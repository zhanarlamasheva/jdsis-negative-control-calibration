"""OULAD Behavioral Confidence Model (MARS v2, piece B).

Trains a LightGBM classifier mapping a student's VLE engagement trajectory
(snapshot at first_deadline) to one of 4 engagement classes. Labels are
synthesized by deterministic rules (no engagement ground truth exists),
mirroring the XES3G5M rule->XGBoost design.
"""
from __future__ import annotations

import argparse
import json
import logging
import pickle
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, f1_score

logger = logging.getLogger(__name__)

# order matters: maps to LightGBM class indices 0-3
LABEL_CLASSES = ["ENGAGED", "AT_RISK", "DISENGAGED", "RECOVERED"]

FEATURE_COLS = [
    "daily_click_rate",
    "resource_diversity",
    "forum_ratio",
    "pre_deadline_intensity",
    "session_gap_days",
    "absence_streak",
    "content_depth",
    "recent_vs_overall_rate",
    "prior_attempt_count",
    "module_week_phase",
    "active_days",
    "score_trend",
]

# Rule thresholds (single place to tune).
ABSENCE_DISENGAGE = 14
PREV_GAP_RECOVER = 10
RECOVER_RATE = 1.2
GAP_RISK = 7
PREDL_RISK = 0.1

# learning_rate: 0.1 (LightGBM default)
LGB_PARAMS = {
    "objective": "multiclass",
    "num_class": len(LABEL_CLASSES),
    "num_leaves": 63,
    "n_estimators": 200,
    "random_state": 42,
    "verbose": -1,
}

ENROLL_KEYS = ["id_student", "code_module", "code_presentation"]


class OULADBcm:
    def __init__(self, processed_dir: str | Path = "data/oulad",
                 data_dir: str | Path = "../../датасет/oulad",
                 model_dir: str | Path = "models",
                 results_dir: str | Path = "results/oulad") -> None:
        self.processed_dir = Path(processed_dir)
        self.data_dir = Path(data_dir)
        self.model_dir = Path(model_dir)
        self.results_dir = Path(results_dir)
        self.model = None
        self._classes: list[str] = list(LABEL_CLASSES)

    def build_features(self, daily_vle: pd.DataFrame, target: pd.DataFrame,
                       score_trend: pd.Series | None = None) -> pd.DataFrame:
        """One feature row per enrollment, snapshot at first_deadline.

        Returns a frame indexed by ENROLL_KEYS with FEATURE_COLS plus the
        rule-only aux columns recent7_clicks and prev_gap_days.
        """
        tgt = target.set_index(ENROLL_KEYS)
        assert not tgt.index.duplicated().any(), "duplicate enrollment keys in target"
        merged = daily_vle.merge(
            target[ENROLL_KEYS + ["first_deadline"]], on=ENROLL_KEYS, how="left")
        merged = merged[merged["date"] < merged["first_deadline"]]

        records = {}
        for key, g in merged.groupby(ENROLL_KEYS, observed=True):
            fd = tgt.loc[key, "first_deadline"]
            records[key] = self._features_for_group(g, fd)

        feats = pd.DataFrame.from_dict(records, orient="index")
        feats.index = pd.MultiIndex.from_tuples(feats.index, names=ENROLL_KEYS)

        # enrollment-level features from target
        feats["prior_attempt_count"] = tgt["num_of_prev_attempts"].reindex(feats.index)
        feats["module_week_phase"] = (
            tgt["first_deadline"] / tgt["module_presentation_length"]).reindex(feats.index)

        # score_trend (default 0)
        if score_trend is None:
            feats["score_trend"] = 0.0
        else:
            feats["score_trend"] = score_trend.reindex(feats.index).fillna(0.0)

        feats[FEATURE_COLS] = feats[FEATURE_COLS].fillna(0.0).astype(float)
        return feats

    @staticmethod
    def _features_for_group(g: pd.DataFrame, first_deadline: float) -> dict:
        daily = g.groupby("date")["sum_click"].sum().sort_index()
        days = daily.index.to_numpy(dtype=float)
        clicks = daily.to_numpy(dtype=float)
        n_active = len(days)
        total = float(clicks.sum())

        # daily_click_rate: mean clicks/day over last 7 *active* days (spec)
        last7 = clicks[-7:]
        daily_click_rate = float(last7.mean()) if last7.size else 0.0
        # recent7_clicks: total clicks in the calendar window [first_deadline-7, first_deadline)
        recent_mask = days >= (first_deadline - 7)
        recent7_clicks = float(clicks[recent_mask].sum())
        overall_rate = float(clicks.mean()) if n_active else 0.0
        recent_vs_overall = daily_click_rate / overall_rate if overall_rate > 0 else 0.0

        resource_diversity = int(g["activity_type"].nunique())
        forum_clicks = float(g.loc[g["activity_type"] == "forumng", "sum_click"].sum())
        forum_ratio = forum_clicks / total if total > 0 else 0.0

        pre_mask = days >= (first_deadline - 3)
        pre_deadline_intensity = float(clicks[pre_mask].sum()) / total if total > 0 else 0.0

        if n_active >= 2:
            internal = np.diff(days) - 1            # inactive days between active days
            absence_streak = float(internal.max())
            session_gap_days = float(days[-1] - days[-2])
            # prev_gap: gaps whose following active day is within last 14 days before cutoff
            recent_following = days[1:] >= (first_deadline - 14)
            prev_gaps = internal[recent_following]
            prev_gap_days = float(prev_gaps.max()) if prev_gaps.size else 0.0
        else:
            absence_streak = 0.0
            session_gap_days = 0.0
            prev_gap_days = 0.0
        # tail inactivity right before deadline counts toward absence too
        tail_gap = float(first_deadline - days[-1] - 1) if n_active else 0.0
        absence_streak = max(absence_streak, tail_gap, 0.0)

        content_depth = total / n_active if n_active else 0.0

        return {
            "daily_click_rate": daily_click_rate,
            "resource_diversity": resource_diversity,
            "forum_ratio": forum_ratio,
            "pre_deadline_intensity": pre_deadline_intensity,
            "session_gap_days": session_gap_days,
            "absence_streak": absence_streak,
            "content_depth": content_depth,
            "recent_vs_overall_rate": recent_vs_overall,
            "active_days": float(n_active),
            "recent7_clicks": recent7_clicks,
            "prev_gap_days": prev_gap_days,
        }

    def load_assessment_features(self, target: pd.DataFrame) -> pd.Series:
        """score_trend per enrollment: slope of the last <=3 TMA scores
        submitted strictly before the enrollment's first_deadline.

        Returns Series indexed by ENROLL_KEYS (0.0 where <2 scores available).
        Note: at a first_deadline cutoff most enrollments have <2 submitted
        TMAs, so score_trend is 0 for the majority by construction.
        """
        assess = pd.read_csv(self.data_dir / "assessments.csv")
        sa = pd.read_csv(self.data_dir / "studentAssessment.csv")
        tma = assess[assess["assessment_type"] == "TMA"][
            ["id_assessment", "code_module", "code_presentation", "date"]]
        df = sa.merge(tma, on="id_assessment", how="inner")
        df = df.merge(target[ENROLL_KEYS + ["first_deadline"]], on=ENROLL_KEYS, how="inner")
        df = df[df["date_submitted"] < df["first_deadline"]]
        df = df.sort_values(["id_student", "code_module", "code_presentation", "date"])

        out = {}
        for key, g in df.groupby(ENROLL_KEYS):
            scores = g["score"].dropna().to_numpy(dtype=float)[-3:]
            if scores.size >= 2:
                x = np.arange(scores.size, dtype=float)
                out[key] = float(np.polyfit(x, scores, 1)[0])
            else:
                out[key] = 0.0
        s = pd.Series(out)
        if not s.empty:
            s.index = pd.MultiIndex.from_tuples(s.index, names=ENROLL_KEYS)
        return s

    @staticmethod
    def rule_labels(feats: pd.DataFrame) -> pd.Series:
        """Ordered rules; later assignments override earlier (priority:
        DISENGAGED > RECOVERED > AT_RISK > ENGAGED)."""
        labels = pd.Series("ENGAGED", index=feats.index)
        at_risk = (
            (feats["score_trend"] < 0)
            | (feats["session_gap_days"] > GAP_RISK)
            | (feats["pre_deadline_intensity"] < PREDL_RISK)
        )
        labels[at_risk] = "AT_RISK"
        recovered = (
            (feats["prev_gap_days"] >= PREV_GAP_RECOVER)
            & (feats["recent_vs_overall_rate"] > RECOVER_RATE)
        )
        labels[recovered] = "RECOVERED"
        disengaged = (
            (feats["recent7_clicks"] == 0)
            & (feats["absence_streak"] >= ABSENCE_DISENGAGE)
        )
        labels[disengaged] = "DISENGAGED"
        return labels

    def train(self, X_train: pd.DataFrame, y_train: pd.Series,
              X_val: pd.DataFrame, y_val: pd.Series) -> "lgb.LGBMClassifier":
        self._classes = list(LABEL_CLASSES)
        y_tr = y_train.map(self._classes.index)
        y_va = y_val.map(self._classes.index)
        model = lgb.LGBMClassifier(**LGB_PARAMS)
        model.fit(
            X_train[FEATURE_COLS], y_tr,
            eval_set=[(X_val[FEATURE_COLS], y_va)],
            callbacks=[lgb.early_stopping(20, verbose=False)],
        )
        self.model = model
        return model

    def predict(self, X: pd.DataFrame) -> list[str]:
        idx = self.model.predict(X[FEATURE_COLS])
        return [self._classes[int(i)] for i in idx]

    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
        y_true = y_test.map(self._classes.index)
        y_pred = self.model.predict(X_test[FEATURE_COLS])
        report = classification_report(
            y_true, y_pred, labels=list(range(len(self._classes))),
            target_names=self._classes, output_dict=True, zero_division=0)
        macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
        per_class = {c: report[c] for c in self._classes}
        return {
            "macro_f1": macro_f1,
            "per_class": per_class,
            "class_balance": y_test.value_counts().to_dict(),
        }

    def build(self) -> dict:
        proc = self.processed_dir / "processed"
        splt = self.processed_dir / "splits"
        daily = pd.read_parquet(proc / "daily_vle.parquet")
        target = pd.read_parquet(proc / "student_target.parquet")

        score_trend = self.load_assessment_features(target)
        feats = self.build_features(daily, target, score_trend=score_trend)
        labels = self.rule_labels(feats)
        logger.info("label balance: %s", labels.value_counts().to_dict())

        def _keys(split_name):
            df = pd.read_parquet(splt / f"{split_name}.parquet")
            return pd.MultiIndex.from_frame(df[ENROLL_KEYS])

        tr_idx = feats.index.intersection(_keys("train"))
        va_idx = feats.index.intersection(_keys("val"))
        te_idx = feats.index.intersection(_keys("test"))

        X = feats
        y = labels
        self.train(X.loc[tr_idx], y.loc[tr_idx], X.loc[va_idx], y.loc[va_idx])
        metrics = self.evaluate(X.loc[te_idx], y.loc[te_idx])

        model_path = self.model_dir / "bcm_oulad.pkl"
        metrics_path = self.results_dir / "bcm_metrics.json"
        model_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        with open(model_path, "wb") as fh:
            pickle.dump({"model": self.model, "classes": self._classes,
                         "features": FEATURE_COLS}, fh)
        metrics["train_balance"] = y.loc[tr_idx].value_counts().to_dict()
        metrics_path.write_text(json.dumps(metrics, indent=2, default=str))
        logger.info("BCM test macro-F1=%.4f", metrics["macro_f1"])
        return {"model_path": str(model_path), "metrics_path": str(metrics_path)}


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description="Train OULAD BCM")
    ap.add_argument("--processed-dir", default="data/oulad")
    ap.add_argument("--data-dir", default="../../датасет/oulad")
    ap.add_argument("--build", action="store_true")
    args = ap.parse_args()
    if args.build:
        OULADBcm(processed_dir=args.processed_dir, data_dir=args.data_dir).build()
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
