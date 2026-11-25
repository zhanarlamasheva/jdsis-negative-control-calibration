"""OULAD causal uplift: effect of on-time first-assessment submission on retention.

Domain adapter over CausalUplift: builds treatment T (on-time first submission),
pre-treatment covariates X (demographics + pre-cutoff click trend), outcome Y (retained).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from models.pred_oulad import (
    CAT_COLS, ENROLL_KEYS, NUM_COLS, SEED, OULADPred, StaticEncoder,
    activity_vocab, click_trend_features,
)
from models.uplift_core import (
    CausalUplift, auuc, evalue_from_rr, policy_value_at_k, qini_coefficient,
)

logger = logging.getLogger("mars.model.uplift_oulad")


class OULADUplift(CausalUplift):
    def __init__(self, processed_dir: str = "data/oulad",
                 data_dir: str = "../../датасет/oulad",
                 results_dir: str = "results/oulad",
                 window_frac: float | None = None, k_folds: int = 5,
                 seed: int = SEED):
        super().__init__(k_folds=k_folds, seed=seed)
        self.window_frac = window_frac
        self.results_dir = Path(results_dir)
        self.pred = OULADPred(processed_dir=processed_dir, data_dir=data_dir,
                              window_frac=window_frac)
        self._enc: StaticEncoder | None = None

    def _treatment(self, target: pd.DataFrame, assess: pd.DataFrame,
                   keys: pd.DataFrame) -> pd.Series:
        """T=1 if the first assessment (min a_deadline) was submitted on-time."""
        keys = keys[ENROLL_KEYS].drop_duplicates().reset_index(drop=True)
        first = (assess.sort_values("a_deadline")
                 .drop_duplicates(["code_module", "code_presentation"])
                 [["code_module", "code_presentation", "id_assessment"]]
                 .rename(columns={"id_assessment": "_first_id"}))
        a = assess.merge(first, on=["code_module", "code_presentation"], how="inner")
        a = a[a["id_assessment"] == a["_first_id"]].copy()
        a["_ontime"] = (a["date_submitted"] <= a["a_deadline"]).astype(int)
        ot = a.groupby(ENROLL_KEYS)["_ontime"].max()
        s = keys.merge(ot.rename("T").reset_index(), on=ENROLL_KEYS, how="left")
        return s["T"].fillna(0).astype(int)

    def _build_TXY(self, daily, target, assess, keys, fit_enc: bool = False):
        """Return (T[N], X[N,d], Y[N], keys_df, feat_names). X is pre-treatment only."""
        keys = keys[ENROLL_KEYS].drop_duplicates().reset_index(drop=True)
        acts = activity_vocab(daily)
        X_seq, lengths, y_drop, _, _ = self.pred.build_sequences(daily, target, keys, acts)
        trend = click_trend_features(X_seq.sum(axis=2), lengths)
        trend_names = [f"trend_{i}" for i in range(trend.shape[1])]

        demo = keys.merge(target[ENROLL_KEYS + CAT_COLS + NUM_COLS],
                          on=ENROLL_KEYS, how="left")
        if fit_enc:
            self._enc = StaticEncoder(cat_cols=CAT_COLS, num_cols=NUM_COLS)
            self._enc.fit(demo)
        if self._enc is None:
            raise RuntimeError("call _build_TXY with fit_enc=True on train data first")
        demo_enc = self._enc.transform(demo)
        demo_names = [f"demo_{i}" for i in range(demo_enc.shape[1])]

        X = np.concatenate([demo_enc, trend], axis=1).astype(np.float32)
        names = demo_names + trend_names
        T = self._treatment(target, assess, keys).to_numpy().astype(int)
        Y = (1 - y_drop_withdrawn(target, keys)).astype(int)
        return T, X, Y, keys, names

    def _load(self):
        proc = self.pred.processed_dir
        daily = pd.read_parquet(proc / "processed" / "daily_vle.parquet")
        target = pd.read_parquet(proc / "processed" / "student_target.parquet")
        assess = self.pred._load_assessments()
        return daily, target, assess

    def build(self) -> dict:
        """ATE cross-fit on the 2013 train cohort; tau ranking + Qini on 2014 test."""
        daily, target, assess = self._load()
        tr_keys = self.pred._load_keys("train")
        te_keys = self.pred._load_keys("test")

        # --- train: cross-fit nuisances -> AIPW ATE + E-value ---
        Ttr, Xtr, Ytr, _, _ = self._build_TXY(daily, target, assess, tr_keys, fit_enc=True)
        e_tr, mu0_tr, mu1_tr = self._fit_nuisances(Xtr, Ttr, Ytr)
        keep = (e_tr >= 0.02) & (e_tr <= 0.98)
        n_trim = int((~keep).sum())
        ate, se, ci = self._aipw_ate(Ttr[keep], Ytr[keep], e_tr[keep],
                                     mu0_tr[keep], mu1_tr[keep])
        # E-value on the SAME trimmed cohort: convert the risk-difference ATE (and its
        # CI limit nearest the null) to a retention risk ratio via the baseline mu0 mean.
        base = float(np.clip(mu0_tr[keep].mean(), 1e-6, 1.0))
        rr = (base + ate) / base
        ev = evalue_from_rr(rr)
        rr_lo = (base + ci[0]) / base          # RR at the CI bound closest to null
        ev_lo = evalue_from_rr(rr_lo) if ci[0] > 0 else 1.0

        # --- test: refit nuisances on all train, predict tau on 2014, Qini ---
        e_full = self._xgb().fit(Xtr, Ttr)
        Tte, Xte, Yte, keys_te, _ = self._build_TXY(daily, target, assess, te_keys)
        e_te = np.clip(e_full.predict_proba(Xte)[:, 1], 0.02, 0.98)
        mu0_te = self._fit_proba1(Xtr[Ttr == 0], Ytr[Ttr == 0], Xte)
        mu1_te = self._fit_proba1(Xtr[Ttr == 1], Ytr[Ttr == 1], Xte)
        tau_te = self._xlearner_tau(Xte, Tte, Yte, e_te, mu0_te, mu1_te)

        res = {
            "ate": ate, "se": se, "ci95": ci,
            "rr_retention": rr, "evalue": float(ev), "evalue_ci_lower": float(ev_lo),
            "qini_coef": qini_coefficient(tau_te, Tte, Yte),
            "auuc": auuc(tau_te, Tte, Yte),
            "policy_at_k": {f"{k}": policy_value_at_k(tau_te, k) for k in (0.1, 0.2, 0.3)},
            "n_train": int(len(Ttr)), "n_test": int(len(Tte)),
            "n_trim": n_trim, "treat_rate_train": float(Ttr.mean()),
        }
        self.results_dir.mkdir(parents=True, exist_ok=True)
        (self.results_dir / "uplift_metrics.json").write_text(json.dumps(res, indent=2))
        keys_out = keys_te.copy()
        keys_out["tau"] = tau_te
        keys_out.to_parquet(self.results_dir / "uplift_tau.parquet", index=False)
        logger.info("uplift ATE=%.4f CI=%s qini=%.4f", ate, ci, res["qini_coef"])
        return res


def y_drop_withdrawn(target: pd.DataFrame, keys: pd.DataFrame) -> np.ndarray:
    """1 if final_result == Withdrawn, aligned to keys order."""
    keys = keys[ENROLL_KEYS].drop_duplicates().reset_index(drop=True)
    m = keys.merge(target[ENROLL_KEYS + ["final_result"]], on=ENROLL_KEYS, how="left")
    return (m["final_result"].astype(str) == "Withdrawn").to_numpy().astype(int)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    up = OULADUplift(window_frac=0.5, k_folds=5)
    print(json.dumps(up.build(), indent=2))
