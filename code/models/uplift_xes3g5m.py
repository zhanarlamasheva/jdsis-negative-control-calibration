"""XES3G5M causal uplift: effect of weak-concept practice on mastery gain.

Domain adapter over CausalUplift. Three-window temporal split of each student's
interaction sequence: X-window (0..x_frac) -> covariates; T-window (x_frac..t_frac)
-> treatment (heavy practice of already-weak concepts); GT half (t_frac..1.0) -> outcome
Y = 1[difficulty-adjusted learning gain > 0]. Covariates are strictly from the X-window
and exclude any T-window practice signal (collider / leaks T).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from models.pred_oulad import SEED
from models.uplift_core import (
    CausalUplift, auuc, evalue_from_rr, policy_value_at_k, qini_coefficient,
)

logger = logging.getLogger("mars.model.uplift_xes3g5m")


class XES3G5MUplift(CausalUplift):
    def __init__(self, data_dir: str = "data", results_dir: str = "results/xes3g5m",
                 x_frac: float = 0.15, t_frac: float = 0.30, k_folds: int = 5,
                 seed: int = SEED):
        super().__init__(k_folds=k_folds, seed=seed)
        self.data_dir = Path(data_dir)
        self.results_dir = Path(results_dir)
        self.x_frac = x_frac
        self.t_frac = t_frac

    def _windows(self, n: int):
        """Index partition (x_idx, t_idx, gt_idx) for a length-n sequence."""
        xe = max(1, int(n * self.x_frac))
        te = max(xe + 1, int(n * self.t_frac))
        te = min(te, n - 1)               # leave >=1 for GT
        idx = np.arange(n)
        return idx[:xe], idx[xe:te], idx[te:]

    @staticmethod
    def _first_tag(tags_val):
        """First concept of an interaction. Real data: ';'-joined str ("30;34;42");
        synthetic/tests: list[int]. Take the first concept either way."""
        if isinstance(tags_val, str):
            return int(tags_val.split(";")[0])
        if isinstance(tags_val, (list, tuple, np.ndarray)):
            return int(tags_val[0]) if len(tags_val) else -1
        return int(tags_val)

    def _weak_practice_count(self, g: pd.DataFrame) -> int:
        """Count of T-window interactions on concepts the student got wrong in X-window."""
        g = g.sort_values("timestamp")
        n = len(g)
        x_idx, t_idx, _ = self._windows(n)
        tags = g["tags"].map(self._first_tag).to_numpy()
        correct = g["correct"].to_numpy()
        weak = set(tags[x_idx][correct[x_idx] == 0].tolist())
        if not weak:
            return 0
        return int(sum(1 for c in tags[t_idx] if c in weak))

    def _treatment(self, df: pd.DataFrame):
        """Return (T[N] binary, uids[N]): T=1 if weak-concept practice > cohort median."""
        counts = (df.groupby("user_id", sort=True)
                  .apply(self._weak_practice_count))
        uids = counts.index.to_numpy()
        cnt = counts.to_numpy().astype(float)
        med = float(np.median(cnt))
        T = (cnt > med).astype(int)
        return T, uids

    def _q_acc(self, df: pd.DataFrame) -> dict:
        """Global question accuracy map q_acc[str(question_id)] = mean(correct).
        Keys are str so int (real) and str (synthetic) question_ids both resolve."""
        acc = df.groupby("question_id")["correct"].mean()
        return {str(k): float(v) for k, v in acc.items()}

    def _student_features(self, g: pd.DataFrame, q_acc: dict):
        """(x_feats dict, y) for one student. X from X-window only; Y=1[lg_adj>0]."""
        g = g.sort_values("timestamp")
        n = len(g)
        x_idx, t_idx, gt_idx = self._windows(n)
        tags = g["tags"].map(self._first_tag).to_numpy()
        correct = g["correct"].to_numpy().astype(float)
        qid = g["question_id"].to_numpy()
        qa = np.array([q_acc.get(str(q), 0.5) for q in qid], dtype=float)

        xc = correct[x_idx]
        xt = tags[x_idx]
        weak = set(xt[xc == 0].tolist())
        x_feats = {
            "early_theta": float((xc - qa[x_idx]).mean()),
            "n_early": float(len(x_idx)),
            "early_accuracy": float(xc.mean()) if len(x_idx) else 0.0,
            "n_distinct_concepts_early": float(len(set(xt.tolist()))),
            "early_weak_count": float(len(weak)),
        }
        # outcome: difficulty-adjusted learning gain, pre = 0..t_frac, post = GT half
        pre = np.concatenate([x_idx, t_idx])
        pre_acc = float(correct[pre].mean())
        post_acc = float(correct[gt_idx].mean())
        pre_exp = float(qa[pre].mean())
        post_exp = float(qa[gt_idx].mean())
        lg_adj = (post_acc - post_exp) - (pre_acc - pre_exp)
        y = int(lg_adj > 0)
        return x_feats, y

    def _build_TXY(self, df: pd.DataFrame):
        """Return (T[N], X[N,d], Y[N], uids, feat_names). X is X-window only."""
        q_acc = self._q_acc(df)
        names = ["early_theta", "n_early", "early_accuracy",
                 "n_distinct_concepts_early", "early_weak_count"]
        rows, ys, uids = [], [], []
        for uid, g in df.groupby("user_id", sort=True):
            xf, y = self._student_features(g, q_acc)
            rows.append([xf[k] for k in names])
            ys.append(y)
            uids.append(uid)
        X = np.asarray(rows, dtype=np.float32)
        Y = np.asarray(ys, dtype=int)
        T, t_uids = self._treatment(df)
        assert list(t_uids) == list(uids), "treatment/feature uid order mismatch"
        return T, X, Y, np.asarray(uids), names

    _COLS = ["user_id", "timestamp", "question_id", "tags", "correct"]

    def _load_full(self) -> pd.DataFrame:
        """Concat the temporal splits into each student's FULL interaction sequence.
        XES3G5M splits are interaction-level over the SAME users (train=early ..
        test=late); concatenating reconstructs the whole per-user timeline."""
        parts = []
        for s in ("train", "val", "test"):
            p = self.data_dir / "splits" / f"{s}.parquet"
            if p.exists():
                parts.append(pd.read_parquet(p, columns=self._COLS))
        return pd.concat(parts, ignore_index=True)

    def _user_split(self, df: pd.DataFrame, test_frac: float = 0.2):
        """Clean USER-level holdout: disjoint student cohorts (no shared users)."""
        users = np.sort(df["user_id"].unique())
        perm = np.random.default_rng(self.seed).permutation(len(users))
        n_test = max(1, int(test_frac * len(users)))
        test_u = set(users[perm[:n_test]].tolist())
        is_test = df["user_id"].isin(test_u)
        return df[~is_test], df[is_test]

    def build(self) -> dict:
        """ATE cross-fit on train cohort; tau ranking + Qini on a disjoint user holdout."""
        full = self._load_full()
        tr, te = self._user_split(full)

        Ttr, Xtr, Ytr, _, _ = self._build_TXY(tr)
        e_tr, mu0_tr, mu1_tr = self._fit_nuisances(Xtr, Ttr, Ytr)
        keep = (e_tr >= 0.02) & (e_tr <= 0.98)
        n_trim = int((~keep).sum())
        ate, se, ci = self._aipw_ate(Ttr[keep], Ytr[keep], e_tr[keep],
                                     mu0_tr[keep], mu1_tr[keep])
        base = float(np.clip(mu0_tr[keep].mean(), 1e-6, 1.0))
        rr = (base + ate) / base
        ev = evalue_from_rr(rr)
        ev_lo = evalue_from_rr((base + ci[0]) / base) if ci[0] > 0 else 1.0

        e_full = self._xgb().fit(Xtr, Ttr)
        Tte, Xte, Yte, uids_te, _ = self._build_TXY(te)
        e_te = np.clip(e_full.predict_proba(Xte)[:, 1], 0.02, 0.98)
        mu0_te = self._fit_proba1(Xtr[Ttr == 0], Ytr[Ttr == 0], Xte)
        mu1_te = self._fit_proba1(Xtr[Ttr == 1], Ytr[Ttr == 1], Xte)
        tau_te = self._xlearner_tau(Xte, Tte, Yte, e_te, mu0_te, mu1_te)

        res = {
            "domain": "xes3g5m", "treatment": "weak_concept_practice_gt_median",
            "outcome": "lg_adj_positive",
            "ate": ate, "se": se, "ci95": ci,
            "rr_mastery": rr, "evalue": float(ev), "evalue_ci_lower": float(ev_lo),
            "qini_coef": qini_coefficient(tau_te, Tte, Yte),
            "auuc": auuc(tau_te, Tte, Yte),
            "policy_at_k": {f"{k}": policy_value_at_k(tau_te, k) for k in (0.1, 0.2, 0.3)},
            "n_train": int(len(Ttr)), "n_test": int(len(Tte)),
            "n_trim": n_trim, "treat_rate_train": float(Ttr.mean()),
        }
        self.results_dir.mkdir(parents=True, exist_ok=True)
        (self.results_dir / "uplift_metrics.json").write_text(json.dumps(res, indent=2))
        pd.DataFrame({"user_id": uids_te, "tau": tau_te}).to_parquet(
            self.results_dir / "uplift_tau.parquet", index=False)
        logger.info("xes3g5m uplift ATE=%.4f CI=%s qini=%.4f", ate, ci, res["qini_coef"])
        return res


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    up = XES3G5MUplift(k_folds=5)
    print(json.dumps(up.build(), indent=2))
