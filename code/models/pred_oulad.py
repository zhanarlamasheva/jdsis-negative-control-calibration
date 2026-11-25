"""OULAD dropout prediction: multi-task LSTM (BCE classifier + Cox survival head).

Piece D of MARS v2 (contribution 3.3). Reads piece-A parquet artifacts, builds a
per-student daily VLE click sequence (date < first_deadline), and trains a shared
LSTM encoder with two heads: binary dropout (ROC-AUC) and Cox proportional-hazards
risk (C-index). Standalone trainable model; outcome-weighted ranking is out of scope.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, brier_score_loss

logger = logging.getLogger("mars.model.pred_oulad")

ENROLL_KEYS = ["id_student", "code_module", "code_presentation"]
DROPOUT_RESULTS = {"Fail", "Withdrawn"}
MAX_DAYS = 61
HIDDEN, LAYERS, MLP_OUT = 128, 2, 64
DROPOUT_P = 0.3
LR, EPOCHS, PATIENCE = 1e-3, 100, 20
LAMBDA_COX = 1.0
BATCH = 256
SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CAT_COLS = ["gender", "region", "highest_education", "imd_band", "age_band", "disability"]
NUM_COLS = ["num_of_prev_attempts", "studied_credits", "date_registration"]


def activity_vocab(daily: pd.DataFrame) -> list[str]:
    """Stable sorted list of VLE activity_type categories (pinned at build time)."""
    return sorted(daily["activity_type"].astype(str).unique().tolist())


def click_trend_features(daily_total: np.ndarray, lengths: np.ndarray) -> np.ndarray:
    """Engineered disengagement-trajectory features from per-day click totals.

    daily_total: [N, MAX_DAYS] summed clicks per day. lengths: [N] window length.
    Returns [N, 7]: total, active_days, mean_per_active_day, clicks_last_14d,
    weekly slope, longest zero-run, days_since_last_click. All within [0, length).
    """
    n = daily_total.shape[0]
    feats = np.zeros((n, 7), dtype=np.float32)
    for i in range(n):
        L = int(lengths[i])
        seq = daily_total[i, :L]
        active = seq > 0
        n_active = int(active.sum())
        total = float(seq.sum())
        feats[i, 0] = total
        feats[i, 1] = float(n_active)
        feats[i, 2] = total / n_active if n_active else 0.0
        feats[i, 3] = float(seq[max(0, L - 14):L].sum())
        n_weeks = max(1, int(np.ceil(L / 7)))
        weekly = np.array([seq[w * 7:(w + 1) * 7].sum() for w in range(n_weeks)],
                          dtype=float)
        if len(weekly) >= 2:
            feats[i, 4] = float(np.polyfit(np.arange(len(weekly)), weekly, 1)[0])
        gap = best = 0
        for v in active:
            if v:
                gap = 0
            else:
                gap += 1
                best = max(best, gap)
        feats[i, 5] = float(best)
        idx = np.flatnonzero(active)
        feats[i, 6] = float(L - 1 - idx[-1]) if len(idx) else float(L)
    return feats


class PredNet(nn.Module):
    """Shared LSTM encoder + masked temporal attention + static fusion + two heads."""

    def __init__(self, n_feat: int, n_static: int = 0):
        super().__init__()
        self.n_static = n_static
        self.lstm = nn.LSTM(n_feat, HIDDEN, LAYERS, dropout=DROPOUT_P, batch_first=True)
        self.attn = nn.Linear(HIDDEN, 1)
        self.mlp = nn.Sequential(nn.Linear(HIDDEN + n_static, MLP_OUT),
                                 nn.ReLU(), nn.Dropout(DROPOUT_P))
        self.cls = nn.Linear(MLP_OUT, 1)
        self.risk = nn.Linear(MLP_OUT, 1)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor,
                static: torch.Tensor | None = None):
        out, _ = self.lstm(x)                                   # [B, T, H]
        T = x.size(1)
        mask = torch.arange(T, device=x.device)[None, :] < lengths[:, None]  # [B, T]
        score = self.attn(out).squeeze(-1)                      # [B, T]
        score = score.masked_fill(~mask, float("-inf"))
        w = torch.softmax(score, dim=1)                         # [B, T]
        ctx = (w.unsqueeze(-1) * out).sum(1)                    # [B, H]
        if self.n_static:
            assert static is not None, "n_static>0 requires static features"
            ctx = torch.cat([ctx, static], dim=1)               # [B, H + S]
        h = self.mlp(ctx)
        return self.cls(h).squeeze(-1), self.risk(h).squeeze(-1)  # [B], [B]


def cox_loss(risk: torch.Tensor, time: torch.Tensor, event: torch.Tensor) -> torch.Tensor:
    """Negative log Cox partial likelihood (Breslow) over the in-batch risk set.

    For each event, the risk set is all samples with time >= the event time.
    # ponytail: in-batch risk set; switch to a global risk set only if C-index is poor.
    """
    order = torch.argsort(time, descending=True)
    risk, event = risk[order], event[order]
    log_risk_set = torch.logcumsumexp(risk, dim=0)             # sum over time >= current
    n_events = event.sum().clamp(min=1.0)
    return -((risk - log_risk_set) * event).sum() / n_events


def c_index(risk: np.ndarray, time: np.ndarray, event: np.ndarray) -> float:
    """Harrell concordance: over comparable pairs (event_i=1, time_i<time_j),
    fraction with risk_i > risk_j (ties counted 0.5). Higher risk = earlier event."""
    risk = np.asarray(risk, dtype=float)
    time = np.asarray(time, dtype=float)
    event = np.asarray(event)
    ev = np.where(event == 1)[0]
    if len(ev) == 0:
        return 0.5
    tr = time[ev][:, None]
    rr = risk[ev][:, None]
    comparable = tr < time[None, :]                           # [n_event, n]
    if not comparable.any():
        return 0.5
    concordant = comparable & (rr > risk[None, :])
    ties = comparable & (rr == risk[None, :])
    return float((concordant.sum() + 0.5 * ties.sum()) / comparable.sum())


class StaticEncoder:
    """Fit-on-train one-hot (categorical) + standardize (numeric) -> dense [N, S].

    Categorical: vocab pinned at fit; unseen/NaN -> all-zero one-hot block plus a
    per-field *_missing flag. Numeric: train mean/std; NaN imputed to the train mean
    (0 after standardization), no separate missing flag. Leak-free: stats come only
    from the frame passed to fit().
    """

    def __init__(self, cat_cols=CAT_COLS, num_cols=NUM_COLS):
        self.cat_cols = list(cat_cols)
        self.num_cols = list(num_cols)
        self.vocab: dict[str, list[str]] = {}
        self.mean: dict[str, float] = {}
        self.std: dict[str, float] = {}
        self.n_static = 0

    def fit(self, df: pd.DataFrame) -> "StaticEncoder":
        for c in self.cat_cols:
            vals = df[c].dropna().astype(str).unique().tolist()
            self.vocab[c] = sorted(vals)
        for c in self.num_cols:
            col = pd.to_numeric(df[c], errors="coerce")
            self.mean[c] = float(col.mean())
            s = float(col.std())
            self.std[c] = s if s > 1e-8 else 1.0
        self.n_static = (sum(len(self.vocab[c]) + 1 for c in self.cat_cols)
                         + len(self.num_cols))
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        n = len(df)
        cols: list[np.ndarray] = []
        for c in self.cat_cols:
            vocab = self.vocab[c]
            idx = {v: i for i, v in enumerate(vocab)}
            block = np.zeros((n, len(vocab)), dtype=np.float32)
            sv = df[c].to_numpy(dtype=object)
            for r in range(n):
                v = sv[r]
                if v is None or (isinstance(v, float) and np.isnan(v)):
                    continue
                j = idx.get(str(v))
                if j is not None:
                    block[r, j] = 1.0
            missing = (block.sum(axis=1) == 0).astype(np.float32)[:, None]
            cols.append(block)
            cols.append(missing)
        for c in self.num_cols:
            col = pd.to_numeric(df[c], errors="coerce").to_numpy(dtype=np.float64)
            z = (np.nan_to_num(col, nan=self.mean[c]) - self.mean[c]) / self.std[c]
            cols.append(z.astype(np.float32)[:, None])
        return np.concatenate(cols, axis=1).astype(np.float32)

    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        return self.fit(df).transform(df)


def _stratified_split(arrays, frac: float = 0.85, seed: int = SEED):
    """Split (X, lengths, static, y, t, e) into two tuples, stratified on y (binary)."""
    X, lengths, static, y, t, e = arrays
    rng = np.random.default_rng(seed)
    fit_mask = np.zeros(len(y), dtype=bool)
    for cls in (0, 1):
        idx = np.where(y == cls)[0]
        rng.shuffle(idx)
        fit_mask[idx[:int(len(idx) * frac)]] = True
    pick = lambda m: (X[m], lengths[m], static[m], y[m], t[m], e[m])
    return pick(fit_mask), pick(~fit_mask)


def _strat_mask(y: np.ndarray, frac: float = 0.85, seed: int = SEED) -> np.ndarray:
    """Boolean fit-mask stratified on binary y (True = fit, False = monitor)."""
    rng = np.random.default_rng(seed)
    mask = np.zeros(len(y), dtype=bool)
    for cls in (0, 1):
        idx = np.where(y == cls)[0]
        rng.shuffle(idx)
        mask[idx[:int(len(idx) * frac)]] = True
    return mask


class OULADPred:
    def __init__(self, processed_dir: str = "data/oulad",
                 data_dir: str = "../../датасет/oulad",
                 model_dir: str = "models", results_dir: str = "results/oulad",
                 window_frac: float | None = None):
        self.processed_dir = Path(processed_dir)
        self.data_dir = Path(data_dir)
        self.model_dir = Path(model_dir)
        self.results_dir = Path(results_dir)
        self.window_frac = window_frac
        self.model: nn.Module | None = None
        self._acts: list[str] = []
        self.enc: StaticEncoder | None = None
        self.n_static = 0
        self._last_submitted_any = None

    def _cut_map(self, target: pd.DataFrame) -> pd.Series:
        """Per-enrollment observation cutoff, indexed by ENROLL_KEYS.

        first_deadline (leak-free default) or window_frac * module length (fallback B).
        Shared by the click window AND the assessment window so both see the same horizon.
        """
        t = target.set_index(ENROLL_KEYS)
        if self.window_frac is None:
            return t["first_deadline"].astype(float)
        return self.window_frac * t["module_presentation_length"].astype(float)

    def build_sequences(self, daily: pd.DataFrame, target: pd.DataFrame,
                        keys: pd.DataFrame, acts: list[str]):
        """Return (X, lengths, y_drop, t_event, e_event) for the given enrollments.

        X: float32 [N, MAX_DAYS, F] daily clicks per activity_type, padded with zeros.
        lengths: int [N] real window length = min(ceil(first_deadline), MAX_DAYS), >=1.
        y_drop: int [N] 1 if final_result in {Fail, Withdrawn}.
        t_event/e_event: survival time (unreg day, else module length) / event (Withdrawn).
        """
        act_idx = {a: i for i, a in enumerate(acts)}
        F = len(acts)
        keys = keys[ENROLL_KEYS].drop_duplicates().reset_index(drop=True)
        enr_idx = {tuple(k): i for i, k in enumerate(keys.itertuples(index=False, name=None))}
        N = len(keys)

        tgt = target.set_index(ENROLL_KEYS)
        X = np.zeros((N, MAX_DAYS, F), dtype=np.float32)
        lengths = np.ones(N, dtype=np.int64)
        y = np.zeros(N, dtype=np.int64)
        t = np.zeros(N, dtype=np.float32)
        e = np.zeros(N, dtype=np.int64)

        # observation cutoff per enrollment, shared with the assessment window.
        cut_map = self._cut_map(target)

        # labels + lengths per enrollment
        for k, i in enr_idx.items():
            row = tgt.loc[k]
            cut = float(cut_map.loc[k])
            # ponytail: cap at MAX_DAYS=61; window_frac=0.25*~257 clips, acceptable
            lengths[i] = max(1, min(int(np.ceil(cut)), MAX_DAYS))
            res = str(row["final_result"])
            y[i] = 1 if res in DROPOUT_RESULTS else 0
            unreg = row["date_unregistration"]
            if res == "Withdrawn" and pd.notna(unreg):
                e[i] = 1
                t[i] = float(unreg)
            else:
                # non-withdrawn, or Withdrawn with missing unreg date -> censored at module end
                e[i] = 0
                t[i] = float(row["module_presentation_length"])

        # clicks: filter to keys + date < cutoff, scatter-add into X
        d = daily.merge(keys, on=ENROLL_KEYS, how="inner").copy()
        d["_cut"] = d.set_index(ENROLL_KEYS).index.map(cut_map)
        d = d[d["date"] < d["_cut"]]
        d = d[d["activity_type"].astype(str).isin(act_idx)]
        if len(d):
            ei = np.array([enr_idx[(r.id_student, r.code_module, r.code_presentation)]
                           for r in d.itertuples(index=False)])
            ri = np.clip(d["date"].to_numpy().astype(int), 0, MAX_DAYS - 1)
            ai = d["activity_type"].astype(str).map(act_idx).to_numpy().astype(int)
            np.add.at(X, (ei, ri, ai), d["sum_click"].to_numpy().astype(np.float32))

        return X, lengths, y, t, e

    def train(self, Xtr, len_tr, str_tr, ytr, ttr, etr,
              Xval, len_val, str_val, yval, tval, eval_, epochs: int = EPOCHS):
        torch.manual_seed(SEED)
        F = Xtr.shape[2]
        S = str_tr.shape[1]
        net = PredNet(F, S).to(DEVICE)
        opt = torch.optim.Adam(net.parameters(), lr=LR)
        bce = nn.BCEWithLogitsLoss()   # no pos_weight: better-calibrated probabilities (brier)

        def to_dev(a, dtype):
            return torch.as_tensor(a, dtype=dtype, device=DEVICE)

        Xtr_t, len_tr_t = to_dev(Xtr, torch.float32), to_dev(len_tr, torch.long)
        str_tr_t = to_dev(str_tr, torch.float32)
        ytr_t, ttr_t, etr_t = (to_dev(ytr, torch.float32), to_dev(ttr, torch.float32),
                               to_dev(etr, torch.float32))
        Xv, lv = to_dev(Xval, torch.float32), to_dev(len_val, torch.long)
        str_v = to_dev(str_val, torch.float32)
        yv, tv, ev = (to_dev(yval, torch.float32), to_dev(tval, torch.float32),
                      to_dev(eval_, torch.float32))

        n = len(Xtr)
        best_val, best_state, bad = float("inf"), None, 0
        for epoch in range(epochs):
            net.train()
            perm = torch.randperm(n, device=DEVICE)
            for s in range(0, n, BATCH):
                idx = perm[s:s + BATCH]
                logit, risk = net(Xtr_t[idx], len_tr_t[idx], str_tr_t[idx])
                loss = bce(logit, ytr_t[idx]) + LAMBDA_COX * cox_loss(
                    risk, ttr_t[idx], etr_t[idx])
                opt.zero_grad()
                loss.backward()
                opt.step()
            net.eval()
            with torch.no_grad():
                vl, vr = net(Xv, lv, str_v)
                vloss = (bce(vl, yv) + LAMBDA_COX * cox_loss(vr, tv, ev)).item()
            if vloss < best_val - 1e-4:
                best_val, best_state, bad = vloss, {k: v.detach().cpu().clone()
                                                    for k, v in net.state_dict().items()}, 0
            else:
                bad += 1
                if bad >= PATIENCE:
                    logger.info("pred early stop epoch %d", epoch)
                    break
        if best_state is not None:
            net.load_state_dict(best_state)
        self.model = net
        return net

    def _lstm_prob(self, X, lengths, static) -> np.ndarray:
        net = self.model
        net.eval()
        with torch.no_grad():
            logit, _ = net(torch.as_tensor(X, dtype=torch.float32, device=DEVICE),
                           torch.as_tensor(lengths, dtype=torch.long, device=DEVICE),
                           torch.as_tensor(static, dtype=torch.float32, device=DEVICE))
            return torch.sigmoid(logit).cpu().numpy()

    def _xgb_prob(self, feat_tr, ytr, feat_mon, ymon, feat_te) -> np.ndarray:
        """Train an XGBoost classifier on tabular features, return P(dropout) on test."""
        from xgboost import XGBClassifier
        clf = XGBClassifier(
            n_estimators=600, max_depth=4, learning_rate=0.03,
            subsample=0.8, colsample_bytree=0.8, eval_metric="auc",
            early_stopping_rounds=40, random_state=SEED, n_jobs=4)
        clf.fit(feat_tr, ytr, eval_set=[(feat_mon, ymon)], verbose=False)
        self._xgb = clf
        return clf.predict_proba(feat_te)[:, 1]

    def evaluate(self, Xte, len_te, str_te, yte, tte, ete) -> dict:
        prob = self._lstm_prob(Xte, len_te, str_te)
        net = self.model
        net.eval()
        with torch.no_grad():
            _, risk = net(torch.as_tensor(Xte, dtype=torch.float32, device=DEVICE),
                          torch.as_tensor(len_te, dtype=torch.long, device=DEVICE),
                          torch.as_tensor(str_te, dtype=torch.float32, device=DEVICE))
            risk = risk.cpu().numpy()
        try:
            auc = float(roc_auc_score(yte, prob))
        except ValueError:
            auc = 0.5
        brier = float(brier_score_loss(yte, prob)) if len(set(yte.tolist())) > 1 else float(
            np.mean((prob - yte) ** 2))
        return {"dropout_auc": auc, "c_index": c_index(risk, tte, ete), "brier": brier}

    def _assessment_features(self, target: pd.DataFrame, assess: pd.DataFrame,
                             keys: pd.DataFrame) -> pd.DataFrame:
        """Per-enrollment pre-cutoff assessment aggregates (date_submitted < cut).

        Columns aligned to `keys`: submitted_any, banked_any, n_submitted,
        mean_score, weighted_score, on_time_rate, submission_ratio.
        Leak-free: only submissions before the same `cut` used for clicks count.
        """
        keys = keys[ENROLL_KEYS].drop_duplicates().reset_index(drop=True)
        cm = self._cut_map(target)                      # MultiIndex on ENROLL_KEYS
        cut_df = cm.rename("_cut").reset_index()

        a = assess.merge(keys, on=ENROLL_KEYS, how="inner").merge(
            cut_df, on=ENROLL_KEYS, how="left")
        sub = a[a["date_submitted"] < a["_cut"]].copy()

        if len(sub):
            g = sub.groupby(ENROLL_KEYS)
            def _wavg(d):
                w = d["weight"].to_numpy(dtype=float)
                s = d["score"].to_numpy(dtype=float)
                return float(np.average(s, weights=w)) if w.sum() > 0 else float(np.nanmean(s))
            agg = pd.DataFrame({
                "n_submitted": g.size(),
                "mean_score": g["score"].mean(),
                "weighted_score": g.apply(_wavg),
                "on_time_rate": g.apply(
                    lambda d: float((d["date_submitted"] <= d["a_deadline"]).mean())),
                "banked_any": g["is_banked"].max().clip(upper=1),
            }).reset_index()
            agg["submitted_any"] = 1.0
        else:
            agg = pd.DataFrame(columns=ENROLL_KEYS + [
                "n_submitted", "mean_score", "weighted_score", "on_time_rate",
                "banked_any", "submitted_any"])

        # available assessments per enrollment: distinct id_assessment for its
        # (module, presentation) whose deadline is before this enrollment's cut.
        # ponytail: proxy "available" by assessments that appear in the data; with
        # thousands of students every assessment is submitted by someone.
        amd = assess.drop_duplicates(
            ["code_module", "code_presentation", "id_assessment"])[
            ["code_module", "code_presentation", "id_assessment", "a_deadline"]]
        kc = keys.merge(cut_df, on=ENROLL_KEYS, how="left").merge(
            amd, on=["code_module", "code_presentation"], how="left")
        kc["_avail"] = (kc["a_deadline"] < kc["_cut"]).astype(float)
        navail = kc.groupby(ENROLL_KEYS)["_avail"].sum().rename("n_available")

        out = keys.merge(agg, on=ENROLL_KEYS, how="left").merge(
            navail.reset_index(), on=ENROLL_KEYS, how="left")
        out["submitted_any"] = out["submitted_any"].fillna(0.0).astype(np.float32)
        out["banked_any"] = out["banked_any"].fillna(0.0).astype(np.float32)
        out["n_submitted"] = out["n_submitted"].fillna(0.0).astype(np.float32)
        denom = out["n_available"].fillna(0.0).to_numpy(dtype=float)
        num = out["n_submitted"].to_numpy(dtype=float)
        ratio = np.divide(num, denom, out=np.zeros_like(num), where=denom > 0)
        out["submission_ratio"] = np.clip(ratio, 0.0, 1.0).astype(np.float32)
        # mean_score / weighted_score / on_time_rate left NaN when no submission
        return out.drop(columns=["n_available"])

    def build_static(self, target: pd.DataFrame, assess: pd.DataFrame,
                     keys: pd.DataFrame, fit: bool = False) -> np.ndarray:
        """Dense [N, S] static matrix for enrollments `keys`, aligned to keys order.

        fit=True pins the StaticEncoder vocab/stats on these rows (train pool);
        fit=False reuses the already-fit encoder (test). Leak-free: assessment
        features use only date_submitted < first_deadline.
        """
        keys = keys[ENROLL_KEYS].drop_duplicates().reset_index(drop=True)
        demo = keys.merge(target[ENROLL_KEYS + CAT_COLS + NUM_COLS],
                          on=ENROLL_KEYS, how="left")
        ea = self._assessment_features(target, assess, keys)
        df = demo.merge(ea, on=ENROLL_KEYS, how="left")
        self._last_submitted_any = df["submitted_any"].to_numpy(dtype=np.float32)

        # numeric cols handled by the encoder: demographics + assessment aggregates
        num_cols = NUM_COLS + ["n_submitted", "mean_score", "weighted_score",
                               "on_time_rate", "submission_ratio"]
        if fit or self.enc is None:
            self.enc = StaticEncoder(cat_cols=CAT_COLS, num_cols=num_cols)
            self.enc.fit(df)
        encoded = self.enc.transform(df)                      # [N, enc.n_static]
        flags = df[["submitted_any", "banked_any"]].to_numpy(dtype=np.float32)
        out = np.concatenate([encoded, flags], axis=1).astype(np.float32)
        self.n_static = out.shape[1]
        return out

    def _load_assessments(self) -> pd.DataFrame:
        a = pd.read_csv(self.data_dir / "studentAssessment.csv")
        m = pd.read_csv(self.data_dir / "assessments.csv")
        m = m[["id_assessment", "code_module", "code_presentation", "date", "weight"]]
        out = a.merge(m, on="id_assessment", how="left").rename(
            columns={"date": "a_deadline"})
        out["a_deadline"] = pd.to_numeric(out["a_deadline"], errors="coerce")
        out["weight"] = pd.to_numeric(out["weight"], errors="coerce").fillna(0.0)
        out["score"] = pd.to_numeric(out["score"], errors="coerce")
        return out

    def _load_keys(self, name: str) -> pd.DataFrame:
        df = pd.read_parquet(self.processed_dir / "splits" / f"{name}.parquet")
        return df[ENROLL_KEYS]

    def build(self, epochs: int = EPOCHS) -> dict:
        daily = pd.read_parquet(self.processed_dir / "processed" / "daily_vle.parquet")
        target = pd.read_parquet(self.processed_dir / "processed" / "student_target.parquet")
        assess = self._load_assessments()
        self._acts = activity_vocab(daily)

        tr_keys = self._load_keys("train")
        va_keys = self._load_keys("val")
        te_keys = self._load_keys("test")

        pool_keys = pd.concat([tr_keys, va_keys], ignore_index=True)
        Xp, lp, yp, tp, ep = self.build_sequences(daily, target, pool_keys, self._acts)
        Sp = self.build_static(target, assess, pool_keys, fit=True)
        trend_p = click_trend_features(Xp.sum(axis=2), lp)
        feat_p = np.concatenate([Sp, trend_p], axis=1).astype(np.float32)

        Xte, lte, yte, tte, ete = self.build_sequences(daily, target, te_keys, self._acts)
        Ste = self.build_static(target, assess, te_keys, fit=False)
        trend_te = click_trend_features(Xte.sum(axis=2), lte)
        feat_te = np.concatenate([Ste, trend_te], axis=1).astype(np.float32)

        m = _strat_mask(yp, frac=0.85, seed=SEED)
        self.train(Xp[m], lp[m], Sp[m], yp[m], tp[m], ep[m],
                   Xp[~m], lp[~m], Sp[~m], yp[~m], tp[~m], ep[~m], epochs=epochs)

        lstm_metrics = self.evaluate(Xte, lte, Ste, yte, tte, ete)
        lstm_prob = self._lstm_prob(Xte, lte, Ste)
        xgb_prob = self._xgb_prob(feat_p[m], yp[m], feat_p[~m], yp[~m], feat_te)
        ens_prob = 0.5 * (lstm_prob + xgb_prob)

        def _auc(p):
            try:
                return float(roc_auc_score(yte, p))
            except ValueError:
                return 0.5
        xgb_auc = _auc(xgb_prob)
        ens_auc = _auc(ens_prob)
        ens_brier = float(brier_score_loss(yte, ens_prob)) if len(set(yte.tolist())) > 1 \
            else float(np.mean((ens_prob - yte) ** 2))

        base = {"n_train": int(len(Xp)), "n_test": int(len(Xte)),
                "n_pos": int(yp.sum()), "n_static": int(self.n_static)}
        xgb_m = {"dropout_auc": xgb_auc, **base}
        ens_m = {"dropout_auc": ens_auc, "c_index": lstm_metrics["c_index"],
                 "brier": ens_brier, "lstm_auc": lstm_metrics["dropout_auc"],
                 "xgb_auc": xgb_auc, **base}

        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        model_path = self.model_dir / "pred_oulad.pt"
        torch.save(self.model.state_dict(), model_path)
        (self.results_dir / "pred_metrics_xgb.json").write_text(json.dumps(xgb_m, indent=2))
        (self.results_dir / "pred_metrics_ensemble.json").write_text(json.dumps(ens_m, indent=2))
        metrics_path = self.results_dir / "pred_metrics.json"
        metrics_path.write_text(json.dumps(ens_m, indent=2))   # final = ensemble
        logger.info("pred(ensemble) auc=%.4f (lstm=%.4f xgb=%.4f) c_index=%.4f brier=%.4f",
                    ens_auc, lstm_metrics["dropout_auc"], xgb_auc,
                    ens_m["c_index"], ens_brier)
        return {"model_path": str(model_path), "metrics_path": str(metrics_path)}


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description="Train OULAD dropout predictor (static fusion)")
    ap.add_argument("--processed-dir", default="data/oulad")
    ap.add_argument("--data-dir", default="../../датасет/oulad")
    ap.add_argument("--window-frac", type=float, default=None,
                    help="widen click window to date < frac*module_length (fallback B)")
    ap.add_argument("--max-days", type=int, default=None,
                    help="raise sequence length cap (default 61); pairs with --window-frac")
    ap.add_argument("--build", action="store_true")
    args = ap.parse_args()
    if args.max_days:
        global MAX_DAYS
        MAX_DAYS = args.max_days  # ponytail: module-global override, fine for a CLI knob
    if args.build:
        OULADPred(processed_dir=args.processed_dir, data_dir=args.data_dir,
                  window_frac=args.window_frac).build()
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
