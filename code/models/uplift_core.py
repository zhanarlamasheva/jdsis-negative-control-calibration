"""Domain-agnostic causal-uplift estimator: cross-fit nuisances, X-learner CATE,
doubly-robust AIPW ATE, E-value sensitivity, Qini/AUUC/policy metrics.

Shared by OULADUplift and XES3G5MUplift; each subclass supplies only the domain
treatment/covariate/outcome builder (`_build_TXY`) and the `build()` driver.
"""

from __future__ import annotations

import numpy as np
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier, XGBRegressor

from models.pred_oulad import SEED


class CausalUplift:
    def __init__(self, k_folds: int = 5, seed: int = SEED):
        self.k_folds = k_folds
        self.seed = seed

    def _xgb(self):
        return XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05,
                             subsample=0.8, colsample_bytree=0.8, eval_metric="logloss",
                             random_state=self.seed, n_jobs=4)

    def _xreg(self):
        return XGBRegressor(n_estimators=300, max_depth=4, learning_rate=0.05,
                            subsample=0.8, colsample_bytree=0.8,
                            random_state=self.seed, n_jobs=4)

    def _oof_prob(self, X, y, folds):
        p = np.zeros(len(y), dtype=float)
        for tr, te in folds:
            m = self._xgb().fit(X[tr], y[tr])
            p[te] = m.predict_proba(X[te])[:, 1]
        return p

    def _fit_proba1(self, X, y, Xpred):
        """P(y=1|Xpred); constant fallback when y is single-class (degenerate arm)."""
        u = np.unique(y)
        if len(u) < 2:
            return np.full(len(Xpred), float(u[0]))
        return self._xgb().fit(X, y).predict_proba(Xpred)[:, 1]

    def _fit_nuisances(self, X, T, Y):
        """Cross-fit out-of-fold propensity e(x) and outcomes mu0(x), mu1(x)."""
        skf = StratifiedKFold(n_splits=self.k_folds, shuffle=True,
                              random_state=self.seed)
        folds = list(skf.split(X, T))
        e = self._oof_prob(X, T, folds)
        mu0 = np.zeros(len(Y), dtype=float)
        mu1 = np.zeros(len(Y), dtype=float)
        for tr, te in folds:
            tr0 = tr[T[tr] == 0]
            tr1 = tr[T[tr] == 1]
            if len(tr0) < 2 or len(tr1) < 2:
                raise ValueError(
                    f"fold has {len(tr0)} controls / {len(tr1)} treated — "
                    "reduce k_folds or check treatment overlap")
            mu0[te] = self._fit_proba1(X[tr0], Y[tr0], X[te])
            mu1[te] = self._fit_proba1(X[tr1], Y[tr1], X[te])
        return e, mu0, mu1

    def _xlearner_tau(self, X, T, Y, e, mu0, mu1):
        """X-learner CATE: imputed-effect regressions blended by propensity."""
        idx1 = np.where(T == 1)[0]
        idx0 = np.where(T == 0)[0]
        d1 = Y[idx1] - mu0[idx1]
        d0 = mu1[idx0] - Y[idx0]
        tau1_model = self._xreg().fit(X[idx1], d1)
        tau0_model = self._xreg().fit(X[idx0], d0)
        tau1 = tau1_model.predict(X)
        tau0 = tau0_model.predict(X)
        return e * tau0 + (1.0 - e) * tau1

    @staticmethod
    def _aipw_ate(T, Y, e, mu0, mu1, clip=(0.02, 0.98)):
        """Doubly-robust AIPW ATE, with influence-function CI95."""
        e = np.clip(e, *clip)
        psi = (mu1 - mu0
               + T / e * (Y - mu1)
               - (1 - T) / (1 - e) * (Y - mu0))
        ate = float(psi.mean())
        se = float(psi.std(ddof=1) / np.sqrt(len(psi)))
        ci = [ate - 1.96 * se, ate + 1.96 * se]
        return ate, se, ci


def qini_curve(tau, T, Y):
    """Qini curve points (fractions[n+1], uplift[n+1]); standard Qini definition."""
    order = np.argsort(-np.asarray(tau))
    t = np.asarray(T)[order]
    y = np.asarray(Y)[order]
    yt = np.cumsum(y * t)
    yc = np.cumsum(y * (1 - t))
    nt = np.cumsum(t)
    nc = np.cumsum(1 - t)
    with np.errstate(divide="ignore", invalid="ignore"):
        uplift = yt - yc * np.where(nc > 0, nt / nc, 0.0)
    n = len(tau)
    frac = np.arange(0, n + 1) / n
    return frac, np.concatenate([[0.0], uplift])


def qini_coefficient(tau, T, Y):
    """Qini AREA between the Qini curve and the random diagonal.

    # ponytail: raw area (units = extra positive-outcome students), NOT normalized
    # to a perfect-ranker coefficient, so dataset-scale-dependent. Within-study
    # ranking diagnostic; normalize by the oracle-curve area only if a reviewer needs
    # a cross-dataset-comparable coefficient.
    """
    frac, uplift = qini_curve(tau, T, Y)
    # NumPy 2.0 removed ``trapz``; ``trapezoid`` is the supported equivalent.
    integrate = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    area_model = integrate(uplift, frac)
    rand = uplift[-1] * frac
    area_rand = integrate(rand, frac)
    return float(area_model - area_rand)


def auuc(tau, T, Y):
    frac, uplift = qini_curve(tau, T, Y)
    integrate = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    return float(integrate(uplift, frac))


def policy_value_at_k(tau, k=0.1):
    """Mean predicted uplift among the top-k fraction by tau.

    # ponytail: mean predicted tau in the top slice; upgrade to an AIPW policy value
    # (realized uplift of the treat-top-k policy) if a reviewer asks.
    """
    tau = np.asarray(tau)
    n = max(1, int(round(k * len(tau))))
    top = np.sort(tau)[::-1][:n]
    return float(top.mean())


def evalue_from_rr(rr):
    """E-value for a risk ratio (VanderWeele & Ding 2017). RR<1 inverted first."""
    rr = float(rr)
    if rr <= 0:
        return 1.0
    if rr < 1.0:
        rr = 1.0 / rr
    return rr + np.sqrt(rr * (rr - 1.0))
