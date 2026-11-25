"""Semi-synthetic positive control on REAL OULAD covariates.

The real-data analyses provide observational estimates whose calibrated conclusions
depend on design and control-panel assumptions. A fully synthetic simulation
(sim_signaware.py) shows recovery under a linear SEM, but a reviewer can object
that its confounding structure is idealised. This script adds a semi-synthetic design:
real OULAD students, real demographic covariates, and a real engagement proxy
(pre-deadline VLE clicks) as the latent confounder, into which a KNOWN treatment
effect is injected. Ground truth is therefore known while the confounding and the
negative-control panel come from the actual joint distribution of the data.

Design
------
- g  = real pre-first-deadline cumulative VLE clicks (standardised) = unmeasured
       confounder (engagement). It is NOT given to the nuisance models.
- X  = observed covariates for the AIPW nuisances: code_module, region,
       module length, registration timing. Disjoint from g and from the controls.
- T  ~ Bernoulli(sigmoid(0.9 g + 0.3 noise))         confounded treatment.
- Y  potential outcomes from a logistic model loaded on g (coef B_LOAD) with a
       known treatment log-odds shift theta; the marginal ATE is computed exactly
       from the simulated potential outcomes (Monte-Carlo over the real cohort).
- Negative controls = six synthetic outcomes confounded by the SAME real
       engagement g (common loading B_LOAD) but T-independent, with a designed
       sign split: N_OPP of the six load on g with the opposite sign. The panel
       therefore shares the target's confounder magnitude while spanning both
       sign conventions -> the sign-divergence regime. The real demographic
       attributes alone are too weakly tied to g to carry the target's bias, so
       only the covariates, the confounder g and the treatment mechanism are
       taken from the data; the calibrating outcomes are simulated on top of g.

The AIPW estimator, the Schuemie empirical null, and the sign-aware orientation
are applied exactly as on the real data. The check is whether the calibrated
estimate brackets the KNOWN ATE: sign-aware should, naive should not.

CPU, ~1 min. Output: results/semisynth_positive.json + console summary.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import KFold

DATA = Path(__file__).parents[1] / "education_models" / "data" / "oulad" / "processed"
OUT = Path(__file__).parent / "results" / "semisynth_positive.json"
KEYS = ["code_module", "code_presentation", "id_student"]
SEED = 20260626
THETA = 0.55     # treatment log-odds shift on the target -> known ATE
B_LOAD = 0.85    # shared confounder loading (g) on every outcome's logit
N_OPP = 4        # controls anti-correlated with g (of 6) -> f = 4/6 sign-opposed
Z = 1.959963985


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def load() -> pd.DataFrame:
    tgt = pd.read_parquet(DATA / "student_target.parquet")
    vle = pd.read_parquet(DATA / "daily_vle.parquet")
    # engagement proxy g: clicks strictly before the first deadline.
    dl = tgt[KEYS + ["first_deadline"]]
    v = vle.merge(dl, on=KEYS, how="inner")
    pre = v[v["date"] <= v["first_deadline"]]
    g = pre.groupby(KEYS)["sum_click"].sum().rename("clicks")
    df = tgt.merge(g, on=KEYS, how="left")
    df["clicks"] = df["clicks"].fillna(0.0)
    return df


def aipw(y: np.ndarray, t: np.ndarray, X: np.ndarray, rng) -> tuple[float, float]:
    """Cross-fit AIPW ATE with gradient-boosted nuisances (binary y)."""
    n = len(y)
    e = np.zeros(n); m0 = np.zeros(n); m1 = np.zeros(n)
    kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
    for tr, te in kf.split(X):
        ps = HistGradientBoostingClassifier(max_iter=120, max_depth=3,
                                            random_state=SEED).fit(X[tr], t[tr])
        e[te] = np.clip(ps.predict_proba(X[te])[:, 1], 0.02, 0.98)
        for arm, mu in ((0, m0), (1, m1)):
            sel = tr[t[tr] == arm]
            if len(np.unique(y[sel])) < 2:        # degenerate arm -> constant
                mu[te] = y[sel].mean()
                continue
            om = HistGradientBoostingClassifier(max_iter=120, max_depth=3,
                                                random_state=SEED).fit(X[sel], y[sel])
            mu[te] = om.predict_proba(X[te])[:, 1]
    psi = m1 - m0 + t * (y - m1) / e - (1 - t) * (y - m0) / (1 - e)
    return float(psi.mean()), float(psi.std(ddof=1) / np.sqrt(n))


def mom_null(taus, ses):
    mu = float(np.mean(taus))
    tau2 = max(0.0, float(np.var(taus, ddof=1) - np.mean(np.square(ses))))
    return mu, np.sqrt(tau2)


def main():
    rng = np.random.default_rng(SEED)
    df = load()
    n = len(df)

    g = (df["clicks"] - df["clicks"].mean()) / (df["clicks"].std() + 1e-9)
    g = g.to_numpy()

    # observed nuisance covariates X (disjoint from g and from the controls).
    Xobs = pd.get_dummies(
        df[["code_module", "region"]].astype(str), drop_first=True
    ).astype(float)
    Xobs["mlen"] = df["module_presentation_length"].fillna(0).to_numpy()
    Xobs["reg"] = df["date_registration"].fillna(df["date_registration"].median()).to_numpy()
    X = Xobs.to_numpy()

    # confounded treatment and known-effect potential outcomes (g loads at B_LOAD).
    t = (rng.standard_normal(n) < 0.9 * g + 0.3 * rng.standard_normal(n)).astype(int)
    eta0 = -0.3 + B_LOAD * g
    y0p, y1p = sigmoid(eta0), sigmoid(eta0 + THETA)
    true_ate = float(np.mean(y1p - y0p))
    u = rng.random(n)
    y = np.where(t == 1, (u < y1p).astype(int), (u < y0p).astype(int))

    # six synthetic negative controls confounded by the SAME g (common magnitude
    # B_LOAD), T-independent, with N_OPP of them sign-opposed to the target.
    ctrl_names = ["nc_pos1", "nc_pos2", "nc_neg1", "nc_neg2", "nc_neg3", "nc_neg4"]
    ctrl_signs = np.array([1.0] * (6 - N_OPP) + [-1.0] * N_OPP)
    # per-control intercepts for varied base rates (real-looking prevalences).
    ctrl_c0 = np.array([0.2, -0.1, 0.0, 0.3, -0.2, 0.1])
    controls = {}
    for nm, s_k, c0 in zip(ctrl_names, ctrl_signs, ctrl_c0):
        p_k = sigmoid(c0 + s_k * B_LOAD * g)
        controls[nm] = (rng.random(n) < p_k).astype(int)

    # target effect estimate (AIPW on observed X only; g stays unmeasured).
    obs, obs_se = aipw(y, t, X, rng)

    taus, ses, orient, names = [], [], [], []
    s_y = np.sign(np.corrcoef(y, g)[0, 1]) or 1.0
    for nm, yc in controls.items():
        tk, sk_se = aipw(yc, t, X, rng)
        s_k = np.sign(np.corrcoef(yc, g)[0, 1]) or 1.0
        taus.append(tk); ses.append(sk_se)
        orient.append((s_k * s_y) * tk)
        names.append(nm)
    taus, ses = np.array(taus), np.array(ses)
    orient = np.array(orient)

    mu_n, sd_n = mom_null(taus, ses)
    mu_a, sd_a = mom_null(orient, ses)
    cal_n = obs - mu_n
    cal_a = obs - s_y * mu_a
    se_n = np.sqrt(obs_se**2 + sd_n**2)
    se_a = np.sqrt(obs_se**2 + sd_a**2)

    def ci(est, se):
        return [round(est - Z * se, 4), round(est + Z * se, 4)]

    def covers(est, se):
        lo, hi = est - Z * se, est + Z * se
        return bool(lo <= true_ate <= hi)

    def block(est, se, mu=None, tau=None):
        d = {"ate": round(est, 4), "ci": ci(est, se),
             "abs_bias": round(abs(est - true_ate), 4),
             "ci_width": round(2 * Z * se, 4),
             "covers_truth": covers(est, se)}
        if mu is not None:
            d = {"mu": round(mu, 4), "tau": round(tau, 4), **d}
        return d

    res = {
        "n": n, "true_ate": round(true_ate, 4), "theta_logodds": THETA,
        "b_load": B_LOAD, "n_opposed": N_OPP, "treat_rate": round(float(t.mean()), 3),
        "uncalibrated": block(obs, obs_se),
        "naive": block(cal_n, se_n, mu_n, sd_n),
        "sign_aware": block(cal_a, se_a, float(s_y * mu_a), sd_a),
        "panel": [{"control": nm, "raw": round(float(tk), 4),
                   "oriented": round(float(o), 4)}
                  for nm, tk, o in zip(names, taus, orient)],
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
