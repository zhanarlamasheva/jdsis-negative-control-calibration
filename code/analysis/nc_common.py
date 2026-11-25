"""Shared estimation and diagnostic code for the reviewer re-analyses.

Mirrors code/analysis/run_real_data_audit.py (cross-fit AIPW with XGBoost, common
support [0.02, 0.98], student-cluster variance, sign-aware equal pooling, MoM
interval) and adds the diagnostics requested in review:
  * proxy-sign stability on the common-support sample, for targets and controls,
    with marginal and covariate-residualised correlations (comment 4);
  * frequency of orientation changes and the signed estimate under resampled
    signs (comment 5);
  * raw, MoM and fixed-sign covariance intervals side by side (comment 5);
  * optional full-pipeline cluster bootstrap that refits all nuisance models.
"""
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold
from xgboost import XGBClassifier

Z = 1.959963984540054


def classifier(seed, fast=False):
    return XGBClassifier(n_estimators=100 if fast else 300, max_depth=4, learning_rate=.05 if not fast else .1,
                         subsample=.8, colsample_bytree=.8, eval_metric='logloss', random_state=seed, n_jobs=4)


def _prob(X, y, Xt, seed, fast):
    if len(np.unique(y)) == 1:
        return np.full(len(Xt), float(y[0]))
    return classifier(seed, fast).fit(X, y).predict_proba(Xt)[:, 1]


def crossfit(X, T, V, groups, seed, fast=False, Xtest=None):
    """Five-fold group cross-fitting. V: n x q binary outcomes (targets then controls)."""
    folds = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed).split(X, T, groups)
    e = np.zeros(len(T)); m0 = np.zeros(V.shape, float); m1 = m0.copy()
    for tr, te in folds:
        e[te] = _prob(X[tr], T[tr], X[te], seed, fast)
        for j in range(V.shape[1]):
            for a, mu in ((0, m0), (1, m1)):
                ii = tr[T[tr] == a]
                mu[te, j] = _prob(X[ii], V[ii, j], X[te], seed, fast)
    hold = None
    if Xtest is not None:
        et = _prob(X, T, Xtest, seed, fast)
        h0 = np.column_stack([_prob(X[T == 0], V[T == 0, j], Xtest, seed, fast) for j in range(V.shape[1])])
        h1 = np.column_stack([_prob(X[T == 1], V[T == 1, j], Xtest, seed, fast) for j in range(V.shape[1])])
        hold = (et, h0, h1)
    return e, m0, m1, hold


def cluster_cov(infl, groups):
    _, idx = np.unique(groups, return_inverse=True); k = int(idx.max() + 1)
    s = np.zeros((k, infl.shape[1])); np.add.at(s, idx, infl)
    return s.T @ s / len(groups) ** 2 * k / max(1, k - 1)


def aipw(T, V, e, m0, m1, groups):
    keep = (e >= .02) & (e <= .98)
    ee = np.clip(e[keep], .02, .98)[:, None]; a = T[keep, None]; y = V[keep]
    psi = m1[keep] - m0[keep] + a * (y - m1[keep]) / ee - (1 - a) * (y - m0[keep]) / (1 - ee)
    est = psi.mean(axis=0)
    return est, cluster_cov(psi - est, groups[keep]), keep


def corr(x, y):
    return float(np.corrcoef(x, y)[0, 1]) if np.std(x) > 0 and np.std(y) > 0 else 0.


def sgn(x):
    s = np.sign(x); return np.where(s == 0, 1, s)


def mom(t, se):
    mu = float(np.mean(t)); h2 = max(0., float(np.var(t, ddof=1) - np.mean(np.asarray(se) ** 2))) if len(t) > 1 else 0.
    return mu, float(np.sqrt(h2))


def calibrate(est, cov, rho, tj, ctrl):
    """Sign-aware equal pooling for target index tj with control indices ctrl."""
    se = np.sqrt(np.diag(cov))
    o = sgn(rho[ctrl]) * sgn(rho[tj])
    mu, h = mom(est[ctrl] * o, se[ctrl])
    cal = float(est[tj] - mu); w_mom = float(np.sqrt(se[tj] ** 2 + h ** 2))
    a = np.zeros(len(est)); a[tj] = 1; a[ctrl] = -o / len(ctrl)
    w_cov = float(np.sqrt(max(0., a @ cov @ a)))
    return dict(raw=float(est[tj]), raw_ci=[float(est[tj] - Z * se[tj]), float(est[tj] + Z * se[tj])],
                orientation=o.astype(int).tolist(), signed_mean=mu, spread=h, corrected=cal,
                mom_ci=[cal - Z * w_mom, cal + Z * w_mom], cov_ci=[cal - Z * w_cov, cal + Z * w_cov])


def loco(est, cov, rho, ctrl, hold=None):
    """Each control in turn as a pseudo-target with a known null."""
    se = np.sqrt(np.diag(cov)); rows = []
    for j in ctrl:
        other = [k for k in ctrl if k != j]
        o = sgn(rho[other]) * sgn(rho[j]); mu, h = mom(est[other] * o, se[other])
        w = np.sqrt(se[j] ** 2 + h ** 2); p = est[j] - mu
        r = dict(control=j, raw=float(est[j]), corrected=float(p), excludes_zero=bool(abs(p) > Z * w))
        if hold is not None:
            he, hc = hold; hse = np.sqrt(hc[j, j]); hp = he[j] - mu; hw = np.sqrt(hse ** 2 + h ** 2)
            r.update(holdout_raw=float(he[j]), holdout_corrected=float(hp), holdout_excludes_zero=bool(abs(hp) > Z * hw))
        rows.append(r)
    out = dict(rows=rows, raw_rmse=float(np.sqrt(np.mean([r['raw'] ** 2 for r in rows]))),
               corrected_rmse=float(np.sqrt(np.mean([r['corrected'] ** 2 for r in rows]))),
               excluded=sum(r['excludes_zero'] for r in rows), k=len(rows))
    if hold is not None:
        out.update(holdout_raw_rmse=float(np.sqrt(np.mean([r['holdout_raw'] ** 2 for r in rows]))),
                   holdout_corrected_rmse=float(np.sqrt(np.mean([r['holdout_corrected'] ** 2 for r in rows]))),
                   holdout_excluded=sum(r['holdout_excludes_zero'] for r in rows))
    return out


def residualise(y, X):
    Xd = np.column_stack((np.ones(len(X)), X))
    beta = np.linalg.lstsq(Xd, y, rcond=None)[0]
    return y - Xd @ beta


def sign_bootstrap(V, g, X, groups, names, est, cov, targets, ctrl, reps=2000, seed=20261029):
    # X: covariates used for residualisation; must not contain the proxy itself
    """Student-cluster bootstrap of proxy signs on the analysis (common-support) sample.

    Returns per-variable retention for marginal and residualised correlations, the
    frequency with which the orientation vector of each target changes, and the
    distribution of the signed estimate when only the signs are resampled."""
    _, gi = np.unique(groups, return_inverse=True); k = gi.max() + 1
    gr = residualise(g, X); Vr = np.column_stack([residualise(V[:, j].astype(float), X) for j in range(V.shape[1])])
    def mom_sums(gv, Vv):
        m = np.column_stack((np.ones(len(gv)), gv, gv * gv, Vv, Vv * Vv, gv[:, None] * Vv))
        s = np.zeros((k, m.shape[1])); np.add.at(s, gi, m); return s
    q = V.shape[1]
    def rho_from(t):
        n = t[:, :1]; sg = t[:, 1:2]; sgg = t[:, 2:3]
        sv = t[:, 3:3 + q]; svv = t[:, 3 + q:3 + 2 * q]; sgv = t[:, 3 + 2 * q:3 + 3 * q]
        return (sgv - sg * sv / n) / np.sqrt(np.maximum((sgg - sg * sg / n) * (svv - sv * sv / n), 1e-30))
    S_marg = mom_sums(g, V.astype(float)); S_res = mom_sums(gr, Vr)
    rng = np.random.default_rng(seed)
    W = rng.multinomial(k, np.ones(k) / k, size=reps)
    out = {}
    se = np.sqrt(np.diag(cov))
    for label, S in (('marginal', S_marg), ('residualised', S_res)):
        point = rho_from(S.sum(axis=0)[None, :])[0]; draws = rho_from(W @ S)
        out[label] = {names[j]: dict(correlation=float(point[j]), ci95=np.quantile(draws[:, j], [.025, .975]).tolist(),
                                     sign_retention=float(np.mean(sgn(draws[:, j]) == sgn(point[j]))))
                      for j in range(q)}
        if label == 'marginal':
            for tj in targets:
                o0 = sgn(point[ctrl]) * sgn(point[tj]); od = sgn(draws[:, ctrl]) * sgn(draws[:, [tj]])
                signed = est[tj] - (est[ctrl][None, :] * od).mean(axis=1)
                out.setdefault('orientation_changes', {})[names[tj]] = dict(
                    any_control_flips=float(np.mean((od != o0).any(axis=1))),
                    per_control_flip_rate={names[c]: float(np.mean(od[:, i] != o0[i])) for i, c in enumerate(ctrl)},
                    signed_estimate_sign_only_ci95=np.quantile(signed, [.025, .975]).tolist())
    return out


def stable_panel(boot, names, ctrl, threshold):
    return [c for c in ctrl if boot['marginal'][names[c]]['sign_retention'] >= threshold]


def full_pipeline_bootstrap(build, groups, reps, seed, targets, ctrl, point_orientation):
    """Resample students, refit every nuisance model, re-estimate signs and pooling.
    build(index_array, new_cluster_labels) must return (est, cov, rho) for the resampled rows."""
    rng = np.random.default_rng(seed); ug = np.unique(groups)
    pos = {g: np.flatnonzero(groups == g) for g in ug}
    res = {t: [] for t in targets}; flips = {t: 0 for t in targets}
    for b in range(reps):
        pick = rng.choice(ug, len(ug), replace=True)
        idx = np.concatenate([pos[g] for g in pick])
        newg = np.concatenate([np.full(len(pos[g]), i) for i, g in enumerate(pick)])
        est, cov, rho = build(idx, newg)
        for t in targets:
            c = calibrate(est, cov, rho, t, ctrl); res[t].append((c['raw'], c['corrected']))
            flips[t] += int(list(c['orientation']) != list(point_orientation[t]))
        print(f'  pipeline bootstrap {b + 1}/{reps}', flush=True)
    return {t: dict(raw_ci95=np.quantile([r[0] for r in v], [.025, .975]).tolist(),
                    corrected_ci95=np.quantile([r[1] for r in v], [.025, .975]).tolist(),
                    orientation_differs_from_point_estimate=flips[t] / reps) for t, v in res.items()}
