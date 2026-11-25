"""Reviewer scenario: a factor that biases every control equally but not the target.
Reuses the data-generating code of run_binary_validation.py (Table 7) and adds the
leave-one-control-out (LOCO) null-prediction diagnostic computed inside each replication."""
import sys, json, csv
import numpy as np
sys.path.insert(0, sys.argv[1])
import run_binary_validation as rb

SIGNS = rb.SIGNS
def loco(d, values, g):
    gc = g - g.mean(); vc = values - values.mean(axis=0)
    cs = np.sign(vc.T @ gc); cs[cs == 0] = 1
    ct = d[1:] * cs[1:]            # controls oriented by their own proxy sign
    raw = np.sqrt(np.mean(d[1:] ** 2))
    corr = np.sqrt(np.mean([(ct[k] - np.delete(ct, k).mean()) ** 2 for k in range(len(ct))]))
    return raw, corr

SCEN = [dict(name='matched', w=[1., 0.], ly=[.8, 0.], lc=[.8, 0.]),
        dict(name='control_only_factor', w=[1., 1.], ly=[.8, 0.], lc=[.8, .8])]
R = int(sys.argv[2]) if len(sys.argv) > 2 else 500
rows = []
for s in SCEN:
    for alpha in [0., .5]:
        truth = rb.true_ate(alpha, s['ly']); rng = np.random.default_rng(20261031)
        est = {'uncalibrated': [], 'sign_mom': []}; lr = []; lc_ = []
        for rep in range(R):
            n = 2000; u = rng.normal(size=(n, 2)); x = rng.normal(size=n)
            a = (u @ np.array(s['w']) + .5 * x + rng.normal(size=n) > 0).astype(float)
            g = u[:, 0] + rng.normal(size=n)
            y = rng.binomial(1, rb.expit(alpha * a + u @ np.array(s['ly']) + .4 * x))
            c = rng.binomial(1, rb.expit((u @ np.array(s['lc']) + .4 * x)[:, None] * SIGNS))
            values = np.column_stack((y, c))
            fold = np.zeros(n, dtype=int); fold[rng.permutation(n)[:n // 2]] = 1
            design = np.column_stack((np.ones(n), x, g))
            d, inf, _ = rb.aipw(values, a, design, fold)
            res = rb.calibrate(d, inf, values, g)
            for m in est: est[m].append(res[m])
            r0, r1 = loco(d, values, g); lr.append(r0); lc_.append(r1)
        lr, lc_ = np.array(lr), np.array(lc_)
        for m, e in est.items():
            e = np.array(e); err = e[:, 0] - truth
            rows.append(dict(scenario=s['name'], alpha=alpha, true_ate=round(truth, 4), method=m,
                bias=round(err.mean(), 4), rmse=round(np.sqrt(np.mean(err ** 2)), 4),
                coverage=round(np.mean((e[:, 1] <= truth) & (truth <= e[:, 2])), 3),
                reject_zero=round(np.mean((e[:, 1] > 0) | (e[:, 2] < 0)), 3),
                loco_raw_rmse=round(lr.mean(), 4), loco_corrected_rmse=round(lc_.mean(), 4),
                loco_pass_rate=round(np.mean(lc_ < lr), 3)))
        print(rows[-1], flush=True)
with open('/home/claude/sim/control_only_factor.csv', 'w', newline='') as f:
    wr = csv.DictWriter(f, fieldnames=list(rows[0])); wr.writeheader(); wr.writerows(rows)
