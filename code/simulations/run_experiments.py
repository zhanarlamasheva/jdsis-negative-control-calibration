"""Finite-sample stress tests of the manuscript's calibration step; NumPy only.

This is NOT a replication of the AIPW analyses or an implementation of the
published EmpiricalCalibration package. It tests the manuscript's arithmetic
mean / method-of-moments (MoM) formulas using difference-in-means estimates.
"""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import platform
import time
import numpy as np

ROOT = Path(__file__).resolve().parent
SIGNS = np.array([1, -1, 1, -1, 1, -1])
Z = 1.959963984540054


def scenarios():
    result = []
    def add(name, experiment, target, controls, weights=(1., 0.), scales=None, shifts=None):
        result.append(dict(name=name, experiment=experiment, target=list(target),
                           controls=np.asarray(controls).tolist(), weights=list(weights),
                           scales=list(np.ones(6) if scales is None else scales),
                           shifts=list(np.zeros(6) if shifts is None else shifts)))
    base = np.column_stack((SIGNS * .1, np.zeros(6)))
    for ratio in [.25, .5, 1., 2., 4.]:
        add(f'magnitude_ratio_{ratio:g}', '1_magnitude', [.1, 0], base * ratio)
    unequal = np.array([.25, .5, .75, 1.25, 1.5, 1.75])
    add('heterogeneous_mean_matched', '1_magnitude', [.1, 0], base * unequal[:, None])
    add('heterogeneous_mean_mismatched', '1_magnitude', [.1, 0], base * (2 * unequal[:, None]))
    add('two_factors_aligned', '2_confounders', [.05, .05],
        SIGNS[:, None] * np.tile([.05, .05], (6, 1)), (1., 1.))
    add('two_factors_proxy_blind', '2_confounders', [.05, .05],
        SIGNS[:, None] * np.tile([-.1, .2], (6, 1)), (1., 1.))
    mixed = np.array([[.2, -.1]] * 3 + [[-.1, .2]] * 3)
    add('two_factors_proxy_mixed', '2_confounders', [.05, .05],
        SIGNS[:, None] * mixed, (1., 1.))
    for scale in [.1, 1., 10.]:
        add(f'all_units_{scale:g}', '3_units', [.1, 0], base, scales=np.full(6, scale))
    one = np.ones(6); one[0] = 10
    add('one_control_units_10', '3_units', [.1, 0], base, scales=one)
    add('affine_shift_100', '3_units', [.1, 0], base, shifts=np.full(6, 100.))
    add('reverse_all_units', '3_units', [.1, 0], base, scales=np.full(6, -1.))
    return result


def population(s):
    w = np.array(s['weights'])
    # A=1[w.U + eta > 0], U,eta independent standard normal.
    delta = 2 * np.sqrt(2 / np.pi) * w / np.sqrt(1 + w @ w)
    by = float(np.array(s['target']) @ delta)
    bc = np.array(s['controls']) @ delta * np.array(s['scales'])
    # g=U1 + independent noise. Cov(Y,g)=target[0]+effect*Cov(A,U1).
    return by, bc


def estimates(values, a):
    n = len(a); p = a.mean()
    m1 = values[a].mean(axis=0); m0 = values[~a].mean(axis=0)
    influence = a[:, None] / p * (values - m1) - (~a)[:, None] / (1-p) * (values - m0)
    return m1-m0, influence.var(axis=0, ddof=1)/n, influence


def calibrations(values, a, g, scenario):
    d, v, influence = estimates(values, a)
    y = values[:, 0]; controls = values[:, 1:]
    centered_g = g - g.mean()
    sg = np.sign((controls-controls.mean(axis=0)).T @ centered_g)
    sy = np.sign((y-y.mean()) @ centered_g)
    sign = sg * sy
    by, bc = population(scenario)
    oracle_sign = np.sign(bc) * np.sign(by)
    sd_ratio = y.std(ddof=1) / controls.std(axis=0, ddof=1)
    answer = {'uncalibrated': (d[0], np.sqrt(v[0]))}
    for name, factor in [('manuscript_mom_raw', np.ones(6)),
                         ('sign_mom', sign),
                         ('standardized_sign_mom', sign * sd_ratio),
                         ('oracle_sign_mom', oracle_sign)]:
        dc = d[1:] * factor
        vc = v[1:] * factor**2
        heterogeneity = max(0., dc.var(ddof=1)-vc.mean())
        answer[name] = (d[0]-dc.mean(), np.sqrt(v[0]+heterogeneity))
    # Known DGP transfer coefficients: infeasible benchmark, not a proposed method.
    transfer = by / bc
    adjusted_if = influence[:, 0] - (influence[:, 1:] * transfer).mean(axis=1)
    answer['oracle_transfer_covariance'] = (
        d[0]-(d[1:]*transfer).mean(), np.sqrt(adjusted_if.var(ddof=1)/len(a)))
    return answer, float(np.mean(sign != oracle_sign))


def summarize(draws, truth):
    est = np.array([x[0] for x in draws]); se = np.array([x[1] for x in draws])
    error = est-truth
    coverage = np.mean(np.abs(error) <= Z*se)
    rejection = np.mean(np.abs(est) > Z*se)
    return dict(bias=float(error.mean()), bias_mcse=float(error.std(ddof=1)/np.sqrt(len(est))),
                rmse=float(np.sqrt(np.mean(error**2))), coverage=float(coverage),
                coverage_mcse=float(np.sqrt(coverage*(1-coverage)/len(est))),
                mean_width=float((2*Z*se).mean()), rejection_rate=float(rejection),
                rejection_mcse=float(np.sqrt(rejection*(1-rejection)/len(est))),
                rejection_interpretation='type_I_error' if truth == 0 else 'power')


def verify():
    # Exact algebra: homogeneous sign correction; heterogeneous residual;
    # units break raw correction; partial proxy can reverse actual bias direction.
    ss = {s['name']: s for s in scenarios()}
    by, bc = population(ss['magnitude_ratio_1'])
    assert np.isclose(by-np.abs(bc).mean(), 0)
    by, bc = population(ss['magnitude_ratio_2'])
    assert np.isclose(by-np.abs(bc).mean(), -by)
    by, bc = population(ss['all_units_10'])
    assert np.isclose(by-np.abs(bc).mean(), -9*by)
    s = ss['two_factors_proxy_blind']; by, bc = population(s)
    proxy_sign = np.sign(np.array(s['controls'])[:, 0])
    assert np.isclose(by-(proxy_sign*bc).mean(), 2*by)
    # Affine invariance of standardization and expected reverse-code orientation.
    rng = np.random.default_rng(73)
    u = rng.normal(size=(3000, 2)); a = (u[:, 0]+rng.normal(size=3000)) > 0
    y = .1*u[:, 0]+rng.normal(size=3000)
    c = u @ np.array(ss['all_units_1']['controls']).T+rng.normal(size=(3000,6))
    g = u[:, 0]+rng.normal(size=3000)
    outputs = {}
    for name in ['all_units_1', 'all_units_10', 'affine_shift_100', 'reverse_all_units']:
        s = ss[name]
        vals = np.column_stack((y, c*np.array(s['scales'])+np.array(s['shifts'])))
        outputs[name] = calibrations(vals, a, g, s)[0]
    for name in outputs:
        assert np.allclose(outputs[name]['standardized_sign_mom'],
                           outputs['all_units_1']['standardized_sign_mom'], atol=1e-12)
    assert np.allclose(outputs['reverse_all_units']['sign_mom'], outputs['all_units_1']['sign_mom'])
    print('Verification: 8 algebra / affine-invariance checks passed.', flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--replications', type=int, default=1000)
    parser.add_argument('--n', type=int, default=4000)
    parser.add_argument('--seed', type=int, default=20261029)
    parser.add_argument('--verify-only', action='store_true')
    args = parser.parse_args()
    verify()
    if args.verify_only: return
    if args.replications < 2 or args.n < 100: raise ValueError('Need R>=2, n>=100')
    start = time.time(); rows=[]; theory=[]; raw=[]
    out = ROOT/'results'; out.mkdir(exist_ok=True)
    config = dict(seed=args.seed, n=args.n, replications=args.replications,
                  effects=[0., .1], scenarios=scenarios(), python=platform.python_version(),
                  numpy=np.__version__, source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  estimator='difference in means', proxy='U1 + independent N(0,1) noise',
                  interval='manuscript MoM plug-in, except covariance-aware oracle transfer',
                  paired_draws='identical seeds across scenarios and effect settings')
    (out/'config.json').write_text(json.dumps(config, indent=2), encoding='utf-8')
    for s in scenarios():
        by, bc = population(s)
        for truth in [0., .1]:
            rng = np.random.default_rng(args.seed)
            accum={}; wrong=[]
            for rep in range(args.replications):
                u=rng.normal(size=(args.n, 2))
                a=(u @ np.array(s['weights'])+rng.normal(size=args.n)) > 0
                g=u[:, 0]+rng.normal(size=args.n)
                y=truth*a + u @ np.array(s['target']) + rng.normal(size=args.n)
                c=u @ np.array(s['controls']).T + rng.normal(size=(args.n, 6))
                c=c*np.array(s['scales'])+np.array(s['shifts'])
                methods, mis=calibrations(np.column_stack((y,c)), a, g, s)
                wrong.append(mis)
                for method, pair in methods.items():
                    accum.setdefault(method, []).append(pair)
                    raw.append([s['name'], truth, rep, method, *pair])
            for method, draws in accum.items():
                rows.append(dict(experiment=s['experiment'], scenario=s['name'], true_effect=truth,
                                 method=method, n=args.n, replications=args.replications,
                                 sign_error_rate=float(np.mean(wrong)), **summarize(draws, truth)))
            # Population proxy sign includes the treatment contribution to target correlation.
            w=np.array(s['weights'])
            target_cov=s['target'][0]+truth*w[0]/np.sqrt(2*np.pi*(1+w@w))
            orientation=np.sign(np.array(s['controls'])[:,0]*np.array(s['scales']))*np.sign(target_cov)
            theory.append(dict(scenario=s['name'], true_effect=truth, target_bias=by,
                               raw_residual=by-bc.mean(), sign_residual=by-(orientation*bc).mean(),
                               oracle_sign_residual=by-np.abs(bc).mean()))
            print(f"{s['name']} effect={truth:g}: sign bias={rows[-4]['bias']:+.4f}", flush=True)
    def write_csv(path, records):
        with path.open('w', newline='', encoding='utf-8') as f:
            writer=csv.DictWriter(f, fieldnames=list(records[0])); writer.writeheader(); writer.writerows(records)
    write_csv(out/'summary.csv', rows); write_csv(out/'population_truth.csv', theory)
    with (out/'replications.csv').open('w', newline='', encoding='utf-8') as f:
        writer=csv.writer(f); writer.writerow(['scenario','true_effect','replication','method','estimate','standard_error']); writer.writerows(raw)
    # Validate unbiased oracle against known truth with conservative MC tolerance.
    oracle=[r for r in rows if r['method']=='oracle_transfer_covariance']
    assert all(abs(r['bias']) < 5*r['bias_mcse'] for r in oracle), 'Oracle implementation / MC check failed'
    manifest=dict(elapsed_seconds=round(time.time()-start,2), summary_rows=len(rows),
                  replication_rows=len(raw), verification='passed',
                  output_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest()
                                 for p in out.glob('*.csv')})
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print(json.dumps(manifest,indent=2), flush=True)


if __name__ == '__main__':
    main()
