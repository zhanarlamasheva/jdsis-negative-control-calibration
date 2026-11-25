"""Known-truth binary benchmark; design fixed in PROTOCOL.md before execution."""
from pathlib import Path
import sys
sys.dont_write_bytecode = True
import argparse
import csv
import gzip
import hashlib
import json
import time
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
WORK = ROOT.parent
OUT = ROOT / 'results'
Z95 = 1.959963984540054
SIGNS = np.array([1., -1., 1., -1., 1., -1.])


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def populations():
    # Exact finite synthetic population: every stratum and latent combination.
    cells = np.array([(z, u, v) for z in range(3) for u in [-1., 1.] for v in [-1., 1.]])
    project = WORK.parent.parent / 'education_models'
    target_path = project / 'data/oulad/processed/student_target.parquet'
    split_path = project / 'data/oulad/splits/train.parquet'
    keys = ['id_student', 'code_module', 'code_presentation']
    frame = pd.read_parquet(split_path)[keys].merge(pd.read_parquet(target_path), on=keys, validate='one_to_one')
    frame = frame.sort_values(keys).drop_duplicates('id_student', keep='first')
    age = frame.age_band.map({'0-35': -1., '35-55': .4, '55<=': 1.})
    if age.isna().any():
        raise ValueError('Unrecognized age coding: ' + str(frame.age_band.unique()))
    strata = np.where(frame.studied_credits < 60, 0, np.where(frame.studied_credits < 120, 1, 2))
    empirical = np.column_stack((strata, age, np.where(frame.num_of_prev_attempts > 0, 1., -1.)))
    result = {}
    for name, a in [('synthetic', cells), ('oulad_profile', empirical)]:
        z = a[:, 0].astype(int)
        p = np.bincount(z, minlength=3) / len(z)
        means = np.array([a[z == k, 1:].mean(axis=0) for k in range(3)])
        cov = np.array([np.cov(a[z == k, 1:], rowvar=False, ddof=0) for k in range(3)])
        result[name] = dict(rows=a, p=p, means=means, cov=cov, sd=float(a[:, 1].std()), mean=float(a[:, 1].mean()))
    inventory = dict(oulad_unique_students=len(frame), source_role='Empirical baseline distribution only; all exposures/outcomes/proxies generated',
                     hashes={str(p): digest(p) for p in [target_path, split_path]},
                     population_aggregates={k: dict(n=len(v['rows']), stratum_weights=v['p'].tolist(),
                         latent_means=v['means'].tolist(), latent_covariances=v['cov'].tolist()) for k, v in result.items()})
    return result, inventory


def scenarios():
    result = []
    def add(family, scale=1., rho=.75, tau=0., n=2000, source='synthetic', kappa=.15):
        ly = np.array([.14, 0.])
        lc = SIGNS[:, None] * np.array([.14 * scale, 0.])[None, :]
        gamma = np.zeros(6)
        trend = np.zeros(6)
        assignment = np.array([kappa, 0.])
        proxy_trend = 0.
        if family == 'second_factor':
            lc = SIGNS[:, None] * np.array([0., .14])[None, :]
        elif family == 'opposed_factors':
            assignment = np.array([kappa / 2, kappa / 2])
            ly = np.array([.07, .07])
            lc = SIGNS[:, None] * np.array([-.14, .28])[None, :]
        elif family.startswith('invalid_'):
            gamma[[0] if family == 'invalid_one' else [0, 2]] = .12
        elif family == 'stratum_sign_reversal':
            trend = -.22 * SIGNS
            proxy_trend = 2.
        tag = f'{source}__{family}__scale{scale:g}__rho{rho:g}__tau{tau:g}__n{n}'
        result.append(dict(id=tag, family=family, source=source, scale=scale, rho=rho, tau=tau, n=n,
                           train_n=1000, ly=ly, lc=lc, gamma=gamma, trend=trend,
                           assignment=assignment, proxy_trend=proxy_trend))
    for scale in [.25, .5, 1., 1.5, 2.]:
        for rho in [0., .25, .75]:
            for tau in [0., .08]:
                add('scale_proxy_grid', scale=scale, rho=rho, tau=tau)
    for family in ['second_factor', 'opposed_factors', 'invalid_one', 'invalid_two', 'stratum_sign_reversal']:
        for tau in [0., .08]:
            add(family, tau=tau)
    for scale in [.5, 1., 2.]:
        for rho in [.25, .75]:
            for tau in [0., .08]:
                add('scale_proxy_grid', source='oulad_profile', scale=scale, rho=rho, tau=tau)
    for n in [500, 8000]:
        for tau in [0., .08]:
            add('sample_size', rho=.25, tau=tau, n=n)
    for tau in [0., .08]:
        add('no_confounding', tau=tau, kappa=0.)
    assert len(result) == 58 and len({s['id'] for s in result}) == 58
    return result


def draw(rng, pop, s):
    n = s['n'] + s['train_n']
    rows = pop['rows'][rng.integers(len(pop['rows']), size=n)]
    z = rows[:, 0].astype(int)
    u = rows[:, 1:]
    ez = np.array([.35, .5, .65])
    ea = ez[z] + (u - pop['means'][z]) @ s['assignment']
    if not ((ea > 0).all() and (ea < 1).all()):
        raise ArithmeticError('Invalid assignment probabilities')
    a = (rng.random(n) < ea).astype(float)
    py = .45 + .05 * (z - 1) + s['tau'] * a + u @ s['ly']
    pc = .5 + (z - 1)[:, None] * s['trend'] + a[:, None] * s['gamma'] + u @ s['lc'].T
    prob = np.column_stack((py, pc))
    if not ((prob >= 0).all() and (prob <= 1).all()):
        raise ArithmeticError('Invalid response probabilities; no clipping permitted')
    values = (rng.random(prob.shape) < prob).astype(float)
    g = s['rho'] * (u[:, 0] - pop['mean']) / pop['sd'] + np.sqrt(1 - s['rho']**2) * rng.normal(size=n)
    g += s['proxy_trend'] * (z - 1)
    cut = s['train_n']
    return (z[:cut], values[:cut], g[:cut]), (z[cut:], u[cut:], a[cut:], values[cut:], ea[cut:])


def orientation(z, values, g, residual=False):
    if residual:
        values = values.copy()
        g = g.copy()
        for k in range(3):
            m = z == k
            values[m] -= values[m].mean(axis=0)
            g[m] -= g[m].mean()
    cross = (values - values.mean(axis=0)).T @ (g - g.mean())
    signs = np.where(cross >= 0, 1., -1.)
    return signs[0] * signs[1:]


def standardize(z, a, values, p):
    n, q = values.shape
    d = np.zeros(q)
    influence = np.zeros_like(values)
    for k in range(3):
        for arm in [0, 1]:
            take = (z == k) & (a == arm)
            count = int(take.sum())
            if count < 2:
                raise ArithmeticError('Empty/singleton treatment-by-stratum cell')
            mean = values[take].mean(axis=0)
            factor = (2 * arm - 1) * p[k]
            d += factor * mean
            influence[take] = factor * n / count * (values[take] - mean)
    cov = influence.T @ influence / (n * (n - 1))
    return d, cov


def structural_quantities(pop, s, orient):
    ez = np.array([.35, .5, .65])
    delta = np.einsum('zij,j->zi', pop['cov'], s['assignment']) / (ez * (1 - ez))[:, None]
    db = pop['p'] @ delta
    by = float(db @ s['ly'])
    bc = s['gamma'] + s['lc'] @ db
    diff = s['ly'] - (orient[:, None] * s['lc']).mean(axis=0)
    invalid = float((orient * s['gamma']).mean())
    bound = float((pop['p'] @ np.abs(delta)) @ np.abs(diff) + abs(invalid))
    residual = float(by - (orient * bc).mean())
    if abs(residual) > bound + 1e-12:
        raise ArithmeticError('Structural bound failed')
    return by, bc, residual, bound


def evaluate(pop, s, train, evaluation):
    z, u, a, values, ea = evaluation
    n = len(a)
    d, cov = standardize(z, a, values, pop['p'])
    raw_signs = orientation(*train)
    residual_signs = orientation(*train, residual=True)
    # True-loading signs: clearly an oracle, not estimated from the panel.
    true_signs = np.sign(s['lc'][:, np.flatnonzero(s['ly'])[0]])
    if np.any(true_signs == 0):
        true_signs = SIGNS.copy()  # Second-factor loading orientation, still cannot repair mismatch.
    answers = {}
    def put(name, point, radius, bias_bound=0.):
        answers[name] = (float(point), float(point - radius), float(point + radius), float(bias_bound))
    put('uncorrected_wald', d[0], Z95 * np.sqrt(cov[0, 0]))
    for name, orient in [('unsigned_mom', np.ones(6)), ('original_sign_mom', raw_signs),
                         ('residual_sign_mom', residual_signs), ('oracle_sign_mom', true_signs)]:
        control = orient * d[1:]
        point = d[0] - control.mean()
        h2 = max(0., control.var(ddof=1) - np.diag(cov)[1:].mean())
        put(name, point, Z95 * np.sqrt(cov[0, 0] + h2))
        weights = np.r_[1., -orient / 6]
        if name in ['original_sign_mom', 'residual_sign_mom']:
            radius = Z95 * np.sqrt(max(0., weights @ cov @ weights))
            put(name.replace('_mom', '_wald'), point, radius)
            if name == 'original_sign_mom':
                for allowance in [.025, .05, .10, .20]:
                    put(f'sign_wald_R{allowance:g}', point, radius + allowance, allowance)
    # A standard random-effects summary comparator; not a replication of the full
    # Schuemie positive-control empirical-calibration model or of NC-DiD.
    for name, orient in [('unsigned_random_effects', np.ones(6)), ('signed_random_effects', raw_signs)]:
        estimates = d[1:] * orient
        variances = np.maximum(np.diag(cov)[1:], 1e-12)
        fixed_w = 1 / variances
        fixed_mean = fixed_w @ estimates / fixed_w.sum()
        q = fixed_w @ (estimates - fixed_mean)**2
        h2 = max(0., (q - 5) / (fixed_w.sum() - (fixed_w @ fixed_w) / fixed_w.sum()))
        w = 1 / (variances + h2)
        correction = w @ estimates / w.sum()
        put(name, d[0] - correction, Z95 * np.sqrt(cov[0, 0] + h2))
    point = d[0] - np.median(raw_signs * d[1:])
    answers['signed_median_point_only'] = (float(point), np.nan, np.nan, 0.)
    oracle_score = a * values[:, 0] / ea - (1 - a) * values[:, 0] / (1 - ea)
    put('oracle_full_assignment_ipw', oracle_score.mean(), Z95 * oracle_score.std(ddof=1) / np.sqrt(n))
    by, bc, bias, bound = structural_quantities(pop, s, raw_signs)
    v = values[:, 0] - values[:, 1:] @ raw_signs / 6
    ez = np.array([.35, .5, .65])[z]
    h = a * v / ez - (1 - a) * v / (1 - ez)
    # Conditional on independent training, H is bounded in [-M/eta,M/eta].
    m = 1 + np.mean(raw_signs < 0)
    eta = .35
    sampling_radius = (2 * m / eta) * np.sqrt(np.log(2 / .05) / (2 * n))
    put('known_e_hoeffding_oracle_R', h.mean(), sampling_radius + bound, bound)
    diagnostics = dict(orientation_error=float(np.mean(raw_signs != true_signs)),
                       residual_orientation_error=float(np.mean(residual_signs != true_signs)),
                       target_bias=by, mean_signed_control_bias=float(raw_signs @ bc / 6),
                       true_residual_bias=bias, oracle_structural_bound=bound,
                       oracle_bound_slack=bound-abs(bias), mean_training_sign=float(raw_signs.mean()))
    return answers, diagnostics


def verify(populations_by_name):
    # Independent enumeration confirms the latent selection-bias decomposition.
    checked = 0
    for s in scenarios():
        pop = populations_by_name[s['source']]
        rows = pop['rows']
        z = rows[:, 0].astype(int)
        u = rows[:, 1:]
        e = np.array([.35, .5, .65])
        ea = e[z] + (u - pop['means'][z]) @ s['assignment']
        y0 = .45 + .05 * (z - 1) + u @ s['ly']
        c0 = .5 + (z - 1)[:, None] * s['trend'] + u @ s['lc'].T
        direct = np.zeros(7)
        for k in range(3):
            take = z == k
            direct[0] += pop['p'][k] * (np.mean(ea[take] * (y0[take] + s['tau'])) / e[k]
                            - np.mean((1-ea[take]) * y0[take]) / (1-e[k]))
            direct[1:] += pop['p'][k] * (np.mean(ea[take,None] * (c0[take] + s['gamma']), axis=0) / e[k]
                            - np.mean((1-ea[take,None]) * c0[take], axis=0) / (1-e[k]))
        for signs in [SIGNS, -SIGNS, np.ones(6)]:
            by, bc, residual, bound = structural_quantities(pop, s, signs)
            assert np.max(np.abs(direct - np.r_[s['tau'] + by, bc])) < 1e-10
            assert abs(direct[0] - signs @ direct[1:] / 6 - s['tau'] - residual) < 1e-10
            assert abs(residual) <= bound + 1e-12
            checked += 1
    # Summary-equivalent models: same population inputs to original MoM, distinct ATEs.
    kappa = .15
    summaries = [dict(tau=tau, loading=lam, observed_contrast=tau+4*kappa*lam,
                      mean_oriented_control=4*kappa*.14) for tau, lam in [(0., .14), (.042, .07)]]
    assert abs(summaries[0]['observed_contrast']-summaries[1]['observed_contrast']) < 1e-12
    return dict(structural_enumeration_checks=checked, status='passed', summary_equivalent_models=summaries,
                caveat='Equivalence of retained population contrasts/signs/marginal variances, not equivalence of the full observed joint distribution.')


def write_csv(path, rows):
    with path.open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--replications', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=2026102907)
    parser.add_argument('--verify-only', action='store_true')
    args = parser.parse_args()
    pop, inventory = populations()
    checks = verify(pop)
    OUT.mkdir(exist_ok=True)
    (OUT/'theory_checks.json').write_text(json.dumps(checks, indent=2), encoding='utf-8')
    (OUT/'population_inventory.json').write_text(json.dumps(inventory, indent=2), encoding='utf-8')
    print(json.dumps(checks, indent=2), flush=True)
    if args.verify_only:
        return
    config = dict(seed=args.seed, replications=args.replications, cells=58,
                  protocol_sha256=digest(ROOT/'PROTOCOL.md'), source_sha256=digest(__file__),
                  numpy=np.__version__, pandas=pd.__version__, python=sys.version,
                  scenario_specs=[{k: v.tolist() if isinstance(v, np.ndarray) else v for k,v in s.items()} for s in scenarios()],
                  estimator='Saturated common-stratum standardization; independent training orientation',
                  oracle_labels=['oracle_sign_mom', 'oracle_full_assignment_ipw', 'known_e_hoeffding_oracle_R'])
    (OUT/'config.json').write_text(json.dumps(config, indent=2), encoding='utf-8')
    start = time.time()
    summary, diagnostics_summary = [], []
    fields = ['scenario','replication','method','truth','estimate','lower','upper','allowance']
    failures = []
    with gzip.open(OUT/'replications.csv.gz', 'wt', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for j, s in enumerate(scenarios()):
            rng = np.random.default_rng(args.seed)
            methods, diagnostics = {}, []
            for rep in range(args.replications):
                train, evaluation = draw(rng, pop[s['source']], s)
                try:
                    answer, diagnostic = evaluate(pop[s['source']], s, train, evaluation)
                except ArithmeticError as error:
                    failures.append(dict(scenario=s['id'], replication=rep, error=str(error)))
                    continue
                diagnostics.append(diagnostic)
                for method, (estimate, lo, hi, allowance) in answer.items():
                    methods.setdefault(method, []).append((estimate, lo, hi, allowance))
                    writer.writerow(dict(scenario=s['id'], replication=rep, method=method, truth=s['tau'],
                                         estimate=estimate, lower=lo, upper=hi, allowance=allowance))
            if not methods:
                raise RuntimeError('Every replication failed in '+s['id'])
            base_error = (np.array(methods['uncorrected_wald'])[:,0] - s['tau'])**2
            nrep = len(base_error)
            for method, entries in methods.items():
                a = np.array(entries)
                error = a[:, 0] - s['tau']
                improvement = base_error - error**2
                mcse = improvement.std(ddof=1) / np.sqrt(nrep)
                finite = np.isfinite(a[:, 1:3]).all(axis=1)
                coverage = np.mean((a[:,1]<=s['tau']) & (s['tau']<=a[:,2])) if finite.all() else np.nan
                rejection = np.mean((a[:,1]>0) | (a[:,2]<0)) if finite.all() else np.nan
                summary.append(dict(scenario=s['id'], family=s['family'], source=s['source'], scale=s['scale'], rho=s['rho'],
                                    tau=s['tau'], n=s['n'], replications=nrep, method=method, bias=float(error.mean()),
                                    bias_mcse=float(error.std(ddof=1)/np.sqrt(nrep)), rmse=float(np.sqrt(np.mean(error**2))),
                                    coverage=float(coverage), coverage_mcse=float(np.sqrt(coverage*(1-coverage)/nrep)),
                                    rejection=float(rejection), rejection_mcse=float(np.sqrt(rejection*(1-rejection)/nrep)),
                                    rejection_type='type_I' if s['tau']==0 else 'power',
                                    width=float(np.mean(a[:,2]-a[:,1])), mse_improvement=float(improvement.mean()),
                                    mse_improvement_mcse=float(mcse), mse_improvement_lower95=float(improvement.mean()-Z95*mcse),
                                    mse_improvement_upper95=float(improvement.mean()+Z95*mcse)))
            diag = dict(scenario=s['id'])
            for key in diagnostics[0]:
                data = np.array([x[key] for x in diagnostics])
                diag[key] = float(data.mean())
                diag[key+'_mcse'] = float(data.std(ddof=1)/np.sqrt(nrep))
            diagnostics_summary.append(diag)
            point = next(x for x in summary if x['scenario']==s['id'] and x['method']=='original_sign_mom')
            print(f'{j+1:02}/58 {s["id"]}: RMSE={point["rmse"]:.4f}, coverage={point["coverage"]:.3f}', flush=True)
    write_csv(OUT/'summary.csv', summary)
    write_csv(OUT/'diagnostics.csv', diagnostics_summary)
    (OUT/'failures.json').write_text(json.dumps(failures, indent=2), encoding='utf-8')
    manifest = dict(elapsed_seconds=round(time.time()-start, 2), scenario_count=58, replications_requested=args.replications,
                    failures=len(failures), summary_rows=len(summary), source_and_protocol_hash_match=True,
                    files={p.name:digest(p) for p in OUT.iterdir() if p.is_file() and p.name!='manifest.json'})
    (OUT/'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == '__main__':
    main()
