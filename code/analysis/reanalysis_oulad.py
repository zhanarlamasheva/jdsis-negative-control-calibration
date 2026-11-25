"""OULAD re-analysis requested in review (comments 1, 4 and 5).

Runs directly on the raw OULAD CSV files (studentInfo, studentRegistration,
assessments, studentAssessment, studentVle) and produces:

  1. A coding audit of the exposure under the manuscript definition: on-time,
     late, missing and banked first-assessment submissions, withdrawals before the
     deadline, late registrations, and early submitters with clicks between their
     submission and the deadline.
  2. Three designs with the same estimator and panel:
       reference  - manuscript timing (clicks before the deadline, all registrations);
       landmark   - time zero at L = deadline - k days: only registrations that are
                    enrolled, not withdrawn and have not submitted by L; features from
                    clicks before L; exposure = on-time submission in (L, deadline];
                    withdrawal counted only after L; banked results excluded;
       landmark_strict - landmark, additionally excluding withdrawals in (L, deadline]
                    (conditions on post-landmark information; sensitivity only).
  3. For every design: raw, MoM and fixed-sign covariance intervals; LOCO null
     prediction; drop-one estimates; proxy-sign stability on the analysis sample for
     targets and controls (marginal and residualised correlations); orientation-change
     frequency; and the estimate using only controls above a prespecified retention
     threshold. Optionally a full-pipeline cluster bootstrap (--pipeline-boot N).

Usage:
  python reanalysis_oulad.py --data /path/to/OULAD --out results_oulad [--landmark-days 7]
         [--threshold 0.95] [--pipeline-boot 200] [--fast]
"""
import argparse, json, sys
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
import nc_common as nc

KEY = ['code_module', 'code_presentation', 'id_student']
TARGETS = ['not_withdrawn', 'successful_completion']
CONTROLS = ['disability', 'male', 'age_35_plus', 'higher_education', 'imd_50_plus', 'previous_attempts']
MAX_DAYS = 61
SEED = 20260626


def load(d):
    d = Path(d)
    info = pd.read_csv(d / 'studentInfo.csv'); reg = pd.read_csv(d / 'studentRegistration.csv')
    ass = pd.read_csv(d / 'assessments.csv'); sa = pd.read_csv(d / 'studentAssessment.csv')
    vle = pd.read_csv(d / 'studentVle.csv', usecols=KEY + ['date', 'sum_click'])
    for f in (ass, sa, reg):
        for c in ('date', 'date_submitted', 'date_registration', 'date_unregistration'):
            if c in f: f[c] = pd.to_numeric(f[c], errors='coerce')
    return info, reg, ass, sa, vle


def first_assessment(ass):
    a = ass.dropna(subset=['date']).sort_values(['code_module', 'code_presentation', 'date', 'id_assessment'])
    return a.drop_duplicates(['code_module', 'code_presentation'])[['code_module', 'code_presentation', 'id_assessment', 'date']] \
        .rename(columns={'date': 'deadline'})


def frame(info, reg, ass, sa, presentations):
    df = info[info.code_presentation.isin(presentations)].merge(reg, on=KEY, how='left', validate='one_to_one')
    df = df.merge(first_assessment(ass), on=['code_module', 'code_presentation'], how='inner')
    sub = sa[['id_assessment', 'id_student', 'date_submitted', 'is_banked']]
    df = df.merge(sub, on=['id_assessment', 'id_student'], how='left', validate='one_to_one')
    df['submitted'] = df.date_submitted.notna()
    df['banked'] = df.is_banked.fillna(0).astype(int) == 1
    df['on_time'] = df.submitted & (df.date_submitted <= df.deadline)
    return df.reset_index(drop=True)


def trend_features(daily, start, end):
    """Seven click-trajectory features over integer days [start, end) (as in pred_oulad)."""
    L = max(1, min(int(np.ceil(end - start)), MAX_DAYS)); seq = daily[:L]
    active = seq > 0; tot = float(seq.sum()); na = int(active.sum())
    weeks = max(1, int(np.ceil(L / 7))); weekly = np.array([seq[w * 7:(w + 1) * 7].sum() for w in range(weeks)], float)
    slope = float(np.polyfit(np.arange(len(weekly)), weekly, 1)[0]) if len(weekly) >= 2 else 0.
    gap = best = 0
    for v in active:
        gap = 0 if v else gap + 1; best = max(best, gap)
    last = np.flatnonzero(active); since = float(L - 1 - last[-1]) if len(last) else float(L)
    return [tot, na, tot / na if na else 0., float(seq[max(0, L - 14):L].sum()), slope, float(best), since]


def features(df, vle, cut_col):
    v = vle.merge(df[KEY + [cut_col]], on=KEY, how='inner')
    v = v[(v.date >= 0) & (v.date < v[cut_col])]
    agg = v.groupby(KEY + ['date']).sum_click.sum().reset_index()
    pos = {k: i for i, k in enumerate(map(tuple, df[KEY].to_numpy()))}
    daily = np.zeros((len(df), MAX_DAYS), np.float32)
    for row in agg.itertuples(index=False):
        if row.date < MAX_DAYS: daily[pos[(row.code_module, row.code_presentation, row.id_student)], int(row.date)] += row.sum_click
    F = np.array([trend_features(daily[i], 0, df[cut_col].iat[i]) for i in range(len(df))], np.float32)
    return F


def outcomes(df):
    imd = pd.to_numeric(df.imd_band.astype(str).str.extract(r'^(\d+)', expand=False), errors='coerce')
    V = np.column_stack(((df.final_result != 'Withdrawn'), df.final_result.isin(['Pass', 'Distinction']),
                         df.disability == 'Y', df.gender == 'M', df.age_band != '0-35',
                         df.highest_education.isin(['HE Qualification', 'Post Graduate Qualification']),
                         imd >= 50, df.num_of_prev_attempts > 0)).astype(int)
    return V, int(imd.isna().sum())


def covariates(df, F):
    reg = pd.get_dummies(df.region.astype(str), prefix='r').to_numpy(float)
    return np.column_stack((reg, df.studied_credits.to_numpy(float), df.date_registration.fillna(0).to_numpy(float), F)).astype(np.float32)


def coding_audit(df, vle):
    d = df
    early = d[d.submitted & ~d.banked & (d.date_submitted < d.deadline)]
    v = vle.merge(early[KEY + ['date_submitted', 'deadline']], on=KEY, how='inner')
    v = v[(v.date >= 0) & (v.date < v.deadline)]
    after = v[v.date > v.date_submitted].groupby(KEY).sum_click.sum()
    total = v.groupby(KEY).sum_click.sum()
    share = (after.reindex(total.index).fillna(0) / total.replace(0, np.nan)).dropna()
    nos = ~d.submitted
    return dict(
        registrations=len(d),
        on_time_non_banked=int((d.on_time & ~d.banked).sum()),
        on_time_banked=int((d.on_time & d.banked).sum()),
        late_submission=int((d.submitted & ~d.on_time).sum()),
        no_submission=int(nos.sum()),
        no_submission_withdrew_before_deadline=int((nos & (d.date_unregistration <= d.deadline)).sum()),
        withdrew_before_deadline_any=int((d.date_unregistration <= d.deadline).sum()),
        withdrew_before_deadline_coded_T0=int(((d.date_unregistration <= d.deadline) & ~d.on_time).sum()),
        registered_after_deadline=int((d.date_registration > d.deadline).sum()),
        banked_any=int(d.banked.sum()),
        early_submitters=int(len(early)),
        early_submitters_with_clicks_after_submission=int((after > 0).sum()),
        median_share_of_window_clicks_after_submission=float(share.median()) if len(share) else None,
        coding_in_manuscript='T=1 if the first assessment was submitted on or before its deadline (banked included); '
                             'late, missing and withdrawn-before-deadline registrations coded T=0')


def design_frames(df, k):
    ref = df.copy(); ref['cut'] = ref.deadline; ref['T'] = ref.on_time.astype(int)
    L = df.deadline - k
    elig = (df.date_registration <= L) & ~(df.date_unregistration <= L) & ~df.banked & \
           ~(df.submitted & (df.date_submitted <= L))
    lm = df[elig].copy(); lm['cut'] = lm.deadline - k
    lm['T'] = (lm.submitted & (lm.date_submitted > lm.cut) & (lm.date_submitted <= lm.deadline)).astype(int)
    strict = lm[~((lm.date_unregistration > lm.cut) & (lm.date_unregistration <= lm.deadline))].copy()
    flow = dict(all=len(df), registered_by_L=int((df.date_registration <= L).sum()),
                not_withdrawn_by_L=int(((df.date_registration <= L) & ~(df.date_unregistration <= L)).sum()),
                not_banked=int(((df.date_registration <= L) & ~(df.date_unregistration <= L) & ~df.banked).sum()),
                not_submitted_by_L=int(elig.sum()), strict_no_withdrawal_before_deadline=len(strict))
    return dict(reference=ref.reset_index(drop=True), landmark=lm.reset_index(drop=True),
                landmark_strict=strict.reset_index(drop=True)), flow


def analyse(d, vle, args, label):
    F = features(d, vle, 'cut'); X = covariates(d, F); V, imd_missing = outcomes(d)
    T = d['T'].to_numpy(); groups = d.id_student.to_numpy(); g = F[:, 0].astype(float)
    names = TARGETS + CONTROLS; ctrl = list(range(2, 8))
    print(f'{label}: {len(d)} registrations, treated share {T.mean():.3f}', flush=True)
    e, m0, m1, _ = nc.crossfit(X, T, V, groups, SEED, args.fast)
    est, cov, keep = nc.aipw(T, V, e, m0, m1, groups)
    Xbase = X[:, :X.shape[1] - F.shape[1]]          # region, credits, registration date (no click features)
    Vk, gk, Xk, grk = V[keep], g[keep], Xbase[keep], groups[keep]
    rho = np.array([nc.corr(Vk[:, j], gk) for j in range(V.shape[1])])
    se = np.sqrt(np.diag(cov))
    out = dict(n=len(d), n_common_support=int(keep.sum()), n_students=int(pd.Series(grk).nunique()),
               treated_share=float(T.mean()), imd_missing=imd_missing,
               estimates={names[j]: dict(estimate=float(est[j]), se=float(se[j]), proxy_correlation=float(rho[j])) for j in range(8)})
    out['targets'] = {names[t]: nc.calibrate(est, cov, rho, t, ctrl) for t in (0, 1)}
    out['drop_one'] = {names[t]: {names[c]: nc.calibrate(est, cov, rho, t, [x for x in ctrl if x != c])['corrected'] for c in ctrl} for t in (0, 1)}
    lo = nc.loco(est, cov, rho, ctrl); lo['rows'] = [dict(r, control=names[r['control']]) for r in lo['rows']]
    out['loco'] = lo
    boot = nc.sign_bootstrap(Vk, gk, Xk, grk, names, est, cov, [0, 1], ctrl, reps=args.sign_boot)
    out['sign_stability_common_support'] = boot
    stable = nc.stable_panel(boot, names, ctrl, args.threshold)
    out['stable_panel'] = dict(threshold=args.threshold, controls=[names[c] for c in stable],
                               targets={names[t]: nc.calibrate(est, cov, rho, t, stable) if len(stable) >= 2 else None for t in (0, 1)})
    if args.pipeline_boot:
        def build(idx, Gb):
            Xb, Tb, Vb = X[idx], T[idx], V[idx]   # resampled students carry new cluster labels
            eb, a0, a1, _ = nc.crossfit(Xb, Tb, Vb, Gb, SEED, True)
            eb_, cb, kb = nc.aipw(Tb, Vb, eb, a0, a1, Gb)
            rb = np.array([nc.corr(Vb[kb][:, j], F[idx][kb][:, 0]) for j in range(Vb.shape[1])])
            return eb_, cb, rb
        out['pipeline_bootstrap'] = nc.full_pipeline_bootstrap(build, groups, args.pipeline_boot, SEED, [0, 1], ctrl,
                                                               {t: out['targets'][names[t]]['orientation'] for t in (0, 1)})
    return out


def fmt_ci(c): return f'[{c[0]:+.3f}, {c[1]:+.3f}]'


def markdown(res):
    L = ['# OULAD re-analysis', '', '## Exposure coding audit (manuscript definition)', '']
    L += [f'- {k}: {v}' for k, v in res['coding_audit'].items()]
    L += ['', '## Landmark sample flow', ''] + [f'- {k}: {v}' for k, v in res['landmark_flow'].items()]
    for name, r in res['designs'].items():
        L += ['', f'## Design: {name} (n = {r["n"]}, common support = {r["n_common_support"]})', '',
              '| Target | Raw [95% CI] | Corrected | MoM interval | Covariance interval (fixed signs) |', '|---|---|---|---|---|']
        for t, c in r['targets'].items():
            L.append(f'| {t} | {c["raw"]:+.3f} {fmt_ci(c["raw_ci"])} | {c["corrected"]:+.3f} | {fmt_ci(c["mom_ci"])} | {fmt_ci(c["cov_ci"])} |')
        lo = r['loco']
        L += ['', f'LOCO null prediction: raw RMSE {lo["raw_rmse"]:.3f}, corrected RMSE {lo["corrected_rmse"]:.3f}, '
                  f'intervals excluding zero {lo["excluded"]}/{lo["k"]}', '',
              '| Variable | Marginal r | Retention | Residualised r | Retention |', '|---|---|---|---|---|']
        b = r['sign_stability_common_support']
        for v in TARGETS + CONTROLS:
            m, s = b['marginal'][v], b['residualised'][v]
            L.append(f'| {v} | {m["correlation"]:+.3f} | {m["sign_retention"]:.3f} | {s["correlation"]:+.3f} | {s["sign_retention"]:.3f} |')
        for t, oc in b['orientation_changes'].items():
            L.append(f'\nOrientation changes for {t}: any control flips in {oc["any_control_flips"]:.3f} of draws; '
                     f'signed estimate (sign uncertainty only) 95% range {fmt_ci(oc["signed_estimate_sign_only_ci95"])}')
        sp = r['stable_panel']
        L.append(f'\nStable panel (retention >= {sp["threshold"]}): {", ".join(sp["controls"]) or "none"}')
        for t, c in sp['targets'].items():
            if c: L.append(f'- {t}: corrected {c["corrected"]:+.3f}, MoM {fmt_ci(c["mom_ci"])}, covariance {fmt_ci(c["cov_ci"])}')
        if 'pipeline_bootstrap' in r:
            for t, pb in r['pipeline_bootstrap'].items():
                L.append(f'- Full-pipeline bootstrap, {TARGETS[int(t)]}: raw {fmt_ci(pb["raw_ci95"])}, corrected {fmt_ci(pb["corrected_ci95"])}, '
                         f'orientation differs from point estimate in {pb["orientation_differs_from_point_estimate"]:.3f} of draws')
    return '\n'.join(L) + '\n'


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', required=True); p.add_argument('--out', default='results_oulad')
    p.add_argument('--presentations', nargs='+', default=['2013B', '2013J'])
    p.add_argument('--landmark-days', type=int, default=7); p.add_argument('--threshold', type=float, default=.95)
    p.add_argument('--sign-boot', type=int, default=2000); p.add_argument('--pipeline-boot', type=int, default=0)
    p.add_argument('--fast', action='store_true', help='100 trees instead of 300 (testing only)')
    args = p.parse_args()
    info, reg, ass, sa, vle = load(args.data)
    df = frame(info, reg, ass, sa, args.presentations)
    designs, flow = design_frames(df, args.landmark_days)
    res = dict(settings=vars(args), coding_audit=coding_audit(df, vle), landmark_flow=flow,
               designs={k: analyse(d, vle, args, k) for k, d in designs.items()})
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    (out / 'oulad_reanalysis.json').write_text(json.dumps(res, indent=2, default=float), encoding='utf-8')
    (out / 'oulad_reanalysis.md').write_text(markdown(res), encoding='utf-8')
    print(markdown(res))


if __name__ == '__main__':
    main()
