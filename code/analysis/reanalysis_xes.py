"""XES3G5M re-analysis requested in review (comments 2, 3 and 6).

Input: question_level/train_valid_sequences_quelevel.csv from the XES3G5M release.

Designs (same estimator and panel as the manuscript):
  fractional - manuscript windows: first 30%, next 30%, final 40% of each user's
               observed sequence (window boundaries depend on total history length);
  fixed      - windows fixed in event counts and independent of future length:
               early = events 1..E, middle = E+1..E+M, late = E+M+1..E+M+Lmax;
               eligibility requires at least E+M+Lmin events.
For each design and treatment (A = middle accuracy, B2 = middle session regularity):
  * sample flow with counts after every exclusion, separately for development and holdout;
  * raw and corrected targets with MoM and fixed-sign covariance intervals;
  * LOCO null prediction (development and holdout) for the full four-control panel and
    for the reduced panel that jointly excludes early accuracy and early ability;
  * proxy-sign stability (student-cluster bootstrap) on the development common support.

Usage:
  python reanalysis_xes.py --source .../train_valid_sequences_quelevel.csv --out results_xes
         [--early 30 --middle 30 --late-max 40 --late-min 10] [--fast]
"""
import argparse, hashlib, json, sys
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
import nc_common as nc

SEED = 20260626
CONTROLS = ['early_accuracy', 'early_ability', 'early_activity', 'early_concept_diversity']


def parse(path):
    raw = pd.read_csv(path); groups = {}; dropped = 0
    for row in raw.itertuples(index=False):
        q = np.fromstring(str(row.questions), sep=',', dtype=np.int64)
        r = np.fromstring(str(row.responses), sep=',', dtype=np.int8)
        t = np.fromstring(str(row.timestamps), sep=',', dtype=np.float64)
        tok = str(row.concepts).split(',')
        c = np.array([int(v.split('_')[0]) if v.split('_')[0].lstrip('-').isdigit() else -1 for v in tok], np.int64)
        keep = (r >= 0) & (r <= 1) & (t >= 0) & (q >= 0); dropped += int((~keep).sum())
        groups.setdefault(str(row.uid), []).append(np.column_stack((q[keep], r[keep], t[keep], c[keep])))
    seqs = {}; dup = conf = 0
    for uid, parts in groups.items():
        a = np.concatenate(parts); a = a[np.argsort(a[:, 2], kind='stable')]
        f = pd.DataFrame(a, columns=['q', 'r', 't', 'c'])
        conf += int((f.groupby(['q', 't']).r.nunique() > 1).sum())
        d = f.duplicated(['q', 't']); dup += int(d.sum()); seqs[uid] = a[~d.to_numpy()]
    h = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    return seqs, dict(source=str(path), sha256=h, source_rows=len(raw), users=len(seqs),
                      padding_or_invalid_removed=dropped, duplicate_events_removed=dup, conflicting_event_keys=conf)


def windows(a, mode, args):
    n = len(a)
    if mode == 'fractional':
        i = max(1, int(.3 * n)); j = min(n - 1, max(i + 1, int(.6 * n)))
        return a[:i], a[i:j], a[j:]
    E, M = args.early, args.middle
    return a[:E], a[E:E + M], a[E + M:E + M + args.late_max]


def eligible(a, mode, args):
    return len(a) >= 20 if mode == 'fractional' else len(a) >= args.early + args.middle + args.late_min


def regularity(middle):
    ts = np.unique(middle[:, 2]); mag = np.median(np.abs(ts))
    if mag >= 1e14: ts = ts / 1e6
    elif mag >= 1e11: ts = ts / 1e3
    sessions = ts[np.r_[True, np.diff(ts) > 1800]]; gaps = np.diff(sessions); gaps = gaps[gaps > 0]
    reg = float(1 / (1 + gaps.std(ddof=1) / gaps.mean())) if len(gaps) >= 3 and gaps.mean() > 0 else np.nan
    return reg, len(sessions)


def build_rows(seqs, mode, args):
    elig = np.array(sorted(u for u, a in seqs.items() if eligible(a, mode, args)))
    size = min(args.sample, len(elig))
    sel = np.array(sorted(np.random.default_rng(SEED).choice(elig, size, replace=False)))
    order = np.random.default_rng(SEED).permutation(sel); nh = int(round(.2 * size))
    hold = set(order[:nh]); dev = [u for u in sel if u not in hold]
    sums, cnt = {}, {}
    for u in dev:
        early = windows(seqs[u], mode, args)[0]
        for q in np.unique(early[:, 0]):
            ans = early[early[:, 0] == q, 1]; sums[int(q)] = sums.get(int(q), 0.) + float(ans.sum()); cnt[int(q)] = cnt.get(int(q), 0) + len(ans)
    diff = {q: sums[q] / cnt[q] for q in sums}
    rows = []
    for u in sel:
        early, middle, late = windows(seqs[u], mode, args)
        reg, ns = regularity(middle)
        rows.append(dict(group=u, holdout=u in hold,
                         early_accuracy=float(early[:, 1].mean()),
                         early_ability=float(np.mean(early[:, 1] - np.array([diff.get(int(q), .5) for q in early[:, 0]]))),
                         early_activity=len(early) if mode == 'fractional' else float(np.ptp(early[:, 2])),
                         early_concept_diversity=len(np.unique(early[:, 3])),
                         early_wrong_concepts=len(np.unique(early[early[:, 1] == 0, 3])),
                         middle_accuracy=float(middle[:, 1].mean()), late_accuracy=float(late[:, 1].mean()),
                         middle_session_regularity=reg, middle_sessions=ns))
    flow = dict(eligible=int(len(elig)), sampled=int(size), development=len(dev), holdout=len(hold))
    return pd.DataFrame(rows), flow


def analyse(rows, design, field, args):
    use = rows if design == 'A' else rows[(rows.middle_sessions >= 4) & rows.middle_session_regularity.notna()]
    dev, hold = use[~use.holdout], use[use.holdout]
    th = {k: float(dev[k].median()) for k in ['late_accuracy', field, *CONTROLS]}
    vals = lambda d: np.column_stack([(d.late_accuracy >= th['late_accuracy'])] + [(d[f] >= th[f]) for f in CONTROLS]).astype(int)
    V, Vh = vals(dev), vals(hold)
    T = (dev[field] >= th[field]).to_numpy().astype(int); Th = (hold[field] >= th[field]).to_numpy().astype(int)
    X = dev[['early_wrong_concepts']].to_numpy(np.float32); Xh = hold[['early_wrong_concepts']].to_numpy(np.float32)
    G, Gh = dev.group.to_numpy(), hold.group.to_numpy()
    e, m0, m1, h = nc.crossfit(X, T, V, G, SEED, args.fast, Xh)
    est, cov, keep = nc.aipw(T, V, e, m0, m1, G)
    he, hc, hkeep = nc.aipw(Th, Vh, *h, Gh)
    g = dev.early_ability.to_numpy()[keep]
    rho = np.array([nc.corr(V[keep][:, j], g) for j in range(V.shape[1])])
    names = ['late_accuracy'] + CONTROLS
    out = dict(flow=dict(development_after_design_filter=len(dev), holdout_after_design_filter=len(hold),
                         development_common_support=int(keep.sum()), holdout_common_support=int(hkeep.sum())),
               treated_share=float(T.mean()))
    for label, ctrl in (('full_panel', [1, 2, 3, 4]), ('without_accuracy_and_ability', [3, 4])):
        lo = nc.loco(est, cov, rho, ctrl, (he, hc)); lo['rows'] = [dict(r, control=names[r['control']]) for r in lo['rows']]
        out[label] = dict(target=nc.calibrate(est, cov, rho, 0, ctrl), loco=lo)
    out['sign_stability_development'] = nc.sign_bootstrap(V[keep], g, X[keep], G[keep], names, est, cov, [0], [1, 2, 3, 4],
                                                           reps=args.sign_boot)
    return out


def md(res):
    f = lambda c: f'[{c[0]:+.3f}, {c[1]:+.3f}]'
    L = ['# XES3G5M re-analysis', '', '## Source and cleaning', ''] + [f'- {k}: {v}' for k, v in res['provenance'].items()]
    for mode, r in res['modes'].items():
        L += ['', f'## Windows: {mode}', ''] + [f'- {k}: {v}' for k, v in r['flow'].items()]
        for d, x in r['designs'].items():
            L += ['', f'### Design {d}', ''] + [f'- {k}: {v}' for k, v in x['flow'].items()]
            L += ['', '| Panel | Raw | Corrected | MoM interval | Cov. interval | LOCO dev raw→corr | LOCO holdout raw→corr | Excl. 0 (dev/hold) |',
                  '|---|---|---|---|---|---|---|---|']
            for p in ('full_panel', 'without_accuracy_and_ability'):
                t, lo = x[p]['target'], x[p]['loco']
                L.append(f'| {p} | {t["raw"]:+.3f} | {t["corrected"]:+.3f} | {f(t["mom_ci"])} | {f(t["cov_ci"])} | '
                         f'{lo["raw_rmse"]:.3f}→{lo["corrected_rmse"]:.3f} | {lo["holdout_raw_rmse"]:.3f}→{lo["holdout_corrected_rmse"]:.3f} | '
                         f'{lo["excluded"]}/{lo["k"]}, {lo["holdout_excluded"]}/{lo["k"]} |')
            b = x['sign_stability_development']['marginal']
            L.append('\nProxy-sign retention (development common support): ' +
                     ', '.join(f'{k} {v["correlation"]:+.3f} ({v["sign_retention"]:.3f})' for k, v in b.items()))
    return '\n'.join(L) + '\n'


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source', required=True); p.add_argument('--out', default='results_xes')
    p.add_argument('--early', type=int, default=30); p.add_argument('--middle', type=int, default=30)
    p.add_argument('--late-max', type=int, default=40); p.add_argument('--late-min', type=int, default=10)
    p.add_argument('--sample', type=int, default=6000); p.add_argument('--sign-boot', type=int, default=2000)
    p.add_argument('--fast', action='store_true')
    args = p.parse_args()
    seqs, prov = parse(args.source)
    res = dict(settings=vars(args), provenance=prov, modes={})
    for mode in ('fractional', 'fixed'):
        rows, flow = build_rows(seqs, mode, args)
        res['modes'][mode] = dict(flow=flow, designs={d: analyse(rows, d, fld, args) for d, fld in
                                                      (('A', 'middle_accuracy'), ('B2', 'middle_session_regularity'))})
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    (out / 'xes_reanalysis.json').write_text(json.dumps(res, indent=2, default=float), encoding='utf-8')
    (out / 'xes_reanalysis.md').write_text(md(res), encoding='utf-8'); print(md(res))


if __name__ == '__main__':
    main()
