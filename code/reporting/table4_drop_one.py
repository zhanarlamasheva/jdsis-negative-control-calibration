"""Reproduce Table 4 (drop-one OULAD estimates) and the LOCO intervals quoted in Section 4.2
from results/real_data/oulad_reanalysis.json, using the same MoM rule as nc_common.calibrate.

Usage: python code/reporting/table4_drop_one.py [results/real_data/oulad_reanalysis.json]
"""
import json, sys
import numpy as np

Z = 1.959963984540054
NAMES = ['not_withdrawn', 'successful_completion', 'disability', 'male', 'age_35_plus',
         'higher_education', 'imd_50_plus', 'previous_attempts']
CTRL = list(range(2, 8))


def sgn(x):
    s = np.sign(x); return np.where(s == 0, 1, s)


def pooled(est, se, rho, ref, ctrl):
    ctrl = np.array(ctrl); t = est[ctrl] * sgn(rho[ctrl]) * sgn(rho[ref])
    h2 = max(0., t.var(ddof=1) - np.mean(se[ctrl] ** 2)) if len(ctrl) > 1 else 0.
    return t.mean(), np.sqrt(h2)


def main(path):
    r = json.load(open(path, encoding='utf-8'))['designs']['reference']
    est = np.array([r['estimates'][n]['estimate'] for n in NAMES])
    se = np.array([r['estimates'][n]['se'] for n in NAMES])
    rho = np.array([r['estimates'][n]['proxy_correlation'] for n in NAMES])
    print('Table 4: non-withdrawal after dropping each control')
    print(f'{"control":20s} {"contrast":>9s} {"proxy r":>8s} {"sign":>5s} {"adjusted":>9s}  nominal interval')
    for j in CTRL:
        mu, h = pooled(est, se, rho, 0, [k for k in CTRL if k != j])
        c = est[0] - mu; w = np.sqrt(se[0] ** 2 + h ** 2)
        print(f'{NAMES[j]:20s} {est[j]:+9.3f} {rho[j]:+8.3f} {int(sgn(rho[j])):+5d} {c:+9.3f}  [{c - Z * w:+.3f}, {c + Z * w:+.3f}]')
    print('\nLeave-one-control-out pseudo-targets (Section 4.2)')
    for j in CTRL:
        mu, h = pooled(est, se, rho, j, [k for k in CTRL if k != j])
        p = est[j] - mu; w = np.sqrt(se[j] ** 2 + h ** 2)
        print(f'{NAMES[j]:20s} raw {est[j]:+.3f} corrected {p:+.3f} [{p - Z * w:+.3f}, {p + Z * w:+.3f}]')


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else 'results/real_data/oulad_reanalysis.json')
