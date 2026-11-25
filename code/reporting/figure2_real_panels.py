"""Panels C and D of Figure 2 from the saved re-analysis outputs.

C: OULAD marginal proxy-sign correlations with 95% cluster-bootstrap intervals
   (reference design, common support).
D: leave-one-control-out null-reference RMSE, raw and corrected, for OULAD and
   XES3G5M (fractional windows, full panels).

Usage:
  python code/reporting/figure2_real_panels.py results/real_data out.png [top_panels.png]
If top_panels.png (panels A and B) is given, the two images are stacked into the full figure.
"""
import json, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

LABELS = {'disability': 'Declared disability', 'male': 'Male indicator', 'age_35_plus': 'Age 35+',
          'higher_education': 'Higher education', 'imd_50_plus': 'IMD 50+', 'previous_attempts': 'Previous attempts'}


def main(results, out, top=None):
    o = json.load(open(Path(results) / 'oulad_reanalysis.json', encoding='utf-8'))['designs']['reference']
    x = json.load(open(Path(results) / 'xes_reanalysis.json', encoding='utf-8'))['modes']['fractional']['designs']
    marg = o['sign_stability_common_support']['marginal']
    bars = [('OULAD\ndev', o['loco']['raw_rmse'], o['loco']['corrected_rmse'])]
    for d, name in (('A', 'XES A'), ('B2', 'XES B2')):
        l = x[d]['full_panel']['loco']
        bars += [(f'{name}\ndev', l['raw_rmse'], l['corrected_rmse']),
                 (f'{name}\nholdout', l['holdout_raw_rmse'], l['holdout_corrected_rmse'])]

    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
    keys = list(LABELS); yy = np.arange(len(keys))
    pt = np.array([marg[k]['correlation'] for k in keys]); ci = np.array([marg[k]['ci95'] for k in keys])
    ax[0].errorbar(pt, yy, xerr=np.vstack((pt - ci[:, 0], ci[:, 1] - pt)), fmt='o', color='#24648d', capsize=3)
    ax[0].axvline(0, color='#777', lw=1)
    ax[0].set(yticks=yy, yticklabels=[LABELS[k] for k in keys], xlabel='Marginal proxy correlation',
              title='C. OULAD sign stability (common support)')
    ax[0].invert_yaxis()
    xx = np.arange(len(bars)); w = .36
    ax[1].bar(xx - w / 2, [b[1] for b in bars], w, color='#8795a1', label='Raw')
    ax[1].bar(xx + w / 2, [b[2] for b in bars], w, color='#24648d', label='LOCO corrected')
    ax[1].set(xticks=xx, xticklabels=[b[0] for b in bars], ylabel='Null-reference RMSE', title='D. Real-panel null prediction')
    ax[1].legend(frameon=False)
    fig.tight_layout()
    if top is None:
        fig.savefig(out, dpi=220); return
    tmp = Path(out).with_suffix('.bottom.png'); fig.savefig(tmp, dpi=220); plt.close(fig)
    from PIL import Image
    a, b = Image.open(top).convert('RGB'), Image.open(tmp).convert('RGB')
    if b.width != a.width:
        b = b.resize((a.width, round(b.height * a.width / b.width)))
    full = Image.new('RGB', (a.width, a.height + b.height), 'white'); full.paste(a, (0, 0)); full.paste(b, (0, a.height))
    full.save(out); tmp.unlink()


if __name__ == '__main__':
    main(*sys.argv[1:])
