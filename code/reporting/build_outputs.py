"""Build article-ready tables and figures from all prospectively specified cells."""
from pathlib import Path
import sys
sys.dont_write_bytecode = True
import os
import json
import hashlib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'paper_supplement'
OUT.mkdir(exist_ok=True)
os.environ['MPLCONFIGDIR'] = str(ROOT / '.mplconfig')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

METHODS = ['uncorrected_wald','original_sign_mom','residual_sign_mom']
COLORS = ['#666666','#2878A8','#D17A22']
LABELS = ['Uncorrected','Original orientation','Residual orientation']


def get(frame, family='scale_proxy_grid', source='synthetic', scale=1., rho=.75, tau=0., n=2000, method='original_sign_mom'):
    take = frame[(frame.family==family)&(frame.source==source)&(frame.scale==scale)&(frame.rho==rho)&
                 (frame.tau==tau)&(frame.n==n)&(frame.method==method)]
    assert len(take)==1, (family,source,scale,rho,tau,n,method)
    return take.iloc[0]


def figures(frame):
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig, axes = plt.subplots(2,2,figsize=(10.5,7.2),constrained_layout=True)
    for row, source in enumerate(['synthetic','oulad_profile']):
        scales = [.25,.5,1.,1.5,2.] if row==0 else [.5,1.,2.]
        rhos = [0.,.25,.75] if row==0 else [.25,.75]
        ratio = np.array([[get(frame,source=source,scale=s,rho=r).rmse /
                           get(frame,source=source,scale=s,rho=r,method='uncorrected_wald').rmse for s in scales] for r in rhos])
        coverage = np.array([[get(frame,source=source,scale=s,rho=r).coverage for s in scales] for r in rhos])
        for col, (data,cmap,vmin,vmax,title) in enumerate([(ratio,'RdYlBu_r',.2,1.5,'RMSE ratio: original / uncorrected'),
                                                        (coverage,'YlGnBu',0,1,'Coverage: original MoM interval')]):
            ax=axes[row,col]
            im=ax.imshow(data,cmap=cmap,vmin=vmin,vmax=vmax,aspect='auto')
            ax.set_xticks(range(len(scales)),[f'{x:g}' for x in scales])
            ax.set_yticks(range(len(rhos)),[f'{x:g}' for x in rhos])
            ax.set_xlabel('Control / target loading ratio')
            ax.set_ylabel('Proxy coefficient rho')
            ax.set_title(('Synthetic' if row==0 else 'OULAD profile: generated outcomes')+'\n'+title,fontsize=10)
            for i in range(len(rhos)):
                for j in range(len(scales)):
                    ax.text(j,i,f'{data[i,j]:.2f}',ha='center',va='center',color='white' if (col==1 and data[i,j]>.6) or (col==0 and data[i,j]>1.3) else 'black')
            fig.colorbar(im,ax=ax,shrink=.8)
    fig.suptitle('Null effect, n=2000; 1000 independent training/evaluation replications per cell',fontsize=11)
    fig.savefig(OUT/'grid.png',dpi=220)
    plt.close(fig)

    families=['scale_proxy_grid','second_factor','opposed_factors','invalid_two','stratum_sign_reversal','no_confounding']
    names=['Matched','Second factor','Opposed factors','Two invalid','Stratum reversal','No confounding']
    fig,axes=plt.subplots(2,1,figsize=(10.5,7),constrained_layout=True)
    x=np.arange(len(families))
    for j,(method,color,label) in enumerate(zip(METHODS,COLORS,LABELS)):
        values=[get(frame,family=f,method=method) for f in families]
        axes[0].bar(x+(j-1)*.24,[v.rmse for v in values],width=.23,color=color,label=label)
        axes[1].bar(x+(j-1)*.24,[v.coverage for v in values],width=.23,color=color)
    axes[0].set_ylabel('RMSE (risk difference)')
    axes[0].legend(ncol=3,loc='upper left',frameon=False)
    axes[0].set_ylim(0,.215)
    axes[1].set_ylabel('Coverage')
    axes[1].axhline(.95,color='black',linestyle='--',lw=1)
    axes[1].set_ylim(0,1.04)
    for ax in axes:
        ax.set_xticks(x,names)
    fig.suptitle('Distinct failures require distinct explanations: null effect, rho=.75, n=2000',fontsize=11)
    fig.savefig(OUT/'failures.png',dpi=220)
    plt.close(fig)

    fig,axes=plt.subplots(1,2,figsize=(10.5,4.2),constrained_layout=True)
    allowances=[0.,.025,.05,.10,.20]
    for family,label,color in zip(['scale_proxy_grid','second_factor','stratum_sign_reversal'],
                                   ['Matched','Second factor','Stratum reversal'],['#2878A8','#68933B','#B34F57']):
        rows=[get(frame,family=family,method='original_sign_wald' if r==0 else f'sign_wald_R{r:g}') for r in allowances]
        axes[0].plot(allowances,[v.coverage for v in rows],marker='o',label=label,color=color)
        axes[1].plot(allowances,[v.width for v in rows],marker='o',color=color)
    axes[0].axhline(.95,color='black',linestyle='--',lw=1)
    axes[0].set_ylabel('Coverage')
    axes[0].legend(frameon=False,fontsize=9,loc='lower right')
    axes[1].set_ylabel('Mean interval width')
    for ax in axes:
        ax.set_xlabel('Prespecified residual-bias allowance R')
    fig.suptitle('Sensitivity displays: more width can restore coverage, not identify the correct allowance',fontsize=11)
    fig.savefig(OUT/'sensitivity.png',dpi=220)
    plt.close(fig)


def tex_rows(frame):
    labels={'scale_proxy_grid':'Scale/proxy','second_factor':'Second factor','opposed_factors':'Opposed factors',
            'invalid_one':'One invalid','invalid_two':'Two invalid','stratum_sign_reversal':'Stratum reversal',
            'sample_size':'Sample size','no_confounding':'No confounding'}
    all_rows=[]
    for source in ['synthetic','oulad_profile']:
        for scenario in frame[frame.source==source].scenario.drop_duplicates():
            cell=frame[frame.scenario==scenario]
            b=cell[cell.method=='uncorrected_wald'].iloc[0]
            s=cell[cell.method=='original_sign_mom'].iloc[0]
            r=cell[cell.method=='residual_sign_mom'].iloc[0]
            all_rows.append(f'{"S" if source=="synthetic" else "O"} & {labels[s.family]} & {s.scale:g} & {s.rho:g} & {s.tau:g} & {int(s.n)} & {b.rmse:.3f} & {s.rmse:.3f} & {s.coverage:.3f} & {r.rmse:.3f} \\\\')
    (OUT/'all_cells.tex').write_text('\n'.join(all_rows)+'\n',encoding='utf-8')
    selected=[]
    for family in ['scale_proxy_grid','second_factor','opposed_factors','invalid_one','invalid_two','stratum_sign_reversal','no_confounding']:
        b=get(frame,family=family,method='uncorrected_wald')
        s=get(frame,family=family)
        r=get(frame,family=family,method='residual_sign_mom')
        selected.append(f'{labels[family] if family!="scale_proxy_grid" else "Matched controls"} & {b.rmse:.4f} & {s.rmse:.4f} & {s.coverage:.3f} & {r.rmse:.4f} & {r.coverage:.3f} \\\\')
    (OUT/'selected_cells.tex').write_text('\n'.join(selected)+'\n',encoding='utf-8')
    selected_table = r'''\begin{tabular}{lrrrrr}\hline
 & Raw RMSE & Original RMSE & Cover. & Residual RMSE & Cover.\\\hline
''' + '\n'.join(selected) + r'''
\hline\end{tabular}
'''
    (OUT/'selected_table.tex').write_text(selected_table,encoding='utf-8')
    all_table = r'''\begin{longtable}{lp{2.5cm}rrrrrrrr}
\caption{Every prespecified cell. S=synthetic, O=OULAD-profile generated outcomes; $r$=control loading scale, $\rho$=proxy coefficient, $\tau$=true ATE. U=uncorrected RMSE; Sgn=original oriented RMSE; C=original MoM coverage; Res=residual-orientation RMSE. Each row contains 1000 replications.}\label{tab:all}\\
\hline Source & Family & $r$ & $\rho$ & $\tau$ & $n$ & U & Sgn & C & Res\\\hline
\endfirsthead
\hline Source & Family & $r$ & $\rho$ & $\tau$ & $n$ & U & Sgn & C & Res\\\hline
\endhead
\hline\endfoot
''' + '\n'.join(all_rows) + r'''
\end{longtable}
'''
    (OUT/'all_table.tex').write_text(all_table,encoding='utf-8')
    matched=get(frame)
    base=get(frame,method='uncorrected_wald')
    sem=get(frame,source='oulad_profile')
    sem_base=get(frame,source='oulad_profile',method='uncorrected_wald')
    reverse=get(frame,family='stratum_sign_reversal')
    reverse_r=get(frame,family='stratum_sign_reversal',method='residual_sign_mom')
    scale2=get(frame,scale=2.)
    macros=dict(MatchedBaseRMSE=f'{base.rmse:.4f}',MatchedRMSE=f'{matched.rmse:.4f}',MatchedCoverage=f'{100*matched.coverage:.1f}',
                MatchedMSEReduction=f'{100*(1-matched.rmse**2/base.rmse**2):.1f}',
                ProfileBaseRMSE=f'{sem_base.rmse:.4f}',ProfileRMSE=f'{sem.rmse:.4f}',ProfileCoverage=f'{100*sem.coverage:.1f}',
                ReverseRMSE=f'{reverse.rmse:.4f}',ReverseResidualRMSE=f'{reverse_r.rmse:.4f}',
                ReverseResidualCoverage=f'{100*reverse_r.coverage:.1f}',DoubleCoverage=f'{100*scale2.coverage:.1f}')
    (OUT/'numbers.tex').write_text('\n'.join('\\newcommand{\\'+key+'}{'+value+'}' for key,value in macros.items())+'\n',encoding='utf-8')
    return macros


def main():
    frame=pd.read_csv(ROOT/'results/summary.csv')
    config=json.loads((ROOT/'results/config.json').read_text(encoding='utf-8'))
    manifest=json.loads((ROOT/'results/manifest.json').read_text(encoding='utf-8'))
    assert len(frame)==58*16 and frame.replications.eq(1000).all() and manifest['failures']==0
    for name,value in manifest['files'].items():
        assert hashlib.sha256((ROOT/'results'/name).read_bytes()).hexdigest()==value,name
    assert hashlib.sha256((ROOT/'run_boundary_benchmark.py').read_bytes()).hexdigest()==config['source_sha256']
    assert hashlib.sha256((ROOT/'PROTOCOL.md').read_bytes()).hexdigest()==config['protocol_sha256']
    figures(frame)
    numbers=tex_rows(frame)
    original=frame[frame.method=='original_sign_mom']
    wins=int((original.mse_improvement_lower95>0).sum())
    losses=int((original.mse_improvement_upper95<0).sum())
    result=dict(cells=58,replications=58000,methods=16,summary_rows=len(frame),failures=0,
                original_paired_mse_wins=wins,original_paired_mse_losses=losses,
                original_paired_mse_unclear=58-wins-losses,
                warning='Counts summarize the designed scenario grid, not a population probability of success; intervals are pointwise Monte Carlo descriptions.',
                key_numbers=numbers,original_coverage_min=float(original.coverage.min()),original_coverage_max=float(original.coverage.max()),
                oracle_finite_interval_coverage_min=float(frame[frame.method=='known_e_hoeffding_oracle_R'].coverage.min()),
                oracle_finite_interval_width_min=float(frame[frame.method=='known_e_hoeffding_oracle_R'].width.min()),
                oracle_finite_interval_width_max=float(frame[frame.method=='known_e_hoeffding_oracle_R'].width.max()))
    (ROOT/'results/report_summary.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':
    main()
