"""Generate manuscript tables, a scientific figure and an editorial revision memo."""
from pathlib import Path
import sys
sys.dont_write_bytecode=True
import os
ROOT=Path(__file__).resolve().parent
os.environ['MPLCONFIGDIR']=str(ROOT/'.mplconfig')
import csv
import hashlib
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

PAPER=ROOT/'manuscript_revised'
EVIDENCE=ROOT/'evidence'
WORK=ROOT.parent


def load(path): return json.loads(path.read_text(encoding='utf-8'))


def main():
    o=load(EVIDENCE/'oulad_common_design.json'); x=load(EVIDENCE/'xes_raw_source_common_design.json')
    boot=load(EVIDENCE/'oulad_orientation_bootstrap.json'); inventory=load(EVIDENCE/'data_inventory.json')
    manifest=load(EVIDENCE/'audit_manifest.json'); config=load(EVIDENCE/'audit_config.json')
    for name,digest in manifest['outputs'].items():
        assert hashlib.sha256((EVIDENCE/name).read_bytes()).hexdigest()==digest
    assert config['source_sha256']==hashlib.sha256((ROOT/'run_real_data_audit.py').read_bytes()).hexdigest()
    assert config['analysis_plan_sha256']==hashlib.sha256((ROOT/'REAL_DATA_AUDIT_PLAN.md').read_bytes()).hexdigest()
    assert inventory['oulad']['matches_archived_input_hashes']
    simulations=list(csv.DictReader((WORK/'effectiveness_validation/binary_results/summary.csv').open(encoding='utf-8')))
    def sim(s,m,alpha=0):
        return next(r for r in simulations if r['scenario']==s and r['method']==m and float(r['alpha'])==alpha)
    def interval(values): return f'[{values[0]:+.3f}, {values[1]:+.3f}]'
    targets=[('OULAD: non-withdrawal',o['n_common_support'],o['diagnostic']['targets']['not_withdrawn']),
             ('OULAD: successful completion',o['n_common_support'],o['diagnostic']['targets']['successful_completion'])]
    for name,d in x['designs'].items():
        targets.append((f'XES: Design {"A" if name=="A" else "B2"}',d['n_common_support'],d['diagnostic']['targets']['late_accuracy_high']))
    tex=[r'\begin{table}[htbp]\centering\small',
         r'\caption{New common-design observational contrasts. Signed correction and nominal MoM intervals retain the manuscript procedure. Designs differ from archived full-adjustment or weighted analyses.}\label{tab:real-results}',
         r'\begin{tabularx}{\textwidth}{Xrrrr}\toprule',
         r'Dataset / endpoint & $n$ & Raw & Corrected & Nominal interval \\\midrule']
    for name,n,t in targets:
        tex.append(f'{name} & {n} & {t["uncalibrated"]:+.3f} & {t["calibrated"]:+.3f} & {interval(t["manuscript_interval"])} '+r'\\')
    tex.extend([r'\bottomrule\end{tabularx}\end{table}',
                f'OULAD retains {o["n_common_support"]} registrations from {o["n_unique_students"]} students on common support. '+
                f'The new XES source contains {x["provenance"]["eligible_users"]} eligible users before the reproducible 6000-user sample. '+
                'The sample is user-disjoint across development and holdout.'])
    (PAPER/'real_results.tex').write_text('\n'.join(tex),encoding='utf-8')
    panels=[('OULAD',o['diagnostic']['pseudo_target_checks'],False)]
    for design,d in x['designs'].items():
        panels.append((f'XES {"A" if design=="A" else "B2"}',d['diagnostic']['pseudo_target_checks'],True))
    pseudo_stats=[]
    tex=[r'\begin{table}[htbp]\centering\small',
         r'\caption{Leave-one-control-out null-reference diagnostics. XES null exclusion depends additionally on the stated observation-process assumption. Counts are descriptive across a small dependent panel, not estimated population coverage.}\label{tab:pseudo-targets}',
         r'\begin{tabularx}{\textwidth}{Xrrrr}\toprule',
         r'Panel / evaluation & $K$ & Raw RMSE & Corrected RMSE & Intervals excluding 0 \\\midrule']
    for name,checks,has_holdout in panels:
        for evaluation in (['development','holdout'] if has_holdout else ['development']):
            rawkey='uncalibrated' if evaluation=='development' else 'holdout_uncalibrated'
            raw=np.array([v[rawkey] for v in checks]); corrected=np.array([v[f'{evaluation}_calibrated'] for v in checks])
            excluded=sum(v[f'{evaluation}_interval_excludes_zero'] for v in checks)
            rr=float(np.sqrt(np.mean(raw**2))); rc=float(np.sqrt(np.mean(corrected**2)))
            pseudo_stats.append(dict(panel=name,evaluation=evaluation,n_controls=len(checks),raw_rmse=rr,calibrated_rmse=rc,excluded=excluded))
            tex.append(f'{name}: {evaluation} & {len(checks)} & {rr:.3f} & {rc:.3f} & {excluded}/{len(checks)} '+r'\\')
    tex.extend([r'\bottomrule\end{tabularx}\end{table}',
                'The OULAD leave-one-control-out correction increases aggregate null-reference error rather than reducing it; previous attempts remains outside zero. '+
                'XES null-reference errors decrease in these designs, including holdout, but the small number of related controls and observation-window assumptions prevent treating this as target-bias validation.',
                f'The question-level source required removal of {x["provenance"]["padding_removed"]:,} padding entries and '+
                f'{x["provenance"]["duplicate_events_removed"]:,} repeated event keys. '+
                f'{x["provenance"]["conflicting_event_keys"]:,} repeated keys had conflicting response labels; the first stable-order record was retained. '+
                'This explicit cleaning convention is a sensitivity limitation.'])
    (PAPER/'pseudo_results.tex').write_text('\n'.join(tex),encoding='utf-8')
    tex=[r'A student-cluster bootstrap (2000 repetitions) of the full OULAD training-key cohort, before the support restriction, examines marginal proxy-sign stability. ',
         r'It does not refit nuisance models or estimate total causal-effect uncertainty.',
         r'\begin{table}[htbp]\centering\small',
         r'\caption{OULAD marginal sign stability in the full 5859-registration training cohort. Bootstrap sign retention is a resampling frequency, not a posterior probability.}\label{tab:orientation}',
         r'\begin{tabularx}{\textwidth}{Xrrr}\toprule Control & Correlation & Bootstrap interval & Sign retained \\\midrule']
    labels={'disability':'Declared disability','male':'Male indicator','age_35_plus':'Age 35+',
            'higher_education':'Higher education','imd_50_plus':'IMD 50+','previous_attempts':'Previous attempts'}
    for name,v in boot['controls'].items():
        tex.append(f'{labels[name]} & {v["correlation"]:+.3f} & {interval(v["ci95"])} & {100*v["probability_same_sign"]:.1f}\% '+r'\\')
    tex.extend([r'\bottomrule\end{tabularx}\end{table}',
                'In particular, the disability and previous-attempt correlations are small and their bootstrap intervals cross zero. '+
                'This weakens the case for treating their estimated orientations as known confounding directions. '+
                f'The leading component explains {100*o["diagnostic"]["first_component_variance_share"]:.1f}\% of the common-support control-correlation variance; '+
                'this is a descriptive statistic and neither confirms nor rejects a particular latent-factor model.'])
    (PAPER/'orientation_results.tex').write_text('\n'.join(tex),encoding='utf-8')
    simlabels={'matched':'Matched controls','different_magnitudes':'Different magnitudes','incompatible_factors':'Incompatible factors','no_confounding':'No confounding'}
    tex=[r'\begin{table}[htbp]\centering\small',
         r'\caption{Known-truth binary simulation of the unchanged sign-aware/MoM rule. Each row uses 500 repetitions, $n=2000$ and two-fold AIPW with logistic working models. The non-null ATE is computed by quadrature.}\label{tab:simulation-results}',
         r'\begin{tabularx}{\textwidth}{Xrrrrr}\toprule Scenario & True ATE & Raw RMSE & Signed RMSE & Coverage & Reject 0 \\\midrule']
    for s in simlabels:
        for alpha in [0,.5]:
            r=sim(s,'sign_mom',alpha); b=sim(s,'uncalibrated',alpha)
            tex.append(f'{simlabels[s]} & {float(r["true_ate"]):.3f} & {float(b["rmse"]):.3f} & {float(r["rmse"]):.3f} & {float(r["coverage"]):.3f} & {float(r["rejection_rate"]):.3f} '+r'\\')
    tex.extend([r'\bottomrule\end{tabularx}\end{table}',
                r'At coverage 0.95, Monte Carlo standard error is about 0.010 for 500 repetitions. Rejection at true zero measures type-I error; at nonzero effects it measures power. ',
                r'\begin{figure}[htbp]\centering\includegraphics[width=\textwidth]{audit_summary.png}',
                r'\caption{Conditional simulation performance and real-panel diagnostics. Top: binary known-truth RMSE and nominal coverage (coverage bars show $\pm1.96$ Monte Carlo SE). Bottom left: OULAD cluster-bootstrap marginal correlations on full training keys. Bottom right: null-reference error before and after leave-one-control-out correction. Real-panel checks do not identify target bias.}\label{fig:audit-summary}\end{figure}'])
    (PAPER/'simulation_results.tex').write_text('\n'.join(tex),encoding='utf-8')
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(2,2,figsize=(13,9)); positions=np.arange(4); width=.36
    ss=list(simlabels); raw=[float(sim(s,'uncalibrated')['rmse']) for s in ss]; signed=[float(sim(s,'sign_mom')['rmse']) for s in ss]
    axes[0,0].bar(positions-width/2,raw,width,label='Uncalibrated',color='#8795a1'); axes[0,0].bar(positions+width/2,signed,width,label='Sign-aware MoM',color='#24648d')
    axes[0,0].set(xticks=positions,xticklabels=['Matched','Different\nmagnitudes','Incompatible\nfactors','No\nconfounding'],ylabel='RMSE (risk difference)',title='A. Binary null-effect simulation'); axes[0,0].legend(frameon=False)
    cov=np.array([float(sim(s,'sign_mom')['coverage']) for s in ss]); err=1.96*np.array([float(sim(s,'sign_mom')['coverage_mcse']) for s in ss])
    axes[0,1].bar(positions,cov,color='#24648d',yerr=err,capsize=3); axes[0,1].axhline(.95,color='#a73d3d',ls='--',label='Nominal 95%')
    axes[0,1].set(xticks=positions,xticklabels=['Matched','Different\nmagnitudes','Incompatible\nfactors','No\nconfounding'],ylim=(0,1.04),ylabel='Interval coverage',title='B. Sign-aware interval performance'); axes[0,1].legend(frameon=False)
    vv=list(boot['controls'].values()); yy=np.arange(6); point=np.array([v['correlation'] for v in vv]); ci=np.array([v['ci95'] for v in vv])
    axes[1,0].errorbar(point,yy,xerr=np.vstack((point-ci[:,0],ci[:,1]-point)),fmt='o',color='#24648d',capsize=3)
    axes[1,0].axvline(0,color='#777',lw=1); axes[1,0].set(yticks=yy,yticklabels=list(labels.values()),xlabel='Marginal proxy correlation',title='C. OULAD sign stability'); axes[1,0].invert_yaxis()
    xx=np.arange(len(pseudo_stats)); axes[1,1].bar(xx-width/2,[v['raw_rmse'] for v in pseudo_stats],width,color='#8795a1',label='Raw')
    axes[1,1].bar(xx+width/2,[v['calibrated_rmse'] for v in pseudo_stats],width,color='#24648d',label='LOCO corrected')
    axes[1,1].set(xticks=xx,xticklabels=[v['panel']+'\n'+('dev' if v['evaluation']=='development' else 'holdout') for v in pseudo_stats],ylabel='Null-reference RMSE',title='D. Real-panel null prediction'); axes[1,1].legend(frameon=False)
    fig.tight_layout(); fig.savefig(PAPER/'audit_summary.png',dpi=220); plt.close(fig)
    abstract=(
        'Negative-control estimates can be aligned in sign before pooling, but orientation alone does not establish that heterogeneous controls measure commensurable residual confounding. '+
        'We retain the sign-aware equal-pooling rule and assess its conditions and limitations in educational data. '+
        'A bias--variance identity gives the exact finite-sample condition for MSE improvement; it does not identify unknown target bias. '+
        'In known-truth binary simulations, matched controls reduce RMSE, whereas nonmatching magnitudes and incompatible factors produce poor coverage. '+
        'New common-design audits use checksum-verified OULAD inputs and an explicitly named XES3G5M sequence source. '+
        'All outcomes share adjustment, cross-fit folds and support, and real-control timing and coding are documented. '+
        'OULAD leave-one-control-out null prediction does not improve overall, and orientations for two consequential controls have weak marginal proxy support. '+
        'XES null-reference errors decrease in development and holdout, conditional on additional observation-process assumptions. '+
        'The results support conditional use and diagnostic reporting rather than a generally identified causal-calibration method. '+
        'Neither non-rejection of a control null nor a target interval crossing zero validates target-bias correction.\n\n'+
        r'\noindent\textbf{Keywords:} negative controls; calibration; unmeasured confounding; educational data mining; sensitivity analysis.\n')
    (PAPER/'abstract.tex').write_text(abstract.replace('\\n','\n'),encoding='utf-8')
    # Honest editorial disposition; not a sendable letter or a claim of readiness.
    lines=['# Доработка по замечаниям редактора','',
           'Исходная процедура сохранена: ориентация по прокси, равные веса и MoM-пул. Расширения с эталоном и адаптивными весами в эту версию не включены. Отдельная копия рукописи подготовлена в manuscript_revised.','',
           '## Что изменено','',
           '| Замечание редактора | Сделанная доработка | Статус |',
           '|---|---|---|',
           '| Выравнивание знаков не устанавливает новый causal method | Переписаны название, abstract, вклад и обсуждение; ориентация названа простой модификацией; добавлено сопоставление с COCA/bridge | Исправлено позиционирование; это не гарантия достаточной новизны для JEDM |',
           '| Не обоснована сопоставимость разнородных контролей | Добавлены точные условия MSE, отдельная карточка для каждого контроля, общий adjustment/support, LOCO и holdout null-reference checks | Проверено диагностически; равенство истинного target bias не доказано |',
           '| Эффективность может быть следствием заданного DGP | Включены совпадающие и нарушенные условия бинарной симуляции без отбора благоприятных строк | Условная эффективность и пределы показаны вместе |','',
           '## Важный результат: замечание о сопоставимости нельзя снять одной редактурой','',
           'На OULAD коррекция не уменьшила суммарную ошибку предсказания нулевых контролей. Для previous attempts nominal LOCO interval по-прежнему исключает ноль. Это не позволяет написать, что панель проверена как общий измеритель bias. В версии статьи это ограничение явно сохранено.','',
           f'Bootstrap полного train-cohort: знак disability сохранялся только в {100*boot["controls"]["disability"]["probability_same_sign"]:.1f}% повторений; '+
           f'для previous attempts — в {100*boot["controls"]["previous_attempts"]["probability_same_sign"]:.1f}%. '+
           'Оба 95%-интервала marginal proxy correlation включают ноль. Bootstrap не включает nuisance refitting и не является total ATE CI.','',
           'XES LOCO/holdout ошибки меньше исходных, но четыре связанных prefix-контроля и окна, зависящие от полной длины последовательности, ограничивают этот аргумент. Ненулевой causal effect реального основного исхода не известен, поэтому его bias непосредственно не измерен.','',
           '## Найденные расхождения кода и текста','',
           '1. OULAD input hashes совпали с архивом. Но код определяет outcome как not Withdrawn, а статья описывала not(Fail or Withdrawn). Новая версия отдельно показывает non-withdrawal и successful completion.',
           '2. Текущий module runner задаёт window_frac=0.5, хотя текст описывал first-deadline cutoff. Новый audit использует window_frac=None. Даже pre-deadline признаки могут следовать фактической ранней submission; эта временная проблема явно оговорена.',
           '3. В прежнем XES-коде для каждого NC удалялся свой признак и заново оценивались propensity/support. Новый audit использует один ограниченный X и один support для всех outcomes; это новый sensitivity design, не идентичная репликация старого full-X.',
           '4. Generic data/splits сейчас имеют EdNet-признаки и user-prefix u. Старые XES commands указывали на эту папку, без архивных входных хешей. Историческая подмена не утверждается, но воспроизводимость старых XES-чисел не подтверждена. Новый audit читает именованный raw XES3G5M question-level файл и фиксирует его SHA-256.',
           '5. XES item accuracy для early ability раньше использовала поздние ответы и отдельно весь holdout. Новая карта сложности строится только по early prefixes development, затем применяется к обеим группам.',
           '6. Session builder раньше сравнивал timestamp с началом сессии, а не с предыдущим interaction. Новый B2 использует соседний gap >1800 секунд.',
           '7. Прежняя B2-weighting formula зависела от наблюдаемого treatment; её нельзя автоматически описывать как doubly robust overlap-ATE. Новый diagnostic использует обычный AIPW на общем support. Orientation/pooling остаётся прежним.',
           '8. Два OULAD script sources отсутствуют (сохранились pyc). Их старые результаты не объявлены полностью воспроизведёнными.','',
           '## Новые результаты: общий диагностический дизайн','',
           '| Набор / endpoint | N на support | До коррекции | После | Nominal MoM interval |',
           '|---|---:|---:|---:|---|']
    for name,n,t in targets:
        lines.append(f'| {name} | {n} | {t["uncalibrated"]:+.4f} | {t["calibrated"]:+.4f} | {interval(t["manuscript_interval"])} |')
    lines+=['','Эти оценки получены при новом явно описанном common-adjustment design. Они не сравниваются со старым +0.049 как если бы изменились только знаки или только один параметр. Положительные/нулевые observational contrasts не считаются доказанными intervention effects.','',
            '## Статус для подачи','',
            'Подготовлена научно более осторожная рабочая версия и новая доказательная база. Заявлять, что все основания отказа сняты и статью можно сразу повторно подавать в JEDM, нельзя. Ключевое эмпирическое равенство bias для исходной OULAD-панели не подтверждено, а простой orientation step не стал новым идентификационным методом.',
            'Сохранить исходный метод можно как анализ условий применимости и диагностический case study. Для сохранения сильного утверждения о causal bias correction нужны более подходящие предметно обоснованные контроли или независимый ориентир target effect. Эта доработка не имитирует такое отсутствующее доказательство.','',
            '## Воспроизводимость','',
            '- REAL_DATA_AUDIT_PLAN.md: решения до нового пересчёта.',
            '- evidence/data_inventory.json: входы, совпадение OULAD-хешей, демографические агрегаты и источники кода.',
            '- evidence/oulad_common_design.json и xes_raw_source_common_design.json: все реальные target/control estimates и pseudo-target checks.',
            '- evidence/oulad_orientation_bootstrap.json: bootstrap marginal proxy signs.',
            '- evidence/audit_config.json и audit_manifest.json: конфигурация, версии и контрольные суммы.',
            '- manuscript_revised: отдельная редактируемая LaTeX-версия; исходная manuscript_updated не изменялась.','']
    (ROOT/'EDITORIAL_REVISION_RU.md').write_text('\n'.join(lines),encoding='utf-8')
    summary=dict(status='evidence_generated',targets=targets,pseudo_target_summaries=pseudo_stats,
                 oulad_orientation_bootstrap=boot['controls'],unresolved='Target-bias commensurability and sufficient methodological novelty are not established')
    (EVIDENCE/'revision_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps(dict(pseudo_target_summaries=pseudo_stats,files='manuscript inputs, figure and revision memo generated'),indent=2),flush=True)


if __name__=='__main__': main()
