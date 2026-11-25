"""Common-design negative-control audit on recovered, explicitly named inputs.

Preserves sign orientation, equal pooling, and manuscript MoM intervals.
Does not claim causal validation from observational negative-control diagnostics.
"""
from pathlib import Path
import sys
sys.dont_write_bytecode=True
import hashlib
import importlib.metadata
import json
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from xgboost import XGBClassifier

ROOT=Path(__file__).resolve().parent
WORK=ROOT.parent
PROJECT=WORK.parent.parent/'education_models'
ARTICLE=WORK.parent
RAW_OULAD=WORK.parent.parent.parent/'датасет/oulad'
OUT=ROOT/'evidence'
SEED=20260626


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()


def classifier():
    return XGBClassifier(n_estimators=300,max_depth=4,learning_rate=.05,subsample=.8,
                         colsample_bytree=.8,eval_metric='logloss',random_state=SEED,n_jobs=4)


def probability(X,y,Xtest):
    if len(np.unique(y))==1: return np.full(len(Xtest),float(y[0]))
    return classifier().fit(X,y).predict_proba(Xtest)[:,1]


def nuisance_joint(X,T,values,groups,Xtest=None):
    folds=list(StratifiedGroupKFold(n_splits=5,shuffle=True,random_state=SEED).split(X,T,groups))
    e=np.zeros(len(T)); m0=np.zeros_like(values,dtype=float); m1=m0.copy()
    for k,(train,test) in enumerate(folds):
        assert not set(groups[train])&set(groups[test])
        e[test]=probability(X[train],T[train],X[test])
        for j in range(values.shape[1]):
            for a,mu in [(0,m0),(1,m1)]:
                ii=train[T[train]==a]
                mu[test,j]=probability(X[ii],values[ii,j],X[test])
        print(f'  cross-fit fold {k+1}/5 completed',flush=True)
    holdout=None
    if Xtest is not None:
        et=probability(X,T,Xtest)
        zero=np.column_stack([probability(X[T==0],values[T==0,j],Xtest) for j in range(values.shape[1])])
        one=np.column_stack([probability(X[T==1],values[T==1,j],Xtest) for j in range(values.shape[1])])
        holdout=(et,zero,one)
    return e,m0,m1,holdout


def clustered_covariance(influence,groups):
    _,idx=np.unique(groups,return_inverse=True); k=int(idx.max()+1)
    sums=np.zeros((k,influence.shape[1])); np.add.at(sums,idx,influence)
    return sums.T@sums/len(groups)**2 * k/max(1,k-1)


def scores(T,values,e,m0,m1,groups):
    keep=(e>=.02)&(e<=.98)
    ee=np.clip(e[keep],.02,.98)[:,None]; a=T[keep,None]; yy=values[keep]
    psi=m1[keep]-m0[keep]+a*(yy-m1[keep])/ee-(1-a)*(yy-m0[keep])/(1-ee)
    estimate=psi.mean(axis=0); influence=psi-estimate
    cov=clustered_covariance(influence,groups[keep])
    return estimate,cov,keep,influence


def correlation(x,y):
    return float(np.corrcoef(x,y)[0,1]) if np.std(x)>0 and np.std(y)>0 else 0.


def mom(estimate,ses):
    mu=float(np.mean(estimate)); spread=float(np.sqrt(max(0.,np.var(estimate,ddof=1)-np.mean(np.array(ses)**2))))
    return mu,spread


def panel_summary(values,g,estimate,cov,names,target_indices,control_indices,holdout=None):
    se=np.sqrt(np.diag(cov)); rho=np.array([correlation(values[:,j],g) for j in range(values.shape[1])])
    controls={names[j]:dict(estimate=float(estimate[j]),se=float(se[j]),
                           prevalence=float(values[:,j].mean()),proxy_correlation=float(rho[j])) for j in control_indices}
    targets={}
    for j in target_indices:
        orient=np.sign(rho[control_indices])*np.sign(rho[j]); orient=np.where(orient==0,1,orient)
        mu,spread=mom(estimate[control_indices]*orient,se[control_indices])
        cal=float(estimate[j]-mu); cal_se=float(np.sqrt(se[j]**2+spread**2))
        a=np.zeros(len(names)); a[j]=1; a[control_indices]=-orient/len(control_indices)
        fixed_sign_se=float(np.sqrt(max(0.,a@cov@a)))
        targets[names[j]]=dict(uncalibrated=float(estimate[j]),se=float(se[j]),
                               ci95=[float(estimate[j]-1.96*se[j]),float(estimate[j]+1.96*se[j])],
                               proxy_correlation=float(rho[j]),orientations=orient.astype(int).tolist(),
                               signed_mean=mu,heterogeneity=spread,calibrated=cal,
                               manuscript_interval=[cal-1.96*cal_se,cal+1.96*cal_se],
                               covariance_se_diagnostic=fixed_sign_se,
                               covariance_interval_diagnostic=[cal-1.96*fixed_sign_se,cal+1.96*fixed_sign_se],
                               interval_status='MoM formula; not independently validated causal total-uncertainty coverage')
    pseudo=[]
    for j in control_indices:
        other=[k for k in control_indices if k!=j]
        orient=np.sign(rho[other])*np.sign(rho[j]); orient=np.where(orient==0,1,orient)
        mu,spread=mom(estimate[other]*orient,se[other])
        point=float(estimate[j]-mu); wid=float(np.sqrt(se[j]**2+spread**2)); ci=[point-1.96*wid,point+1.96*wid]
        entry=dict(pseudo_target=names[j],true_effect=0.,uncalibrated=float(estimate[j]),
                   development_calibrated=point,development_interval=ci,development_interval_excludes_zero=bool(ci[0]>0 or ci[1]<0))
        if holdout is not None:
            ht,hc=holdout; hse=float(np.sqrt(hc[j,j])); hp=float(ht[j]-mu)
            hw=float(np.sqrt(hse*hse+spread*spread)); hci=[hp-1.96*hw,hp+1.96*hw]
            entry.update(holdout_uncalibrated=float(ht[j]),holdout_calibrated=hp,holdout_interval=hci,
                         holdout_interval_excludes_zero=bool(hci[0]>0 or hci[1]<0))
        pseudo.append(entry)
    corr=np.corrcoef(values[:,control_indices],rowvar=False)
    eig=np.linalg.eigvalsh(np.nan_to_num(corr)); eig=np.maximum(eig,0)
    return dict(controls=controls,targets=targets,pseudo_target_checks=pseudo,
                control_correlation=corr.tolist(),control_correlation_eigenvalues=eig.tolist(),
                first_component_variance_share=float(eig[-1]/max(eig.sum(),1e-12)),
                caution='Correlation structure and null pseudo-target checks do not identify target confounding bias')


def oulad():
    sys.path.insert(0,str(PROJECT))
    from models.uplift_oulad import OULADUplift
    from models.pred_oulad import StaticEncoder
    up=OULADUplift(processed_dir=str(PROJECT/'data/oulad'),data_dir=str(RAW_OULAD),
                   results_dir=str(OUT),window_frac=None,k_folds=5,seed=42)
    daily,target,assess=up._load(); keys=up.pred._load_keys('train')
    T,Xsource,ysource,aligned,names=up._build_TXY(daily,target,assess,keys,fit_enc=True)
    keycols=['id_student','code_module','code_presentation']
    demo=aligned.merge(target,on=keycols,how='left',validate='one_to_one')
    assert np.array_equal(ysource,(demo.final_result!='Withdrawn').to_numpy().astype(int))
    enc=StaticEncoder(cat_cols=['region'],num_cols=['studied_credits','date_registration'])
    X=np.column_stack((enc.fit_transform(demo),Xsource[:,-7:])).astype(np.float32)
    imd=pd.to_numeric(demo.imd_band.astype(str).str.extract(r'^(\d+)',expand=False),errors='coerce')
    controls=np.column_stack(((demo.disability=='Y').astype(int),(demo.gender=='M').astype(int),
                              (demo.age_band!='0-35').astype(int),demo.highest_education.isin(['HE Qualification','Post Graduate Qualification']).astype(int),
                              (imd>=50).astype(int),(demo.num_of_prev_attempts>0).astype(int)))
    Ysuccess=demo.final_result.isin(['Pass','Distinction']).to_numpy().astype(int)
    values=np.column_stack((ysource,Ysuccess,controls)); group=demo.id_student.to_numpy(); g=Xsource[:,-7]
    names=['not_withdrawn','successful_completion','disability','male','age_35_plus','higher_education','imd_50_plus','previous_attempts']
    print('OULAD common-design fitting',len(T),'registrations',flush=True)
    e,m0,m1,_=nuisance_joint(X,T,values,group)
    estimate,cov,keep,influence=scores(T,values,e,m0,m1,group)
    summary=panel_summary(values[keep],g[keep],estimate,cov,names,[0,1],list(range(2,8)))
    result=dict(dataset='OULAD',analysis='post-screening common-adjustment diagnostic',n_before_support=len(T),n_common_support=int(keep.sum()),
                n_unique_students=int(pd.Series(group[keep]).nunique()),n_outside_support=int((~keep).sum()),
                imd_missing=int(imd.isna().sum()),cutoff='strictly before first assessment deadline',
                outcome_definitions={'not_withdrawn':'final_result != Withdrawn','successful_completion':'Pass or Distinction'},
                common_covariates=['region','studied_credits','date_registration','seven pre-deadline VLE trend features'],
                covariate_exclusion='All six candidate control fields excluded for every target/control analysis',
                group_crossfit='5 folds by id_student',se='student-cluster influence sums; nuisance models treated through score approximation',
                diagnostic=summary)
    (OUT/'oulad_common_design.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print('OULAD completed',json.dumps(summary['targets']),flush=True)
    return result


def parse_xes():
    path=PROJECT/'data/xes3g5m/XES3G5M/question_level/train_valid_sequences_quelevel.csv'
    raw=pd.read_csv(path); groups={}; dropped=0
    for row in raw.itertuples(index=False):
        q=np.fromstring(str(row.questions),sep=',',dtype=np.int64)
        r=np.fromstring(str(row.responses),sep=',',dtype=np.int8)
        t=np.fromstring(str(row.timestamps),sep=',',dtype=np.float64)
        tokens=str(row.concepts).split(',')
        c=np.array([int(v.split('_')[0]) if v.split('_')[0].lstrip('-').isdigit() else -1 for v in tokens],dtype=np.int64)
        assert len(q)==len(r)==len(t)==len(c)
        keep=(r>=0)&(r<=1)&(t>=0)&(q>=0); dropped+=int((~keep).sum())
        groups.setdefault(str(row.uid),[]).append(np.column_stack((q[keep],r[keep],t[keep],c[keep])))
    sequences={}; duplicates=0; conflicts=0
    for uid,parts in groups.items():
        a=np.concatenate(parts); a=a[np.argsort(a[:,2],kind='stable')]
        frame=pd.DataFrame(a,columns=['q','r','t','c'])
        conflicts+=int((frame.groupby(['q','t']).r.nunique()>1).sum())
        dup=frame.duplicated(['q','t']); duplicates+=int(dup.sum()); a=a[~dup.to_numpy()]
        if len(a)>=20: sequences[uid]=a
    eligible=np.array(sorted(sequences)); assert len(eligible)>=6000
    selected=np.random.default_rng(SEED).choice(eligible,6000,replace=False)
    selected=np.array(sorted(selected)); order=np.random.default_rng(SEED).permutation(selected)
    holdout=set(order[:1200]); development=[u for u in selected if u not in holdout]
    sums={}; counts={}
    for uid in development:
        a=sequences[uid]; early=a[:max(1,int(.3*len(a)))]; unique=np.unique(early[:,0])
        for q in unique:
            ans=early[early[:,0]==q,1]
            sums[int(q)]=sums.get(int(q),0.)+float(ans.sum()); counts[int(q)]=counts.get(int(q),0)+len(ans)
    difficulty={q:sums[q]/counts[q] for q in sums}
    rows=[]
    for uid in selected:
        a=sequences[uid]; n=len(a); i=max(1,int(.3*n)); j=min(n-1,max(i+1,int(.6*n)))
        early=a[:i]; middle=a[i:j]; late=a[j:]
        ability=float(np.mean(early[:,1]-np.array([difficulty.get(int(q),.5) for q in early[:,0]])))
        ts=np.unique(middle[:,2]); magnitude=np.median(np.abs(ts))
        if magnitude>=1e14: ts=ts/1e6
        elif magnitude>=1e11: ts=ts/1e3
        start=np.r_[True,np.diff(ts)>1800]
        sessions=ts[start]; gaps=np.diff(sessions); gaps=gaps[gaps>0]
        reg=float(1/(1+gaps.std(ddof=1)/gaps.mean())) if len(gaps)>=3 and gaps.mean()>0 else np.nan
        rows.append(dict(holdout=uid in holdout,early_ability=ability,early_accuracy=float(early[:,1].mean()),early_activity=len(early),
                         early_concept_diversity=len(np.unique(early[:,3])),early_wrong_concepts=len(np.unique(early[early[:,1]==0,3])),
                         middle_accuracy=float(middle[:,1].mean()),late_accuracy=float(late[:,1].mean()),
                         middle_session_regularity=reg,middle_sessions=len(sessions),group=uid))
    provenance=dict(raw_source=str(path),sha256=sha(path),source_rows=len(raw),eligible_users=len(eligible),selected_users=6000,
                    padding_removed=dropped,duplicate_events_removed=duplicates,conflicting_event_keys=conflicts,
                    difficulty_source='Only early prefixes of development users; shared map applied to holdout',
                    session_definition='Gap >1800 seconds between consecutive unique timestamps',
                    window_caveat='Fractional windows depend on complete observed sequence length; temporal NCO validity requires an additional observation-process assumption',
                    archive_comparability='New raw-source cohort; archived generic-split XES results are not claimed reproduced')
    return pd.DataFrame(rows),provenance


def xes():
    print('Parsing named XES3G5M question-level source',flush=True)
    rows,provenance=parse_xes(); reports={}
    feature_names=['early_accuracy','early_ability','early_activity','early_concept_diversity']
    for design,field in [('A','middle_accuracy'),('B2_gap_corrected','middle_session_regularity')]:
        use=rows.copy()
        if design!='A': use=use[(use.middle_sessions>=4)&use.middle_session_regularity.notna()].copy()
        dev=use[~use.holdout].copy(); hold=use[use.holdout].copy()
        thresholds={name:float(dev[name].median()) for name in ['late_accuracy',field,*feature_names]}
        vals=lambda d:np.column_stack(((d.late_accuracy>=thresholds['late_accuracy']).astype(int),
                                     *[(d[f]>=thresholds[f]).astype(int) for f in feature_names]))
        values=vals(dev); htvalues=vals(hold); T=(dev[field]>=thresholds[field]).to_numpy().astype(int)
        Ttest=(hold[field]>=thresholds[field]).to_numpy().astype(int)
        X=dev[['early_wrong_concepts']].to_numpy(dtype=np.float32); Xt=hold[['early_wrong_concepts']].to_numpy(dtype=np.float32)
        groups=dev.group.to_numpy(); gt=hold.group.to_numpy()
        print('XES',design,'fitting',len(dev),'development users;',len(hold),'holdout',flush=True)
        e,m0,m1,h=nuisance_joint(X,T,values,groups,Xt)
        estimate,cov,keep,_=scores(T,values,e,m0,m1,groups)
        et,hm0,hm1=h; he,hc,hkeep,_=scores(Ttest,htvalues,et,hm0,hm1,gt)
        names=['late_accuracy_high']+[f'{f}_high' for f in feature_names]
        summary=panel_summary(values[keep],dev.early_ability.to_numpy()[keep],estimate,cov,names,[0],[1,2,3,4],(he,hc))
        reports[design]=dict(n_development=len(dev),n_holdout=len(hold),n_common_support=int(keep.sum()),n_holdout_support=int(hkeep.sum()),
                              thresholds=thresholds,common_covariates=['early_wrong_concepts'],estimand='ATE on common estimated propensity support; observational',
                              diagnostic=summary)
        print('XES',design,'completed',json.dumps(summary['targets']),flush=True)
    result=dict(dataset='XES3G5M',provenance=provenance,designs=reports)
    (OUT/'xes_raw_source_common_design.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    return result


def main():
    OUT.mkdir(exist_ok=True); start=time.time()
    plan=ROOT/'REAL_DATA_AUDIT_PLAN.md'
    config=dict(seed=SEED,analysis_plan_sha256=sha(plan),source_sha256=sha(Path(__file__)),python=sys.executable,
                packages={m:importlib.metadata.version(m) for m in ['numpy','pandas','scikit-learn','xgboost','pyarrow']},
                note='Prospective post-editorial diagnostic; original sign/equal-pool/MoM correction retained')
    (OUT/'audit_config.json').write_text(json.dumps(config,indent=2),encoding='utf-8')
    oulad(); xes()
    manifest=dict(status='completed',elapsed_seconds=round(time.time()-start,2),
                  outputs={p.name:sha(p) for p in OUT.glob('*common_design.json')})
    (OUT/'audit_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print(json.dumps(manifest,indent=2),flush=True)


if __name__=='__main__': main()
