"""Student-cluster bootstrap of OULAD proxy signs on the full training cohort.

Aggregate diagnostic only. This is not a refitted total-uncertainty ATE bootstrap.
"""
from pathlib import Path
import sys
sys.dont_write_bytecode=True
import json
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parent
PROJECT=ROOT.parent.parent.parent/'education_models'


def main():
    idx=['id_student','code_module','code_presentation']
    target=pd.read_parquet(PROJECT/'data/oulad/processed/student_target.parquet')
    keys=pd.read_parquet(PROJECT/'data/oulad/splits/train.parquet')[idx].drop_duplicates()
    demo=keys.merge(target,on=idx,validate='one_to_one')
    daily=pd.read_parquet(PROJECT/'data/oulad/processed/daily_vle.parquet')
    frame=daily.merge(demo[idx+['first_deadline']],on=idx,how='inner')
    frame=frame[frame.date<frame.first_deadline]
    totals=frame.groupby(idx).sum_click.sum().rename('g').reset_index()
    demo=demo.merge(totals,on=idx,how='left'); g=demo.g.fillna(0).to_numpy(dtype=float)
    imd=pd.to_numeric(demo.imd_band.astype(str).str.extract(r'^(\d+)',expand=False),errors='coerce')
    names=['disability','male','age_35_plus','higher_education','imd_50_plus','previous_attempts']
    c=np.column_stack(((demo.disability=='Y').astype(int),(demo.gender=='M').astype(int),
                       (demo.age_band!='0-35').astype(int),demo.highest_education.isin(['HE Qualification','Post Graduate Qualification']).astype(int),
                       (imd>=50).astype(int),(demo.num_of_prev_attempts>0).astype(int))).astype(float)
    # Every per-row sufficient moment is additive within student clusters.
    moments=np.column_stack((np.ones(len(g)),g,g*g,c,c*c,g[:,None]*c))
    _,group=np.unique(demo.id_student.to_numpy(),return_inverse=True); nclusters=int(group.max()+1)
    sums=np.zeros((nclusters,moments.shape[1])); np.add.at(sums,group,moments)
    def rho(t):
        n=t[:,0,None]; sg=t[:,1,None]; sgg=t[:,2,None]
        sc=t[:,3:9]; scc=t[:,9:15]; sgc=t[:,15:21]
        return (sgc-sg*sc/n)/np.sqrt(np.maximum((sgg-sg*sg/n)*(scc-sc*sc/n),1e-30))
    point=rho(sums.sum(axis=0)[None,:])[0]
    rng=np.random.default_rng(20261029); repetitions=2000; draws=[]
    for start in range(0,repetitions,100):
        counts=rng.multinomial(nclusters,np.ones(nclusters)/nclusters,size=min(100,repetitions-start))
        draws.append(rho(counts@sums))
    draws=np.concatenate(draws)
    controls={name:dict(correlation=float(point[j]),ci95=np.quantile(draws[:,j],[.025,.975]).tolist(),
                        probability_same_sign=float(np.mean(np.sign(draws[:,j])==np.sign(point[j])))) for j,name in enumerate(names)}
    result=dict(cohort='Full archived OULAD training keys; before common-support restriction',n_registrations=len(demo),
                n_student_clusters=nclusters,bootstrap_repetitions=repetitions,seed=20261029,controls=controls,
                limitation='No nuisance refitting; this measures marginal proxy-sign stability only, not bias comparability or total ATE uncertainty')
    (ROOT/'evidence/oulad_orientation_bootstrap.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__': main()
