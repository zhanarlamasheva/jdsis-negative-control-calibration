"""Independent aggregate reconstruction from the saved replicate table."""
from pathlib import Path
import sys
sys.dont_write_bytecode=True
import json
import hashlib
import numpy as np
import pandas as pd

root=Path(__file__).resolve().parent
results=root/'results'
summary=pd.read_csv(results/'summary.csv')
raw=pd.read_csv(results/'replications.csv.gz',dtype={'scenario':'category','method':'category'})
assert len(raw)==58*1000*16
assert not raw.duplicated(['scenario','replication','method']).any()
assert np.isfinite(raw.estimate).all()
point_only=raw.method=='signed_median_point_only'
assert raw.loc[point_only,['lower','upper']].isna().all().all()
assert np.isfinite(raw.loc[~point_only,['lower','upper']]).all().all()
assert (raw.loc[~point_only,'lower']<=raw.loc[~point_only,'upper']).all()
raw['error']=raw.estimate-raw.truth
raw['squared_error']=raw.error**2
raw['covered']=np.where(point_only,np.nan,((raw.lower<=raw.truth)&(raw.truth<=raw.upper)).astype(float))
raw['rejected']=np.where(point_only,np.nan,((raw.lower>0)|(raw.upper<0)).astype(float))
raw['width']=raw.upper-raw.lower
aggregate=raw.groupby(['scenario','method'],observed=True).agg(n=('estimate','size'),bias=('error','mean'),
            mse=('squared_error','mean'),coverage=('covered','mean'),rejection=('rejected','mean'),width=('width','mean')).reset_index()
aggregate['rmse']=np.sqrt(aggregate.mse)
assert aggregate.n.eq(1000).all()
merged=summary.merge(aggregate,on=['scenario','method'],validate='one_to_one',suffixes=('_saved','_reconstructed'))
errors={}
for metric in ['bias','rmse','coverage','rejection','width']:
    difference=np.abs(merged[metric+'_saved']-merged[metric+'_reconstructed'])
    errors[metric]=float(difference.max())
    assert errors[metric]<1e-11,(metric,errors[metric])
base=raw[raw.method=='uncorrected_wald'][['scenario','replication','squared_error']].rename(columns={'squared_error':'base_error'})
paired=raw.merge(base,on=['scenario','replication'],validate='many_to_one')
paired['improvement']=paired.base_error-paired.squared_error
paired_summary=paired.groupby(['scenario','method'],observed=True).agg(mean=('improvement','mean'),sd=('improvement','std')).reset_index()
merged=summary.merge(paired_summary,on=['scenario','method'],validate='one_to_one')
assert np.max(np.abs(merged.mse_improvement-merged['mean']))<1e-11
assert np.max(np.abs(merged.mse_improvement_mcse-merged.sd/np.sqrt(1000)))<1e-11
config=json.loads((results/'config.json').read_text(encoding='utf-8'))
assert config['source_sha256']==hashlib.sha256((root/'run_boundary_benchmark.py').read_bytes()).hexdigest()
assert config['protocol_sha256']==hashlib.sha256((root/'PROTOCOL.md').read_bytes()).hexdigest()
report=dict(status='passed',replicate_rows=len(raw),scenario_method_groups=len(summary),reconstructed_metrics=errors,
            paired_mse_and_mcse='passed',protocol_and_source_hashes='passed',
            note='Checks numerical consistency, not causal assumptions or novelty.')
(results/'verification.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps(report,indent=2))
