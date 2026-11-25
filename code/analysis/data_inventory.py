"""Read-only inventory of the recovered data and code; aggregate output only."""
from pathlib import Path
import sys
sys.dont_write_bytecode=True
import hashlib
import json
import importlib.metadata
import pandas as pd

ROOT=Path(__file__).resolve().parent
WORK=ROOT.parent
PROJECT=WORK.parent.parent/'education_models'
ARTICLE=WORK.parent


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''): h.update(block)
    return h.hexdigest()


def main():
    out=ROOT/'evidence'; out.mkdir(exist_ok=True)
    manifest=json.loads((ARTICLE/'results/oulad_manifest_strict.json').read_text())
    files=[PROJECT/'data/oulad/processed/student_target.parquet',PROJECT/'data/oulad/processed/daily_vle.parquet']
    hashes={str(p.relative_to(PROJECT)).replace('\\','/'):digest(p) for p in files}
    recorded={k.replace('\\','/'):v for k,v in manifest['input_sha256'].items()}
    target=pd.read_parquet(files[0]); keys=pd.read_parquet(PROJECT/'data/oulad/splits/train.parquet')
    index=['id_student','code_module','code_presentation']
    train=keys[index].drop_duplicates().merge(target,on=index,how='left',validate='one_to_one')
    demo={c:{str(k):int(v) for k,v in train[c].value_counts(dropna=False).items()}
          for c in ['gender','age_band','highest_education','imd_band','disability','num_of_prev_attempts']}
    xes=[]
    for split in ['train','val','test']:
        p=PROJECT/f'data/splits/{split}.parquet'
        frame=pd.read_parquet(p,columns=['user_id','timestamp','question_id','correct'])
        xes.append(dict(split=split,n_interactions=len(frame),n_users=frame['user_id'].nunique(),
                        columns=list(frame.columns),sha256=digest(p)))
    source_paths=[ARTICLE/'run_oulad_strict.py',ARTICLE/'run_xes3g5m_designs.py',ARTICLE/'run_xes3g5m_design_b2.py',
                  PROJECT/'models/uplift_oulad.py',PROJECT/'models/uplift_core.py',PROJECT/'models/pred_oulad.py']
    code={str(p):dict(sha256=digest(p),exists=p.exists()) for p in source_paths}
    result=dict(python=sys.executable,packages={m:importlib.metadata.version(m) for m in ['numpy','pandas','scikit-learn','xgboost','pyarrow']},
                oulad=dict(n_train=len(train),source_hashes=hashes,matches_archived_input_hashes=hashes==recorded,
                           demographics=demo,final_results={str(k):int(v) for k,v in train['final_result'].value_counts().items()},
                           retained_not_withdrawn=float((train.final_result!='Withdrawn').mean()),
                           retained_not_withdrawn_or_fail=float((~train.final_result.isin(['Withdrawn','Fail'])).mean())),
                xes=xes,source_code=code,
                missing_oulad_script_sources=[str(PROJECT/'scripts'/f'{name}.py') for name in ['uplift_evalue_robustness','uplift_pos_multiT']
                                              if not (PROJECT/'scripts'/f'{name}.py').exists()])
    (out/'data_inventory.json').write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding='utf-8')
    print(json.dumps(dict(n_oulad_train=len(train),oulad_input_hash_match=hashes==recorded,
                         final_results=result['oulad']['final_results'],xes=xes,
                         demographics=demo,missing_source_count=len(result['missing_oulad_script_sources'])),ensure_ascii=True,indent=2),flush=True)


if __name__=='__main__': main()
