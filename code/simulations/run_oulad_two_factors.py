"""Semi-synthetic OULAD sensitivity analysis with known binary potential outcomes."""
import csv
import hashlib
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from run_binary_stress import aipw, calibrate, expit

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from semisynth_positive import DATA, KEYS

SEED = 20261117
REPS = 500
N = 2000
SIGNS = np.array([1, 1, -1, -1, -1, -1])
TARGET = np.array([.85, .65])
PANELS = {
    "matched": [TARGET.tolist()] * 6,
    "unequal_proportional": (np.array([.25,.5,.75,1.25,1.5,1.75])[:,None]*TARGET).tolist(),
    "unequal_mixed": [[.2,.1],[.4,-.4],[.6,.05],[1.,-.8],[1.25,.15],[1.5,-1.]],
}

def run():
    out = HERE / "oulad_two_factors"
    out.mkdir(exist_ok=True)
    def log(message):
        stamp = datetime.now(timezone(timedelta(hours=5))).strftime("%H:%M")
        line = f"{stamp} {message}"
        print(line, flush=True)
        with (out/"progress.log").open("a", encoding="utf-8") as stream:
            stream.write(line+"\n")

    start = perf_counter()
    log(f"Loading OULAD, CPU, seed={SEED}, {REPS} repetitions per cell")
    targets = pd.read_parquet(DATA/"student_target.parquet")
    daily = pd.read_parquet(DATA/"daily_vle.parquet", columns=KEYS+["date","sum_click"])
    targets = targets.sort_values(KEYS).drop_duplicates("id_student")
    linked = daily.merge(targets[KEYS+["first_deadline"]], on=KEYS, validate="many_to_one")
    clicks = linked.loc[linked.date < linked.first_deadline].groupby(KEYS).sum_click.sum()
    targets = targets.join(clicks.rename("clicks"),on=KEYS)
    raw = np.log1p(targets.clicks.fillna(0).to_numpy(float))
    engagement = (raw-raw.mean())/raw.std()
    x = targets[["module_presentation_length","date_registration"]].astype(float)
    x = x.fillna(x.median())
    x = (x-x.mean())/x.std(ddof=0)
    x = np.column_stack([x, pd.get_dummies(targets.code_module,drop_first=True,dtype=float)])
    assert np.isfinite(x).all() and np.isfinite(engagement).all()
    assert targets.id_student.is_unique and len(targets)>=N
    log(f"Prepared {len(targets)} unique learners")

    def digest(path):
        sha = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024*1024), b""):
                sha.update(block)
        return sha.hexdigest()

    config = dict(seed=SEED, n=N, pool=len(targets), replications=REPS, panels=PANELS,
                  numpy=np.__version__, pandas=pd.__version__, target=TARGET.tolist(),
                  source_sha256={p.name:digest(p) for p in [Path(__file__),HERE/"run_binary_stress.py"]},
                  data_sha256={name:digest(DATA/name) for name in ["student_target.parquet","daily_vle.parquet"]})
    config_text = json.dumps(config, indent=2)
    config_path = out/"config.json"
    if config_path.exists() and config_path.read_text(encoding="utf-8") != config_text:
        raise ValueError("Existing run configuration differs")
    config_path.write_text(config_text,encoding="utf-8")
    all_rows = []
    for panel, loads in PANELS.items():
        checkpoint = out/f"{panel}.csv"
        if checkpoint.exists():
            with checkpoint.open() as stream:
                all_rows.extend(csv.DictReader(stream))
            log(f"Reused completed panel {panel}")
            continue
        rows = []
        cell_start = perf_counter()
        for rep in range(REPS):
            rng = np.random.default_rng(np.random.SeedSequence([SEED,rep]))
            ix = rng.choice(len(x), N, replace=False)
            u = np.column_stack([engagement[ix],rng.standard_normal(N)])
            measured = x[ix]
            base = .25*measured[:,0]-.20*measured[:,1]
            treatment = rng.binomial(1,expit(.9*u[:,0]+.8*u[:,1]+base))
            proxy = u[:,0]+rng.standard_normal(N)
            controls = rng.binomial(1,expit((u@np.array(loads).T+base[:,None])*SIGNS))
            fold = rng.permutation(np.arange(N)%2)
            design = np.column_stack([np.ones(N),measured,proxy])
            outcome_draw = rng.random(N)
            eta = -.3+u@TARGET+base
            for alpha in (0.,.55):
                true_effect = (expit(eta+alpha)-expit(eta)).mean()
                outcome = outcome_draw<expit(eta+alpha*treatment)
                values = np.column_stack([outcome,controls])
                estimates, scores, _ = aipw(values,treatment,design,fold)
                for method,(point,lower,upper) in calibrate(estimates,scores,values,proxy).items():
                    assert np.isfinite([point,lower,upper]).all() and lower<=upper
                    rows.append(dict(panel=panel,alpha=alpha,rep=rep,method=method,
                                     truth=true_effect,estimate=point,lower=lower,upper=upper))
            if (rep+1)%50==0:
                rate=(rep+1)/(perf_counter()-cell_start)
                log(f"{panel} {rep+1}/{REPS} paired repetitions, {rate:.1f}/s, panel ETA {(REPS-rep-1)/rate:.0f}s")
        temp = checkpoint.with_suffix(".tmp")
        pd.DataFrame(rows).to_csv(temp,index=False)
        temp.replace(checkpoint)
        all_rows.extend(rows)
    data = pd.DataFrame(all_rows).apply(pd.to_numeric,errors="ignore")
    assert len(data)==len(PANELS)*2*REPS*4 and not data.duplicated(["panel","alpha","rep","method"]).any()
    data["bias"] = data.estimate-data.truth
    data["coverage"] = (data.lower<=data.truth)&(data.truth<=data.upper)
    data["width"] = data.upper-data.lower
    data["reject_zero"] = (data.lower>0)|(data.upper<0)
    assert (data.loc[data.alpha==0,"coverage"] != data.loc[data.alpha==0,"reject_zero"]).all()
    summary = data.groupby(["panel","alpha","method"],sort=False).agg(
        truth=("truth","mean"), bias=("bias","mean"), bias_sd=("bias","std"),
        coverage=("coverage","mean"), width=("width","mean"), reject_zero=("reject_zero","mean"))
    summary["bias_mcse"] = summary.pop("bias_sd")/np.sqrt(REPS)
    summary["coverage_mcse"] = np.sqrt(summary.coverage*(1-summary.coverage)/REPS)
    summary.to_csv(out/"summary.csv")
    log(f"Completed in {perf_counter()-start:.1f}s")

if __name__ == "__main__":
    with threadpool_limits(limits=1):
        run()
