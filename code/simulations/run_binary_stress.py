"""Binary outcome cross-fit AIPW check. See BINARY_PROTOCOL_RU.md."""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import time
from datetime import datetime, timedelta, timezone
import numpy as np

ROOT=Path(__file__).resolve().parent
Z=1.959963984540054
SIGNS=np.array([1.,-1.,1.,-1.,1.,-1.])


def expit(x):
    return 1/(1+np.exp(-np.clip(x,-35,35)))


def logistic_fit(x,y):
    beta=np.zeros(x.shape[1])
    penalty=np.diag([1e-8]+[1e-6]*(x.shape[1]-1))
    for _ in range(40):
        prob=expit(x@beta)
        score=x.T@(y-prob)-penalty@beta
        information=x.T@(x*(prob*(1-prob))[:,None])+penalty
        step=np.linalg.solve(information,score)
        beta+=step
        if np.max(np.abs(step))<1e-8: break
    else: raise ArithmeticError('Logistic IRLS did not converge')
    return beta


def true_ate(alpha,load,nodes=100):
    z,w=np.polynomial.hermite.hermgauss(nodes)
    sd=np.sqrt(np.array(load)@np.array(load)+.4**2)
    eta=np.sqrt(2)*sd*z
    return float(w@(expit(eta+alpha)-expit(eta))/np.sqrt(np.pi))


def aipw(values,a,design,fold):
    n=len(a); q=values.shape[1]
    scores=np.empty((n,q)); outside=0
    for k in [0,1]:
        train=fold!=k; test=~train
        e=expit(design[test]@logistic_fit(design[train],a[train]))
        outside+=int(np.sum((e<.02)|(e>.98)))
        e=np.clip(e,.02,.98)
        mu=[]
        for treatment in [0,1]:
            selected=train&(a==treatment)
            fitted=np.column_stack([logistic_fit(design[selected],values[selected,j]) for j in range(q)])
            mu.append(expit(design[test]@fitted))
        aa=a[test,None]
        scores[test]=mu[1]-mu[0]+aa*(values[test]-mu[1])/e[:,None]-(1-aa)*(values[test]-mu[0])/(1-e[:,None])
    return scores.mean(axis=0),scores-scores.mean(axis=0),outside/n


def calibrate(d,inf,values,g):
    cov=np.cov(inf,rowvar=False,ddof=1)/len(inf)
    def ci(point,se): return point,point-Z*se,point+Z*se
    result={'uncalibrated':ci(d[0],np.sqrt(cov[0,0]))}
    gc=g-g.mean(); vc=values-values.mean(axis=0)
    corr_sign=np.sign(vc.T@gc)
    orient=corr_sign[1:]*corr_sign[0]
    for name,factor in [('manuscript_mom_raw',np.ones(6)),('sign_mom',orient)]:
        ctrl=d[1:]*factor
        h2=max(0.,ctrl.var(ddof=1)-np.mean(np.diag(cov)[1:]*factor**2))
        result[name]=ci(d[0]-ctrl.mean(),np.sqrt(cov[0,0]+h2))
    ic=inf[:,0]-(inf[:,1:]*orient).mean(axis=1)
    result['sign_covariance']=ci(result['sign_mom'][0],ic.std(ddof=1)/np.sqrt(len(ic)))
    return result


def log(*parts, **kwargs):
    stamp = datetime.now(timezone(timedelta(hours=5))).strftime('%H:%M')
    line = stamp + ' ' + ' '.join(map(str, parts))
    print(line, flush=True)
    with (ROOT/'binary_progress.log').open('a', encoding='utf-8') as stream:
        stream.write(line + '\n')


def main():
    p=argparse.ArgumentParser(); p.add_argument('--replications',type=int,default=500)
    p.add_argument('--seed',type=int,default=20261031); p.add_argument('--verify-only',action='store_true')
    args=p.parse_args()
    scenarios=[
        dict(name='matched',w=[1.,0.],ly=[.8,0.],loadings=[.8]*6),
        dict(name='weak',w=[1.,0.],ly=[.8,0.],loadings=[.15,.20,.25,.30,.35,.40]),
        dict(name='dominant',w=[1.,0.],ly=[.8,0.],loadings=[3.2,.8,.8,.8,.8,.8])]
    for s in scenarios:
        assert abs(true_ate(.5,s['ly'],60)-true_ate(.5,s['ly'],100))<1e-10
        assert true_ate(0,s['ly'])==0
    # Independent IRLS score check against likelihood first-order conditions.
    rng=np.random.default_rng(91); xx=np.column_stack((np.ones(2000),rng.normal(size=(2000,2))))
    yy=rng.binomial(1,expit(xx@np.array([.2,.5,-.3])))
    bb=logistic_fit(xx,yy)
    assert np.max(np.abs(xx.T@(yy-expit(xx@bb))))<1e-4
    log('Quadrature and logistic score checks passed',flush=True)
    if args.verify_only: return
    start=time.time(); out=ROOT/'binary_results'; out.mkdir(exist_ok=True)
    config=dict(seed=args.seed,n=2000,replications=args.replications,alphas=[0.,.5],scenarios=scenarios,
                numpy=np.__version__,crossfit_folds=2,propensity_bounds=[.02,.98],
                source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                protocol_sha256=hashlib.sha256((ROOT.parent/'effectiveness_validation'/'BINARY_PROTOCOL_RU.md').read_bytes()).hexdigest())
    (out/'config.json').write_text(json.dumps(config,indent=2),encoding='utf-8')
    summary=[]; raw=[]
    for s in scenarios:
        for alpha in [0.,.5]:
            truth=true_ate(alpha,s['ly']); rng=np.random.default_rng(args.seed)
            cell_start=time.perf_counter()
            log(s['name'], f'alpha={alpha:g}, CPU, seed={args.seed}, 0/{args.replications}')
            draws={}; trims=[]
            for rep in range(args.replications):
                n=2000; u=rng.normal(size=(n,2)); x=rng.normal(size=n)
                a=(u@np.array(s['w'])+.5*x+rng.normal(size=n)>0).astype(float)
                g=u[:,0]+rng.normal(size=n)
                eta=u@np.array(s['ly'])+.4*x
                y=rng.binomial(1,expit(alpha*a+eta))
                ec=(u[:,0,None]*np.array(s['loadings'])+.4*x[:,None])*SIGNS
                c=rng.binomial(1,expit(ec)); values=np.column_stack((y,c))
                fold=np.zeros(n,dtype=int); fold[rng.permutation(n)[:n//2]]=1
                design=np.column_stack((np.ones(n),x,g))
                d,inf,trim=aipw(values,a,design,fold); trims.append(trim)
                answers=calibrate(d,inf,values,g)
                for m,(est,lo,hi) in answers.items():
                    assert np.isfinite([est,lo,hi]).all() and lo<=hi
                    draws.setdefault(m,[]).append((est,lo,hi))
                    raw.append(dict(scenario=s['name'],alpha=alpha,true_ate=truth,replication=rep,method=m,estimate=est,lower=lo,upper=hi,propensity_clipped_fraction=trim))
                if (rep+1)%100==0:
                    rate=(rep+1)/(time.perf_counter()-cell_start)
                    log(s['name'], f'alpha={alpha:g}, {rep+1}/{args.replications}, {rate:.1f} rep/s, ETA {(args.replications-rep-1)/rate:.1f}s')
            base=np.array(draws['uncalibrated']); be=(base[:,0]-truth)**2
            bw=np.mean(base[:,2]-base[:,1])
            for m,entries in draws.items():
                arr=np.array(entries); err=arr[:,0]-truth
                cv=np.mean((arr[:,1]<=truth)&(truth<=arr[:,2])); rej=np.mean((arr[:,1]>0)|(arr[:,2]<0))
                width=np.mean(arr[:,2]-arr[:,1]); imp=be-err**2; seimp=imp.std(ddof=1)/np.sqrt(args.replications)
                row=dict(scenario=s['name'],alpha=alpha,true_ate=truth,method=m,n=2000,replications=args.replications,
                         bias=err.mean(),bias_mcse=err.std(ddof=1)/np.sqrt(args.replications),rmse=np.sqrt(np.mean(err**2)),
                         coverage=cv,coverage_mcse=np.sqrt(cv*(1-cv)/args.replications),rejection_rate=rej,
                         rejection_mcse=np.sqrt(rej*(1-rej)/args.replications),rejection_interpretation='type_I_error' if alpha==0 else 'power',
                         mean_width=width,width_ratio=width/bw,mse_improvement=imp.mean(),mse_improvement_mcse=seimp,
                         mse_improvement_lower95=imp.mean()-Z*seimp,propensity_clipped_fraction=np.mean(trims))
                row['conditional_success']=bool(row['mse_improvement_lower95']>0 and .93<=cv<=.97 and width<=2*bw)
                summary.append(row)
            log(s['name'],f'alpha={alpha:g}, ATE={truth:.5f}',f'sign bias={summary[-2]["bias"]:+.4f}, coverage={summary[-2]["coverage"]:.3f}',flush=True)
    for name,rows in [('summary',summary),('replications',raw)]:
        with (out/f'{name}.csv').open('w',encoding='utf-8',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    manifest=dict(elapsed_seconds=round(time.time()-start,2),summary_rows=len(summary),replication_rows=len(raw),verification='passed',
                  sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in out.glob('*.csv')})
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    log(json.dumps(manifest,indent=2),flush=True)


if __name__=='__main__': main()
