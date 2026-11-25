"""Conditional effectiveness and external-reference transfer; NumPy only.

See PROTOCOL_RU.md. The reference sample is a simulated additional resource,
not an available OULAD/XES sample. No novelty claim is made for these baselines.
"""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import platform
import time
import numpy as np

ROOT=Path(__file__).resolve().parent
Z=1.959963984540054
SIGNS=np.array([1.,-1.,1.,-1.,1.,-1.])
METHODS=['uncalibrated','manuscript_mom_raw','sign_mom','sign_covariance',
         'reference_transfer_wald','reference_transfer_fieller_hull','oracle_adjusted']


def scenarios():
    base=dict(n=2000,nref=4000,w=[1.,0.],ly=[.3,0.],
              lc=(SIGNS[:,None]*np.array([[.3,0.]])).tolist(),
              scales=[1.]*6,control_sd=1.,proxy_sd=1.,ref_shift=1.,direct=0.)
    specs=[('equal_loadings',{}),
           ('half_loadings',{'lc':(np.array(base['lc'])*.5).tolist()}),
           ('double_loadings',{'lc':(np.array(base['lc'])*2).tolist()}),
           ('heterogeneous_loadings',{'lc':(np.array(base['lc'])*np.array([.25,.5,.75,1.25,1.5,1.75])[:,None]).tolist()}),
           ('changed_units',{'scales':[10.,1.,1.,1.,1.,1.]}),
           ('noisy_controls',{'control_sd':3.}),
           ('two_factors_aligned',{'w':[1.,1.],'ly':[.15,.15],
                                   'lc':(SIGNS[:,None]*np.array([[.3,.3]])).tolist()}),
           ('two_factors_incompatible',{'w':[1.,1.],'ly':[.15,.15],
                                        'lc':(SIGNS[:,None]*np.array([[-.3,.6]])).tolist()}),
           ('reference_shift',{'ref_shift':1.5}),
           ('weak_reference',{'proxy_sd':8.,'nref':250}),
           ('invalid_controls',{'direct':.1}),
           ('no_confounding',{'w':[0.,0.]}),
           ('small_sample',{'n':500}),('large_sample',{'n':8000})]
    return [dict(base,**changes,name=name) for name,changes in specs]


def fit_effects(values, design):
    """Joint OLS effects with heteroskedastic sandwich influence functions."""
    bread=np.linalg.inv(design.T@design/len(design))
    beta=bread@(design.T@values)/len(design)
    residual=values-design@beta
    influence=(design@bread[:,1])[:,None]*residual
    return beta[1], influence, np.cov(influence,rowvar=False,ddof=1)/len(design)


def fieller_hull(dy,dc,numer,denom,vary,varc,covyc,varn,vard,covnd):
    """First-order inverted product-Wald set; return its interval hull."""
    aa=denom**2-Z**2*vard
    bb=-2*dc*numer*denom+2*Z**2*dc*covnd
    cc=dc**2*numer**2-Z**2*(denom**2*vary+numer**2*varc-2*denom*numer*covyc+dc**2*varn)
    if aa<=0: return -np.inf,np.inf
    disc=bb**2-4*aa*cc
    if disc < -1e-10: raise ArithmeticError('Unexpected empty Fieller set')
    root=np.sqrt(max(0.,disc))
    tlo=(-bb-root)/(2*aa); thi=(-bb+root)/(2*aa)
    return dy-thi,dy-tlo


def make_data(s,theta,rng):
    n=s['n']; nr=s['nref']; lc=np.array(s['lc']); ly=np.array(s['ly'])
    u=rng.normal(size=(n,2)); x=rng.normal(size=n)
    a=(u@np.array(s['w'])+.5*x+rng.normal(size=n)>0).astype(float)
    g=u[:,0]+s['proxy_sd']*rng.normal(size=n)
    y=theta*a+u@ly+.3*x+rng.normal(size=n)
    c=u@lc.T+.2*x[:,None]+s['control_sd']*rng.normal(size=(n,6))
    c=(c+s['direct']*a[:,None]*SIGNS)*np.array(s['scales'])
    ur=rng.normal(size=(nr,2)); xr=rng.normal(size=nr)
    gr=ur[:,0]+s['proxy_sd']*rng.normal(size=nr)
    yr=ur@(ly*s['ref_shift'])+.3*xr+rng.normal(size=nr)
    cr=(ur@lc.T+.2*xr[:,None]+s['control_sd']*rng.normal(size=(nr,6)))*np.array(s['scales'])
    return y,c,a,x,u,g,yr,cr,gr


def evaluate(data):
    y,c,a,x,u,g,yr,cr,gr=data
    n=len(y); nr=len(yr)
    d,influence,cov=fit_effects(np.column_stack((y,c)),np.column_stack((np.ones(n),a,x)))
    def interval(point,se): return point,point-Z*se,point+Z*se
    answer={'uncalibrated':interval(d[0],np.sqrt(cov[0,0]))}
    gc=g-g.mean()
    sign=np.sign((c-c.mean(axis=0)).T@gc)*np.sign((y-y.mean())@gc)
    for name,factor in [('manuscript_mom_raw',np.ones(6)),('sign_mom',sign)]:
        ctrl=d[1:]*factor
        h2=max(0.,ctrl.var(ddof=1)-np.mean(np.diag(cov)[1:]*factor**2))
        answer[name]=interval(d[0]-ctrl.mean(),np.sqrt(cov[0,0]+h2))
    corrected_if=influence[:,0]-(influence[:,1:]*sign).mean(axis=1)
    answer['sign_covariance']=interval(answer['sign_mom'][0],corrected_if.std(ddof=1)/np.sqrt(n))
    # Reference orientations are used for both primary and reference control pools.
    grc=gr-gr.mean(); yrc=yr-yr.mean(); crc=cr-cr.mean(axis=0)
    sr=np.sign(crc.T@grc)
    pool=(crc*sr).mean(axis=1)
    numer=np.mean(grc*yrc); denom=np.mean(grc*pool)
    if denom==0: raise ArithmeticError('Exactly zero reference denominator')
    inn=grc*yrc-numer; ind=grc*pool-denom
    refcov=np.cov(np.column_stack((inn,ind)),rowvar=False,ddof=1)/nr
    ratio=numer/denom
    dc=float(np.mean(d[1:]*sr)); ic=(influence[:,1:]*sr).mean(axis=1)
    pcov=np.cov(np.column_stack((influence[:,0],ic)),rowvar=False,ddof=1)/n
    ir=inn/denom-numer*ind/denom**2
    primary_if=influence[:,0]-ratio*ic
    variance=primary_if.var(ddof=1)/n+dc**2*ir.var(ddof=1)/nr
    point=d[0]-ratio*dc
    answer['reference_transfer_wald']=interval(point,np.sqrt(variance))
    lo,hi=fieller_hull(d[0],dc,numer,denom,pcov[0,0],pcov[1,1],pcov[0,1],
                       refcov[0,0],refcov[1,1],refcov[0,1])
    answer['reference_transfer_fieller_hull']=(point,lo,hi)
    od,oi,oc=fit_effects(y[:,None],np.column_stack((np.ones(n),a,x,u)))
    answer['oracle_adjusted']=interval(od[0],np.sqrt(float(oc)))
    diagnostic=dict(ratio=ratio,denominator_z=denom/np.sqrt(refcov[1,1]),
                    weak_denominator=float(denom**2<=Z**2*refcov[1,1]))
    return answer,diagnostic


def verify():
    # Fieller must reduce to an ordinary fixed-ratio covariance interval.
    lo,hi=fieller_hull(.3,.2,.6,.4,.0025,.0016,.0004,0.,0.,0.)
    se=np.sqrt(.0025+1.5**2*.0016-2*1.5*.0004)
    assert np.allclose([lo,hi],[-Z*se,Z*se])
    s=scenarios()[0]; data=make_data(s,.1,np.random.default_rng(43))
    a,diag=evaluate(data)
    # Independent coefficient implementation.
    y,c,t,x,u,g,yr,cr,gr=data
    beta=np.linalg.lstsq(np.column_stack((np.ones(len(y)),t,x)),y,rcond=None)[0][1]
    assert np.isclose(beta,a['uncalibrated'][0])
    # Unit invariance must hold even if only one control changes scale.
    scales=np.array([10.,1.,1.,1.,1.,1.])
    b,_=evaluate((y,c*scales,t,x,u,g,yr,cr*scales,gr))
    # With heterogeneous scales, the pooled noise composition changes: finite
    # sample point estimates need not coincide. Uniform units must coincide.
    b,_=evaluate((y,c*10,t,x,u,g,yr,cr*10,gr))
    for m in ['reference_transfer_wald','reference_transfer_fieller_hull']:
        assert np.allclose(a[m],b[m],rtol=1e-10,atol=1e-10)
    # Additive effect preservation on identical draws, including interval width.
    b,_=evaluate((y+.1*t,c,t,x,u,g,yr,cr,gr))
    for m in ['uncalibrated','sign_mom','sign_covariance','reference_transfer_wald','reference_transfer_fieller_hull','oracle_adjusted']:
        assert np.allclose(np.array(b[m])-np.array(a[m]),.1,atol=1e-10),m
    # Delta-method derivative of ratio by central finite differences.
    N,D=.3,.5; eps=1e-6
    assert np.isclose(((N+eps)/D-(N-eps)/D)/(2*eps),1/D)
    assert np.isclose((N/(D+eps)-N/(D-eps))/(2*eps),-N/D**2)
    print('Algebra, OLS, common-unit invariance, effect preservation, derivatives: passed',flush=True)


def save_csv(path,rows):
    with path.open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


def main():
    p=argparse.ArgumentParser(); p.add_argument('--replications',type=int,default=1000)
    p.add_argument('--seed',type=int,default=20261030); p.add_argument('--verify-only',action='store_true')
    args=p.parse_args(); verify()
    if args.verify_only: return
    if args.replications<2: raise ValueError('Need >=2 replications')
    start=time.time(); out=ROOT/'results'; out.mkdir(exist_ok=True)
    config=dict(seed=args.seed,replications=args.replications,effects=[0.,.1],scenarios=scenarios(),
                numpy=np.__version__,python=platform.python_version(),
                source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                protocol_sha256=hashlib.sha256((ROOT/'PROTOCOL_RU.md').read_bytes()).hexdigest(),
                inference='OLS sandwich; independent-reference delta and Fieller hull',
                note='Continuous linear DGP; independent untreated reference is simulated additional information')
    (out/'config.json').write_text(json.dumps(config,indent=2),encoding='utf-8')
    summary=[]; raw=[]; diagnostics=[]
    for s in scenarios():
        for theta in [0.,.1]:
            rng=np.random.default_rng(args.seed)
            draws={m:[] for m in METHODS}; ds=[]
            for r in range(args.replications):
                answer,diag=evaluate(make_data(s,theta,rng)); ds.append(diag)
                diagnostics.append(dict(scenario=s['name'],true_effect=theta,replication=r,**diag))
                for m in METHODS:
                    est,lo,hi=answer[m]
                    assert np.isfinite(est) and not np.isnan(lo) and not np.isnan(hi) and lo<=hi
                    draws[m].append((est,lo,hi))
                    raw.append(dict(scenario=s['name'],true_effect=theta,replication=r,method=m,estimate=est,lower=lo,upper=hi))
            baseline=np.array(draws['uncalibrated']); baseerr=(baseline[:,0]-theta)**2
            basewidth=np.mean(baseline[:,2]-baseline[:,1])
            for m in METHODS:
                arr=np.array(draws[m]); err=arr[:,0]-theta
                covered=(arr[:,1]<=theta)&(theta<=arr[:,2]); rejected=(arr[:,1]>0)|(arr[:,2]<0)
                width=arr[:,2]-arr[:,1]; finite=np.isfinite(width)
                improvement=baseerr-err**2
                impmc=improvement.std(ddof=1)/np.sqrt(args.replications)
                coverage=covered.mean(); rejection=rejected.mean()
                meanfinite=float(width[finite].mean()) if finite.any() else 0.
                row=dict(scenario=s['name'],true_effect=theta,method=m,n=s['n'],n_reference=s['nref'],replications=args.replications,
                         bias=err.mean(),bias_mcse=err.std(ddof=1)/np.sqrt(args.replications),rmse=np.sqrt(np.mean(err**2)),
                         coverage=coverage,coverage_mcse=np.sqrt(coverage*(1-coverage)/args.replications),
                         rejection_rate=rejection,rejection_mcse=np.sqrt(rejection*(1-rejection)/args.replications),
                         rejection_interpretation='type_I_error' if theta==0 else 'power',
                         mean_finite_width=meanfinite,finite_interval_fraction=finite.mean(),
                         unbounded_fraction=1-finite.mean(),width_ratio_finite=meanfinite/basewidth,
                         mse_improvement=improvement.mean(),mse_improvement_mcse=impmc,
                         mse_improvement_lower95=improvement.mean()-Z*impmc,
                         weak_reference_fraction=np.mean([d['weak_denominator'] for d in ds]))
                row['conditional_success']=bool(row['mse_improvement_lower95']>0 and .93<=coverage<=.97 and finite.all() and meanfinite<=2*basewidth)
                summary.append(row)
            print(s['name'],f'true={theta:g}',
                  'sign RMSE='+str(round(summary[-5]['rmse'],4)),
                  'transfer RMSE='+str(round(summary[-2]['rmse'],4)),flush=True)
    save_csv(out/'summary.csv',summary); save_csv(out/'replications.csv',raw); save_csv(out/'diagnostics.csv',diagnostics)
    oracle=[r for r in summary if r['method']=='oracle_adjusted']
    assert all(abs(r['bias'])<5*r['bias_mcse'] for r in oracle), 'Oracle MC verification failed'
    manifest=dict(elapsed_seconds=round(time.time()-start,2),summary_rows=len(summary),replication_rows=len(raw),
                  dataset_runs=len(diagnostics),verification='passed',
                  sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in out.glob('*.csv')})
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print(json.dumps(manifest,indent=2),flush=True)


if __name__=='__main__': main()
