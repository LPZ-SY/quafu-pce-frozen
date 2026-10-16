"""disco上的当天校准局部退极化实验；保持当前模型和统一精修不变。"""
import sys,os
from pathlib import Path
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'pce_v4_research'))
from final_objective_accelerated import *
HERE=Path(__file__).resolve().parent
from concurrent.futures import ProcessPoolExecutor,as_completed
import socket,csv,tarfile

def one(rho,u,q,n):
    t=rho.reshape((2,)*(2*n));t=np.moveaxis(np.tensordot(u,t,axes=(1,q)),0,q)
    t=np.moveaxis(np.tensordot(u.conj(),t,axes=(1,n+q)),0,n+q)
    return t.reshape(rho.shape)
def depolarize(rho,qs,p,n):
    if p<=0:return rho
    idx=np.arange(1<<n);mask=sum(1<<(n-1-q) for q in qs);base=idx[(idx&mask)==0]
    groups=[base+sum(((s>>j)&1)*(1<<(n-1-q)) for j,q in enumerate(qs)) for s in range(1<<len(qs))]
    partial=sum(rho[np.ix_(g,g)] for g in groups);out=(1-p)*rho
    for g in groups:out[np.ix_(g,g)]+=p*partial/len(groups)
    return out
def distributions(model,layout,noisy=True):
    n=model['nq'];th=np.array(model['theta']).reshape(model['layers'],n,2)
    rho=np.zeros((1<<n,1<<n),complex);rho[0,0]=1;idx=np.arange(1<<n)
    f1=np.array([float(r['fidelity']) for r in layout['qubits']]);f2=np.array([float(r['fidelity']) for r in layout['edges']])
    for layer in range(model['layers']):
        for q in range(n):
            a,z=th[layer,q];c,s=np.cos(a/2),np.sin(a/2)
            u=np.diag([np.exp(-.5j*z),np.exp(.5j*z)])@np.array([[c,-s],[s,c]])
            rho=one(rho,u,q,n)
            if noisy:rho=depolarize(rho,[q],2*(1-f1[q]),n)
        for q in range(layer%2,n-1,2):
            sign=1-2*(((idx>>(n-1-q))&1)*((idx>>(n-2-q))&1));rho=rho*sign[:,None]*sign[None,:]
            if noisy:rho=depolarize(rho,[q,q+1],4/3*(1-f2[q]),n)
    pp=[];h=np.array([[1,1],[1,-1]])/np.sqrt(2)
    for family in range(3):
        r=rho.copy()
        if family:
            for q in range(n):
                u=h@np.diag([1,-1j]) if family==2 else h;r=one(r,u,q,n)
                if noisy:r=depolarize(r,[q],2*(1-f1[q]),n)
        p=np.real(np.diag(r));assert np.min(p)>-1e-10 and abs(p.sum()-1)<1e-9
        pp.append(np.maximum(p,0)/p.sum())
    return np.stack(pp)
def readout(pp,f0,f1,n):
    out=pp.copy();idx=np.arange(1<<n)
    for q in range(n):
        bit=1<<(n-1-q);lo=idx[(idx&bit)==0];hi=lo|bit;a,b=out[:,lo].copy(),out[:,hi].copy()
        out[:,lo]=f0[q]*a+(1-f1[q])*b;out[:,hi]=(1-f0[q])*a+f1[q]*b
    return out
def job(args):
    chip,key=args;m=read(HERE/'manifest.json');model=m['models'][key];layout=m['devices'][chip]['layouts'][str(model['nq'])]
    path=HERE/'simulation'/chip/f'{key}.json'
    if path.exists():return
    S=setup(model['instance']);E=FinalEvaluator(S,'FINAL_JOINT',model['nq']);n=model['nq'];start=time.perf_counter()
    ideal=probabilities(np.array(model['theta']),n,model['layers']);pp=distributions(model,layout)
    e0=scores_from_moments(walsh(ideal),E.fams,E.masks);ref=E.decode(e0)
    scenarios={f'common_readout_{int(rate*100)}pct':readout(pp,np.full(n,1-rate),np.full(n,1-rate),n) for rate in (0,.02,.05)}
    g=np.array([float(r.get('readout g_fidelity') or 0) for r in layout['qubits']]);e=np.array([float(r.get('readout e_fidelity') or 0) for r in layout['qubits']])
    if np.all((g>0)&(e>0)):scenarios['published_today_readout']=readout(pp,g,e,n)
    results=[];raw=[]
    for scenario,p in scenarios.items():
        rng=np.random.default_rng(seed('chip_today_test',model['instance'],model['seed'],model['arm'],scenario))
        values,counts=sample(p,E.fams,E.masks,rng,batches=64,shots=12288);rows=[]
        for batch,val in enumerate(values):
            v=E.decode(val);_,a=E.hybrid(v)
            rows.append(dict(batch=batch,assignment=v.tolist(),customer_flip_rate=float(np.mean(v!=ref)),
                correlation_mae=float(np.mean(np.abs(val-e0))),correlation_cosine=float(np.dot(val,e0)/max(1e-15,np.linalg.norm(val)*np.linalg.norm(e0))),**a))
        results.append(dict(chip=chip,instance=model['instance'],seed=model['seed'],arm=model['arm'],scenario=scenario,
            raw_feasibility=float(np.mean([r['raw']['feasible'] for r in rows])),raw_overload=float(np.mean([r['raw']['overload'] for r in rows])),
            final_feasibility=float(np.mean([r['final']['feasible'] for r in rows])),final_distance=float(np.mean([r['final']['distance'] for r in rows if r['final']['feasible']])) if any(r['final']['feasible'] for r in rows) else None,
            changed_customers=float(np.mean([r['changed_customers'] for r in rows])),customer_flip_rate=float(np.mean([r['customer_flip_rate'] for r in rows])),
            correlation_mae=float(np.mean([r['correlation_mae'] for r in rows])),correlation_cosine=float(np.mean([r['correlation_cosine'] for r in rows]))))
        for row in rows:row.update(scenario=scenario,chip=chip,model=key)
        raw+=rows;dest=path.parent/f'{key}_{scenario}.npz';dest.parent.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(dest,counts=counts,scores=values,ideal_scores=e0,noisy_probabilities=p)
    save(path,dict(model=key,model_sha256=model['model_sha256'],results=results,rows=raw,seconds=time.perf_counter()-start,ledger=E.ledger()))
    print('芯片模型模拟完成',chip,key,round(time.perf_counter()-start,2),flush=True)
def main():
    assert socket.gethostname()=='lenovo';m=read(HERE/'manifest.json');start=time.perf_counter();verification=[]
    for name,sd in [('B-n31-k5',2001),('A-n45-k7',3001)]:
        key=f'{name}_{sd}_trained';model=m['models'][key];chip=next(c for c,d in m['devices'].items() if str(model['nq']) in d['layouts'])
        pp=distributions(model,m['devices'][chip]['layouts'][str(model['nq'])],False)
        error=float(np.max(np.abs(pp-probabilities(np.array(model['theta']),model['nq'],model['layers']))));assert error<1e-10
        verification.append(dict(instance=name,density_matrix_zero_noise_max_error=error))
    save(HERE/'simulation_verification.json',dict(passed=True,formal_instances=verification,host=socket.gethostname()))
    work=[(chip,key) for chip,d in m['devices'].items() for key,model in m['models'].items() if str(model['nq']) in d['layouts']]
    with ProcessPoolExecutor(max_workers=12) as pool:
        for f in as_completed([pool.submit(job,a) for a in work]):f.result()
    records=[r for p in (HERE/'simulation').glob('*/*.json') for r in read(p)['results']]
    summary=[]
    for chip,name,arm,scenario in sorted({(r['chip'],r['instance'],r['arm'],r['scenario']) for r in records}):
        rr=[r for r in records if (r['chip'],r['instance'],r['arm'],r['scenario'])==(chip,name,arm,scenario)]
        a=dict(chip=chip,instance=name,arm=arm,scenario=scenario,seeds=len(rr),batches_per_seed=64,shots_per_batch=12288)
        for key in ('raw_feasibility','raw_overload','final_feasibility','final_distance','changed_customers','customer_flip_rate','correlation_mae','correlation_cosine'):
            vv=[r[key] for r in rr if r[key] is not None];a[key]=float(np.mean(vv)) if vv else None
        summary.append(a)
    save(HERE/'simulation_summary.json',summary)
    with (HERE/'simulation_summary.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(summary[0]));w.writeheader();w.writerows(summary)
    save(HERE/'simulation_execution.json',dict(alias='disco',host=socket.gethostname(),complete=True,models=len(work),seconds=time.perf_counter()-start))
    with tarfile.open(HERE.parent/'pce_chip_selection_20261016_simulation.tar.gz','w:gz') as tar:
        for folder in ('simulation',):
            for p in (HERE/folder).rglob('*'):
                if p.is_file():tar.add(p,arcname=p.relative_to(HERE))
        for name in ('simulation_summary.json','simulation_summary.csv','simulation_execution.json','simulation_verification.json'):
            tar.add(HERE/name,arcname=name)
    print('芯片模拟全部完成',time.perf_counter()-start,flush=True)
if __name__=='__main__':main()
