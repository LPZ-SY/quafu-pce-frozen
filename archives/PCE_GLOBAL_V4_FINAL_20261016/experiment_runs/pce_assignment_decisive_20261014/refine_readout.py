"""针对第一轮失败方案，保持分配不变，比较线路深度并增大最小关联间隔。"""
import os
for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[name]='1'
import json
import multiprocessing as mp
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp,softmax
from train_assignment import Circuit,assess,problem
from run_decisive import characters
from hardware_assignment import prepare,simulate_qasm

HERE=Path(__file__).resolve().parent
OUT=HERE/'readout_refinement'


def job(args):
    source,layers=args
    cached=OUT/(source['id']+f'_margin_l{layers}.json')
    if cached.exists():return json.loads(cached.read_text())
    circ=Circuit(source['q'],layers)
    chi=characters(source['q'],source['mapping']).astype(float)
    signs=np.where(np.array(source['correlations'])<0,-1.,1.)
    best=None;calls=0
    rng=np.random.default_rng(71800+source['n']*10+source['seed']+layers)
    for restart in range(3):
        theta=np.array(source['theta']) if restart==0 and layers==source['layers'] else rng.uniform(-np.pi,np.pi,circ.q*layers)
        for beta in (2.,8.,30.,100.):
            def fun(x):
                nonlocal best,calls
                state,tape=circ.forward(x)
                corr=chi@(state*state)
                margin=signs*corr
                if best is None or margin.min()>best[0]:best=(float(margin.min()),x.copy(),corr.copy())
                value=logsumexp(-beta*margin)/beta
                d=-signs*softmax(-beta*margin)
                calls+=1
                return float(value),circ.adjoint(x,state,tape,d@chi)
            result=minimize(fun,theta,jac=True,method='L-BFGS-B',options={'maxiter':200,'ftol':1e-12,'gtol':1e-8})
            theta=result.x
    margin,theta,corr=best
    _,_,w=problem(source['n'],source['seed'])
    ev=assess(corr,source['instance'],w,source['model']!='redundant')
    result=dict(source)
    result.update(id=source['id']+f'_margin_l{layers}',layers=layers,theta=theta.tolist(),correlations=corr.tolist(),result=ev,
                  original_id=source['id'],original_margin=source['result']['min_margin'],original_result=source['result'],
                  preserves_assignment=ev['assignment']==source['result']['assignment'],target_signed_margin=margin,
                  refinement_calls=calls,parameters=len(theta),refinement='以已有量子优化分配为目标的监督表示训练；不是新求解成绩')
    (OUT/(result['id']+'.json')).write_text(json.dumps(result,ensure_ascii=False,indent=2))
    return result


def main():
    OUT.mkdir(exist_ok=True)
    manifest=json.loads((HERE/'hardware'/'manifest.json').read_text())
    analysis=json.loads((HERE/'hardware'/'analysis.json').read_text())
    failed={r['source_label'] for r in analysis['rows'] if not r['identical_assignment']}
    sources=[manifest['sources'][label] for label in sorted(failed)]
    jobs=[(s,l) for s in sources for l in (4,8)]
    rows=[]
    with mp.Pool(4) as pool:
        for r in pool.imap_unordered(job,jobs):
            rows.append(r)
            print(json.dumps(dict(id=r['id'],margin=r['target_signed_margin'],preserved=r['preserves_assignment'])),flush=True)
    (OUT/'all_runs.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2))
    # 同批重跑原线路，区分修正效果与设备随时间变化。
    (OUT/'hardware_sources.json').write_text(json.dumps(sources+sorted(rows,key=lambda r:r['id']),ensure_ascii=False,indent=2))


if __name__=='__main__':main()
