"""直接分配目标的关联期望值训练；无路线扰动、无容量修复、无参考解输入。"""
import os
for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[name]='1'
import argparse
import itertools
import json
import multiprocessing as mp
from pathlib import Path
import time
import numpy as np
from scipy.optimize import minimize
from run_decisive import nqubits, characters, make_instance, all_route_costs, save

HERE=Path(__file__).resolve().parent
OUT=HERE/'training'


class Circuit:
    """实振幅线路：每层单比特 Y 旋转，交替相邻受控 Z 门，末层仅旋转。"""
    def __init__(self,q,layers):
        self.q,self.layers=q,layers
        self.size=2**q
        self.pairs=[]
        for bit in range(q):
            low=np.array([s for s in range(self.size) if not(s>>bit&1)])
            self.pairs.append((low,low|(1<<bit)))
        self.cz={}
        s=np.arange(self.size)
        for b in range(q-1):
            self.cz[b]=1-2*((s>>b&1)*(s>>(b+1)&1))

    def rotate(self,state,bit,theta):
        low,high=self.pairs[bit]
        c,s=np.cos(theta/2),np.sin(theta/2)
        result=state.copy()
        result[low]=c*state[low]-s*state[high]
        result[high]=s*state[low]+c*state[high]
        return result

    def forward(self,theta):
        state=np.zeros(self.size)
        state[0]=1
        history=[]
        index=0
        for layer in range(self.layers):
            for bit in range(self.q):
                history.append(('ry',bit,index,state))
                state=self.rotate(state,bit,theta[index])
                index+=1
            if layer<self.layers-1:
                for bit in range(layer%2,self.q-1,2):
                    history.append(('cz',bit,None,None))
                    state=state*self.cz[bit]
        return state,history

    def adjoint(self,theta,state,history,diagonal):
        lam=diagonal*state
        gradient=np.zeros_like(theta)
        for kind,bit,index,before in reversed(history):
            if kind=='cz':
                lam=lam*self.cz[bit]
            else:
                low,high=self.pairs[bit]
                c,s=np.cos(theta[index]/2),np.sin(theta[index]/2)
                derivative=np.empty_like(before)
                derivative[low]=(-s*before[low]-c*before[high])/2
                derivative[high]=(c*before[low]-s*before[high])/2
                gradient[index]=2*np.dot(lam,derivative)
                lam=self.rotate(lam,bit,-theta[index])
        return gradient

    def qasm(self,theta):
        lines=['OPENQASM 2.0;','include "qelib1.inc";',f'qreg q[{self.q}];',f'creg c[{self.q}];']
        for layer in range(self.layers):
            for bit in range(self.q): lines.append(f'ry({theta[layer*self.q+bit]:.17g}) q[{bit}];')
            if layer<self.layers-1:
                for bit in range(layer%2,self.q-1,2): lines.append(f'cz q[{bit}],q[{bit+1}];')
        lines.extend(f'measure q[{bit}] -> c[{bit}];' for bit in range(self.q))
        return '\n'.join(lines)+'\n'


def problem(n,seed):
    inst=make_instance(n,seed,1.15)
    xy=np.array(inst['xy'])
    distance=np.linalg.norm(xy[:,None]-xy[None,:],axis=2)
    weight=distance[1:,1:]/(n*distance[1:,1:].mean())
    return inst,distance,weight


def loss(corr,alpha,inst,weight,complement):
    n=inst['n']
    demand=np.array(inst['demand'])
    cap=np.array(inst['capacities'])
    t=np.tanh(alpha*corr)
    values=(1-t)/2
    a=np.stack([values,1-values],axis=1) if complement else values.reshape(n,2)
    gradient=weight@a
    objective=.5*np.sum(a*gradient)
    loads=demand@a
    excess=np.maximum(loads-cap,0)
    scale=demand.mean()
    objective+=100*np.sum((excess/scale)**2)
    gradient+=200/scale**2*demand[:,None]*excess[None,:]
    unique=a.sum(axis=1)-1
    objective+=100*np.sum(unique**2)
    gradient+=200*unique[:,None]
    objective+=.02*np.sum(a*(1-a))
    gradient+=.02*(1-2*a)
    da=gradient[:,0]-gradient[:,1] if complement else gradient.ravel()
    return float(objective), da*(-alpha/2)*(1-t*t)


def assess(corr,inst,weight,complement):
    bits=(corr<0).astype(int)
    a=np.stack([bits,1-bits],axis=1) if complement else bits.reshape(inst['n'],2)
    demand=np.array(inst['demand'])
    loads=demand@a
    unique=int(np.sum((a.sum(axis=1)-1)**2))
    excess=int(np.maximum(loads-np.array(inst['capacities']),0).sum())
    return dict(feasible=bool(unique==0 and excess==0 and np.all(corr!=0)),
        unique_penalty=unique,overload=excess,loads=loads.tolist(),assignment=a.tolist(),
        proxy=float(.5*np.sum(a*(weight@a))),min_margin=float(np.min(np.abs(corr))))


def one_job(job):
    n,seed,model,restart,layers=job
    ident=f'n{n}_s{seed}_{model}_r{restart}_l{layers}'
    target=OUT/(ident+'.json')
    if target.exists(): return json.loads(target.read_text())
    start=time.time()
    inst,distance,weight=problem(n,seed)
    complement=model!='redundant'
    m=n if complement else 2*n
    q=nqubits(m)
    mapping=list(itertools.combinations(range(q),2))[:m]
    chi=characters(q,mapping).astype(float)
    circ=Circuit(q,layers)
    rng=np.random.default_rng(seed*1000+restart*17+layers)
    classical=model=='classical'
    theta=rng.normal(0,.3,m) if classical else rng.uniform(-np.pi,np.pi,q*layers)
    best=None
    calls=0
    def consider(x,corr):
        nonlocal best
        ev=assess(corr,inst,weight,complement)
        key=(ev['unique_penalty']+ev['overload'],round(ev['proxy'],10),-ev['min_margin'])
        if best is None or key<best[0]: best=(key,x.copy(),corr.copy(),ev)
    history=[]
    for alpha in (2.,5.,10.,20.,40.):
        def fun(x):
            nonlocal calls
            if classical:
                corr=np.tanh(x)
                value,derivative=loss(corr,alpha,inst,weight,complement)
                gradient=derivative*(1-corr*corr)
            else:
                state,tape=circ.forward(x)
                corr=chi@(state*state)
                value,derivative=loss(corr,alpha,inst,weight,complement)
                gradient=circ.adjoint(x,state,tape,derivative@chi)
            calls+=1
            # 所有方法以同一离散分配目标记录训练沿途的最佳可行解，路线距离不参与选参。
            consider(x,corr)
            return value,gradient
        result=minimize(fun,theta,jac=True,method='L-BFGS-B',options={'maxiter':100,'ftol':1e-11,'gtol':1e-7,'maxls':30})
        theta=result.x
        history.append(dict(alpha=alpha,loss=float(result.fun),iterations=int(result.nit),success=bool(result.success)))
    _,best_theta,corr,ev=best
    result=dict(id=ident,n=n,seed=seed,model=model,restart=restart,layers=layers,
        q=0 if classical else q,correlators=0 if classical else m,
        parameters=len(best_theta),instance=inst,mapping=mapping,theta=best_theta.tolist(),correlations=corr.tolist(),
        result=ev,calls=calls,history=history,seconds=time.time()-start,
        selection='只按离散分配目标与关联间隔选参；不输入最优路线或参考分配；不修复容量')
    if not classical:
        (OUT/(ident+'.qasm')).write_text(circ.qasm(best_theta))
    target.write_text(json.dumps(result,ensure_ascii=False,indent=2))
    return result


def verify_gradients():
    rng=np.random.default_rng(78)
    inst,_,w=problem(10,100)
    errors=[]
    for complement in (True,False):
        m=10 if complement else 20
        q=nqubits(m)
        circ=Circuit(q,4)
        mapping=list(itertools.combinations(range(q),2))[:m]
        chi=characters(q,mapping).astype(float)
        theta=rng.uniform(-2,2,q*4)
        state,tape=circ.forward(theta)
        value,d=loss(chi@(state*state),5.,inst,w,complement)
        gradient=circ.adjoint(theta,state,tape,d@chi)
        numerical=[]
        for i in range(len(theta)):
            delta=np.zeros_like(theta);delta[i]=1e-6
            plus=circ.forward(theta+delta)[0];minus=circ.forward(theta-delta)[0]
            numerical.append((loss(chi@(plus*plus),5.,inst,w,complement)[0]-loss(chi@(minus*minus),5.,inst,w,complement)[0])/2e-6)
        errors.append(float(np.max(np.abs(gradient-numerical))))
        assert errors[-1]<1e-5
        assert abs(np.sum(state*state)-1)<1e-12
    (OUT/'gradient_verification.json').write_text(json.dumps(dict(max_absolute_errors=errors)))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--workers',type=int,default=4)
    args=parser.parse_args()
    OUT.mkdir(exist_ok=True)
    verify_gradients()
    jobs=[]
    for n,seeds in ((10,range(100,106)),(16,range(200,204)),(24,range(300,303)),(40,range(400,403)),(60,range(500,503))):
        for seed in seeds:
            for model in (('redundant','complement','classical') if n==10 else ('complement','classical')):
                for restart in range(2):
                    layers=4 if restart==0 else 8
                    jobs.append((n,seed,model,restart,layers))
    rows=[]
    with mp.Pool(args.workers) as pool:
        for result in pool.imap_unordered(one_job,jobs):
            rows.append(result)
            print(json.dumps(dict(completed=len(rows),total=len(jobs),id=result['id'],feasible=result['result']['feasible'],margin=result['result']['min_margin'])),flush=True)
    (OUT/'all_runs.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
