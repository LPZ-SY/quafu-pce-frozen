"""固定8比特4层，真实路线目标，量子独立训练与随机预算阶梯。"""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import sys,json,time,zlib,hashlib
from pathlib import Path
import numpy as np
from numba import njit,types
from numba.typed import Dict
HERE=Path(__file__).resolve().parent
ROOT=HERE.parent.parent
sys.path.insert(0,str(ROOT/'outputs/cvrp_three_way_20261014'))
import compare as C
sys.path.insert(0,str(HERE.parent/'pce_dongling_direct4_20261014'))
import pce_k as K
from scale_pairs import route_cost
NQ,LAYERS,STEPS,REPEATS=8,4,20000,32
BUDGETS=[1,4,16,64,256,1024,4096]

def save(name,data):
    p=HERE/name;t=p.with_suffix('.tmp');t.write_text(json.dumps(data,ensure_ascii=False,indent=1),encoding='utf-8');t.replace(p)

@njit(cache=True)
def expectations(theta,fams,masks):
    psi=np.zeros(256,np.complex128);psi[0]=1
    for layer in range(4):
        for q in range(8):
            bit=1<<(7-q);j=2*(layer*8+q)
            c,s=np.cos(theta[j]/2),np.sin(theta[j]/2)
            z0=np.exp(-.5j*theta[j+1]);z1=np.exp(.5j*theta[j+1])
            for base in range(0,256,2*bit):
                for off in range(bit):
                    a=base+off;b=a+bit;x,y=psi[a],psi[b]
                    psi[a]=z0*(c*x-s*y);psi[b]=z1*(s*x+c*y)
        for q in range(layer%2,7,2):
            a=1<<(7-q);b=1<<(6-q)
            for j in range(256):
                if j&a and j&b:psi[j]=-psi[j]
    out=np.zeros(len(masks))
    for fam in range(3):
        phi=psi.copy()
        if fam:
            for q in range(8):
                bit=1<<(7-q)
                for base in range(0,256,2*bit):
                    for off in range(bit):
                        a=base+off;b=a+bit;x,y=phi[a],phi[b]
                        if fam==2:y=-1j*y
                        phi[a]=(x+y)/np.sqrt(2);phi[b]=(x-y)/np.sqrt(2)
        parity=np.abs(phi)**2
        stride=1
        while stride<256:
            for base in range(0,256,2*stride):
                for off in range(stride):
                    a=base+off;b=a+stride;x,y=parity[a],parity[b]
                    parity[a]=x+y;parity[b]=x-y
            stride*=2
        for j in range(len(masks)):
            if fams[j]==fam:out[j]=parity[masks[j]]
    return out

@njit(cache=True)
def decode(e,n,k):
    v=np.zeros(n,np.int64)
    for i in range(n):
        for j in range(3):
            if e[3*i+j]>0:v[i]+=1<<j
        v[i]%=k
    return v

@njit(cache=True)
def evaluate(v,D,q,cap,k,cache):
    loads=np.zeros(k);keys=np.zeros(k,np.int64)
    for i in range(len(v)):
        loads[v[i]]+=q[i];keys[v[i]]|=1<<i
    over=np.maximum(0,loads-cap).sum();distance=0.
    for vehicle in range(k):
        key=keys[vehicle]
        if key not in cache:
            nodes=np.empty(len(v),np.int64);count=0
            for i in range(len(v)):
                if v[i]==vehicle:nodes[count]=i+1;count+=1
            cache[key]=route_cost(D,nodes[:count])
        distance+=cache[key]
    return over,distance

@njit(cache=True)
def train(D,q,cap,k,fams,masks,seed,steps):
    np.random.seed(seed);cache=Dict.empty(types.int64,types.float64)
    U=(len(q)+k)*D.max()+1
    theta=np.random.uniform(-np.pi,np.pi,64)
    e=expectations(theta,fams,masks);v=decode(e,len(q),k)
    over,dist=evaluate(v,D,q,cap,k,cache);energy=over*U+dist
    best=energy;best_theta=theta.copy();same=0
    trace=np.zeros((5,4));checkpoints=np.array([1,1000,5000,10000,20000]);ci=0
    for t in range(steps):
        frac=t/steps;temp=(.1*U)*(.0001/.1)**frac
        proposal=theta.copy();idx=np.random.randint(64)
        proposal[idx]+=np.random.normal(0,np.pi*(.5*(1-frac)+.03))
        new_e=expectations(proposal,fams,masks);w=decode(new_e,len(q),k)
        if np.all(w==v):same+=1
        no,nd=evaluate(w,D,q,cap,k,cache);score=no*U+nd
        if score<=energy or np.random.random()<np.exp(-(score-energy)/temp):
            theta=proposal;energy=score;v=w
            if score<best:best=score;best_theta=theta.copy()
        if ci<5 and t+1==checkpoints[ci]:
            out=decode(expectations(best_theta,fams,masks),len(q),k)
            bo,bd=evaluate(out,D,q,cap,k,cache)
            trace[ci]=np.array([t+1,bo,bd,len(cache)]);ci+=1
    e=expectations(best_theta,fams,masks);v=decode(e,len(q),k)
    return best_theta,e,v,len(cache),same,trace

@njit(cache=True)
def random_prefixes(D,q,cap,k,seed,budgets):
    np.random.seed(seed);cache=Dict.empty(types.int64,types.float64)
    best_over=1e100;best_dist=1e100;winner=np.zeros(len(q),np.int64)
    outputs=np.zeros((len(budgets),len(q)),np.int64);counts=np.zeros(len(budgets),np.int64);j=0
    for t in range(budgets[-1]):
        v=np.random.randint(0,k,len(q));over,dist=evaluate(v,D,q,cap,k,cache)
        if over<best_over or (over==best_over and dist<best_dist):best_over,best_dist,winner=over,dist,v.copy()
        if t+1==budgets[j]:
            outputs[j]=winner;counts[j]=len(cache);j+=1
    return outputs,counts

def slots(S):
    s=K.slots_k(8,3,S['m'])
    f=np.array(['ZXY'.index(a) for a,_ in s],np.int64)
    masks=np.array([sum(1<<(7-q) for q in qs) for _,qs in s],np.int64)
    return s,f,masks

def verify():
    rng=np.random.default_rng(20261014);error=0.
    for name in C.NAMES:
        S=C.A.setup(name);s,f,m=slots(S);c=K.CircuitK(8,4,True).prepare(s)
        for _ in range(3):
            theta=rng.uniform(-np.pi,np.pi,64)
            reference=c.expect_k(c.run(theta));actual=expectations(theta,f,m)
            error=max(error,float(np.max(np.abs(reference-actual))))
            assert np.allclose(reference,actual,rtol=0,atol=1e-12)
            v=decode(actual,S['n'],S['K']);d=C.A.assess(S,v)
            cache=Dict.empty(types.int64,types.float64)
            over,dist=evaluate(v,S['inst']['D'],S['q'],S['cap'],S['K'],cache)
            assert over==d['overload'] and dist==d['distance']
    save('verification.json',dict(passed=True,max_correlator_error=error,independent_parameter_sets=18))
    print('VERIFIED',error,flush=True)

def job(args):
    name,rep=args;S=C.A.setup(name);s,f,m=slots(S)
    seed=zlib.crc32(f'Q8-route:{name}:{rep}:20261014'.encode());t0=time.perf_counter()
    theta,e,v,unique,same,trace=train(S['inst']['D'],S['q'],S['cap'],S['K'],f,m,seed,STEPS)
    elapsed=time.perf_counter()-t0
    q=C.process(S,v,'quantum_8q4','real_route',dict(repeat=rep,seed=seed,theta=theta.tolist(),outputs=e.tolist(),
       slots=[[a,list(b)] for a,b in s],training_seconds=elapsed,training_evaluations=STEPS+1,
       training_unique_route_evaluations=int(unique),unchanged_assignment_proposals=int(same),trace=trace.tolist(),
       min_abs_correlator=float(np.abs(e).min())))
    seed=zlib.crc32(f'R-route:{name}:{rep}:20261014'.encode());t0=time.perf_counter()
    vs,counts=random_prefixes(S['inst']['D'],S['q'],S['cap'],S['K'],seed,np.array(BUDGETS))
    elapsed=time.perf_counter()-t0;rs=[]
    for i,budget in enumerate(BUDGETS):
        rs.append(C.process(S,vs[i],'random','real_route',dict(repeat=rep,seed=seed,budget=budget,
            training_unique_route_evaluations=int(counts[i]),random_full_prefix_seconds=elapsed)))
    return dict(instance=name,repeat=rep,quantum=q,random=rs)

def run():
    from concurrent.futures import ProcessPoolExecutor,as_completed
    verify()
    protocol=dict(instances=C.NAMES,nq=8,layers=4,parameters=64,cz_gates=14,correlation_order=3,
        quantum_steps=STEPS,independent_repeats=REPEATS,random_budgets=BUDGETS,
        selection='容量可行优先；不可行先最小化总超载量，之后均以实际车内排序总路程比较。',
        training='模拟退火；能量=总超载量×U+实际路程，U=(客户数+车辆数)×最大边长+1，确保一单位超载优先于任何路程差。',
        temperature='T0=0.1U，T1=0.0001U，指数降温；单参数扰动沿用旧幅度规则；无读出裕量惩罚。',
        refinement='复用compare.py，最多10000个候选评价；同一输出前后跟踪，不按精修结果换赢家。',
        measurement='理想模拟训练；本轮无真机任务，关联裕量记录用于后续读出检验。',
        fair_claim='随机预算是质量参照，不与量子搜索步数或测量次数作等成本换算。',
        code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    if (HERE/'protocol.json').exists():
        old=C.read(HERE/'protocol.json');assert old==protocol,'冻结协议不匹配'
    else:save('protocol.json',protocol)
    out=HERE/'runs';out.mkdir(exist_ok=True)
    jobs=[(n,r) for n in C.NAMES for r in range(REPEATS) if not (out/f'{n}_{r:02d}.json').exists()]
    with ProcessPoolExecutor(max_workers=4) as pool:
        futures={pool.submit(job,j):j for j in jobs}
        for future in as_completed(futures):
            rec=future.result();save(f"runs/{rec['instance']}_{rec['repeat']:02d}.json",rec)
            q=rec['quantum'];print(rec['instance'],rec['repeat'],q['before']['feasible'],q['before']['distance'],q['after']['distance'],round(q['training_seconds'],1),flush=True)
    print('COMPLETE',len(list(out.glob('*.json'))),flush=True)

if __name__=='__main__':
    if len(sys.argv)>1 and sys.argv[1]=='verify':verify()
    else:run()
