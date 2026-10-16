"""正式实例上的泡利关联编码、相关测量、原始分车和训练目标。"""
import os
for _name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_name] = '1'
import sys, json, hashlib, itertools, time
from pathlib import Path
import numpy as np
from numba import njit, types
from numba.typed import Dict

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(HERE.parent / 'pce_stable_controls_20261015'))
from shared import route_cost, refine

def save(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=1, allow_nan=False), encoding='utf-8')
    tmp.replace(path)

def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def seed(*parts):
    return int.from_bytes(hashlib.sha256(':'.join(map(str, parts)).encode()).digest()[:4], 'little')

def setup(name):
    x = read(HERE / 'instances' / (name + '.json'))
    return dict(n=x['n'], K=x['K'], cap=x['cap'], q=np.array(x['q']),
                inst=dict(name=name, D=np.array(x['D']), bks=x['bks'], P=np.array(x['P'])))

def mapping(n, k, nq, old=False, mapping_seed=20261015):
    rng = np.random.default_rng(mapping_seed)
    pool = [(a, sum(1 << (nq-1-i) for i in qs), qs)
            for a in range(3) for order in ((3,) if old else (2,3))
            for qs in itertools.combinations(range(nq), order)]
    if old:
        ids = np.random.default_rng(20261014).permutation(len(pool))[:3*n]
        selected = [pool[i] for i in ids]
    else:
        # 按每列的轴、阶数、比特使用计数逐槽平衡；不使用需求或坐标。
        selected = [None]*(n*k); used = set()
        axis = np.zeros((k,3), int); order_count = np.zeros((k,2), int)
        support = np.zeros((k,nq), int)
        for i in range(n):
            for col in rng.permutation(k):
                available = [j for j in range(len(pool)) if j not in used]
                costs = []
                for j in available:
                    a, _, qs = pool[j]
                    costs.append(100*axis[col,a] + 100*order_count[col,len(qs)-2]
                                 + 2*sum(support[col,q] for q in qs) + rng.random())
                j = available[int(np.argmin(costs))]; used.add(j)
                a, mask, qs = pool[j]; selected[i*k+col] = pool[j]
                axis[col,a] += 1; order_count[col,len(qs)-2] += 1
                support[col,list(qs)] += 1
    fams = np.array([x[0] for x in selected], np.int64)
    masks = np.array([x[1] for x in selected], np.int64)
    priorities = np.random.default_rng(seed('ties',mapping_seed,n,k)).random((n,k))
    return fams, masks, priorities

@njit(cache=True)
def probabilities(theta, nq, layers):
    dim = 1 << nq
    psi = np.zeros(dim, np.complex128); psi[0] = 1
    for layer in range(layers):
        for q in range(nq):
            bit = 1 << (nq-1-q); j = 2*(layer*nq+q)
            c,s = np.cos(theta[j]/2),np.sin(theta[j]/2)
            z0,z1 = np.exp(-.5j*theta[j+1]),np.exp(.5j*theta[j+1])
            for base in range(0,dim,2*bit):
                for off in range(bit):
                    a,b = base+off,base+off+bit; x,y = psi[a],psi[b]
                    psi[a],psi[b] = z0*(c*x-s*y),z1*(s*x+c*y)
        for q in range(layer%2,nq-1,2):
            a,b = 1 << (nq-1-q),1 << (nq-2-q)
            for j in range(dim):
                if j&a and j&b: psi[j] = -psi[j]
    pp = np.empty((3,dim))
    for fam in range(3):
        phi = psi.copy()
        if fam:
            for q in range(nq):
                bit = 1 << (nq-1-q)
                for base in range(0,dim,2*bit):
                    for off in range(bit):
                        a,b = base+off,base+off+bit; x,y = phi[a],phi[b]
                        if fam==2: y = -1j*y
                        phi[a],phi[b] = (x+y)/np.sqrt(2),(x-y)/np.sqrt(2)
        pp[fam] = np.abs(phi)**2
        pp[fam] /= pp[fam].sum()
    return pp

@njit(cache=True)
def walsh(pp):
    out = pp.copy(); dim = pp.shape[1]
    for f in range(3):
        stride = 1
        while stride < dim:
            for base in range(0,dim,2*stride):
                for off in range(stride):
                    a,b = base+off,base+off+stride; x,y = out[f,a],out[f,b]
                    out[f,a],out[f,b] = x+y,x-y
            stride *= 2
    return out

@njit(cache=True)
def scores_from_moments(moments, fams, masks):
    return np.array([moments[fams[i],masks[i]] for i in range(len(masks))])

def allocation(shots):
    # 顺序Z、X、Y；余数优先X，与执行书一致。
    out = np.full(3,shots//3,np.int64)
    for f in (1,2,0)[:shots%3]: out[f] += 1
    return out

def sample(pp, fams, masks, rng, shots=4096, batches=1):
    alloc = allocation(shots)
    counts = np.stack([rng.multinomial(int(alloc[f]),pp[f],size=batches) for f in range(3)],axis=1)
    values = np.stack([scores_from_moments(walsh(c/alloc[:,None]),fams,masks) for c in counts])
    return values,counts

@njit(cache=True)
def decode(values, n, k, old, priorities):
    v = np.zeros(n,np.int64)
    for i in range(n):
        if old:
            for b in range(3):
                if values[3*i+b]>0: v[i] += 1 << b
            v[i] %= k
        else:
            best = np.max(values[i*k:(i+1)*k]); winner = -1; p = -1.
            for col in range(k):
                if abs(values[i*k+col]-best)<=1e-12 and priorities[i,col]>p:
                    winner,p = col,priorities[i,col]
            v[i] = winner
    return v

@njit(cache=True)
def evaluate(v,D,q,cap,k,cache):
    loads = np.zeros(k); keys = np.zeros(k,np.int64)
    for i in range(len(v)):
        loads[v[i]] += q[i]; keys[v[i]] |= 1 << i
    over = np.maximum(0,loads-cap).sum(); distance = 0.
    for col in range(k):
        key = keys[col]
        if key not in cache:
            nodes = np.empty(len(v),np.int64); count = 0
            for i in range(len(v)):
                if v[i]==col: nodes[count]=i+1; count+=1
            cache[key] = route_cost(D,nodes[:count])
        distance += cache[key]
    return over,distance,loads

@njit(cache=True)
def margin(values,moments,fams,masks,n,k,priorities,alloc):
    v = decode(values,n,k,False,priorities); z = np.empty(n); gaps = np.empty(n)
    for i in range(n):
        a=i*k+v[i]; best_z=1e20; best_gap=1e20
        for col in range(k):
            if col==v[i]: continue
            b=i*k+col; gap=values[a]-values[b]
            var=(1-values[a]**2)/alloc[fams[a]]+(1-values[b]**2)/alloc[fams[b]]
            if fams[a]==fams[b]:
                cov=moments[fams[a],masks[a]^masks[b]]-values[a]*values[b]
                var-=2*cov/alloc[fams[a]]
            zz=gap/max(np.sqrt(max(0.,var)),1e-8)
            best_z=min(best_z,zz); best_gap=min(best_gap,gap)
        z[i],gaps[i]=best_z,best_gap
    return np.mean(np.maximum(0,1-z/2)**2),z,gaps

class Evaluator:
    def __init__(self,S,method,nq,layers=4,mapping_seed=20261015):
        self.S,self.method,self.nq,self.layers=S,method,nq,layers
        self.old=method=='LEGACY'
        self.fams,self.masks,self.priorities=mapping(S['n'],S['K'],nq,self.old,mapping_seed)
        self.cache=Dict.empty(types.int64,types.float64)
        self.candidates=set(); self.calls=0; self.query_count=0; self.shots=0; self.states=0
        self.scale=max(1.,(S['n']+S['K'])*S['inst']['D'].max())

    def assess(self,v):
        S=self.S; self.calls+=1; self.candidates.add(v.tobytes())
        over,dist,loads=evaluate(v,S['inst']['D'],S['q'],S['cap'],S['K'],self.cache)
        return dict(feasible=bool(over==0),overload=float(over),distance=float(dist),loads=loads.tolist())

    def decode(self,e):
        return decode(e,self.S['n'],self.S['K'],self.old,self.priorities)

    def task_loss(self,a):
        return a['distance']/self.scale+30*a['overload']/self.S['q'].sum()

    def objective(self,theta,rng,diagnostics=True):
        pp=probabilities(theta,self.nq,self.layers); self.states+=1; self.query_count+=1
        moments=walsh(pp); e=scores_from_moments(moments,self.fams,self.masks)
        v=self.decode(e); ref=self.assess(v); f=self.task_loss(ref)
        measured=[]
        if diagnostics:
            ee,_=sample(pp,self.fams,self.masks,rng,batches=4); self.shots+=4*4096
            measured=[self.assess(self.decode(x)) for x in ee]
        if self.old:
            m=0.; loss=f
        else:
            m,_,_=margin(e,moments,self.fams,self.masks,self.S['n'],self.S['K'],self.priorities,allocation(4096))
            loss=f
            if self.method not in ('NEW_DECODER','RANDOM_THETA','RANDOM_SEARCH'):
                loss=.35*f+.55*np.mean([self.task_loss(a) for a in measured])+.1*m
                if self.method in ('V41_SOFT_CAPACITY','V42_GEOMETRY'):
                    scores=e.reshape(self.S['n'],self.S['K'])/.25
                    p=np.exp(scores-scores.max(axis=1,keepdims=True)); p/=p.sum(axis=1,keepdims=True)
                    load=self.S['q']@p
                    loss+=.25*np.mean((load/self.S['cap'])**2)
                    loss+=.25*np.sum(np.maximum(0,load-self.S['cap'])**2)/self.S['cap']**2
                    if self.method=='V42_GEOMETRY':
                        D=self.S['inst']['D'][1:,1:]; denom=max(1.,D.sum())
                        loss+=.05*np.sum((p@p.T)*D)/denom
        return float(loss),dict(ref=ref,measured_feasible=float(np.mean([a['feasible'] for a in measured])) if measured else None,
                               measured_overload=float(np.mean([a['overload'] for a in measured])) if measured else None,margin=float(m))

    def readouts(self,theta,rng,batches=64,shots=4096):
        pp=probabilities(theta,self.nq,self.layers); self.states+=1
        e=scores_from_moments(walsh(pp),self.fams,self.masks); ref=self.decode(e)
        ee,counts=sample(pp,self.fams,self.masks,rng,shots,batches); self.shots+=shots*batches
        rows=[]
        for b,x in enumerate(ee):
            v=self.decode(x); a=self.assess(v)
            rows.append(dict(batch=b,shots=shots,assignment=v.tolist(),**a,
                customer_flip_rate=float(np.mean(v!=ref)),full_assignment_match=bool(np.array_equal(v,ref))))
        return rows,counts,e,ee

    def ledger(self):
        return dict(objective_queries=self.query_count,state_simulations=self.states,logical_candidate_evaluations=self.calls,
                    unique_assignments=len(self.candidates),unique_route_evaluations=len(self.cache),simulated_shots=self.shots,
                    hardware_shots=0)

def summary(rows):
    ok=[r['distance'] for r in rows if r['feasible']]
    return dict(raw_feasibility_rate=float(np.mean([r['feasible'] for r in rows])),
                raw_mean_overload=float(np.mean([r['overload'] for r in rows])),
                feasible_raw_route_distance=float(np.mean(ok)) if ok else None,
                customer_flip_rate=float(np.mean([r.get('customer_flip_rate',0) for r in rows])))

def rank(s):
    return (-s['raw_feasibility_rate'],s['raw_mean_overload'],s['feasible_raw_route_distance'] if s['feasible_raw_route_distance'] is not None else float('inf'),s['customer_flip_rate'])
