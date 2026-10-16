"""disco上的完整规模对照。所有有限测量为完整8位读数，统一解码和精修。"""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import sys,json,time,zlib,socket,hashlib
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent
REPEATS=32
SHOTS=4096
BUDGETS=[1,4,16,64,256,1024,4096]

def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def save(p,x):
    p=Path(p);p.parent.mkdir(exist_ok=True,parents=True);tmp=p.with_suffix('.tmp')
    tmp.write_text(json.dumps(x,ensure_ascii=False,indent=1,allow_nan=False),encoding='utf-8');tmp.replace(p)
def seed(label):return zlib.crc32(('stable-controls-20261015:'+label).encode())

def load_setup(data):
    return dict(inst=dict(name=data['name'],D=np.array(data['D']),bks=data['bks']),
                n=data['n'],K=data['K'],cap=data['cap'],q=np.array(data['q']))

def probabilities(theta):
    state=np.zeros((2,)*8,dtype=complex);state[(0,)*8]=1
    theta=np.array(theta).reshape(4,8,2)
    def gate(psi,u,q):return np.moveaxis(np.tensordot(u,np.moveaxis(psi,q,0),axes=(1,0)),0,q)
    words=np.arange(256)
    for layer in range(4):
        for q in range(8):
            a,b=theta[layer,q];c,s=np.cos(a/2),np.sin(a/2)
            u=np.diag([np.exp(-.5j*b),np.exp(.5j*b)])@np.array([[c,-s],[s,c]])
            state=gate(state,u,q)
        for q in range(layer%2,7,2):
            minus=((words>>(7-q))&1)&((words>>(6-q))&1)
            state=state*(1-2*minus).reshape((2,)*8)
    h=np.array([[1,1],[1,-1]])/np.sqrt(2)
    result={'Z':np.abs(state.reshape(-1))**2}
    for f in ('X','Y'):
        phi=state.copy();u=h if f=='X' else h@np.diag([1,-1j])
        for q in range(8):phi=gate(phi,u,q)
        result[f]=np.abs(phi.reshape(-1))**2
    return result

def make_groups(slots):
    bits=(np.arange(256)[:,None]>>(7-np.arange(8)))&1
    signs=1-2*bits
    return {f:(np.array([i for i,s in enumerate(slots) if s[0]==f]),
               np.column_stack([np.prod(signs[:,s[1]],axis=1) for s in slots if s[0]==f])) for f in 'ZXY'}

def expectations(groups,pp,seed_value=None):
    out=np.zeros(sum(len(idx) for idx,signs in groups.values()));counts={}
    rng=np.random.default_rng(seed_value)
    for f,(idx,signs) in groups.items():
        p=pp[f]/pp[f].sum()
        if seed_value is None:out[idx]=p@signs
        else:
            n=rng.multinomial(SHOTS,p);counts[f]=n.tolist();out[idx]=n@signs/SHOTS
    return out,counts

def assess(S,v):
    from shared import route_cost
    K=S['K'];loads=np.array([S['q'][v==k].sum() for k in range(K)])
    invalid=int(np.sum((v<0)|(v>=K)));over=float(np.maximum(0,loads-S['cap']).sum())
    distance=float(sum(route_cost(S['inst']['D'],np.flatnonzero(v==k).astype(np.int64)+1) for k in range(K)))
    feasible=invalid==0 and over==0
    return dict(feasible=feasible,invalid_codes=invalid,overload=over,distance=distance,
                gap_bks_percent=100*(distance/S['inst']['bks']-1) if feasible else None)

def process(S,v,method,extra=None):
    from shared import refine
    before=assess(S,v);t=time.perf_counter();w,counts=refine(S,v);after=assess(S,w)
    assert (after['overload'],after['distance'])<=(before['overload'],before['distance'])
    return dict(instance=S['inst']['name'],method=method,before=before,after=after,
                reference_distance=S['inst']['bks'],initial_assignment=v.tolist(),final_assignment=w.tolist(),
                refinement=counts,refinement_seconds=time.perf_counter()-t,**(extra or {}))

def circuit_job(key):
    from shared import decode
    data=read(HERE/'inputs.json');src=data['sources'][key];S=load_setup(data['instances'][src['instance']])
    groups=make_groups(src['slots']);target=np.array(src['outputs'])
    variants={'stable_finite':src['theta'],'untrained_finite':src['untrained_theta'],
              'historical_finite':src['original_theta']}
    pp={m:probabilities(th) for m,th in variants.items()}
    pp['no_information']={f:np.ones(256)/256 for f in 'ZXY'}
    rows=[];t=time.perf_counter()
    def add(e,method,rep,counts=None,draw_seed=None):
        v=decode(e,S['n'],S['K']);den=np.linalg.norm(e)*np.linalg.norm(target)
        rows.append(process(S,v,method,dict(source=key,repeat=rep,outputs=e.tolist(),counts=counts,
                    draw_seed=draw_seed,sign_agreement=float(np.mean((e>0)==(target>0))),
                    cosine=float(e@target/den) if den else 0.0,
                    same_stable_assignment=bool(np.array_equal(v,src['initial_assignment'])))))
    for m in ('stable','untrained'):
        e,_=expectations(groups,pp[m+'_finite']);add(e,m+'_exact',0)
    for method,p in pp.items():
        for rep in range(REPEATS):
            sd=seed(f'{key}:{method}:{rep}');e,counts=expectations(groups,p,sd);add(e,method,rep,counts,sd)
    save(HERE/'results'/f'{key}.json',dict(source=key,rows=rows,seconds=time.perf_counter()-t))
    return key,len(rows),time.perf_counter()-t

def global_job(name):
    from shared import random_prefixes
    data=read(HERE/'inputs.json');S=load_setup(data['instances'][name]);rows=[];t=time.perf_counter()
    for rep in range(REPEATS):
        sd=seed(f'global:{name}:{rep}');start=time.perf_counter()
        vs,counts=random_prefixes(S['inst']['D'],S['q'],S['cap'],S['K'],sd,np.array(BUDGETS))
        elapsed=time.perf_counter()-start
        for j,budget in enumerate(BUDGETS):
            rows.append(process(S,vs[j],f'global_{budget}',dict(repeat=rep,seed=sd,budget=budget,
                        full_prefix_seconds=elapsed,unique_vehicle_route_evaluations=int(counts[j]))))
    save(HERE/'results'/f'global_{name}.json',dict(instance=name,rows=rows,seconds=time.perf_counter()-t))
    return 'global_'+name,len(rows),time.perf_counter()-t

def main():
    from concurrent.futures import ProcessPoolExecutor,as_completed
    data=read(HERE/'inputs.json')
    save(HERE/'execution.json',dict(host=socket.gethostname(),python=sys.executable,pid=os.getpid(),
         cwd=str(HERE),numpy=np.__version__,start_unix=time.time(),
         input_sha256=hashlib.sha256((HERE/'inputs.json').read_bytes()).hexdigest(),
         stable_source_hashes=data['hashes']))
    jobs=[(circuit_job,k) for k in data['sources'] if not (HERE/'results'/f'{k}.json').exists()]
    jobs += [(global_job,n) for n in data['instances'] if not (HERE/'results'/f'global_{n}.json').exists()]
    with ProcessPoolExecutor(max_workers=6) as pool:
        for task in as_completed([pool.submit(f,k) for f,k in jobs]):print(*task.result(),flush=True)
    save(HERE/'completion.json',dict(complete=True,host=socket.gethostname(),files=len(list((HERE/'results').glob('*.json'))),finish_unix=time.time()))
    print('COMPLETE',flush=True)

if __name__=='__main__':main()
