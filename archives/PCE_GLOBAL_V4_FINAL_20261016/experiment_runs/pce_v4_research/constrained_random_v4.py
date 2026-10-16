"""仅在disco执行：当前冻结模型的约束内随机对照。"""
from final_objective_accelerated import *
from concurrent.futures import ProcessPoolExecutor,as_completed
import socket,tarfile
OUT=HERE/'constrained_random_v4'
BUDGETS=(1,4,16,64,256,4096)
MAX_TRIES=100

@njit(cache=True)
def generate_candidates(q,cap,K,D,rng,count):
    n=len(q);assignments=np.full((count,n),-1,np.int64);distances=np.full(count,np.inf)
    attempts=np.zeros(count,np.int64)
    for draw in range(count):
        for attempt in range(1,MAX_TRIES+1):
            order=rng.permutation(n);loads=np.zeros(K);v=np.full(n,-1,np.int64);ok=True
            for i in order:
                allowed=np.empty(K,np.int64);nk=0
                for k in range(K):
                    if loads[k]+q[i]<=cap:allowed[nk]=k;nk+=1
                if nk==0:ok=False;break
                k=allowed[rng.integers(0,nk)];v[i]=k;loads[k]+=q[i]
            attempts[draw]=attempt
            if ok:
                assignments[draw]=v;distance=0.
                for k in range(K):
                    nodes=np.empty(n,np.int64);used=0
                    for i in range(n):
                        if v[i]==k:nodes[used]=i+1;used+=1
                    distance+=accelerated_route_cost(D,nodes[:used])
                distances[draw]=distance;break
    return assignments,distances,attempts

def job(args):
    name,sd=args;p=OUT/'results'/name/f'{sd}.json'
    if p.exists():return str(p)
    S=setup(name);E=FinalEvaluator(S,'FINAL_JOINT',8 if S['n']==30 else 9)
    rows=[];candidate_distances=np.empty((64,4096),np.float64);construction_attempts=np.empty((64,4096),np.int64)
    start=time.perf_counter()
    for batch in range(64):
        stream=seed('constrained_random_v4',name,sd,batch);rng=np.random.default_rng(stream)
        best_distance=float('inf');winner=None;offset=0;tries=0;fails=0;generation_seconds=0.
        for budget in BUDGETS:
            t=time.perf_counter();vv,dd,tt=generate_candidates(S['q'],S['cap'],S['K'],S['inst']['D'],rng,budget-offset)
            generation_seconds+=time.perf_counter()-t;candidate_distances[batch,offset:budget]=dd;construction_attempts[batch,offset:budget]=tt
            idx=int(np.argmin(dd));tries+=int(tt.sum());fails+=int(np.sum(~np.isfinite(dd)))
            if dd[idx]<best_distance:best_distance=float(dd[idx]);winner=vv[idx].copy()
            meta=dict(instance=name,seed=sd,batch=batch,method=f'CONSTRAINED_RANDOM_{budget}',candidate_budget=budget,
                stream_seed=stream,construction_attempts=tries,failed_candidate_constructions=fails,
                generation_seconds=generation_seconds,quantum_measurements=0)
            if winner is None:
                rows.append(dict(**meta,construction_failed=True,assignment=None,raw=None,
                    final=dict(feasible=False,overload=None,distance=None),final_assignment=None,changed_customers=None))
            else:
                t=time.perf_counter();_,h=E.hybrid(winner);refine_seconds=time.perf_counter()-t
                assert h['raw']['feasible'] and h['raw']['overload']==0 and h['raw']['distance']==best_distance
                assert h['refinement_candidate_evaluations']<=10000
                rows.append(dict(**meta,construction_failed=False,assignment=winner.tolist(),
                    refinement_seconds=refine_seconds,**h))
            offset=budget
        if (batch+1)%16==0:print('约束内随机',name,sd,batch+1,'/64',round(time.perf_counter()-start,2),flush=True)
    p.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(p.with_suffix('.npz'),candidate_distances=candidate_distances,construction_attempts=construction_attempts)
    save(p,dict(instance=name,seed=sd,rows=rows,seconds=time.perf_counter()-start,ledger=E.ledger(),
        instance_sha256=digest(HERE/'instances'/(name+'.json')),protocol_sha256=digest(OUT/'protocol.json'),
        source_sha256=digest(Path(__file__)),host=socket.gethostname()))
    return str(p)

def main():
    assert socket.gethostname()=='lenovo','本补充计算按用户要求仅在disco执行'
    OUT.mkdir(exist_ok=True)
    protocol=dict(created_unix=time.time(),comparison='当前冻结FINAL_JOINT的4096次测量独立测试，参数不再调整',
        instances={'B-n31-k5':list(range(2001,2021)),'A-n45-k7':list(range(3001,3011))},batches_per_seed=64,budgets=list(BUDGETS),
        generator='每次随机打乱全部客户；每客户在仍能满足容量的车辆中等概率选择；无车可放时废弃该尝试并重新开始；每候选最多100次尝试。',
        distribution='容量感知的顺序随机构造，并非整个可行域的均匀抽样；不使用坐标、距离或老师分配指导构造。',
        failure='100次尝试均失败则该候选失败；预算中保留其消耗，不补抽。预算内没有成功候选则整批失败。',
        selection='各档共享同一随机序列前缀，仅按精修前可行总路线距离选优；并列取更早候选，然后精修一次。',
        refinement=dict(candidate_budget=10000,algorithm='复用当前最终模型的同一确定性跨车移动与交换精修',stop='无严格改善或10000候选',randomness='无随机数'),
        resources='这是经典分配候选预算；4096份不等于量子训练4096次目标查询，后者每查询含参考及4次读出精修。分别报告成本，不声称端到端等成本。',
        statistical_unit='先将每主种子的64批汇总，再在20或10个主种子层面配对；不是1920个独立问题实例。',
        primary='1份约束内随机为主对照；4、16、64、256、4096为预定预算阶梯，全部报告；路线检验在两实例六档共12项上做Holm校正。',
        source_sha256=digest(Path(__file__)),refiner_sha256=digest(HERE/'final_objective_accelerated.py'),
        previous_version='旧约束内随机的6实例结果不混入本轮；本轮无超载后备分配。')
    if not (OUT/'protocol.json').exists():save(OUT/'protocol.json',protocol)
    save(OUT/'execution.json',dict(alias='disco',host=socket.gethostname(),pid=os.getpid(),workers=12,python=sys.executable,
        numpy=np.__version__,started_unix=time.time()))
    # 在正式实例上预编译，并对原路线函数复算构造结果；不采用小规模替代实例。
    verification=[]
    for name in protocol['instances']:
        S=setup(name);vv,dd,tt=generate_candidates(S['q'],S['cap'],S['K'],S['inst']['D'],np.random.default_rng(69031),8)
        for v,d in zip(vv,dd):
            assert np.isfinite(d)
            assert np.all(np.bincount(v,weights=S['q'],minlength=S['K'])<=S['cap'])
            expected=sum(route_cost(S['inst']['D'],(np.flatnonzero(v==k)+1).astype(np.int64)) for k in range(S['K']))
            assert expected==d
        verification.append(dict(instance=name,assignments=8,original_route_cost_matches=True))
    save(OUT/'implementation_verification.json',dict(passed=True,records=verification))
    start=time.perf_counter();work=[(n,sd) for n,seeds in protocol['instances'].items() for sd in seeds]
    with ProcessPoolExecutor(max_workers=12) as pool:
        for f in as_completed([pool.submit(job,x) for x in work]):f.result()
    save(OUT/'completion.json',dict(complete=True,seconds=time.perf_counter()-start,seed_blocks=len(work),candidate_budgets=list(BUDGETS)))
    with tarfile.open(HERE.parent/'constrained_random_v4_results.tar.gz','w:gz') as tar:
        for p in OUT.rglob('*'):
            if p.is_file():tar.add(p,arcname=p.relative_to(HERE))
    print('约束内随机全部完成',time.perf_counter()-start,flush=True)
if __name__=='__main__':main()
