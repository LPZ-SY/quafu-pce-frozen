"""冻结后的独立测试、信息来源、容量约束分配及统一经典精修。"""
from core import *
from concurrent.futures import ProcessPoolExecutor,as_completed

def get_evaluator(model):
    S=setup(model['instance']);method=model['method']
    actual='NEW_JOINT' if method=='DEEP_JOINT' else 'NEW_DECODER' if method=='DEEP_DECODER' else method
    E=Evaluator(S,actual,model['nq'],model['layers'])
    assert np.array_equal(E.fams,model['mapping_fams']) and np.array_equal(E.masks,model['mapping_masks'])
    return E

def raw_job(path):
    path=Path(path);out=path.parent/'test.json'
    if out.exists():return
    m=read(path);E=get_evaluator(m);theta=np.array(m['theta']);start=time.perf_counter();allrows=[];stats={}
    for shots in (1024,4096,16384):
        sd=seed('final_test',m['phase'],m['instance'],m['seed'],shots)
        rows,counts,ref,values=E.readouts(theta,np.random.default_rng(sd),shots=shots)
        stats[str(shots)]=summary(rows)
        for r in rows:r.update(phase=m['phase'],instance=m['instance'],method=m['method'],seed=m['seed'],stream_seed=sd,stage='RAW')
        allrows.extend(rows)
        np.savez_compressed(path.parent/f'test_counts_{shots}.npz',counts=counts,scores=values,reference_scores=ref,
                            shots_per_basis=allocation(shots),axis_order=np.array(['Z','X','Y']))
    pp=probabilities(theta,E.nq,E.layers);mom=walsh(pp);ref=scores_from_moments(mom,E.fams,E.masks)
    if not E.old:
        mm,z,gaps=margin(ref,mom,E.fams,E.masks,E.S['n'],E.S['K'],E.priorities,allocation(4096))
        signal=dict(standardized_gap_quantiles=np.quantile(z,[0,.25,.5,.75,1]).tolist(),
                    raw_gap_quantiles=np.quantile(gaps,[0,.25,.5,.75,1]).tolist(),fraction_gap_below_010=float(np.mean(gaps<.1)))
    else:signal=dict(min_abs_correlator=float(np.abs(ref).min()))
    save(out,dict(model=str(path.relative_to(HERE)),model_sha256=digest(path),summary=stats,rows=allrows,signal=signal,
                  ledger=E.ledger(),seconds=time.perf_counter()-start))

def refine_stages(S,initial):
    """与旧共享精修同样的候选顺序和预算，补存首次容量可行的中间结果。"""
    v=initial.copy();n,K,q,cap=S['n'],S['K'],S['q'],S['cap'];cache={};evals=0;accepted=0
    def cost(nodes):
        key=tuple(sorted(nodes))
        if key not in cache:cache[key]=float(route_cost(S['inst']['D'],np.array(key,np.int64)))
        return cache[key]
    groups=[set((np.flatnonzero(v==k)+1).tolist()) for k in range(K)]
    loads=np.array([sum(q[i-1] for i in g) for g in groups]);costs=[cost(g) for g in groups]
    current=(float(np.maximum(0,loads-cap).sum()),sum(costs));repair=v.copy() if current[0]==0 else None
    repair_evaluations=0 if repair is not None else None;start=time.perf_counter();repair_seconds=0. if repair is not None else None
    while evals<10000:
        best,move=current,None
        def consider(a,b,ga,gb,la,lb):
            nonlocal best,move,evals
            evals+=1
            over=current[0]-max(0,loads[a]-cap)-max(0,loads[b]-cap)+max(0,la-cap)+max(0,lb-cap)
            if over>best[0]:return
            ca,cb=cost(ga),cost(gb);value=(float(over),current[1]-costs[a]-costs[b]+ca+cb)
            if value<best:best,move=value,(a,b,ga,gb,la,lb,ca,cb)
        for i in range(n):
            if evals>=10000:break
            a,node=v[i],i+1
            for b in range(K):
                if evals>=10000:break
                if a!=b:consider(a,b,groups[a]-{node},groups[b]|{node},loads[a]-q[i],loads[b]+q[i])
        for i in range(n):
            if evals>=10000:break
            for j in range(i+1,n):
                if evals>=10000:break
                a,b=v[i],v[j]
                if a!=b:consider(a,b,(groups[a]-{i+1})|{j+1},(groups[b]-{j+1})|{i+1},loads[a]-q[i]+q[j],loads[b]-q[j]+q[i])
        if move is None:break
        a,b,ga,gb,la,lb,ca,cb=move;groups[a],groups[b],loads[a],loads[b],costs[a],costs[b]=ga,gb,la,lb,ca,cb
        for node in ga:v[node-1]=a
        for node in gb:v[node-1]=b
        current=best;accepted+=1
        if repair is None and current[0]==0:repair=v.copy();repair_evaluations=evals;repair_seconds=time.perf_counter()-start
    return v,repair,dict(candidate_evaluations=evals,unique_route_evaluations=len(cache),accepted=accepted,budget=10000,
                         repair_evaluations=repair_evaluations,repair_seconds=repair_seconds,seconds=time.perf_counter()-start)

def process(E,v,extra):
    final,repair,ledger=refine_stages(E.S,v)
    return dict(**extra,raw_assignment=v.tolist(),raw=E.assess(v),repair_assignment=repair.tolist() if repair is not None else None,
        repair=E.assess(repair) if repair is not None else None,final_assignment=final.tolist(),final=E.assess(final),
        final_changed_customers=int(np.sum(final!=v)),final_changed_demand=float(E.S['q'][final!=v].sum()),refinement=ledger)

def cap_assign(E,scores,sd):
    from ortools.sat.python import cp_model
    S=E.S;model=cp_model.CpModel();n,k=S['n'],S['K']
    x=[[model.NewBoolVar(f'x_{i}_{j}') for j in range(k)] for i in range(n)]
    for i in range(n):model.Add(sum(x[i])==1)
    for j in range(k):model.Add(sum(int(S['q'][i])*x[i][j] for i in range(n))<=int(S['cap']))
    ints=np.rint(scores*1e6).astype(np.int64)
    model.Maximize(sum(int(ints[i,j])*x[i][j] for i in range(n) for j in range(k)))
    solver=cp_model.CpSolver();solver.parameters.max_time_in_seconds=30.;solver.parameters.num_search_workers=1
    solver.parameters.random_seed=sd%(2**31-1);start=time.perf_counter();status=solver.Solve(model)
    meta=dict(status=solver.StatusName(status),seconds=time.perf_counter()-start,score_integer_scale=1000000,
              time_limit_seconds=30,workers=1,seed=sd%(2**31-1))
    if status not in (cp_model.OPTIMAL,cp_model.FEASIBLE):return None,meta
    v=np.array([next(j for j in range(k) if solver.Value(x[i][j])) for i in range(n)],np.int64)
    assert E.assess(v)['feasible'];meta.update(objective=float(solver.ObjectiveValue()),bound=float(solver.BestObjectiveBound()))
    return v,meta

def postprocess_job(path):
    path=Path(path);out=path.parent/'postprocess.json'
    if out.exists():return
    m=read(path);E=get_evaluator(m);test=read(path.parent/'test.json');rows=[]
    for r in [r for r in test['rows'] if r['shots']==4096 and r['batch']<8]:
        v=np.array(r['assignment'],np.int64)
        rows.append(process(E,v,dict(batch=r['batch'],method=m['method'],information='TRAINED' if m['method']!='RANDOM_THETA' else 'RANDOM_THETA',pipeline='RAW_FINAL')))
    save(out,dict(rows=rows,model=str(path.relative_to(HERE)),model_sha256=digest(path)))

def attribution_job(path):
    path=Path(path);out=path.parent/'attribution.json'
    if out.exists():return
    m=read(path);E=get_evaluator(m);S=E.S;n,k=S['n'],S['K'];chosen=read(HERE/'PCE_GLOBAL_V4_FINAL.json')['implementation']
    assert m['method']==chosen and not E.old
    z=np.load(path.parent/'test_counts_4096.npz');trained=z['scores'];reference=z['reference_scores']
    rng=np.random.default_rng(seed('attribution',m['phase'],m['instance'],m['seed']))
    uniform=np.full((3,1<<E.nq),1/(1<<E.nq));null,nullcounts=sample(uniform,E.fams,E.masks,rng,batches=64)
    random_theta=np.array(m['initial_theta']);pp=probabilities(random_theta,E.nq,E.layers)
    random_values,random_counts=sample(pp,E.fams,E.masks,rng,batches=64)
    perms=np.stack([rng.permutation(n) for _ in range(20)]);ref=E.decode(reference)
    rows=[];posts=[];caps=[]
    for batch in range(64):
        inputs=[('TRAINED',0,trained[batch]),('RANDOM_THETA',0,random_values[batch]),('NULL_READOUT',0,null[batch])]
        inputs += [('SHUFFLED_ROWS',i,trained[batch].reshape(n,k)[p].ravel()) for i,p in enumerate(perms)]
        for information,perm_id,values in inputs:
            v=E.decode(values);a=E.assess(v)
            rows.append(dict(batch=batch,information=information,permutation=perm_id,assignment=v.tolist(),
                             customer_flip_rate=float(np.mean(v!=ref)),**a))
            if batch<8 and perm_id==0:
                extra=dict(batch=batch,information=information,permutation=perm_id,pipeline='RAW_FINAL')
                posts.append(process(E,v,extra))
                # 第三版约定每规模首10个训练种子，4类输入×8批×10=320次求解。
                if m['seed']%1000<=10:
                    w,meta=cap_assign(E,values.reshape(n,k),seed('cap',m['instance'],m['seed'],batch))
                    entry=dict(batch=batch,information=information,solver=meta,raw_assignment=v.tolist(),
                               changed_customers=int(np.sum(w!=v)) if w is not None else None,
                               changed_demand=float(S['q'][w!=v].sum()) if w is not None else None)
                    if w is not None:entry.update(process(E,w,dict(pipeline='CAP_FINAL')))
                    else:entry.update(pipeline='CAP_FINAL',raw=None,final=None,repair=None)
                    caps.append(entry)
    np.savez_compressed(path.parent/'attribution_counts.npz',null_counts=nullcounts,random_counts=random_counts,
                        permutations=perms,random_theta=random_theta)
    save(out,dict(model=str(path.relative_to(HERE)),rows=rows,postprocess=posts,capacity=caps,ledger=E.ledger(),
                  note='客户行置换20次嵌套在同一父模型内；仅预定置换0进入容量分配与精修。'))
    print('信息来源完成',m['phase'],m['instance'],m['seed'],len(caps),flush=True)

def bias_job(name):
    out=HERE/'decoder_bias'/(name+'.json')
    if out.exists():return
    S=setup(name);n,k=S['n'],S['K'];nq=8 if n==30 else 9
    f,m,p=mapping(n,k,nq);fo,mo,po=mapping(n,k,8,True)
    rng=np.random.default_rng(seed('bias',name));assignments={x:np.empty((10000,n),np.int16) for x in ('GLOBAL_RANDOM_1','IID_SCORES','NULL_OLD','NULL_NEW')}
    for i in range(10000):
        assignments['GLOBAL_RANDOM_1'][i]=rng.integers(k,size=n)
        assignments['IID_SCORES'][i]=decode(rng.normal(size=n*k),n,k,False,p)
        counts=np.stack([rng.multinomial(int(s),np.full(1<<nq,1/(1<<nq))) for s in allocation(4096)])
        moments=walsh(counts/allocation(4096)[:,None])
        e=scores_from_moments(moments,f,m);assignments['NULL_NEW'][i]=decode(e,n,k,False,p)
        oldcounts=counts if nq==8 else counts.reshape(3,256,2).sum(axis=2)
        eo=scores_from_moments(walsh(oldcounts/allocation(4096)[:,None]),fo,mo)
        assignments['NULL_OLD'][i]=decode(eo,n,k,True,po)
    results={}
    for method,v in assignments.items():
        loads=np.stack([(S['q']*(v==col)).sum(axis=1) for col in range(k)],axis=1);over=np.maximum(0,loads-S['cap']).sum(axis=1)
        results[method]=dict(assignments=10000,vehicle_selection_rate=[float(np.mean(v==col)) for col in range(k)],
            demand_share=(loads.mean(axis=0)/S['q'].sum()).tolist(),load_mean=loads.mean(axis=0).tolist(),load_variance=loads.var(axis=0).tolist(),
            load_covariance=np.cov(loads,rowvar=False).tolist(),raw_feasibility_rate=float(np.mean(over==0)),raw_mean_overload=float(over.mean()))
    out.parent.mkdir(exist_ok=True);np.savez_compressed(out.with_suffix('.npz'),**assignments)
    save(out,dict(instance=name,results=results,note='两种空信息管线使用成对完整位串；9比特时取最高8位为旧管线输入。'))
    print('解码偏向完成',name,flush=True)

def baseline_job(args):
    name,sd=args;out=HERE/'classical'/name/f'{sd}.json'
    if out.exists():return
    S=setup(name);E=Evaluator(S,'NEW_DECODER',8 if S['n']==30 else 9);rng=np.random.default_rng(seed('classical',name,sd));rows=[]
    v=None;best=(float('inf'),float('inf'));start=time.perf_counter()
    for t in range(4096):
        candidate=rng.integers(S['K'],size=S['n'],dtype=np.int64);a=E.assess(candidate);score=(a['overload'],a['distance'])
        if score<best:best,v=score,candidate.copy()
        if t+1 in (1,256,4096):rows.append(process(E,v,dict(method=f'GLOBAL_RANDOM_{t+1}',candidate_budget=t+1,search_seconds=time.perf_counter()-start)))
    # 经典直接分车：原生路线与公共车内评价分别保存。
    from ortools.constraint_solver import pywrapcp,routing_enums_pb2
    chosen=read(HERE/('raw_priority_frozen_model.json' if (HERE/'raw_priority_frozen_model.json').exists() else 'PCE_GLOBAL_V4_FINAL.json'))['implementation'];phase='confirmation' if S['n']==30 else 'scale'
    model=read(HERE/'models'/phase/name/f'{chosen}_{sd}'/'result.json')
    seconds=max(.001,model['training_seconds']);manager=pywrapcp.RoutingIndexManager(S['n']+1,S['K'],0)
    routing=pywrapcp.RoutingModel(manager);D=S['inst']['D'];dem=np.r_[0,S['q']]
    cb=routing.RegisterTransitCallback(lambda i,j:int(D[manager.IndexToNode(i),manager.IndexToNode(j)]));routing.SetArcCostEvaluatorOfAllVehicles(cb)
    demand=routing.RegisterUnaryTransitCallback(lambda i:int(dem[manager.IndexToNode(i)]))
    routing.AddDimensionWithVehicleCapacity(demand,0,[int(S['cap'])]*S['K'],True,'Capacity')
    pars=pywrapcp.DefaultRoutingSearchParameters();pars.first_solution_strategy=routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    pars.local_search_metaheuristic=routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    pars.time_limit.FromMilliseconds(int(seconds*1000));start=time.perf_counter();sol=routing.SolveWithParameters(pars)
    if sol:
        v=np.full(S['n'],-1,np.int64);routes=[]
        for col in range(S['K']):
            idx=routing.Start(col);route=[0]
            while not routing.IsEnd(idx):
                node=manager.IndexToNode(idx)
                if node:v[node-1]=col;route.append(node)
                idx=sol.Value(routing.NextVar(idx))
            route.append(0);routes.append(route)
        assert np.all(v>=0)
        rows.append(process(E,v,dict(method='CLASSICAL_DIRECT',native_routes=routes,native_distance=float(sol.ObjectiveValue()),
                    search_seconds=time.perf_counter()-start,time_limit_seconds=seconds)))
    else:rows.append(dict(method='CLASSICAL_DIRECT',raw=None,final=None,status='NO_SOLUTION',time_limit_seconds=seconds))
    # 固定随机起点，局部随机调车；保存首次分配与4096次评价后的结果。
    v=rng.integers(S['K'],size=S['n'],dtype=np.int64);a=E.assess(v);start=time.perf_counter()
    for _ in range(4096):
        w=v.copy();i=rng.integers(S['n']);w[i]=rng.integers(S['K']);b=E.assess(w)
        if (b['overload'],b['distance'])<(a['overload'],a['distance']):v,a=w,b
    rows.append(process(E,v,dict(method='LOCAL_RANDOM_MOVE',candidate_budget=4096,search_seconds=time.perf_counter()-start)))
    # 固定客户最多的两车，枚举其中编号最小的8客户；避免把其他车辆客户混入双车局部问题。
    initial=v.copy();vehicles=sorted(range(S['K']),key=lambda col:(-int(np.sum(v==col)),col))[:2]
    a,b=vehicles;nodes=np.flatnonzero(np.isin(v,vehicles))[:8];assert len(nodes)==8
    best=(float('inf'),float('inf'));winner=None
    for bits in range(256):
        w=initial.copy();w[nodes]=[vehicles[(bits>>i)&1] for i in range(8)];r=E.assess(w)
        if (r['overload'],r['distance'])<best:best=(r['overload'],r['distance']);winner=w
    rows.append(process(E,winner,dict(method='FIXED_8_TWO_VEHICLES_256',candidate_budget=256,
            initial_assignment=initial.tolist(),customers=nodes.tolist(),vehicles=[a,b],selection_version='within_two_vehicles',scope='固定两车内8客户的全部256种分配；不代表完整多轮局部算法')))
    save(out,dict(instance=name,seed=sd,rows=rows,ledger=E.ledger()))

def parallel(function,items):
    with ProcessPoolExecutor(max_workers=4) as pool:
        for f in as_completed([pool.submit(function,x) for x in items]):f.result()

def main():
    mode=sys.argv[1] if len(sys.argv)>1 else 'all'
    models=list((HERE/'models').glob('*/*/*/result.json'))
    if mode in ('raw','all'):parallel(raw_job,models)
    if mode in ('post','all'):
        # 核验新加中间记录没有改变统一精修行为。
        S=setup('B-n31-k5');v=np.random.default_rng(9981).integers(5,size=30,dtype=np.int64)
        a,_,_=refine_stages(S,v);b,_=refine(S,v);assert np.array_equal(a,b)
        parallel(postprocess_job,models)
    if mode in ('bias','all'):parallel(bias_job,['B-n31-k5','A-n45-k7'])
    if mode in ('attribution','all'):
        chosen=read(HERE/'PCE_GLOBAL_V4_FINAL.json')['implementation']
        subset=[p for p in models if p.parent.name.startswith(chosen+'_') and p.parents[2].name in ('confirmation','scale')]
        parallel(attribution_job,subset)
    if mode in ('classical','all'):
        parallel(baseline_job,[(n,s) for n,seeds in [('B-n31-k5',range(2001,2021)),('A-n45-k7',range(3001,3011))] for s in seeds])

if __name__=='__main__':main()
