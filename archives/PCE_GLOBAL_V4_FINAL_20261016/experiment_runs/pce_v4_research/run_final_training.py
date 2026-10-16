"""按用户更新的最终可行硬门槛，直接训练经典精修后目标。"""
from final_objective import *
from concurrent.futures import ProcessPoolExecutor,as_completed

def prepare_protocol():
    old=HERE/'PCE_GLOBAL_V4_FINAL.json';archive=HERE/'raw_priority_frozen_model.json'
    if not archive.exists():save(archive,read(old))
    protocol=dict(version='final_feasibility_hard_gate',source='用户在首轮之后明确更新研究目标，覆盖旧文档的原始可行优先要求。',
        formula='H(a)=I_final_feasible*D_final/D_scale + I_final_failed*(1000000+30*O_final/Q) + 0.35*O_raw/Q + 0.10*N_changed/N；L=0.35*H(reference)+0.55*mean(H(4 readouts))+0.10*mean(customer_flip_rate)',
        hard_gate='最终不可行分配记失败，不以其路线长度排名；最终可行分配才计算路线均值、配对距离与差距。失败率作为首要门槛并始终报告。',
        weights=dict(reference=.35,measurement=.55,instability=.10,raw_overload=.35,changed_customers=.10,final_failure=1000000),
        refinement=dict(algorithm='与原shared.refine相同的确定性最优改进：单客户跨车移动及两客户交换',candidate_budget=10000,
                        stop='无严格改善移动或候选评价数达到10000',rng='精修器本身不使用随机数',
                        cache='完整分配及车内路线可缓存；缓存只复用同一确定性结果，不改算法或预算，真实调用数单列'),
        groups=['FINAL_JOINT','FINAL_RANDOM_SEARCH','RANDOM_THETA','LEGACY','NEW_DECODER','NEW_JOINT','V42_GEOMETRY'],
        optimizer='复用模拟退火单参数扰动；随机搜索每次重新从相同初始化分布抽取全部参数',
        seeds=dict(development=list(range(1001,1011)),confirmation=list(range(2001,2021)),scale=list(range(3001,3011))),
        budgets=dict(objective_queries=4096,readout_batches_per_query=4,shots_per_full_readout=4096,validation_batches=16,test_batches=64),
        selection='验证最终失败率最低→带惩罚目标最低→最终可行路线最短→原始超载最低→净跨车改动最少→翻车率最低→更早检查点',
        equality='新训练与随机搜索使用相同初始参数、查询数、完整测量流和精修规则；旧组原训练预算单列，其测试精修完全相同。',
        final_claim='仅当配对置信区间支持时，声称训练相对随机参数降低原始超载、精修改动并缩短最终可行路线；不预写成功。')
    save(HERE/'final_objective_protocol.json',protocol)

def final_rank(s):
    return (-s['final_feasibility_rate'],s['hybrid_loss'],s['feasible_final_distance'] if s['feasible_final_distance'] is not None else float('inf'),
            s['raw_mean_overload'],s['mean_changed_customers'],s['customer_flip_rate'])

def job(args):
    phase,name,method,sd=args;folder=HERE/'models'/phase/name/f'{method}_{sd}';out=folder/'result.json'
    if out.exists():return str(out)
    folder.mkdir(exist_ok=True,parents=True);S=setup(name);nq=8 if S['n']==30 else 9
    E=FinalEvaluator(S,method,nq);rng=np.random.default_rng(seed('optimizer',sd,nq,4));shot_rng=np.random.default_rng(seed('final_train_readout',name,sd))
    initial=np.random.default_rng(seed('initial',sd,nq,4)).uniform(-np.pi,np.pi,8*nq);theta=initial.copy()
    start=time.perf_counter();loss,diag=E.objective(theta,shot_rng);best=loss;best_theta=theta.copy();trace=[];checkpoints=[]
    for t in range(1,4096):
        frac=t/4096
        if method=='FINAL_RANDOM_SEARCH':candidate=rng.uniform(-np.pi,np.pi,len(theta))
        else:
            candidate=theta.copy();j=rng.integers(len(theta));candidate[j]+=rng.normal(0,np.pi*(.5*(1-frac)+.03))
        value,diag=E.objective(candidate,shot_rng);temperature=.01*(.00001/.01)**frac
        if value<=loss or rng.random()<np.exp(min(0,(loss-value)/temperature)):theta,loss=candidate,value
        if value<best:best,best_theta=value,candidate.copy()
        if (t+1)%64==0:trace.append(dict(query=t+1,training_best_loss=float(best),current_loss=float(loss),**diag))
        if (t+1)%512==0:
            checkpoints.append((t+1,best_theta.copy()));np.savez_compressed(folder/f'checkpoint_{t+1}.npz',theta=best_theta)
            save(folder/'progress.json',dict(query=t+1,best_loss=float(best),seconds=time.perf_counter()-start,ledger=E.ledger()))
            print('精修目标训练',name,method,sd,t+1,round(best,6),round(time.perf_counter()-start,1),flush=True)
    training_ledger=E.ledger();elapsed=time.perf_counter()-start;validation=[]
    for query,th in checkpoints:
        s,rows,counts,_,_=E.validate(th,np.random.default_rng(seed('final_validation',name,sd,query)))
        validation.append(dict(query=query,summary=s,rows=rows,theta=th.tolist()))
        np.savez_compressed(folder/f'validation_{query}.npz',counts=counts)
    winner=min(validation,key=lambda r:(final_rank(r['summary']),r['query']));theta=np.array(winner['theta'])
    ref=E.decode(scores_from_moments(walsh(probabilities(theta,nq,4)),E.fams,E.masks))
    save(out,dict(phase=phase,instance=name,method=method,seed=sd,nq=nq,layers=4,parameters=len(theta),theta=theta.tolist(),
        initial_theta=initial.tolist(),mapping_fams=E.fams.tolist(),mapping_masks=E.masks.tolist(),tie_priorities=E.priorities.tolist(),
        selected_query=winner['query'],validation_summary=winner['summary'],validation=validation,reference_assignment=ref.tolist(),
        reference=E.assess(ref),trace=trace,training_ledger=training_ledger,training_seconds=elapsed,wall_seconds=time.perf_counter()-start,
        instance_sha256=digest(HERE/'instances'/(name+'.json')),protocol_sha256=digest(HERE/'final_objective_protocol.json'),code_sha256=digest(HERE/'final_objective.py')))
    print('精修目标完成',phase,name,method,sd,winner['summary'],round(elapsed,1),flush=True)
    return str(out)

def run_jobs(phase,name,methods,seeds):
    work=[(phase,name,m,s) for m in methods for s in seeds]
    with ProcessPoolExecutor(max_workers=4) as pool:
        for future in as_completed([pool.submit(job,x) for x in work]):future.result()

def freeze():
    rows=[read(p) for p in (HERE/'models'/'final_development'/'B-n31-k5').glob('FINAL_JOINT_*/result.json')]
    assert len(rows)==10
    save(HERE/'PCE_GLOBAL_V4_FINAL.json',dict(name='PCE_GLOBAL_V4_FINAL',implementation='FINAL_JOINT',layers=4,
        objective='最终可行硬门槛下的统一精修后路线，并惩罚原始超载、净跨车改动、有限测量不稳定性',
        protocol_sha256=digest(HERE/'final_objective_protocol.json'),development_models=10,confirmation_seeds=list(range(2001,2021)),
        scale_seeds=list(range(3001,3011)),status='规则冻结；独立确认结果另报，不以模型名称代表通过成功条件',
        supersedes='raw_priority_frozen_model.json',user_requested_revision=True))

def main():
    mode=sys.argv[1] if len(sys.argv)>1 else 'all';prepare_protocol()
    if mode=='scale_joint':
        run_jobs('final_scale','A-n45-k7',['FINAL_JOINT'],range(3001,3011));return
    if mode in ('development','all'):
        run_jobs('final_development','B-n31-k5',['FINAL_JOINT'],range(1001,1011));freeze()
    if mode in ('confirmation','all'):
        run_jobs('final_confirmation','B-n31-k5',['FINAL_JOINT','FINAL_RANDOM_SEARCH'],range(2001,2021))
        run_jobs('final_scale','A-n45-k7',['FINAL_JOINT','FINAL_RANDOM_SEARCH'],range(3001,3011))

if __name__=='__main__':main()
