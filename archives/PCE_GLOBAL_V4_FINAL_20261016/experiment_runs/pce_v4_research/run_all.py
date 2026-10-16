"""从正式实例执行首轮、一次结构修改、冻结、独立确认和规模验证。"""
import os
for _name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[_name]='1'
import sys, json, shutil, time, platform, importlib.metadata
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
import yaml
from core import *

METHODS=['LEGACY','NEW_DECODER','NEW_JOINT','RANDOM_THETA']

def prepare():
    sources=Path('C:/Users/吕佩哲/Downloads')
    for name in ('PCE_CVRP_v3_Codex_实验执行书.md','PCE_CVRP_v4_config.template.yaml'):
        dest=HERE/'protocol_sources'/name; dest.parent.mkdir(exist_ok=True)
        if dest.exists(): assert digest(dest)==digest(sources/name)
        else: shutil.copy2(sources/name,dest)
    cfg=yaml.safe_load((sources/'PCE_CVRP_v4_config.template.yaml').read_text(encoding='utf-8'))
    sys.path.insert(0,str(HERE.parent/'pce_set_partition_20261014'))
    import pipeline as SP
    previous=read(HERE.parent/'pce_stable_controls_20261015/inputs.json')
    instances={}
    for name in ('B-n31-k5','A-n45-k7'):
        s=SP.load(name); old=previous['instances'][name]
        assert s['n']==len(s['q'])-1 and s['q'][0]==0
        assert np.array_equal(s['D'],old['D']) and np.array_equal(s['q'][1:],old['q'])
        assert s['n']==(30 if name=='B-n31-k5' else 44)
        x={k:(v.tolist() if isinstance(v,np.ndarray) else v) for k,v in s.items()}
        x['q']=x['q'][1:]
        raw=SP.DATA/(name+'.vrp'); x['source_file']=str(raw);x['source_sha256']=digest(raw)
        save(HERE/'instances'/(name+'.json'),x)
        instances[name]=dict(n=s['n'],K=s['K'],capacity=s['cap'],total_demand=int(s['q'].sum()),source_sha256=digest(raw))
    protocol=dict(config=cfg,instances=instances,source_hashes={p.name:digest(p) for p in (HERE/'protocol_sources').iterdir()},
        resolution={
            'documents':'用户确认第三版执行书与第四版配置为本轮全部输入；不存在可读取的第四版执行书。',
            'branch_labels':'不冒称文档存在A—F标签；采用第三版第14节故障规则并保存数据触发的明确修改。',
            'legacy':'完整历史20,001次路线查询加30,001次稳定性查询；其额外预算单列，不宣称与新组等成本。',
            'primary':'新方法8比特4层64参数；44客户新表示用9比特4层72参数。旧表示均保留8比特。',
            'new_methods':'4096次目标查询；每次1份参考和4份测量分车；随机参数不训练，另加4096次随机参数搜索。',
            'margin':'执行第三版协方差感知标准化差距目标z=2；第四版margin_target=0.10仅保留为原始差距诊断阈值，不擅自替换第三版公式。',
            'validation':'每512次保存训练最优参数；每候选16份独立验证读出；按可行率、超载、可行路程、翻车率顺序选。',
            'test':'冻结模型后每档1024、4096、16384次测量各64批；测试结果不参与修改选择。',
            'confirmation':'30客户确认使用全新种子2001—2020；44客户使用3001—3010；重复种子不冒称新实例。',
            'revision':'一次结构改造：新解码有收益但联合不足用连续容量项；容量提高而路程恶化超过10%用几何项；普遍失败则加深至6层并配套等深对照。',
            'freeze':'初轮与改造的验证集汇总按预定排序选择模型规则；最终确认结果不用于改写规则。',
            'hardware':'模拟规则冻结后再检查既有权限；全部新真机数据必须有独立任务及原始计数；无权限时只导出任务包。'},
        software={p:importlib.metadata.version(p) for p in ('numpy','numba','scipy','PyYAML','ortools','matplotlib')},python=sys.version)
    target=HERE/'protocol.json'
    if target.exists(): assert read(target)==protocol,'协议已有不同版本，禁止覆盖'
    else: save(target,protocol)
    audit='''正式实验输入审计（2026年10月15日）
使用用户确认的第三版执行书和第四版配置。第四版科研推进执行书未提供，A—F标签不作杜撰。
容量约束车辆路径问题（Capacitated Vehicle Routing Problem，CVRP）的30客户实例为B-n31-k5，非A-n32-k5（31客户）。44客户实例为A-n45-k7。
两者容量均100；总需求分别412和634；仓库需求为0并从客户索引排除。欧氏距离先平方和开根，再floor(d+0.5)取整。
泡利关联编码（Pauli Correlation Encoding，PCE）旧表示为每客户3项三体同轴关联，符号>0置1、零值置0，三位整数对车辆数取余。
旧三位均匀码对5车计数为2、2、2、1、1，故标签概率不均；这不是全部容量损失的因果归因。
新表示使用不重复的二体、三体同轴槽位；车辆列按轴、阶数和物理比特使用计数平衡。映射不读需求、坐标或测试成绩。
物理比特0对应状态整数最高位；Y基先施加逆相位门再施加哈达玛门；同基所有分数从同一完整位串计数恢复。
旧实验每方向4096次、三方向合计12288次；新实验每完整读出合计4096次，X=1366，Y=1365，Z=1365。
旧自主搜索20,000次更新加初始评价，稳定性修正30,000次更新加初始评价；后者固定全部关联符号。新联合训练从第一步允许客户换车。
原始分车仅用固定最近邻及车内路径改进计算距离；共享refine包含跨车移动与交换，必须另列最终结果，不可混入原始可行率。
旧训练参数、映射、随机种子、计数和任务台账存在，但旧结果不是本轮同预算证据；图中53、75、78及36份不可行本轮不采用为新结果。
新结果目录与旧实验隔离；尚无任何第四版真机结果。没有创建经典客户×车辆自由偏置表，也不使用老师标签。
'''
    (HERE/'audit.txt').write_text(audit,encoding='utf-8')
    for name in instances:
        S=setup(name);nq=8 if S['n']==30 else 9
        for old in (False,True):
            q=8 if old else nq; f,m,p=mapping(S['n'],S['K'],q,old)
            save(HERE/'mappings'/f'{name}_{"legacy" if old else "new"}.json',
                 dict(nq=q,axis_order='ZXY',fams=f.tolist(),masks=m.tolist(),tie_priorities=p.tolist(),mapping_seed=20261014 if old else 20261015))
    print('输入冻结完成',instances,flush=True)

def verify():
    sys.path.insert(0,str(HERE.parent/'pce_dongling_direct4_20261014'))
    import pce_k as K
    rng=np.random.default_rng(78645); errors=[]
    for name in ('B-n31-k5','A-n45-k7'):
        S=setup(name);nq=8 if S['n']==30 else 9
        for old in (False,True):
            q=8 if old else nq;E=Evaluator(S,'LEGACY' if old else 'NEW_JOINT',q)
            th=rng.uniform(-np.pi,np.pi,8*q); pp=probabilities(th,q,4)
            slots=[('ZXY'[f],tuple(i for i in range(q) if m&(1<<(q-1-i)))) for f,m in zip(E.fams,E.masks)]
            c=K.CircuitK(q,4,True).prepare(slots); reference=c.probs(c.run(th))
            error=max(float(np.max(np.abs(pp[i]-reference[f]))) for i,f in enumerate('ZXY'))
            assert error<1e-12;errors.append(error)
            val=scores_from_moments(walsh(pp),E.fams,E.masks)
            assert np.allclose(val,c.expect_k(c.run(th)),atol=1e-12)
            v=E.decode(val);assert len(v)==S['n'] and np.all((v>=0)&(v<S['K']))
            if not old:
                order=rng.permutation(S['K']); x=val.reshape(S['n'],S['K'])
                perm=decode(x[:,order].ravel(),S['n'],S['K'],False,E.priorities[:,order])
                assert np.array_equal(order[perm],v)
                zeros=E.decode(np.zeros_like(val));assert len(np.unique(zeros))>1
            vals,counts=sample(pp,E.fams,E.masks,rng,batches=4)
            assert np.all(counts.sum(axis=2)==allocation(4096))
            # 对完整计数逐位计算乘积，独立核对快速变换和位序。
            signs=np.array([[(-1)**bin(int(w&m)).count('1') for w in range(1<<q)] for m in E.masks])
            ref=np.array([(counts[0,f]*sg).sum()/allocation(4096)[f] for f,sg in zip(E.fams,signs)])
            assert np.allclose(vals[0],ref,atol=1e-12)
    assert np.bincount(np.arange(8)%5).tolist()==[2,2,2,1,1]
    save(HERE/'verification.json',dict(passed=True,formal_instances=['B-n31-k5','A-n45-k7'],
         probability_errors=errors,checks=['独立线路模拟','Y基与位序','计数奇偶乘积','车辆置换等价','全零并列可复现','旧取余频次','总测量预算']))
    print('正式规模实现核验通过',max(errors),flush=True)

def train_job(args):
    phase,name,method,sd,layers=args
    folder=HERE/'models'/phase/name/f'{method}_{sd}';result=folder/'result.json'
    if result.exists(): return str(result)
    folder.mkdir(parents=True,exist_ok=True);S=setup(name)
    nq=8 if S['n']==30 or method=='LEGACY' else 9
    actual='NEW_JOINT' if method=='DEEP_JOINT' else 'NEW_DECODER' if method=='DEEP_DECODER' else method
    E=Evaluator(S,actual,nq,layers)
    rng=np.random.default_rng(seed('optimizer',sd,nq,layers))
    init=np.random.default_rng(seed('initial',sd,nq,layers)).uniform(-np.pi,np.pi,2*nq*layers)
    shot_rng=np.random.default_rng(seed('train_readout',phase,name,sd))
    checkpoints=[];trace=[];start=time.perf_counter();extra={}
    if method=='LEGACY':
        sys.path.insert(0,str(HERE.parent/'pce_random_value_20261014'))
        import run as old
        import stability_remote as stable
        # 旧接口使用其原生随机数初始化；完整保留，不冒充单因素消融。
        th,ev,v,unique,same,oldtrace=old.train(S['inst']['D'],S['q'],S['cap'],S['K'],E.fams,E.masks,sd,20000)
        checkpoints.append((20001,th.copy()))
        fixed,ev,accepted,valid=stable.improve(th,E.fams,E.masks,seed('legacy_stability',sd,name))
        checkpoints=[(50002,fixed)]
        trace=[dict(query=int(x[0])+1,reference_overload=float(x[1]),reference_distance=float(x[2])) for x in oldtrace]
        extra=dict(legacy_route_queries=20001,legacy_stability_queries=30001,legacy_unique_routes=int(unique),
                   legacy_unchanged_proposals=int(same),legacy_stability_valid=int(valid),legacy_stability_accepted=int(accepted))
    elif method=='RANDOM_THETA':
        checkpoints=[(0,init)]
    else:
        theta=init.copy();loss,diag=E.objective(theta,shot_rng);best=loss;best_theta=theta.copy()
        for t in range(1,4096):
            frac=t/4096
            if method=='RANDOM_SEARCH': proposal=rng.uniform(-np.pi,np.pi,len(theta))
            else:
                proposal=theta.copy();j=rng.integers(len(theta))
                proposal[j]+=rng.normal(0,np.pi*(.5*(1-frac)+.03))
            new,diag=E.objective(proposal,shot_rng)
            temperature=.01*(.00001/.01)**frac
            if new<=loss or rng.random()<np.exp(min(0,(loss-new)/temperature)):
                theta,loss=proposal,new
            if new<best: best,best_theta=new,proposal.copy()
            if (t+1)%64==0:
                trace.append(dict(query=t+1,training_best_loss=float(best),current_loss=float(loss),
                    reference_overload=diag['ref']['overload'],reference_distance=diag['ref']['distance'],
                    measured_feasible=diag['measured_feasible'],measured_overload=diag['measured_overload'],margin=diag['margin']))
            if (t+1)%512==0:
                checkpoints.append((t+1,best_theta.copy()))
                save(folder/'progress.json',dict(query=t+1,best_loss=float(best),seconds=time.perf_counter()-start))
                np.savez_compressed(folder/f'checkpoint_{t+1}.npz',theta=best_theta)
    training_ledger=E.ledger();training_seconds=time.perf_counter()-start
    if method=='LEGACY': training_ledger.update(objective_queries=50002,state_simulations=50002,**extra)
    validation=[]
    for query,theta in checkpoints:
        vr,vc,_,_=E.readouts(theta,np.random.default_rng(seed('validation',phase,name,sd,query)),batches=16)
        validation.append(dict(query=query,summary=summary(vr),rows=vr,theta=theta.tolist()))
        np.savez_compressed(folder/f'validation_{query}.npz',counts=vc)
    winner=min(range(len(validation)),key=lambda i:(rank(validation[i]['summary']),validation[i]['query']))
    best=validation[winner];theta=np.array(best['theta']);ref=E.decode(scores_from_moments(walsh(probabilities(theta,nq,layers)),E.fams,E.masks))
    metadata=dict(phase=phase,instance=name,method=method,seed=sd,nq=nq,layers=layers,parameters=len(theta),theta=theta.tolist(),
        initial_theta=init.tolist() if method!='LEGACY' else None,mapping_fams=E.fams.tolist(),mapping_masks=E.masks.tolist(),
        tie_priorities=E.priorities.tolist(),selected_query=best['query'],validation_summary=best['summary'],
        validation=validation,reference_assignment=ref.tolist(),reference=E.assess(ref),trace=trace,
        training_ledger=training_ledger,training_seconds=training_seconds,wall_seconds=time.perf_counter()-start,
        instance_sha256=digest(HERE/'instances'/(name+'.json')),protocol_sha256=digest(HERE/'protocol.json'),code_sha256=digest(HERE/'core.py'))
    save(result,metadata)
    print('完成',phase,name,method,sd,best['summary'],round(training_seconds,1),flush=True)
    return str(result)

def jobs(phase,name,methods,seeds):
    work=[(phase,name,m,int(s),6 if m.startswith('DEEP_') else 4) for m in methods for s in seeds]
    with ProcessPoolExecutor(max_workers=4) as pool:
        for f in as_completed([pool.submit(train_job,x) for x in work]): f.result()

def method_summary(phase,name,method):
    rows=[read(p)['validation_summary'] for p in (HERE/'models'/phase/name).glob(method+'_*/result.json')]
    out={k:float(np.mean([r[k] for r in rows if r[k] is not None])) if any(r[k] is not None for r in rows) else None for k in rows[0]}
    out['models']=len(rows);return out

def decide_revision():
    name='B-n31-k5';d={m:method_summary('round1',name,m) for m in METHODS}
    old,new,joint,rnd=[d[m] for m in METHODS]
    if joint['raw_feasibility_rate']>new['raw_feasibility_rate'] and joint['feasible_raw_route_distance'] is not None and new['feasible_raw_route_distance'] is not None and joint['feasible_raw_route_distance']>1.1*new['feasible_raw_route_distance']:
        revision='V42_GEOMETRY';reason='联合训练容量可行率提高，但可行条件路线恶化超过预设10%；增加低权重几何项并保留连续容量项。'
    elif rank(new)<rank(old) and not rank(joint)<rank(new):
        revision='V41_SOFT_CAPACITY';reason='新解码验证排序改善，而联合训练未超过新解码；增加温度0.25的连续容量项，保持量子参数独立搜索。'
    elif joint['raw_feasibility_rate']<.5 or rank(joint)>=rank(rnd):
        revision='DEEP_JOINT';reason='原始可行率低于50%或未优于随机参数；将线路由4层加深到6层，另跑同深度新解码对照。'
    else:
        revision='V41_SOFT_CAPACITY';reason='联合训练已有收益；仍以连续容量项检验有限测量下剩余超载是否可进一步下降，若验证排序不改善则不采用。'
    result=dict(revision=revision,reason=reason,evidence=d,decision_data='独立验证读出；不读取最终测试',
                source='第三版第14节失败诊断与第四版revision_v41/revision_v42；阈值及补充分支为本轮提前冻结的实现规则')
    save(HERE/'revision_decision.json',result);print('结构修改',result,flush=True);return revision

def freeze(revision):
    methods={m:method_summary('round1','B-n31-k5',m) for m in ('NEW_DECODER','NEW_JOINT')}
    methods[revision]=method_summary('revision','B-n31-k5',revision)
    chosen=min(methods,key=lambda m:rank(methods[m]))
    final=dict(name='PCE_GLOBAL_V4_FINAL',implementation=chosen,layers=6 if chosen.startswith('DEEP_') else 4,
               evidence=methods,selection='仅按模型规则在初轮验证集上的预定排序；新种子确认不再修改规则',
               revision_adopted=chosen==revision,confirmation_seeds=list(range(2001,2021)),scale_seeds=list(range(3001,3011)))
    save(HERE/'PCE_GLOBAL_V4_FINAL.json',final);print('规则冻结',final,flush=True);return chosen

def main():
    mode=sys.argv[1] if len(sys.argv)>1 else 'all'
    if mode in ('prepare','all'): prepare();verify()
    if mode=='prepare':return
    if mode in ('round1','all'):
        jobs('round1','B-n31-k5',METHODS,range(1001,1011))
    if mode in ('advance','all'):
        revision=decide_revision()
        methods=[revision]+(['DEEP_DECODER'] if revision=='DEEP_JOINT' else [])
        jobs('revision','B-n31-k5',methods,range(1001,1011))
        chosen=freeze(revision)
        jobs('confirmation','B-n31-k5',list(dict.fromkeys([chosen,'NEW_DECODER','NEW_JOINT','RANDOM_THETA','RANDOM_SEARCH','LEGACY'])),range(2001,2021))
        jobs('scale','A-n45-k7',list(dict.fromkeys([chosen,'NEW_DECODER','NEW_JOINT','RANDOM_THETA','RANDOM_SEARCH','LEGACY'])),range(3001,3011))
    if mode=='verify':verify()

if __name__=='__main__':main()
