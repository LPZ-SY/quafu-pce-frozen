"""各组共用种子、测量流、精修预算和停止条件的最终比较。"""
from final_objective import *
from concurrent.futures import ProcessPoolExecutor,as_completed

def test_job(path):
    path=Path(path);m=read(path);name=m['instance'];sd=m['seed'];method=m['method']
    cohort='development' if sd<2000 else 'confirmation' if sd<3000 else 'scale'
    out=HERE/'final_tests'/cohort/name/f'{method}_{sd}.json'
    if out.exists():return
    E=FinalEvaluator(setup(name),method,m['nq'],m['layers']);theta=np.array(m['theta']);start=time.perf_counter();allrows=[];stats={}
    for shots in (1024,4096,16384):
        stream=seed('unified_final_test',name,sd,shots)
        s,rows,counts,e,values=E.validate(theta,np.random.default_rng(stream),batches=64,shots=shots)
        stats[str(shots)]=s
        for r in rows:r.update(instance=name,seed=sd,method=method,cohort=cohort,stream_seed=stream)
        allrows+=rows
        p=out.parent/f'{method}_{sd}_{shots}.npz';p.parent.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(p,counts=counts,scores=values,reference_scores=e,shots_per_basis=allocation(shots))
    save(out,dict(model=str(path.relative_to(HERE)),model_sha256=digest(path),instance=name,method=method,seed=sd,cohort=cohort,
                   summary=stats,rows=allrows,ledger=E.ledger(),seconds=time.perf_counter()-start))
    print('统一最终测试完成',cohort,name,method,sd,stats['4096'],flush=True)

def attribution_job(path):
    path=Path(path);m=read(path);name=m['instance'];sd=m['seed'];out=HERE/'final_attribution'/name/f'{sd}.json'
    if out.exists():return
    cohort='confirmation' if sd<3000 else 'scale';E=FinalEvaluator(setup(name),'FINAL_JOINT',m['nq']);n,k=E.S['n'],E.S['K']
    data=np.load(HERE/'final_tests'/cohort/name/f'FINAL_JOINT_{sd}_4096.npz');trained=data['scores'];ref=E.decode(data['reference_scores'])
    rng=np.random.default_rng(seed('final_attribution',name,sd));uniform=np.full((3,1<<E.nq),1/(1<<E.nq))
    null,counts=sample(uniform,E.fams,E.masks,rng,batches=64);perms=np.stack([rng.permutation(n) for _ in range(20)])
    rows=[]
    for batch in range(64):
        entries=[('NULL_READOUT',0,null[batch])]+[('SHUFFLED_ROWS',i,trained[batch].reshape(n,k)[p].ravel()) for i,p in enumerate(perms)]
        for info,perm,values in entries:
            v=E.decode(values)
            # 20次行置换均保留原始分车，预定置换0及空信息输入接受完整统一精修。
            row=dict(batch=batch,information=info,permutation=perm,assignment=v.tolist(),**E.assess(v),customer_flip_rate=float(np.mean(v!=ref)))
            if perm==0:_,h=E.hybrid(v);row.update(h)
            rows.append(row)
    out.parent.mkdir(exist_ok=True,parents=True);np.savez_compressed(out.with_suffix('.npz'),null_counts=counts,permutations=perms)
    save(out,dict(instance=name,seed=sd,rows=rows,ledger=E.ledger(),note='每模型64份完整读出；置换重复不是新模型。精修仅用于预定置换0。'))

def mapping_job(args):
    name,sd,ms=args;out=HERE/'mapping_sensitivity'/name/f'{sd}_{ms}.json'
    if out.exists():return
    # 固定参数映射扰动仅用于机制敏感性，不以结果选映射，也不冒称重新训练。
    phase='final_confirmation' if sd<3000 else 'final_scale';m=read(HERE/'models'/phase/name/f'FINAL_JOINT_{sd}'/'result.json')
    E=FinalEvaluator(setup(name),'FINAL_JOINT',m['nq']);E.fams,E.masks,E.priorities=mapping(E.S['n'],E.S['K'],E.nq,False,ms)
    s,rows,counts,e,values=E.validate(np.array(m['theta']),np.random.default_rng(seed('mapping_sensitivity',name,sd,ms)),batches=64)
    save(out,dict(instance=name,seed=sd,mapping_seed=ms,summary=s,rows=rows,
        limitation='冻结参数下更换映射的敏感性；没有重新训练，不能解释为该映射的最佳可达性能。'))

def parallel(fn,args):
    with ProcessPoolExecutor(max_workers=4) as pool:
        for f in as_completed([pool.submit(fn,x) for x in args]):f.result()

def main():
    mode=sys.argv[1] if len(sys.argv)>1 else 'test'
    models=list((HERE/'models').glob('*/*/*/result.json'))
    if mode=='test':parallel(test_job,models)
    if mode=='attribution':parallel(attribution_job,[p for p in models if p.parent.name.startswith('FINAL_JOINT_') and p.parents[2].name in ('final_confirmation','final_scale')])
    if mode=='mapping':parallel(mapping_job,[(name,sd,ms) for name,seeds in [('B-n31-k5',range(2001,2004)),('A-n45-k7',range(3001,3004))] for sd in seeds for ms in (20261016,20261017)])

if __name__=='__main__':main()
