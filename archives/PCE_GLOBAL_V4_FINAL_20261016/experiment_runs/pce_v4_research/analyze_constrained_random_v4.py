"""在disco汇总新增约束内随机对照，保留冻结量子模型不变。"""
from core import *
from report import write_table
import socket,tarfile,collections
OUT=HERE/'constrained_random_v4'
BUDGETS=(1,4,16,64,256,4096)

def effect(values,rng):
    d=np.asarray(values,float);n=len(d)
    ci=np.quantile(d[rng.integers(0,n,size=(20000,n))].mean(axis=1),[.025,.975])
    if n<=10:
        patterns=np.arange(1<<n)[:,None];signs=1-2*((patterns>>np.arange(n))&1)
        p=float(np.mean(np.abs((signs*d).mean(axis=1))>=abs(d.mean())-1e-12))
    else:
        signs=rng.choice([-1,1],size=(100000,n))
        p=float((1+np.sum(np.abs((signs*d).mean(axis=1))>=abs(d.mean())-1e-12))/100001)
    return dict(n=n,mean=float(d.mean()),ci95=ci.tolist(),p_two_sided=p)

def main():
    assert socket.gethostname()=='lenovo'
    assert read(OUT/'completion.json')['complete'];protocol=read(OUT/'protocol.json')
    allgroups={};paired={};tables=[];audit=[];raw_records=[]
    for name,seeds in protocol['instances'].items():
        S=setup(name);cohort='confirmation' if S['n']==30 else 'scale'
        quantum={sd:read(HERE/'final_tests'/cohort/name/f'FINAL_JOINT_{sd}.json') for sd in seeds}
        baseline={sd:read(OUT/'results'/name/f'{sd}.json') for sd in seeds}
        qrows={sd:[r for r in quantum[sd]['rows'] if r['shots']==4096] for sd in seeds}
        qdist=float(np.mean([np.mean([r['final']['distance'] for r in rr if r['final']['feasible']]) for rr in qrows.values()]))
        tables.append(dict(实例=name,方法='训练后的量子流程',候选分配预算=None,独立种子数=len(seeds),每种子批数=64,
            构造成功率=1.,精修前可行率=float(np.mean([r['raw']['feasible'] for rr in qrows.values() for r in rr])),
            精修前平均超载=float(np.mean([r['raw']['overload'] for rr in qrows.values() for r in rr])),
            最终可行率=float(np.mean([r['final']['feasible'] for rr in qrows.values() for r in rr])),
            最终可行路线均值=qdist,平均精修改动人数=float(np.mean([r['changed_customers'] for rr in qrows.values() for r in rr])),
            平均精修候选评价数=float(np.mean([r['refinement_candidate_evaluations'] for rr in qrows.values() for r in rr])),
            预算说明='每模型先训练4096次查询，每查询参考及4份有限测量精修；测试每批4096次测量后精修一次。'))
        for sd,d in baseline.items():
            assert d['source_sha256']==digest(HERE/'constrained_random_v4.py')
            assert d['instance_sha256']==digest(HERE/'instances'/(name+'.json'))
            assert d['protocol_sha256']==digest(OUT/'protocol.json')
            assert len(d['rows'])==64*len(BUDGETS)
            arrays=np.load(OUT/'results'/name/f'{sd}.npz');dd=arrays['candidate_distances'];tt=arrays['construction_attempts']
            assert dd.shape==tt.shape==(64,4096) and np.all((tt>=1)&(tt<=100))
            for r in d['rows']:
                assert r['quantum_measurements']==0
                assert r['stream_seed']==seed('constrained_random_v4',name,sd,r['batch'])
                prefix=dd[r['batch'],:r['candidate_budget']];attempts=tt[r['batch'],:r['candidate_budget']]
                assert r['construction_attempts']==int(attempts.sum()) and r['failed_candidate_constructions']==int((~np.isfinite(prefix)).sum())
                if r['construction_failed']:
                    assert not np.any(np.isfinite(prefix)) and not r['final']['feasible'];continue
                assert np.min(prefix)==r['raw']['distance'] and r['raw']['overload']==0
                for stage,field in [('raw','assignment'),('final','final_assignment')]:
                    v=np.array(r[field]);assert len(v)==S['n'] and np.all((v>=0)&(v<S['K']))
                    over=float(np.maximum(0,np.bincount(v,weights=S['q'],minlength=S['K'])-S['cap']).sum())
                    assert over==r[stage]['overload'] and bool(over==0)==r[stage]['feasible']
                    if r['batch']==0:
                        distance=sum(route_cost(S['inst']['D'],(np.flatnonzero(v==k)+1).astype(np.int64)) for k in range(S['K']))
                        assert distance==r[stage]['distance']
                assert 0<=r['refinement_candidate_evaluations']<=10000
                assert np.sum(np.array(r['assignment'])!=r['final_assignment'])==r['changed_customers']
            raw_records.extend(d['rows'])
            audit.append(dict(instance=name,seed=sd,rows=len(d['rows']),candidate_draws=64*4096,
                failed_candidate_constructions=int(np.sum(~np.isfinite(dd))),construction_attempts=int(tt.sum())))
        for budget in BUDGETS:
            rr={sd:[r for r in baseline[sd]['rows'] if r['candidate_budget']==budget] for sd in seeds}
            assert all(sorted(r['batch'] for r in v)==list(range(64)) for v in rr.values())
            pairs=[]
            for sd in seeds:
                qr={r['batch']:r for r in qrows[sd]};br={r['batch']:r for r in rr[sd]}
                common=[(qr[i],br[i]) for i in range(64) if qr[i]['final']['feasible'] and br[i]['final']['feasible']]
                assert common,'共同可行样本为空，应保留失败另行处理'
                pairs.append(dict(seed=sd,common_feasible_batches=len(common),
                    final_distance=float(np.mean([a['final']['distance']-b['final']['distance'] for a,b in common])),
                    changed_customers=float(np.mean([a['changed_customers']-b['changed_customers'] for a,b in common]))))
            rng=np.random.default_rng(seed('constrained_v4_stats',name,budget))
            e=effect([p['final_distance'] for p in pairs],rng)
            e['per_seed']=pairs;paired[f'{name}:{budget}']=e
            flat=[r for v in rr.values() for r in v];good=[r for r in flat if r['final']['feasible']]
            mean=float(np.mean([r['final']['distance'] for r in good])) if good else None
            tables.append(dict(实例=name,方法=f'约束内随机{budget}份选1',候选分配预算=budget,独立种子数=len(seeds),每种子批数=64,
                构造成功率=float(np.mean([not r['construction_failed'] for r in flat])),
                精修前可行率=float(np.mean([r['raw'] is not None and r['raw']['feasible'] for r in flat])),精修前平均超载=0. if all(r['raw'] for r in flat) else None,
                最终可行率=len(good)/len(flat),最终可行路线均值=mean,
                平均精修改动人数=float(np.mean([r['changed_customers'] for r in good])) if good else None,
                平均精修候选评价数=float(np.mean([r.get('refinement_candidate_evaluations',0) for r in flat])),
                平均构造尝试次数=float(np.mean([r['construction_attempts'] for r in flat])),
                平均生成与选优秒数=float(np.mean([r['generation_seconds'] for r in flat])),
                量子减随机路线差=e['mean'],路线差95区间=e['ci95'],量子相对路线差百分比=100*(qdist-mean)/mean if mean else None,
                预算说明=f'每批生成{budget}个候选，按精修前距离选优，再执行一次相同精修；没有量子测量或训练。'))
    ordered=sorted(paired,key=lambda k:paired[k]['p_two_sided']);previous=0.
    for i,k in enumerate(ordered):
        e=paired[k];previous=max(previous,min(1.,(len(ordered)-i)*e['p_two_sided']));e['p_holm_12']=previous
        e['quantum_significantly_better']=previous<.05 and e['ci95'][1]<0
        e['random_significantly_better']=previous<.05 and e['ci95'][0]>0
    for r in tables:
        budget=r['候选分配预算']
        if budget is not None:
            e=paired[f"{r['实例']}:{budget}"];r['校正后概率值']=e['p_holm_12']
            r['统计判断']='量子流程显著更短' if e['quantum_significantly_better'] else '约束内随机显著更短' if e['random_significantly_better'] else '未检出显著路线差异'
    write_table('table_constrained_random_v4',tables)
    save(OUT/'paired_statistics.json',paired)
    save(OUT/'verification.json',dict(passed=True,seed_blocks=len(audit),rows=len(raw_records),
        candidate_draws=sum(r['candidate_draws'] for r in audit),failed_candidate_constructions=sum(r['failed_candidate_constructions'] for r in audit),
        checks='核对全部主种子、64批、六预算前缀选优、候选重试数、原始及最终容量、客户完整性、精修预算和来源哈希；每种子首批的所有原始及最终路线用原函数复算。',records=audit))
    with (OUT/'raw_results.jsonl').open('w',encoding='utf-8') as f:
        for r in raw_records:f.write(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n')
    lines=['当前最终量子流程与约束内随机的补充比较','',
        '新增生成、精修、统计与核验均在disco服务器执行。量子模型保持冻结，直接复用其既有独立测试。',
        '定义：随机打乱客户顺序，每次只在容量仍允许的车辆中等概率选车；无法继续时重新尝试，单候选最多100次。该构造不使用几何信息，也不是全可行域均匀抽样。',
        '预算：1、4、16、64、256、4096个候选共享前缀，按精修前距离选一份，然后使用完全相同的10000候选精修器。30客户20个主种子、44客户10个主种子，各64批。',
        '统计：先在每种子内汇总64批，再配对种子；两实例六档共12项路线检验统一Holm校正。20种子用100000次符号翻转近似检验，10种子枚举全部符号；95%区间用20000次种子重抽样。','']
    for name in protocol['instances']:
        q=next(r for r in tables if r['实例']==name and r['候选分配预算'] is None)
        lines.append(f"{name}：训练后量子流程最终路线{q['最终可行路线均值']:.3f}，原始可行率{q['精修前可行率']:.2%}，原始平均超载{q['精修前平均超载']:.3f}，平均精修改动{q['平均精修改动人数']:.3f}人。")
        for r in tables:
            if r['实例']!=name or r['候选分配预算'] is None:continue
            lines.append(f"{r['方法']}：最终路线{r['最终可行路线均值']:.3f}，最终可行率{r['最终可行率']:.2%}，精修改动{r['平均精修改动人数']:.3f}人；量子减随机差{r['量子减随机路线差']:.3f}，95%区间[{r['路线差95区间'][0]:.3f}, {r['路线差95区间'][1]:.3f}]，校正后概率值{r['校正后概率值']:.5g}；{r['统计判断']}。")
        lines.append('')
    lines += ['约束内随机若成功生成，其原始超载就是零。因此训练相对无约束随机参数的“减少超载”优势，不能直接转成对约束内随机的优势。',
        '同一精修器只保证后处理规则一致，并不保证端到端成本一致。量子训练4096次目标查询每次含5份精修；本经典对照每个候选仅做车内路线评价，预算内选一份后才跨车精修。4096个经典候选不是等同于4096次量子训练查询。',
        '选优依据是精修前路线，精修后的排名可能变化；因此随着候选预算增加，最终路线均值不保证单调改善。两实例证据不代表普遍量子或经典优势。']
    (OUT/'比较结论.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    save(OUT/'analysis_execution.json',dict(host=socket.gethostname(),alias='disco',python=sys.executable,source_sha256=digest(Path(__file__))))
    sources=[p for p in OUT.rglob('*') if p.is_file() and p.name!='artifact_manifest.json']
    sources += [HERE/'constrained_random_v4.py',Path(__file__),HERE/'tables/table_constrained_random_v4.csv',HERE/'tables/table_constrained_random_v4.md']
    save(OUT/'artifact_manifest.json',dict(files=[dict(path=p.relative_to(HERE).as_posix(),sha256=digest(p),bytes=p.stat().st_size) for p in sources]))
    sources.append(OUT/'artifact_manifest.json')
    with tarfile.open(HERE.parent/'constrained_random_v4_delivery.tar.gz','w:gz') as tar:
        for p in sources:tar.add(p,arcname=p.relative_to(HERE))
    print('\n'.join(lines),flush=True)
if __name__=='__main__':main()
