"""核对正式种子、测量预算、分配可行性、来源哈希和完整交付。"""
from final_objective_accelerated import *
from report import write_table,all_data,LABELS
import collections
def main():
    data,groups=all_data();checks=[];rows_checked=0;count_files=0
    expected=[('development','B-n31-k5',range(1001,1011),['LEGACY','NEW_DECODER','NEW_JOINT','RANDOM_THETA','V42_GEOMETRY','FINAL_JOINT']),
        ('confirmation','B-n31-k5',range(2001,2021),['LEGACY','NEW_DECODER','NEW_JOINT','RANDOM_THETA','V42_GEOMETRY','RANDOM_SEARCH','FINAL_JOINT','FINAL_RANDOM_SEARCH']),
        ('scale','A-n45-k7',range(3001,3011),['LEGACY','NEW_DECODER','NEW_JOINT','RANDOM_THETA','V42_GEOMETRY','RANDOM_SEARCH','FINAL_JOINT','FINAL_RANDOM_SEARCH'])]
    for cohort,name,seeds,methods in expected:
        for method in methods:
            actual=sorted(m['seed'] for m in groups.get((cohort,name,method),[]));assert actual==list(seeds),(cohort,method,actual)
            checks.append(dict(cohort=cohort,instance=name,method=method,seeds=actual))
    evaluators={};stability=[]
    for d in data:
        name=d['instance'];S=setup(name);m=read(HERE/d['model']);assert digest(HERE/d['model'])==d['model_sha256']
        key=(name,d['method']);E=evaluators.setdefault(key,FinalEvaluator(S,d['method'],m['nq'],m['layers']))
        assert E.fams.tolist()==m['mapping_fams'] and E.masks.tolist()==m['mapping_masks']
        assert E.priorities.tolist()==m['tie_priorities']
        for shots in (1024,4096,16384):
            rr=[r for r in d['rows'] if r['shots']==shots];assert sorted(r['batch'] for r in rr)==list(range(64))
            p=HERE/'final_tests'/d['cohort']/name/f"{d['method']}_{d['seed']}_{shots}.npz"
            a=np.load(p);c=a['counts'];assert c.shape==(64,3,1<<m['nq'])
            assert np.all(c.sum(axis=2)==allocation(shots)) and np.all(c>=0);count_files+=1
            assert all(r['stream_seed']==seed('unified_final_test',name,d['seed'],shots) for r in rr)
            vals=scores_from_moments(walsh(c[0]/allocation(shots)[:,None]),E.fams,E.masks)
            assert np.allclose(vals,a['scores'][0],atol=1e-12,rtol=0)
            assert np.array_equal(E.decode(vals),rr[0]['assignment'])
            for r in rr:
                for stage,field in (('raw','assignment'),('final','final_assignment')):
                    v=np.array(r[field],np.int64);assert len(v)==S['n'] and np.all((v>=0)&(v<S['K']))
                    loads=np.bincount(v,weights=S['q'],minlength=S['K']);over=float(np.maximum(0,loads-S['cap']).sum())
                    assert over==r[stage]['overload'] and bool(over==0)==r[stage]['feasible']
                assert np.sum(np.array(r['final_assignment'])!=r['assignment'])==r['changed_customers']
                assert 0<=r['refinement_candidate_evaluations']<=10000
                rows_checked+=1
            # 每模型每测量档首份最终路径用原车内函数独立复算；其余逐项核验容量及客户完整性。
            v=np.array(rr[0]['final_assignment']);dist=sum(route_cost(S['inst']['D'],(np.flatnonzero(v==k)+1).astype(np.int64)) for k in range(S['K']))
            assert dist==rr[0]['final']['distance']
        if d['method'] in ('FINAL_JOINT','FINAL_RANDOM_SEARCH'):
            assert m['training_ledger']['objective_queries']==4096
            assert m['training_ledger']['simulated_shots']==4096*4*4096
            if d['cohort']!='development':
                ref=next(x for x in groups[(d['cohort'],name,'RANDOM_THETA')] if x['seed']==d['seed'])
                rm=read(HERE/ref['model']);assert m['initial_theta']==rm['theta']
    for (cohort,name,method),gg in sorted(groups.items()):
        for shots in (1024,4096,16384):
            rr=[r for d in gg for r in d['rows'] if r['shots']==shots]
            stability.append(dict(阶段=cohort,实例=name,方法=LABELS[method],独立种子数=len(gg),每批测量次数=shots,
                原始可行率=float(np.mean([r['raw']['feasible'] for r in rr])),最终可行率=float(np.mean([r['final']['feasible'] for r in rr])),
                客户翻车率=float(np.mean([r['customer_flip_rate'] for r in rr])),整份分车一致率=float(np.mean([r['customer_flip_rate']==0 for r in rr]))))
    write_table('table_readout_stability',stability)
    bias=[]
    for p in (HERE/'decoder_bias').glob('*.json'):
        d=read(p)
        for method,value in d['results'].items():bias.append(dict(实例=d['instance'],输入组=method,**value))
    write_table('table_decoder_bias',bias)
    capacity=[]
    for p in (HERE/'models').glob('*/*/*/attribution.json'):
        d=read(p)
        for r in d['capacity']:capacity.append(dict(模型=d['model'],**r))
    save(HERE/'capacity_auxiliary_raw.json',capacity)
    capsummary=[];cg=collections.defaultdict(list)
    for r in capacity:cg[(r['模型'].replace('\\','/').split('/')[2],r['information'])].append(r)
    for (name,method),rr in cg.items():
        capsummary.append(dict(实例=name,信息组=method,求解次数=len(rr),最终可行次数=sum(bool(r.get('final') and r['final']['feasible']) for r in rr),
            说明='原始目标冻结阶段的容量约束辅助求解，不进入最终目标训练；逐条状态与预算见capacity_auxiliary_raw.json'))
    write_table('table_capacity_auxiliary',capsummary)
    sensitivity=[]
    for p in (HERE/'mapping_sensitivity').glob('*/*.json'):
        d=read(p);sensitivity.append(dict(实例=d['instance'],训练种子=d['seed'],映射种子=d['mapping_seed'],**d['summary'],解释=d['limitation']))
    assert len(sensitivity)==12;write_table('table_mapping_sensitivity',sensitivity)
    result=dict(passed=True,models=len(data),readout_records=rows_checked,full_count_files=count_files,groups=checks,
        constraints='每客户恰有一个合法车辆编号；全部需求完整保留；容量超载从分配重算；最终失败不记为可行。',
        measurements='每模型每档64批完整三基计数；1024、4096、16384总测量数逐文件核对；跨方法测试主种子一致。',
        model_selection='确认20种子、规模10种子全保留；最终训练与等查询搜索各4096查询、每查询4份4096次测量。',
        refiner='所有组共同确定性精修，10000候选上限；加速版本等价证据另存。',
        hardware='冻结当前已回收快照；未满足预算协议的数据不纳入主比较，不等待未返回任务。')
    save(HERE/'delivery_verification.json',result);print('完整交付核验通过',len(data),rows_checked,count_files,flush=True)
if __name__=='__main__':main()
