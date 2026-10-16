"""冻结最新稳定性修正版；导出不含凭据的 disco 计算包。"""
import sys,json,inspect,hashlib,tarfile
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent
OLD=HERE.parent/'pce_random_value_20261014'
sys.path.insert(0,str(OLD))
import run as R
sys.path.insert(0,str(HERE.parent/'pce_controls_20261015'))
from run_controls import initial_theta

def write(p,x):
    p.write_text(json.dumps(x,ensure_ascii=False,indent=1,allow_nan=False),encoding='utf-8')

def main():
    from worker import probabilities,make_groups,assess,load_setup
    original=R.C.read(OLD/'hardware/logical_tasks.json')['sources']
    stable={x['source']:x for x in R.C.read(OLD/'stability_results.json')}
    assert set(original)==set(stable)
    sources={};instances={}
    for key,old in original.items():
        fixed=stable[key];S=R.C.A.setup(old['instance']);theta=np.array(fixed['theta'])
        slots=[(f,tuple(q)) for f,q in old['slots']]
        c=R.K.CircuitK(8,4,True).prepare(slots)
        e=c.expect_k(c.run(theta))
        assert np.allclose(e,fixed['outputs'],atol=1e-12,rtol=0)
        assert np.array_equal(e>0,np.array(old['outputs'])>0)
        v=R.decode(e,S['n'],S['K']);assert np.array_equal(v,old['initial_assignment'])
        start=initial_theta(old['seed'])
        src=dict(instance=old['instance'],seed=old['seed'],theta=theta.tolist(),outputs=e.tolist(),
                 slots=old['slots'],initial_assignment=v.tolist(),before=old['before'],after=old['after'],
                 untrained_theta=start.tolist(),untrained_outputs=c.expect_k(c.run(start)).tolist(),
                 original_theta=old['theta'],original_outputs=old['outputs'],
                 revision_steps=fixed['steps'],revision_seed=fixed['seed'],
                 min_abs_before=fixed['min_before'],min_abs_after=fixed['min_after'])
        sources[key]=src
        instances[old['instance']]=dict(name=old['instance'],n=S['n'],K=S['K'],cap=S['cap'],
                D=S['inst']['D'].tolist(),q=S['q'].tolist(),bks=S['inst']['bks'])
        for tt in (theta,start,np.array(old['theta'])):
            actual=probabilities(tt);reference=c.probs(c.run(tt))
            assert all(np.allclose(actual[f],reference[f],atol=1e-12,rtol=0) for f in 'ZXY')
    paths=[OLD/'stability_results.json',OLD/'stability_validation.json',OLD/'hardware/logical_tasks.json']
    hashes={str(p.resolve()):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    payload=dict(sources=sources,instances=instances,hashes=hashes)
    write(HERE/'inputs.json',payload)
    # 导出原路线评价与精修函数，不改变候选次序或预算。
    shared='import numpy as np\nfrom numba import njit,types\nfrom numba.typed import Dict\nfrom types import SimpleNamespace\nREFINE_BUDGET=10000\n\n'
    shared+=inspect.getsource(R.C.SP.route_cost.py_func.__globals__['rlen'])+'\n'
    shared+=inspect.getsource(R.C.SP.route_cost)+'\n'
    shared+='SP=SimpleNamespace(route_cost=route_cost)\n\n'
    for f in (R.C.refine,R.decode,R.evaluate,R.random_prefixes):shared+=inspect.getsource(f)+'\n'
    (HERE/'shared.py').write_text(shared,encoding='utf-8')
    # 对独立导出实现做全规模逐条一致性检验。
    import shared as T
    checked=0
    for key,src in sources.items():
        S=load_setup(instances[src['instance']])
        for v in (np.array(src['initial_assignment']),T.decode(np.array(src['untrained_outputs']),S['n'],S['K'])):
            a=assess(S,v);ref=R.C.A.assess(R.C.A.setup(src['instance']),v)
            for field in ('distance','overload','feasible','gap_bks_percent'):assert a[field]==ref[field]
            w,counts=T.refine(S,v)
            ww,cc=R.C.refine(R.C.A.setup(src['instance']),v)
            assert np.array_equal(w,ww) and counts==cc
            checked+=1
    write(HERE/'bundle_verification.json',dict(passed=True,fullscale_assignments_checked=checked,
        probability_sets_checked=36,max_allowed_probability_error=1e-12,source_hashes=hashes))
    protocol=dict(version='20261015_latest_stability_revision',primary='最新稳定性修正后的12条线路',
        historical='修正前参数仅诊断；旧117项真机结果不充当本批修正后真机结果。',
        execution='模拟、全局随机、精修及统计在disco执行；东岭任务由既有本地接口提交和回收。',
        source_hashes=hashes,nq=8,layers=4,parameters=64,three_body_correlators=True,
        customer_counts=sorted(set(x['n'] for x in instances.values())),
        finite_repeats_per_source=32,shots_per_basis=4096,bases=['Z','X','Y'],
        no_information='每方向均匀抽取4096个8位读数，保留同一测量产生的关联结构，使用相同解码。经典生成，不冒称真机。',
        random_parameters='精确恢复每条训练的原始随机参数；相同线路结构、编码、解码。训练前后对照，非等训练计算成本。',
        global_budgets=[1,4,16,64,256,1024,4096],global_repeats_per_instance=32,
        global_selection='同一随机序列前缀，超载量优先、实际路线其次选一份，再精修一次。质量参照，不等同于测量预算。',
        hardware=dict(chip='Dongling',trained_circuits=12,random_parameter_circuits=12,
                      batches=3,science_tasks=216,calibration_tasks=9,total_tasks=225,total_measurements=921600,
                      scheduling='每批校验；训练和未训练线路交错，后续批次反转次序；同一物理路径，原始读出主分析。'),
        refinement='复用既有确定性路线排序及跨车移动/交换；最多10000候选；所有失败保留，精修前后都报告。',
        hypotheses=[
          '稳定性修正版真机读数比无信息读数保留更多训练关联符号。',
          '同一精修后，修正版真机优于无信息读数及全局随机1份，并报告全局随机阶梯。',
          '修正后训练参数优于训练前随机参数，分别检验精确模拟、有限测量与真机。'],
        statistics='六实例等权；读出重复嵌套在线路内，不当成新的独立问题；实例重抽样95%区间为描述性；不从未显著差异推导等效。',
        reference='既有数据集最佳已知值单列；不把启发式经典算法结果写成已证明最优。',
        model_decision='先完成修正后主实验，不根据成绩筛换线路；若失败，再冻结下一修改目标并使用新测量种子/新批次验证。')
    p=HERE/'protocol.json'
    if p.exists():assert R.C.read(p)==protocol
    else:write(p,protocol)
    with tarfile.open(HERE/'disco_bundle.tar.gz','w:gz') as archive:
        for filename in ('inputs.json','protocol.json','bundle_verification.json','shared.py','worker.py'):
            archive.add(HERE/filename,arcname=filename)
    print('READY',len(sources),'verified',checked,'hardware_tasks',225,flush=True)

if __name__=='__main__':main()
