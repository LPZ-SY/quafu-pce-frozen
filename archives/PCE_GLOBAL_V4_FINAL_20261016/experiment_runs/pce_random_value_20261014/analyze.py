"""逐实例聚合，重算全部路线，按实例而非测量重复估计不确定性。"""
import run as R
import numpy as np

def summary(rows):
    out=dict(runs=len(rows))
    for stage in ('before','after'):
        good=[r[stage] for r in rows if r[stage]['feasible']]
        out[stage]=dict(feasible=len(good),mean_gap=float(np.mean([x['gap_bks_percent'] for x in good])) if good else None,
            mean_distance=float(np.mean([x['distance'] for x in good])) if good else None)
    return out

def main():
    records=[R.C.read(p) for p in sorted((R.HERE/'runs').glob('*.json'))]
    assert len(records)==192 and len({(x['instance'],x['repeat']) for x in records})==192
    rows=[x['quantum'] for x in records]+[y for x in records for y in x['random']]
    for row in rows:
        S=R.C.A.setup(row['instance'])
        for stage,key in [('before','initial_assignment'),('after','final_assignment')]:
            assert R.C.A.assess(S,np.array(row[key]))==row[stage]
        if row['method']=='quantum_8q4':
            s,f,m=R.slots(S);actual=R.expectations(np.array(row['theta']),f,m)
            assert np.max(np.abs(actual-row['outputs']))<1e-12
            assert np.array_equal(R.decode(actual,S['n'],S['K']),row['initial_assignment'])
        assert row['refinement']['candidate_evaluations']<=10000
    results=[]
    for label,budget in [('Q',None)]+[('R',b) for b in R.BUDGETS]:
        selected=[r for r in rows if (r['method']=='quantum_8q4' if label=='Q' else r['method']=='random' and r['budget']==budget)]
        results.append(dict(method=label,budget=budget,**summary(selected),instances={n:summary([r for r in selected if r['instance']==n]) for n in R.C.NAMES}))
    comparisons=[];rng=np.random.default_rng(20261014)
    for b in R.BUDGETS:
        q=results[0];r=next(x for x in results if x['budget']==b)
        comparison=dict(budget=b)
        for stage in ['before','after']:
            wins=ties=losses=0
            for rec in records:
                a=rec['quantum'][stage];z=next(x for x in rec['random'] if x['budget']==b)[stage]
                qa=(a['overload'],a['distance']);ra=(z['overload'],z['distance'])
                wins+=qa<ra;ties+=qa==ra;losses+=qa>ra
            obj=dict(quantum_wins=wins,ties=ties,quantum_losses=losses)
            # 只有全部输出可行才比较全部实例均值，避免对可行子集的偏差。
            if q[stage]['feasible']==192 and r[stage]['feasible']==192:
                diff=np.array([r['instances'][n][stage]['mean_gap']-q['instances'][n][stage]['mean_gap'] for n in R.C.NAMES])
                boot=diff[rng.integers(0,6,(20000,6))].mean(1)
                obj.update(mean_random_minus_quantum=float(diff.mean()),instance_bootstrap95=np.quantile(boot,[.025,.975]).tolist(),
                    instance_wins=int(sum(diff>0)),instance_losses=int(sum(diff<0)))
            comparison[stage]=obj
        comparisons.append(comparison)
    R.save('analysis.json',dict(summary=results,comparisons=comparisons,
        cost=dict(quantum_training_seconds_mean=float(np.mean([r['training_seconds'] for r in rows if r['method']=='quantum_8q4'])),
            quantum_training_unique_route_evaluations_mean=float(np.mean([r['training_unique_route_evaluations'] for r in rows if r['method']=='quantum_8q4'])),
            quantum_min_abs_correlator_median=float(np.median([r['min_abs_correlator'] for r in rows if r['method']=='quantum_8q4'])))))
    lines=['8比特完整分车的价值实验：真实路程目标','',
        '6个实例，每实例32次独立训练，量子每次8比特4层、64个参数、14个纠缠门、20000次参数提案。',
        '随机每次生成1/4/16/64/256/1024/4096份分车，按容量可行优先、实际路程选一份。',
        '双方使用同一车内排序和同一跨车精修，每次最多10000候选评价。精修前后追踪同一份输出，不按最终结果更换赢家。',
        '本轮是新训练，不是旧结果重新命名；无信号裕量惩罚，真机稳定性需另验证。随机次数不是量子步数等成本换算。',
        '精修前部分随机输出不可行，下面的路程差距只对可行输出取均值，必须结合可行数阅读。','',
        '| 方法 | 精修前可行/192 | 精修前可行解平均最优差距 | 精修后可行/192 | 精修后平均最优差距 |',
        '|---|---:|---:|---:|---:|']
    for r in results:
        cells=['量子8比特' if r['method']=='Q' else f"随机{r['budget']}份"]
        for stage in ['before','after']:
            x=r[stage];cells += [str(x['feasible']),f"{x['mean_gap']:.2f}%" if x['mean_gap'] is not None else '无可行解']
        lines.append('| '+' | '.join(cells)+' |')
    lines+=['','逐实例真实路程均值（量子）：','| 实例 | 最优值 | 精修前 | 精修后 |','|---|---:|---:|---:|']
    for n,x in results[0]['instances'].items():lines.append(f"| {n} | {R.C.SP.INSTANCES[n]} | {x['before']['mean_distance']:.2f} | {x['after']['mean_distance']:.2f} |")
    lines+=['','精修后随机减量子平均差距（正数表示量子更好，单位为百分点），95%区间按六个实例重抽样；七档比较未作多重校正：']
    for x in comparisons:
        y=x['after'];lines.append(f"随机{x['budget']}份：{y.get('mean_random_minus_quantum')}；区间{y.get('instance_bootstrap95')}；逐实例量子胜{y.get('instance_wins')}负{y.get('instance_losses')}。")
    lines+=['','口径更正：旧白桦直接优化批次实际为10比特、两比特关联，尽管旧文字报告标作8比特；本轮8比特、三比特关联由冻结代码和参数维度核验。',
        '真机任务单独存hardware；未返回数据不得写成真机结论。此模拟比较不证明量子加速，也未排除经典有方向搜索取得相同或更好成绩。']
    (R.HERE/'模拟实验结果.txt').write_text('\n'.join(lines),encoding='utf-8')
    R.save('result_verification.json',dict(passed=True,training_runs=192,total_before_after_records=len(rows),all_quantum_states_recomputed=True))
    for x in results:print(x['method'],x['budget'],x['before'],x['after'],flush=True)

if __name__=='__main__':main()
