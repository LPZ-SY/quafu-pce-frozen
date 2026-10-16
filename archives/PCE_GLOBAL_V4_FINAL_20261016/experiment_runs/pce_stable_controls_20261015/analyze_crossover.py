"""分别报告首次扩展与新种子验证；实例等权，不把重复当新实例。"""
import json
from pathlib import Path
import numpy as np
import worker as W

HERE=Path(__file__).resolve().parent
BUDGETS=(4096,8192,16384)

def main():
    report=W.read(HERE/'analysis.json');inputs=W.read(HERE/'inputs.json');names=list(inputs['instances'])
    setups={name:W.load_setup(inputs['instances'][name]) for name in names}
    target=report['summary']['stable_finite']['per_instance']
    rng=np.random.default_rng(20261015123)
    indexes=rng.integers(0,len(names),size=(20000,len(names)))
    batches={}
    for batch in ('first','holdout'):
        rows=[]
        for name in names:
            if batch=='first':
                original=W.read(HERE/'results'/f'global_{name}.json')['rows']
                extension=W.read(HERE/'extended'/'16384'/f'{name}.json')
                assert extension['complete_repeats']==32
                rows.extend(r for r in original if r['method']=='global_4096')
                rows.extend(extension['rows'])
            else:
                x=W.read(HERE/'extended'/'holdout'/f'{name}.json')
                assert x['start']==32 and x['next_repeat']==64 and len(x['rows'])==96
                rows.extend(x['rows'])
        table={}
        for budget in BUDGETS:
            method=f'global_{budget}'+('_holdout' if batch=='holdout' else '')
            per={}
            feasible=0
            for name in names:
                rr=[r for r in rows if r['instance']==name and r['method']==method]
                assert len(rr)==32 and set(r['repeat'] for r in rr)==(set(range(32)) if batch=='first' else set(range(32,64)))
                for r in rr:
                    assert W.assess(setups[name],np.array(r['final_assignment']))==r['after']
                feasible+=sum(r['after']['feasible'] for r in rr)
                if any(not r['after']['feasible'] for r in rr):
                    per[name]=None
                else:per[name]=float(np.mean([r['after']['gap_bks_percent'] for r in rr]))
            if all(v is not None for v in per.values()):
                gaps=np.array([per[n] for n in names])
                differences=np.array([per[n]-target[n]['after']['gap'] for n in names])
                interval=np.quantile(differences[indexes].mean(axis=1),[.025,.975]).tolist()
                table[str(budget)]=dict(count=192,after_feasible=feasible,
                    mean_gap=float(gaps.mean()),random_minus_quantum=float(differences.mean()),
                    descriptive_cluster_interval=interval,per_instance_gap=per)
            else:table[str(budget)]=dict(count=192,after_feasible=feasible,mean_gap=None,
                                         note='精修后有失败，不能只对成功子集计算平均差距')
        first=next((b for b in BUDGETS if table[str(b)]['mean_gap'] is not None and
                    table[str(b)]['mean_gap']<=report['summary']['stable_finite']['after']['gap']),None)
        batches[batch]=dict(table=table,first_tested_crossover=first)
    result=dict(target='最新稳定性修正版同测量次数模拟',
                target_gap=report['summary']['stable_finite']['after']['gap'],
                instances=names,first_batch=batches['first'],independent_seed_batch=batches['holdout'],
                note='首次扩展与新种子验证分别报告；差值为随机减量子，负数表示随机更好。实例重抽样区间是描述性而非等效性证明。精修后结果不保证随尝试次数单调改善。')
    W.save(HERE/'extended'/'crossover_validation.json',result)
    lines=['全局随机需要尝试多少份：首次扩展与新随机种子验证','',
           f"最新修正版同测量次数模拟目标：精修后平均参考差距 {result['target_gap']:.3f}% 。",
           '随机每个档位先按总超载、精修前路程选一份，再精修一次；六实例等权。',
           '首次扩展使用原随机序列编号0—31；独立验证使用编号32—63。每批每实例32次。','',
           '| 随机份数 | 首次扩展差距 | 新种子验证差距 | 新种子随机减量子 | 新种子实例区间 |',
           '|---:|---:|---:|---:|---:|']
    for budget in BUDGETS:
        a=batches['first']['table'][str(budget)];b=batches['holdout']['table'][str(budget)]
        if b['mean_gap'] is None:lines.append(f'| {budget} | {a["mean_gap"]:.3f}% | 有失败 | — | — |')
        else:
            lo,hi=b['descriptive_cluster_interval']
            lines.append(f'| {budget} | {a["mean_gap"]:.3f}% | {b["mean_gap"]:.3f}% | '
                         f'{b["random_minus_quantum"]:+.3f}个百分点 | [{lo:+.3f}, {hi:+.3f}] |')
    lines+=['',f"首次扩展中首次达到目标的已测档位：{batches['first']['first_tested_crossover']}份。",
            f"独立随机种子中首次达到目标的已测档位：{batches['holdout']['first_tested_crossover'] or '截至16384份尚未达到'}。",
            '这两个结论都只针对同测量次数模拟；东岭真机平均差距14.481%，全局随机1份已低于它。白桦尚未返回。',
            '由于精修只作用于精修前挑中的一份，最终质量不保证随尝试次数严格单调；首次达到是已测档位，不是精确最少份数。']
    (HERE/'extended'/'随机追平验证结果.txt').write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps(dict(first=batches['first']['first_tested_crossover'],
                          holdout=batches['holdout']['first_tested_crossover'],
                          holdout_gaps={b:v['mean_gap'] for b,v in batches['holdout']['table'].items()}),ensure_ascii=False))

if __name__=='__main__':main()
