"""同实例、同种子索引整理真机与已有经典对照；不重训、不选择有利批次。"""
import json, socket, statistics
from pathlib import Path
HERE=Path(__file__).resolve().parent
BASE=HERE.parent/'pce_v4_research'
def read(p): return json.loads(p.read_text(encoding='utf-8'))
def write(p,x): p.write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8')
assert socket.gethostname()=='lenovo'
data=read(HERE/'prior_v4_actual_budget_analysis.json')
seeds=sorted({r['seed'] for r in data['rows']})
batches=sorted({r['batch'] for r in data['rows']})
ref=read(BASE/'instances/B-n31-k5.json')['bks']
out=[]
def add(label,rows,scope):
    feasible=[r['final']['feasible'] for r in rows]
    distance=statistics.mean(r['final']['distance'] for r in rows) if all(feasible) else None
    out.append(dict(method=label,scope=scope,rows=len(rows),final_feasible=sum(feasible),final_distance=distance,
        gap_percent=(distance/ref-1)*100 if distance is not None else None,
        raw_feasible=sum(r['raw']['feasible'] for r in rows),raw_overload=statistics.mean(r['raw']['overload'] for r in rows),
        changed_customers=statistics.mean(r.get('changed_customers',r.get('final_changed_customers')) for r in rows)))
for source,label in [('QPU','东岭真机'),('IDEAL','同测量次数理想模拟')]:
    for arm,name in [('trained','训练后参数'),('random','未训练随机参数')]:
        add(label+'／'+name,[r for r in data['rows'] if r['source']==source and r['arm']==arm],'3个种子×8批，每批6144次测量')
random_rows=[r for sd in seeds for r in read(BASE/'constrained_random_v4/results/B-n31-k5'/f'{sd}.json')['rows'] if r['batch'] in batches]
for budget in (1,4,16,64,256,4096):
    add(f'约束内随机{budget}份选1',[r for r in random_rows if r['candidate_budget']==budget],f'同3个种子、同8个批次索引；生成{budget}份，选1份精修；事后提取既有结果')
classical=[r for sd in seeds for r in read(HERE/'classical_baselines'/f'{sd}.json')['rows']]
for method,label in [('GLOBAL_RANDOM_1','无约束全局随机1份'),('GLOBAL_RANDOM_4096','无约束全局随机4096份选1'),('CLASSICAL_DIRECT','经典直接求解')]:
    add(label,[r for r in classical if r['method']==method],'同3个种子、每种子1份既有结果；搜索预算另计')
write(HERE/'formal_baseline_comparison.json',dict(instance='B-n31-k5',reference=ref,seeds=seeds,batches=batches,table=out,
    limitation='约束内随机仅按既有种子与批次索引事后对齐，不是同运行成本，也没有共享物理测量随机数。经典直接求解是独立强基准，其搜索预算不同。参考值来自实例bks字段，本轮不提供最优性证明。',host=socket.gethostname()))
lines=['30客户／5车正式实例完整对照','实例：B-n31-k5；实例已知最佳参考值：'+str(ref),'',
       '方法 | 份数 | 最终可行数 | 最终路线均值 | 距参考值 | 原始可行数 | 原始超载 | 精修改动人数']
for r in out:
    lines.append(f"{r['method']} | {r['rows']} | {r['final_feasible']} | {r['final_distance']:.4f} | {r['gap_percent']:.2f}% | {r['raw_feasible']} | {r['raw_overload']:.4f} | {r['changed_customers']:.4f}")
lines+=['','范围与解释：','真机与同测量次数模拟采用相同的三个冻结模型，各8批，三测量方向合计6144次。',
        '约束内随机从已完成数据中事后提取同三个种子、批次0—7，未根据结果筛选批次；每份候选选优后仅精修一次。',
        '全部沿用固定精修器及10000候选预算；其他搜索、训练、测量成本不同，不能称为等总成本比较。',
        '经典直接求解每种子一份，作为强经典基准；不复制为8批增加样本数。',
        '参考值来自实例bks字段，本轮没有重新证明全局最优。',
        '只有三个独立种子；各均值差不自动构成显著性证据。真机698.04优于随机参数705.42，但劣于同参数理想模拟694.50及经典直接求解672。']
(HERE/'正式实例完整对照.txt').write_text('\n'.join(lines),encoding='utf-8')
print(json.dumps(out,ensure_ascii=False))
