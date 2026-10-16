"""按完整独立测试给出验收状态与精修机制，不用模型名称代替成功结论。"""
from report import *
def main():
    data,groups=all_data();paired_results=read(HERE/'paired_statistics.json');mechanism=[];acceptance={}
    for cohort,name in [('confirmation','B-n31-k5'),('scale','A-n45-k7')]:
        for method in ('FINAL_JOINT','FINAL_RANDOM_SEARCH','RANDOM_THETA'):
            models=groups[(cohort,name,method)];per_seed=[]
            for m in models:
                rr=[r for r in m['rows'] if r['shots']==4096]
                per_seed.append(dict(seed=m['seed'],
                    repair_evaluations=float(np.mean([r['repair_evaluations'] for r in rr])),
                    repair_changed=float(np.mean([np.sum(np.array(r['repair_assignment'])!=r['assignment']) for r in rr if r['repair_assignment'] is not None])),
                    final_changed=float(np.mean([r['changed_customers'] for r in rr])),
                    total_evaluations=float(np.mean([r['refinement_candidate_evaluations'] for r in rr])),
                    budget_exhaustion=float(np.mean([r['refinement_candidate_evaluations']==10000 for r in rr]))))
            mechanism.append(dict(实例=name,方法=LABELS[method],方法编号=method,独立种子数=len(models),
                首次容量可行前平均候选数=float(np.mean([r['repair_evaluations'] for r in per_seed])),
                仅容量修复平均改动人数=float(np.mean([r['repair_changed'] for r in per_seed])),
                最终精修平均改动人数=float(np.mean([r['final_changed'] for r in per_seed])),
                平均精修候选数=float(np.mean([r['total_evaluations'] for r in per_seed])),
                用尽候选预算比例=float(np.mean([r['budget_exhaustion'] for r in per_seed])),种子记录=per_seed))
        a=aggregate_group(groups[(cohort,name,'FINAL_JOINT')]);p=paired_results[f'{name}:RANDOM_THETA']
        acceptance[name]=dict(final_feasibility_hard_gate_passed=a['最终失败批数']==0,
            all_three_improvements_supported=p['all_three_supported'],effects=p['effects'],
            equal_query_search=paired_results[f'{name}:FINAL_RANDOM_SEARCH']['effects'])
    write_table('table_refinement_mechanism',mechanism)
    save(HERE/'final_acceptance.json',dict(model='PCE_GLOBAL_V4_FINAL',simulation_completed=True,
        all_scales_meet_user_success_conditions=all(v['final_feasibility_hard_gate_passed'] and v['all_three_improvements_supported'] for v in acceptance.values()),
        per_instance=acceptance,hardware='未通过冻结预算校验，待回收任务不阻塞本次模拟交付。'))
    notes=['','## 综合判断与下一步研究重点','',
        '30客户确认达到用户提出的三项改善要求；44客户的最终路线与原始超载显著改善，但精修净改动人数显著增加，因此整体科研目标尚未全部达到。PCE_GLOBAL_V4_FINAL是本轮冻结模型的名称，不能解释为所有规模均验收成功。',
        '相对同目标、同4096次查询的随机参数搜索，30客户路线差的95%区间包含零，尚不足以证明路线更优；44客户路线更短，区间不包含零。两个规模均未超过所列经典直接求解基线。',
        '44客户中，较少的原始超载并不必然意味着较少的最终跨车改动。容量修复与缩短路线使用同一有限候选预算，后者也可能主动跨车调整。表table_refinement_mechanism分别列出首次容量可行前的候选消耗、容量修复改动和最终改动，可定位这一权衡；本轮没有把“容量修复改动减少”替代用户要求的“最终精修改动减少”。']
    for name in ('B-n31-k5','A-n45-k7'):
        a=next(r for r in mechanism if r['实例']==name and r['方法编号']=='FINAL_JOINT');b=next(r for r in mechanism if r['实例']==name and r['方法编号']=='RANDOM_THETA')
        notes.append(f"{name}：训练／未训练随机参数的仅容量修复平均改动人数为{a['仅容量修复平均改动人数']:.3f}／{b['仅容量修复平均改动人数']:.3f}；首次可行前平均候选评价数为{a['首次容量可行前平均候选数']:.1f}／{b['首次容量可行前平均候选数']:.1f}；用尽10000候选预算比例为{a['用尽候选预算比例']:.2%}／{b['用尽候选预算比例']:.2%}。")
    notes += ['','后续结构修改应优先把“精修改动量可接受”做成训练阶段的显式约束或独立的可调约束系数，再在新种子与更多实例上确认；单纯继续增大线路或测量次数没有本轮证据支持。保持最终可行硬门槛和公共精修器不变，检查路线收益能否在更少改动下保持。这是由本轮44客户失败项提出的下一轮假设，不是已经验证的成果。']
    p=HERE/'final_conclusion.md';body=p.read_text(encoding='utf-8').split('\n## 综合判断与下一步研究重点')[0]
    p.write_text(body+'\n'.join(notes)+'\n',encoding='utf-8')
    print(json.dumps({k:v['all_three_improvements_supported'] for k,v in acceptance.items()}),flush=True)
if __name__=='__main__':main()
