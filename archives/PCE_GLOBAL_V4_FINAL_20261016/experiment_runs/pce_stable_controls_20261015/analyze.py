"""最新修正版对照统一分析；只读本批真机记录，旧真机不会进入主表。"""
import json,hashlib
from pathlib import Path
import numpy as np
import worker as W
from shared import decode
HERE=Path(__file__).resolve().parent
LABELS={
 'stable_exact':'稳定性修正版：精确模拟',
 'stable_finite':'稳定性修正版：同测量次数模拟',
 'stable_hardware':'稳定性修正版：东岭真机',
 'stable_hardware_baihua':'稳定性修正版：白桦真机',
 'stable_hardware_shenglian':'稳定性修正版：圣莲真机',
 'no_information':'无训练信息读数（经典生成）',
 'untrained_exact':'训练前随机参数：精确模拟',
 'untrained_finite':'训练前随机参数：同测量次数模拟',
 'untrained_hardware':'训练前随机参数：东岭真机',
 'untrained_hardware_baihua':'训练前随机参数：白桦真机',
 'untrained_hardware_shenglian':'训练前随机参数：圣莲真机',
 'global_one_paired':'全局随机1份：逐线路逐次配对，各精修一次',
 'historical_finite':'修正前参数：同测量次数模拟（历史诊断）'}
EXTRA_BUDGETS=[8192,16384,32768,65536]
LABELS.update({f'global_{n}':f'全局随机分车{n}份，选一份精修'
               for n in W.BUDGETS+EXTRA_BUDGETS})

def hardware_rows(data,device='Dongling'):
    suffix={'Dongling':'','Baihua':'_baihua','Shenglian':'_shenglian'}[device]
    folder=HERE/('hardware'+suffix)
    if not (folder/'manifest.json').exists():return [],dict(state='未准备')
    manifest=W.read(folder/'manifest.json')
    assert manifest['chip']==device
    if not (folder/'ledger.json').exists():return [],dict(state='已准备未提交',tasks=len(manifest['tasks']))
    ledger=W.read(folder/'ledger.json');logical=W.read(folder/'logical_tasks.json')
    assert logical['inputs_sha256']==hashlib.sha256((HERE/'inputs.json').read_bytes()).hexdigest()
    status=dict(state='回收中',tasks=len(manifest['tasks']),submitted=len(ledger),
                done=sum(x['status']=='DONE' for x in ledger.values()),
                failed=sum(x['status']=='FAILED' for x in ledger.values()),chip=device,checks={})
    def probability(label):
        raw=W.read(folder/'raw'/f'{label}.json');counts=raw.get('count') or raw.get('counts')
        assert sum(counts.values())==W.SHOTS
        p=np.zeros(256)
        for word,n in counts.items():
            word=str(word).replace(' ','');assert len(word)==8 and set(word)<=set('01')
            p[int(word[::-1],2)]+=n/W.SHOTS
        return p
    rows=[]
    for batch in range(3):
        controls=[f'{k}_b{batch}' for k in ('zero','one','order')]
        if not all(ledger.get(k,{}).get('status')=='DONE' for k in controls):continue
        checks={label:float(probability(f'{label}_b{batch}')[int(word,2)])
                for label,word in [('zero','00000000'),('one','11111111'),('order','00101101')]}
        order_ok=format(int(np.argmax(probability(f'order_b{batch}'))),'08b')=='00101101'
        status['checks'][str(batch)]=dict(expected_word_probability=checks,order_passed=order_ok)
        if not order_ok:continue
        for k,src in logical['sources'].items():
            labels={f:f'{k}_{f}_b{batch}' for f in 'ZXY'}
            if not all(ledger.get(l,{}).get('status')=='DONE' for l in labels.values()):continue
            original=data['sources'][src['source']]
            assert src['theta']==original['theta' if src['arm']=='stable' else 'untrained_theta']
            groups=W.make_groups(src['slots']);pp={f:probability(l) for f,l in labels.items()}
            e,_=W.expectations(groups,pp);target=np.array(original['outputs'])
            den=np.linalg.norm(e)*np.linalg.norm(target)
            S=W.load_setup(data['instances'][src['instance']]);v=decode(e,S['n'],S['K'])
            rows.append(W.process(S,v,src['arm']+'_hardware'+suffix,dict(source=src['source'],repeat=batch,
                        outputs=e.tolist(),sign_agreement=float(np.mean((e>0)==(target>0))),
                        cosine=float(e@target/den) if den else 0.0,
                        same_stable_assignment=bool(np.array_equal(v,original['initial_assignment'])))))
    status['complete_solutions']=len(rows)
    if status['done']==status['tasks']:status['state']='全部返回'
    if any(not c['order_passed'] for c in status['checks'].values()):status['state']='校验异常，相关批次暂不解码'
    return rows,status

def summarize(rows,names):
    result={}
    for method in LABELS:
        rr=[r for r in rows if r['method']==method]
        if not rr:continue
        per={}
        for name in names:
            group=[r for r in rr if r['instance']==name]
            if not group:continue
            a=dict(count=len(group))
            for stage in ('before','after'):
                values=[r[stage] for r in group];good=[x for x in values if x['feasible']]
                a[stage]=dict(feasible=len(good),feasible_rate=len(good)/len(values),
                    distance=float(np.mean([x['distance'] for x in values])),
                    overload=float(np.mean([x['overload'] for x in values])),
                    gap=float(np.mean([x['gap_bks_percent'] for x in values])) if len(good)==len(values) else None,
                    feasible_only_gap=float(np.mean([x['gap_bks_percent'] for x in good])) if good else None)
            for field in ('sign_agreement','cosine','same_stable_assignment'):
                if all(field in x for x in group):a[field]=float(np.mean([x[field] for x in group]))
            per[name]=a
        aggregate=dict(count=len(rr),instances=len(per),per_instance=per)
        for stage in ('before','after'):
            pp=[x[stage] for x in per.values()]
            aggregate[stage]=dict(feasible=sum(x['feasible'] for x in pp),
                feasible_rate=float(np.mean([x['feasible_rate'] for x in pp])),
                distance=float(np.mean([x['distance'] for x in pp])),
                overload=float(np.mean([x['overload'] for x in pp])),
                gap=float(np.mean([x['gap'] for x in pp])) if all(x['gap'] is not None for x in pp) else None)
        for field in ('sign_agreement','cosine','same_stable_assignment'):
            if all(field in x for x in per.values()):aggregate[field]=float(np.mean([x[field] for x in per.values()]))
        result[method]=aggregate
    return result

def differences(summary,names):
    pairs=[('stable_hardware','no_information'),('stable_hardware','global_one_paired'),
           ('stable_hardware','untrained_hardware'),('stable_hardware','stable_finite'),
           ('stable_finite','untrained_finite'),('stable_exact','untrained_exact'),
           ('stable_finite','historical_finite'),('stable_finite','no_information'),
           ('stable_finite','global_one_paired')]
    pairs += [('stable_hardware_baihua',right) for right in
              ('no_information','global_one_paired','untrained_hardware_baihua','stable_finite','stable_hardware')]
    pairs += [('stable_hardware_shenglian',right) for right in
              ('no_information','global_one_paired','untrained_hardware_shenglian',
               'stable_finite','stable_hardware','stable_hardware_baihua')]
    pairs += [(left,f'global_{budget}') for left in
              ('stable_finite','stable_hardware','stable_hardware_baihua','stable_hardware_shenglian')
              for budget in W.BUDGETS+EXTRA_BUDGETS if budget!=1]
    rng=np.random.default_rng(20261015091);resamples=rng.integers(0,len(names),size=(20000,len(names)))
    result=[]
    for left,right in pairs:
        if left not in summary or right not in summary:continue
        a,b=summary[left]['per_instance'],summary[right]['per_instance']
        if any(name not in a or name not in b for name in names):continue
        for metric in ('after_gap','before_feasibility','sign_agreement'):
            if metric=='after_gap':
                if any(a[n]['after']['gap'] is None or b[n]['after']['gap'] is None for n in names):continue
                values=np.array([b[n]['after']['gap']-a[n]['after']['gap'] for n in names])
            elif metric=='before_feasibility':values=100*np.array([a[n]['before']['feasible_rate']-b[n]['before']['feasible_rate'] for n in names])
            else:
                if any(metric not in a[n] or metric not in b[n] for n in names):continue
                values=100*np.array([a[n][metric]-b[n][metric] for n in names])
            boot=values[resamples].mean(axis=1)
            result.append(dict(left=left,right=right,metric=metric,
                positive_means='左侧更好',mean=float(values.mean()),
                descriptive_cluster_interval=np.quantile(boot,[.025,.975]).tolist(),
                per_instance=dict(zip(names,values.tolist()))))
    return result


def win_rates(rows,names):
    """补充描述：先比较总超载，再比较路程；保留失败，实例等权。"""
    pairs=[('stable_finite','untrained_finite'),('stable_finite','no_information')]
    pairs.append(('stable_finite','global_one_paired'))
    for left,control in [('stable_hardware','untrained_hardware'),
                         ('stable_hardware_baihua','untrained_hardware_baihua'),
                         ('stable_hardware_shenglian','untrained_hardware_shenglian')]:
        pairs.extend((left,r) for r in (control,'no_information','stable_finite','global_one_paired'))
    pairs.extend((left,f'global_{b}') for left in
                 ('stable_finite','stable_hardware','stable_hardware_baihua','stable_hardware_shenglian')
                 for b in W.BUDGETS+EXTRA_BUDGETS)
    rng=np.random.default_rng(20261015092)
    indexes=rng.integers(0,len(names),size=(20000,len(names)))
    result=[]
    for left,right in pairs:
        per={}
        for name in names:
            a=[r['after'] for r in rows if r['instance']==name and r['method']==left]
            b=[r['after'] for r in rows if r['instance']==name and r['method']==right]
            if not a or not b:continue
            scores=[]
            for x in a:
                for y in b:
                    xx=(x['overload'],x['distance']);yy=(y['overload'],y['distance'])
                    scores.append(1.0 if xx<yy else .5 if xx==yy else 0.0)
            per[name]=float(np.mean(scores))
        if len(per)!=len(names):continue
        values=np.array([per[n] for n in names])
        result.append(dict(left=left,right=right,mean=float(values.mean()),per_instance=per,
            descriptive_cluster_interval=np.quantile(values[indexes].mean(axis=1),[.025,.975]).tolist()))
    return result


def device_decisions(summary,status):
    result={}
    for device,suffix in [('Dongling',''),('Baihua','_baihua'),('Shenglian','_shenglian')]:
        trained=summary.get('stable_hardware'+suffix)
        untrained=summary.get('untrained_hardware'+suffix)
        state=status[device]
        if (trained is None or untrained is None or trained['count']!=36 or untrained['count']!=36
                or state.get('done')!=225 or len(state.get('checks',{}))!=3
                or any(not x['order_passed'] for x in state['checks'].values())):
            result[device]=dict(state='等待完整且通过校验的读数')
            continue
        target=trained['after']['gap'];before=trained['before']['feasible']
        random=summary['global_one_paired']['after']['gap']
        candidate=untrained['after']['gap']
        ideal=summary['stable_finite']['after']['gap']
        result[device]=dict(state='完整',before_feasible=f'{before}/36',
            after_feasible=f"{trained['after']['feasible']}/36",
            trained_gap=target,untrained_gap=candidate,random_one_gap=random,
            same_measurements_simulation_gap=ideal,
            trained_minus_simulation=None if target is None else target-ideal,
            random_minus_trained=None if target is None else random-target,
            untrained_minus_trained=None if target is None or candidate is None else candidate-target,
            trained_sign_agreement=trained.get('sign_agreement'),
            untrained_sign_agreement=untrained.get('sign_agreement'),
            mean_quality_conditions_met=bool(target is not None and candidate is not None
                and target<random and target<candidate),
            note='仅对六实例等权平均作描述判断；读出重复不当作独立实例，不能据此证明量子加速。')
    return result

def main():
    data=W.read(HERE/'inputs.json');names=list(data['instances']);rows=[]
    files=sorted((HERE/'results').glob('*.json'));assert len(files)==18
    for p in files:rows.extend(W.read(p)['rows'])
    paired=HERE/'controls'/'paired_one.json'
    if paired.exists():
        control=W.read(paired)
        assert control['input_sha256']==hashlib.sha256((HERE/'inputs.json').read_bytes()).hexdigest()
        assert len(control['rows'])==384 and control['random_per_output']==1
        rows.extend(control['rows'])
    extension_status={}
    for maximum in (16384,65536):
        folder=HERE/'extended'/str(maximum)
        completion=folder/'completion.json'
        if not completion.exists():continue
        state=W.read(completion)
        assert state['complete'] and state['instances']==len(names) and state['repeats']==W.REPEATS
        for name in names:
            item=W.read(folder/f'{name}.json')
            assert item['complete_repeats']==W.REPEATS and item['instance']==name
            assert item['input_sha256']==hashlib.sha256((HERE/'inputs.json').read_bytes()).hexdigest()
            assert len(item['rows'])==W.REPEATS*(len(item['budgets'])-1)
            rows.extend(item['rows'])
        extension_status[str(maximum)]=state
    hw=[];status={}
    for device in ('Dongling','Baihua','Shenglian'):
        device_rows,device_status=hardware_rows(data,device)
        hw.extend(device_rows);status[device]=device_status
    rows.extend(hw)
    # 每个输出均独立复核容量、路线长度；抽取每方法每实例首条复核精修流程。
    checked=0;seen=set()
    from shared import refine
    for row in rows:
        S=W.load_setup(data['instances'][row['instance']])
        for stage,field in [('before','initial_assignment'),('after','final_assignment')]:
            a=W.assess(S,np.array(row[field]));assert a==row[stage]
        key=row['method'],row['instance']
        if key not in seen:
            w,counts=refine(S,np.array(row['initial_assignment']))
            assert w.tolist()==row['final_assignment'] and counts==row['refinement'];seen.add(key)
        checked+=1
    W.save(HERE/'hardware_analysis.json',dict(status=status,rows=hw))
    summary=summarize(rows,names);diff=differences(summary,names);wins=win_rates(rows,names)
    crossings={}
    for target in ('stable_hardware','stable_hardware_baihua','stable_hardware_shenglian',
                   'stable_finite','stable_exact'):
        if target not in summary:continue
        tg=summary[target]['after']['gap']
        matches=[b for b in W.BUDGETS+EXTRA_BUDGETS if tg is not None and f'global_{b}' in summary
                 and summary[f'global_{b}']['after']['gap'] is not None
                 and summary[f'global_{b}']['after']['gap']<=tg]
        crossings[target]=dict(first_tested_budget=matches[0] if matches else None,
             matched_budgets=matches,note='仅比较六实例等权均值的已测试档位；不是精确交点或等成本。')
    decisions=device_decisions(summary,status)
    report=dict(summary=summary,differences=diff,win_rates=wins,crossings=crossings,
                extended_random=extension_status,hardware=status,
                device_decisions=decisions,
                verification=dict(all_rows_checked=checked,refinements_rechecked=len(seen),passed=True))
    W.save(HERE/'analysis.json',report)
    def num(x):return '—' if x is None else f'{x:.2f}'
    lines=['最新稳定性修正版：完整规模三项对照实验','',
      '主对象：2026年10月14日稳定性修正后的12条8量子比特线路，30—44客户，6个既有实例；执行位置disco。',
      '模拟、随机对照、真机解码、精修与统计在disco执行；真机提交与回收由本地接口执行。',
      '旧117项任务不进入本批修正后真机列。东岭、白桦、圣莲各225项：训练后108、训练前108、校验9；每项4096次测量。三台设备分开统计。',
      f"本批真机状态：{status}",'',
      '真机每组预期36份分车。未全部返回时，表内只列已通过校验且三方向凑齐的输出，属于暂定结果；不同组的返回进度可能不同，不能提前宣布优劣。',
      '全局随机1份为主要简单随机对照：每条线路每次有限测量输出对应一次全局随机抽取，各得到一份完整分车、各精修一次。该逐输出配对对照是在原随机阶梯完成后补充，单独标明。',
      '全局随机4至4096份是搜索质量曲线，不称为量子单次输出的等成本对照。',
      '追加的8192份及以上档位用于探索随机多久能追上修正版的同测量次数模拟；按原随机序列继续抽取，先选一份再精修一次。它们是事后追加，不冒充预冻结主实验。',
      '固定名称：局部随机调车=初赛局部方法；全局随机分车=本轮所有客户独立随机选车。',
      '全局随机有放回抽取次数没有数学上限；不重复的带编号车辆分配最多K^n种，其中包含超载与空车分配。',
      '全局随机档位是质量参照；一组完整量子读出含三方向各4096次测量，合起来解码成一份分车，不能当作12288份分车。','',
      '| 方法 | 分车数 | 精修前可行 | 精修前平均超载 | 精修前平均路程† | 精修后可行 | 精修后平均路程 | 精修后距参考值 |',
      '|---|---:|---:|---:|---:|---:|---:|---:|']
    for m,label in LABELS.items():
        if m not in summary:
            if '_hardware' in m:lines.append(f'| {label} | 待真机结果 | — | — | — | — | — | — |')
            continue
        s=summary[m];b,a=s['before'],s['after'];n=s['count']
        lines.append(f"| {label} | {n} | {b['feasible']}/{n} | {b['overload']:.2f} | {b['distance']:.2f} | {a['feasible']}/{n} | {a['distance']:.2f} | {num(a['gap'])}% |")
    lines+=['','†精修前路程含超载分车，仅作诊断，不能当作可行配送成本。所有失败保留；若精修后仍有失败，总体差距记为缺失，不只对成功子集报均值。',
       '六个实例等权平均；模拟的重复测量不当作新增独立线路。各组精修预算相同；训练成本没有省略为零。','',
       '| 方法 | 与修正版精确关联符号一致率 | 完整分车一致率 |','|---|---:|---:|']
    for m,s in summary.items():
        if 'sign_agreement' in s:lines.append(f"| {LABELS[m]} | {100*s['sign_agreement']:.2f}% | {100*s['same_stable_assignment']:.2f}% |")
    lines+=['','差值与描述性95%区间（正数表示左侧更好；仅六个既有实例）']
    for d in diff:
        lo,hi=d['descriptive_cluster_interval']
        lines.append(f"{LABELS[d['left']]} 相对 {LABELS[d['right']]}：{d['metric']} {d['mean']:.3f}，区间[{lo:.3f},{hi:.3f}]。")
    lines+=['','保留失败的补充比较：精修后先比较总超载，超载相同再比较路程。',
            '每实例两组全部输出交叉比较，平局计半；先算实例内胜率，再对六实例等权平均。',
            '这是补充描述分析，不是事先冻结的主要指标；交叉配对不当作独立样本。',
            '| 左侧方法 | 右侧方法 | 左侧胜率 | 按实例重抽样描述性95%区间 |',
            '|---|---|---:|---:|']
    for w in wins:
        lo,hi=w['descriptive_cluster_interval']
        lines.append(f"| {LABELS[w['left']]} | {LABELS[w['right']]} | {100*w['mean']:.2f}% | [{100*lo:.2f}%, {100*hi:.2f}%] |")
    lines+=['','全局随机达到质量的已测试档位：']
    largest=max(b for b in W.BUDGETS+EXTRA_BUDGETS if f'global_{b}' in summary)
    for method,x in crossings.items():
        first=x['first_tested_budget']
        lines.append(f"{LABELS[method]}：{first if first else f'截至{largest}份尚未达到'}；按精修后平均差距判断，不声称等成本。")
    lines+=['','三设备的预设判断：']
    for device,decision in decisions.items():
        if decision['state']!='完整':lines.append(f'{device}：等待完整且通过校验的读数。');continue
        lines.append(f"{device}：训练后精修前可行{decision['before_feasible']}，精修后可行{decision['after_feasible']}；"
                     f"训练后差距{num(decision['trained_gap'])}%、未训练参数{num(decision['untrained_gap'])}%、"
                     f"随机1份{num(decision['random_one_gap'])}%、同测量次数无噪声模拟{num(decision['same_measurements_simulation_gap'])}%。"
                     f"训练后均值同时优于未训练参数和随机1份：{'是' if decision['mean_quality_conditions_met'] else '否'}。")
    lines+=['这些是既有六实例的平均质量判断；若实例间差异区间跨零，不能称稳定优于。更好设备的结果也不能单独证明硬件不可替代。']
    lines+=['','逐实例实际路程与数据集参考值','| 实例 | 客户 | 车辆 | 数据集参考值 | 不重复分配上限K^n |','|---|---:|---:|---:|---:|']
    for n,x in data['instances'].items():lines.append(f"| {n} | {x['n']} | {x['K']} | {x['bks']} | {x['K']**x['n']} |")
    lines+=['参考值来自既有数据集。本轮未调用精确求解器新增最优性证明；也没有把启发式算法结果标为最优值。','',
      '| 实例 | 方法 | 精修前可行 | 精修前平均路程† | 精修后可行 | 精修后平均路程 | 精修后差距 |','|---|---|---:|---:|---:|---:|---:|']
    for n in names:
        for m in LABELS:
            if m not in summary or n not in summary[m]['per_instance']:continue
            s=summary[m]['per_instance'][n];b,a=s['before'],s['after'];count=s['count']
            lines.append(f"| {n} | {LABELS[m]} | {b['feasible']}/{count} | {b['distance']:.2f} | {a['feasible']}/{count} | {a['distance']:.2f} | {num(a['gap'])}% |")
    lines+=['','研究依据与本实验的对应关系：',
      '2025年Nature Communications的泡利关联编码（Pauli Correlation Encoding，PCE）论文给出了关联符号解码与经典后处理框架；不意味着本配送任务已获得量子优势。https://www.nature.com/articles/s41467-024-55346-z',
      '2026年7月电力组合优化预印本用同一贪心后处理的随机初值对照，区分原始读出信息与后处理效果；报告接近零的关联在4096次测量下易翻转。可借鉴其对照思路，不能把其无硬容量约束任务的结果移用于本配送任务。https://arxiv.org/html/2607.24722v1',
      '2026年6月PCE基准预印本分别研究精确期望、有限测量和噪声；其研究明确未覆盖车辆路径问题。可借鉴分层诊断，不作为本算法性能证明。https://arxiv.org/html/2606.18914v1','',
      '能力边界：本实验最多支持读出信息与混合求解质量的作用判断，不能证明量子加速、硬件不可替代、或优于强经典算法。',
      f'复核通过：{checked}份输出前后评价，{len(seen)}条精修过程独立重算。']
    (HERE/'最新修正版实验结果.txt').write_text('\n'.join(lines),encoding='utf-8')
    for m,s in summary.items():print(m,s['before']['feasible'],s['count'],s['after']['gap'],s.get('sign_agreement'),flush=True)
    print('HARDWARE',status,flush=True)

if __name__=='__main__':main()
