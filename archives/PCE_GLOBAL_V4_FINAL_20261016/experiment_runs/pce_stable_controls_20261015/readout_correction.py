"""仅用同批零态/壹态校验线路校正三体关联读出；不依配送答案调参。"""
import json
from pathlib import Path
import numpy as np
import worker as W
from shared import decode

HERE=Path(__file__).resolve().parent
DEVICES={'Dongling':('hardware',''), 'Shenglian':('hardware_shenglian','_shenglian')}

def probability(folder,label):
    raw=W.read(folder/'raw'/f'{label}.json')
    counts=raw.get('count') or raw.get('counts')
    assert sum(counts.values())==4096
    p=np.zeros(256)
    for word,n in counts.items():
        word=str(word).replace(' ','');assert len(word)==8 and set(word)<=set('01')
        p[int(word[::-1],2)]+=n/4096
    return p

def calibration(folder,batch,signs):
    zero=probability(folder,f'zero_b{batch}')
    one=probability(folder,f'one_b{batch}')
    m0=zero@signs;m1=one@signs
    offset=(m0+m1)/2;scale=(m0-m1)/2
    assert np.all(scale>.5) and np.all(scale<=1)
    return offset,scale

def corrected_expectations(slots,pp,signs,offset,scale):
    e=np.empty(len(slots))
    for j,(basis,qubits) in enumerate(slots):
        factors=(signs[:,qubits]-offset[qubits])/scale[qubits]
        e[j]=pp[basis]@np.prod(factors,axis=1)
    return e

def summarize(rows,names):
    result={}
    for device in DEVICES:
        for arm in ('stable','untrained'):
            rr=[r for r in rows if r['device']==device and r['arm']==arm]
            assert len(rr)==36
            per={}
            for name in names:
                subset=[r for r in rr if r['instance']==name]
                assert len(subset)==6
                per[name]=dict(before_feasible=sum(r['before']['feasible'] for r in subset),
                    after_feasible=sum(r['after']['feasible'] for r in subset),
                    gap=float(np.mean([r['after']['gap_bks_percent'] for r in subset]))
                        if all(r['after']['feasible'] for r in subset) else None,
                    sign_agreement=float(np.mean([r['sign_agreement'] for r in subset])))
            good=all(x['gap'] is not None for x in per.values())
            result[f'{device}_{arm}']=dict(count=36,
                before_feasible=sum(x['before_feasible'] for x in per.values()),
                after_feasible=sum(x['after_feasible'] for x in per.values()),
                gap=float(np.mean([x['gap'] for x in per.values()])) if good else None,
                sign_agreement=float(np.mean([x['sign_agreement'] for x in per.values()])),
                per_instance=per)
    return result

def main():
    inputs=W.read(HERE/'inputs.json');names=list(inputs['instances'])
    previous=W.read(HERE/'analysis.json')['summary']
    original_rows=W.read(HERE/'hardware_analysis.json')['rows']
    original={(r['method'],r['source'],r['repeat']):r for r in original_rows}
    bits=(np.arange(256)[:,None]>>(7-np.arange(8)))&1
    signs=1-2*bits
    rows=[];calibrations={}
    for device,(dirname,suffix) in DEVICES.items():
        folder=HERE/dirname
        assert all(v['status']=='DONE' for v in W.read(folder/'ledger.json').values())
        calibrations[device]=[]
        for batch in range(3):
            offset,scale=calibration(folder,batch,signs)
            calibrations[device].append(dict(batch=batch,offset=offset.tolist(),scale=scale.tolist()))
            for source,src in inputs['sources'].items():
                pp={basis:probability(folder,f'{source}_stable_{basis}_b{batch}') for basis in 'ZXY'}
                for arm in ('stable','untrained'):
                    if arm=='untrained':
                        pp={basis:probability(folder,f'{source}_untrained_{basis}_b{batch}') for basis in 'ZXY'}
                    groups=W.make_groups(src['slots']);uncorrected,_=W.expectations(groups,pp)
                    S=W.load_setup(inputs['instances'][src['instance']])
                    uncorrected_assignment=decode(uncorrected,S['n'],S['K'])
                    key=(arm+'_hardware'+suffix,source,batch)
                    assert uncorrected_assignment.tolist()==original[key]['initial_assignment']
                    e=corrected_expectations(src['slots'],pp,signs,offset,scale)
                    v=decode(e,S['n'],S['K'])
                    target=np.array(src['outputs'])
                    outcome=W.process(S,v,'readout_corrected',dict(device=device,arm=arm,source=source,
                        repeat=batch,sign_agreement=float(np.mean((e>0)==(target>0))),
                        changed_correlator_signs=int(np.sum((e>0)!=(uncorrected>0))),
                        changed_customer_assignments=int(np.sum(v!=uncorrected_assignment))))
                    rows.append(outcome)
    corrected=summarize(rows,names)
    comparison={}
    for device,(_,suffix) in DEVICES.items():
        comparison[device]={}
        for arm in ('stable','untrained'):
            old=previous[arm+'_hardware'+suffix]
            new=corrected[f'{device}_{arm}']
            comparison[device][arm]=dict(
                before_feasible=f"{old['before']['feasible']}→{new['before_feasible']}",
                after_feasible=f"{old['after']['feasible']}→{new['after_feasible']}",
                gap_before=old['after']['gap'],gap_after=new['gap'],
                sign_agreement_before=old['sign_agreement'],
                sign_agreement_after=new['sign_agreement'])
    result=dict(calibration=calibrations,comparison=comparison,corrected_summary=corrected,
        verification=dict(total_rows=len(rows),matched_all_original_decodes=True,
                          calibration_from_same_batch_only=True),
        limitation='逐比特独立读出误差模型；未校正双比特门、相干损失和相关读出错误。'
        )
    W.save(HERE/'readout_correction_results.json',result)
    lines=['零态/壹态校验驱动的读出校正：完整规模回顾实验','',
      '对每个比特、每批从零态和壹态校验读数估计读出偏移与缩放；三体关联采用三个单比特校正因子的乘积。',
      '公式在比较配送结果前固定，没有利用最佳已知路程值选择参数。',
      '| 设备 | 线路 | 精修前可行：原始→校正 | 精修后可行：原始→校正 | 精修后差距：原始→校正 | 关联符号一致率：原始→校正 |',
      '|---|---|---:|---:|---:|---:|']
    for device,arms in comparison.items():
        for arm,x in arms.items():
            a='训练后' if arm=='stable' else '未训练参数'
            before=x['gap_before'];after=x['gap_after']
            lines.append(f"| {device} | {a} | {x['before_feasible']} | {x['after_feasible']} | "
                f"{before:.3f}%→{after:.3f}% | "
                f"{100*x['sign_agreement_before']:.2f}%→{100*x['sign_agreement_after']:.2f}% |"
                if before is not None and after is not None else
                f"| {device} | {a} | {x['before_feasible']} | {x['after_feasible']} | 有不可行输出 | — |")
    lines+=['','这项事后诊断只检验独立单比特读出校正是否能挽回现有真机结果。',
            '若校正改善现有结果，仍需新真机批次确认；若不能改善，就不能把失败简单归因于可校正读出偏差。']
    (HERE/'读出校正实验结果.txt').write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps(comparison,ensure_ascii=False))

if __name__=='__main__':main()
