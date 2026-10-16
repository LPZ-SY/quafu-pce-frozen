"""三批同线路读数合并为一次关联估计，再按原规则解码与精修。"""
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
    for word,n in counts.items():p[int(str(word).replace(' ','')[::-1],2)]+=n/4096
    return p

def main():
    inputs=W.read(HERE/'inputs.json');names=list(inputs['instances'])
    earlier=W.read(HERE/'analysis.json')['summary']
    outputs=[]
    for device,(dirname,suffix) in DEVICES.items():
        folder=HERE/dirname
        assert len(W.read(folder/'ledger.json'))==225
        for source,src in inputs['sources'].items():
            S=W.load_setup(inputs['instances'][src['instance']])
            groups=W.make_groups(src['slots']);target=np.array(src['outputs'])
            for arm in ('stable','untrained'):
                pp={basis:np.mean([probability(folder,f'{source}_{arm}_{basis}_b{batch}')
                                   for batch in range(3)],axis=0) for basis in 'ZXY'}
                e,_=W.expectations(groups,pp)
                v=decode(e,S['n'],S['K'])
                outputs.append(W.process(S,v,'pooled_readout',dict(device=device,arm=arm,source=source,
                    sign_agreement=float(np.mean((e>0)==(target>0))),
                    measurements_per_basis=12288)))
    summaries={}
    for device,(_,suffix) in DEVICES.items():
        summaries[device]={}
        for arm in ('stable','untrained'):
            rr=[r for r in outputs if r['device']==device and r['arm']==arm]
            assert len(rr)==12
            per={}
            for name in names:
                subset=[r for r in rr if r['instance']==name]
                assert len(subset)==2
                per[name]=dict(before_feasible=sum(r['before']['feasible'] for r in subset),
                    after_feasible=sum(r['after']['feasible'] for r in subset),
                    gap=float(np.mean([r['after']['gap_bks_percent'] for r in subset]))
                        if all(r['after']['feasible'] for r in subset) else None)
            old=earlier[arm+'_hardware'+suffix]
            summaries[device][arm]=dict(count=12,
                before_feasible=sum(x['before_feasible'] for x in per.values()),
                after_feasible=sum(x['after_feasible'] for x in per.values()),
                gap=float(np.mean([x['gap'] for x in per.values()]))
                    if all(x['gap'] is not None for x in per.values()) else None,
                sign_agreement=float(np.mean([r['sign_agreement'] for r in rr])),
                original_gap=old['after']['gap'],original_sign_agreement=old['sign_agreement'],
                per_instance=per)
    W.save(HERE/'pooled_readout_results.json',dict(summaries=summaries,rows=outputs,
        note='事后诊断：三个4096次读出按相同线路、相同测量方向逐位合并为12288次估计，解码一份分车并统一精修。不是与单批等测量成本。'))
    lines=['三批读数合并实验：完整规模真机','',
           '同一线路、同一测量方向的三批读数合并，得到每方向12288次测量的一份分车，精修一次。',
           '| 设备 | 线路 | 单批精修后差距 | 合并精修后差距 | 合并精修后可行 | 符号一致率：单批→合并 |',
           '|---|---|---:|---:|---:|---:|']
    for device,arms in summaries.items():
        for arm,x in arms.items():
            a='训练后' if arm=='stable' else '未训练参数'
            new=f"{x['gap']:.3f}%" if x['gap'] is not None else '有不可行输出'
            lines.append(f"| {device} | {a} | {x['original_gap']:.3f}% | {new} | "
                         f"{x['after_feasible']}/12 | "
                         f"{100*x['original_sign_agreement']:.2f}%→{100*x['sign_agreement']:.2f}% |")
    lines+=['','合并后每条线路只输出一份，单批三次则输出三份；两种结果不能视为相同测量成本或相同输出数量。',
            '若合并未改善关联符号，提示至少单纯把4096次增为12288次不足以消除偏差；不能从此推断任意更高测量次数也无效。']
    (HERE/'合并读数实验结果.txt').write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps(summaries,ensure_ascii=False))

if __name__=='__main__':main()
