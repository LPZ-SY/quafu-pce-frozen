"""全客户规模线路：分测量方向比较真机与模拟关联符号，并记录校验读数。"""
import json
from pathlib import Path
import numpy as np
import worker as W

HERE=Path(__file__).resolve().parent
FOLDERS={'Dongling':'hardware','Shenglian':'hardware_shenglian'}
MARGIN_BINS=[(0,.05),(.05,.1),(.1,.2),(.2,float('inf'))]

def probability(folder,label):
    raw=W.read(folder/'raw'/f'{label}.json')
    counts=raw.get('count') or raw.get('counts')
    assert sum(counts.values())==4096
    p=np.zeros(256)
    for word,n in counts.items():
        word=str(word).replace(' ','');assert len(word)==8
        p[int(word[::-1],2)]+=n/4096
    return p

def bit_success(folder,batch,tag):
    raw=W.read(folder/'raw'/f'{tag}_b{batch}.json')
    counts=raw.get('count') or raw.get('counts')
    expected='0' if tag=='zero' else '1'
    return sum(n*sum(bit==expected for bit in word.replace(' ',''))/8
               for word,n in counts.items())/4096

def main():
    inputs=W.read(HERE/'inputs.json')
    result={}
    for device,dirname in FOLDERS.items():
        folder=HERE/dirname
        ledger=W.read(folder/'ledger.json')
        assert len(ledger)==225 and all(v['status']=='DONE' for v in ledger.values())
        rows=[];calibration=[]
        for batch in range(3):
            calibration.append(dict(batch=batch,zero_bit_success=bit_success(folder,batch,'zero'),
                                    one_bit_success=bit_success(folder,batch,'one')))
            for key,src in inputs['sources'].items():
                pp={basis:probability(folder,f'{key}_stable_{basis}_b{batch}') for basis in 'ZXY'}
                groups=W.make_groups(src['slots']);e,_=W.expectations(groups,pp)
                ideal=np.array(src['outputs'])
                for basis,(idx,_) in groups.items():
                    if not len(idx):continue
                    rows.extend(dict(instance=src['instance'],source=key,batch=batch,basis=basis,
                                     ideal=float(ideal[i]),hardware=float(e[i]),
                                     sign_correct=bool((ideal[i]>0)==(e[i]>0))) for i in idx)
        by_basis={}
        for basis in 'ZXY':
            rr=[r for r in rows if r['basis']==basis]
            by_basis[basis]=dict(count=len(rr),sign_agreement=float(np.mean([r['sign_correct'] for r in rr])),
                mean_abs_error=float(np.mean([abs(r['hardware']-r['ideal']) for r in rr])),
                correlation_slope=float(sum(r['hardware']*r['ideal'] for r in rr)/
                    sum(r['ideal']*r['ideal'] for r in rr)))
        by_margin={}
        for lo,hi in MARGIN_BINS:
            rr=[r for r in rows if lo<=abs(r['ideal'])<hi]
            by_margin[f'{lo:g}-{hi:g}']=dict(count=len(rr),sign_agreement=
                float(np.mean([r['sign_correct'] for r in rr])) if rr else None)
        result[device]=dict(calibration=calibration,by_basis=by_basis,by_margin=by_margin,
            total_correlators=len(rows),sign_agreement=float(np.mean([r['sign_correct'] for r in rows])))
    W.save(HERE/'device_noise_diagnosis.json',result)
    lines=['完整规模真机读出诊断：东岭与圣莲','',
           '同12条训练后线路、三批读数，每个泡利关联分别与无噪声期望符号比较。',
           '| 设备 | 测量方向 | 关联数 | 符号一致率 | 平均绝对误差 | 关联衰减系数 |',
           '|---|---|---:|---:|---:|---:|']
    for device,x in result.items():
        for basis,y in x['by_basis'].items():
            lines.append(f"| {device} | {basis} | {y['count']} | {100*y['sign_agreement']:.2f}% | "
                         f"{y['mean_abs_error']:.4f} | {y['correlation_slope']:.3f} |")
    lines+=['','零态、壹态校验线路的单比特平均正确率：']
    for device,x in result.items():
        z=float(np.mean([a['zero_bit_success'] for a in x['calibration']]))
        o=float(np.mean([a['one_bit_success'] for a in x['calibration']]))
        lines.append(f'{device}：零态{100*z:.2f}%，壹态{100*o:.2f}%。')
    lines+=['','理想关联绝对值与符号保留率：','| 设备 | 理想关联绝对值范围 | 关联数 | 符号一致率 |',
            '|---|---:|---:|---:|']
    for device,x in result.items():
        for band,y in x['by_margin'].items():
            lines.append(f"| {device} | {band} | {y['count']} | {100*y['sign_agreement']:.2f}% |")
    lines+=['','零态、壹态线路只测试计算基底的准备与读出；无法单独区分双比特门误差、相干损失和其他测量方向误差。',
            '这些是诊断，不替代相同精修后的配送路程；没有从设备标称校准数据直接推断算法质量。']
    (HERE/'真机读出诊断.txt').write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps({k:dict(total_correlators=v['total_correlators'],
                             sign_agreement=v['sign_agreement'],by_basis=v['by_basis'])
                      for k,v in result.items()},ensure_ascii=False))

if __name__=='__main__':main()
