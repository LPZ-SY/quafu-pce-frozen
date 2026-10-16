"""逐批原始读出为主；校正与理想有限测量为诊断，容量失败不排除。"""
import run as R
import numpy as np
import zlib
HERE=R.HERE/'hardware'

def main():
    D=R.C.read(HERE/'logical_tasks.json');ledger=R.C.read(HERE/'ledger.json')
    bits=((np.arange(256)[:,None]>>(7-np.arange(8)))&1);signs=1-2*bits
    def prob(label):
        raw=R.C.read(HERE/'raw'/f'{label}.json');counts=raw.get('count') or raw.get('counts')
        assert sum(counts.values())==4096
        p=np.zeros(256)
        for word,count in counts.items():
            word=str(word).replace(' ','');assert len(word)==8 and set(word)<=set('01')
            p[int(word[::-1],2)]+=count/4096
        return p
    rows=[];controls=[]
    for batch in range(3):
        if any(ledger.get(f'{k}_b{batch}',{}).get('status')!='DONE' for k in ('zero','one','order')):continue
        zero,one,order=[prob(f'{k}_b{batch}') for k in ('zero','one','order')]
        measured=format(int(np.argmax(order)),'08b');assert measured=='00101101'
        p0=np.array([zero[bits[:,q]==0].sum() for q in range(8)])
        p1=np.array([one[bits[:,q]==1].sum() for q in range(8)])
        inv=[np.linalg.inv(np.array([[p0[q],1-p1[q]],[1-p0[q],p1[q]]])) for q in range(8)]
        controls.append(dict(batch=batch,bit_order=measured,p0=p0.tolist(),p1=p1.tolist()))
        for key,src in D['sources'].items():
            labels={b:f'{key}_{b}_b{batch}' for b in 'ZXY'}
            if any(ledger.get(l,{}).get('status')!='DONE' for l in labels.values()):continue
            probs={b:prob(label) for b,label in labels.items()}
            S=R.C.A.setup(src['instance'])
            for mode in ('raw','mitigated'):
                pp={b:p.copy() for b,p in probs.items()}
                if mode=='mitigated':
                    for b,p in pp.items():
                        a=p.reshape((2,)*8)
                        for q in range(8):a=np.moveaxis(np.tensordot(inv[q],np.moveaxis(a,q,0),axes=(1,0)),0,q)
                        pp[b]=a.reshape(-1)
                e=np.array([pp[f]@np.prod(signs[:,qs],axis=1) for f,qs in src['slots']])
                v=R.decode(e,S['n'],S['K'])
                row=R.C.process(S,v,'hardware_'+mode,'dongling_real_route',dict(source=key,hardware_batch=batch,
                    outputs=e.tolist(),correlator_sign_agreement=float(np.mean((e>0)==(np.array(src['outputs'])>0))),
                    same_assignment=bool(np.array_equal(v,src['initial_assignment'])),
                    simulated_before=src['before'],simulated_after=src['after']))
                rows.append(row)
    diagnostics=[]
    # 真机之前已经固定的12条线路：理想无硬件噪声，仅有限测量，检查失败来源。
    for key,src in D['sources'].items():
        S=R.C.A.setup(src['instance']);slots=[(f,tuple(q)) for f,q in src['slots']]
        c=R.K.CircuitK(8,4,True).prepare(slots);pp=c.probs(c.run(np.array(src['theta'])))
        rng=np.random.default_rng(zlib.crc32(('finite:'+key).encode()));ok=same=0
        for rep in range(32):
            e=np.zeros(len(slots))
            for f,(idx,sgn) in c.groups.items():
                p=pp[f]/pp[f].sum();e[idx]=rng.multinomial(4096,p)@sgn/4096
            v=R.decode(e,S['n'],S['K']);ok+=R.C.A.assess(S,v)['feasible'];same+=np.array_equal(v,src['initial_assignment'])
        diagnostics.append(dict(source=key,instance=src['instance'],repeats=32,feasible=int(ok),same=int(same),
            min_abs=float(np.min(np.abs(src['outputs']))),fraction_abs_below_002=float(np.mean(np.abs(src['outputs'])<.02))))
    R.save('hardware_analysis.json',dict(controls=controls,rows=rows,finite_measurement=diagnostics,
        done_tasks=sum(x['status']=='DONE' for x in ledger.values()),total_tasks=len(ledger)))
    lines=['东岭真实路程训练读出实验','',
        '固定6实例、每例第0/1训练输出，共12条线路，三批独立读出，逐批三个方向各4096次。',
        '主要结果为原始读出；校正单列。所有容量失败保留，统一精修计入修复工作。模拟列来自完全相同的参数。',
        f"已返回{sum(x['status']=='DONE' for x in ledger.values())}/{len(ledger)}项任务；以下只含三个方向及校验均齐全的分车。",'',
        '| 方式 | 分车数 | 精修前可行数 | 完整复现模拟分车 | 精修后可行数 | 精修后平均最优差距 |',
        '|---|---:|---:|---:|---:|---:|']
    for mode in ('raw','mitigated'):
        rs=[r for r in rows if r['method']=='hardware_'+mode];good=[r['after']['gap_bks_percent'] for r in rs if r['after']['feasible']]
        lines.append(f"| {'原始' if mode=='raw' else '校正'} | {len(rs)} | {sum(r['before']['feasible'] for r in rs)} | {sum(r['same_assignment'] for r in rs)} | {len(good)} | {np.mean(good) if good else float('nan'):.2f}% |")
    lines+=['','逐条原始读出与同线路模拟（†超载不可行）','| 线路 | 批次 | 模拟前→后 | 真机前→后 | 总超载量 |','|---|---:|---:|---:|---:|']
    for r in rows:
        if r['method']!='hardware_raw':continue
        b,a=r['before'],r['after'];sb,sa=r['simulated_before'],r['simulated_after']
        lines.append(f"| {r['source']} | {r['hardware_batch']} | {sb['distance']:.0f}→{sa['distance']:.0f} | {b['distance']:.0f}{'' if b['feasible'] else '†'}→{a['distance']:.0f}{'' if a['feasible'] else '†'} | {b['overload']:.0f} |")
    lines+=['','无硬件噪声、每方向4096次测量的模拟诊断：',
        f"12条线路各32次，共384次；可行{sum(x['feasible'] for x in diagnostics)}/384；完整复现分车{sum(x['same'] for x in diagnostics)}/384。", 
        '这与量子训练的精确关联期望不同；有限测量失败说明不能把全部失败归咎于芯片噪声。']
    (R.HERE/'东岭真机结果.txt').write_text('\n'.join(lines),encoding='utf-8')
    for mode in ('raw','mitigated'):
        rs=[r for r in rows if r['method']=='hardware_'+mode]
        print(mode,len(rs),sum(r['before']['feasible'] for r in rs),sum(r['after']['feasible'] for r in rs),np.mean([r['after']['gap_bks_percent'] for r in rs if r['after']['feasible']]),flush=True)
    print('finite',sum(x['feasible'] for x in diagnostics),sum(x['same'] for x in diagnostics),flush=True)

if __name__=='__main__':main()
