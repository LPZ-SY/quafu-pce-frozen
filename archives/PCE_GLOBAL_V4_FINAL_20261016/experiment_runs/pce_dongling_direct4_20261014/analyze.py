"""东岭真机结果分析：比特顺序校验、读出校正、正负号一致率、读出方案的可行性与路程。"""
import json, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'pce_set_partition_20261014'))
import assign_pce as A
from descent_test import descend

D = json.loads((HERE / 'logical_tasks.json').read_text(encoding='utf-8'))
M = json.loads((HERE / 'manifest.json').read_text(encoding='utf-8'))
NQ, SHOTS = D['nq'], M['shots']
LEDGER = json.loads((HERE / 'ledger.json').read_text(encoding='utf-8'))


def counts(label):
    raw = json.loads((HERE / 'raw' / f'{label}.json').read_text(encoding='utf-8'))
    src = raw.get('count') or raw.get('counts')
    out = {}
    for k, v in src.items():
        b = str(k).replace(' ', '')
        assert len(b) == NQ and not set(b) - {'0', '1'}
        out[b[::-1]] = out.get(b[::-1], 0) + int(v)          # 平台经典位高位在左 → 逻辑 q0..q7
    assert sum(out.values()) == SHOTS
    return out


def done(label):
    return LEDGER.get(label, {}).get('status') == 'DONE'


def prob_vec(c):
    p = np.zeros(2 ** NQ)
    for b, v in c.items():
        p[int(b, 2)] += v                                     # 下标：q0 为最高位，与模拟一致
    return p / p.sum()


def readout_inverse():
    z = [counts(l) for l in ('zero', 'zero_end') if done(l)]
    o = [counts(l) for l in ('one', 'one_end') if done(l)]
    p0 = np.array([np.mean([sum(v for b, v in c.items() if b[q] == '0') / SHOTS for c in z]) for q in range(NQ)])
    p1 = np.array([np.mean([sum(v for b, v in c.items() if b[q] == '1') / SHOTS for c in o]) for q in range(NQ)])
    inv = [np.linalg.inv(np.array([[p0[q], 1 - p1[q]], [1 - p0[q], p1[q]]])) for q in range(NQ)]
    return p0, p1, inv


def mitigate(p, inv):
    t = p.reshape((2,) * NQ)
    for q in range(NQ):
        t = np.moveaxis(np.tensordot(inv[q], np.moveaxis(t, q, 0), axes=(1, 0)), 0, q)
    return t.reshape(-1)


bits = ((np.arange(2 ** NQ)[:, None] >> (NQ - 1 - np.arange(NQ))) & 1)
Zm = 1 - 2 * bits


def correlators(probs, slots):
    out = np.zeros(len(slots))
    for k, (f, q) in enumerate(slots):
        out[k] = probs[f] @ np.prod(Zm[:, q], axis=1)
    return out


def main():
    L = ['东岭真机：8 比特、三比特关联、8 层，读出 6 个实例中装进线路的好方案', '']
    for lab in ('order', 'order_end'):
        if done(lab):
            c = counts(lab); mode = max(c, key=c.get)
            L.append(f'比特顺序校验 {lab}：众数 {mode}（期望 00101101），{"通过" if mode == "00101101" else "未通过"}')
    p0, p1, inv = readout_inverse()
    L.append('读出正确率 P(0|0)：' + ' '.join(f'{x:.3f}' for x in p0))
    L.append('读出正确率 P(1|1)：' + ' '.join(f'{x:.3f}' for x in p1))
    L.append('')
    rows = []
    for name, src in D['sources'].items():
        S = A.setup(name)
        slots = [(f, list(q)) for f, q in src['slots']]
        ideal = np.array(src['outputs']); target = np.array(src['target'])
        reps = [r for r in (1, 2, 3) if all(done(f'{name}_{b}_r{r}') for b in 'ZXY')]
        if not reps:
            continue
        variants = {}
        for mode in ('raw', 'mitigated'):
            per = []
            for r in reps:
                pr = {b: prob_vec(counts(f'{name}_{b}_r{r}')) for b in 'ZXY'}
                if mode == 'mitigated':
                    pr = {b: mitigate(p, inv) for b, p in pr.items()}
                per.append(correlators(pr, slots))
            variants[mode] = dict(per=per, pooled=np.mean(per, axis=0))
        row = dict(instance=name, m=src['m'], repeats=len(reps), ideal_feasible=src['decoded']['feasible'],
                   ideal_gap=src['decoded']['gap_bks_percent'], target_gap=src['target_assess']['gap_bks_percent'])
        for mode, v in variants.items():
            e = v['pooled']
            agree = float(np.mean(np.sign(e) == np.sign(ideal)))
            slope = float(np.polyfit(ideal, e, 1)[0])
            vh = A.decode(S, (e > 0).astype(int)); d = A.assess(S, vh)
            vt = A.decode(S, (target > 0).astype(int))
            w, _ = descend(S, vh); dd = A.assess(S, w)
            absI = np.abs(ideal); bins = [(0, .05), (.05, .1), (.1, .2), (.2, 1.01)]
            byb = [float(np.mean(np.sign(e[(absI >= a) & (absI < b)]) == np.sign(ideal[(absI >= a) & (absI < b)])))
                   if np.any((absI >= a) & (absI < b)) else None for a, b in bins]
            per_rep = [A.assess(S, A.decode(S, (x > 0).astype(int))) for x in v['per']]
            row[mode] = dict(sign_agree=agree, slope=slope, mae=float(np.mean(np.abs(e - ideal))),
                             customers_changed=int(np.sum(vh != vt)), feasible=d['feasible'], gap=d['gap_bks_percent'],
                             overload=d['overload'], invalid=d['invalid_codes'], agree_by_margin=byb,
                             per_repeat_feasible=[x['feasible'] for x in per_rep],
                             after_descent_gap=dd['gap_bks_percent'], outputs=e.tolist())
        rows.append(row)
        for mode in ('raw', 'mitigated'):
            x = row[mode]
            L.append(f"{name}（{row['m']} 变量，{len(reps)} 次重复，{('校正后' if mode == 'mitigated' else '原始')}）："
                     f"正负号一致 {x['sign_agree']:.3f}，信号缩放 {x['slope']:.2f}，换车客户 {x['customers_changed']}，"
                     f"可行 {x['feasible']}（超载 {x['overload']}，无效编号 {x['invalid']}），差距 "
                     f"{('%.1f%%' % x['gap']) if x['gap'] is not None else '—'}；按信号强度分档一致率 "
                     + '/'.join('—' if b is None else f'{b:.2f}' for b in x['agree_by_margin'])
                     + f"；局部下降后差距 {('%.1f%%' % x['after_descent_gap']) if x['after_descent_gap'] is not None else '—'}")
        L.append(f"   模拟理想读出：可行 {row['ideal_feasible']}，差距 {row['ideal_gap']}；目标解差距 {row['target_gap']:.1f}%")
    txt = '\n'.join(L)
    (HERE / 'REPORT.txt').write_text(txt, encoding='utf-8')
    (HERE / 'RESULT.json').write_text(json.dumps(dict(rows=rows, p0=p0.tolist(), p1=p1.tolist()), ensure_ascii=False), encoding='utf-8')
    print(txt)


if __name__ == '__main__':
    main()
