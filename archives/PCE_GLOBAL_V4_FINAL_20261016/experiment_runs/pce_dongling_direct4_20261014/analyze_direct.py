"""直接优化真机分析：真机读出的方案是否合法、是否与模拟读出相同、路程差距；正负号一致率按信号强度分档。"""
import json, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'pce_set_partition_20261014'))
sys.path.insert(0, str(HERE))
import assign_pce as A
import fold
import analyze as AN
from descent_test import descend

AN.HERE = HERE
AN.D = json.loads((HERE / 'logical_tasks.json').read_text(encoding='utf-8'))
AN.M = json.loads((HERE / 'manifest.json').read_text(encoding='utf-8'))
AN.LEDGER = json.loads((HERE / 'ledger.json').read_text(encoding='utf-8'))


def main():
    D = AN.D
    L = ['东岭真机：完整规模直接优化（8 比特三比特关联；线路由 CVRP 目标直接优化得到，没有经典答案）', '']
    for lab in ('order', 'order_end'):
        if AN.done(lab):
            c = AN.counts(lab); L.append(f'比特顺序校验 {lab}：众数 {max(c, key=c.get)}')
    p0, p1, inv = AN.readout_inverse()
    L.append('P(0|0)：' + ' '.join(f'{x:.3f}' for x in p0) + '；P(1|1)：' + ' '.join(f'{x:.3f}' for x in p1))
    L.append('')
    L.append(f"{'线路':18s}{'变量':>5s}{'模拟合法':>8s}{'模拟差距':>8s}{'信号≥0.3':>9s}{'方式':>5s}{'一致率':>7s}{'缩放':>6s}"
             f"{'真机合法':>8s}{'同模拟':>7s}{'真机差距':>8s}{'下降后':>8s}")
    rows, binned = [], {}
    for key, src in D['sources'].items():
        S = A.setup(src['instance'])
        slots = [(f, list(q)) for f, q in src['slots']]
        ideal = np.array(src['outputs'])
        vs = A.decode(S, (ideal > 0).astype(int))
        reps = [r for r in (1, 2) if all(AN.done(f'{key}_{b}_r{r}') for b in 'ZXY')]
        if not reps:
            continue
        for mode in ('raw', 'mitigated'):
            per = []
            for r in reps:
                pr = {b: AN.prob_vec(AN.counts(f'{key}_{b}_r{r}')) for b in 'ZXY'}
                if mode == 'mitigated':
                    pr = {b: AN.mitigate(p, inv) for b, p in pr.items()}
                per.append(AN.correlators(pr, slots))
            e = np.mean(per, axis=0)
            vh = A.decode(S, (e > 0).astype(int)); d = A.assess(S, vh)
            w, _ = descend(S, vh); dd = A.assess(S, w)
            row = dict(key=key, mode=mode, m=src['m'], sim_feasible=src['assess']['feasible'], sim_gap=src['assess']['gap_bks_percent'],
                       frac_ge_03=src['frac_ge_03'], agree=float(np.mean(np.sign(e) == np.sign(ideal))),
                       slope=float(np.polyfit(ideal, e, 1)[0]), hw_feasible=d['feasible'], same=bool(np.all(vh == vs)),
                       hw_gap=d['gap_bks_percent'], descent_gap=dd['gap_bks_percent'], invalid=d['invalid_codes'], overload=d['overload'],
                       per_repeat_feasible=[A.assess(S, A.decode(S, (x > 0).astype(int)))['feasible'] for x in per], outputs=e.tolist())
            rows.append(row)
            if mode == 'mitigated':
                for a, b in ((0, .1), (.1, .2), (.2, .3), (.3, 2)):
                    msk = (np.abs(ideal) >= a) & (np.abs(ideal) < b)
                    binned.setdefault((src['kind'] + str(src['layers']), a), []).extend((np.sign(e[msk]) == np.sign(ideal[msk])).tolist())
            f = lambda x: '—' if x is None else f'{x:.1f}'
            L.append(f"{key:18s}{row['m']:5d}{('是' if row['sim_feasible'] else '否'):>8s}{f(row['sim_gap']):>8s}{row['frac_ge_03']:9.2f}"
                     f"{('原始' if mode == 'raw' else '校正'):>5s}{row['agree']:7.3f}{row['slope']:6.2f}{('是' if row['hw_feasible'] else '否'):>8s}"
                     f"{('是' if row['same'] else '否'):>7s}{f(row['hw_gap']):>8s}{f(row['descent_gap']):>8s}")
    L.append('')
    L.append('按模拟信号强度分档的真机一致率（校正后）：')
    for (cfg, a), v in sorted(binned.items()):
        L.append(f'  {cfg} |信号|≥{a:.1f} 档：{np.mean(v):.3f}（{len(v)} 个）')
    L.append('')
    for cfg in sorted({r['key'].split('_')[1] for r in rows}):
        rs = [r for r in rows if r['key'].endswith(cfg) and r['mode'] == 'mitigated']
        L.append(f"{cfg}：真机合法 {sum(r['hw_feasible'] for r in rs)}/{len(rs)}，与模拟同方案 {sum(r['same'] for r in rs)}/{len(rs)}，"
                 f"一致率 {np.mean([r['agree'] for r in rs]):.3f}")
    txt = '\n'.join(L)
    (HERE / 'REPORT.txt').write_text(txt, encoding='utf-8')
    (HERE / 'RESULT.json').write_text(json.dumps(rows, ensure_ascii=False), encoding='utf-8')
    print(txt)


if __name__ == '__main__':
    main()
