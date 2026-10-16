"""第二版（约束内嵌编码）结果汇总：可行率、路程差距、测量读出稳定性、配对检验。"""
import json, sys
from pathlib import Path
import numpy as np
from scipy.stats import wilcoxon

HERE = Path(__file__).resolve().parent
OUT = HERE / (sys.argv[1] if len(sys.argv) > 1 else 'results_assign')
rows = json.loads((OUT / 'runs.json').read_text(encoding='utf-8'))
names = list(dict.fromkeys(r['instance'] for r in rows))
GRAD = ('Q', 'P', 'LR', 'C1')
from pipeline import INSTANCES as BKS


def res(r, stage):
    return r['result'] if r['arm'] == 'SA' else r['by_tau'][stage]['ideal']


L = []
L.append(f'约束内嵌编码 + 泡利关联编码：10 比特，原始正负号读出，只排单车路线（{OUT.name}）')
L.append('')
for stage, title in (('cont', '加延续法后'), ('base', '只做第一阶段')):
    if stage == 'base':
        L.append('（附）只做第一阶段训练：')
    L.append(f'一、理想读出直接可行比例（{title}；每格 = 可行起点数/起点数）')
    L.append(f"{'方法':5s}" + ''.join(f'{n:>10s}' for n in names) + f"{'合计':>8s}")
    for arm in GRAD + (('SA',) if stage == 'cont' else ()):
        cells, ok, tot = [], 0, 0
        for n in names:
            rs = [r for r in rows if r['instance'] == n and r['arm'] == arm]
            f = sum(res(r, stage)['feasible'] for r in rs); ok += f; tot += len(rs)
            cells.append(f'{f:>7d}/{len(rs):<2d}')
        L.append(f'{arm:5s}' + ''.join(cells) + f'{ok / max(tot, 1):8.2f}')
    if stage == 'cont':
        L.append(f"{'RND':5s}" + ''.join(f"{next(r for r in rows if r['instance'] == n and r['arm'] == 'RND')['feasible_rate']:10.3f}" for n in names))
    L.append('')
    if stage == 'base':
        break
    L.append('二、可行解与已知最优解的差距（%）：均值 / 最好（可行起点数）')
    L.append(f"{'方法':5s}" + ''.join(f'{n:>18s}' for n in names))
    for arm in GRAD + ('SA',):
        cells = []
        for n in names:
            g = [res(r, 'cont')['gap_bks_percent'] for r in rows if r['instance'] == n and r['arm'] == arm and res(r, 'cont')['feasible']]
            cells.append(f"{np.mean(g):7.1f}/{np.min(g):5.1f}({len(g)})" if g else f"{'—':>18s}")
        L.append(f'{arm:5s}' + ''.join(f'{c:>18s}' for c in cells))
    rnd = [next(r for r in rows if r['instance'] == n and r['arm'] == 'RND') for n in names]
    L.append(f"{'RND':5s}" + ''.join(f"{('%.1f' % (100 * (r['best_distance'] / b - 1))) if r['best_distance'] else '—':>18s}"
                                     for r, b in zip(rnd, [BKS[n] for n in names])))
    L.append('')
    L.append('三、训练目标（代理）在读出解上的值，各实例平均（越低越好）：')
    for arm in GRAD + ('SA',):
        v = [np.mean([res(r, 'cont')['proxy'] for r in rows if r['instance'] == n and r['arm'] == arm]) for n in names]
        L.append(f'  {arm:4s}' + ''.join(f'{x:9.3f}' for x in v))
    L.append('')
    L.append('四、量子线路的测量读出（每方向 4096 次，重复 32 次）：')
    for arm in ('Q', 'P'):
        rs = [r for r in rows if r['arm'] == arm]
        L.append(f"  {arm}: 可行比例 {np.mean([r['by_tau']['cont']['shot_feasible_rate'] for r in rs]):.2f}；"
                 f"与理想读出完全一致 {np.mean([r['by_tau']['cont']['shot_same'] for r in rs]):.2f}；"
                 f"最小间隔中位数 {np.median([r['by_tau']['cont']['min_margin'] for r in rs]):.3f}")
    L.append('')
    L.append('五、配对比较（同一实例同一起点编号；指标 = 代理目标值，越低越好；单侧符号秩检验）：')
    for a, b in (('Q', 'P'), ('Q', 'LR'), ('Q', 'C1')):
        da, db = [], []
        for n in names:
            for k in range(8):
                ra = [r for r in rows if r['instance'] == n and r['arm'] == a and r['restart'] == k]
                rb = [r for r in rows if r['instance'] == n and r['arm'] == b and r['restart'] == k]
                if ra and rb:
                    da.append(res(ra[0], 'cont')['proxy']); db.append(res(rb[0], 'cont')['proxy'])
        da, db = np.array(da), np.array(db)
        w = int(np.sum(da < db)); l = int(np.sum(da > db))
        p = wilcoxon(da, db, alternative='less').pvalue if np.any(da != db) else 1.0
        L.append(f'  {a} 对 {b}：{a} 更好 {w} 次，更差 {l} 次，p = {p:.3g}（共 {len(da)} 对）')
txt = '\n'.join(L)
(OUT / 'REPORT.txt').write_text(txt, encoding='utf-8')
print(txt)
