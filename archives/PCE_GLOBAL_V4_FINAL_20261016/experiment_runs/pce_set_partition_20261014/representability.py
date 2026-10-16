"""可表示性测试：10 比特线路能否“装下”一个好的 CVRP 解？

做法：用模拟退火在约束内嵌编码上找到好解（每个实例 1 个目标），把它的正负号当作监督目标，
让各模型直接拟合：损失 = −Σ_j t_j·tanh(α·e_j)。拟合成功意味着这个解在该模型的可表示集合内。
区分两种失败：表示不出（拟合也失败）与训练找不到（拟合成功、但按 CVRP 目标训练找不到）。
模型：纠缠线路 Q（层数 4/6/10/16）、去掉纠缠门 P（6 层）、参数量相同的经典低秩 LR（120 参数）。
指标：符号匹配比例、完全匹配的起点数、读出解的可行性与路程。
"""
import os
for v in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[v] = '1'
import sys, json, time, zlib
from pathlib import Path
import numpy as np
import assign_pce as A
import pce_scale as ps

HERE = Path(__file__).resolve().parent
OUT = HERE / 'results_representability'
RESTARTS = 2
STEPS = 2000
CONFIGS = [('Q', 4), ('Q', 6), ('Q', 10), ('Q', 16), ('P', 6), ('LR', 6)]


def target(name):
    S = A.setup(name)
    best = None
    for r in range(8):
        v = A.anneal(S, np.random.default_rng([zlib.crc32(name.encode()), 77, r]))
        d = A.assess(S, v)
        if d['feasible'] and (best is None or d['distance'] < best[1]['distance']):
            best = (v, d)
    v, d = best
    bits = ((v[:, None] >> np.arange(S['B'])) & 1).reshape(-1)
    return S, 2 * bits - 1, d


def fit(S, t, kind, layers, rng):
    slots = A.slots_random(S['m'])
    alpha = A.ALPHA
    if kind in ('Q', 'P'):
        circ = ps.Circuit(A.NQ, layers, kind == 'Q')
        def fun(th):
            e = circ.expect(circ.run(th), slots); s = np.tanh(alpha * e)
            return -(t * s).sum(), circ.grad(th, slots, -t * alpha * (1 - s ** 2))
        th, _, _ = ps.adam(fun, rng.uniform(-np.pi, np.pi, circ.npar), steps=STEPS)
        return circ.expect(circ.run(th), slots), circ.npar
    F = {'X': 0, 'Y': 1, 'Z': 2}
    fi = np.array([F[s[0]] for s in slots]); ai = np.array([s[1] for s in slots]); bi = np.array([s[2] for s in slots])
    R = A.RANK
    def out(u):
        U = u.reshape(3, A.NQ, R); return (U[fi, ai] * U[fi, bi]).sum(1) / R
    def fun(u):
        U = u.reshape(3, A.NQ, R); e = out(u); s = np.tanh(alpha * e); c = -t * alpha * (1 - s ** 2)
        g = np.zeros_like(U)
        np.add.at(g, (fi, ai), (c / R)[:, None] * U[fi, bi]); np.add.at(g, (fi, bi), (c / R)[:, None] * U[fi, ai])
        return -(t * s).sum(), g.reshape(-1)
    u, _, _ = ps.adam(fun, rng.normal(0, .3, 3 * A.NQ * R) / np.sqrt(alpha), steps=STEPS)
    return out(u), 3 * A.NQ * R


def job(args):
    name, kind, layers, r = args
    S, t, ref = target(name)
    rng = np.random.default_rng([zlib.crc32(name.encode()), zlib.crc32(kind.encode()), layers, r])
    t0 = time.time()
    e, npar = fit(S, t, kind, layers, rng)
    match = (np.sign(e) == t)
    d = A.assess(S, A.decode(S, (e > 0).astype(int)))
    return dict(instance=name, kind=kind, layers=layers, restart=r, parameters=npar, m=S['m'],
                match_rate=float(match.mean()), exact=bool(match.all()), mismatches=int((~match).sum()),
                min_margin=float(np.abs(e).min()), target_distance=ref['distance'], target_gap=ref['gap_bks_percent'],
                decoded=d, seconds=round(time.time() - t0, 1))


def run():
    OUT.mkdir(exist_ok=False)
    jobs = [(n, k, L, r) for n in A.SP.INSTANCES for k, L in CONFIGS for r in range(RESTARTS)]
    from multiprocessing import Pool
    rows = []
    with Pool(int(os.environ.get('WORKERS', '7'))) as pool:
        for rec in pool.imap_unordered(job, jobs):
            rows.append(rec); (OUT / 'runs.json').write_text(json.dumps(rows, ensure_ascii=False))
            print(len(rows), '/', len(jobs), rec['instance'], rec['kind'], rec['layers'], rec['match_rate'], rec['seconds'], flush=True)
    report()


def report():
    rows = json.loads((OUT / 'runs.json').read_text(encoding='utf-8'))
    L = ['可表示性测试：拟合模拟退火好解的正负号（每格：平均符号匹配率 / 完全匹配起点数 / 读出可行起点数）', '']
    names = list(A.SP.INSTANCES)
    L.append(f"{'模型':10s}{'参数':>6s}" + ''.join(f'{n:>16s}' for n in names))
    for k, Lyr in CONFIGS:
        cells = []; npar = None
        for n in names:
            rs = [r for r in rows if r['instance'] == n and r['kind'] == k and r['layers'] == Lyr]
            if not rs: cells.append(f"{'—':>16s}"); continue
            npar = rs[0]['parameters']
            cells.append(f"{np.mean([r['match_rate'] for r in rs]):6.3f}/{sum(r['exact'] for r in rs)}/{sum(r['decoded']['feasible'] for r in rs):<5d}")
        L.append(f"{k + str(Lyr) + '层':10s}{(npar or 0):6d}" + ''.join(f'{c:>16s}' for c in cells))
    L.append('')
    L.append('目标解（模拟退火）的路程差距%：' + '，'.join(f"{n} {next(r['target_gap'] for r in rows if r['instance'] == n):.1f}" for n in names if any(r['instance'] == n for r in rows)))
    txt = '\n'.join(L)
    (OUT / 'REPORT.txt').write_text(txt, encoding='utf-8'); print(txt)


if __name__ == '__main__':
    {'run': run, 'report': report}[sys.argv[1]]()
