"""决定性实验：完整规模、量子线路直接优化 CVRP（不给任何经典答案），把真机可读出条件写进目标，然后上东岭读出。

假设：8 个比特（三比特关联，168 个变量槽）的 PCE 线路，可以直接对完整 CVRP 实例（90–132 个变量）优化，
并在东岭真机上读出合法方案。
目标 = 约束内嵌编码的离散代理目标（同车两两距离 + 超载罚 + 无效编号罚）
     + LAM · Σ_j max(0, TAU − |⟨Π_j⟩|)            ← 真机可读出条件：东岭 4 层线路上信号 ≥ 0.3 才稳定读对
搜索：线路参数空间的模拟退火（每步扰动一个参数），不做连续松弛，不使用任何经典解作为目标或起点。
方法：Q4、Q6（纠缠线路 4/6 层）、P6（去掉 CZ 的 6 层），每个实例 4 个起点。
"""
import os
for v in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[v] = '1'
import sys, json, zlib, time
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'pce_set_partition_20261014'))
sys.path.insert(0, str(HERE))
import assign_pce as A
import fold
import pce_k as K

NQ, KB = 8, 3
TAU = 0.15
BUDGET = int(os.environ.get('BUDGET', '60000'))
RESTARTS = 8
CONFIGS = (('Q', 4, 2.0), ('P', 4, 2.0))


def objective(S, e, LAM):
    v = A.decode(S, (e > 0).astype(int))
    bits = ((v[:, None] >> np.arange(S['B'])) & 1).astype(float)
    proxy = float(A.loss_p(S, bits)[0])
    weak = np.maximum(0, TAU - np.abs(e))
    return proxy + LAM * weak.sum(), proxy, v


def job(args):
    name, kind, layers, lam, r = args
    S = A.setup(name)
    slots = K.slots_k(NQ, KB, S['m'])
    c = K.CircuitK(NQ, layers, kind == 'Q').prepare(slots)
    rng = np.random.default_rng([zlib.crc32(name.encode()), zlib.crc32(kind.encode()), layers, int(lam * 10), r, 1015])
    t0 = time.time()
    x = rng.uniform(-np.pi, np.pi, c.npar)
    e = c.expect_k(c.run(x)); E, _, _ = objective(S, e, lam)
    best = (E, x.copy()); T0, T1 = 2.0, 0.005; trace = []
    for t in range(BUDGET):
        frac = t / BUDGET; T = T0 * (T1 / T0) ** frac; sigma = np.pi * (0.5 * (1 - frac) + 0.03)
        y = x.copy(); j = rng.integers(c.npar); y[j] += rng.normal(0, sigma)
        F, _, _ = objective(S, c.expect_k(c.run(y)), lam)
        if F <= E or rng.random() < np.exp(-(F - E) / T):
            x, E = y, F
            if E < best[0]: best = (E, x.copy())
        if (t + 1) % 3000 == 0: trace.append(best[0])
    th = best[1]; e = c.expect_k(c.run(th))
    _, proxy, v = objective(S, e, lam)
    d = A.assess(S, v)
    return dict(instance=name, kind=kind, layers=layers, lam=lam, restart=r, m=S['m'], theta=th.tolist(),
                slots=[[f, list(q)] for f, q in slots], outputs=e.tolist(), assess=d, proxy=proxy,
                min_abs=float(np.abs(e).min()), frac_ge_03=float(np.mean(np.abs(e) >= TAU)),
                frac_ge_02=float(np.mean(np.abs(e) >= 0.2)), trace=trace, seconds=round(time.time() - t0, 1))


def run():
    from multiprocessing import Pool
    jobs = [(n, k, L, lam, r) for n in A.SP.INSTANCES for k, L, lam in CONFIGS for r in range(RESTARTS)]
    rows = []
    with Pool(int(os.environ.get('WORKERS', '7'))) as pool:
        for rec in pool.imap_unordered(job, jobs):
            rows.append(rec); (HERE / 'search.json').write_text(json.dumps(rows, ensure_ascii=False))
            print(len(rows), '/', len(jobs), rec['instance'], rec['kind'], rec['layers'], rec['lam'], rec['assess']['feasible'],
                  rec['assess']['gap_bks_percent'], round(rec['frac_ge_03'], 2), rec['seconds'], flush=True)
    report(rows)


def report(rows=None):
    rows = rows or json.loads((HERE / 'search.json').read_text(encoding='utf-8'))
    L = ['完整规模直接优化（8 比特三比特关联，目标含"信号 ≥ 0.3"条件）；每格：合法起点数，合法解平均差距%，信号≥0.3 比例均值', '']
    names = list(A.SP.INSTANCES)
    for k, Lyr, lam in CONFIGS:
        cells = []
        for n in names:
            rs = [r for r in rows if r['instance'] == n and r['kind'] == k and r['layers'] == Lyr and r['lam'] == lam]
            f = [r['assess']['gap_bks_percent'] for r in rs if r['assess']['feasible']]
            cells.append(f"{len(f)}/{len(rs)},{np.mean(f) if f else float('nan'):5.1f},{np.mean([r['frac_ge_03'] for r in rs]):.2f}")
        L.append(f'{k}{Lyr}层 权重{lam} ' + '  '.join(f'{n}:{c}' for n, c in zip(names, cells)))
    txt = '\n'.join(L); (HERE / 'SEARCH_REPORT.txt').write_text(txt, encoding='utf-8'); print(txt)


if __name__ == '__main__':
    {'run': run, 'report': report}[sys.argv[1]]()
