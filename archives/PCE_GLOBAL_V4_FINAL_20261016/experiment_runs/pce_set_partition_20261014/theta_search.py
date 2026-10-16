"""模型修正第五版：在表示的参数空间里直接用离散目标搜索（不做连续松弛）。

依据：可表示性测试表明好解在 10 比特线路的可表示范围内；连续松弛训练找不到，是因为损失是平均场形式。
做法：模拟退火，每步扰动一个参数（正态，步长随温度缩小），读出正负号 → 分组 → 离散代理目标（与训练目标同一函数），
按 Metropolis 规则接受。所有方法用相同的目标评价次数 BUDGET。
方法：Qθ（纠缠线路 6 层，120 参数）、Pθ（去掉 CZ）、LRθ（经典低秩 120 参数）、SAv（直接在客户分组上退火，单客户换车）。
报告：搜索结束时的最好解；再接同一个单客户最陡下降后的结果。
"""
import os
for v in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[v] = '1'
import sys, json, time, zlib
from pathlib import Path
import numpy as np
import assign_pce as A
import pce_scale as ps
from descent_test import descend

HERE = Path(__file__).resolve().parent
OUT = HERE / os.environ.get('OUTDIR', 'results_theta_search')
BUDGET = int(os.environ.get('BUDGET', '20000'))
RESTARTS = int(os.environ.get('RESTARTS', '4'))
KINDS = ('Q', 'P', 'LR', 'SAv')


def proxy_of(S, v):
    bits = ((v[:, None] >> np.arange(S['B'])) & 1).astype(float)
    return float(A.loss_p(S, bits)[0])


def make_decoder(S, kind, rng):
    slots = A.slots_random(S['m'])
    if kind in ('Q', 'P'):
        circ = ps.Circuit(A.NQ, 6, kind == 'Q')
        return (lambda th: A.decode(S, (circ.expect(circ.run(th), slots) > 0).astype(int))), \
               rng.uniform(-np.pi, np.pi, circ.npar), np.pi
    F = {'X': 0, 'Y': 1, 'Z': 2}
    fi = np.array([F[s[0]] for s in slots]); ai = np.array([s[1] for s in slots]); bi = np.array([s[2] for s in slots])
    def dec(u):
        U = u.reshape(3, A.NQ, A.RANK); return A.decode(S, ((U[fi, ai] * U[fi, bi]).sum(1) > 0).astype(int))
    return dec, rng.normal(0, 1, 3 * A.NQ * A.RANK), 1.0


def search_params(S, kind, rng, T0=1.0, T1=0.005):
    dec, x, scale = make_decoder(S, kind, rng)
    v = dec(x); E = proxy_of(S, v); best = (E, v.copy()); trace = []
    for t in range(BUDGET):
        frac = t / BUDGET; T = T0 * (T1 / T0) ** frac; sigma = scale * (0.5 * (1 - frac) + 0.05)
        y = x.copy(); j = rng.integers(len(x)); y[j] += rng.normal(0, sigma)
        w = dec(y); F = proxy_of(S, w)
        if F <= E or rng.random() < np.exp(-(F - E) / T):
            x, v, E = y, w, F
            if E < best[0]: best = (E, v.copy())
        if (t + 1) % 2000 == 0: trace.append(best[0])
    return best[1], trace


def search_assign(S, rng, T0=1.0, T1=0.005):
    n, K, q, cap = S['n'], S['K'], S['q'], S['cap']
    v = rng.integers(0, K, n); E = proxy_of(S, v); best = (E, v.copy()); trace = []
    for t in range(BUDGET):
        T = T0 * (T1 / T0) ** (t / BUDGET)
        w = v.copy(); i = rng.integers(n); b = rng.integers(K - 1); w[i] = b + (b >= v[i])
        F = proxy_of(S, w)
        if F <= E or rng.random() < np.exp(-(F - E) / T):
            v, E = w, F
            if E < best[0]: best = (E, v.copy())
        if (t + 1) % 2000 == 0: trace.append(best[0])
    return best[1], trace


def job(args):
    name, kind, r = args
    S = A.setup(name); rng = np.random.default_rng([zlib.crc32(name.encode()), zlib.crc32(kind.encode()), r, 5])
    t0 = time.time()
    v, trace = search_assign(S, rng) if kind == 'SAv' else search_params(S, kind, rng)
    raw = A.assess(S, v); w, steps = descend(S, v); post = A.assess(S, w)
    return dict(instance=name, kind=kind, restart=r, budget=BUDGET, raw=raw, post=post, descent_steps=steps,
                trace=trace, seconds=round(time.time() - t0, 1))


def run():
    OUT.mkdir(exist_ok=False)
    jobs = [(n, k, r) for k in KINDS for n in A.SP.INSTANCES for r in range(RESTARTS)]
    from multiprocessing import Pool
    rows = []
    with Pool(int(os.environ.get('WORKERS', '7'))) as pool:
        for rec in pool.imap_unordered(job, jobs):
            rows.append(rec); (OUT / 'runs.json').write_text(json.dumps(rows, ensure_ascii=False))
            print(len(rows), '/', len(jobs), rec['instance'], rec['kind'], rec['raw']['feasible'], rec['raw']['gap_bks_percent'], rec['seconds'], flush=True)
    report()


def report():
    rows = json.loads((OUT / 'runs.json').read_text(encoding='utf-8'))
    names = list(A.SP.INSTANCES)
    L = [f'参数空间离散搜索（每次 {BUDGET} 次目标评价）；每格：合法数/起点数，合法解平均差距%（接最陡下降后的平均差距%）', '']
    L.append(f"{'方法':5s}" + ''.join(f'{n:>22s}' for n in names) + f"{'平均差距':>9s}{'下降后':>8s}")
    for k in KINDS:
        cells, g_all, p_all = [], [], []
        for n in names:
            rs = [r for r in rows if r['instance'] == n and r['kind'] == k]
            f = [r['raw']['gap_bks_percent'] for r in rs if r['raw']['feasible']]
            p = [r['post']['gap_bks_percent'] for r in rs if r['post']['feasible']]
            g_all += f; p_all += p
            cells.append(f"{len(f)}/{len(rs)},{np.mean(f) if f else float('nan'):5.1f}({np.mean(p) if p else float('nan'):4.1f})")
        L.append(f'{k:5s}' + ''.join(f'{c:>22s}' for c in cells) + f'{np.mean(g_all) if g_all else float("nan"):9.1f}{np.mean(p_all) if p_all else float("nan"):8.1f}')
    L.append('')
    L.append('最好代理值随评价次数的变化（所有实例平均，每 2000 次一个点）：')
    for k in KINDS:
        tr = np.array([r['trace'] for r in rows if r['kind'] == k])
        L.append(f'  {k:4s}' + ' '.join(f'{x:6.2f}' for x in tr.mean(0)))
    txt = '\n'.join(L)
    (OUT / 'REPORT.txt').write_text(txt, encoding='utf-8'); print(txt)


if __name__ == '__main__':
    {'run': run, 'report': report}[sys.argv[1]]()
