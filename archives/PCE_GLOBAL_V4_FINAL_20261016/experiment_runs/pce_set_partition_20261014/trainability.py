"""模型修正第三版：训练改进（可表示性测试表明好解可以被 10 比特线路表示，瓶颈在训练）。

改动：α 多阶段延续（先平滑后尖锐，参照 Usuki、Fujii 等 2026），之后接二值化延续 γ·Σ(1−s²)；
每个方法在同一组随机起点上训练，按训练目标（读出解的离散代理值）在起点之间选最好的一个，不使用路程或已知最优解做选择。
方法：Q（纠缠线路 6 层，120 参数）、LR（经典低秩 120 参数）、C1（每变量一参数）。
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
OUT = HERE / os.environ.get('OUTDIR', 'results_trainability')
ALPHAS = (3.0, 6.0, 10.0, 15.0, 25.0)
STEPS_A = 500
GAMMAS = (0.003, 0.01, 0.03, 0.1)
STEPS_G = 400
RESTARTS = int(os.environ.get('RESTARTS', '4'))
LAYERS = int(os.environ.get('LAYERS', '6'))


def schedule(make, x0):
    x = x0
    for a in ALPHAS:
        x, _, _ = ps.adam(make(a, 0.0), x, steps=STEPS_A)
    for g in GAMMAS:
        x, _, _ = ps.adam(make(ALPHAS[-1], g), x, steps=STEPS_G)
    return x


def train(S, kind, rng):
    slots = A.slots_random(S['m'])
    if kind == 'Q':
        circ = ps.Circuit(A.NQ, LAYERS, True)
        def make(al, g):
            def fun(th):
                e = circ.expect(circ.run(th), slots); s = np.tanh(al * e)
                L, gs = A.loss_s(S, s); L += g * (1 - s ** 2).sum(); gs = gs - 2 * g * s
                return L, circ.grad(th, slots, gs * al * (1 - s ** 2))
            return fun
        th = schedule(make, rng.uniform(-np.pi, np.pi, circ.npar))
        return circ.expect(circ.run(th), slots), circ.npar
    if kind == 'LR':
        F = {'X': 0, 'Y': 1, 'Z': 2}
        fi = np.array([F[s[0]] for s in slots]); ai = np.array([s[1] for s in slots]); bi = np.array([s[2] for s in slots])
        R = A.RANK
        def out(u):
            U = u.reshape(3, A.NQ, R); return (U[fi, ai] * U[fi, bi]).sum(1) / R
        def make(al, g):
            def fun(u):
                U = u.reshape(3, A.NQ, R); e = out(u); s = np.tanh(al * e)
                L, gs = A.loss_s(S, s); L += g * (1 - s ** 2).sum(); gs = gs - 2 * g * s
                c = gs * al * (1 - s ** 2); G = np.zeros_like(U)
                np.add.at(G, (fi, ai), (c / R)[:, None] * U[fi, bi]); np.add.at(G, (fi, bi), (c / R)[:, None] * U[fi, ai])
                return L, G.reshape(-1)
            return fun
        u = schedule(make, rng.normal(0, .3, 3 * A.NQ * R) / np.sqrt(ALPHAS[0]))
        return out(u), 3 * A.NQ * R
    def make(al, g):                       # C1：s = tanh(al·y/15)，与其他方法同一套 α 进度
        def fun(y):
            s = np.tanh(al * y / 15); L, gs = A.loss_s(S, s); L += g * (1 - s ** 2).sum(); gs = gs - 2 * g * s
            return L, gs * (al / 15) * (1 - s ** 2)
        return fun
    y = schedule(make, rng.normal(0, .1, S['m']))
    return y, S['m']


def job(args):
    name, kind, r = args
    S = A.setup(name); rng = np.random.default_rng([zlib.crc32(name.encode()), zlib.crc32(kind.encode()), r, 3])
    t0 = time.time()
    e, npar = train(S, kind, rng)
    d = A.assess(S, A.decode(S, (e > 0).astype(int)))
    return dict(instance=name, kind=kind, restart=r, parameters=npar, layers=LAYERS, result=d,
                min_margin=float(np.abs(e).min()), outputs=e.tolist(), seconds=round(time.time() - t0, 1))


def run():
    OUT.mkdir(exist_ok=False)
    jobs = [(n, k, r) for k in ('Q', 'LR', 'C1') for n in A.SP.INSTANCES for r in range(RESTARTS)]
    from multiprocessing import Pool
    rows = []
    with Pool(int(os.environ.get('WORKERS', '7'))) as pool:
        for rec in pool.imap_unordered(job, jobs):
            rows.append(rec); (OUT / 'runs.json').write_text(json.dumps(rows, ensure_ascii=False))
            print(len(rows), '/', len(jobs), rec['instance'], rec['kind'], rec['result']['feasible'], rec['result']['gap_bks_percent'], rec['seconds'], flush=True)
    report()


def report():
    rows = json.loads((OUT / 'runs.json').read_text(encoding='utf-8'))
    names = list(A.SP.INSTANCES)
    L = [f'训练改进（α 多阶段延续 + 二值化延续），{LAYERS} 层；每格：合法起点数/起点数，合法解平均差距%，'
         '按训练目标选出的那个起点的差距%', '']
    L.append(f"{'方法':5s}" + ''.join(f'{n:>22s}' for n in names) + f"{'合法比例':>9s}")
    for k in ('Q', 'LR', 'C1'):
        cells, ok, tot = [], 0, 0
        for n in names:
            rs = [r for r in rows if r['instance'] == n and r['kind'] == k]
            f = [r for r in rs if r['result']['feasible']]; ok += len(f); tot += len(rs)
            pick = min(rs, key=lambda r: r['result']['proxy']) if rs else None
            pg = pick['result']['gap_bks_percent'] if pick and pick['result']['feasible'] else None
            cells.append(f"{len(f)}/{len(rs)},{np.mean([r['result']['gap_bks_percent'] for r in f]) if f else float('nan'):5.1f},"
                         f"{pg if pg is not None else float('nan'):5.1f}")
        L.append(f'{k:5s}' + ''.join(f'{c:>22s}' for c in cells) + f'{ok / max(tot, 1):9.2f}')
    L.append('')
    L.append('代理目标值（各实例起点平均）：')
    for k in ('Q', 'LR', 'C1'):
        L.append(f'  {k:3s}' + ''.join(f"{np.mean([r['result']['proxy'] for r in rows if r['instance'] == n and r['kind'] == k]):9.3f}" for n in names))
    txt = '\n'.join(L)
    (OUT / 'REPORT.txt').write_text(txt, encoding='utf-8'); print(txt)


if __name__ == '__main__':
    {'run': run, 'report': report}[sys.argv[1]]()
