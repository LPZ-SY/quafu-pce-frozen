"""模型修正第二版：约束内嵌编码。

集合划分写法失败的原因：合法组合极稀疏且彼此隔离，而泡利关联编码（PCE）的损失只用到每个变量各自的期望值（平均场松弛），
无法协调多个变量同时满足“每个客户恰好一次”。本版把这条硬约束写进编码本身：
  每个客户 i 用 B=⌈log2 K⌉ 个变量写出所在车辆编号的二进制位；任何一组正负号都对应“每个客户恰好一辆车”。
  剩下的约束只有容量（软罚）和无效编号（编号 ≥ K，软罚）。
训练目标（独立伯努利期望，与 PCE 的平均场读法一致）：
  同车客户两两距离之和（归一化）+ A·Σ_v 超载² + A·无效编号概率。
评价：按正负号读出分组；可行 = 编号全部有效且每车不超载；每组用最近邻 + 2-opt + Or-opt 排一条路线，计算真实总路程。
对照：Q（纠缠线路）、P（去掉 CZ）、LR（参数量相同的经典低秩）、C1（每变量一参数）、SA（离散分组上直接模拟退火）、RND（随机位）。
运行：python assign_pce.py run   /   python assign_pce.py report
"""
import os
for v in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[v] = '1'
import sys, json, time, hashlib, itertools, zlib
from pathlib import Path
import numpy as np
import pipeline as SP
import pce_scale as ps

HERE = Path(__file__).resolve().parent
OUT = HERE / 'results_assign'
NQ, LAYERS = 10, 6
ALPHA = 1.5 * NQ
PEN = 5.0
CAP_LIN, CAP_SQ = 20.0, 200.0          # 超载罚：超载 1% 容量约等于 0.2，远大于分组紧凑度的典型差异


def cap_pen(over):
    return CAP_LIN * over + CAP_SQ * over ** 2


def cap_grad(over):
    return np.where(over > 0, CAP_LIN + 2 * CAP_SQ * over, 0.0)
STEPS, STEPS2, TAU, W_M = 1500, 400, 0.1, 20.0
GAMMAS = (0.003, 0.01, 0.03, 0.1)        # 延续法：逐步加大二值化罚 γ·Σ(1−s²)
RESTARTS = int(os.environ.get('RESTARTS', '8'))
SHOTS, SHOT_REPS = 4096, 32
RANK = 4
ARMS = ('Q', 'P', 'LR', 'C1', 'SA', 'RND')


def setup(name):
    inst = SP.load(name)
    n, K = inst['n'], inst['K']
    B = int(np.ceil(np.log2(K))); V = 2 ** B
    D = inst['D'][1:, 1:]; Dn = D / D[np.triu_indices(n, 1)].mean() / n
    codes = ((np.arange(V)[:, None] >> np.arange(B)) & 1).astype(float)   # 编号 v 的各位
    return dict(inst=inst, n=n, K=K, B=B, V=V, m=n * B, Dn=Dn, q=inst['q'][1:].astype(float),
                cap=float(inst['cap']), codes=codes)


def probs(S, p):
    """p: (n,B) 每位取 1 的概率 → (n,V) 分到各编号的概率，以及对 p 的偏导所需的分量。"""
    n, B, V, codes = S['n'], S['B'], S['V'], S['codes']
    f = np.where(codes[None, :, :] == 1, p[:, None, :], 1 - p[:, None, :])     # (n,V,B)
    Pm = f.prod(2)
    return Pm, f


def loss_p(S, p):
    n, B, V, K, codes = S['n'], S['B'], S['V'], S['K'], S['codes']
    Pm, f = probs(S, p)
    DP = S['Dn'] @ Pm                                  # (n,V)
    Lc = 0.5 * (Pm * DP).sum()
    load = S['q'] @ Pm[:, :K]; over = np.maximum(0, load - S['cap']) / S['cap']
    Lp = cap_pen(over).sum()
    Li = PEN * Pm[:, K:].sum()
    dP = DP.copy()
    dP[:, :K] += cap_grad(over)[None, :] * S['q'][:, None] / S['cap']
    dP[:, K:] += PEN
    dp = np.zeros((n, B))
    for b in range(B):
        others = np.prod(np.delete(f, b, axis=2), axis=2)            # (n,V)
        sign = np.where(codes[:, b] == 1, 1.0, -1.0)                  # d f_b / d p_b
        dp[:, b] = (dP * others * sign[None, :]).sum(1)
    return Lc + Lp + Li, dp


def loss_s(S, s):
    p = ((1 + s) / 2).reshape(S['n'], S['B'])
    L, dp = loss_p(S, p)
    return L, 0.5 * dp.reshape(-1)


def decode(S, bits):
    v = (bits.reshape(S['n'], S['B']) * (1 << np.arange(S['B']))).sum(1).astype(int)
    return v


def assess(S, v):
    inst, K = S['inst'], S['K']
    invalid = int(np.sum(v >= K))
    loads = np.array([S['q'][v == k].sum() for k in range(K)])
    over = float(np.maximum(0, loads - S['cap']).sum())
    ok = invalid == 0 and over == 0
    dist = None
    if invalid == 0:
        dist = sum(SP.rcost(inst, frozenset(int(i) + 1 for i in np.flatnonzero(v == k))) for k in range(K) if np.any(v == k))
    return dict(feasible=bool(ok), invalid_codes=invalid, overload=over, distance=dist,
                gap_bks_percent=100 * (dist / inst['bks'] - 1) if ok else None,
                proxy=float(loss_p(S, ((v[:, None] >> np.arange(S['B'])) & 1).astype(float))[0]))


def slots_random(m):
    allslots = [(f, a, b) for f in 'ZXY' for a, b in itertools.combinations(range(NQ), 2)]
    perm = np.random.default_rng(20261014).permutation(len(allslots))[:m]
    return [allslots[i] for i in perm]


def margin(e, tau):
    v = np.maximum(0, tau - np.abs(e)); return W_M * (v ** 2).sum(), -2 * W_M * v * np.sign(e)


def train_circuit(S, slots, ent, rng):
    circ = ps.Circuit(NQ, LAYERS, ent)
    def make(g):
        def fun(th):
            e = circ.expect(circ.run(th), slots); s = np.tanh(ALPHA * e)
            L, gs = loss_s(S, s); L += g * (1 - s ** 2).sum(); gs = gs - 2 * g * s
            return L, circ.grad(th, slots, gs * ALPHA * (1 - s ** 2))
        return fun
    th0, _, _ = ps.adam(make(0.0), rng.uniform(-np.pi, np.pi, circ.npar), steps=STEPS)
    th = th0.copy()
    for g in GAMMAS:
        th, _, _ = ps.adam(make(g), th, steps=STEPS2)
    return circ, {'base': th0, 'cont': th}


def train_lowrank(S, slots, rng):
    F = {'X': 0, 'Y': 1, 'Z': 2}
    fi = np.array([F[s[0]] for s in slots]); ai = np.array([s[1] for s in slots]); bi = np.array([s[2] for s in slots])
    def out(u):
        U = u.reshape(3, NQ, RANK); return (U[fi, ai] * U[fi, bi]).sum(1) / RANK
    def make(gam):
        def fun(u):
            U = u.reshape(3, NQ, RANK); e = out(u); s = np.tanh(ALPHA * e)
            L, gs = loss_s(S, s); L += gam * (1 - s ** 2).sum(); gs = gs - 2 * gam * s
            c = gs * ALPHA * (1 - s ** 2)
            g = np.zeros_like(U)
            np.add.at(g, (fi, ai), (c / RANK)[:, None] * U[fi, bi])
            np.add.at(g, (fi, bi), (c / RANK)[:, None] * U[fi, ai])
            return L, g.reshape(-1)
        return fun
    u0, _, _ = ps.adam(make(0.0), rng.normal(0, .3, 3 * NQ * RANK) / np.sqrt(ALPHA), steps=STEPS)
    u = u0.copy()
    for g in GAMMAS:
        u, _, _ = ps.adam(make(g), u, steps=STEPS2)
    return {'base': out(u0), 'cont': out(u)}


def train_c1(S, rng):
    def make(g):
        def fun(y):
            s = np.tanh(y); L, gs = loss_s(S, s); L += g * (1 - s ** 2).sum(); gs = gs - 2 * g * s
            return L, gs * (1 - s ** 2)
        return fun
    y0, _, _ = ps.adam(make(0.0), rng.normal(0, 1, S['m']), steps=STEPS)
    y = y0.copy()
    for g in GAMMAS:
        y, _, _ = ps.adam(make(g), y, steps=STEPS2)
    return {'base': np.tanh(y0), 'cont': np.tanh(y)}


def anneal(S, rng, sweeps=3000, T0=1.0, T1=0.002):
    """离散分组上的模拟退火：每步把一个客户换到另一辆车，目标与训练目标的离散版相同。"""
    n, K, Dn, q, cap = S['n'], S['K'], S['Dn'], S['q'], S['cap']
    v = rng.integers(0, K, n); load = np.array([q[v == k].sum() for k in range(K)])
    pen = lambda ld: cap_pen(np.maximum(0, ld - cap) / cap)
    E = 0.5 * sum(Dn[i, v == v[i]].sum() for i in range(n)) + pen(load).sum()
    best, bv = E, v.copy(); total = sweeps * n
    for t in range(total):
        T = T0 * (T1 / T0) ** (t / total); i = rng.integers(n); a = v[i]; b = rng.integers(K - 1); b = b + (b >= a)
        d = Dn[i, v == b].sum() - (Dn[i, v == a].sum() - Dn[i, i])
        d += pen(load[a] - q[i]) + pen(load[b] + q[i]) - pen(load[a]) - pen(load[b])
        if d < 0 or rng.random() < np.exp(-d / T):
            v[i] = b; load[a] -= q[i]; load[b] += q[i]; E += d
            if E < best - 1e-12: best, bv = E, v.copy()
    return bv


def sampled(circ, th, slots, rng):
    psi = circ.run(th); out = np.empty((SHOT_REPS, len(slots)))
    for f in 'ZXY':
        idx = [k for k, s in enumerate(slots) if s[0] == f]
        if not idx: continue
        p = np.abs(circ.rotate(psi, ps.VFAM[f]).reshape(-1)) ** 2; p = p / p.sum()
        signs = np.column_stack([circ.Zm[:, slots[k][1]] * circ.Zm[:, slots[k][2]] for k in idx])
        out[:, idx] = rng.multinomial(SHOTS, p, size=SHOT_REPS) @ signs / SHOTS
    return out


def job(args):
    name, arm, r = args
    S = setup(name); rng = np.random.default_rng([zlib.crc32(name.encode()), ARMS.index(arm), r, 2])
    t0 = time.time(); rec = dict(instance=name, arm=arm, restart=r, m=S['m'])
    if arm in ('Q', 'P'):
        slots = slots_random(S['m'])
        circ, ths = train_circuit(S, slots, arm == 'Q', rng)
        rec.update(parameters=circ.npar, cz=circ.ncz, by_tau={})
        for tau, th in ths.items():  # tau 键：base = 只做第一阶段，cont = 加延续法
            e = circ.expect(circ.run(th), slots)
            ideal = assess(S, decode(S, (e > 0).astype(int)))
            sh = [assess(S, decode(S, (o > 0).astype(int))) for o in sampled(circ, th, slots, rng)]
            rec['by_tau'][str(tau)] = dict(ideal=ideal, min_margin=float(np.abs(e).min()),
                shot_feasible_rate=float(np.mean([x['feasible'] for x in sh])),
                shot_same=float(np.mean([x['distance'] == ideal['distance'] and x['feasible'] == ideal['feasible'] for x in sh])),
                shot_mean_distance=float(np.mean([x['distance'] for x in sh if x['feasible']])) if any(x['feasible'] for x in sh) else None,
                theta=th.tolist())
    elif arm == 'LR':
        outs = train_lowrank(S, slots_random(S['m']), rng)
        rec.update(parameters=3 * NQ * RANK, by_tau={str(t): dict(ideal=assess(S, decode(S, (e > 0).astype(int))),
                   min_margin=float(np.abs(e).min())) for t, e in outs.items()})
    elif arm == 'C1':
        outs = train_c1(S, rng)
        rec.update(parameters=S['m'], by_tau={str(t): dict(ideal=assess(S, decode(S, (s > 0).astype(int)))) for t, s in outs.items()})
    elif arm == 'SA':
        rec.update(result=assess(S, anneal(S, rng)))
    else:
        rs = [assess(S, decode(S, rng.integers(0, 2, S['m']))) for _ in range(2000)]
        fe = [x for x in rs if x['feasible']]
        rec.update(feasible_rate=float(np.mean([x['feasible'] for x in rs])),
                   best_distance=min(x['distance'] for x in fe) if fe else None,
                   mean_distance=float(np.mean([x['distance'] for x in fe])) if fe else None)
    rec['seconds'] = round(time.time() - t0, 2)
    return rec


def run():
    OUT.mkdir(parents=True, exist_ok=False)
    (OUT / 'manifest.json').write_text(json.dumps(dict(nq=NQ, layers=LAYERS, alpha=ALPHA, pen=PEN, steps=STEPS,
        steps2=STEPS2, gammas=GAMMAS, restarts=RESTARTS, shots=SHOTS, shot_reps=SHOT_REPS, rank=RANK, arms=ARMS,
        instances=SP.INSTANCES, code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        note='constraint-embedded binary vehicle-index encoding; raw sign decoding; per-cluster routing only; simulator'), indent=2))
    jobs = [(n, a, r) for a in ('Q', 'P', 'LR', 'C1', 'SA', 'RND') for n in SP.INSTANCES
            for r in range(1 if a == 'RND' else RESTARTS)]
    from multiprocessing import Pool
    rows = []; t0 = time.time()
    with Pool(int(os.environ.get('WORKERS', '7'))) as pool:
        for rec in pool.imap_unordered(job, jobs):
            rows.append(rec); (OUT / 'runs.json').write_text(json.dumps(rows, ensure_ascii=False))
            print(len(rows), '/', len(jobs), rec['instance'], rec['arm'], rec['restart'], rec['seconds'], 's',
                  round(time.time() - t0), 's', flush=True)


if __name__ == '__main__':
    {'run': run}[sys.argv[1]]()
