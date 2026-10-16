"""泡利关联编码（PCE）+ 路线集合划分：10 比特、135 条候选路线的最小流水线，只测可行率。

问题：每条候选路线一个二元变量 x_r（选或不选）；要求每个客户恰好被一条选中路线覆盖，选中路线数不超过 K。
目标：E(x) = Σ c̃_r x_r + A·Σ_i (1 − Σ_{r∋i} x_r)² + B·(Σ_r x_r − K)²，c̃_r = c_r / max c。
编码：10 个比特的全部 3×C(10,2)=135 个两体关联 ⟨P_a P_b⟩（P∈{X,Y,Z}），x_r = [⟨Π_r⟩ > 0]。
线路：砖墙结构（RY、RZ + 相邻 CZ），6 层，120 个参数（少于 135 个变量）。
主要指标：原始正负号解码直接可行的比例（不做任何修复）；理想读出与 4096 次测量读出分别统计。
对照：
  Q   纠缠线路，随机变量槽映射
  Qc  纠缠线路，冲突感知映射（同一对比特的 XX/YY/ZZ 三个槽分给互相冲突的路线）
  P   同一线路去掉全部 CZ（乘积态）
  LR  参数量相同（120）的经典低秩模型：e_(f,a,b) = u_a^f·u_b^f / r
  C1  每个变量一个参数（135）的经典连续松弛
  SA  模拟退火直接在离散 QUBO 上搜索
  RND 随机方案（每条路线以 K/m 的概率选中）
  MIP 混合整数规划求池内最优（参照，不计入可行率比较）
运行：python pipeline.py run  然后  python pipeline.py report
"""
import os
for v in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[v] = '1'
import sys, json, time, hashlib, itertools, zlib
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
SUPPORT = HERE.parent / 'pce_cvrp_encoding_20261013' / 'support'
DATA = HERE.parent / 'pce_cvrp_encoding_20261013' / 'data'
sys.path.insert(0, str(SUPPORT))
import pce_scale as ps                      # 线路模拟、伴随梯度、Adam（与此前实验同一份代码）
from scale_pairs import route_cost          # 最近邻 + 2-opt + Or-opt 的单车路线长度

OUT = HERE / 'results'
INSTANCES = {'A-n32-k5': 784, 'B-n31-k5': 672, 'A-n33-k5': 661,
             'B-n34-k5': 788, 'P-n40-k5': 458, 'A-n45-k7': 1146}
NQ, LAYERS, M = 10, 6, 135
PEN_A, PEN_B = 2.0, 2.0
ALPHA = 1.5 * NQ
STEPS, STEPS2, TAU, W_M = 1500, 500, 0.1, 20.0
RESTARTS = int(os.environ.get('RESTARTS', '8'))
SHOTS, SHOT_REPS = 4096, 32
RANK = 4
ORDERS = 600
ARMS = ('Q', 'Qc', 'P', 'LR', 'C1', 'SA', 'RND')


# ---------------- 实例 ----------------
def load(name):
    text = (DATA / f'{name}.vrp').read_text().splitlines()
    sec, xy, q, cap = None, {}, {}, None
    for line in text:
        s = line.strip()
        if not s or s == 'EOF': continue
        if s.endswith('_SECTION'): sec = s; continue
        if sec == 'NODE_COORD_SECTION':
            i, x, y = s.split(); xy[int(i)] = (float(x), float(y))
        elif sec == 'DEMAND_SECTION':
            i, d = s.split(); q[int(i)] = int(d)
        elif sec == 'DEPOT_SECTION':
            pass
        elif ':' in s and s.split(':')[0].strip() == 'CAPACITY':
            cap = int(s.split(':')[1])
    ids = sorted(xy)
    P = np.array([xy[i] for i in ids]); dem = np.array([q[i] for i in ids], np.int64)
    D = np.floor(np.sqrt(((P[:, None] - P[None]) ** 2).sum(-1)) + .5)   # TSPLIB EUC_2D 取整
    K = int(name.split('-k')[1])
    return dict(name=name, P=P, q=dem, D=D, cap=cap, K=K, n=len(ids) - 1, bks=INSTANCES[name])


_rc = {}
def rcost(inst, S):
    key = (inst['name'], S)
    if key not in _rc:
        _rc[key] = float(route_cost(inst['D'], np.array(sorted(S), np.int64)))
    return _rc[key]


def split(inst, order):
    """按给定客户顺序切成不超过 K 段、每段不超载的路线，使段内按序路程之和最小（动态规划）。"""
    D, q, cap, K = inst['D'], inst['q'], inst['cap'], inst['K']
    n = len(order); INF = 1e18
    dp = np.full((K + 1, n + 1), INF); par = np.full((K + 1, n + 1), -1, np.int64); dp[0, 0] = 0
    seg = np.full((n, n), INF)
    for i in range(n):
        load, length, prev = 0, 0., 0
        for j in range(i, n):
            c = order[j]; load += q[c]
            if load > cap: break
            length += D[prev, c]; prev = c
            seg[i, j] = length + D[c, 0]
    for k in range(1, K + 1):
        for e in range(1, n + 1):
            for s in range(e):
                v = dp[k - 1, s] + seg[s, e - 1]
                if v < dp[k, e]: dp[k, e], par[k, e] = v, s
    k = int(np.argmin(dp[:, n]))
    if dp[k, n] >= INF: return None
    routes, e = [], n
    while k > 0:
        s = par[k, e]; routes.append(frozenset(int(c) for c in order[s:e])); e = s; k -= 1
    return routes


def build_pool(inst, rng):
    """随机扫描顺序 + 分段 + 单车路线精修，得到许多完整可行解；按总长从短到长取路线，凑满 135 条。"""
    P, n = inst['P'], inst['n']
    ang = np.arctan2(P[1:, 1] - P[0, 1], P[1:, 0] - P[0, 0])
    sols = {}
    for t in range(ORDERS):
        jitter = (0.0, 0.05, 0.15, 0.3)[t % 4]
        a = (ang - rng.uniform(0, 2 * np.pi) + rng.normal(0, jitter, n)) % (2 * np.pi)
        a = a if t % 2 else -a
        order = np.argsort(a) + 1
        routes = split(inst, order)
        if routes is None: continue
        key = frozenset(routes)
        if key not in sols:
            sols[key] = sum(rcost(inst, r) for r in routes)
    ranked = sorted(sols.items(), key=lambda kv: kv[1])
    pool, full = [], 0
    for sol, _ in ranked:
        new = [r for r in sol if r not in pool]
        if len(pool) + len(new) <= M:
            pool += new; full += 1
        else:
            pool += new[:M - len(pool)]
        if len(pool) == M: break
    assert len(pool) == M, (inst['name'], len(pool))
    return pool, dict(heuristic_solutions=len(sols), full_solutions_in_pool=full,
                      best_heuristic=float(ranked[0][1]))


def qubo(inst, pool):
    n, K = inst['n'], inst['K']
    A = np.zeros((n, M))
    for r, S in enumerate(pool):
        for c in S: A[c - 1, r] = 1
    c = np.array([rcost(inst, S) for S in pool]); cs = c / c.max()
    size = A.sum(0); ov = A.T @ A
    h = cs - PEN_A * size + PEN_B * (1 - 2 * K)
    W = PEN_A * ov + PEN_B; np.fill_diagonal(W, 0)
    const = PEN_A * n + PEN_B * K * K
    return dict(A=A, c=c, cs=cs, h=h, W=W, const=const, K=K)


def energy(Qb, x):
    return Qb['const'] + Qb['h'] @ x + x @ Qb['W'] @ x


def feasible(Qb, x):
    cover = Qb['A'] @ x
    return bool(np.all(cover == 1) and x.sum() <= Qb['K']), int(np.sum(cover != 1))


def pool_optimum(Qb):
    from scipy.optimize import milp, LinearConstraint, Bounds
    cons = [LinearConstraint(Qb['A'], 1, 1), LinearConstraint(np.ones((1, M)), 0, Qb['K'])]
    r = milp(Qb['c'], constraints=cons, integrality=np.ones(M), bounds=Bounds(0, 1))
    assert r.success
    x = np.round(r.x)
    return float(Qb['c'] @ x), x


def count_partitions(Qb, cap=10 ** 6):
    A, K = Qb['A'].astype(bool), Qb['K']
    cover_by = [np.flatnonzero(A[i]) for i in range(A.shape[0])]
    cnt = 0
    def rec(covered, used):
        nonlocal cnt
        if cnt >= cap: return
        if covered.all(): cnt += 1; return
        if used == K: return
        free = np.flatnonzero(~covered)
        i = min(free, key=lambda j: sum(1 for r in cover_by[j] if not (A[:, r] & covered).any()))
        for r in cover_by[i]:
            if not (A[:, r] & covered).any():
                rec(covered | A[:, r], used + 1)
    rec(np.zeros(A.shape[0], bool), 0)
    return cnt


# ---------------- 变量槽映射 ----------------
def slots_random():
    allslots = [(f, a, b) for f in 'ZXY' for a, b in itertools.combinations(range(NQ), 2)]
    perm = np.random.default_rng(20261014).permutation(M)
    return [allslots[i] for i in perm]


def slots_conflict(Qb):
    """把互相冲突（有共同客户）的路线三个一组，放到同一对比特的 XX/YY/ZZ 上：
    同一对比特三者不能同时取大的正值，而冲突路线本来就不该同时选中。"""
    ov = Qb['A'].T @ Qb['A']; np.fill_diagonal(ov, 0)
    left = set(range(M)); groups = []
    while left:
        r = max(left, key=lambda i: ov[i, list(left)].sum()); left.discard(r)
        s = max(left, key=lambda i: ov[r, i]); left.discard(s)
        t = max(left, key=lambda i: ov[r, i] + ov[s, i]); left.discard(t)
        groups.append((r, s, t))
    pairs = list(itertools.combinations(range(NQ), 2))
    slots = [None] * M
    for (a, b), g in zip(pairs, groups):
        for f, r in zip('XYZ', g): slots[r] = (f, a, b)
    return slots


# ---------------- 训练 ----------------
def relaxed(Qb, s):
    x = (1 + s) / 2
    E = energy(Qb, x)
    dEdx = Qb['h'] + 2 * Qb['W'] @ x
    return E, 0.5 * dEdx


def margin_term(e, tau):
    v = np.maximum(0, tau - np.abs(e))
    return W_M * (v ** 2).sum(), -2 * W_M * v * np.sign(e)


def train_circuit(Qb, slots, ent, rng):
    circ = ps.Circuit(NQ, LAYERS, ent)
    def make(tau):
        def fun(th):
            e = circ.expect(circ.run(th), slots); s = np.tanh(ALPHA * e)
            L, gs = relaxed(Qb, s); c = gs * ALPHA * (1 - s ** 2)
            if tau > 0:
                Lm, gm = margin_term(e, tau); L += Lm; c = c + gm
            return L, circ.grad(th, slots, c)
        return fun
    th0, _, it0 = ps.adam(make(0.0), rng.uniform(-np.pi, np.pi, circ.npar), steps=STEPS)
    th1, _, it1 = ps.adam(make(TAU), th0.copy(), steps=STEPS2)
    return circ, {0.0: th0, TAU: th1}, it0 + it1


def train_lowrank(Qb, slots, rng):
    F = {'X': 0, 'Y': 1, 'Z': 2}
    fi = np.array([F[s[0]] for s in slots]); ai = np.array([s[1] for s in slots]); bi = np.array([s[2] for s in slots])
    def unpack(u): return u.reshape(3, NQ, RANK)
    def outputs(u):
        U = unpack(u); return (U[fi, ai] * U[fi, bi]).sum(1) / RANK
    def make(tau):
        def fun(u):
            U = unpack(u); e = outputs(u); s = np.tanh(ALPHA * e)
            L, gs = relaxed(Qb, s); c = gs * ALPHA * (1 - s ** 2)
            if tau > 0:
                Lm, gm = margin_term(e, tau); L += Lm; c = c + gm
            g = np.zeros_like(U)
            np.add.at(g, (fi, ai), (c / RANK)[:, None] * U[fi, bi])
            np.add.at(g, (fi, bi), (c / RANK)[:, None] * U[fi, ai])
            return L, g.reshape(-1)
        return fun
    u0, _, _ = ps.adam(make(0.0), rng.normal(0, .5, 3 * NQ * RANK), steps=STEPS)
    u1, _, _ = ps.adam(make(TAU), u0.copy(), steps=STEPS2)
    return {0.0: outputs(u0), TAU: outputs(u1)}, 3 * NQ * RANK


def train_c1(Qb, rng):
    def make(tau):
        def fun(y):
            s = np.tanh(y); L, gs = relaxed(Qb, s); c = gs * (1 - s ** 2)
            if tau > 0:
                Lm, gm = margin_term(s, tau); L += Lm; c = c + gm * (1 - s ** 2)
            return L, c
        return fun
    y0, _, _ = ps.adam(make(0.0), rng.normal(0, 1, M), steps=STEPS)
    y1, _, _ = ps.adam(make(TAU), y0.copy(), steps=STEPS2)
    return {0.0: np.tanh(y0), TAU: np.tanh(y1)}


def anneal(Qb, rng, sweeps=2000, T0=2.0, T1=0.01):
    x = (rng.random(M) < Qb['K'] / M).astype(float)
    E = energy(Qb, x); best, bx = E, x.copy()
    h, W = Qb['h'], Qb['W']
    for t in range(sweeps * M):
        T = T0 * (T1 / T0) ** (t / (sweeps * M)); i = rng.integers(M)
        d = (1 - 2 * x[i]) * (h[i] + 2 * W[i] @ x)
        if d < 0 or rng.random() < np.exp(-d / T):
            x[i] = 1 - x[i]; E += d
            if E < best - 1e-12: best, bx = E, x.copy()
    return bx


def sampled(circ, th, slots, rng):
    psi = circ.run(th); out = np.empty((SHOT_REPS, M))
    for f in 'ZXY':
        idx = [k for k, s in enumerate(slots) if s[0] == f]
        if not idx: continue
        p = np.abs(circ.rotate(psi, ps.VFAM[f]).reshape(-1)) ** 2; p = p / p.sum()
        signs = np.column_stack([circ.Zm[:, slots[k][1]] * circ.Zm[:, slots[k][2]] for k in idx])
        out[:, idx] = rng.multinomial(SHOTS, p, size=SHOT_REPS) @ signs / SHOTS
    return out


def assess(Qb, opt, x):
    ok, bad = feasible(Qb, x)
    cost = float(Qb['c'] @ x)
    return dict(feasible=ok, violated_customers=bad, selected=int(x.sum()),
                cost=cost if ok else None, gap_percent=100 * (cost / opt - 1) if ok else None,
                energy=float(energy(Qb, x)))


# ---------------- 任务 ----------------
def prepare(name):
    inst = load(name); rng = np.random.default_rng(1014)
    pool, meta = build_pool(inst, rng)
    Qb = qubo(inst, pool)
    opt, xopt = pool_optimum(Qb)
    meta.update(name=name, n=inst['n'], K=inst['K'], cap=inst['cap'], bks=inst['bks'], pool_optimum=opt,
                pool_gap_to_bks_percent=100 * (opt / inst['bks'] - 1),
                feasible_partitions=count_partitions(Qb), routes=[sorted(r) for r in pool],
                route_costs=Qb['c'].tolist(), mean_route_size=float(Qb['A'].sum(0).mean()),
                optimum_energy=float(energy(Qb, xopt)), optimum_selection=np.flatnonzero(xopt).tolist(),
                min_energy_is_optimum=None)
    return inst, pool, Qb, meta


def job(args):
    name, arm, r = args
    inst, pool, Qb, meta = prepare(name)
    opt = meta['pool_optimum']
    rng = np.random.default_rng([zlib.crc32(name.encode()), ARMS.index(arm), r])
    t0 = time.time(); rec = dict(instance=name, arm=arm, restart=r)
    if arm in ('Q', 'Qc', 'P'):
        slots = slots_conflict(Qb) if arm == 'Qc' else slots_random()
        circ, ths, iters = train_circuit(Qb, slots, arm != 'P', rng)
        rec.update(parameters=circ.npar, cz=circ.ncz, iterations=iters, by_tau={})
        for tau, th in ths.items():
            e = circ.expect(circ.run(th), slots)
            ideal = assess(Qb, opt, (e > 0).astype(float))
            shots = [assess(Qb, opt, (o > 0).astype(float)) for o in sampled(circ, th, slots, rng)]
            rec['by_tau'][str(tau)] = dict(ideal=ideal, min_margin=float(np.abs(e).min()),
                median_margin=float(np.median(np.abs(e))),
                shot_feasible_rate=float(np.mean([s['feasible'] for s in shots])),
                shot_same_as_ideal=float(np.mean([s['energy'] == ideal['energy'] for s in shots])),
                shot_mean_violated=float(np.mean([s['violated_customers'] for s in shots])),
                theta=th.tolist())
    elif arm == 'LR':
        outs, npar = train_lowrank(Qb, slots_random(), rng)
        rec.update(parameters=npar, by_tau={str(t): dict(ideal=assess(Qb, opt, (e > 0).astype(float)),
                   min_margin=float(np.abs(e).min())) for t, e in outs.items()})
    elif arm == 'C1':
        outs = train_c1(Qb, rng)
        rec.update(parameters=M, by_tau={str(t): dict(ideal=assess(Qb, opt, (s > 0).astype(float))) for t, s in outs.items()})
    elif arm == 'SA':
        rec.update(result=assess(Qb, opt, anneal(Qb, rng)))
    else:
        rec.update(results=[assess(Qb, opt, (rng.random(M) < Qb['K'] / M).astype(float)) for _ in range(10000)])
        rs = rec.pop('results')
        rec.update(feasible_rate=float(np.mean([s['feasible'] for s in rs])),
                   mean_violated=float(np.mean([s['violated_customers'] for s in rs])))
    rec['seconds'] = round(time.time() - t0, 2)
    return rec


def run():
    OUT.mkdir(parents=True, exist_ok=False)
    metas = {}
    for name in INSTANCES:
        _, _, Qb, meta = prepare(name)
        metas[name] = meta
        print(json.dumps({k: meta[k] for k in ('name', 'n', 'K', 'pool_optimum', 'bks', 'pool_gap_to_bks_percent',
                                               'feasible_partitions', 'full_solutions_in_pool', 'mean_route_size')},
                         ensure_ascii=False), flush=True)
    (OUT / 'pools.json').write_text(json.dumps(metas, ensure_ascii=False))
    (OUT / 'manifest.json').write_text(json.dumps(dict(
        nq=NQ, layers=LAYERS, m=M, pen_a=PEN_A, pen_b=PEN_B, alpha=ALPHA, steps=STEPS, steps2=STEPS2, tau=TAU,
        restarts=RESTARTS, shots=SHOTS, shot_reps=SHOT_REPS, rank=RANK, arms=ARMS, instances=INSTANCES,
        code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        note='raw sign decoding only; no repair, no local search; simulator only'), indent=2))
    jobs = [(n, a, r) for n in INSTANCES for a in ARMS for r in range(1 if a == 'RND' else RESTARTS)]
    from multiprocessing import Pool
    rows = []; t0 = time.time()
    with Pool(int(os.environ.get('WORKERS', '7'))) as pool:
        for rec in pool.imap_unordered(job, jobs):
            rows.append(rec)
            (OUT / 'runs.json').write_text(json.dumps(rows, ensure_ascii=False))
            print(len(rows), '/', len(jobs), rec['instance'], rec['arm'], rec['restart'], rec['seconds'], 's',
                  round(time.time() - t0), 's', flush=True)


def report():
    pools = json.loads((OUT / 'pools.json').read_text(encoding='utf-8'))
    rows = json.loads((OUT / 'runs.json').read_text(encoding='utf-8'))
    L = []
    L.append('路线集合划分 + 泡利关联编码：10 比特、135 条候选路线（纯模拟，原始正负号解码，不做任何修复）')
    L.append('')
    L.append('候选路线池：')
    L.append(f"{'实例':10s}{'客户':>5s}{'车':>3s}{'池内最优':>9s}{'已知最优':>9s}{'池差距%':>8s}{'可行组合数':>10s}{'完整解':>7s}{'平均路线客户数':>12s}")
    for n, p in pools.items():
        L.append(f"{n:10s}{p['n']:5d}{p['K']:3d}{p['pool_optimum']:9.0f}{p['bks']:9d}{p['pool_gap_to_bks_percent']:8.2f}"
                 f"{p['feasible_partitions']:10d}{p['full_solutions_in_pool']:7d}{p['mean_route_size']:12.1f}")
    L.append('')
    names = list(pools)
    def tab(title, key):
        L.append(title)
        L.append(f"{'方法':6s}{'τ':>5s}" + ''.join(f'{n:>10s}' for n in names) + f"{'合计':>8s}")
        for arm in ('Q', 'Qc', 'P', 'LR', 'C1'):
            for tau in ('0.0', '0.1'):
                cells, tot, den = [], 0, 0
                for n in names:
                    rs = [r for r in rows if r['instance'] == n and r['arm'] == arm]
                    if key == 'shot' and arm not in ('Q', 'Qc', 'P'):
                        cells.append(f"{'—':>10s}"); continue
                    v = [r['by_tau'][tau]['ideal']['feasible'] if key == 'ideal' else r['by_tau'][tau]['shot_feasible_rate'] for r in rs]
                    tot += sum(v); den += len(v)
                    cells.append(f"{sum(v) / max(1, len(v)):10.2f}")
                L.append(f"{arm:6s}{tau:>5s}" + ''.join(cells) + (f"{tot / den:8.2f}" if den else f"{'—':>8s}"))
        if key == 'ideal':
            cells, tot, den = [], 0, 0
            for n in names:
                v = [r['result']['feasible'] for r in rows if r['instance'] == n and r['arm'] == 'SA']
                tot += sum(v); den += len(v); cells.append(f"{sum(v) / max(1, len(v)):10.2f}")
            L.append(f"{'SA':6s}{'':>5s}" + ''.join(cells) + f"{tot / den:8.2f}")
            v = [next(r for r in rows if r['instance'] == n and r['arm'] == 'RND')['feasible_rate'] for n in names]
            L.append(f"{'RND':6s}{'':>5s}" + ''.join(f'{x:10.4f}' for x in v) + f"{np.mean(v):8.4f}")
        L.append('')
    tab('一、理想读出（精确期望值）下直接可行的比例（每格 = 可行的随机起点数 / 起点数）：', 'ideal')
    tab('二、每个方向 4096 次测量读出、重复 32 次，直接可行的比例：', 'shot')
    L.append('三、可行时与池内最优的差距（%，可行的起点取平均；括号内为可行起点数）：')
    L.append(f"{'方法':6s}{'τ':>5s}" + ''.join(f'{n:>14s}' for n in names))
    for arm in ('Q', 'Qc', 'P', 'LR', 'C1', 'SA'):
        for tau in (('0.0', '0.1') if arm != 'SA' else ('',)):
            cells = []
            for n in names:
                rs = [r for r in rows if r['instance'] == n and r['arm'] == arm]
                res = [r['result'] if arm == 'SA' else r['by_tau'][tau]['ideal'] for r in rs]
                g = [x['gap_percent'] for x in res if x['feasible']]
                cells.append(f"{(np.mean(g) if g else float('nan')):9.2f}({len(g):2d})  ")
            L.append(f"{arm:6s}{tau:>5s}" + ''.join(cells))
    L.append('')
    L.append('四、不可行时平均违反约束的客户数（理想读出）：')
    for arm in ('Q', 'Qc', 'P', 'LR', 'C1'):
        v = [r['by_tau']['0.1']['ideal']['violated_customers'] for r in rows if r['arm'] == arm and not r['by_tau']['0.1']['ideal']['feasible']]
        L.append(f"  {arm:4s} τ=0.1：{np.mean(v) if v else 0:.2f}（{len(v)} 个不可行起点）")
    L.append('')
    L.append('五、量子线路训练后的关联值间隔（τ=0.1，中位数）：')
    for arm in ('Q', 'Qc', 'P'):
        v = [r['by_tau']['0.1']['min_margin'] for r in rows if r['arm'] == arm]
        L.append(f"  {arm:4s} 最小间隔中位数 {np.median(v):.3f}；测量读出与理想读出完全一致的比例 "
                 f"{np.mean([r['by_tau']['0.1']['shot_same_as_ideal'] for r in rows if r['arm'] == arm]):.2f}")
    txt = '\n'.join(L)
    (HERE / 'REPORT.txt').write_text(txt, encoding='utf-8')
    print(txt)


if __name__ == '__main__':
    {'run': run, 'report': report}[sys.argv[1]]()
