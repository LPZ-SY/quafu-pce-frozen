"""泡利关联编码：规模对比（探索性，未预登记）。

问题：规模越大，量子方法（泡利关联编码 + 纠缠线路）能否比经典方法得到更多、更短的方案？

实例：120 客户，需求 1–10，容量 60，仓库居中；扫描法分车 + L1（搬一个 + 换一个）收敛，作为起点。
      生成与路线计算沿用项目 scale_pairs.py（只把客户数改成 120）。
子问题：同一实例里取相邻的 3 / 6 / 12 辆车（嵌套）。每个客户一个变量：留在原车，或换到子问题里
       离它最近的另一辆车。比特数 8 / 10 / 12，每个变量对应两个比特在 Z / X / Y 某一方向上的关联值正负号。
目标（调参用）：近似路程（现解处的一阶 + 两两二阶差分，不含罚）+ 按松弛装载量精确计算的容量罚 + 正则项。
四种方法（每种 R 个随机起点）：
  Q  量子：砖墙线路（单比特 RY、RZ + 相邻 CZ），模拟器上用伴随法求梯度、Adam 调参；
  P  去掉纠缠门：同一线路删除全部 CZ，其余相同；
  C1 经典直接调变量：不用线路，每个变量一个参数，同一目标、同一优化器；
  C2 经典模拟退火：同一目标（离散），单变量翻转；
  C0 随机换车：每个变量以 1/2 概率换车（检验改进是否只是“打乱后再精修”带来的）。
读出后统一处理：超载修正 → 重新排路线 → 计算整个实例总路程 → 对子问题内的车做 L1 轻精修 → 再算一次。
"""
import os, sys, json, time, itertools
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scale_pairs import route_cost          # 项目原有的路线计算（最近邻 + 2-opt + Or-opt）
from numba import njit

N_CUST, CAP, FILL = 120, 60, 0.8
SIZES = {3: 8, 6: 10, 12: 12}               # 车辆数 → 比特数
R_RESTART = int(os.environ.get("R", "5"))
STEPS, LR = int(os.environ.get("STEPS", "1000")), 0.05
PEN_W, REG_W = 2.0, 0.1


# ---------------- 实例（沿用 scale_pairs.instance 的写法，客户数 120） ----------------
def cost_of(D, S):
    return route_cost(D, np.array(sorted(S), np.int64)) if S else 0.0


def l1(D, q, R, idx=None):
    """搬一个 + 换一个，跑到收敛。idx 给出时只在这些路线之间做。"""
    idx = list(range(len(R))) if idx is None else list(idx)
    C = {k: cost_of(D, R[k]) for k in idx}
    imp = True
    while imp:
        imp = False
        for a, b in itertools.permutations(idx, 2):
            base = C[a] + C[b]; la, lb = q[list(R[a])].sum(), q[list(R[b])].sum()
            for x in list(R[a]):
                if lb + q[x] <= CAP:
                    ca, cb = cost_of(D, R[a] - {x}), cost_of(D, R[b] | {x})
                    if ca + cb < base - 1e-9:
                        R[a].discard(x); R[b].add(x); C[a], C[b] = ca, cb; imp = True; break
                done = False
                for y in list(R[b]):
                    if la - q[x] + q[y] <= CAP and lb - q[y] + q[x] <= CAP:
                        ca, cb = cost_of(D, (R[a] - {x}) | {y}), cost_of(D, (R[b] - {y}) | {x})
                        if ca + cb < base - 1e-9:
                            R[a] = (R[a] - {x}) | {y}; R[b] = (R[b] - {y}) | {x}
                            C[a], C[b] = ca, cb; imp = done = True; break
                if done: break
            if imp: break
    return R


def instance(seed):
    rng = np.random.default_rng(seed)
    P = rng.uniform(0, 100, (N_CUST + 1, 2)); P[0] = 50
    q = np.r_[0, rng.integers(1, 11, N_CUST)].astype(np.int64)
    D = np.linalg.norm(P[:, None] - P[None], axis=-1)
    ang = np.arctan2(P[1:, 1] - 50, P[1:, 0] - 50); off = rng.uniform(0, 2 * np.pi)
    order = np.argsort((ang - off) % (2 * np.pi)) + 1
    R, cur, ld = [], [], 0
    for c in order:
        if ld + q[c] > FILL * CAP: R.append(cur); cur, ld = [], 0
        cur.append(int(c)); ld += q[c]
    R.append(cur)
    R = [set(r) for r in R]
    R = l1(D, q, R)
    return P, q, D, [r for r in R if r]


def total(D, R):
    return sum(cost_of(D, r) for r in R)


# ---------------- 子问题 ----------------
def pick_groups(P, R):
    cen = np.array([P[list(r)].mean(0) for r in R])
    k0 = int(np.argmin(np.linalg.norm(cen - cen.mean(0), axis=1)))     # 最靠中间的一辆车
    order = np.argsort(np.linalg.norm(cen - cen[k0], axis=1))
    return {K: [int(k) for k in order[:K]] for K in SIZES if K <= len(R)}


def build_sub(P, q, D, R, grp):
    cust, orig, targ = [], [], []
    for r in grp:
        for c in sorted(R[r]):
            others = [s for s in grp if s != r]
            dmin = [min(np.linalg.norm(P[c] - P[x]) for x in R[s]) for s in others]
            cust.append(c); orig.append(r); targ.append(others[int(np.argmin(dmin))])
    cust, orig, targ = np.array(cust), np.array(orig), np.array(targ)
    m = len(cust)

    def members(flip):
        mem = {r: set() for r in grp}
        for i in range(m): mem[targ[i] if flip[i] else orig[i]].add(int(cust[i]))
        return mem

    base_mem = members(np.zeros(m, int)); base_c = {r: cost_of(D, base_mem[r]) for r in grp}

    def delta(flips):                      # flips：要翻转的变量下标集合，只重算受影响的路线
        aff = set()
        for i in flips: aff |= {int(orig[i]), int(targ[i])}
        mem = {r: set(base_mem[r]) for r in aff}
        for i in flips: mem[int(orig[i])].discard(int(cust[i])); mem[int(targ[i])].add(int(cust[i]))
        return sum(cost_of(D, mem[r]) - base_c[r] for r in aff)

    lin = np.array([delta([i]) for i in range(m)])
    J = np.zeros((m, m))
    for i, j in itertools.combinations(range(m), 2):
        if {orig[i], targ[i]} & {orig[j], targ[j]}:
            J[i, j] = J[j, i] = delta([i, j]) - lin[i] - lin[j]
    sc = max(np.abs(lin).max(), np.abs(J).max(), 1e-9)
    ridx = {r: k for k, r in enumerate(grp)}
    O = np.array([ridx[r] for r in orig]); T = np.array([ridx[r] for r in targ])
    return dict(cust=cust, orig=orig, targ=targ, m=m, lin=lin / sc, J=J / sc, sc=sc,
                O=O, T=T, qv=q[cust].astype(float), nr=len(grp), members=members)


# ---------------- 目标（松弛形式，u∈[0,1] 表示换车程度） ----------------
def loss_u(sub, u):
    """返回 (目标值, d目标/du)。路程部分是二次近似；容量罚按松弛装载量精确计算。"""
    lin, J, qv, O, T, nr = sub["lin"], sub["J"], sub["qv"], sub["O"], sub["T"], sub["nr"]
    Lr = lin @ u + 0.5 * u @ J @ u
    g = lin + J @ u
    load = np.zeros(nr); np.add.at(load, O, qv * (1 - u)); np.add.at(load, T, qv * u)
    over = np.maximum(0, (load - CAP) / 5.5)
    Lp = PEN_W * (over ** 2).sum()
    dl = 2 * PEN_W * over / 5.5                      # d罚/d装载
    g = g + (-qv * dl[O] + qv * dl[T])
    return Lr + Lp, g


def loss_s(sub, s):                                   # s∈(-1,1)，u=(1-s)/2；加正则
    u = (1 - s) / 2
    L, gu = loss_u(sub, u)
    m = len(s); ms = (s ** 2).mean()
    L += REG_W * ms ** 2
    gs = -0.5 * gu + REG_W * 2 * ms * 2 * s / m
    return L, gs


# ---------------- 量子线路模拟（伴随法求梯度） ----------------
H = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
SDG = np.diag([1, -1j])
VFAM = {"Z": None, "X": H, "Y": H @ SDG}             # 测 Y：先 S†，再 H
PY = np.array([[0, -1j], [1j, 0]]); PZ = np.diag([1, -1]).astype(complex)


def ry(t): c, s = np.cos(t / 2), np.sin(t / 2); return np.array([[c, -s], [s, c]], complex)
def rz(t): return np.diag([np.exp(-0.5j * t), np.exp(0.5j * t)])


def apply1(psi, U, qb, n):
    psi = np.moveaxis(psi, qb, 0)
    psi = np.tensordot(U, psi, axes=(1, 0))
    return np.moveaxis(psi, 0, qb)


def cz_mask(n, pairs):
    bits = ((np.arange(1 << n)[:, None] >> (n - 1 - np.arange(n))) & 1)
    s = np.ones(1 << n)
    for a, b in pairs: s *= np.where(bits[:, a] & bits[:, b], -1, 1)
    return s.reshape((2,) * n)


class Circuit:
    def __init__(self, n, L, ent=True):
        self.n, self.L, self.ent = n, L, ent
        self.masks = [cz_mask(n, [(a, a + 1) for a in range(l % 2, n - 1, 2)]) for l in range(L)]
        self.ncz = sum(len(range(l % 2, n - 1, 2)) for l in range(L)) if ent else 0
        self.npar = 2 * n * L
        bits = ((np.arange(1 << n)[:, None] >> (n - 1 - np.arange(n))) & 1)
        self.Zm = (1 - 2 * bits).astype(float)       # 每个基态下各比特的 ±1

    def ops(self, th):
        th = th.reshape(self.L, self.n, 2)
        for l in range(self.L):
            for qb in range(self.n):
                yield ("ry", qb, th[l, qb, 0], (l, qb, 0))
                yield ("rz", qb, th[l, qb, 1], (l, qb, 1))
            if self.ent: yield ("cz", l, None, None)

    def run(self, th):
        n = self.n; psi = np.zeros((2,) * n, complex); psi[(0,) * n] = 1
        for kind, a, t, _ in self.ops(th):
            if kind == "ry": psi = apply1(psi, ry(t), a, n)
            elif kind == "rz": psi = apply1(psi, rz(t), a, n)
            else: psi = psi * self.masks[a]
        return psi

    def rotate(self, psi, V):
        if V is None: return psi
        for qb in range(self.n): psi = apply1(psi, V, qb, self.n)
        return psi

    def expect(self, psi, slots):
        """slots: [(族, a, b)]，返回各关联值。"""
        out = np.zeros(len(slots)); cache = {}
        for f in "ZXY":
            idx = [k for k, s in enumerate(slots) if s[0] == f]
            if not idx: continue
            p = np.abs(self.rotate(psi, VFAM[f]).reshape(-1)) ** 2
            C = self.Zm.T @ (p[:, None] * self.Zm)
            for k in idx: out[k] = C[slots[k][1], slots[k][2]]
        return out

    def O_apply(self, psi, slots, c):
        res = np.zeros_like(psi)
        for f in "ZXY":
            idx = [k for k, s in enumerate(slots) if s[0] == f]
            if not idx: continue
            Cm = np.zeros((self.n, self.n))
            for k in idx: a, b = slots[k][1], slots[k][2]; Cm[a, b] += c[k] / 2; Cm[b, a] += c[k] / 2
            Dg = ((self.Zm @ Cm) * self.Zm).sum(1).reshape((2,) * self.n)
            V = VFAM[f]
            phi = self.rotate(psi, V) * Dg
            if V is not None: phi = self.rotate(phi, V.conj().T)
            res += phi
        return res

    def grad(self, th, slots, c):
        """d(Σ c_k ⟨Π_k⟩)/dθ，伴随法。"""
        n = self.n; ops = list(self.ops(th))
        psi = self.run(th); lam = self.O_apply(psi, slots, c); phi = psi
        g = np.zeros((self.L, self.n, 2))
        for kind, a, t, key in reversed(ops):
            if kind == "cz":
                phi = phi * self.masks[a]; lam = lam * self.masks[a]; continue
            G = PY if kind == "ry" else PZ
            Gphi = apply1(phi, G, a, n)
            g[key] = np.imag(np.vdot(lam, Gphi))
            U = ry(t) if kind == "ry" else rz(t)
            Ud = U.conj().T
            phi = apply1(phi, Ud, a, n); lam = apply1(lam, Ud, a, n)
        return g.reshape(-1)


def make_slots(n, m):
    pairs = list(itertools.combinations(range(n), 2))
    slots = [(f, a, b) for f in "ZXY" for a, b in pairs]
    assert m <= len(slots), (n, m, len(slots))
    return slots[:m]


def adam(fun, x, steps=STEPS, lr=LR):
    mo = np.zeros_like(x); v = np.zeros_like(x); best = (np.inf, x.copy()); hist = []
    for t in range(1, steps + 1):
        L, g = fun(x)
        hist.append(L)
        if L < best[0]: best = (L, x.copy())
        mo = 0.9 * mo + 0.1 * g; v = 0.999 * v + 0.001 * g * g
        x = x - lr * (mo / (1 - 0.9 ** t)) / (np.sqrt(v / (1 - 0.999 ** t)) + 1e-8)
        if t > 100 and hist[-100] - L < 1e-6 * max(1, abs(L)): break
    return best[1], best[0], len(hist)


def run_quantum(sub, n, ent, rng):
    m = sub["m"]; L = max(4, int(np.ceil(m / n))); circ = Circuit(n, L, ent)
    slots = make_slots(n, m); alpha = 1.5 * n

    def fun(th):
        e = circ.expect(circ.run(th), slots)
        s = np.tanh(alpha * e)
        Lv, gs = loss_s(sub, s)
        c = gs * alpha * (1 - s ** 2)
        return Lv, circ.grad(th, slots, c)
    th, Lbest, it = adam(fun, rng.uniform(-np.pi, np.pi, circ.npar))
    e = circ.expect(circ.run(th), slots)
    return (e < 0).astype(int), dict(loss=float(Lbest), iters=it, ncz=circ.ncz, layers=L,
                                     npar=circ.npar, min_abs_corr=float(np.abs(e).min()))


def run_relax(sub, rng):
    m = sub["m"]

    def fun(y):
        s = np.tanh(y); Lv, gs = loss_s(sub, s)
        return Lv, gs * (1 - s ** 2)
    y, Lbest, it = adam(fun, rng.normal(0, 1, m))
    return (y < 0).astype(int), dict(loss=float(Lbest), iters=it)


@njit(cache=True)
def sa_core(lin, J, qv, O, T, nr, u, steps, T0, T1, seed, pen_w, cap):
    np.random.seed(seed); m = len(u)
    load = np.zeros(nr)
    for i in range(m):
        if u[i]: load[T[i]] += qv[i]
        else: load[O[i]] += qv[i]

    def pen(ld):
        s = 0.0
        for r in range(nr):
            o = max(0.0, (ld[r] - cap) / 5.5); s += o * o
        return pen_w * s
    best_u = u.copy(); cur = 0.0
    for i in range(m):
        cur += lin[i] * u[i]
        for j in range(m): cur += 0.5 * J[i, j] * u[i] * u[j]
    cur += pen(load); best = cur
    for k in range(steps):
        Tt = T0 * (T1 / T0) ** (k / steps); i = np.random.randint(m)
        d = 1 - 2 * u[i]
        dr = d * (lin[i] + (J[i] * u).sum() - J[i, i] * u[i])
        ld = load.copy()
        if u[i]: ld[T[i]] -= qv[i]; ld[O[i]] += qv[i]
        else: ld[O[i]] -= qv[i]; ld[T[i]] += qv[i]
        dd = dr + pen(ld) - pen(load)
        if dd < 0 or np.random.random() < np.exp(-dd / Tt):
            u[i] = 1 - u[i]; load = ld; cur += dd
            if cur < best: best = cur; best_u = u.copy()
    return best_u, best


def run_sa(sub, rng):
    m = sub["m"]; u0 = rng.integers(0, 2, m).astype(np.int64)
    u, b = sa_core(sub["lin"], sub["J"], sub["qv"], sub["O"].astype(np.int64), sub["T"].astype(np.int64),
                   sub["nr"], u0, 300 * m, 0.5, 0.002, int(rng.integers(1 << 30)), PEN_W, float(CAP))
    return u.astype(int), dict(loss=float(b))


# ---------------- 读出后的统一处理 ----------------
def evaluate(P, q, D, R, grp, sub, flip, start_total):
    mem = sub["members"](flip)
    over_routes = sum(1 for r in grp if q[list(mem[r])].sum() > CAP)
    over_amt = int(sum(max(0, q[list(mem[r])].sum() - CAP) for r in grp))
    # 超载修正：从超载车里移出“移走后增加最少”的客户，放到能装下的车里增加最少的位置
    guard = 0
    while guard < 200:
        bad = [r for r in grp if q[list(mem[r])].sum() > CAP]
        if not bad: break
        guard += 1; best = None
        for r in bad:
            cr = cost_of(D, mem[r])
            for x in mem[r]:
                rem = cost_of(D, mem[r] - {x}) - cr
                for s in grp:
                    if s == r or q[list(mem[s])].sum() + q[x] > CAP: continue
                    add = cost_of(D, mem[s] | {x}) - cost_of(D, mem[s])
                    if best is None or rem + add < best[0]: best = (rem + add, r, s, x)
        if best is None: return dict(feasible=False, over_routes=over_routes, over_amt=over_amt)
        _, r, s, x = best; mem[r].discard(x); mem[s].add(x)
    R2 = [set(r) for r in R]
    for r in grp: R2[r] = set(mem[r])
    before = total(D, R2)
    R2 = l1(D, q, R2, grp)
    after = total(D, R2)
    return dict(feasible=True, over_routes=over_routes, over_amt=over_amt, nflip=int(flip.sum()),
                before=float(before), after=float(after),
                impr_before=float((start_total - before) / start_total * 100),
                impr_after=float((start_total - after) / start_total * 100))


def job(seed):
    t0 = time.time()
    P, q, D, R = instance(seed)
    start = total(D, R); groups = pick_groups(P, R)
    out = dict(seed=seed, start=float(start), nroutes=len(R), sizes={})
    for K, grp in groups.items():
        n = SIZES[K]; sub = build_sub(P, q, D, R, grp)
        rng = np.random.default_rng(seed * 100 + K)
        res = {}
        for meth in ("Q", "P", "C1", "C2", "C0"):
            runs = []
            for r in range(R_RESTART):
                t1 = time.time()
                if meth == "Q": flip, info = run_quantum(sub, n, True, rng)
                elif meth == "P": flip, info = run_quantum(sub, n, False, rng)
                elif meth == "C1": flip, info = run_relax(sub, rng)
                elif meth == "C2": flip, info = run_sa(sub, rng)
                else: flip, info = rng.integers(0, 2, sub["m"]), {}
                ev = evaluate(P, q, D, R, grp, sub, flip, start)
                ev.update(info); ev["proxy_disc"] = float(loss_u(sub, flip.astype(float))[0]); ev["sec"] = round(time.time() - t1, 2)
                ev["flip"] = "".join(map(str, flip))
                runs.append(ev)
            res[meth] = runs
        out["sizes"][K] = dict(n=n, m=sub["m"], routes=grp, res=res)
    out["sec"] = round(time.time() - t0, 1)
    return out


if __name__ == "__main__":
    from multiprocessing import Pool
    s0, s1 = int(sys.argv[1]), int(sys.argv[2])
    nproc = int(sys.argv[3]) if len(sys.argv) > 3 else os.cpu_count()
    tag = sys.argv[4] if len(sys.argv) > 4 else "main"
    rows = []; t0 = time.time()
    with Pool(nproc) as pool:
        for r in pool.imap_unordered(job, range(s0, s1)):
            rows.append(r); json.dump(rows, open(f"pce_scale_{tag}.json", "w"))
            print(len(rows), "个实例已完成", round(time.time() - t0), "秒", flush=True)
    print("done", flush=True)
