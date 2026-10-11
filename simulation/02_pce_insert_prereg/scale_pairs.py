"""成对近邻子问题的规模扩展：8 / 12 / 16 比特（纯模拟）。
自包含重建（原 qrf/pair_study.py 不在磁盘上）：
  实例：60 客户，需求 1–10，容量 60，仓库居中；扫描法 + 路由 + L1（relocate+swap）收敛作为经典现解。
  子问题：取质心最近、且各有 ≥8 个客户的两条路线；贪心配出最紧的 m 对跨路线客户（a∈A, b∈B），
          第 k 对占比特 (2k, 2k+1)；x_i=1 表示把该客户换到另一辆车。m=4/6/8 → 8/12/16 比特，前 m 对嵌套一致。
  芯片：路径拓扑 0-1-2-…-(n-1)（对内边 + 对间连接边），Shenglian 上可原生嵌入；单层 QAOA 需 2(n-1) 个 CZ。
  代理（P_REPAIR 类）：在现解处做二阶差分，只取路径上的边；不可行用超载平方罚。
  真实目标：两条路线的路由路程（最近邻 + 2-opt + Or-opt 到收敛），超载记为不可行；全部 2^n 态精确枚举。
  三组：PRODUCT / WARM_NOZZ / WARM_QAOA，定义同 study31；参数按代理能量选。
  指标：真实路程前 5% 的富集；噪声 = 全局去极化（整体保真度 F=f^{#CZ}）+ 逐比特读出。
"""
import sys, json, time, itertools
import numpy as np
from numba import njit
from multiprocessing import Pool

GG = np.linspace(-np.pi, np.pi, 61); BB = np.linspace(0, np.pi, 31)
GRID = np.array([(g, b) for g in GG for b in BB])
EPS = (0.0, 0.05, 0.15, 0.25)
RO0, RO1 = 0.031, 0.037          # Shenglian v4s 8 比特读出误差均值（0→1, 1→0）
import os
N_CUST, CAP, PEN, FILL = 60, 60, 5.0, 0.8
LS = os.environ.get('LS', '1') == '1'


# ---------------- 路由（numba） ----------------
@njit(cache=True)
def rlen(D, r, n):
    if n == 0: return 0.0
    s = D[0, r[0]] + D[r[n - 1], 0]
    for i in range(n - 1): s += D[r[i], r[i + 1]]
    return s


@njit(cache=True)
def route_cost(D, nodes):
    n = len(nodes)
    if n == 0: return 0.0
    r = np.empty(n, np.int64); used = np.zeros(n, np.bool_); cur = 0
    for k in range(n):                               # 最近邻
        bj, bd = -1, 1e18
        for j in range(n):
            if not used[j] and D[cur, nodes[j]] < bd: bd, bj = D[cur, nodes[j]], j
        used[bj] = True; r[k] = nodes[bj]; cur = nodes[bj]
    best = rlen(D, r, n); tmp = np.empty(n, np.int64)
    imp = True
    while imp:
        imp = False
        for i in range(n - 1):                       # 2-opt
            for j in range(i + 1, n):
                for k in range(n): tmp[k] = r[k]
                for k in range(j - i + 1): tmp[i + k] = r[j - k]
                v = rlen(D, tmp, n)
                if v < best - 1e-9:
                    best = v; r[:] = tmp; imp = True
        for L in range(1, 4):                        # Or-opt
            if L >= n: break
            for i in range(n - L + 1):
                for kk in range(n - L + 1):
                    if kk == i: continue
                    for rev in range(2):
                        # 取出 r[i:i+L]，插到剩余序列的位置 kk
                        rest = np.empty(n - L, np.int64); t = 0
                        for q in range(n):
                            if q < i or q >= i + L: rest[t] = r[q]; t += 1
                        t = 0
                        for q in range(kk): tmp[t] = rest[q]; t += 1
                        for q in range(L): tmp[t] = r[i + (L - 1 - q if rev else q)]; t += 1
                        for q in range(kk, n - L): tmp[t] = rest[q]; t += 1
                        v = rlen(D, tmp, n)
                        if v < best - 1e-9:
                            best = v; r[:] = tmp; imp = True
    return best


@njit(cache=True)
def enumerate_true(D, q, cap, A0, B0, mov_inA, mov, nbits):
    """mov: 可动客户；mov_inA[i]=1 表示现解中在 A。返回 2^n 个真实路程（不可行 inf）。"""
    N = 1 << nbits; out = np.empty(N, np.float64)
    for m in range(N):
        la = 0; lb = 0
        A = np.empty(len(A0) + nbits, np.int64); B = np.empty(len(B0) + nbits, np.int64); na = 0; nb = 0
        for c in A0: A[na] = c; na += 1; la += q[c]
        for c in B0: B[nb] = c; nb += 1; lb += q[c]
        for i in range(nbits):
            inA = mov_inA[i] ^ ((m >> i) & 1)
            if inA: A[na] = mov[i]; na += 1; la += q[mov[i]]
            else: B[nb] = mov[i]; nb += 1; lb += q[mov[i]]
        if la > cap or lb > cap: out[m] = np.inf
        else: out[m] = route_cost(D, A[:na]) + route_cost(D, B[:nb])
    return out


# ---------------- 实例与经典现解 ----------------
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
    cost = lambda S: route_cost(D, np.array(sorted(S), np.int64))
    R = [set(r) for r in R]; imp = LS
    while imp:                                        # L1：relocate + swap 到收敛
        imp = False
        for a, b in itertools.permutations(range(len(R)), 2):
            base = cost(R[a]) + cost(R[b]); la, lb = q[list(R[a])].sum(), q[list(R[b])].sum()
            for x in list(R[a]):
                if lb + q[x] <= CAP and cost(R[a] - {x}) + cost(R[b] | {x}) < base - 1e-9:
                    R[a].discard(x); R[b].add(x); imp = True; break
                done = False
                for y in list(R[b]):
                    if la - q[x] + q[y] <= CAP and lb - q[y] + q[x] <= CAP and \
                       cost((R[a] - {x}) | {y}) + cost((R[b] - {y}) | {x}) < base - 1e-9:
                        R[a] = (R[a] - {x}) | {y}; R[b] = (R[b] - {y}) | {x}; imp = done = True; break
                if done: break
            if imp: break
        R = [r for r in R if r]
    return P, q, D, R


def pick_pairs(P, R, mmax=8):
    cen = [P[list(r)].mean(0) for r in R]
    cand = [(np.linalg.norm(cen[a] - cen[b]), a, b) for a, b in itertools.combinations(range(len(R)), 2)
            if len(R[a]) >= mmax and len(R[b]) >= mmax]
    if not cand: return None
    _, a, b = min(cand); A, B = sorted(R[a]), sorted(R[b])
    D2 = np.linalg.norm(P[A][:, None] - P[B][None], axis=-1); pairs = []
    while len(pairs) < mmax:
        i, j = np.unravel_index(np.argmin(D2), D2.shape); pairs.append((A[i], B[j]))
        D2[i, :] = np.inf; D2[:, j] = np.inf
    mid = np.array([(P[x] + P[y]) / 2 for x, y in pairs])
    return a, b, pairs, mid


# ---------------- 代理与线路 ----------------
def bits_table(n): return ((np.arange(1 << n)[:, None] >> np.arange(n)) & 1).astype(np.int8)


def build_proxy(D, q, A0, B0, mov, inA, n):
    edges = [(i, i + 1) for i in range(n - 1)]
    def f(flip):
        A = list(A0); B = list(B0)
        for i in range(n): ((A if inA[i] ^ flip[i] else B)).append(mov[i])
        over = max(0, q[A].sum() - CAP) ** 2 + max(0, q[B].sum() - CAP) ** 2
        return route_cost(D, np.array(A, np.int64)) + route_cost(D, np.array(B, np.int64)) + PEN * over
    z = np.zeros(n, int); f0 = f(z); fi = []
    for i in range(n):
        e = z.copy(); e[i] = 1; fi.append(f(e))
    lin = np.array(fi) - f0; J = {}
    for i, j in edges:
        e = z.copy(); e[i] = e[j] = 1; J[(i, j)] = f(e) - fi[i] - fi[j] + f0
    sc = max(np.abs(lin).max(), max(abs(v) for v in J.values()), 1e-9)
    return lin / sc, {k: v / sc for k, v in J.items()}


def ising_lambda(lin, J):
    h = -lin / 2; Jz = []
    for (a, b), c in J.items(): h[a] -= c / 4; h[b] -= c / 4; Jz.append(c / 4)
    return float(np.abs(Jz).sum() / (np.abs(Jz).sum() + np.abs(h).sum()))


def fwht(a, n):
    N = 1 << n; h = 1
    while h < N:
        a = a.reshape(a.shape[:-1] + (N // (2 * h), 2, h)); x, y = a[..., 0, :].copy(), a[..., 1, :].copy()
        a[..., 0, :], a[..., 1, :] = x + y, x - y; a = a.reshape(a.shape[:-3] + (N,)); h *= 2
    return a / np.sqrt(N)


def best_std(Ec, E, n, XEIG):
    best = None
    for s in range(0, len(GRID), 64):
        g = GRID[s:s + 64]
        psi = np.exp(-1j * g[:, :1] * Ec[None, :]).astype(np.complex64) / np.sqrt(1 << n)
        psi = fwht(fwht(psi, n) * np.exp(-1j * g[:, 1:2] * XEIG[None, :]), n)
        pr = np.abs(psi) ** 2; v = pr @ E; k = int(np.argmin(v))
        if best is None or v[k] < best[0]: best = (v[k], pr[k] / pr[k].sum())
    return best[1]


def product_state(c, BITS):
    a = np.ones(len(BITS))
    for i in range(BITS.shape[1]): a *= np.where(BITS[:, i] == 1, np.sqrt(c[i]), np.sqrt(1 - c[i]))
    return a.astype(np.complex64)


def best_warm(Ec, E, c, n, BITS):
    th = 2 * np.arcsin(np.sqrt(c)); p0 = product_state(c, BITS); best = None
    Ry = lambda s: np.array([[np.cos(s / 2), -np.sin(s / 2)], [np.sin(s / 2), np.cos(s / 2)]])
    for s in range(0, len(GRID), 64):
        g = GRID[s:s + 64]; G = len(g)
        x = (p0[None, :] * np.exp(-1j * g[:, :1] * Ec[None, :])).astype(np.complex64).reshape((G,) + (2,) * n)
        ph = np.stack([np.exp(-1j * g[:, 1]), np.exp(1j * g[:, 1])], 1)
        for i in range(n):
            ax = n - i; U = np.einsum("ij,bj,jk->bik", Ry(th[i]), ph, Ry(-th[i])).astype(np.complex64)
            x = np.moveaxis(x, ax, 1); x = np.einsum("bij,bj...->bi...", U, x); x = np.moveaxis(x, 1, ax)
        pr = np.abs(x.reshape(G, -1)) ** 2; v = pr @ E; k = int(np.argmin(v))
        if best is None or v[k] < best[0]: best = (v[k], pr[k] / pr[k].sum())
    return best[1]


def readout(pr, n):
    x = pr.reshape((2,) * n); M = np.array([[1 - RO0, RO1], [RO0, 1 - RO1]])
    for i in range(n):
        ax = n - 1 - i
        x = np.moveaxis(np.tensordot(M, np.moveaxis(x, ax, 0), axes=(1, 0)), 0, ax)
    return x.reshape(-1)


def job(args):
    seed, ns = args
    P, q, D, R = instance(seed)
    pk = pick_pairs(P, R)
    if pk is None: return None
    a, b, pairs, mid = pk
    out = dict(seed=seed)
    for n in ns:
        m = n // 2; prs = pairs[:m]
        # 对沿交界排序（按中点相对仓库的角度），使相邻对在路径上相邻
        ang = np.arctan2(mid[:m, 1] - 50, mid[:m, 0] - 50); ang = (ang - ang.mean() + np.pi) % (2 * np.pi)
        prs = [prs[k] for k in np.argsort(ang)]
        mov = np.array([c for pr in prs for c in pr], np.int64); inA = np.array([1, 0] * m, np.int64)
        A0 = np.array(sorted(R[a] - set(mov)), np.int64); B0 = np.array(sorted(R[b] - set(mov)), np.int64)
        t0 = time.time()
        T = enumerate_true(D, q, CAP, A0, B0, inA, mov, n)
        lin, J = build_proxy(D, q, A0, B0, mov, inA, n)
        BITS = bits_table(n)
        E = BITS @ lin + sum(c * BITS[:, i] * BITS[:, j] for (i, j), c in J.items())
        E_lin = BITS @ lin + sum(c * (BITS[:, i] + BITS[:, j]) / 2 for (i, j), c in J.items())
        XEIG = (n - 2 * BITS.sum(1)).astype(float)
        fin = np.isfinite(T); ngood = int(np.ceil(0.05 * (1 << n)))
        good = np.argsort(np.where(fin, T, np.inf), kind="stable")[:ngood]
        from scipy.stats import spearmanr
        o = dict(lam=ising_lambda(lin.copy(), J), rho=float(spearmanr(E[fin], T[fin])[0]),
                 room=float((T[0] - T[fin].min()) / T[0]), n_feas=int(fin.sum()))
        pr_prod = best_std(E_lin, E, n, XEIG)
        c = np.array([pr_prod[BITS[:, i] == 1].sum() for i in range(n)])
        arms = {"PRODUCT": pr_prod}
        for arm, Ec in (("WARM_NOZZ", E_lin), ("WARM_QAOA", E)):
            best = None
            for eps in EPS:
                pr = best_warm(Ec, E, np.clip(c, eps, 1 - eps), n, BITS)
                if best is None or pr @ E < best[0] - 1e-12: best = (pr @ E, pr)
            arms[arm] = best[1]
        uni = readout(np.full(1 << n, 1 / (1 << n)), n)
        for arm, pr in arms.items():
            o[arm] = dict(ideal=float(pr[good].sum() * (1 << n) / ngood),
                          ro=float(readout(pr, n)[good].sum() * (1 << n) / ngood))
        o["uni_ro"] = float(uni[good].sum() * (1 << n) / ngood)
        o["sec"] = round(time.time() - t0, 1)
        out[n] = o
    return out


if __name__ == "__main__":
    ns = [int(x) for x in sys.argv[1].split(",")]
    seeds = range(int(sys.argv[2]), int(sys.argv[3]))
    tag = sys.argv[4]
    rows = []; t0 = time.time()
    with Pool(2) as pool:
        for r in pool.imap_unordered(job, [(s, ns) for s in seeds]):
            if r: rows.append(r); json.dump(rows, open(f"scale_{tag}.json", "w"), default=float)
            print(len(rows), round(time.time() - t0), r and {n: r[n]["sec"] for n in ns}, flush=True)
    print("done", round(time.time() - t0))
