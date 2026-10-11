"""独立检查 pce_scale.py 里的泡利关联编码实现。
1. 线路与关联值：与一套独立写的稠密矩阵模拟器逐项对比；
2. 梯度：伴随法 vs 有限差分；
3. 表达能力：训练线路去复现随机给定的 0/1 方案（纠缠线路 vs 去掉 CZ）；
4. 复现论文：最大割，论文的损失函数（tanh + 正则项）、α=1.5n、一轮单比特翻转搜索，与精确最优比较。"""
import sys, os, itertools, json, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pce_scale as S
from numba import njit

rng = np.random.default_rng(0)
I2 = np.eye(2); X = np.array([[0, 1], [1, 0]]); Y = np.array([[0, -1j], [1j, 0]]); Z = np.diag([1, -1])
PAULI = {"X": X, "Y": Y, "Z": Z}


def kron_all(ms):
    out = np.array([[1.0 + 0j]])
    for m in ms: out = np.kron(out, m)
    return out


def dense_state(n, L, th, ent):
    """独立实现：逐门构造 2^n×2^n 矩阵。比特 0 为最高位（与 pce_scale 一致）。"""
    th = th.reshape(L, n, 2); psi = np.zeros(1 << n, complex); psi[0] = 1
    CZ2 = np.diag([1, 1, 1, -1])
    for l in range(L):
        for q in range(n):
            a = th[l, q, 0]; RY = np.array([[np.cos(a / 2), -np.sin(a / 2)], [np.sin(a / 2), np.cos(a / 2)]])
            psi = kron_all([RY if k == q else I2 for k in range(n)]) @ psi
            b = th[l, q, 1]; RZ = np.diag([np.exp(-0.5j * b), np.exp(0.5j * b)])
            psi = kron_all([RZ if k == q else I2 for k in range(n)]) @ psi
        if ent:
            for a in range(l % 2, n - 1, 2):
                ms = [I2] * n; ms[a] = None
                ops = []; k = 0
                while k < n:
                    if k == a: ops.append(CZ2); k += 2
                    else: ops.append(I2); k += 1
                psi = kron_all(ops) @ psi
    return psi


def dense_expect(psi, n, slots):
    out = []
    for f, a, b in slots:
        P = kron_all([PAULI[f] if k in (a, b) else I2 for k in range(n)])
        out.append(np.real(np.vdot(psi, P @ psi)))
    return np.array(out)


def check_sim():
    worst = 0
    for n, L, ent in [(5, 3, True), (6, 4, True), (6, 4, False), (7, 3, True)]:
        circ = S.Circuit(n, L, ent); slots = S.make_slots(n, 3 * n * (n - 1) // 2)
        for _ in range(3):
            th = rng.uniform(-np.pi, np.pi, circ.npar)
            e1 = circ.expect(circ.run(th), slots); e2 = dense_expect(dense_state(n, L, th, ent), n, slots)
            worst = max(worst, np.abs(e1 - e2).max())
    return worst


def check_grad():
    n, L = 6, 4; circ = S.Circuit(n, L, True); slots = S.make_slots(n, 40)
    th = rng.uniform(-np.pi, np.pi, circ.npar); c = rng.normal(size=40)
    g = circ.grad(th, slots, c); fd = np.zeros_like(th); h = 1e-6
    for i in range(len(th)):
        tp = th.copy(); tp[i] += h; tm = th.copy(); tm[i] -= h
        fd[i] = (c @ circ.expect(circ.run(tp), slots) - c @ circ.expect(circ.run(tm), slots)) / (2 * h)
    return np.abs(g - fd).max()


def fit_target(n, m, ent, target, steps=1500):
    """训练线路使 sgn⟨Π_i⟩ = target_i。返回复现正确的比例。"""
    L = max(4, int(np.ceil(m / n))); circ = S.Circuit(n, L, ent); slots = S.make_slots(n, m); al = 1.5 * n

    def fun(th):
        e = circ.expect(circ.run(th), slots); t = np.tanh(al * e)
        return -(target * t).sum(), circ.grad(th, slots, -target * al * (1 - t ** 2))
    th, _, _ = S.adam(fun, rng.uniform(-np.pi, np.pi, circ.npar), steps=steps)
    e = circ.expect(circ.run(th), slots)
    return float((np.sign(e) == target).mean())


# ---------------- 最大割 ----------------
def rand_graph(m, deg, seed):
    r = np.random.default_rng(seed)
    while True:
        E = set()
        while len(E) < m * deg // 2:
            a, b = r.choice(m, 2, replace=False); E.add((min(a, b), max(a, b)))
        E = np.array(sorted(E)); W = np.zeros((m, m)); W[E[:, 0], E[:, 1]] = W[E[:, 1], E[:, 0]] = 1
        return E, W


@njit(cache=True)
def exact_maxcut(Wb, m):
    best = 0
    for x in range(1 << (m - 1)):          # 固定最后一个点在 0 一侧
        c = 0
        for i in range(m):
            for j in range(i + 1, m):
                if Wb[i, j] and (((x >> i) & 1) != ((x >> j) & 1)): c += 1
        if c > best: best = c
    return best


def cut(W, x):
    s = 1 - 2 * x
    return 0.25 * (W * (1 - np.outer(s, s))).sum()


def bitswap(W, x):
    x = x.copy(); v = cut(W, x)
    for i in range(len(x)):
        x[i] ^= 1; v2 = cut(W, x)
        if v2 > v: v = v2
        else: x[i] ^= 1
    return x, v


def pce_maxcut(W, E, n, ent, seed, L=None):
    m = len(W); L = L or max(4, int(np.ceil(m / n))); circ = S.Circuit(n, L, ent); slots = S.make_slots(n, m)
    al = 1.5 * n; beta = 0.5; nu = len(E) / 2 + (m - 1) / 4
    r = np.random.default_rng(seed)

    def fun(th):
        e = circ.expect(circ.run(th), slots); t = np.tanh(al * e)
        # 论文式(2)：最小化 Σ_{(i,j)∈E} W_ij t_i t_j + β ν [mean t²]²
        Lc = sum(t[i] * t[j] for i, j in E); g = np.zeros(m)
        for i, j in E: g[i] += t[j]; g[j] += t[i]
        ms = (t ** 2).mean(); Lr = beta * nu * ms ** 2
        g = g + beta * nu * 2 * ms * 2 * t / m
        return Lc + Lr, circ.grad(th, slots, g * al * (1 - t ** 2))
    th, _, _ = S.adam(fun, r.uniform(-np.pi, np.pi, circ.npar), steps=2000)
    e = circ.expect(circ.run(th), slots); x = (e < 0).astype(int)
    v0 = cut(W, x); _, v1 = bitswap(W, x)
    return v0, v1, float(np.abs(e).min()), circ.ncz


if __name__ == "__main__":
    out = {}
    out["sim_max_abs_err"] = float(check_sim()); print("1. 线路关联值 最大误差", out["sim_max_abs_err"], flush=True)
    out["grad_max_abs_err"] = float(check_grad()); print("2. 梯度 最大误差", out["grad_max_abs_err"], flush=True)
    res = {}
    for n, m in [(6, 30), (8, 60), (8, 84)]:
        for ent in (True, False):
            acc = [fit_target(n, m, ent, rng.choice([-1, 1], m)) for _ in range(5)]
            res[f"n{n}_m{m}_{'纠缠' if ent else '无纠缠'}"] = acc
            print(f"3. 复现随机方案 n={n} m={m} {'纠缠' if ent else '无纠缠'}：正确比例 {np.round(acc, 3)}", flush=True)
    out["expressivity"] = res
    mc = []
    for seed in range(8):
        m = 26; E, W = rand_graph(m, 4, seed); opt = exact_maxcut(W.astype(np.int64), m)
        rnd = [bitswap(W, np.random.default_rng(seed * 10 + k).integers(0, 2, m))[1] for k in range(50)]
        row = dict(seed=seed, opt=int(opt), naive_mean=float(np.mean(rnd)), naive_std=float(np.std(rnd)))
        for ent in (True, False):
            r = [pce_maxcut(W, E, 8, ent, seed * 100 + k) for k in range(3)]
            row["Q" if ent else "P"] = dict(r_before=[v[0] / opt for v in r], r_after=[v[1] / opt for v in r],
                                           min_gap=[v[2] for v in r], ncz=r[0][3])
        mc.append(row)
        print(f"4. 最大割 m=26 seed={seed} 最优={opt}  随机+翻转搜索={row['naive_mean']/opt:.3f}±{row['naive_std']/opt:.3f}  "
              f"纠缠 前/后={np.round(row['Q']['r_before'],3)}/{np.round(row['Q']['r_after'],3)}  "
              f"无纠缠 后={np.round(row['P']['r_after'],3)}", flush=True)
    out["maxcut"] = mc
    json.dump(out, open("verify_pce.json", "w"), ensure_ascii=False, indent=1)
