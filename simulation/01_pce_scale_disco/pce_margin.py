"""泡利关联编码：间隔下限训练（已预登记，见项目文档《泡利关联编码_间隔下限训练_预登记》）。

要回答的问题：
  训练时要求每个变量的关联期望值绝对值（间隔）不小于 τ，能不能让真机噪声下读出的方案
  和无噪声时一致，同时方案质量损失不大？比特数多给 2 个，会不会更容易做到？

实例、子问题、线路、训练目标、优化器全部沿用 pce_scale.py。只改两处：
  1. 第二阶段训练：从第一阶段的最好参数出发，目标加一项  W_M · Σ max(0, τ − |⟨Π_i⟩|)²，再训练 STEPS2 步。
  2. 读出评估：除无噪声外，按“整体去极化 + 不对称读出误差 + 有限测量次数”的噪声模型，重复 NREP 次读出。

噪声模型（与 scale_pairs.py 一致的写法）：
  整体保真度 F = f_CZ^(两比特门数)，所有泡利期望值乘 F；
  读出：0 读成 1 的概率 RO0，1 读成 0 的概率 RO1；
  有限测量次数：每种测量设置 SHOTS 次，期望值加上标准差 sqrt((1−E²)/SHOTS) 的正态误差；
  读出误差校准：用已知的 RO0、RO1 反解（另报告不校准的结果）。

方法：Q（纠缠线路）与 P（去掉纠缠门，两比特门数为 0，所以只受读出误差和测量次数影响）；
      C2（经典模拟退火，同一离散目标）只用来给出“同一目标下经典能达到的值”，作为质量的上端参照。

运行（在 disco 上，与 pce_scale.py 放在同一文件夹）：
  python3 pce_margin.py 40001 40011 $(nproc) main
  python3 analyze_margin.py pce_margin_main.json > pce_margin_main_analysis.txt
可调环境变量：R（随机起点，默认 3）、STEPS（第一阶段，默认 1000）、STEPS2（第二阶段，默认 500）、
              TAUS（默认 "0.05,0.1,0.15,0.2"）、W_M（默认 20）、NREP（默认 50）、SHOTS（默认 4096）、
              SIZES（默认 "3,6,12"）、EXTRA（多给的比特数，默认 2；12 车不加）。
"""
import os, sys, json, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pce_scale as ps

R_RESTART = int(os.environ.get("R", "3"))
STEPS2 = int(os.environ.get("STEPS2", "500"))
TAUS = [float(x) for x in os.environ.get("TAUS", "0.05,0.1,0.15,0.2").split(",")]
W_M = float(os.environ.get("W_M", "20"))
NREP = int(os.environ.get("NREP", "50"))
SHOTS = int(os.environ.get("SHOTS", "4096"))
SIZES = [int(x) for x in os.environ.get("SIZES", "3,6,12").split(",")]
EXTRA = int(os.environ.get("EXTRA", "2"))
FCZ = (0.977, 0.99)                         # 每个两比特门的保真度：此前模拟采用值，以及较乐观值
RO0, RO1 = 0.031, 0.037                     # 与 scale_pairs.py 相同（Shenglian 读出误差均值）

# 路线路程缓存（结果与原函数相同，只是不重复计算）
_raw = ps.cost_of; _cache = {}
def _cc(D, S):
    k = frozenset(S); v = _cache.get(k)
    if v is None: v = _raw(D, S); _cache[k] = v
    return v
ps.cost_of = _cc


def full_expect(circ, psi, slots):
    """返回各变量的关联期望值，以及各族每个比特的单比特期望值（读出偏差要用）。"""
    corr = np.zeros(len(slots)); single = {}
    for f in "ZXY":
        p = np.abs(circ.rotate(psi, ps.VFAM[f]).reshape(-1)) ** 2
        single[f] = circ.Zm.T @ p
        C = circ.Zm.T @ (p[:, None] * circ.Zm)
        for k, s in enumerate(slots):
            if s[0] == f: corr[k] = C[s[1], s[2]]
    return corr, single


def noisy_readouts(corr, single, slots, ncz, fcz, rng, mitigate):
    """按噪声模型重复读出 NREP 次，返回 (NREP, m) 的期望值估计。"""
    F = fcz ** ncz
    a, b = RO1 - RO0, 1 - RO0 - RO1          # 读出后 E[z'] = a + b·z
    m = len(slots); out = np.zeros((NREP, m))
    sa = np.array([single[s[0]][s[1]] for s in slots]) * F
    sb = np.array([single[s[0]][s[2]] for s in slots]) * F
    E = a * a + a * b * (sa + sb) + b * b * corr * F
    Ea, Eb = a + b * sa, a + b * sb
    for r in range(NREP):
        e = E + rng.normal(0, 1, m) * np.sqrt(np.clip(1 - E ** 2, 0, 1) / SHOTS)
        if mitigate:
            ea = Ea + rng.normal(0, 1, m) * np.sqrt(np.clip(1 - Ea ** 2, 0, 1) / SHOTS)
            eb = Eb + rng.normal(0, 1, m) * np.sqrt(np.clip(1 - Eb ** 2, 0, 1) / SHOTS)
            za, zb = (ea - a) / b, (eb - a) / b
            e = (e - a * a - a * b * (za + zb)) / (b * b)
        out[r] = e
    return out


def train(sub, n, ent, rng):
    """第一阶段：与 pce_scale.run_quantum 相同；第二阶段：每个 τ 从第一阶段参数出发加间隔下限。"""
    m = sub["m"]; L = max(4, int(np.ceil(m / n))); circ = ps.Circuit(n, L, ent)
    slots = ps.make_slots(n, m); alpha = 1.5 * n

    def make_fun(tau):
        def fun(th):
            e = circ.expect(circ.run(th), slots)
            s = np.tanh(alpha * e)
            Lv, gs = ps.loss_s(sub, s)
            c = gs * alpha * (1 - s ** 2)
            if tau > 0:
                v = np.maximum(0, tau - np.abs(e))
                Lv = Lv + W_M * (v ** 2).sum()
                c = c - 2 * W_M * v * np.sign(e)
            return Lv, circ.grad(th, slots, c)
        return fun
    th0, _, _ = ps.adam(make_fun(0.0), rng.uniform(-np.pi, np.pi, circ.npar))
    out = {0.0: th0}
    for tau in TAUS:
        th, _, _ = ps.adam(make_fun(tau), th0.copy(), steps=STEPS2)
        out[tau] = th
    return circ, slots, out


def job(seed):
    t0 = time.time()
    P, q, D, R = ps.instance(seed)
    start = ps.total(D, R); groups = ps.pick_groups(P, R)
    res = dict(seed=seed, start=float(start), sizes={})
    for K in SIZES:
        if K not in groups: continue
        grp = groups[K]; sub = ps.build_sub(P, q, D, R, grp); m = sub["m"]
        rng = np.random.default_rng(seed * 100 + K)
        # 质量参照：随机方案的平均目标值（下端）与模拟退火最好值（上端）
        Lr = float(np.mean([ps.loss_u(sub, rng.integers(0, 2, m).astype(float))[0] for _ in range(200)]))
        Lsa = min(float(ps.loss_u(sub, ps.run_sa(sub, rng)[0].astype(float))[0]) for _ in range(5))
        nlist = [ps.SIZES[K]] + ([ps.SIZES[K] + EXTRA] if EXTRA and K != 12 else [])
        runs = []
        for n in nlist:
            if m > 3 * n * (n - 1) // 2: continue
            for meth, ent in (("Q", True), ("P", False)):
                for r in range(R_RESTART):
                    circ, slots, ths = train(sub, n, ent, rng)
                    for tau, th in ths.items():
                        corr, single = full_expect(circ, circ.run(th), slots)
                        flip0 = (corr < 0).astype(int)
                        L0 = float(ps.loss_u(sub, flip0.astype(float))[0])
                        row = dict(meth=meth, n=n, tau=tau, restart=r, ncz=circ.ncz, layers=circ.L,
                                   min_margin=float(np.abs(corr).min()), med_margin=float(np.median(np.abs(corr))),
                                   frac_ge_tau=float(np.mean(np.abs(corr) >= tau - 1e-9)),
                                   L_clean=L0, noise={})
                        for fcz in FCZ:
                            for mit in (False, True):
                                E = noisy_readouts(corr, single, slots, circ.ncz, fcz, rng, mit)
                                fl = (E < 0).astype(int)
                                Ls = [float(ps.loss_u(sub, f.astype(float))[0]) for f in fl]
                                row["noise"][f"{fcz}_{int(mit)}"] = dict(
                                    exact=float(np.mean((fl == flip0).all(1))),
                                    flipfrac=float(np.mean(fl != flip0)),
                                    L_mean=float(np.mean(Ls)))
                        if tau in (0.0, 0.1):     # 真实路程只算两档，省时间
                            ev = ps.evaluate(P, q, D, R, grp, sub, flip0, start)
                            row["impr_before"] = ev.get("impr_before"); row["impr_after"] = ev.get("impr_after")
                        runs.append(row)
        res["sizes"][K] = dict(m=m, L_rand=Lr, L_sa=Lsa, runs=runs)
    res["sec"] = round(time.time() - t0, 1)
    return res


if __name__ == "__main__":
    from multiprocessing import Pool
    s0, s1 = int(sys.argv[1]), int(sys.argv[2])
    nproc = int(sys.argv[3]) if len(sys.argv) > 3 else os.cpu_count()
    tag = sys.argv[4] if len(sys.argv) > 4 else "main"
    rows = []; t0 = time.time()
    with Pool(nproc) as pool:
        for r in pool.imap_unordered(job, range(s0, s1)):
            rows.append(r); json.dump(rows, open(f"pce_margin_{tag}.json", "w"))
            print(len(rows), "个实例已完成", round(time.time() - t0), "秒", flush=True)
    print("done", flush=True)
