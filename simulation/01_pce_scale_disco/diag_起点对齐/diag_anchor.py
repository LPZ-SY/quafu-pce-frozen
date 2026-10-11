"""诊断：原实现 vs 起点对齐 + 改动规模惩罚 + 间隔下限。只做 3 辆车（8 比特）。
用法：python3 diag_anchor.py s0 s1 nproc"""
import sys, os, json, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pce_scale as S

MU = 0.05          # 每换一个客户的惩罚（目标已按最大系数归一化）
TAU, MW = 0.15, 5.0  # 间隔下限与权重


def run_q(sub, n, ent, rng, anchor, margin, alpha=None):
    m = sub["m"]; L = max(4, int(np.ceil(m / n))); circ = S.Circuit(n, L, ent)
    slots = S.make_slots(n, m); alpha = alpha or 1.5 * n
    th0 = rng.uniform(-np.pi, np.pi, circ.npar)
    c = np.sign(circ.expect(circ.run(th0), slots)) if anchor else np.ones(m)
    c[c == 0] = 1

    def fun(th):
        e = circ.expect(circ.run(th), slots)
        t = np.tanh(alpha * e); s = c * t
        if margin:
            u = (1 - s) / 2
            Lv, gu = S.loss_u(sub, u)
            Lv += MU * u.sum(); gs = -0.5 * (gu + MU)
            r = np.maximum(0, TAU - np.abs(e))
            Lv += MW * (r ** 2).sum()
            ge_m = -2 * MW * r * np.sign(e)
        else:
            Lv, gs = S.loss_s(sub, s); ge_m = 0
        ge = gs * c * alpha * (1 - t ** 2) + ge_m
        return Lv, circ.grad(th, slots, ge)
    th, Lb, it = S.adam(fun, th0)
    e = circ.expect(circ.run(th), slots)
    flip = (c * e < 0).astype(int)
    return flip, dict(min_gap=float(np.abs(e).min()), med_gap=float(np.median(np.abs(e))), iters=it)


def run_c1(sub, rng, anchor):
    m = sub["m"]

    def fun(y):
        s = np.tanh(y); u = (1 - s) / 2
        Lv, gu = S.loss_u(sub, u)
        if anchor: Lv += MU * u.sum(); gu = gu + MU
        return Lv, -0.5 * gu * (1 - s ** 2)
    y0 = np.abs(rng.normal(0, 1, m)) if anchor else rng.normal(0, 1, m)
    y, Lb, it = S.adam(fun, y0)
    return (y < 0).astype(int), {}


def job(seed):
    P, q, D, R = S.instance(seed); start = S.total(D, R)
    grp = S.pick_groups(P, R)[3]; n = 8; sub = S.build_sub(P, q, D, R, grp)
    rng = np.random.default_rng(seed)
    out = dict(seed=seed, m=sub["m"], res={})
    arms = {"Q原": lambda: run_q(sub, n, True, rng, False, False),
            "Q对齐": lambda: run_q(sub, n, True, rng, True, True),
            "P对齐": lambda: run_q(sub, n, False, rng, True, True),
            "C1对齐": lambda: run_c1(sub, rng, True)}
    for name, f in arms.items():
        runs = []
        for r in range(3):
            flip, info = f()
            ev = S.evaluate(P, q, D, R, grp, sub, flip, start); ev.update(info); runs.append(ev)
        out["res"][name] = runs
        if name == "Q对齐":       # 随机对照：换车个数与 Q对齐 逐次相同
            rr = []
            for ev in runs:
                k = ev.get("nflip", 0); fl = np.zeros(sub["m"], int)
                fl[rng.choice(sub["m"], k, replace=False)] = 1
                rr.append(S.evaluate(P, q, D, R, grp, sub, fl, start))
            out["res"]["随机同个数"] = rr
    return out


if __name__ == "__main__":
    from multiprocessing import Pool
    s0, s1, npr = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
    rows = []; t0 = time.time()
    with Pool(npr) as p:
        for r in p.imap_unordered(job, range(s0, s1)):
            rows.append(r); json.dump(rows, open("diag_anchor.json", "w"), ensure_ascii=False)
            print(len(rows), round(time.time() - t0), flush=True)
