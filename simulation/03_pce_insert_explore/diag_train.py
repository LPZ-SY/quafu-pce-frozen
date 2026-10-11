"""探索性诊断（不属于预登记正式实验）：
A. 从能复现最优的线路参数出发，按正式损失继续训练，是否漂离最优；
B. softmax 放大系数从小逐步加大（多阶段）对比固定 β=12，命中最优的比例。
用法：python3 diag_train.py 种子起 种子止 进程数"""
import sys, os, json, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pce_insert as I, pce_scale as S

SCHED = [1.5, 3, 6, 12, 24, 48]     # 多阶段 β
STEP_STAGE = 300

def circ_of(sb, ent=True):
    m = sb["m"]; L = max(4, int(np.ceil(m / I.NQ)))
    return S.Circuit(I.NQ, L, ent), S.make_slots(I.NQ, m)

def fun(sb, circ, slots, beta, margin):
    def f(th):
        old = I.BETA; I.BETA = beta
        e = circ.expect(circ.run(th), slots); Lv, ge = I.loss_e(sb, e, margin)
        I.BETA = old
        return Lv, circ.grad(th, slots, ge)
    return f

def gap(sb, en, ch):
    h = I.hard(sb, ch)
    return (h["route"] - en["opt_route"]) / sb["start"] * 100 if h["feasible"] else None

def job(seed):
    sb = I.build(seed, 2); en = I.enumerate_all(sb); rng = np.random.default_rng(seed + 7)
    out = dict(seed=seed, A=[], fixed=[], multi=[], fixed_long=[])
    circ, slots = circ_of(sb)
    # A：先复现最优
    tgt = [[v for v in b if sb["var"][v][1] == en["opt_choice"][i]][0] for i, b in enumerate(sb["blocks"])]
    def sup(th):
        e = circ.expect(circ.run(th), slots); L_ = 0.0; ge = np.zeros(sb["m"])
        for i, b in enumerate(sb["blocks"]):
            t = tgt[i]
            for v in b:
                if v == t: continue
                d = e[t] - e[v]
                if d < I.TAU: L_ += (I.TAU - d) ** 2; ge[t] -= 2 * (I.TAU - d); ge[v] += 2 * (I.TAU - d)
        return L_, circ.grad(th, slots, ge)
    for r in range(3):
        th, _, _ = S.adam(sup, rng.uniform(-np.pi, np.pi, circ.npar), steps=3000)
        e0 = circ.expect(circ.run(th), slots); ok0 = I.decode(sb, e0) == en["opt_choice"]
        if not ok0: continue
        rec = dict(start_ok=True)
        for beta in (12.0, 48.0):
            th2, _, _ = S.adam(fun(sb, circ, slots, beta, False), th.copy(), steps=1000)
            ch = I.decode(sb, circ.expect(circ.run(th2), slots)); rec[f"stay_beta{int(beta)}"] = ch == en["opt_choice"]; rec[f"gap_beta{int(beta)}"] = gap(sb, en, ch)
        out["A"].append(rec); break
    # B：固定 β=12（与正式实验相同：1000 + 500 步，第二段加间隔项）对比多阶段
    for r in range(5):
        th0 = rng.uniform(-np.pi, np.pi, circ.npar)
        t1, _, _ = S.adam(fun(sb, circ, slots, 12.0, False), th0.copy(), steps=1000)
        t2, _, _ = S.adam(fun(sb, circ, slots, 12.0, True), t1, steps=500)
        out["fixed"].append(gap(sb, en, I.decode(sb, circ.expect(circ.run(t2), slots))))
        t = th0.copy()
        for k, beta in enumerate(SCHED):
            t, _, _ = S.adam(fun(sb, circ, slots, beta, k == len(SCHED) - 1), t, steps=STEP_STAGE)
        out["multi"].append(gap(sb, en, I.decode(sb, circ.expect(circ.run(t), slots))))
        # 对照：同样总步数（1800）但固定在最终 β=48
        t3, _, _ = S.adam(fun(sb, circ, slots, 48.0, False), th0.copy(), steps=1500)
        t3, _, _ = S.adam(fun(sb, circ, slots, 48.0, True), t3, steps=300)
        out["fixed_long"].append(gap(sb, en, I.decode(sb, circ.expect(circ.run(t3), slots))))
    return out

if __name__ == "__main__":
    from multiprocessing import Pool
    s0, s1, npr = map(int, sys.argv[1:4]); rows = []; t0 = time.time()
    with Pool(npr) as p:
        for r in p.imap_unordered(job, range(s0, s1)):
            rows.append(r); json.dump(rows, open("diag_train.json", "w"))
            print(len(rows), r["seed"], round(time.time() - t0), flush=True)
