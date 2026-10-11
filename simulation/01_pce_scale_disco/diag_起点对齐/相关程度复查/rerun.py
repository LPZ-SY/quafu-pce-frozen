import sys, os, json, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pce_scale as S, diag_two as D2

def exact_delta(sub, D, flip):
    mem = sub["members"](flip); base = sub["members"](np.zeros(sub["m"], int))
    return sum(S.cost_of(D, mem[r]) - S.cost_of(D, base[r]) for r in mem)

def job(seed):
    P, q, D, R = S.instance(seed); start = S.total(D, R)
    grp = S.pick_groups(P, R)[3]; n = 8; sub = S.build_sub(P, q, D, R, grp)
    rng = np.random.default_rng(seed)
    out = dict(seed=seed, m=sub["m"], start=start, res={})
    arms = {"Q原": lambda: D2.run_q(sub, n, True, rng, False, False),
            "Q对齐": lambda: D2.run_q(sub, n, True, rng, True, True, two=True),
            "P对齐": lambda: D2.run_q(sub, n, False, rng, True, True, two=True),
            "C1对齐": lambda: D2.run_c1(sub, rng, True)}
    def rec(flip):
        ev = S.evaluate(P, q, D, R, grp, sub, flip, start)
        u = flip.astype(float)
        ev["proxy"] = float(S.loss_u(sub, u)[0])            # 近似路程 + 容量罚（训练目标的离散值）
        ev["proxy_mu"] = ev["proxy"] + D2.MU * float(u.sum())  # 加上换车惩罚
        ev["exact_sub"] = float(exact_delta(sub, D, flip) / sub["sc"])  # 子问题内精确路程变化（同一归一化）
        ev["flip"] = "".join(map(str, flip))
        return ev
    for name, f in arms.items():
        runs = []
        for r in range(3):
            flip, info = f(); ev = rec(flip); ev.update(info); runs.append(ev)
        out["res"][name] = runs
        if name == "Q对齐":
            rr = []
            for ev in runs:
                k = ev.get("nflip", 0); fl = np.zeros(sub["m"], int)
                fl[rng.choice(sub["m"], k, replace=False)] = 1
                rr.append(rec(fl))
            out["res"]["随机同个数"] = rr
    return out

if __name__ == "__main__":
    from multiprocessing import Pool
    rows = []; t0 = time.time()
    with Pool(2) as p:
        for r in p.imap_unordered(job, range(20001, 20021)):
            rows.append(r); json.dump(rows, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "rerun.json"), "w"), ensure_ascii=False)
            print(len(rows), round(time.time() - t0), flush=True)
