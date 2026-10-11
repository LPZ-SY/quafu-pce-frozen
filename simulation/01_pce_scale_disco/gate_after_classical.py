"""量子在经典之后出场：第 0 步闸门（纯经典，已预登记，见项目文档《量子在经典之后出场_预登记》）。

要回答的问题：
  先用强经典方法把方案改到改不动，在量子能表达的改动范围里（每个客户留在原车，或换到子问题里离它最近的
  另一辆车；一次最多换 8 个客户），还有没有能让路线变短的改动？随机挑客户换，要试多少次才能碰上？

实例、子问题、读出后的统一处理（超载修正 → 重新排路线 → 对子问题内的车做“搬一个＋换一个”）
全部沿用 pce_scale.py，只是把起点从“搬一个＋换一个收敛”换成强局部最优。

强局部最优（迭代局部搜索）：
  每次随机取一辆车和离它质心最近的 3 辆车之一，从两车客户里随机取 2–6 个，各自随机放回两车之一（不超载），
  再对这两辆车做“搬一个＋换一个”；总路程变短就接受，否则撤回。连续 ILS_STOP 次没有改进时停止。

每个子问题的两种探测（候选都按 pce_scale.evaluate 统一处理）：
  随机：换车客户数 k = 1..8，每个 k 随机挑 N_RAND 组客户；
  近似路程引导：用 pce_scale 的近似路程（现解处一阶 + 两两二阶差分 + 容量罚）做束搜索，
               每层保留 BEAM 组，最多 8 个客户，取近似路程最低的 N_PROXY 组（不重复）去真实评估。

运行（在 disco 上，与 pce_scale.py 放在同一文件夹）：
  python3 gate_after_classical.py 30001 30021 $(nproc) gate
  python3 analyze_gate.py pce_gate_gate.json > pce_gate_gate_analysis.txt
可调环境变量：SIZES（默认 "3,6"；加 12 很慢）、ILS_STOP（默认 200）、N_RAND（默认 60）、
              N_PROXY（默认 200）、BEAM（默认 50）、KMAX（默认 8）。
"""
import os, sys, json, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pce_scale as ps

SIZES = [int(x) for x in os.environ.get("SIZES", "3,6").split(",")]
ILS_STOP = int(os.environ.get("ILS_STOP", "200"))
N_RAND = int(os.environ.get("N_RAND", "60"))
N_PROXY = int(os.environ.get("N_PROXY", "200"))
BEAM = int(os.environ.get("BEAM", "50"))
KMAX = int(os.environ.get("KMAX", "8"))
THR = 0.01                                   # 有效改进阈值（%）

# ---------- 路线路程缓存：结果与原函数完全相同，只是不重复计算 ----------
_raw_cost = ps.cost_of
_cache = {}


def _cached_cost(D, S):
    k = frozenset(S)
    v = _cache.get(k)
    if v is None:
        v = _raw_cost(D, S); _cache[k] = v
    return v


ps.cost_of = _cached_cost


# ---------- 强局部最优：迭代局部搜索 ----------
def ils(P, q, D, R, rng):
    R = [set(r) for r in R]
    cur = ps.total(D, R); fails = 0; kicks = 0; acc = 0
    while fails < ILS_STOP:
        kicks += 1
        cen = np.array([P[list(r)].mean(0) for r in R])
        a = int(rng.integers(len(R)))
        near = [int(k) for k in np.argsort(np.linalg.norm(cen - cen[a], axis=1))[1:4]]
        b = near[int(rng.integers(len(near)))]
        pool = list(R[a] | R[b]); k = int(rng.integers(2, 7))
        if len(pool) < k: fails += 1; continue
        pick = rng.choice(pool, k, replace=False)
        Ra, Rb = R[a] - set(pick), R[b] - set(pick)
        ok = True
        for c in pick:
            first, second = (Ra, Rb) if rng.random() < 0.5 else (Rb, Ra)
            if q[list(first)].sum() + q[c] <= ps.CAP: first.add(int(c))
            elif q[list(second)].sum() + q[c] <= ps.CAP: second.add(int(c))
            else: ok = False; break
        if not ok or not Ra or not Rb: fails += 1; continue
        R2 = [set(r) for r in R]; R2[a], R2[b] = Ra, Rb
        R2 = ps.l1(D, q, R2, [a, b])
        new = ps.total(D, R2)
        if new < cur - 1e-9:
            R, cur, fails = [r for r in R2 if r], new, 0; acc += 1
        else:
            fails += 1
    R = ps.l1(D, q, R)                        # 最后全部车再做一遍，保证“搬一个＋换一个”也改不动
    return R, dict(kicks=kicks, accepted=acc)


# ---------- 近似路程引导的束搜索 ----------
def proxy_candidates(sub):
    m = sub["m"]

    def val(fs):
        u = np.zeros(m); u[list(fs)] = 1
        return ps.loss_u(sub, u)[0]
    layer = [frozenset()]; seen = {}
    for _ in range(KMAX):
        nxt = {}
        for fs in layer:
            for i in range(m):
                if i in fs: continue
                g = fs | {i}
                if g not in nxt and g not in seen: nxt[g] = val(g)
        if not nxt: break
        seen.update(nxt)
        layer = [g for g, _ in sorted(nxt.items(), key=lambda t: t[1])[:BEAM]]
    best = sorted(seen.items(), key=lambda t: t[1])[:N_PROXY]
    return [sorted(g) for g, _ in best]


def run_cand(P, q, D, R, grp, sub, flips, start):
    f = np.zeros(sub["m"], int); f[list(flips)] = 1
    ev = ps.evaluate(P, q, D, R, grp, sub, f, start)
    return dict(k=len(flips), impr=float(ev["impr_after"]) if ev.get("feasible") else None,
                flips=[int(x) for x in flips])


def job(seed):
    t0 = time.time()
    P, q, D, R0 = ps.instance(seed)
    l1_total = ps.total(D, R0)
    rng = np.random.default_rng(seed * 7 + 1)
    R, info = ils(P, q, D, R0, rng)
    R = [r for r in R if r]
    start = ps.total(D, R)
    out = dict(seed=seed, l1_total=float(l1_total), start=float(start),
               ils_gain_pct=float((l1_total - start) / l1_total * 100), nroutes=len(R), ils=info, sizes={})
    groups = ps.pick_groups(P, R)
    for K in SIZES:
        if K not in groups or K > len(R): continue
        grp = groups[K]; sub = ps.build_sub(P, q, D, R, grp); m = sub["m"]
        srng = np.random.default_rng(seed * 100 + K)
        rand = []
        for k in range(1, KMAX + 1):
            for _ in range(N_RAND):
                if k > m: break
                rand.append(run_cand(P, q, D, R, grp, sub, srng.choice(m, k, replace=False), start))
        prox = [run_cand(P, q, D, R, grp, sub, fs, start) for fs in proxy_candidates(sub)]
        out["sizes"][K] = dict(m=m, routes=grp, rand=rand, proxy=prox)
    out["sec"] = round(time.time() - t0, 1)
    return out


if __name__ == "__main__":
    from multiprocessing import Pool
    s0, s1 = int(sys.argv[1]), int(sys.argv[2])
    nproc = int(sys.argv[3]) if len(sys.argv) > 3 else os.cpu_count()
    tag = sys.argv[4] if len(sys.argv) > 4 else "gate"
    rows = []; t0 = time.time()
    with Pool(nproc) as pool:
        for r in pool.imap_unordered(job, range(s0, s1)):
            rows.append(r); json.dump(rows, open(f"pce_gate_{tag}.json", "w"))
            print(len(rows), "个实例已完成", round(time.time() - t0), "秒", flush=True)
    print("done", flush=True)
