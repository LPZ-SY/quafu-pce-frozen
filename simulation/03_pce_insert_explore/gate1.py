"""第一步出题检查（纯经典，探索性）：不看答案的选题规则 + 枚举只用于打分。
规则：在所有"互为最近的 3 辆车"组合里，取总装载率最高的一组；在其中最满的车里随机选一个客户，
连同 3 辆车内离它最近的 K-1 个客户一起拿走；每个客户保留插入代价最小的 ⌊84/K⌋ 个空位。
用法：python3 gate1.py K 种子起 种子止 进程数"""
import sys, os, json, time, itertools
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pce_insert2 as I, pce_scale as S

def pick_tight(P, q, R):
    cen = np.array([P[list(r)].mean(0) for r in R]); best = None
    for k in range(len(R)):
        nb = np.argsort(np.linalg.norm(cen - cen[k], axis=1))[:3]
        load = sum(q[list(R[j])].sum() for j in nb) / (3 * I.CAP)
        if best is None or load > best[0]: best = (load, sorted(int(j) for j in nb))
    return best[1], best[0]

def build_tight(seed, K):
    inst = S.instance(seed); P, q, D, R = inst
    grp, fill = pick_tight(P, q, R)
    rng = np.random.default_rng(seed * 10 + K)
    fullest = max(grp, key=lambda r: q[list(R[r])].sum())
    c0 = int(rng.choice(sorted(R[fullest])))
    pool = sorted(set().union(*[R[r] for r in grp]))
    rem = sorted(pool, key=lambda c: (D[c0, c], c))[:K]
    sb = I.build(seed, K, grp=grp, rem=rem, inst=inst); sb["fill"] = fill
    return sb

def enum_fast(sb):
    K = len(sb["rem"]); cands = [np.array(sorted(int(sb["var"][v][1]) for v in b)) for b in sb["blocks"]]
    G = sb["G"]; ins = np.zeros((K, G)); 
    for v, (i, g) in enumerate(sb["var"]): ins[i, g] = sb["lin"][v] * sb["sc"]
    pc = {}
    for (v1, v2), w in zip(sb["pv"], sb["pw"]):
        (i1, g1), (i2, g2) = sb["var"][v1], sb["var"][v2]; pc[(i1, i2, g1)] = w * sb["sc"]
    grids = np.stack(np.meshgrid(*cands, indexing="ij"), -1).reshape(-1, K)
    tot = sum(ins[i, grids[:, i]] for i in range(K))
    for i, j in itertools.combinations(range(K), 2):
        same = grids[:, i] == grids[:, j]
        if same.any():
            gv = grids[same, i]; tot[same] += np.array([pc.get((i, j, g), 0.0) for g in gv])
    route_of = np.array([g[0] for g in sb["gaps"]]); qrem = np.array([sb["q"][c] for c in sb["rem"]])
    load = np.tile(sb["loads"], (len(grids), 1))
    for i in range(K): np.add.at(load, (np.arange(len(grids)), route_of[grids[:, i]]), qrem[i])
    feas = (load <= I.CAP).all(1)
    occ = np.zeros((len(grids), G), int)
    for i in range(K): np.add.at(occ, (np.arange(len(grids)), grids[:, i]), 1)
    feas &= (occ <= 2).all(1)
    tot[~feas] = np.inf; k = int(np.argmin(tot))
    return list(map(int, grids[k])), float(tot[k]), int(len(grids)), int(feas.sum())

def job(args):
    seed, K = args; t0 = time.time()
    sb = build_tight(seed, K)
    ch, opt, n, nf = enum_fast(sb)
    h = I.hard(sb, ch); assert abs(h["route"] - opt) < 1e-6 and h["feasible"], (h, opt)
    lf = I.lightfull(sb, ch)
    out = dict(seed=seed, K=K, m=sb["m"], fill=sb["fill"], n=n, nfeas=nf, opt_route=opt, opt_light=lf["light"], opt_full=lf["full"],
               moved=int(sum(sb["gaps"][g][0] != [k for k, r in enumerate(sb["grp"]) if c in sb["R"][r]][0] for c, g in zip(sb["rem"], ch))))
    for nm, rg in (("后悔值", True), ("贪心", False)):
        hh = I.heuristic(sb, rg); out[nm] = dict(route=hh["route"], feasible=hh["feasible"],
                                               gap=(hh["route"] - opt) / sb["start"] * 100, **I.lightfull(sb, seqs=hh["seqs"]))
    out["sec"] = round(time.time() - t0, 1)
    return out

if __name__ == "__main__":
    from multiprocessing import Pool
    K, s0, s1, npr = map(int, sys.argv[1:5]); rows = []; t0 = time.time()
    with Pool(npr) as p:
        for r in p.imap_unordered(job, [(s, K) for s in range(s0, s1)]):
            rows.append(r); json.dump(rows, open(f"gate1_K{K}.json", "w"), ensure_ascii=False)
            print(len(rows), r["seed"], round(time.time() - t0), flush=True)
