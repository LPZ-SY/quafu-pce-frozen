"""泡利关联编码：精确插入损失的机制检验（预登记：泡利关联编码_精确插入损失_机制检验_预登记）。
用法：python3 pce_insert.py K 种子起 种子止 进程数 输出名 [容量惩罚系数]
"""
import sys, os, json, time, itertools
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pce_scale as S
from scale_pairs import rlen
from numba import njit

CAP = S.CAP
NQ = 8                      # 比特数
BETA = 1.5 * NQ             # softmax 放大系数
TAU, MW = 0.1, 5.0          # 第二段：第一名领先第二名的下限与权重
A3 = 5.0                    # 同一空位 3 人及以上的惩罚系数
A_CAP = float(os.environ.get("A_CAP", "0.5"))   # 容量惩罚系数：在 59001–59005 上按 TUNING_RULE.txt 调定为 0.5，已冻结
STEP1, STEP2, RESTART = 1000, 500, 5
EXPR_RESTART, EXPR_STEPS = 10, 3000
MAXV = 3 * NQ * (NQ - 1) // 2   # 84


@njit(cache=True)
def route_order(D, nodes):
    """与 scale_pairs.route_cost 完全相同的算法，返回顺序。"""
    n = len(nodes)
    r = np.empty(n, np.int64)
    if n == 0: return r
    used = np.zeros(n, np.bool_); cur = 0
    for k in range(n):
        bj, bd = -1, 1e18
        for j in range(n):
            if not used[j] and D[cur, nodes[j]] < bd: bd, bj = D[cur, nodes[j]], j
        used[bj] = True; r[k] = nodes[bj]; cur = nodes[bj]
    best = rlen(D, r, n); tmp = np.empty(n, np.int64)
    imp = True
    while imp:
        imp = False
        for i in range(n - 1):
            for j in range(i + 1, n):
                for k in range(n): tmp[k] = r[k]
                for k in range(j - i + 1): tmp[i + k] = r[j - k]
                v = rlen(D, tmp, n)
                if v < best - 1e-9:
                    best = v; r[:] = tmp; imp = True
        for L in range(1, 4):
            if L >= n: break
            for i in range(n - L + 1):
                for kk in range(n - L + 1):
                    if kk == i: continue
                    for rev in range(2):
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
    return r


def seq_len(D, seq):
    return float(rlen(D, np.asarray(seq, np.int64), len(seq))) if len(seq) else 0.0


# ---------------- 子问题 ----------------
def build(seed, K):
    P, q, D, R = S.instance(seed); start = S.total(D, R)
    grp = S.pick_groups(P, R)[3]
    rng = np.random.default_rng(seed * 10 + K)
    pool = sorted(set().union(*[R[r] for r in grp]))
    c0 = int(rng.choice(pool))
    rem_c = sorted(pool, key=lambda c: (D[c0, c], c))[:K]          # c0 及最近的 K-1 个
    seqs, loads = [], []
    for r in grp:
        left = sorted(R[r] - set(rem_c))
        seqs.append([int(x) for x in route_order(D, np.array(left, np.int64))])
        loads.append(int(q[left].sum()) if left else 0)
    gaps = []                                                       # (车序号, 位置, a, b)
    for k, sq in enumerate(seqs):
        ext = [0] + sq + [0]
        for j in range(len(sq) + 1): gaps.append((k, j, ext[j], ext[j + 1]))
    G = len(gaps)
    ins = np.array([[D[a, c] + D[c, b] - D[a, b] for (_, _, a, b) in gaps] for c in rem_c])
    per = G if K * G <= MAXV else MAXV // K
    cand = [list(np.argsort(ins[i], kind="stable")[:per]) if per < G else list(range(G)) for i in range(K)]
    var = [(i, g) for i in range(K) for g in sorted(cand[i])]
    m = len(var)
    blocks, s = [], 0
    for i in range(K): n_i = len(cand[i]); blocks.append(np.arange(s, s + n_i)); s += n_i
    sc = float(np.abs([ins[i, g] for i, g in var]).max())
    lin = np.array([ins[i, g] for i, g in var]) / sc
    pairs = []                                                      # (v1, v2, 修正项/sc)
    for v1, v2 in itertools.combinations(range(m), 2):
        (i1, g1), (i2, g2) = var[v1], var[v2]
        if i1 != i2 and g1 == g2:
            _, _, a, b = gaps[g1]; c1, c2 = rem_c[i1], rem_c[i2]
            both = min(D[a, c1] + D[c1, c2] + D[c2, b], D[a, c2] + D[c2, c1] + D[c1, b]) - D[a, b]
            pairs.append((v1, v2, (both - ins[i1, g1] - ins[i2, g2]) / sc))
    pv = np.array([[a, b] for a, b, _ in pairs], int).reshape(-1, 2)
    pw = np.array([w for _, _, w in pairs])
    vroute = np.array([gaps[g][0] for _, g in var]); vgap = np.array([g for _, g in var])
    vq = np.array([q[rem_c[i]] for i, _ in var], float)
    base_len = sum(seq_len(D, sq) for sq in seqs)
    orig_grp = sum(S.cost_of(D, R[r]) for r in grp)
    return dict(seed=seed, K=K, P=P, q=q, D=D, R=R, start=start, grp=grp, rem=rem_c, seqs=seqs,
                loads=np.array(loads, float), gaps=gaps, G=G, var=var, m=m, blocks=blocks, sc=sc,
                lin=lin, pv=pv, pw=pw, vroute=vroute, vgap=vgap, vq=vq, base_len=base_len,
                orig_grp=orig_grp, trimmed=per < G)


# ---------------- 损失 ----------------
def penalties(sb, w):
    load = sb["loads"].copy(); np.add.at(load, sb["vroute"], sb["vq"] * w)
    over = np.maximum(0, (load - CAP) / 5.5)
    occ = np.zeros(sb["G"]); np.add.at(occ, sb["vgap"], w)
    ex = np.maximum(0, occ - 2)
    return A_CAP * (over ** 2).sum() + A3 * (ex ** 2).sum(), over, ex


def loss_w(sb, w):
    """返回（损失，对 w 的梯度）。w 为每个变量的软分配。"""
    pv, pw = sb["pv"], sb["pw"]
    L = sb["lin"] @ w
    g = sb["lin"].copy()
    if len(pw):
        L += (pw * w[pv[:, 0]] * w[pv[:, 1]]).sum()
        np.add.at(g, pv[:, 0], pw * w[pv[:, 1]]); np.add.at(g, pv[:, 1], pw * w[pv[:, 0]])
    Lp, over, ex = penalties(sb, w)
    g = g + (2 * A_CAP * over / 5.5)[sb["vroute"]] * sb["vq"] + (2 * A3 * ex)[sb["vgap"]]
    return L + Lp, g


def soft(sb, e):
    w = np.empty_like(e)
    for b in sb["blocks"]:
        z = BETA * e[b]; z = z - z.max(); ez = np.exp(z); w[b] = ez / ez.sum()
    return w


def loss_e(sb, e, margin):
    w = soft(sb, e); L, gw = loss_w(sb, w)
    ge = np.empty_like(e)
    for b in sb["blocks"]:
        ge[b] = BETA * w[b] * (gw[b] - (w[b] * gw[b]).sum())
    if margin:
        for b in sb["blocks"]:
            if len(b) < 2: continue
            o = np.argsort(-e[b]); i1, i2 = b[o[0]], b[o[1]]; d = e[i1] - e[i2]
            if d < TAU:
                L += MW * (TAU - d) ** 2; ge[i1] -= 2 * MW * (TAU - d); ge[i2] += 2 * MW * (TAU - d)
    return L, ge


# ---------------- 解码与评分 ----------------
def decode(sb, e):
    return [int(sb["var"][b[int(np.argmax(e[b]))]][1]) for b in sb["blocks"]]


def onehot(sb, choice):
    w = np.zeros(sb["m"])
    for i, b in enumerate(sb["blocks"]):
        for v in b:
            if sb["var"][v][1] == choice[i]: w[v] = 1
    return w


def build_seqs(sb, choice):
    """在固定车内顺序下插入；同一空位多人时按 permutations 取最短。"""
    D = sb["D"]; seqs = [list(s) for s in sb["seqs"]]
    bygap = {}
    for i, g in enumerate(choice): bygap.setdefault(g, []).append(sb["rem"][i])
    for g in sorted(bygap, key=lambda g: (sb["gaps"][g][0], -sb["gaps"][g][1])):   # 同车从后往前插，位置不乱
        k, j, a, b = sb["gaps"][g]; cs = bygap[g]
        best = min(itertools.permutations(cs), key=lambda p: D[a, p[0]] + sum(D[p[t], p[t + 1]] for t in range(len(p) - 1)) + D[p[-1], b])
        seqs[k] = seqs[k][:j] + list(best) + seqs[k][j:]
    return seqs


def hard(sb, choice):
    seqs = build_seqs(sb, choice)
    dl = sum(seq_len(sb["D"], s) for s in seqs) - sb["base_len"]
    w = onehot(sb, choice); Lp, over, ex = penalties(sb, w)
    load = sb["loads"].copy(); np.add.at(load, sb["vroute"], sb["vq"] * w)
    return dict(hard=float(dl / sb["sc"] + Lp), route=float(dl), feasible=bool((load <= CAP + 1e-9).all()),
                over_amt=float(np.maximum(0, load - CAP).sum()), triple=int((ex > 0).sum()))


def lightfull(sb, choice=None, seqs=None):
    """轻处理（超载修正 + 车内重排）与完整精修（搬一个、换一个到收敛）。返回占起点总路程的百分比缩短。"""
    D, q, grp = sb["D"], sb["q"], sb["grp"]
    if seqs is None: seqs = build_seqs(sb, choice)
    mem = {r: set(seqs[k]) for k, r in enumerate(grp)}
    guard = 0
    while guard < 200:
        bad = [r for r in grp if q[list(mem[r])].sum() > CAP]
        if not bad: break
        guard += 1; best = None
        for r in bad:
            cr = S.cost_of(D, mem[r])
            for x in mem[r]:
                rem = S.cost_of(D, mem[r] - {x}) - cr
                for s2 in grp:
                    if s2 == r or q[list(mem[s2])].sum() + q[x] > CAP: continue
                    add = S.cost_of(D, mem[s2] | {x}) - S.cost_of(D, mem[s2])
                    if best is None or rem + add < best[0]: best = (rem + add, r, s2, x)
        if best is None: return dict(light=None, full=None)
        _, r, s2, x = best; mem[r].discard(x); mem[s2].add(x)
    R2 = [set(r) for r in sb["R"]]
    for r in grp: R2[r] = set(mem[r])
    t1 = S.total(D, R2); R2 = S.l1(D, q, R2, grp); t2 = S.total(D, R2)
    st = sb["start"]
    return dict(light=float((st - t1) / st * 100), full=float((st - t2) / st * 100))


def enumerate_all(sb):
    cands = [sorted(int(sb["var"][v][1]) for v in b) for b in sb["blocks"]]
    best = None; best_any = None; n = 0
    for ch in itertools.product(*cands):
        h = hard(sb, list(ch)); n += 1
        if h["triple"]: continue                      # 同一空位 3 人及以上不属于合法方案（损失中被禁止）
        if best_any is None or h["hard"] < best_any[0]: best_any = (h["hard"], list(ch))
        if h["feasible"] and (best is None or h["route"] < best[0]): best = (h["route"], list(ch), h["hard"])
    return dict(n=n, opt_route=best[0] if best else None, opt_choice=best[1] if best else None,
                opt_hard=best[2] if best else None, min_hard=best_any[0], min_hard_choice=best_any[1])


# ---------------- 经典插入启发式 ----------------
def heuristic(sb, regret):
    D, q = sb["D"], sb["q"]; seqs = [list(s) for s in sb["seqs"]]; loads = list(sb["loads"])
    left = list(range(len(sb["rem"])))
    while left:
        opts = {}
        for i in left:
            c = sb["rem"][i]; o = []
            for k, sq in enumerate(seqs):
                ext = [0] + sq + [0]
                for j in range(len(sq) + 1):
                    cost = D[ext[j], c] + D[c, ext[j + 1]] - D[ext[j], ext[j + 1]]
                    o.append((cost, loads[k] + q[c] > CAP, k, j))
            o.sort(key=lambda t: (t[1], t[0])); opts[i] = o
        if regret:
            def rg(i):
                f = [t for t in opts[i] if not t[1]]
                return (f[1][0] - f[0][0]) if len(f) > 1 else 1e9
            i = max(left, key=rg)
        else:
            i = left[0]
        cost, ov, k, j = opts[i][0]
        seqs[k].insert(j, sb["rem"][i]); loads[k] += q[sb["rem"][i]]; left.remove(i)
    dl = sum(seq_len(D, s) for s in seqs) - sb["base_len"]
    feas = all(l <= CAP for l in loads)
    return dict(route=float(dl), feasible=feas, seqs=seqs)


# ---------------- 三组优化方法 ----------------
def train(sb, kind, rng):
    m = sb["m"]; t0 = time.time(); nev = 0
    if kind in ("Q", "P"):
        L = max(4, int(np.ceil(m / NQ))); circ = S.Circuit(NQ, L, kind == "Q"); slots = S.make_slots(NQ, m)
        th0 = rng.uniform(-np.pi, np.pi, circ.npar); npar = circ.npar

        def E(th): return circ.expect(circ.run(th), slots)

        def fun_m(margin):
            def f(th):
                e = E(th); Lv, ge = loss_e(sb, e, margin)
                return Lv, circ.grad(th, slots, ge)
            return f
    else:
        th0 = rng.normal(0, 1, m); npar = m

        def E(th): return np.tanh(th)

        def fun_m(margin):
            def f(th):
                e = np.tanh(th); Lv, ge = loss_e(sb, e, margin)
                return Lv, ge * (1 - e ** 2)
            return f
    th1, _, it1 = S.adam(fun_m(False), th0, steps=STEP1)
    th2, _, it2 = S.adam(fun_m(True), th1, steps=STEP2)
    e = E(th2); w = soft(sb, e)
    softv = float(loss_w(sb, w)[0])
    gaps_top = []
    for b in sb["blocks"]:
        s2 = np.sort(e[b])[::-1]; gaps_top.append(float(s2[0] - s2[1]) if len(b) > 1 else 1.0)
    ch = decode(sb, e)
    out = dict(soft=softv, choice=ch, npar=int(npar), nev=int(it1 + it2), sec=round(time.time() - t0, 2),
               min_lead=float(min(gaps_top)), min_abs=float(np.abs(e).min()))
    out.update(hard(sb, ch))
    return out


def express(sb, kind, target, rng):
    """以 target（每个客户的空位）为目标，训练线路复现其最大值位置。"""
    m = sb["m"]; L = max(4, int(np.ceil(m / NQ))); circ = S.Circuit(NQ, L, kind == "Q"); slots = S.make_slots(NQ, m)
    tgt = []
    for i, b in enumerate(sb["blocks"]):
        tgt.append([v for v in b if sb["var"][v][1] == target[i]][0])

    def f(th):
        e = circ.expect(circ.run(th), slots); L_ = 0.0; ge = np.zeros(m)
        for i, b in enumerate(sb["blocks"]):
            t = tgt[i]
            for v in b:
                if v == t: continue
                d = e[t] - e[v]
                if d < TAU:
                    L_ += (TAU - d) ** 2; ge[t] -= 2 * (TAU - d); ge[v] += 2 * (TAU - d)
        return L_, circ.grad(th, slots, ge)
    res = []
    for r in range(EXPR_RESTART):
        th, Lb, it = S.adam(f, rng.uniform(-np.pi, np.pi, circ.npar), steps=EXPR_STEPS)
        e = circ.expect(circ.run(th), slots)
        ok = decode(sb, e) == list(target)
        lead = min(float(e[tgt[i]] - max(e[v] for v in b if v != tgt[i])) for i, b in enumerate(sb["blocks"]) if len(b) > 1)
        res.append(dict(ok=bool(ok), lead=lead, loss=float(Lb), iters=int(it)))
        if ok and lead >= TAU - 1e-9: break       # 已以足够间隔复现，不再试
    return res


def job(args):
    seed, K, do_expr = args
    t0 = time.time(); sb = build(seed, K)
    rng = np.random.default_rng(seed * 1000 + K)
    out = dict(seed=seed, K=K, m=sb["m"], G=sb["G"], trimmed=sb["trimmed"], start=sb["start"], sc=sb["sc"])
    en = enumerate_all(sb); out["enum"] = en
    if en["opt_choice"] is not None: out["enum"].update({"opt_lf": lightfull(sb, en["opt_choice"])})
    out["res"] = {}
    for kind in ("Q", "P", "C1"):
        runs = []
        for r in range(RESTART):
            x = train(sb, kind, rng); x.update(lightfull(sb, x["choice"])); runs.append(x)
        out["res"][kind] = runs
    rr = []
    for r in range(RESTART):
        ch = [int(rng.choice([sb["var"][v][1] for v in b])) for b in sb["blocks"]]
        x = dict(choice=ch); x.update(hard(sb, ch)); x.update(lightfull(sb, ch)); rr.append(x)
    out["res"]["随机"] = rr
    for nm, rg in (("后悔值插入", True), ("贪心插入", False)):
        h = heuristic(sb, rg); lf = lightfull(sb, seqs=h["seqs"])
        out["res"][nm] = [dict(route=h["route"], feasible=h["feasible"], **lf)]
    if do_expr and en["opt_choice"] is not None:
        out["expr"] = {k: express(sb, k, en["opt_choice"], rng) for k in ("Q", "P")}
    out["sec"] = round(time.time() - t0, 1)
    return out


if __name__ == "__main__":
    from multiprocessing import Pool
    K, s0, s1, npr, tag = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
    do_expr = os.environ.get("EXPR", "1") == "1"
    rows = []; t0 = time.time()
    with Pool(npr) as p:
        for r in p.imap_unordered(job, [(s, K, do_expr) for s in range(s0, s1)]):
            rows.append(r); json.dump(rows, open(f"ins_{tag}.json", "w"), ensure_ascii=False)
            print(len(rows), r["seed"], round(time.time() - t0), flush=True)
    print("done", flush=True)
