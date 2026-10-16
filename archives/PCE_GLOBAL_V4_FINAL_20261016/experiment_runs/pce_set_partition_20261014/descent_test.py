"""起点价值测试：PCE 读出解 → 单客户换车的最陡下降（离散代理目标）；对照：随机起点 + 同样的下降。"""
import json, zlib, numpy as np, assign_pce as A
def descend(S, v):
    v = v.copy(); n, K, Dn, q, cap = S['n'], S['K'], S['Dn'], S['q'], S['cap']
    v = np.where(v >= K, v % K, v)                     # 无效编号先折回（随机起点同样处理）
    load = np.array([q[v == k].sum() for k in range(K)])
    pen = lambda ld: A.cap_pen(np.maximum(0, ld - cap) / cap)
    steps = 0
    while True:
        best = (-1e-12, None, None)
        for i in range(n):
            a = v[i]; base = Dn[i, v == a].sum() - Dn[i, i]
            for b in range(K):
                if b == a: continue
                d = Dn[i, v == b].sum() - base + pen(load[a]-q[i]) + pen(load[b]+q[i]) - pen(load[a]) - pen(load[b])
                if d < best[0]: best = (d, i, b)
        if best[1] is None: return v, steps
        _, i, b = best; load[v[i]] -= q[i]; load[b] += q[i]; v[i] = b; steps += 1
if __name__ == '__main__':
    rows = json.load(open('results_trainability/runs.json', encoding='utf-8'))
    out = {}
    print(f"{'实例':10s}{'方法':6s}{'下降前合法':>8s}{'下降后合法':>8s}{'下降后差距%均值':>14s}{'最好':>7s}{'下降步数':>8s}")
    for n in A.SP.INSTANCES:
        S = A.setup(n)
        for kind in ('Q', 'LR'):
            rs = [r for r in rows if r['instance'] == n and r['kind'] == kind]
            before = after = 0; gaps = []; st = []
            for r in rs:
                v = A.decode(S, (np.array(r['outputs']) > 0).astype(int)); before += A.assess(S, v)['feasible']
                w, s = descend(S, v); d = A.assess(S, w); after += d['feasible']; st.append(s)
                if d['feasible']: gaps.append(d['gap_bks_percent'])
            print(f"{n:10s}{kind:6s}{before:>5d}/{len(rs)}{after:>6d}/{len(rs)}{np.mean(gaps) if gaps else float('nan'):14.1f}{min(gaps) if gaps else float('nan'):7.1f}{np.mean(st):8.1f}")
            out[(n, kind)] = gaps
        rng = np.random.default_rng(zlib.crc32(n.encode())); gaps = []; st = []; ok = 0
        for r in range(64):
            w, s = descend(S, rng.integers(0, S['K'], S['n'])); d = A.assess(S, w); ok += d['feasible']; st.append(s)
            if d['feasible']: gaps.append(d['gap_bks_percent'])
        g4 = [np.mean(gaps[k:k+4]) for k in range(0, len(gaps)-3, 4)]
        print(f"{n:10s}{'随机':6s}{'':>8s}{ok:>6d}/64{np.mean(gaps):14.1f}{min(gaps):7.1f}{np.mean(st):8.1f}   每4个一组的均值范围 {min(g4):.1f}–{max(g4):.1f}", flush=True)
