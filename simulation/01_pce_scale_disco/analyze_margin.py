"""间隔下限训练的分析。用法：python3 analyze_margin.py pce_margin_main.json
相对质量 = (随机方案平均目标值 − 该方案目标值) / (随机方案平均目标值 − 模拟退火最好值)：1 表示与模拟退火一样好，0 表示与随机一样。
预登记主检验（每个规模、每种比特数分别做；逐实例配对，先对随机起点取平均；单侧 Wilcoxon）：
  H1：纠缠线路 τ=0.1 的含噪相对质量 > 纠缠线路 τ=0 的含噪相对质量（f_CZ=0.977，读出校准）
  H2：纠缠线路 τ=0.1 的含噪相对质量 > 去掉纠缠门 τ=0.1 的含噪相对质量（同一噪声设置）
真机就绪条件（任一组合满足即可）：纠缠线路某个 τ 下，含噪读出与无噪声完全一致的比例 ≥ 0.8，
  且含噪相对质量不低于同一线路无噪声相对质量 0.05 以上。
"""
import sys, json
import numpy as np
from scipy.stats import wilcoxon

d = json.load(open(sys.argv[1]))
KEY = "0.977_1"
print(f"实例数：{len(d)}；主噪声设置：每个两比特门保真度 0.977，读出误差已校准，每种测量设置 4096 次")


def rq(L, Lr, Ls):
    return (Lr - L) / (Lr - Ls) if Lr != Ls else np.nan


def table(rows_by_inst, meth, n, tau, key=KEY):
    """返回逐实例（随机起点平均后）的各指标。"""
    out = []
    for x in rows_by_inst:
        rs = [r for r in x["runs"] if r["meth"] == meth and r["n"] == n and abs(r["tau"] - tau) < 1e-9]
        if not rs: out.append(None); continue
        Lr, Ls = x["L_rand"], x["L_sa"]
        out.append(dict(
            clean=np.mean([rq(r["L_clean"], Lr, Ls) for r in rs]),
            noisy=np.mean([rq(r["noise"][key]["L_mean"], Lr, Ls) for r in rs]),
            exact=np.mean([r["noise"][key]["exact"] for r in rs]),
            flip=np.mean([r["noise"][key]["flipfrac"] for r in rs]),
            minm=np.median([r["min_margin"] for r in rs]),
            fge=np.mean([r["frac_ge_tau"] for r in rs]),
            ncz=rs[0]["ncz"]))
    return out


def paired(a, b, k):
    pa = [(x[k], y[k]) for x, y in zip(a, b) if x and y]
    dd = np.array([u - v for u, v in pa]); nz = dd[np.abs(dd) > 1e-12]
    w, l = int((nz > 0).sum()), int((nz < 0).sum())
    p = wilcoxon(nz, alternative="greater").pvalue if len(nz) >= 1 else 1.0
    return w, l, p


sizes = sorted({int(k) for r in d for k in r["sizes"]})
ready = []
for K in sizes:
    xs = [r["sizes"][str(K)] for r in d if str(K) in r["sizes"]]
    ns = sorted({r["n"] for x in xs for r in x["runs"]})
    taus = sorted({r["tau"] for x in xs for r in x["runs"]})
    print(f"\n==== {K} 辆车（变量中位 {int(np.median([x['m'] for x in xs]))} 个）====")
    for n in ns:
        print(f"-- {n} 个比特 --")
        print(f"{'方法':10s}{'τ':>6s}{'两比特门':>8s}{'最小间隔':>9s}{'达标比例':>9s}{'无噪声质量':>11s}{'含噪质量':>9s}{'完全一致':>9s}{'翻转比例':>9s}")
        for meth, nm in (("Q", "纠缠线路"), ("P", "去掉纠缠门")):
            for tau in taus:
                t = [v for v in table(xs, meth, n, tau) if v]
                if not t: continue
                g = lambda k: np.mean([v[k] for v in t])
                print(f"{nm:10s}{tau:6.2f}{t[0]['ncz']:8d}{np.median([v['minm'] for v in t]):9.3f}{g('fge'):9.2f}"
                      f"{g('clean'):11.3f}{g('noisy'):9.3f}{g('exact'):9.2f}{g('flip'):9.3f}")
                if meth == "Q" and g("exact") >= 0.8 and g("noisy") >= g("clean") - 0.05:
                    ready.append((K, n, tau, g("noisy"), g("exact")))
        q0, q1, p1 = table(xs, "Q", n, 0.0), table(xs, "Q", n, 0.1), table(xs, "P", n, 0.1)
        w, l, p = paired(q1, q0, "noisy"); print(f"  H1 纠缠线路 τ=0.1 对 τ=0（含噪质量）：{w}胜/{l}负 p={p:.3g}")
        w, l, p = paired(q1, p1, "noisy"); print(f"  H2 纠缠线路对去掉纠缠门（τ=0.1，含噪质量）：{w}胜/{l}负 p={p:.3g}")
        for key, nm in (("0.977_0", "不校准读出"), ("0.99_1", "两比特门保真度 0.99")):
            a = [v for v in table(xs, "Q", n, 0.1, key) if v]
            if a: print(f"  参考（纠缠线路 τ=0.1，{nm}）：含噪质量 {np.mean([v['noisy'] for v in a]):.3f}，完全一致 {np.mean([v['exact'] for v in a]):.2f}")
print("\n真机就绪的组合（规模, 比特数, τ, 含噪质量, 完全一致比例）：")
for r in ready: print(f"  {r[0]} 辆车, {r[1]} 比特, τ={r[2]}, {r[3]:.3f}, {r[4]:.2f}")
if not ready: print("  无")
