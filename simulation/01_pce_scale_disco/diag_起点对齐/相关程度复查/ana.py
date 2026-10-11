import json, sys, numpy as np
from scipy.stats import spearmanr
W = sys.argv[1]
new = {r["seed"]: r for r in json.load(open(W + "/rerun.json"))}
old = {r["seed"]: r for r in json.load(open(W + "/diag_two_orig.json"))}
# 复现检查
mis = tot = 0
for s, r in new.items():
    for a, runs in r["res"].items():
        for x, y in zip(runs, old[s]["res"][a]):
            tot += 1
            if not (x.get("feasible") == y.get("feasible") and abs(x.get("impr_after", 0) - y.get("impr_after", 0)) < 1e-6): mis += 1
print(f"复现检查：{tot - mis}/{tot} 次运行与原数据一致")
ARMS = ["Q原", "Q对齐", "P对齐", "C1对齐", "随机同个数"]
def corrs(arms, xk, yk, sign=-1):
    out = []
    for s, r in sorted(new.items()):
        xs, ys = [], []
        for a in arms:
            for e in r["res"][a]:
                if e.get("feasible"): xs.append(e[xk]); ys.append(e[yk])
        if len(set(xs)) > 1 and len(set(ys)) > 1:
            out.append(sign * spearmanr(xs, ys).correlation)
    return np.array(out)
def show(lbl, arms, xk, yk):
    c = corrs(arms, xk, yk)
    print(f"  {lbl:<34} 可算实例 {len(c):2d}  中位 {np.median(c):+.2f}  平均 {c.mean():+.2f}  正相关实例 {int((c>0).sum())}/{len(c)}")
print("\n秩相关（已把符号统一成：越大＝训练分数越好的方案，最终路线也越好）")
for xk, xl in [("proxy", "训练目标（近似路程+容量罚）"), ("proxy_mu", "训练目标+换车惩罚")]:
    for yk, yl in [("impr_after", "精修后缩短"), ("impr_before", "精修前缩短")]:
        print(f"{xl} 对 {yl}：")
        show("全部方法混在一起（同旧口径）", ARMS, xk, yk)
        show("只看对齐后的方法（Q对齐/P对齐/C1/随机）", ARMS[1:], xk, yk)
print("\n近似目标对精确路程的准确度（子问题内，同一归一化）：")
show("近似路程+罚 对 精确路程变化", ARMS, "proxy", "exact_sub")
# 各方法：换车个数、改动比例、改动且变好比例
print("\n各方法概况：")
for a in ARMS:
    E = [e for r in new.values() for e in r["res"][a] if e.get("feasible")]
    nf = np.array([e["nflip"] for e in E]); ia = np.array([e["impr_after"] for e in E]); ib = np.array([e["impr_before"] for e in E])
    ch = nf > 0
    print(f"  {a:<6} 运行 {len(E)}  有改动 {ch.mean():.2f}  有改动中精修前变好 {((ib>1e-9)&ch).sum()}/{ch.sum()}  精修后变好 {((ia>1e-9)&ch).sum()}/{ch.sum()}  精修后变差 {((ia<-1e-9)&ch).sum()}/{ch.sum()}")
# 只在有改动的运行上，跨实例合并：训练分数 与 精修后缩短
for a in ["Q对齐", "C1对齐", "Q原"]:
    E = [e for r in new.values() for e in r["res"][a] if e.get("feasible") and e["nflip"] > 0]
    if len(E) > 3:
        rho = spearmanr([-e["proxy"] for e in E], [e["impr_after"] for e in E]).correlation
        rho2 = spearmanr([-e["proxy"] for e in E], [e["impr_before"] for e in E]).correlation
        print(f"  {a} 有改动的运行合并（{len(E)} 次）：训练分数对精修后 {rho:+.2f}，对精修前 {rho2:+.2f}")
