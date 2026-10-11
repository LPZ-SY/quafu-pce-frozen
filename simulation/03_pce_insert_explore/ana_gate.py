import json, sys, numpy as np
R = sorted(json.load(open(sys.argv[1])), key=lambda r: r["seed"]); n = len(R)
print(f"K={R[0]['K']} 实例 {n}，变量数中位 {np.median([r['m'] for r in R]):.0f}，三车装载率中位 {np.median([r['fill'] for r in R]):.2f}")
lo = np.array([r["opt_light"] for r in R]); fo = np.array([r["opt_full"] for r in R])
print(f"最好插法经轻处理后比原方案短的题：{(lo > 1e-6).sum()}/{n}（缩短中位 {np.median(lo[lo>1e-6]) if (lo>1e-6).any() else 0:.3f}%）；完整精修后：{(fo > 1e-6).sum()}/{n}")
print(f"最好插法中至少一个客户换了车：{sum(r['moved']>0 for r in R)}/{n}")
for nm in ("后悔值", "贪心"):
    miss = [r for r in R if not (r[nm]["feasible"] and r[nm]["gap"] < 1e-6)]
    print(f"{nm}插入没找到最好插法：{len(miss)}/{n}；其中最好插法比原方案短的：{sum(r['opt_light']>1e-6 for r in miss)}")
both = [r for r in R if r["opt_light"] > 1e-6 and not (r["后悔值"]["feasible"] and r["后悔值"]["gap"] < 1e-6)]
print(f"同时满足“最好插法能变短”且“后悔值插入找不到”的题：{len(both)}/{n}")
