"""汇总 pce_scale 的结果。用法：python3 analyze_pce.py pce_scale_main.json"""
import sys, json
import numpy as np
from scipy.stats import wilcoxon

rows = json.load(open(sys.argv[1]))
METH = ["Q", "P", "C1", "C2", "C0"]
NAME = {"Q": "量子（纠缠线路）", "P": "去掉纠缠门", "C1": "经典直接调变量", "C2": "经典模拟退火", "C0": "随机换车"}


def best(runs, key):
    v = [r[key] for r in runs if r.get("feasible") and key in r]
    return max(v) if v else np.nan


def cnt(runs, key):
    return sum(1 for r in runs if r.get("feasible") and r.get(key, -1) > 1e-9)


def minproxy(runs):
    return min(r["proxy_disc"] for r in runs)


def test(a, b):
    d = np.array(a) - np.array(b); d = d[~np.isnan(d)]
    w, l = int((d > 1e-9).sum()), int((d < -1e-9).sum())
    if w + l == 0: return w, l, 1.0
    p = wilcoxon(d[np.abs(d) > 1e-9], alternative="greater").pvalue if w + l >= 1 else 1.0
    return w, l, p


print(f"实例数：{len(rows)}；每种方法每个实例 {len(rows[0]['sizes'][next(iter(rows[0]['sizes']))]['res']['Q'])} 次随机起点")
for K in sorted({int(k) for r in rows for k in r["sizes"]}):
    sub = [r["sizes"][str(K)] if str(K) in r["sizes"] else r["sizes"].get(K) for r in rows]
    sub = [s for s in sub if s]
    print(f"\n==== {K} 辆车：客户（变量）{np.mean([s['m'] for s in sub]):.0f} 个，比特 {sub[0]['n']} 个，"
          f"量子线路两比特门 {sub[0]['res']['Q'][0].get('ncz')} 个 ====")
    tab = {}
    print(f"{'方法':<14}{'调参目标最好值':>12}{'更短次数(精修前)':>16}{'更短次数(精修后)':>16}{'最好缩短%(精修前)':>18}{'最好缩短%(精修后)':>18}{'超载车(修正前)':>14}")
    for m in METH:
        R = [s["res"][m] for s in sub]
        tab[m] = dict(px=[minproxy(x) for x in R], bb=[best(x, "impr_before") for x in R],
                      ba=[best(x, "impr_after") for x in R], cb=[cnt(x, "impr_before") for x in R],
                      ca=[cnt(x, "impr_after") for x in R], ov=[np.mean([r["over_routes"] for r in x]) for x in R])
        t = tab[m]
        print(f"{NAME[m]:<14}{np.mean(t['px']):>12.3f}{np.mean(t['cb']):>16.2f}{np.mean(t['ca']):>16.2f}"
              f"{np.nanmean(t['bb']):>18.3f}{np.nanmean(t['ba']):>18.3f}{np.mean(t['ov']):>14.2f}")
    print("逐实例配对比较（量子减对照；目标值越小越好，其余越大越好；单侧检验量子更好）：")
    for m in ["P", "C1", "C2", "C0"]:
        w, l, p = test([-x for x in tab["Q"]["px"]], [-x for x in tab[m]["px"]])
        w2, l2, p2 = test(tab["Q"]["ba"], tab[m]["ba"])
        w3, l3, p3 = test(tab["Q"]["ca"], tab[m]["ca"])
        print(f"  对 {NAME[m]}：调参目标 {w}胜/{l}负 p={p:.3g}；最好缩短(精修后) {w2}胜/{l2}负 p={p2:.3g}；"
              f"更短次数(精修后) {w3}胜/{l3}负 p={p3:.3g}")
