import json, sys
import numpy as np
from scipy.stats import wilcoxon
rows = json.load(open(sys.argv[1] if len(sys.argv) > 1 else "diag_anchor.json"))
arms = ["Q原", "Q对齐", "P对齐", "C1对齐", "随机同个数"]
print("实例数", len(rows), "变量数中位", np.median([r["m"] for r in rows]))
T = {}
for a in arms:
    R = [r["res"][a] for r in rows]
    nf = np.median([np.median([e.get("nflip", np.nan) for e in x]) for x in R])
    best = [max([e["impr_after"] for e in x if e.get("feasible")] or [np.nan]) for x in R]
    mean = [np.nanmean([e["impr_after"] for e in x if e.get("feasible")] or [np.nan]) for x in R]
    bef = [np.nanmean([e["impr_before"] for e in x if e.get("feasible")] or [np.nan]) for x in R]
    over = np.mean([np.mean([e["over_routes"] for e in x]) for x in R])
    gmin = [np.median([e["min_gap"] for e in x]) for x in R] if "min_gap" in R[0][0] else None
    gmed = [np.median([e["med_gap"] for e in x]) for x in R] if "med_gap" in R[0][0] else None
    T[a] = dict(best=best, mean=mean)
    s = f"{a:<8} 换车个数中位 {nf:5.1f}  精修前平均% {np.nanmean(bef):6.2f}  精修后平均% {np.nanmean(mean):5.2f}  3取最好% {np.nanmean(best):5.2f}  超载车 {over:4.2f}"
    if gmin: s += f"  最小间隔中位 {np.median(gmin):.3f}  间隔中位 {np.median(gmed):.3f}"
    print(s)


def cmp(a, b, k):
    d = np.array(T[a][k]) - np.array(T[b][k]); d = d[~np.isnan(d)]
    w, l = int((d > 1e-9).sum()), int((d < -1e-9).sum())
    p = wilcoxon(d[np.abs(d) > 1e-9], alternative="greater").pvalue if w + l else 1
    return f"{w}/{l} p={p:.3g}"


for a, b in [("Q对齐", "Q原"), ("Q对齐", "P对齐"), ("Q对齐", "C1对齐"), ("Q对齐", "随机同个数")]:
    print(f"{a} 对 {b}：精修后平均 {cmp(a, b, 'mean')}；3取最好 {cmp(a, b, 'best')}")
