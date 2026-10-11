import json, sys, numpy as np
from scipy.stats import spearmanr
sys.path.insert(0, sys.argv[1]); import pce_scale as S
rows = json.load(open(sys.argv[1] + "/rerun.json"))
ARMS = ["Q原", "Q对齐", "P对齐", "C1对齐", "随机同个数"]
c1, c2, c3, n3 = [], [], [], []
for r in rows:
    P, q, D, R = S.instance(r["seed"]); grp = S.pick_groups(P, R)[3]; sub = S.build_sub(P, q, D, R, grp)
    xs, ys, ps, ia, ch = [], [], [], [], []
    for a in ARMS:
        for e in r["res"][a]:
            u = np.array(list(e["flip"]), float)
            xs.append(sub["lin"] @ u + 0.5 * u @ sub["J"] @ u); ys.append(e["exact_sub"])
            if e.get("feasible"): ps.append(e["proxy_mu"]); ia.append(e["impr_after"]); ch.append(e["nflip"] > 0)
    if len(set(np.round(xs, 9))) > 2: c1.append(spearmanr(xs, ys).correlation)
    ps, ia, ch = np.array(ps), np.array(ia), np.array(ch)
    if ch.sum() > 3 and len(set(ia[ch])) > 1: c3.append(-spearmanr(ps[ch], ia[ch]).correlation); n3.append(ch.sum())
c1, c3 = np.array(c1), np.array(c3)
print(f"实例数 {len(rows)}")
print(f"近似路程（不含罚）对 精确路程变化：可算 {len(c1)} 个实例，秩相关中位 {np.median(c1):+.2f}，平均 {c1.mean():+.2f}，正相关 {(c1>0).sum()}/{len(c1)}")
print(f"只看有改动的运行（排除“不改动”）：训练分数 对 精修后缩短：可算 {len(c3)} 个实例，中位 {np.median(c3):+.2f}，平均 {c3.mean():+.2f}，正相关 {(c3>0).sum()}/{len(c3)}")
