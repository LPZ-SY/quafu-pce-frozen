import json, numpy as np
R = sorted(json.load(open("diag_train.json")), key=lambda r: r["seed"])
print("实例", len(R))
A = [a for r in R for a in r["A"]]
print(f"A 从最优出发：可复现 {len(A)}/{len(R)}；按 β=12 训练后仍在最优 {sum(a['stay_beta12'] for a in A)}/{len(A)}；按 β=48 仍在最优 {sum(a['stay_beta48'] for a in A)}/{len(A)}")
for k, nm in [("fixed", "固定 β=12（正式实验做法）"), ("multi", "多阶段 β 1.5→48"), ("fixed_long", "固定 β=48，同等步数")]:
    g = [x for r in R for x in r[k]]; hit = [x is not None and x < 1e-6 for x in g]
    best = [min([x for x in r[k] if x is not None], default=None) for r in R]
    bh = [b is not None and b < 1e-6 for b in best]
    feas = [x for x in g if x is not None]
    print(f"{nm:<22} 单次命中 {np.mean(hit):.2f}  5 次取最好命中 {np.mean(bh):.2f}  单次平均差距 {np.mean(feas):.3f}%  超载 {sum(x is None for x in g)}/{len(g)}")
