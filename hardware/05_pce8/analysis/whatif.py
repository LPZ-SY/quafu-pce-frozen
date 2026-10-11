import numpy as np
exec(open('diag_hw.py').read().split("ehw = np.concatenate")[0])
par = np.array([0.837, 0.113, -0.205, 0.126, 0.217, -0.051, -0.091, 1.757])
for name, p in (("拟合值（含 6-7 边 1.76 弧度）", par), ("把 6-7 边的相位误差置 0", np.r_[par[:7], 0.0])):
    pr = predict(p, "edge_phase"); res = []
    for q, d in zip(pr, data):
        c = np.array(d[1]["anchor_sign"]); eid = d[3]
        res.append(int(((c*q<0)!=(c*eid<0)).sum()))
    print(name, "各实例与模拟不一致的变量数：", res)
