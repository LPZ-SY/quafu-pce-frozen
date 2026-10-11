import numpy as np, json
exec(open('diag_hw.py').read().split("ehw = np.concatenate")[0])
# 错误位置
for sd, A, slots, eid, ehw in data:
    c = np.array(A["anchor_sign"]); wrong = [slots[k] for k in range(len(slots)) if (c[k]*eid[k]<0)!=(c[k]*ehw[k]<0)]
    print(sd, "错的变量对应的比特对:", wrong)
allD = list(data); out = []
for i in range(len(allD)):
    data[:] = [d for j, d in enumerate(allD) if j != i]
    best = None
    for st in range(3):
        x0 = np.r_[0.8, np.random.default_rng(st).normal(0, 0.2, 7)]
        r = minimize(loss, x0, args=("edge_phase",), method="Powell", options=dict(maxiter=4000, xtol=1e-3, ftol=1e-6))
        if best is None or r.fun < best.fun: best = r
    data[:] = [allD[i]]
    p = predict(best.x, "edge_phase")[0]; p0 = predict([best.x[0]], "depol")[0]
    sd, A, slots, eid, ehw = allD[i]; c = np.array(A["anchor_sign"])
    sst = ((ehw - ehw.mean())**2).sum()
    out.append(dict(seed=sd, edge_phase=best.x.round(2).tolist(), R2_holdout=round(1-((p-ehw)**2).sum()/sst,3),
                    R2_depol_holdout=round(1-((p0-ehw)**2).sum()/sst,3), sign_match=round(float(((c*p<0)==(c*ehw<0)).mean()),3)))
    print(out[-1], flush=True)
    data[:] = allD
json.dump(out, open('/home/claude/pce8/run/analysis/error_model_loo.json','w'), indent=1)
