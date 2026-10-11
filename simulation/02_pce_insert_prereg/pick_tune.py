import json, numpy as np
out = []
for a in ["0.5", "1", "2", "5"]:
    rows = json.load(open(f"ins_tune_{a}.json")); fe = {}; gp = []
    for arm in ("Q", "C1"):
        xs = [(r, x) for r in rows for x in r["res"][arm]]
        fe[arm] = np.mean([x["feasible"] for _, x in xs])
        gp += [(x["route"] - r["enum"]["opt_route"]) / r["start"] * 100 for r, x in xs if x["feasible"] and r["enum"]["opt_route"] is not None]
    out.append((a, fe["Q"], fe["C1"], np.mean(gp)))
    print(f"A_CAP={a}: 无超载比例 Q {fe['Q']:.2f} C1 {fe['C1']:.2f}；无超载运行离最优差距平均 {np.mean(gp):.3f}%")
ok = [o for o in out if o[1] >= 0.9 and o[2] >= 0.9]
if ok: pick = min(ok, key=lambda o: (round(o[3], 9), float(o[0])))
else: pick = max(out, key=lambda o: min(o[1], o[2])); print("没有候选值满足无超载 ≥0.9，按规则取无超载比例最高者")
print("选定 A_CAP =", pick[0])
