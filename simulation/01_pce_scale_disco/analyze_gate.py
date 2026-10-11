"""第 0 步闸门的分析：按预登记的三条通过条件逐个规模判定。"""
import sys, json
import numpy as np

THR = 0.01
d = json.load(open(sys.argv[1]))
print(f"实例数：{len(d)}")
g = [r["ils_gain_pct"] for r in d]
print(f"强局部最优比“搬一个＋换一个”收敛方案再短：中位 {np.median(g):.2f}%（范围 {min(g):.2f}–{max(g):.2f}%）")
sizes = sorted({int(k) for r in d for k in r["sizes"]})
for K in sizes:
    rows = [r["sizes"][str(K)] for r in d if str(K) in r["sizes"]]
    n = len(rows)
    print(f"\n==== {K} 辆车（{n} 个子问题，变量中位 {int(np.median([x['m'] for x in rows]))} 个）====")
    has, mink, best = 0, [], []
    rand_hits = {k: [0, 0] for k in range(1, 9)}
    prox_has = 0; rand_has = 0
    for x in rows:
        imp = [c for c in x["rand"] + x["proxy"] if c["impr"] is not None and c["impr"] > THR]
        for c in x["rand"]:
            rand_hits[c["k"]][1] += 1
            if c["impr"] is not None and c["impr"] > THR: rand_hits[c["k"]][0] += 1
        if any(c["impr"] is not None and c["impr"] > THR for c in x["proxy"]): prox_has += 1
        if any(c["impr"] is not None and c["impr"] > THR for c in x["rand"]): rand_has += 1
        if imp:
            has += 1; mink.append(min(c["k"] for c in imp)); best.append(max(c["impr"] for c in imp))
    print(f"有改进的子问题：{has}/{n}（随机找到 {rand_has}，近似路程引导找到 {prox_has}）")
    if best: print(f"  最好改进幅度：中位 {np.median(best):.3f}%，最大 {max(best):.3f}%")
    if mink: print(f"  每个子问题“最少要换几个客户才有改进”：{sorted(mink)}")
    print("随机挑客户换，每试一次碰上改进的比例（按换车客户数）：")
    tot_h = tot_n = 0
    for k in range(1, 9):
        h, t = rand_hits[k]; tot_h += h; tot_n += t
        if t: print(f"  换 {k} 个：{h}/{t} = {h / t:.4f}" + (f"（约每 {t / h:.0f} 次碰上一次）" if h else ""))
    p = tot_h / tot_n if tot_n else 0
    c1 = has / n >= 0.2
    c2 = bool(mink) and np.mean(np.array(mink) >= 3) > 0.5
    c3 = (p == 0) or (1 / p >= 200)
    print("预登记通过条件：")
    print(f"  1. 有改进的子问题 ≥ 20%：{has / n:.0%} → {'通过' if c1 else '不通过'}")
    print(f"  2. 改进中最少要换 ≥3 个客户的占一半以上：" +
          (f"{np.mean(np.array(mink) >= 3):.0%}" if mink else "无改进") + f" → {'通过' if c2 else '不通过'}")
    print(f"  3. 随机平均 ≥200 次才碰上一次：" + (f"约每 {1 / p:.0f} 次" if p else "一次都没碰上") +
          f" → {'通过' if c3 else '不通过'}")
    print(f"  {K} 辆车总判定：{'通过，进入第 1 步' if (c1 and c2 and c3) else '不通过'}")
