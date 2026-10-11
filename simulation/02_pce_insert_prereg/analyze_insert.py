"""按预登记汇总。用法：python3 analyze_insert.py ins_xxx.json"""
import sys, json
import numpy as np
from scipy.stats import spearmanr, wilcoxon
rows = sorted(json.load(open(sys.argv[1])), key=lambda r: r["seed"])
K = rows[0]["K"]; OPT = [r for r in rows if r["enum"]["opt_route"] is not None]
print(f"K={K}，实例 {len(rows)}（有可行最优解 {len(OPT)}），变量数中位 {np.median([r['m'] for r in rows]):.0f}，"
      f"裁剪候选空位的实例 {sum(r['trimmed'] for r in rows)}，枚举组合数中位 {np.median([r['enum']['n'] for r in rows]):.0f}")
ARMS = ["Q", "P", "C1"]; NAME = {"Q": "有纠缠线路", "P": "无纠缠门线路", "C1": "经典直接调变量", "随机": "随机插入",
                                  "后悔值插入": "后悔值插入", "贪心插入": "贪心插入"}

def gap(r, x):  # 路程与枚举最优之差，占起点总路程 %
    return (x["route"] - r["enum"]["opt_route"]) / r["start"] * 100

# ---------- 缺口 1 ----------
print("\n【缺口 1】软分数与硬解码分数的秩相关（每实例 5 次运行；方向均为越小越好）")
def sp(xs, ys):
    if len(set(np.round(xs, 12))) < 2 or len(set(np.round(ys, 12))) < 2: return None
    return spearmanr(xs, ys).correlation
M1 = {}
for arms, lbl in [(["Q"], "有纠缠线路"), (["P"], "无纠缠门线路"), (["C1"], "经典直接调变量"), (ARMS, "三组合并")]:
    cs = []; diff = []
    for r in rows:
        xs = [x["soft"] for a in arms for x in r["res"][a]]; ys = [x["hard"] for a in arms for x in r["res"][a]]
        c = sp(xs, ys); diff += [abs(a - b) for a, b in zip(xs, ys)]
        if c is not None: cs.append(c)
    cs = np.array(cs); npos = int((cs > 0).sum())
    ok = len(cs) > 0 and np.median(cs) >= 0.7 and npos >= 2 / 3 * len(cs)
    M1[lbl] = ok
    print(f"  {lbl:<10} 可算 {len(cs):2d}/{len(rows)}  中位 {np.median(cs) if len(cs) else float('nan'):+.2f}  正相关 {npos}/{len(cs)}  "
          f"|软−硬| 中位 {np.median(diff):.4f}  {'（达到 M1 门槛）' if ok else ''}")

# ---------- 缺口 2 ----------
print("\n【缺口 2】离枚举最优有多远（只计无超载的解码方案；占起点总路程 %）")
best = {}; single = {}; hit = {}
allgap = {r["seed"]: [] for r in OPT}
for a in ARMS + ["随机", "后悔值插入", "贪心插入"]:
    for r in OPT:
        allgap[r["seed"]] += [gap(r, x) for x in r["res"][a] if x["feasible"]]
worst = {s: (max(v) if v else 0.0) for s, v in allgap.items()}
for a in ARMS + ["随机", "后悔值插入", "贪心插入"]:
    b, sg, h, inf = [], [], [], 0
    for r in OPT:
        xs = r["res"][a]; f = [gap(r, x) for x in xs if x["feasible"]]; inf += sum(not x["feasible"] for x in xs)
        bb = min(f) if f else worst[r["seed"]]; b.append(bb); h.append(bb < 1e-6)
        sg.append(np.mean([gap(r, x) if x["feasible"] else worst[r["seed"]] for x in xs]))
    best[a], single[a], hit[a] = np.array(b), np.array(sg), np.array(h)
    nrun = sum(len(r["res"][a]) for r in OPT)
    print(f"  {NAME[a]:<10} 取最好命中最优 {np.mean(h):.2f}  取最好差距中位 {np.median(b):.3f}%  单次平均差距 {np.mean(sg):.3f}%  解码超载 {inf}/{nrun}")
m2 = np.mean(hit["Q"]) >= 0.5 or np.median(best["Q"]) <= 0.1
print(f"  门槛 M2（有纠缠线路命中 ≥50% 或差距中位 ≤0.1%）：{'通过' if m2 else '不通过'}")
def wl(a, b):
    d = np.array(a) - np.array(b); return int((d < -1e-9).sum()), int((np.abs(d) <= 1e-9).sum()), int((d > 1e-9).sum())
print("  逐实例：有纠缠线路取最好 对 各组（胜/平/负，越小越好）：" + "；".join(f"{NAME[a]} {wl(best['Q'], best[a])}" for a in ["P", "C1", "随机", "后悔值插入", "贪心插入"]))

# ---------- 其他记录 ----------
print("\n【其他记录】")
tri = sum(x.get("triple", 0) > 0 for r in rows for a in ARMS for x in r["res"][a])
print(f"  解码中同一空位 3 人及以上：{tri} 次")
for a in ARMS + ["随机", "后悔值插入", "贪心插入"]:
    L = [x["light"] for r in rows for x in r["res"][a] if x.get("light") is not None]
    F = [x["full"] for r in rows for x in r["res"][a] if x.get("full") is not None]
    print(f"  {NAME[a]:<10} 轻处理后缩短平均 {np.mean(L):+.3f}%  完整精修后 {np.mean(F):+.3f}%")
lo = [r["enum"]["opt_lf"]["light"] for r in OPT]; fo = [r["enum"]["opt_lf"]["full"] for r in OPT]
print(f"  枚举最优    轻处理后缩短平均 {np.mean(lo):+.3f}%  完整精修后 {np.mean(fo):+.3f}%")
for a in ARMS:
    xs = [x for r in rows for x in r["res"][a]]
    print(f"  计算预算 {NAME[a]:<10} 参数 {np.mean([x['npar'] for x in xs]):.0f}  目标函数计算次数 {np.mean([x['nev'] for x in xs]):.0f}  "
          f"每次运行秒数 {np.mean([x['sec'] for x in xs]):.1f}  第一名领先第二名最小值中位 {np.median([x['min_lead'] for x in xs]):.3f}")

# ---------- 表达诊断 ----------
if "expr" in rows[0]:
    print("\n【表达能力诊断】以枚举最优为目标复现最大值位置（最多 10 个起点，每个 3000 步）")
    for k in ("Q", "P"):
        E = [r["expr"][k] for r in rows if "expr" in r]
        ok = [any(x["ok"] for x in e) for e in E]; okm = [any(x["ok"] and x["lead"] >= 0.1 - 1e-9 for x in e) for e in E]
        ntry = [next((i + 1 for i, x in enumerate(e) if x["ok"]), None) for e in E]
        print(f"  {NAME[k]:<10} 复现成功 {np.mean(ok):.2f}  且领先 ≥τ {np.mean(okm):.2f}  成功所需起点数中位 {np.median([t for t in ntry if t]) if any(ntry) else '—'}")

# ---------- 结论规则 ----------
print("\n【按预登记的结论】")
ceil = np.mean(hit["Q"]) >= 0.8 and np.mean(hit["C1"]) >= 0.8
if ceil: print("  天花板：有纠缠线路和经典直接调变量都在 ≥80% 的实例上命中最优，增量在本规模上无法识别（不能推出机制无效或无增量）。")
q_ok = M1["有纠缠线路"] and m2
c1_m2 = np.mean(hit["C1"]) >= 0.5 or np.median(best["C1"]) <= 0.1
p_m2 = np.mean(hit["P"]) >= 0.5 or np.median(best["P"]) <= 0.1
if q_ok: print("  M1、M2 都通过：机制可行" + ("，进入第二阶段。" if K == 2 else "。"))
if not M1["有纠缠线路"] and not M1["经典直接调变量"]: print("  有纠缠线路与经典直接调变量都未达 M1：当前“连续放宽 + 最大值解码”组合有问题。")
if c1_m2 and not m2: print("  经典直接调变量达到 M2、有纠缠线路未达到：优先检查线路的表达、优化方法与参数化（不能推出纠缠无用）。")
if m2 and not p_m2: print("  有纠缠线路达到 M2、无纠缠门线路未达到：纠缠对所测线路族有帮助（不构成量子优于经典）。")
if m2 and np.mean(best["Q"] <= best["后悔值插入"] + 1e-9) < 1 and wl(best["Q"], best["后悔值插入"])[0] == 0:
    print("  有纠缠线路未胜过后悔值插入：机制可行性另论，尚无实际竞争优势。")
if K >= 3:
    print("\n【第二阶段增量检验】（单侧符号秩检验，越小越好）")
    for b in ("C1", "P"):
        for nm, arr in (("取最好差距", best), ("单次平均差距", single)):
            d = arr["Q"] - arr[b]; dz = d[np.abs(d) > 1e-12]
            p = wilcoxon(dz, alternative="less").pvalue if len(dz) else 1.0
            print(f"  有纠缠线路 对 {NAME[b]}，{nm}：{wl(arr['Q'], arr[b])}，p={p:.3g}{'  通过' if p < 0.05 else ''}")
