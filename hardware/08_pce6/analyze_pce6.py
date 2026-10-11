"""pce6 分析（在 Claude 的环境里运行）。用法：python3 analyze_pce6.py <包目录> [结果文件夹名，默认 live_pce6]"""
import sys, os, json
from math import comb
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import pce_scale as S
PKG = sys.argv[1]; LV = sys.argv[2] if len(sys.argv) > 2 else "live_pce6"
MAN = json.load(open(os.path.join(PKG, "task_manifest.json"), encoding="utf-8"))
LED = json.load(open(os.path.join(PKG, LV, "ledger.json"), encoding="utf-8"))
N = MAN["n_qubits"]; ORDER = LED.get("_bit_order", "c_last_first")
lines = []
def say(*a):
    s = " ".join(str(x) for x in a); print(s); lines.append(s)

def bits_of(order):
    f = os.path.join(PKG, LV, "raw", f"{order:03d}.json")
    if not os.path.exists(f): return None
    r = json.load(open(f, encoding="utf-8")); c = r.get("count") or r.get("counts") or {}
    rows = []
    for k, v in c.items():
        k = str(k).replace(" ", "")
        rows.append(([int(k[N - 1 - i]) for i in range(N)] if ORDER == "c_last_first" else [int(k[i]) for i in range(N)], int(v)))
    return np.repeat(np.array([b for b, _ in rows]), [v for _, v in rows], axis=0)

cal = {0: [], 1: []}
for t in MAN["tasks"]:
    if t["kind"] == "calibration":
        b = bits_of(t["order"])
        if b is not None: cal[t["bits_c0_first"][0]].append((b != t["bits_c0_first"][0]).mean(0))
p01, p10 = np.mean(cal[0], 0), np.mean(cal[1], 0)
say("读出误差 读1|真0：", np.round(p01, 3), " 读0|真1：", np.round(p10, 3))
Minv = [np.linalg.inv(np.array([[1 - p01[q], p10[q]], [p01[q], 1 - p10[q]]])) for q in range(N)]

def corr(bits, a, b):
    pm = np.array([np.mean((bits[:, a] == x) & (bits[:, b] == y)) for x in (0, 1) for y in (0, 1)])
    pm = np.kron(Minv[a], Minv[b]) @ pm
    return pm[0] - pm[1] - pm[2] + pm[3]

def sign_test(a, b):
    w = sum(x > y + 1e-9 for x, y in zip(a, b)); l = sum(x < y - 1e-9 for x, y in zip(a, b)); n = w + l
    return w, l, (sum(comb(n, k) for k in range(w, n + 1)) / 2 ** n if n else 1.0)

res = []
for inst in MAN["instances"]:
    sd = inst["seed"]; P_, q, Dm, R = S.instance(sd); start = S.total(Dm, R); grp = S.pick_groups(P_, R)[3]
    sub = S.build_sub(P_, q, Dm, R, grp); assert sub["m"] == inst["m"]
    row = dict(seed=sd, m=inst["m"])
    for arm in ("Q", "P"):
        A = inst["arms"][arm]; slots = [tuple(s) for s in A["slots"]]; c = np.array(A["anchor_sign"])
        e_id = np.array(A["e_ideal"]); fl_id = np.array(A["flip_ideal"]); e = np.full(len(slots), np.nan)
        for f in "ZXY":
            ts = [t for t in MAN["tasks"] if t["kind"] == "algorithm" and t["instance"] == sd and t["arm"] == arm and t["family"] == f]
            bl = [bits_of(t["order"]) for t in ts]; bl = [b for b in bl if b is not None]
            if not bl: continue
            bits = np.vstack(bl)
            for k, s in enumerate(slots):
                if s[0] == f: e[k] = corr(bits, s[1], s[2])
        if np.isnan(e).any(): row[arm] = None; continue
        fl = (c * e < 0).astype(int); ev = S.evaluate(P_, q, Dm, R, grp, sub, fl, start)
        score = max(ev.get("impr_after") or 0.0, 0.0) if ev["feasible"] else 0.0
        rnd_scores = []
        if arm == "Q":       # 对照：随机改同样个数的客户（每题 30 次，取平均），检查有纠缠的收益是否只是"随便改一改"带来的
            rr = np.random.default_rng(sd)
            for _ in range(30):
                fr = np.zeros(len(fl), int); fr[rr.choice(len(fl), int(fl.sum()), replace=False)] = 1
                evr = S.evaluate(P_, q, Dm, R, grp, sub, fr, start)
                rnd_scores.append(max(evr.get("impr_after") or 0.0, 0.0) if evr["feasible"] else 0.0)
        row[arm] = dict(score_hw=score, score_random=float(np.mean(rnd_scores)) if rnd_scores else None, impr_raw=ev.get("impr_after"), feasible=ev["feasible"], nflip=int(fl.sum()),
                        score_ideal=A["score_ideal"], nflip_ideal=int(fl_id.sum()), exact=bool((fl == fl_id).all()),
                        n_wrong=int((fl != fl_id).sum()), retention=float((e * e_id).sum() / (e_id ** 2).sum()),
                        pred=A["prediction"])
    res.append(row)
    say(f"实例 {sd}（{inst['m']} 变量）：" + "；".join(
        f"{arm} 真机 {row[arm]['score_hw']:.3f}%（模拟 {row[arm]['score_ideal']:.3f}%，换车 {row[arm]['nflip']}/{row[arm]['nflip_ideal']}，读错 {row[arm]['n_wrong']}）"
        if row.get(arm) else f"{arm} 缺数据" for arm in ("Q", "P")))
ok = [r for r in res if r.get("Q") and r.get("P")]
qh = [r["Q"]["score_hw"] for r in ok]; ph = [r["P"]["score_hw"] for r in ok]
qi = [r["Q"]["score_ideal"] for r in ok]; pi = [r["P"]["score_ideal"] for r in ok]
say(f"\n==== 主判据（真机，{len(ok)} 题）====")
w, l, p = sign_test(qh, ph); say(f"有纠缠 对 无纠缠（路线缩短，比原方案差按 0 计）：{w} 胜 {l} 负 {len(ok) - w - l} 平，单侧符号检验 p = {p:.4f}")
say(f"平均缩短：有纠缠 {np.mean(qh):.3f}%，无纠缠 {np.mean(ph):.3f}%")
w2, l2, p2 = sign_test(qi, pi); say(f"（参考）模拟中：{w2} 胜 {l2} 负，p = {p2:.4f}；平均 {np.mean(qi):.3f}% 对 {np.mean(pi):.3f}%")
rq = [r["Q"]["score_random"] for r in ok]; w3, l3, p3 = sign_test(qh, rq)
say(f"（对照）有纠缠 对 随机改同样个数：{w3} 胜 {l3} 负，p = {p3:.4f}；随机平均 {np.mean(rq):.3f}%")
say("\n==== 辅助指标 ====")
for arm in ("Q", "P"):
    say(f"{arm}：与模拟完全一致 {sum(r[arm]['exact'] for r in ok)}/{len(ok)}；平均读错 {np.mean([r[arm]['n_wrong'] for r in ok]):.2f} 个；"
        f"信号保留 {np.mean([r[arm]['retention'] for r in ok]):.2f}；真机缩短 / 模拟缩短 = {np.sum([r[arm]['score_hw'] for r in ok]):.3f} / {np.sum([r[arm]['score_ideal'] for r in ok]):.3f}")
os.makedirs(os.path.join(PKG, "analysis"), exist_ok=True)
json.dump(dict(readout=dict(p01=p01.tolist(), p10=p10.tolist()), layout=LED.get("_layout"), results=res),
          open(os.path.join(PKG, "analysis", "analysis_pce6.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=float)
open(os.path.join(PKG, "analysis", "analysis_pce6.txt"), "w", encoding="utf-8").write("\n".join(lines) + "\n")
