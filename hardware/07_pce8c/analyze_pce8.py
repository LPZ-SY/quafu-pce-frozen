"""泡利关联编码真机展示 pce8：分析（在 Claude 的环境里运行，需要 pce_scale.py、scale_pairs.py）
用法：python3 analyze_pce8.py <包目录>     # 包目录里有 task_manifest.json 和 live/
输出：<包目录>/analysis/analysis_pce8.txt 和 .json"""
import sys, os, json, glob
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import pce_scale as S

PKG = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "pkg")
MAN = json.load(open(os.path.join(PKG, "task_manifest.json"), encoding="utf-8"))
LED = json.load(open(os.path.join(PKG, "live", "ledger.json"), encoding="utf-8"))
N = MAN["n_qubits"]; ORDER = LED.get("_bit_order", "c7…c0")
os.makedirs(os.path.join(PKG, "analysis"), exist_ok=True)
lines = []
def say(*a):
    s = " ".join(str(x) for x in a); print(s); lines.append(s)


def counts(order):
    f = os.path.join(PKG, "live", "raw", f"{order:03d}.json")
    if not os.path.exists(f): return None
    r = json.load(open(f, encoding="utf-8")); c = r.get("count") or r.get("counts") or {}
    out = np.zeros((0, N), int); rows = []
    for k, v in c.items():
        k = str(k).replace(" ", "")
        b = [int(k[N - 1 - i]) for i in range(N)] if ORDER == "c7…c0" else [int(k[i]) for i in range(N)]
        rows.append((b, int(v)))
    return rows


def to_bits(rows):
    return np.repeat(np.array([b for b, _ in rows]), [v for _, v in rows], axis=0)


T = {t["order"]: t for t in MAN["tasks"]}
# ---------- 读出标定 ----------
cal = {}
for t in MAN["tasks"]:
    if t["kind"] != "calibration": continue
    rows = counts(t["order"])
    if rows is None: continue
    bits = to_bits(rows); b = t["bits_c0_to_c7"][0]
    err = (bits != b).mean(0)                         # 每个比特读错的比例
    cal.setdefault(b, []).append(err)
p01 = np.mean(cal[0], 0); p10 = np.mean(cal[1], 0)
say("读出误差 P(读1|真0)：", np.round(p01, 3)); say("读出误差 P(读0|真1)：", np.round(p10, 3))
if len(cal[0]) > 1: say("首尾漂移（最大绝对差）：0→1", np.round(np.abs(cal[0][0] - cal[0][-1]).max(), 3), " 1→0", np.round(np.abs(cal[1][0] - cal[1][-1]).max(), 3))
Minv = [np.linalg.inv(np.array([[1 - p01[q], p10[q]], [p01[q], 1 - p10[q]]])) for q in range(N)]


def corr(bits, a, b, mitigate):
    pm = np.array([np.mean((bits[:, a] == x) & (bits[:, b] == y)) for x in (0, 1) for y in (0, 1)])
    if mitigate: pm = np.kron(Minv[a], Minv[b]) @ pm
    return pm[0] - pm[1] - pm[2] + pm[3]


BITS = ((np.arange(1 << N)[:, None] >> (N - 1 - np.arange(N))) & 1)
res = []
for inst in MAN["instances"]:
    sd = inst["seed"]
    P_, q, Dm, R = S.instance(sd); start = S.total(Dm, R); grp = S.pick_groups(P_, R)[3]
    sub = S.build_sub(P_, q, Dm, R, grp)
    assert sub["m"] == inst["m"] and sub["cust"].tolist() == inst["customers"]
    row = dict(seed=sd, m=inst["m"], arms={})
    for arm in ("Q", "P"):
        A = inst["arms"][arm]; slots = [tuple(s) for s in A["slots"]]; c = np.array(A["anchor_sign"])
        e_id = np.array(A["e_ideal"]); fl_id = np.array(A["flip_ideal"])
        out = {}
        for mit in (False, True):
            e = np.zeros(len(slots)); tvd = {}
            per_round = []
            for f in "ZXY":
                ts = [t for t in MAN["tasks"] if t["kind"] == "algorithm" and t["instance"] == sd and t["arm"] == arm and t["family"] == f]
                rows = [counts(t["order"]) for t in ts]; rows = [r for r in rows if r]
                if not rows: continue
                bits = np.vstack([to_bits(r) for r in rows])
                idx = [k for k, s in enumerate(slots) if s[0] == f]
                for k in idx: e[k] = corr(bits, slots[k][1], slots[k][2], mit)
                p_hw = np.zeros(1 << N); np.add.at(p_hw, bits @ (1 << (N - 1 - np.arange(N))), 1); p_hw /= len(bits)
                p_id = np.load(os.path.join(PKG, "ideal", f"s{sd}_{arm}_{f}.npy"))
                tvd[f] = float(0.5 * np.abs(p_hw - p_id).sum())
            fl = (c * e < 0).astype(int)
            ev = S.evaluate(P_, q, Dm, R, grp, sub, fl, start)
            keep = np.abs(e_id) > 0
            out["mitigated" if mit else "raw"] = dict(
                sign_agreement=float((fl == fl_id).mean()), n_disagree=int((fl != fl_id).sum()), exact=bool((fl == fl_id).all()),
                retention=float((e * e_id).sum() / (e_id ** 2).sum()), nflip=int(fl.sum()), tvd=tvd,
                impr_after=ev.get("impr_after"), feasible=ev["feasible"], min_abs_hw=float(np.abs(e).min()))
        out["ideal"] = dict(nflip=int(fl_id.sum()), impr_after=A["eval_ideal"].get("impr_after"), min_gap=A["min_gap"], ncz=A["ncz"])
        out["prediction"] = A["prediction"]
        row["arms"][arm] = out
        say(f"\n实例 {sd}（{inst['m']} 个变量）{arm}：CZ {A['ncz']}，模拟换车 {out['ideal']['nflip']}、精修后缩短 {out['ideal']['impr_after']:.3f}%，最小间隔 {A['min_gap']:.3f}")
        for k in ("raw", "mitigated"):
            o = out[k]
            say(f"  真机（{'校准后' if k == 'mitigated' else '未校准'}）：正负号一致 {o['sign_agreement']:.3f}（错 {o['n_disagree']} 个），完全一致 {o['exact']}，"
                f"信号保留 {o['retention']:.2f}，换车 {o['nflip']}，精修后缩短 {o['impr_after'] if o['impr_after'] is None else round(o['impr_after'], 3)}%，"
                f"分布差异 {{{', '.join(f'{f}:{v:.2f}' for f, v in o['tvd'].items())}}}")
        say("  事先预测：", [{k: (round(v, 3) if isinstance(v, float) else v) for k, v in p.items()} for p in A["prediction"]])
    res.append(row)

say("\n==== 汇总（校准后）====")
for arm in ("Q", "P"):
    sa = [r["arms"][arm]["mitigated"]["sign_agreement"] for r in res]
    ex = sum(r["arms"][arm]["mitigated"]["exact"] for r in res)
    ret = [r["arms"][arm]["mitigated"]["retention"] for r in res]
    ia = [r["arms"][arm]["mitigated"]["impr_after"] for r in res]; ii = [r["arms"][arm]["ideal"]["impr_after"] for r in res]
    say(f"{arm}：正负号一致 平均 {np.mean(sa):.3f}（最低 {np.min(sa):.3f}）；完全一致 {ex}/{len(res)}；信号保留平均 {np.mean(ret):.2f}；"
        f"精修后缩短 真机 {np.round(ia, 3)} 模拟 {np.round(ii, 3)}")
qa = [r["arms"]["Q"]["mitigated"]["impr_after"] or -9 for r in res]; pa = [r["arms"]["P"]["mitigated"]["impr_after"] or -9 for r in res]
say(f"真机上 Q 对 P（精修后缩短）：{sum(a > b + 1e-9 for a, b in zip(qa, pa))} 胜 {sum(a < b - 1e-9 for a, b in zip(qa, pa))} 负（4 个实例，只作报告）")
json.dump(dict(readout=dict(p01=p01.tolist(), p10=p10.tolist()), layout=LED.get("_layout"), bit_order=ORDER, instances=res),
          open(os.path.join(PKG, "analysis", "analysis_pce8.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=float)
open(os.path.join(PKG, "analysis", "analysis_pce8.txt"), "w", encoding="utf-8").write("\n".join(lines) + "\n")
