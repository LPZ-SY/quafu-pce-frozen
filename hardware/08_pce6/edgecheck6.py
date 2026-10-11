"""门相位检测（6 比特链）：估计每条边 CZ 的条件相位偏差和单比特相位偏差。
做法与 pce8c 的 edgecheck.py 相同：一层里互不相邻的边同时检测，层的划分与实验线路的 CZ 层完全一致
（偶数层 0-1,2-3,4-5；奇数层 1-2,3-4）。每条边 (a,b)：a、b 先变成 |+⟩，做一次 CZ；一个比特在 X 或 Y 方向测量，
另一个在 Z 方向测量。按 Z 方向比特的结果分组算相位：ψ0（理想 0）、ψ1（理想 π）。
  条件相位偏差 δ = (ψ1 − ψ0) − π：无法用单比特门补偿，作为是否可用的判据（|δ| ≤ 0.3 弧度）。
  单比特相位偏差 = ψ0：可以在每层 CZ 之后加 rz(−ψ0) 补偿；超过 1.0 弧度视为异常，该边不用。
每层 4 种设置 × 2 层 = 8 个任务；每个任务跑 2 次，每次 1024 次测量。"""
import numpy as np
N = 6
LAYERS = {"even": [(0, 1), (2, 3), (4, 5)], "odd": [(1, 2), (3, 4)]}
SETTINGS = [("first", "X"), ("first", "Y"), ("second", "X"), ("second", "Y")]
COND_MAX = 0.3      # 弧度：条件相位偏差上限（预登记）
SINGLE_MAX = 1.0    # 弧度：单比特相位偏差上限（超过视为异常，预登记）


def circuit(layer, setting):
    who, basis = setting
    L = ["OPENQASM 2.0;", 'include "qelib1.inc";', f"qreg q[{N}];", f"creg c[{N}];"]
    qs = sorted({q for e in LAYERS[layer] for q in e})
    L += [f"h q[{q}];" for q in qs]
    L += [f"cz q[{a}],q[{b}];" for a, b in LAYERS[layer]]
    for a, b in LAYERS[layer]:
        t = a if who == "first" else b
        if basis == "X": L.append(f"h q[{t}];")
        else: L += [f"rz({-np.pi / 2:.17g}) q[{t}];", f"h q[{t}];"]
    L += [f"measure q[{q}] -> c[{q}];" for q in range(N)]
    return "\n".join(L) + "\n"


def wrap(x): return (x + np.pi) % (2 * np.pi) - np.pi


def estimate(bits_by_setting, layer):
    out = {}
    for a, b in LAYERS[layer]:
        r = {}
        for who in ("first", "second"):
            t, c = (a, b) if who == "first" else (b, a)
            ph = {}
            for cv in (0, 1):
                xs = bits_by_setting[(who, "X")]; ys = bits_by_setting[(who, "Y")]
                ex = 1 - 2 * xs[xs[:, c] == cv][:, t].mean(); ey = 1 - 2 * ys[ys[:, c] == cv][:, t].mean()
                ph[cv] = float(np.arctan2(ey, ex)); r[f"amp_{who}_{cv}"] = float(np.hypot(ex, ey))
            r[f"cond_dev_{who}"] = float(wrap(ph[1] - ph[0] - np.pi)); r[f"single_dev_{who}"] = float(wrap(ph[0]))
        r["cond_dev"] = float(np.mean([r["cond_dev_first"], r["cond_dev_second"]]))
        r["max_cond"] = float(max(abs(r["cond_dev_first"]), abs(r["cond_dev_second"])))
        r["max_single"] = float(max(abs(r["single_dev_first"]), abs(r["single_dev_second"])))
        r["pass"] = r["max_cond"] <= COND_MAX and r["max_single"] <= SINGLE_MAX
        out[(a, b)] = r
    return out


def compensate(txt, verdict):
    """在逻辑线路的每层 CZ 之后加单比特相位补偿。verdict[(a,b)] 含 single_dev_first（a 的相位）、single_dev_second（b 的相位）。
    同一层的 CZ 是连续的几行；在这一层最后一个 CZ 之后统一插入补偿。"""
    lines = txt.splitlines(); out = []; pend = []
    for i, ln in enumerate(lines):
        out.append(ln)
        if ln.startswith("cz "):
            a, b = [int(x.split("[")[1].rstrip("];")) for x in ln[3:].split(",")]
            pend.append((min(a, b), max(a, b)))
            nxt = lines[i + 1] if i + 1 < len(lines) else ""
            if not nxt.startswith("cz "):
                for e in pend:
                    v = verdict[e]
                    out.append(f"rz({-v['single_dev_first']:.17g}) q[{e[0]}];"); out.append(f"rz({-v['single_dev_second']:.17g}) q[{e[1]}];")
                pend = []
    return "\n".join(out) + "\n"
