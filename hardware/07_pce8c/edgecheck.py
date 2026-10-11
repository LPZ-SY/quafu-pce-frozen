"""门相位检测：生成检测线路、从测量结果估计每条边 CZ 门的条件相位偏差和单比特相位偏差。
做法：一层里互不相邻的边（与实验线路的 CZ 层完全相同：偶数层 0-1,2-3,4-5,6-7；奇数层 1-2,3-4,5-6）同时检测。
每条边 (a,b)：a、b 都先变成 |+⟩，做一次 CZ；然后一个比特在 X 或 Y 方向测量、另一个在 Z 方向测量。
按 Z 方向比特的结果（0 或 1）分组，算 X/Y 方向比特的相位：
  ψ0 = 另一比特为 0 时的相位（理想为 0），ψ1 = 另一比特为 1 时的相位（理想为 π），
  条件相位偏差 δ = (ψ1 − ψ0) − π，单比特相位偏差 = ψ0。两个方向各测一次（a 读相位、b 读相位）。
每层 4 种设置 × 2 层 = 8 个任务，每个 1024 次。"""
import numpy as np
N = 8
LAYERS = {"even": [(0, 1), (2, 3), (4, 5), (6, 7)], "odd": [(1, 2), (3, 4), (5, 6)]}
SETTINGS = [("first", "X"), ("first", "Y"), ("second", "X"), ("second", "Y")]   # 哪个比特读相位、在哪个方向
THRESH = 0.3   # 弧度，预登记的门槛


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
    """bits_by_setting[(who,basis)] = 形状 (次数, 8) 的 0/1 数组（下标为逻辑比特）。返回每条边的估计。"""
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
        r["max_abs_dev"] = float(max(abs(r["cond_dev_first"]), abs(r["cond_dev_second"]), abs(r["single_dev_first"]), abs(r["single_dev_second"])))
        r["pass"] = r["max_abs_dev"] <= THRESH
        out[(a, b)] = r
    return out
