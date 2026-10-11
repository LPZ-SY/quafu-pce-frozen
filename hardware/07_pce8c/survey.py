"""全芯片门相位普查（筛选用）：把芯片上所有可用的边分成若干组，每组内的边互不共用比特，同一组的边同时检测。
每组 4 个任务（读相位的比特 × X/Y 方向），每个 1024 次。检测原理与 edgecheck.py 相同。
普查只用来缩小候选范围；选中的链仍须通过 edgecheck.py 的正式检测（预登记规则不变）。"""
import numpy as np

SETTINGS = [("first", "X"), ("first", "Y"), ("second", "X"), ("second", "Y")]


def color_edges(edges):
    """贪心边着色：返回若干组，每组内的边不共用比特。"""
    groups = []
    for e in sorted(edges, key=lambda e: (e[0], e[1])):
        for g in groups:
            if all(e[0] not in f and e[1] not in f for f in g): g.append(e); break
        else: groups.append([e])
    return groups


def circuit(group, setting, nq):
    """group 为物理边列表 (a,b)，a<b。返回 (qasm, 测量的物理比特列表)。"""
    who, basis = setting
    qs = sorted({q for e in group for q in e})
    L = ["OPENQASM 2.0;", 'include "qelib1.inc";', f"qreg q[{nq}];", f"creg c[{len(qs)}];"]
    L += [f"h q[{q}];" for q in qs]
    L += [f"cz q[{a}],q[{b}];" for a, b in group]
    for a, b in group:
        t = a if who == "first" else b
        if basis == "X": L.append(f"h q[{t}];")
        else: L += [f"rz({-np.pi / 2:.17g}) q[{t}];", f"h q[{t}];"]
    L += [f"measure q[{q}] -> c[{i}];" for i, q in enumerate(qs)]
    return "\n".join(L) + "\n", qs


def basis_circuit(qs, nq, ones):
    L = ["OPENQASM 2.0;", 'include "qelib1.inc";', f"qreg q[{nq}];", f"creg c[{len(qs)}];"]
    L += [f"ry({np.pi:.17g}) q[{q}];" for i, q in enumerate(qs) if i in ones]
    L += [f"measure q[{q}] -> c[{i}];" for i, q in enumerate(qs)]
    return "\n".join(L) + "\n"


def bits(cnt, ncl, order):
    rows = []
    for k, v in cnt.items():
        k = str(k).replace(" ", "")
        if len(k) != ncl: continue
        b = [int(k[ncl - 1 - i]) for i in range(ncl)] if order == "c_last_first" else [int(k[i]) for i in range(ncl)]
        rows.append((b, v))
    return np.repeat(np.array([b for b, _ in rows]), [v for _, v in rows], axis=0)


def wrap(x): return (x + np.pi) % (2 * np.pi) - np.pi


def estimate(bits_by_setting, group, qs):
    idx = {q: i for i, q in enumerate(qs)}; out = {}
    for a, b in group:
        r = {}
        for who in ("first", "second"):
            t, c = (idx[a], idx[b]) if who == "first" else (idx[b], idx[a])
            ph = {}
            for cv in (0, 1):
                xs, ys = bits_by_setting[(who, "X")], bits_by_setting[(who, "Y")]
                sx, sy = xs[xs[:, c] == cv], ys[ys[:, c] == cv]
                if len(sx) < 20 or len(sy) < 20: ph[cv] = np.nan; continue
                ph[cv] = float(np.arctan2(1 - 2 * sy[:, t].mean(), 1 - 2 * sx[:, t].mean()))
            r[f"cond_dev_{who}"] = float(wrap(ph[1] - ph[0] - np.pi)); r[f"single_dev_{who}"] = float(wrap(ph[0]))
        devs = [r[k] for k in r]
        r["max_abs_dev"] = float(np.max(np.abs(devs))) if not np.any(np.isnan(devs)) else 9.0
        out[f"{a}-{b}"] = r
    return out
