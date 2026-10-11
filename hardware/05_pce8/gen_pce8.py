"""泡利关联编码真机展示 pce8：生成线路与任务清单（离线，在提交前冻结）

内容：3 辆车子问题（8 比特，约 26 个客户变量，14 个 CZ），起点对齐 + 换车惩罚 + 间隔下限两段训练
     （与 diag_two.py 相同，诊断记录见项目文档“泡利关联编码_实现复查与起点对齐诊断”）。
两组线路：Q（纠缠线路）、P（同一结构去掉全部 CZ，单独训练）。每组 3 种测量设置（全 Z / 全 X / 全 Y）。
选题规则（机械执行）：从种子 20001 起按顺序；每个实例 Q、P 各训练 3 个随机起点，各自取“无噪声读出 → 超载修正 →
     轻精修”后总路程缩短最多的一次；若 Q 选中的那次换车客户数 ≥1 且缩短 >0，则入选，否则跳过并记录。取满 4 个实例。
     这是展示用的选题规则：它挑的是“模拟里线路确实提出了改动”的实例，结论只针对这类实例。
"""
import os, sys, json, hashlib, itertools, random
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import pce_scale as S
import diag_two as D

N, K_VEH, NINST, NSTART, SEED0 = 8, 3, 4, 3, 20001
ROUNDS, SHOTS = 4, 1024
OUT = os.path.join(HERE, "pkg")
sha = lambda s: hashlib.sha256(s.encode()).hexdigest()
FAM = "ZXY"


def train(sub, ent, rng):
    """与 diag_two.run_q(two=True) 相同的训练，额外返回参数与锚定正负号。"""
    m = sub["m"]; L = max(4, int(np.ceil(m / N))); circ = S.Circuit(N, L, ent)
    slots = S.make_slots(N, m); alpha = 1.5 * N
    th0 = rng.uniform(-np.pi, np.pi, circ.npar)
    c = np.sign(circ.expect(circ.run(th0), slots)); c[c == 0] = 1
    th1, _, _ = S.adam(D.make_fun(sub, circ, slots, c, alpha, True, 0.0), th0)
    th, _, _ = S.adam(D.make_fun(sub, circ, slots, c, alpha, True, D.MW), th1)
    e = circ.expect(circ.run(th), slots)
    return dict(theta=th, c=c, e=e, flip=(c * e < 0).astype(int), L=L, ncz=circ.ncz, slots=slots)


def qasm_logical(th, L, ent, fam):
    th = th.reshape(L, N, 2)
    out = ["OPENQASM 2.0;", 'include "qelib1.inc";', f"qreg q[{N}];", f"creg c[{N}];"]
    for l in range(L):
        for q in range(N):
            out += [f"ry({th[l, q, 0]:.17g}) q[{q}];", f"rz({th[l, q, 1]:.17g}) q[{q}];"]
        if ent:
            out += [f"cz q[{a}],q[{a + 1}];" for a in range(l % 2, N - 1, 2)]
    for q in range(N):
        if fam == "X": out += [f"h q[{q}];"]
        if fam == "Y": out += [f"rz({-np.pi / 2:.17g}) q[{q}];", f"h q[{q}];"]
    out += [f"measure q[{q}] -> c[{q}];" for q in range(N)]
    return "\n".join(out) + "\n"


# ---------- 独立的 qasm 模拟器（只认本包用到的门），用于核对线路文件 ----------
def sim_qasm(txt):
    psi = np.zeros((2,) * N, complex); psi[(0,) * N] = 1
    Hm = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
    for line in txt.splitlines():
        line = line.strip()
        if not line or line.startswith(("OPENQASM", "include", "qreg", "creg", "measure")): continue
        g, rest = line.split(" ", 1); args = [int(x.split("[")[1].rstrip("];,")) for x in rest.split(",")]
        if g == "h": U = Hm
        elif g.startswith("ry"):
            t = float(g[3:-1]); U = np.array([[np.cos(t / 2), -np.sin(t / 2)], [np.sin(t / 2), np.cos(t / 2)]])
        elif g.startswith("rz"):
            t = float(g[3:-1]); U = np.diag([np.exp(-0.5j * t), np.exp(0.5j * t)])
        elif g == "cz":
            a, b = args; idx = [slice(None)] * N; idx[a] = 1; idx[b] = 1; psi[tuple(idx)] *= -1; continue
        else: raise ValueError(g)
        q = args[0]; psi = np.moveaxis(np.tensordot(U, np.moveaxis(psi, q, 0), axes=(1, 0)), 0, q)
    return (np.abs(psi.reshape(-1)) ** 2)       # 下标的二进制：比特 0 为最高位


BITS = ((np.arange(1 << N)[:, None] >> (N - 1 - np.arange(N))) & 1)


def corr_from_probs(p, pairs):
    Zm = 1 - 2 * BITS
    C = Zm.T @ (p[:, None] * Zm)
    return np.array([C[a, b] for a, b in pairs])


def predict(probs_by_fam, slots, c, flip_ideal, f_cz, ncz, rng, reps=200, p01=0.031, p10=0.037):
    """含噪预测：整体去极化 + 不对称读出误差 + 每种设置 SHOTS×ROUNDS 次 + 逐比特读出校准。"""
    F = f_cz ** ncz; agree, exact = [], 0
    M = np.array([[1 - p01, p10], [p01, 1 - p10]]); Minv = np.linalg.inv(M)
    for _ in range(reps):
        e = np.zeros(len(slots))
        for f in FAM:
            idx = [k for k, s in enumerate(slots) if s[0] == f]
            if not idx: continue
            p = F * probs_by_fam[f] + (1 - F) / (1 << N)
            cnt = rng.multinomial(SHOTS * ROUNDS, p)
            bits = np.repeat(BITS, cnt, axis=0)
            fl = np.where(bits == 1, rng.random(bits.shape) < p10, rng.random(bits.shape) < p01)
            bits = bits ^ fl
            for k in idx:
                _, a, b = slots[k]
                pm = np.array([np.mean((bits[:, a] == x) & (bits[:, b] == y)) for x in (0, 1) for y in (0, 1)])
                pc = np.kron(Minv, Minv) @ pm
                e[k] = pc[0] - pc[1] - pc[2] + pc[3]
        fl_hw = (c * e < 0).astype(int)
        agree.append(np.mean(fl_hw == flip_ideal)); exact += int((fl_hw == flip_ideal).all())
    return dict(f_cz=f_cz, circuit_fidelity=F, sign_agreement_mean=float(np.mean(agree)),
                sign_agreement_min=float(np.min(agree)), exact_match_rate=exact / reps)


def main():
    for d in ("circuits", "ideal", "diagnostics", "calibration"): os.makedirs(os.path.join(OUT, d), exist_ok=True)
    scan, insts, s = [], [], SEED0
    while len(insts) < NINST:
        P_, q, Dm, R = S.instance(s); start = S.total(Dm, R); grp = S.pick_groups(P_, R)[K_VEH]
        sub = S.build_sub(P_, q, Dm, R, grp); rng = np.random.default_rng(s)
        rec = dict(seed=s, start_total=start, routes=grp, m=sub["m"], customers=sub["cust"].tolist(),
                   orig=sub["orig"].tolist(), targ=sub["targ"].tolist(), arms={})
        for arm, ent in (("Q", True), ("P", False)):
            best = None; tried = []
            for r in range(NSTART):
                t = train(sub, ent, rng)
                ev = S.evaluate(P_, q, Dm, R, grp, sub, t["flip"], start)
                tried.append(dict(nflip=int(t["flip"].sum()), impr_after=ev.get("impr_after"), feasible=ev["feasible"]))
                key = ev["impr_after"] if ev["feasible"] else -1e9
                if best is None or key > best[0]: best = (key, t, ev)
            rec["arms"][arm] = dict(train=best[1], eval=best[2], tried=tried)
        q_ok = rec["arms"]["Q"]["eval"]["feasible"] and rec["arms"]["Q"]["train"]["flip"].sum() >= 1 \
            and rec["arms"]["Q"]["eval"]["impr_after"] > 0
        scan.append(dict(seed=s, taken=bool(q_ok), Q=rec["arms"]["Q"]["tried"], P=rec["arms"]["P"]["tried"]))
        print(s, "入选" if q_ok else "跳过", scan[-1], flush=True)
        if q_ok: insts.append(rec)
        s += 1
    tasks, inst_out = [], []
    prng = np.random.default_rng(7)
    for rec in insts:
        sd = rec["seed"]; io = {k: v for k, v in rec.items() if k != "arms"}; io["arms"] = {}
        for arm in ("Q", "P"):
            t = rec["arms"][arm]["train"]; ev = rec["arms"][arm]["eval"]; ent = arm == "Q"
            files, probs = {}, {}
            for f in FAM:
                txt = qasm_logical(t["theta"], t["L"], ent, f); p = sim_qasm(txt); probs[f] = p
                fn = f"circuits/s{sd}_{arm}_{f}.qasm"
                open(os.path.join(OUT, fn), "w", encoding="utf-8", newline="\n").write(txt)
                np.save(os.path.join(OUT, "ideal", f"s{sd}_{arm}_{f}.npy"), p)
                files[f] = dict(file=fn, sha256=sha(txt), cz=txt.count("cz "))
            # 核对：qasm 模拟出的关联值 = 训练时的关联值
            e_q = np.zeros(rec["m"])
            for f in FAM:
                idx = [k for k, sl in enumerate(t["slots"]) if sl[0] == f]
                cc = corr_from_probs(probs[f], [(t["slots"][k][1], t["slots"][k][2]) for k in idx])
                e_q[idx] = cc
            err = float(np.abs(e_q - t["e"]).max()); assert err < 1e-9, (sd, arm, err)
            pred = [predict(probs, t["slots"], t["c"], t["flip"], fz, t["ncz"], prng) for fz in (0.977, 0.99)] if ent else \
                   [predict(probs, t["slots"], t["c"], t["flip"], 1.0, 0, prng)]
            io["arms"][arm] = dict(files=files, theta=t["theta"].tolist(), anchor_sign=t["c"].tolist(), layers=t["L"],
                                   ncz=t["ncz"], slots=[list(x) for x in t["slots"]], e_ideal=t["e"].tolist(),
                                   flip_ideal=t["flip"].tolist(), min_gap=float(np.abs(t["e"]).min()),
                                   eval_ideal=ev, tried=rec["arms"][arm]["tried"], qasm_check_err=err, prediction=pred)
            print(sd, arm, "CZ", t["ncz"], "换车", int(t["flip"].sum()), "最小间隔", round(float(np.abs(t["e"]).min()), 3),
                  "精修后缩短%", round(ev.get("impr_after", float("nan")), 3), "预测", pred, flush=True)
        inst_out.append(io)
    # 诊断（位序）与读出标定
    def basis(bits):
        L = ["OPENQASM 2.0;", 'include "qelib1.inc";', f"qreg q[{N}];", f"creg c[{N}];"]
        L += [f"ry({np.pi:.17g}) q[{i}];" for i in range(N) if bits[i]]
        L += [f"measure q[{i}] -> c[{i}];" for i in range(N)]
        return "\n".join(L) + "\n"
    for nm, bits in (("diag_low4", [1, 1, 1, 1, 0, 0, 0, 0]), ("diag_q0q1q4", [1, 1, 0, 0, 1, 0, 0, 0])):
        txt = basis(bits); fn = f"diagnostics/{nm}.qasm"; open(os.path.join(OUT, fn), "w", encoding="utf-8", newline="\n").write(txt)
        tasks.append(dict(kind="diagnostic", name=nm, file=fn, sha256=sha(txt), bits_c0_to_c7=bits))
    for nm, b in (("ro_all0", 0), ("ro_all1", 1)):
        txt = basis([b] * N); fn = f"calibration/{nm}.qasm"; open(os.path.join(OUT, fn), "w", encoding="utf-8", newline="\n").write(txt)
        tasks.append(dict(kind="calibration", name=nm, file=fn, sha256=sha(txt), bits_c0_to_c7=[b] * N))
    rnd = random.Random(20261010)
    for r in range(1, ROUNDS + 1):
        jobs = [(io, arm, f) for io in inst_out for arm in ("Q", "P") for f in FAM]; rnd.shuffle(jobs)
        for io, arm, f in jobs:
            fi = io["arms"][arm]["files"][f]
            tasks.append(dict(kind="algorithm", round=r, instance=io["seed"], arm=arm, family=f,
                              name=f"s{io['seed']}_{arm}_{f}_r{r}", file=fi["file"], sha256=fi["sha256"]))
    for nm, b in (("ro_all0_end", 0), ("ro_all1_end", 1)):
        fn = f"calibration/{nm[:7]}.qasm"; txt = open(os.path.join(OUT, fn), encoding="utf-8").read()
        tasks.append(dict(kind="calibration", name=nm, file=fn, sha256=sha(txt), bits_c0_to_c7=[b] * N))
    for i, t in enumerate(tasks, 1): t.update(order=i, shots=SHOTS)
    man = dict(created="2026-10-10", version="pce8", design=__doc__, n_qubits=N, rounds=ROUNDS, shots=SHOTS,
               logical_cz_edges=[[a, a + 1] for a in range(N - 1)], scan_log=scan, instances=inst_out, tasks=tasks,
               submit_rule="提交时读取 Baihua、Dongling 当天标定，在两块芯片上枚举所有 8 比特链，按 Σ(每条边使用次数×ln CZ保真度) + Σ ln 读出保真度 取最高者；"
                           "逻辑线路不变，只把 q[i] 换成物理比特。")
    json.dump(man, open(os.path.join(OUT, "task_manifest.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=float)
    print("任务数", len(tasks))


if __name__ == "__main__":
    main()
