"""泡利关联编码 pce6：有纠缠与无纠缠的公平比较（离线生成，提交前冻结）

题目：种子 20201–20220，共 20 题，按顺序全部使用，不看任何结果挑题。每题取相邻 3 辆车组成子问题（与 pce8 相同）。
电路：6 比特链，4 层；Q = 有纠缠（每层后相邻比特 CZ，奇偶交替，共 10 个 CZ）；P = 同结构去掉全部 CZ。
训练：与 pce8 相同（起点对齐 + 换车惩罚 + 两段训练间隔下限 0.15）。Q、P 各 3 个随机起点，
      各取"无噪声读出 → 超载修正 → 轻精修"后路线缩短最多的一次（比原方案差的按 0 计）；两组规则完全相同。
测量：每个电路 3 种设置（全 Z / 全 X / 全 Y），每种跑 2 次，每次 1024 次。
"""
import os, sys, json, hashlib, random
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import pce_scale as S
import diag_two as D

N, LAYERS, K_VEH, NSTART = 6, 4, 3, 3
SEEDS = list(range(20201, 20221))
ROUNDS, SHOTS = 2, 1024
OUT = os.path.join(HERE, "out")
FAM = "ZXY"
sha = lambda s: hashlib.sha256(s.encode()).hexdigest()
RET, NOISE = 0.79, 0.064      # 预测用：pce8c 真机的信号保留与剩余误差


def train(sub, ent, rng):
    circ = S.Circuit(N, LAYERS, ent); m = sub["m"]; slots = S.make_slots(N, m); alpha = 1.5 * N
    th0 = rng.uniform(-np.pi, np.pi, circ.npar)
    c = np.sign(circ.expect(circ.run(th0), slots)); c[c == 0] = 1
    th1, _, _ = S.adam(D.make_fun(sub, circ, slots, c, alpha, True, 0.0), th0)
    th, _, _ = S.adam(D.make_fun(sub, circ, slots, c, alpha, True, D.MW), th1)
    e = circ.expect(circ.run(th), slots)
    return dict(theta=th, c=c, e=e, flip=(c * e < 0).astype(int), ncz=circ.ncz, slots=slots)


def qasm_logical(th, ent, fam):
    th = th.reshape(LAYERS, N, 2)
    out = ["OPENQASM 2.0;", 'include "qelib1.inc";', f"qreg q[{N}];", f"creg c[{N}];"]
    for l in range(LAYERS):
        for q in range(N):
            out += [f"ry({th[l, q, 0]:.17g}) q[{q}];", f"rz({th[l, q, 1]:.17g}) q[{q}];"]
        if ent: out += [f"cz q[{a}],q[{a + 1}];" for a in range(l % 2, N - 1, 2)]
    for q in range(N):
        if fam == "X": out += [f"h q[{q}];"]
        if fam == "Y": out += [f"rz({-np.pi / 2:.17g}) q[{q}];", f"h q[{q}];"]
    out += [f"measure q[{q}] -> c[{q}];" for q in range(N)]
    return "\n".join(out) + "\n"


def sim_qasm(txt, err=None):
    """独立的 qasm 模拟器。err[(a,b)] = (a 的附加相位, b 的附加相位, 条件相位偏差)，加在每个 CZ 上。"""
    psi = np.zeros((2,) * N, complex); psi[(0,) * N] = 1
    Hm = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
    for line in txt.splitlines():
        line = line.strip()
        if not line or line.startswith(("OPENQASM", "include", "qreg", "creg", "measure")): continue
        g, rest = line.split(" ", 1); args = [int(x.split("[")[1].rstrip("];,")) for x in rest.split(",")]
        if g == "cz":
            a, b = sorted(args); pa, pb, dl = (err or {}).get((a, b), (0, 0, 0))
            for i in (0, 1):
                for j in (0, 1):
                    idx = [slice(None)] * N; idx[a] = i; idx[b] = j
                    psi[tuple(idx)] *= (-1 if i and j else 1) * np.exp(1j * (pa * i + pb * j + dl * i * j))
            continue
        if g == "h": U = Hm
        elif g.startswith("ry"):
            t = float(g[3:-1]); U = np.array([[np.cos(t / 2), -np.sin(t / 2)], [np.sin(t / 2), np.cos(t / 2)]])
        elif g.startswith("rz"):
            t = float(g[3:-1]); U = np.diag([np.exp(-0.5j * t), np.exp(0.5j * t)])
        else: raise ValueError(g)
        q = args[0]; psi = np.moveaxis(np.tensordot(U, np.moveaxis(psi, q, 0), axes=(1, 0)), 0, q)
    return np.abs(psi.reshape(-1)) ** 2


BITS = ((np.arange(1 << N)[:, None] >> (N - 1 - np.arange(N))) & 1)


def corr(p, slots, fam):
    Zm = 1 - 2 * BITS; C = Zm.T @ (p[:, None] * Zm)
    return {k: C[s[1], s[2]] for k, s in enumerate(slots) if s[0] == fam}


def job(seed):
    P_, q, Dm, R = S.instance(seed); start = S.total(Dm, R); grp = S.pick_groups(P_, R)[K_VEH]
    sub = S.build_sub(P_, q, Dm, R, grp); rng = np.random.default_rng(seed)
    rec = dict(seed=seed, start_total=start, routes=grp, m=sub["m"], customers=sub["cust"].tolist(), arms={})
    for arm, ent in (("Q", True), ("P", False)):
        best, tried = None, []
        for r in range(NSTART):
            t = train(sub, ent, rng); ev = S.evaluate(P_, q, Dm, R, grp, sub, t["flip"], start)
            key = max(ev.get("impr_after") or 0.0, 0.0) if ev["feasible"] else 0.0
            tried.append(dict(nflip=int(t["flip"].sum()), impr_after=ev.get("impr_after"), feasible=ev["feasible"], score=key))
            if best is None or key > best[0] + 1e-12: best = (key, t, ev)
        key, t, ev = best
        files, probs = {}, {}
        for f in FAM:
            txt = qasm_logical(t["theta"], ent, f); probs[f] = sim_qasm(txt)
            files[f] = dict(file=f"circuits/s{seed}_{arm}_{f}.qasm", text=txt, sha256=sha(txt), cz=txt.count("cz "))
        e_q = np.zeros(sub["m"])
        for f in FAM:
            for k, v in corr(probs[f], t["slots"], f).items(): e_q[k] = v
        err = float(np.abs(e_q - t["e"]).max()); assert err < 1e-9, (seed, arm, err)
        # 预测：每条边条件相位偏差 U(-0.3,0.3)、补偿后剩余单比特相位 U(-0.1,0.1)；信号缩小到 0.79；加标准差 0.064 的误差
        prng = np.random.default_rng(seed + (0 if ent else 7)); hw_impr, exact = [], 0
        for d in range(30):
            em = {(a, a + 1): (prng.uniform(-.1, .1), prng.uniform(-.1, .1), prng.uniform(-.3, .3)) for a in range(N - 1)} if ent else None
            e = np.zeros(sub["m"])
            for f in FAM:
                for k, v in corr(sim_qasm(files[f]["text"], em), t["slots"], f).items(): e[k] = v
            hw = RET * e + prng.normal(0, NOISE, len(e)); fl = (t["c"] * hw < 0).astype(int)
            exact += int((fl == t["flip"]).all()); ev2 = S.evaluate(P_, q, Dm, R, grp, sub, fl, start)
            hw_impr.append(max(ev2.get("impr_after") or 0.0, 0.0) if ev2["feasible"] else 0.0)
        rec["arms"][arm] = dict(files=files, theta=t["theta"].tolist(), anchor_sign=t["c"].tolist(), layers=LAYERS, ncz=t["ncz"],
                                slots=[list(x) for x in t["slots"]], e_ideal=t["e"].tolist(), flip_ideal=t["flip"].tolist(),
                                min_gap=float(np.abs(t["e"]).min()), eval_ideal=ev, score_ideal=key, tried=tried, qasm_check_err=err,
                                probs={f: probs[f].tolist() for f in FAM},
                                prediction=dict(exact_rate=exact / 30, impr_hw_mean=float(np.mean(hw_impr))))
        print(seed, arm, "CZ", t["ncz"], "变量", sub["m"], "换车", int(t["flip"].sum()), "模拟缩短%", round(key, 3),
              "预测真机缩短%", round(float(np.mean(hw_impr)), 3), "预测全对率", exact / 30, flush=True)
    return rec


def main(nproc):
    from multiprocessing import Pool
    for d in ("circuits", "ideal", "diagnostics", "calibration"): os.makedirs(os.path.join(OUT, d), exist_ok=True)
    with Pool(nproc) as p: recs = sorted(p.map(job, SEEDS), key=lambda r: r["seed"])
    tasks, insts = [], []
    for rec in recs:
        for arm in ("Q", "P"):
            A = rec["arms"][arm]
            for f in FAM:
                fi = A["files"][f]
                open(os.path.join(OUT, fi["file"]), "w", encoding="utf-8", newline="\n").write(fi.pop("text"))
                np.save(os.path.join(OUT, "ideal", f"s{rec['seed']}_{arm}_{f}.npy"), np.array(A["probs"][f]))
            A.pop("probs")
        insts.append(rec)

    def basis(bits):
        L = ["OPENQASM 2.0;", 'include "qelib1.inc";', f"qreg q[{N}];", f"creg c[{N}];"]
        L += [f"ry({np.pi:.17g}) q[{i}];" for i in range(N) if bits[i]]
        L += [f"measure q[{i}] -> c[{i}];" for i in range(N)]
        return "\n".join(L) + "\n"
    for nm, bits in (("diag_low3", [1, 1, 1, 0, 0, 0]), ("diag_q0q1q4", [1, 1, 0, 0, 1, 0])):
        txt = basis(bits); fn = f"diagnostics/{nm}.qasm"; open(os.path.join(OUT, fn), "w", encoding="utf-8", newline="\n").write(txt)
        tasks.append(dict(kind="diagnostic", name=nm, file=fn, sha256=sha(txt), bits_c0_first=bits))
    for nm, b in (("ro_all0", 0), ("ro_all1", 1)):
        txt = basis([b] * N); fn = f"calibration/{nm}.qasm"; open(os.path.join(OUT, fn), "w", encoding="utf-8", newline="\n").write(txt)
        tasks.append(dict(kind="calibration", name=nm, file=fn, sha256=sha(txt), bits_c0_first=[b] * N))
    rnd = random.Random(20261011)
    for r in range(1, ROUNDS + 1):
        jobs = [(io, arm, f) for io in insts for arm in ("Q", "P") for f in FAM]; rnd.shuffle(jobs)
        for io, arm, f in jobs:
            fi = io["arms"][arm]["files"][f]
            tasks.append(dict(kind="algorithm", round=r, instance=io["seed"], arm=arm, family=f,
                              name=f"s{io['seed']}_{arm}_{f}_r{r}", file=fi["file"], sha256=fi["sha256"]))
    for nm, b in (("ro_all0_end", 0), ("ro_all1_end", 1)):
        fn = f"calibration/{nm[:7]}.qasm"; txt = open(os.path.join(OUT, fn), encoding="utf-8").read()
        tasks.append(dict(kind="calibration", name=nm, file=fn, sha256=sha(txt), bits_c0_first=[b] * N))
    for i, t in enumerate(tasks, 1): t.update(order=i, shots=SHOTS)
    man = dict(created="2026-10-11", version="pce6", design=__doc__, n_qubits=N, layers=LAYERS, rounds=ROUNDS, shots=SHOTS,
               logical_cz_edges=[[a, a + 1] for a in range(N - 1)], instances=insts, tasks=tasks)
    json.dump(man, open(os.path.join(OUT, "task_manifest.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=float)
    print("任务数", len(tasks))


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 2)
