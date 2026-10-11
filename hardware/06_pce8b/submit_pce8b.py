"""泡利关联编码真机展示 pce8b（pce8 的重跑）：提交 / 收取

与 pce8 的唯一区别：选比特链时，先对候选链的每条边做“门相位检测”（edgecheck.py），
有任何一条边的相位偏差超过 0.3 弧度，就排除这条边、换下一条链再检测（最多 4 轮）。
实验线路、任务清单、判据与 pce8 完全相同（线路文件逐字相同，哈希核对）。

用法（双击 run_pce8b.bat 即可）：
    python submit_pce8b.py --check     # 只做本地检查，不联网
    python submit_pce8b.py --run       # 检测选链 → 位序诊断 → 其余 100 个任务 → 收取
中途断了直接再运行 --run：已提交的任务不会重复提交。
提交方式不变：compile=False、compiler=None、correct=False、open_dd=None。"""
import argparse, hashlib, json, os, sys, time, math, re
from pathlib import Path
import numpy as np

PKG = Path(__file__).resolve().parent
sys.path.insert(0, str(PKG))
import edgecheck as EC
LIVE = PKG / "live"; RAW = LIVE / "raw"; RAWC = LIVE / "raw_check"; PHY = LIVE / "physical"
MAN = json.load(open(PKG / "task_manifest.json", encoding="utf-8"))
N = MAN["n_qubits"]; CHIPS = ["Baihua", "Dongling"]; CHECK_REPS = 2; MAX_ROUNDS = 4
USE = {}
for inst in MAN["instances"]:
    cnt = {}
    for l in range(inst["arms"]["Q"]["layers"]):
        for a in range(l % 2, N - 1, 2): cnt[(a, a + 1)] = cnt.get((a, a + 1), 0) + 1
    for e, v in cnt.items(): USE[e] = max(USE.get(e, 0), v)


def token():
    t = os.environ.get("QUAFU_API_TOKEN", "").strip()
    for p in (Path.cwd() / ".env", PKG.parent / ".env", PKG.parent.parent / ".env"):
        if t: break
        if p.exists():
            for line in p.read_text(encoding="utf-8").splitlines():
                if line.strip().startswith("QUAFU_API_TOKEN="):
                    t = line.split("=", 1)[1].strip().strip('"').strip("'")
    return t


def load_ledger():
    f = LIVE / "ledger.json"
    return json.load(open(f, encoding="utf-8")) if f.exists() else {}


def save_ledger(L):
    LIVE.mkdir(exist_ok=True); tmp = LIVE / "ledger.tmp"
    json.dump(L, open(tmp, "w", encoding="utf-8"), ensure_ascii=False, indent=1); os.replace(tmp, LIVE / "ledger.json")


def check():
    bad = 0
    for t in MAN["tasks"]:
        txt = (PKG / t["file"]).read_text(encoding="utf-8")
        ok = hashlib.sha256(txt.encode()).hexdigest() == t["sha256"]
        gates = {l.split()[0].split("(")[0] for l in txt.splitlines() if l and not l.startswith(("OPENQASM", "include", "qreg", "creg"))}
        ok &= gates <= {"h", "ry", "rz", "cz", "measure"}
        if not ok: bad += 1; print("检查失败:", t["order"], t["file"], gates)
    print(f"本地检查：{len(MAN['tasks'])} 个任务，失败 {bad} 个"); return bad == 0


def norm_counts(raw):
    c = raw.get("count") or raw.get("counts") or {}
    out = {}
    for k, v in (c.items() if isinstance(c, dict) else []):
        try: out[str(k).replace(" ", "")] = out.get(str(k).replace(" ", ""), 0) + int(v)
        except (TypeError, ValueError): pass
    return out


def edges_of(info):
    E = {}
    for c in info["couplers_info"].values():
        a, b = c["qubits_index"]; f = c.get("fidelity") or 0
        if 0 < f <= 1: E[(min(a, b), max(a, b))] = f
    return E


def best_chain(info, excluded):
    E = {e: f for e, f in edges_of(info).items() if e not in excluded}
    adj = {}
    for a, b in E: adj.setdefault(a, set()).add(b); adj.setdefault(b, set()).add(a)

    def ro(q):
        i = info["qubits_info"].get(f"Q{q}", {}); g, e = i.get("readout g_fidelity") or 0, i.get("readout e_fidelity") or 0
        return math.log((g + e) / 2) if g > 0 and e > 0 else 0.0
    best = (None, None)

    def rec(path):
        nonlocal best
        if len(path) == N:
            s = sum(USE.get((k, k + 1), 0) * math.log(E[(min(path[k], path[k + 1]), max(path[k], path[k + 1]))]) for k in range(N - 1))
            s += sum(ro(q) for q in path)
            if best[0] is None or s > best[0]: best = (s, list(path))
            return
        for nb in adj.get(path[-1], ()):
            if nb not in path: rec(path + [nb])
    for q0 in adj: rec([q0])
    if best[1] is None: return None
    path = best[1]
    fs = [E[(min(path[k], path[k + 1]), max(path[k], path[k + 1]))] for k in range(N - 1)]
    F = math.prod(fs[k] ** USE.get((k, k + 1), 0) for k in range(N - 1))
    return dict(score=best[0], phys=path, cz_fidelity=fs, circuit_cz_fidelity=F)


def to_physical(txt, phys):
    out = []
    for line in txt.splitlines():
        if line.startswith("qreg"): out.append(f"qreg q[{max(phys) + 1}];")
        elif line.startswith(("OPENQASM", "include", "creg")): out.append(line)
        else: out.append(re.sub(r"q\[(\d+)\]", lambda m: f"q[{phys[int(m.group(1))]}]", line))
    return "\n".join(out) + "\n"


def bits_from(cnt, order):
    rows = []
    for k, v in cnt.items():
        b = [int(k[N - 1 - i]) for i in range(N)] if order == "c7…c0" else [int(k[i]) for i in range(N)]
        rows.append((b, v))
    return np.repeat(np.array([b for b, _ in rows]), [v for _, v in rows], axis=0)


def run_task(mgr, key, name, chip, txt, L, rawdir):
    """提交一个任务（已提交则跳过），返回 key。"""
    if key in L and L[key].get("task_id"): return key
    PHY.mkdir(parents=True, exist_ok=True); (PHY / f"{key}.qasm").write_text(txt, encoding="utf-8")
    payload = {"chip": chip, "name": name, "circuit": txt, "shots": MAN["shots"], "compile": False,
               "options": {"compiler": None, "correct": False, "open_dd": None, "target_qubits": []}}
    L[key] = dict(name=name, intent_time=time.time(), status="SUBMITTING", rawdir=str(rawdir.name)); save_ledger(L)
    tid = str(mgr.run(payload)); L[key].update(task_id=tid, submit_time=time.time(), status="SUBMITTED"); save_ledger(L)
    print(f"已提交 {key} -> {tid}", flush=True); return key


def collect(mgr, keys, L, timeout=10800):
    t0 = time.time()
    while True:
        pending = [k for k in keys if L[k].get("status") not in ("DONE", "FAILED")]
        if not pending: return True
        for k in pending:
            e = L[k]; rd = LIVE / e["rawdir"]; rd.mkdir(parents=True, exist_ok=True)
            try: r = mgr.result(e["task_id"])
            except Exception as ex: print("查询出错", k, ex); continue
            if not isinstance(r, dict): continue
            st = str(r.get("status", "")).lower(); cnt = norm_counts(r)
            if cnt or st in ("finished", "completed", "done", "success", "failed", "error", "cancelled"):
                json.dump(r, open(rd / f"{k}.json", "w", encoding="utf-8"), ensure_ascii=False, default=str)
                n = sum(cnt.values())
                e.update(status="DONE" if n == MAN["shots"] else "FAILED", shots_returned=n, platform_status=st, done_time=time.time())
                save_ledger(L); print(f"收到 {k} status={st} shots={n}", flush=True)
        if time.time() - t0 > timeout: print("等待超时，稍后再运行 --run 继续收取"); return False
        time.sleep(15)


def diagnostics(mgr, L, chip, phys):
    """位序诊断（每条候选链都要先确认位序，才能解读检测结果）。返回位序或 None。"""
    keys = []
    for t in [t for t in MAN["tasks"] if t["kind"] == "diagnostic"]:
        txt = (PKG / t["file"]).read_text(encoding="utf-8")
        assert hashlib.sha256(txt.encode()).hexdigest() == t["sha256"]
        key = f"{t['order']:03d}" if L.get("_layout") else f"pre_{chip}_{'-'.join(map(str, phys))}_{t['name']}"
        keys.append((run_task(mgr, key, f"QRF_pce8b_{key}", chip, to_physical(txt, phys), L, RAW if L.get("_layout") else RAWC), t))
    if not collect(mgr, [k for k, _ in keys], L): return None
    order = None
    for k, t in keys:
        cnt = norm_counts(json.load(open(LIVE / L[k]["rawdir"] / f"{k}.json", encoding="utf-8")))
        top = max(cnt, key=cnt.get) if cnt else ""
        b = t["bits_c0_to_c7"]; this = "c7…c0" if top == "".join(map(str, b[::-1])) else ("c0…c7" if top == "".join(map(str, b)) else None)
        print(f"  位序诊断 {t['name']}: 实测最多 {top} 占比 {cnt.get(top, 0) / MAN['shots']:.2f} → {this}")
        if this is None or (order and order != this): return None
        order = this
    return order


def select_layout(mgr, L, chip_arg):
    from quark.circuit import Backend
    infos = {}
    for chip in ([chip_arg] if chip_arg else CHIPS):
        try: infos[chip] = dict(Backend(chip).chip_info)
        except Exception as ex: print(chip, "读取标定失败：", ex); continue
        json.dump(infos[chip], open(LIVE / f"{chip}_calibration_at_submit.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        bg = set(infos[chip].get("basis_gates") or [])
        if bg and not {"h", "ry", "rz", "cz"} <= bg: print(chip, "基础门不含 h/ry/rz/cz：", bg); del infos[chip]
    L.setdefault("_excluded", {c: [] for c in infos}); L.setdefault("_check_rounds", []); save_ledger(L)
    for rnd in range(len(L["_check_rounds"]) + 1, MAX_ROUNDS + 1):
        cands = []
        for chip, info in infos.items():
            r = best_chain(info, {tuple(e) for e in L["_excluded"].get(chip, [])})
            if r: r["chip"] = chip; r["calibration_time"] = info.get("calibration_time"); cands.append(r)
            if r: print(f"第 {rnd} 轮候选 {chip}：链 {r['phys']} CZ 保真度 {[round(x, 4) for x in r['cz_fidelity']]} 总保真度 {r['circuit_cz_fidelity']:.3f} 得分 {r['score']:.3f}")
        if not cands: return None
        pick = max(cands, key=lambda r: r["score"]); chip, phys = pick["chip"], pick["phys"]
        print(f"第 {rnd} 轮检测：{chip} {phys}")
        order = diagnostics(mgr, L, chip, phys)
        if order is None: print("  位序诊断不通过，停止。"); return None
        keys = {}
        for layer in EC.LAYERS:
            for s in EC.SETTINGS:
                for rep in range(1, CHECK_REPS + 1):
                    key = f"chk{rnd}_{chip}_{layer}_{s[0]}{s[1]}_r{rep}"
                    keys[key] = (layer, s)
                    run_task(mgr, key, f"QRF_pce8b_{key}", chip, to_physical(EC.circuit(layer, s), phys), L, RAWC)
        if not collect(mgr, list(keys), L): return None
        verdict, fails = {}, []
        for layer in EC.LAYERS:
            bits = {}
            for s in EC.SETTINGS:
                arr = [bits_from(norm_counts(json.load(open(RAWC / f"{k}.json", encoding="utf-8"))), order)
                       for k, (ly, ss) in keys.items() if ly == layer and ss == s]
                bits[s] = np.vstack(arr)
            for (a, b), r in EC.estimate(bits, layer).items():
                pe = sorted((phys[a], phys[b])); verdict[f"{pe[0]}-{pe[1]}"] = r
                print(f"  边 Q{pe[0]}-Q{pe[1]}（逻辑 {a}-{b}）：条件相位偏差 {r['cond_dev_first']:+.2f}/{r['cond_dev_second']:+.2f}，"
                      f"单比特相位偏差 {r['single_dev_first']:+.2f}/{r['single_dev_second']:+.2f}，最大 {r['max_abs_dev']:.2f} → {'通过' if r['pass'] else '不通过'}")
                if not r["pass"]: fails.append(pe)
        L["_check_rounds"].append(dict(round=rnd, chip=chip, phys=phys, bit_order=order, verdict=verdict, fails=fails)); save_ledger(L)
        if not fails:
            pick["bit_order"] = order; return pick
        L["_excluded"].setdefault(chip, []).extend(fails); save_ledger(L)
        print(f"  有 {len(fails)} 条边不通过，排除后重新选链。")
    return None


def run(chip_arg):
    tok = token()
    if not tok: sys.exit("没有找到 QUAFU_API_TOKEN（环境变量或 .env），未提交任何任务")
    os.environ["QPU_API_TOKEN"] = tok
    from quark import Task
    mgr = Task(tok); L = load_ledger()
    if not L.get("_layout"):
        pick = select_layout(mgr, L, chip_arg)
        if pick is None: sys.exit("没有找到所有边都通过检测的比特链（或诊断失败）：未提交实验任务。请把 live 文件夹发回 Claude。")
        L["_layout"] = pick; save_ledger(L); print("选用：", pick["chip"], pick["phys"])
    chip, phys = L["_layout"]["chip"], L["_layout"]["phys"]
    order = diagnostics(mgr, L, chip, phys)                       # 正式的 2 个位序诊断（任务 1、2）
    L["_bit_order"] = order; L["_diagnostics_passed"] = order is not None; save_ledger(L)
    if order is None: sys.exit("位序诊断不通过：按预案停止。请把 live 文件夹发回 Claude。")
    rest = [t for t in MAN["tasks"] if t["kind"] != "diagnostic"]
    keys = []
    for t in rest:
        txt = (PKG / t["file"]).read_text(encoding="utf-8")
        assert hashlib.sha256(txt.encode()).hexdigest() == t["sha256"], "线路哈希不符，停止"
        key = f"{t['order']:03d}"
        keys.append(run_task(mgr, key, f"QRF_pce8b_{key}_{t['name']}", chip, to_physical(txt, phys), L, RAW))
    if collect(mgr, keys, L):
        print("全部完成。请告诉 Claude：pce8b 重跑结果已回收（live 文件夹）。")


class _Tee:
    def __init__(self, *s): self.s = s
    def write(self, x):
        for f in self.s: f.write(x); f.flush()
    def flush(self):
        for f in self.s: f.flush()


if __name__ == "__main__":
    LIVE.mkdir(exist_ok=True)
    _log = open(LIVE / "run_log.txt", "a", encoding="utf-8")
    sys.stdout = _Tee(sys.__stdout__, _log); sys.stderr = _Tee(sys.__stderr__, _log)
    print("==== 开始", time.strftime("%Y-%m-%d %H:%M:%S"), "cwd=", os.getcwd())
    ap = argparse.ArgumentParser(); ap.add_argument("--check", action="store_true"); ap.add_argument("--run", action="store_true")
    ap.add_argument("--chip", choices=CHIPS)
    a = ap.parse_args()
    if a.run:
        if check(): run(a.chip)
    else:
        check()
