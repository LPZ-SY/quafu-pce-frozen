"""泡利关联编码真机展示 pce8：提交 / 收取（Baihua 或 Dongling，102 个任务 × 1024 次）

用法（与以前的真机包相同的环境；双击 run_pce8.bat 即可）：
    python submit_pce8.py --check            # 只做本地检查，不联网
    python submit_pce8.py --run              # 选芯片和比特链 → 交 2 个位序诊断 → 核验 → 交其余任务 → 收取
    python submit_pce8.py --run --chip Baihua   # 指定芯片（默认在 Baihua、Dongling 中按规则自动选）
中途断了直接再运行 --run：已提交的任务不会重复提交，只继续收取。
提交方式与以前相同：compile=False、compiler=None、correct=False、open_dd=None。
选链规则（预登记）：读当天标定，枚举所有 8 个比特相连的链，
    得分 = Σ 每条边在线路中使用的次数 × ln(CZ 保真度) + Σ ln(读出保真度，取 g、e 平均；无数据则不计)，取最高者。
逻辑线路文件不变（哈希核对），只把 q[i] 换成物理比特，物理线路另存到 live/physical/。"""
import argparse, hashlib, json, os, sys, time, math, re
from pathlib import Path

PKG = Path(__file__).resolve().parent
LIVE = PKG / "live"; RAW = LIVE / "raw"; PHY = LIVE / "physical"
MAN = json.load(open(PKG / "task_manifest.json", encoding="utf-8"))
N = MAN["n_qubits"]; CHIPS = ["Baihua", "Dongling"]
USE = {}                                                     # 每条逻辑边在 Q 线路中的最多使用次数
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


def best_chain(info):
    E = {}
    for c in info["couplers_info"].values():
        a, b = c["qubits_index"]; f = c.get("fidelity") or 0
        if 0 < f <= 1: E[(min(a, b), max(a, b))] = f
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


def preflight(chip_arg):
    from quark.circuit import Backend
    LIVE.mkdir(exist_ok=True); cands = []
    for chip in ([chip_arg] if chip_arg else CHIPS):
        try: info = dict(Backend(chip).chip_info)
        except Exception as ex: print(chip, "读取标定失败：", ex); continue
        json.dump(info, open(LIVE / f"{chip}_calibration_at_submit.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        bg = set(info.get("basis_gates") or [])
        if bg and not {"h", "ry", "rz", "cz"} <= bg: print(chip, "基础门不含 h/ry/rz/cz：", bg); continue
        r = best_chain(info)
        if r is None: print(chip, "找不到 8 比特链"); continue
        r["chip"] = chip; r["calibration_time"] = info.get("calibration_time"); cands.append(r)
        print(f"{chip}（标定 {r['calibration_time']}）最好链 {r['phys']}  CZ 保真度 {[round(x, 4) for x in r['cz_fidelity']]}"
              f"  Q 线路全部 CZ 的总保真度 {r['circuit_cz_fidelity']:.3f}  得分 {r['score']:.3f}")
    if not cands: return None
    pick = max(cands, key=lambda r: r["score"])
    print("选用：", pick["chip"], pick["phys"])
    return pick


def to_physical(txt, phys):
    out = []
    for line in txt.splitlines():
        if line.startswith("qreg"): out.append(f"qreg q[{max(phys) + 1}];")
        elif line.startswith(("OPENQASM", "include", "creg")): out.append(line)
        else: out.append(re.sub(r"q\[(\d+)\]", lambda m: f"q[{phys[int(m.group(1))]}]", line))
    return "\n".join(out) + "\n"


def submit(mgr, t, L):
    key = str(t["order"])
    if key in L and L[key].get("task_id"): return
    txt = (PKG / t["file"]).read_text(encoding="utf-8")
    assert hashlib.sha256(txt.encode()).hexdigest() == t["sha256"], "线路哈希不符，停止"
    phys_txt = to_physical(txt, L["_layout"]["phys"])
    PHY.mkdir(parents=True, exist_ok=True)
    (PHY / f"{t['order']:03d}_{t['name']}.qasm").write_text(phys_txt, encoding="utf-8")
    name = f"QRF_pce8_{t['order']:03d}_{t['name']}"
    payload = {"chip": L["_layout"]["chip"], "name": name, "circuit": phys_txt, "shots": t["shots"], "compile": False,
               "options": {"compiler": None, "correct": False, "open_dd": None, "target_qubits": []}}
    L[key] = dict(order=t["order"], file=t["file"], name=t["name"], intent_time=time.time(), status="SUBMITTING"); save_ledger(L)
    tid = str(mgr.run(payload))
    L[key].update(task_id=tid, submit_time=time.time(), status="SUBMITTED"); save_ledger(L)
    print(f"已提交 #{t['order']:03d} {name} -> {tid}", flush=True)


def collect(mgr, orders, L, timeout=10800):
    t0 = time.time(); RAW.mkdir(parents=True, exist_ok=True)
    while True:
        pending = [o for o in orders if L[str(o)].get("status") not in ("DONE", "FAILED")]
        if not pending: return True
        for o in pending:
            e = L[str(o)]
            try: r = mgr.result(e["task_id"])
            except Exception as ex: print("查询出错", o, ex); continue
            if not isinstance(r, dict): continue
            st = str(r.get("status", "")).lower(); cnt = norm_counts(r)
            if cnt or st in ("finished", "completed", "done", "success", "failed", "error", "cancelled"):
                json.dump(r, open(RAW / f"{o:03d}.json", "w", encoding="utf-8"), ensure_ascii=False, default=str)
                n = sum(cnt.values())
                e.update(status="DONE" if n == MAN["shots"] else "FAILED", shots_returned=n, platform_status=st, done_time=time.time())
                save_ledger(L); print(f"收到 #{o:03d} status={st} shots={n}", flush=True)
        if time.time() - t0 > timeout: print("等待超时，稍后再运行 --run 继续收取"); return False
        time.sleep(15)


def run(chip_arg):
    tok = token()
    if not tok: sys.exit("没有找到 QUAFU_API_TOKEN（环境变量或 .env），未提交任何任务")
    os.environ["QPU_API_TOKEN"] = tok
    L = load_ledger()
    if not L.get("_layout"):
        pick = preflight(chip_arg)
        if pick is None: sys.exit("两块芯片都没有可用的 8 比特链：未提交任何任务。请把 live 文件夹发回 Claude。")
        L["_layout"] = pick; save_ledger(L)
    from quark import Task
    mgr = Task(tok)
    diags = [t for t in MAN["tasks"] if t["kind"] == "diagnostic"]
    for t in diags: submit(mgr, t, L)
    if not collect(mgr, [t["order"] for t in diags], L): return
    ok, order = True, None
    for t in diags:
        cnt = norm_counts(json.load(open(RAW / f"{t['order']:03d}.json", encoding="utf-8")))
        top = max(cnt, key=cnt.get) if cnt else ""
        b = t["bits_c0_to_c7"]; k_rev = "".join(map(str, b[::-1])); k_fwd = "".join(map(str, b))
        this = "c7…c0" if top == k_rev else ("c0…c7" if top == k_fwd else None)
        print(f"诊断 {t['name']}: 实测最多 {top} 占比 {cnt.get(top, 0) / MAN['shots']:.2f}，判定位序 {this}")
        ok &= this is not None and (order is None or order == this); order = order or this
    L["_diagnostics_passed"] = ok; L["_bit_order"] = order; save_ledger(L)
    if not ok: sys.exit("位序诊断不通过：按预案停止，未提交其余任务。请把 live 文件夹发回 Claude。")
    rest = [t for t in MAN["tasks"] if t["kind"] != "diagnostic"]
    for t in rest: submit(mgr, t, L)
    if collect(mgr, [t["order"] for t in rest], L):
        print("全部完成。请告诉 Claude：泡利关联编码真机展示 pce8 结果已回收（live 文件夹）。")


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
