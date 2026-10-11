"""泡利关联编码 pce6：有纠缠与无纠缠的公平比较（6 比特链，10 个 CZ，单比特相位补偿）
流程（双击 run_pce6.bat 即可）：普查 Baihua、Dongling 的门相位 → 选 6 比特链（条件相位 ≤0.3）→ 正式检测（最多 4 轮）
→ 用正式检测测得的单比特相位做补偿 → 位序诊断 → 240 个实验任务 + 4 个读出标定 → 收取。
    python submit_pce6.py --check     # 只做本地检查，不联网
    python submit_pce6.py --all       # 普查 + 选链检测 + 实验，一次完成
"""
import argparse, hashlib, json, os, sys, time, math, re
from pathlib import Path
import numpy as np

PKG = Path(__file__).resolve().parent
sys.path.insert(0, str(PKG))
import edgecheck6 as EC
import survey as SV
LIVE = PKG / os.environ.get("PCE_LIVE", "live_pce6"); RAW = LIVE / "raw"; RAWC = LIVE / "raw_check"; RAWS = LIVE / "raw_survey"; PHY = LIVE / "physical"
MAN = json.load(open(PKG / "task_manifest.json", encoding="utf-8"))
N = MAN["n_qubits"]; CHIPS = ["Baihua", "Dongling", "Shenglian"]; DEFAULT_CHIPS = ["Baihua", "Dongling"]; CHECK_REPS = 2; MAX_ROUNDS = 4; POLL = float(os.environ.get("PCE_POLL", "15"))
USE = {}
for inst in MAN["instances"]:
    cnt = {}
    for l in range(MAN["layers"]):
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


def dead_qubits(info):
    """标定里 T1 或单比特保真度为 0 的比特（例如 Baihua 的 Q66、Q111）：这些比特不能用。"""
    return {int(k[1:]) for k, v in info["qubits_info"].items()
            if not (v.get("fidelity") or 0) > 0 or not (v.get("T1") or 0) > 0}


def edges_of(info, keep_dead=False):
    """标定里保真度 > 0 的边。默认去掉连着坏比特的边（耦合器保真度 > 0 但比特本身不可用，任务会直接失败）。"""
    E = {}; dead = set() if keep_dead else dead_qubits(info)
    for c in info["couplers_info"].values():
        a, b = c["qubits_index"]; f = c.get("fidelity") or 0
        if 0 < f <= 1 and a not in dead and b not in dead: E[(min(a, b), max(a, b))] = f
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
        b = [int(k[N - 1 - i]) for i in range(N)] if order == "c_last_first" else [int(k[i]) for i in range(N)]
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
        keys.append((run_task(mgr, key, f"QRF_pce6_{key}", chip, to_physical(txt, phys), L, RAW if L.get("_layout") else RAWC), t))
    if not collect(mgr, [k for k, _ in keys], L): return None
    order = None
    for k, t in keys:
        cnt = norm_counts(json.load(open(LIVE / L[k]["rawdir"] / f"{k}.json", encoding="utf-8")))
        top = max(cnt, key=cnt.get) if cnt else ""
        b = t["bits_c0_first"]; this = "c_last_first" if top == "".join(map(str, b[::-1])) else ("c_first_first" if top == "".join(map(str, b)) else None)
        print(f"  位序诊断 {t['name']}: 实测最多 {top} 占比 {cnt.get(top, 0) / MAN['shots']:.2f} → {this}")
        if this is None or (order and order != this): return None
        order = this
    return order


def notify(title, msg):
    """在 Windows 上弹窗并响铃（不阻塞脚本）；在服务器上只打印。"""
    print(f"\n========== {title} ==========\n{msg}\n", flush=True)
    try:
        import threading, ctypes, winsound
        winsound.MessageBeep()
        threading.Thread(target=lambda: ctypes.windll.user32.MessageBoxW(0, msg, title, 0x40 | 0x1000), daemon=True).start()
    except Exception:
        pass


def load_infos(chips):
    from quark.circuit import Backend
    infos = {}; LIVE.mkdir(parents=True, exist_ok=True)
    for chip in chips:
        f = LIVE / f"{chip}_calibration_at_submit.json"
        if f.exists():
            infos[chip] = json.load(open(f, encoding="utf-8")); continue
        try: infos[chip] = dict(Backend(chip).chip_info)
        except Exception as ex: print(chip, "读取标定失败：", ex); continue
        json.dump(infos[chip], open(f, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        bg = set(infos[chip].get("basis_gates") or [])
        if bg and not {"h", "ry", "rz", "cz"} <= bg: print(chip, "基础门不含 h/ry/rz/cz：", bg); del infos[chip]
    return infos


MAX_EDGES = 47      # 每个普查任务最多 47 条边（94 个比特）：Baihua 上 126 比特 / 63 条边的任务全部失败，94 比特的正常


def surv_ok(r):
    """普查判据：条件相位 ≤ COND_MAX 且单比特相位 ≤ SINGLE_MAX（单比特相位会在实验中补偿）。"""
    return max(abs(r["cond_dev_first"]), abs(r["cond_dev_second"])) <= EC.COND_MAX and \
        max(abs(r["single_dev_first"]), abs(r["single_dev_second"])) <= EC.SINGLE_MAX


def plan_groups(edges, dead=()):
    """先按互不共用比特分组，再把超过 MAX_EDGES 的组切成几块，最后去掉连着坏比特的边。返回 [(名字, 边列表)]。
    分组用的是全部边（和之前一样），这样已经成功返回的组保持原样、可以沿用；
    去掉了坏边的组改名（加 x），不会和之前失败的任务混在一起。"""
    out = []
    for gi, g in enumerate(SV.color_edges(edges)):
        if len(g) <= MAX_EDGES: parts = [(f"g{gi}", g)]
        else:
            k = -(-len(g) // MAX_EDGES); size = -(-len(g) // k)
            parts = [(f"g{gi}{'abcdefgh'[j]}", g[j * size:(j + 1) * size]) for j in range(k)]
        for name, p in parts:
            keep = [e for e in p if e[0] not in dead and e[1] not in dead]
            if keep: out.append((name if len(keep) == len(p) else name + "x", keep))
    return out


def submit_survey(mgr, L, infos):
    """把所有芯片的普查任务一次性交上去（不等结果）。已经成功返回的任务会沿用，不会重交。"""
    L.setdefault("_survey", {})
    for chip, info in infos.items():
        sv = L["_survey"].setdefault(chip, {})
        if sv.get("done"): continue
        nq = max(int(k[1:]) for k in info["qubits_info"]) + 1
        dead = dead_qubits(info)
        plan = plan_groups(sorted(edges_of(info, keep_dead=True)), dead); edges = sorted(edges_of(info))
        if dead: print(f"{chip}：标定显示不可用的比特 {sorted(dead)}，连着它们的边不检测", flush=True)
        big = max((g for _, g in plan), key=len); qs_big = sorted({q for e in big for q in e})
        # 位序诊断：上一次已成功、且比特数相同的诊断任务直接沿用；否则换一个没用过的名字重新交
        old = sv.get("dkey")
        if old and L.get(old, {}).get("status") == "DONE" and sv.get("diag_qs") == qs_big: dkey = old
        else:
            j = 1
            while f"sv_{chip}_diag{'' if j == 1 else j}" in L: j += 1
            dkey = f"sv_{chip}_diag{'' if j == 1 else j}"
        sv.update(groups=[[list(e) for e in g] for _, g in plan], group_names=[n for n, _ in plan], n_edges=len(edges),
                  diag_qs=qs_big, dkey=dkey, jobs={}, dead=sorted(dead))
        run_task(mgr, dkey, f"QRF_pce6_{dkey}", chip, SV.basis_circuit(qs_big, nq, {0, 1, 2}), L, RAWS)
        for gi, (name, g) in enumerate(plan):
            for st in SV.SETTINGS:
                txt, qs = SV.circuit(g, st, nq); key = f"sv_{chip}_{name}_{st[0]}{st[1]}"
                sv["jobs"][key] = [gi, st[0], st[1], qs]; run_task(mgr, key, f"QRF_pce6_{key}", chip, txt, L, RAWS)
        save_ledger(L); print(f"{chip}：普查任务 {len(sv['jobs']) + 1} 个（已返回的沿用，其余已提交）", flush=True)


def poll_once(mgr, keys, L):
    for k in [k for k in keys if L[k].get("status") not in ("DONE", "FAILED")]:
        e = L[k]; rd = LIVE / e["rawdir"]; rd.mkdir(parents=True, exist_ok=True)
        try: r = mgr.result(e["task_id"])
        except Exception as ex: print("查询出错", k, ex); continue
        if not isinstance(r, dict): continue
        st = str(r.get("status", "")).lower(); cnt = norm_counts(r)
        if cnt or st in ("finished", "completed", "done", "success", "failed", "error", "cancelled", "fail"):
            json.dump(r, open(rd / f"{k}.json", "w", encoding="utf-8"), ensure_ascii=False, default=str)
            n = sum(cnt.values())
            e.update(status="DONE" if n == MAN["shots"] else "FAILED", shots_returned=n, platform_status=st, done_time=time.time())
            save_ledger(L)
        elif e.get("last_platform_status") != st:          # 平台返回了没见过的状态：记下来，方便以后识别
            e["last_platform_status"] = st; save_ledger(L)
            json.dump(r, open(rd / f"{k}.pending.json", "w", encoding="utf-8"), ensure_ascii=False, default=str)
    return [k for k in keys if L[k].get("status") not in ("DONE", "FAILED")]


STALL_MIN = float(os.environ.get("PCE_STALL_MIN", "30"))


def mark_stalled(keys, L):
    """同一芯片的其它任务都已返回，而某个任务交上去超过 STALL_MIN 分钟、且比最后一个返回的任务晚了 STALL_MIN 分钟还没结果：
    平台上这种任务实际是 Failed（查询接口不报失败），按失败处理。"""
    done_t = [L[k].get("done_time", 0) for k in keys if L[k].get("status") == "DONE"]
    pend = [k for k in keys if L[k].get("status") not in ("DONE", "FAILED")]
    if not pend or not done_t: return []                 # 一个都没返回时可能只是排队，不判
    now = time.time(); last = max(done_t)
    out = [k for k in pend if now - L[k].get("submit_time", now) > STALL_MIN * 60 and now - last > STALL_MIN * 60]
    for k in out: L[k].update(status="FAILED", platform_status="stalled", done_time=now)
    if out: save_ledger(L)
    return out


def process_chip(chip, info, L, failed=()):
    sv = L["_survey"][chip]; groups = [[tuple(e) for e in g] for g in sv["groups"]]
    cnt = norm_counts(json.load(open(RAWS / f"{sv['dkey']}.json", encoding="utf-8")))
    n = len(sv["diag_qs"]); order = None; rd = ""
    # 几十个比特同时读出时，整串完全读对的次数很少，所以逐个比特判断：设为 1 的 3 个比特多数读到 1、其余多数读到 0
    for o in ("c_last_first", "c_first_first"):
        m = SV.bits(cnt, n, o).mean(0)
        if m[:3].min() > 0.5 and np.median(m[3:]) < 0.5:
            order = o; noisy = [sv["diag_qs"][i] for i in range(3, n) if m[i] > 0.25]
            rd = f"；全 0 比特读成 1 的比例中位 {np.median(m[3:]):.3f}，超过 25% 的比特：{noisy}"
    lines = [f"{chip} 普查（标定 {info.get('calibration_time')}）", f"位序诊断（逐比特）：{order}{rd}"]
    if sv.get("dead"): lines.append(f"标定显示不可用的比特：{sv['dead']}（连着它们的边不检测、不使用）")
    if failed: lines.append(f"有 {len(failed)} 个普查任务失败，这些组的边按不通过处理：{list(failed)}")
    if order is None:
        sv.update(done=True, ok=False); save_ledger(L); lines.append("位序诊断不通过，这块芯片不参与选链。")
    else:
        res = {}
        for gi, g in enumerate(groups):
            bb, qsg = {}, None
            for key, (gj, w, bs, qs) in sv["jobs"].items():
                if gj == gi and key not in failed:
                    try: bb[(w, bs)] = SV.bits(norm_counts(json.load(open(RAWS / f"{key}.json", encoding="utf-8"))), len(qs), order); qsg = qs
                    except Exception: pass
            if len(bb) == 4: res.update(SV.estimate(bb, g, qsg))
        good = {k for k, r in res.items() if surv_ok(r)}
        sv.update(done=True, ok=True, edges=res); save_ledger(L)
        devs = sorted(res.items(), key=lambda kv: -max(abs(kv[1]["cond_dev_first"]), abs(kv[1]["cond_dev_second"])))
        lines.append(f"{sv['n_edges']} 条边分 {len(groups)} 组；{len(good)} 条通过（条件相位 ≤{EC.COND_MAX}、单比特相位 ≤{EC.SINGLE_MAX} 弧度），{len(res) - len(good)} 条不通过，{sv['n_edges'] - len(res)} 条没有结果。")
        lines.append("条件相位偏差最大的 8 条边：" + "，".join(f"Q{k}（{max(abs(r['cond_dev_first']), abs(r['cond_dev_second'])):.2f}）" for k, r in devs[:8]))
        bad = {tuple(map(int, k.split("-"))) for k in res if k not in good} | {e for g in groups for e in g if f"{e[0]}-{e[1]}" not in res}
        r = best_chain(info, bad)
        lines.append(f"只用通过的边，最好的 6 比特链：{r['phys']}，CZ 总保真度 {r['circuit_cz_fidelity']:.3f}" if r else "只用通过的边，找不到 6 比特链。")
    txt = "\n".join(lines); (LIVE / f"survey_{chip}.txt").write_text(txt + "\n", encoding="utf-8")
    notify(f"{chip} 普查结果已出", txt)


def survey(mgr, L, chips, wait_hours=24):
    infos = load_infos(chips)
    submit_survey(mgr, L, infos)
    t0 = time.time(); last = 0
    while True:
        for c in infos:                                      # 旧版位序诊断误判的芯片，用新判断重新处理一次
            sv = L["_survey"][c]
            if sv.get("done") and not sv.get("ok") and not sv.get("rechecked"):
                sv["rechecked"] = True; sv["done"] = False; save_ledger(L)
        pending = [c for c in infos if not L["_survey"][c].get("done")]
        if not pending: break
        for c in pending:
            sv = L["_survey"][c]; ks = [sv["dkey"]] + list(sv["jobs"])
            stalled = mark_stalled(ks, L) if poll_once(mgr, ks, L) else []
            if stalled: print(time.strftime("%H:%M"), f"{c}：{stalled} 长时间没有结果，按失败处理", flush=True)
            if not [k for k in ks if L[k].get("status") not in ("DONE", "FAILED")]:
                failed = [k for k in ks if L[k].get("status") == "FAILED"]
                if sv["dkey"] in failed:
                    sv.update(done=True, ok=False, rechecked=True); save_ledger(L)
                    notify(f"{c} 普查有任务失败", f"{c} 的位序诊断任务失败：{failed}。请把 {LIVE.name} 文件夹发回 Claude。")
                else: process_chip(c, infos[c], L, failed)        # 失败的组没有结果，组里的边按不通过处理
        pending = [c for c in infos if not L["_survey"][c].get("done")]
        if not pending: break
        if time.time() - last > 300:
            last = time.time()
            for c in pending:
                sv = L["_survey"][c]; ks = [sv["dkey"]] + list(sv["jobs"])
                print(time.strftime("%H:%M"), f"{c} 普查：已完成 {sum(L[k].get('status') == 'DONE' for k in ks)}/{len(ks)}", flush=True)
        if time.time() - t0 > wait_hours * 3600: print("等待超时，稍后再运行继续收取。"); return False
        time.sleep(POLL)
    notify("普查完成", f"普查结果已写入 {LIVE.name} 文件夹（survey_*.txt），接着选链检测。")
    return True


def select_layout(mgr, L, chips):
    done = [c for c in chips if L.get("_survey", {}).get(c, {}).get("done")]
    for c in chips:
        if c not in done: print(f"{c} 普查还没有完成，这次不参与选链。")
    infos = load_infos(done)
    L.setdefault("_excluded", {c: [] for c in infos}); L.setdefault("_check_rounds", []); save_ledger(L)
    surv_bad = {}
    for chip in infos:
        sv = L["_survey"].get(chip, {})
        if not sv.get("ok"): surv_bad[chip] = None; continue
        good = {tuple(map(int, k.split("-"))) for k, r in sv["edges"].items() if surv_ok(r)}
        surv_bad[chip] = set(edges_of(infos[chip])) - good          # 普查不通过或没有结果的边都不用
    for rnd in range(len(L["_check_rounds"]) + 1, MAX_ROUNDS + 1):
        cands = []
        for chip, info in infos.items():
            if surv_bad.get(chip) is None: continue
            r = best_chain(info, {tuple(e) for e in L["_excluded"].get(chip, [])} | surv_bad[chip])
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
                    run_task(mgr, key, f"QRF_pce6_{key}", chip, to_physical(EC.circuit(layer, s), phys), L, RAWC)
        if not collect(mgr, list(keys), L): return None
        verdict, vlog, fails = {}, {}, []
        for layer in EC.LAYERS:
            bits = {}
            for s in EC.SETTINGS:
                arr = [bits_from(norm_counts(json.load(open(RAWC / f"{k}.json", encoding="utf-8"))), order)
                       for k, (ly, ss) in keys.items() if ly == layer and ss == s]
                bits[s] = np.vstack(arr)
            for (a, b), r in EC.estimate(bits, layer).items():
                pe = sorted((phys[a], phys[b])); verdict[f"{pe[0]}-{pe[1]}"] = r; vlog[f"{a}-{b}"] = r
                print(f"  边 Q{pe[0]}-Q{pe[1]}（逻辑 {a}-{b}）：条件相位偏差 {r['cond_dev_first']:+.2f}/{r['cond_dev_second']:+.2f}，"
                      f"单比特相位偏差 {r['single_dev_first']:+.2f}/{r['single_dev_second']:+.2f}（将补偿）→ {'通过' if r['pass'] else '不通过'}")
                if not r["pass"]: fails.append(pe)
        L["_check_rounds"].append(dict(round=rnd, chip=chip, phys=phys, bit_order=order, verdict=verdict, verdict_logical=vlog, fails=fails)); save_ledger(L)
        if not fails:
            pick["bit_order"] = order; pick["verdict_logical"] = vlog; return pick
        L["_excluded"].setdefault(chip, []).extend(fails); save_ledger(L)
        print(f"  有 {len(fails)} 条边不通过，排除后重新选链。")
    return None


def run(chips, force=False):
    tok = token()
    if not tok: sys.exit("没有找到 QUAFU_API_TOKEN（环境变量或 .env），未提交任何任务")
    os.environ["QPU_API_TOKEN"] = tok
    from quark import Task
    mgr = Task(tok); L = load_ledger()
    if not L.get("_layout") and force:
        # 用户决定：用最近一轮正式检测过的链直接跑，不要求 7 条边全部过线；在记录里写明哪些边没过
        if not L.get("_check_rounds"): sys.exit("还没有做过正式检测，不能直接跑。")
        r = L["_check_rounds"][-1]
        L["_layout"] = dict(chip=r["chip"], phys=r["phys"], bit_order=r["bit_order"], forced=True, verdict_logical=r["verdict_logical"],
                            gate_failed_edges={k: round(v["max_cond"], 3) for k, v in r["verdict"].items() if not v["pass"]},
                            note="正式检测未全部通过（条件相位门槛 0.3 弧度），按用户决定照常运行实验，结果需注明")
        save_ledger(L); print("按用户决定直接使用：", r["chip"], r["phys"], "未过线的边：", L["_layout"]["gate_failed_edges"])
    if not L.get("_layout"):
        pick = select_layout(mgr, L, chips)
        if pick is None:
            notify("pce6 正式检测未通过", "没有找到所有边都通过检测的 6 比特链（或诊断失败），未提交实验任务。请告诉 Claude。")
            sys.exit(f"没有找到所有边都通过检测的比特链（或诊断失败）：未提交实验任务。请把 {LIVE.name} 文件夹发回 Claude。")
        L["_layout"] = pick; save_ledger(L); print("选用：", pick["chip"], pick["phys"])
    chip, phys = L["_layout"]["chip"], L["_layout"]["phys"]
    vl = L["_layout"]["verdict_logical"]
    comp = {tuple(map(int, k.split("-"))): v for k, v in vl.items()}         # 逻辑边 → 正式检测结果（补偿用）
    print("单比特相位补偿（逻辑边：a 的补偿, b 的补偿）：", {k: (round(-v["single_dev_first"], 3), round(-v["single_dev_second"], 3)) for k, v in vl.items()})
    order = diagnostics(mgr, L, chip, phys)                       # 正式的 2 个位序诊断（任务 1、2）
    L["_bit_order"] = order; L["_diagnostics_passed"] = order is not None; save_ledger(L)
    if order is None: sys.exit(f"位序诊断不通过：按预案停止。请把 {LIVE.name} 文件夹发回 Claude。")
    rest = [t for t in MAN["tasks"] if t["kind"] != "diagnostic"]
    keys = []
    for t in rest:
        txt = (PKG / t["file"]).read_text(encoding="utf-8")
        assert hashlib.sha256(txt.encode()).hexdigest() == t["sha256"], "线路哈希不符，停止"
        key = f"{t['order']:03d}"
        if "cz " in txt: txt = EC.compensate(txt, comp)                    # 有纠缠电路：每层 CZ 后补偿单比特相位
        keys.append(run_task(mgr, key, f"QRF_pce6_{key}_{t['name']}", chip, to_physical(txt, phys), L, RAW))
    if collect(mgr, keys, L):
        notify("pce6 实验完成", f"全部 {len(keys) + 2} 个任务已返回。请告诉 Claude：pce6 实验结果已回收（{LIVE.name} 文件夹）。")


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
    ap.add_argument("--survey", action="store_true"); ap.add_argument("--force", action="store_true"); ap.add_argument("--all", action="store_true")
    ap.add_argument("--chips", nargs="+", choices=CHIPS, default=DEFAULT_CHIPS)
    a = ap.parse_args()
    if a.all:
        if check():
            tok = token()
            if not tok: sys.exit("没有找到 QUAFU_API_TOKEN（环境变量或 .env），未提交任何任务")
            os.environ["QPU_API_TOKEN"] = tok
            from quark import Task
            survey(Task(tok), load_ledger(), a.chips, wait_hours=2)
            run(a.chips, a.force)
    elif a.survey:
        if check():
            tok = token()
            if not tok: sys.exit("没有找到 QUAFU_API_TOKEN（环境变量或 .env），未提交任何任务")
            os.environ["QPU_API_TOKEN"] = tok
            from quark import Task
            survey(Task(tok), load_ledger(), a.chips)
    elif a.run:
        if check(): run(a.chips, a.force)
    else:
        check()
