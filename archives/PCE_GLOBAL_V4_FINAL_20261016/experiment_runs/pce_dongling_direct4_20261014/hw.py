"""东岭真机：查询、编译、提交、回收（在装有 quark 的环境中运行；只依赖 numpy 与 quark）。"""
import contextlib, io, json, hashlib, re, sys, time, os
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
ENV = HERE.parent.parent / '2.QuantumRouteForge：基于精简版增添实验及结果（选择观看）' / 'QuantumRouteForge-isolated-study' / '.env'
sys.path.insert(0, str(HERE.parent / 'cpdptw_pce_hardware_20261012'))
CHIP = os.environ.get('CHIP', 'Dongling')
SHOTS = 4096
N = 8


def now():
    return datetime.now(timezone.utc).isoformat()


def save(p, v):
    p.parent.mkdir(parents=True, exist_ok=True)
    t = p.with_suffix('.tmp')
    t.write_text(json.dumps(v, indent=1, ensure_ascii=False, default=str), encoding='utf-8')
    t.replace(p)


def read(p):
    return json.loads(p.read_text(encoding='utf-8'))


def token():
    for line in ENV.read_text(encoding='utf-8-sig').splitlines():
        if '=' in line and not line.lstrip().startswith('#'):
            k, v = line.split('=', 1)
            if k.strip() in ('QPU_API_TOKEN', 'QUAFU_API_TOKEN'):
                return v.strip().strip('"').strip("'")
    raise RuntimeError('缺少真机凭据')


def quiet():
    return contextlib.ExitStack()


def manager():
    from quark import Task
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        return Task(token())


def backend():
    os.environ['QPU_API_TOKEN'] = token()
    from quark.circuit import Backend
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        b = Backend(CHIP)
        cal = dict(b.chip_info)
    return b, cal


def status():
    mgr = manager()
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        q = mgr.status()
    print(json.dumps(q, ensure_ascii=False))
    save(HERE / 'queue_status.json', dict(utc=now(), queue=q))


def calibration():
    from compile_physical import quality_path
    b, cal = backend()
    save(HERE / f'calibration_{CHIP}.json', cal)
    path, edges, qubits, score = quality_path(cal, N)
    qi = {int(r['index']): r for r in cal['qubits_info'].values()}
    ef = [edges[tuple(sorted((a, c)))] for a, c in zip(path, path[1:])]
    info = dict(chip=CHIP, calibration_time=cal.get('calibration_time'), path=path,
                qubit_fidelity=[qi[q].get('fidelity') for q in path], T1=[qi[q].get('T1') for q in path],
                T2=[qi[q].get('T2') for q in path], coupler_fidelity=ef)
    print(json.dumps(info, ensure_ascii=False))
    save(HERE / f'chain_{CHIP}.json', info)


def compile_all():
    from quark.circuit import Transpiler
    from compile_physical import quality_path
    if (HERE / 'manifest.json').exists():
        raise RuntimeError('已有清单，不覆盖')
    b, cal = backend()
    path, edges, qubits, score = quality_path(cal, N)
    save(HERE / 'calibration_at_compile.json', cal)
    logical = read(HERE / 'logical_tasks.json')
    tasks = []
    for t in logical['tasks']:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            c = Transpiler(b).run(t['qasm'], target_qubits=path, optimize_level=1, niter=5)
        phys = str(c.to_openqasm2)
        meas = re.findall(r'measure\s+q\[(\d+)\]\s*->\s*c\[(\d+)\]', phys)
        assert sorted(int(x) for _, x in meas) == list(range(N))
        assert sorted(int(q) for q, _ in meas) == sorted(path)
        gates = [tuple(sorted(map(int, p))) for p in re.findall(r'(?:cz|cx)\s+q\[(\d+)\]\s*,\s*q\[(\d+)\]', phys)]
        assert all(g in edges for g in gates)
        f = HERE / 'physical' / f"{t['label']}.qasm"
        f.parent.mkdir(exist_ok=True)
        f.write_text(phys, encoding='utf-8')
        tasks.append(dict(label=t['label'], file=str(f.relative_to(HERE)), sha256=hashlib.sha256(f.read_bytes()).hexdigest(),
                          physical_two_qubit_gates=len(gates), measurements=meas))
    save(HERE / 'manifest.json', dict(created_utc=now(), chip=CHIP, shots=SHOTS, path=path,
         calibration_time=cal.get('calibration_time'), tasks=tasks,
         readout_mitigation='analysis only, from zero/one circuits'))
    print(json.dumps(dict(tasks=len(tasks), path=path,
                          max_2q=max(t['physical_two_qubit_gates'] for t in tasks)), ensure_ascii=False))


def submit():
    m = read(HERE / 'manifest.json')
    lp = HERE / 'ledger.json'
    ledger = read(lp) if lp.exists() else {}
    mgr = manager()
    for t in m['tasks']:
        if t['label'] in ledger:
            if not ledger[t['label']].get('task_id'):
                raise RuntimeError('已有不确定提交状态，不重复提交')
            continue
        f = HERE / t['file']
        assert hashlib.sha256(f.read_bytes()).hexdigest() == t['sha256']
        ledger[t['label']] = dict(status='SUBMITTING', utc=now())
        save(lp, ledger)
        payload = dict(chip=m['chip'], name=f"PCE_CVRP_DIRECT4_20261014_{t['label']}", circuit=f.read_text(encoding='utf-8'),
                       shots=SHOTS, compile=False, options=dict(compiler=None, correct=False, open_dd=None, target_qubits=[]))
        try:
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                tid = mgr.run(payload)
            assert isinstance(tid, int) and tid > 0
        except Exception as e:
            ledger[t['label']].update(status='UNKNOWN', error=type(e).__name__)
            save(lp, ledger)
            raise RuntimeError('提交状态不确定，已记入台账，不自动重试') from None
        ledger[t['label']].update(status='SUBMITTED', task_id=tid, submitted_utc=now())
        save(lp, ledger)
        print(json.dumps(dict(event='submitted', label=t['label'], task_id=tid)), flush=True)


def collect(loop=False):
    m = read(HERE / 'manifest.json')
    lp = HERE / 'ledger.json'
    mgr = manager()
    while True:
        ledger = read(lp)
        for t in m['tasks']:
            e = ledger.get(t['label'], {})
            if e.get('status') in ('DONE', 'FAILED') or not e.get('task_id'):
                continue
            try:
                with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                    raw = mgr.result(e['task_id'])
            except Exception:
                continue
            if not isinstance(raw, dict):
                continue
            if raw.get('count') or raw.get('counts'):
                save(HERE / 'raw' / f"{t['label']}.json", raw)
                e.update(status='DONE', received_utc=now())
            elif str(raw.get('status', '')).lower() in ('failed', 'error', 'cancelled'):
                save(HERE / 'raw' / f"{t['label']}.json", raw)
                e.update(status='FAILED', received_utc=now())
            ledger[t['label']] = e
        save(lp, ledger)
        done = sum(v.get('status') == 'DONE' for v in ledger.values())
        failed = sum(v.get('status') == 'FAILED' for v in ledger.values())
        print(json.dumps(dict(utc=now(), total=len(m['tasks']), done=done, failed=failed)), flush=True)
        if not loop or done + failed == len(m['tasks']):
            return
        time.sleep(60)


if __name__ == '__main__':
    a = sys.argv[1]
    try:
        {'status': status, 'calibration': calibration, 'compile': compile_all, 'submit': submit,
         'collect': lambda: collect(False), 'wait': lambda: collect(True)}[a]()
    except Exception as exc:
        print(json.dumps(dict(action=a, error_type=type(exc).__name__, message=str(exc)[:200]), ensure_ascii=False))
        raise SystemExit(1)
