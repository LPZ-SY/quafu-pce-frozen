"""最新修正版与同初始化未训练参数的多设备对照；显式分离准备、提交、回收。"""
import sys,json,hashlib,contextlib,io,re,importlib.util
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent
OLD=HERE.parent/'pce_random_value_20261014'
CHIP=sys.argv[2] if len(sys.argv)>2 else 'Dongling'
FOLDERS={'Dongling':'hardware','Baihua':'hardware_baihua','Shenglian':'hardware_shenglian'}
if CHIP not in FOLDERS:raise ValueError('不支持的设备')
spec=importlib.util.spec_from_file_location('stable_old_hardware_helpers',OLD/'hardware.py')
B=importlib.util.module_from_spec(spec);spec.loader.exec_module(B)
H=B.H
OUT=HERE/FOLDERS[CHIP];OUT.mkdir(exist_ok=True)
H.HERE=OUT
H.CHIP=CHIP

def prepare():
    if (OUT/'manifest.json').exists():raise RuntimeError('本批线路已冻结，不覆盖')
    data=H.read(HERE/'inputs.json');protocol=H.read(HERE/'protocol.json')
    stable_hash=hashlib.sha256((OLD/'stability_results.json').read_bytes()).hexdigest()
    assert stable_hash in data['hashes'].values()
    logical={};sources={};paired=[]
    for key,src in data['sources'].items():
        for arm in ('stable','untrained'):
            k=f'{key}_{arm}';theta=src['theta'] if arm=='stable' else src['untrained_theta']
            sources[k]=dict(source=key,arm=arm,theta=theta,instance=src['instance'],
                           slots=src['slots'],target_outputs=src['outputs'])
            for basis in 'ZXY':logical[f'{k}_{basis}']=B.qasm(theta,basis)
        for basis in 'ZXY':paired.extend([f'{key}_stable_{basis}',f'{key}_untrained_{basis}'])
    for label,pattern in [('zero','00000000'),('one','11111111'),('order','00101101')]:
        lines=['OPENQASM 2.0;','include "qelib1.inc";','qreg q[8];','creg c[8];']
        lines.extend(f'x q[{q}];' for q,v in enumerate(pattern) if v=='1')
        lines.extend(f'measure q[{q}] -> c[{q}];' for q in range(8))
        logical[label]='\n'.join(lines)+'\n'
    backend,cal=H.backend()
    from compile_physical import quality_path
    from quark.circuit import Transpiler
    path,edges,_,_=quality_path(cal,8)
    H.save(OUT/'calibration_at_compile.json',cal)
    sim=B.independent_simulator();templates={}
    for label,txt in logical.items():
        with contextlib.redirect_stdout(io.StringIO()),contextlib.redirect_stderr(io.StringIO()):
            compiled=Transpiler(backend).run(txt,target_qubits=path,optimize_level=1,niter=5)
        physical=str(compiled.to_openqasm2)
        meas=re.findall(r'measure\s+q\[(\d+)\]\s*->\s*c\[(\d+)\]',physical)
        assert sorted(int(c) for _,c in meas)==list(range(8))
        assert sorted(int(q) for q,_ in meas)==sorted(path)
        gates=[tuple(sorted(map(int,x))) for x in re.findall(r'(?:cz|cx)\s+q\[(\d+)\]\s*,\s*q\[(\d+)\]',physical)]
        assert all(g in edges for g in gates)
        error=float(np.max(np.abs(sim(physical,meas)-sim(txt,[(i,i) for i in range(8)]))))
        assert error<1e-9
        if label not in ('zero','one','order'):assert len(gates)==14
        p=OUT/'physical'/f'{label}.qasm';p.parent.mkdir(exist_ok=True);p.write_text(physical,encoding='utf-8')
        templates[label]=dict(file=str(p.relative_to(OUT)),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),
                             physical_two_qubit_gates=len(gates),measurements=meas,probability_error=error)
        print('compiled',label,'gates',len(gates),'error',error,flush=True)
    tasks=[]
    for batch in range(3):
        order=['zero','one','order']+(paired if batch%2==0 else list(reversed(paired)))
        for label in order:tasks.append(dict(label=f'{label}_b{batch}',template=label,batch=batch,**templates[label]))
    assert len(tasks)==protocol['hardware']['total_tasks']==225
    H.save(OUT/'logical_tasks.json',dict(nq=8,sources=sources,logical=logical,
            stable_parameter_sha256=stable_hash,inputs_sha256=hashlib.sha256((HERE/'inputs.json').read_bytes()).hexdigest()))
    H.save(OUT/'manifest.json',dict(created_utc=H.now(),chip=CHIP,shots=4096,path=path,tasks=tasks,
           total_measurements=921600,stable_parameter_sha256=stable_hash,
           max_probability_error=max(t['probability_error'] for t in tasks),
           description='最新稳定性修正版12条与对应原始随机参数12条，三批、每方向4096次；校验9项。'))
    H.save(OUT/'device_protocol.json',dict(chip=CHIP,total_tasks=225,total_measurements=921600,
           stable_parameter_sha256=stable_hash,
           inputs_sha256=hashlib.sha256((HERE/'inputs.json').read_bytes()).hexdigest(),
           shared_protocol_sha256=hashlib.sha256((HERE/'protocol.json').read_bytes()).hexdigest(),
           note='复用同一最新修正版与随机参数对照；各设备分别编译、提交、回收、统计，不合并读数。'))
    print('READY',CHIP,len(tasks),'measurements',921600,'path',path,flush=True)

def submit():
    m=H.read(OUT/'manifest.json');ledger_path=OUT/'ledger.json'
    ledger=H.read(ledger_path) if ledger_path.exists() else {}
    assert m['stable_parameter_sha256']==hashlib.sha256((OLD/'stability_results.json').read_bytes()).hexdigest()
    mgr=H.manager()
    for task in m['tasks']:
        label=task['label']
        if label in ledger:
            if not ledger[label].get('task_id'):raise RuntimeError('已有不确定提交，不重复发送')
            continue
        p=OUT/task['file'];assert hashlib.sha256(p.read_bytes()).hexdigest()==task['sha256']
        ledger[label]=dict(status='SUBMITTING',utc=H.now());H.save(ledger_path,ledger)
        payload=dict(chip=CHIP,name='PCE_STABLE_CONTROLS_20261015_'+CHIP+'_'+label,
                     circuit=p.read_text(encoding='utf-8'),shots=4096,compile=False,
                     options=dict(compiler=None,correct=False,open_dd=None,target_qubits=[]))
        try:
            with contextlib.redirect_stdout(io.StringIO()),contextlib.redirect_stderr(io.StringIO()):
                tid=mgr.run(payload)
            assert isinstance(tid,int) and tid>0
        except Exception as exc:
            ledger[label].update(status='UNKNOWN',error=type(exc).__name__);H.save(ledger_path,ledger)
            raise RuntimeError('提交状态不确定；已保存，禁止自动重试') from None
        ledger[label].update(status='SUBMITTED',task_id=tid,submitted_utc=H.now());H.save(ledger_path,ledger)
        print('submitted',label,tid,flush=True)

def collect():
    manifest=H.read(OUT/'manifest.json');lp=OUT/'ledger.json';ledger=H.read(lp)
    assert len(ledger)==len(manifest['tasks'])==225
    assert all(e.get('task_id') for e in ledger.values())
    mgr=H.manager();queried=0;errors=0
    for task in manifest['tasks']:
        label=task['label'];entry=ledger[label]
        if entry.get('status') in ('DONE','FAILED'):continue
        queried+=1
        try:
            with contextlib.redirect_stdout(io.StringIO()),contextlib.redirect_stderr(io.StringIO()):
                raw=mgr.result(entry['task_id'])
        except Exception:
            errors+=1;continue
        if isinstance(raw,dict):
            complete=bool(raw.get('count') or raw.get('counts'))
            failed=str(raw.get('status','')).lower() in ('failed','error','cancelled')
            if complete or failed:
                H.save(OUT/'raw'/f'{label}.json',raw)
                entry.update(status='DONE' if complete else 'FAILED',received_utc=H.now())
                H.save(lp,ledger)
        if queried%25==0:print('collected',CHIP,queried,'done',sum(e['status']=='DONE' for e in ledger.values()),flush=True)
    progress=dict(utc=H.now(),chip=CHIP,total=len(manifest['tasks']),queried=queried,query_errors=errors,
                  done=sum(e['status']=='DONE' for e in ledger.values()),
                  failed=sum(e['status']=='FAILED' for e in ledger.values()))
    H.save(OUT/'collection_status.json',progress)
    print(json.dumps(progress,ensure_ascii=False),flush=True)

if __name__=='__main__':
    try:
        {'prepare':prepare,'submit':submit,'status':H.status,'collect':collect}[sys.argv[1]]()
    except Exception as exc:
        print('ERROR',type(exc).__name__,flush=True);raise SystemExit(1)
