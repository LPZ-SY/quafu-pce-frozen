"""冻结线路的真机任务包、既有权限核验、编译、提交和原始结果回收。"""
from core import *
import importlib.util,contextlib,io,re
from datetime import datetime,timezone
OUT=HERE/('hardware_scale' if 'scale_only' in sys.argv else 'hardware');OUT.mkdir(exist_ok=True)

def now():return datetime.now(timezone.utc).isoformat()

def helpers():
    sys.path.insert(0,str(HERE.parent/'pce_cmaes_fixed12_20261014/hardware_prepared/.sdk'))
    spec=importlib.util.spec_from_file_location('v4_existing_hardware',HERE.parent/'pce_dongling_direct4_20261014/hw.py')
    H=importlib.util.module_from_spec(spec);spec.loader.exec_module(H);H.HERE=OUT
    from quark import Task
    original=Task.session.request
    def bounded(method,url,**kw):
        kw.setdefault('timeout',(10,30));return original(method,url,**kw)
    Task.session.request=bounded
    return H

def package():
    if (OUT/'ledger.json').exists():raise RuntimeError('已有真机提交台账，禁止重写任务清单')
    sys.path.insert(0,str(HERE.parent/'pce_dongling_direct4_20261014'));import pce_k as K
    chosen=read(HERE/'PCE_GLOBAL_V4_FINAL.json')['implementation'];tasks=[];models={}
    prefix='final_' if chosen=='FINAL_JOINT' else ''
    specifications=[('B-n31-k5',prefix+'confirmation',range(2001,2004)),('A-n45-k7',prefix+'scale',range(3001,3004))]
    if 'primary_only' in sys.argv:specifications=specifications[:1]
    if 'scale_only' in sys.argv:specifications=specifications[1:]
    for name,phase,seeds in specifications:
        for sd in seeds:
            path=HERE/'models'/phase/name/f'{chosen}_{sd}'/'result.json';m=read(path)
            for arm in ('TRAINED_QPU','RANDOM_QPU'):
                key=f'{name}_{sd}_{arm}';th=m['theta'] if arm=='TRAINED_QPU' else m['initial_theta']
                models[key]=dict(model=str(path.relative_to(HERE)),model_sha256=digest(path),theta=th,seed=sd,instance=name,arm=arm,
                                 nq=m['nq'],layers=m['layers'],fams=m['mapping_fams'],masks=m['mapping_masks'],priorities=m['tie_priorities'])
                for basis,shot in zip('ZXY',allocation(4096)):
                    txt=K.qasm(th,m['nq'],m['layers'],basis);p=OUT/'logical'/f'{key}_{basis}.qasm';p.parent.mkdir(exist_ok=True);p.write_text(txt,encoding='utf-8')
                    for batch in range(8):tasks.append(dict(label=f'{key}_{basis}_{batch}',key=key,basis=basis,shots=int(shot),batch=batch,
                                                            file=str(p.relative_to(OUT)),sha256=digest(p),nq=m['nq']))
    # 每批固定独立位序与读出校验，测量次数另计，不混入科学组。
    sizes=sorted(set(t['nq'] for t in tasks))
    for nq in sizes:
        for kind,pattern in [('zero','0'*nq),('one','1'*nq),('order',('001011010')[:nq])]:
            key=f'control_{nq}_{kind}';p=OUT/'logical'/f'{key}.qasm'
            p.write_text(K.qasm_pattern(nq,pattern),encoding='utf-8')
            for batch in range(8):tasks.append(dict(label=f'{key}_{batch}',key=key,basis='Z',shots=256,batch=batch,
                file=str(p.relative_to(OUT)),sha256=digest(p),nq=nq,control=True,pattern=pattern))
    tasks.sort(key=lambda x:(x['batch'],not x.get('control',False),x['key'].split('_')[0],x['key'].split('_')[1],x['basis'],x['key']))
    save(OUT/'logical_manifest.json',dict(models=models,tasks=tasks,total_tasks=len(tasks),total_shots=sum(t['shots'] for t in tasks),
        per_instance_science_shots=196608,calibration_shots=6144*len(sizes),groups=['TRAINED_QPU','TRAINED_IDEAL_SHOT','RANDOM_QPU'],frozen_model_hash=digest(HERE/'PCE_GLOBAL_V4_FINAL.json'),
        note='每实例固定前三个确认种子；每参数8份独立完整读出；真机训练后与未训练组交错提交。模拟组另行按编译电路采样。'))
    print('逻辑任务包完成',len(tasks),sum(t['shots'] for t in tasks),flush=True)

def preflight():
    token=None
    try:
        H=helpers();token=H.token();mgr=H.manager()
        with contextlib.redirect_stdout(io.StringIO()),contextlib.redirect_stderr(io.StringIO()):
            status=mgr.status();backend,cal=H.backend()
        if not isinstance(status,dict):raise RuntimeError('平台返回非结构化状态，无法核实执行权限')
        save(OUT/'calibration.json',cal)
        result=dict(utc=now(),chip='Dongling',reachable=True,queue=status,n_qubits=len(cal.get('qubits_info',{})),submitted_tasks=0,
                    quota='既有接口未公开独立余额字段；本轮仅使用原凭据已有权限，不购买额度。')
    except Exception as e:
        message=str(e)
        if token:message=message.replace(token,'[已隐藏凭据]')
        result=dict(utc=now(),reachable=False,error_type=type(e).__name__,message=message[:600],submitted_tasks=0)
    save(OUT/'preflight.json',result);print(json.dumps(result,ensure_ascii=False),flush=True)
    return result

def compile_all():
    H=helpers();backend,cal=H.backend();source=read(OUT/'logical_manifest.json')
    sys.path.insert(0,str(HERE.parent/'cpdptw_pce_hardware_20261012'))
    from compile_physical import quality_path
    from quark.circuit import Transpiler
    # 复用既有独立的编译后电路模拟器，仅提取函数，无旧任务写入。
    import ast
    src=(HERE.parent/'pce_assignment_decisive_20261014/hardware_assignment.py').read_text(encoding='utf-8')
    fn=next(n for n in ast.parse(src).body if isinstance(n,ast.FunctionDef) and n.name=='simulate_qasm')
    env={'np':np,'re':re};exec(compile(ast.Module(body=[fn],type_ignores=[]),'<compiled_circuit_verifier>','exec'),env);simulate=env['simulate_qasm']
    layouts={};compiled={}
    for nq in sorted(set(t['nq'] for t in source['tasks'])):
        path,edges,_,_=quality_path(cal,nq);layouts[nq]=(path,edges)
    for task in source['tasks']:
        key=task['file']
        if key in compiled:continue
        text=(OUT/key).read_text(encoding='utf-8');nq=task['nq'];layout,edges=layouts[nq]
        with contextlib.redirect_stdout(io.StringIO()),contextlib.redirect_stderr(io.StringIO()):
            c=Transpiler(backend).run(text,target_qubits=layout,optimize_level=1,niter=5)
        physical=str(c.to_openqasm2);meas=re.findall(r'measure\s+q\[(\d+)\]\s*->\s*c\[(\d+)\]',physical)
        assert sorted(int(c) for _,c in meas)==list(range(nq))
        probs=simulate(physical,meas);reference=simulate(text,[(i,i) for i in range(nq)])
        error=float(np.max(np.abs(probs-reference)));assert error<1e-9
        gates=re.findall(r'(?:cz|cx)\s+q\[(\d+)\]\s*,\s*q\[(\d+)\]',physical)
        assert all(tuple(sorted(map(int,g))) in edges for g in gates)
        p=OUT/'physical'/Path(key).name;p.parent.mkdir(exist_ok=True);p.write_text(physical,encoding='utf-8')
        depths={}
        for line in physical.splitlines():
            if line.strip().startswith(('OPENQASM','include','qreg','creg','measure','barrier')):continue
            qs=[int(x) for x in re.findall(r'q\[(\d+)\]',line)]
            if qs:
                dep=1+max(depths.get(q,0) for q in qs)
                for q in qs:depths[q]=dep
        compiled[key]=dict(file=str(p.relative_to(OUT)),sha256=digest(p),nq=nq,layout=layout,measurements=meas,
                           two_qubit_gates=len(gates),circuit_depth=max(depths.values(),default=0),probability_error=error,ideal_probabilities=np.asarray(probs).tolist())
    tasks=[dict(t,**{k:v for k,v in compiled[t['file']].items() if k not in ('nq',)}) for t in source['tasks']]
    save(OUT/'compiled_manifest.json',dict(tasks=tasks,models=source['models'],layouts={str(k):v[0] for k,v in layouts.items()},
                                         total_shots=source['total_shots'],created=now()))
    print('编译核验完成',len(compiled),flush=True)

def submit():
    H=helpers();mgr=H.manager();manifest=read(OUT/'compiled_manifest.json');p=OUT/'ledger.json';ledger=read(p) if p.exists() else {}
    for task in manifest['tasks']:
        label=task['label']
        if label in ledger:
            if not ledger[label].get('task_id'):raise RuntimeError('已有不确定提交，必须查询，不自动重复提交')
            continue
        path=OUT/task['file'];assert digest(path)==task['sha256']
        ledger[label]=dict(status='SUBMITTING',submitted_utc=now());save(p,ledger)
        payload=dict(chip='Dongling',name='PCE_GLOBAL_V4_'+label,circuit=path.read_text(encoding='utf-8'),shots=task['shots'],compile=False,
                     options=dict(compiler=None,correct=False,open_dd=None,target_qubits=[]))
        try:
            with contextlib.redirect_stdout(io.StringIO()),contextlib.redirect_stderr(io.StringIO()):tid=mgr.run(payload)
            assert isinstance(tid,int) and tid>0
        except Exception as e:
            ledger[label].update(status='SUBMISSION_UNCERTAIN',error_type=type(e).__name__);save(p,ledger);raise
        ledger[label].update(status='SUBMITTED',task_id=tid);save(p,ledger)
    print('真机任务已提交',len(ledger),flush=True)

def collect():
    H=helpers();mgr=H.manager();p=OUT/'ledger.json';ledger=read(p)
    for label,entry in ledger.items():
        if entry['status'] in ('DONE','FAILED') or not entry.get('task_id'):continue
        try:
            with contextlib.redirect_stdout(io.StringIO()),contextlib.redirect_stderr(io.StringIO()):raw=mgr.result(entry['task_id'])
        except Exception:continue
        if not isinstance(raw,dict):continue
        if raw.get('count') or raw.get('counts'):
            save(OUT/'raw'/(label+'.json'),raw);entry.update(status='DONE',received_utc=now())
        elif str(raw.get('status','')).lower() in ('failed','error','cancelled'):
            save(OUT/'raw'/(label+'.json'),raw);entry.update(status='FAILED',received_utc=now())
    save(p,ledger);print('真机回收', {s:sum(e['status']==s for e in ledger.values()) for s in set(e['status'] for e in ledger.values())},flush=True)

if __name__=='__main__':
    action=sys.argv[1] if len(sys.argv)>1 else 'package'
    {'package':package,'preflight':preflight,'compile':compile_all,'submit':submit,'collect':collect}[action]()
