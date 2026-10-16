"""东岭全规模决定性实验；固定每实例重复0/1，三批读出，禁止不确定提交重试。"""
import sys,os,json,hashlib,contextlib,io,re,ast
from pathlib import Path
HERE=Path(__file__).resolve().parent
ROOT=HERE.parent.parent
sys.path.insert(0,str(HERE.parent/'pce_cmaes_fixed12_20261014/hardware_prepared/.sdk'))
import numpy as np
import importlib.util
spec=importlib.util.spec_from_file_location('old_hw',HERE.parent/'pce_dongling_direct4_20261014/hw.py')
H=importlib.util.module_from_spec(spec);spec.loader.exec_module(H)
OUT=HERE/'hardware';OUT.mkdir(exist_ok=True)
H.HERE=OUT;H.CHIP='Dongling';H.N=8
from quark import Task
original_request=Task.session.request
def bounded(method,url,**kw):
    kw.setdefault('timeout',(15,45));return original_request(method,url,**kw)
Task.session.request=bounded

def save(name,x):H.save(OUT/name,x)
def qasm(theta,basis):
    a=np.array(theta).reshape(4,8,2)
    lines=['OPENQASM 2.0;','include "qelib1.inc";','qreg q[8];','creg c[8];']
    for l in range(4):
        for q in range(8):lines.extend([f'ry({a[l,q,0]:.17g}) q[{q}];',f'rz({a[l,q,1]:.17g}) q[{q}];'])
        lines.extend(f'cz q[{q}],q[{q+1}];' for q in range(l%2,7,2))
    if basis=='Y':lines.extend(f'sdg q[{q}];' for q in range(8))
    if basis in 'XY':lines.extend(f'h q[{q}];' for q in range(8))
    lines.extend(f'measure q[{q}] -> c[{q}];' for q in range(8))
    return '\n'.join(lines)+'\n'

def independent_simulator():
    source=(HERE.parent/'pce_assignment_decisive_20261014/hardware_assignment.py').read_text(encoding='utf-8')
    node=next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name=='simulate_qasm')
    env={'np':np,'re':re};exec(compile(ast.Module(body=[node],type_ignores=[]),'<independent_qasm_simulator>','exec'),env)
    return env['simulate_qasm']

def prepare():
    if (OUT/'manifest.json').exists():raise RuntimeError('已冻结真机清单，不覆盖')
    protocol=H.read(HERE/'protocol.json');names=protocol['instances']
    save('protocol.json',dict(hypothesis='8比特真实路线训练在30至44客户完整分车上，输出价值可在东岭读出后保留。',
        instances=names,selected_training_repeats=[0,1],selection='预先固定重复编号，含不可行结果；不按训练或真机成绩筛选。',
        repeats=3,shots_per_basis=4096,bases=['Z','X','Y'],raw_primary=True,
        corrections='读出校正单列；原始每批单独报告，不以跨批合并掩盖失败；跨批合并另作诊断。',
        total_tasks=117,total_measurements=479232,refinement='同一10000候选上限，原始分车与精修结果分别报告。'))
    sources={};logical={}
    for name in names:
        for rep in (0,1):
            p=HERE/'runs'/f'{name}_{rep:02d}.json';r=H.read(p)['quantum'];key=f'{name}_s{rep}'
            sources[key]=r
            for b in 'ZXY':logical[f'{key}_{b}']=qasm(r['theta'],b)
    for label,pattern in [('zero','00000000'),('one','11111111'),('order','00101101')]:
        lines=['OPENQASM 2.0;','include "qelib1.inc";','qreg q[8];','creg c[8];']
        lines.extend(f'x q[{q}];' for q,v in enumerate(pattern) if v=='1')
        lines.extend(f'measure q[{q}] -> c[{q}];' for q in range(8))
        logical[label]='\n'.join(lines)+'\n'
    H.manager();backend,cal=H.backend()
    from compile_physical import quality_path
    from quark.circuit import Transpiler
    path,edges,_,_=quality_path(cal,8)
    save('calibration_at_compile.json',cal)
    sim=independent_simulator();templates={}
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
        p=OUT/'physical'/f'{label}.qasm';p.parent.mkdir(exist_ok=True);p.write_text(physical,encoding='utf-8')
        templates[label]=dict(file=str(p.relative_to(OUT)),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),
            physical_two_qubit_gates=len(gates),measurements=meas,probability_error=error)
        print('compiled',label,len(gates),error,flush=True)
    tasks=[];science=[x for x in logical if x not in ('zero','one','order')]
    for batch in range(3):
        order=['zero','one','order']+(science if batch%2==0 else list(reversed(science)))
        for label in order:tasks.append(dict(label=f'{label}_b{batch}',template=label,batch=batch,**templates[label]))
    save('logical_tasks.json',dict(nq=8,sources=sources,logical=logical))
    save('manifest.json',dict(created_utc=H.now(),chip='Dongling',shots=4096,path=path,tasks=tasks,
        max_probability_error=max(x['probability_error'] for x in tasks)))
    print('READY',len(tasks),path,flush=True)

def submit():
    m=H.read(OUT/'manifest.json');lp=OUT/'ledger.json';ledger=H.read(lp) if lp.exists() else {};mgr=H.manager()
    for t in m['tasks']:
        label=t['label']
        if label in ledger:
            if not ledger[label].get('task_id'):raise RuntimeError('已有不确定提交状态，须查询，不自动重试')
            continue
        p=OUT/t['file'];assert hashlib.sha256(p.read_bytes()).hexdigest()==t['sha256']
        ledger[label]=dict(status='SUBMITTING',utc=H.now());H.save(lp,ledger)
        payload=dict(chip='Dongling',name='PCE_ROUTE_VALUE_20261014_'+label,circuit=p.read_text(encoding='utf-8'),
            shots=4096,compile=False,options=dict(compiler=None,correct=False,open_dd=None,target_qubits=[]))
        try:
            with contextlib.redirect_stdout(io.StringIO()),contextlib.redirect_stderr(io.StringIO()):tid=mgr.run(payload)
            assert isinstance(tid,int) and tid>0
        except Exception as exc:
            ledger[label].update(status='UNKNOWN',error=type(exc).__name__);H.save(lp,ledger)
            raise RuntimeError('提交状态不确定，已保存，禁止自动重试') from None
        ledger[label].update(status='SUBMITTED',task_id=tid,submitted_utc=H.now());H.save(lp,ledger)
        print('submitted',label,tid,flush=True)

if __name__=='__main__':
    try:{'status':H.status,'prepare':prepare,'submit':submit,'collect':lambda:H.collect(False)}[sys.argv[1]]()
    except Exception as exc:
        print('ERROR',type(exc).__name__,flush=True);raise SystemExit(1)
