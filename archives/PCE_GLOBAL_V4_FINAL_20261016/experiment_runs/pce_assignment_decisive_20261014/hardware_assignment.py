"""真机直接分配读出：复用已有认证接口，所有结果存入本轮独立目录。"""
import argparse
import contextlib
import hashlib
import io
import json
from pathlib import Path
import re
import sys
import time
import numpy as np
from train_assignment import Circuit, assess, problem
from run_decisive import characters

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'pce_cvrp_encoding_20261013'))
import hardware_round as H
OUT=HERE/'hardware'
SHOTS=4096
SOURCE_RUNS=None


def save(path,value): H.save(path,value)
def read(path): return H.read(path)


def simulate_qasm(text,measurements):
    """按最终经典测量位把物理编号压紧，独立模拟已编译的基本门。"""
    index={int(q):int(c) for q,c in measurements}
    n=len(index)
    state=np.zeros(2**n,complex);state[0]=1
    basis=np.arange(2**n)
    def one(matrix,bit):
        nonlocal state
        low=basis[(basis>>bit&1)==0];high=low|(1<<bit)
        out=state.copy()
        out[low]=matrix[0,0]*state[low]+matrix[0,1]*state[high]
        out[high]=matrix[1,0]*state[low]+matrix[1,1]*state[high]
        state=out
    x=np.array([[0,1],[1,0]],complex);y=np.array([[0,-1j],[1j,0]]);z=np.diag([1,-1])
    gates={'x':x,'y':y,'z':z,'h':np.array([[1,1],[1,-1]])/np.sqrt(2),'s':np.diag([1,1j]),'sdg':np.diag([1,-1j]),'id':np.eye(2)}
    for statement in text.split(';'):
        line=statement.strip()
        if not line or line.startswith(('OPENQASM','include','qreg','creg','measure','barrier','//')): continue
        match=re.fullmatch(r'(\w+)(?:\(([^)]+)\))?\s+q\[(\d+)\](?:\s*,\s*q\[(\d+)\])?',line)
        if not match: raise ValueError('无法独立核验的量子门语句：'+line)
        gate,angle,qa,qb=match.groups();bit=index[int(qa)]
        if gate in ('cz','cx'):
            other=index[int(qb)]
            if gate=='cz': state*=1-2*((basis>>bit&1)*(basis>>other&1))
            else:
                permutation=basis.copy()
                permutation[(basis>>bit&1)==1]^=1<<other
                state=state[permutation]
        elif gate in gates: one(gates[gate],bit)
        elif gate in ('rx','ry','rz'):
            theta=float(angle)
            generator={'rx':x,'ry':y,'rz':z}[gate]
            one(np.cos(theta/2)*np.eye(2)-1j*np.sin(theta/2)*generator,bit)
        elif gate in ('u','u3'):
            t,p,l=map(float,angle.split(','))
            one(np.array([[np.cos(t/2),-np.exp(1j*l)*np.sin(t/2)],
                          [np.exp(1j*p)*np.sin(t/2),np.exp(1j*(p+l))*np.cos(t/2)]]),bit)
        elif gate in ('u1','p'): one(np.diag([1,np.exp(1j*float(angle))]),bit)
        else: raise ValueError('未实现量子门模拟：'+gate)
    return np.abs(state)**2


def prepare(chip):
    if (OUT/'manifest.json').exists(): raise RuntimeError('已有真机实验包，不覆盖')
    H.manager()
    sys.path.insert(0,str(H.LEGACY))
    from compile_physical import quality_path
    from quark.circuit import Backend,Transpiler
    with contextlib.redirect_stdout(io.StringIO()),contextlib.redirect_stderr(io.StringIO()):
        backend=Backend(chip)
        calibration=dict(backend.chip_info)
    save(OUT/'calibration.json',calibration)
    runs=read(SOURCE_RUNS) if SOURCE_RUNS else read(HERE/'training'/'all_runs.json')
    # 按规模选择已公开的实例编号，两次重启按训练目标选优，包含失败个案。
    cases=[(10,100,'redundant'),(10,101,'redundant'),(10,100,'complement'),(10,101,'complement'),
           (16,200,'complement'),(24,300,'complement'),(40,400,'complement'),(60,500,'complement')]
    sources={}
    if SOURCE_RUNS:
        sources={r['id']:r for r in runs}
    else:
        for n,seed,model in cases:
            candidates=[r for r in runs if (r['n'],r['seed'],r['model'])==(n,seed,model)]
            source=min(candidates,key=lambda r:(r['result']['unique_penalty']+r['result']['overload'],r['result']['proxy'],-r['result']['min_margin']))
            sources[source['id']]=source
    sizes=sorted({s['q'] for s in sources.values()})
    physical_paths={}
    all_edges={}
    for q in sizes:
        physical_paths[q],all_edges[q],_,_=quality_path(calibration,q)
    logical={}
    for q in sizes:
        pattern=('0010110101010110')[:q]
        lines=['OPENQASM 2.0;','include "qelib1.inc";',f'qreg q[{q}];',f'creg c[{q}];']
        lines.extend(f'x q[{i}];' for i,b in enumerate(pattern) if b=='1')
        lines.extend(f'measure q[{i}] -> c[{i}];' for i in range(q))
        p=np.zeros(2**q);p[sum(int(b)<<i for i,b in enumerate(pattern))]=1
        logical[f'bit_order_q{q}']=(q,'\n'.join(lines)+'\n',p,pattern)
    for label,source in sources.items():
        circ=Circuit(source['q'],source['layers'])
        state=circ.forward(np.array(source['theta']))[0]
        p=state*state
        corr=characters(source['q'],source['mapping'])@p
        assert np.max(np.abs(corr-source['correlations']))<1e-10
        logical[label]=(source['q'],circ.qasm(source['theta']),p,None)
    compiled={}
    for label,(q,text,p,pattern) in logical.items():
        with contextlib.redirect_stdout(io.StringIO()),contextlib.redirect_stderr(io.StringIO()):
            result=Transpiler(backend).run(text,target_qubits=physical_paths[q],optimize_level=1,niter=5)
        physical=str(result.to_openqasm2)
        measurements=re.findall(r'measure\s+q\[(\d+)\]\s*->\s*c\[(\d+)\]',physical)
        assert sorted(int(c) for _,c in measurements)==list(range(q))
        assert sorted(int(a) for a,_ in measurements)==sorted(physical_paths[q])
        edges=[tuple(sorted(map(int,pair))) for pair in re.findall(r'(?:cz|cx)\s+q\[(\d+)\]\s*,\s*q\[(\d+)\]',physical)]
        assert all(edge in all_edges[q] for edge in edges)
        simulated=simulate_qasm(physical,measurements)
        error=float(np.max(np.abs(simulated-p)))
        assert error<1e-7,error
        for folder,content in (('logical',text),('physical',physical)):
            path=OUT/folder/(label+'.qasm');path.parent.mkdir(parents=True,exist_ok=True);path.write_text(content)
        compiled[label]=dict(q=q,file='physical/'+label+'.qasm',sha256=hashlib.sha256(physical.encode()).hexdigest(),
                             two_qubit_gates=len(edges),probability_max_error=error,pattern=pattern,shots=SHOTS)
    tasks=[dict(label=label,source_label=label,repetition=0,**compiled[label]) for label in logical if label.startswith('bit_order')]
    for repeat in range(3):
        tasks.extend(dict(label=label+f'_r{repeat+1}',source_label=label,repetition=repeat+1,**compiled[label]) for label in sources)
    save(OUT/'manifest.json',dict(chip=chip,sources=sources,tasks=tasks,physical_paths=physical_paths,
         calibration_time=calibration.get('calibration_time'),shots=SHOTS,repeats=3,total_measurements=len(tasks)*SHOTS,
         mode='模拟训练后固定线路真机读出；无硬件优化；无读出缓解；无容量修复',created_utc=H.now()))
    print(json.dumps(dict(prepared=len(tasks),measurements=len(tasks)*SHOTS,sizes=sizes)),flush=True)


def submit():
    manifest=read(OUT/'manifest.json')
    ledger=read(OUT/'ledger.json') if (OUT/'ledger.json').exists() else {}
    mgr=H.manager()
    with contextlib.redirect_stdout(io.StringIO()): queue=mgr.status()
    if not isinstance(queue.get(manifest['chip']),int): raise RuntimeError('设备不接受任务')
    for task in manifest['tasks']:
        label=task['label']
        if label in ledger:
            if not ledger[label].get('task_id'): raise RuntimeError('已有不确定提交，不重复提交')
            continue
        path=OUT/task['file']
        assert hashlib.sha256(path.read_bytes()).hexdigest()==task['sha256']
        ledger[label]=dict(status='SUBMITTING',utc=H.now());save(OUT/'ledger.json',ledger)
        payload=dict(chip=manifest['chip'],name='PCE_Assignment_20261014_'+label,circuit=path.read_text(),shots=SHOTS,compile=False,
                     options=dict(compiler=None,correct=False,open_dd=None,target_qubits=[]))
        try:
            with contextlib.redirect_stdout(io.StringIO()),contextlib.redirect_stderr(io.StringIO()): task_id=mgr.run(payload)
            if not isinstance(task_id,int) or task_id<=0: raise RuntimeError('没有有效任务编号')
        except Exception as e:
            ledger[label].update(status='UNKNOWN',error_type=type(e).__name__);save(OUT/'ledger.json',ledger)
            raise RuntimeError('提交状态不确定，未重试') from None
        ledger[label].update(status='SUBMITTED',task_id=task_id,submitted_utc=H.now());save(OUT/'ledger.json',ledger)
        print(json.dumps(dict(submitted=label,task_id=task_id)),flush=True)


def collect():
    manifest=read(OUT/'manifest.json');ledger=read(OUT/'ledger.json');mgr=H.manager()
    for task in manifest['tasks']:
        label=task['label'];item=ledger.get(label,{})
        if item.get('status') in ('DONE','FAILED') or not item.get('task_id'): continue
        try:
            with contextlib.redirect_stdout(io.StringIO()),contextlib.redirect_stderr(io.StringIO()): raw=mgr.result(item['task_id'])
        except Exception as e:
            print(json.dumps(dict(label=label,error=type(e).__name__)),flush=True);continue
        if not isinstance(raw,dict):continue
        if raw.get('count') or raw.get('counts'):
            counts=raw.get('count') or raw.get('counts')
            assert sum(counts.values())==SHOTS
            assert all(len(str(b).replace(' ',''))==task['q'] for b in counts)
            save(OUT/'raw'/(label+'.json'),raw)
            ledger[label].update(status='DONE',received_utc=H.now())
        elif str(raw.get('status','')).lower() in ('error','failed','cancelled'):
            save(OUT/'raw'/(label+'.json'),raw);ledger[label].update(status='FAILED',received_utc=H.now())
        save(OUT/'ledger.json',ledger)
    print(json.dumps(dict(total=len(manifest['tasks']),done=sum(v['status']=='DONE' for v in ledger.values()),failed=sum(v['status']=='FAILED' for v in ledger.values()))),flush=True)


def analyze():
    manifest=read(OUT/'manifest.json');ledger=read(OUT/'ledger.json')
    if not all(ledger.get(t['label'],{}).get('status')=='DONE' for t in manifest['tasks']): raise RuntimeError('真机结果尚未齐全')
    rows=[];bit_checks=[]
    for task in manifest['tasks']:
        raw=read(OUT/'raw'/(task['label']+'.json'))
        counts=raw.get('count') or raw.get('counts')
        p=np.zeros(2**task['q'])
        for b,count in counts.items():p[int(str(b).replace(' ',''),2)]+=count/SHOTS
        if task['pattern'] is not None:
            actual=format(int(p.argmax()),f"0{task['q']}b")[::-1]
            bit_checks.append(dict(q=task['q'],expected=task['pattern'],actual=actual,passed=actual==task['pattern']))
            continue
        source=manifest['sources'][task['source_label']]
        chi=characters(source['q'],source['mapping'])
        corr=chi@p
        _,_,weight=problem(source['n'],source['seed'])
        ev=assess(corr,source['instance'],weight,source['model']!='redundant')
        rows.append(dict(label=task['label'],source_label=task['source_label'],n=source['n'],q=source['q'],model=source['model'],
             repetition=task['repetition'],correlations=corr.tolist(),result=ev,ideal=source['result'],
             identical_assignment=ev['assignment']==source['result']['assignment'],
             original_id=source.get('original_id',source['id']),
             original_identical=ev['assignment']==source.get('original_result',source['result'])['assignment'],
             refinement_preserves_assignment=source.get('preserves_assignment',True),
             variable_accuracy=float(np.mean((corr<0)==(np.array(source['correlations'])<0))),
             correlation_mae=float(np.mean(np.abs(corr-source['correlations'])))))
    assert all(r['passed'] for r in bit_checks)
    save(OUT/'analysis.json',dict(bit_checks=bit_checks,rows=rows,
         feasible=sum(r['result']['feasible'] for r in rows),identical=sum(r['identical_assignment'] for r in rows),
         total=len(rows),task_ids={label:item['task_id'] for label,item in ledger.items()}))
    print(json.dumps(dict(total=len(rows),feasible=sum(r['result']['feasible'] for r in rows),identical=sum(r['identical_assignment'] for r in rows))),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['prepare','submit','collect','analyze']);parser.add_argument('--chip',default='Baihua')
    parser.add_argument('--output',default='hardware');parser.add_argument('--source-runs');args=parser.parse_args()
    OUT=HERE/args.output
    SOURCE_RUNS=HERE/args.source_runs if args.source_runs else None
    if args.action=='prepare':prepare(args.chip)
    else:globals()[args.action]()
