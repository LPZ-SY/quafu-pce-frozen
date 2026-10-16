"""今日正式8/9比特线路：冻结物理布局、编译核验及小批真机比较。"""
from device_access import *
import hashlib,ast,re
sys.path.insert(0,str(ROOT/'pce_v4_research'))
import numpy as np
from types import SimpleNamespace
core_source=(ROOT/'pce_v4_research/core.py').read_text(encoding='utf-8')
core_fn=next(n for n in ast.parse(core_source).body if isinstance(n,ast.FunctionDef) and n.name=='probabilities')
core_fn.decorator_list=[]
core_env={'np':np};exec(compile(ast.Module(body=[core_fn],type_ignores=[]),'<training_probability_reference>','exec'),core_env)
C=SimpleNamespace(read=read,probabilities=core_env['probabilities'])
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    if (HERE/'manifest.json').exists():raise RuntimeError('已有冻结清单，不覆盖')
    h=access();h.manager()
    sys.path.insert(0,str(ROOT/'cpdptw_pce_hardware_20261012'))
    from compile_physical import quality_path
    from quark.circuit import Transpiler
    k_source=(ROOT/'pce_dongling_direct4_20261014/pce_k.py').read_text(encoding='utf-8')
    k_nodes=[n for n in ast.parse(k_source).body if isinstance(n,ast.FunctionDef) and n.name in ('qasm','qasm_pattern')]
    k_env={'np':np};exec(compile(ast.Module(body=k_nodes,type_ignores=[]),'<frozen_circuit_export>','exec'),k_env)
    K=SimpleNamespace(qasm=k_env['qasm'],qasm_pattern=k_env['qasm_pattern'])
    source=(ROOT/'pce_assignment_decisive_20261014/hardware_assignment.py').read_text(encoding='utf-8')
    fn=next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name=='simulate_qasm')
    env={'np':np,'re':re};exec(compile(ast.Module(body=[fn],type_ignores=[]),'<independent_simulator>','exec'),env);simulate=env['simulate_qasm']
    status=read(HERE/'queue_snapshot.json')['status'];chips=[c for c in ('Baihua','Dongling','Shenglian') if isinstance(status.get(c),int)]
    models={}
    for name,phase,seeds in [('B-n31-k5','final_confirmation',range(2001,2004)),('A-n45-k7','final_scale',range(3001,3004))]:
        for sd in seeds:
            p=ROOT/'pce_v4_research/models'/phase/name/f'FINAL_JOINT_{sd}/result.json';m=C.read(p)
            for arm in ('trained','random'):
                key=f'{name}_{sd}_{arm}';models[key]=dict(instance=name,seed=sd,arm=arm,nq=m['nq'],layers=m['layers'],
                    theta=m['theta'] if arm=='trained' else m['initial_theta'],fams=m['mapping_fams'],masks=m['mapping_masks'],
                    priorities=m['tie_priorities'],model_file=str(p.relative_to(ROOT)),model_sha256=digest(p))
    tasks=[];devices={}
    for chip in chips:
        h.CHIP=chip;backend,cal=h.backend();folder=HERE/'devices'/chip;save(folder/'calibration.json',cal)
        layouts={};layout_failures={}
        for nq in (8,9):
            try:path,edges,qubits,score=quality_path(cal,nq)
            except RuntimeError as exc:
                layout_failures[str(nq)]=str(exc);continue
            layouts[str(nq)]=dict(path=path,selection_score=score,qubits=[qubits[q] for q in path],
                edges=[dict(qubits=[a,b],fidelity=edges[tuple(sorted((a,b)))]) for a,b in zip(path,path[1:])])
        save(folder/'layouts.json',layouts);devices[chip]=dict(calibration_time=cal.get('calibration_time'),queue_snapshot=status[chip],layouts=layouts,layout_failures=layout_failures)
        logical=[]
        for key,m in models.items():
            if m['seed'] not in (2001,3001) or str(m['nq']) not in layouts:continue
            for basis in 'ZXY':logical.append(dict(label=key+'_'+basis,key=key,basis=basis,nq=m['nq'],control=False,
                qasm=K.qasm(m['theta'],m['nq'],m['layers'],basis)))
        for nq in (8,9):
            if str(nq) not in layouts:continue
            for kind,pattern in [('zero','0'*nq),('one','1'*nq),('order',('001011010')[:nq])]:
                logical.append(dict(label=f'control_{nq}_{kind}',key=f'control_{nq}_{kind}',basis='Z',nq=nq,control=True,
                    pattern=pattern,qasm=K.qasm_pattern(nq,pattern)))
        logical.sort(key=lambda t:(not t['control'],t['nq'],t['basis'],t['key']))
        for task in logical:
            text=task.pop('qasm');nq=task['nq'];layout=layouts[str(nq)]['path']
            with contextlib.redirect_stdout(io.StringIO()),contextlib.redirect_stderr(io.StringIO()):
                circuit=Transpiler(backend).run(text,target_qubits=layout,optimize_level=1,niter=5)
            physical=str(circuit.to_openqasm2);meas=re.findall(r'measure\s+q\[(\d+)\]\s*->\s*c\[(\d+)\]',physical)
            assert sorted(int(c) for _,c in meas)==list(range(nq))
            assert sorted(int(q) for q,_ in meas)==sorted(layout)
            reverse=np.array([int(format(i,f'0{nq}b')[::-1],2) for i in range(1<<nq)])
            pp=simulate(physical,meas)[reverse]
            if task['control']:
                target=np.zeros(1<<nq);target[int(task['pattern'],2)]=1.
            else:
                m=models[task['key']];target=C.probabilities(np.array(m['theta']),nq,m['layers'])['ZXY'.index(task['basis'])]
            error=float(np.max(np.abs(pp-target)));assert error<1e-9,(chip,task['label'],error)
            path=folder/'physical'/(task['label']+'.qasm');path.parent.mkdir(parents=True,exist_ok=True);path.write_text(physical,encoding='utf-8')
            lp=folder/'logical'/(task['label']+'.qasm');lp.parent.mkdir(parents=True,exist_ok=True);lp.write_text(text,encoding='utf-8')
            gates=re.findall(r'(?:cz|cx)\s+q\[(\d+)\]\s*,\s*q\[(\d+)\]',physical)
            edges={tuple(sorted(map(int,x['qubits_index']))) for x in cal['couplers_info'].values() if float(x.get('fidelity') or 0)>0}
            assert all(tuple(sorted(map(int,g))) in edges for g in gates)
            task.update(chip=chip,file=path.relative_to(HERE).as_posix(),sha256=digest(path),shots=4096,
                layout=layout,measurements=meas,two_qubit_gates=len(gates),probability_error=error,ideal_probabilities=pp.tolist())
            tasks.append(task)
        print('芯片编译通过',chip,cal.get('calibration_time'),'任务',len(logical),flush=True)
    # 各设备按同一逻辑任务轮流提交；校验在前，训练与随机使用完全相同物理布局。
    tasks.sort(key=lambda t:(not t['control'],t['nq'],t['basis'],t['key'],t['chip']))
    save(HERE/'manifest.json',dict(created_utc=now(),devices=devices,models=models,tasks=tasks,total_tasks=len(tasks),
        total_measurements=sum(t['shots'] for t in tasks),shots_per_basis=4096,shots_per_complete_readout=12288,
        scope='当天选片小批验证：每规模固定第一个确认种子，训练与原始随机参数各1份完整真机读出；不是最终统计确认。',
        later_confirmation='推荐设备后独立追加其余预定种子及重复批次；本次不根据试验成绩改参数。'))
    save(HERE/'protocol.json',dict(date='2026-10-16',chips=chips,hardware_budget=dict(tasks=len(tasks),measurements=sum(t['shots'] for t in tasks)),
        simulation='每芯片、每规模固定前三个训练种子及对应未训练参数；按当天所选物理路径门保真度建局部退极化模型；64份完整读出，每份三基合计12288次。',
        noise_limit='门保真度转换为独立退极化近似；不重复叠加已可能包含在门误差内的T1/T2损耗；不覆盖串扰、漂移或相干误差。',
        readout='白桦和圣莲今日校准未公开有效读出保真度，不能把零字段当零误差。共同读出误差0%、2%、5%为敏感性情景；东岭另用今日公开读出混淆参数。',
        refinement='沿用全部组相同10000候选精修预算、同一停止条件与确定性实现。最终不可行记失败。',
        selection='分别比较任务相关关联误差、客户分配变化、最终可行率与路线；校准模拟排序和真实队列分别报告，真机小批仅初步验证。',
        hardware_scope='用户已请求当天选片实验与真机效果；先提交三台可排队设备的小批共54项上限，不扩大训练、重试未知提交或购买额度。'))
    print('冻结完成',len(tasks),sum(t['shots'] for t in tasks),flush=True)
if __name__=='__main__':main()
