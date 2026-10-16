"""仅解析本轮真实任务，保留原始计数并配对编译电路理想读出。"""
from final_objective import *

def counts_array(raw,nq,reverse):
    data=raw.get('count') or raw.get('counts');out=np.zeros(1<<nq,np.int64)
    if not isinstance(data,dict):raise ValueError('没有完整计数字典')
    for word,value in data.items():
        word=str(word).replace(' ','')
        assert len(word)==nq and set(word)<=set('01')
        out[int(word[::-1] if reverse else word,2)]+=int(value)
    return out

def analyze(folder):
    folder=Path(folder);manifest=read(folder/'compiled_manifest.json');ledger=read(folder/'ledger.json');rows=[];controls=[];rejected=[];probability_checks=[]
    lookup={t['label']:t for t in manifest['tasks']};bykey={}
    for task in manifest['tasks']:bykey[(task['key'],task['basis'],task['batch'])]=task
    orientation={}
    for task in manifest['tasks']:
        if not task.get('control') or not task['key'].endswith('_order'):continue
        label=task['label'];nq=task['nq'];batch=task['batch']
        if ledger.get(label,{}).get('status')!='DONE':continue
        raw=read(folder/'raw'/(label+'.json'));direct=counts_array(raw,nq,False);reverse=counts_array(raw,nq,True);target=int(task['pattern'],2)
        use_reverse=reverse[target]>direct[target];array=reverse if use_reverse else direct
        order_passed=bool(int(np.argmax(array))==target)
        passed=bool(order_passed and array.sum()==task['shots'])
        controls.append(dict(nq=nq,batch=batch,reverse=bool(use_reverse),passed=passed,target=task['pattern'],
                             target_frequency=float(array[target]/array.sum()),task_id=ledger[label]['task_id'],
                             bit_order_passed=order_passed,requested_shots=task['shots'],returned_shots=int(array.sum())))
        if order_passed:orientation[(nq,batch)]=bool(use_reverse)
    for key,model in manifest['models'].items():
        S=setup(model['instance']);E=FinalEvaluator(S,'FINAL_JOINT',model['nq'],model['layers'])
        assert E.fams.tolist()==model['fams'] and E.masks.tolist()==model['masks']
        # 独立QASM模拟器按最低位编号量子比特；训练器按最高位编号。显式转换并与训练器核验。
        reverse_indices=np.array([int(format(i,f'0{model["nq"]}b')[::-1],2) for i in range(1<<model['nq'])])
        pp=np.stack([np.array(bykey[(key,basis,0)]['ideal_probabilities'])[reverse_indices] for basis in 'ZXY'])
        error=float(np.max(np.abs(pp-probabilities(np.array(model['theta']),model['nq'],model['layers']))))
        assert error<1e-9,(key,error)
        probability_checks.append(dict(model=key,max_error=error,passed=True))
        ref=E.decode(scores_from_moments(walsh(pp),E.fams,E.masks))
        for batch in range(8):
            tasks=[bykey[(key,b,batch)] for b in 'ZXY'];available=all(ledger.get(t['label'],{}).get('status')=='DONE' for t in tasks)
            meta=dict(instance=model['instance'],seed=model['seed'],batch=batch,nq=model['nq'],arm=model['arm'],
                parameter_source=key,shots=4096,two_qubit_gates=[t['two_qubit_gates'] for t in tasks],
                depths=[t['circuit_depth'] for t in tasks],layouts=[t['layout'] for t in tasks],
                task_ids=[ledger.get(t['label'],{}).get('task_id') for t in tasks],model_sha256=model['model_sha256'])
            if available and (model['nq'],batch) in orientation:
                counts=np.stack([counts_array(read(folder/'raw'/(t['label']+'.json')),model['nq'],orientation[(model['nq'],batch)]) for t in tasks])
                if np.array_equal(counts.sum(axis=1),allocation(4096)) and all(c['passed'] for c in controls if c['nq']==model['nq'] and c['batch']==batch):
                    values=scores_from_moments(walsh(counts/allocation(4096)[:,None]),E.fams,E.masks);v=E.decode(values);_,h=E.hybrid(v)
                    rows.append(dict(**meta,source='QPU',assignment=v.tolist(),scores=values.tolist(),customer_flip_rate=float(np.mean(v!=ref)),**h))
                else:
                    rejected.append(dict(**meta,requested_per_basis=allocation(4096).tolist(),returned_per_basis=counts.sum(axis=1).tolist(),
                        reason='实际返回测量次数或校验任务预算与冻结协议不一致；保留原始数据，不进入主比较。'))
            # 同参数、同编译电路、同测量次数的独立理想读出；不把它作为真机结果。
            rng=np.random.default_rng(seed('compiled_hardware_ideal',model['instance'],model['seed'],model['arm'],batch))
            values,cc=sample(pp,E.fams,E.masks,rng,batches=1);v=E.decode(values[0]);_,h=E.hybrid(v)
            arm='TRAINED_IDEAL_SHOT' if model['arm']=='TRAINED_QPU' else 'RANDOM_IDEAL_SHOT'
            rows.append(dict(**{**meta,'arm':arm},source='IDEAL_COMPILED',assignment=v.tolist(),scores=values[0].tolist(),
                             customer_flip_rate=float(np.mean(v!=ref)),counts=cc[0].tolist(),**h))
    table=[]
    for name in sorted(set(r['instance'] for r in rows)):
        for arm in ('TRAINED_QPU','TRAINED_IDEAL_SHOT','RANDOM_QPU','RANDOM_IDEAL_SHOT'):
            rr=[r for r in rows if r['instance']==name and r['arm']==arm];good=[r['final']['distance'] for r in rr if r['final']['feasible']]
            table.append(dict(实例=name,组=arm,模型数=len(set(r['seed'] for r in rr)),已完成完整读出批数=len(rr),预定完整读出批数=24,
                每批测量次数=4096,原始可行率=float(np.mean([r['raw']['feasible'] for r in rr])) if rr else None,
                原始平均超载=float(np.mean([r['raw']['overload'] for r in rr])) if rr else None,
                最终可行率=float(np.mean([r['final']['feasible'] for r in rr])) if rr else None,
                最终可行路线均值=float(np.mean(good)) if good else None,
                平均净跨车客户数=float(np.mean([r['changed_customers'] for r in rr])) if rr else None,
                客户翻车率=float(np.mean([r['customer_flip_rate'] for r in rr])) if rr else None,
                真实任务证据='ledger.json及raw原始计数' if arm.endswith('_QPU') else '编译后电路理想分布，无设备噪声'))
    report=dict(table=table,rows=rows,controls=controls,rejected_batches=rejected,probability_checks=probability_checks,total_tasks=len(manifest['tasks']),
        done_tasks=sum(x['status']=='DONE' for x in ledger.values()),failed_tasks=sum(x['status']=='FAILED' for x in ledger.values()),
        unreturned_tasks=sum(x['status'] not in ('DONE','FAILED') for x in ledger.values()),
        scope='模拟训练后冻结参数的真机读出验证，不是真机闭环训练。')
    save(folder/'analysis.json',report);print('真机分析',str(folder),report['done_tasks'],report['total_tasks'],table,flush=True)

if __name__=='__main__':
    for name in ('hardware','hardware_scale'):
        folder=HERE/name
        if (folder/'compiled_manifest.json').exists() and (folder/'ledger.json').exists():analyze(folder)
