"""disco分析旧版最新模型的真实6144次读出，以及今日选片小批；分开报告。"""
import sys,re
from pathlib import Path
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'pce_v4_research'))
from final_objective_accelerated import *
HERE=Path(__file__).resolve().parent
import socket,collections,csv
def count_array(raw,nq,reverse):
    out=np.zeros(1<<nq,np.int64)
    for word,count in (raw.get('count') or raw.get('counts')).items():
        word=str(word).replace(' ','');assert len(word)==nq and set(word)<=set('01')
        out[int(word[::-1] if reverse else word,2)]+=int(count)
    return out
def analyze(prior=False):
    folder=HERE/'prior_v4/source' if prior else HERE/'today_source'
    if not (folder/'ledger.json').exists():return
    m=read(folder/'manifest.json');ledger=read(folder/'ledger.json');models=m['models'];controls=[];orientation={};rows=[];missing=[]
    def ident(t):return t['label'] if prior else t['chip']+'__'+t['label']
    for t in m['tasks']:
        if not t.get('control') or not t['key'].endswith('_order'):continue
        label=ident(t);entry=ledger.get(label,{})
        if entry.get('status')!='DONE':continue
        chip=t.get('chip','Dongling');batch=t.get('batch',0);nq=t['nq'];r=read(folder/'raw'/(label+'.json'))
        direct=count_array(r,nq,False);reverse=count_array(r,nq,True);target=int(t['pattern'],2)
        rev=reverse[target]>direct[target];v=reverse if rev else direct;passed=bool(np.argmax(v)==target)
        controls.append(dict(chip=chip,nq=nq,batch=batch,passed=passed,reverse=bool(rev),shots=int(v.sum()),target_frequency=float(v[target]/v.sum())))
        if passed:orientation[(chip,nq,batch)]=bool(rev)
    grouped=collections.defaultdict(dict)
    for t in m['tasks']:
        if not t.get('control'):grouped[(t.get('chip','Dongling'),t['key'],t.get('batch',0))][t['basis']]=t
    for (chip,key,batch),tasks in grouped.items():
        model=models[key];nq=model['nq'];label=[ident(tasks[b]) for b in 'ZXY'];entries=[ledger.get(s,{}) for s in label]
        if not all(e.get('status')=='DONE' for e in entries):missing.append(dict(chip=chip,model=key,batch=batch,reason='任务未完整返回'));continue
        if (chip,nq,batch) not in orientation:missing.append(dict(chip=chip,model=key,batch=batch,reason='位序校验未通过或未返回'));continue
        counts=np.stack([count_array(read(folder/'raw'/(s+'.json')),nq,orientation[(chip,nq,batch)]) for s in label])
        shots=int(counts.sum());assert np.array_equal(counts.sum(axis=1),allocation(shots))
        assert shots==(6144 if prior else 12288),(chip,key,shots)
        E=FinalEvaluator(setup(model['instance']),'FINAL_JOINT',nq);assert E.fams.tolist()==model['fams'] and E.masks.tolist()==model['masks']
        pp=probabilities(np.array(model['theta']),nq,model['layers']);e0=scores_from_moments(walsh(pp),E.fams,E.masks);ref=E.decode(e0)
        values=scores_from_moments(walsh(counts/counts.sum(axis=1)[:,None]),E.fams,E.masks)
        arm='trained' if model['arm'] in ('trained','TRAINED_QPU') else 'random'
        for source,val in [('QPU',values),('IDEAL',sample(pp,E.fams,E.masks,np.random.default_rng(seed('actual_chip_ideal',chip,key,batch,shots)),batches=1,shots=shots)[0][0])]:
            v=E.decode(val);_,h=E.hybrid(v)
            rows.append(dict(chip=chip,instance=model['instance'],seed=model['seed'],arm=arm,batch=batch,source=source,shots=shots,
                task_ids=[e['task_id'] for e in entries],assignment=v.tolist(),customer_flip_rate=float(np.mean(v!=ref)),
                correlation_mae=float(np.mean(np.abs(val-e0))),correlation_cosine=float(np.dot(val,e0)/max(1e-15,np.linalg.norm(val)*np.linalg.norm(e0))),**h))
    table=[]
    for chip,name,arm,source in sorted({(r['chip'],r['instance'],r['arm'],r['source']) for r in rows}):
        rr=[r for r in rows if (r['chip'],r['instance'],r['arm'],r['source'])==(chip,name,arm,source)]
        good=[r['final']['distance'] for r in rr if r['final']['feasible']]
        table.append(dict(chip=chip,instance=name,arm=arm,source=source,models=len(set(r['seed'] for r in rr)),batches=len(rr),shots=rr[0]['shots'],
            raw_feasibility=float(np.mean([r['raw']['feasible'] for r in rr])),raw_overload=float(np.mean([r['raw']['overload'] for r in rr])),
            final_feasibility=float(np.mean([r['final']['feasible'] for r in rr])),final_distance=float(np.mean(good)) if good else None,
            changed_customers=float(np.mean([r['changed_customers'] for r in rr])),customer_flip_rate=float(np.mean([r['customer_flip_rate'] for r in rr])),
            correlation_mae=float(np.mean([r['correlation_mae'] for r in rr])),correlation_cosine=float(np.mean([r['correlation_cosine'] for r in rr]))))
    pairs=[]
    for chip,name in sorted({(r['chip'],r['instance']) for r in rows}):
        for sd in sorted(set(r['seed'] for r in rows if r['chip']==chip and r['instance']==name)):
            train={r['batch']:r for r in rows if r['chip']==chip and r['instance']==name and r['seed']==sd and r['arm']=='trained' and r['source']=='QPU'}
            random={r['batch']:r for r in rows if r['chip']==chip and r['instance']==name and r['seed']==sd and r['arm']=='random' and r['source']=='QPU'}
            common=[(train[b],random[b]) for b in train.keys()&random.keys() if train[b]['final']['feasible'] and random[b]['final']['feasible']]
            if common:pairs.append(dict(chip=chip,instance=name,seed=sd,paired_batches=len(common),trained_minus_random_distance=float(np.mean([a['final']['distance']-b['final']['distance'] for a,b in common]))))
    output='prior_v4_actual_budget_analysis' if prior else 'today_hardware_analysis'
    save(HERE/(output+'.json'),dict(table=table,rows=rows,controls=controls,missing=missing,paired_seeds=pairs,
        status=dict(total=len(m['tasks']),done=sum(x.get('status')=='DONE' for x in ledger.values())),
        limitation='按实际6144次测量事后追加分析，不纳入原定4096次主比较；仅3个训练种子，不能以24批读出当24个独立模型。' if prior else '当天选片小批，每规模1个模型、1批；只作初步设备诊断，不作总体优势推断。'))
    if table:
        with (HERE/(output+'.csv')).open('w',encoding='utf-8-sig',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(table[0]));w.writeheader();w.writerows(table)
    print(output,json.dumps(table,ensure_ascii=False),flush=True)
if __name__=='__main__':
    assert socket.gethostname()=='lenovo';analyze(True);analyze(False)
