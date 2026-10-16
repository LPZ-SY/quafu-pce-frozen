"""只操作本轮冻结任务及昨日已登记待回收任务；不重试未知提交。"""
from device_access import *
import hashlib
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def collect_prior():
    h=access();mgr=h.manager();folder=ROOT/'pce_v4_research/hardware';lp=folder/'ledger.json';ledger=read(lp)
    snapshot=HERE/'prior_v4/ledger_before_collection.json'
    if not snapshot.exists():save(snapshot,ledger)
    queried=0
    for label,item in ledger.items():
        if item['status'] in ('DONE','FAILED') or not item.get('task_id'):continue
        queried+=1
        try:
            with contextlib.redirect_stdout(io.StringIO()),contextlib.redirect_stderr(io.StringIO()):r=mgr.result(item['task_id'])
        except Exception:continue
        if isinstance(r,dict) and (r.get('count') or r.get('counts')):
            save(folder/'raw'/(label+'.json'),r);item.update(status='DONE',received_utc=now());save(lp,ledger)
        elif isinstance(r,dict) and str(r.get('status')).lower() in ('failed','error','cancelled'):
            save(folder/'raw'/(label+'.json'),r);item.update(status='FAILED',received_utc=now());save(lp,ledger)
    result=dict(retrieved_utc=now(),queried=queried,status={s:sum(x['status']==s for x in ledger.values()) for s in sorted(set(x['status'] for x in ledger.values()))})
    save(HERE/'prior_v4/collection_status.json',result);print(json.dumps(result,ensure_ascii=False),flush=True)
def submit():
    m=read(HERE/'manifest.json');assert len(m['tasks'])<=54 and m['total_measurements']<=221184
    h=access();mgr=h.manager();p=HERE/'ledger.json';ledger=read(p) if p.exists() else {}
    with contextlib.redirect_stdout(io.StringIO()),contextlib.redirect_stderr(io.StringIO()):queue=mgr.status()
    save(HERE/'queue_at_submission.json',dict(utc=now(),status=queue))
    for t in m['tasks']:
        key=t['chip']+'__'+t['label']
        if key in ledger:
            if not ledger[key].get('task_id'):raise RuntimeError('已有不确定提交，不自动重试')
            continue
        if not isinstance(queue.get(t['chip']),int):continue
        f=HERE/t['file'];assert digest(f)==t['sha256'] and t['shots']==4096
        ledger[key]=dict(status='SUBMITTING',chip=t['chip'],label=t['label'],submitted_utc=now());save(p,ledger)
        payload=dict(chip=t['chip'],name='PCE_CHIP_20261016_'+t['label'],circuit=f.read_text(encoding='utf-8'),shots=4096,compile=False,
            options=dict(compiler=None,correct=False,open_dd=None,target_qubits=[]))
        try:
            with contextlib.redirect_stdout(io.StringIO()),contextlib.redirect_stderr(io.StringIO()):tid=mgr.run(payload)
            assert isinstance(tid,int) and tid>0
        except Exception as exc:
            ledger[key].update(status='SUBMISSION_UNCERTAIN',error_type=type(exc).__name__);save(p,ledger);raise RuntimeError('提交状态不确定，台账已保留') from None
        ledger[key].update(status='SUBMITTED',task_id=tid);save(p,ledger)
        print('已提交',t['chip'],t['label'],tid,flush=True)
def collect():
    h=access();mgr=h.manager();p=HERE/'ledger.json';ledger=read(p)
    for key,item in ledger.items():
        if item['status'] in ('DONE','FAILED') or not item.get('task_id'):continue
        try:
            with contextlib.redirect_stdout(io.StringIO()),contextlib.redirect_stderr(io.StringIO()):r=mgr.result(item['task_id'])
        except Exception:continue
        if isinstance(r,dict) and (r.get('count') or r.get('counts')):
            save(HERE/'raw'/(key+'.json'),r);item.update(status='DONE',received_utc=now());save(p,ledger)
        elif isinstance(r,dict) and str(r.get('status')).lower() in ('failed','error','cancelled'):
            save(HERE/'raw'/(key+'.json'),r);item.update(status='FAILED',received_utc=now());save(p,ledger)
    result=dict(retrieved_utc=now(),chips={c:{s:sum(i['chip']==c and i['status']==s for i in ledger.values()) for s in ('SUBMITTED','DONE','FAILED','SUBMISSION_UNCERTAIN')} for c in sorted(set(i['chip'] for i in ledger.values()))})
    save(HERE/'collection_status.json',result);print(json.dumps(result,ensure_ascii=False),flush=True)
if __name__=='__main__':{'prior_collect':collect_prior,'submit':submit,'collect':collect}[sys.argv[1]]()
