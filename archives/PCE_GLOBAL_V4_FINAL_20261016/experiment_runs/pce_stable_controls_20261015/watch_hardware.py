"""仅回收本批已登记的真机任务；在disco分析并取回报告，不提交任何新任务。"""
import concurrent.futures
import io
import json
import os
import subprocess
import sys
import tarfile
import time
from datetime import datetime, timezone
from pathlib import Path

HERE=Path(__file__).resolve().parent
REMOTE='/tmp/pzlv_pce_stable_controls_20261015'
PYTHON='/tmp/pzlv_search_budget_audit_20261014_r01/.venv/bin/python'
FOLDERS=('hardware','hardware_baihua','hardware_shenglian')

def now():return datetime.now(timezone.utc).isoformat()
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def save(p,v):
    tmp=p.with_suffix('.tmp')
    tmp.write_text(json.dumps(v,ensure_ascii=False,indent=1),encoding='utf-8');tmp.replace(p)

def run(args,timeout):
    result=subprocess.run(args,cwd=HERE,capture_output=True,text=True,encoding='utf-8',errors='replace',
                          timeout=timeout,creationflags=subprocess.CREATE_NO_WINDOW)
    if result.returncode:raise RuntimeError('子程序执行失败：'+Path(args[0]).name)
    return result.stdout

def snapshot():
    state={}
    for folder in FOLDERS:
        m=read(HERE/folder/'manifest.json');ledger=read(HERE/folder/'ledger.json')
        assert len(m['tasks'])==len(ledger)==225
        assert all(v.get('task_id') for v in ledger.values())
        state[m['chip']]=dict(total=225,submitted=225,
            done=sum(v['status']=='DONE' for v in ledger.values()),
            failed=sum(v['status']=='FAILED' for v in ledger.values()))
    return state

def synchronize():
    # 白名单打包；真机原始记录仅传计数与状态，不传服务端其他元数据。
    archive=HERE/'hardware_snapshot.tar.gz'
    with tarfile.open(archive,'w:gz') as tar:
        tar.add(HERE/'analyze.py',arcname='analyze.py')
        for folder in FOLDERS:
            for name in ('manifest.json','logical_tasks.json','ledger.json','device_protocol.json','collection_status.json'):
                p=HERE/folder/name
                if p.exists():tar.add(p,arcname=f'{folder}/{name}')
            for p in sorted((HERE/folder/'raw').glob('*.json')):
                raw=read(p);clean={k:raw[k] for k in ('count','counts','status') if k in raw}
                content=json.dumps(clean).encode('utf-8')
                info=tarfile.TarInfo(f'{folder}/raw/{p.name}');info.size=len(content)
                tar.addfile(info,io.BytesIO(content))
    opts=['-o','BatchMode=yes','-o','ConnectTimeout=20']
    run(['scp',*opts,str(archive),f'disco:{REMOTE}/hardware_snapshot.tar.gz'],180)
    run(['ssh',*opts,'disco',f'cd {REMOTE} && tar -xzf hardware_snapshot.tar.gz && '
         f'{PYTHON} analyze.py > analysis.log 2>&1'],1800)
    run(['scp',*opts,*[f'disco:{REMOTE}/{name}' for name in
         ('analysis.json','hardware_analysis.json','最新修正版实验结果.txt','analysis.log')],str(HERE)],180)

def main():
    import msvcrt
    lock=open(HERE/'watch.lock','a+b');lock.seek(0)
    if lock.read(1)==b'':lock.write(b'0');lock.flush()
    lock.seek(0)
    try:msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
    except OSError:raise RuntimeError('已有回收进程运行') from None
    started=time.time();last_synced=None;state=snapshot()
    status=dict(pid=os.getpid(),started_utc=now(),state='运行中',devices=state,
                scope='仅回收三设备已登记675项任务；每轮间隔300秒；最长7天；不新增任务')
    save(HERE/'watch_status.json',status)
    while time.time()-started<7*86400:
        state=snapshot()
        if state!=last_synced:
            try:
                synchronize();last_synced=state
                status['last_analysis_utc']=now();status.pop('sync_error',None)
            except Exception as exc:status['sync_error']=type(exc).__name__
        status.update(devices=state,last_update_utc=now())
        if all(s['done']+s['failed']==225 for s in state.values()) and state==last_synced:
            status['state']='回收结束，分析已同步';save(HERE/'watch_status.json',status);return
        save(HERE/'watch_status.json',status)
        errors={}
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
            pending={pool.submit(run,[sys.executable,str(HERE/'hardware_stable.py'),'collect',chip],1800):chip
                     for chip,s in state.items() if s['done']+s['failed']<225}
            for future in concurrent.futures.as_completed(pending):
                try:print(future.result(),flush=True)
                except Exception as exc:errors[pending[future]]=type(exc).__name__
        status.update(collection_errors=errors,devices=snapshot(),last_update_utc=now())
        save(HERE/'watch_status.json',status)
        print(json.dumps(status,ensure_ascii=False),flush=True)
        # 新结果先同步，再进入等待；避免最终结果返回后多等一轮。
        if snapshot()!=last_synced:continue
        time.sleep(300)
    status.update(state='到达7天回收上限，需续接',last_update_utc=now())
    save(HERE/'watch_status.json',status)

if __name__=='__main__':main()
