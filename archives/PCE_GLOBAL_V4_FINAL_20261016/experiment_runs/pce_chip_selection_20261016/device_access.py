"""今日芯片状态快照；复用既有授权，不输出凭据。"""
import sys,json,contextlib,io,os,importlib.util,datetime
from pathlib import Path
HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
SDK=ROOT/'pce_cmaes_fixed12_20261014/hardware_prepared/.sdk'
sys.path.insert(0,str(SDK))
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def save(p,d):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(d,ensure_ascii=False,indent=1),encoding='utf-8');t.replace(p)
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def access():
    spec=importlib.util.spec_from_file_location('chip_selection_existing_access',ROOT/'pce_dongling_direct4_20261014/hw.py')
    h=importlib.util.module_from_spec(spec);spec.loader.exec_module(h)
    h.HERE=HERE
    from quark import Task
    original=Task.session.request
    def bounded(method,url,**kwargs):
        kwargs.setdefault('timeout',(10,30));return original(method,url,**kwargs)
    Task.session.request=bounded
    return h
def snapshot():
    h=access();mgr=h.manager()
    with contextlib.redirect_stdout(io.StringIO()),contextlib.redirect_stderr(io.StringIO()):status=mgr.status()
    save(HERE/'queue_snapshot.json',dict(retrieved_utc=now(),status=status))
    print(json.dumps(dict(retrieved_utc=now(),status=status),ensure_ascii=False),flush=True)
    chips=sorted(set(['Baihua','Dongling','Shenglian'])|set(status.keys())) if isinstance(status,dict) else ['Baihua','Dongling','Shenglian']
    records=[]
    for chip in chips:
        if not isinstance(chip,str):continue
        h.CHIP=chip
        try:
            backend,cal=h.backend();save(HERE/'calibrations'/f'{chip}.json',cal)
            rec=dict(chip=chip,retrieved_utc=now(),queue=status.get(chip),calibration_time=cal.get('calibration_time'),
                qubits=len(cal.get('qubits_info',{})),couplers=len(cal.get('couplers_info',{})),available=True,
                calibration_fields=list(cal))
        except Exception as exc:rec=dict(chip=chip,retrieved_utc=now(),available=False,error_type=type(exc).__name__)
        records.append(rec);save(HERE/'snapshot_summary.json',records);print(json.dumps(rec,ensure_ascii=False),flush=True)
if __name__=='__main__':snapshot()
