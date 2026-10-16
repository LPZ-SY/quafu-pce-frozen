"""服务器剩余正式训练、测试及证据回传。"""
import os,sys,subprocess,socket,json,time,tarfile
from pathlib import Path
H=Path(__file__).resolve().parent
def run(name,*args):
    subprocess.run([sys.executable,str(H/name),*args],cwd=H,check=True)
def main():
    started=time.time();os.environ['PCE_WORKERS']='12'
    (H/'disco_execution.json').write_text(json.dumps(dict(host=socket.gethostname(),alias='disco',pid=os.getpid(),
        python=sys.executable,started_unix=started,workers=12,cwd=str(H)),indent=1))
    run('verify_accelerated.py')
    run('run_remote_training.py','confirmation')
    source=(H/'final_evaluation.py').read_text().replace('from final_objective import *','from final_objective_accelerated import *').replace('max_workers=4','max_workers=12')
    (H/'remote_evaluation.py').write_text(source)
    for mode in ('test','attribution','mapping'):run('remote_evaluation.py',mode)
    (H/'disco_completion.json').write_text(json.dumps(dict(complete=True,seconds=time.time()-started,host=socket.gethostname())))
    with tarfile.open(H.parent/'completed_results.tar.gz','w:gz') as tar:
        for name in ('models','final_tests','final_attribution','mapping_sensitivity'):
            for p in (H/name).rglob('*'):
                if p.is_file() and p.suffix in ('.json','.npz'):tar.add(p,arcname=p.relative_to(H))
        for name in ('disco_execution.json','disco_completion.json','complete_refiner_acceleration_verification.json','remote_evaluation.py'):
            tar.add(H/name,arcname=name)
    print('服务器剩余实验全部完成',time.time()-started,flush=True)
if __name__=='__main__':main()
