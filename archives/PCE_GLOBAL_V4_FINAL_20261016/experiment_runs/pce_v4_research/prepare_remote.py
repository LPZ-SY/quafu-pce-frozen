"""迁移已完成模型，归档未完成运行，冻结等价加速版本。"""
from pathlib import Path
import json,tarfile,shutil,datetime,hashlib
H=Path(__file__).resolve().parent
stamp=datetime.datetime.now(datetime.timezone.utc).isoformat()
aborted=[]
for folder in (H/'models').glob('final_*/*/*'):
    if (folder/'result.json').exists():continue
    if not list(folder.iterdir()):continue
    dest=H/'migration_interrupted'/folder.relative_to(H/'models')
    dest.parent.mkdir(parents=True,exist_ok=True)
    if dest.exists():raise RuntimeError('归档目标已存在')
    progress=json.loads((folder/'progress.json').read_text()) if (folder/'progress.json').exists() else {}
    aborted.append(dict(folder=str(folder.relative_to(H)),progress=progress))
    shutil.move(str(folder),str(dest))
(H/'migration.json').write_text(json.dumps(dict(utc=stamp,remote_alias='disco',interrupted=aborted,
    policy='已完成模型保留；中止运行的已记录成本保留，未完成种子从头执行完整4096次查询。'),ensure_ascii=False,indent=1),encoding='utf-8')
original=(H/'final_objective.py').read_text(encoding='utf-8')
fast=original.replace('from core import *','from core import *\nfrom route_acceleration import accelerated_route_cost').replace('cache[key]=route_cost(D,nodes[:j])','cache[key]=accelerated_route_cost(D,nodes[:j])')
(H/'final_objective_accelerated.py').write_text(fast,encoding='utf-8')
trainer=(H/'run_final_training.py').read_text(encoding='utf-8').replace('from final_objective import *','from final_objective_accelerated import *').replace('max_workers=4','max_workers=int(os.environ.get("PCE_WORKERS", "12"))').replace("code_sha256=digest(HERE/'final_objective.py')","code_sha256=digest(HERE/'final_objective_accelerated.py'),execution_host=__import__('socket').gethostname(),route_code_sha256=digest(HERE/'route_acceleration.py')")
(H/'run_remote_training.py').write_text(trainer,encoding='utf-8')
with tarfile.open(H/'disco_remaining.tar.gz','w:gz') as tar:
    for p in H.glob('*.py'):tar.add(p,arcname='pce_v4_research/'+p.name)
    for p in H.glob('*.json'):tar.add(p,arcname='pce_v4_research/'+p.name)
    for folder in ('instances','models','final_tests','final_attribution'):
        for p in (H/folder).rglob('*'):
            if p.is_file() and p.suffix in ('.json','.npz'):tar.add(p,arcname='pce_v4_research/'+p.relative_to(H).as_posix())
    p=H.parent/'pce_stable_controls_20261015/shared.py';tar.add(p,arcname='pce_stable_controls_20261015/shared.py')
print('已归档中止运行',len(aborted),'计算包字节',(H/'disco_remaining.tar.gz').stat().st_size)
