"""准备无凭据的真机计数分析输入。"""
import json, shutil, tarfile
from pathlib import Path
HERE = Path(__file__).resolve().parent
for source, destination, manifest in [
    (HERE.parent/'pce_v4_research/hardware', HERE/'prior_v4/source', 'compiled_manifest.json'),
    (HERE, HERE/'today_source', 'manifest.json'),
]:
    destination.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source/manifest, destination/'manifest.json')
    shutil.copyfile(source/'ledger.json', destination/'ledger.json')
    (destination/'raw').mkdir(exist_ok=True)
    for p in (source/'raw').glob('*.json'):
        raw = json.loads(p.read_text(encoding='utf-8'))
        clean = {k: raw[k] for k in ('count', 'counts', 'status') if k in raw}
        (destination/'raw'/p.name).write_text(json.dumps(clean), encoding='utf-8')
with tarfile.open(HERE/'analysis_input.tar.gz', 'w:gz') as tar:
    for p in (HERE/'prior_v4/source', HERE/'today_source', HERE/'analyze_hardware_readouts.py'):
        tar.add(p, arcname=str(p.relative_to(HERE.parent)))
print('分析输入已准备')
