"""离线校验已入库文件；完整附件解压后可校验全部数据。"""
import ast, hashlib, json
from pathlib import Path
root=Path(__file__).resolve().parent
m=json.loads((root/'archive_manifest.json').read_text(encoding='utf-8'))
checked=0;optional=0
for item in m['files']:
    p=root/item['path']
    if not p.exists():
        assert not item['in_git'], '缺少已入库文件：'+item['path']
        optional+=1;continue
    data=p.read_bytes()
    assert hashlib.sha256(data).hexdigest()==item['sha256'], '文件校验失败：'+item['path']
    if p.suffix=='.py':ast.parse(data.decode('utf-8-sig'),filename=item['path'])
    checked+=1
for name,n,k,phase,seeds in [('B-n31-k5',30,5,'final_confirmation',range(2001,2021)),('A-n45-k7',44,7,'final_scale',range(3001,3011))]:
    base=root/'experiment_runs/pce_v4_research'
    inst=json.loads((base/'instances'/f'{name}.json').read_text(encoding='utf-8'))
    assert inst['n']==n and inst['K']==k
    for sd in seeds:
        model=json.loads((base/'models'/phase/name/f'FINAL_JOINT_{sd}'/'result.json').read_text(encoding='utf-8'))
        assert len(model['theta'])==model['nq']*model['layers']*2
        assert len(model['mapping_masks'])==n*k
print(json.dumps(dict(passed=True,checked_files=checked,optional_release_files_not_extracted=optional),ensure_ascii=False))
