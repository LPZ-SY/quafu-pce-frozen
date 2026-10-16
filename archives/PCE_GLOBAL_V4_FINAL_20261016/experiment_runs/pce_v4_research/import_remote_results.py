"""安全回收服务器结果，已存在证据不覆盖，并核对重复读出。"""
from pathlib import Path
import tarfile,json,io,hashlib
import numpy as np
H=Path(__file__).resolve().parent
def main():
    written=[];retained=[];matched=[]
    with tarfile.open(H/'disco_completed_results.tar.gz','r:gz') as tar:
        for item in tar.getmembers():
            if not item.isfile():raise RuntimeError('归档中含非普通文件')
            p=(H/item.name).resolve()
            if H not in p.parents:raise RuntimeError('归档路径越界')
            content=tar.extractfile(item).read()
            if p.exists():
                if item.name.startswith('models/') and p.name=='result.json':assert p.read_bytes()==content,item.name
                if item.name.startswith('final_tests/') and p.suffix=='.npz':
                    a=np.load(p);b=np.load(io.BytesIO(content))
                    assert np.array_equal(a['counts'],b['counts']),item.name
                    assert np.array_equal(a['scores'],b['scores']),item.name
                    matched.append(item.name)
                if item.name.startswith('final_tests/') and p.suffix=='.json':
                    a=json.loads(p.read_text(encoding='utf-8'));b=json.loads(content)
                    assert a['rows']==b['rows'] and a['summary']==b['summary'],item.name
                retained.append(item.name);continue
            p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(content);written.append(item.name)
    (H/'remote_import_verification.json').write_text(json.dumps(dict(passed=True,new_files=len(written),retained_files=len(retained),
        repeated_count_files_exactly_equal=matched,archive_sha256=hashlib.sha256((H/'disco_completed_results.tar.gz').read_bytes()).hexdigest(),
        policy='已有结果不覆盖；重复模型字节、重复计数与分数、重复测试全部逐项一致。'),ensure_ascii=False,indent=1),encoding='utf-8')
    print('回收完成',len(written),'保留已有',len(retained),'重复计数完全一致',len(matched))
if __name__=='__main__':main()
