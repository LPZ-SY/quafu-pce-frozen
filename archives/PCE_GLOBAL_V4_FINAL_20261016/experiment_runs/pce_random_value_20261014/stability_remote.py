"""固定自主求出的分车，仅增强关联符号；独立验证之前不接触测量结果。"""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import json,time,zlib
from pathlib import Path
import numpy as np
from numba import njit
from kernel import expectations
HERE=Path(__file__).resolve().parent

@njit(cache=True)
def improve(theta,fams,masks,seed):
    np.random.seed(seed)
    initial=expectations(theta,fams,masks);target=np.where(initial>0,1.,-1.)
    x=theta.copy();e=initial.copy();loss=np.maximum(0,.05-target*e).mean()
    best=loss;winner=x.copy();accepted=0;valid=0
    for t in range(30000):
        fraction=t/30000
        y=x.copy();j=np.random.randint(64);y[j]+=np.random.normal(0,.3*(1-fraction)+.005)
        f=expectations(y,fams,masks);value=np.maximum(0,.05-target*f).mean()
        # 平滑目标允许中途跨符号，但最终只保留所有符号都保持的参数。
        temp=.0003*(.000001/.0003)**fraction
        if value<=loss or np.random.random()<np.exp(-(value-loss)/temp):
            x,e,loss=y,f,value;accepted+=1
        if np.all((f>0)==(initial>0)):
            valid+=1
            if value<best:best=value;winner=y.copy()
    return winner,expectations(winner,fams,masks),accepted,valid

def job(item):
    key,src=item
    fams=np.array(['ZXY'.index(f) for f,qs in src['slots']],np.int64)
    masks=np.array([sum(1<<(7-q) for q in qs) for f,qs in src['slots']],np.int64)
    t0=time.perf_counter();seed=zlib.crc32(('stability:'+key).encode())
    theta,e,accepted,valid=improve(np.array(src['theta']),fams,masks,seed)
    assert np.array_equal(e>0,np.array(src['outputs'])>0)
    return dict(source=key,theta=theta.tolist(),outputs=e.tolist(),seed=seed,steps=30000,
        accepted=accepted,valid_proposals=valid,seconds=time.perf_counter()-t0,
        min_before=float(np.abs(src['outputs']).min()),min_after=float(np.abs(e).min()),
        loss_before=float(np.maximum(0,.05-np.abs(src['outputs'])).mean()),
        loss_after=float(np.maximum(0,.05-np.abs(e)).mean()))

if __name__=='__main__':
    from concurrent.futures import ProcessPoolExecutor,as_completed
    src=json.loads((HERE/'targets.json').read_text())['sources'];rows=[]
    (HERE/'protocol.json').write_text(json.dumps(dict(target_margin=.05,steps=30000,
        note='同8比特4层，固定原自主训练输出的全部关联正负号；优化平均符号裕量缺口，最终不允许改变分车；独立测量验证在本地另做。')))
    with ProcessPoolExecutor(max_workers=6) as pool:
        for f in as_completed([pool.submit(job,x) for x in src.items()]):
            r=f.result();rows.append(r)
            (HERE/'results.json').write_text(json.dumps(rows))
            print(r['source'],r['min_before'],r['min_after'],r['loss_before'],r['loss_after'],flush=True)
    (HERE/'done.txt').write_text(str(len(rows)))
