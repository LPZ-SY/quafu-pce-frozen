"""用未参与首次门槛判断的随机种子，验证8192与16384份的质量。"""
import hashlib
import time
from concurrent.futures import ProcessPoolExecutor,as_completed
from pathlib import Path
import numpy as np
import worker as W
from shared import random_prefixes

HERE=Path(__file__).resolve().parent
BUDGETS=np.array([4096,8192,16384],dtype=np.int64)
START,STOP=32,64

def one_instance(name):
    data=W.read(HERE/'inputs.json');S=W.load_setup(data['instances'][name])
    dest=HERE/'extended'/'holdout'/f'{name}.json'
    if dest.exists():
        saved=W.read(dest)
        assert saved['instance']==name and saved['budgets']==BUDGETS.tolist() and saved['start']==START
        rows=saved['rows'];first=saved['next_repeat']
    else:rows=[];first=START
    for rep in range(first,STOP):
        seed=W.seed(f'global:{name}:{rep}')
        proposals,counts=random_prefixes(S['inst']['D'],S['q'],S['cap'],S['K'],seed,BUDGETS)
        for j,budget in enumerate(BUDGETS):
            rows.append(W.process(S,proposals[j],f'global_{budget}_holdout',dict(
                repeat=rep,seed=seed,budget=int(budget),
                unique_vehicle_route_evaluations=int(counts[j]),
                selection='按总超载与精修前路程选一份，仅精修一次')))
        W.save(dest,dict(instance=name,budgets=BUDGETS.tolist(),start=START,
            next_repeat=rep+1,rows=rows,host='disco',
            input_sha256=hashlib.sha256((HERE/'inputs.json').read_bytes()).hexdigest()))
        if (rep+1)%8==0:print(name,rep+1,'/',STOP,flush=True)
    return name,len(rows)

def main():
    data=W.read(HERE/'inputs.json');start=time.time()
    with ProcessPoolExecutor(max_workers=6) as pool:
        for completed in as_completed([pool.submit(one_instance,name) for name in data['instances']]):
            print('COMPLETE',*completed.result(),flush=True)
    W.save(HERE/'extended'/'holdout'/'completion.json',dict(host='disco',complete=True,
        budgets=BUDGETS.tolist(),start=START,stop=STOP,instances=len(data['instances']),
        seconds=time.time()-start))

if __name__=='__main__':main()
