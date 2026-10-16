"""延长已有全局随机序列，估计追上修正版模拟输出的首次测试档位。"""
import hashlib
import sys
import time
from concurrent.futures import ProcessPoolExecutor,as_completed
from pathlib import Path
import numpy as np
import worker as W
from shared import random_prefixes

HERE=Path(__file__).resolve().parent
STAGES={16384:[4096,8192,16384],65536:[4096,32768,65536]}

def one_instance(name,maximum):
    data=W.read(HERE/'inputs.json');S=W.load_setup(data['instances'][name])
    existing=W.read(HERE/'results'/f'global_{name}.json')['rows']
    previous={r['repeat']:r for r in existing if r['method']=='global_4096'}
    assert len(previous)==W.REPEATS
    budgets=np.array(STAGES[maximum],dtype=np.int64)
    dest=HERE/'extended'/str(maximum)/f'{name}.json'
    if dest.exists():
        saved=W.read(dest);assert saved['budgets']==budgets.tolist() and saved['instance']==name
        rows=saved['rows'];first=saved['complete_repeats']
    else:rows=[];first=0
    for rep in range(first,W.REPEATS):
        seed=W.seed(f'global:{name}:{rep}')
        proposals,counts=random_prefixes(S['inst']['D'],S['q'],S['cap'],S['K'],seed,budgets)
        assert proposals[0].tolist()==previous[rep]['initial_assignment']
        assert int(counts[0])==previous[rep]['unique_vehicle_route_evaluations']
        for index,budget in enumerate(budgets[1:],1):
            rows.append(W.process(S,proposals[index],f'global_{budget}',dict(
                repeat=rep,seed=seed,budget=int(budget),
                unique_vehicle_route_evaluations=int(counts[index]),
                selection='按总超载与精修前路程选一份，仅精修一次')))
        W.save(dest,dict(instance=name,budgets=budgets.tolist(),complete_repeats=rep+1,rows=rows,
            input_sha256=hashlib.sha256((HERE/'inputs.json').read_bytes()).hexdigest(),host='disco'))
        if (rep+1)%8==0:print(name,maximum,rep+1,'/',W.REPEATS,flush=True)
    return name,len(rows)

def main():
    maximum=int(sys.argv[1]);assert maximum in STAGES
    data=W.read(HERE/'inputs.json')
    t=time.time()
    with ProcessPoolExecutor(max_workers=6) as pool:
        futures=[pool.submit(one_instance,name,maximum) for name in data['instances']]
        for future in as_completed(futures):print('COMPLETE',*future.result(),flush=True)
    W.save(HERE/'extended'/str(maximum)/'completion.json',dict(host='disco',maximum=maximum,
        budgets=STAGES[maximum],instances=len(data['instances']),repeats=W.REPEATS,
        seconds=time.time()-t,complete=True))

if __name__=='__main__':main()
