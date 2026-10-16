"""每条线路每次读出配一份全局随机分车；统一精修。"""
import hashlib
import time
from pathlib import Path
import numpy as np
import worker as W

HERE=Path(__file__).resolve().parent

def main():
    inputs=W.read(HERE/'inputs.json');rows=[];started=time.time()
    for key,source in inputs['sources'].items():
        S=W.load_setup(inputs['instances'][source['instance']])
        for rep in range(W.REPEATS):
            draw_seed=W.seed(f'paired-global-one:{key}:{rep}')
            assignment=np.random.default_rng(draw_seed).integers(0,S['K'],size=S['n'],dtype=np.int64)
            rows.append(W.process(S,assignment,'global_one_paired',dict(source=key,repeat=rep,
                seed=draw_seed,budget=1,selection='仅生成一份，无预选')))
        print(key,'complete',flush=True)
    assert len(rows)==12*W.REPEATS==384
    W.save(HERE/'controls'/'paired_one.json',dict(rows=rows,random_per_output=1,
        repeated_per_source=W.REPEATS,source_count=len(inputs['sources']),
        input_sha256=hashlib.sha256((HERE/'inputs.json').read_bytes()).hexdigest(),
        host='disco',seconds=time.time()-started,
        note='与每条训练后有限测量模拟输出按线路及重复编号配对；每次只生成一份全局随机分车，并精修一次。'))
    print('COMPLETE',len(rows),flush=True)

if __name__=='__main__':main()
