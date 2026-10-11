import json, numpy as np, os, sys
sys.path.insert(0,'.')
import gen_pce8 as G
man=json.load(open('pkg/task_manifest.json',encoding='utf-8'))
rng=np.random.default_rng(11)
for inst in man['instances']:
    A=inst['arms']['Q']; probs={f:np.load(f"pkg/ideal/s{inst['seed']}_Q_{f}.npy") for f in 'ZXY'}
    slots=[tuple(s) for s in A['slots']]
    p=G.predict(probs,slots,np.array(A['anchor_sign']),np.array(A['flip_ideal']),0.92,A['ncz'],rng,reps=100)
    A['prediction'].append(p); print(inst['seed'],p)
json.dump(man,open('pkg/task_manifest.json','w',encoding='utf-8'),ensure_ascii=False,indent=1,default=float)
