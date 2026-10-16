import json, numpy as np, pipeline as P
from multiprocessing import Pool
def one(a):
    name,arm,r=a; rec=P.job((name,arm,r))
    for t in rec['by_tau'].values(): t.pop('theta',None)
    return rec
if __name__=='__main__':
    jobs=[(n,a,r) for n in P.INSTANCES for a in ('Q','Qc') for r in range(2)]
    with Pool(6) as pool: rows=pool.map(one,jobs)
    json.dump(rows,open('sp_quantum_check.json','w'),ensure_ascii=False)
    for r in rows:
        d=r['by_tau']['0.1']['ideal']; print(r['instance'],r['arm'],r['restart'],'feasible',d['feasible'],'violated',d['violated_customers'],'selected',d['selected'])
