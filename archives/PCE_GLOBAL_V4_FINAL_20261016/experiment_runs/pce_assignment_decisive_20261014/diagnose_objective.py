"""区分分组目标与真实路程不一致，以及训练未达到分组最优两种损失。"""
import json
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr
from train_assignment import problem
from run_decisive import all_route_costs

HERE=Path(__file__).resolve().parent
rows=[]
for n,seeds in ((10,range(100,106)),(16,range(200,204))):
    for seed in seeds:
        inst,distance,w=problem(n,seed)
        costs=all_route_costs(distance)
        a=(np.arange(2**n)[:,None]>>np.arange(n))&1
        load=a@np.array(inst['demand'])
        feasible=(load<=inst['capacities'][0])&(sum(inst['demand'])-load<=inst['capacities'][1])
        route=(costs+costs[::-1])[feasible]
        proxy=(.5*(np.sum(a*(a@w),axis=1)+np.sum((1-a)*((1-a)@w),axis=1)))[feasible]
        best=np.isclose(proxy,proxy.min(),rtol=0,atol=1e-10)
        rows.append(dict(n=n,seed=seed,feasible_assignments=int(feasible.sum()),route_optimum=float(route.min()),
                         proxy_optimum=float(proxy.min()),proxy_optimal_best_route=float(route[best].min()),
                         proxy_optimal_route_gap_percent=float(100*(route[best].min()/route.min()-1)),
                         spearman=float(spearmanr(proxy,route).statistic)))
(HERE/'results'/'objective_diagnostics.json').write_text(json.dumps(rows,indent=2))
print(json.dumps(rows),flush=True)
