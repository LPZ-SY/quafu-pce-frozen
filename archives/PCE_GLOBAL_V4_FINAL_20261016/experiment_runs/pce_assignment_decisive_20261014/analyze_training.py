"""统一评估原始分配、精确路线参考与有限测量稳定性。"""
import json
from pathlib import Path
import numpy as np
from train_assignment import Circuit, problem, assess
from run_decisive import all_route_costs, characters

HERE=Path(__file__).resolve().parent


def heuristic_route(distance,nodes):
    remaining=set(map(int,nodes));route=[0]
    while remaining:
        node=min(remaining,key=lambda v:(distance[route[-1],v],v))
        route.append(node);remaining.remove(node)
    route.append(0)
    improved=True
    while improved:
        improved=False
        for i in range(1,len(route)-2):
            for j in range(i+1,len(route)-1):
                delta=distance[route[i-1],route[j]]+distance[route[i],route[j+1]]-distance[route[i-1],route[i]]-distance[route[j],route[j+1]]
                if delta < -1e-9:
                    route[i:j+1]=route[i:j+1][::-1];improved=True
    return route,float(sum(distance[a,b] for a,b in zip(route,route[1:])))


def main():
    rows=json.loads((HERE/'training'/'all_runs.json').read_text())
    cache={};evaluated=[]
    rng=np.random.default_rng(55903)
    for r in rows:
        key=(r['n'],r['seed'])
        if key not in cache:
            inst,distance,w=problem(*key)
            data=dict(instance=inst,distance=distance,weight=w)
            if r['n']<=16:
                n=r['n'];cost=all_route_costs(distance)
                a=(np.arange(2**n)[:,None]>>np.arange(n))&1
                load=a@np.array(inst['demand'])
                feasible=(load<=inst['capacities'][0]) & (sum(inst['demand'])-load<=inst['capacities'][1])
                distance_all=cost+cost[::-1]
                proxy=.5*(np.sum(a*(a@w),axis=1)+np.sum((1-a)*((1-a)@w),axis=1))
                data.update(cost=cost,route_optimum=float(distance_all[feasible].min()),proxy_optimum=float(proxy[feasible].min()))
            cache[key]=data
        data=cache[key];a=np.array(r['result']['assignment'])
        ev=dict(id=r['id'],n=r['n'],seed=r['seed'],model=r['model'],restart=r['restart'],layers=r['layers'],
                q=r['q'],feasible=r['result']['feasible'],min_margin=r['result']['min_margin'],calls=r['calls'],seconds=r['seconds'],
                proxy=r['result']['proxy'],proxy_optimum=data.get('proxy_optimum'),exact_route_optimum=data.get('route_optimum'))
        routes=[];heuristic=0
        for k in range(2):
            route,length=heuristic_route(data['distance'],list(np.flatnonzero(a[:,k])+1))
            routes.append(route);heuristic+=length
        ev.update(routes=routes,heuristic_distance=heuristic if ev['feasible'] else None)
        if r['n']<=16 and ev['feasible']:
            mask=int(a[:,0]@(1<<np.arange(r['n'])))
            exact=float(data['cost'][mask]+data['cost'][(2**r['n']-1)^mask])
            ev.update(exact_distance_for_assignment=exact,route_gap_percent=100*(exact/data['route_optimum']-1),
                      proxy_gap_percent=100*(ev['proxy']/data['proxy_optimum']-1))
        if r['model']!='classical':
            circ=Circuit(r['q'],r['layers']);state=circ.forward(np.array(r['theta']))[0]
            p=state*state;p/=p.sum()
            chi=characters(r['q'],r['mapping'])
            counts=rng.multinomial(4096,p,size=128)
            correlators=counts@chi.T/4096
            metrics=[assess(c,r['instance'],data['weight'],r['model']!='redundant') for c in correlators]
            ev.update(measurement_repeats=128,measurement_feasible=sum(x['feasible'] for x in metrics),
                      measurement_identical=sum(x['assignment']==r['result']['assignment'] for x in metrics))
        evaluated.append(ev)
    summary=[]
    for n,model in sorted({(r['n'],r['model']) for r in evaluated}):
        group=[r for r in evaluated if (r['n'],r['model'])==(n,model)]
        gaps=[r['route_gap_percent'] for r in group if 'route_gap_percent' in r]
        summary.append(dict(n=n,model=model,runs=len(group),feasible=sum(r['feasible'] for r in group),
            q=group[0]['q'],mean_calls=float(np.mean([r['calls'] for r in group])),
            mean_seconds=float(np.mean([r['seconds'] for r in group])),
            mean_heuristic_distance=float(np.mean([r['heuristic_distance'] for r in group if r['feasible']])) if any(r['feasible'] for r in group) else None,
            conditional_mean_exact_route_gap_percent=float(np.mean(gaps)) if gaps else None,
            measurement_feasible=sum(r.get('measurement_feasible',0) for r in group),
            measurement_identical=sum(r.get('measurement_identical',0) for r in group),
            measurement_trials=sum(r.get('measurement_repeats',0) for r in group)))
    output=dict(summary=summary,rows=evaluated,scope='独立原始分配；未做容量修复；大于16客户的距离采用相同最近邻及二边交换，不冒称最优；两重启同时改变深度，不能据此归因深度效应；各方法迭代上限相同但评价次数不同，不声称严格同计算预算')
    (HERE/'training'/'evaluation.json').write_text(json.dumps(output,ensure_ascii=False,indent=2))
    print(json.dumps(summary,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
