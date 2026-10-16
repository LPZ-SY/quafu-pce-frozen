"""固定真机原始分配，只做车内排序；不修复分车，不跨车移动。"""
import json
from functools import lru_cache
from pathlib import Path
import numpy as np
from train_assignment import problem
from analyze_training import heuristic_route

HERE=Path(__file__).resolve().parent


def exact_tour(distance,nodes):
    nodes=list(map(int,nodes))
    if not nodes:return [0,0],0.
    @lru_cache(None)
    def dp(mask,last):
        previous=mask^(1<<last)
        if not previous:return float(distance[0,nodes[last]]),-1
        return min((dp(previous,k)[0]+float(distance[nodes[k],nodes[last]]),k)
                   for k in range(len(nodes)) if previous>>k&1)
    mask=(1<<len(nodes))-1
    cost,last=min((dp(mask,k)[0]+distance[nodes[k],0],k) for k in range(len(nodes)))
    reverse=[]
    while mask:
        reverse.append(nodes[last]);previous=dp(mask,last)[1];mask^=1<<last;last=previous
    route=[0]+reverse[::-1]+[0]
    assert abs(sum(distance[a,b] for a,b in zip(route,route[1:]))-cost)<1e-9
    return route,float(cost)


def main():
    output={}
    for folder in ('hardware','hardware_refined'):
        manifest=json.loads((HERE/folder/'manifest.json').read_text())
        analysis=json.loads((HERE/folder/'analysis.json').read_text())
        results=[]
        for row in analysis['rows']:
            source=manifest['sources'][row['source_label']]
            inst,distance,w=problem(source['n'],source['seed'])
            record=dict(label=row['label'],source_label=row['source_label'],feasible=row['result']['feasible'],
                        route_method='精确车内排序' if source['n']<=16 else '最近邻及二边交换',routes=None,distance=None)
            if record['feasible']:
                a=np.array(row['result']['assignment'])
                solver=exact_tour if source['n']<=16 else heuristic_route
                routes=[];total=0
                for k in range(2):
                    route,cost=solver(distance,np.flatnonzero(a[:,k])+1)
                    routes.append(route);total+=cost
                visited=[v for route in routes for v in route[1:-1]]
                assert sorted(visited)==list(range(1,source['n']+1))
                assert all(sum(inst['demand'][v-1] for v in route[1:-1])<=inst['capacities'][k] for k,route in enumerate(routes))
                record.update(routes=routes,distance=total)
            results.append(record)
        output[folder]=results
    (HERE/'results'/'hardware_routes.json').write_text(json.dumps(output,ensure_ascii=False,indent=2))
    print(json.dumps({k:dict(rows=len(v),feasible=sum(r['feasible'] for r in v)) for k,v in output.items()}),flush=True)


if __name__=='__main__':main()
