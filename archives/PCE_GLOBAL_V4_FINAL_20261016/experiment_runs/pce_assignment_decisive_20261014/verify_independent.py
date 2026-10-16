"""独立核验路线动态规划、真机编译模拟器与表示模型的区别。"""
import itertools
import json
from pathlib import Path
import numpy as np
from run_decisive import all_route_costs,characters
from pauli_z import constraints
from hardware_assignment import simulate_qasm

HERE=Path(__file__).resolve().parent
rng=np.random.default_rng(781)
xy=rng.normal(size=(7,2))
distance=np.linalg.norm(xy[:,None]-xy[None,:],axis=2)
costs=all_route_costs(distance)
error=0.
for mask in range(64):
    nodes=[i+1 for i in range(6) if mask>>i&1]
    brute=min(sum(distance[a,b] for a,b in zip((0,)+p,p+(0,))) for p in itertools.permutations(nodes))
    error=max(error,abs(costs[mask]-brute))
assert error<1e-10
bell='OPENQASM 2.0; include "qelib1.inc"; qreg q[3]; creg c[2]; h q[2]; cx q[2],q[0]; measure q[2] -> c[0]; measure q[0] -> c[1];'
p=simulate_qasm(bell,[('2','0'),('0','1')])
assert np.max(np.abs(p-np.array([.5,0,0,.5])))<1e-12
mapping=list(itertools.combinations(range(7),2))[:20]
chi=characters(7,mapping)
poly=constraints(mapping,[1]*10,[5,5],lambda_a=1,lambda_c=0)
operator_average=sum(float(poly.evaluate(s)) for s in range(128))/128
mean_assignment=(1-chi.mean(axis=1))/2
mean_penalty=float(np.sum((mean_assignment.reshape(10,2).sum(axis=1)-1)**2))
assert operator_average==5 and mean_penalty==0
summary=dict(exact_route_vs_bruteforce_max_error=error,route_subsets=64,
             compiled_simulator_bell_test=True,uniform_state_operator_unique_penalty=operator_average,
             uniform_state_penalty_of_means=mean_penalty)
(HERE/'results'/'independent_verification.json').write_text(json.dumps(summary,indent=2))
print(json.dumps(summary),flush=True)
