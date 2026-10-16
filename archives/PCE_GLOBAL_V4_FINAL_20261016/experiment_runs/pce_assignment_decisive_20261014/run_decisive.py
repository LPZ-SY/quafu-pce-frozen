"""直接算符编码的决定性检验、容量修正、期望值符号表示复验。"""
import os
for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[name] = '1'
import csv
import hashlib
import itertools
import json
import math
from pathlib import Path
import platform
import time
import numpy as np
from scipy.optimize import linprog
from numba import njit
from pauli_z import ZPoly, compile_expression, constraints, from_diagonal

HERE = Path(__file__).resolve().parent
OUT = HERE/'results'


def save(name, value):
    path = OUT/name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def table(name, rows):
    path = OUT/name
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def nqubits(m):
    return math.ceil((1+math.sqrt(1+8*m))/2)


def characters(q, mapping):
    bits = (np.arange(2**q, dtype=np.int64)[:, None] >> np.arange(q)) & 1
    z = 1-2*bits
    return np.array([z[:, i]*z[:, j] for i, j in mapping], dtype=np.int8)


def mapping_list(n, random_count=64, conditioned_count=16):
    q = nqubits(2*n)
    edges = list(itertools.combinations(range(q), 2))
    rows = [('lexicographic', edges[:2*n])]
    for seed in range(random_count):
        rng = np.random.default_rng(8000+n*100+seed)
        order = rng.permutation(len(edges))[:2*n]
        rows.append((f'random_{seed:03}', [edges[i] for i in order]))
    split = next(s for s in range(1, q) if s*(q-s) >= n and len(edges)-s*(q-s) >= n)
    cross = [e for e in edges if (e[0] < split) != (e[1] < split)]
    same = [e for e in edges if (e[0] < split) == (e[1] < split)]
    for seed in range(conditioned_count):
        rng = np.random.default_rng(9000+n*100+seed)
        c = [cross[i] for i in rng.permutation(len(cross))[:n]]
        s = [same[i] for i in rng.permutation(len(same))[:n]]
        mapping = []
        for a, b in zip(c, s):
            mapping.extend((a, b) if rng.integers(2) else (b, a))
        rows.append((f'conditioned_{seed:03}', mapping))
    return q, rows


def affine_count(q, mapping):
    """两车辆唯一归属等价于每客户两条边奇偶值相反；在二元域精确消元。"""
    basis = {}
    consistent = True
    for i in range(0, len(mapping), 2):
        mask = 0
        for v in mapping[i]+mapping[i+1]: mask ^= 1 << v
        rhs = 1
        while mask:
            pivot = mask.bit_length()-1
            if pivot not in basis:
                basis[pivot] = (mask, rhs)
                break
            mask ^= basis[pivot][0]
            rhs ^= basis[pivot][1]
        if not mask and rhs: consistent = False
    parents = list(range(q))
    def root(v):
        while parents[v] != v: v = parents[v]
        return v
    for a, b in mapping: parents[root(a)] = root(b)
    components = len({root(v) for v in range(q)})
    states = 2**(q-len(basis)) if consistent else 0
    return dict(rank=len(basis), consistent=consistent, states=states,
                unique_assignments=states//2**components, graph_components=components)


@njit(cache=True)
def all_route_costs(distance):
    """Held–Karp 动态规划：所有客户子集的精确最短仓库回路。"""
    n = len(distance)-1
    dp = np.full((1 << n, n), np.inf)
    costs = np.full(1 << n, np.inf)
    costs[0] = 0.
    for i in range(n): dp[1 << i, i] = distance[0, i+1]
    for mask in range(1, 1 << n):
        for last in range(n):
            if not (mask >> last & 1): continue
            previous = mask ^ (1 << last)
            if previous:
                value = np.inf
                for k in range(n):
                    if previous >> k & 1:
                        value = min(value, dp[previous, k]+distance[k+1, last+1])
                dp[mask, last] = value
            costs[mask] = min(costs[mask], dp[mask, last]+distance[last+1, 0])
    return costs


def make_instance(n, seed, ratio):
    rng = np.random.default_rng(seed)
    xy = rng.uniform(0, 100, (n+1, 2))
    xy[0] = 50
    demand = rng.integers(1, 10, n)
    capacity = int(math.ceil(demand.sum()*ratio/2))
    return dict(n=n, k=2, seed=seed, capacity_ratio=ratio, demand=demand.tolist(),
                capacities=[capacity]*2, xy=xy.tolist(), distance_convention='欧氏距离，不取整')


def decode_metrics(assignment, demand, capacity):
    unique = ((assignment.sum(axis=2)-1)**2).sum(axis=1)
    load = np.einsum('sik,i->sk', assignment, demand)
    equal = ((load-capacity)**2).sum(axis=1)
    excess = (np.maximum(load-capacity, 0)**2).sum(axis=1)
    feasible = (unique == 0) & (excess == 0)
    return unique, load, equal, excess, feasible


def verify_compiler():
    rng = np.random.default_rng(742)
    errors = []
    for q in (3, 5, 7, 9):
        pairs = list(itertools.combinations(range(q), 2))
        mapping = [pairs[i] for i in rng.permutation(len(pairs))[:2*(len(pairs)//2)]]
        n = len(mapping)//2
        demand = rng.integers(1, 10, n).tolist()
        capacity = [sum(demand)//2+2]*2
        poly = constraints(mapping, demand, capacity, 17, 3)
        assignments = ((1-characters(q, mapping).T)//2).reshape(-1, n, 2)
        unique, _, equal, excess, _ = decode_metrics(assignments, demand, capacity)
        direct = 17*unique+3*equal
        errors.append(max(abs(float(poly.evaluate(s))-v) for s, v in enumerate(direct)))
        corrected = from_diagonal((17*unique+3*excess).tolist())
        assert all(corrected.evaluate(s) == int(17*unique[s]+3*excess[s]) for s in range(2**q))
    a = (1-ZPoly.z(0, 1))/2
    b = (1-ZPoly.z(1, 2))/2
    assert (a*a).terms == a.terms
    assert (ZPoly.z(0, 1)*ZPoly.z(1, 2)).terms == ZPoly.z(0, 2).terms
    assert compile_expression('(sum(a)-1)**2', {'a': [a, b]}).terms == ((a+b-1)**2).terms
    assert (ZPoly.z(0, 1)*ZPoly.z(2, 3)*ZPoly.z(4, 5)).terms == ZPoly.z(0,1,2,3,4,5).terms
    assert max(errors) == 0
    return dict(max_energy_error=max(errors), tested_qubits=[3,5,7,9],
                exact_rational_arithmetic=True, all_states_checked=2**3+2**5+2**7+2**9)


def run_operator():
    batch = []
    examples = {}
    map_cache = {}
    for n, seeds in ((10, range(100,110)), (16, range(200,204))):
        q, mappings = mapping_list(n)
        map_cache[n] = (q, mappings)
        decoded = {}
        for label, mapping in mappings:
            a = ((1-characters(q, mapping).T)//2).reshape(-1,n,2)
            decoded[label] = a
            predicted = affine_count(q, mapping)
            valid = np.all(a.sum(axis=2)==1, axis=1)
            assert valid.sum() == predicted['states']
        for seed in seeds:
            first = make_instance(n, seed, 1.02)
            xy = np.array(first['xy'])
            distance = np.linalg.norm(xy[:,None]-xy[None,:], axis=2)
            costs = all_route_costs(distance)
            bits = ((np.arange(2**n)[:,None] >> np.arange(n)) & 1)
            demand = np.array(first['demand'])
            load0 = bits@demand
            full_distance = costs+costs[::-1]
            for ratio in (1.02, 1.15, 1.35):
                inst = make_instance(n, seed, ratio)
                cap = np.array(inst['capacities'])
                full_feasible = (load0 <= cap[0]) & (demand.sum()-load0 <= cap[1])
                optimum = float(full_distance[full_feasible].min()) if full_feasible.any() else None
                ident = f'n{n}_s{seed}_r{ratio}'
                inst.update(full_feasible_assignments=int(full_feasible.sum()), exact_route_optimum=optimum)
                save(Path('instances')/(ident+'.json'), inst)
                for label, mapping in mappings:
                    a = decoded[label]
                    unique, loads, equal, excess, feasible = decode_metrics(a, demand, cap)
                    ids = a[:,:,0]@(1 << np.arange(n))
                    covered = np.unique(ids[feasible])
                    best = float(full_distance[covered].min()) if len(covered) else None
                    row = dict(instance=ident,n=n,q=q,mapping=label,
                        represented_onehot=int(np.unique(ids[unique==0]).size),
                        full_feasible=int(full_feasible.sum()),represented_feasible=int(len(covered)),
                        coverage=float(len(covered)/full_feasible.sum()) if full_feasible.any() else None,
                        exact_route_optimum=optimum,encoded_best_route=best,
                        encoded_route_gap_percent=100*(best/optimum-1) if best is not None and optimum else None)
                    for la in (1,100,10000):
                        energy = la*unique+equal
                        ground = energy == energy.min()
                        row[f'equality_ground_feasible_la{la}'] = int(feasible[ground].sum())
                        row[f'equality_ground_count_la{la}'] = int(ground.sum())
                    corrected = unique+excess
                    ground = corrected == corrected.min()
                    row['inequality_ground_feasible'] = int(feasible[ground].sum())
                    row['inequality_ground_count'] = int(ground.sum())
                    batch.append(row)
                    if n==10 and seed==100 and ratio==1.15 and label in ('lexicographic','conditioned_000'):
                        original = constraints(mapping, inst['demand'], inst['capacities'])
                        fixed = from_diagonal(corrected.tolist())
                        assert all(original.evaluate(s)==int(unique[s]+equal[s]) for s in range(2**q))
                        assert all(fixed.evaluate(s)==int(corrected[s]) for s in range(2**q))
                        table(f'{label}_mapping.csv', [dict(customer=i//2+1,vehicle=i%2+1,qubit_a=e[0],qubit_b=e[1]) for i,e in enumerate(mapping)])
                        table(f'{label}_hamiltonian_original.csv', original.rows())
                        table(f'{label}_hamiltonian_inequality.csv', fixed.rows())
                        rows = []
                        for s in np.argsort(unique+equal,kind='stable'):
                            rows.append(dict(state=int(s),bits_q0_first=''.join(str((int(s)>>i)&1) for i in range(q)),
                                energy=int(unique[s]+equal[s]),inequality_energy=int(corrected[s]),
                                assignment=';'.join(''.join(map(str,v)) for v in a[s]),
                                unique_penalty=int(unique[s]),load_v1=int(loads[s,0]),load_v2=int(loads[s,1]),
                                equality_penalty=int(equal[s]),overload_penalty=int(excess[s]),feasible=bool(feasible[s])))
                        table(f'{label}_all_states_sorted.csv', rows)
                        examples[label] = dict(instance=inst,mapping=mapping,
                            original_terms=len(original.terms),original_max_order=max(bin(m).count('1') for m in original.terms),
                            inequality_terms=len(fixed.terms),inequality_max_order=max(bin(m).count('1') for m in fixed.terms),
                            ground_states=[r for r in rows if r['energy']==rows[0]['energy']],summary=row)
            print(json.dumps(dict(stage='operator',n=n,seed=seed,rows=len(batch))),flush=True)
    table('operator_batch.csv',batch)
    save('examples.json', examples)
    summaries = []
    for n in (10,16):
        for kind in ('lexicographic','random','conditioned'):
            selected = [r for r in batch if r['n']==n and r['mapping'].startswith(kind)]
            gaps = [r['encoded_route_gap_percent'] for r in selected if r['encoded_route_gap_percent'] is not None]
            summaries.append(dict(n=n,kind=kind,count=len(selected),
                any_feasible=sum(r['represented_feasible']>0 for r in selected),
                mean_onehot=float(np.mean([r['represented_onehot'] for r in selected])),
                mean_coverage=float(np.mean([r['coverage'] for r in selected if r['coverage'] is not None])),
                conditional_mean_route_gap_percent=float(np.mean(gaps)) if gaps else None,
                equality_ground_any_feasible=sum(r['equality_ground_feasible_la1']>0 for r in selected),
                inequality_ground_any_feasible=sum(r['inequality_ground_feasible']>0 for r in selected)))
    save('operator_summary.json',summaries)
    return summaries


def run_representation():
    n,q=10,7
    mapping=list(itertools.combinations(range(q),2))[:20]
    chi=characters(q,mapping).astype(float)
    outcomes=[]
    targets=[]
    distributions=[]
    for mask in range(2**n):
        a0=(mask>>np.arange(n))&1
        a=np.stack([a0,1-a0],axis=1).ravel()
        target=1-2*a
        # 对全部物理 Z 测量分布求最大统一符号间隔，不把已知目标的表示检验冒称求解。
        objective=np.r_[np.zeros(2**q),-1.]
        result=linprog(objective,A_ub=np.c_[-target[:,None]*chi,np.ones(20)],b_ub=np.zeros(20),
                       A_eq=np.r_[np.ones(2**q),0.][None,:],b_eq=[1.],bounds=[(0,None)]*(2**q)+[(0,1)],method='highs')
        if not result.success: raise RuntimeError(result.message)
        p=result.x[:-1]
        corr=chi@p
        gamma=float(np.min(target*corr))
        assert gamma>0 and np.max(np.abs((np.sqrt(p)**2)@chi.T-corr))<1e-12
        outcomes.append(dict(assignment_mask=mask,max_margin=gamma,support=int((p>1e-10).sum())))
        targets.append(target)
        distributions.append(p)
        if (mask+1)%256==0: print(json.dumps(dict(stage='margin',completed=mask+1)),flush=True)
    table('all_1024_assignment_margins.csv',outcomes)
    np.savez_compressed(OUT/'representation_states.npz', probabilities=np.array(distributions), targets=np.array(targets),chi=chi)
    selected=sorted(set([0,1,17,85,170,341,511,512,682,853,1022,1023,
                        int(np.argmin([r['max_margin'] for r in outcomes]))]))
    measurements=[]
    rng=np.random.default_rng(55001)
    for mask in selected:
        target=targets[mask]
        optimal=distributions[mask]
        constructive=(1+(target@chi)/20)/(2**q)
        assert constructive.min()>=-1e-14
        assert np.max(np.abs(chi@constructive-target/20))<1e-12
        for method,p in (('constructive',constructive),('max_margin',optimal)):
            for shots in (256,1024,4096,16384):
                counts=rng.multinomial(shots,p,size=256)
                estimate=counts@chi.T/shots
                correct=np.all(estimate*target>0,axis=1)
                decoded=(estimate<0).reshape(-1,n,2)
                onehot=np.all(decoded.sum(axis=2)==1,axis=1) & np.all(estimate!=0,axis=1)
                measurements.append(dict(mask=mask,method=method,shots=shots,repeats=256,
                    all_variables_correct=int(correct.sum()),onehot_feasible=int(onehot.sum()),
                    min_margin=float(np.min(target*(chi@p)))))
    table('measurement_recovery.csv',measurements)
    summary=dict(target_count=len(outcomes),represented=sum(r['max_margin']>0 for r in outcomes),
                 min_optimal_margin=min(r['max_margin'] for r in outcomes),
                 median_optimal_margin=float(np.median([r['max_margin'] for r in outcomes])),
                 max_optimal_margin=max(r['max_margin'] for r in outcomes),
                 constructive_margin=1/20,scope='已知目标的物理表示与读出检验，不是优化算法成绩')
    save('representation_summary.json',summary)
    return summary


def run_scaling():
    rows=[]
    for n in (10,16,24,40,60):
        q,mappings=mapping_list(n,256,32)
        for label,mapping in mappings:
            result=affine_count(q,mapping)
            rows.append(dict(n=n,k=2,variables=2*n,q=q,mapping=label,
                nominal_compression=2*n/q,independent_binary_baseline=n,
                full_onehot_assignments=2**n,coverage=result['unique_assignments']/2**n,**result))
    table('structural_scaling.csv',rows)
    return [dict(n=n,q=nqubits(2*n),max_onehot=max(r['unique_assignments'] for r in rows if r['n']==n),
                 random_consistent=sum(r['consistent'] for r in rows if r['n']==n and r['mapping'].startswith('random')))
            for n in (10,16,24,40,60)]


def main():
    start=time.time()
    OUT.mkdir(exist_ok=True)
    verification=verify_compiler()
    save('compiler_verification.json',verification)
    structural=run_scaling()
    save('structural_summary.json',structural)
    operator=run_operator()
    representation=run_representation()
    save('run_summary.json',dict(host=platform.node(),python=platform.python_version(),numpy=np.__version__,
         seconds=time.time()-start,compiler=verification,structure=structural,operator=operator,representation=representation,
         source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in HERE.glob('*.py')}))
    print(json.dumps(dict(done=True,seconds=time.time()-start)),flush=True)


if __name__=='__main__': main()
