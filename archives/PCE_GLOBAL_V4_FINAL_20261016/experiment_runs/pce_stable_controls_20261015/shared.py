import numpy as np
from numba import njit,types
from numba.typed import Dict
from types import SimpleNamespace
REFINE_BUDGET=10000

@njit(cache=True)
def rlen(D, r, n):
    if n == 0: return 0.0
    s = D[0, r[0]] + D[r[n - 1], 0]
    for i in range(n - 1): s += D[r[i], r[i + 1]]
    return s

@njit(cache=True)
def route_cost(D, nodes):
    n = len(nodes)
    if n == 0: return 0.0
    r = np.empty(n, np.int64); used = np.zeros(n, np.bool_); cur = 0
    for k in range(n):                               # 最近邻
        bj, bd = -1, 1e18
        for j in range(n):
            if not used[j] and D[cur, nodes[j]] < bd: bd, bj = D[cur, nodes[j]], j
        used[bj] = True; r[k] = nodes[bj]; cur = nodes[bj]
    best = rlen(D, r, n); tmp = np.empty(n, np.int64)
    imp = True
    while imp:
        imp = False
        for i in range(n - 1):                       # 2-opt
            for j in range(i + 1, n):
                for k in range(n): tmp[k] = r[k]
                for k in range(j - i + 1): tmp[i + k] = r[j - k]
                v = rlen(D, tmp, n)
                if v < best - 1e-9:
                    best = v; r[:] = tmp; imp = True
        for L in range(1, 4):                        # Or-opt
            if L >= n: break
            for i in range(n - L + 1):
                for kk in range(n - L + 1):
                    if kk == i: continue
                    for rev in range(2):
                        # 取出 r[i:i+L]，插到剩余序列的位置 kk
                        rest = np.empty(n - L, np.int64); t = 0
                        for q in range(n):
                            if q < i or q >= i + L: rest[t] = r[q]; t += 1
                        t = 0
                        for q in range(kk): tmp[t] = rest[q]; t += 1
                        for q in range(L): tmp[t] = r[i + (L - 1 - q if rev else q)]; t += 1
                        for q in range(kk, n - L): tmp[t] = rest[q]; t += 1
                        v = rlen(D, tmp, n)
                        if v < best - 1e-9:
                            best = v; r[:] = tmp; imp = True
    return best

SP=SimpleNamespace(route_cost=route_cost)

def refine(S, initial):
    """确定性最优改进；单客户换车与跨车两客户交换；同一候选评价上限。"""
    v = initial.copy()
    n, K, q, cap = S['n'], S['K'], S['q'], S['cap']
    cache, evals, accepted, route_calls = {}, 0, 0, 0
    def cost(nodes):
        nonlocal route_calls
        key = tuple(sorted(nodes))
        if key not in cache:
            cache[key] = float(SP.route_cost(S['inst']['D'], np.asarray(key, dtype=np.int64)))
            route_calls += 1
        return cache[key]
    groups = [set((np.flatnonzero(v == k)+1).tolist()) for k in range(K)]
    loads = np.array([sum(q[i-1] for i in g) for g in groups])
    costs = [cost(g) for g in groups]
    current = (float(np.maximum(0, loads-cap).sum()), sum(costs))
    initial_rank = current
    while evals < REFINE_BUDGET:
        best, move = current, None
        def consider(a,b,ga,gb,la,lb):
            nonlocal best, move, evals
            evals += 1
            over = current[0] - max(0,loads[a]-cap)-max(0,loads[b]-cap)+max(0,la-cap)+max(0,lb-cap)
            if over > best[0]: return
            ca, cb = cost(ga), cost(gb)
            value = (float(over), current[1]-costs[a]-costs[b]+ca+cb)
            if value < best:
                best, move = value, (a,b,ga,gb,la,lb,ca,cb)
        for i in range(n):
            if evals >= REFINE_BUDGET: break
            a, node = v[i], i+1
            for b in range(K):
                if evals >= REFINE_BUDGET: break
                if a != b:
                    consider(a,b,groups[a]-{node},groups[b]|{node},loads[a]-q[i],loads[b]+q[i])
        for i in range(n):
            if evals >= REFINE_BUDGET: break
            for j in range(i+1,n):
                if evals >= REFINE_BUDGET: break
                a,b = v[i],v[j]
                if a != b:
                    consider(a,b,(groups[a]-{i+1})|{j+1},(groups[b]-{j+1})|{i+1},loads[a]-q[i]+q[j],loads[b]-q[j]+q[i])
        if move is None: break
        a,b,ga,gb,la,lb,ca,cb = move
        groups[a],groups[b],loads[a],loads[b],costs[a],costs[b] = ga,gb,la,lb,ca,cb
        for node in ga: v[node-1] = a
        for node in gb: v[node-1] = b
        current = best
        accepted += 1
    assert current <= initial_rank
    return v, dict(candidate_evaluations=evals, unique_route_evaluations=route_calls, accepted=accepted, budget=REFINE_BUDGET)

@njit(cache=True)
def decode(e,n,k):
    v=np.zeros(n,np.int64)
    for i in range(n):
        for j in range(3):
            if e[3*i+j]>0:v[i]+=1<<j
        v[i]%=k
    return v

@njit(cache=True)
def evaluate(v,D,q,cap,k,cache):
    loads=np.zeros(k);keys=np.zeros(k,np.int64)
    for i in range(len(v)):
        loads[v[i]]+=q[i];keys[v[i]]|=1<<i
    over=np.maximum(0,loads-cap).sum();distance=0.
    for vehicle in range(k):
        key=keys[vehicle]
        if key not in cache:
            nodes=np.empty(len(v),np.int64);count=0
            for i in range(len(v)):
                if v[i]==vehicle:nodes[count]=i+1;count+=1
            cache[key]=route_cost(D,nodes[:count])
        distance+=cache[key]
    return over,distance

@njit(cache=True)
def random_prefixes(D,q,cap,k,seed,budgets):
    np.random.seed(seed);cache=Dict.empty(types.int64,types.float64)
    best_over=1e100;best_dist=1e100;winner=np.zeros(len(q),np.int64)
    outputs=np.zeros((len(budgets),len(q)),np.int64);counts=np.zeros(len(budgets),np.int64);j=0
    for t in range(budgets[-1]):
        v=np.random.randint(0,k,len(q));over,dist=evaluate(v,D,q,cap,k,cache)
        if over<best_over or (over==best_over and dist<best_dist):best_over,best_dist,winner=over,dist,v.copy()
        if t+1==budgets[j]:
            outputs[j]=winner;counts[j]=len(cache);j+=1
    return outputs,counts

