"""对整数对称距离使用等价增量边长，保持原车内搜索的顺序和严格改进条件。"""
from core import np,njit,route_cost,setup,save,HERE

@njit(cache=True)
def accelerated_route_cost(D,nodes):
    n=len(nodes)
    if n==0:return 0.
    r=np.empty(n,np.int64);used=np.zeros(n,np.bool_);cur=0
    for k in range(n):
        bj,bd=-1,1e18
        for j in range(n):
            if not used[j] and D[cur,nodes[j]]<bd:bd,bj=D[cur,nodes[j]],j
        used[bj]=True;r[k]=nodes[bj];cur=nodes[bj]
    best=D[0,r[0]]+D[r[n-1],0]
    for i in range(n-1):best+=D[r[i],r[i+1]]
    temp=np.empty(n,np.int64);rest=np.empty(n,np.int64);imp=True
    while imp:
        imp=False
        for i in range(n-1):
            for j in range(i+1,n):
                a=0 if i==0 else r[i-1];b=0 if j==n-1 else r[j+1]
                delta=D[a,r[j]]+D[r[i],b]-D[a,r[i]]-D[r[j],b]
                if delta < -1e-9:
                    for t in range((j-i+1)//2):r[i+t],r[j-t]=r[j-t],r[i+t]
                    best+=delta;imp=True
        for L in range(1,4):
            if L>=n:break
            for i in range(n-L+1):
                for kk in range(n-L+1):
                    if kk==i:continue
                    for rev in range(2):
                        first,last=r[i],r[i+L-1];prev=0 if i==0 else r[i-1];after=0 if i+L==n else r[i+L]
                        left=0 if kk==0 else r[kk-1 if kk-1<i else kk-1+L]
                        right=0 if kk==n-L else r[kk if kk<i else kk+L]
                        sf,sl=(last,first) if rev else (first,last)
                        delta=D[prev,after]-D[prev,first]-D[last,after]+D[left,sf]+D[sl,right]-D[left,right]
                        if delta < -1e-9:
                            t=0
                            for q in range(n):
                                if q<i or q>=i+L:rest[t]=r[q];t+=1
                            t=0
                            for q in range(kk):temp[t]=rest[q];t+=1
                            for q in range(L):temp[t]=r[i+(L-1-q if rev else q)];t+=1
                            for q in range(kk,n-L):temp[t]=rest[q];t+=1
                            r[:]=temp;best+=delta;imp=True
    return best

def verify():
    import time
    rng=np.random.default_rng(87023);records=[]
    for name in ('B-n31-k5','A-n45-k7'):
        S=setup(name);D=S['inst']['D'];assert np.array_equal(D,D.T) and np.array_equal(D,np.rint(D))
        groups=[np.sort(rng.choice(np.arange(1,S['n']+1),size=size,replace=False)).astype(np.int64) for size in list(range(0,18))*12]
        accelerated_route_cost(D,groups[0]);route_cost(D,groups[0])
        start=time.perf_counter();expected=[route_cost(D,g) for g in groups];old=time.perf_counter()-start
        start=time.perf_counter();actual=[accelerated_route_cost(D,g) for g in groups];new=time.perf_counter()-start
        assert expected==actual
        records.append(dict(instance=name,customer_subsets=len(groups),identical=True,original_seconds=old,incremental_seconds=new))
    save(HERE/'route_acceleration_verification.json',dict(passed=True,records=records,scope='只用于已核对的整数对称距离；客户集合全部来自正式实例。'))
    print(records,flush=True)

if __name__=='__main__':verify()
