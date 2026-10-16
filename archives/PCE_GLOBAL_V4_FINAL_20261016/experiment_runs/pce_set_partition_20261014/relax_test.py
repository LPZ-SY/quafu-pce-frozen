import numpy as np, pipeline as P, pce_scale as ps, sys
GAMMAS=[float(g) for g in sys.argv[1].split(',')]
def c1_cont(Qb,rng):
    y=rng.normal(0,1,P.M)
    for g in GAMMAS:
        def fun(y):
            s=np.tanh(y); L,gs=P.relaxed(Qb,s)
            L+=g*(1-s**2).sum(); gs=gs-2*g*s
            return L,gs*(1-s**2)
        y,_,_=ps.adam(fun,y,steps=800)
    return np.tanh(y)
tot=0
for n in P.INSTANCES:
    inst,pool,Qb,meta=P.prepare(n); rng=np.random.default_rng(1); f=0; gaps=[]; v=[]
    for r in range(16):
        x=(c1_cont(Qb,rng)>0).astype(float); ok,b=P.feasible(Qb,x); f+=ok; v.append(b)
        if ok: gaps.append(100*(Qb['c']@x/meta['pool_optimum']-1))
    tot+=f
    print(n,f,'/16 violated',np.mean(v),'gap',np.round(gaps,1))
print('total',tot,'/96')
