import numpy as np, pipeline as P, pce_scale as ps
OUTER, INNER, RHO = 20, 300, 0.5
def alm_loss(Qb, lam, g):
    A, K = Qb['A'], Qb['K']
    def f(s):
        x=(1+s)/2; L,gs=P.relaxed(Qb,s)
        viol=1-A@x; L+= lam@viol; gx=-A.T@lam
        L+=g*(1-s**2).sum()
        return L, gs+0.5*gx-2*g*s
    return f
def alm_c1(Qb, rng):
    y=rng.normal(0,1,P.M); lam=np.zeros(Qb['A'].shape[0])
    for o in range(OUTER):
        g=0.02*o; f=alm_loss(Qb,lam,g)
        def fun(y):
            s=np.tanh(y); L,gs=f(s); return L,gs*(1-s**2)
        y,_,_=ps.adam(fun,y,steps=INNER)
        x=(np.tanh(y)>0).astype(float); lam=lam+RHO*(1-Qb['A']@x)
        if P.feasible(Qb,x)[0]: return x,o
    return x,OUTER
if __name__=='__main__':
    tot=0
    for n in P.INSTANCES:
        inst,pool,Qb,meta=P.prepare(n); rng=np.random.default_rng(1); f=0; gaps=[]; v=[]; its=[]
        for r in range(16):
            x,o=alm_c1(Qb,rng); ok,b=P.feasible(Qb,x); f+=ok; v.append(b); its.append(o)
            if ok: gaps.append(100*(Qb['c']@x/meta['pool_optimum']-1))
        tot+=f; print(n,f,'/16 violated',np.mean(v),'outer',np.mean(its),'gap',np.round(gaps,1),flush=True)
    print('total',tot,'/96')
