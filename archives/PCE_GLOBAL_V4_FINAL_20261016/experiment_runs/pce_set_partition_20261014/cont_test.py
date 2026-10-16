import numpy as np, assign_pce as A, pce_scale as ps, sys
GS=[float(x) for x in sys.argv[1].split(',')]
def c1(S,rng):
    y=rng.normal(0,1,S['m'])
    for g in [0.0]+GS:
        def fun(y):
            s=np.tanh(y); L,gs=A.loss_s(S,s); L+=g*(1-s**2).sum(); gs=gs-2*g*s
            return L,gs*(1-s**2)
        y,_,_=ps.adam(fun,y,steps=1500 if g==0 else 400)
    return np.tanh(y)
tot=0; R=8
for n in A.SP.INSTANCES:
    S=A.setup(n); rng=np.random.default_rng(3); res=[]
    for r in range(R):
        d=A.assess(S,A.decode(S,(c1(S,rng)>0).astype(int))); res.append(d)
    f=[d for d in res if d['feasible']]; tot+=len(f)
    print(n,len(f),'/',R,'over',np.mean([d['overload'] for d in res]),'gap',np.round([d['gap_bks_percent'] for d in f],1),flush=True)
print('total',tot)
