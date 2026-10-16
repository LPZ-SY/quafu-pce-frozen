import numpy as np, pipeline as P
from relax_test import c1_cont
def descend(Qb,x):
    x=x.copy()
    while True:
        d=(1-2*x)*(Qb['h']+2*Qb['W']@x); i=np.argmin(d)
        if d[i]>=-1e-12: return x
        x[i]=1-x[i]
for n in P.INSTANCES:
    inst,pool,Qb,meta=P.prepare(n); rng=np.random.default_rng(1)
    under=over=cov=fd=rd=0; R=16
    for r in range(R):
        x=(c1_cont(Qb,rng)>0).astype(float); c=Qb['A']@x
        under+=np.sum(c==0); over+=np.sum(c>1); cov+= (np.all(c>=1) and x.sum()<=Qb['K'])
        fd+=P.feasible(Qb,descend(Qb,x))[0]
        xr=(rng.random(P.M)<Qb['K']/P.M).astype(float); rd+=P.feasible(Qb,descend(Qb,xr))[0]
    print(n,'under',under/R,'over',over/R,'covering-feasible',cov,'/16','| +one-flip descent',fd,'/16 | random+descent',rd,'/16')
