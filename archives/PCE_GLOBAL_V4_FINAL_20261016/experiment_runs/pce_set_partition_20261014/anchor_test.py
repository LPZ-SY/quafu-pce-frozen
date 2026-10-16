"""对称破缺测试：按极角均匀选 K 个锚点客户，用强罚把锚点 k 固定到车辆 k。"""
import numpy as np, assign_pce as A, pce_scale as ps, zlib, sys
ANCHOR_W = 5.0
def anchors(S):
    P = S['inst']['P']; ang = np.arctan2(P[1:,1]-P[0,1], P[1:,0]-P[0,0]); order = np.argsort(ang)
    return order[(np.arange(S['K']) * S['n']) // S['K']]
def loss_s_anchor(S, s, anc):
    L, g = A.loss_s(S, s)
    p = ((1 + s) / 2).reshape(S['n'], S['B']); gp = np.zeros_like(p)
    for k, i in enumerate(anc):
        code = (k >> np.arange(S['B'])) & 1
        f = np.where(code == 1, p[i], 1 - p[i]); Pk = f.prod()
        L += ANCHOR_W * (1 - Pk)
        for b in range(S['B']):
            gp[i, b] += -ANCHOR_W * np.prod(np.delete(f, b)) * (1 if code[b] else -1)
    return L, g + 0.5 * gp.reshape(-1)
def run(kind, use_anchor, S, rng):
    slots = A.slots_random(S['m']); anc = anchors(S)
    loss = (lambda s: loss_s_anchor(S, s, anc)) if use_anchor else (lambda s: A.loss_s(S, s))
    if kind == 'C1':
        def mk(g):
            def fun(y):
                s = np.tanh(y); L, gs = loss(s); L += g*(1-s**2).sum(); gs = gs-2*g*s; return L, gs*(1-s**2)
            return fun
        y = rng.normal(0, 1, S['m']); y,_,_ = ps.adam(mk(0), y, steps=1500)
        for g in A.GAMMAS: y,_,_ = ps.adam(mk(g), y, steps=400)
        return np.tanh(y)
    F={'X':0,'Y':1,'Z':2}; fi=np.array([F[s[0]] for s in slots]); ai=np.array([s[1] for s in slots]); bi=np.array([s[2] for s in slots]); R=A.RANK
    out=lambda u:(u.reshape(3,A.NQ,R)[fi,ai]*u.reshape(3,A.NQ,R)[fi,bi]).sum(1)/R
    def mk(al,g):
        def fun(u):
            U=u.reshape(3,A.NQ,R); e=out(u); s=np.tanh(al*e); L,gs=loss(s); L+=g*(1-s**2).sum(); gs=gs-2*g*s
            c=gs*al*(1-s**2); G=np.zeros_like(U)
            np.add.at(G,(fi,ai),(c/R)[:,None]*U[fi,bi]); np.add.at(G,(fi,bi),(c/R)[:,None]*U[fi,ai]); return L,G.reshape(-1)
        return fun
    u=rng.normal(0,.3,3*A.NQ*R)/np.sqrt(3)
    for al in (3.,6.,10.,15.,25.): u,_,_=ps.adam(mk(al,0),u,steps=500)
    for g in A.GAMMAS: u,_,_=ps.adam(mk(25.,g),u,steps=400)
    return out(u)
if __name__=='__main__':
    for kind in ('C1','LR'):
        for use in (False, True):
            feas=0; gaps=[]; prox=[]
            for n in A.SP.INSTANCES:
                S=A.setup(n)
                for r in range(4):
                    rng=np.random.default_rng([zlib.crc32(n.encode()), r, 9])
                    d=A.assess(S, A.decode(S,(run(kind,use,S,rng)>0).astype(int)))
                    feas+=d['feasible']; prox.append(d['proxy'])
                    if d['feasible']: gaps.append(d['gap_bks_percent'])
            print(kind,'锚点' if use else '无锚点','合法',feas,'/24','平均差距%',round(np.mean(gaps),1) if gaps else None,'代理',round(np.mean(prox),2),flush=True)
