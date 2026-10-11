"""真机误差诊断（事后分析，不在预登记内）：哪种误差模型能解释纠缠线路的偏差？"""
import json, os, numpy as np
from scipy.optimize import minimize
PKG = "/home/claude/pce8/run"; N = 8
MAN = json.load(open(f"{PKG}/task_manifest.json", encoding="utf-8"))
def counts(o):
    r = json.load(open(f"{PKG}/live/raw/{o:03d}.json", encoding="utf-8")); c = r.get("count") or r.get("counts")
    rows = [([int(k[N-1-i]) for i in range(N)], v) for k, v in c.items()]
    return np.repeat(np.array([b for b, _ in rows]), [v for _, v in rows], axis=0)
cal = {0: [], 1: []}
for t in MAN["tasks"]:
    if t["kind"] == "calibration": cal[t["bits_c0_to_c7"][0]].append((counts(t["order"]) != t["bits_c0_to_c7"][0]).mean(0))
p01, p10 = np.mean(cal[0], 0), np.mean(cal[1], 0)
Minv = [np.linalg.inv(np.array([[1-p01[q], p10[q]], [p01[q], 1-p10[q]]])) for q in range(N)]
def corr(bits, a, b):
    pm = np.array([np.mean((bits[:, a]==x)&(bits[:, b]==y)) for x in (0,1) for y in (0,1)])
    pc = np.kron(Minv[a], Minv[b]) @ pm; return pc[0]-pc[1]-pc[2]+pc[3]

H = np.array([[1,1],[1,-1]])/np.sqrt(2)
def ap(psi, U, q): return np.moveaxis(np.tensordot(U, np.moveaxis(psi, q, 0), axes=(1,0)), 0, q)
def run(theta, L, fam, dphi, zph):
    th = np.array(theta).reshape(L, N, 2); psi = np.zeros((2,)*N, complex); psi[(0,)*N] = 1
    for l in range(L):
        for q in range(N):
            a, b = th[l,q]; psi = ap(psi, np.array([[np.cos(a/2),-np.sin(a/2)],[np.sin(a/2),np.cos(a/2)]]), q)
            psi = ap(psi, np.diag([np.exp(-.5j*b), np.exp(.5j*b)]), q)
        for a in range(l % 2, N-1, 2):
            idx = [slice(None)]*N; idx[a] = 1; idx[a+1] = 1; psi[tuple(idx)] *= np.exp(1j*(np.pi + dphi[a]))
            for q in (a, a+1): psi = ap(psi, np.diag([1, np.exp(1j*zph[q])]), q)
    for q in range(N):
        if fam == "X": psi = ap(psi, H, q)
        if fam == "Y": psi = ap(psi, H @ np.diag([1, -1j]), q)
    p = np.abs(psi.reshape(-1))**2
    B = ((np.arange(256)[:, None] >> (N-1-np.arange(N))) & 1); Z = 1-2*B
    return Z.T @ (p[:, None]*Z)

data = []
for inst in MAN["instances"]:
    A = inst["arms"]["Q"]; slots = [tuple(s) for s in A["slots"]]; e_hw = np.zeros(len(slots))
    for f in "ZXY":
        ts = [t for t in MAN["tasks"] if t["kind"]=="algorithm" and t["instance"]==inst["seed"] and t["arm"]=="Q" and t["family"]==f]
        bits = np.vstack([counts(t["order"]) for t in ts])
        for k, s in enumerate(slots):
            if s[0]==f: e_hw[k] = corr(bits, s[1], s[2])
    data.append((inst["seed"], A, slots, np.array(A["e_ideal"]), e_hw))

def predict(par, model):
    F = par[0]; dphi = np.zeros(N-1); zph = np.zeros(N)
    if model == "common_phase": dphi[:] = par[1]
    if model == "edge_phase": dphi[:] = par[1:8]
    if model == "zphase": zph[:] = par[1:9]
    if model == "edge+z": dphi[:] = par[1:8]; zph[:] = par[8:16]
    out = []
    for sd, A, slots, eid, ehw in data:
        C = {f: run(A["theta"], A["layers"], f, dphi, zph) for f in "ZXY"}
        out.append(F*np.array([C[s[0]][s[1], s[2]] for s in slots]))
    return out
def loss(par, model): return sum(((p - d[4])**2).sum() for p, d in zip(predict(par, model), data))
ehw = np.concatenate([d[4] for d in data]); eid = np.concatenate([d[3] for d in data])
print("观测数", len(ehw), "总平方和", round((ehw**2).sum(), 3))
res = {}
for model, npar in (("depol", 1), ("common_phase", 2), ("edge_phase", 8), ("zphase", 9), ("edge+z", 16)):
    best = None
    for st in range(3 if npar > 1 else 1):
        x0 = np.r_[0.8, np.random.default_rng(st).normal(0, 0.2, npar-1)]
        r = minimize(loss, x0, args=(model,), method="Powell", options=dict(maxiter=4000, xtol=1e-3, ftol=1e-6))
        if best is None or r.fun < best.fun: best = r
    pr = np.concatenate(predict(best.x, model))
    # 用拟合的模型解码，看正负号能否复现真机
    agree = []
    for p, d in zip(predict(best.x, model), data):
        c = np.array(d[1]["anchor_sign"]); agree.append(float(((c*p<0)==(c*d[4]<0)).mean()))
    res[model] = dict(params=best.x.round(3).tolist(), sse=round(best.fun, 3), R2=round(1-best.fun/((ehw-ehw.mean())**2).sum(), 3), sign_match_hw=np.round(agree,3).tolist())
    print(model, res[model], flush=True)
json.dump(res, open(f"{PKG}/analysis/error_model_fit.json", "w"), ensure_ascii=False, indent=1)
