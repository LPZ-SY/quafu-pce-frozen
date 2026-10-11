"""实现正确性检查：(1) 离散点上损失公式 == 逐条重建路线的精确值；(2) 梯度 == 有限差分。"""
import sys, os, itertools, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pce_insert as I, pce_scale as S
rep = {}
for seed, K in [(59001, 2), (59002, 2), (59003, 3)]:
    sb = I.build(seed, K)
    cands = [sorted(int(sb["var"][v][1]) for v in b) for b in sb["blocks"]]
    worst = 0.0; n = 0; same = 0
    combos = list(itertools.product(*cands))
    if len(combos) > 4000: combos = [combos[i] for i in np.random.default_rng(0).choice(len(combos), 4000, replace=False)]
    for ch in combos:
        ch = list(ch); f = I.loss_w(sb, I.onehot(sb, ch))[0]; h = I.hard(sb, ch)["hard"]
        # 第二种重建：直接把客户插进 [0]+seq+[0]，同空位多人取最短排列，再整条算长度
        worst = max(worst, abs(f - h)); n += 1; same += len(set(ch)) < len(ch)
    rng = np.random.default_rng(1)
    e = rng.uniform(-0.5, 0.5, sb["m"]); errs = []
    for margin in (False, True):
        L, g = I.loss_e(sb, e, margin)
        for k in rng.choice(sb["m"], 10, replace=False):
            d = np.zeros(sb["m"]); d[k] = 1e-6
            fd = (I.loss_e(sb, e + d, margin)[0] - I.loss_e(sb, e - d, margin)[0]) / 2e-6
            errs.append(abs(fd - g[k]))
    circ = S.Circuit(8, max(4, int(np.ceil(sb["m"] / 8))), True); slots = S.make_slots(8, sb["m"])
    th = rng.uniform(-np.pi, np.pi, circ.npar)
    def F(t): return I.loss_e(sb, circ.expect(circ.run(t), slots), True)[0]
    _, ge = I.loss_e(sb, circ.expect(circ.run(th), slots), True); gth = circ.grad(th, slots, ge); cerr = []
    for k in rng.choice(circ.npar, 8, replace=False):
        d = np.zeros(circ.npar); d[k] = 1e-6; cerr.append(abs((F(th + d) - F(th - d)) / 2e-6 - gth[k]))
    rep[f"{seed}_K{K}"] = dict(m=sb["m"], combos=n, same_gap_combos=int(same), max_formula_vs_rebuild=worst,
                              max_grad_err_e=float(max(errs)), max_grad_err_circuit=float(max(cerr)))
    print(seed, K, rep[f"{seed}_K{K}"], flush=True)
json.dump(rep, open("verify.json", "w"), ensure_ascii=False, indent=1)
