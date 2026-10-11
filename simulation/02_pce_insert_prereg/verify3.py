import sys, os, itertools
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pce_insert as I
sb = I.build(59003, 3); cands = [sorted(int(sb["var"][v][1]) for v in b) for b in sb["blocks"]]
w2 = w3 = 0.0; n2 = n3 = 0
for ch in itertools.product(*cands):
    ch = list(ch); d = abs(I.loss_w(sb, I.onehot(sb, ch))[0] - I.hard(sb, ch)["hard"])
    if len(set(ch)) == 1: w3 = max(w3, d); n3 += 1
    else: w2 = max(w2, d); n2 += 1
print("trimmed", sb["trimmed"], "m", sb["m"], "最多2人同空位的组合", n2, "最大误差", w2, "| 3人同空位", n3, "最大误差", w3)
