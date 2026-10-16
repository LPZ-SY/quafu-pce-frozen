"""车号取余：编号 v ≥ K 时当作 v − K 号车，任何读出都不会出现无效车号。导入后替换 assign_pce.decode。"""
import numpy as np
import assign_pce as A
_orig = A.decode
def decode(S, bits):
    return _orig(S, bits) % S['K']
A.decode = decode
