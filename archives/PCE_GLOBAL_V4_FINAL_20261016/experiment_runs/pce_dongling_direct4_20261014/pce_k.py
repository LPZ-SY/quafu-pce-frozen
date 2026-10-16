"""任意阶泡利关联（同一方向上 k 个比特的乘积）的线路期望值与伴随梯度；线路结构与 pce_scale.Circuit 相同。"""
import itertools
import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'pce_cvrp_encoding_20261013' / 'support'))
import pce_scale as ps


class CircuitK(ps.Circuit):
    def prepare(self, slots):
        """slots: [(方向, 比特下标元组)]；预先按方向分组并计算每个基态下的乘积符号。"""
        self.slots = slots
        self.groups = {}
        for f in 'ZXY':
            idx = [k for k, s in enumerate(slots) if s[0] == f]
            if idx:
                signs = np.column_stack([np.prod(self.Zm[:, list(slots[k][1])], axis=1) for k in idx])
                self.groups[f] = (np.array(idx), signs)
        return self

    def expect_k(self, psi):
        out = np.zeros(len(self.slots))
        for f, (idx, signs) in self.groups.items():
            p = np.abs(self.rotate(psi, ps.VFAM[f]).reshape(-1)) ** 2
            out[idx] = p @ signs
        return out

    def probs(self, psi):
        return {f: np.abs(self.rotate(psi, ps.VFAM[f]).reshape(-1)) ** 2 for f in self.groups}

    def O_apply_k(self, psi, c):
        res = np.zeros_like(psi)
        for f, (idx, signs) in self.groups.items():
            Dg = (signs @ c[idx]).reshape((2,) * self.n)
            V = ps.VFAM[f]
            phi = self.rotate(psi, V) * Dg
            if V is not None:
                phi = self.rotate(phi, V.conj().T)
            res += phi
        return res

    def grad_k(self, th, c):
        n = self.n; ops = list(self.ops(th))
        psi = self.run(th); lam = self.O_apply_k(psi, c); phi = psi
        g = np.zeros((self.L, self.n, 2))
        for kind, a, t, key in reversed(ops):
            if kind == 'cz':
                phi = phi * self.masks[a]; lam = lam * self.masks[a]; continue
            G = ps.PY if kind == 'ry' else ps.PZ
            g[key] = np.imag(np.vdot(lam, ps.apply1(phi, G, a, n)))
            U = ps.ry(t) if kind == 'ry' else ps.rz(t)
            Ud = U.conj().T
            phi = ps.apply1(phi, Ud, a, n); lam = ps.apply1(lam, Ud, a, n)
        return g.reshape(-1)


def slots_k(nq, k, m, seed=20261014):
    allslots = [(f, tuple(c)) for f in 'ZXY' for c in itertools.combinations(range(nq), k)]
    assert m <= len(allslots), (m, len(allslots))
    perm = np.random.default_rng(seed).permutation(len(allslots))[:m]
    return [allslots[i] for i in perm]


def qasm(theta, nq, layers, basis, ent=True):
    th = np.asarray(theta).reshape(layers, nq, 2)
    lines = ['OPENQASM 2.0;', 'include "qelib1.inc";', f'qreg q[{nq}];', f'creg c[{nq}];']
    for l in range(layers):
        for q in range(nq):
            lines.append(f'ry({th[l, q, 0]:.17g}) q[{q}];')
            lines.append(f'rz({th[l, q, 1]:.17g}) q[{q}];')
        if ent:
            lines.extend(f'cz q[{q}],q[{q + 1}];' for q in range(l % 2, nq - 1, 2))
    if basis == 'Y':
        lines.extend(f'sdg q[{q}];' for q in range(nq))
    if basis in ('X', 'Y'):
        lines.extend(f'h q[{q}];' for q in range(nq))
    lines.extend(f'measure q[{q}] -> c[{q}];' for q in range(nq))
    return '\n'.join(lines) + '\n'


def qasm_pattern(nq, bits):
    lines = ['OPENQASM 2.0;', 'include "qelib1.inc";', f'qreg q[{nq}];', f'creg c[{nq}];']
    lines.extend(f'x q[{q}];' for q, b in enumerate(bits) if b == '1')
    lines.extend(f'measure q[{q}] -> c[{q}];' for q in range(nq))
    return '\n'.join(lines) + '\n'
