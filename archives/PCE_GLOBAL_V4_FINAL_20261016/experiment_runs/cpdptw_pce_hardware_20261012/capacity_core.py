# -*- coding: utf-8 -*-
"""泡利关联编码真机容量实验的公共函数。"""
from __future__ import annotations

import hashlib
import itertools
import json
import math
import os
from pathlib import Path

import numpy as np


BASES = ("Z", "X", "Y")


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temp, path)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def balanced_slots(n, orders=(2, 3), limit=None):
    """按 Z、X、Y 交错返回同方向二体/三体槽，避免方向偏置。"""
    by_order = {order: list(itertools.combinations(range(n), order)) for order in orders}
    slots = []
    max_len = max((len(items) for items in by_order.values()), default=0)
    for rank in range(max_len):
        for order in orders:
            items = by_order[order]
            if rank >= len(items):
                continue
            qubits = items[rank]
            for basis in BASES:
                slots.append((basis, *qubits))
                if limit is not None and len(slots) >= limit:
                    return slots
    return slots


def slot_capacity(n):
    pair = 3 * math.comb(n, 2)
    triple = 3 * math.comb(n, 3)
    return {"pair": pair, "triple": triple, "total": pair + triple}


def qasm_header(n):
    return [
        "OPENQASM 2.0;",
        'include "qelib1.inc";',
        f"qreg q[{n}];",
        f"creg c[{n}];",
    ]


def qasm_measure(lines, n, basis="Z"):
    if basis == "X":
        lines.extend(f"h q[{q}];" for q in range(n))
    elif basis == "Y":
        lines.extend(f"sdg q[{q}];" for q in range(n))
        lines.extend(f"h q[{q}];" for q in range(n))
    elif basis != "Z":
        raise ValueError(basis)
    lines.extend(f"measure q[{q}] -> c[{q}];" for q in range(n))
    return "\n".join(lines) + "\n"


def sign_patterns(n):
    return {
        "all_zero": "0" * n,
        "all_one": "1" * n,
        "alternating": "".join(str(q % 2) for q in range(n)),
        "pairs": "".join(str((q // 2) % 2) for q in range(n)),
        "quartets": "".join(str((q // 4) % 2) for q in range(n)),
    }


def product_qasm(n, basis, signs):
    """制备给定泡利方向的正负本征态，再在同方向测量。"""
    lines = qasm_header(n)
    for q, bit in enumerate(signs):
        if basis == "Z":
            if bit == "1":
                lines.append(f"x q[{q}];")
        elif basis == "X":
            angle = "-pi/2" if bit == "1" else "pi/2"
            lines.append(f"ry({angle}) q[{q}];")
        elif basis == "Y":
            angle = "pi/2" if bit == "1" else "-pi/2"
            lines.append(f"rx({angle}) q[{q}];")
        else:
            raise ValueError(basis)
    return qasm_measure(lines, n, basis)


def readout_qasm(n, bits):
    lines = qasm_header(n)
    lines.extend(f"x q[{q}];" for q, bit in enumerate(bits) if bit == "1")
    return qasm_measure(lines, n, "Z")


def ghz_qasm(n, basis):
    """在互不重叠的三元组上制备 GHZ 态；X/Y 三体关联有已知答案。"""
    if basis not in ("X", "Y"):
        raise ValueError("GHZ 诊断只用于 X 或 Y 方向")
    lines = qasm_header(n)
    for a in range(0, n - 2, 3):
        lines.extend((f"h q[{a}];", f"cx q[{a}],q[{a + 1}];", f"cx q[{a + 1}],q[{a + 2}];"))
        if basis == "Y":
            lines.append(f"s q[{a}];")
    return qasm_measure(lines, n, basis)


def pce_angles(n, layers=4, seed=20261011):
    rng = np.random.default_rng(seed + 1009 * n + layers)
    return rng.uniform(-math.pi, math.pi, size=(layers, n, 2))


def pce_qasm(n, basis, layers=4, seed=20261011):
    angles = pce_angles(n, layers, seed)
    lines = qasm_header(n)
    for layer in range(layers):
        for q in range(n):
            lines.append(f"ry({angles[layer, q, 0]:.17g}) q[{q}];")
            lines.append(f"rz({angles[layer, q, 1]:.17g}) q[{q}];")
        for q in range(layer % 2, n - 1, 2):
            lines.append(f"cz q[{q}],q[{q + 1}];")
    return qasm_measure(lines, n, basis)


def _apply_one(state, matrix, q, n):
    shaped = state.reshape((2,) * n)
    shaped = np.moveaxis(shaped, q, 0)
    shaped = np.tensordot(matrix, shaped, axes=(1, 0))
    return np.moveaxis(shaped, 0, q).reshape(-1)


def pce_state(n, layers=4, seed=20261011):
    state = np.zeros(1 << n, dtype=np.complex128)
    state[0] = 1.0
    angles = pce_angles(n, layers, seed)
    indices = np.arange(1 << n, dtype=np.uint64)
    for layer in range(layers):
        for q in range(n):
            theta, phi = angles[layer, q]
            c, s = math.cos(theta / 2), math.sin(theta / 2)
            state = _apply_one(state, np.array([[c, -s], [s, c]], complex), q, n)
            state = _apply_one(state, np.diag([np.exp(-0.5j * phi), np.exp(0.5j * phi)]), q, n)
        for q in range(layer % 2, n - 1, 2):
            bit_a = n - 1 - q
            bit_b = n - 2 - q
            mask = (((indices >> bit_a) & 1) & ((indices >> bit_b) & 1)).astype(bool)
            state[mask] *= -1
    return state


def rotate_for_measurement(state, n, basis):
    if basis == "Z":
        return state
    h = np.array([[1, 1], [1, -1]], complex) / math.sqrt(2)
    sdg = np.diag([1, -1j])
    matrix = h if basis == "X" else h @ sdg
    out = state
    for q in range(n):
        out = _apply_one(out, matrix, q, n)
    return out


def probabilities_by_basis(state, n, basis):
    rotated = rotate_for_measurement(state, n, basis)
    probabilities = np.abs(rotated) ** 2
    probabilities /= probabilities.sum()
    return probabilities


def correlations_from_probabilities(probabilities, n, slots):
    """用快速沃尔什变换一次得到所有 Z 子集矩，避免逐关联枚举状态。"""
    moments = np.asarray(probabilities, dtype=float).copy()
    if len(moments) != 1 << n:
        raise ValueError("概率向量长度与量子比特数不符")
    width = 1
    while width < len(moments):
        view = moments.reshape(-1, width * 2)
        left = view[:, :width].copy()
        right = view[:, width:].copy()
        view[:, :width] = left + right
        view[:, width:] = left - right
        width *= 2
    output = []
    for slot in slots:
        mask = 0
        for q in slot[1:]:
            mask |= 1 << (n - 1 - q)
        output.append(float(moments[mask]))
    return np.asarray(output)


def sample_counts(probabilities, n, shots, rng, readout_error=0.0):
    states = rng.choice(len(probabilities), size=shots, p=probabilities)
    if readout_error:
        flips = rng.random((shots, n)) < readout_error
        masks = np.zeros(shots, dtype=np.int64)
        for q in range(n):
            masks |= flips[:, q].astype(np.int64) << (n - 1 - q)
        states ^= masks
    unique, numbers = np.unique(states, return_counts=True)
    return {format(int(state), f"0{n}b"): int(count) for state, count in zip(unique, numbers)}


def normalize_counts(counts, n):
    out = {}
    for key, value in counts.items():
        bits = str(key).replace(" ", "")
        if len(bits) != n or set(bits) - {"0", "1"}:
            raise ValueError(f"非法位串：{key}")
        number = float(value)
        if number < 0:
            raise ValueError("计数不能为负数")
        out[bits] = out.get(bits, 0.0) + number
    if not out or sum(out.values()) <= 0:
        raise ValueError("缺少有效计数")
    return out


def bit_correct_probabilities(counts, expected):
    counts = normalize_counts(counts, len(expected))
    shots = sum(counts.values())
    return [sum(number for bits, number in counts.items() if bits[q] == expected[q]) / shots
            for q in range(len(expected))]


def readout_parameters(zero_counts, one_counts, n):
    p00 = np.array(bit_correct_probabilities(zero_counts, "0" * n))
    p11 = np.array(bit_correct_probabilities(one_counts, "1" * n))
    return {"p00": p00, "p11": p11, "a": p00 - p11, "b": p00 + p11 - 1.0}


def correlations_from_counts(counts, slots, correction=None):
    first_key = next(iter(counts), "")
    n = len(str(first_key).replace(" ", ""))
    if not n:
        raise ValueError("缺少有效计数")
    counts = normalize_counts(counts, n)
    shots = sum(counts.values())
    probabilities = np.zeros(1 << n, float)
    for bits, number in counts.items():
        probabilities[int(bits, 2)] += number / shots
    moments = probabilities.copy()
    width = 1
    while width < len(moments):
        view = moments.reshape(-1, width * 2)
        left = view[:, :width].copy()
        right = view[:, width:].copy()
        view[:, :width] = left + right
        view[:, width:] = left - right
        width *= 2
    output = np.zeros(len(slots), float)
    a = correction["a"] if correction is not None else None
    b = correction["b"] if correction is not None else None
    if correction is not None and np.any(np.abs(b) < 0.1):
        raise ValueError("读出修正矩阵接近奇异，拒绝放大噪声")
    for index, slot in enumerate(slots):
        qubits = tuple(slot[1:])
        if correction is None:
            mask = sum(1 << (n - 1 - q) for q in qubits)
            output[index] = moments[mask]
            continue
        numerator = 0.0
        # 展开 Π(o_i-a_i)，二体最多4项、三体最多8项。
        for choose in range(1 << len(qubits)):
            moment_mask = 0
            coefficient = 1.0
            for position, q in enumerate(qubits):
                if (choose >> position) & 1:
                    moment_mask |= 1 << (n - 1 - q)
                else:
                    coefficient *= -a[q]
            numerator += coefficient * moments[moment_mask]
        output[index] = numerator / float(np.prod(b[list(qubits)]))
    return output


def expected_product_correlations(signs, slots):
    values = np.array([1.0 if bit == "0" else -1.0 for bit in signs])
    return np.array([float(np.prod(values[list(slot[1:])])) for slot in slots])


def rank_correlation(left, right):
    """无并列时等同斯皮尔曼相关；有并列时使用平均秩。"""
    def ranks(values):
        order = np.argsort(values, kind="mergesort")
        result = np.empty(len(values), float)
        pos = 0
        while pos < len(values):
            end = pos + 1
            while end < len(values) and values[order[end]] == values[order[pos]]:
                end += 1
            result[order[pos:end]] = 0.5 * (pos + end - 1)
            pos = end
        return result
    x, y = ranks(np.asarray(left)), ranks(np.asarray(right))
    if np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])
