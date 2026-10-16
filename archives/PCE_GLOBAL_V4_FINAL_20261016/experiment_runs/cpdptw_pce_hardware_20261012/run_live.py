# -*- coding: utf-8 -*-
"""Baihua 与 Dongling 上的 CPDPTW 二体/三体 PCE 端到端真机实验。"""
from __future__ import annotations

import contextlib
import io
import itertools
import json
import math
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

os.environ["CPDPTW_K_REQUESTS"] = "4"
os.environ["CPDPTW_ACTIONS_PER_REQUEST"] = "8"
os.environ["CPDPTW_SELECTION_STRATEGY"] = "distinct_routes"

import cpdptw_core as M
from capacity_core import (
    atomic_json, balanced_slots, correlations_from_counts, expected_product_correlations,
    readout_parameters, sha256,
)
from compile_physical import quality_path

HERE = Path(__file__).absolute().parent
CHIPS = ("Baihua", "Dongling")
N = 8
LAYERS = 4
SHOTS = 4096
SEED = 74001


def now():
    return datetime.now(timezone.utc).isoformat()


def token():
    values = {}
    for line in (HERE / ".env").read_text(encoding="utf-8-sig").splitlines():
        if line.strip() and not line.lstrip().startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            values[key.strip()] = value.strip().strip('"').strip("'")
    value = values.get("QPU_API_TOKEN") or values.get("QUAFU_API_TOKEN")
    if not value:
        raise RuntimeError("缺少真机凭据")
    return value


def build_problem():
    for attempt in range(20):
        seed = SEED + attempt * 1_000_000
        try:
            problem = M.build_problem(seed)
            return problem, M.build_quadratic_model(problem), seed
        except RuntimeError as exc:
            if "可行复合动作不足" not in str(exc):
                raise
    raise RuntimeError("无法生成冻结的 CPDPTW 实例")


def arbitrary_expectation(circuit, theta, slots):
    state = circuit.run(theta)
    out = np.zeros(len(slots), float)
    for basis in "ZXY":
        indexes = [i for i, slot in enumerate(slots) if slot[0] == basis]
        if not indexes:
            continue
        probability = np.abs(circuit.rotate(state, M.S.VFAM[basis]).reshape(-1)) ** 2
        for index in indexes:
            columns = list(slots[index][1:])
            out[index] = float(probability @ np.prod(circuit.Zm[:, columns], axis=1))
    return out


def train(model, slots):
    rng = np.random.default_rng(20261012)
    circuit = M.S.Circuit(N, LAYERS, True)
    theta = rng.uniform(-math.pi, math.pi, circuit.npar)
    best = (math.inf, theta.copy())
    history = []
    step_index = 0
    for stage, beta in enumerate((3.0, 7.0, 12.0)):
        for local in range(180):
            step_index += 1
            a = 0.12 / ((step_index + 20) ** 0.602)
            c = 0.16 / (step_index ** 0.101)
            delta = rng.choice((-1.0, 1.0), size=len(theta))
            plus = arbitrary_expectation(circuit, theta + c * delta, slots)
            minus = arbitrary_expectation(circuit, theta - c * delta, slots)
            lp = M.loss_and_gradient(model, plus, beta, stage == 2)[0]
            lm = M.loss_and_gradient(model, minus, beta, stage == 2)[0]
            theta -= a * ((lp - lm) / (2 * c)) * delta
            current_e = arbitrary_expectation(circuit, theta, slots)
            current = M.loss_and_gradient(model, current_e, beta, stage == 2)[0]
            if current < best[0]:
                best = (float(current), theta.copy())
            if local % 20 == 0:
                history.append({"stage": stage, "iteration": local, "loss": float(current)})
    return circuit, best[1], arbitrary_expectation(circuit, best[1], slots), history


def qasm(theta, basis):
    values = theta.reshape(LAYERS, N, 2)
    lines = ["OPENQASM 2.0;", 'include "qelib1.inc";', f"qreg q[{N}];", f"creg c[{N}];"]
    for layer in range(LAYERS):
        for q in range(N):
            lines.append(f"ry({values[layer,q,0]:.17g}) q[{q}];")
            lines.append(f"rz({values[layer,q,1]:.17g}) q[{q}];")
        for q in range(layer % 2, N - 1, 2):
            lines.append(f"cz q[{q}],q[{q+1}];")
    if basis == "X":
        lines.extend(f"h q[{q}];" for q in range(N))
    elif basis == "Y":
        lines.extend(f"sdg q[{q}];" for q in range(N))
        lines.extend(f"h q[{q}];" for q in range(N))
    lines.extend(f"measure q[{q}] -> c[{q}];" for q in range(N))
    return "\n".join(lines) + "\n"


def pattern_qasm(bits):
    lines = ["OPENQASM 2.0;", 'include "qelib1.inc";', f"qreg q[{N}];", f"creg c[{N}];"]
    lines.extend(f"x q[{q}];" for q, bit in enumerate(bits) if bit == "1")
    lines.extend(f"measure q[{q}] -> c[{q}];" for q in range(N))
    return "\n".join(lines) + "\n"


def prepare(problem, model, generation_seed):
    slots = balanced_slots(N, limit=len(model["lin"]))
    circuit, theta, ideal_e, history = train(model, slots)
    ideal_choices, ideal_result, evaluated = M.decode(problem, ideal_e, M.TOPK)
    optimum = M.exact_enumeration(problem)
    logical = {
        "zero": pattern_qasm("0" * N), "one": pattern_qasm("1" * N),
        "alternating": pattern_qasm("01" * (N // 2)), "pairs": pattern_qasm("0011" * (N // 4)),
        **{f"pce_{basis}": qasm(theta, basis) for basis in "ZXY"},
    }
    (HERE / "logical").mkdir(exist_ok=True)
    for label, text in logical.items():
        (HERE / "logical" / f"{label}.qasm").write_text(text, encoding="utf-8")
    protocol = {
        "created_utc": now(), "seed": SEED, "generation_seed": generation_seed,
        "n_qubits": N, "layers": LAYERS, "shots": SHOTS,
        "slots": [list(slot) for slot in slots], "pair_slots": sum(len(slot) == 3 for slot in slots),
        "triple_slots": sum(len(slot) == 4 for slot in slots),
        "ideal_correlations": ideal_e.tolist(), "ideal_choices": ideal_choices,
        "ideal_result": M.clean(ideal_result), "joint_candidates_evaluated": evaluated,
        "exact_optimum": M.clean(optimum), "problem": M.clean(M.compact_problem(problem)),
        "training_history": history,
        "logical_sha256": {label: sha256(HERE / "logical" / f"{label}.qasm") for label in logical},
    }
    atomic_json(HERE / "protocol.json", protocol)
    return protocol


def compile_chip(chip, logical_labels):
    from quark.circuit import Backend, Transpiler
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        backend = Backend(chip)
        calibration = dict(backend.chip_info)
    path, edges, qubits, _ = quality_path(calibration, N)
    targets = path[:N]
    folder = HERE / chip / "circuits"; folder.mkdir(parents=True, exist_ok=True)
    tasks = []
    for label in logical_labels:
        logical = (HERE / "logical" / f"{label}.qasm").read_text(encoding="utf-8")
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            compiled = Transpiler(backend).run(logical, target_qubits=targets, optimize_level=1, niter=5)
        output = folder / f"{label}.qasm"
        output.write_text(str(compiled.to_openqasm2), encoding="utf-8")
        tasks.append({"label": label, "file": str(output.relative_to(HERE)).replace("\\", "/"),
                      "sha256": sha256(output), "shots": SHOTS})
    info = {"chip": chip, "calibration_time": calibration.get("calibration_time"),
            "n_physical_qubits": len(calibration.get("qubits_info", {})),
            "n_couplers": len(calibration.get("couplers_info", {})), "nested_path": path,
            "targets": targets, "minimum_qubit_fidelity": min(float(qubits[q].get("fidelity") or 0) for q in path),
            "minimum_edge_fidelity": min(edges[tuple(sorted(edge))] for edge in zip(path, path[1:])),
            "tasks": tasks}
    atomic_json(HERE / chip / "manifest.json", info)
    atomic_json(HERE / chip / "calibration.json", calibration)
    return info


def normalize(raw):
    source = raw.get("count") or raw.get("counts") or {}
    counts = {}
    for key, value in source.items():
        bits = str(key).replace(" ", "")
        if len(bits) != N:
            raise ValueError("返回位串长度错误")
        logical = bits[::-1]
        counts[logical] = counts.get(logical, 0) + int(value)
    if sum(counts.values()) != SHOTS:
        raise ValueError("返回测量次数错误")
    return counts


def submit_and_collect(chip, manifest):
    from quark import Task
    manager = Task(token())
    ledger_path = HERE / chip / "ledger.json"
    ledger = json.loads(ledger_path.read_text()) if ledger_path.exists() else {}
    for task in manifest["tasks"]:
        label = task["label"]
        if label in ledger:
            continue
        ledger[label] = {"status": "SUBMITTING", "intended_utc": now(), "sha256": task["sha256"]}
        atomic_json(ledger_path, ledger)
        payload = {"chip": chip, "name": f"CPDPTW_PCE_20261012_{chip}_{label}",
                   "circuit": (HERE / task["file"]).read_text(encoding="utf-8"), "shots": SHOTS,
                   "compile": False, "options": {"compiler": None, "correct": False,
                                                   "open_dd": None, "target_qubits": []}}
        tid = manager.run(payload)
        ledger[label].update({"status": "SUBMITTED", "task_id": int(tid), "submitted_utc": now()})
        atomic_json(ledger_path, ledger)
        print(json.dumps({"event": "submitted", "chip": chip, "label": label, "task_id": tid}, ensure_ascii=False), flush=True)
    deadline = time.monotonic() + 43200
    raw_dir = HERE / chip / "raw"; raw_dir.mkdir(exist_ok=True)
    while True:
        pending = [task for task in manifest["tasks"] if ledger[task["label"]]["status"] != "DONE"]
        if not pending:
            break
        for task in pending:
            label = task["label"]
            try:
                raw = manager.result(ledger[label]["task_id"])
            except Exception:
                continue
            if not isinstance(raw, dict) or not (raw.get("count") or raw.get("counts")):
                continue
            counts = normalize(raw)
            atomic_json(raw_dir / f"{label}.json", raw)
            ledger[label].update({"status": "DONE", "received_utc": now(), "shots_returned": sum(counts.values())})
            atomic_json(ledger_path, ledger)
            print(json.dumps({"event": "received", "chip": chip, "label": label}, ensure_ascii=False), flush=True)
        if time.monotonic() > deadline:
            raise TimeoutError("等待真机返回超时")
        time.sleep(10)
    return {task["label"]: normalize(json.loads((raw_dir / f"{task['label']}.json").read_text()))
            for task in manifest["tasks"]}


def analyze(chip, counts, protocol):
    slots = [tuple(slot) for slot in protocol["slots"]]
    readout = readout_parameters(counts["zero"], counts["one"], N)
    ideal = np.asarray(protocol["ideal_correlations"])
    raw_e = np.zeros(len(slots)); corrected_e = np.zeros(len(slots))
    for basis in "ZXY":
        indexes = [i for i, slot in enumerate(slots) if slot[0] == basis]
        subset = [slots[i] for i in indexes]
        raw_e[indexes] = correlations_from_counts(counts[f"pce_{basis}"], subset)
        corrected_e[indexes] = correlations_from_counts(counts[f"pce_{basis}"], subset, readout)
    problem, _, _ = build_problem()
    outputs = {}
    for name, values in (("ideal", ideal), ("hardware_raw", raw_e), ("hardware_corrected", corrected_e)):
        choices, result, evaluated = M.decode(problem, values, M.TOPK)
        outputs[name] = {"choices": choices, "result": M.clean(result), "evaluated": evaluated}
    pair = np.array([len(slot) == 3 for slot in slots]); triple = ~pair
    report = {
        "chip": chip, "readout_p00": readout["p00"].tolist(), "readout_p11": readout["p11"].tolist(),
        "readout_minimum": float(min(readout["p00"].min(), readout["p11"].min())),
        "pair_mean_absolute_error_raw": float(np.mean(np.abs(raw_e[pair] - ideal[pair]))),
        "pair_mean_absolute_error_corrected": float(np.mean(np.abs(corrected_e[pair] - ideal[pair]))),
        "triple_mean_absolute_error_raw": float(np.mean(np.abs(raw_e[triple] - ideal[triple]))),
        "triple_mean_absolute_error_corrected": float(np.mean(np.abs(corrected_e[triple] - ideal[triple]))),
        "outputs": outputs, "exact_optimum": protocol["exact_optimum"],
    }
    atomic_json(HERE / chip / "result.json", report)
    return report


def main():
    problem, model, generation_seed = build_problem()
    protocol = prepare(problem, model, generation_seed)
    manifests = {chip: compile_chip(chip, ("zero", "one", "alternating", "pairs", "pce_Z", "pce_X", "pce_Y"))
                 for chip in CHIPS}
    atomic_json(HERE / "manifests.json", manifests)
    for chip in CHIPS:
        counts = submit_and_collect(chip, manifests[chip])
        report = analyze(chip, counts, protocol)
        print(json.dumps({"event": "complete", "chip": chip, "readout_minimum": report["readout_minimum"],
                          "raw": report["outputs"]["hardware_raw"],
                          "corrected": report["outputs"]["hardware_corrected"]}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
