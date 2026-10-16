# -*- coding: utf-8 -*-
"""依据当前标定选择嵌套物理路径，并将逻辑线路编译为 Shenglian 物理线路。"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import math
import re
from pathlib import Path

from capacity_core import atomic_json, sha256

HERE = Path(__file__).absolute().parent


def quality_path(calibration, length=20, beam_width=12000):
    qubits = {int(row["index"]): row for row in calibration["qubits_info"].values()}
    edges = {}
    graph = {q: set() for q in qubits}
    for row in calibration["couplers_info"].values():
        a, b = map(int, row["qubits_index"])
        fidelity = float(row.get("fidelity") or 0.0)
        if fidelity <= 0:
            continue
        edge = tuple(sorted((a, b)))
        edges[edge] = fidelity
        graph[a].add(b); graph[b].add(a)

    def node_score(q):
        row = qubits[q]
        fidelity = max(float(row.get("fidelity") or 0.0), 1e-6)
        t1 = max(float(row.get("T1") or 0.0), 1.0)
        t2 = max(float(row.get("T2") or 0.0), 1.0)
        return 5.0 * math.log(fidelity) + 0.15 * math.log(t1) + 0.20 * math.log(t2)

    def edge_score(a, b):
        return 12.0 * math.log(max(edges[tuple(sorted((a, b)))], 1e-6))

    beam = [(node_score(q), (q,)) for q in graph if graph[q]]
    for _ in range(1, length):
        candidates = []
        for score, path in beam:
            for nxt in graph[path[-1]]:
                if nxt in path:
                    continue
                candidates.append((score + node_score(nxt) + edge_score(path[-1], nxt), path + (nxt,)))
        if not candidates:
            raise RuntimeError(f"当前有效耦合图不存在长度为{length}的简单路径")
        candidates.sort(key=lambda item: item[0], reverse=True)
        seen = set(); beam = []
        for item in candidates:
            signature = (item[1][-1], frozenset(item[1]))
            if signature in seen:
                continue
            seen.add(signature); beam.append(item)
            if len(beam) >= beam_width:
                break
    score, path = max(beam, key=lambda item: item[0])
    # 选择前缀质量更好的一端作为小规模起点。
    reverse = tuple(reversed(path))
    def prefix_score(candidate):
        return sum((length - i) * node_score(q) for i, q in enumerate(candidate))
    if prefix_score(reverse) > prefix_score(path):
        path = reverse
    return list(path), edges, qubits, score


def compile_all(root):
    root = Path(root)
    if (root / "hardware_ledger.json").exists():
        raise RuntimeError("已存在真机台账，禁止重新编译和改变映射")
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    calibration = json.loads((root / "calibration.json").read_text(encoding="utf-8"))
    config_path = root / "backend_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8")) if config_path.exists() else {"chip": "Shenglian"}
    chip = str(config["chip"])
    path, edges, qubits, score = quality_path(calibration, max(manifest["sizes"]))
    from quark.circuit import Backend, Transpiler
    backend = Backend(chip)
    compiled_tasks = []
    for task in manifest["tasks"]:
        n = int(task["n"])
        targets = path[:n]
        logical_path = root / task["file"]
        if sha256(logical_path) != task["sha256"]:
            raise RuntimeError(f"逻辑线路散列不匹配：{task['file']}")
        logical = logical_path.read_text(encoding="utf-8")
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            compiled = Transpiler(backend).run(logical, target_qubits=targets, optimize_level=1, niter=5)
        physical = str(compiled.to_openqasm2)
        measured = sorted({int(value) for value in re.findall(r"measure\s+q\[(\d+)\]", physical)})
        if measured != sorted(targets):
            raise RuntimeError(f"测量物理映射不匹配：{task['label']} {measured} != {sorted(targets)}")
        used_edges = []
        for match in re.finditer(r"(?:cz|cx)\s+q\[(\d+)\]\s*,\s*q\[(\d+)\]", physical, re.I):
            edge = tuple(sorted((int(match.group(1)), int(match.group(2)))))
            used_edges.append(edge)
            if edge not in edges:
                raise RuntimeError(f"线路使用未标定耦合边：{task['label']} {edge}")
        relative = Path("physical_circuits") / f"n{n:02d}" / f"{task['label']}.qasm"
        output = root / relative
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(physical, encoding="utf-8")
        compiled_tasks.append({**task, "logical_file": task["file"], "logical_sha256": task["sha256"],
                               "file": relative.as_posix(), "sha256": sha256(output),
                               "target_qubits": targets, "physical_two_qubit_gates": len(used_edges)})
    mapping = {
        "chip": chip, "calibration_time": calibration.get("calibration_time"),
        "nested_path": path, "path_score": score,
        "qubits": {str(q): {"fidelity": qubits[q].get("fidelity"), "T1": qubits[q].get("T1"),
                             "T2": qubits[q].get("T2")} for q in path},
        "edges": [{"qubits": [a, b], "fidelity": edges[tuple(sorted((a, b)))]}
                  for a, b in zip(path, path[1:])],
    }
    atomic_json(root / "mapping.json", mapping)
    physical_manifest = {key: value for key, value in manifest.items() if key != "tasks"}
    physical_manifest.update({"chip": chip, "compile": False, "mapping_file": "mapping.json",
                              "tasks": compiled_tasks})
    atomic_json(root / "physical_manifest.json", physical_manifest)
    seal = {task["file"]: task["sha256"] for task in compiled_tasks}
    seal["mapping.json"] = sha256(root / "mapping.json")
    seal["physical_manifest.json"] = sha256(root / "physical_manifest.json")
    atomic_json(root / "physical_input_seal.json", {"files_sha256": seal})
    print(json.dumps({"path": path, "tasks": len(compiled_tasks),
                      "minimum_qubit_fidelity": min(float(qubits[q].get("fidelity") or 0) for q in path),
                      "minimum_edge_fidelity": min(edges[tuple(sorted(edge))] for edge in zip(path, path[1:]))},
                     ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=HERE)
    compile_all(parser.parse_args().root)
