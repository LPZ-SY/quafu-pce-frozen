# -*- coding: utf-8 -*-
"""CPDPTW 复合插入动作的泡利关联编码机制实验。"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import itertools
import json
import math
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

HERE = Path(__file__).absolute().parent
WORKSPACE = HERE.parents[1]
LOCAL_PCE_CORE = HERE / "pce_scale.py"
WORKSPACE_PCE_CORE = WORKSPACE / "experiment_runs" / "pce_margin_20261010" / "pce_scale.py"
PCE_CORE = LOCAL_PCE_CORE if LOCAL_PCE_CORE.exists() else WORKSPACE_PCE_CORE


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载 {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


S = load_module("cpdptw_pce_core", PCE_CORE)

NQ = 8
N_VEHICLES = 4
N_REQUESTS = 12
CAPACITY = 24
HORIZON = 520.0
SERVICE = 4.0
K_REQUESTS = int(os.environ.get("CPDPTW_K_REQUESTS", "4"))
SELECTION_STRATEGY = os.environ.get("CPDPTW_SELECTION_STRATEGY", "distinct_routes")
ACTIONS_PER_REQUEST = int(os.environ.get("CPDPTW_ACTIONS_PER_REQUEST", "8"))
TOPK = 3
RESTARTS = 3
STEPS = (250, 250, 250)
BETAS = (3.0, 7.0, 12.0)
METHODS = ("Q", "P", "C")


def distance_matrix(points):
    return np.linalg.norm(points[:, None, :] - points[None, :, :], axis=-1)


def evaluate_route(instance, route):
    D = instance["distance"]
    demand = instance["demand"]
    earliest = instance["earliest"]
    latest = instance["latest"]
    service = instance["service"]
    now = 0.0
    load = 0.0
    distance = 0.0
    violation = 0.0
    previous = 0
    minimum_slack = math.inf
    for node in route:
        travel = float(D[previous, node])
        distance += travel
        now += travel
        start = max(now, float(earliest[node]))
        violation += max(0.0, start - float(latest[node]))
        minimum_slack = min(minimum_slack, float(latest[node]) - start)
        now = start + float(service[node])
        load += float(demand[node])
        violation += max(0.0, -load) + max(0.0, load - CAPACITY)
        previous = node
    distance += float(D[previous, 0])
    now += float(D[previous, 0])
    violation += max(0.0, now - HORIZON)
    if minimum_slack is math.inf:
        minimum_slack = HORIZON
    return {
        "feasible": bool(violation <= 1e-9),
        "distance": distance,
        "end_time": now,
        "violation": violation,
        "minimum_slack": minimum_slack,
    }


def make_instance(seed):
    """先生成可行基准路线，再围绕基准到达时刻生成时间窗。"""
    rng = np.random.default_rng(seed)
    points = [np.array([50.0, 50.0])]
    demand = [0.0]
    service = [0.0]
    request_nodes = []
    routes = [[] for _ in range(N_VEHICLES)]
    quantities = rng.integers(3, 9, N_REQUESTS)
    for request in range(N_REQUESTS):
        vehicle = request % N_VEHICLES
        angle = 2 * np.pi * vehicle / N_VEHICLES + rng.normal(0, 0.12)
        radius = rng.uniform(17, 31)
        center = np.array([50 + radius * np.cos(angle), 50 + radius * np.sin(angle)])
        pickup = np.clip(center + rng.normal(0, 5, 2), 2, 98)
        direction = np.array([np.cos(angle + rng.normal(0, 0.45)), np.sin(angle + rng.normal(0, 0.45))])
        delivery = np.clip(pickup + direction * rng.uniform(4, 13) + rng.normal(0, 2, 2), 2, 98)
        pickup_id = len(points)
        points.append(pickup)
        demand.append(float(quantities[request]))
        service.append(SERVICE)
        delivery_id = len(points)
        points.append(delivery)
        demand.append(float(-quantities[request]))
        service.append(SERVICE)
        request_nodes.append((pickup_id, delivery_id))
        routes[vehicle].extend([pickup_id, delivery_id])
    points = np.asarray(points, float)
    D = distance_matrix(points)
    earliest = np.zeros(len(points), float)
    latest = np.full(len(points), HORIZON, float)
    # 先计算无时间窗基准到达，再生成有余量但并不宽松的时间窗。
    for route in routes:
        now = 0.0
        previous = 0
        for node in route:
            now += D[previous, node]
            width_before = rng.uniform(18, 34)
            width_after = rng.uniform(35, 70)
            earliest[node] = max(0.0, now - width_before)
            latest[node] = now + width_after
            now += SERVICE
            previous = node
    instance = {
        "seed": seed,
        "points": points,
        "distance": D,
        "demand": np.asarray(demand),
        "service": np.asarray(service),
        "earliest": earliest,
        "latest": latest,
        "request_nodes": request_nodes,
        "routes": routes,
    }
    checks = [evaluate_route(instance, route) for route in routes]
    if not all(check["feasible"] for check in checks):
        raise RuntimeError("基准路线生成失败")
    return instance


def route_cost(instance, routes):
    results = [evaluate_route(instance, route) for route in routes]
    return {
        "feasible": all(result["feasible"] for result in results),
        "distance": float(sum(result["distance"] for result in results)),
        "violation": float(sum(result["violation"] for result in results)),
        "minimum_slack": float(min(result["minimum_slack"] for result in results)),
    }


def valid_gap_orders(events):
    if len(events) <= 1:
        return [tuple(events)]
    output = []
    for order in itertools.permutations(events):
        position = {event: index for index, event in enumerate(order)}
        ok = True
        for request, kind, _ in events:
            pickup = (request, 0, next(node for req, k, node in events if req == request and k == 0)) if any(req == request and k == 0 for req, k, _ in events) else None
            delivery = (request, 1, next(node for req, k, node in events if req == request and k == 1)) if any(req == request and k == 1 for req, k, _ in events) else None
            if pickup is not None and delivery is not None and position[pickup] > position[delivery]:
                ok = False
                break
        if ok:
            output.append(order)
    return output


def best_route_with_events(instance, skeleton, events_by_gap):
    gap_options = []
    for gap in range(len(skeleton) + 1):
        events = events_by_gap.get(gap, [])
        gap_options.append(valid_gap_orders(events))
    best = None
    for orders in itertools.product(*gap_options):
        route = []
        for gap, ordered in enumerate(orders):
            route.extend(event[2] for event in ordered)
            if gap < len(skeleton):
                route.append(skeleton[gap])
        result = evaluate_route(instance, route)
        rank = (not result["feasible"], result["violation"], result["distance"], tuple(route))
        if best is None or rank < best[0]:
            best = (rank, route, result)
    return best[1], best[2]


def apply_actions(problem, choices):
    instance = problem["instance"]
    events = [{} for _ in range(N_VEHICLES)]
    for block, action_index in enumerate(choices):
        action = problem["actions"][block][int(action_index)]
        vehicle = action["vehicle"]
        events[vehicle].setdefault(action["pickup_gap"], []).append((block, 0, action["pickup"]))
        events[vehicle].setdefault(action["delivery_gap"], []).append((block, 1, action["delivery"]))
    routes = []
    route_results = []
    for vehicle, skeleton in enumerate(problem["skeletons"]):
        route, result = best_route_with_events(instance, skeleton, events[vehicle])
        routes.append(route)
        route_results.append(result)
    return {
        "feasible": all(result["feasible"] for result in route_results),
        "distance": float(sum(result["distance"] for result in route_results)),
        "violation": float(sum(result["violation"] for result in route_results)),
        "minimum_slack": float(min(result["minimum_slack"] for result in route_results)),
        "routes": routes,
    }


def original_gap(original_route, skeleton, node):
    position = original_route.index(node)
    selected = set(original_route) - set(skeleton)
    return sum(1 for item in original_route[:position] if item not in selected)


def build_problem(seed):
    instance = make_instance(seed)
    rng = np.random.default_rng(seed + 991)
    if SELECTION_STRATEGY == "distinct_routes":
        if K_REQUESTS > N_VEHICLES:
            raise ValueError("distinct_routes 要求局部请求数不超过车辆数")
        # 每条路线各取一个请求，作为低冲突基准。
        selected = []
        for vehicle in range(K_REQUESTS):
            candidates = [request for request in range(N_REQUESTS) if request % N_VEHICLES == vehicle]
            selected.append(int(rng.choice(candidates)))
    elif SELECTION_STRATEGY == "cluster_221":
        if K_REQUESTS != 5:
            raise ValueError("cluster_221 固定要求五个局部请求")
        # 连续三条路线按2、2、1个请求取样，制造共享路线和时间窗竞争。
        first = int(rng.integers(0, N_VEHICLES))
        counts = (2, 2, 1)
        selected = []
        for offset, count in enumerate(counts):
            vehicle = (first + offset) % N_VEHICLES
            candidates = np.asarray([request for request in range(N_REQUESTS) if request % N_VEHICLES == vehicle], int)
            selected.extend(map(int, rng.choice(candidates, size=count, replace=False)))
    else:
        raise ValueError(f"未知请求选择策略：{SELECTION_STRATEGY}")
    selected_nodes = set(node for request in selected for node in instance["request_nodes"][request])
    skeletons = [[node for node in route if node not in selected_nodes] for route in instance["routes"]]
    skeleton_cost = route_cost(instance, skeletons)["distance"]
    actions = []
    for block, request in enumerate(selected):
        pickup, delivery = instance["request_nodes"][request]
        candidates = []
        for vehicle, skeleton in enumerate(skeletons):
            for pickup_gap in range(len(skeleton) + 1):
                for delivery_gap in range(pickup_gap, len(skeleton) + 1):
                    temporary = {
                        "instance": instance,
                        "skeletons": skeletons,
                        "actions": [[{
                            "vehicle": vehicle, "pickup_gap": pickup_gap, "delivery_gap": delivery_gap,
                            "pickup": pickup, "delivery": delivery,
                        }]],
                    }
                    result = apply_actions(temporary, [0])
                    if result["feasible"]:
                        candidates.append({
                            "vehicle": vehicle,
                            "pickup_gap": pickup_gap,
                            "delivery_gap": delivery_gap,
                            "pickup": pickup,
                            "delivery": delivery,
                            "single_distance": result["distance"],
                            "single_delta": result["distance"] - skeleton_cost,
                            "single_slack": result["minimum_slack"],
                        })
        source_vehicle = selected[block] % N_VEHICLES
        original = instance["routes"][source_vehicle]
        original_action_key = (
            source_vehicle,
            original_gap(original, skeletons[source_vehicle], pickup),
            original_gap(original, skeletons[source_vehicle], delivery),
        )
        candidates.sort(key=lambda action: (action["single_delta"], -action["single_slack"], action["vehicle"], action["pickup_gap"], action["delivery_gap"]))
        kept = candidates[:ACTIONS_PER_REQUEST]
        original_action = next((action for action in candidates if (action["vehicle"], action["pickup_gap"], action["delivery_gap"]) == original_action_key), None)
        if original_action is None:
            raise RuntimeError(f"原始复合动作未通过可行性检查：请求={request}，键={original_action_key}，可行动作数={len(candidates)}")
        if original_action not in kept:
            kept[-1] = original_action
            kept.sort(key=lambda action: (action["single_delta"], action["vehicle"], action["pickup_gap"], action["delivery_gap"]))
        if len(kept) < ACTIONS_PER_REQUEST:
            raise RuntimeError("可行复合动作不足")
        for index, action in enumerate(kept):
            action["index"] = index
        actions.append(kept)
    problem = {"instance": instance, "selected_requests": selected, "skeletons": skeletons, "actions": actions}
    original_choices = []
    for block, request in enumerate(selected):
        pickup, delivery = instance["request_nodes"][request]
        vehicle = request % N_VEHICLES
        key = (vehicle, original_gap(instance["routes"][vehicle], skeletons[vehicle], pickup), original_gap(instance["routes"][vehicle], skeletons[vehicle], delivery))
        original_choices.append(next(index for index, action in enumerate(actions[block]) if (action["vehicle"], action["pickup_gap"], action["delivery_gap"]) == key))
    baseline = apply_actions(problem, original_choices)
    if not baseline["feasible"]:
        raise RuntimeError("基准复合动作不可行")
    problem["original_choices"] = original_choices
    problem["baseline"] = baseline
    problem["skeleton_cost"] = skeleton_cost
    return problem


def build_quadratic_model(problem):
    m = K_REQUESTS * ACTIONS_PER_REQUEST
    lin = np.zeros(m, float)
    for block in range(K_REQUESTS):
        for action in range(ACTIONS_PER_REQUEST):
            lin[block * ACTIONS_PER_REQUEST + action] = problem["actions"][block][action]["single_delta"]
    pairs = []
    weights = []
    feasible_pairs = 0
    for left in range(K_REQUESTS):
        for right in range(left + 1, K_REQUESTS):
            for a in range(ACTIONS_PER_REQUEST):
                for b in range(ACTIONS_PER_REQUEST):
                    result = apply_actions(problem, [a if block == left else b if block == right else problem["original_choices"][block] for block in range(K_REQUESTS)])
                    i = left * ACTIONS_PER_REQUEST + a
                    j = right * ACTIONS_PER_REQUEST + b
                    base_pair = lin[i] + lin[j]
                    interaction = result["distance"] - problem["skeleton_cost"] - base_pair
                    if not result["feasible"]:
                        interaction += 50.0 + 10.0 * result["violation"]
                    else:
                        feasible_pairs += 1
                    pairs.append((i, j))
                    weights.append(interaction)
    scale = max(np.quantile(np.abs(np.r_[lin, weights]), 0.9), 1.0)
    return {
        "lin": lin / scale,
        "pairs": np.asarray(pairs, int),
        "weights": np.asarray(weights, float) / scale,
        "scale": float(scale),
        "feasible_pair_fraction": feasible_pairs / len(pairs),
    }


def softmax_blocks(e, beta):
    w = np.empty_like(e)
    for block in range(K_REQUESTS):
        indexes = slice(block * ACTIONS_PER_REQUEST, (block + 1) * ACTIONS_PER_REQUEST)
        z = beta * e[indexes]
        z -= z.max()
        exp = np.exp(z)
        w[indexes] = exp / exp.sum()
    return w


def loss_and_gradient(model, e, beta, margin):
    w = softmax_blocks(e, beta)
    lin = model["lin"]
    pairs = model["pairs"]
    weights = model["weights"]
    value = float(lin @ w + np.sum(weights * w[pairs[:, 0]] * w[pairs[:, 1]]))
    gradient_w = lin.copy()
    np.add.at(gradient_w, pairs[:, 0], weights * w[pairs[:, 1]])
    np.add.at(gradient_w, pairs[:, 1], weights * w[pairs[:, 0]])
    gradient_e = np.empty_like(e)
    for block in range(K_REQUESTS):
        indexes = np.arange(block * ACTIONS_PER_REQUEST, (block + 1) * ACTIONS_PER_REQUEST)
        gradient_e[indexes] = beta * w[indexes] * (gradient_w[indexes] - np.sum(w[indexes] * gradient_w[indexes]))
        if margin:
            order = indexes[np.argsort(-e[indexes], kind="stable")]
            lead = e[order[0]] - e[order[1]]
            if lead < 0.1:
                value += 5.0 * (0.1 - lead) ** 2
                gradient_e[order[0]] -= 10.0 * (0.1 - lead)
                gradient_e[order[1]] += 10.0 * (0.1 - lead)
    return value, gradient_e


def adam(fun, x, steps):
    first = np.zeros_like(x)
    second = np.zeros_like(x)
    best = (math.inf, x.copy())
    for step in range(1, steps + 1):
        value, gradient = fun(x)
        if value < best[0]:
            best = (float(value), x.copy())
        first = 0.9 * first + 0.1 * gradient
        second = 0.999 * second + 0.001 * gradient * gradient
        x = x - S.LR * (first / (1 - 0.9**step)) / (np.sqrt(second / (1 - 0.999**step)) + 1e-8)
    return best[1], best[0]


def decode(problem, e, topk):
    candidates = []
    for block in range(K_REQUESTS):
        indexes = np.arange(block * ACTIONS_PER_REQUEST, (block + 1) * ACTIONS_PER_REQUEST)
        local = indexes[np.argsort(-e[indexes], kind="stable")[:topk]] - block * ACTIONS_PER_REQUEST
        candidates.append(list(map(int, local)))
    best = None
    evaluated = 0
    for choices in itertools.product(*candidates):
        result = apply_actions(problem, choices)
        evaluated += 1
        rank = (not result["feasible"], result["violation"], result["distance"], tuple(choices))
        if best is None or rank < best[0]:
            best = (rank, list(choices), result)
    return best[1], best[2], evaluated


def train(problem, model, method, restart, seed):
    rng = np.random.default_rng(seed * 1000 + restart)
    m = len(model["lin"])
    slots = S.make_slots(NQ, m)
    if method in ("Q", "P"):
        circuit = S.Circuit(NQ, max(4, int(np.ceil(m / NQ))), method == "Q")
        parameters = rng.uniform(-np.pi, np.pi, circuit.npar)

        def expectation(theta):
            return circuit.expect(circuit.run(theta), slots)

        def objective(beta, margin):
            def wrapped(theta):
                e = expectation(theta)
                value, gradient_e = loss_and_gradient(model, e, beta, margin)
                return value, circuit.grad(theta, slots, gradient_e)
            return wrapped
        ncz = circuit.ncz
        npar = circuit.npar
    else:
        parameters = rng.normal(0, 1, m)

        def expectation(theta):
            return np.tanh(theta)

        def objective(beta, margin):
            def wrapped(theta):
                e = expectation(theta)
                value, gradient_e = loss_and_gradient(model, e, beta, margin)
                return value, gradient_e * (1 - e * e)
            return wrapped
        ncz = 0
        npar = m
    stage_loss = []
    started = time.time()
    for index, (steps, beta) in enumerate(zip(STEPS, BETAS)):
        parameters, value = adam(objective(beta, index == len(STEPS) - 1), parameters, steps)
        stage_loss.append(value)
    e = expectation(parameters)
    argmax_choices, argmax_result, _ = decode(problem, e, 1)
    joint_choices, joint_result, evaluated = decode(problem, e, TOPK)
    return {
        "method": method,
        "restart": restart,
        "stage_loss": stage_loss,
        "argmax_choices": argmax_choices,
        "argmax": argmax_result,
        "joint_choices": joint_choices,
        "joint": joint_result,
        "joint_candidates_evaluated": evaluated,
        "ncz": ncz,
        "npar": npar,
        "objective_evaluations": sum(STEPS),
        "seconds": round(time.time() - started, 3),
    }


def exact_enumeration(problem):
    best = None
    feasible = 0
    total = 0
    for choices in itertools.product(range(ACTIONS_PER_REQUEST), repeat=K_REQUESTS):
        result = apply_actions(problem, choices)
        total += 1
        if result["feasible"]:
            feasible += 1
            rank = (result["distance"], tuple(choices))
            if best is None or rank < best[0]:
                best = (rank, list(choices), result)
    return {"choices": best[1], "result": best[2], "feasible": feasible, "total": total}


def greedy_regret(problem):
    chosen = [None] * K_REQUESTS
    remaining = set(range(K_REQUESTS))
    while remaining:
        block_options = {}
        for block in remaining:
            scored = []
            for action in range(ACTIONS_PER_REQUEST):
                trial = [problem["original_choices"][b] if chosen[b] is None else chosen[b] for b in range(K_REQUESTS)]
                trial[block] = action
                result = apply_actions(problem, trial)
                scored.append((not result["feasible"], result["violation"], result["distance"], action))
            scored.sort()
            regret = scored[1][2] - scored[0][2]
            block_options[block] = (regret, scored[0])
        block = max(remaining, key=lambda b: (block_options[b][0], -b))
        chosen[block] = int(block_options[block][1][3])
        remaining.remove(block)
    return {"choices": chosen, "result": apply_actions(problem, chosen)}


def clean(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    return value


def compact_problem(problem):
    instance = problem["instance"]
    return {
        "seed": instance["seed"],
        "selected_requests": problem["selected_requests"],
        "original_choices": problem["original_choices"],
        "baseline": problem["baseline"],
        "actions": problem["actions"],
        "time_windows": [[float(instance["earliest"][node]), float(instance["latest"][node])] for node in range(len(instance["points"]))],
        "demands": instance["demand"],
    }


def run_instance(seed):
    started = time.time()
    problem = None
    generation_seed = None
    retries = 0
    for attempt in range(20):
        candidate_seed = seed + attempt * 1_000_000
        try:
            problem = build_problem(candidate_seed)
            generation_seed = candidate_seed
            retries = attempt
            break
        except RuntimeError as error:
            if "可行复合动作不足" not in str(error):
                raise
    if problem is None:
        raise RuntimeError(f"外层种子 {seed} 在20次机械重试后仍没有足够可行动作")
    model = build_quadratic_model(problem)
    optimum = exact_enumeration(problem)
    methods = {method: [train(problem, model, method, restart, generation_seed) for restart in range(RESTARTS)] for method in METHODS}
    return {
        "seed": seed,
        "generation_seed": generation_seed,
        "input_retries": retries,
        "problem": compact_problem(problem),
        "model": {"scale": model["scale"], "feasible_pair_fraction": model["feasible_pair_fraction"]},
        "optimum": optimum,
        "greedy_regret": greedy_regret(problem),
        "methods": methods,
        "seconds": round(time.time() - started, 3),
    }


def summarize(rows):
    output = {"instances": len(rows), "methods": {}}
    for method in METHODS:
        hit = []
        best_regret = []
        argmax_regret = []
        feasible_runs = 0
        for row in rows:
            optimum = row["optimum"]["result"]["distance"]
            baseline = row["problem"]["baseline"]["distance"]
            runs = row["methods"][method]
            joint = [100 * (run["joint"]["distance"] - optimum) / baseline for run in runs if run["joint"]["feasible"]]
            argmax = [100 * (run["argmax"]["distance"] - optimum) / baseline for run in runs if run["argmax"]["feasible"]]
            feasible_runs += sum(run["joint"]["feasible"] for run in runs)
            value = min(joint) if joint else 100.0
            best_regret.append(value)
            argmax_regret.append(min(argmax) if argmax else 100.0)
            hit.append(value < 1e-9)
        output["methods"][method] = {
            "best_of_restarts_optimum_hit_rate": float(np.mean(hit)),
            "best_of_restarts_mean_regret_pct_baseline": float(np.mean(best_regret)),
            "argmax_mean_regret_pct_baseline": float(np.mean(argmax_regret)),
            "joint_decode_gain_percentage_points": float(np.mean(np.asarray(argmax_regret) - np.asarray(best_regret))),
            "joint_feasible_run_rate": feasible_runs / (len(rows) * RESTARTS),
        }
    greedy_regret_values = []
    greedy_hits = []
    feasible_fraction = []
    for row in rows:
        optimum = row["optimum"]["result"]["distance"]
        baseline = row["problem"]["baseline"]["distance"]
        value = 100 * (row["greedy_regret"]["result"]["distance"] - optimum) / baseline
        greedy_regret_values.append(value)
        greedy_hits.append(value < 1e-9)
        feasible_fraction.append(row["optimum"]["feasible"] / row["optimum"]["total"])
    output["greedy_regret"] = {
        "optimum_hit_rate": float(np.mean(greedy_hits)),
        "mean_regret_pct_baseline": float(np.mean(greedy_regret_values)),
    }
    output["mean_feasible_combination_fraction"] = float(np.mean(feasible_fraction))
    return output


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(clean(value), ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed-start", type=int, default=71001)
    parser.add_argument("--instances", type=int, default=12)
    parser.add_argument("--processes", type=int, default=max(1, min(10, os.cpu_count() or 1)))
    parser.add_argument("--tag", default="main")
    args = parser.parse_args()
    tasks = list(range(args.seed_start, args.seed_start + args.instances))
    rows = []
    started = time.time()
    with Pool(args.processes) as pool:
        for row in pool.imap_unordered(run_instance, tasks):
            rows.append(row)
            rows.sort(key=lambda item: item["seed"])
            write_json(HERE / "results" / f"runs_{args.tag}.json", rows)
            print(f"完成 {len(rows)}/{len(tasks)}：种子 {row['seed']}，累计 {time.time() - started:.1f} 秒", flush=True)
    summary = summarize(rows)
    write_json(HERE / "results" / f"summary_{args.tag}.json", summary)
    write_json(HERE / "results" / f"manifest_{args.tag}.json", {
        "seed_start": args.seed_start,
        "instances": args.instances,
        "requests_per_subproblem": K_REQUESTS,
        "selection_strategy": SELECTION_STRATEGY,
        "actions_per_request": ACTIONS_PER_REQUEST,
        "full_combinations": ACTIONS_PER_REQUEST ** K_REQUESTS,
        "joint_decode_combinations": TOPK ** K_REQUESTS,
        "restarts": RESTARTS,
        "objective_evaluations": sum(STEPS),
        "methods": METHODS,
        "pce_core_sha256": sha256(PCE_CORE),
        "runner_sha256": sha256(Path(__file__)),
        "elapsed_seconds": round(time.time() - started, 3),
    })
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
