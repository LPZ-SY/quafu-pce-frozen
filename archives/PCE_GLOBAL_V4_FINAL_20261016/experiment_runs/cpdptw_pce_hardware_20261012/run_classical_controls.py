# -*- coding: utf-8 -*-
"""冻结真机实例上的81组合经典对照与完整CPDPTW强求解器对照。"""
from __future__ import annotations

import itertools
import json
import math
import os
import time
from pathlib import Path

import numpy as np
from ortools.constraint_solver import pywrapcp, routing_enums_pb2

os.environ["CPDPTW_K_REQUESTS"] = "4"
os.environ["CPDPTW_ACTIONS_PER_REQUEST"] = "8"
os.environ["CPDPTW_SELECTION_STRATEGY"] = "distinct_routes"

import cpdptw_core as M


HERE = Path(__file__).parent
RESULTS = HERE / "classical_controls"
SEED = 20261012
LOCAL_BUDGET = 81
REPEATS = 100
CLASSICAL_RESTARTS = 10
DISTANCE_SCALE = 1000
FULL_SOLVER_SECONDS = 30


def clean(value):
    return M.clean(value)


def load_problem():
    protocol = json.loads((HERE / "protocol.json").read_text(encoding="utf-8"))
    problem = M.build_problem(int(protocol["generation_seed"]))
    frozen = protocol["problem"]
    if problem["selected_requests"] != frozen["selected_requests"]:
        raise RuntimeError("冻结实例请求不一致")
    if problem["original_choices"] != frozen["original_choices"]:
        raise RuntimeError("冻结实例原始动作不一致")
    return protocol, problem


def rank(result, choices):
    return (not result["feasible"], result["violation"], result["distance"], tuple(choices))


def evaluate_choices(problem, choices_list):
    best = None
    for choices in choices_list:
        choices = list(map(int, choices))
        result = M.apply_actions(problem, choices)
        item = (rank(result, choices), choices, result)
        if best is None or item[0] < best[0]:
            best = item
    return {
        "choices": best[1], "result": clean(best[2]),
        "exact_route_evaluations": len(choices_list),
    }


def cartesian_top3(problem, candidates):
    choices = list(itertools.product(*candidates))
    if len(choices) != LOCAL_BUDGET:
        raise RuntimeError("候选组合数不是81")
    output = evaluate_choices(problem, choices)
    output["candidate_actions"] = [list(map(int, row)) for row in candidates]
    return output


def individual_insertion_top3(problem):
    candidates = []
    for actions in problem["actions"]:
        order = sorted(range(len(actions)), key=lambda a: (
            actions[a]["single_delta"], -actions[a]["single_slack"], a
        ))
        candidates.append(order[:3])
    return cartesian_top3(problem, candidates)


def surrogate_energy(model, choices):
    selected = {block * M.ACTIONS_PER_REQUEST + int(action) for block, action in enumerate(choices)}
    value = sum(model["lin"][index] for index in selected)
    for (left, right), weight in zip(model["pairs"], model["weights"]):
        if int(left) in selected and int(right) in selected:
            value += weight
    return float(value)


def quadratic_marginal_top3(problem, model):
    reference = list(problem["original_choices"])
    candidates = []
    for block in range(M.K_REQUESTS):
        scored = []
        for action in range(M.ACTIONS_PER_REQUEST):
            choices = reference.copy()
            choices[block] = action
            scored.append((surrogate_energy(model, choices), action))
        candidates.append([action for _, action in sorted(scored)[:3]])
    return cartesian_top3(problem, candidates)


def greedy_regret_81(problem):
    chosen = [None] * M.K_REQUESTS
    remaining = set(range(M.K_REQUESTS))
    evaluated = 0
    while remaining:
        block_options = {}
        for block in remaining:
            scored = []
            for action in range(M.ACTIONS_PER_REQUEST):
                trial = [problem["original_choices"][b] if chosen[b] is None else chosen[b]
                         for b in range(M.K_REQUESTS)]
                trial[block] = action
                result = M.apply_actions(problem, trial)
                evaluated += 1
                scored.append((not result["feasible"], result["violation"], result["distance"], action))
            scored.sort()
            regret = scored[1][2] - scored[0][2]
            block_options[block] = (regret, scored[0])
        block = max(remaining, key=lambda b: (block_options[b][0], -b))
        chosen[block] = int(block_options[block][1][3])
        remaining.remove(block)
    result = M.apply_actions(problem, chosen)
    evaluated += 1
    if evaluated != LOCAL_BUDGET:
        raise RuntimeError(f"贪心遗憾插入检查了{evaluated}个组合")
    return {"choices": chosen, "result": clean(result), "exact_route_evaluations": evaluated}


def random_search_81(problem, rng):
    indexes = rng.choice(M.ACTIONS_PER_REQUEST ** M.K_REQUESTS, size=LOCAL_BUDGET, replace=False)
    choices = []
    for value in indexes:
        row = []
        number = int(value)
        for _ in range(M.K_REQUESTS):
            row.append(number % M.ACTIONS_PER_REQUEST)
            number //= M.ACTIONS_PER_REQUEST
        choices.append(row)
    return evaluate_choices(problem, choices)


def simulated_annealing_81(problem, rng):
    current = list(problem["original_choices"])
    current_result = M.apply_actions(problem, current)
    best = (rank(current_result, current), current.copy(), current_result)
    evaluated = 1
    temperature_start = 12.0
    temperature_end = 0.15
    for step in range(1, LOCAL_BUDGET):
        proposal = current.copy()
        block = int(rng.integers(M.K_REQUESTS))
        alternatives = [a for a in range(M.ACTIONS_PER_REQUEST) if a != proposal[block]]
        proposal[block] = int(rng.choice(alternatives))
        result = M.apply_actions(problem, proposal)
        evaluated += 1
        delta = result["distance"] - current_result["distance"]
        if not result["feasible"]:
            delta += 1000.0 + 100.0 * result["violation"]
        temperature = temperature_start * (temperature_end / temperature_start) ** (step / (LOCAL_BUDGET - 1))
        if delta <= 0 or rng.random() < math.exp(-delta / temperature):
            current, current_result = proposal, result
        item = (rank(result, proposal), proposal.copy(), result)
        if item[0] < best[0]:
            best = item
    return {"choices": best[1], "result": clean(best[2]), "exact_route_evaluations": evaluated}


def classical_direct_top3(problem, model, restart):
    trained = M.train(problem, model, "C", restart, SEED)
    # M.train已经使用相同的前三名乘积解码，因此精确检查数固定为81。
    return {
        "choices": trained["joint_choices"], "result": clean(trained["joint"]),
        "candidate_actions": None,
        "exact_route_evaluations": int(trained["joint_candidates_evaluated"]),
        "surrogate_objective_evaluations": int(trained["objective_evaluations"]),
        "training_seconds": float(trained["seconds"]),
    }


def summarize_runs(runs, local_optimum):
    distances = np.asarray([row["result"]["distance"] for row in runs], float)
    feasible = np.asarray([row["result"]["feasible"] for row in runs], bool)
    gaps = 100.0 * (distances - local_optimum) / local_optimum
    return {
        "runs": len(runs),
        "feasible_rate": float(feasible.mean()),
        "optimum_hit_rate": float(np.mean(np.abs(distances - local_optimum) <= 1e-9)),
        "distance_mean": float(distances.mean()),
        "distance_median": float(np.median(distances)),
        "distance_best": float(distances.min()),
        "distance_worst": float(distances.max()),
        "gap_pct_mean": float(gaps.mean()),
        "gap_pct_median": float(np.median(gaps)),
        "gap_pct_best": float(gaps.min()),
        "gap_pct_worst": float(gaps.max()),
    }


def audit_full_routes(instance, routes):
    flattened = [node for route in routes for node in route]
    expected = list(range(1, len(instance["points"])))
    pair_checks = []
    for request, (pickup, delivery) in enumerate(instance["request_nodes"]):
        pickup_vehicle = next((v for v, route in enumerate(routes) if int(pickup) in route), None)
        delivery_vehicle = next((v for v, route in enumerate(routes) if int(delivery) in route), None)
        same_vehicle = pickup_vehicle is not None and pickup_vehicle == delivery_vehicle
        precedence = bool(same_vehicle and routes[pickup_vehicle].index(int(pickup)) < routes[pickup_vehicle].index(int(delivery)))
        pair_checks.append({
            "request": request, "pickup": int(pickup), "delivery": int(delivery),
            "same_vehicle": same_vehicle, "pickup_before_delivery": precedence,
        })
    route_checks = [M.evaluate_route(instance, route) for route in routes]
    return {
        "all_non_depot_nodes_once": sorted(flattened) == expected and len(flattened) == len(set(flattened)),
        "all_pairs_same_vehicle": all(row["same_vehicle"] for row in pair_checks),
        "all_pickups_before_deliveries": all(row["pickup_before_delivery"] for row in pair_checks),
        "all_routes_capacity_and_time_feasible": all(row["feasible"] for row in route_checks),
        "pair_checks": pair_checks,
        "route_checks": clean(route_checks),
    }


def solve_full_cpdptw(instance):
    started = time.perf_counter()
    node_count = len(instance["points"])
    manager = pywrapcp.RoutingIndexManager(node_count, M.N_VEHICLES, 0)
    routing = pywrapcp.RoutingModel(manager)
    distance = np.rint(instance["distance"] * DISTANCE_SCALE).astype(np.int64)
    service = np.rint(instance["service"] * DISTANCE_SCALE).astype(np.int64)
    earliest = np.rint(instance["earliest"] * DISTANCE_SCALE).astype(np.int64)
    latest = np.rint(instance["latest"] * DISTANCE_SCALE).astype(np.int64)
    demands = np.rint(instance["demand"]).astype(np.int64)

    def distance_callback(from_index, to_index):
        return int(distance[manager.IndexToNode(from_index), manager.IndexToNode(to_index)])

    distance_id = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(distance_id)

    def time_callback(from_index, to_index):
        left = manager.IndexToNode(from_index)
        right = manager.IndexToNode(to_index)
        return int(service[left] + distance[left, right])

    time_id = routing.RegisterTransitCallback(time_callback)
    horizon = int(round(M.HORIZON * DISTANCE_SCALE))
    routing.AddDimension(time_id, horizon, horizon, True, "时间")
    time_dimension = routing.GetDimensionOrDie("时间")
    for node in range(1, node_count):
        index = manager.NodeToIndex(node)
        time_dimension.CumulVar(index).SetRange(int(earliest[node]), int(latest[node]))
    for vehicle in range(M.N_VEHICLES):
        time_dimension.CumulVar(routing.End(vehicle)).SetMax(horizon)

    def demand_callback(index):
        return int(demands[manager.IndexToNode(index)])

    demand_id = routing.RegisterUnaryTransitCallback(demand_callback)
    routing.AddDimensionWithVehicleCapacity(
        demand_id, 0, [M.CAPACITY] * M.N_VEHICLES, True, "容量"
    )

    solver = routing.solver()
    for pickup, delivery in instance["request_nodes"]:
        pickup_index = manager.NodeToIndex(int(pickup))
        delivery_index = manager.NodeToIndex(int(delivery))
        routing.AddPickupAndDelivery(pickup_index, delivery_index)
        solver.Add(routing.VehicleVar(pickup_index) == routing.VehicleVar(delivery_index))
        solver.Add(time_dimension.CumulVar(pickup_index) <= time_dimension.CumulVar(delivery_index))

    parameters = pywrapcp.DefaultRoutingSearchParameters()
    parameters.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PARALLEL_CHEAPEST_INSERTION
    parameters.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    parameters.time_limit.FromSeconds(FULL_SOLVER_SECONDS)
    parameters.log_search = False

    initial = routing.ReadAssignmentFromRoutes(instance["routes"], True)
    solution = routing.SolveFromAssignmentWithParameters(initial, parameters)
    if solution is None:
        solution = routing.SolveWithParameters(parameters)
    if solution is None:
        raise RuntimeError("完整CPDPTW强求解器未找到可行解")

    routes = []
    for vehicle in range(M.N_VEHICLES):
        index = routing.Start(vehicle)
        route = []
        while not routing.IsEnd(index):
            node = manager.IndexToNode(index)
            if node != 0:
                route.append(int(node))
            index = solution.Value(routing.NextVar(index))
        routes.append(route)
    result = M.route_cost(instance, routes)
    result["routes"] = routes
    audit = audit_full_routes(instance, routes)
    if not all(audit[key] for key in (
        "all_non_depot_nodes_once", "all_pairs_same_vehicle",
        "all_pickups_before_deliveries", "all_routes_capacity_and_time_feasible",
    )):
        raise RuntimeError("完整CPDPTW求解结果未通过独立审计")
    return {
        "solver": "OR-Tools RoutingModel",
        "strategy": "并行最便宜插入 + 引导式局部搜索",
        "time_limit_seconds": FULL_SOLVER_SECONDS,
        "wall_seconds": time.perf_counter() - started,
        "integer_objective": int(solution.ObjectiveValue()),
        "result": clean(result),
        "audit": audit,
    }


def main():
    RESULTS.mkdir(exist_ok=True)
    protocol, problem = load_problem()
    model = M.build_quadratic_model(problem)
    local_optimum = float(protocol["exact_optimum"]["result"]["distance"])
    started = time.perf_counter()

    deterministic = {
        "individual_insertion_top3": individual_insertion_top3(problem),
        "quadratic_marginal_top3": quadratic_marginal_top3(problem, model),
        "greedy_regret_81": greedy_regret_81(problem),
    }
    classical_runs = [classical_direct_top3(problem, model, restart)
                      for restart in range(CLASSICAL_RESTARTS)]
    random_runs = []
    annealing_runs = []
    for repeat in range(REPEATS):
        random_runs.append(random_search_81(problem, np.random.default_rng(SEED + repeat)))
        annealing_runs.append(simulated_annealing_81(problem, np.random.default_rng(SEED + 10000 + repeat)))

    full_solver = solve_full_cpdptw(problem["instance"])
    original_full = M.route_cost(problem["instance"], problem["instance"]["routes"])
    report = {
        "experiment": "冻结真机实例的公平经典对照与完整CPDPTW强求解器",
        "seed": SEED,
        "generation_seed": protocol["generation_seed"],
        "local_problem": {
            "requests": M.K_REQUESTS,
            "actions_per_request": M.ACTIONS_PER_REQUEST,
            "full_combinations": M.ACTIONS_PER_REQUEST ** M.K_REQUESTS,
            "fair_exact_route_evaluation_budget": LOCAL_BUDGET,
            "budget_fraction": LOCAL_BUDGET / (M.ACTIONS_PER_REQUEST ** M.K_REQUESTS),
            "original_feasible_route": clean(problem["baseline"]),
            "exact_local_optimum": protocol["exact_optimum"],
            "deterministic_controls": deterministic,
            "classical_direct_runs": classical_runs,
            "classical_direct_summary": summarize_runs(classical_runs, local_optimum),
            "random_search_runs": random_runs,
            "random_search_summary": summarize_runs(random_runs, local_optimum),
            "simulated_annealing_runs": annealing_runs,
            "simulated_annealing_summary": summarize_runs(annealing_runs, local_optimum),
            "hardware": {
                chip: json.loads((HERE / chip / "result.json").read_text(encoding="utf-8"))["outputs"]
                for chip in ("Baihua", "Dongling")
            },
        },
        "full_cpdptw": {
            "requests": M.N_REQUESTS,
            "vehicles": M.N_VEHICLES,
            "original_feasible_route": clean(original_full),
            "strong_solver": full_solver,
            "scope_note": "完整24节点、12请求、4车辆问题；不是4请求局部邻域。",
        },
        "total_wall_seconds": time.perf_counter() - started,
    }
    path = RESULTS / "results.json"
    path.write_text(json.dumps(clean(report), ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "output": str(path),
        "deterministic": {name: row["result"]["distance"] for name, row in deterministic.items()},
        "classical_direct": report["local_problem"]["classical_direct_summary"],
        "random_search": report["local_problem"]["random_search_summary"],
        "simulated_annealing": report["local_problem"]["simulated_annealing_summary"],
        "full_cpdptw_original": original_full["distance"],
        "full_cpdptw_solver": full_solver["result"]["distance"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
