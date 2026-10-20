"""Запрос демонстраций до достижения одинакового числа успешных записей."""

from __future__ import annotations

import random
from collections import Counter

from act_recovery.settings import FAILURE_TYPES


def resolve_condition(method, condition, budget, seed, attempt):

    accepted, attempts = ([], [])
    requested = set()
    accepted_types = Counter()

    def query(candidate, kind=None):
        
        if candidate in requested:
            raise RuntimeError("Selector повторно запросил один candidate.")
        requested.add(candidate)
        row = dict(attempt(candidate))
        if row.get("candidate_id") != candidate or not isinstance(row.get("accepted"), bool):
            raise RuntimeError("Неверный результат expert attempt.")
        attempts.append(row)
        if row["accepted"]:
            accepted.append(candidate)
            accepted_types[row["failure_type"]] += 1
        if kind is not None and row["failure_type"] != kind:
            raise RuntimeError("Reference label expert attempt отличается от группы selector.")

    if method in ("U", "Random"):
        for candidate in condition["query_order"]:
            if len(accepted) == budget:
                break
            query(candidate)
    elif method == "F_ref":
        queues = condition["query_orders_by_type"]
        positions = {}
        for kind in queues:
            positions[kind] = 0

        def next_of_type(kind):
            
            candidate = queues[kind][positions[kind]]
            positions[kind] += 1
            query(candidate, kind)

        for kind in FAILURE_TYPES:
            while accepted_types[kind] < condition["quotas"][kind] and positions[kind] < len(queues[kind]):
                next_of_type(kind)
        rng = random.Random(seed)
        while len(accepted) < budget:
            available = []
            for kind in sorted(queues):
                if positions[kind] < len(queues[kind]):
                    available.append(kind)
            if not available:
                break
            minimum = min((accepted_types[kind] for kind in available))
            tied = []
            for kind in available:
                if accepted_types[kind] == minimum:
                    tied.append(kind)
            next_of_type(rng.choice(tied))
    else:
        raise ValueError(f"Неизвестный selector: {method}")
    return {
        "selected_successful_candidates": accepted,
        "attempts": attempts,
        "target_budget": budget,
        "accepted_count": len(accepted),
        "budget_met": len(accepted) == budget,
        "failure_counts": dict(accepted_types),
        "n_attempts": len(attempts),
        "expert_action_timesteps_including_failures": sum((row["expert_steps"] for row in attempts)),
        "expert_execution_seconds_logical": sum((row["expert_execution_wall_seconds"] for row in attempts)),
        "cached_attempts": sum((bool(row.get("cache_hit")) for row in attempts)),
    }
