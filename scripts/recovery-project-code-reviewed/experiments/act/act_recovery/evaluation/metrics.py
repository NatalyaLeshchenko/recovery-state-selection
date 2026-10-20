"""Число успехов, доли успеха и парное сравнение на одинаковых состояниях."""

from __future__ import annotations

from act_recovery.evaluation.models import require
from act_recovery.settings import FAILURE_TYPES, METHODS, VARIANTS


def paired_counts(first, second):
    """Вход: first — first; second — second.
    Выход: четыре парных исхода и разница успеха в процентных пунктах."""
    require(len(first) == len(second) > 0, "Для разницы нужны парные результаты на одинаковых states.")
    return {
        "both_success": sum((a and b for a, b in zip(first, second))),
        "both_fail": sum((not a and (not b) for a, b in zip(first, second))),
        "first_only_success": sum((a and (not b) for a, b in zip(first, second))),
        "second_only_success": sum((not a and b for a, b in zip(first, second))),
        "difference_percentage_points": 100
        * sum((int(a) - int(b) for a, b in zip(first, second)))
        / len(first),
    }


def recovery_summary(cases, results, budget):
    """Вход: cases — cases; results — results; budget — число требуемых демонстраций.
    Выход: итоговые показатели восстановления четырёх политик."""
    require(bool(cases), "Нет test cases.")
    ids = []
    for case in cases:
        ids.append(case["candidate_id"])
    require(len(set(ids)) == len(ids) and set(results) == set(ids), "Результаты покрывают другой test set.")
    init_state_ids = set()
    for c in cases:
        init_state_ids.add(c["init_state_id"])
    summary = {
        "n_states": len(ids),
        "n_initial_configurations": len(init_state_ids),
        "demonstration_budget": budget,
        "models": {},
        "F_ref_minus_U": {},
        "scope": "Fixed independent single-task test; ACT ensemble proxy, reference failure labels",
        "uncertainty_note": "Small single-training-seed pilot. Two failure starts from the same configuration are related; no reliable population CI or significance claim is made.",
    }
    series = {}
    for variant in VARIANTS:
        require(
            all((set(results[cid]) == set(VARIANTS) for cid in ids)),
            "Не все четыре политики оценены на всех states.",
        )
        successes_for_variant = []
        for cid in ids:
            successes_for_variant.append(results[cid][variant]["success"])
        series[variant] = successes_for_variant
        require(
            all((isinstance(x, bool) for x in series[variant])),
            "Success должен быть boolean benchmark outcome.",
        )
        count = sum(series[variant])
        groups = {}
        for kind in FAILURE_TYPES:
            successes = []
            for c in cases:
                if c["failure_type"] == kind:
                    successes.append(results[c["candidate_id"]][variant]["success"])
            require(bool(successes), "Нет test starts одной категории.")
            groups[kind] = {
                "n_success": sum(successes),
                "n_episodes": len(successes),
                "pc_success": 100 * sum(successes) / len(successes),
            }
        summary["models"][variant] = {
            "n_success": count,
            "n_episodes": len(ids),
            "pc_success": 100 * count / len(ids),
            "per_failure_type": groups,
            "macro_pc_success": sum((g["pc_success"] for g in groups.values())) / len(groups),
        }
    summary["F_ref_minus_U"] = paired_counts(series["F_ref"], series["U"])
    summary["F_ref_minus_U"].update(first="F_ref", second="U")
    against_initial = {}
    for variant in METHODS:
        against_initial[variant] = paired_counts(series[variant], series["initial"])
    summary["against_initial"] = against_initial
    return summary
