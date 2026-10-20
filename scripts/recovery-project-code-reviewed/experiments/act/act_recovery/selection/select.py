"""Сохранение плана отбора с одинаковым бюджетом демонстраций."""

from __future__ import annotations

from act_recovery.files import (
    file_hash,
    local_path,
    new_folder,
    read_json,
    resolve_pointer,
    write_json,
    write_pointer,
)
from act_recovery.pool.records import load_pool
from act_recovery.selection.methods import query_plan
from act_recovery.settings import CODE_ROOT, SCORE_POINTER


def select(args, root):
    folder = local_path(root, args.scores) if args.scores else resolve_pointer(root, SCORE_POINTER) #score results
    record = read_json(folder / "scores.json")

    if record.get("status") != "complete":
        raise RuntimeError("Scoring не завершён.")
    
    pool, protocol = load_pool(root, record["pool"])

    expected = {}

    plan = query_plan(record["cases"], args.budget, args.seed) #  {"U": "selected_candidates": ["A", "B"]}, {F_ref: ..}

    output = new_folder(root, "selection")
    result = read_json((CODE_ROOT / "selection_pilot.py").with_name("selection_defaults.json"))[
        "selection_notes"
    ]

    result.update(
        pool=str(pool),
        scores=str(folder),
        seed=args.seed,
        target_successful_demonstrations=args.budget,
        conditions=plan,
    )
    write_json(output / "selection.json", result)
    write_pointer(root, "selection_plan_path.txt", output)
    print(f"\nSELECTION READY: target B={args.budget}, seed={args.seed}")
    for method, condition in plan.items():
        print(f"{method}: {condition['selected_candidates']}; types={condition['failure_counts']}")
    print("Plan:", output / "selection.json") 

    """U: ['A', 'B']; types={'missed_grasp': 2}
    F_ref: ['B', 'C']; types={'missed_grasp': 1, 'object_dropped': 1}
    Random: ['D', 'B']; types={'object_dropped': 1, 'missed_grasp': 1} """
