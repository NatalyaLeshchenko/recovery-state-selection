"""Проверка обычной задачи без вмешательств."""

from __future__ import annotations

from datetime import datetime

from act_recovery.evaluation.metrics import paired_counts
from act_recovery.evaluation.models import require
from act_recovery.evaluation.records import load_test, run_folder
from act_recovery.files import write_json, write_pointer
from act_recovery.processes import run_clean_child
from act_recovery.settings import CLEAN_POINTER, METHODS, VARIANTS


def clean(args, root):
    
    pool, info = load_test(root, args.pool)
    output, results = run_folder(root, "clean", args, pool, info)
    for variant in VARIANTS:
        if variant in results:
            continue
        print(
            f"\nCLEAN {variant}: {len(info['clean_init_state_ids'])} nominal initial configurations",
            flush=True,
        )
        folder = output / variant / datetime.now().strftime("attempt_%Y%m%d_%H%M%S_%f")
        request = {
            "project_root": str(root),
            "output": str(folder),
            "variant": variant,
            "models": info["provenance"]["models"],
            "init_ids": info["clean_init_state_ids"],
            "excluded_init_ids": info["provenance"]["training_init_state_ids"],
        }
        result = run_clean_child(root, request)
        if variant != "initial":
            require(
                result["starts"] == results["initial"]["starts"],
                "Clean policies получили разные исходные состояния.",
            )
        results[variant] = result
        write_json(output / "partial_results.json", results)
        print(f"{variant}: {result['n_success']}/{result['n_episodes']}", flush=True)
    load_test(root, pool)
    differences = {}
    for variant in METHODS:
        differences[variant] = results[variant]["pc_success"] - results["initial"]["pc_success"]
    paired_results = {}
    for variant in METHODS:
        paired_results[variant] = paired_counts(
            results[variant]["successes"], results["initial"]["successes"]
        )
    summary = {
        "n_states": len(info["clean_init_state_ids"]),
        "models": results,
        "difference_from_initial_percentage_points": differences,
        "paired_against_initial": paired_results,
        "scope": "Clean-task retention on the same ordinary initial states, without induced failures or expert assistance",
        "note": "Single-task, single-training-seed pilot; exact point estimates, no significance claim.",
    }
    write_json(output / "summary.json", summary)
    write_pointer(root, CLEAN_POINTER, output)
    print("\n=== CLEAN-TASK RETENTION ===")
    for variant in VARIANTS:
        row = results[variant]
        change = (
            ""
            if variant == "initial"
            else f"; относительно initial: {summary['difference_from_initial_percentage_points'][variant]:+.1f} п.п."
        )
        print(f"{variant}: {row['n_success']}/{row['n_episodes']} = {row['pc_success']:.1f}%{change}")
    print("Output:", output)
