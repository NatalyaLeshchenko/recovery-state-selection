"""Проверка четырёх ACT на одинаковых состояниях после ошибок."""

from __future__ import annotations

from datetime import datetime

from act_recovery.evaluation.metrics import recovery_summary
from act_recovery.evaluation.models import require
from act_recovery.evaluation.records import load_test, run_folder
from act_recovery.files import file_hash, write_json, write_pointer
from act_recovery.models import verify_model
from act_recovery.processes import run_recovery_worker
from act_recovery.settings import RECOVERY_POINTER, VARIANTS


def recovery(args, root):
    """Вход: args — аргументы команды терминала; root — папка с pilot_runs.
    Выход: результаты и таблица успеха на recovery-стартах."""
    pool, info = load_test(root, args.pool)
    output, results = run_folder(root, "recovery", args, pool, info)
    protocol, identities = (info["protocol"], info["provenance"]["models"])
    for index, case in enumerate(info["cases"], 1):
        candidate = case["candidate_id"]
        row = results.setdefault(candidate, {})
        for variant in VARIANTS:
            if variant in row:
                continue
            print(f"\nRECOVERY {index}/{len(info['cases'])}, {candidate}, {variant}", flush=True)
            verify_model(identities[variant])
            require(
                file_hash(pool / candidate / "start.npz") == case["snapshot_sha256"], "Snapshot изменился."
            )
            attempt_folder = (
                output / candidate / variant / datetime.now().strftime("attempt_%Y%m%d_%H%M%S_%f")
            )
            request = {
                "stage": "evaluate",
                "variant": variant,
                "protocol": protocol,
                "project_root": str(root),
                "output": str(attempt_folder),
                "case_dir": str(pool / candidate),
                "init_state_id": case["init_state_id"],
                "init_definition_sha256": case["init_definition_sha256"],
                "model_path": identities[variant]["path"],
            }
            result = run_recovery_worker(root, request)
            require(
                result["status"] == "evaluated"
                and result["variant"] == variant
                and (result["model"] == identities[variant]["path"])
                and (result["init_state_id"] == case["init_state_id"])
                and (1 <= result["steps"] <= 280)
                and (result["restore"]["recovery_horizon_steps"] == 280),
                "Evaluator вернул результат другого model/state/protocol.",
            )
            result.update(
                candidate_id=candidate,
                failure_type=case["failure_type"],
                source_snapshot_sha256=case["snapshot_sha256"],
                result_folder=str(attempt_folder),
            )
            row[variant] = result
            write_json(output / "partial_results.json", results)
            print(
                f"{variant}: success={result['success']}; steps={result['steps']}; restore error={result['restore']['max_state_error']:.3g}",
                flush=True,
            )
    load_test(root, pool)
    summary = recovery_summary(info["cases"], results, info["provenance"]["demonstration_budget"])
    write_json(output / "summary.json", summary)
    write_pointer(root, RECOVERY_POINTER, output)
    print("\n=== INDEPENDENT SELECTOR COMPARISON ===")
    for variant, row in summary["models"].items():
        print(f"{variant}: {row['n_success']}/{row['n_episodes']} = {row['pc_success']:.1f}%")
        type_counts = {}
        for k, g in row["per_failure_type"].items():
            type_counts[k] = f"{g['n_success']}/{g['n_episodes']}"
        print("  By failure type:", type_counts)
    print(f"F_ref − U: {summary['F_ref_minus_U']['difference_percentage_points']:+.1f} percentage points")
    print("Output:", output)
