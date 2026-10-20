"""Общий сбор состояний после промаха и падения объекта."""

from __future__ import annotations

from collections import Counter

from act_recovery.files import (
    StageNotReady,
    file_hash,
    new_folder,
    read_json,
    write_json,
    write_pointer,
)
from act_recovery.pool.records import load_ensemble
from act_recovery.pool.worker import run_worker
from act_recovery.settings import (
    CODE_ROOT,
    FAILURE_TYPES,
    INSTRUCTION,
    POOL_POINTER,
    RESERVED_TEST_IDS,
)


def collect(args, root):
    
    from act_recovery.models import runtime_versions


    ensemble_folder, ensemble = load_ensemble(root, args.ensemble)

    output = new_folder(root, "pool")
    protocol = read_json((CODE_ROOT / "selection_pilot.py").with_name("selection_defaults.json"))["protocol"]
    protocol.update(
        instruction=INSTRUCTION,
        excluded_init_state_ids=list(RESERVED_TEST_IDS),
        init_state_ids=args.init_ids,
        reserved_independent_recovery_test_ids=list(RESERVED_TEST_IDS),
        failure_types=list(FAILURE_TYPES),
        ensemble_folder=str(ensemble_folder),
        ensemble_sha256=file_hash(ensemble_folder / "ensemble.json"),
        models={"initial": ensemble["models"][0]["identity"]},
        runtime_versions=runtime_versions(),
        manifest_sha256=file_hash(root / "pilot_runs/manifest.json"),
    )
    write_json(output / "protocol.json", protocol)
    print("Pool:", output, flush=True)

    for init_id in args.init_ids:
        for kind in FAILURE_TYPES:
            candidate_id = f"init_{init_id:03d}_{kind}"
            print(f"\nCOLLECT {candidate_id}: initial ACT; controlled intervention", flush=True)
            request = {
                "stage": "collect",
                "protocol": protocol,
                "project_root": str(root),
                "output": str(output / candidate_id),
                "candidate_id": candidate_id,
                "failure_type": kind,
                "init_state_id": init_id,
                "model_path": protocol["models"]["initial"]["path"],
            }
            try:
                result = run_worker(root, request)
            except Exception:
                protocol["status"] = "error"
                write_json(output / "protocol.json", protocol)
                raise
            protocol["attempts"].append(result)
            if result["status"] == "accepted":
                protocol["cases"].append(result)
                print("Accepted:", kind, result["verification"], flush=True)
            else:
                print("Skipped:", result["reason"], flush=True)
            write_json(output / "protocol.json", protocol)
    counts = Counter((case["failure_type"] for case in protocol["cases"]))
    protocol["complete"] = all((counts[k] > 0 for k in FAILURE_TYPES))
    protocol["status"] = "complete" if protocol["complete"] else "insufficient_failure_types"
    protocol["counts"] = dict(counts)
    write_json(output / "protocol.json", protocol)

    
    if protocol["complete"]:
        write_pointer(root, POOL_POINTER, output)
    print("\nPOOL READY" if protocol["complete"] else "POOL INCOMPLETE: missing failure type")
    print("Failure types:", dict(counts))
    print("Output:", output)


    if not protocol["complete"]:
        raise StageNotReady(
            "Неполный пул сохранён для диагностики. score и select требуют обоих failure types."
        )
