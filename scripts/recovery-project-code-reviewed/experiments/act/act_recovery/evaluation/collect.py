"""Сбор отдельных начальных состояний для проверки восстановления."""

from __future__ import annotations

from collections import Counter
from copy import deepcopy

from act_recovery.evaluation.models import check_split, load_models, require
from act_recovery.files import (
    StageNotReady,
    file_hash,
    new_folder,
    write_json,
    write_pointer,
)
from act_recovery.models import runtime_versions, verify_model
from act_recovery.pool.worker import run_worker
from act_recovery.settings import (
    CODE_ROOT,
    EVALUATION_SCOPE,
    FAILURE_TYPES,
    RESERVED_TEST_IDS,
    TEST_FILE,
    TEST_POINTER,
)


def collect(args, root):
    
    provenance, source = load_models(root, args.models)
    check_split(args.init_ids, provenance)
    check_split(list(RESERVED_TEST_IDS), provenance)
    protocol = deepcopy(source)
    for key in ("cases", "attempts", "complete", "status", "counts"):
        protocol.pop(key, None)
    protocol.update(
        init_state_ids=args.init_ids,
        excluded_init_state_ids=provenance["training_init_state_ids"],
        scope=EVALUATION_SCOPE,
        purpose="evaluation_only",
        models={"initial": provenance["models"]["initial"]},
    )
    require(
        protocol["n_action_steps"] == 1 and protocol["recovery_horizon_steps"] == 280,
        "Ожидается зафиксированный pilot protocol: n_action_steps=1, horizon=280.",
    )
    output = new_folder(root, "test_selection")
    info = {
        "status": "running",
        "scope": EVALUATION_SCOPE,
        "purpose": "evaluation_only",
        "protocol": protocol,
        "provenance": provenance,
        "requested_init_state_ids": args.init_ids,
        "clean_init_state_ids": list(RESERVED_TEST_IDS),
        "cases": [],
        "attempts": [],
        "runtime_versions": runtime_versions(),
        "inclusion_rule": "All valid physical events on the predeclared configurations; no outcome filtering",
        "ood_note": "Held out from recovery learning; initial configurations may occur in nominal D0",
        "script_sha256": file_hash(CODE_ROOT / "evaluate_selection.py"),
    }
    write_json(output / TEST_FILE, info)
    print("Test starts:", output, flush=True)
    try:
        for init_id in args.init_ids:
            for kind in FAILURE_TYPES:
                candidate = f"init_{init_id:03d}_{kind}"
                print(f"\nTEST COLLECT {candidate}: initial ACT; fixed intervention protocol", flush=True)
                request = {
                    "stage": "collect",
                    "project_root": str(root),
                    "protocol": protocol,
                    "output": str(output / candidate),
                    "candidate_id": candidate,
                    "failure_type": kind,
                    "init_state_id": init_id,
                    "model_path": provenance["models"]["initial"]["path"],
                }
                result = run_worker(root, request)
                if result["status"] == "accepted":
                    require(
                        result["init_definition_sha256"] not in provenance["training_init_definition_sha256"],
                        "Test configuration физически совпадает с training configuration.",
                    )
                    result.update(
                        scope=EVALUATION_SCOPE, purpose="evaluation_only", independent_recovery_test=True
                    )
                    write_json(output / candidate / "case.json", result)
                    write_json(output / candidate / "worker_result.json", result)
                    info["cases"].append(result)
                    print("Accepted:", kind, result["verification"], flush=True)
                else:
                    print("Skipped:", result["reason"], flush=True)
                info["attempts"].append(result)
                write_json(output / TEST_FILE, info)
        verify_model(provenance["models"]["initial"])
        counts = Counter((case["failure_type"] for case in info["cases"]))
        info["counts"] = dict(counts)
        info["status"] = (
            "test_complete" if all((counts[kind] > 0 for kind in FAILURE_TYPES)) else "incomplete"
        )
        write_json(output / TEST_FILE, info)
    except BaseException as error:
        info.update(status="interrupted", error=f"{type(error).__name__}: {error}")
        write_json(output / TEST_FILE, info)
        raise
    if info["status"] != "test_complete":
        raise StageNotReady(f"Нет обоих failure types. Состояния и причины пропусков сохранены: {output}")
    write_pointer(root, TEST_POINTER, output)
    print(f"\nTEST STATES READY: {len(info['cases'])} starts; types={dict(counts)}")
    print("Expert recovery and uncertainty scoring were not run for the test states.")
