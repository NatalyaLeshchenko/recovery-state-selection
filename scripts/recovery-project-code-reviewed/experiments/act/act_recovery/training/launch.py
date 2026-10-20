"""Три дообучения исходной ACT по одному рецепту."""

from __future__ import annotations

from act_recovery.expert.records import load_collection
from act_recovery.files import (
    StageNotReady,
    file_hash,
    local_path,
    new_folder,
    read_json,
    resolve_pointer,
    write_json,
    write_pointer,
)
from act_recovery.processes import run_child
from act_recovery.settings import (
    DATASETS_POINTER,
    METHODS,
    MODELS_POINTER,
    RECOVERY_SUPERVISION,
    WEIGHTS,
)


def train(args, root):

    datasets_folder = (
        local_path(root, args.datasets) if args.datasets else resolve_pointer(root, DATASETS_POINTER)
    ) #ищем готовые данные
    datasets = read_json(datasets_folder / "datasets.json")
    
    collection, info = load_collection(root, datasets["collection"]) #info["target_budget"] = 2, info["source_initial_model"] - ACT
    
    recipe = {
        "steps": args.steps,
        "seed": args.seed,
        "lr": 1e-05,
        "n_action_steps": 1,
        "recovery_supervision_steps": RECOVERY_SUPERVISION, #10 действий эксперта
        "weights": list(WEIGHTS),
        "checkpoint_selection": "last fixed update; no test-based checkpoint selection",
        "normalization": "fixed initial-policy processors",
        "training_source": "fixed initial ACT, independently reloaded for every selector",
    }
    output = new_folder(root, "models_selection") #results
    summary = {
        "status": "running",
        "recipe": recipe,
        "target_budget": info["target_budget"],
        "datasets": str(datasets_folder),
        "datasets_sha256": file_hash(datasets_folder / "datasets.json"),
        "collection": str(collection),
        "initial_policy": info["source_initial_model"],
        "models": {},
        "scope": "Single-task ACT disagreement/reference-label pilot; not INSIGHT or VLA validation",
    }
    write_json(output / "models.json", summary)
    try:
        for method in METHODS: #METHODS = ("U", "F_ref", "Random")
            print(
                f"\n=== TRAIN {method}: B={info['target_budget']}, updates={args.steps}, seed={args.seed} ===",
                flush=True,
            )
            worker_output = output / method
            request = { #пар-ры одного запуска
                "project_root": str(root),
                "output": str(worker_output),
                "method": method,
                "dataset": datasets["datasets"][method],
                "initial_policy": info["source_initial_model"],
                "budget": info["target_budget"],
                "recipe": recipe,
            }
            result = run_child(root, "_train-worker", request, worker_output, live=True) #выполни одно дообучение
            summary["models"][method] = result["model"]
            write_json(output / "models.json", summary)

            
        summary["status"] = "complete"
        write_json(output / "models.json", summary)
        write_pointer(root, MODELS_POINTER, output)
    except BaseException as error:
        summary.update(status="interrupted", error=f"{type(error).__name__}: {error}")
        write_json(output / "models.json", summary)
        raise
    print("\nTHREE SELECTOR POLICIES READY")
    for method, model in summary["models"].items():
        print(f"{method}: {model['identity']['path']}")
    print("Models:", output / "models.json")
