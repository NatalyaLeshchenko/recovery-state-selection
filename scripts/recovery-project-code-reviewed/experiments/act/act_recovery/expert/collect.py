"""Сбор выбранных демонстраций с общей калибровкой и учётом затрат."""

from __future__ import annotations

import numpy as np

from act_recovery.expert.budget import resolve_condition
from act_recovery.expert.plan import load_plan
from act_recovery.files import (
    StageNotReady,
    file_hash,
    new_folder,
    write_json,
    write_pointer,
)
from act_recovery.processes import run_child
from act_recovery.settings import COLLECTION_POINTER, METHODS


def expert(args, root):
    
    from lerobot.envs.configs import LiberoEnv

    from act_recovery.expert.calibration import calibrate
    from act_recovery.models import verify_identity

    plan_path, plan, pool, protocol = load_plan(root, args.plan)
    output = new_folder(root, "expert_selection")
    nominal_path = root / "pilot_runs/lift_trace/actions.npz"
    with np.load(nominal_path, allow_pickle=False) as saved:
        nominal = saved["actions"].copy()
    cfg = LiberoEnv(
        task="libero_spatial",
        task_ids=[0],
        control_mode="relative",
        init_states=True,
        hard_reset=True,
        observation_height=256,
        observation_width=256,
        episode_length=len(nominal) + 20,
    )
    grasp_offset, placement_offset = calibrate(cfg, nominal)
    print("Expert grasp offset =", grasp_offset, flush=True)
    info = {
        "status": "running",
        "complete": False,
        "pool": str(pool),
        "instruction": protocol["instruction"],
        "pool_protocol_sha256": file_hash(pool / "protocol.json"),
        "selection_plan": str(plan_path),
        "selection_plan_sha256": file_hash(plan_path),
        "source_initial_model": protocol["models"]["initial"],
        "manifest_sha256": protocol["manifest_sha256"],
        "target_budget": plan["target_successful_demonstrations"],
        "selection_seed": plan["seed"],
        "expert": "fixed_privileged_scripted_controller",
        "max_expert_steps": args.max_expert_steps,
        "nominal_calibration_actions": str(nominal_path),
        "nominal_actions_sha256": file_hash(nominal_path),
        "grasp_offset": grasp_offset.tolist(),
        "placement_offset": placement_offset.tolist(),
        "conditions": {},
        "unique_attempts": {},
        "cost_note": "Shared candidates are physically collected once. Per-condition logical acquisition counts include cached attempts; physical total is reported separately. These are controller times, not human teleoperation times.",
        "scope": "ACT selected-data learning pilot; independent evaluation is a separate stage",
    }
    write_json(output / "collection.json", info)
    cases = {}
    for case in protocol["cases"]:
        cases[case["candidate_id"]] = case

    def attempt(candidate):
        """Вход: номер кандидата. Выход: результат эксперта из новой попытки или общей сохранённой записи."""
        if candidate not in cases:
            raise RuntimeError("Selector запросил candidate вне common pool.")
        if candidate in info["unique_attempts"]:
            return {**info["unique_attempts"][candidate], "cache_hit": True}
        case = cases[candidate]
        verify_identity(info["source_initial_model"])
        folder = output / candidate
        print(f"\nEXPERT {candidate}: restore → corrective actions → benchmark goal", flush=True)
        request = {
            "project_root": str(root),
            "output": str(folder),
            "case_folder": str(pool / candidate),
            "case": case,
            "protocol": protocol,
            "max_expert_steps": args.max_expert_steps,
            "grasp_offset": grasp_offset.tolist(),
            "placement_offset": placement_offset.tolist(),
        }
        result = run_child(root, "_expert-worker", request, folder)["attempt"]
        info["unique_attempts"][candidate] = result
        write_json(output / "collection.json", info)
        print(f"accepted={result['accepted']}; expert_steps={result['expert_steps']}", flush=True)
        return {**result, "cache_hit": False}

    try:
        for method in METHODS:
            print(f"\n=== {method}: target B={info['target_budget']} ===", flush=True)
            info["conditions"][method] = resolve_condition(
                method, plan["conditions"][method], info["target_budget"], plan["seed"], attempt
            )
            write_json(output / "collection.json", info)
        verify_identity(info["source_initial_model"])
        if file_hash(pool / "protocol.json") != info["pool_protocol_sha256"]:
            raise RuntimeError("Common pool изменился во время expert collection.")
        info["complete"] = all((row["budget_met"] for row in info["conditions"].values()))
        info["status"] = "complete" if info["complete"] else "unmet_budget"
        info["physical_unique_attempts"] = len(info["unique_attempts"])
        info["physical_expert_action_timesteps"] = sum(
            (row["expert_steps"] for row in info["unique_attempts"].values())
        )
        info["physical_expert_execution_seconds"] = sum(
            (row["expert_execution_wall_seconds"] for row in info["unique_attempts"].values())
        )
        write_json(output / "collection.json", info)
    except BaseException as error:
        info.update(status="interrupted", error=f"{type(error).__name__}: {error}")
        write_json(output / "collection.json", info)
        raise
    for method, row in info["conditions"].items():
        print(
            f"{method}: {row['accepted_count']}/{info['target_budget']} accepted; types={row['failure_counts']}"
        )
    print("Output:", output)
    if not info["complete"]:
        raise StageNotReady("Не все selectors получили B успешных demonstrations; дообучение остановлено.")
    write_pointer(root, COLLECTION_POINTER, output)
    print("\nEXPERT DATA READY")
