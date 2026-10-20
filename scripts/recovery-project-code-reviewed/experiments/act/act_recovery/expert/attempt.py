"""Восстановление кандидата и одна попытка программного эксперта."""

from __future__ import annotations

import shutil
import time
from pathlib import Path

import numpy as np

from act_recovery.environment import goal_at_start, make_vector
from act_recovery.files import file_hash


def expert_one(request):

    from act_recovery.evaluation.replay import EvaluationHooks
    from act_recovery.expert.controller import run_expert
    from act_recovery.expert.recording import save_attempt
    from act_recovery.expert.scene import GoalReached, find_robot_env, geometry

    output = Path(request["output"])
    case = request["case"]
    folder = Path(request["case_folder"])
    if file_hash(folder / "start.npz") != case["snapshot_sha256"]:
        raise RuntimeError("Candidate snapshot изменился перед expert collection.")
    with np.load(folder / "start.npz", allow_pickle=False) as saved:
        prefix_steps = len(saved["past_actions"])
    protocol = dict(request["protocol"], recovery_horizon_steps=request["max_expert_steps"])
    replay = {
        "stage": "evaluate",
        "variant": "scripted_expert",
        "protocol": protocol,
        "case_dir": str(folder),
        "output": str(output),
        "init_state_id": case["init_state_id"],
        "init_definition_sha256": case["init_definition_sha256"],
    }
    records = {
        "observations": {},
        "actions": [],
        "rewards": [],
        "phases": [],
        "geometry": [],
        "success": False,
        "ended": False,
    }
    metadata = {
        "candidate_id": case["candidate_id"],
        "suite": protocol["suite"],
        "task_id": protocol["task_id"],
        "instruction": protocol["instruction"],
        "seed": protocol["seed"],
        "init_state_id": case["init_state_id"],
        "init_definition_sha256": case["init_definition_sha256"],
        "failure_type": case["failure_type"],
        "label_source": case["label_source"],
        "source_snapshot_sha256": case["snapshot_sha256"],
        "source_case": str(folder),
        "expert": "scripted_controller_with_privileged_simulator_geometry",
        "control_frequency_hz": 20,
        "observation_timing": "before_action",
        "grasp_offset": request["grasp_offset"],
        "placement_offset": request["placement_offset"],
        "max_expert_steps": request["max_expert_steps"],
        "calibration_note": "Fixed successful nominal init-0 calibration; no per-case or per-selector retuning",
        "scope": "Expert training/development demonstration; no autonomous policy test result",
    }
    vec = make_vector(protocol, prefix_steps)
    try:
        hooks = EvaluationHooks(replay, type(vec).reset, type(vec).step)
        obs, _ = hooks.reset(vec, seed=protocol["seed"])
        raw = find_robot_env(vec)
        if goal_at_start(vec):
            raise RuntimeError("Recovery start уже удовлетворяет goal.")
        metadata["restore_check"] = hooks.restore
        metadata["start_geometry"] = geometry(raw)
        shutil.copy2(folder / "start.npz", output / "start.npz")
        started = time.perf_counter()
        try:
            run_expert(
                vec,
                raw,
                obs,
                np.asarray(request["grasp_offset"]),
                np.asarray(request["placement_offset"]),
                request["max_expert_steps"],
                records,
            )
        except GoalReached:
            pass
        except RuntimeError as error:
            metadata["error"] = str(error)
            print("EXPERT ATTEMPT FAILED:", error, flush=True)
        metadata["expert_execution_wall_seconds"] = time.perf_counter() - started
        metadata["end_geometry"] = geometry(raw)
        final_goal = goal_at_start(vec)
        metadata["goal_predicate_at_end"] = final_goal
        if final_goal != records["success"]:
            raise RuntimeError("Benchmark predicate и recorded success reward расходятся.")
        if records["geometry"]:
            recorded_times = []
            for row in records["geometry"]:
                recorded_times.append(row["simulator_time"])
            times = recorded_times + [metadata["end_geometry"]["simulator_time"]]
            if not np.allclose(np.diff(times), 0.05, atol=1e-08, rtol=0):
                raise RuntimeError("Частота expert control отличается от 20 Hz.")
        save_attempt(output, records, metadata, make_gif=True)
        return {
            "candidate_id": case["candidate_id"],
            "failure_type": case["failure_type"],
            "folder": str(output),
            "accepted": records["success"],
            "init_state_id": case["init_state_id"],
            "init_definition_sha256": case["init_definition_sha256"],
            "source_snapshot_sha256": case["snapshot_sha256"],
            "expert_steps": len(records["actions"]),
            "error": metadata.get("error"),
            "expert_execution_wall_seconds": metadata["expert_execution_wall_seconds"],
            "metadata_sha256": file_hash(output / "metadata.json"),
            "demonstration_sha256": file_hash(output / "demonstration.npz") if records["success"] else None,
        }
    finally:
        vec.close()
