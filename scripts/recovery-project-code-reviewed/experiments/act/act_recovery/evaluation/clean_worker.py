"""сохранили ли умение таскать миску"""

from __future__ import annotations

import os
import runpy
import sys
from pathlib import Path

from act_recovery.environment import goal_at_start
from act_recovery.evaluation.models import require
from act_recovery.files import read_json, write_json
from act_recovery.models import array_hash, verify_model
from act_recovery.observations import raw_observation
from act_recovery.scene import choose_initial_state, state_and_ctrl


def clean_worker(request):
    from gymnasium.vector import SyncVectorEnv

    root, output = (Path(request["project_root"]), Path(request["output"]))
    os.chdir(root)

    identities, variant = (request["models"], request["variant"])#проверяет модель
    verify_model(identities[variant])
    init_ids, excluded = (request["init_ids"], request["excluded_init_ids"])
    original_reset, old_argv = (SyncVectorEnv.reset, sys.argv)
    resets, vectors = ([], [])

    def reset(vec, *args, **kwargs):
        
        index = len(resets)
        require(index < len(init_ids), "Clean evaluator запросил лишний reset.")
        init_id = init_ids[index]
        definition = choose_initial_state(vec, init_id, excluded)
        result = original_reset(vec, *args, **kwargs)
        vectors.append(vec)
        state, ctrl = state_and_ctrl(vec)
        obs = raw_observation(result[0])
        require(not goal_at_start(vec), "Обычное начальное состояние уже удовлетворяет goal.")
        observation_hashes = {}
        for key, value in obs.items():
            observation_hashes[key] = array_hash(value)
        resets.append(
            {
                "init_state_id": init_id,
                "init_definition_sha256": definition,
                "state_sha256": array_hash(state),
                "ctrl_sha256": array_hash(ctrl),
                "observations_sha256": observation_hashes,
            }
        )
        write_json(output / "clean_starts.json", resets)
        return result

    sys.argv = [
        "lerobot-eval",
        f"--policy.path={identities[variant]['path']}",
        "--policy.device=cuda",
        "--policy.n_action_steps=1",
        "--env.type=libero",
        "--env.task=libero_spatial",
        "--env.task_ids=[0]",
        "--env.control_mode=relative",
        "--env.init_states=true",
        "--env.hard_reset=true",
        "--env.max_parallel_tasks=1",
        "--env.observation_height=256",
        "--env.observation_width=256",
        "--env.episode_length=280",
        "--eval.batch_size=1",
        f"--eval.n_episodes={len(init_ids)}",
        "--eval.use_async_envs=false",
        "--seed=0",
        f"--output_dir={output / 'evaluation'}",
    ]
    try:
        SyncVectorEnv.reset = reset
        runpy.run_module("lerobot.scripts.lerobot_eval", run_name="__main__")
    finally:
        SyncVectorEnv.reset, sys.argv = (original_reset, old_argv)
        unique_environments = {}
        for vec in vectors:
            unique_environments[id(vec)] = vec
        for vec in unique_environments.values():
            if not getattr(vec, "closed", False):
                vec.close()
    info = read_json(output / "evaluation/eval_info.json")
    require(len(info["per_task"]) == 1, "Clean evaluation должна содержать одну task.")
    successes = info["per_task"][0]["metrics"]["successes"]
    require(
        len(successes) == len(resets) == len(init_ids)
        and all((isinstance(x, bool) for x in successes))
        and (sum(successes) == info["overall"]["n_success"]),
        "Неполная clean evaluation.",
    )
    verify_model(identities[variant])
    return {
        "status": "evaluated",
        "variant": variant,
        "successes": successes,
        "starts": resets,
        "n_success": sum(successes),
        "n_episodes": len(init_ids),
        "pc_success": 100 * sum(successes) / len(init_ids),
        "videos": info["overall"].get("video_paths", []),
        "result_folder": str(output),
    }
