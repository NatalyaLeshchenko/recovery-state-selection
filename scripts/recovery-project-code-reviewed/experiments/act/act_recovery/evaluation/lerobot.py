"""Вызов штатного оценщика LeRobot с нужной средой и политикой."""

from __future__ import annotations

import runpy
import sys
from pathlib import Path


def official_evaluation(request, hooks):

    from gymnasium.vector import SyncVectorEnv

    protocol = request["protocol"]
    if request["stage"] == "collect":
        horizon = protocol["nominal_horizon_steps"] + protocol["open_steps"]
    else:
        horizon = len(hooks.saved["past_actions"]) + protocol["recovery_horizon_steps"]
    old_reset, old_step, old_argv = (SyncVectorEnv.reset, SyncVectorEnv.step, sys.argv)
    sys.argv = [
        "lerobot-eval",
        f"--policy.path={request['model_path']}",
        "--policy.device=cuda",
        f"--policy.n_action_steps={protocol['n_action_steps']}",
        "--env.type=libero",
        "--env.task=libero_spatial",
        "--env.task_ids=[0]",
        "--env.control_mode=relative",
        "--env.init_states=true",
        "--env.hard_reset=true",
        "--env.max_parallel_tasks=1",
        "--env.observation_height=256",
        "--env.observation_width=256",
        f"--env.episode_length={horizon}",
        "--eval.batch_size=1",
        "--eval.n_episodes=1",
        "--eval.use_async_envs=false",
        f"--seed={protocol['seed']}",
        f"--output_dir={Path(request['output']) / 'evaluation'}",
    ]
    try:

        def reset_environment(vec, *args, **kwargs):
            """Вход: среда и параметры reset. Выход: наблюдение после повтора сохранённых действий."""
            return hooks.reset(vec, *args, **kwargs)

        SyncVectorEnv.reset = reset_environment

        def step_environment(vec, action):
            """Вход: среда и команда ACT. Выход: следующий шаг среды с записью в журнал."""
            return hooks.step(vec, action)

        SyncVectorEnv.step = step_environment
        runpy.run_module("lerobot.scripts.lerobot_eval", run_name="__main__")
    finally:
        SyncVectorEnv.reset, SyncVectorEnv.step, sys.argv = (old_reset, old_step, old_argv)
        unique_environments = {}
        for vec in hooks.vectors:
            unique_environments[id(vec)] = vec
        for vec in unique_environments.values():
            if not getattr(vec, "closed", False):
                vec.close()
