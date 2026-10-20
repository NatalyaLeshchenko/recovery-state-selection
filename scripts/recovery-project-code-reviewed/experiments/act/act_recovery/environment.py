"""Создание одной среды LIBERO и проверка цели задачи."""

from __future__ import annotations

import numpy as np


def create_environment(episode_length):
    """Вход: лимит шагов. Выход: одна среда LIBERO в том же интерфейсе LeRobot."""
    from lerobot.envs.configs import LiberoEnv
    from lerobot.envs.factory import make_env

    config = LiberoEnv(
        task="libero_spatial",
        task_ids=[0],
        control_mode="relative",
        init_states=True,
        hard_reset=True,
        max_parallel_tasks=1,
        observation_height=256,
        observation_width=256,
        episode_length=episode_length,
    )
    environments = make_env(config, n_envs=1, use_async_envs=False)
    return environments["libero_spatial"][0]


def make_vector(protocol, prefix_steps):
    """Вход: протокол и длина префикса. Выход: среда для последующих скриптов проекта.

    Сохраняет прежнее имя, которое импортирует selection_experiment.py.
    """
    return create_environment(prefix_steps + protocol["recovery_horizon_steps"])


def goal_at_start(vec):
    """Вход: среда. Выход: выполнена ли уже цель LIBERO."""
    from act_recovery.scene import wrapper_nodes

    for obj in wrapper_nodes(vec):
        for name in ("check_success", "_check_success"):
            method = getattr(obj, name, None)
            if callable(method):
                result = np.asarray(method())
                if result.size != 1:
                    raise RuntimeError("Ожидался один LIBERO success flag.")
                return bool(result.item())
    raise RuntimeError("Не найдена проверка LIBERO goal predicate.")
