"""Общие смещения захвата и размещения из одного обычного выполнения."""

from __future__ import annotations

import numpy as np

from act_recovery.expert.scene import find_robot_env, geometry


def make_vec(cfg):

    from lerobot.envs.factory import make_env

    return make_env(cfg, n_envs=1, use_async_envs=False)["libero_spatial"][0]


def calibrate(cfg, nominal_actions): #
    
    vec = make_vec(cfg)
    try:
        vec.reset(seed=0)
        raw = find_robot_env(vec)
        grasp_offset = None
        nominal_success = False

        for index, action in enumerate(nominal_actions, start=1):
            _, reward, terminated, truncated, _ = vec.step(action)
            current = geometry(raw) #узнаем, где теперь находятся захват, миска и тарелка

            if index == 42:
                grasp_offset = np.asarray(current["eef_pos"]) - current["bowl_pos"] 

                """eef_pos  = [0.20, 0.10, 0.95]
                    bowl_pos = [0.20, 0.10, 0.90]

                    grasp_offset = [0, 0, 0.05]"""

                #проверяем саксесс
            nominal_success = nominal_success or bool(np.any(np.asarray(reward) > 0))
            if np.any(terminated) or np.any(truncated):
                break
        if grasp_offset is None or not nominal_success:
            raise RuntimeError("Калибровочный rollout не достиг цели либо не дошёл до шага 42.")

        #положение миски относительно тарелки
        placement_offset = np.asarray(current["bowl_pos"]) - current["plate_pos"]
        return (grasp_offset, placement_offset) # положение захвата относительно миски, положение миски относительно тарелки
    finally:
        vec.close()
