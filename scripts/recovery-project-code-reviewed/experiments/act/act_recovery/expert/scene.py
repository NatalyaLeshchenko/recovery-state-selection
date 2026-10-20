"""Координаты рабочего конца, миски и тарелки для контроллера эксперта."""

from __future__ import annotations

import numpy as np


class GoalReached(Exception):

    pass


def find_robot_env(vec):
    
    queue = [vec.envs[0]]
    visited = set()
    while queue:
        obj = queue.pop(0)
        if id(obj) in visited:
            continue
        visited.add(id(obj))
        if getattr(obj, "robots", None):
            return obj
        for attr in ("env", "_env", "unwrapped"):
            child = getattr(obj, attr, None)
            if child is not None:
                queue.append(child)
    raise RuntimeError("Не найден robot environment.")


def geometry(raw):
    
    sim = raw.sim
    names = list(sim.model.body_names)
    bowl_id = names.index("akita_black_bowl_1_main")
    plate_id = names.index("plate_1_main") #Найти номер объекта с заранее известным именем
    site_id = sim.model.site_name2id(raw.robots[0].controller_config["eef_name"])
    return {
        "simulator_time": float(sim.data.time),
        "eef_pos": sim.data.site_xpos[site_id].copy().tolist(), #Прочитать текущие координаты [x, y, z]
        "bowl_pos": sim.data.body_xpos[bowl_id].copy().tolist(),
        "plate_pos": sim.data.body_xpos[plate_id].copy().tolist(),
    }


def flatten_observation(value, path="observation"):
    
    result = {}
    if isinstance(value, dict):
        for key, child in value.items():
            result.update(flatten_observation(child, f"{path}/{key}"))
    else:
        array = np.asarray(value)
        if array.dtype.kind not in "biuf" or array.ndim == 0 or array.shape[0] != 1:
            raise RuntimeError(f"Неожиданное наблюдение {path}: {array.shape}, {array.dtype}")
        result[path] = array[0].copy()
    return result
