"""Начальные состояния LIBERO и чтение состояния MuJoCo."""

from __future__ import annotations

import numpy as np

from act_recovery.models import array_hash


class SkipCase(Exception):
    pass


def wrapper_nodes(vec):
    
    queue, seen = ([vec.envs[0]], set())
    while queue:
        obj = queue.pop(0)
        if id(obj) in seen:
            continue
        seen.add(id(obj))
        yield obj
        for name in ("env", "_env", "unwrapped"):
            child = getattr(obj, name, None)
            if child is not None:
                queue.append(child)

#выбрать, с какой расстановки начать
def choose_initial_state(vec, init_id, excluded):

    for obj in wrapper_nodes(vec):
        attributes = vars(obj)
        if "init_state_id" in attributes and "_init_states" in attributes:
            states = attributes["_init_states"]
            
            selected = np.asarray(states[init_id], dtype=np.float64)

            
            for other_id in excluded:
                other = np.asarray(states[other_id], dtype=np.float64)
                if selected.shape == other.shape and np.allclose(selected, other, atol=1e-08, rtol=0):
                    raise SkipCase(f"init-state ID {init_id} совпадает с исключённым ID {other_id}.")
                
            obj.init_state_id = init_id #при след ресет - старт 20
            return array_hash(selected) #отпечаток состояния среды
        
    raise RuntimeError(
        "Не найден LeRobot LiberoEnv с init_state_id и _init_states; нужен текущий исходник envs/libero.py."
    )

#vector environment - to start - vec.reset()  
def find_raw(vec):
    
    for obj in wrapper_nodes(vec):
        if getattr(obj, "robots", None) and getattr(obj, "sim", None) is not None: #ищем, где есть сим и робот
            return obj
    raise RuntimeError("Не найден robot environment.")


def geometry(vec):


    """ {
    "simulator_time": 2.5,
    "bowl_pos": [0.10, 0.20, 0.90],
    "plate_pos": [0.30, 0.20, 0.90],
    "eef_pos": [0.10, 0.20, 1.05],
    "bowl_plate_distance_xy": 0.20,} """
    
    raw = find_raw(vec)
    sim = raw.sim

    names = list(sim.model.body_names)
    bowl = sim.data.body_xpos[names.index("akita_black_bowl_1_main")].copy() #миски - тарелки
    plate = sim.data.body_xpos[names.index("plate_1_main")].copy()

    site_id = sim.model.site_name2id(raw.robots[0].controller_config["eef_name"])
    eef = sim.data.site_xpos[site_id].copy() #эндеффектор сюда
    return {
        "simulator_time": float(sim.data.time),
        "bowl_pos": bowl.tolist(),
        "plate_pos": plate.tolist(),
        "eef_pos": eef.tolist(),
        "bowl_plate_distance_xy": float(np.linalg.norm(bowl[:2] - plate[:2])),
    }


def state_and_ctrl(vec):
    
    sim = find_raw(vec).sim
    state = sim.get_state().flatten().copy() #состояние и скорости
    ctrl = sim.data.ctrl.copy() #для приводов
    if not np.isfinite(state).all() or not np.isfinite(ctrl).all():
        raise RuntimeError("Симулятор содержит неконечные значения.")
    return (state, ctrl)


def exact_error(actual, expected, label):
    
    if actual.shape != expected.shape:
        raise RuntimeError(f"Размер {label} изменился: {actual.shape} / {expected.shape}")
    error = float(np.max(np.abs(actual - expected)))
    if error > 1e-08:
        raise RuntimeError(f"Replay не воспроизвёл {label}: max error={error:.12g}")
    return error


def stopped(result):
    return bool(np.any(result[2]) or np.any(result[3]))

#успешно ли
def observed_success(result):
    info = result[4]
    if isinstance(info, dict) and "is_success" in info:
        if np.any(info["is_success"]):
            return True
    return bool(np.any(np.asarray(result[1]) > 0))


def checked_action(action):
    action = np.asarray(action).copy()
    if action.shape != (1, 7) or not np.isfinite(action).all():
        raise RuntimeError(f"Ожидался конечный action (1, 7), получено {action.shape}")
    return action
