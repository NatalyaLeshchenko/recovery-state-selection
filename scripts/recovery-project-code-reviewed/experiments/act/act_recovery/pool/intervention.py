"""ОТут вмешиваемся"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from act_recovery.files import file_hash
from act_recovery.observations import remember_observation, skipped_result
from act_recovery.pool.contacts import finger_contacts
from act_recovery.pool.events import check_drop_event, check_miss_event
from act_recovery.settings import RAW_SHAPES


def capture_candidate(request, vec, data, step, intended, before, held_ever): 
    
    """request	- какой тип ошибки создаем и куда сохраняем
        vec - 	текущую работающую среду
        data -	накопленные наблюдения и выполненные команды
        intended -	команду, которую ACT собиралась выполнить
        before	- координаты перед вмешательством
        held_ever	- было ли ранее подтверждено удержание миски"""

    from act_recovery.scene import geometry, observed_success, state_and_ctrl, stopped

    protocol = request["protocol"]
    output = Path(request["output"])
    kind = request["failure_type"]
    for item in data["history"]:
        data["frames"].append(item["observation/pixels/image"][0].copy())
    commands = [] #команды вмешательства


    if kind == "missed_grasp":
        blocked = intended.copy()
        blocked[0, -1] = -1.0 # закрытие меняем на открытие 


        commands.append(blocked)
        #движение вверх 24 шага - 1.2 сек
        lift = np.zeros((1, 7), dtype=np.float32)
        lift[0, 2] = protocol["miss_lift_action"]
        lift[0, -1] = -1.0


        for number in range(protocol["miss_lift_steps"]):
            commands.append(lift.copy())
    else:
        opened = np.zeros((1, 7), dtype=np.float32)
        opened[0, -1] = -1.0 #для object_dropped, открыли, пока он держит миску

        for number in range(protocol["open_steps"]):
            commands.append(opened.copy())
    for action in commands:
        result = vec.step(action)
        data["actions"].append(action.copy())
        remember_observation(data, result[0])
        data["frames"].append(data["last_observation"]["observation/pixels/image"][0].copy())
        data["event_rows"].append(
            {
                "geometry": geometry(vec),
                "contacts": finger_contacts(vec),
                "executed_action": action[0].tolist(),
            }
        )


        if stopped(result) or observed_success(result):
            return skipped_result(request, "Задача или эпизод завершились во время вмешательства.")

        
    if kind == "missed_grasp":
        valid, evidence = check_miss_event(data["initial_geometry"], before, data["event_rows"], protocol) #получилась ли нужная физическая ошибка
    else:
        valid, evidence = check_drop_event(before, data["event_rows"], held_ever, protocol)
    if not valid:
        return skipped_result(request, f"Условия reference label {kind} не выполнены: {evidence}")
    state, ctrl = state_and_ctrl(vec)
    saved = data["last_observation"].copy()
    saved.update(
        simulator_state=state,
        ctrl=ctrl,
        reset_state=data["reset_state"],
        reset_ctrl=data["reset_ctrl"],
        past_actions=np.stack(data["actions"]),
    )
    np.savez_compressed(output / "start.npz", **saved)
    history = {}
    for key in RAW_SHAPES:
        values = []
        for item in data["history"]:
            values.append(item[key])
        history[key] = np.concatenate(values, axis=0)
    np.savez_compressed(output / "history.npz", **history)
    return {
        "status": "accepted",
        "candidate_id": request["candidate_id"],
        "failure_type": kind,
        "label_source": "explicit_simulator_event_rules",
        "init_state_id": request["init_state_id"],
        "init_definition_sha256": data["init_hash"],
        "nominal_steps": step,
        "prefix_steps": len(data["actions"]),
        "instruction": protocol["instruction"],
        "initial_geometry": data["initial_geometry"],
        "before_intervention": before,
        "after_intervention": data["event_rows"][-1]["geometry"],
        "verification": evidence,
        "history_observations": len(data["history"]),
        "nominal_help_score_used_for_collection": False,
        "autonomous_recovery_outcome_used_for_collection": False,
        "scope": "Induced collection/development state; not an independent evaluation state",
        "snapshot_sha256": file_hash(output / "start.npz"),
        "history_sha256": file_hash(output / "history.npz"),
    }
