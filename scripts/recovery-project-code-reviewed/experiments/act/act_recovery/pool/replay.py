"""делаем новую среду, проверяем воспроизводимость """

from __future__ import annotations

from pathlib import Path

import numpy as np

from act_recovery.environment import create_environment, goal_at_start
from act_recovery.files import write_json
from act_recovery.observations import raw_observation


def check_restoration(request, case):
    
    from act_recovery.scene import (
        choose_initial_state,
        exact_error,
        observed_success,
        state_and_ctrl,
        stopped,
    )

    protocol = request["protocol"]
    output = Path(request["output"])
    vec = create_environment(case["prefix_steps"] + protocol["recovery_horizon_steps"]) #новая среда, прежний старт 
    try:
        actual_hash = choose_initial_state(vec, request["init_state_id"], protocol["excluded_init_state_ids"])
        if actual_hash != case["init_definition_sha256"]:
            raise RuntimeError("Определение начального состояния изменилось.")
        
        obs, info = vec.reset(seed=protocol["seed"]) #устанавливаем нач среду
        state, ctrl = state_and_ctrl(vec)

        with np.load(output / "start.npz", allow_pickle=False) as saved: 
            reset_error = exact_error(state, saved["reset_state"], "reset state") # # сохранённое начальное состояние симулятора
            reset_ctrl_error = exact_error(ctrl, saved["reset_ctrl"] 
                                           , "reset ctrl") # сохранённые начальные сигналы управления

            
            for action in saved["past_actions"]: # повтор движений
                result = vec.step(action) 

                """result = {
                        observation,  # новое наблюдение
                        reward,       # награда
                        terminated,   # эпизод завершён
                        truncated,    # достигнут лимит
                        info,         # дополнительные сведения
                    } """
                obs = result[0]
                if stopped(result) or observed_success(result):
                    raise RuntimeError("Эпизод завершился до повторяемого recovery start.")
            state, ctrl = state_and_ctrl(vec)
            restore = {
                "max_reset_state_error": reset_error,
                "max_reset_ctrl_error": reset_ctrl_error,
                "max_state_error": exact_error(state, saved["simulator_state"], "recovery state"),
                "max_ctrl_error": exact_error(ctrl, saved["ctrl"], "recovery ctrl"),
                "prefix_steps": len(saved["past_actions"]),
                "init_state_id": request["init_state_id"],
                "recovery_horizon_steps": protocol["recovery_horizon_steps"],
            }
            write_json(output / "restore_check.json", restore)
            if goal_at_start(vec):
                raise RuntimeError("Сохранённый старт уже удовлетворяет цели.")
            errors = {}

            #что модель видит, observations сравниваем
            for key, value in raw_observation(obs).items():
                difference = float(np.max(np.abs(value.astype(np.float64) - saved[key].astype(np.float64))))
                errors[key] = difference
                tolerance = 0 if "/pixels/" in key else 1e-06
                if difference > tolerance:
                    raise RuntimeError(f"Не совпадает наблюдение {key}: ошибка {difference}")
        return dict(restore, observation_errors=errors, goal_at_start=False)
    finally:
        vec.close()
