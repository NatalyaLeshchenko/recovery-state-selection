"""Подключение повтора состояния и журнала действий к оценке LeRobot."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from act_recovery.files import write_json
from act_recovery.scene import (
    checked_action,
    choose_initial_state,
    exact_error,
    geometry,
    observed_success,
    state_and_ctrl,
    stopped,
)


class EvaluationHooks:
    """Хранит сохранённый старт, повторяет действия при reset и записывает шаги оценки."""

    def __init__(self, request, original_reset, original_step):
        """Вход: параметры оценки и исходные методы reset/step.
        Выход: объект с загруженным snapshot и пустым журналом действий."""
        self.request, self.reset_base, self.step_base = (request, original_reset, original_step)
        self.protocol = request["protocol"]
        self.output = Path(request["output"])
        self.vectors, self.rows = ([], [])
        self.restore = None
        with np.load(Path(request["case_dir"]) / "start.npz", allow_pickle=False) as data:
            saved_arrays = {}
            for key in ("simulator_state", "ctrl", "reset_state", "reset_ctrl", "past_actions"):
                saved_arrays[key] = data[key].copy()
            self.saved = saved_arrays
        prefix = self.saved["past_actions"]
        if prefix.ndim != 3 or prefix.shape[1:] != (1, 7) or (not np.isfinite(prefix).all()):
            raise RuntimeError("Неожиданный сохранённый action prefix.")

    def reset(self, vec, *args, **kwargs):
        """Вход: среда и параметры reset.
        Выход: наблюдение в восстановленном recovery-старте и измерения точности повтора."""
        self.vectors.append(vec)
        actual_hash = choose_initial_state(
            vec, self.request["init_state_id"], self.protocol["excluded_init_state_ids"]
        )
        if actual_hash != self.request["init_definition_sha256"]:
            raise RuntimeError("Определение LIBERO initial state изменилось после collection.")
        obs, info = self.reset_base(vec, *args, **kwargs)
        state, ctrl = state_and_ctrl(vec)
        reset_error = exact_error(state, self.saved["reset_state"], "reset state")
        reset_ctrl_error = exact_error(ctrl, self.saved["reset_ctrl"], "reset ctrl")
        for action in self.saved["past_actions"]:
            result = self.step_base(vec, action)
            obs, info = (result[0], result[4])
            if stopped(result) or observed_success(result):
                raise RuntimeError("Эпизод завершился при replay до recovery start.")
        state, ctrl = state_and_ctrl(vec)
        state_error = exact_error(state, self.saved["simulator_state"], "recovery state")
        ctrl_error = exact_error(ctrl, self.saved["ctrl"], "recovery ctrl")
        self.restore = {
            "max_reset_state_error": reset_error,
            "max_reset_ctrl_error": reset_ctrl_error,
            "max_state_error": state_error,
            "max_ctrl_error": ctrl_error,
            "prefix_steps": len(self.saved["past_actions"]),
            "init_state_id": self.request["init_state_id"],
            "recovery_horizon_steps": self.protocol["recovery_horizon_steps"],
        }
        write_json(self.output / "restore_check.json", self.restore)
        self.rows.clear()
        print(f"RESTORE OK: state_error={state_error:.3g}, ctrl_error={ctrl_error:.3g}", flush=True)
        return (obs, info)

    def step(self, vec, action):
        """Вход: среда и команда ACT.
        Выход: результат шага; геометрия и команда добавляются в журнал."""
        action = checked_action(action)
        if self.restore is None:
            raise RuntimeError("Получен step без проверенного восстановления.")
        if len(self.rows) >= self.protocol["recovery_horizon_steps"]:
            raise RuntimeError("Evaluator продолжил эпизод после лимита recovery.")
        before = geometry(vec)
        result = self.step_base(vec, action)
        after = geometry(vec)
        obs, reward, terminated, truncated, info = result
        at_limit = len(self.rows) + 1 == self.protocol["recovery_horizon_steps"]
        if at_limit and (not stopped(result)):
            truncated = np.ones_like(np.asarray(truncated), dtype=bool)
        self.rows.append(
            {
                "step": len(self.rows),
                "action": action[0].tolist(),
                "before": before,
                "after": after,
                "reward": np.asarray(reward).reshape(-1).tolist(),
                "terminated": bool(np.any(terminated)),
                "truncated": bool(np.any(truncated)),
                "fixed_recovery_horizon_reached": at_limit,
            }
        )
        return (obs, reward, terminated, truncated, info)
