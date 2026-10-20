"""Движения эксперта: подойти, захватить, поднять, перенести и отпустить."""

from __future__ import annotations

import numpy as np

from act_recovery.expert.scene import GoalReached, flatten_observation, geometry


def run_expert(vec, raw, obs, grasp_offset, placement_offset, max_steps, records):


    """ 
    vec - среда, которой отправляем команды: vec.step(action)
    raw	- внутренняя среда робота: через неё читаем координаты объектов и настройки контроллера
    obs - текущие наблюдения: изображения камер и состояние робота
    grasp_offset - на сколько сместить захват относительно центра миски, чтобы удобно её взять
    placement_offset - где должна оказаться миска относительно центра тарелки
    max_steps - максимальное число команд на исправление
    records	- словарь, в который складываем записанную демонстрацию
    """
   
    scale = np.asarray(raw.robots[0].controller.output_max[:3], dtype=float)
    if scale.shape != (3,) or np.any(scale <= 0):
        raise RuntimeError("Неожиданная шкала OSC controller.")

    #запомнили нач высоту миски
    start_bowl_z = geometry(raw)["bowl_pos"][2]

    def eef_pos():
         # X, Y, Z в метрах
        return np.asarray(geometry(raw)["eef_pos"])

    def bowl_pos():
        
        return np.asarray(geometry(raw)["bowl_pos"])

    def step(action, phase):
        #action — команда из семи чисел. phase — подпись этапа,"lift"

        #Когда получим новое наблюдение, обновить переменную obs из run_expert, чтобы следующее движение использовало уже новую картинку и состояние
        nonlocal obs

        if len(records["actions"]) >= max_steps:
            raise RuntimeError("Достигнут лимит действий expert.")
        
        before = flatten_observation(obs)


        action = np.asarray(action, dtype=np.float32).reshape(1, 7)

        #Запоминаем координаты перед движением, выполняем команду в симуляторе и получаем новое наблюдение
        before_geometry = geometry(raw)
        obs, reward, terminated, truncated, _ = vec.step(action)

        for key, value in before.items():
            records["observations"].setdefault(key, []).append(value)

        records["actions"].append(action[0].copy()) # наблюдение ДО команды
        records["rewards"].append(float(np.asarray(reward).reshape(-1)[0])) # полученная награда
        records["phases"].append(phase)
        records["geometry"].append(before_geometry) # координаты ДО команды
        records["success"] = bool(np.any(np.asarray(reward) > 0))
        records["ended"] = bool(np.any(terminated) or np.any(truncated))
        if len(records["actions"]) % 20 == 0 or records["success"]:
            print(
                f"Expert steps={len(records['actions'])}; phase={phase}; success={records['success']}",
                flush=True,
            )
        if records["success"]:
            raise GoalReached()
        if records["ended"]:
            raise RuntimeError("Эпизод завершился без достижения цели.")

    def hold(gripper, steps, phase):
        #сколько держим команду захвата
        for _ in range(steps):
            action = np.zeros(7)
            action[-1] = gripper
            step(action, phase)

    def move_to(target, gripper, phase, max_move_steps=40):
        
        target = np.asarray(target)
        for _ in range(max_move_steps):
            error = target - eef_pos()
            if np.linalg.norm(error) < 0.008:
                return
            action = np.zeros(7)
            action[:3] = np.clip(0.6 * error / scale, -0.35, 0.35)
            action[-1] = gripper
            step(action, phase)
        if np.linalg.norm(target - eef_pos()) >= 0.008:
            raise RuntimeError(f"Захват не достиг позиции: {phase}.")

    print("PHASE: approach_grasp", flush=True)

    #Сначала открываем захват и подходим к миске
    hold(-1, 6, "open_before_grasp")
    grasp_target = bowl_pos() + grasp_offset
    print("Target displacement (m):", (grasp_target - eef_pos()).round(5), flush=True)
    move_to(grasp_target, -1, "approach_grasp")
    print("PHASE: close_gripper_and_lift", flush=True)

    #Закрываем захват и поднимаем
    hold(1, 12, "close_gripper")
    lift_target = eef_pos()
    lift_target[2] += 0.1
    move_to(lift_target, 1, "lift")

    #точку на 10 см выше текущего положения захвата
    rise = float(bowl_pos()[2] - start_bowl_z)
    records["bowl_rise_m"] = rise
    print("Bowl rise (m):", rise, flush=True)
    if rise < 0.04:
        raise RuntimeError("Миска не поднялась: попытка не принимается как успешная.")
    print("PHASE: transport", flush=True)

    #перенос на тарелку
    desired_bowl = np.asarray(geometry(raw)["plate_pos"]) + placement_offset
    placement_target = desired_bowl + (eef_pos() - bowl_pos())

    """desired_bowl = [0.40, 0.20, 0.92]
    eef_pos() - bowl_pos() = [0, 0, 0.03]   ----> placement_target = [0.40, 0.20, 0.95]"""

    above_plate = placement_target.copy()
    above_plate[2] = max(eef_pos()[2], placement_target[2] + 0.1)
    move_to(above_plate, 1, "transport")
    print("PHASE: lower_and_release", flush=True)
    move_to(placement_target, 1, "lower")
    hold(-1, 12, "release")
    hold(-1, 20, "settle")
    raise RuntimeError("Expert завершил последовательность, но benchmark не подтвердил успех.")
