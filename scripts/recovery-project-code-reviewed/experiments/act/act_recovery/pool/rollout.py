"""хочу: состояние после ошибки"""

from __future__ import annotations

import random
from collections import deque

import numpy as np

from act_recovery.environment import create_environment
from act_recovery.observations import (
    policy_observation,
    remember_observation,
    skipped_result,
)
from act_recovery.pool.contacts import finger_contacts
from act_recovery.pool.events import eligible_miss
from act_recovery.pool.intervention import capture_candidate


def collect_candidate(request, data):
    import torch
    from lerobot.policies.act.modeling_act import ACTPolicy
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.processor.env_processor import LiberoProcessorStep

    from act_recovery.scene import (
        choose_initial_state,
        geometry,
        observed_success,
        state_and_ctrl,
        stopped,
    )

    protocol = request["protocol"] #загрузка модели + достали настройки
    random.seed(protocol["seed"])
    np.random.seed(protocol["seed"])
    torch.manual_seed(protocol["seed"])


    policy = ACTPolicy.from_pretrained(request["model_path"], local_files_only=True, strict=True) #исходная ACT с ее весами

    policy.config.n_action_steps = protocol["n_action_steps"]
    policy.to("cuda").eval()
    #Подготавливает и нормализует входы ACT. Выход аст - в команды для выполнения
    preprocessor, postprocessor = make_pre_post_processors(
        policy.config,
        pretrained_path=request["model_path"],
        preprocessor_overrides={"device_processor": {"device": "cuda"}},
    )
    processor = LiberoProcessorStep() #от либеро - наблюдение в 7 чисел
    wait_steps = max(protocol["open_steps"], 1 + protocol["miss_lift_steps"])
    vec = create_environment(protocol["nominal_horizon_steps"] + wait_steps)
    try:
        #среду создали
        data["init_hash"] = choose_initial_state(
            vec, request["init_state_id"], protocol["excluded_init_state_ids"]
        )
        obs, info = vec.reset(seed=protocol["seed"])
        data["reset_state"], data["reset_ctrl"] = state_and_ctrl(vec)
        data["initial_geometry"] = geometry(vec)
        data["actions"] = [] #выполненные команды
        data["history"] = deque(maxlen=protocol["history_observations"]) #последнее наблюдение
        remember_observation(data, obs)
        policy.reset()
        held_ever = False
        hold_count = 0

        #главный цикл движения
        for step in range(protocol["nominal_horizon_steps"]):

            #Возьми последние картинки и состояние робота, преобразуй их в тензоры и подготовь для ACT
            with torch.inference_mode(), torch.autocast(device_type="cuda", enabled=policy.config.use_amp):
                batch = preprocessor(policy_observation(data["last_observation"], processor, torch))

                """ batch = {
                        "observation.images.image": ...,   # первая картинка: (1, 3, 256, 256)
                        "observation.images.image2": ...,  # вторая картинка: (1, 3, 256, 256)
                        "observation.state": ...,          # состояние робота: (1, 8)
                    }"""

                for key in batch:
                    batch[key] = batch[key].to("cuda")

                    """action от ACT"""
                action = postprocessor(policy.select_action(batch)).detach().cpu().numpy()



            #пора ли вмешиваться?
            before = geometry(vec)
            contacts = finger_contacts(vec)
            rise = before["bowl_pos"][2] - data["initial_geometry"]["bowl_pos"][2] #подъем миски 
            previous_close = bool(data["actions"] and data["actions"][-1][0, -1] > 0)

            #точно ли держит миску? 
            if contacts["bilateral"] and previous_close and (rise >= protocol["held_min_bowl_rise_m"]):
                hold_count += 1
            else:
                hold_count = 0
            held_ever = held_ever or hold_count >= protocol["held_consecutive_observations"]
            data["trace"].append(
                {
                    "nominal_step": step,
                    "geometry": before,
                    "contacts": contacts,
                    "intended_action": action[0].tolist(),
                    "held_streak": hold_count,
                    "held_ever": held_ever,
                }
            )
            #по какому типу вмешательство
            kind = request["failure_type"]
            miss = kind == "missed_grasp" and eligible_miss(
                action, before, data["initial_geometry"], held_ever, protocol
            )

            #Нужно создать падение; робот уже доказанно держал миску, сейчас продолжает держать её обоими пальцами и поднял достаточно высоко.
            drop = (
                kind == "object_dropped"
                and held_ever
                and contacts["bilateral"]
                and (rise >= protocol["lift_threshold_m"])
            )
            if miss or drop:
                return capture_candidate(request, vec, data, step, action, before, held_ever) #решили, когда вмешаться
            result = vec.step(action)
            data["actions"].append(action.copy())
            obs = result[0]
            remember_observation(data, obs)
            if stopped(result) or observed_success(result):
                return skipped_result(request, "Эпизод завершился до подходящего вмешательства.")
        return skipped_result(request, "За отведённые шаги не найден момент вмешательства.")
    finally:
        vec.close()
