"""Преобразование двух изображений и состояния робота во вход ACT."""

from __future__ import annotations

import numpy as np

from act_recovery.settings import RAW_SHAPES


def remember_observation(data, obs):

    data["last_observation"] = raw_observation(obs)
    data["history"].append(data["last_observation"])


def skipped_result(request, reason):
   
    return {
        "status": "skipped",
        "reason": reason,
        "candidate_id": request["candidate_id"],
        "failure_type": request["failure_type"],
    }


def raw_observation(obs):
    
    result = {}
    for key in RAW_SHAPES:
        value = obs
        for part in key.split("/")[1:]:
            value = value[part]
        result[key] = np.asarray(value).copy()
    return result


def policy_observation(saved, processor, torch):
    
    raw = {
        "observation.images.image": torch.from_numpy(saved["observation/pixels/image"].copy()).permute(
            0, 3, 1, 2
        ),
        "observation.images.image2": torch.from_numpy(saved["observation/pixels/image2"].copy()).permute(
            0, 3, 1, 2
        ),
        "observation.robot_state": {
            "eef": {
                "pos": torch.from_numpy(saved["observation/robot_state/eef/pos"].copy()),
                "quat": torch.from_numpy(saved["observation/robot_state/eef/quat"].copy()),
            },
            "gripper": {"qpos": torch.from_numpy(saved["observation/robot_state/gripper/qpos"].copy())},
        },
    }
    processed = processor.observation(raw)
    for key in ("observation.images.image", "observation.images.image2"):
        processed[key] = processed[key].float() / 255.0
    return processed


def compare_normalization(model_paths, torch):
    
    from safetensors.torch import load_file

    reference = None
    for folder in model_paths:
        files = sorted(folder.glob("*normalizer*.safetensors"))
        if not files:
            raise RuntimeError(f"Не найдены сохранённые normalization stats: {folder}")
        current = {}
        for path in files:
            current[path.name] = load_file(str(path), device="cpu")
        if reference is None:
            reference = current
        if set(current) != set(reference):
            raise RuntimeError("Normalization processor files различаются между nominal members.")
        for name, tensors in current.items():
            if set(tensors) != set(reference[name]):
                raise RuntimeError("Normalization tensor keys различаются между nominal members.")
            for key, value in tensors.items():
                expected = reference[name][key]
                if value.shape != expected.shape or not torch.equal(value, expected):
                    raise RuntimeError(f"Нет общей D0 normalization: {folder / name}, tensor={key}")
    normalization_keys = {}
    for name, tensors in reference.items():
        normalization_keys[name] = sorted(tensors)
    return normalization_keys
