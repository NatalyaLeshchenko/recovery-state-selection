"""Подготовка изображений, состояния и действий демонстрации."""

from __future__ import annotations

import numpy as np

from act_recovery.settings import INPUT_KEYS


def as_numpy(value):
    """Вход: Tensor или массив. Выход: массив NumPy в памяти CPU."""
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def image_hwc_uint8(value):
    """ Выход: RGB uint8 формы (256, 256, 3)."""
    array = as_numpy(value)
    if array.shape == (3, 256, 256):
        array = array.transpose(1, 2, 0)
    if array.shape != (256, 256, 3):
        raise RuntimeError(f"Неожиданная форма RGB: {array.shape}")
    if array.dtype != np.uint8:
        array = np.rint(np.clip(array, 0, 1) * 255).astype(np.uint8)
    return np.ascontiguousarray(array)


def checked_frame(sample, instruction):

    state = as_numpy(sample["observation.state"]).astype(np.float32)
    action = as_numpy(sample["action"]).astype(np.float32)
    if (
        state.shape != (8,)
        or action.shape != (7,)
        or (not np.isfinite(state).all())
        or (not np.isfinite(action).all())
    ):
        raise RuntimeError("Нужны конечные observation.state (8,) и action (7,).")
    return {
        "observation.images.image": image_hwc_uint8(sample["observation.images.image"]),
        "observation.images.image2": image_hwc_uint8(sample["observation.images.image2"]),
        "observation.state": state,
        "action": action,
        "task": instruction,
    } #state.shape == (8,) -> action.shape == (7,)


def processed_correction_frame(data, index, processor, torch):
    """Вход: запись эксперта, номер кадра, обработчик LIBERO и модуль torch.
    Выход: один кадр в формате входов ACT и соответствующее действие."""
    raw = {
        "observation.images.image": torch.from_numpy(data["observation/pixels/image"][index].copy())
        .permute(2, 0, 1)
        .unsqueeze(0), #(256, 256, 3) → (1, 3, 256, 256)
        "observation.images.image2": torch.from_numpy(data["observation/pixels/image2"][index].copy())
        .permute(2, 0, 1)
        .unsqueeze(0),
        "observation.robot_state": {
            "eef": {
                "pos": torch.from_numpy(data["observation/robot_state/eef/pos"][index].copy()).unsqueeze(0),
                "quat": torch.from_numpy(data["observation/robot_state/eef/quat"][index].copy()).unsqueeze(0),
            },
            "gripper": {
                "qpos": torch.from_numpy(
                    data["observation/robot_state/gripper/qpos"][index].copy()
                ).unsqueeze(0)
            },
        },
    }
    processed = processor.observation(raw)
    frame = {}
    for key in INPUT_KEYS:
        frame[key] = processed[key][0]
    return frame | {"action": data["action"][index]}
