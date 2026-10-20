"""Чтение успешных записей эксперта и границы эпизодов."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from act_recovery.files import (
    StageNotReady,
    file_hash,
    local_path,
    read_json,
    resolve_pointer,
)
from act_recovery.settings import COLLECTION_POINTER, METHODS


def load_collection(root, value=None):
    """Вход: root — папка с pilot_runs; value — переданное значение.
    Выход: папка и описание экспертных демонстраций."""
    folder = local_path(root, value) if value else resolve_pointer(root, COLLECTION_POINTER)
    info = read_json(folder / "collection.json")
    if (
        not info.get("complete")
        or info.get("status") != "complete"
        or set(info["conditions"]) != set(METHODS)
    ):
        raise StageNotReady("Нужен завершённый expert collection для всех трёх selectors.")
    if (
        file_hash(Path(info["pool"]) / "protocol.json") != info["pool_protocol_sha256"]
        or file_hash(info["selection_plan"]) != info["selection_plan_sha256"]
        or file_hash(root / "pilot_runs/manifest.json") != info["manifest_sha256"]
    ):
        raise RuntimeError("Common pool, plan или исходный D0 изменились после collection.")
    from act_recovery.models import verify_identity

    verify_identity(info["source_initial_model"])
    for method, row in info["conditions"].items():
        accepted = row["selected_successful_candidates"]
        if (
            len(accepted) != info["target_budget"]
            or len(set(accepted)) != len(accepted)
            or (not row["budget_met"])
        ):
            raise RuntimeError(f"Не выполнен demonstration budget для {method}.")
        for candidate in accepted:
            attempt = info["unique_attempts"][candidate]
            if not attempt["accepted"] or attempt["candidate_id"] != candidate:
                raise RuntimeError("Selected successful dataset содержит failed attempt.")
    return (folder, info)


def load_successful_demo(attempt, instruction):
    """Вход: attempt — attempt; instruction — instruction.
    Выход: метаданные и массивы успешной демонстрации."""
    from act_recovery.files import file_hash

    folder = Path(attempt["folder"])
    if attempt.get("accepted") is not True:
        raise RuntimeError("Неуспешная попытка не является demonstration.")
    for filename, key in (
        ("metadata.json", "metadata_sha256"),
        ("demonstration.npz", "demonstration_sha256"),
    ):
        if file_hash(folder / filename) != attempt[key]:
            raise RuntimeError(f"Исходные данные изменились: {folder / filename}")
    metadata = read_json(folder / "metadata.json")
    if (
        metadata.get("accepted") is not True
        or metadata.get("success") is not True
        or metadata.get("observation_timing") != "before_action"
        or (metadata.get("instruction") != instruction)
        or (metadata.get("control_frequency_hz") != 20)
        or (metadata.get("init_definition_sha256") != attempt["init_definition_sha256"])
    ):
        raise RuntimeError("Метаданные expert demonstration не соответствуют протоколу.")
    with np.load(folder / "demonstration.npz", allow_pickle=False) as source:
        data = {}
        for key in source.files:
            data[key] = source[key].copy()
    n = len(data["action"])
    shapes = {
        "observation/pixels/image": (n, 256, 256, 3),
        "observation/pixels/image2": (n, 256, 256, 3),
        "observation/robot_state/eef/pos": (n, 3),
        "observation/robot_state/eef/quat": (n, 4),
        "observation/robot_state/gripper/qpos": (n, 2),
        "action": (n, 7),
        "reward": (n,),
        "phase": (n,),
    }
    for key, shape in shapes.items():
        if key not in data or data[key].shape != shape:
            raise RuntimeError(f"Неверная форма {key}; ожидается {shape}.")
        if data[key].dtype.kind in "fiu" and (not np.isfinite(data[key]).all()):
            raise RuntimeError(f"Неконечные числовые значения: {key}.")
    for key in ("observation/pixels/image", "observation/pixels/image2"):
        if data[key].dtype != np.uint8:
            raise RuntimeError("Ожидаются исходные RGB uint8.")
    if not n or n != metadata["n_frames"] or (not np.any(data["reward"] > 0)):
        raise RuntimeError("Пустая demonstration, неверная длина или отсутствует success reward.")
    if np.any(np.abs(data["action"]) > 1.00001):
        raise RuntimeError("Expert actions выходят за пределы OSC action limits.")
    if np.any(data["reward"][:-1] > 0) or data["reward"][-1] <= 0:
        raise RuntimeError("Запись должна останавливаться на первом benchmark success.")
    return (metadata, data)


def validate_ranges(episodes, n_frames):
    """Вход: episodes — episodes; n_frames — n_frames.
    Выход: ошибка при неполном или неверном разбиении на эпизоды."""
    cursor = 0
    for number, episode in enumerate(episodes):
        if episode["episode_index"] != number or episode["start"] != cursor or episode["stop"] <= cursor:
            raise RuntimeError("Неверные границы recovery episodes.")
        cursor = episode["stop"]
    if cursor != n_frames or not episodes:
        raise RuntimeError("Границы episodes не покрывают dataset.")
