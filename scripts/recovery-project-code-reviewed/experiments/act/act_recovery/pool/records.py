"""Загрузка ансамбля и сохранённого набора кандидатов."""

from __future__ import annotations

from pathlib import Path

from act_recovery.files import (
    StageNotReady,
    file_hash,
    local_path,
    read_json,
    resolve_pointer,
)
from act_recovery.settings import FAILURE_TYPES, POOL_POINTER


def load_ensemble(root, value=None):
    
    from act_recovery.models import policy_settings, verify_identity

    folder = local_path(root, value) if value else resolve_pointer(root, "uncertainty_ensemble_path.txt") #найти описание трех обычных ACT
    record = read_json(folder / "ensemble.json")
    models = record.get("models", [])
    if (
        record.get("complete") is not True
        or record.get("status") != "complete"
        or record.get("smoke_only") is not False
        or (len(models) != 3)
    ):
        raise RuntimeError("Нужен завершённый ensemble из трёх nominal ACT, не smoke run.")
    if models[0].get("role") != "fixed_initial_actor_and_member0":
        raise RuntimeError("Не определена общая исходная ACT для collection и fine-tuning.")
    seeds = set()
    for m in models:
        seeds.add(m["seed"])
    if len(seeds) != 3:
        raise RuntimeError("Seeds nominal members должны различаться.")
    weight_hashes = set()
    for m in models:
        weight_hashes.add(m["identity"]["sha256"]["model.safetensors"])
    if len(weight_hashes) != 3:
        raise RuntimeError("Weights nominal members должны различаться.")
    initial_settings = None
    for model in models:
        verify_identity(model["identity"])
        cfg = read_json(Path(model["identity"]["path"]) / "config.json")
        settings = policy_settings(cfg)
        if initial_settings is None:
            initial_settings = settings
        if cfg.get("type") != "act" or settings != initial_settings:
            raise RuntimeError("Архитектура и training settings members должны совпадать.")
    if file_hash(root / "pilot_runs/manifest.json") != record["manifest_sha256"]:
        raise RuntimeError("Исходный D0 manifest изменился после обучения ensemble.")
    return (folder, record)


def load_pool(root, value=None): #найти описание сохраненных состояний после ошибок
    
    pool = local_path(root, value) if value else resolve_pointer(root, POOL_POINTER)
    protocol = read_json(pool / "protocol.json")
    if protocol.get("complete") is not True or protocol.get("status") != "complete":
        raise StageNotReady(
            "Пул collection не завершён или не содержит два failure types. Сначала нужен POOL READY."
        )
    cases = protocol["cases"]
    candidate_ids = set()
    for case in cases:
        candidate_ids.add(case["candidate_id"])
    if len(candidate_ids) != len(cases):
        raise RuntimeError("Повторяющиеся candidate IDs.")
    if set((case["failure_type"] for case in cases)) != set(FAILURE_TYPES):
        raise RuntimeError("Нужны оба reference failure types.")
    for case in cases:
        if case.get("label_source") != "explicit_simulator_event_rules":
            raise RuntimeError("Нужны проверенные reference event labels.")
        directory = pool / case["candidate_id"]
        for name, digest in (("start.npz", case["snapshot_sha256"]), ("history.npz", case["history_sha256"])):
            if file_hash(directory / name) != digest:
                raise RuntimeError(f"Изменились candidate данные: {directory / name}")
    return (pool, protocol)
