"""Чтение плана и проверка его связи с общим набором состояний."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from act_recovery.files import file_hash, local_path, read_json, resolve_pointer
from act_recovery.pool.records import load_ensemble, load_pool
from act_recovery.selection.methods import disagreement, query_plan
from act_recovery.settings import METHODS


def load_plan(root, value=None):
    """Вход: root — папка с pilot_runs; value — переданное значение.
    Выход: план отбора, общий пул и протокол."""
    folder = local_path(root, value) if value else resolve_pointer(root, "selection_plan_path.txt")
    plan_path = folder if folder.is_file() else folder / "selection.json"
    plan = read_json(plan_path)
    if set(plan.get("conditions", {})) != set(METHODS):
        raise RuntimeError("Нужен общий selection plan для U, F_ref и Random.")
    pool, protocol = load_pool(root, plan["pool"])
    scores_folder = Path(plan["scores"])
    scores = read_json(scores_folder / "scores.json")
    if (
        scores.get("status") != "complete"
        or Path(scores["pool"]).resolve() != pool
        or file_hash(pool / "protocol.json") != scores["pool_protocol_sha256"]
        or (file_hash(scores_folder / "predictions.npz") != scores["predictions_sha256"])
    ):
        raise RuntimeError("Scores и plan не соответствуют зафиксированному common pool.")
    pool_cases = {}
    for case in protocol["cases"]:
        pool_cases[case["candidate_id"]] = case
    score_cases = scores["cases"]
    if len(score_cases) != len(pool_cases) or {row["candidate_id"] for row in score_cases} != set(pool_cases):
        raise RuntimeError("Score rows покрывают другой candidate pool.")
    with np.load(scores_folder / "predictions.npz", allow_pickle=False) as saved:
        candidate_ids = []
        for row in score_cases:
            candidate_ids.append(row["candidate_id"])
        if saved["candidate_ids"].tolist() != candidate_ids:
            raise RuntimeError("Predictions и score rows имеют разный порядок candidates.")
        recalculated = disagreement(saved["normalized_actions"])
    for row, value in zip(score_cases, recalculated):
        case = pool_cases[row["candidate_id"]]
        if (
            row["snapshot_sha256"] != case["snapshot_sha256"]
            or row["failure_type"] != case["failure_type"]
            or (not np.isclose(row["U"], value, atol=1e-12, rtol=1e-12))
        ):
            raise RuntimeError("Uncertainty score или reference label не соответствует сохранённым данным.")
    budget = plan["target_successful_demonstrations"]
    if isinstance(budget, bool) or not isinstance(budget, int) or budget < 1:
        raise RuntimeError("Некорректный demonstration budget.")
    expected = query_plan(scores["cases"], budget, plan["seed"])
    if expected != plan["conditions"]:
        raise RuntimeError("Selection plan изменён или не воспроизводится из сохранённых scores.")
    _, ensemble = load_ensemble(root, protocol["ensemble_folder"])
    if (
        ensemble["models"][0]["identity"] != protocol["models"]["initial"]
        or file_hash(Path(protocol["ensemble_folder"]) / "ensemble.json") != protocol["ensemble_sha256"]
    ):
        raise RuntimeError("Исходная ACT или nominal ensemble изменились.")
    return (plan_path, plan, pool, protocol)
