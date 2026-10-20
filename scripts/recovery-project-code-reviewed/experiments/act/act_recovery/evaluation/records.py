"""Чтение тестового набора и возобновление прерванной оценки."""

from __future__ import annotations

from act_recovery.evaluation.models import check_split, load_models, require
from act_recovery.files import (
    StageNotReady,
    file_hash,
    local_path,
    new_folder,
    read_json,
    resolve_pointer,
    write_json,
)
from act_recovery.settings import FAILURE_TYPES, TEST_FILE, TEST_POINTER


def load_test(root, value=None):
    """Вход: root — папка с pilot_runs; value — переданное значение.
    Выход: папка и описание отдельного тестового набора."""
    pool = local_path(root, value) if value else resolve_pointer(root, TEST_POINTER)
    info = read_json(pool / TEST_FILE)
    if info.get("status") != "test_complete" or info.get("purpose") != "evaluation_only":
        raise StageNotReady("Сначала нужен TEST STATES READY с обоими failure types.")
    current, _ = load_models(root, info["provenance"]["models_file"])
    require(current == info["provenance"], "Модели или training provenance изменились после фиксации test.")
    check_split(info["requested_init_state_ids"], current)
    check_split(info["clean_init_state_ids"], current)
    cases = info["cases"]
    require(
        len(cases) == len({case["candidate_id"] for case in cases})
        and set((case["failure_type"] for case in cases)) == set(FAILURE_TYPES),
        "Неполный test или повторные cases.",
    )
    for case in cases:
        require(
            case["init_state_id"] in info["requested_init_state_ids"]
            and case["init_definition_sha256"] not in current["training_init_definition_sha256"]
            and (case.get("independent_recovery_test") is True)
            and (case["label_source"] == "explicit_simulator_event_rules"),
            "Нарушено отделение test от training.",
        )
        for filename, digest in (
            ("start.npz", case["snapshot_sha256"]),
            ("history.npz", case["history_sha256"]),
        ):
            require(
                file_hash(pool / case["candidate_id"] / filename) == digest,
                "Test snapshot/history изменились.",
            )
    protocol = info["protocol"]
    require(
        protocol["n_action_steps"] == 1
        and protocol["recovery_horizon_steps"] == 280
        and (protocol["excluded_init_state_ids"] == current["training_init_state_ids"]),
        "Изменён общий test protocol.",
    )
    return (pool, info)


def run_folder(root, mode, args, pool, info):
    """Вход: root — папка с pilot_runs; mode — mode; args — аргументы команды терминала; pool — pool; info — info.
    Выход: папка проверки и ранее сохранённые частичные результаты."""
    frozen = {
        "mode": mode,
        "test_pool": str(pool),
        "test_pool_sha256": file_hash(pool / TEST_FILE),
        "provenance": info["provenance"],
        "n_action_steps": 1,
        "horizon_steps": 280,
    }
    if mode == "clean":
        frozen["clean_init_state_ids"] = info["clean_init_state_ids"]
    if args.resume:
        output = local_path(root, args.resume)
        require(read_json(output / "run_info.json") == frozen, "Resume относится к другому test или моделям.")
        partial = read_json(output / "partial_results.json")
    else:
        output = new_folder(root, mode + "_selection_eval")
        write_json(output / "run_info.json", frozen)
        partial = {}
        write_json(output / "partial_results.json", partial)
    print("Evaluation:", output, flush=True)
    return (output, partial)
