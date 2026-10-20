"""Загрузка четырёх сравниваемых политик и разделение train/test."""

from __future__ import annotations

from pathlib import Path

from act_recovery.expert.records import load_collection
from act_recovery.files import (
    StageNotReady,
    file_hash,
    local_path,
    read_json,
    resolve_pointer,
)
from act_recovery.models import verify_model
from act_recovery.pool.records import load_pool
from act_recovery.settings import (
    METHODS,
    MODELS_POINTER,
    RESERVED_TEST_IDS,
    SOURCE_FILE,
)


def require(condition, message):
    """Вход: условие и текст ошибки. Выход: ошибка, если условие ложно."""
    if not condition:
        raise RuntimeError(message)


def load_models(root, value=None):
    """Вход: root — папка с pilot_runs; value — переданное значение.
    Выход: описание сравниваемых политик и протокол их исходных данных."""
    folder = local_path(root, value) if value else resolve_pointer(root, MODELS_POINTER)
    filename = folder if folder.is_file() else folder / "models.json"
    info = read_json(filename)
    if info.get("status") != "complete" or set(info.get("models", {})) != set(METHODS):
        raise StageNotReady("Нужны завершённые U, F_ref и Random: THREE SELECTOR POLICIES READY.")
    dataset_file = Path(info["datasets"]) / "datasets.json"
    require(file_hash(dataset_file) == info["datasets_sha256"], "Описание training datasets изменилось.")
    datasets = read_json(dataset_file)
    collection, acquisition = load_collection(root, info["collection"])
    require(
        Path(datasets["collection"]).resolve() == collection
        and file_hash(collection / "collection.json") == datasets["collection_sha256"],
        "Datasets относятся к другому expert collection.",
    )
    pool, source = load_pool(root, acquisition["pool"])
    initial, recipe = (info["initial_policy"], info["recipe"])
    require(
        initial == acquisition["source_initial_model"] == source["models"]["initial"],
        "Модели имеют разные исходные ACT.",
    )
    require(
        info["target_budget"] == acquisition["target_budget"] and recipe["n_action_steps"] == 1,
        "Нужен один demonstration budget и n_action_steps=1.",
    )
    identities = {"initial": initial}
    training_ids = set()
    for case in source["cases"]:
        training_ids.add(case["init_state_id"])
    training_hashes = set()
    for case in source["cases"]:
        training_hashes.add(case["init_definition_sha256"])
    for method in METHODS:
        row = info["models"][method]
        dataset = datasets["datasets"][method]
        require(
            row["initial_policy"] == initial and row["recipe"] == recipe and (row["dataset"] == dataset),
            f"Не совпадает общий training recipe или dataset: {method}.",
        )
        require(
            file_hash(Path(row["training_folder"]) / "run_config.json") == row["run_config_sha256"],
            f"Training record изменился: {method}.",
        )
        run = read_json(Path(row["training_folder"]) / "run_config.json")
        require(
            run.get("status") == "complete"
            and run["completed_updates"] == recipe["steps"]
            and (run["saved_policy"] == row["identity"]),
            f"Не завершено одинаковое дообучение: {method}.",
        )
        sources_file = Path(dataset["root"]) / SOURCE_FILE
        require(
            file_hash(sources_file) == dataset["sources_sha256"], f"Dataset sources изменились: {method}."
        )
        sources = read_json(sources_file)
        require(
            sources["initial_policy"] == initial
            and sources["selector"] == method
            and (sources["n_episodes"] == dataset["n_episodes"] == info["target_budget"])
            and (
                {episode["candidate_id"] for episode in sources["episodes"]}
                == set(acquisition["conditions"][method]["selected_successful_candidates"])
            ),
            f"Training data не соответствуют выбранным demonstrations: {method}.",
        )
        training_ids.update(sources["used_init_state_ids"])
        training_hashes.update(sources["used_init_definition_sha256"])
        identities[method] = row["identity"]
    for identity in identities.values():
        verify_model(identity)
    provenance = {
        "models_file": str(filename),
        "models_sha256": file_hash(filename),
        "models": identities,
        "recipe": recipe,
        "demonstration_budget": info["target_budget"],
        "training_pool": str(pool),
        "training_pool_sha256": file_hash(pool / "protocol.json"),
        "training_init_state_ids": sorted(training_ids),
        "training_init_definition_sha256": sorted(training_hashes),
    }
    return (provenance, source)


def check_split(ids, provenance):
    """Вход: ids — ids; provenance — provenance.
    Выход: ошибка при пересечении обучающих и тестовых стартов."""
    require(bool(ids) and len(ids) == len(set(ids)), "Нужны разные --init-ids.")
    require(
        set(ids).issubset(RESERVED_TEST_IDS),
        f"Для test заранее зарезервированы только IDs {RESERVED_TEST_IDS}.",
    )
    require(
        not set(ids).intersection(provenance["training_init_state_ids"]),
        "Test configuration использована в collection/training или selector pool.",
    )
