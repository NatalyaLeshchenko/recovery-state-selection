"""Параметры одного дообучения и поля CSV с ошибками модели."""

from act_recovery.models import runtime_versions
from act_recovery.settings import SOURCE_FILE, WEIGHTS


def training_record(context):

    from act_recovery.files import file_hash

    request = context["request"]
    recipe = context["recipe"]
    manifest = context["manifest"]
    sources = context["sources"]
    return {
        "status": "running",
        "source_initial_model": context["identity"],
        "recovery_dataset": str(context["dataset_root"]),
        "recovery_source_metadata_sha256": file_hash(context["dataset_root"] / SOURCE_FILE),
        "original_dataset": manifest["dataset_repo"],
        "original_revision": manifest["dataset_revision"],
        "original_train_episodes": manifest["train_episodes"],
        "steps": recipe["steps"],
        "seed": recipe["seed"],
        "lr": recipe["lr"],
        "n_action_steps": recipe["n_action_steps"],
        "chunk_size": context["chunk_size"],
        "weights": dict(zip(("original", "uniform", "grasp", "release"), WEIGHTS)),
        "sampling": "Original frames uniform; recovery episodes uniform, then frames uniform; phase samples episode-balanced.",
        "weight_decay": 0.0001,
        "gradient_clip": 10.0,
        "original_fps": context["original_fps"],
        "recovery_fps": sources["fps"],
        "normalization": "Unchanged initial-policy processors",
        "trainable_parameters": "All parameters already marked requires_grad",
        "used_init_state_ids": sources["used_init_state_ids"],
        "used_init_definition_sha256": sources["used_init_definition_sha256"],
        "runtime_versions": runtime_versions(),
        "selector": request["method"],
        "demonstration_budget": request["budget"],
        "recovery_supervision_steps": recipe["recovery_supervision_steps"],
        "scope": "ACT selected-data comparison; no independent test result in training logs",
        "common_recipe": recipe,
        "dataset_sources_sha256": request["dataset"]["sources_sha256"],
    }


def log_fields():
    """Вход: без аргументов. Выход: названия столбцов losses.csv."""
    fields = ["step", "loss", "l1", "kl"]
    for prefix in ("l1", "valid"):
        for name in ("original", "uniform", "grasp", "release"):
            fields.append(prefix + "_" + name)
    fields.extend(["original_frame", "uniform_frame", "grasp_frame", "release_frame", "peak_allocated_gib"])
    return fields
