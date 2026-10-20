"""Чтение исходной ACT и проверка ее обучающей выборки."""

from __future__ import annotations
from pathlib import Path
from act_recovery.files import file_hash, read_json
from act_recovery.models import identity, policy_settings


def checked_source(train, policy, manifest):

    
    if policy.get("type") != "act" or train.get("policy", {}).get("type") != "act":
        raise RuntimeError("Нужна исходная ACT, обученная на D0.")
    
    if train.get("resume") is True or train["policy"].get("pretrained_path"):
        raise RuntimeError(
            "Исходный запуск был resume/fine-tuning; нельзя считать его независимым nominal member."
        )
    
    for key in ("dataset_repo", "dataset_revision", "train_episodes"):
        if key not in manifest:
            raise RuntimeError(f"Нет {key} в pilot_runs/manifest.json.")
        
    dataset = train.get("dataset", {})

    episodes = dataset.get("episodes")
    expected = manifest["train_episodes"]
    if (
        not isinstance(episodes, list)
        or not episodes
        or len(episodes) != len(set(episodes))
        or (sorted(episodes) != sorted(expected))
    ):
        raise RuntimeError("Train episode IDs не совпадают с исходным D0.")
    
    if set(episodes) & set(manifest.get("validation_episodes", [])):
        raise RuntimeError("Nominal validation episodes включены в train split.")
    for key in ("steps", "batch_size", "seed"):
        if not isinstance(train.get(key), int) or isinstance(train[key], bool):
            raise RuntimeError(f"Некорректное поле {key} в исходном train_config.json.")
        
    job = train.get("job")

    return {"repo_id": dataset["repo_id"], "revision": dataset["revision"], "episodes": list(episodes)}

#словарик
def context(root, initial_path=None):

    
    path = initial_path or Path("pilot_runs/training/initial/checkpoints/010000/pretrained_model")
    path = Path(path).expanduser()

    initial = (path if path.is_absolute() else root / path).resolve()
    source = identity(initial) #сведения о файлах модели
    train, policy = (read_json(initial / "train_config.json"), read_json(initial / "config.json")) #словарь настроек

    manifest_path = root / "pilot_runs/manifest.json"
    manifest = read_json(manifest_path) #читаем описание эксперимента
    
    d0 = checked_source(train, policy, manifest)
    checkpoint_name = initial.parent.name

   
    if not checkpoint_name.isdigit():
        raise RuntimeError(
            "Укажите исходный numbered checkpoint, например checkpoints/010000/pretrained_model."
        )
    return {
        "root": root,
        "initial": initial,
        "identity": source,
        "train": train,
        "policy": policy,
        "d0": d0,
        "manifest_sha256": file_hash(manifest_path),
    }
