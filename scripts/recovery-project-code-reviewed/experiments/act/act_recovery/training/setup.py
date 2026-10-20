"""Загрузка исходной ACT и двух источников данных для одного способа отбора."""

import random
from pathlib import Path

import numpy as np

from act_recovery.expert.records import validate_ranges
from act_recovery.files import file_hash, read_json, resolve_pointer
from act_recovery.models import verify_identity, verify_model
from act_recovery.settings import SOURCE_FILE
from act_recovery.training.sampling import focus_groups


def prepare_training(request):

    import torch
    from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
    from lerobot.policies.act.modeling_act import ACTPolicy
    from lerobot.policies.factory import make_pre_post_processors

    root = Path(request["project_root"])
    dataset = request["dataset"]
    dataset_root = Path(dataset["root"])
    sources = read_json(dataset_root / SOURCE_FILE)
    manifest = read_json(root / "pilot_runs/manifest.json")
    recipe = request["recipe"]

    
    identity = sources["initial_policy"]
    verify_identity(identity)
    verify_model(identity)
    source = Path(identity["path"])
    episodes = sources["episodes"]
    validate_ranges(episodes, sources["n_frames"])
    grasps = focus_groups(episodes, sources["grasp_focus_indices"])
    releases = focus_groups(episodes, sources["release_focus_indices"])
    random.seed(recipe["seed"])
    np.random.seed(recipe["seed"])
    torch.manual_seed(recipe["seed"])
    torch.cuda.manual_seed_all(recipe["seed"])
    rng = np.random.default_rng(recipe["seed"])
    policy = ACTPolicy.from_pretrained(source, local_files_only=True, strict=True)
    indices = list(policy.config.action_delta_indices)
    if policy.config.n_obs_steps != 1:
        raise RuntimeError("Expected ACT with one observation per query.")
    if not indices or indices != list(range(len(indices))) or recipe["n_action_steps"] > len(indices):
        raise RuntimeError("Invalid ACT action chunk.")
    policy.config.n_action_steps = recipe["n_action_steps"]
    policy.to("cuda")
    preprocessor, _ = make_pre_post_processors(policy.config, pretrained_path=source)
    original_fps = float(
        LeRobotDatasetMetadata(manifest["dataset_repo"], revision=manifest["dataset_revision"]).fps
    )
    if original_fps <= 0 or float(sources["fps"]) != 20:
        raise RuntimeError("Unexpected dataset frequency.")
    original_times = []
    recovery_times = []
    for index in indices:
        original_times.append(index / original_fps)
        recovery_times.append(index / sources["fps"])
    original = LeRobotDataset(
        manifest["dataset_repo"],
        revision=manifest["dataset_revision"],
        episodes=manifest["train_episodes"],
        video_backend="pyav",
        delta_timestamps={"action": original_times},
    )
    recovery = LeRobotDataset(
        dataset["repo_id"], root=dataset_root, delta_timestamps={"action": recovery_times}
    )
    if not len(original) or len(recovery) != sources["n_frames"]:
        raise RuntimeError("The original or recovery dataset has changed.")
    for episode in episodes:
        sample = recovery[episode["stop"] - 1]
        if int(sample["episode_index"].item()) != episode["episode_index"] or bool(
            sample["action_is_pad"][0].item()
        ):
            raise RuntimeError("Incorrect recovery episode boundary.")
        if len(indices) > 1 and (not bool(sample["action_is_pad"][1:].all().item())):
            raise RuntimeError("An action chunk crosses a recovery episode boundary.")
    return {
        "request": request,
        "root": root,
        "manifest": manifest,
        "recipe": recipe,
        "sources": sources,
        "dataset_root": dataset_root,
        "source": source,
        "identity": identity,
        "policy": policy,
        "preprocessor": preprocessor,
        "original": original,
        "recovery": recovery,
        "episodes": episodes,
        "grasps": grasps,
        "releases": releases,
        "rng": rng,
        "original_fps": original_fps,
        "chunk_size": len(indices),
    }
