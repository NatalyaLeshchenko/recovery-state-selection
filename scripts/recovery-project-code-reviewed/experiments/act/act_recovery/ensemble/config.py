"""Настройки независимых моделей ансамбля."""

from __future__ import annotations

import importlib.metadata
import shutil
import sys
from copy import deepcopy
from pathlib import Path

from act_recovery.settings import ENSEMBLE_SCOPE

#настройки доп ACT
def member_config(source, output, seed, smoke=False):
    
    if seed == source["seed"]:
        raise RuntimeError("Seed нового member должен отличаться от исходного.")


    #настройки
    result = deepcopy(source)
    result.update(
        seed=seed, 
        output_dir=str(output),
        resume=False, 
        num_workers=0,
        job_name=f"act_uncertainty_seed{seed}", 
        env=None,
        env_eval_freq=0,
    )
    if smoke:
        result["steps"] = min(20, source["steps"]) #на 20, just to check

    result["save_freq"] = result["steps"] #частота обновления чекпоинта
    result["save_checkpoint"] = True
    result["log_freq"] = min(20, result["steps"])
    result["policy"]["device"] = "cuda"
    result["policy"]["pretrained_path"] = None
    result["policy"]["push_to_hub"] = False

    if isinstance(result.get("wandb"), dict):
        result["wandb"]["enable"] = False
    if "save_checkpoint_to_hub" in result:
        result["save_checkpoint_to_hub"] = False
    if "persistent_workers" in result:
        result["persistent_workers"] = False
    if "checkpoint_path" in result:
        result["checkpoint_path"] = None
    return result


def training_command(config_path):
    

    return [sys.executable, "-m", "lerobot.scripts.lerobot_train", f"--config_path={config_path}"]

#собираем тех отчет
def memory_info():
   
    info = {}
    path = Path("/proc/meminfo")
    if path.is_file():
        for line in path.read_text().splitlines():
            name, value = line.split(":", 1)
            if name in ("MemTotal", "MemAvailable", "SwapTotal", "SwapFree"):
                info[name + "_GiB"] = round(int(value.strip().split()[0]) / 1024**2, 2)
    return info


def runtime_report(ctx):
   
    packages = {}
    for name in ("lerobot", "torch", "hf-libero"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = "not installed"
    return {
        "scope": ENSEMBLE_SCOPE,
        "initial_model": str(ctx["initial"]),
        "initial_seed": ctx["train"]["seed"],
        "updates_per_member": ctx["train"]["steps"],
        "batch_size": ctx["train"]["batch_size"],
        "new_num_workers": 0,
        "D0": ctx["d0"],
        "memory": memory_info(),
        "disk_free_GiB": round(shutil.disk_usage(ctx["root"]).free / 1024**3, 2),
        "versions": packages,
    }
