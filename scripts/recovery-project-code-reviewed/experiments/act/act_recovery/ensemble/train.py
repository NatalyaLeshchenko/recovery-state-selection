"""Последовательное обучение двух ACT на обычных демонстрациях."""

from __future__ import annotations

import os
import subprocess
from datetime import datetime

from act_recovery.ensemble.config import member_config, runtime_report, training_command
from act_recovery.files import file_hash, read_json, write_json
from act_recovery.models import identity, policy_settings, verify_identity
from act_recovery.settings import ENSEMBLE_SCOPE


def run_members(ctx, smoke):
   
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA недоступна; обучение не запущено.")
    
    root = ctx["root"]
    verify_identity(ctx["identity"])
    output = (
        root
        / "pilot_runs/uncertainty"
        / datetime.now().strftime(("smoke" if smoke else "ensemble") + "_%Y%m%d_%H%M%S_%f")
    )
    output.mkdir(parents=True, exist_ok=False)

    initial_seed = ctx["train"]["seed"]

    
    seeds = [initial_seed + 1] if smoke else [initial_seed + 1, initial_seed + 2]

    report = {
        "status": "running",
        "complete": False,
        "smoke_only": smoke,
        "scope": ENSEMBLE_SCOPE,
        "runtime": runtime_report(ctx),
        "D0": ctx["d0"],
        "manifest_sha256": ctx["manifest_sha256"],
        "models": [
            {"seed": initial_seed, "identity": ctx["identity"], "role": "fixed_initial_actor_and_member0"}
        ],
        "attempts": [],
        "execution": "One training process at a time; no simultaneous models on CUDA",
    }

    #сохранили настройки
    write_json(output / "ensemble.json", report)
    print("Output:", output, flush=True)
    child_env = dict(
        os.environ,
        WANDB_MODE="disabled",
        HF_HUB_DISABLE_TELEMETRY="1",
        OMP_NUM_THREADS="4",
        MKL_NUM_THREADS="4",
    )
    try:
        for seed in seeds:
            target = output / f"seed_{seed}"
            cfg = member_config(ctx["train"], target, seed, smoke=smoke)
            config_path = output / f"seed_{seed}_train_config.json"
            write_json(config_path, cfg)

            command = training_command(config_path)

            attempt = {"seed": seed, "command": command, "config": str(config_path), "complete": False}
            report["attempts"].append(attempt)

            write_json(output / "ensemble.json", report)

            print(
                f"\nACT seed={seed}: D0, updates={cfg['steps']}, batch={cfg['batch_size']}, workers=0",
                flush=True,
            )

            #обучение через Леробот
            subprocess.run(command, cwd=root, env=child_env, check=True)

            trained = (target / "checkpoints/last/pretrained_model").resolve()
            record = identity(trained)
            if policy_settings(read_json(trained / "config.json")) != policy_settings(ctx["policy"]):
                raise RuntimeError("Архитектура trained member отличается от исходной ACT.")
            saved = read_json(trained / "train_config.json")
            if (
                saved.get("seed") != seed
                or saved.get("steps") != cfg["steps"]
                or saved.get("batch_size") != cfg["batch_size"]
                or (saved.get("dataset") != cfg["dataset"])
            ):
                raise RuntimeError("Сохранённый training config отличается от запланированного.")
            if saved["policy"].get("pretrained_path") or saved.get("resume") is True:
                raise RuntimeError(
                    "Member загрузил готовые action-model weights вместо fresh initialization."
                )
            existing_hashes = []
            for m in report["models"]:
                existing_hashes.append(m["identity"]["sha256"]["model.safetensors"])

            if record["sha256"]["model.safetensors"] in existing_hashes:
                raise RuntimeError("Получены идентичные weights; независимость members не подтверждена.")
            
            attempt["complete"] = True
            report["models"].append({"seed": seed, "identity": record, "role": "uncertainty_member"})
            write_json(output / "ensemble.json", report)
        verify_identity(ctx["identity"])

        if file_hash(root / "pilot_runs/manifest.json") != ctx["manifest_sha256"]:
            raise RuntimeError("D0 manifest изменился во время обучения.")
        
        report.update(status="complete", complete=True)
        write_json(output / "ensemble.json", report)
        
        if not smoke:
            write_pointer = root / "pilot_runs/uncertainty_ensemble_path.txt"
            temporary = write_pointer.with_suffix(".tmp")
            temporary.write_text(str(output) + "\n", encoding="utf-8")
            temporary.replace(write_pointer)
            print("\nENSEMBLE MEMBERS READY: 3", flush=True)
        else:
            print("\nSMOKE COMPLETE", flush=True)
        print("Output:", output, flush=True)
    
    except BaseException as error:
        report.update(
            status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed", error=str(error)
        )
        write_json(output / "ensemble.json", report)
        raise
