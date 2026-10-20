"""Запуск отдельных этапов в том же Python с сохранением журнала."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from act_recovery.evaluation.models import require
from act_recovery.files import read_json, write_json
from act_recovery.settings import CODE_ROOT

#
def run_child(root, command, request, output, live=False):
    
    output.mkdir(parents=True, exist_ok=False) #создать папку результата
    request_path = output / "request.json"
    write_json(request_path, request) #параметры в request.json
    call = [
        sys.executable, #путь, где сейчас наш код
        str((CODE_ROOT / "selection_experiment.py").resolve()), #путь к selection_experiment.py
        command, #command = "_train-worker"
        "--request", #файл с параметрами
        str(request_path),
    ]
    env = dict(
        os.environ,
        OMP_NUM_THREADS="4",
        MKL_NUM_THREADS="4",
        WANDB_MODE="disabled",
        HF_HUB_DISABLE_TELEMETRY="1",
    )

    if live:
        completed = subprocess.run(call, cwd=root, env=env)
    else:
        with (output / "worker.log").open("w", encoding="utf-8") as log:
            completed = subprocess.run(call, cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT)
    result_path = output / "worker_result.json"
    if completed.returncode != 0 or not result_path.is_file():
        if not live:
            print(
                "\n".join((output / "worker.log").read_text(errors="replace").splitlines()[-45:]), flush=True
            )
        raise RuntimeError(f"Этап остановлен; данные и лог: {output}")
    result = read_json(result_path)
    if result.get("status") != "complete":
        raise RuntimeError(f"Незавершённый worker: {result}")
    return result


def run_clean_child(root, request):
    
    output = Path(request["output"])
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "request.json", request)
    env = dict(
        os.environ,
        OMP_NUM_THREADS="4",
        MKL_NUM_THREADS="4",
        WANDB_MODE="disabled",
        HF_HUB_DISABLE_TELEMETRY="1",
    )
    with (output / "worker.log").open("w", encoding="utf-8") as log:
        
        
        #Выполнить команду из списка call и дождаться её завершения!
        completed = subprocess.run(
            [
                sys.executable,
                str((CODE_ROOT / "evaluate_selection.py").resolve()),
                "_clean-worker",
                "--request",
                str(output / "request.json"),
            ],
            cwd=root,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
    if completed.returncode or not (output / "worker_result.json").is_file():
        print("\n".join((output / "worker.log").read_text(errors="replace").splitlines()[-45:]), flush=True)
        raise RuntimeError(f"Clean evaluation остановилась. Лог: {output / 'worker.log'}")
    result = read_json(output / "worker_result.json")
    require(
        result.get("status") == "evaluated" and result["variant"] == request["variant"],
        "Некорректный clean result.",
    )
    return result


def run_recovery_worker(root, request):
    
    output = Path(request["output"])
    output.mkdir(parents=True, exist_ok=False)
    request_path = output / "request.json"
    write_json(request_path, request)
    with (output / "worker.log").open("w", encoding="utf-8") as log:
        completed = subprocess.run(
            [
                sys.executable,
                str((CODE_ROOT / "evaluate_selection.py").resolve()),
                "_recovery-worker",
                "--request",
                str(request_path),
            ],
            cwd=root,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
    result_path = output / "worker_result.json"
    if completed.returncode != 0 or not result_path.is_file():
        lines = (output / "worker.log").read_text(errors="replace").splitlines()
        print("\n".join(lines[-45:]), flush=True)
        raise RuntimeError(f"Проверка остановлена; полный лог: {output / 'worker.log'}")
    result = json.loads(result_path.read_text())
    if result.get("status") not in ("accepted", "skipped", "evaluated"):
        raise RuntimeError(f"Неожиданный результат worker: {result}")
    return result
