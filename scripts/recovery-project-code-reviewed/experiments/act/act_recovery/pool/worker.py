"""Запустить исходную ACT из расстановки №20, создать промах захвата, 
сохранить состояние после него и проверить, что это состояние можно воспроизвести.."""

from __future__ import annotations

import os
import subprocess
import sys
import traceback
from pathlib import Path

import numpy as np

from act_recovery.files import read_json, write_json
from act_recovery.observations import skipped_result
from act_recovery.pool.replay import check_restoration
from act_recovery.pool.rollout import collect_candidate
from act_recovery.scene import SkipCase
from act_recovery.settings import CODE_ROOT


def run_worker(root, request):

    output = Path(request["output"])
    output.mkdir(parents=True, exist_ok=False)
    request_path = output / "request.json"
    write_json(request_path, request)
    command = [
        sys.executable,
        str((CODE_ROOT / "selection_pilot.py").resolve()),
        "_worker",
        "--request",
        str(request_path),
    ]
    with (output / "worker.log").open("w", encoding="utf-8") as log:
        completed = subprocess.run(command, cwd=root, stdout=log, stderr=subprocess.STDOUT)
    if completed.returncode != 0:
        raise RuntimeError(f"Сбор остановился. Причина в файле {output / 'worker.log'}")
    return read_json(output / "worker_result.json")


def worker(path):

    request = read_json(path)
    os.chdir(request["project_root"])
    output = Path(request["output"])
    data = {"trace": [], "event_rows": [], "frames": []}
    try:
        result = collect_candidate(request, data) #main

        if result["status"] == "accepted":
            result["restore_check"] = check_restoration(request, result) #можно ли в новой среде повторить сохранённые действия и получить то же recovery-состояние
            write_json(output / "case.json", result)
        write_json(output / "worker_result.json", result)
    except SkipCase as error:
        write_json(output / "worker_result.json", skipped_result(request, str(error)))
    except Exception as error:
        write_json(
            output / "worker_result.json",
            {"status": "error", "error": str(error), "traceback": traceback.format_exc()},
        )
        raise
    finally:
        write_json(
            output / "event_trace.json", {"nominal": data["trace"], "intervention": data["event_rows"]}
        )
        if data["frames"]:
            from PIL import Image

            images = []
            for frame in data["frames"]:
                images.append(Image.fromarray(np.ascontiguousarray(frame[::-1, ::-1])))
            images[0].save(
                output / "intervention.gif", save_all=True, append_images=images[1:], duration=50, loop=0
            )
