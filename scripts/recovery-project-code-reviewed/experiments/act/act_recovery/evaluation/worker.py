"""Одна оценка ACT после воспроизведения сохранённого recovery-старта."""

import os
import traceback
from pathlib import Path

from act_recovery.evaluation.lerobot import official_evaluation
from act_recovery.evaluation.replay import EvaluationHooks
from act_recovery.files import read_json, write_json


def recovery_worker(request_path):
    """Вход: JSON с моделью, сохранённым стартом и лимитом шагов.
    Выход: worker_result.json, видео и журнал выполненных действий.
    """
    from gymnasium.vector import SyncVectorEnv

    request = read_json(request_path)
    os.chdir(request["project_root"])
    output = Path(request["output"])
    hooks = None
    try:
        hooks = EvaluationHooks(request, SyncVectorEnv.reset, SyncVectorEnv.step)
        official_evaluation(request, hooks)
        info = read_json(output / "evaluation/eval_info.json")["overall"]
        if info["n_episodes"] != 1 or hooks.restore is None or (not hooks.rows):
            raise RuntimeError("Expected one complete recovery evaluation episode.")
        result = {
            "status": "evaluated",
            "success": bool(info["n_success"] == 1),
            "steps": len(hooks.rows),
            "videos": info.get("video_paths", []),
            "model": request["model_path"],
            "variant": request["variant"],
            "init_state_id": request["init_state_id"],
            "restore": hooks.restore,
        }
        write_json(output / "worker_result.json", result)
    except BaseException as error:
        write_json(
            output / "worker_result.json",
            {"status": "error", "error": str(error), "traceback": traceback.format_exc()},
        )
        raise
    finally:
        if hooks is not None and hooks.rows:
            write_json(
                output / "rollout_trace.json",
                {
                    "steps": hooks.rows,
                    "variant": request["variant"],
                    "case_dir": request["case_dir"],
                    "n_action_steps": request["protocol"]["n_action_steps"],
                    "geometry_note": "Height/contact diagnostics alone do not establish stable grasp; success is the LIBERO goal outcome.",
                },
            )
