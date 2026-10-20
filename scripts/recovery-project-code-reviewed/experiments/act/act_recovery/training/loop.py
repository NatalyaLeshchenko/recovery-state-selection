"""Обновления весов ACT на обычных и выбранных recovery-демонстрациях."""

import csv
import time
from pathlib import Path

from act_recovery.files import file_hash, write_json, write_pointer
from act_recovery.models import verify_identity, verify_model
from act_recovery.settings import WEIGHTS
from act_recovery.training.batch import training_batch
from act_recovery.training.checkpoint import save_model
from act_recovery.training.loss import focused_loss
from act_recovery.training.record import log_fields, training_record
from act_recovery.training.setup import prepare_training


def train_one(request):



    import torch

    context = prepare_training(request)
    policy = context["policy"]
    recipe = context["recipe"]
    if list(WEIGHTS) != recipe["weights"]:
        raise RuntimeError("The common sampling weights have changed.")
    output = Path(request["output"]) / "training"
    output.mkdir(parents=True, exist_ok=False)
    run_info = training_record(context)
    write_json(output / "run_config.json", run_info)
    weights = torch.tensor(WEIGHTS, device="cuda", dtype=torch.float32)
    parameters = []
    for parameter in policy.parameters():
        if parameter.requires_grad:
            parameters.append(parameter)


    optimizer = torch.optim.AdamW(parameters, lr=recipe["lr"], weight_decay=0.0001)
    policy.train()

    torch.cuda.reset_peak_memory_stats()
    completed = 0
    started = time.perf_counter()
    try:
        with (output / "losses.csv").open("x", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=log_fields())
            writer.writeheader()
            for step in range(1, recipe["steps"] + 1):

                #
                batch, indices = training_batch(context)
                optimizer.zero_grad(set_to_none=True) #важно - в пайторче накапливаются
                loss, metrics = focused_loss(policy, batch, weights)

               
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), 10.0)
                if not torch.isfinite(norm).item():
                    raise RuntimeError("The training gradients are not finite.")
                optimizer.step()
                completed = step
                value = float(loss.detach().cpu())
                peak = torch.cuda.max_memory_allocated() / 1024**3
                writer.writerow(
                    {
                        "step": step,
                        "loss": value,
                        **metrics,
                        "original_frame": indices[0],
                        "uniform_frame": indices[1],
                        "grasp_frame": indices[2],
                        "release_frame": indices[3],
                        "peak_allocated_gib": peak,
                    }
                )
                if step == 1 or step % 50 == 0 or step == recipe["steps"]:
                    stream.flush()
                    print(f"step={step}/{recipe['steps']} loss={value:.4f} peak={peak:.2f} GiB", flush=True)
        verify_model(context["identity"])
        run_info.update(
            status="complete",
            completed_updates=completed,
            training_wall_seconds=time.perf_counter() - started,
        )
        folder = output / "checkpoints/last/pretrained_model"
        saved = save_model(policy, context["source"], context["identity"], folder, run_info)
        run_info["saved_policy"] = saved
        write_json(output / "run_config.json", run_info)
        write_pointer(context["root"], "selection_" + request["method"] + "_policy_path.txt", folder)
    except BaseException as error:
        run_info.update(
            status="interrupted", completed_updates=completed, error=f"{type(error).__name__}: {error}"
        )
        write_json(output / "run_config.json", run_info)
        raise
    verify_identity(request["initial_policy"])
    return {
        "method": request["method"],
        "identity": saved,
        "training_folder": str(output),
        "run_config_sha256": file_hash(output / "run_config.json"),
        "initial_policy": request["initial_policy"],
        "dataset": request["dataset"],
        "recipe": recipe,
    }
