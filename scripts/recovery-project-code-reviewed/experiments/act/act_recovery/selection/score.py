"""Предсказания трёх ACT для одинаковых сохранённых наблюдений."""

from __future__ import annotations

import gc
import time
from pathlib import Path

import numpy as np

from act_recovery.files import file_hash, new_folder, write_json, write_pointer
from act_recovery.observations import compare_normalization, policy_observation
from act_recovery.pool.records import load_ensemble, load_pool
from act_recovery.selection.methods import disagreement
from act_recovery.settings import SCORE_POINTER


def score(args, root):
    
    pool, protocol = load_pool(root, args.pool)
    import torch
    from lerobot.policies.act.modeling_act import ACTPolicy
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.processor.env_processor import LiberoProcessorStep

   
    ensemble_folder, ensemble = load_ensemble(root, args.ensemble or protocol["ensemble_folder"])

    
    paths = []
    for model in ensemble["models"]:
        paths.append(Path(model["identity"]["path"])) #Path("/.../initial/pretrained_model")
    shared_stats = compare_normalization(paths, torch)
    output = new_folder(root, "scores")
    started = time.perf_counter()


    record = {
        "status": "running",
        "pool": str(pool), #from where
        "pool_protocol_sha256": file_hash(pool / "protocol.json"),
        "ensemble": str(ensemble_folder),
        "ensemble_sha256": protocol["ensemble_sha256"],
        "members": ensemble["models"], #какая ACT
        "normalization": shared_stats, #общ шкала
        "score_horizon": 1,
        "score_definition": "Mean squared ensemble disagreement in shared D0-normalized action space, first action only",
        "scope": "ACT disagreement proxy; not INSIGHT, help probability, or demonstrated epistemic uncertainty",
        "models_loaded_sequentially": True,
        "expert_actions_used": False,
        "cases": [],
    }
    write_json(output / "scores.json", record)


    #модель обрабатывает
    predictions = []

    for member, folder in zip(ensemble["models"], paths):
        print(f"\nSCORE nominal ACT seed={member['seed']}: {len(protocol['cases'])} states", flush=True)
        torch.manual_seed(0)

        policy = ACTPolicy.from_pretrained(folder, local_files_only=True, strict=True)
        policy.config.n_action_steps = 1
        policy.to("cuda").eval()# на GPU
        preprocessor, postprocessor = make_pre_post_processors(policy.config, pretrained_path=folder) 
        processor = LiberoProcessorStep() #от либеро : 9 ->7
        values = []
        with torch.inference_mode():
            for case in protocol["cases"]:
                path = pool / case["candidate_id"] / "start.npz"
                with np.load(path, allow_pickle=False) as saved:
                    batch = policy_observation(saved, processor, torch)
                batch = preprocessor(batch)
                for key in batch:
                    if torch.is_tensor(batch[key]):
                        batch[key] = batch[key].to("cuda")
                policy.reset()
                normalized = policy.predict_action_chunk(batch) #(1, k, 7)
                values.append(normalized[0, :1].detach().cpu().numpy().copy())
                print(f"  {case['candidate_id']}", flush=True)
        predictions.append(np.stack(values))

        del policy, preprocessor, postprocessor, processor, batch, normalized #clean
        gc.collect()
        torch.cuda.empty_cache()

    predictions = np.stack(predictions) #(3, 4, 1, 7) (models, 4 candidates,1 comand, 7 component)
    scores = disagreement(predictions) #[0.06, 0.04, 0.02, 0.01] -> [A. B. C. D]
    candidate_names = []
    for case in protocol["cases"]:
        candidate_names.append(case["candidate_id"]) #["A", "B", "C", "D"]
    
    np.savez_compressed(
        output / "predictions.npz", normalized_actions=predictions, candidate_ids=np.asarray(candidate_names)
    )
    for case, value in zip(protocol["cases"], scores):
        record["cases"].append(
            {
                "candidate_id": case["candidate_id"],
                "init_state_id": case["init_state_id"],
                "failure_type": case["failure_type"],
                "U": float(value),
                "snapshot_sha256": case["snapshot_sha256"],
            }
        )
    record.update(
        status="complete",
        scoring_seconds=time.perf_counter() - started,
        predictions_sha256=file_hash(output / "predictions.npz"),
    )
    load_pool(root, pool)


    if file_hash(pool / "protocol.json") != record["pool_protocol_sha256"]:
        raise RuntimeError("Pool protocol изменился во время scoring")
    write_json(output / "scores.json", record)
    write_pointer(root, SCORE_POINTER, output)
    print("\nUNCERTAINTY SCORES READY")

    def uncertainty_order(row):
        """больше U — ближе к началу списка."""
        return -row["U"]

    for row in sorted(record["cases"], key=uncertainty_order):
        print(f"{row['candidate_id']}: U={row['U']:.6g}")
    print("Output:", output)
