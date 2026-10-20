"""Преобразование выбранных демонстраций в три датасета LeRobot."""

from __future__ import annotations

from act_recovery.expert.records import load_collection
from act_recovery.files import file_hash, new_folder, write_json, write_pointer
from act_recovery.settings import DATASETS_POINTER, METHODS, SOURCE_FILE


def prepare(args, root):
    
    import torch
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.processor.env_processor import LiberoProcessorStep

    from act_recovery.expert.records import load_successful_demo, validate_ranges
    from act_recovery.training.frames import checked_frame, processed_correction_frame

    collection, info = load_collection(root, args.collection)
    output = new_folder(root, "datasets_selection")
    summary = {
        "status": "running",
        "collection": str(collection),
        "collection_sha256": file_hash(collection / "collection.json"),
        "target_budget": info["target_budget"],
        "datasets": {},
    }
    write_json(output / "datasets.json", summary)
    features = {
        "observation.images.image": {
            "dtype": "image",
            "shape": (256, 256, 3),
            "names": ["height", "width", "channel"],
        },
        "observation.images.image2": {
            "dtype": "image",
            "shape": (256, 256, 3),
            "names": ["height", "width", "channel"],
        },
        "observation.state": {"dtype": "float32", "shape": (8,), "names": ["state"]},
        "action": {"dtype": "float32", "shape": (7,), "names": ["actions"]},
    }
    try:
        for method in METHODS:
            selected = info["conditions"][method]["selected_successful_candidates"]
            folder = output / method
            repo_id = f"local/selection-{method.lower().replace('_', '-')}"
            writer = LeRobotDataset.create(repo_id, fps=20, features=features, root=folder, use_videos=False)
            processor = LiberoProcessorStep()
            episodes, grasp_indices, release_indices, attempts = ([], [], [], [])
            count = 0
            for candidate in selected:
                attempt = info["unique_attempts"][candidate]
                metadata, data = load_successful_demo(attempt, info["instruction"])
                if (
                    metadata.get("candidate_id") != candidate
                    or metadata["source_snapshot_sha256"] != attempt["source_snapshot_sha256"]
                    or metadata["failure_type"] != attempt["failure_type"]
                ):
                    raise RuntimeError("Expert demo не соответствует selected candidate.")
                start, closed = (count, False)
                for index in range(len(data["action"])):
                    frame = checked_frame(
                        processed_correction_frame(data, index, processor, torch), info["instruction"]
                    )
                    grip = float(frame["action"][-1])
                    if str(data["phase"][index]) in ("approach_grasp", "close_gripper"):
                        grasp_indices.append(count)
                    if grip < 0 and closed:
                        release_indices.append(count)
                    closed = closed or grip > 0
                    writer.add_frame(frame)
                    count += 1
                writer.save_episode()
                episodes.append(
                    {
                        "episode_index": len(episodes),
                        "start": start,
                        "stop": count,
                        "candidate_id": candidate,
                        "source_kind": "selected_expert_recovery",
                        "init_state_id": attempt["init_state_id"],
                        "failure_type": attempt["failure_type"],
                        "init_definition_sha256": attempt["init_definition_sha256"],
                        "source_folder": attempt["folder"],
                    }
                )
                attempts.append(attempt)
                del metadata, data
            writer.finalize()
            validate_ranges(episodes, count)
            if not grasp_indices or not release_indices:
                raise RuntimeError(
                    f"В данных {method} нет grasp/release examples для общего sampling recipe."
                )
            init_state_ids = set()
            for row in episodes:
                init_state_ids.add(row["init_state_id"])
            init_definition_hashes = set()
            for row in episodes:
                init_definition_hashes.add(row["init_definition_sha256"])
            sources = {
                "status": "complete",
                "repo_id": repo_id,
                "fps": 20,
                "n_frames": count,
                "n_episodes": len(episodes),
                "selector": method,
                "demonstration_budget": info["target_budget"],
                "episodes": episodes,
                "grasp_focus_indices": grasp_indices,
                "release_focus_indices": release_indices,
                "initial_policy": info["source_initial_model"],
                "collection": str(collection),
                "collection_sha256": summary["collection_sha256"],
                "new_demonstrations": attempts,
                "used_init_state_ids": sorted(init_state_ids),
                "used_init_definition_sha256": sorted(init_definition_hashes),
                "historical_recovery_demonstrations_included": False,
                "observation_timing": "before_action",
                "image_processing": "Raw RGB rotated once by LiberoProcessorStep; unchanged D0 policy normalization",
            }
            write_json(folder / SOURCE_FILE, sources)
            check = LeRobotDataset(repo_id, root=folder)
            if len(check) != count or len(episodes) != info["target_budget"]:
                raise RuntimeError("Число сохранённых frames/episodes не соответствует budget.")
            for episode in episodes:
                for index in (episode["start"], episode["stop"] - 1):
                    sample = check[index]
                    checked_frame(sample, info["instruction"])
                    if int(sample["episode_index"].item()) != episode["episode_index"]:
                        raise RuntimeError("Граница demonstration episodes не сохранилась.")
            del check, writer, processor
            summary["datasets"][method] = {
                "root": str(folder),
                "repo_id": repo_id,
                "n_frames": count,
                "n_episodes": len(episodes),
                "sources_sha256": file_hash(folder / SOURCE_FILE),
            }
            write_json(output / "datasets.json", summary)
            write_pointer(root, f"selection_{method}_dataset_path.txt", folder)
            print(f"{method}: {len(episodes)} demonstrations, {count} frames; {folder}", flush=True)
        summary["status"] = "complete"
        write_json(output / "datasets.json", summary)
        write_pointer(root, DATASETS_POINTER, output)
    except BaseException as error:
        summary.update(status="interrupted", error=f"{type(error).__name__}: {error}")
        write_json(output / "datasets.json", summary)
        raise
    print("\nSELECTED DATASETS READY")
