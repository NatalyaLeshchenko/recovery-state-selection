"""Запись наблюдений, действий, результата и GIF экспертной попытки."""

from __future__ import annotations

import json

import numpy as np


def save_attempt(output, records, metadata, make_gif):
    
    arrays = {}
    for key, values in records["observations"].items():
        arrays[key] = np.stack(values)
    if records["actions"]:
        arrays.update(
            action=np.stack(records["actions"]),
            reward=np.asarray(records["rewards"], dtype=np.float32),
            phase=np.asarray(records["phases"]),
        )
        filename = "demonstration.npz" if records["success"] else "failed_attempt.npz"
        np.savez_compressed(output / filename, **arrays)
    metadata.update(
        success=records["success"],
        accepted=records["success"],
        n_frames=len(records["actions"]),
        episode_ended=records["ended"],
        bowl_rise_m=records.get("bowl_rise_m"),
    )
    (output / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output / "expert_geometry.json").write_text(json.dumps(records["geometry"], indent=2) + "\n")
    if make_gif and records["actions"]:
        try:
            from PIL import Image

            images = next(
                (
                    values
                    for key, values in records["observations"].items()
                    if "/pixels/" in key and np.asarray(values[0]).shape == (256, 256, 3)
                )
            )
            frames = []
            for frame in images:
                frames.append(Image.fromarray(np.flip(frame, axis=(0, 1)).copy().astype(np.uint8)))
            frames[0].save(
                output / "correction.gif", save_all=True, append_images=frames[1:], duration=50, loop=0
            )
        except Exception as error:
            print("GIF write failed:", error, flush=True)
