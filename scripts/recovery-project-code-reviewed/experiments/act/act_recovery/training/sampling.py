"""Выбор обычного кадра, recovery-кадра, захвата и отпускания."""

from __future__ import annotations

from act_recovery.expert.records import validate_ranges


def focus_groups(episodes, indices):
    
    validate_ranges(episodes, episodes[-1]["stop"] if episodes else 0)
    sorted_indices = sorted(set((int(i) for i in indices)))
    if not sorted_indices or sorted_indices[0] < 0 or sorted_indices[-1] >= episodes[-1]["stop"]:
        raise RuntimeError("Неверные phase-focus indices.")
    groups = []
    for episode in episodes:
        group = []
        for index in sorted_indices:
            if episode["start"] <= index < episode["stop"]:
                group.append(index)
        if group:
            groups.append(group)
    return groups


def sample_frames(rng, episodes, grasps, releases):
    
    episode = episodes[int(rng.integers(len(episodes)))]
    uniform = int(rng.integers(episode["start"], episode["stop"]))
    grasp_group = grasps[int(rng.integers(len(grasps)))]
    release_group = releases[int(rng.integers(len(releases)))]
    return (uniform, int(rng.choice(grasp_group)), int(rng.choice(release_group)))
