"""Условия подтверждённого промаха захвата и падения объекта."""

from __future__ import annotations

import numpy as np


def eligible_miss(intended, before, initial, held_ever, cfg):
    
    rise = before["bowl_pos"][2] - initial["bowl_pos"][2]
    distance = float(np.linalg.norm(np.asarray(before["eef_pos"]) - before["bowl_pos"]))
    return (
        not held_ever
        and rise < cfg["miss_max_bowl_rise_m"]
        and (intended[0, -1] >= cfg["close_command_threshold"])
        and (distance <= cfg["miss_trigger_distance_m"])
    )


def check_miss_event(initial, before, event_rows, cfg):
    
    after = event_rows[-1]["geometry"]
    bowl_rise = max((row["geometry"]["bowl_pos"][2] for row in event_rows)) - initial["bowl_pos"][2]
    eef_rise = after["eef_pos"][2] - before["eef_pos"][2]
    moved_xy = float(np.linalg.norm(np.asarray(after["bowl_pos"][:2]) - before["bowl_pos"][:2]))
    no_hold = not any((row["contacts"]["bilateral"] for row in event_rows))
    valid = (
        bowl_rise < cfg["miss_max_bowl_rise_m"]
        and eef_rise >= cfg["miss_min_empty_lift_m"]
        and (moved_xy <= cfg["miss_max_bowl_shift_m"])
        and no_hold
    )
    return (
        valid,
        {
            "max_bowl_rise_m": bowl_rise,
            "empty_eef_lift_m": eef_rise,
            "bowl_shift_xy_m": moved_xy,
            "no_bilateral_hold_during_attempt": no_hold,
        },
    )


def check_drop_event(before, event_rows, held_ever, cfg):
    
    after = event_rows[-1]["geometry"]
    fall = before["bowl_pos"][2] - after["bowl_pos"][2]
    lost_hold = not event_rows[-1]["contacts"]["bilateral"]
    valid = held_ever and fall >= cfg["drop_threshold_m"] and lost_hold
    return (
        valid,
        {
            "fall_m": fall,
            "held_before_intervention": held_ever,
            "bilateral_hold_lost": lost_hold,
            "release_was_forced_before_task_success": True,
        },
    )
