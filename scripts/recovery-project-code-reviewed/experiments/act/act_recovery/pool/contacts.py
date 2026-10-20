"""Поиск контактов миски с левым и правым пальцами захвата."""

from __future__ import annotations

import numpy as np


def descendant_ids(model, root_id):
   
    parents = np.asarray(model.body_parentid)
    result = {int(root_id)}
    for body in range(len(parents)):
        node = body
        while node != root_id and parents[node] != node:
            node = int(parents[node])
        if node == root_id:
            result.add(body)
    return result


def finger_contacts(vec):
   
    from act_recovery.scene import find_raw

    sim = find_raw(vec).sim #находит миску и два пальца
    names = list(sim.model.body_names)
    roots = {
        """akita_black_bowl_1_main — нужная миска
            gripper0_leftfinger    — левый палец
            gripper0_rightfinger   — правый палец"""
        "bowl": names.index("akita_black_bowl_1_main"), 
        "left": names.index("gripper0_leftfinger"),
        "right": names.index("gripper0_rightfinger"),
    }
    groups = {}
    for key, body in roots.items():
        groups[key] = descendant_ids(sim.model, body)
    geom_bodies = np.asarray(sim.model.geom_bodyid)
    sides = set()
    for index in range(int(sim.data.ncon)):
        contact = sim.data.contact[index] #«палец касается миски»
        bodies = {int(geom_bodies[contact.geom1]), int(geom_bodies[contact.geom2])}
        if not bodies & groups["bowl"]:
            continue
        for side in ("left", "right"):
            if bodies & groups[side]:
                sides.add(side)
    return {"left": "left" in sides, "right": "right" in sides, "bilateral": {"left", "right"} <= sides}
