"""Расхождение предсказаний и правила U, F_ref и Random."""

from __future__ import annotations

import random
from collections import Counter

import numpy as np

from act_recovery.settings import FAILURE_TYPES

#насколько разное модели предложили: #3+3+1
def disagreement(predictions): #[0, 0, 0, 0, 0, 0, 0]; [0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3]; [0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6]

    value = np.asarray(predictions, dtype=np.float64)
    mean = value.mean(axis=0, keepdims=True) #делаем одну среднюю команду из 3 [0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3]
    return np.square(value - mean).mean(axis=(0, 2, 3)) #как в DaGer
#U: 0,06



#for F_ref
def balanced_quotas(counts, budget, seed):

    
    rng = random.Random(seed) 
    quotas = {}
    for kind in sorted(counts):
        quotas[kind] = 0

    for _ in range(budget): # counts = {"missed_grasp": 250, "object_dropped": 4}; budget = 6 -> quotas = {"missed_grasp": 3, "object_dropped": 3}
        available = []
        for kind in sorted(counts):
            if quotas[kind] < counts[kind]:
                available.append(kind)
        minimum = min((quotas[kind] for kind in available))
        tied = []
        for kind in available:
            if quotas[kind] == minimum:
                tied.append(kind)
        quotas[rng.choice(tied)] += 1
    return quotas

#какие состояния запросить 
def query_plan(rows, budget, seed): # {"candidate_id": "A", "failure_type": "missed_grasp",   "U": 0.06},

    ids = []
    by_id = {}
    for row in rows:
        ids.append(row["candidate_id"])
        by_id[row["candidate_id"]] = row #ids = ["A", "B", "C", "D"]

    #Random
    rng = random.Random(seed)
    base = sorted(ids)
    random_order = rng.sample(base, len(base)) #random_order = ["D", "B", "A", "C"]

    #U
    tie_order = rng.sample(base, len(base)) #if одинаковые U
    ranked = []

    for row in rows:
        candidate = row["candidate_id"]
        ranked.append((-row["U"], tie_order.index(candidate), candidate)) #(-0.06, 3, "A") U, tie, id (особенность sort)

    ranked.sort()  
    uncertainty_order = []#uncertainty_order = ["A", "B", "C", "D"] {−0,06 < −0,04 < −0,02 < −0,01}
    for value, priority, candidate in ranked:
        uncertainty_order.append(candidate)

    #F_ref
    groups = {}
    counts = {}

    for kind in FAILURE_TYPES:
        group = []
        for row in rows:
            if row["failure_type"] == kind:
                group.append(row["candidate_id"])
        groups[kind] = sorted(group) #groups["missed_grasp"]  # ["A", "B"]
        counts[kind] = len(group) #queues["missed_grasp"] = ["B", "A"] ; queues["object_dropped"] = ["C", "D"]; balanced = ["B", "C"]
    quotas = balanced_quotas(counts, budget, seed)


    queues = {}
    balanced = []
    for kind in FAILURE_TYPES:
        queues[kind] = rng.sample(groups[kind], len(groups[kind]))
        balanced.extend(queues[kind][: quotas[kind]])


    plan = {
        "U": {"selected_candidates": uncertainty_order[:budget], "query_order": uncertainty_order},
        "F_ref": {"selected_candidates": balanced, "quotas": quotas, "query_orders_by_type": queues},
        "Random": {"selected_candidates": random_order[:budget], "query_order": random_order}, #U	["A", "B"] -> selected_types = ["missed_grasp", "missed_grasp"]
    }
    for condition in plan.values():
        selected_types = []
        for candidate in condition["selected_candidates"]:
            selected_types.append(by_id[candidate]["failure_type"])
        condition["failure_counts"] = dict(Counter(selected_types))
    return plan
