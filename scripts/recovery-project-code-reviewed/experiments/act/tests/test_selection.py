import unittest

import numpy as np
from act_recovery.selection.methods import disagreement, query_plan


class SelectionTests(unittest.TestCase):
    def setUp(self):
        self.rows = []
        for index in range(8):
            kind = "missed_grasp" if index % 2 == 0 else "object_dropped"
            self.rows.append({"candidate_id": str(index), "failure_type": kind, "U": float(index)})

    def test_same_seed_repeats_the_whole_plan(self):
        self.assertEqual(query_plan(self.rows, 2, 0), query_plan(list(reversed(self.rows)), 2, 0))

    def test_highest_uncertainty_is_selected_first(self):
        self.assertEqual(query_plan(self.rows, 2, 0)["U"]["selected_candidates"], ["7", "6"])

    def test_two_requests_cover_both_failure_types(self):
        plan = query_plan(self.rows, 2, 0)["F_ref"]
        self.assertEqual(plan["failure_counts"], {"missed_grasp": 1, "object_dropped": 1})

    def test_selectors_do_not_repeat_candidates(self):
        for method in query_plan(self.rows, 8, 0).values():
            selected = method["selected_candidates"]
            self.assertEqual(len(selected), len(set(selected)))

    def test_identical_predictions_have_zero_disagreement(self):
        predictions = np.ones((3, 8, 1, 7))
        np.testing.assert_array_equal(disagreement(predictions), np.zeros(8))

    def test_disagreement_uses_normalized_actions(self):
        predictions = np.zeros((3, 1, 1, 7))
        predictions[0] = 1
        predictions[2] = -1
        np.testing.assert_allclose(disagreement(predictions), [2 / 3])
