import unittest

from act_recovery.evaluation.metrics import paired_counts, recovery_summary


class MetricTests(unittest.TestCase):
    def test_paired_difference_uses_shared_starts(self):
        result = paired_counts([True, True, False, False], [True, False, True, False])
        self.assertEqual(
            result,
            {
                "both_success": 1,
                "both_fail": 1,
                "first_only_success": 1,
                "second_only_success": 1,
                "difference_percentage_points": 0.0,
            },
        )

    def test_summary_keeps_initial_and_adapted_results_separate(self):
        cases = [
            {"candidate_id": "a", "failure_type": "missed_grasp", "init_state_id": 15},
            {"candidate_id": "b", "failure_type": "object_dropped", "init_state_id": 15},
        ]
        results = {
            "a": {
                "initial": {"success": True},
                "U": {"success": False},
                "F_ref": {"success": True},
                "Random": {"success": True},
            },
            "b": {
                "initial": {"success": True},
                "U": {"success": True},
                "F_ref": {"success": True},
                "Random": {"success": False},
            },
        }
        summary = recovery_summary(cases, results, 2)
        self.assertEqual(summary["models"]["initial"]["n_success"], 2)
        self.assertEqual(summary["models"]["U"]["n_success"], 1)
        self.assertEqual(summary["F_ref_minus_U"]["difference_percentage_points"], 50.0)
