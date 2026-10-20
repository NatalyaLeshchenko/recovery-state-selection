import unittest

from act_recovery.pool.events import check_drop_event, check_miss_event


class EventTests(unittest.TestCase):
    def setUp(self):
        self.config = {
            "miss_max_bowl_rise_m": 0.015,
            "miss_min_empty_lift_m": 0.035,
            "miss_max_bowl_shift_m": 0.03,
            "drop_threshold_m": 0.03,
        }
        self.before = {"bowl_pos": [0.0, 0.0, 1.0], "eef_pos": [0.0, 0.0, 1.0]}
        self.after = {"bowl_pos": [0.0, 0.0, 1.0], "eef_pos": [0.0, 0.0, 1.04]}

    def test_empty_lift_with_static_bowl_is_a_missed_grasp(self):
        rows = [{"geometry": self.after, "contacts": {"bilateral": False}}]
        accepted, evidence = check_miss_event(self.before, self.before, rows, self.config)
        self.assertTrue(accepted)
        self.assertAlmostEqual(evidence["empty_eef_lift_m"], 0.04)

    def test_bilateral_hold_rejects_the_missed_grasp_label(self):
        rows = [{"geometry": self.after, "contacts": {"bilateral": True}}]
        self.assertFalse(check_miss_event(self.before, self.before, rows, self.config)[0])

    def test_a_rising_bowl_rejects_the_missed_grasp_label(self):
        self.after["bowl_pos"][2] = 1.02
        rows = [{"geometry": self.after, "contacts": {"bilateral": False}}]
        self.assertFalse(check_miss_event(self.before, self.before, rows, self.config)[0])

    def test_drop_requires_a_prior_hold_and_lost_contact(self):
        self.after["bowl_pos"][2] = 0.96
        rows = [{"geometry": self.after, "contacts": {"bilateral": False}}]
        self.assertTrue(check_drop_event(self.before, rows, True, self.config)[0])
        self.assertFalse(check_drop_event(self.before, rows, False, self.config)[0])
        rows[0]["contacts"]["bilateral"] = True
        self.assertFalse(check_drop_event(self.before, rows, True, self.config)[0])
