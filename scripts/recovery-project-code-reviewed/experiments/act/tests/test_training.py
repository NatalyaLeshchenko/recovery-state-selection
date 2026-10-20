import unittest

import numpy as np
from act_recovery.training.loss import recovery_padding
from act_recovery.training.sampling import focus_groups, sample_frames


class TrainingTests(unittest.TestCase):
    def setUp(self):
        self.episodes = [
            {"episode_index": 0, "start": 0, "stop": 8},
            {"episode_index": 1, "start": 8, "stop": 20},
        ]

    def test_recovery_mask_does_not_shorten_nominal_supervision(self):
        mask = np.zeros((4, 20), dtype=bool)
        result = recovery_padding(mask, 10)
        self.assertFalse(result[0].any())
        self.assertFalse(result[1:, :10].any())
        self.assertTrue(result[1:, 10:].all())
        self.assertFalse(mask.any())

    def test_phase_frames_are_grouped_by_episode(self):
        self.assertEqual(focus_groups(self.episodes, [0, 2, 8, 10]), [[0, 2], [8, 10]])

    def test_sampling_returns_valid_frames_of_each_phase(self):
        rng = np.random.default_rng(0)
        for _ in range(50):
            uniform, grasp, release = sample_frames(rng, self.episodes, [[0, 2], [8, 10]], [[6, 7], [15, 18]])
            self.assertIn(uniform, range(20))
            self.assertIn(grasp, [0, 2, 8, 10])
            self.assertIn(release, [6, 7, 15, 18])
