import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class CommandTests(unittest.TestCase):
    def test_public_entry_points_work_from_another_directory(self):
        code = Path(__file__).resolve().parent.parent
        commands = {
            "train_uncertainty_ensemble.py": [[], ["inspect"], ["train"]],
            "selection_pilot.py": [[], ["collect"], ["score"], ["select"]],
            "selection_experiment.py": [[], ["expert"], ["prepare"], ["train"]],
            "evaluate_selection.py": [[], ["collect"], ["recovery"], ["clean"]],
        }
        with tempfile.TemporaryDirectory() as folder:
            for filename, options in commands.items():
                for arguments in options:
                    command = [sys.executable, str(code / filename), *arguments, "--help"]
                    result = subprocess.run(command, cwd=folder, capture_output=True, text=True)
                    self.assertEqual(result.returncode, 0, result.stderr)
                    self.assertIn("usage:", result.stdout)
