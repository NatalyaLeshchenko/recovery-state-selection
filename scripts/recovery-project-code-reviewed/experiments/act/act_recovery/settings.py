"""Постоянные эксперимента: задача, типы ошибок, тестовые старты и имена файлов."""

from __future__ import annotations

from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parent.parent
FAILURE_TYPES = ("missed_grasp", "object_dropped")
RESERVED_TEST_IDS = (15, 16, 17, 18, 19, 45, 46, 47, 48, 49)
POOL_POINTER = "selection_pool_path.txt"
SCORE_POINTER = "selection_scores_path.txt"
INSTRUCTION = "pick up the black bowl between the plate and the ramekin and place it on the plate"
RAW_SHAPES = {
    "observation/pixels/image": (1, 256, 256, 3),
    "observation/pixels/image2": (1, 256, 256, 3),
    "observation/robot_state/eef/pos": (1, 3),
    "observation/robot_state/eef/quat": (1, 4),
    "observation/robot_state/gripper/qpos": (1, 2),
}
ENSEMBLE_SCOPE = "ACT ensemble-disagreement proxy; not INSIGHT or VLA validation"
POLICY_IGNORE = {"device", "pretrained_path", "push_to_hub", "repo_id"}
REQUIRED = (
    "config.json",
    "model.safetensors",
    "train_config.json",
    "policy_preprocessor.json",
    "policy_postprocessor.json",
)
METHODS = ("U", "F_ref", "Random")
COLLECTION_POINTER = "selection_expert_collection_path.txt"
DATASETS_POINTER = "selection_datasets_path.txt"
MODELS_POINTER = "selection_models_path.txt"
SOURCE_FILE = "selection_sources.json"
WEIGHTS = (0.5, 0.2, 0.15, 0.15) #для лосс
RECOVERY_SUPERVISION = 10
VARIANTS = ("initial", "U", "F_ref", "Random")
TEST_POINTER = "selection_test_pool_path.txt"
RECOVERY_POINTER = "selection_recovery_evaluation_path.txt"
CLEAN_POINTER = "selection_clean_evaluation_path.txt"
TEST_FILE = "test_protocol.json"
EVALUATION_SCOPE = "Independent induced-error ACT test; held out from recovery learning and selector tuning"
INPUT_KEYS = ("observation.images.image", "observation.images.image2", "observation.state")
