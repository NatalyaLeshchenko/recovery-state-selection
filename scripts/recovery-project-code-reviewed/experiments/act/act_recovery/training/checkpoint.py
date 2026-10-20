"""сохраненная версия модели на этапе обучения"""

import shutil
from pathlib import Path

from act_recovery.files import write_json
from act_recovery.models import model_identity


def save_model(policy, source, identity, folder, run_info):

    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=False)
    policy.save_pretrained(folder) #сохраняем веса и конфиг (украдем из Lerobot)

    for pattern in ("policy_preprocessor*", "policy_postprocessor*"):
        for file in Path(source).glob(pattern):
            if file.is_file():
                shutil.copy2(file, folder / file.name)
                
    write_json(folder / "pilot_training_metadata.json", run_info)
    saved = model_identity(folder)
    for name, digest in identity["sha256"].items():
        if name.startswith("policy_") and saved["sha256"].get(name) != digest:
            raise RuntimeError("Policy normalization changed while saving the model.")
    return saved
