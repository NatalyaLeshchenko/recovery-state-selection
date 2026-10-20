"""Хочу: выбрать один обычный пример и три recovery-примера, объединить их, обработать, перенести на GPU и подготовить маску для расчёта ошибки."""

from act_recovery.training.loss import recovery_padding
from act_recovery.training.sampling import sample_frames


def training_batch(context):

    import torch

    rng = context["rng"]
    original = context["original"] # D0
    recovery = context["recovery"] # дополнительные recovery-демонстрации

    original_index = int(rng.integers(len(original))) #original[735] - obs, joints etc

    uniform, grasp, release = sample_frames(rng, context["episodes"], context["grasps"], context["releases"]) #uniform = 42 grasp = 18 release = 95
    selected = [original[original_index], recovery[uniform], recovery[grasp], recovery[release]] #selected = [original[735], recovery[42],recovery[18],recovery[95]]

    """ {
    "observation.images.camera1": изображение,
    "observation.images.camera2": изображение,
    "observation.state": состояние_робота,
    "action": последовательность_целевых_действий,
    "action_is_pad": маска_заполнения,
} """

    keys = list(context["policy"].config.input_features) + ["action", "action_is_pad"]
    batch = {}

    for key in keys:
        values = []
        for sample in selected:
            values.append(sample[key])
        batch[key] = torch.stack(values)
    batch = context["preprocessor"](batch) #GPU
    for key, value in batch.items():
        if torch.is_tensor(value):
            batch[key] = value.to("cuda")
    batch["action_is_pad"] = recovery_padding(
        batch["action_is_pad"], context["recipe"]["recovery_supervision_steps"] #10
    )
    return (batch, (original_index, uniform, grasp, release))
