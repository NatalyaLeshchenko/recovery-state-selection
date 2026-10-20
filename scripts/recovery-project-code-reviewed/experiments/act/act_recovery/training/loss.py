"""Ошибка предсказания ACT, веса групп данных и маска recovery-действий."""

from __future__ import annotations


def recovery_padding(mask, horizon):
    
    
    result = mask.clone() if hasattr(mask, "clone") else mask.copy()
    result[1:, horizon:] = True

    """ 
    для horizon == 10:

        №0 — D0	Все доступные действия чанка
        №1 — recovery uniform	Первые 10 доступных действий
        №2 — recovery grasp	Первые 10 доступных действий
        №3 — recovery release	Первые 10 доступных действий
    
    """

    return result


def focused_loss(policy, batch, weights):

    from lerobot.utils.constants import ACTION, OBS_IMAGES

    model_batch = dict(batch)
    if policy.config.image_features:
        images = []
        for key in policy.config.image_features:
            images.append(model_batch[key])
        model_batch[OBS_IMAGES] = images

        #получили предикшнс
    prediction, (mu, log_variance) = policy.model(model_batch)
    l1, per_sample, counts = weighted_reconstruction_loss(
        prediction, model_batch[ACTION], model_batch["action_is_pad"], weights
    )

    kl = l1.new_zeros(())
    if policy.config.use_vae and log_variance is not None:
        kl = ((-0.5 * (1 + log_variance - mu.pow(2) - log_variance.exp())).sum(-1) * weights).sum()
    loss = l1 + policy.config.kl_weight * kl
    names = ("original", "uniform", "grasp", "release")
    metrics = {"l1": float(l1.detach().cpu()), "kl": float(kl.detach().cpu())}
    for index, name in enumerate(names):
        metrics[f"l1_{name}"] = float(per_sample[index].detach().cpu())
        metrics[f"valid_{name}"] = int(counts[index].detach().cpu())
    return (loss, metrics)


def weighted_reconstruction_loss(prediction, target, is_pad, weights):
    """ prediction  # команды, которые предсказала ACT
        target      # model_batch[ACTION]
        is_pad      # какие шаги не учитывать
        weights     # вклад каждого из четырёх примеров """
    
    valid = (~is_pad.bool()).unsqueeze(-1).to(prediction.dtype) #маска

    #число учитываемых действий для каждого примера и умножаем на 7 компонен
    valid_counts = valid.sum(dim=(1, 2))

    #одну среднюю L1-ошибку на каждый из четырёх примеров
    absolute_error = (prediction - target).abs()

    """ 
    target     = [0.2, 0, 0, 0, 0, 0, 1]  # команда демонстрации
        prediction = [0.5, 0, 0, 0, 0, 0, 1]  # предсказание ACT

        absolute_error = [0.3, 0, 0, 0, 0, 0, 0]
    """

    denominators = valid_counts * prediction.shape[-1]

    """

    D0:       0.10 × 0.50 = 0.05
    Recovery: 0.20 × 0.20 = 0.04
    Grasp:    0.40 × 0.15 = 0.06
    Release:  0.20 × 0.15 = 0.03

    Итоговая L1 = 0.18

    """

    per_sample = (absolute_error * valid).sum(dim=(1, 2)) / denominators
    return ((per_sample * weights).sum(), per_sample, valid_counts)
