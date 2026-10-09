#!/usr/bin/env python3
from __future__ import annotations
import argparse
import csv
from datetime import datetime
import json
from pathlib import Path
import random
import shutil

def positive_int(value: str) -> int:
    result = int(value)
    if result <= 0:
        raise argparse.ArgumentTypeError('Нужно положительное число.')
    return result

def local_path(root: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    return (path if path.is_absolute() else root / path).resolve()

def weighted_reconstruction_loss(prediction, target, is_pad, weights):
    if prediction.shape != target.shape or prediction.ndim != 3:
        raise RuntimeError(f'Неожиданные формы actions: {prediction.shape}, {target.shape}')
    if is_pad.shape != prediction.shape[:2]:
        raise RuntimeError(f'Неожиданная форма action_is_pad: {is_pad.shape}')
    valid = (~is_pad.bool()).unsqueeze(-1).to(prediction.dtype)
    valid_counts = valid.sum(dim=(1, 2))
    if bool((valid_counts < 1).any().item()):
        raise RuntimeError('В обучающий batch попал пример без действительных действий.')
    absolute_error = (prediction - target).abs()
    denominators = valid_counts * prediction.shape[-1]
    per_sample = (absolute_error * valid).sum(dim=(1, 2)) / denominators
    return ((per_sample * weights).sum(), per_sample, valid_counts)

def phase_focused_loss(policy, batch, weights):
    from lerobot.utils.constants import ACTION, OBS_IMAGES
    model_batch = dict(batch)
    if policy.config.image_features:
        source_items_1 = policy.config.image_features
        items_1 = []
        for key in source_items_1:
            items_1.append(model_batch[key])
        model_batch[OBS_IMAGES] = items_1
    actions_hat, (mu_hat, log_variance_hat) = policy.model(model_batch)
    reconstruction, per_sample, valid_counts = weighted_reconstruction_loss(actions_hat, model_batch[ACTION], model_batch['action_is_pad'], weights)
    kl = reconstruction.new_zeros(())
    if policy.config.use_vae and log_variance_hat is not None:
        kl_per_sample = (-0.5 * (1 + log_variance_hat - mu_hat.pow(2) - log_variance_hat.exp())).sum(-1)
        kl = (kl_per_sample * weights).sum()
    loss = reconstruction + policy.config.kl_weight * kl
    return (loss, {'l1': float(reconstruction.detach().cpu()), 'kl': float(kl.detach().cpu()), 'l1_original': float(per_sample[0].detach().cpu()), 'l1_recovery_uniform': float(per_sample[1].detach().cpu()), 'l1_release': float(per_sample[2].detach().cpu()), 'valid_original': int(valid_counts[0].detach().cpu()), 'valid_recovery_uniform': int(valid_counts[1].detach().cpu()), 'valid_release': int(valid_counts[2].detach().cpu())})

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root', type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument('--policy-path', type=Path, help='Источник весов; по умолчанию текущая recovery ACT.')
    parser.add_argument('--steps', type=positive_int, default=1000)
    parser.add_argument('--lr', type=float, default=1e-05)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--log-every', type=positive_int, default=20)
    parser.add_argument('--check-every', type=positive_int, default=200)
    parser.add_argument('--output-dir', type=Path)
    args = parser.parse_args()
    if not 0 < args.lr < 1:
        parser.error('--lr должен быть больше нуля и меньше единицы.')
    import numpy as np
    import torch
    from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
    from lerobot.policies.act.modeling_act import ACTPolicy
    from lerobot.policies.factory import make_pre_post_processors
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA недоступна: обучение не началось.')
    root = args.project_root.resolve()
    manifest = json.loads((root / 'pilot_runs/manifest.json').read_text(encoding='utf-8'))
    source = args.policy_path
    if source is None:
        source = (root / 'pilot_runs/recovery_policy_path.txt').read_text().strip()
    source = local_path(root, source)
    dataset_root = local_path(root, (root / 'pilot_runs/recovery_dataset_path.txt').read_text().strip())
    for name in ('config.json', 'model.safetensors', 'policy_preprocessor.json', 'policy_postprocessor.json'):
        if not (source / name).is_file():
            raise FileNotFoundError(f'Не найден файл модели: {source / name}')
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    rng = np.random.default_rng(args.seed)
    policy = ACTPolicy.from_pretrained(source, local_files_only=True, strict=True)
    policy.to('cuda')
    if policy.config.n_obs_steps != 1:
        raise RuntimeError('Этот pilot рассчитан на ACT с n_obs_steps=1.')
    action_indices = list(policy.config.action_delta_indices)
    if not action_indices or action_indices[0] != 0:
        raise RuntimeError('Ожидался action chunk, начинающийся с текущего действия.')
    preprocessor, postprocessor = make_pre_post_processors(policy.config, pretrained_path=source)
    nominal_meta = LeRobotDatasetMetadata(manifest['dataset_repo'], revision=manifest['dataset_revision'])
    nominal_fps = float(nominal_meta.fps)
    recovery_probe = LeRobotDataset('local/recovery-drop-pilot', root=dataset_root)
    recovery_fps = float(recovery_probe.meta.fps)
    del recovery_probe
    if nominal_fps <= 0 or recovery_fps <= 0:
        raise RuntimeError('FPS датасета должен быть положительным.')
    source_items_2 = action_indices
    items_2 = []
    for index in source_items_2:
        items_2.append(index / nominal_fps)
    original = LeRobotDataset(manifest['dataset_repo'], episodes=manifest['train_episodes'], revision=manifest['dataset_revision'], video_backend='pyav', delta_timestamps={'action': items_2})
    source_items_3 = action_indices
    items_3 = []
    for index in source_items_3:
        items_3.append(index / recovery_fps)
    recovery = LeRobotDataset('local/recovery-drop-pilot', root=dataset_root, delta_timestamps={'action': items_3})
    if not len(original) or not len(recovery):
        raise RuntimeError('Один из датасетов пуст.')
    input_keys = list(policy.config.input_features)
    keys = input_keys + ['action', 'action_is_pad']
    recovery_samples = []
    release_indices = []
    closed_earlier = {}
    for index in range(len(recovery)):
        sample = recovery[index]
        episode = int(sample['episode_index'].item())
        gripper = float(sample['action'][0, -1].item())
        previously_closed = closed_earlier.get(episode, False)
        if gripper < 0 and previously_closed:
            release_indices.append(index)
        closed_earlier[episode] = previously_closed or gripper > 0
        source_items_4 = keys
        items_4 = {}
        for key in source_items_4:
            items_4[key] = sample[key]
        recovery_samples.append(items_4)
    if not release_indices:
        raise RuntimeError('Не найдены команды открытия после предыдущего закрытия gripper.')
    output = args.output_dir or Path('pilot_runs/training') / datetime.now().strftime('release_focus_%Y%m%d_%H%M%S_%f')
    output = local_path(root, output)
    output.mkdir(parents=True, exist_ok=False)
    weights = torch.tensor([0.5, 0.25, 0.25], device='cuda', dtype=torch.float32)
    run_info = {'status': 'running', 'experiment': 'Phase-focused training-data diagnostic; not a selector comparison', 'source_policy': str(source), 'recovery_dataset': str(dataset_root), 'original_dataset': manifest['dataset_repo'], 'original_revision': manifest['dataset_revision'], 'original_train_episodes': manifest['train_episodes'], 'seed': args.seed, 'steps': args.steps, 'lr': args.lr, 'weight_decay': 0.0001, 'gradient_clip': 10.0, 'sample_weights': {'original': 0.5, 'recovery_uniform': 0.25, 'release': 0.25}, 'l1_normalization': 'Per sample, divided by its valid action count times action dimensions', 'release_indices': release_indices, 'release_definition': 'Expert opening command after an earlier closure in the same episode', 'chunk_size': len(action_indices), 'original_fps': nominal_fps, 'recovery_fps': recovery_fps, 'trainable_parameters': 'All existing parameters marked requires_grad', 'evaluation': 'Only a training-observation diagnostic is run here'}

    def write_info():
        (output / 'run_config.json').write_text(json.dumps(run_info, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')

    def move_batch(batch):
        source_items_5 = batch.items()
        items_5 = {}
        for key, value in source_items_5:
            items_5[key] = value.to('cuda') if torch.is_tensor(value) else value
        return items_5
    diagnostic_rows = []

    def check_release(step):
        policy.eval()
        predicted = []
        with torch.inference_mode():
            for index in release_indices:
                sample = recovery_samples[index]
                source_items_6 = input_keys
                batch = {}
                for key in source_items_6:
                    batch[key] = sample[key].unsqueeze(0)
                batch = move_batch(preprocessor(batch))
                policy.reset()
                action = postprocessor(policy.select_action(batch))
                if not torch.is_tensor(action) or action.numel() != 7:
                    raise RuntimeError('Неожиданный формат inference action.')
                predicted.append(float(action.detach().cpu().reshape(-1)[-1]))
        policy.reset()
        policy.train()
        row = {'step': step, 'frames': release_indices, 'gripper_predictions': predicted, 'mean_gripper': float(np.mean(predicted)), 'fraction_negative': float(np.mean(np.asarray(predicted) < 0))}
        diagnostic_rows.append(row)
        (output / 'release_diagnostic.json').write_text(json.dumps(diagnostic_rows, indent=2) + '\n', encoding='utf-8')
        print(f"RELEASE CHECK step={step}: mean_gripper={row['mean_gripper']:.4f} negative_fraction={row['fraction_negative']:.2f} commands={np.round(predicted, 3).tolist()}", flush=True)

    def save_model(folder):
        folder.mkdir(parents=True, exist_ok=False)
        policy.save_pretrained(folder)
        for pattern in ('policy_preprocessor*', 'policy_postprocessor*'):
            for processor_file in source.glob(pattern):
                if processor_file.is_file():
                    shutil.copy2(processor_file, folder / processor_file.name)
        (folder / 'pilot_training_metadata.json').write_text(json.dumps(run_info, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    write_info()
    print('Source model:', source, flush=True)
    print('Nominal frames:', len(original), 'Recovery frames:', len(recovery), flush=True)
    print('Action chunk length:', len(action_indices), 'FPS:', nominal_fps, recovery_fps, flush=True)
    print('Release frame indices:', release_indices, flush=True)
    print('Loss weights: original=0.50, recovery_uniform=0.25, release=0.25', flush=True)
    print('Run:', output, flush=True)
    optimizer = torch.optim.AdamW((parameter for parameter in policy.parameters() if parameter.requires_grad), lr=args.lr, weight_decay=0.0001)
    torch.cuda.reset_peak_memory_stats()
    step = 0
    fields = ['step', 'loss', 'l1', 'kl', 'l1_original', 'l1_recovery_uniform', 'l1_release', 'valid_original', 'valid_recovery_uniform', 'valid_release', 'recovery_uniform_frame', 'release_frame', 'peak_allocated_gib']
    try:
        check_release(0)
        with (output / 'losses.csv').open('x', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for step in range(1, args.steps + 1):
                original_index = int(rng.integers(len(original)))
                uniform_index = int(rng.integers(len(recovery_samples)))
                focused_index = int(rng.choice(release_indices))
                samples = [original[original_index], recovery_samples[uniform_index], recovery_samples[focused_index]]
                batch = {key: torch.stack([sample[key] for sample in samples]) for key in keys}
                batch = move_batch(preprocessor(batch))
                optimizer.zero_grad(set_to_none=True)
                loss, metrics = phase_focused_loss(policy, batch, weights)
                if not bool(torch.isfinite(loss).item()):
                    raise RuntimeError(f'Loss стал нечисловым на шаге {step}.')
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), 10.0)
                if not bool(torch.isfinite(grad_norm).item()):
                    raise RuntimeError(f'Градиенты стали нечисловыми на шаге {step}.')
                optimizer.step()
                peak = torch.cuda.max_memory_allocated() / 1024 ** 3
                writer.writerow({'step': step, 'loss': float(loss.detach().cpu()), **metrics, 'recovery_uniform_frame': uniform_index, 'release_frame': focused_index, 'peak_allocated_gib': peak})
                if step == 1 or step % args.log_every == 0 or step == args.steps:
                    stream.flush()
                    print(f"step={step}/{args.steps} loss={loss.item():.4f} l1_release={metrics['l1_release']:.4f} valid_actions={[metrics['valid_original'], metrics['valid_recovery_uniform'], metrics['valid_release']]} peak_allocated={peak:.2f} GiB", flush=True)
                if step % args.check_every == 0 or step == args.steps:
                    check_release(step)
                if step % 500 == 0 and step < args.steps:
                    run_info['completed_steps'] = step
                    save_model(output / 'checkpoints' / f'{step:06d}' / 'pretrained_model')
        run_info['status'] = 'completed'
        run_info['completed_steps'] = step
        destination = output / 'checkpoints/last/pretrained_model'
        save_model(destination)
        write_info()
        (root / 'pilot_runs/release_focused_policy_path.txt').write_text(str(destination) + '\n')
    except BaseException as error:
        run_info.update(status='stopped', completed_steps=step, error=f'{type(error).__name__}: {error}')
        write_info()
        raise
    print('\nPHASE-FOCUSED FINETUNING COMPLETE', flush=True)
    print('Model:', destination, flush=True)
    print('Pointer: pilot_runs/release_focused_policy_path.txt', flush=True)
if __name__ == '__main__':
    main()
