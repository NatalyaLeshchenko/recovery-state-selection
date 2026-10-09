#!/usr/bin/env python3
from __future__ import annotations
import argparse
import csv
from datetime import datetime
import json
from pathlib import Path
import random
import shutil
import numpy as np
REPO_ID = 'local/recovery-corrections-pilot'
INPUT_KEYS = ('observation.images.image', 'observation.images.image2', 'observation.state')

def positive_int(value):
    value = int(value)
    if value <= 0:
        raise argparse.ArgumentTypeError('Нужно положительное число.')
    return value

def local_path(root, value):
    path = Path(value).expanduser()
    return (path if path.is_absolute() else root / path).resolve()

def from_pointer(root, name):
    return local_path(root, (root / 'pilot_runs' / name).read_text().strip())

def as_numpy(value):
    if hasattr(value, 'detach'):
        value = value.detach().cpu().numpy()
    return np.asarray(value)

def image_hwc_uint8(value):
    array = as_numpy(value)
    if array.shape == (3, 256, 256):
        array = array.transpose(1, 2, 0)
    if array.shape != (256, 256, 3):
        raise RuntimeError(f'Неожиданная форма RGB: {array.shape}')
    if array.dtype != np.uint8:
        if not np.isfinite(array).all() or array.min() < 0 or array.max() > 1 + 1e-06:
            raise RuntimeError('Float RGB должен быть в диапазоне [0, 1], без нормализации mean/std.')
        array = np.rint(np.clip(array, 0, 1) * 255).astype(np.uint8)
    return np.ascontiguousarray(array)

def checked_frame(sample, instruction):
    state = as_numpy(sample['observation.state']).astype(np.float32)
    action = as_numpy(sample['action']).astype(np.float32)
    if state.shape != (8,) or action.shape != (7,) or (not np.isfinite(state).all()) or (not np.isfinite(action).all()):
        raise RuntimeError('Нужны конечные observation.state (8,) и action (7,).')
    return {'observation.images.image': image_hwc_uint8(sample['observation.images.image']), 'observation.images.image2': image_hwc_uint8(sample['observation.images.image2']), 'observation.state': state, 'action': action, 'task': instruction}

def load_correction(folder, instruction):
    metadata = json.loads((folder / 'metadata.json').read_text())
    if metadata.get('success') is not True or metadata.get('accepted') is not True:
        raise RuntimeError('Новая correction не подтверждена как успешная.')
    if metadata.get('instruction') != instruction or metadata.get('observation_timing') != 'before_action':
        raise RuntimeError('Инструкция или момент записи новой correction не соответствуют проекту.')
    if float(metadata.get('control_frequency_hz', 0)) != 20:
        raise RuntimeError('Correction должна быть записана с частотой 20 Hz.')
    with np.load(folder / 'demonstration.npz', allow_pickle=False) as source:
        source_items_1 = source.files
        data = {}
        for key in source_items_1:
            data[key] = source[key].copy()
    n = len(data['action'])
    shapes = {'observation/pixels/image': (n, 256, 256, 3), 'observation/pixels/image2': (n, 256, 256, 3), 'observation/robot_state/eef/pos': (n, 3), 'observation/robot_state/eef/quat': (n, 4), 'observation/robot_state/gripper/qpos': (n, 2), 'action': (n, 7), 'reward': (n,), 'phase': (n,)}
    for key, expected in shapes.items():
        if key not in data or data[key].shape != expected:
            raise RuntimeError(f'Неверная форма correction: {key}; ожидается {expected}.')
        if data[key].dtype.kind in 'fiu' and (not np.isfinite(data[key]).all()):
            raise RuntimeError(f'Некорректные числовые значения: {key}')
    if not n or n != metadata['n_frames'] or (not np.any(data['reward'] > 0)):
        raise RuntimeError('Пустая correction, неверная длина или отсутствует success reward.')
    if not np.any(np.isin(data['phase'], ('approach_grasp', 'close_gripper'))):
        raise RuntimeError('В correction не найдены этапы повторного захвата.')
    return (metadata, data)

def processed_correction_frame(data, index, processor, torch):
    raw = {'observation.images.image': torch.from_numpy(data['observation/pixels/image'][index].copy()).permute(2, 0, 1).unsqueeze(0), 'observation.images.image2': torch.from_numpy(data['observation/pixels/image2'][index].copy()).permute(2, 0, 1).unsqueeze(0), 'observation.robot_state': {'eef': {'pos': torch.from_numpy(data['observation/robot_state/eef/pos'][index].copy()).unsqueeze(0), 'quat': torch.from_numpy(data['observation/robot_state/eef/quat'][index].copy()).unsqueeze(0)}, 'gripper': {'qpos': torch.from_numpy(data['observation/robot_state/gripper/qpos'][index].copy()).unsqueeze(0)}}}
    processed = processor.observation(raw)
    source_items_2 = INPUT_KEYS
    items_2 = {}
    for key in source_items_2:
        items_2[key] = processed[key][0]
    return items_2 | {'action': data['action'][index]}

def write_dataset(writer, original, correction, instruction, processor, torch):
    previous_episode = None
    episode_count = 0
    for index in range(len(original)):
        sample = original[index]
        episode = int(as_numpy(sample['episode_index']).item())
        if previous_episode is not None and episode != previous_episode:
            writer.save_episode()
            episode_count += 1
        writer.add_frame(checked_frame(sample, instruction))
        previous_episode = episode
    if previous_episode is None:
        raise RuntimeError('Исходный recovery dataset пуст.')
    writer.save_episode()
    episode_count += 1
    grasp_indices = []
    for index in range(len(correction['action'])):
        sample = processed_correction_frame(correction, index, processor, torch)
        writer.add_frame(checked_frame(sample, instruction))
        if str(correction['phase'][index]) in ('approach_grasp', 'close_gripper'):
            grasp_indices.append(len(original) + index)
        if (index + 1) % 20 == 0:
            print(f"Corrections converted {index + 1}/{len(correction['action'])}", flush=True)
    writer.save_episode()
    writer.finalize()
    return (grasp_indices, episode_count + 1)

def prepare(args):
    import torch
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.processor.env_processor import LiberoProcessorStep
    root = args.project_root.resolve()
    manifest = json.loads((root / 'pilot_runs/manifest.json').read_text())
    original_root = from_pointer(root, 'recovery_dataset_path.txt')
    correction_root = local_path(root, args.correction_dir) if args.correction_dir else from_pointer(root, 'last_correction_demo.txt')
    metadata, correction = load_correction(correction_root, manifest['instruction'])
    original = LeRobotDataset('local/recovery-drop-pilot', root=original_root, return_uint8=True)
    if float(original.meta.fps) != 20:
        raise RuntimeError('Исходная recovery demonstration должна иметь FPS=20.')
    if (original_root / 'correction_sources.json').exists():
        raise RuntimeError('Исходный recovery pointer должен указывать на первоначальный dataset.')
    output = args.output_dir or Path('pilot_runs/datasets') / datetime.now().strftime('corrected_%Y%m%d_%H%M%S_%f')
    output = local_path(root, output)
    if output.exists():
        raise FileExistsError(f'Папка уже существует: {output}')
    output.parent.mkdir(parents=True, exist_ok=True)
    features = {'observation.images.image': {'dtype': 'image', 'shape': (256, 256, 3), 'names': ['height', 'width', 'channel']}, 'observation.images.image2': {'dtype': 'image', 'shape': (256, 256, 3), 'names': ['height', 'width', 'channel']}, 'observation.state': {'dtype': 'float32', 'shape': (8,), 'names': ['state']}, 'action': {'dtype': 'float32', 'shape': (7,), 'names': ['actions']}}
    writer = LeRobotDataset.create(REPO_ID, fps=20, features=features, root=output, use_videos=False)
    grasp_indices, n_episodes = write_dataset(writer, original, correction, manifest['instruction'], LiberoProcessorStep(), torch)
    sources = {'repo_id': REPO_ID, 'original_recovery_dataset': str(original_root), 'correction_demo': str(correction_root), 'original_frames': len(original), 'correction_frames': len(correction['action']), 'n_frames': len(original) + len(correction['action']), 'n_episodes': n_episodes, 'fps': 20, 'grasp_focus_indices': grasp_indices, 'takeover_step': metadata['takeover_step'], 'image_processing': 'Original recovery copied as-is; raw correction rotated once by LiberoProcessorStep'}
    (output / 'correction_sources.json').write_text(json.dumps(sources, indent=2) + '\n')
    check = LeRobotDataset(REPO_ID, root=output)
    if len(check) != sources['n_frames']:
        raise RuntimeError('Число кадров после сохранения не совпало.')
    for index in (0, len(original), len(check) - 1):
        checked_frame(check[index], manifest['instruction'])
    (root / 'pilot_runs/corrected_recovery_dataset_path.txt').write_text(str(output) + '\n')
    print('\nCORRECTED DATASET READY', flush=True)
    print('Output:', output, flush=True)
    print(f"Episodes: {n_episodes}; frames: {len(original)} + {len(correction['action'])} = {len(check)}", flush=True)

def focused_loss(policy, batch, weights):
    from finetune_release import weighted_reconstruction_loss
    from lerobot.utils.constants import ACTION, OBS_IMAGES
    model_batch = dict(batch)
    if policy.config.image_features:
        source_items_3 = policy.config.image_features
        items_3 = []
        for key in source_items_3:
            items_3.append(model_batch[key])
        model_batch[OBS_IMAGES] = items_3
    prediction, (mu, log_variance) = policy.model(model_batch)
    l1, per_sample, counts = weighted_reconstruction_loss(prediction, model_batch[ACTION], model_batch['action_is_pad'], weights)
    kl = l1.new_zeros(())
    if policy.config.use_vae and log_variance is not None:
        kl = ((-0.5 * (1 + log_variance - mu.pow(2) - log_variance.exp())).sum(-1) * weights).sum()
    loss = l1 + policy.config.kl_weight * kl
    names = ('original', 'uniform', 'grasp', 'release')
    metrics = {'l1': float(l1.detach().cpu()), 'kl': float(kl.detach().cpu())}
    for index, name in enumerate(names):
        metrics[f'l1_{name}'] = float(per_sample[index].detach().cpu())
        metrics[f'valid_{name}'] = int(counts[index].detach().cpu())
    return (loss, metrics)

def train(args):
    import torch
    from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
    from lerobot.policies.act.modeling_act import ACTPolicy
    from lerobot.policies.factory import make_pre_post_processors
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA недоступна: обучение не началось.')
    root = args.project_root.resolve()
    manifest = json.loads((root / 'pilot_runs/manifest.json').read_text())
    source = local_path(root, args.policy_path) if args.policy_path else from_pointer(root, 'release_focused_policy_path.txt')
    dataset_root = from_pointer(root, 'corrected_recovery_dataset_path.txt')
    sources = json.loads((dataset_root / 'correction_sources.json').read_text())
    for name in ('config.json', 'model.safetensors', 'policy_preprocessor.json', 'policy_postprocessor.json'):
        if not (source / name).is_file():
            raise FileNotFoundError(source / name)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    rng = np.random.default_rng(args.seed)
    policy = ACTPolicy.from_pretrained(source, local_files_only=True, strict=True)
    policy.to('cuda')
    if policy.config.n_obs_steps != 1:
        raise RuntimeError('Этот pilot рассчитан на ACT с n_obs_steps=1.')
    indices = list(policy.config.action_delta_indices)
    if not indices or indices[0] != 0:
        raise RuntimeError('Action chunk должен начинаться с текущего действия.')
    preprocessor, postprocessor = make_pre_post_processors(policy.config, pretrained_path=source)
    original_fps = float(LeRobotDatasetMetadata(manifest['dataset_repo'], revision=manifest['dataset_revision']).fps)
    source_items_4 = indices
    items_4 = []
    for index in source_items_4:
        items_4.append(index / original_fps)
    original = LeRobotDataset(manifest['dataset_repo'], revision=manifest['dataset_revision'], episodes=manifest['train_episodes'], video_backend='pyav', delta_timestamps={'action': items_4})
    source_items_5 = indices
    items_5 = []
    for index in source_items_5:
        items_5.append(index / sources['fps'])
    recovery = LeRobotDataset(REPO_ID, root=dataset_root, delta_timestamps={'action': items_5})
    input_keys = list(policy.config.input_features)
    sample_keys = input_keys + ['action', 'action_is_pad']
    samples = []
    releases = []
    previously_closed = {}
    for index in range(len(recovery)):
        sample = recovery[index]
        episode = int(sample['episode_index'].item())
        gripper = float(sample['action'][0, -1].item())
        if gripper < 0 and previously_closed.get(episode, False):
            releases.append(index)
        previously_closed[episode] = previously_closed.get(episode, False) or gripper > 0
        source_items_6 = sample_keys
        items_6 = {}
        for key in source_items_6:
            items_6[key] = sample[key]
        samples.append(items_6)
    grasps = sources['grasp_focus_indices']
    if not len(original) or not samples or (not releases) or (not grasps) or any((not 0 <= i < len(samples) for i in grasps)):
        raise RuntimeError('Пустые данные или неверные grasp/release indices.')
    boundary = int(sources['original_frames']) - 1
    if len(indices) > 1 and (not bool(samples[boundary]['action_is_pad'][1:].all().item())):
        raise RuntimeError('Action chunk пересекает границу старой и новой demonstrations.')
    output = args.output_dir or Path('pilot_runs/training') / datetime.now().strftime('corrected_%Y%m%d_%H%M%S_%f')
    output = local_path(root, output)
    output.mkdir(parents=True, exist_ok=False)
    weights = torch.tensor([0.5, 0.2, 0.15, 0.15], dtype=torch.float32, device='cuda')
    run_info = {'status': 'running', 'source_policy': str(source), 'dataset': str(dataset_root), 'original_dataset': manifest['dataset_repo'], 'original_revision': manifest['dataset_revision'], 'original_train_episodes': manifest['train_episodes'], 'steps': args.steps, 'seed': args.seed, 'lr': args.lr, 'weight_decay': 0.0001, 'gradient_clip': 10.0, 'sample_weights': {'original': 0.5, 'uniform': 0.2, 'grasp': 0.15, 'release': 0.15}, 'grasp_focus_indices': grasps, 'release_indices': releases, 'original_fps': original_fps, 'recovery_fps': sources['fps'], 'chunk_size': len(indices), 'normalization': 'Copied unchanged from source policy', 'trainable_parameters': 'All existing requires_grad parameters', 'experiment': 'Two recovery episodes with phase-focused sampling; not a selector comparison'}

    def write_info():
        (output / 'run_config.json').write_text(json.dumps(run_info, indent=2) + '\n')

    def move_batch(batch):
        source_items_7 = batch.items()
        items_7 = {}
        for key, value in source_items_7:
            items_7[key] = value.to('cuda') if torch.is_tensor(value) else value
        return items_7
    diagnostics = []
    probe_indices = sorted(set([int(sources['original_frames']), int(grasps[0]), int(grasps[-1])] + releases))

    def check_actions(step):
        policy.eval()
        results = []
        with torch.inference_mode():
            for index in probe_indices:
                source_items_8 = input_keys
                items_8 = {}
                for key in source_items_8:
                    items_8[key] = samples[index][key].unsqueeze(0)
                batch = move_batch(preprocessor(items_8))
                policy.reset()
                action = postprocessor(policy.select_action(batch))
                if not torch.is_tensor(action) or action.numel() != 7 or (not torch.isfinite(action).all().item()):
                    raise RuntimeError('Неверный diagnostic inference action.')
                results.append({'frame': index, 'expert': samples[index]['action'][0].tolist(), 'predicted': action.detach().cpu().reshape(7).tolist()})
        policy.reset()
        policy.train()
        diagnostics.append({'step': step, 'frames': results, 'evaluation': 'First inference action on training observations'})
        (output / 'action_checks.json').write_text(json.dumps(diagnostics, indent=2) + '\n')
        print(f'ACTION CHECK step={step}:', flush=True)
        for row in results:
            print(f"  frame={row['frame']}: expert dz={row['expert'][2]:.3f}, gripper={row['expert'][-1]:.3f}; ACT dz={row['predicted'][2]:.3f}, gripper={row['predicted'][-1]:.3f}", flush=True)

    def save_model(folder):
        folder.mkdir(parents=True, exist_ok=False)
        policy.save_pretrained(folder)
        for pattern in ('policy_preprocessor*', 'policy_postprocessor*'):
            for file in source.glob(pattern):
                if file.is_file():
                    shutil.copy2(file, folder / file.name)
        (folder / 'pilot_training_metadata.json').write_text(json.dumps(run_info, indent=2) + '\n')
    write_info()
    print('Source model:', source, flush=True)
    print('Nominal frames:', len(original), 'Recovery frames:', len(recovery), flush=True)
    print('Run:', output, flush=True)
    print('Loss weights: original=0.50, uniform=0.20, grasp=0.15, release=0.15', flush=True)
    optimizer = torch.optim.AdamW((p for p in policy.parameters() if p.requires_grad), lr=args.lr, weight_decay=0.0001)
    torch.cuda.reset_peak_memory_stats()
    source_items_9 = ('original', 'uniform', 'grasp', 'release')
    items_9 = []
    for n in source_items_9:
        items_9.append(f'l1_{n}')
    source_items_10 = ('original', 'uniform', 'grasp', 'release')
    items_10 = []
    for n in source_items_10:
        items_10.append(f'valid_{n}')
    fields = ['step', 'loss', 'l1', 'kl'] + items_9 + items_10 + ['uniform_frame', 'grasp_frame', 'release_frame', 'peak_allocated_gib']
    step = 0
    try:
        check_actions(0)
        with (output / 'losses.csv').open('x', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for step in range(1, args.steps + 1):
                original_index = int(rng.integers(len(original)))
                uniform_index = int(rng.integers(len(samples)))
                grasp_index = int(rng.choice(grasps))
                release_index = int(rng.choice(releases))
                selected = [original[original_index], samples[uniform_index], samples[grasp_index], samples[release_index]]
                batch = move_batch(preprocessor({key: torch.stack([s[key] for s in selected]) for key in sample_keys}))
                optimizer.zero_grad(set_to_none=True)
                loss, metrics = focused_loss(policy, batch, weights)
                if not torch.isfinite(loss).item():
                    raise RuntimeError('Loss стал нечисловым.')
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), 10.0)
                if not torch.isfinite(grad_norm).item():
                    raise RuntimeError('Градиенты стали нечисловыми.')
                optimizer.step()
                peak = torch.cuda.max_memory_allocated() / 1024 ** 3
                writer.writerow({'step': step, 'loss': float(loss.detach().cpu()), **metrics, 'uniform_frame': uniform_index, 'grasp_frame': grasp_index, 'release_frame': release_index, 'peak_allocated_gib': peak})
                if step == 1 or step % 20 == 0 or step == args.steps:
                    stream.flush()
                    print(f"step={step}/{args.steps} loss={loss.item():.4f} l1_grasp={metrics['l1_grasp']:.4f} l1_release={metrics['l1_release']:.4f} peak_allocated={peak:.2f} GiB", flush=True)
                if step % 200 == 0 or step == args.steps:
                    check_actions(step)
                if step % 500 == 0 and step < args.steps:
                    run_info['completed_steps'] = step
                    save_model(output / 'checkpoints' / f'{step:06d}' / 'pretrained_model')
        run_info.update(status='completed', completed_steps=step)
        destination = output / 'checkpoints/last/pretrained_model'
        save_model(destination)
        write_info()
        (root / 'pilot_runs/corrected_policy_path.txt').write_text(str(destination) + '\n')
    except BaseException as error:
        run_info.update(status='stopped', completed_steps=step, error=f'{type(error).__name__}: {error}')
        write_info()
        raise
    print('\nCORRECTION FINETUNING COMPLETE', flush=True)
    print('Model:', destination, flush=True)
    print('Pointer: pilot_runs/corrected_policy_path.txt', flush=True)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    prepare_parser = commands.add_parser('prepare', help='Объединить старую recovery и новую correction в отдельные эпизоды.')
    train_parser = commands.add_parser('train', help='Дообучить ACT на объединённых recovery данных с replay.')
    for command in (prepare_parser, train_parser):
        command.add_argument('--project-root', type=Path, default=Path(__file__).resolve().parent)
        command.add_argument('--output-dir', type=Path)
    prepare_parser.add_argument('--correction-dir', type=Path)
    train_parser.add_argument('--policy-path', type=Path)
    train_parser.add_argument('--steps', type=positive_int, default=1000)
    train_parser.add_argument('--seed', type=int, default=0)
    train_parser.add_argument('--lr', type=float, default=1e-05)
    args = parser.parse_args()
    if args.command == 'prepare':
        prepare(args)
    else:
        if not 0 < args.lr < 1:
            parser.error('--lr должен быть больше нуля и меньше единицы.')
        train(args)
if __name__ == '__main__':
    main()
