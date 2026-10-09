#!/usr/bin/env python3
from __future__ import annotations
import argparse
import csv
from datetime import datetime
import json
import os
from pathlib import Path
import random
import shutil
import time
os.environ.setdefault('MUJOCO_GL', 'egl')
import numpy as np
REPO_ID = 'local/diverse-recovery-pilot'
COLLECTION_POINTER = 'diverse_recovery_collection_path.txt'
DATASET_POINTER = 'diverse_recovery_dataset_path.txt'
POLICY_POINTER = 'diverse_recovery_policy_path.txt'
SOURCE_FILE = 'diverse_sources.json'
WEIGHTS = (0.5, 0.2, 0.15, 0.15)
EXPECTED_HELPERS = ('validate_recovery.py', 'check_passive_recovery.py', 'collect_correction.py', 'train_corrections.py', 'finetune_release.py')

def positive_int(value):
    value = int(value)
    if value <= 0:
        raise argparse.ArgumentTypeError('Нужно положительное число.')
    return value

def local_path(root, value):
    path = Path(value).expanduser()
    return (path if path.is_absolute() else root / path).resolve()

def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))

def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')

def from_pointer(root, name):
    return local_path(root, (root / 'pilot_runs' / name).read_text().strip())

def publish_pointer(root, name, path):
    pointer = root / 'pilot_runs' / name
    temporary = pointer.with_suffix('.tmp')
    temporary.write_text(str(path) + '\n', encoding='utf-8')
    temporary.replace(pointer)

def new_folder(root, parent, prefix):
    folder = root / 'pilot_runs' / parent / datetime.now().strftime(prefix + '_%Y%m%d_%H%M%S_%f')
    folder.mkdir(parents=True, exist_ok=False)
    return folder

def new_records():
    return {'observations': {}, 'actions': [], 'rewards': [], 'phases': [], 'geometry': [], 'success': False, 'ended': False}

def check_manifest(root):
    manifest = read_json(root / 'pilot_runs/manifest.json')
    if manifest['suite'] != 'libero_spatial' or manifest['task_id'] != 0:
        raise RuntimeError('Этот pilot рассчитан на libero_spatial, task 0.')
    return manifest

def collect_one(pool, case, protocol, output, grasp_offset, placement_offset, args):
    from check_passive_recovery import benchmark_goal, make_vector
    from collect_correction import GoalReached, find_robot_env, geometry, run_expert, save_attempt
    from validate_recovery import EvaluationHooks
    output.mkdir(parents=True, exist_ok=False)
    case_folder = pool / f"init_{int(case['init_state_id']):03d}"
    with np.load(case_folder / 'start.npz', allow_pickle=False) as data:
        prefix_steps = len(data['past_actions'])
    expert_protocol = dict(protocol, recovery_horizon_steps=args.max_expert_steps)
    request = {'stage': 'evaluate', 'variant': 'scripted_expert', 'protocol': expert_protocol, 'case_dir': str(case_folder), 'output': str(output), 'init_state_id': case['init_state_id'], 'init_definition_sha256': case['init_definition_sha256']}
    write_json(output / 'request.json', request)
    records = new_records()
    metadata = {'suite': protocol['suite'], 'task_id': protocol['task_id'], 'instruction': protocol['instruction'], 'seed': protocol['seed'], 'init_state_id': case['init_state_id'], 'init_definition_sha256': case['init_definition_sha256'], 'source_case': str(case_folder), 'source_snapshot_sha256': case['snapshot_sha256'], 'source_case_metadata_sha256': case['metadata_sha256'], 'failure_type': 'object_dropped', 'failure_source': 'forced_open_gripper_after_verified_height_rise', 'expert': 'scripted_controller_with_privileged_simulator_geometry', 'control_frequency_hz': 20, 'observation_timing': 'before_action', 'max_expert_steps': args.max_expert_steps, 'grasp_offset': grasp_offset.tolist(), 'placement_offset': placement_offset.tolist(), 'calibration_note': 'Fixed offsets from the original successful init-0 trajectory; no per-case retuning.', 'evaluation_note': 'Training/development demonstration, not a test episode.'}
    vec = None
    try:
        vec = make_vector(expert_protocol, prefix_steps)
        hooks = EvaluationHooks(request, type(vec).reset, type(vec).step)
        obs, _ = hooks.reset(vec, seed=protocol['seed'])
        raw = find_robot_env(vec)
        metadata['restore_check'] = hooks.restore
        if benchmark_goal(vec) is True:
            raise RuntimeError('Сохранённое состояние уже удовлетворяет goal; продолжение не собирается.')
        metadata['start_geometry'] = geometry(raw)
        shutil.copy2(case_folder / 'start.npz', output / 'start.npz')
        started = time.perf_counter()
        try:
            run_expert(vec, raw, obs, grasp_offset, placement_offset, args.max_expert_steps, records)
        except GoalReached:
            pass
        except RuntimeError as error:
            metadata['error'] = str(error)
            print('EXPERT ATTEMPT FAILED:', error, flush=True)
        metadata['expert_execution_wall_seconds'] = time.perf_counter() - started
        metadata['end_geometry'] = geometry(raw)
        final_goal = benchmark_goal(vec)
        metadata['goal_predicate_at_end'] = final_goal
        if final_goal is not None and final_goal != records['success']:
            raise RuntimeError('Benchmark goal и success reward расходятся; данные не принимаются.')
        source_items_1 = records['geometry']
        times = []
        for row in source_items_1:
            times.append(row['simulator_time'])
        if times:
            times.append(metadata['end_geometry']['simulator_time'])
            if not np.allclose(np.diff(times), 0.05, atol=1e-08, rtol=0):
                raise RuntimeError('Частота управления отличается от 20 Hz.')
        save_attempt(output, records, metadata, make_gif=not args.no_gif)
        from validate_recovery import file_hash
        metadata['metadata_sha256'] = file_hash(output / 'metadata.json')
        if records['success']:
            metadata['demonstration_sha256'] = file_hash(output / 'demonstration.npz')
        return {'folder': str(output), 'init_state_id': case['init_state_id'], 'init_definition_sha256': case['init_definition_sha256'], 'accepted': records['success'], 'expert_steps': len(records['actions']), 'metadata_sha256': metadata['metadata_sha256'], 'demonstration_sha256': metadata.get('demonstration_sha256'), 'error': metadata.get('error'), 'expert_execution_wall_seconds': metadata['expert_execution_wall_seconds']}
    finally:
        if vec is not None and (not getattr(vec, 'closed', False)):
            vec.close()

def collect(args, root):
    from check_passive_recovery import load_pool
    from collect_correction import calibrate
    from lerobot.envs.configs import LiberoEnv
    from validate_recovery import file_hash, runtime_versions, verify_model
    manifest = check_manifest(root)
    pool = local_path(root, args.pool)
    protocol, cases = load_pool(pool, args.init_ids)
    if protocol['runtime_versions'] != runtime_versions():
        raise RuntimeError('Версии среды изменились после сбора пула; точное replay не гарантировано.')
    if protocol['instruction'] != manifest['instruction']:
        raise RuntimeError('Инструкция пула не соответствует pilot_runs/manifest.json.')
    verify_model(protocol['models']['initial'])
    nominal_path = root / 'pilot_runs/lift_trace/actions.npz'
    with np.load(nominal_path, allow_pickle=False) as data:
        nominal_actions = data['actions'].copy()
    if nominal_actions.ndim != 3 or nominal_actions.shape[1:] != (1, 7) or (not np.isfinite(nominal_actions).all()):
        raise RuntimeError('Неверная форма калибровочных действий.')
    output = new_folder(root, 'recovery_demos', 'diverse_collection')
    source_items_2 = cases
    items_2 = []
    for case in source_items_2:
        items_2.append(int(case['init_state_id']))
    info = {'status': 'running', 'complete': False, 'pool': str(pool), 'pool_protocol_sha256': file_hash(pool / 'protocol.json'), 'instruction': manifest['instruction'], 'source_initial_model': protocol['models']['initial'], 'runtime_versions': runtime_versions(), 'requested_starts': len(cases), 'selected_init_ids': items_2, 'selection_rule': 'All specified pool cases in stored order; no filter by autonomous-policy outcome.', 'attempts_per_start': 1, 'attempts': [], 'accepted_count': 0, 'calibration_actions': str(nominal_path), 'calibration_actions_sha256': file_hash(nominal_path), 'scope': 'Diverse expert-data pilot. Reused evaluation starts become development data.'}
    write_json(output / 'collection.json', info)
    print('Output:', output, flush=True)
    try:
        cfg = LiberoEnv(task='libero_spatial', task_ids=[0], episode_length=max(len(nominal_actions) + 50, 400), observation_height=256, observation_width=256)
        grasp_offset, placement_offset = calibrate(cfg, nominal_actions)
        info['grasp_offset'] = grasp_offset.tolist()
        info['placement_offset'] = placement_offset.tolist()
        for number, case in enumerate(cases, start=1):
            init_id = int(case['init_state_id'])
            print(f'\nEXPERT {number}/{len(cases)}, init_id={init_id}', flush=True)
            result = collect_one(pool, case, protocol, output / f'init_{init_id:03d}', grasp_offset, placement_offset, args)
            info['attempts'].append(result)
            info['accepted_count'] += int(result['accepted'])
            write_json(output / 'collection.json', info)
            print(f"accepted={result['accepted']}; expert_steps={result['expert_steps']}; successful demonstrations={info['accepted_count']}/{number}", flush=True)
        info.update(status='complete', complete=True)
        write_json(output / 'collection.json', info)
        publish_pointer(root, COLLECTION_POINTER, output)
    except BaseException as error:
        info.update(status='interrupted', error=f'{type(error).__name__}: {error}')
        write_json(output / 'collection.json', info)
        raise
    print(f"\nEXPERT COLLECTION COMPLETE: {info['accepted_count']}/{len(cases)}", flush=True)
    print('Output:', output, flush=True)

def load_successful_demo(attempt, instruction):
    from validate_recovery import file_hash
    folder = Path(attempt['folder'])
    if attempt.get('accepted') is not True:
        raise RuntimeError('Неуспешная попытка не является demonstration.')
    for filename, key in (('metadata.json', 'metadata_sha256'), ('demonstration.npz', 'demonstration_sha256')):
        if file_hash(folder / filename) != attempt[key]:
            raise RuntimeError(f'Исходные данные изменились: {folder / filename}')
    metadata = read_json(folder / 'metadata.json')
    if metadata.get('accepted') is not True or metadata.get('success') is not True or metadata.get('observation_timing') != 'before_action' or (metadata.get('instruction') != instruction) or (metadata.get('control_frequency_hz') != 20) or (metadata.get('init_definition_sha256') != attempt['init_definition_sha256']):
        raise RuntimeError('Метаданные expert demonstration не соответствуют протоколу.')
    with np.load(folder / 'demonstration.npz', allow_pickle=False) as source:
        source_items_3 = source.files
        data = {}
        for key in source_items_3:
            data[key] = source[key].copy()
    n = len(data['action'])
    shapes = {'observation/pixels/image': (n, 256, 256, 3), 'observation/pixels/image2': (n, 256, 256, 3), 'observation/robot_state/eef/pos': (n, 3), 'observation/robot_state/eef/quat': (n, 4), 'observation/robot_state/gripper/qpos': (n, 2), 'action': (n, 7), 'reward': (n,), 'phase': (n,)}
    for key, shape in shapes.items():
        if key not in data or data[key].shape != shape:
            raise RuntimeError(f'Неверная форма {key}; ожидается {shape}.')
        if data[key].dtype.kind in 'fiu' and (not np.isfinite(data[key]).all()):
            raise RuntimeError(f'Неконечные числовые значения: {key}.')
    for key in ('observation/pixels/image', 'observation/pixels/image2'):
        if data[key].dtype != np.uint8:
            raise RuntimeError('Ожидаются исходные RGB uint8.')
    if not n or n != metadata['n_frames'] or (not np.any(data['reward'] > 0)):
        raise RuntimeError('Пустая demonstration, неверная длина или отсутствует success reward.')
    if np.any(np.abs(data['action']) > 1.00001):
        raise RuntimeError('Expert actions выходят за пределы OSC action limits.')
    if np.any(data['reward'][:-1] > 0) or data['reward'][-1] <= 0:
        raise RuntimeError('Запись должна останавливаться на первом benchmark success.')
    return (metadata, data)

def validate_ranges(episodes, n_frames):
    cursor = 0
    for number, episode in enumerate(episodes):
        if episode['episode_index'] != number or episode['start'] != cursor or episode['stop'] <= cursor:
            raise RuntimeError('Неверные границы recovery episodes.')
        cursor = episode['stop']
    if cursor != n_frames or not episodes:
        raise RuntimeError('Границы episodes не покрывают dataset.')

def prepare(args, root):
    import torch
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.processor.env_processor import LiberoProcessorStep
    from train_corrections import as_numpy, checked_frame, processed_correction_frame
    from validate_recovery import file_hash
    manifest = check_manifest(root)
    collection_root = local_path(root, args.collection) if args.collection else from_pointer(root, COLLECTION_POINTER)
    collection = read_json(collection_root / 'collection.json')
    if not collection.get('complete') or collection.get('instruction') != manifest['instruction']:
        raise RuntimeError('Нужен завершённый expert collection для текущей задачи.')
    source_items_4 = collection['attempts']
    accepted = []
    for attempt in source_items_4:
        if attempt['accepted']:
            accepted.append(attempt)
    if len(accepted) != collection['accepted_count']:
        raise RuntimeError('Счётчик successful demonstrations не соответствует collection.json.')
    source_items_5 = accepted
    items_5 = set()
    for a in source_items_5:
        items_5.add(a['init_definition_sha256'])
    if len(items_5) < args.min_new_layouts:
        raise RuntimeError(f'Нужно хотя бы {args.min_new_layouts} новых успешных раскладок; сейчас {len(accepted)}. Проверьте результаты expert attempts.')
    base_root = from_pointer(root, 'corrected_recovery_dataset_path.txt')
    base_sources = read_json(base_root / 'correction_sources.json')
    base = LeRobotDataset('local/recovery-corrections-pilot', root=base_root, return_uint8=True)
    if float(base.meta.fps) != 20 or not len(base) or len(base) != base_sources['n_frames']:
        raise RuntimeError('Старые recovery данные должны быть непустыми, с FPS=20.')
    for attempt in accepted:
        metadata, data = load_successful_demo(attempt, manifest['instruction'])
        del metadata, data
    output = new_folder(root, 'datasets', 'diverse_recovery')
    features = {'observation.images.image': {'dtype': 'image', 'shape': (256, 256, 3), 'names': ['height', 'width', 'channel']}, 'observation.images.image2': {'dtype': 'image', 'shape': (256, 256, 3), 'names': ['height', 'width', 'channel']}, 'observation.state': {'dtype': 'float32', 'shape': (8,), 'names': ['state']}, 'action': {'dtype': 'float32', 'shape': (7,), 'names': ['actions']}}
    output.rmdir()
    writer = LeRobotDataset.create(REPO_ID, fps=20, features=features, root=output, use_videos=False)
    episodes, grasp_indices, release_indices = ([], [], [])
    original_grasps = set(base_sources['grasp_focus_indices'])
    if any((not 0 <= index < len(base) for index in original_grasps)):
        raise RuntimeError('Неверные grasp indices старого dataset.')
    count, start, previous, seen, closed = (0, 0, None, set(), False)

    def add(sample, grasp=False):
        nonlocal count, closed
        frame = checked_frame(sample, manifest['instruction'])
        gripper = float(frame['action'][-1])
        if gripper < 0 and closed:
            release_indices.append(count)
        closed = closed or gripper > 0
        if grasp:
            grasp_indices.append(count)
        writer.add_frame(frame)
        count += 1

    def finish(kind, init_id, init_hash=None, folder=None):
        nonlocal start, closed
        writer.save_episode()
        episodes.append({'episode_index': len(episodes), 'start': start, 'stop': count, 'source_kind': kind, 'init_state_id': init_id, 'init_definition_sha256': init_hash, 'source_folder': folder})
        start, closed = (count, False)
    for index in range(len(base)):
        sample = base[index]
        episode = int(as_numpy(sample['episode_index']).item())
        if previous is not None and episode != previous:
            finish('historical_init0_recovery', 0, folder=str(base_root))
            if episode in seen:
                raise RuntimeError('Старые recovery episodes расположены не последовательно.')
        seen.add(episode)
        add(sample, grasp=index in original_grasps)
        previous = episode
    finish('historical_init0_recovery', 0, folder=str(base_root))
    if len(episodes) != base_sources['n_episodes']:
        raise RuntimeError('Количество исторических recovery episodes не совпало.')
    processor = LiberoProcessorStep()
    for number, attempt in enumerate(accepted, start=1):
        _, data = load_successful_demo(attempt, manifest['instruction'])
        for index in range(len(data['action'])):
            sample = processed_correction_frame(data, index, processor, torch)
            add(sample, grasp=str(data['phase'][index]) in ('approach_grasp', 'close_gripper'))
        finish('diverse_expert_recovery', int(attempt['init_state_id']), attempt['init_definition_sha256'], attempt['folder'])
        print(f'Converted episodes: {number}/{len(accepted)}; total frames={count}', flush=True)
        del data
    writer.finalize()
    validate_ranges(episodes, count)
    if not grasp_indices or not release_indices:
        raise RuntimeError('В recovery данных отсутствуют grasp/release кадры для существующего sampler.')
    source_items_6 = episodes
    items_6 = set()
    for ep in source_items_6:
        items_6.add(int(ep['init_state_id']))
    source_items_7 = episodes
    items_7 = set()
    for ep in source_items_7:
        if ep['init_definition_sha256']:
            items_7.add(ep['init_definition_sha256'])
    sources = {'status': 'complete', 'repo_id': REPO_ID, 'fps': 20, 'n_frames': count, 'n_episodes': len(episodes), 'episodes': episodes, 'grasp_focus_indices': grasp_indices, 'release_focus_indices': release_indices, 'initial_policy': collection['source_initial_model'], 'collection': str(collection_root), 'collection_sha256': file_hash(collection_root / 'collection.json'), 'base_dataset': str(base_root), 'base_source_metadata_sha256': file_hash(base_root / 'correction_sources.json'), 'new_demonstrations': accepted, 'used_init_state_ids': sorted(items_6), 'used_init_definition_sha256': sorted(items_7), 'image_processing': 'Historical converted images unchanged; new raw images rotated once by LiberoProcessorStep.', 'observation_timing': 'before_action', 'evaluation_note': 'These source layouts are development/training data, not independent final-test layouts.'}
    check = LeRobotDataset(REPO_ID, root=output)
    if len(check) != count:
        raise RuntimeError('Количество сохранённых кадров не совпало.')
    for episode in episodes:
        for index in (episode['start'], episode['stop'] - 1):
            sample = check[index]
            checked_frame(sample, manifest['instruction'])
            if int(as_numpy(sample['episode_index']).item()) != episode['episode_index']:
                raise RuntimeError('Граница episodes не сохранилась.')
    write_json(output / SOURCE_FILE, sources)
    publish_pointer(root, DATASET_POINTER, output)
    print(f'\nDIVERSE DATASET READY: {len(episodes)} episodes, {count} frames', flush=True)
    print('Initial-state IDs:', sources['used_init_state_ids'], flush=True)
    print('Output:', output, flush=True)

def focus_groups(episodes, indices):
    validate_ranges(episodes, episodes[-1]['stop'] if episodes else 0)
    sorted_indices = sorted(set((int(i) for i in indices)))
    if not sorted_indices or sorted_indices[0] < 0 or sorted_indices[-1] >= episodes[-1]['stop']:
        raise RuntimeError('Неверные phase-focus indices.')
    groups = []
    for episode in episodes:
        source_items_8 = sorted_indices
        group = []
        for index in source_items_8:
            if episode['start'] <= index < episode['stop']:
                group.append(index)
        if group:
            groups.append(group)
    return groups

def sample_frames(rng, episodes, grasps, releases):
    episode = episodes[int(rng.integers(len(episodes)))]
    uniform = int(rng.integers(episode['start'], episode['stop']))
    grasp_group = grasps[int(rng.integers(len(grasps)))]
    release_group = releases[int(rng.integers(len(releases)))]
    return (uniform, int(rng.choice(grasp_group)), int(rng.choice(release_group)))

def train(args, root):
    import torch
    from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
    from lerobot.policies.act.modeling_act import ACTPolicy
    from lerobot.policies.factory import make_pre_post_processors
    from train_corrections import focused_loss
    from validate_recovery import file_hash, model_identity, runtime_versions, verify_model
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA недоступна; обучение не запущено.')
    manifest = check_manifest(root)
    dataset_root = from_pointer(root, DATASET_POINTER)
    sources = read_json(dataset_root / SOURCE_FILE)
    if sources.get('status') != 'complete' or sources['repo_id'] != REPO_ID:
        raise RuntimeError('Нужен завершённый diverse recovery dataset.')
    identity = sources['initial_policy']
    verify_model(identity)
    source = Path(identity['path'])
    episodes = sources['episodes']
    validate_ranges(episodes, sources['n_frames'])
    grasps = focus_groups(episodes, sources['grasp_focus_indices'])
    releases = focus_groups(episodes, sources['release_focus_indices'])
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    rng = np.random.default_rng(args.seed)
    policy = ACTPolicy.from_pretrained(source, local_files_only=True, strict=True)
    if policy.config.n_obs_steps != 1:
        raise RuntimeError('Этот pilot поддерживает ACT с n_obs_steps=1.')
    indices = list(policy.config.action_delta_indices)
    if indices != list(range(len(indices))) or not indices or args.n_action_steps > len(indices):
        raise RuntimeError('Неверный action chunk или n_action_steps.')
    policy.config.n_action_steps = args.n_action_steps
    policy.to('cuda')
    preprocessor, _ = make_pre_post_processors(policy.config, pretrained_path=source)
    original_fps = float(LeRobotDatasetMetadata(manifest['dataset_repo'], revision=manifest['dataset_revision']).fps)
    if original_fps <= 0 or float(sources['fps']) != 20:
        raise RuntimeError('Некорректная частота датасетов.')
    source_items_9 = indices
    items_9 = []
    for index in source_items_9:
        items_9.append(index / original_fps)
    original = LeRobotDataset(manifest['dataset_repo'], revision=manifest['dataset_revision'], episodes=manifest['train_episodes'], video_backend='pyav', delta_timestamps={'action': items_9})
    source_items_10 = indices
    items_10 = []
    for index in source_items_10:
        items_10.append(index / sources['fps'])
    recovery = LeRobotDataset(REPO_ID, root=dataset_root, delta_timestamps={'action': items_10})
    if not len(original) or len(recovery) != sources['n_frames']:
        raise RuntimeError('Пустые исходные данные или изменился recovery dataset.')
    for episode in episodes:
        sample = recovery[episode['stop'] - 1]
        if int(sample['episode_index'].item()) != episode['episode_index'] or bool(sample['action_is_pad'][0].item()) or (len(indices) > 1 and (not bool(sample['action_is_pad'][1:].all().item()))):
            raise RuntimeError('Action chunk пересекает границу recovery episodes.')
    output = new_folder(root, 'training', 'diverse_from_initial')
    run_info = {'status': 'running', 'source_initial_model': identity, 'recovery_dataset': str(dataset_root), 'recovery_source_metadata_sha256': file_hash(dataset_root / SOURCE_FILE), 'original_dataset': manifest['dataset_repo'], 'original_revision': manifest['dataset_revision'], 'original_train_episodes': manifest['train_episodes'], 'steps': args.steps, 'seed': args.seed, 'lr': args.lr, 'n_action_steps': args.n_action_steps, 'chunk_size': len(indices), 'weights': dict(zip(('original', 'uniform', 'grasp', 'release'), WEIGHTS)), 'sampling': 'Original frames uniform; recovery episodes uniform, then frames uniform; phase samples episode-balanced.', 'weight_decay': 0.0001, 'gradient_clip': 10.0, 'original_fps': original_fps, 'recovery_fps': sources['fps'], 'normalization': 'Unchanged initial-policy processors', 'trainable_parameters': 'All parameters already marked requires_grad', 'used_init_state_ids': sources['used_init_state_ids'], 'used_init_definition_sha256': sources['used_init_definition_sha256'], 'runtime_versions': runtime_versions(), 'scope': 'ACT learning pilot, not a data-selector comparison; closed-loop evaluation is separate.'}
    write_json(output / 'run_config.json', run_info)
    input_keys = list(policy.config.input_features)
    keys = input_keys + ['action', 'action_is_pad']
    weights = torch.tensor(WEIGHTS, device='cuda', dtype=torch.float32)
    optimizer = torch.optim.AdamW((p for p in policy.parameters() if p.requires_grad), lr=args.lr, weight_decay=0.0001)

    def save_model(folder):
        folder.mkdir(parents=True, exist_ok=False)
        policy.save_pretrained(folder)
        for pattern in ('policy_preprocessor*', 'policy_postprocessor*'):
            for file in source.glob(pattern):
                if file.is_file():
                    shutil.copy2(file, folder / file.name)
        write_json(folder / 'pilot_training_metadata.json', run_info)
        saved = model_identity(folder)
        for name, sha256 in identity['sha256'].items():
            if name.startswith('policy_') and saved['sha256'].get(name) != sha256:
                raise RuntimeError('Нормализация изменилась при сохранении.')
        return saved
    names = ('original', 'uniform', 'grasp', 'release')
    source_items_11 = ('l1', 'valid')
    items_11 = []
    for prefix in source_items_11:
        for name in names:
            items_11.append(f'{prefix}_{name}')
    fields = ['step', 'loss', 'l1', 'kl'] + items_11 + ['original_frame', 'uniform_frame', 'grasp_frame', 'release_frame', 'peak_allocated_gib']
    print('Source model:', source, flush=True)
    print(f'Recovery: {len(episodes)} episodes, {len(recovery)} frames', flush=True)
    print('Training initial-state IDs:', sources['used_init_state_ids'], flush=True)
    print('Loss weights: original=0.50, uniform=0.20, grasp=0.15, release=0.15', flush=True)
    print('Run:', output, flush=True)
    policy.train()
    torch.cuda.reset_peak_memory_stats()
    completed = 0
    started = time.perf_counter()
    try:
        with (output / 'losses.csv').open('x', newline='', encoding='utf-8') as stream:
            csv_writer = csv.DictWriter(stream, fieldnames=fields)
            csv_writer.writeheader()
            for step in range(1, args.steps + 1):
                original_index = int(rng.integers(len(original)))
                uniform, grasp, release = sample_frames(rng, episodes, grasps, releases)
                selected = [original[original_index], recovery[uniform], recovery[grasp], recovery[release]]
                batch = preprocessor({key: torch.stack([sample[key] for sample in selected]) for key in keys})
                source_items_12 = batch.items()
                batch = {}
                for key, value in source_items_12:
                    batch[key] = value.to('cuda') if torch.is_tensor(value) else value
                optimizer.zero_grad(set_to_none=True)
                loss, metrics = focused_loss(policy, batch, weights)
                if not torch.isfinite(loss).item():
                    raise RuntimeError('Loss стал нечисловым.')
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), 10.0)
                if not torch.isfinite(grad_norm).item():
                    raise RuntimeError('Градиенты стали нечисловыми.')
                optimizer.step()
                completed = step
                value = float(loss.detach().cpu())
                peak = torch.cuda.max_memory_allocated() / 1024 ** 3
                csv_writer.writerow({'step': step, 'loss': value, **metrics, 'original_frame': original_index, 'uniform_frame': uniform, 'grasp_frame': grasp, 'release_frame': release, 'peak_allocated_gib': peak})
                if step == 1 or step % args.log_every == 0 or step == args.steps:
                    stream.flush()
                    print(f"step={step}/{args.steps} loss={value:.4f} l1_recovery={metrics['l1_uniform']:.4f} peak_allocated={peak:.2f} GiB", flush=True)
                if step % args.save_every == 0 and step < args.steps:
                    run_info['completed_updates'] = completed
                    save_model(output / 'checkpoints' / f'{step:06d}' / 'pretrained_model')
        verify_model(identity)
        run_info.update(status='complete', completed_updates=completed, training_wall_seconds=time.perf_counter() - started)
        model_folder = output / 'checkpoints/last/pretrained_model'
        saved_identity = save_model(model_folder)
        run_info['saved_policy'] = saved_identity
        write_json(output / 'run_config.json', run_info)
        publish_pointer(root, POLICY_POINTER, model_folder)
    except BaseException as error:
        run_info.update(status='interrupted', completed_updates=completed, error=f'{type(error).__name__}: {error}')
        write_json(output / 'run_config.json', run_info)
        raise
    print('\nDIVERSE FINETUNING COMPLETE', flush=True)
    print('Model:', model_folder, flush=True)
    print('Pointer:', root / 'pilot_runs' / POLICY_POINTER, flush=True)

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--project-root', type=Path, default=Path(__file__).resolve().parent)
    sub = parser.add_subparsers(dest='command', required=True)
    collection = sub.add_parser('collect', help='Expert continuations из сохранённых recovery starts.')
    collection.add_argument('--pool', type=Path, required=True)
    collection.add_argument('--init-ids', type=int, nargs='+', help='По умолчанию все состояния указанного пула.')
    collection.add_argument('--max-expert-steps', type=positive_int, default=280)
    collection.add_argument('--no-gif', action='store_true')
    preparation = sub.add_parser('prepare', help='Старые и новые successful demos в одном dataset.')
    preparation.add_argument('--collection', type=Path, help='По умолчанию последний завершённый diverse collection.')
    preparation.add_argument('--min-new-layouts', type=positive_int, default=2)
    training = sub.add_parser('train', help='Новое обучение от исходной ACT с исходной нормализацией.')
    training.add_argument('--steps', type=positive_int, default=1000)
    training.add_argument('--lr', type=float, default=1e-05)
    training.add_argument('--seed', type=int, default=0)
    training.add_argument('--n-action-steps', type=positive_int, default=10)
    training.add_argument('--log-every', type=positive_int, default=20)
    training.add_argument('--save-every', type=positive_int, default=500)
    args = parser.parse_args()
    if args.command == 'train' and (not 0 < args.lr < 1):
        parser.error('--lr должен быть больше нуля и меньше единицы.')
    root = args.project_root.expanduser().resolve()
    source_items_13 = EXPECTED_HELPERS
    missing = []
    for name in source_items_13:
        if not (root / name).is_file():
            missing.append(name)
    if missing:
        parser.error(f"Сохраните файл в папку проекта рядом с: {', '.join(missing)}")
    os.chdir(root)
    {'collect': collect, 'prepare': prepare, 'train': train}[args.command](args, root)
if __name__ == '__main__':
    main()
