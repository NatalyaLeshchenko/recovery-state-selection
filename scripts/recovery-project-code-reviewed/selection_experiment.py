#!/usr/bin/env python3
from __future__ import annotations
import argparse
from collections import Counter
from datetime import datetime
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import time
import traceback
from types import SimpleNamespace
import numpy as np
os.environ.setdefault('MUJOCO_GL', 'egl')
import selection_pilot as pilot
METHODS = ('U', 'F_ref', 'Random')
COLLECTION_POINTER = 'selection_expert_collection_path.txt'
DATASETS_POINTER = 'selection_datasets_path.txt'
MODELS_POINTER = 'selection_models_path.txt'
SOURCE_FILE = 'selection_sources.json'
WEIGHTS = (0.5, 0.2, 0.15, 0.15)
RECOVERY_SUPERVISION = 10

def positive_int(value):
    value = int(value)
    if value < 1:
        raise argparse.ArgumentTypeError('Нужно положительное целое число.')
    return value

def load_plan(root, value=None):
    folder = pilot.local_path(root, value) if value else pilot.resolve_pointer(root, 'selection_plan_path.txt')
    plan_path = folder if folder.is_file() else folder / 'selection.json'
    plan = pilot.read_json(plan_path)
    if set(plan.get('conditions', {})) != set(METHODS):
        raise RuntimeError('Нужен общий selection plan для U, F_ref и Random.')
    pool, protocol = pilot.load_pool(root, plan['pool'])
    scores_folder = Path(plan['scores'])
    scores = pilot.read_json(scores_folder / 'scores.json')
    if scores.get('status') != 'complete' or Path(scores['pool']).resolve() != pool or pilot.file_hash(pool / 'protocol.json') != scores['pool_protocol_sha256'] or (pilot.file_hash(scores_folder / 'predictions.npz') != scores['predictions_sha256']):
        raise RuntimeError('Scores и plan не соответствуют зафиксированному common pool.')
    source_items_1 = protocol['cases']
    pool_cases = {}
    for case in source_items_1:
        pool_cases[case['candidate_id']] = case
    score_cases = scores['cases']
    if len(score_cases) != len(pool_cases) or {row['candidate_id'] for row in score_cases} != set(pool_cases):
        raise RuntimeError('Score rows покрывают другой candidate pool.')
    with np.load(scores_folder / 'predictions.npz', allow_pickle=False) as saved:
        source_items_2 = score_cases
        items_2 = []
        for row in source_items_2:
            items_2.append(row['candidate_id'])
        if saved['candidate_ids'].tolist() != items_2:
            raise RuntimeError('Predictions и score rows имеют разный порядок candidates.')
        recalculated = pilot.disagreement(saved['normalized_actions'])
    for row, value in zip(score_cases, recalculated):
        case = pool_cases[row['candidate_id']]
        if row['snapshot_sha256'] != case['snapshot_sha256'] or row['failure_type'] != case['failure_type'] or (not np.isclose(row['U'], value, atol=1e-12, rtol=1e-12)):
            raise RuntimeError('Uncertainty score или reference label не соответствует сохранённым данным.')
    budget = plan['target_successful_demonstrations']
    if isinstance(budget, bool) or not isinstance(budget, int) or budget < 1:
        raise RuntimeError('Некорректный demonstration budget.')
    expected = pilot.query_plan(scores['cases'], budget, plan['seed'])
    if expected != plan['conditions']:
        raise RuntimeError('Selection plan изменён или не воспроизводится из сохранённых scores.')
    _, ensemble = pilot.load_ensemble(root, protocol['ensemble_folder'])
    if ensemble['models'][0]['identity'] != protocol['models']['initial'] or pilot.file_hash(Path(protocol['ensemble_folder']) / 'ensemble.json') != protocol['ensemble_sha256']:
        raise RuntimeError('Исходная ACT или nominal ensemble изменились.')
    return (plan_path, plan, pool, protocol)

def resolve_condition(method, condition, budget, seed, attempt):
    accepted, attempts = ([], [])
    requested = set()
    accepted_types = Counter()

    def query(candidate, kind=None):
        if candidate in requested:
            raise RuntimeError('Selector повторно запросил один candidate.')
        requested.add(candidate)
        row = dict(attempt(candidate))
        if row.get('candidate_id') != candidate or not isinstance(row.get('accepted'), bool):
            raise RuntimeError('Неверный результат expert attempt.')
        attempts.append(row)
        if row['accepted']:
            accepted.append(candidate)
            accepted_types[row['failure_type']] += 1
        if kind is not None and row['failure_type'] != kind:
            raise RuntimeError('Reference label expert attempt отличается от группы selector.')
    if method in ('U', 'Random'):
        for candidate in condition['query_order']:
            if len(accepted) == budget:
                break
            query(candidate)
    elif method == 'F_ref':
        queues = condition['query_orders_by_type']
        source_items_3 = queues
        positions = {}
        for kind in source_items_3:
            positions[kind] = 0

        def next_of_type(kind):
            candidate = queues[kind][positions[kind]]
            positions[kind] += 1
            query(candidate, kind)
        for kind in pilot.FAILURE_TYPES:
            while accepted_types[kind] < condition['quotas'][kind] and positions[kind] < len(queues[kind]):
                next_of_type(kind)
        rng = random.Random(seed)
        while len(accepted) < budget:
            source_items_4 = sorted(queues)
            available = []
            for kind in source_items_4:
                if positions[kind] < len(queues[kind]):
                    available.append(kind)
            if not available:
                break
            minimum = min((accepted_types[kind] for kind in available))
            source_items_5 = available
            tied = []
            for kind in source_items_5:
                if accepted_types[kind] == minimum:
                    tied.append(kind)
            next_of_type(rng.choice(tied))
    else:
        raise ValueError(f'Неизвестный selector: {method}')
    return {'selected_successful_candidates': accepted, 'attempts': attempts, 'target_budget': budget, 'accepted_count': len(accepted), 'budget_met': len(accepted) == budget, 'failure_counts': dict(accepted_types), 'n_attempts': len(attempts), 'expert_action_timesteps_including_failures': sum((row['expert_steps'] for row in attempts)), 'expert_execution_seconds_logical': sum((row['expert_execution_wall_seconds'] for row in attempts)), 'cached_attempts': sum((bool(row.get('cache_hit')) for row in attempts))}

def expert_one(request):
    from collect_correction import GoalReached, find_robot_env, geometry, run_expert, save_attempt
    from validate_recovery import EvaluationHooks
    output = Path(request['output'])
    case = request['case']
    folder = Path(request['case_folder'])
    if pilot.file_hash(folder / 'start.npz') != case['snapshot_sha256']:
        raise RuntimeError('Candidate snapshot изменился перед expert collection.')
    with np.load(folder / 'start.npz', allow_pickle=False) as saved:
        prefix_steps = len(saved['past_actions'])
    protocol = dict(request['protocol'], recovery_horizon_steps=request['max_expert_steps'])
    replay = {'stage': 'evaluate', 'variant': 'scripted_expert', 'protocol': protocol, 'case_dir': str(folder), 'output': str(output), 'init_state_id': case['init_state_id'], 'init_definition_sha256': case['init_definition_sha256']}
    records = {'observations': {}, 'actions': [], 'rewards': [], 'phases': [], 'geometry': [], 'success': False, 'ended': False}
    metadata = {'candidate_id': case['candidate_id'], 'suite': protocol['suite'], 'task_id': protocol['task_id'], 'instruction': protocol['instruction'], 'seed': protocol['seed'], 'init_state_id': case['init_state_id'], 'init_definition_sha256': case['init_definition_sha256'], 'failure_type': case['failure_type'], 'label_source': case['label_source'], 'source_snapshot_sha256': case['snapshot_sha256'], 'source_case': str(folder), 'expert': 'scripted_controller_with_privileged_simulator_geometry', 'control_frequency_hz': 20, 'observation_timing': 'before_action', 'grasp_offset': request['grasp_offset'], 'placement_offset': request['placement_offset'], 'max_expert_steps': request['max_expert_steps'], 'calibration_note': 'Fixed successful nominal init-0 calibration; no per-case or per-selector retuning', 'scope': 'Expert training/development demonstration; no autonomous policy test result'}
    vec = pilot.make_vector(protocol, prefix_steps)
    try:
        hooks = EvaluationHooks(replay, type(vec).reset, type(vec).step)
        obs, _ = hooks.reset(vec, seed=protocol['seed'])
        raw = find_robot_env(vec)
        if pilot.goal_at_start(vec):
            raise RuntimeError('Recovery start уже удовлетворяет goal.')
        metadata['restore_check'] = hooks.restore
        metadata['start_geometry'] = geometry(raw)
        shutil.copy2(folder / 'start.npz', output / 'start.npz')
        started = time.perf_counter()
        try:
            run_expert(vec, raw, obs, np.asarray(request['grasp_offset']), np.asarray(request['placement_offset']), request['max_expert_steps'], records)
        except GoalReached:
            pass
        except RuntimeError as error:
            metadata['error'] = str(error)
            print('EXPERT ATTEMPT FAILED:', error, flush=True)
        metadata['expert_execution_wall_seconds'] = time.perf_counter() - started
        metadata['end_geometry'] = geometry(raw)
        final_goal = pilot.goal_at_start(vec)
        metadata['goal_predicate_at_end'] = final_goal
        if final_goal != records['success']:
            raise RuntimeError('Benchmark predicate и recorded success reward расходятся.')
        if records['geometry']:
            source_items_6 = records['geometry']
            items_6 = []
            for row in source_items_6:
                items_6.append(row['simulator_time'])
            times = items_6 + [metadata['end_geometry']['simulator_time']]
            if not np.allclose(np.diff(times), 0.05, atol=1e-08, rtol=0):
                raise RuntimeError('Частота expert control отличается от 20 Hz.')
        save_attempt(output, records, metadata, make_gif=True)
        return {'candidate_id': case['candidate_id'], 'failure_type': case['failure_type'], 'folder': str(output), 'accepted': records['success'], 'init_state_id': case['init_state_id'], 'init_definition_sha256': case['init_definition_sha256'], 'source_snapshot_sha256': case['snapshot_sha256'], 'expert_steps': len(records['actions']), 'error': metadata.get('error'), 'expert_execution_wall_seconds': metadata['expert_execution_wall_seconds'], 'metadata_sha256': pilot.file_hash(output / 'metadata.json'), 'demonstration_sha256': pilot.file_hash(output / 'demonstration.npz') if records['success'] else None}
    finally:
        vec.close()

def run_child(root, command, request, output, live=False):
    output.mkdir(parents=True, exist_ok=False)
    request_path = output / 'request.json'
    pilot.write_json(request_path, request)
    call = [sys.executable, str(Path(__file__).resolve()), command, '--request', str(request_path)]
    env = dict(os.environ, OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', WANDB_MODE='disabled', HF_HUB_DISABLE_TELEMETRY='1')
    if live:
        completed = subprocess.run(call, cwd=root, env=env)
    else:
        with (output / 'worker.log').open('w', encoding='utf-8') as log:
            completed = subprocess.run(call, cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT)
    result_path = output / 'worker_result.json'
    if completed.returncode != 0 or not result_path.is_file():
        if not live:
            print('\n'.join((output / 'worker.log').read_text(errors='replace').splitlines()[-45:]), flush=True)
        raise RuntimeError(f'Этап остановлен; данные и лог: {output}')
    result = pilot.read_json(result_path)
    if result.get('status') != 'complete':
        raise RuntimeError(f'Незавершённый worker: {result}')
    return result

def expert(args, root):
    from collect_correction import calibrate
    from lerobot.envs.configs import LiberoEnv
    from train_uncertainty_ensemble import verify_identity
    plan_path, plan, pool, protocol = load_plan(root, args.plan)
    output = pilot.new_folder(root, 'expert_selection')
    nominal_path = root / 'pilot_runs/lift_trace/actions.npz'
    with np.load(nominal_path, allow_pickle=False) as saved:
        nominal = saved['actions'].copy()
    cfg = LiberoEnv(task='libero_spatial', task_ids=[0], control_mode='relative', init_states=True, hard_reset=True, observation_height=256, observation_width=256, episode_length=len(nominal) + 20)
    grasp_offset, placement_offset = calibrate(cfg, nominal)
    print('Expert grasp offset =', grasp_offset, flush=True)
    info = {'status': 'running', 'complete': False, 'pool': str(pool), 'instruction': protocol['instruction'], 'pool_protocol_sha256': pilot.file_hash(pool / 'protocol.json'), 'selection_plan': str(plan_path), 'selection_plan_sha256': pilot.file_hash(plan_path), 'source_initial_model': protocol['models']['initial'], 'manifest_sha256': protocol['manifest_sha256'], 'target_budget': plan['target_successful_demonstrations'], 'selection_seed': plan['seed'], 'expert': 'fixed_privileged_scripted_controller', 'max_expert_steps': args.max_expert_steps, 'nominal_calibration_actions': str(nominal_path), 'nominal_actions_sha256': pilot.file_hash(nominal_path), 'grasp_offset': grasp_offset.tolist(), 'placement_offset': placement_offset.tolist(), 'conditions': {}, 'unique_attempts': {}, 'cost_note': 'Shared candidates are physically collected once. Per-condition logical acquisition counts include cached attempts; physical total is reported separately. These are controller times, not human teleoperation times.', 'scope': 'ACT selected-data learning pilot; independent evaluation is a separate stage'}
    pilot.write_json(output / 'collection.json', info)
    source_items_7 = protocol['cases']
    cases = {}
    for case in source_items_7:
        cases[case['candidate_id']] = case

    def attempt(candidate):
        if candidate not in cases:
            raise RuntimeError('Selector запросил candidate вне common pool.')
        if candidate in info['unique_attempts']:
            return {**info['unique_attempts'][candidate], 'cache_hit': True}
        case = cases[candidate]
        verify_identity(info['source_initial_model'])
        folder = output / candidate
        print(f'\nEXPERT {candidate}: restore → corrective actions → benchmark goal', flush=True)
        request = {'project_root': str(root), 'output': str(folder), 'case_folder': str(pool / candidate), 'case': case, 'protocol': protocol, 'max_expert_steps': args.max_expert_steps, 'grasp_offset': grasp_offset.tolist(), 'placement_offset': placement_offset.tolist()}
        result = run_child(root, '_expert-worker', request, folder)['attempt']
        info['unique_attempts'][candidate] = result
        pilot.write_json(output / 'collection.json', info)
        print(f"accepted={result['accepted']}; expert_steps={result['expert_steps']}", flush=True)
        return {**result, 'cache_hit': False}
    try:
        for method in METHODS:
            print(f"\n=== {method}: target B={info['target_budget']} ===", flush=True)
            info['conditions'][method] = resolve_condition(method, plan['conditions'][method], info['target_budget'], plan['seed'], attempt)
            pilot.write_json(output / 'collection.json', info)
        verify_identity(info['source_initial_model'])
        if pilot.file_hash(pool / 'protocol.json') != info['pool_protocol_sha256']:
            raise RuntimeError('Common pool изменился во время expert collection.')
        info['complete'] = all((row['budget_met'] for row in info['conditions'].values()))
        info['status'] = 'complete' if info['complete'] else 'unmet_budget'
        info['physical_unique_attempts'] = len(info['unique_attempts'])
        info['physical_expert_action_timesteps'] = sum((row['expert_steps'] for row in info['unique_attempts'].values()))
        info['physical_expert_execution_seconds'] = sum((row['expert_execution_wall_seconds'] for row in info['unique_attempts'].values()))
        pilot.write_json(output / 'collection.json', info)
    except BaseException as error:
        info.update(status='interrupted', error=f'{type(error).__name__}: {error}')
        pilot.write_json(output / 'collection.json', info)
        raise
    for method, row in info['conditions'].items():
        print(f"{method}: {row['accepted_count']}/{info['target_budget']} accepted; types={row['failure_counts']}")
    print('Output:', output)
    if not info['complete']:
        raise pilot.StageNotReady('Не все selectors получили B успешных demonstrations; дообучение остановлено.')
    pilot.write_pointer(root, COLLECTION_POINTER, output)
    print('\nEXPERT DATA READY')

def load_collection(root, value=None):
    folder = pilot.local_path(root, value) if value else pilot.resolve_pointer(root, COLLECTION_POINTER)
    info = pilot.read_json(folder / 'collection.json')
    if not info.get('complete') or info.get('status') != 'complete' or set(info['conditions']) != set(METHODS):
        raise pilot.StageNotReady('Нужен завершённый expert collection для всех трёх selectors.')
    if pilot.file_hash(Path(info['pool']) / 'protocol.json') != info['pool_protocol_sha256'] or pilot.file_hash(info['selection_plan']) != info['selection_plan_sha256'] or pilot.file_hash(root / 'pilot_runs/manifest.json') != info['manifest_sha256']:
        raise RuntimeError('Common pool, plan или исходный D0 изменились после collection.')
    from train_uncertainty_ensemble import verify_identity
    verify_identity(info['source_initial_model'])
    for method, row in info['conditions'].items():
        accepted = row['selected_successful_candidates']
        if len(accepted) != info['target_budget'] or len(set(accepted)) != len(accepted) or (not row['budget_met']):
            raise RuntimeError(f'Не выполнен demonstration budget для {method}.')
        for candidate in accepted:
            attempt = info['unique_attempts'][candidate]
            if not attempt['accepted'] or attempt['candidate_id'] != candidate:
                raise RuntimeError('Selected successful dataset содержит failed attempt.')
    return (folder, info)

def prepare(args, root):
    import torch
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.processor.env_processor import LiberoProcessorStep
    from train_corrections import checked_frame, processed_correction_frame
    from expand_recovery_pilot import load_successful_demo, validate_ranges
    collection, info = load_collection(root, args.collection)
    output = pilot.new_folder(root, 'datasets_selection')
    summary = {'status': 'running', 'collection': str(collection), 'collection_sha256': pilot.file_hash(collection / 'collection.json'), 'target_budget': info['target_budget'], 'datasets': {}}
    pilot.write_json(output / 'datasets.json', summary)
    features = {'observation.images.image': {'dtype': 'image', 'shape': (256, 256, 3), 'names': ['height', 'width', 'channel']}, 'observation.images.image2': {'dtype': 'image', 'shape': (256, 256, 3), 'names': ['height', 'width', 'channel']}, 'observation.state': {'dtype': 'float32', 'shape': (8,), 'names': ['state']}, 'action': {'dtype': 'float32', 'shape': (7,), 'names': ['actions']}}
    try:
        for method in METHODS:
            selected = info['conditions'][method]['selected_successful_candidates']
            folder = output / method
            repo_id = f"local/selection-{method.lower().replace('_', '-')}"
            writer = LeRobotDataset.create(repo_id, fps=20, features=features, root=folder, use_videos=False)
            processor = LiberoProcessorStep()
            episodes, grasp_indices, release_indices, attempts = ([], [], [], [])
            count = 0
            for candidate in selected:
                attempt = info['unique_attempts'][candidate]
                metadata, data = load_successful_demo(attempt, info['instruction'])
                if metadata.get('candidate_id') != candidate or metadata['source_snapshot_sha256'] != attempt['source_snapshot_sha256'] or metadata['failure_type'] != attempt['failure_type']:
                    raise RuntimeError('Expert demo не соответствует selected candidate.')
                start, closed = (count, False)
                for index in range(len(data['action'])):
                    frame = checked_frame(processed_correction_frame(data, index, processor, torch), info['instruction'])
                    grip = float(frame['action'][-1])
                    if str(data['phase'][index]) in ('approach_grasp', 'close_gripper'):
                        grasp_indices.append(count)
                    if grip < 0 and closed:
                        release_indices.append(count)
                    closed = closed or grip > 0
                    writer.add_frame(frame)
                    count += 1
                writer.save_episode()
                episodes.append({'episode_index': len(episodes), 'start': start, 'stop': count, 'candidate_id': candidate, 'source_kind': 'selected_expert_recovery', 'init_state_id': attempt['init_state_id'], 'failure_type': attempt['failure_type'], 'init_definition_sha256': attempt['init_definition_sha256'], 'source_folder': attempt['folder']})
                attempts.append(attempt)
                del metadata, data
            writer.finalize()
            validate_ranges(episodes, count)
            if not grasp_indices or not release_indices:
                raise RuntimeError(f'В данных {method} нет grasp/release examples для общего sampling recipe.')
            source_items_8 = episodes
            items_8 = set()
            for row in source_items_8:
                items_8.add(row['init_state_id'])
            source_items_9 = episodes
            items_9 = set()
            for row in source_items_9:
                items_9.add(row['init_definition_sha256'])
            sources = {'status': 'complete', 'repo_id': repo_id, 'fps': 20, 'n_frames': count, 'n_episodes': len(episodes), 'selector': method, 'demonstration_budget': info['target_budget'], 'episodes': episodes, 'grasp_focus_indices': grasp_indices, 'release_focus_indices': release_indices, 'initial_policy': info['source_initial_model'], 'collection': str(collection), 'collection_sha256': summary['collection_sha256'], 'new_demonstrations': attempts, 'used_init_state_ids': sorted(items_8), 'used_init_definition_sha256': sorted(items_9), 'historical_recovery_demonstrations_included': False, 'observation_timing': 'before_action', 'image_processing': 'Raw RGB rotated once by LiberoProcessorStep; unchanged D0 policy normalization'}
            pilot.write_json(folder / SOURCE_FILE, sources)
            check = LeRobotDataset(repo_id, root=folder)
            if len(check) != count or len(episodes) != info['target_budget']:
                raise RuntimeError('Число сохранённых frames/episodes не соответствует budget.')
            for episode in episodes:
                for index in (episode['start'], episode['stop'] - 1):
                    sample = check[index]
                    checked_frame(sample, info['instruction'])
                    if int(sample['episode_index'].item()) != episode['episode_index']:
                        raise RuntimeError('Граница demonstration episodes не сохранилась.')
            del check, writer, processor
            summary['datasets'][method] = {'root': str(folder), 'repo_id': repo_id, 'n_frames': count, 'n_episodes': len(episodes), 'sources_sha256': pilot.file_hash(folder / SOURCE_FILE)}
            pilot.write_json(output / 'datasets.json', summary)
            pilot.write_pointer(root, f'selection_{method}_dataset_path.txt', folder)
            print(f'{method}: {len(episodes)} demonstrations, {count} frames; {folder}', flush=True)
        summary['status'] = 'complete'
        pilot.write_json(output / 'datasets.json', summary)
        pilot.write_pointer(root, DATASETS_POINTER, output)
    except BaseException as error:
        summary.update(status='interrupted', error=f'{type(error).__name__}: {error}')
        pilot.write_json(output / 'datasets.json', summary)
        raise
    print('\nSELECTED DATASETS READY')

def recovery_padding(mask, horizon):
    if mask.ndim != 2 or mask.shape[0] != 4 or (not 1 <= horizon < mask.shape[1]):
        raise RuntimeError('Нужен batch [D0, recovery-uniform, grasp, release] с полным ACT chunk.')
    result = mask.clone() if hasattr(mask, 'clone') else mask.copy()
    result[1:, horizon:] = True
    return result

def train_one(request):
    import expand_recovery_pilot as trainer
    import train_corrections as training_module
    from train_uncertainty_ensemble import verify_identity
    root = Path(request['project_root'])
    dataset = request['dataset']
    sources = pilot.read_json(Path(dataset['root']) / SOURCE_FILE)
    if pilot.file_hash(Path(dataset['root']) / SOURCE_FILE) != dataset['sources_sha256'] or sources['initial_policy'] != request['initial_policy'] or sources['n_episodes'] != request['budget']:
        raise RuntimeError('Dataset или общая исходная ACT изменились перед дообучением.')
    verify_identity(request['initial_policy'])
    if trainer.WEIGHTS != WEIGHTS:
        raise RuntimeError('Sampling weights существующего trainer отличаются от общего recipe.')
    recipe = request['recipe']
    target = Path(request['output']) / 'training'
    pointer = f"selection_{request['method']}_policy_path.txt"
    dataset_pointer = f"selection_{request['method']}_dataset_path.txt"
    if pilot.resolve_pointer(root, dataset_pointer) != Path(dataset['root']):
        raise RuntimeError('Dataset pointer указывает на другой эксперимент.')
    old_loss = training_module.focused_loss
    source_items_10 = ('REPO_ID', 'SOURCE_FILE', 'DATASET_POINTER', 'POLICY_POINTER', 'new_folder', 'write_json')
    old = {}
    for key in source_items_10:
        old[key] = getattr(trainer, key)

    def loss(policy, batch, weights):
        masked = dict(batch)
        masked['action_is_pad'] = recovery_padding(batch['action_is_pad'], recipe['recovery_supervision_steps'])
        return old_loss(policy, masked, weights)

    def new_folder(project_root, parent, prefix):
        target.mkdir(parents=True, exist_ok=False)
        return target

    def write_json(path, value):
        if isinstance(value, dict) and 'recovery_dataset' in value and ('source_initial_model' in value):
            value = {**value, 'selector': request['method'], 'demonstration_budget': request['budget'], 'recovery_supervision_steps': recipe['recovery_supervision_steps'], 'scope': 'ACT selected-data comparison; no independent test result in training logs', 'common_recipe': recipe, 'dataset_sources_sha256': dataset['sources_sha256']}
        old['write_json'](path, value)
    try:
        trainer.REPO_ID, trainer.SOURCE_FILE = (dataset['repo_id'], SOURCE_FILE)
        trainer.DATASET_POINTER, trainer.POLICY_POINTER = (dataset_pointer, pointer)
        trainer.new_folder, trainer.write_json = (new_folder, write_json)
        training_module.focused_loss = loss
        args = SimpleNamespace(steps=recipe['steps'], seed=recipe['seed'], lr=recipe['lr'], n_action_steps=recipe['n_action_steps'], log_every=50, save_every=recipe['steps'] + 1)
        trainer.train(args, root)
    finally:
        training_module.focused_loss = old_loss
        for name, value in old.items():
            setattr(trainer, name, value)
    run_info = pilot.read_json(target / 'run_config.json')
    verify_identity(request['initial_policy'])
    if run_info.get('status') != 'complete' or run_info['completed_updates'] != recipe['steps']:
        raise RuntimeError('Не завершены все одинаковые training updates.')
    return {'method': request['method'], 'identity': run_info['saved_policy'], 'training_folder': str(target), 'run_config_sha256': pilot.file_hash(target / 'run_config.json'), 'initial_policy': request['initial_policy'], 'dataset': dataset, 'recipe': recipe}

def train(args, root):
    datasets_folder = pilot.local_path(root, args.datasets) if args.datasets else pilot.resolve_pointer(root, DATASETS_POINTER)
    datasets = pilot.read_json(datasets_folder / 'datasets.json')
    if datasets.get('status') != 'complete' or set(datasets.get('datasets', {})) != set(METHODS):
        raise pilot.StageNotReady('Сначала нужны подготовленные datasets для всех трёх методов.')
    collection, info = load_collection(root, datasets['collection'])
    if pilot.file_hash(collection / 'collection.json') != datasets['collection_sha256']:
        raise RuntimeError('Expert collection изменился после преобразования данных.')
    recipe = {'steps': args.steps, 'seed': args.seed, 'lr': 1e-05, 'n_action_steps': 1, 'recovery_supervision_steps': RECOVERY_SUPERVISION, 'weights': list(WEIGHTS), 'checkpoint_selection': 'last fixed update; no test-based checkpoint selection', 'normalization': 'fixed initial-policy processors', 'training_source': 'fixed initial ACT, independently reloaded for every selector'}
    output = pilot.new_folder(root, 'models_selection')
    summary = {'status': 'running', 'recipe': recipe, 'target_budget': info['target_budget'], 'datasets': str(datasets_folder), 'datasets_sha256': pilot.file_hash(datasets_folder / 'datasets.json'), 'collection': str(collection), 'initial_policy': info['source_initial_model'], 'models': {}, 'scope': 'Single-task ACT disagreement/reference-label pilot; not INSIGHT or VLA validation'}
    pilot.write_json(output / 'models.json', summary)
    try:
        for method in METHODS:
            print(f"\n=== TRAIN {method}: B={info['target_budget']}, updates={args.steps}, seed={args.seed} ===", flush=True)
            worker_output = output / method
            request = {'project_root': str(root), 'output': str(worker_output), 'method': method, 'dataset': datasets['datasets'][method], 'initial_policy': info['source_initial_model'], 'budget': info['target_budget'], 'recipe': recipe}
            result = run_child(root, '_train-worker', request, worker_output, live=True)
            summary['models'][method] = result['model']
            pilot.write_json(output / 'models.json', summary)
        summary['status'] = 'complete'
        pilot.write_json(output / 'models.json', summary)
        pilot.write_pointer(root, MODELS_POINTER, output)
    except BaseException as error:
        summary.update(status='interrupted', error=f'{type(error).__name__}: {error}')
        pilot.write_json(output / 'models.json', summary)
        raise
    print('\nTHREE SELECTOR POLICIES READY')
    for method, model in summary['models'].items():
        print(f"{method}: {model['identity']['path']}")
    print('Models:', output / 'models.json')

def internal_worker(args):
    request = pilot.read_json(args.request)
    os.chdir(request['project_root'])
    output = Path(request['output'])
    try:
        if args.command == '_expert-worker':
            result = {'status': 'complete', 'attempt': expert_one(request)}
        else:
            result = {'status': 'complete', 'model': train_one(request)}
        pilot.write_json(output / 'worker_result.json', result)
    except BaseException as error:
        pilot.write_json(output / 'worker_result.json', {'status': 'error', 'error': str(error), 'traceback': traceback.format_exc()})
        raise

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--project-root', type=Path, default=Path(__file__).resolve().parent)
    commands = parser.add_subparsers(dest='command', required=True)
    collection = commands.add_parser('expert', help='Expert continuation для selected candidates; одинаковый budget.')
    collection.add_argument('--plan', type=Path)
    collection.add_argument('--max-expert-steps', type=positive_int, default=280)
    preparation = commands.add_parser('prepare', help='Отдельный dataset только выбранных demos для каждого метода.')
    preparation.add_argument('--collection', type=Path)
    training = commands.add_parser('train', help='Три копии исходной ACT, один общий training recipe.')
    training.add_argument('--datasets', type=Path)
    training.add_argument('--steps', type=positive_int, default=1000)
    training.add_argument('--seed', type=int, default=0)
    for name in ('_expert-worker', '_train-worker'):
        command = commands.add_parser(name, help=argparse.SUPPRESS)
        command.add_argument('--request', type=Path, required=True)
    args = parser.parse_args()
    root = args.project_root.expanduser().resolve()
    os.chdir(root)
    if args.command.startswith('_'):
        internal_worker(args)
    else:
        {'expert': expert, 'prepare': prepare, 'train': train}[args.command](args, root)
if __name__ == '__main__':
    try:
        main()
    except pilot.StageNotReady as error:
        print(f'\nERROR: {error}', file=sys.stderr)
        sys.exit(2)
