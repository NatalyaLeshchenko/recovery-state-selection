#!/usr/bin/env python3
from __future__ import annotations
import argparse
from collections import Counter, deque
from datetime import datetime
import gc
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time
import traceback
import numpy as np
os.environ.setdefault('MUJOCO_GL', 'egl')
FAILURE_TYPES = ('missed_grasp', 'object_dropped')
RESERVED_TEST_IDS = (15, 16, 17, 18, 19, 45, 46, 47, 48, 49)
POOL_POINTER = 'selection_pool_path.txt'
SCORE_POINTER = 'selection_scores_path.txt'
INSTRUCTION = 'pick up the black bowl between the plate and the ramekin and place it on the plate'
RAW_SHAPES = {'observation/pixels/image': (1, 256, 256, 3), 'observation/pixels/image2': (1, 256, 256, 3), 'observation/robot_state/eef/pos': (1, 3), 'observation/robot_state/eef/quat': (1, 4), 'observation/robot_state/gripper/qpos': (1, 2)}

class StageNotReady(RuntimeError):
    pass

def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))

def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    temporary.replace(path)

def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:

        def read_chunk():
            return stream.read(1024 * 1024)
        for block in iter(read_chunk, b''):
            digest.update(block)
    return digest.hexdigest()

def local_path(root, path):
    path = Path(path).expanduser()
    return (path if path.is_absolute() else root / path).resolve()

def resolve_pointer(root, filename):
    path = root / 'pilot_runs' / filename
    if not path.is_file():
        hints = {POOL_POINTER: 'Пул ещё не готов. Сначала выполните collect и дождитесь POOL READY с обоими типами ошибок.', SCORE_POINTER: 'Uncertainty scores ещё не рассчитаны. Сначала завершите collect, затем score.', 'uncertainty_ensemble_path.txt': 'Не найден готовый nominal ensemble. Сначала завершите train_uncertainty_ensemble.py train.'}
        raise StageNotReady(hints.get(filename, f'Не найден указатель: {path}'))
    value = path.read_text(encoding='utf-8').strip()
    if not value:
        raise RuntimeError(f'Пустой указатель: {path}')
    return local_path(root, value)

def write_pointer(root, filename, path):
    target = root / 'pilot_runs' / filename
    temporary = target.with_suffix('.tmp')
    temporary.write_text(str(path) + '\n', encoding='utf-8')
    temporary.replace(target)

def new_folder(root, prefix):
    path = root / 'pilot_runs/selection' / datetime.now().strftime(prefix + '_%Y%m%d_%H%M%S_%f')
    path.mkdir(parents=True, exist_ok=False)
    return path

def load_ensemble(root, value=None):
    from train_uncertainty_ensemble import policy_settings, verify_identity
    folder = local_path(root, value) if value else resolve_pointer(root, 'uncertainty_ensemble_path.txt')
    record = read_json(folder / 'ensemble.json')
    models = record.get('models', [])
    if record.get('complete') is not True or record.get('status') != 'complete' or record.get('smoke_only') is not False or (len(models) != 3):
        raise RuntimeError('Нужен завершённый ensemble из трёх nominal ACT, не smoke run.')
    if models[0].get('role') != 'fixed_initial_actor_and_member0':
        raise RuntimeError('Не определена общая исходная ACT для collection и fine-tuning.')
    source_items_1 = models
    items_1 = set()
    for m in source_items_1:
        items_1.add(m['seed'])
    if len(items_1) != 3:
        raise RuntimeError('Seeds nominal members должны различаться.')
    source_items_2 = models
    items_2 = set()
    for m in source_items_2:
        items_2.add(m['identity']['sha256']['model.safetensors'])
    if len(items_2) != 3:
        raise RuntimeError('Weights nominal members должны различаться.')
    initial_settings = None
    for model in models:
        verify_identity(model['identity'])
        cfg = read_json(Path(model['identity']['path']) / 'config.json')
        settings = policy_settings(cfg)
        if initial_settings is None:
            initial_settings = settings
        if cfg.get('type') != 'act' or settings != initial_settings:
            raise RuntimeError('Архитектура и training settings members должны совпадать.')
    if file_hash(root / 'pilot_runs/manifest.json') != record['manifest_sha256']:
        raise RuntimeError('Исходный D0 manifest изменился после обучения ensemble.')
    return (folder, record)

def raw_observation(obs):
    result = {}
    for key, expected in RAW_SHAPES.items():
        value = obs
        for part in key.split('/')[1:]:
            value = value[part]
        value = np.asarray(value)
        if value.shape != expected or not np.isfinite(value).all():
            raise RuntimeError(f'Некорректное observation: {key}, shape={value.shape}')
        if '/pixels/' in key and value.dtype != np.uint8:
            raise RuntimeError(f'RGB должен иметь dtype uint8: {key}')
        result[key] = value.copy()
    return result

def descendant_ids(model, root_id):
    parents = np.asarray(model.body_parentid)
    result = {int(root_id)}
    for body in range(len(parents)):
        node, seen = (body, set())
        while node not in seen:
            if node == root_id:
                result.add(body)
                break
            seen.add(node)
            parent = int(parents[node])
            if parent == node:
                break
            node = parent
    return result

def finger_contacts(vec):
    from validate_recovery import find_raw
    sim = find_raw(vec).sim
    names = list(sim.model.body_names)
    roots = {'bowl': names.index('akita_black_bowl_1_main'), 'left': names.index('gripper0_leftfinger'), 'right': names.index('gripper0_rightfinger')}
    source_items_3 = roots.items()
    groups = {}
    for key, body in source_items_3:
        groups[key] = descendant_ids(sim.model, body)
    geom_bodies = np.asarray(sim.model.geom_bodyid)
    sides = set()
    for index in range(int(sim.data.ncon)):
        contact = sim.data.contact[index]
        bodies = {int(geom_bodies[contact.geom1]), int(geom_bodies[contact.geom2])}
        if not bodies & groups['bowl']:
            continue
        for side in ('left', 'right'):
            if bodies & groups[side]:
                sides.add(side)
    return {'left': 'left' in sides, 'right': 'right' in sides, 'bilateral': {'left', 'right'} <= sides}

def eligible_miss(intended, before, initial, held_ever, cfg):
    rise = before['bowl_pos'][2] - initial['bowl_pos'][2]
    distance = float(np.linalg.norm(np.asarray(before['eef_pos']) - before['bowl_pos']))
    return not held_ever and rise < cfg['miss_max_bowl_rise_m'] and (intended[0, -1] >= cfg['close_command_threshold']) and (distance <= cfg['miss_trigger_distance_m'])

def check_miss_event(initial, before, event_rows, cfg):
    after = event_rows[-1]['geometry']
    bowl_rise = max((row['geometry']['bowl_pos'][2] for row in event_rows)) - initial['bowl_pos'][2]
    eef_rise = after['eef_pos'][2] - before['eef_pos'][2]
    moved_xy = float(np.linalg.norm(np.asarray(after['bowl_pos'][:2]) - before['bowl_pos'][:2]))
    no_hold = not any((row['contacts']['bilateral'] for row in event_rows))
    valid = bowl_rise < cfg['miss_max_bowl_rise_m'] and eef_rise >= cfg['miss_min_empty_lift_m'] and (moved_xy <= cfg['miss_max_bowl_shift_m']) and no_hold
    return (valid, {'max_bowl_rise_m': bowl_rise, 'empty_eef_lift_m': eef_rise, 'bowl_shift_xy_m': moved_xy, 'no_bilateral_hold_during_attempt': no_hold})

def check_drop_event(before, event_rows, held_ever, cfg):
    after = event_rows[-1]['geometry']
    fall = before['bowl_pos'][2] - after['bowl_pos'][2]
    lost_hold = not event_rows[-1]['contacts']['bilateral']
    valid = held_ever and fall >= cfg['drop_threshold_m'] and lost_hold
    return (valid, {'fall_m': fall, 'held_before_intervention': held_ever, 'bilateral_hold_lost': lost_hold, 'release_was_forced_before_task_success': True})

class PoolHooks:

    def __init__(self, request, original_reset, original_step):
        self.request, self.reset_base, self.step_base = (request, original_reset, original_step)
        self.protocol = request['protocol']
        self.output = Path(request['output'])
        self.vectors, self.actions, self.trace, self.event_rows = ([], [], [], [])
        self.history = deque(maxlen=self.protocol['history_observations'])
        self.frames = []
        self.metadata = None
        self.nominal_steps = 0
        self.hold_streak = 0
        self.held_ever = False
        self.last_observation = None

    def remember(self, obs):
        self.last_observation = raw_observation(obs)
        self.history.append(self.last_observation)

    def reset(self, vec, *args, **kwargs):
        from validate_recovery import choose_initial_state, geometry, state_and_ctrl
        self.vectors.append(vec)
        self.init_hash = choose_initial_state(vec, self.request['init_state_id'], self.protocol['excluded_init_state_ids'])
        result = self.reset_base(vec, *args, **kwargs)
        self.reset_state, self.reset_ctrl = state_and_ctrl(vec)
        self.initial_geometry = geometry(vec)
        self.actions.clear()
        self.trace.clear()
        self.history.clear()
        self.nominal_steps = self.hold_streak = 0
        self.held_ever = False
        self.remember(result[0])
        return result

    def step(self, vec, action):
        from validate_recovery import Captured, SkipCase, checked_action, geometry, observed_success, stopped
        action = checked_action(action)
        if self.nominal_steps >= self.protocol['nominal_horizon_steps']:
            raise SkipCase('Nominal rollout не достиг условий вмешательства за 280 шагов.')
        before = geometry(vec)
        contacts = finger_contacts(vec)
        rise = before['bowl_pos'][2] - self.initial_geometry['bowl_pos'][2]
        previous_close = bool(self.actions and self.actions[-1][0, -1] > 0)
        if contacts['bilateral'] and previous_close and (rise >= self.protocol['held_min_bowl_rise_m']):
            self.hold_streak += 1
        else:
            self.hold_streak = 0
        self.held_ever = self.held_ever or self.hold_streak >= self.protocol['held_consecutive_observations']
        self.trace.append({'nominal_step': self.nominal_steps, 'geometry': before, 'contacts': contacts, 'intended_action': action[0].tolist(), 'held_streak': self.hold_streak, 'held_ever': self.held_ever})
        kind = self.request['failure_type']
        if kind == 'missed_grasp' and eligible_miss(action, before, self.initial_geometry, self.held_ever, self.protocol):
            self.intervene(vec, action, before)
            raise Captured()
        if kind == 'object_dropped' and self.held_ever and contacts['bilateral'] and (rise >= self.protocol['lift_threshold_m']):
            self.intervene(vec, action, before)
            raise Captured()
        result = self.step_base(vec, action)
        self.actions.append(action.copy())
        self.nominal_steps += 1
        self.remember(result[0])
        if stopped(result) or observed_success(result):
            raise SkipCase('Эпизод завершился до подходящего вмешательства.')
        return result

    def intervene(self, vec, intended, before):
        from validate_recovery import SkipCase, geometry, observed_success, state_and_ctrl, stopped
        kind = self.request['failure_type']
        print(f"INTERVENTION {kind}, init_id={self.request['init_state_id']}, nominal_step={self.nominal_steps}", flush=True)
        source_items_4 = self.history
        items_4 = []
        for item in source_items_4:
            items_4.append(item['observation/pixels/image'][0].copy())
        self.frames = items_4
        if kind == 'missed_grasp':
            blocked = intended.copy()
            blocked[0, -1] = -1.0
            lift = np.zeros((1, 7), dtype=np.float32)
            lift[0, 2] = self.protocol['miss_lift_action']
            lift[0, -1] = -1.0
            source_items_5 = range(self.protocol['miss_lift_steps'])
            items_5 = []
            for _ in source_items_5:
                items_5.append(lift.copy())
            intervention = [blocked] + items_5
        else:
            opened = np.zeros((1, 7), dtype=np.float32)
            opened[0, -1] = -1.0
            source_items_6 = range(self.protocol['open_steps'])
            intervention = []
            for _ in source_items_6:
                intervention.append(opened.copy())
        for action in intervention:
            result = self.step_base(vec, action)
            self.actions.append(action.copy())
            self.remember(result[0])
            self.frames.append(self.last_observation['observation/pixels/image'][0].copy())
            self.event_rows.append({'geometry': geometry(vec), 'contacts': finger_contacts(vec), 'executed_action': action[0].tolist()})
            if stopped(result) or observed_success(result):
                raise SkipCase('Задача или эпизод завершились во время вмешательства.')
        if kind == 'missed_grasp':
            valid, evidence = check_miss_event(self.initial_geometry, before, self.event_rows, self.protocol)
        else:
            valid, evidence = check_drop_event(before, self.event_rows, self.held_ever, self.protocol)
        if not valid:
            raise SkipCase(f'Условия reference label {kind} не выполнены: {evidence}')
        state, ctrl = state_and_ctrl(vec)
        np.savez_compressed(self.output / 'start.npz', simulator_state=state, ctrl=ctrl, reset_state=self.reset_state, reset_ctrl=self.reset_ctrl, past_actions=np.stack(self.actions), **self.last_observation)
        np.savez_compressed(self.output / 'history.npz', **{key: np.concatenate([item[key] for item in self.history], axis=0) for key in RAW_SHAPES})
        self.metadata = {'status': 'accepted', 'candidate_id': self.request['candidate_id'], 'failure_type': kind, 'label_source': 'explicit_simulator_event_rules', 'init_state_id': self.request['init_state_id'], 'init_definition_sha256': self.init_hash, 'nominal_steps': self.nominal_steps, 'prefix_steps': len(self.actions), 'instruction': self.protocol['instruction'], 'initial_geometry': self.initial_geometry, 'before_intervention': before, 'after_intervention': self.event_rows[-1]['geometry'], 'verification': evidence, 'history_observations': len(self.history), 'nominal_help_score_used_for_collection': False, 'autonomous_recovery_outcome_used_for_collection': False, 'scope': 'Induced collection/development state; not an independent evaluation state', 'snapshot_sha256': file_hash(self.output / 'start.npz'), 'history_sha256': file_hash(self.output / 'history.npz')}

def make_vector(protocol, prefix_steps):
    from lerobot.envs.configs import LiberoEnv
    from lerobot.envs.factory import make_env
    cfg = LiberoEnv(task='libero_spatial', task_ids=[0], control_mode='relative', init_states=True, hard_reset=True, max_parallel_tasks=1, observation_height=256, observation_width=256, episode_length=prefix_steps + protocol['recovery_horizon_steps'])
    envs = make_env(cfg, n_envs=1, use_async_envs=False)
    return envs['libero_spatial'][0]

def goal_at_start(vec):
    from validate_recovery import wrapper_nodes
    for obj in wrapper_nodes(vec):
        for name in ('check_success', '_check_success'):
            method = getattr(obj, name, None)
            if callable(method):
                result = np.asarray(method())
                if result.size != 1:
                    raise RuntimeError('Ожидался один LIBERO success flag.')
                return bool(result.item())
    raise RuntimeError('Не найдена проверка LIBERO goal predicate.')

def check_restoration(request, case):
    from validate_recovery import EvaluationHooks
    vec = make_vector(request['protocol'], case['prefix_steps'])
    replay = {**request, 'stage': 'evaluate', 'case_dir': request['output'], 'init_definition_sha256': case['init_definition_sha256']}
    hooks = EvaluationHooks(replay, type(vec).reset, type(vec).step)
    try:
        obs, _ = hooks.reset(vec, seed=request['protocol']['seed'])
        if goal_at_start(vec):
            raise RuntimeError('Сохранённый recovery start уже удовлетворяет цели.')
        actual = raw_observation(obs)
        errors = {}
        with np.load(Path(request['output']) / 'start.npz', allow_pickle=False) as saved:
            for key, value in actual.items():
                difference = float(np.max(np.abs(value.astype(np.float64) - saved[key].astype(np.float64))))
                errors[key] = difference
                tolerance = 0 if '/pixels/' in key else 1e-06
                if difference > tolerance:
                    raise RuntimeError(f'Replay observation mismatch: {key}, max error={difference}')
        return {**hooks.restore, 'observation_errors': errors, 'goal_at_start': False}
    finally:
        vec.close()

def worker(path):
    request = read_json(path)
    os.chdir(request['project_root'])
    output = Path(request['output'])
    from gymnasium.vector import SyncVectorEnv
    from validate_recovery import Captured, SkipCase, official_evaluation
    hooks = PoolHooks(request, SyncVectorEnv.reset, SyncVectorEnv.step)
    try:
        try:
            evaluator_request = {**request, 'protocol': {**request['protocol'], 'open_steps': max(request['protocol']['open_steps'], 1 + request['protocol']['miss_lift_steps'])}}
            official_evaluation(evaluator_request, hooks)
        except Captured:
            if hooks.metadata is None:
                raise RuntimeError('Capture завершён без snapshot.')
        if hooks.metadata is None:
            raise SkipCase('Нет подходящего recovery start.')
        result = dict(hooks.metadata)
        result['restore_check'] = check_restoration(request, result)
        write_json(output / 'case.json', result)
        write_json(output / 'worker_result.json', result)
        print(f"ACCEPTED {result['candidate_id']}; prefix={result['prefix_steps']}; replay OK", flush=True)
    except SkipCase as error:
        write_json(output / 'worker_result.json', {'status': 'skipped', 'reason': str(error), 'candidate_id': request['candidate_id'], 'failure_type': request['failure_type']})
    except Exception as error:
        write_json(output / 'worker_result.json', {'status': 'error', 'error': str(error), 'traceback': traceback.format_exc()})
        raise
    finally:
        write_json(output / 'event_trace.json', {'nominal': hooks.trace, 'intervention': hooks.event_rows})
        if hooks.frames:
            from PIL import Image
            source_items_7 = hooks.frames
            images = []
            for frame in source_items_7:
                images.append(Image.fromarray(np.ascontiguousarray(frame[::-1, ::-1])))
            images[0].save(output / 'intervention.gif', save_all=True, append_images=images[1:], duration=50, loop=0)

def run_worker(root, request):
    output = Path(request['output'])
    output.mkdir(parents=True, exist_ok=False)
    request_path = output / 'request.json'
    write_json(request_path, request)
    child_env = dict(os.environ, OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', WANDB_MODE='disabled', HF_HUB_DISABLE_TELEMETRY='1')
    with (output / 'worker.log').open('w', encoding='utf-8') as log:
        completed = subprocess.run([sys.executable, str(Path(__file__).resolve()), '_worker', '--request', str(request_path)], cwd=root, env=child_env, stdout=log, stderr=subprocess.STDOUT)
    result_path = output / 'worker_result.json'
    if completed.returncode != 0 or not result_path.is_file():
        print('\n'.join((output / 'worker.log').read_text(errors='replace').splitlines()[-45:]), flush=True)
        raise RuntimeError(f"Collection остановлен; полный лог: {output / 'worker.log'}")
    result = read_json(result_path)
    if result.get('status') not in ('accepted', 'skipped'):
        raise RuntimeError(f'Неожиданный результат worker: {result}')
    return result

def collect(args, root):
    from validate_recovery import runtime_versions
    if len(args.init_ids) != len(set(args.init_ids)) or any((i < 0 for i in args.init_ids)):
        raise ValueError('Нужны различные неотрицательные --init-ids.')
    if set(args.init_ids) & set(RESERVED_TEST_IDS):
        raise ValueError(f'Для отдельной будущей test evaluation зарезервированы IDs {RESERVED_TEST_IDS}; не используйте их для collection.')
    ensemble_folder, ensemble = load_ensemble(root, args.ensemble)
    manifest = read_json(root / 'pilot_runs/manifest.json')
    if manifest.get('suite') != 'libero_spatial' or manifest.get('task_id') != 0 or manifest.get('instruction') != INSTRUCTION:
        raise RuntimeError('Этот pilot рассчитан на libero_spatial task 0: чёрная миска между тарелкой и ramekin.')
    output = new_folder(root, 'pool')
    protocol = {'collection_protocol_version': 2, 'suite': 'libero_spatial', 'task_id': 0, 'instruction': INSTRUCTION, 'seed': 0, 'n_action_steps': 1, 'control_frequency_hz': 20, 'nominal_horizon_steps': 280, 'recovery_horizon_steps': 280, 'excluded_init_state_ids': list(RESERVED_TEST_IDS), 'init_state_ids': args.init_ids, 'reserved_independent_recovery_test_ids': list(RESERVED_TEST_IDS), 'test_split_scope': 'Held out from this selection/training experiment; may be familiar from nominal D0', 'failure_types': list(FAILURE_TYPES), 'history_observations': 48, 'held_min_bowl_rise_m': 0.02, 'held_consecutive_observations': 3, 'lift_threshold_m': 0.06, 'drop_threshold_m': 0.03, 'open_steps': 12, 'miss_trigger_distance_m': 0.1, 'close_command_threshold': 0.5, 'miss_max_bowl_rise_m': 0.015, 'miss_min_empty_lift_m': 0.035, 'miss_max_bowl_shift_m': 0.03, 'miss_lift_action': 0.35, 'miss_lift_steps': 24, 'ensemble_folder': str(ensemble_folder), 'ensemble_sha256': file_hash(ensemble_folder / 'ensemble.json'), 'models': {'initial': ensemble['models'][0]['identity']}, 'runtime_versions': runtime_versions(), 'manifest_sha256': file_hash(root / 'pilot_runs/manifest.json'), 'collection_rule': 'First eligible event per requested layout and intervention type; no uncertainty or recovery-outcome filtering', 'scope': 'Single-task induced-error collection/development pool; ACT proxy; not INSIGHT, VLA, or a test set', 'label_definitions': {'missed_grasp': 'Policy close command near target; closure blocked; no established prior lifted hold; empty lift and bowl remains on table', 'object_dropped': 'Consecutive bilateral finger contact while target lifted; forced open before goal; target falls >=3cm and bilateral hold lost'}, 'attempts': [], 'cases': [], 'complete': False}
    write_json(output / 'protocol.json', protocol)
    print('Pool:', output, flush=True)
    from train_uncertainty_ensemble import verify_identity
    for init_id in args.init_ids:
        for kind in FAILURE_TYPES:
            candidate_id = f'init_{init_id:03d}_{kind}'
            print(f'\nCOLLECT {candidate_id}: initial ACT; controlled intervention', flush=True)
            verify_identity(protocol['models']['initial'])
            request = {'stage': 'collect', 'protocol': protocol, 'project_root': str(root), 'output': str(output / candidate_id), 'candidate_id': candidate_id, 'failure_type': kind, 'init_state_id': init_id, 'model_path': protocol['models']['initial']['path']}
            try:
                result = run_worker(root, request)
            except Exception:
                protocol['status'] = 'error'
                write_json(output / 'protocol.json', protocol)
                raise
            protocol['attempts'].append(result)
            if result['status'] == 'accepted':
                protocol['cases'].append(result)
                print('Accepted:', kind, result['verification'], flush=True)
            else:
                print('Skipped:', result['reason'], flush=True)
            write_json(output / 'protocol.json', protocol)
    counts = Counter((case['failure_type'] for case in protocol['cases']))
    protocol['complete'] = all((counts[k] > 0 for k in FAILURE_TYPES))
    protocol['status'] = 'complete' if protocol['complete'] else 'insufficient_failure_types'
    protocol['counts'] = dict(counts)
    write_json(output / 'protocol.json', protocol)
    if protocol['complete']:
        write_pointer(root, POOL_POINTER, output)
    print('\nPOOL READY' if protocol['complete'] else 'POOL INCOMPLETE: missing failure type')
    print('Failure types:', dict(counts))
    print('Output:', output)
    if protocol['complete']:
        pass
    else:
        raise StageNotReady('Неполный пул сохранён для диагностики. score и select требуют обоих failure types.')

def load_pool(root, value=None):
    pool = local_path(root, value) if value else resolve_pointer(root, POOL_POINTER)
    protocol = read_json(pool / 'protocol.json')
    if protocol.get('complete') is not True or protocol.get('status') != 'complete':
        raise StageNotReady('Пул collection не завершён или не содержит два failure types. Сначала нужен POOL READY.')
    cases = protocol['cases']
    source_items_8 = cases
    items_8 = set()
    for case in source_items_8:
        items_8.add(case['candidate_id'])
    if len(items_8) != len(cases):
        raise RuntimeError('Повторяющиеся candidate IDs.')
    if set((case['failure_type'] for case in cases)) != set(FAILURE_TYPES):
        raise RuntimeError('Нужны оба reference failure types.')
    for case in cases:
        if case.get('label_source') != 'explicit_simulator_event_rules':
            raise RuntimeError('Нужны проверенные reference event labels.')
        directory = pool / case['candidate_id']
        for name, digest in (('start.npz', case['snapshot_sha256']), ('history.npz', case['history_sha256'])):
            if file_hash(directory / name) != digest:
                raise RuntimeError(f'Изменились candidate данные: {directory / name}')
    return (pool, protocol)

def compare_normalization(model_paths, torch):
    from safetensors.torch import load_file
    reference = None
    for folder in model_paths:
        files = sorted(folder.glob('*normalizer*.safetensors'))
        if not files:
            raise RuntimeError(f'Не найдены сохранённые normalization stats: {folder}')
        source_items_9 = files
        current = {}
        for path in source_items_9:
            current[path.name] = load_file(str(path), device='cpu')
        if reference is None:
            reference = current
        if set(current) != set(reference):
            raise RuntimeError('Normalization processor files различаются между nominal members.')
        for name, tensors in current.items():
            if set(tensors) != set(reference[name]):
                raise RuntimeError('Normalization tensor keys различаются между nominal members.')
            for key, value in tensors.items():
                expected = reference[name][key]
                if value.shape != expected.shape or not torch.equal(value, expected):
                    raise RuntimeError(f'Нет общей D0 normalization: {folder / name}, tensor={key}')
    source_items_10 = reference.items()
    items_10 = {}
    for name, tensors in source_items_10:
        items_10[name] = sorted(tensors)
    return items_10

def policy_observation(saved, processor, torch):
    raw = {'observation.images.image': torch.from_numpy(saved['observation/pixels/image'].copy()).permute(0, 3, 1, 2), 'observation.images.image2': torch.from_numpy(saved['observation/pixels/image2'].copy()).permute(0, 3, 1, 2), 'observation.robot_state': {'eef': {'pos': torch.from_numpy(saved['observation/robot_state/eef/pos'].copy()), 'quat': torch.from_numpy(saved['observation/robot_state/eef/quat'].copy())}, 'gripper': {'qpos': torch.from_numpy(saved['observation/robot_state/gripper/qpos'].copy())}}}
    processed = processor.observation(raw)
    for key in ('observation.images.image', 'observation.images.image2'):
        processed[key] = processed[key].float() / 255.0
    expected = {'observation.state', 'observation.images.image', 'observation.images.image2'}
    if set(processed) != expected or tuple(processed['observation.state'].shape) != (1, 8):
        raise RuntimeError('Неожиданные ACT inputs после LIBERO processor.')
    return processed

def disagreement(predictions):
    value = np.asarray(predictions, dtype=np.float64)
    if value.ndim != 4 or value.shape[0] < 2 or value.shape[2:] != (1, 7) or (not np.isfinite(value).all()):
        raise ValueError('Нужны конечные predictions (M, N, 1, 7), M>=2.')
    mean = value.mean(axis=0, keepdims=True)
    return np.square(value - mean).mean(axis=(0, 2, 3))

def score(args, root):
    pool, protocol = load_pool(root, args.pool)
    from train_uncertainty_ensemble import verify_identity
    import torch
    from lerobot.policies.act.modeling_act import ACTPolicy
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.processor.env_processor import LiberoProcessorStep
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA недоступна; scoring не запущен.')
    ensemble_folder, ensemble = load_ensemble(root, args.ensemble or protocol['ensemble_folder'])
    if file_hash(ensemble_folder / 'ensemble.json') != protocol['ensemble_sha256'] or ensemble['models'][0]['identity'] != protocol['models']['initial']:
        raise RuntimeError('Ensemble или общая исходная ACT изменились после collection.')
    source_items_11 = ensemble['models']
    paths = []
    for model in source_items_11:
        paths.append(Path(model['identity']['path']))
    shared_stats = compare_normalization(paths, torch)
    output = new_folder(root, 'scores')
    started = time.perf_counter()
    record = {'status': 'running', 'pool': str(pool), 'pool_protocol_sha256': file_hash(pool / 'protocol.json'), 'ensemble': str(ensemble_folder), 'ensemble_sha256': protocol['ensemble_sha256'], 'members': ensemble['models'], 'normalization': shared_stats, 'score_horizon': 1, 'score_definition': 'Mean squared ensemble disagreement in shared D0-normalized action space, first action only', 'scope': 'ACT disagreement proxy; not INSIGHT, help probability, or demonstrated epistemic uncertainty', 'models_loaded_sequentially': True, 'expert_actions_used': False, 'cases': []}
    write_json(output / 'scores.json', record)
    predictions = []
    for member, folder in zip(ensemble['models'], paths):
        verify_identity(member['identity'])
        print(f"\nSCORE nominal ACT seed={member['seed']}: {len(protocol['cases'])} states", flush=True)
        torch.manual_seed(0)
        policy = ACTPolicy.from_pretrained(folder, local_files_only=True, strict=True)
        if policy.config.n_obs_steps != 1 or policy.config.temporal_ensemble_coeff is not None:
            raise RuntimeError('Нужна текущая ACT: n_obs_steps=1, без temporal ensemble.')
        expected = {'observation.state', 'observation.images.image', 'observation.images.image2'}
        if set(policy.config.input_features) != expected:
            raise RuntimeError('Неожиданные observation inputs ACT.')
        policy.config.n_action_steps = 1
        policy.to('cuda').eval()
        preprocessor, postprocessor = make_pre_post_processors(policy.config, pretrained_path=folder)
        processor = LiberoProcessorStep()
        values = []
        with torch.inference_mode():
            for case in protocol['cases']:
                path = pool / case['candidate_id'] / 'start.npz'
                with np.load(path, allow_pickle=False) as saved:
                    batch = policy_observation(saved, processor, torch)
                batch = preprocessor(batch)
                source_items_12 = batch.items()
                batch = {}
                for key, value in source_items_12:
                    batch[key] = value.to('cuda') if torch.is_tensor(value) else value
                policy.reset()
                normalized = policy.predict_action_chunk(batch)
                if normalized.shape != (1, policy.config.chunk_size, 7):
                    raise RuntimeError(f'Неожиданный action chunk: {tuple(normalized.shape)}')
                values.append(normalized[0, :1].detach().cpu().numpy().copy())
                print(f"  {case['candidate_id']}", flush=True)
        predictions.append(np.stack(values))
        verify_identity(member['identity'])
        del policy, preprocessor, postprocessor, processor, batch, normalized
        gc.collect()
        torch.cuda.empty_cache()
    predictions = np.stack(predictions)
    scores = disagreement(predictions)
    source_items_13 = protocol['cases']
    items_13 = []
    for case in source_items_13:
        items_13.append(case['candidate_id'])
    np.savez_compressed(output / 'predictions.npz', normalized_actions=predictions, candidate_ids=np.asarray(items_13))
    for case, value in zip(protocol['cases'], scores):
        record['cases'].append({'candidate_id': case['candidate_id'], 'init_state_id': case['init_state_id'], 'failure_type': case['failure_type'], 'U': float(value), 'snapshot_sha256': case['snapshot_sha256']})
    record.update(status='complete', scoring_seconds=time.perf_counter() - started, predictions_sha256=file_hash(output / 'predictions.npz'))
    load_pool(root, pool)
    if file_hash(pool / 'protocol.json') != record['pool_protocol_sha256']:
        raise RuntimeError('Pool protocol изменился во время scoring.')
    write_json(output / 'scores.json', record)
    write_pointer(root, SCORE_POINTER, output)
    print('\nUNCERTAINTY SCORES READY')

    def uncertainty_order(row):
        return -row['U']
    for row in sorted(record['cases'], key=uncertainty_order):
        print(f"{row['candidate_id']}: U={row['U']:.6g}")
    print('Output:', output)

def balanced_quotas(counts, budget, seed):
    if budget < 1 or budget > sum(counts.values()) or any((count < 0 for count in counts.values())):
        raise ValueError('Budget должен лежать между 1 и размером candidate pool.')
    rng = random.Random(seed)
    source_items_14 = sorted(counts)
    quotas = {}
    for kind in source_items_14:
        quotas[kind] = 0
    for _ in range(budget):
        source_items_15 = sorted(counts)
        available = []
        for kind in source_items_15:
            if quotas[kind] < counts[kind]:
                available.append(kind)
        minimum = min((quotas[kind] for kind in available))
        source_items_16 = available
        tied = []
        for kind in source_items_16:
            if quotas[kind] == minimum:
                tied.append(kind)
        quotas[rng.choice(tied)] += 1
    return quotas

def query_plan(rows, budget, seed):
    source_items_17 = rows
    ids = []
    for row in source_items_17:
        ids.append(row['candidate_id'])
    if len(ids) != len(set(ids)) or not 1 <= budget <= len(ids):
        raise ValueError('Нужны различные кандидаты и budget <= размер пула.')
    if any((row['failure_type'] not in FAILURE_TYPES or not np.isfinite(row['U']) for row in rows)):
        raise ValueError('Некорректные labels или uncertainty scores.')
    base = sorted(ids)
    rng = random.Random(seed)
    random_order = rng.sample(base, len(base))
    tie_order = rng.sample(base, len(base))
    source_items_18 = enumerate(tie_order)
    tie_rank = {}
    for index, candidate in source_items_18:
        tie_rank[candidate] = index

    def uncertainty_order_2(row):
        return (-row['U'], tie_rank[row['candidate_id']])
    ordered_u = sorted(rows, key=uncertainty_order_2)
    groups = {kind: sorted((row['candidate_id'] for row in rows if row['failure_type'] == kind)) for kind in FAILURE_TYPES}
    source_items_19 = groups.items()
    items_19 = {}
    for kind, group in source_items_19:
        items_19[kind] = len(group)
    quotas = balanced_quotas(items_19, budget, seed)
    source_items_20 = groups.items()
    queues = {}
    for kind, group in source_items_20:
        queues[kind] = rng.sample(group, len(group))
    source_items_21 = FAILURE_TYPES
    balanced = []
    for kind in source_items_21:
        for candidate in queues[kind][:quotas[kind]]:
            balanced.append(candidate)
    source_items_22 = rows
    by_id = {}
    for row in source_items_22:
        by_id[row['candidate_id']] = row

    def condition(selected, **extra):
        return {'selected_candidates': selected, 'failure_counts': dict(Counter((by_id[candidate]['failure_type'] for candidate in selected))), **extra}
    source_items_23 = ordered_u[:budget]
    items_23 = []
    for row in source_items_23:
        items_23.append(row['candidate_id'])
    source_items_24 = ordered_u
    items_24 = []
    for row in source_items_24:
        items_24.append(row['candidate_id'])
    return {'U': condition(items_23, query_order=items_24), 'F_ref': condition(balanced, quotas=quotas, query_orders_by_type=queues), 'Random': condition(random_order[:budget], query_order=random_order)}

def select(args, root):
    folder = local_path(root, args.scores) if args.scores else resolve_pointer(root, SCORE_POINTER)
    record = read_json(folder / 'scores.json')
    if record.get('status') != 'complete':
        raise RuntimeError('Scoring не завершён.')
    pool, protocol = load_pool(root, record['pool'])
    if file_hash(pool / 'protocol.json') != record['pool_protocol_sha256']:
        raise RuntimeError('Common pool изменился после scoring.')
    if file_hash(folder / 'predictions.npz') != record['predictions_sha256']:
        raise RuntimeError('Сохранённые predictions изменились.')
    source_items_25 = protocol['cases']
    expected = {}
    for case in source_items_25:
        expected[case['candidate_id']] = case
    if len(record['cases']) != len(expected) or {row['candidate_id'] for row in record['cases']} != set(expected):
        raise RuntimeError('Scoring покрывает другой candidate pool.')
    for row in record['cases']:
        case = expected[row['candidate_id']]
        if row['snapshot_sha256'] != case['snapshot_sha256'] or row['failure_type'] != case['failure_type']:
            raise RuntimeError('Scores или labels не соответствуют candidate snapshots.')
    plan = query_plan(record['cases'], args.budget, args.seed)
    output = new_folder(root, 'selection')
    result = {'pool': str(pool), 'scores': str(folder), 'seed': args.seed, 'target_successful_demonstrations': args.budget, 'conditions': plan, 'scope': 'Initial candidate query plan; not collected demonstrations or measured recovery success', 'quota_rule': 'Repeatedly add one slot to a least-filled nonempty group; seeded ties; exhausted groups skipped', 'collection_rule': 'U/Random advance their fixed order after failed attempts. F_ref tries the next candidate in the same type; if exhausted, redistribute unfilled slots by the same quota rule among remaining available groups. Record every attempt. Stop only after B accepted successes, or report an unmet budget.', 'training_rule': 'All future variants start from the fixed initial ACT; same training parameters and replay. No old recovery-adapted weights.', 'evaluation_rule': 'Use independent common test starts, never states in this collection pool.'}
    write_json(output / 'selection.json', result)
    write_pointer(root, 'selection_plan_path.txt', output)
    print(f'\nSELECTION READY: target B={args.budget}, seed={args.seed}')
    for method, condition in plan.items():
        print(f"{method}: {condition['selected_candidates']}; types={condition['failure_counts']}")
    print('Plan:', output / 'selection.json')

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--project-root', type=Path, default=Path(__file__).resolve().parent)
    commands = parser.add_subparsers(dest='command', required=True)
    collection = commands.add_parser('collect', help='Сохранить два типа induced recovery starts.')
    collection.add_argument('--init-ids', type=int, nargs='+', required=True)
    collection.add_argument('--ensemble', type=Path)
    scoring = commands.add_parser('score', help='Disagreement nominal ensemble на сохранённых observations.')
    scoring.add_argument('--pool', type=Path)
    scoring.add_argument('--ensemble', type=Path)
    selection = commands.add_parser('select', help='Сравнить query plan U / F_ref / Random без обучения.')
    selection.add_argument('--scores', type=Path)
    selection.add_argument('--budget', type=int, required=True)
    selection.add_argument('--seed', type=int, default=0)
    internal = commands.add_parser('_worker', help=argparse.SUPPRESS)
    internal.add_argument('--request', type=Path, required=True)
    args = parser.parse_args()
    root = args.project_root.expanduser().resolve()
    os.chdir(root)
    if args.command == '_worker':
        worker(args.request)
    else:
        {'collect': collect, 'score': score, 'select': select}[args.command](args, root)
if __name__ == '__main__':
    try:
        main()
    except StageNotReady as error:
        print(f'\nERROR: {error}', file=sys.stderr)
        sys.exit(2)
    except Exception as error:
        print(f'\nERROR: {error}', file=sys.stderr)
        raise
