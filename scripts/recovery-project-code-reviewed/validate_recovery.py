#!/usr/bin/env python3
from __future__ import annotations
import argparse
import csv
from datetime import datetime
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import runpy
import subprocess
import sys
import traceback
import numpy as np

class Captured(Exception):
    pass

class SkipCase(Exception):
    pass

def positive_int(value):
    result = int(value)
    if result < 1:
        raise argparse.ArgumentTypeError('Нужно положительное целое число.')
    return result

def positive_float(value):
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise argparse.ArgumentTypeError('Нужно положительное конечное число.')
    return result

def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8')

def local_path(root, value):
    path = Path(value).expanduser()
    return (path if path.is_absolute() else root / path).resolve()

def pointer_path(root, filename):
    value = (root / 'pilot_runs' / filename).read_text().strip()
    if not value:
        raise RuntimeError(f'Пустой указатель: {filename}')
    return local_path(root, value)

def new_folder(root, prefix):
    path = root / 'pilot_runs/heldout_recovery' / (prefix + datetime.now().strftime('_%Y%m%d_%H%M%S_%f'))
    path.mkdir(parents=True, exist_ok=False)
    return path

def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:

        def read_chunk():
            return stream.read(1024 * 1024)
        for chunk in iter(read_chunk, b''):
            digest.update(chunk)
    return digest.hexdigest()

def array_hash(value):
    value = np.ascontiguousarray(value, dtype=np.float64)
    return hashlib.sha256(value.tobytes()).hexdigest()

def model_identity(path):
    path = path.resolve()
    required = ('config.json', 'model.safetensors', 'policy_preprocessor.json', 'policy_postprocessor.json')
    for name in required:
        if not (path / name).is_file():
            raise FileNotFoundError(path / name)
    config = json.loads((path / 'config.json').read_text())
    if config.get('type') != 'act':
        raise RuntimeError(f"Ожидается ACT, получено {config.get('type')}: {path}")
    names = set(required)
    for pattern in ('policy_preprocessor*', 'policy_postprocessor*'):
        names.update((p.name for p in path.glob(pattern) if p.is_file()))
    source_items_1 = sorted(names)
    items_1 = {}
    for name in source_items_1:
        items_1[name] = file_hash(path / name)
    return {'path': str(path), 'sha256': items_1}

def verify_model(identity):
    path = Path(identity['path'])
    for name, expected in identity['sha256'].items():
        if file_hash(path / name) != expected:
            raise RuntimeError(f'Модель или processors изменились после фиксации пула: {path / name}')

def runtime_versions():
    result = {'python': sys.version.split()[0]}
    for package in ('lerobot', 'hf-libero', 'gymnasium', 'torch', 'numpy'):
        try:
            result[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            result[package] = None
    return result

def wrapper_nodes(vec):
    queue, seen = ([vec.envs[0]], set())
    while queue:
        obj = queue.pop(0)
        if id(obj) in seen:
            continue
        seen.add(id(obj))
        yield obj
        for name in ('env', '_env', 'unwrapped'):
            child = getattr(obj, name, None)
            if child is not None:
                queue.append(child)

def choose_initial_state(vec, init_id, excluded):
    for obj in wrapper_nodes(vec):
        attributes = vars(obj)
        if 'init_state_id' in attributes and '_init_states' in attributes:
            states = attributes['_init_states']
            if states is None or not 0 <= init_id < len(states):
                raise RuntimeError(f'LIBERO init-state ID {init_id} вне доступного диапазона.')
            selected = np.asarray(states[init_id], dtype=np.float64)
            if not np.isfinite(selected).all():
                raise RuntimeError('LIBERO init state содержит неконечные значения.')
            for other_id in excluded:
                other = np.asarray(states[other_id], dtype=np.float64)
                if selected.shape == other.shape and np.allclose(selected, other, atol=1e-08, rtol=0):
                    raise SkipCase(f'init-state ID {init_id} совпадает с исключённым ID {other_id}.')
            obj.init_state_id = init_id
            return array_hash(selected)
    raise RuntimeError('Не найден LeRobot LiberoEnv с init_state_id и _init_states; нужен текущий исходник envs/libero.py.')

def find_raw(vec):
    for obj in wrapper_nodes(vec):
        if getattr(obj, 'robots', None) and getattr(obj, 'sim', None) is not None:
            return obj
    raise RuntimeError('Не найден robot environment.')

def geometry(vec):
    raw = find_raw(vec)
    sim = raw.sim
    names = list(sim.model.body_names)
    bowl = sim.data.body_xpos[names.index('akita_black_bowl_1_main')].copy()
    plate = sim.data.body_xpos[names.index('plate_1_main')].copy()
    site_id = sim.model.site_name2id(raw.robots[0].controller_config['eef_name'])
    eef = sim.data.site_xpos[site_id].copy()
    return {'simulator_time': float(sim.data.time), 'bowl_pos': bowl.tolist(), 'plate_pos': plate.tolist(), 'eef_pos': eef.tolist(), 'bowl_plate_distance_xy': float(np.linalg.norm(bowl[:2] - plate[:2]))}

def state_and_ctrl(vec):
    sim = find_raw(vec).sim
    state = sim.get_state().flatten().copy()
    ctrl = sim.data.ctrl.copy()
    if not np.isfinite(state).all() or not np.isfinite(ctrl).all():
        raise RuntimeError('Симулятор содержит неконечные значения.')
    return (state, ctrl)

def exact_error(actual, expected, label):
    if actual.shape != expected.shape:
        raise RuntimeError(f'Размер {label} изменился: {actual.shape} / {expected.shape}')
    error = float(np.max(np.abs(actual - expected)))
    if error > 1e-08:
        raise RuntimeError(f'Replay не воспроизвёл {label}: max error={error:.12g}')
    return error

def stopped(result):
    return bool(np.any(result[2]) or np.any(result[3]))

def observed_success(result):
    info = result[4]
    if isinstance(info, dict) and 'is_success' in info:
        if np.any(info['is_success']):
            return True
    return bool(np.any(np.asarray(result[1]) > 0))

def checked_action(action):
    action = np.asarray(action).copy()
    if action.shape != (1, 7) or not np.isfinite(action).all():
        raise RuntimeError(f'Ожидался конечный action (1, 7), получено {action.shape}')
    return action

class CollectionHooks:

    def __init__(self, request, original_reset, original_step):
        self.request, self.reset_base, self.step_base = (request, original_reset, original_step)
        self.protocol = request['protocol']
        self.output = Path(request['output'])
        self.vectors, self.actions, self.drop_rows, self.frames = ([], [], [], [])
        self.reset_state = self.reset_ctrl = None
        self.initial_geometry = None
        self.init_hash = None
        self.nominal_steps = 0
        self.metadata = None

    def reset(self, vec, *args, **kwargs):
        self.vectors.append(vec)
        self.init_hash = choose_initial_state(vec, self.request['init_state_id'], self.protocol['excluded_init_state_ids'])
        result = self.reset_base(vec, *args, **kwargs)
        self.reset_state, self.reset_ctrl = state_and_ctrl(vec)
        self.initial_geometry = geometry(vec)
        self.actions.clear()
        self.nominal_steps = 0
        return result

    def step(self, vec, action):
        action = checked_action(action)
        if self.reset_state is None:
            raise RuntimeError('Получен step до reset.')
        if self.nominal_steps >= self.protocol['nominal_horizon_steps']:
            raise SkipCase('Лимит nominal rollout: требуемый подъём не наблюдался.')
        before = geometry(vec)
        rise = before['bowl_pos'][2] - self.initial_geometry['bowl_pos'][2]
        boundary = self.nominal_steps % self.protocol['n_action_steps'] == 0
        if boundary and rise >= self.protocol['lift_threshold_m']:
            self.capture_drop(vec, before)
            raise Captured()
        result = self.step_base(vec, action)
        self.actions.append(action)
        self.nominal_steps += 1
        if stopped(result):
            reason = 'Задача завершена до вмешательства.' if observed_success(result) else 'Nominal rollout завершился до подъёма.'
            raise SkipCase(reason)
        return result

    def capture_drop(self, vec, before):
        print(f"CAPTURE init_id={self.request['init_state_id']} step={self.nominal_steps} bowl_z={before['bowl_pos'][2]:.4f}", flush=True)
        for _ in range(self.protocol['open_steps']):
            action = np.zeros((1, 7), dtype=np.float32)
            action[0, -1] = -1
            result = self.step_base(vec, action)
            self.actions.append(action.copy())
            current = geometry(vec)
            self.drop_rows.append(current)
            obs = result[0]
            if isinstance(obs, dict) and 'pixels' in obs and ('image' in obs['pixels']):
                self.frames.append(np.asarray(obs['pixels']['image'])[0].copy())
            if stopped(result) or observed_success(result):
                raise SkipCase('Задача или эпизод завершились во время принудительного открывания.')
        after = geometry(vec)
        fall = before['bowl_pos'][2] - after['bowl_pos'][2]
        if fall < self.protocol['drop_threshold_m']:
            raise SkipCase(f'Падение не подтверждено: изменение высоты {fall:.4f} м.')
        saved, ctrl = state_and_ctrl(vec)
        for reference in self.protocol['training_recovery_states']:
            with np.load(reference, allow_pickle=False) as data:
                training = data['simulator_state']
                if saved.shape == training.shape and np.allclose(saved[1:], training[1:], atol=1e-08, rtol=0):
                    raise SkipCase('Полученное физическое состояние совпало с обучающим recovery start.')
        prefix = np.stack(self.actions)
        np.savez_compressed(self.output / 'start.npz', simulator_state=saved, ctrl=ctrl, reset_state=self.reset_state, reset_ctrl=self.reset_ctrl, past_actions=prefix)
        self.metadata = {'status': 'accepted', 'init_state_id': self.request['init_state_id'], 'init_definition_sha256': self.init_hash, 'nominal_steps': self.nominal_steps, 'prefix_steps': len(prefix), 'fall_m': fall, 'initial_geometry': self.initial_geometry, 'before_drop': before, 'after_drop': after, 'failure_protocol': 'forced_open_gripper_after_verified_height_rise', 'held_out_from_recorded_recovery_demos': True, 'recovery_outcome_used_for_collection': False}
        write_json(self.output / 'case.json', self.metadata)
        write_json(self.output / 'intervention_geometry.json', self.drop_rows)

class EvaluationHooks:

    def __init__(self, request, original_reset, original_step):
        self.request, self.reset_base, self.step_base = (request, original_reset, original_step)
        self.protocol = request['protocol']
        self.output = Path(request['output'])
        self.vectors, self.rows = ([], [])
        self.restore = None
        with np.load(Path(request['case_dir']) / 'start.npz', allow_pickle=False) as data:
            source_items_2 = ('simulator_state', 'ctrl', 'reset_state', 'reset_ctrl', 'past_actions')
            items_2 = {}
            for key in source_items_2:
                items_2[key] = data[key].copy()
            self.saved = items_2
        prefix = self.saved['past_actions']
        if prefix.ndim != 3 or prefix.shape[1:] != (1, 7) or (not np.isfinite(prefix).all()):
            raise RuntimeError('Неожиданный сохранённый action prefix.')

    def reset(self, vec, *args, **kwargs):
        self.vectors.append(vec)
        actual_hash = choose_initial_state(vec, self.request['init_state_id'], self.protocol['excluded_init_state_ids'])
        if actual_hash != self.request['init_definition_sha256']:
            raise RuntimeError('Определение LIBERO initial state изменилось после collection.')
        obs, info = self.reset_base(vec, *args, **kwargs)
        state, ctrl = state_and_ctrl(vec)
        reset_error = exact_error(state, self.saved['reset_state'], 'reset state')
        reset_ctrl_error = exact_error(ctrl, self.saved['reset_ctrl'], 'reset ctrl')
        for action in self.saved['past_actions']:
            result = self.step_base(vec, action)
            obs, info = (result[0], result[4])
            if stopped(result) or observed_success(result):
                raise RuntimeError('Эпизод завершился при replay до recovery start.')
        state, ctrl = state_and_ctrl(vec)
        state_error = exact_error(state, self.saved['simulator_state'], 'recovery state')
        ctrl_error = exact_error(ctrl, self.saved['ctrl'], 'recovery ctrl')
        self.restore = {'max_reset_state_error': reset_error, 'max_reset_ctrl_error': reset_ctrl_error, 'max_state_error': state_error, 'max_ctrl_error': ctrl_error, 'prefix_steps': len(self.saved['past_actions']), 'init_state_id': self.request['init_state_id'], 'recovery_horizon_steps': self.protocol['recovery_horizon_steps']}
        write_json(self.output / 'restore_check.json', self.restore)
        self.rows.clear()
        print(f'RESTORE OK: state_error={state_error:.3g}, ctrl_error={ctrl_error:.3g}', flush=True)
        return (obs, info)

    def step(self, vec, action):
        action = checked_action(action)
        if self.restore is None:
            raise RuntimeError('Получен step без проверенного восстановления.')
        if len(self.rows) >= self.protocol['recovery_horizon_steps']:
            raise RuntimeError('Evaluator продолжил эпизод после лимита recovery.')
        before = geometry(vec)
        result = self.step_base(vec, action)
        after = geometry(vec)
        obs, reward, terminated, truncated, info = result
        at_limit = len(self.rows) + 1 == self.protocol['recovery_horizon_steps']
        if at_limit and (not stopped(result)):
            truncated = np.ones_like(np.asarray(truncated), dtype=bool)
        self.rows.append({'step': len(self.rows), 'action': action[0].tolist(), 'before': before, 'after': after, 'reward': np.asarray(reward).reshape(-1).tolist(), 'terminated': bool(np.any(terminated)), 'truncated': bool(np.any(truncated)), 'fixed_recovery_horizon_reached': at_limit})
        return (obs, reward, terminated, truncated, info)

def official_evaluation(request, hooks):
    from gymnasium.vector import SyncVectorEnv
    protocol = request['protocol']
    if request['stage'] == 'collect':
        horizon = protocol['nominal_horizon_steps'] + protocol['open_steps']
    else:
        horizon = len(hooks.saved['past_actions']) + protocol['recovery_horizon_steps']
    old_reset, old_step, old_argv = (SyncVectorEnv.reset, SyncVectorEnv.step, sys.argv)
    sys.argv = ['lerobot-eval', f"--policy.path={request['model_path']}", '--policy.device=cuda', f"--policy.n_action_steps={protocol['n_action_steps']}", '--env.type=libero', '--env.task=libero_spatial', '--env.task_ids=[0]', '--env.control_mode=relative', '--env.init_states=true', '--env.hard_reset=true', '--env.max_parallel_tasks=1', '--env.observation_height=256', '--env.observation_width=256', f'--env.episode_length={horizon}', '--eval.batch_size=1', '--eval.n_episodes=1', '--eval.use_async_envs=false', f"--seed={protocol['seed']}", f"--output_dir={Path(request['output']) / 'evaluation'}"]
    try:

        def reset_environment(vec, *args, **kwargs):
            return hooks.reset(vec, *args, **kwargs)
        SyncVectorEnv.reset = reset_environment

        def step_environment(vec, action):
            return hooks.step(vec, action)
        SyncVectorEnv.step = step_environment
        runpy.run_module('lerobot.scripts.lerobot_eval', run_name='__main__')
    finally:
        SyncVectorEnv.reset, SyncVectorEnv.step, sys.argv = (old_reset, old_step, old_argv)
        source_items_3 = hooks.vectors
        items_3 = {}
        for vec in source_items_3:
            items_3[id(vec)] = vec
        for vec in items_3.values():
            if not getattr(vec, 'closed', False):
                vec.close()

def worker(request_path):
    request = json.loads(request_path.read_text())
    output = Path(request['output'])
    os.chdir(request['project_root'])
    os.environ['MUJOCO_GL'] = 'egl'
    from gymnasium.vector import SyncVectorEnv
    original_reset, original_step = (SyncVectorEnv.reset, SyncVectorEnv.step)
    hooks = None
    try:
        hooks = (CollectionHooks if request['stage'] == 'collect' else EvaluationHooks)(request, original_reset, original_step)
        try:
            official_evaluation(request, hooks)
        except Captured:
            if request['stage'] != 'collect' or hooks.metadata is None:
                raise RuntimeError('Некорректное завершение collection.')
        if request['stage'] == 'collect':
            if hooks.metadata is None:
                raise SkipCase('Rollout завершён без подходящего подъёма миски.')
            if hooks.frames:
                from PIL import Image
                source_items_4 = hooks.frames
                frames = []
                for frame in source_items_4:
                    frames.append(Image.fromarray(np.ascontiguousarray(frame[::-1, ::-1], dtype=np.uint8)))
                frames[0].save(output / 'intervention.gif', save_all=True, append_images=frames[1:], duration=50, loop=0)
            result = hooks.metadata
        else:
            info = json.loads((output / 'evaluation/eval_info.json').read_text())
            overall = info['overall']
            if overall['n_episodes'] != 1 or hooks.restore is None or (not hooks.rows):
                raise RuntimeError('Evaluator не сохранил одну полную проверенную recovery episode.')
            result = {'status': 'evaluated', 'success': bool(overall['n_success'] == 1), 'steps': len(hooks.rows), 'videos': overall.get('video_paths', []), 'model': request['model_path'], 'variant': request['variant'], 'init_state_id': request['init_state_id'], 'restore': hooks.restore}
        write_json(output / 'worker_result.json', result)
    except SkipCase as exc:
        if request['stage'] != 'collect':
            write_json(output / 'worker_result.json', {'status': 'error', 'error': str(exc)})
            raise
        write_json(output / 'worker_result.json', {'status': 'skipped', 'init_state_id': request['init_state_id'], 'reason': str(exc)})
    except Exception as exc:
        write_json(output / 'worker_result.json', {'status': 'error', 'error': str(exc), 'traceback': traceback.format_exc()})
        raise
    finally:
        if isinstance(hooks, EvaluationHooks) and hooks.rows:
            write_json(output / 'rollout_trace.json', {'steps': hooks.rows, 'variant': request['variant'], 'case_dir': request['case_dir'], 'n_action_steps': request['protocol']['n_action_steps'], 'geometry_note': 'Height/contact diagnostics alone do not establish stable grasp; success is the LIBERO goal outcome.'})

def run_worker(root, request):
    output = Path(request['output'])
    output.mkdir(parents=True, exist_ok=False)
    request_path = output / 'request.json'
    write_json(request_path, request)
    with (output / 'worker.log').open('w', encoding='utf-8') as log:
        completed = subprocess.run([sys.executable, str(Path(__file__).resolve()), '_worker', '--request', str(request_path)], cwd=root, stdout=log, stderr=subprocess.STDOUT)
    result_path = output / 'worker_result.json'
    if completed.returncode != 0 or not result_path.is_file():
        lines = (output / 'worker.log').read_text(errors='replace').splitlines()
        print('\n'.join(lines[-45:]), flush=True)
        raise RuntimeError(f"Проверка остановлена; полный лог: {output / 'worker.log'}")
    result = json.loads(result_path.read_text())
    if result.get('status') not in ('accepted', 'skipped', 'evaluated'):
        raise RuntimeError(f'Неожиданный результат worker: {result}')
    return result

def collect(args, root):
    manifest = json.loads((root / 'pilot_runs/manifest.json').read_text())
    if manifest['suite'] != 'libero_spatial' or manifest['task_id'] != 0:
        raise RuntimeError('Этот pilot предназначен для libero_spatial task_id=0.')
    initial = root / 'pilot_runs/training/initial/checkpoints/last/pretrained_model'
    adapted = local_path(root, args.policy_path) if args.policy_path else pointer_path(root, 'corrected_policy_path.txt')
    references = [root / 'pilot_runs/drop_step50/perturbed_start.npz']
    correction_pointer = root / 'pilot_runs/last_correction_demo.txt'
    if correction_pointer.exists():
        references.append(pointer_path(root, 'last_correction_demo.txt') / 'start.npz')
    for reference in references:
        if not reference.is_file():
            raise FileNotFoundError(reference)
    source_items_5 = references
    items_5 = []
    for p in source_items_5:
        items_5.append(str(p.resolve()))
    protocol = {'suite': 'libero_spatial', 'task_id': 0, 'instruction': manifest['instruction'], 'n_action_steps': args.n_action_steps, 'seed': args.seed, 'excluded_init_state_ids': [0], 'first_init_state_id': args.start_id, 'requested_count': args.count, 'max_attempts': args.max_attempts, 'lift_threshold_m': args.lift_threshold, 'drop_threshold_m': args.drop_threshold, 'open_steps': args.open_steps, 'nominal_horizon_steps': 280, 'recovery_horizon_steps': args.horizon, 'training_recovery_states': items_5, 'models': {'initial': model_identity(initial), 'recovery': model_identity(adapted)}, 'runtime_versions': runtime_versions(), 'scope': 'New forced-drop starts in the same task, excluded from the two recorded recovery demonstrations; not necessarily unseen in D0.', 'selection_rule': 'First eligible distinct layouts in ID order; no filter by autonomous recovery success.', 'cases': [], 'attempts': [], 'complete': False}
    output = new_folder(root, 'pool')
    write_json(output / 'protocol.json', protocol)
    hashes = set()
    for attempt in range(args.max_attempts):
        init_id = args.start_id + attempt
        case_dir = output / f'init_{init_id:03d}'
        print(f'\nCOLLECT init_id={init_id}: initial ACT; lift; gripper opening', flush=True)
        request = {'stage': 'collect', 'protocol': protocol, 'project_root': str(root), 'output': str(case_dir), 'init_state_id': init_id, 'model_path': protocol['models']['initial']['path']}
        result = run_worker(root, request)
        if result['status'] == 'accepted':
            if result['init_definition_sha256'] in hashes:
                result['status'], result['reason'] = ('skipped', 'Повтор стартовой конфигурации в пуле.')
                write_json(case_dir / 'case.json', result)
                write_json(case_dir / 'worker_result.json', result)
            else:
                hashes.add(result['init_definition_sha256'])
                protocol['cases'].append({'case_dir': str(case_dir), 'init_state_id': init_id, 'init_definition_sha256': result['init_definition_sha256'], 'snapshot_sha256': file_hash(case_dir / 'start.npz'), 'metadata_sha256': file_hash(case_dir / 'case.json')})
                print(f"Accepted {len(protocol['cases'])}/{args.count}: fall {result['fall_m']:.4f} m; prefix={result['prefix_steps']}", flush=True)
        if result['status'] == 'skipped':
            print('Skipped:', result['reason'], flush=True)
        protocol['attempts'].append(result)
        protocol['complete'] = len(protocol['cases']) == args.count
        write_json(output / 'protocol.json', protocol)
        if protocol['complete']:
            break
    if not protocol['complete']:
        raise RuntimeError(f"Получено {len(protocol['cases'])}/{args.count} состояний. Пул сохранён в {output}; оценка не запускается на неполном пуле.")
    (root / 'pilot_runs/last_heldout_pool.txt').write_text(str(output))
    print('\nPOOL READY:', output)
    print('New starts:', len(protocol['cases']))

def wilson_interval(successes, total):
    z = 1.959963984540054
    p = successes / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    radius = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return [max(0, center - radius), min(1, center + radius)]

def paired_summary(rows):
    total = len(rows)
    if not total:
        raise RuntimeError('Нет завершённых парных проверок.')
    result = {'n_states': total, 'models': {}, 'paired_outcomes': {'both_success': sum((r['initial_success'] and r['recovery_success'] for r in rows)), 'both_fail': sum((not r['initial_success'] and (not r['recovery_success']) for r in rows)), 'adapted_only_success': sum((not r['initial_success'] and r['recovery_success'] for r in rows)), 'initial_only_success': sum((r['initial_success'] and (not r['recovery_success']) for r in rows))}}
    for variant in ('initial', 'recovery'):
        count = sum((r[f'{variant}_success'] for r in rows))
        result['models'][variant] = {'n_success': count, 'n_episodes': total, 'pc_success': 100 * count / total, 'sr_wilson95': wilson_interval(count, total)}
    result['difference_percentage_points'] = result['models']['recovery']['pc_success'] - result['models']['initial']['pc_success']
    result['note'] = 'Small single-task pilot; Wilson intervals are for each success rate, not a CI for the paired difference. No F-versus-U comparison is performed.'
    return result

def evaluate(args, root):
    pool = local_path(root, args.pool) if args.pool else pointer_path(root, 'last_heldout_pool.txt')
    protocol = json.loads((pool / 'protocol.json').read_text())
    if not protocol['complete'] or len(protocol['cases']) != protocol['requested_count']:
        raise RuntimeError('Пул не завершён; сначала выполните collect.')
    if runtime_versions() != protocol['runtime_versions']:
        raise RuntimeError('Версии среды изменились после collection. Нужен новый воспроизводимый пул.')
    for identity in protocol['models'].values():
        verify_model(identity)
    for case in protocol['cases']:
        folder = Path(case['case_dir'])
        if file_hash(folder / 'start.npz') != case['snapshot_sha256'] or file_hash(folder / 'case.json') != case['metadata_sha256']:
            raise RuntimeError(f'Сохранённый case изменился: {folder}')
    output = new_folder(root, 'evaluation')
    write_json(output / 'run_info.json', {'pool': str(pool), 'protocol': protocol})
    rows = []
    for index, case in enumerate(protocol['cases'], start=1):
        results = {}
        for variant in ('initial', 'recovery'):
            print(f"\nEVALUATE {index}/{len(protocol['cases'])}, init_id={case['init_state_id']}, {variant}", flush=True)
            request = {'stage': 'evaluate', 'protocol': protocol, 'project_root': str(root), 'output': str(output / f"init_{case['init_state_id']:03d}" / variant), 'case_dir': case['case_dir'], 'init_state_id': case['init_state_id'], 'init_definition_sha256': case['init_definition_sha256'], 'model_path': protocol['models'][variant]['path'], 'variant': variant}
            result = run_worker(root, request)
            results[variant] = result
            print(f"{variant}: success={result['success']}; steps={result['steps']}; restore error={result['restore']['max_state_error']:.3g}", flush=True)
        rows.append({'init_state_id': case['init_state_id'], 'initial_success': results['initial']['success'], 'recovery_success': results['recovery']['success'], 'initial_steps': results['initial']['steps'], 'recovery_steps': results['recovery']['steps'], 'initial_video': next(iter(results['initial']['videos']), ''), 'recovery_video': next(iter(results['recovery']['videos']), '')})
        write_json(output / 'paired_results.json', rows)
        with (output / 'paired_results.csv').open('w', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    summary = paired_summary(rows)
    write_json(output / 'summary.json', summary)
    (root / 'pilot_runs/last_heldout_evaluation.txt').write_text(str(output))
    print('\n=== NEW RECOVERY STATES ===')
    for variant, result in summary['models'].items():
        print(f"{variant}: {result['n_success']}/{result['n_episodes']}")
    print(f"Difference: {summary['difference_percentage_points']:+.1f} percentage points")
    print('Output:', output)

def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--project-root', type=Path, default=Path(__file__).resolve().parent)
    sub = parser.add_subparsers(dest='command', required=True)
    collection = sub.add_parser('collect', help='Сохранить новые recovery starts с помощью исходной ACT.')
    collection.add_argument('--count', type=positive_int, default=5)
    collection.add_argument('--start-id', type=positive_int, default=10)
    collection.add_argument('--max-attempts', type=positive_int, default=10)
    collection.add_argument('--policy-path', type=Path, help='Дообученная модель, которая будет зафиксирована до оценки; default=corrected_policy_path.txt.')
    collection.add_argument('--n-action-steps', type=positive_int, default=10)
    collection.add_argument('--seed', type=int, default=0)
    collection.add_argument('--lift-threshold', type=positive_float, default=0.06)
    collection.add_argument('--drop-threshold', type=positive_float, default=0.03)
    collection.add_argument('--open-steps', type=positive_int, default=12)
    collection.add_argument('--horizon', type=positive_int, default=280)
    evaluation = sub.add_parser('evaluate', help='Парная оценка двух зафиксированных policies.')
    evaluation.add_argument('--pool', type=Path, help='Default=last_heldout_pool.txt.')
    internal = sub.add_parser('_worker', help='Служебный запуск; используется автоматически.')
    internal.add_argument('--request', type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'collect' and args.max_attempts < args.count:
        parser.error('--max-attempts не может быть меньше --count.')
    return args

def main():
    args = parse_args()
    if args.command == '_worker':
        worker(args.request.resolve())
        return
    root = args.project_root.resolve()
    os.chdir(root)
    if args.command == 'collect':
        collect(args, root)
    else:
        evaluate(args, root)
if __name__ == '__main__':
    main()
