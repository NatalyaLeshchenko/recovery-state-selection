#!/usr/bin/env python3
from __future__ import annotations
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import numpy as np

class GoalReached(Exception):
    pass

def nonnegative_int(value):
    result = int(value)
    if result < 0:
        raise argparse.ArgumentTypeError('Нужно неотрицательное число.')
    return result

def positive_int(value):
    result = nonnegative_int(value)
    if not result:
        raise argparse.ArgumentTypeError('Нужно положительное число.')
    return result

def local_path(root, path):
    path = Path(path).expanduser()
    return (path if path.is_absolute() else root / path).resolve()

def find_robot_env(vec):
    queue = [vec.envs[0]]
    visited = set()
    while queue:
        obj = queue.pop(0)
        if id(obj) in visited:
            continue
        visited.add(id(obj))
        if getattr(obj, 'robots', None):
            return obj
        for attr in ('env', '_env', 'unwrapped'):
            child = getattr(obj, attr, None)
            if child is not None:
                queue.append(child)
    raise RuntimeError('Не найден robot environment.')

def geometry(raw):
    sim = raw.sim
    names = list(sim.model.body_names)
    bowl_id = names.index('akita_black_bowl_1_main')
    plate_id = names.index('plate_1_main')
    site_id = sim.model.site_name2id(raw.robots[0].controller_config['eef_name'])
    return {'simulator_time': float(sim.data.time), 'eef_pos': sim.data.site_xpos[site_id].copy().tolist(), 'bowl_pos': sim.data.body_xpos[bowl_id].copy().tolist(), 'plate_pos': sim.data.body_xpos[plate_id].copy().tolist()}

def geometry_error(actual, expected):
    return max((float(np.max(np.abs(np.asarray(actual[key]) - expected[key]))) for key in ('eef_pos', 'bowl_pos', 'plate_pos')))

def check_geometry(raw, expected, tolerance=1e-06):
    actual = geometry(raw)
    error = geometry_error(actual, expected)
    time_error = abs(actual['simulator_time'] - expected['simulator_time'])
    if error > tolerance or time_error > tolerance:
        raise RuntimeError(f'Replay не совпал с трассой: координаты={error:.3g} м, время={time_error:.3g} с.')
    return error

def flatten_observation(value, path='observation'):
    result = {}
    if isinstance(value, dict):
        for key, child in value.items():
            result.update(flatten_observation(child, f'{path}/{key}'))
    else:
        array = np.asarray(value)
        if array.dtype.kind not in 'biuf' or array.ndim == 0 or array.shape[0] != 1:
            raise RuntimeError(f'Неожиданное наблюдение {path}: {array.shape}, {array.dtype}')
        result[path] = array[0].copy()
    return result

def make_vec(cfg):
    from lerobot.envs.factory import make_env
    return make_env(cfg, n_envs=1, use_async_envs=False)['libero_spatial'][0]

def calibrate(cfg, nominal_actions):
    vec = make_vec(cfg)
    try:
        vec.reset(seed=0)
        raw = find_robot_env(vec)
        grasp_offset = None
        nominal_success = False
        for index, action in enumerate(nominal_actions, start=1):
            _, reward, terminated, truncated, _ = vec.step(action)
            current = geometry(raw)
            if index == 42:
                grasp_offset = np.asarray(current['eef_pos']) - current['bowl_pos']
            nominal_success = nominal_success or bool(np.any(np.asarray(reward) > 0))
            if np.any(terminated) or np.any(truncated):
                break
        if grasp_offset is None or not nominal_success:
            raise RuntimeError('Калибровочный rollout не достиг цели либо не дошёл до шага 42.')
        placement_offset = np.asarray(current['bowl_pos']) - current['plate_pos']
        return (grasp_offset, placement_offset)
    finally:
        vec.close()

def restore_start(vec, prefix, saved_state, trace_rows, takeover_step):
    obs, _ = vec.reset(seed=0)
    raw = find_robot_env(vec)
    for action in prefix:
        obs, _, terminated, truncated, _ = vec.step(action)
        if np.any(terminated) or np.any(truncated):
            raise RuntimeError('Эпизод завершился до сохранённого падения.')
    error = float(np.max(np.abs(raw.sim.get_state().flatten() - saved_state)))
    print(f'Restore error: {error:.12g}', flush=True)
    if error > 1e-08:
        raise RuntimeError('Не удалось воспроизвести исходное состояние после падения.')
    max_geometry_error = check_geometry(raw, trace_rows[0]['before'])
    replayed = []
    for index in range(takeover_step):
        row = trace_rows[index]
        max_geometry_error = max(max_geometry_error, check_geometry(raw, row['before']))
        action = np.asarray(row['action'], dtype=np.float32)[None, :]
        obs, reward, terminated, truncated, _ = vec.step(action)
        if np.any(np.asarray(reward) > 0) or np.any(terminated) or np.any(truncated):
            raise RuntimeError('Rollout завершился до выбранной точки передачи управления.')
        max_geometry_error = max(max_geometry_error, check_geometry(raw, row['after']))
        replayed.append(action.copy())
    max_geometry_error = max(max_geometry_error, check_geometry(raw, trace_rows[takeover_step]['before']))
    full_prefix = np.concatenate((prefix, np.asarray(replayed).reshape(-1, 1, 7)), axis=0)
    return (obs, raw, full_prefix, {'max_state_error': error, 'max_geometry_error_m': max_geometry_error})

def run_expert(vec, raw, obs, grasp_offset, placement_offset, max_steps, records):
    scale = np.asarray(raw.robots[0].controller.output_max[:3], dtype=float)
    if scale.shape != (3,) or np.any(scale <= 0):
        raise RuntimeError('Неожиданная шкала OSC controller.')
    start_bowl_z = geometry(raw)['bowl_pos'][2]

    def eef_pos():
        return np.asarray(geometry(raw)['eef_pos'])

    def bowl_pos():
        return np.asarray(geometry(raw)['bowl_pos'])

    def step(action, phase):
        nonlocal obs
        if len(records['actions']) >= max_steps:
            raise RuntimeError('Достигнут лимит действий expert.')
        before = flatten_observation(obs)
        if records['observations'] and set(before) != set(records['observations']):
            raise RuntimeError('Состав наблюдений изменился.')
        action = np.asarray(action, dtype=np.float32).reshape(1, 7)
        before_geometry = geometry(raw)
        obs, reward, terminated, truncated, _ = vec.step(action)
        for key, value in before.items():
            records['observations'].setdefault(key, []).append(value)
        records['actions'].append(action[0].copy())
        records['rewards'].append(float(np.asarray(reward).reshape(-1)[0]))
        records['phases'].append(phase)
        records['geometry'].append(before_geometry)
        records['success'] = bool(np.any(np.asarray(reward) > 0))
        records['ended'] = bool(np.any(terminated) or np.any(truncated))
        if len(records['actions']) % 20 == 0 or records['success']:
            print(f"Expert steps={len(records['actions'])}; phase={phase}; success={records['success']}", flush=True)
        if records['success']:
            raise GoalReached()
        if records['ended']:
            raise RuntimeError('Эпизод завершился без достижения цели.')

    def hold(gripper, steps, phase):
        for _ in range(steps):
            action = np.zeros(7)
            action[-1] = gripper
            step(action, phase)

    def move_to(target, gripper, phase, max_move_steps=40):
        target = np.asarray(target)
        for _ in range(max_move_steps):
            error = target - eef_pos()
            if np.linalg.norm(error) < 0.008:
                return
            action = np.zeros(7)
            action[:3] = np.clip(0.6 * error / scale, -0.35, 0.35)
            action[-1] = gripper
            step(action, phase)
        if np.linalg.norm(target - eef_pos()) >= 0.008:
            raise RuntimeError(f'Захват не достиг позиции: {phase}.')
    print('PHASE: approach_grasp', flush=True)
    hold(-1, 6, 'open_before_grasp')
    grasp_target = bowl_pos() + grasp_offset
    print('Target displacement (m):', (grasp_target - eef_pos()).round(5), flush=True)
    move_to(grasp_target, -1, 'approach_grasp')
    print('PHASE: close_gripper_and_lift', flush=True)
    hold(1, 12, 'close_gripper')
    lift_target = eef_pos()
    lift_target[2] += 0.1
    move_to(lift_target, 1, 'lift')
    rise = float(bowl_pos()[2] - start_bowl_z)
    records['bowl_rise_m'] = rise
    print('Bowl rise (m):', rise, flush=True)
    if rise < 0.04:
        raise RuntimeError('Миска не поднялась: попытка не принимается как успешная.')
    print('PHASE: transport', flush=True)
    desired_bowl = np.asarray(geometry(raw)['plate_pos']) + placement_offset
    placement_target = desired_bowl + (eef_pos() - bowl_pos())
    above_plate = placement_target.copy()
    above_plate[2] = max(eef_pos()[2], placement_target[2] + 0.1)
    move_to(above_plate, 1, 'transport')
    print('PHASE: lower_and_release', flush=True)
    move_to(placement_target, 1, 'lower')
    hold(-1, 12, 'release')
    hold(-1, 20, 'settle')
    raise RuntimeError('Expert завершил последовательность, но benchmark не подтвердил успех.')

def save_attempt(output, records, metadata, make_gif):
    source_items_1 = records['observations'].items()
    arrays = {}
    for key, values in source_items_1:
        arrays[key] = np.stack(values)
    if records['actions']:
        arrays.update(action=np.stack(records['actions']), reward=np.asarray(records['rewards'], dtype=np.float32), phase=np.asarray(records['phases']))
        filename = 'demonstration.npz' if records['success'] else 'failed_attempt.npz'
        np.savez_compressed(output / filename, **arrays)
    metadata.update(success=records['success'], accepted=records['success'], n_frames=len(records['actions']), episode_ended=records['ended'], bowl_rise_m=records.get('bowl_rise_m'))
    (output / 'metadata.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    (output / 'expert_geometry.json').write_text(json.dumps(records['geometry'], indent=2) + '\n')
    if make_gif and records['actions']:
        try:
            from PIL import Image
            images = next((values for key, values in records['observations'].items() if '/pixels/' in key and np.asarray(values[0]).shape == (256, 256, 3)))
            source_items_2 = images
            frames = []
            for frame in source_items_2:
                frames.append(Image.fromarray(np.flip(frame, axis=(0, 1)).copy().astype(np.uint8)))
            frames[0].save(output / 'correction.gif', save_all=True, append_images=frames[1:], duration=50, loop=0)
        except Exception as error:
            print('GIF write failed:', error, flush=True)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root', type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument('--trace-path', type=Path, help='Recovery rollout_trace.json; default uses last_recovery_comparison.txt.')
    parser.add_argument('--takeover-step', type=nonnegative_int, default=85)
    parser.add_argument('--max-expert-steps', type=positive_int, default=200)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--no-gif', action='store_true')
    args = parser.parse_args()
    root = args.project_root.resolve()
    os.chdir(root)
    os.environ['MUJOCO_GL'] = 'egl'
    from lerobot.envs.configs import LiberoEnv
    manifest = json.loads((root / 'pilot_runs/manifest.json').read_text())
    if manifest['suite'] != 'libero_spatial' or manifest['task_id'] != 0:
        raise RuntimeError('Этот expert рассчитан на libero_spatial, task_id=0.')
    trace_path = args.trace_path
    if trace_path is None:
        comparison = local_path(root, (root / 'pilot_runs/last_recovery_comparison.txt').read_text().strip())
        trace_path = comparison / 'recovery/rollout_trace.json'
    trace_path = local_path(root, trace_path)
    trace = json.loads(trace_path.read_text())
    rows = trace['steps']
    if trace.get('variant') != 'recovery' or not 0 <= args.takeover_step < len(rows):
        raise RuntimeError('Нужна трасса recovery ACT и существующий индекс takeover-step.')
    for index, row in enumerate(rows):
        if row['step'] != index or np.asarray(row['action']).shape != (7,):
            raise RuntimeError('Трасса содержит пропуски шагов или неверную форму действий.')
    with np.load(root / 'pilot_runs/drop_step50/perturbed_start.npz', allow_pickle=False) as data:
        prefix = data['past_actions'].copy()
        saved_state = data['simulator_state'].copy()
    with np.load(root / 'pilot_runs/lift_trace/actions.npz', allow_pickle=False) as data:
        nominal_actions = data['actions'].copy()
    if prefix.ndim != 3 or prefix.shape[1:] != (1, 7):
        raise RuntimeError('Неожиданная форма записанного префикса действий.')
    horizon = max(len(prefix) + args.takeover_step + args.max_expert_steps + 50, len(nominal_actions) + 50)
    cfg = LiberoEnv(task='libero_spatial', task_ids=[0], episode_length=horizon, observation_height=256, observation_width=256)
    grasp_offset, placement_offset = calibrate(cfg, nominal_actions)
    print('Grasp offset (m):', grasp_offset, flush=True)
    output = args.output_dir or Path('pilot_runs/recovery_demos') / datetime.now().strftime(f'correction_before_step{args.takeover_step}_%Y%m%d_%H%M%S_%f')
    output = local_path(root, output)
    output.mkdir(parents=True, exist_ok=False)
    metadata = {'suite': 'libero_spatial', 'task_id': 0, 'instruction': manifest['instruction'], 'seed': 0, 'failure_type': 'object_dropped', 'failure_source': 'forced_gripper_open_after_step50_then_recovery_ACT', 'collection_reason': 'Correct the incomplete regasp in a policy-visited state', 'expert': 'scripted_controller_with_privileged_simulator_geometry', 'trace_path': str(trace_path), 'source_policy': trace.get('model'), 'takeover_step': args.takeover_step, 'takeover_timing': 'before_trace_action', 'control_frequency_hz': 20, 'observation_timing': 'before_action', 'max_expert_steps': args.max_expert_steps, 'collection_episode_length': horizon, 'grasp_offset': grasp_offset.tolist(), 'placement_offset': placement_offset.tolist(), 'evaluation_note': 'A training-data diagnostic, not held-out recovery evaluation'}
    records = {'observations': {}, 'actions': [], 'rewards': [], 'phases': [], 'geometry': [], 'success': False, 'ended': False}
    vec = None
    error = None
    try:
        vec = make_vec(cfg)
        obs, raw, full_prefix, restore_info = restore_start(vec, prefix, saved_state, rows, args.takeover_step)
        metadata['restore_check'] = restore_info
        current = geometry(raw)
        metadata['start_geometry'] = current
        np.savez_compressed(output / 'start.npz', simulator_state=raw.sim.get_state().flatten().copy(), ctrl=raw.sim.data.ctrl.copy(), past_actions=full_prefix)
        print(f"REPLAY OK: max_position_error={restore_info['max_geometry_error_m']:.3g} m", flush=True)
        print(f'Expert takeover before ACT step {args.takeover_step}.', flush=True)
        run_expert(vec, raw, obs, grasp_offset, placement_offset, args.max_expert_steps, records)
    except GoalReached:
        pass
    except Exception as failure:
        error = failure
        metadata['error'] = f'{type(failure).__name__}: {failure}'
        print('\nSTOP:', failure, flush=True)
    finally:
        try:
            if vec is not None:
                vec.close()
        finally:
            save_attempt(output, records, metadata, not args.no_gif)
    print('\nSUCCESS:', records['success'], flush=True)
    print('Expert actions:', len(records['actions']), flush=True)
    print('Output:', output, flush=True)
    if records['success']:
        (root / 'pilot_runs/last_correction_demo.txt').write_text(str(output) + '\n')
        print('CORRECTION DEMONSTRATION SAVED', flush=True)
    elif error is not None:
        raise SystemExit(1)
if __name__ == '__main__':
    main()
