#!/usr/bin/env python3
from __future__ import annotations
import argparse
from datetime import datetime
import inspect
import json
import os
from pathlib import Path
os.environ.setdefault('MUJOCO_GL', 'egl')
import numpy as np
from validate_recovery import EvaluationHooks, file_hash, geometry, local_path, observed_success, positive_int, runtime_versions, stopped, wrapper_nodes, write_json

def passive_action():
    action = np.zeros((1, 7), dtype=np.float32)
    action[0, -1] = -1.0
    return action

def benchmark_goal(vec):
    for obj in wrapper_nodes(vec):
        for name in ('check_success', '_check_success'):
            check = getattr(obj, name, None)
            if callable(check):
                result = np.asarray(check())
                if result.size != 1:
                    raise RuntimeError('Ожидался один benchmark success flag.')
                return bool(result.item())
    return None

def capture_frame(obs, frames):
    pixels = obs.get('pixels', {}) if isinstance(obs, dict) else {}
    image = pixels.get('image') if isinstance(pixels, dict) else None
    if image is None:
        return
    image = np.asarray(image)
    if image.ndim != 4 or image.shape[0] != 1 or image.shape[-1] != 3:
        raise RuntimeError(f'Неожиданная форма camera RGB: {image.shape}')
    if image.dtype != np.uint8:
        raise RuntimeError('Ожидался RGB uint8 из симулятора.')
    frames.append(image[0].copy())

def save_gif(frames, path, frequency_hz):
    if not frames:
        return
    from PIL import Image
    source_items_1 = frames
    images = []
    for frame in source_items_1:
        images.append(Image.fromarray(frame))
    images[0].save(path, save_all=True, append_images=images[1:], duration=round(1000 / frequency_hz), loop=0)

def make_vector(protocol, prefix_steps):
    from lerobot.envs.configs import LiberoEnv
    from lerobot.envs.factory import make_env
    cfg = LiberoEnv(task=protocol['suite'], task_ids=[protocol['task_id']], control_mode='relative', init_states=True, hard_reset=True, max_parallel_tasks=1, observation_height=256, observation_width=256, episode_length=prefix_steps + protocol['recovery_horizon_steps'])
    envs = make_env(cfg, n_envs=1, use_async_envs=False)
    return envs[protocol['suite']][protocol['task_id']]

def run_case(pool, case, protocol, output):
    init_id = int(case['init_state_id'])
    folder = pool / f'init_{init_id:03d}'
    output.mkdir(parents=True, exist_ok=False)
    with np.load(folder / 'start.npz', allow_pickle=False) as data:
        prefix_steps = len(data['past_actions'])
    request = {'stage': 'evaluate', 'variant': 'passive_open_gripper', 'protocol': protocol, 'case_dir': str(folder), 'output': str(output), 'init_state_id': init_id, 'init_definition_sha256': case['init_definition_sha256']}
    write_json(output / 'request.json', request)
    vec = make_vector(protocol, prefix_steps)
    hooks = None
    frames = []
    result = None
    try:
        hooks = EvaluationHooks(request, type(vec).reset, type(vec).step)
        obs, _ = hooks.reset(vec, seed=protocol['seed'])
        capture_frame(obs, frames)
        start_geometry = geometry(vec)
        goal_at_restore = benchmark_goal(vec)
        success = goal_at_restore is True
        goal_checks = []
        for _ in range(0 if success else protocol['recovery_horizon_steps']):
            step_result = hooks.step(vec, passive_action())
            capture_frame(step_result[0], frames)
            goal = benchmark_goal(vec)
            signal = observed_success(step_result)
            goal_checks.append({'step': len(hooks.rows) - 1, 'success_signal': signal, 'goal_predicate': goal})
            success = signal
            if success or stopped(step_result):
                break
        end_geometry = geometry(vec)
        source_items_2 = hooks.rows
        items_2 = []
        for r in source_items_2:
            items_2.append(r['after']['simulator_time'])
        times = [start_geometry['simulator_time']] + items_2
        if len(times) > 1:
            intervals = np.diff(times)
            if not np.allclose(intervals, 0.05, atol=1e-08, rtol=0):
                raise RuntimeError('Частота управления отличается от проверенных 20 Hz.')
        result = {'init_state_id': init_id, 'success': success, 'steps': len(hooks.rows), 'goal_at_restore': goal_at_restore, 'action': passive_action()[0].tolist(), 'restore': hooks.restore, 'horizon': protocol['recovery_horizon_steps'], 'control_frequency_hz': 20, 'start_geometry': start_geometry, 'end_geometry': end_geometry, 'goal_signal_consistent': all((r['goal_predicate'] is None or r['goal_predicate'] == r['success_signal'] for r in goal_checks)), 'note': 'No ACT after restore; OSC feedback and object/contact dynamics remain active.'}
        write_json(output / 'result.json', result)
        write_json(output / 'goal_checks.json', goal_checks)
        save_gif(frames, output / 'passive.gif', 20)
        return result
    finally:
        write_json(output / 'trace.json', {'variant': 'passive_open_gripper', 'steps': hooks.rows if hooks is not None else []})
        if not getattr(vec, 'closed', False):
            vec.close()

def load_pool(pool, selected_ids):
    protocol = json.loads((pool / 'protocol.json').read_text())
    if protocol.get('complete') is not True:
        raise RuntimeError('Нужен завершённый candidate pool.')
    if protocol.get('suite') != 'libero_spatial' or protocol.get('task_id') != 0:
        raise RuntimeError('Этот контроль рассчитан на текущий libero_spatial task 0.')
    cases = protocol['cases']
    source_items_3 = cases
    ids = []
    for case in source_items_3:
        ids.append(int(case['init_state_id']))
    if len(ids) != len(set(ids)):
        raise RuntimeError('В пуле повторяются init-state IDs.')
    if selected_ids is not None:
        if len(selected_ids) != len(set(selected_ids)):
            raise RuntimeError('Не повторяйте IDs в --init-ids.')
        missing = sorted(set(selected_ids) - set(ids))
        if missing:
            raise RuntimeError(f'Эти IDs отсутствуют в пуле: {missing}')
        selected = set(selected_ids)
        source_items_4 = cases
        cases = []
        for case in source_items_4:
            if int(case['init_state_id']) in selected:
                cases.append(case)
    if not cases:
        raise RuntimeError('Не выбраны состояния для проверки.')
    for case in cases:
        folder = pool / f"init_{int(case['init_state_id']):03d}"
        for filename, key in (('start.npz', 'snapshot_sha256'), ('case.json', 'metadata_sha256')):
            if file_hash(folder / filename) != case[key]:
                raise RuntimeError(f'Файл изменился после collection: {folder / filename}')
        metadata = json.loads((folder / 'case.json').read_text())
        if metadata['init_definition_sha256'] != case['init_definition_sha256'] or metadata['status'] != 'accepted':
            raise RuntimeError('Метаданные recovery start не соответствуют protocol.json.')
    return (protocol, cases)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pool', type=Path, required=True, help='Папка завершённого пула с protocol.json.')
    parser.add_argument('--init-ids', type=int, nargs='+', help='Конкретные IDs; по умолчанию все состояния пула.')
    parser.add_argument('--horizon', type=positive_int, help='По умолчанию тот же post-start horizon, что у policies.')
    parser.add_argument('--project-root', type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    root = args.project_root.expanduser().resolve()
    pool = local_path(root, args.pool)
    protocol, cases = load_pool(pool, args.init_ids)
    protocol = dict(protocol)
    original_horizon = protocol['recovery_horizon_steps']
    if args.horizon is not None:
        protocol['recovery_horizon_steps'] = args.horizon
    os.chdir(root)
    output = root / 'pilot_runs/heldout_recovery' / datetime.now().strftime('passive_check_%Y%m%d_%H%M%S_%f')
    output.mkdir(parents=True, exist_ok=False)
    source_items_5 = cases
    items_5 = []
    for case in source_items_5:
        items_5.append(int(case['init_state_id']))
    info = {'status': 'running', 'pool': str(pool), 'protocol_sha256': file_hash(pool / 'protocol.json'), 'init_state_ids': items_5, 'source_horizon': original_horizon, 'diagnostic_horizon': protocol['recovery_horizon_steps'], 'runtime_versions': runtime_versions(), 'action': passive_action()[0].tolist(), 'helper_sha256': file_hash(Path(inspect.getfile(EvaluationHooks))), 'scope': 'Fixed open-gripper control after exact replay; no policy inference, demonstrations, or training.'}
    write_json(output / 'run_info.json', info)
    results = []
    try:
        for index, case in enumerate(cases, start=1):
            init_id = int(case['init_state_id'])
            print(f'\nPASSIVE {index}/{len(cases)}, init_id={init_id}: no movement; gripper open', flush=True)
            result = run_case(pool, case, protocol, output / f'init_{init_id:03d}')
            results.append(result)
            write_json(output / 'results.json', results)
            print(f"success={result['success']}; steps={result['steps']}; restore error={result['restore']['max_state_error']:.3g}", flush=True)
        total_success = sum((result['success'] for result in results))
        if not all((result['goal_signal_consistent'] for result in results)):
            print('Success signal and goal predicate differ; see goal_checks.json.', flush=True)
        summary = {'n_states': len(results), 'n_success': total_success, 'results': [{k: r[k] for k in ('init_state_id', 'success', 'steps', 'goal_at_restore', 'goal_signal_consistent')} for r in results], 'note': 'Selected diagnostic cases; this is not an unbiased policy evaluation or recovery expert.'}
        write_json(output / 'summary.json', summary)
        info.update(status='completed', completed_cases=len(results))
        write_json(output / 'run_info.json', info)
    except BaseException as error:
        info.update(status='stopped', completed_cases=len(results), error=f'{type(error).__name__}: {error}')
        write_json(output / 'run_info.json', info)
        raise
    print(f'\nPASSIVE CONTROL: {total_success}/{len(results)} successful starts', flush=True)
    print('Output:', output, flush=True)
if __name__ == '__main__':
    main()
