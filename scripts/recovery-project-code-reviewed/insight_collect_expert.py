#!/usr/bin/env python3
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import html
import os
from pathlib import Path
import shutil
import sys
import time
import numpy as np
import collect_correction as expert
import insight_pipeline as p
import insight_select as selection
import run_vla_libero as base
POINTER = 'pilot_runs/insight_expert_collection_path.txt'
METHODS = ('INSIGHT', 'F_geometry', 'Random')

def check_sources(hashes):
    for path, expected in hashes.items():
        p.require(Path(path).is_file() and p.digest(path) == expected, f'Исходный файл изменился: {path}')

def load_plan(root, value=None):
    folder = p.resolve(root, value) if value else p.pointer(root, selection.POINTER)
    path = folder if folder.is_file() else folder / 'selection.json'
    plan = p.read(path)
    p.require(plan['status'] == 'planned' and set(plan['conditions']) == set(METHODS), 'Нужен готовый план insight_select.py для трёх методов.')
    budget = plan['initial_query_budget']
    p.require(type(budget) is int and budget > 0, 'Некорректный query budget.')
    check_sources(plan['source_hashes'])
    _, _, actual, _, hashes = selection.read_candidates(root, plan['pool'])
    source_items_1 = actual
    current = {}
    for row in source_items_1:
        current[row['candidate_id']] = row
    source_items_2 = plan['candidates']
    indexed = {}
    for row in source_items_2:
        indexed[row['candidate_id']] = row
    p.require(len(indexed) == len(plan['candidates']) and set(current) == set(indexed), 'Common pool изменился после selection.')
    for identity, row in indexed.items():
        p.require(all((row[key] == current[identity][key] for key in ('init_id', 'failure_type', 'snapshot', 'snapshot_sha256', 'query'))), f'Selected start изменился: {identity}')
    for method in METHODS:
        ids = plan['conditions'][method]['selected_candidates']
        p.require(len(ids) == budget and len(set(ids)) == budget and (set(ids) <= set(indexed)), f'Неверный список запросов {method}.')
    frozen, _ = p.frozen_protocol(root)
    p.require(p.digest(frozen) == plan['frozen_protocol_sha256'], 'Frozen protocol изменился.')
    hashes.update(plan['source_hashes'])
    hashes[str(path)] = p.digest(path)
    hashes[str(frozen)] = p.digest(frozen)
    return (path, plan, indexed, hashes)

def max_error(actual, expected, name, tolerance=1e-08):
    a, b = (np.asarray(actual), np.asarray(expected))
    p.require(a.shape == b.shape and a.size and np.isfinite(a).all() and np.isfinite(b).all(), f'Некорректная форма/значения: {name}.')
    error = float(np.max(np.abs(a.astype(np.float64) - b.astype(np.float64))))
    p.require(error <= tolerance, f'Restore не совпал: {name}, error={error:.12g}.')
    return error

def restore(vec, row):
    snapshot = Path(row['snapshot'])
    p.require(p.digest(snapshot) == row['snapshot_sha256'], 'Snapshot изменился перед replay.')
    case_dir = snapshot.parent.parent
    case = p.read(case_dir / 'case.json')
    reset_path = case_dir / 'reset.npz'
    p.require(p.digest(reset_path) == case['reset_snapshot_sha256'], 'Reset snapshot изменился.')
    with np.load(snapshot, allow_pickle=False) as saved:
        source_items_3 = saved.files
        data = {}
        for key in source_items_3:
            data[key] = saved[key].copy()
    with np.load(reset_path, allow_pickle=False) as saved:
        definition = saved['init_definition'].copy()
        p.require(np.array_equal(saved['simulator_state'], data['reset_state']) and np.array_equal(saved['ctrl'], data['reset_ctrl']), 'Candidate и reset относятся к разным начальным состояниям.')
    wrapper = base.libero_wrapper(vec)
    init_id = row['init_id']
    p.require(case['init_state_id'] == init_id and 20 <= init_id < 30 and (wrapper._init_states is not None) and (init_id < len(wrapper._init_states)), 'Неверный selection init ID.')
    current_definition = np.asarray(wrapper._init_states[init_id], dtype=np.float64)
    p.require(np.array_equal(current_definition, definition) and hashlib.sha256(current_definition.tobytes()).hexdigest() == case['init_definition_sha256'], 'LIBERO initial definition отличается от сохранённой.')
    wrapper.init_state_id = init_id
    observation, _ = vec.reset(seed=case['seed'])
    raw = base.raw_environment(vec)
    wrapper = base.libero_wrapper(vec)
    p.require(str(wrapper.task_description) == case['instruction'] and int(wrapper.control_freq) == case['control_hz'] == 20, 'Изменились задача или control frequency.')
    state, ctrl = base.physical_state(vec)
    info = {'reset_state_error': max_error(state, data['reset_state'], 'reset state'), 'reset_ctrl_error': max_error(ctrl, data['reset_ctrl'], 'reset ctrl')}
    prefix = data['past_actions']
    p.require(prefix.ndim == 3 and prefix.shape[1:] == (1, 7) and (len(prefix) > 0) and np.isfinite(prefix).all() and (np.max(np.abs(prefix)) <= 1), 'Некорректный префикс действий.')
    p.require(len(prefix) == case['recovery_start_step'], 'Префикс не заканчивается в selected start.')
    for action in prefix:
        observation, _, terminated, truncated, _ = vec.step(action)
        p.require(not (np.any(terminated) or np.any(truncated) or base.benchmark_success(vec)), 'Эпизод завершился во время replay.')
    state, ctrl = base.physical_state(vec)
    info.update(state_error=max_error(state, data['simulator_state'], 'candidate state'), ctrl_error=max_error(ctrl, data['ctrl'], 'candidate ctrl'), prefix_steps=len(prefix))
    observed = base.flatten_observation(observation)
    source_items_4 = data.items()
    expected = {}
    for key, value in source_items_4:
        if key.startswith('observation/'):
            expected[key] = value
    p.require(set(observed) == set(expected), 'Изменился состав observation.')
    errors = {}
    for key, value in observed.items():
        if '/pixels/' in key:
            p.require(np.array_equal(value, expected[key]), f'Камера отличается при replay: {key}')
            errors[key] = 0.0
        else:
            errors[key] = max_error(value, expected[key], key, tolerance=1e-06)
    info['observation_errors'] = errors
    p.require(not base.benchmark_success(vec), 'Selected start уже удовлетворяет цели.')
    print(f"RESTORE OK: init_id={init_id}; prefix={len(prefix)}; state_error={info['state_error']:.3g}; ctrl_error={info['ctrl_error']:.3g}", flush=True)
    return (observation, raw, case, info)

class ObservationTracker:

    def __init__(self, vec, observation):
        self.vec, self.observation = (vec, observation)

    def step(self, action):
        result = self.vec.step(action)
        self.observation = result[0]
        return result

    def __getattr__(self, name):
        return getattr(self.vec, name)

def approach_above_bowl(vec, raw, grasp_offset, records, max_steps, max_move_steps=100):
    scale = np.asarray(raw.robots[0].controller.output_max[:3], dtype=float)
    p.require(scale.shape == (3,) and np.isfinite(scale).all() and np.all(scale > 0), 'Неверная шкала OSC controller.')

    def step(action, phase):
        p.require(len(records['actions']) < max_steps, 'Достигнут лимит expert steps.')
        before = expert.flatten_observation(vec.observation)
        geometry = expert.geometry(raw)
        action = np.asarray(action, dtype=np.float32).reshape(1, 7)
        _, reward, terminated, truncated, _ = vec.step(action)
        for key, value in before.items():
            records['observations'].setdefault(key, []).append(value)
        records['actions'].append(action[0].copy())
        records['rewards'].append(float(np.asarray(reward).reshape(-1)[0]))
        records['phases'].append(phase)
        records['geometry'].append(geometry)
        records['success'] = base.benchmark_success(vec)
        records['ended'] = bool(np.any(terminated) or np.any(truncated))
        if records['success']:
            raise expert.GoalReached()
        p.require(not records['ended'], 'Эпизод завершился во время подхода expert.')

    def move(target, phase):
        for _ in range(max_move_steps):
            error = target - np.asarray(expert.geometry(raw)['eef_pos'])
            if np.linalg.norm(error) < 0.008:
                return
            action = np.zeros(7)
            action[:3] = np.clip(0.6 * error / scale, -0.35, 0.35)
            action[-1] = -1
            step(action, phase)
        distance = float(np.linalg.norm(target - np.asarray(expert.geometry(raw)['eef_pos'])))
        if distance < 0.008:
            return
        raise RuntimeError(f'Expert не достиг waypoint: {phase}; distance={distance:.4f} m; limit={max_move_steps}')
    print('PHASE: prepare_grasp', flush=True)
    for _ in range(6):
        step([0, 0, 0, 0, 0, 0, -1], 'open_for_clearance')
    geometry = expert.geometry(raw)
    above = np.asarray(geometry['bowl_pos']) + grasp_offset
    above[2] = max(float(geometry['eef_pos'][2]), float(above[2] + 0.1))
    raised = np.asarray(geometry['eef_pos']).copy()
    raised[2] = above[2]
    move(raised, 'raise_for_clearance')
    move(above, 'approach_above_bowl')

def run_expert(vec, raw, grasp_offset, placement_offset, records, max_steps, max_move_steps):
    scale = np.asarray(raw.robots[0].controller.output_max[:3], dtype=float)
    p.require(scale.shape == (3,) and np.all(scale > 0) and np.isfinite(scale).all(), 'Неверная шкала OSC.')
    start_bowl_z = float(expert.geometry(raw)['bowl_pos'][2])

    def eef_pos():
        return np.asarray(expert.geometry(raw)['eef_pos'])

    def bowl_pos():
        return np.asarray(expert.geometry(raw)['bowl_pos'])

    def step(action, phase):
        p.require(len(records['actions']) < max_steps, 'Достигнут общий лимит действий expert.')
        before = expert.flatten_observation(vec.observation)
        action = np.asarray(action, dtype=np.float32).reshape(1, 7)
        before_geometry = expert.geometry(raw)
        _, reward, terminated, truncated, _ = vec.step(action)
        for key, value in before.items():
            records['observations'].setdefault(key, []).append(value)
        records['actions'].append(action[0].copy())
        records['rewards'].append(float(np.asarray(reward).reshape(-1)[0]))
        records['phases'].append(phase)
        records['geometry'].append(before_geometry)
        records['success'] = base.benchmark_success(vec)
        records['ended'] = bool(np.any(terminated) or np.any(truncated))
        if len(records['actions']) % 20 == 0 or records['success']:
            print(f"Expert steps={len(records['actions'])}; phase={phase}; success={records['success']}", flush=True)
        if records['success']:
            raise expert.GoalReached()
        p.require(not records['ended'], 'Эпизод завершился без достижения цели.')

    def hold(gripper, steps, phase):
        for _ in range(steps):
            action = np.zeros(7)
            action[-1] = gripper
            step(action, phase)

    def move_to(target, gripper, phase):
        target = np.asarray(target, dtype=float)
        first_step = len(records['actions'])
        check = {'phase': phase, 'target_eef_pos': target.tolist(), 'max_move_steps': max_move_steps, 'tolerance_m': 0.008, 'start_error_m': float(np.linalg.norm(target - eef_pos()))}
        records.setdefault('waypoint_checks', []).append(check)
        try:
            for _ in range(max_move_steps):
                error = target - eef_pos()
                if np.linalg.norm(error) < 0.008:
                    return
                action = np.zeros(7)
                action[:3] = np.clip(0.6 * error / scale, -0.35, 0.35)
                action[-1] = gripper
                step(action, phase)
            distance = float(np.linalg.norm(target - eef_pos()))
            p.require(distance < 0.008, f'Захват не достиг позиции: {phase}; distance={distance:.4f} m; limit={max_move_steps}.')
        finally:
            check.update(end_error_m=float(np.linalg.norm(target - eef_pos())), executed_actions=len(records['actions']) - first_step)
    print('PHASE: approach_grasp', flush=True)
    hold(-1, 6, 'open_before_grasp')
    move_to(bowl_pos() + grasp_offset, -1, 'approach_grasp')
    print('PHASE: close_gripper_and_lift', flush=True)
    hold(1, 12, 'close_gripper')
    lifted = eef_pos()
    lifted[2] += 0.1
    move_to(lifted, 1, 'lift')
    rise = float(bowl_pos()[2] - start_bowl_z)
    records['bowl_rise_m'] = rise
    print('Bowl rise (m):', rise, flush=True)
    p.require(rise >= 0.04, 'Миска не поднялась: повторный захват не подтверждён.')
    print('PHASE: transport', flush=True)
    desired_bowl = np.asarray(expert.geometry(raw)['plate_pos']) + placement_offset
    placement_target = desired_bowl + (eef_pos() - bowl_pos())
    above_plate = placement_target.copy()
    above_plate[2] = max(eef_pos()[2], placement_target[2] + 0.1)
    move_to(above_plate, 1, 'transport')
    print('PHASE: lower_and_release', flush=True)
    move_to(placement_target, 1, 'lower')
    hold(-1, 12, 'release')
    hold(-1, 20, 'settle')
    raise RuntimeError('Expert завершил последовательность, но benchmark не подтвердил успех.')

def collect_one(root, out, row, cfg, grasp_offset, placement_offset, max_steps, max_move_steps=100):
    out.mkdir()
    records = {'observations': {}, 'actions': [], 'rewards': [], 'phases': [], 'geometry': [], 'success': False, 'ended': False}
    metadata = {'candidate_id': row['candidate_id'], 'failure_type': row['failure_type'], 'init_state_id': row['init_id'], 'source_snapshot': row['snapshot'], 'source_snapshot_sha256': row['snapshot_sha256'], 'suite': 'libero_spatial', 'task_id': 0, 'expert': 'fixed_scripted_OSC_controller_with_privileged_simulator_geometry', 'control_frequency_hz': 20, 'observation_timing': 'before_action', 'max_expert_steps': max_steps, 'max_move_steps': max_move_steps, 'expert_controller_revision': 2, 'grasp_offset': grasp_offset.tolist(), 'placement_offset': placement_offset.tolist(), 'scope': 'Expert demonstration from a selection start; no autonomous policy evaluation'}
    vec, tracker, raw = (None, None, None)
    started = time.perf_counter()
    try:
        vec = expert.make_vec(cfg)
        observation, raw, case, restore_info = restore(vec, row)
        metadata.update(restore_check=restore_info, instruction=case['instruction'], seed=case['seed'], init_definition_sha256=case['init_definition_sha256'], start_geometry=expert.geometry(raw))
        shutil.copy2(row['snapshot'], out / 'start.npz')
        tracker = ObservationTracker(vec, observation)
        try:
            approach_above_bowl(tracker, raw, grasp_offset, records, max_steps, max_move_steps)
            run_expert(tracker, raw, grasp_offset, placement_offset, records, max_steps, max_move_steps)
        except expert.GoalReached:
            pass
        predicate = base.benchmark_success(vec)
        p.require(predicate == records['success'], 'Benchmark predicate и recorded success расходятся.')
        metadata.update(goal_predicate_at_end=predicate, end_geometry=expert.geometry(raw))
        source_items_5 = records['geometry']
        items_5 = []
        for g in source_items_5:
            items_5.append(g['simulator_time'])
        times = items_5 + [metadata['end_geometry']['simulator_time']]
        p.require(records['actions'] and np.allclose(np.diff(times), 0.05, atol=1e-08, rtol=0), 'Expert control frequency отличается от 20 Hz.')
        p.require(records['success'], 'Expert не завершил задачу в установленном бюджете.')
    except Exception as error:
        records['success'] = False
        metadata['error'] = f'{type(error).__name__}: {error}'
        print('EXPERT ATTEMPT NOT ACCEPTED:', error, flush=True)
    finally:
        metadata['wall_seconds_including_restore'] = time.perf_counter() - started
        metadata['last_phase'] = records['phases'][-1] if records['phases'] else None
        metadata['waypoint_checks'] = records.get('waypoint_checks', [])
        if raw is not None:
            try:
                metadata.update(end_geometry=expert.geometry(raw), goal_predicate_at_end=base.benchmark_success(vec))
            except Exception as error:
                metadata['end_diagnostic_error'] = str(error)
        try:
            expert.save_attempt(out, records, metadata, make_gif=False)
            try:
                if records['actions'] and tracker is not None:
                    first = records['observations']['observation/pixels/image']
                    second = records['observations']['observation/pixels/image2']
                    source_items_6 = zip(first, second)
                    frames = []
                    for a, b in source_items_6:
                        frames.append(np.ascontiguousarray(np.concatenate((a[::-1, ::-1], b[::-1, ::-1]), axis=1)))
                    final = base.flatten_observation(tracker.observation)
                    frames.append(np.ascontiguousarray(np.concatenate((final['observation/pixels/image'][0, ::-1, ::-1], final['observation/pixels/image2'][0, ::-1, ::-1]), axis=1)))
                    base.save_video(out / 'expert.mp4', frames, 20)
            except Exception as error:
                metadata['video_error'] = str(error)
                p.write(out / 'metadata.json', metadata)
                print('Video write failed:', error, flush=True)
        finally:
            if vec is not None:
                vec.close()
    demo = out / 'demonstration.npz'
    return {'candidate_id': row['candidate_id'], 'failure_type': row['failure_type'], 'init_id': row['init_id'], 'accepted': records['success'], 'expert_steps': len(records['actions']), 'folder': str(out), 'demonstration': str(demo) if records['success'] else None, 'demonstration_sha256': p.digest(demo) if records['success'] else None, 'metadata_sha256': p.digest(out / 'metadata.json'), 'wall_seconds_including_restore': metadata['wall_seconds_including_restore'], 'error': metadata.get('error'), 'last_phase': metadata['last_phase'], 'waypoint_checks': metadata['waypoint_checks'], 'max_move_steps': max_move_steps}

def condition_results(plan, attempts, history=None):
    results = {}
    for method in METHODS:
        selected = plan['conditions'][method]['selected_candidates']
        source_items_7 = selected
        tried = []
        for key in source_items_7:
            if key in attempts:
                tried.append(attempts[key])
        source_items_8 = tried
        accepted = []
        for row in source_items_8:
            if row['accepted']:
                accepted.append(row)
        source_items_9 = selected
        all_attempts = []
        for key in source_items_9:
            for row in (history or {}).get(key, [attempts[key]] if key in attempts else []):
                all_attempts.append(row)
        source_items_10 = accepted
        items_10 = []
        for row in source_items_10:
            items_10.append(row['candidate_id'])
        results[method] = {'selected_candidates': selected, 'n_requested': len(selected), 'n_attempted': len(tried), 'n_accepted': len(accepted), 'budget_met': len(accepted) == len(selected), 'accepted_candidates': items_10, 'accepted_failure_counts': dict(Counter((row['failure_type'] for row in accepted))), 'n_attempts_including_retries': len(all_attempts), 'expert_action_timesteps_including_failures': sum((row['expert_steps'] for row in all_attempts)), 'logical_wall_seconds_including_restore': sum((row['wall_seconds_including_restore'] for row in all_attempts))}
    return results

def gallery(out, info):
    cards = []
    for method in METHODS:
        content = []
        for identity in info['conditions'][method]['selected_candidates']:
            result = info['unique_attempts'].get(identity)
            if result is None:
                content.append(f'<p>{html.escape(identity)}: not attempted</p>')
                continue
            state = 'Benchmark success' if result['accepted'] else 'Stopped before verified success'
            video_path = Path(result['folder']) / 'expert.mp4'
            video = html.escape(os.path.relpath(video_path, out), quote=True)
            player = f'<video controls preload="metadata" src="{video}"></video>' if video_path.is_file() else ''
            detail = f"<p>{html.escape(result['error'])}</p>" if result.get('error') else ''
            content.append(f"<article><h3>{html.escape(identity)}</h3>{player}<p>{state} · {result['expert_steps']} actions · 20 Hz</p>{detail}</article>")
        cards.append(f"<section><h2>{method}</h2>{''.join(content)}</section>")
    text = f"""<!doctype html><meta charset="utf-8"><title>Selected expert recoveries</title><style>body{{background:#0f1726;color:#e7edf7;font:16px system-ui;margin:30px}}main{{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:20px}}section,article{{background:#182337;padding:16px;border-radius:12px;margin-bottom:16px}}article{{background:#223047}}video{{width:100%}}h1,h2{{font-weight:600}}@media(max-width:850px){{main{{grid-template-columns:1fr}}}}</style><h1>Expert recovery demonstrations</h1><p>Same selected starts and fixed expert. Shared starts reuse the same recording. These videos show expert control, not a fine-tuned VLA.</p><main>{''.join(cards)}</main>"""
    (out / 'gallery.html').write_text(text, encoding='utf-8')

def previous_collection(root, value, path, selected, indexed, max_steps, max_move_steps):
    folder = p.resolve(root, value) if value else p.pointer(root, POINTER)
    record_path = folder / 'collection.json'
    previous = p.read(record_path)
    p.require(previous['status'] == 'complete' and previous['selection_plan_sha256'] == p.digest(path) and (previous['unique_selected_candidates'] == selected), 'Предыдущий collection относится к другому selection plan или не завершён.')
    own_path = Path(__file__).resolve()
    source_items_11 = previous['source_hashes'].items()
    previous_inputs = {}
    for name, value in source_items_11:
        if Path(name).resolve() != own_path:
            previous_inputs[name] = value
    check_sources(previous_inputs)
    p.require(max_move_steps >= max(50, previous.get('max_move_steps', 40)), 'При reuse успешных попыток нельзя уменьшать лимит перемещения.')
    p.require(set(previous['unique_attempts']) == set(selected), 'Не хватает предыдущих attempts.')
    history = previous.get('attempt_history') or {key: [result] for key, result in previous['unique_attempts'].items()}
    for identity in selected:
        p.require(identity in history and history[identity] and (history[identity][-1] == previous['unique_attempts'][identity]), 'Нарушена история attempts.')
        for attempt in history[identity]:
            metadata_path = Path(attempt['folder']) / 'metadata.json'
            p.require(attempt['candidate_id'] == identity and p.digest(metadata_path) == attempt['metadata_sha256'], 'Изменились предыдущие metadata.')
            metadata = p.read(metadata_path)
            row = indexed[identity]
            p.require(metadata['source_snapshot_sha256'] == row['snapshot_sha256'] and metadata['init_state_id'] == row['init_id'] and (metadata['failure_type'] == row['failure_type']) and (metadata['accepted'] == attempt['accepted']) and (metadata['n_frames'] == attempt['expert_steps']), 'Attempt относится к другому start.')
            previous_inputs[str(metadata_path)] = attempt['metadata_sha256']
            if attempt['accepted']:
                demo = Path(attempt['demonstration'])
                p.require(metadata.get('goal_predicate_at_end') is True and attempt['expert_steps'] <= max_steps and (p.digest(demo) == attempt['demonstration_sha256']), 'Нельзя reuse неподтверждённую demonstration.')
                previous_inputs[str(demo)] = attempt['demonstration_sha256']
    previous_inputs[str(record_path)] = p.digest(record_path)
    source_items_12 = ('grasp_offset', 'placement_offset')
    offsets = []
    for key in source_items_12:
        offsets.append(np.asarray(previous[key], dtype=float))
    p.require(all((value.shape == (3,) and np.isfinite(value).all() for value in offsets)), 'Неверная expert calibration.')
    return (folder, previous, history, previous_inputs, offsets)

def run(root, args):
    os.chdir(root)
    os.environ.setdefault('MUJOCO_GL', 'egl')
    path, plan, indexed, hashes = load_plan(root, args.plan)
    nominal_path = p.resolve(root, args.nominal_actions)
    with np.load(nominal_path, allow_pickle=False) as data:
        nominal = data['actions'].copy()
    p.require(nominal.ndim == 3 and nominal.shape[1:] == (1, 7) and (len(nominal) >= 42) and np.isfinite(nominal).all(), 'Нужна проверенная nominal init-0 action trace.')
    hashes[str(nominal_path)] = p.digest(nominal_path)
    hashes[str(Path(expert.__file__).resolve())] = p.digest(Path(expert.__file__).resolve())
    hashes[str(Path(__file__).resolve())] = p.digest(Path(__file__).resolve())
    selected = list(dict.fromkeys((identity for method in METHODS for identity in plan['conditions'][method]['selected_candidates'])))
    previous, previous_path, offsets = (None, None, None)
    attempts, history = ({}, {})
    if args.retry_failed:
        previous_path, previous, history, previous_inputs, offsets = previous_collection(root, args.previous_collection, path, selected, indexed, args.max_expert_steps, args.max_move_steps)
        attempts = dict(previous['unique_attempts'])
        hashes.update(previous_inputs)
        hashes[str(Path(__file__).resolve())] = p.digest(Path(__file__).resolve())
    source_items_13 = selected
    pending = []
    for identity in source_items_13:
        if identity not in attempts or not attempts[identity]['accepted']:
            pending.append(identity)
    if not pending:
        print('All selected demonstrations were accepted. Output:', previous_path, flush=True)
        return previous
    prefix_max = 0
    for identity in selected:
        with np.load(indexed[identity]['snapshot'], allow_pickle=False) as data:
            prefix_max = max(prefix_max, len(data['past_actions']))
    from lerobot.envs.configs import LiberoEnv
    cfg = LiberoEnv(task='libero_spatial', task_ids=[0], control_mode='relative', init_states=True, hard_reset=True, max_parallel_tasks=1, observation_height=256, observation_width=256, episode_length=max(prefix_max + args.max_expert_steps + 20, len(nominal) + 20))
    out = root / 'pilot_runs/vla_insight' / f'selected_expert_{p.stamp()}'
    out.mkdir(parents=True)
    info = {'status': 'running', 'selection_plan': str(path), 'selection_plan_sha256': p.digest(path), 'source_hashes': hashes, 'target_budget_per_method': plan['initial_query_budget'], 'selection_seed': plan['seed'], 'unique_selected_candidates': selected, 'unique_attempts': attempts, 'attempt_history': history, 'conditions': condition_results(plan, attempts, history), 'max_expert_steps': args.max_expert_steps, 'max_move_steps': args.max_move_steps, 'expert_controller_revision': 2, 'acquisition_rule': 'Keep exactly the selected candidates. Explicit retries of incomplete starts are counted; never replace a candidate using its autonomous outcome.', 'previous_collection': str(previous_path) if previous_path else None, 'retry_rule': 'Repeat every incomplete selected start with the same increased move limit; preserve accepted recordings and all earlier attempt costs.' if args.retry_failed else None, 'shared_data_rule': 'Identical selected start is physically recorded once, reused unchanged by all selecting methods.', 'cost_note': 'Per-method logical costs include shared attempts. Physical total counts each recording once. These are scripted controller costs, not human annotation times.', 'policy_trained_here': False, 'autonomous_success_measured_here': False, 'scope': 'Expert data acquisition from a frozen VLA selection pool; not a selector learning-gain test'}
    p.write(out / 'collection.json', info)
    p.set_pointer(root, POINTER, out)
    print('EXPERT COLLECTION:', out, flush=True)
    print(f"Requests per method: {plan['initial_query_budget']}; starts in this run: {len(pending)}; move limit={args.max_move_steps}. The VLA is not loaded.", flush=True)
    try:
        if offsets is None:
            grasp_offset, placement_offset = expert.calibrate(cfg, nominal)
        else:
            grasp_offset, placement_offset = offsets
        info.update(grasp_offset=grasp_offset.tolist(), placement_offset=placement_offset.tolist(), expert_rule='Fixed init-0 calibration and OSC gains; common configurable move limit; open/approach/regasp/lift/transport/lower/release. No per-method or per-start tuning.')
        p.write(out / 'collection.json', info)
        for identity in pending:
            check_sources(hashes)
            print(f'\nEXPERT {identity}', flush=True)
            result = collect_one(root, out / identity, indexed[identity], cfg, grasp_offset, placement_offset, args.max_expert_steps, args.max_move_steps)
            info['unique_attempts'][identity] = result
            info['attempt_history'].setdefault(identity, []).append(result)
            info['conditions'] = condition_results(plan, info['unique_attempts'], info['attempt_history'])
            p.write(out / 'collection.json', info)
            print(f"accepted={result['accepted']}; expert_steps={result['expert_steps']}", flush=True)
        check_sources(hashes)
        info['status'] = 'complete'
    except BaseException as error:
        info.update(status='interrupted', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        info['conditions'] = condition_results(plan, info['unique_attempts'], info['attempt_history'])
        info['all_budgets_met'] = all((row['budget_met'] for row in info['conditions'].values()))
        source_items_14 = info['attempt_history'].values()
        all_attempts = []
        for values in source_items_14:
            for attempt in values:
                all_attempts.append(attempt)
        info['physical_recordings'] = len(all_attempts)
        info['physical_action_timesteps'] = sum((row['expert_steps'] for row in all_attempts))
        p.write(out / 'collection.json', info)
        gallery(out, info)
    print('\nSELECTED EXPERT COLLECTION COMPLETE', flush=True)
    for method, condition in info['conditions'].items():
        print(f"{method}: demonstrations={condition['n_accepted']}/{condition['n_requested']}; attempts_with_retries={condition['n_attempts_including_retries']}; types={condition['accepted_failure_counts']}", flush=True)
    print('Output:', out, flush=True)
    print('Gallery:', out / 'gallery.html', flush=True)
    return info

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', help='Папка selector_comparison или selection.json; по умолчанию последний plan.')
    parser.add_argument('--nominal-actions', default='pilot_runs/lift_trace/actions.npz')
    parser.add_argument('--max-expert-steps', type=expert.positive_int, default=280)
    parser.add_argument('--max-move-steps', type=expert.positive_int, default=100, help='Общий для всех expert starts лимит действий на одно перемещение.')
    parser.add_argument('--retry-failed', action='store_true', help='Повторить только незавершённые starts из previous collection.')
    parser.add_argument('--previous-collection', help='Для retry; по умолчанию текущий expert collection pointer.')
    args = parser.parse_args()
    p.require(not args.previous_collection or args.retry_failed, '--previous-collection используется с --retry-failed.')
    result = run(Path(__file__).resolve().parent, args)
    if not result['all_budgets_met']:
        print('Expert collection incomplete: accepted demonstration budgets differ.', file=sys.stderr)
        sys.exit(2)
if __name__ == '__main__':
    try:
        main()
    except (Exception, KeyboardInterrupt) as error:
        print('\nSTOP:', error, file=sys.stderr)
        sys.exit(1)
