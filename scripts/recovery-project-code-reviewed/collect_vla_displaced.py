#!/usr/bin/env python3
from __future__ import annotations
import argparse
from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import numpy as np
import run_vla_libero as base
bridge = base.bridge
require = base.require
json_atomic = base.json_atomic
FAILURE_TYPES = ('missed_grasp', 'object_dropped')
ROLE_IDS = {'detector_train': range(0, 10), 'detector_validation': range(10, 15), 'selection': range(20, 30), 'evaluation': range(40, 50)}
PROTOCOL = {'held_min_bowl_rise_m': 0.02, 'held_consecutive_observations': 3, 'lift_threshold_m': 0.06, 'drop_threshold_m': 0.03, 'open_steps': 12, 'miss_trigger_distance_m': 0.1, 'close_command_threshold': 0.5, 'miss_max_bowl_rise_m': 0.015, 'miss_min_empty_lift_m': 0.035, 'miss_max_bowl_shift_m': 0.03, 'miss_lift_action': 0.35, 'miss_lift_steps': 24}
PROTOCOL.update({'intervention_profile': 'displaced_v1', 'horizontal_eef_offset_m': [-0.1, -0.08], 'held_extra_lift_m': 0.03, 'waypoint_tolerance_m': 0.006, 'waypoint_gain': 0.6, 'waypoint_max_abs_action': 0.35, 'waypoint_max_steps': 70, 'protocol_fixed_before_rollout': True, 'outcome_used_to_filter_candidates': False})
SCOPE = 'Single-task displaced recovery training diagnostic; no trained help detector or selector comparison'

def intervention_step_limit():
    return max(1 + PROTOCOL['miss_lift_steps'] + PROTOCOL['waypoint_max_steps'], 2 * PROTOCOL['waypoint_max_steps'] + PROTOCOL['open_steps'])

class SkipCase(Exception):
    pass

def validate_arguments(args):
    require(args.purpose == 'detector_train', 'Эта малая проба использует только detector_train (init IDs 0..9).')
    require(len(set(args.init_ids)) == len(args.init_ids), '--init-ids должны быть разными.')
    allowed = ROLE_IDS[args.purpose]
    require(all((i in allowed for i in args.init_ids)), f'Для {args.purpose} отведены init IDs {allowed.start}..{allowed.stop - 1}.')
    require(args.nominal_steps > 0 and args.recovery_steps > 0, 'Лимиты шагов должны быть положительными.')
    require(len(set(args.failure_types)) == len(args.failure_types), 'Типы ошибок не должны повторяться.')

def descendant_ids(model, root_id):
    parents, result = (np.asarray(model.body_parentid), set())
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

class Diagnostics:

    def __init__(self, vec):
        raw = base.raw_environment(vec)
        self.sim = raw.sim
        model = self.sim.model
        names = list(model.body_names)
        for name in ('akita_black_bowl_1_main', 'plate_1_main', 'gripper0_leftfinger', 'gripper0_rightfinger'):
            require(name in names, f'Неподдерживаемая сцена: отсутствует {name}.')
        self.bowl_id = names.index('akita_black_bowl_1_main')
        self.plate_id = names.index('plate_1_main')
        self.site_id = model.site_name2id(raw.robots[0].controller_config['eef_name'])
        robot = raw.robots[0]
        require(robot.controller_config.get('type') == 'OSC_POSE' and robot.controller_config.get('control_delta') is True, 'Смещение требует OSC_POSE с относительным управлением.')
        self.translation_scale = np.asarray(robot.controller.output_max[:3], dtype=np.float64).copy()
        require(self.translation_scale.shape == (3,) and np.isfinite(self.translation_scale).all() and np.all(self.translation_scale > 0), 'Некорректный масштаб OSC перемещения.')
        self.groups = {'bowl': descendant_ids(model, self.bowl_id), 'left': descendant_ids(model, names.index('gripper0_leftfinger')), 'right': descendant_ids(model, names.index('gripper0_rightfinger'))}
        self.geom_bodies = np.asarray(model.geom_bodyid)

    def geometry(self):
        data = self.sim.data
        return {'simulator_time': float(data.time), 'bowl_pos': data.body_xpos[self.bowl_id].tolist(), 'plate_pos': data.body_xpos[self.plate_id].tolist(), 'eef_pos': data.site_xpos[self.site_id].tolist()}

    def contacts(self):
        sides = set()
        for index in range(int(self.sim.data.ncon)):
            contact = self.sim.data.contact[index]
            bodies = {int(self.geom_bodies[contact.geom1]), int(self.geom_bodies[contact.geom2])}
            if bodies & self.groups['bowl']:
                sides.update((side for side in ('left', 'right') if bodies & self.groups[side]))
        return {'left': 'left' in sides, 'right': 'right' in sides, 'bilateral': {'left', 'right'} <= sides}

def eligible_event(kind, action, before, initial, contacts, held_ever):
    rise = before['bowl_pos'][2] - initial['bowl_pos'][2]
    if kind == 'object_dropped':
        return held_ever and contacts['bilateral'] and (rise >= PROTOCOL['lift_threshold_m'])
    distance = float(np.linalg.norm(np.asarray(before['eef_pos']) - before['bowl_pos']))
    return not held_ever and rise < PROTOCOL['miss_max_bowl_rise_m'] and (action[-1] >= PROTOCOL['close_command_threshold']) and (distance <= PROTOCOL['miss_trigger_distance_m'])

def waypoint_actions(diagnostics, target, gripper, *, held=False):
    target = np.asarray(target, dtype=np.float64)
    require(target.shape == (3,) and np.isfinite(target).all(), 'Некорректная цель вмешательства.')
    for _ in range(PROTOCOL['waypoint_max_steps']):
        if held and (not diagnostics.contacts()['bilateral']):
            raise SkipCase('Удержание миски потеряно до запланированного отпускания.')
        error = target - np.asarray(diagnostics.geometry()['eef_pos'])
        if np.linalg.norm(error) <= PROTOCOL['waypoint_tolerance_m']:
            return
        action = np.zeros(7, dtype=np.float32)
        action[:3] = np.clip(PROTOCOL['waypoint_gain'] * error / diagnostics.translation_scale, -PROTOCOL['waypoint_max_abs_action'], PROTOCOL['waypoint_max_abs_action'])
        action[-1] = gripper
        yield action
    if held and (not diagnostics.contacts()['bilateral']):
        raise SkipCase('Удержание миски потеряно до запланированного отпускания.')
    if np.linalg.norm(target - np.asarray(diagnostics.geometry()['eef_pos'])) > PROTOCOL['waypoint_tolerance_m']:
        raise SkipCase('Захват не достиг waypoint вмешательства за фиксированный лимит.')

def intervention_actions(kind, intended, diagnostics):
    opened = np.zeros(7, dtype=np.float32)
    opened[-1] = -1
    horizontal = np.asarray(PROTOCOL['horizontal_eef_offset_m'])
    if kind == 'object_dropped':
        target = np.asarray(diagnostics.geometry()['eef_pos'], dtype=np.float64)
        target[2] += PROTOCOL['held_extra_lift_m']
        yield from waypoint_actions(diagnostics, target, 1, held=True)
        target[:2] += horizontal
        yield from waypoint_actions(diagnostics, target, 1, held=True)
        if not diagnostics.contacts()['bilateral']:
            raise SkipCase('Миска не удерживается перед запланированным отпусканием.')
        for _ in range(PROTOCOL['open_steps']):
            yield opened.copy()
        return
    blocked = intended.copy()
    blocked[-1] = -1
    lift = opened.copy()
    lift[2] = PROTOCOL['miss_lift_action']
    yield blocked
    for _ in range(PROTOCOL['miss_lift_steps']):
        yield lift.copy()
    target = np.asarray(diagnostics.geometry()['eef_pos'], dtype=np.float64)
    target[:2] += horizontal
    yield from waypoint_actions(diagnostics, target, -1)

def verify_event(kind, initial, before, rows, held_ever):
    require(bool(rows), 'Нет действий вмешательства.')
    after = rows[-1]['geometry_after']
    if kind == 'object_dropped':
        release_index = next((i for i, r in enumerate(rows) if r['executed_action'][-1] < 0), None)
        require(release_index is not None, 'Нет запланированного открывания захвата.')
        at_release = rows[release_index]['geometry_before']
        held_at_release = bool(rows[release_index]['contacts_before']['bilateral'])
        fall = at_release['bowl_pos'][2] - after['bowl_pos'][2]
        lost = not rows[-1]['contacts_after']['bilateral']
        shift = np.asarray(after['bowl_pos'][:2]) - before['bowl_pos'][:2]
        evidence = {'fall_m': fall, 'held_before_intervention': bool(held_ever), 'held_immediately_before_release': held_at_release, 'bilateral_hold_lost': lost, 'release_forced_before_goal': True, 'bowl_displacement_xy_m': shift.tolist(), 'eef_displacement_xy_m': (np.asarray(at_release['eef_pos'][:2]) - before['eef_pos'][:2]).tolist(), 'release_step_within_intervention': release_index}
        return (bool(held_ever and held_at_release and lost and (fall >= PROTOCOL['drop_threshold_m'])), evidence)
    source_items_1 = rows
    items_1 = []
    for r in source_items_1:
        items_1.append(r['geometry_after']['bowl_pos'][2])
    rise = max([before['bowl_pos'][2]] + items_1) - initial['bowl_pos'][2]
    empty_lift = after['eef_pos'][2] - before['eef_pos'][2]
    shift = float(np.linalg.norm(np.asarray(after['bowl_pos'][:2]) - before['bowl_pos'][:2]))
    no_hold = not any((r['contacts_after']['bilateral'] for r in rows))
    evidence = {'max_bowl_rise_m': rise, 'empty_eef_lift_m': empty_lift, 'bowl_shift_xy_m': shift, 'no_bilateral_hold_during_attempt': no_hold, 'eef_displacement_xy_m': (np.asarray(after['eef_pos'][:2]) - before['eef_pos'][:2]).tolist()}
    return (bool(not held_ever and rise < PROTOCOL['miss_max_bowl_rise_m'] and (empty_lift >= PROTOCOL['miss_min_empty_lift_m']) and (shift <= PROTOCOL['miss_max_bowl_shift_m']) and no_hold), evidence)

@dataclass
class Recorder:
    vec: object
    observation: dict
    diagnostics: Diagnostics
    actions: list = field(default_factory=list)
    rewards: list = field(default_factory=list)
    raw_rows: list = field(default_factory=list)
    frames: list = field(default_factory=list)
    trajectory: list = field(default_factory=list)
    success: bool = False
    ended: bool = False

    def apply(self, action, phase, inference_step=None, chunk_index=None):
        require(not self.ended, 'Нельзя выполнять действие после окончания эпизода.')
        predicted = np.asarray(action, dtype=np.float32)
        require(predicted.shape == (7,) and np.isfinite(predicted).all(), 'Некорректное действие.')
        actual = np.clip(predicted, -1, 1)
        raw = base.flatten_observation(self.observation)
        state, ctrl = base.physical_state(self.vec)
        row = {'step': len(self.actions), 'phase': phase, 'inference_step': inference_step, 'chunk_index': chunk_index, 'geometry_before': self.diagnostics.geometry(), 'contacts_before': self.diagnostics.contacts(), 'predicted_action': predicted.tolist(), 'executed_action': actual.tolist()}
        obs, reward, terminated, truncated, _ = self.vec.step(actual[None])
        self.success = base.benchmark_success(self.vec)
        self.ended = self.success or bool(np.any(terminated) or np.any(truncated))
        row.update(geometry_after=self.diagnostics.geometry(), contacts_after=self.diagnostics.contacts(), success_after_action=self.success, episode_ended_after_action=self.ended)
        self.raw_rows.append({**raw, 'simulator_state': state, 'ctrl': ctrl})
        self.frames.append(np.ascontiguousarray(raw['observation/pixels/image'][0, ::-1, ::-1]))
        self.actions.append(actual.copy())
        self.rewards.append(float(np.asarray(reward).reshape(-1)[0]))
        self.trajectory.append(row)
        self.observation = obs
        return row

    def snapshot(self, path, reset_state, reset_ctrl):
        state, ctrl = base.physical_state(self.vec)
        np.savez_compressed(path, simulator_state=state, ctrl=ctrl, reset_state=reset_state, reset_ctrl=reset_ctrl, past_actions=np.asarray(self.actions, dtype=np.float32).reshape(-1, 1, 7), **base.flatten_observation(self.observation))

def run_case(vec, client, output, init_id, kind, args):
    case_id = f'init_{init_id:03d}_{kind}'
    folder = output / case_id
    folder.mkdir()
    (folder / 'queries').mkdir()
    wrapper = base.libero_wrapper(vec)
    require(wrapper._init_states is not None and 0 <= init_id < len(wrapper._init_states), f'Initial-state ID {init_id} недоступен.')
    definition = np.asarray(wrapper._init_states[init_id], dtype=np.float64).copy()
    wrapper.init_state_id = init_id
    observation, _ = vec.reset(seed=args.seed)
    require(not base.benchmark_success(vec), 'Начальное состояние уже удовлетворяет цели.')
    wrapper = base.libero_wrapper(vec)
    instruction, hz = (str(wrapper.task_description), int(wrapper.control_freq))
    require(hz == base.CONTROL_HZ, f'Неожиданный control_freq={hz}.')
    diagnostics = Diagnostics(vec)
    initial = diagnostics.geometry()
    recorder = Recorder(vec, observation, diagnostics)
    reset_state, reset_ctrl = base.physical_state(vec)
    np.savez_compressed(folder / 'reset.npz', init_definition=definition, simulator_state=reset_state, ctrl=reset_ctrl, **base.flatten_observation(observation))
    record = {'status': 'running', 'scope': SCOPE, 'case_id': case_id, 'purpose': args.purpose, 'suite': 'libero_spatial', 'task_id': 0, 'init_state_id': init_id, 'seed': args.seed, 'requested_failure_type': kind, 'verified_failure_type': None, 'instruction': instruction, 'init_definition_sha256': hashlib.sha256(definition.tobytes()).hexdigest(), 'reset_snapshot_sha256': bridge.digest(folder / 'reset.npz'), 'nominal_horizon': args.nominal_steps, 'recovery_horizon': args.recovery_steps, 'replan_steps': args.replan_steps, 'control_hz': hz, 'initial_geometry': initial, 'decoder_protocol': base.DECODER_PROTOCOL, 'intervention_profile': PROTOCOL['intervention_profile'], 'candidate_accepted': False, 'recovery_success': None, 'help_score': None, 'inferences': [], 'intervention': None}
    labels = []
    hold_streak, held_ever, recovery_start = (0, False, None)

    def query_policy(phase, limit):
        step = len(recorder.actions)
        query = folder / 'queries' / f'step_{step:04d}'
        query.mkdir()
        recorder.snapshot(query / 'start.npz', reset_state, reset_ctrl)
        if phase == 'recovery' and 'candidate_query' not in record:
            record['candidate_query'] = str(query.relative_to(output))
        invalid = None
        try:
            chunk, info = client.infer(query, base.openpi_observation(recorder.observation, instruction))
        except base.InferenceOutputError as error:
            invalid = error
            chunk, info = (None, error.info)
        boundary = {'step': step, 'phase': phase, 'query': str(query.relative_to(output)), 'snapshot_sha256': bridge.digest(query / 'start.npz'), 'features_sha256': bridge.digest(query / 'token_features.npz') if (query / 'token_features.npz').is_file() else None, 'expected_executed_actions': min(args.replan_steps, limit), 'executed_actions': 0, 'inference': info}
        record['inferences'].append(boundary)
        labels.append({'step': step, 'phase': phase, 'query': boundary['query'], 'help_required': None, 'annotator': None, 'failure_type': None, 'features_valid_for_classifier': info.get('features_valid_for_classifier'), 'verified_event_before_boundary': kind if recovery_start is not None else None, 'note': "Review task progress of this prediction's executed prefix; do not copy the episode outcome."})
        if invalid is not None:
            raise invalid
        print(f"{case_id}: {phase} step={step}; tokens={info['generated_tokens']}; help_score=None", flush=True)
        return (chunk[:boundary['expected_executed_actions']], boundary)
    try:
        json_atomic(folder / 'case.json', record)
        while len(recorder.actions) < args.nominal_steps and (not recorder.ended) and (recovery_start is None):
            chunk, boundary = query_policy('nominal', args.nominal_steps - len(recorder.actions))
            for chunk_index, action in enumerate(chunk):
                before, contacts = (diagnostics.geometry(), diagnostics.contacts())
                previous_close = bool(recorder.actions and recorder.actions[-1][-1] > 0)
                rise = before['bowl_pos'][2] - initial['bowl_pos'][2]
                if contacts['bilateral'] and previous_close and (rise >= PROTOCOL['held_min_bowl_rise_m']):
                    hold_streak += 1
                else:
                    hold_streak = 0
                held_ever = held_ever or hold_streak >= PROTOCOL['held_consecutive_observations']
                if eligible_event(kind, action, before, initial, contacts, held_ever):
                    event_start, event_rows = (len(recorder.actions), [])
                    record['intervention'] = {'start_step': event_start, 'before': before, 'held_ever': held_ever, 'intended_action': action.tolist()}
                    print(f'INTERVENTION {case_id}, step={event_start}', flush=True)
                    for forced in intervention_actions(kind, action, diagnostics):
                        event_rows.append(recorder.apply(forced, 'intervention'))
                        if recorder.ended:
                            raise SkipCase('Задача или эпизод завершились во время вмешательства.')
                    verified, evidence = verify_event(kind, initial, before, event_rows, held_ever)
                    record['intervention'].update(end_step=len(recorder.actions), verification=evidence)
                    if not verified:
                        raise SkipCase(f'Событие {kind} не подтверждено: {evidence}')
                    require(not base.benchmark_success(vec), 'Recovery start уже удовлетворяет цели.')
                    recovery_start = len(recorder.actions)
                    (folder / 'candidate').mkdir()
                    recorder.snapshot(folder / 'candidate/start.npz', reset_state, reset_ctrl)
                    record.update(candidate_accepted=True, verified_failure_type=kind, recovery_start_step=recovery_start, candidate_snapshot_sha256=bridge.digest(folder / 'candidate/start.npz'))
                    print(f'VERIFIED {case_id}: {evidence}', flush=True)
                    break
                recorder.apply(action, 'nominal', boundary['step'], chunk_index)
                boundary['executed_actions'] += 1
                if recorder.ended:
                    break
            json_atomic(folder / 'case.json', record)
        if recovery_start is None:
            raise SkipCase('VLA завершила эпизод или не достигла условий вмешательства.')
        while not recorder.ended and len(recorder.actions) - recovery_start < args.recovery_steps:
            remaining = args.recovery_steps - (len(recorder.actions) - recovery_start)
            chunk, boundary = query_policy('recovery', remaining)
            if 'candidate_query' not in record:
                record['candidate_query'] = boundary['query']
            for chunk_index, action in enumerate(chunk):
                recorder.apply(action, 'recovery', boundary['step'], chunk_index)
                boundary['executed_actions'] += 1
                if recorder.ended:
                    break
            record.update(recovery_steps=len(recorder.actions) - recovery_start)
            json_atomic(folder / 'case.json', record)
        record.update(status='complete', recovery_success=bool(recorder.success), recovery_steps=len(recorder.actions) - recovery_start, total_steps=len(recorder.actions))
    except SkipCase as error:
        record.update(status='skipped', skip_reason=str(error), recovery_success=None, total_steps=len(recorder.actions))
        print(f'SKIPPED {case_id}: {error}', flush=True)
    except base.InferenceOutputError as error:
        record.update(status='invalid_output', recovery_success=None, recovery_steps=len(recorder.actions) - recovery_start if recovery_start is not None else None, total_steps=len(recorder.actions), error_code=error.code, error=str(error), task_goal_at_stop=bool(recorder.success), fallback_actions_executed=False)
        print(f'CASE STOPPED {case_id}: {error.code}: {error}', flush=True)
    except BaseException as error:
        record.update(status='interrupted', recovery_success=None, total_steps=len(recorder.actions), error=f'{type(error).__name__}: {error}')
        raise
    finally:
        if record['intervention'] is not None:
            record['intervention'].setdefault('end_step', min(len(recorder.actions), recovery_start if recovery_start is not None else len(recorder.actions)))
        if recorder.raw_rows:
            arrays = {key: np.stack([row[key] for row in recorder.raw_rows]) for key in recorder.raw_rows[0]}
            np.savez_compressed(folder / 'rollout.npz', **arrays, action=np.asarray(recorder.actions, dtype=np.float32), reward=np.asarray(recorder.rewards, dtype=np.float32))
            try:
                base.save_video(folder / 'rollout.mp4', recorder.frames, hz)
                record['video'] = str(folder / 'rollout.mp4')
                if record.get('candidate_accepted'):
                    lo = max(0, record['intervention']['start_step'] - 10)
                    hi = record['recovery_start_step']
                    context_frames = recorder.frames[lo:hi] + [np.ascontiguousarray(recorder.raw_rows[hi]['observation/pixels/image'][0, ::-1, ::-1]) if hi < len(recorder.raw_rows) else np.ascontiguousarray(base.flatten_observation(recorder.observation)['observation/pixels/image'][0, ::-1, ::-1])]
                    base.save_video(folder / 'intervention.mp4', context_frames, hz)
            except Exception as error:
                record['video_error'] = str(error)
                print('Video write failed:', error, flush=True)
        state, ctrl = base.physical_state(vec)
        np.savez_compressed(folder / 'final.npz', simulator_state=state, ctrl=ctrl, **base.flatten_observation(recorder.observation))
        json_atomic(folder / 'trajectory.json', recorder.trajectory)
        json_atomic(folder / 'help_labels_template.json', {'status': 'unannotated', 'purpose': args.purpose, 'labels': labels, 'note': 'A verified failure event is not itself a Strong INSIGHT help label.'})
        json_atomic(folder / 'case.json', record)
    print(f"CASE {case_id}: status={record['status']}; recovery_success={record['recovery_success']}", flush=True)
    return record

def summary_for(record):
    cases = record['cases']
    source_items_2 = cases
    accepted = []
    for c in source_items_2:
        if c['candidate_accepted']:
            accepted.append(c)
    source_items_3 = accepted
    complete = []
    for c in source_items_3:
        if c['status'] == 'complete':
            complete.append(c)
    by_type = {}
    for kind in FAILURE_TYPES:
        source_items_4 = complete
        group = []
        for c in source_items_4:
            if c['verified_failure_type'] == kind:
                group.append(c)
        by_type[kind] = {'n_verified_starts': sum((c.get('verified_failure_type') == kind for c in accepted)), 'n_complete_continuations': len(group), 'n_invalid_outputs': sum((c.get('verified_failure_type') == kind and c['status'] == 'invalid_output' for c in accepted)), 'n_autonomous_success': sum((c['recovery_success'] for c in group)), 'n_autonomous_failure': sum((not c['recovery_success'] for c in group))}
    return {'status': record['status'], 'purpose': record['purpose'], 'scope': SCOPE, 'intervention_profile': record.get('protocol', {}).get('intervention_profile'), 'decoder_protocol': record.get('decoder_protocol', 1), 'n_requested': len(record['init_state_ids']) * len(record['failure_types']), 'n_recorded': len(cases), 'n_verified_starts': len(accepted), 'n_complete_continuations': len(complete), 'n_autonomous_success': sum((c['recovery_success'] for c in complete)), 'n_autonomous_failure': sum((not c['recovery_success'] for c in complete)), 'n_invalid_outputs': sum((c['status'] == 'invalid_output' for c in cases)), 'n_interrupted': sum((c['status'] in ('interrupted', 'running') for c in cases)), 'n_skipped': sum((c['status'] == 'skipped' for c in cases)), 'by_type': by_type, 'classifier_trained': False, 'help_labels': 'unannotated', 'note': 'Invalid outputs are reported separately, not hidden as completed rollouts. Outcomes do not filter candidates or supply help labels.'}

def saved_cases(output):
    source_items_5 = sorted(output.glob('init_*/case.json'))
    items_5 = []
    for path in source_items_5:
        items_5.append(bridge.read_json(path))
    return items_5

def inspect_run(root, value):
    output = Path(value).expanduser()
    if not output.is_absolute():
        output = root / output
    require(output.is_dir(), f'Нет папки: {output}')
    record = bridge.read_json(output / 'run_info.json')
    record['cases'] = saved_cases(output)
    summary = summary_for(record)
    legacy = record.get('decoder_protocol', 1) < 2
    summary['legacy_decoder_guard'] = legacy
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    for case in record['cases']:
        print(f"{case['case_id']}: status={case['status']}; candidate={case['candidate_accepted']}; recovery_success={case['recovery_success']}; error={case.get('error', '')}")
    if legacy:
        print('The legacy decoder guard could accept a zero fallback. These continuations are not a clean recovery evaluation.')

def collect(root, args):
    validate_arguments(args)
    require(getattr(base, 'DECODER_PROTOCOL', None) == 3, 'Нужен run_vla_libero.py с DECODER_PROTOCOL=3 (valid payload may be no-EOS).')
    bridge.check_repositories(root)
    checkpoint, files = bridge.ensure_checkpoint()
    output = root / 'pilot_runs/vla_insight' / ('failures_displaced_' + datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f'))
    output.mkdir(parents=True, exist_ok=False)
    source_items_6 = ROLE_IDS.items()
    items_6 = {}
    for k, v in source_items_6:
        items_6[k] = list(v)
    record = {'status': 'running', 'scope': SCOPE, 'purpose': args.purpose, 'source_policy': 'pi0-FAST', 'checkpoint': bridge.CHECKPOINT, 'checkpoint_local_path': str(checkpoint), 'checkpoint_files': files, 'openpi_commit': bridge.OPENPI_COMMIT, 'insight_commit': bridge.INSIGHT_COMMIT, 'normalization': 'zscore', 'allocator': 'platform', 'control_hz': base.CONTROL_HZ, 'suite': 'libero_spatial', 'task_id': 0, 'init_state_ids': args.init_ids, 'failure_types': args.failure_types, 'seed': args.seed, 'replan_steps': args.replan_steps, 'nominal_horizon': args.nominal_steps, 'recovery_horizon': args.recovery_steps, 'protocol': PROTOCOL, 'initial_configuration_split': items_6, 'feature_order': bridge.FEATURE_ORDER, 'trim_head': 3, 'trim_tail': 2, 'help_detector_trained': False, 'autonomous_outcome_used_for_selection': False, 'decoder_protocol': base.DECODER_PROTOCOL, 'invalid_output_handling': 'stop_case_continue_next_no_fallback', 'parent_versions': base.versions(), 'script_sha256': bridge.digest(Path(__file__)), 'rollout_script_sha256': bridge.digest(root / 'run_vla_libero.py'), 'bridge_sha256': bridge.digest(root / 'start_insight.py'), 'cases': []}
    json_atomic(output / 'run_info.json', record)
    json_atomic(output / 'summary.json', summary_for(record))
    pointer = root / 'pilot_runs/vla_displaced_collection_path.txt'
    pointer.write_text(str(output) + '\n', encoding='utf-8')
    print('Output:', output, '\nPurpose:', args.purpose, flush=True)
    print('Intervention profile displaced_v1: x=-0.10 m, y=-0.08 m; drop after an additional 0.03 m lift.', flush=True)
    print(f'After event verification: {args.recovery_steps} autonomous VLA steps; labels remain unannotated.', flush=True)
    client, vec = (base.WorkerClient(root, output), None)
    try:
        client.start()
        os.environ.setdefault('MUJOCO_GL', 'egl')
        from lerobot.envs.configs import LiberoEnv
        from lerobot.envs.factory import make_env
        total_horizon = args.nominal_steps + intervention_step_limit() + args.recovery_steps + 1
        cfg = LiberoEnv(task='libero_spatial', task_ids=[0], control_mode='relative', init_states=True, hard_reset=True, max_parallel_tasks=1, observation_height=256, observation_width=256, episode_length=total_horizon)
        vec = make_env(cfg, n_envs=1, use_async_envs=False)['libero_spatial'][0]
        for init_id in args.init_ids:
            for kind in args.failure_types:
                record['cases'].append(run_case(vec, client, output, init_id, kind, args))
                json_atomic(output / 'run_info.json', record)
                json_atomic(output / 'summary.json', summary_for(record))
        record['status'] = 'complete'
    except BaseException as error:
        record.update(status='interrupted', error=f'{type(error).__name__}: {error}')
        record['cases'] = saved_cases(output)
        raise
    finally:
        try:
            if vec is not None:
                vec.close()
        finally:
            try:
                client.close()
            finally:
                json_atomic(output / 'run_info.json', record)
                json_atomic(output / 'summary.json', summary_for(record))
    summary = summary_for(record)
    print(f"\nVLA FAILURE COLLECTION COMPLETE: {summary['n_verified_starts']}/{summary['n_requested']} verified starts")
    for kind, row in summary['by_type'].items():
        print(f"{kind}: VLA recovery={row['n_autonomous_success']}/{row['n_complete_continuations']}; invalid outputs={row['n_invalid_outputs']}")
    print('Output:', output)

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--init-ids', type=int, nargs='+', default=[0, 1])
    parser.add_argument('--purpose', choices=('detector_train',), default='detector_train')
    parser.add_argument('--failure-types', choices=FAILURE_TYPES, nargs='+', default=list(FAILURE_TYPES))
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--replan-steps', type=int, choices=range(1, 11), default=5)
    parser.add_argument('--nominal-steps', type=int, default=280)
    parser.add_argument('--recovery-steps', type=int, default=280)
    parser.add_argument('--inspect-run', help='Просмотр старого запуска без загрузки VLA и без изменения файлов.')
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    if args.inspect_run:
        inspect_run(root, args.inspect_run)
    else:
        collect(root, args)
if __name__ == '__main__':
    try:
        main()
    except (Exception, KeyboardInterrupt) as error:
        print(f'\nSTOP: {error}', file=sys.stderr, flush=True)
        sys.exit(1)
