#!/usr/bin/env python3
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import numpy as np
ROLES = {'detector_train': range(0, 10), 'detector_validation': range(10, 15), 'detector_test': range(15, 20), 'selection': range(20, 30), 'evaluation': range(40, 50)}
PROFILES = {'shift_a': {'horizontal_eef_offset_m': [-0.1, -0.08], 'held_extra_lift_m': 0.03}, 'shift_b': {'horizontal_eef_offset_m': [-0.18, -0.08], 'held_extra_lift_m': 0.03}, 'shift_c': {'horizontal_eef_offset_m': [-0.1, -0.16], 'held_extra_lift_m': 0.03}}
FEATURE_ORDER = ['AU', 'EU', 'entropy', 'chosen_token_log_probability']
EXPERIMENT_PROTOCOL = 'insight_calibration_action_progress_v1'
COLLECTION_POINTER = 'pilot_runs/insight_calibration_collection_path.txt'
REVIEW_POINTER = 'pilot_runs/insight_calibration_review_path.txt'
FROZEN_POINTER = 'pilot_runs/insight_frozen_protocol_path.txt'
MODEL_POINTER = 'pilot_runs/insight_help_model_path.txt'

def require(condition, message):
    if not condition:
        raise ValueError(message)

def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))

def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()

def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    temporary.replace(path)

def stamp():
    return datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')

def resolve(root, value):
    path = Path(value).expanduser()
    return (path if path.is_absolute() else root / path).resolve()

def pointer(root, name):
    path = root / name
    require(path.is_file(), f'Нет {name}: сначала выполни предшествующий этап.')
    return resolve(root, path.read_text(encoding='utf-8').strip())

def set_pointer(root, name, value):
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(str(value) + '\n', encoding='utf-8')
    temporary.replace(path)

def signature(record):
    source_items_1 = ('checkpoint', 'openpi_commit', 'insight_commit', 'normalization', 'feature_order', 'suite', 'task_id', 'control_hz', 'replan_steps', 'trim_head', 'trim_tail')
    items_1 = {}
    for key in source_items_1:
        items_1[key] = record.get(key)
    return items_1

def validate_signature(value):
    require(value['suite'] == 'libero_spatial' and value['task_id'] == 0, 'Нужна текущая задача libero_spatial/0.')
    require(value['normalization'] == 'zscore' and value['feature_order'] == FEATURE_ORDER, 'Не совпадает порядок признаков или normalization.')
    require(value['control_hz'] == 20 and value['replan_steps'] == 5 and (value['trim_head'] == 3) and (value['trim_tail'] == 2), 'Протокол требует control=20Hz, replan=5, trim=3+2.')

def frozen_protocol(root):
    path = pointer(root, FROZEN_POINTER)
    record = read(path)
    require(record['experiment_protocol'] == EXPERIMENT_PROTOCOL and record['frozen'] is True, 'Неверный замороженный протокол.')
    require(record['profile'] in PROFILES and record['parameters'] == PROFILES[record['profile']], 'Параметры профиля изменились.')
    return (path, record)

def chosen_ids(args):
    defaults = {'detector_train': [0, 1] if not args.frozen else [2, 3], 'detector_validation': [10, 11], 'detector_test': [15, 16, 17], 'selection': [20, 21, 22, 23], 'evaluation': [40, 41, 42, 43, 44]}
    ids = args.init_ids if args.init_ids is not None else defaults[args.purpose]
    require(ids and len(ids) == len(set(ids)) and all((i in ROLES[args.purpose] for i in ids)), f'Для {args.purpose}: init IDs {list(ROLES[args.purpose])}, без повторений.')
    return ids

def collect(root, args):
    import collect_vla_displaced as collector
    import run_vla_libero as base
    require(getattr(base, 'DECODER_PROTOCOL', 0) >= 3, 'Нужен уже проверенный исправленный декодер (DECODER_PROTOCOL>=3). Старые файлы не заменены.')
    args.init_ids = chosen_ids(args)
    require(args.recovery_steps > 0 and args.nominal_steps > 0, 'Лимиты шагов должны быть положительными.')
    frozen_path = frozen = None
    if args.frozen:
        frozen_path, frozen = frozen_protocol(root)
        profiles = [frozen['profile']]
        require(args.profiles is None, 'После freeze профиль задаётся сохранённым протоколом.')
    else:
        require(args.purpose == 'detector_train', 'До freeze собираем только обучающую калибровку.')
        require(not (root / FROZEN_POINTER).exists(), 'Протокол уже заморожен. Используй collect --frozen для новых данных.')
        profiles = list(PROFILES) if args.profiles is None else args.profiles
        require(len(set(profiles)) == len(profiles), 'Профили не должны повторяться.')
    base.bridge.check_repositories(root)
    checkpoint, files = base.bridge.ensure_checkpoint()
    output = root / 'pilot_runs/vla_insight' / ('insight_collection_' + stamp())
    output.mkdir(parents=True)
    parent = {'status': 'running', 'experiment_protocol': EXPERIMENT_PROTOCOL, 'purpose': args.purpose, 'init_state_ids': args.init_ids, 'profiles': profiles, 'calibration_only': not args.frozen, 'frozen_protocol_sha256': digest(frozen_path) if frozen_path else None, 'checkpoint_local_path': str(checkpoint), 'checkpoint': base.bridge.CHECKPOINT, 'openpi_commit': base.bridge.OPENPI_COMMIT, 'insight_commit': base.bridge.INSIGHT_COMMIT, 'normalization': 'zscore', 'feature_order': FEATURE_ORDER, 'suite': 'libero_spatial', 'task_id': 0, 'control_hz': 20, 'replan_steps': 5, 'trim_head': 3, 'trim_tail': 2, 'decoder_protocol': base.DECODER_PROTOCOL, 'runs': [], 'script_sha256': digest(Path(__file__)), 'checkpoint_files': files, 'seed': 0, 'nominal_horizon': args.nominal_steps, 'recovery_horizon': args.recovery_steps}
    write(output / 'run_info.json', parent)
    set_pointer(root, COLLECTION_POINTER, output)
    print('COLLECTION:', output, flush=True)
    print(f'{len(profiles) * len(args.init_ids) * 2} cases; one GPU worker; labels are not created automatically.', flush=True)
    client, vec = (base.WorkerClient(root, output), None)
    original_protocol = deepcopy(collector.PROTOCOL)
    try:
        client.start()
        os.environ.setdefault('MUJOCO_GL', 'egl')
        from lerobot.envs.configs import LiberoEnv
        from lerobot.envs.factory import make_env
        horizon = args.nominal_steps + collector.intervention_step_limit() + args.recovery_steps + 1
        config = LiberoEnv(task='libero_spatial', task_ids=[0], control_mode='relative', init_states=True, hard_reset=True, max_parallel_tasks=1, observation_height=256, observation_width=256, episode_length=horizon)
        vec = make_env(config, n_envs=1, use_async_envs=False)['libero_spatial'][0]
        args.seed, args.replan_steps = (0, 5)
        for profile in profiles:
            collector.PROTOCOL.clear()
            collector.PROTOCOL.update(deepcopy(original_protocol), **deepcopy(PROFILES[profile]))
            collector.PROTOCOL['intervention_profile'] = profile
            run = output / profile
            run.mkdir()
            record = {**parent, 'status': 'running', 'profile': profile, 'protocol': deepcopy(collector.PROTOCOL), 'failure_types': list(collector.FAILURE_TYPES), 'seed': 0, 'cases': [], 'nominal_horizon': args.nominal_steps, 'recovery_horizon': args.recovery_steps, 'scope': 'Calibration or frozen one-task VLA collection; no automatic help labels'}
            record.pop('runs', None)
            write(run / 'run_info.json', record)
            parent['runs'].append(str(run))
            write(output / 'run_info.json', parent)
            for init_id in args.init_ids:
                for kind in collector.FAILURE_TYPES:
                    print(f'\nPROFILE={profile}; init_id={init_id}; type={kind}', flush=True)
                    try:
                        collector.run_case(vec, client, run, init_id, kind, args)
                    finally:
                        record['cases'] = collector.saved_cases(run)
                        write(run / 'run_info.json', record)
                        write(run / 'summary.json', collector.summary_for(record))
            record['status'] = 'complete'
            write(run / 'run_info.json', record)
            write(run / 'summary.json', collector.summary_for(record))
        parent['status'] = 'complete'
    except BaseException as error:
        parent.update(status='interrupted', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        collector.PROTOCOL.clear()
        collector.PROTOCOL.update(original_protocol)
        try:
            if vec is not None:
                vec.close()
        finally:
            try:
                client.close()
            finally:
                write(output / 'run_info.json', parent)
                write(output / 'summary.json', collection_summary(output, parent))
    print('\nCOLLECTION COMPLETE:', output)

def collection_summary(output, parent):
    summaries = {}
    for run_path in parent['runs']:
        run = resolve(output, run_path)
        if (run / 'summary.json').is_file():
            summaries[run.name] = read(run / 'summary.json')
    return {'status': parent['status'], 'purpose': parent['purpose'], 'experiment_protocol': EXPERIMENT_PROTOCOL, 'calibration_only': parent['calibration_only'], 'profiles': summaries, 'help_classifier_trained': False, 'note': 'Autonomous outcomes describe collection only and are never copied into help labels.'}

def find_runs(root, explicit=None):
    paths = [resolve(root, p) for p in explicit] if explicit else sorted((root / 'pilot_runs/vla_insight').glob('insight_collection_*/shift_*/run_info.json'))
    candidates = []
    for path in paths:
        if path.is_file():
            candidates.append(path.parent)
        elif (path / 'run_info.json').is_file():
            record = read(path / 'run_info.json')
            if 'runs' in record:
                candidates.extend((resolve(path, p) for p in record['runs']))
            else:
                candidates.append(path)
    freeze_path = root / FROZEN_POINTER
    frozen = frozen_protocol(root)[1] if freeze_path.exists() else None
    result = []
    for run in sorted(set(candidates)):
        record = read(run / 'run_info.json')
        require(record.get('experiment_protocol') == EXPERIMENT_PROTOCOL, 'Нужен сбор из insight_pipeline.py.')
        if record['purpose'] not in ('detector_train', 'detector_validation', 'detector_test'):
            continue
        if frozen:
            if record.get('profile') != frozen['profile']:
                continue
            source_items_2 = frozen['parameters']
            items_2 = {}
            for k in source_items_2:
                items_2[k] = record['protocol'][k]
            require(items_2 == frozen['parameters'], 'Параметры сбора изменены.')
        else:
            require(record['purpose'] == 'detector_train', 'Калибровка до freeze не использует validation/test.')
        result.append(run)
    require(result, 'Нет подходящих collections. Сначала python insight_pipeline.py collect')
    return result

def uniform_queries(boundaries, limit):
    if len(boundaries) <= limit:
        return boundaries
    ids = np.unique(np.linspace(0, len(boundaries) - 1, limit).round().astype(int))
    source_items_3 = ids
    items_3 = []
    for i in source_items_3:
        items_3.append(boundaries[i])
    return items_3

def configure_review():
    import label_vla_help as labels
    import review_vla_actions as progress
    source_items_4 = ('detector_train', 'detector_validation', 'detector_test')
    items_4 = {}
    for role in source_items_4:
        items_4[role] = ROLES[role]
    labels.ROLES = items_4
    return (labels, progress)

def prepare_review(root, runs, limit):
    labels, progress = configure_review()
    require(1 <= limit <= 100, 'Лимит queries на case: 1..100.')
    output = root / 'pilot_runs/vla_insight' / ('insight_review_' + stamp())
    output.mkdir(parents=True)
    (output / 'assets').mkdir()
    inherited = {}
    prior = []
    for name in (progress.POINTER, REVIEW_POINTER):
        if (root / name).exists():
            prior.append(pointer(root, name))
    for previous in dict.fromkeys(prior):
        old_manifest, old_labels = (read(previous / 'manifest.json'), read(previous / 'labels.json'))
        require(old_labels['manifest_sha256'] == digest(previous / 'manifest.json'), 'Изменился прежний manifest.')
        require(old_manifest['label_protocol'] == progress.PROTOCOL, 'Нельзя переносить субъективные legacy labels.')
        inherited.update(old_labels['answers'])
    samples, excluded, dedup = ([], [], {})
    policy_signature = None
    for run in runs:
        run_record = read(run / 'run_info.json')
        current = signature(run_record)
        validate_signature(current)
        require(policy_signature is None or current == policy_signature, 'Разные VLA / feature protocols.')
        policy_signature = current
        role = run_record['purpose']
        for path in sorted(run.glob('init_*/case.json')):
            case = read(path)
            init_id = case['init_state_id']
            require(case['purpose'] == role and type(init_id) is int and (init_id in ROLES[role]), 'Нарушен split по init IDs.')
            require(case.get('decoder_protocol', 0) >= 2, 'Legacy decoder 1 не подходит.')
            data = progress.case_data(path.parent)

            def boundary_step(b):
                return b['step']
            boundaries = sorted(case['inferences'], key=boundary_step)
            source_items_5 = boundaries
            items_5 = set()
            for b in source_items_5:
                items_5.add(b['step'])
            require(len(items_5) == len(boundaries), 'Повтор query step.')
            source_items_6 = boundaries
            eligible = []
            for b in source_items_6:
                if b.get('executed_actions', 0) > 0:
                    eligible.append(b)
            for boundary in uniform_queries(eligible, limit):
                query = labels.checked_path(run, boundary['query'])
                features, reason = labels.classifier_features(query, boundary)
                if not reason:
                    preview, reason = progress.action_preview(output, query, boundary, data)
                if reason:
                    excluded.append({'query': str(query), 'reason': reason})
                    continue
                source_items_7 = boundaries
                items_7 = []
                for b in source_items_7:
                    if max(0, boundary['step'] - 40) <= b['step'] < boundary['step']:
                        items_7.append(b)
                previous_queries = items_7[-4:]
                context = []
                for b in previous_queries + [boundary]:
                    q = labels.checked_path(run, b['query'])
                    images = labels.snapshot_images(q, b.get('snapshot_sha256'))
                    source_items_8 = images
                    items_8 = []
                    for im in source_items_8:
                        items_8.append(labels.save_image(output, im))
                    context.append({'step': b['step'], 'views': items_8})
                require(context[-1] == preview['frames'][0], 'Начальный кадр и preview расходятся.')
                visible = {'label_protocol': progress.PROTOCOL, 'role': role, 'init_id': init_id, 'instruction': case['instruction'], 'context': context, 'preview': preview}
                identity = hashlib.sha256(json.dumps(visible, sort_keys=True).encode() + features.tobytes()).hexdigest()
                source = {'run': str(run), 'query': str(query), 'features_path': str(query / 'token_features.npz'), 'features_sha256': digest(query / 'token_features.npz'), 'snapshot_sha256': digest(query / 'start.npz'), 'case_digests': data['digests'].copy()}
                if identity in dedup:
                    dedup[identity]['sources'].append(source)
                    if run_record['profile'] not in dedup[identity]['profiles']:
                        dedup[identity]['profiles'].append(run_record['profile'])
                    continue
                sample = {'sample_id': identity, 'purpose': role, 'init_id': init_id, 'instruction': case['instruction'], 'step': boundary['step'], 'context': context, 'action_preview': preview, 'sources': [source], 'profiles': [run_record['profile']]}
                samples.append(sample)
                dedup[identity] = sample
            print(f"Prepared {run_record['profile']}/{path.parent.name}: {len(samples)} unique samples", flush=True)
            del data
    require(samples, 'Нет сохранённых допустимых token features и проверенного движения. См. excluded_queries.json.')
    random.Random(0).shuffle(samples)
    source_items_9 = enumerate(sorted({(s['purpose'], s['init_id']) for s in samples}))
    items_9 = {}
    for i, key in source_items_9:
        items_9[key] = f'E{i + 1:02d}'
    aliases = items_9
    for sample in samples:
        sample['episode_alias'] = aliases[sample['purpose'], sample['init_id']]
    source_items_10 = runs
    items_10 = []
    for p in source_items_10:
        items_10.append(str(p))
    source_items_11 = ROLES.items()
    items_11 = {}
    for r, v in source_items_11:
        items_11[r] = list(v)
    manifest = {'schema_version': 2, 'experiment_protocol': EXPERIMENT_PROTOCOL, 'label_protocol': progress.PROTOCOL, 'label_question': progress.QUESTION, 'samples': samples, 'policy_protocol': policy_signature, 'created_at': stamp(), 'runs': items_10, 'max_queries_per_case': limit, 'scope': 'Simulation adaptation of INSIGHT Strong: progress of executed prefix, not ten-action counterfactual.', 'fixed_split': items_11, 'hidden_from_reviewer': ['outcome', 'failure_type', 'profile', 'uncertainty', 'old subjective labels']}
    write(output / 'manifest.json', manifest)
    write(output / 'excluded_queries.json', excluded)
    source_items_12 = samples
    answers = {}
    for s in source_items_12:
        if s['sample_id'] in inherited:
            answers[s['sample_id']] = inherited[s['sample_id']]
    write(output / 'labels.json', {'schema_version': 2, 'label_protocol': progress.PROTOCOL, 'manifest_sha256': digest(output / 'manifest.json'), 'label_question': progress.QUESTION, 'answers': answers})
    set_pointer(root, REVIEW_POINTER, output)
    print('REVIEW:', output)
    print('Prepared:', dict(Counter((s['purpose'] for s in samples))), '; restored labels:', len(answers))
    return output

def review_command(root, args):
    _, progress = configure_review()
    if args.resume:
        output = resolve(root, args.workspace) if args.workspace else pointer(root, REVIEW_POINTER)
    else:
        output = prepare_review(root, find_runs(root, args.runs), args.max_queries)
    if not args.prepare_only:
        progress.serve(output, args.port, not args.no_open)

def review_state(root, value=None):
    path = resolve(root, value) if value else pointer(root, REVIEW_POINTER)
    manifest, labels = (read(path / 'manifest.json'), read(path / 'labels.json'))
    require(labels['manifest_sha256'] == digest(path / 'manifest.json'), 'Manifest изменился после разметки.')
    require(manifest.get('experiment_protocol') == EXPERIMENT_PROTOCOL, 'Нужна новая insight_review_* папка.')
    require(manifest['label_protocol'] == 'action_progress_executed_prefix_v1' and labels['label_protocol'] == manifest['label_protocol'], 'Неверный критерий labels.')
    source_items_13 = manifest['samples']
    identities = []
    for s in source_items_13:
        identities.append(s['sample_id'])
    require(len(identities) == len(set(identities)), 'Повтор sample ID.')
    for sample in manifest['samples']:
        require(sample['purpose'] in ROLES and type(sample['init_id']) is int and (sample['init_id'] in ROLES[sample['purpose']]), 'Нарушен split.')
    require(set(labels['answers']) <= set(identities), 'Labels содержат неизвестные sample IDs.')
    for answer in labels['answers'].values():
        v = answer['help_required']
        require(v is None or (type(v) is int and v in (0, 1)), 'Labels должны быть 0/1/null.')
    return (path, manifest, labels)

def label_counts(samples, answers):
    counts = Counter({'continue': 0, 'help': 0, 'unknown': 0, 'unreviewed': 0})
    positive_ids = set()
    for sample in samples:
        answer = answers.get(sample['sample_id'])
        key = 'unreviewed' if answer is None else 'unknown' if answer['help_required'] is None else 'help' if answer['help_required'] == 1 else 'continue'
        counts[key] += 1
        if key == 'help':
            positive_ids.add(sample['init_id'])
    return {**counts, 'help_init_ids': sorted(positive_ids)}

def audit(root, args):
    _, manifest, labels = review_state(root, args.workspace)
    result = {role: label_counts([s for s in manifest['samples'] if s['purpose'] == role], labels['answers']) for role in ('detector_train', 'detector_validation', 'detector_test')}
    print(json.dumps(result, ensure_ascii=False, indent=2))

def freeze(root, args):
    path, manifest, labels = review_state(root, args.workspace)
    if (root / FROZEN_POINTER).exists():
        saved_path, saved = frozen_protocol(root)
        print('Frozen profile:', saved['profile'], saved_path)
        return
    counts = {}
    for profile in PROFILES:
        source_items_14 = manifest['samples']
        group = []
        for s in source_items_14:
            if s['purpose'] == 'detector_train' and profile in s['profiles']:
                group.append(s)
        counts[profile] = label_counts(group, labels['answers'])
    print(json.dumps(counts, indent=2, ensure_ascii=False))
    source_items_15 = PROFILES
    eligible = []
    for p in source_items_15:
        if counts[p]['help'] >= 5 and counts[p]['continue'] >= 10 and (len(counts[p]['help_init_ids']) >= 2) and (counts[p]['unreviewed'] == 0):
            eligible.append(p)
    require(eligible, 'Нет профиля с >=5 help на >=2 train init IDs и >=10 continue при завершённом просмотре. Метки не меняем ради баланса; нужна дополнительная обучающая калибровка.')
    profile = eligible[0]
    destination = root / 'pilot_runs/vla_insight' / ('frozen_insight_protocol_' + stamp() + '.json')
    write(destination, {'experiment_protocol': EXPERIMENT_PROTOCOL, 'frozen': True, 'profile': profile, 'parameters': PROFILES[profile], 'created_at': stamp(), 'policy_protocol': manifest['policy_protocol'], 'label_protocol': manifest['label_protocol'], 'calibration_manifest_sha256': digest(path / 'manifest.json'), 'calibration_labels_sha256': digest(path / 'labels.json'), 'calibration_counts': counts, 'rule': 'first predefined profile with >=5 help on >=2 training init IDs, >=10 continue, all reviewed', 'validation_or_test_used': False, 'autonomous_outcome_used': False})
    set_pointer(root, FROZEN_POINTER, destination)
    print('FROZEN:', profile, destination)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    c = sub.add_parser('collect')
    c.add_argument('--purpose', choices=tuple(ROLES), default='detector_train')
    c.add_argument('--init-ids', nargs='+', type=int)
    c.add_argument('--profiles', nargs='+', choices=tuple(PROFILES))
    c.add_argument('--frozen', action='store_true')
    c.add_argument('--nominal-steps', type=int, default=280)
    c.add_argument('--recovery-steps', type=int, default=280)
    r = sub.add_parser('review')
    r.add_argument('--runs', nargs='+')
    r.add_argument('--workspace')
    r.add_argument('--resume', action='store_true')
    r.add_argument('--max-queries', type=int, default=8)
    r.add_argument('--port', type=int, default=8767)
    r.add_argument('--no-open', action='store_true')
    r.add_argument('--prepare-only', action='store_true')
    for name in ('audit', 'freeze'):
        sub.add_parser(name).add_argument('--workspace')
    t = sub.add_parser('train')
    t.add_argument('--workspace')
    t.add_argument('--epochs', type=int, default=60)
    t.add_argument('--seed', type=int, default=0)
    e = sub.add_parser('evaluate')
    e.add_argument('--workspace')
    e.add_argument('--model')
    sub.add_parser('smoke')
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    if args.command == 'collect':
        collect(root, args)
    elif args.command == 'review':
        review_command(root, args)
    elif args.command == 'audit':
        audit(root, args)
    elif args.command == 'freeze':
        freeze(root, args)
    else:
        import insight_training as training
        {'train': training.train, 'evaluate': training.evaluate, 'smoke': training.smoke}[args.command](root, args)
if __name__ == '__main__':
    try:
        main()
    except (Exception, KeyboardInterrupt) as error:
        print(f'\nSTOP: {error}', file=sys.stderr, flush=True)
        sys.exit(1)
