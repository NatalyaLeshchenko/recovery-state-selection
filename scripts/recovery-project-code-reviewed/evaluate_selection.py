#!/usr/bin/env python3
from __future__ import annotations
import argparse
from collections import Counter
from copy import deepcopy
from datetime import datetime
import os
from pathlib import Path
import runpy
import subprocess
import sys
import traceback
import numpy as np
os.environ.setdefault('MUJOCO_GL', 'egl')
import selection_pilot as pilot
import selection_experiment as experiment
import validate_recovery as validation
VARIANTS = ('initial', 'U', 'F_ref', 'Random')
TEST_POINTER = 'selection_test_pool_path.txt'
RECOVERY_POINTER = 'selection_recovery_evaluation_path.txt'
CLEAN_POINTER = 'selection_clean_evaluation_path.txt'
TEST_FILE = 'test_protocol.json'
SCOPE = 'Independent induced-error ACT test; held out from recovery learning and selector tuning'

def require(condition, message):
    if not condition:
        raise RuntimeError(message)

def load_models(root, value=None):
    folder = pilot.local_path(root, value) if value else pilot.resolve_pointer(root, experiment.MODELS_POINTER)
    filename = folder if folder.is_file() else folder / 'models.json'
    info = pilot.read_json(filename)
    if info.get('status') != 'complete' or set(info.get('models', {})) != set(experiment.METHODS):
        raise pilot.StageNotReady('Нужны завершённые U, F_ref и Random: THREE SELECTOR POLICIES READY.')
    dataset_file = Path(info['datasets']) / 'datasets.json'
    require(pilot.file_hash(dataset_file) == info['datasets_sha256'], 'Описание training datasets изменилось.')
    datasets = pilot.read_json(dataset_file)
    collection, acquisition = experiment.load_collection(root, info['collection'])
    require(Path(datasets['collection']).resolve() == collection and pilot.file_hash(collection / 'collection.json') == datasets['collection_sha256'], 'Datasets относятся к другому expert collection.')
    pool, source = pilot.load_pool(root, acquisition['pool'])
    initial, recipe = (info['initial_policy'], info['recipe'])
    require(initial == acquisition['source_initial_model'] == source['models']['initial'], 'Модели имеют разные исходные ACT.')
    require(info['target_budget'] == acquisition['target_budget'] and recipe['n_action_steps'] == 1, 'Нужен один demonstration budget и n_action_steps=1.')
    identities = {'initial': initial}
    source_items_1 = source['cases']
    training_ids = set()
    for case in source_items_1:
        training_ids.add(case['init_state_id'])
    source_items_2 = source['cases']
    training_hashes = set()
    for case in source_items_2:
        training_hashes.add(case['init_definition_sha256'])
    for method in experiment.METHODS:
        row = info['models'][method]
        dataset = datasets['datasets'][method]
        require(row['initial_policy'] == initial and row['recipe'] == recipe and (row['dataset'] == dataset), f'Не совпадает общий training recipe или dataset: {method}.')
        require(pilot.file_hash(Path(row['training_folder']) / 'run_config.json') == row['run_config_sha256'], f'Training record изменился: {method}.')
        run = pilot.read_json(Path(row['training_folder']) / 'run_config.json')
        require(run.get('status') == 'complete' and run['completed_updates'] == recipe['steps'] and (run['saved_policy'] == row['identity']), f'Не завершено одинаковое дообучение: {method}.')
        sources_file = Path(dataset['root']) / experiment.SOURCE_FILE
        require(pilot.file_hash(sources_file) == dataset['sources_sha256'], f'Dataset sources изменились: {method}.')
        sources = pilot.read_json(sources_file)
        require(sources['initial_policy'] == initial and sources['selector'] == method and (sources['n_episodes'] == dataset['n_episodes'] == info['target_budget']) and ({episode['candidate_id'] for episode in sources['episodes']} == set(acquisition['conditions'][method]['selected_successful_candidates'])), f'Training data не соответствуют выбранным demonstrations: {method}.')
        training_ids.update(sources['used_init_state_ids'])
        training_hashes.update(sources['used_init_definition_sha256'])
        identities[method] = row['identity']
    for identity in identities.values():
        validation.verify_model(identity)
    provenance = {'models_file': str(filename), 'models_sha256': pilot.file_hash(filename), 'models': identities, 'recipe': recipe, 'demonstration_budget': info['target_budget'], 'training_pool': str(pool), 'training_pool_sha256': pilot.file_hash(pool / 'protocol.json'), 'training_init_state_ids': sorted(training_ids), 'training_init_definition_sha256': sorted(training_hashes)}
    return (provenance, source)

def check_split(ids, provenance):
    require(bool(ids) and len(ids) == len(set(ids)), 'Нужны разные --init-ids.')
    require(set(ids).issubset(pilot.RESERVED_TEST_IDS), f'Для test заранее зарезервированы только IDs {pilot.RESERVED_TEST_IDS}.')
    require(not set(ids).intersection(provenance['training_init_state_ids']), 'Test configuration использована в collection/training или selector pool.')

def collect(args, root):
    provenance, source = load_models(root, args.models)
    check_split(args.init_ids, provenance)
    check_split(list(pilot.RESERVED_TEST_IDS), provenance)
    protocol = deepcopy(source)
    for key in ('cases', 'attempts', 'complete', 'status', 'counts'):
        protocol.pop(key, None)
    protocol.update(init_state_ids=args.init_ids, excluded_init_state_ids=provenance['training_init_state_ids'], scope=SCOPE, purpose='evaluation_only', models={'initial': provenance['models']['initial']})
    require(protocol['n_action_steps'] == 1 and protocol['recovery_horizon_steps'] == 280, 'Ожидается зафиксированный pilot protocol: n_action_steps=1, horizon=280.')
    output = pilot.new_folder(root, 'test_selection')
    info = {'status': 'running', 'scope': SCOPE, 'purpose': 'evaluation_only', 'protocol': protocol, 'provenance': provenance, 'requested_init_state_ids': args.init_ids, 'clean_init_state_ids': list(pilot.RESERVED_TEST_IDS), 'cases': [], 'attempts': [], 'runtime_versions': validation.runtime_versions(), 'inclusion_rule': 'All valid physical events on the predeclared configurations; no outcome filtering', 'ood_note': 'Held out from recovery learning; initial configurations may occur in nominal D0', 'script_sha256': pilot.file_hash(Path(__file__))}
    pilot.write_json(output / TEST_FILE, info)
    print('Test starts:', output, flush=True)
    try:
        for init_id in args.init_ids:
            for kind in pilot.FAILURE_TYPES:
                candidate = f'init_{init_id:03d}_{kind}'
                print(f'\nTEST COLLECT {candidate}: initial ACT; fixed intervention protocol', flush=True)
                request = {'stage': 'collect', 'project_root': str(root), 'protocol': protocol, 'output': str(output / candidate), 'candidate_id': candidate, 'failure_type': kind, 'init_state_id': init_id, 'model_path': provenance['models']['initial']['path']}
                result = pilot.run_worker(root, request)
                if result['status'] == 'accepted':
                    require(result['init_definition_sha256'] not in provenance['training_init_definition_sha256'], 'Test configuration физически совпадает с training configuration.')
                    result.update(scope=SCOPE, purpose='evaluation_only', independent_recovery_test=True)
                    pilot.write_json(output / candidate / 'case.json', result)
                    pilot.write_json(output / candidate / 'worker_result.json', result)
                    info['cases'].append(result)
                    print('Accepted:', kind, result['verification'], flush=True)
                else:
                    print('Skipped:', result['reason'], flush=True)
                info['attempts'].append(result)
                pilot.write_json(output / TEST_FILE, info)
        validation.verify_model(provenance['models']['initial'])
        counts = Counter((case['failure_type'] for case in info['cases']))
        info['counts'] = dict(counts)
        info['status'] = 'test_complete' if all((counts[kind] > 0 for kind in pilot.FAILURE_TYPES)) else 'incomplete'
        pilot.write_json(output / TEST_FILE, info)
    except BaseException as error:
        info.update(status='interrupted', error=f'{type(error).__name__}: {error}')
        pilot.write_json(output / TEST_FILE, info)
        raise
    if info['status'] != 'test_complete':
        raise pilot.StageNotReady(f'Нет обоих failure types. Состояния и причины пропусков сохранены: {output}')
    pilot.write_pointer(root, TEST_POINTER, output)
    print(f"\nTEST STATES READY: {len(info['cases'])} starts; types={dict(counts)}")
    print('Expert recovery and uncertainty scoring were not run for the test states.')

def load_test(root, value=None):
    pool = pilot.local_path(root, value) if value else pilot.resolve_pointer(root, TEST_POINTER)
    info = pilot.read_json(pool / TEST_FILE)
    if info.get('status') != 'test_complete' or info.get('purpose') != 'evaluation_only':
        raise pilot.StageNotReady('Сначала нужен TEST STATES READY с обоими failure types.')
    current, _ = load_models(root, info['provenance']['models_file'])
    require(current == info['provenance'], 'Модели или training provenance изменились после фиксации test.')
    check_split(info['requested_init_state_ids'], current)
    check_split(info['clean_init_state_ids'], current)
    cases = info['cases']
    require(len(cases) == len({case['candidate_id'] for case in cases}) and set((case['failure_type'] for case in cases)) == set(pilot.FAILURE_TYPES), 'Неполный test или повторные cases.')
    for case in cases:
        require(case['init_state_id'] in info['requested_init_state_ids'] and case['init_definition_sha256'] not in current['training_init_definition_sha256'] and (case.get('independent_recovery_test') is True) and (case['label_source'] == 'explicit_simulator_event_rules'), 'Нарушено отделение test от training.')
        for filename, digest in (('start.npz', case['snapshot_sha256']), ('history.npz', case['history_sha256'])):
            require(pilot.file_hash(pool / case['candidate_id'] / filename) == digest, 'Test snapshot/history изменились.')
    protocol = info['protocol']
    require(protocol['n_action_steps'] == 1 and protocol['recovery_horizon_steps'] == 280 and (protocol['excluded_init_state_ids'] == current['training_init_state_ids']), 'Изменён общий test protocol.')
    return (pool, info)

def paired_counts(first, second):
    require(len(first) == len(second) > 0, 'Для разницы нужны парные результаты на одинаковых states.')
    return {'both_success': sum((a and b for a, b in zip(first, second))), 'both_fail': sum((not a and (not b) for a, b in zip(first, second))), 'first_only_success': sum((a and (not b) for a, b in zip(first, second))), 'second_only_success': sum((not a and b for a, b in zip(first, second))), 'difference_percentage_points': 100 * sum((int(a) - int(b) for a, b in zip(first, second))) / len(first)}

def recovery_summary(cases, results, budget):
    require(bool(cases), 'Нет test cases.')
    source_items_3 = cases
    ids = []
    for case in source_items_3:
        ids.append(case['candidate_id'])
    require(len(set(ids)) == len(ids) and set(results) == set(ids), 'Результаты покрывают другой test set.')
    source_items_4 = cases
    items_4 = set()
    for c in source_items_4:
        items_4.add(c['init_state_id'])
    summary = {'n_states': len(ids), 'n_initial_configurations': len(items_4), 'demonstration_budget': budget, 'models': {}, 'F_ref_minus_U': {}, 'scope': 'Fixed independent single-task test; ACT ensemble proxy, reference failure labels', 'uncertainty_note': 'Small single-training-seed pilot. Two failure starts from the same configuration are related; no reliable population CI or significance claim is made.'}
    series = {}
    for variant in VARIANTS:
        require(all((set(results[cid]) == set(VARIANTS) for cid in ids)), 'Не все четыре политики оценены на всех states.')
        source_items_5 = ids
        items_5 = []
        for cid in source_items_5:
            items_5.append(results[cid][variant]['success'])
        series[variant] = items_5
        require(all((isinstance(x, bool) for x in series[variant])), 'Success должен быть boolean benchmark outcome.')
        count = sum(series[variant])
        groups = {}
        for kind in pilot.FAILURE_TYPES:
            source_items_6 = cases
            successes = []
            for c in source_items_6:
                if c['failure_type'] == kind:
                    successes.append(results[c['candidate_id']][variant]['success'])
            require(bool(successes), 'Нет test starts одной категории.')
            groups[kind] = {'n_success': sum(successes), 'n_episodes': len(successes), 'pc_success': 100 * sum(successes) / len(successes)}
        summary['models'][variant] = {'n_success': count, 'n_episodes': len(ids), 'pc_success': 100 * count / len(ids), 'per_failure_type': groups, 'macro_pc_success': sum((g['pc_success'] for g in groups.values())) / len(groups)}
    summary['F_ref_minus_U'] = paired_counts(series['F_ref'], series['U'])
    summary['F_ref_minus_U'].update(first='F_ref', second='U')
    source_items_7 = experiment.METHODS
    items_7 = {}
    for variant in source_items_7:
        items_7[variant] = paired_counts(series[variant], series['initial'])
    summary['against_initial'] = items_7
    return summary

def run_folder(root, mode, args, pool, info):
    frozen = {'mode': mode, 'test_pool': str(pool), 'test_pool_sha256': pilot.file_hash(pool / TEST_FILE), 'provenance': info['provenance'], 'n_action_steps': 1, 'horizon_steps': 280}
    if mode == 'clean':
        frozen['clean_init_state_ids'] = info['clean_init_state_ids']
    if args.resume:
        output = pilot.local_path(root, args.resume)
        require(pilot.read_json(output / 'run_info.json') == frozen, 'Resume относится к другому test или моделям.')
        partial = pilot.read_json(output / 'partial_results.json')
    else:
        output = pilot.new_folder(root, mode + '_selection_eval')
        pilot.write_json(output / 'run_info.json', frozen)
        partial = {}
        pilot.write_json(output / 'partial_results.json', partial)
    print('Evaluation:', output, flush=True)
    return (output, partial)

def recovery(args, root):
    pool, info = load_test(root, args.pool)
    output, results = run_folder(root, 'recovery', args, pool, info)
    protocol, identities = (info['protocol'], info['provenance']['models'])
    for index, case in enumerate(info['cases'], 1):
        candidate = case['candidate_id']
        row = results.setdefault(candidate, {})
        for variant in VARIANTS:
            if variant in row:
                continue
            print(f"\nRECOVERY {index}/{len(info['cases'])}, {candidate}, {variant}", flush=True)
            validation.verify_model(identities[variant])
            require(pilot.file_hash(pool / candidate / 'start.npz') == case['snapshot_sha256'], 'Snapshot изменился.')
            attempt_folder = output / candidate / variant / datetime.now().strftime('attempt_%Y%m%d_%H%M%S_%f')
            request = {'stage': 'evaluate', 'variant': variant, 'protocol': protocol, 'project_root': str(root), 'output': str(attempt_folder), 'case_dir': str(pool / candidate), 'init_state_id': case['init_state_id'], 'init_definition_sha256': case['init_definition_sha256'], 'model_path': identities[variant]['path']}
            result = validation.run_worker(root, request)
            require(result['status'] == 'evaluated' and result['variant'] == variant and (result['model'] == identities[variant]['path']) and (result['init_state_id'] == case['init_state_id']) and (1 <= result['steps'] <= 280) and (result['restore']['recovery_horizon_steps'] == 280), 'Evaluator вернул результат другого model/state/protocol.')
            result.update(candidate_id=candidate, failure_type=case['failure_type'], source_snapshot_sha256=case['snapshot_sha256'], result_folder=str(attempt_folder))
            row[variant] = result
            pilot.write_json(output / 'partial_results.json', results)
            print(f"{variant}: success={result['success']}; steps={result['steps']}; restore error={result['restore']['max_state_error']:.3g}", flush=True)
    load_test(root, pool)
    summary = recovery_summary(info['cases'], results, info['provenance']['demonstration_budget'])
    pilot.write_json(output / 'summary.json', summary)
    pilot.write_pointer(root, RECOVERY_POINTER, output)
    print('\n=== INDEPENDENT SELECTOR COMPARISON ===')
    for variant, row in summary['models'].items():
        print(f"{variant}: {row['n_success']}/{row['n_episodes']} = {row['pc_success']:.1f}%")
        source_items_8 = row['per_failure_type'].items()
        items_8 = {}
        for k, g in source_items_8:
            items_8[k] = f"{g['n_success']}/{g['n_episodes']}"
        print('  By failure type:', items_8)
    print(f"F_ref − U: {summary['F_ref_minus_U']['difference_percentage_points']:+.1f} percentage points")
    print('Output:', output)

def clean_worker(request):
    from gymnasium.vector import SyncVectorEnv
    root, output = (Path(request['project_root']), Path(request['output']))
    os.chdir(root)
    identities, variant = (request['models'], request['variant'])
    validation.verify_model(identities[variant])
    init_ids, excluded = (request['init_ids'], request['excluded_init_ids'])
    original_reset, old_argv = (SyncVectorEnv.reset, sys.argv)
    resets, vectors = ([], [])

    def reset(vec, *args, **kwargs):
        index = len(resets)
        require(index < len(init_ids), 'Clean evaluator запросил лишний reset.')
        init_id = init_ids[index]
        definition = validation.choose_initial_state(vec, init_id, excluded)
        result = original_reset(vec, *args, **kwargs)
        vectors.append(vec)
        state, ctrl = validation.state_and_ctrl(vec)
        obs = pilot.raw_observation(result[0])
        require(not pilot.goal_at_start(vec), 'Обычное начальное состояние уже удовлетворяет goal.')
        source_items_9 = obs.items()
        items_9 = {}
        for key, value in source_items_9:
            items_9[key] = validation.array_hash(value)
        resets.append({'init_state_id': init_id, 'init_definition_sha256': definition, 'state_sha256': validation.array_hash(state), 'ctrl_sha256': validation.array_hash(ctrl), 'observations_sha256': items_9})
        pilot.write_json(output / 'clean_starts.json', resets)
        return result
    sys.argv = ['lerobot-eval', f"--policy.path={identities[variant]['path']}", '--policy.device=cuda', '--policy.n_action_steps=1', '--env.type=libero', '--env.task=libero_spatial', '--env.task_ids=[0]', '--env.control_mode=relative', '--env.init_states=true', '--env.hard_reset=true', '--env.max_parallel_tasks=1', '--env.observation_height=256', '--env.observation_width=256', '--env.episode_length=280', '--eval.batch_size=1', f'--eval.n_episodes={len(init_ids)}', '--eval.use_async_envs=false', '--seed=0', f"--output_dir={output / 'evaluation'}"]
    try:
        SyncVectorEnv.reset = reset
        runpy.run_module('lerobot.scripts.lerobot_eval', run_name='__main__')
    finally:
        SyncVectorEnv.reset, sys.argv = (original_reset, old_argv)
        source_items_10 = vectors
        items_10 = {}
        for vec in source_items_10:
            items_10[id(vec)] = vec
        for vec in items_10.values():
            if not getattr(vec, 'closed', False):
                vec.close()
    info = pilot.read_json(output / 'evaluation/eval_info.json')
    require(len(info['per_task']) == 1, 'Clean evaluation должна содержать одну task.')
    successes = info['per_task'][0]['metrics']['successes']
    require(len(successes) == len(resets) == len(init_ids) and all((isinstance(x, bool) for x in successes)) and (sum(successes) == info['overall']['n_success']), 'Неполная clean evaluation.')
    validation.verify_model(identities[variant])
    return {'status': 'evaluated', 'variant': variant, 'successes': successes, 'starts': resets, 'n_success': sum(successes), 'n_episodes': len(init_ids), 'pc_success': 100 * sum(successes) / len(init_ids), 'videos': info['overall'].get('video_paths', []), 'result_folder': str(output)}

def run_clean_child(root, request):
    output = Path(request['output'])
    output.mkdir(parents=True, exist_ok=False)
    pilot.write_json(output / 'request.json', request)
    env = dict(os.environ, OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', WANDB_MODE='disabled', HF_HUB_DISABLE_TELEMETRY='1')
    with (output / 'worker.log').open('w', encoding='utf-8') as log:
        completed = subprocess.run([sys.executable, str(Path(__file__).resolve()), '_clean-worker', '--request', str(output / 'request.json')], cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT)
    if completed.returncode or not (output / 'worker_result.json').is_file():
        print('\n'.join((output / 'worker.log').read_text(errors='replace').splitlines()[-45:]), flush=True)
        raise RuntimeError(f"Clean evaluation остановилась. Лог: {output / 'worker.log'}")
    result = pilot.read_json(output / 'worker_result.json')
    require(result.get('status') == 'evaluated' and result['variant'] == request['variant'], 'Некорректный clean result.')
    return result

def clean(args, root):
    pool, info = load_test(root, args.pool)
    output, results = run_folder(root, 'clean', args, pool, info)
    for variant in VARIANTS:
        if variant in results:
            continue
        print(f"\nCLEAN {variant}: {len(info['clean_init_state_ids'])} nominal initial configurations", flush=True)
        folder = output / variant / datetime.now().strftime('attempt_%Y%m%d_%H%M%S_%f')
        request = {'project_root': str(root), 'output': str(folder), 'variant': variant, 'models': info['provenance']['models'], 'init_ids': info['clean_init_state_ids'], 'excluded_init_ids': info['provenance']['training_init_state_ids']}
        result = run_clean_child(root, request)
        if variant != 'initial':
            require(result['starts'] == results['initial']['starts'], 'Clean policies получили разные исходные состояния.')
        results[variant] = result
        pilot.write_json(output / 'partial_results.json', results)
        print(f"{variant}: {result['n_success']}/{result['n_episodes']}", flush=True)
    load_test(root, pool)
    source_items_11 = experiment.METHODS
    items_11 = {}
    for variant in source_items_11:
        items_11[variant] = results[variant]['pc_success'] - results['initial']['pc_success']
    source_items_12 = experiment.METHODS
    items_12 = {}
    for variant in source_items_12:
        items_12[variant] = paired_counts(results[variant]['successes'], results['initial']['successes'])
    summary = {'n_states': len(info['clean_init_state_ids']), 'models': results, 'difference_from_initial_percentage_points': items_11, 'paired_against_initial': items_12, 'scope': 'Clean-task retention on the same ordinary initial states, without induced failures or expert assistance', 'note': 'Single-task, single-training-seed pilot; exact point estimates, no significance claim.'}
    pilot.write_json(output / 'summary.json', summary)
    pilot.write_pointer(root, CLEAN_POINTER, output)
    print('\n=== CLEAN-TASK RETENTION ===')
    for variant in VARIANTS:
        row = results[variant]
        change = '' if variant == 'initial' else f"; относительно initial: {summary['difference_from_initial_percentage_points'][variant]:+.1f} п.п."
        print(f"{variant}: {row['n_success']}/{row['n_episodes']} = {row['pc_success']:.1f}%{change}")
    print('Output:', output)

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--project-root', type=Path, default=Path(__file__).resolve().parent)
    commands = parser.add_subparsers(dest='command', required=True)
    collection = commands.add_parser('collect', help='Новые TEST starts; без expert и uncertainty scoring.')
    collection.add_argument('--init-ids', type=int, nargs='+', default=[15, 45])
    collection.add_argument('--models', type=Path)
    for name in ('recovery', 'clean'):
        command = commands.add_parser(name)
        command.add_argument('--pool', type=Path)
        command.add_argument('--resume', type=Path)
    worker = commands.add_parser('_clean-worker', help=argparse.SUPPRESS)
    worker.add_argument('--request', type=Path, required=True)
    args = parser.parse_args()
    root = args.project_root.resolve()
    try:
        if args.command == '_clean-worker':
            request = pilot.read_json(args.request)
            try:
                result = clean_worker(request)
            except BaseException as error:
                pilot.write_json(Path(request['output']) / 'worker_result.json', {'status': 'error', 'error': str(error), 'traceback': traceback.format_exc()})
                raise
            pilot.write_json(Path(request['output']) / 'worker_result.json', result)
        else:
            {'collect': collect, 'recovery': recovery, 'clean': clean}[args.command](args, root)
    except (pilot.StageNotReady, FileNotFoundError, ValueError, RuntimeError) as error:
        print(f'\nERROR: {error}', file=sys.stderr, flush=True)
        sys.exit(2)
if __name__ == '__main__':
    main()
