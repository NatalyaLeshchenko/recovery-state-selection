#!/usr/bin/env python3
from __future__ import annotations
import argparse
from copy import deepcopy
from datetime import datetime
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
SCOPE = 'ACT ensemble-disagreement proxy; not INSIGHT or VLA validation'
POLICY_IGNORE = {'device', 'pretrained_path', 'push_to_hub', 'repo_id'}
REQUIRED = ('config.json', 'model.safetensors', 'train_config.json', 'policy_preprocessor.json', 'policy_postprocessor.json')

def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))

def write_json(path, value):
    temporary = Path(path).with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    temporary.replace(path)

def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:

        def read_chunk():
            return stream.read(1024 * 1024)
        for chunk in iter(read_chunk, b''):
            digest.update(chunk)
    return digest.hexdigest()

def policy_settings(config):
    source_items_1 = config.items()
    items_1 = {}
    for key, value in source_items_1:
        if key not in POLICY_IGNORE:
            items_1[key] = value
    return items_1

def checked_source(train, policy, manifest):
    if policy.get('type') != 'act' or train.get('policy', {}).get('type') != 'act':
        raise RuntimeError('Нужна исходная ACT, обученная на D0.')
    if train.get('resume') is True or train['policy'].get('pretrained_path'):
        raise RuntimeError('Исходный запуск был resume/fine-tuning; нельзя считать его независимым nominal member.')
    for key in ('dataset_repo', 'dataset_revision', 'train_episodes'):
        if key not in manifest:
            raise RuntimeError(f'Нет {key} в pilot_runs/manifest.json.')
    dataset = train.get('dataset', {})
    if dataset.get('repo_id') != manifest['dataset_repo']:
        raise RuntimeError('Dataset исходного обучения отличается от D0 в manifest.')
    if dataset.get('revision') != manifest['dataset_revision']:
        raise RuntimeError('Dataset revision отличается от зафиксированного D0.')
    episodes = dataset.get('episodes')
    expected = manifest['train_episodes']
    if not isinstance(episodes, list) or not episodes or len(episodes) != len(set(episodes)) or (sorted(episodes) != sorted(expected)):
        raise RuntimeError('Train episode IDs не совпадают с исходным D0.')
    if set(episodes) & set(manifest.get('validation_episodes', [])):
        raise RuntimeError('Nominal validation episodes включены в train split.')
    for key in ('steps', 'batch_size', 'seed'):
        if not isinstance(train.get(key), int) or isinstance(train[key], bool):
            raise RuntimeError(f'Некорректное поле {key} в исходном train_config.json.')
    if train['steps'] <= 0 or not 1 <= train['batch_size'] <= 2:
        raise RuntimeError('Нужно исходное обучение с batch_size 1 или 2 и положительным steps.')
    if policy_settings(train['policy']) != policy_settings(policy):
        raise RuntimeError('Policy settings в train_config.json и config.json различаются.')
    if train.get('reward_model') is not None or train.get('peft') is not None:
        raise RuntimeError('Этот скрипт рассчитан на обычное nominal ACT training без reward model/PEFT.')
    job = train.get('job')
    if isinstance(job, dict) and (job.get('is_remote') is True or job.get('provider') not in (None, '', 'local')):
        raise RuntimeError('Исходный config настроен на remote job; нужен локальный training config.')
    return {'repo_id': dataset['repo_id'], 'revision': dataset['revision'], 'episodes': list(episodes)}

def identity(path):
    names = set(REQUIRED)
    for pattern in ('policy_preprocessor*', 'policy_postprocessor*'):
        names.update((p.name for p in path.glob(pattern) if p.is_file()))
    for name in names:
        if not (path / name).is_file():
            raise FileNotFoundError(path / name)
    source_items_2 = sorted(names)
    items_2 = {}
    for name in source_items_2:
        items_2[name] = file_hash(path / name)
    return {'path': str(path), 'sha256': items_2}

def verify_identity(record):
    path = Path(record['path'])
    if any((file_hash(path / name) != expected for name, expected in record['sha256'].items())):
        raise RuntimeError('Файлы исходной модели изменились во время подготовки ensemble.')

def context(root, initial_path=None):
    path = initial_path or Path('pilot_runs/training/initial/checkpoints/010000/pretrained_model')
    path = Path(path).expanduser()
    initial = (path if path.is_absolute() else root / path).resolve()
    source = identity(initial)
    train, policy = (read_json(initial / 'train_config.json'), read_json(initial / 'config.json'))
    manifest_path = root / 'pilot_runs/manifest.json'
    manifest = read_json(manifest_path)
    d0 = checked_source(train, policy, manifest)
    checkpoint_name = initial.parent.name
    if checkpoint_name.isdigit() and int(checkpoint_name) != train['steps']:
        raise RuntimeError('Checkpoint не соответствует полному числу updates исходного обучения.')
    if not checkpoint_name.isdigit():
        raise RuntimeError('Укажите исходный numbered checkpoint, например checkpoints/010000/pretrained_model.')
    return {'root': root, 'initial': initial, 'identity': source, 'train': train, 'policy': policy, 'd0': d0, 'manifest_sha256': file_hash(manifest_path)}

def member_config(source, output, seed, smoke=False):
    if seed == source['seed']:
        raise RuntimeError('Seed нового member должен отличаться от исходного.')
    result = deepcopy(source)
    result.update(seed=seed, output_dir=str(output), resume=False, num_workers=0, job_name=f'act_uncertainty_seed{seed}', env=None, env_eval_freq=0)
    if smoke:
        result['steps'] = min(20, source['steps'])
    result['save_freq'] = result['steps']
    result['save_checkpoint'] = True
    result['log_freq'] = min(20, result['steps'])
    result['policy']['device'] = 'cuda'
    result['policy']['pretrained_path'] = None
    result['policy']['push_to_hub'] = False
    if isinstance(result.get('wandb'), dict):
        result['wandb']['enable'] = False
    if 'save_checkpoint_to_hub' in result:
        result['save_checkpoint_to_hub'] = False
    if 'persistent_workers' in result:
        result['persistent_workers'] = False
    if 'checkpoint_path' in result:
        result['checkpoint_path'] = None
    return result

def training_command(config_path):
    return [sys.executable, '-m', 'lerobot.scripts.lerobot_train', f'--config_path={config_path}']

def memory_info():
    info = {}
    path = Path('/proc/meminfo')
    if path.is_file():
        for line in path.read_text().splitlines():
            name, value = line.split(':', 1)
            if name in ('MemTotal', 'MemAvailable', 'SwapTotal', 'SwapFree'):
                info[name + '_GiB'] = round(int(value.strip().split()[0]) / 1024 ** 2, 2)
    return info

def runtime_report(ctx):
    packages = {}
    for name in ('lerobot', 'torch', 'hf-libero'):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = 'not installed'
    return {'scope': SCOPE, 'initial_model': str(ctx['initial']), 'initial_seed': ctx['train']['seed'], 'updates_per_member': ctx['train']['steps'], 'batch_size': ctx['train']['batch_size'], 'new_num_workers': 0, 'D0': ctx['d0'], 'memory': memory_info(), 'disk_free_GiB': round(shutil.disk_usage(ctx['root']).free / 1024 ** 3, 2), 'versions': packages}

def run_members(ctx, smoke):
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA недоступна; обучение не запущено.')
    root = ctx['root']
    verify_identity(ctx['identity'])
    output = root / 'pilot_runs/uncertainty' / datetime.now().strftime(('smoke' if smoke else 'ensemble') + '_%Y%m%d_%H%M%S_%f')
    output.mkdir(parents=True, exist_ok=False)
    initial_seed = ctx['train']['seed']
    seeds = [initial_seed + 1] if smoke else [initial_seed + 1, initial_seed + 2]
    report = {'status': 'running', 'complete': False, 'smoke_only': smoke, 'scope': SCOPE, 'runtime': runtime_report(ctx), 'D0': ctx['d0'], 'manifest_sha256': ctx['manifest_sha256'], 'models': [{'seed': initial_seed, 'identity': ctx['identity'], 'role': 'fixed_initial_actor_and_member0'}], 'attempts': [], 'execution': 'One training process at a time; no simultaneous models on CUDA'}
    write_json(output / 'ensemble.json', report)
    print('Output:', output, flush=True)
    child_env = dict(os.environ, WANDB_MODE='disabled', HF_HUB_DISABLE_TELEMETRY='1', OMP_NUM_THREADS='4', MKL_NUM_THREADS='4')
    try:
        for seed in seeds:
            target = output / f'seed_{seed}'
            cfg = member_config(ctx['train'], target, seed, smoke=smoke)
            config_path = output / f'seed_{seed}_train_config.json'
            write_json(config_path, cfg)
            command = training_command(config_path)
            attempt = {'seed': seed, 'command': command, 'config': str(config_path), 'complete': False}
            report['attempts'].append(attempt)
            write_json(output / 'ensemble.json', report)
            print(f"\nACT seed={seed}: D0, updates={cfg['steps']}, batch={cfg['batch_size']}, workers=0", flush=True)
            subprocess.run(command, cwd=root, env=child_env, check=True)
            trained = (target / 'checkpoints/last/pretrained_model').resolve()
            record = identity(trained)
            if policy_settings(read_json(trained / 'config.json')) != policy_settings(ctx['policy']):
                raise RuntimeError('Архитектура trained member отличается от исходной ACT.')
            saved = read_json(trained / 'train_config.json')
            if saved.get('seed') != seed or saved.get('steps') != cfg['steps'] or saved.get('batch_size') != cfg['batch_size'] or (saved.get('dataset') != cfg['dataset']):
                raise RuntimeError('Сохранённый training config отличается от запланированного.')
            if saved['policy'].get('pretrained_path') or saved.get('resume') is True:
                raise RuntimeError('Member загрузил готовые action-model weights вместо fresh initialization.')
            source_items_3 = report['models']
            existing_hashes = []
            for m in source_items_3:
                existing_hashes.append(m['identity']['sha256']['model.safetensors'])
            if record['sha256']['model.safetensors'] in existing_hashes:
                raise RuntimeError('Получены идентичные weights; независимость members не подтверждена.')
            attempt['complete'] = True
            report['models'].append({'seed': seed, 'identity': record, 'role': 'uncertainty_member'})
            write_json(output / 'ensemble.json', report)
        verify_identity(ctx['identity'])
        if file_hash(root / 'pilot_runs/manifest.json') != ctx['manifest_sha256']:
            raise RuntimeError('D0 manifest изменился во время обучения.')
        report.update(status='complete', complete=True)
        write_json(output / 'ensemble.json', report)
        if not smoke:
            write_pointer = root / 'pilot_runs/uncertainty_ensemble_path.txt'
            temporary = write_pointer.with_suffix('.tmp')
            temporary.write_text(str(output) + '\n', encoding='utf-8')
            temporary.replace(write_pointer)
            print('\nENSEMBLE MEMBERS READY: 3', flush=True)
        else:
            print('\nSMOKE COMPLETE', flush=True)
        print('Output:', output, flush=True)
    except BaseException as error:
        report.update(status='interrupted' if isinstance(error, KeyboardInterrupt) else 'failed', error=str(error))
        write_json(output / 'ensemble.json', report)
        raise

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('mode', choices=('inspect', 'smoke', 'train'))
    parser.add_argument('--project-root', type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument('--initial-policy', type=Path)
    args = parser.parse_args()
    root = args.project_root.expanduser().resolve()
    ctx = context(root, args.initial_policy)
    print(json.dumps(runtime_report(ctx), ensure_ascii=False, indent=2), flush=True)
    if args.mode != 'inspect':
        run_members(ctx, smoke=args.mode == 'smoke')
    else:
        print('INSPECT OK')
if __name__ == '__main__':
    try:
        main()
    except (RuntimeError, FileNotFoundError, FileExistsError, subprocess.CalledProcessError) as exc:
        print('ERROR:', exc, file=sys.stderr)
        raise SystemExit(1)
