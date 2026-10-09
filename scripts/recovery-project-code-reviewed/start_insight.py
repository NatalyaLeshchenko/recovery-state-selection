#!/usr/bin/env python3
# Token sampling adapted from OpenPI (Apache-2.0), commit 215abfb217dbac7d5f1273282331b9b1866c0479.
# Upstream: https://github.com/Physical-Intelligence/openpi
# INSIGHT: https://github.com/ulaskarli/insight-vla-help-triggers
from __future__ import annotations
import argparse
import base64
from contextlib import contextmanager
from datetime import datetime, timezone
import dataclasses
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import traceback
from types import MethodType
import urllib.request
import urllib.parse
OPENPI_COMMIT = '215abfb217dbac7d5f1273282331b9b1866c0479'
INSIGHT_COMMIT = 'de2eba0ce723fa683f00f4729959d0e24fb36fad'
CHECKPOINT = 'gs://openpi-assets/checkpoints/pi0_fast_libero'
FEATURE_ORDER = ['AU', 'EU', 'entropy', 'chosen_token_log_probability']
GIB = 1024 ** 3
CHECKPOINT_PREFIX = 'checkpoints/pi0_fast_libero/'
INVENTORY_URL = 'https://storage.googleapis.com/storage/v1/b/openpi-assets/o?prefix=checkpoints/pi0_fast_libero/&maxResults=1000'
RESTORE_CONCURRENT_GB = 3
MIN_WORKER_RAM = 11 * GIB

def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8')
    temporary.replace(path)

def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))

def digest(path):
    with Path(path).open('rb') as file:
        result = hashlib.file_digest(file, 'sha256')
    return result.hexdigest()

def new_run(root, name):
    stamp = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')
    path = root / 'pilot_runs' / 'vla_insight' / (name + '_' + stamp)
    path.mkdir(parents=True, exist_ok=False)
    return path

def run(argv, *, cwd=None, env=None):
    print('RUN:', ' '.join(map(str, argv)), flush=True)
    subprocess.run(list(map(str, argv)), cwd=cwd, env=env, check=True)

def public_json(url):
    request = urllib.request.Request(url, headers={'User-Agent': 'vla-insight-probe'})
    with urllib.request.urlopen(request, timeout=45) as response:
        return json.load(response)

def checkpoint_inventory():
    curl = shutil.which('curl')
    if not curl:
        raise RuntimeError('Нужен curl: sudo apt install curl. Затем повтори download.')
    result = subprocess.run([curl, '--disable', '--http1.1', '--location', '--fail', '--silent', '--show-error', '--connect-timeout', '30', '--max-time', '45', '--retry', '2', '--retry-all-errors', '--retry-delay', '2', INVENTORY_URL], capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError('Не удалось получить список весов: ' + result.stderr[-2000:])
    return parse_checkpoint_inventory(json.loads(result.stdout))

def parse_checkpoint_inventory(inventory):
    if inventory.get('nextPageToken') or not inventory.get('items'):
        raise RuntimeError('Не удалось проверить полный публичный checkpoint inventory.')
    files = []
    for item in inventory['items']:
        name = item['name']
        if not name.startswith(CHECKPOINT_PREFIX):
            raise RuntimeError('Checkpoint inventory содержит посторонний путь.')
        if name.endswith('/') and int(item['size']) == 0:
            continue
        relative = name[len(CHECKPOINT_PREFIX):]
        if not relative or any((part in ('', '.', '..') for part in relative.split('/'))):
            raise RuntimeError('Некорректный путь в checkpoint inventory.')
        checksum = item.get('md5Hash')
        if not checksum or len(base64.b64decode(checksum, validate=True)) != 16:
            raise RuntimeError('Нет ожидаемой контрольной суммы MD5 для ' + name)
        generation = str(item['generation'])
        if not generation.isdecimal():
            raise RuntimeError('Некорректная версия объекта checkpoint.')
        files.append({'name': name, 'relative': relative, 'size': int(item['size']), 'generation': generation, 'md5Hash': checksum})
    source_items_1 = files
    paths = set()
    for item in source_items_1:
        paths.add(item['relative'])
    if len(paths) != len(files) or not {'params/manifest.ocdbt', 'assets/physical-intelligence/libero/norm_stats.json'}.issubset(paths):
        raise RuntimeError('Checkpoint inventory не содержит обязательных файлов.')

    def checkpoint_file_order(item):
        return (item['size'], item['relative'])
    return sorted(files, key=checkpoint_file_order)

def checkpoint_cache():
    base = Path(os.environ.get('OPENPI_DATA_HOME', '~/.cache/openpi')).expanduser().resolve()
    return base / 'openpi-assets' / 'checkpoints' / 'pi0_fast_libero'

@contextmanager
def checkpoint_lock(checkpoint):
    import fcntl
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    with checkpoint.with_suffix('.lock').open('a') as file:
        try:
            fcntl.flock(file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError('Предыдущая загрузка ещё работает. Останови её Ctrl+C и повтори download.') from error
        try:
            yield
        finally:
            fcntl.flock(file.fileno(), fcntl.LOCK_UN)

def verified_file(path, item):
    if not path.is_file() or path.stat().st_size != item['size']:
        return False
    with path.open('rb') as file:

        def new_md5():
            return hashlib.md5(usedforsecurity=False)
        value = hashlib.file_digest(file, new_md5).digest()
    return base64.b64encode(value).decode('ascii') == item['md5Hash']

def object_url(item):
    name = urllib.parse.quote(item['name'], safe='')
    query = urllib.parse.urlencode({'alt': 'media', 'generation': item['generation']})
    return f'https://storage.googleapis.com/download/storage/v1/b/openpi-assets/o/{name}?{query}'

def download_object(item, destination, old_partial):
    if verified_file(destination, item):
        print('  VERIFIED:', item['relative'], flush=True)
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + '.download')
    sources = [destination, partial, old_partial / item['relative'], old_partial / 'pi0_fast_libero' / item['relative']]
    for source in sources:
        if source != destination and verified_file(source, item):
            source.replace(destination)
            print('  REUSED:', item['relative'], flush=True)
            return
    source_items_2 = sources
    resumable = []
    for path in source_items_2:
        if path.is_file() and 0 < path.stat().st_size < item['size']:
            resumable.append(path)
    if resumable:

        def file_size(path):
            return path.stat().st_size
        best = max(resumable, key=file_size)
        if best != partial:
            best.replace(partial)
    elif partial.exists():
        partial.replace(partial.with_name(partial.name + '.invalid_' + str(time.time_ns())))
    for attempt in range(1, 7):
        if partial.exists() and partial.stat().st_size >= item['size']:
            partial.replace(partial.with_name(partial.name + '.invalid_' + str(time.time_ns())))
        offset = partial.stat().st_size if partial.exists() else 0
        print(f"  DOWNLOAD {attempt}/6: {item['relative']} ({offset / GIB:.3f}/{item['size'] / GIB:.3f} GiB)", flush=True)
        result = subprocess.run([shutil.which('curl'), '--disable', '--http1.1', '--location', '--fail', '--connect-timeout', '30', '--speed-limit', '1024', '--speed-time', '60', '--continue-at', '-', '--output', str(partial), '--url', object_url(item)])
        if verified_file(partial, item):
            partial.replace(destination)
            print('  MD5 OK:', item['relative'], flush=True)
            return
        if result.returncode in (22, 23, 33, 60):
            raise RuntimeError(f"curl exit={result.returncode} для {item['relative']}. Неполный файл сохранён; checkpoint не принят.")
        print('  CHECKSUM MISMATCH: retrying download', flush=True)
        if attempt < 6:
            time.sleep(2)
    raise RuntimeError('Не удалось скачать ' + item['relative'] + '. Прогресс сохранён; повтори download позже.')

def ensure_checkpoint(files=None):
    files = checkpoint_inventory() if files is None else files
    checkpoint = checkpoint_cache()
    size = sum((item['size'] for item in files))
    print(f'Checkpoint on disk: {size / GIB:.2f} GiB.', flush=True)
    with checkpoint_lock(checkpoint):
        old_partial = checkpoint.with_suffix('.partial')
        checkpoint.mkdir(parents=True, exist_ok=True)
        missing = sum((item['size'] for item in files if not verified_file(checkpoint / item['relative'], item)))
        if shutil.disk_usage(checkpoint).free < missing + 2 * GIB:
            raise RuntimeError('Недостаточно места на SSD для недостающих весов и кеша.')
        for number, item in enumerate(files, start=1):
            print(f'CHECKPOINT FILE {number}/{len(files)}', flush=True)
            download_object(item, checkpoint / item['relative'], old_partial)
        write_json(checkpoint / 'verified_download.json', {'checkpoint': CHECKPOINT, 'files': files, 'verified_at': datetime.now(timezone.utc).isoformat(), 'transport': 'curl HTTPS'})
    print('CHECKPOINT VERIFIED:', checkpoint, flush=True)
    return (checkpoint, files)

def memory_info():
    values = {}
    path = Path('/proc/meminfo')
    if path.exists():
        for line in path.read_text().splitlines():
            key, value = line.split(':', 1)
            if key in ('MemTotal', 'MemAvailable', 'SwapTotal', 'SwapFree'):
                values[key] = int(value.split()[0]) * 1024
    return values

def gpu_info():
    result = subprocess.run(['nvidia-smi', '--query-gpu=name,driver_version,memory.total,memory.free', '--format=csv,noheader,nounits'], capture_output=True, text=True, check=True)
    return result.stdout.strip()

def isolated_env(root):
    env = dict(os.environ)
    env.pop('VIRTUAL_ENV', None)
    env.pop('PYTHONPATH', None)
    env['GIT_LFS_SKIP_SMUDGE'] = '1'
    env['UV_PROJECT_ENVIRONMENT'] = str(root / 'vla_insight' / 'openpi' / '.venv')
    env['OPENPI_DATA_HOME'] = str(root / 'vla_insight' / 'model_cache')
    env['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
    env['XLA_PYTHON_CLIENT_ALLOCATOR'] = 'platform'
    env['JAX_PLATFORMS'] = 'cuda'
    env['WANDB_MODE'] = 'disabled'
    env['HF_HUB_DISABLE_TELEMETRY'] = '1'
    env['OMP_NUM_THREADS'] = '2'
    return env

def pin_repository(folder, url, commit):
    if not folder.exists():
        run(['git', 'clone', '--no-checkout', url, folder])
        run(['git', 'checkout', '--detach', commit], cwd=folder)
    actual = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=folder, text=True).strip()
    changes = subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=no'], cwd=folder, text=True).strip()
    if actual != commit or changes:
        raise RuntimeError(f'{folder}: требуется неизменённый commit {commit}; существующие файлы не перезаписываются.')

def check_repositories(root):
    for name, commit in (('openpi', OPENPI_COMMIT), ('insight', INSIGHT_COMMIT)):
        folder = root / 'vla_insight' / name
        if not folder.exists():
            raise RuntimeError('Сначала: python start_insight.py prepare')
        pin_repository(folder, '', commit)

def freeze_act(root):
    paths = {'models': ('selection_models_path.txt', 'models.json'), 'recovery': ('selection_recovery_evaluation_path.txt', 'summary.json'), 'clean': ('selection_clean_evaluation_path.txt', 'summary.json')}
    record = {'scope': 'ACT selection pilot; ensemble U, reference F, Random', 'is_INSIGHT': False, 'is_language_conditioned_VLA': False, 'sources': {}}
    for name, (pointer, filename) in paths.items():
        folder = Path((root / 'pilot_runs' / pointer).read_text().strip())
        if not folder.is_absolute():
            folder = root / folder
        source = folder / filename
        value = read_json(source)
        if value.get('status') not in (None, 'complete'):
            raise RuntimeError(f'Оценка ещё не завершена: {source}')
        record['sources'][name] = {'path': str(source.resolve()), 'sha256': digest(source), 'record': value}
    output = new_run(root, 'act_record')
    write_json(output / 'act_results.json', record)
    print('ACT RESULTS SAVED:', output / 'act_results.json')

def cpu_author_smoke(root):
    import numpy as np
    import torch
    sys.path.insert(0, str(root / 'vla_insight' / 'insight'))
    from insight.models.single_transformer import SingleStepTransformer
    from insight.strong_trigger import StrongHelpTrigger
    torch.set_num_threads(2)
    torch.manual_seed(0)
    model = SingleStepTransformer().cpu().eval()
    with torch.no_grad():
        result = model(torch.zeros(1, 8, 4), torch.zeros(1, 8, dtype=torch.bool))
    assert result.shape == (1,) and torch.isfinite(result).all()
    trigger = StrongHelpTrigger.__new__(StrongHelpTrigger)
    trigger.trim_head, trigger.trim_tail = (3, 2)
    source_items_3 = enumerate(StrongHelpTrigger.FEATURE_KEYS)
    raw = {}
    for i, k in source_items_3:
        raw[k] = np.arange(10, dtype=np.float32) + 100 * i
    features = trigger.features_from_policy_output(raw)
    np.testing.assert_array_equal(features[:, 0], np.arange(3, 8))
    np.testing.assert_array_equal(features[:, 3], np.arange(303, 308))
    record = {'status': 'CPU author architecture / feature-contract test passed', 'parameters': sum((p.numel() for p in model.parameters())), 'trained_detector': False, 'author_commit': INSIGHT_COMMIT, 'feature_order': FEATURE_ORDER, 'trim_head': 3, 'trim_tail': 2}
    write_json(root / 'vla_insight' / 'cpu_author_check.json', record)
    print('INSIGHT AUTHOR CODE CPU CHECK OK')

def prepare(root):
    base = root / 'vla_insight'
    base.mkdir(exist_ok=True)
    pin_repository(base / 'insight', 'https://github.com/ulaskarli/insight-vla-help-triggers.git', INSIGHT_COMMIT)
    pin_repository(base / 'openpi', 'https://github.com/Physical-Intelligence/openpi.git', OPENPI_COMMIT)
    cpu_author_smoke(root)
    releases = public_json('https://api.github.com/repos/ulaskarli/insight-vla-help-triggers/releases')
    write_json(base / 'public_releases.json', releases)
    source_items_4 = releases
    assets = []
    for release in source_items_4:
        for asset in release.get('assets', []):
            assets.append(asset['name'])
    print('INSIGHT release assets:', assets)
    print('GPU:', gpu_info())
    source_items_5 = memory_info().items()
    items_5 = {}
    for key, value in source_items_5:
        items_5[key] = round(value / GIB, 2)
    print('RAM:', items_5)
    env = isolated_env(root)
    tools_python = base / 'tools' / 'bin' / 'python'
    if not tools_python.exists():
        run([sys.executable, '-m', 'venv', base / 'tools'])
    if not (base / 'tools' / 'bin' / 'uv').exists():
        run([tools_python, '-m', 'pip', 'install', 'uv'])
    uv = base / 'tools' / 'bin' / 'uv'
    run(['git', 'submodule', 'update', '--init', '--recursive'], cwd=base / 'openpi', env=env)
    run([uv, 'sync', '--python', '3.11', '--no-dev'], cwd=base / 'openpi', env=env)
    jax_check(root)
    print('\nPREPARE COMPLETE')

def jax_check(root):
    check_repositories(root)
    python = root / 'vla_insight' / 'openpi' / '.venv' / 'bin' / 'python'
    if not python.exists():
        raise RuntimeError('Не создана отдельная среда openpi. Сначала prepare.')
    run([python, Path(__file__).resolve(), '_jax-worker'], cwd=root, env=isolated_env(root))

def jax_worker(root):
    import jax
    import jax.numpy as jnp
    import numpy as np
    devices = jax.devices()
    if not devices or any((device.platform != 'gpu' for device in devices)):
        raise RuntimeError(f'JAX не видит CUDA GPU: {devices}; CPU fallback запрещён.')
    a = jnp.ones((512, 512), dtype=jnp.bfloat16)
    value = float(jnp.sum(a @ a, dtype=jnp.float32).block_until_ready())
    if not np.isfinite(value) or value != 512.0 ** 3:
        raise RuntimeError('JAX CUDA matmul не прошёл.')
    source_items_6 = devices
    items_6 = []
    for d in source_items_6:
        items_6.append(str(d))
    record = {'jax': jax.__version__, 'devices': items_6, 'cuda_bfloat16_matmul': 'OK'}
    write_json(root / 'vla_insight' / 'jax_check.json', record)
    print('JAX CUDA OK:', record, flush=True)

def saved_observation(root, candidate_id=None):
    import numpy as np
    pointer = root / 'pilot_runs' / 'selection_pool_path.txt'
    pool = Path(pointer.read_text().strip())
    if not pool.is_absolute():
        pool = root / pool
    protocol = read_json(pool / 'protocol.json')
    if protocol.get('status') != 'complete':
        raise RuntimeError('Нужен завершённый ACT pool для одного integration probe.')
    candidates = protocol['cases']
    if candidate_id is not None:
        source_items_7 = candidates
        candidates = []
        for case in source_items_7:
            if case['candidate_id'] == candidate_id:
                candidates.append(case)
    if not candidates:
        raise RuntimeError('Не найден candidate для проверки интерфейса.')
    case = candidates[0]
    snapshot = pool / case['candidate_id'] / 'start.npz'
    if digest(snapshot) != case['snapshot_sha256']:
        raise RuntimeError('Изменился исходный snapshot.')
    with np.load(snapshot, allow_pickle=False) as data:
        pos = data['observation/robot_state/eef/pos'][0].astype(np.float32)
        quat = data['observation/robot_state/eef/quat'][0].astype(np.float32)
        if not np.isclose(np.linalg.norm(quat), 1, atol=0.0001):
            raise RuntimeError('EEF quaternion должен быть единичным, в порядке xyzw.')
        w = np.clip(quat[3], -1.0, 1.0)
        denominator = np.sqrt(max(0.0, 1.0 - w * w))
        axis_angle = np.zeros(3, dtype=np.float32)
        if denominator > 1e-10:
            axis_angle = (quat[:3] / denominator * (2.0 * np.arccos(w))).astype(np.float32)
        grip = data['observation/robot_state/gripper/qpos'][0].astype(np.float32)
        state = np.concatenate((pos, axis_angle, grip))
        image = np.ascontiguousarray(data['observation/pixels/image'][0, ::-1, ::-1])
        wrist = np.ascontiguousarray(data['observation/pixels/image2'][0, ::-1, ::-1])
    if state.shape != (8,) or image.ndim != 3 or image.shape[-1] != 3:
        raise RuntimeError('Некорректные observation shapes.')
    observation = {'observation/state': state, 'observation/image': image, 'observation/wrist_image': wrist, 'prompt': protocol['instruction']}
    source = {'candidate_id': case['candidate_id'], 'snapshot': str(snapshot), 'snapshot_sha256': digest(snapshot), 'source_policy': 'ACT', 'use': 'one input for VLA integration probe only, not VLA training/evaluation'}
    return (observation, source)

def sample_with_features(self, rng, observation, *, max_decoding_steps=256):
    import jax
    import jax.numpy as jnp
    from openpi.models import model as _model
    from openpi.models.pi0_fast import PALIGEMMA_EOS_TOKEN, left_to_right_align, make_attn_mask
    observation = _model.preprocess_observation(None, observation, train=False, image_keys=list(observation.images.keys()))
    prefix, prefix_mask, ar_mask = self.embed_inputs(observation)
    attn = make_attn_mask(prefix_mask, ar_mask)
    prefix, prefix_mask, attn = left_to_right_align(prefix, prefix_mask, attn)
    size = prefix.shape[1]
    length = jnp.sum(prefix_mask, axis=-1)
    start = size - length
    attn = jnp.pad(attn, ((0, 0), (0, 0), (0, max_decoding_steps)))
    positions = jnp.cumsum(prefix_mask, axis=-1) - 1
    hidden, cache, _ = self.PaliGemma.llm(embedded_prefix=prefix, mask=attn, positions=positions, decode=True, return_prelogits=True)
    logits, _ = self.PaliGemma.llm(pre_logits=hidden[:, -1:])
    batch = logits.shape[0]
    tokens = jnp.zeros((batch, max_decoding_steps), dtype=jnp.float32)
    tracks = jnp.zeros((batch, max_decoding_steps, 4), dtype=jnp.float32)

    def step(carry):
        last_logits, tokens, tracks, cache, _, index = carry
        z = last_logits[:, 0].astype(jnp.float32)
        token = jnp.argmax(last_logits, axis=-1).astype(jnp.int32)
        logp = jax.nn.log_softmax(z, axis=-1)
        entropy = -jnp.sum(jnp.exp(logp) * logp, axis=-1)
        chosen_logp = jnp.take_along_axis(logp, token, axis=-1)[:, 0]
        top_logits, _ = jax.lax.top_k(z, 30)
        alpha = jax.nn.relu(top_logits) + 1e-06
        total = jnp.sum(alpha, axis=-1)
        au = -jnp.sum(alpha / total[:, None] * (jax.scipy.special.digamma(alpha + 1) - jax.scipy.special.digamma(total[:, None] + 1)), axis=-1)
        eu = 30.0 / (total + 30.0)
        features = jnp.stack((au, eu, entropy, chosen_logp), axis=-1)
        tokens = tokens.at[:, index].set(token[:, 0].astype(jnp.float32))
        tracks = tracks.at[:, index].set(features)
        done = jnp.all(jnp.any(token == PALIGEMMA_EOS_TOKEN, axis=-1))
        embedding = self.PaliGemma.llm(token, embed_only=True)
        positions = length[:, None] + index + 1
        mask = jnp.logical_and(jnp.arange(size + max_decoding_steps)[None, None] >= start[:, None, None], jnp.arange(size + max_decoding_steps)[None, None] < jnp.broadcast_to(size + index + 1, (start.shape[0], 1, 1)))
        next_logits, cache, _ = self.PaliGemma.llm(embedded_prefix=embedding, mask=mask, positions=positions, decode=True, kv_cache=cache)
        return (next_logits, tokens, tracks, cache, done, index + 1)

    def condition(carry):
        return ~carry[4] & (carry[5] < max_decoding_steps)
    _, tokens, tracks, _, done, count = jax.lax.while_loop(condition, step, (logits, tokens, tracks, cache, False, 0))
    return (tokens, tracks, count, done)

class ZScoreFactory:

    def __init__(self, original):
        self.original = original

    def create(self, *args, **kwargs):
        return dataclasses.replace(self.original.create(*args, **kwargs), use_quantile_norm=False)

def memory_note(output, stage):
    import resource
    status = {}
    path = Path('/proc/self/status')
    if path.exists():
        for line in path.read_text().splitlines():
            name, value = line.split(':', 1)
            if name in ('VmRSS', 'VmHWM'):
                status[name] = int(value.split()[0]) * 1024
    record = {'stage': stage, 'pid': os.getpid(), 'timestamp': time.time(), 'rss_bytes': status.get('VmRSS'), 'rss_peak_bytes': int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024}
    cgroup = Path('/proc/self/cgroup')
    if cgroup.exists():
        for line in cgroup.read_text().splitlines():
            if line.startswith('0::'):
                folder = Path('/sys/fs/cgroup') / line[3:].lstrip('/')
                record['cgroup_path'] = str(folder)
                for name in ('memory.current', 'memory.peak', 'memory.max', 'memory.events'):
                    try:
                        record[name] = (folder / name).read_text().strip()
                    except OSError:
                        pass
    write_json(Path(output) / 'memory_stage.json', record)
    rss = record['rss_bytes'] or 0
    print(f"STAGE: {stage}; RSS={rss / GIB:.2f} GiB; RSS peak={record['rss_peak_bytes'] / GIB:.2f} GiB", flush=True)

def restore_params_bounded(params_path, *, restore_type=None, dtype=None, sharding=None, concurrent_gb=RESTORE_CONCURRENT_GB, report=None):
    import jax
    import orbax.checkpoint as ocp
    from flax import traverse_util
    from orbax.checkpoint._src.serialization import tensorstore_utils
    params_path = Path(params_path).resolve()
    restore_type = jax.Array if restore_type is None else restore_type
    if restore_type is jax.Array and sharding is None:
        mesh = jax.sharding.Mesh(jax.devices(), ('x',))
        sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
    original_context = tensorstore_utils.get_ts_context

    def limited_context(*args, **kwargs):
        kwargs.setdefault('file_io_concurrency_limit', 2)
        kwargs.setdefault('data_copy_concurrency_limit', 2)
        return original_context(*args, **kwargs)
    tensorstore_utils.get_ts_context = limited_context
    try:
        handler = ocp.PyTreeCheckpointHandler(restore_concurrent_gb=concurrent_gb)
        with ocp.Checkpointer(handler) as checkpointer:
            metadata = checkpointer.metadata(params_path)
            item = {'params': metadata['params']}
            if report:
                report('restore_metadata_ready')
            print(f'BOUNDED RESTORE: concurrent_reads={concurrent_gb} GB; file_IO_threads=2; data_copy_threads=2; target_dtype=' + str(dtype), flush=True)

            def array_restore_arguments(_):
                return ocp.ArrayRestoreArgs(sharding=sharding, restore_type=restore_type, dtype=dtype)
            params = checkpointer.restore(params_path, ocp.args.PyTreeRestore(item=item, restore_args=jax.tree.map(array_restore_arguments, item)))['params']

        def wait_for_array(value):
            if isinstance(value, jax.Array):
                return value.block_until_ready()
            else:
                return value
        jax.tree.map(wait_for_array, params)
    finally:
        tensorstore_utils.get_ts_context = original_context
    flat = traverse_util.flatten_dict(params)
    if all((key[-1] == 'value' for key in flat)):
        source_items_8 = flat.items()
        flat = {}
        for key, value in source_items_8:
            flat[key[:-1]] = value
    if report:
        report('restore_weights_ready')
    return traverse_util.unflatten_dict(flat)

@contextmanager
def bounded_openpi_restore(model_module, output):
    original = model_module.restore_params

    def bounded(*args, **kwargs):

        def record_memory(stage):
            return memory_note(output, stage)
        return restore_params_bounded(*args, **kwargs, report=record_memory)
    model_module.restore_params = bounded
    try:
        yield
    finally:
        model_module.restore_params = original

def guarded_worker(root, output, args, *, worker_command='_features-worker'):
    if worker_command not in ('_features-worker', '_stock-worker'):
        raise ValueError(f'Неизвестный GPU worker: {worker_command}')
    notes = output if worker_command == '_features-worker' else output / 'stock_check'
    notes.mkdir(parents=True, exist_ok=True)
    python = root / 'vla_insight' / 'openpi' / '.venv' / 'bin' / 'python'
    argv = [str(python), str(Path(__file__).resolve()), worker_command, '--output', str(output)]
    if worker_command == '_features-worker' and args.detector_checkpoint:
        argv += ['--detector-checkpoint', str(Path(args.detector_checkpoint).expanduser().resolve())]
    if worker_command == '_features-worker' and args.verify_tokens:
        argv += ['--verify-tokens']
    memory = memory_info()
    if not args.no_memory_guard:
        if not shutil.which('systemd-run'):
            raise RuntimeError('Не найден systemd-run для ограничения RAM. Запуск остановлен до загрузки модели.')
        subprocess.run(['systemctl', '--user', 'show-environment'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        cap = min(memory['MemAvailable'] - GIB, memory['MemTotal'] - 4 * GIB)
        if cap < MIN_WORKER_RAM:
            raise RuntimeError(f'RAM CHECK STOP: лимит worker {cap / GIB:.2f} GiB, нужно не менее {MIN_WORKER_RAM / GIB:.2f} GiB с сохранением запаса RAM для системы. Проверь python run_vla_libero.py --memory-check.')
        print(f'Worker RAM limit: {cap / GIB:.2f} GiB; swap=0.', flush=True)
        argv = ['systemd-run', '--user', '--scope', '--quiet', '-p', f'MemoryMax={cap}', '-p', 'MemorySwapMax=0', '--', *argv]
    log = notes / 'worker.log'
    print('Log:', log, flush=True)
    with log.open('w', encoding='utf-8') as file:
        process = subprocess.Popen(argv, cwd=root, env=isolated_env(root), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, start_new_session=True)
        try:
            for line in process.stdout:
                print(line, end='', flush=True)
                file.write(line)
                file.flush()
            code = process.wait()
        except BaseException:
            if process.poll() is None:
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    process.wait(timeout=10)
            raise
        finally:
            process.stdout.close()
    if code:
        note = ''
        stage = notes / 'memory_stage.json'
        if stage.exists():
            info = read_json(stage)
            note = f" Последний этап: {info['stage']}; замеры: {stage}."
        write_json(notes / 'worker_exit.json', {'returncode': code, 'last_stage': read_json(stage) if stage.exists() else None})
        label = 'Проба VLA' if worker_command == '_features-worker' else 'Stock token check'
        raise RuntimeError(f'{label} остановлена (exit={code}).{note} Лог: {log}. Это ещё не результат INSIGHT selection.')

def compare_probe_tokens(output):
    import numpy as np
    with np.load(output / 'token_features.npz', allow_pickle=False) as saved:
        instrumented = saved['padded_token_ids'].copy()
    with np.load(output / 'stock_check' / 'tokens.npz', allow_pickle=False) as saved:
        stock = saved['padded_token_ids'].copy()
    comparison = {'status': 'matched' if np.array_equal(instrumented, stock) else 'mismatch', 'instrumented_shape': list(instrumented.shape), 'stock_shape': list(stock.shape), 'includes_eos_and_padding': True}
    if comparison['status'] == 'mismatch' and instrumented.shape == stock.shape:
        index = tuple(np.argwhere(instrumented != stock)[0])
        source_items_9 = index
        items_9 = []
        for i in source_items_9:
            items_9.append(int(i))
        comparison.update(first_difference_index=items_9, instrumented_token=int(instrumented[index]), stock_token=int(stock[index]))
    return comparison

def verify_probe_tokens(root, output, args):
    result = read_json(output / 'result.json')
    if result.get('status') != 'VLA token-feature inference passed':
        raise RuntimeError('Перед stock check требуется успешная проба с сохранёнными признаками.')
    metadata = {'method': 'separate sequential GPU processes', 'rng_seed': 0, 'temperature': 0.0, 'max_decoding_steps': 256, 'stock_log': str(output / 'stock_check' / 'worker.log')}
    print('\nSTOCK TOKEN CHECK', flush=True)
    try:
        guarded_worker(root, output, args, worker_command='_stock-worker')
        comparison = compare_probe_tokens(output)
    except (Exception, KeyboardInterrupt) as error:
        result['token_check'] = 'not verified: stock check did not complete'
        result['token_verification'] = {**metadata, 'status': 'interrupted' if isinstance(error, KeyboardInterrupt) else 'failed', 'error': str(error) or type(error).__name__}
        write_json(output / 'result.json', result)
        if isinstance(error, KeyboardInterrupt):
            raise
        raise RuntimeError(f"Token check не завершён. Признаки сохранены: {output / 'token_features.npz'}. Причина: {error}") from error
    result['token_verification'] = {**metadata, **comparison}
    result['token_check'] = 'exact match to stock openpi greedy tokens' if comparison['status'] == 'matched' else 'token mismatch to stock openpi'
    write_json(output / 'result.json', result)
    if comparison['status'] != 'matched':
        raise RuntimeError(f"Token ids отличаются. Оба массива сохранены; сравнение: {output / 'result.json'}.")
    print('TOKEN CHECK OK: stock OpenPI tokens match', flush=True)

def features(root, args):
    check_repositories(root)
    jax_check(root)
    observation, source = saved_observation(root, args.candidate)
    import numpy as np
    output = new_run(root, 'token_probe')
    source_items_10 = observation.items()
    items_10 = {}
    for k, v in source_items_10:
        if k != 'prompt':
            items_10[k] = v
    np.savez_compressed(output / 'input.npz', **items_10)
    write_json(output / 'input.json', {'prompt': observation['prompt'], 'source': source})
    checkpoint, files = ensure_checkpoint()
    size = sum((item['size'] for item in files))
    write_json(output / 'run_info.json', {'checkpoint': CHECKPOINT, 'openpi_commit': OPENPI_COMMIT, 'insight_commit': INSIGHT_COMMIT, 'purpose': 'One offline VLA inference and true INSIGHT feature extraction', 'normalization': 'zscore, released LIBERO FAST checkpoint compatibility', 'feature_order': FEATURE_ORDER, 'top_k': 30, 'evidence': 'relu(logits)+1e-6', 'trim_head': 3, 'trim_tail': 2, 'source': source, 'trained_detector_supplied': bool(args.detector_checkpoint), 'checkpoint_download_bytes': size, 'memory_before': memory_info(), 'gpu_before': gpu_info(), 'checkpoint_local_path': str(checkpoint), 'checkpoint_files': files, 'checkpoint_download': 'verified curl HTTPS; no gcsfs', 'restore_concurrent_gb': RESTORE_CONCURRENT_GB, 'restore_IO_threads': 2, 'token_verification_requested': args.verify_tokens, 'token_verification_method': 'separate sequential GPU processes' if args.verify_tokens else None, 'rng_seed': 0, 'max_decoding_steps': 256, 'temperature': 0.0})
    guarded_worker(root, output, args)
    if args.verify_tokens:
        verify_probe_tokens(root, output, args)
    print('\nResult:', output / 'result.json')
    print('TOKEN FEATURES SAVED:', output / 'token_features.npz')

def load_probe_policy_and_observation(output, notes):
    import jax
    import jax.numpy as jnp
    import numpy as np
    from openpi.models import model as _model
    from openpi.policies import policy_config
    from openpi.training import config
    print('LOADING OFFICIAL pi0-FAST LIBERO (BF16, JAX).', flush=True)
    memory_note(notes, 'imports_ready')
    record = read_json(output / 'run_info.json')
    checkpoint = Path(record['checkpoint_local_path'])
    if not checkpoint.is_dir():
        raise RuntimeError('Не найден предварительно проверенный checkpoint.')
    cfg = config.get_config('pi0_fast_libero')
    cfg = dataclasses.replace(cfg, data=ZScoreFactory(cfg.data))
    data_cfg = cfg.data.create(cfg.assets_dirs, cfg.model)
    assert data_cfg.use_quantile_norm is False
    with bounded_openpi_restore(_model, notes):
        policy = policy_config.create_trained_policy(cfg, checkpoint)
    memory_note(notes, 'policy_ready')
    with np.load(output / 'input.npz', allow_pickle=False) as saved:
        source_items_11 = saved.files
        obs = {}
        for key in source_items_11:
            obs[key] = saved[key].copy()
    obs['prompt'] = read_json(output / 'input.json')['prompt']
    inputs = policy._input_transform(dict(obs))

    def add_batch_dimension(value):
        return jnp.asarray(value)[None]
    inputs = jax.tree.map(add_batch_dimension, inputs)
    return (policy, inputs, _model.Observation.from_dict(inputs))

def features_worker(root, args):
    output = Path(args.output)
    try:
        import jax
        import numpy as np
        from openpi.shared import nnx_utils
        os.environ['PYTHONUNBUFFERED'] = '1'
        start_time = time.monotonic()
        policy, inputs, observation = load_probe_policy_and_observation(output, output)
        print('VLA WEIGHTS LOADED', flush=True)
        bound = MethodType(sample_with_features, policy._model)
        sampler = nnx_utils.module_jit(bound)
        rng = jax.random.key(0)
        memory_note(output, 'inference_compile_begin')
        tokens, tracks, count, done = sampler(rng, observation)
        tokens, tracks, count, done = jax.device_get((tokens, tracks, count, done))
        memory_note(output, 'inference_complete')
        count = int(count)
        if not bool(done) or count <= 5:
            raise RuntimeError(f'FAST sequence не завершена EOS или слишком короткая: n={count}, EOS={done}.')
        raw = np.asarray(tracks[0, :count], dtype=np.float32)
        if raw.shape != (count, 4) or not np.all(np.isfinite(raw)):
            raise RuntimeError('Некорректные uncertainty features.')
        if np.any(raw[:, 2] < -1e-05) or np.any(raw[:, 3] > 1e-05) or np.any(raw[:, 1] > 1.00001):
            raise RuntimeError('Не выполнены базовые ограничения entropy/log p/EU.')
        from run_vla_libero import checked_output_transform
        decoded = checked_output_transform(policy, np.asarray(inputs['state'][0]), tokens[0])
        actions = np.asarray(decoded['actions'])
        if actions.shape != (10, 7) or not np.all(np.isfinite(actions)):
            raise RuntimeError(f'Некорректные VLA actions: {actions.shape}.')
        sys.path.insert(0, str(root / 'vla_insight' / 'insight'))
        from insight.strong_trigger import StrongHelpTrigger
        trigger = StrongHelpTrigger.__new__(StrongHelpTrigger)
        trigger.trim_head, trigger.trim_tail = (3, 2)
        policy_output = {'au': raw[:, 0], 'eu': raw[:, 1], 'entropy': raw[:, 2], 'perplexity': raw[:, 3]}
        trimmed = trigger.features_from_policy_output(policy_output)
        token_check = 'pending separate stock worker' if args.verify_tokens else 'not requested'
        np.savez_compressed(output / 'token_features.npz', token_ids=tokens[0, :count].astype(np.int32), padded_token_ids=tokens.astype(np.int32), raw_features=raw, features=trimmed, actions=actions)
        result = {'status': 'VLA token-feature inference passed', 'checkpoint': CHECKPOINT, 'generated_tokens': count, 'classifier_tokens': len(trimmed), 'feature_order': FEATURE_ORDER, 'actions_shape': list(actions.shape), 'token_check': token_check, 'help_score': None, 'scope': 'integration probe; no closed-loop evaluation', 'elapsed_seconds_including_load': time.monotonic() - start_time, 'gpu_after': gpu_info(), 'host_memory_after': memory_info()}
        stats = jax.devices()[0].memory_stats()
        if stats:
            source_items_12 = stats.items()
            items_12 = {}
            for k, v in source_items_12:
                if isinstance(v, (int, float)):
                    items_12[k] = int(v)
            result['jax_memory_stats_bytes'] = items_12
        if args.detector_checkpoint:
            trained = StrongHelpTrigger([args.detector_checkpoint], device='cpu')
            decision = trained.predict_features(trimmed)
            result['help_score'] = decision.probability
            result['help_logit'] = decision.logit
            result['detector_sha256'] = digest(args.detector_checkpoint)
            result['detector_note'] = 'Supplied checkpoint; its training/split provenance must be verified separately.'
        else:
            result['detector_note'] = 'No trained Strong weights supplied. Need separate labeled VLA rollouts and detector training.'
        write_json(output / 'result.json', result)
        print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)
    except Exception as error:
        write_json(output / 'result.json', {'status': 'error', 'error': str(error), 'traceback': traceback.format_exc()})
        raise

def stock_worker(root, args):
    output = Path(args.output)
    notes = output / 'stock_check'
    notes.mkdir(parents=True, exist_ok=True)
    try:
        import jax
        import numpy as np
        start_time = time.monotonic()
        policy, _, observation = load_probe_policy_and_observation(output, notes)
        print('STOCK CHECK: OpenPI greedy sampler', flush=True)
        memory_note(notes, 'stock_inference_compile_begin')
        tokens = np.asarray(jax.device_get(policy._sample_actions(jax.random.key(0), observation)))
        memory_note(notes, 'stock_inference_complete')
        if tokens.shape != (1, 256) or not np.all(np.isfinite(tokens)) or np.any(tokens != np.floor(tokens)):
            raise RuntimeError(f'Некорректный массив stock tokens: {tokens.shape}.')
        np.savez_compressed(notes / 'tokens.npz', padded_token_ids=tokens.astype(np.int32))
        result = {'status': 'stock inference complete; comparison performed by parent', 'tokens_shape': list(tokens.shape), 'rng_seed': 0, 'temperature': 0.0, 'max_decoding_steps': 256, 'elapsed_seconds_including_load': time.monotonic() - start_time, 'gpu_after': gpu_info(), 'host_memory_after': memory_info()}
        write_json(notes / 'result.json', result)
        print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)
    except Exception as error:
        write_json(notes / 'result.json', {'status': 'error', 'error': str(error), 'traceback': traceback.format_exc()})
        raise

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('command', choices=['freeze-act', 'prepare', 'download', 'jax-check', 'features', '_jax-worker', '_features-worker', '_stock-worker'])
    parser.add_argument('--candidate', help='Optional ACT candidate ID for one interface probe')
    parser.add_argument('--detector-checkpoint', help='Optional actual trained INSIGHT Strong .pt file')
    parser.add_argument('--verify-tokens', action='store_true', help='Compare to original OpenPI in a second sequential GPU process; preserve features if it fails')
    parser.add_argument('--no-memory-guard', action='store_true', help='Explicitly disable the RAM process limit')
    parser.add_argument('--output', help=argparse.SUPPRESS)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    if args.command == 'freeze-act':
        freeze_act(root)
    elif args.command == 'prepare':
        prepare(root)
    elif args.command == 'download':
        ensure_checkpoint()
    elif args.command == 'jax-check':
        jax_check(root)
    elif args.command == 'features':
        features(root, args)
    elif args.command == '_jax-worker':
        jax_worker(root)
    elif args.command == '_features-worker':
        features_worker(root, args)
    else:
        stock_worker(root, args)
if __name__ == '__main__':
    try:
        main()
    except (Exception, KeyboardInterrupt) as error:
        print(f'\nSTOP: {error}', file=sys.stderr, flush=True)
        sys.exit(1)
