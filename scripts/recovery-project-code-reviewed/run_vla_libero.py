#!/usr/bin/env python3
from __future__ import annotations
import argparse
import dataclasses
from datetime import datetime, timezone
import importlib.metadata
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
import numpy as np
import start_insight as bridge
SUITE_HORIZONS = {'libero_spatial': 280, 'libero_object': 280, 'libero_goal': 300, 'libero_10': 520, 'libero_90': 400}
CONTROL_HZ = 20
RUN_SCOPE = 'VLA development rollout; no selector comparison or recovery training'
DECODER_PROTOCOL = 3
MIN_WORKER_RAM = 11 * bridge.GIB
AVAILABLE_RAM_RESERVE = bridge.GIB
TOTAL_RAM_RESERVE = 4 * bridge.GIB

def worker_memory_plan(memory=None):
    memory = bridge.memory_info() if memory is None else memory
    require('MemTotal' in memory and 'MemAvailable' in memory, 'Не удалось прочитать RAM из /proc/meminfo.')
    total, available = (int(memory['MemTotal']), int(memory['MemAvailable']))
    require(total > 0 and 0 <= available <= total, 'Некорректные показатели RAM.')
    total_ceiling = max(0, total - TOTAL_RAM_RESERVE)
    cap = max(0, min(available - AVAILABLE_RAM_RESERVE, total_ceiling))
    return {'total': total, 'available': available, 'cap': cap, 'ready': cap >= MIN_WORKER_RAM, 'required_available': MIN_WORKER_RAM + AVAILABLE_RAM_RESERVE, 'additional_available': max(0, MIN_WORKER_RAM + AVAILABLE_RAM_RESERVE - available), 'total_shortfall': max(0, MIN_WORKER_RAM - total_ceiling)}

def print_memory_plan(plan):
    print(f"RAM: total={plan['total'] / bridge.GIB:.2f} GiB; available={plan['available'] / bridge.GIB:.2f} GiB.", flush=True)
    print(f"Worker RAM limit={plan['cap'] / bridge.GIB:.2f} GiB; minimum for this probe={MIN_WORKER_RAM / bridge.GIB:.2f} GiB (observed restore peak=10.12 GiB).", flush=True)
    if plan['ready']:
        print('RAM CHECK OK', flush=True)

def memory_failure_message(plan):
    if plan['total_shortfall']:
        return f"RAM CHECK STOP: при сохранении запаса RAM для системы максимум процесса {max(0, plan['total'] - TOTAL_RAM_RESERVE) / bridge.GIB:.2f} GiB, ниже порога {MIN_WORKER_RAM / bridge.GIB:.2f} GiB. Закрытия приложений недостаточно для этого ограничителя."
    return f"RAM CHECK STOP: освободите ещё не менее {plan['additional_available'] / bridge.GIB:.2f} GiB RAM (нужно MemAvailable >= {plan['required_available'] / bridge.GIB:.2f} GiB). Закройте VS Code, лишние вкладки браузера и видеоплееры, затем повторите --memory-check. VLA worker не запущен."

class InferenceOutputError(RuntimeError):

    def __init__(self, code, message, *, arrays=None, info=None):
        super().__init__(message)
        self.code = code
        self.arrays = {} if arrays is None else arrays
        self.info = {} if info is None else info

def strict_fast_extract(extractor, tokens):
    from scipy.fft import idct
    tokenizer = extractor.tokenizer
    horizon, dim = (extractor.action_horizon, extractor.action_dim)
    text = tokenizer._paligemma_tokenizer.decode(np.asarray(tokens, dtype=np.int32).tolist())
    if 'Action: ' not in text:
        raise InferenceOutputError('missing_action_marker', 'FAST output не содержит Action: payload.')
    if '|' not in text.split('Action: ', 1)[1]:
        raise InferenceOutputError('missing_action_separator', 'FAST output не содержит завершающий Action separator.')
    payload = text.split('Action: ')[1].split('|')[0].strip()
    raw_tokens = np.asarray(tokenizer._paligemma_tokenizer.encode(payload))
    action_tokens = tokenizer._act_tokens_to_paligemma_tokens(raw_tokens)
    processor = tokenizer._fast_tokenizer
    try:
        decoded = processor.bpe_tokenizer.decode(action_tokens.tolist())
        coefficients = np.asarray(list(map(ord, decoded))) + processor.min_token
        expected = horizon * dim
        if coefficients.size != expected:
            raise InferenceOutputError('fast_decode', f'FAST: {coefficients.size} DCT coefficients; expected {expected} ({horizon} x {dim}).')
        require(np.isfinite(processor.scale) and processor.scale > 0, 'Некорректный FAST scale.')
        return idct(coefficients.reshape(horizon, dim) / processor.scale, axis=0, norm='ortho')
    except InferenceOutputError:
        raise
    except (ValueError, TypeError, OverflowError, AssertionError) as error:
        raise InferenceOutputError('fast_decode', f'FAST action decoding: {type(error).__name__}: {error}') from error

def checked_output_transform(policy, state, tokens):
    transforms = getattr(policy._output_transform, 'transforms', None)
    require(transforms is not None, 'Не найден pinned OpenPI CompositeTransform.')
    require(sum((type(t).__name__ == 'ExtractFASTActions' for t in transforms)) == 1, 'Ожидается один ExtractFASTActions в output pipeline.')
    data = {'state': state, 'actions': tokens}
    extracted = False
    for transform in transforms:
        if type(transform).__name__ == 'ExtractFASTActions':
            data = {**data, 'actions': strict_fast_extract(transform, data['actions'])}
            extracted = True
        else:
            if type(transform).__name__ == 'Unnormalize':
                require(extracted, 'FAST decoding должен предшествовать unnormalization.')
            data = transform(data)
    return data

def require(condition, message):
    if not condition:
        raise RuntimeError(message)

def json_atomic(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    temporary.replace(path)

def versions():
    result = {'python': sys.version.split()[0]}
    for package in ('lerobot', 'hf-libero', 'jax', 'orbax-checkpoint', 'torch', 'av'):
        try:
            result[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            result[package] = None
    return result

def flatten_observation(observation, prefix='observation'):
    result = {}
    for key, value in observation.items():
        path = f'{prefix}/{key}'
        if isinstance(value, dict):
            result.update(flatten_observation(value, path))
        else:
            array = np.asarray(value)
            require(array.dtype != object, f'Object array запрещён в observation: {path}')
            result[path] = array.copy()
    return result

def openpi_observation(observation, instruction):
    eef = observation['robot_state']['eef']
    pos = np.asarray(eef['pos'][0], dtype=np.float32)
    quat = np.asarray(eef['quat'][0], dtype=np.float32)
    grip = np.asarray(observation['robot_state']['gripper']['qpos'][0], dtype=np.float32)
    require(pos.shape == (3,) and quat.shape == (4,) and (grip.shape == (2,)), 'Некорректные размеры proprioception.')
    require(np.isclose(np.linalg.norm(quat), 1, atol=0.0001), 'EEF quaternion должен быть единичным xyzw.')
    w = np.clip(quat[3], -1, 1)
    denominator = np.sqrt(max(0.0, 1.0 - w * w))
    angle = np.zeros(3, dtype=np.float32)
    if denominator > 1e-10:
        angle = (quat[:3] / denominator * (2.0 * np.arccos(w))).astype(np.float32)
    state = np.concatenate((pos, angle, grip))
    require(np.all(np.isfinite(state)), 'Неконечный state.')
    images = []
    for key in ('image', 'image2'):
        image = np.asarray(observation['pixels'][key][0])
        require(image.shape == (256, 256, 3) and image.dtype == np.uint8, f'Ожидается RGB uint8 (256,256,3): {key}, {image.shape}, {image.dtype}.')
        images.append(np.ascontiguousarray(image[::-1, ::-1]))
    return {'observation/state': state, 'observation/image': images[0], 'observation/wrist_image': images[1], 'prompt': str(instruction)}

def wrapper_nodes(vec):
    queue, visited = ([vec.envs[0]], set())
    while queue:
        obj = queue.pop(0)
        if id(obj) in visited:
            continue
        visited.add(id(obj))
        yield obj
        for name in ('env', '_env', 'unwrapped'):
            child = getattr(obj, name, None)
            if child is not None:
                queue.append(child)

def libero_wrapper(vec):
    for obj in wrapper_nodes(vec):
        if 'init_state_id' in vars(obj) and '_init_states' in vars(obj):
            return obj
    raise RuntimeError('Не найден LiberoEnv с фиксированными initial states.')

def raw_environment(vec):
    for obj in wrapper_nodes(vec):
        if getattr(obj, 'robots', None) and getattr(obj, 'sim', None) is not None:
            return obj
    raise RuntimeError('Не найден физический LIBERO environment.')

def physical_state(vec):
    sim = raw_environment(vec).sim
    state, ctrl = (sim.get_state().flatten().copy(), sim.data.ctrl.copy())
    require(np.isfinite(state).all() and np.isfinite(ctrl).all(), 'Неконечные physics/control values.')
    return (state, ctrl)

def benchmark_success(vec):
    raw = raw_environment(vec)
    for name in ('check_success', '_check_success'):
        predicate = getattr(raw, name, None)
        if callable(predicate):
            value = np.asarray(predicate())
            require(value.size == 1, 'Ожидается один benchmark success flag.')
            return bool(value.item())
    raise RuntimeError('Не найден LIBERO benchmark success predicate.')

def geometry(vec):
    sim = raw_environment(vec).sim
    record = {'simulator_time': float(sim.data.time)}
    for name in list(sim.model.body_names):
        if name in ('akita_black_bowl_1_main', 'plate_1_main'):
            index = sim.model.body_name2id(name)
            record[name] = sim.data.body_xpos[index].tolist()
    return record

def save_video(filename, frames, fps):
    import av
    require(bool(frames), 'Нельзя записать пустое видео.')
    height, width = frames[0].shape[:2]
    with av.open(str(filename), 'w') as container:
        stream = container.add_stream('libx264', rate=int(fps))
        stream.width, stream.height, stream.pix_fmt = (width, height, 'yuv420p')
        stream.options = {'crf': '18', 'preset': 'fast'}
        for image in frames:
            frame = av.VideoFrame.from_ndarray(image, format='rgb24')
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)

class WorkerClient:

    def __init__(self, root, output):
        self.root, self.output = (Path(root), Path(output))
        self.queue = self.output / 'ipc'
        self.queue.mkdir()
        self.process = self.log_file = None
        self.log_offset, self.index = (0, 0)

    def start(self):
        plan = worker_memory_plan()
        print_memory_plan(plan)
        require(plan['ready'], memory_failure_message(plan))
        python = self.root / 'vla_insight/openpi/.venv/bin/python'
        require(python.is_file(), 'Нет подготовленной OpenPI venv. Нужен start_insight.py prepare.')
        require(shutil.which('systemd-run'), 'Нужен systemd-run для ограничения RAM.')
        subprocess.run(['systemctl', '--user', 'show-environment'], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        cap = plan['cap']
        argv = ['systemd-run', '--user', '--scope', '--quiet', '-p', f'MemoryMax={cap}', '-p', 'MemorySwapMax=0', '--', str(python), str(Path(__file__).resolve()), '--worker', '--output', str(self.output), '--parent-pid', str(os.getpid())]
        env = bridge.isolated_env(self.root)
        env['XLA_PYTHON_CLIENT_ALLOCATOR'] = 'platform'
        env['PYTHONUNBUFFERED'] = '1'
        self.log_file = (self.output / 'worker.log').open('w', encoding='utf-8')
        print(f'VLA worker: allocator=platform; RAM limit={cap / bridge.GIB:.2f} GiB; swap=0.', flush=True)
        print('Log:', self.output / 'worker.log', flush=True)
        self.process = subprocess.Popen(argv, cwd=self.root, env=env, start_new_session=True, stdout=self.log_file, stderr=subprocess.STDOUT)
        self.wait(self.output / 'worker_ready.json', 'загрузка VLA')

    def show_log(self):
        log = self.output / 'worker.log'
        if log.exists():
            with log.open(encoding='utf-8', errors='replace') as file:
                file.seek(self.log_offset)
                content = file.read()
                self.log_offset = file.tell()
            if content:
                print(content, end='', flush=True)

    def wait(self, path, stage, timeout=1200):
        deadline, update = (time.monotonic() + timeout, time.monotonic() + 30)
        while True:
            self.show_log()
            if path.is_file():
                result = bridge.read_json(path)
                require(result.get('status') in ('ready', 'ok', 'invalid_output'), f"VLA worker: {result.get('error', result)}; лог {self.output / 'worker.log'}")
                return result
            code = self.process.poll()
            if code is not None:
                hint = ' SIGKILL: возможен OOM; причину нужно подтверждать в kernel journal.' if code == -signal.SIGKILL else ''
                raise RuntimeError(f"VLA worker остановлен (exit={code});{hint} лог {self.output / 'worker.log'}")
            require(time.monotonic() < deadline, f"Превышен лимит ожидания ({stage}); лог {self.output / 'worker.log'}")
            if time.monotonic() >= update:
                print(f'Waiting: {stage}...', flush=True)
                update = time.monotonic() + 30
            time.sleep(0.2)

    def infer(self, query, observation):
        query = Path(query)
        source_items_1 = observation.items()
        items_1 = {}
        for k, v in source_items_1:
            if k != 'prompt':
                items_1[k] = v
        np.savez_compressed(query / 'input.npz', **items_1)
        json_atomic(query / 'input.json', {'prompt': observation['prompt']})
        request_id = self.index
        request = {'command': 'infer', 'request_id': request_id, 'query': str(query.relative_to(self.output))}
        json_atomic(self.queue / f'request_{request_id:05d}.json', request)
        reply = self.wait(self.queue / f'response_{request_id:05d}.json', 'inference')
        require(reply['request_id'] == request_id and reply['query'] == request['query'], 'Ответ VLA относится к другому запросу.')
        self.index += 1
        if reply['status'] == 'invalid_output':
            raise InferenceOutputError(reply['error_code'], reply['error'], info=reply)
        with np.load(query / 'token_features.npz', allow_pickle=False) as data:
            actions = data['actions'].copy()
        require(actions.shape == (10, 7) and np.isfinite(actions).all(), 'Некорректный action chunk.')
        return (actions, reply)

    def close(self):
        if self.process is not None:
            json_atomic(self.queue / 'stop.json', {'command': 'stop'})
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(self.process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                try:
                    self.process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    try:
                        os.killpg(self.process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    self.process.wait(timeout=5)
            self.show_log()
        if self.log_file is not None:
            self.log_file.close()

def infer_with_features(policy, sampler, observation, trigger):
    import jax
    import jax.numpy as jnp
    from openpi.models import model as model_module
    started = time.monotonic()
    inputs = policy._input_transform(dict(observation))

    def add_batch_dimension(value):
        return jnp.asarray(value)[None]
    inputs = jax.tree.map(add_batch_dimension, inputs)
    obs = model_module.Observation.from_dict(inputs)
    tokens, tracks, count, done = jax.device_get(sampler(jax.random.key(0), obs))
    count = int(count)
    require(0 < count <= tracks.shape[1], 'Некорректное число generated tokens.')
    raw = np.asarray(tracks[0, :count], dtype=np.float32)
    require(raw.shape == (count, 4) and np.isfinite(raw).all(), 'Некорректные token features.')
    require(np.all(raw[:, 2] >= -1e-05) and np.all(raw[:, 3] <= 1e-05) and np.all(raw[:, 1] >= 0) and np.all(raw[:, 1] <= 1.00001), 'Некорректные entropy/log-p/EU.')
    token_sequence_complete = bool(done)
    classifier_ready = token_sequence_complete and count > 5
    features = trigger.features_from_policy_output({'au': raw[:, 0], 'eu': raw[:, 1], 'entropy': raw[:, 2], 'perplexity': raw[:, 3]}) if classifier_ready else np.empty((0, 4), dtype=np.float32)
    if classifier_ready:
        require(features.shape == (count - 5, 4), 'Нарушено обрезание служебных токенов INSIGHT.')
    arrays = {'token_ids': tokens[0, :count].astype(np.int32), 'raw_features': raw, 'features': features, 'eos': np.asarray(bool(done))}
    info = {'generated_tokens': count, 'classifier_tokens': len(features), 'feature_order': bridge.FEATURE_ORDER, 'help_score': None, 'eos': bool(done), 'decoder_protocol': DECODER_PROTOCOL, 'token_sequence_complete': token_sequence_complete, 'features_valid_for_classifier': classifier_ready, 'eos_policy': 'diagnostic_only_for_action_execution;required_for_strong_features', 'elapsed_seconds': time.monotonic() - started}
    if count <= 5:
        raise InferenceOutputError('short_token_sequence', f'Слишком короткая FAST token sequence: tokens={count}, EOS={done}.', arrays=arrays, info=info)
    if not token_sequence_complete:
        info.update(token_warning='missing_eos', feature_exclusion_reason='missing_eos')
        print(f'WARNING: FAST EOS was not received before the limit; checking the action payload: tokens={count}.', flush=True)
    try:
        decoded = checked_output_transform(policy, np.asarray(inputs['state'][0]), tokens[0])
        actions = np.asarray(decoded['actions'], dtype=np.float32)
        if actions.shape != (10, 7) or not np.isfinite(actions).all():
            raise InferenceOutputError('invalid_actions', 'Некорректные VLA actions.')
    except InferenceOutputError as error:
        if error.code in ('missing_action_marker', 'missing_action_separator'):
            arrays['features'] = np.empty((0, 4), dtype=np.float32)
            info.update(classifier_tokens=0, features_valid_for_classifier=False)
        info['elapsed_seconds'] = time.monotonic() - started
        error.arrays, error.info = (arrays, info)
        raise
    arrays['actions'] = actions
    info['elapsed_seconds'] = time.monotonic() - started
    return (arrays, info)

def save_inference_response(query, queue, index, infer):
    try:
        arrays, info = infer()
        info.update(status='ok')
    except InferenceOutputError as error:
        arrays = error.arrays
        info = {**error.info, 'status': 'invalid_output', 'error_code': error.code, 'error': str(error), 'fallback_actions_executed': False}
        print(f'INVALID MODEL OUTPUT: {error.code}: {error}', flush=True)
    np.savez_compressed(query / 'token_features.npz', **arrays)
    info.update(request_id=index, query=str(query.relative_to(queue.parent)))
    json_atomic(query / 'inference.json', info)
    json_atomic(queue / f'response_{index:05d}.json', info)
    return info

def parent_alive(pid):
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False

def model_worker(root, args):
    output, queue = (Path(args.output), Path(args.output) / 'ipc')
    index = 0
    try:
        import jax
        from openpi.models import model as model_module
        from openpi.policies import policy_config
        from openpi.shared import nnx_utils
        from openpi.training import config
        require(all((device.platform == 'gpu' for device in jax.devices())), 'JAX CUDA недоступен.')
        bridge.memory_note(output, 'imports_ready')
        record = bridge.read_json(output / 'run_info.json')
        cfg = config.get_config('pi0_fast_libero')
        cfg = dataclasses.replace(cfg, data=bridge.ZScoreFactory(cfg.data))
        with bridge.bounded_openpi_restore(model_module, output):
            policy = policy_config.create_trained_policy(cfg, Path(record['checkpoint_local_path']))
        bridge.memory_note(output, 'policy_ready')
        sampler = nnx_utils.module_jit(MethodType(bridge.sample_with_features, policy._model))
        sys.path.insert(0, str(root / 'vla_insight/insight'))
        from insight.strong_trigger import StrongHelpTrigger
        trigger = StrongHelpTrigger.__new__(StrongHelpTrigger)
        trigger.trim_head, trigger.trim_tail = (3, 2)
        json_atomic(output / 'worker_ready.json', {'status': 'ready', 'pid': os.getpid(), 'versions': versions(), 'gpu': bridge.gpu_info()})
        print('VLA WEIGHTS LOADED', flush=True)
        last_active = time.monotonic()
        while parent_alive(args.parent_pid) and (not (queue / 'stop.json').exists()):
            require(time.monotonic() - last_active < 1200, 'Worker слишком долго не получает запросы.')
            request_path = queue / f'request_{index:05d}.json'
            if not request_path.is_file():
                time.sleep(0.1)
                continue
            request = bridge.read_json(request_path)
            require(request['command'] == 'infer' and request['request_id'] == index, 'Некорректная очередь запросов.')
            query = (output / request['query']).resolve()
            require(query.is_relative_to(output.resolve()), 'Query должен быть внутри папки этого запуска.')
            with np.load(query / 'input.npz', allow_pickle=False) as data:
                source_items_2 = data.files
                observation = {}
                for key in source_items_2:
                    observation[key] = data[key].copy()
            observation['prompt'] = bridge.read_json(query / 'input.json')['prompt']
            if index == 0:
                bridge.memory_note(output, 'inference_compile_begin')

            def run_inference():
                return infer_with_features(policy, sampler, observation, trigger)
            info = save_inference_response(query, queue, index, run_inference)
            if index == 0:
                bridge.memory_note(output, 'first_inference_complete')
            index += 1
            last_active = time.monotonic()
        bridge.memory_note(output, 'worker_complete')
    except BaseException as error:
        record = {'status': 'error', 'error': f'{type(error).__name__}: {error}', 'traceback': traceback.format_exc()}
        json_atomic(output / 'worker_error.json', record)
        if not (output / 'worker_ready.json').exists():
            json_atomic(output / 'worker_ready.json', record)
        else:
            json_atomic(queue / f'response_{index:05d}.json', record)
        raise

def run_episode(vec, client, output, init_id, args):
    folder = output / f'init_{init_id:03d}'
    folder.mkdir()
    (folder / 'queries').mkdir()
    wrapper = libero_wrapper(vec)
    require(wrapper._init_states is not None and 0 <= init_id < len(wrapper._init_states), f'Initial-state ID {init_id} вне доступного диапазона.')
    definition = np.asarray(wrapper._init_states[init_id], dtype=np.float64)
    wrapper.init_state_id = init_id
    observation, _ = vec.reset(seed=args.seed)
    require(not benchmark_success(vec), 'Начальное состояние уже удовлетворяет цели.')
    instruction = str(libero_wrapper(vec).task_description)
    hz = int(libero_wrapper(vec).control_freq)
    require(hz == CONTROL_HZ, f'Изменился control_freq: {hz}.')
    reset_state, reset_ctrl = physical_state(vec)
    np.savez_compressed(folder / 'reset.npz', init_definition=definition, simulator_state=reset_state, ctrl=reset_ctrl, **flatten_observation(observation))
    horizon = args.max_steps or SUITE_HORIZONS[args.suite]
    record = {'status': 'running', 'scope': RUN_SCOPE, 'suite': args.suite, 'task_id': args.task_id, 'instruction': instruction, 'init_state_id': init_id, 'seed': args.seed, 'max_steps': horizon, 'replan_steps': args.replan_steps, 'control_hz': hz, 'success': None, 'intervention': None, 'inferences': []}
    json_atomic(folder / 'episode.json', record)
    actions, rewards, frames, raw_records, trajectory, labels = ([], [], [], [], [], [])
    success = False
    ended = False
    try:
        while len(actions) < horizon and (not ended):
            step = len(actions)
            query = folder / 'queries' / f'step_{step:04d}'
            query.mkdir()
            state, ctrl = physical_state(vec)
            raw_obs = flatten_observation(observation)
            prefix = np.asarray(actions, dtype=np.float32).reshape(-1, 1, 7)
            np.savez_compressed(query / 'start.npz', simulator_state=state, ctrl=ctrl, reset_state=reset_state, reset_ctrl=reset_ctrl, past_actions=prefix, **raw_obs)
            chunk, info = client.infer(query, openpi_observation(observation, instruction))
            expected_steps = min(args.replan_steps, horizon - step)
            boundary = {'step': step, 'query': str(query.relative_to(output)), 'snapshot_sha256': bridge.digest(query / 'start.npz'), 'features_sha256': bridge.digest(query / 'token_features.npz'), 'expected_executed_actions': expected_steps, 'inference': info}
            record['inferences'].append(boundary)
            labels.append({'step': step, 'query': boundary['query'], 'help_required': None, 'failure_type': None, 'annotator': None, 'note': 'Label only with context ending at this inference boundary.'})
            print(f"init_id={init_id}: step={step}; tokens={info['generated_tokens']}; inference={info['elapsed_seconds']:.2f}s; help_score=None", flush=True)
            executed = 0
            for action in chunk[:expected_steps]:
                raw_obs = flatten_observation(observation)
                actual = np.clip(np.asarray(action, dtype=np.float32), -1, 1)
                state, ctrl = physical_state(vec)
                row = {'step': len(actions), 'geometry': geometry(vec), 'predicted_action': action.tolist(), 'executed_action': actual.tolist(), 'inference_step': step, 'chunk_index': executed}
                next_observation, reward, terminated, truncated, _ = vec.step(actual[None])
                success = benchmark_success(vec)
                ended = success or bool(np.any(terminated) or np.any(truncated))
                row.update(success_after_action=success)
                actions.append(actual.copy())
                rewards.append(float(np.asarray(reward).reshape(-1)[0]))
                raw_records.append({**raw_obs, 'simulator_state': state, 'ctrl': ctrl})
                frames.append(np.ascontiguousarray(raw_obs['observation/pixels/image'][0, ::-1, ::-1]))
                trajectory.append(row)
                observation = next_observation
                executed += 1
                if ended:
                    break
            boundary['executed_actions'] = executed
            record.update(steps=len(actions), success=success)
            json_atomic(folder / 'episode.json', record)
        record.update(status='complete', success=success, steps=len(actions), ended=ended)
    except BaseException as error:
        record.update(status='interrupted', success=None, steps=len(actions), error=f'{type(error).__name__}: {error}')
        raise
    finally:
        if raw_records:
            arrays = {key: np.stack([row[key] for row in raw_records]) for key in raw_records[0]}
            np.savez_compressed(folder / 'rollout.npz', action=np.asarray(actions, dtype=np.float32), reward=np.asarray(rewards, dtype=np.float32), **arrays)
            try:
                save_video(folder / 'rollout.mp4', frames, hz)
                record['video'] = str(folder / 'rollout.mp4')
            except Exception as error:
                record['video_error'] = str(error)
                print('Video write failed:', error, flush=True)
        terminal_state, terminal_ctrl = physical_state(vec)
        np.savez_compressed(folder / 'final.npz', simulator_state=terminal_state, ctrl=terminal_ctrl, **flatten_observation(observation))
        json_atomic(folder / 'trajectory.json', trajectory)
        json_atomic(folder / 'help_labels_template.json', {'status': 'unannotated', 'labels': labels, 'note': 'An episode outcome is not a Strong INSIGHT step-level help label.'})
        json_atomic(folder / 'episode.json', record)
    print(f"EPISODE init_id={init_id}: success={success}; steps={len(actions)}; video={folder / 'rollout.mp4'}", flush=True)
    return record

def run_rollouts(root, args):
    require(len(set(args.init_ids)) == len(args.init_ids) and all((i >= 0 for i in args.init_ids)), 'Нужны разные неотрицательные --init-ids.')
    require(args.max_steps is None or args.max_steps > 0, '--max-steps должен быть положительным.')
    bridge.check_repositories(root)
    checkpoint, files = bridge.ensure_checkpoint()
    output = root / 'pilot_runs/vla_insight' / ('rollouts_' + datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f'))
    output.mkdir(parents=True, exist_ok=False)
    record = {'status': 'running', 'scope': RUN_SCOPE, 'source_policy': 'pi0-FAST', 'checkpoint': bridge.CHECKPOINT, 'checkpoint_local_path': str(checkpoint), 'checkpoint_files': files, 'openpi_commit': bridge.OPENPI_COMMIT, 'insight_commit': bridge.INSIGHT_COMMIT, 'normalization': 'zscore', 'allocator': 'platform', 'restore_concurrent_gb': bridge.RESTORE_CONCURRENT_GB, 'suite': args.suite, 'task_id': args.task_id, 'init_state_ids': args.init_ids, 'seed': args.seed, 'replan_steps': args.replan_steps, 'max_steps': args.max_steps or SUITE_HORIZONS[args.suite], 'feature_order': bridge.FEATURE_ORDER, 'help_detector_trained': False, 'decoder_protocol': DECODER_PROTOCOL, 'control_hz': CONTROL_HZ, 'trim_head': 3, 'trim_tail': 2, 'parent_versions': versions(), 'script_sha256': bridge.digest(Path(__file__)), 'bridge_sha256': bridge.digest(root / 'start_insight.py'), 'episodes': []}
    json_atomic(output / 'run_info.json', record)
    print('Output:', output, flush=True)
    client, vec = (WorkerClient(root, output), None)
    try:
        client.start()
        os.environ.setdefault('MUJOCO_GL', 'egl')
        from lerobot.envs.configs import LiberoEnv
        from lerobot.envs.factory import make_env
        horizon = record['max_steps']
        cfg = LiberoEnv(task=args.suite, task_ids=[args.task_id], control_mode='relative', init_states=True, hard_reset=True, max_parallel_tasks=1, observation_height=256, observation_width=256, episode_length=horizon)
        envs = make_env(cfg, n_envs=1, use_async_envs=False)
        vec = envs[args.suite][args.task_id]
        for init_id in args.init_ids:
            episode = run_episode(vec, client, output, init_id, args)
            record['episodes'].append(episode)
            json_atomic(output / 'run_info.json', record)
        record.update(status='complete', n_episodes=len(record['episodes']), n_success=sum((row['success'] for row in record['episodes'])))
        record['pc_success'] = 100.0 * record['n_success'] / record['n_episodes']
    except BaseException as error:
        record.update(status='interrupted', error=f'{type(error).__name__}: {error}')
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
    source_items_3 = ('status', 'scope', 'source_policy', 'suite', 'task_id', 'n_episodes', 'n_success', 'pc_success')
    items_3 = {}
    for key in source_items_3:
        items_3[key] = record[key]
    json_atomic(output / 'summary.json', items_3)
    pointer = root / 'pilot_runs/vla_last_rollouts.txt'
    pointer.write_text(str(output) + '\n', encoding='utf-8')
    print(f"\nVLA ROLLOUTS COMPLETE: {record['n_success']}/{record['n_episodes']}")
    print('Output:', output)
    print('INSIGHT features saved. The help classifier and the policy were not trained here.')

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--init-ids', type=int, nargs='+', default=[0])
    parser.add_argument('--suite', choices=tuple(SUITE_HORIZONS), default='libero_spatial')
    parser.add_argument('--task-id', type=int, default=0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--replan-steps', type=int, choices=range(1, 11), default=5)
    parser.add_argument('--max-steps', type=int)
    parser.add_argument('--memory-check', action='store_true', help='Проверить RAM без загрузки VLA или запуска симулятора.')
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--output', help=argparse.SUPPRESS)
    parser.add_argument('--parent-pid', type=int, help=argparse.SUPPRESS)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    if args.worker:
        require(args.output and args.parent_pid, 'Worker arguments отсутствуют.')
        model_worker(root, args)
    elif args.memory_check:
        plan = worker_memory_plan()
        print_memory_plan(plan)
        require(plan['ready'], memory_failure_message(plan))
    else:
        run_rollouts(root, args)
if __name__ == '__main__':
    try:
        main()
    except (Exception, KeyboardInterrupt) as error:
        print(f'\nSTOP: {error}', file=sys.stderr, flush=True)
        sys.exit(1)
