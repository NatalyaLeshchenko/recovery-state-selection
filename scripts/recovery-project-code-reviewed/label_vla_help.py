#!/usr/bin/env python3
from __future__ import annotations
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import random
import re
import secrets
import sys
import threading
from urllib.parse import parse_qs, urlsplit
import webbrowser
import numpy as np
from PIL import Image
FEATURE_ORDER = ['AU', 'EU', 'entropy', 'chosen_token_log_probability']
ROLES = {'detector_train': range(0, 10), 'detector_validation': range(10, 15)}
POINTER = 'pilot_runs/vla_help_annotation_path.txt'
QUESTION = 'Would you request an expert intervention now, from the task and visual context up to this boundary?'

def require(ok, message):
    if not ok:
        raise ValueError(message)

def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))

def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    temporary.replace(path)

def sha256(path):
    with Path(path).open('rb') as file:
        return hashlib.file_digest(file, 'sha256').hexdigest()

def timestamp():
    return datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')

def checked_path(run, relative):
    path = (run / relative).resolve()
    require(path.is_relative_to(run.resolve()), 'Query path выходит за пределы collection.')
    return path

def snapshot_images(query, expected_digest=None):
    path = query / 'start.npz'
    if expected_digest is not None:
        require(sha256(path) == expected_digest, f'Изменился snapshot: {path}')
    with np.load(path, allow_pickle=False) as data:
        result = []
        for key in ('observation/pixels/image', 'observation/pixels/image2'):
            image = data[key]
            require(image.dtype == np.uint8 and image.ndim == 4 and (image.shape[0] == 1) and (image.shape[-1] == 3), f'Некорректное RGB image: {key}')
            result.append(np.ascontiguousarray(image[0, ::-1, ::-1]))
    return result

def save_image(output, image):
    identity = hashlib.sha256(str(image.shape).encode() + image.tobytes()).hexdigest()
    relative = f'assets/{identity}.png'
    path = output / relative
    if not path.exists():
        Image.fromarray(image).save(path, compress_level=2)
    return relative

def classifier_features(query, boundary):
    path = query / 'token_features.npz'
    if not path.is_file():
        return (None, 'no_token_features')
    if boundary.get('features_sha256') is not None:
        require(sha256(path) == boundary['features_sha256'], f'Изменились features: {path}')
    info_path = query / 'inference.json'
    info = read_json(info_path) if info_path.is_file() else boundary.get('inference', {})
    if info.get('features_valid_for_classifier') is not True:
        return (None, info.get('error_code', 'features_not_valid_for_classifier'))
    require(info.get('feature_order') == FEATURE_ORDER, 'Изменился порядок INSIGHT features.')
    with np.load(path, allow_pickle=False) as data:
        features = data['features'].astype(np.float32, copy=True)
        eos = bool(data['eos']) if 'eos' in data.files else info.get('eos') is True
    require(eos and features.ndim == 2 and (features.shape[1] == 4) and (len(features) > 0) and np.isfinite(features).all(), f'Некорректная Strong sequence: {path}')
    require(len(features) == info['generated_tokens'] - 5, 'Не выполнено обрезание 3+2 служебных токенов.')
    return (features, None)

def prepare(root, runs=None, workspace=None):
    paths = [Path(path).expanduser().resolve() for path in runs] if runs else sorted((root / 'pilot_runs/vla_insight').glob('failures_*'))
    require(bool(paths), 'Не найдены VLA collections. Нужны папки failures_* рядом с проектом.')
    output = Path(workspace).expanduser().resolve() if workspace else root / 'pilot_runs/vla_insight' / ('help_annotation_' + timestamp())
    output.mkdir(parents=True, exist_ok=False)
    (output / 'assets').mkdir()
    samples, audit, run_sources, dedup = ([], [], [], {})
    signature = None
    for run in paths:
        record_path = run / 'run_info.json'
        if not record_path.is_file():
            continue
        record = read_json(record_path)
        role = record.get('purpose')
        if role not in ROLES or record.get('decoder_protocol') not in (2, 3):
            audit.append({'run': str(run), 'reason': 'not_detector_data_or_legacy_decoder'})
            continue
        source_items_1 = ('checkpoint', 'openpi_commit', 'insight_commit', 'normalization', 'feature_order', 'suite', 'task_id', 'control_hz', 'replan_steps', 'trim_head', 'trim_tail')
        current = {}
        for key in source_items_1:
            current[key] = record.get(key)
        require(current['feature_order'] == FEATURE_ORDER and current['normalization'] == 'zscore', f'Неожиданный feature / normalization protocol: {run}')
        require(current['suite'] == 'libero_spatial' and current['task_id'] == 0, 'Этот разметчик рассчитан на текущий однотасковый pilot.')
        require(current['control_hz'] == 20 and current['replan_steps'] == 5 and (current['trim_head'] == 3) and (current['trim_tail'] == 2), 'Изменился control / Strong feature protocol: нужны 20 Hz, replan=5, trim=3+2.')
        if signature is None:
            signature = current
        require(current == signature, 'Collections используют разные policy / feature protocols.')
        run_sources.append({'path': str(run), 'run_info_sha256': sha256(record_path), 'purpose': role})
        for case_path in sorted(run.glob('init_*/case.json')):
            case = read_json(case_path)
            init_id = case['init_state_id']
            require(type(init_id) is int and init_id in ROLES[role] and (case['purpose'] == role), f'Нарушен фиксированный init-ID split: {case_path}')
            require(case.get('decoder_protocol') in (2, 3), f'Legacy case: {case_path}')

            def boundary_step(boundary):
                return boundary['step']
            boundaries = sorted(case['inferences'], key=boundary_step)
            source_items_2 = boundaries
            items_2 = set()
            for b in source_items_2:
                items_2.add(b['step'])
            require(len(items_2) == len(boundaries), 'Повторяющиеся query steps.')
            frame_cache = {}

            def frame(boundary):
                step = boundary['step']
                if step not in frame_cache:
                    query = checked_path(run, boundary['query'])
                    images = snapshot_images(query, boundary.get('snapshot_sha256'))
                    source_items_3 = images
                    items_3 = []
                    for image in source_items_3:
                        items_3.append(save_image(output, image))
                    frame_cache[step] = {'step': step, 'views': items_3}
                return frame_cache[step]
            for boundary in boundaries:
                step = boundary['step']
                require(type(step) is int and step >= 0, 'Некорректный query step.')
                query = checked_path(run, boundary['query'])
                features, reason = classifier_features(query, boundary)
                if reason:
                    audit.append({'query': str(query), 'purpose': role, 'init_id': init_id, 'reason': reason})
                    continue
                source_items_4 = boundaries
                items_4 = []
                for b in source_items_4:
                    if max(0, step - 40) <= b['step'] < step:
                        items_4.append(b)
                previous = items_4[-4:]
                source_items_5 = previous
                items_5 = []
                for b in source_items_5:
                    items_5.append(frame(b))
                context = items_5 + [frame(boundary)]
                identity = hashlib.sha256(json.dumps({'role': role, 'init_id': init_id, 'instruction': case['instruction'], 'context': context}, sort_keys=True).encode() + features.tobytes()).hexdigest()
                source = {'query': str(query), 'features_path': str(query / 'token_features.npz'), 'features_sha256': sha256(query / 'token_features.npz'), 'snapshot_sha256': sha256(query / 'start.npz'), 'run': str(run)}
                if identity in dedup:
                    dedup[identity]['sources'].append(source)
                    continue
                sample = {'sample_id': identity, 'purpose': role, 'init_id': init_id, 'instruction': case['instruction'], 'step': step, 'context': context, 'classifier_tokens': len(features), 'sources': [source]}
                samples.append(sample)
                dedup[identity] = sample
        print(f'Prepared from {run.name}: уникальных observations {len(samples)}', flush=True)
    require(bool(samples), 'Нет queries с допустимыми Strong features; проверь audit и decoder protocol.')
    ordered = []
    for role in ROLES:
        source_items_6 = samples
        group = []
        for sample in source_items_6:
            if sample['purpose'] == role:
                group.append(sample)
        random.Random(0).shuffle(group)
        ordered.extend(group)
    source_items_7 = enumerate(sorted({(s['purpose'], s['init_id']) for s in ordered}))
    items_7 = {}
    for index, key in source_items_7:
        items_7[key] = f'E{index + 1:02d}'
    aliases = items_7
    for sample in ordered:
        sample['episode_alias'] = aliases[sample['purpose'], sample['init_id']]
    source_items_8 = ROLES.items()
    items_8 = {}
    for role, ids in source_items_8:
        items_8[role] = list(ids)
    manifest = {'schema_version': 1, 'created_at': datetime.now(timezone.utc).isoformat(), 'label_question': QUESTION, 'label_values': {'0': 'continue autonomously', '1': 'request expert now', 'null': 'unknown / unreviewed; excluded from training'}, 'context_protocol': 'current pre-action views + up to 4 earlier query views within 40 control steps', 'hidden_from_reviewer': ['outcomes', 'failure_type', 'phase', 'uncertainty', 'decoder_status'], 'fixed_split': items_8, 'policy_protocol': signature, 'runs': run_sources, 'samples': ordered}
    write_json(output / 'manifest.json', manifest)
    write_json(output / 'excluded_queries.json', audit)
    write_json(output / 'labels.json', {'schema_version': 1, 'manifest_sha256': sha256(output / 'manifest.json'), 'label_question': QUESTION, 'answers': {}})
    pointer = root / POINTER
    pointer.parent.mkdir(parents=True, exist_ok=True)
    pointer.write_text(str(output) + '\n', encoding='utf-8')
    print('\nReview:', output)
    print('Unique observations:', dict(Counter((s['purpose'] for s in ordered))))
    return output

def resolve_workspace(root, value=None):
    output = Path(value).expanduser().resolve() if value else Path((root / POINTER).read_text().strip())
    require((output / 'manifest.json').is_file(), f'Нет workspace: {output}')
    return output

class Review:

    def __init__(self, output):
        self.output = output
        self.manifest = read_json(output / 'manifest.json')
        self.samples = self.manifest['samples']
        self.labels = read_json(output / 'labels.json')
        require(self.labels['manifest_sha256'] == sha256(output / 'manifest.json'), 'Manifest изменился после начала разметки.')
        self.lock = threading.Lock()
        self.token = secrets.token_urlsafe(24)

    def stats(self):
        result = {}
        for role in ROLES:
            counts = Counter({'continue': 0, 'help': 0, 'unknown': 0, 'unreviewed': 0})
            for sample in self.samples:
                if sample['purpose'] != role:
                    continue
                answer = self.labels['answers'].get(sample['sample_id'])
                key = 'unreviewed' if answer is None else 'unknown' if answer['help_required'] is None else 'help' if answer['help_required'] == 1 else 'continue'
                counts[key] += 1
            result[role] = dict(counts)
        first = next((i for i, s in enumerate(self.samples) if s['sample_id'] not in self.labels['answers']), 0)
        return {'counts': result, 'total': len(self.samples), 'first_unreviewed': first}

    def item(self, index):
        require(0 <= index < len(self.samples), 'Observation index вне диапазона.')
        sample = self.samples[index]
        source_items_9 = ('sample_id', 'purpose', 'instruction', 'step', 'context', 'episode_alias')
        items_9 = {}
        for key in source_items_9:
            items_9[key] = sample[key]
        return items_9 | {'answer': self.labels['answers'].get(sample['sample_id']), 'index': index}

    def save(self, data):
        identity = data.get('sample_id')
        source_items_10 = self.samples
        items_10 = set()
        for s in source_items_10:
            items_10.add(s['sample_id'])
        require(identity in items_10, 'Неизвестный sample ID.')
        value = data.get('help_required')
        require(value is None or (type(value) is int and value in (0, 1)), 'Help label должен быть 0, 1 или null.')
        annotator = data.get('annotator', '')
        note = data.get('note', '')
        require(isinstance(annotator, str) and 0 < len(annotator.strip()) <= 80, 'Укажи имя annotator.')
        require(isinstance(note, str) and len(note) <= 1000, 'Слишком длинный комментарий.')
        with self.lock:
            self.labels['answers'][identity] = {'help_required': value, 'reviewed': True, 'annotator': annotator.strip(), 'note': note.strip(), 'updated_at': datetime.now(timezone.utc).isoformat()}
            write_json(self.output / 'labels.json', self.labels)
HTML = '<!doctype html><html lang="ru"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">\n<title>VLA · Help labels</title><style>\n*{box-sizing:border-box}body{margin:0;background:#f4f5f8;color:#212a36;font:16px system-ui,sans-serif}main{max-width:1060px;margin:24px auto;padding:0 18px}.card{background:white;border:1px solid #dde2e9;border-radius:14px;padding:18px;margin:14px 0}h1{font-size:24px;margin:0}p{line-height:1.5}.muted{color:#657080;font-size:14px}.top,.nav,.actions,.views,.fields{display:flex;gap:12px;align-items:center;justify-content:space-between}.views>div{flex:1;min-width:0}.views img{width:100%;max-height:380px;object-fit:contain;background:#e9edf3;border-radius:8px}.timeline{display:flex;gap:10px;overflow:auto;margin-top:16px}.timeline button{padding:5px;flex:0 0 112px}.timeline img{width:100px;display:block}button,input,textarea{font:inherit}button{border:1px solid #c7d0dc;border-radius:8px;background:#fff;padding:10px 14px;cursor:pointer}button:disabled{opacity:.5;cursor:wait}button.selected{outline:3px solid #8aa9d1}.actions button{flex:1;min-height:58px}.continue{background:#eaf3fe}.help{background:#fff0e1}.unknown{background:#f0f1f4}input,textarea{border:1px solid #c7d0dc;border-radius:7px;padding:9px;width:100%}label{display:block}.fields>label:first-child{flex:1}.fields>label:last-child{flex:3}.status{min-height:24px;color:#12643c}.error{color:#b22424}#task{font-size:18px;margin:10px 0}.badge{font-size:13px;background:#edf0f5;padding:5px 9px;border-radius:20px}@media(max-width:650px){.views,.fields,.actions{flex-direction:column}.actions button,.views>div,.fields>label{width:100%}.top{align-items:flex-start}.timeline button{flex-basis:85px}.timeline img{width:73px}}\n</style><main><div class="top"><h1>VLA · Help labels</h1><span class="badge">Локальная разметка</span></div>\n<div class="card"><p style="margin-top:0"><b>Ты оператор. Попросила бы ты expert вмешаться прямо сейчас?</b></p><p class="muted">Оцени задачу и кадры до текущего запроса. Падение или промах сами по себе не задают ответ. Если по кадрам нельзя решить, выбери «Не уверена». Последующие кадры и исход эпизода здесь не показываются.</p><div id="counts" class="muted"></div></div>\n<div class="card"><div class="nav"><button id="prev">← Назад</button><b id="position"></b><button id="next">Далее →</button></div><div id="meta" class="muted" style="margin-top:12px"></div><p id="task"></p>\n<div class="views"><div><div class="muted">Внешняя камера</div><img id="base" alt="Наблюдение внешней камеры"></div><div><div class="muted">Камера на захвате</div><img id="wrist" alt="Наблюдение камеры на захвате"></div></div><div id="shown" class="muted"></div><div id="timeline" class="timeline"></div></div>\n<div class="card"><div class="fields"><label>Кто размечает<input id="annotator" maxlength="80" placeholder="Имя"></label><label>Комментарий, если нужен<textarea id="note" rows="2" maxlength="1000"></textarea></label></div>\n<div class="actions" style="margin-top:14px"><button class="continue" data-label="0">0 · Продолжить самостоятельно</button><button class="help" data-label="1">1 · Expert сейчас</button><button class="unknown" data-label="unknown">U · Не уверена</button></div><p id="status" class="status"></p><div class="muted">Ответ сохраняется в labels.json на этом компьютере. Можно закрыть окно и продолжить позже. Клавиши: 0, 1, U; стрелки — переход.</div></div></main>\n<script>\nconst TOKEN=\'__TOKEN__\';let index=0,item=null,busy=false,loading=false,total=0;const $=id=>document.getElementById(id);\n$(\'annotator\').value=localStorage.getItem(\'vla_annotator\')||\'\';\nfunction message(text,error=false){$(\'status\').textContent=text;$(\'status\').className=\'status\'+(error?\' error\':\'\');}\nasync function api(path,body){const r=await fetch(path,body===undefined?{}:{method:\'POST\',headers:{\'Content-Type\':\'application/json\',\'X-Annotation-Token\':TOKEN},body:JSON.stringify(body)});const data=await r.json();if(!r.ok)throw Error(data.error||r.statusText);return data;}\nasync function state(){const s=await api(\'/api/state\');total=s.total;$(\'counts\').textContent=Object.entries(s.counts).map(([role,c])=>`${role===\'detector_train\'?\'Обучение\':\'Проверка\'}: 0 — ${c.continue}, 1 — ${c.help}, не уверена — ${c.unknown}, осталось — ${c.unreviewed}`).join(\' · \');return s;}\nfunction controls(){document.querySelectorAll(\'button\').forEach(b=>b.disabled=busy||loading||!item);$(\'prev\').disabled=busy||loading||!item||index===0;$(\'next\').disabled=busy||loading||!item||index===total-1;}\nfunction loadImage(element,path){return new Promise((resolve,reject)=>{element.onload=()=>resolve();element.onerror=()=>reject(Error(\'Не удалось открыть кадр; ответ не сохраняй.\'));element.src=\'/\'+path;});}\nasync function displayFrame(frame){await Promise.all([loadImage($(\'base\'),frame.views[0]),loadImage($(\'wrist\'),frame.views[1])]);$(\'shown\').textContent=`Показан шаг ${frame.step}. Текущий запрос: шаг ${item.step}.`;document.querySelectorAll(\'#timeline button\').forEach(b=>b.classList.toggle(\'selected\',Number(b.dataset.step)===frame.step));}\nasync function view(frame){if(loading||busy)return;loading=true;controls();try{await displayFrame(frame);}catch(e){item=null;throw e;}finally{loading=false;controls();}}\nasync function show(i){if(loading)return;loading=true;controls();try{const next=Math.max(0,Math.min(total-1,i));const data=await api(\'/api/item?index=\'+next);index=next;item=data;$(\'position\').textContent=`${index+1} / ${total}`;$(\'meta\').textContent=`${item.purpose===\'detector_train\'?\'Обучение\':\'Проверка\'} · эпизод ${item.episode_alias} · запрос на шаге ${item.step}`;$(\'task\').textContent=item.instruction;$(\'note\').value=item.answer?.note||\'\';$(\'timeline\').replaceChildren();for(const frame of item.context){const b=document.createElement(\'button\');b.dataset.step=frame.step;const img=document.createElement(\'img\');img.src=\'/\'+frame.views[0];img.alt=\'Предыдущее наблюдение\';b.append(img,document.createTextNode(\'Шаг \'+frame.step));b.onclick=()=>view(frame).catch(e=>message(e.message,true));$(\'timeline\').append(b);}await displayFrame(item.context.at(-1));document.querySelectorAll(\'[data-label]\').forEach(b=>b.classList.toggle(\'selected\',!!item.answer&&(b.dataset.label===\'unknown\'?item.answer.help_required===null:Number(b.dataset.label)===item.answer.help_required)));}catch(e){item=null;throw e;}finally{loading=false;controls();}}\nasync function save(value){if(busy||loading||!item)return;const annotator=$(\'annotator\').value.trim();if(!annotator){message(\'Укажи имя annotator.\',true);$(\'annotator\').focus();return;}busy=true;controls();try{await api(\'/api/label\',{sample_id:item.sample_id,help_required:value,annotator,note:$(\'note\').value});localStorage.setItem(\'vla_annotator\',annotator);const s=await state();await show(index+1);message(Object.values(s.counts).every(c=>c.unreviewed===0)?\'Сохранено. Все observations просмотрены; окно можно закрыть.\':\'Сохранено.\');}catch(e){message(e.message,true);}finally{busy=false;controls();}}\ndocument.querySelectorAll(\'[data-label]\').forEach(b=>b.onclick=()=>save(b.dataset.label===\'unknown\'?null:Number(b.dataset.label)));\n$(\'prev\').onclick=()=>show(index-1).catch(e=>message(e.message,true));$(\'next\').onclick=()=>show(index+1).catch(e=>message(e.message,true));\ndocument.addEventListener(\'keydown\',e=>{if(busy||loading||!item||[\'INPUT\',\'TEXTAREA\'].includes(document.activeElement.tagName))return;if(e.key===\'0\'||e.key===\'1\'){e.preventDefault();save(Number(e.key));}else if(e.key.toLowerCase()===\'u\'){e.preventDefault();save(null);}else if(e.key===\'ArrowLeft\'||e.key===\'ArrowRight\'){e.preventDefault();show(index+(e.key===\'ArrowLeft\'?-1:1)).catch(x=>message(x.message,true));}});\ncontrols();(async()=>{const s=await state();await show(s.first_unreviewed);})().catch(e=>message(e.message,true));\n</script></html>'

def handler_for(review):

    class Handler(BaseHTTPRequestHandler):

        def log_message(self, format, *args):
            pass

        def send(self, code, payload, content_type='application/json; charset=utf-8'):
            data = payload if isinstance(payload, bytes) else json.dumps(payload, ensure_ascii=False).encode()
            self.send_response(code)
            self.send_header('Content-Type', content_type)
            self.send_header('Content-Length', str(len(data)))
            self.send_header('Cache-Control', 'no-store')
            self.send_header('X-Content-Type-Options', 'nosniff')
            self.end_headers()
            self.wfile.write(data)

        def local_request(self):
            allowed = {f'127.0.0.1:{self.server.server_port}', f'localhost:{self.server.server_port}'}
            require(self.headers.get('Host') in allowed, 'Только localhost.')
            origin = self.headers.get('Origin')
            require(origin is None or origin in {'http://' + host for host in allowed}, 'Другой Origin запрещён.')

        def do_GET(self):
            try:
                self.local_request()
                parsed = urlsplit(self.path)
                if parsed.path == '/':
                    self.send(200, HTML.replace('__TOKEN__', review.token).encode(), 'text/html; charset=utf-8')
                elif parsed.path == '/api/state':
                    with review.lock:
                        self.send(200, review.stats())
                elif parsed.path == '/api/item':
                    index = int(parse_qs(parsed.query)['index'][0])
                    with review.lock:
                        self.send(200, review.item(index))
                elif re.fullmatch('/assets/[0-9a-f]{64}\\.png', parsed.path):
                    path = (review.output / parsed.path.lstrip('/')).resolve()
                    require(path.is_relative_to((review.output / 'assets').resolve()), 'Некорректный asset path.')
                    self.send(200, path.read_bytes(), 'image/png')
                else:
                    self.send(404, {'error': 'Не найдено'})
            except (ValueError, KeyError, OSError) as error:
                self.send(400, {'error': str(error)})

        def do_POST(self):
            try:
                self.local_request()
                require(self.path == '/api/label' and self.headers.get('X-Annotation-Token') == review.token, 'Некорректный запрос сохранения.')
                size = int(self.headers.get('Content-Length', '0'))
                require(0 < size <= 8192, 'Некорректный размер запроса.')
                data = json.loads(self.rfile.read(size))
                require(isinstance(data, dict), 'Нужен JSON object.')
                review.save(data)
                self.send(200, {'saved': True})
            except (ValueError, KeyError, OSError) as error:
                self.send(400, {'error': str(error)})
    return Handler

def serve(output, port=8765, open_browser=True):
    review = Review(output)
    with ThreadingHTTPServer(('127.0.0.1', port), handler_for(review)) as server:
        url = f'http://127.0.0.1:{server.server_port}'
        print('\nReview:', url, '\nLabels:', output / 'labels.json', flush=True)
        print('Labels: 0=continue, 1=help, U=unknown. Ctrl+C stops the server.', flush=True)
        if open_browser:
            webbrowser.open(url)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            print('\nLabels saved.')

def export(output):
    review = Review(output)
    rows = []
    for sample in review.samples:
        answer = review.labels['answers'].get(sample['sample_id'])
        if answer is None or answer['help_required'] is None:
            continue
        for source in sample['sources']:
            require(sha256(source['features_path']) == source['features_sha256'], 'Изменились исходные features.')
            require(sha256(Path(source['query']) / 'start.npz') == source['snapshot_sha256'], 'Изменились исходные observations.')
        rows.append({'sample_id': sample['sample_id'], 'purpose': sample['purpose'], 'init_id': sample['init_id'], 'step': sample['step'], 'sources': sample['sources'], **answer})
    require(bool(rows), 'Пока нет размеченных ответов 0/1.')
    path = output / ('reviewed_labels_' + timestamp() + '.json')
    write_json(path, {'schema_version': 1, 'label_question': QUESTION, 'manifest_sha256': review.labels['manifest_sha256'], 'policy_protocol': review.manifest['policy_protocol'], 'samples': rows, 'counts': review.stats()['counts'], 'scope': 'Human pre-action would-intervene labels; simulation adaptation of INSIGHT Strong'})
    print('Export:', path)
    print(json.dumps(review.stats()['counts'], ensure_ascii=False, indent=2))

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('command', nargs='?', choices=('prepare', 'serve', 'export'), default='serve')
    parser.add_argument('--runs', nargs='+', help='Optional explicit failures_* collections')
    parser.add_argument('--workspace', help='Annotation workspace; default: last saved workspace')
    parser.add_argument('--port', type=int, default=8765)
    parser.add_argument('--no-open', action='store_true')
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    if args.command == 'prepare':
        prepare(root, args.runs, args.workspace)
    elif args.command == 'export':
        export(resolve_workspace(root, args.workspace))
    else:
        if args.runs or (not args.workspace and (not (root / POINTER).is_file())):
            output = prepare(root, args.runs, args.workspace)
        else:
            output = resolve_workspace(root, args.workspace)
        serve(output, args.port, not args.no_open)
if __name__ == '__main__':
    try:
        main()
    except (Exception, KeyboardInterrupt) as error:
        print('\nSTOP:', error, file=sys.stderr)
        sys.exit(1)
