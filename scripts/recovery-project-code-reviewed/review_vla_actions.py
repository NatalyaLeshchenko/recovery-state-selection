#!/usr/bin/env python3
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import sys
import webbrowser
import numpy as np
from PIL import Image
import label_vla_help as base
POINTER = 'pilot_runs/vla_progress_annotation_path.txt'
QUESTION = "Does this executed VLA action prefix fail to contribute to the instructed task's progress?"
PROTOCOL = 'action_progress_executed_prefix_v1'
PIXELS = ('observation/pixels/image', 'observation/pixels/image2')

def old_workspace(root, value=None):
    return base.resolve_workspace(root, value)

def load_old(output):
    manifest = base.read_json(output / 'manifest.json')
    labels = base.read_json(output / 'labels.json')
    base.require(labels['manifest_sha256'] == base.sha256(output / 'manifest.json'), 'Исходный manifest изменился после разметки.')
    base.require(manifest.get('schema_version') == 1 and 'label_protocol' not in manifest, 'Нужна исходная папка help_annotation_*, не новый progress review.')
    return (manifest, labels)

def audit_old(output):
    manifest, labels = load_old(output)
    result, positives = ({}, [])
    for role in base.ROLES:
        counts = Counter({'continue': 0, 'help': 0, 'unknown': 0, 'unreviewed': 0})
        groups = set()
        for sample in manifest['samples']:
            if sample['purpose'] != role:
                continue
            answer = labels['answers'].get(sample['sample_id'])
            key = 'unreviewed' if answer is None else 'unknown' if answer['help_required'] is None else 'help' if answer['help_required'] == 1 else 'continue'
            counts[key] += 1
            if key == 'help':
                groups.add(sample['init_id'])
                source_items_1 = sample['sources']
                items_1 = []
                for s in source_items_1:
                    items_1.append(s['query'])
                positives.append({'purpose': role, 'init_id': sample['init_id'], 'step': sample['step'], 'sample_id': sample['sample_id'], 'note': answer.get('note', ''), 'queries': items_1})
        labeled = counts['continue'] + counts['help']
        result[role] = {**dict(counts), 'binary_samples': labeled, 'always_continue_accuracy': counts['continue'] / labeled if labeled else None, 'positive_init_ids': sorted(groups), 'positive_recall_of_always_continue': 0.0 if counts['help'] else None}
    return {'source_workspace': str(output), 'label_protocol': 'legacy_would_intervene', 'counts': result, 'positive_samples': positives, 'classifier_trained': False, 'note': 'Pilot labels use the previous subjective question; accuracy alone is misleading.'}

def source_query(source):
    run = Path(source['run']).resolve()
    query = Path(source['query']).resolve()
    base.require(query.is_relative_to(run), 'Query выходит за collection.')
    base.require(base.sha256(query / 'start.npz') == source['snapshot_sha256'], 'Изменился snapshot.')
    base.require(Path(source['features_path']).resolve() == query / 'token_features.npz', 'Некорректный features path.')
    base.require(base.sha256(query / 'token_features.npz') == source['features_sha256'], 'Изменились token features.')
    return query

def case_data(folder):
    record = base.read_json(folder / 'case.json')
    path = folder / 'rollout.npz'
    trajectory_path = folder / 'trajectory.json'
    if not path.is_file() or not trajectory_path.is_file():
        return None
    with np.load(path, allow_pickle=False) as data:
        actions = data['action'].copy()
        source_items_2 = PIXELS
        images = []
        for key in source_items_2:
            images.append(data[key].copy())
    base.require(actions.ndim == 2 and actions.shape[1] == 7 and np.isfinite(actions).all(), 'Некорректный action rollout.')
    for image in images:
        base.require(image.dtype == np.uint8 and image.ndim == 5 and (image.shape[0] == len(actions)) and (image.shape[1] == 1) and (image.shape[-1] == 3), 'Некорректные rollout cameras.')
    trajectory = base.read_json(trajectory_path)
    base.require(len(trajectory) == len(actions), 'Trajectory и rollout имеют разную длину.')
    source_items_3 = (folder / 'case.json', path, trajectory_path)
    items_3 = {}
    for p in source_items_3:
        items_3[str(p)] = base.sha256(p)
    return {'record': record, 'actions': actions, 'images': images, 'trajectory': trajectory, 'digests': items_3}

def action_preview(output, query, boundary, data):
    info = base.read_json(query / 'inference.json')
    count = boundary.get('executed_actions', 0)
    if info.get('status') != 'ok':
        return (None, 'undecodable_prediction_not_a_human_help_label')
    if count == 0:
        return (None, 'no_executed_prefix')
    if data is None:
        return (None, 'recorded_motion_missing')
    base.require(type(count) is int and 0 < count <= 5, 'Некорректный executed prefix.')
    with np.load(query / 'token_features.npz', allow_pickle=False) as saved:
        predicted = saved['actions'].astype(np.float32)
    base.require(predicted.shape == (10, 7) and np.isfinite(predicted).all(), 'Некорректная VLA prediction.')
    step = boundary['step']
    base.require(0 <= step < len(data['actions']) and step + count <= len(data['actions']), 'Action prefix выходит за пределы rollout.')
    actual = np.clip(predicted[:count], -1, 1)
    base.require(np.allclose(actual, data['actions'][step:step + count], rtol=0, atol=1e-07), 'Recorded actions отличаются от VLA prediction.')
    for index, row in enumerate(data['trajectory'][step:step + count]):
        base.require(row['step'] == step + index and row['inference_step'] == step and (row['chunk_index'] == index) and (row['phase'] in ('nominal', 'recovery')), 'В preview попало вмешательство или другое действие VLA.')
    starting = base.snapshot_images(query, boundary.get('snapshot_sha256'))
    for i in range(2):
        base.require(np.array_equal(starting[i], data['images'][i][step, 0, ::-1, ::-1]), 'Preview начинается с другого observation.')
    frames = []
    for offset in range(count + 1):
        position = step + offset
        if position < len(data['actions']):
            source_items_4 = data['images']
            views = []
            for image in source_items_4:
                views.append(np.ascontiguousarray(image[position, 0, ::-1, ::-1]))
        else:
            with np.load(query.parent.parent / 'final.npz', allow_pickle=False) as final:
                source_items_5 = PIXELS
                views = []
                for key in source_items_5:
                    views.append(np.ascontiguousarray(final[key][0, ::-1, ::-1]))
            data['digests'][str(query.parent.parent / 'final.npz')] = base.sha256(query.parent.parent / 'final.npz')
        source_items_6 = views
        items_6 = []
        for image in source_items_6:
            items_6.append(base.save_image(output, image))
        frames.append({'step': position, 'views': items_6})
    return ({'executed_actions': count, 'prediction_length': len(predicted), 'frames': frames, 'commands': actual.tolist()}, None)

def prepare(root, source_output):
    manifest, labels = load_old(source_output)
    output = root / 'pilot_runs/vla_insight' / ('progress_review_' + base.timestamp())
    output.mkdir(parents=True, exist_ok=False)
    (output / 'assets').mkdir()
    grouped = defaultdict(list)
    for sample in manifest['samples']:
        base.require(type(sample['init_id']) is int and sample['purpose'] in base.ROLES and (sample['init_id'] in base.ROLES[sample['purpose']]), 'Нарушен фиксированный detector split.')
        source_items_7 = sample['context']
        steps = []
        for frame in source_items_7:
            steps.append(frame['step'])
        base.require(1 <= len(steps) <= 5 and all((type(step) is int for step in steps)) and (steps == sorted(set(steps))) and (steps[-1] == sample['step']) and all((max(0, sample['step'] - 40) <= step <= sample['step'] for step in steps)), 'В context попал будущий кадр или неверная история.')
        for source in sample['sources']:
            query = source_query(source)
            grouped[query.parent.parent].append((sample, source, query))
    samples, excluded, dedup = ([], [], {})
    context_cache = {}
    for folder, members in sorted(grouped.items()):
        data = case_data(folder)
        record = data['record'] if data is not None else base.read_json(folder / 'case.json')
        source_items_8 = record['inferences']
        boundaries = {}
        for b in source_items_8:
            boundaries[b['step']] = b
        for old, source, query in members:
            base.require(record['purpose'] == old['purpose'] and record['init_state_id'] == old['init_id'], 'Case и исходная разметка относятся к разным starts.')
            boundary = boundaries[old['step']]
            base.require(base.checked_path(Path(source['run']), boundary['query']) == query, 'Boundary относится к другому query.')
            features, reason = base.classifier_features(query, boundary)
            if reason:
                excluded.append({'query': str(query), 'reason': reason})
                continue
            preview, reason = action_preview(output, query, boundary, data)
            if reason:
                excluded.append({'query': str(query), 'reason': reason})
                continue
            context = []
            for frame in old['context']:
                views = []
                for relative in frame['views']:
                    if relative not in context_cache:
                        original = (source_output / relative).resolve()
                        base.require(original.is_relative_to(source_output.resolve()), 'Некорректный context path.')
                        with Image.open(original) as image:
                            pixels = np.asarray(image.convert('RGB'))
                        context_cache[relative] = base.save_image(output, pixels)
                    views.append(context_cache[relative])
                context.append({'step': frame['step'], 'views': views})
            base.require(context[-1] == preview['frames'][0], 'Исходный context и начало действия различаются.')
            identity = hashlib.sha256(json.dumps({'label_protocol': PROTOCOL, 'role': old['purpose'], 'init_id': old['init_id'], 'instruction': old['instruction'], 'context': context, 'preview': preview}, sort_keys=True).encode() + features.tobytes()).hexdigest()
            provenance = {**source, 'case_digests': data['digests'].copy()}
            if identity in dedup:
                dedup[identity]['sources'].append(provenance)
                if old['sample_id'] not in dedup[identity]['supersedes_sample_ids']:
                    dedup[identity]['supersedes_sample_ids'].append(old['sample_id'])
                continue
            sample = {'sample_id': identity, 'purpose': old['purpose'], 'init_id': old['init_id'], 'instruction': old['instruction'], 'step': old['step'], 'context': context, 'episode_alias': old['episode_alias'], 'action_preview': preview, 'sources': [provenance], 'supersedes_sample_ids': [old['sample_id']]}
            samples.append(sample)
            dedup[identity] = sample
        print(f'Prepared {folder.name}: observations {len(samples)}', flush=True)
        del data
    base.require(bool(samples), 'Нет подтверждённых записей движения VLA для просмотра.')
    ordered = []
    for role in base.ROLES:
        source_items_9 = samples
        group = []
        for s in source_items_9:
            if s['purpose'] == role:
                group.append(s)
        random.Random(0).shuffle(group)
        ordered.extend(group)
    new_manifest = {'schema_version': 2, 'label_protocol': PROTOCOL, 'label_question': QUESTION, 'created_at': datetime.now(timezone.utc).isoformat(), 'policy_protocol': manifest['policy_protocol'], 'source_workspace': str(source_output), 'source_manifest_sha256': base.sha256(source_output / 'manifest.json'), 'source_labels_sha256': base.sha256(source_output / 'labels.json'), 'samples': ordered, 'hidden_from_reviewer': ['old_answers', 'episode_outcome', 'failure_type', 'uncertainty', 'decoder_status'], 'scope': 'Simulation adaptation: progress of executed prefix; not complete ten-action counterfactual', 'preview_protocol': 'Only verified actions from this inference; no forced interventions or later VLA queries'}
    base.write_json(output / 'manifest.json', new_manifest)
    base.write_json(output / 'excluded_queries.json', excluded)
    base.write_json(output / 'legacy_label_audit.json', audit_old(source_output))
    base.write_json(output / 'labels.json', {'schema_version': 2, 'label_protocol': PROTOCOL, 'manifest_sha256': base.sha256(output / 'manifest.json'), 'label_question': QUESTION, 'answers': {}})
    pointer = root / POINTER
    pointer.parent.mkdir(parents=True, exist_ok=True)
    pointer.write_text(str(output) + '\n', encoding='utf-8')
    print('\nReview:', output)
    print('Review samples:', dict(Counter((s['purpose'] for s in ordered))))
    return output

class ProgressReview(base.Review):

    def __init__(self, output):
        super().__init__(output)
        base.require(self.manifest.get('label_protocol') == PROTOCOL and self.labels.get('label_protocol') == PROTOCOL, 'Нужна новая папка progress_review_*.')

    def item(self, index):
        return {**super().item(index), 'action_preview': self.samples[index]['action_preview']}
HTML = base.HTML.replace('VLA · Help labels', 'VLA · Task progress')
HTML = HTML.replace('Ты оператор. Попросила бы ты expert вмешаться прямо сейчас?', 'Помогает ли показанное действие VLA выполнить задачу?')
HTML = HTML.replace('Оцени задачу и кадры до текущего запроса. Падение или промах сами по себе не задают ответ. Если по кадрам нельзя решить, выбери «Не уверена». Последующие кадры и исход эпизода здесь не показываются.', 'Оцени наблюдение и короткое движение из одного запроса VLA. 0 — действие продвигает задачу; 1 — действие ошибочное или не способствует прогрессу; U — не хватает информации. Подготовка захвата и его закрывание тоже могут быть полезными действиями. Возможный будущий успех не определяет метку. Исход эпизода скрыт.')
HTML = HTML.replace('0 · Продолжить самостоятельно', '0 · Действие полезно')
HTML = HTML.replace('1 · Expert сейчас', '1 · Нет прогресса / ошибка')
HTML = HTML.replace('<div class="card"><div class="fields">', '<div class="card"><b>Движение из текущего запроса VLA</b><p id="clip-description" class="muted"></p><div class="views"><div id="clip-base"></div><div id="clip-wrist"></div></div><div class="nav" style="margin-top:12px"><button id="clip-play">Воспроизвести</button><span id="clip-position"></span><input id="clip-slider" type="range" min="0" value="0" style="width:45%"></div><details style="margin-top:12px"><summary>Команды для этого движения</summary><div id="commands" style="overflow:auto"></div></details></div><div class="card"><div class="fields">')
HTML = HTML.replace("await displayFrame(item.context.at(-1));document.querySelectorAll('[data-label]')", "await displayFrame(item.context.at(-1));await renderAction(item);document.querySelectorAll('[data-label]')")
HTML = HTML.replace('if(busy||loading||!item)return;const annotator=', 'if(busy||loading||!item||!clipDone)return;const annotator=')
HTML = HTML.replace('controls();(async()=>', "\nlet clipImages=[],clipFrames=[],clipIndex=0,clipGeneration=0,playing=false,clipDone=false,seenFrames=new Set();\nfunction controls(){document.querySelectorAll('button').forEach(b=>b.disabled=busy||loading||!item||(b.hasAttribute('data-label')&&!clipDone));$('prev').disabled=busy||loading||!item||index===0;$('next').disabled=busy||loading||!item||index===total-1;}\nfunction clipFrame(i){clipIndex=i;$('clip-base').replaceChildren(clipImages[i][0]);$('clip-wrist').replaceChildren(clipImages[i][1]);$('clip-slider').value=i;$('clip-position').textContent=`Шаг ${clipFrames[i].step}: ${i} / ${clipFrames.length-1} действий`;seenFrames.add(i);clipDone=seenFrames.size===clipFrames.length;controls();}\nasync function renderAction(data){clipGeneration++;playing=false;clipDone=false;seenFrames=new Set();clipFrames=data.action_preview.frames;$('clip-play').textContent='Воспроизвести';$('clip-slider').max=clipFrames.length-1;$('clip-description').textContent=`Показаны ${data.action_preview.executed_actions} выполненных действий из ${data.action_preview.prediction_length} предсказанных. Сначала воспроизведи движение, затем выбери метку. Скорость замедлена примерно в 3 раза. Вмешательство и следующие запросы сюда не входят.`;clipImages=await Promise.all(clipFrames.map(f=>Promise.all(f.views.map(path=>new Promise((resolve,reject)=>{const img=new Image();img.alt='Наблюдение во время текущего действия';img.onload=()=>resolve(img);img.onerror=()=>reject(Error('Не удалось загрузить движение VLA.'));img.src='/'+path;})))));clipFrame(0);const table=document.createElement('table');table.style.cssText='width:100%;font-size:13px;border-collapse:collapse';const head=document.createElement('tr');for(const name of ['Действие','dx','dy','dz','rx','ry','rz','gripper']){const th=document.createElement('th');th.textContent=name;th.style.padding='6px';head.append(th);}table.append(head);data.action_preview.commands.forEach((command,i)=>{const row=document.createElement('tr');[String(i+1),...command.map(x=>x.toFixed(3))].forEach(value=>{const td=document.createElement('td');td.textContent=value;td.style.cssText='padding:6px;text-align:right;border-top:1px solid #ddd';row.append(td);});table.append(row);});$('commands').replaceChildren(table);}\n$('clip-slider').oninput=()=>{clipGeneration++;playing=false;$('clip-play').textContent='Воспроизвести';clipFrame(Number($('clip-slider').value));};\n$('clip-play').onclick=()=>{clipGeneration++;if(playing){playing=false;$('clip-play').textContent='Воспроизвести';return;}playing=true;const generation=clipGeneration;$('clip-play').textContent='Пауза';let i=0;const advance=()=>{if(generation!==clipGeneration||!playing)return;clipFrame(i);if(i===clipFrames.length-1){playing=false;$('clip-play').textContent='Воспроизвести';return;}i++;setTimeout(advance,i===1?350:150);};advance();};\ncontrols();(async()=>")

def serve(output, port=8766, open_browser=True):
    review = ProgressReview(output)
    base.HTML = HTML
    with base.ThreadingHTTPServer(('127.0.0.1', port), base.handler_for(review)) as server:
        url = f'http://127.0.0.1:{server.server_port}'
        print('\nReview:', url, '\nLabels:', output / 'labels.json', flush=True)
        print('Labels: 0=progress, 1=error/no progress, U=unknown. Ctrl+C stops the server.', flush=True)
        if open_browser:
            webbrowser.open(url)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            print('\nLabels saved.')

def resolve_new(root, value=None):
    output = Path(value).expanduser().resolve() if value else Path((root / POINTER).read_text().strip())
    base.require((output / 'manifest.json').is_file(), 'Не найдена новая папка progress_review_*.')
    return output

def export(output):
    review = ProgressReview(output)
    rows, checked = ([], {})
    for sample in review.samples:
        answer = review.labels['answers'].get(sample['sample_id'])
        if answer is None or answer['help_required'] is None:
            continue
        for source in sample['sources']:
            source_query(source)
            for path, digest in source['case_digests'].items():
                if path not in checked:
                    checked[path] = base.sha256(path)
                base.require(checked[path] == digest, 'Изменилась запись движения VLA.')
        rows.append({**sample, **answer})
    base.require(bool(rows), 'Пока нет новых ответов 0/1.')
    path = output / ('reviewed_progress_labels_' + base.timestamp() + '.json')
    base.write_json(path, {'schema_version': 2, 'label_protocol': PROTOCOL, 'label_question': QUESTION, 'manifest_sha256': review.labels['manifest_sha256'], 'policy_protocol': review.manifest['policy_protocol'], 'scope': review.manifest['scope'], 'samples': rows, 'counts': review.stats()['counts'], 'classifier_trained': False})
    print('Export:', path)
    print(json.dumps(review.stats()['counts'], ensure_ascii=False, indent=2))

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('command', nargs='?', choices=('audit', 'prepare', 'serve', 'export'), default='serve')
    parser.add_argument('--source', help='Old help_annotation_* folder; defaults to the old saved pointer')
    parser.add_argument('--workspace', help='New progress_review_* folder')
    parser.add_argument('--port', type=int, default=8766)
    parser.add_argument('--no-open', action='store_true')
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    if args.command == 'audit':
        print(json.dumps(audit_old(old_workspace(root, args.source)), ensure_ascii=False, indent=2))
    elif args.command == 'prepare':
        prepare(root, old_workspace(root, args.source))
    elif args.command == 'export':
        export(resolve_new(root, args.workspace))
    else:
        output = prepare(root, old_workspace(root, args.source)) if args.source or (not args.workspace and (not (root / POINTER).is_file())) else resolve_new(root, args.workspace)
        serve(output, args.port, not args.no_open)
if __name__ == '__main__':
    try:
        main()
    except (Exception, KeyboardInterrupt) as error:
        print('\nSTOP:', error, file=sys.stderr)
        sys.exit(1)
