#!/usr/bin/env python3
from __future__ import annotations
from collections import Counter
import csv
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import random
import sys
import numpy as np
import insight_pipeline as protocol
AUTHOR_SOURCE_SHA256 = '3b08cb4ade9e47653193008e2dd4394532718916e8cec431eb84cbe846843318'

def fingerprint(rows):

    def sample_identifier(r):
        return r['sample_id']
    source_items_1 = sorted(rows, key=sample_identifier)
    payload = []
    for r in source_items_1:
        payload.append({'sample_id': r['sample_id'], 'purpose': r['purpose'], 'init_id': r['init_id'], 'label': r['label'], 'features_sha256': hashlib.sha256(r['features'].tobytes()).hexdigest()})
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()

def load_rows(root, workspace, roles):
    path, manifest, labels = protocol.review_state(root, workspace)
    _, frozen = protocol.frozen_protocol(root)
    protocol.require(manifest['policy_protocol'] == frozen['policy_protocol'], 'VLA/feature protocol отличается от freeze.')
    import label_vla_help as helper
    rows, counts = ([], {})
    checksums = {}

    def checked_digest(filename):
        filename = Path(filename).resolve()
        if filename not in checksums:
            checksums[filename] = protocol.digest(filename)
        return checksums[filename]
    for role in roles:
        source_items_2 = manifest['samples']
        samples = []
        for s in source_items_2:
            if s['purpose'] == role:
                samples.append(s)
        counts[role] = protocol.label_counts(samples, labels['answers'])
        protocol.require(samples and counts[role]['unreviewed'] == 0, f'{role}: просмотри все выбранные queries; unknown разрешены и исключаются.')
        for sample in samples:
            answer = labels['answers'][sample['sample_id']]
            if answer['help_required'] is None:
                continue
            protocol.require(answer.get('reviewed') is True, 'Нет подтверждения human review.')
            protocol.require(sample['profiles'] == [frozen['profile']], 'После freeze нужны данные только выбранного профиля.')
            arrays = []
            for source in sample['sources']:
                run = protocol.read(Path(source['run']) / 'run_info.json')
                protocol.require(run['profile'] == frozen['profile'] and run['purpose'] == role, 'Исходный collection относится к другому профилю или split.')
                query = Path(source['query'])
                protocol.require(checked_digest(query / 'start.npz') == source['snapshot_sha256'] and checked_digest(query / 'token_features.npz') == source['features_sha256'], 'Изменились observations/features после human review.')
                for recorded, checksum in source['case_digests'].items():
                    protocol.require(checked_digest(recorded) == checksum, 'Изменился action preview source.')
                features, reason = helper.classifier_features(query, {})
                protocol.require(reason is None, f'Не подходят Strong features: {reason}')
                arrays.append(features)
            protocol.require(arrays and all((np.array_equal(arrays[0], a) for a in arrays)), 'Deduplicated sources различаются.')
            rows.append({'sample_id': sample['sample_id'], 'purpose': role, 'init_id': sample['init_id'], 'step': sample['step'], 'label': answer['help_required'], 'features': arrays[0]})
    groups = {role: {r['init_id'] for r in rows if r['purpose'] == role} for role in roles}
    for i, left in enumerate(roles):
        for right in roles[i + 1:]:
            protocol.require(not groups[left] & groups[right], 'Train/validation/test init IDs пересекаются.')
    return (path, manifest, rows, counts)

def metrics(labels, scores, threshold):
    y = np.asarray(labels, dtype=np.int64)
    values = np.asarray(scores, dtype=np.float64)
    protocol.require(y.ndim == values.ndim == 1 and y.shape == values.shape and (len(y) > 0) and np.isin(y, [0, 1]).all() and np.isfinite(values).all() and math.isfinite(float(threshold)), 'Некорректные labels, scores или threshold.')
    predicted = values >= threshold
    tp = int(np.sum(predicted & (y == 1)))
    fp = int(np.sum(predicted & (y == 0)))
    fn = int(np.sum(~predicted & (y == 1)))
    tn = int(np.sum(~predicted & (y == 0)))
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    specificity = tn / (tn + fp) if tn + fp else None
    f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None
    return {'n': len(y), 'n_help': tp + fn, 'n_continue': tn + fp, 'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn, 'accuracy': (tp + tn) / len(y), 'precision': precision, 'recall': recall, 'f1': f1, 'balanced_accuracy': (recall + specificity) / 2 if recall is not None and specificity is not None else None, 'false_ask_rate': fp / (tn + fp) if tn + fp else None, 'miss_help_rate': fn / (tp + fn) if tp + fn else None, 'trigger_rate': (tp + fp) / len(y)}

def fit_threshold(labels, scores):
    y = np.asarray(labels)
    values = np.asarray(scores, dtype=np.float64)
    protocol.require(set(y.tolist()) == {0, 1}, 'Для threshold calibration нужны оба класса validation.')
    unique = np.unique(values)
    protocol.require(np.isfinite(unique).all(), 'Не конечные validation scores.')
    candidates = [float(np.nextafter(unique[0], -np.inf)), float(np.nextafter(unique[-1], np.inf))]
    source_items_3 = zip(unique[:-1], unique[1:])
    items_3 = []
    for a, b in source_items_3:
        items_3.append(float(a / 2 + b / 2))
    candidates += items_3
    best = None
    for threshold in candidates:
        m = metrics(y, values, threshold)
        key = (m['balanced_accuracy'], m['f1'] or 0, -m['trigger_rate'], threshold)
        if best is None or key > best[0]:
            best = (key, threshold, m)
    return {'threshold': best[1], 'validation_metrics': best[2], 'rule': 'maximize validation balanced accuracy; tie: F1, lower trigger rate, larger threshold'}

def check_coverage(rows, role, *, training=False):
    source_items_4 = rows
    group = []
    for r in source_items_4:
        if r['purpose'] == role:
            group.append(r)
    counts = Counter((r['label'] for r in group))
    protocol.require(set(counts) == {0, 1}, f'{role}: нужны human labels обоих классов; сейчас {dict(counts)}.')
    source_items_5 = group
    items_5 = set()
    for r in source_items_5:
        items_5.add(r['init_id'])
    protocol.require(len(items_5) >= 2, f'{role}: нужны хотя бы два разных init IDs.')
    if training:
        protocol.require(counts[1] >= 5 and counts[0] >= 10 and (len({r['init_id'] for r in group if r['label'] == 1}) >= 2), 'Train pilot: минимум 5 help на двух init IDs и 10 continue; это gate пригодности, не гарантия качества.')

def torch_model(root):
    try:
        import torch
    except ImportError as error:
        raise RuntimeError('Запусти в основной .venv с установленным PyTorch. GPU для classifier не нужен.') from error
    source = root / 'vla_insight/insight/insight/models/single_transformer.py'
    protocol.require(source.is_file(), 'Нет pinned author SingleStepTransformer; нужен существующий start_insight.py prepare.')
    protocol.require(protocol.digest(source) == AUTHOR_SOURCE_SHA256, 'Author SingleStepTransformer отличается от проверенного pinned source.')
    spec = importlib.util.spec_from_file_location('pilot_insight_author_transformer', source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    torch.set_num_threads(min(4, torch.get_num_threads()))
    model = module.SingleStepTransformer(d_in=4, d_h=64, nhead=4, nlayers=1).cpu()
    return (torch, model, source)

def batches(torch, rows, batch_size=16, order=None):
    indices = np.arange(len(rows)) if order is None else np.asarray(order)
    for lo in range(0, len(indices), batch_size):
        source_items_6 = indices[lo:lo + batch_size]
        group = []
        for i in source_items_6:
            group.append(rows[int(i)])
        max_tokens = max((len(r['features']) for r in group))
        x = torch.zeros((len(group), max_tokens, 4), dtype=torch.float32)
        pad = torch.ones((len(group), max_tokens), dtype=torch.bool)
        for i, row in enumerate(group):
            n = len(row['features'])
            x[i, :n] = torch.from_numpy(row['features'])
            pad[i, :n] = False
        source_items_7 = group
        items_7 = []
        for r in source_items_7:
            items_7.append(r['label'])
        yield (x, pad, torch.tensor(items_7, dtype=torch.float32))

def predict(torch, model, rows):
    model.eval()
    with torch.no_grad():
        source_items_8 = batches(torch, rows)
        values = []
        for x, pad, _ in source_items_8:
            values.append(model(x, pad).cpu().numpy())
    return np.concatenate(values).astype(np.float64)

def scalar_baselines(rows):
    source_items_9 = rows
    items_9 = []
    for r in source_items_9:
        items_9.append(float(r['features'][:, 2].mean()))
    source_items_10 = rows
    items_10 = []
    for r in source_items_10:
        items_10.append(-float(r['features'][:, 3].mean()))
    return {'entropy': np.array(items_9), 'mean_negative_log_probability': np.array(items_10)}

def train(root, args):
    protocol.require(1 <= args.epochs <= 1000, 'Число epochs должно быть 1..1000.')
    path, manifest, rows, counts = load_rows(root, args.workspace, ('detector_train', 'detector_validation'))
    for role in ('detector_train', 'detector_validation'):
        check_coverage(rows, role, training=role == 'detector_train')
    source_items_11 = rows
    tr = []
    for r in source_items_11:
        if r['purpose'] == 'detector_train':
            tr.append(r)
    source_items_12 = rows
    va = []
    for r in source_items_12:
        if r['purpose'] == 'detector_validation':
            va.append(r)
    torch, model, source = torch_model(root)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    _, model, _ = torch_model(root)
    out = root / 'pilot_runs/vla_insight' / ('strong_detector_' + protocol.stamp())
    out.mkdir(parents=True)
    frozen_path, frozen = protocol.frozen_protocol(root)
    record = {'status': 'training', 'architecture': 'author SingleStepTransformer (4,64,4,1)', 'author_source_sha256': protocol.digest(source), 'frozen_protocol_sha256': protocol.digest(frozen_path), 'profile': frozen['profile'], 'parameters': frozen['parameters'], 'policy_protocol': manifest['policy_protocol'], 'label_protocol': manifest['label_protocol'], 'sampling_max_queries_per_case': manifest['max_queries_per_case'], 'seed': args.seed, 'epochs_limit': args.epochs, 'device': 'cpu', 'torch_version': torch.__version__, 'feature_transform': 'none; existing trimmed raw AU/EU/entropy/chosen-token log probability', 'training_rows_sha256': fingerprint(tr), 'validation_rows_sha256': fingerprint(va), 'counts': counts, 'test_used_for_training': False, 'test_used_for_threshold': False, 'scope': 'Exploratory single-task simulation adaptation, not paper reproduction or policy adaptation'}
    protocol.write(out / 'train_info.json', record)
    positives = sum((r['label'] for r in tr))
    weight = (len(tr) - positives) / positives
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(weight))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    best = float('inf')
    wait = 0
    history = []
    generator = np.random.default_rng(args.seed)
    checkpoint = out / 'single_fold0.pt'
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0
        for x, pad, y in batches(torch, tr, order=generator.permutation(len(tr))):
            optimizer.zero_grad()
            loss = loss_fn(model(x, pad), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += float(loss.detach()) * len(y)
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x, pad, y in batches(torch, va):
                val_loss += float(loss_fn(model(x, pad), y)) * len(y)
        val_loss /= len(va)
        history.append({'epoch': epoch, 'train_weighted_bce': train_loss / len(tr), 'validation_weighted_bce': val_loss})
        print(f'epoch={epoch}/{args.epochs} train={train_loss / len(tr):.4f} validation={val_loss:.4f}', flush=True)
        if val_loss < best - 1e-06:
            best = val_loss
            wait = 0
            torch.save(model.state_dict(), checkpoint)
            record['best_epoch'] = epoch
        else:
            wait += 1
            if wait >= 10:
                break
    model.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=True))
    scores = predict(torch, model, va)
    source_items_13 = va
    y = []
    for r in source_items_13:
        y.append(r['label'])
    calibration = {'INSIGHT_Strong': fit_threshold(y, scores)}
    for name, values in scalar_baselines(va).items():
        calibration[name] = fit_threshold(y, values)
    record.update(status='trained', classifier_trained=True, policy_adapted=False, positive_weight=weight, threshold_logit=calibration['INSIGHT_Strong']['threshold'], checkpoint_sha256=protocol.digest(checkpoint))
    protocol.write(out / 'calibration.json', calibration)
    protocol.write(out / 'history.json', history)
    record['calibration_sha256'] = protocol.digest(out / 'calibration.json')
    protocol.write(out / 'train_info.json', record)
    protocol.write(out / 'deployment.json', {'checkpoint': str(checkpoint), 'device': 'cpu', 'threshold_logit': record['threshold_logit'], 'trim_head': 3, 'trim_tail': 2, 'feature_order': protocol.FEATURE_ORDER})
    protocol.set_pointer(root, protocol.MODEL_POINTER, out)
    print('\nHELP CLASSIFIER TRAINED:', out)

def evaluate(root, args):
    out = protocol.resolve(root, args.model) if args.model else protocol.pointer(root, protocol.MODEL_POINTER)
    record = protocol.read(out / 'train_info.json')
    calibration = protocol.read(out / 'calibration.json')
    protocol.require(record['status'] == 'trained', 'Нет завершённого classifier checkpoint.')
    protocol.require(protocol.digest(out / 'calibration.json') == record['calibration_sha256'], 'Threshold calibration изменена после fit.')
    frozen_path, _ = protocol.frozen_protocol(root)
    protocol.require(protocol.digest(frozen_path) == record['frozen_protocol_sha256'], 'Протокол изменён после fit.')
    _, manifest, rows, counts = load_rows(root, args.workspace, ('detector_train', 'detector_validation', 'detector_test'))
    source_items_14 = rows
    tr = []
    for r in source_items_14:
        if r['purpose'] == 'detector_train':
            tr.append(r)
    source_items_15 = rows
    va = []
    for r in source_items_15:
        if r['purpose'] == 'detector_validation':
            va.append(r)
    source_items_16 = rows
    te = []
    for r in source_items_16:
        if r['purpose'] == 'detector_test':
            te.append(r)
    protocol.require(fingerprint(tr) == record['training_rows_sha256'] and fingerprint(va) == record['validation_rows_sha256'], 'Train/validation labels или выборка изменились после fit. Этот test не относится к сохранённому fit.')
    check_coverage(te, 'detector_test')
    protocol.require(manifest['policy_protocol'] == record['policy_protocol'], 'Изменился VLA protocol.')
    protocol.require(protocol.digest(out / 'single_fold0.pt') == record['checkpoint_sha256'], 'Изменился checkpoint.')
    torch, model, source = torch_model(root)
    protocol.require(protocol.digest(source) == record['author_source_sha256'], 'Изменилась author architecture.')
    model.load_state_dict(torch.load(out / 'single_fold0.pt', map_location='cpu', weights_only=True))
    values = {'INSIGHT_Strong': predict(torch, model, te), **scalar_baselines(te)}
    source_items_17 = te
    y = []
    for r in source_items_17:
        y.append(r['label'])
    source_items_18 = values
    thresholds = {}
    for name in source_items_18:
        thresholds[name] = calibration[name]['threshold']
    source_items_19 = values.items()
    items_19 = {}
    for name, v in source_items_19:
        items_19[name] = metrics(y, v, thresholds[name])
    source_items_20 = te
    items_20 = set()
    for r in source_items_20:
        items_20.add(r['init_id'])
    report = {'models': items_19, 'always_continue': metrics(y, np.zeros(len(te)), 1.0), 'n_init_ids': len(items_20), 'test_counts': counts['detector_test'], 'test_rows_sha256': fingerprint(te), 'thresholds': thresholds, 'threshold_fit_on': 'detector_validation only', 'classifier_checkpoint_sha256': record['checkpoint_sha256'], 'policy_success_measured': False, 'selectors_compared': False, 'valid_feature_queries_only': True, 'note': 'Sparse human action-progress labels from valid token-feature queries; unknown labels and invalid sequences are excluded. Queries within an init are correlated; this is not final VLA policy evaluation or OOD testing.'}
    per_init = {}
    source_items_21 = te
    items_21 = set()
    for r in source_items_21:
        items_21.add(r['init_id'])
    for init_id in sorted(items_21):
        source_items_22 = enumerate(te)
        indices = []
        for i, r in source_items_22:
            if r['init_id'] == init_id:
                indices.append(i)
        source_items_23 = values.items()
        items_23 = {}
        for name, v in source_items_23:
            items_23[name] = metrics(np.asarray(y)[indices], v[indices], thresholds[name])
        per_init[str(init_id)] = items_23
    report['per_init_id'] = per_init
    result = root / 'pilot_runs/vla_insight' / ('strong_test_' + protocol.stamp())
    result.mkdir(parents=True)
    protocol.write(result / 'summary.json', report)
    with (result / 'predictions.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.writer(stream)
        writer.writerow(['sample_id', 'init_id', 'step', 'human_label', *values])
        for i, r in enumerate(te):
            source_items_24 = values.values()
            items_24 = []
            for v in source_items_24:
                items_24.append(v[i])
            writer.writerow([r['sample_id'], r['init_id'], r['step'], r['label'], *items_24])
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(9, 4.5), constrained_layout=True)
        names = list(report['models']) + ['Always continue']
        results = list(report['models'].values()) + [report['always_continue']]
        x = np.arange(len(names))
        source_items_25 = results
        items_25 = []
        for r in source_items_25:
            items_25.append(r['balanced_accuracy'])
        ax.bar(x - 0.16, items_25, 0.32, label='Balanced accuracy', color='#486DCD')
        source_items_26 = results
        items_26 = []
        for r in source_items_26:
            items_26.append(r['f1'] or 0)
        ax.bar(x + 0.16, items_26, 0.32, label='Help F1', color='#168572')
        ax.set_xticks(x, ['INSIGHT Strong', 'Entropy', 'Mean NLL', 'Always continue'])
        ax.set_ylim(0, 1.1)
        ax.legend(frameon=False)
        ax.set_ylabel('Observed test score')
        ax.set_title(f"Independent initial-state test: {len(te)} human labels, {report['n_init_ids']} init IDs\nThresholds fitted on validation; no policy adaptation measured")
        fig.savefig(result / 'detector_comparison.png', dpi=170)
        plt.close(fig)
    except ImportError:
        pass
    print(json.dumps(report['models'], indent=2))
    print('\nINDEPENDENT DETECTOR TEST:', result)

def smoke(root, args):
    torch, model, source = torch_model(root)
    torch.manual_seed(0)
    x = torch.randn(2, 12, 4)
    pad = torch.zeros((2, 12), dtype=torch.bool)
    pad[1, 8:] = True
    model.train()
    loss = torch.nn.functional.binary_cross_entropy_with_logits(model(x, pad), torch.tensor([0.0, 1.0]))
    loss.backward()
    protocol.require(math.isfinite(float(loss.detach())) and all((p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())), 'CPU model smoke failed.')
    print('AUTHOR MODEL CPU FORWARD/BACKWARD OK; synthetic check only, classifier is not trained.')
    print('Source:', source)
