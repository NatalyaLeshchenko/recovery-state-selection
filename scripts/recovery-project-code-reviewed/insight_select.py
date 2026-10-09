#!/usr/bin/env python3
from __future__ import annotations
import argparse
from collections import Counter
import csv
import html
from pathlib import Path
import sys
import numpy as np
import insight_pipeline as p
import insight_training as training
POINTER = 'pilot_runs/insight_selection_plan_path.txt'
KINDS = ('missed_grasp', 'object_dropped')

def within(parent, value):
    child = p.resolve(parent, value)
    p.require(child.is_relative_to(parent.resolve()), 'Путь выходит за пределы collection.')
    return child

def same_snapshot(candidate, query):
    with np.load(candidate, allow_pickle=False) as a, np.load(query, allow_pickle=False) as b:
        required = {'simulator_state', 'ctrl', 'reset_state', 'reset_ctrl', 'past_actions', 'observation/pixels/image', 'observation/pixels/image2'}
        p.require(required <= set(a.files) and set(a.files) == set(b.files) and all((np.array_equal(a[key], b[key]) for key in a.files)), 'Первый recovery query относится к другому simulator/observation state.')

def verify_geometry(case, trajectory, run):
    import collect_vla_displaced as collector
    kind = case['verified_failure_type']
    p.require(kind in KINDS and case['requested_failure_type'] == kind, 'Неизвестный тип ошибки.')
    event = case['intervention']
    lo, hi = (event['start_step'], event['end_step'])
    p.require(type(lo) is type(hi) is int and 0 <= lo < hi <= len(trajectory), 'Некорректные границы вмешательства.')
    rows = trajectory[lo:hi]
    p.require(all((row['phase'] == 'intervention' and row['step'] == lo + i for i, row in enumerate(rows))), 'Трасса вмешательства не совпадает с event boundaries.')
    p.require(rows[0]['geometry_before'] == event['before'], 'Изменились исходные координаты event.')
    for key in ('drop_threshold_m', 'miss_max_bowl_rise_m', 'miss_min_empty_lift_m', 'miss_max_bowl_shift_m'):
        p.require(run['protocol'][key] == collector.PROTOCOL[key], 'Изменились геометрические критерии.')
    verified, evidence = collector.verify_event(kind, case['initial_geometry'], event['before'], rows, event['held_ever'])
    p.require(verified, 'Тип ошибки не подтверждается сохранённой геометрией и контактами.')
    return (kind, evidence)

def read_candidates(root, pool):
    import label_vla_help as labels
    pool = p.resolve(root, pool) if pool else p.pointer(root, p.COLLECTION_POINTER)
    parent = p.read(pool / 'run_info.json')
    p.require(parent['status'] == 'complete' and parent['purpose'] == 'selection', 'Нужен завершённый selection pool. Запусти: python insight_pipeline.py collect --frozen --purpose selection')
    frozen_path, frozen = p.frozen_protocol(root)
    p.require(parent['experiment_protocol'] == p.EXPERIMENT_PROTOCOL and parent['frozen_protocol_sha256'] == p.digest(frozen_path) and (parent['profiles'] == [frozen['profile']]), 'Collection не соответствует замороженному протоколу.')
    p.validate_signature(p.signature(parent))
    p.require(p.signature(parent) == frozen['policy_protocol'], 'Изменилась VLA или признаки.')
    ids = parent['init_state_ids']
    p.require(ids and len(ids) == len(set(ids)) and all((type(i) is int and i in p.ROLES['selection'] for i in ids)), 'Selection должен использовать отдельные init IDs 20..29.')
    source_items_1 = ids
    expected = set()
    for i in source_items_1:
        for kind in KINDS:
            expected.add(f'init_{i:03d}_{kind}')
    rows, excluded, seen = ([], [], set())
    hashes = {str(pool / 'run_info.json'): p.digest(pool / 'run_info.json')}
    for value in parent['runs']:
        run_path = within(pool, value)
        run = p.read(run_path / 'run_info.json')
        hashes[str(run_path / 'run_info.json')] = p.digest(run_path / 'run_info.json')
        p.require(run['purpose'] == 'selection' and run['profile'] == frozen['profile'] and ({k: run['protocol'][k] for k in frozen['parameters']} == frozen['parameters']), 'Run относится к другому split или профилю.')
        for path in sorted(run_path.glob('init_*/case.json')):
            case = p.read(path)
            identity = case['case_id']
            p.require(identity in expected and identity not in seen, 'Лишний или повторный кандидат.')
            seen.add(identity)
            hashes[str(path)] = p.digest(path)
            p.require(case['purpose'] == 'selection' and case['init_state_id'] in ids and (case['decoder_protocol'] >= 3), 'Нарушен split или decoder protocol.')
            if not case['candidate_accepted']:
                excluded.append({'candidate_id': identity, 'reason': 'no_verified_physical_start'})
                continue
            candidate = path.parent / 'candidate/start.npz'
            p.require(p.digest(candidate) == case['candidate_snapshot_sha256'], 'Изменился candidate snapshot.')
            hashes[str(candidate)] = p.digest(candidate)
            trajectory_path = path.parent / 'trajectory.json'
            hashes[str(trajectory_path)] = p.digest(trajectory_path)
            kind, evidence = verify_geometry(case, p.read(trajectory_path), run)
            source_items_2 = case['inferences']
            first = []
            for b in source_items_2:
                if b['phase'] == 'recovery':
                    first.append(b)
            if not first:
                excluded.append({'candidate_id': identity, 'reason': 'no_first_recovery_query'})
                continue

            def boundary_step(b):
                return b['step']
            boundary = min(first, key=boundary_step)
            p.require(boundary['step'] == case['recovery_start_step'], 'Первый query не относится к candidate start.')
            query = within(run_path, boundary['query'])
            p.require(p.digest(query / 'start.npz') == boundary['snapshot_sha256'], 'Изменился query snapshot.')
            same_snapshot(candidate, query / 'start.npz')
            for name in ('start.npz', 'token_features.npz', 'inference.json'):
                if (query / name).is_file():
                    hashes[str(query / name)] = p.digest(query / name)
            features, reason = labels.classifier_features(query, boundary)
            if reason:
                excluded.append({'candidate_id': identity, 'failure_type': kind, 'reason': reason})
                continue
            rows.append({'candidate_id': identity, 'init_id': case['init_state_id'], 'failure_type': kind, 'query': str(query), 'snapshot': str(candidate), 'snapshot_sha256': case['candidate_snapshot_sha256'], 'features': features, 'geometry_evidence': evidence, 'intervention_video': str(path.parent / 'intervention.mp4')})
    p.require(seen == expected, 'В завершённом collection отсутствуют запрошенные cases.')
    return (pool, parent, rows, excluded, hashes)

def fixed_detector(root, model):
    folder = p.resolve(root, model) if model else p.pointer(root, p.MODEL_POINTER)
    record = p.read(folder / 'train_info.json')
    calibration = p.read(folder / 'calibration.json')
    frozen_path, _ = p.frozen_protocol(root)
    p.require(record['status'] == 'trained' and record['frozen_protocol_sha256'] == p.digest(frozen_path), 'Нужен обученный классификатор замороженного протокола.')
    p.require(p.digest(folder / 'single_fold0.pt') == record['checkpoint_sha256'] and p.digest(folder / 'calibration.json') == record['calibration_sha256'], 'Checkpoint или validation-порог изменились после fit.')
    threshold = float(calibration['INSIGHT_Strong']['threshold'])
    p.require(np.isfinite(threshold) and threshold == record['threshold_logit'], 'Некорректный validation-порог.')
    return (folder, record, threshold)

def predict(root, folder, record, rows):
    torch, model, source = training.torch_model(root)
    p.require(p.digest(source) == record['author_source_sha256'], 'Изменилась авторская архитектура.')
    model.load_state_dict(torch.load(folder / 'single_fold0.pt', map_location='cpu', weights_only=True))
    model.eval()
    values = []
    with torch.no_grad():
        for row in rows:
            x = torch.from_numpy(row['features'])[None]
            padding = torch.zeros((1, x.shape[1]), dtype=torch.bool)
            values.append(float(model(x, padding).item()))
    return np.asarray(values, dtype=np.float64)

def plan(rows, budget, seed):
    from selection_pilot import query_plan
    p.require(set((row['failure_type'] for row in rows)) == set(KINDS), 'Общий valid-feature pool должен содержать оба типа ошибки. Исключения смотри в diagnostic.json.')
    original = query_plan(rows, budget, seed)
    return {'INSIGHT': original['U'], 'F_geometry': original['F_ref'], 'Random': original['Random']}

def visuals(out, rows, conditions, budget):
    import label_vla_help as labels
    (out / 'assets').mkdir()
    source_items_3 = rows
    indexed = {}
    for row in source_items_3:
        indexed[row['candidate_id']] = row
    colors = {'INSIGHT': '#6d8cff', 'F_geometry': '#43cfb1', 'Random': '#edb65c'}
    cards = []
    for method, choice in conditions.items():
        examples = []
        for identity in choice['selected_candidates']:
            row = indexed[identity]
            images = labels.snapshot_images(Path(row['query']))
            source_items_4 = images
            images = []
            for image in source_items_4:
                images.append(labels.save_image(out, image))
            examples.append('<article><b>' + html.escape(identity) + '</b><p>' + html.escape(row['failure_type']) + f""" · help logit {row['U']:.4f}</p><div class="views">""" + ''.join(('<img src="' + html.escape(im, quote=True) + '" alt="Saved robot camera">' for im in images)) + '</div><p><a href="' + html.escape(Path(row['intervention_video']).as_uri(), quote=True) + '">Intervention video</a></p></article>')
        cards.append('<section style="border-top:4px solid ' + colors[method] + '"><h2>' + method + '</h2><p>' + html.escape(str(choice['failure_counts'])) + '</p>' + ''.join(examples) + '</section>')
    page = '<!doctype html><html lang="en"><meta charset="utf-8"><title>Recovery state selection</title><style>'
    page += 'body{background:#101827;color:#e8eef9;font:16px system-ui;margin:0;padding:32px}main{max-width:1400px;margin:auto}.grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:20px}section{background:#1b2639;padding:20px;border-radius:12px}article{border-top:1px solid #354157;padding-top:16px;margin-top:20px}.views{display:flex;gap:8px}.views img{width:calc(50% - 4px);height:auto}a{color:#a9c6ff}p{line-height:1.5;color:#bac8df}@media(max-width:850px){.grid{grid-template-columns:1fr}}</style><main>'
    page += f'<h1>Recovery state selection · B = {budget}</h1><p>Same saved pi0-FAST candidate starts. Fixed INSIGHT scores, simulator-verified failure coverage, and seeded random selection. These are actual camera snapshots. This page shows a query plan; expert demonstrations and adapted-policy success have not been measured.</p><div class="grid">' + ''.join(cards) + '</div></main></html>'
    (out / 'gallery.html').write_text(page, encoding='utf-8')
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        names = list(conditions)
        fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
        bottom = np.zeros(len(names))
        for kind, color in zip(KINDS, ('#6d8cff', '#43cfb1')):
            source_items_5 = names
            items_5 = []
            for name in source_items_5:
                items_5.append(conditions[name]['failure_counts'].get(kind, 0))
            count = np.asarray(items_5)
            ax.bar(names, count, bottom=bottom, color=color, label=kind)
            bottom += count
        ax.set_ylabel('Selected candidate states')
        ax.set_ylim(0, budget + 0.6)
        ax.set_yticks(range(budget + 1))
        ax.set_title(f'Same common pool · query budget B={budget}\nCoverage comparison; no adapted-policy success measured')
        ax.legend(frameon=False)
        fig.savefig(out / 'coverage.png', dpi=170)
        plt.close(fig)
    except ImportError:
        print('Matplotlib unavailable; gallery and tables saved.')

def run(root, args):
    root = Path(root).resolve()
    pool, parent, rows, excluded, hashes = read_candidates(root, args.pool)
    folder, record, threshold = fixed_detector(root, args.model)
    p.require(p.signature(parent) == record['policy_protocol'], 'Scoring использует другую VLA.')
    p.require(args.budget >= 1 and args.seed >= 0, 'Budget >=1, seed >=0.')
    out = root / 'pilot_runs/vla_insight' / ('selector_comparison_' + p.stamp())
    out.mkdir(parents=True)
    diagnostic = {'pool': str(pool), 'n_requested': len(parent['init_state_ids']) * 2, 'n_valid_feature_candidates': len(rows), 'excluded': excluded, 'common_pool_failure_counts': dict(Counter((row['failure_type'] for row in rows))), 'scope': 'Valid first-query features only; exclusions apply to all selectors. No autonomous outcome filters candidates.'}
    p.write(out / 'diagnostic.json', diagnostic)
    p.require(rows and args.budget <= len(rows), f"Недостаточно valid-feature кандидатов для B={args.budget}: {out / 'diagnostic.json'}")
    p.require(set((row['failure_type'] for row in rows)) == set(KINDS), f"Нет обоих типов в общем пуле: {out / 'diagnostic.json'}")
    print(f'CPU SCORE: {len(rows)} candidates; excluded={len(excluded)}; fixed Strong detector')
    values = predict(root, folder, record, rows)
    p.require(values.shape == (len(rows),) and np.isfinite(values).all(), 'Некорректные detector scores.')
    for row, value in zip(rows, values):
        row.update(U=float(value), help_probability=float(np.exp(-np.logaddexp(0.0, -value))), above_validation_threshold=bool(value >= threshold))
        print(f"{row['candidate_id']}: help logit={value:.5f}; {row['failure_type']}")
    for path, expected in hashes.items():
        p.require(p.digest(path) == expected, 'Collection изменился во время scoring.')
    conditions = plan(rows, args.budget, args.seed)
    serializable = [{key: value for key, value in row.items() if key != 'features'} for row in rows]
    result = {**diagnostic, 'status': 'planned', 'conditions': conditions, 'candidates': serializable, 'model': str(folder), 'classifier_checkpoint_sha256': record['checkpoint_sha256'], 'calibration_sha256': record['calibration_sha256'], 'source_hashes': hashes, 'frozen_protocol_sha256': record['frozen_protocol_sha256'], 'seed': args.seed, 'target_successful_demonstrations': args.budget, 'initial_query_budget': args.budget, 'INSIGHT_rule': 'Descending first-post-intervention help logit. Fixed-budget ranking, even below the binary threshold; no retraining or threshold tuning.', 'geometry_rule': 'Balanced quotas over two contact/motion-verified induced failure types; seeded choice within type; privileged simulator reference.', 'random_rule': 'Uniform seeded candidate permutation without replacement; same common pool.', 'model_trained_here': False, 'human_labels_required_here': False, 'expert_demonstrations_collected': False, 'policy_success_measured': False, 'note': 'Selection coverage is descriptive, not proof of recovery-success gains. This adapts INSIGHT to pool ranking; it is not the original first-trigger online protocol.'}
    p.write(out / 'selection.json', result)
    with (out / 'scores.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.writer(stream)
        writer.writerow(['candidate_id', 'init_id', 'failure_type', 'help_logit', 'help_probability', 'above_validation_threshold'])
        for row in rows:
            source_items_6 = ('candidate_id', 'init_id', 'failure_type', 'U', 'help_probability', 'above_validation_threshold')
            items_6 = []
            for k in source_items_6:
                items_6.append(row[k])
            writer.writerow(items_6)
    visuals(out, rows, conditions, args.budget)
    p.set_pointer(root, POINTER, out)
    print(f'\nSELECTOR QUERY PLAN READY: B={args.budget}, seed={args.seed}')
    for method, choice in conditions.items():
        print(method, choice['selected_candidates'], choice['failure_counts'])
    print('Output:', out)
    print('Gallery:', out / 'gallery.html')
    return out

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pool', help='Завершённый selection collection; по умолчанию текущий collection pointer.')
    parser.add_argument('--model', help='Фиксированный Strong detector; по умолчанию model pointer.')
    parser.add_argument('--budget', type=int, default=2)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    run(Path(__file__).resolve().parent, args)
if __name__ == '__main__':
    try:
        main()
    except (Exception, KeyboardInterrupt) as error:
        print('\nSTOP:', error, file=sys.stderr)
        sys.exit(1)
