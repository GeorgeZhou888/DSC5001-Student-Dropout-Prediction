"""B-stage fixed-fold training. Never trains on or evaluates the held-out test set."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime
import hashlib
from importlib import metadata
import json
from pathlib import Path
import platform
import re
import sys
import time
import warnings
from zoneinfo import ZoneInfo

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, make_scorer, precision_score, recall_score
from sklearn.model_selection import GridSearchCV, ParameterGrid, PredefinedSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[2]
LABELS = ['Dropout', 'Enrolled', 'Graduate']
FEATURE_SETS = ['enrollment', 'enrollment_plus_sem1']
MODEL_NAMES = ['logistic_regression', 'decision_tree', 'random_forest']
FORBIDDEN = {'row_id', 'Target', 'dataset_split', 'cv_fold'}
INPUTS = ['data/manifest.json', 'data/processed/students.csv',
          'data/splits/split_manifest.csv', 'configs/feature_sets.json',
          'configs/protocol.json', 'environment/python-version.txt',
          'environment/requirements-lock.txt']
# Frozen in code before this run; no test-set feedback is used to define grids.
GRIDS = {
    'logistic_regression': {'classifier__C': [0.1, 1.0, 10.0],
                            'classifier__class_weight': [None, 'balanced']},
    'decision_tree': {'classifier__max_depth': [3, 6, None],
                      'classifier__min_samples_leaf': [5, 20],
                      'classifier__class_weight': [None, 'balanced']},
    'random_forest': {'classifier__max_depth': [12, None],
                      'classifier__min_samples_leaf': [1, 5],
                      'classifier__class_weight': [None, 'balanced']},
}


def now():
    return datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(timespec='seconds')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2,
                               allow_nan=False) + '\n', encoding='utf-8', newline='\n')


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False, encoding='utf-8', lineterminator='\n')


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def inside(root, relative):
    if not isinstance(relative, str) or '\\' in relative:
        raise ValueError('Manifest paths must be relative POSIX paths')
    parts = relative.split('/')
    if any(p in ('', '.', '..') or ':' in p for p in parts):
        raise ValueError('Unsafe manifest path')
    path = root.joinpath(*parts).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError('Manifest path leaves the project root')
    return path


def validate_environment(root):
    wanted_python = (root / 'environment/python-version.txt').read_text().strip().removeprefix('Python ')
    if platform.python_version().split('.')[:2] != wanted_python.split('.')[:2]:
        raise ValueError(f'Expected Python {wanted_python} major/minor; got {platform.python_version()}')
    versions = {}
    for line in (root / 'environment/requirements-lock.txt').read_text().splitlines():
        if not line.strip() or line.lstrip().startswith('#'):
            continue
        match = re.fullmatch(r'([A-Za-z0-9_.-]+)==([^\s]+)', line.strip())
        if not match:
            raise ValueError('Environment lock must contain exact package pins only')
        name, wanted = match.groups()
        actual = metadata.version(name)
        if actual != wanted:
            raise ValueError(f'{name}: expected {wanted}, found {actual}')
        versions[name] = actual
    if not {'numpy', 'pandas', 'scikit-learn', 'joblib', 'threadpoolctl'}.issubset(versions):
        raise ValueError('Incomplete environment lock')
    return {'python_version': platform.python_version(), 'a_recorded_python': wanted_python,
            'python_policy': 'same major/minor; patch recorded explicitly',
            'system': platform.system(), 'architecture': platform.machine(),
            'packages': versions, 'exact_package_lock_match': True}


def validate_input_hashes(root, allow_lf_equivalent=False):
    manifest = read_json(root / 'data/manifest.json')
    required = set(INPUTS) - {'data/manifest.json'}
    if not required.issubset(manifest['files']):
        raise ValueError('A manifest does not cover every required B input')
    records = []
    for relative, expected in manifest['files'].items():
        path = inside(root, relative)
        content = path.read_bytes()
        actual = hashlib.sha256(content).hexdigest()
        status = 'exact'
        canonical = None
        if actual != expected['sha256']:
            # Only a proven CRLF->LF equivalence is allowed. Do not rewrite files.
            allowed_text = (path.suffix in {'.py', '.json', '.md', '.txt', '.csv'}
                            and not relative.startswith('data/raw/'))
            if allow_lf_equivalent and allowed_text and b'\x00' not in content:
                content.decode('utf-8-sig')
                canonical = hashlib.sha256(content.replace(b'\r\n', b'\n')).hexdigest()
            if canonical != expected['sha256']:
                raise ValueError(f'A input checksum mismatch: {relative}. '
                                 'Only --allow-lf-equivalent can accept proven CRLF/LF differences.')
            status = 'lf_equivalent_only'
        elif len(content) != expected['bytes']:
            raise ValueError(f'A input byte count mismatch: {relative}')
        records.append({'file': relative, 'expected_sha256': expected['sha256'],
                        'actual_sha256': actual, 'expected_bytes': expected['bytes'],
                        'actual_bytes': len(content), 'status': status,
                        'lf_canonical_sha256': canonical})
    return manifest, {'status': 'CONTENT_VERIFIED_ACCEPTANCE_PENDING',
        'policy': 'explicit LF-equivalence exception; upstream files are never changed'
                  if allow_lf_equivalent else 'exact bytes only',
        'files': records, 'exact_count': sum(r['status'] == 'exact' for r in records),
        'lf_equivalent_count': sum(r['status'] == 'lf_equivalent_only' for r in records),
        'formal_A_acceptance': 'Not asserted; A/leader must confirm the frozen input version.'}


def validate_protocol(protocol, features):
    expected = {'random_seed': 42, 'target_column': 'Target', 'row_id_column': 'row_id',
                'test_fraction': 0.2, 'labels': LABELS, 'models': MODEL_NAMES,
                'baseline': 'majority_class', 'selection_metric': 'macro_f1',
                'processed_data': 'data/processed/students.csv',
                'split_manifest': 'data/splits/split_manifest.csv',
                'feature_sets_file': 'configs/feature_sets.json',
                'learned_preprocessing': 'fit_within_each_training_fold',
                'test_usage': 'final_evaluation_only_after_model_freeze'}
    if any(protocol.get(k) != v for k, v in expected.items()):
        raise ValueError('Protocol differs from the agreed B-stage experiment')
    if protocol.get('status') != 'ready' or features.get('status') != 'ready':
        raise ValueError('A configuration is not ready')
    if protocol['cross_validation'] != {'n_splits': 5, 'stratified': True,
                                      'shuffle': True, 'fold_column': 'cv_fold'}:
        raise ValueError('Unexpected cross-validation protocol')
    if protocol['outer_split'] != {'stratified': True, 'shuffle': True}:
        raise ValueError('Unexpected outer split protocol')
    first, second = (features[k] for k in FEATURE_SETS)
    cats, nums = features['categorical_features'], features['numeric_features']
    if len(first) != len(set(first)) or len(first) != 18 or len(second) != len(set(second)) or len(second) != 24:
        raise ValueError('Expected unique 18/24 feature lists')
    if not set(first) < set(second) or set(second) & FORBIDDEN:
        raise ValueError('Invalid feature nesting or forbidden predictor')
    if any('2nd sem' in c for c in second) or set(second) & set(features['excluded_fields_with_reasons']):
        raise ValueError('Excluded predictor present')
    if len(cats) != 14 or len(nums) != 10 or set(cats) & set(nums) or set(cats + nums) != set(second):
        raise ValueError('Invalid categorical/numeric partition')


def read_training_only(root, features):
    """Discard held-out records before building any model/metric DataFrame."""
    with (root / 'data/splits/split_manifest.csv').open(encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != ['row_id', 'dataset_split', 'cv_fold']:
            raise ValueError('Unexpected split columns')
        splits = list(reader)
    split_by_id = {r['row_id']: r for r in splits}
    if len(split_by_id) != len(splits) or len(splits) != 4424:
        raise ValueError('Duplicate/missing split IDs')
    for r in splits:
        if r['dataset_split'] == 'test' and r['cv_fold'] == '':
            continue
        if r['dataset_split'] != 'train' or r['cv_fold'] not in {'0', '1', '2', '3', '4'}:
            raise ValueError('Invalid split/fold values')
    rows, seen = [], []
    with (root / 'data/processed/students.csv').open(encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f)
        if len(reader.fieldnames) != 38 or len(set(reader.fieldnames)) != 38:
            raise ValueError('Unexpected processed columns')
        if not set(features['enrollment_plus_sem1'] + ['Target', 'row_id']).issubset(reader.fieldnames):
            raise ValueError('Required input columns absent')
        for row in reader:
            rid = row['row_id']
            seen.append(rid)
            if rid not in split_by_id:
                raise ValueError('Unmatched processed ID')
            split = split_by_id[rid]
            if split['dataset_split'] != 'train':
                continue
            rows.append({k: row[k] for k in ['row_id', 'Target'] + features['enrollment_plus_sem1']}
                        | {'cv_fold': int(split['cv_fold'])})
    if seen != [f'S{i:06d}' for i in range(1, 4425)] or set(seen) != set(split_by_id):
        raise ValueError('Processed IDs are not stable or split coverage differs')
    train = pd.DataFrame(rows)
    if len(train) != 3539 or sum(r['dataset_split'] == 'test' for r in splits) != 885:
        raise ValueError('Expected fixed 3539/885 split')
    if train['Target'].value_counts().to_dict() != {'Graduate': 1767, 'Dropout': 1137, 'Enrolled': 635}:
        raise ValueError('Unexpected training labels/counts')
    if train.groupby('cv_fold').size().tolist() != [708, 708, 708, 708, 707]:
        raise ValueError('Unexpected fixed-fold sizes')
    for c in features['enrollment_plus_sem1']:
        train[c] = pd.to_numeric(train[c], errors='raise')
    if not np.isfinite(train[features['enrollment_plus_sem1']].to_numpy(dtype=float)).all():
        raise ValueError('Training predictors have missing/nonfinite values')
    for label in LABELS:
        fold_counts = train.loc[train.Target == label].groupby('cv_fold').size().reindex(range(5), fill_value=0)
        if fold_counts.max() - fold_counts.min() > 1:
            raise ValueError('Fixed folds are not stratified')
    return train


def build_pipeline(features, feature_set, model_name):
    selected = features[feature_set]
    cats = [c for c in selected if c in features['categorical_features']]
    nums = [c for c in selected if c in features['numeric_features']]
    preprocessor = ColumnTransformer([
        ('categorical', OneHotEncoder(handle_unknown='ignore', sparse_output=True), cats),
        ('numeric', StandardScaler() if model_name == 'logistic_regression' else 'passthrough', nums)
    ], remainder='drop', sparse_threshold=1.0)
    models = {
        'logistic_regression': LogisticRegression(solver='lbfgs', max_iter=5000, random_state=42),
        'decision_tree': DecisionTreeClassifier(random_state=42),
        'random_forest': RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=1),
        'majority_baseline': DummyClassifier(strategy='most_frequent', random_state=42),
    }
    return Pipeline([('preprocess', preprocessor), ('classifier', models[model_name])])


def scorers():
    return {'macro_f1': make_scorer(f1_score, labels=LABELS, average='macro', zero_division=0),
            'dropout_precision': make_scorer(precision_score, labels=['Dropout'], average='macro', zero_division=0),
            'dropout_recall': make_scorer(recall_score, labels=['Dropout'], average='macro', zero_division=0)}


def compute_metrics(y_true, predicted):
    return {'macro_f1': float(f1_score(y_true, predicted, labels=LABELS, average='macro', zero_division=0)),
            'dropout_precision': float(precision_score(y_true, predicted, labels=['Dropout'], average='macro', zero_division=0)),
            'dropout_recall': float(recall_score(y_true, predicted, labels=['Dropout'], average='macro', zero_division=0))}


def fit_search(pipe, grid, X, y, folds, jobs):
    search = GridSearchCV(pipe, grid, scoring=scorers(), refit=False,
                          cv=PredefinedSplit(folds), n_jobs=jobs, error_score='raise',
                          return_train_score=False)
    # Thread backend avoids repeated interpreter launch on Windows. Limit native BLAS threads.
    with joblib.parallel_backend('threading', n_jobs=jobs), threadpool_limits(limits=1):
        search.fit(X, y)
    results = search.cv_results_
    means = np.asarray(results['mean_test_macro_f1'])
    if not np.isfinite(means).all():
        raise ValueError('Nonfinite CV scores')
    # ParameterGrid order is fixed in the source. Exact ties choose first candidate.
    best = int(np.argmax(means))
    return results, best


def save_model(root, pipe, X, y, filename):
    with threadpool_limits(limits=1):
        pipe.fit(X, y)
    if list(pipe.classes_) != LABELS or list(pipe.feature_names_in_) != list(X.columns):
        raise ValueError('Unexpected model labels or predictor order')
    # Smoke-test only training rows; do not report these as validation scores.
    sample = X.iloc[:7]
    before = pipe.predict_proba(sample)
    if not np.isfinite(before).all() or not np.allclose(before.sum(axis=1), 1):
        raise ValueError('Invalid model probabilities')
    path = root / 'models' / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.joblib.tmp')
    joblib.dump(pipe, temporary, compress=3)
    if temporary.stat().st_size > 20_000_000:
        raise ValueError('Model exceeds the project 20 MB limit')
    temporary.replace(path)
    restored = joblib.load(path)  # Only the artifact just authored by this run is loaded.
    if not np.array_equal(restored.predict(sample), pipe.predict(sample)) or not np.array_equal(restored.predict_proba(sample), before):
        raise ValueError('Serialized model round-trip differs')
    return {'path': path.relative_to(root).as_posix(), 'sha256': sha(path),
            'bytes': path.stat().st_size, 'classes': list(pipe.classes_),
            'features': list(X.columns), 'training_rows': len(X),
            'training_only_serialization_smoke_check': 'PASS'}


def train_models(root, args):
    started = now()
    t0 = time.perf_counter()
    result_dir = root / 'results/B_validation'
    manifest_path = result_dir / 'run_manifest.json'
    if manifest_path.exists() and not args.overwrite:
        raise ValueError('Existing B run found; use --overwrite only to intentionally replace it')
    upstream, input_check = validate_input_hashes(root, args.allow_lf_equivalent)
    environment = validate_environment(root)
    features = read_json(root / 'configs/feature_sets.json')
    protocol = read_json(root / 'configs/protocol.json')
    validate_protocol(protocol, features)
    train = read_training_only(root, features)
    inputs_before = {name: sha(root / name) for name in INPUTS}
    frozen_before = {r['file']: r['actual_sha256'] for r in input_check['files']}
    own_sources = sorted((root / 'src/B_models').glob('*.py'))
    source_hashes = {p.relative_to(root).as_posix(): sha(p) for p in own_sources}
    portable_command = ('python src/B_models/train_models.py'
        + (' --allow-lf-equivalent' if args.allow_lf_equivalent else '')
        + f' --source-commit {args.source_commit}'
        + (f' --source-zip-sha256 {args.source_zip_sha256}' if args.source_zip_sha256 else '')
        + f' --jobs {args.jobs}')
    run = {'schema_version': 1, 'stage': 'B_training_and_freeze', 'status': 'RUNNING',
           'started_at': started, 'timezone': 'Asia/Shanghai',
           'source_snapshot_commit': args.source_commit,
           'source_zip_sha256': args.source_zip_sha256,
           'A_manifest_input_baseline_commit': upstream.get('input_baseline_commit'),
           'A_manifest_repository_head_at_run': upstream.get('repository_head_at_run'),
           'inputs_actual_sha256': inputs_before, 'source_sha256': source_hashes,
           'environment': environment, 'random_seed': 42, 'command': portable_command,
           'n_training_rows': len(train), 'n_test_rows_reserved': 885,
           'test_policy': 'Test rows discarded before model DataFrame creation; no test predictions or metrics.',
           'folds': 'Read cv_fold from A manifest; never regenerate folds',
           'grid': GRIDS, 'selection_metric': 'mean five-fold macro_f1',
           'tie_break': 'First candidate in deterministic sklearn ParameterGrid order on exact score ties',
           'metric_labels': LABELS, 'zero_division': 0,
           'representative_model_policy': 'Highest mean CV macro_f1 among six groups; exact ties use FEATURE_SETS then MODEL_NAMES order.',
           'limits': ['CV is used for tuning; selected CV scores are not an unbiased final generalization estimate.',
                      'A formal acceptance and B/C/leader sign-off are not asserted by this script.',
                      'First-semester comparison is retrospective, not a validated real-time warning system.']}
    write_json(result_dir / 'input_verification.json', input_check)
    write_json(result_dir / 'environment.json', environment)
    write_json(manifest_path, run)
    (result_dir / 'requirements-lock.txt').write_text(
        ''.join(f'{k}=={v}\n' for k, v in sorted(environment['packages'].items())),
        encoding='utf-8', newline='\n')
    selected = {'schema_version': 1, 'selection_metric': 'macro_f1',
                'tuning_scope': 'training-only, fixed five-fold CV', 'groups': []}
    catalog = {'schema_version': 1, 'probability_rule': 'Use each loaded pipeline.classes_; never assume numeric label order.',
               'labels': LABELS, 'models': []}
    all_metrics, selected_metrics, candidates, summary = [], [], [], []
    y, folds = train['Target'], train['cv_fold'].to_numpy(dtype=int)
    try:
        # A convergence warning fails the run rather than silently freezing an unconverged model.
        warnings.filterwarnings('error', category=ConvergenceWarning)
        for feature_set in FEATURE_SETS:
            X = train.loc[:, features[feature_set]]
            for model_name in MODEL_NAMES:
                print(f'Training {feature_set} / {model_name}: {len(ParameterGrid(GRIDS[model_name]))} candidates x 5 folds', flush=True)
                pipeline = build_pipeline(features, feature_set, model_name)
                results, best = fit_search(pipeline, GRIDS[model_name], X, y, folds, args.jobs)
                for candidate_id, params in enumerate(results['params']):
                    candidate = {'feature_set': feature_set, 'model': model_name,
                                 'candidate_id': candidate_id, 'is_selected': candidate_id == best,
                                 'params_json': json.dumps(params, sort_keys=True),
                                 'mean_fit_seconds': float(results['mean_fit_time'][candidate_id])}
                    for metric in scorers():
                        candidate[f'{metric}_mean'] = float(results[f'mean_test_{metric}'][candidate_id])
                        candidate[f'{metric}_std'] = float(results[f'std_test_{metric}'][candidate_id])
                    candidates.append(candidate)
                    for fold in range(5):
                        row = {'feature_set': feature_set, 'model': model_name,
                               'candidate_id': candidate_id, 'cv_fold': fold,
                               'is_selected': candidate_id == best,
                               'n_train': int((folds != fold).sum()), 'n_validation': int((folds == fold).sum())}
                        for metric in scorers():
                            row[metric] = float(results[f'split{fold}_test_{metric}'][candidate_id])
                        all_metrics.append(row)
                        if candidate_id == best:
                            selected_metrics.append(row)
                best_params = results['params'][best]
                model = save_model(root, clone(pipeline).set_params(**best_params), X, y,
                                   f'{feature_set}__{model_name}.joblib')
                model.update({'model': model_name, 'feature_set': feature_set})
                catalog['models'].append(model)
                item = {'feature_set': feature_set, 'model': model_name,
                        'selected_candidate_id': best, 'params': best_params,
                        'selection_metric': 'macro_f1', 'mean_validation_score': float(results['mean_test_macro_f1'][best]),
                        'std_validation_score': float(results['std_test_macro_f1'][best]),
                        'candidate_count': len(results['params']), 'fold_count': 5,
                        'model_file': model['path']}
                selected['groups'].append(item)
                selected_candidate = next(c for c in candidates if c['feature_set'] == feature_set and c['model'] == model_name and c['is_selected'])
                summary.append({k: v for k, v in selected_candidate.items() if k not in {'is_selected', 'params_json', 'mean_fit_seconds'}})
                print(f"  selected macro-F1={item['mean_validation_score']:.6f}", flush=True)
        # One shared majority baseline: the input feature values do not affect its prediction.
        print('Training shared majority baseline: 5 fixed folds', flush=True)
        X = train.loc[:, features['enrollment']]
        baseline = build_pipeline(features, 'enrollment', 'majority_baseline')
        baseline_rows = []
        for fold, (fit_idx, val_idx) in enumerate(PredefinedSplit(folds).split()):
            fitted = clone(baseline).fit(X.iloc[fit_idx], y.iloc[fit_idx])
            row = {'feature_set': 'shared_majority', 'model': 'majority_baseline',
                   'candidate_id': 0, 'cv_fold': fold, 'is_selected': True,
                   'n_train': len(fit_idx), 'n_validation': len(val_idx),
                   **compute_metrics(y.iloc[val_idx], fitted.predict(X.iloc[val_idx]))}
            baseline_rows.append(row)
        all_metrics.extend(baseline_rows)
        selected_metrics.extend(baseline_rows)
        baseline_summary = {'feature_set': 'shared_majority', 'model': 'majority_baseline', 'candidate_id': 0}
        for metric in scorers():
            values = [r[metric] for r in baseline_rows]
            baseline_summary.update({f'{metric}_mean': float(np.mean(values)), f'{metric}_std': float(np.std(values, ddof=0))})
        summary.append(baseline_summary)
        baseline_file = save_model(root, baseline, X, y, 'majority_baseline.joblib')
        baseline_file.update({'model': 'majority_baseline', 'feature_set': 'shared_majority',
                              'input_feature_set': 'enrollment', 'applies_to': FEATURE_SETS})
        catalog['models'].append(baseline_file)
        selected['baseline'] = {'strategy': 'most_frequent', 'model_file': baseline_file['path'],
                                'input_feature_set': 'enrollment',
                                'mean_validation_score': baseline_summary['macro_f1_mean']}
        representative = max(selected['groups'], key=lambda r: r['mean_validation_score'])
        selected['representative_model_for_C'] = {k: representative[k] for k in ['feature_set', 'model', 'model_file', 'mean_validation_score']}
        write_csv(result_dir / 'cv_metrics.csv', selected_metrics)
        write_csv(result_dir / 'cv_all_candidates.csv', all_metrics)
        write_csv(result_dir / 'cv_search_results.csv', candidates)
        write_csv(result_dir / 'cv_summary.csv', summary)
        write_json(result_dir / 'selected_params.json', selected)
        write_json(result_dir / 'model_catalog.json', catalog)
        if any(sha(root / name) != expected for name, expected in frozen_before.items()) or any(sha(root / name) != expected for name, expected in inputs_before.items()):
            raise ValueError('Upstream inputs changed during training')
        if any(sha(root / name) != expected for name, expected in source_hashes.items()):
            raise ValueError('B source changed during training')
        training_ids = train['row_id'].tolist()
        write_csv(result_dir / 'training_fold_membership.csv', train[['row_id', 'cv_fold']].to_dict('records'))
        run.update({'status': 'COMPLETE_PENDING_TEAM_ACCEPTANCE', 'finished_at': now(),
                    'duration_seconds': round(time.perf_counter() - t0, 3),
                    'training_ids_sha256': hashlib.sha256(('\n'.join(training_ids) + '\n').encode()).hexdigest(),
                    'n_cv_rows_selected': len(selected_metrics), 'n_cv_rows_all_candidates': len(all_metrics),
                    'n_candidate_configurations': len(candidates), 'n_saved_models': len(catalog['models']),
                    'all_upstream_files_unchanged': True,
                    'representative_model_for_C': selected['representative_model_for_C'],
                    'acceptance_note': 'B execution complete; explicit LF-equivalence exception and A/B/C/leader acceptance remain documented.'})
        output_files = list(result_dir.glob('*.csv')) + [result_dir / n for n in ['selected_params.json', 'model_catalog.json', 'input_verification.json', 'environment.json', 'requirements-lock.txt']]
        output_files += [root / r['path'] for r in catalog['models']]
        run['output_sha256'] = {p.relative_to(root).as_posix(): sha(p) for p in sorted(output_files)}
        write_json(manifest_path, run)
        print('B stage complete: seven saved pipelines; no test-set evaluation.', flush=True)
    except Exception as exc:
        run.update({'status': 'FAILED', 'finished_at': now(), 'error_type': type(exc).__name__, 'error': str(exc)})
        write_json(manifest_path, run)
        raise
    return run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--allow-lf-equivalent', action='store_true', help='Accept only hash-proven CRLF/LF differences without editing A inputs')
    parser.add_argument('--source-commit', required=True, help='40-hex commit of the supplied source snapshot')
    parser.add_argument('--source-zip-sha256', default=None)
    parser.add_argument('--jobs', type=int, default=2)
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()
    if not re.fullmatch(r'[a-fA-F0-9]{40}', args.source_commit):
        parser.error('--source-commit must be a full Git SHA')
    if args.source_zip_sha256 and not re.fullmatch(r'[a-fA-F0-9]{64}', args.source_zip_sha256):
        parser.error('--source-zip-sha256 must be a SHA-256')
    if args.jobs not in range(1, 9):
        parser.error('--jobs must be 1 to 8')
    train_models(ROOT, args)


if __name__ == '__main__':
    main()
