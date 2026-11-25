"""Verify B artifacts; --recheck-cv independently refits selected fixed folds only."""
import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import confusion_matrix
from threadpoolctl import threadpool_limits

import train_models as b


def independent_scores(y, predicted):
    matrix = confusion_matrix(y, predicted, labels=b.LABELS).astype(float)
    tp = np.diag(matrix)
    precision = np.divide(tp, matrix.sum(axis=0), out=np.zeros(3), where=matrix.sum(axis=0) != 0)
    recall = np.divide(tp, matrix.sum(axis=1), out=np.zeros(3), where=matrix.sum(axis=1) != 0)
    f1 = np.divide(2 * tp, matrix.sum(axis=0) + matrix.sum(axis=1), out=np.zeros(3),
                   where=matrix.sum(axis=0) + matrix.sum(axis=1) != 0)
    return {'macro_f1': float(f1.mean()), 'dropout_precision': float(precision[0]),
            'dropout_recall': float(recall[0])}


def verify(root, recheck_cv=False):
    result_dir = root / 'results/B_validation'
    run = b.read_json(result_dir / 'run_manifest.json')
    if run['status'] != 'COMPLETE_PENDING_TEAM_ACCEPTANCE':
        raise ValueError('Training is incomplete')
    checks = {}
    checks['actual_upstream_hashes_match_training_run'] = all(b.sha(root / name) == wanted for name, wanted in run['inputs_actual_sha256'].items())
    checks['training_source_hashes_match'] = all(b.sha(root / name) == wanted for name, wanted in run['source_sha256'].items())
    checks['training_output_hashes_match'] = all(b.sha(root / name) == wanted for name, wanted in run['output_sha256'].items())
    if not all(checks.values()):
        raise ValueError('Hash verification failed; do not load model artifacts')
    b.validate_environment(root)
    checks['locked_environment_matches'] = True
    _, upstream = b.validate_input_hashes(root, allow_lf_equivalent=True)
    features = b.read_json(root / 'configs/feature_sets.json')
    protocol = b.read_json(root / 'configs/protocol.json')
    b.validate_protocol(protocol, features)
    train = b.read_training_only(root, features)
    membership = pd.read_csv(result_dir / 'training_fold_membership.csv')
    checks['fixed_training_ids_and_folds_match'] = membership.equals(train[['row_id', 'cv_fold']])
    metrics = pd.read_csv(result_dir / 'cv_metrics.csv')
    all_candidates = pd.read_csv(result_dir / 'cv_all_candidates.csv')
    search = pd.read_csv(result_dir / 'cv_search_results.csv')
    summary = pd.read_csv(result_dir / 'cv_summary.csv')
    selected = b.read_json(result_dir / 'selected_params.json')
    catalog = b.read_json(result_dir / 'model_catalog.json')
    checks['six_groups_and_baseline_present'] = len(selected['groups']) == 6 and len(catalog['models']) == 7
    checks['expected_validation_row_counts'] = len(metrics) == 35 and len(all_candidates) == 265 and len(search) == 52 and len(summary) == 7
    checks['selected_rows_match_candidate_results'] = metrics.equals(all_candidates.loc[all_candidates.is_selected].reset_index(drop=True))
    checks['scores_are_finite_and_in_range'] = bool(np.isfinite(metrics[['macro_f1', 'dropout_precision', 'dropout_recall']].to_numpy()).all()
        and metrics[['macro_f1', 'dropout_precision', 'dropout_recall']].ge(0).all().all()
        and metrics[['macro_f1', 'dropout_precision', 'dropout_recall']].le(1).all().all())
    checks['models_under_20mb'] = all((root / r['path']).stat().st_size <= 20_000_000 for r in catalog['models'])
    means_match, best_match, model_checks, reproduced = True, True, [], []
    for record in catalog['models']:
        model_name, feature_set = record['model'], record['feature_set']
        chosen = metrics.loc[(metrics.model == model_name) & (metrics.feature_set == feature_set)].sort_values('cv_fold')
        if chosen.cv_fold.tolist() != list(range(5)):
            raise ValueError('A group is missing fixed validation folds')
        summary_row = summary.loc[(summary.model == model_name) & (summary.feature_set == feature_set)].iloc[0]
        for metric in ['macro_f1', 'dropout_precision', 'dropout_recall']:
            means_match &= bool(np.isclose(summary_row[f'{metric}_mean'], chosen[metric].mean(), atol=1e-12, rtol=0))
            means_match &= bool(np.isclose(summary_row[f'{metric}_std'], chosen[metric].std(ddof=0), atol=1e-12, rtol=0))
        model = joblib.load(root / record['path'])  # Hash-checked artifact from this B delivery only.
        X, y = train[record['features']], train.Target
        model_ok = (list(model.classes_) == b.LABELS and list(model.feature_names_in_) == record['features']
                    and record['training_rows'] == 3539 and b.sha(root / record['path']) == record['sha256'])
        p = model.predict_proba(X.iloc[:5])
        model_ok &= bool(np.isfinite(p).all() and np.allclose(p.sum(axis=1), 1))
        if model_name == 'logistic_regression':
            scaler = model.named_steps['preprocess'].named_transformers_['numeric']
            model_ok &= int(scaler.n_samples_seen_) == 3539
        if model_name == 'decision_tree':
            model_ok &= int(model.named_steps['classifier'].tree_.n_node_samples[0]) == 3539
        if model_name == 'majority_baseline':
            model_ok &= bool(np.array_equal(model.named_steps['classifier'].class_prior_,
                 np.array([(y == label).mean() for label in b.LABELS])))
        if model_name != 'majority_baseline':
            group = next(g for g in selected['groups'] if g['feature_set'] == feature_set and g['model'] == model_name)
            options = search.loc[(search.feature_set == feature_set) & (search.model == model_name)]
            expected_best = int(options.loc[options.macro_f1_mean.idxmax(), 'candidate_id'])
            best_match &= expected_best == group['selected_candidate_id']
            best_match &= all(model.get_params()[key] == value for key, value in group['params'].items())
        model_checks.append({'model': model_name, 'feature_set': feature_set, 'pass': bool(model_ok)})
        if recheck_cv:
            print(f'Rechecking selected model: {feature_set} / {model_name}', flush=True)
            # Independent masking, rather than calling the training script's GridSearchCV.
            for fold in range(5):
                fit_mask = train.cv_fold.to_numpy() != fold
                val_mask = ~fit_mask
                fitted = clone(model)
                with threadpool_limits(limits=1):
                    fitted.fit(X.loc[fit_mask], y.loc[fit_mask])
                actual = independent_scores(y.loc[val_mask], fitted.predict(X.loc[val_mask]))
                expected = chosen.loc[chosen.cv_fold == fold].iloc[0]
                agrees = all(np.isclose(actual[key], expected[key], atol=1e-12, rtol=0) for key in actual)
                reproduced.append({'model': model_name, 'feature_set': feature_set, 'cv_fold': fold,
                                   'scores_match': bool(agrees), **actual})
    checks['summary_mean_and_std_match_folds'] = bool(means_match)
    checks['parameters_selected_by_cv_macro_f1'] = bool(best_match)
    checks['saved_models_labels_features_and_fit_scope'] = all(r['pass'] for r in model_checks)
    representative = max(selected['groups'], key=lambda r: r['mean_validation_score'])
    checks['representative_selected_before_test'] = representative['model_file'] == selected['representative_model_for_C']['model_file']
    checks['upstream_unchanged_after_verification'] = all(b.sha(root / name) == wanted for name, wanted in run['inputs_actual_sha256'].items())
    if recheck_cv:
        checks['all_35_selected_fold_scores_independently_reproduced'] = len(reproduced) == 35 and all(r['scores_match'] for r in reproduced)
        b.write_csv(result_dir / 'cv_reproduction.csv', reproduced)
    report = {'status': 'PASS' if all(checks.values()) else 'FAIL', 'verified_at': b.now(),
              'training_run_manifest_sha256': b.sha(result_dir / 'run_manifest.json'),
              'checks': checks, 'model_checks': model_checks, 'independent_cv_recheck': recheck_cv,
              'upstream_lf_equivalent_files': upstream['lf_equivalent_count'],
              'test_set_evaluated': False,
              'team_acceptance': 'Not asserted; PR, C confirmation and leader acceptance remain manual.'}
    b.write_json(result_dir / 'delivery_check.json', report)
    print(f"B verification: {report['status']} ({sum(checks.values())}/{len(checks)})", flush=True)
    if report['status'] != 'PASS':
        raise ValueError('Some delivery checks failed')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--recheck-cv', action='store_true', help='Refit 35 selected training/validation fold models, never test rows')
    args = parser.parse_args()
    verify(Path(__file__).resolve().parents[2], args.recheck_cv)
