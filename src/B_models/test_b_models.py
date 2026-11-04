"""Targeted B-stage checks: integrity, fixed folds, leakage and model interchange."""
import csv
import json
from pathlib import Path
import shutil
import tempfile
import unittest

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import PredefinedSplit

import train_models as b


class BProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = Path(__file__).resolve().parents[2]
        cls.features = b.read_json(cls.root / 'configs/feature_sets.json')
        cls.train = b.read_training_only(cls.root, cls.features)

    def test_checksum_policy_distinguishes_line_endings_and_content(self):
        with tempfile.TemporaryDirectory(prefix='b_integrity_') as directory:
            root = Path(directory)
            (root / 'data').mkdir()
            (root / 'configs').mkdir()
            records = {}
            for name in set(b.INPUTS) - {'data/manifest.json'}:
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                original = b'value\n'
                records[name] = {'sha256': b.hashlib.sha256(original).hexdigest(), 'bytes': len(original)}
                path.write_bytes(original)
            b.write_json(root / 'data/manifest.json', {'files': records})
            target = root / 'configs/protocol.json'
            target.write_bytes(b'value\r\n')
            with self.assertRaisesRegex(ValueError, 'checksum mismatch'):
                b.validate_input_hashes(root)
            _, check = b.validate_input_hashes(root, True)
            self.assertEqual(check['lf_equivalent_count'], 1)
            target.write_bytes(b'CHANGED\r\n')
            with self.assertRaisesRegex(ValueError, 'checksum mismatch'):
                b.validate_input_hashes(root, True)

    def test_manifest_path_cannot_escape_root(self):
        for name in ['../outside', '/outside', 'C:/outside', 'data/../../outside', 'data\\file.csv']:
            with self.subTest(name=name), self.assertRaises(ValueError):
                b.inside(self.root, name)

    def test_test_labels_and_predictors_are_never_used(self):
        with tempfile.TemporaryDirectory(prefix='b_heldout_') as directory:
            root = Path(directory)
            (root / 'data/processed').mkdir(parents=True)
            (root / 'data/splits').mkdir(parents=True)
            shutil.copy2(self.root / 'data/splits/split_manifest.csv', root / 'data/splits/split_manifest.csv')
            with (root / 'data/splits/split_manifest.csv').open(encoding='utf-8', newline='') as f:
                test_ids = {r['row_id'] for r in csv.DictReader(f) if r['dataset_split'] == 'test'}
            with (self.root / 'data/processed/students.csv').open(encoding='utf-8', newline='') as f:
                reader = csv.DictReader(f)
                fieldnames, rows = reader.fieldnames, list(reader)
            for row in rows:
                if row['row_id'] in test_ids:
                    for key in row:
                        if key != 'row_id':
                            row[key] = 'HELD_OUT_MUST_NEVER_BE_USED'
            with (root / 'data/processed/students.csv').open('w', encoding='utf-8', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)
            actual = b.read_training_only(root, self.features)
            pd.testing.assert_frame_equal(actual, self.train)
            self.assertTrue(set(actual.row_id).isdisjoint(test_ids))

    def test_exact_supplied_fold_membership(self):
        folds = self.train.cv_fold.to_numpy()
        validation_ids = []
        for fold, (fit_idx, val_idx) in enumerate(PredefinedSplit(folds).split()):
            self.assertTrue(np.all(folds[val_idx] == fold))
            self.assertTrue(np.all(folds[fit_idx] != fold))
            self.assertTrue(set(fit_idx).isdisjoint(val_idx))
            self.assertEqual(len(fit_idx) + len(val_idx), 3539)
            validation_ids.extend(self.train.iloc[val_idx].row_id)
        self.assertEqual(len(validation_ids), 3539)
        self.assertEqual(len(validation_ids), len(set(validation_ids)))
        self.assertEqual(set(validation_ids), set(self.train.row_id))

    def test_scaler_and_categories_learn_only_from_fit_fold(self):
        fit = self.train.loc[self.train.cv_fold != 0]
        names = self.features['enrollment']
        pipe = b.build_pipeline(self.features, 'enrollment', 'logistic_regression')
        pipe.fit(fit[names], fit.Target)
        prep = pipe.named_steps['preprocess']
        numeric = [c for c in names if c in self.features['numeric_features']]
        expected = fit[numeric].mean().to_numpy()
        np.testing.assert_allclose(prep.named_transformers_['numeric'].mean_, expected)
        self.assertFalse(np.allclose(expected, self.train[numeric].mean().to_numpy()))
        categorical = [c for c in names if c in self.features['categorical_features']]
        for name, learned in zip(categorical, prep.named_transformers_['categorical'].categories_):
            np.testing.assert_array_equal(learned, np.sort(fit[name].unique()))

    def test_unknown_categories_and_probability_labels(self):
        fit = self.train.loc[self.train.cv_fold != 0]
        names = self.features['enrollment']
        pipe = b.build_pipeline(self.features, 'enrollment', 'decision_tree')
        pipe.fit(fit[names], fit.Target)
        example = self.train.loc[self.train.cv_fold == 0, names].iloc[:3].copy()
        example['Course'] = -999999
        prob = pipe.predict_proba(example)
        self.assertEqual(list(pipe.classes_), b.LABELS)
        self.assertEqual(prob.shape, (3, 3))
        self.assertTrue(np.isfinite(prob).all())
        np.testing.assert_allclose(prob.sum(axis=1), 1)

    def test_missing_class_metrics_are_explicit(self):
        scores = b.compute_metrics(b.LABELS, ['Graduate'] * 3)
        self.assertAlmostEqual(scores['macro_f1'], 1 / 6)
        self.assertEqual(scores['dropout_precision'], 0)
        self.assertEqual(scores['dropout_recall'], 0)

    def test_serialized_majority_pipeline_is_portable(self):
        X, y = self.train[self.features['enrollment']], self.train.Target
        pipe = b.build_pipeline(self.features, 'enrollment', 'majority_baseline')
        with tempfile.TemporaryDirectory(prefix='b_model_') as directory:
            artifact = b.save_model(Path(directory), clone(pipe), X, y, 'baseline.joblib')
            self.assertEqual(artifact['classes'], b.LABELS)
            self.assertEqual(artifact['training_only_serialization_smoke_check'], 'PASS')
            model = b.joblib.load(Path(directory) / artifact['path'])
            self.assertEqual(set(model.predict(X.iloc[:20])), {'Graduate'})


if __name__ == '__main__':
    unittest.main(verbosity=2)
