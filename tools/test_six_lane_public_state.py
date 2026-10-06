"""Invented archive/schema fixtures only; no product imports or model execution."""
from types import SimpleNamespace
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from bench_board_state import normalize_public_fitted_state, public_fitted_state_paths
from six_lane_evidence import capture_model


def pca_archive(name='PCA'):
    return dict(format=np.array(['mojolearn-pca-1']), estimator=np.array([name]),
                numeric_mode=np.array(['identical']), svd_solver=np.array(['full']),
                components=np.zeros((2, 3), dtype='<f4'), mean=np.zeros(3, dtype='<f4'),
                singular_values=np.zeros(2, dtype='<f4'), explained_variance=np.zeros(2, dtype='<f4'),
                explained_variance_ratio=np.zeros(2, dtype='<f4'), noise_variance=np.zeros(1, dtype='<f8'),
                meta=np.array([2, 3, 5, 0], dtype='<i8'))


def kmeans_archive(name='KMeans'):
    return dict(format=np.array(['mojolearn-kmeans-1']), estimator=np.array([name]),
                numeric_mode=np.array(['identical']), metric=np.array(['euclidean']), init=np.array(['array']),
                centers=np.zeros((2, 3), dtype='<f4'), labels=np.zeros(5, dtype='<i4'),
                meta=np.array([2, 3, 5, 4, 0, 2, 300, 1, 0], dtype='<i8'),
                reals=np.array([0, 1, 1, 0.0001, 2], dtype='<f8'))


def fixture_model(name, module, arrays, error=None):
    def save(self, path):
        self.saves += 1
        if error:
            raise ValueError(error)
        np.savez(path, **arrays)

    def forbidden(self, *args, **kwargs):
        raise AssertionError('No estimator operation is permitted in this fixture')

    cls = type(name, (), {'__module__': module, 'save': save, 'saves': 0,
                          'fit': forbidden, 'predict': forbidden, 'transform': forbidden})
    return cls()


class PublicStateMetadataTest(unittest.TestCase):
    def test_retention_keeps_original_export_and_typed_state_after_capture(self):
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / 'scored-A.model-state.json'
            model = fixture_model('PCA', 'mojolearn.decomposition', pca_archive())
            result = capture_model(SimpleNamespace(est=model), retain_path=destination)
            self.assertEqual(result['status'], 'CAPTURED')
            exported = result['provenance']['retained_export']
            raw = Path(exported['path']).read_bytes()
            self.assertEqual(exported['sha256'], hashlib.sha256(raw).hexdigest())
            self.assertEqual(exported['bytes'], len(raw))
            self.assertEqual(result['retained_values'], str(destination))
            tree = json.loads(destination.read_text())['tree']
            arrays = [value for _, value in tree['dict'] if 'array' in value]
            self.assertTrue(arrays)
            for leaf in arrays:
                self.assertEqual(leaf['sha256'], hashlib.sha256(Path(leaf['array']).read_bytes()).hexdigest())
            duplicate = capture_model(SimpleNamespace(est=model), retain_path=destination)
            self.assertEqual(duplicate['status'], 'UNAVAILABLE')
            self.assertEqual(Path(exported['path']).read_bytes(), raw)

    def test_partial_export_retained_without_upgrading_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            model = fixture_model('OtherPCA', 'mojolearn.testing', pca_archive())
            result = capture_model(SimpleNamespace(est=model),
                                   retain_path=Path(directory) / 'partial.json')
            self.assertEqual(result['status'], 'UNAVAILABLE')
            exported = result['partial_export']['retained_export']
            self.assertEqual(hashlib.sha256(Path(exported['path']).read_bytes()).hexdigest(), exported['sha256'])

    def test_state_dict_values_retained_without_a_save_export(self):
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / 'state.json'
            model = SimpleNamespace(state_dict=lambda: {'weights': np.zeros(2, dtype='<f4')})
            result = capture_model(SimpleNamespace(est=model), ['$.weights'], retain_path=destination)
            self.assertEqual(result['completeness'], 'complete_declared_scope')
            leaf = json.loads(destination.read_text())['tree']['dict'][0][1]
            self.assertEqual(hashlib.sha256(Path(leaf['array']).read_bytes()).hexdigest(), leaf['sha256'])

    def capture(self, name='PCA', arrays=None, paths=None, module=None, error=None):
        if arrays is None:
            arrays = pca_archive(name) if name.endswith('PCA') else kmeans_archive(name)
        if module is None:
            module = ('mojolearn._classical_host' if name.startswith('Host') else
                      'mojolearn.decomposition' if name == 'PCA' else 'mojolearn.cluster')
        model = fixture_model(name, module, arrays, error)
        result = capture_model(SimpleNamespace(est=model), paths)
        self.assertEqual(model.saves, 1)
        return result

    def test_pca_complete_contract_has_every_declared_typed_leaf(self):
        result = self.capture()
        self.assertEqual(result['status'], 'CAPTURED')
        self.assertEqual(result['completeness'], 'complete_declared_scope')
        self.assertEqual(result['missing_state'], [])
        self.assertEqual({r['path'] for r in result['manifest']}, set(public_fitted_state_paths('PCA')))
        self.assertEqual(result['provenance']['contract'], 'mojolearn.public-fitted-state/pca-1')
        self.assertIn('complete_export_sha256', result['provenance'])

    def test_kmeans_captures_labels_diagnostics_and_controls(self):
        result = self.capture('KMeans')
        leaves = {r['path']: r for r in result['manifest']}
        self.assertEqual(set(leaves), set(public_fitted_state_paths('KMeans')))
        self.assertEqual(leaves['$.labels']['dtype'], '<i4')
        self.assertEqual(leaves['$.meta']['shape'], [9])
        self.assertEqual(leaves['$.reals']['shape'], [5])
        self.assertEqual(result['completeness'], 'complete_declared_scope')

    def test_normalization_changes_only_non_numerical_class_metadata(self):
        arrays = pca_archive('HostPCA')
        normalized = normalize_public_fitted_state(arrays, 'PCA', 'HostPCA')
        self.assertEqual(normalized['estimator'], 'PCA')
        for key in arrays:
            if key != 'estimator':
                self.assertIs(normalized[key], arrays[key])
        result = self.capture('HostPCA', arrays)
        self.assertEqual(result['provenance']['normalized_metadata']['estimator']['saved'], 'HostPCA')

    def test_known_host_subclass_capture_uses_shared_logical_scope(self):
        base = self.capture('KMeans')
        host = self.capture('HostKMeans')
        self.assertEqual(base['scope'], host['scope'])
        self.assertEqual(base['manifest'], host['manifest'])
        self.assertEqual(base['sha256'], host['sha256'])
        self.assertNotEqual(base['provenance']['complete_export_sha256'], host['provenance']['complete_export_sha256'])

    def test_missing_extra_or_unreviewed_export_never_becomes_complete(self):
        for mutation in ('missing', 'extra', 'format', 'wrong-owner', 'object'):
            with self.subTest(mutation=mutation):
                arrays = pca_archive()
                if mutation == 'missing':
                    del arrays['explained_variance_ratio']
                elif mutation == 'extra':
                    arrays['new_fitted_state'] = np.zeros(1, dtype='<f4')
                elif mutation == 'format':
                    arrays['format'] = np.array(['mojolearn-pca-future'])
                elif mutation == 'wrong-owner':
                    arrays['estimator'] = np.array(['OtherPCA'])
                else:
                    arrays['mean'] = np.array([object()], dtype=object)
                result = self.capture(arrays=arrays)
                self.assertEqual(result['status'], 'UNAVAILABLE')
                self.assertTrue(result['missing_state'])
                self.assertNotIn('sha256', result)

    def test_wrong_shape_dtype_and_control_codes_are_refused(self):
        arrays = pca_archive()
        arrays['components'] = np.zeros((3, 2), dtype='<f4')
        self.assertEqual(self.capture(arrays=arrays)['status'], 'UNAVAILABLE')
        arrays = pca_archive()
        arrays['components'] = arrays['components'].astype('<f8')
        self.assertEqual(self.capture(arrays=arrays)['status'], 'UNAVAILABLE')
        arrays = kmeans_archive()
        arrays['meta'][4] = 1
        self.assertEqual(self.capture('KMeans', arrays)['status'], 'UNAVAILABLE')

    def test_partial_or_incompatible_recipe_declaration_stays_unqualified(self):
        for paths in ([], ['$.components'], public_fitted_state_paths('PCA') + ['$.unavailable']):
            with self.subTest(paths=paths):
                result = self.capture(paths=paths)
                self.assertEqual(result['status'], 'CAPTURED')
                self.assertEqual(result['completeness'], 'scope_not_qualified')
                self.assertIn('Recipe model-state paths differ', result['reason'])
        self.assertEqual(self.capture(paths=public_fitted_state_paths('PCA'))['completeness'], 'complete_declared_scope')

    def test_ols_prediction_archive_remains_partial(self):
        arrays = dict(format=np.array(['mojolearn-linear-1']), estimator=np.array(['LinearRegression']),
                      numeric_mode=np.array(['identical']), coef=np.zeros(3, dtype='<f4'),
                      intercept=np.zeros(1, dtype='<f8'), meta=np.array([3, 1], dtype='<i8'))
        model = fixture_model('LinearRegression', 'mojolearn.linear_model', arrays)
        result = capture_model(SimpleNamespace(est=model))
        self.assertEqual(result['status'], 'UNAVAILABLE')
        self.assertEqual(model.saves, 1)
        self.assertEqual(result['missing_state'], ['retained training mean _x_mean', 'retained training mean _y_mean'])
        self.assertEqual(result['partial_export']['status'], 'ok')
        self.assertNotIn('sha256', result)

    def test_export_failure_does_not_invent_state_or_retry(self):
        result = self.capture(error='fixture exporter failure')
        self.assertEqual(result['status'], 'UNAVAILABLE')
        self.assertIn('fixture exporter failure', result['reason'])
        self.assertNotIn('sha256', result)

    def test_generic_save_is_not_upgraded_by_a_matching_field_set(self):
        model = fixture_model('OtherPCA', 'mojolearn.testing', pca_archive())
        result = capture_model(SimpleNamespace(est=model))
        self.assertEqual(result['status'], 'UNAVAILABLE')
        self.assertIn('partial_export', result)

    def test_existing_state_dict_contract_is_preserved(self):
        model = SimpleNamespace(state_dict=lambda: {'weights': np.zeros(2, dtype='<f4')})
        result = capture_model(SimpleNamespace(est=model), ['$.weights'])
        self.assertEqual(result['scope'], 'public state_dict')
        self.assertEqual(result['completeness'], 'complete_declared_scope')


if __name__ == '__main__':
    unittest.main()
