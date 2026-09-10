# SPDX-License-Identifier: Apache-2.0
"""Fold metadata matches sklearn; built-in CV never needs its dependencies."""
import os
import subprocess
import sys
import textwrap

import pytest

from mojolearn.model_selection import _clone, _default_folds


@pytest.mark.parametrize('labels', [
    [0, 1] * 11,
    ['z'] * 8 + ['a'] * 7 + ['m'] * 6,
    [2, 0, 1, 1, 2, 0] * 4,
    [0] * 3 + [1] * 9,
])
@pytest.mark.parametrize('splits', [2, 3])
def test_stratified_metadata_matches_reference(labels, splits):
    reference = pytest.importorskip('sklearn.model_selection')
    expected = list(reference.StratifiedKFold(splits).split(labels, labels))
    actual = list(_default_folds(labels, splits, True))
    assert actual == [(train.tolist(), test.tolist()) for train, test in expected]


def test_kfold_metadata_matches_reference():
    reference = pytest.importorskip('sklearn.model_selection')
    for n in [7, 11, 22]:
        for splits in [2, 3, 5]:
            expected = reference.KFold(splits).split(range(n))
            assert list(_default_folds(list(range(n)), splits, False)) == [
                (train.tolist(), test.tolist()) for train, test in expected]


def test_pipeline_clone_keeps_external_protocol():
    pipeline = pytest.importorskip('sklearn.pipeline')
    preprocessing = pytest.importorskip('sklearn.preprocessing')
    original = pipeline.Pipeline([('scale', preprocessing.StandardScaler())])
    cloned = _clone(original)
    assert cloned is not original
    assert cloned.steps[0][1] is not original.steps[0][1]
    assert cloned.get_params()['scale__with_mean'] is True


def test_default_and_explicit_cv_without_numpy_or_sklearn():
    # Fresh process guarantees an already-imported dependency cannot mask reach.
    program = textwrap.dedent('''
        import builtins, ctypes
        original_import = builtins.__import__
        def blocked(name, *args, **kwargs):
            if name.split('.')[0] in ('numpy', 'sklearn'):
                raise AssertionError('unexpected dependency: ' + name)
            return original_import(name, *args, **kwargs)
        builtins.__import__ = blocked
        from mojolearn import Array
        from mojolearn import _buffer
        from mojolearn.model_selection import cross_val_score
        calls = []
        def gather(src, dst, indices, source_rows, output_rows, row_bytes):
            for output, source in enumerate((ctypes.c_int64 * output_rows).from_address(indices)):
                assert 0 <= source < source_rows
                ctypes.memmove(dst + output * row_bytes, src + source * row_bytes, row_bytes)
        _buffer._NATIVE['gather_rows_bytes'] = gather
        class Estimator:
            _estimator_type = 'classifier'
            def __init__(self, setting=None): self.setting = setting
            def get_params(self, deep=True): return {'setting': self.setting}
            def fit(self, X, y):
                assert not hasattr(self, 'fitted_')
                assert self.setting == {'nested': [1, 2]}
                self.setting['nested'].append(3)
                self.fitted_ = True
                calls.append((X.tolist(), y.tolist()))
                return self
            def score(self, X, y):
                assert self.fitted_
                return 0.75
        X = Array.from_list([[float(i), float(i+100)] for i in range(12)], '<f4')
        y = Array.from_list([0, 1] * 6, '<i4')
        model = Estimator({'nested': [1, 2]})
        assert cross_val_score(model, X, y).tolist() == [0.75] * 5
        assert model.setting == {'nested': [1, 2]}
        assert not hasattr(model, 'fitted_')
        assert cross_val_score(model, X, y, cv=[([0, 1, 2, 3], [4, 5])]).tolist() == [0.75]
        class Splitter:
            def split(self, X, y, groups):
                assert groups == ['a'] * 12
                yield [0, 1], [8, 9]
        assert cross_val_score(model, X, y, cv=Splitter(), groups=['a'] * 12).tolist() == [0.75]
        assert calls[-1][0] == [[0., 100.], [1., 101.]]
    ''')
    result = subprocess.run([sys.executable, '-c', program], capture_output=True, text=True,
                            env=os.environ.copy())
    assert result.returncode == 0, result.stdout + result.stderr
