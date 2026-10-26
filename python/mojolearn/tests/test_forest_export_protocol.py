# SPDX-License-Identifier: Apache-2.0
"""Same-fit model export through the real public fits (small inputs): both
MOJOLEARN_FOREST_EXPORT selections (into, verify) give the same model bytes and
the same archive, and an unknown selection is refused by name."""
import numpy as np
import pytest

from mojolearn import (ExtraTreesClassifier, ExtraTreesRegressor,
                       RandomForestClassifier, RandomForestRegressor)

CLASSES = [ExtraTreesClassifier, ExtraTreesRegressor,
           RandomForestClassifier, RandomForestRegressor]
_ARRAYS = ('_offsets', '_colid', '_quesval', '_left_child', '_leaves')


def _fit(cls):
    rng = np.random.default_rng(0)
    X = rng.random((40, 4), dtype=np.float32)
    y = ((X[:, 0] > 0.5).astype(np.int32) if cls._estimator_type == 'classifier'
         else (2.0 * X[:, 0] + X[:, 1]).astype(np.float32))
    return cls(n_estimators=3, max_depth=3, random_state=0, inference_engine='sequential').fit(X, y)


@pytest.mark.parametrize('cls', CLASSES)
def test_every_selection_exports_the_same_fit(cls, monkeypatch, tmp_path):
    models, archives = {}, {}
    for selection in ('into', 'verify'):
        monkeypatch.setenv('MOJOLEARN_FOREST_EXPORT', selection)
        model = _fit(cls)
        models[selection] = [getattr(model, name).tobytes() for name in _ARRAYS]
        path = tmp_path / f'{selection}.npz'
        model.save(path)
        archives[selection] = path.read_bytes()
    monkeypatch.delenv('MOJOLEARN_FOREST_EXPORT', raising=False)
    default = _fit(cls)
    assert [getattr(default, name).tobytes() for name in _ARRAYS] == models['into']
    assert models['into'] == models['verify']
    assert archives['into'] == archives['verify']


@pytest.mark.parametrize('cls', CLASSES)
@pytest.mark.parametrize('selection', ('typo', 'legacy'))
def test_invalid_selection_refused(cls, selection, monkeypatch):
    monkeypatch.setenv('MOJOLEARN_FOREST_EXPORT', selection)
    with pytest.raises(ValueError, match='into or verify'):
        _fit(cls)
