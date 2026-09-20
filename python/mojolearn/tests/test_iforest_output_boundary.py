"""Isolation Forest allocates only the result buffer selected by `want`."""
import ctypes
from types import SimpleNamespace

import numpy as np
import pytest

from mojolearn import IsolationForest
from mojolearn._buffer import as_f32_c


@pytest.mark.parametrize("method,want", [
    ("score_samples", 0), ("decision_function", 1), ("predict", 2),
])
def test_iforest_passes_null_for_unused_output(monkeypatch, method, want):
    seen = []

    def run(train, query, out_f32, out_i32, info, params):
        seen.append((out_f32, out_i32, params[-1]))
        metadata = (ctypes.c_double * 3).from_address(info)
        metadata[:] = (-0.5, 4.0, 2.0)
        if params[-1] == 2:
            result = (ctypes.c_int32 * params[2]).from_address(out_i32)
            result[:] = (-1, 1, 1)
        else:
            result = (ctypes.c_float * params[2]).from_address(out_f32)
            result[:] = (-0.75, -0.25, -0.125)
        return params[2]

    model = IsolationForest(n_estimators=2, max_samples=4, random_state=7)
    model._x = as_f32_c(np.ones((4, 2), np.float32), name="X")[0]
    model.n_features_in_ = 2
    monkeypatch.setattr(model, "_bind", lambda *args: SimpleNamespace(iforest_run=run))
    result = getattr(model, method)(np.ones((3, 2), np.float32))
    assert len(result) == 3
    assert seen[0][2] == want
    if want == 2:
        assert seen[0][0] == 0 and seen[0][1] != 0
    else:
        assert seen[0][0] != 0 and seen[0][1] == 0
