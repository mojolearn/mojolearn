# SPDX-License-Identifier: Apache-2.0
"""Fault interception observes descriptor storage before any kernel executes."""
import ctypes
import numpy as np
import pytest
import mojolearn as ml
from mojolearn import _backend, _expansion_prep


@pytest.mark.parametrize('operation', ['fit', 'transform'])
def test_range_descriptors_and_program_survive_native_call_envelope(operation, monkeypatch):
    # This deliberately substitutes a test controller, not a native product
    # path. It reads the ABI buffers and stops before GPU execution; actual
    # public numerical/native-boundary tests qualify the restored route.
    monkeypatch.setenv('MOJOLEARN_ARENA_RANGES', '1')
    monkeypatch.setenv('MOJOLEARN_XPREP_R3_OFF', 'work')
    monkeypatch.setenv('MOJOLEARN_XPREP_DIRECT', '0')
    entries = _expansion_prep._ROBUST_CONTEXT['scaler_bindings'][_backend._DEFAULT_MODE]
    captured = []
    def capture(arena, program, output, sizes, ranges):
        ia, ni, oa, no = map(int, ranges)
        assert 0 < ni <= 8 and 0 < no <= 8
        captured.append((list(sizes), list((ctypes.c_int32*(3*ni)).from_address(ia)),
                         list((ctypes.c_int32*(4*no)).from_address(oa)),
                         ctypes.c_int32.from_address(int(program)).value))
        raise RuntimeError('descriptor test stopped before kernel execution')
    monkeypatch.setitem(entries, 'x_prep_run_ranges', capture)
    model = ml.RobustScaler()
    if operation == 'transform':
        vars(model).update(center_=None, scale_=np.ones(5, dtype='f4'),
                           numeric_mode_=_backend._DEFAULT_MODE, n_features_in_=5,
                           _mojolearn_fitted=True)
    with pytest.raises(ValueError, match='descriptor test stopped'):
        getattr(model, operation)(np.arange(95, dtype='f4').reshape(19,5))
    if operation == 'fit':
        assert captured == [([273,0,0,5], [0,95,-1,220,223,-1],
                             [95,220,-1,1,223,273,-1,1], 0)]
    else:
        assert captured == [([195,0,0,1], [0,100,-1], [100,195,-1,1], 3)]
