# SPDX-License-Identifier: Apache-2.0
"""Same-fit byte export and ownership cleanup; native fit gates are separate."""
import ctypes
from types import SimpleNamespace

import pytest

from mojolearn import _serialize
from mojolearn._array import Array
from mojolearn._forest_protocol import _export_fit_result


@pytest.fixture
def export_native():
    # Ragged forest, signed zero and Float32 subnormal bytes in live/dead slots.
    fields = ([0, 3, 4], [1, -1, -1, -1], [0.25, -0.0, 2**-149, 1.5],
              [1, -1, -1, -1], [-0.0, 0.125, 0.75, 2**-149])
    arrays = [Array.from_list(v, t) for v, t in zip(fields, ('<i4','<i4','<f4','<i4','<f4'))]
    live, releases = {17}, []
    def export(handle, *args):
        assert handle in live
        assert args[-1] == [2, 4, 1]
        for destination, source in zip(args[:-1], arrays):
            ctypes.memmove(destination, source.tobytes(), source.nbytes)
    def legacy(handle):
        assert handle in live
        return [*fields, [2, 1, 7]]
    def release(handle):
        if handle not in live:
            raise ValueError('unknown or released handle')
        releases.append(handle)
        live.remove(handle)
    native = SimpleNamespace(forest_export=export, forest_export_legacy=legacy,
                             forest_export_release=release)
    return native, [17, 2, 4, 1, [2, 1, 7]], arrays, releases


def test_same_fit_export_bytes_and_archive_hash(export_native, tmp_path):
    native, descriptor, expected, released = export_native
    result = _export_fit_result(native, descriptor, compare_legacy=True)
    assert released == [17]
    assert result[-1] == [2, 1, 7]
    assert [v.tobytes() for v in result[:5]] == [v.tobytes() for v in expected]
    keys = ['offsets', 'colid', 'quesval', 'left_child', 'leaves']
    a, b = tmp_path/'old.npz', tmp_path/'new.npz'
    _serialize.write_npz(a, dict(zip(keys, expected)))
    _serialize.write_npz(b, dict(zip(keys, result[:5])))
    assert a.read_bytes() == b.read_bytes()
    with pytest.raises(ValueError, match='released'):
        native.forest_export_release(17)


@pytest.mark.parametrize('failure', ['export', 'diagnostic', 'allocation', 'counts', 'metadata'])
def test_failure_releases_owned_fit(export_native, monkeypatch, failure):
    native, descriptor, _, released = export_native
    def fail(*args, **kwargs):
        raise RuntimeError('injected failure')
    if failure == 'export':
        native.forest_export = fail
    elif failure == 'diagnostic':
        native.forest_export_legacy = fail
    elif failure == 'allocation':
        monkeypatch.setattr('mojolearn._buffer.empty', fail)
    elif failure == 'counts':
        descriptor[2] = 2147483648
    else:
        descriptor[4] = [3, 1]
    with pytest.raises((RuntimeError, ValueError)):
        _export_fit_result(native, descriptor, compare_legacy=True)
    assert released == [17]


def test_diagnostic_sabotage_is_detected(export_native):
    native, descriptor, _, released = export_native
    original = native.forest_export
    def corrupt(handle, *args):
        original(handle, *args)
        ctypes.c_int32.from_address(args[1]).value = 999
    native.forest_export = corrupt
    with pytest.raises(RuntimeError, match='byte mismatch: _colid'):
        _export_fit_result(native, descriptor, compare_legacy=True)
    assert released == [17]


@pytest.mark.parametrize('selection', [None, 'into', 'verify'])
def test_candidate_routes_same_fit_and_retains_exported_storage(export_native, monkeypatch, selection):
    from mojolearn._forest_protocol import _forest_fit_function, _forest_fit_arrays
    native, descriptor, _, released = export_native
    calls = []
    def fit(*args):
        calls.append(args)
        return descriptor
    native.et_classifier_fit_export = fit
    if selection is None:
        monkeypatch.delenv('MOJOLEARN_FOREST_EXPORT', raising=False)
    else:
        monkeypatch.setenv('MOJOLEARN_FOREST_EXPORT', selection)
    result = _forest_fit_function(native, 'et_classifier_fit')(111, 222, [3, 2])
    assert calls == [(111, 222, [3, 2])]
    assert released == [17]
    attached = _forest_fit_arrays(result)
    assert all(a is b for a, b in zip(attached[:5], result[:5]))


def test_legacy_override_unchanged_and_invalid_selection_refused(monkeypatch):
    from mojolearn._forest_protocol import _forest_fit_function
    entry = object()
    native = SimpleNamespace(rf_regressor_fit=entry)
    monkeypatch.setenv('MOJOLEARN_FOREST_EXPORT', 'legacy')
    assert _forest_fit_function(native, 'rf_regressor_fit') is entry
    monkeypatch.setenv('MOJOLEARN_FOREST_EXPORT', 'typo')
    with pytest.raises(ValueError, match='legacy, into or verify'):
        _forest_fit_function(native, 'rf_regressor_fit')
