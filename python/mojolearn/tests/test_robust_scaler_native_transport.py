# SPDX-License-Identifier: Apache-2.0
"""Preserve public scaler output ownership and established arena transports."""
import gc
from contextlib import nullcontext
import mmap
import re
import numpy as np
import pytest
import mojolearn as ml
from mojolearn import _arena_io, _backend
from mojolearn._array import Array
from mojolearn._verify_native_runtime import audit_native_call


def _ranges():
    binding = _backend.binding('_mojolearn_x_prep', 'identical')
    try:
        binding.x_prep_run_ranges
        binding.x_prep_dev_live
    except (AttributeError, ImportError):
        pytest.skip('GPU ranges/store ABI is not applicable to the CPU facade')
    return binding


@pytest.mark.parametrize('off', ['', 'work', 'mapped', 'view', 'all'])
@pytest.mark.parametrize('ranges', ['0', '1'])
def test_transport_switches_preserve_full_public_result(off, ranges, monkeypatch):
    x = np.arange(16400, dtype='f4').reshape(2050, 8)
    x[::7, 3] = np.nan
    monkeypatch.setenv('MOJOLEARN_XPREP_R3_OFF', 'all')
    monkeypatch.setenv('MOJOLEARN_ARENA_RANGES', '0')
    expected = ml.RobustScaler().fit_transform(x)
    monkeypatch.setenv('MOJOLEARN_XPREP_R3_OFF', off)
    monkeypatch.setenv('MOJOLEARN_ARENA_RANGES', ranges)
    actual = audit_native_call(ml.RobustScaler().fit_transform, x).assert_native()
    assert actual.tobytes() == expected.tobytes()


@pytest.mark.parametrize('rows', [False, True])
def test_resident_snapshot_counters_and_slot_lifetime_are_preserved(rows, monkeypatch):
    binding = _ranges()
    monkeypatch.setenv('MOJOLEARN_ARENA_RANGES', '1')
    monkeypatch.delenv('MOJOLEARN_XPREP_R3_OFF', raising=False)
    base = Array.from_list([[1.,2.],[3.,4.],[5.,6.]], '<f4')
    source = base
    calls = []
    if rows:
        def forbidden(*args):
            calls.append(True)
            raise AssertionError('Python gather ran')
        source = _arena_io.DeviceRows(base, Array.from_list([0,2], '<i8'), forbidden)
    initial = int(binding.x_prep_dev_live())
    with _arena_io.resident(min_words=1) as scope:
        model = audit_native_call(ml.RobustScaler().fit, source).assert_native()
        before = model.center_.tobytes(), model.scale_.tobytes()
        first = scope.stats()['x_prep']
        base._mv[0] = 100.
        audit_native_call(model.fit, source).assert_native()
        second = scope.stats()['x_prep']
        assert (model.center_.tobytes(), model.scale_.tobytes()) == before
        assert first == (2, 0) and second == (3, 1)
        # Per-call gather slots have already been freed; only cache uploads
        # remain. Scope cleanup is application/test control outside the call.
        assert int(binding.x_prep_dev_live()) == initial + 3
        if rows:
            assert source._host is None and not calls
    assert int(binding.x_prep_dev_live()) == initial


@pytest.mark.parametrize('resident', [False, True])
def test_device_row_failure_releases_operation_slots_and_retains_scope_owner(resident, monkeypatch):
    binding = _ranges()
    monkeypatch.setenv('MOJOLEARN_ARENA_RANGES', '1')
    base = Array.from_list([[1.,2.],[3.,4.]], '<f4')
    source = _arena_io.DeviceRows(base, Array.from_list([99], '<i8'), None)
    initial = int(binding.x_prep_dev_live())
    with _arena_io.resident(min_words=1) if resident else nullcontext():
        audit = audit_native_call(ml.RobustScaler().fit, source)
        assert isinstance(audit.exception, ValueError), audit.report()
        assert 'index out of range' in str(audit.exception)
        report = audit.report()
        assert not report['python_frames'] and not report['forbidden_native_calls']
        assert int(binding.x_prep_dev_live()) == initial + int(resident)
    assert int(binding.x_prep_dev_live()) == initial


@pytest.mark.parametrize('primary_failure', [False, True])
def test_every_owned_release_is_attempted_and_primary_error_is_retained(primary_failure, monkeypatch):
    binding = _ranges()
    monkeypatch.setenv('MOJOLEARN_ARENA_RANGES', '1')
    from mojolearn import _expansion_prep
    entries = _expansion_prep._ROBUST_CONTEXT['scaler_bindings']['identical']
    base = Array.from_list([[1.,2.],[3.,4.]], '<f4')
    source = _arena_io.DeviceRows(base, Array.from_list([99 if primary_failure else 0], '<i8'), None)
    initial = int(binding.x_prep_dev_live())
    slots = []
    try:
        with monkeypatch.context() as patch:
            # Native ABI fault: this fixed native zero-argument query rejects
            # the slot argument. No Python cleanup callback is installed.
            patch.setitem(entries, 'x_prep_dev_free', binding.x_prep_numeric_mode)
            audit = audit_native_call(ml.RobustScaler().fit, source)
        assert isinstance(audit.exception, RuntimeError), audit.report()
        text = str(audit.exception)
        slots = [int(value) for value in re.findall(r'slot (\d+):', text)]
        assert len(slots) == (1 if primary_failure else 2), text
        assert len(set(slots)) == len(slots)
        if primary_failure:
            assert 'index out of range' in text and 'cleanup also failed' in text
        report = audit.report()
        assert not report['python_frames'] and not report['forbidden_native_calls']
        assert int(binding.x_prep_dev_live()) == initial + len(slots)
    finally:
        for slot in slots:
            binding.x_prep_dev_free(slot)
    assert int(binding.x_prep_dev_live()) == initial


def _owns_mapping(value):
    while value is not None:
        if isinstance(value, mmap.mmap):
            return True
        value = value.obj if isinstance(value, memoryview) else getattr(value, 'base', None)
    return False


@pytest.mark.parametrize('off', ['', 'mapped', 'view'])
def test_large_transform_output_retains_mapping_until_consumed(off, monkeypatch):
    monkeypatch.setenv('MOJOLEARN_XPREP_R3_OFF', off)
    model = ml.RobustScaler().fit(np.array([[0.],[2.],[4.]], dtype='f4'))
    x = np.arange(1 << 20, dtype='f4').reshape(-1, 1)
    result = audit_native_call(model.transform, x).assert_native()
    del model, x
    gc.collect()
    assert _owns_mapping(result) == (off == '')
    assert result[0, 0] == -1 and result[-1, 0] == ((1 << 20)-3)/2


@pytest.mark.parametrize('hook', ['dict', 'finalizer', 'base'])
def test_resident_class_hooks_are_refused_before_cache_allocation(hook, monkeypatch):
    _ranges()
    monkeypatch.setenv('MOJOLEARN_ARENA_RANGES', '1')
    calls = []
    def forbidden(*args):
        calls.append(True)
        raise AssertionError('cache metadata hook ran')
    with _arena_io.resident(min_words=1):
        cls = _arena_io.DeviceCache
        if hook == 'dict':
            class ForeignCache:
                __dict__ = property(forbidden)
            from mojolearn import _expansion_prep
            monkeypatch.setitem(_expansion_prep._ROBUST_CONTEXT, 'arena_cache_class', ForeignCache)
        elif hook == 'finalizer':
            monkeypatch.setattr(cls, '__del__', forbidden, raising=False)
        else:
            class Foreign:
                __del__ = forbidden
            # Changing __bases__ is rejected by CPython for object-backed
            # heap classes on some versions; test an existing cache class
            # subclass replacement through scope metadata instead.
            class ForeignCache(cls, Foreign):
                pass
            from mojolearn import _expansion_prep
            monkeypatch.setitem(_expansion_prep._ROBUST_CONTEXT, 'arena_cache_class', ForeignCache)
        audit = audit_native_call(ml.RobustScaler().fit, np.ones((3,2), dtype='f4'))
        assert isinstance(audit.exception, TypeError), audit.report()
        assert not audit.report()['python_frames'] and not calls
