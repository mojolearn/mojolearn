# SPDX-License-Identifier: Apache-2.0
"""DEVIATION 3120: the `into=` forms of `ParallelByteLanguageModelTrainer`'s
exports (`export_raw`, `export_gradients`, `fold_export`) against their
no-argument forms, and the runner's `ExportBuffers`. Host mocks only: the
binding is a stand-in that writes a known byte pattern at the addresses it
is handed, so these tests prove what the Python side sends and refuses, not
what a device returns (the device proof is the H100 leg in
bench/results/byte_lm_export_fast_2026-09-25/)."""
import ctypes
import hashlib
import importlib.util
from pathlib import Path
import struct
import sys
import threading
import types

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
try:
    import mojolearn.parallel_training as pt
except ImportError:
    # a checkout without built bindings: load the modules under test
    # without the package's binding selection
    for name in [m for m in sys.modules if m == 'mojolearn' or m.startswith('mojolearn.')]:
        del sys.modules[name]
    pkg = types.ModuleType('mojolearn')
    pkg.__path__ = [str(ROOT / 'python' / 'mojolearn')]
    sys.modules['mojolearn'] = pkg
    import mojolearn.parallel_training as pt
from mojolearn._buffer import empty, flat_bytes
from mojolearn._byte_lm_config import ByteLanguageModelConfig

SHAPE = ByteLanguageModelConfig()
N, NT = SHAPE.n_total, SHAPE.n_tensors


class FakeBinding:
    """Writes a pattern that depends on the array and on `version`, so a
    reused buffer is seen to be overwritten, and refuses what the native
    slot table refuses (overlapping outputs)."""

    def __init__(self):
        self.version = 1
        self.calls = 0

    def _fill(self, address, count, salt, integer=False):
        kind = ctypes.c_int32 if integer else ctypes.c_float
        arr = np.ctypeslib.as_array((kind * count).from_address(address))
        if integer:
            arr[:] = (np.arange(count) + salt + self.version) % 2
        else:
            arr[:] = (np.arange(count, dtype=np.float32) * np.float32(0.5 + salt)
                      + np.float32(self.version)).astype(np.float32)

    def _spans(self, addresses, cells):
        spans = sorted((a, a + 4 * c) for a, c in zip(addresses, cells))
        for (a0, a1), (b0, b1) in zip(spans, spans[1:]):
            if b0 < a1:
                raise RuntimeError('byte LM: output overlaps another live span')

    def byte_lm_parallel_export(self, session, addresses, rank, gradients):
        self.calls += 1
        if gradients:
            assert len(addresses) == 1
            self._fill(addresses[0], N, 7)
        else:
            assert len(addresses) == 4
            self._spans(addresses, [N, N, N, NT])
            for i, a in enumerate(addresses[:3]):
                self._fill(a, N, i)
            self._fill(addresses[3], NT, 3, integer=True)
        return 5

    def byte_lm_parallel_shard_gradient(self, session, addresses, completed):
        raise AssertionError('not called here')

    def byte_lm_parallel_fold_export(self, session, addresses):
        self.calls += 1
        assert len(addresses) == 1
        self._fill(addresses[0], N, 11)
        return N


def trainer(binding=None):
    t = object.__new__(pt.ParallelByteLanguageModelTrainer)
    t.pool_optimizer, t.devices, t.logical_shards = False, (0,), 1  # the split-step form, which the fold needs
    t._state = {'completed_steps': 5}
    t._shape = SHAPE
    t._session = object()
    t._binding = binding or FakeBinding()
    t._closed = t._lost = False
    t._lock = threading.RLock()
    return t


def raw_bytes(x):
    if isinstance(x, (bytes, bytearray)):
        return bytes(x)
    if isinstance(x, np.ndarray):
        return x.tobytes()
    return bytes(flat_bytes(x))


def state_bufs(kind='numpy'):
    if kind == 'numpy':
        return dict(parameters=np.zeros(N, np.float32), m=np.zeros(N, np.float32),
                    v=np.zeros(N, np.float32), flags=np.zeros(NT, np.int32))
    return dict(parameters=empty((N,), '<f4'), m=empty((N,), '<f4'), v=empty((N,), '<f4'),
                flags=empty((NT,), '<i4'))


@pytest.mark.parametrize('kind', ['numpy', 'array'])
def test_export_raw_into_same_bytes_and_same_objects(kind):
    t = trainer()
    fresh = t.export_raw()
    bufs = state_bufs(kind)
    ids = {k: id(v) for k, v in bufs.items()}
    out = t.export_raw(into=bufs)
    assert out is bufs and {k: id(v) for k, v in out.items()} == ids
    for k in pt.STATE_ARRAYS:
        assert hashlib.sha256(raw_bytes(out[k])).digest() == hashlib.sha256(raw_bytes(fresh[k])).digest()
    # a reused buffer is overwritten by the next export
    t._binding.version = 2
    again = t.export_raw(into=bufs)
    assert raw_bytes(again['parameters']) == raw_bytes(t.export_raw()['parameters'])
    assert raw_bytes(again['parameters']) != raw_bytes(fresh['parameters'])


def test_into_allocates_nothing(monkeypatch):
    t = trainer()
    bufs, g, f = state_bufs(), np.zeros(N, np.float32), np.zeros(N, np.float32)

    def refuse(*a, **k):
        raise AssertionError('an into= export allocated')
    monkeypatch.setattr(pt, 'empty', refuse)
    assert t.export_raw(into=bufs) is bufs
    assert t.export_gradients(into=g) is g
    assert t.fold_export(into=f) is f


@pytest.mark.parametrize('kind', ['numpy', 'array'])
def test_gradients_and_fold_into_same_bytes(kind):
    t = trainer()
    g = np.zeros(N, np.float32) if kind == 'numpy' else empty((N,), '<f4')
    f = np.zeros(N, np.float32) if kind == 'numpy' else empty((N,), '<f4')
    assert raw_bytes(t.export_gradients(into=g)) == raw_bytes(t.export_gradients())
    fresh = t.fold_export()
    assert isinstance(fresh, bytes) and len(fresh) == 4 * N
    assert raw_bytes(t.fold_export(into=f)) == fresh


def bad_targets():
    ro = np.zeros(N, np.float32)
    ro.flags.writeable = False
    return [
        ('float64', np.zeros(N, np.float64), TypeError),
        ('int32 for float', np.zeros(N, np.int32), TypeError),
        ('bytearray', bytearray(4 * N), TypeError),
        ('one short', np.zeros(N - 1, np.float32), ValueError),
        ('one long', np.zeros(N + 1, np.float32), ValueError),
        ('two-dimensional', np.zeros((N, 1), np.float32), ValueError),
        ('strided', np.zeros(2 * N, np.float32)[::2], ValueError),
        ('read-only', ro, ValueError),
        ('bytes', bytes(4 * N), ValueError),
        ('not a buffer', [0.0] * N, TypeError),
    ]


@pytest.mark.parametrize('label,target,exc', bad_targets(), ids=[b[0] for b in bad_targets()])
def test_float_targets_refused_before_the_binding(label, target, exc):
    for call in ('gradients', 'fold', 'parameters', 'v'):
        t = trainer()
        with pytest.raises(exc):
            if call == 'gradients':
                t.export_gradients(into=target)
            elif call == 'fold':
                t.fold_export(into=target)
            else:
                bufs = state_bufs()
                bufs[call] = target
                t.export_raw(into=bufs)
        assert t._binding.calls == 0, (label, call)


@pytest.mark.parametrize('label,target,exc', [
    ('float32 flags', np.zeros(NT, np.float32), TypeError),
    ('int64 flags', np.zeros(NT, np.int64), TypeError),
    ('flags one short', np.zeros(NT - 1, np.int32), ValueError),
    ('flags sized like parameters', np.zeros(N, np.int32), ValueError),
])
def test_flags_refused(label, target, exc):
    t = trainer()
    bufs = state_bufs()
    bufs['flags'] = target
    with pytest.raises(exc):
        t.export_raw(into=bufs)
    assert t._binding.calls == 0


@pytest.mark.parametrize('into', [
    [np.zeros(N, np.float32)] * 4,
    dict(parameters=np.zeros(N, np.float32), m=np.zeros(N, np.float32), v=np.zeros(N, np.float32)),
    dict(state_bufs(), extra=np.zeros(N, np.float32)),
], ids=['list', 'missing flags', 'extra key'])
def test_into_keys_refused(into):
    t = trainer()
    with pytest.raises(ValueError):
        t.export_raw(into=into)
    assert t._binding.calls == 0


def test_same_buffer_twice_is_refused_by_the_slot_table():
    t = trainer()
    bufs = state_bufs()
    bufs['m'] = bufs['parameters']
    with pytest.raises(RuntimeError, match='overlaps'):
        t.export_raw(into=bufs)


def test_no_argument_forms_unchanged():
    t = trainer()
    raw = t.export_raw()
    assert set(raw) == set(pt.STATE_ARRAYS)
    assert raw['parameters'].shape == (N,) and raw['flags'].shape == (NT,)
    assert struct.unpack('<f', raw_bytes(raw['m'])[4:8])[0] == np.float32(1 + 1.5)
    assert raw_bytes(t.export_gradients())[:4] == struct.pack('<f', 1.0)


# ------------------------------------------------------------- the runner

def _runner():
    spec = importlib.util.spec_from_file_location('lm_segment_export_into', ROOT / 'tools' / 'lm_segment.py')
    mod = importlib.util.module_from_spec(spec)
    saved = sys.argv
    sys.argv = [saved[0]]
    try:
        spec.loader.exec_module(mod)
    finally:
        sys.argv = saved
    return mod


class OldTrainer:
    """A package that predates `into=`: the runner must call the old forms."""

    def __init__(self):
        self.t = trainer()

    def export_raw(self, *, rank=0):
        return self.t.export_raw()

    def export_gradients(self, *, rank=0):
        return self.t.export_gradients()


def test_runner_reuses_buffers_and_digests_are_unchanged():
    seg = _runner()
    t = trainer()
    ex = seg.ExportBuffers(t)
    assert ex.reuse
    first = ex.state(t)
    g1 = ex.gradients(t)
    for step in (2, 3):
        t._binding.version = step
        raw = ex.state(t)
        g = ex.gradients(t)
        assert raw is first and g is g1
        fresh, fresh_g = t.export_raw(), t.export_gradients()
        for scheme in (seg.SCHEME_V1, seg.SCHEME_V2):
            assert seg._hash_arrays(raw, scheme) == seg._hash_arrays(fresh, scheme)
            assert seg._hash_gradient(g, scheme) == seg._hash_gradient(fresh_g, scheme)


def test_runner_falls_back_without_into():
    seg = _runner()
    old = OldTrainer()
    ex = seg.ExportBuffers(old)
    assert not ex.reuse
    a, b = ex.state(old), ex.state(old)
    assert a is not b and raw_bytes(a['v']) == raw_bytes(b['v'])
    assert ex.gradients(old) is not ex.gradients(old)
