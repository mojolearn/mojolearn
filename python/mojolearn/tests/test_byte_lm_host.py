# SPDX-License-Identifier: Apache-2.0
"""Host plumbing for CPU inference (DEVIATIONS 2610, 2614, 2615).

A fake binding stands in for `_mojolearn_byte_lm_host`, so these tests prove
wiring, refusals and the CPU-only selector, never arithmetic. The arithmetic
is qualified by tools/byte_lm_host_gate.py against retained GPU captures.
"""
import struct
import os
import sys
import types

import numpy as np
import pytest

from mojolearn import ByteLanguageModelConfig
from mojolearn import _backend, _buffer
from mojolearn import _byte_lm_host as host_mod
from mojolearn.tests.test_byte_lm_surface import buffer


@pytest.fixture
def fake_host(monkeypatch, tmp_path):
    stand_in = tmp_path / '_mojolearn_byte_lm_host.so'
    stand_in.write_bytes(b'')
    monkeypatch.setenv('MOJOLEARN_BYTE_LM_HOST_BINARY', str(stand_in))
    monkeypatch.delenv('MOJOLEARN_BYTE_LM_HOST_ALLOW_SABOTAGE', raising=False)
    shape = ByteLanguageModelConfig()
    m = types.ModuleType(host_mod._MODULE_NAME)
    m.calls = []
    m.byte_lm_host_numeric_mode = lambda: 1
    m.byte_lm_host_vendor = lambda: 'cpu'
    m.byte_lm_host_sabotage = lambda: False
    # bindings/_mojolearn_byte_lm_host.mojo: the IDENTICAL toggle suffix the
    # wrapper appends to the expected profile (empty for a default build).
    m.byte_lm_host_arithmetic_suffix = lambda: ''
    # the real binding answers the profile of whatever shape it is handed
    m.byte_lm_host_profile = lambda native: ByteLanguageModelConfig(*native).profile

    def logits(addresses, dims, native, threaded, threads):
        m.calls.append(('logits', list(dims), threaded, threads))
        batch, length = dims
        out = buffer(addresses[2], batch * length * shape.vocab_size)
        out[:] = 0
        for b in range(batch):
            last = ((b * length) + length - 1) * shape.vocab_size
            out[last + 7] = 2.0
            out[last + 3] = 2.0
        return batch * length * shape.vocab_size

    m.byte_lm_host_logits = logits

    def next_bytes(addresses, dims, native, threads):
        m.calls.append(('next', list(dims), threads))
        batch, _ = dims
        out = buffer(addresses[2], batch, integer=True)
        out[:] = 3
        return batch

    m.byte_lm_host_next = next_bytes
    m.byte_lm_host_loss = (
        lambda addresses, native, threaded, threads: m.calls.append(('loss', threaded, threads)) or 0x3F800000)
    m.all_finite_f32 = lambda addr, n: int(np.isfinite(buffer(addr, n)).all())
    monkeypatch.setitem(sys.modules, host_mod._MODULE_NAME, m)
    # the loader's one-module cache lives in `_backend` (lane/lazy-imports)
    monkeypatch.delitem(_backend._DOOR_HOST_MODULES, host_mod._EXTENSION, raising=False)
    # A PRIVATE helper cache, not an item planted in the process-wide one.
    # `_buffer._native` caches every helper it resolves for the life of the
    # process, so any earlier module that converted an array (a transpose,
    # a float64 cast) leaves its real binding's callable there, and
    # `_native(key)` then answers from the cache without ever reaching the
    # resolver these tests fake. Swapping the dict for the test and letting
    # monkeypatch restore it makes the result independent of test order.
    monkeypatch.setattr(_buffer, '_NATIVE', {'all_finite_f32': m.all_finite_f32})
    yield m
    # The test's own load_door_host() call planted THIS fake under the extension key; delitem above only
    # undoes a key that existed before the test, so without this the fake survived into every later test
    # of the process (test_ragged_lengths saw its 2.0 at vocab index 3; release rehearsal 0.8.39, 2026-10-12).
    _backend._DOOR_HOST_MODULES.pop(host_mod._EXTENSION, None)


# The inference owner (bindings/byte_lm_host_trainer_native.mojo) calls the
# host binding through its compiled descriptor (`byte_lm_host_api_v1`), which
# a Python stand-in cannot provide, and it connects the binding ONCE, when
# `_byte_lm_host` is imported (`_connect_host_trainer` -> `_load()`). So the
# wiring tests below run on the connected compiled host binding (skipped where
# it is not built), and the load-time refusals are tested at `_load()`, which
# the import-time connection runs.
@pytest.fixture
def real_host():
    binding = host_mod._HOST_TRAIN_CONTEXT.get('byte_lm_host_binding')
    if binding is None or isinstance(binding, str):
        pytest.skip(f"compiled host binding unavailable: {binding}")
    return binding


def _zeros():
    return np.zeros(34944, np.float32)


def test_loss_bits_round_trip(real_host):
    model = host_mod.LanguageModelInference(_zeros())
    batch = np.zeros((2, 33), np.int32)
    bits = model.loss_bits(batch)
    assert isinstance(bits, int) and 0 <= bits < (1 << 32)
    assert struct.unpack('<f', struct.pack('<I', bits))[0] == model.loss(batch)


def test_threaded_flag_is_a_bool_and_both_paths_agree(real_host):
    model = host_mod.LanguageModelInference(_zeros(), threaded=True)
    batch = (np.arange(66, dtype=np.int32) % 256).reshape(2, 33)
    assert model.loss_bits(batch) == model.loss_bits(batch, threaded=False)
    ids = batch[:1, :4]
    threaded = np.asarray(model.logits(ids))
    assert threaded.tobytes() == np.asarray(model.logits(ids, threaded=False)).tobytes()
    with pytest.raises(TypeError):
        model.logits(np.zeros((1, 4), np.int32), threaded=1)
    with pytest.raises(TypeError):
        host_mod.LanguageModelInference(_zeros(), threaded='yes')


def test_default_is_the_threaded_path_on_every_core(real_host):
    model = host_mod.LanguageModelInference(_zeros())
    assert vars(model)['_threaded'] is True
    assert vars(model)['_threads'] is None


def test_out_of_range_ids_raise_value_error_before_the_binding(real_host):
    model = host_mod.LanguageModelInference(_zeros())
    for bad in (-1, 256, -100, 1 << 30):
        ids = np.zeros((1, 4), np.int32)
        ids[0, 2] = bad
        with pytest.raises(ValueError, match='byte values'):
            model.logits(ids)
        with pytest.raises(ValueError, match='byte values'):
            model.next_bytes(ids, threaded=False)
        batch = np.zeros((2, 33), np.int32)
        batch[1, 5] = bad
        with pytest.raises(ValueError, match='byte values'):
            model.loss_bits(batch)
    for bad in (-1, 256):
        batch = np.zeros((2, 33), np.int32)
        batch[0, 32] = bad
        with pytest.raises(ValueError, match='byte values'):
            model.loss_bits(batch)
    # -100 in the target-only last column is the loss oracle's ignore_index.
    batch = np.zeros((2, 33), np.int32)
    batch[1, 32] = -100
    bits = model.loss_bits(batch)
    assert isinstance(bits, int)


def test_thread_count_is_kept_and_refused_out_of_range(real_host):
    model = host_mod.LanguageModelInference(_zeros(), threaded=True, threads=3)
    assert vars(model)['_threads'] == 3
    batch = (np.arange(66, dtype=np.int32) % 256).reshape(2, 33)
    assert model.loss_bits(batch) == host_mod.LanguageModelInference(_zeros()).loss_bits(batch)
    assert tuple(model.logits(np.zeros((1, 4), np.int32), threads=2).shape) == (1, 4, 256)
    for bad, error in ((0, ValueError), (1025, ValueError), (True, TypeError), ('3', TypeError)):
        with pytest.raises(error):
            host_mod.LanguageModelInference(_zeros(), threads=bad)
        with pytest.raises(error):
            model.logits(np.zeros((1, 4), np.int32), threads=bad)


def test_logits_shape_and_greedy_ties_go_low(real_host):
    # Zero parameters give equal logits everywhere: every tie goes to id 0.
    model = host_mod.LanguageModelInference(_zeros())
    out = model.logits(np.zeros((2, 5), np.int32))
    assert tuple(out.shape) == (2, 5, 256)
    assert model.next_bytes(np.zeros((2, 5), np.int32)) == [0, 0]


def test_unthreaded_next_bytes_uses_full_logits_fallback(real_host):
    model = host_mod.LanguageModelInference(_zeros())
    ids = (np.arange(10, dtype=np.int32) * 7 % 256).reshape(2, 5)
    assert model.next_bytes(ids, threaded=False) == model.next_bytes(ids)


def test_refusals(real_host):
    model = host_mod.LanguageModelInference(_zeros())
    with pytest.raises(ValueError):
        model.loss_bits(np.zeros((2, 32), np.int32))
    with pytest.raises(ValueError):
        model.logits(np.zeros((1, 33), np.int32))
    bad = _zeros()
    bad[5] = np.nan
    with pytest.raises(ValueError):
        host_mod.LanguageModelInference(bad)
    with pytest.raises(ValueError):
        host_mod.LanguageModelInference(np.zeros(10, np.float32))


def test_sabotage_build_refused_outside_the_gate(fake_host, monkeypatch):
    fake_host.byte_lm_host_sabotage = lambda: True
    with pytest.raises(RuntimeError, match='SABOTAGE'):
        host_mod._load()
    monkeypatch.setenv('MOJOLEARN_BYTE_LM_HOST_ALLOW_SABOTAGE', '1')
    monkeypatch.delitem(_backend._DOOR_HOST_MODULES, host_mod._EXTENSION, raising=False)
    assert host_mod._load() is fake_host


def test_wrong_mode_or_vendor_refused(fake_host, monkeypatch):
    fake_host.byte_lm_host_vendor = lambda: 'metal'
    with pytest.raises(RuntimeError):
        host_mod._load()
    fake_host.byte_lm_host_vendor = lambda: 'cpu'
    fake_host.byte_lm_host_numeric_mode = lambda: 0
    monkeypatch.delitem(_backend._DOOR_HOST_MODULES, host_mod._EXTENSION, raising=False)
    with pytest.raises(RuntimeError):
        host_mod._load()


def test_connected_load_refusal_is_raised_by_construction(monkeypatch):
    # A binding refused at import-time connection is kept as its refusal's
    # text, and construction raises it with its class.
    monkeypatch.setitem(host_mod._HOST_TRAIN_CONTEXT, 'byte_lm_host_binding',
                        'RuntimeError: byte LM host binding is a SABOTAGE build')
    with pytest.raises(RuntimeError, match='SABOTAGE'):
        host_mod.LanguageModelInference(_zeros())


def test_native_helpers_fall_back_to_the_cpu_binding_only_for_the_three(fake_host, monkeypatch):
    def no_gpu_set(*args, **kwargs):
        raise ImportError('no identical set')
    monkeypatch.setattr(_backend, 'binding', no_gpu_set)
    monkeypatch.delitem(_buffer._NATIVE, 'all_finite_f32')
    assert _buffer._native('all_finite_f32') is fake_host.all_finite_f32
    with pytest.raises(ImportError):
        _buffer._native('transpose_f32')


def test_cpu_only_selector_stubs_raise_by_name(monkeypatch):
    pkg = types.ModuleType('fakepkg')
    monkeypatch.setattr(_backend, '_CPU_ONLY', None)
    monkeypatch.setattr(_backend, '_SELECTED', None)
    monkeypatch.setattr(_backend, '_MISSING', [])
    try:
        assert _backend._select_cpu_only(pkg, 'identical', 'NO SUPPORTED GPU FOUND (test)') == 'identical'
        assert _backend.vendor() == 'cpu'
        assert _backend.gpu_arch() is None
        # Byte LM's host trainer has its own loader; the GPU session binding
        # remains a named refusal. Mamba now has a routed CPU binding.
        stub = sys.modules['fakepkg._mojolearn_byte_lm']
        with pytest.raises(ImportError, match='NO GPU binary set') as info:
            stub.byte_lm_session_open
        assert 'NO SUPPORTED GPU FOUND (test)' in str(info.value)
        # A family whose host binding is BUILT on this box is routed to it, not
        # stubbed (phase 1 routing); the classical host lane's estimators
        # binding was the first to sit beside this test on a developer Mac.
        routed = {name for name, basename in _backend._HOST_MODULES.items()
                  if os.path.exists(_backend.host_module_path(basename))}
        assert set(_backend._MISSING) == set(_backend._MODULES) - routed
        assert '_mojolearn_byte_lm' in _backend._MISSING
    finally:
        for key in [k for k in sys.modules if k.startswith('fakepkg.')]:
            del sys.modules[key]
