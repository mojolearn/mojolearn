# SPDX-License-Identifier: Apache-2.0
"""Host plumbing for CPU inference (DEVIATIONS 2610, 2614, 2615).

A fake binding stands in for `_mojolearn_byte_lm_host`, so these tests prove
wiring, refusals and the CPU-only selector, never arithmetic. The arithmetic
is qualified by tools/byte_lm_host_gate.py against retained GPU captures.
"""
import struct
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
    m.byte_lm_host_profile = (
        lambda native: shape.profile if list(native) == host_mod._native_shape(shape) else 'other')

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
    m.byte_lm_host_loss = (
        lambda addresses, native, threaded, threads: m.calls.append(('loss', threaded, threads)) or 0x3F800000)
    m.all_finite_f32 = lambda addr, n: int(np.isfinite(buffer(addr, n)).all())
    monkeypatch.setitem(sys.modules, host_mod._MODULE_NAME, m)
    monkeypatch.setattr(host_mod, '_MODULE', None)
    monkeypatch.setitem(_buffer._NATIVE, 'all_finite_f32', m.all_finite_f32)
    return m


def test_loss_bits_round_trip(fake_host):
    model = host_mod.LanguageModelInference(np.zeros(34944, np.float32))
    assert model.loss_bits(np.zeros((2, 33), np.int32)) == 0x3F800000
    assert model.loss(np.zeros((2, 33), np.int32)) == 1.0


def test_threaded_flag_reaches_the_binding_as_an_int(fake_host):
    model = host_mod.LanguageModelInference(np.zeros(34944, np.float32), threaded=True)
    model.loss_bits(np.zeros((2, 33), np.int32))
    model.loss_bits(np.zeros((2, 33), np.int32), threaded=False)
    model.logits(np.zeros((1, 4), np.int32))
    assert fake_host.calls == [('loss', 1, 0), ('loss', 0, 0), ('logits', [1, 4], 1, 0)]
    with pytest.raises(TypeError):
        model.logits(np.zeros((1, 4), np.int32), threaded=1)
    with pytest.raises(TypeError):
        host_mod.LanguageModelInference(np.zeros(34944, np.float32), threaded='yes')


def test_thread_count_reaches_the_binding_and_is_refused_out_of_range(fake_host):
    model = host_mod.LanguageModelInference(np.zeros(34944, np.float32), threaded=True, threads=3)
    model.loss_bits(np.zeros((2, 33), np.int32))
    model.logits(np.zeros((1, 4), np.int32), threads=2)
    assert fake_host.calls == [('loss', 1, 3), ('logits', [1, 4], 1, 2)]
    for bad, error in ((0, ValueError), (1025, ValueError), (True, TypeError), ('3', TypeError)):
        with pytest.raises(error):
            host_mod.LanguageModelInference(np.zeros(34944, np.float32), threads=bad)
        with pytest.raises(error):
            model.logits(np.zeros((1, 4), np.int32), threads=bad)


def test_logits_shape_and_greedy_ties_go_low(fake_host):
    model = host_mod.LanguageModelInference(np.zeros(34944, np.float32))
    out = model.logits(np.zeros((2, 5), np.int32))
    assert tuple(out.shape) == (2, 5, 256)
    assert model.next_bytes(np.zeros((2, 5), np.int32)) == [3, 3]


def test_refusals(fake_host):
    model = host_mod.LanguageModelInference(np.zeros(34944, np.float32))
    with pytest.raises(ValueError):
        model.loss_bits(np.zeros((2, 32), np.int32))
    with pytest.raises(ValueError):
        model.logits(np.zeros((1, 33), np.int32))
    bad = np.zeros(34944, np.float32)
    bad[5] = np.nan
    with pytest.raises(ValueError):
        host_mod.LanguageModelInference(bad)
    with pytest.raises(ValueError):
        host_mod.LanguageModelInference(np.zeros(10, np.float32))


def test_sabotage_build_refused_outside_the_gate(fake_host, monkeypatch):
    fake_host.byte_lm_host_sabotage = lambda: True
    with pytest.raises(RuntimeError, match='SABOTAGE'):
        host_mod.LanguageModelInference(np.zeros(34944, np.float32))
    monkeypatch.setenv('MOJOLEARN_BYTE_LM_HOST_ALLOW_SABOTAGE', '1')
    monkeypatch.setattr(host_mod, '_MODULE', None)
    host_mod.LanguageModelInference(np.zeros(34944, np.float32))


def test_wrong_mode_or_vendor_refused(fake_host, monkeypatch):
    fake_host.byte_lm_host_vendor = lambda: 'metal'
    with pytest.raises(RuntimeError):
        host_mod.LanguageModelInference(np.zeros(34944, np.float32))
    fake_host.byte_lm_host_vendor = lambda: 'cpu'
    fake_host.byte_lm_host_numeric_mode = lambda: 0
    monkeypatch.setattr(host_mod, '_MODULE', None)
    with pytest.raises(RuntimeError):
        host_mod.LanguageModelInference(np.zeros(34944, np.float32))


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
        stub = sys.modules['fakepkg._mojolearn_gbdt']
        with pytest.raises(ImportError, match='NO GPU binary set') as info:
            stub.gbdt_fit
        assert 'NO SUPPORTED GPU FOUND (test)' in str(info.value)
        assert set(_backend._MISSING) == set(_backend._MODULES)
    finally:
        for key in [k for k in sys.modules if k.startswith('fakepkg.')]:
            del sys.modules[key]
