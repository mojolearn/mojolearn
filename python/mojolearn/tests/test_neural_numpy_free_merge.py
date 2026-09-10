"""Host packing/state checks; GPU arithmetic is mocked, never reimplemented."""
import array
import ctypes
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

from mojolearn import _buffer as buffers, _bufcheck as checks
from mojolearn import _training_impl as training
from mojolearn._transformer_impl import TransformerState
from mojolearn._samba_impl import SambaConfig, SambaStack


def test_ring_order_and_linear_alias():
    raw = buffers.empty(24, '<f4')
    view = checks.flat_view(raw, 'f')
    for i in range(24):
        view[i] = i
    state = TransformerState(2, 2, 2, 3, raw, raw, 5, window=3)
    expected = np.arange(24, dtype=np.float32).reshape(2, 2, 3, 2)[:, :, [2, 0, 1], :]
    result = state.keys()
    np.testing.assert_array_equal(result, expected)
    checks.flat_view(result, 'f')[0] = -99
    assert view[4] == 4  # Ring access is a copy.
    linear = TransformerState(2, 2, 2, 3, raw, raw, 2)
    keys = linear.keys()
    checks.flat_view(keys, 'f')[0] = 42
    assert view[0] == 42  # Linear access is a borrowed view.


def test_accumulation_packs_strides_and_plain_buffers(monkeypatch):
    captured = []
    class Binding:
        def accumulate(self, addresses, params):
            n, count, tokens = params
            captured.append(list((ctypes.c_float * (n * count)).from_address(addresses[1])))
            output = (ctypes.c_float * n).from_address(addresses[0])
            for i in range(n):
                output[i] = i + 20  # Sentinel native result, no CPU gradient math.
    monkeypatch.setattr(training, '_load', lambda mode: Binding())
    a = np.arange(12, dtype=np.float32).reshape(3, 4)[:, ::2]
    result = training.accumulate_grads([a, a + 2], tokens=None)
    assert captured[-1] == list(a.ravel()) + list((a + 2).ravel())
    np.testing.assert_array_equal(result, np.arange(20, 26).reshape(3, 2))
    result = training.accumulate_grads([array.array('f', [1, 2]), array.array('f', [3, 4])], None)
    assert captured[-1] == [1, 2, 3, 4]
    assert result.shape == (2,)


def test_samba_parameter_alias_checkpoint(tmp_path):
    config = SambaConfig(11, 32, ['attention'], n_heads=2, n_kv_heads=1,
                         head_dim=16, intermediate=48)
    model = SambaStack(config, weights={n: buffers.full(s, .25, '<f4') for n, s in config.registry()})
    first = model.names[0]
    checks.flat_view(model.parameters()[first], 'f')[0] = 7
    assert checks.flat_view(model.flat, 'f')[0] == 7
    path = tmp_path / 'state.json'
    model.save_checkpoint(path)
    restored = SambaStack.from_checkpoint(path)
    assert checks.le_bytes(restored.flat, 'f') == checks.le_bytes(model.flat, 'f')
    checks.flat_view(restored.parameters()[first], 'f')[0] = 9
    assert checks.flat_view(restored.flat, 'f')[0] == 9


def test_runtime_imports_and_fixture_without_numpy():
    code = '''
import sys
class BlockNumpy:
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'numpy' or fullname.startswith('numpy.'):
            raise AssertionError('runtime NumPy import: ' + fullname)
sys.meta_path.insert(0, BlockNumpy())
from mojolearn import _byte_lm_impl, _mamba_impl, _samba_impl, _training_impl, _transformer_impl, _verify
from mojolearn import _buffer, _bufcheck
from mojolearn._samba_impl import SambaConfig, SambaStack
config = SambaConfig(11, 32, ['attention'], n_heads=2, n_kv_heads=1, head_dim=16, intermediate=48)
model = SambaStack(config, weights={n: _buffer.zeros(s, '<f4') for n, s in config.registry()})
assert model.n_total > 0
x, c = _verify.build_fixture()
def fnv(data):
    h = 14695981039346656037
    for b in data:
        h = ((h ^ b) * 1099511628211) & ((1 << 64) - 1)
    return h
assert fnv(_verify._le_bytes(x)) == _verify.FIXTURE_X_FNV1A64
assert fnv(_verify._le_bytes(c)) == _verify.FIXTURE_C_FNV1A64
assert 'numpy' not in sys.modules
'''
    env = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[2]))
    subprocess.run([sys.executable, '-c', code], env=env, check=True, capture_output=True, text=True)
