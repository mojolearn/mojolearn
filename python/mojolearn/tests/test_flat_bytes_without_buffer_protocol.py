"""On Python 3.10 and 3.11 the package's `Array` cannot export the buffer
protocol (DEVIATION 2305), so `memoryview(array)` raises TypeError there.
The live worker's fold export and the cross-vendor state hash read arrays
through `_buffer.flat_bytes`, which works on every Python. These tests take
`__buffer__` away from `Array` to stand in for those Pythons."""
import hashlib
import struct

import pytest

from mojolearn import _buffer, cross_vendor
from mojolearn._array import Array


@pytest.fixture
def no_buffer_protocol(monkeypatch):
    if hasattr(Array, "__buffer__"):
        monkeypatch.delattr(Array, "__buffer__")
    a = Array.from_list([1.5, -2.0, 3.25], dtype="<f4")
    with pytest.raises(TypeError):
        memoryview(a)
    return a


def test_flat_bytes_reads_an_array_without_the_buffer_protocol(no_buffer_protocol):
    a = no_buffer_protocol
    mv = _buffer.flat_bytes(a)
    assert mv.format == "B" and mv.nbytes == 12 and mv.readonly
    assert mv.tobytes() == struct.pack("<3f", 1.5, -2.0, 3.25) == a.tobytes()


def test_flat_bytes_reads_bytes_and_numpy_too():
    import numpy as np
    raw = struct.pack("<3f", 1.5, -2.0, 3.25)
    assert _buffer.flat_bytes(raw).tobytes() == raw
    assert _buffer.flat_bytes(np.frombuffer(raw, dtype="<f4")).tobytes() == raw
    with pytest.raises(TypeError):
        _buffer.flat_bytes(object(), name="thing")


def test_state_hash_and_f32_on_arrays_without_the_buffer_protocol(no_buffer_protocol):
    a = no_buffer_protocol
    raw = a.tobytes()
    state = dict(parameters=a, m=a, v=a, flags=a)
    assert cross_vendor.state_hash(state) == hashlib.sha256(raw * 4).hexdigest()
    assert cross_vendor.state_hash(dict(parameters=raw, m=raw, v=raw, flags=raw)) == cross_vendor.state_hash(state)
    assert list(cross_vendor._f32(a)) == [1.5, -2.0, 3.25]
    assert cross_vendor.ordered_fold([a, a]) == cross_vendor.ordered_fold([raw, raw])


def test_a_perturbed_array_changes_the_hash(no_buffer_protocol):
    a = no_buffer_protocol
    b = Array.from_list([1.5, -2.0, 3.0], dtype="<f4")
    assert cross_vendor.state_hash(dict(parameters=a, m=a, v=a, flags=a)) != cross_vendor.state_hash(dict(parameters=b, m=a, v=a, flags=a))
