"""Small public Transformer boundary controls. Run after the affected native compile.

These are not cross-vendor identity or performance evidence. Test data and
independent expected cache views are prepared in Python outside the operation.
"""
import array
import pickle

import numpy as np
import pytest

from mojolearn import TransformerBlock, TransformerState


def _cache_state(*, window=0, cached=0, batch=2, kv=2, hd=2, capacity=8):
    slots = window or capacity
    keys = array.array("f", range(batch * kv * slots * hd))
    values = array.array("f", (100 + i for i in range(len(keys))))
    return TransformerState(batch, kv, hd, capacity, keys, values, cached, window)


def _block():
    d, it = 16, 32
    rng = np.random.default_rng(27)
    names = TransformerBlock._W_NAMES
    shapes = ((d,), (d,), (d, d), (d, d), (d, d), (d, d), (it, d), (it, d), (d, it))
    weights = {name: rng.normal(0, .01, shape).astype(np.float32) for name, shape in zip(names, shapes)}
    weights[names[0]].fill(1)
    weights[names[1]].fill(1)
    return TransformerBlock(weights, n_heads=1, numeric_mode="identical")


def test_state_unwindowed_views_retain_owner_and_alias_cache():
    state = _cache_state(cached=3)
    keys = state.keys()
    values = state.values()
    assert keys.shape == values.shape == (2, 2, 3, 2)
    state.k_cache[1] = 71
    assert np.asarray(keys).ravel()[1] == 71
    original = np.asarray(values).copy()
    del state
    np.testing.assert_array_equal(np.asarray(values), original)


@pytest.mark.parametrize("cached", [0, 2, 3, 5, 8])
def test_state_ring_is_chronological_copy(cached):
    state = _cache_state(window=3, cached=cached)
    held = min(cached, 3)
    source = np.frombuffer(state.k_cache, dtype=np.float32).reshape(2, 2, 3, 2)
    order = [(cached - held + i) % 3 for i in range(held)]
    expected = source[:, :, order, :].copy()
    actual = state.keys()
    np.testing.assert_array_equal(np.asarray(actual), expected)
    state.k_cache[0] = -99
    np.testing.assert_array_equal(np.asarray(actual), expected)
    assert state.capacity == 3


def test_state_checkpoint_roundtrip_keeps_cache_layout():
    state = _cache_state(window=3, cached=5)
    restored = pickle.loads(pickle.dumps(state))
    np.testing.assert_array_equal(np.asarray(restored.keys()), np.asarray(state.keys()))
    assert restored.cached_tokens == 5 and restored.capacity == 3


def test_state_rejects_scalar_hooks_before_conversion():
    calls = []
    class Hook:
        def __int__(self):
            calls.append("int")
            return 1
        def __index__(self):
            calls.append("index")
            return 1
    with pytest.raises(TypeError):
        TransformerState(Hook(), 1, 1, 1, array.array("f", [0]), array.array("f", [0]))
    assert calls == []
    state = _cache_state()
    state.cached_tokens = Hook()
    with pytest.raises(TypeError):
        state.keys()
    assert calls == []


def test_state_dictionary_property_is_not_invoked():
    calls = []
    class HookState(TransformerState):
        @property
        def __dict__(self):
            calls.append("dict")
            return {}
    state = object.__new__(HookState)
    with pytest.raises(TypeError):
        state.keys()
    assert calls == []


def test_state_buffer_protocol_is_not_invoked():
    calls = []
    class BufferHook:
        def __buffer__(self, flags):
            calls.append("buffer")
            return memoryview(array.array("f", [0]))
    with pytest.raises(TypeError):
        TransformerState(1, 1, 1, 1, BufferHook(), array.array("f", [0]))
    assert calls == []


def test_tape_lifecycle_and_gradient_shapes():
    block = _block()
    x = np.full((1, 2, 16), .01, dtype=np.float32)
    y, tape = block.forward_with_tape(x, activation_budget_bytes=0)
    assert y.shape == x.shape and tape.shape == x.shape
    with pytest.raises(TypeError):
        pickle.dumps(tape)
    gradients = block.backward_from_tape(tape, np.ones_like(x))
    assert gradients["x"].shape == x.shape
    assert set(gradients) == {"x", *block._W_NAMES}
    with pytest.raises(ValueError):
        block.backward_from_tape(tape, np.ones_like(x))
    tape.close()


def test_tape_close_does_not_discard_newer_ticket():
    block = _block()
    x = np.full((1, 2, 16), .01, dtype=np.float32)
    _, old = block.forward_with_tape(x)
    _, current = block.forward_with_tape(x)
    old.close()
    assert block.backward_from_tape(current, np.ones_like(x))["x"].shape == x.shape


def test_tape_rejects_input_and_mutable_model_hooks():
    block = _block()
    calls = []
    class ArrayHook(np.ndarray):
        @property
        def shape(self):
            calls.append("shape")
            return (1, 2, 16)
    with pytest.raises(TypeError):
        block.forward_with_tape(np.ones((1, 2, 16), dtype=np.float32).view(ArrayHook))
    assert calls == []
    class ScalarHook:
        def __int__(self):
            calls.append("int")
            return 16
    block.d_model = ScalarHook()
    with pytest.raises(TypeError):
        block.forward_with_tape(np.ones((1, 2, 16), dtype=np.float32))
    assert calls == []


def test_decode_session_state_roundtrip_and_original_route_bits():
    block = _block()
    source = np.full((1, 2, 16), .01, dtype=np.float32)
    reference = block.allocate_state(1, 6)
    expected = np.asarray(block.forward(source, reference)).copy()
    state = block.allocate_state(1, 6)
    with block.decode_session(state) as session:
        actual = session.forward(source)
        assert np.asarray(actual).tobytes() == expected.tobytes()
        session.sync_state()
        snapshot = np.asarray(state.keys()).copy()
        session.load_state()
        np.testing.assert_array_equal(np.asarray(state.keys()), snapshot)
    assert not session.is_open
    assert state.cached_tokens == 2
    np.testing.assert_array_equal(np.asarray(state.keys()), np.asarray(reference.keys()))


def test_decode_rejects_mutable_state_hooks_without_invocation():
    block = _block()
    state = block.allocate_state(1, 6)
    calls = []
    class Hook:
        def __int__(self):
            calls.append("int")
            return 0
    with block.decode_session(state) as session:
        state.cached_tokens = Hook()
        with pytest.raises(TypeError):
            session.step(np.zeros((1, 1, 16), dtype=np.float32))
        assert calls == []
        state.cached_tokens = 0


def test_decode_rejects_replaced_session_owners():
    block = _block()
    state = block.allocate_state(1, 6)
    session = block.decode_session(state)
    original = session._state
    try:
        session._state = object()
        with pytest.raises(TypeError):
            session.step(np.zeros((1, 1, 16), dtype=np.float32))
    finally:
        session._state = original
        session.discard()
