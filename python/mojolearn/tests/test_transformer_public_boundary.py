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


# TOMBSTONE: TRANSFORMER_FORWARD_TAPE_API (DELETED: superseded by the attention step op and the tape rule, lane x6) deleted 2026-11-09 by e49d0ee90d; code recoverable at 4f41e3d67d.
# Restore: git apply experiments/removed/TRANSFORMER_FORWARD_TAPE_API.patch
# (the forward_with_tape / backward_from_tape lifecycle, ticket and hook tests.)


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


def test_decode_session_kept_steps_match_single_forwards():
    # Eight tokens through one kept session (words admitted once, the step
    # reads only cached_tokens) against eight single-token forwards on a
    # fresh state: the same bits and the same cache position after each.
    block = _block()
    rng = np.random.default_rng(31)
    tokens = rng.normal(0, .1, (8, 1, 1, 16)).astype(np.float32)
    reference = block.allocate_state(1, 8)
    expected = [np.asarray(block.forward(t, reference)).copy() for t in tokens]
    state = block.allocate_state(1, 8)
    with block.decode_session(state) as session:
        for i, t in enumerate(tokens):
            actual = session.forward(t)
            assert np.asarray(actual).tobytes() == expected[i].tobytes()
            assert state.cached_tokens == i + 1
    np.testing.assert_array_equal(np.asarray(state.keys()), np.asarray(reference.keys()))


def test_decode_session_sees_a_foreign_cached_tokens_write():
    # A write to the state between steps is seen by the kept (row N177) step
    # path: the session refuses a cached_tokens that differs from the
    # resident cache, and steps on once the caller's word matches again.
    block = _block()
    x = np.full((1, 1, 16), .02, dtype=np.float32)
    state = block.allocate_state(1, 6)
    with block.decode_session(state) as session:
        session.step(x)
        session.step(x)
        assert state.cached_tokens == 2
        state.cached_tokens = 0
        with pytest.raises(ValueError, match="cached_tokens = 0"):
            session.step(x)
        state.cached_tokens = 2
        session.step(x)
        assert state.cached_tokens == 3


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


# TOMBSTONE: TRANSFORMER_FORWARD_TAPE_API (DELETED: superseded by the attention step op and the tape rule, lane x6) deleted 2026-11-09 by e49d0ee90d; code recoverable at 4f41e3d67d.
# Restore: git apply experiments/removed/TRANSFORMER_FORWARD_TAPE_API.patch
# (the budget-aware tape retention and replay test.)
