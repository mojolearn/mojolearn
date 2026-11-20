# SPDX-License-Identifier: Apache-2.0
"""Program.decode and carried state (lane x1-executor-state-and-step-ops):
a captured next-token graph runs n times as a device loop, the tokens of
every step are recorded on the device and returned together, and the result
is the same for every ``unroll``; integer state is carried word for word by
while_loop; an ids iteration input is an int32 position; Session.state makes
a state tensor that decode can reset. Tiny data. Written unrun (the lane
does not run tests); the orchestrator runs it once after the compile.
`Session.embedding` returns one row per id (flattened) and `Session.ids`
keeps a (B, 1) array two-dimensional, so the next-token argmax keeps its
axis (keepdims=True) for the state to keep its rank."""
import numpy as np
import pytest

from mojolearn import _backend


def _has_entries():
    try:
        return hasattr(_backend.binding("_mojolearn_x_sequence"), "ops_session_call")
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _has_entries(),
                                reason="the sequence binding has no ops_session_call entry (rebuild it)")


def _next_token_table(vocab):
    """A (vocab, vocab) table whose row t has its maximum at (t + 1) % vocab:
    embedding + argmax walks the tokens 0, 1, 2, ... deterministically."""
    table = np.zeros((vocab, vocab), np.float32)
    for t in range(vocab):
        table[t, (t + 1) % vocab] = 1.0
    return table


def _reference(first, n, vocab):
    out, cur = [], np.asarray(first, np.int64).reshape(-1)
    for _ in range(n):
        cur = (cur + 1) % vocab
        out.append(cur.copy())
    return np.stack(out).astype(np.int32)


@pytest.mark.parametrize("unroll", [1, 2, 3, 8])
def test_decode_records_every_step(unroll):
    from mojolearn import ops
    vocab = 7
    first = np.array([[2], [5]])
    with ops.Session() as s:
        table = s.tensor(_next_token_table(vocab))
        ids = s.ids(first)
        logits = s.embedding(table, ids)                      # (2, vocab): one row per id
        nxt = s.argmax(logits, axis=-1, keepdims=True, int32=True)
        step = s.capture([ids], [nxt])
        out = step.decode(first, n=10, unroll=unroll)
        assert out["iterations"] == 10
        assert out["tokens"].dtype == np.int32 and out["tokens"].shape == (10, 2)
        assert np.array_equal(out["tokens"], _reference(first, 10, vocab))
        assert out["outputs"][0].dtype == np.int32


def test_decode_zero_steps():
    from mojolearn import ops
    with ops.Session() as s:
        table = s.tensor(_next_token_table(4))
        ids = s.ids(np.array([[1]]))
        step = s.capture([ids], [s.argmax(s.embedding(table, ids), axis=-1, keepdims=True, int32=True)])
        out = step.decode(np.array([[1]]), n=0)
        assert out["iterations"] == 0 and out["tokens"].shape == (0, 1)


@pytest.mark.parametrize("every", [1, 4])
def test_while_loop_carries_integer_state(every):
    from mojolearn import ops
    vocab = 5
    with ops.Session() as s:
        table = s.tensor(_next_token_table(vocab))
        ids = s.ids(np.array([[0]]))
        nxt = s.argmax(s.embedding(table, ids), axis=-1, keepdims=True, int32=True)
        body = s.capture([ids], [nxt])
        cond = s.capture([ids], [s.full((1, 1), 1)])
        out = body.while_loop(np.array([[0]]), condition=cond, max_iterations=3, check_every=every)
        assert out["iterations"] == 3
        assert out["outputs"][0].dtype == np.int32 and int(out["outputs"][0].reshape(-1)[0]) == 3


def test_state_tensor_is_reset_by_decode():
    from mojolearn import ops
    with ops.Session() as s:
        pos = s.state(np.array([[4]]), ids=True)
        assert s.numpy(pos).view(np.int32).reshape(-1)[0] == 4
        table = s.tensor(_next_token_table(3))
        ids = s.ids(np.array([[0]]))
        step = s.capture([ids], [s.argmax(s.embedding(table, ids), axis=-1, keepdims=True, int32=True)])
        step.decode(np.array([[0]]), n=1, state={pos: np.array([[0]])})
        assert s.numpy(pos).view(np.int32).reshape(-1)[0] == 0


def test_decode_refuses_a_non_state_handle():
    from mojolearn import ops
    with ops.Session() as s:
        w = s.tensor(np.zeros((1, 1), np.float32))
        table = s.tensor(_next_token_table(3))
        ids = s.ids(np.array([[0]]))
        step = s.capture([ids], [s.argmax(s.embedding(table, ids), axis=-1, keepdims=True, int32=True)])
        with pytest.raises(ValueError):
            step.decode(np.array([[0]]), n=1, state={w: np.zeros((1, 1), np.float32)})
