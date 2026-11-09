# SPDX-License-Identifier: Apache-2.0
"""Program.while_loop as the executor's bounded loop: the carried state is
updated in place, the condition stops the loop on the device, and the host
reads the loop's status every ``check_every`` iterations. The outputs, the
iteration count and the stop flag are the same for every ``check_every``;
the iteration register is readable by the body and the condition; a body
that swaps its state tensors commits both from the same iteration; a body
that resizes its state needs ``check_every=1`` (the default, None, finds
that on the first iteration); refusals hold. Tiny data."""
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


def _words(a):
    return np.ascontiguousarray(a, dtype=np.float32).view(np.uint32)


@pytest.mark.parametrize("every", [None, 1, 2, 3, 8, 64])
def test_check_every_does_not_change_the_result(every):
    from mojolearn import ops
    a = np.array([[0.0]], np.float32)
    with ops.Session() as s:
        x = s.tensor(a)
        condition = s.capture(x, s.less(x, s.full((1, 1), 5)))
        body = s.capture(x, s.add(x, s.full((1, 1), 1)))
        for limit, iterations, stopped in [(10, 5, True), (3, 3, False), (0, 0, False), (5, 5, True)]:
            ref = body.while_loop(a, condition=condition, max_iterations=limit)
            out = body.while_loop(a, condition=condition, max_iterations=limit, check_every=every)
            assert ref["iterations"] == iterations and ref["stopped"] is stopped
            assert out["iterations"] == ref["iterations"] and out["stopped"] is ref["stopped"]
            assert np.array_equal(_words(out["outputs"]), _words(ref["outputs"]))


@pytest.mark.parametrize("every", [1, 4])
def test_swapped_state_commits_from_one_iteration(every):
    from mojolearn import ops
    a = np.array([[1.0, 2.0]], np.float32)
    b = np.array([[10.0, 20.0]], np.float32)
    with ops.Session() as s:
        x = s.tensor(a)
        y = s.tensor(b)
        condition = s.capture([x, y], s.less(s.sum(x), s.full((1, 1), 1.0e9)))
        body = s.capture([x, y], [y, x])
        out = body.while_loop(a, b, condition=condition, max_iterations=3, check_every=every)
        assert out["iterations"] == 3 and out["stopped"] is False
        first, second = out["outputs"]
        assert np.array_equal(_words(first), _words(b))
        assert np.array_equal(_words(second), _words(a))


@pytest.mark.parametrize("every", [1, 5])
def test_iteration_register_is_a_program_value(every):
    from mojolearn import ops
    a = np.array([[0.0]], np.float32)
    with ops.Session() as s:
        x = s.tensor(a)
        i = s.tensor(np.array([[0.0]], np.float32))
        # x accumulates 0 + 1 + 2 + ... over the iterations; stop at i == 4.
        condition = s.capture([x, i], s.less(i, s.full((1, 1), 4)))
        body = s.capture([x, i], s.add(x, i))
        out = body.while_loop(a, condition=condition, max_iterations=100, check_every=every, iteration=True)
        assert out["iterations"] == 4 and out["stopped"] is True
        assert np.array_equal(_words(out["outputs"]), _words(np.array([[6.0]], np.float32)))


def test_resizing_body_needs_check_every_one():
    from mojolearn import ops
    a = np.ones((1, 1), np.float32)
    with ops.Session() as s:
        x = s.tensor(a)
        condition = s.capture(x, s.less(s.sum(x), s.full((1, 1), 4)))
        body = s.capture(x, s.concat([x, x], axis=0))
        out = body.while_loop(a, condition=condition, max_iterations=5)
        assert out["iterations"] == 2 and out["stopped"] is True
        assert np.array_equal(out["outputs"], np.ones((4, 1), np.float32))
        with pytest.raises(ValueError, match="keeps each state tensor's shape"):
            body.while_loop(a, condition=condition, max_iterations=5, check_every=2)


def test_refusals():
    from mojolearn import ops
    a = np.array([[0.0]], np.float32)
    with ops.Session() as s:
        x = s.tensor(a)
        condition = s.capture(x, s.less(x, s.full((1, 1), 3)))
        body = s.capture(x, s.add(x, s.full((1, 1), 1)))
        with pytest.raises(ValueError, match="check_every"):
            body.while_loop(a, condition=condition, max_iterations=3, check_every=0)
        with pytest.raises(ValueError, match="at most 16777216"):
            body.while_loop(a, condition=condition, max_iterations=16777217)
        with pytest.raises(Exception, match="iteration=True|one tensor per input"):
            body.while_loop(a, condition=condition, max_iterations=3, iteration=True)
        nonfinite = s.capture(x, x)
        with pytest.raises(Exception, match="finite"):
            body.while_loop(np.array([[np.nan]], np.float32), condition=nonfinite, max_iterations=3, check_every=4)
