# SPDX-License-Identifier: Apache-2.0
"""mojolearn.ops.gradcheck (lane gradcheck): the MyTanh rule passes; a rule that
drops one path of an input, or that assumes an upstream gradient of 1, fails
and names that input (and only it); a straight-through surrogate fails the
finite difference but passes against its explicit expected rule."""
import numpy as np
import pytest

from mojolearn import _backend


def _has_entries():
    try:
        b = _backend.binding("_mojolearn_x_sequence")
        return hasattr(b, "ops_session_call")
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _has_entries(),
                                reason="the sequence binding has no ops_session_call entry (rebuild it)")


def _x(shape, seed):
    return np.random.default_rng(seed).standard_normal(shape).astype(np.float32)


def _my_tanh(cap, shape):
    zeros = np.zeros(shape, np.float32)
    x = cap.tensor(zeros)
    forward = cap.capture(inputs=[x], outputs=[cap.tanh(x)])
    y, g = cap.tensor(zeros), cap.tensor(zeros)
    backward = cap.capture(inputs=[y, g], outputs=[cap.mul(g, cap.sub(cap.full((1, 1), 1.0), cap.mul(y, y)))])
    return forward, backward


def _two_inputs(cap, shape, rule):
    """forward(a, b) = tanh(a) + a * b, saving tanh(a), a and b; `rule` picks the backward:
    "right": da = g (1 - t^2) + g b, db = g a
    "drop_path": da = g (1 - t^2) (the a * b path of a is dropped), db = g a
    "upstream_one": da right, db = a (the incoming gradient is taken as 1)"""
    zeros = np.zeros(shape, np.float32)
    a, b = cap.tensor(zeros), cap.tensor(zeros)
    t = cap.tanh(a)
    forward = cap.capture(inputs=[a, b], outputs=[cap.add(t, cap.mul(a, b)), t, a, b])
    st, sa, sb, g = (cap.tensor(zeros) for _ in range(4))
    tanh_path = cap.mul(g, cap.sub(cap.full((1, 1), 1.0), cap.mul(st, st)))
    # the dropped path is kept at weight 0 so every backward input is read
    da = cap.add(tanh_path, cap.scale(cap.mul(g, sb), 0.0 if rule == "drop_path" else 1.0))
    db = cap.mul(cap.full((1, 1), 1.0), sa) if rule == "upstream_one" else cap.mul(g, sa)
    backward = cap.capture(inputs=[st, sa, sb, g], outputs=[da, db])
    return forward, backward


def test_my_tanh_passes():
    from mojolearn import ops
    with ops.Session() as cap:
        forward, backward = _my_tanh(cap, (6, 4))
        assert ops.gradcheck(forward, backward, [_x((6, 4), 0)], saved=[0]) is True


def test_right_two_input_rule_passes():
    from mojolearn import ops
    with ops.Session() as cap:
        forward, backward = _two_inputs(cap, (5, 4), "right")
        assert ops.gradcheck(forward, backward, [_x((5, 4), 1), _x((5, 4), 2)],
                             saved=[1, 2, 3], outputs=1) is True


@pytest.mark.parametrize("rule, bad", [("drop_path", 0), ("upstream_one", 1)])
def test_wrong_rule_fails_and_names_the_input(rule, bad):
    from mojolearn import ops
    inputs = [_x((5, 4), 1), _x((5, 4), 2)]
    with ops.Session() as cap:
        forward, backward = _two_inputs(cap, (5, 4), rule)
        with pytest.raises(ops.GradcheckError, match=f"for input {bad}\\n") as caught:
            ops.gradcheck(forward, backward, inputs, saved=[1, 2, 3], outputs=1)
        assert [m["input"] for m in caught.value.mismatches] == [bad]
        m = caught.value.mismatches[0]
        assert m["difference"] > m["tolerance"]
        assert ops.gradcheck(forward, backward, inputs, saved=[1, 2, 3], outputs=1,
                             raise_exception=False) is False


def _straight_through(cap, shape, factor):
    """forward round(x); backward factor * g (straight-through when factor is 1);
    the expected rule (x, g) -> g."""
    zeros = np.zeros(shape, np.float32)
    x = cap.tensor(zeros)
    forward = cap.capture(inputs=[x], outputs=[cap.round(x)])
    g = cap.tensor(zeros)
    backward = cap.capture(inputs=[g], outputs=[cap.scale(g, factor)])
    ex, eg = cap.tensor(zeros), cap.tensor(zeros)
    expected = cap.capture(inputs=[ex, eg], outputs=[cap.add(eg, cap.scale(ex, 0.0))])
    return forward, backward, expected


def test_surrogate_rule_checks_against_its_expected_rule():
    from mojolearn import ops
    shape = (4, 3)
    # k + 0.2: a step of eps never crosses a rounding boundary, so the finite difference is 0
    x0 = (np.random.default_rng(3).integers(-3, 4, shape) + 0.2).astype(np.float32)
    with ops.Session() as cap:
        forward, backward, expected = _straight_through(cap, shape, 1.0)
        with pytest.raises(ops.GradcheckError, match="finite difference"):
            ops.gradcheck(forward, backward, [x0])
        assert ops.gradcheck(forward, backward, [x0], expected=expected) is True
        forward, backward, expected = _straight_through(cap, shape, 2.0)
        with pytest.raises(ops.GradcheckError, match="expected rule for input 0") as caught:
            ops.gradcheck(forward, backward, [x0], expected=expected)
        assert [m["input"] for m in caught.value.mismatches] == [0]


def test_input_count_is_refused():
    from mojolearn import ops
    with ops.Session() as cap:
        forward, backward = _my_tanh(cap, (4, 3))
        with pytest.raises(ValueError, match="takes 1 inputs, got 2"):
            ops.gradcheck(forward, backward, [_x((4, 3), 0), _x((4, 3), 1)], saved=[0])
