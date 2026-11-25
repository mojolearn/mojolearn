# SPDX-License-Identifier: Apache-2.0
"""`mojolearn.ops.gradcheck`: check a user-defined backward rule.

A developer testing tool, like `torch.autograd.gradcheck` (AGENTS.md "The verify
command is protected": testing tooling is not product execution). It is not a
verify lane and records no reference. Every computation on tensor data is a
native session operation (custom forward and backward, add, scale, mul, sum,
abs, max); the Python here draws the seeded random directions, reads back a few
scalars per input and compares them.
"""
import math

__all__ = ["gradcheck", "GradcheckError"]

# float32 unit roundoff (2 ** -24): the rounding of one float32 operation.
_FLOAT32_ROUNDOFF = 2.0 ** -24


class GradcheckError(RuntimeError):
    """Raised by :func:`gradcheck` when a backward rule disagrees.

    Attributes
    ----------
    mismatches : list of dict
        One entry per input that disagrees: ``input`` (its position among the
        forward program's inputs), ``shape``, ``analytical``, ``numerical`` (or
        ``expected``), ``difference`` and ``tolerance``.
    """

    def __init__(self, message, mismatches):
        super().__init__(message)
        self.mismatches = mismatches


def gradcheck(forward, backward, inputs, *, saved=(), outputs=None, expected=None, eps=5e-3,
              atol=1e-5, rtol=1e-2, seed=0, raise_exception=True, numeric_mode=None, device=None):
    """Check a backward rule for :meth:`Session.custom_op` against its forward.

    Like ``torch.autograd.gradcheck(..., fast_mode=True)``: a random upstream
    gradient ``v`` (standard normal, so neither 1 nor constant) is applied to
    the operation's outputs, the rule's vector-Jacobian product ``J^T v`` is
    computed with :meth:`Session.backward` exactly as in training, and for each
    input ``x_i`` it is compared, along a random direction ``d_i`` (uniform in
    ``[-1, 1]``), with the central difference of the forward::

        numerical_i  = (L(x_i + eps * d_i) - L(x_i - eps * d_i)) / (2 * eps)
        analytical_i = sum((J_i^T v) * d_i),   where L(x) = sum(v * forward(x))

    the other inputs held fixed. Input ``i`` agrees when
    ``|analytical_i - numerical_i| <= atol + rtol * max(|analytical_i|, |numerical_i|) + noise``,
    where ``noise = (2 + log2(n)) * 2**-24 * sum(|v * y|) / eps`` bounds the
    float32 rounding of the two forward evaluations (``n`` output elements).
    Every input is checked, and an error names each one that disagrees and by
    how much.

    For a deliberate surrogate gradient (a straight-through estimator,
    gradient reversal), whose rule is not the derivative of the forward, pass
    the rule you intend as ``expected``: the rule's gradients are then compared
    elementwise with it, ``|rule - expected| <= atol + rtol * |expected|``,
    instead of with the finite difference.

    Parameters
    ----------
    forward, backward : Program
        The two programs, as passed to :meth:`Session.custom_op`.
    inputs : sequence of ndarray
        One array per input of ``forward``, of its captured shape (an integer
        array for an input made with :meth:`Session.ids`, which takes no
        gradient and is not checked). Choose values where the forward is
        smooth: a finite difference across a kink or a jump disagrees with
        any rule.
    saved, outputs
        As in :meth:`Session.custom_op`.
    expected : Program or None, default None
        ``(x_0, ..., x_{n-1}, grad_y_0, ..., grad_y_{m-1}) -> (grad_x, ...)``:
        the inputs and the incoming gradient of each returned output, giving
        one gradient per float input, in input order. None checks against the
        finite difference.
    eps : float, default 5e-3
        The finite-difference step, near the cube root of float32's machine
        epsilon (the step that balances truncation and rounding for inputs of
        order 1). Scale it with your inputs' magnitude.
    atol, rtol : float, default 1e-5 and 1e-2
        The absolute and relative tolerances above. float32 central
        differences agree to about three significant digits, so ``rtol`` is
        much larger than the float64 default of PyTorch's gradcheck.
    seed : int, default 0
        Seed of the upstream gradient and the directions.
    raise_exception : bool, default True
        Raise :class:`GradcheckError` on a disagreement; with False, return
        False instead.
    numeric_mode, device
        As in :class:`Session`.

    Returns
    -------
    bool
        True when every input agrees.

    Example
    -------
    ``MyTanh`` (forward ``y = tanh(x)`` saving ``y``, backward
    ``grad_y * (1 - y * y)``) passes:

    >>> import numpy as np
    >>> from mojolearn import ops
    >>> zeros = np.zeros((4, 3), np.float32)
    >>> with ops.Session() as cap:
    ...     x = cap.tensor(zeros)
    ...     forward = cap.capture(inputs=[x], outputs=[cap.tanh(x)])
    ...     y, grad_y = cap.tensor(zeros), cap.tensor(zeros)
    ...     dx = cap.mul(grad_y, cap.sub(cap.full((1, 1), 1.0), cap.mul(y, y)))
    ...     backward = cap.capture(inputs=[y, grad_y], outputs=[dx])
    ...     x0 = np.random.default_rng(0).standard_normal((4, 3)).astype(np.float32)
    ...     ops.gradcheck(forward, backward, [x0], saved=[0])
    True
    """
    from ._optional_numpy import require_numpy
    from .ops import Session
    np = require_numpy("mojolearn.ops.gradcheck")

    kinds = tuple(forward.input_kinds)
    if len(inputs) != len(kinds):
        raise ValueError(f"gradcheck: the forward program takes {len(kinds)} inputs, got {len(inputs)}")
    returned = len(forward.output_shapes) if outputs is None else int(outputs)
    out_shapes = [tuple(shape) for shape in forward.output_shapes[:returned]]
    floats = [i for i, kind in enumerate(kinds) if kind == 0]
    rng = np.random.default_rng(seed)
    # glue: seeded test data of the declared shapes (testing tool), no math on the user's data
    arrays = [np.ascontiguousarray(a, dtype=np.float32) if kind == 0 else np.ascontiguousarray(a)
              for a, kind in zip(inputs, kinds)]
    upstream = [rng.standard_normal(shape).astype(np.float32) for shape in out_shapes]
    directions = {i: rng.uniform(-1.0, 1.0, arrays[i].shape).astype(np.float32) for i in floats}

    # The rule's vector-Jacobian products, through the native tape as in training.
    with Session(record=True, numeric_mode=numeric_mode, device=device) as s:
        op = s.custom_op(forward, backward, saved=saved, outputs=outputs)
        xs = [s.tensor(a, requires_grad=True) if kind == 0 else s.ids(a) for a, kind in zip(arrays, kinds)]
        s.backward(_weighted_sum(s, _as_tuple(s.custom(op, *xs)), [s.tensor(v) for v in upstream]))
        rule = {i: s.grad(xs[i]) for i in floats}

    mismatches = []
    with Session(numeric_mode=numeric_mode, device=device) as e:
        xs = [e.tensor(a) if kind == 0 else e.ids(a) for a, kind in zip(arrays, kinds)]
        vs = [e.tensor(v) for v in upstream]
        if expected is not None:
            want = _as_tuple(expected.run(*arrays, *upstream))
            if len(want) != len(floats):
                raise ValueError(f"gradcheck: the expected program returns {len(want)} gradients; "
                                 f"it must return {len(floats)}, one per float input")
            for slot, i in enumerate(floats):
                got, ref = e.tensor(rule[i]), e.tensor(np.asarray(want[slot], dtype=np.float32))
                if e.shape(got) != e.shape(ref):
                    raise ValueError(f"gradcheck: expected gradient {slot} has shape {e.shape(ref)}; "
                                     f"input {i} has shape {e.shape(got)}")
                diff = e.abs(e.sub(got, ref))
                largest = e.scalar(e.max(diff))
                excess = e.scalar(e.max(e.sub(diff, e.scale(e.abs(ref), rtol))))
                entry = {"input": i, "shape": arrays[i].shape, "difference": largest,
                         "tolerance": atol, "expected_max": e.scalar(e.max(e.abs(ref)))}
                if not excess <= atol:  # NaN disagrees
                    mismatches.append(entry)
        else:
            op = e.custom_op(forward, backward, saved=saved, outputs=outputs)
            magnitude = e.scalar(_weighted_sum(e, _as_tuple(e.custom(op, *xs)), vs, absolute=True))
            count = sum(int(np.prod(shape)) for shape in out_shapes)
            noise = (2.0 + math.log2(max(count, 1))) * _FLOAT32_ROUNDOFF * magnitude / eps
            for i in floats:
                d = e.tensor(directions[i])
                step = e.scale(d, eps)
                sides = []
                for moved in (e.add(xs[i], step), e.sub(xs[i], step)):
                    args = list(xs)
                    args[i] = moved
                    sides.append(_weighted_sum(e, _as_tuple(e.custom(op, *args)), vs))
                numerical = e.scalar(e.scale(e.sub(sides[0], sides[1]), 0.5 / eps))
                analytical = e.scalar(e.sum(e.mul(e.tensor(rule[i]), d)))
                difference = abs(analytical - numerical)
                tolerance = atol + rtol * max(abs(analytical), abs(numerical)) + noise
                entry = {"input": i, "shape": arrays[i].shape, "analytical": analytical,
                         "numerical": numerical, "difference": difference, "tolerance": tolerance}
                if not difference <= tolerance:  # NaN disagrees
                    mismatches.append(entry)

    if not mismatches:
        return True
    if not raise_exception:
        return False
    raise GradcheckError(_message(mismatches, expected is not None), mismatches)


def _as_tuple(value):
    return tuple(value) if isinstance(value, (tuple, list)) else (value,)


def _weighted_sum(s, ys, vs, absolute=False):
    """sum_j sum(v_j * y_j) (or of |v_j * y_j|) as a (1, 1) tensor, natively."""
    total = None
    for y, v in zip(ys, vs):
        term = s.mul(y, v)
        term = s.sum(s.abs(term) if absolute else term)
        total = term if total is None else s.add(total, term)
    return total


def _message(mismatches, against_expected):
    if against_expected:
        lines = [f"input {m['input']} (shape {m['shape']}): largest |rule - expected| = {m['difference']:.6g} "
                 f"exceeds atol {m['tolerance']:.3g} + rtol * |expected| (largest |expected| "
                 f"{m['expected_max']:.6g})" for m in mismatches]
        head = "gradcheck: the backward rule disagrees with the expected rule for "
    else:
        lines = [f"input {m['input']} (shape {m['shape']}): analytical {m['analytical']:.6g}, numerical "
                 f"{m['numerical']:.6g}, difference {m['difference']:.6g} > tolerance {m['tolerance']:.6g}"
                 for m in mismatches]
        head = "gradcheck: the backward rule disagrees with the finite difference of the forward for "
    names = ", ".join(f"input {m['input']}" for m in mismatches)
    return head + names + "\n  " + "\n  ".join(lines)
