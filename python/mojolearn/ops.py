# SPDX-License-Identifier: Apache-2.0
"""Numerical operations to compose your own algorithms on the GPU.

A :class:`Session` holds tensors on the device and runs one native operation
per call: matrix and batched products, elementwise math with broadcasting,
reductions over any axes, reshapes and axis permutations, concatenation and
splitting, softmax, dense and layer-norm layers, 2-D convolution and
pooling, dropout, row permutations and gathers, losses and metrics.
Chain them to build an inference or hybrid algorithm (a mixture of experts,
an ensemble with learned routing, statistics combined with a learned
representation) without writing a kernel.

In `identical` mode, mojolearn returns the same bits on Apple, NVIDIA and AMD GPUs.
Every operation here is the one mojolearn's own trainers run, so a
composition gives the same output bits on every vendor for the same inputs.

Tensors are float32 arrays of up to 8 dimensions addressed by integer
handles; a scalar is a ``(1, 1)`` tensor and a 1-D input becomes one column.
Operations on a matrix keep the shapes they always had: ``sum(x, axis=0)``
of an ``(n, k)`` matrix is ``(1, k)``. Values stay on the
device until you read them back with :meth:`Session.numpy` or
:meth:`Session.scalar`. A read-back is a host step: use :meth:`Session.scalar`
when your Python code must branch on a computed value, and keep everything
else on the device.

With ``Session(record=True)`` the same composition is differentiable: create
the trainable inputs with ``requires_grad=True``, compute a scalar loss, call
:meth:`Session.backward` and read each gradient with :meth:`Session.grad`.
To train, make an optimizer with :meth:`Session.optimizer` and call
:meth:`Session.step` after each backward: the parameters are updated in
place on the device, with the optimizer's state kept there too, so a
training loop reads nothing back unless you ask for it.

``backward(loss, create_graph=True)`` records the gradient computation too,
so each gradient is a tensor (:meth:`Session.grad_tensor`) that you can
differentiate again: second derivatives, Hessian-vector products
(:meth:`Session.hvp`), Jacobian-vector products (:meth:`Session.jvp`) and
vector-Jacobian products (:meth:`Session.vjp`).

:meth:`Session.capture` turns the operations of an inference session into a
:class:`Program`: :meth:`Program.run` runs the whole model on new inputs in
one call, with the same output bits as the operations called one by one, and
:meth:`Program.save` writes it for :meth:`Session.load_program` on any
install.

Example
-------
>>> import numpy as np
>>> from mojolearn import ops
>>> rng = np.random.default_rng(0)
>>> with ops.Session() as s:
...     x = s.tensor(rng.standard_normal((8, 4), dtype=np.float32))
...     w = s.tensor(rng.standard_normal((4, 3), dtype=np.float32))
...     p = s.softmax(s.matmul(x, w))
...     probs = s.numpy(p)
"""
import os

from ._optional_numpy import require_numpy
np = require_numpy('mojolearn.ops')

from . import _backend
from ._distributed_program import DistributedProgram as _DistributedProgram

__all__ = ["Session", "Program"]

# autograd/ops_api.mojo OPS_*
_OP = {
    "matmul": 1, "add": 2, "sub": 3, "mul": 4, "div": 5, "relu": 6, "sigmoid": 7,
    "tanh": 8, "exp": 9, "log": 10, "sqrt": 11, "scale": 12, "transpose": 13,
    "reshape": 14, "sum": 15, "mean": 16, "softmax": 17, "log_softmax": 18,
    "linear": 19, "layer_norm": 20, "dropout": 21, "gather_rows": 22,
    "permutation": 23, "mse": 24, "mae": 25, "huber": 26, "bce_with_logits": 27,
    "cross_entropy": 28, "detach": 29, "full": 30, "copy": 31,
    "embedding": 32,
    "reshape_n": 33, "permute": 34, "sum_n": 35, "mean_n": 36, "max": 37, "matmul_n": 38,
    "concat": 39, "narrow": 40, "take": 41, "softmax_n": 42, "log_softmax_n": 43,
    "conv2d": 44, "max_pool2d": 45, "avg_pool2d": 46,
    "erf": 48, "gelu": 49, "softplus": 50, "silu": 51, "leaky_relu": 52, "elu": 53,
    "hard_sigmoid": 54, "hard_swish": 55, "clip": 56, "abs": 57, "floor": 58, "ceil": 59,
    "round": 60, "sin": 61, "cos": 62, "sign": 63, "reciprocal": 64, "pow": 65,
    "minimum": 66, "maximum": 67, "where": 68, "compare": 69, "logical": 70,
    "expand": 71, "argmax": 72, "pad": 73, "any_all": 74, "cumsum": 75, "index": 76,
}
# autograd/ops_api.mojo OPS_COMPARE and OPS_LOGICAL kinds
_COMPARE = {"eq": 0, "gt": 1, "lt": 2, "ge": 3, "le": 4, "ne": 5}
_GELU = {"none": 0, "tanh": 1}
# the largest tensor rank (autograd/rankn.mojo RN_MAX_RANK)
_MAX_RANK = 8
_METRIC = {"accuracy": 1, "log_loss": 2, "roc_auc": 3}
_ACTIVATION = {None: 0, "none": 0, "relu": 1, "sigmoid": 2, "tanh": 3, "softmax": 4}
_REDUCTION = {"mean": 0, "sum": 1}
_AXIS = {None: -1, 0: 0, 1: 1}
# sequence/ops.mojo OPT_* and autograd/model.mojo SCHED_* (Sequential.compile's codes)
_OPTIMIZER = {"sgd": 0, "adam": 1, "adamw": 2}
_SCHEDULE = {"constant": 0, "step": 1, "exponential": 2}
# the state buffers autograd/ops_api.mojo optimizer_state returns, by kind
_STATE_NAMES = {"sgd": ("momentum_buffer", None, None), "adam": ("exp_avg", "exp_avg_sq", None),
                "adamw": ("exp_avg", "exp_avg_sq", None)}
# autograd/ops_api.mojo OPS_PACK_*: the packed argument block of one call
_PACK_H, _PACK_I, _PACK_F = 3, 7, 15
_PACK_HANDLES, _PACK_INTS, _PACK_FLOATS = 4, 8, 4
_PACK_WORDS = _PACK_F + (_PACK_FLOATS + 1) // 2


def _int(value, name, low=None):
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer")
    value = int(value)
    if low is not None and value < low:
        raise ValueError(f"{name} must be >= {low}")
    return value


def _seed_words(seed):
    seed = _int(seed, "seed", 0)
    if seed >= 2 ** 64:
        raise ValueError("seed must be in [0, 2**64)")
    return [seed & 0xFFFFFFFF, seed >> 32]


def _reduction(reduction):
    if reduction not in _REDUCTION:
        raise ValueError("reduction must be 'mean' or 'sum'")
    return _REDUCTION[reduction]


def _axis(axis):
    if isinstance(axis, bool) or axis not in _AXIS:
        raise ValueError("axis must be None (all elements), 0 (down the rows) or 1 (across the columns)")
    return _AXIS[axis]


def _axes(axis, name="axis"):
    """[] for None, else the listed axes as integers (Mojo normalizes them)."""
    if axis is None:
        return []
    if isinstance(axis, (tuple, list)):
        if len(axis) > _MAX_RANK:
            raise ValueError(f"{name} lists more than {_MAX_RANK} axes")
        return [_int(a, name) for a in axis]  # glue: at most eight axis arguments
    return [_int(axis, name)]


def _keepdims(keepdims):
    if keepdims is None:
        return -1
    return 1 if keepdims else 0


def _dims(shape, name="shape", low=None):
    if isinstance(shape, (int, np.integer)) and not isinstance(shape, bool):
        shape = (shape,)
    shape = tuple(shape)
    if not shape or len(shape) > _MAX_RANK:
        raise ValueError(f"{name} must have 1 to {_MAX_RANK} dimensions")
    return [_int(d, name, low) for d in shape]  # glue: at most eight dimension arguments


class Session:
    """Device tensors and the native operations that combine them.

    Parameters
    ----------
    record : bool, default False
        Record each operation so :meth:`backward` can differentiate the
        composition. Values are the same bits either way.
    numeric_mode : {'identical', 'fast'} or None, default None
        Numeric mode. None uses the library default. ``'fast'`` is available
        on Apple GPUs only.

    Every operation takes tensor handles and returns the integer handle of a
    new tensor; nothing is copied back to the host until you call
    :meth:`numpy`, :meth:`scalar` or :meth:`grad`. The metrics
    (:meth:`accuracy`, :meth:`log_loss`, :meth:`roc_auc`) return a float.

    A session is a context manager; :meth:`close` (or leaving the ``with``
    block) returns its device memory.

    Examples
    --------
    Compose an inference step and read a scalar back:

    >>> import numpy as np
    >>> from mojolearn import ops
    >>> rng = np.random.default_rng(0)
    >>> with ops.Session() as s:
    ...     x = s.tensor(rng.standard_normal((32, 4), dtype=np.float32))
    ...     w = s.tensor(rng.standard_normal((4, 2), dtype=np.float32))
    ...     h = s.linear(x, w, activation="tanh")
    ...     total = s.scalar(s.sum(h))

    Train a linear model with Adam, the parameters staying on the device:

    >>> X = rng.standard_normal((64, 4), dtype=np.float32)
    >>> Y = rng.standard_normal((64, 1), dtype=np.float32)
    >>> with ops.Session(record=True) as s:
    ...     w = s.tensor(np.zeros((4, 1), dtype=np.float32), requires_grad=True)
    ...     b = s.tensor(np.zeros((1, 1), dtype=np.float32), requires_grad=True)
    ...     x, y = s.tensor(X), s.tensor(Y)
    ...     opt = s.optimizer("adam", lr=0.01)
    ...     for epoch in range(20):
    ...         s.backward(s.mse(s.linear(x, w, b), y))
    ...         s.step(opt, [w, b])
    ...         s.reset()
    ...     weights = s.numpy(w)

    Record the same composition and differentiate a loss:

    >>> y = rng.standard_normal((32, 2), dtype=np.float32)
    >>> with ops.Session(record=True) as s:
    ...     x = s.tensor(rng.standard_normal((32, 4), dtype=np.float32))
    ...     w = s.tensor(rng.standard_normal((4, 2), dtype=np.float32), requires_grad=True)
    ...     loss = s.mse(s.matmul(x, w), s.tensor(y))
    ...     s.backward(loss)
    ...     grad_w = s.grad(w)
    """

    def __init__(self, record=False, numeric_mode=None):
        self.record = bool(record)
        self.numeric_mode = numeric_mode
        self._b = _backend.binding("_mojolearn_x_sequence", numeric_mode)
        self._id = int(self._b.ops_open(1 if self.record else 0))
        # One argument block per session, written in place each call and read
        # by address in Mojo (no list crosses into the binding). A binding
        # built before the packed entry takes the list call.
        self._packed = hasattr(self._b, "ops_op_packed")
        self._args = np.zeros(_PACK_WORDS, dtype=np.int64)
        self._argf = self._args[_PACK_F:].view(np.float32)
        self._args_addr = self._args.ctypes.data
        self._opt_kind = {}

    # ------------------------------------------------------------ lifetime
    def close(self):
        """Free every tensor of the session and its device memory."""
        sid, self._id = getattr(self, "_id", None), None
        if sid is not None:
            self._b.ops_close(sid)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def _sid(self):
        if self._id is None:
            raise ValueError("this session is closed")
        return self._id

    def reset(self):
        """Drop every computed tensor (and the recorded graph); tensors made
        with :meth:`tensor` stay valid. Call it between independent passes to
        reuse their memory."""
        self._b.ops_reset(self._sid())

    def free(self, h):
        """Release tensor ``h``; its memory is reused after :meth:`reset`."""
        self._b.ops_free(self._sid(), _int(h, "handle", 1))

    def profile_reset(self):
        """Clear the process-wide launch profile (recorded when the
        MOJOLEARN_AG_PROFILE environment variable is set)."""
        self._b.ops_profile_reset()

    def profile_report(self):
        """The process-wide launch profile as text: launches per operation."""
        return str(self._b.ops_profile_report())  # glue: diagnostic launch-count text over profile rows, no data

    # ------------------------------------------------------------ data
    def tensor(self, x, requires_grad=False):
        """Copy an array to the device; returns its handle.

        ``x`` is a scalar or an array of up to 8 dimensions of bool,
        integers, float16, float32 or float64, used as float32. A scalar is
        a ``(1, 1)`` tensor and a 1-D array one column; a 2-D array may have
        any memory layout; an array of 3 or more dimensions keeps its shape
        and must be C-contiguous (pass ``numpy.ascontiguousarray(x)``).
        ``requires_grad=True`` (recording sessions only) makes it a
        trainable input whose gradient :meth:`grad` returns after
        :meth:`backward`.
        """
        from ._cast_in import as_input, describe, matrix_view
        a = as_input(x)
        if a.ndim > _MAX_RANK:
            raise ValueError(f"x must have at most {_MAX_RANK} dimensions")
        if a.size == 0:
            raise ValueError("x must not be empty")
        flag = 1 if requires_grad else 0
        if a.ndim > 2:
            # the same memory as its row-major (rows, last axis) matrix view,
            # plus the full shape; it stays referenced until the binding has
            # copied it
            m = matrix_view(a, "x")
            return int(self._b.ops_upload(self._sid(), describe(m, "x"), [flag, a.ndim] + list(a.shape)))
        # `a` as it is (Mojo casts it to float32 on the executor); it stays
        # referenced until the binding has copied it
        return int(self._b.ops_upload(self._sid(), describe(a, "x"), [flag]))

    def ids(self, x):
        """Copy an array of integer ids to the device; returns its handle.

        ``x`` is an array of up to 8 dimensions of any integer dtype, held
        as int32 values. Pass the handle as the ``ids`` of
        :meth:`embedding` or the ``indices`` of :meth:`take`: the result is
        the same as passing the array itself, and :meth:`capture` can then
        name the ids as an input of a :class:`Program`, so a language model
        captured once runs on new token ids. The tensor takes no gradient;
        read its values with ``s.numpy(h).view(numpy.int32)``.

        Example
        -------
        >>> import numpy as np
        >>> from mojolearn import ops
        >>> rng = np.random.default_rng(0)
        >>> with ops.Session() as s:
        ...     table = s.tensor(rng.standard_normal((10, 4), dtype=np.float32))
        ...     tokens = s.ids(np.array([[3, 1, 4], [1, 5, 9]]))
        ...     h = s.embedding(table, tokens)
        ...     model = s.capture(inputs=[tokens], outputs=[h])
        ...     rows = model.run(np.array([[2, 7, 1], [8, 2, 8]]))
        """
        from ._cast_in import as_input, describe, matrix_view
        a = as_input(x)
        if a.dtype.kind not in "iu":
            raise TypeError(f"ids must be integers, got dtype {a.dtype}")
        if a.ndim > _MAX_RANK:
            raise ValueError(f"ids must have at most {_MAX_RANK} dimensions")
        if a.size == 0:
            raise ValueError("ids must not be empty")
        # flag word bit 1: an ids tensor (Mojo casts the integers to int32
        # words on the executor); `a` stays referenced until the binding has
        # copied it
        if a.ndim > 2:
            m = matrix_view(a, "ids")
            return int(self._b.ops_upload(self._sid(), describe(m, "ids"), [2, a.ndim] + list(a.shape)))
        return int(self._b.ops_upload(self._sid(), describe(a, "ids"), [2]))

    @staticmethod
    def _is_handle(v):
        return isinstance(v, (int, np.integer)) and not isinstance(v, bool)

    def shape(self, h):
        """The shape of tensor ``h`` as a tuple: ``(rows, cols)`` for a
        matrix, every dimension for a tensor of another rank."""
        dims = self._b.ops_shape(self._sid(), _int(h, "handle", 1))
        return tuple(int(d) for d in dims)  # glue: at most eight dimensions

    def numpy(self, h):
        """Tensor ``h``'s values as a new float32 array of its shape."""
        out = np.empty(self.shape(h), dtype=np.float32)
        self._b.ops_download(self._sid(), h, out.ctypes.data)
        return out

    def scalar(self, h):
        """The value of a ``(1, 1)`` tensor as a Python float, for branching
        in your code on a computed value."""
        return float(self._b.ops_scalar(self._sid(), _int(h, "handle", 1)))

    def full(self, shape, value):
        """A new tensor of ``shape`` (up to 8 dimensions) with every
        element ``value``."""
        return self._op("full", (), _dims(shape, low=1), [float(value)])

    def copy(self, x):
        """A copy of ``x`` that carries no gradient."""
        return self._op("copy", (x,))

    def detach(self, x):
        """``x`` as a constant: later operations pass no gradient back to it."""
        return self._op("detach", (x,))

    # ------------------------------------------------------------ the call
    def _op(self, name, handles, ints=(), floats=()):
        if self._packed and len(handles) <= _PACK_HANDLES and len(ints) <= _PACK_INTS and len(floats) <= _PACK_FLOATS:
            a = self._args
            a[0], a[1], a[2] = len(handles), len(ints), len(floats)
            for k, h in enumerate(handles):  # glue: at most three input handles
                a[_PACK_H + k] = _int(h, "handle", 1)
            for k, v in enumerate(ints):  # glue: at most four validated integer arguments
                a[_PACK_I + k] = v
            for k, v in enumerate(floats):  # glue: at most one float argument
                self._argf[k] = v
            return int(self._b.ops_op_packed(self._sid(), _OP[name], self._args_addr))
        hs = [_int(h, "handle", 1) for h in handles]  # glue: at most three input handles
        return int(self._b.ops_op(self._sid(), _OP[name], hs, list(ints), list(floats)))

    # ------------------------------------------------------------ elementwise and products
    def matmul(self, a, b):
        """The matrix product of the last two axes, ``a @ b``.

        Leading axes are batch axes and broadcast as in NumPy: a
        ``(batch, heads, m, k)`` tensor times a ``(batch, heads, k, n)`` or
        a ``(k, n)`` tensor gives ``(batch, heads, m, n)``.
        """
        return self._op("matmul", (a, b))

    def add(self, a, b):
        """``a + b`` with NumPy broadcasting: shapes are aligned on the
        right and an axis of length 1 stretches. For matrices, a ``(1, 1)``,
        ``(1, cols)`` or ``(rows, 1)`` operand broadcasts."""
        return self._op("add", (a, b))

    def sub(self, a, b):
        """``a - b`` with the broadcasting of :meth:`add`."""
        return self._op("sub", (a, b))

    def mul(self, a, b):
        """``a * b`` elementwise with the broadcasting of :meth:`add`."""
        return self._op("mul", (a, b))

    def div(self, a, b):
        """``a / b`` elementwise with the broadcasting of :meth:`add`."""
        return self._op("div", (a, b))

    def relu(self, x):
        """``max(x, 0)``."""
        return self._op("relu", (x,))

    def sigmoid(self, x):
        """``1 / (1 + exp(-x))``."""
        return self._op("sigmoid", (x,))

    def tanh(self, x):
        """The hyperbolic tangent."""
        return self._op("tanh", (x,))

    def exp(self, x):
        """The natural exponential."""
        return self._op("exp", (x,))

    def log(self, x):
        """The natural logarithm."""
        return self._op("log", (x,))

    def sqrt(self, x):
        """The square root."""
        return self._op("sqrt", (x,))

    def scale(self, x, factor):
        """``factor * x``."""
        return self._op("scale", (x,), (), [float(factor)])

    # ------------------------------------------------------------ shape and reductions
    def transpose(self, x, a=-2, b=-1):
        """``x`` with axes ``a`` and ``b`` swapped; the matrix transpose by
        default."""
        if a == -2 and b == -1:
            return self._op("transpose", (x,))
        return self._op("permute", (x,), [1, _int(a, "a"), _int(b, "b")])

    def permute(self, x, axes):
        """``x`` with its axes reordered: output axis ``k`` is input axis
        ``axes[k]``.

        Example
        -------
        >>> with ops.Session() as s:
        ...     x = s.tensor(np.zeros((2, 3, 4), dtype=np.float32))
        ...     s.shape(s.permute(x, (2, 0, 1)))
        (4, 2, 3)
        """
        return self._op("permute", (x,), [0] + _axes(tuple(axes), "axes"))

    def reshape(self, x, shape):
        """``x``'s elements, row-major, in a new ``shape`` with the same
        number of elements; one dimension may be -1 and is inferred."""
        dims = _dims(shape)
        if len(dims) == 2 and -1 not in dims:
            return self._op("reshape", (x,), [_int(dims[0], "rows", 1), _int(dims[1], "cols", 1)])
        return self._op("reshape_n", (x,), [0] + dims)

    def flatten(self, x, start=1, end=-1):
        """``x`` with axes ``start`` to ``end`` (inclusive) merged into
        one; by default every axis after the first."""
        return self._op("reshape_n", (x,), [1, _int(start, "start"), _int(end, "end")])

    def squeeze(self, x, axis=None):
        """``x`` without its length-1 axes (all of them, or those listed in
        ``axis``)."""
        return self._op("reshape_n", (x,), [2] + _axes(axis))

    def unsqueeze(self, x, axis):
        """``x`` with a new length-1 axis at position ``axis``."""
        return self._op("reshape_n", (x,), [3, _int(axis, "axis")])

    def sum(self, x, axis=None, keepdims=None):
        """The sum over ``axis``.

        ``axis=None`` sums every element into a ``(1, 1)`` tensor; an
        integer or a tuple of integers sums over those axes (negative axes
        count from the end). ``keepdims=True`` keeps each summed axis with
        length 1 and ``False`` removes it; the default keeps it for a
        matrix and removes it otherwise, so ``sum(x, axis=0)`` of an
        ``(n, k)`` matrix is ``(1, k)``.
        """
        return self._op("sum_n", (x,), [_keepdims(keepdims)] + _axes(axis))

    def mean(self, x, axis=None, keepdims=None):
        """The mean over ``axis``, with the axes and shapes of :meth:`sum`."""
        return self._op("mean_n", (x,), [_keepdims(keepdims)] + _axes(axis))

    def max(self, x, axis=None, keepdims=None):
        """The largest value over ``axis``, with the axes and shapes of
        :meth:`sum`. A NaN is larger than any number; among equal values the
        first position wins, and the gradient flows to that position only."""
        return self._op("max", (x,), [_keepdims(keepdims)] + _axes(axis))

    def softmax(self, x, axis=-1):
        """The softmax along ``axis``; by default the last axis (each row
        of a matrix)."""
        if axis == -1:
            return self._op("softmax", (x,))
        return self._op("softmax_n", (x,), [_int(axis, "axis")])

    def log_softmax(self, x, axis=-1):
        """The log-softmax along ``axis``; by default the last axis."""
        if axis == -1:
            return self._op("log_softmax", (x,))
        return self._op("log_softmax_n", (x,), [_int(axis, "axis")])

    # ------------------------------------------------------------ joining, splitting, picking
    def concat(self, xs, axis=0):
        """The tensors ``xs`` joined along ``axis``; every other axis must
        match."""
        hs = tuple(xs)
        if not hs:
            raise ValueError("concat needs at least one tensor")
        return self._op("concat", hs, [_int(axis, "axis")])

    def split(self, x, sizes_or_sections, axis=0):
        """``x`` cut along ``axis`` into a list of tensors.

        ``sizes_or_sections`` is the number of equal pieces, or a list of
        piece lengths that sum to the axis length. Each piece is a view of
        ``x`` (no copy); in a recording session the gradients of the pieces
        flow back into their places in ``x``'s gradient.
        """
        axis = _int(axis, "axis")
        if isinstance(sizes_or_sections, (int, np.integer)) and not isinstance(sizes_or_sections, bool):
            count = _int(sizes_or_sections, "sections", 1)
            tail = [1, count]
        else:
            sizes = [_int(v, "sizes", 1) for v in sizes_or_sections]  # glue: the piece lengths
            if not sizes:
                raise ValueError("sizes must list at least one length")
            count = len(sizes)
            tail = [0] + sizes
        return [self._op("narrow", (x,), [axis, k] + tail) for k in range(count)]  # glue: one view per piece

    def take(self, x, indices, axis=0):
        """The slices of ``x`` at ``indices`` along ``axis``.

        ``indices`` is an array of integers (any integer dtype) or a
        tensor made with :meth:`ids`, flattened in row-major order; the
        result replaces ``axis`` by one slice per index. An index outside
        ``[0, x.shape[axis])`` raises an error. In a recording session the
        gradient of ``x`` sums the gradients of every slice taken from each
        position, in ascending index position.
        """
        from ._cast_in import as_input, describe
        if self._is_handle(indices):
            # an ids tensor: Mojo reads its int32 words on the device
            return self._op("take", (x, indices), [_int(axis, "axis")])
        a = as_input(indices)
        if a.dtype.kind not in "iu":
            raise TypeError(f"indices must be integers, got dtype {a.dtype}")
        if a.size == 0:
            raise ValueError("indices must not be empty")
        if a.ndim > 2:
            if not a.flags.c_contiguous:
                raise ValueError("indices with more than two dimensions must be C-contiguous; "
                                 "pass numpy.ascontiguousarray(indices)")
            a = a.reshape(-1)
        # `a` as it is (Mojo casts it to int32 on the executor, row-major);
        # it stays referenced until the binding has copied it
        return self._op("take", (x,), describe(a, "indices") + [_int(axis, "axis")])

    # ------------------------------------------------------------ convolution and pooling
    def conv2d(self, x, weight, bias=None, stride=1, padding=0, groups=1):
        """A 2-D convolution (cross-correlation) of ``x``.

        ``x`` is ``(N, C, H, W)``, ``weight`` ``(out_channels, C // groups,
        kh, kw)`` and ``bias`` an optional tensor of ``out_channels`` values.
        The input is padded with ``padding`` zeros on every side and the
        kernel moves ``stride`` positions at a time; the result is
        ``(N, out_channels, H_out, W_out)``. With ``groups`` greater than 1
        the channels split into ``groups`` equal blocks and each block of
        outputs sees only its block of inputs; ``groups=C`` is a depthwise
        convolution.

        Example::

            y = s.conv2d(x, w, stride=1, padding=1, groups=4)
        """
        handles = (x, weight) if bias is None else (x, weight, bias)
        return self._op("conv2d", handles, [_int(stride, "stride", 1), _int(padding, "padding", 0),
                                            _int(groups, "groups", 1)])

    def max_pool2d(self, x, kernel_size, stride=None):
        """The largest value of each ``kernel_size`` window of an
        ``(N, C, H, W)`` tensor, windows ``stride`` apart (by default
        ``kernel_size``)."""
        k = _int(kernel_size, "kernel_size", 1)
        return self._op("max_pool2d", (x,), [k, k if stride is None else _int(stride, "stride", 1)])

    def avg_pool2d(self, x, kernel_size, stride=None):
        """The mean of each ``kernel_size`` window of an ``(N, C, H, W)``
        tensor, windows ``stride`` apart (by default ``kernel_size``)."""
        k = _int(kernel_size, "kernel_size", 1)
        return self._op("avg_pool2d", (x,), [k, k if stride is None else _int(stride, "stride", 1)])

    # ------------------------------------------------------------ more elementwise math
    def erf(self, x):
        """The error function."""
        return self._op("erf", (x,))

    def gelu(self, x, approximate="none"):
        """The Gaussian error linear unit ``x * Phi(x)``, with ``Phi`` the
        standard normal distribution function; ``approximate='tanh'`` uses
        the tanh approximation
        ``0.5 * x * (1 + tanh(sqrt(2 / pi) * (x + 0.044715 * x**3)))``."""
        if approximate not in _GELU:
            raise ValueError("approximate must be 'none' or 'tanh'")
        return self._op("gelu", (x,), [_GELU[approximate]])

    def softplus(self, x):
        """``log(1 + exp(x))``."""
        return self._op("softplus", (x,))

    def silu(self, x):
        """``x * sigmoid(x)`` (also called swish)."""
        return self._op("silu", (x,))

    def leaky_relu(self, x, alpha=0.01):
        """``x`` where ``x >= 0``, else ``alpha * x``."""
        return self._op("leaky_relu", (x,), (), [float(alpha)])

    def elu(self, x, alpha=1.0):
        """``x`` where ``x >= 0``, else ``alpha * (exp(x) - 1)``."""
        return self._op("elu", (x,), (), [float(alpha)])

    def hard_sigmoid(self, x, alpha=0.2, beta=0.5):
        """``alpha * x + beta`` clipped to ``[0, 1]``."""
        return self._op("hard_sigmoid", (x,), (), [float(alpha), float(beta)])

    def hard_swish(self, x):
        """``x * hard_sigmoid(x, alpha=1/6, beta=0.5)``."""
        return self._op("hard_swish", (x,))

    def clip(self, x, min=None, max=None):
        """``x`` limited to ``[min, max]``; a bound left as None does not
        limit. A NaN stays NaN."""
        low = float("-inf") if min is None else float(min)
        high = float("inf") if max is None else float(max)
        return self._op("clip", (x,), (), [low, high])

    def abs(self, x):
        """The absolute value."""
        return self._op("abs", (x,))

    def floor(self, x):
        """The largest integer not above ``x``. No gradient flows through it."""
        return self._op("floor", (x,))

    def ceil(self, x):
        """The smallest integer not below ``x``. No gradient flows through it."""
        return self._op("ceil", (x,))

    def round(self, x):
        """``x`` rounded to the nearest integer, a half to the even one (as
        ``numpy.round``). No gradient flows through it."""
        return self._op("round", (x,))

    def sin(self, x):
        """The sine."""
        return self._op("sin", (x,))

    def cos(self, x):
        """The cosine."""
        return self._op("cos", (x,))

    def sign(self, x):
        """-1, 0 or 1 by the sign of ``x``. No gradient flows through it."""
        return self._op("sign", (x,))

    def reciprocal(self, x):
        """``1 / x``."""
        return self._op("reciprocal", (x,))

    def pow(self, x, exponent):
        """``x ** exponent`` elementwise, ``exponent`` a tensor that
        broadcasts with ``x`` as in :meth:`add`. A negative ``x`` takes an
        integer exponent. For a number exponent use :meth:`pow_scalar`."""
        return self._op("pow", (x, exponent))

    def pow_scalar(self, x, exponent):
        """``x ** exponent`` for a number ``exponent``. A negative ``x``
        takes an integer exponent."""
        # a handle is an int too, so the number form is its own method
        return self._op("pow", (x,), (), [float(exponent)])

    def minimum(self, a, b):
        """The smaller of ``a`` and ``b`` elementwise, with the broadcasting
        of :meth:`add`. A NaN in either gives NaN; ``-0.0`` is below ``0.0``."""
        return self._op("minimum", (a, b))

    def maximum(self, a, b):
        """The larger of ``a`` and ``b`` elementwise, with the broadcasting
        of :meth:`add`. A NaN in either gives NaN; ``0.0`` is above ``-0.0``."""
        return self._op("maximum", (a, b))

    def where(self, condition, a, b):
        """``a`` where ``condition`` is nonzero, else ``b``; the three
        broadcast together as in NumPy. The gradient flows to ``a`` and
        ``b``."""
        return self._op("where", (condition, a, b))

    def _compare(self, a, b, kind):
        return self._op("compare", (a, b), [_COMPARE[kind]])

    def equal(self, a, b):
        """1.0 where ``a == b``, else 0.0, with the broadcasting of :meth:`add`."""
        return self._compare(a, b, "eq")

    def not_equal(self, a, b):
        """1.0 where ``a != b``, else 0.0."""
        return self._compare(a, b, "ne")

    def greater(self, a, b):
        """1.0 where ``a > b``, else 0.0."""
        return self._compare(a, b, "gt")

    def greater_equal(self, a, b):
        """1.0 where ``a >= b``, else 0.0."""
        return self._compare(a, b, "ge")

    def less(self, a, b):
        """1.0 where ``a < b``, else 0.0."""
        return self._compare(a, b, "lt")

    def less_equal(self, a, b):
        """1.0 where ``a <= b``, else 0.0."""
        return self._compare(a, b, "le")

    def logical_not(self, x):
        """1.0 where ``x`` is zero, else 0.0."""
        return self._op("logical", (x,), [0])

    def logical_and(self, a, b):
        """1.0 where both ``a`` and ``b`` are nonzero, else 0.0."""
        return self._op("logical", (a, b), [1])

    def logical_or(self, a, b):
        """1.0 where ``a`` or ``b`` is nonzero, else 0.0."""
        return self._op("logical", (a, b), [2])

    def expand(self, x, shape):
        """``x`` broadcast to ``shape`` as NumPy broadcasts it (the result's
        shape is the broadcast of the two). The gradient sums back over the
        broadcast axes."""
        return self._op("expand", (x,), _dims(shape, low=1))

    def argmax(self, x, axis=-1, keepdims=False, int32=False):
        """The position of the largest value along ``axis``. A NaN is larger
        than any number and among equal values the first position wins.

        The positions are float values by default, so later operations can
        use them; ``int32=True`` stores them as int32 words instead, to read
        with ``s.numpy(h).view(np.int32)``. ``keepdims=True`` keeps the axis
        with length 1. No gradient flows through it.
        """
        return self._op("argmax", (x,), [_int(axis, "axis"), 1 if keepdims else 0, 1 if not int32 else 0])

    def pad(self, x, pads, value=0.0):
        """``x`` padded with ``value``.

        ``pads`` lists two counts per axis, as ONNX ``Pad`` does: every
        axis's count before its first element, then every axis's count after
        its last one. A negative count removes elements instead.
        """
        counts = [_int(p, "pads") for p in pads]  # glue: two integer arguments per axis
        if not counts or len(counts) > 2 * _MAX_RANK:
            raise ValueError(f"pads must list two counts for each of 1 to {_MAX_RANK} axes")
        return self._op("pad", (x,), counts, [float(value)])

    def any(self, x, axis=None, keepdims=None):
        """1.0 where some element over ``axis`` is nonzero, else 0.0, with
        the axes and shapes of :meth:`sum`. A NaN counts as nonzero. No
        gradient flows through it."""
        return self._op("any_all", (x,), [0, _keepdims(keepdims)] + _axes(axis))

    def all(self, x, axis=None, keepdims=None):
        """1.0 where every element over ``axis`` is nonzero, else 0.0, with
        the axes and shapes of :meth:`sum`. A NaN counts as nonzero. No
        gradient flows through it."""
        return self._op("any_all", (x,), [1, _keepdims(keepdims)] + _axes(axis))

    def cumsum(self, x, axis, exclusive=False, reverse=False):
        """The running sum of ``x`` along ``axis``; the result has ``x``'s shape.

        ``exclusive=True`` leaves each position's own value out (the first
        position is 0) and ``reverse=True`` sums from the last position to
        the first, as ONNX ``CumSum`` does. The additions follow one fixed
        order, so the result has the same bits on every device. The
        gradient flows back to ``x``.

        >>> h = s.cumsum(s.tensor(np.ones((2, 4), dtype=np.float32)), axis=1)
        >>> s.numpy(h)[0]
        array([1., 2., 3., 4.], dtype=float32)
        """
        return self._op("cumsum", (x,), [_int(axis, "axis"), 1 if exclusive else 0, 1 if reverse else 0])

    def index(self, x, indices, axes=None):
        """``x`` picked at several index tensors at once, as ``x[i, j]`` with
        integer arrays in NumPy or torch.

        ``indices`` lists tensors of integer positions (stored as float
        values, a negative one counting from the end of its axis), one for
        each of ``axes`` (ascending; by default the leading axes). The
        index tensors broadcast together; the result's axes are their
        broadcast shape followed by ``x``'s other axes, except that when the
        indexed axes are consecutive the broadcast axes stand where the
        first of them stood. A position outside its axis raises an error.
        The gradient flows back to ``x``; the index tensors take none.

        >>> x = s.tensor(np.arange(12, dtype=np.float32).reshape(3, 4))
        >>> i = s.tensor(np.array([[0, 2]], dtype=np.float32))
        >>> j = s.tensor(np.array([[1, 3]], dtype=np.float32))
        >>> s.numpy(s.index(x, (i, j)))
        array([[ 1., 11.]], dtype=float32)
        """
        indices = tuple(indices)
        if not indices or len(indices) > _MAX_RANK:
            raise ValueError(f"index takes 1 to {_MAX_RANK} index tensors")
        if axes is None:
            axes = range(len(indices))
        axes = [_int(a, "axes") for a in axes]  # glue: at most eight axis arguments
        if len(axes) != len(indices):
            raise ValueError("index takes one axis for each index tensor")
        return self._op("index", (x,) + indices, axes)

    # ------------------------------------------------------------ layers
    def linear(self, x, weight, bias=None, activation=None):
        """A dense layer ``activation(x @ weight + bias)``.

        The last axis of ``x`` is the feature axis: an ``(..., inputs)``
        tensor gives ``(..., outputs)``.
        ``weight`` is ``(inputs, outputs)``, ``bias`` an optional
        ``(1, outputs)`` tensor, ``activation`` one of None, ``'relu'``,
        ``'sigmoid'``, ``'tanh'`` or ``'softmax'``.
        """
        if activation not in _ACTIVATION:
            raise ValueError("activation must be None, 'relu', 'sigmoid', 'tanh' or 'softmax'")
        handles = (x, weight) if bias is None else (x, weight, bias)
        return self._op("linear", handles, [_ACTIVATION[activation]])

    def layer_norm(self, x, weight, bias, eps=1e-5):
        """Normalize along the last axis to zero mean and unit variance,
        then scale by ``weight`` and shift by ``bias`` (both ``(1, features)``,
        the last axis's length)."""
        if not eps > 0:
            raise ValueError("eps must be positive")
        return self._op("layer_norm", (x, weight, bias), (), [float(eps)])

    def dropout(self, x, p, seed=0, counter=0):
        """Zero each element with probability ``p`` and scale the rest by
        ``1 / (1 - p)``. The mask depends only on ``seed``, ``counter`` and
        the element position: use a new ``counter`` for every step."""
        if not 0 <= p < 1:
            raise ValueError("p must be in [0, 1)")
        counter = _int(counter, "counter", 0)
        return self._op("dropout", (x,), _seed_words(seed) + [counter], [float(p)])

    # ------------------------------------------------------------ data steps
    def permutation(self, n, seed=0, epoch=0):
        """A random order of the rows ``0 .. n-1`` as an ``(n, 1)`` tensor,
        determined by ``seed`` and ``epoch``."""
        n = _int(n, "n", 1)
        return self._op("permutation", (), [n] + _seed_words(seed) + [_int(epoch, "epoch", 0)])

    def gather_rows(self, x, indices, start=0, count=None):
        """The rows of ``x`` listed in ``indices[start:start + count]``.

        ``indices`` is a :meth:`permutation` of at most ``x``'s row count.
        The result carries no gradient.
        """
        count = 0 if count is None else _int(count, "count", 1)
        return self._op("gather_rows", (x, indices), [_int(start, "start", 0), count])

    def embedding(self, table, ids, padding_idx=None):
        """Rows of ``table`` picked by integer ``ids``: ``table[ids]``.

        ``table`` is a ``(num_embeddings, embedding_dim)`` tensor and ``ids``
        an array of integers (any integer dtype) in ``[0, num_embeddings)``
        or a tensor of them made with :meth:`ids`, flattened in row-major
        order;
        the result has one row per id. An id outside that range raises an
        error. In a recording session the gradient of ``table`` sums, for
        each row, the gradients of every position that picked it, in
        ascending position order, and row ``padding_idx`` gets a zero
        gradient.

        Example
        -------
        >>> with ops.Session(record=True) as s:
        ...     w = s.tensor(np.ones((10, 4), dtype=np.float32), requires_grad=True)
        ...     x = s.embedding(w, np.array([3, 1, 3]))
        ...     s.backward(s.sum(x))
        ...     g = s.grad(w)
        """
        from ._cast_in import as_input, describe
        if self._is_handle(ids):
            pad = -1 if padding_idx is None else _int(padding_idx, "padding_idx", 0)
            # an ids tensor: Mojo reads its int32 words on the device
            return self._op("embedding", (table, ids), [pad])
        a = as_input(ids)
        if a.dtype.kind not in "iu":
            raise TypeError(f"ids must be integers, got dtype {a.dtype}")
        if a.size == 0:
            raise ValueError("ids must not be empty")
        if a.ndim > 2:
            # a C-contiguous array flattens to a view of the same memory
            if not a.flags.c_contiguous:
                raise ValueError("ids with more than two dimensions must be C-contiguous; "
                                 "pass numpy.ascontiguousarray(ids)")
            a = a.reshape(-1)
        pad = -1 if padding_idx is None else _int(padding_idx, "padding_idx", 0)
        # `a` as it is (Mojo casts it to int32 on the executor, row-major);
        # it stays referenced until the binding has copied it
        return self._op("embedding", (table,), describe(a, "ids") + [pad])

    # ------------------------------------------------------------ losses (differentiable)
    def mse(self, prediction, target, reduction="mean"):
        """The squared error, averaged or summed: a ``(1, 1)`` tensor."""
        return self._op("mse", (prediction, target), [_reduction(reduction)])

    def mae(self, prediction, target, reduction="mean"):
        """The absolute error, averaged or summed."""
        return self._op("mae", (prediction, target), [_reduction(reduction)])

    def huber(self, prediction, target, delta=1.0, reduction="mean"):
        """The Huber loss with threshold ``delta``, averaged or summed."""
        if not delta > 0:
            raise ValueError("delta must be positive")
        return self._op("huber", (prediction, target), [_reduction(reduction)], [float(delta)])

    def bce_with_logits(self, logits, target, reduction="mean"):
        """Binary cross-entropy of ``sigmoid(logits)`` against targets in
        ``[0, 1]``, computed stably from the logits."""
        return self._op("bce_with_logits", (logits, target), [_reduction(reduction)])

    def cross_entropy(self, logits, codes, reduction="mean"):
        """Softmax cross-entropy of ``(..., classes)`` logits against class
        codes (the class index as a float), one per position of the leading
        axes: ``(rows, 1)`` codes for ``(rows, classes)`` logits."""
        return self._op("cross_entropy", (logits, codes), [_reduction(reduction)])

    # ------------------------------------------------------------ metrics (values)
    def _metric(self, name, y_true, prediction):
        hs = [_int(y_true, "y_true", 1), _int(prediction, "prediction", 1)]
        return float(self._b.ops_metric(self._sid(), _METRIC[name], hs))

    def accuracy(self, y_true, prediction):
        """The fraction of rows classified correctly, as a float.

        ``y_true`` holds class codes ``(n, 1)``; ``prediction`` holds codes
        ``(n, 1)`` or scores ``(n, classes)`` (the first highest wins)."""
        return self._metric("accuracy", y_true, prediction)

    def log_loss(self, y_true, probability):
        """The mean negative log-likelihood of the class codes ``(n, 1)``
        under the probabilities ``(n, classes)`` (or ``(n, 1)``, the
        positive class), as a float."""
        return self._metric("log_loss", y_true, probability)

    def roc_auc(self, y_true, scores):
        """The area under the ROC curve of binary labels ``(n, 1)`` (1
        positive) against scores ``(n, 1)``, as a float."""
        return self._metric("roc_auc", y_true, scores)

    # ------------------------------------------------------------ gradients
    def backward(self, loss, accumulate=False, create_graph=False):
        """Differentiate the ``(1, 1)`` tensor ``loss`` with respect to every
        input made with ``requires_grad=True`` (recording sessions only).

        Parameters
        ----------
        loss : int
            Handle of a ``(1, 1)`` tensor.
        accumulate : bool, default False
            Add to the gradients of the previous backward instead of
            replacing them.
        create_graph : bool, default False
            Record the gradient computation itself, so each gradient is a
            tensor (:meth:`grad_tensor`) you can keep computing with and
            differentiate again with a second backward. Cannot be combined
            with ``accumulate=True``: add the previous gradient tensor with
            :meth:`add` instead.

        Example
        -------
        The second derivative of ``x ** 3`` at ``x = 2`` is ``12``:

        >>> with ops.Session(record=True) as s:
        ...     x = s.tensor(np.array([[2.0]], dtype=np.float32), requires_grad=True)
        ...     y = s.mul(s.mul(x, x), x)
        ...     s.backward(y, create_graph=True)
        ...     s.backward(s.grad_tensor(x))
        ...     d2 = s.grad(x)
        """
        loss = _int(loss, "loss", 1)
        if create_graph:
            if accumulate:
                raise ValueError("accumulate=True cannot be combined with create_graph=True: "
                                 "add the previous gradient tensor with add() instead")
            self._b.ops_backward_graph(self._sid(), loss)
            return
        self._b.ops_backward(self._sid(), loss, 1 if accumulate else 0)

    def grad(self, h):
        """The gradient of tensor ``h`` from the last :meth:`backward`, as a
        float32 array of its shape."""
        out = np.empty(self.shape(h), dtype=np.float32)
        self._b.ops_grad(self._sid(), h, out.ctypes.data)
        return out

    def grad_tensor(self, h):
        """The gradient of tensor ``h`` from the last
        ``backward(..., create_graph=True)``, as a tensor handle.

        Unlike :meth:`grad`, which copies the values back as an array, the
        handle stays on the device and is differentiable: use it in further
        operations and call :meth:`backward` on a scalar built from it to
        get second derivatives. Raises if the last backward did not record
        its gradients.
        """
        return int(self._b.ops_grad_handle(self._sid(), _int(h, "handle", 1)))

    def vjp(self, output, inputs, v, create_graph=False):
        """The vector-Jacobian products of ``output`` for the seed ``v``.

        Parameters
        ----------
        output : int
            Handle of the tensor to differentiate, of any shape.
        inputs : int or list of int
            The tensors to differentiate with respect to; each must be made
            with ``requires_grad=True`` or computed from one.
        v : int
            Handle of a tensor of ``output``'s shape.
        create_graph : bool, default False
            Return differentiable tensors that a later backward can
            differentiate again.

        Returns
        -------
        list of int
            One handle per input, of that input's shape: ``J^T v``, where
            ``J`` is the Jacobian of ``output`` with respect to the input.
            Like :meth:`backward`, it replaces the gradients :meth:`grad`
            returns.

        Example
        -------
        >>> with ops.Session(record=True) as s:
        ...     x = s.tensor(np.ones((3, 2), dtype=np.float32), requires_grad=True)
        ...     y = s.tanh(x)
        ...     (jt_v,) = s.vjp(y, [x], s.full((3, 2), 1.0))
        ...     values = s.numpy(jt_v)
        """
        hs = self._handles(inputs)
        res = self._b.ops_vjp(self._sid(), _int(output, "output", 1), hs, _int(v, "v", 1),
                              1 if create_graph else 0)
        return [int(h) for h in res]  # glue: one result handle per input

    def jvp(self, output, inputs, u):
        """The Jacobian-vector product of ``output`` along the directions ``u``.

        Parameters
        ----------
        output : int
            Handle of the tensor to differentiate, of any shape.
        inputs : int or list of int
            The tensors to differentiate with respect to; each must be made
            with ``requires_grad=True`` or computed from one.
        u : int or list of int
            One direction per input, each of its input's shape.

        Returns
        -------
        int
            Handle of a tensor of ``output``'s shape: the sum over the
            inputs of ``J u``, where ``J`` is the Jacobian of ``output`` with
            respect to that input. It replaces the gradients :meth:`grad`
            returns.

        Example
        -------
        >>> with ops.Session(record=True) as s:
        ...     x = s.tensor(np.ones((3, 2), dtype=np.float32), requires_grad=True)
        ...     y = s.tanh(x)
        ...     j_u = s.numpy(s.jvp(y, [x], [s.full((3, 2), 1.0)]))
        """
        hs = self._handles(inputs)
        us = self._handles(u)
        if len(us) != len(hs):
            raise ValueError("u must give one direction per input")
        return int(self._b.ops_jvp(self._sid(), _int(output, "output", 1), hs, us))

    def hvp(self, loss, params, v):
        """The Hessian-vector products of a scalar ``loss`` along ``v``.

        Parameters
        ----------
        loss : int
            Handle of a ``(1, 1)`` tensor.
        params : int or list of int
            The tensors the Hessian is taken with respect to; each must be
            made with ``requires_grad=True`` or computed from one.
        v : int or list of int
            One direction per parameter, each of its parameter's shape.

        Returns
        -------
        list of int
            One handle per parameter, of its shape: the parameter's block of
            ``H v``, where ``H`` is the Hessian of ``loss`` with respect to
            all of ``params`` together. Where the loss has no curvature the
            result is zeros. It replaces the gradients :meth:`grad` returns.

        Example
        -------
        >>> rng = np.random.default_rng(0)
        >>> with ops.Session(record=True) as s:
        ...     x = s.tensor(rng.standard_normal((16, 3), dtype=np.float32))
        ...     y = s.tensor(rng.standard_normal((16, 1), dtype=np.float32))
        ...     w = s.tensor(np.zeros((3, 1), dtype=np.float32), requires_grad=True)
        ...     loss = s.mse(s.linear(x, w, activation="tanh"), y)
        ...     (hv,) = s.hvp(loss, [w], [s.full((3, 1), 1.0)])
        ...     values = s.numpy(hv)
        """
        hs = self._handles(params)
        vs = self._handles(v)
        if len(vs) != len(hs):
            raise ValueError("v must give one direction per parameter")
        res = self._b.ops_hvp(self._sid(), _int(loss, "loss", 1), hs, vs)
        return [int(h) for h in res]  # glue: one result handle per parameter

    @staticmethod
    def _handles(hs):
        if isinstance(hs, (int, np.integer)) and not isinstance(hs, bool):
            hs = [hs]
        hs = [_int(h, "handle", 1) for h in hs]  # glue: the call's input handles
        if not hs:
            raise ValueError("give at least one tensor")
        return hs

    # ------------------------------------------------------------ captured programs
    def capture(self, inputs, outputs):
        """Capture the operations that compute ``outputs`` from ``inputs``
        as a :class:`Program` that runs again on new arrays in one call.

        The session must not record (``Session(record=False)``). Every
        operation since the last :meth:`reset` that ``outputs`` depend on is
        kept, in order, with its arguments; tensors made with
        :meth:`tensor` that the outputs read and that are not inputs are the
        program's parameters, read where they are at each run (so a
        :meth:`step` between runs is seen by the next run). :meth:`embedding`
        and :meth:`take` are captured when their ids are a tensor made with
        :meth:`ids` (name it as an input to run on new ids); called with an
        index array they cannot be captured.

        Parameters
        ----------
        inputs : int or list of int
            Handles of the tensors that :meth:`Program.run` replaces with
            new arrays, in the order ``run`` takes them.
        outputs : int or list of int
            Handles of the tensors ``run`` returns.

        Returns
        -------
        Program

        Examples
        --------
        >>> import numpy as np
        >>> from mojolearn import ops
        >>> rng = np.random.default_rng(0)
        >>> with ops.Session() as s:
        ...     x = s.tensor(rng.standard_normal((16, 4), dtype=np.float32))
        ...     w = s.tensor(rng.standard_normal((4, 3), dtype=np.float32))
        ...     p = s.softmax(s.linear(x, w, activation="tanh"))
        ...     model = s.capture(inputs=[x], outputs=[p])
        ...     probs = model.run(rng.standard_normal((16, 4), dtype=np.float32))
        """
        ins = self._handles(inputs)
        outs = self._handles(outputs)
        # Mojo walks the session's operation log from the outputs
        info = self._b.ops_capture(self._sid(), ins, outs)
        return Program(self, info)

    def run(self, program, *arrays, dynamic_shapes=False):
        """Run ``program`` on new input arrays: one call that uploads them,
        runs every captured operation on the device and returns the outputs.

        ``arrays`` are in the order of ``capture``'s ``inputs``, each of the
        shape the captured input had (any dtype :meth:`tensor` takes; an
        input made with :meth:`ids` takes an integer array). The
        outputs are the values the same operations give when called one by
        one, to the bit. Returns one float32 array for a program with one
        output, else a tuple of arrays.

        With ``dynamic_shapes=True``, input dimensions may change while
        their ranks stay fixed. The captured operations and parameter
        shapes must accept the new dimensions. Output shapes are computed
        natively. A captured reshape should use ``-1`` for an inferred
        dimension when its element count can change.
        """
        from ._cast_in import as_input, describe, matrix_view
        if not isinstance(program, Program) or program._s is not self:
            raise ValueError("program must be a Program of this session")
        if dynamic_shapes:
            return self._b.ops_program_dynamic(self._sid(), program._id, arrays)
        if len(arrays) != len(program.input_shapes):
            raise ValueError(f"this program takes {len(program.input_shapes)} input arrays, got {len(arrays)}")
        keep, descs = [], []
        for k, x in enumerate(arrays):  # glue: one descriptor per program input
            a = as_input(x)
            want = program.input_shapes[k]
            have = a.shape if a.ndim > 1 else (a.size, 1)
            if tuple(have) != want:
                raise ValueError(f"input {k} must have the shape {want}, got {tuple(a.shape)}")
            if program.input_kinds[k] == 1 and a.dtype.kind not in "iu":
                raise TypeError(f"input {k} holds ids and must be an integer array, got dtype {a.dtype}")
            m = matrix_view(a, "x") if a.ndim > 2 else a
            keep.append(m)
            descs.append(describe(m, "x"))
        outs = [np.empty(sh, dtype=np.float32) for sh in program.output_shapes]  # glue: one destination per output
        addrs = [o.ctypes.data for o in outs]  # glue: one destination address per program output
        self._b.ops_program_run(self._sid(), program._id, descs, addrs)
        del keep
        return outs[0] if len(outs) == 1 else tuple(outs)

    def load_distributed(self, path, devices=(0,)):
        """Restore a complete distributed program onto the given devices."""
        return _DistributedProgram(self._b, self._b.ops_distributed_load(path, devices))

    def load_program(self, path):
        """Load a :class:`Program` saved with :meth:`Program.save`.

        Its parameters become new tensors of this session. The session must
        not record. A program saved on one install loads on any other,
        a GPU install or the CPU-only install, and gives the same outputs.
        """
        if self.record:
            raise ValueError("programs run in an inference session (Session(record=False))")
        info = self._b.ops_load_program(self._sid(), os.fspath(path))  # cpu-route: program file read on the host
        return Program(self, info)

    # ------------------------------------------------------------ checkpoints
    def save_state(self, path, params, optimizer=None, *, seed=0, epoch=0, step=0):
        """Save parameters, optimizer state and training counters to ``path``.

        Together with :meth:`load_state` this resumes a custom training
        loop exactly: the parameters and the optimizer's buffers are saved
        word for word, and ``seed``, ``epoch`` and ``step`` are the counters
        your loop passes to :meth:`permutation` and :meth:`dropout`, so the
        resumed loop draws the same row orders and dropout masks it would
        have drawn without stopping. A file saved on one GPU vendor (or on
        the CPU) resumes to the same values on another.

        Parameters
        ----------
        path : str or path-like
            Destination file.
        params : list of int
            The parameter handles, in the order :meth:`load_state` returns
            them.
        optimizer : optimizer handle or None, default None
            The optimizer from :meth:`optimizer` whose state is saved with
            the parameters.
        seed, epoch, step : int, default 0
            Your loop's random seed and counters, returned by
            :meth:`load_state`.

        Examples
        --------
        >>> import numpy as np
        >>> from mojolearn import ops
        >>> rng = np.random.default_rng(0)
        >>> with ops.Session(record=True) as s:  # doctest: +SKIP
        ...     w = s.tensor(rng.standard_normal((4, 2), dtype=np.float32), requires_grad=True)
        ...     opt = s.optimizer("adam", lr=1e-3)
        ...     s.save_state("state.ckpt", [w], optimizer=opt, seed=0, epoch=3, step=120)
        """
        hs = [_int(h, "handle", 1) for h in params]  # glue: the parameter handles
        if not hs:
            raise ValueError("params must hold at least one tensor handle")
        opt = 0 if optimizer is None else self._kind_of(optimizer)[0]
        counters = _seed_words(seed) + [_int(epoch, "epoch", 0), _int(step, "step", 0)]
        # Mojo reads each parameter's optimizer state (step count,
        # bias-correction products, buffers) and writes the file
        self._b.ops_save_state(self._sid(), os.fspath(path), hs, opt, counters)  # cpu-route: checkpoint file written on the host

    def load_state(self, path, optimizer=None, requires_grad=True):
        """Load a file written by :meth:`save_state`.

        The parameters become new tensors of this session (trainable when
        ``requires_grad`` and the session records). With ``optimizer``, the
        saved optimizer state is written into that optimizer, so its next
        :meth:`step` continues exactly; create it with the same settings as
        the one that was saved.

        Parameters
        ----------
        path : str or path-like
            A file written by :meth:`save_state`.
        optimizer : optimizer handle or None, default None
            The optimizer that continues the training.
        requires_grad : bool, default True
            Make the loaded parameters trainable (recording sessions).

        Returns
        -------
        dict
            ``params`` (the new handles, in the saved order), ``seed``,
            ``epoch`` and ``step``.

        Examples
        --------
        >>> from mojolearn import ops
        >>> with ops.Session(record=True) as s:  # doctest: +SKIP
        ...     opt = s.optimizer("adam", lr=1e-3)
        ...     state = s.load_state("state.ckpt", optimizer=opt)
        ...     (w,) = state["params"]
        """
        path = os.fspath(path)
        grad = 1 if (requires_grad and self.record) else 0
        opt = 0 if optimizer is None else self._kind_of(optimizer)[0]
        # Mojo uploads the parameters and restores the optimizer state
        # through the session's load_optimizer_state
        hs, _, lo, hi, epoch, step = self._b.ops_load_state(self._sid(), path, grad, opt)  # cpu-route: checkpoint file read on the host
        hs = [int(h) for h in hs]  # glue: the new handles
        return dict(params=hs, seed=(int(hi) << 32) | int(lo), epoch=int(epoch), step=int(step))
    def load_weights(self, path, names, *, source=None):
        """Load named tensors from a safetensors checkpoint (or a PyTorch
        state_dict) into tensors of this session.

        Each tensor in ``names`` must have been made with :meth:`tensor`;
        its values are replaced in place, so the handle, its
        ``requires_grad`` and its optimizer state stay. float16, bfloat16,
        float32 and float64 sources are converted to float32 on the device
        (float16 and bfloat16 exactly, float64 rounded to the nearest
        float32).

        Parameters
        ----------
        path : str, path-like or mapping
            The path of a ``.safetensors`` file or of a checkpoint directory
            (``model.safetensors``, or ``model.safetensors.index.json`` and
            its shards), or a mapping of names to tensors such as
            ``module.state_dict()``.
        names : list of tuple
            ``(handle, name)`` pairs: tensor ``name`` of the source goes
            into tensor ``handle``, which has the source's shape (size-1
            axes aside: a ``(n,)`` bias fills a ``(1, n)`` tensor).
            ``(handle, name, "transpose")`` loads a 2-D ``(a, b)`` source
            into an ``(b, a)`` tensor, for example a PyTorch linear weight
            ``(out_features, in_features)`` into the ``(inputs, outputs)``
            weight :meth:`linear` takes.
        source : {'torch', 'safetensors'} or None, default None
            None takes ``'safetensors'`` for a path and ``'torch'`` for a
            mapping.

        Returns
        -------
        list of int
            The handles, in the order given.

        Examples
        --------
        >>> import numpy as np
        >>> from mojolearn import ops
        >>> rng = np.random.default_rng(0)
        >>> state = {"fc.weight": rng.standard_normal((2, 4), dtype=np.float32),
        ...          "fc.bias": np.zeros(2, dtype=np.float32)}
        >>> with ops.Session() as s:
        ...     w = s.tensor(np.zeros((4, 2), dtype=np.float32))
        ...     b = s.tensor(np.zeros((1, 2), dtype=np.float32))
        ...     _ = s.load_weights(state, [(w, "fc.weight", "transpose"), (b, "fc.bias")])
        ...     x = s.tensor(rng.standard_normal((8, 4), dtype=np.float32))
        ...     probs = s.numpy(s.linear(x, w, b, activation="softmax"))
        """
        from ._weight_import import LAYOUT_AS_IS, LAYOUT_TRANSPOSE, WeightSource
        pairs = list(names)
        if not pairs:
            raise ValueError("names must hold at least one (handle, name) pair")
        entries = []
        for pair in pairs:  # glue: the (handle, name[, layout]) arguments
            if not isinstance(pair, (tuple, list)) or len(pair) not in (2, 3):
                raise ValueError("names takes (handle, name) or (handle, name, 'transpose') pairs")
            layout = LAYOUT_AS_IS
            if len(pair) == 3:
                if pair[2] != "transpose":
                    raise ValueError("the third item of a names entry must be 'transpose'")
                layout = LAYOUT_TRANSPOSE
            if not isinstance(pair[1], str):
                raise TypeError("a tensor name must be a string")
            entries.append((_int(pair[0], "handle", 1), pair[1], layout))
        keys = [k for _, k, _ in entries]  # glue: the requested tensor names
        sid = self._sid()
        with WeightSource(path, source, what="load_weights") as src:
            missing = src.missing(keys)
            if missing:
                held = src.names()
                raise ValueError(f"load_weights: the source has no tensor {missing[0]!r}"
                                 + (f" ({len(missing)} missing)" if len(missing) > 1 else "")
                                 + f"; it holds {held[:6]}" + (" ..." if len(held) > 6 else ""))
            recs = []
            for h, key, layout in entries:  # glue: one record per source tensor
                recs += src.record(key, h, 0, layout)
            # Mojo checks every dtype and shape, then writes every tensor
            self._b.ops_load_weights(sid, recs, keys)
        return [h for h, _, _ in entries]  # glue: the handles, in the order given

    def zero_grad(self):
        """Clear the gradient of every tensor made with ``requires_grad=True``.

        The next ``backward(loss, accumulate=True)`` then starts each
        gradient from zero. Use it to sum the gradients of several losses
        before one :meth:`step`; a plain :meth:`backward` already replaces
        the previous gradients.
        """
        self._b.ops_zero_grad(self._sid())

    # ------------------------------------------------------------ optimizers
    def optimizer(self, kind="adam", lr=1e-3, *, momentum=0.0, nesterov=False, beta1=0.9, beta2=0.999,
                  eps=1e-8, weight_decay=0.0, schedule="constant", step_size=1, gamma=0.1):
        """Create an optimizer for :meth:`step`; returns its integer handle.

        Parameters
        ----------
        kind : {'sgd', 'adam', 'adamw'}, default 'adam'
            The update rule, as in PyTorch's ``torch.optim``.
        lr : float, default 1e-3
            Learning rate (> 0).
        momentum : float, default 0.0
            Momentum of ``'sgd'``, in ``[0, 1)``.
        nesterov : bool, default False
            Nesterov momentum for ``'sgd'`` (needs ``momentum > 0``).
        beta1, beta2, eps : float
            Moment decays and denominator term of ``'adam'`` and ``'adamw'``.
        weight_decay : float, default 0.0
            L2 penalty added to the gradient (``'sgd'``, ``'adam'``) or
            decoupled decay (``'adamw'``).
        schedule : {'constant', 'step', 'exponential'}, default 'constant'
            Learning-rate schedule over epochs: ``'step'`` multiplies the
            rate by ``gamma`` every ``step_size`` epochs, ``'exponential'``
            by ``gamma`` every epoch. Advance it with :meth:`scheduler_step`.
        step_size : int, default 1
            Epochs per decay of the ``'step'`` schedule.
        gamma : float, default 0.1
            Decay factor of the ``'step'`` and ``'exponential'`` schedules.

        The options mean what they mean in :meth:`mojolearn.Sequential.compile`,
        and the update is the one :meth:`mojolearn.Sequential.fit` runs:
        the same model, data, starting weights and options give the same
        weights, bit for bit.

        Example
        -------
        >>> with ops.Session(record=True) as s:
        ...     opt = s.optimizer("sgd", lr=0.1, momentum=0.9, nesterov=True)
        """
        if kind not in _OPTIMIZER:
            raise ValueError(f"kind must be one of {sorted(_OPTIMIZER)}")
        if schedule not in _SCHEDULE:
            raise ValueError(f"schedule must be one of {sorted(_SCHEDULE)}")
        ip = [_OPTIMIZER[kind], 1 if nesterov else 0, _SCHEDULE[schedule], _int(step_size, "step_size", 1)]
        fp = [float(lr), float(momentum), float(beta1), float(beta2), float(eps), float(weight_decay),
              float(gamma)]
        opt = int(self._b.ops_optimizer(self._sid(), ip, fp))
        self._opt_kind[opt] = kind
        return opt

    def _kind_of(self, optimizer):
        optimizer = _int(optimizer, "optimizer", 1)
        if optimizer not in self._opt_kind:
            raise ValueError(f"{optimizer} is not an optimizer of this session")
        return optimizer, self._opt_kind[optimizer]

    def step(self, optimizer, params):
        """Update the parameters in place from the last :meth:`backward`.

        ``params`` lists tensors made with ``requires_grad=True`` (or is one
        such handle). Each is updated on the device by ``optimizer``; its
        optimizer state (momentum or moment estimates and the step count)
        stays on the device in the session, separately for every parameter.
        Nothing is read back. Call :meth:`reset` before recording the next
        forward pass.
        """
        optimizer, _ = self._kind_of(optimizer)
        if isinstance(params, (int, np.integer)) and not isinstance(params, bool):
            params = [params]
        hs = [_int(p, "handle", 1) for p in params]  # glue: the step's parameter handles
        self._b.ops_step(self._sid(), optimizer, hs)

    def scheduler_step(self, optimizer):
        """Advance ``optimizer``'s learning-rate schedule by one epoch."""
        optimizer, _ = self._kind_of(optimizer)
        self._b.ops_scheduler_step(self._sid(), optimizer)

    def learning_rate(self, optimizer):
        """``optimizer``'s current learning rate, as a float."""
        optimizer, _ = self._kind_of(optimizer)
        return float(self._b.ops_learning_rate(self._sid(), optimizer))

    def optimizer_state(self, optimizer, param):
        """The optimizer's state for one parameter, as a dict.

        ``'step'`` is the number of updates applied to ``param``. ``'adam'``
        and ``'adamw'`` add ``'beta1_power'`` and ``'beta2_power'`` (the
        bias-correction products) and the tensor handles ``'exp_avg'`` and
        ``'exp_avg_sq'``; ``'sgd'`` with momentum adds
        ``'momentum_buffer'``. The tensors are copies, valid until
        :meth:`reset`: read them with :meth:`numpy` to save a checkpoint,
        and restore it with :meth:`load_optimizer_state`.
        """
        optimizer, kind = self._kind_of(optimizer)
        w = self._b.ops_optimizer_state(self._sid(), optimizer, _int(param, "param", 1))
        state = {"step": int(w[0])}
        if kind in ("adam", "adamw"):
            state["beta1_power"] = float(w[1])
            state["beta2_power"] = float(w[2])
        for name, h in zip(_STATE_NAMES[kind], w[4:7]):  # glue: three named state buffers
            if name is not None and int(h) != 0:
                state[name] = int(h)
        return state

    def load_optimizer_state(self, optimizer, param, state):
        """Restore the state :meth:`optimizer_state` returned.

        ``state`` holds the same keys, with each buffer a tensor handle of
        ``param``'s shape (for example ``s.tensor(saved_array)``). The
        values are copied on the device; the next :meth:`step` continues
        exactly where the saved run stopped.
        """
        optimizer, kind = self._kind_of(optimizer)
        if "step" not in state:
            raise KeyError("state needs 'step'")
        names = _STATE_NAMES[kind]
        hs = [_int(state[n], n, 1) if n is not None and n in state else 0 for n in names]  # glue: three named state buffers
        fp = [1.0, 1.0, 1.0]
        if kind in ("adam", "adamw"):
            if "beta1_power" not in state or "beta2_power" not in state:
                raise KeyError("an adam state needs 'beta1_power' and 'beta2_power'")
            fp = [float(state["beta1_power"]), float(state["beta2_power"]), 1.0]
        ip = [_int(state["step"], "step", 0)] + hs
        self._b.ops_load_optimizer_state(self._sid(), optimizer, _int(param, "param", 1), ip, fp)


class Program:
    """A model captured from an inference :class:`Session`, run in one call.

    Made by :meth:`Session.capture` or :meth:`Session.load_program`.
    :meth:`run` uploads new input arrays, runs every captured operation on
    the device and returns the outputs: the same values, to the bit, as
    calling the operations one by one, with one call instead of one per
    operation.

    Attributes
    ----------
    input_shapes : tuple of tuple of int
        The shape each input array of :meth:`run` must have.
    input_kinds : tuple of int
        0 for an input of values, 1 for an input of integer ids (a tensor
        made with :meth:`Session.ids`).
    output_shapes : tuple of tuple of int
        The shapes of the arrays :meth:`run` returns.
    params : tuple of int
        Handles of the session tensors the program reads as parameters.

    Examples
    --------
    >>> import numpy as np
    >>> from mojolearn import ops
    >>> rng = np.random.default_rng(0)
    >>> W1 = rng.standard_normal((4, 8), dtype=np.float32)
    >>> W2 = rng.standard_normal((8, 2), dtype=np.float32)
    >>> with ops.Session() as s:
    ...     x = s.tensor(np.zeros((32, 4), np.float32))
    ...     h = s.linear(x, s.tensor(W1), activation="relu")
    ...     y = s.softmax(s.linear(h, s.tensor(W2)))
    ...     model = s.capture(inputs=[x], outputs=[y])
    ...     s.reset()
    ...     proba = model.run(rng.standard_normal((32, 4), dtype=np.float32))
    """

    def __init__(self, session, info):
        self._s = session
        self._id = int(info[0])
        self.input_shapes = tuple(tuple(int(d) for d in sh) for sh in info[1])  # glue: at most eight dimensions per input
        self.output_shapes = tuple(tuple(int(d) for d in sh) for sh in info[2])  # glue: at most eight dimensions per output
        self.params = tuple(int(h) for h in info[3])  # glue: the parameter handles
        # a binding built before the input kinds returns four entries: every input holds values
        kinds = info[4] if len(info) > 4 else [0] * len(self.input_shapes)
        self.input_kinds = tuple(int(k) for k in kinds)  # glue: one kind per program input

    def run(self, *arrays, dynamic_shapes=False):
        """Run the program on new input arrays (see :meth:`Session.run`)."""
        return self._s.run(self, *arrays, dynamic_shapes=dynamic_shapes)

    def fit(self, *arrays, targets=None, optimizer, parameters=None, loss="mse",
            epochs=10, batch_size=32, shuffle=True, seed=0, huber_delta=1.0):
        """Train this captured graph in one native call.

        Parameters
        ----------
        arrays : ndarray
            Input arrays in capture order, with equal sample counts on
            their first axis. Batching preserves the remaining dimensions.
        targets : ndarray or None
            Targets for the chosen loss. Cross-entropy takes numeric class
            codes. With ``loss="output"``, omit targets and capture a scalar
            loss as the program's one output.
        optimizer : int
            Handle returned by the owning session's ``optimizer`` method.
            Its state and schedule continue across calls.
        parameters : sequence of int or None
            Captured parameter handles to update. None trains every
            captured parameter; specify a subset to keep constants frozen.
        loss : {'mse', 'mae', 'huber', 'bce_with_logits', 'cross_entropy', 'output'}
            Loss applied to the program's one output.
        epochs, batch_size : int
            Epoch count and maximum rows per batch.
        shuffle : bool
            Shuffle rows with the native seeded permutation each epoch.
        seed : int
            Seed of the row order; reuse it when continuing training.
        huber_delta : float
            Transition point for the Huber loss.

        Returns
        -------
        Program
            This model, with ``loss_curve_`` holding this call's epoch
            losses. Parameters and optimizer state stay in the session.

        Examples
        --------
        >>> opt = session.optimizer("adam", lr=0.01)
        >>> model.fit(X, targets=y, optimizer=opt, epochs=20)
        """
        kinds = {"output": 0, "mse": 1, "mae": 2, "huber": 3,
                 "bce_with_logits": 4, "cross_entropy": 5}
        if loss not in kinds:
            raise ValueError(f"unknown loss {loss!r}")
        if (loss == "output") != (targets is None):
            raise ValueError("omit targets only when loss='output'")
        lo, hi = _seed_words(seed)
        params = self.params if parameters is None else self._s._handles(parameters)
        ip = [_int(optimizer, "optimizer", 1), kinds[loss], _int(epochs, "epochs", 1),
              _int(batch_size, "batch_size", 1), int(bool(shuffle)), lo, hi]
        self.loss_curve_ = self._s._b.ops_program_fit(
            self._s._sid(), self._id, arrays, targets, params, ip, [float(huber_delta)])
        self.n_iter_ = int(epochs)
        return self

    def distribute(self, devices=(0,), *, logical_shards=2, optimizer,
                   parameters=(), seed=0):
        """Copy this program into native workers with sharded optimizer state.

        Keep ``logical_shards`` unchanged when comparing physical topologies.
        ``optimizer`` is a session optimizer handle; its state is copied too.
        ``parameters`` lists trainable handles, or is empty to train every
        floating-point parameter. ``seed`` controls native row permutations.
        """
        return _DistributedProgram(self._s._b, self._s._b.ops_distributed_open(
            self._s._sid(), self._id, devices, logical_shards, optimizer, parameters, seed))

    def save(self, path):
        """Write the program and its parameters' current values to ``path``,
        for :meth:`Session.load_program` on any install.

        Beside it, ``<path>.verify.json`` records this install's report: the
        program run once on seeded inputs, with the digest of every input and
        output. On another install, ``python -m mojolearn verify-model
        <path>`` runs the saved program on the same inputs and checks that
        every output has the same bits.
        """
        self._s._b.ops_save_program(self._s._sid(), self._id, os.fspath(path))  # cpu-route: program file written on the host
        from ._verify_model import record_program
        record_program(self, path)

    def free(self):
        """Release the program (its parameters stay session tensors)."""
        if self._id is not None and self._s._id is not None:
            self._s._b.ops_program_free(self._s._sid(), self._id)
        self._id = None
