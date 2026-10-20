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

__all__ = ["Session"]

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
}
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

        ``indices`` is an array of integers (any integer dtype), flattened
        in row-major order; the result replaces ``axis`` by one slice per
        index. An index outside ``[0, x.shape[axis])`` raises an error. In a
        recording session the gradient of ``x`` sums the gradients of every
        slice taken from each position, in ascending index position.
        """
        from ._cast_in import as_input, describe
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
    def conv2d(self, x, weight, bias=None, stride=1, padding=0):
        """A 2-D convolution (cross-correlation) of ``x``.

        ``x`` is ``(N, C, H, W)``, ``weight`` ``(out_channels, C, kh, kw)``
        and ``bias`` an optional tensor of ``out_channels`` values. The
        input is padded with ``padding`` zeros on every side and the kernel
        moves ``stride`` positions at a time; the result is
        ``(N, out_channels, H_out, W_out)``.
        """
        handles = (x, weight) if bias is None else (x, weight, bias)
        return self._op("conv2d", handles, [_int(stride, "stride", 1), _int(padding, "padding", 0)])

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
        an array of integers (any integer dtype) in ``[0, num_embeddings)``,
        flattened in row-major order;
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
    def backward(self, loss, accumulate=False):
        """Differentiate the ``(1, 1)`` tensor ``loss`` with respect to every
        input made with ``requires_grad=True`` (recording sessions only).
        ``accumulate=True`` adds to the gradients of the previous backward."""
        self._b.ops_backward(self._sid(), _int(loss, "loss", 1), 1 if accumulate else 0)

    def grad(self, h):
        """The gradient of tensor ``h`` from the last :meth:`backward`, as a
        float32 array of its shape."""
        out = np.empty(self.shape(h), dtype=np.float32)
        self._b.ops_grad(self._sid(), h, out.ctypes.data)
        return out

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
