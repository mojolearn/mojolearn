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

In `identical` mode, Mojolearn returns the same bits on Apple, NVIDIA and AMD GPUs.
Every operation here is the one Mojolearn's own trainers run, so a
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
from types import SimpleNamespace as _SimpleNamespace

from ._optional_numpy import require_numpy as _require_numpy
_require_numpy('mojolearn.ops')

from . import _backend
from ._distributed_program import DistributedProgram as _DistributedProgram
from ._partitioned_program import PartitionedProgram as _PartitionedProgram

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
    # nn-blocks-scatter begin
    "scatter_add": 113, "segment_sum": 114, "segment_mean": 115, "segment_max": 116,
    # nn-blocks-scatter end
    # nn-blocks-qkv begin
    "attention": 122,
    # nn-blocks-qkv end
    # session-ops-conv begin
    "conv1d": 130, "conv2d_dilated": 131, "conv3d": 132, "conv_transpose1d": 133, "conv_transpose2d": 134,
    # session-ops-conv end
}
_GELU = {"none": 0, "tanh": 1}
# the state buffers autograd/ops_api.mojo optimizer_state returns, by kind
_STATE_NAMES = {"sgd": ("momentum_buffer", None, None), "adam": ("exp_avg", "exp_avg_sq", None),
                "adamw": ("exp_avg", "exp_avg_sq", None)}
# autograd/ops_api.mojo OPS_PACK_*: the packed argument block of one call
_PACK_H, _PACK_I, _PACK_F = 3, 7, 15
_PACK_HANDLES, _PACK_INTS, _PACK_FLOATS = 4, 8, 4


# The Python `_int`, `_seed_words`, `_axes`, `Session._sid` and
# `Session._handles` refusals had no caller left (lane py-tail, 2026-11-15):
# every Session operation is one `_SESSION_CALL` and the ops door refuses its
# arguments (providers/python/autograd_ops_frontend_pyapi_door.mojo `_int`,
# `_axes`, `_seed_words`).


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

    Train a score at every step of sequences of different lengths: an
    embedding of each sequence's category beside its per-step features, a
    GRU over every step, a linear head on every step's state and a binary
    loss at every step, the padded steps masked out and the positive steps
    weighted 30 times (``x`` is time first):

    >>> T, B, F, E, H = 12, 4, 3, 2, 8
    >>> X = rng.standard_normal((T, B, F), dtype=np.float32)
    >>> lengths = np.array([12, 9, 5, 7], dtype=np.float32)
    >>> category = np.array([0, 2, 1, 2])
    >>> Y = (rng.random((T, B, 1)) < 0.2).astype(np.float32)
    >>> valid = (np.arange(T)[:, None] < lengths[None, :]).astype(np.float32)[:, :, None]
    >>> W = (valid * np.where(Y > 0, 30.0, 1.0)).astype(np.float32)
    >>> def init(*shape):
    ...     return rng.uniform(-0.3, 0.3, shape).astype(np.float32)
    >>> with ops.Session(record=True) as s:
    ...     table = s.tensor(init(3, E), requires_grad=True)
    ...     w_ih = s.tensor(init(3 * H, F + E), requires_grad=True)
    ...     w_hh = s.tensor(init(3 * H, H), requires_grad=True)
    ...     b_ih = s.tensor(init(1, 3 * H), requires_grad=True)
    ...     b_hh = s.tensor(init(1, 3 * H), requires_grad=True)
    ...     w_out = s.tensor(init(H, 1), requires_grad=True)
    ...     b_out = s.tensor(init(1, 1), requires_grad=True)
    ...     params = [table, w_ih, w_hh, b_ih, b_hh, w_out, b_out]
    ...     x, y, w, n = s.tensor(X), s.tensor(Y), s.tensor(W), s.tensor(lengths)
    ...     opt = s.optimizer("adam", lr=0.01)
    ...     for epoch in range(5):
    ...         e = s.expand(s.reshape(s.embedding(table, category), (1, B, E)), (T, B, E))
    ...         h = s.gru(s.concat([x, e], axis=2), w_ih, w_hh, b_ih, b_hh, lengths=n)
    ...         logits = s.linear(h, w_out, b_out)
    ...         s.backward(s.bce_with_logits(logits, y, weight=w))
    ...         s.step(opt, params)
    ...         s.reset()
    ...     head = s.numpy(w_out)
    """

    def __init__(self, record=False, numeric_mode=None, device=None):
        return _SESSION_INITIALIZE(self, record, numeric_mode, _SESSION_CONNECTIONS, device, Program)

    # ------------------------------------------------------------ lifetime
    def close(self):
        """Free every tensor of the session and its device memory."""
        return _SESSION_CALL(self, "close", ())

    def __enter__(self):
        return _SESSION_CALL(self, "enter", ())

    def __exit__(self, *exc):
        return _SESSION_CALL(self, "exit", ())

    def reset(self):
        """Drop every computed tensor (and the recorded graph); tensors made
        with :meth:`tensor` stay valid. Call it between independent passes to
        reuse their memory."""
        return _SESSION_CALL(self, "reset", ())

    def free(self, h):
        """Release tensor ``h``; its memory is reused after :meth:`reset`."""
        return _SESSION_CALL(self, "free", (h,))

    def gradient_report(self, handles, *, values=True):
        """Explain gradient availability and report nonfinite or zero values for tensors."""
        return _SESSION_CALL(self, "gradient_report", (handles, values))

    def execution_profile_begin(self):
        """Begin executor allocation, transfer, synchronization and operation accounting."""
        return _SESSION_CALL(self, "execution_profile_begin", ())

    def execution_profile_end(self):
        """Drain the executor and return counters for the active profiling interval."""
        return _SESSION_CALL(self, "execution_profile_end", ())

    def profile_reset(self):
        """Clear the process-wide launch profile (recorded when the
        MOJOLEARN_AG_PROFILE environment variable is set)."""
        return _SESSION_CALL(self, "profile_reset", ())

    def profile_report(self):
        """The process-wide launch profile as text: launches per operation."""
        return _SESSION_CALL(self, "profile_report", ())

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
        return _SESSION_CALL(self, "tensor", (x, requires_grad))

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
        return _SESSION_CALL(self, "ids", (x,))

    def state(self, x, ids=False):
        """Copy an array to the device as carried state; returns its handle.

        A state tensor is carried from one call to the next and
        :meth:`Session.capture` records it as state of the program, so
        :meth:`Program.decode` and :meth:`Program.while_loop` advance it
        without copies. An operation that keeps state between calls
        writes its state tensors in place. ``x`` is accepted as in
        :meth:`tensor`, or as in :meth:`ids` with ``ids=True`` (integer
        state such as a position). A state tensor takes no gradient.

        Example
        -------
        >>> import numpy as np
        >>> from mojolearn import ops
        >>> with ops.Session() as s:
        ...     position = s.state(np.zeros((1, 1), np.int32), ids=True)
        """
        return _SESSION_CALL(self, "state", (x, ids))

    def shape(self, h):
        """The shape of tensor ``h`` as a tuple: ``(rows, cols)`` for a
        matrix, every dimension for a tensor of another rank."""
        return _SESSION_CALL(self, "shape", (h,))

    def numpy(self, h):
        """Tensor ``h``'s values as a new float32 array of its shape."""
        return _SESSION_CALL(self, "numpy", (h,))

    def scalar(self, h):
        """The value of a ``(1, 1)`` tensor as a Python float, for branching
        in your code on a computed value."""
        return _SESSION_CALL(self, "scalar", (h,))

    def full(self, shape, value):
        """A new tensor of ``shape`` (up to 8 dimensions) with every
        element ``value``."""
        return _SESSION_CALL(self, "full", (shape, value))

    def copy(self, x):
        """A copy of ``x`` that carries no gradient."""
        return _SESSION_CALL(self, "copy", (x,))

    def detach(self, x):
        """``x`` as a constant: later operations pass no gradient back to it."""
        return _SESSION_CALL(self, "detach", (x,))

    # ------------------------------------------------------------ the call
    # ------------------------------------------------------------ elementwise and products
    def matmul(self, a, b):
        """The matrix product of the last two axes, ``a @ b``.

        Leading axes are batch axes and broadcast as in NumPy: a
        ``(batch, heads, m, k)`` tensor times a ``(batch, heads, k, n)`` or
        a ``(k, n)`` tensor gives ``(batch, heads, m, n)``.
        """
        return _SESSION_CALL(self, "matmul", (a, b))

    def add(self, a, b):
        """``a + b`` with NumPy broadcasting: shapes are aligned on the
        right and an axis of length 1 stretches. For matrices, a ``(1, 1)``,
        ``(1, cols)`` or ``(rows, 1)`` operand broadcasts."""
        return _SESSION_CALL(self, "add", (a, b))

    def sub(self, a, b):
        """``a - b`` with the broadcasting of :meth:`add`."""
        return _SESSION_CALL(self, "sub", (a, b))

    def mul(self, a, b):
        """``a * b`` elementwise with the broadcasting of :meth:`add`."""
        return _SESSION_CALL(self, "mul", (a, b))

    def div(self, a, b):
        """``a / b`` elementwise with the broadcasting of :meth:`add`."""
        return _SESSION_CALL(self, "div", (a, b))

    def relu(self, x):
        """``max(x, 0)``."""
        return _SESSION_CALL(self, "relu", (x,))

    def sigmoid(self, x):
        """``1 / (1 + exp(-x))``."""
        return _SESSION_CALL(self, "sigmoid", (x,))

    def tanh(self, x):
        """The hyperbolic tangent."""
        return _SESSION_CALL(self, "tanh", (x,))

    def exp(self, x):
        """The natural exponential."""
        return _SESSION_CALL(self, "exp", (x,))

    def log(self, x):
        """The natural logarithm."""
        return _SESSION_CALL(self, "log", (x,))

    def sqrt(self, x):
        """The square root."""
        return _SESSION_CALL(self, "sqrt", (x,))

    def scale(self, x, factor):
        """``factor * x``."""
        return _SESSION_CALL(self, "scale", (x, factor))

    # ------------------------------------------------------------ shape and reductions
    def transpose(self, x, a=-2, b=-1):
        """``x`` with axes ``a`` and ``b`` swapped; the matrix transpose by
        default."""
        return _SESSION_CALL(self, "transpose", (x, a, b))

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
        return _SESSION_CALL(self, "permute", (x, axes))

    def reshape(self, x, shape):
        """``x``'s elements, row-major, in a new ``shape`` with the same
        number of elements; one dimension may be -1 and is inferred."""
        return _SESSION_CALL(self, "reshape", (x, shape))

    def flatten(self, x, start=1, end=-1):
        """``x`` with axes ``start`` to ``end`` (inclusive) merged into
        one; by default every axis after the first."""
        return _SESSION_CALL(self, "flatten", (x, start, end))

    def squeeze(self, x, axis=None):
        """``x`` without its length-1 axes (all of them, or those listed in
        ``axis``)."""
        return _SESSION_CALL(self, "squeeze", (x, axis))

    def unsqueeze(self, x, axis):
        """``x`` with a new length-1 axis at position ``axis``."""
        return _SESSION_CALL(self, "unsqueeze", (x, axis))

    def sum(self, x, axis=None, keepdims=None):
        """The sum over ``axis``.

        ``axis=None`` sums every element into a ``(1, 1)`` tensor; an
        integer or a tuple of integers sums over those axes (negative axes
        count from the end). ``keepdims=True`` keeps each summed axis with
        length 1 and ``False`` removes it; the default keeps it for a
        matrix and removes it otherwise, so ``sum(x, axis=0)`` of an
        ``(n, k)`` matrix is ``(1, k)``.
        """
        return _SESSION_CALL(self, "sum", (x, axis, keepdims))

    def mean(self, x, axis=None, keepdims=None):
        """The mean over ``axis``, with the axes and shapes of :meth:`sum`."""
        return _SESSION_CALL(self, "mean", (x, axis, keepdims))

    def max(self, x, axis=None, keepdims=None):
        """The largest value over ``axis``, with the axes and shapes of
        :meth:`sum`. A NaN is larger than any number; among equal values the
        first position wins, and the gradient flows to that position only."""
        return _SESSION_CALL(self, "max", (x, axis, keepdims))

    def softmax(self, x, axis=-1):
        """The softmax along ``axis``; by default the last axis (each row
        of a matrix)."""
        return _SESSION_CALL(self, "softmax", (x, axis))

    def log_softmax(self, x, axis=-1):
        """The log-softmax along ``axis``; by default the last axis."""
        return _SESSION_CALL(self, "log_softmax", (x, axis))

    # ------------------------------------------------------------ joining, splitting, picking
    def concat(self, xs, axis=0):
        """The tensors ``xs`` joined along ``axis``; every other axis must
        match."""
        return _SESSION_CALL(self, "concat", (xs, axis))

    def split(self, x, sizes_or_sections, axis=0):
        """``x`` cut along ``axis`` into a list of tensors.

        ``sizes_or_sections`` is the number of equal pieces, or a list of
        piece lengths that sum to the axis length. Each piece is a view of
        ``x`` (no copy); in a recording session the gradients of the pieces
        flow back into their places in ``x``'s gradient.
        """
        return _SESSION_CALL(self, "split", (x, sizes_or_sections, axis))

    def take(self, x, indices, axis=0):
        """The slices of ``x`` at ``indices`` along ``axis``.

        ``indices`` is an array of integers (any integer dtype) or a
        tensor made with :meth:`ids`, flattened in row-major order; the
        result replaces ``axis`` by one slice per index. An index outside
        ``[0, x.shape[axis])`` raises an error. In a recording session the
        gradient of ``x`` sums the gradients of every slice taken from each
        position, in ascending index position.
        """
        return _SESSION_CALL(self, "take", (x, indices, axis))

    # ------------------------------------------------------------ convolution and pooling
    def conv2d(self, x, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
        """A 2-D convolution (cross-correlation) of ``x``, as ``torch.nn.functional.conv2d``.

        ``x`` is ``(N, C, H, W)``, ``weight`` ``(out_channels, C // groups,
        kh, kw)`` and ``bias`` an optional tensor of ``out_channels`` values.
        ``stride``, ``padding`` and ``dilation`` are an integer or a pair
        ``(height, width)``: the input is padded with ``padding`` zeros at
        both ends of each axis, the kernel moves ``stride`` positions at a
        time and its taps are ``dilation`` positions apart. ``padding`` may
        also be ``'valid'`` (no padding) or ``'same'`` (the output keeps the
        input's size; it needs stride 1 and ``dilation * (k - 1)`` even on
        each axis). The result is ``(N, out_channels, H_out, W_out)`` with
        ``H_out = (H + 2 * padding - dilation * (kh - 1) - 1) // stride + 1``
        and ``W_out`` likewise. With ``groups`` greater than 1 the channels
        split into ``groups`` equal blocks and each block of outputs sees
        only its block of inputs; ``groups=C`` is a depthwise convolution.

        Example::

            y = s.conv2d(x, w, stride=1, padding=1, groups=4)
            z = s.conv2d(x, w, padding=(2, 1), dilation=(2, 1))
        """
        return _SESSION_CALL(self, "conv2d", (x, weight, bias, stride, padding, dilation, groups))

    def max_pool2d(self, x, kernel_size, stride=None):
        """The largest value of each ``kernel_size`` window of an
        ``(N, C, H, W)`` tensor, windows ``stride`` apart (by default
        ``kernel_size``)."""
        return _SESSION_CALL(self, "max_pool2d", (x, kernel_size, stride))

    def avg_pool2d(self, x, kernel_size, stride=None):
        """The mean of each ``kernel_size`` window of an ``(N, C, H, W)``
        tensor, windows ``stride`` apart (by default ``kernel_size``)."""
        return _SESSION_CALL(self, "avg_pool2d", (x, kernel_size, stride))

    # ------------------------------------------------------------ more elementwise math
    def erf(self, x):
        """The error function."""
        return _SESSION_CALL(self, "erf", (x,))

    def gelu(self, x, approximate="none"):
        """The Gaussian error linear unit ``x * Phi(x)``, with ``Phi`` the
        standard normal distribution function; ``approximate='tanh'`` uses
        the tanh approximation
        ``0.5 * x * (1 + tanh(sqrt(2 / pi) * (x + 0.044715 * x**3)))``."""
        return _SESSION_CALL(self, "gelu", (x, approximate))

    def softplus(self, x):
        """``log(1 + exp(x))``."""
        return _SESSION_CALL(self, "softplus", (x,))

    def silu(self, x):
        """``x * sigmoid(x)`` (also called swish)."""
        return _SESSION_CALL(self, "silu", (x,))

    def leaky_relu(self, x, alpha=0.01):
        """``x`` where ``x >= 0``, else ``alpha * x``."""
        return _SESSION_CALL(self, "leaky_relu", (x, alpha))

    def elu(self, x, alpha=1.0):
        """``x`` where ``x >= 0``, else ``alpha * (exp(x) - 1)``."""
        return _SESSION_CALL(self, "elu", (x, alpha))

    def hard_sigmoid(self, x, alpha=0.2, beta=0.5):
        """``alpha * x + beta`` clipped to ``[0, 1]``."""
        return _SESSION_CALL(self, "hard_sigmoid", (x, alpha, beta))

    def hard_swish(self, x):
        """``x * hard_sigmoid(x, alpha=1/6, beta=0.5)``."""
        return _SESSION_CALL(self, "hard_swish", (x,))

    def clip(self, x, min=None, max=None):
        """``x`` limited to ``[min, max]``; a bound left as None does not
        limit. A NaN stays NaN."""
        return _SESSION_CALL(self, "clip", (x, min, max))

    def abs(self, x):
        """The absolute value."""
        return _SESSION_CALL(self, "abs", (x,))

    def floor(self, x):
        """The largest integer not above ``x``. No gradient flows through it."""
        return _SESSION_CALL(self, "floor", (x,))

    def ceil(self, x):
        """The smallest integer not below ``x``. No gradient flows through it."""
        return _SESSION_CALL(self, "ceil", (x,))

    def round(self, x):
        """``x`` rounded to the nearest integer, a half to the even one (as
        ``numpy.round``). No gradient flows through it."""
        return _SESSION_CALL(self, "round", (x,))

    def sin(self, x):
        """The sine."""
        return _SESSION_CALL(self, "sin", (x,))

    def cos(self, x):
        """The cosine."""
        return _SESSION_CALL(self, "cos", (x,))

    def sign(self, x):
        """-1, 0 or 1 by the sign of ``x``. No gradient flows through it."""
        return _SESSION_CALL(self, "sign", (x,))

    def reciprocal(self, x):
        """``1 / x``."""
        return _SESSION_CALL(self, "reciprocal", (x,))

    def pow(self, x, exponent):
        """``x ** exponent`` elementwise, ``exponent`` a tensor that
        broadcasts with ``x`` as in :meth:`add`. A negative ``x`` takes an
        integer exponent. For a number exponent use :meth:`pow_scalar`."""
        return _SESSION_CALL(self, "pow", (x, exponent))

    def pow_scalar(self, x, exponent):
        """``x ** exponent`` for a number ``exponent``. A negative ``x``
        takes an integer exponent."""
        # a handle is an int too, so the number form is its own method
        return _SESSION_CALL(self, "pow_scalar", (x, exponent))

    def minimum(self, a, b):
        """The smaller of ``a`` and ``b`` elementwise, with the broadcasting
        of :meth:`add`. A NaN in either gives NaN; ``-0.0`` is below ``0.0``."""
        return _SESSION_CALL(self, "minimum", (a, b))

    def maximum(self, a, b):
        """The larger of ``a`` and ``b`` elementwise, with the broadcasting
        of :meth:`add`. A NaN in either gives NaN; ``0.0`` is above ``-0.0``."""
        return _SESSION_CALL(self, "maximum", (a, b))

    def where(self, condition, a, b):
        """``a`` where ``condition`` is nonzero, else ``b``; the three
        broadcast together as in NumPy. The gradient flows to ``a`` and
        ``b``."""
        return _SESSION_CALL(self, "where", (condition, a, b))

    def equal(self, a, b):
        """1.0 where ``a == b``, else 0.0, with the broadcasting of :meth:`add`."""
        return _SESSION_CALL(self, "equal", (a, b))

    def not_equal(self, a, b):
        """1.0 where ``a != b``, else 0.0."""
        return _SESSION_CALL(self, "not_equal", (a, b))

    def greater(self, a, b):
        """1.0 where ``a > b``, else 0.0."""
        return _SESSION_CALL(self, "greater", (a, b))

    def greater_equal(self, a, b):
        """1.0 where ``a >= b``, else 0.0."""
        return _SESSION_CALL(self, "greater_equal", (a, b))

    def less(self, a, b):
        """1.0 where ``a < b``, else 0.0."""
        return _SESSION_CALL(self, "less", (a, b))

    def less_equal(self, a, b):
        """1.0 where ``a <= b``, else 0.0."""
        return _SESSION_CALL(self, "less_equal", (a, b))

    def logical_not(self, x):
        """1.0 where ``x`` is zero, else 0.0."""
        return _SESSION_CALL(self, "logical_not", (x,))

    def logical_and(self, a, b):
        """1.0 where both ``a`` and ``b`` are nonzero, else 0.0."""
        return _SESSION_CALL(self, "logical_and", (a, b))

    def logical_or(self, a, b):
        """1.0 where ``a`` or ``b`` is nonzero, else 0.0."""
        return _SESSION_CALL(self, "logical_or", (a, b))

    def expand(self, x, shape):
        """``x`` broadcast to ``shape`` as NumPy broadcasts it (the result's
        shape is the broadcast of the two). The gradient sums back over the
        broadcast axes."""
        return _SESSION_CALL(self, "expand", (x, shape))

    def argmax(self, x, axis=-1, keepdims=False, int32=False):
        """The position of the largest value along ``axis``. A NaN is larger
        than any number and among equal values the first position wins.

        The positions are float values by default, so later operations can
        use them; ``int32=True`` stores them as int32 words instead, to read
        with ``s.numpy(h).view(np.int32)``. ``keepdims=True`` keeps the axis
        with length 1. No gradient flows through it.
        """
        return _SESSION_CALL(self, "argmax", (x, axis, keepdims, int32))

    def pad(self, x, pads, value=0.0):
        """``x`` padded with ``value``.

        ``pads`` lists two counts per axis, as ONNX ``Pad`` does: every
        axis's count before its first element, then every axis's count after
        its last one. A negative count removes elements instead.
        """
        return _SESSION_CALL(self, "pad", (x, pads, value))

    def any(self, x, axis=None, keepdims=None):
        """1.0 where some element over ``axis`` is nonzero, else 0.0, with
        the axes and shapes of :meth:`sum`. A NaN counts as nonzero. No
        gradient flows through it."""
        return _SESSION_CALL(self, "any", (x, axis, keepdims))

    def all(self, x, axis=None, keepdims=None):
        """1.0 where every element over ``axis`` is nonzero, else 0.0, with
        the axes and shapes of :meth:`sum`. A NaN counts as nonzero. No
        gradient flows through it."""
        return _SESSION_CALL(self, "all", (x, axis, keepdims))

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
        return _SESSION_CALL(self, "cumsum", (x, axis, exclusive, reverse))

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
        return _SESSION_CALL(self, "index", (x, indices, axes))

    # ------------------------------------------------------------ layers
    def linear(self, x, weight, bias=None, activation=None):
        """A dense layer ``activation(x @ weight + bias)``.

        The last axis of ``x`` is the feature axis: an ``(..., inputs)``
        tensor gives ``(..., outputs)``.
        ``weight`` is ``(inputs, outputs)``, ``bias`` an optional
        ``(1, outputs)`` tensor, ``activation`` one of None, ``'relu'``,
        ``'sigmoid'``, ``'tanh'`` or ``'softmax'``.
        """
        return _SESSION_CALL(self, "linear", (x, weight, bias, activation))

    def layer_norm(self, x, weight, bias, eps=1e-5):
        """Normalize along the last axis to zero mean and unit variance,
        then scale by ``weight`` and shift by ``bias`` (both ``(1, features)``,
        the last axis's length)."""
        return _SESSION_CALL(self, "layer_norm", (x, weight, bias, eps))

    def dropout(self, x, p, seed=0, counter=0):
        """Zero each element with probability ``p`` and scale the rest by
        ``1 / (1 - p)``. The mask depends only on ``seed``, ``counter`` and
        the element position: use a new ``counter`` for every step."""
        return _SESSION_CALL(self, "dropout", (x, p, seed, counter))

    def _transformer_softmax(self, x):
        return _SESSION_CALL(self, "transformer_softmax", (x,))

    def _transformer_linear(self, x, weight):
        return _SESSION_CALL(self, "transformer_linear", (x, weight))

    def rms_norm(self, x, weight, eps=1e-6):
        """Normalize each last-axis row by its root mean square, then scale by weight."""
        return _SESSION_CALL(self, "rms_norm", (x, weight, eps))

    def rotary(self, x, head_dim, *, offset=0, theta=10000.0):
        """Apply full-width rotary embeddings to sequence-by-concatenated-head features."""
        return _SESSION_CALL(self, "rotary", (x, head_dim, offset, theta))

    # ------------------------------------------------------------ blocks
    def mamba_block(self, weights, *, version=1, batch_size=1, dt_limit=(0.0, float("inf")),
                    requires_grad=False):
        """Put one Mamba block on the device; returns the block.

        The block is the one :class:`mojolearn.Mamba1Block`,
        :class:`mojolearn.Mamba2Block` and :class:`mojolearn.Mamba3Block`
        run, so :meth:`mamba_step` returns the same bits as those classes
        for the same weights and input.

        Parameters
        ----------
        weights : list or tuple of arrays
            The block's weights in the order the block class lists them.
            For ``version=1``: ``norm.weight``, ``in_proj.weight``,
            ``conv1d.weight``, ``conv1d.bias``, ``x_proj.weight``,
            ``dt_proj.weight``, ``dt_proj.bias``, ``A_log``, ``D``,
            ``out_proj.weight``. For ``version=2``: ``block_norm.weight``,
            ``in_proj.weight``, ``conv1d.weight``, ``conv1d.bias``,
            ``dt_bias``, ``A_log``, ``D``, ``norm.weight``,
            ``out_proj.weight``. For ``version=3``: the nine arrays of
            :class:`mojolearn.Mamba3Block`, in its order. Shapes are those
            the block class documents; ``d_model`` is read from the first
            array.
        version : {1, 2, 3}, default 1
            Mamba-1, Mamba-2 or Mamba-3.
        batch_size : int, default 1
            Number of sequences the block's state holds.
        dt_limit : tuple of float, default (0.0, inf)
            Clamp ``(lo, hi)`` applied to dt. Used by ``version=2``.
        requires_grad : bool, default False
            Make the weights trainable (recording sessions only).

        Returns
        -------
        dict
            ``'weights'`` is the handle of one tensor holding every weight,
            ``'state'`` the handle of the block's state, which starts at
            zero. Pass the dict to :meth:`mamba_step`. With
            ``requires_grad=True`` pass ``block['weights']`` to :meth:`step`
            to train the block and to :meth:`grad` to read its gradient.

        Example
        -------
        >>> with ops.Session() as s:
        ...     block = s.mamba_block(weights, version=1, batch_size=2)
        ...     y = s.mamba_step(s.tensor(x), block)   # x: (2, tokens, d_model)
        """
        return _SESSION_CALL(self, "mamba_block", (weights, version, batch_size, dt_limit, requires_grad))

    def mamba_step(self, x, block):
        """Run a Mamba block over the next tokens; returns the output.

        ``x`` holds ``batch_size * tokens * d_model`` values, usually as a
        ``(batch_size, tokens, d_model)`` tensor; the result has ``x``'s
        shape. ``block`` comes from :meth:`mamba_block`. One token per call
        is a decode step and several tokens a prefill; the block's state is
        updated in place either way, so the next call continues the
        sequence.

        In a recording session each call starts from a zero state and is
        differentiable: after :meth:`backward`, :meth:`grad` of
        ``block['weights']`` and of ``x`` return the gradients.
        """
        return _SESSION_CALL(self, "mamba_step", (x, block))

    def attention_block(self, weights, *, n_heads, max_positions, batch_size=1, n_kv_heads=None, window=0,
                        requires_grad=False, rope_theta=10000.0, rope_scaling=None, rope_dim=None,
                        qkv_bias=False, o_bias=False, norm="rmsnorm", norm_eps=1e-6, norm_bias=False,
                        mlp="swiglu", mlp_bias=False, qk_norm=False, attn_softcap=None):
        """Put one transformer decoder block on the device; returns the block.

        The block is :class:`mojolearn.TransformerBlock` with its default
        options: input RMSNorm, causal self-attention with rotary position
        embeddings and grouped-query attention, residual, RMSNorm, a
        SiLU-gated MLP, residual. It keeps a key/value cache, so decoding a
        token costs the same at any position.

        Parameters
        ----------
        weights : list or tuple of arrays
            Nine arrays in this order, in the ``(out, in)`` layout:
            ``input_layernorm.weight``, ``post_attention_layernorm.weight``,
            ``q_proj.weight``, ``k_proj.weight``, ``v_proj.weight``,
            ``o_proj.weight``, ``gate_proj.weight``, ``up_proj.weight``,
            ``down_proj.weight``. ``d_model`` is read from the first array
            and the MLP width from ``gate_proj.weight``.
        n_heads : int
            Number of attention heads; ``d_model`` must be a multiple of it.
        max_positions : int
            The longest sequence the cache holds.
        batch_size : int, default 1
            Number of sequences the cache holds.
        n_kv_heads : int or None, default None
            Number of key/value heads; None means ``n_heads``.
        window : int, default 0
            Sliding-window length; 0 attends to every earlier position.
        requires_grad : bool, default False
            Make the weights trainable (recording sessions only).
        rope_theta, rope_scaling, rope_dim, qkv_bias, o_bias, norm, norm_eps, norm_bias, mlp, mlp_bias, qk_norm, attn_softcap
            The block options of :class:`mojolearn.TransformerBlock`, with
            the same meanings and defaults. An option that adds weights
            adds their arrays after the nine above, in this order:
            ``q_proj.bias``, ``k_proj.bias``, ``v_proj.bias`` (`qkv_bias`),
            ``o_proj.bias`` (`o_bias`), ``input_layernorm.bias``,
            ``post_attention_layernorm.bias`` (`norm_bias`),
            ``up_proj.bias``, ``down_proj.bias``, ``gate_proj.bias``
            (`mlp_bias`; the last only for a gated MLP), ``q_norm.weight``,
            ``k_norm.weight`` (`qk_norm`). An ungated MLP (``mlp='gelu'``
            or ``'gelu_tanh'``) has no ``gate_proj.weight``: leave it out.

        Returns
        -------
        dict
            ``'weights'`` is the handle of one tensor holding every weight
            and ``'state'`` the handle of the key/value cache, which starts
            empty. Pass the dict to :meth:`attention_step`.

        Example
        -------
        Two blocks stacked:

        >>> with ops.Session() as s:
        ...     first = s.attention_block(w0, n_heads=4, max_positions=64)
        ...     second = s.attention_block(w1, n_heads=4, max_positions=64)
        ...     h = s.attention_step(s.tensor(x), first)   # x: (1, tokens, d_model)
        ...     h = s.attention_step(h, second)
        """
        return _SESSION_CALL(self, "attention_block",
                             (weights, batch_size, n_heads, n_kv_heads, max_positions, window, requires_grad,
                              rope_theta, rope_scaling, rope_dim, qkv_bias, o_bias, norm, norm_eps, norm_bias,
                              mlp, mlp_bias, qk_norm, attn_softcap))

    def attention_step(self, x, block):
        """Run a transformer block over the next tokens; returns the output.

        ``x`` holds ``batch_size * tokens * d_model`` values, usually as a
        ``(batch_size, tokens, d_model)`` tensor; the result has ``x``'s
        shape. ``block`` comes from :meth:`attention_block`. The tokens are
        appended to the block's key/value cache: one token per call is a
        decode step, several a prefill, and decoding token by token gives
        the same bits as one prefill over the whole sequence.

        In a recording session each call starts from an empty cache and is
        differentiable: after :meth:`backward`, :meth:`grad` of
        ``block['weights']`` and of ``x`` return the gradients.
        """
        return _SESSION_CALL(self, "attention_step", (x, block))

    def small_mlp(self, x, weight1, bias1, weight2, bias2):
        """A two-layer perceptron ``relu(x @ weight1.T + bias1) @ weight2.T + bias2``.

        ``x`` is ``(rows, inputs)``, ``weight1`` ``(hidden, inputs)``,
        ``bias1`` holds ``hidden`` values, ``weight2`` is
        ``(outputs, hidden)`` and ``bias2`` holds ``outputs`` values; the
        result is ``(rows, outputs)``. One operation for the whole network
        and, in a recording session, one for its gradients. Its products are
        summed in a different order from two :meth:`linear` calls, so the
        two forms can differ in the last bits; each is the same on every
        vendor.
        """
        return _SESSION_CALL(self, "small_mlp", (x, weight1, bias1, weight2, bias2))

    def lm_head_loss(self, hidden, weight, codes):
        """Mean cross-entropy of ``hidden @ weight.T`` against class codes.

        ``hidden`` holds ``rows * width`` values, ``weight`` is
        ``(vocabulary, width)`` and ``codes`` holds one class code per row;
        the result is a ``(1, 1)`` tensor. The ``(rows, vocabulary)`` logits
        are never stored, so a large vocabulary needs no extra memory. In a
        recording session the loss is differentiable with respect to
        ``hidden`` and ``weight``.
        """
        return _SESSION_CALL(self, "lm_head_loss", (hidden, weight, codes))

    def lstm(self, x, weight_ih, weight_hh, bias_ih, bias_hh, h_0=None, c_0=None, lengths=None,
             reverse=False, return_state=False):
        """One LSTM layer over a whole sequence; returns every hidden state.

        ``x`` is ``(steps, batch, inputs)``, time first, as
        ``torch.nn.LSTM`` takes it by default. The weights have PyTorch's
        shapes and gate order (input, forget, cell, output):
        ``weight_ih`` ``(4 * hidden, inputs)``, ``weight_hh``
        ``(4 * hidden, hidden)``, ``bias_ih`` and ``bias_hh`` as
        ``(1, 4 * hidden)`` tensors. The result is ``(steps, batch,
        hidden)``. To stack layers, pass the result as the next layer's
        ``x``. In a recording session the layer is differentiable with
        respect to ``x`` and all four parameters.

        Parameters
        ----------
        h_0, c_0 : tensor or None, default None
            The initial hidden and cell states, each ``(batch, hidden)``
            (PyTorch's ``(1, batch, hidden)`` for one layer is accepted
            too). None starts that state at zero. They are values: no
            gradient reaches them.
        lengths : tensor or None, default None
            One length per sequence (``batch`` values, each from 1 to
            ``steps``) for a batch of padded sequences. The steps of a
            sequence at or past its length are padding: their inputs are
            ignored and the state is carried through them unchanged, so
            the outputs there repeat the state of the sequence's last
            step, as a packed sequence in PyTorch gives.
        reverse : bool, default False
            Run the sequence from its last step to its first (from each
            sequence's own last step with ``lengths``); the outputs stay in
            time order. A bidirectional layer is the forward and the
            reverse layer joined on the last axis.
        return_state : bool, default False
            Also return the final states: ``(outputs, h_n, c_n)``, each
            state ``(batch, hidden)``, the state after the last step run
            (the first step when ``reverse``). ``h_n`` carries the outputs'
            gradient; ``c_n`` is a value.

        Examples
        --------
        A bidirectional layer::

            fwd = s.lstm(x, w_ih, w_hh, b_ih, b_hh, lengths=n)
            bwd = s.lstm(x, r_ih, r_hh, rb_ih, rb_hh, lengths=n, reverse=True)
            both = s.concat([fwd, bwd], axis=2)  # (steps, batch, 2 * hidden)

        One step at a time, carrying the state (each ``x_t`` is
        ``(1, batch, inputs)``)::

            h, c = None, None
            for x_t in steps:
                y, h, c = s.lstm(x_t, w_ih, w_hh, b_ih, b_hh, h_0=h, c_0=c, return_state=True)

        Each step gives the same bits as the whole sequence in one call
        when ``inputs`` is at most 64.
        """
        return _SESSION_CALL(self, "lstm", (x, weight_ih, weight_hh, bias_ih, bias_hh, h_0, c_0, lengths,
                                            reverse, return_state))

    def gru(self, x, weight_ih, weight_hh, bias_ih, bias_hh, h_0=None, lengths=None, reverse=False,
            return_state=False):
        """One GRU layer over a whole sequence; returns every hidden state.

        As :meth:`lstm`, with PyTorch's GRU gate order (reset, update, new):
        ``weight_ih`` is ``(3 * hidden, inputs)``, ``weight_hh``
        ``(3 * hidden, hidden)`` and each bias ``(1, 3 * hidden)``. There
        is no cell state: ``return_state=True`` returns ``(outputs, h_n)``.

        One step at a time, carrying the state, gives the same bits as the
        whole sequence in one call when ``inputs`` is at most 64::

            h = None
            for x_t in steps:
                y, h = s.gru(x_t, w_ih, w_hh, b_ih, b_hh, h_0=h, return_state=True)
        """
        return _SESSION_CALL(self, "gru", (x, weight_ih, weight_hh, bias_ih, bias_hh, h_0, None, lengths,
                                           reverse, return_state))

    def rnn(self, x, weight_ih, weight_hh, bias_ih, bias_hh, h_0=None, lengths=None, reverse=False,
            return_state=False):
        """One tanh RNN layer over a whole sequence; returns every hidden state.

        As :meth:`gru`, with ``weight_ih`` ``(hidden, inputs)``,
        ``weight_hh`` ``(hidden, hidden)`` and each bias ``(1, hidden)``.
        """
        return _SESSION_CALL(self, "rnn", (x, weight_ih, weight_hh, bias_ih, bias_hh, h_0, None, lengths,
                                           reverse, return_state))

    def conv_block(self, x, weight, bias, *, padding=0, pool_size=None):
        """A convolution block ``max_pool(relu(conv2d(x, weight) + bias))``.

        ``x`` is ``(N, C, H, W)``, ``weight`` ``(out_channels, C, k, k)``
        and ``bias`` holds ``out_channels`` values. The convolution has
        stride 1 and zero ``padding`` on every side; ``pool_size=p`` adds a
        ``p`` by ``p`` max pool with stride ``p``, and None leaves the pool
        out. One operation for the whole block, as
        :class:`mojolearn.CNNClassifier` runs it, and one for its gradients
        in a recording session.
        """
        return _SESSION_CALL(self, "conv_block", (x, weight, bias, padding, pool_size))

    def conv_linear(self, x, weight, bias):
        """A dense layer ``x @ weight.T + bias`` in PyTorch's layout.

        ``x`` holds ``rows * inputs`` values (a ``(N, C, H, W)`` block
        output is read as ``N`` rows), ``weight`` is ``(outputs, inputs)``
        as in ``torch.nn.Linear`` and ``bias`` holds ``outputs`` values. It
        is the output layer :class:`mojolearn.CNNClassifier` puts after its
        convolution blocks.
        """
        return _SESSION_CALL(self, "conv_linear", (x, weight, bias))

    def lm_embedding(self, table, ids, padding_idx=None):
        """Rows of ``table`` picked by a tensor of token ids.

        The values are those of :meth:`embedding`. ``ids`` is a tensor made
        with :meth:`ids`. In a recording session the gradient of ``table``
        is computed with the method the language-model trainers use, which
        is faster for a long sequence over a large vocabulary; it gives the
        same gradient bits as :meth:`embedding`.
        """
        return _SESSION_CALL(self, "lm_embedding", (table, ids, padding_idx))

    def lm_rows(self, batch_size, eos_token_id=None):
        """Create the bookkeeping of one text generation; returns its handle.

        It records, for each of ``batch_size`` sequences, whether it has
        finished and how many tokens it has produced. ``eos_token_id`` is
        an id, or a list of ids, that ends a sequence. Pass the handle to
        :meth:`lm_pick` and :meth:`lm_alive`.
        """
        return _SESSION_CALL(self, "lm_rows", (batch_size, eos_token_id))

    def lm_pick(self, logits, rows, *, pad_token_id=0, do_sample=False, temperature=1.0, top_k=0,
                top_p=1.0, seed=0):
        """Choose the next token of every sequence; returns the ids.

        ``logits`` is ``(batch_size, tokens, vocabulary)`` or
        ``(batch_size, vocabulary)``; the last position of each sequence is
        read. The result is a ``(batch_size, 1)`` tensor of int32 ids, as
        :meth:`ids` makes: feed it to :meth:`lm_embedding` for the next
        step. A finished sequence gets ``pad_token_id``.

        ``do_sample=False`` takes the largest logit (the lowest id on a
        tie). ``do_sample=True`` samples after ``temperature``, ``top_k``
        (0 keeps every token) and ``top_p``, as in
        :meth:`mojolearn.CausalLM.generate`; the draw depends only on
        ``seed``, the sequence and its position, so it is the same on every
        vendor. ``rows`` (from :meth:`lm_rows`) is updated in place.
        """
        return _SESSION_CALL(self, "lm_pick",
                             (logits, rows, pad_token_id, do_sample, temperature, top_k, top_p, seed))

    def lm_alive(self, ids, rows):
        """Whether the generation should continue, as a ``(1, 1)`` tensor.

        1.0 while at least one sequence is unfinished, else 0.0. ``ids`` is
        the result of :meth:`lm_pick`. Capture it as the condition of
        :meth:`Program.while_loop` to stop a decode loop on the device.
        """
        return _SESSION_CALL(self, "lm_alive", (ids, rows))

    # ------------------------------------------------------------ data steps
    def permutation(self, n, seed=0, epoch=0):
        """A random order of the rows ``0 .. n-1`` as an ``(n, 1)`` tensor,
        determined by ``seed`` and ``epoch``."""
        return _SESSION_CALL(self, "permutation", (n, seed, epoch))

    def gather_rows(self, x, indices, start=0, count=None):
        """The rows of ``x`` listed in ``indices[start:start + count]``.

        ``indices`` is a :meth:`permutation` of at most ``x``'s row count.
        The result carries no gradient.
        """
        return _SESSION_CALL(self, "gather_rows", (x, indices, start, count))

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
        return _SESSION_CALL(self, "embedding", (table, ids, padding_idx))

    # ------------------------------------------------------------ losses (differentiable)
    def mse(self, prediction, target, reduction="mean", weight=None):
        """The squared error, averaged or summed: a ``(1, 1)`` tensor.

        ``weight`` (None or a tensor) weights each term: one value per
        element of ``prediction``, or one per row (``(rows, 1)``, the rows
        of the ``(rows, last axis)`` view). The mean is then
        ``sum(weight * term) / sum(weight)`` and the sum
        ``sum(weight * term)``; a weight of 0 leaves the term out, so a 0/1
        weight is a mask. The weight takes no gradient. A total weight of
        zero gives a loss of 0 and zero gradients (the smallest normal
        float32, 2**-126, is added to the total, which changes no result
        for a total above 2**-100).

        To average over several batches as one, take each batch's weighted
        sum (``reduction="sum"``) and its total weight (:meth:`sum` of the
        weight) and divide the totals, so a batch of 2 real tokens and one
        of 20 count every token once::

            num_a = s.cross_entropy(logits_a, codes_a, reduction="sum", weight=mask_a)  # 2 tokens
            num_b = s.cross_entropy(logits_b, codes_b, reduction="sum", weight=mask_b)  # 20 tokens
            loss = s.div(s.add(num_a, num_b), s.add(s.sum(mask_a), s.sum(mask_b)))
        """
        return _SESSION_CALL(self, "mse", (prediction, target, reduction, weight))

    def mae(self, prediction, target, reduction="mean", weight=None):
        """The absolute error, averaged or summed; ``weight`` as in
        :meth:`mse`."""
        return _SESSION_CALL(self, "mae", (prediction, target, reduction, weight))

    def huber(self, prediction, target, delta=1.0, reduction="mean", weight=None):
        """The Huber loss with threshold ``delta``, averaged or summed;
        ``weight`` as in :meth:`mse`."""
        return _SESSION_CALL(self, "huber", (prediction, target, delta, reduction, weight))

    def bce_with_logits(self, logits, target, reduction="mean", weight=None):
        """Binary cross-entropy of ``sigmoid(logits)`` against targets in
        ``[0, 1]``, computed stably from the logits; ``weight`` as in
        :meth:`mse` (for example 0 on padded steps and 30 on positives)."""
        return _SESSION_CALL(self, "bce_with_logits", (logits, target, reduction, weight))

    def cross_entropy(self, logits, codes, reduction="mean", weight=None):
        """Softmax cross-entropy of ``(..., classes)`` logits against class
        codes (the class index as a float), one per position of the leading
        axes: ``(rows, 1)`` codes for ``(rows, classes)`` logits.

        ``weight`` (None or a tensor) holds one value per position (the
        codes' count); the mean is ``sum(weight * loss) / sum(weight)``, as
        ``torch.nn.CrossEntropyLoss`` normalizes a weighted mean, and the
        rest as in :meth:`mse`.
        """
        return _SESSION_CALL(self, "cross_entropy", (logits, codes, reduction, weight))

    # ------------------------------------------------------------ metrics (values)
    def accuracy(self, y_true, prediction):
        """The fraction of rows classified correctly, as a float.

        ``y_true`` holds class codes ``(n, 1)``; ``prediction`` holds codes
        ``(n, 1)`` or scores ``(n, classes)`` (the first highest wins)."""
        return _SESSION_CALL(self, "accuracy", (y_true, prediction))

    def log_loss(self, y_true, probability):
        """The mean negative log-likelihood of the class codes ``(n, 1)``
        under the probabilities ``(n, classes)`` (or ``(n, 1)``, the
        positive class), as a float."""
        return _SESSION_CALL(self, "log_loss", (y_true, probability))

    def roc_auc(self, y_true, scores):
        """The area under the ROC curve of binary labels ``(n, 1)`` (1
        positive) against scores ``(n, 1)``, as a float."""
        return _SESSION_CALL(self, "roc_auc", (y_true, scores))

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
        return _SESSION_CALL(self, "backward", (loss, accumulate, create_graph))

    def grad(self, h):
        """The gradient of tensor ``h`` from the last :meth:`backward`, as a
        float32 array of its shape."""
        return _SESSION_CALL(self, "grad", (h,))

    def grad_tensor(self, h):
        """The gradient of tensor ``h`` from the last
        ``backward(..., create_graph=True)``, as a tensor handle.

        Unlike :meth:`grad`, which copies the values back as an array, the
        handle stays on the device and is differentiable: use it in further
        operations and call :meth:`backward` on a scalar built from it to
        get second derivatives. Raises if the last backward did not record
        its gradients.
        """
        return _SESSION_CALL(self, "grad_tensor", (h,))

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
        return _SESSION_CALL(self, "vjp", (output, inputs, v, create_graph))

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
        return _SESSION_CALL(self, "jvp", (output, inputs, u))

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
        return _SESSION_CALL(self, "hvp", (loss, params, v))

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
        return _SESSION_CALL(self, "capture", (inputs, outputs))

    def run(self, program, *arrays, dynamic_shapes=False, optimize=False):
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
        return _SESSION_CALL(self, "program_run", (program, arrays, dynamic_shapes, optimize))

    # nn-blocks-scan begin
    def scan(self, cell, xs, init_state, *, params=None, reverse=False, lengths=None, counter=0):
        """Apply a captured cell at every step of a sequence, carrying its state.

        ``cell`` is a :class:`Program` captured with
        ``inputs=[state_1, ..., state_k, x_t]`` and
        ``outputs=[next_state_1, ..., next_state_k, y_t]``: one step of any
        recurrence written with this session's operations (a custom RNN or
        LSTM cell, a state-space update, a filter). ``scan`` runs it once per
        step of ``xs``, on the device, in one call: step ``t`` reads
        ``xs[t]`` and the state the previous step returned, the state stays
        on the device between steps and nothing is read back. Like
        ``jax.lax.scan`` and PyTorch's recurrent layers, the outputs are
        stacked with the steps first.

        In a recording session (``Session(record=True)``) the scan is
        differentiable: :meth:`backward` reaches ``init_state``, ``xs`` and
        every tensor in ``params``, and :meth:`step` trains them. The
        gradient keeps every step's values (backward through time). In an
        inference session the scan is captured with the rest of the model
        by :meth:`capture`, and a saved program keeps the cell.

        Parameters
        ----------
        cell : Program
            The step, captured from an inference session.
        xs : int
            The handle of a ``(steps, ...)`` tensor; ``xs[t]`` must have the
            shape of the cell's last input.
        init_state : int or sequence of int
            The initial state tensors, in the cell's input order, each of
            the shape the cell's input had. A sequence may be empty (a map
            over the steps).
        params : sequence of int, optional
            Tensors of this session for the tensors the cell reads as
            parameters (:attr:`Program.params`), in that order and of the
            same shapes. Required when the cell was captured in another
            session (for example, a training session's own trainable
            weights); by default the cell reads its own parameters.
        reverse : bool, default False
            Run from the last step to the first. The outputs keep their
            time order (``ys[t]`` belongs to ``xs[t]``), and the final state
            is the state after step 0.
        lengths : int, optional
            The handle of one length per sequence (float values), for ``xs``
            of shape ``(steps, batch, ...)`` and states and outputs with the
            batch first. A sequence's steps at or past its length keep its
            state and give zero outputs, as a packed sequence in PyTorch.
        counter : int, default 0
            The random key of this call: a :meth:`dropout` inside the cell
            draws a new mask at every step, from its own seed, this key and
            the step index (not from the sequence length). Use a new
            ``counter`` for every training step, as with :meth:`dropout`;
            :meth:`Program.fit` advances it every minibatch by itself.

        Returns
        -------
        tuple
            ``(ys, final_state)``: ``ys`` the ``(steps, ...)`` outputs, and
            ``final_state`` one handle when ``init_state`` was one handle,
            else a tuple of handles.

        Examples
        --------
        A tanh RNN cell, trained through a scan:

        >>> import numpy as np
        >>> from mojolearn import ops
        >>> rng = np.random.default_rng(0)
        >>> W = (0.3 * rng.standard_normal((3, 4))).astype(np.float32)
        >>> U = (0.3 * rng.standard_normal((4, 4))).astype(np.float32)
        >>> with ops.Session() as c:
        ...     h, x = c.tensor(np.zeros((2, 4), np.float32)), c.tensor(np.zeros((2, 3), np.float32))
        ...     h2 = c.tanh(c.add(c.matmul(x, c.tensor(W)), c.matmul(h, c.tensor(U))))
        ...     cell = c.capture(inputs=[h, x], outputs=[h2, h2])
        ...     with ops.Session(record=True) as s:
        ...         w, u = s.tensor(W, requires_grad=True), s.tensor(U, requires_grad=True)
        ...         xs = s.tensor(rng.standard_normal((5, 2, 3), dtype=np.float32))
        ...         ys, h_n = s.scan(cell, xs, s.tensor(np.zeros((2, 4), np.float32)), params=[w, u])
        ...         s.backward(s.mean(s.mul(ys, ys)))
        ...         s.step(s.optimizer("adam", lr=0.01), [w, u])
        """
        return _SESSION_CALL(self, "scan", (cell, xs, init_state, params, reverse, lengths, counter))
    # nn-blocks-scan end

    def load_partitioned(self, path, devices=(0,)):
        """Restore a complete partitioned program onto physical devices."""
        return _SESSION_CALL(self, "load_partitioned", (path, devices))

    def load_distributed(self, path, devices=(0,)):
        """Restore a complete distributed program onto the given devices."""
        return _SESSION_CALL(self, "load_distributed", (path, devices))

    def load_control(self, path):
        """Load control components and shared parameters from one checkpoint."""
        return _SESSION_CALL(self, "load_control", (path,))

    def load_program(self, path):
        """Load a :class:`Program` saved with :meth:`Program.save`.

        Its parameters become new tensors of this session. The session must
        not record. A program saved on one install loads on any other,
        a GPU install or the CPU-only install, and gives the same outputs.
        """
        return _SESSION_CALL(self, "load_program", (path,))

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
        return _SESSION_CALL(self, "save_state", (path, params, optimizer, seed, epoch, step))

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
        return _SESSION_CALL(self, "load_state", (path, optimizer, requires_grad))
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
        return _SESSION_CALL(self, "load_weights", (path, names, source))

    def zero_grad(self):
        """Clear the gradient of every tensor made with ``requires_grad=True``.

        The next ``backward(loss, accumulate=True)`` then starts each
        gradient from zero. Use it to sum the gradients of several losses
        before one :meth:`step`; a plain :meth:`backward` already replaces
        the previous gradients.
        """
        return _SESSION_CALL(self, "zero_grad", ())

    # ------------------------------------------------------------ optimizers
    def optimizer(self, kind="adam", lr=1e-3, *, momentum=0.0, nesterov=False, beta1=0.9, beta2=0.999,
                  eps=1e-8, weight_decay=0.0, schedule="constant", step_size=1, gamma=0.1,
                  alpha=0.99, centered=False, lr_decay=0.0, initial_accumulator_value=0.0,
                  clip_grad_norm=None, clip_grad_value=None):
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
        return _SESSION_CALL(self, "optimizer", (kind, lr, momentum, nesterov, beta1, beta2, eps, weight_decay, schedule, step_size, gamma,
                                                 alpha, centered, lr_decay, initial_accumulator_value,
                                                 clip_grad_norm, clip_grad_value))

    def step(self, optimizer, params):
        """Update the parameters in place from the last :meth:`backward`.

        ``params`` lists tensors made with ``requires_grad=True`` (or is one
        such handle). Each is updated on the device by ``optimizer``; its
        optimizer state (momentum or moment estimates and the step count)
        stays on the device in the session, separately for every parameter.
        Nothing is read back. Call :meth:`reset` before recording the next
        forward pass.
        """
        return _SESSION_CALL(self, "step", (optimizer, params))

    def scheduler_step(self, optimizer):
        """Advance ``optimizer``'s learning-rate schedule by one epoch."""
        return _SESSION_CALL(self, "scheduler_step", (optimizer,))

    def learning_rate(self, optimizer):
        """``optimizer``'s current learning rate, as a float."""
        return _SESSION_CALL(self, "learning_rate", (optimizer,))

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
        return _SESSION_CALL(self, "optimizer_state", (optimizer, param))

    def load_optimizer_state(self, optimizer, param, state):
        """Restore the state :meth:`optimizer_state` returned.

        ``state`` holds the same keys, with each buffer a tensor handle of
        ``param``'s shape (for example ``s.tensor(saved_array)``). The
        values are copied on the device; the next :meth:`step` continues
        exactly where the saved run stopped.
        """
        return _SESSION_CALL(self, "load_optimizer_state", (optimizer, param, state))

    # nn-blocks-qkv begin
    def attention(self, q, k, v, mask=None, causal=False, scale=None):
        """Scaled dot-product attention over your own queries, keys and values.

        Computes ``softmax(scale * q @ k^T + mask) @ v`` with the argument
        meanings of PyTorch's ``torch.nn.functional.scaled_dot_product_attention``:
        ``q`` is ``(batch, heads, Lq, d)``, ``k`` is ``(batch, kv_heads, Lk, d)``
        and ``v`` is ``(batch, kv_heads, Lk, dv)``; the result is
        ``(batch, heads, Lq, dv)``. Inputs without the heads axis,
        ``(batch, L, features)``, give ``(batch, Lq, dv)``. ``Lq`` and ``Lk``
        may differ (cross-attention: queries from one sequence, keys and
        values from another).

        Grouped heads: ``heads`` may be a multiple of ``kv_heads``; query head
        ``h`` attends with key/value head ``h // (heads // kv_heads)``, as
        PyTorch's ``enable_gqa=True``. The keys and values are never copied
        per group.

        Parameters
        ----------
        q, k, v : int
            Tensor handles of the shapes above.
        mask : int or None, default None
            A tensor that broadcasts to the scores ``(batch, heads, Lq, Lk)``
            (aligned on the right, as NumPy broadcasts). As in PyTorch, its
            kind decides how it is read: a tensor made with :meth:`ids` is a
            boolean mask, where a nonzero value takes part in attention and a
            zero is left out (pass ``s.ids(mask.astype(np.int8))`` for a
            boolean array); a float tensor made with :meth:`tensor` is added
            to the scaled scores, and ``-inf`` there leaves the position out.
            A float mask made with ``requires_grad=True`` receives its
            gradient (summed over the axes it broadcasts along), as in
            PyTorch; a boolean mask takes none.
        causal : bool, default False
            Query ``i`` attends to keys ``0 .. i`` only (PyTorch's
            ``is_causal``, the lower triangle aligned at the top left when
            ``Lq`` and ``Lk`` differ). It cannot be combined with ``mask``;
            put the causal pattern into the mask instead.
        scale : float or None, default None
            The factor on ``q @ k^T``; None means ``1 / sqrt(d)``.

        Returns
        -------
        int
            The handle of the attention output.

        A query whose every key is left out (by the mask or by ``causal``)
        gets an output row of zeros and passes zero gradient back; PyTorch
        returns NaN for such a row. In a recording session the gradients
        flow to ``q``, ``k`` and ``v`` through :meth:`backward`, :meth:`vjp`
        and :meth:`Program.fit`. The softmax sums each row in one fixed
        order, so the result has the same bits on every device. The call can
        be captured into a :class:`Program` like any other operation, the
        mask included (as an input, or as a fixed tensor of the program).

        Example
        -------
        >>> import numpy as np
        >>> from mojolearn import ops
        >>> rng = np.random.default_rng(0)
        >>> with ops.Session(record=True) as s:
        ...     q = s.tensor(rng.standard_normal((2, 4, 5, 8), dtype=np.float32), requires_grad=True)
        ...     k = s.tensor(rng.standard_normal((2, 2, 7, 8), dtype=np.float32), requires_grad=True)
        ...     v = s.tensor(rng.standard_normal((2, 2, 7, 8), dtype=np.float32), requires_grad=True)
        ...     keep = np.ones((2, 1, 1, 7), dtype=np.int8)
        ...     keep[1, ..., 5:] = 0                       # the second sequence has 5 keys
        ...     out = s.attention(q, k, v, mask=s.ids(keep))
        ...     s.backward(s.sum(out))
        ...     dq, dk, dv = s.grad(q), s.grad(k), s.grad(v)
        """
        return _SESSION_CALL(self, "attention", (q, k, v, mask, causal, scale))
    # nn-blocks-qkv end
    # nn-blocks-pg begin: optimizer parameter groups (autograd/nn_pg.mojo)
    def param_group_optimizer(self, param_groups, kind="adam", lr=1e-3, *, momentum=0.0, nesterov=False,
                              beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.0, schedule="constant",
                              step_size=1, gamma=0.1, alpha=0.99, centered=False, lr_decay=0.0,
                              initial_accumulator_value=0.0, clip_grad_norm=None, clip_grad_value=None):
        """Create an optimizer with parameter groups for :meth:`step`; returns its integer handle.

        As ``torch.optim`` with a list of parameter groups: each group is a
        dict with ``'params'`` (the tensors of the group, made with
        ``requires_grad=True``) and, optionally, its own options, which
        override the optimizer-wide ones given as keywords. A typical use is
        a new head at a higher learning rate than a pretrained backbone, or
        biases without weight decay.

        Parameters
        ----------
        param_groups : list of dict
            One dict per group. Keys: ``'params'`` and, for each kind,
            ``'lr'``, ``'weight_decay'`` and ``'sgd'``: ``'momentum'``,
            ``'nesterov'``, ``'dampening'``; ``'adam'``, ``'adamw'``,
            ``'adamax'``, ``'nadam'``: ``'betas'`` (or ``'beta1'`` and
            ``'beta2'``), ``'eps'``; ``'lion'``: ``'betas'``; ``'rmsprop'``:
            ``'alpha'``, ``'eps'``, ``'momentum'``, ``'centered'``;
            ``'adagrad'``: ``'eps'``, ``'lr_decay'``,
            ``'initial_accumulator_value'``. A parameter belongs to one group.
        kind, lr, momentum, nesterov, beta1, beta2, eps, weight_decay, alpha, centered, lr_decay, initial_accumulator_value
            The defaults of every group, as in :meth:`optimizer`.
        schedule, step_size, gamma
            One learning-rate schedule for the optimizer; it scales every
            group's rate. Advance it with :meth:`scheduler_step`.
        clip_grad_norm : float or None, default None
            Clip by the norm of ALL the gradients of every group that one
            :meth:`step` updates (one norm, one coefficient for every group,
            as ``torch.nn.utils.clip_grad_norm_`` over all parameters).
        clip_grad_value : float or None, default None
            Clamp every gradient to ``[-clip_grad_value, clip_grad_value]``.

        Each group's update is the one :meth:`optimizer` with that group's
        options gives, bit for bit. :meth:`optimizer_state` and
        :meth:`load_optimizer_state` work per parameter as for any optimizer;
        :meth:`optimizer_state_dict` saves every group at once.

        Example
        -------
        >>> with ops.Session(record=True) as s:
        ...     body = s.tensor(np.zeros((4, 8), dtype=np.float32), requires_grad=True)
        ...     head = s.tensor(np.zeros((8, 1), dtype=np.float32), requires_grad=True)
        ...     bias = s.tensor(np.zeros((1, 1), dtype=np.float32), requires_grad=True)
        ...     opt = s.param_group_optimizer(
        ...         [{"params": [body], "lr": 1e-4},
        ...          {"params": [head], "lr": 1e-2},
        ...          {"params": [bias], "lr": 1e-2, "weight_decay": 0.0}],
        ...         "adamw", weight_decay=0.01, clip_grad_norm=1.0)
        """
        return _SESSION_CALL(self, "param_group_optimizer", (kind, lr, momentum, nesterov, beta1, beta2, eps,
                                                             weight_decay, schedule, step_size, gamma, alpha,
                                                             centered, lr_decay, initial_accumulator_value,
                                                             clip_grad_norm, clip_grad_value, param_groups))

    def learning_rates(self, optimizer):
        """Every parameter group's current learning rate, as a list of floats
        (an optimizer made by :meth:`param_group_optimizer`)."""
        return _SESSION_CALL(self, "learning_rates", (optimizer,))

    def optimizer_state_dict(self, optimizer):
        """The whole state of an optimizer with parameter groups, as a dict.

        ``torch.optim``'s layout: ``'state'`` maps each parameter's index
        (its position across the groups, in group order) to the dict
        :meth:`optimizer_state` returns for it, and ``'param_groups'`` lists
        each group's options with ``'params'`` holding those indices;
        ``'epoch'`` is the schedule's epoch. The state tensors are copies,
        valid until :meth:`reset`: read them with :meth:`numpy` to save a
        checkpoint.
        """
        return _SESSION_CALL(self, "optimizer_state_dict", (optimizer,))

    def load_optimizer_state_dict(self, optimizer, state_dict):
        """Restore what :meth:`optimizer_state_dict` returned, into an
        optimizer built with the same groups (the same number of groups and
        of parameters in each; state tensors as handles of this session).
        Each group's options are taken from the dict, every parameter's
        state is copied on the device, and the next :meth:`step` continues
        exactly where the saved run stopped."""
        return _SESSION_CALL(self, "load_optimizer_state_dict", (optimizer, state_dict))
    # nn-blocks-pg end

    # nn-blocks-scatter begin
    # ------------------------------------------------------------ indexed writes and grouped reductions
    def scatter_add(self, x, dim, index, src):
        """``x`` with every value of ``src`` added at the position ``index`` names along ``dim``, as
        ``torch.Tensor.scatter_add``.

        For a 2-D ``x`` and ``dim=0`` the result is ``x`` with ``out[index[i, j], j] += src[i, j]`` for
        every ``(i, j)``; any rank and any ``dim`` work the same way. ``index`` is an integer tensor made
        with :meth:`ids`, of ``src``'s shape, each value in ``[0, x.shape[dim])``; ``src`` has ``x``'s size
        on every dimension except ``dim``. The result has ``x``'s shape.

        An index may repeat: the values that land on one position are added to ``x``'s value one after
        another in ascending position in ``src``, so the result is the same bits whatever the hardware.
        A position no index names keeps ``x``'s value.

        In a recording session the gradient flows to ``x`` (unchanged) and to ``src`` (the gradient at
        the position each value was added to); ``index`` takes none.

        Example
        -------
        >>> import numpy as np
        >>> from mojolearn import ops
        >>> with ops.Session() as s:
        ...     x = s.tensor(np.zeros((3, 2), dtype=np.float32))
        ...     src = s.tensor(np.ones((4, 2), dtype=np.float32))
        ...     index = s.ids(np.array([[0, 1], [2, 1], [0, 0], [2, 1]]))
        ...     s.numpy(s.scatter_add(x, 0, index, src))
        array([[2., 1.],
               [0., 3.],
               [2., 0.]], dtype=float32)
        """
        return _SESSION_CALL(self, "scatter_add", (x, dim, index, src))

    def segment_sum(self, data, segment_ids, num_segments):
        """The rows of ``data`` added up by segment: row ``s`` of the result is the sum of the rows ``r``
        of ``data`` with ``segment_ids[r] == s``.

        ``data`` is ``(n, ...)``; ``segment_ids`` is an integer tensor made with :meth:`ids` holding one
        id per row of ``data``, each in ``[0, num_segments)``, in any order (as
        ``tf.math.unsorted_segment_sum``); the result is ``(num_segments, ...)``. A segment no row names
        is 0. The rows of a segment are added in ascending row order, so the result is the same bits
        whatever the hardware. The gradient of ``data`` is the gradient of each row's segment.

        This is the aggregation of a graph neural network (sum the messages of each edge into its target
        node), of a bag of embeddings (sum the vectors of each bag) and of pooling by group.

        Example
        -------
        >>> import numpy as np
        >>> from mojolearn import ops
        >>> with ops.Session() as s:
        ...     data = s.tensor(np.array([[1, 2], [3, 4], [5, 6]], dtype=np.float32))
        ...     ids = s.ids(np.array([2, 0, 2]))
        ...     s.numpy(s.segment_sum(data, ids, num_segments=3))
        array([[3., 4.],
               [0., 0.],
               [6., 8.]], dtype=float32)
        """
        return _SESSION_CALL(self, "segment_sum", (data, segment_ids, num_segments))

    def segment_mean(self, data, segment_ids, num_segments):
        """The rows of ``data`` averaged by segment: :meth:`segment_sum` divided by the number of rows in
        each segment.

        Arguments as :meth:`segment_sum`. A segment no row names is 0 (the sum divided by
        ``max(count, 1)``). The gradient of a row is its segment's gradient divided by the segment's row
        count.

        Example
        -------
        >>> import numpy as np
        >>> from mojolearn import ops
        >>> with ops.Session() as s:
        ...     data = s.tensor(np.array([[1, 2], [3, 4], [5, 6]], dtype=np.float32))
        ...     ids = s.ids(np.array([2, 0, 2]))
        ...     s.numpy(s.segment_mean(data, ids, num_segments=3))
        array([[3., 4.],
               [0., 0.],
               [3., 4.]], dtype=float32)
        """
        return _SESSION_CALL(self, "segment_mean", (data, segment_ids, num_segments))

    def segment_max(self, data, segment_ids, num_segments):
        """The largest value of each column within each segment of ``data``'s rows.

        Arguments as :meth:`segment_sum`. A segment no row names is 0 (as ``torch_scatter.scatter_max``,
        so the result stays finite). A NaN in a segment makes that value NaN. In a recording session the
        gradient of each result value goes to the row that holds the maximum; when several rows hold it,
        to the first of them (the lowest row index); the other rows get 0.

        Example
        -------
        >>> import numpy as np
        >>> from mojolearn import ops
        >>> with ops.Session() as s:
        ...     data = s.tensor(np.array([[1, 8], [3, 4], [5, 6]], dtype=np.float32))
        ...     ids = s.ids(np.array([2, 0, 2]))
        ...     s.numpy(s.segment_max(data, ids, num_segments=3))
        array([[3., 4.],
               [0., 0.],
               [5., 8.]], dtype=float32)
        """
        return _SESSION_CALL(self, "segment_max", (data, segment_ids, num_segments))
    # nn-blocks-scatter end

    # nn-blocks-cbw begin ---------------------------------------- user-defined backward rules
    def custom_op(self, forward, backward, *, saved=(), outputs=None):
        """Give a captured operation your own backward rule; returns its handle.

        ``forward`` computes the operation's outputs from its inputs, and
        ``backward`` computes the gradient of each input from the saved
        tensors and the incoming gradient of each output: a vector-Jacobian
        product, like ``torch.autograd.Function``'s ``forward`` and
        ``backward``. Both are :class:`Program` objects captured from an
        inference session with the operations of this module. They are
        checked against each other here, once: the counts, shapes and kinds
        of the inputs, outputs, saved tensors and gradients must agree, and a
        mismatch raises ``ValueError`` naming it.

        Call the operation with :meth:`custom`. In a recording session,
        :meth:`backward` runs ``backward`` on the device with the gradient
        that reaches each output from the rest of the model (for
        ``loss = 3 * sum(f(x))`` that gradient is 3, not 1), and adds each
        gradient it returns to the input's other contributions, so an input
        used along several paths receives all of them. The same holds in
        :meth:`Program.fit` for a program captured with the operation, and a
        saved program keeps both graphs.

        Parameters
        ----------
        forward : Program
            ``(x_0, ..., x_{n-1}) -> (y_0, ..., y_{m-1}, extra, ...)``: the
            operation's outputs, then any further tensors the backward needs.
        backward : Program
            ``(saved_0, ..., grad_y_0, ..., grad_y_{m-1}) -> (grad_x, ...)``:
            the saved tensors, then the incoming gradient of each output (of
            that output's shape); it returns one gradient per float input of
            ``forward``, in input order, each of its input's shape (an input
            made with :meth:`ids` takes no gradient).
        saved : sequence of int, default ()
            Indices into ``forward``'s outputs: the tensors ``backward``
            receives first, in this order. They stay on the device from the
            forward to the backward.
        outputs : int or None, default None
            How many leading outputs of ``forward`` the operation returns;
            None returns every output. Each later output must be saved.

        Returns
        -------
        int
            The operation's handle, for :meth:`custom`.

        The first version has these limits. Both programs are built from the
        operations of this module (a new device kernel needs a compiled Mojo
        extension). Neither may read a tensor of its session as a parameter:
        capture every tensor it reads as an input (a learnable weight then
        gets its gradient from ``backward``) and make constants with
        :meth:`full`. The rule is first order: a second derivative through
        it (``backward(..., create_graph=True)``, :meth:`hvp`, :meth:`jvp`)
        is refused.

        Use it for a gradient that differs from the forward's: a
        straight-through estimator, gradient reversal, a clipped gradient, or
        a numerically stable gradient of a custom activation.

        Example
        -------
        ``MyTanh``: forward ``y = tanh(x)``, saving ``y``; backward
        ``grad_y * (1 - y * y)``.

        >>> import numpy as np
        >>> from mojolearn import ops
        >>> zeros = np.zeros((4, 3), np.float32)
        >>> with ops.Session() as cap:
        ...     x = cap.tensor(zeros)
        ...     forward = cap.capture(inputs=[x], outputs=[cap.tanh(x)])
        ...     y, grad_y = cap.tensor(zeros), cap.tensor(zeros)
        ...     one = cap.full((1, 1), 1.0)
        ...     dx = cap.mul(grad_y, cap.sub(one, cap.mul(y, y)))
        ...     backward = cap.capture(inputs=[y, grad_y], outputs=[dx])
        ...     with ops.Session(record=True) as s:
        ...         my_tanh = s.custom_op(forward, backward, saved=[0])
        ...         w = s.tensor(np.ones((4, 3), np.float32), requires_grad=True)
        ...         loss = s.scale(s.sum(s.custom(my_tanh, w)), 3.0)
        ...         s.backward(loss)
        ...         grad_w = s.grad(w)
        """
        return _SESSION_CALL(self, "custom_op", (forward, backward, saved, outputs))

    def custom(self, op, *inputs):
        """Call the operation :meth:`custom_op` returned on tensor handles.

        ``inputs`` are in the order of the forward program's inputs, each of
        its captured rank. Returns the handle of the output, or a tuple of
        handles when the operation returns several. In an inference session
        :meth:`capture` keeps the call like any other operation.

        Example
        -------
        >>> y = s.custom(my_tanh, w)  # doctest: +SKIP
        """
        return _SESSION_CALL(self, "custom", (op, inputs))
    # nn-blocks-cbw end

    # session-ops-conv begin ------------------------------------ Sequential's convolutions
    def conv1d(self, x, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
        """A 1-D convolution (cross-correlation) of ``x``, as ``torch.nn.functional.conv1d``.

        ``x`` is ``(N, C, L)``, ``weight`` ``(out_channels, C // groups, k)``
        and ``bias`` an optional tensor of ``out_channels`` values. The
        input is padded with ``padding`` zeros at both ends, the kernel
        moves ``stride`` positions at a time and its taps are ``dilation``
        positions apart; ``padding`` may also be ``'valid'`` or ``'same'``
        (stride 1 and ``dilation * (k - 1)`` even). The result is
        ``(N, out_channels, L_out)`` with
        ``L_out = (L + 2 * padding - dilation * (k - 1) - 1) // stride + 1``.
        With ``groups`` greater than 1 each block of outputs sees only its
        block of input channels. The values and gradients are those of
        :class:`Sequential`'s ``conv1d`` layer.

        Example
        -------
        >>> import numpy as np
        >>> from mojolearn import ops
        >>> with ops.Session(record=True) as s:
        ...     x = s.tensor(np.ones((2, 3, 8), np.float32))
        ...     w = s.tensor(np.full((4, 3, 3), 0.1, np.float32), requires_grad=True)
        ...     y = s.conv1d(x, w, padding=2, dilation=2)
        ...     s.backward(s.sum(y))
        ...     s.shape(y)
        (2, 4, 8)
        """
        return _SESSION_CALL(self, "conv1d", (x, weight, bias, stride, padding, dilation, groups))

    def conv3d(self, x, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
        """A 3-D convolution (cross-correlation) of ``x``, as ``torch.nn.functional.conv3d``.

        ``x`` is ``(N, C, D, H, W)``, ``weight`` ``(out_channels, C // groups,
        kd, kh, kw)`` and ``bias`` an optional tensor of ``out_channels``
        values. ``stride``, ``padding`` and ``dilation`` are an integer or a
        triple ``(depth, height, width)``; ``padding`` may also be
        ``'valid'`` or ``'same'`` (stride 1 and ``dilation * (k - 1)`` even
        on each axis). The result is ``(N, out_channels, D_out, H_out,
        W_out)`` with ``D_out = (D + 2 * padding - dilation * (kd - 1) - 1)
        // stride + 1`` and the others likewise. The values and gradients
        are those of :class:`Sequential`'s ``conv3d`` layer.

        Example
        -------
        >>> import numpy as np
        >>> from mojolearn import ops
        >>> with ops.Session(record=True) as s:
        ...     x = s.tensor(np.ones((1, 2, 4, 6, 6), np.float32))
        ...     w = s.tensor(np.full((3, 2, 3, 3, 3), 0.05, np.float32), requires_grad=True)
        ...     y = s.conv3d(x, w, stride=(1, 2, 2), padding=1)
        ...     s.backward(s.sum(y))
        ...     s.shape(y)
        (1, 3, 4, 3, 3)
        """
        return _SESSION_CALL(self, "conv3d", (x, weight, bias, stride, padding, dilation, groups))

    def conv_transpose1d(self, x, weight, bias=None, stride=1, padding=0, output_padding=0, groups=1,
                         dilation=1):
        """A 1-D transposed convolution of ``x``, as ``torch.nn.functional.conv_transpose1d``.

        ``x`` is ``(N, C_in, L)``, ``weight`` ``(C_in, C_out // groups, k)``
        and ``bias`` an optional tensor of ``C_out`` values. The result is
        ``(N, C_out, L_out)`` with
        ``L_out = (L - 1) * stride - 2 * padding + dilation * (k - 1) + output_padding + 1``:
        the gradient of ``conv1d`` with the same weight, stride, padding and
        dilation with respect to its input, plus the bias. It is the usual
        learned upsampling of a decoder. ``output_padding`` must be smaller
        than ``stride``. The values and gradients are those of
        :class:`Sequential`'s ``conv_transpose1d`` layer.

        Example
        -------
        >>> import numpy as np
        >>> from mojolearn import ops
        >>> with ops.Session(record=True) as s:
        ...     x = s.tensor(np.ones((2, 4, 5), np.float32))
        ...     w = s.tensor(np.full((4, 2, 3), 0.1, np.float32), requires_grad=True)
        ...     y = s.conv_transpose1d(x, w, stride=2, padding=1, output_padding=1)
        ...     s.backward(s.sum(y))
        ...     s.shape(y)
        (2, 2, 10)
        """
        return _SESSION_CALL(self, "conv_transpose1d",
                             (x, weight, bias, stride, padding, output_padding, groups, dilation))

    def conv_transpose2d(self, x, weight, bias=None, stride=1, padding=0, output_padding=0, groups=1,
                         dilation=1):
        """A 2-D transposed convolution of ``x``, as ``torch.nn.functional.conv_transpose2d``.

        ``x`` is ``(N, C_in, H, W)``, ``weight`` ``(C_in, C_out // groups,
        kh, kw)`` and ``bias`` an optional tensor of ``C_out`` values.
        ``stride``, ``padding``, ``output_padding`` and ``dilation`` are an
        integer or a pair ``(height, width)``. The result is ``(N, C_out,
        H_out, W_out)`` with ``H_out = (H - 1) * stride - 2 * padding +
        dilation * (kh - 1) + output_padding + 1`` and ``W_out`` likewise:
        the gradient of ``conv2d`` with the same weight, stride, padding and
        dilation with respect to its input, plus the bias. ``output_padding``
        must be smaller than ``stride`` on each axis. The values and
        gradients are those of :class:`Sequential`'s ``conv_transpose2d``
        layer.

        Example
        -------
        >>> import numpy as np
        >>> from mojolearn import ops
        >>> with ops.Session(record=True) as s:
        ...     x = s.tensor(np.ones((1, 4, 3, 3), np.float32))
        ...     w = s.tensor(np.full((4, 2, 2, 2), 0.1, np.float32), requires_grad=True)
        ...     y = s.conv_transpose2d(x, w, stride=2)
        ...     s.backward(s.sum(y))
        ...     s.shape(y)
        (1, 2, 6, 6)
        """
        return _SESSION_CALL(self, "conv_transpose2d",
                             (x, weight, bias, stride, padding, output_padding, groups, dilation))
    # session-ops-conv end


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
        return _SESSION_CALL(session, "program_init", (self, info))

    def run(self, *arrays, dynamic_shapes=False, optimize=False):
        """Run the program on new input arrays (see :meth:`Session.run`)."""
        return _PROGRAM_CALL(self, "run", (arrays, dynamic_shapes, optimize, False))

    def inspect(self, *, optimize=False):
        """Return the graph's operations, references, attributes and shapes."""
        return _PROGRAM_CALL(self, "inspect", (optimize,))

    def profile(self, *arrays, dynamic_shapes=False, optimize=False):
        """Run and return outputs with per-node timing and storage accounting."""
        return _PROGRAM_CALL(self, "profile", (arrays, dynamic_shapes, optimize, True))

    def run_if(self, condition, otherwise, *arrays, optimize=False):
        """Evaluate a scalar condition graph and execute its selected branch."""
        return _PROGRAM_CALL(self, "run_if", (condition, otherwise, arrays, 0, optimize, None, None))

    def run_dynamic(self, *arrays, optimize=False, profile=False):
        """Replay on compatible input ranks and dimensions, deriving every output shape."""
        return _PROGRAM_CALL(self, "run_dynamic", (arrays, 2, optimize, profile))

    def vjp_if(self, condition, otherwise, *arrays, seeds, parameters=None):
        """Differentiate the selected branch against output seeds in one native call."""
        return _PROGRAM_CALL(self, "vjp_if", (condition, otherwise, arrays, 0, False, seeds, parameters))

    def vjp_while(self, *arrays, condition, max_iterations, seeds, parameters=None):
        """Differentiate the executed loop iterations against final output seeds."""
        return _PROGRAM_CALL(self, "vjp_while", (condition, None, arrays, max_iterations, False, seeds, parameters))

    def while_loop(self, *arrays, condition, max_iterations, optimize=False, check_every=None, iteration=False):
        """Apply this state graph while the scalar condition holds, up to a limit.

        The loop runs on the device: the state arrays are uploaded once and
        updated in place, the condition is tested on the device, and the
        final state is downloaded once.

        Parameters
        ----------
        arrays : ndarray
            The initial state, one array per input of this graph (without
            the iteration input when ``iteration=True``).
        condition : Program
            A graph over the same inputs that returns one value; the loop
            continues while it is nonzero.
        max_iterations : int
            The most body applications, at most 16777216.
        optimize : bool
            Optimize the component graphs before running.
        check_every : int or None
            Iterations between two reads of the loop's status on the host.
            Larger values wait less on the device; up to ``check_every - 1``
            extra iterations may be computed after the stop and are
            discarded, including their writes to state tensors
            (:meth:`Session.state`), so the result is the same for every
            value. Above 1, the body must keep each state array's shape.
            None lets the loop choose from the body's cost.
        iteration : bool
            When true, this graph and the condition take one more input, a
            single value holding the number of iterations completed so far.

        Returns
        -------
        dict
            ``outputs`` (the final state), ``iterations`` (body applications)
            and ``stopped`` (whether the condition ended the loop).

        Examples
        --------
        >>> out = body.while_loop(x, condition=cond, max_iterations=100, check_every=8)
        >>> out["iterations"]
        """
        return _PROGRAM_CALL(self, "while_loop", (condition, None, arrays, max_iterations, optimize,
                                                  check_every, iteration))

    def decode(self, *arrays, n, state=None, unroll=None, iteration=False):
        """Generate ``n`` steps with this graph, on the device, in one call.

        The graph maps the current tokens (its first input, an ids input)
        to the next tokens, for example an embedding, the model's blocks and
        an argmax. It runs ``n`` times as a device loop: the tokens and the
        model's state tensors (:meth:`Session.state`) are updated in place,
        and the tokens of every step are recorded on the device and returned
        together.

        Parameters
        ----------
        arrays : ndarray
            The first tokens, one array per input of this graph (without the
            iteration input when ``iteration=True``).
        n : int
            The number of steps, at most 16777216.
        state : dict or None
            Values written into the session's state tensors before the first
            step, as ``{handle: array}``; None keeps the state they hold.
        unroll : int or None
            Steps enqueued between two reads of the loop's status on the
            host. The result is the same for every value. None lets the loop
            choose (a fixed step count reads almost only at the end).
        iteration : bool
            When true, this graph takes one more input, the number of steps
            completed so far (an integer when that input is an ids input).

        Returns
        -------
        dict
            ``tokens`` (one row per step: the tokens that step produced),
            ``outputs`` (the final value of every input) and ``iterations``.

        Examples
        --------
        >>> out = step.decode(np.array([[1]]), n=32)
        >>> out["tokens"].shape
        (32, 1)
        """
        return _PROGRAM_CALL(self, "decode", (arrays, n, unroll, iteration, state))

    def fit(self, *arrays, targets=None, optimizer, parameters=None, loss="mse",
            epochs=10, batch_size=32, shuffle=True, seed=0, huber_delta=1.0,
            tol=None, n_iter_no_change=5):
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
        tol : float or None
            Stopping criterion. When not None, training stops once the
            epoch loss has not fallen below the best loss so far minus
            ``tol`` for ``n_iter_no_change`` epochs in a row. None runs
            every epoch.
        n_iter_no_change : int
            Epochs without improvement before stopping, used with ``tol``.

        Returns
        -------
        Program
            This model, with ``loss_curve_`` holding this call's epoch
            losses and ``n_iter_`` the epochs run. Parameters and optimizer
            state stay in the session.

        Examples
        --------
        >>> opt = session.optimizer("adam", lr=0.01)
        >>> model.fit(X, targets=y, optimizer=opt, epochs=20)
        """
        return _PROGRAM_CALL(self, "fit", (None, None, arrays, targets,
            (optimizer, parameters, loss, epochs, batch_size, shuffle, seed, huber_delta, tol, n_iter_no_change)))

    def fit_if(self, condition, otherwise, *arrays, targets=None, optimizer,
               parameters=None, loss="mse", epochs=10, batch_size=32,
               shuffle=True, seed=0, huber_delta=1.0, tol=None, n_iter_no_change=5):
        """Train the branch selected by a scalar predicate for each minibatch.

        Components belong to one session and accept the same input ranks and
        kinds. The predicate is discrete. Shared parameters have one optimizer
        state; parameters absent from the selected branch are unchanged.
        Returns this program with epoch losses and per-call branch counts.
        ``tol`` and ``n_iter_no_change`` stop training early as in :meth:`fit`.
        """
        return _PROGRAM_CALL(self, "fit_if", (condition, otherwise, arrays, targets,
            (optimizer, parameters, loss, epochs, batch_size, shuffle, seed, huber_delta, tol, n_iter_no_change)))

    def partition(self, devices=(0,), *, stages=2, tensor_shards=1,
                  microbatches=2, optimizer, parameters=(), seed=0):
        """Partition captured operations and model state across native workers.

        ``stages`` and ``tensor_shards`` fix the logical graph; ``microbatches``
        fixes the pipeline and gradient order. Physical devices may change
        when restoring a checkpoint. Eligible matrix products split independent
        output columns; stage-owned parameters and moments stay on one device.
        """
        return _PROGRAM_CALL(self, "partition", (devices,
            (stages, tensor_shards, microbatches, optimizer, parameters, seed)))

    def distribute(self, devices=(0,), *, logical_shards=2, optimizer,
                   parameters=(), seed=0):
        """Copy this program into native workers with sharded optimizer state.

        Keep ``logical_shards`` unchanged when comparing physical topologies.
        ``optimizer`` is a session optimizer handle; its state is copied too.
        ``parameters`` lists trainable handles, or is empty to train every
        floating-point parameter. ``seed`` controls native row permutations.
        """
        return _PROGRAM_CALL(self, "distribute", (devices, (logical_shards, optimizer, parameters, seed)))

    def save_control(self, path, *, condition, otherwise=None, max_iterations=None,
                     optimizer=None, seed=0):
        """Save control components, shared parameters and optional optimizer state."""
        return _PROGRAM_CALL(self, "save_control", (path, condition, otherwise, max_iterations, optimizer, seed))

    def save(self, path, *, verify=True):
        """Write the program and its parameters' current values to ``path``,
        for :meth:`Session.load_program` on any install.

        With ``verify=True`` (the default), ``<path>.verify.json`` beside it
        records this install's report: the program run once on seeded
        inputs, with the digest of every input and output. On another
        install, ``python -m mojolearn verify-model <path>`` runs the saved
        program on the same inputs and checks that every output has the same
        bits. ``verify=False`` writes only the program file (and removes a
        report left by an earlier save to the same path), for checkpoints
        saved often during training.

        Example
        -------
        >>> program.save("model.mlp", verify=False)  # doctest: +SKIP
        """
        return _PROGRAM_CALL(self, "save", (path, verify))

    def free(self):
        """Release the program (its parameters stay session tensors)."""
        return _PROGRAM_CALL(self, "free", ())


# Import-time connection only; save never calls a Python layout/provenance helper.
from ._native_program_save_bridge import connect_program as _connect_program_save
_connect_program_save(Program, Session)
del _connect_program_save


# Distribution setup: resolve installed extension modules and CPU proxy
# attributes while importing this connector. Session construction and every
# later operation use cached native entries; no backend Python is called.
def _connect_session_mode(mode):
    return _connect_session_module(_backend.exact_host_module(_backend.binding("_mojolearn_x_sequence", mode)))


def _connect_session_module(module):
    # Import-time registration of common native provider/resource owner types.
    from .tokenizer import _binding
    _binding()
    # The exact compiled module, resolved once here: a session stores it as
    # `_b`, and native admission (`neural_descriptor_entry`, the device
    # provider's exact-module check) refuses the CPU `_HostBinding` facade
    # ("native provider requires an exact compiled module": every imported
    # program and Session on a CPU-only install, smoke 2026-11-15).
    distributed = _SimpleNamespace(
        ops_distributed_open=getattr(module, "ops_distributed_open", None),
        ops_distributed_load=getattr(module, "ops_distributed_load", None),
        ops_distributed_fit=getattr(module, "ops_distributed_fit", None),
        ops_distributed_run=getattr(module, "ops_distributed_run", None),
        ops_distributed_state=getattr(module, "ops_distributed_state", None),
        ops_distributed_residency=getattr(module, "ops_distributed_residency", None),
        ops_distributed_save=getattr(module, "ops_distributed_save", None),
        ops_distributed_close=getattr(module, "ops_distributed_close", None),
        ops_distributed_enter=getattr(module, "ops_distributed_enter", None),
        ops_distributed_exit=getattr(module, "ops_distributed_exit", None),
        ops_partitioned_open=getattr(module, "ops_partitioned_open", None),
        ops_partitioned_load=getattr(module, "ops_partitioned_load", None),
        ops_partitioned_fit=getattr(module, "ops_partitioned_fit", None),
        ops_partitioned_run=getattr(module, "ops_partitioned_run", None),
        ops_partitioned_state=getattr(module, "ops_partitioned_state", None),
        ops_partitioned_residency=getattr(module, "ops_partitioned_residency", None),
        ops_partitioned_save=getattr(module, "ops_partitioned_save", None),
        ops_partitioned_close=getattr(module, "ops_partitioned_close", None),
        ops_partitioned_enter=getattr(module, "ops_partitioned_enter", None),
        ops_partitioned_exit=getattr(module, "ops_partitioned_exit", None),
        ops_partitioned_profile=getattr(module, "ops_partitioned_profile", None),
    )
    result = _SimpleNamespace(
        binding=module, initialize=module.ops_session_initialize,
        numeric_mode=module.x_sequence_numeric_mode, open=module.ops_session_create,
        call=module.ops_session_call, execute=module.ops_program_execute,
        inspect=module.ops_program_inspect, control=module.ops_program_control,
        fit=module.ops_program_fit_native, distributed=distributed)
    from ._native_family_bridge import shared_context
    admission = shared_context()
    names = ("__dict__", "__getattribute__", "__getattr__", "__setattr__", "__delattr__")
    layout = tuple((name, vars(Session).get(name)) for name in names)  # glue: connect exact Session storage and attribute identities at import
    admission.update(ops_session_class=Session, ops_program_class=Program,
                     ops_connection_type=_SimpleNamespace, ops_session_layout=layout)
    result.native_module = module.model_native_api_v1.__self__
    result.public_provider = module.ops_session_public_connect(result.native_module)
    result.native_provider = result.initialize(None, None, None, result.native_module, None, admission)
    from ._native_program_save_bridge import connect_session
    connect_session(result)
    return result


_SESSION_CONNECTIONS = {}
# Both supported modes connect at import; a missing optional mode retains its
# exact refusal for the native initializer. The default remains selectable.
for _session_mode in ("identical", "fast"):
    try:
        _SESSION_CONNECTIONS[_session_mode] = _connect_session_mode(_session_mode)
    except (ImportError, ValueError, AttributeError) as _session_error:
        _SESSION_CONNECTIONS[_session_mode] = type(_session_error).__name__ + ": " + str(_session_error)

_session_connected = _SESSION_CONNECTIONS["identical"]
if isinstance(_session_connected, str):
    _session_connected = _SESSION_CONNECTIONS["fast"]
if isinstance(_session_connected, str):
    # Both connections failed. Preserve the required/default identical owner
    # error; an unavailable optional tier must not hide the missing owner.
    raise ImportError(_SESSION_CONNECTIONS["identical"])
_SESSION_INITIALIZE = _session_connected.initialize

# The CPU connection of `Session(device="cpu")`, connected once here beside
# the mode connections on an installation with a GPU set (the session
# initializer selects one connection per session, once, and every operation of
# the session and of its Programs runs through it). In a CPU-only process the
# mode connection already is the CPU one and nothing more is loaded. A host
# binary that is not built is kept as its refusal text, said when a
# `device="cpu"` session is opened.
if _backend.vendor() != "cpu":
    from ._native_family_bridge import shared_context as _session_shared_context
    try:
        _session_cpu = _connect_session_module(_backend.load_host_module("_mojolearn_x_sequence_host"))  # import-time: the CPU connection of device="cpu"
    except (ImportError, ValueError, AttributeError, TypeError, RuntimeError, OSError) as _session_error:
        _session_cpu = "ImportError: device='cpu' is not connected in this installation (" + type(_session_error).__name__ + ": " + str(_session_error) + ")"
    _session_shared_context()["ops_session_cpu_connection"] = _session_cpu
    del _session_shared_context, _session_cpu

_SESSION_CALL = _session_connected.call

_PROGRAM_CALL = _session_connected.binding.ops_program_public_call

from ._distributed_program import _connect_owned_programs
_connect_owned_programs(_SESSION_CONNECTIONS)
del _connect_owned_programs
