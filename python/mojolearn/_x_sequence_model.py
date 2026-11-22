# SPDX-License-Identifier: Apache-2.0
"""A feed-forward model you assemble layer by layer and train on the GPU."""
# Native model descriptions and numerical execution retain the existing
# Sequential ABI. Each public operation delegates one native call.
import os

from ._optional_numpy import require_numpy
np = require_numpy('_x_sequence_model')
from . import _backend
from ._mode import CheckMixin as _CheckMixin

# Description codes (autograd/model.mojo LAYER_*, LOSS_*, ...).
_LAYER = {"linear": 1, "relu": 2, "sigmoid": 3, "tanh": 4, "softmax": 5,
          "layer_norm": 6, "dropout": 7, "residual": 8,
          "reshape": 9, "permute": 10, "flatten": 11, "conv2d": 12, "max_pool2d": 13, "avg_pool2d": 14,
          "conv1d": 15, "max_pool1d": 16, "avg_pool1d": 17, "adaptive_avg_pool1d": 18, "branch": 19,
          "concat": 20, "residual_block": 0, "parallel": 0,
          "silu": 32, "gelu": 33, "softplus": 34, "rms_norm": 35, "rotary": 36, "multiply": 37,
          "narrow": 38, "embedding": 39, "attention": 40, "rnn": 41, "lstm": 41, "gru": 41, "gated": 0,
          "selective_scan": 42,
          "lm_head": 45}  # lm_head (lane neural-blocks-3-lm)
_LOSS = {"mse": 0, "cross_entropy": 1, "bce_with_logits": 2, "huber": 3, "token_cross_entropy": 4}
_TARGET_DENSE, _TARGET_CODES, _TARGET_CODE_VALUES = 0, 1, 2
_SCHED = {"constant": 0, "step": 1, "exponential": 2}
_OUT_RAW, _OUT_SOFTMAX, _OUT_SIGMOID, _OUT_PROBA2, _OUT_ARGMAX, _OUT_THRESHOLD = 0, 1, 2, 3, 4, 5
# sequence/ops.mojo OPT_*: "momentum" is SGD with a momentum buffer.
_OPT = {"sgd": 0, "momentum": 0, "adam": 1, "adamw": 2}
# core/cast_in.mojo CAST_KIND_* and item size -> the classes_ dtype of a
# loaded checkpoint
_CLASS_DTYPE = {(0, 1): "u1", (0, 2): "u2", (0, 4): "u4", (0, 8): "u8",
                (1, 1): "i1", (1, 2): "i2", (1, 4): "i4", (1, 8): "i8",
                (2, 2): "f2", (2, 4): "f4", (2, 8): "f8", (3, 1): "?"}


class Sequential(_CheckMixin):
    """A feed-forward model built layer by layer and trained on the GPU.

    Add layers with :meth:`add` (or the shortcuts :meth:`linear`,
    :meth:`relu`, ...), choose the loss and optimizer with :meth:`compile`,
    then call :meth:`fit`. Parameter initialization, shuffling, batching, the
    forward and backward passes and the optimizer all run natively on the
    device.

    In `identical` mode, mojolearn returns the same bits on Apple, NVIDIA and AMD GPUs.
    For the same data and seed, the trained parameters and the predictions
    of a ``Sequential`` model are the same on every vendor.

    Parameters
    ----------
    numeric_mode : {'identical', 'fast'} or None, default None
        Numeric mode. None uses the library default. ``'fast'`` is available
        on Apple GPUs only.
    input_shape : tuple of int or None, default None
        The shape of one sample, for example ``(channels, height, width)``.
        None takes it from ``X`` at fit. With an input shape, ``fit`` and
        ``predict`` also accept a 2-D ``X`` whose rows hold each sample's
        values in row-major order.

    A ``linear`` layer acts on the last axis of each sample. The model's
    output must be one vector per sample: end a model on images or
    sequences with :meth:`flatten` and a ``linear`` layer.

    Attributes
    ----------
    params_ : ndarray of float32
        All trained parameters as one flat vector, layers in order (a linear
        layer's weight ``(in, out)`` row-major, then its bias; a layer norm's
        weight, then its bias; a conv2d layer's weight
        ``(out_channels, in_channels, kernel_size, kernel_size)`` row-major,
        then its bias).
    loss_curve_ : list of float
        Mean training loss of every epoch.
    optimizer_m_ : ndarray of float32
        The optimizer's first-moment state after the last step, laid out as
        ``params_`` (the momentum buffer for ``'sgd'`` and ``'momentum'``, Adam's running
        mean of the gradient for ``'adam'`` and ``'adamw'``). Not set on a
        model loaded from an export.
    optimizer_v_ : ndarray of float32
        The optimizer's second-moment state, laid out as ``params_`` (Adam's
        running mean of the squared gradient). Not set on a model loaded
        from an export.
    n_iter_ : int
        Number of epochs run.
    n_features_in_ : int
        Number of input features.
    classes_ : ndarray
        Sorted class labels (``cross_entropy`` and ``bce_with_logits`` only).

    Examples
    --------
    >>> import numpy as np, mojolearn
    >>> rng = np.random.default_rng(0)
    >>> X = rng.random((256, 4), dtype=np.float32)
    >>> y = (X[:, 0] - 2 * X[:, 1]) + (0.5 * X[:, 2] + 3 * X[:, 3])
    >>> model = mojolearn.Sequential().linear(16).relu().linear(1)
    >>> model = model.compile(loss="mse", optimizer="adam", lr=0.01)
    >>> model = model.fit(X, y, epochs=20, batch_size=32, seed=0)
    >>> pred = model.predict(X)

    A classifier with class probabilities:

    >>> labels = (X[:, 0] > X[:, 1]).astype(np.int64)
    >>> clf = mojolearn.Sequential().linear(16).relu().linear(2)
    >>> clf = clf.compile(loss="cross_entropy", optimizer="adam", lr=0.01)
    >>> clf = clf.fit(X, labels, epochs=20, batch_size=32, seed=0)
    >>> proba = clf.predict_proba(X)
    """

    def __init__(self, numeric_mode=None, *, input_shape=None, device=None):
        return _SEQUENTIAL_ENTRY(self, "construct", (numeric_mode, input_shape, device), _SEQUENTIAL_CONTEXT)

    # ------------------------------------------------------------ building
    def add(self, kind, *args, **kwargs):
        """Append a layer and return the model.

        Parameters
        ----------
        kind : str
            One of ``'linear'`` (``units``, ``bias=True``), ``'relu'``,
            ``'sigmoid'``, ``'tanh'``, ``'softmax'``, ``'layer_norm'``
            (``eps=1e-5``), ``'residual'`` (``source=-1``: add the output
            of layer ``source``, or the model input for -1, to the current
            output), ``'reshape'`` (``shape``), ``'permute'`` (``axes``),
            ``'flatten'``, ``'conv2d'`` (``out_channels``, ``kernel_size``,
            ``stride=1``, ``padding=0``, ``bias=True``), ``'max_pool2d'``
            or ``'avg_pool2d'`` (``kernel_size``, ``stride=None``),
            ``'conv1d'``, ``'max_pool1d'``, ``'avg_pool1d'``,
            ``'adaptive_avg_pool1d'``, ``'branch'``, ``'concat'``,
            ``'residual_block'`` and ``'parallel'`` (the arguments of the
            methods of the same names).
        *args, **kwargs
            The layer's arguments, as listed for ``kind``.

        Returns
        -------
        Sequential
            The model, so calls chain.

        Examples
        --------
        >>> import mojolearn
        >>> model = mojolearn.Sequential().add("linear", 8).add("relu").add("linear", 1)
        """
        return _SEQUENTIAL_ENTRY(self, "add", (kind, args, kwargs), _SEQUENTIAL_CONTEXT)

    def _linear(self, units, bias=True):
        return _SEQUENTIAL_ENTRY(self, "add", ("linear", (units, bias), _SEQUENTIAL_NO_KEYWORDS), _SEQUENTIAL_CONTEXT)

    def _residual(self, source=-1):
        return _SEQUENTIAL_ENTRY(self, "add", ("residual", (source,), _SEQUENTIAL_NO_KEYWORDS), _SEQUENTIAL_CONTEXT)

    def linear(self, units, bias=True):
        """Append a fully connected layer with ``units`` outputs."""
        return _SEQUENTIAL_ENTRY(self, "add", ("linear", (units, bias), _SEQUENTIAL_NO_KEYWORDS), _SEQUENTIAL_CONTEXT)

    def relu(self):
        """Append a ReLU activation."""
        return _SEQUENTIAL_ENTRY(self, "add", ("relu", (), _SEQUENTIAL_NO_KEYWORDS), _SEQUENTIAL_CONTEXT)

    def sigmoid(self):
        """Append a logistic sigmoid activation."""
        return _SEQUENTIAL_ENTRY(self, "add", ("sigmoid", (), _SEQUENTIAL_NO_KEYWORDS), _SEQUENTIAL_CONTEXT)

    def tanh(self):
        """Append a hyperbolic tangent activation."""
        return _SEQUENTIAL_ENTRY(self, "add", ("tanh", (), _SEQUENTIAL_NO_KEYWORDS), _SEQUENTIAL_CONTEXT)

    def softmax(self):
        """Append a row-wise softmax."""
        return _SEQUENTIAL_ENTRY(self, "add", ("softmax", (), _SEQUENTIAL_NO_KEYWORDS), _SEQUENTIAL_CONTEXT)

    def layer_norm(self, eps=1e-5):
        """Append a layer normalization with a learned weight and bias."""
        return _SEQUENTIAL_ENTRY(self, "add", ("layer_norm", (eps,), _SEQUENTIAL_NO_KEYWORDS), _SEQUENTIAL_CONTEXT)

    def dropout(self, p):
        """Append dropout with drop probability ``p`` (active in fit only)."""
        return _SEQUENTIAL_ENTRY(self, "add", ("dropout", (p,), _SEQUENTIAL_NO_KEYWORDS), _SEQUENTIAL_CONTEXT)

    def residual(self, source=-1):
        """Add the output of layer ``source`` (-1: the model input) to the current output."""
        return _SEQUENTIAL_ENTRY(self, "add", ("residual", (source,), _SEQUENTIAL_NO_KEYWORDS), _SEQUENTIAL_CONTEXT)

    def reshape(self, shape):
        """Append a reshape of each sample to ``shape`` (up to 6
        dimensions, one of them -1 to infer it)."""
        return _SEQUENTIAL_ENTRY(self, "add", ("reshape", (shape,), _SEQUENTIAL_NO_KEYWORDS), _SEQUENTIAL_CONTEXT)

    def permute(self, axes):
        """Append a reordering of each sample's axes: output axis ``k`` is
        input axis ``axes[k]`` (axes count from 0 within a sample)."""
        return _SEQUENTIAL_ENTRY(self, "add", ("permute", (axes,), _SEQUENTIAL_NO_KEYWORDS), _SEQUENTIAL_CONTEXT)

    def flatten(self):
        """Append a flattening of each sample into one vector."""
        return _SEQUENTIAL_ENTRY(self, "add", ("flatten", (), _SEQUENTIAL_NO_KEYWORDS), _SEQUENTIAL_CONTEXT)

    def conv2d(self, out_channels, kernel_size, stride=1, padding=0, bias=True, groups=1):
        """Append a 2-D convolution of ``(channels, height, width)``
        samples with ``out_channels`` square ``kernel_size`` kernels,
        ``stride`` and zero ``padding`` on every side. With ``groups``
        greater than 1 the channels split into ``groups`` equal blocks and
        each block of outputs sees only its block of inputs (``groups``
        equal to the input channels is a depthwise convolution); the weight
        is ``(out_channels, channels // groups, kernel_size, kernel_size)``.

        Example::

            model.conv2d(32, 3, padding=1, groups=32)
        """
        return _SEQUENTIAL_ENTRY(self, "add", ("conv2d", (out_channels, kernel_size, stride, padding, bias, groups), _SEQUENTIAL_NO_KEYWORDS), _SEQUENTIAL_CONTEXT)

    def max_pool2d(self, kernel_size, stride=None):
        """Append a max pooling of ``(channels, height, width)`` samples
        over square ``kernel_size`` windows, ``stride`` apart (by default
        ``kernel_size``)."""
        return _SEQUENTIAL_ENTRY(self, "add", ("max_pool2d", (kernel_size, stride), _SEQUENTIAL_NO_KEYWORDS), _SEQUENTIAL_CONTEXT)

    def avg_pool2d(self, kernel_size, stride=None):
        """Append an average pooling of ``(channels, height, width)``
        samples over square ``kernel_size`` windows, ``stride`` apart (by
        default ``kernel_size``)."""
        return _SEQUENTIAL_ENTRY(self, "add", ("avg_pool2d", (kernel_size, stride), _SEQUENTIAL_NO_KEYWORDS), _SEQUENTIAL_CONTEXT)

    def conv1d(self, out_channels, kernel_size, stride=1, padding=0, bias=True, groups=1):
        """Append a 1-D convolution of ``(channels, length)`` samples with
        ``out_channels`` kernels of length ``kernel_size``, ``stride`` and
        ``padding`` zeros at both ends. With ``groups`` greater than 1 the
        channels split into ``groups`` equal blocks, each block of outputs
        seeing only its block of inputs; the weight is ``(out_channels,
        channels // groups, kernel_size)``. A ``kernel_size`` of 1 is a
        pointwise (bottleneck) convolution.

        Example::

            model.conv1d(32, 7, padding=3)
        """
        return _SEQUENTIAL_ENTRY(self, "add", ("conv1d", (out_channels, kernel_size, stride, padding, bias, groups), _SEQUENTIAL_NO_KEYWORDS), _SEQUENTIAL_CONTEXT)

    def max_pool1d(self, kernel_size, stride=None, padding=0):
        """Append a max pooling of ``(channels, length)`` samples over
        windows of ``kernel_size``, ``stride`` apart (by default
        ``kernel_size``), with ``padding`` positions at both ends that are
        never chosen (at most ``kernel_size // 2``)."""
        return _SEQUENTIAL_ENTRY(self, "add", ("max_pool1d", (kernel_size, stride, padding), _SEQUENTIAL_NO_KEYWORDS), _SEQUENTIAL_CONTEXT)

    def avg_pool1d(self, kernel_size, stride=None, padding=0):
        """Append an average pooling of ``(channels, length)`` samples over
        windows of ``kernel_size``, ``stride`` apart (by default
        ``kernel_size``), with ``padding`` zeros at both ends counted in each
        window (at most ``kernel_size // 2``)."""
        return _SEQUENTIAL_ENTRY(self, "add", ("avg_pool1d", (kernel_size, stride, padding), _SEQUENTIAL_NO_KEYWORDS), _SEQUENTIAL_CONTEXT)

    def adaptive_avg_pool1d(self, output_size=1):
        """Append an average pooling of ``(channels, length)`` samples to
        ``(channels, output_size)``: each output is the mean of one of
        ``output_size`` equal windows, so ``output_size`` must divide the
        length. ``output_size=1`` is the global average over the length."""
        return _SEQUENTIAL_ENTRY(self, "add", ("adaptive_avg_pool1d", (output_size,), _SEQUENTIAL_NO_KEYWORDS), _SEQUENTIAL_CONTEXT)

    def branch(self, source=-1):
        """Append a layer whose output is the output of layer ``source``
        (-1: the model input), so the layers after it start a new branch
        from that point."""
        return _SEQUENTIAL_ENTRY(self, "add", ("branch", (source,), _SEQUENTIAL_NO_KEYWORDS), _SEQUENTIAL_CONTEXT)

    def concat(self, sources, axis=0):
        """Append the outputs of the layers ``sources`` (1 to 6 layer
        indices, -1 for the model input) joined along sample axis ``axis``
        (0: the first axis, the channels; 1 joins the features of
        ``(length, features)`` samples). The other axes must agree."""
        return _SEQUENTIAL_ENTRY(self, "add", ("concat", (sources, axis), _SEQUENTIAL_NO_KEYWORDS), _SEQUENTIAL_CONTEXT)

    def residual_block(self, body, shortcut=None):
        """Append a residual block: ``body(h) + shortcut(h)``, where ``h`` is
        the current output and ``shortcut`` is the identity when None.

        Parameters
        ----------
        body : Sequential
            The layers of the main path, for example two convolutions.
        shortcut : Sequential or None, default None
            The layers applied to the block's input before the addition,
            typically a ``conv1d`` of kernel size 1 when the body changes
            the number of channels. Its output must have the body's shape.

        The layers of ``body`` and ``shortcut`` are copied into this model
        (they take their place in :attr:`layers` and in the parameter
        names); the two models themselves are not changed or trained.

        Examples
        --------
        >>> import mojolearn
        >>> body = mojolearn.Sequential().conv1d(16, 3, padding=1).relu().conv1d(16, 3, padding=1)
        >>> model = mojolearn.Sequential(input_shape=(4, 32)).conv1d(16, 1)
        >>> model = model.residual_block(body).relu()
        """
        return _SEQUENTIAL_ENTRY(self, "add", ("residual_block", (body, shortcut), _SEQUENTIAL_NO_KEYWORDS), _SEQUENTIAL_CONTEXT)

    def parallel(self, *branches):
        """Append parallel branches whose outputs are joined along the
        channels: each branch (a ``Sequential``) reads the current output,
        and the result is their outputs concatenated in order along the
        first axis of each sample. An empty branch passes its input
        through. 1 to 6 branches; their outputs must agree outside the
        channel axis.

        The branches' layers are copied into this model; the branch
        models themselves are not changed or trained.

        Examples
        --------
        >>> import mojolearn
        >>> a = mojolearn.Sequential().conv1d(8, 9, padding=4)
        >>> b = mojolearn.Sequential().conv1d(8, 5, padding=2)
        >>> c = mojolearn.Sequential().max_pool1d(3, stride=1, padding=1).conv1d(8, 1)
        >>> model = mojolearn.Sequential(input_shape=(4, 32)).parallel(a, b, c).relu()
        """
        return _SEQUENTIAL_ENTRY(self, "add", ("parallel", branches, _SEQUENTIAL_NO_KEYWORDS), _SEQUENTIAL_CONTEXT)

    def silu(self):
        """Append a SiLU (swish) activation, ``x * sigmoid(x)``."""
        return _SEQUENTIAL_ENTRY(self, "add", ("silu", (), _SEQUENTIAL_NO_KEYWORDS), _SEQUENTIAL_CONTEXT)

    def gelu(self, approximate="none"):
        """Append a GELU activation: ``x * Phi(x)`` with the normal CDF
        ``Phi`` (``approximate='none'``), or its tanh approximation
        (``approximate='tanh'``)."""
        return _SEQUENTIAL_ENTRY(self, "add", ("gelu", (approximate,), _SEQUENTIAL_NO_KEYWORDS), _SEQUENTIAL_CONTEXT)

    def softplus(self):
        """Append a softplus activation, ``log(1 + exp(x))``."""
        return _SEQUENTIAL_ENTRY(self, "add", ("softplus", (), _SEQUENTIAL_NO_KEYWORDS), _SEQUENTIAL_CONTEXT)

    def rms_norm(self, eps=1e-6):
        """Append a root-mean-square normalization of the last axis with a
        learned weight: ``x / sqrt(mean(x ** 2) + eps) * weight``.

        Example::

            model.rms_norm(1e-6)
        """
        return _SEQUENTIAL_ENTRY(self, "add", ("rms_norm", (eps,), _SEQUENTIAL_NO_KEYWORDS), _SEQUENTIAL_CONTEXT)

    def rotary(self, head_dim, theta=10000.0):
        """Append rotary position embeddings over ``(length, features)``
        samples: each block of ``head_dim`` features is rotated by angles of
        its position ``0 .. length - 1`` (the two halves of the block form
        the pairs). ``head_dim`` is even and divides the features."""
        return _SEQUENTIAL_ENTRY(self, "add", ("rotary", (head_dim, theta), _SEQUENTIAL_NO_KEYWORDS), _SEQUENTIAL_CONTEXT)

    def multiply(self, source=-1):
        """Multiply the current output elementwise by the output of layer
        ``source`` (-1: the model input); the shapes must agree."""
        return _SEQUENTIAL_ENTRY(self, "add", ("multiply", (source,), _SEQUENTIAL_NO_KEYWORDS), _SEQUENTIAL_CONTEXT)

    def narrow(self, axis, start, length):
        """Keep positions ``start .. start + length - 1`` of sample axis
        ``axis`` (axes count from 0 within a sample).

        Example: a causal depthwise convolution over a ``(channels, length)``
        sample of length ``L``::

            model.conv1d(64, 4, padding=3, groups=64).narrow(1, 0, L)
        """
        return _SEQUENTIAL_ENTRY(self, "add", ("narrow", (axis, start, length), _SEQUENTIAL_NO_KEYWORDS), _SEQUENTIAL_CONTEXT)

    def embedding(self, num_embeddings, embedding_dim):
        """Append an embedding lookup: each sample holds integer ids (stored
        as floats) in ``[0, num_embeddings)`` and each id becomes the
        ``embedding_dim`` values of its row in a learned table, so a
        ``(length,)`` sample becomes ``(length, embedding_dim)``."""
        return _SEQUENTIAL_ENTRY(self, "add", ("embedding", (num_embeddings, embedding_dim), _SEQUENTIAL_NO_KEYWORDS), _SEQUENTIAL_CONTEXT)

    def attention(self, heads, kv_heads=None, rotary=True, theta=10000.0, window=0, causal=True):
        """Append multi-head self-attention over ``(length, features)``
        samples: query, key, value and output projections (no bias), scaled
        dot-product attention per head, with rotary position embeddings when
        ``rotary`` is True.

        Parameters
        ----------
        heads : int
            Number of query heads; it divides the features.
        kv_heads : int or None, default None
            Number of key and value heads (grouped-query attention); it
            divides ``heads``. None uses ``heads``.
        rotary : bool, default True
            Rotate queries and keys by their positions.
        theta : float, default 10000.0
            Rotary base.
        window : int, default 0
            With ``window > 0`` each position attends only to the ``window``
            most recent positions, itself included (sliding-window
            attention).
        causal : bool, default True
            Each position attends only to itself and earlier positions.

        Example: a pre-norm Transformer layer::

            S = mojolearn.Sequential
            mlp = S().rms_norm().gated(S().linear(256, bias=False).silu(),
                                       S().linear(256, bias=False)).linear(64, bias=False)
            model = (S(input_shape=(32,)).embedding(512, 64)
                     .residual_block(S().rms_norm().attention(4))
                     .residual_block(mlp))
        """
        return _SEQUENTIAL_ENTRY(self, "add", ("attention", (heads, kv_heads, rotary, theta, window, causal), _SEQUENTIAL_NO_KEYWORDS), _SEQUENTIAL_CONTEXT)

    def lstm(self, hidden_size):
        """Append an LSTM layer over ``(length, features)`` samples from a
        zero state; the output is every step's hidden state,
        ``(length, hidden_size)``. Weights follow PyTorch's ``nn.LSTM``
        layout and gate order."""
        return _SEQUENTIAL_ENTRY(self, "add", ("lstm", (hidden_size,), _SEQUENTIAL_NO_KEYWORDS), _SEQUENTIAL_CONTEXT)

    def gru(self, hidden_size):
        """Append a GRU layer over ``(length, features)`` samples from a zero
        state; the output is every step's hidden state."""
        return _SEQUENTIAL_ENTRY(self, "add", ("gru", (hidden_size,), _SEQUENTIAL_NO_KEYWORDS), _SEQUENTIAL_CONTEXT)

    def rnn(self, hidden_size):
        """Append a tanh recurrent layer over ``(length, features)`` samples
        from a zero state; the output is every step's hidden state."""
        return _SEQUENTIAL_ENTRY(self, "add", ("rnn", (hidden_size,), _SEQUENTIAL_NO_KEYWORDS), _SEQUENTIAL_CONTEXT)

    def selective_scan(self, d_state=16):
        """Append Mamba's selective state-space scan over ``(length,
        features)`` samples whose features are ``[u | delta | B | C]``:
        ``dim`` input channels ``u``, their ``dim`` step sizes ``delta``
        (already positive, for example after ``softplus``), and the
        ``d_state`` input and output projections ``B`` and ``C`` of each
        position, so ``features = 2 * dim + 2 * d_state``. The output is
        ``(length, dim)``: ``y + u * D`` with ``y[t] = C[t] . h[t]`` and
        ``h[t] = exp(delta[t] * A) * h[t - 1] + delta[t] * B[t] * u[t]``
        from ``h = 0``, where ``A = -exp(A_log)``. ``A_log``
        ``(dim, d_state)`` and ``D`` ``(dim,)`` are learned, initialized as
        in Mamba (``A_log[:, n] = log(n + 1)``, ``D = 1``).

        Parameters
        ----------
        d_state : int, default 16
            State size per channel: 4, 8 or 16.

        Example: the core of a Mamba mixer over ``(length, 64)`` samples::

            # u, delta, b, c: indices of the layers giving the four parts
            model.concat([u, delta, b, c], axis=1).selective_scan(16)
        """
        return _SEQUENTIAL_ENTRY(self, "add", ("selective_scan", (d_state,), _SEQUENTIAL_NO_KEYWORDS), _SEQUENTIAL_CONTEXT)

    def lm_head(self, vocab):
        """Append a language-model head: the ``vocab`` logits of every
        position, ``h @ weight.T`` with a learned ``weight`` of shape
        ``(vocab, features)`` (no bias), so a ``(length, features)`` sample
        becomes ``(length, vocab)``. Train it with
        ``compile(loss="token_cross_entropy")`` and ``y`` the
        ``(n_samples, length)`` int32 next-token ids: the loss is the mean
        cross-entropy over every token, computed without forming the
        logits. As the last layer, ``predict`` returns every position's
        logits, ``(n_samples, length * vocab)`` in row-major order.

        Parameters
        ----------
        vocab : int
            Vocabulary size (at least 2).

        Example: a small causal language model over token ids::

            model = (mojolearn.Sequential().embedding(256, 64)
                     .attention(4).lm_head(256)
                     .compile(loss="token_cross_entropy", optimizer="adamw"))
        """
        return _SEQUENTIAL_ENTRY(self, "add", ("lm_head", (vocab,), _SEQUENTIAL_NO_KEYWORDS), _SEQUENTIAL_CONTEXT)

    def gated(self, gate, up):
        """Append a gated block: ``gate(h) * up(h)``, where ``h`` is the
        current output (a gated MLP: SwiGLU is ``gate`` a linear layer and
        :meth:`silu`, ``up`` a linear layer).

        Parameters
        ----------
        gate, up : Sequential
            The layers of the two paths; their outputs must have the same
            shape. Their layers are copied into this model; the two models
            themselves are not changed or trained.

        Examples
        --------
        >>> import mojolearn
        >>> gate = mojolearn.Sequential().linear(32, bias=False).silu()
        >>> up = mojolearn.Sequential().linear(32, bias=False)
        >>> model = mojolearn.Sequential(input_shape=(8, 16)).gated(gate, up).linear(16)
        """
        return _SEQUENTIAL_ENTRY(self, "add", ("gated", (gate, up), _SEQUENTIAL_NO_KEYWORDS), _SEQUENTIAL_CONTEXT)

    @property
    def layers(self):
        """The layers in order: ``(kind, arg0, arg1, float_arg)`` tuples,
        and ``(kind, arguments, float_arg)`` for ``reshape``, ``permute``,
        ``flatten``, ``conv2d`` and the pools, with ``arguments`` the
        layer's six integer arguments."""
        return _SEQUENTIAL_ENTRY(self, "layers", (), _SEQUENTIAL_CONTEXT)

    def compile(self, loss="mse", optimizer="adam", lr=1e-3, *, momentum=0.9, nesterov=False,
                beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.0, schedule="constant",
                step_size=1, gamma=0.1, huber_delta=1.0, tol=1e-4, n_iter_no_change=0):
        """Choose the loss, optimizer and learning-rate schedule.

        Parameters
        ----------
        loss : {'mse', 'cross_entropy', 'bce_with_logits', 'huber', 'token_cross_entropy'}, default 'mse'
            ``cross_entropy`` takes logits and integer class labels;
            ``bce_with_logits`` takes one logit per row and binary labels;
            ``token_cross_entropy`` trains a model ending in
            :meth:`lm_head` on ``(n_samples, length)`` int32 token ids.
        optimizer : {'sgd', 'momentum', 'adam', 'adamw'}, default 'adam'
            ``'momentum'`` is SGD with ``momentum``.
        lr : float, default 1e-3
            Learning rate (> 0).
        momentum : float, default 0.9
            Momentum of ``'momentum'``.
        nesterov : bool, default False
            Nesterov momentum for ``'momentum'``.
        beta1, beta2, eps : float
            Adam and AdamW moment decays and denominator term.
        weight_decay : float, default 0.0
            L2 penalty (sgd, momentum, adam) or decoupled decay (adamw).
        schedule : {'constant', 'step', 'exponential'}, default 'constant'
            ``'step'`` multiplies the rate by ``gamma`` every ``step_size``
            epochs; ``'exponential'`` by ``gamma`` every epoch.
        huber_delta : float, default 1.0
            Threshold of the Huber loss.
        tol, n_iter_no_change : float, int
            Stop when the epoch loss has not improved by more than ``tol``
            for ``n_iter_no_change`` epochs; 0 trains every epoch.

        Returns
        -------
        Sequential
            The model.

        Examples
        --------
        >>> import mojolearn
        >>> model = mojolearn.Sequential().linear(8).relu().linear(1)
        >>> model = model.compile(loss="huber", optimizer="momentum", lr=0.01, nesterov=True)
        """
        return _SEQUENTIAL_ENTRY(self, "compile", (loss, optimizer, lr, momentum, nesterov, beta1, beta2, eps, weight_decay, schedule, step_size, gamma, huber_delta, tol, n_iter_no_change), _SEQUENTIAL_CONTEXT)

    # ------------------------------------------------------------ description
    def _program(self, n_features, sample=None):
        return _SEQUENTIAL_ENTRY(self, "program", (n_features, sample), _SEQUENTIAL_CONTEXT)

    def _table(self):
        return _SEQUENTIAL_ENTRY(self, "table", (), _SEQUENTIAL_CONTEXT)


    # ------------------------------------------------------------ training
    def fit(self, X, y, *, epochs=10, batch_size=None, shuffle=None, seed=0, resume=False,
            validation_data=None, restore_best_weights=False):
        """Train the model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features) or (n_samples, ...)
            Training inputs: a bool, integer, float16, float32 or float64
            array, used as float32. A 2-D array may have any memory layout;
            an array of 3 or more dimensions (one sample per index of the
            first axis) must be C-contiguous.
        y : array-like
            Targets: (n_samples,) or (n_samples, n_outputs) for ``mse`` and
            ``huber``; one label per row for ``cross_entropy`` (any number
            of classes, one model output per class) and ``bce_with_logits``
            (two classes, one model output); the ``(n_samples, length)``
            int32 next-token ids for ``token_cross_entropy``.
        epochs : int, default 10
            Passes over the data (with ``resume=True``, passes added to
            the ones already run).
        batch_size : int or None, default None
            Rows per optimizer step. None uses 32, or the previous fit's
            batch size when resuming.
        shuffle : bool or None, default None
            Visit the rows in a new random order every epoch. None uses
            True, or the previous fit's choice when resuming.
        seed : int, default 0
            Seed of the parameter initialization and the row order. A
            resumed fit keeps the seed of the fit it continues.
        resume : bool, default False
            Continue the previous fit (or the training state loaded with
            :meth:`from_checkpoint`) instead of starting over: the
            parameters, the optimizer state, the epoch and step counters
            and the row order carry on, so ``fit(epochs=2)`` followed by
            ``fit(epochs=2, resume=True)`` gives the same parameters as
            ``fit(epochs=4)``. Pass the same training data.
        validation_data : tuple of (X, y) or None, default None
            Held-out data evaluated after each epoch with dropout disabled.
            When provided, early stopping monitors its loss. Monitoring
            starts afresh for each fit call.
        restore_best_weights : bool, default False
            With validation data, restore the best epoch's parameters,
            optimizer and counters together. ``n_iter_`` is the retained
            epoch; ``epochs_run_`` counts epochs executed in this call.
            ``best_epoch_`` names the best absolute epoch and
            ``validation_loss_curve_`` keeps every validation loss from
            this call. Resuming continues from the retained epoch.

        Returns
        -------
        Sequential
            The fitted model, with ``params_``, ``loss_curve_``, ``n_iter_``
            and ``n_features_in_`` set (and ``classes_`` for classifiers).
        """
        return _SEQUENTIAL_ENTRY(self, "fit", (X, y, epochs, batch_size, shuffle, seed, resume, validation_data, restore_best_weights), _SEQUENTIAL_CONTEXT)

    # ------------------------------------------------------------ weight import

    def load_state_dict(self, state, *, source=None, names=None, prefix="", strict=True):
        """Load trained weights from PyTorch or a safetensors checkpoint.

        Each layer with parameters takes the tensors ``<name>.weight`` and
        ``<name>.bias``, where ``<name>`` is the layer's index in the model
        (as in a ``torch.nn.Sequential`` of the same layers, activations
        included) with ``prefix`` in front. Weights use PyTorch's layouts:
        a ``linear`` weight is ``(out_features, in_features)``, a
        ``conv2d`` weight ``(out_channels, in_channels // groups,
        kernel_size, kernel_size)``, a ``layer_norm`` weight and bias ``(width,)``.
        float16, bfloat16, float32 and float64 tensors are accepted and
        converted to float32 on the device (float16 and bfloat16 exactly,
        float64 rounded to the nearest float32).

        After loading, :meth:`predict` uses the weights, and the next
        :meth:`fit` (without ``resume``) starts from them.

        Parameters
        ----------
        state : mapping or str or path-like
            A mapping of names to tensors, such as ``module.state_dict()``
            (tensors on the CPU, or NumPy arrays), or the path of a
            ``.safetensors`` file or of a checkpoint directory
            (``model.safetensors``, or ``model.safetensors.index.json`` and
            its shards).
        source : {'torch', 'safetensors'} or None, default None
            ``'torch'`` for a mapping, ``'safetensors'`` for a path. None
            takes ``'safetensors'`` for a path and ``'torch'`` otherwise.
        names : dict or None, default None
            ``{layer index: name}`` for layers whose tensors are named
            differently, for example ``{0: "fc1", 2: "fc2"}``.
        prefix : str, default ''
            Put in front of every name, for example ``"model."``.
        strict : bool, default True
            Refuse a source that holds tensors the model does not use.

        Returns
        -------
        Sequential
            The model, with ``params_``, ``n_features_in_`` and
            ``n_outputs_`` set.

        Raises
        ------
        ValueError
            A tensor is missing, unused (with ``strict``) or of the wrong
            shape, or the model needs ``input_shape`` to size its layers.

        Examples
        --------
        >>> import numpy as np, mojolearn
        >>> rng = np.random.default_rng(0)
        >>> state = {"0.weight": rng.standard_normal((8, 4), dtype=np.float32),
        ...          "0.bias": np.zeros(8, dtype=np.float32),
        ...          "2.weight": rng.standard_normal((1, 8), dtype=np.float32),
        ...          "2.bias": np.zeros(1, dtype=np.float32)}
        >>> model = mojolearn.Sequential().linear(8).relu().linear(1)
        >>> model = model.load_state_dict(state)
        >>> pred = model.predict(rng.random((16, 4), dtype=np.float32))
        """
        return _SEQUENTIAL_ENTRY(self, "load_weights", (state, source, names, prefix, strict), _SEQUENTIAL_CONTEXT)

    def state_dict(self):
        """The model's weights by name, as ``torch.nn.Sequential`` names them.

        Each layer with parameters gives ``"<i>.weight"`` and, when it has
        one, ``"<i>.bias"``, where ``<i>`` is the layer's index in the model
        (activations included). The arrays use PyTorch's layouts: a
        ``linear`` weight is ``(out_features, in_features)``, a ``conv2d``
        weight ``(out_channels, in_channels // groups, kernel_size,
        kernel_size)``. ``model.load_state_dict(model.state_dict())`` gives
        the same parameters back, and the mapping loads into the PyTorch
        module of the same layers.

        Returns
        -------
        dict of str to ndarray of float32
            A copy of the parameters, in layer order.

        Examples
        --------
        >>> import numpy as np, mojolearn
        >>> state = {"0.weight": np.ones((8, 4), dtype=np.float32), "0.bias": np.zeros(8, dtype=np.float32)}
        >>> model = mojolearn.Sequential().linear(8).load_state_dict(state)
        >>> model.state_dict()["0.weight"].shape
        (8, 4)
        """
        return _SEQUENTIAL_ENTRY(self, "export_weights", (0,), _SEQUENTIAL_CONTEXT)

    def parameters(self):
        """An iterator over the weight arrays, in :meth:`state_dict` order.

        Returns
        -------
        iterator of ndarray of float32
        """
        return _SEQUENTIAL_ENTRY(self, "export_weights", (1,), _SEQUENTIAL_CONTEXT)

    def named_parameters(self):
        """An iterator over ``(name, array)`` pairs, in :meth:`state_dict` order.

        Returns
        -------
        iterator of (str, ndarray of float32)
        """
        return _SEQUENTIAL_ENTRY(self, "export_weights", (2,), _SEQUENTIAL_CONTEXT)

    def to(self, *args, **kwargs):
        """Move the model to a device, as ``torch.nn.Module.to`` does.

        Parameters
        ----------
        device : str, optional
            ``"gpu"`` or ``"cpu"``, as the ``device`` constructor argument
            takes them.
        dtype : optional
            ``torch.float32``, NumPy's float32 or ``"float32"``. The model
            runs float32 only; any other dtype is refused.

        Returns
        -------
        Sequential
            The model itself, with its parameters, on the new device.
        """
        return _SEQUENTIAL_ENTRY(self, "to", (args, kwargs), _SEQUENTIAL_CONTEXT)

    # ------------------------------------------------------------ checkpoints
    def _save(self, path, export):
        return _SEQUENTIAL_ENTRY(self, "save", (path, export), _SEQUENTIAL_CONTEXT)

    def save_checkpoint(self, path):
        """Save the model and its full training state to ``path``.

        The file holds the layers, the loss, optimizer and schedule chosen
        with :meth:`compile`, the trained parameters, the optimizer state,
        the loss curve and the epoch and step counters of the row order and
        the optimizer. :meth:`from_checkpoint` restores all of it, and
        ``fit(..., resume=True)`` then continues training exactly where it
        stopped, on any install: a checkpoint written on one GPU vendor (or
        on the CPU) resumes to the same parameters on another.

        Beside it, ``<path>.verify.json`` records this install's report (the
        model's outputs on seeded inputs, by digest), for
        ``python -m mojolearn verify-model <path>`` on another install.

        Parameters
        ----------
        path : str or path-like
            Destination file.

        Examples
        --------
        >>> import numpy as np, mojolearn
        >>> rng = np.random.default_rng(0)
        >>> X = rng.random((256, 4), dtype=np.float32)
        >>> y = X @ np.array([1.0, -2.0, 0.5, 3.0], dtype=np.float32)
        >>> model = mojolearn.Sequential().linear(16).relu().linear(1)
        >>> model = model.compile(loss="mse", optimizer="adam", lr=0.01)
        >>> model = model.fit(X, y, epochs=2, batch_size=32, seed=0)
        >>> model.save_checkpoint("model.ckpt")  # doctest: +SKIP
        >>> again = mojolearn.Sequential.from_checkpoint("model.ckpt")  # doctest: +SKIP
        >>> again = again.fit(X, y, epochs=2, resume=True)  # doctest: +SKIP
        """
        return _SEQUENTIAL_ENTRY(self, "save", (path, False), _SEQUENTIAL_CONTEXT)

    def export(self, path):
        """Save the trained model for inference only.

        The file holds the layers and the trained parameters (and the
        classes of a classifier), without the optimizer state.
        :meth:`load` reads it back on any install, GPU or CPU, and
        :meth:`predict` and :meth:`predict_proba` give the same values
        there as here. Beside it, ``<path>.verify.json`` records this
        install's report, for ``python -m mojolearn verify-model <path>`` on
        another install.

        Parameters
        ----------
        path : str or path-like
            Destination file.

        Examples
        --------
        >>> import numpy as np, mojolearn
        >>> rng = np.random.default_rng(0)
        >>> X = rng.random((256, 4), dtype=np.float32)
        >>> labels = (X[:, 0] > X[:, 1]).astype(np.int64)
        >>> clf = mojolearn.Sequential().linear(16).relu().linear(2)
        >>> clf = clf.compile(loss="cross_entropy", optimizer="adam", lr=0.01)
        >>> clf = clf.fit(X, labels, epochs=5, batch_size=32, seed=0)
        >>> clf.export("classifier.ckpt")  # doctest: +SKIP
        >>> served = mojolearn.Sequential.load("classifier.ckpt")  # doctest: +SKIP
        >>> proba = served.predict_proba(X)  # doctest: +SKIP
        """
        return _SEQUENTIAL_ENTRY(self, "save", (path, True), _SEQUENTIAL_CONTEXT)

    @classmethod
    def from_checkpoint(cls, path, numeric_mode=None, device=None):
        """Load a model saved with :meth:`save_checkpoint` or :meth:`export`.

        Parameters
        ----------
        path : str or path-like
            A file written by :meth:`save_checkpoint` or :meth:`export`.
        numeric_mode : {'identical', 'fast'} or None, default None
            Numeric mode of the loaded model. None uses the library default.
            ``'fast'`` is available on Apple GPUs only.

        Returns
        -------
        Sequential
            The model, ready for :meth:`predict`; from a
            :meth:`save_checkpoint` file also ready for
            ``fit(..., resume=True)``.

        Raises
        ------
        ValueError
            The file is not a complete model checkpoint (wrong format,
            truncated or damaged).
        """
        return _SEQUENTIAL_ENTRY(cls, "load_checkpoint", (path, numeric_mode, device), _SEQUENTIAL_CONTEXT)

    @classmethod
    def load(cls, path, numeric_mode=None, device=None):
        """Load a model saved with :meth:`export` (or :meth:`save_checkpoint`).

        The same as :meth:`from_checkpoint`.

        Examples
        --------
        >>> import mojolearn
        >>> model = mojolearn.Sequential.load("classifier.ckpt")  # doctest: +SKIP
        """
        return _SEQUENTIAL_ENTRY(cls, "load_checkpoint", (path, numeric_mode, device), _SEQUENTIAL_CONTEXT)

    def _targets(self, y, n, n_out, loss):
        return _SEQUENTIAL_ENTRY(self, "targets", (y, n, n_out, loss), _SEQUENTIAL_CONTEXT)

    # ------------------------------------------------------------ prediction
    def _forward(self, X, transform, width=None):
        return _SEQUENTIAL_ENTRY(self, "forward", (X, transform, width), _SEQUENTIAL_CONTEXT)

    def vendor_used(self):
        """The accelerator API of the compiled binary this model runs on.

        Returns
        -------
        str
            ``'cpu'`` for a ``device="cpu"`` model, otherwise this
            installation's vendor (``'metal'``, ``'cuda'``, ``'hip'`` or
            ``'cpu'``).
        """
        return _SEQUENTIAL_ENTRY(self, "vendor_used", (), _SEQUENTIAL_CONTEXT)

    # `model.check(X)` (`_mode.CheckMixin`) answers X with this fitted model
    # on the GPU and on the CPU in one process. The two hooks below are what
    # the check needs from a class without `get_params` (`_check_both.on_device`);
    # only the check calls them, never fit or predict.
    _mojolearn_device = True

    def _mojolearn_device_twin(self, device):
        # glue (check only): the same unfitted model on `device`: the
        # constructor's arguments, then the layer list and the compile
        # options as the constructor's caller built them. The lists are the
        # twin's own, so adding a layer to one model never reaches the other.
        fields = object.__getattribute__(self, "__dict__")
        twin = type(self)(fields.get("numeric_mode"), input_shape=fields.get("input_shape"), device=device)
        target = object.__getattribute__(twin, "__dict__")
        for name in ("_layers", "_options"):  # glue: two description fields, not data
            value = fields.get(name)
            target[name] = type(value)(value) if type(value) in (list, dict) else value
        return twin

    def _mojolearn_device_state(self, twin, device):
        # glue (check only): hand the fitted state to the twin in memory. It
        # is every field but the three constructor arguments and the two
        # description fields the twin already holds: host arrays and scalars
        # (the parameters, the fit metadata, the optimizer state), shared,
        # not copied. The device route is not a field: the twin resolves its
        # own at its first operation.
        fields = object.__getattribute__(self, "__dict__")
        target = object.__getattribute__(twin, "__dict__")
        for name, value in fields.items():  # glue: over the model's fields, not data
            if name not in ("numeric_mode", "device", "input_shape", "_layers", "_options"):
                target[name] = value

    def predict(self, X):
        """Predict with the fitted model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Inputs: a bool, integer, float16, float32 or float64 array in
            any memory layout, used as float32.

        Returns
        -------
        ndarray
            The model outputs, (n_samples,) or (n_samples, n_outputs) in the
            shape ``y`` had at fit; for ``cross_entropy`` and
            ``bce_with_logits`` models, the predicted class labels
            (n_samples,).
        """
        return _SEQUENTIAL_ENTRY(self, "predict", (X,), _SEQUENTIAL_CONTEXT)

    def predict_proba(self, X):
        """Class probabilities of a ``cross_entropy`` or ``bce_with_logits`` model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Inputs: a bool, integer, float16, float32 or float64 array in
            any memory layout, used as float32.

        Returns
        -------
        ndarray of float32, shape (n_samples, n_classes)
            Each row sums to one; columns follow ``classes_``.
        """
        return _SEQUENTIAL_ENTRY(self, "predict_proba", (X,), _SEQUENTIAL_CONTEXT)


# Package-import ABI connection. No model operation calls this setup again.
from ._native_runtime_sequential_bridge import connect as _connect_sequential_owner
_SEQUENTIAL_NO_KEYWORDS = {}
_SEQUENTIAL_CONTEXT = {
    "integer": np.integer, "layer_codes": _LAYER, "loss_codes": _LOSS,
    "optimizer_codes": _OPT, "schedule_codes": _SCHED,
    "layer_names": ("", "linear", "relu", "sigmoid", "tanh", "softmax",
                    "layer_norm", "dropout", "residual", "reshape", "permute",
                    "flatten", "conv2d", "max_pool2d", "avg_pool2d", "conv1d", "max_pool1d",
                    "avg_pool1d", "adaptive_avg_pool1d", "branch", "concat",
                    "", "", "", "", "", "", "", "", "", "", "", "silu", "gelu", "softplus", "rms_norm",
                    "rotary", "multiply", "narrow", "embedding", "attention", "recurrent",
                    "selective_scan", "", "", "lm_head"),
}


def _connect_sequential_bindings():
    # ABI connection at module import only; no Sequential operation calls it.
    from types import BuiltinFunctionType, ModuleType
    from collections import OrderedDict, defaultdict
    from pathlib import PosixPath, PurePosixPath
    import ctypes
    import mmap
    import sys
    from array import array
    from ._array import Array, SUPPORTED_DTYPES, _ARRAY_CONTEXT
    from ._version import __version__
    selected = _backend.default_mode()
    entries = {}
    names = ("x_sequence_vendor", "x_sequence_numeric_mode", "model_native_api_v1", "model_n_params", "model_fit", "model_predict", "model_labels", "model_gather_labels", "model_predict_labels", "model_fit_labels", "model_import_params", "model_save_checkpoint", "model_load_checkpoint", "model_checkpoint_info", "verify_digest", "verify_file_sha256", "verify_model_info", "verify_fill")

    def connected(module):
        # glue: resolve existing compiled ABI symbols at import
        symbols = {}
        for name in names:  # glue: resolve existing compiled ABI symbols at import
            entry = getattr(module, name)
            if not isinstance(entry, BuiltinFunctionType):
                raise TypeError(f"Sequential requires a compiled {name}")
            symbols[name] = entry
        return symbols

    errors = (ImportError, AttributeError, ValueError, RuntimeError)
    for mode in (selected, "fast" if selected == "identical" else "identical"):  # glue: connect native modes at package import
        try:
            entries[mode] = connected(_backend.binding("_mojolearn_x_sequence", mode))
        except errors as error:
            entries[mode] = type(error).__name__ + ": " + str(error)
    # The CPU table of `device="cpu"`, connected once here beside the device
    # table (bindings/sequential_route_door.mojo resolves a model against one
    # of the two, once per model). In a CPU-only process the selected table
    # already is the CPU table and nothing more is loaded. A host binary that
    # is not built is kept as its refusal text, said when a `device="cpu"`
    # model first runs.
    cpu_entries = None
    if _backend.vendor() != "cpu":
        try:
            cpu = connected(_backend.load_host_module("_mojolearn_x_sequence_host"))  # import-time: the CPU table of device="cpu"
        except errors + (TypeError, OSError) as error:
            cpu = type(error).__name__ + ": " + str(error)
        cpu_entries = {"identical": cpu}
    class_dtypes = {key: np.dtype(value) for key, value in _CLASS_DTYPE.items()}  # glue: connect checkpoint dtype classes once at package import
    numeric_dtypes = {("uifb"[kind], size): dtype for (kind, size), dtype in class_dtypes.items()}  # glue: connect native numeric dtype metadata once at package import
    buffer_codes = {("b", 1): "b", ("i", 1): "b", ("i", 2): "h", ("i", 4): "i", ("i", 8): "q", ("u", 1): "B", ("u", 2): "H", ("u", 4): "I", ("u", 8): "Q", ("f", 2): "H", ("f", 4): "f", ("f", 8): "d"}
    scalar_types = tuple(dtype.type for dtype in class_dtypes.values())  # glue: connect exact native scalar types once at package import
    integer_types = (int,) + tuple(dtype.type for (kind, _), dtype in class_dtypes.items() if kind in (0, 1))  # glue: connect exact native integer types once at package import
    array_fields = {name: vars(Array)[name].__get__ for name in ("_mv", "shape", "dtype", "order")}  # glue: retain native slot descriptors once at package import
    array_storage_slots = _ARRAY_CONTEXT["from_buffer_slots"]  # glue: reuse the complete native storage descriptor inventory retained at import
    return {
        "array_storage_slots": array_storage_slots, "array_storage_dtypes": SUPPORTED_DTYPES,
        "sequential_class": Sequential, "module_type": ModuleType, "modules": sys.modules,
        "mapping_types": (dict, OrderedDict, defaultdict), "scalar_types": scalar_types, "integer_types": integer_types,
        "native_bytes_type": bytes, "native_bytearray_type": bytearray,
        "defaultdict_class": defaultdict, "native_default_factories": (list, tuple, dict, OrderedDict, defaultdict, bool, int, float, str, bytes, bytearray),
        "native_value_types": (type(None), str, bytes, bytearray, bool, int, float, range, np.ndarray, Array, array, PosixPath, PurePosixPath, type(iter([])), type(iter(())), type(iter(range(0))), type(reversed([]))),
        "buffer_owner_types": (type(None), bytes, bytearray, array, np.ndarray, mmap.mmap), "ctypes_array_base": ctypes.Array, "array_fields": array_fields,
        "model_fields": ("numeric_mode", "device", "input_shape", "_layers", "_options", "params_", "n_features_in_", "n_outputs_", "_sample_shape", "n_iter_", "loss_curve_", "_fit_state", "_imported", "_fit_loss", "_target", "_fit_batch", "_fit_shuffle", "_y_1d", "classes_", "optimizer_m_", "optimizer_v_", "_curve", "validation_loss_curve_", "epochs_run_", "best_epoch_"),
        "numeric_dtypes": numeric_dtypes, "buffer_codes": buffer_codes, "scalar_type": np.generic,
        "version": __version__, "uname": os.uname, "class_dtypes": class_dtypes,
        "loss_names": ("mse", "cross_entropy", "bce_with_logits", "huber", "token_cross_entropy"),
        "bindings": entries, "sequential_cpu_bindings": cpu_entries,
        "backend_fields": vars(_backend), "vendor": _backend.vendor(),
        "ndarray": np.ndarray, "array_type": array, "array_class": Array,
        "float_types": (float, np.floating), "int_types": (int, np.integer),
        "bool_types": (bool, np.bool_), "int64_min": -(2 ** 63),
        "int64_max": 2 ** 63 - 1, "uint64_max": 2 ** 64 - 1,
        "float32": np.dtype("f4"), "float64": np.dtype("f8"),
        "int32": np.dtype("i4"), "int64": np.dtype("i8"),
        "uint64": np.dtype("u8"), "bool": np.dtype("?"),
    }


_SEQUENTIAL_CONTEXT.update(_connect_sequential_bindings())
_SEQUENTIAL_ENTRY = _connect_sequential_owner(_SEQUENTIAL_CONTEXT)
_SEQUENTIAL_ENTRY(None, "initialize", (), _SEQUENTIAL_CONTEXT)
