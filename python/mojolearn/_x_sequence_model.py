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
          "selective_scan": 42, "ssd_scan": 43, "trapezoid_scan": 44,
          "lm_head": 45,
          # nn-attention begin: autograd/nn_attention.mojo; the encoder and decoder layers and stacks are
          # containers (code 0) written from existing layers by the description door
          "multihead_attention": 67, "positional_encoding": 72,
          "transformer_encoder_layer": 0, "transformer_decoder_layer": 0,
          "transformer_encoder": 0, "transformer_decoder": 0,
          # nn-attention end
          "batch_norm1d": 46, "batch_norm2d": 46,  # lm_head (lane neural-blocks-3-lm)
          # nn-conv begin
          "conv_transpose1d": 56, "conv_transpose2d": 57, "adaptive_avg_pool2d": 58,
          "adaptive_max_pool2d": 59, "adaptive_max_pool1d": 60, "global_max_pool": 61,
          "zero_pad1d": 62, "zero_pad2d": 62, "constant_pad1d": 62, "constant_pad2d": 62,
          "reflection_pad1d": 63, "reflection_pad2d": 63, "replication_pad1d": 64, "replication_pad2d": 64,
          "upsample": 65, "pixel_shuffle": 66, "pixel_unshuffle": 66,
          # nn-conv end
          # nn-activations begin
          "identity": 47, "leaky_relu": 48, "elu": 49, "prelu": 50, "mish": 51, "hardswish": 52,
          "hardsigmoid": 53,
          # nn-activations end
          # nn-norms begin
          "group_norm": 54, "instance_norm1d": 55, "instance_norm2d": 55,
          # nn-norms end
          # nn-losses-2 begin
          "log_softmax": 77,
          # nn-losses-2 end
          # nn-conv3d begin
          "conv3d": 73, "max_pool3d": 74, "avg_pool3d": 75, "adaptive_avg_pool3d": 76,
          # nn-conv3d end
          }
_LOSS = {"mse": 0, "cross_entropy": 1, "bce_with_logits": 2, "huber": 3, "token_cross_entropy": 4,
         # nn-losses begin
         "l1": 5, "smooth_l1": 6, "nll": 7, "kl_div": 8,
         # nn-losses-2 begin
         "ctc": 9,
         # nn-losses-2 end
         }
_REDUCTION = {"mean": 0, "sum": 1, "batchmean": 2}
# nn-losses end
_TARGET_DENSE, _TARGET_CODES, _TARGET_CODE_VALUES = 0, 1, 2
_SCHED = {"constant": 0, "step": 1, "exponential": 2,
          # lane nn-optim begin (autograd/nn_optim.mojo NN_SCHED_*)
          "multi_step": 3, "cosine": 4, "cosine_warm_restarts": 5, "linear_warmup": 6, "one_cycle": 7,
          "reduce_on_plateau": 8,
          # lane nn-optim end
          }
_OUT_RAW, _OUT_SOFTMAX, _OUT_SIGMOID, _OUT_PROBA2, _OUT_ARGMAX, _OUT_THRESHOLD = 0, 1, 2, 3, 4, 5
# sequence/ops.mojo OPT_*: "momentum" is SGD with a momentum buffer.
_OPT = {"sgd": 0, "momentum": 0, "adam": 1, "adamw": 2,
        # lane nn-optim begin (autograd/nn_optim.mojo NN_OPT_*)
        "rmsprop": 3, "adagrad": 4, "lamb": 10,
        # lane nn-optim end
        }
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

    In `identical` mode, Mojolearn returns the same bits on Apple, NVIDIA and AMD GPUs.
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
        Sorted class labels (``cross_entropy``, ``nll`` and ``bce_with_logits`` only).

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
            (``eps=1e-5``), ``'dropout'`` (``p``), ``'residual'``
            (``source=-1``: add the output
            of layer ``source``, or the model input for -1, to the current
            output), ``'reshape'`` (``shape``), ``'permute'`` (``axes``),
            ``'flatten'``, ``'conv2d'`` (``out_channels``, ``kernel_size``,
            ``stride=1``, ``padding=0``, ``bias=True``), ``'max_pool2d'``
            or ``'avg_pool2d'`` (``kernel_size``, ``stride=None``),
            or the name of any other layer method of this class
            (``'conv1d'``, ``'max_pool1d'``, ``'avg_pool1d'``,
            ``'adaptive_avg_pool1d'``, ``'branch'``, ``'concat'``,
            ``'residual_block'``, ``'parallel'``, ``'gated'``, ``'silu'``,
            ``'gelu'``, ``'softplus'``, ``'rms_norm'``, ``'rotary'``,
            ``'multiply'``, ``'narrow'``, ``'embedding'``, ``'attention'``,
            ``'lstm'``, ``'gru'``, ``'rnn'``, ``'batch_norm1d'``,
            ``'batch_norm2d'``, ``'selective_scan'``, ``'ssd_scan'``,
            ``'trapezoid_scan'``, ``'lm_head'``,
            ``'multihead_attention'``, ``'positional_encoding'``,
            ``'transformer_encoder_layer'``, ``'transformer_decoder_layer'``,
            ``'transformer_encoder'``, ``'transformer_decoder'``,
            ``'conv_transpose1d'``, ``'conv_transpose2d'``,
            ``'adaptive_avg_pool2d'``, ``'adaptive_max_pool2d'``,
            ``'adaptive_max_pool1d'``, ``'global_max_pool'``, ``'zero_pad1d'``,
            ``'zero_pad2d'``, ``'constant_pad1d'``, ``'constant_pad2d'``,
            ``'reflection_pad1d'``, ``'reflection_pad2d'``,
            ``'replication_pad1d'``, ``'replication_pad2d'``, ``'upsample'``,
            ``'pixel_shuffle'``, ``'pixel_unshuffle'``, ``'identity'``,
            ``'leaky_relu'``, ``'elu'``, ``'prelu'``, ``'mish'``,
            ``'hardswish'``, ``'hardsigmoid'``, ``'group_norm'``,
            ``'instance_norm1d'``, ``'instance_norm2d'``,
            ``'log_softmax'``, ``'conv3d'``, ``'max_pool3d'``, ``'avg_pool3d'``,
            ``'adaptive_avg_pool3d'``), with that
            method's arguments.
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

    # nn-losses-2 begin
    def log_softmax(self, dim=-1):
        """Append a log-softmax (PyTorch's ``nn.LogSoftmax``):
        ``x - log(sum(exp(x)))`` along ``dim``, computed stably (the
        maximum is subtracted before the exponentials).

        Parameters
        ----------
        dim : int, default -1
            The axis, counted as in PyTorch with the batch axis as axis 0:
            ``-1`` is the last axis of each sample, ``1`` the first. The
            batch axis itself cannot be normalized.

        Its output is log-probabilities, the input of the ``nll``,
        ``kl_div`` and ``ctc`` losses (:meth:`compile`). A model whose
        samples are ``(length, classes)`` and that ends in
        ``log_softmax()`` gives each step's log-probabilities, the input
        of ``ctc``; its predictions are those ``length * classes`` values
        of each sample, flat.

        Example::

            model = mojolearn.Sequential().linear(16).relu().linear(3).log_softmax()
            model.compile(loss="nll")
        """
        return _SEQUENTIAL_ENTRY(self, "add", ("log_softmax", (dim,), _SEQUENTIAL_NO_KEYWORDS), _SEQUENTIAL_CONTEXT)
    # nn-losses-2 end

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
        never chosen (at most ``kernel_size // 2``), as PyTorch's
        ``nn.MaxPool1d``.

        Example::

            model.conv1d(16, 3, padding=1).relu().max_pool1d(2)
        """
        return _SEQUENTIAL_ENTRY(self, "add", ("max_pool1d", (kernel_size, stride, padding), _SEQUENTIAL_NO_KEYWORDS), _SEQUENTIAL_CONTEXT)

    def avg_pool1d(self, kernel_size, stride=None, padding=0):
        """Append an average pooling of ``(channels, length)`` samples over
        windows of ``kernel_size``, ``stride`` apart (by default
        ``kernel_size``), with ``padding`` zeros at both ends counted in each
        window (at most ``kernel_size // 2``), as PyTorch's
        ``nn.AvgPool1d`` with ``count_include_pad=True``.

        Example::

            model.avg_pool1d(3, stride=1, padding=1)
        """
        return _SEQUENTIAL_ENTRY(self, "add", ("avg_pool1d", (kernel_size, stride, padding), _SEQUENTIAL_NO_KEYWORDS), _SEQUENTIAL_CONTEXT)

    def adaptive_avg_pool1d(self, output_size=1):
        """Append an average pooling of ``(channels, length)`` samples to
        ``(channels, output_size)``: each output is the mean of one of
        ``output_size`` equal windows, so ``output_size`` must divide the
        length. ``output_size=1`` is the global average over the length.

        Example: global average pooling before a classifier head::

            model.conv1d(32, 3, padding=1).relu().adaptive_avg_pool1d(1).flatten().linear(2)
        """
        return _SEQUENTIAL_ENTRY(self, "add", ("adaptive_avg_pool1d", (output_size,), _SEQUENTIAL_NO_KEYWORDS), _SEQUENTIAL_CONTEXT)

    # nn-conv3d begin
    def conv3d(self, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
        """Append a 3-D convolution of ``(channels, depth, height, width)``
        samples, as PyTorch's ``nn.Conv3d`` with zero padding.

        Parameters
        ----------
        out_channels : int
            The number of output channels.
        kernel_size : int or tuple of 3 ints
            The window along depth, height and width.
        stride : int or tuple of 3 ints, default 1
            The step between windows.
        padding : int, tuple of 3 ints or 'valid', default 0
            Zeros added at both ends of each axis.
        dilation : int or tuple of 3 ints, default 1
            The spacing between kernel taps.
        groups : int, default 1
            With ``groups`` greater than 1 the channels split into
            ``groups`` equal blocks and each block of outputs sees only its
            block of inputs; it divides the input channels and
            ``out_channels``.
        bias : bool, default True
            Add a learned per-channel bias.

        The weight is ``(out_channels, channels // groups, kd, kh, kw)``
        and, as in PyTorch, the weight and bias start uniform in
        ``[-1/sqrt(fan_in), 1/sqrt(fan_in)]`` with ``fan_in = channels //
        groups * kd * kh * kw``.

        Example::

            model = (mojolearn.Sequential(input_shape=(1, 16, 16, 16))
                     .conv3d(8, 3, padding=1).relu().max_pool3d(2)
                     .adaptive_avg_pool3d(1).flatten().linear(2))
        """
        return _SEQUENTIAL_ENTRY(self, "add", ("conv3d", (out_channels, kernel_size, stride, padding, dilation, groups, bias), _SEQUENTIAL_NO_KEYWORDS), _SEQUENTIAL_CONTEXT)

    def max_pool3d(self, kernel_size, stride=None, padding=0, dilation=1, ceil_mode=False):
        """Append a max pooling of ``(channels, depth, height, width)``
        samples, as PyTorch's ``nn.MaxPool3d``.

        Each output is the largest value of its window; padded positions
        are never chosen, and of tied values the first in (depth, height,
        width) order receives the gradient.

        Parameters
        ----------
        kernel_size : int or tuple of 3 ints
            The window along depth, height and width.
        stride : int, tuple of 3 ints or None, default None
            The step between windows; None is ``kernel_size``.
        padding : int or tuple of 3 ints, default 0
            Positions added at both ends of each axis, at most half of the
            window.
        dilation : int or tuple of 3 ints, default 1
            The spacing between window taps.
        ceil_mode : bool, default False
            Round the output sizes up instead of down.

        Example::

            model.max_pool3d(2)
        """
        return _SEQUENTIAL_ENTRY(self, "add", ("max_pool3d", (kernel_size, stride, padding, dilation, ceil_mode), _SEQUENTIAL_NO_KEYWORDS), _SEQUENTIAL_CONTEXT)

    def avg_pool3d(self, kernel_size, stride=None, padding=0, ceil_mode=False, count_include_pad=True,
                   divisor_override=None):
        """Append an average pooling of ``(channels, depth, height, width)``
        samples, as PyTorch's ``nn.AvgPool3d``.

        Parameters
        ----------
        kernel_size : int or tuple of 3 ints
            The window along depth, height and width.
        stride : int, tuple of 3 ints or None, default None
            The step between windows; None is ``kernel_size``.
        padding : int or tuple of 3 ints, default 0
            Zeros added at both ends of each axis, at most half of the
            window.
        ceil_mode : bool, default False
            Round the output sizes up instead of down.
        count_include_pad : bool, default True
            Count the zero padding in each window's size.
        divisor_override : int or None, default None
            When given, the divisor of every window.

        Example::

            model.avg_pool3d(2, stride=2)
        """
        return _SEQUENTIAL_ENTRY(self, "add", ("avg_pool3d", (kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override), _SEQUENTIAL_NO_KEYWORDS), _SEQUENTIAL_CONTEXT)

    def adaptive_avg_pool3d(self, output_size):
        """Append an average pooling of ``(channels, depth, height, width)``
        samples to ``(channels, *output_size)``, as PyTorch's
        ``nn.AdaptiveAvgPool3d``: ``output_size=1`` is the global average
        over each volume, and output sizes that divide the input's make each
        output the mean of one equal tile. Other output sizes are refused.

        Parameters
        ----------
        output_size : int or tuple of 3 ints
            The output depth, height and width.

        Example::

            model.adaptive_avg_pool3d(1)
        """
        return _SEQUENTIAL_ENTRY(self, "add", ("adaptive_avg_pool3d", (output_size,), _SEQUENTIAL_NO_KEYWORDS), _SEQUENTIAL_CONTEXT)
    # nn-conv3d end

    def branch(self, source=-1):
        """Append a layer whose output is the output of layer ``source``
        (-1: the model input), so the layers after it start a new branch
        from that point.

        Example: two heads read the same features (layer 1)::

            model.linear(32).relu().linear(8).branch(1).linear(4).concat([2, 4])
        """
        return _SEQUENTIAL_ENTRY(self, "add", ("branch", (source,), _SEQUENTIAL_NO_KEYWORDS), _SEQUENTIAL_CONTEXT)

    def concat(self, sources, axis=0):
        """Append the outputs of the layers ``sources`` (1 to 6 layer
        indices, -1 for the model input) joined along sample axis ``axis``
        (0: the first axis, the channels; 1 joins the features of
        ``(length, features)`` samples). The other axes must agree.

        Example: join the outputs of layers 0 and 2 along the channels::

            model.conv1d(8, 3, padding=1).relu().conv1d(8, 5, padding=2).concat([0, 2])
        """
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
        """Append a SiLU (swish) activation, ``x * sigmoid(x)``, as PyTorch's
        ``nn.SiLU``.

        Example::

            model.linear(64).silu()
        """
        return _SEQUENTIAL_ENTRY(self, "add", ("silu", (), _SEQUENTIAL_NO_KEYWORDS), _SEQUENTIAL_CONTEXT)

    def gelu(self, approximate="none"):
        """Append a GELU activation: ``x * Phi(x)`` with the normal CDF
        ``Phi`` (``approximate='none'``), or its tanh approximation
        (``approximate='tanh'``), as PyTorch's ``nn.GELU``.

        Example::

            model.linear(256).gelu(approximate="tanh")
        """
        return _SEQUENTIAL_ENTRY(self, "add", ("gelu", (approximate,), _SEQUENTIAL_NO_KEYWORDS), _SEQUENTIAL_CONTEXT)

    def softplus(self):
        """Append a softplus activation, ``log(1 + exp(x))``.

        Example: positive step sizes for a state-space scan::

            model.linear(64).softplus()
        """
        return _SEQUENTIAL_ENTRY(self, "add", ("softplus", (), _SEQUENTIAL_NO_KEYWORDS), _SEQUENTIAL_CONTEXT)

    # nn-activations begin
    def identity(self):
        """Append an identity layer, which returns its input unchanged
        (PyTorch's ``nn.Identity``).

        It is a placeholder: put it where a module may later be switched off,
        or swapped for another, without changing the model's structure (the
        layer indices, and so the names in :meth:`state_dict`, stay the same).
        It has no parameters and no computation runs for it.

        Example::

            model = mojolearn.Sequential().linear(16).relu().identity().linear(1)
        """
        return _SEQUENTIAL_ENTRY(self, "add", ("identity", (), _SEQUENTIAL_NO_KEYWORDS), _SEQUENTIAL_CONTEXT)

    def leaky_relu(self, negative_slope=0.01):
        """Append a leaky ReLU activation (PyTorch's ``nn.LeakyReLU``):
        ``x`` where ``x >= 0``, else ``negative_slope * x``.

        Parameters
        ----------
        negative_slope : float, default 0.01
            The slope for negative inputs.

        Example::

            model = mojolearn.Sequential().linear(16).leaky_relu(0.1).linear(1)
        """
        return _SEQUENTIAL_ENTRY(self, "add", ("leaky_relu", (negative_slope,), _SEQUENTIAL_NO_KEYWORDS), _SEQUENTIAL_CONTEXT)

    def elu(self, alpha=1.0):
        """Append an ELU activation (PyTorch's ``nn.ELU``): ``x`` where
        ``x >= 0``, else ``alpha * (exp(x) - 1)``.

        Parameters
        ----------
        alpha : float, default 1.0
            The value the output approaches for large negative inputs is
            ``-alpha``.

        Example::

            model = mojolearn.Sequential().linear(16).elu().linear(1)
        """
        return _SEQUENTIAL_ENTRY(self, "add", ("elu", (alpha,), _SEQUENTIAL_NO_KEYWORDS), _SEQUENTIAL_CONTEXT)

    def prelu(self, num_parameters=1, init=0.25):
        """Append a PReLU activation with a learned slope (PyTorch's
        ``nn.PReLU``): ``x`` where ``x >= 0``, else ``weight * x``.

        Parameters
        ----------
        num_parameters : int, default 1
            1 shares one slope across every element. Otherwise it must equal
            the size of the first axis of each sample (the channels of
            ``(channels, ...)`` samples, the features of ``(features,)``
            samples), and each channel learns its own slope.
        init : float, default 0.25
            The initial value of every slope; it must be greater than -1.

        The slopes are a parameter tensor ``weight`` of shape
        ``(num_parameters,)`` in :attr:`params_` and :meth:`state_dict`, and
        are trained with the other parameters.

        Example::

            model = mojolearn.Sequential(input_shape=(4, 16)).conv1d(8, 3, padding=1).prelu(8)
        """
        return _SEQUENTIAL_ENTRY(self, "add", ("prelu", (num_parameters, init), _SEQUENTIAL_NO_KEYWORDS), _SEQUENTIAL_CONTEXT)

    def mish(self):
        """Append a Mish activation (PyTorch's ``nn.Mish``):
        ``x * tanh(softplus(x))``.

        Example::

            model = mojolearn.Sequential().linear(16).mish().linear(1)
        """
        return _SEQUENTIAL_ENTRY(self, "add", ("mish", (), _SEQUENTIAL_NO_KEYWORDS), _SEQUENTIAL_CONTEXT)

    def hardswish(self):
        """Append a hard swish activation (PyTorch's ``nn.Hardswish``):
        ``x * relu6(x + 3) / 6``, that is 0 for ``x <= -3``, ``x`` for
        ``x >= 3`` and ``x * (x + 3) / 6`` between.

        Example::

            model = mojolearn.Sequential().linear(16).hardswish().linear(1)
        """
        return _SEQUENTIAL_ENTRY(self, "add", ("hardswish", (), _SEQUENTIAL_NO_KEYWORDS), _SEQUENTIAL_CONTEXT)

    def hardsigmoid(self):
        """Append a hard sigmoid activation (PyTorch's ``nn.Hardsigmoid``):
        ``relu6(x + 3) / 6``, that is 0 for ``x <= -3``, 1 for ``x >= 3`` and
        ``x / 6 + 1 / 2`` between.

        Example::

            model = mojolearn.Sequential().linear(16).hardsigmoid().linear(1)
        """
        return _SEQUENTIAL_ENTRY(self, "add", ("hardsigmoid", (), _SEQUENTIAL_NO_KEYWORDS), _SEQUENTIAL_CONTEXT)
    # nn-activations end

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
        the pairs). ``head_dim`` is even and divides the features.

        Example::

            model.linear(64).rotary(16)
        """
        return _SEQUENTIAL_ENTRY(self, "add", ("rotary", (head_dim, theta), _SEQUENTIAL_NO_KEYWORDS), _SEQUENTIAL_CONTEXT)

    def multiply(self, source=-1):
        """Multiply the current output elementwise by the output of layer
        ``source`` (-1: the model input); the shapes must agree.

        Example: a sigmoid gate on the output of layer 0::

            model.linear(32).linear(32).sigmoid().multiply(0)
        """
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
        ``(length,)`` sample becomes ``(length, embedding_dim)``.

        Example: token ids of 32 positions into 64 features::

            mojolearn.Sequential(input_shape=(32,)).embedding(512, 64)
        """
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

    def lstm(self, hidden_size, num_layers=1, *, bidirectional=False, dropout=0.0, output="sequence"):
        """Append an LSTM over ``(length, features)`` samples from a zero
        state, as PyTorch's ``nn.LSTM``: the same weight layout, gate order
        and initialization (every tensor uniform in
        ``[-1/sqrt(hidden_size), 1/sqrt(hidden_size)]``).

        Parameters
        ----------
        hidden_size : int
            Size of the hidden state.
        num_layers : int, default 1
            Number of stacked layers; layer ``k + 1`` reads layer ``k``'s
            output sequence.
        bidirectional : bool, default False
            Add a reverse direction to every layer. It reads each sample's
            steps in reverse order, starting at its last valid step when
            ``lengths`` are given, and its states are concatenated after the
            forward direction's on the features.
        dropout : float, default 0.0
            Dropout probability on the output of every layer except the last,
            during :meth:`fit` only (no effect with one layer).
        output : {'sequence', 'final'}, default 'sequence'
            ``'sequence'``: every step's state of the last layer,
            ``(length, directions * hidden_size)`` (PyTorch's ``output``).
            ``'final'``: the last layer's final state of each direction,
            ``(directions, hidden_size)`` (PyTorch's ``h_n`` of the last
            layer).

        With ``lengths`` given to :meth:`fit` and :meth:`predict`, a sample's
        steps past its length do not change its state; the ``'sequence'``
        output there repeats the forward direction's final state and is 0
        for the reverse direction.

        With more than one layer, ``bidirectional=True`` or
        ``output='final'``, the module is written into the model as several
        layers (one recurrent layer per layer and direction, the joins of the
        directions and the dropout layers), in PyTorch's parameter order:
        ``weight_ih``, ``weight_hh``, ``bias_ih``, ``bias_hh`` of layer 0's
        forward direction, then of its reverse direction, then layer 1.

        Example: classify variable-length sequences of 3 features padded to
        20 steps by the final states of a two-layer bidirectional LSTM::

            model = (mojolearn.Sequential(input_shape=(20, 3))
                     .lstm(32, num_layers=2, bidirectional=True, output="final")
                     .flatten().linear(2)
                     .compile(loss="cross_entropy"))
            model.fit(X, y, lengths=lengths)
        """
        return _SEQUENTIAL_ENTRY(self, "add", ("lstm", (hidden_size, num_layers, bidirectional, dropout, output), _SEQUENTIAL_NO_KEYWORDS), _SEQUENTIAL_CONTEXT)

    def gru(self, hidden_size, num_layers=1, *, bidirectional=False, dropout=0.0, output="sequence"):
        """Append a GRU over ``(length, features)`` samples from a zero
        state, as PyTorch's ``nn.GRU``: the same weight layout, gate order
        and initialization. The arguments, ``lengths`` and the output are as
        for :meth:`lstm`.

        Example::

            mojolearn.Sequential(input_shape=(20, 3)).gru(32, output="final").flatten().linear(1)
        """
        return _SEQUENTIAL_ENTRY(self, "add", ("gru", (hidden_size, num_layers, bidirectional, dropout, output), _SEQUENTIAL_NO_KEYWORDS), _SEQUENTIAL_CONTEXT)

    def rnn(self, hidden_size, num_layers=1, *, bidirectional=False, dropout=0.0, output="sequence"):
        """Append a tanh recurrent layer over ``(length, features)`` samples
        from a zero state, as PyTorch's ``nn.RNN`` with
        ``nonlinearity='tanh'``. The arguments, ``lengths`` and the output
        are as for :meth:`lstm`.

        Example::

            mojolearn.Sequential(input_shape=(20, 3)).rnn(32).narrow(0, 19, 1).flatten().linear(1)
        """
        return _SEQUENTIAL_ENTRY(self, "add", ("rnn", (hidden_size, num_layers, bidirectional, dropout, output), _SEQUENTIAL_NO_KEYWORDS), _SEQUENTIAL_CONTEXT)

    def batch_norm1d(self, num_features, eps=1e-5, momentum=0.1):
        """Append batch normalization over the channels of ``(channels,)`` or
        ``(channels, length)`` samples, as PyTorch's ``nn.BatchNorm1d``.

        During :meth:`fit` each channel is normalized with the mean and
        (biased) variance of the batch over its rows and positions, and the
        running statistics are updated as
        ``running = (1 - momentum) * running + momentum * batch`` (the
        variance unbiased). :meth:`predict` and validation normalize with the
        running statistics. Then a learned per-channel ``weight`` (initially
        1) and ``bias`` (initially 0) are applied. The running statistics are
        saved with the model (``params_``, :meth:`state_dict`, checkpoints)
        and are never changed by the optimizer.

        Parameters
        ----------
        num_features : int
            The number of channels, the first axis of each sample.
        eps : float, default 1e-5
            Added to the variance before its square root.
        momentum : float, default 0.1
            Weight of the batch statistics in the running update, in [0, 1].

        Example::

            model = (mojolearn.Sequential(input_shape=(3, 64))
                     .conv1d(16, 5, padding=2).batch_norm1d(16).relu()
                     .adaptive_avg_pool1d(1).flatten().linear(2))
        """
        return _SEQUENTIAL_ENTRY(self, "add", ("batch_norm1d", (num_features, eps, momentum), _SEQUENTIAL_NO_KEYWORDS), _SEQUENTIAL_CONTEXT)

    def batch_norm2d(self, num_features, eps=1e-5, momentum=0.1):
        """Append batch normalization over the channels of
        ``(channels, height, width)`` samples, as PyTorch's
        ``nn.BatchNorm2d``: :meth:`batch_norm1d` with the statistics taken
        over the rows and both spatial axes.

        Example::

            model = (mojolearn.Sequential(input_shape=(1, 28, 28))
                     .conv2d(8, 3, padding=1).batch_norm2d(8).relu()
                     .flatten().linear(10))
        """
        return _SEQUENTIAL_ENTRY(self, "add", ("batch_norm2d", (num_features, eps, momentum), _SEQUENTIAL_NO_KEYWORDS), _SEQUENTIAL_CONTEXT)

    # nn-norms begin
    def group_norm(self, num_groups, num_channels, eps=1e-5, affine=True):
        """Append group normalization, as PyTorch's ``nn.GroupNorm``.

        The channels of each sample (its first axis) are split into
        ``num_groups`` groups of consecutive channels. Every group of every
        sample is normalized with its own mean and (biased) variance over
        its channels and the remaining axes, so the result does not depend
        on the other rows of the batch and is the same in :meth:`fit` and
        :meth:`predict`. With ``affine`` a learned per-channel ``weight``
        (initially 1) and ``bias`` (initially 0) are then applied.

        Parameters
        ----------
        num_groups : int
            The number of groups; it must divide ``num_channels``.
            ``num_groups=1`` normalizes every sample as a whole (layer norm
            over all its values); ``num_groups=num_channels`` normalizes each
            channel separately (instance norm).
        num_channels : int
            The number of channels, the first axis of each sample.
        eps : float, default 1e-5
            Added to the variance before its square root.
        affine : bool, default True
            Whether to learn the per-channel ``weight`` and ``bias``.

        Example::

            model = (mojolearn.Sequential(input_shape=(3, 64))
                     .conv1d(16, 5, padding=2).group_norm(4, 16).relu()
                     .adaptive_avg_pool1d(1).flatten().linear(2))
        """
        return _SEQUENTIAL_ENTRY(self, "add", ("group_norm", (num_groups, num_channels, eps, affine), _SEQUENTIAL_NO_KEYWORDS), _SEQUENTIAL_CONTEXT)

    def instance_norm1d(self, num_features, eps=1e-5, momentum=0.1, affine=False, track_running_stats=False):
        """Append instance normalization over ``(channels, length)``
        samples, as PyTorch's ``nn.InstanceNorm1d``.

        Each channel of each sample is normalized with its own mean and
        (biased) variance over the length. With ``affine`` a learned
        per-channel ``weight`` (initially 1) and ``bias`` (initially 0) are
        then applied.

        With ``track_running_stats`` the layer also keeps running statistics
        per channel: during :meth:`fit` they are updated as
        ``running = (1 - momentum) * running + momentum * stat``, with
        ``stat`` the batch's average of the per-sample means and (unbiased)
        variances, and :meth:`predict` and validation normalize with them.
        Without it every call normalizes with the sample's own statistics.
        The running statistics are saved with the model (``params_``,
        :meth:`state_dict`, checkpoints) and are never changed by the
        optimizer.

        Parameters
        ----------
        num_features : int
            The number of channels, the first axis of each sample.
        eps : float, default 1e-5
            Added to the variance before its square root.
        momentum : float, default 0.1
            Weight of the batch statistics in the running update, in [0, 1].
        affine : bool, default False
            Whether to learn the per-channel ``weight`` and ``bias``.
        track_running_stats : bool, default False
            Whether to keep running statistics and use them in
            :meth:`predict`.

        Example::

            model = (mojolearn.Sequential(input_shape=(3, 64))
                     .conv1d(16, 5, padding=2).instance_norm1d(16, affine=True)
                     .relu().adaptive_avg_pool1d(1).flatten().linear(2))
        """
        return _SEQUENTIAL_ENTRY(self, "add", ("instance_norm1d", (num_features, eps, momentum, affine, track_running_stats), _SEQUENTIAL_NO_KEYWORDS), _SEQUENTIAL_CONTEXT)

    def instance_norm2d(self, num_features, eps=1e-5, momentum=0.1, affine=False, track_running_stats=False):
        """Append instance normalization over ``(channels, height, width)``
        samples, as PyTorch's ``nn.InstanceNorm2d``: :meth:`instance_norm1d`
        with the statistics taken over both spatial axes.

        Example::

            model = (mojolearn.Sequential(input_shape=(1, 28, 28))
                     .conv2d(8, 3, padding=1).instance_norm2d(8, affine=True)
                     .relu().flatten().linear(10))
        """
        return _SEQUENTIAL_ENTRY(self, "add", ("instance_norm2d", (num_features, eps, momentum, affine, track_running_stats), _SEQUENTIAL_NO_KEYWORDS), _SEQUENTIAL_CONTEXT)
    # nn-norms end

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

    def ssd_scan(self, heads):
        """Append Mamba-2's state-space dual scan over ``(length,
        features)`` samples whose features are ``[x | B | C | dt]``:
        ``heads`` groups of 64 input channels ``x``, the 128 input and
        128 output projections ``B`` and ``C`` of each position (one group
        shared by every head), and one step size ``dt`` per head (already
        positive, for example after ``softplus``), so ``features = 65 *
        heads + 256``. The output is ``(length, 64 * heads)``: per head
        ``y[t] = C[t] . h[t] + D * x[t]`` with ``h[t] = exp(dt[t] * A) *
        h[t - 1] + dt[t] * B[t] x[t]`` from ``h = 0``, where ``A =
        -exp(A_log)``, computed by Mamba-2's chunked algorithm.
        ``A_log`` and ``D`` (one value per head) are learned, initialized
        to ``A_log = log(1), .., log(heads)`` and ``D = 1``.

        Parameters
        ----------
        heads : int
            Number of heads of 64 channels.

        Example: the core of a two-head Mamba-2 mixer over ``(length,
        386)`` samples::

            model.ssd_scan(2)
        """
        return _SEQUENTIAL_ENTRY(self, "add", ("ssd_scan", (heads,), _SEQUENTIAL_NO_KEYWORDS), _SEQUENTIAL_CONTEXT)

    def trapezoid_scan(self, heads, d_state):
        """Append Mamba-3's trapezoid state-space scan over ``(length,
        features)`` samples whose features are ``[v | k | q | dt | A |
        trap]``: per head, ``headdim`` values ``v``, ``d_state`` keys ``k``
        and queries ``q``, and one step size ``dt``, decay rate ``A`` and
        trapezoid logit ``trap`` per position, so ``features = heads *
        (headdim + 2 * d_state + 3)``. The output is ``(length, heads *
        headdim)``. Per head, from ``h = 0``::

            alpha = exp(A dt), sigma = sigmoid(trap)
            h[t] = alpha h[t-1] + dt (1 - sigma) alpha k[t-1] v[t-1]
                   + dt sigma k[t] v[t]
            y[t] = q[t] . h[t] + D v[t]

        ``D`` ``(heads,)`` is learned, initialized to 1.

        Parameters
        ----------
        heads : int
            Number of heads.
        d_state : int
            State size per head.

        Example: the core of a Mamba-3 mixer with 2 heads of 16 channels
        and 32 states over ``(length, 2 * (16 + 64 + 3))`` samples::

            model.trapezoid_scan(heads=2, d_state=32)
        """
        return _SEQUENTIAL_ENTRY(self, "add", ("trapezoid_scan", (heads, d_state), _SEQUENTIAL_NO_KEYWORDS), _SEQUENTIAL_CONTEXT)

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

    # nn-attention begin
    def multihead_attention(self, embed_dim, num_heads, dropout=0.0, bias=True, add_bias_kv=False,
                            add_zero_attn=False, kdim=None, vdim=None, batch_first=True, *, source=None,
                            is_causal=False, attn_mask=None, key_padding_mask=False):
        """Append multi-head attention, as PyTorch's ``nn.MultiheadAttention``
        with ``batch_first=True``, over ``(length, embed_dim)`` samples.

        The queries are the current output. The keys and values are the
        current output too (self-attention), or the output of layer
        ``source`` (cross-attention, for example a decoder attending to an
        encoder's output). Each projection is ``x W^T + b``; per head the
        result is ``softmax(q k^T / sqrt(head_dim) + mask) v``; the heads
        are joined and projected by ``out_proj``. The output has the shape
        of the queries.

        Parameters
        ----------
        embed_dim : int
            The features of the queries and of the output.
        num_heads : int
            Number of heads; it divides ``embed_dim``.
        dropout : float, default 0.0
            Dropout on the attention weights during :meth:`fit`.
        bias : bool, default True
            Add the projection biases ``in_proj_bias`` and ``out_proj.bias``.
            Without bias they stay zero and are never trained.
        add_bias_kv, add_zero_attn : bool, default False
            Must be False.
        kdim, vdim : int or None, default None
            The features of the keys and values (equal: both come from
            ``source``). None takes ``embed_dim``.
        batch_first : bool, default True
            Must be True: samples are ``(length, features)``.
        source : int or None, default None
            None for self-attention, -1 for the model input, or the index of
            an earlier layer whose output gives the keys and values.
        is_causal : bool, default False
            Each query attends only to keys at its own position or before.
        attn_mask : None or 'causal', default None
            ``'causal'`` is the same as ``is_causal=True``.
        key_padding_mask : bool, default False
            Ignore the keys at or past each sample's length, the
            ``lengths`` given to :meth:`fit` and :meth:`predict` (PyTorch's
            ``key_padding_mask`` True there). The keys must have the model
            input's length.

        The parameters are PyTorch's: ``in_proj_weight`` ``(3 * embed_dim,
        embed_dim)`` (the query, key and value weights stacked),
        ``out_proj.weight``, ``in_proj_bias`` and ``out_proj.bias``,
        initialized as PyTorch does. With a ``kdim`` other than
        ``embed_dim``, ``in_proj_weight`` is one vector: the query weight
        ``(embed_dim, embed_dim)`` then the key and value weights side by side
        ``(kdim, 2 * embed_dim)``, each laid out (inputs, outputs).

        Examples
        --------
        >>> import mojolearn
        >>> model = (mojolearn.Sequential(input_shape=(10,)).embedding(20, 16)
        ...          .positional_encoding(16, dropout=0.0)
        ...          .multihead_attention(16, 4, is_causal=True)
        ...          .flatten().linear(3))
        """
        return _SEQUENTIAL_ENTRY(self, "add", ("multihead_attention", (embed_dim, num_heads, dropout, bias, add_bias_kv, add_zero_attn, kdim, vdim, batch_first, source, is_causal, attn_mask, key_padding_mask), _SEQUENTIAL_NO_KEYWORDS), _SEQUENTIAL_CONTEXT)

    def positional_encoding(self, d_model, dropout=0.1, max_len=5000):
        """Append the sinusoidal positional encoding of "Attention Is All You
        Need" (the ``PositionalEncoding`` of PyTorch's transformer
        tutorial): ``x + PE[:length]`` with ``PE[p, 2j] = sin(p / 10000^(2j /
        d_model))`` and ``PE[p, 2j + 1] = cos(p / 10000^(2j / d_model))``,
        then dropout during :meth:`fit`. It has no parameters.

        Parameters
        ----------
        d_model : int
            The features of each position (even).
        dropout : float, default 0.1
            Dropout after the addition.
        max_len : int, default 5000
            The longest sequence accepted (at most 8192).

        Examples
        --------
        >>> import mojolearn
        >>> model = mojolearn.Sequential(input_shape=(12,)).embedding(50, 32).positional_encoding(32)
        """
        return _SEQUENTIAL_ENTRY(self, "add", ("positional_encoding", (d_model, dropout, max_len), _SEQUENTIAL_NO_KEYWORDS), _SEQUENTIAL_CONTEXT)

    def transformer_encoder_layer(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu",
                                  layer_norm_eps=1e-5, batch_first=True, norm_first=False, bias=True, *,
                                  is_causal=False, key_padding_mask=False):
        """Append a Transformer encoder layer, as PyTorch's
        ``nn.TransformerEncoderLayer`` with ``batch_first=True``.

        With ``norm_first=False``: ``x = norm1(x + dropout(self_attn(x)))``
        then ``x = norm2(x + dropout(ff(x)))``, where ``ff`` is ``linear(
        dim_feedforward)``, the activation, dropout and ``linear(d_model)``.
        With ``norm_first=True`` the norms come first:
        ``x = x + dropout(self_attn(norm1(x)))``, ``x = x + dropout(ff(norm2(
        x)))``. The layer is written into the model as its modules
        (:meth:`multihead_attention`, :meth:`dropout`, :meth:`residual`,
        :meth:`layer_norm`, :meth:`linear` and the activation), which appear
        in :attr:`layers` and in the parameter names.

        Parameters
        ----------
        d_model : int
            The features of each position.
        nhead : int
            Attention heads; it divides ``d_model``.
        dim_feedforward : int, default 2048
            The width of the feed-forward network.
        dropout : float, default 0.1
            Dropout in attention, after attention and in the feed-forward
            network, during :meth:`fit`.
        activation : {'relu', 'gelu'}, default 'relu'
        layer_norm_eps : float, default 1e-5
        batch_first : bool, default True
            Must be True.
        norm_first : bool, default False
        bias : bool, default True
            Must be True.
        is_causal : bool, default False
            A causal mask in self-attention.
        key_padding_mask : bool, default False
            Ignore the positions at or past each sample's ``lengths`` (see
            :meth:`multihead_attention`).

        Examples
        --------
        >>> import mojolearn
        >>> model = (mojolearn.Sequential(input_shape=(16,)).embedding(100, 32)
        ...          .positional_encoding(32)
        ...          .transformer_encoder_layer(32, 4, dim_feedforward=64)
        ...          .flatten().linear(2))
        """
        return _SEQUENTIAL_ENTRY(self, "add", ("transformer_encoder_layer", (d_model, nhead, dim_feedforward, dropout, activation, layer_norm_eps, batch_first, norm_first, bias, is_causal, key_padding_mask), _SEQUENTIAL_NO_KEYWORDS), _SEQUENTIAL_CONTEXT)

    def transformer_decoder_layer(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu",
                                  layer_norm_eps=1e-5, batch_first=True, norm_first=False, bias=True, *,
                                  memory=None, tgt_is_causal=False, tgt_key_padding_mask=False,
                                  memory_key_padding_mask=False):
        """Append a Transformer decoder layer, as PyTorch's
        ``nn.TransformerDecoderLayer`` with ``batch_first=True``: self-attention
        over the current output, attention to ``memory`` (the output of an
        earlier layer, typically an encoder's), and the feed-forward network,
        each with dropout, a residual connection and a layer norm (first, with
        ``norm_first=True``). The layer is written into the model as its
        modules, as for :meth:`transformer_encoder_layer`.

        Parameters
        ----------
        d_model, nhead, dim_feedforward, dropout, activation, layer_norm_eps, batch_first, norm_first, bias
            As for :meth:`transformer_encoder_layer`.
        memory : int or None, default None
            The layer whose output is attended to (-1: the model input); it
            has ``d_model`` features. None only in a layer given to
            :meth:`transformer_decoder`, which sets it.
        tgt_is_causal : bool, default False
            A causal mask in self-attention (each position sees only itself
            and earlier ones), as a decoder trained with teacher forcing needs.
        tgt_key_padding_mask, memory_key_padding_mask : bool, default False
            Ignore the target or memory positions at or past each sample's
            ``lengths`` (see :meth:`multihead_attention`).

        Examples
        --------
        >>> import mojolearn
        >>> S = mojolearn.Sequential
        >>> enc = S(input_shape=(8, 16)).transformer_encoder_layer(16, 2, 32, dropout=0.0)
        >>> memory = len(enc.layers) - 1   # the encoder layer's last module
        >>> model = (enc.transformer_decoder_layer(16, 2, 32, dropout=0.0, memory=memory, tgt_is_causal=True)
        ...          .flatten().linear(4))
        """
        return _SEQUENTIAL_ENTRY(self, "add", ("transformer_decoder_layer", (d_model, nhead, dim_feedforward, dropout, activation, layer_norm_eps, batch_first, norm_first, bias, memory, tgt_is_causal, tgt_key_padding_mask, memory_key_padding_mask), _SEQUENTIAL_NO_KEYWORDS), _SEQUENTIAL_CONTEXT)

    def transformer_encoder(self, encoder_layer, num_layers, norm=None):
        """Append ``num_layers`` copies of ``encoder_layer`` (a ``Sequential``
        holding a :meth:`transformer_encoder_layer`), then ``norm``, as
        PyTorch's ``nn.TransformerEncoder``. Each copy has its own
        parameters; ``encoder_layer`` itself is not changed or trained.

        Parameters
        ----------
        encoder_layer : Sequential
        num_layers : int
        norm : Sequential or None, default None
            Layers after the last copy, for example
            ``Sequential().layer_norm()``.

        Examples
        --------
        >>> import mojolearn
        >>> S = mojolearn.Sequential
        >>> layer = S().transformer_encoder_layer(32, 4, dim_feedforward=64)
        >>> model = (S(input_shape=(16,)).embedding(100, 32).positional_encoding(32)
        ...          .transformer_encoder(layer, 2).flatten().linear(2))
        """
        return _SEQUENTIAL_ENTRY(self, "add", ("transformer_encoder", (encoder_layer, num_layers, norm), _SEQUENTIAL_NO_KEYWORDS), _SEQUENTIAL_CONTEXT)

    def transformer_decoder(self, decoder_layer, num_layers, memory, norm=None):
        """Append ``num_layers`` copies of ``decoder_layer`` (a ``Sequential``
        holding a :meth:`transformer_decoder_layer` built without
        ``memory``), each attending to the output of layer ``memory``, then
        ``norm``, as PyTorch's ``nn.TransformerDecoder``.

        Parameters
        ----------
        decoder_layer : Sequential
        num_layers : int
        memory : int
            The layer whose output every copy attends to (-1: the model
            input).
        norm : Sequential or None, default None

        Examples
        --------
        >>> import mojolearn
        >>> S = mojolearn.Sequential
        >>> layer = S().transformer_decoder_layer(16, 2, 32, dropout=0.0, tgt_is_causal=True)
        >>> model = (S(input_shape=(8, 16)).linear(16)
        ...          .transformer_decoder(layer, 2, memory=-1).flatten().linear(4))
        """
        return _SEQUENTIAL_ENTRY(self, "add", ("transformer_decoder", (decoder_layer, num_layers, memory, norm), _SEQUENTIAL_NO_KEYWORDS), _SEQUENTIAL_CONTEXT)
    # nn-attention end

    # nn-conv begin
    def conv_transpose1d(self, out_channels, kernel_size, stride=1, padding=0, output_padding=0, groups=1,
                         bias=True, dilation=1):
        """Append a 1-D transposed convolution of ``(channels, length)``
        samples, as PyTorch's ``nn.ConvTranspose1d`` (the gradient of
        :meth:`conv1d` with respect to its input, so it is often used to
        upsample in a decoder).

        The output length is ``(length - 1) * stride - 2 * padding +
        dilation * (kernel_size - 1) + output_padding + 1``. The weight is
        ``(channels, out_channels // groups, kernel_size)`` and the bias
        ``(out_channels,)``, both initialized uniformly in
        ``[-1 / sqrt(fan_in), 1 / sqrt(fan_in)]`` with ``fan_in =
        out_channels // groups * kernel_size``.

        Parameters
        ----------
        out_channels : int
            Number of output channels.
        kernel_size : int
            Kernel length.
        stride : int, default 1
            Step between the positions an input value is spread to.
        padding : int, default 0
            ``dilation * (kernel_size - 1) - padding`` zeros are added at both
            ends of the output.
        output_padding : int, default 0
            Extra length added at the end of the output; smaller than
            ``stride``.
        groups : int, default 1
            Number of channel groups; it divides the channels and
            ``out_channels``.
        bias : bool, default True
            Whether to add a learned bias.
        dilation : int, default 1
            Spacing between kernel taps.

        Example: a convolutional autoencoder over ``(1, 32)`` signals::

            model = (mojolearn.Sequential(input_shape=(1, 32))
                     .conv1d(8, 5, stride=2, padding=2).tanh()
                     .conv_transpose1d(1, 4, stride=2, padding=1).flatten())
        """
        return _SEQUENTIAL_ENTRY(self, "add", ("conv_transpose1d", (out_channels, kernel_size, stride, padding, output_padding, groups, bias, dilation), _SEQUENTIAL_NO_KEYWORDS), _SEQUENTIAL_CONTEXT)

    def conv_transpose2d(self, out_channels, kernel_size, stride=1, padding=0, output_padding=0, groups=1,
                         bias=True, dilation=1):
        """Append a 2-D transposed convolution of ``(channels, height,
        width)`` samples with square kernels, as PyTorch's
        ``nn.ConvTranspose2d``: :meth:`conv_transpose1d` along both spatial
        axes. The weight is ``(channels, out_channels // groups,
        kernel_size, kernel_size)``.

        Example: double the height and width of 16-channel feature maps::

            model.conv_transpose2d(8, 4, stride=2, padding=1)
        """
        return _SEQUENTIAL_ENTRY(self, "add", ("conv_transpose2d", (out_channels, kernel_size, stride, padding, output_padding, groups, bias, dilation), _SEQUENTIAL_NO_KEYWORDS), _SEQUENTIAL_CONTEXT)

    def adaptive_avg_pool2d(self, output_size):
        """Append an adaptive average pooling of ``(channels, height,
        width)`` samples to ``(channels, out_height, out_width)``, as
        PyTorch's ``nn.AdaptiveAvgPool2d``: output ``(i, j)`` is the mean of
        rows ``floor(i * height / out_height)`` to ``ceil((i + 1) * height /
        out_height)`` (exclusive) and the columns likewise, so any output
        size works.

        Parameters
        ----------
        output_size : int or tuple
            ``out_height`` and ``out_width``, one int for both; ``None`` in
            the tuple keeps that axis's size.

        Example::

            model.conv2d(16, 3, padding=1).relu().adaptive_avg_pool2d((4, 4)).flatten().linear(10)
        """
        return _SEQUENTIAL_ENTRY(self, "add", ("adaptive_avg_pool2d", (output_size,), _SEQUENTIAL_NO_KEYWORDS), _SEQUENTIAL_CONTEXT)

    def adaptive_max_pool2d(self, output_size):
        """Append an adaptive max pooling of ``(channels, height, width)``
        samples, as PyTorch's ``nn.AdaptiveMaxPool2d``: the windows of
        :meth:`adaptive_avg_pool2d`, each output the largest value of its
        window (of equal values the first in row-major order receives the
        gradient).

        Example::

            model.adaptive_max_pool2d(2)
        """
        return _SEQUENTIAL_ENTRY(self, "add", ("adaptive_max_pool2d", (output_size,), _SEQUENTIAL_NO_KEYWORDS), _SEQUENTIAL_CONTEXT)

    def adaptive_max_pool1d(self, output_size):
        """Append an adaptive max pooling of ``(channels, length)`` samples
        to ``(channels, output_size)``, as PyTorch's ``nn.AdaptiveMaxPool1d``:
        output ``i`` is the largest value of positions ``floor(i * length /
        output_size)`` to ``ceil((i + 1) * length / output_size)``
        (exclusive).

        Example::

            model.conv1d(16, 5, padding=2).relu().adaptive_max_pool1d(4).flatten().linear(2)
        """
        return _SEQUENTIAL_ENTRY(self, "add", ("adaptive_max_pool1d", (output_size,), _SEQUENTIAL_NO_KEYWORDS), _SEQUENTIAL_CONTEXT)

    def global_max_pool(self):
        """Append a global max pooling: each channel of a ``(channels,
        length)`` or ``(channels, height, width)`` sample becomes its largest
        value, giving ``(channels,)`` (PyTorch's ``AdaptiveMaxPool1d(1)`` or
        ``AdaptiveMaxPool2d(1)`` followed by a flatten). Of equal values the
        first receives the gradient.

        Example::

            model.conv1d(32, 3, padding=1).relu().global_max_pool().linear(2)
        """
        return _SEQUENTIAL_ENTRY(self, "add", ("global_max_pool", (), _SEQUENTIAL_NO_KEYWORDS), _SEQUENTIAL_CONTEXT)

    def zero_pad1d(self, padding):
        """Append zero padding of ``(channels, length)`` samples, as
        PyTorch's ``nn.ZeroPad1d``: ``padding`` zeros at both ends, or
        ``(left, right)``. A negative pad removes values.

        Example::

            model.zero_pad1d((2, 0))
        """
        return _SEQUENTIAL_ENTRY(self, "add", ("zero_pad1d", (padding,), _SEQUENTIAL_NO_KEYWORDS), _SEQUENTIAL_CONTEXT)

    def zero_pad2d(self, padding):
        """Append zero padding of ``(channels, height, width)`` samples, as
        PyTorch's ``nn.ZeroPad2d``: ``padding`` zeros on every side, or
        ``(left, right, top, bottom)``. A negative pad removes values.

        Example::

            model.zero_pad2d(1)
        """
        return _SEQUENTIAL_ENTRY(self, "add", ("zero_pad2d", (padding,), _SEQUENTIAL_NO_KEYWORDS), _SEQUENTIAL_CONTEXT)

    def constant_pad1d(self, padding, value):
        """Append constant padding of ``(channels, length)`` samples with
        ``value``, as PyTorch's ``nn.ConstantPad1d``: ``padding`` at both
        ends, or ``(left, right)``. A negative pad removes values.

        Example::

            model.constant_pad1d(2, -1.0)
        """
        return _SEQUENTIAL_ENTRY(self, "add", ("constant_pad1d", (padding, value), _SEQUENTIAL_NO_KEYWORDS), _SEQUENTIAL_CONTEXT)

    def constant_pad2d(self, padding, value):
        """Append constant padding of ``(channels, height, width)`` samples
        with ``value``, as PyTorch's ``nn.ConstantPad2d``: ``padding`` on
        every side, or ``(left, right, top, bottom)``.

        Example::

            model.constant_pad2d((1, 1, 2, 0), 0.5)
        """
        return _SEQUENTIAL_ENTRY(self, "add", ("constant_pad2d", (padding, value), _SEQUENTIAL_NO_KEYWORDS), _SEQUENTIAL_CONTEXT)

    def reflection_pad1d(self, padding):
        """Append reflection padding of ``(channels, length)`` samples, as
        PyTorch's ``nn.ReflectionPad1d``: the values mirrored at each end,
        the end value itself not repeated. ``padding`` (both ends) or
        ``(left, right)``, each smaller than the length.

        Example::

            model.reflection_pad1d(2).conv1d(8, 5)
        """
        return _SEQUENTIAL_ENTRY(self, "add", ("reflection_pad1d", (padding,), _SEQUENTIAL_NO_KEYWORDS), _SEQUENTIAL_CONTEXT)

    def reflection_pad2d(self, padding):
        """Append reflection padding of ``(channels, height, width)``
        samples, as PyTorch's ``nn.ReflectionPad2d``: ``padding`` (every
        side) or ``(left, right, top, bottom)``, each smaller than the
        padded axis.

        Example::

            model.reflection_pad2d(1).conv2d(16, 3)
        """
        return _SEQUENTIAL_ENTRY(self, "add", ("reflection_pad2d", (padding,), _SEQUENTIAL_NO_KEYWORDS), _SEQUENTIAL_CONTEXT)

    def replication_pad1d(self, padding):
        """Append replication padding of ``(channels, length)`` samples, as
        PyTorch's ``nn.ReplicationPad1d``: each end value repeated.
        ``padding`` (both ends) or ``(left, right)``.

        Example::

            model.replication_pad1d((4, 0))
        """
        return _SEQUENTIAL_ENTRY(self, "add", ("replication_pad1d", (padding,), _SEQUENTIAL_NO_KEYWORDS), _SEQUENTIAL_CONTEXT)

    def replication_pad2d(self, padding):
        """Append replication padding of ``(channels, height, width)``
        samples, as PyTorch's ``nn.ReplicationPad2d``: the edge values
        repeated. ``padding`` (every side) or ``(left, right, top,
        bottom)``.

        Example::

            model.replication_pad2d(1)
        """
        return _SEQUENTIAL_ENTRY(self, "add", ("replication_pad2d", (padding,), _SEQUENTIAL_NO_KEYWORDS), _SEQUENTIAL_CONTEXT)

    def upsample(self, size=None, scale_factor=None, mode="nearest", align_corners=None):
        """Append upsampling of ``(channels, length)`` or ``(channels,
        height, width)`` samples, as PyTorch's ``nn.Upsample``.

        Parameters
        ----------
        size : int or tuple, optional
            The output length, or ``(height, width)`` (one int for both).
        scale_factor : float, optional
            The output size is ``floor(size * scale_factor)`` on each spatial
            axis. Give either ``size`` or ``scale_factor``. It is stored as a
            float32.
        mode : {'nearest', 'linear', 'bilinear'}, default 'nearest'
            ``'nearest'`` copies the value at ``floor(i * scale)``;
            ``'linear'`` (``(channels, length)`` samples) and ``'bilinear'``
            (``(channels, height, width)`` samples) interpolate between the
            two nearest values on each axis. ``scale`` is ``input_size /
            output_size``, or ``1 / scale_factor`` when one is given.
        align_corners : bool or None, default None
            For ``'linear'`` and ``'bilinear'``: True aligns the corner
            values of input and output; None or False uses PyTorch's
            half-pixel rule.

        Example: a decoder that doubles its feature maps twice::

            model.upsample(scale_factor=2).conv2d(8, 3, padding=1).relu() \
                 .upsample(scale_factor=2, mode="bilinear").conv2d(1, 3, padding=1)
        """
        return _SEQUENTIAL_ENTRY(self, "add", ("upsample", (size, scale_factor, mode, align_corners), _SEQUENTIAL_NO_KEYWORDS), _SEQUENTIAL_CONTEXT)

    def pixel_shuffle(self, upscale_factor):
        """Append a pixel shuffle, as PyTorch's ``nn.PixelShuffle``:
        ``(channels * r * r, height, width)`` samples become ``(channels,
        height * r, width * r)``, output ``[c, h * r + i, w * r + j]`` being
        input ``[c * r * r + i * r + j, h, w]`` with ``r = upscale_factor``.

        Example: sub-pixel upsampling by 2::

            model.conv2d(4 * 3, 3, padding=1).pixel_shuffle(2)
        """
        return _SEQUENTIAL_ENTRY(self, "add", ("pixel_shuffle", (upscale_factor,), _SEQUENTIAL_NO_KEYWORDS), _SEQUENTIAL_CONTEXT)

    def pixel_unshuffle(self, downscale_factor):
        """Append a pixel unshuffle, the inverse of :meth:`pixel_shuffle`
        (PyTorch's ``nn.PixelUnshuffle``): ``(channels, height * r, width *
        r)`` samples become ``(channels * r * r, height, width)``.

        Example::

            model.pixel_unshuffle(2)
        """
        return _SEQUENTIAL_ENTRY(self, "add", ("pixel_unshuffle", (downscale_factor,), _SEQUENTIAL_NO_KEYWORDS), _SEQUENTIAL_CONTEXT)
    # nn-conv end

    @property
    def layers(self):
        """The layers in order: ``(kind, arg0, arg1, float_arg)`` tuples,
        and ``(kind, arguments, float_arg)`` for ``reshape``, ``permute``,
        ``flatten``, ``conv2d`` and the pools, with ``arguments`` the
        layer's six integer arguments."""
        return _SEQUENTIAL_ENTRY(self, "layers", (), _SEQUENTIAL_CONTEXT)

    def compile(self, loss="mse", optimizer="adam", lr=1e-3, *, momentum=None, nesterov=False,
                beta1=0.9, beta2=0.999, eps=None, weight_decay=None, schedule="constant",
                step_size=1, gamma=0.1, huber_delta=1.0, tol=1e-4, n_iter_no_change=0,
                reduction="mean", pos_weight=1.0, beta=1.0, blank=0, zero_infinity=False,
                alpha=0.99, centered=False, lr_decay=0.0, initial_accumulator_value=0.0,
                max_grad_norm=1.0, trust_clip=False, always_adapt=False,
                milestones=None, T_max=None, eta_min=0.0, T_0=None, T_mult=1,
                warmup_steps=0, total_steps=None, pct_start=0.3, anneal_strategy="cos",
                div_factor=25.0, final_div_factor=1e4, three_phase=False,
                factor=0.1, patience=10, threshold=1e-4, cooldown=0, min_lr=0.0, cycle_momentum=True):
        """Choose the loss, optimizer and learning-rate schedule.

        Parameters
        ----------
        loss : {'mse', 'l1', 'smooth_l1', 'huber', 'cross_entropy', 'nll', 'bce_with_logits', 'kl_div', \
'ctc', 'token_cross_entropy'}, default 'mse'
            The loss of PyTorch's module of the same name: ``mse``
            (``MSELoss``), ``l1`` (``L1Loss``, the absolute error),
            ``smooth_l1`` (``SmoothL1Loss``: ``0.5 d**2 / beta`` where
            ``|d| < beta``, else ``|d| - 0.5 beta``) and ``huber``
            (``HuberLoss``) take real targets of the output's shape.
            ``cross_entropy`` takes logits and integer class labels;
            ``nll`` (``NLLLoss``) takes log-probabilities, one a class,
            and integer class labels; ``bce_with_logits``
            (``BCEWithLogitsLoss``) takes one logit per row and binary
            labels, computed stably from the logits; ``kl_div``
            (``KLDivLoss`` with ``log_target=False``) takes
            log-probabilities and target probabilities of the output's
            shape, ``sum y (log y - x)``, zero where ``y`` is zero;
            ``ctc`` (``CTCLoss``) takes each step's log-probabilities of a
            model whose samples are ``(length, classes)`` and that ends in
            :meth:`log_softmax`, and label sequences with their
            ``target_lengths`` (:meth:`fit`): the negative log-likelihood
            of the labels over every alignment of the sample's steps;
            ``token_cross_entropy`` trains a model ending in
            :meth:`lm_head` on ``(n_samples, length)`` int32 token ids.
        optimizer : {'sgd', 'momentum', 'adam', 'adamw', 'rmsprop', 'adagrad', 'lamb'}, default 'adam'
            ``'momentum'`` is SGD with ``momentum``. ``'rmsprop'`` and
            ``'adagrad'`` follow ``torch.optim.RMSprop`` and
            ``torch.optim.Adagrad``; ``'lamb'`` is LAMB as in timm's
            ``Lamb`` (bias correction and gradient averaging on, the
            layer-wise trust ratio applied to each parameter tensor).
        lr : float, default 1e-3
            Learning rate (> 0); the peak rate of ``'one_cycle'`` and
            ``'linear_warmup'``.
        momentum : float or None, default None
            Momentum of ``'momentum'`` (None: 0.9) and of ``'rmsprop'``
            (None: 0).
        nesterov : bool, default False
            Nesterov momentum for ``'momentum'``.
        beta1, beta2 : float
            Moment decays of ``'adam'``, ``'adamw'`` and ``'lamb'``.
        eps : float or None, default None
            Denominator term. None: 1e-8, or 1e-10 for ``'adagrad'`` and
            1e-6 for ``'lamb'``.
        weight_decay : float or None, default None
            L2 penalty (sgd, momentum, adam, rmsprop, adagrad) or
            decoupled decay (adamw, lamb). None: 0, or 0.01 for ``'lamb'``.
        schedule : str, default 'constant'
            The learning-rate schedule:

            - ``'constant'``.
            - ``'step'``: times ``gamma`` every ``step_size`` epochs.
            - ``'multi_step'``: times ``gamma`` at each epoch in
              ``milestones``.
            - ``'exponential'``: times ``gamma`` every epoch.
            - ``'cosine'``: ``eta_min + (lr - eta_min)(1 + cos(pi e /
              T_max)) / 2`` at epoch ``e``, PyTorch's closed form: past
              ``T_max`` epochs the rate rises again over the next
              ``T_max``, a period of ``2 T_max``.
            - ``'cosine_warm_restarts'``: the same cosine over cycles of
              ``T_0`` epochs, each ``T_mult`` times the last.
            - ``'linear_warmup'``: per optimizer step, rises linearly to
              ``lr`` over ``warmup_steps`` steps, then falls linearly to 0
              at ``total_steps``.
            - ``'one_cycle'``: ``torch.optim.lr_scheduler.OneCycleLR`` per
              optimizer step with ``max_lr=lr`` over ``total_steps`` steps;
              it cycles the learning rate only, so it needs
              ``cycle_momentum=False``.
            - ``'reduce_on_plateau'``: ``ReduceLROnPlateau`` in ``'min'``
              mode on the validation loss when ``fit`` has
              ``validation_data``, else on the epoch's training loss.
        step_size : int, default 1
            Epochs between ``'step'`` decays.
        gamma : float, default 0.1
            Decay factor of ``'step'``, ``'multi_step'`` and ``'exponential'``.
        huber_delta : float, default 1.0
            Threshold of the Huber loss.
        reduction : {'mean', 'sum', 'batchmean'}, default 'mean'
            How a batch's terms become its loss, as in PyTorch: ``'mean'``
            averages over the elements (over the samples for
            ``cross_entropy`` and ``nll``; for ``ctc``, each sample's loss
            divided by its target length, then averaged over the samples),
            ``'sum'`` adds them, and
            ``'batchmean'`` (``kl_div`` only) sums and divides by the
            samples, the divergence per sample. ``loss_curve_`` holds each
            epoch's loss per sample (the mean of the batch losses, weighted
            by their samples, for a mean; the summed loss over the
            samples for a sum). ``token_cross_entropy`` takes ``'mean'``.
        pos_weight : float, default 1.0
            Weight of the positive term of ``bce_with_logits`` (> 0), the
            same for every output: ``-[pos_weight y log s + (1 - y)
            log(1 - s)]``, ``s`` the sigmoid of the logit.
        beta : float, default 1.0
            Threshold of ``smooth_l1`` (>= 0; 0 is the L1 loss).
        blank : int, default 0
            The class index of the ``ctc`` blank (a class of the model's
            output that no label may use).
        zero_infinity : bool, default False
            ``ctc`` only: a sample whose labels no alignment of its steps
            can produce (more labels, with the blanks needed between
            repeats, than steps) has an infinite loss; True gives it loss
            0 and no gradient instead.
        tol, n_iter_no_change : float, int
            Stop when the epoch loss has not improved by more than ``tol``
            for ``n_iter_no_change`` epochs; 0 trains every epoch.
        alpha : float, default 0.99
            Smoothing constant of ``'rmsprop'``.
        centered : bool, default False
            Centered ``'rmsprop'`` (normalize by the estimated variance);
            available without momentum.
        lr_decay : float, default 0.0
            Learning-rate decay of ``'adagrad'``.
        initial_accumulator_value : float, default 0.0
            Starting value of ``'adagrad'``'s sum of squares.
        max_grad_norm : float or None, default 1.0
            ``'lamb'``: clip the global gradient norm to this value before
            the update; None does not clip.
        trust_clip, always_adapt : bool, default False
            ``'lamb'``: clip the trust ratio to 1; apply the trust ratio
            even without weight decay.
        milestones : sequence of int or None, default None
            ``'multi_step'``: 1 to 8 rising epochs.
        T_max : int or None, default None
            ``'cosine'``: epochs of the half cosine (required).
        eta_min : float, default 0.0
            The lowest rate of ``'cosine'`` and ``'cosine_warm_restarts'``.
        T_0 : int or None, default None
            ``'cosine_warm_restarts'``: epochs of the first cycle (required).
        T_mult : int, default 1
            ``'cosine_warm_restarts'``: growth of each cycle.
        warmup_steps : int, default 0
            ``'linear_warmup'``: optimizer steps of the warmup.
        total_steps : int or None, default None
            ``'linear_warmup'`` and ``'one_cycle'``: the optimizer steps of
            the schedule (required). A ``'one_cycle'`` fit may not run more.
        pct_start, anneal_strategy, div_factor, final_div_factor, three_phase
            ``'one_cycle'``: as ``OneCycleLR`` (defaults 0.3, ``'cos'``, 25,
            1e4, False).
        cycle_momentum : bool, default True
            ``'one_cycle'``: PyTorch's default True also cycles the
            momentum, which this schedule does not; pass False (True is
            refused with that reason).
        factor, patience, threshold, cooldown, min_lr
            ``'reduce_on_plateau'``: as ``ReduceLROnPlateau`` (defaults 0.1,
            10, 1e-4 relative, 0, 0).

        Returns
        -------
        Sequential
            The model.

        Examples
        --------
        >>> import mojolearn
        >>> model = mojolearn.Sequential().linear(8).relu().linear(1)
        >>> model = model.compile(loss="huber", optimizer="momentum", lr=0.01, nesterov=True)
        >>> model = model.compile(loss="smooth_l1", beta=0.5, reduction="sum")
        >>> model = model.compile(optimizer="rmsprop", lr=0.01, schedule="cosine", T_max=20)
        """
        # two tuples of at most 30 values: CPython builds a longer tuple literal through a list (BUILD_LIST and
        # LIST_APPEND), work in the wrapper; the native UNPACK joins them (lane nn-integrate, 2026-11-23)
        return _SEQUENTIAL_ENTRY(self, "compile", ((loss, optimizer, lr, momentum, nesterov, beta1, beta2, eps, weight_decay, schedule, step_size, gamma, huber_delta, tol, n_iter_no_change, reduction, pos_weight, beta, blank, zero_infinity),
                                                   (alpha, centered, lr_decay, initial_accumulator_value, max_grad_norm, trust_clip, always_adapt, milestones, T_max, eta_min, T_0, T_mult, warmup_steps, total_steps, pct_start, anneal_strategy, div_factor, final_div_factor, three_phase, factor, patience, threshold, cooldown, min_lr, cycle_momentum)), _SEQUENTIAL_CONTEXT)

    # ------------------------------------------------------------ description
    def _program(self, n_features, sample=None):
        return _SEQUENTIAL_ENTRY(self, "program", (n_features, sample), _SEQUENTIAL_CONTEXT)

    def _table(self):
        return _SEQUENTIAL_ENTRY(self, "table", (), _SEQUENTIAL_CONTEXT)


    # ------------------------------------------------------------ training
    def fit(self, X, y, *, epochs=10, batch_size=None, shuffle=None, seed=0, resume=False,
            validation_data=None, restore_best_weights=False, lengths=None, target_lengths=None):
        """Train the model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features) or (n_samples, ...)
            Training inputs: a bool, integer, float16, float32 or float64
            array, used as float32. A 2-D array may have any memory layout;
            an array of 3 or more dimensions (one sample per index of the
            first axis) must be C-contiguous.
        y : array-like
            Targets: (n_samples,) or (n_samples, n_outputs) for ``mse``,
            ``l1``, ``smooth_l1``, ``huber`` and ``kl_div`` (target
            probabilities for ``kl_div``); one label per row for
            ``cross_entropy`` and ``nll`` (any number of classes, one model
            output per class) and ``bce_with_logits``
            (two classes, one model output); the ``(n_samples, length)``
            int32 next-token ids for ``token_cross_entropy``; for ``ctc``,
            the ``(n_samples, max_target_length)`` label sequences (class
            indices other than ``blank``), each padded past its target
            length with any value.
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
        lengths : array-like of int of shape (n_samples,) or None, default None
            For models with recurrent layers (:meth:`lstm`, :meth:`gru`,
            :meth:`rnn`): each sample's number of valid steps, from 1 to the
            padded length of the sequence axis. Steps past a sample's length
            are padding: their values are ignored, they do not change the
            recurrent state, and the recurrent layers' outputs there repeat
            the state at the sample's last valid step, so the last step holds
            each sample's final state. None treats every step as valid. Not combined with
            ``validation_data``. With ``ctc`` they are also the number of
            output steps each sample's alignment uses.
        target_lengths : array-like of int of shape (n_samples,) or None, default None
            ``ctc`` only (and required there): each sample's number of
            labels, from 0 to the width of ``y``. Not combined with
            ``validation_data``.

        Returns
        -------
        Sequential
            The fitted model, with ``params_``, ``loss_curve_``, ``n_iter_``
            and ``n_features_in_`` set (and ``classes_`` for classifiers).
        """
        return _SEQUENTIAL_ENTRY(self, "fit", (X, y, epochs, batch_size, shuffle, seed, resume, validation_data, restore_best_weights, lengths, target_lengths), _SEQUENTIAL_CONTEXT)

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

    def predict(self, X, lengths=None):
        """Predict with the fitted model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Inputs: a bool, integer, float16, float32 or float64 array in
            any memory layout, used as float32.
        lengths : array-like of int of shape (n_samples,) or None, default None
            For models with recurrent layers (:meth:`lstm`, :meth:`gru`,
            :meth:`rnn`): each sample's number of valid steps, as in
            :meth:`fit`. None treats every step as valid.

        Returns
        -------
        ndarray
            The model outputs, (n_samples,) or (n_samples, n_outputs) in the
            shape ``y`` had at fit; for ``cross_entropy``, ``nll`` and
            ``bce_with_logits`` models, the predicted class labels
            (n_samples,).
        """
        return _SEQUENTIAL_ENTRY(self, "predict", (X, lengths), _SEQUENTIAL_CONTEXT)

    def predict_proba(self, X, lengths=None):
        """Class probabilities of a ``cross_entropy``, ``nll`` or ``bce_with_logits`` model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Inputs: a bool, integer, float16, float32 or float64 array in
            any memory layout, used as float32.
        lengths : array-like of int of shape (n_samples,) or None, default None
            For models with recurrent layers (:meth:`lstm`, :meth:`gru`,
            :meth:`rnn`): each sample's number of valid steps, from 1 to the
            padded length of the sequence axis. Steps past a sample's length
            are padding: their values are ignored, they do not change the
            recurrent state, and the recurrent layers' outputs there repeat
            the state at the sample's last valid step, so the last step holds
            each sample's final state. None treats every step as valid.

        Returns
        -------
        ndarray of float32, shape (n_samples, n_classes)
            Each row sums to one; columns follow ``classes_``.
        """
        return _SEQUENTIAL_ENTRY(self, "predict_proba", (X, lengths), _SEQUENTIAL_CONTEXT)


# Package-import ABI connection. No model operation calls this setup again.
from ._native_runtime_sequential_bridge import connect as _connect_sequential_owner
_SEQUENTIAL_NO_KEYWORDS = {}
_SEQUENTIAL_CONTEXT = {
    "integer": np.integer, "layer_codes": _LAYER, "loss_codes": _LOSS,
    "optimizer_codes": _OPT, "schedule_codes": _SCHED,
    "reduction_codes": _REDUCTION,  # nn-losses
    # layer_names[code]: the name `model.layers` reports for each layer code (tools/block_tables_check.py checks
    # it against every LAYER_ code). 21..31 are not codes; 68..71 were never assigned (model_blocks
    # BLOCK_UNUSED_FIRST..BLOCK_UNUSED_LAST) and stay empty.
    "layer_names": ("", "linear", "relu", "sigmoid", "tanh", "softmax",
                    "layer_norm", "dropout", "residual", "reshape", "permute",
                    "flatten", "conv2d", "max_pool2d", "avg_pool2d", "conv1d", "max_pool1d",
                    "avg_pool1d", "adaptive_avg_pool1d", "branch", "concat",
                    "", "", "", "", "", "", "", "", "", "", "",                                  # 21..31
                    "silu", "gelu", "softplus", "rms_norm",                                       # 32..35
                    "rotary", "multiply", "narrow", "embedding", "attention", "recurrent",        # 36..41
                    "selective_scan", "ssd_scan", "trapezoid_scan", "lm_head", "batch_norm",      # 42..46
                    "identity", "leaky_relu", "elu", "prelu", "mish", "hardswish",                # 47..52 nn-activations
                    "hardsigmoid",                                                                # 53
                    "group_norm", "instance_norm",                                                # 54..55 nn-norms
                    "conv_transpose1d", "conv_transpose2d", "adaptive_avg_pool2d",                # 56..58 nn-conv
                    "adaptive_max_pool2d", "adaptive_max_pool1d", "global_max_pool",              # 59..61
                    "constant_pad", "reflection_pad", "replication_pad", "upsample",              # 62..65
                    "pixel_shuffle",                                                              # 66
                    "multihead_attention",                                                        # 67 nn-attention
                    "", "", "", "",                                                               # 68..71 unused
                    "positional_encoding",                                                        # 72 nn-attention
                    "conv3d", "max_pool3d", "avg_pool3d", "adaptive_avg_pool3d",                  # 73..76 nn-conv3d
                    "log_softmax",                                                                # 77 nn-losses-2
                    ),
}
# nn-conv3d begin
# layer_names is read by code (bindings/sequential_description_pack.mojo sequential_layers subscripts it), so
# the 3-D layers' names join it as a mapping, whatever form the other lanes left it in
_names = _SEQUENTIAL_CONTEXT["layer_names"]
_names = dict(_names) if isinstance(_names, dict) else dict(enumerate(_names))
_names.update({73: "conv3d", 74: "max_pool3d", 75: "avg_pool3d", 76: "adaptive_avg_pool3d"})
_SEQUENTIAL_CONTEXT["layer_names"] = _names
del _names
# nn-conv3d end




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
        "loss_names": ("mse", "cross_entropy", "bce_with_logits", "huber", "token_cross_entropy",
                       "l1", "smooth_l1", "nll", "kl_div",  # nn-losses: l1 .. kl_div
                       "ctc"),  # nn-losses-2
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
