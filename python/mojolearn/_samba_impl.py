# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""A Samba-shaped stack on the GPU: embedding -> [Mamba3Block |
TransformerBlock]* (each block carries its own pre-norm and residual) ->
final RMSNorm -> tied or untied LM head -> cross-entropy, with a full
backward, AdamW with clipping, a learning-rate schedule, clause 9.2 gradient
accumulation, a position-keyed RNG and streamed checksummed checkpoints.

PRIVATE MODULE, AND IT HOLDS NO NUMERICS. Every arithmetic step is a call
into `_mojolearn_training` (embedding, RMSNorm, head GEMM, accumulate, RNG,
loss, optimizer), `_mojolearn_mamba` (Mamba3Block forward/backward) or
`_mojolearn_transformer` (TransformerBlock forward and IDENTICAL zero-state
prefill backward). Attention layers use full causal attention, and each
backward recomputes its saved layer input from a fresh cache. This file
decides the ORDER of the tensor registry (part of the clipped answer), the
order of the blocks, and what goes in a checkpoint.

The tensor registry, in order, is the clip's cross-tensor summation order:

    embed.weight
    layers.{i}.<block tensor names, in the block class's _W_NAMES order>
    norm_f.weight
    lm_head.weight            only when tie_embeddings is False
"""

from . import _buffer as _buffers, _bufcheck as _checks
from ._array import Array as _Array
import json
from pathlib import Path

from . import _portable_math as math
from ._training_impl import _round_f32

from . import _backend
from . import _native_neural_bridge
from . import _numeric_profile
from . import _ragged
from . import _training_impl as T
from ._mamba_impl import Mamba3Block
from ._transformer_impl import TransformerBlock

__all__ = ["SambaConfig", "SambaDecodeSession", "SambaStack", "SambaState"]

PROFILE = "mojolearn.samba-stack.fp32.v1"
_STATE_SCHEMA = "mojolearn.samba-stack-state.v1"
_CHECKPOINT_SCHEMA = "mojolearn.samba-stack-json-checkpoint.v1"
_CHECKPOINT_LIMIT = 256 * 1024 * 1024

_M3_HEADDIM = 64
_M3_EXPAND = 2
_M3_D_STATE = 128
_M3_NGROUPS = 1
_M3_NUM_ROPE_ANGLES = 32

_LAYER_KINDS = ("mamba3", "attention")


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False).encode("ascii")


class SambaConfig(object):
    """Shape of a `SambaStack`: vocabulary, width and the sequence of layer kinds.

    Parameters
    ----------
    vocab : int
        Vocabulary size, >= 2.
    d_model : int
        Model dimension, >= 1. Must be a multiple of 32 when any layer is
        'mamba3', and equal ``n_heads * head_dim`` when any layer is 'attention'.
    layers : sequence of {'mamba3', 'attention'}
        Layer kinds in stack order; at least one.
    n_heads : int or None, default None
        Attention heads. Required when any layer is 'attention'.
    n_kv_heads : int or None, default None
        Key/value heads for grouped-query attention. None uses `n_heads`.
    head_dim : int or None, default None
        Attention head dimension. None uses ``d_model // n_heads``.
    intermediate : int or None, default None
        Hidden size of the attention layers' gated MLP. Required when any layer is
        'attention'.
    tie_embeddings : bool, default True
        Share the embedding matrix with the output head.
    norm_eps : float, default 1e-5
        Epsilon of the final RMSNorm (>= 0). Each block keeps its own constants.
    dropout : float, default 0.0
        Dropout rate after the embedding during `SambaStack.train_step`, in [0, 1).

    Invalid values raise `ValueError`.

    Examples
    --------
    >>> import mojolearn
    >>> cfg = mojolearn.SambaConfig(vocab=256, d_model=64, layers=("mamba3", "attention"),
    ...                             n_heads=4, intermediate=128)
    """

    def __init__(self, vocab, d_model, layers, n_heads=None, n_kv_heads=None,
                 head_dim=None, intermediate=None, tie_embeddings=True,
                 norm_eps=1e-5, dropout=0.0):
        return _native_neural_bridge.RUNTIME.samba_config_call(
            "initialize", (self, (vocab, d_model, layers, n_heads, n_kv_heads,
                                   head_dim, intermediate, tie_embeddings,
                                   norm_eps, dropout)), _native_samba_bridge.CONTEXT)

    def to_dict(self):
        """The configuration as a plain dict (the inverse of `from_dict`)."""
        return _native_neural_bridge.RUNTIME.samba_config_call("dict", (self,), _native_samba_bridge.CONTEXT)

    @classmethod
    def from_dict(cls, d):
        """Build a configuration from a dict made by `to_dict`."""
        return _native_neural_bridge.RUNTIME.samba_config_call("from_dict", (cls, d), _native_samba_bridge.CONTEXT)

    # -- the registry ---------------------------------------------------
    def block_shapes(self, kind):
        """Names and shapes of one block's tensors.

        Parameters
        ----------
        kind : {'mamba3', 'attention'}
            Layer kind.

        Returns
        -------
        list of (str, tuple)
            ``(name, shape)`` pairs in the block's weight order.
        """
        return _native_neural_bridge.RUNTIME.samba_config_call("block_shapes", (self, kind), _native_samba_bridge.CONTEXT)

    def registry(self):
        """Names and shapes of every parameter tensor of the stack, in order.

        The order is ``embed.weight``, then ``layers.{i}.<block tensor>`` for each layer,
        then ``norm_f.weight``, then ``lm_head.weight`` when embeddings are not tied.
        It is also the summation order of gradient clipping.

        Returns
        -------
        list of (str, tuple)
        """
        return _native_neural_bridge.RUNTIME.samba_config_call("registry", (self,), _native_samba_bridge.CONTEXT)


class SambaState(object):
    """The stack's decode state (2026-09-15): `layers[i]` is layer i's own
    block state (`Mamba3State` or `TransformerState`), caller-owned and laid
    out as its block class documents, for `batch_size` rows and up to
    `max_tokens` positions. `SambaStack.forward(inputs, state)` and
    `SambaStack.step` update every piece in place."""

    def __init__(self, batch_size, max_tokens, layers):
        return _native_samba_bridge.RUNTIME.samba_native_call(
            "state_construct", (self, batch_size, max_tokens, layers), _native_samba_bridge.CONTEXT)


# Accumulation follows optimizer contract clause 9.2 (leaf size, T mod L, A divides P, A a power of
# two). The embedding backward is a run-sorted fold over each row's occurrences, not a leaf tree,
# so A > 1 can change embed.weight's last bits; tests/test_samba_surface.py records it as "no claim".
# Holds no numerics: every arithmetic step calls _mojolearn_training / _mojolearn_mamba /
# _mojolearn_transformer (see the module docstring).
class SambaStack(object):
    """A Samba-style language model with its own training loop.

    Embedding, then a stack of `Mamba3Block` and `TransformerBlock` layers (each with
    its own pre-norm and residual), a final RMSNorm, a tied or untied LM head and the
    cross-entropy loss, with a full backward pass. Training uses AdamW over the
    parameter registry (`SambaConfig.registry`), optional gradient-norm clipping, an
    optional learning-rate schedule and gradient accumulation. All parameters live
    in one flat float32 buffer. Attention layers use full causal attention.

    Pass exactly one of `weights` or `generator` to supply the initial values;
    otherwise `ValueError`.

    With ``accumulation_steps > 1`` the batch rows are split into microbatches whose
    gradients are combined in a fixed tree. A split the accumulation rule does not
    allow is refused with `ValueError` before any gradient is computed. An allowed
    split reproduces the unsplit step's block, final-norm and LM-head gradients bit
    for bit, but not the embedding gradient's, which is summed per token occurrence:
    after a step with accumulation the embedding can differ from the unsplit step in
    the last bits.

    Parameters
    ----------
    config : SambaConfig
        Model shape. Anything else raises `TypeError`.
    weights : dict of str to float32 buffer or None, default None
        Initial parameters keyed by the registry names, with exact names and shapes.
    generator : Generator or None, default None
        Seeded `mojolearn.training.Generator` used to initialize the parameters
        and to draw dropout masks.
    lr : float, default 1e-3
        Learning rate, used when `lr_schedule` is None.
    betas : tuple of float, default (0.9, 0.999)
        AdamW moment decays.
    eps : float, default 1e-8
        AdamW denominator term.
    weight_decay : float, default 0.01
        AdamW decoupled weight decay.
    lr_schedule : schedule or None, default None
        For example `WarmupCosineLR` or `WarmupLinearLR`.
    max_norm : float or None, default None
        Clip the global gradient L2 norm to this value; None turns clipping off.
    accumulation_steps : int, default 1
        Number of microbatches per optimizer step; must divide the batch rows.
    numeric_mode : {'identical', 'fast'} or None, default None
        Numeric mode. None uses the library default. ``'fast'`` is available on Apple GPUs only.

    Attributes
    ----------
    config : SambaConfig
    names : list of str
        Registry names in order.
    optimizer : AdamW
        The optimizer.

    Examples
    --------
    >>> import numpy as np, mojolearn
    >>> from mojolearn.training import Generator, WarmupCosineLR
    >>> cfg = mojolearn.SambaConfig(vocab=256, d_model=64, layers=("mamba3", "mamba3"))
    >>> stack = mojolearn.SambaStack(cfg, generator=Generator(7), lr=1e-3,
    ...                              lr_schedule=WarmupCosineLR(1e-3, 8, 64),
    ...                              max_norm=1.0, accumulation_steps=2)
    >>> ids = np.random.default_rng(0).integers(0, 256, (4, 32)).astype(np.int32)
    >>> out = stack.train_step(ids[:, :-1], ids[:, 1:])
    >>> stack.save_checkpoint("run.ckpt")
    """

    def __init__(self, config, weights=None, generator=None, lr=1e-3,
                 betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01,
                 lr_schedule=None, max_norm=None, accumulation_steps=1,
                 numeric_mode=None):
        return _native_samba_bridge.RUNTIME.samba_native_call("construct", (self, (config, weights, generator, lr, betas, eps, weight_decay, lr_schedule, max_norm, accumulation_steps, numeric_mode)), _native_samba_bridge.CONTEXT)

    # -- weights ----------------------------------------------------------
    def load_weights(self, weights):
        """Copy parameters in from a dict keyed by registry name.

        The key set and every shape must match exactly; values must be float32.
        """
        return _native_samba_bridge.RUNTIME.samba_native_call("load_weights", (self, weights), _native_samba_bridge.CONTEXT)

    def parameters(self):
        """The parameters as ``{name: array}``, views of the flat buffer in registry order."""
        return _native_samba_bridge.RUNTIME.samba_native_call("parameters", (self,), _native_samba_bridge.CONTEXT)

    # -- forward ------------------------------------------------------------
    def forward(self, inputs, state=None, *, lengths=None):
        """Compute logits for a batch of token ids, without dropout.

        Parameters
        ----------
        inputs : int array, shape (B, L)
            Token ids.
        state : SambaState or None, default None
            None runs every block from a zero state. Pass a state from
            `allocate_state` to carry it: each block's state is read at entry and
            updated in place, so a later `forward` or `step` continues the sequence.
        lengths : sequence of int or None, default None
            Makes the batch ragged: B integers in [1, L]. Row i is real at positions
            [0, lengths[i]) and padding after. Each real position's logits equal the row
            run alone at its own length, and every padding position's logits are exactly
            +0.0. Not allowed together with `state`.

        Returns
        -------
        logits : mojolearn.Array, shape (B, L, vocab)
        """
        return _native_samba_bridge.RUNTIME.samba_native_call(
            "forward", (self, inputs, state, lengths), _native_samba_bridge.CONTEXT)

    # -- decode (2026-09-15) --------------------------------------------------
    def allocate_state(self, batch_size, max_tokens):
        """Return a zero decode state for `batch_size` sequences of up to `max_tokens` positions.

        The state holds one block state per layer in stack order (`Mamba3State` or
        `TransformerState`), owned by the caller.

        Parameters
        ----------
        batch_size : int
        max_tokens : int

        Returns
        -------
        SambaState
        """
        return _native_samba_bridge.RUNTIME.samba_native_call(
            "allocate_state", (self, batch_size, max_tokens), _native_samba_bridge.CONTEXT)

    def step(self, inputs, state):
        """Decode one token per row, updating `state` in place.

        The stateful `forward` at L = 1. No dropout.

        Parameters
        ----------
        inputs : int array, shape (B,) or (B, 1)
            Token ids.
        state : SambaState
            Decode state from `allocate_state`.

        Returns
        -------
        logits : mojolearn.Array, shape (B, vocab)
        """
        return _native_samba_bridge.RUNTIME.samba_native_call(
            "step", (self, inputs, state), _native_samba_bridge.CONTEXT)

    def decode_session(self, state):
        """Open a `SambaDecodeSession` for fast token-by-token decoding.

        On a GPU the session keeps the parameters, every layer's state and the
        hidden state between layers on the device across steps; its `step` gives
        the same bytes as `SambaStack.step`. The session copies the parameters and
        the state when it opens:

        - until `close()` or `sync_state()`, the state's arrays are stale, and
          `forward`/`step` on this stack refuse that state;
        - parameter changes after opening (training steps, `load_weights`) are not
          seen; close and open a new session;
        - edits to the state arrays are not seen until `load_state()`.

        Parameters
        ----------
        state : SambaState
            Decode state from `allocate_state`.

        Returns
        -------
        SambaDecodeSession

        Examples
        --------
        >>> state = stack.allocate_state(batch_size=2, max_tokens=64)
        >>> with stack.decode_session(state) as session:
        ...     for ids_t in tokens:
        ...         logits = session.step(ids_t)
        """
        return _native_samba_bridge.RUNTIME.samba_native_call(
            "decode_session_open", (self, state), _native_samba_bridge.CONTEXT)

    def loss(self, inputs, targets):
        """Mean cross-entropy of `targets` given `inputs`, both (B, L) ids; no dropout, no gradient."""
        return _native_samba_bridge.RUNTIME.samba_native_call(
            "loss", (self, inputs, targets), _native_samba_bridge.CONTEXT)

    # -- backward -----------------------------------------------------------
    def _refuse_no_backward(self):
        for i, kind in enumerate(self.config.layers):  # glue: checks each layer kind
            if kind == "attention" and not callable(getattr(TransformerBlock, "backward", None)):
                raise NotImplementedError(
                    "mojolearn.SambaStack: layer %d is an attention block and "
                    "loaded TransformerBlock wrapper has no callable backward; "
                    "install a matching wrapper and IDENTICAL transformer "
                    "extension. The layer gradient cannot be skipped "
                    "(python/mojolearn/_samba_impl.py)" % i)

    def loss_and_grads(self, inputs, targets, num_items=None,
                       dropout_stream=None, token_offset=0):
        """Run one forward and the full backward pass.

        Parameters
        ----------
        inputs, targets : int array, shape (B, L)
            Token ids and next-token targets. Shapes must match.
        num_items : int or None, default None
            Divisor of the summed loss and gradients. None uses this call's own number of
            targets; pass the whole step's count to make a microbatch's gradient its
            slice of the unsplit one.
        dropout_stream : int or None, default None
            Generator stream for the dropout mask; None applies no dropout.
        token_offset : int, default 0
            Position offset of this batch in the dropout stream.

        Returns
        -------
        loss : float
        grads : list of mojolearn.Array
            One gradient per parameter, in registry order.
        """
        return _native_samba_bridge.RUNTIME.samba_native_call(
            "model_gradients", (self, inputs, targets, num_items,
                                dropout_stream, token_offset), _native_samba_bridge.CONTEXT)

    def train_step(self, inputs, targets):
        """Run one optimizer step on a batch.

        The batch rows are split into `accumulation_steps` microbatches; each one's
        gradient uses the whole step's divisor and the pieces are combined in a fixed
        tree before clipping and the AdamW update. B must be divisible by
        `accumulation_steps`, otherwise `ValueError`.

        Parameters
        ----------
        inputs, targets : int array, shape (B, L)
            Token ids and next-token targets.

        Returns
        -------
        dict
            ``loss``, ``lr``, ``step`` and ``total_norm`` (the gradient norm before
            clipping, or None when `max_norm` is None).
        """
        return _native_samba_bridge.RUNTIME.samba_native_call(
            "model_train_step", (self, inputs, targets), _native_samba_bridge.CONTEXT)

    # -- state ----------------------------------------------------------------
    def _experiment_profile(self):
        """Native numerical-contract metadata; no tensor work in this shell."""
        try:
            profile = getattr(T._load(self.numeric_mode), "training_experiment_profile", None)
        except (ImportError, AttributeError):
            profile = None
        return str(profile()) if callable(profile) else "baseline"

    def state_dict(self, *, _copy_arrays=True):
        """The complete training state as a dict.

        Holds the configuration, parameter registry and values, optimizer moments and
        step, schedule, generator state and numeric mode, so `load_state_dict` resumes
        training exactly.
        """
        return _native_samba_bridge.RUNTIME.samba_native_call(
            "snapshot", (self, _copy_arrays), _native_samba_bridge.CONTEXT)

    def load_state_dict(self, state):
        """Restore a state made by `state_dict`; returns self.

        Raises `ValueError` when the schema, configuration, numeric profile or sizes do
        not match this stack.
        """
        return _native_samba_bridge.RUNTIME.samba_native_call(
            "publish", (self, state), _native_samba_bridge.CONTEXT)

    # -- checkpoint (streamed arrays; legacy JSON remains readable) --------
    _ARRAYS = (("parameters", "<f4"), ("exp_avg", "<f4"),
               ("exp_avg_sq", "<f4"), ("buf_initialized", "<i4"))

    def save_checkpoint(self, path):
        """Write the training state to `path` as a checksummed checkpoint.

        The file is written atomically. Saving reads the live parameter and optimizer
        storage, so do not train or modify the stack while it saves.
        """
        return _native_samba_bridge.RUNTIME.samba_native_call(
            "checkpoint_save", (self, path), _native_samba_bridge.CONTEXT)

    @classmethod
    def from_checkpoint(cls, path, numeric_mode=None):
        """Build a stack from a checkpoint written by `save_checkpoint`.

        Parameters
        ----------
        path : str or path-like
        numeric_mode : str or None, default None
            None uses the mode stored in the checkpoint.

        Returns
        -------
        SambaStack

        Raises `ValueError` when the checkpoint fails its integrity or schema checks.
        """
        return _native_samba_bridge.RUNTIME.samba_native_call(
            "checkpoint_load", (cls, path, numeric_mode), _native_samba_bridge.CONTEXT)

    @classmethod
    def _read_legacy_checkpoint(cls, path):
        return _native_samba_bridge.RUNTIME.samba_native_call(
            "checkpoint_read", (path,), _native_samba_bridge.CONTEXT)


class SambaDecodeSession(object):
    """Token-by-token decoding on one `SambaStack` and one `SambaState`.

    Made by `SambaStack.decode_session(state)`. Every output is byte for byte what
    `SambaStack.step` gives on the same stack and state. On a GPU the parameters,
    every layer's state and the hidden state between layers stay on the device
    across steps, so a step uploads the token ids and downloads the logits only.
    A session is not thread-safe. Use it as a context manager to close it
    automatically.

    Attributes
    ----------
    state : SambaState
        The state this session decodes from.
    is_open : bool
        Whether the session is open.

    Examples
    --------
    >>> with stack.decode_session(state) as session:
    ...     for ids_t in tokens:
    ...         logits = session.step(ids_t)
    """

    # Every method is one native call (bindings/samba_decode_native.mojo); the
    # resident device session, the open flag and the row count live in the
    # instance record the native door writes.
    def __init__(self, stack, state):
        return _native_samba_bridge.RUNTIME.samba_native_call(
            "decode_session_init", (self, stack, state), _native_samba_bridge.CONTEXT)

    @property
    def state(self):
        """The `SambaState` this session decodes from."""
        return _native_samba_bridge.RUNTIME.samba_native_call(
            "decode_session_state", (self,), _native_samba_bridge.CONTEXT)

    @property
    def is_open(self):
        """Whether the session is open."""
        return _native_samba_bridge.RUNTIME.samba_native_call(
            "decode_session_is_open", (self,), _native_samba_bridge.CONTEXT)

    @property
    def _native(self):
        return _native_samba_bridge.RUNTIME.samba_native_call(
            "decode_session_device", (self,), _native_samba_bridge.CONTEXT)

    def step(self, inputs):
        """Decode one token per row.

        Parameters
        ----------
        inputs : int array, shape (B,) or (B, 1)
            Token ids; B must equal the session's batch size.

        Returns
        -------
        logits : mojolearn.Array, shape (B, vocab)
        """
        return _native_samba_bridge.RUNTIME.samba_native_call(
            "decode_session_step", (self, inputs), _native_samba_bridge.CONTEXT)

    def decode(self, inputs):
        """Decode `N` tokens per row in one call.

        The logits are byte for byte what `N` calls to `step` give, token by
        token, and the state ends where they leave it.

        Parameters
        ----------
        inputs : int array, shape (B, N)
            Token ids; B must equal the session's batch size.

        Returns
        -------
        logits : mojolearn.Array, shape (B, N, vocab)

        Examples
        --------
        >>> with stack.decode_session(state) as session:
        ...     logits = session.decode(ids)
        """
        return _native_samba_bridge.RUNTIME.samba_native_call(
            "decode_session_decode", (self, inputs), _native_samba_bridge.CONTEXT)

    def sync_state(self):
        """Copy the session's current state into the `SambaState` arrays and return the state."""
        return _native_samba_bridge.RUNTIME.samba_native_call(
            "decode_session_sync", (self,), _native_samba_bridge.CONTEXT)

    def load_state(self):
        """Reload the session from the `SambaState` arrays (after you edit them) and return the state."""
        return _native_samba_bridge.RUNTIME.samba_native_call(
            "decode_session_load", (self,), _native_samba_bridge.CONTEXT)

    def close(self):
        """Copy the state back, release the session's buffers and close it. Closing twice is a no-op."""
        return _native_samba_bridge.RUNTIME.samba_native_call(
            "decode_session_close", (self,), _native_samba_bridge.CONTEXT)

    def __enter__(self):
        return _native_samba_bridge.RUNTIME.samba_native_call(
            "decode_session_enter", (self,), _native_samba_bridge.CONTEXT)

    def __exit__(self, *exc):
        return _native_samba_bridge.RUNTIME.samba_native_call(
            "decode_session_exit", (self,), _native_samba_bridge.CONTEXT)

    def __del__(self):
        return _native_samba_bridge.RUNTIME.samba_native_call(
            "decode_session_drop", (self,), _native_samba_bridge.CONTEXT)

    def __repr__(self):
        return _native_samba_bridge.RUNTIME.samba_native_call(
            "decode_session_repr", (self,), _native_samba_bridge.CONTEXT)


# Import-time native ABI connection; legacy public methods remain until their
# complete supported operation has a native owner and whole-call audit.
from . import _native_samba_bridge
_native_samba_bridge.RUNTIME = _native_neural_bridge.RUNTIME
_native_samba_bridge.CONTEXT = _native_samba_bridge.connect(
    SambaConfig, SambaStack, T, _native_neural_bridge, SambaState, SambaDecodeSession)
