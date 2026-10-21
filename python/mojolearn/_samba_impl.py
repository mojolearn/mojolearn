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
import hashlib
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

__all__ = ["SambaConfig", "SambaStack", "SambaState"]

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


def _init_tensor(gen, name, shape):
    """Initial bits from the position-keyed stream: normal(0, 0.02) for the
    embedding and an untied head, torch's Linear default
    (kaiming-uniform, bound 1/sqrt(fan_in)) for every projection, ones for
    the norms / D / B_bias / C_bias, and uniform(-4, -2) for dt_bias
    (softplus of that is a dt near 0.02 to 0.13)."""
    last = name.split(".")[-1]
    if name in ("embed.weight", "lm_head.weight"):
        return gen.normal(shape, 0.0, 0.02)
    if last == "weight" and len(shape) == 2:
        return gen.kaiming_uniform(shape, fan_in=shape[1])
    if last == "dt_bias":
        return gen.uniform(shape, -4.0, -2.0)
    return _buffers.full(shape, 1.0, '<f4')




def _count_targets(y):
    """How many int32 targets are not the ignore index. The equality mask
    and its integer sum run in the core Mojo helpers (`Array.__eq__` against
    a scalar and `Array.sum`, bindings/hotpath_helpers.mojo); Python only
    subtracts (pyglue-sweep 2026-10-03: the Python scan is gone)."""
    return int(y.size) - int((y == T._IGNORE_INDEX_DEFAULT).sum())  # glue: mask and sum run in Mojo helpers


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

    def _make_block(self, i):
        kind = self.config.layers[i]
        w = {n: self.arrays["layers.%d.%s" % (i, n)]
             for n, _ in self.config.block_shapes(kind)}  # glue: dict of named tensors
        if kind == "mamba3":
            block = Mamba3Block(w, numeric_mode=self.numeric_mode)
            # lane/fam2-lm: the stack reads no report stage of its blocks.
            block._discard_reports = True
            return block
        c = self.config
        return TransformerBlock(w, n_heads=c.n_heads, n_kv_heads=c.n_kv_heads,
                                head_dim=c.head_dim, numeric_mode=self.numeric_mode)

    def _block(self, i):
        # A small number of wiring tests deliberately construct a partial
        # stack with ``__new__`` so they can substitute the arithmetic.  Keep
        # that diagnostic path able to materialize a wrapper on demand; all
        # ordinarily constructed stacks use the cache above.
        blocks = getattr(self, "_blocks", None)
        return self._make_block(i) if blocks is None else blocks[i]

    def _head_weight(self):
        return self.arrays["embed.weight" if self.config.tie_embeddings
                           else "lm_head.weight"]

    # -- the Apple FAST fused tail (lane afn-samba, 2026-10-03) ---------------
    _AFN_ENTRIES = ("samba_afn_norm_head_forward", "samba_afn_tail_train",
                    "samba_afn_embedding_backward_tied")

    def _afn(self):
        """The training binding's fused Samba entries as `{name: fn}`, or
        None. They are registered only by an Apple FAST build with
        MOJOLEARN_AFN_SAMBA_FUSE (or MOJOLEARN_AFN_SAMBA_ALL); the IDENTICAL
        binding and every other build have none, and MOJOLEARN_HOTPATH=python
        asks for the per-op reference arm. The kernels, operands and order
        are the per-op path's; what changes is that the final norm's output,
        the logits' gradient and the tied embedding pair never cross the bus
        between binding calls (training/samba_afn.mojo)."""
        cached = getattr(self, "_afn_cache", None)
        if cached is not None:
            return cached or None
        entries = None
        if _buffers.hotpath_enabled():
            binding = T._load(self.numeric_mode)
            found = {}
            for name in self._AFN_ENTRIES:  # glue: looks up the named binding entries
                try:
                    found[name] = getattr(binding, name, None)
                except (AttributeError, ImportError):
                    found[name] = None
            if all(callable(found[name]) for name in self._AFN_ENTRIES):  # glue: checks the named binding entries
                entries = found
        self._afn_cache = entries if entries is not None else {}
        return entries

    def _afn_norm_head(self, afn, h2):
        """`logits (M, V)` of the final norm then the head over `h2 (M, D)`,
        one binding call."""
        c = self.config
        h2 = T._c32(h2, "h", "samba_afn_norm_head_forward")
        nw = T._c32(self.arrays["norm_f.weight"], "weight", "samba_afn_norm_head_forward")
        hw = T._c32(self._head_weight(), "weight", "samba_afn_norm_head_forward")
        m, k = h2.shape
        n = c.vocab
        logits = _buffers.empty((m, n), '<f4')
        afn["samba_afn_norm_head_forward"](
            [T._addr(logits), T._addr_ro(h2), T._addr_ro(nw), T._addr_ro(hw)],
            [int(m), int(n), int(k), float(c.norm_eps)])
        return logits

    def _afn_tail_train(self, afn, h2, y, items):
        """`(loss, dh (M, D), d_norm_w (D,), d_head_w (V, D))` of the final
        norm, the head, the sum-reduced loss over `items` and both
        backwards, one binding call."""
        c = self.config
        h2 = T._c32(h2, "h", "samba_afn_tail_train")
        nw = T._c32(self.arrays["norm_f.weight"], "weight", "samba_afn_tail_train")
        hw = T._c32(self._head_weight(), "weight", "samba_afn_tail_train")
        m, k = h2.shape
        n = c.vocab
        loss_out = _buffers.zeros((1,), '<f4')
        row_out = _buffers.empty((m,), '<f4')
        dh = _buffers.empty((m, k), '<f4')
        dnw = _buffers.empty((k,), '<f4')
        dhw = _buffers.empty((n, k), '<f4')
        afn["samba_afn_tail_train"](
            [T._addr(loss_out), T._addr(row_out), T._addr(dh), T._addr(dnw),
             T._addr(dhw), T._addr_ro(h2), T._addr_ro(nw), T._addr_ro(hw),
             T._addr_ro(y)],
            [int(m), int(n), int(k), float(c.norm_eps),
             int(T._IGNORE_INDEX_DEFAULT), 1, int(items), 0.0])
        return float(loss_out[0]), dh, dnw, dhw

    def _afn_embedding_backward_tied(self, afn, dy2, ids1, pair):
        """`embedding_backward(dy2, ids1) + pair`, the tied gradient, one
        binding call."""
        c = self.config
        dy2 = T._c32(dy2, "dy", "samba_afn_embedding_backward_tied")
        pair = T._c32(pair, "pair", "samba_afn_embedding_backward_tied")
        n, d = dy2.shape
        dw = _buffers.empty((c.vocab, d), '<f4')
        afn["samba_afn_embedding_backward_tied"](
            [T._addr(dw), T._addr_ro(dy2), T._addr_ro(ids1), T._addr_ro(pair)],
            [int(n), int(c.vocab), int(d)])
        return dw

    # -- the device-resident forward and step (lane S1 samba-resident, 2026-10-07)
    _RESIDENT_ENTRIES = ("samba_resident_open", "samba_resident_forward",
                         "samba_resident_step")

    def _resident(self):
        """The training binding when it carries the IDENTICAL resident Samba
        entries (the IDENTICAL default since 2026-10-08; -D
        MOJOLEARN_IDN_SAMBA_RESIDENT_STEP_OFF drops them; training/samba_resident.mojo)
        and this stack can take them: no dropout (the dropout mask is the
        generator's, drawn per microbatch on the per-op route). None
        otherwise, and the per-op route below runs as before (arm B). The
        chunked-head profile (NI34) spells a different head, so a
        chunked-head build leaves the entries out
        (training/neural_identical_experiments.mojo);
        it is re-checked here so a mixed build still takes one route."""
        cached = getattr(self, "_resident_cache", None)
        if cached is not None:
            return cached or None
        binding = None
        if self.config.dropout == 0.0:
            try:
                b = T._load(self.numeric_mode)
                enabled = getattr(b, "samba_resident_enabled", None)
                ok = callable(enabled) and bool(enabled())
                if ok:
                    chunked = getattr(b, "training_chunked_lm_head_enabled", None)
                    ok = not (callable(chunked) and bool(chunked()))
                if ok and all(callable(getattr(b, n, None)) for n in self._RESIDENT_ENTRIES):  # glue: checks binding entry names
                    binding = b
            except (AttributeError, ImportError):
                binding = None
        self._resident_cache = binding if binding is not None else False
        return binding

    def _resident_session(self, binding):
        """The stack's `_SambaResidentSession` (opened once: the registry
        and its gradient on the device, the per-layer weight views). The
        host registry `self.flat` stays authoritative: every call uploads it
        and the step writes it back, so checkpoints, `parameters()` and
        `load_state_dict` are unchanged."""
        session = getattr(self, "_resident_handle", None)
        if session is None:
            c = self.config
            kinds = [0 if k == "mamba3" else 1 for k in c.layers]  # glue: layer kind codes for the binding
            session = binding.samba_resident_open(
                kinds,
                [int(c.vocab), int(c.d_model), int(c.n_heads or 0), int(c.n_kv_heads or 0),
                 int(c.head_dim or 0), int(c.intermediate or 0),
                 1 if c.tie_embeddings else 0, float(c.norm_eps), int(self.n_total)],
                T._addr_ro(self.flat))
            self._resident_handle = session
        return session

    def _forward_resident(self, binding, ids):
        """`(B, L)` ids -> `(B, L, vocab)` logits in one binding call. The
        id range check is the embedding entry's own device refusal, so no
        NumPy pass over the ids runs here."""
        c = self.config
        session = self._resident_session(binding)
        b, l = ids.shape
        logits = _buffers.empty((b * l, c.vocab), '<f4')
        binding.samba_resident_forward(
            session, [T._addr_ro(self.flat), T._addr_ro(ids), T._addr(logits)],
            [int(b), int(l)])
        return logits.reshape((b, l, c.vocab))

    def _train_step_resident(self, binding, ids, y):
        """One optimizer step in one binding call (accumulation_steps 1):
        the forward, the sum-reduced loss over the step's own target count,
        the backward and `identical_optimizer_step` with the optimizer's
        resident moments. The optimizer's scalar bookkeeping is its own
        (`_step_begin` / `_step_finish`). None when the optimizer has no
        resident moments in this build (the per-op route then runs)."""
        o = self.optimizer
        o._open_resident(binding)
        if o._res is None:
            return None
        session = self._resident_session(binding)
        b, l = ids.shape
        cfg, plist = o._step_begin(self.max_norm)
        info = _buffers.zeros((3,), '<f4')
        loss_out = _buffers.zeros((1,), '<f4')
        binding.samba_resident_step(
            session,
            [T._addr(self.flat), T._addr_ro(ids), T._addr_ro(y), T._addr_ro(o.offsets),
             T._addr(o.buf_initialized), T._addr(info), T._addr(loss_out)],
            [int(b), int(l)] + plist, int(o._res))
        # the moments moved on the device; the host copies are stale until read
        o._host_fresh = False
        total = o._step_finish(info, cfg)
        self.last_ = {"loss": float(loss_out[0]), "lr": o.lr_,
                      "step": o.t, "total_norm": total}
        return dict(self.last_)

    # -- forward ------------------------------------------------------------
    @staticmethod
    def _ids(x, what):
        pb = _checks.probe(x)
        if len(pb.shape) != 2 or not _checks.is_integer(pb.format):
            raise ValueError("mojolearn.SambaStack: %s must be (B, L) integer ids" % what)
        return _buffers.as_i32_c(x, ndim=2, name=what)[0]

    def _forward(self, inputs, dropout_stream=None, token_offset=0, head=True,
                 norm=True, retain_tapes=False):
        """The forward with every block input kept for the backward.
        `head=False` stops after the final norm (`logits` is None): the
        fused `samba_head_loss` runs the head itself. `norm=False` (with
        `head=False`) stops after the last block (`hn` is None too): the
        Apple FAST tail (`_afn`) runs the final norm itself. With `_afn`
        present and both on, the final norm and the head are one call."""
        c = self.config
        ids = self._ids(inputs, "inputs")
        b, l = ids.shape
        if ids.min() < 0 or ids.max() >= c.vocab:
            raise ValueError("mojolearn.SambaStack: inputs must be in [0, vocab)")
        x = T.embedding_forward(self.arrays["embed.weight"], ids.reshape(-1),
                                self.numeric_mode).reshape((b, l, c.d_model))
        key = None
        if c.dropout > 0.0 and dropout_stream is not None:
            x, key = self.generator.dropout(x, c.dropout,
                                            offset=token_offset * c.d_model,
                                            stream=dropout_stream)
        xs = []
        tapes = []
        for i in range(len(c.layers)):  # glue: dispatches each layer block
            xs.append(x)
            block = self._block(i)
            forward_tape = getattr(block, "_forward_with_tape", None)
            if retain_tapes and callable(forward_tape):
                x, tape = forward_tape(x)
            else:
                x, tape = block.forward(x), None
            tapes.append(tape)
        hn = None
        logits = None
        afn = self._afn() if (norm and head) else None
        if afn is not None:
            logits = self._afn_norm_head(afn, x.reshape((b * l, c.d_model)))
        elif norm:
            hn = T.rms_norm_forward(x, self.arrays["norm_f.weight"], c.norm_eps,
                                    self.numeric_mode)
            if head:
                logits = T.linear_forward(hn.reshape((b * l, c.d_model)),
                                          self._head_weight(), self.numeric_mode)
        return {"ids": ids, "key": key, "xs": xs, "tapes": tapes, "h": x, "hn": hn,
                "logits": logits}

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

    def _forward_state(self, inputs, state, step):
        c = self.config
        what = "mojolearn.SambaStack.step" if step else "mojolearn.SambaStack.forward"
        if not isinstance(state, SambaState):
            raise TypeError("%s: state must be a SambaState (allocate_state)" % what)
        ids = self._ids(inputs, "inputs")
        b, l = ids.shape
        if ids.min() < 0 or ids.max() >= c.vocab:
            raise ValueError("%s: inputs must be in [0, vocab)" % what)
        if state.batch_size != b or len(state.layers) != len(c.layers):
            raise ValueError("%s: the state holds %d rows and %d layers, the call "
                             "has B = %d and the stack %d layers"
                             % (what, state.batch_size, len(state.layers), b, len(c.layers)))
        x = T.embedding_forward(self.arrays["embed.weight"], ids.reshape(-1),
                                self.numeric_mode).reshape((b, l, c.d_model))
        for i in range(len(c.layers)):  # glue: dispatches each layer block
            blk = self._block(i)
            x = blk.step(x, state.layers[i]) if step else blk.forward(x, state.layers[i])
        hn = T.rms_norm_forward(x, self.arrays["norm_f.weight"], c.norm_eps,
                                self.numeric_mode)
        logits = T.linear_forward(hn.reshape((b * l, c.d_model)),
                                  self._head_weight(), self.numeric_mode)
        return logits.reshape((b, l, c.vocab))

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


# Import-time native ABI connection; legacy public methods remain until their
# complete supported operation has a native owner and whole-call audit.
from . import _native_samba_bridge
_native_samba_bridge.RUNTIME = _native_neural_bridge.RUNTIME
_native_samba_bridge.CONTEXT = _native_samba_bridge.connect(
    SambaConfig, SambaStack, T, _native_neural_bridge, SambaState)
