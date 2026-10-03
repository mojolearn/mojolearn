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
    """The stack's shape. `layers` is a sequence of "mamba3" / "attention"
    in stack order. `d_model` must be a multiple of 32 for a Mamba-3 layer
    (headdim 64, expand 2) and equal `n_heads * head_dim` for an attention
    layer. `norm_eps` is the FINAL norm's eps; the blocks carry their own
    profile constants."""

    def __init__(self, vocab, d_model, layers, n_heads=None, n_kv_heads=None,
                 head_dim=None, intermediate=None, tie_embeddings=True,
                 norm_eps=1e-5, dropout=0.0):
        self.vocab = int(vocab)
        self.d_model = int(d_model)
        self.layers = tuple(str(k) for k in layers)  # glue: layer kind config names
        self.n_heads = None if n_heads is None else int(n_heads)
        self.n_kv_heads = (None if n_kv_heads is None else int(n_kv_heads))
        self.head_dim = None if head_dim is None else int(head_dim)
        self.intermediate = None if intermediate is None else int(intermediate)
        self.tie_embeddings = bool(tie_embeddings)
        self.norm_eps = float(_round_f32(norm_eps))
        self.dropout = float(_round_f32(dropout))
        if self.vocab < 2 or self.d_model < 1 or not self.layers:
            raise ValueError("mojolearn.SambaConfig: vocab >= 2, d_model >= 1 "
                             "and at least one layer are required")
        for k in self.layers:  # glue: validates layer kind names
            if k not in _LAYER_KINDS:
                raise ValueError("mojolearn.SambaConfig: layer kind %r; the "
                                 "kinds are %r" % (k, _LAYER_KINDS))
        if "mamba3" in self.layers and self.d_model % (_M3_HEADDIM // _M3_EXPAND):
            raise ValueError("mojolearn.SambaConfig: a mamba3 layer needs "
                             "d_model to be a multiple of 32 (Mamba3Block)")
        if "attention" in self.layers:
            if self.n_heads is None or self.intermediate is None:
                raise ValueError("mojolearn.SambaConfig: an attention layer "
                                 "needs n_heads and intermediate")
            if self.head_dim is None:
                self.head_dim = self.d_model // self.n_heads
            if self.n_kv_heads is None:
                self.n_kv_heads = self.n_heads
            if self.n_heads * self.head_dim != self.d_model:
                raise ValueError("mojolearn.SambaConfig: d_model must equal "
                                 "n_heads * head_dim")
        if not (0.0 <= self.dropout < 1.0):
            raise ValueError("mojolearn.SambaConfig: dropout must be in [0, 1)")
        if not (self.norm_eps >= 0.0):
            raise ValueError("mojolearn.SambaConfig: norm_eps must be >= 0")

    def to_dict(self):
        return {"vocab": self.vocab, "d_model": self.d_model,
                "layers": list(self.layers), "n_heads": self.n_heads,
                "n_kv_heads": self.n_kv_heads, "head_dim": self.head_dim,
                "intermediate": self.intermediate,
                "tie_embeddings": self.tie_embeddings,
                "norm_eps": self.norm_eps, "dropout": self.dropout}

    @classmethod
    def from_dict(cls, d):
        return cls(d["vocab"], d["d_model"], d["layers"], d["n_heads"],
                   d["n_kv_heads"], d["head_dim"], d["intermediate"],
                   d["tie_embeddings"], d["norm_eps"], d["dropout"])

    # -- the registry ---------------------------------------------------
    def block_shapes(self, kind):
        """`[(name, shape)]` of one block's tensors, in the block class's
        `_W_NAMES` order."""
        dm = self.d_model
        if kind == "mamba3":
            di = _M3_EXPAND * dm
            nh = di // _M3_HEADDIM
            dip = 2 * di + 2 * _M3_NGROUPS * _M3_D_STATE + 3 * nh + _M3_NUM_ROPE_ANGLES
            shapes = {
                "block_norm.weight": (dm,), "in_proj.weight": (dip, dm),
                "dt_bias": (nh,), "B_norm.weight": (_M3_D_STATE,),
                "C_norm.weight": (_M3_D_STATE,), "B_bias": (nh, _M3_D_STATE),
                "C_bias": (nh, _M3_D_STATE), "D": (nh,),
                "out_proj.weight": (dm, di),
            }
            return [(n, shapes[n]) for n in Mamba3Block._W_NAMES]  # glue: weight names and shapes
        qw = self.n_heads * self.head_dim
        kw = self.n_kv_heads * self.head_dim
        it = self.intermediate
        shapes = {
            "input_layernorm.weight": (dm,),
            "post_attention_layernorm.weight": (dm,),
            "q_proj.weight": (qw, dm), "k_proj.weight": (kw, dm),
            "v_proj.weight": (kw, dm), "o_proj.weight": (dm, qw),
            "gate_proj.weight": (it, dm), "up_proj.weight": (it, dm),
            "down_proj.weight": (dm, it),
        }
        return [(n, shapes[n]) for n in TransformerBlock._W_NAMES]  # glue: weight names and shapes

    def registry(self):
        """`[(name, shape)]` in the clip's order."""
        out = [("embed.weight", (self.vocab, self.d_model))]
        for i, kind in enumerate(self.layers):  # glue: registry of named tensors
            out.extend(("layers.%d.%s" % (i, n), s)
                       for n, s in self.block_shapes(kind))  # glue: registry of named tensors
        out.append(("norm_f.weight", (self.d_model,)))
        if not self.tie_embeddings:
            out.append(("lm_head.weight", (self.vocab, self.d_model)))
        return out


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
        self.batch_size = int(batch_size)
        self.max_tokens = int(max_tokens)
        self.layers = list(layers)


class SambaStack(object):
    """The stack, its optimizer, schedule and RNG, over one flat float32
    parameter buffer whose registry views the blocks read directly.

        cfg = SambaConfig(vocab=256, d_model=64, layers=("mamba3", "mamba3"))
        stack = SambaStack(cfg, generator=Generator(7), lr=1e-3,
                           lr_schedule=WarmupCosineLR(1e-3, 8, 64),
                           max_norm=1.0, accumulation_steps=2)
        out = stack.train_step(inputs, targets)     # (B, L) int32 each
        stack.save_checkpoint("run.json")

    `weights` (a dict keyed by the registry names) or `generator` (a
    `Generator`) supplies the initial bits; exactly one of them. The
    optimizer is AdamW over the registry in order; `max_norm=None` turns
    the clip off. `accumulation_steps` splits the batch rows into that many
    microbatches and combines by the clause 9.2 tree; a split the clause
    does not admit is refused BY NAME before any gradient is computed. An
    admitted split reproduces the unsplit step's block, final-norm and
    `lm_head` gradients bit for bit, but NOT `embed.weight`'s: the embedding
    backward is a run-sorted fold over each row's occurrences, not a leaf
    tree, so a row seen in two microbatches sums in a different order
    (`tests/test_samba_surface.py` records it as "no claim"). The whole
    parameter buffer after an A > 1 step can therefore differ from A = 1 in
    the last bits of the embedding.
    """

    def __init__(self, config, weights=None, generator=None, lr=1e-3,
                 betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01,
                 lr_schedule=None, max_norm=None, accumulation_steps=1,
                 numeric_mode=None):
        if not isinstance(config, SambaConfig):
            raise TypeError("mojolearn.SambaStack: config must be a SambaConfig")
        if (weights is None) == (generator is None):
            raise ValueError("mojolearn.SambaStack: pass exactly one of "
                             "weights= or generator=")
        self.config = config
        self.numeric_mode = numeric_mode
        self.names = [n for n, _ in config.registry()]  # glue: registry of named tensors
        self.shapes = {n: tuple(s) for n, s in config.registry()}  # glue: registry of named tensors
        self.offsets = [0]
        for n in self.names:  # glue: offsets over named tensors
            self.offsets.append(self.offsets[-1] + int(math.prod(self.shapes[n])))  # glue: offsets over named tensors
        self.n_total = self.offsets[-1]
        self.flat = _buffers.zeros(self.n_total, '<f4')
        self.arrays = {}
        for j, n in enumerate(self.names):  # glue: views per named tensor
            self.arrays[n] = _Array.from_buffer(_checks.flat_view(self.flat, 'f')[self.offsets[j]:self.offsets[j + 1]]).reshape(self.shapes[n])
        self.generator = generator if generator is not None else T.Generator(0, numeric_mode)
        if weights is not None:
            self.load_weights(weights)
        else:
            for n in self.names:  # glue: copies each named tensor
                _checks.flat_view(self.arrays[n], 'f')[:] = _checks.flat_view(_init_tensor(self.generator, n, self.shapes[n]), 'f')
        self.max_norm = None if max_norm is None else float(max_norm)
        self.optimizer = T.AdamW([self.arrays[n] for n in self.names], lr=lr,  # glue: optimizer gets named tensors
                                 betas=betas, eps=eps, weight_decay=weight_decay,
                                 lr_schedule=lr_schedule,
                                 accumulation_steps=accumulation_steps,
                                 numeric_mode=numeric_mode)
        # The block wrappers retain views of ``self.flat``; optimizer steps
        # and checkpoint restores update that storage in place.  Constructing
        # them again for every forward/backward only repeats Python shape and
        # option validation, and also prevents TransformerBlock from reusing
        # its per-instance runtime resources.  Keep one wrapper per registry
        # block, just as SambaInference does.  No tensor is copied or cached.
        self._blocks = [self._make_block(i) for i in range(len(config.layers))]  # glue: builds one block per layer
        self.last_ = None

    # -- weights ----------------------------------------------------------
    def load_weights(self, weights):
        """Copy the registry's tensors in from a dict keyed by name, exact
        key set and exact shapes, float32 only."""
        missing = [n for n in self.names if n not in weights]  # glue: checks weight dict names
        extra = [n for n in weights if n not in self.shapes]  # glue: checks weight dict names
        if missing or extra:
            raise ValueError("mojolearn.SambaStack: weight dict mismatch; "
                             "missing %r, unknown %r" % (missing, extra))
        for n in self.names:  # glue: checks each named tensor
            a = weights[n]
            pb = _checks.probe(a)
            if not _checks.is_native_f32(pb.format):
                raise TypeError("mojolearn.SambaStack: %s has dtype %s; float32 only"
                                % (n, pb.format))
            if pb.shape != self.shapes[n]:
                raise ValueError("mojolearn.SambaStack: %s has shape %r, want %r"
                                 % (n, pb.shape, self.shapes[n]))
            a = _buffers.as_f32_c(a, ndim=None, name=n)[0]
            if not _buffers.all_finite(a):
                raise ValueError("mojolearn.SambaStack: %s is not finite" % n)
            _checks.flat_view(self.arrays[n], 'f')[:] = _checks.flat_view(a, 'f')

    def parameters(self):
        """The registry as `{name: array}` (views of the flat buffer)."""
        return {n: self.arrays[n] for n in self.names}  # glue: dict of named tensors

    def _make_block(self, i):
        kind = self.config.layers[i]
        w = {n: self.arrays["layers.%d.%s" % (i, n)]
             for n, _ in self.config.block_shapes(kind)}  # glue: dict of named tensors
        if kind == "mamba3":
            return Mamba3Block(w, numeric_mode=self.numeric_mode)
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
            for name in self._AFN_ENTRIES:
                try:
                    found[name] = getattr(binding, name, None)
                except (AttributeError, ImportError):
                    found[name] = None
            if all(callable(found[name]) for name in self._AFN_ENTRIES):
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

    # -- forward ------------------------------------------------------------
    @staticmethod
    def _ids(x, what):
        pb = _checks.probe(x)
        if len(pb.shape) != 2 or not _checks.is_integer(pb.format):
            raise ValueError("mojolearn.SambaStack: %s must be (B, L) integer ids" % what)
        return _buffers.as_i32_c(x, ndim=2, name=what)[0]

    def _forward(self, inputs, dropout_stream=None, token_offset=0, head=True,
                 norm=True):
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
        for i in range(len(c.layers)):  # glue: dispatches each layer block
            xs.append(x)
            x = self._block(i).forward(x)
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
        return {"ids": ids, "key": key, "xs": xs, "h": x, "hn": hn,
                "logits": logits}

    def forward(self, inputs, state=None, *, lengths=None):
        """`(B, L)` ids in, `(B, L, vocab)` float32 logits out, no dropout.

        `state=None` is the TRAINING forward: the same `_forward` that
        `loss` and `loss_and_grads` run, every block from a zero state.
        Pass a `SambaState` (`allocate_state`) to carry the decode state
        instead: every block's own state is read at entry and updated in
        place, so a later `forward` or `step` on that state continues the
        sequence (2026-09-15, the rlpair part of tools/identity_break.py).

        `lengths` (2026-09-15) makes the batch RAGGED: `B` integers in
        `[1, L]`, row `i` real at positions `[0, lengths[i])` and padding
        after. Every real position's logits are byte for byte the row run
        alone at its own length (the stack is a per-token embedding, causal
        blocks, a per-token norm and head; no arithmetic changes,
        `_ragged.py` says why) and every padding position's logits are
        exactly `+0.0`, whatever id the input held there. It applies to the
        stateless forward only; `lengths` with a `state` is refused."""
        if lengths is not None:
            if state is not None:
                raise ValueError("mojolearn.SambaStack.forward: lengths= applies to the "
                                 "stateless forward only; pass state=None")
            ids = self._ids(inputs, "inputs")
            return _ragged.ragged_forward(self.forward, ids, None, lengths, "<i4",
                                          "SambaStack.forward")[0]
        if state is not None:
            return self._forward_state(inputs, state, step=False)
        acts = self._forward(inputs)
        b, l = acts["ids"].shape
        return acts["logits"].reshape((b, l, self.config.vocab))

    # -- decode (2026-09-15) --------------------------------------------------
    def allocate_state(self, batch_size, max_tokens):
        """The zero decode state for `batch_size` sequences of up to
        `max_tokens` positions: one block state per layer, in stack order
        (`Mamba3Block.allocate_state(B)`, `TransformerBlock.allocate_state(B,
        max_tokens)`). Every piece is caller-owned and documented by its
        block class; nothing is hidden here."""
        b, smax = int(batch_size), int(max_tokens)
        if b < 1 or smax < 1:
            raise ValueError("mojolearn.SambaStack.allocate_state: batch_size "
                             "and max_tokens must be positive")
        layers = []
        for i, kind in enumerate(self.config.layers):  # glue: allocates state per layer
            blk = self._block(i)
            layers.append(blk.allocate_state(b) if kind == "mamba3"
                          else blk.allocate_state(b, smax))
        return SambaState(b, smax, layers)

    def step(self, inputs, state):
        """One decode token per row: `(B,)` or `(B, 1)` ids in, `(B, vocab)`
        float32 logits out, `state` updated in place. It is the stateful
        `forward` at L = 1 with each block's `step` (the blocks' one
        spelling for decode, their contracts' section on prefill
        resumption); the embedding, the final RMSNorm, the head and the
        loss arithmetic are the training primitives `_forward` calls. No
        dropout."""
        if state is None:
            raise ValueError("mojolearn.SambaStack.step: state is required "
                             "(allocate_state(B, max_tokens) makes the fresh one)")
        pb = _checks.probe(inputs)
        if not _checks.is_integer(pb.format) or len(pb.shape) not in (1, 2) \
                or (len(pb.shape) == 2 and pb.shape[1] != 1):
            raise ValueError("mojolearn.SambaStack.step: inputs must be (B,) or "
                             "(B, 1) integer ids")
        ids = _buffers.as_i32_c(inputs, ndim=None, name="inputs")[0].reshape((pb.shape[0], 1))
        out = self._forward_state(ids, state, step=True)
        return out.reshape((pb.shape[0], self.config.vocab))

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
        """Mean cross-entropy over the targets (no dropout, no gradient)."""
        acts = self._forward(inputs)
        y = self._ids(targets, "targets").reshape(-1)
        return float(T.cross_entropy(acts["logits"], y, numeric_mode=self.numeric_mode))

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
        """One forward and the full backward. Returns `(loss, grads)` with
        `grads` a list in registry order. `num_items` (an integer) makes the
        divisor of the sum-reduced loss and gradient the whole step's target
        count, which is how a microbatch's gradient equals its slice of the
        unsplit one; `None` divides by this call's own target count."""
        self._refuse_no_backward()
        c = self.config
        # The fused head (lane/py-lm): the head GEMM, the loss and the head
        # backward in one call, the logits never crossing back and forth.
        # Same kernels, operands and order as the three-call arm below,
        # which stays as the reference for a binding without the entry.
        # The Apple FAST tail (lane afn-samba): the final norm, the head, the
        # loss and both backwards in one call, then the tied embedding
        # gradient in one call. The per-op arms below are the reference.
        afn = self._afn()
        fused = afn is None and callable(T._optional_samba_head(T._load(self.numeric_mode)))
        acts = self._forward(inputs, dropout_stream, token_offset,
                             head=not fused and afn is None, norm=afn is None)
        ids = acts["ids"]
        b, l = ids.shape
        y = self._ids(targets, "targets")
        if y.shape != ids.shape:
            raise ValueError("mojolearn.SambaStack: targets must match inputs' shape")
        y = y.reshape(-1)
        count = _count_targets(y)
        items = count if num_items is None else int(num_items)
        grads = {}
        if afn is not None:
            loss, dh, grads["norm_f.weight"], dw_head = self._afn_tail_train(
                afn, acts["h"].reshape((b * l, c.d_model)), y, items)
            dh = dh.reshape((b, l, c.d_model))
        else:
            hn2 = acts["hn"].reshape((b * l, c.d_model))
            out = T.samba_head_loss(hn2, self._head_weight(), y, items,
                                    self.numeric_mode) if fused else None
            if out is not None:
                loss, dhn, dw_head = out
            else:
                logits = acts["logits"]
                if logits is None:
                    logits = T.linear_forward(hn2, self._head_weight(), self.numeric_mode)
                loss, dlogits = T.cross_entropy(logits, y, reduction="sum",
                                                num_items=items, return_grad=True,
                                                numeric_mode=self.numeric_mode)
                dhn, dw_head = T.linear_backward(dlogits, hn2, self._head_weight(),
                                                 self.numeric_mode)
            dh, grads["norm_f.weight"] = T.rms_norm_backward(
                dhn.reshape((b, l, c.d_model)), acts["h"], self.arrays["norm_f.weight"],
                c.norm_eps, self.numeric_mode)
        for i in reversed(range(len(c.layers))):  # glue: dispatches each layer backward
            g = self._block(i).backward(acts["xs"][i], dh)
            dh = g.pop("x")
            for n, v in g.items():  # glue: renames gradient dict keys
                grads["layers.%d.%s" % (i, n)] = v
        if acts["key"] is not None:
            dh = self.generator.dropout_backward(dh, acts["key"])
        if c.tie_embeddings and afn is not None:
            d_emb = self._afn_embedding_backward_tied(
                afn, dh.reshape((b * l, c.d_model)), ids.reshape(-1), dw_head)
        else:
            d_emb = T.embedding_backward(dh.reshape((b * l, c.d_model)), ids.reshape(-1),
                                         c.vocab, self.numeric_mode)
            if c.tie_embeddings:
                # The tied gradient is ONE pair add, embedding first, no
                # alignment claim (tokens=None).
                d_emb = T.accumulate_grads([d_emb, dw_head], tokens=None,
                                           numeric_mode=self.numeric_mode)
        if not c.tie_embeddings:
            grads["lm_head.weight"] = dw_head
        grads["embed.weight"] = d_emb
        return float(loss), [grads[n] for n in self.names]  # glue: orders gradients by name

    def train_step(self, inputs, targets):
        """One optimizer step over `(B, L)` inputs and targets: the batch
        rows are split into `accumulation_steps` microbatches, each
        microbatch's gradient is computed with the whole step's divisor and
        the pieces are combined by the clause 9.2 tree. Returns a dict with
        `loss`, `lr`, `step`, `total_norm`."""
        ids = self._ids(inputs, "inputs")
        y = self._ids(targets, "targets")
        if y.shape != ids.shape:
            raise ValueError("mojolearn.SambaStack: targets must match inputs' shape")
        b, l = ids.shape
        a = self.optimizer.accumulation_steps
        tokens = b * l
        if b % a != 0:
            raise ValueError("mojolearn.SambaStack.train_step: batch rows B=%d "
                             "are not divisible by accumulation_steps=%d" % (b, a))
        if a > 1 and not T.accumulation_is_aligned(tokens, a, self.numeric_mode):
            raise ValueError(
                "mojolearn.SambaStack.train_step: MISALIGNED microbatch split, "
                "T = %d tokens at A = %d does not satisfy optimizer contract "
                "clause 9.2 (leaf size, T mod L, A divides P, A a power of "
                "two); refused before any gradient is computed "
                "(python/mojolearn/_samba_impl.py)" % (tokens, a))
        count = _count_targets(y)
        stream = (self.generator.next_stream()
                  if self.config.dropout > 0.0 else None)
        rows = b // a
        losses, parts = [], []
        for k in range(a):  # glue: dispatches each microbatch binding call
            sl = slice(k * rows, (k + 1) * rows)
            loss_k, g_k = self.loss_and_grads(ids[sl], y[sl], num_items=count,
                                              dropout_stream=stream,
                                              token_offset=k * rows * l)
            losses.append(_Array.from_list([loss_k], '<f4'))
            parts.append(g_k)
        if a == 1:
            loss = float(losses[0][0])
            total = self.optimizer.step(parts[0], max_norm=self.max_norm)
        else:
            loss = float(T.accumulate_grads(losses, tokens=None,
                                            numeric_mode=self.numeric_mode)[0])
            total = self.optimizer.step_accumulated(parts, tokens,
                                                    max_norm=self.max_norm)
        self.last_ = {"loss": loss, "lr": self.optimizer.lr_,
                      "step": self.optimizer.t, "total_norm": total}
        return dict(self.last_)

    # -- state ----------------------------------------------------------------
    def state_dict(self, *, _copy_arrays=True):
        o = self.optimizer
        sched = None if o.lr_schedule is None else o.lr_schedule.config()
        return {
            "schema": _STATE_SCHEMA, "profile": PROFILE,
            "numeric_mode": _backend._CODE_MODE.get(
                T._load(self.numeric_mode).training_numeric_mode(), "unknown"),
            "config": self.config.to_dict(),
            "registry": [{"name": n, "shape": list(self.shapes[n]),
                          "offset": self.offsets[j],
                          "size": self.offsets[j + 1] - self.offsets[j]}
                         for j, n in enumerate(self.names)],  # glue: checkpoint registry of tensors
            "parameters": self.flat.copy() if _copy_arrays else self.flat,
            "exp_avg": o.exp_avg.copy() if _copy_arrays else o.exp_avg,
            "exp_avg_sq": o.exp_avg_sq.copy() if _copy_arrays else o.exp_avg_sq,
            "buf_initialized": (o.buf_initialized.copy() if _copy_arrays
                                else o.buf_initialized), "t": int(o.t),
            "optimizer": {"kind": "adamw", "lr": o.lr, "beta1": o.betas[0],
                          "beta2": o.betas[1], "eps": o.eps,
                          "weight_decay": o.weight_decay,
                          "max_norm": self.max_norm,
                          "accumulation_steps": o.accumulation_steps},
            "schedule": sched,
            "rng": self.generator.state_dict(),
        }

    def load_state_dict(self, state):
        if state.get("schema") != _STATE_SCHEMA or state.get("profile") != PROFILE:
            raise ValueError("mojolearn.SambaStack: state schema/profile mismatch")
        if state["config"] != self.config.to_dict():
            raise ValueError("mojolearn.SambaStack: state config differs from this stack's")
        # A state written under another numeric profile is refused by name; a
        # state with no field is fp32_v1, which is what this stack computes.
        _numeric_profile.check_saved(state, _numeric_profile.TRAINING_DEFAULT, "mojolearn.SambaStack state")
        p = _buffers.as_f32_c(state['parameters'], ndim=1, name='parameters')[0]
        if p.shape != (self.n_total,):
            raise ValueError("mojolearn.SambaStack: parameters hold %d floats, "
                             "the registry is %d" % (p.size, self.n_total))
        _checks.flat_view(self.flat, 'f')[:] = _checks.flat_view(p, 'f')
        self.optimizer.load_state_dict({
            "t": state["t"], "exp_avg": state["exp_avg"],
            "exp_avg_sq": state["exp_avg_sq"],
            "buf_initialized": state["buf_initialized"]})
        oc = state["optimizer"]
        self.optimizer.lr = float(oc["lr"])
        self.optimizer.betas = (float(oc["beta1"]), float(oc["beta2"]))
        self.optimizer.eps = float(oc["eps"])
        self.optimizer.weight_decay = float(oc["weight_decay"])
        self.max_norm = None if oc["max_norm"] is None else float(oc["max_norm"])
        if int(oc["accumulation_steps"]) != self.optimizer.accumulation_steps:
            raise ValueError("mojolearn.SambaStack: accumulation_steps in the "
                             "state differs from this stack's; the microbatch "
                             "count is part of the run's numerical specification")
        self.optimizer.lr_schedule = (None if state["schedule"] is None
                                      else T._Schedule.from_config(state["schedule"]))
        self.generator.load_state_dict(state["rng"])
        return self

    # -- checkpoint (streamed arrays; legacy JSON remains readable) --------
    _ARRAYS = (("parameters", "<f4"), ("exp_avg", "<f4"),
               ("exp_avg_sq", "<f4"), ("buf_initialized", "<i4"))

    def save_checkpoint(self, path):
        """Atomically stream checksummed state arrays without a total size cap.

        Saving borrows parameter and optimizer storage; callers must not train
        or mutate this stack concurrently with saving.
        """
        from ._samba_checkpoint import save
        return save(path, self.state_dict(_copy_arrays=False))

    @classmethod
    def from_checkpoint(cls, path, numeric_mode=None):
        from ._samba_checkpoint import MAGIC, load
        with Path(path).open("rb") as stream:
            streamed = stream.read(len(MAGIC)) == MAGIC
        if streamed:
            payload = load(path)
        else:
            payload = cls._read_legacy_checkpoint(path)
        if numeric_mode is None:
            numeric_mode = payload["numeric_mode"]
        config = SambaConfig.from_dict(payload["config"])
        oc = payload["optimizer"]
        sched = (None if payload["schedule"] is None
                 else T._Schedule.from_config(payload["schedule"]))
        weights = {}
        flat = _checks.flat_view(payload["parameters"], 'f')
        for entry in payload["registry"]:  # glue: views per checkpoint tensor
            weights[entry["name"]] = _Array.from_buffer(flat[
                entry["offset"]:entry["offset"] + entry["size"]]).reshape(entry["shape"])
        stack = cls(config, weights=weights, lr=oc["lr"],
                    betas=(oc["beta1"], oc["beta2"]), eps=oc["eps"],
                    weight_decay=oc["weight_decay"], lr_schedule=sched,
                    max_norm=oc["max_norm"],
                    accumulation_steps=oc["accumulation_steps"],
                    numeric_mode=numeric_mode)
        return stack.load_state_dict(payload)

    @classmethod
    def _read_legacy_checkpoint(cls, path):
        with Path(path).open("rb") as stream:
            encoded = stream.read(_CHECKPOINT_LIMIT + 1)
        if len(encoded) > _CHECKPOINT_LIMIT:
            raise ValueError("mojolearn.SambaStack: checkpoint exceeds the size limit")
        envelope = json.loads(encoded)
        if (not isinstance(envelope, dict)
                or set(envelope) != {"schema", "payload", "payload_sha256"}
                or envelope["schema"] != _CHECKPOINT_SCHEMA):
            raise ValueError("mojolearn.SambaStack: checkpoint schema mismatch")
        payload = envelope["payload"]
        if hashlib.sha256(_canonical(payload)).hexdigest() != envelope["payload_sha256"]:
            raise ValueError("mojolearn.SambaStack: checkpoint integrity mismatch")
        for key, dtype in cls._ARRAYS:  # glue: checks each checkpoint array
            d = payload[key]
            if d["dtype"] != dtype:
                raise ValueError("mojolearn.SambaStack: checkpoint tensor dtype mismatch")
            raw = bytes.fromhex(d["hex"])
            payload[key] = _buffers.frombytes(raw, dtype, d["shape"])
        return payload
