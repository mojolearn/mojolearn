# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Public neural INFERENCE on the CPU from weights trained on a GPU
(lane/inference-tokenizer-neural, 2026-09-15).

Two forward-only classes over the shipped host binding
`mojolearn/host/_mojolearn_neural_host.so` (`bindings/build_neural_host.sh`),
loaded by path like the byte LM's, the forest's and the tokenizer's:

  `MLPInference`                the small 8-16-3 MLP's logits, from a
                                `SmallMLPTrainer` checkpoint or its four
                                weights
  `TransformerBlockInference`   `TransformerBlock.forward` (full causal or a
                                sliding window, ragged `lengths` included),
                                `allocate_state` and `step`, from the block's
                                nine named weights
  `Mamba1BlockInference`,       the Mamba blocks' `forward` (ragged
  `Mamba2BlockInference`,       `lengths` included), `allocate_state` and
  `Mamba3BlockInference`        `step`, from their named weights
                                (lane/inference-neural-forward)
  `SambaInference`              `SambaStack.forward`'s logits, plus
                                `allocate_state` and `step`, from a
                                `SambaStack.save_checkpoint` file or a
                                config and its registry's weights

INCREMENTAL DECODING IS PUBLIC HERE SINCE lane/stateful-cpu-decoding
(2026-09-16) and it is not a second model. A carried state, a `step` and a
cache allocation were refused by name until then because the binding
exported the fresh entries alone; it now exports the state-carrying ones as
well, and each class reaches its GPU parent's own `_call`, so prefill and
decode are ONE spelling. `tools/step_vs_full_check.py` measures on the CPU
column that decoding a sequence one token at a time with a carried state is
BITWISE the same sequence run as one fresh-state forward pass, position by
position, and fires with the first differing position and both values when
one carried cell is moved by one ULP.

None trains, and none can: no optimizer, loss or backward is exported by
the binding. Training stays on a GPU (and, for internal
verification only, on the source reference host bindings). The arithmetic is
the same host functions the reference bindings call for the same steps, so
the answer is meant to be the GPU columns' bits; `tools/identity_break.py`'s
`mlp`, `transformer`, `transformer-window` and `*-decode` lanes measure that
on a CPU column, where their held-out and batch cells run through these
classes.

The binding is resolved on first use, so an install without it still
imports and raises BY NAME when touched.
"""
import hashlib
import json
from pathlib import Path

from . import _backend
from . import _ragged
from . import lowbit as _lowbit
from ._buffer import addr, addr_ro, all_finite, empty
from ._mlp_impl import (
    _FILE_LIMIT, _FILE_SCHEMA, _NAMES, _SHAPES, _array, _batch, _canonical,
    _decode_state, _unique_object, _validate_state,
)
from ._transformer_impl import TransformerBlock
from ._mamba_impl import Mamba1Block, Mamba2Block, Mamba3Block, _f32_strict
from ._arrays import _addr, _addr_ro
from ._bufcheck import le_bytes, probe
from ._buffer import as_i32_c, frombytes

_EXTENSION = "_mojolearn_neural_host"

__all__ = ["MLPInference", "TransformerBlockInference", "Mamba1BlockInference",
           "Mamba2BlockInference", "Mamba3BlockInference", "SambaInference"]


def _binding():
    return _backend.load_host_module(_EXTENSION)


class MLPInference:
    """Forward-only `SmallMLPTrainer` (8 -> 16 ReLU -> 3, float32) on the
    CPU. The weights are copied at construction and never change.

    `predict_logits(X)` is `SmallMLPTrainer.predict_logits` from the same
    weights: `X` is `(rows, 8)` float32 with 1..256 rows, the answer a
    `(rows, 3)` float32 `mojolearn.Array`. Each row's logits depend on that
    row alone (the GEMM profile's cells are row-independent)."""

    def __init__(self, weight1, bias1, weight2, bias2):
        # lane/identical-lowbit-inference (2026-09-17): the two matrices may
        # arrive packed (mojolearn.lowbit); materialized exactly, fp32 after.
        self.weight_format = _lowbit.format_of([weight1, weight2])
        weight1 = _lowbit.materialize_one(weight1, "weight1")
        weight2 = _lowbit.materialize_one(weight2, "weight2")
        self._weights = [_array(value, shape, name) for value, shape, name in
                         zip((weight1, bias1, weight2, bias2), _SHAPES, _NAMES)]
        self._m = _binding()

    @classmethod
    def from_checkpoint(cls, path):
        """The weights of a `mojolearn.small-mlp-checkpoint.v1` file
        (`SmallMLPTrainer.save_checkpoint`), after the trainer's own size,
        schema and SHA-256 checks. The optimizer state is validated and
        dropped."""
        with Path(path).open("rb") as stream:
            encoded = stream.read(_FILE_LIMIT + 1)
        if len(encoded) > _FILE_LIMIT:
            raise ValueError("MLPInference checkpoint exceeds its size bound")
        try:
            envelope = json.loads(encoded, object_pairs_hook=_unique_object)
        except (ValueError, UnicodeDecodeError, RecursionError) as exc:
            raise ValueError("MLPInference checkpoint is not valid bounded JSON") from exc
        if not isinstance(envelope, dict) or set(envelope) != {"schema", "payload", "payload_sha256"}:
            raise ValueError("MLPInference checkpoint envelope mismatch")
        if envelope["schema"] != _FILE_SCHEMA:
            raise ValueError("MLPInference checkpoint schema mismatch")
        if hashlib.sha256(_canonical(envelope["payload"])).hexdigest() != envelope["payload_sha256"]:
            raise ValueError("MLPInference checkpoint integrity mismatch")
        weights, _, _, _ = _validate_state(_decode_state(envelope["payload"]))
        return cls(*weights)

    @property
    def weights_(self):
        """An independent copy of the four weights, by name."""
        return {name: w.copy() for name, w in zip(_NAMES, self._weights)}

    def predict_logits(self, X):
        x = _batch(X)
        rows = int(x.shape[0])
        out = empty((rows, 3), "<f4")
        w1, b1, w2, b2 = self._weights
        wrote = int(self._m.mlp_forward_logits(
            [addr_ro(x, name="X"), addr_ro(w1, name="weight1"), addr_ro(b1, name="bias1"),
             addr_ro(w2, name="weight2"), addr_ro(b2, name="bias2"), addr(out, name="logits")],
            [rows]))
        if wrote != rows * 3 or not all_finite(out):
            raise RuntimeError("MLPInference returned an invalid result")
        return out

    def __repr__(self):
        return "MLPInference(architecture=[8, 16, 3])"


class TransformerBlockInference(TransformerBlock):
    """Forward-only `TransformerBlock` on the CPU: the same constructor
    (`weights` by the nine names, `n_heads`, `n_kv_heads`, `head_dim`,
    `window`) and the same weight checks, and the block's whole INFERENCE
    surface over the shipped neural binding -- `forward(x, lengths=None)`
    from a zero state, `allocate_state(B, max_tokens)`, `forward(x, state)`
    and `step(x, state)`. There is no output head in a block, so there are
    no logits to expose; a model's logits are the caller's projection of
    this output. Only `backward` is refused by name: the binding exports no
    VJP.

    NOTHING IS OVERRIDDEN HERE EXCEPT THE BINDING (lane/stateful-cpu-
    decoding, 2026-09-16). Until that lane this class carried its own
    `forward` plus three refusals, and the refusals were honest -- the
    binding exported `transformer_forward_fresh` alone. It now exports
    `transformer_forward` and `transformer_decode_step` as well, so the
    parent's own `_call` is the arithmetic, which is the point: a decode
    step is the prefill entry at L = 1 with the cache carried, ONE
    spelling, and `tools/step_vs_full_check.py` measures on this column
    that the two agree bitwise at every position."""

    def _extension(self):
        return _binding()

    def backward(self, x, grad_output):
        raise NotImplementedError(
            "mojolearn TransformerBlockInference.backward: inference is forward only; "
            "the backward pass runs on TransformerBlock over a training binding")


# ---------------------------------------------------------------- Mamba (lane/inference-neural-forward, 2026-09-15)

class _RecurrentBlockInference:
    """Forward-only Mamba block on the CPU: the GPU class's constructor and
    weight checks, and the block's whole INFERENCE surface over the shipped
    neural binding -- `forward(x, lengths=None)` from a zero state,
    `allocate_state(B)`, `forward(x, state)` and `step(x, state)`. Only
    `backward` is refused by name: the binding exports no VJP.

    NOTHING IS OVERRIDDEN HERE EXCEPT THE BINDING AND `backward`
    (lane/stateful-cpu-decoding, 2026-09-16). Until that lane each class
    reached an eleven- or twelve-address `*_forward_fresh` entry of its own
    and refused `step`, `allocate_state` and a carried state, and the
    refusals were honest -- the binding exported no state-carrying entry.
    It now exports `mamba{1,2,3}_forward` and `mamba{1,2,3}_decode_step`
    with the block classes' own contracts, so the parent's `_call` is the
    arithmetic. The three block oracles run prefill and decode through ONE
    call site (a decode step is the same oracle at l = 1 carrying the
    state), and `tools/step_vs_full_check.py` measures on this column that
    a step-by-step decode is bitwise the one-shot prefill at every
    position."""

    def _extension(self):
        return _binding()

    def backward(self, x, grad_output):
        raise NotImplementedError(
            f"mojolearn {type(self).__name__}.backward: inference is forward only; "
            "the backward pass runs on the GPU block class over a training binding")


class Mamba1BlockInference(_RecurrentBlockInference, Mamba1Block):
    """Forward-only `Mamba1Block` on the CPU, from its ten named weights."""


class Mamba2BlockInference(_RecurrentBlockInference, Mamba2Block):
    """Forward-only `Mamba2Block` on the CPU, from its nine named weights and
    `dt_limit` (default `(0.0, inf)`)."""


class Mamba3BlockInference(_RecurrentBlockInference, Mamba3Block):
    """Forward-only `Mamba3Block` on the CPU, from its nine named weights."""


# ---------------------------------------------------------------- Samba

class SambaInference:
    """Forward-only `SambaStack` on the CPU: `(B, L)` ids in, `(B, L, vocab)`
    float32 logits out, from a `SambaStack.save_checkpoint` file
    (`from_checkpoint`) or a `SambaConfig` and the registry's weights by name.
    The arithmetic is the stack's training forward with no dropout: the
    embedding gather, each block's forward in stack order
    (`Mamba3BlockInference`, `TransformerBlockInference`), the final RMSNorm
    and the tied or untied head, each over the shipped neural binding. The
    weights are copied at construction. `allocate_state(B, max_tokens)`,
    `forward(ids, state)` and `step(ids, state)` carry the decode state
    (lane/stateful-cpu-decoding, 2026-09-16); no optimizer, loss or backward
    is reachable."""

    def __init__(self, config, weights):
        from ._samba_impl import SambaConfig
        if not isinstance(config, SambaConfig):
            raise TypeError("mojolearn.SambaInference: config must be a SambaConfig")
        if not hasattr(weights, "keys"):
            raise TypeError("mojolearn.SambaInference: weights must be a dict keyed by the registry names")
        # lane/identical-lowbit-inference (2026-09-17): packed registry
        # matrices (mojolearn.lowbit) materialized exactly, fp32 path after.
        weights, self.weight_format = _lowbit.unpack(weights, "SambaInference")
        shapes = dict(config.registry())
        missing = [n for n in shapes if n not in weights]
        extra = [n for n in weights if n not in shapes]
        if missing or extra:
            raise ValueError("mojolearn.SambaInference: weight dict mismatch; missing %r, unknown %r"
                             % (missing, extra))
        self.config = config
        self._w = {}
        for n, shape in config.registry():
            a = _f32_strict(weights[n], "SambaInference", n)
            if tuple(a.shape) != tuple(shape):
                raise ValueError("mojolearn.SambaInference: %s has shape %r, want %r" % (n, tuple(a.shape), tuple(shape)))
            if not all_finite(a):
                raise ValueError("mojolearn.SambaInference: %s is not finite" % n)
            self._w[n] = frombytes(le_bytes(a, "f"), "<f4", tuple(shape))
        c = config
        self._blocks = []
        for i, kind in enumerate(c.layers):
            w = {n: self._w["layers.%d.%s" % (i, n)] for n, _ in c.block_shapes(kind)}
            if kind == "mamba3":
                self._blocks.append(Mamba3BlockInference(w))
            else:
                self._blocks.append(TransformerBlockInference(w, n_heads=c.n_heads, n_kv_heads=c.n_kv_heads,
                                                              head_dim=c.head_dim))

    @classmethod
    def from_checkpoint(cls, path):
        """The config and parameters of a `SambaStack.save_checkpoint` file
        (the streamed v2 archive or the legacy JSON), after the stack's own
        integrity checks. The optimizer, schedule and RNG state are
        validated by the reader and dropped."""
        from ._samba_checkpoint import MAGIC, load
        from ._samba_impl import PROFILE, SambaConfig, SambaStack, _STATE_SCHEMA
        with Path(path).open("rb") as stream:
            streamed = stream.read(len(MAGIC)) == MAGIC
        payload = load(path) if streamed else SambaStack._read_legacy_checkpoint(path)
        if payload.get("schema") != _STATE_SCHEMA or payload.get("profile") != PROFILE:
            raise ValueError("mojolearn.SambaInference: checkpoint schema/profile mismatch")
        config = SambaConfig.from_dict(payload["config"])
        want = config.registry()
        entries = payload["registry"]
        if [(e["name"], tuple(e["shape"])) for e in entries] != [(n, tuple(s)) for n, s in want]:
            raise ValueError("mojolearn.SambaInference: checkpoint registry differs from its config's")
        from ._bufcheck import flat_view
        flat = flat_view(payload["parameters"], "f")
        weights = {}
        for e in entries:
            weights[e["name"]] = frombytes(
                bytes(flat[e["offset"]:e["offset"] + e["size"]].cast("B")), "<f4", tuple(e["shape"]))
        return cls(config, weights)

    def parameters(self):
        """An independent copy of every weight, by registry name."""
        return {n: a.copy() for n, a in self._w.items()}

    @staticmethod
    def _ids(x):
        pb = probe(x)
        if len(pb.shape) != 2:
            raise ValueError("mojolearn.SambaInference: inputs must be (B, L) integer ids")
        return as_i32_c(x, ndim=2, name="inputs")[0]

    def forward(self, inputs, state=None, *, lengths=None):
        """`(B, L)` ids in, `(B, L, vocab)` float32 logits out.

        `state=None` is the stateless forward, every block from a zero
        state. Pass a `SambaState` (`allocate_state`) to carry the decode
        state instead: every block's own state is read at entry and updated
        in place, so a later `forward` or `step` on that state continues the
        sequence (lane/stateful-cpu-decoding, 2026-09-16). `lengths` applies
        to the stateless forward only."""
        what = "SambaInference.forward"
        ids = self._ids(inputs)
        if lengths is not None:
            if state is not None:
                raise ValueError(f"mojolearn {what}: lengths= applies to the stateless "
                                 "forward only; pass state=None")
            return _ragged.ragged_forward(self.forward, ids, None, lengths, "<i4", what)[0]
        return self._run(ids, state, False, what)

    logits = forward
    __call__ = forward

    def allocate_state(self, batch_size, max_tokens):
        """The zero decode state for `batch_size` sequences of up to
        `max_tokens` positions: one block state per layer, in stack order
        (`Mamba3BlockInference.allocate_state(B)`,
        `TransformerBlockInference.allocate_state(B, max_tokens)`). Every
        piece is caller-owned and documented by its block class, exactly as
        `SambaStack.allocate_state` hands them out."""
        from ._samba_impl import SambaState
        b, smax = int(batch_size), int(max_tokens)
        if b < 1 or smax < 1:
            raise ValueError("mojolearn.SambaInference.allocate_state: batch_size "
                             "and max_tokens must be positive")
        layers = []
        for kind, blk in zip(self.config.layers, self._blocks):
            layers.append(blk.allocate_state(b) if kind == "mamba3"
                          else blk.allocate_state(b, smax))
        return SambaState(b, smax, layers)

    def step(self, inputs, state):
        """One decode token per row: `(B,)` or `(B, 1)` ids in, `(B, vocab)`
        float32 logits out, `state` updated in place. It is the stateful
        `forward` at L = 1 with each block's `step`; the embedding, the
        final RMSNorm and the head are the same three binding entries the
        stateless forward calls, and they are per-token, so a step's logits
        are the logits the one-shot forward writes at that position."""
        what = "SambaInference.step"
        if state is None:
            raise ValueError("mojolearn.SambaInference.step: state is required "
                             "(allocate_state(B, max_tokens) makes the fresh one)")
        pb = probe(inputs)
        if len(pb.shape) not in (1, 2) or (len(pb.shape) == 2 and pb.shape[1] != 1):
            raise ValueError("mojolearn.SambaInference.step: inputs must be (B,) or "
                             "(B, 1) integer ids")
        rows = int(pb.shape[0])
        ids = as_i32_c(inputs, ndim=None, name="inputs")[0].reshape((rows, 1))
        out = self._run(ids, state, True, what)
        return out.reshape((rows, self.config.vocab))

    def _run(self, ids, state, step, what):
        """The stack's arithmetic: the embedding gather, each block in stack
        order, the final RMSNorm and the tied or untied head. ONE spelling
        for the stateless forward, the stateful forward and the decode step
        -- the only difference is what each block is handed."""
        from ._samba_impl import SambaState
        c = self.config
        b, l = int(ids.shape[0]), int(ids.shape[1])
        if b < 1 or l < 1:
            raise ValueError(f"mojolearn {what}: B and L must be positive")
        if ids.min() < 0 or ids.max() >= c.vocab:
            raise ValueError(f"mojolearn {what}: inputs must be in [0, vocab)")
        if state is not None:
            if not isinstance(state, SambaState):
                raise TypeError(f"mojolearn {what}: state must be a SambaState (allocate_state)")
            if state.batch_size != b or len(state.layers) != len(c.layers):
                raise ValueError(
                    f"mojolearn {what}: the state holds {state.batch_size} rows and "
                    f"{len(state.layers)} layers, the call has B = {b} and the stack "
                    f"{len(c.layers)} layers")
        ext = _binding()
        n, d = b * l, c.d_model
        flat_ids = ids.reshape((n,))
        x = empty((n, d), "<f4")
        emb = self._w["embed.weight"]
        ext.embedding_forward([_addr(x), _addr_ro(emb), _addr_ro(flat_ids)], [n, c.vocab, d])
        x = x.reshape((b, l, d))
        for i, blk in enumerate(self._blocks):
            if state is None:
                x = blk.forward(x)
            elif step:
                x = blk.step(x, state.layers[i])
            else:
                x = blk.forward(x, state.layers[i])
        xf = x.reshape((n, d))
        hn = empty((n, d), "<f4")
        ext.rms_norm_forward([_addr(hn), _addr_ro(xf), _addr_ro(self._w["norm_f.weight"])], [n, d, c.norm_eps])
        head = self._w["embed.weight" if c.tie_embeddings else "lm_head.weight"]
        logits = empty((n, c.vocab), "<f4")
        ext.linear_forward([_addr(logits), _addr_ro(hn), _addr_ro(head)], [n, c.vocab, d])
        return logits.reshape((b, l, c.vocab))

    def _refuse(self, name):
        raise NotImplementedError(
            f"mojolearn SambaInference.{name}: inference is forward only; losses, "
            "backward and training run on SambaStack")

    def loss(self, inputs, targets):
        self._refuse("loss")

    def train_step(self, inputs, targets):
        self._refuse("train_step")

    def __repr__(self):
        return "SambaInference(layers=%r, d_model=%d, vocab=%d)" % (
            list(self.config.layers), self.config.d_model, self.config.vocab)
