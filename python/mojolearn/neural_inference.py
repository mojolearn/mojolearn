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
  `TransformerBlockInference`   `TransformerBlock.forward` from a zero state
                                (the stateless prefill, full causal or a
                                sliding window, ragged `lengths` included),
                                from the block's nine named weights
  `Mamba1BlockInference`,       the Mamba blocks' `forward` from a zero
  `Mamba2BlockInference`,       state (ragged `lengths` included), from
  `Mamba3BlockInference`        their named weights
                                (lane/inference-neural-forward)
  `SambaInference`              `SambaStack.forward`'s logits, from a
                                `SambaStack.save_checkpoint` file or a
                                config and its registry's weights

None trains, and none can: no optimizer, loss, backward or decode
cache is exported by the binding. Training stays on a GPU (and, for internal
verification only, on the source reference host bindings). The arithmetic is
the same host functions the reference bindings call for the same steps, so
the answer is meant to be the GPU columns' bits; `tools/identity_break.py`'s
`mlp`, `transformer` and `transformer-window` lanes measure that on a CPU
column, where their held-out and batch cells run through these classes.

The binding is resolved on first use, so an install without it still
imports and raises BY NAME when touched.
"""
import hashlib
import json
from pathlib import Path

from . import _backend
from . import _ragged
from ._buffer import addr, addr_ro, all_finite, empty
from ._mlp_impl import (
    _FILE_LIMIT, _FILE_SCHEMA, _NAMES, _SHAPES, _array, _batch, _canonical,
    _decode_state, _unique_object, _validate_state,
)
from ._transformer_impl import TransformerBlock, _batch_tokens
from ._mamba_impl import Mamba1Block, Mamba2Block, Mamba3Block
from ._mamba_impl import _batch_tokens as _mamba_batch_tokens, _f32_strict
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
    `window`) and the same weight checks, and `forward(x, lengths=None)`
    from a zero state, which is `TransformerBlock.forward(x)` with
    `state=None`. There is no output head in a block, so there are no
    logits to expose; a model's logits are the caller's projection of this
    output. A carried state, `step`, `allocate_state` and `backward` are
    refused by name."""

    def _extension(self):
        return _binding()

    def forward(self, x, state=None, *, lengths=None):
        what = "TransformerBlockInference.forward"
        if state is not None:
            raise ValueError(
                f"mojolearn {what}: a carried state is not supported; this class runs "
                "the stateless prefill only (TransformerBlock carries state on a GPU)")
        x = _batch_tokens(x, what, self.d_model, False)
        ext = self._extension()
        if lengths is None:
            return self._call_fresh(x, ext)
        return _ragged.ragged_forward(lambda xp: self._call_fresh(xp, ext), x, None, lengths, "<f4", what)[0]

    def _refuse(self, name):
        raise NotImplementedError(
            f"mojolearn TransformerBlockInference.{name}: inference is the stateless "
            "forward only; decode steps, carried states and backward run on TransformerBlock")

    def step(self, x, state):
        self._refuse("step")

    def allocate_state(self, batch_size, max_tokens):
        self._refuse("allocate_state")

    def backward(self, x, grad_output):
        self._refuse("backward")


# ---------------------------------------------------------------- Mamba (lane/inference-neural-forward, 2026-09-15)

class _RecurrentBlockInference:
    """Forward-only Mamba block on the CPU: the GPU class's constructor and
    weight checks, and `forward(x, lengths=None)` from a zero state, which is
    the GPU block's `forward(x)` with `state=None` (the final state is
    discarded). Carrying a state, `step`, `allocate_state` and `backward`
    are refused by name. The binding writes only `y`."""

    _ENTRY = None

    def _extension(self):
        return _binding()

    def _params(self, b, l):
        return [b, l, self.d_model]

    def _fresh(self, x, ext):
        b, l = int(x.shape[0]), int(x.shape[1])
        y = empty((b, l, self.d_model), "<f4")
        addrs = [addr_ro(x, name="x")] + [addr_ro(w, name="weight") for w in self._w] + [addr(y, name="y")]
        wrote = int(getattr(ext, self._ENTRY)(addrs, self._params(b, l)))
        if wrote != b * l * self.d_model:
            raise RuntimeError(f"mojolearn {type(self).__name__}: the binding wrote {wrote} cells")
        return y

    def forward(self, x, state=None, *, lengths=None):
        what = type(self).__name__ + ".forward"
        if state is not None:
            raise ValueError(
                f"mojolearn {what}: a carried state is not supported; this class runs the "
                "zero-state prefill only (the GPU block carries state)")
        x = _mamba_batch_tokens(x, what, self.d_model, False)
        ext = self._extension()
        if lengths is None:
            return self._fresh(x, ext)
        return _ragged.ragged_forward(lambda xp: self._fresh(xp, ext), x, None, lengths, "<f4", what)[0]

    __call__ = forward

    def _refuse(self, name):
        raise NotImplementedError(
            f"mojolearn {type(self).__name__}.{name}: inference is the zero-state forward only; "
            "decode steps, carried states and backward run on the GPU block class")

    def step(self, x, state):
        self._refuse("step")

    def allocate_state(self, batch_size):
        self._refuse("allocate_state")

    def backward(self, x, grad_output):
        self._refuse("backward")


class Mamba1BlockInference(_RecurrentBlockInference, Mamba1Block):
    """Forward-only `Mamba1Block` on the CPU, from its ten named weights."""
    _ENTRY = "mamba1_forward_fresh"


class Mamba2BlockInference(_RecurrentBlockInference, Mamba2Block):
    """Forward-only `Mamba2Block` on the CPU, from its nine named weights and
    `dt_limit` (default `(0.0, inf)`)."""
    _ENTRY = "mamba2_forward_fresh"

    def _params(self, b, l):
        lo, hi = self.dt_limit
        return [b, l, self.d_model, lo, hi]


class Mamba3BlockInference(_RecurrentBlockInference, Mamba3Block):
    """Forward-only `Mamba3Block` on the CPU, from its nine named weights."""
    _ENTRY = "mamba3_forward_fresh"


# ---------------------------------------------------------------- Samba

class SambaInference:
    """Forward-only `SambaStack` on the CPU: `(B, L)` ids in, `(B, L, vocab)`
    float32 logits out, from a `SambaStack.save_checkpoint` file
    (`from_checkpoint`) or a `SambaConfig` and the registry's weights by name.
    The arithmetic is the stack's training forward with no dropout: the
    embedding gather, each block's zero-state forward in stack order
    (`Mamba3BlockInference`, `TransformerBlockInference`), the final RMSNorm
    and the tied or untied head, each over the shipped neural binding. The
    weights are copied at construction. No optimizer, loss, state, step or
    backward is reachable."""

    def __init__(self, config, weights):
        from ._samba_impl import SambaConfig
        if not isinstance(config, SambaConfig):
            raise TypeError("mojolearn.SambaInference: config must be a SambaConfig")
        if not hasattr(weights, "keys"):
            raise TypeError("mojolearn.SambaInference: weights must be a dict keyed by the registry names")
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
        what = "SambaInference.forward"
        if state is not None:
            raise ValueError(f"mojolearn {what}: a carried state is not supported; this class runs "
                             "the stateless forward only (SambaStack carries state)")
        ids = self._ids(inputs)
        if lengths is not None:
            return _ragged.ragged_forward(self.forward, ids, None, lengths, "<i4", what)[0]
        c = self.config
        b, l = int(ids.shape[0]), int(ids.shape[1])
        if b < 1 or l < 1:
            raise ValueError(f"mojolearn {what}: B and L must be positive")
        if ids.min() < 0 or ids.max() >= c.vocab:
            raise ValueError(f"mojolearn {what}: inputs must be in [0, vocab)")
        ext = _binding()
        n, d = b * l, c.d_model
        flat_ids = ids.reshape((n,))
        x = empty((n, d), "<f4")
        emb = self._w["embed.weight"]
        ext.embedding_forward([_addr(x), _addr_ro(emb), _addr_ro(flat_ids)], [n, c.vocab, d])
        x = x.reshape((b, l, d))
        for blk in self._blocks:
            x = blk.forward(x)
        xf = x.reshape((n, d))
        hn = empty((n, d), "<f4")
        ext.rms_norm_forward([_addr(hn), _addr_ro(xf), _addr_ro(self._w["norm_f.weight"])], [n, d, c.norm_eps])
        head = self._w["embed.weight" if c.tie_embeddings else "lm_head.weight"]
        logits = empty((n, c.vocab), "<f4")
        ext.linear_forward([_addr(logits), _addr_ro(hn), _addr_ro(head)], [n, c.vocab, d])
        return logits.reshape((b, l, c.vocab))

    logits = forward
    __call__ = forward

    def _refuse(self, name):
        raise NotImplementedError(
            f"mojolearn SambaInference.{name}: inference is the stateless forward only; decode "
            "steps, carried states, losses and training run on SambaStack")

    def step(self, inputs, state):
        self._refuse("step")

    def allocate_state(self, batch_size, max_tokens):
        self._refuse("allocate_state")

    def loss(self, inputs, targets):
        self._refuse("loss")

    def train_step(self, inputs, targets):
        self._refuse("train_step")

    def __repr__(self):
        return "SambaInference(layers=%r, d_model=%d, vocab=%d)" % (
            list(self.config.layers), self.config.d_model, self.config.vocab)
