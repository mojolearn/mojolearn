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

Neither trains, and neither can: no optimizer, loss, backward or decode
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

_EXTENSION = "_mojolearn_neural_host"

__all__ = ["MLPInference", "TransformerBlockInference"]


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
