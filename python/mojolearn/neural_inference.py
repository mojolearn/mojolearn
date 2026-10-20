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
    """Forward-only `SmallMLPTrainer` network (8 -> 16 ReLU -> 3, float32) on the CPU.

    Computes the same logits as `SmallMLPTrainer.predict_logits` from the same weights.
    The weights are copied at construction and never change; the weight matrices may
    also be given packed by `mojolearn.lowbit`, in which case they are unpacked
    exactly to float32. Each row's logits depend on that row alone.

    Parameters
    ----------
    weight1 : array of float32
        First-layer weights.
    bias1 : array of float32
        First-layer bias.
    weight2 : array of float32
        Second-layer weights.
    bias2 : array of float32
        Second-layer bias.

    Attributes
    ----------
    weight_format : str
        Format the weight matrices were given in: "float32", or the packed `mojolearn.lowbit` format.

    Examples
    --------
    >>> import mojolearn
    >>> model = mojolearn.MLPInference.from_checkpoint("mlp.ckpt")
    >>> logits = model.predict_logits(X)    # X: (rows, 8) float32, 1..256 rows
    """

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
        """Build from a `SmallMLPTrainer.save_checkpoint` file.

        The file's size, schema, SHA-256 and arithmetic profile are checked (any mismatch
        raises `ValueError`); the optimizer state is validated and dropped.

        Parameters
        ----------
        path : str or path-like
            Checkpoint file.

        Returns
        -------
        MLPInference
        """
        with Path(path).open("rb") as stream:
            encoded = stream.read(_FILE_LIMIT + 1)
        if len(encoded) > _FILE_LIMIT:
            raise ValueError("MLPInference checkpoint exceeds its size bound")
        try:
            envelope = json.loads(encoded, object_pairs_hook=_unique_object)
        except (ValueError, UnicodeDecodeError, RecursionError) as exc:
            raise ValueError("MLPInference checkpoint is not valid bounded JSON") from exc
        if not isinstance(envelope, dict) or set(envelope) not in ({"schema", "payload", "payload_sha256"}, {"schema", "payload", "payload_sha256", "arithmetic_profile"}):
            raise ValueError("MLPInference checkpoint envelope mismatch")
        if envelope["schema"] != _FILE_SCHEMA:
            raise ValueError("MLPInference checkpoint schema mismatch")
        if hashlib.sha256(_canonical(envelope["payload"])).hexdigest() != envelope["payload_sha256"]:
            raise ValueError("MLPInference checkpoint integrity mismatch")
        expected_profile = str(_binding().neural_arithmetic_profile())
        if envelope.get("arithmetic_profile", "mojolearn.neural-training.fp32.v1") != expected_profile:
            raise ValueError("MLPInference checkpoint arithmetic profile mismatch")
        weights, _, _, _ = _validate_state(_decode_state(envelope["payload"]))
        return cls(*weights)

    @property
    def weights_(self):
        """An independent copy of the four weights, keyed by name."""
        return {name: w.copy() for name, w in zip(_NAMES, self._weights)}

    def predict_logits(self, X):
        """Logits for `X`.

        Parameters
        ----------
        X : array of float32, shape (rows, 8)
            Input rows, 1 to 256 of them.

        Returns
        -------
        mojolearn.Array of float32, shape (rows, 3)
        """
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


# Nothing is overridden except the binding (lane/stateful-cpu-decoding): the host binding
# exports transformer_forward and transformer_decode_step, so the parent's _call is the
# arithmetic (decode = prefill entry at L = 1 with the cache carried);
# tools/step_vs_full_check.py measures step-vs-full bitwise agreement on this column.
class TransformerBlockInference(TransformerBlock):
    """Forward-only `TransformerBlock` on the CPU, from weights trained on a GPU.

    Takes the same constructor arguments as `TransformerBlock` (the nine named
    weights, `n_heads`, `n_kv_heads`, `head_dim`, `window`) and applies the same
    weight checks. Supports `forward(x, lengths=None)` from a zero state,
    `allocate_state(B, max_tokens)`, `forward(x, state)` and `step(x, state)`;
    decoding one token at a time with a carried state gives the same bits as one
    forward pass over the whole sequence. A block has no output head, so it returns
    hidden states, not logits. Runs on the host binding shipped with mojolearn,
    loaded on first use.

    Examples
    --------
    >>> blk = mojolearn.TransformerBlockInference(weights, n_heads=4)
    >>> y = blk.forward(x)
    """

    def _extension(self):
        return _binding()

    def backward(self, x, grad_output):
        """Not available: inference is forward only. Always raises `NotImplementedError`;
        train with the GPU block class.
        """
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


# Shared base _RecurrentBlockInference: only the binding and backward are overridden; the
# binding exports mamba{1,2,3}_forward / mamba{1,2,3}_decode_step, so the parent's _call is
# the arithmetic; tools/step_vs_full_check.py checks step-by-step decode against prefill.
class Mamba1BlockInference(_RecurrentBlockInference, Mamba1Block):
    """Forward-only `Mamba1Block` on the CPU, from weights trained on a GPU.

    Takes the same constructor arguments as `Mamba1Block` (its ten named weights) and
    applies the same weight checks. Supports `forward(x, lengths=None)` from a zero
    state, `allocate_state(B)`, `forward(x, state)` and `step(x, state)`; decoding
    one token at a time with a carried state gives the same bits as one forward pass
    over the whole sequence. `backward` raises `NotImplementedError`. Runs on the
    host binding shipped with mojolearn; it is loaded on first use, so an install
    without it still imports and raises when the class is used.
    """


class Mamba2BlockInference(_RecurrentBlockInference, Mamba2Block):
    """Forward-only `Mamba2Block` on the CPU, from weights trained on a GPU.

    Takes the same constructor arguments as `Mamba2Block` (its nine named weights and `dt_limit`, default ``(0.0, inf)``) and
    applies the same weight checks. Supports `forward(x, lengths=None)` from a zero
    state, `allocate_state(B)`, `forward(x, state)` and `step(x, state)`; decoding
    one token at a time with a carried state gives the same bits as one forward pass
    over the whole sequence. `backward` raises `NotImplementedError`. Runs on the
    host binding shipped with mojolearn; it is loaded on first use, so an install
    without it still imports and raises when the class is used.
    """


class Mamba3BlockInference(_RecurrentBlockInference, Mamba3Block):
    """Forward-only `Mamba3Block` on the CPU, from weights trained on a GPU.

    Takes the same constructor arguments as `Mamba3Block` (its nine named weights) and
    applies the same weight checks. Supports `forward(x, lengths=None)` from a zero
    state, `allocate_state(B)`, `forward(x, state)` and `step(x, state)`; decoding
    one token at a time with a carried state gives the same bits as one forward pass
    over the whole sequence. `backward` raises `NotImplementedError`. Runs on the
    host binding shipped with mojolearn; it is loaded on first use, so an install
    without it still imports and raises when the class is used.
    """


# ---------------------------------------------------------------- Samba

# Arithmetic is the same host functions the reference bindings call; tools/identity_break.py's
# mlp / transformer / transformer-window / *-decode lanes measure it on a CPU column.
class SambaInference:
    """Forward-only `SambaStack` language model on the CPU.

    Maps ``(B, L)`` integer token ids to ``(B, L, vocab)`` float32 logits with the
    stack's forward (no dropout): embedding, each block in stack order, final RMSNorm
    and the tied or untied output head. Build it from a `SambaStack.save_checkpoint`
    file with `from_checkpoint`, or from a `SambaConfig` and a dict of weights keyed by
    the config's registry names. Weights may be packed by `mojolearn.lowbit`; they are
    unpacked exactly to float32 and copied at construction. `allocate_state`,
    ``forward(ids, state)`` and `step` carry a decode state. There is no loss,
    backward or training: `loss` and `train_step` raise `NotImplementedError`.

    Parameters
    ----------
    config : SambaConfig
        Model configuration; anything else raises `TypeError`.
    weights : dict
        float32 arrays keyed by registry name. Missing or unknown names, wrong shapes
        and non-finite values raise `ValueError`.

    Attributes
    ----------
    config : SambaConfig
        The configuration.
    weight_format : str
        Format the weights were given in: "float32", or the packed `mojolearn.lowbit` format.

    Examples
    --------
    >>> import mojolearn
    >>> lm = mojolearn.SambaInference.from_checkpoint("samba.ckpt")
    >>> logits = lm.forward(ids)                  # ids: (B, L) integers
    >>> st = lm.allocate_state(batch_size=1, max_tokens=256)
    >>> logits = lm.forward(prompt, st)
    >>> next_logits = lm.step(next_ids, st)       # (B, vocab)
    """

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
        """Build from a `SambaStack.save_checkpoint` file (the streamed archive or the older JSON format).

        The checkpoint's integrity checks run, and its schema, profile and weight registry
        must match its config, otherwise `ValueError`. Optimizer, schedule and random
        state are validated and dropped.

        Parameters
        ----------
        path : str or path-like
            Checkpoint file.

        Returns
        -------
        SambaInference
        """
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
        """An independent copy of every weight, keyed by registry name."""
        return {n: a.copy() for n, a in self._w.items()}

    @staticmethod
    def _ids(x):
        pb = probe(x)
        if len(pb.shape) != 2:
            raise ValueError("mojolearn.SambaInference: inputs must be (B, L) integer ids")
        return as_i32_c(x, ndim=2, name="inputs")[0]

    def forward(self, inputs, state=None, *, lengths=None):
        """Logits for token ids; also available as `logits` and by calling the model.

        With ``state=None`` every block starts from a zero state. Pass a state from
        `allocate_state` to carry it: each block's state is read and updated in place, so
        a later `forward` or `step` on that state continues the sequence.

        Parameters
        ----------
        inputs : array of int, shape (B, L)
            Token ids in [0, vocab).
        state : SambaState or None, default None
            Decode state to read and update.
        lengths : sequence of int or None, default None
            Ragged batch: row i is real at positions ``[0, lengths[i])``. Stateless
            forward only; with a `state` it raises `ValueError`.

        Returns
        -------
        mojolearn.Array of float32, shape (B, L, vocab)
        """
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
        """A zero decode state for `batch_size` sequences of up to `max_tokens` positions.

        Holds one block state per layer, in stack order, as `SambaStack.allocate_state`
        does. Both arguments must be positive, otherwise `ValueError`.

        Parameters
        ----------
        batch_size : int
            Number of sequences.
        max_tokens : int
            Maximum total positions per sequence.

        Returns
        -------
        SambaState
        """
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
        """Decode one token per row and update `state` in place.

        A step's logits equal the logits a one-shot `forward` gives at that position.

        Parameters
        ----------
        inputs : array of int, shape (B,) or (B, 1)
            Token ids.
        state : SambaState
            Required decode state from `allocate_state`.

        Returns
        -------
        mojolearn.Array of float32, shape (B, vocab)
        """
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
        """Not available: inference only. Raises `NotImplementedError`; use `SambaStack`."""
        self._refuse("loss")

    def train_step(self, inputs, targets):
        """Not available: inference only. Raises `NotImplementedError`; use `SambaStack`."""
        self._refuse("train_step")

    def __repr__(self):
        return "SambaInference(layers=%r, d_model=%d, vocab=%d)" % (
            list(self.config.layers), self.config.d_model, self.config.vocab)


# Import-time connection metadata for the causal native owner. This helper is
# never called by a model operation and cannot choose a GPU arithmetic route.
def _causal_native_connections():
    entries = {}
    try:
        neural = _binding()
    except (ImportError, RuntimeError, ValueError, OSError) as exc:
        neural = str(exc)
    entries.update(transformer=neural, mamba1=neural, mamba2=neural, mamba3=neural, primitives=neural)
    try:
        entries["linalg"] = _backend.load_host_module("_mojolearn_linalg_host")
    except (ImportError, RuntimeError, ValueError, OSError) as exc:
        entries["linalg"] = str(exc)
    return entries
