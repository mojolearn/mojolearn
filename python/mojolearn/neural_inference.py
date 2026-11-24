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
  `SambaInference`              `SambaStack.forward`'s logits, plus
                                `allocate_state` and `step`, from a
                                `SambaStack.save_checkpoint` file or a
                                config and its registry's weights
  (`LanguageModelInference` lives in `language_model`.)

A single block on the CPU is the block class itself with ``device="cpu"``
(`TransformerBlock`, `Mamba1Block`, `Mamba2Block`, `Mamba3Block`): the four
`*BlockInference` subclasses ran the same kernels and gave the same bits for
forward and decode, and were removed (lane neural-consolidate-2, 2026-11-22).
`tools/step_vs_full_check.py` measures on the CPU column that decoding a
sequence one token at a time with a carried state is BITWISE the same
sequence run as one fresh-state forward pass, position by position.

The binding is resolved on first use, so an install without it still
imports and raises BY NAME when touched.
"""
from . import _backend
from . import _native_samba_bridge
from . import lowbit as _lowbit
from ._mlp_impl import _NATIVE as _MLP_NATIVE

_EXTENSION = "_mojolearn_neural_host"

__all__ = ["MLPInference", "SambaInference"]


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
        return _MLP_NATIVE("inference_initialize", (self, weight1, bias1, weight2, bias2))

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
        return _MLP_NATIVE("inference_checkpoint", (cls, path))

    @property
    def weights_(self):
        """An independent copy of the four weights, keyed by name."""
        return _MLP_NATIVE("inference_weights", (self,))

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
        return _MLP_NATIVE("inference_predict", (self, X))

    def __repr__(self):
        return _MLP_NATIVE("inference_repr", (self,))


# glue: connect only MLPInference's native host entries and packed-weight types.
# No inference operation invokes a Python loader or constructor callback.
def _connect_mlp_inference():
    from types import BuiltinFunctionType
    entries = {}
    try:
        module = _binding()
        entries["module"] = module
        for name in ("mlp_forward_logits", "sha256_hex", "neural_arithmetic_profile"):  # glue: fixed host ABI symbols
            entry = getattr(module, name)
            if not isinstance(entry, BuiltinFunctionType):
                raise TypeError("MLPInference requires compiled " + name)
            entries[name] = entry
    except (ImportError, RuntimeError, ValueError, OSError, AttributeError, TypeError) as error:
        entries = type(error).__name__ + ": " + str(error)
    return {"class": MLPInference, "entries": entries, "lowbit": _lowbit._NATIVE_CONTEXT}


_MLP_NATIVE("inference_connect", (_connect_mlp_inference(),))


# TOMBSTONE: TransformerBlockInference, Mamba1BlockInference, Mamba2BlockInference,
# Mamba3BlockInference and their base _RecurrentBlockInference (subclasses of the GPU block
# classes that forced the CPU route and refused backward; same kernels and bits as
# Block(device="cpu")) deleted 2026-11-22 by lane neural-consolidate-2; code recoverable at
# 3b21e609f3. Use TransformerBlock(..., device="cpu") and the Mamba blocks likewise.


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
        return _native_samba_bridge.RUNTIME.samba_native_call(
            "inference_construct", (self, config, weights), _native_samba_bridge.CONTEXT)

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
        return _native_samba_bridge.RUNTIME.samba_native_call(
            "inference_checkpoint", (cls, path), _native_samba_bridge.CONTEXT)

    def parameters(self):
        """An iterator over independent copies of the weights, in registry order."""
        return _native_samba_bridge.RUNTIME.samba_native_call(
            "inference_parameters", (self, 1), _native_samba_bridge.CONTEXT)

    def named_parameters(self):
        """An iterator over ``(name, array)`` pairs of independent weight copies, in registry order."""
        return _native_samba_bridge.RUNTIME.samba_native_call(
            "inference_parameters", (self, 2), _native_samba_bridge.CONTEXT)

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
        return _native_samba_bridge.RUNTIME.samba_native_call(
            "inference_forward", (self, inputs, state, lengths), _native_samba_bridge.CONTEXT)

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
        return _native_samba_bridge.RUNTIME.samba_native_call(
            "allocate_state", (self, batch_size, max_tokens), _native_samba_bridge.CONTEXT)

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
        return _native_samba_bridge.RUNTIME.samba_native_call(
            "inference_step", (self, inputs, state), _native_samba_bridge.CONTEXT)

    def loss(self, inputs, targets):
        """Not available: inference only. Raises `NotImplementedError`; use `SambaStack`."""
        return _native_samba_bridge.RUNTIME.samba_native_call(
            "inference_refuse", ("loss",), _native_samba_bridge.CONTEXT)

    def train_step(self, inputs, targets):
        """Not available: inference only. Raises `NotImplementedError`; use `SambaStack`."""
        return _native_samba_bridge.RUNTIME.samba_native_call(
            "inference_refuse", ("train_step",), _native_samba_bridge.CONTEXT)

    def __repr__(self):
        return _native_samba_bridge.RUNTIME.samba_native_call(
            "inference_repr", (self,), _native_samba_bridge.CONTEXT)


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
