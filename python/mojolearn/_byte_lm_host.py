# SPDX-License-Identifier: Apache-2.0
"""CPU inference for the byte-level decoder language model (DEVIATION 2610).

`LanguageModelInference.from_checkpoint(path)` loads the parameters of a
checkpoint written by `SmallByteLanguageModelTrainer.export_checkpoint` and
runs the forward pass on the CPU through `_mojolearn_byte_lm_host`, which
composes the host FP32 oracles the GPU kernels are gated against.

This module holds no arithmetic. What it promises is what the gate measured:
docs/BYTE_LM_CPU_INFERENCE.md lists the CPUs on which the loss bytes of the
retained Metal, CUDA and HIP captures were reproduced. A CPU not listed there
is not certified, whatever this code returns on it.

The binary is loaded by path and not through `_backend.load_set`, because
that selector refuses a binary whose vendor read-back is not a GPU API, and
this one reads back `cpu` by design. WHICH DIRECTORY is still
`_backend.host_module_path`'s answer, so `MOJOLEARN_HOST_DIR` moves this
door along with every other host binding. It did not until
lane/host-path-resolution (2026-09-16): this door read `mojolearn/host/`
alone, and under the override the byte LM TRAINING door opened (it goes
through `load_host_module`) while this one refused.
`MOJOLEARN_BYTE_LM_HOST_BINARY` names a different file and still wins over
the directory, which is how the gate loads its sabotage build; a sabotage
build is refused unless `MOJOLEARN_BYTE_LM_HOST_ALLOW_SABOTAGE=1`.
"""
import importlib.machinery
import importlib.util
import os
import struct
import sys
from pathlib import Path

from . import _backend
from . import _numeric_profile
from ._buffer import addr, addr_ro, all_finite, as_f32_c, as_i32_c, frombytes, zeros
from ._bufcheck import flat_view, le_bytes
from ._byte_lm_config import ByteLanguageModelConfig
from . import _ragged

_EXTENSION = '_mojolearn_byte_lm_host'
_MODULE_NAME = 'mojolearn._host.' + _EXTENSION
_IDENTICAL_CODE = 1


def _parameters_sha256(parameters):
    """SHA-256 of float32 parameters' little-endian bytes, by the Mojo SHA-256
    (core/sha256.mojo) behind the CPU tokenizer binding's `sha256_hex`."""
    return str(_backend.load_host_module('_mojolearn_tokenizer_host').sha256_hex(le_bytes(parameters, 'f')))


# THE LOADER LIVES IN `_backend` (lane/lazy-imports, 2026-10-10):
# `_buffer._host_native` needs only this binary, and importing this whole
# door from the buffer helpers put its imports in every lane's closure.
def binary_path():
    """The binary this process loads, or would load: the file
    `MOJOLEARN_BYTE_LM_HOST_BINARY` names, else this install's host binding
    of that name (`_backend.door_host_binary_path`)."""
    return _backend.door_host_binary_path(_EXTENSION)


def _load():
    """`_mojolearn_byte_lm_host`, loaded once and read back (`_backend.load_door_host`)."""
    return _backend.load_door_host(_EXTENSION)


def _native_shape(shape):
    if not isinstance(shape, ByteLanguageModelConfig):
        raise TypeError('shape must be a ByteLanguageModelConfig')
    # The config owns the ABI, including its optional chunked-head selector.
    # Both inference and trainer adapters must pass the same complete shape.
    return list(shape.native_shape)


_MAX_THREADS = 1024
#: torch's `ignore_index`, which the loss oracle skips as a target.
_IGNORE_INDEX = -100


def _refuse_ids(tokens, vocab, batch, width, target_column):
    """ValueError unless every id is a byte value in [0, vocab). When
    `target_column` is set, that column of each row feeds only the loss as a
    target and may also hold `_IGNORE_INDEX`, exactly what the native loss
    admits; every other position is a model input."""
    # Native min/max (reduce_stat) admits the common case in one pass; the
    # Python scan below runs only to name the first offending id (and to
    # admit `_IGNORE_INDEX` in the target column).
    if tokens.size and tokens.min() >= 0 and tokens.max() < vocab:
        return
    flat = flat_view(tokens, 'i')
    for r in range(batch):
        base = r * width
        for c in range(width):
            v = flat[base + c]
            if 0 <= v < vocab:
                continue
            if target_column is not None and c == target_column and v == _IGNORE_INDEX:
                continue
            raise ValueError(f'ids must be byte values in [0, {vocab}); got {v} at row {r}, position {c}')


def _logits_ids(ids, shape):
    """`(tokens, copied)` for a logits call, int32 ids `[batch, length]` with
    `batch >= 1`, `1 <= length <= shape.length` and every id a byte value,
    refused with ValueError otherwise. Shared with the trainer's GPU logits
    (DEVIATION 2658) so both surfaces admit the same ids."""
    tokens, copied = as_i32_c(ids, ndim=2, name='ids')
    batch, length = tokens.shape
    if batch <= 0 or not 0 < length <= shape.length:
        raise ValueError(f'ids must be [batch, 1..{shape.length}]')
    _refuse_ids(tokens, shape.vocab_size, batch, length, None)
    return tokens, copied


def _greedy_next_bytes(logits):
    """The greedy next byte after each row of float32 logits
    `[batch, length, vocab]`, read at the last position; ties go to the
    lowest byte value. Shared with the trainer's GPU logits (DEVIATION
    2658), so equal logits bytes pick equal bytes."""
    batch, length, vocab = logits.shape
    flat = flat_view(logits, 'f')
    if batch and vocab and hasattr(logits, '_addr'):
        # The same scan in the base binding's `argmax_rows_f32` (DEVIATION
        # 2500: strict `>` from index 0, so ties keep the lowest byte and a
        # NaN never replaces); `_labels.argmax_rows` falls back to the loop
        # below's rule when the helper is missing.
        from ._labels import argmax_rows
        last = b''.join(bytes(flat[((b * length) + length - 1) * vocab:((b * length) + length) * vocab])
                        for b in range(batch))
        rows = frombytes(last, '<f4', (batch, vocab))
        return [int(i) for i in flat_view(argmax_rows(rows), 'q')]
    result = []
    for b in range(batch):
        base = ((b * length) + length - 1) * vocab
        best = 0
        for v in range(1, vocab):
            if flat[base + v] > flat[base + best]:
                best = v
        result.append(best)
    return result


def _thread_count(value):
    """None is one thread per physical core, sent to the binding as 0."""
    if value is None:
        return 0
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError('threads must be an int or None')
    if not 1 <= value <= _MAX_THREADS:
        raise ValueError(f'threads must be in [1, {_MAX_THREADS}]')
    return value


def _flatten_parameters(params, shape):
    """The registry dict (names from `shape.parameter_names`, values float32
    or packed) to the flat float32 vector the binding takes, and the one
    format the packed tensors share. NumPy-free: the materialized tensors
    are `mojolearn.Array`s and the copy goes through the flat buffer views."""
    from . import lowbit as _lowbit
    from ._buffer import empty as _empty
    from ._bufcheck import flat_view as _flat
    names = shape.parameter_names
    missing = [n for n in names if n not in params]
    extra = [n for n in params if n not in names]
    if missing or extra:
        raise ValueError(f"mojolearn.LanguageModelInference: parameter dict mismatch; missing {missing!r}, unknown {extra!r}")
    fmt = _lowbit.format_of(params)
    flat = _empty((shape.n_total,), '<f4')
    dst = _flat(flat, 'f')
    offsets = shape.offsets
    for j, (name, shp) in enumerate(zip(names, shape.parameter_shapes)):
        a, _ = as_f32_c(_lowbit.materialize_one(params[name], name), ndim=None, name=name)
        if tuple(a.shape) != tuple(shp):
            raise ValueError(f"mojolearn.LanguageModelInference: {name} has shape {tuple(a.shape)}, want {tuple(shp)}")
        dst[offsets[j]:offsets[j + 1]] = _flat(a, 'f')
    return flat, fmt


# Forward-only byte LM on the CPU. The parameters are copied at
# construction and never change afterward.
#
# By default (`threaded=True`) calls run the threaded path (DEVIATIONS
# 2616, 2640): host kernels that spend the oracles' arithmetic in the
# oracles' order without their per-cell allocation, advanced as SIMD lanes
# and split across at most `threads` threads (None: one per physical core)
# along axes the contracts make independent. `threaded=False` runs the
# reference path, the oracles as written on one thread. Both give the same
# bits, which the gate and tools/byte_lm_host_path_sweep.py check on every
# certified CPU. Each call may override both instance defaults.
class LanguageModelInference:
    """Run a trained decoder language model on the CPU, with no GPU.

    Forward pass only: logits, loss and greedy next-token prediction for a
    model trained with :class:`~mojolearn.LanguageModelTrainer`, giving the
    same bits as the GPU forward pass. The parameters are copied at
    construction and never change.

    By default (``threaded=True``) calls use SIMD and up to ``threads`` threads
    (None: one per physical core); ``threaded=False`` runs single-threaded
    reference code. Both give the same bits. Each call can override both
    settings.

    Parameters
    ----------
    parameters : array-like or dict
        Flat float32 parameters of ``shape.n_total`` values (finite), or a dict
        of named tensors as listed by
        :meth:`LanguageModelTrainer.parameter_registry`; packed low-bit tensors
        from ``mojolearn.lowbit`` are expanded exactly.
    shape : LanguageModelConfig or None, default None
        Model dimensions; None uses the defaults.
    threaded : bool, default True
        Use the multithreaded path.
    threads : int or None, default None
        Thread count; None means one per physical core.

    Attributes
    ----------
    weight_format : str
        ``"float32"``, or the packed format of the parameters given.

    Examples
    --------
    >>> import numpy as np, mojolearn
    >>> lm = mojolearn.LanguageModelInference.from_checkpoint("model.json")
    >>> ids = np.frombuffer(b"Hello, wor", dtype=np.uint8).astype(np.int32).reshape(1, -1)
    >>> next_ids = lm.next_bytes(ids)
    """

    def __init__(self, parameters, *, shape=None, threaded=True, threads=None):
        _HOST_TRAIN(self, "inference_construct", (parameters, shape, threaded, threads), _HOST_TRAIN_CONTEXT)

    @classmethod
    def from_checkpoint(cls, path, *, threaded=True, threads=None):
        """Load the parameters of a JSON checkpoint written by the trainer.

        Reads a file written by :meth:`LanguageModelTrainer.export_checkpoint`,
        with the trainer's own integrity checks.

        Parameters
        ----------
        path : str or path-like
            The checkpoint file.
        threaded : bool, default True
            As for the constructor.
        threads : int or None, default None
            As for the constructor.

        Returns
        -------
        LanguageModelInference
        """
        from ._byte_lm_impl import _CHECKPOINT_LIMIT, _decode_checkpoint
        with Path(path).open('rb') as stream:
            encoded = stream.read(_CHECKPOINT_LIMIT + 1)
        state, shape = _decode_checkpoint(encoded)
        return cls(state['parameters'], shape=shape, threaded=threaded, threads=threads)

    def _threads_flag(self, threaded):
        value = self._threaded if threaded is None else threaded
        if not isinstance(value, bool):
            raise TypeError('threaded must be a bool or None')
        return 1 if value else 0

    def _threads_arg(self, threads):
        return _thread_count(self._threads if threads is None else threads)

    @property
    def shape(self):
        """The model dimensions, a :class:`~mojolearn.LanguageModelConfig`."""
        return _HOST_TRAIN(self, "shape", (), _HOST_TRAIN_CONTEXT)

    @property
    def profile(self):
        """The name of the arithmetic profile the CPU binding computes."""
        return _HOST_TRAIN(self, "profile", (), _HOST_TRAIN_CONTEXT)

    def parameters_sha256(self):
        """SHA-256 of the parameters' little-endian float32 bytes."""
        return _HOST_TRAIN(self, "parameters_sha256", (), _HOST_TRAIN_CONTEXT)

    def logits(self, ids, *, lengths=None, threaded=None, threads=None):
        """Float32 logits for every position.

        Parameters
        ----------
        ids : array-like of shape (batch, length), int32
            Token ids, starting at position 0; ``length`` at most
            ``shape.length``.
        lengths : sequence of int or None, default None
            Makes the batch ragged: row ``i`` is real at positions
            ``[0, lengths[i])`` and padding after, whatever the padding ids are.
            Real positions get exactly the logits of the row alone at its own
            length; padding positions are ``+0.0``.
        threaded : bool or None, default None
            Override the instance setting.
        threads : int or None, default None
            Override the instance setting.

        Returns
        -------
        Array of shape (batch, length, vocab_size), float32
        """
        return _HOST_TRAIN(self, "logits", (ids, lengths, threaded, threads), _HOST_TRAIN_CONTEXT)

    def loss_bits(self, ids, *, threaded=None, threads=None):
        """IEEE-754 bit pattern of the mean next-token loss.

        Parameters
        ----------
        ids : array-like of shape (shape.batch, shape.length + 1), int32
            The training batch layout. The last column is only a target; -100 there
            is ignored.
        threaded : bool or None, default None
            Override the instance setting.
        threads : int or None, default None
            Override the instance setting.

        Returns
        -------
        int
            The float32 loss's bits, for exact comparison.
        """
        return _HOST_TRAIN(self, "loss_bits", (ids, None, threaded, threads), _HOST_TRAIN_CONTEXT)

    def loss(self, ids, *, threaded=None, threads=None):
        """Mean next-token loss as a float; see :meth:`loss_bits`."""
        return _HOST_TRAIN(self, "inference_loss", (ids, None, threaded, threads), _HOST_TRAIN_CONTEXT)

    def next_bytes(self, ids, *, lengths=None, threaded=None, threads=None):
        """Greedy next token after each row.

        Picks the highest logit at each row's last position (with ``lengths``, its
        last real position), ties going to the lowest id.

        Parameters
        ----------
        ids : array-like of shape (batch, length), int32
            Token ids.
        lengths : sequence of int or None, default None
            Ragged row lengths, as for :meth:`logits`.
        threaded : bool or None, default None
            Override the instance setting.
        threads : int or None, default None
            Override the instance setting.

        Returns
        -------
        list of int
            One token id per row.
        """
        return _HOST_TRAIN(self, "next_bytes", (ids, lengths, threaded, threads), _HOST_TRAIN_CONTEXT)



class LanguageModelHostTrainer:
    """One byte LM training step on the CPU (DEVIATION 2680).

    Forward, backward and the AdamW update, on the reference path, through
    `byte_lm_host_train_step`. This module still holds no arithmetic; the step
    is a composition of host oracles that are each the normative answer of
    their own profile.

    NOT CERTIFIED YET, AND NOTHING HERE SAYS OTHERWISE. The CPU inference
    surface above is certified on the CPUs listed in
    docs/BYTE_LM_CPU_INFERENCE.md. This one has a gate,
    `tools/byte_lm_cpu_train_gate.py`, which compares a step's gradient, both
    Adam moments and the loss against the retained Apple, NVIDIA and AMD
    captures byte for byte, and docs/BYTE_LM_CPU_TRAINING.md records what that
    gate has actually shown. Read it before believing a number from this class.

    NO THREADED PATH. The threaded forward has no backward twin, because a
    weight gradient sums over every row of the batch and so crosses every
    thread boundary; that needs a fixed cross-thread fold, not threads
    accumulating as they finish.

    Batch composition is part of any claim made with this. Nine of the
    gradients contract over the token count, so the gradient at one batch shape
    is not the bits of the same tokens presented as two smaller batches.
    """

    def __init__(self, parameters, *, m=None, v=None, shape=None,
                 completed_steps=0, lr=1e-3, betas=(.9, .999), eps=1e-8,
                 weight_decay=.01):
        return _HOST_TRAIN(self, "construct", (parameters, m, v, shape, completed_steps,
                                               lr, betas, eps, weight_decay), _HOST_TRAIN_CONTEXT)

    def _moment(self, given, name, n):
        if given is None:
            return zeros((n,), '<f4')
        array, _ = as_f32_c(given, ndim=1, name=name)
        if array.size != n:
            raise ValueError(f'{name} must hold {n} float32 values')
        if not all_finite(array):
            raise ValueError(f'{name} must be finite')
        return frombytes(le_bytes(array, 'f'), '<f4', (n,))

    @classmethod
    def from_state(cls, parameters, m, v, *, completed_steps=0, **kwargs):
        """A trainer resuming a recorded step's starting state, which is how
        the gate replays one step of the retained capture in isolation."""
        return _HOST_TRAIN(cls, "from_state", (parameters, m, v, completed_steps, kwargs), _HOST_TRAIN_CONTEXT)

    @property
    def shape(self):
        return _HOST_TRAIN(self, "shape", (), _HOST_TRAIN_CONTEXT)

    @property
    def profile(self):
        return _HOST_TRAIN(self, "profile", (), _HOST_TRAIN_CONTEXT)

    @property
    def completed_steps(self):
        return _HOST_TRAIN(self, "completed_steps", (), _HOST_TRAIN_CONTEXT)

    @property
    def parameters_(self):
        return _HOST_TRAIN(self, "parameters", (), _HOST_TRAIN_CONTEXT)

    @property
    def m_(self):
        return _HOST_TRAIN(self, "m", (), _HOST_TRAIN_CONTEXT)

    @property
    def v_(self):
        return _HOST_TRAIN(self, "v", (), _HOST_TRAIN_CONTEXT)

    @property
    def gradient_(self):
        """The last step's gradient, or None before any step. A failed step
        leaves this None rather than a stale value."""
        return _HOST_TRAIN(self, "gradient", (), _HOST_TRAIN_CONTEXT)

    def parameters_sha256(self):
        return _HOST_TRAIN(self, "parameters_sha256", (), _HOST_TRAIN_CONTEXT)

    def train_step(self, ids):
        """One step on int32 ids `[shape.batch, shape.length + 1]`, the
        training batch layout, returning the loss's IEEE-754 bits.

        The last column is only ever a target and may hold the ignore index,
        exactly as `loss_bits` admits it. On success the parameters and both
        moments are replaced and the step count advances; if the native call
        raises, none of them move and the gradient is cleared, because a
        half-applied step is worse than a refused one."""
        return _HOST_TRAIN(self, "train_step", (ids,), _HOST_TRAIN_CONTEXT)

    def loss(self, ids):
        """`train_step`'s loss as a float. The bits are the comparable value;
        this is for reading."""
        return _HOST_TRAIN(self, "loss", (ids,), _HOST_TRAIN_CONTEXT)


def _connect_host_trainer():
    # glue: import-time connection of the native `train_step` owner
    # (bindings/byte_lm_host_trainer_native.mojo); the compiled step it runs
    # is the trainer's own binding entry, retained at construction
    from .tokenizer import _binding
    from . import lowbit, _linalg_impl
    from ._byte_lm_config import _DEFAULT_PROFILE
    from ._native_classical_bridge import _state_context
    context = _state_context()
    _numeric_profile.default_profile("training")  # glue: connect profile metadata at import
    try:
        binding = _load()
    except (ImportError, AttributeError, ValueError, RuntimeError, TypeError, OSError) as error:
        binding = type(error).__name__ + ": " + str(error)
    context.update(byte_lm_bf16_class=lowbit.BF16Weight, byte_lm_int8_class=lowbit.Int8Weight,
                   byte_lm_linalg_context=_linalg_impl._NATIVE_CONTEXT, byte_lm_default_profile=_DEFAULT_PROFILE,
                   byte_lm_host_trainer_class=LanguageModelHostTrainer,
                   byte_lm_host_inference_class=LanguageModelInference,
                   byte_lm_config_class=ByteLanguageModelConfig, byte_lm_host_binding=binding,
                   byte_lm_profile_defaults=_numeric_profile._defaults, byte_lm_profiles=_numeric_profile.PROFILES)
    return _binding()._module.byte_lm_host_trainer_call, context


_HOST_TRAIN, _HOST_TRAIN_CONTEXT = _connect_host_trainer()
