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
from ._byte_lm_config import ByteLanguageModelConfig
from . import _ragged

_EXTENSION = '_mojolearn_byte_lm_host'
_MODULE_NAME = 'mojolearn._host.' + _EXTENSION


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


# The shape, thread-count and parameter-dict admission (`_native_shape`,
# `_thread_count`, `_flatten_parameters`) run in the native host trainer door; their
# dead Python twins are gone (lane py-lm-neural, 2026-11-14).

#: torch's `ignore_index`, which the loss oracle skips as a target.
_IGNORE_INDEX = -100


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
        return _HOST_TRAIN(cls, "inference_from_checkpoint", (path, threaded, threads), _HOST_TRAIN_CONTEXT)

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



# TOMBSTONE: LanguageModelHostTrainer deleted 2026-11-22 by lane neural-consolidate: one CPU training
# step is LanguageModelTrainer(device="cpu") (the same composed host step, byte_host_train_native, bit for
# bit: ~/mojolearn-evidence/briefs-20261122/neural-consolidate-work/same_kernels_before_e282.json); code
# recoverable at the lane's parent commit.


def _connect_host_trainer():
    # glue: import-time connection of the native `train_step` owner
    # (bindings/byte_lm_host_trainer_native_door.mojo); the compiled step it runs
    # is the trainer's own binding entry, retained at construction
    from .tokenizer import _binding
    from . import lowbit, _linalg_impl
    from ._byte_lm_impl import _LM_TRAINER_CONTEXT
    from ._byte_lm_config import _DEFAULT_PROFILE
    from ._native_classical_bridge import _state_context
    context = _state_context()
    _numeric_profile.default_profile("training")  # glue: connect profile metadata at import
    try:
        binding = _load()
    except (ImportError, AttributeError, ValueError, RuntimeError, TypeError, OSError) as error:
        binding = type(error).__name__ + ": " + str(error)
    context.update(byte_lm_checkpoint_context=_LM_TRAINER_CONTEXT, byte_lm_bf16_class=lowbit.BF16Weight, byte_lm_int8_class=lowbit.Int8Weight,
                   byte_lm_linalg_context=_linalg_impl._NATIVE_CONTEXT, byte_lm_default_profile=_DEFAULT_PROFILE,
                   byte_lm_host_inference_class=LanguageModelInference,
                   byte_lm_config_class=ByteLanguageModelConfig, byte_lm_host_binding=binding,
                   byte_lm_profile_defaults=_numeric_profile._defaults, byte_lm_profiles=_numeric_profile.PROFILES)
    return _binding()._module.byte_lm_host_trainer_call, context


_HOST_TRAIN, _HOST_TRAIN_CONTEXT = _connect_host_trainer()
