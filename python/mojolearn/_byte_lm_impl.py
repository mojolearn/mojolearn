# SPDX-License-Identifier: Apache-2.0
"""Public wrapper for a runtime-shaped decoder language model.

This module contains no forward/backward/update arithmetic. All numerical
work belongs to _mojolearn_byte_lm, or, on a process that loaded no GPU set,
to the CPU byte LM binding through `_byte_lm_trainer_host` (2026-09-15). No
learning, performance, mathematical correctness or cross-vendor identity
claim follows from this source alone.

NUMPY-FREE (numpy-free-0.7, DEVIATIONS 2428-2432). Inputs are read through
the buffer protocol (`_buffer.view`): a NumPy array, an `array.array`, a
`mojolearn.Array`. Every array this class hands back -- parameters,
gradients, the state snapshot -- is a `mojolearn.Array` (`numpy.asarray`
on it is zero-copy). THE CHECKPOINT BYTES ARE UNCHANGED: the JSON/hex
envelope, its canonical serialization and its SHA-256 are produced from
the same little-endian `<f4` / `<i4` bytes the NumPy spelling emitted
(`_bufcheck.le_bytes`), so a file written by 0.6.x loads here and a file
written here loads there, byte for byte, and the resume/compare tooling
under tools/ reads the same file (see `save_checkpoint`).

DEVICE-OWNED STEP (DEVIATION 2514). With `resident=True`
the native session's device buffers are the ONLY copy of the parameters,
the optimizer moments and the last gradient while it is open: this object's
`_state['parameters']`, `['m']` and `['v']` are `None` from the call that
opened the session until `close()`, `load_state_dict()` or a lost session.
State crosses the boundary at admission (the first call after
construction, `load_state_dict()` or `close()` hands the host state to the
native call, which validates and uploads it once), at export
(`export_state`, `export_gradients`, `export_checkpoint`, `close`) and never
per step; a step moves the ids in and the loss and the flags out. The
stateless path (`resident=False`) hands the state in and takes the new
state out on every call. Since cpu2-l11-neural (2026-10-04) `resident`
defaults to True on a GPU install (the stateless path is an explicit
opt-out and the route of a binding without session entries). Since
2026-11-06 (Andrew) `step_result` defaults to 'lean' unless `resident=False`
is given, on every vendor and the CPU column: the per-step result is the loss,
counters and flags; 'full' adds the gradients, same bytes as before.

ONE NATIVE CALL PER METHOD (lane neural-byte-lm-trainer, 2026-10-21).
`train_step`, `evaluate`, `logits`, `next_bytes`, `export_state` (also
`state_dict`, `parameters_`, `close`) and `export_gradients` each make one
call into the byte LM binding (`byte_lm_trainer_*`,
training/byte_lm_session_ops.mojo `byte_owner_*`), which admits, runs,
exports, splits the gradient by name and rolls back on failure. Python
checks the arguments, allocates the result arrays, passes the buffers and
the native session handle, and assigns what the call returned.

GPU LOGITS (DEVIATION 2658). `logits(ids)` and `next_bytes(ids)` run the
forward alone, through the native `byte_lm_logits` (stateless) and
`byte_lm_session_logits` (resident) entries, and return the numbers the
trainer's loss is computed from. They write no state on either path.
tools/byte_lm_gpu_logits_sweep.py compares them byte for byte with the CPU
reference path of DEVIATION 2610, `LanguageModelInference(threaded=False)`.
"""
from . import _bufcheck as _checks
from ._array import Array as _Array
import json
from . import _numeric_profile


class _LogitsPool:
    """Host storage for `logits()` outputs, reused across calls of one
    trainer (lane/neural-pass61, 2026-10-01). A fresh 64 MiB output per
    call cost ~30 of the L40S's ~50 ms lm-forward (zero fill plus first
    touch), the native call being ~19 ms. A store comes back here when the
    caller drops the Array it was handed, so no two live arrays ever share
    memory; the next call takes it instead of allocating. At most `keep`
    stores of the current size are held; `close()` empties it. The pool is
    native storage: `byte_lm_trainer_native_door.mojo` creates it (`keep`,
    `free`), takes and gives stores; its Python `take`/`give` twins are gone
    (lane py-lm-neural, 2026-11-14)."""


class _PooledLogits(_Array):
    """An owned logits Array whose storage returns to `_pool` when the
    last reference to the Array goes away. Every value in it is written by
    the native logits download that produced it. Its `__del__` is the
    native finalizer installed by `_connect_trainer`."""


from pathlib import Path
import threading

from . import _backend
from ._arrays import _addr, _addr_ro
from ._byte_lm_config import ByteLanguageModelConfig, state_shape
from . import _byte_lm_checkpoint
from . import _ragged

PROFILE = 'mojolearn.byte-lm.b2-l32-d32-h4-kv2-ff64-v256-blocks2.fp32.v1'
# Compatibility constants describe only the original regression fixture.
PARAMETER_NAMES = ByteLanguageModelConfig().parameter_names
PARAMETER_SHAPES = ByteLanguageModelConfig().parameter_shapes
_OFFSETS = ByteLanguageModelConfig().offsets
_N = ByteLanguageModelConfig().n_total
_EXTENSION = '_mojolearn_byte_lm'
_CHECKPOINT_SCHEMA = 'mojolearn.small-byte-lm-json-checkpoint.v1'
_CHECKPOINT_LIMIT = 2 * 1024 * 1024
_LOGITS_MAX_BATCH = 1024
_LOGITS_MAX_CELLS = 268435456


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'),
                      ensure_ascii=True, allow_nan=False).encode('ascii')


#: The tiers the byte LM ships and the code its binding reads back
#: (`byte_lm_numeric_mode`). IDENTICAL is the default; FAST (afn-lm,
#: 2026-10-03) runs the same device step with the pins on the free schedule
#: and promises quality, never bits.
_NATIVE_MODE_CODE = {'identical': 1, 'fast': 0}


def _load(shape=None):
    """The byte LM binding of the process tier for `shape` (None: the
    regression fixture), admitted natively: the profile, mode, vendor and
    runtime-shape checks run in `byte_lm_trainer_native_door.mojo`
    (`_load_snapshot`, operation "binding"). Tools and tests read it."""
    return _LM_TRAINER(None, "binding", (shape,), _LM_TRAINER_CONTEXT)


def _runtime_profile(shape, mode=None):
    """Bind shape metadata to the arithmetic of the actual loaded binary."""
    binding = _backend.binding(_EXTENSION, mode) if mode else _load()
    # The suffix is the default profile's tail past PROFILE (see `_load`).
    return shape.profile + str(binding.byte_lm_profile())[len(PROFILE):]


def _sha256_hex(data):
    """The SHA-256 of the bytes `data` as 64 hex digits, computed by the Mojo
    SHA-256 (core/sha256.mojo) behind the loaded byte LM binding's
    `sha256_hex`: the same digest as before, over the same bytes."""
    return str(_backend.binding(_EXTENSION, _backend.default_mode()).sha256_hex(data))


# The ids admission, the greedy pick, the state and schedule admission and
# the gradient outputs live in the native door (bindings/byte_lm_trainer_native_door.mojo:
# `_gradient_outputs`, the logits ids refusals, `training_schedule_json`); the
# unreached Python copies were deleted (pyout-neural-lm, 2026-11-11).


# FP32 decoder language-model trainer with runtime dimensions.
#
# Defaults B2/L32/DM32/H4/KV2/FF64/V256; pass ByteLanguageModelConfig
# as shape to configure dimensions, layer count and vocabulary.
# no biases/dropout/final norm, untied embedding/head. Supply a flat FP32
# array or the configured named tensors exposed by parameter_registry(shape). There
# is no hidden initialization, tokenizer, padding, truncation or RNG.
#
# logits(ids)/next_bytes(ids) (DEVIATION 2658) run the same device forward
# alone on int32[batch, length] IDs and change no state; the logits are the
# numbers the loss is computed from, and next_bytes picks greedily with
# ties going to the lowest byte value.
#
# train_step/evaluate require actual int32[B,L+1] IDs in [0, shape.vocab_size).
# The first L positions predict the next L; loss averages B*L targets. All arithmetic runs in the native CUDA/HIP/Metal
# profile of the process-selected tier: IDENTICAL (default, same bits on every vendor) or FAST (quality, never bits).
#
# data_schedule is a bounded caller-supplied JSON descriptor. Retain corpus
# and actual token-order SHA256, batch offsets, and planned steps there.
# next_batch_index equals completed_steps; callers must provide the matching
# next batch after restoration. This class does not fetch/reorder a corpus.
#
# NOTHING VALIDATES ITS CONTENTS, and a reader of a published run has to
# know that (lane/data-ordering-determinism, 2026-09-16). The descriptor is
# checked for size, depth, key count and JSON round-trip and for nothing
# else: no key is required, no value is compared against the IDs handed to
# train_step, and {'dataset': 'test'} trains. What it buys is that once a
# checkpoint is written the descriptor is covered by the envelope's
# payload_sha256, so it cannot be swapped without detection. That makes it
# TAMPER-EVIDENT, not TRUE. The run is reproducible from it only to the
# extent the caller made it so, and this class cannot tell the difference.
#
# resident=True retains an owned native context/model/optimizer across
# calls, and (DEVIATION 2514) the device buffers are then the ONLY copy of
# the parameters, moments and last gradient while the session is open:
# nothing is mirrored to the host per step. The state is validated once at
# admission (the first call after construction, `load_state_dict()` or
# `close()`), validated on the device after every update, and validated
# again when it is exported. `export_state()` (also `state_dict()` and
# `parameters_`), `export_gradients()` and `export_checkpoint()` copy it
# out; `close()` exports first and then releases the device, so the next
# call re-admits from host state. A step that fails after its update is
# rolled back on the device; a failure the Python layer detects after
# native success is rolled back through `byte_lm_session_rollback`. What
# is weaker than the stateless path: a LOST context (a rollback whose
# re-scan does not answer) loses the steps since the last export; the
# object then raises "session lost at step k; last exported state is step
# j" on every call until `load_state_dict()`.
#
# step_result selects what `train_step` returns. 'full' returns the
# complete dict (loss, gradients, state); on a resident session it is the
# lean step plus `export_gradients()`, so the bytes are the stateless
# path's. 'lean' returns `loss, step, completed_steps, next_batch_index,
# flags` and leaves the gradient on the device (or, on the stateless route
# an AUTO trainer resolved to, with the trainer) until `export_gradients()`.
# The default (None) is 'lean' unless resident=False is given (Andrew,
# 2026-11-06; formerly 'lean' only for resident=True): DEVIATION 2514 gate G5 on the
# H100 (2026-09-11) measured a lean target-shape step at 0.565 s against
# 8.6 s full and 45 s before the device-owned step, with every lean step
# bit-equal to the full path's and to the pre-change trainer's.
#
# Complete state is copied and updates commit only after success. On the
# stateless path evaluation requires byte-unchanged parameters, moments,
# flags and counter; on a resident session that invariance is a native
# gate (design G1), not a per-call download. Public checkpoints use an
# explicitly separate bounded JSON/hex schema; they are not
# training/checkpoint.mojo's native binary v1 files. The JSON codec
# retains its 2 MiB bound; larger models export `export_state()` arrays.
# Learning, numerical correctness and cross-vendor identity remain
# unqualified in this slice.
class SmallByteLanguageModelTrainer:
    """FP32 decoder language model trainer (AdamW, mean cross-entropy).

    Trains a decoder-only language model whose dimensions come from
    :class:`~mojolearn.LanguageModelConfig` (``shape=``); the defaults are a
    small byte-level model (batch 2, length 32, width 32, 4 heads, 2 key/value
    heads, MLP 64, vocabulary 256). The model has no biases, no dropout and no
    final norm, and the embedding and output head are separate (untied).
    ``LanguageModelTrainer`` is the same class.

    You supply everything: the initial parameters (there is no hidden
    initialization), the token ids of every step (there is no tokenizer,
    padding, truncation or random number generation inside), and a
    ``data_schedule`` describing them. All arithmetic runs on the GPU in the
    process numeric mode: ``"identical"`` (the default) gives the same bits on
    Apple, NVIDIA and AMD.

    Resident sessions. On a GPU install ``resident`` defaults to True: the
    parameters, AdamW moments and last gradient live on the device between
    steps, and only the ids go in and the loss comes out. They are copied to
    the host only when you ask (:meth:`export_state`, :meth:`export_gradients`,
    :meth:`export_checkpoint`, :meth:`close`). A step that fails is rolled back.
    If the device context is lost, the steps since the last export are lost
    and every call raises RuntimeError until :meth:`load_state_dict`.
    ``resident=False`` runs each step statelessly with the state on the host.

    The ``data_schedule`` is a small JSON-serializable dict you provide
    (for example from :meth:`mojolearn.lm_corpus.TokenBatches.data_schedule`):
    record the corpus, token order, batch offsets and planned steps there. It
    is checked only for size and JSON round-trip, not against the ids you
    pass, but once a checkpoint is written it is covered by the checkpoint's
    SHA-256, so it cannot be changed undetected. ``next_batch_index`` equals
    the completed step count; after a restore, pass the matching next batch.

    Parameters
    ----------
    parameters : array-like or dict
        Initial parameters: one flat float32 array of ``shape.n_total``
        values, or a dict of named tensors as listed by
        :meth:`parameter_registry`.
    data_schedule : dict
        The JSON descriptor described above. When it records a vocabulary,
        its ``n_vocab`` must equal ``shape.vocab_size``.
    lr : float, default 1e-3
        AdamW learning rate.
    betas : tuple of float, default (0.9, 0.999)
        AdamW moment decay rates.
    eps : float, default 1e-8
        AdamW epsilon.
    weight_decay : float, default 0.01
        AdamW decoupled weight decay.
    shape : LanguageModelConfig or None, default None
        Model dimensions; None uses the defaults.
    resident : bool or None, default None
        Keep the training state on the device between steps. None chooses the
        resident session whenever the loaded GPU binaries support it.
    step_result : {"full", "lean"} or None, default None
        What :meth:`train_step` returns. ``"lean"`` returns the loss, the step
        counters and the flags; the last step's gradient is available from
        :meth:`export_gradients`. ``"full"`` also returns the pre-update
        gradients (``flat_gradients`` and ``gradients``) from every step.
        None means ``"lean"``, or ``"full"`` when ``resident=False`` is given.
        ``"lean"`` cannot be combined with ``resident=False``.

    Attributes
    ----------
    step_ : int
        Completed training steps.
    parameters_ : Array
        A copy of the current flat parameters.
    data_schedule : dict
        A copy of the ``data_schedule`` descriptor.

    Examples
    --------
    >>> import numpy as np, mojolearn
    >>> shape = mojolearn.LanguageModelConfig()
    >>> rng = np.random.default_rng(0)
    >>> weights = (rng.standard_normal(shape.n_total) * 0.02).astype(np.float32)
    >>> trainer = mojolearn.LanguageModelTrainer(weights, data_schedule={"corpus": "demo"})
    >>> ids = rng.integers(0, 256, (shape.batch, shape.length + 1), dtype=np.int32)
    >>> loss = trainer.train_step(ids)["loss"]
    """

    def __init__(self, parameters, *, data_schedule, lr=1e-3, betas=(.9, .999),
                 eps=1e-8, weight_decay=.01, shape=None, resident=None, step_result=None):
        _LM_TRAINER(self, "construct", (parameters, data_schedule, lr, betas, eps,
                    weight_decay, shape, resident, step_result), _LM_TRAINER_CONTEXT)

    @property
    def data_schedule(self):
        """A copy of the ``data_schedule`` descriptor, as checkpoints keep it."""
        return _LM_TRAINER(self, "data_schedule", (), _LM_TRAINER_CONTEXT)

    @staticmethod
    def parameter_registry(shape=None):
        """Name, shape, offset and size of every parameter tensor.

        Parameters
        ----------
        shape : LanguageModelConfig or None, default None
            Model dimensions; None uses the defaults.

        Returns
        -------
        list of dict
            One ``{"name", "shape", "offset", "size"}`` record per tensor, in
            the order of the flat parameter vector.
        """
        return _LM_TRAINER(None, "parameter_registry", (shape,), _LM_TRAINER_CONTEXT)

    @property
    def step_(self):
        """Number of completed training steps."""
        return _LM_TRAINER(self, "step", (), _LM_TRAINER_CONTEXT)

    @property
    def parameters_(self):
        """A copy of the current flat float32 parameters."""
        return _LM_TRAINER(self, "parameters", (), _LM_TRAINER_CONTEXT)

    def state_dict(self):
        """The complete training state; same as :meth:`export_state`."""
        return _LM_TRAINER(self, "export_state", (), _LM_TRAINER_CONTEXT)

    # The complete state as fresh `mojolearn.Array` copies plus metadata
    # (the `state_dict()` shape). On an open resident session: the device
    # state is re-scanned natively, downloaded once and validated here by
    # the same `_validate_state` that admits an import; on a closed or
    # stateless trainer the host arrays are copied. Mutating the returned
    # arrays cannot reach the device (design section 3).
    def export_state(self):
        """The complete training state as fresh copies.

        On a resident session this downloads the device state once and
        validates it. Changing the returned arrays does not affect the trainer.

        Returns
        -------
        dict
            Parameters, AdamW moments ``m`` and ``v``, flags, step counters,
            configuration, ``data_schedule`` and metadata, as accepted by
            :meth:`load_state_dict`.
        """
        return _LM_TRAINER(self, "export_state", (), _LM_TRAINER_CONTEXT)

    def export_gradients(self, named=True):
        """The pre-update gradient of the last completed training step.

        Available after a :meth:`train_step` with ``step_result="lean"`` (the
        default) and valid until the next :meth:`train_step` starts. With
        ``step_result="full"`` on a stateless trainer (``resident=False``),
        :meth:`train_step` returns the gradients itself.

        Parameters
        ----------
        named : bool, default True
            Also return the gradient split into named tensors.

        Returns
        -------
        dict
            ``{"flat_gradients": Array}`` and, when ``named``,
            ``{"gradients": {name: Array}}``.

        Raises
        ------
        RuntimeError
            If no resident session is open, or no step has completed since it was
            opened, restored or rolled back.
        """
        return _LM_TRAINER(self, "export_gradients", (named,), _LM_TRAINER_CONTEXT)

    # DEVIATION 3010: what the OPEN resident session's quadratic
    # attention stages are holding RIGHT NOW. `None` when no session is
    # open, or when the loaded binding predates this report.
    #
    # Keys: `forward_eager_cells`, `forward_aexp_cells`,
    # `backward_eager_cells` and the matching `*_bytes` at four bytes a
    # cell; `layers_grown_forward`, `layers_grown_backward`,
    # `layers_full_aexp`, `layers`; `eager_bytes` (forward eager +
    # backward eager, the two that only the eager fallback grows) and
    # `total_bytes` (those plus `aexp`). New bindings also report per-layer
    # `forward_status`, `backward_status`, their named `*_counts`, and
    # `attn_materialized` after the latest step. Status -1 means not
    # attempted; 0 ran, 1 refused regime, 2 corner, 3 DEVIATION 3110's
    # latch (this layer refused before, so nothing was launched at all and
    # the eager path ran alone). Materialization may
    # have occurred during backward recomputation. Retained capacity does
    # not say which path ran this step.
    #
    # WHAT IT IS FOR. The ten `[B, n_heads, L, S]` attention arrays are
    # allocated at ONE element for the fused path and GROW ON DEMAND the
    # first time a layer takes the eager path -- a refused regime, or a
    # `FUSED_CORNER` hit, both of which depend on the DATA and so on the
    # step. With the legacy retention policy they stay for the session.
    # `release_eager` builds release dead forward scratch after forward
    # and remaining eager stages after each layer backward; the report's
    # `released_eager_bytes` witnesses that work separately from capacity.
    # `sticky_eager` and `layers_prefer_eager` report the policy which
    # chooses eager before launch after a layer's first corner refusal.
    # `repair_masked_tail` identifies the replay build; per-layer
    # `backward_repair_sites` is a bitmask of estash repairs actually executed:
    # 1 for zdot, 2 for dQ, 3 for both, 0 for neither.
    # A legacy run therefore has no single device footprint: it can step up
    # once, at a step nobody chose, and every capacity figure taken
    # before that step is wrong afterwards. A three-step probe cannot
    # see it. Read this between steps and the step where it moved is
    # named rather than inferred.
    #
    # `forward_aexp_cells` is reported APART from the other forward
    # arrays because two unrelated mechanisms grow `aexp`: the eager
    # fallback grows all four together, while a build carrying the
    # fused exp stash grows `aexp` alone on a call that refused
    # nothing. `eager_bytes` excludes it for exactly that reason.
    #
    # Host metadata only: buffer lengths and saved launcher statuses. Nothing is launched, downloaded or
    # synchronized, and no step path calls this.
    def attention_stage_report(self):
        """Memory held by the open session's attention stages.

        The attention work arrays start small and grow the first time a layer
        needs the unfused attention path, which depends on the data. Read this
        between steps to see the current device footprint of those arrays.
        Nothing is launched or downloaded.

        Returns
        -------
        dict or None
            Cell and byte counts per stage (``forward_eager_cells``,
            ``forward_aexp_cells``, ``backward_eager_cells`` and their ``*_bytes``,
            ``eager_bytes``, ``total_bytes``), layer counts and per-layer status.
            None when no resident session is open or the loaded binaries do not
            report it.
        """
        return _LM_TRAINER(self, "attention_stage_report", (), _LM_TRAINER_CONTEXT)

    def close(self):
        """Copy the device state to the host and release the device.

        A later call re-opens the session from that host state. On a lost
        session nothing can be exported: the device is released and the trainer
        stays lost until :meth:`load_state_dict`.
        """
        return _LM_TRAINER(self, "close", (), _LM_TRAINER_CONTEXT)

    def load_state_dict(self, state):
        """Replace the complete training state.

        The replacement is validated before anything changes; a refused state
        leaves the trainer untouched. Any resident session is released without
        export and the next call opens one from the replacement.

        Parameters
        ----------
        state : dict
            A state from :meth:`export_state` (or :meth:`state_dict`).
        """
        return _LM_TRAINER(self, "load_state", (state,), _LM_TRAINER_CONTEXT)

    def _binding(self):
        # glue: the loaded byte LM binding of the process tier, read only. Its
        # remaining callers are tools/byte_lm_model_pool_check.py and
        # tools/byte_lm_offload_check.py (fault-hook probes). The trainer door
        # admits the binding, suspends a session on a change and computes the
        # run metadata natively (bindings/byte_lm_trainer_native_door.mojo `_binding`).
        return _load(state_shape(self._state))

    def run_metadata(self):
        """Profile, configuration, schedule and binary details of this run.

        Reads constants only; runs no model computation. Keep it with any results
        you record.

        Returns
        -------
        dict
        """
        return _LM_TRAINER(self, "run_metadata", (), _LM_TRAINER_CONTEXT)

    def train_step(self, ids):
        """Run one training step: mean cross-entropy loss and an AdamW update.

        Parameters
        ----------
        ids : array-like of shape (batch, length + 1), int32
            Token ids in ``[0, shape.vocab_size)``. The first ``length``
            positions predict the next ``length``; the loss averages
            ``batch * length`` targets.

        Returns
        -------
        dict
            ``loss``, ``step``, ``completed_steps`` and ``next_batch_index``. By
            default (``step_result="lean"``) also ``flags``; call
            :meth:`export_gradients` for this step's gradient. With
            ``step_result="full"`` also ``flat_gradients`` and ``gradients`` (by
            tensor name), the pre-update gradient.
        """
        # the lock, the ids, the binding and the session lifecycle and the
        # step in one native call (bindings/byte_lm_trainer_native_door.mojo)
        return _LM_TRAINER(self, "train_step", (ids,), _LM_TRAINER_CONTEXT)

    def train_steps(self, ids):
        """Run several training steps in one call.

        Equivalent to calling :meth:`train_step` once per batch, in order,
        with the same losses and the same final state, but the batches are
        handed over together and the steps run back to back on the device.

        Parameters
        ----------
        ids : array-like of shape (n_steps, batch, length + 1), int32
            ``n_steps`` batches of token ids, each as for :meth:`train_step`.

        Returns
        -------
        dict
            ``losses`` (float32 array of shape ``(n_steps,)``, one loss per
            step), and for the last step ``loss``, ``step``,
            ``completed_steps``, ``next_batch_index`` and, by default,
            ``flags``. With ``step_result="full"`` the last step's
            ``flat_gradients`` and ``gradients`` instead of ``flags``.

        Examples
        --------
        >>> import numpy as np, mojolearn
        >>> shape = mojolearn.LanguageModelConfig()
        >>> rng = np.random.default_rng(0)
        >>> weights = (rng.standard_normal(shape.n_total) * 0.02).astype(np.float32)
        >>> trainer = mojolearn.LanguageModelTrainer(weights, data_schedule={"corpus": "demo"})
        >>> ids = rng.integers(0, 256, (4, shape.batch, shape.length + 1), dtype=np.int32)
        >>> losses = trainer.train_steps(ids)["losses"]
        """
        # one native call: the lock, the ids, the binding and the steps
        # (bindings/byte_lm_trainer_native_door.mojo byte_lm_train_steps)
        return _LM_TRAINER(self, "train_steps", (ids,), _LM_TRAINER_CONTEXT)

    def evaluate(self, ids):
        """Mean loss on ``ids`` without changing any state.

        Parameters
        ----------
        ids : array-like of shape (batch, length + 1), int32
            Token ids, as for :meth:`train_step`.

        Returns
        -------
        float
        """
        return _LM_TRAINER(self, "evaluate", (ids,), _LM_TRAINER_CONTEXT)

    # Float32 logits `[batch, length, vocab]` for int32 ids `[batch,
    # length]` (DEVIATION 2658), for positions 0 to length - 1 of each row,
    # prefilled from absolute position 0.
    #
    # `lengths` (2026-09-15) makes the batch RAGGED: `batch` integers in
    # `[1, length]`, row `i` real at positions `[0, lengths[i])` and
    # padding after, whatever int32 values the padding holds. Real
    # positions' logits are byte for byte the row alone at its own length
    # (causal attention, no arithmetic changes, `_ragged.py` says why) and
    # padding positions' logits are exactly `+0.0`. `batch` must be in [1, 1024],
    # `length` in [1, shape.length], `batch * length * vocab` at most
    # 268435456 and every id in [0, shape.vocab_size); anything else is
    # refused with ValueError before any native call.
    #
    # The arithmetic is the IDENTICAL device forward the trainer's loss
    # uses, on Metal, CUDA or HIP, and these are the numbers it computes
    # before the loss. tools/byte_lm_gpu_logits_sweep.py compares them
    # byte for byte with the CPU reference path,
    # `LanguageModelInference(threaded=False)`. The native entries refuse
    # a non-finite logit with an error, because NaN payloads are shaped by
    # the vendor and cannot be part of an identity claim; the CPU class
    # returns them.
    #
    # No state changes on either path. A resident trainer runs on its
    # session (opening it first if needed, as `train_step` does) and the
    # parameters, moments, flags and step counter do not move. A failed
    # call has nothing to roll back; the last gradient is no longer
    # exportable, as after any failure, and a session that no longer
    # reports itself usable is lost. A stateless trainer validates its
    # state as `evaluate` does, builds and destroys one device context,
    # and requires the parameter and id bytes it handed over unchanged
    # afterward.
    def logits(self, ids, *, lengths=None):
        """Float32 logits for every position, without changing any state.

        Runs the same GPU forward pass the training loss uses, starting at
        position 0 of each row.

        Parameters
        ----------
        ids : array-like of shape (batch, length), int32
            Token ids in ``[0, shape.vocab_size)``. ``batch`` in [1, 1024],
            ``length`` in [1, shape.length], and ``batch * length * vocab_size`` at
            most 268435456.
        lengths : sequence of int or None, default None
            Makes the batch ragged: row ``i`` is real at positions
            ``[0, lengths[i])`` and padding after, whatever the padding ids are.
            Real positions get exactly the logits of the row alone at its own
            length; padding positions are ``+0.0``.

        Returns
        -------
        Array of shape (batch, length, vocab_size), float32

        Raises
        ------
        ValueError
            For ids or lengths outside the limits above.
        RuntimeError
            If a logit is not finite.
        """
        return _LM_TRAINER(self, "logits", (ids, lengths), _LM_TRAINER_CONTEXT)

    # The greedy next byte after each row of ids `[batch, length]`, ties
    # to the lowest byte value, picked from `logits(ids)` by the helper
    # `LanguageModelInference.next_bytes` uses (DEVIATION 2658). With
    # `lengths` (a ragged batch, see `logits`) the byte after each row's
    # LAST REAL position.
    def next_bytes(self, ids, *, lengths=None):
        """Greedy next token after each row.

        Picks the highest logit at each row's last position (with ``lengths``,
        its last real position), ties going to the lowest id.

        Parameters
        ----------
        ids : array-like of shape (batch, length), int32
            Token ids, as for :meth:`logits`.
        lengths : sequence of int or None, default None
            Ragged row lengths, as for :meth:`logits`.

        Returns
        -------
        list of int
            One token id per row.
        """
        return _LM_TRAINER(self, "next_bytes", (ids, lengths), _LM_TRAINER_CONTEXT)

    def save_checkpoint(self, path):
        """Same as :meth:`export_checkpoint`."""
        return _LM_TRAINER(self, "export_checkpoint", (path,), _LM_TRAINER_CONTEXT)

    def export_checkpoint(self, path):
        """Write the complete state to ``path`` as a JSON checkpoint.

        The file is written atomically, holds no pickle, and is limited to 2 MiB
        (about 87,000 parameters). For larger models use
        :meth:`export_checkpoint_binary`.

        Parameters
        ----------
        path : str or path-like
            Destination file.
        """
        return _LM_TRAINER(self, "export_checkpoint", (path,), _LM_TRAINER_CONTEXT)

    # Atomically write the complete state as a streamed BINARY archive
    # (`_byte_lm_checkpoint`, schema `mojolearn.byte-lm-stream.v1`), from
    # `export_state()`. Returns the file's SHA-256.
    #
    # THIS IS THE PATH AT SCALE, AND IT IS A SECOND FORMAT, NOT A WIDER
    # FIRST ONE. `export_checkpoint` keeps its JSON/hex envelope, its
    # bytes and its 2 MiB bound untouched; hex doubles the payload, so
    # that bound stops at about 87,381 parameters and the
    # 162,147,840-parameter shape has always been refused by it. Here the
    # four arrays are streamed as their little-endian `<f4`/`<i4` bytes
    # after a bounded canonical JSON header, so the file is about 1.95 GB
    # at that shape instead of impossible, and a round trip is bit-exact
    # by construction: nothing passes through decimal text, so `-0.0`,
    # NaN payloads and subnormals all survive.
    #
    # THE OPTIMIZER MOMENTS ARE IN THE FILE AND THAT IS THE POINT. A
    # restore that dropped `m` and `v` would produce the same loss and
    # the same gradient on its first step and a different update, so the
    # divergence would surface a step later looking like
    # nondeterminism.
    #
    # No device work beyond the one `export_state()` download a resident
    # session already owes, and no arithmetic.
    def export_checkpoint_binary(self, path):
        """Write the complete state to ``path`` as a binary checkpoint.

        For models of any size. The parameters and the AdamW moments are stored
        as raw little-endian bytes after a JSON header, so a round trip is
        bit-exact (``-0.0``, NaN payloads and subnormals survive) and a resumed
        run continues exactly. The file is written atomically.

        Parameters
        ----------
        path : str or path-like
            Destination file.

        Returns
        -------
        str
            The file's SHA-256.
        """
        return _LM_TRAINER(self, "export_checkpoint_binary", (path,), _LM_TRAINER_CONTEXT)

    @classmethod
    def from_checkpoint_binary(cls, path, *, resident=None):
        """Restore a trainer from :meth:`export_checkpoint_binary`.

        The whole file is verified (header and per-array digests, exact size)
        before anything is allocated, and the state is validated as
        :meth:`load_state_dict` validates it.

        Parameters
        ----------
        path : str or path-like
            The checkpoint file.
        resident : bool or None, default None
            As for the constructor.

        Returns
        -------
        LanguageModelTrainer
        """
        return _LM_TRAINER(cls, "from_checkpoint_binary", (path, resident), _LM_TRAINER_CONTEXT)

    @classmethod
    def from_checkpoint(cls, path, *, resident=None):
        """Restore a trainer from a JSON checkpoint written by :meth:`export_checkpoint`.

        Parameters
        ----------
        path : str or path-like
            The checkpoint file.
        resident : bool or None, default None
            As for the constructor.

        Returns
        -------
        LanguageModelTrainer
        """
        return _LM_TRAINER(cls, "from_checkpoint", (path, resident), _LM_TRAINER_CONTEXT)

    @classmethod
    def from_checkpoint_bytes(cls, encoded, *, resident=None):
        """Restore a trainer from the bytes of a JSON checkpoint.

        Runs the same checks as :meth:`from_checkpoint` without opening a file.

        Parameters
        ----------
        encoded : bytes
            The checkpoint contents (exactly ``bytes``; copy mutable buffers
            first).
        resident : bool or None, default None
            As for the constructor.

        Returns
        -------
        LanguageModelTrainer
        """
        return _LM_TRAINER(cls, "from_checkpoint_bytes", (encoded, resident), _LM_TRAINER_CONTEXT)


def _connect_trainer():
    # glue: connect the GPU trainer's native step owner and its fixed
    # metadata once, at package import
    from .tokenizer import _binding as _tokenizer_binding
    from ._native_classical_bridge import _state_context
    from types import MethodType
    context = _state_context()
    # glue: the native trainer builds its token/output Arrays through the shared
    # array materializer (tsa_array -> arena_materialize_native._owned), which reads
    # the exact storage descriptors; connect them once, as the TSA bridge does.
    from ._native_owner_lifetime import connect_owner_context
    ownership = dict(context)
    connect_owner_context(ownership)
    context.update(owner_array_slots=ownership["owner_array_slots"],
                   arena_native_types=ownership["owner_native_types"], arena_memoryview=memoryview,
                   owner_mmap_class=ownership["owner_mmap_class"])
    _numeric_profile.default_profile("training")  # glue: connect the training profile before public construction
    context.update(byte_lm_source_root=str(Path(__file__).resolve().parents[2]), byte_lm_wrapper_file=__file__,
                   byte_lm_source_names=('python/mojolearn/_byte_lm_impl.py', 'python/mojolearn/language_model.py',
                                        'bindings/_mojolearn_byte_lm.mojo', 'training/byte_lm.mojo',
                                        'training/byte_lm_config.mojo', 'python/mojolearn/_byte_lm_config.py'))
    context.update(byte_lm_trainer_class=SmallByteLanguageModelTrainer, byte_lm_config_class=ByteLanguageModelConfig,
                   byte_lm_profile_defaults=_numeric_profile._defaults, byte_lm_profiles=_numeric_profile.PROFILES,
                   byte_lm_lock=type(threading.RLock()))
    _backend.default_mode()  # glue: resolve mode before any native product operation
    bindings = {}
    for mode in _NATIVE_MODE_CODE:  # glue: the two numeric modes, connected once
        try:
            bindings[mode] = _backend.binding(_EXTENSION, mode)
        except (ImportError, AttributeError, ValueError, RuntimeError, TypeError, OSError) as error:
            bindings[mode] = type(error).__name__ + ": " + str(error)
    context.update(byte_lm_bindings=bindings,
                   byte_lm_vendors=('cpu',) if _backend._CPU_ONLY is not None else ('cuda', 'hip', 'metal'),
                   byte_lm_profile=PROFILE, byte_lm_logits_pool=_LogitsPool, backend=vars(_backend))
    module = _tokenizer_binding()._module
    context.update(byte_lm_pooled_class=_PooledLogits,
                   byte_lm_store_getter=vars(_Array)["_store"].__get__)
    _PooledLogits.__del__ = module.classical_attribute_descriptor(
        MethodType(module.byte_lm_pooled_finalize_call, context))
    # glue: pin the existing native slot and exact pooled storage hooks once.
    # Native logits checks replacements before allocation, then reads this slot
    # once; Array.__array_interface__ is never an output-address fallback.
    from types import MemberDescriptorType
    address_slot = vars(_Array)["_addr"]
    if type(address_slot) is not MemberDescriptorType:
        raise TypeError("Byte-LM pooled logits require the native Array address slot")
    mro = _PooledLogits.__mro__
    hooks = ('__getattribute__', '__getattr__', '__setattr__', '__del__', '__dict__',
             '__array_interface__', '_pool', '_store', '_base', '_pin', '_mv', '_addr',
             '_readonly', 'shape', 'dtype', 'ndim', 'size', 'nbytes', 'itemsize', 'strides', 'order')
    layout = tuple((base, tuple((name, vars(base).get(name)) for name in hooks))  # glue: fixed native storage hooks
                   for base in mro)  # glue: exact connected pooled Array ancestry
    context.update(byte_lm_addr_getter=address_slot.__get__, byte_lm_pooled_mro=mro,
                   byte_lm_pooled_layout=layout,
                   byte_lm_type_mro_getter=type.__dict__['__mro__'].__get__,
                   byte_lm_type_dict_getter=type.__dict__['__dict__'].__get__,
                   # glue: the unfilled pooled store's "f" view cast, pinned once
                   byte_lm_view_cast=memoryview.cast, byte_lm_view_format="f",
                   byte_lm_store_type=bytearray,
                   # glue: the pool keeps that view with its store (one view per
                   # store): the `_mv` slot getter, the view type and its exporter getter
                   byte_lm_view_getter=vars(_Array)["_mv"].__get__, byte_lm_view_type=memoryview,
                   byte_lm_view_obj=memoryview.obj.__get__)
    return module.byte_lm_trainer_call, context


_LM_TRAINER, _LM_TRAINER_CONTEXT = _connect_trainer()
