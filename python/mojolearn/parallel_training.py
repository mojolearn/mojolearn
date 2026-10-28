# SPDX-License-Identifier: Apache-2.0
"""Ordered data-parallel byte LM training and single-GPU accumulation replay.

Uses the existing IDENTICAL model kernels. Shard gradients (each a mean CE)
are SUMMED, not averaged. Keep logical_shards, shard order, shape, inputs and
starting state fixed when changing the number of physical devices. Hardware
identity requires cloud qualification; this module alone is not evidence.
"""

class ParallelByteLanguageModelTrainer:
    """Resident model replicas with pooled optimizer state and a fixed left fold.

    Construct from a SmallByteLanguageModelTrainer.state_dict(). devices=(0,)
    replays the same logical shards on one GPU with one resident model. No GPU
    is opened until the first step/export. Import does no device work.
    AdamW moments and rollback copies are owned in disjoint device ranges by
    default; model parameters, gradients and activations remain replicated.
    pool_optimizer=False retains complete optimizer replicas for comparison.
    """

    def __init__(self, state, *, devices=(0,), logical_shards=1, pool_optimizer=True):
        return _native_parallel_byte("init", (self, state, devices, logical_shards, pool_optimizer))

    @property
    def step_(self):
        return _native_parallel_byte("step", (self,))

    @property
    def _state(self):
        # Diagnostic: a detached copy of the native owner's admitted state.
        return _native_parallel_byte("admitted_state", (self,))

    def _open(self):
        return _native_parallel_byte("open", (self,))

    def train_step(self, shards):
        """Consume exactly logical_shards microbatches; return per-shard losses.

        Each microbatch has shape (batch, length + 1). The loss vector retains
        logical order; no host floating-point reduction participates in training.
        """
        return _native_parallel_byte("train_step", (self, shards))

    def state_dict(self, *, rank=0):
        """Export a replica in the existing portable byte-LM state format."""
        return _native_parallel_byte("state", (self, rank))

    def export_raw(self, *, rank=0, into=None):
        """The four state arrays of one replica, freshly downloaded, as
        `{'parameters', 'm', 'v', 'flags'}` with NO admission pass. For
        hashing and streaming a checkpoint at scale: `state_dict()` runs
        `_validate_state`, which copies and scans every state array (the scans
        are native since lane/py-lm, but the copies remain), which a per-step
        hash chain need not pay. What comes back is bytes for a digest or a file, not a
        state that anything may train from; a restore still enters through
        `_validate_state`.

        `into`, when given, is a dict with exactly those four keys holding
        caller-owned buffers (float32[n_total] three times, int32[n_tensors]
        for flags) that are written in place and returned as the dict;
        nothing is allocated. The bytes are the same as the no-argument
        form's. Each buffer must be a writable, C-contiguous, one-dimensional
        buffer of exactly its element count and dtype; anything else is
        refused before anything is written."""
        return _native_parallel_byte("export_raw", (self, rank, into))

    def set_lr(self, lr):
        """Set the learning rate every replica uses at its NEXT update (a
        per-step schedule computed on the host). The value is admitted as
        float32 exactly as the constructor's `lr` is (finite, positive);
        the native session applies it through `byte_lm_parallel_set_lr`
        and returns the float32 bits it stored, which must be the bits
        that were sent. The exported state's `config['lr']` follows, so a
        checkpoint written after this call records the rate in effect."""
        return _native_parallel_byte("set_lr", (self, lr))

    def export_gradients(self, *, rank=0, into=None):
        """The summed gradient of the last committed step, float32[n_total].
        `into`, when given, is a caller-owned float32[n_total] buffer written
        in place and returned; nothing is allocated (same bytes)."""
        return _native_parallel_byte("export_gradients", (self, rank, into))

    def _require_split_step(self, name):
        return _native_parallel_byte("require_split", (self, name))

    def shard_gradient(self, ids):
        """One logical shard's gradient from the CURRENT state, with no
        update: `(loss, float32[n_total] Array)`. The worker half of a step
        whose shards run in other processes, on any vendor
        (`mojolearn.cross_vendor`). Same kernels as one shard of
        `train_step`."""
        return _native_parallel_byte("shard_gradient", (self, ids))

    def apply_gradient(self, total):
        """Commit one step with `total`, the ordered left fold of every
        shard's gradient (`mojolearn.cross_vendor.ordered_fold`). The update
        is the one `train_step` runs after its own fold."""
        return _native_parallel_byte("apply_gradient", (self, total))

    # ---- the device fold for a live worker (mojolearn.cross_vendor, chained) ----

    def _require_fold(self):
        return _native_parallel_byte("require_fold", (self,))

    @property
    def has_device_fold(self):
        """True when the binding folds on the device (a live worker uses it
        instead of `cross_vendor.ordered_fold` on the host)."""
        return _native_parallel_byte("has_device_fold", (self,))

    def fold_reset(self, prefix=None):
        """Start this worker's device fold: empty, or from `prefix`, an
        already folded run of the shards before this worker's block."""
        return _native_parallel_byte("fold_reset", (self, prefix))

    def shard_gradient_fold(self, ids):
        """One shard's gradient, folded into the device total with the
        ordered add `train_step` uses; returns the loss. No update."""
        return _native_parallel_byte("shard_gradient_fold", (self, ids))

    def fold_add(self, gradient):
        """Fold a host-held shard gradient into the device total."""
        return _native_parallel_byte("fold_add", (self, gradient))

    def fold_export(self, *, into=None):
        """The device fold's total, as float32 bytes. With `into` (a
        caller-owned float32[n_total] buffer) the total is written there
        and `into` is returned instead, with no allocation and no copy to
        `bytes`; its bytes are the ones the no-argument form returns."""
        return _native_parallel_byte("fold_export", (self, into))

    def optimizer_ownership(self):
        """Actual native ownership and moment/rollback/reduction bytes per device."""
        return _native_parallel_byte("ownership", (self,))

    def checkpoint(self):
        """Portable state plus the logical reduction contract required for replay."""
        return _native_parallel_byte("checkpoint", (self,))

    @classmethod
    def from_checkpoint(cls, checkpoint, *, devices=(0,), pool_optimizer=True):
        return _native_parallel_byte("from_checkpoint", (cls, checkpoint, devices, pool_optimizer))

    def close(self):
        """Release device resources. Export/checkpoint before closing to retain state."""
        return _native_parallel_byte("close", (self,))

    def __enter__(self):
        return _native_parallel_byte("enter", (self,))

    def __exit__(self, *exc):
        return _native_parallel_byte("close", (self,))


def ordered_sum_gradients(parts):
    """Copy shard zero, then add shards 1..K-1 using IDENTICAL GPU pair adds.

    Separate from training.accumulate_grads, whose balanced-tree contract is
    unchanged. Neither this fold nor the byte-LM fold divides by shard count.
    """
    return _native_parallel_neural("ordered_sum", (parts,))


class ParallelNeuralTrainer:
    """Ordered multi-GPU gradients for SmallMLPTrainer and SambaStack.

    Workers evaluate frozen snapshots concurrently. Gradient columns, whole
    clipping tensors and optimizer ranges are distributed across the selected
    GPUs; the first GPU computes the original small cross-tensor norm. All owners join
    before publishing the owner state. Host transport
    preserves bytes. This trades
    transport cost for reuse of the existing public kernels and optimizer.
    Treat the supplied model as exclusively owned until close().
    """

    def __init__(self, model, *, devices=(0,), logical_shards=1):
        return _native_parallel_neural("init", (self, model, devices, logical_shards))

    def train_step(self, shards):
        """Each shard is (inputs, targets); loss normalization is per microbatch."""
        return _native_parallel_neural("train", (self, shards))

    def export_gradients(self):
        return _native_parallel_neural("export_gradients", (self,))

    def checkpoint(self):
        return _native_parallel_neural("checkpoint", (self,))

    @classmethod
    def from_checkpoint(cls, checkpoint, *, devices=(0,)):
        return _native_parallel_neural("restore", (cls, checkpoint, devices))

    def close(self):
        return _native_parallel_neural("close", (self,))

    def __enter__(self):
        return _native_parallel_neural("enter", (self,))

    def __exit__(self, *exc):
        return _native_parallel_neural("exit", (self,))


def _connect_parallel_byte_native():
    # glue: import-time ABI connection only. Product methods hand off once to
    # native Mojo (bindings/parallel_byte_native.mojo), which admits the exact
    # trainer classes and the installed binding's schedule descriptor table.
    from . import _backend
    from ._array import Array
    from ._byte_lm_config import ByteLanguageModelConfig
    from .tokenizer import _binding as _tokenizer_binding
    runtime = _tokenizer_binding()._module.parallel_byte_call
    mode = _backend.default_mode()
    vendor = _backend.vendor()
    try:  # glue: installed byte-LM binding selection at import
        selected = _backend.binding("_mojolearn_byte_lm", mode)
    except (ImportError, ValueError):
        # ValueError: a CPU-only install refuses a non-identical tier by
        # name (_backend._cpu_only_binding); it imports the same way.
        # A set built without the byte LM still imports: merged_groups.json
        # packs `_mojolearn_byte_lm` into the Linux identical set only, and
        # only with MOJOLEARN_PACKAGE_BYTE_LM=1, so a Linux process in fast
        # mode (or any set built with it off) has none. The byte-LM trainers
        # then refuse at use ("Byte-LM numeric mode is not installed",
        # bindings/parallel_byte_native.mojo) instead of the whole package
        # failing at `import mojolearn` (AMD verify fast, 2026-10-23).
        modes = {}
    else:
        # The registry validates the installed CPU owner before constructing its
        # legacy adapter. Cache its stored native module, never adapter methods.
        # Byte-LM owns registered Mojo types: preserve the one-mode-per-process
        # contract, including a fresh process selected as Apple fast at import.
        modes = {mode: selected._host if vendor == "cpu" else selected}  # glue: installed native mode connection
    runtime("initialize", (Array, ByteLanguageModelConfig, modes, vendor))
    runtime("admit_class", (ParallelByteLanguageModelTrainer, "replica"))
    return runtime


_native_parallel_byte = _connect_parallel_byte_native()

from ._parallel_neural import _native_parallel_neural
