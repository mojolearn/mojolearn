# SPDX-License-Identifier: Apache-2.0
"""Ordered data-parallel byte LM training and single-GPU accumulation replay.

Uses the existing IDENTICAL model kernels. Shard gradients (each a mean CE)
are SUMMED, not averaged. Keep logical_shards, shard order, shape, inputs and
starting state fixed when changing the number of physical devices. Hardware
identity requires cloud qualification; this module alone is not evidence.
"""
import threading

from ._byte_lm_impl import SmallByteLanguageModelTrainer, _array, _float32, _validate_state
from ._byte_lm_config import state_shape
from ._buffer import Buf, addr, addr_ro, empty, flat_bytes, typestr_of


def _missing_entry(driver, what):
    """The refusal for a byte-LM binding without the `driver` session
    entries. On a CPU-only install the binding is the host byte-LM binding,
    which has no parallel, pooled or offloaded session BY DESIGN: the
    cooperative driver's declared sentence (`_parallel_pool._cpu_refusal`),
    so the identity harness reads the par-* host cell N/A rather than a
    stale build. Elsewhere it is a GPU binding older than the entries."""
    from ._backend import vendor
    if vendor() == 'cpu':
        return NotImplementedError(
            'no CPU implementation of the cooperative multi-GPU driver ' + driver + ' yet: its '
            'sessions live inside the GPU byte-LM binding, which no host binding restates')
    return ImportError('rebuild bindings/build_byte_lm.sh for ' + what)

STATE_ARRAYS = ('parameters', 'm', 'v', 'flags')


def _export_target(obj, count, typestr, name):
    """The address of a caller-provided export buffer (the `into=` forms).

    DEVIATION 3120: a per-step export at 162M parameters allocated four
    fresh zero-filled 648.6 MB arrays every call (about 1.4 s a step on an
    H100 host). A caller that exports every step passes the same buffers
    back instead. Each must be a writable, C-contiguous, one-dimensional
    buffer of exactly `count` elements of `typestr`; anything else is
    refused BEFORE the binding is called, so nothing is written to it. The
    caller keeps `obj` alive for the call, as for `addr`. If the export
    then raises, the buffer holds an unspecified partial export."""
    with Buf(obj, writable=True, name=name) as b:
        if b.readonly:
            raise ValueError('mojolearn: %s is read-only, refusing to write to it' % name)
        try:
            got = typestr_of(b)
        except TypeError:
            got = b.format
        if got != typestr:
            raise TypeError('mojolearn: %s must be %s, got %s' % (name, typestr, got))
        if b.ndim != 1 or not b.c_contiguous:
            raise ValueError('mojolearn: %s must be a contiguous one-dimensional buffer' % name)
        if b.shape[0] != count or b.nbytes != count * b.itemsize:
            raise ValueError('mojolearn: %s must hold exactly %d elements, got %d' % (name, count, b.shape[0]))
        return b.addr


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
        form's (`_export_target` says what is refused)."""
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


def _floats(value):
    """A float32 buffer from raw little-endian bytes (a gradient off the wire)
    or any float32 array; `_array` then copies and admits it."""
    if isinstance(value, (bytes, bytearray)):
        import array as _pyarray
        out = _pyarray.array('f')
        out.frombytes(value)
        return out
    return value


def ordered_sum_gradients(parts):
    """Copy shard zero, then add shards 1..K-1 using IDENTICAL GPU pair adds.

    Separate from training.accumulate_grads, whose balanced-tree contract is
    unchanged. Neither this fold nor the byte-LM fold divides by shard count.
    """
    from ._training_impl import accumulate_grads
    parts = list(parts)
    if not parts:
        raise ValueError('ordered reduction needs at least one shard')
    total = [x.copy() for x in parts[0]]  # glue: copies each parameter tensor buffer
    for part in parts[1:]:  # glue: one native accumulate call per shard
        total = accumulate_grads([total, part], tokens=None, numeric_mode='identical')
    return total


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
        from ._mlp_impl import SmallMLPTrainer
        from ._samba_impl import SambaStack
        from ._parallel_pool import DevicePool
        if type(logical_shards) is not int or not 1 <= logical_shards <= 1024:
            raise ValueError('logical_shards must be in [1, 1024]')
        if isinstance(model, SmallMLPTrainer):
            self._operation = 'mlp_gradient'
        elif isinstance(model, SambaStack):
            if model.optimizer.accumulation_steps != 1:
                raise ValueError('parallel Samba requires accumulation_steps=1; logical_shards owns accumulation')
            if model.numeric_mode not in (None, 'identical'):
                raise ValueError('parallel training requires IDENTICAL numeric mode')
            self._operation = 'samba_gradient'
        else:
            raise TypeError('ParallelNeuralTrainer supports SmallMLPTrainer and SambaStack')
        if model.state_dict().get('numeric_mode') != 'identical':
            raise ValueError('parallel training requires an IDENTICAL model')
        self._pool = DevicePool(devices)
        if len(self._pool.devices) > logical_shards:
            self._pool.close()
            raise ValueError('physical device count exceeds logical shard count')
        self._update_pool = DevicePool(self._pool.devices, cooperative=True)
        self.model = model
        self.logical_shards = logical_shards
        self._lock = threading.RLock()
        self._closed = False
        self._gradients = None

    def train_step(self, shards):
        """Each shard is (inputs, targets); loss normalization is per microbatch."""
        with self._lock:
            if self._closed:
                raise RuntimeError('parallel trainer is closed')
            shards = list(shards)
            if len(shards) != self.logical_shards:
                raise ValueError('logical shard count mismatch')
            snapshot = self.model.state_dict()
            self._gradients = None
            try:
                requests = []
                if self._operation == 'samba_gradient':
                    stream = (self.model.generator.next_stream()
                              if self.model.config.dropout > 0 else None)
                    offset = 0
                    for inputs, targets in shards:  # glue: one worker request per shard
                        inputs = self.model._ids(inputs, 'inputs')
                        targets = self.model._ids(targets, 'targets')
                        requests.append((self._operation, snapshot, (inputs, targets, stream, offset)))
                        offset += inputs.size
                else:
                    from ._mlp_impl import _batch, _targets
                    for inputs, targets in shards:  # glue: one worker request per shard
                        inputs = _batch(inputs)
                        targets = _targets(targets, len(inputs))
                        requests.append((self._operation, snapshot, (inputs, targets)))
                results = self._pool.map(requests)
                losses = tuple(part[0] for part in results)  # glue: one scalar loss per shard
                update_state = snapshot
                if self._operation == 'samba_gradient':
                    update_state = dict(snapshot, rng=self.model.generator.state_dict())
                # The cooperative worker owns gradient columns, whole clipping
                # tensors and update ranges, retaining the original norm tree.
                updated, retained, step = self._update_pool.map([(
                    self._operation.replace('_gradient', '_update'), update_state,
                    [part[1] for part in results])])[0]  # glue: one gradient list per shard
                result = dict(losses=losses, completed_steps=step,
                              logical_shards=self.logical_shards, reduction='ordered_sum')
                self.model.load_state_dict(updated)
            except BaseException:
                self.model.load_state_dict(snapshot)
                raise
            self._gradients = retained
            return result

    def export_gradients(self):
        with self._lock:
            if self._gradients is None:
                raise RuntimeError('no committed parallel gradient')
            return [g.copy() for g in self._gradients]  # glue: copies each parameter gradient buffer

    def checkpoint(self):
        with self._lock:
            return dict(schema='mojolearn.parallel-neural.v1', operation=self._operation,
                        logical_shards=self.logical_shards, reduction='ordered_sum',
                        state=self.model.state_dict())

    @classmethod
    def from_checkpoint(cls, checkpoint, *, devices=(0,)):
        if (checkpoint.get('schema') != 'mojolearn.parallel-neural.v1'
                or checkpoint.get('reduction') != 'ordered_sum'):
            raise ValueError('unsupported parallel neural checkpoint contract')
        state = checkpoint['state']
        operation = checkpoint.get('operation')
        if operation == 'mlp_gradient':
            from ._mlp_impl import SmallMLPTrainer, _validate_state
            weights, _, _, schedule = _validate_state(state)
            model = SmallMLPTrainer(*weights, data_schedule=schedule)
        elif operation == 'samba_gradient':
            from ._samba_impl import SambaStack, SambaConfig
            from ._training_impl import Generator
            model = SambaStack(SambaConfig.from_dict(state['config']),
                               generator=Generator(0, 'identical'), numeric_mode='identical')
        else:
            raise ValueError('unsupported parallel neural checkpoint operation')
        model.load_state_dict(state)
        return cls(model, devices=devices, logical_shards=checkpoint['logical_shards'])

    def close(self):
        with self._lock:
            self._closed = True
            self._pool.close()
            self._update_pool.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def _connect_parallel_byte_native():
    # ABI connection only. Product methods below hand off once to native Mojo.
    from . import _backend
    from ._array import Array
    from ._byte_lm_config import ByteLanguageModelConfig
    from .tokenizer import _binding as _tokenizer_binding
    runtime = _tokenizer_binding()._module.parallel_byte_call
    _backend.default_mode()
    vendor = _backend.vendor()
    if vendor == "cpu":
        # cpu-route: only the explicitly selected CPU installation reads host profile metadata
        modes = {"identical": _backend.load_host_module("_mojolearn_byte_lm_host")}
    else:
        modes = {"identical": _backend.binding("_mojolearn_byte_lm", "identical")}
    if vendor == "metal":
        modes["fast"] = _backend.binding("_mojolearn_byte_lm", "fast")
    runtime("initialize", (Array, ByteLanguageModelConfig, modes, vendor))
    return runtime


_native_parallel_byte = _connect_parallel_byte_native()
