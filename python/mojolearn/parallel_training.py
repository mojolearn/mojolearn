# SPDX-License-Identifier: Apache-2.0
"""Ordered data-parallel byte LM training and single-GPU accumulation replay.

Uses the existing IDENTICAL model kernels. Shard gradients (each a mean CE)
are SUMMED, not averaged. Keep logical_shards, shard order, shape, inputs and
starting state fixed when changing the number of physical devices. Hardware
identity requires cloud qualification; this module alone is not evidence.
"""
import threading

from ._byte_lm_impl import SmallByteLanguageModelTrainer, _array, _validate_state
from ._byte_lm_config import state_shape
from ._buffer import addr, addr_ro, empty


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
        devices = tuple(devices)
        if (type(logical_shards) is not int or not 1 <= logical_shards <= 1024
                or not 1 <= len(devices) <= logical_shards):
            raise ValueError('require 1 <= device count <= logical_shards <= 1024')
        if any(type(i) is not int or i < 0 for i in devices) or len(set(devices)) != len(devices):
            raise ValueError('devices must be distinct nonnegative integer indices')
        if type(pool_optimizer) is not bool:
            raise ValueError("pool_optimizer must be bool")
        self.pool_optimizer = pool_optimizer
        self.devices = devices
        self.logical_shards = logical_shards
        self._state = _validate_state(state)
        self._shape = state_shape(self._state)
        self._session = None
        self._binding = None
        self._closed = False
        self._lost = False
        self._lock = threading.RLock()

    @property
    def step_(self):
        return self._state['completed_steps']

    def _open(self):
        if self._closed or self._lost:
            raise RuntimeError('parallel trainer is closed or lost; restore a state export')
        if self._session is not None:
            return
        # Reuse the existing binding resolver and profile/vendor admission.
        seed = self._state
        helper = SmallByteLanguageModelTrainer(seed['parameters'],
            data_schedule=seed['data_schedule'], shape=self._shape)
        binding = helper._binding()
        for name in ('create', 'open', 'close', 'step', 'export', 'rollback'):
            if not callable(getattr(binding, 'byte_lm_parallel_' + name, None)):
                raise ImportError('rebuild bindings/build_byte_lm.sh for parallel training')
        open_name = 'byte_lm_parallel_open_pooled' if self.pool_optimizer else 'byte_lm_parallel_open'
        opener = getattr(binding, open_name, None)
        if not callable(opener):
            raise ImportError('rebuild bindings/build_byte_lm.sh for optimizer pooling')
        if not callable(getattr(binding, 'byte_lm_parallel_reduction_pool_available', None)):
            raise ImportError('rebuild bindings/build_byte_lm.sh for distributed reduction buffers')
        if binding.byte_lm_parallel_reduction_pool_available() != 1:
            raise RuntimeError('binding refused distributed reduction availability')
        cfg = seed['config']
        params = [0, self.step_, cfg['kind'], cfg['lr'], cfg['beta1'], cfg['beta2'],
                  cfg['eps'], cfg['weight_decay'], cfg['momentum'], cfg['dampening'],
                  int(cfg['nesterov']), cfg['max_norm']]
        session = binding.byte_lm_parallel_create()
        try:
            completed = opener(session,
                [addr_ro(seed[k], name=k) for k in ('parameters', 'm', 'v', 'flags')],
                params, list(self._shape.native_shape), list(self.devices), self.logical_shards)
            if completed != self.step_:
                raise RuntimeError('parallel admission returned wrong step')
        except BaseException:
            binding.byte_lm_parallel_close(session)
            raise
        self._binding, self._session = binding, session
        self._state = dict(seed, parameters=None, m=None, v=None)

    def train_step(self, shards):
        """Consume exactly logical_shards microbatches; return per-shard losses.

        Each microbatch has shape (batch, length + 1). The loss vector retains
        logical order; no host floating-point reduction participates in training.
        """
        with self._lock:
            shards = list(shards)
            if len(shards) != self.logical_shards:
                raise ValueError('logical shard count mismatch')
            shape = (self._shape.batch, self._shape.length + 1)
            tokens = [_array(x, shape, 'shard', '<i4') for x in shards]
            self._open()
            before = self.step_
            # Native failures already recover this transaction. An admission
            # refusal must never roll back the preceding successful step.
            losses = self._binding.byte_lm_parallel_step(self._session,
                [addr_ro(x, name='shard') for x in tokens], before)
            try:
                if len(losses) != self.logical_shards:
                    raise RuntimeError('parallel step returned wrong loss count')
                result = dict(losses=tuple(losses), completed_steps=before + 1,
                              logical_shards=self.logical_shards, reduction='ordered_sum')
            except BaseException:
                try:
                    restored = self._binding.byte_lm_parallel_rollback(self._session)
                    if restored != before:
                        self._lost = True
                except BaseException:
                    self._lost = True
                raise
            self._state['completed_steps'] = before + 1
            self._state['next_batch_index'] = before + 1
            return result

    def state_dict(self, *, rank=0):
        """Export a replica in the existing portable byte-LM state format."""
        with self._lock:
            self._open()
            state = dict(self._state)
            for key in ('parameters', 'm', 'v'):
                state[key] = empty((self._shape.n_total,), '<f4')
            state['flags'] = empty((self._shape.n_tensors,), '<i4')
            step = self._binding.byte_lm_parallel_export(self._session,
                [addr(state[k], name=k) for k in ('parameters', 'm', 'v', 'flags')], rank, False)
            if step != self.step_:
                raise RuntimeError('parallel export returned wrong step')
            return _validate_state(state)

    def export_gradients(self, *, rank=0):
        with self._lock:
            self._open()
            out = empty((self._shape.n_total,), '<f4')
            step = self._binding.byte_lm_parallel_export(self._session,
                [addr(out, name='gradients')], rank, True)
            if step != self.step_:
                raise RuntimeError('parallel gradient export returned wrong step')
            return out

    def _require_split_step(self, name):
        if self.pool_optimizer or len(self.devices) != 1:
            raise ValueError(name + ' requires devices=(d,) and pool_optimizer=False')
        if not callable(getattr(self._binding, 'byte_lm_parallel_' + name, None)):
            raise RuntimeError('this byte-LM binary predates ' + name + '; rebuild bindings/build_byte_lm.sh')

    def shard_gradient(self, ids):
        """One logical shard's gradient from the CURRENT state, with no
        update: `(loss, float32[n_total] Array)`. The worker half of a step
        whose shards run in other processes, on any vendor
        (`mojolearn.cross_vendor`). Same kernels as one shard of
        `train_step`."""
        with self._lock:
            tokens = _array(ids, (self._shape.batch, self._shape.length + 1), 'shard', '<i4')
            self._open()
            self._require_split_step('shard_gradient')
            out = empty((self._shape.n_total,), '<f4')
            loss = self._binding.byte_lm_parallel_shard_gradient(self._session,
                [addr_ro(tokens, name='shard'), addr(out, name='gradient')], self.step_)
            return float(loss), out

    def apply_gradient(self, total):
        """Commit one step with `total`, the ordered left fold of every
        shard's gradient (`mojolearn.cross_vendor.ordered_fold`). The update
        is the one `train_step` runs after its own fold."""
        with self._lock:
            total = _array(total, (self._shape.n_total,), 'summed gradient')
            self._open()
            self._require_split_step('apply_gradient')
            before = self.step_
            # A native failure has already rolled the replica back to `before`.
            done = self._binding.byte_lm_parallel_apply_gradient(self._session,
                [addr_ro(total, name='summed gradient')], before)
            if done != before + 1:
                self._lost = True
                raise RuntimeError('apply_gradient returned the wrong step')
            self._state['completed_steps'] = before + 1
            self._state['next_batch_index'] = before + 1
            return done

    def optimizer_ownership(self):
        """Actual native ownership and moment/rollback/reduction bytes per device."""
        with self._lock:
            self._open()
            rows = self._binding.byte_lm_parallel_ownership(self._session)
            return tuple(dict(device=device, first=int(row[0]), count=int(row[1]),
                              moment_bytes=int(row[2]), rollback_bytes=int(row[3]),
                              reduction_bytes=int(row[4]))
                         for device, row in zip(self.devices, rows))

    def checkpoint(self):
        """Portable state plus the logical reduction contract required for replay."""
        return dict(schema='mojolearn.parallel-byte-lm.v1',
                    logical_shards=self.logical_shards, reduction='ordered_sum',
                    state=self.state_dict())

    @classmethod
    def from_checkpoint(cls, checkpoint, *, devices=(0,), pool_optimizer=True):
        if (checkpoint.get('schema') != 'mojolearn.parallel-byte-lm.v1'
                or checkpoint.get('reduction') != 'ordered_sum'):
            raise ValueError('unsupported parallel checkpoint contract')
        return cls(checkpoint['state'], devices=devices,
                   logical_shards=checkpoint['logical_shards'], pool_optimizer=pool_optimizer)

    def close(self):
        """Release device resources. Export/checkpoint before closing to retain state."""
        with self._lock:
            self._closed = True
            if self._session is not None:
                self._binding.byte_lm_parallel_close(self._session)
                self._session = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def ordered_sum_gradients(parts):
    """Copy shard zero, then add shards 1..K-1 using IDENTICAL GPU pair adds.

    Separate from training.accumulate_grads, whose balanced-tree contract is
    unchanged. Neither this fold nor the byte-LM fold divides by shard count.
    """
    from ._training_impl import accumulate_grads
    parts = list(parts)
    if not parts:
        raise ValueError('ordered reduction needs at least one shard')
    total = [x.copy() for x in parts[0]]
    for part in parts[1:]:
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
                    for inputs, targets in shards:
                        inputs = self.model._ids(inputs, 'inputs')
                        targets = self.model._ids(targets, 'targets')
                        requests.append((self._operation, snapshot, (inputs, targets, stream, offset)))
                        offset += inputs.size
                else:
                    from ._mlp_impl import _batch, _targets
                    for inputs, targets in shards:
                        inputs = _batch(inputs)
                        targets = _targets(targets, len(inputs))
                        requests.append((self._operation, snapshot, (inputs, targets)))
                results = self._pool.map(requests)
                losses = tuple(part[0] for part in results)
                update_state = snapshot
                if self._operation == 'samba_gradient':
                    update_state = dict(snapshot, rng=self.model.generator.state_dict())
                # The cooperative worker owns gradient columns, whole clipping
                # tensors and update ranges, retaining the original norm tree.
                updated, retained, step = self._update_pool.map([(
                    self._operation.replace('_gradient', '_update'), update_state,
                    [part[1] for part in results])])[0]
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
            return [g.copy() for g in self._gradients]

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
