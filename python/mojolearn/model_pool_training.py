# SPDX-License-Identifier: Apache-2.0
"""Layer-partitioned byte-LM training with ordered microbatch replay."""
import threading

from .parallel_training import ParallelByteLanguageModelTrainer
from ._byte_lm_impl import SmallByteLanguageModelTrainer, _validate_state
from ._byte_lm_config import state_shape
from ._buffer import addr_ro


class _ModelPoolBinding:
    """Use the existing step/export transaction adapter with a model owner."""

    def __init__(self, binding):
        for name in ('create', 'open', 'close', 'step', 'export', 'rollback', 'ownership'):
            function = getattr(binding, 'byte_lm_model_pool_' + name, None)
            if not callable(function):
                raise ImportError('rebuild bindings/build_byte_lm.sh for model pooling')
            setattr(self, 'byte_lm_parallel_' + name, function)


class PooledByteLanguageModelTrainer(ParallelByteLanguageModelTrainer):
    """One byte LM whose layers and optimizer state span the selected GPUs.

    Decoder layers have one owner. Embedding/head live on devices[0]; each
    individual layer and the head must fit its owner. This first schedule runs
    layers sequentially. It pools model capacity and makes no speedup claim.
    Logical microbatch count/order is independent of physical device count.
    Checkpoints use the same ordered-sum contract as the replica trainer.
    """

    def __init__(self, state, *, devices=(0,), logical_shards=1):
        devices = tuple(devices)
        if type(logical_shards) is not int or not 1 <= logical_shards <= 1024:
            raise ValueError('logical_shards must be in [1, 1024]')
        if any(type(i) is not int or i < 0 for i in devices) or len(set(devices)) != len(devices):
            raise ValueError('devices must be distinct nonnegative integer indices')
        self._state = _validate_state(state)
        self._shape = state_shape(self._state)
        if not 1 <= len(devices) <= min(64, self._shape.n_layers):
            raise ValueError('require 1 <= device count <= min(64, n_layers)')
        self.devices = devices
        self.logical_shards = logical_shards
        self._session = None
        self._binding = None
        self._closed = False
        self._lost = False
        self._lock = threading.RLock()

    def _open(self):
        if self._closed or self._lost:
            raise RuntimeError('pooled trainer is closed or lost; restore a state export')
        if self._session is not None:
            return
        seed = self._state
        helper = SmallByteLanguageModelTrainer(seed['parameters'],
            data_schedule=seed['data_schedule'], shape=self._shape)
        binding = _ModelPoolBinding(helper._binding())
        cfg = seed['config']
        params = [0, self.step_, cfg['kind'], cfg['lr'], cfg['beta1'], cfg['beta2'],
                  cfg['eps'], cfg['weight_decay'], cfg['momentum'], cfg['dampening'],
                  int(cfg['nesterov']), cfg['max_norm']]
        session = binding.byte_lm_parallel_create()
        try:
            completed = binding.byte_lm_parallel_open(session,
                [addr_ro(seed[k], name=k) for k in ('parameters', 'm', 'v', 'flags')],
                params, list(self._shape.native_shape), list(self.devices), self.logical_shards)
            if completed != self.step_:
                raise RuntimeError('model pool admission returned wrong step')
        except BaseException:
            binding.byte_lm_parallel_close(session)
            raise
        self._binding, self._session = binding, session
        self._state = dict(seed, parameters=None, m=None, v=None)

    def state_dict(self, *, rank=0):
        """Export the complete canonical state; there are no model replicas."""
        return super().state_dict(rank=rank)

    def model_ownership(self):
        """Actual canonical parameter/state allocations for each owned chunk.

        Counts exclude copied kernel weights, activations and workspaces;
        they are not a total GPU-memory estimate.
        """
        with self._lock:
            self._open()
            rows = self._binding.byte_lm_parallel_ownership(self._session)
            return tuple(dict(device=self.devices[int(row[0])], first=int(row[1]),
                              count=int(row[2]), parameter_bytes=int(row[3]),
                              moment_bytes=int(row[4]), rollback_bytes=int(row[5]),
                              gradient_bytes=int(row[6])) for row in rows)

    def optimizer_ownership(self):
        return self.model_ownership()

    @classmethod
    def from_checkpoint(cls, checkpoint, *, devices=(0,)):
        if (checkpoint.get('schema') != 'mojolearn.parallel-byte-lm.v1'
                or checkpoint.get('reduction') != 'ordered_sum'):
            raise ValueError('unsupported parallel checkpoint contract')
        return cls(checkpoint['state'], devices=devices,
                   logical_shards=checkpoint['logical_shards'])
