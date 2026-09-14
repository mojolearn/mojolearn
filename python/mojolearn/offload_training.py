# SPDX-License-Identifier: Apache-2.0
"""Memory-bounded single-GPU replay of ordered byte-LM checkpoints."""
import threading

from .parallel_training import ParallelByteLanguageModelTrainer
from ._byte_lm_impl import _load, _validate_state
from ._byte_lm_config import state_shape
from ._buffer import addr_ro


class _OffloadBinding:
    """Use the existing step/export transaction adapter with a model owner."""

    def __init__(self, binding):
        for name in ('create', 'open', 'close', 'step', 'export', 'rollback'):
            function = getattr(binding, 'byte_lm_offload_' + name, None)
            if not callable(function):
                raise ImportError('rebuild bindings/build_byte_lm.sh for offloaded replay')
            setattr(self, 'byte_lm_parallel_' + name, function)


class OffloadedByteLanguageModelTrainer(ParallelByteLanguageModelTrainer):
    """Replay a pooled checkpoint with one decoder layer on one GPU at a time.

    Parameters, optimizer state, gradient sums and saved activations live in
    host memory. Every arithmetic operation still runs the original GPU
    kernels. A decoder layer plus the embedding/head must fit the device.
    The schedule trades transfers and recomputation for GPU memory; it does
    not promise pooled-run throughput. Logical shard order is preserved.
    """

    def __init__(self, state, *, devices=(0,), logical_shards=1):
        devices = tuple(devices)
        if type(logical_shards) is not int or not 1 <= logical_shards <= 1024:
            raise ValueError('logical_shards must be in [1, 1024]')
        if any(type(i) is not int or i < 0 for i in devices) or len(set(devices)) != len(devices):
            raise ValueError('devices must be distinct nonnegative integer indices')
        self._state = _validate_state(state)
        self._shape = state_shape(self._state)
        if len(devices) != 1:
            raise ValueError('offloaded replay requires exactly one device')
        self.devices = devices
        self.logical_shards = logical_shards
        self._session = None
        self._binding = None
        self._closed = False
        self._lost = False
        self._lock = threading.RLock()

    def _open(self):
        if self._closed or self._lost:
            raise RuntimeError('offloaded trainer is closed or lost; restore a state export')
        if self._session is not None:
            return
        seed = self._state
        binding = _OffloadBinding(_load(self._shape))
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
                raise RuntimeError('offloaded replay admission returned wrong step')
        except BaseException:
            binding.byte_lm_parallel_close(session)
            raise
        self._binding, self._session = binding, session
        self._state = dict(seed, parameters=None, m=None, v=None)

    def state_dict(self, *, rank=0):
        """Export the complete canonical state; there are no model replicas."""
        return super().state_dict(rank=rank)

    def optimizer_ownership(self):
        raise NotImplementedError('offloaded state lives on the host; no resident GPU optimizer owners')

    @classmethod
    def from_checkpoint(cls, checkpoint, *, devices=(0,)):
        if (checkpoint.get('schema') != 'mojolearn.parallel-byte-lm.v1'
                or checkpoint.get('reduction') != 'ordered_sum'):
            raise ValueError('unsupported parallel checkpoint contract')
        return cls(checkpoint['state'], devices=devices,
                   logical_shards=checkpoint['logical_shards'])
