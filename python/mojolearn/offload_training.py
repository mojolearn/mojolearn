# SPDX-License-Identifier: Apache-2.0
"""Memory-bounded single-GPU replay of ordered byte-LM checkpoints."""
from .parallel_training import ParallelByteLanguageModelTrainer, _native_parallel_byte


class OffloadedByteLanguageModelTrainer(ParallelByteLanguageModelTrainer):
    """Replay a pooled checkpoint with one decoder layer on one GPU at a time.

    Parameters, optimizer state, gradient sums and saved activations live in
    host memory. Every arithmetic operation still runs the original GPU
    kernels. A decoder layer plus the embedding/head must fit the device.
    The schedule trades transfers and recomputation for GPU memory; it does
    not promise pooled-run throughput. Logical shard order is preserved.
    """

    def __init__(self, state, *, devices=(0,), logical_shards=1):
        # One native admission for every byte-LM schedule; this exact class,
        # admitted at import, selects the one-device layer-at-a-time session
        # (bindings/parallel_byte_native.mojo).
        return _native_parallel_byte("init", (self, state, devices, logical_shards, False))

    def _open(self):
        return _native_parallel_byte("open", (self,))

    def state_dict(self, *, rank=0):
        """Export the complete canonical state; there are no model replicas."""
        return _native_parallel_byte("state", (self, rank))

    def optimizer_ownership(self):
        """Refused: offloaded state lives on the host, with no resident GPU
        optimizer owners."""
        return _native_parallel_byte("ownership", (self,))

    @classmethod
    def from_checkpoint(cls, checkpoint, *, devices=(0,)):
        return _native_parallel_byte("from_checkpoint", (cls, checkpoint, devices, False))


_native_parallel_byte("admit_class", (OffloadedByteLanguageModelTrainer, "offload"))  # glue: import-time class admission
