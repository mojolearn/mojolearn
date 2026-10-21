# SPDX-License-Identifier: Apache-2.0
"""Layer-partitioned byte-LM training with ordered microbatch replay."""
from .parallel_training import ParallelByteLanguageModelTrainer, _native_parallel_byte


class PooledByteLanguageModelTrainer(ParallelByteLanguageModelTrainer):
    """One byte LM whose layers and optimizer state span the selected GPUs.

    Decoder layers have one owner. Embedding/head live on devices[0]; each
    individual layer and the head must fit its owner. This first schedule runs
    layers sequentially. It pools model capacity and makes no speedup claim.
    Logical microbatch count/order is independent of physical device count.
    Checkpoints use the same ordered-sum contract as the replica trainer.
    """

    def __init__(self, state, *, devices=(0,), logical_shards=1):
        # One native admission for every byte-LM schedule; "model_pool"
        # selects the layer-owner session (bindings/parallel_byte_native.mojo).
        return _native_parallel_byte("init", (self, state, devices, logical_shards, False, "model_pool"))

    def _open(self):
        return _native_parallel_byte("open", (self,))

    def state_dict(self, *, rank=0):
        """Export the complete canonical state; there are no model replicas."""
        return super().state_dict(rank=rank)

    def model_ownership(self):
        """Actual canonical parameter/state allocations for each owned chunk.

        Counts exclude copied kernel weights, activations and workspaces;
        they are not a total GPU-memory estimate.
        """
        return _native_parallel_byte("ownership", (self,))

    def optimizer_ownership(self):
        return self.model_ownership()

    @classmethod
    def from_checkpoint(cls, checkpoint, *, devices=(0,)):
        return _native_parallel_byte("from_checkpoint", (cls, checkpoint, devices, False, "model_pool"))
