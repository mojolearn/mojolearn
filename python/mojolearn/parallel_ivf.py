# SPDX-License-Identifier: Apache-2.0
"""Disjoint IVF candidate storage with replicated coarse centers.

A fitted/loaded index is partitioned without retraining its quantizer. Worker
processes retain only their candidate rows and keep that shard resident on
their GPU from the first search (lane/py-dn-ann, 2026-09-28). Coarse centers and queries are replicated. Building the original
index and extending a distributed index are outside this API.
"""
__all__ = ['DistributedIVFIndex']

from ._ivf_impl import IVFIndex
from . import _native_parallel_shard as _shard
from ._native_parallel_shard import _native_parallel_shard

# Import-time class metadata; admission, storage and partial searches are native.
_shard.register(ivf_class=IVFIndex)


class DistributedIVFIndex:
    """Context-managed search over disjoint stored candidate shards.

    ``from_index(index, devices=(0, 1))`` preserves the global coarse quantizer
    and its selected probes. Local IDs are monotonic maps of original IDs, so
    exact-distance ties retain original row order. Final selection uses squared
    distances; Euclidean rooting occurs only after the global order is fixed.
    Individual shards and all coarse centers must each fit on one GPU.
    """
    def __init__(self):
        _native_parallel_shard('ivf_init', (self,))

    @classmethod
    def from_index(cls, index, *, devices=(0, 1)):
        return _native_parallel_shard('ivf_from_index', (cls, index, devices))

    def search(self, queries):
        # One native call (bindings/parallel_shard_native.mojo `ivf_search`).
        return _native_parallel_shard('ivf_search', (self, queries))

    def close(self):
        """Stop the worker processes and release the stored shards."""
        _native_parallel_shard('ivf_close', (self,))

    def __enter__(self):
        return _native_parallel_shard('ivf_enter', (self,))

    def __exit__(self, *exc):
        _native_parallel_shard('ivf_close', (self,))


_shard.register(distributed_ivf_class=DistributedIVFIndex)
