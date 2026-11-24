# SPDX-License-Identifier: Apache-2.0
"""Disjoint IVF candidate storage with replicated coarse centers.

A fitted/loaded index is partitioned without retraining its quantizer. Worker
processes retain only their candidate rows and keep that shard resident on
their GPU from the first search (lane/py-dn-ann, 2026-09-28). Coarse centers and queries are replicated. Building the original
index and extending a distributed index are outside this API.
"""
__all__ = ['DistributedIVFIndex']

from ._ivf_impl import IVFIndex as _IVFIndex
from . import _native_parallel_shard as _shard
from ._native_parallel_shard import _native_parallel_shard

# Import-time class metadata; admission, storage and partial searches are native.
_shard.register(ivf_class=_IVFIndex)


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
    def from_index(cls, index, *, devices=(0, 1), device=None):
        """Split a fitted ``IVFIndex``'s stored candidates across GPUs.

        Parameters
        ----------
        index : IVFIndex
            A fitted index.
        devices : tuple of int, default=(0, 1)
            Distinct GPU indices; each holds one shard.
        device : {None, 'gpu', 'cpu'}, default None
            Where the computation runs. None and 'gpu' run on the GPUs in
            ``devices``; 'cpu' raises ValueError on a GPU installation, since this
            driver has no CPU form. Other values raise ValueError.

        Returns
        -------
        DistributedIVFIndex
        """
        return _native_parallel_shard('ivf_from_index', (cls, index, devices, device))

    def search(self, queries):
        # One native call (bindings/parallel_shard_native_door.mojo `ivf_search`).
        return _native_parallel_shard('ivf_search', (self, queries))

    def close(self):
        """Stop the worker processes and release the stored shards."""
        _native_parallel_shard('ivf_close', (self,))

    def __enter__(self):
        return _native_parallel_shard('ivf_enter', (self,))

    def __exit__(self, *exc):
        _native_parallel_shard('ivf_close', (self,))


_shard.register(distributed_ivf_class=DistributedIVFIndex)
