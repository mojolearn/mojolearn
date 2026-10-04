# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Large float32 outputs on lazily zeroed memory (lane neighbors-apple3).

`_buffer.empty` allocates an `array.array`, whose zeros one core writes
before the kernel's answer overwrites them: about 0.1 s at 2 GB on the M4
Pro (RBFSampler.transform and SkewedChi2Sampler.transform at 1M x 500).
`_empty_out` backs an output of 64 MB or more by an anonymous memory map
instead; the system hands out its zero pages when they are first written.
The same zeros and the same Array (`Array.from_buffer`, which keeps the map
alive).

OPT-IN until its A/B passes: MOJOLEARN_XN_LAZY_OUT=1. Used by the neighbors
family's outputs (kernel_methods.py, _expansion_neighbors.py).
"""
import os

from ._array import Array
from ._buffer import empty

_LAZY_OUT = os.environ.get("MOJOLEARN_XN_LAZY_OUT", "") == "1"
_LAZY_OUT_MIN_BYTES = 1 << 26


def _mapped_out(shape):
    """A float32 output backed by an anonymous map when it is 64 MB or more
    (else `empty`), whatever MOJOLEARN_XN_LAZY_OUT says: the route a binding
    compiled with an opt-in define chooses (lane apple-fast-w2-kfeat,
    RBFSampler.fit_transform under MOJOLEARN_KM_FAST_RBF_RESIDENT), so no
    core writes zeros the device copy then overwrites."""
    dims = tuple(int(v) for v in (shape if isinstance(shape, (tuple, list)) else (shape,)))  # glue: validates the shape argument
    count = 1
    for v in dims:  # glue: product of shape dimensions
        count *= v
    if count * 4 >= _LAZY_OUT_MIN_BYTES:
        import mmap
        store = mmap.mmap(-1, count * 4, flags=mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS)
        return Array.from_buffer(memoryview(store).cast("f")).reshape(dims)
    return empty(shape, "<f4")


def _empty_out(shape, dtype="<f4"):
    """`empty(shape, dtype)` for an output a kernel will fill."""
    if _LAZY_OUT and dtype == "<f4":
        dims = tuple(int(v) for v in (shape if isinstance(shape, (tuple, list)) else (shape,)))  # glue: validates the shape argument
        count = 1
        for v in dims:  # glue: product of shape dimensions
            count *= v
        if count * 4 >= _LAZY_OUT_MIN_BYTES:
            import mmap
            store = mmap.mmap(-1, count * 4, flags=mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS)
            return Array.from_buffer(memoryview(store).cast("f")).reshape(dims)
    return empty(shape, dtype)
