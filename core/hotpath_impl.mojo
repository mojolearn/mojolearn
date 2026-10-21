# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Typed kernels of the shared host helpers (`bindings/hotpath_helpers.mojo`).

The Python door stays in `bindings/hotpath_helpers.mojo`: it unpacks
PythonObject arguments, calls one typed function here and converts the
result back. Everything that computes (casts, comparisons, counts, folds,
draws) lives here with no Python import, word for word as it was in the
binding (lane thin-core, 2026-10-21: a pure move, no bit change).
"""
from std.math import fma, isfinite, sqrt
from std.memory import bitcast
from std.sys.compile import is_defined
from sequence.schedule import fill_epoch_order, splitmix64
from checks.numerics import portable_cosf, portable_log64
from metrics.checks.pinned_sum import canonicalize_nan


#: THE NEGATIVE CONTROL. `-D MOJOLEARN_HOST_SABOTAGE=1` is the core host
#: binding's existing sabotage define, and `-D MOJOLEARN_HOTPATH_SABOTAGE=1`
#: sabotages THESE HELPERS ALONE, so a divergence under it cannot be the
#: k-NN or k-means fold's (`core_host_sabotage()` reports either and
#: `_backend` refuses to load such a binary without
#: MOJOLEARN_HOST_ALLOW_SABOTAGE=1). Under either every helper here answers
#: WRONG ON PURPOSE in a way that keeps its output well formed: the cast
#: writes each run of eight elements reversed, min and max trade places,
#: the float sum folds descending, argmax takes the LAST maximum, equality
#: is inverted, the encoder's codes are reversed, the
#: gather reads the next table slot, a duplicate index is accepted, an
#: overlap is denied and the fold ids are rotated by one.
#: `tests/test_hotpath_native.py` must FAIL against such a build in every
#: group; a differential test that passes against it compares nothing.
comptime HOTPATH_SABOTAGE = (
    is_defined["MOJOLEARN_HOST_SABOTAGE"]() or is_defined["MOJOLEARN_HOTPATH_SABOTAGE"]()
)

#: dtype codes shared with `python/mojolearn/_array.py::_NATIVE_CODE`.
comptime HP_F32 = 0
comptime HP_F64 = 1
comptime HP_I32 = 2
comptime HP_I64 = 3
comptime HP_U32 = 4
comptime HP_U8 = 5
#: lane py-runtime round 2: bf16 bits (uint16), int8 codes and float16, the
#: dtypes whose Python routines were the last ones left in `_array.py`
comptime HP_U16 = 6
comptime HP_I8 = 7
comptime HP_F16 = 8
#: equal_elements only (lane py-runtime round 3): a int64, b float64, exact
comptime HP_I64_F64 = 9

comptime HP_W = 8


@always_inline
def _ptr[dt: DType](addr: Int) raises -> MutPointer[Scalar[dt], MutUntrackedOrigin]:
    if addr == 0:
        raise Error("mojolearn: null buffer address")
    return MutPointer[Scalar[dt], MutUntrackedOrigin](unsafe_from_address=addr)


def _tasks(n: Int) -> Int:
    """One task on the calling thread (cpu-gpu-cleanup c-core: the GPU
    binding runs no host pool; the SIMD loop is a copy/compare)."""
    return 1
