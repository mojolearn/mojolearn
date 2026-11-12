# SPDX-License-Identifier: Apache-2.0
# Typed native matrix/device lifetimes. No Python _M attribute hook or _DevBuf
# finalizer participates in a manifold operation, including temporary cleanup.
# The PACK half of the former `bindings/manifold_storage_native.mojo` (fence-flip split, 2026-11-12);
# the UNPACK half is `bindings/manifold_storage_native_unpack.mojo`.
from std.ffi import c_int
from std.python import Python, PythonObject
from std.python._cpython import ExternalFunction, PyObjectPtr
from bindings.cpython_table import cpython_table

# The C entries a release and the kit lists use come from the one CPython
# table (ManifoldCalls, MojolearnManifoldCallsV1, folded into
# bindings/cpython_table.mojo by lane ne1-e-table, 2026-11-06).


from bindings.manifold_storage_native_unpack import (
    ManifoldMatrix,
)


def words_new(count: Int) raises -> PythonObject:
    """A words holder of `count` zero float32 values."""
    return PythonObject(alloc=ManifoldMatrix(words_of=count))
