# SPDX-License-Identifier: Apache-2.0
# The explicit CPU density digest hashes stored model fields, not NPZ bytes.
# Array.tobytes uses physical storage order; ndarray.tobytes uses logical C
# order. Preserve that distinction while owning iteration and SHA-256 here.
from std.python import Python, PythonObject
from std.python._cpython import PyObjectPtr
from bindings.py_balanced import py_call, py_type, py_get, py_contains
from bindings.classical_attributes import classical_connect_numpy, classical_getattribute
from bindings.host_tree_native import host_tree_constructor_fields
from bindings.native_numpy_metadata_unpack import native_numpy_dtype
from bindings.sequential_admission import _one_of
from bindings.optimizer_public_words_unpack import stash_python_error
from core.sha256 import Sha256


from bindings.host_density_hash_native_pack import (
    _host_fields_hash,
)


def host_density_hash(model: PythonObject, args: PythonObject, context: PythonObject) raises -> PythonObject:
    classical_connect_numpy(context)
    _ = host_tree_constructor_fields(model, context)
    var family = String(args[0])
    var plan = context["classical_public_layouts"][py_type(model)]
    var mro = plan["mro"]
    var known = False
    for i in range(len(mro)):  # driver-loop: registered density class ancestry
        var kind = py_get(context["host_inference_kinds"], mro[i])
        known = known or (kind is not None and String(kind) == family)
    if not known or (family != "kde" and family != "iforest"):
        raise Error("TypeError: density hash requires its connected native host class")
    return _host_fields_hash(model, context)


def host_bound_hash(model: PythonObject, args: PythonObject, context: PythonObject) raises -> PythonObject:
    """`_HostBound.model_sha256` for every explicit host model without its own
    digest: SHA-256 over the declared `_HOST_ARRAYS` fields in sorted name
    order, a `None` field skipped, an Array's physical bytes and a NumPy
    array's logical C-order bytes (the Python loop over `hashlib` it replaces,
    lane pyout-runtime-boundary 2026-11-11)."""
    classical_connect_numpy(context)
    return _host_fields_hash(model, context)
