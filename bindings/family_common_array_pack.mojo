# SPDX-License-Identifier: Apache-2.0
# PACK helper shared by the cluster, harmony and marker doors: one flat `array.array` result becomes an
# Array's store without a second copy. Moved unchanged out of bindings/cluster_public_native_pack.mojo
# (lane door-cut-1, 2026-11-28) so that a door in another family does not import the cluster query door,
# and with it the cluster state, for this one function. `cluster_keys` is the key table it always used.
from std.python import PythonObject
from bindings.py_balanced import py_call, py_type, py_item
from bindings.classical_arrays_door import classical_wrap_array
from bindings.cluster_keys_unpack import cluster_keys


def _xc_array(flat: PythonObject, shape: PythonObject, typestr: String, code: String,
              context: PythonObject) raises -> PythonObject:
    """`Array._from_flat(flat, shape, typestr)` without its copy: the
    entry returns each result as one `array.array` of the dtype's store code
    (providers/python/x_cluster_out_pyapi.mojo, one memcpy from the result), which becomes the
    Array's store as it is. Anything else is copied once, as `_from_flat` did.
    Same values either way."""
    var store = flat
    var nk = cluster_keys()
    if py_type(flat) is not py_item(context, nk[].k_array_type) or String(py=flat.typecode) != code:
        store = py_call(py_item(context, nk[].k_array_type), PythonObject(code), flat)
    return classical_wrap_array(store, shape, typestr, "C", context)
