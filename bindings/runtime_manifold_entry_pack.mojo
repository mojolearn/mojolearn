# SPDX-License-Identifier: Apache-2.0
# Manifold orchestration and its helpers compile in one native owner. Existing
# matrix/device owners are passed by retained Python references; no MKit value
# or exposed Mojo type crosses this C-entry boundary.
from std.python import PythonObject
from std.python._cpython import PyObjectPtr
from std.sys.compile import is_defined
from bindings.classical_error_boundary_pack import classical_python_error
from bindings.manifold_native_pack import manifold_operation, manifold_unknown_operation
from bindings.native_operation_codes import FR_MANI, family_route_owner
from bindings.native_operation_names_unpack import operation_lookup, operation_code

# This owner reports the existing whole-host-set negative-control flag. Its
# connected kernel owners provide the arithmetic controls; the extraction
# adds no numerical branch or standalone owner control.
comptime RUNTIME_MANIFOLD_SABOTAGE = is_defined["MOJOLEARN_HOST_SABOTAGE"]()


def runtime_manifold_c(py_self: PyObjectPtr, args_ptr: PyObjectPtr) abi("C") -> PyObjectPtr:
    try:
        var call = PythonObject(from_borrowed=args_ptr)
        # One dictionary lookup of the caller's own str (lane bw2-manifold;
        # was a String copy, a prefix test and the owners' name chains per
        # call). Only a name that is not a manifold name of the table builds
        # its String, for the unchanged error texts.
        var names = operation_lookup()
        var code = operation_code(names, call[1])
        if code < 0 or family_route_owner(names[].table.family_routes[code]) != FR_MANI:
            var name = String(call[1])
            if not name.startswith("family_mani_"):
                raise Error("RuntimeError: unknown family operation " + name)
            manifold_unknown_operation(name)
        var result = manifold_operation(call[0], names[].table.leaves[code], call[2], call[3])
        return result.steal_data()
    except error:
        return classical_python_error(error^, args_ptr)
