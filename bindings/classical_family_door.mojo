# SPDX-License-Identifier: Apache-2.0
# The `family_*` operations of `classical_native_call` as their own compiled
# door (2026-11-07 host split; docs/build/HOT_GROUPS.md section 6).
# bindings/classical_native.mojo forwards them here with the caller's own
# argument tuple (PyObject_Call of `context["classical_family_door"]`, no
# Python frame): the family admission, the dispatcher
# (bindings/family_native.mojo) and the error mapping are the ones the
# classical door ran before, moved unchanged.
from std.python import PythonObject
from std.python._cpython import PyObjectPtr
from bindings.classical_error_boundary_pack import classical_python_error
from bindings.family_native_door import family_operation, family_operation_routed
from bindings.native_operation_codes import operation_table, OP_ROUTE_FAMILY
from bindings.native_operation_names_unpack import operation_lookup, operation_code
from bindings.py_balanced import py_type
from bindings.admitted_call_door import AdmittedCall
from providers.python.classical_family_admission_pyapi_unpack import classical_family_admit
from bindings.classical_door_routes import classical_family_operation


def _family_coded(model: PythonObject, code: Int, args: PythonObject, context: PythonObject) raises -> PythonObject:
    """A `family_*` operation by its integer code: the route word and the
    name come from the table, as the classical door's coded path read them."""
    var table = operation_table()
    if code < 0 or code >= len(table[].names):
        raise Error("TypeError: unknown native operation code")
    var route = table[].routes[code]
    if route != OP_ROUTE_FAMILY:
        # A tree code, or a classical-door string operation's code
        # (OP_ROUTE_CLASSICAL, lane bw2-classical): not a family operation.
        raise Error("TypeError: native operation code is not a family operation")
    _ = classical_family_admit(model)
    # bw2-core: the name is the table's own String, borrowed (it was copied
    # per call); a converted owner reads the leaf code and never the name.
    return family_operation_routed(model, table[].family_routes[code], table[].leaves[code], table[].names[code],
                                   args, context)


def classical_family_c(py_self: PyObjectPtr, args_ptr: PyObjectPtr) abi("C") -> PyObjectPtr:
    """`family_native_call(model, operation, args, context)`: the same four
    arguments `classical_native_call` received."""
    try:
        var request = PythonObject(from_borrowed=args_ptr)
        if len(request) != 4:
            raise Error("TypeError: family_native_call requires model, operation, arguments and context")
        var call = AdmittedCall()
        var b = call.builtins()
        if py_type(request[1]) is b.int and py_type(request[3]) is b.dict:
            var coded = _family_coded(request[0], Int(py=request[1]), request[2], request[3])
            return coded.steal_data()
        if py_type(request[1]) is not b.str or py_type(request[3]) is not b.dict:
            raise Error("TypeError: native classical operations require an exact operation string and context dictionary")
        var model = request[0]
        # bw2-core: a name of the operation table is ONE dictionary lookup of
        # the caller's own str (it was copied into a String and walked through
        # the prefix chain of `family_route`, then the owner's chains, per
        # call). Only a name the table does not hold (or a tree name, refused
        # below as before) builds its String, for the old route and the
        # unknown-operation error text.
        var table = operation_lookup()
        var code = operation_code(table, request[1])
        if code >= 0 and table[].table.routes[code] == OP_ROUTE_FAMILY:
            _ = classical_family_admit(model)
            var routed = family_operation_routed(model, table[].table.family_routes[code],
                                                 table[].table.leaves[code], table[].table.names[code],
                                                 request[2], request[3])
            return routed.steal_data()
        var operation = String(request[1])
        if not classical_family_operation(operation):
            raise Error("RuntimeError: unknown family operation " + operation)
        _ = classical_family_admit(model)
        var family = family_operation(model, operation, request[2], request[3])
        return family.steal_data()
    except error:
        return classical_python_error(error^, args_ptr)
