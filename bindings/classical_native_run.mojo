# SPDX-License-Identifier: Apache-2.0
# RUN part of bindings/classical_native.mojo: typed Mojo only (no Python
# imports); the door in classical_native.mojo keeps the unpack and pack.
from bindings.native_operation_codes import operation_table, OP_ROUTE_TREES, OP_ROUTE_HOST_TREE


def classical_coded_trees(code: Int) raises -> Bool:
    """Whether the operation with this integer code is the tree door's (else
    the family door's)."""
    var table = operation_table()
    if code < 0 or code >= len(table[].names):
        raise Error("TypeError: unknown native operation code")
    var route = table[].routes[code]
    return route == OP_ROUTE_TREES or route == OP_ROUTE_HOST_TREE
