# SPDX-License-Identifier: Apache-2.0
# UNPACK of an operation NAME into its code (lane bw2-core, 2026-11-20): the
# str half of bindings/native_operation_codes.mojo.
#
# WHAT IT REPLACES. A door that received the operation as a str copied it into
# a Mojo String (`String(request[1])`), and every dispatcher under it walked a
# chain of prefix tests on that String (`family_route`: up to 45
# `startswith`; then the owner's own prefix and exact-name chains), on every
# call. A dispatcher that forwards to another owner could not use a family's
# own `ops` dictionary (bindings/py_keys.mojo), because the name belongs to
# the owner it forwards to.
#
# THE FORM. `operation_code(table, name)` is ONE `PyDict_GetItemWithError`
# with the caller's own str object in the table's `lookup` dictionary (every
# name of the table, interned, to its code). The code indexes the table's
# columns: `routes`, `family_routes`, `leaves` (the converted owner's integer)
# and `names` (the table's own String, borrowed: for an owner that is not
# converted yet, and for error text). -1 is a name the table does not hold;
# only that path builds the name String.
#
# COST (AGENTS.md "No refactor adds cost on a hot path"). Per call: the slot
# fetch the coded path already made, one flag test and one dictionary lookup,
# in place of a String build and the prefix chain. The dictionary is filled
# once per process, at the first str call.
from std.python import Python, PythonObject
from bindings.native_operation_codes import OperationTable, OPERATION_TABLE
from bindings.py_keys import py_op_add, py_op_code


def _operation_lookup_fill(table: UnsafePointer[OperationTable, MutUntrackedOrigin]) raises:
    # Once per process. `lookup_ready` is set last, so a fill that raised is
    # run again whole by the next call.
    var lookup = Python.dict()
    for i in range(len(table[].names)):  # driver-loop: over the fixed operation names, once per process
        py_op_add(lookup, table[].names[i], i)
    table[].lookup = lookup
    table[].lookup_ready = True


@always_inline
def operation_lookup() raises -> UnsafePointer[OperationTable, MutUntrackedOrigin]:
    """The process-wide operation table with its name dictionary filled: one
    slot fetch and one flag test. Fetch once per call, outside loops."""
    var table = OPERATION_TABLE.get_or_create_ptr()
    if not table[].lookup_ready:
        _operation_lookup_fill(table)
    return table


@always_inline
def operation_code(table: UnsafePointer[OperationTable, MutUntrackedOrigin],
                   name: PythonObject) raises -> Int:
    """The code of an operation name (the caller's own str object), or -1 for
    a name the table does not hold. No String is built."""
    return py_op_code(table[].lookup, name)
