# SPDX-License-Identifier: Apache-2.0
"""`operator.index` through the one CPython table, for the tokenizer, causal
LM and byte LM bindings. The table this module held moved into
`bindings/py_cpython_table.mojo` (lane ne1-e-table, 2026-11-06); call sites use
`cpython_table()[].<Name>`."""

from std.python import PythonObject
from bindings.py_cpython_table import cpython_table


def lm_try_index(value: PythonObject, mut word: Int) raises -> Bool:
    """`operator.index(value)` as a 64-bit word through the table: True and
    `word` set on success. False (with no Python error pending) for the bool
    singletons and for any value that is not an index or does not fit, so
    the caller runs its original admission path and keeps its wording."""
    if value is PythonObject(True) or value is PythonObject(False):
        return False
    var table = cpython_table()
    var indexed = table[].PyNumber_Index(value._obj_ptr)
    if not indexed:
        table[].PyErr_Clear()
        return False
    var owned = PythonObject(from_owned=indexed)
    var result = table[].PyLong_AsLongLong(owned._obj_ptr)
    if result == -1 and table[].PyErr_Occurred():
        table[].PyErr_Clear()
        return False
    # NumPy bool scalars implement __index__ (as 0 or 1) before NumPy 2.3, the
    # NumPy that Python 3.10 resolves; 2.3+ refuses them. Refuse them here on
    # every NumPy so a bool is never admitted as an integer. Only a 0/1 result
    # from an object that is not itself the int pays the type-name read;
    # an exact int comes back from PyNumber_Index as the same object.
    if (result == 0 or result == 1) and not (owned is value):
        var type_name = String(py=value.__class__.__name__)
        if type_name == "bool" or type_name == "bool_":
            return False
    word = Int(result)
    return True
