# SPDX-License-Identifier: Apache-2.0
# Constant words of a loaded preprocessing binding, read once per binding.
#
# Getters such as `preprocessing_numeric_mode`, `x_prep_numeric_mode`,
# `x_prep_idn_fam` or `x_prep_c08_routes` return a compile-time constant of the
# binary they belong to. The preprocessing doors used to call them on every
# public operation (two to four extra native round trips per fit, transform or
# predict). The first operation now calls the getter and stores its word in
# the import-time entries dict under a "cached:" key (the same dict that
# already retains the compiled entries; small_mlp keeps its provider there the
# same way), so every later operation reads one dict slot instead.
from std.python import PythonObject
from bindings.py_balanced import py_get, py_setitem
from bindings.py_typed_call import py_call_typed


def prep_entry_word(entries: PythonObject, name: String, missing: Int = 0) raises -> Int:
    """The integer the compiled getter `entries[name]` returns, or `missing`
    when the binding does not export it. Called once per binding and name."""
    var key = PythonObject(String("cached:") + name)
    var cached = py_get(entries, key)
    if cached is not None:
        return Int(py=cached)
    var entry = py_get(entries, PythonObject(name))
    var word = missing if entry is None else Int(py=py_call_typed(entry))
    py_setitem(entries, key, PythonObject(word))
    return word


def prep_entry_text(entries: PythonObject, name: String) raises -> String:
    """The string the compiled getter `entries[name]` returns, or "" when the
    binding does not export it. Called once per binding and name."""
    var key = PythonObject(String("cached:") + name)
    var cached = py_get(entries, key)
    if cached is not None:
        return String(py=cached)
    var entry = py_get(entries, PythonObject(name))
    var text = String("") if entry is None else String(py=py_call_typed(entry))
    py_setitem(entries, key, PythonObject(text))
    return text


def prep_mode_checked(entries: PythonObject, expected: Int) raises -> Bool:
    """True when both the preprocessing and the x_prep binding of `entries`
    report the numeric mode `expected` (1 identical, 0 fast). The two native
    probes run once per binding; later calls compare the cached words."""
    return (prep_entry_word(entries, "preprocessing_numeric_mode", -1) == expected
            and prep_entry_word(entries, "x_prep_numeric_mode", -1) == expected)
