# SPDX-License-Identifier: Apache-2.0
"""tools/boundary_keys_plain.py: a held-key door reads back as its by-name form, line for line."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from boundary_keys_plain import is_keys_file, plain  # noqa: E402

DOOR = "bindings/tsa_public_native_door.mojo"


def test_reads_and_writes_come_back_by_name():
    text = "\n".join([
        "    var nk = tsa_keys()",
        "    var mode = py_item(py_item(context, nk[].k_backend_fields), nk[].k__DEFAULT_MODE)",
        "    var init = py_item(context, nk[].k_hw_init_codes)[py_item(fields, nk[].k_initialization_method)]",
        "    var saved = py_get(fields, nk[].k_numeric_mode, PythonObject(None))",
        "    py_setitem(fields, nk[].k_coef_, _wrap(py_item(entries, nk[].k_fit)(x, \"a,b\")))",
    ])
    assert plain(DOOR, text).split("\n") == [
        "    var nk = tsa_keys()",
        '    var mode = context["backend_fields"]["_DEFAULT_MODE"]',
        '    var init = context["hw_init_codes"][fields["initialization_method"]]',
        '    var saved = py_get(fields, "numeric_mode", PythonObject(None))',
        '    py_setitem(fields, "coef_", _wrap(entries["fit"](x, "a,b")))',
    ]


def test_operation_codes_come_back_as_names():
    text = "\n".join([
        "from bindings.tsa_keys_unpack import (tsa_keys, TSA_OPN_HW_FIT,",
        "    TSA_OPN_HW_PREDICT)",
        "def owner(model: PythonObject, operation: Int, context: PythonObject) raises:",
        "        var operation = py_op_code(nk[].ops, call[1])",
        "        if operation == TSA_OPN_HW_FIT or operation == X_LINEAR_OPN_CD_FIT:",
        "            return inner(model, TSA_OPN_HW_PREDICT, context)",
    ])
    assert plain(DOOR, text).split("\n") == [
        "from bindings.tsa_keys_unpack import (tsa_keys, TSA_OPN_HW_FIT,",
        "    TSA_OPN_HW_PREDICT)",
        "def owner(model: PythonObject, operation: String, context: PythonObject) raises:",
        "        var operation = String(py=call[1])",
        '        if operation == "hw_fit" or operation == "cd_fit":',
        '            return inner(model, "hw_predict", context)',
    ]


def test_operation_leaf_comes_back_as_the_name():
    # bw2-classical: the classical door's leaf of the caller's str (one table lookup).
    text = "\n".join([
        "def classical_operation_leaf(name: PythonObject) raises -> Int:",
        "        var operation = classical_operation_leaf(request[1])",
        "        if operation == CLASSICAL_OPN_RIDGE_FIT or operation == CLASSICAL_OPN_OLS_FIT:",
    ])
    assert plain(DOOR, text).split("\n") == [
        "def classical_operation_leaf(name: PythonObject) raises -> Int:",
        "        var operation = String(py=request[1])",
        '        if operation == "ridge_fit" or operation == "ols_fit":',
    ]


def test_line_count_and_untouched_text():
    text = 'def f(operation: Int):\n    return fields["alpha"]\n'
    assert plain(DOOR, text) == text                      # no held key, no code: as written
    assert plain("tools/x.py", "nk[].k_a") == "nk[].k_a"   # Mojo only
    keys = 'nk[].k_alpha = py_key("alpha")\npy_op_add(ops, "hw_fit", TSA_OPN_HW_FIT)'
    assert is_keys_file("bindings/tsa_keys_unpack.mojo")
    assert plain("bindings/tsa_keys_unpack.mojo", keys) == keys
    names = 'comptime GP_OPN_FAMILY_GP_GPR_FIT = 7\n    leaves.append(GP_OPN_FAMILY_GP_GPR_FIT)'
    assert is_keys_file("bindings/operation_names_gp.mojo")
    assert plain("bindings/operation_names_gp.mojo", names) == names
    assert plain(DOOR, "if operation == GP_OPN_FAMILY_GP_GPR_FIT:") == 'if operation == "family_gp_gpr_fit":'
    long = "\n".join(["    x = py_item(a, nk[].k_b)"] * 7)
    assert plain(DOOR, long).count("\n") == long.count("\n")
    assert plain(DOOR, None) is None
