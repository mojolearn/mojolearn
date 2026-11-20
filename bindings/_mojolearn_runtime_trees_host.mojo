# SPDX-License-Identifier: Apache-2.0
# Host owner of the tree, density, saved-model inference and host archive operations of `classical_native_call`.
# Split out of bindings/_mojolearn_runtime_classical_host.mojo (2026-11-07 host split, docs/build/HOT_GROUPS.md
# section 6) so it compiles as its own merged group: the same compiled functions, registered under the same
# names. bindings/_mojolearn_tokenizer_host.mojo keeps a native refusal under each name, and import setup
# replaces it with this module's entry (python/mojolearn/host_surface.py HOST_IMPORT_ALIASES). The Python types
# these entries allocate stay registered by their existing owner, which import setup loads first.
from core.experiment_guards_linear import LINEAR_GUARDS_OK
from std.os import abort
from std.python import PythonObject
from std.python.bindings import PythonModuleBuilder
from checks.kernel_matrix import COLUMN_CPU, TARGET_COLUMN, column_name
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
from std.sys.compile import is_defined
from bindings.classical_trees_door import classical_trees_c

comptime RUNTIME_TREES_SABOTAGE = is_defined["MOJOLEARN_HOST_SABOTAGE"]()


def runtime_trees_host_numeric_mode_binding() raises -> PythonObject:
    comptime assert GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL, (
        "runtime trees host requires MOJOLEARN_NUMERIC_IDENTICAL=1"
    )
    return PythonObject(GLOBAL_NUMERIC_MODE)


def runtime_trees_host_vendor_binding() raises -> PythonObject:
    return PythonObject("cpu")


def runtime_trees_host_column_binding() raises -> PythonObject:
    comptime assert TARGET_COLUMN == COLUMN_CPU, (
        "runtime trees host requires MOJOLEARN_COLUMN_CPU"
    )
    return PythonObject(column_name(TARGET_COLUMN))


def runtime_trees_host_sabotage_binding() raises -> PythonObject:
    return PythonObject(RUNTIME_TREES_SABOTAGE)


def build__mojolearn_runtime_trees_host() -> PythonObject:
    comptime assert LINEAR_GUARDS_OK, "invalid integrated experiment configuration (linear guards)"
    try:
        var module = PythonModuleBuilder("_mojolearn_runtime_trees_host")
        module.def_py_c_function(classical_trees_c, "trees_native_call")
        module.def_function[runtime_trees_host_numeric_mode_binding]("runtime_trees_host_numeric_mode")
        module.def_function[runtime_trees_host_vendor_binding]("runtime_trees_host_vendor")
        module.def_function[runtime_trees_host_column_binding]("runtime_trees_host_column")
        module.def_function[runtime_trees_host_sabotage_binding]("runtime_trees_host_sabotage")
        return module.finalize()
    except error:
        abort(String("failed to create _mojolearn_runtime_trees_host: ", error))
