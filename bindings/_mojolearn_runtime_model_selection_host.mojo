# SPDX-License-Identifier: Apache-2.0
# Logical host owner for model selection. Release builds merge this builder
# with the other host builders; development builds give it an independent
# source/cache boundary. No native exposed type is registered here.
from std.os import abort
from std.python import PythonObject
from std.python.bindings import PythonModuleBuilder
from checks.kernel_matrix import COLUMN_CPU, TARGET_COLUMN, column_name
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
from bindings.runtime_model_selection_entry import (
    runtime_model_selection_c, RUNTIME_MODEL_SELECTION_SABOTAGE,
)


def runtime_model_selection_host_numeric_mode_binding() raises -> PythonObject:
    comptime assert GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL, (
        "model selection host requires MOJOLEARN_NUMERIC_IDENTICAL=1"
    )
    return PythonObject(GLOBAL_NUMERIC_MODE)


def runtime_model_selection_host_vendor_binding() raises -> PythonObject:
    return PythonObject("cpu")


def runtime_model_selection_host_column_binding() raises -> PythonObject:
    comptime assert TARGET_COLUMN == COLUMN_CPU, (
        "model selection host requires MOJOLEARN_COLUMN_CPU"
    )
    return PythonObject(column_name(TARGET_COLUMN))


def runtime_model_selection_host_sabotage_binding() raises -> PythonObject:
    return PythonObject(RUNTIME_MODEL_SELECTION_SABOTAGE)


def build__mojolearn_runtime_model_selection_host() -> PythonObject:
    try:
        var module = PythonModuleBuilder("_mojolearn_runtime_model_selection_host")
        module.def_py_c_function(runtime_model_selection_c, "model_selection_native_call")
        module.def_function[runtime_model_selection_host_numeric_mode_binding]("runtime_model_selection_host_numeric_mode")
        module.def_function[runtime_model_selection_host_vendor_binding]("runtime_model_selection_host_vendor")
        module.def_function[runtime_model_selection_host_column_binding]("runtime_model_selection_host_column")
        module.def_function[runtime_model_selection_host_sabotage_binding]("runtime_model_selection_host_sabotage")
        return module.finalize()
    except error:
        abort(String("failed to create _mojolearn_runtime_model_selection_host: ", error))
