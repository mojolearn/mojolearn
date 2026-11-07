# SPDX-License-Identifier: Apache-2.0
# Host owner of the preprocessing and label doors (scalers, stateless and classifying transforms, encoders,
# string labels). Split out of bindings/_mojolearn_runtime_classical_host.mojo (2026-11-07 host split,
# docs/build/HOT_GROUPS.md section 6) so it compiles as its own merged group: the same compiled functions,
# registered under the same names. bindings/_mojolearn_tokenizer_host.mojo keeps a native refusal under each
# name, and import setup replaces it with this module's entry (python/mojolearn/host_surface.py
# HOST_IMPORT_ALIASES). The Python types these entries allocate stay registered by their existing owner, which
# import setup loads first.
from core.experiment_guards_linear import LINEAR_GUARDS_OK
from std.os import abort
from std.python import PythonObject
from std.python.bindings import PythonModuleBuilder
from checks.kernel_matrix import COLUMN_CPU, TARGET_COLUMN, column_name
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
from std.sys.compile import is_defined
from bindings.scaler_native import scaler_fit_c
from bindings.scaler_lifecycle_native import scaler_lifecycle_c, scaler_attribute_c
from bindings.minmax_native import minmax_lifecycle_c
from bindings.maxabs_native import maxabs_lifecycle_c
from bindings.prep_stateless_native import prep_stateless_c
from bindings.prep_classify_native import prep_classify_c
from bindings.prep_owner_native import prep_owner_c
from bindings.str_labels_native import str_labels_c
from bindings.labels_native_entry import labels_native_c
from bindings.robust_native import robust_lifecycle_c

comptime RUNTIME_PREP_SABOTAGE = is_defined["MOJOLEARN_HOST_SABOTAGE"]()


def runtime_prep_host_numeric_mode_binding() raises -> PythonObject:
    comptime assert GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL, (
        "runtime prep host requires MOJOLEARN_NUMERIC_IDENTICAL=1"
    )
    return PythonObject(GLOBAL_NUMERIC_MODE)


def runtime_prep_host_vendor_binding() raises -> PythonObject:
    return PythonObject("cpu")


def runtime_prep_host_column_binding() raises -> PythonObject:
    comptime assert TARGET_COLUMN == COLUMN_CPU, (
        "runtime prep host requires MOJOLEARN_COLUMN_CPU"
    )
    return PythonObject(column_name(TARGET_COLUMN))


def runtime_prep_host_sabotage_binding() raises -> PythonObject:
    return PythonObject(RUNTIME_PREP_SABOTAGE)


def build__mojolearn_runtime_prep_host() -> PythonObject:
    comptime assert LINEAR_GUARDS_OK, "invalid integrated experiment configuration (linear guards)"
    try:
        var module = PythonModuleBuilder("_mojolearn_runtime_prep_host")
        module.def_py_c_function(scaler_fit_c, "standard_scaler_fit_native")
        module.def_py_c_function(scaler_lifecycle_c, "standard_scaler_native_call")
        module.def_py_c_function(minmax_lifecycle_c, "minmax_scaler_native_call")
        module.def_py_c_function(maxabs_lifecycle_c, "maxabs_scaler_native_call")
        module.def_py_c_function(prep_stateless_c, "prep_stateless_native_call")
        module.def_py_c_function(prep_classify_c, "prep_classify_native_call")
        module.def_py_c_function(prep_owner_c, "prep_encode_native_call")
        module.def_py_c_function(str_labels_c, "str_labels_native_call")
        module.def_py_c_function(labels_native_c, "labels_native_call")
        module.def_py_c_function(robust_lifecycle_c, "robust_scaler_native_call")
        module.def_py_c_function(scaler_attribute_c, "standard_scaler_attribute_call")
        module.def_function[runtime_prep_host_numeric_mode_binding]("runtime_prep_host_numeric_mode")
        module.def_function[runtime_prep_host_vendor_binding]("runtime_prep_host_vendor")
        module.def_function[runtime_prep_host_column_binding]("runtime_prep_host_column")
        module.def_function[runtime_prep_host_sabotage_binding]("runtime_prep_host_sabotage")
        return module.finalize()
    except error:
        abort(String("failed to create _mojolearn_runtime_prep_host: ", error))
