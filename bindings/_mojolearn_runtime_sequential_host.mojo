# SPDX-License-Identifier: Apache-2.0
# Logical host owner for sequential. Release builds merge this builder
# with the other host builders; development builds give it an independent
# source/cache boundary. Native admission snapshots are registered here.
from std.os import abort
from std.python import PythonObject
from std.python.bindings import PythonModuleBuilder
from checks.kernel_matrix import COLUMN_CPU, TARGET_COLUMN, column_name
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
from std.sys.compile import is_defined
from bindings.sequential_native import sequential_c
from bindings.native_identity_key import NativeIdentityKey
from bindings.sequential_admission import SequentialAdmissions
from bindings.sequential_description import SequentialDescription

comptime RUNTIME_SEQUENTIAL_SABOTAGE = is_defined["MOJOLEARN_HOST_SABOTAGE"]()


def runtime_sequential_host_numeric_mode_binding() raises -> PythonObject:
    comptime assert GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL, (
        "sequential host requires MOJOLEARN_NUMERIC_IDENTICAL=1"
    )
    return PythonObject(GLOBAL_NUMERIC_MODE)


def runtime_sequential_host_vendor_binding() raises -> PythonObject:
    return PythonObject("cpu")


def runtime_sequential_host_column_binding() raises -> PythonObject:
    comptime assert TARGET_COLUMN == COLUMN_CPU, (
        "sequential host requires MOJOLEARN_COLUMN_CPU"
    )
    return PythonObject(column_name(TARGET_COLUMN))


def runtime_sequential_host_sabotage_binding() raises -> PythonObject:
    return PythonObject(RUNTIME_SEQUENTIAL_SABOTAGE)


def build__mojolearn_runtime_sequential_host() -> PythonObject:
    try:
        var module = PythonModuleBuilder("_mojolearn_runtime_sequential_host")
        _ = module.add_type[NativeIdentityKey]("_NativeIdentityKey")
        _ = module.add_type[SequentialAdmissions]("_SequentialAdmissions")
        _ = module.add_type[SequentialDescription]("_SequentialDescription")
        module.def_py_c_function(sequential_c, "sequential_native_call")
        module.def_function[runtime_sequential_host_numeric_mode_binding]("runtime_sequential_host_numeric_mode")
        module.def_function[runtime_sequential_host_vendor_binding]("runtime_sequential_host_vendor")
        module.def_function[runtime_sequential_host_column_binding]("runtime_sequential_host_column")
        module.def_function[runtime_sequential_host_sabotage_binding]("runtime_sequential_host_sabotage")
        return module.finalize()
    except error:
        abort(String("failed to create _mojolearn_runtime_sequential_host: ", error))
