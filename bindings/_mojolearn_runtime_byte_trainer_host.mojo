# SPDX-License-Identifier: Apache-2.0
# Host owner of the byte language model trainer doors (host, device and multi-device). Split out of
# bindings/_mojolearn_tokenizer_host.mojo (2026-11-07 host split, docs/build/HOT_GROUPS.md section 6) so it
# compiles as its own merged group: the same compiled functions, registered under the same names.
# bindings/_mojolearn_tokenizer_host.mojo keeps a native refusal under each name, and import setup replaces it
# with this module's entry (python/mojolearn/host_surface.py HOST_IMPORT_ALIASES). The Python types these
# entries allocate stay registered by their existing owner, which import setup loads first.
from core.experiment_guards_neural import NEURAL_GUARDS_OK
from std.os import abort
from std.python import PythonObject
from std.python.bindings import PythonModuleBuilder
from checks.kernel_matrix import COLUMN_CPU, TARGET_COLUMN, column_name
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
from std.sys.compile import is_defined
from bindings.byte_lm_host_trainer_native import byte_lm_host_trainer_c
from bindings.byte_lm_trainer_native import byte_lm_trainer_c, byte_lm_pooled_finalize_c
from bindings.parallel_byte_native import parallel_byte_c, ParallelByteOwner
from std.python.bindings import lookup_py_type_object
from std.reflection import reflect
from bindings.py_type_registry import py_type_absent, py_type_attach

comptime RUNTIME_BYTE_TRAINER_SABOTAGE = is_defined["MOJOLEARN_HOST_SABOTAGE"]()


def runtime_byte_trainer_host_numeric_mode_binding() raises -> PythonObject:
    comptime assert GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL, (
        "runtime byte trainer host requires MOJOLEARN_NUMERIC_IDENTICAL=1"
    )
    return PythonObject(GLOBAL_NUMERIC_MODE)


def runtime_byte_trainer_host_vendor_binding() raises -> PythonObject:
    return PythonObject("cpu")


def runtime_byte_trainer_host_column_binding() raises -> PythonObject:
    comptime assert TARGET_COLUMN == COLUMN_CPU, (
        "runtime byte trainer host requires MOJOLEARN_COLUMN_CPU"
    )
    return PythonObject(column_name(TARGET_COLUMN))


def runtime_byte_trainer_host_sabotage_binding() raises -> PythonObject:
    return PythonObject(RUNTIME_BYTE_TRAINER_SABOTAGE)


# Registered by this owner, the one that allocates it (moved here from
# bindings/_mojolearn_tokenizer_host.mojo with its entry). Lookup-guarded like
# the classical host's registrars (bindings/py_type_registry.mojo): a second
# initialization in one process attaches the existing type object.
def register_parallelbyteowner(mut module: PythonModuleBuilder) raises:
    var existing = PythonObject(None)
    try:
        existing = lookup_py_type_object[ParallelByteOwner]()
    except error:
        if not py_type_absent(error, String(reflect[ParallelByteOwner].name())):
            raise error
        _ = module.add_type[ParallelByteOwner]("_ParallelByteOwner")
        return
    py_type_attach(module.module, "_ParallelByteOwner", existing)


def build__mojolearn_runtime_byte_trainer_host() -> PythonObject:
    comptime assert NEURAL_GUARDS_OK, "invalid integrated experiment configuration (neural guards)"
    try:
        var module = PythonModuleBuilder("_mojolearn_runtime_byte_trainer_host")
        module.def_py_c_function(byte_lm_host_trainer_c, "byte_lm_host_trainer_call")
        module.def_py_c_function(byte_lm_trainer_c, "byte_lm_trainer_call")
        module.def_py_c_function(byte_lm_pooled_finalize_c, "byte_lm_pooled_finalize_call")
        module.def_py_c_function(parallel_byte_c, "parallel_byte_call")
        register_parallelbyteowner(module)
        module.def_function[runtime_byte_trainer_host_numeric_mode_binding]("runtime_byte_trainer_host_numeric_mode")
        module.def_function[runtime_byte_trainer_host_vendor_binding]("runtime_byte_trainer_host_vendor")
        module.def_function[runtime_byte_trainer_host_column_binding]("runtime_byte_trainer_host_column")
        module.def_function[runtime_byte_trainer_host_sabotage_binding]("runtime_byte_trainer_host_sabotage")
        return module.finalize()
    except error:
        abort(String("failed to create _mojolearn_runtime_byte_trainer_host: ", error))
