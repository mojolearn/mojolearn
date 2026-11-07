# SPDX-License-Identifier: Apache-2.0
# Host owner of the model catalog and the language-model corpus doors. Split out of
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
from bindings.model_catalog_native import catalog_public_c
from bindings.lm_corpus_public_native import lm_corpus_public_c

comptime RUNTIME_CATALOG_SABOTAGE = is_defined["MOJOLEARN_HOST_SABOTAGE"]()


def runtime_catalog_host_numeric_mode_binding() raises -> PythonObject:
    comptime assert GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL, (
        "runtime catalog host requires MOJOLEARN_NUMERIC_IDENTICAL=1"
    )
    return PythonObject(GLOBAL_NUMERIC_MODE)


def runtime_catalog_host_vendor_binding() raises -> PythonObject:
    return PythonObject("cpu")


def runtime_catalog_host_column_binding() raises -> PythonObject:
    comptime assert TARGET_COLUMN == COLUMN_CPU, (
        "runtime catalog host requires MOJOLEARN_COLUMN_CPU"
    )
    return PythonObject(column_name(TARGET_COLUMN))


def runtime_catalog_host_sabotage_binding() raises -> PythonObject:
    return PythonObject(RUNTIME_CATALOG_SABOTAGE)


def build__mojolearn_runtime_catalog_host() -> PythonObject:
    comptime assert NEURAL_GUARDS_OK, "invalid integrated experiment configuration (neural guards)"
    try:
        var module = PythonModuleBuilder("_mojolearn_runtime_catalog_host")
        module.def_py_c_function(catalog_public_c, "catalog_public_call")
        module.def_py_c_function(lm_corpus_public_c, "lm_corpus_public_call")
        module.def_function[runtime_catalog_host_numeric_mode_binding]("runtime_catalog_host_numeric_mode")
        module.def_function[runtime_catalog_host_vendor_binding]("runtime_catalog_host_vendor")
        module.def_function[runtime_catalog_host_column_binding]("runtime_catalog_host_column")
        module.def_function[runtime_catalog_host_sabotage_binding]("runtime_catalog_host_sabotage")
        return module.finalize()
    except error:
        abort(String("failed to create _mojolearn_runtime_catalog_host: ", error))
