# SPDX-License-Identifier: Apache-2.0
# Host owner of the multi-device drivers, the neighbor query doors, forecasting and the small MLP. Split out of
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
from bindings.parallel_shard_native import parallel_shard_c
from bindings.neighbor_query_native import neighbor_query_c, neighbor_query_attribute_c
from bindings.parallel_forecast_native import parallel_forecast_c
from bindings.tsa_public_native import tsa_c
from bindings.parallel_neural_native import parallel_neural_c
from bindings.parallel_mlp_native import parallel_mlp_c
from bindings.small_mlp_native import small_mlp_c, small_mlp_finalize_c
from bindings.parallel_causal_native import parallel_causal_c

comptime RUNTIME_PARALLEL_SABOTAGE = is_defined["MOJOLEARN_HOST_SABOTAGE"]()


def runtime_parallel_host_numeric_mode_binding() raises -> PythonObject:
    comptime assert GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL, (
        "runtime parallel host requires MOJOLEARN_NUMERIC_IDENTICAL=1"
    )
    return PythonObject(GLOBAL_NUMERIC_MODE)


def runtime_parallel_host_vendor_binding() raises -> PythonObject:
    return PythonObject("cpu")


def runtime_parallel_host_column_binding() raises -> PythonObject:
    comptime assert TARGET_COLUMN == COLUMN_CPU, (
        "runtime parallel host requires MOJOLEARN_COLUMN_CPU"
    )
    return PythonObject(column_name(TARGET_COLUMN))


def runtime_parallel_host_sabotage_binding() raises -> PythonObject:
    return PythonObject(RUNTIME_PARALLEL_SABOTAGE)


def build__mojolearn_runtime_parallel_host() -> PythonObject:
    comptime assert NEURAL_GUARDS_OK, "invalid integrated experiment configuration (neural guards)"
    try:
        var module = PythonModuleBuilder("_mojolearn_runtime_parallel_host")
        module.def_py_c_function(parallel_shard_c, "parallel_shard_call")
        module.def_py_c_function(neighbor_query_c, "neighbor_query_call")
        module.def_py_c_function(neighbor_query_attribute_c, "neighbor_query_attribute_call")
        module.def_py_c_function(parallel_forecast_c, "parallel_forecast_call")
        module.def_py_c_function(tsa_c, "tsa_native_call")
        module.def_py_c_function(parallel_neural_c, "parallel_neural_call")
        module.def_py_c_function(parallel_mlp_c, "parallel_mlp_call")
        module.def_py_c_function(small_mlp_c, "small_mlp_call")
        module.def_py_c_function(small_mlp_finalize_c, "small_mlp_finalize")
        module.def_py_c_function(parallel_causal_c, "parallel_causal_call")
        module.def_function[runtime_parallel_host_numeric_mode_binding]("runtime_parallel_host_numeric_mode")
        module.def_function[runtime_parallel_host_vendor_binding]("runtime_parallel_host_vendor")
        module.def_function[runtime_parallel_host_column_binding]("runtime_parallel_host_column")
        module.def_function[runtime_parallel_host_sabotage_binding]("runtime_parallel_host_sabotage")
        return module.finalize()
    except error:
        abort(String("failed to create _mojolearn_runtime_parallel_host: ", error))
