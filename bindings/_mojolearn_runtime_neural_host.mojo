# SPDX-License-Identifier: Apache-2.0
# Host owner of the public neural doors: transformer, Mamba and Samba blocks, neural blocks, causal language
# models and decoding. Split out of bindings/_mojolearn_tokenizer_host.mojo (2026-11-07 host split,
# docs/build/HOT_GROUPS.md section 6) so it compiles as its own merged group: the same compiled functions,
# registered under the same names. bindings/_mojolearn_tokenizer_host.mojo keeps a native refusal under each
# name, and import setup replaces it with this module's entry (python/mojolearn/host_surface.py
# HOST_IMPORT_ALIASES). The Python types these entries allocate stay registered by their existing owner, which
# import setup loads first.
from core.experiment_guards_neural import NEURAL_GUARDS_OK
from std.os import abort
from std.python import PythonObject
from std.python.bindings import PythonModuleBuilder
from checks.kernel_matrix import COLUMN_CPU, TARGET_COLUMN, column_name
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
from std.sys.compile import is_defined
from bindings.native_transport import transport_initialize
from bindings.transformer_public_native import transformer_public_c
from providers.python.transformer_decode_pyapi import transformer_decode_native_c
from bindings.mamba_public_native import mamba_native_c
from bindings.samba_config_native import samba_config_c
from bindings.samba_public_native import samba_public_c
from bindings.neural_block_public import neural_block_c
from bindings.causal_lm_construct import causal_model_c
from bindings.causal_lm_public import causal_public_c

comptime RUNTIME_NEURAL_SABOTAGE = is_defined["MOJOLEARN_HOST_SABOTAGE"]()


def runtime_neural_host_numeric_mode_binding() raises -> PythonObject:
    comptime assert GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL, (
        "runtime neural host requires MOJOLEARN_NUMERIC_IDENTICAL=1"
    )
    return PythonObject(GLOBAL_NUMERIC_MODE)


def runtime_neural_host_vendor_binding() raises -> PythonObject:
    return PythonObject("cpu")


def runtime_neural_host_column_binding() raises -> PythonObject:
    comptime assert TARGET_COLUMN == COLUMN_CPU, (
        "runtime neural host requires MOJOLEARN_COLUMN_CPU"
    )
    return PythonObject(column_name(TARGET_COLUMN))


def runtime_neural_host_sabotage_binding() raises -> PythonObject:
    return PythonObject(RUNTIME_NEURAL_SABOTAGE)


def build__mojolearn_runtime_neural_host() -> PythonObject:
    comptime assert NEURAL_GUARDS_OK, "invalid integrated experiment configuration (neural guards)"
    try:
        var module = PythonModuleBuilder("_mojolearn_runtime_neural_host")
        # The transport boundary these entries reach, set up as the tokenizer host does
        # (the same values; the tokenizer host initializes it first in one process).
        transport_initialize()
        module.def_py_c_function(transformer_public_c, "transformer_public_call")
        module.def_py_c_function(transformer_decode_native_c, "transformer_decode_native_call")
        module.def_py_c_function(mamba_native_c, "mamba_native_call")
        module.def_py_c_function(samba_config_c, "samba_config_call")
        module.def_py_c_function(samba_public_c, "samba_native_call")
        module.def_py_c_function(neural_block_c[0], "neural_block_init")
        module.def_py_c_function(neural_block_c[4], "neural_block_set_params")
        module.def_py_c_function(neural_block_c[5], "neural_block_getstate")
        module.def_py_c_function(neural_block_c[6], "neural_block_setstate")
        module.def_py_c_function(neural_block_c[1], "neural_block_allocate")
        module.def_py_c_function(neural_block_c[2], "neural_block_forward")
        module.def_py_c_function(neural_block_c[3], "neural_block_backward")
        module.def_py_c_function(neural_block_c[7], "neural_block_inference_backward")
        module.def_py_c_function(causal_model_c[0], "causal_model_init")
        module.def_py_c_function(causal_model_c[1], "causal_model_load")
        module.def_py_c_function(causal_model_c[2], "causal_model_read_weights")
        module.def_py_c_function(causal_public_c[1], "causal_lm_allocate")
        module.def_py_c_function(causal_public_c[2], "causal_lm_reset")
        module.def_py_c_function(causal_public_c[3], "causal_lm_forward")
        module.def_py_c_function(causal_public_c[4], "causal_lm_stream")
        module.def_py_c_function(causal_public_c[5], "causal_lm_read")
        module.def_py_c_function(causal_public_c[6], "causal_lm_info")
        module.def_py_c_function(causal_public_c[7], "causal_lm_generate")
        module.def_py_c_function(causal_public_c[8], "causal_lm_text_stream")
        module.def_py_c_function(causal_public_c[9], "causal_lm_generate_text")
        module.def_py_c_function(causal_public_c[10], "causal_lm_resident")
        module.def_py_c_function(causal_public_c[11], "causal_lm_snapshot")
        module.def_py_c_function(causal_public_c[12], "causal_lm_resume")
        module.def_py_c_function(causal_public_c[13], "causal_lm_parameters")
        module.def_py_c_function(causal_public_c[14], "causal_lm_blocks")
        module.def_py_c_function(causal_public_c[15], "causal_state_init")
        module.def_py_c_function(causal_public_c[16], "causal_lm_repr")
        module.def_py_c_function(causal_public_c[17], "causal_state_repr")
        module.def_function[runtime_neural_host_numeric_mode_binding]("runtime_neural_host_numeric_mode")
        module.def_function[runtime_neural_host_vendor_binding]("runtime_neural_host_vendor")
        module.def_function[runtime_neural_host_column_binding]("runtime_neural_host_column")
        module.def_function[runtime_neural_host_sabotage_binding]("runtime_neural_host_sabotage")
        return module.finalize()
    except error:
        abort(String("failed to create _mojolearn_runtime_neural_host: ", error))
