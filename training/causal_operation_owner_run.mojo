# SPDX-License-Identifier: Apache-2.0
"""RUN half of the causal LM operation owner (lane rs-neural, wave 8 run
split, from bindings/causal_operation_owner.mojo): the names of the model
and block fields a call reads live instead of from its capture. Typed Mojo
only, no Python import (tools/python_free_dirs.sh). Same bodies."""


def _causal_model_runtime_names() -> List[String]:
    # Model fields a call itself updates (lifecycle words, resident sessions):
    # read and normalized from the live model on every call, as before.
    return [String("_parameters_exposed"), String("_resident_busy"), String("_resident_sessions"),
            String("_decode_program"), String("_closed"), String("_failed")]


def _causal_block_runtime_names() -> List[String]:
    # Block fields an operation writes back (sessions, provider caches,
    # arithmetic profile words, reports): taken live on every call.
    return [String("_arithmetic_profile"), String("_arithmetic_profile_mode"), String("_arithmetic_profile_changed"),
            String("_native_session"), String("h_last_"),
            String("_native_block_provider_cache"), String("_native_mamba_block_provider_cache"), String("_native_mamba_state"),
            String("_native_primitives_provider_cache"),
            String("_native_mamba_provider_cache")]
