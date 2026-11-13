# SPDX-License-Identifier: Apache-2.0
"""Connection slots for the compiled standalone neural owner.

`RUNTIME` and `CONTEXT` are written by `models/causal_lm.py` when it imports
(`_connect_causal_native`). Nothing defines them here: the first read of
either one imports `models.causal_lm`, which connects them, so every neural
family (`_transformer_impl`, `_mamba_impl`, `_samba_impl`, `neural_inference`)
reaches the same connection whatever is imported first.
"""


def __getattr__(name):
    # glue: lazy family import (lane lazy-import-fix, 2026-11-12). The block
    # modules used to import `models.causal_lm` at their own module end to run
    # this connection; when `neural_inference` imported first, that closed the
    # cycle neural_inference -> _transformer_impl -> models.causal_lm ->
    # neural_inference (partially initialized). The connection now runs at the
    # first read of a slot, after every class it pins exists.
    if name not in ("RUNTIME", "CONTEXT"):
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib
    importlib.import_module(__name__.rpartition(".")[0] + ".models.causal_lm")
    slots = globals()
    if name not in slots:
        # Read while models.causal_lm itself is still connecting.
        raise AttributeError(f"mojolearn: the native neural connection ({__name__}.{name}) is read "
                             "before models.causal_lm finished connecting it")
    return slots[name]


def connect_training(backend, vendor, inference_bindings):
    # Import-time ABI connection only. Native owners receive plain containers
    # of checked C functions, never a lazy Python binding proxy.
    from types import BuiltinFunctionType, SimpleNamespace
    from . import _buffer
    selected_mode = backend.default_mode()
    route = "cpu" if vendor in (None, "cpu") else "gpu"
    families = {
        "transformer": ("_mojolearn_transformer", (
            "transformer_vendor", "transformer_numeric_mode", "transformer_backward",
            "transformer_block_api_v1", "transformer_session_backward",
            "transformer_arithmetic_profile", "transformer_arithmetic_profile_changed")),
        "mamba": ("_mojolearn_mamba", (
            "mamba_vendor", "mamba_numeric_mode", "mamba_state_api_v1", "mamba_block_api_v1",
            "mamba3_tape_create_native_v1", "mamba3_tape_handle_native_v1",
            "mamba1_backward", "mamba2_backward", "mamba3_backward",
            # The Mamba-3 prefill session is a native registry handle reached
            # through the neural table (lane mamba3-prefill-handle); only the
            # B-arm entries of MOJOLEARN_NEURAL_TABLE_ROUTE_OFF remain.
            "mamba3_prefill_session_backward", "mamba3_prefill_session_backward_owned",
            "mamba3_prefill_session_install_weights", "mamba3_owned_weights_enabled",
            "mamba3_forward_fresh", "mamba3_forward_tape_enabled", "mamba3_forward_tape",
            "mamba3_backward_tape", "mamba3_close_tape")),
    }
    result = {}
    validator = _buffer._native("all_finite_f32")
    if not isinstance(validator, BuiltinFunctionType):
        raise TypeError("neural training input validation must be a compiled native entry")
    modes = [selected_mode]
    if route == "gpu":
        modes = []
        for key in inference_bindings:  # glue: enumerate modules already connected at import
            if key.startswith("gpu:"):
                modes.append(key.split(":", 1)[1])
    for mode in modes:  # glue: reuse the already connected native mode modules
        connected = {}
        for family, (target, names) in families.items():  # glue: connect native ABI symbols at import
            entries = {}
            try:
                selected = (backend.binding(target, mode) if route == "cpu" else
                            inference_bindings["gpu:" + mode]["mamba1" if family == "mamba" else family])
                if isinstance(selected, str):
                    connected[family] = selected
                    continue
                for name in names:  # glue: connect native ABI symbols at import
                    try:
                        entry = getattr(selected, name)
                    except (ImportError, AttributeError, NotImplementedError):
                        continue
                    if not isinstance(entry, BuiltinFunctionType):
                        raise TypeError(f"{target}.{name} must be a compiled native entry")
                    entries[name] = entry
                connected[family] = SimpleNamespace(canonical=selected, **entries)
            except (ImportError, RuntimeError, ValueError, OSError, AttributeError, NotImplementedError) as exc:
                connected[family] = f"native training binding {target} for {mode} is unavailable: {exc}"
        connected["all_finite_f32"] = validator
        result[route + ":" + mode] = connected
    return result
