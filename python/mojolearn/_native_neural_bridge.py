# SPDX-License-Identifier: Apache-2.0
"""Import-time connection slots for the compiled standalone neural owner."""
RUNTIME = None
CONTEXT = None


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
            "transformer_session_create", "transformer_session_close",
            "transformer_session_backward", "transformer_session_invalidate_tape",
            "transformer_arithmetic_profile", "transformer_arithmetic_profile_changed",
            "transformer_forward_tape_enabled", "transformer_forward_tape",
            "transformer_backward_tape", "transformer_close_tape",
            "transformer_session_forward_tape", "transformer_session_backward_tape")),
        "mamba": ("_mojolearn_mamba", (
            "mamba_vendor", "mamba_numeric_mode", "mamba1_backward", "mamba2_backward",
            "mamba3_backward", "mamba3_forward", "mamba3_decode_step", "mamba3_prefill_session_create",
            "mamba3_prefill_session_backward", "mamba3_prefill_session_backward_owned",
            "mamba3_prefill_session_forward", "mamba3_prefill_session_forward_owned",
            "mamba3_prefill_session_reports_optional", "mamba3_prefill_session_close",
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
