# SPDX-License-Identifier: Apache-2.0
"""Import-time connection records for the native Samba execution owner."""
RUNTIME = None
CONTEXT = None


def connect(config_cls, model_cls, training, neural):
    # ABI connection at package import only; no operation invokes this function.
    from types import BuiltinFunctionType, SimpleNamespace
    from . import _backend, _buffer
    entries = {}
    mode = _backend.default_mode()
    binding = _backend.binding("_mojolearn_training", mode)
    names = (
        "training_vendor", "training_numeric_mode", "training_experiment_profile",
        "embedding_forward", "embedding_backward", "rms_norm_forward", "rms_norm_backward",
        "linear_forward", "linear_backward", "ce_loss", "accumulate", "neural_rng",
        "optimizer_step", "optimizer_resident_download", "optimizer_resident_upload",
        "optimizer_resident_open", "optimizer_resident_close", "optimizer_resident_step",
        "optimizer_resident_step_io", "optimizer_maximize_dev", "accumulation_is_aligned",
        "samba_resident_enabled", "samba_resident_open", "samba_resident_forward", "samba_resident_step",
        "optimizer_parallel_available", "accumulate_parallel_available", "clip_parallel_available",
        "samba_head_loss", "training_chunked_lm_head_enabled", "chunked_lm_head_v2_train",
        "samba_afn_norm_head_forward", "samba_afn_tail_train", "samba_afn_embedding_backward_tied",
    )
    for name in names:  # glue: connect existing compiled symbols at package import
        try:
            entry = getattr(binding, name)
        except (AttributeError, ImportError, NotImplementedError):
            continue
        if not isinstance(entry, BuiltinFunctionType):
            raise TypeError(f"Samba native execution requires a compiled {name}")
        entries[name] = entry
    exact_schedule = training._lr_native("lr_schedule_exact")  # cpu-route: connects the existing exact host-scalar learning-rate entry at import; native owner evaluates one scalar per step
    if not isinstance(exact_schedule, BuiltinFunctionType):
        raise TypeError("Samba schedule requires a compiled exact schedule entry")
    core = {}
    for name in ("equal_elements", "reduce_stat", "all_finite_f32"):  # glue: connect existing native input admission entries at import
        entry = _buffer._native(name)
        if not isinstance(entry, BuiltinFunctionType):
            raise TypeError(f"Samba input admission requires a compiled {name}")
        core[name] = entry
    # Reuse connected modules, without loading a second type-owning tier.
    # These records let Generator preserve its per-instance numeric mode.
    import _operator
    generator_bindings = {mode: {name: getattr(binding, name) for name in  # glue: import-time ABI symbols
                               ("training_vendor", "training_numeric_mode", "neural_rng")}}
    if _backend.vendor() not in (None, "cpu"):
        for route, connected in neural.CONTEXT["bindings"].items():  # glue: ABI connection records only
            if not route.startswith("gpu:"):
                continue
            selected = connected["primitives"]
            key = route.split(":", 1)[1]
            if isinstance(selected, str):
                generator_bindings[key] = selected
                continue
            symbols = {name: getattr(selected, name) for name in  # glue: import-time ABI symbols
                       ("training_vendor", "training_numeric_mode", "neural_rng")}
            if not all(isinstance(value, BuiltinFunctionType) for value in symbols.values()):  # glue: validate connection entries at import
                raise TypeError("Generator execution requires compiled native RNG entries")
            generator_bindings[key] = symbols
    return {
        **core,
        "generator_bindings": generator_bindings, "generator_backend": _backend,
        "generator_index": _operator.index,
        "neural": neural.CONTEXT, "entries": SimpleNamespace(**entries),
        "mode": mode, "vendor": _backend.vendor(), "config_class": config_cls,
        "model_class": model_cls, "optimizer_class": training.AdamW,
        "generator_class": training.Generator, "exact_schedule": exact_schedule,
        "schedule_classes": {"constant": training.ConstantLR,
                             "linear": training.WarmupLinearLR,
                             "cosine": training.WarmupCosineLR},
    }
