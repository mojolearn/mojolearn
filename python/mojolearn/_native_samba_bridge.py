# SPDX-License-Identifier: Apache-2.0
"""Import-time connection records for the native Samba execution owner."""
RUNTIME = None
CONTEXT = None


def connect(config_cls, model_cls, training, neural):
    # ABI connection at package import only; no operation invokes this function.
    from types import BuiltinFunctionType, SimpleNamespace
    from . import _backend
    entries = {}
    mode = _backend.default_mode()
    binding = _backend.binding("_mojolearn_training", mode)
    names = (
        "training_vendor", "training_numeric_mode", "training_experiment_profile",
        "embedding_forward", "embedding_backward", "rms_norm_forward", "rms_norm_backward",
        "linear_forward", "linear_backward", "ce_loss", "accumulate", "neural_rng",
        "optimizer_step", "optimizer_resident_download", "optimizer_resident_upload",
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
    return {
        "neural": neural.CONTEXT, "entries": SimpleNamespace(**entries),
        "mode": mode, "vendor": _backend.vendor(), "config_class": config_cls,
        "model_class": model_cls, "optimizer_class": training.AdamW,
        "generator_class": training.Generator, "exact_schedule": exact_schedule,
        "schedule_classes": {"constant": training.ConstantLR,
                             "linear": training.WarmupLinearLR,
                             "cosine": training.WarmupCosineLR},
    }
