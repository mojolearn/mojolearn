# SPDX-License-Identifier: Apache-2.0
"""Import-time connection records for the native Samba execution owner."""
RUNTIME = None
CONTEXT = None

def _connect_admission(config_cls, model_cls, state_cls, training, neural):
    # Package connection only: retain C descriptors and exact native classes.
    # NumPy remains optional and may be imported by a caller after mojolearn.
    import array
    import mmap
    import sys
    import _thread
    from collections import OrderedDict, defaultdict
    from pathlib import PosixPath, PurePosixPath
    from types import ModuleType
    from .neural_inference import SambaInference
    from ._mamba_impl import Mamba3State
    nc = neural.CONTEXT
    fields = (
        "config", "numeric_mode", "arrays", "flat", "names", "shapes", "offsets", "n_total",
        "optimizer", "generator", "max_norm", "_blocks", "_w", "weight_format", "_native_cpu_inference",
        "vocab", "d_model", "layers", "n_heads", "n_kv_heads", "head_dim", "intermediate",
        "tie_embeddings", "norm_eps", "dropout", "params", "_m", "_v", "buf_initialized", "t",
        "lr", "betas", "eps", "weight_decay", "accumulation_steps", "lr_schedule", "maximize",
        "_host_fresh", "_pristine", "_p_dev", "_g_dev", "_resident_wanted", "seed", "counter",
        "peak_lr", "min_lr", "warmup_steps", "total_steps", "_wopt", "_int15", "nheads", "d_inner",
        "d_in_proj", "window", "_opts_tail", "_extended", "numeric_profile", "_native_kind", "_native_route",
        "_owned_generation", "_owned_restore_pending", "_discard_reports", "dt_limit", "_dt_limit",
        "_arithmetic_profile", "_arithmetic_profile_mode", "_arithmetic_profile_changed",
        "batch_size", "max_tokens", "k_cache", "v_cache", "cached_tokens", "theta", "h",
        "buffer_qrot", "buffer_krot", "buffer_v", "buffer_dt", "buffer_sig", "buffer_adt",
        "pending_k", "pending_v", "buffered_tokens", "pending", "bits", "codes", "exponents", "shape",
        "conv_window", "profile_id", "absolute_position", "affine_boundary", "affine_a", "affine_b",
        "_nn34_usable", "buffer_xbc", "buffer_dtraw", "last_", "packed_", "lr_", "clip_coef_", "total_norm_",
        "_runtime_lock", "_W_NAMES", "_opts_flags", "_gated", "max_positions", "rope_dim", "conv_dim", "dt_rank",
        "kind", "device", "block_options", "block_kwargs",
    )
    classes = [config_cls, model_cls, state_cls, SambaInference, Mamba3State,
               training.AdamW, training.Generator, training.ConstantLR,
               training.WarmupLinearLR, training.WarmupCosineLR,
               nc["bf16_class"], nc["int8_class"], nc["model_class"], nc["config_context"]["plan_class"]]
    classes.extend(nc["generation_classes"][2].values())
    for group in nc["block_classes"].values():  # glue: fixed ABI owner classes at package connection
        classes.extend(group.values())
    classes = tuple(dict.fromkeys(classes))
    records = []
    for cls in classes:  # glue: retain native __dict__ getters for admitted records
        descriptor = next(vars(base)["__dict__"] for base in cls.__mro__ if "__dict__" in vars(base))
        records.append((cls, descriptor.__get__, fields))
    array_cls = nc["array_class"]
    array_names = tuple(array_cls.__slots__)
    guards = []
    for cls in classes + (array_cls,):  # glue: retained class metadata, never runtime callbacks
        names = array_names if cls is array_cls else fields + ("__dict__",)
        namespace = vars(cls)
        keys = names + ("__getattribute__", "__getattr__", "__setattr__")
        guards.append((namespace, tuple((name, namespace.get(name)) for name in keys), cls, cls.__mro__))
    return {
        "records": tuple(records), "guards": tuple(guards), "array_class": array_cls,
        "array_getters": {name: vars(array_cls)[name].__get__ for name in array_names},
        "array_metadata": ("shape", "dtype", "order", "_addr", "_readonly", "ndim", "size", "nbytes", "itemsize", "strides"),
        "mapping_types": (dict, OrderedDict, defaultdict),
        "defaultdict_type": defaultdict,
        "native_factories": (dict, list, tuple, set, bool, int, float, complex, str, bytes, bytearray),
        "scalar_types": (type(None), bool, int, float, complex, str, bytes),
        "buffer_types": (bytes, bytearray, array.array, mmap.mmap),
        "opaque_types": (_thread.LockType, type(_thread.RLock())),
        "path_types": (PosixPath, PurePosixPath), "module_type": ModuleType, "modules": sys.modules,
        "numpy_names": ("ndarray", "bool_", "int8", "uint8", "int16", "uint16", "int32", "uint32", "int64", "uint64", "float16", "float32", "float64", "longdouble"),
        "type_flags": vars(type)["__flags__"].__get__,
        "type_mro": vars(type)["__mro__"].__get__,
        "schedule_types": (training.ConstantLR, training.WarmupLinearLR, training.WarmupCosineLR),
    }

def connect(config_cls, model_cls, training, neural, state_cls):
    # ABI connection at package import only; no operation invokes this function.
    from types import BuiltinFunctionType, SimpleNamespace
    from . import _backend, _buffer
    from ._mamba_impl import Mamba3State
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
    from . import _portable_math
    try:
        # cpu-route: connect the existing portable binary64 scalar sqrt ABI;
        # native initializer preparation calls this C function directly.
        generator_sqrt = _portable_math._load().mojolearn_sqrt
        generator_sqrt_type = type(generator_sqrt)
    except OSError as exc:
        # Preserve import without the optional scalar library. Only an
        # initializer requiring sqrt refuses, before consuming a stream.
        generator_sqrt, generator_sqrt_type = str(exc), None
    return {
        **core,
        "admission": _connect_admission(config_cls, model_cls, state_cls, training, neural),
        "generator_bindings": generator_bindings, "generator_backend": _backend,
        "generator_index": _operator.index,
        "generator_sqrt": generator_sqrt, "generator_sqrt_type": generator_sqrt_type,
        "neural": neural.CONTEXT, "entries": SimpleNamespace(**entries),
        "mode": mode, "vendor": _backend.vendor(), "config_class": config_cls,
        "model_class": model_cls, "state_class": state_cls, "mamba3_state_class": Mamba3State,
        "optimizer_class": training.AdamW,
        "generator_class": training.Generator, "exact_schedule": exact_schedule,
        "schedule_classes": {"constant": training.ConstantLR,
                             "linear": training.WarmupLinearLR,
                             "cosine": training.WarmupCosineLR},
    }
