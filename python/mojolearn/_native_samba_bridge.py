# SPDX-License-Identifier: Apache-2.0
"""Import-time connection records for the native Samba execution owner."""
RUNTIME = None
CONTEXT = None

def _connect_admission(config_cls, model_cls, state_cls, training, neural, session_cls=None):
    # Package connection only: retain C descriptors and exact native classes.
    # NumPy remains optional and may be imported by a caller after mojolearn.
    import array
    import mmap
    import sys
    import _thread
    from ._native_operation_lock import _OperationLock
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
        "_resident_wanted", "seed", "counter",
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
    if session_cls is not None:
        # SambaDecodeSession: admitted as a record; none of its own fields is
        # in `fields`, so a step walks no model or state behind it.
        classes.append(session_cls)
    classes.extend(nc["generation_classes"][2].values())
    classes.extend(nc["block_classes"].values())
    classes = tuple(dict.fromkeys(classes))
    buffer_or_owner_fields = {
        "config", "optimizer", "generator", "arrays", "flat", "params", "_m", "_v", "buf_initialized",
        "_blocks", "_w", "_wopt", "_int15", "layers", "offsets", "packed_", "_runtime_lock",
        "k_cache", "v_cache", "theta", "h", "buffer_qrot", "buffer_krot", "buffer_v", "buffer_dt",
        "buffer_sig", "buffer_adt", "pending_k", "pending_v", "bits", "codes", "exponents",
        "conv_window", "affine_boundary", "affine_a", "affine_b", "buffer_xbc", "buffer_dtraw",
    }
    records = []
    for cls in classes:  # glue: retain native __dict__ getters for admitted records
        descriptor = next(vars(base)["__dict__"] for base in cls.__mro__ if "__dict__" in vars(base))  # glue: retain the fixed native dictionary ABI at package connection
        records.append((cls, descriptor.__get__, fields))
    array_cls = nc["array_class"]
    array_names = tuple(array_cls.__slots__)
    guards = []
    for cls in classes + (array_cls,):  # glue: retained class metadata, never runtime callbacks
        names = array_names if cls is array_cls else fields + ("__dict__",)
        namespace = vars(cls)
        keys = names + ("__getattribute__", "__getattr__", "__setattr__")
        guards.append((namespace, tuple((name, namespace.get(name)) for name in keys), cls, cls.__mro__))  # glue: snapshot fixed class descriptors at package connection
    return {
        "records": tuple(records), "guards": tuple(guards), "array_class": array_cls,
        "metadata_fields": tuple(name for name in fields if name not in buffer_or_owner_fields),  # glue: connect fixed ABI field categories once at import
        "config_class": config_cls, "optimizer_class": training.AdamW,
        "array_getters": {name: vars(array_cls)[name].__get__ for name in array_names},  # glue: connect native Array slot getters once at package import
        "array_metadata": ("shape", "dtype", "order", "_addr", "_readonly", "ndim", "size", "nbytes", "itemsize", "strides"),
        "mapping_types": (dict, OrderedDict, defaultdict),
        "defaultdict_type": defaultdict,
        "native_factories": (dict, list, tuple, set, bool, int, float, complex, str, bytes, bytearray),
        "scalar_types": (type(None), bool, int, float, complex, str, bytes),
        "buffer_types": (bytes, bytearray, array.array, mmap.mmap),
        "opaque_types": (_thread.LockType, type(_thread.RLock()), _OperationLock),
        "path_types": (PosixPath, PurePosixPath), "module_type": ModuleType, "modules": sys.modules,
        "numpy_names": ("ndarray", "bool_", "int8", "uint8", "int16", "uint16", "int32", "uint32", "int64", "uint64", "float16", "float32", "float64", "longdouble"),
        "type_flags": vars(type)["__flags__"].__get__,
        "type_mro": vars(type)["__mro__"].__get__,
        "schedule_types": (training.ConstantLR, training.WarmupLinearLR, training.WarmupCosineLR),
    }

def connect(config_cls, model_cls, training, neural, state_cls, session_cls=None):
    # ABI connection at package import only; no operation invokes this function.
    from types import BuiltinFunctionType, SimpleNamespace
    from . import _backend, _buffer
    from ._mamba_impl import Mamba3State
    entries = {}
    mode = _backend.default_mode()
    try:
        binding = _backend.binding("_mojolearn_training", mode)
        unavailable = None
    except (ImportError, ValueError) as exc:
        # glue: an install without this tier (a CPU-only install under
        # MOJOLEARN_NUMERIC_MODE=fast refuses the fast tier by name) still
        # imports; every Samba entry stays unconnected and the native owner
        # refuses at use, as parallel_training does for a missing byte LM.
        binding, unavailable = None, str(exc)
    names = (
        "training_vendor", "training_numeric_mode", "training_experiment_profile", "neural_primitives_api_v1",
        "embedding_forward", "embedding_backward", "rms_norm_forward", "rms_norm_backward",
        "linear_forward", "linear_backward", "ce_loss", "accumulate", "neural_rng",
        "optimizer_native_api_v1", "accumulation_is_aligned",
        "optimizer_resident_step_io", "optimizer_maximize_dev",
        "samba_resident_enabled",
        "optimizer_parallel_available", "accumulate_parallel_available", "clip_parallel_available",
        "samba_head_loss", "training_chunked_lm_head_enabled", "chunked_lm_head_v2_train",
        "samba_afn_tail_train", "samba_afn_embedding_backward_tied",
    )
    for name in names:  # glue: connect existing compiled symbols at package import
        try:
            entry = getattr(binding, name)
        except (AttributeError, ImportError, NotImplementedError):
            continue
        if not isinstance(entry, BuiltinFunctionType):
            raise TypeError(f"Samba native execution requires a compiled {name}")
        entries[name] = entry
    core = {}
    for name in ("equal_elements", "reduce_stat", "all_finite_f32"):  # glue: connect existing native input admission entries at import
        entry = _buffer._native(name)
        if not isinstance(entry, BuiltinFunctionType):
            raise TypeError(f"Samba input admission requires a compiled {name}")
        core[name] = entry
    # Which numeric modes have a connected training image, for Generator
    # (it keeps a per-instance numeric mode): True, or the reason it is not
    # connected. The draw itself goes through the image's entry table
    # (bindings/generator_draw_native_door.mojo), so no symbol is held here.
    import _operator
    generator_bindings = {mode: True if binding is not None else unavailable}
    if _backend.vendor() not in (None, "cpu"):
        for route, connected in neural.CONTEXT["bindings"].items():  # glue: ABI connection records only
            if not route.startswith("gpu:"):
                continue
            selected = connected["primitives"]
            generator_bindings[route.split(":", 1)[1]] = selected if isinstance(selected, str) else True
    result = {
        **core,
        "admission": _connect_admission(config_cls, model_cls, state_cls, training, neural, session_cls),
        "generator_bindings": generator_bindings, "generator_backend": _backend,
        "generator_index": _operator.index,
        "neural": neural.CONTEXT, "entries": SimpleNamespace(canonical=binding, **entries),
        "mode": mode, "vendor": _backend.vendor(), "config_class": config_cls,
        "model_class": model_cls, "state_class": state_cls, "mamba3_state_class": Mamba3State,
        "decode_session_class": session_cls,
        "optimizer_class": training.AdamW,
        "optimizer_step_io": "optimizer_resident_step_io" in entries,
        "optimizer_maximize_device": "optimizer_maximize_dev" in entries,
        "generator_class": training.Generator,
        # A restored schedule is built by the optimizer owner's native
        # `schedule_setstate` (dictionary fill, admission, one block bind).
        "schedule_native": training._OPT_NATIVE, "schedule_context": training._OPT_CONTEXT,
        "schedule_classes": {"constant": training.ConstantLR,
                             "linear": training.WarmupLinearLR,
                             "cosine": training.WarmupCosineLR},
        # `device=` of Generator and the schedules (lane vdf-sched;
        # bindings/generator_device_unpack.mojo): the shared UNPACK reads the
        # installation record here.
        "backend_fields": vars(_backend),
    }
    # A Generator constructed with device="cpu" beside a GPU set draws on the host
    # training image's Samba table entry (vendor 0), which the image publishes when it
    # loads: loaded at the first device="cpu" admission of the process, never at import.
    if _backend._CPU_ONLY is None:
        from ._native_family_bridge import cpu_load_later

        def _connect_generator_cpu():
            # glue: lazy load of the CPU identical training image, once per process
            try:
                _backend.load_host_module("_mojolearn_training_host")
                result["generator_host"] = True
            except (ImportError, AttributeError, ValueError, RuntimeError, TypeError, OSError) as error:
                result["generator_host"] = type(error).__name__ + ": " + str(error)
        cpu_load_later(result, _connect_generator_cpu)
    return result
