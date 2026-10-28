# SPDX-License-Identifier: Apache-2.0
"""Import-time connection of optimizer native ABI entries and admitted types."""


def connect(training):
    # glue: connect fixed ABI symbols and native type/descriptor identities once
    import array
    from collections import OrderedDict
    import mmap
    import sys
    from types import BuiltinFunctionType, MethodType, ModuleType
    from . import _backend, _buffer, _numeric_profile
    from .tokenizer import _binding
    from ._array import Array
    from ._native_classical_bridge import _numpy_free_context
    from ._native_classical_warnings import connect_warnings
    owners = (training.SGD, training.Adam, training.AdamW)
    schedules = (training.ConstantLR, training.WarmupLinearLR, training.WarmupCosineLR)
    guarded = (Array, training._Optimizer, training._Schedule, training._LrTable,
               training.TrainDeviceArray) + owners + schedules
    names = tuple(Array.__slots__) + ("__getattribute__", "__getattr__", "__setattr__", "__dict__")
    guards = tuple((cls, cls.__mro__, vars(cls), tuple((name, vars(cls).get(name)) for name in names))  # glue: connect fixed native descriptor identities at import
                   for cls in guarded)  # glue: snapshot native record and array layouts
    bindings = {}
    metadata_bindings = {}
    entries = ("optimizer_door_step", "optimizer_door_io", "optimizer_door_state",
               "optimizer_door_state_set", "optimizer_door_load_state", "optimizer_door_zero",
               "optimizer_door_clip", "optimizer_door_accumulate", "training_numeric_mode",
               "train_dev_free", "train_dev_alloc", "train_dev_put", "train_dev_get",
               "linear_forward", "linear_backward", "linear_forward_dev", "linear_backward_dev",
               "rms_norm_forward", "rms_norm_backward", "embedding_forward", "embedding_backward", "ce_loss", "ce_loss_dev", "accumulation_is_aligned", "optimizer_resident_download", "optimizer_resident_upload",
               "chunked_lm_head_v2_loss", "chunked_lm_head_v2_train", "mlp_sessions_rows", "mlp_sessions",
               "residual_dropout", "residual_dropout_backward")
    optional_entries = ("linear_forward_dev", "linear_backward_dev", "ce_loss_dev", "chunked_lm_head_v2_loss",
                        "chunked_lm_head_v2_train", "mlp_sessions_rows", "mlp_sessions", "residual_dropout",
                        "residual_dropout_backward")
    for mode in ("identical", "fast"):  # glue: connect every supported numeric-mode ABI at import
        try:
            binding = _backend.binding("_mojolearn_training", mode)
            # glue: metadata is available independently of optional training
            # kernels; retain its actual compiled entries once at import.
            try:
                metadata_entries = tuple(getattr(binding, name) for name in ("training_numeric_mode", "training_vendor"))  # glue: fixed compiled metadata exports
                if any(type(entry) is not BuiltinFunctionType for entry in metadata_entries):  # glue: admit only native metadata entry points
                    raise TypeError("training metadata requires compiled native entries")
                metadata_bindings[mode] = metadata_entries
            except (ImportError, AttributeError, TypeError, ValueError, RuntimeError, NotImplementedError) as error:
                metadata_bindings[mode] = type(error).__name__ + ": " + str(error)
            cells = {}
            for name in entries:  # glue: retain compiled symbols, no runtime backend selection
                try:
                    entry = getattr(binding, name)
                except (ImportError, AttributeError, NotImplementedError):
                    if name.startswith("train_dev_") or name in optional_entries:
                        continue
                    raise
                if not isinstance(entry, BuiltinFunctionType):
                    raise TypeError(f"optimizer requires compiled {name}")
                cells[name] = entry
            cells["module"] = binding
            bindings[mode] = cells
        except (ImportError, AttributeError, TypeError, ValueError, RuntimeError, NotImplementedError) as error:
            bindings[mode] = type(error).__name__ + ": " + str(error)
            if mode not in metadata_bindings:
                metadata_bindings[mode] = bindings[mode]
    init = _buffer._native("optimizer_door_init")  # cpu-route: native parameter registry metadata only
    if not isinstance(init, BuiltinFunctionType):
        raise TypeError("optimizer requires compiled optimizer_door_init")
    _numeric_profile.default_profile("training")  # glue: initialize the package's numeric-profile connection
    _backend.default_mode()  # glue: initialize metadata's shared default before public native calls
    input_context = _numpy_free_context()
    input_context["native_module_type"] = ModuleType
    schedule_entries = {}
    for name in ("lr_schedule_at", "lr_schedule_series"):  # glue: retain native scalar schedule entries at import
        entry = training._lr_native(name)  # cpu-route: compiled exact learning-rate scalar ABI, never a training/data kernel
        if not isinstance(entry, BuiltinFunctionType):
            raise TypeError(f"schedule requires compiled {name}")
        schedule_entries[name] = entry
    context = {
        "schedule_entries": schedule_entries, "owners": owners, "schedules": schedules, "device_class": training.TrainDeviceArray,
        "array_class": Array, "array_type": array.array,
        "buffer_types": (bytes, bytearray, array.array, mmap.mmap), "mapping_types": (dict, OrderedDict),
        # Fixed native identities; mutable builtins names cannot admit hooks.
        "container_types": (list, tuple, dict, set, frozenset, memoryview),
        "scalar_types": (type(None), bool, int, float, complex, str, bytes),
        "module_type": ModuleType, "modules": sys.modules,
        "type_flags": vars(type)["__flags__"].__get__, "type_mro": vars(type)["__mro__"].__get__,
        "numpy_names": ("ndarray", "bool_", "int8", "uint8", "int16", "uint16", "int32", "uint32", "int64", "uint64", "longlong", "ulonglong", "float16", "float32", "float64", "longdouble"),
        "guards": guards, "bindings": bindings, "init": init,
        "training_metadata_bindings": metadata_bindings,
        "training_metadata_types": (str, int, BuiltinFunctionType, tuple),
        "training_metadata_strip": str.strip, "training_metadata_lower": str.lower,
        "backend": vars(_backend), "profile_defaults": _numeric_profile._defaults,
        "profiles": _numeric_profile.PROFILES, "input_context": input_context,
        "array_getters": {name: vars(Array)[name].__get__ for name in Array.__slots__},  # glue: retain fixed native Array slot descriptors at import
    }
    context["embedding_index_cast"] = _buffer._native("cast_elements")  # glue: retain the existing buffer conversion ABI at import
    if not isinstance(context["embedding_index_cast"], BuiltinFunctionType):
        raise TypeError("embedding requires compiled cast_elements")
    context["training_scalar_warnings"] = connect_warnings()
    context["training_scalar_locations"] = {
        name: (getattr(training, name).__code__.co_filename,
               max(line for _, _, line in getattr(training, name).__code__.co_lines() if line is not None),  # glue: retain fixed wrapper source metadata at import
               training.__name__)
        for name in ("rms_norm_forward", "rms_norm_backward", "embedding_backward", "cross_entropy")  # glue: retain public scalar-option warning locations at import
    }
    module = _binding()._module
    native = module.optimizer_native
    if not isinstance(native, BuiltinFunctionType):
        raise TypeError("optimizer requires compiled optimizer_native")
    finalizer = module.optimizer_device_finalize
    descriptor = module.classical_attribute_descriptor
    if not isinstance(finalizer, BuiltinFunctionType) or not isinstance(descriptor, BuiltinFunctionType):
        raise TypeError("training device ownership requires compiled finalizer descriptors")
    context["device_finalizer_types"] = (dict, int, BuiltinFunctionType, str, list)
    context["device_fields_getter"] = vars(training.TrainDeviceArray)["__dict__"].__get__
    training.TrainDeviceArray.__del__ = descriptor(MethodType(finalizer, context))
    optimizer_finalizer = module.optimizer_finalize_descriptor
    if not isinstance(optimizer_finalizer, BuiltinFunctionType):
        raise TypeError("optimizer ownership requires a compiled finalizer descriptor")
    training._Optimizer.__del__ = descriptor(MethodType(optimizer_finalizer, context))
    context["device_finalizer_descriptor"] = vars(training.TrainDeviceArray)["__del__"]
    names += ("__del__",)
    context["guards"] = tuple((cls, cls.__mro__, vars(cls), tuple((name, vars(cls).get(name)) for name in names))  # glue: retain finalizer and layout identities after native descriptor connection
                              for cls in guarded)  # glue: snapshot admitted owner layouts at import
    return native, context


def connect_sequence(training, sequence):
    # glue: register built-in sequence schedule identities once when that module loads
    context = training._OPT_CONTEXT
    classes = (sequence.StepLR, sequence.ExponentialLR, sequence.OneCycleLR)
    context["schedules"] = context["schedules"][:3] + classes
    guarded = (sequence._Sched, sequence._PowSched) + classes
    names = ("__dict__", "__getattribute__", "__getattr__", "__setattr__")
    guards = tuple((cls, cls.__mro__, vars(cls), tuple((name, vars(cls).get(name)) for name in names))  # glue: retain fixed native metadata layouts at module import
                   for cls in guarded)  # glue: register only fixed built-in schedule classes at import
    context["guards"] += guards


def connect_sequence_optimizers(training, optim, device):
    # glue: connect exact sequence optimizer owner layouts at import
    context = training._OPT_CONTEXT
    classes = (optim.RMSprop, optim.Adagrad, optim.Lion, optim.Adamax,
               optim.NAdam, optim.Adafactor, optim.LAMB)
    context["sequence_owners"] = classes
    guarded = (optim._ResidentState, optim._SeqOptimizer) + classes
    names = ("__dict__", "__getattribute__", "__getattr__", "__setattr__")
    context["guards"] += tuple((cls, cls.__mro__, vars(cls), tuple((name, vars(cls).get(name)) for name in names))  # glue: connect fixed native descriptor identities at import
                                for cls in guarded)  # glue: snapshot sequence native owner layouts
    return training._OPT_NATIVE, context


def connect_sequence_device(training, device):
    # glue: connect sequence buffer/optimizer ABI symbols once at module import
    from types import BuiltinFunctionType
    from _contextvars import ContextVar
    from . import _backend
    from ._native_classical_warnings import connect_warnings
    import re
    context = training._OPT_CONTEXT
    context["sequence_attribute_error"] = AttributeError  # glue: fixed native missing-member category
    context["sequence_print_context_type"] = ContextVar
    context["sequence_print_context_get"] = ContextVar.get
    warning = connect_warnings()
    code = device.SequenceDeviceTensor.__array__.__code__
    warning.update(warning_filename=code.co_filename,
                   warning_lineno=max(line for _, _, line in code.co_lines() if line is not None),  # glue: retain the public array-protocol warning location
                   warning_module=device.__name__, warning_source_line="")
    context["sequence_array_warnings"] = warning
    context["sequence_dtype_pattern"] = re.compile(r"[<>=|]?(?:[?bBhHiIuUlLqQpPnNefdgFDGOSTVcmM][0-9]*(?:\[[0-9a-zA-Z]+\])?|(?:datetime64|timedelta64)(?:\[[0-9a-zA-Z]+\])?)")
    context["sequence_dtype_fullmatch"] = re.Pattern.fullmatch
    context["sequence_dtype_match"] = re.Pattern.match
    context["sequence_dtype_compact"] = re.compile(r"([<>|=]?)( *[(]?[ ,0-9]*[)]? *)([<>|=]?)([A-Za-z0-9.?]*(?:\[[a-zA-Z0-9,.]+\])?)")
    context["sequence_dtype_separator"] = re.compile(r"\s*,\s*")
    context["sequence_dtype_space"] = re.compile(r"\s+$")
    context["sequence_device_class"] = device.SequenceDeviceTensor
    context["sequence_finalizer_types"] = (tuple, dict, int, BuiltinFunctionType)
    slots = tuple(device.SequenceDeviceTensor.__slots__)
    names = slots + ("__dict__", "__getattribute__", "__getattr__", "__setattr__")
    guarded = (device.SequenceDeviceTensor,)
    context["guards"] += tuple((cls, cls.__mro__, vars(cls), tuple((name, vars(cls).get(name)) for name in names))  # glue: connect fixed native descriptor identities at import
                                for cls in guarded)  # glue: snapshot sequence native owner layouts
    context["sequence_device_getters"] = {name: vars(device.SequenceDeviceTensor)[name].__get__ for name in slots}  # glue: retain native slot getters at import
    context["sequence_device_setters"] = {name: vars(device.SequenceDeviceTensor)[name].__set__ for name in slots if name != "__weakref__"}  # glue: retain native slot setters at import
    bindings = {}
    required = ("seq_optimizer_init", "seq_optimizer_step", "seq_lamb_step", "seq_adafactor_step",
                "seq_optimizer_state", "seq_optimizer_export", "seq_optimizer_import", "seq_optimizer_close")
    optional = ("seq_tensor_alloc", "seq_tensor_free", "seq_tensor_upload", "seq_tensor_download", "seq_tensor_copy",
                "optimizer_resident_open", "optimizer_resident_close", "optimizer_resident_move", "optimizer_resident_step", "optimizer_resident_step_dev",
                "adafactor_resident_open", "adafactor_resident_step", "adafactor_resident_step_dev",
                "lamb_resident_open", "lamb_resident_step", "lamb_resident_step_dev",
                "layer_norm", "layer_norm_dev", "moe_forward", "moe_forward_dev", "moe_weights_put", "moe_weights_free")
    for mode in ("identical", "fast"):  # glue: connect supported sequence numeric-mode ABI symbols at import
        try:
            binding = _backend.binding("_mojolearn_x_sequence", mode)
            cells = {"module": binding}
            for name in required + optional:  # glue: retain fixed compiled symbols at import
                entry = getattr(binding, name, None)
                if entry is None and name in optional:
                    continue
                if not isinstance(entry, BuiltinFunctionType):
                    raise TypeError(f"sequence optimizer requires compiled {name}")
                cells[name] = entry
            bindings[mode] = cells
        except (ImportError, AttributeError, TypeError, ValueError, RuntimeError, NotImplementedError) as error:
            bindings[mode] = type(error).__name__ + ": " + str(error)
    context["sequence_bindings"] = bindings
    # Embedding.to_device uses the embedding module's own device context.
    # Connect its fixed native buffer ABI here, at import, so native tensor
    # admission/readback/finalization recognize that owner without callbacks.
    extra = []
    for mode in ("identical", "fast"):  # glue: existing numeric-mode owners
        try:
            owner = _backend.binding("_mojolearn_embedding", mode)
            cells = {"module": owner}
            for name in ("seq_tensor_alloc", "seq_tensor_free", "seq_tensor_upload", "seq_tensor_download"):  # glue: connect four fixed native ABI entries at import
                entry = getattr(owner, name, None)
                if not isinstance(entry, BuiltinFunctionType):
                    raise TypeError("embedding tensor owner requires compiled " + name)
                cells[name] = entry
            extra.append(cells)
        except (ImportError, AttributeError, TypeError, ValueError, RuntimeError, NotImplementedError):
            pass  # Unavailable optional owners are refused by their public loader.
    context["sequence_extra_bindings"] = tuple(extra)
    context["sequence_device_io_off"] = device._DEVICE_IO_OFF
    return training._OPT_NATIVE, context


def connect_embedding(training, embedding_cls, archive_context):
    # glue: connect the Embedding layer's compiled entries and its native
    # lifetime descriptors once, at package import
    from types import BuiltinFunctionType, MethodType
    from . import _backend
    from .tokenizer import _binding
    try:
        from . import _x_sequence_device  # glue: connect the resident owner before public operations when its optional dependency exists
    except ModuleNotFoundError as error:
        if error.name != "numpy":
            raise
    context = training._OPT_CONTEXT
    bindings = {}
    names = ("embedding_forward", "embedding_backward", "embedding_numeric_mode", "embedding_table_release", "seq_tensor_alloc",
             "embedding_forward_dev", "embedding_backward_dev", "seq_tensor_free", "seq_tensor_upload",
             "seq_tensor_download", "seq_tensor_copy")
    for mode in ("identical", "fast"):  # glue: connect every supported numeric-mode ABI at import
        try:
            module = _backend.binding("_mojolearn_embedding", mode)
            cells = {"module": module}
            for name in names:  # glue: retain the compiled symbols the layer reaches
                entry = getattr(module, name, None)
                if isinstance(entry, BuiltinFunctionType):
                    cells[name] = entry
            bindings[mode] = cells
        except (ImportError, AttributeError, TypeError, ValueError, RuntimeError, NotImplementedError) as error:
            bindings[mode] = type(error).__name__ + ": " + str(error)
    context["embedding_bindings"] = bindings
    context["embedding_class"] = embedding_cls
    if "sequence_device_class" in context:
        native_owners = tuple(entry for entry in bindings.values()  # glue: preserve the full resident embedding ABI, including copy
                              if isinstance(entry, dict) and "seq_tensor_upload" in entry)
        context["sequence_extra_bindings"] = tuple(
            entry for entry in context.get("sequence_extra_bindings", ())  # glue: retain unrelated native device owners
            if not any(entry["module"] is owner["module"] for owner in native_owners)  # glue: compare import-connected native module identities
        ) + native_owners
    # The host inference class uses an import-connected native lookup owner;
    # a public operation never invokes the Python binding loader.
    context["embedding_host_module"] = "mojolearn._host._mojolearn_embedding_infer_host"
    # glue: connect the optional host lookup ABI once, including host_model.
    # Unavailable installs retain a refusal string instead of a runtime loader.
    try:
        host_module = _backend.load_host_module("_mojolearn_embedding_infer_host")  # cpu-route: import-time connection for explicitly selected CPU-only host_model lookup
        context["embedding_host_binding"] = {"module": host_module, **{
            name: getattr(host_module, name) for name in names  # glue: fixed host ABI connection at import
            if isinstance(getattr(host_module, name, None), BuiltinFunctionType)
        }}
    except (ImportError, AttributeError, TypeError, ValueError, RuntimeError, NotImplementedError) as error:
        context["embedding_host_binding"] = type(error).__name__ + ": " + str(error)
    context["embedding_host_names"] = names
    context["embedding_archive_context"] = archive_context.get("host_archive_context", archive_context)
    module = _binding()._module
    lifetime = module.embedding_layer_lifetime
    descriptor = module.classical_attribute_descriptor
    if not isinstance(lifetime, BuiltinFunctionType) or not isinstance(descriptor, BuiltinFunctionType):
        raise TypeError("Embedding requires its compiled lifetime entry")
    embedding_cls.__setattr__ = descriptor(MethodType(lifetime, context))
    embedding_cls.__del__ = descriptor(MethodType(lifetime, context))
    names = ("__dict__", "__getattribute__", "__getattr__", "__setattr__", "__del__")
    context["guards"] += tuple((cls, cls.__mro__, vars(cls), tuple((name, vars(cls).get(name)) for name in names))  # glue: retain native Embedding metadata identities
                                for cls in (embedding_cls,))  # glue: retain the single native Embedding class guard
    return training._OPT_NATIVE, context
