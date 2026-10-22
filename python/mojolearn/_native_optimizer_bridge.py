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
    owners = (training.SGD, training.Adam, training.AdamW)
    schedules = (training.ConstantLR, training.WarmupLinearLR, training.WarmupCosineLR)
    guarded = (Array, training._Optimizer, training._Schedule, training._LrTable,
               training.TrainDeviceArray) + owners + schedules
    names = tuple(Array.__slots__) + ("__getattribute__", "__getattr__", "__setattr__", "__dict__")
    guards = tuple((cls, cls.__mro__, vars(cls), tuple((name, vars(cls).get(name)) for name in names))  # glue: connect fixed native descriptor identities at import
                   for cls in guarded)  # glue: snapshot native record and array layouts
    bindings = {}
    entries = ("optimizer_door_step", "optimizer_door_io", "optimizer_door_state",
               "optimizer_door_state_set", "optimizer_door_load_state", "optimizer_door_zero",
               "optimizer_door_clip", "optimizer_door_accumulate", "training_numeric_mode",
               "train_dev_free", "train_dev_alloc", "train_dev_put", "train_dev_get", "accumulation_is_aligned", "optimizer_resident_download", "optimizer_resident_upload")
    for mode in ("identical", "fast"):  # glue: connect every supported numeric-mode ABI at import
        try:
            binding = _backend.binding("_mojolearn_training", mode)
            cells = {}
            for name in entries:  # glue: retain compiled symbols, no runtime backend selection
                try:
                    entry = getattr(binding, name)
                except (ImportError, AttributeError, NotImplementedError):
                    if name.startswith("train_dev_"):
                        continue
                    raise
                if not isinstance(entry, BuiltinFunctionType):
                    raise TypeError(f"optimizer requires compiled {name}")
                cells[name] = entry
            cells["module"] = binding
            bindings[mode] = cells
        except (ImportError, AttributeError, TypeError, ValueError, RuntimeError, NotImplementedError) as error:
            bindings[mode] = type(error).__name__ + ": " + str(error)
    init = _buffer._native("optimizer_door_init")  # cpu-route: native parameter registry metadata only
    if not isinstance(init, BuiltinFunctionType):
        raise TypeError("optimizer requires compiled optimizer_door_init")
    _numeric_profile.default_profile("training")  # glue: initialize the package's numeric-profile connection
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
        "scalar_types": (type(None), bool, int, float, complex, str, bytes),
        "module_type": ModuleType, "modules": sys.modules,
        "type_flags": vars(type)["__flags__"].__get__, "type_mro": vars(type)["__mro__"].__get__,
        "numpy_names": ("ndarray", "bool_", "int8", "uint8", "int16", "uint16", "int32", "uint32", "int64", "uint64", "float16", "float32", "float64", "longdouble"),
        "guards": guards, "bindings": bindings, "init": init,
        "backend": vars(_backend), "profile_defaults": _numeric_profile._defaults,
        "profiles": _numeric_profile.PROFILES, "input_context": input_context,
        "array_getters": {name: vars(Array)[name].__get__ for name in Array.__slots__},  # glue: retain fixed native Array slot descriptors at import
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
                "lamb_resident_open", "lamb_resident_step", "lamb_resident_step_dev")
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
    context["sequence_device_io_off"] = device._DEVICE_IO_OFF
    return training._OPT_NATIVE, context
