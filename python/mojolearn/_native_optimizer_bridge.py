# SPDX-License-Identifier: Apache-2.0
"""Import-time connection of optimizer native ABI entries and admitted types."""


def connect(training):
    # glue: connect fixed ABI symbols and native type/descriptor identities once
    import array
    import mmap
    import sys
    from types import BuiltinFunctionType, ModuleType
    from . import _backend, _buffer, _numeric_profile
    from .tokenizer import _binding
    from ._array import Array
    from ._native_classical_bridge import _numpy_free_context
    owners = (training.SGD, training.Adam, training.AdamW)
    schedules = (training.ConstantLR, training.WarmupLinearLR, training.WarmupCosineLR)
    guarded = (Array, training._Optimizer, training._Schedule, training._LrTable,
               training.TrainDeviceArray) + owners + schedules
    names = tuple(Array.__slots__) + ("__getattribute__", "__getattr__", "__setattr__", "__dict__")
    guards = tuple((cls, cls.__mro__, vars(cls), tuple((name, vars(cls).get(name)) for name in names))
                   for cls in guarded)  # glue: snapshot native record and array layouts
    bindings = {}
    entries = ("optimizer_door_step", "optimizer_door_io", "optimizer_door_state",
               "optimizer_door_state_set", "optimizer_door_load_state", "optimizer_door_zero",
               "optimizer_door_clip", "optimizer_door_accumulate", "training_numeric_mode",
               "train_dev_free", "accumulation_is_aligned", "optimizer_resident_download", "optimizer_resident_upload")
    for mode in ("identical", "fast"):  # glue: connect every supported numeric-mode ABI at import
        try:
            binding = _backend.binding("_mojolearn_training", mode)
            cells = {}
            for name in entries:  # glue: retain compiled symbols, no runtime backend selection
                try:
                    entry = getattr(binding, name)
                except (ImportError, AttributeError, NotImplementedError):
                    if name == "train_dev_free":
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
    context = {
        "owners": owners, "schedules": schedules, "device_class": training.TrainDeviceArray,
        "array_class": Array, "array_type": array.array,
        "buffer_types": (bytes, bytearray, array.array, mmap.mmap),
        "scalar_types": (type(None), bool, int, float, complex, str, bytes),
        "module_type": ModuleType, "modules": sys.modules,
        "type_flags": vars(type)["__flags__"].__get__, "type_mro": vars(type)["__mro__"].__get__,
        "numpy_names": ("ndarray", "bool_", "int8", "uint8", "int16", "uint16", "int32", "uint32", "int64", "uint64", "float16", "float32", "float64", "longdouble"),
        "guards": guards, "bindings": bindings, "init": init,
        "backend": vars(_backend), "profile_defaults": _numeric_profile._defaults,
        "profiles": _numeric_profile.PROFILES, "input_context": input_context,
        "array_getters": {name: vars(Array)[name].__get__ for name in Array.__slots__},
    }
    native = _binding()._module.optimizer_native
    if not isinstance(native, BuiltinFunctionType):
        raise TypeError("optimizer requires compiled optimizer_native")
    return native, context
