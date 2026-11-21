# SPDX-License-Identifier: Apache-2.0
# UNPACK of PyTorch's `Module.to(...)` arguments, shared by every Mojolearn
# class that offers `to()` (the neural blocks, Sequential). Lane api-torch,
# 2026-11-21 (round 2: moved here from bindings/neural_block_public_door.mojo
# so the blocks and Sequential admit the same words through one function).
#
# `to(device=None, dtype=None)` as torch spells it: a str positional is the
# device, a dtype positional is the dtype; `non_blocking` and `copy` are
# accepted and ignored (a Mojolearn `to()` is synchronous and never shares
# device storage with the old route). The neural models are float32 only, so
# a dtype is admitted or refused in words, never converted. The device word
# itself is admitted by the one device rule (core/device_route.mojo) where the
# caller applies it. Cold API call: no loop or repeated call reaches it.
from std.python import PythonObject
from bindings.py_balanced import py_call, py_type
from bindings.native_loaded_module_door import NativeModules


def torch_float32_word(mut mods: NativeModules, value: PythonObject, what: String) raises -> Bool:
    # True for `torch.float32`, NumPy's float32 dtype or the word "float32";
    # False for any other dtype. Exact-type admission: a str, or a dtype
    # class named `dtype` (torch.dtype; NumPy's `dtype[float32]` or, from
    # NumPy 2, `Float32DType`: C types whose str is their name); nothing
    # else is read.
    var bi = mods.module("builtins")
    var kind = py_type(value)
    if kind is bi.str:
        var word = String(value)
        return word == "float32" or word == "torch.float32"
    var kind_name = String(py_call(bi.type.__dict__["__name__"].__get__, kind))
    if kind_name != "dtype" and not kind_name.startswith("dtype[") and not kind_name.endswith("DType"):
        raise Error("TypeError: " + what + ": dtype must be torch.float32, numpy.float32's dtype or 'float32'")
    var word = String(py_call(bi.str, value))
    return word == "float32" or word == "torch.float32"


def torch_to_device(mut mods: NativeModules, args: PythonObject, kwargs: PythonObject, what: String) raises -> PythonObject:
    """The device `to(*args, **kwargs)` names, or None when it names none.
    A dtype other than float32 is refused here."""
    var bi = mods.module("builtins")
    if py_type(args) is not bi.tuple or py_type(kwargs) is not bi.dict:
        raise Error("TypeError: " + what + ": to() requires ordinary positional and keyword arguments")
    if len(args) > 2:
        raise Error("TypeError: " + what + ": to() takes a device and a dtype")
    var device = PythonObject(None)
    var dtype = PythonObject(None)
    var has_device = False
    for i in range(len(args)):  # driver-loop: over at most two positional arguments
        var item = args[i]
        if item is None:
            continue
        if py_type(item) is bi.str and String(item) != "float32" and String(item) != "torch.float32":
            if has_device:
                raise Error("TypeError: " + what + ": to() got two devices")
            device = item
            has_device = True
        else:
            dtype = item
    var keys = py_call(bi.list, kwargs)
    for i in range(len(keys)):  # driver-loop: over the keyword names given
        var key = keys[i]
        if py_type(key) is not bi.str:
            raise Error("TypeError: " + what + ": to() keywords must be names")
        var name = String(key)
        if name == "device":
            if has_device:
                raise Error("TypeError: " + what + ": to() got two devices")
            device = kwargs[key]
            has_device = device is not None
        elif name == "dtype":
            dtype = kwargs[key]
        elif name == "non_blocking" or name == "copy":
            continue
        else:
            raise Error("TypeError: " + what + ": to() got an unexpected keyword argument '" + name + "'")
    if dtype is not None and not torch_float32_word(mods, dtype, what):
        raise Error("ValueError: " + what + ": runs float32 only (identical mode defines its bits in float32), got dtype "
                    + String(py_call(bi.repr, dtype)))
    return device
