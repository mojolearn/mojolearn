# SPDX-License-Identifier: Apache-2.0
# Torch's ordinary Tensor type can still dispatch to Python mode hooks.
# Read only Torch's native thread-local stack/dispatch-key metadata before
# touching a Tensor. Never disable modes or call Python torch.overrides /
# torch.utils._python_dispatch helpers. The C entries are the ones used by
# those helpers in the supported installed Torch, and are checked before cache.
from std.ffi import _Global
from bindings.native_loaded_module import native_loaded_module
from std.python import Python, PythonObject
from bindings.py_balanced import py_call, py_type, py_get, py_fields, py_setitem


struct NativeTorchBoundary(Defaultable, Movable):
    var module: PythonObject
    var base: PythonObject
    var tensor: PythonObject
    var parameter: PythonObject
    var entries: PythonObject
    var methods: PythonObject
    var mode_keys: PythonObject
    var pre_dispatch: PythonObject

    def __init__(out self):
        self.module = PythonObject(None)
        self.base = PythonObject(None)
        self.tensor = PythonObject(None)
        self.parameter = PythonObject(None)
        self.entries = PythonObject(None)
        self.methods = PythonObject(None)
        self.mode_keys = PythonObject(None)
        self.pre_dispatch = PythonObject(None)


comptime NATIVE_TORCH = _Global[StorageType=NativeTorchBoundary,
    name="MojoNativeTorchAdmission", init_fn=NativeTorchBoundary.__init__]


def _torch_type_fields(cls: PythonObject) raises -> PythonObject:
    var bi = native_loaded_module("builtins")
    # Use type's native dictionary descriptor, bypassing a custom metaclass's
    # __getattribute__ and any user property named __dict__.
    return py_call(bi.type.__dict__["__dict__"].__get__, cls)


def native_torch_exact(cls: PythonObject) raises -> Bool:
    var bi = native_loaded_module("builtins")
    var sys = native_loaded_module("sys")
    var module = py_get(sys.modules, "torch._tensor")
    if module is None or py_type(module) is not py_type(sys):
        return False
    var fields = py_fields(module)
    if cls is py_get(fields, "Tensor"):
        return True
    var parameter = py_get(sys.modules, "torch.nn.parameter")
    if parameter is not None and py_type(parameter) is py_type(sys):
        return cls is py_get(py_fields(parameter), "Parameter")
    return False


def _torch_connect() raises:
    var cache = NATIVE_TORCH.get_or_create_ptr()
    var bi = native_loaded_module("builtins")
    var sys = native_loaded_module("sys")
    var module = py_get(sys.modules, "torch._C")
    if module is None or py_type(module) is not py_type(sys):
        raise Error("TypeError: Torch native mode admission requires its compiled extension")
    if cache[].module is module:
        return
    var fields = py_fields(module)
    var entries = Python.dict()
    # small-loop(the three fixed Torch mode metadata entry names)
    for name in [String("_len_torch_function_stack"), String("_len_torch_dispatch_stack"),
                 String("_dispatch_tls_is_dispatch_key_included")]:
        var entry = py_get(fields, name)
        if entry is None or py_type(entry) is not py_type(bi.len):
            raise Error("TypeError: Torch lacks compiled mode metadata entry " + name)
        if String(entry.__module__) != "torch._C" or String(entry.__name__) != name:
            raise Error("TypeError: Torch mode metadata entry provenance differs: " + name)
        py_setitem(entries, name, entry)
    var base = py_get(fields, "TensorBase")
    if base is None:
        # Torch 2.1 registers the same native tensor descriptor family only
        # under _TensorBase. Later releases export both names. Admission below
        # still verifies the native methods and the actual Tensor ancestry.
        base = py_get(fields, "_TensorBase")
    if base is None:
        raise Error("TypeError: Torch lacks native TensorBase metadata")
    var base_fields = _torch_type_fields(base)
    var methods = Python.dict()
    for name in [String("detach"), String("numpy"), String("data_ptr"), String("stride")]:  # small-loop(the four fixed tensor method names)
        var entry = py_get(base_fields, name)
        if entry is None or py_type(entry) is not py_type(bi.list.append):
            raise Error("TypeError: Torch tensor conversion requires native method " + name)
        py_setitem(methods, name, entry)
    for name in [String("device"), String("dtype"), String("shape")]:  # small-loop(the three fixed tensor descriptor names)
        var entry = py_get(base_fields, name)
        if entry is None or py_type(entry) is not py_type(bi.memoryview.obj):
            raise Error("TypeError: Torch tensor metadata requires native descriptor " + name)
        py_setitem(methods, name, entry)
    var mode_type = py_get(fields, "_TorchDispatchModeKey")
    var dispatch_type = py_get(fields, "DispatchKey")
    if dispatch_type is None:
        raise Error("TypeError: Torch lacks native dispatch key metadata")
    var keys = Python.list()
    var get_mode = py_get(fields, "_get_dispatch_mode")
    if mode_type is None and get_mode is None:
        # Verified Torch 2.1 native TLS stores every dispatch mode in stack_.
        # Its C stack length therefore covers the modes that modern Torch
        # stores in separate infrastructure slots. Never infer this layout
        # merely from missing APIs in a newer/unknown release.
        # Source: pytorch/v2.1.0/c10/core/impl/TorchDispatchModeTLS.h; actual
        # 2.1.2 C-entry probe retained in the internal compatibility evidence.
        var version_module = py_get(sys.modules, "torch.version")
        if version_module is None or py_type(version_module) is not py_type(sys):
            raise Error("TypeError: Torch unified dispatch metadata requires verified version metadata")
        var version = py_get(py_fields(version_module), "__version__")
        if py_type(version) is not bi.str or not Bool(py=py_call(version.startswith, "2.1.")):
            raise Error("TypeError: Torch lacks native infrastructure mode metadata for this API family")
    else:
        if mode_type is None or get_mode is None or py_type(get_mode) is not py_type(bi.len):
            raise Error("TypeError: Torch lacks compiled infrastructure mode metadata")
        if String(get_mode.__module__) != "torch._C" or String(get_mode.__name__) != "_get_dispatch_mode":
            raise Error("TypeError: Torch infrastructure mode metadata provenance differs")
        py_setitem(entries, "_get_dispatch_mode", get_mode)
        var modes = _torch_type_fields(mode_type)
        for name in [String("FAKE"), String("PROXY"), String("FUNCTIONAL")]:  # small-loop(the three fixed dispatch mode keys)
            var key = py_get(modes, name)
            if key is None or py_type(key) is not mode_type:
                raise Error("TypeError: Torch lacks native dispatch mode key " + name)
            py_call(keys.append, key)
    var dispatch = _torch_type_fields(dispatch_type)
    var pre = py_get(dispatch, "PreDispatch")
    if pre is None or py_type(pre) is not dispatch_type:
        raise Error("TypeError: Torch lacks native PreDispatch key metadata")
    var torch = py_get(sys.modules, "torch._tensor")
    if torch is None or py_type(torch) is not py_type(sys):
        raise Error("TypeError: Torch native Tensor class metadata is unavailable")
    var public = py_fields(torch)
    cache[].tensor = py_get(public, "Tensor")
    cache[].base = base
    var parameter = py_get(sys.modules, "torch.nn.parameter")
    cache[].parameter = PythonObject(None)
    if parameter is not None and py_type(parameter) is py_type(sys):
        cache[].parameter = py_get(py_fields(parameter), "Parameter")
    cache[].entries = entries
    cache[].methods = methods
    cache[].mode_keys = keys
    cache[].pre_dispatch = pre
    cache[].module = module


def native_torch_admit(value: PythonObject) raises:
    _torch_connect()
    var cache = NATIVE_TORCH.get_or_create_ptr()
    var cls = py_type(value)
    if cls is not cache[].tensor and cls is not cache[].parameter:
        raise Error("TypeError: Torch buffers require an exact Tensor or Parameter without Python subclass hooks")
    var bi = native_loaded_module("builtins")
    var hierarchy = py_call(bi.type.__dict__["__mro__"].__get__, cls)
    var has_base = False
    for i in range(len(hierarchy)):  # driver-loop: over native Tensor class ancestry metadata
        has_base = has_base or hierarchy[i] is cache[].base
    if not has_base:
        raise Error("TypeError: Torch Tensor class metadata does not derive from its native TensorBase")
    if Int(py=cache[].entries["_len_torch_function_stack"]()) != 0:
        raise Error("TypeError: Torch buffers cannot be used while a Python TorchFunctionMode is active")
    if Int(py=cache[].entries["_len_torch_dispatch_stack"]()) != 0:
        raise Error("TypeError: Torch buffers cannot be used while a Python TorchDispatchMode is active")
    for i in range(len(cache[].mode_keys)):  # driver-loop: over native Torch infrastructure mode keys
        if py_call(cache[].entries["_get_dispatch_mode"], cache[].mode_keys[i]) is not None:
            raise Error("TypeError: Torch buffers cannot be used while a Python infrastructure dispatch mode is active")
    # Torch's _ops.py sets this native TLS key whenever its Python pre-dispatch
    # stack is nonempty. Refusing the included key needs no Python stack walk.
    if Bool(py=py_call(cache[].entries["_dispatch_tls_is_dispatch_key_included"], cache[].pre_dispatch)):
        raise Error("TypeError: Torch buffers cannot be used while Python PreDispatch is active")


def native_torch_metadata(value: PythonObject) raises -> PythonObject:
    native_torch_admit(value)
    var cache = NATIVE_TORCH.get_or_create_ptr()
    var methods = cache[].methods
    # Invoke native base descriptors directly: an instance can shadow a
    # non-data Tensor method with a Python callable despite having exact type.
    var device = py_call(methods["device"].__get__, value)
    var dtype = py_call(methods["dtype"].__get__, value)
    var shape = py_call(methods["shape"].__get__, value)
    var address = py_call(methods["data_ptr"], value)
    var strides = py_call(methods["stride"], value)
    return Python.tuple(dtype, device.type, shape, strides, address)


def native_torch_numpy(value: PythonObject) raises -> PythonObject:
    native_torch_admit(value)
    var cache = NATIVE_TORCH.get_or_create_ptr()
    var device = py_call(cache[].methods["device"].__get__, value)
    if String(device.type) != "cpu":
        raise Error("TypeError: Torch input must be a host buffer; copy device tensors to CPU explicitly")
    var detached = py_call(cache[].methods["detach"], value)
    var result = py_call(cache[].methods["numpy"], detached)
    _ = detached^
    return result
