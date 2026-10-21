# SPDX-License-Identifier: Apache-2.0
# Torch's ordinary Tensor type can still dispatch to Python mode hooks.
# Read only Torch's native thread-local stack/dispatch-key metadata before
# touching a Tensor. Never disable modes or call Python torch.overrides /
# torch.utils._python_dispatch helpers. The C entries are the ones used by
# those helpers in the supported installed Torch, and are checked before cache.
from std.ffi import _Global
from std.python import Python, PythonObject


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
    var bi = Python.import_module("builtins")
    # Use type's native dictionary descriptor, bypassing a custom metaclass's
    # __getattribute__ and any user property named __dict__.
    return bi.type.__dict__["__dict__"].__get__(cls)


def native_torch_exact(cls: PythonObject) raises -> Bool:
    var bi = Python.import_module("builtins")
    var sys = Python.import_module("sys")
    var module = sys.modules.get("torch._tensor")
    if module is None or bi.type(module) is not bi.type(sys):
        return False
    var fields = bi.object.__getattribute__(module, "__dict__")
    if cls is fields.get("Tensor"):
        return True
    var parameter = sys.modules.get("torch.nn.parameter")
    if parameter is not None and bi.type(parameter) is bi.type(sys):
        return cls is bi.object.__getattribute__(parameter, "__dict__").get("Parameter")
    return False


def _torch_connect() raises:
    var cache = NATIVE_TORCH.get_or_create_ptr()
    var bi = Python.import_module("builtins")
    var sys = Python.import_module("sys")
    var module = sys.modules.get("torch._C")
    if module is None or bi.type(module) is not bi.type(sys):
        raise Error("TypeError: Torch native mode admission requires its compiled extension")
    if cache[].module is module:
        return
    var fields = bi.object.__getattribute__(module, "__dict__")
    var entries = Python.dict()
    for name in [String("_len_torch_function_stack"), String("_len_torch_dispatch_stack"),
                 String("_get_dispatch_mode"), String("_dispatch_tls_is_dispatch_key_included")]:
        var entry = fields.get(name)
        if entry is None or bi.type(entry) is not bi.type(bi.len):
            raise Error("TypeError: Torch lacks compiled mode metadata entry " + name)
        if String(entry.__module__) != "torch._C" or String(entry.__name__) != name:
            raise Error("TypeError: Torch mode metadata entry provenance differs: " + name)
        entries[name] = entry
    var base = fields.get("TensorBase")
    if base is None:
        raise Error("TypeError: Torch lacks native TensorBase metadata")
    var base_fields = _torch_type_fields(base)
    var methods = Python.dict()
    for name in [String("detach"), String("numpy"), String("data_ptr"), String("stride")]:
        var entry = base_fields.get(name)
        if entry is None or bi.type(entry) is not bi.type(bi.list.append):
            raise Error("TypeError: Torch tensor conversion requires native method " + name)
        methods[name] = entry
    for name in [String("device"), String("dtype"), String("shape")]:
        var entry = base_fields.get(name)
        if entry is None or bi.type(entry) is not bi.type(bi.memoryview.obj):
            raise Error("TypeError: Torch tensor metadata requires native descriptor " + name)
        methods[name] = entry
    var mode_type = fields.get("_TorchDispatchModeKey")
    var dispatch_type = fields.get("DispatchKey")
    if mode_type is None or dispatch_type is None:
        raise Error("TypeError: Torch lacks native dispatch key metadata")
    var modes = _torch_type_fields(mode_type)
    var keys = Python.list()
    for name in [String("FAKE"), String("PROXY"), String("FUNCTIONAL")]:
        var key = modes.get(name)
        if key is None or bi.type(key) is not mode_type:
            raise Error("TypeError: Torch lacks native dispatch mode key " + name)
        keys.append(key)
    var dispatch = _torch_type_fields(dispatch_type)
    var pre = dispatch.get("PreDispatch")
    if pre is None or bi.type(pre) is not dispatch_type:
        raise Error("TypeError: Torch lacks native PreDispatch key metadata")
    var torch = sys.modules.get("torch._tensor")
    if torch is None or bi.type(torch) is not bi.type(sys):
        raise Error("TypeError: Torch native Tensor class metadata is unavailable")
    var public = bi.object.__getattribute__(torch, "__dict__")
    cache[].tensor = public.get("Tensor")
    cache[].base = base
    var parameter = sys.modules.get("torch.nn.parameter")
    cache[].parameter = PythonObject(None)
    if parameter is not None and bi.type(parameter) is bi.type(sys):
        cache[].parameter = bi.object.__getattribute__(parameter, "__dict__").get("Parameter")
    cache[].entries = entries
    cache[].methods = methods
    cache[].mode_keys = keys
    cache[].pre_dispatch = pre
    cache[].module = module


def native_torch_admit(value: PythonObject) raises:
    _torch_connect()
    var cache = NATIVE_TORCH.get_or_create_ptr()
    var cls = Python.import_module("builtins").type(value)
    if cls is not cache[].tensor and cls is not cache[].parameter:
        raise Error("TypeError: Torch buffers require an exact Tensor or Parameter without Python subclass hooks")
    var bi = Python.import_module("builtins")
    var hierarchy = bi.type.__dict__["__mro__"].__get__(cls)
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
        if cache[].entries["_get_dispatch_mode"](cache[].mode_keys[i]) is not None:
            raise Error("TypeError: Torch buffers cannot be used while a Python infrastructure dispatch mode is active")
    # Torch's _ops.py sets this native TLS key whenever its Python pre-dispatch
    # stack is nonempty. Refusing the included key needs no Python stack walk.
    if Bool(py=cache[].entries["_dispatch_tls_is_dispatch_key_included"](cache[].pre_dispatch)):
        raise Error("TypeError: Torch buffers cannot be used while Python PreDispatch is active")


def native_torch_metadata(value: PythonObject) raises -> PythonObject:
    native_torch_admit(value)
    var cache = NATIVE_TORCH.get_or_create_ptr()
    var methods = cache[].methods
    # Invoke native base descriptors directly: an instance can shadow a
    # non-data Tensor method with a Python callable despite having exact type.
    var device = methods["device"].__get__(value)
    var dtype = methods["dtype"].__get__(value)
    var shape = methods["shape"].__get__(value)
    var address = methods["data_ptr"](value)
    var strides = methods["stride"](value)
    return Python.tuple(dtype, device.type, shape, strides, address)


def native_torch_numpy(value: PythonObject) raises -> PythonObject:
    native_torch_admit(value)
    var cache = NATIVE_TORCH.get_or_create_ptr()
    var device = cache[].methods["device"].__get__(value)
    if String(device.type) != "cpu":
        raise Error("TypeError: Torch input must be a host buffer; copy device tensors to CPU explicitly")
    var detached = cache[].methods["detach"](value)
    var result = cache[].methods["numpy"](detached)
    _ = detached^
    return result
