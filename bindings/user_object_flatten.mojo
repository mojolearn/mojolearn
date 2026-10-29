# SPDX-License-Identifier: Apache-2.0
# Operation-local flatten admission: no user getattr/tolist/iteration hooks.
from std.collections import Dict
from std.memory import bitcast
from std.python import Python, PythonObject
from std.python._cpython import PyObjectPtr
from bindings.native_loaded_module import NativeModules, native_namespace_string_keys
from bindings.native_array_snapshot import native_array_snapshot
from bindings.mode_metadata_native import _class_fields
from bindings.py_balanced import py_call, py_type, py_get


struct UserObjectFlatten(Movable):
    var mods: NativeModules
    var b: PythonObject
    var list_type: PythonObject
    var tuple_type: PythonObject
    var leaf_types: List[PythonObject]
    var methods: Dict[Int, PythonObject]
    var owners: List[PythonObject]
    var array_context: PythonObject
    var array_class: PythonObject
    var foreign_ready: Bool

    def __init__(out self) raises:
        self.mods = NativeModules()
        self.b = self.mods.module("builtins")
        self.list_type = self.b.list
        self.tuple_type = self.b.tuple
        self.leaf_types = List[PythonObject]()
        for cls in [self.b.str, self.b.bytes, self.b.bytearray, self.b.int,  # small-loop(fixed primitive leaf types)
                    self.b.float, self.b.bool, self.b.complex, py_type(PythonObject(None))]:
            self.leaf_types.append(cls)
        self.methods = Dict[Int, PythonObject]()
        self.owners = List[PythonObject]()
        self.array_context = PythonObject(None)
        self.array_class = PythonObject(None)
        self.foreign_ready = False

    def prepare_foreign(mut self) raises:
        if self.foreign_ready:
            return
        var sys = self.mods.module("sys")
        var module = py_get(sys.modules, "mojolearn._array")
        if module is not None:
            if py_type(module) is not py_type(sys):
                raise Error("TypeError: user-object Array metadata requires its native module")
            ref cpy = Python().cpython()
            var raw = cpy.lib.get_function[PyObjectPtr]("PyModule_GetDict")(module._obj_ptr)
            if not raw or not native_namespace_string_keys(raw, self.b.str):
                raise Error("TypeError: user-object Array metadata requires native string fields")
            var fields = PythonObject(from_borrowed=raw)
            var context = py_get(fields, "_ARRAY_CONTEXT")
            if context is not None:
                if py_type(context) is not self.b.dict or not native_namespace_string_keys(context._obj_ptr, self.b.str):
                    raise Error("TypeError: user-object Array context requires native string fields")
                var cls = py_get(context, "array_class")
                if cls is not py_get(fields, "Array") or py_type(cls) is not self.b.type:
                    raise Error("TypeError: user-object Array class metadata changed")
                self.array_context = Python.dict(array_class=cls)
                self.array_class = cls
        self.foreign_ready = True

    def native_method(mut self, cls: PythonObject) raises -> PythonObject:
        var identity = Int(cls._obj_ptr)
        if identity in self.methods:
            return self.methods[identity]
        # Only known native storage types may supply a conversion descriptor.
        # Heap subclasses and Python metaclasses cannot acquire this admission.
        var array_type = self.mods.module("array").array
        var supported = cls is array_type or cls is self.b.memoryview
        ref cpy = Python().cpython()
        var flags = cpy.lib.get_function[UInt64]("PyType_GetFlags")(cls._obj_ptr)
        if not supported and py_type(cls) is self.b.type and (flags & UInt64(512)) == 0:
            var module_name = py_call(self.b.type.__getattribute__, cls, "__module__")
            supported = py_type(module_name) is self.b.str and String(module_name) == "numpy"
        if not supported:
            raise Error("TypeError: user-object flatten requires native containers or scalars without Python conversion hooks")
        var descriptor = py_call(self.b.type.__getattribute__, cls, "tolist")
        var types = self.mods.module("types")
        if py_type(descriptor) is not types.MethodDescriptorType:
            raise Error("TypeError: user-object flatten requires a native tolist descriptor")
        self.owners.append(cls)
        self.methods[identity] = descriptor
        return descriptor

    def array_into(mut self, value: PythonObject, output: PythonObject) raises:
        var captured = native_array_snapshot(self.mods, value, self.array_context, True)
        var shape = List[Int]()
        var strides = List[Int]()
        for axis in range(len(captured.shape)):  # small-loop(captured native Array dimensions)
            shape.append(Int(py=captured.shape[axis]))
            strides.append(Int(py=captured.strides[axis]))
        var count = Int(py=captured.size)
        var dtype = String(captured.dtype)
        var address = Int(py=captured._addr)
        ref cpy = Python().cpython()
        var append = cpy.lib.get_function[Int32]("PyList_Append")
        for index in range(count):  # cpu-route: publish completed captured native Array scalars in logical C order
            var remaining = index
            var offset = 0
            for reverse_axis in range(len(shape)):  # small-loop(native rank/stride words)
                var axis = len(shape) - reverse_axis - 1
                offset += (remaining % shape[axis]) * strides[axis]
                remaining //= shape[axis]
            var at = address + offset
            var item: PythonObject
            if dtype == "<f4":
                item = PythonObject(Float64(MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=at)[0]))
            elif dtype == "<f8":
                item = PythonObject(MutPointer[Float64, MutUntrackedOrigin](unsafe_from_address=at)[0])
            elif dtype == "<f2":
                item = PythonObject(Float64(bitcast[DType.float16](MutPointer[UInt16, MutUntrackedOrigin](unsafe_from_address=at)[0])))
            else:
                var integer = Int64(0)
                if dtype == "<i8":
                    integer = MutPointer[Int64, MutUntrackedOrigin](unsafe_from_address=at)[0]
                elif dtype == "<i4":
                    integer = Int64(MutPointer[Int32, MutUntrackedOrigin](unsafe_from_address=at)[0])
                elif dtype == "<u4":
                    integer = Int64(MutPointer[UInt32, MutUntrackedOrigin](unsafe_from_address=at)[0])
                elif dtype == "<u2":
                    integer = Int64(MutPointer[UInt16, MutUntrackedOrigin](unsafe_from_address=at)[0])
                elif dtype == "<i1":
                    integer = Int64(MutPointer[Int8, MutUntrackedOrigin](unsafe_from_address=at)[0])
                else:
                    integer = Int64(MutPointer[UInt8, MutUntrackedOrigin](unsafe_from_address=at)[0])
                item = PythonObject(integer)
            if append(output._obj_ptr, item._obj_ptr) != 0:
                raise cpy.unsafe_get_error()
        _ = captured

    def into(mut self, value: PythonObject, output: PythonObject, converted: Bool = False) raises:
        ref cpy = Python().cpython()
        var cls = py_type(value)
        if cls is self.list_type or cls is self.tuple_type:
            var size = len(value)
            var get_list = cpy.lib.get_function[PyObjectPtr]("PyList_GetItem")
            var get_tuple = cpy.lib.get_function[PyObjectPtr]("PyTuple_GetItem")
            for i in range(size):  # cpu-route: walk exact native nested list/tuple slots once
                var raw = get_list(value._obj_ptr, i) if cls is self.list_type else get_tuple(value._obj_ptr, i)
                if not raw:
                    raise cpy.unsafe_get_error()
                var item = PythonObject(from_borrowed=raw)
                self.into(item, output)
            _ = value
            return
        var primitive = converted
        for kind in self.leaf_types:  # small-loop(fixed primitive scalar types)
            primitive = primitive or cls is kind
        if primitive:
            if cpy.lib.get_function[Int32]("PyList_Append")(output._obj_ptr, value._obj_ptr) != 0:
                raise cpy.unsafe_get_error()
            return
        self.prepare_foreign()
        if cls is self.array_class:
            self.array_into(value, output)
            return
        var method = self.native_method(cls)
        var result = py_call(method, value)
        # A native scalar's tolist may return itself (e.g. longdouble). The
        # original contract boxes once, then treats a non-container as a leaf.
        self.into(result, output, True)


def user_object_flatten(value: PythonObject) raises -> PythonObject:
    var reader = UserObjectFlatten()
    var output = Python.list()
    reader.into(value, output)
    return output
