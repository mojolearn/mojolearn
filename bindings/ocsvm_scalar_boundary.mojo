# SPDX-License-Identifier: Apache-2.0
"""Exact scalar admission before OneClassSVM's native C option comparisons."""
from std.python import Python, PythonObject
from std.python._cpython import PyObjectPtr
from bindings.py_balanced import py_type, py_get, py_call
from bindings.native_loaded_module import native_loaded_module, native_namespace_string_keys
from core.shared_slot import SharedSlot
from bindings.native_numpy_metadata import native_numpy_dtype
from bindings.decomp_scalar_compare import decomp_equal_admitted
from bindings.family_common import family_repr
from bindings.p3_array_arena_records import P3_DICT_NATIVE_STRING_KEYS, p3_dict_admitted, p3_dict_admit


struct OcScalarTypes(Defaultable, Movable):
    var constructor: PythonObject
    var arrays: List[PythonObject]
    var scalars: List[PythonObject]
    var objects: List[Bool]

    def __init__(out self):
        self.constructor = PythonObject(None)
        self.arrays = List[PythonObject]()
        self.scalars = List[PythonObject]()
        self.objects = List[Bool]()

    def connect(mut self, constructor: PythonObject) raises:
        if self.constructor is constructor:
            return
        self.arrays = List[PythonObject]()
        self.scalars = List[PythonObject]()
        self.objects = List[Bool]()
        for code in ["?", "b", "B", "h", "H", "i", "I", "l", "L", "q", "Q", "e", "f", "d", "g", "F", "D", "G", "U", "S", "m", "M", "V", "O", "T"]:
            var native = PythonObject(None)
            try:
                native = py_call(constructor, code)
            except:
                continue  # version-dependent native dtype aliases
            self.arrays.append(py_type(native))
            self.scalars.append(native.type if code != "O" and code != "V" else PythonObject(None))
            self.objects.append(code == "O")
        self.constructor = constructor


# Boundary metadata only. GIL-held setup pins immutable native dtype/scalar
# classes once; no user object is admitted by a name or by a mutable type list.
comptime OC_SCALAR_TYPES = SharedSlot[OcScalarTypes, "MojolearnOcScalarTypesV1"]


def oc_dictionary(value: PythonObject) raises:
    var b = native_loaded_module("builtins")
    if py_type(value) is not b.dict:
        raise Error("TypeError: OneClassSVM metadata requires an exact native dictionary")
    if not p3_dict_admitted(P3_DICT_NATIVE_STRING_KEYS, value):
        if not native_namespace_string_keys(value._obj_ptr, b.str):
            raise Error("TypeError: OneClassSVM metadata requires native string keys")
        p3_dict_admit(P3_DICT_NATIVE_STRING_KEYS, value)


def oc_scalar_admit(value: PythonObject, context: PythonObject) raises:
    var current = value
    var visited = List[Int]()
    var b = native_loaded_module("builtins")
    var ndarray = py_get(context, "ndarray")
    while True:  # boundary-only traversal of a scalar object-array owner chain
        var cls = py_type(current)
        if (cls is b.float or cls is b.int or cls is b.bool or cls is b.str or cls is b.bytes
            or cls is b.bytearray or cls is b.complex or cls is py_type(PythonObject(None))):
            return
        if ndarray is None:
            raise Error("TypeError: OneClassSVM parameters require native scalar values without Python hooks")
        var constructor = native_numpy_dtype(context)
        var plan = OC_SCALAR_TYPES.get_or_create_ptr()
        plan[].connect(constructor)
        var dtype = current.dtype if cls is ndarray else PythonObject(None)
        var object_array = False
        for index in range(len(plan[].arrays)):  # immutable native dtype/scalar identity plan
            if cls is ndarray and py_type(dtype) is plan[].arrays[index]:
                if plan[].objects[index]:
                    object_array = True
                    break
                if not Bool(py=dtype.hasobject):
                    return
            elif cls is not ndarray and cls is plan[].scalars[index]:
                return
        if not object_array:
            raise Error("TypeError: OneClassSVM parameters require native scalar values without Python hooks")
        var size = Int(py=current.size)
        if size == 0:
            return  # no object comparison can run; native truth keeps its own refusal
        if size != 1:
            raise Error("ValueError: The truth value of an array with more than one element is ambiguous. Use a.any() or a.all()")
        var address = Int(current._obj_ptr)
        for previous in visited:  # scalar nesting identities, never tensor values
            if previous == address:
                raise Error("TypeError: OneClassSVM parameters require an acyclic native scalar owner")
        visited.append(address)
        # ndarray is exact; item() is its native C storage accessor. It
        # returns the sole pinned object without invoking that object's hooks.
        current = py_call(current.item)


def oc_name(value: PythonObject, name: String, context: PythonObject) raises -> Bool:
    oc_scalar_admit(value, context)
    return decomp_equal_admitted(value, PythonObject(name))


def oc_kernel_admit(value: PythonObject, context: PythonObject) raises -> Bool:
    oc_scalar_admit(value, context)
    for name in ["linear", "poly", "rbf", "sigmoid", "precomputed"]:  # five native option names
        if decomp_equal_admitted(value, PythonObject(name)):
            return True
    return False


def oc_parameter_repr(value: PythonObject, context: PythonObject) raises -> String:
    if py_type(value) is py_get(context, "ndarray"):
        return "<native scalar array>"
    return family_repr(value)


def oc_mode_value(context: PythonObject, mode: Int) raises -> PythonObject:
    oc_dictionary(context)
    var backend = context["backend_fields"]
    oc_dictionary(backend)
    var mapping = py_get(backend, "_CODE_MODE", PythonObject(None))
    if mapping is None:
        return PythonObject(None)
    var b = native_loaded_module("builtins")
    if py_type(mapping) is not b.dict:
        raise Error("TypeError: OneClassSVM mode metadata requires an exact native dictionary")
    ref cpy = Python().cpython()
    var position = Int(0)
    var key = PyObjectPtr()
    var item = PyObjectPtr()
    var next = cpy.lib.get_function[Int32]("PyDict_Next")
    for i in range(len(mapping)):  # native metadata keys; no hashing or equality during admission
        if next(mapping._obj_ptr, MutPointer(to=position), MutPointer(to=key), MutPointer(to=item)) == 0:
            break
        var pinned = PythonObject(from_borrowed=key)
        var cls = py_type(pinned)
        if cls is not b.tuple and cls is not b.frozenset and cls is not b.range:
            oc_scalar_admit(pinned, context)
    return py_get(mapping, PythonObject(mode), PythonObject(None))
