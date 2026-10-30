# SPDX-License-Identifier: Apache-2.0
# Shared exact label scalar and connected ClassLabels admission. No numbers
# ABC or Python subtype/iteration/conversion hook is part of this boundary.
from std.collections import Dict
from std.python import Python, PythonObject
from bindings.native_loaded_module import native_loaded_module
from bindings.native_numpy_metadata import native_numpy_dtype, native_numpy_module_fields
from bindings.py_balanced import py_call, py_type, py_get, py_contains
from bindings.py_attr import py_attr

def labels_primitive_kind(cls: PythonObject, context: PythonObject) raises -> Int:
    # 0 unsupported, 1 string, 2 bool, 3 integer, 4 real.
    var b = native_loaded_module("builtins")
    if cls is b.str:
        return 1
    if cls is b.bool:
        return 2
    if cls is b.int:
        return 3
    if cls is b.float:
        return 4
    var integers = context["integer_types"]
    for i in range(len(integers)):  # driver-loop: exact native integer classes
        if cls is integers[i]:
            return 3
    if py_get(context, "ndarray") is not None:
        if cls is context["numeric_dtypes"][Python.tuple("b", 1)].type:
            return 2
        var scalars = context["scalar_types"]
        for i in range(len(scalars)):  # driver-loop: remaining exact native real scalar classes
            if cls is scalars[i]:
                return 4
        # NumPy's extension exports dtype, but need not export scalar types
        # (2.5.4 does not export str_). The admitted static dtype factory
        # supplies the canonical text scalar identity without calling a
        # replaceable public scalar constructor or a user's conversion hook.
        var factory = native_numpy_dtype(context)
        var text_dtype = py_call(factory, "U")
        var text_type = text_dtype.type
        var numpy_fields = native_numpy_module_fields(context["modules"]["numpy"])
        if py_get(numpy_fields, "str_") is not text_type:
            raise Error("TypeError: np.str_ must remain NumPy's native type; Python callbacks cannot execute")
        if cls is text_type:
            return 1
        for code in ["q", "Q", "g"]:  # small-loop(native long-long and long-double scalar aliases)
            if cls is py_call(factory, code).type:
                return 4 if code == "g" else 3
    return 0


struct LabelTypeKinds(Movable):
    """Exact scalar classification reused only within one native label walk.

    Every element still supplies its own exact type. Native identity keys do
    not call metaclass hash/equality; retained owners prevent address reuse.
    Start a fresh table after any numerical dispatch/GIL-release boundary.
    """
    var kinds: Dict[Int, Int]
    var owners: List[PythonObject]

    def __init__(out self):
        self.kinds = Dict[Int, Int]()
        self.owners = List[PythonObject]()

    def kind(mut self, cls: PythonObject, context: PythonObject) raises -> Int:
        var identity = Int(cls._obj_ptr)
        if identity in self.kinds:
            return self.kinds[identity]
        # Preserve the existing ordered classifier on the first encounter,
        # rather than assuming an exact outer list has homogeneous elements.
        var kind = labels_primitive_kind(cls, context)
        self.owners.append(cls)
        self.kinds[identity] = kind
        return kind

    def admit(mut self, value: PythonObject, context: PythonObject) raises -> Int:
        var kind = self.kind(py_type(value), context)
        if kind == 0:
            raise Error("TypeError: labels require exact native strings or real numeric scalars without Python protocols")
        return kind


def labels_admit_primitive(value: PythonObject, context: PythonObject) raises -> Int:
    var kind = labels_primitive_kind(py_type(value), context)
    if kind == 0:
        raise Error("TypeError: labels require exact native strings or real numeric scalars without Python protocols")
    return kind


def _labels_class(context: PythonObject) raises -> PythonObject:
    var b = native_loaded_module("builtins")
    var cls = context["classifier_labels_class"]
    if py_type(cls) is not b.type:
        raise Error("TypeError: native class-label output requires connected class metadata")
    var mro = py_call(b.type.__dict__["__mro__"].__get__, cls)
    var ancestry = context["classifier_labels_mro"]
    if len(mro) != len(ancestry):
        raise Error("TypeError: native class-label output ancestry changed")
    for i in range(len(mro)):  # driver-loop: retained ancestry before allocation or releasing old labels
        if mro[i] is not ancestry[i]:
            raise Error("TypeError: native class-label output ancestry changed")
    var fields = py_call(b.type.__dict__["__dict__"].__get__, cls)
    var connected = context["classifier_labels_namespace"]
    var names = py_call(b.list, fields)
    if len(names) != len(connected):
        raise Error("TypeError: native class-label output metadata changed")
    for i in range(len(names)):  # driver-loop: immutable connected output class metadata
        if py_type(names[i]) is not b.str or not py_contains(connected, names[i]) or fields[names[i]] is not connected[names[i]]:
            raise Error("TypeError: native class-label output cannot invoke modified Python hooks")
    return cls


def classifier_label_list(value: PythonObject, context: PythonObject) raises -> PythonObject:
    var cls = py_get(context, "classifier_labels_class")
    if cls is None or py_type(value) is not cls:
        return value
    _ = _labels_class(context)
    # list.copy is the native list implementation, not an iterable protocol.
    return py_call(py_attr(py_type(Python.list()), "copy"), value)


