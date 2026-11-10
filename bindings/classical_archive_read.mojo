# SPDX-License-Identifier: Apache-2.0
# Archive path/file UNPACK and result/model PACK. Native Mojo owns ZIP/NPY
# framing, bounds, dtype/layout plans, Unicode decoding and decompression.
from std.collections import Dict
from std.python import Python, PythonObject
from std.python._cpython import PyObjectPtr
from bindings.cpython_table import cpython_table
from bindings.py_balanced import py_call, py_type, py_get, py_setitem, py_contains
from bindings.classical_archive import _archive_methods, _archive_path, _archive_file_call, _archive_class_name_unpack
from bindings.optimizer_public_words import stash_python_error
from bindings.classical_arrays import classical_numpy, classical_is_array, classical_dtype_str
from bindings.classical_archive_input import ArchiveReadInput, archive_input_unpack
from bindings.classical_archive_output import archive_members_pack, archive_events_pack
from bindings.classical_archive_linear import archive_class_names_unpack, archive_print_mode_unpack, archive_linear_pack
from core.native_archive_linear import native_archive_string, native_linear_archive_plan
from core.native_archive_read import native_archive_read_status
from core.native_archive_repr import NativeArchiveReprNode, archive_repr_format


def _archive_builtin_repr(value: PythonObject) raises -> String:
    # Call only with the exact native scalar/file types admitted below. A C
    # repr entry is not an admission check for an arbitrary Python object.
    ref cpy = Python().cpython()
    var result = cpy.lib.get_function[PyObjectPtr]("PyObject_Repr")(value._obj_ptr)
    if not result:
        var failure = stash_python_error()
        raise failure^
    return String(PythonObject(from_owned=result))


def _archive_model_fields(value: PythonObject) raises -> PythonObject:
    # A connected model owns an ordinary instance dictionary. Read that native
    # storage directly, without looking up the mutable builtins.object name.
    ref cpy = Python().cpython()
    var result = cpython_table()[].PyObject_GenericGetDict(value._obj_ptr, PyObjectPtr())
    if not result:
        raise cpy.unsafe_get_error()
    return PythonObject(from_owned=result)


def _archive_read_path(value: PythonObject, context: PythonObject) raises -> PythonObject:
    # Only pathlib component storage receives lexical normalization; exact
    # str and bytes retain their original filename spelling and error state.
    return _archive_path(value, context, True)


def _archive_repr_types() raises -> PythonObject:
    # Public constructors establish immutable builtin identities without
    # consulting mutable builtins.complex/set/frozenset module attributes.
    ref cpy = Python().cpython()
    var complex_ptr = cpy.lib.get_function[PyObjectPtr]("PyComplex_FromDoubles")(Float64(0), Float64(0))
    if not complex_ptr:
        var failure = stash_python_error()
        raise failure^
    var complex_value = PythonObject(from_owned=complex_ptr)
    var set_ptr = cpy.lib.get_function[PyObjectPtr]("PySet_New")(PyObjectPtr())
    if not set_ptr:
        var failure = stash_python_error()
        raise failure^
    var set_value = PythonObject(from_owned=set_ptr)
    var frozen_ptr = cpy.lib.get_function[PyObjectPtr]("PyFrozenSet_New")(PyObjectPtr())
    if not frozen_ptr:
        var failure = stash_python_error()
        raise failure^
    var frozen_value = PythonObject(from_owned=frozen_ptr)
    return Python.tuple(py_type(complex_value), py_type(set_value), py_type(frozen_value))


def _archive_repr_kind(value: PythonObject, context: PythonObject,
                       native_types: PythonObject) raises -> Int:
    var cls = py_type(value)
    if (cls is py_type(PythonObject("")) or cls is context["archive_bytes_type"]
            or cls is py_type(PythonObject(0)) or cls is py_type(PythonObject(True))
            or cls is py_type(PythonObject(Float64(0)))
            or cls is native_types[0] or cls is context["archive_bytearray_type"]
            or cls is context["archive_memoryview_type"] or value is None):
        return 0
    var paths = context["archive_path_types"]
    if cls is paths[0] or cls is paths[1]:
        return 1
    if cls is py_type(Python.list()):
        return 2
    if cls is py_type(Python.tuple()):
        return 3
    if cls is py_type(Python.dict()):
        return 4
    if cls is native_types[1]:
        return 5
    if cls is native_types[2]:
        return 6
    var types = context["archive_file_types"]
    for i in range(len(types)):  # driver-loop: connected exact native file identities
        if cls is types[i]:
            return 7 + i
    raise Error("TypeError: archive error paths require native names without Python repr callbacks")


def _archive_repr_file_name(value: PythonObject, context: PythonObject, kind: Int) raises -> PythonObject:
    _ = _archive_methods(value, context)
    if kind == 8 and Bool(py=value.closed):
        return Python.tuple()
    # Buffered .name delegates to its exact native FileIO owner. Guard every
    # dictionary key before lookup, including nested peer files. Native dict
    # lookup alone could otherwise execute a foreign key's comparison hook.
    var raw_file = value if kind == 8 else value.raw
    var fields = _archive_model_fields(raw_file)
    if py_type(fields) is not py_type(Python.dict()):
        raise Error("TypeError: archive file names require native dictionary storage")
    var keys = py_call(py_type(Python.list()), fields)
    for i in range(len(keys)):  # driver-loop: native FileIO metadata keys
        if py_type(keys[i]) is not py_type(PythonObject("")):
            raise Error("TypeError: archive file metadata requires exact string keys without Python comparison hooks")
    # Read through the public C API so a genuine native failure keeps its
    # class and traceback. Only AttributeError means the name was deleted;
    # allocation and other native errors must not become a nameless display.
    ref cpy = Python().cpython()
    var attribute_error = context["archive_attribute_error"]
    var matches = cpy.lib.get_function[Int32]("PyErr_ExceptionMatches")
    var clear = cpython_table()[].PyErr_Clear
    var key = String("name")
    var pointer = cpy.lib.get_function[PyObjectPtr]("PyObject_GetAttrString")(value._obj_ptr, key.as_c_string_slice().unsafe_ptr())
    if not pointer:
        if matches(attribute_error._obj_ptr) != 0:
            clear()
            return Python.tuple()
        var failure = stash_python_error()
        raise failure^
    var name = PythonObject(from_owned=pointer)
    return Python.tuple(name)


def _archive_repr_members(value: PythonObject, kind: Int) raises -> PythonObject:
    if kind == 4:
        return py_call(py_type(Python.list()), py_call(value.items))
    if kind == 5 or kind == 6:
        return py_call(py_type(Python.list()), value)
    return value


def _archive_repr_admit(value: PythonObject, context: PythonObject,
                        native_types: PythonObject) raises -> Bool:
    # Admission is iterative and visits each identity once. A recursive Mojo
    # walk consumed much more C stack than builtin repr; invoking another
    # PythonObject helper near its guard exhausted CPython's error headroom.
    var pending = List[PythonObject]()
    pending.append(value)
    var seen = Dict[Int, Bool]()
    var has_path = False
    while len(pending):  # driver-loop: native representation graph admission
        var item = pending.pop()
        var identity = Int(item._obj_ptr)
        if identity in seen:
            continue
        seen[identity] = True
        var kind = _archive_repr_kind(item, context, native_types)
        if kind == 1:
            # Admit stored path components now; never call pathlib.__repr__.
            _ = _archive_read_path(item, context)
            has_path = True
        elif kind >= 2 and kind <= 6:
            var members = _archive_repr_members(item, kind)
            for i in range(len(members)):  # driver-loop: exact container references
                if kind == 4:
                    pending.append(members[i][0])
                    pending.append(members[i][1])
                else:
                    pending.append(members[i])
        elif kind >= 8:
            var name = _archive_repr_file_name(item, context, kind)
            if len(name):
                pending.append(name[0])
    return has_path


struct _ArchiveReprFailure(Movable):
    # Boundary-only exception pins. A scalar repr can fail before a later
    # graph node; native RUN selects that same first failure in display order.
    var kind: PythonObject
    var value: PythonObject
    var traceback: PythonObject

    def __init__(out self) raises:
        ref cpy = Python().cpython()
        var words = cpy.PyErr_FetchTriple()
        self.kind = PythonObject(from_owned=words[0]) if words[0] else PythonObject(None)
        self.value = PythonObject(from_owned=words[1]) if words[1] else PythonObject(None)
        self.traceback = PythonObject(from_owned=words[2]) if words[2] else PythonObject(None)

    def raise_saved(self) raises:
        ref cpy = Python().cpython()
        var kind = PyObjectPtr() if self.kind is None else cpy.Py_NewRef(self.kind._obj_ptr)
        var value = PyObjectPtr() if self.value is None else cpy.Py_NewRef(self.value._obj_ptr)
        var traceback = PyObjectPtr() if self.traceback is None else cpy.Py_NewRef(self.traceback._obj_ptr)
        cpy.lib.get_function[NoneType]("PyErr_Restore")(kind, value, traceback)
        var failure = stash_python_error()
        raise failure^


def _archive_repr_leaf_unpack(value: PythonObject, mut node: NativeArchiveReprNode,
                              mut failures: List[_ArchiveReprFailure]) raises:
    # Exact native leaves were admitted by the complete first pass. Capture
    # their C-API failures without replacing the pending exception TLS slot.
    ref cpy = Python().cpython()
    var result = cpy.lib.get_function[PyObjectPtr]("PyObject_Repr")(value._obj_ptr)
    if not result:
        node.failure = len(failures)
        failures.append(_ArchiveReprFailure())
        return
    node.text = String(PythonObject(from_owned=result))


def _archive_repr_child_unpack(value: PythonObject, mut pins: List[PythonObject],
                               mut nodes: List[NativeArchiveReprNode],
                               mut indices: Dict[Int, Int]) raises -> Int:
    var identity = Int(value._obj_ptr)
    if identity in indices:
        return indices[identity]
    var index = len(nodes)
    indices[identity] = index
    pins.append(value)
    nodes.append(NativeArchiveReprNode())
    return index


def _archive_repr_paths(value: PythonObject, context: PythonObject,
                        native_types: PythonObject) raises -> String:
    # UNPACK snapshots native container edges, scalar display values and owner
    # pins. Native RUN owns formatting order, cycle handling and text state.
    var pins = List[PythonObject]()
    var nodes = List[NativeArchiveReprNode]()
    var indices = Dict[Int, Int]()
    var failures = List[_ArchiveReprFailure]()
    _ = _archive_repr_child_unpack(value, pins, nodes, indices)
    var index = 0
    while index < len(pins):  # unpack: admitted native object graph and owners
        var item = pins[index]
        var kind = _archive_repr_kind(item, context, native_types)
        var node = NativeArchiveReprNode(kind)
        if kind == 0 or kind == 7:
            _archive_repr_leaf_unpack(item, node, failures)
        elif kind == 1:
            var normalized = _archive_read_path(item, context)
            node.pure_path = py_type(item) is context["archive_path_types"][1]
            _archive_repr_leaf_unpack(normalized, node, failures)
        elif kind >= 2 and kind <= 6:
            var members = _archive_repr_members(item, kind)
            for i in range(len(members)):  # unpack: exact container references
                if kind == 4:
                    node.children.append(_archive_repr_child_unpack(members[i][0], pins, nodes, indices))
                    node.children.append(_archive_repr_child_unpack(members[i][1], pins, nodes, indices))
                else:
                    node.children.append(_archive_repr_child_unpack(members[i], pins, nodes, indices))
        else:
            var name = _archive_repr_file_name(item, context, kind)
            if len(name) == 0:
                node.kind = 0
                _archive_repr_leaf_unpack(item, node, failures)
            else:
                node.children.append(_archive_repr_child_unpack(name[0], pins, nodes, indices))
                if kind == 8:
                    _archive_repr_leaf_unpack(item.mode, node, failures)
                    node.closefd = Bool(py=item.closefd)
        nodes[index] = node^
        index += 1
    var result = archive_repr_format(nodes)
    if result.failure >= 0:
        failures[result.failure].raise_saved()
    _ = pins
    return result.text^


def _archive_read_repr(value: PythonObject, context: PythonObject) raises -> String:
    var native_types = _archive_repr_types()
    if _archive_repr_admit(value, context, native_types):
        return _archive_repr_paths(value, context, native_types)
    # The complete graph is now exact-native, including every file-name dict
    # and nested member. Builtin scalar/container and exact FileIO/BufferedIO
    # repr retain the GIL and only traverse this admitted metadata (CPython
    # fileio_repr/buffered_repr); no user hook or I/O is reached here. Let
    # CPython own its recursion guard and unwind before fetching its exception.
    return _archive_builtin_repr(value)


def _archive_read_input(path: PythonObject, context: PythonObject) raises -> ArchiveReadInput:
    var file = path
    var close = True
    var types = context["archive_file_types"]
    for k in range(len(types)):
        if py_type(path) is types[k]:
            close = False
    if close:
        file = _archive_file_call(context["archive_open"], _archive_read_path(path, context), "rb")
    return archive_input_unpack(file, close, context)


def classical_archive_read(path: PythonObject, context: PythonObject) raises -> PythonObject:
    # Generic format boundary used by the remaining saved-model families.
    var admitted = _archive_read_input(path, context)
    var minor = Int(py=context["archive_python_minor"])
    var micro = Int(py=context["archive_python_micro"])
    try:
        var members = native_archive_read_status(admitted.span, minor, micro)
        return archive_members_pack(members, context)
    finally:
        admitted.close()


def _archive_dtype_name(value: PythonObject, context: PythonObject) raises -> String:
    # The dtype a refusal names, as `_serialize._dtype_name` named it.
    var strings = py_type(PythonObject(""))
    var bytes = context["archive_bytes_type"]
    var lists = py_type(Python.list())
    if classical_is_array(value, context) or (classical_numpy(context) and py_type(value) is context["ndarray"]):
        return classical_dtype_str(value, context)
    if py_type(value) is strings:
        return "<U"
    if py_type(value) is bytes:
        return "|S"
    if py_type(value) is lists:
        var inner = value
        while py_type(inner) is lists and len(inner) != 0:  # driver-loop: over the nesting depth of a stored field
            inner = inner[0]
        if py_type(inner) is strings:
            return "<U"
        if py_type(inner) is bytes:
            return "|S"
        return "list"
    return _archive_class_name_unpack(py_type(value))


def _archive_missing(name: String) -> String:
    return "mojolearn: model file is missing field '" + name + "'"


def _archive_exact(arrays: PythonObject, name: String, dtype: String, context: PythonObject) raises -> PythonObject:
    # The refusals keep `_serialize.exact`'s wording: a cast could change
    # bits, so a dtype mismatch is refused, never converted.
    if not py_contains(arrays, name):
        raise Error(_archive_missing(name))
    var value = arrays[name]
    var numeric = py_type(value) is context["ndarray"] if classical_numpy(context) else classical_is_array(value, context)
    if not numeric or classical_dtype_str(value, context) != dtype:
        raise Error("mojolearn: model field '" + name + "' has dtype " + _archive_dtype_name(value, context)
                    + ", the format stores " + dtype + "; refusing to cast a model file")
    return value


def _archive_string(arrays: PythonObject, name: String, context: PythonObject) raises -> PythonObject:
    if not py_contains(arrays, name):
        raise Error(_archive_missing(name))
    var value = arrays[name]
    if py_type(value) is context["array_class"]:
        # A NumPy-free read: a text field never decodes to numbers.
        raise Error("model string field " + name + " is not text")
    if classical_numpy(context) and py_type(value) is context["ndarray"]:
        if Int(py=value.size) == 0:
            raise Error("model string field is empty")
        value = value.flat[0]
    while py_type(value) is py_type(Python.list()):  # driver-loop: over the nesting depth of a stored string field
        if len(value) == 0:
            raise Error("model string field is empty")
        value = value[0]
    return py_call(py_type(PythonObject("")), value)


def classical_archive_load(cls: PythonObject, path: PythonObject, ridge: Bool, context: PythonObject) raises -> PythonObject:
    var admitted = _archive_read_input(path, context)
    var minor = Int(py=context["archive_python_minor"])
    var micro = Int(py=context["archive_python_micro"])
    try:
        var members = native_archive_read_status(admitted.span, minor, micro)
        archive_events_pack(members, context)
        var ancestry = archive_class_names_unpack(cls)
        var print_mode = archive_print_mode_unpack(members, context)
        var plan = native_linear_archive_plan(members, ancestry, ridge, classical_numpy(context), print_mode)
        if plan.class_mismatch:
            raise Error("mojolearn: " + _archive_read_repr(path, context) + " was saved by "
                        + plan.saved_as + ", not " + ancestry[0])
        return archive_linear_pack(cls, plan, members, ridge, context)
    finally:
        admitted.close()


def classical_host_model(path: PythonObject, context: PythonObject) raises -> PythonObject:
    """Load an explicit host linear model from one native archive execution."""
    var admitted = _archive_read_input(path, context)
    var minor = Int(py=context["archive_python_minor"])
    var micro = Int(py=context["archive_python_micro"])
    try:
        var members = native_archive_read_status(admitted.span, minor, micro)
        archive_events_pack(members, context)
        var print_mode = archive_print_mode_unpack(members, context)
        var numpy = classical_numpy(context)
        var saved_as = native_archive_string(members, "estimator", numpy, print_mode)
        var cls = py_get(context["classical_host_estimators"], PythonObject(saved_as))
        if cls is None:
            raise Error("mojolearn: " + _archive_read_repr(path, context) + " was saved by "
                        + saved_as + ", which mojolearn-linear-1 does not hold")
        var ridge = Bool(py=context["classical_kinds"][cls])
        var ancestry = archive_class_names_unpack(cls)
        var plan = native_linear_archive_plan(members, ancestry, ridge, numpy, print_mode)
        if plan.class_mismatch:
            raise Error("mojolearn: " + _archive_read_repr(path, context) + " was saved by "
                        + plan.saved_as + ", not " + ancestry[0])
        var model = archive_linear_pack(cls, plan, members, ridge, context)
        py_setitem(_archive_model_fields(model), "estimator", PythonObject(saved_as))
        return model
    finally:
        admitted.close()
