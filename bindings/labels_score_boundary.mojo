# SPDX-License-Identifier: Apache-2.0
"""Exact scalar admission and output owners for typed classification keys."""
from std.python import Python, PythonObject
from std.python._cpython import ExternalFunction, PyObjectPtr, GILReleased
from core.shared_slot import SharedSlot
from bindings.cpython_table import cpython_table
from bindings.cpython_borrowed import borrowed_item, borrowed_type_word
from bindings.native_loaded_module import native_loaded_module
from bindings.py_balanced import py_type, py_get
from core.labels_score_keys import (
    ScoreLabelKey, ScoreKeyIndex, score_key_join, SCORE_INT, SCORE_BIGINT, SCORE_UTF8, SCORE_UCS4,
)


comptime _ScoreNumberToBase = ExternalFunction[
    "PyNumber_ToBase", def(PyObjectPtr, Int32) thin abi("C") -> PyObjectPtr]
comptime _ScoreUnicodeFromObject = ExternalFunction[
    "PyUnicode_FromObject", def(PyObjectPtr) thin abi("C") -> PyObjectPtr]
comptime _ScoreUnicodeGetLength = ExternalFunction[
    "PyUnicode_GetLength", def(PyObjectPtr) thin abi("C") -> Int]


struct ScoreScalarApi(Defaultable, Movable):
    var number_to_base: _ScoreNumberToBase.type
    var unicode_from_object: _ScoreUnicodeFromObject.type
    var unicode_length: _ScoreUnicodeGetLength.type

    def __init__(out self):
        ref cpy = Python().cpython()
        self.number_to_base = _ScoreNumberToBase.load(cpy.lib.borrow())
        self.unicode_from_object = _ScoreUnicodeFromObject.load(cpy.lib.borrow())
        self.unicode_length = _ScoreUnicodeGetLength.load(cpy.lib.borrow())


comptime SCORE_SCALAR_API = SharedSlot[ScoreScalarApi, "MojolearnScoreScalarApiV1"]


struct ScoreKeyOwner(Movable):
    # `values` contains only admitted output objects; `pins` keeps borrowed
    # immutable key bytes alive. Algorithms read only `keys`.
    var keys: List[ScoreLabelKey]
    var values: List[PythonObject]
    var pins: List[PythonObject]
    var points: List[List[UInt32]]

    def __init__(out self):
        self.keys = List[ScoreLabelKey]()
        self.values = List[PythonObject]()
        self.pins = List[PythonObject]()
        self.points = List[List[UInt32]]()


def score_keys_unpack(values: PythonObject, string_kind: Bool, context: PythonObject) raises -> ScoreKeyOwner:
    # The caller completed shape and exact primitive admission first, so
    # these stable scalar C APIs cannot enter user conversion slots.
    var result = ScoreKeyOwner()
    var b = native_loaded_module("builtins")
    var is_tuple = py_type(values) is b.tuple
    var unicode_error = b.UnicodeEncodeError
    var npbool = py_get(context, "ndarray")
    if npbool is not None:
        npbool = context["numeric_dtypes"][Python.tuple("b", 1)].type
    var api = cpython_table()
    var scalar = SCORE_SCALAR_API.get_or_create_ptr()
    ref cpy = Python().cpython()
    var length = List[Int](length=1, fill=0)
    var lp = length.unsafe_ptr().unsafe_mut_cast[True]().unsafe_origin_cast[MutUntrackedOrigin]()
    var overflow = List[Int32](length=1, fill=0)
    var op = overflow.unsafe_ptr().unsafe_mut_cast[True]().unsafe_origin_cast[MutUntrackedOrigin]()
    # Exact CPython slot reads require the GIL; RUN hashes/sorts/lookups split
    # the admitted native keys across the host worker pool.
    for i in range(len(values)):  # UNPACK exact scalar slots under the GIL
        var raw = borrowed_item(values, i, is_tuple)
        var identity = borrowed_type_word(raw)
        var value = PythonObject(from_borrowed=raw)
        if string_kind:
            if identity != Int(b.str._obj_ptr):
                var normalized = scalar[].unicode_from_object(raw)
                if not normalized:
                    raise cpy.unsafe_get_error()
                value = PythonObject(from_owned=normalized)
            var address = api[].PyUnicode_AsUTF8AndSize(value._obj_ptr, Int(lp))
            if address != 0:
                result.keys.append(ScoreLabelKey(SCORE_UTF8, 0, address, length[0], False))
            else:
                # As in the existing text path, surrogate-containing strings
                # have no UTF-8 cache. Preserve their exact Unicode points.
                if api[].PyErr_ExceptionMatches(unicode_error._obj_ptr) == 0:
                    raise cpy.unsafe_get_error()
                api[].PyErr_Clear()
                var count = scalar[].unicode_length(value._obj_ptr)
                if count < 0:
                    raise cpy.unsafe_get_error()
                var points = List[UInt32](unsafe_uninit_length=count)
                for at in range(count):  # UNPACK one exact str's native code-point storage
                    points[at] = api[].PyUnicode_ReadChar(value._obj_ptr, at)
                var key = ScoreLabelKey(SCORE_UCS4, 0, Int(points.unsafe_ptr()), count, False)
                result.points.append(points^)
                result.keys.append(key)
            result.values.append(value)
            continue
        if identity != Int(b.int._obj_ptr):
            var normalized = raw
            if identity == Int(b.bool._obj_ptr) or (npbool is not None and identity == Int(npbool._obj_ptr)):
                var truth = api[].PyObject_IsTrue(raw)
                if truth < 0:
                    raise cpy.unsafe_get_error()
                normalized = api[].PyLong_FromLongLong(Int64(truth))
            else:
                normalized = api[].PyNumber_Index(raw)
            if not normalized:
                raise cpy.unsafe_get_error()
            value = PythonObject(from_owned=normalized)
        overflow[0] = 0
        var word = api[].PyLong_AsLongLongAndOverflow(value._obj_ptr, Int(op))
        if overflow[0] == 0:
            if word == -1 and cpy.PyErr_Occurred():
                raise cpy.unsafe_get_error()
            result.keys.append(ScoreLabelKey(SCORE_INT, word, 0, 0, False))
        else:
            # Public stable C API, base16 is not subject to the interpreter's
            # decimal digit limit. No private PyLong layout/capsule is used.
            var raw_hex = scalar[].number_to_base(value._obj_ptr, Int32(16))
            if not raw_hex:
                raise cpy.unsafe_get_error()
            var hexadecimal = PythonObject(from_owned=raw_hex)
            var address = api[].PyUnicode_AsUTF8AndSize(raw_hex, Int(lp))
            if address == 0:
                raise cpy.unsafe_get_error()
            var negative = overflow[0] < 0
            var prefix = 3 if negative else 2
            result.keys.append(ScoreLabelKey(SCORE_BIGINT, 0, address + prefix, length[0] - prefix, negative))
            result.pins.append(hexadecimal)
        result.values.append(value)
    _ = length^
    _ = overflow^
    _ = values
    return result^


def score_keys_pack(owner: ScoreKeyOwner, rows: List[Int], identity: Bool = False) raises -> PythonObject:
    # PACK public class objects in the native plan's order. Numeric fast-path
    # keys have no Python class intermediates; only final classes are boxed.
    var api = cpython_table()
    ref cpy = Python().cpython()
    var count = len(owner.keys) if identity else len(rows)
    var raw = api[].PyList_New(count)
    if not raw:
        raise cpy.unsafe_get_error()
    var result = PythonObject(from_owned=raw)
    for i in range(count):
        var row = i if identity else rows[i]
        var item = PythonObject(None)
        if len(owner.values):
            item = owner.values[row]
        else:
            item = PythonObject(owner.keys[row].number)
        if api[].PyList_SetItem(raw, i, cpy.Py_NewRef(item._obj_ptr)) != 0:
            raise cpy.unsafe_get_error()
    _ = owner
    return result


def score_keys_pack_pair(a: ScoreKeyOwner, b: ScoreKeyOwner, rows: List[Int]) raises -> PythonObject:
    """PACK the sorted union; joined keys borrow the two boundary owners."""
    var api = cpython_table()
    ref cpy = Python().cpython()
    var raw = api[].PyList_New(len(rows))
    if not raw:
        raise cpy.unsafe_get_error()
    var result = PythonObject(from_owned=raw)
    var split = len(a.keys)
    for i in range(len(rows)):  # PACK boxed public class labels under the GIL
        var row = rows[i]
        var item = PythonObject(None)
        if row < split:
            item = a.values[row] if len(a.values) else PythonObject(a.keys[row].number)
        else:
            row -= split
            item = b.values[row] if len(b.values) else PythonObject(b.keys[row].number)
        if api[].PyList_SetItem(raw, i, cpy.Py_NewRef(item._obj_ptr)) != 0:
            raise cpy.unsafe_get_error()
    _ = a
    _ = b
    return result


def score_keys_index(keys: List[ScoreLabelKey], keep_last: Bool = False,
                     ordered: Bool = True) raises -> ScoreKeyIndex:
    try:
        with GILReleased(Python()):
            return ScoreKeyIndex(Int(keys.unsafe_ptr()), len(keys), keep_last, ordered)
    finally:
        _ = keys


def score_keys_union(a: ScoreKeyOwner, b: ScoreKeyOwner) raises -> List[ScoreLabelKey]:
    try:
        with GILReleased(Python()):
            return score_key_join(a.keys, b.keys)
    finally:
        _ = a
        _ = b
