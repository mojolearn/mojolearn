# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Change stamps for dictionaries and classes on CPython 3.10 and 3.11.

Lane ne2-i-j (2026-11-07; plan NATIVE_ENTRY_POINTS_AFTER_0866 (git history)
section 2I). The "check once" records (native_class_admission
`class_record` / `dict_keys_admitted`, the module-connection stamp and the
model generation word of py_model_records, all through
native_loaded_module `shared_dict_watch` / `shared_dict_epoch`) key on
CPython 3.12's type and dictionary watchers. The wheels declare
`python_requires >= 3.10`, so 3.10 and 3.11 ran the full check on every call.
Those two versions carry the stamps the watchers replaced, in the objects
themselves:

- Dictionaries (PEP 509): `PyDictObject.ma_version_tag`, a 64-bit word set
  from one process-wide counter at creation and at every insertion,
  replacement, deletion and clear. Equal tags mean the same dictionary in the
  same state; a new dictionary at a recycled address has a new tag. The
  callers' epoch is that word.
- Classes: `PyTypeObject.tp_version_tag` with `Py_TPFLAGS_VALID_VERSION_TAG`.
  A class is given a tag (from one counter, never reused) only when every
  base holds one; `PyType_Modified` (any class attribute assignment or
  deletion, `__bases__` assignment) clears the flag on the class and, through
  `tp_subclasses`, on every subclass that holds one. So "flag set and tag
  equal to the recorded tag" means neither the class nor an ancestor changed:
  the guarantee the 3.12 type watcher gives.

LAYOUT. The offsets and the versions they are read on are
bindings/cpython_layout.mojo (THE table, checked by ctypes in
tools/tests/test_cpython_layout.py); the words are read only when the running
interpreter's row lists their group (3.10 / 3.11 release layout, LP64): `ma_used` +16, `ma_version_tag` +24; `tp_flags` +168,
`tp_version_tag` +384. Each offset is PROVEN once per process before use, on
objects made for the probe: a new dictionary's `ma_used` is 0, becomes 1 on
one insertion (and `PyDict_Size` agrees), and its tag moves; a new class's
flags at +168 equal `PyType_GetFlags`, a lookup gives it the valid flag and a
nonzero tag, an attribute assignment clears the flag, and the next lookup
gives a different tag. Any mismatch (a debug `Py_TRACE_REFS` build, another
layout) leaves the mode off and every caller on its full check, as before.
On 3.12 and later the mode is off: the watchers serve.

The caller holds the object (a live dictionary or class) whenever it reads a
stamp here: no word is read through a stale address. Nothing here runs Python
code of the caller's objects except one attribute lookup of `__init__` on a
class that passed its full admission (`type_version_tag_assign`), which
CPython resolves in C for functions and slot wrappers.
"""
from core.native_string import unsafe_string_from_utf8_cstr
from std.python import Python, PythonObject
from std.python._cpython import PyObjectPtr
from bindings.py_cpython_table import cpython_table
from bindings.py_attr import py_attr
from bindings.cpython_layout import (
    DICT_MA_USED, DICT_MA_VERSION_TAG, TYPE_TP_FLAGS, TYPE_TP_VERSION_TAG, TYPE_TPFLAGS_VALID_VERSION_TAG,
    LAYOUT_DICT_VERSION, LAYOUT_TYPE_VERSION, CPYTHON_LAYOUT_MODE, cpython_layout_row_of_version,
    layout_word64, layout_word32,
)


def cpython_layout_groups() raises -> Int:
    """The running interpreter's row of the version table
    (bindings/cpython_layout.mojo), resolved once per process from one
    `Py_GetVersion`: the word groups that may be read by offset here. A
    version the table does not list reads no word."""
    var g = CPYTHON_LAYOUT_MODE.get_or_create_ptr()
    if g[].groups != -2:
        return g[].groups
    var row = cpython_layout_row_of_version(native_python_version())
    g = CPYTHON_LAYOUT_MODE.get_or_create_ptr()
    g[].groups = row
    return row


def native_python_version() raises -> String:
    """`Py_GetVersion()`: the running interpreter's version text."""
    ref cpy = Python().cpython()
    var version_ptr = cpy.lib.get_function[Pointer[Int8, MutUntrackedOrigin]]("Py_GetVersion")()
    return unsafe_string_from_utf8_cstr(version_ptr)


def dict_version_tags() raises -> Bool:
    """True on CPython 3.10 / 3.11 once the dictionary layout is proven
    (module docstring): `dict_version_tag` is then the dictionary's stamp."""
    var g = CPYTHON_LAYOUT_MODE.get_or_create_ptr()
    if g[].dicts != -2:
        return g[].dicts == 1
    g[].dicts = 0
    if (cpython_layout_groups() & LAYOUT_DICT_VERSION) == 0:
        return False
    var t = cpython_table()
    var probe = Python.dict()
    var later = Python.dict()
    var a = Int(probe._obj_ptr)
    var used0 = layout_word64(a + DICT_MA_USED)
    var tag0 = layout_word64(a + DICT_MA_VERSION_TAG)
    var tag_later = layout_word64(Int(later._obj_ptr) + DICT_MA_VERSION_TAG)
    probe["mojolearn_probe"] = PythonObject(1)
    var used1 = layout_word64(a + DICT_MA_USED)
    var tag1 = layout_word64(a + DICT_MA_VERSION_TAG)
    var proven = (used0 == 0 and used1 == 1 and t[].PyDict_Size(probe._obj_ptr) == 1
                  and tag0 != 0 and tag_later != tag0 and tag1 != tag0 and tag1 != tag_later)
    _ = probe
    _ = later
    if proven:
        g = CPYTHON_LAYOUT_MODE.get_or_create_ptr()
        g[].dicts = 1
    return proven


def type_version_tags() raises -> Bool:
    """True on CPython 3.10 / 3.11 once the class layout is proven (module
    docstring): `type_version_tag` is then the class's stamp."""
    var g = CPYTHON_LAYOUT_MODE.get_or_create_ptr()
    if g[].types != -2:
        return g[].types == 1
    g[].types = 0
    if (cpython_layout_groups() & LAYOUT_TYPE_VERSION) == 0:
        return False
    var t = cpython_table()
    ref cpy = Python().cpython()
    # `type`, reached as the class of the dict class (no module import)
    var probe_dict = Python.dict()
    var dict_type = PythonObject(from_owned=cpy.PyObject_Type(probe_dict._obj_ptr))
    var type_type = PythonObject(from_owned=cpy.PyObject_Type(dict_type._obj_ptr))
    var cls = type_type(PythonObject("_MojolearnVersionTagProbe"), Python.tuple(), Python.dict())
    var a = Int(cls._obj_ptr)
    var proven = layout_word64(a + TYPE_TP_FLAGS) == t[].PyType_GetFlags(cls._obj_ptr)
    var tag0 = 0
    if proven:
        _ = py_attr(cls, "__init__")
        proven = (layout_word64(a + TYPE_TP_FLAGS) & TYPE_TPFLAGS_VALID_VERSION_TAG) != 0
        tag0 = Int(layout_word32(a + TYPE_TP_VERSION_TAG))
        proven = proven and tag0 != 0
    if proven:
        var one = PythonObject(1)
        if cpy.PyObject_SetAttrString(cls._obj_ptr, String("mojolearn_probe"), one._obj_ptr) == -1:
            raise cpy.unsafe_get_error()
        proven = (layout_word64(a + TYPE_TP_FLAGS) & TYPE_TPFLAGS_VALID_VERSION_TAG) == 0
        proven = proven and layout_word64(a + TYPE_TP_FLAGS) == t[].PyType_GetFlags(cls._obj_ptr)
    if proven:
        _ = py_attr(cls, "__init__")
        proven = (layout_word64(a + TYPE_TP_FLAGS) & TYPE_TPFLAGS_VALID_VERSION_TAG) != 0
        var tag1 = Int(layout_word32(a + TYPE_TP_VERSION_TAG))
        proven = proven and tag1 != 0 and tag1 != tag0
    _ = cls
    _ = probe_dict
    if proven:
        g = CPYTHON_LAYOUT_MODE.get_or_create_ptr()
        g[].types = 1
    return proven


def type_version_tag(cls: PythonObject) -> Int:
    """The live class's `tp_version_tag` when it holds a valid one, else 0
    (call only while `type_version_tags()` is True)."""
    var a = Int(cls._obj_ptr)
    if (layout_word64(a + TYPE_TP_FLAGS) & TYPE_TPFLAGS_VALID_VERSION_TAG) == 0:
        return 0
    return Int(layout_word32(a + TYPE_TP_VERSION_TAG))


def type_version_tag_assign(cls: PythonObject) raises -> Int:
    """`type_version_tag(cls)` after one lookup that makes CPython assign the
    class (and its bases) a tag when it has none: 0 when it still has none
    (the tag counter is exhausted), and the caller admits no record."""
    var tag = type_version_tag(cls)
    if tag != 0:
        return tag
    # The lookup is CPython's own MRO cache lookup (`_PyType_Lookup`, exported
    # by 3.10 / 3.11, the only versions that reach here): it assigns the tag
    # and returns a borrowed entry, sets no error and runs no descriptor. An
    # attribute read (`getattr(cls, "__init__")`) went through the class's
    # METACLASS `__getattribute__`, so a value whose class has a Python
    # metaclass ran a Python frame inside get_params on 3.10
    # (test_leaf_metaclass_hooks_are_not_needed_to_read_native_storage).
    var name = PythonObject("__init__")
    ref cpy = Python().cpython()
    _ = cpy.lib.get_function[PyObjectPtr]("_PyType_Lookup")(cls._obj_ptr, name._obj_ptr)
    _ = name
    return type_version_tag(cls)
