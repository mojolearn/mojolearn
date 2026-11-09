# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Change stamps for dictionaries and classes on CPython 3.10 and 3.11.

Lane ne2-i-j (2026-11-07; docs/plans/NATIVE_ENTRY_POINTS_AFTER_0866.md
section 2I). The "check once" records (native_class_admission
`class_record` / `dict_keys_admitted`, the module-connection stamp and the
model generation word of native_model_records, all through
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

LAYOUT. The words are read at their offsets in the 3.10 / 3.11 release
layout (LP64): `ma_used` +16, `ma_version_tag` +24; `tp_flags` +176,
`tp_version_tag` +392. Each offset is PROVEN once per process before use, on
objects made for the probe: a new dictionary's `ma_used` is 0, becomes 1 on
one insertion (and `PyDict_Size` agrees), and its tag moves; a new class's
flags at +176 equal `PyType_GetFlags`, a lookup gives it the valid flag and a
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
from std.python import Python, PythonObject
from std.python._cpython import PyObjectPtr
from bindings.cpython_table import cpython_table
from bindings.py_attr import py_attr
from core.shared_slot import SharedSlot

comptime _DICT_USED = 16
comptime _DICT_VERSION = 24
comptime _TYPE_FLAGS = 176
comptime _TYPE_VERSION = 392
#: Py_TPFLAGS_VALID_VERSION_TAG (Include/object.h, 3.10 and 3.11)
comptime _TPFLAGS_VALID_VERSION_TAG = UInt64(1 << 19)


struct VersionTagMode(Defaultable, Movable):
    # -2 not probed yet, 0 off, 1 on (proven on this interpreter)
    var dicts: Int
    var types: Int

    def __init__(out self):
        self.dicts = -2
        self.types = -2


# Shared by every Mojo extension in the process: bump the name's version
# whenever VersionTagMode's layout changes.
comptime VERSION_TAG_MODE = SharedSlot[VersionTagMode, "MojolearnVersionTagModeV2"]


@always_inline
def _word64(address: Int) -> UInt64:
    return MutPointer[UInt64, MutUntrackedOrigin](unsafe_from_address=address)[0]


@always_inline
def _word32(address: Int) -> UInt32:
    return MutPointer[UInt32, MutUntrackedOrigin](unsafe_from_address=address)[0]


def _python_310_or_311() raises -> Bool:
    var version = native_python_version()
    return version.startswith("3.10.") or version.startswith("3.11.")


def native_python_version() raises -> String:
    """`Py_GetVersion()`: the running interpreter's version text."""
    ref cpy = Python().cpython()
    var version_ptr = cpy.lib.get_function[Pointer[Int8, MutUntrackedOrigin]]("Py_GetVersion")()
    return String(unsafe_from_utf8_ptr=version_ptr)


def dict_version_tags() raises -> Bool:
    """True on CPython 3.10 / 3.11 once the dictionary layout is proven
    (module docstring): `dict_version_tag` is then the dictionary's stamp."""
    var g = VERSION_TAG_MODE.get_or_create_ptr()
    if g[].dicts != -2:
        return g[].dicts == 1
    g[].dicts = 0
    if not _python_310_or_311():
        return False
    var t = cpython_table()
    var probe = Python.dict()
    var later = Python.dict()
    var a = Int(probe._obj_ptr)
    var used0 = _word64(a + _DICT_USED)
    var tag0 = _word64(a + _DICT_VERSION)
    var tag_later = _word64(Int(later._obj_ptr) + _DICT_VERSION)
    probe["mojolearn_probe"] = PythonObject(1)
    var used1 = _word64(a + _DICT_USED)
    var tag1 = _word64(a + _DICT_VERSION)
    var proven = (used0 == 0 and used1 == 1 and t[].PyDict_Size(probe._obj_ptr) == 1
                  and tag0 != 0 and tag_later != tag0 and tag1 != tag0 and tag1 != tag_later)
    _ = probe
    _ = later
    if proven:
        g = VERSION_TAG_MODE.get_or_create_ptr()
        g[].dicts = 1
    return proven


def dict_version_tag(address: Int) -> Int:
    """The `ma_version_tag` of the live exact dictionary at `address` (call
    only while `dict_version_tags()` is True and the caller holds it)."""
    return Int(_word64(address + _DICT_VERSION))


def type_version_tags() raises -> Bool:
    """True on CPython 3.10 / 3.11 once the class layout is proven (module
    docstring): `type_version_tag` is then the class's stamp."""
    var g = VERSION_TAG_MODE.get_or_create_ptr()
    if g[].types != -2:
        return g[].types == 1
    g[].types = 0
    if not _python_310_or_311():
        return False
    var t = cpython_table()
    ref cpy = Python().cpython()
    # `type`, reached as the class of the dict class (no module import)
    var probe_dict = Python.dict()
    var dict_type = PythonObject(from_owned=cpy.PyObject_Type(probe_dict._obj_ptr))
    var type_type = PythonObject(from_owned=cpy.PyObject_Type(dict_type._obj_ptr))
    var cls = type_type(PythonObject("_MojolearnVersionTagProbe"), Python.tuple(), Python.dict())
    var a = Int(cls._obj_ptr)
    var proven = _word64(a + _TYPE_FLAGS) == t[].PyType_GetFlags(cls._obj_ptr)
    var tag0 = 0
    if proven:
        _ = py_attr(cls, "__init__")
        proven = (_word64(a + _TYPE_FLAGS) & _TPFLAGS_VALID_VERSION_TAG) != 0
        tag0 = Int(_word32(a + _TYPE_VERSION))
        proven = proven and tag0 != 0
    if proven:
        var one = PythonObject(1)
        if cpy.PyObject_SetAttrString(cls._obj_ptr, String("mojolearn_probe"), one._obj_ptr) == -1:
            raise cpy.unsafe_get_error()
        proven = (_word64(a + _TYPE_FLAGS) & _TPFLAGS_VALID_VERSION_TAG) == 0
        proven = proven and _word64(a + _TYPE_FLAGS) == t[].PyType_GetFlags(cls._obj_ptr)
    if proven:
        _ = py_attr(cls, "__init__")
        proven = (_word64(a + _TYPE_FLAGS) & _TPFLAGS_VALID_VERSION_TAG) != 0
        var tag1 = Int(_word32(a + _TYPE_VERSION))
        proven = proven and tag1 != 0 and tag1 != tag0
    _ = cls
    _ = probe_dict
    if proven:
        g = VERSION_TAG_MODE.get_or_create_ptr()
        g[].types = 1
    return proven


def type_version_tag(cls: PythonObject) -> Int:
    """The live class's `tp_version_tag` when it holds a valid one, else 0
    (call only while `type_version_tags()` is True)."""
    var a = Int(cls._obj_ptr)
    if (_word64(a + _TYPE_FLAGS) & _TPFLAGS_VALID_VERSION_TAG) == 0:
        return 0
    return Int(_word32(a + _TYPE_VERSION))


def type_version_tag_assign(cls: PythonObject) raises -> Int:
    """`type_version_tag(cls)` after one lookup that makes CPython assign the
    class (and its bases) a tag when it has none: 0 when it still has none
    (the tag counter is exhausted), and the caller admits no record."""
    var tag = type_version_tag(cls)
    if tag != 0:
        return tag
    _ = py_attr(cls, "__init__")
    return type_version_tag(cls)
