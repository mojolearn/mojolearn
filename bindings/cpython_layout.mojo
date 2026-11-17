# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""THE table of CPython object words Mojolearn reads or writes by offset
(lane fb5-cpython-layout, 2026-11-17).

WHY. bindings/py_version_tags_run.mojo read PyTypeObject.tp_flags and
tp_version_tag at +176/+392 on CPython 3.10/3.11 (real +168/+384), so the
class stamp never held and every public call re-walked its class (fixed in
15cca53af1). Any word read inside a CPython object by offset is outside the
public C API for a struct that is not in the stable ABI; this module is the
ONE place such an offset is written, and the ONE place that says on which
interpreter versions it may be used.

THE RULE. A Mojo read or write of a CPython object word by offset:
  * takes its offset from a `comptime` below (each carries its reference),
  * runs only when `cpython_layout_groups()` (bindings/py_version_tags.mojo:
    the running interpreter's row of the version table, resolved once per
    process from `Py_GetVersion`) holds its group, AND
    its own once-per-process proof on objects made for the probe passed
    (py_version_tags.mojo for the class and dictionary words,
    native_array_storage_door.mojo for the array.array words). An
    interpreter the table does not list (3.15+, a free-threaded build) reads
    no word: the caller takes its public C-API path, as before. A layout that
    fails the proof (a debug `Py_TRACE_REFS` build) reads no word either.
  * is checked by `tools/cpython_layout_check.py` (ctypes on the running
    interpreter, and on every installed 3.10-3.14 from
    `tools/tests/test_cpython_layout.py`): each offset below against the live
    object for every version row that lists its group. That test parses this
    file: keep one `comptime NAME = <int>` per offset and, per version, one
    `if minor == N:` line followed by its `return <groups>` line.
Public layouts (Py_buffer's ten words, core/boundary_layout.mojo;
PyObject.ob_type at +8 read by the standard library's `Py_TYPE`) are part of
the stable ABI and are checked by the same tool, not gated here.

All offsets are for the LP64 release (GIL) build, the only builds the wheels
target (python/pyproject.toml: CPython 3.10-3.14, no free-threaded build).
"""
from core.shared_slot import SharedSlot

# ---------------------------------------------------------------- offsets
#: PyDictObject.ma_used (Include/cpython/dictobject.h): the first member
#: after PyObject_HEAD (ob_refcnt, ob_type). Checked 3.10.21, 3.11.16.
comptime DICT_MA_USED = 16
#: PyDictObject.ma_version_tag (PEP 509, uint64_t after ma_used). Bumped on
#: every mutation on 3.10/3.11; deprecated 3.12 (watcher bits share it),
#: replaced by `_ma_watcher_tag` in 3.14. Checked 3.10.21, 3.11.16.
comptime DICT_MA_VERSION_TAG = 24
#: PyTypeObject.tp_flags (Include/cpython/object.h): PyVarObject head (24)
#: then 18 eight-byte members tp_name .. tp_as_buffer. Equals
#: `PyType_GetFlags` and `type.__flags__`. Checked 3.10.21 .. 3.14.6.
comptime TYPE_TP_FLAGS = 168
#: PyTypeObject.tp_version_tag (unsigned int after tp_del): tp_flags + 8
#: then 26 eight-byte members tp_doc .. tp_del. Checked 3.10.21, 3.11.16.
comptime TYPE_TP_VERSION_TAG = 384
#: Py_TPFLAGS_VALID_VERSION_TAG (Include/object.h, 3.10 and 3.11).
comptime TYPE_TPFLAGS_VALID_VERSION_TAG = UInt64(1 << 19)
#: array.array `arrayobject` (Modules/arraymodule.c, not a public header):
#: PyObject_VAR_HEAD, so ob_size at +16, then `char *ob_item` at +24 and
#: `Py_ssize_t allocated` at +32. Checked 3.10.21 .. 3.14.6.
comptime ARRAY_OB_SIZE = 16
comptime ARRAY_OB_ITEM = 24
comptime ARRAY_ALLOCATED = 32

# ---------------------------------------------------------------- groups
#: DICT_MA_USED and DICT_MA_VERSION_TAG are the dictionary's change stamp.
comptime LAYOUT_DICT_VERSION = 1
#: TYPE_TP_FLAGS, TYPE_TP_VERSION_TAG and the valid-tag flag are the class
#: change stamp.
comptime LAYOUT_TYPE_VERSION = 2
#: ARRAY_OB_SIZE, ARRAY_OB_ITEM and ARRAY_ALLOCATED: the item block of an
#: array.array that native_array_unfilled installs.
comptime LAYOUT_ARRAY_BLOCK = 4


def cpython_layout_row(minor: Int, free_threaded: Bool) -> Int:
    """The version table: the word groups this binary may read on CPython
    3.<minor> (release GIL build), 0 for a version the table does not list.
    The class and dictionary stamps are read on 3.10 / 3.11 only (3.12 and
    later have type and dictionary watchers, the callers' first choice)."""
    if free_threaded:
        return 0  # a different PyObject header (ob_tid, ob_ref_local, ob_ref_shared): no row
    if minor == 10:
        return LAYOUT_DICT_VERSION | LAYOUT_TYPE_VERSION | LAYOUT_ARRAY_BLOCK
    if minor == 11:
        return LAYOUT_DICT_VERSION | LAYOUT_TYPE_VERSION | LAYOUT_ARRAY_BLOCK
    if minor == 12:
        return LAYOUT_ARRAY_BLOCK
    if minor == 13:
        return LAYOUT_ARRAY_BLOCK
    if minor == 14:
        return LAYOUT_ARRAY_BLOCK
    return 0


struct CPythonLayoutMode(Defaultable, Movable):
    # -2 not resolved yet; otherwise the running interpreter's row
    var groups: Int
    # array.array block proof (native_array_storage_door): -2 not probed, 0 off, 1 on
    var arrays: Int
    # dictionary / class stamp proofs (py_version_tags): -2 not probed, 0 off, 1 on
    var dicts: Int
    var types: Int

    def __init__(out self):
        self.groups = -2
        self.arrays = -2
        self.dicts = -2
        self.types = -2


# Shared by every Mojo extension in the process: bump the name's version
# whenever CPythonLayoutMode's layout changes (core/shared_slot.mojo rule).
comptime CPYTHON_LAYOUT_MODE = SharedSlot[CPythonLayoutMode, "MojolearnCPythonLayoutModeV1"]


def cpython_layout_row_of_version(version: String) -> Int:
    """The version table's row for `Py_GetVersion()` text ("3.11.16 (main,
    ...) [Clang ...]"); a free-threaded build says "free-threading build" in
    the same text (3.13: "experimental free-threading build")."""
    if not version.startswith("3."):
        return 0
    var text = version.as_bytes()
    var minor = 0
    var digits = 0
    var i = 2
    while i < len(text) and digits < 3:  # small-loop(version text): the minor number's digits, two at most
        var b = Int(text[i])
        if b < 48 or b > 57:
            break
        minor = minor * 10 + (b - 48)
        digits += 1
        i += 1
    if digits == 0 or i >= len(text) or Int(text[i]) != 46:  # "3.<minor>." or no row
        return 0
    return cpython_layout_row(minor, "free-threading" in version)


@always_inline
def layout_word64(address: Int) -> UInt64:
    return MutPointer[UInt64, MutUntrackedOrigin](unsafe_from_address=address)[0]


@always_inline
def layout_word32(address: Int) -> UInt32:
    return MutPointer[UInt32, MutUntrackedOrigin](unsafe_from_address=address)[0]
