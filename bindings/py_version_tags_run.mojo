# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
# RUN part of bindings/py_version_tags.mojo: typed Mojo only (no Python
# imports). The layout offsets, the process-wide mode slot and the word reads;
# the probes that prove the layout stay in py_version_tags.mojo.
from core.shared_slot import SharedSlot

comptime _DICT_USED = 16
comptime _DICT_VERSION = 24
# lane fb2-smallcall (2026-11-17): PyTypeObject on 3.10 / 3.11 (LP64, release
# build): the 24-byte PyVarObject head, then 18 eight-byte members (tp_name ..
# tp_as_buffer) put `tp_flags` at +168 (was 176, which is `tp_doc`), and
# `tp_version_tag` (unsigned int, after tp_del) at +384 (was 392, which is
# `tp_finalize`). With the old words the type proof never passed, so on the
# board box's CPython 3.11 every `class_record` missed and every door re-ran
# its full class-layout walk on every call (the fixed +0.1 to 0.3 ms per small
# GPU call on build d95f805cce vs 0.8.55). Checked on CPython 3.10.21 and
# 3.11.16 with ctypes: PyType_GetFlags equals the word at +168 only, and an
# attribute assignment then a lookup moves the u32 at +384 only.
comptime _TYPE_FLAGS = 168
comptime _TYPE_VERSION = 384
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


def dict_version_tag(address: Int) -> Int:
    """The `ma_version_tag` of the live exact dictionary at `address` (call
    only while `dict_version_tags()` is True and the caller holds it)."""
    return Int(_word64(address + _DICT_VERSION))
