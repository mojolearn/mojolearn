# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
# RUN part of bindings/py_version_tags.mojo: typed Mojo only (no Python
# imports). The layout offsets, the process-wide mode slot and the word reads;
# the probes that prove the layout stay in py_version_tags.mojo.
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


def dict_version_tag(address: Int) -> Int:
    """The `ma_version_tag` of the live exact dictionary at `address` (call
    only while `dict_version_tags()` is True and the caller holds it)."""
    return Int(_word64(address + _DICT_VERSION))
