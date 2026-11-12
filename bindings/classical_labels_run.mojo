# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""RUN half of the label door (bindings/classical_labels_door.mojo): the
typed dtype numbering and environment reads the label encoder and decoder
use, and the typed codes source. No Python import
(tools/python_free_dirs.sh). Moved word for word (lane admitted-x-view,
2026-11-12); per-call scalar decisions, nothing to split across cores."""

from std.os import getenv


def labels_env(name: String, default: String) -> String:
    var value = getenv(name)
    return value if value else default


def _native_code(typestr: String) raises -> Int:
    """`_array._NATIVE_CODE`: the core helpers' dtype numbering."""
    if typestr == "<f4":
        return 0
    if typestr == "<f8":
        return 1
    if typestr == "<i4":
        return 2
    if typestr == "<i8":
        return 3
    if typestr == "<u4":
        return 4
    if typestr == "<u1":
        return 5
    if typestr == "<u2":
        return 6
    if typestr == "<i1":
        return 7
    if typestr == "<f2":
        return 8
    raise Error("TypeError: " + typestr + " has no native element code")


def _encode_code(typestr: String) -> Int:
    """`_labels._NATIVE_ENCODE`, or -1."""
    if typestr == "<f4":
        return 0
    if typestr == "<f8":
        return 1
    if typestr == "<i4":
        return 2
    if typestr == "<i8":
        return 3
    if typestr == "<u4":
        return 4
    if typestr == "<u1":
        return 5
    return -1


trait _LabelsCodeSource:
    def address(self) raises -> Int: ...


@fieldwise_init
struct _NativeLabelCodes(Copyable, Movable, _LabelsCodeSource):
    var data: Int

    def address(self) raises -> Int:
        return self.data
