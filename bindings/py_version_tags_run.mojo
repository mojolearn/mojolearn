# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
# RUN part of bindings/py_version_tags.mojo: typed Mojo only (no Python
# imports). The word reads of the dictionary stamp; the offsets, the version
# rows and the process-wide mode slot are bindings/cpython_layout.mojo (THE
# table, lane fb5-cpython-layout, 2026-11-17), and the probes that prove the
# layout stay in py_version_tags.mojo.
#
# History: lane fb2-smallcall (2026-11-17) found tp_flags / tp_version_tag
# read at +176 / +392 on 3.10 / 3.11 (real +168 / +384, 15cca53af1); the
# offsets now live in the table with their references and a ctypes test
# (tools/tests/test_cpython_layout.py).
from bindings.cpython_layout import DICT_MA_VERSION_TAG, layout_word64


def dict_version_tag(address: Int) -> Int:
    """The `ma_version_tag` of the live exact dictionary at `address` (call
    only while `dict_version_tags()` is True and the caller holds it)."""
    return Int(layout_word64(address + DICT_MA_VERSION_TAG))
