# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""RUN half of `bindings/msel_host.mojo` (wave 8 run split): the parallel
forest's host offset rebase, called by the core host image's parallel table.
Typed Mojo only: no Python import."""
from core.msel_host_impl import msel_rebase_check_parts
from core.msel_host import host_rebase_offsets_i32


def msel_rebase_offsets_i32_host_run(ra: Int, nr: Int, la: Int, p: Int, no: Int, d: Int) raises:
    """The parallel forest's merged int32 tree offsets, on the host. The
    core host image's parallel table entry (bindings/parallel_base_entry_host.mojo)
    calls it; the Python door `msel_rebase_offsets_i32` had no caller left
    (lane calltable-parallel-archive, 2026-11-11)."""
    if p < 1 or nr < 1 or no < 1 or ra == 0 or la == 0 or d == 0:
        raise Error("msel_rebase_offsets_i32: bad sizes")
    msel_rebase_check_parts(la, p, nr, no)
    if not host_rebase_offsets_i32(ra, nr, la, p, no, d):
        raise Error("msel_rebase_offsets_i32: a merged offset leaves int32")
