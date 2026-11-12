# SPDX-License-Identifier: Apache-2.0
"""RUN of the x_prep owner (bindings/x_prep_native_owner.mojo): the R3
transport switch read. No Python import (tools/python_free_dirs.sh). Lane
rs-shared-infra-b (2026-11-12): moved word for word. The owner's output-range
merge (`_xprep_merge`) was a copy of `_merge_pairs` kept so the owner did not
import the arena runner; it now calls `_merge_pairs` in the Python-free
bindings/arena_prog_run.mojo."""
from std.os import getenv


def xprep_r3(name: String) -> Bool:
    # These three existing R3 choices default on; OFF wins over ON.
    # p3-array-arena: the switch stays a live read (the transport tests and
    # A/B jobs flip it inside one process), but the usual unset case returns
    # before any parsing or allocation.
    var raw = String(getenv("MOJOLEARN_XPREP_R3_OFF",""))
    if raw.byte_length() == 0:
        return True
    var names = raw.replace("+",",").split(",")
    for i in range(len(names)):  # driver-loop: configured transport-switch names
        if String(names[i]).strip() == name or String(names[i]).strip() == "all":
            return False
    return True
