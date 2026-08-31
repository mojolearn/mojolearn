# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Runs `mojo_only/vendor_correctness_check.mojo` and PRINTS the table.

The correctness column in `VENDOR_LIBRARIES.md` is a transcript of this
program's output. Regenerate it here rather than trusting the file:

    tools/with_build_lock.sh pixi run \
      --manifest-path ../mojotrees/pixi.toml \
      mojo build -I . vendor_main.mojo -o /tmp/vendor_probe
    /tmp/vendor_probe

Exit status is 1 if any primitive WIRED into this tree tested WRONG, so this
is usable as a gate. A WRONG verdict on an UNWIRED primitive exits 0: nothing
is broken today, and that row is there so nothing substitutes it tomorrow.

    /tmp/vendor_probe --transpose

runs the one probe that is not in the table, because `linalg.transpose` on
device buffers does not raise, it ABORTS the process, and a probe of it
inside the main run would take the table down with it.
"""

from std.sys import argv, exit

from mojo_only.vendor_correctness_check import (
    check_transpose_aborts,
    run_vendor_correctness,
)


def main() raises:
    var args = argv()
    for i in range(len(args)):
        if args[i] == "--transpose":
            check_transpose_aborts()
            return
    var failures = run_vendor_correctness()
    if failures != 0:
        exit(1)
