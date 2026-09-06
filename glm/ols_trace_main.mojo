# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Emit the OLS identity certificate. DEVIATION 527.

    MOJOLEARN_IDENTITY_TRACE=/tmp/mac.card \
        pixi run mojo run -I . glm/ols_trace_main.mojo

    # the other machine, same commit, same mode
    MOJOLEARN_IDENTITY_TRACE=/tmp/amd.card \
        pixi run mojo run -I . glm/ols_trace_main.mojo

    python3 tools/identity_trace_diff.py /tmp/mac.card /tmp/amd.card

Eleven records, one fit, `core/identity_trace.mojo`'s v1 format. See
`glm/checks/ols_trace.mojo` for the stage list and for why the fixture is
built with no floating-point operation in it.

THE RUN-TO-RUN CONTROL IS NOT OPTIONAL AND IS NOT SEPARATE. Set

    MOJOLEARN_OLS_CARD_CONTROL=/tmp/mac.control.card

and this driver fits the SAME fixture a second time into that file and
compares the two IN PROCESS. A card that does not match its own control on
one machine cannot be compared against another machine's: the difference you
would then be reading is this device's run-to-run noise, which under
`NUMERIC_FAST` is a real and expected thing (float atomics have no ordering
guarantee -- `numerics.mojo` says so at the top). Under `NUMERIC_IDENTICAL`
a control mismatch is a FAILURE and this driver exits non-zero on it.

THE MODE IS PRINTED BY THE BINARY AND WRITTEN INTO THE CARD'S HEADER.
Four sessions share this checkout and `GLOBAL_NUMERIC_MODE` is a line in a
shared file, so a run that compiled inside another session's flip window
would otherwise carry the wrong label. `tools/with_identical_mode.sh` takes
the build lock; this is the second, independent witness, and it is IN THE
ARTIFACT rather than only on the terminal, because the artifact is what gets
shipped to the other machine.
"""

from max.gpu.host import DeviceContext
from std.os import getenv
from std.sys import exit

from core.identity_trace import IdentityTrace, first_divergence
from glm.checks.ols_trace import (
    OLS_CARD_COLS,
    OLS_CARD_ROWS,
    emit_ols_card,
    ols_card_mode_name,
)
from glm.impl.linalg.detail.lstsq import OLS_ELEM_TPB
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL


comptime CARD_ENV = "MOJOLEARN_IDENTITY_TRACE"
comptime CARD_DUMP_ENV = "MOJOLEARN_IDENTITY_TRACE_DUMP"
comptime CARD_CONTROL_ENV = "MOJOLEARN_OLS_CARD_CONTROL"


def _header(mut t: IdentityTrace, what: String) raises:
    t.header(
        "mojolearn OLS identity card (DEVIATION 527) -- " + what
    )
    t.header("numeric mode: " + ols_card_mode_name())
    t.header(
        "shape: n_rows="
        + String(OLS_CARD_ROWS)
        + " n_cols="
        + String(OLS_CARD_COLS)
        + " elem_tpb="
        + String(OLS_ELEM_TPB)
    )
    t.header(
        "fixture: splitmix64 bit-assembled f32, no host float arithmetic"
    )


def main() raises:
    var path = String(getenv(CARD_ENV))
    var dump = String(getenv(CARD_DUMP_ENV))
    var control = String(getenv(CARD_CONTROL_ENV))

    print("== glm/ols_trace_main.mojo [" + ols_card_mode_name() + "] ==")
    if path == "":
        print(
            "  " + CARD_ENV + " is unset, so there is nowhere to write a"
            " card. Set it to a path and re-run; see this file's docstring."
        )
        exit(2)

    # TRUNCATE. The env constructor deliberately APPENDS (two fits in one
    # process share one file in the GBDT harness), and a card re-run against
    # the path it used last time would then be its own previous run
    # concatenated with this one -- which the differ reads as a structural
    # divergence with no cause. `to_path` is the truncating constructor and
    # is what a one-fit-per-file card wants.
    var t = IdentityTrace.to_path(path, dump, True)
    _header(t, "primary")
    var coef = emit_ols_card(DeviceContext(), t)
    print("  wrote " + String(t.seq) + " records to " + path)
    print("  coef[0] = " + String(coef[0]) + " (decimal, NOT the record --")
    print("            the records are FNV-1a64 over raw bytes)")

    if control == "":
        print(
            "  no " + CARD_CONTROL_ENV + ": the run-to-run control was NOT"
            " taken. A card without one cannot be compared across machines"
            " -- see this file's docstring."
        )
        return

    var c = IdentityTrace.to_path(control, dump, True)
    _header(c, "run-to-run control, same fixture, same process")
    var coef2 = emit_ols_card(DeviceContext(), c)
    _ = len(coef2)
    print("  wrote " + String(c.seq) + " control records to " + control)

    var diff = first_divergence(path, control)
    comptime identical = GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL
    if diff == "":
        print(
            "  control MATCHES: all "
            + String(t.seq)
            + " stages bit-identical across two fits of the same fixture"
            " in one process ["
            + ols_card_mode_name()
            + "]"
        )
        return
    comptime if identical:
        raise Error(
            "ols_trace_main: the card does not match its own run-to-run"
            " control under NUMERIC_IDENTICAL, which is a failure of this"
            " machine before any second vendor is involved. First"
            " divergence:\n  "
            + diff
        )
    print(
        "  control DIFFERS [FAST]: this is the measurement, not a bug --"
        " the shipped mode makes no run-to-run promise. First divergence:"
    )
    print("    " + diff)
