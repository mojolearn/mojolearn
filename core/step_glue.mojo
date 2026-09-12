# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""DEVIATIONS 2645 to 2648: the byte LM step glue arms
(docs/lanes/BRIEF_step_glue_2026-09-11.md).

WHICH BUILDS COMPILE THE GLUE LAUNCHES (DEVIATION 2649 changed this). Until
that deviation every new launch path here was compiled only under
`-D MOJOLEARN_STEP_GLUE_TRIAL=1` (`comptime STEP_GLUE_TRIAL`). It is now
compiled under the trial define OR on a shipped build whose COLUMN DEFAULT
carries the matching bits, which is how the NVIDIA winner reaches the
shipped path: `step_glue_default_arm_for` in checks/kernel_matrix.mojo is
the routing row, `STEP_GLUE_ARM_DEFAULT` is this column's word, and the two
predicates `STEP_GLUE_SHIPPED_UPDATE` and `STEP_GLUE_SHIPPED_ROWS` are what
each caller ORs into its `comptime if`. On a build that is neither,
`step_glue_arm_from_env()` returns `STEP_GLUE_ARM_DEFAULT` (which is
`STEP_GLUE_SHIPPED`, 0, on every column but NVIDIA) without reading the
environment, so that build compiles the shipped launches only, exactly as
before.

`step_glue_sabotage()` is NOT part of that widening. It stays False on
every build without the trial define, so a shipped build never takes the
floor in `step_glue_blocks` and never reads the sabotage variable.

THE ARM NAME. `MOJOLEARN_STEP_GLUE_ARM` is `shipped` (or unset or empty),
or tokens in this order joined by `_`:

    optskip     DEVIATION 2646: the optimizer's four entry scans are not run
                on the byte LM step (brief section 5.2)
    noshadow    DEVIATION 2647: AdamW writes into the shadow buffers and the
                step swaps handles; no shadow copy (brief sections 4.3, 5.3)
    rows16      DEVIATION 2645: the RMSNorm row kernels launch 16, 8 or 4
    rows8       threads per block instead of 128 (brief section 4.1)
    rows4

so 16 names in all (`shipped`, `optskip`, `noshadow`, `optskip_noshadow`,
and each of those with one rows token, the bare rows token standing for
`shipped` plus rows). The parser compares the whole string against that
table, so an unknown token, a repeated token, two rows tokens or tokens out
of order all raise.

`MOJOLEARN_STEP_GLUE_ARM_SABOTAGE=1` (trial builds only) is the REACH
sabotage of brief section 6: the rows launches and the out-of-place update
launch cover floor(count / threads) blocks instead of the ceiling, so the
tail they leave unwritten names the threads per block that reached the
launch. Never set on a timed run.

Nothing here reads or writes a device buffer.
"""

from std.os import getenv
from std.sys.compile import is_defined

# DEVIATION 2649: the shipped arm comes from the kernel matrix's routing row,
# so this file names no column. The matrix imports nothing of ours, so there
# is no cycle (core/ already imports it in eight other files).
from checks.kernel_matrix import (
    STEP_GLUE_DEFAULT_WORD_OPTSKIP_NOSHADOW_ROWS16,
    TARGET_COLUMN,
    column_max_block_size,
    step_glue_default_arm_for,
)

comptime STEP_GLUE_TRIAL = is_defined["MOJOLEARN_STEP_GLUE_TRIAL"]()

comptime STEP_GLUE_SHIPPED = 0
comptime STEP_GLUE_OPTSKIP = 1
comptime STEP_GLUE_NOSHADOW = 2
comptime STEP_GLUE_ROWS16 = 4
comptime STEP_GLUE_ROWS8 = 8
comptime STEP_GLUE_ROWS4 = 16
comptime STEP_GLUE_UPDATE_BITS = 3
"""`optskip | noshadow`: an arm with either bit takes the glue update path."""
comptime STEP_GLUE_ROWS_BITS = 28
comptime STEP_GLUE_ALL_BITS = 31

comptime STEP_GLUE_ARM_OPTSKIP_NOSHADOW_ROWS16 = (
    STEP_GLUE_OPTSKIP | STEP_GLUE_NOSHADOW | STEP_GLUE_ROWS16
)
"""`optskip_noshadow_rows16`: DEVIATIONS 2646, 2647 and 2645 together, the
arm DEVIATION 2649 flipped on NVIDIA (brief sections 2 and 7)."""

comptime STEP_GLUE_ARM_DEFAULT = step_glue_default_arm_for[TARGET_COLUMN]()
"""THE SHIPPED ARM for this column, from the kernel matrix ROUTING row
(DEVIATION 2649). `STEP_GLUE_SHIPPED` (0) on Apple and every column but
NVIDIA, so those builds are unchanged."""

comptime STEP_GLUE_SHIPPED_UPDATE = (
    (not STEP_GLUE_TRIAL) and (STEP_GLUE_ARM_DEFAULT & STEP_GLUE_UPDATE_BITS) != 0
)
"""DEVIATION 2649: a shipped build whose column default carries `optskip`
and/or `noshadow`, so `_byte_step_device` takes `_byte_glue_update`. The
analogue of `ATTN_SHIPPED_BWD_ESTASH` (DEVIATION 2657)."""


def step_glue_trial_build() -> Bool:
    """True only in a build compiled with `-D MOJOLEARN_STEP_GLUE_TRIAL=1`."""
    comptime if STEP_GLUE_TRIAL:
        return True
    return False


def step_glue_arm_valid(arm: Int) -> Bool:
    """A word the parser can produce: known bits only, at most one rows bit."""
    if arm < 0 or arm > STEP_GLUE_ALL_BITS:
        return False
    var rows = arm & STEP_GLUE_ROWS_BITS
    return (
        rows == 0
        or rows == STEP_GLUE_ROWS16
        or rows == STEP_GLUE_ROWS8
        or rows == STEP_GLUE_ROWS4
    )


def step_glue_rows_of(arm: Int) -> Int:
    """Threads per block the arm's RMSNorm row launches use, or 0 for the
    shipped geometry (`LLAMA_TPB` / `BWD_TPB`, 128)."""
    var rows = arm & STEP_GLUE_ROWS_BITS
    if rows == STEP_GLUE_ROWS16:
        return 16
    if rows == STEP_GLUE_ROWS8:
        return 8
    if rows == STEP_GLUE_ROWS4:
        return 4
    return 0


comptime STEP_GLUE_DEFAULT_ROWS = step_glue_rows_of(STEP_GLUE_ARM_DEFAULT)
"""The default arm's RMSNorm threads per block, 0 for the shipped 128."""

comptime STEP_GLUE_ROWS_FIT = (
    STEP_GLUE_DEFAULT_ROWS == 0
    or STEP_GLUE_DEFAULT_ROWS <= column_max_block_size(TARGET_COLUMN)
)
"""The fit predicate, the analogue of `ATTN_ES_FITS` (DEVIATION 2657): the
default's block is inside this column's dispatch cap. Every rows token is a
small power of two, so this holds on every column today; it is written down
so a future rows token cannot silently name a geometry a column refuses."""

comptime STEP_GLUE_SHIPPED_ROWS = (
    (not STEP_GLUE_TRIAL) and STEP_GLUE_DEFAULT_ROWS != 0 and STEP_GLUE_ROWS_FIT
)
"""DEVIATION 2649: a shipped build whose column default carries a rows
token, so the two RMSNorm row launchers use it."""


def step_glue_arm_name(arm: Int) -> String:
    """The name `step_glue_arm_parse` reads back as `arm`. A word the parser
    cannot produce is spelled `bits<N>`, which the parser refuses, so a name
    printed beside a timing is never another arm's name."""
    if not step_glue_arm_valid(arm):
        return String("bits") + String(arm)
    if arm == STEP_GLUE_SHIPPED:
        return String("shipped")
    var name = String("")
    if (arm & STEP_GLUE_OPTSKIP) != 0:
        name = String("optskip")
    if (arm & STEP_GLUE_NOSHADOW) != 0:
        if name != "":
            name = name + "_"
        name = name + "noshadow"
    var rows = step_glue_rows_of(arm)
    if rows > 0:
        if name != "":
            name = name + "_"
        name = name + "rows" + String(rows)
    return name


def step_glue_arm_parse(name: String) raises -> Int:
    """The arm word for `name` (empty is `shipped`). Raises on anything that
    is not one of the 16 names `step_glue_arm_name` spells."""
    if name == "":
        return STEP_GLUE_SHIPPED
    for arm in range(STEP_GLUE_ALL_BITS + 1):
        if step_glue_arm_valid(arm) and step_glue_arm_name(arm) == name:
            return arm
    raise Error(
        "step glue arm '" + name + "' is not a valid name: shipped, or the"
        + " tokens optskip, noshadow, then one of rows16, rows8, rows4, in"
        + " that order, joined by '_' (DEVIATIONS 2645 to 2647)"
    )


def step_glue_arm_from_env() raises -> Int:
    """The glue arm for THIS call, read on the host. Trial builds read
    `MOJOLEARN_STEP_GLUE_ARM` (raising on an invalid name); every other
    build returns `STEP_GLUE_ARM_DEFAULT`, its column's shipped word, without
    reading the environment (DEVIATION 2649; it was the literal
    `STEP_GLUE_SHIPPED` before that deviation, which is what the word still
    is on every column but NVIDIA).

    The early return matters for more than clarity: the two RMSNorm
    launchers call this ONCE PER LAUNCH, so a shipped build must not reach
    `getenv` or the 32-iteration name search below.

    Every launcher calls this, so the build-time contract of the routing row
    is asserted here, the way `fused_attention_arm_from_env` asserts the
    attention words."""
    comptime assert (
        STEP_GLUE_DEFAULT_WORD_OPTSKIP_NOSHADOW_ROWS16
        == STEP_GLUE_ARM_OPTSKIP_NOSHADOW_ROWS16
    ), (
        "checks/kernel_matrix.mojo STEP_GLUE_DEFAULT_WORD_OPTSKIP_NOSHADOW_ROWS16"
        " no longer spells this file's optskip_noshadow_rows16 bits"
        " (DEVIATION 2649); fix the literal there"
    )
    comptime assert step_glue_arm_valid(STEP_GLUE_ARM_DEFAULT), (
        "step_glue_default_arm_for names a word the parser cannot produce"
    )
    comptime assert STEP_GLUE_TRIAL or STEP_GLUE_ROWS_FIT, (
        "step_glue_default_arm_for names a rows token past this column's"
        " dispatch cap; the shipped build would launch a geometry the vendor"
        " refuses"
    )
    comptime if not STEP_GLUE_TRIAL:
        return STEP_GLUE_ARM_DEFAULT
    return step_glue_arm_parse(String(getenv("MOJOLEARN_STEP_GLUE_ARM")))


def step_glue_sabotage() -> Bool:
    """The reach sabotage switch (module docstring). False on every build
    without the trial define, without reading the environment."""
    comptime if not STEP_GLUE_TRIAL:
        return False
    return String(getenv("MOJOLEARN_STEP_GLUE_ARM_SABOTAGE")) == "1"


def step_glue_blocks(count: Int, threads: Int) -> Int:
    """Blocks for a `count`-thread launch at `threads` per block: the
    ceiling, never 0. Under the reach sabotage the FLOOR (still never 0), so
    the last `count mod threads` threads are not launched."""
    var blocks = (count + threads - 1) // threads
    if step_glue_sabotage():
        blocks = count // threads
    if blocks < 1:
        blocks = 1
    return blocks
