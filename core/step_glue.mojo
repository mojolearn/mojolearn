# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""DEVIATIONS 2645 to 2648: the byte LM step glue arms
(docs/lanes/BRIEF_step_glue_2026-09-11.md).

EVERY NEW LAUNCH PATH BEHIND THESE HELPERS IS COMPILED ONLY UNDER
`-D MOJOLEARN_STEP_GLUE_TRIAL=1` (`comptime STEP_GLUE_TRIAL`). On a build
without the define `step_glue_arm_from_env()` returns `STEP_GLUE_SHIPPED`
without reading the environment and `step_glue_sabotage()` returns False,
and every caller guards its new branch with `comptime if STEP_GLUE_TRIAL`,
so a shipped build compiles the shipped launches only.

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
    build returns `STEP_GLUE_SHIPPED` without reading the environment."""
    comptime if not STEP_GLUE_TRIAL:
        return STEP_GLUE_SHIPPED
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
