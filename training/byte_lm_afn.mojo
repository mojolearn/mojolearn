# SPDX-License-Identifier: Apache-2.0
"""Apple FAST neural experiments for the byte LM (lane afn-lm, 2026-10-03).

`BYTE_LM_FAST_APPLE` is true only on a FAST build (no
`-D MOJOLEARN_NUMERIC_IDENTICAL`) whose host is Apple silicon. Every
experiment below is that AND its own `-D MOJOLEARN_AFN_LM_<NAME>` define
(or `-D MOJOLEARN_AFN_LM_ALL`), default OFF, so an IDENTICAL build, and a
FAST build on NVIDIA or AMD, compiles main's code unchanged.
docs/apple-fast/notes/neural-lm.md lists what each one changes.
"""
from std.sys.compile import is_defined
from std.sys.info import has_apple_gpu_accelerator
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_FAST

comptime BYTE_LM_FAST_APPLE = (
    GLOBAL_NUMERIC_MODE == NUMERIC_FAST and has_apple_gpu_accelerator()
)
