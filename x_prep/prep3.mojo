# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The prep3 lane's FAST + Apple switch (lane/apple-fast-prep3, 2026-10-02).

Honoured only in a FAST build for the Apple GPU: the IDENTICAL binding, the
host binding and the other vendors compile main's code unchanged.

  PREP3_MAXABS  MaxAbsScaler.fit from the caller's own buffer in one upload
                (x_prep/fastmaxabs.mojo; the entry `x_prep_maxabs_fit_direct`
                exists in no other build). FAST + Apple DEFAULT since M3 A/B
                prep3-maxabs-istella-x-m3: maxabs-scaler istella 121.7 ->
                104.2 ms (-14.4%), output digest bit-identical (M2 A/B
                prep3-maxabs-istella-x 133.0 -> 99.1 ms).
                `-D MOJOLEARN_PREP3_MAXABS_OFF` reverts to the program route.
  PREP3_MAXABS_POOL  (FAST + Apple default, rollback
                -D MOJOLEARN_PREP3_MAXABS_POOL_OFF, on PREP3_MAXABS): the direct
                fit's three device buffers (X's n d floats, 880 MB at
                istella, the partials and the 2 d outputs) come from
                core/device_pool (exact size, kept between calls) instead of
                three fresh allocations and frees per fit. Storage only: the
                same kernels, the same words.
"""
from std.sys.compile import is_defined
from std.sys.info import has_apple_gpu_accelerator
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_FAST, NUMERIC_IDENTICAL

comptime PREP3_FAST_APPLE = GLOBAL_NUMERIC_MODE == NUMERIC_FAST and has_apple_gpu_accelerator()
#: lane/idn-gates (2026-10-04): PREP3_MAXABS is also the IDENTICAL default on
#: every vendor's GPU binding (a maximum has one answer in any order: the
#: program route's max_abs_ and scale_ words); -D MOJOLEARN_IDN_GATES_OFF (or
#: the _OFF) restores the program route in IDENTICAL.
comptime PREP3_IDN = GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL and not (is_defined["MOJOLEARN_IDN_GATES_OFF"]() or is_defined["MOJOLEARN_IDN_ALL_OFF"]())
comptime PREP3_MAXABS = (PREP3_FAST_APPLE or PREP3_IDN) and not is_defined["MOJOLEARN_PREP3_MAXABS_OFF"]()
# KEEP, 2026-10-04: M3 w2-w4s-maxabs-istella, source 74d233862d74e2c8e48b7db38a24ec85a22b80a1,
# maxabs-scaler istella 104.6 -> 17.0 ms. w2-w4s-maxabs-q PASS:
# 12/12 output arrays byte-identical, including repeated pool reuse; max_abs_
# also exact against NumPy. Same kernels and caller-owned output, download
# synchronized before return (no deferred first-read cost). See EXPERIMENTS.md.
# _OFF restores fresh device allocations; IDENTICAL and other vendors unchanged.
# merge 2026-10-04: the pool stays FAST + Apple (where it was measured); PREP3_MAXABS is now also
# the IDENTICAL default (lane idn-gates), and the pool in IDENTICAL is an unmeasured candidate.
comptime PREP3_MAXABS_POOL = PREP3_FAST_APPLE and PREP3_MAXABS and not is_defined["MOJOLEARN_PREP3_MAXABS_POOL_OFF"]()
