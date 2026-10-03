# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The prep3 lane's FAST + Apple switch (lane/apple-fast-prep3, 2026-10-02).

Honoured only in a FAST build for the Apple GPU: the IDENTICAL binding, the
host binding and the other vendors compile main's code unchanged.

  PREP3_MAXABS  MaxAbsScaler.fit from the caller's own buffer in one upload
                (x_prep/fastmaxabs.mojo; the entry `x_prep_maxabs_fit_direct`
                exists in no other build). FAST + Apple DEFAULT since M2 A/B
                prep3-maxabs-istella-x: maxabs-scaler istella 133.0 -> 99.1 ms,
                the same max_abs_ / scale_ words.
                `-D MOJOLEARN_PREP3_MAXABS_OFF` reverts to the program route.
"""
from std.sys.compile import is_defined
from std.sys.info import has_apple_gpu_accelerator
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_FAST

comptime PREP3_FAST_APPLE = GLOBAL_NUMERIC_MODE == NUMERIC_FAST and has_apple_gpu_accelerator()
comptime PREP3_MAXABS = PREP3_FAST_APPLE and is_defined["MOJOLEARN_PREP3_MAXABS"]()
