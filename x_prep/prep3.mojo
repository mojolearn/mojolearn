# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The prep3 lane's FAST + Apple switches (lane/apple-fast-prep3, 2026-10-02).

Each is a build-time define, default OFF, honoured only in a FAST build for
the Apple GPU: the IDENTICAL binding, the host binding and the other vendors
compile main's code unchanged whatever is defined.

  -D MOJOLEARN_PREP3_LABELS  the labels' run scan (uniq_count / uniq_scan /
                             uniq_write) and chunk_neg by flag-and-scan
                             threadgroups (x_prep/fastlabels.mojo): the same
                             words, no thread walking a chunk of rows alone.
  -D MOJOLEARN_PREP3_MAXABS  MaxAbsScaler.fit from the caller's own buffer in
                             one upload (x_prep/fastmaxabs.mojo; the entry
                             `x_prep_maxabs_fit_direct` exists in no other build).
  -D MOJOLEARN_PREP3_SPLINE  SplineTransformer: the blocked count / min / max
                             units and no dead sorted-column block (Python
                             route, taken when `x_prep_prep3_spline` exists).
"""
from std.sys.compile import is_defined
from std.sys.info import has_apple_gpu_accelerator
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_FAST

comptime PREP3_FAST_APPLE = GLOBAL_NUMERIC_MODE == NUMERIC_FAST and has_apple_gpu_accelerator()
comptime PREP3_LABELS = PREP3_FAST_APPLE and is_defined["MOJOLEARN_PREP3_LABELS"]()
comptime PREP3_MAXABS = PREP3_FAST_APPLE and is_defined["MOJOLEARN_PREP3_MAXABS"]()
comptime PREP3_SPLINE = PREP3_FAST_APPLE and is_defined["MOJOLEARN_PREP3_SPLINE"]()
