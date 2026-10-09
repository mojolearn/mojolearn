# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""2026-10-07 lane trees-hist-ideas: IDENTICAL GBDT histogram switches.

All default OFF (define absent). Each one changes WHERE an integer lands or
HOW MANY integer atomics carry it, never which integer: every addend is the
same `hist2_quantize(stat, fixed_scale, hist2_dither(position))` Int32 the
incumbent kernels add, and integer addition is associative, so the
histogram cells -- and therefore every split -- keep their bits on every
column. Separate from `trees_identical_switches.mojo` (lane trees-cleanup
owns that file).

NOT COMPILED -- NOT TESTED -- IDENTITY NOT VERIFIED -- NOT MEASURED.

| define | kind | legal values | gates |
|---|---|---|---|
| MOJOLEARN_TREES_HIST_PACKED_GH   | switch   | on/off        | hist_2_one_byte_8bit.mojo `h8_add_point` |
| MOJOLEARN_TREES_HIST_WARP_AGG    | switch   | on/off        | hist_2_one_byte_8bit.mojo body loops |
| MOJOLEARN_TREES_HIST_SYM_FEATURE_PARALLEL | switch | on/off | greedy_search_helper.mojo `replication_for` |
"""
from std.sys.compile import is_defined
from std.sys.defines import get_defined_int
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL

comptime _IDN = GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL

# Idea 1: grad+hess of one point in ONE 64-bit shared atomic (NVIDIA/AMD;
# Apple has no 64-bit atomics and keeps the two Int32 atomics). Range proof
# in `hist_2_one_byte_8bit.h8_add_point`.
comptime HIST_PACKED_GH = _IDN and is_defined["MOJOLEARN_TREES_HIST_PACKED_GH"]()

# Idea 2: wave-aggregated atomics. When every lane of a hardware wave holds
# the same (feature, bin) cell, the wave sums its addends with one shuffle
# reduction and ONE lane issues the atomic. See `h8_add_point_wave`.
comptime HIST_WARP_AGG = _IDN and is_defined["MOJOLEARN_TREES_HIST_WARP_AGG"]()

# Tried 2026-10-08 (MOJOLEARN_TREES_HIST_MULTISTAT = 4 | 8, run ge123e6f9): MultiClass one-byte histograms, one cindex
# walk per 4 or 8 stat planes (launch_hist2_8bit_wide[NS, 1]); NV/AMD gbdt-multiclass istella 1.04x/1.41x (4), 0.99x/1.31x (8),
# taxi 1.07x/1.11x (4), 1.11x/1.14x (8) SLOWER; accuracy and mlogloss SAME -> deleted. Recoverable at main 42d1e42c6.
# (Idea 3; the wide kernel stays for idea 4.)

# Idea 4: cost-chosen feature-parallel vs row-parallel histogram grid for
# the SymmetricTree greedy searcher. See `sym_feature_parallel_replicas`.
comptime HIST_SYM_FEATURE_PARALLEL = _IDN and is_defined[
    "MOJOLEARN_TREES_HIST_SYM_FEATURE_PARALLEL"
]()
