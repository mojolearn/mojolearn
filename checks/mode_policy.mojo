# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of Mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The mode policy table: what `fast` may relax, row by row, and the contract.

`fast` is `identical` with named relaxations switched on. This module is the
one place that says which relaxations a build has on. It is a leaf: it reads
`core/package_profile.mojo` and `core/accelerator_target.mojo` only, so
`checks/numerics.mojo` and `checks/kernel_matrix.mojo` can both import it.
Every value is a `comptime` constant, so no kernel pays a lookup: a seam that
reads a row is a `comptime if` on a constant.

The contract constant
---------------------
`CONTRACT_IDENTICAL` is True when the build promises the same bits on every
vendor. It is separate from "which kernel form is compiled"
(`GLOBAL_NUMERIC_MODE` in `checks/numerics.mojo`), so that a build can compile
the identical kernels and still be allowed to relax a row.

The three kinds of build
------------------------
- `POLICY_BUILD_IDENTICAL`: `-D MOJOLEARN_NUMERIC_IDENTICAL`. The contract
  holds; every row is off; a row define is a compile error.
- `POLICY_BUILD_FAST_ROWS`: `-D MOJOLEARN_NUMERIC_IDENTICAL
  -D MOJOLEARN_MODE_FAST_ROWS`. The identical kernels with the contract off.
  A row is off unless its define is passed or the table below says it is the
  measured default of the column (MATRIX_UNITS on NVIDIA and AMD). This is
  `fast` on NVIDIA and AMD (and the form Apple `fast` moves to once it is
  measured there).
- `POLICY_BUILD_LEGACY_FAST`: no `MOJOLEARN_NUMERIC_IDENTICAL`. The `fast`
  set as it was built before this table existed (Apple's shipped `fast` set):
  the five seam rows are on, as the seams always compiled them there.

The table (True = relaxed)
--------------------------
    row                 identical      fast rows      fast rows      legacy fast
                        every column   NVIDIA / AMD   Apple / host   (Apple fast today)
    FOLD_FREE           False          False          False          True
    CONTRACT_FREE       False          False          False          True
    FLUSH_OFF           False          False          False          True
    VENDOR_MATH         False          False          False          True
    MATRIX_UNITS        False          True           False          False (its own gates)
    WAITS_ELIDED        False          False          False          False (its own gates)
    TILE_FREE           False          False          False          True

The host column reads the same rows as the device columns of its set, so a
host routine that mirrors a device fold stays on the same side of every row.

A relaxation is one define, `-D MOJOLEARN_MODE_FAST_<ROW>`, default off. It is
effective only where the contract is off: passing one in an identical build
stops the build (`_row`). A family narrows a row with its own define
(`-D MOJOLEARN_<AREA>_FAST_<ROW>`) as one extra term at that family's seam
call, under the experiment rules.

Who reads the rows
------------------
    FOLD_FREE       core/pinned_reduce.mojo: pinned_block_sum, pinned_block_prefix_sum
    CONTRACT_FREE   checks/numerics.mojo: identical_mul_add, identical_mul, identical_mul64,
                    identical_mul_add_simd
    FLUSH_OFF       checks/numerics.mojo: ftz, ftz_simd
    VENDOR_MATH     checks/numerics.mojo: identical_exp ... identical_clamp
    TILE_FREE       checks/kernel_matrix.mojo: lib_block_size_for
    MATRIX_UNITS    gemm/fast_matrix_units.mojo: GEMM_FAST_MATRIX_UNITS (the neural products on the
                    matrix unit, the route below)
    WAITS_ELIDED    no seam yet (only waits that can move a bit)

`MODE_POLICY_MASK` has bit i set when row i is relaxed. Bindings export it so
verification and the compiled-kernel audits read the rows from the binary.

The matrix-units route (the default of `fast` on NVIDIA and AMD)
----------------------------------------------------------------
    route (constant)          row           NVIDIA            AMD               Apple Metal             host
    ROUTE_GEMM_MATRIX_UNITS   MATRIX_UNITS  MAX matmul, TF32  MAX matmul, MFMA  afn route (AFN_GEMM_*)  exact

  gemm/fast_matrix_units.mojo: the large neural FP32 products (neural dispatch, every caller; the sequence
  executor; the public `linalg.matmul` entry) run on MAX `linalg.matmul`. One source for NVIDIA and AMD; on AMD
  the fp32 MFMA keeps full fp32 operands, so the relaxation there is the vendor's fold order and K grouping,
  not operand width. On Apple the products already have their matrix-unit route (gemm/afn_apple_fast.mojo), so
  the row stays off there and adds no second path; the host column has no matrix unit and stays exact.

  PROMOTED 2026-11-20 (lane fx-verdicts6) on A/B abf2: NVIDIA L40S host 79f003b8e8ae, base wheel e41b5148b24d,
  base and arm back to back in one job (nvc1-0001, R2 tag nvc-ab-20261120T224801Z-458a59), one run per arm, full
  datasets, arm `fast-matrix-units-tf32-all`. 12 of 13 cells faster, 1 flat, 0 slower, median arm / base 0.93:
  MLP fit 0.96-0.97 (4 cells), MLP infer 0.85-0.95, gemm 0.93, lm-forward 0.90, lm-train-step 0.82,
  transformer-forward 0.91, transformer-decode 1.00. Quality equal: mlp-clf accuracy 0.76790 vs 0.76795 and
  0.94345 vs 0.94326; mlp-reg r2 0.5224 vs 0.5206 and 0.93011 vs 0.93015; gemm max_rel_err_vs_fp64 2.8e-4
  (base 2.4e-7, expected of TF32 operands). NVIDIA only (AGENTS.md part 8); AMD takes the same source untimed
  (Andrew, 2026-11-20: a win on NVIDIA is the default on every vendor without its own measurement). Sample: one
  run per arm. Limits: no AMD or Apple timing; the size rule's constants are reasoned, not measured.
  The old `fast` route (every product on the exact FP32 kernels, row off) is
  `-D MOJOLEARN_GEMM_FAST_MATRIX_UNITS_OFF`. `identical` builds are untouched: the row is False under the
  contract on every column and the route is not compiled.

  TOMBSTONES (2026-11-20, lane fx-verdicts6, the same A/B; docs/TOMBSTONES.md "A/B abf2 verdicts"):
  # TOMBSTONE: MOJOLEARN_GEMM_FAST_MATRIX_UNITS (the tiers: kind 1 tf32 open callers only, 10 faster / 3 flat,
  #   median 0.96, dominated by kind 3 on every cell; kind 3 is the default) deleted 2026-11-20 by
  #   lane/fx-verdicts6; code recoverable at 06d1e8ff34.
  # Restore: git apply experiments/removed/MOJOLEARN_GEMM_FAST_MATRIX_UNITS.tiers.patch
  # TOMBSTONE: MOJOLEARN_GEMM_FAST_BF16_UNITS and its row MOJOLEARN_MODE_FAST_BF16_UNITS (row 7, mask bit 128;
  #   median 0.96 but mlp-clf/taxi infer 1.86, gemm max_rel_err 2.2e-3; dominated by tf32-all on every cell)
  #   deleted 2026-11-20 by lane/fx-verdicts6; code recoverable at 06d1e8ff34.
  # Restore: git apply experiments/removed/MOJOLEARN_GEMM_FAST_BF16_UNITS.patch
  # TOMBSTONE: MOJOLEARN_DECOMP_FAST_MATRIX_UNITS (ROUTE_DECOMP_MATRIX_UNITS; NOISE: median 1.00, 8 cells
  #   0.93-1.00) deleted 2026-11-20 by lane/fx-verdicts6; code recoverable at 06d1e8ff34.
  # Restore: git apply experiments/removed/MOJOLEARN_DECOMP_FAST_MATRIX_UNITS.patch
  # TOMBSTONE: MOJOLEARN_NEIGHBORS_FAST_MATRIX_UNITS (ROUTE_NEIGHBORS_MATRIX_UNITS; NOISE: median 0.99, 6 cells
  #   0.98-1.01) deleted 2026-11-20 by lane/fx-verdicts6; code recoverable at 06d1e8ff34.
  # Restore: git apply experiments/removed/MOJOLEARN_NEIGHBORS_FAST_MATRIX_UNITS.patch
"""

from std.sys.compile import is_defined
from core.package_profile import identical_mode_enabled
from core.accelerator_target import (
    TARGET_HAS_AMD_GPU,
    TARGET_HAS_APPLE_GPU,
    TARGET_HAS_NVIDIA_GPU,
)


# The rows. The value is the bit position in MODE_POLICY_MASK.
comptime POLICY_ROW_FOLD_FREE = 0
comptime POLICY_ROW_CONTRACT_FREE = 1
comptime POLICY_ROW_FLUSH_OFF = 2
comptime POLICY_ROW_VENDOR_MATH = 3
comptime POLICY_ROW_MATRIX_UNITS = 4
comptime POLICY_ROW_WAITS_ELIDED = 5
comptime POLICY_ROW_TILE_FREE = 6
comptime POLICY_ROW_COUNT = 7

# The kinds of build.
comptime POLICY_BUILD_IDENTICAL = 0
comptime POLICY_BUILD_FAST_ROWS = 1
comptime POLICY_BUILD_LEGACY_FAST = 2

# The columns. The numbers are those of `checks/kernel_matrix.mojo`
# (COLUMN_APPLE, COLUMN_NVIDIA, COLUMN_AMD, COLUMN_CPU), which this leaf
# cannot import.
comptime POLICY_COLUMN_APPLE = 1
comptime POLICY_COLUMN_NVIDIA = 2
comptime POLICY_COLUMN_AMD = 3
comptime POLICY_COLUMN_HOST = 8


def policy_row_default(row: Int, build: Int, column: Int) -> Bool:
    """The table: whether `row` is relaxed in a `build` of kind
    POLICY_BUILD_* on `column`, before any row define is applied."""
    if build == POLICY_BUILD_IDENTICAL:
        # The contract: nothing is relaxed on any column or on the host.
        return False
    if build == POLICY_BUILD_FAST_ROWS:
        # The identical kernels with the contract off. A row is relaxed on a
        # column only once it is measured for speed and checked for quality;
        # the measured default of a column is written here, with its A/B.
        # MATRIX_UNITS on NVIDIA and AMD: A/B abf2, 2026-11-20, NVIDIA L40S,
        # arm fast-matrix-units-tf32-all, median 0.93 over 13 cells, 0 slower,
        # quality equal (the numbers are in the module docstring). AMD takes
        # the same `linalg.matmul` source untimed. Apple keeps its own
        # matrix-unit route (gemm/afn_apple_fast.mojo) and the host has no
        # matrix unit, so the row is off on both.
        if column == POLICY_COLUMN_NVIDIA:
            return row == POLICY_ROW_MATRIX_UNITS
        if column == POLICY_COLUMN_AMD:
            return row == POLICY_ROW_MATRIX_UNITS
        if column == POLICY_COLUMN_APPLE:
            return False
        return False
    # POLICY_BUILD_LEGACY_FAST: what the seams compiled in a fast set before
    # the table existed. They tested the mode and never the vendor, so the
    # five seam rows are on for every column; matrix units and waits were
    # never seam decisions there (their own Apple gates decide them).
    if row == POLICY_ROW_MATRIX_UNITS:
        return False
    if row == POLICY_ROW_WAITS_ELIDED:
        return False
    return True


def _policy_build() -> Int:
    comptime if identical_mode_enabled():
        comptime if is_defined["MOJOLEARN_MODE_FAST_ROWS"]():
            return POLICY_BUILD_FAST_ROWS
        return POLICY_BUILD_IDENTICAL
    return POLICY_BUILD_LEGACY_FAST


def _policy_column() -> Int:
    comptime if TARGET_HAS_APPLE_GPU:
        return POLICY_COLUMN_APPLE
    comptime if TARGET_HAS_NVIDIA_GPU:
        return POLICY_COLUMN_NVIDIA
    comptime if TARGET_HAS_AMD_GPU:
        return POLICY_COLUMN_AMD
    return POLICY_COLUMN_HOST


comptime POLICY_BUILD = _policy_build()

# Which kernel form this build compiles. KERNEL_FORM_IDENTICAL is the constant
# `GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL` of `checks/numerics.mojo` (both
# are `identical_mode_enabled()`); it stays True in a fast-rows build, which
# compiles the identical kernels. A module that needs its own name for the
# mode imports one of these under that name
# (`from checks.mode_policy import KERNEL_FORM_IDENTICAL as IDENTICAL`) and
# does not define a copy.
comptime KERNEL_FORM_IDENTICAL = POLICY_BUILD != POLICY_BUILD_LEGACY_FAST
comptime KERNEL_FORM_LEGACY_FAST = POLICY_BUILD == POLICY_BUILD_LEGACY_FAST
comptime POLICY_COLUMN = _policy_column()

# The contract: does this build promise the same bits on every vendor?
comptime CONTRACT_IDENTICAL = POLICY_BUILD == POLICY_BUILD_IDENTICAL

# The old `fast` route of the matrix-units row: `-D MOJOLEARN_GEMM_FAST_MATRIX_UNITS_OFF` keeps every product on
# the exact FP32 kernels and clears the row's bit (the B arm of the promotion; the numbers are in the module
# docstring). It relaxes nothing, so it is legal in every build and is not a row define.
comptime _GEMM_MATRIX_UNITS_OFF = is_defined["MOJOLEARN_GEMM_FAST_MATRIX_UNITS_OFF"]()
# The columns of the route: MAX `linalg.matmul` has NVIDIA (sm80 / sm90 / sm100: TF32 operands for fp32 inputs
# unless `use_tf32=False`), AMD CDNA (MFMA 16x16x4 for fp32) and Apple backends (max.modular.com,
# api/mojo/linalg/matmul/gpu, read 2026-11-19). The route is TAKEN on NVIDIA and AMD. On Apple the products
# already have their matrix-unit route (`gemm/afn_apple_fast.mojo` AFN_GEMM_FP32_MMA): the row points at that and
# adds no second path. The host column has no matrix unit.
comptime _MATRIX_UNITS_COLUMN = (TARGET_HAS_NVIDIA_GPU or TARGET_HAS_AMD_GPU) and not TARGET_HAS_APPLE_GPU

# Any row define at all. Read by the compile error below, which must not
# depend on the resolved rows.
comptime _ANY_ROW_DEFINE = (
    is_defined["MOJOLEARN_MODE_FAST_FOLD_FREE"]()
    or is_defined["MOJOLEARN_MODE_FAST_CONTRACT_FREE"]()
    or is_defined["MOJOLEARN_MODE_FAST_FLUSH_OFF"]()
    or is_defined["MOJOLEARN_MODE_FAST_VENDOR_MATH"]()
    or is_defined["MOJOLEARN_MODE_FAST_MATRIX_UNITS"]()
    or is_defined["MOJOLEARN_MODE_FAST_WAITS_ELIDED"]()
    or is_defined["MOJOLEARN_MODE_FAST_TILE_FREE"]()
)


def _row(row: Int, requested: Bool) -> Bool:
    """Resolve one row for this build. Every row passes through here, so any
    module that reads a row also compiles the refusal."""
    comptime assert not (CONTRACT_IDENTICAL and _ANY_ROW_DEFINE), "a MOJOLEARN_MODE_FAST_* relaxation was requested in a build that promises identical bits; relaxations are effective only in a fast build (-D MOJOLEARN_MODE_FAST_ROWS)"
    return policy_row_default(row, POLICY_BUILD, POLICY_COLUMN) or requested


# Reduction order: a fold may take the vendor's own shape.
comptime ROW_FOLD_FREE = _row(POLICY_ROW_FOLD_FREE, is_defined["MOJOLEARN_MODE_FAST_FOLD_FREE"]())
# Fused multiply-add: the compiler may contract or split a*b + c.
comptime ROW_CONTRACT_FREE = _row(POLICY_ROW_CONTRACT_FREE, is_defined["MOJOLEARN_MODE_FAST_CONTRACT_FREE"]())
# Denormal flush: subnormal results are not flushed to zero.
comptime ROW_FLUSH_OFF = _row(POLICY_ROW_FLUSH_OFF, is_defined["MOJOLEARN_MODE_FAST_FLUSH_OFF"]())
# Math library: exp, log, sqrt, division and the rest use the vendor's forms.
comptime ROW_VENDOR_MATH = _row(POLICY_ROW_VENDOR_MATH, is_defined["MOJOLEARN_MODE_FAST_VENDOR_MATH"]())
# Matrix units: the large neural products on the vendor's matrix unit (TF32 operands on NVIDIA, fp32 MFMA on
# AMD). The default of a fast-rows build on those two columns (`policy_row_default`); the row define sets the
# bit on any column of a fast build; `-D MOJOLEARN_GEMM_FAST_MATRIX_UNITS_OFF` clears the default.
comptime _MATRIX_UNITS_ROW_DEFINED = is_defined["MOJOLEARN_MODE_FAST_MATRIX_UNITS"]()
comptime ROW_MATRIX_UNITS = _row(POLICY_ROW_MATRIX_UNITS, _MATRIX_UNITS_ROW_DEFINED) and (
    _MATRIX_UNITS_ROW_DEFINED or not _GEMM_MATRIX_UNITS_OFF
)
# The route of the row (the table in the module docstring): True in a fast-rows build (never the legacy fast
# set, whose products are the old vendor arms, and never under the contract, where the row is False) on the
# NVIDIA or AMD column, unless the `_OFF` define keeps the old route. gemm/fast_matrix_units.mojo imports it;
# no other module reads a define of the row. Resolved once, at compile time: no call pays a lookup.
comptime ROUTE_GEMM_MATRIX_UNITS = (
    POLICY_BUILD == POLICY_BUILD_FAST_ROWS and ROW_MATRIX_UNITS and _MATRIX_UNITS_COLUMN
    and not _GEMM_MATRIX_UNITS_OFF
)
# Waits: only the waits whose removal can move a bit.
comptime ROW_WAITS_ELIDED = _row(POLICY_ROW_WAITS_ELIDED, is_defined["MOJOLEARN_MODE_FAST_WAITS_ELIDED"]())
# Tile and block shape: only the tiles that bound a float fold.
comptime ROW_TILE_FREE = _row(POLICY_ROW_TILE_FREE, is_defined["MOJOLEARN_MODE_FAST_TILE_FREE"]())

comptime ANY_ROW_RELAXED = (
    ROW_FOLD_FREE or ROW_CONTRACT_FREE or ROW_FLUSH_OFF or ROW_VENDOR_MATH
    or ROW_MATRIX_UNITS or ROW_WAITS_ELIDED or ROW_TILE_FREE
)

# Bit i set = row i relaxed. 0 in every identical build; 79 in a legacy fast
# build (rows 0, 1, 2, 3 and 6); 16 in a fast-rows build on NVIDIA or AMD
# (row 4, the matrix-units default).
comptime MODE_POLICY_MASK = (
    (1 if ROW_FOLD_FREE else 0)
    + (2 if ROW_CONTRACT_FREE else 0)
    + (4 if ROW_FLUSH_OFF else 0)
    + (8 if ROW_VENDOR_MATH else 0)
    + (16 if ROW_MATRIX_UNITS else 0)
    + (32 if ROW_WAITS_ELIDED else 0)
    + (64 if ROW_TILE_FREE else 0)
)


def mode_policy_mask() -> Int:
    """The rows this binary was compiled with, for a binding to export."""
    comptime assert not (CONTRACT_IDENTICAL and MODE_POLICY_MASK != 0), "a mode-policy row is relaxed in a build that promises identical bits"
    return MODE_POLICY_MASK
