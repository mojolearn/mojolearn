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
  Every row is off unless its define is passed. This is `fast` on NVIDIA and
  AMD (and the form Apple `fast` moves to once it is measured there).
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
    MATRIX_UNITS        False          False          False          False (its own gates)
    WAITS_ELIDED        False          False          False          False (its own gates)
    BF16_UNITS          False          False          False          False (its own gates)
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
    MATRIX_UNITS    gemm/fast_matrix_units.mojo: GEMM_FAST_MATRIX_UNITS (the NVIDIA TF32 arm,
                    `-D MOJOLEARN_GEMM_FAST_MATRIX_UNITS=<kind>`, a term of the row), and the
                    classical family routes below
    BF16_UNITS      gemm/fast_bf16_units.mojo: GEMM_FAST_BF16_UNITS (`-D MOJOLEARN_GEMM_FAST_BF16_UNITS=<kind>`)
    WAITS_ELIDED    no seam yet (only waits that can move a bit)

`MODE_POLICY_MASK` has bit i set when row i is relaxed. Bindings export it so
verification and the compiled-kernel audits read the rows from the binary.

The family routes of a row
--------------------------
A family route is one family's use of a row: one define, default off, a term
of its row (so the row's mask bit is set and an identical build stops), and
one constant here that the family's route module reads back. A route is taken
only in a fast-rows build on the column it is written for; on any other
column of a fast-rows build the define relaxes nothing there and the constant
is False, so a host binding of the same set compiles and stays exact.

    route (constant)              define                                   row           NVIDIA            AMD               Apple Metal                 host
    GEMM matrix units             MOJOLEARN_GEMM_FAST_MATRIX_UNITS=<kind>  MATRIX_UNITS  MAX matmul, TF32  MAX matmul, MFMA  afn route (AFN_GEMM_*)      exact
    ROUTE_DECOMP_MATRIX_UNITS     MOJOLEARN_DECOMP_FAST_MATRIX_UNITS       MATRIX_UNITS  MAX matmul, TF32  MAX matmul, MFMA  afn route (PCA Gram tile)   exact
    ROUTE_NEIGHBORS_MATRIX_UNITS  MOJOLEARN_NEIGHBORS_FAST_MATRIX_UNITS    MATRIX_UNITS  MAX matmul, TF32  MAX matmul, MFMA  Apple MMA distance kernels  exact
    ROUTE_GEMM_BF16_UNITS         MOJOLEARN_GEMM_FAST_BF16_UNITS=<kind>    BF16_UNITS    MAX matmul, bf16  MAX matmul, bf16  afn route (AFN_GEMM_BF16_MMA)  exact

  GEMM: gemm/fast_matrix_units.mojo (neural dispatch, sequence executor). DECOMP:
  decomposition/gram_matrix_units_route.mojo (PCA / truncated SVD: the Gram past the split-K width, the
  projections). NEIGHBORS: neighbors/distance_matrix_units_route.mojo (brute-force k-NN: the expanded L2
  distance product at wide features). One source for NVIDIA and AMD; on AMD the fp32 MFMA keeps full fp32
  operands, so the relaxation there is the vendor's fold order and K grouping, not operand width.

  BF16_UNITS is its own row because it is a different precision class (8 significant bits per operand, twice
  the TF32 truncation bound) and its second reference is torch BF16. Its route is gemm/fast_bf16_units.mojo
  (lane fx-fast-bf16, 2026-11-20): the neural dispatch, the sequence executor and the public `linalg.matmul`
  entry narrow both operands to bf16 in scratch and run one MAX `linalg.matmul` on the half-precision matrix
  unit (NVIDIA 16x8x16, AMD 16x16x16 / 32x32x8; api/mojo/layout/tensor_core/TensorCore), FP32 accumulation and
  FP32 output; the master weights, the loss, the normalisations and the optimizer stay FP32. MAX has bf16
  products on Apple M5 only and the Apple set has its own bf16 simdgroup route, so the Apple column points at
  that and the define relaxes nothing there. `MOJOLEARN_MODE_FAST_BF16_UNITS` alone sets the row's bit and
  changes no kernel: the family define is what a route reads.

Each route has a quality check (the route's result against the exact FP32
result, within a bound derived from the unit's operand width) and an owed A/B
with quality beside the time (docs/apple-fast/EXPERIMENTS.md). None is a
default: `policy_row_default` stays False until its A/B is in.
"""

from std.sys.compile import is_defined
from std.sys.defines import get_defined_int
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
comptime POLICY_ROW_BF16_UNITS = 7
comptime POLICY_ROW_COUNT = 8

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
        # The identical kernels with the contract off. Zero rows relaxed on
        # NVIDIA, AMD, Apple and the host until a relaxation is measured for
        # speed and checked for quality on that column; the measured default
        # of a column is written here, with its A/B.
        if column == POLICY_COLUMN_NVIDIA:
            return False
        if column == POLICY_COLUMN_AMD:
            return False
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
    if row == POLICY_ROW_BF16_UNITS:
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

# The family arms that are a term of a row. A leaf cannot import the family,
# so the row reads the family's define here and the family reads the row
# back (never the define alone): the table stays the single source.
#   MOJOLEARN_GEMM_FAST_MATRIX_UNITS=<kind> (gemm/fast_matrix_units.mojo, the
#   NVIDIA tensor-core products): any non-zero kind relaxes MATRIX_UNITS.
comptime _GEMM_MATRIX_UNITS_DEFINED = get_defined_int["MOJOLEARN_GEMM_FAST_MATRIX_UNITS", 0]() != 0
#   MOJOLEARN_DECOMP_FAST_MATRIX_UNITS (decomposition/gram_matrix_units_route.mojo: the PCA / truncated-SVD Gram
#   past the split-K width and the projections on the NVIDIA tensor cores).
comptime _DECOMP_MATRIX_UNITS_DEFINED = is_defined["MOJOLEARN_DECOMP_FAST_MATRIX_UNITS"]()
#   MOJOLEARN_NEIGHBORS_FAST_MATRIX_UNITS (neighbors/distance_matrix_units_route.mojo: the brute-force k-NN
#   expanded-L2 distance product at wide features on the NVIDIA tensor cores).
comptime _NEIGHBORS_MATRIX_UNITS_DEFINED = is_defined["MOJOLEARN_NEIGHBORS_FAST_MATRIX_UNITS"]()
#   MOJOLEARN_GEMM_FAST_BF16_UNITS=<kind> (gemm/fast_bf16_units.mojo: the neural products on the bf16 matrix
#   unit, FP32 accumulation): any non-zero kind relaxes BF16_UNITS.
comptime _GEMM_BF16_UNITS_DEFINED = get_defined_int["MOJOLEARN_GEMM_FAST_BF16_UNITS", 0]() != 0
# The matrix-units routes are vendor-neutral in source: one call of MAX `linalg.matmul`, whose GPU backends are
# NVIDIA (sm80 / sm90 / sm100: TF32 operands for fp32 inputs unless `use_tf32=False`), AMD CDNA (MFMA 16x16x4 for
# fp32) and Apple (the 8x8 simdgroup product on M1-M4, a structured kernel on M5); max.modular.com,
# api/mojo/linalg/matmul/gpu, read 2026-11-19. They are TAKEN on NVIDIA and AMD. On Apple each family already has
# its matrix-unit route (`gemm/afn_apple_fast.mojo` AFN_GEMM_FP32_MMA for the products and the PCA Gram,
# `neighbors/checks/apple_mma_distance.mojo` for the k-NN distance): the row points at those and adds no second
# path, so the define relaxes nothing on the Apple column. The host column has no matrix unit. Where the define
# relaxes nothing the term is False, the row's bit stays clear and the kernels are the exact ones.
comptime _FAMILY_UNITS_COLUMN = (TARGET_HAS_NVIDIA_GPU or TARGET_HAS_AMD_GPU) and not TARGET_HAS_APPLE_GPU
comptime _GEMM_MATRIX_UNITS_REQUESTED = _GEMM_MATRIX_UNITS_DEFINED and _FAMILY_UNITS_COLUMN
comptime _DECOMP_MATRIX_UNITS_REQUESTED = _DECOMP_MATRIX_UNITS_DEFINED and _FAMILY_UNITS_COLUMN
comptime _NEIGHBORS_MATRIX_UNITS_REQUESTED = _NEIGHBORS_MATRIX_UNITS_DEFINED and _FAMILY_UNITS_COLUMN
# The bf16 route has the same columns: MAX's half-precision matrix-unit product on NVIDIA and AMD; Apple has its
# own bf16 route (gemm/afn_apple_fast.mojo AFN_GEMM_BF16_MMA; MAX's is M5 only); the host has no unit.
comptime _GEMM_BF16_UNITS_REQUESTED = _GEMM_BF16_UNITS_DEFINED and _FAMILY_UNITS_COLUMN

# Any row define at all. Read by the compile error below, which must not
# depend on the resolved rows.
comptime _ANY_ROW_DEFINE = (
    _GEMM_MATRIX_UNITS_DEFINED
    or _DECOMP_MATRIX_UNITS_DEFINED
    or _NEIGHBORS_MATRIX_UNITS_DEFINED
    or _GEMM_BF16_UNITS_DEFINED
    or (
        is_defined["MOJOLEARN_MODE_FAST_FOLD_FREE"]()
        or is_defined["MOJOLEARN_MODE_FAST_CONTRACT_FREE"]()
        or is_defined["MOJOLEARN_MODE_FAST_FLUSH_OFF"]()
        or is_defined["MOJOLEARN_MODE_FAST_VENDOR_MATH"]()
        or is_defined["MOJOLEARN_MODE_FAST_MATRIX_UNITS"]()
        or is_defined["MOJOLEARN_MODE_FAST_WAITS_ELIDED"]()
        or is_defined["MOJOLEARN_MODE_FAST_TILE_FREE"]()
        or is_defined["MOJOLEARN_MODE_FAST_BF16_UNITS"]()
    )
)


def _row(row: Int, requested: Bool) -> Bool:
    """Resolve one row for this build. Every row passes through here, so any
    module that reads a row also compiles the refusal."""
    comptime assert not (CONTRACT_IDENTICAL and _ANY_ROW_DEFINE), "a MOJOLEARN_MODE_FAST_* relaxation (or a family arm of one: MOJOLEARN_GEMM_FAST_MATRIX_UNITS, MOJOLEARN_DECOMP_FAST_MATRIX_UNITS, MOJOLEARN_NEIGHBORS_FAST_MATRIX_UNITS, MOJOLEARN_GEMM_FAST_BF16_UNITS) was requested in a build that promises identical bits; relaxations are effective only in a fast build (-D MOJOLEARN_MODE_FAST_ROWS)"
    return policy_row_default(row, POLICY_BUILD, POLICY_COLUMN) or requested


# Reduction order: a fold may take the vendor's own shape.
comptime ROW_FOLD_FREE = _row(POLICY_ROW_FOLD_FREE, is_defined["MOJOLEARN_MODE_FAST_FOLD_FREE"]())
# Fused multiply-add: the compiler may contract or split a*b + c.
comptime ROW_CONTRACT_FREE = _row(POLICY_ROW_CONTRACT_FREE, is_defined["MOJOLEARN_MODE_FAST_CONTRACT_FREE"]())
# Denormal flush: subnormal results are not flushed to zero.
comptime ROW_FLUSH_OFF = _row(POLICY_ROW_FLUSH_OFF, is_defined["MOJOLEARN_MODE_FAST_FLUSH_OFF"]())
# Math library: exp, log, sqrt, division and the rest use the vendor's forms.
comptime ROW_VENDOR_MATH = _row(POLICY_ROW_VENDOR_MATH, is_defined["MOJOLEARN_MODE_FAST_VENDOR_MATH"]())
# Matrix units: reduced-precision products (TF32, BF16, MFMA forms).
comptime ROW_MATRIX_UNITS = _row(
    POLICY_ROW_MATRIX_UNITS,
    (
        is_defined["MOJOLEARN_MODE_FAST_MATRIX_UNITS"]()
        or _GEMM_MATRIX_UNITS_REQUESTED
        or _DECOMP_MATRIX_UNITS_REQUESTED
        or _NEIGHBORS_MATRIX_UNITS_REQUESTED
    ),
)
# The classical family routes of the matrix-units row (the table in the module docstring). True only in a
# fast-rows build (never the legacy fast set, whose products are the old vendor arms, and never under the
# contract: `_row` stops that build) on the NVIDIA or AMD column with the family's define passed. The family's route
# module imports its constant; no other module reads the define.
comptime ROUTE_GEMM_MATRIX_UNITS = (
    POLICY_BUILD == POLICY_BUILD_FAST_ROWS and ROW_MATRIX_UNITS and _GEMM_MATRIX_UNITS_REQUESTED
)
comptime ROUTE_DECOMP_MATRIX_UNITS = (
    POLICY_BUILD == POLICY_BUILD_FAST_ROWS and ROW_MATRIX_UNITS and _DECOMP_MATRIX_UNITS_REQUESTED
)
comptime ROUTE_NEIGHBORS_MATRIX_UNITS = (
    POLICY_BUILD == POLICY_BUILD_FAST_ROWS and ROW_MATRIX_UNITS and _NEIGHBORS_MATRIX_UNITS_REQUESTED
)
# Waits: only the waits whose removal can move a bit.
comptime ROW_WAITS_ELIDED = _row(POLICY_ROW_WAITS_ELIDED, is_defined["MOJOLEARN_MODE_FAST_WAITS_ELIDED"]())
# Tile and block shape: only the tiles that bound a float fold.
comptime ROW_TILE_FREE = _row(POLICY_ROW_TILE_FREE, is_defined["MOJOLEARN_MODE_FAST_TILE_FREE"]())
# BF16 matrix units: half-precision operands on the matrix unit, FP32 accumulation (gemm/fast_bf16_units.mojo).
comptime ROW_BF16_UNITS = _row(
    POLICY_ROW_BF16_UNITS,
    is_defined["MOJOLEARN_MODE_FAST_BF16_UNITS"]() or _GEMM_BF16_UNITS_REQUESTED,
)
# The GEMM route of the bf16 row: True only in a fast-rows build on the NVIDIA or AMD column with the family
# define passed. gemm/fast_bf16_units.mojo imports it; no other module reads the define.
comptime ROUTE_GEMM_BF16_UNITS = (
    POLICY_BUILD == POLICY_BUILD_FAST_ROWS and ROW_BF16_UNITS and _GEMM_BF16_UNITS_REQUESTED
)

comptime ANY_ROW_RELAXED = (
    ROW_FOLD_FREE or ROW_CONTRACT_FREE or ROW_FLUSH_OFF or ROW_VENDOR_MATH
    or ROW_MATRIX_UNITS or ROW_WAITS_ELIDED or ROW_TILE_FREE or ROW_BF16_UNITS
)

# Bit i set = row i relaxed. 0 in every identical build; 79 in a legacy fast
# build (rows 0, 1, 2, 3 and 6).
comptime MODE_POLICY_MASK = (
    (1 if ROW_FOLD_FREE else 0)
    + (2 if ROW_CONTRACT_FREE else 0)
    + (4 if ROW_FLUSH_OFF else 0)
    + (8 if ROW_VENDOR_MATH else 0)
    + (16 if ROW_MATRIX_UNITS else 0)
    + (32 if ROW_WAITS_ELIDED else 0)
    + (64 if ROW_TILE_FREE else 0)
    + (128 if ROW_BF16_UNITS else 0)
)


def mode_policy_mask() -> Int:
    """The rows this binary was compiled with, for a binding to export."""
    comptime assert not (CONTRACT_IDENTICAL and MODE_POLICY_MASK != 0), "a mode-policy row is relaxed in a build that promises identical bits"
    return MODE_POLICY_MASK
