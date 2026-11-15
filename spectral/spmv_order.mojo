# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The Lanczos matvec's fold order (lane fam2-cluster, 2026-10-04).

Row `r` of `u = A x` folds in SPMV_LANES fixed lanes and one fixed tree:
lane `l` takes the row's entries `lo + l, lo + l + SPMV_LANES, ...` (the
canonical ascending-column order, strided), `acc = ftz(fma(val, x[col],
acc))` from `+0.0`; the lane sums then combine pairwise, `part[l] =
ftz(part[l] + part[l + w])` for `w = SPMV_LANES / 2, ..., 1`, and `part[0]`
is the row. The order is a function of the row's entries alone (no launch
shape, no warp width), so the device (`lanczos.mojo::id_spmv_lanes_kernel`,
one thread a lane) and the host column (`spectral_oracle.mojo::host_spmv`)
write the same words on every vendor. The chain it replaces was one thread
walking the whole row (thousands of terms on a kNN graph at k = n / 10).
The device and the host column both read this constant.
"""

from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL

comptime SPMV_LANES = 32

# TOMBSTONE: MOJOLEARN_IDN_ALL_OFF (old path, no A/B recorded beside the default) deleted 2026-11-15 by lane/expdel2-13-cluster (non-owner; owner expdel2-37-series); code recoverable at c089be9c4a.
# Restore: git apply experiments/removed/MOJOLEARN_IDN_ALL_OFF.expdel2-13-cluster.patch; record in docs/TOMBSTONES.md.
# TOMBSTONE: MOJOLEARN_IDN_SPMV_LANES_OFF (old path, no A/B recorded beside the default) deleted 2026-11-15 by lane/expdel2-13-cluster; code recoverable at 4f4208e8d5.
# Restore: git apply experiments/removed/MOJOLEARN_IDN_SPMV_LANES_OFF.patch; record in docs/TOMBSTONES.md.
comptime IDN_SPMV_LANES = GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL


#: lane fix-c1-cluster (2026-10-04, audit B10): the Laplacian's degree fold
#: (`laplacian.mojo::degree_kernel`, DEVIATION 776) in the same lanes and
#: tree as the matvec: row `r`'s lane `l` sums the entries `lo + l, lo + l +
#: SPMV_LANES, ...` ascending, `acc = ftz(acc + val)` from `+0.0`, then
#: `part[l] = ftz(part[l] + part[l + w])` for `w = SPMV_LANES / 2, ..., 1`.
#: It replaces one thread walking the whole row. BITS: a weighted
#: precomputed graph's degrees move (the kNN graph's 0.5 / 1 values sum the
#: same in any order); the device (`degree_lanes_kernel`) and the host column
#: (`spectral_oracle.mojo::host_laplacian`) both read this constant.
# TOMBSTONE: MOJOLEARN_IDN_ALL_OFF (old path, no A/B recorded beside the default) deleted 2026-11-15 by lane/expdel2-13-cluster (non-owner; owner expdel2-37-series); code recoverable at c089be9c4a.
# Restore: git apply experiments/removed/MOJOLEARN_IDN_ALL_OFF.expdel2-13-cluster.patch; record in docs/TOMBSTONES.md.
# TOMBSTONE: MOJOLEARN_IDN_LAP_DEGREE_LANES_OFF (old path, no A/B recorded beside the default) deleted 2026-11-15 by lane/expdel2-13-cluster; code recoverable at 1cc466b0db.
# Restore: git apply experiments/removed/MOJOLEARN_IDN_LAP_DEGREE_LANES_OFF.patch; record in docs/TOMBSTONES.md.
comptime IDN_LAP_DEGREE_LANES = GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL
