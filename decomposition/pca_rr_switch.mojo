# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The one switch `eig_and_truncate` (decomposition/impl/linalg/detail/pca.mojo)
and its host column `host_eig_and_truncate` (decomposition/host/pca_oracle.mojo)
both read, so the device and the host change eigensolver together."""
from std.sys.compile import is_defined
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
from x_decomp.rr import RR_EIGH_SWEEPS

#: lane idn-shap-pca (2026-10-04), the IDENTICAL default on every column
#: (NVIDIA, AMD, Apple, host): PCA's and TruncatedSVD's covariance / Gram
#: eigendecomposition runs the round-robin two-sided Jacobi (x_decomp/rr.mojo,
#: x_decomp/jacobi_par.mojo: m - 1 rounds a sweep, the m / 2 disjoint
#: rotations of a round in parallel across the GPU) in place of the one-block
#: cyclic `jacobi_eigh_kernel` (n (n - 1) / 2 serial rotations a sweep on 256
#: threads). BITS CHANGE (a different rotation order): the device kernels and
#: the host column switch together here. -D MOJOLEARN_PCA_RR_EIGH_OFF
#: restores the cyclic solver on both (the A/B arm; pass it to the device
#: AND the host binding builds). FAST builds are unchanged.
comptime PCA_RR_EIGH = (
    GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL and not (is_defined["MOJOLEARN_PCA_RR_EIGH_OFF"]() or is_defined["MOJOLEARN_IDN_ALL_OFF"]())
)

#: The round-robin solve's sweep budget in PCA / TruncatedSVD, the device and
#: the host column alike. RECORDED REASON (lane idn-all, 2026-10-04): it is
#: RR_EIGH_SWEEPS (60), the budget the SAME solver (same rounds, same test,
#: same tolerance) runs under in x_decomp on every column since cgr-decomp
#: (2026-10-03, verified nv = amd = M2 = host), so PCA cannot refuse a matrix
#: `x_decomp` eigh accepts. The cyclic solver's 15 was measured for a
#: different rotation order and does not transfer; no round-robin sweep count
#: at PCA's shapes (220 columns) is recorded, so a tighter number would be a
#: guess. The budget moves no bit of a converged solve (the sweeps run are
#: the solve's own) and, with PCA_RR_FLAG_TEST, costs nothing: the device
#: stops at the converged test. A solve that needs more raises, as the cyclic
#: one did.
comptime PCA_RR_SWEEPS = RR_EIGH_SWEEPS

#: lane idn-all: the device's convergence flag is read between sweeps (six
#: flag words, one wait per sweep; no matrix data crosses) and the enqueuing
#: stops at the converged test, instead of enqueuing every budgeted sweep
#: (about 2 (n - 1) launches each: 438 at 220 columns) as no-ops. Device
#: only (the host column's loop already stops there); moves no bit.
#: -D MOJOLEARN_PCA_RR_FLAG_TEST_OFF restores the enqueue-everything form.
comptime PCA_RR_FLAG_TEST = PCA_RR_EIGH and not (
    is_defined["MOJOLEARN_PCA_RR_FLAG_TEST_OFF"]() or is_defined["MOJOLEARN_IDN_ALL_OFF"]()
)
