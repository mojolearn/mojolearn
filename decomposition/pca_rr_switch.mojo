# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The one switch `eig_and_truncate` (decomposition/impl/linalg/detail/pca.mojo)
and its host column `host_eig_and_truncate` (decomposition/host/pca_oracle.mojo)
both read, so the device and the host change eigensolver together."""
from std.sys.compile import is_defined
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL

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
    GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL and not is_defined["MOJOLEARN_PCA_RR_EIGH_OFF"]()
)
