# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""HOST COLUMN of the UMAP spectral-init post-pass.

The device form is `umap/spectral_init.mojo::_spectral_post_pass_device`
(`spectral_post_pivot_kernel`, `spectral_post_scale_kernel`). This is the
same arithmetic through the same seams, for the host oracle
(`umap_oracle.mojo::host_umap_spectral_initialize`): the first row of
largest magnitude per column is the pivot, `scale = identical_div(10, peak)`
negated when the pivot is negative, and every cell becomes
`ftz(identical_mul(v, scale))`."""

from std.memory import bitcast

from checks.numerics import ftz, identical_div, identical_mul


def host_spectral_post_pass(
    mut embedding: List[Float32], n_samples: Int, n_components: Int
) raises:
    """Row-major `n_samples x n_components`, in place. Refusals in the device
    route's order: column by column, a non-finite value before a zero peak."""
    for c in range(n_components):
        var pivot = 0
        var peak_bits = UInt32(0)
        var bad = False
        for i in range(n_samples):
            var bits = bitcast[DType.uint32](embedding[i * n_components + c])
            if ((bits >> UInt32(23)) & UInt32(0xFF)) == UInt32(0xFF):
                bad = True
            var mag = bits & UInt32(0x7FFFFFFF)
            if mag > peak_bits:
                peak_bits = mag
                pivot = i
        if bad:
            raise Error("UMAP spectral solver returned a non-finite value")
        if peak_bits == UInt32(0):
            raise Error("UMAP spectral solver returned a zero component")
        var scale = identical_div(Float32(10.0), bitcast[DType.float32](peak_bits))
        var pivot_bits = bitcast[DType.uint32](embedding[pivot * n_components + c])
        if (pivot_bits >> UInt32(31)) != UInt32(0):
            scale = -scale
        for i in range(n_samples):
            var at = i * n_components + c
            embedding[at] = ftz(identical_mul(embedding[at], scale))
