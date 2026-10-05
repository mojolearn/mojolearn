# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""PCA's rank selection for a float `n_components` (and the noise variance of
the dropped tail) after the fit (lane py-runtime, 2026-10-05). It ran in
Python (`decomposition.PCA.fit`); Python is glue only, so it is host Mojo
here, shared word for word by the GPU binding and the CPU column. It stays
on the host on every vendor: `nc` binary64 additions, and Apple has no
float64 on the device.

SAME BITS AS THE PYTHON IT REPLACES: the cumulative explained-variance ratio
is one ascending binary64 chain over the float32 ratios, the first index
where it passes `frac` (strictly) keeps `i + 1` components (all when none
does); the noise variance is the ascending binary64 sum of the dropped
float32 explained variances divided by their count, rounded once to float32
(0.0 when nothing is dropped)."""


def pca_rank_finish(
    ratio: MutPointer[Float32, MutUntrackedOrigin],
    ev: MutPointer[Float32, MutUntrackedOrigin],
    nc: Int,
    keep_in: Int,
    frac: Float64,
    mut noise: Float64,
) -> Int:
    """The kept component count (`keep_in` when it is >= 0, the MLE rank,
    else the float-`n_components` rank) and, in `noise`, the dropped tail's
    noise variance."""
    var keep = nc
    if keep_in >= 0:
        keep = keep_in
    else:
        var cum: Float64 = 0.0
        for i in range(nc):  # small-loop(nc: component count): one ratio per fitted component
            cum += Float64(ratio[i])
            if cum > frac:
                keep = i + 1
                break
    if keep > nc:
        keep = nc
    var tail: Float64 = 0.0
    for i in range(keep, nc):  # small-loop(nc: component count): the dropped components' variances
        tail += Float64(ev[i])
    noise = Float64(Float32(tail / Float64(nc - keep))) if nc > keep else 0.0
    return keep
