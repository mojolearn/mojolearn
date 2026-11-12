# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""RUN half of `bindings/_mojolearn_mixture.mojo` (wave 8 run split,
2026-11-12): the scoring entries' model rebuild from typed values. The door
(`_rebuild_model`) reads the Python address and parameter lists into ints and
floats and checks their lengths; this file copies the model buffers and builds
the model. Typed Mojo only, no Python import (tools/python_free_dirs.sh). Same
reads in the same order as before the split, so the bits are unchanged."""
from core.hostptr import read_f32
from mixture.estimator import GaussianMixtureModel


def rebuild_model_run(
    k: Int,
    d: Int,
    n_iter: Int,
    converged: Bool,
    lower_bound: Float32,
    weights_addr: Int,
    means_addr: Int,
    precisions_addr: Int,
    log_det_addr: Int,
) raises -> GaussianMixtureModel:
    """The model from typed scoring-entry words: weights (k), means (k*d),
    precisions_chol (k*d*d) and log_det_chol (k) are copied from their
    addresses; covariances_ stays empty."""
    var weights = read_f32(weights_addr, max(0, k))
    var means = read_f32(means_addr, max(0, k * d))
    # covariances_ (k*d*d) is read by no scoring, predict or sample kernel
    # (they use precisions_chol, means, weights and log_det_chol): it is not
    # copied (p3 rank 296; the address stays in the entry's word layout).
    var covariances = List[Float32]()
    var precisions = read_f32(precisions_addr, max(0, k * d * d))
    var log_det = read_f32(log_det_addr, max(0, k))
    return GaussianMixtureModel(
        k,
        d,
        weights^,
        means^,
        covariances^,
        precisions^,
        log_det^,
        n_iter,
        converged,
        lower_bound,
    )
