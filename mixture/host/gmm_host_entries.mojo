# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Typed host entries behind `bindings/mixture_host_scoring.mojo` and
`bindings/_mojolearn_mixture_host.mojo` (lane thin-trees-metrics,
2026-10-21: the sampling body and `_gmm_fit_run` moved word for word out of
those shells; no Python here)."""
from std.memory import bitcast

from gemm.host.identical_gemm import GEMM_ORACLE_HOST_SABOTAGE
from mixture.checks.sample import gmm_sample_host
from mixture.host.gmm_host_oracle import GmmHostParams, gmmh_fit


def gmm_sample_into(
    weights: List[Float32],
    means: List[Float32],
    precisions: List[Float32],
    k: Int,
    d: Int,
    n: Int,
    seed: UInt64,
    xp: MutPointer[Float32, MutUntrackedOrigin],
    yp: MutPointer[Int32, MutUntrackedOrigin],
) raises:
    """`sample(n_samples)`: `gmm_sample_host`, then the `n * d` cells to
    `xp` (with the sample sabotage arm) and the `n` components to `yp`."""
    var labels = List[Int32](length=max(0, n), fill=Int32(0))
    var x = gmm_sample_host(weights, means, precisions, k, d, n, seed, labels)
    for i in range(n * d):
        var v = x[i]
        comptime if GEMM_ORACLE_HOST_SABOTAGE:
            # THE SAMPLE SABOTAGE ARM (the neighbors and density inference
            # lane, 2026-09-15): the lowest bit of every sampled cell, wrong
            # on purpose. gmm_sample_host reads no GEMM leaf, so the
            # descending-leaf arm cannot reach a sample drawn from a saved
            # model; without this the host sabotage build could not be seen
            # to move gmm-sample inference.
            v = bitcast[DType.float32](bitcast[DType.uint32](v) ^ UInt32(1))
        xp.unsafe_store(i, v)
    for i in range(n):
        yp.unsafe_store(i, labels[i])


def _gmm_fit_run(
    x: List[Float32],
    n: Int,
    d: Int,
    params: GmmHostParams,
    wp: MutPointer[Float32, MutUntrackedOrigin],
    mp: MutPointer[Float32, MutUntrackedOrigin],
    cp: MutPointer[Float32, MutUntrackedOrigin],
    pp: MutPointer[Float32, MutUntrackedOrigin],
    lp: MutPointer[Float32, MutUntrackedOrigin],
    sp: MutPointer[Float64, MutUntrackedOrigin],
) raises -> Int:
    var model = gmmh_fit(x, n, d, params)
    var k = model.n_components
    for i in range(k):
        wp.unsafe_store(i, model.weights[i])
        lp.unsafe_store(i, model.log_det_chol[i])
    for i in range(k * d):
        mp.unsafe_store(i, model.means[i])
    for i in range(k * d * d):
        cp.unsafe_store(i, model.covariances[i])
        pp.unsafe_store(i, model.precisions_cholesky[i])
    sp.unsafe_store(0, Float64(model.n_iter))
    sp.unsafe_store(1, Float64(1 if model.converged else 0))
    sp.unsafe_store(2, Float64(model.lower_bound))
    var n_iter = model.n_iter
    _ = model^
    return n_iter
