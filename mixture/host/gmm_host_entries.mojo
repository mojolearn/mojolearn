# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Typed host entries behind `bindings/mixture_host_scoring.mojo`'s Python
doors (lane thin-trees-metrics, 2026-10-21: moved word for word out of that
shell; no Python here)."""
from std.memory import bitcast

from gemm.host.identical_gemm import GEMM_ORACLE_HOST_SABOTAGE
from mixture.checks.sample import gmm_sample_host


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
