# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""MinCovDet's resident fit drains at most once a C-step (lane mcd-per-step-cost, 2026-11-14).

2000 x 16 fixture rows (splitmix64 draws mapped to [-1, 1)), sklearn's default support size, the
plan the binding computes (bindings/mcd_run.mojo). `fast_mcd_dev` returns the resident kit's drain
count (`DKit.sync`) and its C-step count (log determinants evaluated); the check asserts
drains <= C-steps + 3: one wait a C-step for the log determinant the loop branches on (the narrow
pinvh's status and the selections' NaN words ride that wait), plus the final write. It also checks
the support holds h rows.

Scope: counts `DKit.sync` only. The wide (n > 32) eigh's per-sweep waits and the pool's trims run
`ctx.synchronize` directly and are not counted; at d = 16 the C-step runs neither. GPU build only;
no timing, no identity.
"""
from checks.fixture_rng import splitmix64_next
from x_decomp.cells import F32Ptr, I32Ptr
from x_decomp.kit import Mat
from x_decomp.kit_device import fast_mcd_dev


def require(condition: Bool, message: String) raises:
    if not condition:
        raise Error(message)


def main() raises:
    var n = 2000
    var d = 16
    var h = (n + d + 2) // 2  # ceil(0.5 (n + d + 1)), sklearn's n_support
    var X = Mat(n, d)
    var state = UInt64(20261114)
    for i in range(n * d):
        var u = splitmix64_next(state) >> 40  # 24 bits
        X.d[i] = Float32(Float64(Int(u)) / 8388608.0 - 1.0)
    # the binding's plan (bindings/mcd_run.mojo mcd_fit_run, n > 500)
    var n_sub = n // 300
    var n_ss = n // n_sub
    var n_m = min(1500, n)
    var share = Float64(h) / Float64(n)
    var h_sub = Int(Float64(n_ss) * share)
    if Float64(h_sub) < Float64(n_ss) * share:
        h_sub += 1
    var h_m = Int(Float64(n_m) * share)
    if Float64(h_m) < Float64(n_m) * share:
        h_m += 1
    var plan: List[Int] = [n, d, h, 7, n_sub, n_ss, h_sub, max(10, 500 // n_sub), n_m, h_m, 10 if n > 1500 else 1]
    var loc = List[Float32](length=d, fill=Float32(0))
    var cov = List[Float32](length=d * d, fill=Float32(0))
    var sup = List[Int32](length=n, fill=Int32(0))
    var dist = List[Float32](length=n, fill=Float32(0))
    var steps = fast_mcd_dev(
        X, plan,
        F32Ptr(unsafe_from_address=Int(loc.unsafe_ptr())),
        F32Ptr(unsafe_from_address=Int(cov.unsafe_ptr())),
        I32Ptr(unsafe_from_address=Int(sup.unsafe_ptr())),
        F32Ptr(unsafe_from_address=Int(dist.unsafe_ptr())),
    )
    var support = 0
    for i in range(n):
        support += Int(sup[i])
    print("MCD_STEP_COST n=", n, " d=", d, " h=", h, " csteps=", steps.csteps, " drains=", steps.drains,
          " support=", support)
    require(steps.csteps > 0, "no C-step counted (the FAST batched route ran, or the counter is not wired)")
    require(steps.drains <= steps.csteps + 3, "more than one drain a C-step: " + String(steps.drains)
            + " drains for " + String(steps.csteps) + " C-steps")
    require(support == h, "the support does not hold h rows")
    _ = X^
    _ = loc^
    _ = cov^
    _ = dist^
    print("MCD_STEP_COST drains_per_cstep=pass")
