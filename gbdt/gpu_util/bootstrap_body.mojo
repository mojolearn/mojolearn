# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The bootstrap's per-row draw and its seed, one body (lane gbdt-leaf-body,
2026-11-23) for the device kernels (`gbdt/gpu_util/kernel/bootstrap.mojo`:
`bootstrap_kernel`, `bootstrap_seed_fill_kernel`) and the CPU column
(`gbdt/host/gbdt_oracle.mojo`: `_bootstrap_pass`, `gbdt_bootstrap_seeds`),
which restated both statement for statement. GPU-free, so the host bindings
import it."""
from checks.numerics import identical_log, identical_pow
from gbdt.gpu_util.kernel.random_gen import next_poisson_f, next_uniform_f

comptime BOOTSTRAP_BODY_BAYESIAN = 0
comptime BOOTSTRAP_BODY_BERNOULLI = 1
comptime BOOTSTRAP_BODY_POISSON = 2


@always_inline
def bootstrap_seed(base_seed: UInt64, i: Int) -> UInt64:
    """Seed `i` of `TGpuAwareRandom::CreateSeeds`: splitmix64 of `base_seed +
    (i + 1) * 0x9E3779B97F4A7C15`."""
    var z = base_seed + UInt64(i + 1) * UInt64(0x9E3779B97F4A7C15)
    z = (z ^ (z >> 30)) * UInt64(0xBF58476D1CE4E5B9)
    z = (z ^ (z >> 27)) * UInt64(0x94D049BB133111EB)
    return z ^ (z >> 31)


@always_inline
def bootstrap_draw(kind: Int, param: Float32, mut s: UInt64) -> Float32:
    """One row's bootstrap weight, the seed `s` advanced: their three draws
    (`bootstrap.cu`), BAYESIAN `-log(u + 1e-20)` raised to the temperature
    unless 1 (`:41-47`), BERNOULLI `u < subsample` (`:56-60`), POISSON
    `NextPoisson(lambda)` (`:12-16`). The device passes its comptime kind, so
    the branch folds away there."""
    if kind == BOOTSTRAP_BODY_BAYESIAN:
        var draw = next_uniform_f(s)
        s = draw[1]
        var tmp = -identical_log(draw[0] + Float32(1e-20))
        if param != Float32(1.0):
            return identical_pow(tmp, param)
        return tmp
    elif kind == BOOTSTRAP_BODY_BERNOULLI:
        var draw = next_uniform_f(s)
        s = draw[1]
        return Float32(1.0) if draw[0] < param else Float32(0.0)
    var draw = next_poisson_f(s, param)
    s = draw[1]
    return draw[0]
