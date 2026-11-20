# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The standard normal distribution in Mojo (lane py-runtime round 3):
`_portable_math.erfc` (fdlibm's s_erf.c erfc), `normal_cdf`,
`normal_inv_cdf` (Wichura's AS241) and IterativeImputer's
truncated-normal posterior draw (`_truncnorm_host`), which ran per missing
entry in Python.

SAME BITS AS THE PYTHON THEY REPLACE, by construction: the same binary64
operations in the same order, every product that meets an addition spelled
`pinned_mul_f64` (Python never fuses; Mojo's default fp-mode would), the
pinned exp and log (`checks/numerics.mojo` portable_exp64 / portable_log64,
which packaging/portable_math/portable_math.c, the Python side's library,
translates) and the correctly rounded sqrt. The draw stream is the same
splitmix64 words, consumed only by the entries that draw.

The draw runs on the host column on every install: the predictions come
from the user's estimator (IterativeImputer's user-estimator route, a
CPU-only input step), and it is float64 transcendental work Apple's GPU
cannot do."""
from std.python import Python, PythonObject

# RUN: the typed draw (bindings/normal_dist_helpers_run.mojo).
from bindings.normal_dist_helpers_run import truncnorm_draws


def truncnorm_draws_binding(
    mus_addr: PythonObject, sig_addr: PythonObject, m: PythonObject, bounds: PythonObject,
    state: PythonObject, out_addr: PythonObject,
) raises -> PythonObject:
    """IterativeImputer's `_truncnorm_host` for m entries: mu beyond a bound
    -> the bound, sigma <= 0 (or NaN) -> mu, else the inversion of the
    truncated normal at a 53-bit splitmix64 uniform (only drawing entries
    advance the stream). bounds = [lo, hi] (+-inf allowed); state = the
    stream's 64-bit word as [high 32 bits, low 32 bits]. Writes m float64
    values; returns the new state as [high, low]."""
    var n = Int(py=m)
    var lo = Float64(py=bounds[0])
    var hi = Float64(py=bounds[1])
    var s = (UInt64(Int(py=state[0])) << 32) | UInt64(Int(py=state[1]))
    s = truncnorm_draws(Int(py=mus_addr), Int(py=sig_addr), n, lo, hi, s, Int(py=out_addr))
    var out = Python.list()
    out.append(PythonObject(Int(s >> 32)))
    out.append(PythonObject(Int(s & UInt64(0xFFFFFFFF))))
    return out
