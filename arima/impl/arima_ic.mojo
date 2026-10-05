# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""ARIMA's AIC and BIC, `-2 loglike + penalty` in binary64, on the device
(lane cpu3-seq, 2026-10-04).

DEVIATION 991 computed them in the Python wrapper's native host helper
(`ic_running_min_f64`) from the read-back float32 log-likelihood. The fit
now writes them: `arima_ic_kernel` runs on the device over the resident
log-likelihood (every vendor, Apple included, which has no float64 unit),
and the host column (`bindings/_mojolearn_arima_host.mojo`) calls the same
`arima_ic_bits` per series.

THE ARITHMETIC. `checks/soft_f64.mojo` (integer binary64, correctly
rounded, the IEEE result): the float32 log-likelihood widened exactly, times
-2 (exact: a power of two), plus the penalty, ONE rounding. That is the
float64 `-2.0 * Float64(ll) + pen` the host helper computed, bit for bit,
for every non-NaN log-likelihood (the product is exact, so contraction
cannot move it). BITS: a NaN criterion is now the canonical quiet NaN
`SF64_NAN` on every column (the hardware form kept the log-likelihood's
payload); no other value moves.

The penalties are the wrapper's scalars (2N; log(T) N, T after
differencing), handed over as binary64 bit patterns.
"""

from std.gpu import block_dim, block_idx, thread_idx
from std.math import inf

from checks.soft_f64 import sf64_add, sf64_from_f32, sf64_mul

#: -2.0 as binary64
comptime ARIMA_IC_NEG_TWO = UInt64(0xC000000000000000)
comptime ARIMA_IC_TPB = 128


@always_inline
def arima_ic_bits(ll: Float32, pen: UInt64) -> UInt64:
    """`-2 * ll + pen` in binary64, one rounding (the product is exact)."""
    return sf64_add(sf64_mul(ARIMA_IC_NEG_TWO, sf64_from_f32(ll)), pen)


def arima_ic_kernel(
    d_ic: MutPointer[UInt64, MutAnyOrigin],
    d_ll: MutPointer[Float32, MutAnyOrigin],
    info0: MutPointer[Int32, MutAnyOrigin],
    info1: MutPointer[Int32, MutAnyOrigin],
    batch_size_in: Int32,
    pen_aic: UInt64,
    pen_bic: UInt64,
):
    """One thread per series: the log-likelihood at the fitted point (the
    constant -inf when either Kalman refusal code is set, `batched_loglike_x`'s
    `infeasible_inf` rule), then `d_ic[b]` = AIC and `d_ic[bs + b]` = BIC as
    binary64 bit patterns."""
    var b = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var bs = Int(batch_size_in)
    if b >= bs:
        return
    var v = d_ll.unsafe_load(b)
    if info0.unsafe_load(b) != Int32(0) or info1.unsafe_load(b) != Int32(0):
        v = -inf[DType.float32]()
    d_ic.unsafe_store(b, arima_ic_bits(v, pen_aic))
    d_ic.unsafe_store(bs + b, arima_ic_bits(v, pen_bic))
