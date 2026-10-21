# SPDX-License-Identifier: Apache-2.0
"""Typed Holt-Winters host fit body moved word for word from bindings/_mojolearn_tsa_host.mojo."""
from std.math import isfinite
from std.memory import bitcast
from bindings.hostptr import read_f32
from holtwinters.host.hw_oracle import (
    HW_ORACLE_HOST_SABOTAGE,
    oracle_fit,
)
from holtwinters.impl.runner import (
    holtwinters_validate_data,
    holtwinters_validate_params,
)
from holtwinters.impl.tsa.holtwinters_params import seasonal_from_name


def holtwinters_fit_host_into(
    data_address: Int, n: Int, batch_size: Int, frequency: Int, start_periods: Int, eps: Float32,
    init_method: Int, sname: String,
    cp: MutPointer[Float32, MutUntrackedOrigin], sp: MutPointer[Float32, MutUntrackedOrigin],
    fp: MutPointer[Int32, MutUntrackedOrigin],
) raises -> Int:
    var components_len = 0
    # `holtwinters_fit_ptr`'s guards on the extents the read dereferences
    # (DEVIATION 931), in its words, before the read.
    if batch_size < 1:
        raise Error(
            "holtwinters fit: batch_size must be >= 1 (batch_size="
            + String(batch_size) + "); the input read is batch_size * n cells"
        )
    if n < 1:
        raise Error(
            "holtwinters fit: n must be >= 1 (n=" + String(n)
            + "); the input read is batch_size * n cells"
        )
    var cells = batch_size * n
    if cells < 0 or cells // batch_size != n:
        raise Error(
            "holtwinters fit: batch_size * n overflowed (batch_size="
            + String(batch_size) + ", n=" + String(n) + ")"
        )
    var data = read_f32(data_address, cells)
    # `holtwinters_fit_host_traced`'s validation, in its order.
    var st = seasonal_from_name(sname)
    holtwinters_validate_params(n, batch_size, frequency, start_periods, eps)
    holtwinters_validate_data(data, n, batch_size, st)
    # THE ONE CALL THAT COMPUTES ANYTHING. trace_iters is a record, not
    # an arithmetic input; 0 keeps no per-iteration trace.
    var fitted = oracle_fit[DType.float32](
        data, n, batch_size, frequency, start_periods, st, eps, 0,
        init_method=init_method,
    )
    components_len = len(fitted.level)
    if (
        components_len != (n - frequency) * batch_size
        or len(fitted.trend) != components_len
        or len(fitted.season) != components_len
        or len(fitted.sse) != batch_size
        or len(fitted.alpha) != batch_size
        or len(fitted.beta) != batch_size
        or len(fitted.gamma) != batch_size
        or len(fitted.niter) != batch_size
        or len(fitted.criterion) != batch_size
    ):
        raise Error(
            "holtwinters_fit: the host oracle returned components of an"
            " unexpected length; nothing written"
        )
    for i in range(components_len):
        cp[i] = fitted.level[i]
        cp[components_len + i] = fitted.trend[i]
        cp[2 * components_len + i] = fitted.season[i]
    for b in range(batch_size):
        sp[b] = fitted.sse[b]
        sp[batch_size + b] = fitted.alpha[b]
        sp[2 * batch_size + b] = fitted.beta[b]
        sp[3 * batch_size + b] = fitted.gamma[b]
        fp[b] = Int32(fitted.niter[b])
        fp[batch_size + b] = Int32(fitted.criterion[b])
    comptime if HW_ORACLE_HOST_SABOTAGE:
        # THE FITTED-STATE ARM (lane/inference-holtwinters, 2026-09-15):
        # the split SSE multiply-add leaves the fitted bytes unchanged on
        # some fixtures (denormal, denormal_ftz, wide), so a saved model's
        # file could not move under the negative control. Every finite
        # per-series float (sse, alpha, beta, gamma) also has its lowest
        # bit flipped, a value perturbation, so every saved file moves.
        # The components are left alone: a flip there, followed by the
        # forecast binding's own output flip, restored the forecast's
        # bytes on two fixtures (the third x86 pod, DIVERGENT=16).
        for i in range(4 * batch_size):
            if isfinite(sp[i]):
                sp[i] = bitcast[DType.float32](bitcast[DType.uint32](sp[i]) ^ UInt32(1))
    return components_len
