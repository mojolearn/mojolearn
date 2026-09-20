# SPDX-License-Identifier: Apache-2.0
"""CPU Samba RMSNorm row-schedule price and bitwise-output receipt."""

from std.memory import bitcast
from std.time import perf_counter_ns

from core.identity_trace import FNV_OFFSET, FNV_PRIME
from training.host.samba_ops_oracle import (
    HOST_RMS_FORCE_SERIAL,
    host_samba_rms_norm_backward,
    host_samba_rms_norm_forward,
)


def _mix(h: UInt64, v: Float32) -> UInt64:
    var bits = UInt64(bitcast[DType.uint32](v))
    var out = h
    for b in range(4):
        out = (out ^ ((bits >> UInt64(8 * b)) & UInt64(0xFF))) * FNV_PRIME
    return out


def _hash(h: UInt64, x: List[Float32]) -> UInt64:
    var out = h
    for i in range(len(x)):
        out = _mix(out, x[i])
    return out


def _value(i: Int, salt: Int) -> Float32:
    var z = UInt64(i + 1) * UInt64(0x9E3779B97F4A7C15) + UInt64(salt)
    z = (z ^ (z >> 30)) * UInt64(0xBF58476D1CE4E5B9)
    z = (z ^ (z >> 27)) * UInt64(0x94D049BB133111EB)
    z = z ^ (z >> 31)
    return Float32(Int((z >> 40) & UInt64(0xFFFF))) / Float32(
        32768.0
    ) - Float32(1.0)


def main() raises:
    var m = 2048
    var dm = 512
    var cells = m * dm
    var x = List[Float32](length=cells, fill=Float32(0.0))
    var dy = List[Float32](length=cells, fill=Float32(0.0))
    var w = List[Float32](length=dm, fill=Float32(0.0))
    for i in range(cells):
        x[i] = _value(i, 11)
        dy[i] = _value(i, 17)
    for j in range(dm):
        w[j] = Float32(1.0) + _value(j, 23) * Float32(0.25)

    # Warm the worker pool and compiled math before taking five samples.
    _ = host_samba_rms_norm_forward(x, w, m, dm, Float32(1.0e-5))
    _ = host_samba_rms_norm_backward(dy, x, w, m, dm, Float32(1.0e-5))
    for rep in range(5):
        var t0 = perf_counter_ns()
        var y = host_samba_rms_norm_forward(x, w, m, dm, Float32(1.0e-5))
        var t1 = perf_counter_ns()
        var bwd = host_samba_rms_norm_backward(dy, x, w, m, dm, Float32(1.0e-5))
        var t2 = perf_counter_ns()
        var h = _hash(FNV_OFFSET, y)
        h = _hash(h, bwd[0])
        h = _hash(h, bwd[1])
        print(
            "RMS_CPU_PRICE",
            "serial" if HOST_RMS_FORCE_SERIAL else "parallel",
            rep,
            Float64(t1 - t0) / 1.0e6,
            Float64(t2 - t1) / 1.0e6,
            h,
        )
