"""Completion-timed public Mamba2 prefill backward price card."""

from std.memory import bitcast
from std.time import perf_counter_ns

from mamba.checks.mamba2_fixture import m2_case_weights, m2_case_x
from mamba.impl.modules.mamba2_prefill_backward import (
    Mamba2PrefillGradients,
    mamba2_prefill_backward,
)

comptime B = 2
comptime L = 770
comptime LAYERS = 12
comptime REPS = 3


def fold_hash(mut h: UInt64, xs: List[Float32]) -> UInt64:
    for x in xs:
        h = (h ^ UInt64(bitcast[DType.uint32](x))) * UInt64(0x100000001B3)
    return h


def grad_hash(g: Mamba2PrefillGradients) -> UInt64:
    var h = UInt64(0xCBF29CE484222325)
    h = fold_hash(h, g.x)
    h = fold_hash(h, g.block_norm_weight)
    h = fold_hash(h, g.in_proj_weight)
    h = fold_hash(h, g.conv1d_weight)
    h = fold_hash(h, g.conv1d_bias)
    h = fold_hash(h, g.dt_bias)
    h = fold_hash(h, g.A_log)
    h = fold_hash(h, g.D)
    h = fold_hash(h, g.norm_weight)
    return fold_hash(h, g.out_proj_weight)


def run_weighted() raises -> UInt64:
    var w = m2_case_weights(9)
    var x = m2_case_x(9)
    var dy = x.copy()
    var h = UInt64(0)
    for _ in range(LAYERS):
        var g = mamba2_prefill_backward(
            w, x, dy, B, L, 0.0, bitcast[DType.float32](UInt32(0x7F800000))
        )
        h = grad_hash(g)
    return h


def main() raises:
    print("MAMBA2_BWD warm_hash=", run_weighted())
    for rep in range(REPS):
        var t0 = perf_counter_ns()
        var h = run_weighted()
        var t1 = perf_counter_ns()
        print(
            "MAMBA2_BWD rep=", rep, " layers=", LAYERS,
            " ms=", Float64(t1 - t0) / 1e6, " hash=", h,
        )
