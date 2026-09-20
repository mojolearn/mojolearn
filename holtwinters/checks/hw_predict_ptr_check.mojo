# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn.
"""Bitwise gate for packed-pointer Holt-Winters host inference."""

from std.memory import bitcast

from holtwinters.host.hw_predict import (
    hw_forecast_from_state,
    hw_forecast_from_state_ptr,
    hw_predict_in_sample,
    hw_predict_in_sample_ptr,
)


def _same(a: List[Float32], b: List[Float32], label: String) raises:
    if len(a) != len(b):
        raise Error(label + ": length mismatch")
    for i in range(len(a)):
        if bitcast[DType.uint32](a[i]) != bitcast[DType.uint32](b[i]):
            raise Error(label + ": bit mismatch at " + String(i))


def main() raises:
    var n = 41
    var batch_size = 7
    var frequency = 6
    var components_len = (n - frequency) * batch_size
    var packed = List[Float32](length=3 * components_len, fill=Float32(0.0))
    for i in range(3 * components_len):
        # Exact binary values include signs and distinct component blocks.
        packed[i] = Float32((i % 37) - 18) * Float32(0.03125)
    var level = List[Float32](length=components_len, fill=Float32(0.0))
    var trend = List[Float32](length=components_len, fill=Float32(0.0))
    var season = List[Float32](length=components_len, fill=Float32(0.0))
    for i in range(components_len):
        level[i] = packed[i]
        trend[i] = packed[components_len + i]
        season[i] = packed[2 * components_len + i]
    for additive in [True, False]:
        var reference_fc = hw_forecast_from_state[DType.float32](
            level, trend, season, n, batch_size, frequency, additive, 19
        )
        var pointer_fc = hw_forecast_from_state_ptr(
            packed.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin](),
            components_len,
            n,
            batch_size,
            frequency,
            additive,
            19,
        )
        _same(reference_fc, pointer_fc, "forecast")
        var reference_in = hw_predict_in_sample(
            level, trend, season, n, batch_size, frequency, additive, 0, n
        )
        var pointer_in = hw_predict_in_sample_ptr(
            packed.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin](),
            components_len,
            n,
            batch_size,
            frequency,
            additive,
            0,
            n,
        )
        _same(reference_in, pointer_in, "predict")
    print("HOLTWINTERS PACKED POINTER PREDICT PASS")
