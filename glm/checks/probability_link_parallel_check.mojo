# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn.
"""CPU softmax row parallelism preserves every probability bit.

The production host links are `core/classical_host_predict.mojo`'s
`host_qn_sigmoid_into` / `host_qn_softmax_into` (the CPU binding's
predict_proba; the GPU binding computes them on the device). The duplicate
`glm/estimator.mojo` copies were deleted (cpu-gpu-cleanup c-linear)."""

from std.time import perf_counter_ns

from checks.numerics import identical_exp64
from core.classical_host_predict import host_qn_sigmoid_into, host_qn_softmax_into
from core.host_predict_threads import HostF32Ptr, HostF64Ptr, host_predict_task_count




def _sigmoid_serial(scores: List[Float32], mut dst: List[Float64]):
    for i in range(len(scores)):
        var z = Float64(scores[i])
        var p = 1.0 / (1.0 + identical_exp64(-z))
        dst[2 * i] = 1.0 - p
        dst[2 * i + 1] = p


def _softmax_serial(scores: List[Float32], mut dst: List[Float64], rows: Int, classes: Int):
    for i in range(rows):
        var base = i * classes
        var m = Float64(scores[base])
        for c in range(1, classes):
            var v = Float64(scores[base + c])
            if v > m:
                m = v
        var s = 0.0
        for c in range(classes):
            s = s + identical_exp64(Float64(scores[base + c]) - m)
        for c in range(classes):
            dst[base + c] = identical_exp64(Float64(scores[base + c]) - m) / s


def main() raises:
    var rows = 1_000_000
    var classes = 8
    var binary = List[Float32](length=rows, fill=0.0)
    var multi = List[Float32](length=rows * classes, fill=0.0)
    for i in range(rows):
        binary[i] = Float32((i % 257) - 128) * Float32(0.03125)
    for i in range(rows * classes):
        multi[i] = Float32((i * 17 % 509) - 254) * Float32(0.015625)

    var bs = List[Float64](length=2 * rows, fill=0.0)
    var bp = List[Float64](length=2 * rows, fill=0.0)
    var ms = List[Float64](length=rows * classes, fill=0.0)
    var mp = List[Float64](length=rows * classes, fill=0.0)

    var t0 = perf_counter_ns()
    _sigmoid_serial(binary, bs)
    var serial_sigmoid_ms = Float64(perf_counter_ns() - t0) / 1.0e6
    t0 = perf_counter_ns()
    host_qn_sigmoid_into(rebind[HostF32Ptr](binary.unsafe_ptr()), rebind[HostF64Ptr](bp.unsafe_ptr()), rows,
                         host_predict_task_count(rows))
    var parallel_sigmoid_ms = Float64(perf_counter_ns() - t0) / 1.0e6
    for i in range(2 * rows):
        if bs[i] != bp[i]:
            raise Error("parallel sigmoid moved output cell " + String(i))

    t0 = perf_counter_ns()
    _softmax_serial(multi, ms, rows, classes)
    var serial_softmax_ms = Float64(perf_counter_ns() - t0) / 1.0e6
    t0 = perf_counter_ns()
    host_qn_softmax_into(rebind[HostF32Ptr](multi.unsafe_ptr()), rebind[HostF64Ptr](mp.unsafe_ptr()), rows, classes,
                         host_predict_task_count(rows))
    var parallel_softmax_ms = Float64(perf_counter_ns() - t0) / 1.0e6
    for i in range(rows * classes):
        if ms[i] != mp[i]:
            raise Error("parallel softmax moved output cell " + String(i))

    print("softmax row parallelism OK: serial sigmoid control ", serial_sigmoid_ms, " vs production ", parallel_sigmoid_ms, " ms; softmax ", serial_softmax_ms, " -> ", parallel_softmax_ms, " ms; all ", 10 * rows, " cells identical")
