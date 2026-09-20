# SPDX-License-Identifier: Apache-2.0
"""Bitwise gate for the CPU LogisticRegression probability row split."""

from core.classical_host_predict import (
    host_qn_sigmoid_into,
    host_qn_softmax_into,
)
from core.host_predict_threads import HostF32Ptr, HostF64Ptr


def check_sigmoid_parallel_bits() raises:
    var n = 8193
    var scores = alloc[Float32](n)
    for i in range(n):
        scores.unsafe_store(i, Float32((i % 257) - 128) / Float32(17.0))
    var serial = alloc[Float64](2 * n)
    var parallel = alloc[Float64](2 * n)
    host_qn_sigmoid_into(HostF32Ptr(scores), HostF64Ptr(serial), n, 1)
    host_qn_sigmoid_into(HostF32Ptr(scores), HostF64Ptr(parallel), n, 7)
    for i in range(2 * n):
        if serial.unsafe_load(i) != parallel.unsafe_load(i):
            raise Error("host sigmoid parallel result moved at cell " + String(i))
    scores.unsafe_free()
    serial.unsafe_free()
    parallel.unsafe_free()


def check_softmax_parallel_bits() raises:
    var n = 4099
    var c = 7
    var scores = alloc[Float32](n * c)
    for i in range(n * c):
        # Repeated maxima exercise the strict-`>` tie rule as well as mixed
        # signs and rows that do not divide evenly among tasks.
        scores.unsafe_store(i, Float32((i % 31) - 15) / Float32(5.0))
    var serial = alloc[Float64](n * c)
    var parallel = alloc[Float64](n * c)
    host_qn_softmax_into(HostF32Ptr(scores), HostF64Ptr(serial), n, c, 1)
    host_qn_softmax_into(HostF32Ptr(scores), HostF64Ptr(parallel), n, c, 11)
    for i in range(n * c):
        if serial.unsafe_load(i) != parallel.unsafe_load(i):
            raise Error("host softmax parallel result moved at cell " + String(i))
    scores.unsafe_free()
    serial.unsafe_free()
    parallel.unsafe_free()


def main() raises:
    check_sigmoid_parallel_bits()
    check_softmax_parallel_bits()
    print("host probability pointer/parallel bits OK")
