# SPDX-License-Identifier: Apache-2.0
"""Typed scaler buffer admission and returned-statistic validation.

Fit arithmetic remains in preprocessing.estimator and x_prep's existing kernels.
These routines preserve the former buffer casts and public validation semantics.
"""
from std.memory import bitcast
from bindings.classical_cast import classical_f32_bits

alias F32P = MutPointer[Float32, MutUntrackedOrigin]


def scaler_finite(value: Float32) -> Bool:
    return (bitcast[DType.uint32](value) & UInt32(0x7f800000)) != UInt32(0x7f800000)


def scaler_inf(value: Float32) -> Bool:
    return (bitcast[DType.uint32](value) & UInt32(0x7fffffff)) == UInt32(0x7f800000)


def scaler_cast(addr: Int, s0: Int, s1: Int, rows: Int, cols: Int, item: Int, kind: Int, native: Bool, dest: F32P):
    for r in range(rows):
        for c in range(cols):
            var bytes = MutPointer[UInt8, MutUntrackedOrigin](unsafe_from_address=addr + r * s0 + c * s1)
            var raw = UInt64(0)
            for k in range(item):
                raw |= UInt64(bytes[k if native else item - 1 - k]) << UInt64(k * 8)
            dest[r * cols + c] = bitcast[DType.float32](classical_f32_bits(raw, kind, item))


def scaler_all_finite(x: F32P, size: Int) -> Bool:
    for i in range(size):
        if not scaler_finite(x[i]):
            return False
    return True


def scaler_check_stats(mean: F32P, variance: F32P, scale: F32P, d: Int) raises:
    for j in range(d):
        if not scaler_finite(mean[j]) or not scaler_finite(variance[j]) or not scaler_finite(scale[j]) or variance[j] < 0 or scale[j] <= 0:
            raise Error("StandardScaler statistics are nonfinite, variance negative, or scale nonpositive in Float32")


def scaler_fill(weights: F32P, n: Int, value: Float32):
    for i in range(n):
        weights[i] = value


def scaler_check_missing(arena: F32P, st: Int, out: Int, sc: Int, d: Int) raises -> Bool:
    var all_same = True
    for j in range(d):
        if scaler_inf(arena[st+3*d+j]) or scaler_inf(arena[st+4*d+j]):
            raise Error("Scaler input contains infinity; only NaN may be missing")
        if arena[out+j] != 0:
            scaler_check_stats(arena+out+d+j, arena+out+2*d+j, arena+sc+j, 1)
        all_same = all_same and arena[out+j] == arena[out]
    return all_same


def scaler_counts(source: F32P, dest: MutPointer[Int64, MutUntrackedOrigin], d: Int):
    for j in range(d):
        dest[j] = Int64(source[j])
