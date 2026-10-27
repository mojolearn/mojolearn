# SPDX-License-Identifier: Apache-2.0
"""Typed scaler buffer admission and returned-statistic validation.

Fit arithmetic remains in preprocessing.estimator and x_prep's existing kernels.
These routines preserve the former buffer casts and public validation semantics.
"""
from std.memory import bitcast
from core.classical_cast import classical_f32_bits

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


def scaler_check_missing(arena: F32P, st: Int, stats_base: Int, sc: Int, d: Int) raises -> Bool:
    var all_same = True
    for j in range(d):
        if scaler_inf(arena[st+3*d+j]) or scaler_inf(arena[st+4*d+j]):
            raise Error("Scaler input contains infinity; only NaN may be missing")
        if arena[stats_base+j] != 0:
            scaler_check_stats(arena+stats_base+d+j, arena+stats_base+2*d+j, arena+sc+j, 1)
        all_same = all_same and arena[stats_base+j] == arena[stats_base]
    return all_same


def scaler_counts(source: F32P, dest: MutPointer[Int64, MutUntrackedOrigin], d: Int):
    for j in range(d):
        dest[j] = Int64(source[j])


def scaler_refuse_infinite_columns(arena: F32P, stats_base: Int, d: Int) raises:
    for j in range(d):
        if scaler_inf(arena[stats_base + 3*d+j]) or scaler_inf(arena[stats_base + 4*d+j]):
            raise Error("Scaler input contains infinity; only NaN may be missing")


def scaler_refuse_infinite(x: F32P, size: Int, message: String) raises:
    # Refuses an infinite entry of a fitted statistic (NaN is a missing
    # value and passes): a column maximum |x| is infinite exactly when the
    # column holds an infinity, so the d statistics decide for the n*d input.
    for j in range(size):
        if scaler_inf(x[j]):
            raise Error(message)


def scaler_feature_indices(dest: F32P, d: Int):
    for j in range(d):
        dest[j] = Float32(j)


def scaler_counts_f32(source_addr: Int, source_kind: Int, scalar: Float64, dest: F32P, d: Int):
    # kind 0 is a native scalar; 1 an int64 vector; 2 a float32 vector.
    var integers = MutPointer[Int64, MutUntrackedOrigin](unsafe_from_address=source_addr)
    var floats = F32P(unsafe_from_address=source_addr)
    for j in range(d):
        dest[j] = Float32(scalar) if source_kind == 0 else (Float32(integers[j]) if source_kind == 1 else floats[j])


def scaler_counts_i64(source_addr: Int, vector: Bool, scalar: Int64, dest: MutPointer[Int64, MutUntrackedOrigin], d: Int):
    var source = MutPointer[Int64, MutUntrackedOrigin](unsafe_from_address=source_addr)
    for j in range(d):
        dest[j] = source[j] if vector else scalar


def scaler_equal_f32(values: F32P, d: Int) -> Bool:
    for j in range(1, d):
        if values[j] != values[0]:
            return False
    return True


def scaler_equal_i64(values: MutPointer[Int64, MutUntrackedOrigin], d: Int) -> Bool:
    for j in range(1, d):
        if values[j] != values[0]:
            return False
    return True


def scaler_counts_f64(source_addr: Int, source_kind: Int, scalar: Float64, dest: MutPointer[Float64,MutUntrackedOrigin], d: Int):
    var integers = MutPointer[Int64,MutUntrackedOrigin](unsafe_from_address=source_addr)
    var floats = F32P(unsafe_from_address=source_addr)
    for j in range(d):
        dest[j] = scalar if source_kind == 0 else (Float64(integers[j]) if source_kind == 1 else Float64(floats[j]))


def scaler_merge_count_only(a: MutPointer[Float64,MutUntrackedOrigin], b: MutPointer[Float64,MutUntrackedOrigin], d: Int) -> Bool:
    var same = True
    for j in range(d):
        a[j] += b[j]
        same = same and a[j] == a[0]
    return same


def scaler_narrow_counts(source: MutPointer[Float64,MutUntrackedOrigin], dest_addr: Int, weighted: Bool, d: Int):
    var floats = F32P(unsafe_from_address=dest_addr)
    var integers = MutPointer[Int64,MutUntrackedOrigin](unsafe_from_address=dest_addr)
    for j in range(d):
        if weighted:
            floats[j] = Float32(source[j])
        else:
            integers[j] = Int64(source[j])
