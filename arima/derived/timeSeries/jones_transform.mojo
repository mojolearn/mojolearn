# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
# Derivative work: the upstream file and its pinned commit are recorded in this lane's DERIVATION_MAP.tsv and in this file's own docstring. See NOTICE.
"""The Jones (1980) transform: unconstrained parameters to a stationary AR /
invertible MA polynomial, and back.

PORT OF `cuml/cpp/src_prims/timeSeries/jones_transform.cuh` at cuML 265b9da6
(v26.08.00): `transform` (:35-67), `invtransform` (:71-91),
`jones_transform_kernel` (:105-138), `jones_transform` (:156-182). COPY, DO
NOT IMPROVE. One thread per series; `parameter <= 8` (their local arrays
are `DataT[8]`; `ASSERT(parameter >= 1 && parameter <= 8)`).

THE ARITHMETIC, line for line (`transform`):
    tmp[i] = tanh(tmp[i] * 0.5); new[i] = tmp[i]                 (:38-41)
    for j in 1..p-1: a = new[j]; for k < j: tmp[k] += sign*(a*new[j-k-1])
                     new[0..j) = tmp[0..j)                         (:44-52)
    clamp: new[i] = max(-0.9999, min(new[i], 0.9999))             (:57)
`invtransform`:
    for j = p-1..1: a = new[j]; tmp[k] = (new[k] + sign*(a*new[j-k-1])) / (1 - a*a)
                    new[0..j) = tmp[0..j)                          (:75-85)
    new[i] = 2 * atanh(new[i])                                     (:88)

ROW 9 / ROW 10 / ROW 12 SEAMS. `tmp[k] += sign * (a * new[..])` contracts
as `fma(sign, round(a * new), tmp)`: the multiply that FEEDS the add is
`sign * (...)`, so `a * new` is a SEPARATE rounding and only the outer
product fuses. TWO roundings, not one. (An earlier revision of this file
read the tree as `fma(sign*a, new, tmp)` -- ONE rounding -- and both the
kernel and the host replay were spelled that way; the audit of 2026-08-23
against `jones_transform.cuh:47` corrected it. `sign` is +-1, so
`fma(sign, prod, tmp)` is an exact-signed add, but `prod` is rounded
first and that is the bit that moved.) `1 - (a*a)` DOES contract to
`fma(-a, a, 1)` -- the multiply feeds the subtract directly --
`identical_mul_add(-a, a, 1)`, hoisted out of the `k` loop because it is
loop-invariant in theirs too. `tanh` and `atanh` are
`raft::tanh` / `raft::atanh` -> the device `tanhf` / `atanhf`, a vendor
transcendental each (row 12); their IDENTICAL spelling is DEVIATION 675.
The clamp is spelled VALUE-FIRST (`max(min(v, 0.9999), -0.9999)`, ADDENDUM
11); both zeros cannot meet there (a `tanh` output is clamped only near
+-1) but the spelling is the rule.

=============================================================================
DEVIATION 675: tanh AND atanh THROUGH identical_exp / identical_log
=============================================================================
THEIRS. `raft::tanh(x * 0.5)` and `2 * raft::atanh(v)`: the vendor's
`tanhf` / `atanhf`, whose last bit is a VENDOR CHOICE (IDENTITY_PATHS row
12's class; `numerics.mojo` carries exp/log/sqrt/cos/pow but no tanh).
OURS, under IDENTICAL:
    tanh(x/2)  = (e^x - 1) / (e^x + 1)     with e^x = identical_exp(x),
                 +-1 returned outright for |x| > 80 (e^x would overflow)
    2 atanh(v) = log((1 + v) / (1 - v))    through identical_log
one local per op through `ftz`. Under FAST they are `std.math.tanh(x * 0.5)`
and `2 * std.math.atanh(v)`, the stdlib device path verbatim (Apple FAST
bits do not move). Neither identity is the libm algorithm, so IDENTICAL
bits differ from FAST bits by design, as every row-12 seam does; what is
purchased is one arithmetic on every backend.

MEASURED 2026-08-23, Apple M4, n_obs 24, batch 6
(`arima/original/arima_check.mojo::check_jones_device_equals_oracle`):
device == host replay BITWISE under IDENTICAL, 0 cells differing, for
p = 1..4, AR and MA, forward and inverse (16 stage comparisons, 6 to 24
cells each).

THE ROUND TRIP IS WORSE THAN THIS FILE USED TO CLAIM. An earlier revision
said `inverse(forward(x))` returns to x "within 4 ulp (Float32,
reported)". It does not, and nothing had ever run when that was written.
The measured worst relative error is 2.73e-6, which is about 23 Float32
ulp, uniform across p = 1..4 and both AR and MA. STRUCK and replaced with
the number.

That is DEVIATION 675's price and it is a real one: `(e^x - 1)` cancels for
small `|x|`, and `atanh` near 0 is where the optimizer lives. It is
REPORTED and bounded loosely (the gate asserts only < 1e-2), not asserted
tight, because the goal here is one arithmetic on every backend and not
accuracy. OWED: decide whether `expm1(x)/(expm1(x)+2)` is worth a numbered
replacement.
"""

from max.gpu.host import DeviceBuffer, DeviceContext
from std.gpu import block_dim, block_idx, thread_idx
from std.math import atanh, tanh

from original.numerics import (
    GLOBAL_NUMERIC_MODE,
    NUMERIC_IDENTICAL,
    ftz,
    identical_exp,
    identical_log,
    identical_mul_add,
)


comptime JONES_MAX_PARAMS = 8
comptime JONES_TPB = 256
comptime JONES_CLAMP = Float32(0.9999)


@always_inline
def tanh_half(x: Float32) -> Float32:
    """`raft::tanh(x * 0.5)` (DEVIATION 675)."""
    comptime if GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL:
        if x > Float32(80.0):
            return Float32(1.0)
        if x < Float32(-80.0):
            return Float32(-1.0)
        var e = ftz(identical_exp(x))
        var num = ftz(e - Float32(1.0))
        var den = ftz(e + Float32(1.0))
        return ftz(num / den)
    else:
        return tanh(x * Float32(0.5))


@always_inline
def two_atanh(v: Float32) -> Float32:
    """`2 * raft::atanh(v)` (DEVIATION 675)."""
    comptime if GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL:
        var num = ftz(Float32(1.0) + v)
        var den = ftz(Float32(1.0) - v)
        return ftz(identical_log(ftz(num / den)))
    else:
        return Float32(2.0) * atanh(v)


@always_inline
def jones_clamp(v: Float32) -> Float32:
    """`max(-0.9999, min(v, 0.9999))`, value-first."""
    return max(min(v, JONES_CLAMP), -JONES_CLAMP)


def jones_transform_kernel(
    new_params: MutPointer[Float32, MutAnyOrigin],
    params: MutPointer[Float32, MutAnyOrigin],
    batch_size_in: Int32,
    parameter_in: Int32,
    is_ar_in: Int32,
    is_inv_in: Int32,
    clamp_in: Int32,
):
    """`jones_transform.cuh:105-138`: load, transform or invert, store."""
    var model = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if model >= Int(batch_size_in):
        return
    var parameter = Int(parameter_in)
    var is_ar = is_in_ar(is_ar_in)
    var tmp = InlineArray[Float32, JONES_MAX_PARAMS](fill=Float32(0.0))
    var mine = InlineArray[Float32, JONES_MAX_PARAMS](fill=Float32(0.0))
    for i in range(parameter):
        var v = ftz(params.unsafe_load(model * parameter + i))
        tmp[i] = v
        mine[i] = v
    if is_inv_in != 0:
        # invtransform (:72-90)
        var sign = Float32(1.0) if is_ar else Float32(-1.0)
        var j = parameter - 1
        while j > 0:
            var a = mine[j]
            var den = ftz(identical_mul_add(-a, a, Float32(1.0)))
            for k in range(j):
                var prod = ftz(a * mine[j - k - 1])
                var num = ftz(identical_mul_add(sign, prod, mine[k]))
                tmp[k] = ftz(num / den)
            for it in range(j):
                mine[it] = tmp[it]
            j -= 1
        for i in range(parameter):
            mine[i] = two_atanh(mine[i])
    else:
        # transform (:36-60)
        for i in range(parameter):
            tmp[i] = tanh_half(tmp[i])
            mine[i] = tmp[i]
        var sign = Float32(-1.0) if is_ar else Float32(1.0)
        for j in range(1, parameter):
            var a = mine[j]
            for k in range(j):
                var prod = ftz(a * mine[j - k - 1])
                tmp[k] = ftz(identical_mul_add(sign, prod, tmp[k]))
            for it in range(j):
                mine[it] = tmp[it]
        if clamp_in != 0:
            for i in range(parameter):
                mine[i] = jones_clamp(mine[i])
    for i in range(parameter):
        new_params.unsafe_store(model * parameter + i, mine[i])


@always_inline
def is_in_ar(v: Int32) -> Bool:
    return v != 0


def jones_transform(
    ctx: DeviceContext,
    mut params: DeviceBuffer[DType.float32],
    batch_size: Int,
    parameter: Int,
    mut new_params: DeviceBuffer[DType.float32],
    is_ar: Bool,
    is_inv: Bool,
    clamp: Bool = True,
) raises:
    """`jones_transform.cuh:156-182`: the guards, then one launch. Their
    `raft::copy(newParams, params)` before the kernel is a no-op on the
    result (every cell is overwritten) and is not issued."""
    if batch_size < 1:
        raise Error("jones_transform: unsupported batchSize " + String(batch_size))
    if parameter < 1 or parameter > JONES_MAX_PARAMS:
        raise Error("jones_transform: unsupported parameter " + String(parameter))
    var grid = (batch_size + JONES_TPB - 1) // JONES_TPB
    ctx.enqueue_function[jones_transform_kernel](
        new_params.unsafe_ptr(), params.unsafe_ptr(), Int32(batch_size),
        Int32(parameter), Int32(1 if is_ar else 0), Int32(1 if is_inv else 0),
        Int32(1 if clamp else 0),
        grid_dim=(grid, 1, 1), block_dim=(JONES_TPB, 1, 1),
    )


# ---------------------------------------------------------------------------
# host replay (NOT a port: the oracle, spelled separately over Lists)
# ---------------------------------------------------------------------------


def jones_transform_host(
    params: List[Float32], batch_size: Int, parameter: Int,
    is_ar: Bool, is_inv: Bool, clamp: Bool = True,
) raises -> List[Float32]:
    if parameter < 1 or parameter > JONES_MAX_PARAMS:
        raise Error("jones_transform_host: unsupported parameter " + String(parameter))
    var out = List[Float32]()
    for model in range(batch_size):
        var tmp = List[Float32]()
        var mine = List[Float32]()
        for i in range(parameter):
            var v = ftz(params[model * parameter + i])
            tmp.append(v)
            mine.append(v)
        if is_inv:
            var sign = Float32(1.0) if is_ar else Float32(-1.0)
            var j = parameter - 1
            while j > 0:
                var a = mine[j]
                var den = ftz(identical_mul_add(-a, a, Float32(1.0)))
                for k in range(j):
                    var prod = ftz(a * mine[j - k - 1])
                    var num = ftz(identical_mul_add(sign, prod, mine[k]))
                    tmp[k] = ftz(num / den)
                for it in range(j):
                    mine[it] = tmp[it]
                j -= 1
            for i in range(parameter):
                mine[i] = two_atanh(mine[i])
        else:
            for i in range(parameter):
                tmp[i] = tanh_half(tmp[i])
                mine[i] = tmp[i]
            var sign = Float32(-1.0) if is_ar else Float32(1.0)
            for j in range(1, parameter):
                var a = mine[j]
                for k in range(j):
                    var prod = ftz(a * mine[j - k - 1])
                    tmp[k] = ftz(identical_mul_add(sign, prod, tmp[k]))
                for it in range(j):
                    mine[it] = tmp[it]
            if clamp:
                for i in range(parameter):
                    mine[i] = jones_clamp(mine[i])
        for i in range(parameter):
            out.append(mine[i])
    return out^
