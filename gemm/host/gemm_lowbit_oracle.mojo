# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
# SHIPS: compiled into the CPU host bindings that serve a low-bit weight
# format (python/mojolearn/host_surface.py names which); product, not only
# a check.
"""THE NORMATIVE ANSWERS of the two low-bit profiles, on the host.

    mojolearn.identical.gemm.bf16f32.v1   bf16 operands, fp32.v1 arithmetic
    mojolearn.identical.gemm.int8i32.v1   int8 operands, int32 accumulation,
                                          power-of-two dequantization

Contract: `gemm/IDENTICAL_LOWBIT_CONTRACT.md`. Lane
lane/identical-lowbit-inference, 2026-09-17, DEVIATIONS 2900 to 2909.

WHAT THE bf16 PROFILE IS, IN ONE SENTENCE. A bf16 is the top half of a
float32, so widening it is a shift that cannot round; the profile is
therefore `mojolearn.identical.gemm.fp32.v1` applied to the exactly widened
operands, and this file does not re-spell one line of that arithmetic: it
widens and calls `gemm_oracle`. That is what makes the bf16 profile inherit
the fp32 profile's certificate at every clause the widening does not touch,
and it is why the sabotage arm is fp32.v1's own (`MOJOLEARN_HOST_SABOTAGE`
reaches it through `gemm_oracle`).

WHAT THE int8 PROFILE IS, IN ONE SENTENCE. `q_a . q_b` is an integer sum of
integer products, exact and therefore order-free, so contract sections 4
through 7 of the fp32 profile become vacuous; what is NOT vacuous is the
scale, and the profile pins it to a power of two per row so that the
dequantization is a single exact multiply (L-5, L-6). The int8 profile
covers OP_NT only: the activation rows and the weight rows are each
quantized along `k`, which is the only orientation in which a per-row
exponent is a per-row exponent on both sides.

The sabotage arm of the int8 profile is a VALUE flip on the dequantized
cell, not an order arm, for the reason `gemm_oracle.mojo` records at its
`GEMM_ORACLE_HOST_SABOTAGE`: an exact sum folds an order arm away, and
every int8 sum is exact.
"""

from std.memory import bitcast
from std.sys.compile import is_defined

from checks.numerics import (
    bf16_bits_to_f32,
    dequant_int8_pinned,
    f32_to_bf16_bits_rne,
    ftz,
    int8_row_exponent,
    quantize_int8_value,
)
from gemm.host.gemm_oracle import (
    GEMM_ORACLE_HOST_SABOTAGE,
    OP_NN,
    OP_NT,
    OP_TN,
    gemm_oracle,
    gemm_oracle_sabotage_value_flip,
)

#: The largest `k` the int8 profile accepts: `127 * 127 * k < 2^31` holds
#: up to 133,152, and the profile stops at the power of two below it so the
#: bound is a number a reader can check. Contract L-7.
comptime INT8_MAX_K = 131072

#: THE CONVERSION SEAMS' OWN NEGATIVE CONTROL (lane/laneless-public-classes,
#: 2026-09-19). The four functions below -- `widen_bf16`, `narrow_bf16`,
#: `quantize_rows_int8`, `dequantize_rows_int8` -- are the whole of contract
#: clauses L-1 through L-6 and they are what `mojolearn.lowbit.pack_one`,
#: `materialize_one` and `widen_bf16` and `mojolearn.linalg.to_bf16`,
#: `from_bf16`, `quantize_int8` and `dequantize_int8` compute on a CPU
#: column. Until this define they had NO arm at all: `GEMM_ORACLE_HOST_SABOTAGE`
#: reaches `gemm_oracle`'s leaf and `gemm_int8_oracle`'s dequantized cell and
#: stops there, so a build carrying `-D MOJOLEARN_HOST_SABOTAGE=1` left every
#: conversion byte where it found it. A lane over the conversions alone was
#: therefore REACHED AND INERT, which is the defect `linalg-eigh` and
#: `bpe-trainer` each turned out to be; the `lowbit-conversions` lane of
#: tools/identity_break.py is watched failing under THIS define, and
#: host_surface.GATE_SABOTAGE_OWN_DEFINES names it for the linalg family so
#: the CPU identity gate's sabotage set carries it beside the family one.
#:
#: Each arm perturbs a VALUE and not an order, for the reason
#: `gemm_oracle.mojo` records: every seam here is exact, so an order arm
#: would fold away. A bf16 bit pattern and an int8 code take the low bit
#: flipped, which no input can make a no-op; a float32 result takes
#: `gemm_oracle_sabotage_value_flip`, which moves a zero and a subnormal off
#: the flush as well.
# Spelled on ONE LINE on purpose: test_host_surface greps each family's own
# define as `is_defined["<NAME>"]`, and a wrapped call makes that probe
# return nothing forever while reading exactly like a pass.
comptime LOWBIT_CONVERT_SABOTAGE = is_defined["MOJOLEARN_LOWBIT_CONVERT_SABOTAGE"]()

#: The profile version the bindings read back. The leaf rule and fold
#: topology are fp32.v1's; the low-bit seams are this file's; a change to
#: either makes v2.
comptime LOWBIT_PROFILE_VERSION = 1


# ===========================================================================
# bf16f32.v1
# ===========================================================================


def widen_bf16(bits: List[UInt16]) -> List[Float32]:
    """Contract L-1: every element exactly widened."""
    var out = List[Float32]()
    for i in range(len(bits)):
        var v = bf16_bits_to_f32(bits[i])
        comptime if LOWBIT_CONVERT_SABOTAGE:
            v = gemm_oracle_sabotage_value_flip(v)
        out.append(v)
    return out^


def narrow_bf16(x: List[Float32]) -> List[UInt16]:
    """Contract L-2: every element flushed, then rounded to nearest even."""
    var out = List[UInt16]()
    for i in range(len(x)):
        var b = f32_to_bf16_bits_rne(x[i])
        comptime if LOWBIT_CONVERT_SABOTAGE:
            b = b ^ UInt16(1)
        out.append(b)
    return out^


def gemm_bf16_oracle(
    a: List[Float32],
    b_bits: List[UInt16],
    op: Int,
    m: Int,
    n: Int,
    k: Int,
) -> List[Float32]:
    """**THE NORMATIVE ANSWER of `mojolearn.identical.gemm.bf16f32.v1`**
    with a float32 left operand and a bf16 right operand, the shape of every
    projection in the library (activations times `[out, in]` weights).
    `gemm_oracle` on the widened right operand, nothing else."""
    var b = widen_bf16(b_bits)
    return gemm_oracle(a, b, op, m, n, k)


def gemm_bf16_both_oracle(
    a_bits: List[UInt16],
    b_bits: List[UInt16],
    op: Int,
    m: Int,
    n: Int,
    k: Int,
) -> List[Float32]:
    """The same profile with both operands bf16."""
    var a = widen_bf16(a_bits)
    var b = widen_bf16(b_bits)
    return gemm_oracle(a, b, op, m, n, k)


# ===========================================================================
# int8i32.v1
# ===========================================================================


struct Int8Rows(Movable):
    """A row-major matrix quantized row by row: `q` holds the int8 codes and
    `e` one exponent per row, so row `r` is `q[r, :] * 2^e[r]`."""

    var q: List[Int8]
    var e: List[Int32]
    var rows: Int
    var cols: Int

    def __init__(out self, var q: List[Int8], var e: List[Int32], rows: Int, cols: Int):
        self.q = q^
        self.e = e^
        self.rows = rows
        self.cols = cols


def row_absmax(x: List[Float32], row: Int, cols: Int) -> Float32:
    """The largest magnitude in a row, after the flush. A maximum is exact
    and order-free, so no fold rule is needed here (contract L-3)."""
    var best = Float32(0.0)
    for c in range(cols):
        var v = ftz(x[row * cols + c])
        if v < Float32(0.0):
            v = -v
        if v > best:
            best = v
    return best


def quantize_rows_int8(x: List[Float32], rows: Int, cols: Int) -> Int8Rows:
    """Contract L-3 and L-4: per row, the power-of-two exponent from the
    row's absmax, then every value scaled, rounded to nearest even and
    clamped."""
    var q = List[Int8]()
    var e = List[Int32]()
    for r in range(rows):
        var ex = int8_row_exponent(row_absmax(x, r, cols))
        e.append(Int32(ex))
        for c in range(cols):
            var code = quantize_int8_value(x[r * cols + c], ex)
            comptime if LOWBIT_CONVERT_SABOTAGE:
                code = code ^ Int8(1)
            q.append(code)
    return Int8Rows(q^, e^, rows, cols)


def dequantize_rows_int8(qr: Int8Rows) -> List[Float32]:
    """`q * 2^e`, exact, the float32 matrix an int8 store stands for. This
    is what a block that takes int8 WEIGHTS and keeps fp32 arithmetic
    computes with (the `int8w` formats of the inference classes)."""
    var out = List[Float32]()
    for r in range(qr.rows):
        for c in range(qr.cols):
            var v = dequant_int8_pinned(
                Int32(qr.q[r * qr.cols + c]), Int(qr.e[r])
            )
            comptime if LOWBIT_CONVERT_SABOTAGE:
                v = gemm_oracle_sabotage_value_flip(v)
            out.append(v)
    return out^


def int8_dot_cell(
    qa: List[Int8], qb: List[Int8], i: Int, j: Int, k: Int
) -> Int32:
    """The exact integer sum of products for cell `(i, j)` at OP_NT, `p`
    ascending. Order cannot move a bit here; ascending is kept so the
    device kernel and this loop are one spelling (contract L-7)."""
    var acc = Int32(0)
    for p in range(k):
        acc += Int32(qa[i * k + p]) * Int32(qb[j * k + p])
    return acc


def gemm_int8_oracle(
    qa: List[Int8],
    ea: List[Int32],
    qb: List[Int8],
    eb: List[Int32],
    m: Int,
    n: Int,
    k: Int,
) -> List[Float32]:
    """**THE NORMATIVE ANSWER of `mojolearn.identical.gemm.int8i32.v1`** at
    OP_NT: `C[i, j] = ftz( f32(q_a[i] . q_b[j]) * 2^(ea[i] + eb[j]) )`.
    Row-major `m x n`. `ea` has `m` entries and `eb` has `n`."""
    var c = List[Float32]()
    for i in range(m):
        for j in range(n):
            var acc = int8_dot_cell(qa, qb, i, j, k)
            var v = dequant_int8_pinned(acc, Int(ea[i]) + Int(eb[j]))
            comptime if GEMM_ORACLE_HOST_SABOTAGE:
                # THE SABOTAGE ARM: a value whose bits differ from the
                # dequantized cell on every fixture, exact ones included.
                v = gemm_oracle_sabotage_value_flip(v)
            c.append(v)
    return c^


def gemm_int8_from_f32_oracle(
    a: List[Float32], b: List[Float32], m: Int, n: Int, k: Int
) -> List[Float32]:
    """Quantize both operands by the profile's own rule, then the product.
    The one-call form the checks and the bindings' convenience path use."""
    var qa = quantize_rows_int8(a, m, k)
    var qb = quantize_rows_int8(b, n, k)
    return gemm_int8_oracle(qa.q, qa.e, qb.q, qb.e, m, n, k)
