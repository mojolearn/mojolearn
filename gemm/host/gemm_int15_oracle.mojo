# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
# SHIPS: compiled into the CPU host bindings that serve the int15 profile
# (python/mojolearn/host_surface.py names which); product, not only a check.
"""THE NORMATIVE ANSWER of the fifteen-bit profile, on the host.

    mojolearn.identical.gemm.int15i64.v1   15-bit codes in Int16, an exact
                                           integer sum, power-of-two
                                           dequantization

Contract: `gemm/IDENTICAL_LOWBIT_CONTRACT.md`, the section headed THE
FIFTEEN-BIT PROFILE. Lane lane/lowbit-int15, 2026-09-29, DEVIATIONS 2965 to
2973. Seams `checks/numerics_int15.mojo`.

WHAT THE PROFILE IS, IN ONE SENTENCE. `int8i32.v1` one step wider: both
operands are integer codes with one power-of-two scale per row, the dot
product is an exact integer sum, and the only floating steps are the
conversion of that sum to float32, one multiply by `2^(ea + eb)` and the
flush. OP_NT only, for `int8i32.v1`'s reason.

WHY IT EXISTS. The quality lane measured, on SmolLM2-360M, fifteen-bit
codes on both operands inside the noise of the fp32 baseline and
`int8i32.v1` far outside it. The codes here ARE that lane's
(`bench/lowbit_quality/arith.py`, kind `int15`), and
`gemm/checks/gemm_int15_sim_check.mojo` holds this oracle to that
simulation bit for bit.

TWO SPELLINGS OF ONE SUM, BOTH HERE.
  `int15_dot_cell`         the sum of code products in Int64, `p`
                           ascending. THE DEFINITION.
  `int15_dot_cell_pieces`  the same sum from four int8 piece products
                           accumulated in Int32 and recombined in Int64
                           (clauses W-3 to W-5): what a matrix unit
                           computes. `gemm_int15_check.mojo` requires the
                           two to agree on every fixture and every planted
                           case, on the host, before any device is asked.

The sabotage arm is a VALUE flip on the dequantized cell, not an order arm,
for the reason `gemm_oracle.mojo` records: an exact sum folds an order arm
away, and every sum here is exact.
"""

from std.sys.compile import is_defined

from checks.numerics_int15 import (
    dequant_int15_code,
    dequant_int15_pinned,
    int15_piece_hi,
    int15_piece_lo,
    int15_recombine,
    int15_row_exponent,
    quantize_int15_value,
)
from gemm.host.gemm_lowbit_oracle import row_absmax
from gemm.host.gemm_oracle import (
    GEMM_ORACLE_HOST_SABOTAGE,
    gemm_oracle_sabotage_value_flip,
)

#: THE LARGEST `k` THE PROFILE ACCEPTS, clause W-4. The bounds, each
#: derived in the contract (section W, THE BOUNDS):
#:   one piece sum       `128 * 128 * k <= 2^31 - 1`        k <= 131071
#:   HL + LH in one Int32 `2 * 128 * 127 * k <= 2^31 - 1`   k <= 66052
#: The smallest is 66052 and the profile stops at the power of two below
#: it, so the bound is a number a reader can check. `int8i32.v1`'s bound
#: (131072) does NOT carry over: it assumes a magnitude of 127, and a high
#: piece reaches -128, whose square times 131072 is `2^31` exactly.
from gemm.contract import INT15_MAX_K, INT15_PROFILE_VERSION  # the profile bound and version, shared with the device kernels

#: The bound of clause W-4 (d) before the profile rounds it down: the
#: largest `k` at which `HL + LH` cannot leave an Int32. The gate plants a
#: row at this `k` and one above it and requires the wrapped Int32 to
#: differ from the exact sum at the second and not at the first.
comptime INT15_MID_BOUND_K = 66052

#: The bound of clause W-4 (c): the largest `k` at which one piece sum
#: cannot leave an Int32.
comptime INT15_PIECE_BOUND_K = 131071

#: The conversions' own negative control, the define the low-bit
#: conversions already answer to (`gemm_lowbit_oracle.mojo`): the codes take
#: the low bit flipped and a dequantized value takes the value flip.
# Spelled on ONE LINE on purpose: test_host_surface greps each family's own
# define as `is_defined["<NAME>"]`.
comptime INT15_CONVERT_SABOTAGE = is_defined["MOJOLEARN_LOWBIT_CONVERT_SABOTAGE"]()


struct Int15Rows(Movable):
    """A row-major matrix quantized row by row: `q` holds the fifteen-bit
    codes and `e` one exponent per row, so row `r` is `q[r, :] * 2^e[r]`."""

    var q: List[Int16]
    var e: List[Int32]
    var rows: Int
    var cols: Int

    def __init__(out self, var q: List[Int16], var e: List[Int32], rows: Int, cols: Int):
        self.q = q^
        self.e = e^
        self.rows = rows
        self.cols = cols


struct Int15Planes(Movable):
    """The two int8 planes of a matrix of codes, clause W-3: `hi` and `lo`
    with `q = hi * 128 + lo`, `lo` in `[0, 127]`, `hi` in `[-128, 127]`."""

    var hi: List[Int8]
    var lo: List[Int8]

    def __init__(out self, var hi: List[Int8], var lo: List[Int8]):
        self.hi = hi^
        self.lo = lo^


def quantize_rows_int15(x: List[Float32], rows: Int, cols: Int) -> Int15Rows:
    """Clauses W-1 and W-2: per row, the power-of-two exponent from the
    row's absmax, then every value scaled, rounded to nearest even and
    clamped. The absmax is `int8i32.v1`'s (`row_absmax`): a maximum after
    the flush, exact and order-free, which a NaN never wins."""
    var q = List[Int16]()
    var e = List[Int32]()
    for r in range(rows):
        var ex = int15_row_exponent(row_absmax(x, r, cols))
        e.append(Int32(ex))
        for c in range(cols):
            var code = quantize_int15_value(x[r * cols + c], ex)
            comptime if INT15_CONVERT_SABOTAGE:
                code = code ^ Int16(1)
            q.append(code)
    return Int15Rows(q^, e^, rows, cols)


def quantize_cols_int15(x: List[Float32], rows: Int, cols: Int) -> Int15Rows:
    """The codes of the COLUMNS of a `rows x cols` matrix, written as the
    rows of its transpose (`cols x rows`): what a product that contracts
    over the matrix's row index reads (OP_NN's right operand, both of
    OP_TN's). Clauses W-1 and W-2 on the transposed values and nothing
    else, so it is `quantize_rows_int15` of the transpose, bit for bit."""
    var xt = List[Float32]()
    for c in range(cols):
        for r in range(rows):
            xt.append(x[r * cols + c])
    return quantize_rows_int15(xt, cols, rows)


def dequantize_rows_int15(qr: Int15Rows) -> List[Float32]:
    """`q * 2^e`, exact unless it leaves the normal range: the float32
    matrix a fifteen-bit store stands for."""
    var out = List[Float32]()
    for r in range(qr.rows):
        for c in range(qr.cols):
            var v = dequant_int15_code(qr.q[r * qr.cols + c], Int(qr.e[r]))
            comptime if INT15_CONVERT_SABOTAGE:
                v = gemm_oracle_sabotage_value_flip(v)
            out.append(v)
    return out^


def split_int15(q: List[Int16]) -> Int15Planes:
    """Clause W-3: every code into its two int8 pieces."""
    var hi = List[Int8]()
    var lo = List[Int8]()
    for i in range(len(q)):
        hi.append(int15_piece_hi(q[i]))
        lo.append(int15_piece_lo(q[i]))
    return Int15Planes(hi^, lo^)


def join_int15(hi: List[Int8], lo: List[Int8]) -> List[Int16]:
    """Clause W-3 read backwards: the codes two planes stand for,
    `hi * 128 + lo`. Exact: a shift and an addition of small integers."""
    var q = List[Int16]()
    for i in range(len(hi)):
        q.append(Int16(Int(hi[i]) * 128 + Int(lo[i])))
    return q^


def int15_dot_cell(
    qa: List[Int16], qb: List[Int16], i: Int, j: Int, k: Int
) -> Int64:
    """THE DEFINITION: the exact integer sum of code products for cell
    `(i, j)` at OP_NT, `p` ascending, in Int64. A product is below `2^28`
    in magnitude and is formed in Int32; the sum of `k <= 65536` of them is
    below `2^44`."""
    var acc = Int64(0)
    for p in range(k):
        acc += Int64(Int32(qa[i * k + p]) * Int32(qb[j * k + p]))
    return acc


def int15_dot_cell_pieces(
    ah: List[Int8],
    al: List[Int8],
    bh: List[Int8],
    bl: List[Int8],
    i: Int,
    j: Int,
    k: Int,
) -> Int64:
    """The same sum the way a matrix unit forms it, clauses W-4 and W-5:
    three Int32 accumulators (`HH`, `HL + LH`, `LL`), then the
    recombination in Int64. The Int32 additions WRAP, as a unit's do, so a
    `k` above the bound gives a wrong integer here and not a trap: that is
    what lets the gate show the bound is where the contract says it is."""
    var hh = Int32(0)
    var mid = Int32(0)
    var ll = Int32(0)
    for p in range(k):
        var a_hi = Int32(ah[i * k + p])
        var a_lo = Int32(al[i * k + p])
        var b_hi = Int32(bh[j * k + p])
        var b_lo = Int32(bl[j * k + p])
        # Plain `+` on Int32 wraps; nothing here is spelled `&+`, which
        # Mojo reads as a bitwise AND.
        hh = hh + a_hi * b_hi
        mid = mid + a_hi * b_lo
        mid = mid + a_lo * b_hi
        ll = ll + a_lo * b_lo
    return int15_recombine(hh, mid, ll)


def _refuse_k(k: Int, who: String) raises:
    if k > INT15_MAX_K:
        raise Error(
            who + ": k must be at most " + String(INT15_MAX_K)
            + " so no Int32 piece sum can overflow (contract W-4), got "
            + String(k)
        )


def gemm_int15_oracle_cell(
    qa: List[Int16],
    ea: List[Int32],
    qb: List[Int16],
    eb: List[Int32],
    i: Int,
    j: Int,
    k: Int,
) -> Float32:
    """ONE CELL of the normative answer, for a check that cannot afford
    the whole product on the host (a 512 x 4096 x 14336 product is 3e10
    steps). The arithmetic and the sabotage arm of `gemm_int15_oracle`."""
    var v = dequant_int15_pinned(int15_dot_cell(qa, qb, i, j, k), Int(ea[i]) + Int(eb[j]))
    comptime if GEMM_ORACLE_HOST_SABOTAGE:
        v = gemm_oracle_sabotage_value_flip(v)
    return v


def gemm_int15_oracle(
    qa: List[Int16],
    ea: List[Int32],
    qb: List[Int16],
    eb: List[Int32],
    m: Int,
    n: Int,
    k: Int,
) raises -> List[Float32]:
    """**THE NORMATIVE ANSWER of `mojolearn.identical.gemm.int15i64.v1`**
    at OP_NT: `C[i, j] = ftz( f32(q_a[i] . q_b[j]) * 2^(ea[i] + eb[j]) )`.
    Row-major `m x n`. `ea` has `m` entries and `eb` has `n`. A `k` above
    `INT15_MAX_K` is refused by name."""
    _refuse_k(k, String("gemm_int15_oracle"))
    var c = List[Float32]()
    for i in range(m):
        for j in range(n):
            # THE SABOTAGE ARM lives in the cell: a value whose bits differ
            # from the dequantized cell on every fixture, exact ones included.
            c.append(gemm_int15_oracle_cell(qa, ea, qb, eb, i, j, k))
    return c^


def gemm_int15_pieces_oracle(
    qa: List[Int16],
    ea: List[Int32],
    qb: List[Int16],
    eb: List[Int32],
    m: Int,
    n: Int,
    k: Int,
) raises -> List[Float32]:
    """The same answer through the pieces. NOT the definition and carries
    no sabotage arm: it is the host's own check that clauses W-3 to W-5
    describe the sum `gemm_int15_oracle` computes."""
    _refuse_k(k, String("gemm_int15_pieces_oracle"))
    var pa = split_int15(qa)
    var pb = split_int15(qb)
    var c = List[Float32]()
    for i in range(m):
        for j in range(n):
            var acc = int15_dot_cell_pieces(pa.hi, pa.lo, pb.hi, pb.lo, i, j, k)
            c.append(dequant_int15_pinned(acc, Int(ea[i]) + Int(eb[j])))
    return c^


def gemm_int15_from_f32_oracle(
    a: List[Float32], b: List[Float32], m: Int, n: Int, k: Int
) raises -> List[Float32]:
    """Quantize both operands by the profile's own rule, then the product.
    The one-call form the checks and the bindings' convenience path use."""
    var qa = quantize_rows_int15(a, m, k)
    var qb = quantize_rows_int15(b, n, k)
    return gemm_int15_oracle(qa.q, qa.e, qb.q, qb.e, m, n, k)
