# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Typed host entries of `bindings/_mojolearn_linalg_host.mojo`: the reads,
oracle calls and result copies each door ran with the GIL released, moved
here with no Python import (lane thin-core, 2026-10-21: a pure move, no bit
change). The binding keeps the argument checks and calls one function."""
from core.hostptr import read_f32, read_i8, read_i32, read_u16
from cholesky.host.chol_oracle import (
    CholHostFactor,
    chol_host_potrf,
    chol_host_solve,
)
from gemm.host.identical_gemm import gemm_oracle
from decomposition.host.linalg_public import (
    host_qr_r,
    host_svdvals,
)
from gemm.host.gemm_lowbit_oracle import (
    Int8Rows,
    dequantize_rows_int8,
    gemm_bf16_both_oracle,
    gemm_bf16_oracle,
    gemm_int8_oracle,
    narrow_bf16,
    quantize_rows_int8,
    widen_bf16,
)
from gemm.f16_widen import f16_bits_to_f32
from gemm.host.gemm_int15_oracle import (
    Int15Rows,
    dequantize_rows_int15,
    gemm_int15_oracle,
    join_int15,
    quantize_rows_int15,
    split_int15,
)


def gemm_host_run(cp: MutPointer[Float32, MutUntrackedOrigin], a_address: Int, b_address: Int, op: Int, m: Int, n: Int, k: Int) raises -> Int:
    """The GIL-free body of `gemm_binding` (moved word for word)."""
    var a = read_f32(a_address, m * k)
    var b = read_f32(b_address, n * k)
    # THE ONE LINE THAT COMPUTES ANYTHING.
    var c = gemm_oracle(a, b, op, m, n, k)
    for i in range(m * n):
        cp[i] = c[i]
    return m * n


def cholesky_solve_host_run(factor: CholHostFactor, b: List[Float32], n: Int, nrhs: Int, xp: MutPointer[Float32, MutUntrackedOrigin]) raises:
    """The GIL-free body of `cholesky_solve_binding` (moved word for word)."""
    var x = chol_host_solve(factor, b, nrhs)
    for i in range(n * nrhs):
        xp.unsafe_store(i, x[i])
    _ = x^


def gemm_bf16_host_run(cp: MutPointer[Float32, MutUntrackedOrigin], a_address: Int, b_address: Int, op: Int, m: Int, n: Int, k: Int, a_bf16: Bool) raises -> Int:
    """The GIL-free body of `gemm_bf16_binding` (moved word for word)."""
    var b = read_u16(b_address, n * k)
    var c = List[Float32]()
    if a_bf16:
        var a_bits = read_u16(a_address, m * k)
        c = gemm_bf16_both_oracle(a_bits, b, op, m, n, k)
    else:
        var a = read_f32(a_address, m * k)
        c = gemm_bf16_oracle(a, b, op, m, n, k)
    for i in range(m * n):
        cp[i] = c[i]
    return m * n


def gemm_int8_host_run(cp: MutPointer[Float32, MutUntrackedOrigin], qa_address: Int, ea_address: Int, qb_address: Int, eb_address: Int, m: Int, n: Int, k: Int) raises -> Int:
    """The GIL-free body of `gemm_int8_binding` (moved word for word)."""
    var qa = read_i8(qa_address, m * k)
    var ea = read_i32(ea_address, m)
    var qb = read_i8(qb_address, n * k)
    var eb = read_i32(eb_address, n)
    var c = gemm_int8_oracle(qa, ea, qb, eb, m, n, k)
    for i in range(m * n):
        cp[i] = c[i]
    return m * n


def quantize_int8_host_run(qp: MutPointer[Int8, MutUntrackedOrigin], ep: MutPointer[Int32, MutUntrackedOrigin], x_address: Int, rows: Int, cols: Int) raises -> Int:
    """The GIL-free body of `quantize_int8_binding` (moved word for word)."""
    var x = read_f32(x_address, rows * cols)
    var qr = quantize_rows_int8(x, rows, cols)
    for i in range(rows * cols):
        qp[i] = qr.q[i]
    for r in range(rows):
        ep[r] = qr.e[r]
    return rows * cols


def dequantize_int8_host_run(yp: MutPointer[Float32, MutUntrackedOrigin], q_address: Int, e_address: Int, rows: Int, cols: Int) raises -> Int:
    """The GIL-free body of `dequantize_int8_binding` (moved word for word)."""
    var q = read_i8(q_address, rows * cols)
    var e = read_i32(e_address, rows)
    var y = dequantize_rows_int8(Int8Rows(q^, e^, rows, cols))
    for i in range(rows * cols):
        yp[i] = y[i]
    return rows * cols


def to_bf16_host_run(dp: MutPointer[UInt16, MutUntrackedOrigin], src_address: Int, count: Int) raises:
    """The GIL-free body of `to_bf16_binding` (moved word for word)."""
    var x = read_f32(src_address, count)
    var bits = narrow_bf16(x)
    for i in range(count):
        dp[i] = bits[i]


def from_bf16_host_run(dp: MutPointer[Float32, MutUntrackedOrigin], src_address: Int, count: Int) raises:
    """The GIL-free body of `from_bf16_binding` (moved word for word)."""
    var bits = read_u16(src_address, count)
    var x = widen_bf16(bits)
    for i in range(count):
        dp[i] = x[i]


def from_f16_host_run(dp: MutPointer[Float32, MutUntrackedOrigin], src_address: Int, count: Int) raises:
    """The GIL-free body of `from_f16_binding` (moved word for word)."""
    var bits = read_u16(src_address, count)
    for i in range(count):
        dp[i] = f16_bits_to_f32(bits[i])


def qr_r_host_run(a: List[Float32], n_rows: Int, n_cols: Int, rp: MutPointer[Float32, MutUntrackedOrigin]) raises:
    """The GIL-free body of `qr_r_binding` (moved word for word)."""
    var r = host_qr_r(a, n_rows, n_cols)
    for i in range(n_cols * n_cols):
        rp.unsafe_store(i, r[i])
    _ = r^


def svdvals_host_run(a: List[Float32], n_rows: Int, n_cols: Int, sp: MutPointer[Float32, MutUntrackedOrigin]) raises:
    """The GIL-free body of `svdvals_binding` (moved word for word)."""
    var s = host_svdvals(a, n_rows, n_cols)
    for i in range(n_cols):
        sp.unsafe_store(i, s[i])
    _ = s^


def gemm_int15_host_run(cp: MutPointer[Float32, MutUntrackedOrigin], ah_address: Int, al_address: Int, ea_address: Int, bh_address: Int, bl_address: Int, eb_address: Int, m: Int, n: Int, k: Int) raises -> Int:
    """The GIL-free body of `gemm_int15_binding` (moved word for word)."""
    var qa = join_int15(read_i8(ah_address, m * k), read_i8(al_address, m * k))
    var ea = read_i32(ea_address, m)
    var qb = join_int15(read_i8(bh_address, n * k), read_i8(bl_address, n * k))
    var eb = read_i32(eb_address, n)
    var c = gemm_int15_oracle(qa, ea, qb, eb, m, n, k)
    for i in range(m * n):
        cp[i] = c[i]
    return m * n


def quantize_int15_host_run(hp: MutPointer[Int8, MutUntrackedOrigin], lp: MutPointer[Int8, MutUntrackedOrigin], ep: MutPointer[Int32, MutUntrackedOrigin], x_address: Int, rows: Int, cols: Int) raises -> Int:
    """The GIL-free body of `quantize_int15_binding` (moved word for word)."""
    var x = read_f32(x_address, rows * cols)
    var qr = quantize_rows_int15(x, rows, cols)
    var planes = split_int15(qr.q)
    for i in range(rows * cols):
        hp[i] = planes.hi[i]
        lp[i] = planes.lo[i]
    for r in range(rows):
        ep[r] = qr.e[r]
    return rows * cols


def dequantize_int15_host_run(yp: MutPointer[Float32, MutUntrackedOrigin], h_address: Int, l_address: Int, e_address: Int, rows: Int, cols: Int) raises -> Int:
    """The GIL-free body of `dequantize_int15_binding` (moved word for word)."""
    var q = join_int15(read_i8(h_address, rows * cols), read_i8(l_address, rows * cols))
    var e = read_i32(e_address, rows)
    var y = dequantize_rows_int15(Int15Rows(q^, e^, rows, cols))
    for i in range(rows * cols):
        yp[i] = y[i]
    return rows * cols


def cholesky_factor_host_run(
    a: List[Float32],
    n: Int,
    jitter: Float32,
    lp: MutPointer[Float32, MutUntrackedOrigin],
    sp: MutPointer[Float64, MutUntrackedOrigin],
) raises -> Int:
    """The GIL-free half of `cholesky_factor_binding`."""
    var f = chol_host_potrf(a, n, jitter)
    for i in range(n * n):
        lp.unsafe_store(i, f.l[i])
    # info, nb, logdet, jitter -- the GPU binding's order.
    sp.unsafe_store(0, Float64(f.info))
    sp.unsafe_store(1, Float64(f.nb))
    sp.unsafe_store(2, Float64(f.logdet))
    sp.unsafe_store(3, Float64(f.jitter))
    var info = f.info
    _ = f^
    return info
