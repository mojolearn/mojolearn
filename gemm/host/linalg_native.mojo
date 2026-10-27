# SPDX-License-Identifier: Apache-2.0
# CPU numerical owners admitted through the same shared provider ABI as GPU.
# Bodies preserve the existing host binding validation and *_host_run calls;
# descriptor admission and GIL ownership belong to the common provider boundary.
from core.hostptr import f32_ptr, i8_ptr, i32_ptr, u16_ptr
from gemm.contract import INT8_MAX_K, INT15_MAX_K, OP_NN, OP_NT, OP_TN
from decomposition.linalg_host_impl import (
    gemm_host_run,
    gemm_bf16_host_run,
    gemm_int8_host_run,
    quantize_int8_host_run,
    dequantize_int8_host_run,
    gemm_int15_host_run,
    quantize_int15_host_run,
    dequantize_int15_host_run,
    to_bf16_host_run,
    from_bf16_host_run,
    from_f16_host_run,
)

comptime LINALG_HOST_MAX_EXTENT = 1073741824


def gemm_native(
    c_addr: Int,
    a_addr: Int,
    b_addr: Int,
    params: List[Int],
) raises -> Int:
    if len(params) != 4:
        raise Error(
            "gemm: params must contain 4 values (m, n, k, op), got "
            + String(len(params))
        )
    var cp = f32_ptr(c_addr)
    var a_address = a_addr
    var b_address = b_addr
    var m = params[0]
    var n = params[1]
    var k = params[2]
    var op = params[3]
    if m <= 0 or n <= 0 or k <= 0:
        raise Error(
            "identical_gemm_host: m, n and k must all be positive, got m="
            + String(m) + " n=" + String(n) + " k=" + String(k)
            + " (contract section 8 specifies the degenerate shapes; this"
            " surface has no gate on them and refuses rather than guesses)"
        )
    if op != OP_NN and op != OP_NT and op != OP_TN:
        raise Error(
            "identical_gemm_host: op must be 0 (OP_NN), 1 (OP_NT) or 2"
            " (OP_TN), got " + String(op)
        )
    if (
        m > LINALG_HOST_MAX_EXTENT or n > LINALG_HOST_MAX_EXTENT
        or k > LINALG_HOST_MAX_EXTENT
    ):
        raise Error("gemm: m, n and k must each be at most 2^30")
    var wrote = 0
    wrote = gemm_host_run(cp, a_address, b_address, op, m, n, k)
    return wrote


def gemm_bf16_native(
    c_addr: Int,
    a_addr: Int,
    b_addr: Int,
    params: List[Int],
) raises -> Int:
    if len(params) != 5:
        raise Error("gemm_bf16: params must contain 5 values (m, n, k, op, a_bf16), got " + String(len(params)))
    var cp = f32_ptr(c_addr)
    var a_address = a_addr
    var b_address = b_addr
    var m = params[0]
    var n = params[1]
    var k = params[2]
    var op = params[3]
    var a_bf16 = params[4] != 0
    if m <= 0 or n <= 0 or k <= 0:
        raise Error("gemm_bf16: m, n and k must all be positive, got m=" + String(m) + " n=" + String(n) + " k=" + String(k))
    if op != OP_NN and op != OP_NT and op != OP_TN:
        raise Error("gemm_bf16: op must be 0 (OP_NN), 1 (OP_NT) or 2 (OP_TN), got " + String(op))
    if m > LINALG_HOST_MAX_EXTENT or n > LINALG_HOST_MAX_EXTENT or k > LINALG_HOST_MAX_EXTENT:
        raise Error("gemm_bf16: m, n and k must each be at most 2^30")
    var wrote = 0
    wrote = gemm_bf16_host_run(cp, a_address, b_address, op, m, n, k, a_bf16)
    return wrote


def gemm_int8_native(
    c_addr: Int,
    qa_addr: Int,
    ea_addr: Int,
    qb_addr: Int,
    eb_addr: Int,
    params: List[Int],
) raises -> Int:
    if len(params) != 3:
        raise Error("gemm_int8: params must contain 3 values (m, n, k), got " + String(len(params)))
    var cp = f32_ptr(c_addr)
    var qa_address = qa_addr
    var ea_address = ea_addr
    var qb_address = qb_addr
    var eb_address = eb_addr
    var m = params[0]
    var n = params[1]
    var k = params[2]
    if m <= 0 or n <= 0 or k <= 0:
        raise Error("gemm_int8: m, n and k must all be positive, got m=" + String(m) + " n=" + String(n) + " k=" + String(k))
    if k > INT8_MAX_K:
        raise Error("gemm_int8: k must be at most " + String(INT8_MAX_K) + " (contract L-7), got " + String(k))
    if m > LINALG_HOST_MAX_EXTENT or n > LINALG_HOST_MAX_EXTENT:
        raise Error("gemm_int8: m and n must each be at most 2^30")
    var wrote = 0
    wrote = gemm_int8_host_run(cp, qa_address, ea_address, qb_address, eb_address, m, n, k)
    return wrote


def quantize_int8_native(
    q_addr: Int,
    e_addr: Int,
    x_addr: Int,
    params: List[Int],
) raises -> Int:
    if len(params) != 2:
        raise Error("quantize_int8: params must contain 2 values (rows, cols), got " + String(len(params)))
    var qp = i8_ptr(q_addr)
    var ep = i32_ptr(e_addr)
    var x_address = x_addr
    var rows = params[0]
    var cols = params[1]
    if rows <= 0 or cols <= 0:
        raise Error("quantize_int8: rows and cols must be positive, got " + String(rows) + " x " + String(cols))
    if rows > LINALG_HOST_MAX_EXTENT or cols > LINALG_HOST_MAX_EXTENT:
        raise Error("quantize_int8: rows and cols must each be at most 2^30")
    var wrote = 0
    wrote = quantize_int8_host_run(qp, ep, x_address, rows, cols)
    return wrote


def dequantize_int8_native(
    y_addr: Int,
    q_addr: Int,
    e_addr: Int,
    params: List[Int],
) raises -> Int:
    if len(params) != 2:
        raise Error("dequantize_int8: params must contain 2 values (rows, cols), got " + String(len(params)))
    var yp = f32_ptr(y_addr)
    var q_address = q_addr
    var e_address = e_addr
    var rows = params[0]
    var cols = params[1]
    if rows <= 0 or cols <= 0:
        raise Error("dequantize_int8: rows and cols must be positive, got " + String(rows) + " x " + String(cols))
    if rows > LINALG_HOST_MAX_EXTENT or cols > LINALG_HOST_MAX_EXTENT:
        raise Error("dequantize_int8: rows and cols must each be at most 2^30")
    var wrote = 0
    wrote = dequantize_int8_host_run(yp, q_address, e_address, rows, cols)
    return wrote


def gemm_int15_native(addrs: List[Int], params: List[Int]) raises -> Int:
    if len(addrs) != 7:
        raise Error("gemm_int15: addrs must contain 7 addresses (c, a hi, a lo, a exponents, b hi, b lo, b exponents), got " + String(len(addrs)))
    if len(params) != 3:
        raise Error("gemm_int15: params must contain 3 values (m, n, k), got " + String(len(params)))
    var cp = f32_ptr(addrs[0])
    var ah_address = addrs[1]
    var al_address = addrs[2]
    var ea_address = addrs[3]
    var bh_address = addrs[4]
    var bl_address = addrs[5]
    var eb_address = addrs[6]
    var m = params[0]
    var n = params[1]
    var k = params[2]
    if m <= 0 or n <= 0 or k <= 0:
        raise Error("gemm_int15: m, n and k must all be positive, got m=" + String(m) + " n=" + String(n) + " k=" + String(k))
    if k > INT15_MAX_K:
        raise Error("gemm_int15: k must be at most " + String(INT15_MAX_K) + " (contract W-4), got " + String(k))
    if m > LINALG_HOST_MAX_EXTENT or n > LINALG_HOST_MAX_EXTENT:
        raise Error("gemm_int15: m and n must each be at most 2^30")
    var wrote = 0
    wrote = gemm_int15_host_run(cp, ah_address, al_address, ea_address, bh_address, bl_address, eb_address, m, n, k)
    return wrote


def quantize_int15_native(addrs: List[Int], params: List[Int]) raises -> Int:
    if len(addrs) != 4:
        raise Error("quantize_int15: addrs must contain 4 addresses (hi, lo, exponents, x), got " + String(len(addrs)))
    if len(params) != 2:
        raise Error("quantize_int15: params must contain 2 values (rows, cols), got " + String(len(params)))
    var hp = i8_ptr(addrs[0])
    var lp = i8_ptr(addrs[1])
    var ep = i32_ptr(addrs[2])
    var x_address = addrs[3]
    var rows = params[0]
    var cols = params[1]
    if rows <= 0 or cols <= 0:
        raise Error("quantize_int15: rows and cols must be positive, got " + String(rows) + " x " + String(cols))
    if rows > LINALG_HOST_MAX_EXTENT or cols > LINALG_HOST_MAX_EXTENT:
        raise Error("quantize_int15: rows and cols must each be at most 2^30")
    var wrote = 0
    wrote = quantize_int15_host_run(hp, lp, ep, x_address, rows, cols)
    return wrote


def dequantize_int15_native(addrs: List[Int], params: List[Int]) raises -> Int:
    if len(addrs) != 4:
        raise Error("dequantize_int15: addrs must contain 4 addresses (y, hi, lo, exponents), got " + String(len(addrs)))
    if len(params) != 2:
        raise Error("dequantize_int15: params must contain 2 values (rows, cols), got " + String(len(params)))
    var yp = f32_ptr(addrs[0])
    var h_address = addrs[1]
    var l_address = addrs[2]
    var e_address = addrs[3]
    var rows = params[0]
    var cols = params[1]
    if rows <= 0 or cols <= 0:
        raise Error("dequantize_int15: rows and cols must be positive, got " + String(rows) + " x " + String(cols))
    if rows > LINALG_HOST_MAX_EXTENT or cols > LINALG_HOST_MAX_EXTENT:
        raise Error("dequantize_int15: rows and cols must each be at most 2^30")
    var wrote = 0
    wrote = dequantize_int15_host_run(yp, h_address, l_address, e_address, rows, cols)
    return wrote


def to_bf16_native(
    dst_addr: Int, src_addr: Int, params: List[Int]
) raises -> Int:
    if len(params) != 1:
        raise Error("to_bf16: params must contain 1 value (count)")
    var dp = u16_ptr(dst_addr)
    var src_address = src_addr
    var count = params[0]
    if count <= 0:
        raise Error("to_bf16: count must be positive, got " + String(count))
    if count > LINALG_HOST_MAX_EXTENT:
        raise Error("to_bf16: count must be at most 2^30")
    to_bf16_host_run(dp, src_address, count)
    return count


def from_bf16_native(
    dst_addr: Int, src_addr: Int, params: List[Int]
) raises -> Int:
    if len(params) != 1:
        raise Error("from_bf16: params must contain 1 value (count)")
    var dp = f32_ptr(dst_addr)
    var src_address = src_addr
    var count = params[0]
    if count <= 0:
        raise Error("from_bf16: count must be positive, got " + String(count))
    if count > LINALG_HOST_MAX_EXTENT:
        raise Error("from_bf16: count must be at most 2^30")
    from_bf16_host_run(dp, src_address, count)
    return count


def from_f16_native(
    dst_addr: Int, src_addr: Int, params: List[Int]
) raises -> Int:
    if len(params) != 1:
        raise Error("from_f16: params must contain 1 value (count)")
    var dp = f32_ptr(dst_addr)
    var src_address = src_addr
    var count = params[0]
    if count <= 0:
        raise Error("from_f16: count must be positive, got " + String(count))
    if count > LINALG_HOST_MAX_EXTENT:
        raise Error("from_f16: count must be at most 2^30")
    from_f16_host_run(dp, src_address, count)
    return count
