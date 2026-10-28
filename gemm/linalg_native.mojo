# SPDX-License-Identifier: Apache-2.0
# Shared native linalg transport. Extracted numerical bodies keep their original
# launch, synchronization and release order; both ABI doors call these owners.
from core.hostptr import f32_ptr, f64_ptr, i8_ptr, i32_ptr, read_f32, u16_ptr
from std.os import abort, getenv
from std.time import perf_counter_ns

from checks.vendor import COMPILED_VENDOR

from max.gpu.host import DeviceContext
from core.neural_context import process_ctx
from checks.numerics import GLOBAL_NUMERIC_MODE as _DEVCTX_MODE, NUMERIC_IDENTICAL as _DEVCTX_IDENTICAL

#: This binding's ONE process-lifetime DeviceContext (core/neural_context.mojo,
#: lane/devctx-lifetime): a context per call exhausts Metal command queues.
comptime _DEVCTX_SLOT = "MojoLinalgContextIdentical" if _DEVCTX_MODE == _DEVCTX_IDENTICAL else "MojoLinalgContextFast"


from x_decomp.cells import F32Ptr
from x_decomp.device import DevExec
from decomposition.linalg_public_device import (
    device_qr_r,
    device_svdvals,
)
from gemm.host_entry import identical_gemm_host
from gemm.host_transport import (
    GemmHostLease,
    ROLE_A,
    ROLE_B,
    ROLE_C,
    ROLE_X,
    gemm_down_f32,
    gemm_up_f32,
    gemm_up_u16,
)
from std.ffi import _Global
from gemm.checks.gemm_lowbit import (
    LowbitWorkspace,
    bf16_narrow,
    bf16_widen,
    dequantize_rows_int8_device,
    identical_gemm_bf16w_into,
    identical_gemm_int8_into,
    quantize_rows_int8_device,
)
from gemm.contract import INT8_MAX_K, LOWBIT_PROFILE_VERSION
from gemm.contract import GEMM_NUMERICAL_PROFILE, GEMM_NUMERICAL_PROFILE_VERSION
from gemm.afn_apple_fast import (
    AFN_EPI_BIAS,
    AFN_EPI_BIAS_GELU,
    AFN_EPI_BIAS_RESID,
    AFN_EPI_BIAS_SILU,
    AFN_EPI_NONE,
    AFN_GEMM_BF16_MMA,
    AFN_GEMM_EPILOGUE,
    afn_gemm_bf16_bits_into,
    afn_gemm_fused_into,
)
from decomposition.linalg_door_impl import from_f16_device, qr_r_device_into, svdvals_device_into
from std.gpu import block_dim, block_idx, thread_idx
from gemm.checks.gemm_int15 import (
    Int15QuantWorkspace,
    dequantize_planes_int15_device,
    identical_gemm_int15_planes_into,
    quantize_planes_int15_parallel_device,
)
from gemm.contract import INT15_MAX_K, INT15_PROFILE_VERSION, OP_NN, OP_NT, OP_TN
from max.gpu.host import DeviceBuffer
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL



def _f32_ptr(addr: Int) raises -> MutPointer[Float32, MutUntrackedOrigin]:
    return f32_ptr(addr)

def _dev_f32(ctx: DeviceContext, addr: Int, count: Int) raises -> DeviceBuffer[DType.float32]:
    var n = count
    if n < 1:
        n = 1
    var d = ctx.enqueue_create_buffer[DType.float32](n)
    if count > 0:
        ctx.enqueue_copy(dst_buf=d, src_ptr=f32_ptr(addr))
    return d^

def _dev_u16(ctx: DeviceContext, addr: Int, count: Int) raises -> DeviceBuffer[DType.uint16]:
    var n = count
    if n < 1:
        n = 1
    var d = ctx.enqueue_create_buffer[DType.uint16](n)
    if count > 0:
        ctx.enqueue_copy(dst_buf=d, src_ptr=u16_ptr(addr))
    return d^

def _dev_i8(ctx: DeviceContext, addr: Int, count: Int) raises -> DeviceBuffer[DType.int8]:
    var n = count
    if n < 1:
        n = 1
    var d = ctx.enqueue_create_buffer[DType.int8](n)
    if count > 0:
        ctx.enqueue_copy(dst_buf=d, src_ptr=i8_ptr(addr))
    return d^

def _dev_i32(ctx: DeviceContext, addr: Int, count: Int) raises -> DeviceBuffer[DType.int32]:
    var n = count
    if n < 1:
        n = 1
    var d = ctx.enqueue_create_buffer[DType.int32](n)
    if count > 0:
        ctx.enqueue_copy(dst_buf=d, src_ptr=i32_ptr(addr))
    return d^

def _refuse_lowbit_shape(m: Int, n: Int, k: Int, op: Int, who: String) raises:
    if m <= 0 or n <= 0 or k <= 0:
        raise Error(
            who + ": m, n and k must all be positive, got m=" + String(m)
            + " n=" + String(n) + " k=" + String(k)
        )
    if op != OP_NN and op != OP_NT and op != OP_TN:
        raise Error(who + ": op must be 0 (OP_NN), 1 (OP_NT) or 2 (OP_TN), got " + String(op))

struct _LowbitSlot(Defaultable, Movable):
    var work: Optional[LowbitWorkspace]

    def __init__(out self):
        self.work = Optional[LowbitWorkspace]()

comptime _LOWBIT_POOL = _Global[StorageType=_LowbitSlot, name="MojoGemmLowbitPool", init_fn=_LowbitSlot.__init__]

def _lowbit_run(
    ctx: DeviceContext,
    lease: GemmHostLease,
    mut dc: DeviceBuffer[DType.float32],
    mut da: DeviceBuffer[DType.float32],
    mut db: DeviceBuffer[DType.uint16],
    m: Int, n: Int, k: Int, op: Int,
) raises:
    """`identical_gemm_bf16w_into` then a wait, on the pooled workspace when
    the lease holds the pool (only the lease holder touches the slot)."""
    if lease.held:
        var slot = _LOWBIT_POOL.get_or_create_ptr()
        if not slot[].work:
            slot[].work = LowbitWorkspace(ctx)
        identical_gemm_bf16w_into(ctx, dc, da, db, slot[].work.value(), m, n, k, op)
        ctx.synchronize()
        return
    var work = LowbitWorkspace(ctx)
    identical_gemm_bf16w_into(ctx, dc, da, db, work, m, n, k, op)
    ctx.synchronize()
    _ = work^

def _lowbit_tick(ctx: DeviceContext, on: Bool, mut t: Int, name: String) raises:
    """MOJOLEARN_LOWBIT_TIMING: synchronize, print `timing int8.<name> <ms>
    ms`, advance `t`. A no-op when off."""
    if not on:
        return
    ctx.synchronize()
    var now = Int(perf_counter_ns())
    print("timing int8." + name + " " + String(Float64(now - t) / 1000000.0) + " ms")
    t = now

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
    var cp = _f32_ptr(c_addr)
    var ap = _f32_ptr(a_addr)
    var bp = _f32_ptr(b_addr)
    var m = params[0]
    var n = params[1]
    var k = params[2]
    var op = params[3]
    var ctx = process_ctx[_DEVCTX_SLOT]()
    identical_gemm_host(ctx, cp, ap, bp, m, n, k, op)
    # DEVIATION 3010: the buffers above are gone; DRAIN the frees they
    # enqueued before this block's end destroys the context. Without it
    # the MAX runtime allocator's lock is left held and the NEXT
    # enqueue_create_buffer in the process blocks for ever (DEVIATION
    # 2520's mechanism; the native backtrace of the `transformer-bf16w`
    # hang blames exactly this library). Host-side drain, no arithmetic.
    ctx.synchronize()
    return m * n

def gemm_bf16_native(
    c_addr: Int,
    a_addr: Int,
    b_addr: Int,
    params: List[Int],
) raises -> Int:
    if len(params) != 5:
        raise Error("gemm_bf16: params must contain 5 values (m, n, k, op, a_bf16), got " + String(len(params)))
    var c_address = c_addr
    var a_address = a_addr
    var b_address = b_addr
    var m = params[0]
    var n = params[1]
    var k = params[2]
    var op = params[3]
    var a_bf16 = params[4] != 0
    _refuse_lowbit_shape(m, n, k, op, String("gemm_bf16"))
    var ctx = process_ctx[_DEVCTX_SLOT]()
    # lane/gap-neural-models (2026-10-02): pooled buffers and workspace,
    # staged download (gemm/host_transport.mojo; MOJOLEARN_GEMM_POOL=0
    # restores fresh buffers). Copies only, the same kernels.
    var lease = GemmHostLease()
    var db = lease.u16(ctx, ROLE_B, n * k)
    gemm_up_u16(ctx, lease, db, u16_ptr(b_address), n * k)
    var dc = lease.f32(ctx, ROLE_C, m * n)
    var da = lease.f32(ctx, ROLE_X, m * k)
    if a_bf16:
        var da_bits = lease.u16(ctx, ROLE_A, m * k)
        gemm_up_u16(ctx, lease, da_bits, u16_ptr(a_address), m * k)
        # lane/apple-fast-neural-gemm (2026-10-03): under FAST on Apple
        # with MOJOLEARN_AFN_GEMM_BF16_MMA both operands' bits go straight
        # to the simdgroup kernel (no widen launch, no float32 image of
        # A); otherwise, and under IDENTICAL, the lines below unchanged.
        var served = False
        comptime if AFN_GEMM_BF16_MMA:
            served = afn_gemm_bf16_bits_into(ctx, dc, da_bits, db, m, n, k, op)
            if served:
                ctx.synchronize()
        if not served:
            bf16_widen(ctx, da, da_bits, m * k)
            _lowbit_run(ctx, lease, dc, da, db, m, n, k, op)
        _ = da_bits
    else:
        gemm_up_f32(ctx, lease, da, f32_ptr(a_address), m * k)
        _lowbit_run(ctx, lease, dc, da, db, m, n, k, op)
    gemm_down_f32(ctx, lease, dc, f32_ptr(c_address), m * n)
    lease.release()
    _ = db^
    _ = dc^
    _ = da^
    # DEVIATION 3010: the buffers above are gone; DRAIN the frees they
    # enqueued before this block's end destroys the context. Without it
    # the MAX runtime allocator's lock is left held and the NEXT
    # enqueue_create_buffer in the process blocks for ever (DEVIATION
    # 2520's mechanism; the native backtrace of the `transformer-bf16w`
    # hang blames exactly this library). Host-side drain, no arithmetic.
    ctx.synchronize()
    return m * n

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
    var c_address = c_addr
    var qa_address = qa_addr
    var ea_address = ea_addr
    var qb_address = qb_addr
    var eb_address = eb_addr
    var m = params[0]
    var n = params[1]
    var k = params[2]
    _refuse_lowbit_shape(m, n, k, OP_NT, String("gemm_int8"))
    if k > INT8_MAX_K:
        raise Error("gemm_int8: k must be at most " + String(INT8_MAX_K) + " (contract L-7), got " + String(k))
    # lane/neural-net-experiment (2026-09-30): MOJOLEARN_LOWBIT_TIMING=1
    # prints one line per phase of this call (`timing int8.<phase> <ms>
    # ms`), a synchronize around each, so the board's gemm-int8 cell can be
    # read as context + uploads + kernel + download + drains. Measurement
    # only: the phases and their order are unchanged when it is off.
    var ton = String(getenv("MOJOLEARN_LOWBIT_TIMING")) != ""
    var tk = Int(perf_counter_ns())
    var ctx = process_ctx[_DEVCTX_SLOT]()
    _lowbit_tick(ctx, ton, tk, "context")
    var dqa = _dev_i8(ctx, qa_address, m * k)
    var dea = _dev_i32(ctx, ea_address, m)
    var dqb = _dev_i8(ctx, qb_address, n * k)
    var deb = _dev_i32(ctx, eb_address, n)
    var dc = ctx.enqueue_create_buffer[DType.float32](m * n)
    _lowbit_tick(ctx, ton, tk, "uploads_and_alloc")
    identical_gemm_int8_into(ctx, dc, dqa, dea, dqb, deb, m, n, k)
    ctx.synchronize()
    _lowbit_tick(ctx, ton, tk, "kernel")
    ctx.enqueue_copy(dst_ptr=f32_ptr(c_address), src_buf=dc)
    ctx.synchronize()
    _lowbit_tick(ctx, ton, tk, "download")
    _ = dqa^
    _ = dea^
    _ = dqb^
    _ = deb^
    _ = dc^
    # DEVIATION 3010: the buffers above are gone; DRAIN the frees they
    # enqueued before this block's end destroys the context. Without it
    # the MAX runtime allocator's lock is left held and the NEXT
    # enqueue_create_buffer in the process blocks for ever (DEVIATION
    # 2520's mechanism; the native backtrace of the `transformer-bf16w`
    # hang blames exactly this library). Host-side drain, no arithmetic.
    ctx.synchronize()
    _lowbit_tick(ctx, ton, tk, "free_and_drain")
    return m * n

def quantize_int8_native(
    q_addr: Int,
    e_addr: Int,
    x_addr: Int,
    params: List[Int],
) raises -> Int:
    if len(params) != 2:
        raise Error("quantize_int8: params must contain 2 values (rows, cols), got " + String(len(params)))
    var q_address = q_addr
    var e_address = e_addr
    var x_address = x_addr
    var rows = params[0]
    var cols = params[1]
    if rows <= 0 or cols <= 0:
        raise Error("quantize_int8: rows and cols must be positive, got " + String(rows) + " x " + String(cols))
    var ctx = process_ctx[_DEVCTX_SLOT]()
    var dx = _dev_f32(ctx, x_address, rows * cols)
    var dq = ctx.enqueue_create_buffer[DType.int8](rows * cols)
    var de = ctx.enqueue_create_buffer[DType.int32](rows)
    quantize_rows_int8_device(ctx, dq, de, dx, rows, cols)
    ctx.synchronize()
    ctx.enqueue_copy(dst_ptr=i8_ptr(q_address), src_buf=dq)
    ctx.enqueue_copy(dst_ptr=i32_ptr(e_address), src_buf=de)
    ctx.synchronize()
    _ = dx^
    _ = dq^
    _ = de^
    # DEVIATION 3010: the buffers above are gone; DRAIN the frees they
    # enqueued before this block's end destroys the context. Without it
    # the MAX runtime allocator's lock is left held and the NEXT
    # enqueue_create_buffer in the process blocks for ever (DEVIATION
    # 2520's mechanism; the native backtrace of the `transformer-bf16w`
    # hang blames exactly this library). Host-side drain, no arithmetic.
    ctx.synchronize()
    return rows * cols

def dequantize_int8_native(
    y_addr: Int,
    q_addr: Int,
    e_addr: Int,
    params: List[Int],
) raises -> Int:
    if len(params) != 2:
        raise Error("dequantize_int8: params must contain 2 values (rows, cols), got " + String(len(params)))
    var y_address = y_addr
    var q_address = q_addr
    var e_address = e_addr
    var rows = params[0]
    var cols = params[1]
    if rows <= 0 or cols <= 0:
        raise Error("dequantize_int8: rows and cols must be positive, got " + String(rows) + " x " + String(cols))
    var ctx = process_ctx[_DEVCTX_SLOT]()
    var dq = _dev_i8(ctx, q_address, rows * cols)
    var de = _dev_i32(ctx, e_address, rows)
    var dy = ctx.enqueue_create_buffer[DType.float32](rows * cols)
    dequantize_rows_int8_device(ctx, dy, dq, de, rows, cols)
    ctx.synchronize()
    ctx.enqueue_copy(dst_ptr=f32_ptr(y_address), src_buf=dy)
    ctx.synchronize()
    _ = dq^
    _ = de^
    _ = dy^
    # DEVIATION 3010: the buffers above are gone; DRAIN the frees they
    # enqueued before this block's end destroys the context. Without it
    # the MAX runtime allocator's lock is left held and the NEXT
    # enqueue_create_buffer in the process blocks for ever (DEVIATION
    # 2520's mechanism; the native backtrace of the `transformer-bf16w`
    # hang blames exactly this library). Host-side drain, no arithmetic.
    ctx.synchronize()
    return rows * cols

def gemm_int15_native(addrs: List[Int], params: List[Int]) raises -> Int:
    if len(addrs) != 7:
        raise Error("gemm_int15: addrs must contain 7 addresses (c, a hi, a lo, a exponents, b hi, b lo, b exponents), got " + String(len(addrs)))
    if len(params) != 3:
        raise Error("gemm_int15: params must contain 3 values (m, n, k), got " + String(len(params)))
    var c_address = addrs[0]
    var ah_address = addrs[1]
    var al_address = addrs[2]
    var ea_address = addrs[3]
    var bh_address = addrs[4]
    var bl_address = addrs[5]
    var eb_address = addrs[6]
    var m = params[0]
    var n = params[1]
    var k = params[2]
    _refuse_lowbit_shape(m, n, k, OP_NT, String("gemm_int15"))
    if k > INT15_MAX_K:
        raise Error("gemm_int15: k must be at most " + String(INT15_MAX_K) + " (contract W-4), got " + String(k))
    var ctx = process_ctx[_DEVCTX_SLOT]()
    var dah = _dev_i8(ctx, ah_address, m * k)
    var dal = _dev_i8(ctx, al_address, m * k)
    var dea = _dev_i32(ctx, ea_address, m)
    var dbh = _dev_i8(ctx, bh_address, n * k)
    var dbl = _dev_i8(ctx, bl_address, n * k)
    var deb = _dev_i32(ctx, eb_address, n)
    var dc = ctx.enqueue_create_buffer[DType.float32](m * n)
    identical_gemm_int15_planes_into(ctx, dc, dah, dal, dea, dbh, dbl, deb, m, n, k)
    ctx.synchronize()
    ctx.enqueue_copy(dst_ptr=f32_ptr(c_address), src_buf=dc)
    ctx.synchronize()
    _ = dah^
    _ = dal^
    _ = dea^
    _ = dbh^
    _ = dbl^
    _ = deb^
    _ = dc^
    # DEVIATION 3010: drain the frees the buffers above enqueued.
    ctx.synchronize()
    return m * n

def quantize_int15_native(addrs: List[Int], params: List[Int]) raises -> Int:
    if len(addrs) != 4:
        raise Error("quantize_int15: addrs must contain 4 addresses (hi, lo, exponents, x), got " + String(len(addrs)))
    if len(params) != 2:
        raise Error("quantize_int15: params must contain 2 values (rows, cols), got " + String(len(params)))
    var h_address = addrs[0]
    var l_address = addrs[1]
    var e_address = addrs[2]
    var x_address = addrs[3]
    var rows = params[0]
    var cols = params[1]
    if rows <= 0 or cols <= 0:
        raise Error("quantize_int15: rows and cols must be positive, got " + String(rows) + " x " + String(cols))
    var ctx = process_ctx[_DEVCTX_SLOT]()
    var dx = _dev_f32(ctx, x_address, rows * cols)
    var dh = ctx.enqueue_create_buffer[DType.int8](rows * cols)
    var dl = ctx.enqueue_create_buffer[DType.int8](rows * cols)
    var de = ctx.enqueue_create_buffer[DType.int32](rows)
    var quant = Int15QuantWorkspace(ctx)
    quantize_planes_int15_parallel_device(ctx, dh, dl, de, dx, quant, rows, cols, False)
    ctx.synchronize()
    ctx.enqueue_copy(dst_ptr=i8_ptr(h_address), src_buf=dh)
    ctx.enqueue_copy(dst_ptr=i8_ptr(l_address), src_buf=dl)
    ctx.enqueue_copy(dst_ptr=i32_ptr(e_address), src_buf=de)
    ctx.synchronize()
    _ = dx^
    _ = dh^
    _ = dl^
    _ = de^
    _ = quant^
    # DEVIATION 3010: drain the frees the buffers above enqueued.
    ctx.synchronize()
    return rows * cols

def dequantize_int15_native(addrs: List[Int], params: List[Int]) raises -> Int:
    if len(addrs) != 4:
        raise Error("dequantize_int15: addrs must contain 4 addresses (y, hi, lo, exponents), got " + String(len(addrs)))
    if len(params) != 2:
        raise Error("dequantize_int15: params must contain 2 values (rows, cols), got " + String(len(params)))
    var y_address = addrs[0]
    var h_address = addrs[1]
    var l_address = addrs[2]
    var e_address = addrs[3]
    var rows = params[0]
    var cols = params[1]
    if rows <= 0 or cols <= 0:
        raise Error("dequantize_int15: rows and cols must be positive, got " + String(rows) + " x " + String(cols))
    var ctx = process_ctx[_DEVCTX_SLOT]()
    var dh = _dev_i8(ctx, h_address, rows * cols)
    var dl = _dev_i8(ctx, l_address, rows * cols)
    var de = _dev_i32(ctx, e_address, rows)
    var dy = ctx.enqueue_create_buffer[DType.float32](rows * cols)
    dequantize_planes_int15_device(ctx, dy, dh, dl, de, rows, cols)
    ctx.synchronize()
    ctx.enqueue_copy(dst_ptr=f32_ptr(y_address), src_buf=dy)
    ctx.synchronize()
    _ = dh^
    _ = dl^
    _ = de^
    _ = dy^
    # DEVIATION 3010: drain the frees the buffers above enqueued.
    ctx.synchronize()
    return rows * cols

def to_bf16_native(
    dst_addr: Int, src_addr: Int, params: List[Int]
) raises -> Int:
    if len(params) != 1:
        raise Error("to_bf16: params must contain 1 value (count)")
    var dst_address = dst_addr
    var src_address = src_addr
    var count = params[0]
    if count <= 0:
        raise Error("to_bf16: count must be positive, got " + String(count))
    var ctx = process_ctx[_DEVCTX_SLOT]()
    var dsrc = _dev_f32(ctx, src_address, count)
    var ddst = ctx.enqueue_create_buffer[DType.uint16](count)
    bf16_narrow(ctx, ddst, dsrc, count)
    ctx.synchronize()
    ctx.enqueue_copy(dst_ptr=u16_ptr(dst_address), src_buf=ddst)
    ctx.synchronize()
    _ = dsrc^
    _ = ddst^
    # DEVIATION 3010: the buffers above are gone; DRAIN the frees they
    # enqueued before this block's end destroys the context. Without it
    # the MAX runtime allocator's lock is left held and the NEXT
    # enqueue_create_buffer in the process blocks for ever (DEVIATION
    # 2520's mechanism; the native backtrace of the `transformer-bf16w`
    # hang blames exactly this library). Host-side drain, no arithmetic.
    ctx.synchronize()
    return count

def from_bf16_native(
    dst_addr: Int, src_addr: Int, params: List[Int]
) raises -> Int:
    if len(params) != 1:
        raise Error("from_bf16: params must contain 1 value (count)")
    var dst_address = dst_addr
    var src_address = src_addr
    var count = params[0]
    if count <= 0:
        raise Error("from_bf16: count must be positive, got " + String(count))
    var ctx = process_ctx[_DEVCTX_SLOT]()
    var dsrc = _dev_u16(ctx, src_address, count)
    var ddst = ctx.enqueue_create_buffer[DType.float32](count)
    bf16_widen(ctx, ddst, dsrc, count)
    ctx.synchronize()
    ctx.enqueue_copy(dst_ptr=f32_ptr(dst_address), src_buf=ddst)
    ctx.synchronize()
    _ = dsrc^
    _ = ddst^
    # DEVIATION 3010: the buffers above are gone; DRAIN the frees they
    # enqueued before this block's end destroys the context. Without it
    # the MAX runtime allocator's lock is left held and the NEXT
    # enqueue_create_buffer in the process blocks for ever (DEVIATION
    # 2520's mechanism; the native backtrace of the `transformer-bf16w`
    # hang blames exactly this library). Host-side drain, no arithmetic.
    ctx.synchronize()
    return count

def from_f16_native(
    dst_addr: Int, src_addr: Int, params: List[Int]
) raises -> Int:
    if len(params) != 1:
        raise Error("from_f16: params must contain 1 value (count)")
    var dst_address = dst_addr
    var src_address = src_addr
    var count = params[0]
    if count <= 0:
        raise Error("from_f16: count must be positive, got " + String(count))
    var ctx = process_ctx[_DEVCTX_SLOT]()
    from_f16_device(ctx, f32_ptr(dst_address), u16_ptr(src_address), count)
    return count

