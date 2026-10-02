# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""SVC's binary64 epilogues and row glue ON THE DEVICE (cgfin-c-svm, 2026-10-02).

Until this module the GPU binding imported `svm/host/svc_proba.mojo` and ran
these on host threads: the per-row epilogues of `decision_function`,
`predict`, `predict_proba` and `predict_log_proba`, Platt's `sigmoid_train`,
the probability shuffle and the `_portable_math` check door. They run here
as grid kernels now, plus three row-glue ops `_svm_impl.py` ran as Python
loops over n (the pair row selection, the gathers of rows and columns, the
per-row bounds `C * w * class_weight`).

BINARY64 WITHOUT A FLOAT64 UNIT. The Apple GPU has no float64, so every
binary64 value is a UInt64 bit pattern and every operation is
`checks/soft_f64.mojo`'s correctly rounded integer arithmetic. The host
column (`svm/host/svc_proba.mojo`, the CPU binding) calls the SAME functions
below, so NVIDIA, AMD, Apple and the host compute the same words.

THE SAME BITS AS BEFORE, EXCEPT PLATT AND THE SHUFFLE:
  * the epilogues transcribe the old host code operation for operation
    (`sp_exp` is `pm_exp`, i.e. `mojolearn_exp` of portable_math.c with its
    two roundings of `x * log2(e) + 0.5`; `sf64_log` is `pm_log` statement
    for statement), so their outputs keep their bits;
  * Platt's sums over the rows were one sequential chain. They are
    fixed-order grid folds now, `grid_fold.mojo`'s shape over binary64
    words (chunks of FOLD_TPB rows, a halving tree per chunk, levels until
    one chunk remains; padding +0.0), and `h11`, `h22` are `sigma + sum`.
    probA_ and probB_ change in the last places;
  * the shuffle was a serial Fisher-Yates. It is `core/shuffle_iterator`'s
    Feistel bijection with its cycle walk now, one thread per index, keyed
    by the low 32 bits of one SplitMix64 step of the 64-bit seed. The fold
    membership of the probability cross-validation changes.
"""

from std.gpu import block_idx, thread_idx
from std.memory import bitcast, stack_allocation
from std.python import Python, PythonObject
from std.python._cpython import GILReleased
from max.gpu.host import DeviceBuffer, DeviceContext
from max.gpu.memory import AddressSpace
from max.gpu.sync import barrier

from checks.soft_f64 import SF64_ZERO, sf64_add, sf64_log
from core.shuffle_iterator import FeistelBijection
from svm.estimator import _family_ctx
from svm.impl.svc_rows import (
    EPI_BINARY_CODES,
    EPI_LOG_PROBA,
    EPI_OVO,
    EPI_OVR,
    EPI_PROBA,
    FOLD_TPB,
    GLUE_C_ROWS,
    GLUE_GATHER,
    GLUE_SELECT,
    I32P,
    PlattSums,
    ST_OK,
    U32P,
    U64P,
    _F32_ONE,
    c_row,
    epilogue_row,
    epilogue_scratch,
    fold_blocks,
    gather_cell,
    platt_channels,
    platt_solve,
    platt_terms,
    shuffle_seed32,
    sp_exp,
)

def epilogue_kernel(
    mode: Int32, dec: U32P, n: Int32, row0: Int32, rows: Int32, n_pairs: Int32,
    k: Int32, pi: I32P, ab: U64P, label1: UInt64, scr: U64P, per_row: Int32,
    out32: U32P, out64: U64P, status: I32P,
):
    """One thread per row of `[row0, row0 + rows)`; rows are independent."""
    var t = Int(block_idx.x) * FOLD_TPB + Int(thread_idx.x)
    if t >= Int(rows):
        return
    var st = epilogue_row(
        Int(mode), dec, Int(n), Int(n_pairs), Int(k), pi, ab, label1,
        Int(row0) + t, scr + t * Int(per_row), out32, out64,
    )
    if st != ST_OK:
        status[0] = Int32(st)


def platt_level0_kernel(
    mode: Int32, dst: U64P, dec: U64P, lab: U64P, n_in: Int32,
    a: UInt64, b: UInt64, hi_t: UInt64, lo_t: UInt64, nb: Int32,
):
    """The first fold level: block `blk` forms its chunk's terms and folds
    each channel by the halving tree into `dst[ch * nb + blk]`."""
    var tid = Int(thread_idx.x)
    var blk = Int(block_idx.x)
    var i = blk * FOLD_TPB + tid
    var v = SIMD[DType.uint64, 8](0)
    if i < Int(n_in):
        v = platt_terms(Int(mode), dec[i], lab[i], a, b, hi_t, lo_t)
    var s = stack_allocation[FOLD_TPB, Scalar[DType.uint64], address_space = AddressSpace.SHARED]()
    for ch in range(platt_channels(Int(mode))):
        s[tid] = v[ch]
        barrier()
        var step = FOLD_TPB // 2
        while step > 0:
            if tid < step:
                s[tid] = sf64_add(s[tid], s[tid + step])
            barrier()
            step //= 2
        if tid == 0:
            dst[ch * Int(nb) + blk] = s[0]
        barrier()


def sf64_level_kernel(dst: U64P, src: U64P, n_in: Int32, n_ch: Int32, nb: Int32):
    """A later fold level over `n_ch` channels laid out `src[ch * n_in + i]`."""
    var tid = Int(thread_idx.x)
    var blk = Int(block_idx.x)
    var i = blk * FOLD_TPB + tid
    var s = stack_allocation[FOLD_TPB, Scalar[DType.uint64], address_space = AddressSpace.SHARED]()
    for ch in range(Int(n_ch)):
        var v = SF64_ZERO
        if i < Int(n_in):
            v = src[ch * Int(n_in) + i]
        s[tid] = v
        barrier()
        var step = FOLD_TPB // 2
        while step > 0:
            if tid < step:
                s[tid] = sf64_add(s[tid], s[tid + step])
            barrier()
            step //= 2
        if tid == 0:
            dst[ch * Int(nb) + blk] = s[0]
        barrier()


struct DevicePlatt(PlattSums, Movable):
    """The Platt sums on the device: the rows uploaded once, one grid fold
    per evaluation, one 64 B read back."""

    var ctx: DeviceContext
    var n: Int
    var dec: DeviceBuffer[DType.uint64]
    var lab: DeviceBuffer[DType.uint64]
    var pa: DeviceBuffer[DType.uint64]
    var pb: DeviceBuffer[DType.uint64]

    def __init__(out self, ctx: DeviceContext, dec_addr: Int, lab_addr: Int, n: Int) raises:
        self.ctx = ctx
        self.n = n
        self.dec = ctx.enqueue_create_buffer[DType.uint64](n)
        self.lab = ctx.enqueue_create_buffer[DType.uint64](n)
        var m = 5 * fold_blocks(n) + 8
        self.pa = ctx.enqueue_create_buffer[DType.uint64](m)
        self.pb = ctx.enqueue_create_buffer[DType.uint64](m)
        ctx.enqueue_copy(dst_buf=self.dec, src_ptr=U64P(unsafe_from_address=dec_addr))
        ctx.enqueue_copy(dst_buf=self.lab, src_ptr=U64P(unsafe_from_address=lab_addr))

    def platt_sums(
        mut self, mode: Int, a: UInt64, b: UInt64, hi_t: UInt64, lo_t: UInt64
    ) raises -> SIMD[DType.uint64, 8]:
        var nch = platt_channels(mode)
        var nb = fold_blocks(self.n)
        self.ctx.enqueue_function[platt_level0_kernel](
            Int32(mode), self.pa.unsafe_ptr(), self.dec.unsafe_ptr(), self.lab.unsafe_ptr(),
            Int32(self.n), a, b, hi_t, lo_t, Int32(nb),
            grid_dim=nb, block_dim=FOLD_TPB,
        )
        var cur_a = True
        var n_cur = nb
        while n_cur > 1:
            var nb2 = fold_blocks(n_cur)
            var src = self.pa.unsafe_ptr() if cur_a else self.pb.unsafe_ptr()
            var dst = self.pb.unsafe_ptr() if cur_a else self.pa.unsafe_ptr()
            self.ctx.enqueue_function[sf64_level_kernel](
                dst, src, Int32(n_cur), Int32(nch), Int32(nb2),
                grid_dim=nb2, block_dim=FOLD_TPB,
            )
            cur_a = not cur_a
            n_cur = nb2
        var h = self.ctx.enqueue_create_host_buffer[DType.uint64](8)
        if cur_a:
            self.ctx.enqueue_copy(dst_ptr=h.unsafe_ptr(), src_buf=self.pa.create_sub_buffer[DType.uint64](0, 8))
        else:
            self.ctx.enqueue_copy(dst_ptr=h.unsafe_ptr(), src_buf=self.pb.create_sub_buffer[DType.uint64](0, 8))
        self.ctx.synchronize()
        var out = SIMD[DType.uint64, 8](0)
        for ch in range(nch):
            out[ch] = h.unsafe_ptr()[ch]
        _ = h^
        return out


# ------------------------------------------------------------ the shuffle
def perm_kernel(dst: I32P, n: Int32, seed: UInt32):
    """`dst[i]` = the Feistel bijection of `[0, n)` at i (cycle walk)."""
    var i = Int(block_idx.x) * FOLD_TPB + Int(thread_idx.x)
    if i < Int(n):
        var f = FeistelBijection(Int(n), seed)
        dst[i] = Int32(f(i))


def pmath_kernel(dst: U64P, src: U64P, n: Int32, which: Int32):
    var i = Int(block_idx.x) * FOLD_TPB + Int(thread_idx.x)
    if i < Int(n):
        dst[i] = sp_exp(src[i]) if Int(which) == 0 else sf64_log(src[i])


# ------------------------------------------------------------ the row glue
def gather_kernel(
    src: U32P, n_src_rows: Int64, n_src_cols: Int64, rows: I32P, has_rows: Int32,
    cols: I32P, has_cols: Int32, n_out_cols: Int64, total: Int64, out: U32P, status: I32P,
):
    var cell = Int(block_idx.x) * FOLD_TPB + Int(thread_idx.x)
    if cell < Int(total):
        var st = gather_cell(
            cell, src, Int(n_src_rows), Int(n_src_cols), rows, has_rows != 0,
            cols, has_cols != 0, Int(n_out_cols), out,
        )
        if st != ST_OK:
            status[0] = Int32(st)


def select_count_kernel(codes: I32P, n: Int32, ci: Int32, cj: Int32, part: I32P):
    """Block `blk`'s count of the rows of class ci or cj in its chunk."""
    var tid = Int(thread_idx.x)
    var blk = Int(block_idx.x)
    var i = blk * FOLD_TPB + tid
    var s = stack_allocation[FOLD_TPB, Scalar[DType.int32], address_space = AddressSpace.SHARED]()
    var f = Int32(0)
    if i < Int(n):
        var c = codes[i]
        if c == ci or c == cj:
            f = 1
    s[tid] = f
    barrier()
    var step = FOLD_TPB // 2
    while step > 0:
        if tid < step:
            s[tid] = s[tid] + s[tid + step]
        barrier()
        step //= 2
    if tid == 0:
        part[blk] = s[0]


def select_emit_kernel(
    codes: I32P, n: Int32, ci: Int32, cj: Int32, part: I32P, nb: Int32,
    cvec: U32P, has_c: Int32, out_idx: I32P, out_lab: U32P, out_c: U32P, count: I32P,
):
    """The rows of class ci or cj in row order: block `blk` sums the counts
    of the blocks before it (integers, exact in any order), scans its
    flags, and writes each kept row's index, its label (1.0 for class cj,
    0.0 for ci) and, when given, its bound."""
    var tid = Int(thread_idx.x)
    var blk = Int(block_idx.x)
    var s = stack_allocation[FOLD_TPB, Scalar[DType.int32], address_space = AddressSpace.SHARED]()
    var acc = Int32(0)
    for t in range(tid, blk, FOLD_TPB):
        acc += part[t]
    s[tid] = acc
    barrier()
    var step = FOLD_TPB // 2
    while step > 0:
        if tid < step:
            s[tid] = s[tid] + s[tid + step]
        barrier()
        step //= 2
    var off = s[0]
    barrier()
    var i = blk * FOLD_TPB + tid
    var f = Int32(0)
    var c = Int32(-1)
    if i < Int(n):
        c = codes[i]
        if c == ci or c == cj:
            f = 1
    s[tid] = f
    barrier()
    var d = 1
    while d < FOLD_TPB:
        var v = s[tid]
        if tid >= d:
            v += s[tid - d]
        barrier()
        s[tid] = v
        barrier()
        d *= 2
    if f != 0:
        var pos = Int(off + s[tid] - 1)
        out_idx[pos] = Int32(i)
        out_lab[pos] = _F32_ONE if c == cj else UInt32(0)
        if has_c != 0:
            out_c[pos] = cvec[i]
    if blk == Int(nb) - 1 and tid == FOLD_TPB - 1:
        count[0] = off + s[tid]


def c_rows_kernel(
    sw: U64P, has_sw: Int32, codes: I32P, cw: U64P, has_cw: Int32, k: Int32,
    c: UInt64, n: Int32, out: U32P, status: I32P,
):
    var i = Int(block_idx.x) * FOLD_TPB + Int(thread_idx.x)
    if i < Int(n):
        var st = c_row(i, sw, has_sw != 0, codes, cw, has_cw != 0, Int(k), c, out)
        if st != ST_OK:
            status[0] = Int32(st)


# --------------------------------------------------------- device drivers
def _grid(count: Int) -> Int:
    return max(1, (count + FOLD_TPB - 1) // FOLD_TPB)


def _up_u32(ctx: DeviceContext, addr: Int, count: Int) raises -> DeviceBuffer[DType.uint32]:
    var buf = ctx.enqueue_create_buffer[DType.uint32](max(1, count))
    if count > 0 and addr != 0:
        ctx.enqueue_copy(dst_buf=buf.create_sub_buffer[DType.uint32](0, count), src_ptr=U32P(unsafe_from_address=addr))
    return buf^


def _up_i32(ctx: DeviceContext, addr: Int, count: Int) raises -> DeviceBuffer[DType.int32]:
    var buf = ctx.enqueue_create_buffer[DType.int32](max(1, count))
    if count > 0 and addr != 0:
        ctx.enqueue_copy(dst_buf=buf.create_sub_buffer[DType.int32](0, count), src_ptr=I32P(unsafe_from_address=addr))
    return buf^


def _up_u64(ctx: DeviceContext, addr: Int, count: Int) raises -> DeviceBuffer[DType.uint64]:
    var buf = ctx.enqueue_create_buffer[DType.uint64](max(1, count))
    if count > 0 and addr != 0:
        ctx.enqueue_copy(dst_buf=buf.create_sub_buffer[DType.uint64](0, count), src_ptr=U64P(unsafe_from_address=addr))
    return buf^


def _status(ctx: DeviceContext) raises -> DeviceBuffer[DType.int32]:
    var st = ctx.enqueue_create_buffer[DType.int32](1)
    ctx.enqueue_memset(st, Int32(0))
    return st^


def _read_status(ctx: DeviceContext, st: DeviceBuffer[DType.int32]) raises -> Int:
    var h = ctx.enqueue_create_host_buffer[DType.int32](1)
    ctx.enqueue_copy(dst_ptr=h.unsafe_ptr(), src_buf=st)
    ctx.synchronize()
    var v = Int(h.unsafe_ptr()[0])
    _ = h^
    return v


def device_pair_epilogue(
    mode: Int, dec_addr: Int, n: Int, n_pairs: Int, k: Int, pairs_addr: Int,
    ab_addr: Int, label1: UInt64, out_addr: Int,
) raises:
    """The epilogue of `mode` over n rows on the device, written at
    `out_addr` (float32 n x P for EPI_OVO; 8-byte words otherwise)."""
    var ctx = _family_ctx()
    var d_dec = _up_u32(ctx, dec_addr, n_pairs * n)
    var npi = 2 * n_pairs if mode != EPI_BINARY_CODES else 0
    var d_pi = _up_i32(ctx, pairs_addr, npi)
    var nab = 2 * n_pairs if mode == EPI_PROBA or mode == EPI_LOG_PROBA else 0
    var d_ab = _up_u64(ctx, ab_addr, nab)
    var n_out32 = n * n_pairs if mode == EPI_OVO else 0
    var n_out64 = 0
    if mode == EPI_OVR or mode == EPI_PROBA or mode == EPI_LOG_PROBA:
        n_out64 = n * k
    elif mode == EPI_VOTES or mode == EPI_BINARY_CODES:
        n_out64 = n
    var d_o32 = ctx.enqueue_create_buffer[DType.uint32](max(1, n_out32))
    var d_o64 = ctx.enqueue_create_buffer[DType.uint64](max(1, n_out64))
    var per_row = epilogue_scratch(mode, k)
    var chunk = n if per_row == 0 else max(1, min(n, (1 << 23) // per_row))
    var d_scr = ctx.enqueue_create_buffer[DType.uint64](max(1, chunk * per_row))
    var st = _status(ctx)
    var row0 = 0
    while row0 < n:
        var rows = min(chunk, n - row0)
        ctx.enqueue_function[epilogue_kernel](
            Int32(mode), d_dec.unsafe_ptr(), Int32(n), Int32(row0), Int32(rows), Int32(n_pairs),
            Int32(k), d_pi.unsafe_ptr(), d_ab.unsafe_ptr(), label1, d_scr.unsafe_ptr(),
            Int32(per_row), d_o32.unsafe_ptr(), d_o64.unsafe_ptr(), st.unsafe_ptr(),
            grid_dim=_grid(rows), block_dim=FOLD_TPB,
        )
        row0 += rows
    if n_out32 > 0:
        ctx.enqueue_copy(dst_ptr=U32P(unsafe_from_address=out_addr), src_buf=d_o32.create_sub_buffer[DType.uint32](0, n_out32))
    if n_out64 > 0:
        ctx.enqueue_copy(dst_ptr=U64P(unsafe_from_address=out_addr), src_buf=d_o64.create_sub_buffer[DType.uint64](0, n_out64))
    var status = _read_status(ctx, st)
    _ = d_dec^
    _ = d_pi^
    _ = d_ab^
    _ = d_o32^
    _ = d_o64^
    _ = d_scr^
    _ = st^
    _ = ctx^
    if status != ST_OK:
        raise Error("svc epilogue: a row raised (float division by zero or math domain error)")


def device_gather(
    src_addr: Int, n_src_rows: Int, n_src_cols: Int, rows_addr: Int, n_out_rows: Int,
    cols_addr: Int, n_out_cols: Int, out_addr: Int,
) raises:
    """`out = src[rows][:, cols]` (an absent index list is the identity)
    as a device gather of 4-byte cells."""
    var total = n_out_rows * n_out_cols
    if total == 0:
        return
    var ctx = _family_ctx()
    var d_src = _up_u32(ctx, src_addr, n_src_rows * n_src_cols)
    var d_rows = _up_i32(ctx, rows_addr, n_out_rows if rows_addr != 0 else 0)
    var d_cols = _up_i32(ctx, cols_addr, n_out_cols if cols_addr != 0 else 0)
    var d_out = ctx.enqueue_create_buffer[DType.uint32](total)
    var st = _status(ctx)
    ctx.enqueue_function[gather_kernel](
        d_src.unsafe_ptr(), Int64(n_src_rows), Int64(n_src_cols), d_rows.unsafe_ptr(),
        Int32(1 if rows_addr != 0 else 0), d_cols.unsafe_ptr(), Int32(1 if cols_addr != 0 else 0),
        Int64(n_out_cols), Int64(total), d_out.unsafe_ptr(), st.unsafe_ptr(),
        grid_dim=_grid(total), block_dim=FOLD_TPB,
    )
    ctx.enqueue_copy(dst_ptr=U32P(unsafe_from_address=out_addr), src_buf=d_out)
    var status = _read_status(ctx, st)
    _ = d_src^
    _ = d_rows^
    _ = d_cols^
    _ = d_out^
    _ = st^
    _ = ctx^
    if status != ST_OK:
        raise Error("svc gather: an index is outside the source matrix")


def device_select(
    codes_addr: Int, n: Int, ci: Int, cj: Int, c_addr: Int, idx_addr: Int,
    lab_addr: Int, c_out_addr: Int,
) raises -> Int:
    """The rows of class ci or cj, in row order: their indices (int32),
    labels (float32 1.0 for cj, 0.0 for ci) and, when `c_addr` is given,
    their float32 bounds. Returns the count."""
    if n == 0:
        return 0
    var ctx = _family_ctx()
    var nb = fold_blocks(n)
    var d_codes = _up_i32(ctx, codes_addr, n)
    var has_c = c_addr != 0 and c_out_addr != 0
    var d_c = _up_u32(ctx, c_addr if has_c else 0, n if has_c else 0)
    var d_part = ctx.enqueue_create_buffer[DType.int32](nb)
    var d_idx = ctx.enqueue_create_buffer[DType.int32](n)
    var d_lab = ctx.enqueue_create_buffer[DType.uint32](n)
    var d_cout = ctx.enqueue_create_buffer[DType.uint32](n if has_c else 1)
    var d_count = ctx.enqueue_create_buffer[DType.int32](1)
    ctx.enqueue_function[select_count_kernel](
        d_codes.unsafe_ptr(), Int32(n), Int32(ci), Int32(cj), d_part.unsafe_ptr(),
        grid_dim=nb, block_dim=FOLD_TPB,
    )
    ctx.enqueue_function[select_emit_kernel](
        d_codes.unsafe_ptr(), Int32(n), Int32(ci), Int32(cj), d_part.unsafe_ptr(), Int32(nb),
        d_c.unsafe_ptr(), Int32(1 if has_c else 0), d_idx.unsafe_ptr(), d_lab.unsafe_ptr(),
        d_cout.unsafe_ptr(), d_count.unsafe_ptr(),
        grid_dim=nb, block_dim=FOLD_TPB,
    )
    ctx.enqueue_copy(dst_ptr=I32P(unsafe_from_address=idx_addr), src_buf=d_idx)
    ctx.enqueue_copy(dst_ptr=U32P(unsafe_from_address=lab_addr), src_buf=d_lab)
    if has_c:
        ctx.enqueue_copy(dst_ptr=U32P(unsafe_from_address=c_out_addr), src_buf=d_cout)
    var count = _read_status(ctx, d_count)
    _ = d_codes^
    _ = d_c^
    _ = d_part^
    _ = d_idx^
    _ = d_lab^
    _ = d_cout^
    _ = d_count^
    _ = ctx^
    return count


def device_c_rows(
    sw_addr: Int, codes_addr: Int, cw_addr: Int, k: Int, c: UInt64, n: Int, out_addr: Int
) raises:
    """`_c_rows`'s per-row bounds on the device (float32 at `out_addr`)."""
    if n == 0:
        return
    var ctx = _family_ctx()
    var has_cw = cw_addr != 0
    var d_sw = _up_u64(ctx, sw_addr, n if sw_addr != 0 else 0)
    var d_codes = _up_i32(ctx, codes_addr if has_cw else 0, n if has_cw else 0)
    var d_cw = _up_u64(ctx, cw_addr, k if has_cw else 0)
    var d_out = ctx.enqueue_create_buffer[DType.uint32](n)
    var st = _status(ctx)
    ctx.enqueue_function[c_rows_kernel](
        d_sw.unsafe_ptr(), Int32(1 if sw_addr != 0 else 0), d_codes.unsafe_ptr(), d_cw.unsafe_ptr(),
        Int32(1 if has_cw else 0), Int32(k), c, Int32(n), d_out.unsafe_ptr(), st.unsafe_ptr(),
        grid_dim=_grid(n), block_dim=FOLD_TPB,
    )
    ctx.enqueue_copy(dst_ptr=U32P(unsafe_from_address=out_addr), src_buf=d_out)
    var status = _read_status(ctx, st)
    _ = d_sw^
    _ = d_codes^
    _ = d_cw^
    _ = d_out^
    _ = st^
    _ = ctx^
    if status != ST_OK:
        raise Error("svc C rows: a class code is outside class_weight")


# ------------------------------------------------------------ Python doors
def _ix(v: PythonObject) raises -> Int:
    var x = Int(py=v)
    if x < 0:
        raise Error("svc epilogue: negative size")
    return x


def _f64_bits(v: PythonObject) raises -> UInt64:
    return bitcast[DType.uint64](Float64(py=v))


def check_pairs(pairs_addr: Int, n_pairs: Int, k: Int) raises:
    """The pair codes (2 per pair, a host list of K(K-1)/2 entries) are in
    range and distinct."""
    var pi = I32P(unsafe_from_address=pairs_addr)
    for pr in range(n_pairs):
        var i = Int(pi[2 * pr])
        var j = Int(pi[2 * pr + 1])
        if i < 0 or j < 0 or i >= k or j >= k or i == j:
            raise Error("svc_pair_epilogue: invalid class pair")


def svc_pair_epilogue_device_binding(
    dec_addr: PythonObject, pairs_addr: PythonObject, ab_addr: PythonObject,
    out_addr: PythonObject, params: PythonObject,
) raises -> PythonObject:
    """params[0] is the mode. Modes 0..5 (params [mode, n_rows, n_pairs,
    n_classes, label1]): 0 ovo (float32 n x P), 1 ovr (float64 n x K), 2
    vote codes (int64 n), 3 probabilities (float64 n x K; ab = 2P float64
    (A, B) per pair), 4 their logs, 5 binary codes from the device's labels
    (int64 n); returns n.
    6 GATHER: dec=src float32, pairs=row indices int32 or 0, ab=column
      indices int32 or 0, out float32; params [6, n_src_rows, n_src_cols,
      n_out_rows, n_out_cols]; returns n_out_rows.
    7 SELECT: dec=class codes int32 (n), pairs=per-row bounds float32 or 0,
      ab=labels out float32 (n), out=indices int32 (n); params [7, n, ci, cj,
      bounds-out address or 0]; returns the count.
    8 C_ROWS: dec=sample weights float64 or 0, pairs=class codes int32 or 0,
      ab=class weights float64 (K) or 0, out float32 (n); params [8, n, K,
      C, 0]; returns n."""
    if len(params) != 5:
        raise Error("svc_pair_epilogue: params must contain 5 values")
    var mode = _ix(params[0])
    var dst = Int(py=out_addr)
    if dst == 0:
        raise Error("svc_pair_epilogue: null output")
    if mode == GLUE_GATHER:
        var nsr = _ix(params[1])
        var nsc = _ix(params[2])
        var nor = _ix(params[3])
        var noc = _ix(params[4])
        var ra = Int(py=pairs_addr)
        var ca = Int(py=ab_addr)
        if (ra == 0 and nor != nsr) or (ca == 0 and noc != nsc):
            raise Error("svc gather: an identity axis must keep its length")
        var sa = Int(py=dec_addr)
        with GILReleased(Python()):
            device_gather(sa, nsr, nsc, ra, nor, ca, noc, dst)
        return PythonObject(nor)
    if mode == GLUE_SELECT:
        var n = _ix(params[1])
        var ci = Int(py=params[2])
        var cj = Int(py=params[3])
        var cout = Int(py=params[4])
        var ca = Int(py=dec_addr)
        var cb = Int(py=pairs_addr)
        var la = Int(py=ab_addr)
        var count = 0
        with GILReleased(Python()):
            count = device_select(ca, n, ci, cj, cb, dst, la, cout)
        return PythonObject(count)
    if mode == GLUE_C_ROWS:
        var n = _ix(params[1])
        var k = _ix(params[2])
        var c = _f64_bits(params[3])
        var swa = Int(py=dec_addr)
        var coa = Int(py=pairs_addr)
        var cwa = Int(py=ab_addr)
        with GILReleased(Python()):
            device_c_rows(swa, coa, cwa, k, c, n, dst)
        return PythonObject(n)
    var n = _ix(params[1])
    var n_pairs = _ix(params[2])
    var k = _ix(params[3])
    var label1 = _f64_bits(params[4])
    if mode > EPI_BINARY_CODES:
        raise Error("svc_pair_epilogue: unknown mode")
    if n == 0:
        return PythonObject(0)
    if n_pairs < 1 or k < 2:
        raise Error("svc_pair_epilogue: needs a pair and two classes")
    var pa = Int(py=pairs_addr)
    if mode != EPI_BINARY_CODES:
        check_pairs(pa, n_pairs, k)
    var da = Int(py=dec_addr)
    var aa = Int(py=ab_addr) if mode == EPI_PROBA or mode == EPI_LOG_PROBA else 0
    with GILReleased(Python()):
        device_pair_epilogue(mode, da, n, n_pairs, k, pa, aa, label1, dst)
    return PythonObject(n)


def svc_platt_train_device_binding(
    dec_addr: PythonObject, labels_addr: PythonObject, out_addr: PythonObject, n: PythonObject
) raises -> PythonObject:
    """libsvm's `sigmoid_train` over n float64 decision values and +1/-1
    float64 labels, the row sums on the device; writes (A, B) as two float64
    at out_addr. Returns n."""
    var count = _ix(n)
    var op = U64P(unsafe_from_address=Int(py=out_addr))
    var da = Int(py=dec_addr)
    var la = Int(py=labels_addr)
    if count == 0:
        raise Error("svc_platt_train: no decision values")
    with GILReleased(Python()):
        var ctx = _family_ctx()
        var src = DevicePlatt(ctx, da, la, count)
        var r = platt_solve(src, count)
        op[0] = r[0]
        op[1] = r[1]
        _ = src^
        _ = ctx^
    return PythonObject(count)


def svc_splitmix_perm_device_binding(
    out_addr: PythonObject, n: PythonObject, seed_lo: PythonObject, seed_hi: PythonObject
) raises -> PythonObject:
    """The probability shuffle of `[0, n)` into n int32 at out_addr, the
    64-bit seed handed in as two 32-bit halves. Returns n."""
    var count = _ix(n)
    if count == 0:
        return PythonObject(0)
    if count > 2147483647:
        raise Error("svc_splitmix_perm: n exceeds int32")
    var lo = UInt64(_ix(seed_lo)) & UInt64(0xFFFFFFFF)
    var hi = UInt64(_ix(seed_hi)) & UInt64(0xFFFFFFFF)
    var key = shuffle_seed32((hi << 32) | lo)
    var dst = Int(py=out_addr)
    with GILReleased(Python()):
        var ctx = _family_ctx()
        var d = ctx.enqueue_create_buffer[DType.int32](count)
        ctx.enqueue_function[perm_kernel](
            d.unsafe_ptr(), Int32(count), key, grid_dim=_grid(count), block_dim=FOLD_TPB,
        )
        ctx.enqueue_copy(dst_ptr=I32P(unsafe_from_address=dst), src_buf=d)
        ctx.synchronize()
        _ = d^
        _ = ctx^
    return PythonObject(count)


def svc_portable_math_device_binding(
    in_addr: PythonObject, out_addr: PythonObject, n: PythonObject, which: PythonObject
) raises -> PythonObject:
    """THE TWIN'S OWN CHECK DOOR: `sp_exp` (which 0) or `sf64_log` (which 1)
    over n float64 on the device, so a job can hold them to
    `_portable_math.exp/log` bit for bit. Returns n."""
    var count = _ix(n)
    if count == 0:
        return PythonObject(0)
    var w = Int(py=which)
    var ia = Int(py=in_addr)
    var oa = Int(py=out_addr)
    with GILReleased(Python()):
        var ctx = _family_ctx()
        var d_in = _up_u64(ctx, ia, count)
        var d_out = ctx.enqueue_create_buffer[DType.uint64](count)
        ctx.enqueue_function[pmath_kernel](
            d_out.unsafe_ptr(), d_in.unsafe_ptr(), Int32(count), Int32(w),
            grid_dim=_grid(count), block_dim=FOLD_TPB,
        )
        ctx.enqueue_copy(dst_ptr=U64P(unsafe_from_address=oa), src_buf=d_out)
        ctx.synchronize()
        _ = d_in^
        _ = d_out^
        _ = ctx^
    return PythonObject(count)
