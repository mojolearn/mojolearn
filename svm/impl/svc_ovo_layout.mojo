# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""SVC's one-vs-one attribute layout and `coef_` ON THE DEVICE
(lane/apple-fast-py2mojo-linear).

`python/mojolearn/_svm_impl.py::SVC._set_ovo` built scikit-learn's
multiclass attributes in Python: a dict per class of support rows, a sort,
the support-vector matrix row by row and the (K - 1, n_SV) dual matrix;
`_dual_times_sv` (`coef_` for kernel='linear') was a Python float64 loop
over n_SV * d. Both run here now; the host column
(`svm/host/svc_ovo_host.mojo`) computes the same words with plain loops.

LAYOUT (`svc_ovo_layout`), the old rules exactly:
  * a support vector's class is its pair side: class i when its dual is
    negative (`d < 0.0`), class j otherwise (`mark_kernel`; every pair names
    a row's own class, so concurrent writes write the same value);
  * `support_` is class-major, ascending row index within a class: per
    class, block counts (`count_kernel`), one exclusive scan over the
    class-major counts (`chunk_total_kernel`, `scan_kernel`), then each
    block places its rows of that class (`emit_kernel`); integer counts,
    exact in any order;
  * `dual_coef_[j - 1 if side == i else i][col] = -d` (`dual_kernel`), every
    other cell 0.0.
`support_vectors_` is then the device gather of X at `support_`
(`svc_pair_epilogue` GATHER), the same float32 bytes the dict held.

COEF (`svc_dual_gemv`): `acc[j] += a * sv[a][j]` in binary64, ascending a,
from +0.0, rounded once to float32: one thread per feature, the chain in
`checks/soft_f64.mojo` words (no float64 unit on Apple), each product exact
(two float32 significands fit binary64), so the SAME bits as the Python loop.
"""

from std.gpu import block_idx, thread_idx
from std.memory import stack_allocation
from std.python import Python, PythonObject
from std.python._cpython import GILReleased
from max.gpu.host import DeviceBuffer, DeviceContext
from max.gpu.memory import AddressSpace
from max.gpu.sync import barrier

from checks.soft_f64 import SF64_ZERO, sf64_add, sf64_from_f32, sf64_mul, sf64_to_f32

comptime OVO_TPB = 256

comptime _I32P = MutPointer[Int32, MutAnyOrigin]
comptime _F32P = MutPointer[Float32, MutAnyOrigin]


@always_inline
def ovo_side(d: Float32, ci: Int, cj: Int) -> Int:
    """The class of a pair's support vector: i when its dual is negative."""
    return ci if d < Float32(0.0) else cj


@always_inline
def ovo_dual_row(side: Int, ci: Int, cj: Int) -> Int:
    """The dual_coef_ row of pair (i, j)'s coefficient for a vector of
    class `side`: j - 1 for class i, i for class j."""
    return cj - 1 if side == ci else ci


@always_inline
def dual_gemv_column(dual: _F32P, sv: _F32P, n_sv: Int, d: Int, j: Int) -> Float32:
    """`sum_a dual[a] * sv[a, j]` in binary64, ascending a from +0.0,
    rounded once to float32."""
    var acc = SF64_ZERO
    for a in range(n_sv):
        acc = sf64_add(acc, sf64_mul(sf64_from_f32(dual[a]), sf64_from_f32(sv[a * d + j])))
    return sf64_to_f32(acc)


# ---------------------------------------------------------------- kernels
def mark_kernel(sup: _I32P, dual: _F32P, m: Int32, ci: Int32, cj: Int32, n_bound: Int32,
                cls: _I32P, status: _I32P):
    var e = Int(block_idx.x) * OVO_TPB + Int(thread_idx.x)
    if e < Int(m):
        var r = sup[e]
        if r < 0 or r >= n_bound:
            status[0] = Int32(1)
        else:
            cls[Int(r)] = Int32(ovo_side(dual[e], Int(ci), Int(cj)))


def count_kernel(cls: _I32P, n: Int32, k: Int32, nb: Int32, part: _I32P):
    """part[c * nb + blk]: block blk's count of class-c rows."""
    var tid = Int(thread_idx.x)
    var blk = Int(block_idx.x)
    var i = blk * OVO_TPB + tid
    var mine = Int32(-1)
    if i < Int(n):
        mine = cls[i]
    var s = stack_allocation[OVO_TPB, Scalar[DType.int32], address_space = AddressSpace.SHARED]()
    for c in range(Int(k)):
        s[tid] = Int32(1) if mine == Int32(c) else Int32(0)
        barrier()
        var step = OVO_TPB // 2
        while step > 0:
            if tid < step:
                s[tid] = s[tid] + s[tid + step]
            barrier()
            step //= 2
        if tid == 0:
            part[c * Int(nb) + blk] = s[0]
        barrier()


def chunk_total_kernel(part: _I32P, total: Int32, ctot: _I32P):
    var tid = Int(thread_idx.x)
    var blk = Int(block_idx.x)
    var i = blk * OVO_TPB + tid
    var s = stack_allocation[OVO_TPB, Scalar[DType.int32], address_space = AddressSpace.SHARED]()
    s[tid] = part[i] if i < Int(total) else Int32(0)
    barrier()
    var step = OVO_TPB // 2
    while step > 0:
        if tid < step:
            s[tid] = s[tid] + s[tid + step]
        barrier()
        step //= 2
    if tid == 0:
        ctot[blk] = s[0]


def scan_kernel(part: _I32P, total: Int32, ctot: _I32P, excl: _I32P):
    """excl[t + 1] = sum part[0 .. t] (excl[0] = 0) over `total` counts."""
    var tid = Int(thread_idx.x)
    var blk = Int(block_idx.x)
    var s = stack_allocation[OVO_TPB, Scalar[DType.int32], address_space = AddressSpace.SHARED]()
    var acc = Int32(0)
    for t in range(tid, blk, OVO_TPB):
        acc += ctot[t]
    s[tid] = acc
    barrier()
    var step = OVO_TPB // 2
    while step > 0:
        if tid < step:
            s[tid] = s[tid] + s[tid + step]
        barrier()
        step //= 2
    var off = s[0]
    barrier()
    var i = blk * OVO_TPB + tid
    s[tid] = part[i] if i < Int(total) else Int32(0)
    barrier()
    var d = 1
    while d < OVO_TPB:
        var v = s[tid]
        if tid >= d:
            v += s[tid - d]
        barrier()
        s[tid] = v
        barrier()
        d *= 2
    if i < Int(total):
        excl[i + 1] = off + s[tid]
    if blk == 0 and tid == 0:
        excl[0] = Int32(0)


def emit_kernel(cls: _I32P, n: Int32, k: Int32, nb: Int32, excl: _I32P, support: _I32P, colof: _I32P):
    """Block blk places its class-c rows, ascending, at excl[c * nb + blk]."""
    var tid = Int(thread_idx.x)
    var blk = Int(block_idx.x)
    var i = blk * OVO_TPB + tid
    var mine = Int32(-1)
    if i < Int(n):
        mine = cls[i]
    var s = stack_allocation[OVO_TPB, Scalar[DType.int32], address_space = AddressSpace.SHARED]()
    for c in range(Int(k)):
        var off = excl[c * Int(nb) + blk]
        var f = Int32(1) if mine == Int32(c) else Int32(0)
        s[tid] = f
        barrier()
        var w = 1
        while w < OVO_TPB:
            var v = s[tid]
            if tid >= w:
                v += s[tid - w]
            barrier()
            s[tid] = v
            barrier()
            w *= 2
        if f != 0:
            var pos = Int(off + s[tid] - 1)
            support[pos] = Int32(i)
            colof[i] = Int32(pos)
        barrier()


def nsup_kernel(excl: _I32P, k: Int32, nb: Int32, nsup: _I32P):
    var c = Int(block_idx.x) * OVO_TPB + Int(thread_idx.x)
    if c < Int(k):
        nsup[c] = excl[(c + 1) * Int(nb)] - excl[c * Int(nb)]


def dual_kernel(sup: _I32P, dual: _F32P, m: Int32, ci: Int32, cj: Int32, colof: _I32P,
                n_sv: Int32, dst: _F32P):
    var e = Int(block_idx.x) * OVO_TPB + Int(thread_idx.x)
    if e < Int(m):
        var d = dual[e]
        var side = ovo_side(d, Int(ci), Int(cj))
        var row = ovo_dual_row(side, Int(ci), Int(cj))
        dst[row * Int(n_sv) + Int(colof[Int(sup[e])])] = -d


def gemv_kernel(dual: _F32P, sv: _F32P, n_sv: Int32, d: Int32, dst: _F32P):
    var j = Int(block_idx.x) * OVO_TPB + Int(thread_idx.x)
    if j < Int(d):
        dst[j] = dual_gemv_column(dual, sv, Int(n_sv), Int(d), j)


# ---------------------------------------------------------------- drivers
@always_inline
def _blocks(count: Int) -> Int:
    return max(1, (count + OVO_TPB - 1) // OVO_TPB)


def _read_i32(ctx: DeviceContext, buf: DeviceBuffer[DType.int32], at: Int) raises -> Int:
    var h = ctx.enqueue_create_host_buffer[DType.int32](1)
    ctx.enqueue_copy(dst_ptr=h.unsafe_ptr(), src_buf=buf.create_sub_buffer[DType.int32](at, 1))
    ctx.synchronize()
    var v = Int(h.unsafe_ptr()[0])
    _ = h^
    return v


@fieldwise_init
struct OvoPairs(Copyable, Movable):
    """The per-pair inputs: support addresses (int32), dual addresses
    (float32), and (i, j, n_support) per pair."""
    var sup: List[Int]
    var dual: List[Int]
    var ci: List[Int]
    var cj: List[Int]
    var m: List[Int]


def ovo_pairs_from_python(sup_addrs: PythonObject, dual_addrs: PythonObject, meta: PythonObject) raises -> OvoPairs:
    var p = Int(py=meta[2])
    if len(sup_addrs) != p or len(dual_addrs) != p or len(meta) != 3 + 3 * p:
        raise Error("svc_ovo_layout: one support and one dual address per pair, meta [k, n, P, (i, j, m) * P]")
    var out = OvoPairs(List[Int](), List[Int](), List[Int](), List[Int](), List[Int]())
    for q in range(p):
        out.sup.append(Int(py=sup_addrs[q]))
        out.dual.append(Int(py=dual_addrs[q]))
        out.ci.append(Int(py=meta[3 + 3 * q]))
        out.cj.append(Int(py=meta[4 + 3 * q]))
        out.m.append(Int(py=meta[5 + 3 * q]))
        if out.m[q] < 0 or (out.m[q] > 0 and (out.sup[q] == 0 or out.dual[q] == 0)):
            raise Error("svc_ovo_layout: a pair's support or dual buffer is null")
    return out^


def ovo_layout_device(
    ctx: DeviceContext, pairs: OvoPairs, k: Int, n_bound: Int, cap: Int,
    support_addr: Int, nsup_addr: Int, dual_addr: Int,
) raises -> Int:
    """Writes support_ (n_sv int32), n_support_ (k int32) and dual_coef_
    ((k - 1) * n_sv float32, row stride n_sv) at the three addresses;
    returns n_sv, or -1 when a support index is outside [0, n_bound), -2
    when n_sv exceeds `cap`."""
    var n = max(1, n_bound)
    var nb = _blocks(n)
    var total = k * nb
    var nc = _blocks(total)
    var d_cls = ctx.enqueue_create_buffer[DType.int32](n)
    ctx.enqueue_memset(d_cls, Int32(-1))
    var d_st = ctx.enqueue_create_buffer[DType.int32](1)
    ctx.enqueue_memset(d_st, Int32(0))
    var offs = List[Int]()
    var e_total = 0
    for q in range(len(pairs.m)):
        offs.append(e_total)
        e_total += pairs.m[q]
    var d_sup = ctx.enqueue_create_buffer[DType.int32](max(1, e_total))
    var d_dl = ctx.enqueue_create_buffer[DType.float32](max(1, e_total))
    for q in range(len(pairs.m)):
        if pairs.m[q] > 0:
            ctx.enqueue_copy(dst_buf=d_sup.create_sub_buffer[DType.int32](offs[q], pairs.m[q]),
                             src_ptr=_I32P(unsafe_from_address=pairs.sup[q]))
            ctx.enqueue_copy(dst_buf=d_dl.create_sub_buffer[DType.float32](offs[q], pairs.m[q]),
                             src_ptr=_F32P(unsafe_from_address=pairs.dual[q]))
            ctx.enqueue_function[mark_kernel](
                d_sup.unsafe_ptr() + offs[q], d_dl.unsafe_ptr() + offs[q], Int32(pairs.m[q]),
                Int32(pairs.ci[q]), Int32(pairs.cj[q]), Int32(n_bound), d_cls.unsafe_ptr(),
                d_st.unsafe_ptr(), grid_dim=_blocks(pairs.m[q]), block_dim=OVO_TPB,
            )
    var d_part = ctx.enqueue_create_buffer[DType.int32](total)
    var d_ctot = ctx.enqueue_create_buffer[DType.int32](nc)
    var d_excl = ctx.enqueue_create_buffer[DType.int32](total + 1)
    var d_support = ctx.enqueue_create_buffer[DType.int32](n)
    var d_colof = ctx.enqueue_create_buffer[DType.int32](n)
    var d_nsup = ctx.enqueue_create_buffer[DType.int32](max(1, k))
    ctx.enqueue_function[count_kernel](
        d_cls.unsafe_ptr(), Int32(n_bound), Int32(k), Int32(nb), d_part.unsafe_ptr(),
        grid_dim=nb, block_dim=OVO_TPB,
    )
    ctx.enqueue_function[chunk_total_kernel](
        d_part.unsafe_ptr(), Int32(total), d_ctot.unsafe_ptr(), grid_dim=nc, block_dim=OVO_TPB,
    )
    ctx.enqueue_function[scan_kernel](
        d_part.unsafe_ptr(), Int32(total), d_ctot.unsafe_ptr(), d_excl.unsafe_ptr(),
        grid_dim=nc, block_dim=OVO_TPB,
    )
    ctx.enqueue_function[emit_kernel](
        d_cls.unsafe_ptr(), Int32(n_bound), Int32(k), Int32(nb), d_excl.unsafe_ptr(),
        d_support.unsafe_ptr(), d_colof.unsafe_ptr(), grid_dim=nb, block_dim=OVO_TPB,
    )
    ctx.enqueue_function[nsup_kernel](
        d_excl.unsafe_ptr(), Int32(k), Int32(nb), d_nsup.unsafe_ptr(),
        grid_dim=_blocks(k), block_dim=OVO_TPB,
    )
    if _read_i32(ctx, d_st, 0) != 0:
        return -1
    var n_sv = _read_i32(ctx, d_excl, total)
    if n_sv > cap:
        return -2
    ctx.enqueue_copy(dst_ptr=_I32P(unsafe_from_address=nsup_addr),
                     src_buf=d_nsup.create_sub_buffer[DType.int32](0, k))
    if n_sv > 0:
        var cells = (k - 1) * n_sv
        var d_dual = ctx.enqueue_create_buffer[DType.float32](cells)
        ctx.enqueue_memset(d_dual, Float32(0.0))
        for q in range(len(pairs.m)):
            if pairs.m[q] > 0:
                ctx.enqueue_function[dual_kernel](
                    d_sup.unsafe_ptr() + offs[q], d_dl.unsafe_ptr() + offs[q], Int32(pairs.m[q]),
                    Int32(pairs.ci[q]), Int32(pairs.cj[q]), d_colof.unsafe_ptr(), Int32(n_sv),
                    d_dual.unsafe_ptr(), grid_dim=_blocks(pairs.m[q]), block_dim=OVO_TPB,
                )
        ctx.enqueue_copy(dst_ptr=_I32P(unsafe_from_address=support_addr),
                         src_buf=d_support.create_sub_buffer[DType.int32](0, n_sv))
        ctx.enqueue_copy(dst_ptr=_F32P(unsafe_from_address=dual_addr), src_buf=d_dual)
        ctx.synchronize()
        _ = d_dual^
    else:
        ctx.synchronize()
    _ = d_sup^
    _ = d_dl^
    _ = d_cls^
    _ = d_st^
    _ = d_part^
    _ = d_ctot^
    _ = d_excl^
    _ = d_support^
    _ = d_colof^
    _ = d_nsup^
    return n_sv


def ovo_check_result(n_sv: Int) raises:
    if n_sv == -1:
        raise Error("svc_ovo_layout: a support index is outside the training rows")
    if n_sv == -2:
        raise Error("svc_ovo_layout: more support vectors than the output holds")


def svc_ovo_layout_device_binding(
    ctx: DeviceContext, sup_addrs: PythonObject, dual_addrs: PythonObject, meta: PythonObject,
    out_addrs: PythonObject,
) raises -> PythonObject:
    """`svc_ovo_layout(sup_addrs, dual_addrs, meta, out_addrs)`: meta
    [k, n_bound, P, (i, j, n_support) per pair]; out_addrs [support (int32,
    cap), n_support (int32, k), dual_coef (float32, (k - 1) * cap), cap].
    Returns n_sv (dual_coef_ written with row stride n_sv)."""
    var pairs = ovo_pairs_from_python(sup_addrs, dual_addrs, meta)
    var k = Int(py=meta[0])
    var n_bound = Int(py=meta[1])
    if len(out_addrs) != 4:
        raise Error("svc_ovo_layout: out_addrs [support, n_support, dual_coef, cap]")
    var sa = Int(py=out_addrs[0])
    var na = Int(py=out_addrs[1])
    var da = Int(py=out_addrs[2])
    var cap = Int(py=out_addrs[3])
    if k < 2 or n_bound < 0 or sa == 0 or na == 0 or da == 0:
        raise Error("svc_ovo_layout: needs two classes and three output buffers")
    var n_sv = 0
    with GILReleased(Python()):
        n_sv = ovo_layout_device(ctx, pairs, k, n_bound, cap, sa, na, da)
    ovo_check_result(n_sv)
    return PythonObject(n_sv)


def svc_dual_gemv_device_binding(
    ctx: DeviceContext, dual_addr: PythonObject, sv_addr: PythonObject, dims: PythonObject,
    out_addr: PythonObject,
) raises -> PythonObject:
    """`svc_dual_gemv(dual, sv, [n_sv, d], out)`: coef_ = dual @ sv (one
    row), float32 out of d. Returns d."""
    var n_sv = Int(py=dims[0])
    var d = Int(py=dims[1])
    var oa = Int(py=out_addr)
    if n_sv < 0 or d < 0 or oa == 0:
        raise Error("svc_dual_gemv: bad dims or null output")
    if d == 0:
        return PythonObject(0)
    var dl = Int(py=dual_addr)
    var sl = Int(py=sv_addr)
    with GILReleased(Python()):
        var d_dual = ctx.enqueue_create_buffer[DType.float32](max(1, n_sv))
        var d_sv = ctx.enqueue_create_buffer[DType.float32](max(1, n_sv * d))
        if n_sv > 0:
            ctx.enqueue_copy(dst_buf=d_dual.create_sub_buffer[DType.float32](0, n_sv),
                             src_ptr=_F32P(unsafe_from_address=dl))
            ctx.enqueue_copy(dst_buf=d_sv.create_sub_buffer[DType.float32](0, n_sv * d),
                             src_ptr=_F32P(unsafe_from_address=sl))
        var d_out = ctx.enqueue_create_buffer[DType.float32](d)
        ctx.enqueue_function[gemv_kernel](
            d_dual.unsafe_ptr(), d_sv.unsafe_ptr(), Int32(n_sv), Int32(d), d_out.unsafe_ptr(),
            grid_dim=_blocks(d), block_dim=OVO_TPB,
        )
        ctx.enqueue_copy(dst_ptr=_F32P(unsafe_from_address=oa), src_buf=d_out)
        ctx.synchronize()
        _ = d_dual^
        _ = d_sv^
        _ = d_out^
    return PythonObject(d)
