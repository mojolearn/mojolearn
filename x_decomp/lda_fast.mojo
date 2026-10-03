# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""LatentDirichletAllocation's E-step and sufficient statistics in ONE launch
(lane apple-fast-nb, 2026-10-02). FAST + Apple ONLY, behind
-D MOJOLEARN_LDA_FUSED_SS: the IDENTICAL binding, the other vendors and a
FAST build without the define do not export `x_decomp_dev_lda_estep_ss`, and
`_expansion_decomp.LatentDirichletAllocation._e_step` keeps main's chain
(`lda_block_kernel`, then norm_phi = Et exp_dir (n x v), X / norm_phi and
the (k x n)(n x v) gemm).

Main's `lda_block_kernel` (x_decomp/device.mojo) is one 128-thread block per
document with about 20 KB of threadgroup memory: at the board's taxi-zones
(about 300k documents of a dozen nonzero dropoff zones each) one or two
blocks fit a core, the topic fold runs on 16 of the 128 threads, and the
sufficient statistics then take two dense n x v passes and a gemm whose
inner dimension is n. Here a block of LFS_TPB threads carries LFS_DPB
documents, LFS_TPD lanes each (about 12 KB of threadgroup memory), and the
block is persistent: it walks document groups b, b + G, b + 2G, ... and
accumulates its own k x v partial of the statistics in device memory (no
atomics: the block owns it, groups ascending), then `lda_ss_fold_kernel`
sums the G partials ascending and applies exp_dir. The per-document
sequence is `lda_block_kernel`'s (words compacted ascending, norm_phi's
topic fold ascending, each topic's word fold ascending, total, digamma and
change in topic order), so Dt and Et are the same bits; the statistics take
a new (fixed, run-to-run identical) fold order under FAST.

Caps: k <= LFS_K_CAP topics, v <= LFS_V_CAP words (the dense per-document
norm_phi row lives in threadgroup memory). Past them the entry returns 0 and
Python keeps main's chain.
"""
from std.sys.compile import is_defined
from std.sys.info import has_apple_gpu_accelerator
from std.gpu import block_dim, block_idx, thread_idx
from std.memory import stack_allocation
from std.python import PythonObject
from max.gpu.memory import AddressSpace
from max.gpu.host import DeviceContext
from max.gpu.sync import barrier
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_FAST, ftz, identical_mul_add
from x_decomp.cells import F32Ptr, add, mul, sub, div0, exp_c, digamma
from x_decomp.device import TPB, _blocks, xd_ctx
from x_decomp.resident import _id, _n, _ptr, pool_alloc, pool_free

#: The switch: FAST, Apple, and the define (default OFF).
comptime LDA_FUSED_SS = (
    GLOBAL_NUMERIC_MODE == NUMERIC_FAST and has_apple_gpu_accelerator() and is_defined["MOJOLEARN_LDA_FUSED_SS"]()
)
#: Lanes per document (one SIMD group).
comptime LFS_TPD = 32
#: Documents per block.
comptime LFS_DPB = 4
comptime LFS_TPB = LFS_TPD * LFS_DPB
#: Largest vocabulary (and so nonzero count) a document keeps in threadgroup memory.
comptime LFS_V_CAP = 320
#: Largest n_components.
comptime LFS_K_CAP = 32
#: Persistent blocks (each owns one k x v partial of the statistics).
comptime LFS_BLOCKS = 2048


def lda_fused_kernel(
    x: F32Ptr, ew: F32Ptr, d: F32Ptr, e: F32Ptr, part: F32Ptr, n: Int32, k: Int32, v: Int32,
    prior: Float32, max_iter: Int32, tol: Float32, groups: Int32,
):
    """Block b: documents of groups b, b + groups, ... (LFS_DPB per group,
    LFS_TPD lanes each): `lda_doc_row`'s iterations, Dt and Et written back,
    and part[b * k * v + t * v + w] += Et_final[i, t] * x[i, w] / (norm_phi[i, w]
    + eps) over the block's documents ascending (zero words add exactly 0)."""
    var b = Int(block_idx.x)
    var tid = Int(thread_idx.x)
    var dp = tid // LFS_TPD
    var lane = tid - dp * LFS_TPD
    var kk = Int(k)
    var vv = Int(v)
    var nn = Int(n)
    var kv = kk * vv
    var eps = Float32(2.220446049250313e-16)
    var idx = stack_allocation[LFS_DPB * LFS_V_CAP, Scalar[DType.int32], address_space = AddressSpace.SHARED]()
    var sw = stack_allocation[LFS_DPB * LFS_V_CAP, Scalar[DType.float32], address_space = AddressSpace.SHARED]()
    var ds = stack_allocation[LFS_DPB * LFS_K_CAP, Scalar[DType.float32], address_space = AddressSpace.SHARED]()
    var es = stack_allocation[LFS_DPB * LFS_K_CAP, Scalar[DType.float32], address_space = AddressSpace.SHARED]()
    var dif = stack_allocation[LFS_DPB * LFS_K_CAP, Scalar[DType.float32], address_space = AddressSpace.SHARED]()
    var cnt = stack_allocation[LFS_DPB * (LFS_TPD + 1), Scalar[DType.int32], address_space = AddressSpace.SHARED]()
    var misc = stack_allocation[LFS_DPB * 2, Scalar[DType.float32], address_space = AddressSpace.SHARED]()
    var done = stack_allocation[LFS_DPB, Scalar[DType.int32], address_space = AddressSpace.SHARED]()
    var pb = b * kv
    var q0 = tid
    while q0 < kv:
        part.unsafe_store(pb + q0, Float32(0))
        q0 += LFS_TPB
    var io = dp * LFS_V_CAP
    var ko = dp * LFS_K_CAP
    var co = dp * (LFS_TPD + 1)
    var seg = (vv + LFS_TPD - 1) // LFS_TPD
    var lo = lane * seg
    var hi = lo + seg
    if hi > vv:
        hi = vv
    var ngroups = (nn + LFS_DPB - 1) // LFS_DPB
    var g = b
    while g < ngroups:
        var i = g * LFS_DPB + dp
        var live = i < nn
        var base = i * kk
        var xb = i * vv
        if live:
            var t0 = lane
            while t0 < kk:
                ds[ko + t0] = d.unsafe_load(base + t0)
                es[ko + t0] = e.unsafe_load(base + t0)
                t0 += LFS_TPD
            var w0 = lane
            while w0 < vv:
                sw[io + w0] = Float32(0)
                w0 += LFS_TPD
            var mine = 0
            for w in range(lo, hi):
                if ftz(x.unsafe_load(xb + w)) != Float32(0):
                    mine += 1
            cnt[co + lane] = Int32(mine)
        if lane == 0:
            done[dp] = Int32(0) if live else Int32(1)
        barrier()
        if live and lane == 0:
            var run = 0
            for t in range(LFS_TPD):
                var cn = Int(cnt[co + t])
                cnt[co + t] = Int32(run)
                run += cn
            cnt[co + LFS_TPD] = Int32(run)
        barrier()
        var nnz = 0
        if live:
            nnz = Int(cnt[co + LFS_TPD])
            var at = Int(cnt[co + lane])
            for w in range(lo, hi):
                if ftz(x.unsafe_load(xb + w)) != Float32(0):
                    idx[io + at] = Int32(w)
                    at += 1
        barrier()
        for _ in range(Int(max_iter)):
            var any_live = False
            for s in range(LFS_DPB):
                if done[s] == Int32(0):
                    any_live = True
            if not any_live:
                break
            var active = live and done[dp] == Int32(0)
            if active:
                # norm_phi per nonzero word (topics ascending), x_w / (norm_phi_w + eps)
                var j = lane
                while j < nnz:
                    var w = Int(idx[io + j])
                    var xw = ftz(x.unsafe_load(xb + w))
                    var acc = Float32(0)
                    for t in range(kk):
                        acc = ftz(identical_mul_add(ftz(es[ko + t]), ftz(ew.unsafe_load(t * vv + w)), acc))
                    sw[io + w] = div0(xw, add(acc, eps))
                    j += LFS_TPD
            barrier()
            if active:
                # each topic's fold over the words ascending
                var t = lane
                while t < kk:
                    var acc = Float32(0)
                    var eb = t * vv
                    for j in range(nnz):
                        var w = Int(idx[io + j])
                        acc = ftz(identical_mul_add(ftz(sw[io + w]), ftz(ew.unsafe_load(eb + w)), acc))
                    var dt = add(mul(es[ko + t], acc), prior)
                    var old = ds[ko + t]
                    ds[ko + t] = dt
                    dif[ko + t] = abs(sub(old, dt))
                    t += LFS_TPD
            barrier()
            if active and lane == 0:
                var total = Float32(0)
                for q in range(kk):
                    total = add(total, ds[ko + q])
                misc[dp * 2] = digamma(total)
                var change = Float32(0)
                for q in range(kk):
                    change = add(change, dif[ko + q])
                misc[dp * 2 + 1] = Float32(1) if div0(change, Float32(kk)) < tol else Float32(0)
            barrier()
            if active:
                var psi_total = misc[dp * 2]
                var t = lane
                while t < kk:
                    es[ko + t] = exp_c(sub(digamma(ds[ko + t]), psi_total))
                    t += LFS_TPD
                if lane == 0 and misc[dp * 2 + 1] != Float32(0):
                    done[dp] = Int32(1)
            barrier()
        if live:
            var t0 = lane
            while t0 < kk:
                d.unsafe_store(base + t0, ds[ko + t0])
                e.unsafe_store(base + t0, es[ko + t0])
                t0 += LFS_TPD
            # norm_phi with the final Et (the chain's `mm(Et, exp_dir)`), x_w / (norm_phi_w + eps)
            var j = lane
            while j < nnz:
                var w = Int(idx[io + j])
                var xw = ftz(x.unsafe_load(xb + w))
                var acc = Float32(0)
                for t in range(kk):
                    acc = ftz(identical_mul_add(ftz(es[ko + t]), ftz(ew.unsafe_load(t * vv + w)), acc))
                sw[io + w] = div0(xw, add(acc, eps))
                j += LFS_TPD
        barrier()
        # the block's partial: every (topic, word) cell over the group's documents ascending
        var q = tid
        while q < kv:
            var t = q // vv
            var w = q - t * vv
            var acc = part.unsafe_load(pb + q)
            for s in range(LFS_DPB):
                if g * LFS_DPB + s < nn:
                    acc = add(acc, mul(es[s * LFS_K_CAP + t], sw[s * LFS_V_CAP + w]))
            part.unsafe_store(pb + q, acc)
            q += LFS_TPB
        barrier()
        g += Int(groups)


def lda_ss_fold_kernel(part: F32Ptr, ew: F32Ptr, ss: F32Ptr, kv: Int32, groups: Int32):
    """ss[q] = exp_dir[q] * sum_b part[b * kv + q], b ascending from zero
    (the chain's `mul(mm(Et, R, ta=True), exp_dir)`)."""
    var q = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var n = Int(kv)
    if q >= n:
        return
    var s = Float32(0)
    for b in range(Int(groups)):
        s = add(s, part.unsafe_load(b * n + q))
    ss.unsafe_store(q, mul(s, ew.unsafe_load(q)))


def launch_lda_fused_ss(
    ctx: DeviceContext, x: F32Ptr, ew: F32Ptr, d: F32Ptr, e: F32Ptr, part: F32Ptr, ss: F32Ptr, n: Int, k: Int,
    v: Int, prior: Float32, max_iter: Int, tol: Float32, groups: Int,
) raises:
    """The two launches: part holds groups * k * v floats of scratch."""
    ctx.enqueue_function[lda_fused_kernel](
        x, ew, d, e, part, Int32(n), Int32(k), Int32(v), prior, Int32(max_iter), tol, Int32(groups),
        grid_dim=groups, block_dim=LFS_TPB,
    )
    ctx.enqueue_function[lda_ss_fold_kernel](
        part, ew, ss, Int32(k * v), Int32(groups), grid_dim=_blocks(k * v), block_dim=TPB,
    )


def dev_lda_estep_ss_py(
    x: PythonObject, ew: PythonObject, d: PythonObject, e: PythonObject, ss: PythonObject, p: PythonObject,
    f: PythonObject,
) raises -> PythonObject:
    """`x_decomp_dev_lda_estep_ss` on device matrices: d and e (n x k)
    updated in place as `x_decomp_dev_lda_rows` does, and ss (k x v) = the
    E-step's sufficient statistics. p = [n, k, v, max_iter], f = [prior,
    tol]. Returns 1, or 0 when the shape is past the kernel's caps (the
    caller then runs main's chain)."""
    var n = _n(p, 0)
    var k = _n(p, 1)
    var v = _n(p, 2)
    var max_iter = _n(p, 3)
    if n < 1 or k < 1 or v < 1 or k > LFS_K_CAP or v > LFS_V_CAP:
        return PythonObject(0)
    var groups = min(LFS_BLOCKS, (n + LFS_DPB - 1) // LFS_DPB)
    var kv = k * v
    if n * v > 2147483647 or groups * kv > 2147483647:
        return PythonObject(0)
    var prior = Float32(Float64(py=f[0]))
    var tol = Float32(Float64(py=f[1]))
    var px = _ptr(_id(x), n * v)
    var pw = _ptr(_id(ew), kv)
    var pd = _ptr(_id(d), n * k)
    var pe = _ptr(_id(e), n * k)
    var ps = _ptr(_id(ss), kv)
    var sid = pool_alloc(groups * kv)
    launch_lda_fused_ss(xd_ctx(), px, pw, pd, pe, _ptr(sid, groups * kv), ps, n, k, v, prior, max_iter, tol, groups)
    pool_free(sid)
    return PythonObject(1)
