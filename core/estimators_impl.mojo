# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Typed device work of the `_mojolearn_estimators` binding.

The Python door stays in `bindings/_mojolearn_estimators.mojo`: it unpacks
PythonObject arguments, takes the binding's process context, calls one
function here and returns. The kernels, their launches and the whitening
overlap refusal live here with no Python import, word for word as they were
in the binding (lane thin-core, 2026-10-21: a pure move, no bit change).
"""
from max.gpu.host import DeviceContext
from std.gpu import block_dim, block_idx, grid_dim, thread_idx
from checks.soft_f64 import (
    SF64_ONE, SF64_ZERO, sf64_add, sf64_div, sf64_exp, sf64_from_f32, sf64_gt, sf64_neg,
    sf64_sub, sf64_mul, sf64_to_f32,
)

comptime _F32P = MutPointer[Float32, MutUntrackedOrigin]
comptime _F64P = MutPointer[Float64, MutUntrackedOrigin]


def pca_whiten_overlap_check(
    xa: Int, ma: Int, ca: Int, sa: Int, oa: Int,
    input_count: Int, nf: Int, nc: Int, output_count: Int,
) raises:
    """The whitening entries' refusal of an output span that overlaps any
    input span (x, mean, components, singular values)."""
    var starts = List[Int]()
    starts.append(xa)
    starts.append(ma)
    starts.append(ca)
    starts.append(sa)
    var counts = List[Int]()
    counts.append(input_count)
    counts.append(nf)
    counts.append(nc * nf)
    counts.append(nc)
    for i in range(4):
        if oa < starts[i] + counts[i] * 4 and starts[i] < oa + output_count * 4:
            raise Error("PCA whitening output must not overlap any input")


def ridge_multi_icpt_kernel(
    xmean: MutPointer[Float32, MutAnyOrigin],
    coef: MutPointer[Float32, MutAnyOrigin],
    ymean: MutPointer[Float32, MutAnyOrigin],
    icpt: MutPointer[Float32, MutAnyOrigin],
    n_features_in: Int64,
    n_targets_in: Int64,
):
    """cpu2-l6-bindings: `icpt[j] = ymean[j] - xmean . coef[j, :]`, one
    thread per target, the host loop's statements over
    `checks/soft_f64.mojo`'s binary64 (the Apple GPU has no float64): each
    product of two widened floats is exact, the sum is serial ascending in
    features, the difference and the narrowing are round-to-nearest-even,
    which are the words the host loop wrote."""
    var nf = Int(n_features_in)
    var j = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if j >= Int(n_targets_in):
        return
    var dot = SF64_ZERO
    for c in range(nf):
        dot = sf64_add(
            dot,
            sf64_mul(
                sf64_from_f32(xmean.unsafe_load(c)),
                sf64_from_f32(coef.unsafe_load(j * nf + c)),
            ),
        )
    icpt.unsafe_store(
        j, sf64_to_f32(sf64_sub(sf64_from_f32(ymean.unsafe_load(j)), dot))
    )


def ridge_multi_icpt_device(
    ctx: DeviceContext, xm: _F32P, wp: _F32P, mp: _F32P, ic: _F32P, nf: Int, nt: Int,
) raises:
    """`ridge_fit_multi`'s intercepts on the device
    (`ridge_multi_icpt_kernel`), one thread per target."""
    var d_xm = ctx.enqueue_create_buffer[DType.float32](nf)
    var d_w = ctx.enqueue_create_buffer[DType.float32](nt * nf)
    var d_m = ctx.enqueue_create_buffer[DType.float32](nt)
    var d_ic = ctx.enqueue_create_buffer[DType.float32](nt)
    ctx.enqueue_copy(dst_buf=d_xm, src_ptr=xm)
    ctx.enqueue_copy(dst_buf=d_w, src_ptr=wp)
    ctx.enqueue_copy(dst_buf=d_m, src_ptr=mp)
    ctx.enqueue_function[ridge_multi_icpt_kernel](
        d_xm.unsafe_ptr(), d_w.unsafe_ptr(), d_m.unsafe_ptr(),
        d_ic.unsafe_ptr(), Int64(nf), Int64(nt),
        grid_dim=(nt + 255) // 256, block_dim=256,
    )
    ctx.enqueue_copy(dst_ptr=ic, src_buf=d_ic)
    ctx.synchronize()
    _ = d_xm^
    _ = d_w^
    _ = d_m^
    _ = d_ic^


comptime QN_SOFTMAX_TPB = 128
comptime QN_SOFTMAX_MAX_BLOCKS = 65535


def qn_sigmoid_kernel(
    scores: MutPointer[Float32, MutAnyOrigin],
    dst: MutPointer[UInt64, MutAnyOrigin],
    n_rows_in: Int64,
):
    """The binary `predict_proba` link on the device, one thread per row,
    grid-stride (cpu-gpu-cleanup w2-core-scope; was `qn_sigmoid_host` on
    the host, DEVIATION 549): `p1 = 1 / (1 + exp(-z))`, `p0 = 1 - p1` over
    `checks/soft_f64.mojo`'s binary64 (the Apple GPU has no float64;
    `sf64_exp` is `portable_exp64` statement for statement). Under
    IDENTICAL these are the host column's words
    (`core/classical_host_predict.mojo::host_qn_sigmoid`). `dst` receives
    the binary64 bit patterns, (n_rows, 2) row-major."""
    var n_rows = Int(n_rows_in)
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var stride = Int(grid_dim.x) * Int(block_dim.x)
    while i < n_rows:
        var nz = sf64_neg(sf64_from_f32(scores.unsafe_load(i)))
        var p = sf64_div(SF64_ONE, sf64_add(SF64_ONE, sf64_exp(nz)))
        dst.unsafe_store(2 * i, sf64_sub(SF64_ONE, p))
        dst.unsafe_store(2 * i + 1, p)
        i += stride


def qn_sigmoid_device(ctx: DeviceContext, sp: _F32P, op: _F64P, nr: Int) raises:
    """`qn_sigmoid`'s upload, launch and read-back (`qn_sigmoid_kernel`)."""
    var d_scores = ctx.enqueue_create_buffer[DType.float32](nr)
    ctx.enqueue_copy(dst_buf=d_scores, src_ptr=sp)
    var d_out = ctx.enqueue_create_buffer[DType.uint64](2 * nr)
    ctx.enqueue_function[qn_sigmoid_kernel](
        d_scores.unsafe_ptr(), d_out.unsafe_ptr(), Int64(nr),
        grid_dim=min((nr + QN_SOFTMAX_TPB - 1) // QN_SOFTMAX_TPB, QN_SOFTMAX_MAX_BLOCKS),
        block_dim=QN_SOFTMAX_TPB,
    )
    ctx.enqueue_copy(dst_ptr=op.bitcast[UInt64](), src_buf=d_out)
    ctx.synchronize()
    _ = d_scores^
    _ = d_out^


def qn_softmax_kernel(
    scores: MutPointer[Float32, MutAnyOrigin],
    dst: MutPointer[UInt64, MutAnyOrigin],
    n_rows_in: Int64,
    n_classes_in: Int64,
):
    """The multinomial `predict_proba` link on the device, one thread per
    row, grid-stride (cpu-gpu-cleanup t-gbdt). `qn_softmax_host`'s
    statements (`glm/estimator.mojo`, lane/logistic-multiclass) over
    `checks/soft_f64.mojo`'s binary64 (the Apple GPU has no float64; the
    soft ops are IEEE round-to-nearest, `sf64_exp` is `portable_exp64`
    statement for statement): `m` the first maximum under a strict `>` from
    the first entry, `s` the serial ascending sum of `exp(z_c - m)`,
    `p_c = exp(z_c - m) / s`. Under IDENTICAL these are the host column's
    words (`core/classical_host_predict.mojo::host_qn_softmax`). `out`
    receives the binary64 bit patterns."""
    var n_rows = Int(n_rows_in)
    var nc = Int(n_classes_in)
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var stride = Int(grid_dim.x) * Int(block_dim.x)
    while i < n_rows:
        var base = i * nc
        var m = sf64_from_f32(scores.unsafe_load(base))
        for c in range(1, nc):
            var v = sf64_from_f32(scores.unsafe_load(base + c))
            if sf64_gt(v, m):
                m = v
        var acc = SF64_ZERO
        for c in range(nc):
            var z = sf64_from_f32(scores.unsafe_load(base + c))
            acc = sf64_add(acc, sf64_exp(sf64_sub(z, m)))
        for c in range(nc):
            var z = sf64_from_f32(scores.unsafe_load(base + c))
            dst.unsafe_store(base + c, sf64_div(sf64_exp(sf64_sub(z, m)), acc))
        i += stride




def qn_softmax_device(ctx: DeviceContext, sp: _F32P, op: _F64P, nr: Int, nc: Int) raises:
    """`qn_softmax`'s upload, launch and read-back (`qn_softmax_kernel`)."""
    var n = nr * nc
    var d_scores = ctx.enqueue_create_buffer[DType.float32](n)
    ctx.enqueue_copy(dst_buf=d_scores, src_ptr=sp)
    var d_out = ctx.enqueue_create_buffer[DType.uint64](n)
    ctx.enqueue_function[qn_softmax_kernel](
        d_scores.unsafe_ptr(), d_out.unsafe_ptr(), Int64(nr), Int64(nc),
        grid_dim=min((nr + QN_SOFTMAX_TPB - 1) // QN_SOFTMAX_TPB, QN_SOFTMAX_MAX_BLOCKS),
        block_dim=QN_SOFTMAX_TPB,
    )
    ctx.enqueue_copy(dst_ptr=op.bitcast[UInt64](), src_buf=d_out)
    ctx.synchronize()
    _ = d_scores^
    _ = d_out^
