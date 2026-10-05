# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
# SHIPS: compiled into the GPU linalg binding (bindings/_mojolearn_linalg.mojo); product, not only a check.
"""THE DEVICE ROUTE OF THE THREE PUBLIC DECOMPOSITIONS (2026-09-19).

`decomposition/host/linalg_public.mojo` is the host twin of this file and
was written first. It said, in its own header and in
`python/mojolearn/_linalg_impl.py`, that the host route was taken on EVERY
box on purpose, "because there is no one-shot device door for them yet".
That sentence described a missing file, not a decision, and it cost the
three doors every GPU column they could have had: a decomposition that only
ever runs on the host cannot be checked across vendors, which is the one
claim this library exists to make. THIS IS THAT FILE. `qr`, `eigh` and
`svdvals` now run the SAME KERNELS `PCA(svd_solver='full')` runs, on the
device, on a GPU install.

THE RANKING BETWEEN THE TWO FILES, so nobody has to guess it again: THIS
ONE IS THE PRODUCT AND THE HOST TWIN IS THE VERIFIER. The kernels are what a
caller on a GPU box runs; the oracles exist to re-derive that answer
serially so the two can be diffed. A CPU-only box has no device to run, so
there the verifier is also the route -- that is the only case in which host
arithmetic is somebody's answer rather than somebody's check.

NO NEW ARITHMETIC, EXACTLY AS THE HOST TWIN HAS NONE. Not one fold,
rotation, reflector or square root is written here. Each entry validates on
the HOST and refuses BY NAME before any upload, then uploads, launches the
SHIPPING kernels and downloads:

  `device_qr_r`      `core/householder_qr.mojo::qr_factor` -- both slice
                     arms, the scratch sized by `qr_slice_count` exactly as
                     `pca_fit_full` sizes it.
  (eigh)             deleted (cgr-decomp, 2026-10-03): the one-block
                     cyclic `jacobi_eigh_kernel`; numpy's eigh is x_decomp's
                     round-robin Jacobi (`DevExec.eigh`) on every column.
  `device_svdvals`   `qr_factor` then `decomposition/impl/linalg/detail/
                     svd_full.mojo::svd_of_r`, which is `pca_fit_full`'s
                     tall arm with the centering left out.

THE PERMUTATION is one definition: the host twin's `_argsort_desc` and the
device's `enqueue_eigh_ascending` / `enqueue_svdvals_descending`
(decomposition/spectrum_order_device.mojo) both rank by `spectrum_rank_desc`,
so there is one order per public name in this tree and not two. That matters
more than it looks: the host and the device can agree on every fold and
still disagree on the ANSWER if one of them sorts differently, and a hash
would report that as a divergence without saying which half moved.

WHY HOST IN AND HOST OUT. This is `cholesky/estimator.mojo::
cholesky_factor_host`'s shape and it is the shape the CPython bindings know
how to call. A caller that keeps a matrix on the device across many
operations should call `qr_factor`, `svd_of_r` and `jacobi_eigh_kernel`
directly with its own buffers, exactly as `pca_fit_full` does; this is the
ONE-SHOT form, and one shot is what a public `numpy.linalg` name means.

WHAT THIS FILE MAY NOT BECOME. It must never grow a branch that computes an
answer on the host when the device is present. The whole value of the door
is that the two routes are the same arithmetic reached two ways and can
therefore be held against each other bit for bit; a silent fallback would
turn a divergence into a pass.
"""

from bindings.hostptr import copy_f32
from max.gpu.host import DeviceBuffer, DeviceContext
from core.neural_context import process_ctx
from checks.numerics import GLOBAL_NUMERIC_MODE as _DEVCTX_MODE, NUMERIC_IDENTICAL as _DEVCTX_IDENTICAL

#: This binding's ONE process-lifetime DeviceContext (core/neural_context.mojo,
#: lane/devctx-lifetime): a context per call exhausts Metal command queues.
comptime _DEVCTX_SLOT = "MojoLinalgContextIdentical" if _DEVCTX_MODE == _DEVCTX_IDENTICAL else "MojoLinalgContextFast"


from core.householder_qr import qr_factor, qr_slice_count
from decomposition.linalg_types import _validate_shape
from decomposition.spectrum_order_device import enqueue_svdvals_descending
from decomposition.impl.linalg.detail.svd_full import svd_of_r


def _upload(
    ctx: DeviceContext, values: List[Float32]
) raises -> DeviceBuffer[DType.float32]:
    """`cholesky/estimator.mojo::_upload`, for its reason: one staged host
    buffer, one copy, one wait."""
    var n = len(values)
    var buf = ctx.enqueue_create_buffer[DType.float32](n)
    var host = ctx.enqueue_create_host_buffer[DType.float32](n)
    copy_f32(values.unsafe_ptr(), host.unsafe_ptr(), n)
    ctx.enqueue_copy(dst_buf=buf, src_ptr=host.unsafe_ptr())
    ctx.synchronize()
    _ = host^
    return buf^


def _download(
    ctx: DeviceContext, mut buf: DeviceBuffer[DType.float32], n: Int
) raises -> List[Float32]:
    # the result lands straight in the list (no host copy loop)
    var out = List[Float32](length=n, fill=Float32(0.0))
    if n > 0:
        var head = buf.create_sub_buffer[DType.float32](0, n)
        ctx.enqueue_copy(dst_ptr=out.unsafe_ptr(), src_buf=head)
        ctx.synchronize()
        _ = head^
    return out^


def device_qr_r(
    a: List[Float32], n_rows: Int, n_cols: Int
) raises -> List[Float32]:
    """`numpy.linalg.qr(a, mode='r')` on the device: R, `n_cols x n_cols`
    row major. The device twin of `host_qr_r` and bit for bit its answer.

    `a` is `n_rows x n_cols` row major and is NOT destroyed -- `qr_factor`
    destroys the buffer it is given, and the buffer it is given here is the
    upload, never the caller's list.

    `r_scratch` is sized by `qr_slice_count`, the SAME function the shipped
    caller sizes it with (`pca_full_scratch_cells`). Sizing it here from a
    count computed some other way is the bug that arm has already had once:
    a scratch sized for one slice arm and a dispatch that took the other is
    an out-of-bounds write a small shape does not show you.
    """
    return device_qr_r(process_ctx[_DEVCTX_SLOT](), a, n_rows, n_cols)


def device_qr_r(
    ctx: DeviceContext, a: List[Float32], n_rows: Int, n_cols: Int
) raises -> List[Float32]:
    """`device_qr_r` on a caller's context (a binding with one
    process-lifetime context: x_decomp). Every buffer is freed and drained
    before return; the context is the caller's to keep."""
    _validate_shape(n_rows, n_cols, "qr")
    var da = _upload(ctx, a)
    var scratch = ctx.enqueue_create_buffer[DType.float32](
        qr_slice_count(n_rows, n_cols) * n_cols * n_cols
    )
    var r_buf = ctx.enqueue_create_buffer[DType.float32](n_cols * n_cols)
    ctx.synchronize()
    _ = qr_factor(ctx, da, scratch, r_buf, n_rows, n_cols)
    var r = _download(ctx, r_buf, n_cols * n_cols)
    _ = da^
    _ = scratch^
    _ = r_buf^
    # DEVIATION 3010: the buffers above are gone; DRAIN the frees they
    # enqueued before this scope's end destroys the context. Host-side
    # drain, no arithmetic.
    ctx.synchronize()
    return r^


def device_svdvals(
    a: List[Float32], n_rows: Int, n_cols: Int
) raises -> List[Float32]:
    """`numpy.linalg.svdvals(a)` on the device: the values, DESCENDING.

    `pca_fit_full`'s tall arm without the centering: the Householder QR of
    `a`, then the one-sided Jacobi SVD of R. The singular values of R ARE
    the singular values of `a` because Q is orthogonal, and that identity is
    why this entry can exist without forming Q at all.

    `svd_of_r` owns the convergence refusal and raises by name (DEVIATION
    590); it is not re-stated here, because a second copy of a refusal is a
    second wording of it.
    """
    _validate_shape(n_rows, n_cols, "svdvals")
    var ctx = process_ctx[_DEVCTX_SLOT]()
    var da = _upload(ctx, a)
    var scratch = ctx.enqueue_create_buffer[DType.float32](
        qr_slice_count(n_rows, n_cols) * n_cols * n_cols
    )
    var r_buf = ctx.enqueue_create_buffer[DType.float32](n_cols * n_cols)
    var v_buf = ctx.enqueue_create_buffer[DType.float32](n_cols * n_cols)
    var s_buf = ctx.enqueue_create_buffer[DType.float32](n_cols)
    ctx.synchronize()
    _ = qr_factor(ctx, da, scratch, r_buf, n_rows, n_cols)
    svd_of_r(ctx, r_buf, v_buf, s_buf, n_cols)
    # descending on the device (decomposition/spectrum_order_device.mojo)
    var so = ctx.enqueue_create_buffer[DType.float32](n_cols)
    var spos = ctx.enqueue_create_buffer[DType.int32](n_cols)
    enqueue_svdvals_descending(
        ctx, s_buf.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin](), n_cols, spos,
        so.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin](),
    )
    var s = _download(ctx, so, n_cols)
    _ = so^
    _ = spos^
    _ = da^
    _ = scratch^
    _ = r_buf^
    _ = v_buf^
    _ = s_buf^
    ctx.synchronize()
    _ = ctx^
    return s^
