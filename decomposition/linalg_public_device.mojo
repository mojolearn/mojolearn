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
  `device_eigh`      `decomposition/checks/jacobi_eigh_device.mojo::
                     jacobi_eigh_kernel` then `sign_flip_kernel`, the pair
                     `eig_and_truncate` launches, at the same launch width
                     and the same sweep budget.
  `device_svdvals`   `qr_factor` then `decomposition/impl/linalg/detail/
                     svd_full.mojo::svd_of_r`, which is `pca_fit_full`'s
                     tall arm with the centering left out.

THE PERMUTATION IS NOT REPEATED HERE EITHER. `eigh_ascending` and
`svdvals_descending` live in the host twin and BOTH routes call them, so
there is one order per public name in this tree and not two. That matters
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

from core.device_zero import enqueue_fill
from core.householder_qr import qr_factor, qr_slice_count
from decomposition.checks.jacobi_eigh_device import (
    JACOBI_INFO_UNWRITTEN,
    JACOBI_ROT_TPB,
    JACOBI_SWEEPS,
    JACOBI_TOL,
    jacobi_eigh_kernel,
)
from decomposition.host.linalg_public import (
    EighHostResult,
    _validate_shape,
    _validate_square,
    eigh_ascending,
    svdvals_descending,
)
from decomposition.impl.linalg.detail.pca import SIGNFLIP_TPB, sign_flip_kernel
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
    var h = ctx.enqueue_create_host_buffer[DType.float32](n)
    ctx.enqueue_copy(dst_ptr=h.unsafe_ptr(), src_buf=buf)
    ctx.synchronize()
    var out = List[Float32]()
    for i in range(n):
        out.append(h.unsafe_ptr().unsafe_load(i))
    _ = h^
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
    _validate_shape(n_rows, n_cols, "qr")
    var ctx = DeviceContext()
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
    # DEVIATION 1946: the context dies LAST, after every value built on it.
    _ = ctx^
    return r^


def device_eigh(a: List[Float32], n: Int) raises -> EighHostResult:
    """`numpy.linalg.eigh(a)` on the device, symmetric `a`, ASCENDING.

    The pair `eig_and_truncate` launches -- `jacobi_eigh_kernel` at
    `JACOBI_ROT_TPB` and then `sign_flip_kernel` -- at the sweep budget and
    tolerance every shipped caller uses. The sign flip is not optional and
    not cosmetic: without it two boxes agreeing bit for bit could still hand
    back `v` and `-v`, because the sweep fixes a vector only up to sign.

    THE INFO BUFFER IS PRE-FILLED WITH `JACOBI_INFO_UNWRITTEN` and the two
    failures are separated by name. A kernel that never ran leaves the
    sentinel; a kernel that ran and did not converge writes 0.0. Reading
    both as "not converged" is how a launch failure gets reported as a
    numerical one, which sends the reader at the data instead of at the
    build (`glm/impl/linalg/detail/lstsq.mojo` carries the same guard and
    the same argument).
    """
    _validate_square(n, "eigh")
    var ctx = DeviceContext()
    var da = _upload(ctx, a)
    var dv = ctx.enqueue_create_buffer[DType.float32](n * n)
    var dinfo = ctx.enqueue_create_buffer[DType.float32](3)
    enqueue_fill(ctx, dinfo, JACOBI_INFO_UNWRITTEN)
    ctx.synchronize()
    ctx.enqueue_function[jacobi_eigh_kernel[JACOBI_ROT_TPB]](
        da.unsafe_ptr(),
        dv.unsafe_ptr(),
        dinfo.unsafe_ptr(),
        Int32(n),
        Int32(JACOBI_SWEEPS),
        Float32(JACOBI_TOL),
        grid_dim=(1, 1, 1),
        block_dim=(JACOBI_ROT_TPB, 1, 1),
    )
    ctx.enqueue_function[sign_flip_kernel](
        dv.unsafe_ptr(),
        Int32(n),
        grid_dim=(n, 1, 1),
        block_dim=(SIGNFLIP_TPB, 1, 1),
    )
    ctx.synchronize()
    var info = _download(ctx, dinfo, 3)
    var work = _download(ctx, da, n * n)
    var vecs = _download(ctx, dv, n * n)
    _ = da^
    _ = dv^
    _ = dinfo^
    ctx.synchronize()
    _ = ctx^

    if info[0] == JACOBI_INFO_UNWRITTEN:
        raise Error(
            "eigh: the device Jacobi eigensolver DID NOT WRITE its info"
            " buffer, so it never ran or its launch failed. This is NOT a"
            " convergence failure and must not be reported as one: -1.0 is a"
            " value the kernel never stores. Check that the binding is built"
            " for this device."
        )
    if info[0] == Float32(0.0):
        raise Error(
            "eigh: the Jacobi eigensolver did not converge in "
            + String(JACOBI_SWEEPS)
            + " sweeps at n = "
            + String(n)
            + ": ||offdiag(A)||_F / ||A||_F is still "
            + String(info[1])
            + ". An unconverged decomposition is not returned as if it were"
            " one; see DEVIATION 590. The remedy is more sweeps, the same one"
            " cuSOLVER's syevj has"
        )

    var diag = List[Float32]()
    for i in range(n):
        diag.append(work[i * n + i])
    return eigh_ascending(diag, vecs, n, True, Int(info[2]))


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
    var ctx = DeviceContext()
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
    var s = _download(ctx, s_buf, n_cols)
    _ = da^
    _ = scratch^
    _ = r_buf^
    _ = v_buf^
    _ = s_buf^
    ctx.synchronize()
    _ = ctx^
    return svdvals_descending(s, n_cols)
