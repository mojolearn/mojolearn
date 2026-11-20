# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The Laplace Newton loop of Gaussian process classification.

ROUTE (lane restore-classical-measured-routes, 2026-11-13).
`gpc_newton_device_batch` runs each class through `gpc_newton_host_loop`,
0.8.66's measured loop (`potrf_lower` + `cho_solve`, one stop-test read per
iteration). The executor's bounded device loop on the Cholesky ops (lane
n8-gp-krr-smo, -D MOJOLEARN_IDN_CHOL_OPS_FIT) was the B arm of its A/B: the
same speed on the L40S (ba08dac006), deleted 2026-11-17 (docs/TOMBSTONES.md).

ONE LOOP. The binary fit (`classifier.mojo::gpc_fit_binary_host`) and the
one-vs-rest fit (`gpc_ovr.mojo::gpc_fit_all_device`) each kept their own
copy of the same loop; both now call `gpc_newton_device` here.
"""

from std.atomic import Atomic
from max.gpu import block_idx, thread_idx
from max.gpu.host import DeviceBuffer, DeviceContext, HostBuffer

from cholesky.checks.potrf import (
    CHOL_ELEM_TPB,
    CHOL_NB_PINNED,
    CHOL_PANEL_TPB,
    add_jitter,
    chol_default_nb_hint,
    chol_nb_for,
    chol_workspace_floats,
    enqueue_logdet,
    potrf_lower,
)
from cholesky.checks.trsm import CHOL_SOLVE_TPB, cho_solve
from cholesky.impl.matrix.detail.matrix import copy_vector_from_matrix_diagonal_kernel
from cholesky.logdet_fold import logdet_blocks
from core.identity_trace import IdentityTrace
from checks.numerics import ftz, identical_mul
from core.device_zero import enqueue_fill
from gaussian_process.gpc_common import gpc_neg_inf32, gpc_stop
from gaussian_process.gpc_items import (
    gpc_a_item,
    gpc_fold_blocks,
    gpc_lml_fin,
    gpc_lml_part_item,
    gpc_rhs_item,
    gpc_scale_item,
    gpc_weight_item,
)
from gemm.checks.gemm_identical import identical_gemm_into
from gemm.contract import OP_TN

comptime _GP = MutPointer[Float32, MutAnyOrigin]
comptime _GI = MutPointer[Int32, MutAnyOrigin]

#: SCHEDULING: one thread per row (or per cell of B), the classifier's
#: GPC_STEP_TPB and GPC_B_TPB.
comptime GPC_NEWTON_TPB = 256


@always_inline
def _tid() -> Int:
    return Int(block_idx.x) * GPC_NEWTON_TPB + Int(thread_idx.x)


@always_inline
def _blocks(count: Int) -> Int:
    return (count + GPC_NEWTON_TPB - 1) // GPC_NEWTON_TPB if count > 0 else 1


@always_inline
def _fp(buf: DeviceBuffer[DType.float32]) -> _GP:
    return _GP(unsafe_from_address=Int(buf.unsafe_ptr()))


@always_inline
def _ip(buf: DeviceBuffer[DType.int32]) -> _GI:
    return _GI(unsafe_from_address=Int(buf.unsafe_ptr()))


# TOMBSTONE: MOJOLEARN_IDN_CHOL_OPS_FIT (decided B arm, the executor Cholesky ops; L40S ba08dac006 14 cells B 0.91-1.10x, same speed) deleted 2026-11-17 by lane/fb12-loser-arms; code recoverable at 2203db7094.
# Restore: git apply experiments/removed/MOJOLEARN_IDN_CHOL_OPS_FIT.fb12-loser-arms.patch; record in docs/TOMBSTONES.md.
# (the bounded device loop: GPCNewtonStep, its gated kernels, the criterion and join kernels, GPCNewtonBatch)


@fieldwise_init
struct GPCNewtonRun(ImplicitlyCopyable, Movable):
    """What the Newton loop leaves besides the resident factor, pi and W_sr:
    the likelihood kept (DEVIATION 2830), the iterations and the Cholesky
    panel width that ran."""

    var lml: Float32
    var n_iter: Int
    var nb: Int


# ---------------------------------------------------------------------------
# The measured 0.8.66 Newton loop (the default; lane
# restore-classical-measured-routes, 2026-11-13).
#
# The bounded device loop (-D MOJOLEARN_IDN_CHOL_OPS_FIT, deleted 2026-11-17)
# went in (24c4eed3a5, e28e5af236, 2026-11-09) without an NVIDIA + AMD
# measurement; AMD measured it 1.20x slower (MI325X, 2026-11-13) and the
# L40S A/B the same speed (ba08dac006). Each class runs 0.8.66's loop
# (`gpc_ovr.mojo::gpc_newton_device` at 24c4eed3a5^, copied): `potrf_lower`
# with its info read, log|B| left on the device (`enqueue_logdet`, the
# 0.8.66 IDENTICAL default GPC_IDN_NEWTON_LOGDET), `cho_solve`, and one read
# of the likelihood and the NaN index per iteration for the stop test. The
# kernels below are 0.8.66's ungated Newton kernels (classifier.mojo /
# gpc_ovr.mojo at 24c4eed3a5^).
# ---------------------------------------------------------------------------


def gpc_w_kernel(f: _GP, pi: _GP, w: _GP, wsr: _GP, n: Int32):
    var i = _tid()
    if i < Int(n):
        gpc_weight_item(i, f, pi, w, wsr)


def gpc_b_matrix_kernel(b: _GP, k: _GP, wsr: _GP, n_in: Int32):
    var n = Int(n_in)
    var e = Int(block_idx.x) * GPC_NEWTON_TPB + Int(thread_idx.x)
    if e >= n * n:
        return
    var i = e // n
    var j = e - i * n
    var ww = ftz(identical_mul(ftz(wsr.unsafe_load(i)), ftz(wsr.unsafe_load(j))))
    var cell = ftz(identical_mul(ww, ftz(k.unsafe_load(e))))
    if i == j:
        cell = ftz(Float32(1.0) + cell)
    b.unsafe_store(e, cell)


def gpc_rhs_kernel(w: _GP, f: _GP, y: _GP, pi: _GP, dst: _GP, n: Int32):
    var i = _tid()
    if i < Int(n):
        gpc_rhs_item(i, w, f, y, pi, dst)


def gpc_scale_nan_kernel(wsr: _GP, v: _GP, dst: _GP, nan_at: _GI, n: Int32):
    """`gpc_scale`, and the lowest index whose value is NaN (DEVIATION 1638's
    refusal of a NaN right-hand side) by an atomic min."""
    var i = _tid()
    if i < Int(n):
        gpc_scale_item(i, wsr, v, dst)
        var c = dst.unsafe_load(i)
        if c != c:
            _ = Atomic[Int32].min(nan_at, Int32(i))


def gpc_a_kernel(b: _GP, wsr: _GP, x: _GP, dst: _GP, n: Int32):
    var i = _tid()
    if i < Int(n):
        gpc_a_item(i, b, wsr, x, dst)


def gpc_lml_part_kernel(a: _GP, f: _GP, y: _GP, n: Int32, pdot: _GP, pt2: _GP, nb: Int32):
    var blk = _tid()
    if blk < Int(nb):
        gpc_lml_part_item(blk, a, f, y, Int(n), pdot, pt2)


def gpc_lml_fin_dev_kernel(pdot: _GP, pt2: _GP, nb: Int32, logdet: _GP, dst: _GP):
    """The likelihood with log|B| read from the device scalar
    `enqueue_logdet` wrote, not passed through the host."""
    if _tid() == 0:
        dst.unsafe_store(0, gpc_lml_fin(pdot, pt2, Int(nb), logdet.unsafe_load(0)))


struct GPCHostLoopScratch(Movable):
    """The per-class vectors of `gpc_newton_host_loop` (f, W, b, K b, c, a,
    the likelihood fold's partials and scalar, the NaN guard's word, the
    log-determinant fold's partials, and the two host words of the stop
    test), allocated ONCE per batch and reused by every class's loop: every
    class of a batch has the same n. Lane fix-gmm-chol-leftovers
    (2026-11-13): each class's loop allocated and freed all thirteen. Every
    vector is written before it is read in each loop (f is zero-filled at
    the loop's start), so reuse changes no word."""
    var df: DeviceBuffer[DType.float32]
    var dw: DeviceBuffer[DType.float32]
    var dbv: DeviceBuffer[DType.float32]
    var dkb: DeviceBuffer[DType.float32]
    var dc: DeviceBuffer[DType.float32]
    var da: DeviceBuffer[DType.float32]
    var dpd: DeviceBuffer[DType.float32]
    var dpt: DeviceBuffer[DType.float32]
    var dlml: DeviceBuffer[DType.float32]
    var dnan: DeviceBuffer[DType.int32]
    var dlparts: DeviceBuffer[DType.float32]
    var hlml: HostBuffer[DType.float32]
    var hnan: HostBuffer[DType.int32]

    def __init__(out self, ctx: DeviceContext, n: Int) raises:
        var nbf = gpc_fold_blocks(n)
        self.df = ctx.enqueue_create_buffer[DType.float32](n)
        self.dw = ctx.enqueue_create_buffer[DType.float32](n)
        self.dbv = ctx.enqueue_create_buffer[DType.float32](n)
        self.dkb = ctx.enqueue_create_buffer[DType.float32](n)
        self.dc = ctx.enqueue_create_buffer[DType.float32](n)
        self.da = ctx.enqueue_create_buffer[DType.float32](n)
        self.dpd = ctx.enqueue_create_buffer[DType.float32](max(nbf, 1))
        self.dpt = ctx.enqueue_create_buffer[DType.float32](max(nbf, 1))
        self.dlml = ctx.enqueue_create_buffer[DType.float32](1)
        self.dnan = ctx.enqueue_create_buffer[DType.int32](1)
        self.dlparts = ctx.enqueue_create_buffer[DType.float32](2 * logdet_blocks(n))
        self.hlml = ctx.enqueue_create_host_buffer[DType.float32](1)
        self.hnan = ctx.enqueue_create_host_buffer[DType.int32](1)


def gpc_newton_host_loop(
    ctx: DeviceContext,
    mut scratch: GPCHostLoopScratch,
    mut dk: DeviceBuffer[DType.float32],
    mut dy: DeviceBuffer[DType.float32],
    mut db: DeviceBuffer[DType.float32],
    mut ws: DeviceBuffer[DType.float32],
    mut dwork: DeviceBuffer[DType.float32],
    mut dgw: DeviceBuffer[DType.float32],
    mut dpi: DeviceBuffer[DType.float32],
    mut dwsr: DeviceBuffer[DType.float32],
    n: Int,
    max_iter_predict: Int,
) raises -> GPCNewtonRun:
    """0.8.66's Laplace Newton loop of one class against a resident K and
    resident targets: the same launches in the same order. On return `db`
    holds the last factor, `dpi` and `dwsr` the last pi and W_sr. `ws` holds
    `chol_workspace_floats(n, chol_nb_for(n, CHOL_NB_PINNED))` floats,
    `dwork` n + 1, `scratch` the loop's vectors at this n (one per batch,
    `GPCHostLoopScratch`). One word pair crosses back per iteration for the
    stop test: the likelihood and the NaN guard's index."""
    # handle copies of the batch's scratch: the same device memory, no
    # allocation per class (GPCHostLoopScratch)
    var df = scratch.df.copy()
    var dw = scratch.dw.copy()
    var dbv = scratch.dbv.copy()
    var dkb = scratch.dkb.copy()
    var dc = scratch.dc.copy()
    var da = scratch.da.copy()
    var nbf = gpc_fold_blocks(n)
    var dpd = scratch.dpd.copy()
    var dpt = scratch.dpt.copy()
    var dlml = scratch.dlml.copy()
    var dnan = scratch.dnan.copy()
    var dlparts = scratch.dlparts.copy()
    var diag = dwork.create_sub_buffer[DType.float32](0, n)
    var scalar = dwork.create_sub_buffer[DType.float32](n, 1)
    enqueue_fill(ctx, df, Float32(0.0))
    var trace = IdentityTrace()
    var previous = gpc_neg_inf32()
    var n_iter = 0
    var nb = 0
    for it in range(max_iter_predict):  # driver-loop: over Newton iterations (one stop-test read each, the 0.8.66 route)
        ctx.enqueue_function[gpc_w_kernel](
            _fp(df), _fp(dpi), _fp(dw), _fp(dwsr), Int32(n), grid_dim=_blocks(n), block_dim=GPC_NEWTON_TPB
        )
        ctx.enqueue_function[gpc_b_matrix_kernel](
            _fp(db), _fp(dk), _fp(dwsr), Int32(n), grid_dim=_blocks(n * n), block_dim=GPC_NEWTON_TPB
        )
        add_jitter(ctx, db, n, Float32(0.0), CHOL_ELEM_TPB)
        var run = potrf_lower(ctx, db, ws, n, trace, chol_default_nb_hint(), CHOL_PANEL_TPB, CHOL_ELEM_TPB)
        if run.info != 0:
            raise Error(
                "gpc_fit_host: the factorization of B = I + W_sr K W_sr failed"
                " at Newton iteration "
                + String(it + 1)
                + " (info="
                + String(run.info)
                + "). B's eigenvalues are at least 1 for any finite kernel"
                " matrix, so this means a non-finite latent value"
            )
        # `chol_logdet`'s two launches, its scalar left where it lies
        ctx.enqueue_function[copy_vector_from_matrix_diagonal_kernel](
            diag.unsafe_ptr(), db.unsafe_ptr(), Int32(n), Int32(n),
            grid_dim=((n + CHOL_ELEM_TPB - 1) // CHOL_ELEM_TPB, 1, 1), block_dim=(CHOL_ELEM_TPB, 1, 1),
        )
        enqueue_logdet(ctx, _fp(diag), _fp(dlparts), _fp(scalar), n)
        ctx.enqueue_function[gpc_rhs_kernel](
            _fp(dw), _fp(df), _fp(dy), _fp(dpi), _fp(dbv), Int32(n), grid_dim=_blocks(n), block_dim=GPC_NEWTON_TPB
        )
        identical_gemm_into(ctx, dkb, dk, dbv, dgw, n, 1, n, OP_TN)
        enqueue_fill(ctx, dnan, Int32(n))
        ctx.enqueue_function[gpc_scale_nan_kernel](
            _fp(dwsr), _fp(dkb), _fp(dc), _ip(dnan), Int32(n), grid_dim=_blocks(n), block_dim=GPC_NEWTON_TPB,
        )
        cho_solve(ctx, db, dc, n, 1, trace, CHOL_SOLVE_TPB)
        ctx.enqueue_function[gpc_a_kernel](
            _fp(dbv), _fp(dwsr), _fp(dc), _fp(da), Int32(n), grid_dim=_blocks(n), block_dim=GPC_NEWTON_TPB
        )
        identical_gemm_into(ctx, df, dk, da, dgw, n, 1, n, OP_TN)
        if nbf > 0:
            ctx.enqueue_function[gpc_lml_part_kernel](
                _fp(da), _fp(df), _fp(dy), Int32(n), _fp(dpd), _fp(dpt), Int32(nbf),
                grid_dim=_blocks(nbf), block_dim=GPC_NEWTON_TPB,
            )
        ctx.enqueue_function[gpc_lml_fin_dev_kernel](  # small-launch(nbf: fold blocks of the likelihood): one thread combines the partials into one scalar
            _fp(dpd), _fp(dpt), Int32(nbf), _fp(scalar), _fp(dlml), grid_dim=1, block_dim=1
        )
        ctx.enqueue_copy(dst_ptr=scratch.hlml.unsafe_ptr(), src_buf=dlml)
        ctx.enqueue_copy(dst_ptr=scratch.hnan.unsafe_ptr(), src_buf=dnan)
        ctx.synchronize()
        var nan_at = Int(scratch.hnan.unsafe_ptr().unsafe_load(0))
        if nan_at < n:
            raise Error(
                "cholesky_solve_host: the right-hand side contains NaN at"
                " flat index "
                + String(nan_at)
                + "; refused by name (DEVIATION 1638)"
            )
        var lml = scratch.hlml.unsafe_ptr().unsafe_load(0)
        n_iter = it + 1
        nb = run.nb
        if gpc_stop(lml, previous):
            break
        previous = lml
    ctx.synchronize()
    _ = diag^
    _ = scalar^
    _ = df^
    _ = dw^
    _ = dbv^
    _ = dkb^
    _ = dc^
    _ = da^
    _ = dpd^
    _ = dpt^
    _ = dlml^
    _ = dnan^
    _ = dlparts^
    return GPCNewtonRun(previous, n_iter, nb)


def gpc_newton_device_batch(
    ctx: DeviceContext,
    mut dk: DeviceBuffer[DType.float32],
    dys: List[DeviceBuffer[DType.float32]],
    dbs: List[DeviceBuffer[DType.float32]],
    mut dgw: DeviceBuffer[DType.float32],
    dpis: List[DeviceBuffer[DType.float32]],
    dwsrs: List[DeviceBuffer[DType.float32]],
    n: Int,
    max_iter_predict: Int,
) raises -> List[GPCNewtonRun]:
    """The Laplace Newton loop of every class in `dys` against ONE resident
    K (`dk`, n x n), one class after another (`gpc_newton_host_loop`). Class c
    leaves its last factor in `dbs[c]` and its pi and W_sr in `dpis[c]`,
    `dwsrs[c]`; the GEMV workspace `dgw` is shared (one stream: the classes'
    GEMVs run one after another). Raises by name for the lowest class whose
    factor failed or whose right-hand side was NaN, as that class's own loop
    did."""
    var k = len(dys)
    if len(dbs) != k or len(dpis) != k or len(dwsrs) != k or k < 1:
        raise Error("gpc_newton_device_batch: one target, factor, pi and W_sr buffer per class")
    var runs = List[GPCNewtonRun](capacity=k)
    if max_iter_predict < 1:
        # no iteration: the likelihood kept is the start's -inf
        for _c in range(k):  # driver-loop: over the classes (model structure)
            runs.append(GPCNewtonRun(gpc_neg_inf32(), 0, 0))
        return runs^
    # TOMBSTONE: MOJOLEARN_IDN_CHOL_OPS_FIT (decided B arm, the executor Cholesky ops; L40S ba08dac006 14 cells B 0.91-1.10x, same speed) deleted 2026-11-17 by lane/fb12-loser-arms; code recoverable at 2203db7094.
    # Restore: git apply experiments/removed/MOJOLEARN_IDN_CHOL_OPS_FIT.fb12-loser-arms.patch; record in docs/TOMBSTONES.md.
    # The measured 0.8.66 route (see `gpc_newton_host_loop`): each
    # class's loop after the previous one, sharing one Cholesky
    # workspace as 0.8.66's one-vs-rest fit did.
    var nb_pin = chol_nb_for(n, CHOL_NB_PINNED)
    var ws = ctx.enqueue_create_buffer[DType.float32](chol_workspace_floats(n, nb_pin))
    var dwork = ctx.enqueue_create_buffer[DType.float32](n + 1)
    # the loop's vectors, once per batch (every class has this n)
    var hscratch = GPCHostLoopScratch(ctx, n)
    var dkh = dk.copy()  # a handle copy: the same device memory
    var dgwh = dgw.copy()
    for c in range(k):  # driver-loop: over the classes (model structure)
        var dy = dys[c].copy()
        var db = dbs[c].copy()
        var dpi = dpis[c].copy()
        var dwsr = dwsrs[c].copy()
        runs.append(
            gpc_newton_host_loop(ctx, hscratch, dkh, dy, db, ws, dwork, dgwh, dpi, dwsr, n, max_iter_predict)
        )
    ctx.synchronize()
    _ = ws^
    _ = dwork^
    _ = hscratch^
    _ = dkh^
    _ = dgwh^
    return runs^

def gpc_newton_device(
    ctx: DeviceContext,
    mut dk: DeviceBuffer[DType.float32],
    mut dy: DeviceBuffer[DType.float32],
    mut db: DeviceBuffer[DType.float32],
    mut dgw: DeviceBuffer[DType.float32],
    mut dpi: DeviceBuffer[DType.float32],
    mut dwsr: DeviceBuffer[DType.float32],
    n: Int,
    max_iter_predict: Int,
) raises -> GPCNewtonRun:
    """The Laplace Newton loop against a resident K (`dk`, n x n) and
    resident targets (`dy`): `gpc_newton_device_batch` with one class (one
    loop for the binary and the one-vs-rest fits). On return `db` holds the
    last factor, `dpi` and `dwsr` the last pi and W_sr. `dgw` holds
    `identical_gemm_workspace_max_floats(n, 1, n)` floats. Raises by name
    when B's factor fails or a right-hand side is NaN."""
    var dys: List[DeviceBuffer[DType.float32]] = [dy.copy()]
    var dbs: List[DeviceBuffer[DType.float32]] = [db.copy()]
    var dpis: List[DeviceBuffer[DType.float32]] = [dpi.copy()]
    var dwsrs: List[DeviceBuffer[DType.float32]] = [dwsr.copy()]
    var runs = gpc_newton_device_batch(ctx, dk, dys, dbs, dgw, dpis, dwsrs, n, max_iter_predict)
    return runs[0]
