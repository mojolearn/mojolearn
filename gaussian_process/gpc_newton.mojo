# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The Laplace Newton loop of Gaussian process classification as a step of
the executor's bounded device loop (lane n8-gp-krr-smo, 2026-11-09;
docs/plans/EXECUTOR_CONSOLIDATION_AND_CLEANUP.md section 4, row "GP
regressor / GPC Laplace").

ONE LOOP. The binary fit (`classifier.mojo::gpc_fit_binary_host`) and the
one-vs-rest fit (`gpc_ovr.mojo::gpc_fit_all_device`) each kept their own
copy of the same loop; both now call `gpc_newton_device` here.

WHAT MOVED. Each iteration read three things home: the Cholesky's `info`
(inside `potrf_lower`), log|B| (`chol_logdet`, unless
GPC_IDN_NEWTON_LOGDET), and the likelihood with the NaN guard's index for
the stop test. Now the iteration is `GPCNewtonStep.enqueue_step`: the same
item kernels in the same order, each gated on the loop's done word; the
factor, log-determinant and solve are the executor's Cholesky ops
(cholesky/chol_ops.mojo) whose `info` joins the loop's predicate; the NaN
guard and DEVIATION 2830's stop test are one criterion kernel
(`gpc_newton_criterion_kernel`) that also keeps the likelihood the fit
returns on the device. `bounded_device_loop` reads the register block once
per GPC_NEWTON_CHECK iterations; one more read after the loop brings the
likelihood home.

BITS. The same items, the same block sizes (256, the classifier's
GPC_STEP_TPB and GPC_B_TPB), the same identical GEMVs, and the ops' factor,
solve and log-determinant fold are `potrf_lower`'s / `cho_solve`'s /
`enqueue_logdet`'s words (docs/plans/EXECUTOR_N7_OPS.md). The `+0.0` ridge
launch (`add_jitter`) the loop ran on B is gone: B's diagonal is already
`ftz(1 + cell)`, which is never -0.0, and `ftz(d + 0.0) == d` for every
other float32, so it changed no bit. Iterations enqueued past the stop leave
f, pi, W_sr and the factor as the stopping iteration left them: every
kernel that writes them is gated, and the two GEMVs re-form f and K b from
frozen inputs (the same words).
"""

from std.atomic import Atomic
from std.gpu import block_idx, thread_idx
from max.gpu.host import DeviceBuffer, DeviceContext

from autograd.program_loop import LOOP_WORDS, LoopTimers
from autograd.program_loop_device import LoopRegisters, LoopStep, bounded_device_loop
from cholesky.checks.potrf import CHOL_NB_PINNED, chol_nb_for
from cholesky.chol_ops import (
    chol_op_begin,
    chol_op_cho_solve,
    chol_op_factor,
    chol_op_info_joins_loop,
    chol_op_logdet,
    chol_op_logdet_scratch_floats,
)
from checks.numerics import ftz, identical_mul
from core.device_zero import enqueue_fill
from core.step_gate import STEP_DONE, STEP_FAIL, STEP_ITER, STEP_STOP_ITER
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
#: Newton iterations enqueued between two reads of the register block. An
#: iteration past the stop costs two GEMVs and a few one-word launches (the
#: factor and the solve are gated off), so eight bounds the waste below one
#: iteration's factor at every n. A size-free rule.
comptime GPC_NEWTON_CHECK = 8
#: The failure code of a NaN right-hand side (DEVIATION 1638). A Cholesky
#: failure's code is its `info`, 1 .. n, and n never reaches this value.
comptime GPC_NEWTON_FAIL_NAN = Int32(0x7FFF0000)


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


@always_inline
def _open(gate: _GI) -> Bool:
    return gate.unsafe_load(STEP_DONE) == Int32(0)


def gpc_newton_w_kernel(gate: _GI, f: _GP, pi: _GP, w: _GP, wsr: _GP, n: Int32):
    """`gpc_weight_item` per row while the loop runs."""
    var i = _tid()
    if i < Int(n) and _open(gate):
        gpc_weight_item(i, f, pi, w, wsr)


def gpc_newton_b_kernel(gate: _GI, b: _GP, k: _GP, wsr: _GP, n_in: Int32):
    """B = I + W_sr K W_sr per cell (`gpc_b_matrix`'s arithmetic: the
    weights multiplied first, each product `identical_mul` and flushed, the
    diagonal's `1 + cell`) while the loop runs."""
    var n = Int(n_in)
    var e = Int(block_idx.x) * GPC_NEWTON_TPB + Int(thread_idx.x)
    if e >= n * n or not _open(gate):
        return
    var i = e // n
    var j = e - i * n
    var ww = ftz(identical_mul(ftz(wsr.unsafe_load(i)), ftz(wsr.unsafe_load(j))))
    var cell = ftz(identical_mul(ww, ftz(k.unsafe_load(e))))
    if i == j:
        cell = ftz(Float32(1.0) + cell)
    b.unsafe_store(e, cell)


def gpc_newton_rhs_kernel(gate: _GI, w: _GP, f: _GP, y: _GP, pi: _GP, dst: _GP, n: Int32):
    var i = _tid()
    if i < Int(n) and _open(gate):
        gpc_rhs_item(i, w, f, y, pi, dst)


def gpc_newton_nan_reset_kernel(gate: _GI, nan_at: _GI, n: Int32):
    """One thread: the NaN guard's index back to `n` (none) while the loop
    runs."""
    if _tid() == 0 and _open(gate):
        nan_at.unsafe_store(0, n)


def gpc_newton_scale_nan_kernel(gate: _GI, wsr: _GP, v: _GP, dst: _GP, nan_at: _GI, n: Int32):
    """`gpc_scale`, and the lowest index whose value is NaN (DEVIATION 1638's
    refusal of a NaN right-hand side) by an atomic min, while the loop runs."""
    var i = _tid()
    if i < Int(n) and _open(gate):
        gpc_scale_item(i, wsr, v, dst)
        var c = dst.unsafe_load(i)
        if c != c:
            _ = Atomic[DType.int32].min(nan_at, Int32(i))


def gpc_newton_a_kernel(gate: _GI, b: _GP, wsr: _GP, x: _GP, dst: _GP, n: Int32):
    var i = _tid()
    if i < Int(n) and _open(gate):
        gpc_a_item(i, b, wsr, x, dst)


def gpc_newton_lml_part_kernel(gate: _GI, a: _GP, f: _GP, y: _GP, n: Int32, pdot: _GP, pt2: _GP, nb: Int32):
    var blk = _tid()
    if blk < Int(nb) and _open(gate):
        gpc_lml_part_item(blk, a, f, y, Int(n), pdot, pt2)


def gpc_newton_lml_fin_kernel(gate: _GI, pdot: _GP, pt2: _GP, nb: Int32, logdet: _GP, dst: _GP):
    """The likelihood with log|B| read from the device scalar the
    log-determinant op wrote."""
    if _tid() == 0 and _open(gate):
        dst.unsafe_store(0, gpc_lml_fin(pdot, pt2, Int(nb), logdet.unsafe_load(0)))


def gpc_newton_criterion_kernel(words: _GI, lml: _GP, kept: _GP, nan_at: _GI, n: Int32, it: Int32):
    """One thread, after iteration `it` (1-based). While the loop runs: the
    iteration register; a NaN right-hand side stops the loop with
    GPC_NEWTON_FAIL_NAN; DEVIATION 2830's test (`gpc_stop(lml, kept)`)
    stops it with `kept` unchanged; otherwise `kept = lml`. The fit returns
    `kept` (scikit-learn's `log_marginal_likelihood` before the stopping
    iteration)."""
    if _tid() != 0:
        return
    if words.unsafe_load(STEP_DONE) != Int32(0):
        return
    words.unsafe_store(STEP_ITER, it)
    if nan_at.unsafe_load(0) < n:
        words.unsafe_store(STEP_FAIL, GPC_NEWTON_FAIL_NAN)
        words.unsafe_store(STEP_STOP_ITER, it)
        words.unsafe_store(STEP_DONE, Int32(1))
        return
    var l = lml.unsafe_load(0)
    if gpc_stop(l, kept.unsafe_load(0)):
        words.unsafe_store(STEP_STOP_ITER, it)
        words.unsafe_store(STEP_DONE, Int32(1))
        return
    kept.unsafe_store(0, l)


struct GPCNewtonStep(Copyable, LoopStep, Movable):
    """One Laplace Newton iteration (`_gpc.py:_posterior_mode`) as a step of
    `bounded_device_loop`. The fields are handle copies of the fit's
    buffers (`DeviceBuffer.copy()` shares the memory)."""
    var dk: DeviceBuffer[DType.float32]
    var dy: DeviceBuffer[DType.float32]
    var db: DeviceBuffer[DType.float32]
    var dgw: DeviceBuffer[DType.float32]
    var dpi: DeviceBuffer[DType.float32]
    var dwsr: DeviceBuffer[DType.float32]
    var df: DeviceBuffer[DType.float32]
    var dw: DeviceBuffer[DType.float32]
    var dbv: DeviceBuffer[DType.float32]
    var dkb: DeviceBuffer[DType.float32]
    var dc: DeviceBuffer[DType.float32]
    var da: DeviceBuffer[DType.float32]
    var dpd: DeviceBuffer[DType.float32]
    var dpt: DeviceBuffer[DType.float32]
    var dlml: DeviceBuffer[DType.float32]
    var dkept: DeviceBuffer[DType.float32]
    var dnan: DeviceBuffer[DType.int32]
    var dinfo: DeviceBuffer[DType.int32]
    var dldscr: DeviceBuffer[DType.float32]
    var dldet: DeviceBuffer[DType.float32]
    var n: Int
    var nbf: Int

    def __init__(
        out self,
        ctx: DeviceContext,
        dk: DeviceBuffer[DType.float32],
        dy: DeviceBuffer[DType.float32],
        db: DeviceBuffer[DType.float32],
        dgw: DeviceBuffer[DType.float32],
        dpi: DeviceBuffer[DType.float32],
        dwsr: DeviceBuffer[DType.float32],
        n: Int,
    ) raises:
        self.dk = dk.copy()
        self.dy = dy.copy()
        self.db = db.copy()
        self.dgw = dgw.copy()
        self.dpi = dpi.copy()
        self.dwsr = dwsr.copy()
        self.n = n
        self.nbf = gpc_fold_blocks(n)
        self.df = ctx.enqueue_create_buffer[DType.float32](n)
        self.dw = ctx.enqueue_create_buffer[DType.float32](n)
        self.dbv = ctx.enqueue_create_buffer[DType.float32](n)
        self.dkb = ctx.enqueue_create_buffer[DType.float32](n)
        self.dc = ctx.enqueue_create_buffer[DType.float32](n)
        self.da = ctx.enqueue_create_buffer[DType.float32](n)
        self.dpd = ctx.enqueue_create_buffer[DType.float32](max(self.nbf, 1))
        self.dpt = ctx.enqueue_create_buffer[DType.float32](max(self.nbf, 1))
        self.dlml = ctx.enqueue_create_buffer[DType.float32](1)
        self.dkept = ctx.enqueue_create_buffer[DType.float32](1)
        self.dnan = ctx.enqueue_create_buffer[DType.int32](1)
        self.dinfo = ctx.enqueue_create_buffer[DType.int32](1)
        self.dldscr = ctx.enqueue_create_buffer[DType.float32](chol_op_logdet_scratch_floats(n))
        self.dldet = ctx.enqueue_create_buffer[DType.float32](1)
        enqueue_fill(ctx, self.df, Float32(0.0))
        enqueue_fill(ctx, self.dkept, gpc_neg_inf32())
        enqueue_fill(ctx, self.dnan, Int32(n))

    def enqueue_step(
        mut self, ctx: DeviceContext, mut registers: LoopRegisters, it: Int, mut timers: LoopTimers,
    ) raises:
        self.enqueue_gated(ctx, registers.words, it)

    def enqueue_gated(mut self, ctx: DeviceContext, mut words: DeviceBuffer[DType.int32], it: Int) raises:
        """Iteration `it` gated on the register block `words` (the loop's
        own, or one class's block inside `GPCNewtonBatch`)."""
        var n = self.n
        var g = _ip(words)
        ctx.enqueue_function[gpc_newton_w_kernel](
            g, _fp(self.df), _fp(self.dpi), _fp(self.dw), _fp(self.dwsr), Int32(n),
            grid_dim=_blocks(n), block_dim=GPC_NEWTON_TPB,
        )
        ctx.enqueue_function[gpc_newton_b_kernel](
            g, _fp(self.db), _fp(self.dk), _fp(self.dwsr), Int32(n),
            grid_dim=_blocks(n * n), block_dim=GPC_NEWTON_TPB,
        )
        chol_op_begin(ctx, words, self.dinfo)
        chol_op_factor(ctx, self.db, self.dinfo, n)
        chol_op_info_joins_loop(ctx, words, self.dinfo, it)
        chol_op_logdet(ctx, self.db, self.dinfo, self.dldscr, self.dldet, n)
        ctx.enqueue_function[gpc_newton_rhs_kernel](
            g, _fp(self.dw), _fp(self.df), _fp(self.dy), _fp(self.dpi), _fp(self.dbv), Int32(n),
            grid_dim=_blocks(n), block_dim=GPC_NEWTON_TPB,
        )
        # K b from frozen inputs once the loop is done: the same words
        identical_gemm_into(ctx, self.dkb, self.dk, self.dbv, self.dgw, n, 1, n, OP_TN)
        ctx.enqueue_function[gpc_newton_nan_reset_kernel](  # small-launch(n: the sentinel value stored, not a bound): one thread stores one word, the NaN guard's index
            g, _ip(self.dnan), Int32(n), grid_dim=1, block_dim=1,
        )
        ctx.enqueue_function[gpc_newton_scale_nan_kernel](
            g, _fp(self.dwsr), _fp(self.dkb), _fp(self.dc), _ip(self.dnan), Int32(n),
            grid_dim=_blocks(n), block_dim=GPC_NEWTON_TPB,
        )
        chol_op_cho_solve(ctx, self.db, self.dc, self.dinfo, n, 1)
        ctx.enqueue_function[gpc_newton_a_kernel](
            g, _fp(self.dbv), _fp(self.dwsr), _fp(self.dc), _fp(self.da), Int32(n),
            grid_dim=_blocks(n), block_dim=GPC_NEWTON_TPB,
        )
        # f = K a, from a frozen `a` once the loop is done: the same words
        identical_gemm_into(ctx, self.df, self.dk, self.da, self.dgw, n, 1, n, OP_TN)
        if self.nbf > 0:
            ctx.enqueue_function[gpc_newton_lml_part_kernel](
                g, _fp(self.da), _fp(self.df), _fp(self.dy), Int32(n), _fp(self.dpd), _fp(self.dpt), Int32(self.nbf),
                grid_dim=_blocks(self.nbf), block_dim=GPC_NEWTON_TPB,
            )
        ctx.enqueue_function[gpc_newton_lml_fin_kernel](  # small-launch(nbf: fold blocks of the likelihood): one thread combines the partials into one scalar
            g, _fp(self.dpd), _fp(self.dpt), Int32(self.nbf), _fp(self.dldet), _fp(self.dlml),
            grid_dim=1, block_dim=1,
        )
        ctx.enqueue_function[gpc_newton_criterion_kernel](  # small-launch(n: the sentinel compared, not a bound): one thread runs the loop criterion over three words
            g, _fp(self.dlml), _fp(self.dkept), _ip(self.dnan), Int32(n), Int32(it),
            grid_dim=1, block_dim=1,
        )


@fieldwise_init
struct GPCNewtonRun(ImplicitlyCopyable, Movable):
    """What the Newton loop leaves besides the resident factor, pi and W_sr:
    the likelihood kept (DEVIATION 2830), the iterations and the Cholesky
    panel width that ran."""

    var lml: Float32
    var n_iter: Int
    var nb: Int


def gpc_newton_join_kernel(outer: _GI, blocks: MutPointer[Int64, MutAnyOrigin], k: Int32, it: Int32):
    """One thread, after iteration `it` of every class: the batch loop's
    register block is done (stopping at `it`) once every class's block is
    done (converged or failed). Once done, it changes nothing."""
    if _tid() != 0:
        return
    if outer.unsafe_load(STEP_DONE) != Int32(0):
        return
    outer.unsafe_store(STEP_ITER, it)
    for c in range(Int(k)):
        var w = _GI(unsafe_from_address=Int(blocks.unsafe_load(c)))
        if w.unsafe_load(STEP_DONE) == Int32(0):
            return
    outer.unsafe_store(STEP_STOP_ITER, it)
    outer.unsafe_store(STEP_DONE, Int32(1))


struct GPCNewtonBatch(LoopStep, Movable):
    """Several binary Laplace fits (one-vs-rest classes) as ONE step of
    `bounded_device_loop` (lane m3-linalg-and-housekeeping, 2026-11-09).
    Each class keeps its own register block, so it stops (or fails) on its
    own iteration exactly as alone; the batch's block is done once every
    class's is. Every class's kernels are the single fit's, in the single
    fit's order, on the same stream: the same words. What changes is the
    host side: one read of the register blocks per GPC_NEWTON_CHECK
    iterations for all classes, where the classes ran one loop (and its
    reads) after another."""
    var steps: List[GPCNewtonStep]
    var words: List[DeviceBuffer[DType.int32]]
    var blocks: DeviceBuffer[DType.int64]
    var k: Int

    def __init__(out self, ctx: DeviceContext, var steps: List[GPCNewtonStep]) raises:
        var k = len(steps)
        var words = List[DeviceBuffer[DType.int32]](capacity=k)
        var addrs = List[Int64](length=k, fill=Int64(0))
        for c in range(k):  # driver-loop: over the batch's classes (model structure)
            var w = ctx.enqueue_create_buffer[DType.int32](LOOP_WORDS)
            ctx.enqueue_memset(w, Int32(0))
            addrs[c] = Int64(Int(w.unsafe_ptr()))
            words.append(w^)
        var blocks = ctx.enqueue_create_buffer[DType.int64](max(k, 1))
        ctx.enqueue_copy(
            dst_buf=blocks, src_ptr=MutPointer[Int64, MutAnyOrigin](unsafe_from_address=Int(addrs.unsafe_ptr()))
        )
        ctx.synchronize()
        _ = addrs^
        self.k = k
        self.steps = steps^
        self.words = words^
        self.blocks = blocks^

    def enqueue_step(
        mut self, ctx: DeviceContext, mut registers: LoopRegisters, it: Int, mut timers: LoopTimers,
    ) raises:
        for c in range(self.k):  # driver-loop: over the batch's classes (model structure)
            var w = self.words[c].copy()  # a handle copy: the same device words
            self.steps[c].enqueue_gated(ctx, w, it)
        ctx.enqueue_function[gpc_newton_join_kernel](  # small-launch(k: classes): one thread joins the classes' done words
            _ip(registers.words), MutPointer[Int64, MutAnyOrigin](unsafe_from_address=Int(self.blocks.unsafe_ptr())),
            Int32(self.k), Int32(it), grid_dim=1, block_dim=1,
        )


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
    K (`dk`, n x n) in one bounded device loop (`GPCNewtonBatch`). Class c
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
    var steps = List[GPCNewtonStep](capacity=k)
    for c in range(k):  # driver-loop: over the classes (model structure)
        steps.append(GPCNewtonStep(ctx, dk, dys[c], dbs[c], dgw, dpis[c], dwsrs[c], n))
    var batch = GPCNewtonBatch(ctx, steps^)
    var registers = LoopRegisters(ctx)
    var timers = LoopTimers("gpc.newton")
    _ = bounded_device_loop(ctx, batch, registers, 1, max_iter_predict, GPC_NEWTON_CHECK, timers)
    timers.report()
    # one read: every class's block, likelihood kept and NaN index
    var hw = ctx.enqueue_create_host_buffer[DType.int32](k * LOOP_WORDS)
    var hkept = ctx.enqueue_create_host_buffer[DType.float32](k)
    var hnan = ctx.enqueue_create_host_buffer[DType.int32](k)
    for c in range(k):  # driver-loop: over the classes (model structure)
        ctx.enqueue_copy(dst_ptr=hw.unsafe_ptr() + c * LOOP_WORDS, src_buf=batch.words[c])
        ctx.enqueue_copy(dst_ptr=hkept.unsafe_ptr() + c, src_buf=batch.steps[c].dkept)
        ctx.enqueue_copy(dst_ptr=hnan.unsafe_ptr() + c, src_buf=batch.steps[c].dnan)
    ctx.synchronize()
    for c in range(k):  # driver-loop: over the classes (model structure)
        var w = hw.unsafe_ptr() + c * LOOP_WORDS
        var code = Int(w.unsafe_load(STEP_FAIL))
        var done = Int(w.unsafe_load(STEP_DONE)) != 0
        var stop = Int(w.unsafe_load(STEP_STOP_ITER))
        if code == Int(GPC_NEWTON_FAIL_NAN):
            raise Error(
                "cholesky_solve_host: the right-hand side contains NaN at"
                " flat index "
                + String(Int(hnan.unsafe_ptr().unsafe_load(c)))
                + "; refused by name (DEVIATION 1638)"
            )
        if code != 0:
            raise Error(
                "gpc_fit_host: the factorization of B = I + W_sr K W_sr failed"
                " at Newton iteration "
                + String(stop)
                + " (info="
                + String(code)
                + "). B's eigenvalues are at least 1 for any finite kernel"
                " matrix, so this means a non-finite latent value"
            )
        var n_iter = stop if done else max(max_iter_predict, 0)
        var nb = chol_nb_for(n, CHOL_NB_PINNED) if n_iter > 0 else 0
        runs.append(GPCNewtonRun(hkept.unsafe_ptr().unsafe_load(c), n_iter, nb))
    _ = hw^
    _ = hkept^
    _ = hnan^
    _ = batch^
    _ = registers^
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
