# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""x_decomp/nmf.mojo's NMF solver loops on resident device matrices (lane
py-runtime-b): the same statements on `DKit` (x_decomp/kit_device.mojo),
the same cells on the same values, so the same words as the host column.
GPU binding only.

Lane n4-mds-labelprop-nmf-loops (2026-11-09): both solvers run as the
executor's bounded device loop (autograd/program_loop_device.mojo). W and H
(Ht for CD) are the loop's carried state, written in place only through
`DKit.put_gated`; the stopping test is a one-thread criterion kernel over the
summed words on the device (`nmf_err_f32`, `nmf_mu_stop_f32`,
`nmf_cd_stop_f32` of x_decomp/nmf.mojo, the host column's functions). The host
reads the loop's register block once per chunk: MU every 10 iterations (its
criterion period, so no iteration runs past the stop), CD every
NMF_CD_LOOP_CHUNK. Before, MU read 1-3 error words home per 10 iterations and
CD 1-2 violation words per iteration."""
# TOMBSTONE: nmf-mu-cd-host-word-loops (REFACTORED onto autograd/program_loop_device.mojo) deleted 2026-11-09 by lane/n4-mds-labelprop-nmf-loops; code recoverable at f59d913b0b.
# Restore: git apply -3 experiments/removed/N4_MDS_LABELPROP_NMF_HOST_LOOPS.patch; record in docs/TOMBSTONES.md.
from experiments.classical_identical_ideas.linear_controls import C26_PRODUCTS, C26_UPDATE_FUSED
from max.gpu import block_idx, thread_idx
from max.gpu.host import DeviceContext
from autograd.program_loop import LoopTimers
from autograd.program_loop_device import LoopRegisters, LoopStep, bounded_device_loop, loop_lap
from core.step_gate import STEP_DONE, STEP_FAIL, STEP_ITER, STEP_STOP_ITER
from x_decomp.cells import F32Ptr, I32Ptr, OP_CLASSICAL_MU_IS, add
from x_decomp.device import xd_ctx
from x_decomp.kit import (
    Mat, OP_ABS, OP_ADD, OP_ADDS, OP_AXPY, OP_DIV, OP_GTS, OP_LOGS, OP_MAXS, OP_MUL, OP_MUZ, OP_RECIP,
    OP_MINS, OP_SCALE, OP_SELECT, OP_SQ, OP_SQDIFF, OP_SQRT, mat_const, mat_eye, mat_from,
)
from x_decomp.kit_device import DKit, DMat
from x_decomp.nmf import (
    NMF_ERR_FRO, NMF_ERR_KL, NMF_F32_EPS, NMF_LOG_FLOOR, NMF_ZERO_ERR0, NmfArgs, _f32_prod, _nrm_or0,
    nmf_cd_stop_f32, nmf_err_f32, nmf_err_kind, nmf_mu_stop_f32,
)

#: CD's iterations enqueued between two reads of the loop's register block.
#: The stop is tested every iteration on the device; up to this many minus
#: one sweeps run gated past it (their state writes are no-ops). A cost
#: rule, not a shape: one drain per eight sweeps against at most seven
#: wasted sweeps (KMeans' Lloyd loop reads every 8 as well).
comptime NMF_CD_LOOP_CHUNK = 8
#: MU tests its error every 10 iterations (`_mu`'s period).
comptime NMF_MU_CHECK_PERIOD = 10
#: The failure code the MU criterion stores when the first error is 0.
comptime NMF_FAIL_ZERO_ERR0 = 1


def _view(D: DMat) -> DMat:
    """A non-owning view of all of D (the loop writes D's own buffer)."""
    return DMat(rows_of=D, row0=0, rows=D.r)


def _gate(mut registers: LoopRegisters) -> I32Ptr:
    return registers.words.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]()


def _err_words_dev(k: DKit, M: DMat, W: DMat, H: DMat, kind: Int) raises -> List[DMat]:
    """`NMF._err`'s summed words, on the device, in `nmf_err_f32`'s order."""
    var out = List[DMat]()
    if kind == NMF_ERR_FRO:
        out.append(k.ew1(OP_SQRT, k.total(k.ew2(OP_SQDIFF, M, k.mm(W, H, False, False))), 0.0))
        return out^
    var WH = k.ew1(OP_MAXS, k.mm(W, H, False, False), NMF_F32_EPS)
    var keep = k.ew1(OP_GTS, M, NMF_F32_EPS)
    var div = k.ew2(OP_DIV, M, WH)
    if kind == NMF_ERR_KL:
        out.append(k.total(k.ew2(OP_MUL, k.ew2(OP_MUL, M, k.ew1(OP_LOGS, div, NMF_LOG_FLOOR)), keep)))
        out.append(k.mm(k.colsum(W), k.rowsum(H), False, False))
        out.append(k.total(k.ew2(OP_MUL, M, keep)))
        return out^
    out.append(k.total(k.ew2(OP_MUL, div, keep)))
    out.append(k.total(k.ew2(OP_MUL, k.ew1(OP_LOGS, div, NMF_LOG_FLOOR), keep)))
    return out^


def nmf_mu_crit_kernel(
    words: I32Ptr, a: F32Ptr, b: F32Ptr, c: F32Ptr, state: F32Ptr, kind: Int32, mc: Float32, tol: Float32,
    it: Int32,
):
    """One thread. it == 0 (before the loop): the first error into state[0]
    (err0) and state[1] (prev). Else, while the loop runs: a zero err0 stops
    it with NMF_FAIL_ZERO_ERR0 (Python's ZeroDivisionError), `nmf_mu_stop_f32`
    stops it, otherwise prev = err."""
    if Int(block_idx.x) != 0 or Int(thread_idx.x) != 0:
        return
    var err = nmf_err_f32(Int(kind), a.unsafe_load(0), b.unsafe_load(0), c.unsafe_load(0), mc)
    if it == Int32(0):
        state.unsafe_store(0, err)
        state.unsafe_store(1, err)
        return
    if words.unsafe_load(STEP_DONE) != Int32(0):
        return
    words.unsafe_store(STEP_ITER, it)
    var err0 = state.unsafe_load(0)
    if err0 == Float32(0):
        words.unsafe_store(STEP_FAIL, Int32(NMF_FAIL_ZERO_ERR0))
        words.unsafe_store(STEP_STOP_ITER, it)
        words.unsafe_store(STEP_DONE, Int32(1))
        return
    if nmf_mu_stop_f32(state.unsafe_load(1), err, err0, tol):
        words.unsafe_store(STEP_STOP_ITER, it)
        words.unsafe_store(STEP_DONE, Int32(1))
        return
    state.unsafe_store(1, err)


def nmf_cd_crit_kernel(words: I32Ptr, v1: F32Ptr, v2: F32Ptr, two: Int32, state: F32Ptr, tol: Float32, it: Int32):
    """One thread, while the loop runs: the sweep's violation (both halves
    summed when `two`), the first one kept in state[0], and `nmf_cd_stop_f32`."""
    if Int(block_idx.x) != 0 or Int(thread_idx.x) != 0:
        return
    if words.unsafe_load(STEP_DONE) != Int32(0):
        return
    words.unsafe_store(STEP_ITER, it)
    var v = v1.unsafe_load(0)
    if two != Int32(0):
        v = add(v, v2.unsafe_load(0))
    if it == Int32(1):
        state.unsafe_store(0, v)
    if nmf_cd_stop_f32(v, state.unsafe_load(0), tol):
        words.unsafe_store(STEP_STOP_ITER, it)
        words.unsafe_store(STEP_DONE, Int32(1))


def _mu_ratio_dev(k: DKit, M: DMat, W: DMat, H: DMat, beta: Float64, mut P: DMat) raises -> DMat:
    """`NMF._mu_ratio`: X / WH (KL) or X / WH^2 (IS) with WH clamped at
    EPSILON; P = WH^-1 for IS."""
    var WH = k.ew1(OP_MAXS, k.mm(W, H, False, False), NMF_F32_EPS)
    if beta == 1.0:
        return k.ew2(OP_DIV, M, WH)
    P = k.ew1(OP_RECIP, WH, 0.0)
    return k.ew2(OP_DIV, k.ew2(OP_DIV, M, WH), WH)


struct _MuStep(LoopStep, Movable):
    """One `_mu` / `_mu_beta` iteration as a step of the bounded device loop:
    W and H (views of the caller's matrices) written through `put_gated`,
    then every NMF_MU_CHECK_PERIOD iterations (tol > 0) the error words and
    the criterion kernel."""
    var k: DKit
    var M: DMat
    var W: DMat
    var H: DMat
    var a: NmfArgs
    var kind: Int
    var eps: DMat
    var one: DMat
    var frozen_num: DMat
    var frozen_gram: DMat
    var state: DMat

    def __init__(out self, M: DMat, W: DMat, H: DMat, a: NmfArgs) raises:
        var k = DKit()
        var eps = k.upload(mat_const(NMF_F32_EPS, 1, 1))
        var one = k.upload(mat_const(1.0, 1, 1))
        var frozen_num = DMat(0, 0)
        var frozen_gram = DMat(0, 0)
        comptime if C26_PRODUCTS:
            if not a.update_h and a.beta == 2.0:
                frozen_num = k.mm(M, H, False, True)
                frozen_gram = k.mm(H, H, False, True)
        self.k = k^
        self.M = _view(M)
        self.W = _view(W)
        self.H = _view(H)
        self.a = a.copy()
        self.kind = nmf_err_kind(a.beta)
        self.eps = eps^
        self.one = one^
        self.frozen_num = frozen_num^
        self.frozen_gram = frozen_gram^
        self.state = DMat(1, 2)

    def criterion(self, mut registers: LoopRegisters, it: Int) raises:
        """The error words of the current W, H and the criterion kernel
        (it == 0: err0, before the loop)."""
        var w = _err_words_dev(self.k, self.M, self.W, self.H, self.kind)
        var pa = w[0].p()
        var pb = pa
        var pc = pa
        if len(w) > 1:
            pb = w[1].p()
        if len(w) > 2:
            pc = w[2].p()
        self.k.ctx.enqueue_function[nmf_mu_crit_kernel](
            _gate(registers), pa, pb, pc, self.state.p(), Int32(self.kind), Float32(self.M.r * self.M.c),
            Float32(self.a.tol), Int32(it), grid_dim=1, block_dim=1,
        )
        _ = w^

    def enqueue_step(
        mut self, ctx: DeviceContext, mut registers: LoopRegisters, it: Int, mut timers: LoopTimers,
    ) raises:
        var gate = _gate(registers)
        var beta = self.a.beta
        if beta == 2.0:
            var num: DMat
            var den: DMat
            if C26_PRODUCTS and not self.a.update_h:
                num = _view(self.frozen_num)
                den = self.k.mm(self.W, self.frozen_gram, False, False)
            else:
                num = self.k.mm(self.M, self.H, False, True)
                den = self.k.mm(self.W, self.k.mm(self.H, self.H, False, True), False, False)
            if self.a.l1w > 0:
                den = self.k.ew1(OP_ADDS, den, self.a.l1w)
            if self.a.l2w > 0:
                den = self.k.ew2s(OP_AXPY, den, self.W, self.a.l2w)
            self.k.put_gated(gate, self.W, self.k.ew3(OP_MUZ, self.W, num, den, NMF_F32_EPS))
            if self.a.update_h:
                num = self.k.mm(self.W, self.M, True, False)
                den = self.k.mm(self.k.mm(self.W, self.W, True, False), self.H, False, False)
                if self.a.l1h > 0:
                    den = self.k.ew1(OP_ADDS, den, self.a.l1h)
                if self.a.l2h > 0:
                    den = self.k.ew2s(OP_AXPY, den, self.H, self.a.l2h)
                self.k.put_gated(gate, self.H, self.k.ew3(OP_MUZ, self.H, num, den, NMF_F32_EPS))
        else:
            var P = DMat(0, 0)
            var R = _mu_ratio_dev(self.k, self.M, self.W, self.H, beta, P)
            var num = self.k.mm(R, self.H, False, True)
            var den: DMat
            if beta == 1.0:
                den = self.k.ew2(OP_ADD, self.k.zeros(self.W.r, self.W.c), self.k.vec_t(self.k.rowsum(self.H)))
            else:
                den = self.k.mm(P, self.H, False, True)
            if self.a.l1w > 0:
                den = self.k.ew1(OP_ADDS, den, self.a.l1w)
            if self.a.l2w > 0:
                den = self.k.ew2s(OP_AXPY, den, self.W, self.a.l2w)
            comptime if C26_UPDATE_FUSED:
                self.k.put_gated(
                    gate, self.W, self.k.ew3(OP_CLASSICAL_MU_IS if beta == 0.0 else OP_MUZ, self.W, num, den, NMF_F32_EPS)
                )
            else:
                den = self.k.ew3(OP_SELECT, self.k.ew1(OP_ABS, den, 0.0), den, self.eps, 0.0)
                var delta = self.k.ew2(OP_DIV, num, den)
                if beta == 0.0:
                    delta = self.k.ew1(OP_SQRT, delta, 0.0)
                self.k.put_gated(gate, self.W, self.k.ew2(OP_MUL, self.W, delta))
            if self.a.update_h:
                R = _mu_ratio_dev(self.k, self.M, self.W, self.H, beta, P)
                num = self.k.mm(self.W, R, True, False)
                if beta == 1.0:
                    var ws = self.k.colsum(self.W)
                    ws = self.k.ew3(OP_SELECT, self.k.ew1(OP_ABS, ws, 0.0), ws, self.one, 0.0)
                    den = self.k.ew2(OP_ADD, self.k.zeros(self.H.r, self.H.c), self.k.vec_t(ws^))
                else:
                    den = self.k.mm(self.W, P, True, False)
                if self.a.l1h > 0:
                    den = self.k.ew1(OP_ADDS, den, self.a.l1h)
                if self.a.l2h > 0:
                    den = self.k.ew2s(OP_AXPY, den, self.H, self.a.l2h)
                comptime if C26_UPDATE_FUSED:
                    self.k.put_gated(
                        gate, self.H,
                        self.k.ew3(OP_CLASSICAL_MU_IS if beta == 0.0 else OP_MUZ, self.H, num, den, NMF_F32_EPS),
                    )
                else:
                    den = self.k.ew3(OP_SELECT, self.k.ew1(OP_ABS, den, 0.0), den, self.eps, 0.0)
                    var delta = self.k.ew2(OP_DIV, num, den)
                    if beta == 0.0:
                        delta = self.k.ew1(OP_SQRT, delta, 0.0)
                    self.k.put_gated(gate, self.H, self.k.ew2(OP_MUL, self.H, delta))
        loop_lap(ctx, timers, "mu_update")
        if self.a.tol > 0 and it % NMF_MU_CHECK_PERIOD == 0:
            self.criterion(registers, it)
            loop_lap(ctx, timers, "criterion")


def nmf_mu_dev(M: DMat, mut W: DMat, mut H: DMat, a: NmfArgs) raises -> Int:
    """`NMF._mu` (and `_mu_beta` for beta 1 / 0): W and H updated in place.
    Returns the iteration count, or NMF_ZERO_ERR0."""
    var ctx = xd_ctx()
    var registers = LoopRegisters(ctx)
    var step = _MuStep(M, W, H, a)
    if a.tol > 0:
        step.criterion(registers, 0)
    # tol <= 0 has no criterion: the whole bound is one chunk, one read
    var check = NMF_MU_CHECK_PERIOD if a.tol > 0 else max(a.max_iter, 1)
    var timers = LoopTimers("nmf.mu")
    var outcome = bounded_device_loop(ctx, step, registers, 1, a.max_iter, check, timers)
    timers.report()
    step.k.sync()  # the step kit's held uploads, before it goes
    if outcome.converged:
        if registers.failure() == NMF_FAIL_ZERO_ERR0:
            return NMF_ZERO_ERR0
        return outcome.stop_iteration
    return max(a.max_iter, 0)


def _cd_sweep_dev(k: DKit, gate: I32Ptr, W: DMat, G: DMat, C: DMat, perm: DMat) raises -> DMat:
    """`cd_rows` on a copy of W, the copy written back through `put_gated`;
    the violation word stays on the device."""
    var Wc = k.copy(W)
    var v = k.cd_rows_enq(Wc, G, C, perm)
    k.put_gated(gate, W, Wc)
    return v^


def _cd_side_dev(k: DKit, gate: I32Ptr, M: DMat, W: DMat, Ht: DMat, l1: Float64, l2: Float64,
                 perm: DMat, eye: DMat, trans: Bool) raises -> DMat:
    """`NMF._cd_side`: one half sweep of W against Ht (W's buffer, gated).
    `eye` is the k x k identity, uploaded once per fit when an l2 term is set."""
    var HHt = k.mm(Ht, Ht, True, False)
    var XHt = k.mm(M, Ht, trans, False)
    if l2 != 0.0:
        HHt = k.ew2s(OP_AXPY, HHt, eye, l2)
    if l1 != 0.0:
        XHt = k.ew1(OP_ADDS, XHt, -l1)
    return _cd_sweep_dev(k, gate, W, HHt, XHt, perm)


struct _CdStep(LoopStep, Movable):
    """One `_cd` iteration as a step of the bounded device loop: the W half
    sweep (and with update_H the Ht half sweep), each on a copy written back
    through `put_gated`, then the violation criterion kernel. The coordinate
    order is the identity (uploaded once) or a Philox draw's stable order,
    both on the device; the draw counter advances per enqueued half sweep in
    the host column's order."""
    var k: DKit
    var M: DMat
    var W: DMat
    var Ht: DMat
    var a: NmfArgs
    var kc: Int
    var draws: Int
    var ident: DMat
    var eye: DMat
    var fixed_gram: DMat
    var fixed_cross: DMat
    var state: DMat

    def __init__(out self, M: DMat, W: DMat, Ht: DMat, a: NmfArgs) raises:
        var k = DKit()
        var kc = W.c
        # the per-fit constants, up once
        var ident = k.iota_i32(kc) if not a.shuffle else DMat(0, 0)
        var eye = k.upload(mat_eye(kc)) if (a.l2w != 0.0 or a.l2h != 0.0) else DMat(0, 0)
        var fixed_gram = DMat(0, 0)
        var fixed_cross = DMat(0, 0)
        comptime if C26_PRODUCTS:
            if not a.update_h:
                fixed_gram = k.mm(Ht, Ht, True, False)
                fixed_cross = k.mm(M, Ht, False, False)
                if a.l2w != 0.0:
                    fixed_gram = k.ew2s(OP_AXPY, fixed_gram, eye, a.l2w)
                if a.l1w != 0.0:
                    fixed_cross = k.ew1(OP_ADDS, fixed_cross, -a.l1w)
        self.k = k^
        self.M = _view(M)
        self.W = _view(W)
        self.Ht = _view(Ht)
        self.a = a.copy()
        self.kc = kc
        self.draws = 0
        self.ident = ident^
        self.eye = eye^
        self.fixed_gram = fixed_gram^
        self.fixed_cross = fixed_cross^
        self.state = DMat(1, 1)

    def _perm(mut self) raises -> DMat:
        """`NMF._perm`: a view of the identity, or the next Philox draw's
        stable order (int32 words, on the device)."""
        if not self.a.shuffle:
            return _view(self.ident)
        self.draws += 1
        var r = self.k.rand(1, self.kc, self.a.seed, 200 + self.draws, 0)
        return self.k.order_small_dev(r)

    def enqueue_step(
        mut self, ctx: DeviceContext, mut registers: LoopRegisters, it: Int, mut timers: LoopTimers,
    ) raises:
        var gate = _gate(registers)
        var p1 = self._perm()
        var v1: DMat
        if C26_PRODUCTS and not self.a.update_h:
            v1 = _cd_sweep_dev(self.k, gate, self.W, self.fixed_gram, self.fixed_cross, p1)
        else:
            v1 = _cd_side_dev(self.k, gate, self.M, self.W, self.Ht, self.a.l1w, self.a.l2w, p1, self.eye, False)
        loop_lap(ctx, timers, "cd_w")
        var two = self.a.update_h
        var v2 = DMat(0, 0)
        if two:
            var p2 = self._perm()
            v2 = _cd_side_dev(self.k, gate, self.M, self.Ht, self.W, self.a.l1h, self.a.l2h, p2, self.eye, True)
            loop_lap(ctx, timers, "cd_h")
        var p1v = v1.p()
        var p2v = p1v
        if two:
            p2v = v2.p()
        self.k.ctx.enqueue_function[nmf_cd_crit_kernel](
            gate, p1v, p2v, Int32(1 if two else 0), self.state.p(), Float32(self.a.tol),
            Int32(it), grid_dim=1, block_dim=1,
        )
        loop_lap(ctx, timers, "criterion")


def nmf_cd_dev(M: DMat, mut W: DMat, mut H: DMat, a: NmfArgs) raises -> Int:
    """`NMF._cd`: W and (update_H) H updated. Returns the iteration count."""
    var k = DKit()
    var Ht = k.t(H)
    var ctx = xd_ctx()
    var registers = LoopRegisters(ctx)
    var step = _CdStep(M, W, Ht, a)
    var timers = LoopTimers("nmf.cd")
    var outcome = bounded_device_loop(ctx, step, registers, 1, a.max_iter, NMF_CD_LOOP_CHUNK, timers)
    timers.report()
    step.k.sync()  # the step kit's held uploads, before it goes
    if a.update_h:
        H = k.t(Ht)
    if outcome.converged:
        return outcome.stop_iteration
    return max(a.max_iter, 0)


def nmf_solve_dev_run(
    px: F32Ptr, pw: F32Ptr, ph: F32Ptr, n: Int, d: Int, nc: Int, solver: Int, a: NmfArgs
) raises -> Int:
    """`nmf_solve_run` on the resident kit."""
    if n * d > 2147483647 or n * nc > 2147483647 or nc * d > 2147483647:
        raise Error("x_decomp: nmf shape out of range")
    if solver == 0 and a.beta != 2.0:
        raise Error("x_decomp: nmf cd takes beta 2")
    var it = 0
    var k = DKit()
    var X = k.upload(mat_from(px, n, d))
    var W = k.upload(mat_from(pw, n, nc))
    var H = k.upload(mat_from(ph, nc, d))
    if solver == 0:
        it = nmf_cd_dev(X, W, H, a)
    else:
        it = nmf_mu_dev(X, W, H, a)
    var hw = k.get(W)
    var hh = k.get(H)
    k.sync()
    for i in range(n * nc):
        pw.unsafe_store(i, hw.d[i])
    for i in range(nc * d):
        ph.unsafe_store(i, hh.d[i])
    return it


def nmf_nndsvd_dev_run(
    pu: F32Ptr, ps: F32Ptr, pv: F32Ptr, pw: F32Ptr, ph: F32Ptr, n: Int, d: Int, r: Int, nc: Int
) raises -> Int:
    """`nmf_nndsvd_run` on the resident kit (the same cells per component;
    U's column j taken as the row of U^T, the same words in an n x 1 shape)."""
    if nc > r or n * r > 2147483647 or r * d > 2147483647 or n * nc > 2147483647:
        raise Error("x_decomp: nndsvd shape out of range")
    var k = DKit()
    var Ut = k.t(k.upload(mat_from(pu, n, r)))
    var Sd = k.upload(mat_from(ps, r, 1))
    var V = k.upload(mat_from(pv, r, d))
    var Wt = k.zeros(nc, n)
    var H = k.zeros(nc, d)
    for j in range(nc):  # small-loop(nc: components): one factor pair per component, each a whole-matrix cell chain
        var x = k.vec_t(k.rows(Ut, j, j + 1))
        var y = k.rows(V, j, j + 1)
        var sj = k.rows(Sd, j, j + 1)
        var wc: DMat
        var hr: DMat
        if j == 0:
            var rr = k.ew1(OP_SQRT, sj, 0.0)
            wc = k.ew2(OP_MUL, k.ew1(OP_ABS, x, 0.0), rr)
            hr = k.ew2(OP_MUL, k.ew1(OP_ABS, y, 0.0), rr)
        else:
            var xp = k.ew1(OP_MAXS, x, 0.0)
            var yp = k.ew1(OP_MAXS, y, 0.0)
            var xn = k.ew1(OP_ABS, k.ew1(OP_MINS, x, 0.0), 0.0)
            var yn = k.ew1(OP_ABS, k.ew1(OP_MINS, y, 0.0), 0.0)
            var xpn = _vnorm_dev(k, xp)
            var ypn = _vnorm_dev(k, yp)
            var xnn = _vnorm_dev(k, xn)
            var ynn = _vnorm_dev(k, yn)
            var mp = _f32_prod(xpn, ypn)
            var mn = _f32_prod(xnn, ynn)
            var uu: DMat
            var vv: DMat
            var sigma: Float64
            if mp > mn:
                uu = k.ew1(OP_SCALE, xp, _nrm_or0(xpn))
                vv = k.ew1(OP_SCALE, yp, _nrm_or0(ypn))
                sigma = mp
            else:
                uu = k.ew1(OP_SCALE, xn, _nrm_or0(xnn))
                vv = k.ew1(OP_SCALE, yn, _nrm_or0(ynn))
                sigma = mn
            var lbd = k.ew1(OP_SQRT, k.ew1(OP_SCALE, sj, sigma), 0.0)
            wc = k.ew2(OP_MUL, uu, lbd)
            hr = k.ew2(OP_MUL, vv, lbd)
        k.place_row(Wt, wc, j)
        k.place_row(H, hr, j)
    var Wd = k.t(Wt)
    var hw = k.get(Wd)
    var hh = k.get(H)
    k.sync()
    _ = Wd^
    for i in range(n * nc):
        pw.unsafe_store(i, hw.d[i])
    for i in range(nc * d):
        ph.unsafe_store(i, hh.d[i])
    return nc


def _vnorm_dev(mut k: DKit, v: DMat) raises -> Float64:
    return k.word(k.ew1(OP_SQRT, k.total(k.ew1(OP_SQ, v, 0.0)), 0.0))
