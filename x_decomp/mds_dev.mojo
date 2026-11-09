# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""x_decomp/mds.mojo's SMACOF on resident device matrices (lane
py-runtime-b): the same statements on `DKit`, non-metric disparities by
x_decomp/mds_iso_dev.mojo's setup and isotonic kernels (`_nm_native`'s
resident form). GPU binding only.

Lane n4-mds-labelprop-nmf-loops (2026-11-09): one start's iterations run as
the executor's bounded device loop (autograd/program_loop_device.mojo). Y,
the distances d and the raw stress and ssd words are the loop's carried
state, written only through `DKit.put_gated`; the stop test is
`mds_smacof_crit_kernel` (`mds_stop_f32`, the host column's function) and
the non-metric scale is a device scalar (`mds_disp_scale_f32`'s cells), so
no word comes home inside the loop. The host reads the register block once
per MDS_LOOP_CHUNK iterations and the stress once per start; before, every
iteration read two or three words (the stress, ssd and the disparities'
sum of squares)."""
# TOMBSTONE: mds-smacof-host-word-loop (REFACTORED onto autograd/program_loop_device.mojo) deleted 2026-11-09 by lane/n4-mds-labelprop-nmf-loops; code recoverable at f59d913b0b.
# Restore: git show f59d913b0b:x_decomp/mds_dev.mojo; record in docs/TOMBSTONES.md.
from std.math import sqrt
from std.gpu import block_idx, thread_idx
from max.gpu.host import DeviceContext
from autograd.program_loop import LoopTimers
from autograd.program_loop_device import LoopRegisters, LoopStep, bounded_device_loop, loop_lap
from core.step_gate import STEP_DONE, STEP_ITER, STEP_STOP_ITER
from x_decomp.cells import F32Ptr, I32Ptr
from x_decomp.device import xd_ctx
from x_decomp.kit import (
    Mat, OP_ADD, OP_DIV, OP_FMA, OP_MUL, OP_SCALE, OP_SELECT, OP_SQ, OP_SQDIFF, OP_SQRT, mat_const, mat_eye,
    mat_from,
)
from x_decomp.kit_device import DKit, DMat
from x_decomp.mds import MdsArgs, mds_half_f32, mds_half_pairs, mds_stop_f32
from x_decomp.mds_iso_dev import mds_disp_dev, mds_setup_dev

#: SMACOF iterations enqueued between two reads of the loop's register
#: block. The stop is tested every iteration on the device; up to this many
#: minus one iterations run gated past it (their state writes are no-ops).
#: A cost rule, not a shape: one drain per eight iterations against at most
#: seven wasted ones (KMeans' Lloyd loop reads every 8 as well).
comptime MDS_LOOP_CHUNK = 8


struct _NmDev(Movable):
    var bufs: List[DMat]
    var m: Int
    var G: Int

    def __init__(out self, Dis: DMat, native: Bool) raises:
        self.bufs = List[DMat]()
        self.m = 0
        self.G = 0
        if not native:
            return
        var n = Dis.r
        var N2 = 1
        for _ in range(62):  # small-loop(N2: doublings to the pair count's power of two): Python's 1 << bit_length
            if N2 >= n * n:
                break
            N2 *= 2
        # keys, idx, gid, tmp (N2), gst (N2 + 1), word (1)
        var setup = List[DMat]()
        for _ in range(4):  # small-loop(N2: four pair-sized buffers): keys, idx, gid, tmp
            setup.append(DMat(1, N2))
        setup.append(DMat(1, N2 + 1))
        setup.append(DMat(1, 1))
        var ids = List[Int]()
        for i in range(6):  # small-loop(i: six buffer ids): the setup's buffers
            ids.append(setup[i].id)
        var r = mds_setup_dev(Dis.id, ids, n, N2)
        self.m = r[0]
        self.G = r[1]
        # keep keys, idx, gid, gst; then sm, wt, end, prv, last, hf, hd0, hd1, gv
        self.bufs.append(setup.pop(0))
        self.bufs.append(setup.pop(0))
        self.bufs.append(setup.pop(0))
        _ = setup.pop(0)
        self.bufs.append(setup.pop(0))
        for _ in range(9):  # small-loop(G: nine group-sized buffers): the isotonic step's scratch
            self.bufs.append(DMat(1, max(self.G, 1)))

    def ids(self) -> List[Int]:
        var out = List[Int]()
        for i in range(len(self.bufs)):  # small-loop(i: thirteen buffer ids): the disparity step's buffers
            out.append(self.bufs[i].id)
        return out^


def _dist_dev(k: DKit, Y: DMat) raises -> DMat:
    return k.ew1(OP_SQRT, k.sqdist(Y, Y), 0.0)


def _view(D: DMat) -> DMat:
    """A non-owning view of all of D."""
    return DMat(rows_of=D, row0=0, rows=D.r)


def mds_smacof_crit_kernel(words: I32Ptr, stress_w: F32Ptr, ssd_w: F32Ptr, old: F32Ptr, eps: Float32, it: Int32):
    """One thread, while the loop runs: the stress (half the raw word) against
    the previous one by `mds_stop_f32` (from the second iteration on, as
    `have_old`), else the stress kept as the next `old`."""
    if Int(block_idx.x) != 0 or Int(thread_idx.x) != 0:
        return
    if words.unsafe_load(STEP_DONE) != Int32(0):
        return
    words.unsafe_store(STEP_ITER, it)
    var stress = mds_half_f32(stress_w.unsafe_load(0))
    if it > Int32(1) and mds_stop_f32(old.unsafe_load(0), stress, ssd_w.unsafe_load(0), eps):
        words.unsafe_store(STEP_STOP_ITER, it)
        words.unsafe_store(STEP_DONE, Int32(1))
        return
    old.unsafe_store(0, stress)


struct _SmacofStep(LoopStep, Movable):
    """One `MDS._single` iteration as a step of the bounded device loop."""
    var k: DKit
    var Dis: DMat
    var Y: DMat
    var d: DMat
    var stress_w: DMat
    var ssd_w: DMat
    var old: DMat
    var P: DMat
    var nm: _NmDev
    var native: Bool
    var n: Int
    var eye: DMat
    var floor: DMat
    var half_pairs: DMat
    var eps: Float32

    def __init__(out self, Dis: DMat, Y0: DMat, a: MdsArgs) raises:
        var n = Dis.r
        var native = not a.metric
        var k = DKit()
        var nm = _NmDev(Dis, native)
        var Y = k.copy(Y0)
        var d = _dist_dev(k, Y)
        # the state before the first iteration: stress 0, ssd of the start's
        # distances (what the final normalization reads when nothing ran)
        var stress_w = k.zeros(1, 1)
        var ssd_w = k.total(k.ew1(OP_SQ, d, 0.0))
        var eye = k.upload(mat_eye(n))
        var floor = k.upload(mat_const(1e-5, 1, 1))
        var half_pairs = k.upload(mat_const(Float64(mds_half_pairs(n)), 1, 1))
        var P = DMat(0, 0)
        if native:
            P = DMat(n, n)
        self.k = k^
        self.Dis = _view(Dis)
        self.Y = Y^
        self.d = d^
        self.stress_w = stress_w^
        self.ssd_w = ssd_w^
        self.old = DMat(1, 1)
        self.P = P^
        self.nm = nm^
        self.native = native
        self.n = n
        self.eye = eye^
        self.floor = floor^
        self.half_pairs = half_pairs^
        self.eps = Float32(a.eps)

    def enqueue_step(
        mut self, ctx: DeviceContext, mut registers: LoopRegisters, it: Int, mut timers: LoopTimers,
    ) raises:
        var gate: I32Ptr = registers.words.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]()
        var n = self.n
        var disp: DMat
        if self.native:
            mds_disp_dev(
                self.P.id if it == 1 else self.d.id, self.P.id, self.nm.ids(), n, self.nm.m, self.nm.G, it == 1
            )
            var ss = self.k.total(self.k.ew1(OP_SQ, self.P, 0.0))
            var f = self.k.ew1(OP_SQRT, self.k.ew2(OP_DIV, self.half_pairs, ss), 0.0)
            var Ps = self.k.ew2(OP_MUL, self.P, f)
            disp = self.k.ew2(OP_ADD, Ps, self.k.t(Ps))
            loop_lap(ctx, timers, "disparities")
        else:
            disp = _view(self.Dis)
        var dz = self.k.ew3(OP_SELECT, self.d, self.d, self.floor, 0.0)
        var ratio = self.k.ew2(OP_DIV, disp, dz)
        var B = self.k.ew1(OP_SCALE, ratio, -1.0)
        var rs = self.k.rowsum(ratio)
        B = self.k.ew3(OP_FMA, self.eye, rs, B, 0.0)
        var Yn = self.k.ew1(OP_SCALE, self.k.mm(B, self.Y, False, False), 1.0 / Float64(n))
        var dn = _dist_dev(self.k, Yn)
        var sw = self.k.total(self.k.ew2(OP_SQDIFF, dn, disp))
        var ssd = self.k.total(self.k.ew1(OP_SQ, dn, 0.0))
        self.k.put_gated(gate, self.Y, Yn)
        self.k.put_gated(gate, self.d, dn)
        self.k.put_gated(gate, self.stress_w, sw)
        self.k.put_gated(gate, self.ssd_w, ssd)
        loop_lap(ctx, timers, "guttman")
        self.k.ctx.enqueue_function[mds_smacof_crit_kernel](
            gate, self.stress_w.p(), self.ssd_w.p(), self.old.p(), self.eps, Int32(it), grid_dim=1, block_dim=1,
        )
        loop_lap(ctx, timers, "criterion")


def mds_single_dev(mut k: DKit, Dis: DMat, var Y: DMat, a: MdsArgs, mut stress: Float64, mut it: Int) raises -> DMat:
    """One start's SMACOF run: the final Y, its stress (normalized when
    `a.norm`) and the iteration count."""
    var ctx = xd_ctx()
    var registers = LoopRegisters(ctx)
    var step = _SmacofStep(Dis, Y, a)
    var timers = LoopTimers("mds.smacof")
    var outcome = bounded_device_loop(ctx, step, registers, 1, a.max_iter, MDS_LOOP_CHUNK, timers)
    timers.report()
    it = outcome.stop_iteration if outcome.converged else max(a.max_iter, 0)
    # the start's one read: its stress and ssd words
    var hs = k.get(step.stress_w)
    var hd = k.get(step.ssd_w)
    k.sync()
    step.k.sync()  # the step kit's held uploads, before it goes
    stress = Float64(mds_half_f32(hs.d[0]))
    if a.norm:
        var ssd = Float64(hd.d[0])
        stress = sqrt(stress / (ssd / 2)) if ssd != 0.0 else 0.0
    return k.copy(step.Y)


def mds_fit_dev_run(
    pd: F32Ptr, ps: F32Ptr, py_: F32Ptr, n: Int, nc: Int, ns: Int, a: MdsArgs, mut best_s: Float64
) raises -> Int:
    """`mds_fit_run` on the resident kit."""
    var best_it = 0
    var k = DKit()
    var D = k.upload(mat_from(pd, n, n))
    var S = k.upload(mat_from(ps, ns * n, nc))
    var best = DMat(n, nc)
    for r in range(ns):  # small-loop(ns: restarts): one SMACOF run per start
        var s = 0.0
        var it = 0
        var Y0 = k.rows(S, r * n, (r + 1) * n)
        var Y = mds_single_dev(k, D, Y0^, a, s, it)
        if r == 0 or s < best_s:
            best_s = s
            best_it = it
            best = Y^
    var h = k.get(best)
    k.sync()
    for i in range(n * nc):
        py_.unsafe_store(i, h.d[i])
    return best_it
