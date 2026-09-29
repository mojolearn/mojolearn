# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""THE GLM FIT PAST ONE BLOCK, SAME BITS (lane/linfit-speed, 2026-09-29).

x_linear/glm.mojo `glm_fit` on the device is ONE block of LINEAR_TPB
threads (x_linear/device.mojo): its gradient and Hessian are m + m(m+1)/2
cells (24,752 at the board's Istella shape, 1M x 220), each one thread's
fold over all n rows, so 256 threads walked ~97 cells each, one after the
other, on one multiprocessor, and every objective's fold over the n loss
terms ran on the lead alone. PoissonRegressor took 897 s there on an L40S.

Here (IDENTICAL) the same fit runs its row passes over the whole GPU and
its control on the host. NOTHING about any value's sequence changes:

  * the row map (the linear predictor, the loss term or the two
    derivatives of row i) is `glm_fit`'s expression for row i, one thread
    per row, as the team dealt it;
  * every gradient and Hessian cell is ONE thread's chain over the rows
    ascending, the team's `chain_fmad` / `chain_fmad_scaled` / `fold_fa`
    for that cell. When a launch covers only rows [r0, r1), the cell's
    chain stops there, its accumulator goes to device memory as a float32
    word and the next launch continues the same chain from it (`init`):
    storing and reloading a word is exact, so the chain is the one
    unbroken fold;
  * the objective's fold over the loss terms (rows ascending) and the small
    dense step (the Cholesky solve, the Armijo test, the stopping rules)
    are the lead thread's code, run on the host: the CPU column runs this
    same source and every column agrees with it bit for bit.

Every Apple launch is BOUNDED in work (macOS aborts a Metal launch that
holds the GPU for seconds, leaving its output partly written, with no
error): the rows are cut into launches of at most GW_ROW_STEPS row terms
and GW_CELL_STEPS cell-rows (elsewhere a pass is one launch). Every output of every launch is POISONED first
(a quiet NaN word no computation here produces), read back after the launch
and checked; a surviving poison word raises. The cut is scheduling only
(the chains carry across it), so the words do not depend on it:
MOJOLEARN_X_LINEAR_GW_STEPS=<k> sets both budgets at run time, which the
gate uses to prove exactly that. MOJOLEARN_X_LINEAR_GLM_TEAM=1 runs the
one-block team fit instead (the A/B arm). MOJOLEARN_X_LINEAR_GW_TRACE=1
prints where the fit's wall time went (`GLM-WIDE ...`, stdout).
MOJOLEARN_X_LINEAR_GW_GRAM=0 runs every cell on the one-cell kernel
instead of the tiled kernel (the A/B arm; the same words).
"""
from std.gpu import block_idx, block_dim, thread_idx
from std.os import getenv
from std.sys.info import has_apple_gpu_accelerator
from std.time import perf_counter_ns
from std.memory import bitcast, stack_allocation
from max.gpu.host import DeviceBuffer, DeviceContext, HostBuffer
from max.gpu.memory import AddressSpace
from max.gpu.sync import barrier
from x_linear.ops import (
    FP, fa, fs, fm, fd, fmad, flog, fabs, fmax, ld, st, i2f, fill, copy, row_dot,
    cholesky, chol_solve, mean_of,
)
from x_linear.ops import fz, xmad
from x_linear.tops import fold_fa, chain_fmad, chain_fmad_scaled, _fm
from x_linear.glm import _unit, GLM_LINK_LOG

#: Threads per block (<= 256: the M2 Pro silently drops larger dispatches).
comptime GW_TPB = 256
#: Row terms (rows x d) per row-map launch on Apple. SCHEDULING only.
comptime GW_ROW_STEPS = 1 << 26
#: Cell-rows (cells x rows) per gradient/Hessian launch on Apple. SCHEDULING only.
comptime GW_CELL_STEPS = 1 << 27
#: Elsewhere (no watchdog on a compute GPU) a pass is one launch unless it
#: exceeds these. SCHEDULING only.
comptime GW_ROW_STEPS_OTHER = 1 << 40
comptime GW_CELL_STEPS_OTHER = 1 << 40
#: The poison word: a quiet NaN whose payload no operation here produces
#: (the inputs are finite, and a computed NaN is the default NaN).
comptime GW_POISON = UInt32(0x7FC0DEAD)


def gw_rows_kernel(
    x: FP, y: FP, th: FP, rw: FP, r0_in: Int32, r1_in: Int32, n_in: Int32, d_in: Int32,
    flags_in: Int32, what_in: Int32,
):
    """Row i of [r0, r1): `glm_fit`'s row map. th: theta (d, then the
    intercept) | power at d + 1. flags: bit 0 fit_intercept, bit 1
    sample_weight, bits 8.. the link. what 0: the loss term into rw[i];
    what 1: d/deta into rw[i], d2/deta2 (clamped at zero) into rw[n + i]."""
    var i = Int(r0_in) + Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i >= Int(r1_in):
        return
    var n = Int(n_in)
    var d = Int(d_in)
    var flags = Int(flags_in)
    var fi = (flags & 1) != 0
    var sw = (flags & 2) != 0
    var link = flags >> 8
    var power = ld(th, d + 1)
    var b = ld(th, d) if fi else Float32(0)
    var e = fa(row_dot(x, i, d, th, 0), b)
    var yi = ld(y, i)
    if Int(what_in) == 0:
        var l = _unit(power, link, yi, e, 0)
        if sw:
            l = fm(ld(y, n + i), l)
        st(rw, i, l)
    else:
        var gi = _unit(power, link, yi, e, 1)
        var hi = fmax(Float32(0), _unit(power, link, yi, e, 2))
        if sw:
            gi = fm(ld(y, n + i), gi)
            hi = fm(ld(y, n + i), hi)
        st(rw, i, gi)
        st(rw, n + i, hi)


def gw_cells_kernel(
    x: FP, rw: FP, src: FP, dst: FP, r0_in: Int32, rows_in: Int32, n_in: Int32, d_in: Int32,
    m_in: Int32, first_in: Int32, edge_in: Int32,
):
    """Cell c (`glm_fit`'s order: m gradient cells, then the Hessian's lower
    triangle row by row) over rows [r0, r0 + rows): its chain continued
    from src[c] (from zero when `first`), into dst[c]. `edge`: only the
    cells a tiled kernel does not take (the gradient, and the intercept's
    Hessian row when m = d + 1), thread t -> the t-th of them."""
    var c = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var n = Int(n_in)
    var d = Int(d_in)
    var m = Int(m_in)
    var cells = m + m * (m + 1) // 2
    if Int(edge_in) != 0:
        if c >= m:
            if m == d or c >= 2 * m:
                return
            c = m + d * (d + 1) // 2 + (c - m)
    if c >= cells:
        return
    var r0 = Int(r0_in)
    var rows = Int(rows_in)
    var init = Float32(0) if Int(first_in) != 0 else ld(src, c)
    var acc: Float32
    if c < m:
        if c < d:
            acc = chain_fmad(rw, r0, 1, x, r0 * d + c, d, rows, init)
        else:
            acc = fold_fa(rw, r0, 1, rows, init)
    else:
        var q = c - m
        var j = 0
        while (j + 1) * (j + 2) // 2 <= q:
            j += 1
        var k = q - j * (j + 1) // 2
        if j < d:
            acc = chain_fmad_scaled(rw + (n + r0), x + r0 * d, j, k, d, rows, init)
        elif k < d:
            acc = chain_fmad(rw, n + r0, 1, x, r0 * d + k, d, rows, init)
        else:
            acc = fold_fa(rw, n + r0, 1, rows, init)
    st(dst, c, acc)


#: THE TILED CELL PASS. Every cell of the pass is a chain
#: acc = fz(fma(t_ij, u_ik, acc)) over rows i ascending, for one (j, k) of a
#: table of "rows" j = 0..m and "columns" k = 0..m-1:
#:   t_ij = fm(h_i, x_ij) (j < d), fm(h_i, 1) = fz(h_i) (j = d, the intercept),
#:          fm(g_i, 1) = fz(g_i) (j = m, the gradient);
#:   u_ik = fz(x_ik) (k < d), 1 (k = d).
#: Each is EXACTLY the one-cell kernel's chain for that cell:
#:   Hessian (j, k < d): chain_fmad_scaled, acc = fz(fma(fm(h, x_j), fz(x_k), acc));
#:   Hessian (d, k < d): chain_fmad(h, x_k), acc = fz(fma(fz(h), fz(x_k), acc));
#:   Hessian (d, d): fold_fa(h), acc = fz(acc + fz(h)) = fz(fma(fz(h), 1, acc))
#:     (a * 1 + c rounds once, to the sum's word);
#:   gradient (k < d): chain_fmad(g, x_k); gradient (d): fold_fa(g), the same way
#:   (fm(v, 1) = fz(v): v * 1 is exact).
#: A block owns a GT x GT square of (j, k), one cell per thread; per row it
#: stages the GT t's and GT u's in threadgroup memory once, so a thread's
#: step is two shared loads, one fma and one compare.
comptime GT = 16
comptime GW_GRAM_TPB = GT * GT
#: Rows a block stages at a time (two buffers of GR x 2GT words).
comptime GR = 16
comptime GROW = 2 * GT
#: Staged words per thread per chunk.
comptime GPER = (GR * GROW + GW_GRAM_TPB - 1) // GW_GRAM_TPB


@always_inline
def _tile_fetch(
    x: FP, rw: FP, n: Int, d: Int, m: Int, r0: Int, rows: Int, bj: Int, bk: Int, tid: Int, cix: Int,
) -> InlineArray[Float32, GPER]:
    """This thread's share of chunk `cix`'s staged words: row r of the chunk
    at r * GROW, the block's GT t's then its GT u's (zero past the rows or
    the table); computed from loads, not yet stored."""
    var stage = InlineArray[Float32, GPER](fill=Float32(0))
    comptime for q in range(GPER):
        var e = tid + q * GW_GRAM_TPB
        var v = Float32(0)
        if e < GR * GROW:
            var r = e // GROW
            var c = e % GROW
            var i = cix * GR + r
            if i < rows:
                var gi = r0 + i
                if c < GT:
                    var j = bj * GT + c
                    if j < d:
                        v = _fm(ld(rw, n + gi), ld(x, gi * d + j))
                    elif j < m:
                        v = _fm(ld(rw, n + gi), Float32(1))
                    elif j == m:
                        v = _fm(ld(rw, gi), Float32(1))
                else:
                    var k = bk * GT + (c - GT)
                    if k < d:
                        v = fz(ld(x, gi * d + k))
                    elif k < m:
                        v = Float32(1)
        stage[q] = v
    return stage^


def gw_tile_kernel(
    x: FP, rw: FP, src: FP, dst: FP, r0_in: Int32, rows_in: Int32, n_in: Int32, d_in: Int32,
    m_in: Int32, first_in: Int32,
):
    """Every cell of the pass (see THE TILED CELL PASS) over rows
    [r0, r0 + rows), each chain continued from src (zero when `first`) into
    dst in `glm_fit`'s cell order. The chain runs without its flush while no
    partial is subnormal (then the flush is the identity); a compare off the
    chain watches every partial, and a cell that saw one below the smallest
    normal (a subnormal, or a zero) is recomputed exactly by the one-cell
    function from the same init over the same rows. The same word either way."""
    comptime MIN_NORMAL = Float32(1.17549435e-38)
    var n = Int(n_in)
    var d = Int(d_in)
    var m = Int(m_in)
    var r0 = Int(r0_in)
    var rows = Int(rows_in)
    var tid = Int(thread_idx.x)
    var p = Int(block_idx.x)
    var bj = 0
    while (bj + 1) * (bj + 2) // 2 <= p:
        bj += 1
    var bk = p - bj * (bj + 1) // 2
    var j = bj * GT + tid // GT
    var k = bk * GT + tid % GT
    # this thread's cell in glm_fit's order, or -1
    var cell = -1
    if j < m and k <= j:
        cell = m + j * (j + 1) // 2 + k
    elif j == m and k < m:
        cell = k
    var init = Float32(0)
    if Int(first_in) == 0 and cell >= 0:
        init = ld(src, cell)
    var acc = init
    var bad = False
    var sm = stack_allocation[2 * GR * GROW, Float32, address_space=AddressSpace.SHARED]()
    var chunks = (rows + GR - 1) // GR
    var stage = _tile_fetch(x, rw, n, d, m, r0, rows, bj, bk, tid, 0)
    comptime for q in range(GPER):
        var e = tid + q * GW_GRAM_TPB
        if e < GR * GROW:
            sm[e] = stage[q]
    barrier()
    var tj = tid // GT
    var tk = GT + tid % GT
    for cix in range(chunks):
        if cix + 1 < chunks:
            stage = _tile_fetch(x, rw, n, d, m, r0, rows, bj, bk, tid, cix + 1)
        var base = (cix % 2) * GR * GROW
        if (cix + 1) * GR <= rows:
            comptime for r in range(GR):
                var v = xmad(sm[base + r * GROW + tj], sm[base + r * GROW + tk], acc)
                acc = v
                bad = bad | (abs(v) < MIN_NORMAL)
        else:
            for r in range(rows - cix * GR):
                var v = xmad(sm[base + r * GROW + tj], sm[base + r * GROW + tk], acc)
                acc = v
                bad = bad | (abs(v) < MIN_NORMAL)
        if cix + 1 < chunks:
            var nb = ((cix + 1) % 2) * GR * GROW
            comptime for q in range(GPER):
                var e = tid + q * GW_GRAM_TPB
                if e < GR * GROW:
                    sm[nb + e] = stage[q]
        barrier()
    if cell < 0:
        return
    if bad:
        # the exact one-cell chain (gw_cells_kernel's) from the same init
        if j < d:
            acc = chain_fmad_scaled(rw + (n + r0), x + r0 * d, j, k, d, rows, init)
        elif j < m:
            if k < d:
                acc = chain_fmad(rw, n + r0, 1, x, r0 * d + k, d, rows, init)
            else:
                acc = fold_fa(rw, n + r0, 1, rows, init)
        else:
            if k < d:
                acc = chain_fmad(rw, r0, 1, x, r0 * d + k, d, rows, init)
            else:
                acc = fold_fa(rw, r0, 1, rows, init)
    st(dst, cell, acc)


def _budget(default: Int) -> Int:
    var s = getenv("MOJOLEARN_X_LINEAR_GW_STEPS", "")
    if s == "":
        return default
    try:
        var v = Int(s)
        return v if v > 0 else default
    except:
        return default


def use_glm_wide() -> Bool:
    """False when MOJOLEARN_X_LINEAR_GLM_TEAM=1 asks for the one-block fit."""
    return getenv("MOJOLEARN_X_LINEAR_GLM_TEAM", "") != "1"


def _poison() -> Float32:
    return bitcast[DType.float32](GW_POISON)


def _check(p: FP, off: Int, count: Int, what: String) raises:
    for i in range(count):
        if bitcast[DType.uint32](ld(p, off + i)) == GW_POISON:
            raise Error(
                "x_linear GLM: a device launch (" + what + ") left output word " + String(i)
                + " unwritten (on Apple: a launch the system aborted)"
            )


struct GW(Movable):
    """The fit's device and host buffers (fields, so they outlive every
    launch that takes their pointers)."""

    var dx: DeviceBuffer[DType.float32]
    var dy: DeviceBuffer[DType.float32]
    var dth: DeviceBuffer[DType.float32]
    var hth: HostBuffer[DType.float32]
    var drw: DeviceBuffer[DType.float32]
    var hrw: HostBuffer[DType.float32]
    var dca: DeviceBuffer[DType.float32]
    var dcb: DeviceBuffer[DType.float32]
    var hc: HostBuffer[DType.float32]
    var n: Int
    var d: Int
    var m: Int
    var cells: Int
    var flags: Int
    var row_steps: Int
    var cell_steps: Int
    var rows_ns: Int
    var cells_ns: Int
    var n_rows: Int
    var n_cells: Int
    var gram: Bool

    def __init__(
        out self, ctx: DeviceContext, x: FP, n_x: Int, y: FP, n_y: Int, n: Int, d: Int, m: Int, flags: Int,
    ) raises:
        self.n = n
        self.d = d
        self.m = m
        self.cells = m + m * (m + 1) // 2
        self.flags = flags
        comptime if has_apple_gpu_accelerator():
            self.row_steps = _budget(GW_ROW_STEPS)
            self.cell_steps = _budget(GW_CELL_STEPS)
        else:
            self.row_steps = _budget(GW_ROW_STEPS_OTHER)
            self.cell_steps = _budget(GW_CELL_STEPS_OTHER)
        self.rows_ns = 0
        self.cells_ns = 0
        self.n_rows = 0
        self.n_cells = 0
        self.gram = getenv("MOJOLEARN_X_LINEAR_GW_GRAM", "") != "0"
        self.dx = ctx.enqueue_create_buffer[DType.float32](max(n_x, 1))
        self.dy = ctx.enqueue_create_buffer[DType.float32](max(n_y, 1))
        self.dth = ctx.enqueue_create_buffer[DType.float32](d + 2)
        self.hth = ctx.enqueue_create_host_buffer[DType.float32](d + 2)
        self.drw = ctx.enqueue_create_buffer[DType.float32](2 * n)
        self.hrw = ctx.enqueue_create_host_buffer[DType.float32](2 * n)
        self.dca = ctx.enqueue_create_buffer[DType.float32](self.cells)
        self.dcb = ctx.enqueue_create_buffer[DType.float32](self.cells)
        self.hc = ctx.enqueue_create_host_buffer[DType.float32](self.cells)
        if n_x > 0:
            ctx.enqueue_copy(dst_buf=self.dx, src_ptr=x)
        if n_y > 0:
            ctx.enqueue_copy(dst_buf=self.dy, src_ptr=y)
        var hp = self.hth.unsafe_ptr()
        for j in range(d + 2):
            hp.unsafe_store(j, Float32(0))

    def upload(mut self, ctx: DeviceContext, th: FP, toff: Int, power: Float32) raises:
        """theta (m words) and the power, to the device."""
        var hp = self.hth.unsafe_ptr()
        for j in range(self.m):
            hp.unsafe_store(j, ld(th, toff + j))
        hp.unsafe_store(self.d + 1, power)
        ctx.enqueue_copy(dst_buf=self.dth, src_buf=self.hth)

    def rows(mut self, ctx: DeviceContext, what: Int, link: Int) raises -> FP:
        """The row map over every row (bounded launches), read back and
        checked: the host copy of rw (n words, or 2n for what 1)."""
        var t0 = perf_counter_ns()
        var words = self.n if what == 0 else 2 * self.n
        self.drw.enqueue_fill(_poison())
        var per = max(1, self.row_steps // max(self.d, 1))
        var r0 = 0
        while r0 < self.n:
            var r1 = min(self.n, r0 + per)
            ctx.enqueue_function[gw_rows_kernel](
                self.dx.unsafe_ptr(), self.dy.unsafe_ptr(), self.dth.unsafe_ptr(), self.drw.unsafe_ptr(),
                Int32(r0), Int32(r1), Int32(self.n), Int32(self.d), Int32(self.flags | (link << 8)),
                Int32(what),
                grid_dim=(r1 - r0 + GW_TPB - 1) // GW_TPB, block_dim=GW_TPB,
            )
            ctx.synchronize()
            r0 = r1
        ctx.enqueue_copy(dst_buf=self.hrw, src_buf=self.drw)
        ctx.synchronize()
        var hp = FP(unsafe_from_address=Int(self.hrw.unsafe_ptr()))
        _check(hp, 0, words, "row map")
        self.rows_ns += perf_counter_ns() - t0
        self.n_rows += 1
        return hp

    def launch(mut self, ctx: DeviceContext, src: FP, dst: FP, r0: Int, rows: Int, first: Int32) raises:
        """One bounded pass of every cell: the tiled kernel (or the one-cell
        kernel with MOJOLEARN_X_LINEAR_GW_GRAM=0)."""
        if not self.gram:
            ctx.enqueue_function[gw_cells_kernel](
                self.dx.unsafe_ptr(), self.drw.unsafe_ptr(), src, dst,
                Int32(r0), Int32(rows), Int32(self.n), Int32(self.d), Int32(self.m), first, Int32(0),
                grid_dim=(self.cells + GW_TPB - 1) // GW_TPB, block_dim=GW_TPB,
            )
            return
        var nbj = (self.m + 1 + GT - 1) // GT
        ctx.enqueue_function[gw_tile_kernel](
            self.dx.unsafe_ptr(), self.drw.unsafe_ptr(), src, dst,
            Int32(r0), Int32(rows), Int32(self.n), Int32(self.d), Int32(self.m), first,
            grid_dim=nbj * (nbj + 1) // 2, block_dim=GW_GRAM_TPB,
        )

    def cell_pass(mut self, ctx: DeviceContext) raises -> FP:
        """Every gradient/Hessian cell over all rows, from the d/deta and
        d2/deta2 rows `rows(1)` left in drw: the chains cut into bounded
        launches (ping-pong), each launch's cells poisoned, read back and
        checked. The host copy of the cells."""
        var t0 = perf_counter_ns()
        var per = max(1, self.cell_steps // self.cells)
        var r0 = 0
        var first = True
        var into_b = False
        while r0 < self.n:
            var rows = min(self.n - r0, per)
            var f = Int32(1 if first else 0)
            if into_b:
                self.dca.enqueue_fill(_poison())
                self.launch(ctx, self.dcb.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin](), self.dca.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin](), r0, rows, f)
                ctx.enqueue_copy(dst_buf=self.hc, src_buf=self.dca)
            else:
                self.dcb.enqueue_fill(_poison())
                self.launch(ctx, self.dca.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin](), self.dcb.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin](), r0, rows, f)
                ctx.enqueue_copy(dst_buf=self.hc, src_buf=self.dcb)
            ctx.synchronize()
            var hp = FP(unsafe_from_address=Int(self.hc.unsafe_ptr()))
            _check(hp, 0, self.cells, "gradient/Hessian cells")
            first = False
            into_b = not into_b
            r0 += rows
            self.n_cells += 1
        self.cells_ns += perf_counter_ns() - t0
        return FP(unsafe_from_address=Int(self.hc.unsafe_ptr()))


def _objective(
    mut b: GW, ctx: DeviceContext, theta: FP, toff: Int, power: Float32, link: Int, alpha: Float32,
    den: Float32,
) raises -> Float32:
    """`glm.mojo _objective_team`: the loss terms on the device, then the
    lead's fold (rows ascending) and penalty, here on the host."""
    b.upload(ctx, theta, toff, power)
    var lt = b.rows(ctx, 0, link)
    var acc = fold_fa(lt, 0, 1, b.n)
    var reg = Float32(0)
    for j in range(b.d):
        var w = ld(theta, toff + j)
        reg = fmad(w, w, reg)
    return fa(fd(acc, den), fm(fm(Float32(0.5), alpha), reg))


def glm_fit_wide(
    ctx: DeviceContext, x: FP, n_x: Int, y: FP, n_y: Int, n: Int, d: Int,
    ip: List[Int32], fp: List[Float32], res: FP,
) raises:
    """x_linear/glm.mojo `glm_fit`, operation for operation, with its row
    passes over the whole GPU (see the header). res (host): coef d,
    intercept, n_iter, converged."""
    var max_iter = Int(ip[0])
    var fi = Int(ip[1]) != 0
    var link = Int(ip[2])
    var sw = Int(ip[3]) != 0
    var power = fp[0]
    var alpha = fp[1]
    var tol = fp[2]
    var t_start = perf_counter_ns()
    var m = d + 1 if fi else d
    var flags = (1 if fi else 0) | (2 if sw else 0)
    var b = GW(ctx, x, n_x, y, n_y, n, d, m, flags)
    var den = i2f(n)
    if sw:
        den = Float32(0)
        for i in range(n):
            den = fa(den, ld(y, n + i))
    # the lead's scratch: grad m | H m*m | step m | trial m
    var hw = List[Float32](capacity=3 * m + m * m)
    for _ in range(3 * m + m * m):
        hw.append(Float32(0))
    var g = FP(unsafe_from_address=Int(hw.unsafe_ptr()))
    var h = g + m
    var step = h + m * m
    var trial = step + m
    fill(res, 0, d + 3, Float32(0))
    if fi:
        var ym = mean_of(y, n)
        if sw:
            var acc = Float32(0)
            for i in range(n):
                acc = fmad(ld(y, n + i), ld(y, i), acc)
            ym = fd(acc, den)
        st(res, d, flog(ym) if link == GLM_LINK_LOG else ym)
    var iters = 0
    var converged = False
    var f = _objective(b, ctx, res, 0, power, link, alpha, den)
    for it in range(max_iter):
        # gradient and Hessian at res: the row derivatives, then every cell
        b.upload(ctx, res, 0, power)
        _ = b.rows(ctx, 1, link)
        var cp = b.cell_pass(ctx)
        for c in range(m):
            st(g, c, ld(cp, c))
        for j in range(m):
            for k in range(j + 1):
                st(h, j * m + k, ld(cp, m + j * (j + 1) // 2 + k))
        # the small dense step (m x m): glm_fit's lead code
        var flag = 0  # 0 continue, 1 converged, 2 stop (no descent)
        var slope = Float32(0)
        var inv_n = fd(Float32(1), den)
        var gmax = Float32(0)
        for j in range(m):
            var gj = fm(ld(g, j), inv_n)
            if j < d:
                gj = fmad(alpha, ld(res, j), gj)
            st(g, j, gj)
            gmax = fmax(gmax, fabs(gj))
        if gmax <= tol:
            flag = 1
        else:
            for j in range(m):
                for k in range(j + 1):
                    var v = fm(ld(h, j * m + k), inv_n)
                    if j == k and j < d:
                        v = fa(v, alpha)
                    st(h, j * m + k, v)
                    st(h, k * m + j, v)
            for j in range(m):
                st(step, j, -ld(g, j))
            var ok = cholesky(h, 0, m)
            if ok:
                chol_solve(h, 0, m, step, 0)
            for j in range(m):
                slope = fmad(ld(g, j), ld(step, j), slope)
            if not (slope < 0):
                flag = 2
        if flag == 1:
            converged = True
            break
        iters = it + 1
        if flag == 2:
            break
        var tt = Float32(1)
        var accepted = False
        for _ in range(40):
            for j in range(m):
                st(trial, j, fmad(tt, ld(step, j), ld(res, j)))
            var ft = _objective(b, ctx, trial, 0, power, link, alpha, den)
            if ft == ft and ft <= fa(f, fm(fm(Float32(1e-4), tt), slope)):
                copy(res, 0, trial, 0, m)
                f = ft
                accepted = True
                break
            tt = fm(tt, Float32(0.5))
        if not accepted:
            # glm_fit recomputes the objective at res here (it restores the
            # predictor for no later use) and stops; nothing it computes is kept
            break
    if not fi:
        st(res, d, Float32(0))
    st(res, d + 1, i2f(iters))
    st(res, d + 2, Float32(1) if converged else Float32(0))
    if getenv("MOJOLEARN_X_LINEAR_GW_TRACE", "") != "":
        var total = perf_counter_ns() - t_start
        print(
            "GLM-WIDE n", n, "d", d, "iters", iters, "row_passes", b.n_rows, "cell_launches", b.n_cells,
            "total_ms", Float64(total) / 1.0e6, "rows_ms", Float64(b.rows_ns) / 1.0e6,
            "cells_ms", Float64(b.cells_ns) / 1.0e6,
            "host_ms", Float64(total - b.rows_ns - b.cells_ns) / 1.0e6,
        )
    _ = hw^
    _ = b^
