# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""OPTICS xi extraction on the device (lane cgr4-device-optim-optics,
2026-10-03). The reference's sequential steep-region walk
(`_xi_cluster`, `_extend_region`, `_update_filter_sdas`,
`_extract_xi_labels`) as parallel phases over the reachability plot, one
thread an index, every phase a kernel whose thread t calls the
`x_cluster/optics_xi_cells.mojo` cell for t (the formulation, and why no
step needs a serial walk, is that file's docstring). Integer scans are
`device_post`'s; sparse tables and pointer doubling take log2 n launches.

No phase runs one thread or one block over the plot. The host reads back
scalars only to size grids: the event and down-event counts, and each pair
chunk's kept count; then the clusters and the labels at the end. The host
column (`optics_xi_cells.optics_xi_host`) runs the same cells in the same
phases. Only the GPU binding imports this file."""
from std.gpu import block_dim, block_idx, thread_idx
from max.gpu.host import DeviceBuffer, DeviceContext

from x_cluster.bodies import FPtr, IPtr
from x_cluster.device_post import PTPB, RTPB, SCAN_PER, pgrid, scan_out_kernel, scan_part_kernel
from x_cluster.optics_xi_cells import (
    XB_N,
    XI_PAIR_CAP,
    xi_acc_out_cell,
    xi_brk_cell,
    xi_event_cell,
    xi_first_cell,
    xi_first_out_cell,
    xi_flag_cell,
    xi_gnext_cell,
    xi_jump_cell,
    xi_label_cell,
    xi_levels,
    xi_mark_cell,
    xi_pair_cell,
    xi_pair_out_cell,
    xi_plot_cell,
    xi_region_cell,
    xi_scatter_cell,
    xi_split_cell,
    xi_st0_cell,
    xi_st_cell,
    xi_w_cell,
)


@always_inline
def _tid() -> Int:
    return Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)


def xi_plot_kernel(reach: FPtr, ordering: IPtr, n: Int32, plot: FPtr, pos: IPtr, m: Int32):
    var i = _tid()
    if i < Int(m):
        xi_plot_cell(reach, ordering, Int(n), plot, pos, i)


def xi_flag_kernel(plot: FPtr, n: Int32, xc: Float32, f5: IPtr, m: Int32):
    var i = _tid()
    if i < Int(m):
        xi_flag_cell(plot, Int(n), xc, f5, i)


def xi_scatter_kernel(f: IPtr, ex: IPtr, pos: IPtr, n: Int32, m: Int32):
    var i = _tid()
    if i < Int(m):
        xi_scatter_cell(f, ex, pos, Int(n), i)


def xi_brk_kernel(f5: IPtr, ex5: IPtr, p5: IPtr, n: Int32, min_samples: Int32, b2: IPtr, m: Int32):
    var i = _tid()
    if i < Int(m):
        xi_brk_cell(f5, ex5, p5, Int(n), Int(min_samples), b2, i)


def xi_region_kernel(
    f5: IPtr, ex5: IPtr, p5: IPtr, exb: IPtr, pb: IPtr, n: Int32, end_: IPtr, jmp: IPtr, on: IPtr, m: Int32
):
    var i = _tid()
    if i < Int(m):
        xi_region_cell(f5, ex5, p5, exb, pb, Int(n), end_, jmp, on, i)


def xi_mark_kernel(on: IPtr, jmp: IPtr, m: Int32):
    var i = _tid()
    if i < Int(m):
        xi_mark_cell(on, jmp, i)


def xi_jump_kernel(jmp: IPtr, dst: IPtr, m: Int32):
    var i = _tid()
    if i < Int(m):
        xi_jump_cell(jmp, dst, i)


def xi_st0_kernel(src: FPtr, t: FPtr, m: Int32):
    var i = _tid()
    if i < Int(m):
        xi_st0_cell(src, t, i)


def xi_st_kernel(t: FPtr, m: Int32, k: Int32, is_max: Int32):
    var i = _tid()
    if i < Int(m):
        xi_st_cell(t, Int(m), Int(k), is_max != 0, i)


def xi_w_kernel(pred: IPtr, ordering: IPtr, pos: IPtr, w: FPtr, m: Int32):
    var i = _tid()
    if i < Int(m):
        xi_w_cell(pred, ordering, pos, w, i)


def xi_event_kernel(ev: IPtr, end_: IPtr, f5: IPtr, n: Int32, pmax: FPtr, mib: FPtr, isd: IPtr, ne: IPtr):
    var e = _tid()
    if e < Int(ne[0]):
        xi_event_cell(ev, end_, f5, Int(n), pmax, mib, isd, e)


def xi_split_kernel(isd: IPtr, exd: IPtr, dl: IPtr, ul: IPtr, ne: IPtr):
    var e = _tid()
    if e < Int(ne[0]):
        xi_split_cell(isd, exd, dl, ul, e)


def xi_pair_kernel(
    ul: IPtr, dl: IPtr, ev: IPtr, end_: IPtr, plot: FPtr, pmin: FPtr, wmax: FPtr, n: Int32,
    mibt: FPtr, ne: Int32, xc: Float32, min_cluster_size: Int32, pc: Int32, nd: Int32, r0: Int32,
    flag: IPtr, cs_: IPtr, ce_: IPtr, m: Int32,
):
    var t = _tid()
    if t < Int(m):
        xi_pair_cell(
            ul, dl, ev, end_, plot, pmin, wmax, Int(n), mibt, Int(ne), xc, Int(min_cluster_size), pc != 0,
            Int(nd), Int(r0), flag, cs_, ce_, t,
        )


def xi_pair_out_kernel(
    flag: IPtr, ex: IPtr, cs_: IPtr, ce_: IPtr, r0: Int32, nd: Int32, off: Int32, cl: IPtr, rw: IPtr, m: Int32
):
    var t = _tid()
    if t < Int(m):
        xi_pair_out_cell(flag, ex, cs_, ce_, Int(r0), Int(nd), Int(off), cl, rw, t)


def xi_first_kernel(rw: IPtr, ff: IPtr, fsf: FPtr, m: Int32):
    var k = _tid()
    if k < Int(m):
        xi_first_cell(rw, ff, fsf, k)


def xi_first_out_kernel(ff: IPtr, exf: IPtr, cl: IPtr, fs_: IPtr, fe_: IPtr, fsf: FPtr, m: Int32):
    var k = _tid()
    if k < Int(m):
        xi_first_out_cell(ff, exf, cl, fs_, fe_, fsf, k)


def xi_gnext_kernel(fe_: IPtr, fsmax: FPtr, c: Int32, ng: IPtr, jmp: IPtr, on: IPtr):
    var g = _tid()
    if g <= Int(c):
        xi_gnext_cell(fe_, fsmax, Int(c), ng, jmp, on, g)


def xi_acc_out_kernel(on: IPtr, exa: IPtr, fs_: IPtr, fe_: IPtr, as_: IPtr, ae_: IPtr, m: Int32):
    var g = _tid()
    if g < Int(m):
        xi_acc_out_cell(on, exa, fs_, fe_, as_, ae_, g)


def xi_label_kernel(ordering: IPtr, as_: IPtr, ae_: IPtr, na: IPtr, labels: IPtr, m: Int32):
    var q = _tid()
    if q < Int(m):
        xi_label_cell(ordering, as_, ae_, na, labels, q)


struct _XiPool(Movable):
    """The driver's buffers, zeroed on creation, alive until it returns."""

    var ctx: DeviceContext
    var i: List[DeviceBuffer[DType.int32]]
    var f: List[DeviceBuffer[DType.float32]]

    def __init__(out self, ctx: DeviceContext):
        self.ctx = ctx
        self.i = List[DeviceBuffer[DType.int32]]()
        self.f = List[DeviceBuffer[DType.float32]]()

    def ib(mut self, n: Int) raises -> Int:
        var buf = self.ctx.enqueue_create_buffer[DType.int32](n if n > 0 else 1)
        self.ctx.enqueue_memset(buf, Int32(0))
        self.i.append(buf^)
        return len(self.i) - 1

    def fb(mut self, n: Int) raises -> Int:
        var buf = self.ctx.enqueue_create_buffer[DType.float32](n if n > 0 else 1)
        self.ctx.enqueue_memset(buf, Float32(0))
        self.f.append(buf^)
        return len(self.f) - 1

    def ip(mut self, s: Int) -> IPtr:
        return self.i[s].unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]()

    def fp(mut self, s: Int) -> FPtr:
        return self.f[s].unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]()

    def scan_into(mut self, f: IPtr, m: Int, part: Int, ex: Int, tot: Int) raises:
        """The exclusive integer scan of m flags into ex, the total into tot[0]."""
        var nb = (m + SCAN_PER - 1) // SCAN_PER
        if nb < 1:
            nb = 1
        self.ctx.enqueue_function[scan_part_kernel](f, Int32(m), self.ip(part), grid_dim=nb, block_dim=RTPB)
        self.ctx.enqueue_function[scan_out_kernel](
            f, Int32(m), self.ip(part), Int32(nb), self.ip(ex), self.ip(tot), grid_dim=nb, block_dim=RTPB,
        )

    def scan(mut self, f: IPtr, m: Int) raises -> Tuple[Int, Int]:
        """(ex, tot) slots of a new scan."""
        var nb = (m + SCAN_PER - 1) // SCAN_PER
        var part = self.ib(nb if nb > 0 else 1)
        var ex = self.ib(m)
        var tot = self.ib(1)
        self.scan_into(f, m, part, ex, tot)
        return (ex, tot)

    def read1(mut self, s: Int) raises -> Int:
        var h = List[Int32](length=1, fill=Int32(0))
        var view = self.i[s].create_sub_buffer[DType.int32](0, 1)
        self.ctx.enqueue_copy(dst_ptr=h.unsafe_ptr(), src_buf=view)
        self.ctx.synchronize()
        return Int(h[0])

    def table(mut self, src: FPtr, m: Int, is_max: Bool) raises -> Int:
        """A sparse min / max table over m values (level k at k * m)."""
        var lv = xi_levels(m)
        var t = self.fb(lv * m)
        self.ctx.enqueue_function[xi_st0_kernel](src, self.fp(t), Int32(m), grid_dim=pgrid(m), block_dim=PTPB)
        for k in range(1, lv):
            self.ctx.enqueue_function[xi_st_kernel](
                self.fp(t), Int32(m), Int32(k), Int32(1 if is_max else 0), grid_dim=pgrid(m), block_dim=PTPB,
            )
        return t

    def mark(mut self, on: IPtr, jmp: Int, m: Int) raises:
        """Pointer doubling over m + 1 nodes (node m the sentinel)."""
        var lv = xi_levels(m + 1)
        var a = jmp
        var b = self.ib(m + 1)
        for lvl in range(lv):
            self.ctx.enqueue_function[xi_mark_kernel](
                on, self.ip(a), Int32(m + 1), grid_dim=pgrid(m + 1), block_dim=PTPB,
            )
            if lvl + 1 < lv:
                self.ctx.enqueue_function[xi_jump_kernel](
                    self.ip(a), self.ip(b), Int32(m + 1), grid_dim=pgrid(m + 1), block_dim=PTPB,
                )
                var s = a
                a = b
                b = s


def optics_xi_device(
    ctx: DeviceContext, ordering: IPtr, reach: FPtr, pred: IPtr, n: Int, xc: Float32, min_samples: Int,
    min_cluster_size: Int, pc: Bool, labels: IPtr,
) raises -> List[Int32]:
    """`_xi_cluster` and `_extract_xi_labels` on the device: labels (point
    order) into `labels`; returns the clusters (start, end) flattened, in the
    reference's order. `optics_xi_cells.optics_xi_host` is the host column."""
    var P = _XiPool(ctx)
    var m1 = n + 1
    var plot = P.fb(m1)
    var pos = P.ib(m1)
    ctx.enqueue_function[xi_plot_kernel](
        reach, ordering, Int32(n), P.fp(plot), P.ip(pos), Int32(m1), grid_dim=pgrid(m1), block_dim=PTPB,
    )
    var f5 = P.ib(XB_N * n + 1)
    ctx.enqueue_function[xi_flag_kernel](P.fp(plot), Int32(n), xc, P.ip(f5), Int32(m1), grid_dim=pgrid(m1), block_dim=PTPB)
    var s5 = P.scan(P.ip(f5), XB_N * n + 1)
    var p5 = P.ib(XB_N * n + 1)
    ctx.enqueue_function[xi_scatter_kernel](
        P.ip(f5), P.ip(s5[0]), P.ip(p5), Int32(n), Int32(XB_N * n), grid_dim=pgrid(XB_N * n), block_dim=PTPB,
    )
    var b2 = P.ib(2 * n + 1)
    ctx.enqueue_function[xi_brk_kernel](
        P.ip(f5), P.ip(s5[0]), P.ip(p5), Int32(n), Int32(min_samples), P.ip(b2), Int32(m1),
        grid_dim=pgrid(m1), block_dim=PTPB,
    )
    var sb = P.scan(P.ip(b2), 2 * n + 1)
    var pb = P.ib(2 * n + 1)
    ctx.enqueue_function[xi_scatter_kernel](
        P.ip(b2), P.ip(sb[0]), P.ip(pb), Int32(n), Int32(2 * n), grid_dim=pgrid(2 * n), block_dim=PTPB,
    )
    var end_ = P.ib(m1)
    var jmp = P.ib(m1)
    var on = P.ib(m1)
    ctx.enqueue_function[xi_region_kernel](
        P.ip(f5), P.ip(s5[0]), P.ip(p5), P.ip(sb[0]), P.ip(pb), Int32(n), P.ip(end_), P.ip(jmp), P.ip(on),
        Int32(m1), grid_dim=pgrid(m1), block_dim=PTPB,
    )
    P.mark(P.ip(on), jmp, n)
    var so = P.scan(P.ip(on), n)
    var ev = P.ib(m1)
    ctx.enqueue_function[xi_scatter_kernel](
        P.ip(on), P.ip(so[0]), P.ip(ev), Int32(n), Int32(n), grid_dim=pgrid(n), block_dim=PTPB,
    )
    var pmax = P.table(P.fp(plot), m1, True)
    var pmin = P.table(P.fp(plot), m1, False)
    var w = P.fb(n)
    ctx.enqueue_function[xi_w_kernel](
        pred, ordering, P.ip(pos), P.fp(w), Int32(n), grid_dim=pgrid(n), block_dim=PTPB,
    )
    var wmax = P.table(P.fp(w), n, True)
    var mib = P.fb(m1)
    var isd = P.ib(m1)
    ctx.enqueue_function[xi_event_kernel](
        P.ip(ev), P.ip(end_), P.ip(f5), Int32(n), P.fp(pmax), P.fp(mib), P.ip(isd), P.ip(so[1]),
        grid_dim=pgrid(n), block_dim=PTPB,
    )
    var sd = P.scan(P.ip(isd), n)
    var dl = P.ib(m1)
    var ul = P.ib(m1)
    ctx.enqueue_function[xi_split_kernel](
        P.ip(isd), P.ip(sd[0]), P.ip(dl), P.ip(ul), P.ip(so[1]), grid_dim=pgrid(n), block_dim=PTPB,
    )
    # the grid sizes: the event and down-event counts (scalars, one wait)
    var hv = List[Int32](length=2, fill=Int32(0))
    var v0 = P.i[so[1]].create_sub_buffer[DType.int32](0, 1)
    var v1 = P.i[sd[1]].create_sub_buffer[DType.int32](0, 1)
    ctx.enqueue_copy(dst_ptr=hv.unsafe_ptr(), src_buf=v0)
    ctx.enqueue_copy(dst_ptr=hv.unsafe_ptr() + 1, src_buf=v1)
    ctx.synchronize()
    var ne = Int(hv[0])
    var nd = Int(hv[1])
    var nu = ne - nd
    var ne1 = ne if ne > 0 else 1
    var mibt = P.table(P.fp(mib), ne1, True)
    var c = 0
    var cl = -1
    var rw = -1
    if nd > 0 and nu > 0:
        var rows = XI_PAIR_CAP // nd
        if rows < 1:
            rows = 1
        if rows > nu:
            rows = nu
        var mmax = rows * nd
        var flag = P.ib(mmax)
        var cs_ = P.ib(mmax)
        var ce_ = P.ib(mmax)
        var nbm = (mmax + SCAN_PER - 1) // SCAN_PER
        var part = P.ib(nbm if nbm > 0 else 1)
        var ex = P.ib(mmax)
        var tot = P.ib(1)
        var counts = List[Int]()
        var r0 = 0
        while r0 < nu:
            var rr = rows if r0 + rows <= nu else nu - r0
            var m = rr * nd
            ctx.enqueue_function[xi_pair_kernel](
                P.ip(ul), P.ip(dl), P.ip(ev), P.ip(end_), P.fp(plot), P.fp(pmin), P.fp(wmax), Int32(n),
                P.fp(mibt), Int32(ne1), xc, Int32(min_cluster_size), Int32(1 if pc else 0), Int32(nd),
                Int32(r0), P.ip(flag), P.ip(cs_), P.ip(ce_), Int32(m), grid_dim=pgrid(m), block_dim=PTPB,
            )
            P.scan_into(P.ip(flag), m, part, ex, tot)
            var k = P.read1(tot)
            counts.append(k)
            c += k
            r0 += rr
        cl = P.ib(2 * c)
        rw = P.ib(c)
        var one = len(counts) == 1
        var off = 0
        r0 = 0
        var q = 0
        while r0 < nu:
            var rr = rows if r0 + rows <= nu else nu - r0
            var m = rr * nd
            if not one:
                ctx.enqueue_function[xi_pair_kernel](
                    P.ip(ul), P.ip(dl), P.ip(ev), P.ip(end_), P.fp(plot), P.fp(pmin), P.fp(wmax), Int32(n),
                    P.fp(mibt), Int32(ne1), xc, Int32(min_cluster_size), Int32(1 if pc else 0), Int32(nd),
                    Int32(r0), P.ip(flag), P.ip(cs_), P.ip(ce_), Int32(m), grid_dim=pgrid(m), block_dim=PTPB,
                )
                P.scan_into(P.ip(flag), m, part, ex, tot)
            ctx.enqueue_function[xi_pair_out_kernel](
                P.ip(flag), P.ip(ex), P.ip(cs_), P.ip(ce_), Int32(r0), Int32(nd), Int32(off), P.ip(cl), P.ip(rw),
                Int32(m), grid_dim=pgrid(m), block_dim=PTPB,
            )
            off += counts[q]
            q += 1
            r0 += rr
    if c > 0:
        var ff = P.ib(c)
        var fsf = P.fb(c)
        ctx.enqueue_function[xi_first_kernel](P.ip(rw), P.ip(ff), P.fp(fsf), Int32(c), grid_dim=pgrid(c), block_dim=PTPB)
        var sf = P.scan(P.ip(ff), c)
        var fs_ = P.ib(c)
        var fe_ = P.ib(c)
        ctx.enqueue_function[xi_first_out_kernel](
            P.ip(ff), P.ip(sf[0]), P.ip(cl), P.ip(fs_), P.ip(fe_), P.fp(fsf), Int32(c),
            grid_dim=pgrid(c), block_dim=PTPB,
        )
        var fsmax = P.table(P.fp(fsf), c, True)
        var gj = P.ib(c + 1)
        var gon = P.ib(c + 1)
        ctx.enqueue_function[xi_gnext_kernel](
            P.ip(fe_), P.fp(fsmax), Int32(c), P.ip(sf[1]), P.ip(gj), P.ip(gon), grid_dim=pgrid(c + 1), block_dim=PTPB,
        )
        P.mark(P.ip(gon), gj, c)
        var sa = P.scan(P.ip(gon), c)
        var as_ = P.ib(c)
        var ae_ = P.ib(c)
        ctx.enqueue_function[xi_acc_out_kernel](
            P.ip(gon), P.ip(sa[0]), P.ip(fs_), P.ip(fe_), P.ip(as_), P.ip(ae_), Int32(c),
            grid_dim=pgrid(c), block_dim=PTPB,
        )
        ctx.enqueue_function[xi_label_kernel](
            ordering, P.ip(as_), P.ip(ae_), P.ip(sa[1]), labels, Int32(n), grid_dim=pgrid(n), block_dim=PTPB,
        )
    else:
        var z = P.ib(1)
        ctx.enqueue_function[xi_label_kernel](
            ordering, P.ip(z), P.ip(z), P.ip(z), labels, Int32(n), grid_dim=pgrid(n), block_dim=PTPB,
        )
    var out = List[Int32](length=2 * c, fill=Int32(0))
    if c > 0:
        var view = P.i[cl].create_sub_buffer[DType.int32](0, 2 * c)
        ctx.enqueue_copy(dst_ptr=out.unsafe_ptr(), src_buf=view)
    ctx.synchronize()
    _ = P^
    return out^
