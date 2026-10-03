# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
# SHIPS: compiled into both x_cluster bindings (no std.gpu, no max.gpu).
"""OPTICS xi extraction as parallel cells (lane cgr4-device-optim-optics,
2026-10-03): one source for the device kernels
(`x_cluster/optics_xi_device.mojo`) and the host column's loops
(`optics_xi_host` below). Reference: scikit-learn `sklearn/cluster/_optics.py`
(`_xi_cluster`, `_extend_region`, `_update_filter_sdas`,
`_correct_predecessor`, `_extract_xi_labels`).

The reference walks the reachability plot once, a state machine. Here every
step of that walk is a function of an index, so it runs one thread an index
and every result is an integer or an exact compare: any launch shape and any
thread interleaving give the same words, and the host column's loops (same
cells, same phases, same order) give them too.

1. Flags, one thread a plot index: steep down / steep up, up / down, from
   plot[i] and plot[i + 1] (`xi_flag_cell`).
2. `_extend_region` as scans. The walk from a steep point s stops at the
   first later index that is xward, or that ends a run of more than
   min_samples consecutive neutral points (the run length is i minus the last
   non-neutral index at or before i, a scan); the region end is the last
   steep index before the stop (a scan). So every steep point's region end
   is known at once (`xi_brk_cell`, `xi_region_cell`).
3. The visited steep points. The walk skips steep points inside the last
   region: from s the next visited point is the first steep index after
   end(s) (`nxt`). The visited set is the chain from the first steep point;
   it is marked by pointer doubling (level k marks J_k(marked), then
   J_{k+1} = J_k o J_k; log2 n levels; a mark is only ever set, so any
   interleaving ends in the same set, `xi_mark_cell`, `xi_jump_cell`).
4. Events. The visited points in order; event e's mib is the max of the plot
   over its gap (end of event e-1, s_e] (a sparse max table).
5. The SDA list as range maxima. An SDA made at down event d is alive at a
   later event u iff every mib since d passed the filter; the filter is
   monotone in the mib, so alive iff rmax(mib, d+1..u) is finite and
   <= plot[start_d] * xc, and the SDA's mib at u IS that range max. So every
   (up event, down event) pair is decided at once (`xi_pair_cell`): the
   cluster start / end walks and the predecessor correction are "first /
   last index in a range whose value passes a threshold", binary lifting over
   sparse min / max tables (O(log n) a pair, no loop over the plot).
   Pairs run row by row (up events), the down events of a row in reverse
   (the reference's `reversed(sdas)`); the valid ones are compacted by a scan.
6. Labels (`_extract_xi_labels`). Within one up event's clusters the starts
   fall and every end is past every earlier start, so all of them overlap the
   row's first; and every cluster of an earlier row ends before this row's
   up region. So only a row's first cluster can take a label, and it does iff
   its start is past the end of the last labelled first. That is a chain
   again (next(g) = the first later first whose start passes g's end; a
   sparse max table and binary lifting) marked by the same pointer doubling;
   the labelled clusters are disjoint and sorted, and each point finds its
   own by a binary search.

THE RATIOS. The reference compares Float64 ratios of the Float32 plot. Here
xc = 1 - xi is rounded to Float32 once and every test is the exact one in
real numbers: `ratio <= xc` is `a <= b * xc`, `ratio >= 1 / xc` is
`b <= a * xc`, decided by the exact product (`two_prod`: the rounded
product and its error) with no division (`xi_le_mul`); up / down are
`a < b` / `a > b`; 0/0 and inf/inf compare false, as the reference's NaN
does. A BIT CHANGE against the Float64 walk only where a ratio sits within a
Float64 rounding of xc or 1/xc, or within xc's Float32 rounding.

Positions (n < 2^24) ride in Float32 sparse tables exactly."""
from x_cluster.bodies import FPtr, IPtr
from x_linear.ff import two_prod

#: flag blocks of `f5` (each n long, then one trailing zero)
comptime XB_SD = 0  # steep down
comptime XB_NND = 1  # not neutral for a down region (steep down or up)
comptime XB_SU = 2  # steep up
comptime XB_NNU = 3  # not neutral for an up region (steep up or down)
comptime XB_STEEP = 4  # steep either way
comptime XB_N = 5
#: lifting modes: the value passes when v <= thr, v < thr (min tables),
#: v >= thr, v > thr (max tables)
comptime XF_LE = 0
comptime XF_LT = 1
comptime XF_GE = 2
comptime XF_GT = 3
#: pairs a chunk of the pair grid holds at most
comptime XI_PAIR_CAP = 1 << 22


@always_inline
def xi_inf() -> Float32:
    return Float32.MAX * Float32(2)


@always_inline
def xi_le_mul(v: Float32, u: Float32, xc: Float32) -> Bool:
    """v <= u * xc in real numbers (v, u >= 0 or +inf; 0 < xc < 1): the
    product's rounded value and its exact error decide a tie."""
    var t = two_prod(u, xc)
    if t.hi == xi_inf():
        return True
    if v < t.hi:
        return True
    if v > t.hi:
        return False
    return t.lo >= Float32(0)


@always_inline
def xi_lg(m: Int) -> Int:
    """floor(log2(m)), m >= 1."""
    var k = 0
    while (2 << k) <= m:
        k += 1
    return k


@always_inline
def xi_levels(m: Int) -> Int:
    """Sparse-table levels of an m-long array: 2^levels > m."""
    return xi_lg(m if m > 0 else 1) + 1


@always_inline
def xi_cnt(ex: IPtr, n: Int, k: Int, j: Int) -> Int:
    """The flags of block k in [0, j), j in [0, n]."""
    return Int(ex[k * n + j]) - Int(ex[k * n])


@always_inline
def xi_last(ex: IPtr, pos: IPtr, n: Int, k: Int, j: Int) -> Int:
    """The last flagged index <= j of block k, or -1."""
    if j < 0:
        return -1
    var c = xi_cnt(ex, n, k, j + 1)
    return -1 if c == 0 else Int(pos[k * n + c - 1])


@always_inline
def xi_next(ex: IPtr, pos: IPtr, n: Int, k: Int, j: Int) -> Int:
    """The first flagged index >= j of block k, or n."""
    var c = xi_cnt(ex, n, k, j)
    return Int(pos[k * n + c]) if c < xi_cnt(ex, n, k, n) else n


@always_inline
def xi_scatter_cell(f: IPtr, ex: IPtr, pos: IPtr, n: Int, i: Int):
    """Compaction: block k = i // n's flagged indices, in order, at pos[k * n ..]."""
    if f[i] != 0:
        var k = i // n
        pos[k * n + Int(ex[i]) - Int(ex[k * n])] = Int32(i - k * n)


@always_inline
def xi_plot_cell(reach: FPtr, ordering: IPtr, n: Int, plot: FPtr, pos: IPtr, i: Int):
    """plot[i] = reach[ordering[i]] (i < n), plot[n] = inf; pos the inverse ordering."""
    if i < n:
        var o = Int(ordering[i])
        plot[i] = reach[o]
        pos[o] = Int32(i)
    else:
        plot[n] = xi_inf()


@always_inline
def xi_flag_cell(plot: FPtr, n: Int, xc: Float32, f5: IPtr, i: Int):
    if i >= n:
        f5[XB_N * n] = Int32(0)
        return
    var a = plot[i]
    var b = plot[i + 1]
    var nan = (a == Float32(0) and b == Float32(0)) or (a == xi_inf() and b == xi_inf())
    var up = a < b
    var dn = a > b
    var su = (not nan) and xi_le_mul(a, b, xc)
    var sd = (not nan) and xi_le_mul(b, a, xc)
    f5[XB_SD * n + i] = Int32(1) if sd else Int32(0)
    f5[XB_NND * n + i] = Int32(1) if (sd or up) else Int32(0)
    f5[XB_SU * n + i] = Int32(1) if su else Int32(0)
    f5[XB_NNU * n + i] = Int32(1) if (su or dn) else Int32(0)
    f5[XB_STEEP * n + i] = Int32(1) if (sd or su) else Int32(0)


@always_inline
def xi_brk_cell(f5: IPtr, ex5: IPtr, p5: IPtr, n: Int, min_samples: Int, b2: IPtr, i: Int):
    """b2 block 0 (down) / 1 (up): index i stops a region walk (xward, or
    the (min_samples + 1)-th neutral in a row)."""
    if i >= n:
        b2[2 * n] = Int32(0)
        return
    for dir in range(2):
        var sb = XB_SD if dir == 0 else XB_SU
        var nb = XB_NND if dir == 0 else XB_NNU
        var steep = f5[sb * n + i] != 0
        var nn = f5[nb * n + i] != 0
        var brk = False
        if nn:
            brk = not steep
        else:
            brk = i - xi_last(ex5, p5, n, nb, i) > min_samples
        b2[dir * n + i] = Int32(1) if brk else Int32(0)


@always_inline
def xi_region_cell(
    f5: IPtr, ex5: IPtr, p5: IPtr, exb: IPtr, pb: IPtr, n: Int, end_: IPtr, jmp: IPtr, on: IPtr, i: Int
):
    """Steep i: its region end (`_extend_region`) and the next visited steep
    point. Index n: the sentinel, and the first steep point marked."""
    if i >= n:
        jmp[n] = Int32(n)
        var head = xi_next(ex5, p5, n, XB_STEEP, 0)
        if head < n:
            on[head] = Int32(1)
        return
    if f5[XB_STEEP * n + i] == 0:
        end_[i] = Int32(i)
        jmp[i] = Int32(n)
        return
    var dir = 0 if f5[XB_SD * n + i] != 0 else 1
    var stop = xi_next(exb, pb, n, dir, i + 1)
    var e = xi_last(ex5, p5, n, XB_SD if dir == 0 else XB_SU, stop - 1)
    end_[i] = Int32(e)
    jmp[i] = Int32(xi_next(ex5, p5, n, XB_STEEP, e + 1))


@always_inline
def xi_mark_cell(on: IPtr, jmp: IPtr, i: Int):
    """Pointer doubling: a marked node marks its jump (marks are only set)."""
    if on[i] != 0:
        on[Int(jmp[i])] = Int32(1)


@always_inline
def xi_jump_cell(jmp: IPtr, dst: IPtr, i: Int):
    dst[i] = jmp[Int(jmp[i])]


@always_inline
def xi_st0_cell(src: FPtr, t: FPtr, i: Int):
    t[i] = src[i]


@always_inline
def xi_st_cell(t: FPtr, m: Int, k: Int, is_max: Bool, i: Int):
    """Level k of a sparse table over m values: the min / max of [i, i + 2^k)."""
    var half = 1 << (k - 1)
    var a = t[(k - 1) * m + i]
    var j = i + half
    var b = t[(k - 1) * m + (j if j < m else m - 1)]
    var r: Float32
    if is_max:
        r = a if a >= b else b
    else:
        r = a if a <= b else b
    t[k * m + i] = r


@always_inline
def xi_rmax(t: FPtr, m: Int, l: Int, r: Int) -> Float32:
    """max over [l, r] of a max table."""
    var k = xi_lg(r - l + 1)
    var a = t[k * m + l]
    var b = t[k * m + r - (1 << k) + 1]
    return a if a >= b else b


@always_inline
def _xi_pass(v: Float32, thr: Float32, mode: Int) -> Bool:
    if mode == XF_LE:
        return v <= thr
    if mode == XF_LT:
        return v < thr
    if mode == XF_GE:
        return v >= thr
    return v > thr


@always_inline
def xi_first(t: FPtr, m: Int, l: Int, r: Int, thr: Float32, mode: Int) -> Int:
    """The first j in [l, r] whose value passes (binary lifting over the
    table), or -1."""
    if l > r:
        return -1
    var pos = l
    var k = xi_levels(m) - 1
    while k >= 0:
        var w = 1 << k
        if pos + w - 1 <= r and not _xi_pass(t[k * m + pos], thr, mode):
            pos += w
        k -= 1
    return pos if pos <= r else -1


@always_inline
def xi_last_pass(t: FPtr, m: Int, l: Int, r: Int, thr: Float32, mode: Int) -> Int:
    """The last j in [l, r] whose value passes, or -1."""
    if l > r:
        return -1
    var pos = r
    var k = xi_levels(m) - 1
    while k >= 0:
        var w = 1 << k
        if pos - w + 1 >= l and not _xi_pass(t[k * m + pos - w + 1], thr, mode):
            pos -= w
        k -= 1
    return pos if pos >= l else -1


@always_inline
def xi_w_cell(pred: IPtr, ordering: IPtr, pos: IPtr, w: FPtr, i: Int):
    """The plot position of i's predecessor when it is before i, else -1."""
    var p = Int(pred[Int(ordering[i])])
    var r = Float32(-1)
    if p >= 0:
        var pp = Int(pos[p])
        if pp < i:
            r = Float32(pp)
    w[i] = r


@always_inline
def xi_event_cell(ev: IPtr, end_: IPtr, f5: IPtr, n: Int, pmax: FPtr, mib: FPtr, isd: IPtr, e: Int):
    """Event e (the e-th visited steep point): its mib (the plot max over its
    gap) and its kind."""
    var s = Int(ev[e])
    var lo = 0 if e == 0 else Int(end_[Int(ev[e - 1])]) + 1
    mib[e] = xi_rmax(pmax, n + 1, lo, s)
    isd[e] = Int32(1) if f5[XB_SD * n + s] != 0 else Int32(0)


@always_inline
def xi_split_cell(isd: IPtr, exd: IPtr, dl: IPtr, ul: IPtr, e: Int):
    var r = Int(exd[e])
    if isd[e] != 0:
        dl[r] = Int32(e)
    else:
        ul[e - r] = Int32(e)


@always_inline
def xi_pair_cell(
    ul: IPtr, dl: IPtr, ev: IPtr, end_: IPtr, plot: FPtr, pmin: FPtr, wmax: FPtr, n: Int,
    mibt: FPtr, ne: Int, xc: Float32, min_cluster_size: Int, pc: Bool, nd: Int, r0: Int,
    flag: IPtr, cs_: IPtr, ce_: IPtr, t: Int,
):
    """(up event row r0 + t // nd, down event nd - 1 - t % nd): the
    reference's SDA check and cluster for that pair, flag 1 when it is kept."""
    flag[t] = Int32(0)
    var ur = r0 + t // nd
    var dr = nd - 1 - t % nd
    var ue = Int(ul[ur])
    var de = Int(dl[dr])
    if de >= ue:
        return
    var rm = xi_rmax(mibt, ne, de + 1, ue)
    if rm == xi_inf():
        return
    var ds = Int(ev[de])
    var dend = Int(end_[ds])
    var us = Int(ev[ue])
    var uend = Int(end_[us])
    var dmax = plot[ds]
    if not xi_le_mul(rm, dmax, xc):
        return
    var c_start = ds
    var c_end = uend
    var thr = plot[c_end + 1]
    if not xi_le_mul(rm, thr, xc):
        return
    var m1 = n + 1
    if xi_le_mul(thr, dmax, xc):
        var j = xi_first(pmin, m1, ds + 1, dend, thr, XF_LE)
        c_start = j - 1 if j >= 0 else dend
    elif xi_le_mul(dmax, thr, xc):
        var j = xi_last_pass(pmin, m1, us, uend - 1, dmax, XF_LE)
        c_end = j + 1 if j >= 0 else us
    if pc:
        if c_start >= c_end:
            return
        var a = xi_last_pass(pmin, m1, c_start + 1, c_end, plot[c_start], XF_LT)
        var b = xi_last_pass(wmax, n, c_start + 1, c_end, Float32(c_start), XF_GE)
        var c = a if a >= b else b
        if c < 0:
            return
        c_end = c
    if c_end - c_start + 1 < min_cluster_size:
        return
    if c_start > dend:
        return
    if c_end < us:
        return
    flag[t] = Int32(1)
    cs_[t] = Int32(c_start)
    ce_[t] = Int32(c_end)


@always_inline
def xi_pair_out_cell(
    flag: IPtr, ex: IPtr, cs_: IPtr, ce_: IPtr, r0: Int, nd: Int, off: Int, cl: IPtr, rw: IPtr, t: Int
):
    """The kept pairs of a chunk, in order, at off.. of the cluster list."""
    if flag[t] != 0:
        var k = off + Int(ex[t])
        cl[2 * k] = cs_[t]
        cl[2 * k + 1] = ce_[t]
        rw[k] = Int32(r0 + t // nd)


@always_inline
def xi_first_cell(rw: IPtr, ff: IPtr, fsf: FPtr, k: Int):
    """Cluster k opens its row; fsf (the firsts' starts) starts at -1."""
    ff[k] = Int32(1) if (k == 0 or rw[k] != rw[k - 1]) else Int32(0)
    fsf[k] = Float32(-1)


@always_inline
def xi_first_out_cell(ff: IPtr, exf: IPtr, cl: IPtr, fs_: IPtr, fe_: IPtr, fsf: FPtr, k: Int):
    if ff[k] != 0:
        var g = Int(exf[k])
        fs_[g] = cl[2 * k]
        fe_[g] = cl[2 * k + 1]
        fsf[g] = Float32(Int(cl[2 * k]))


@always_inline
def xi_gnext_cell(fe_: IPtr, fsmax: FPtr, c: Int, ng: IPtr, jmp: IPtr, on: IPtr, g: Int):
    """First g: the next first whose start is past g's end (c: none); the
    head marked. g == c: the sentinel."""
    var G = Int(ng[0])
    if g >= c:
        jmp[c] = Int32(c)
        if G > 0:
            on[0] = Int32(1)
        return
    if g >= G:
        jmp[g] = Int32(c)
        return
    var h = xi_first(fsmax, c, g + 1, c - 1, Float32(Int(fe_[g])), XF_GT)
    jmp[g] = Int32(h if h >= 0 else c)


@always_inline
def xi_acc_out_cell(on: IPtr, exa: IPtr, fs_: IPtr, fe_: IPtr, as_: IPtr, ae_: IPtr, g: Int):
    if on[g] != 0:
        var r = Int(exa[g])
        as_[r] = fs_[g]
        ae_[r] = fe_[g]


@always_inline
def xi_label_cell(ordering: IPtr, as_: IPtr, ae_: IPtr, na: IPtr, labels: IPtr, q: Int):
    """Point at plot position q: the labelled cluster holding it (binary
    search over the sorted, disjoint labelled clusters), or -1."""
    var A = Int(na[0])
    var lo = 0
    var hi = A
    while lo < hi:
        var mid = (lo + hi) // 2
        if Int(as_[mid]) <= q:
            lo = mid + 1
        else:
            hi = mid
    var lab = Int32(-1)
    if lo > 0 and q <= Int(ae_[lo - 1]):
        lab = Int32(lo - 1)
    labels[Int(ordering[q])] = lab


# ---------------------------------------------------------------- host column
def _fpp(mut v: List[Float32]) -> FPtr:
    return v.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]()


def _ipp(mut v: List[Int32]) -> IPtr:
    return v.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]()


def _scan_host(f: IPtr, m: Int, mut ex: List[Int32]) -> Int:
    """Exclusive scan of m flags (integers: the device's words)."""
    ex = List[Int32](length=m if m > 0 else 1, fill=Int32(0))
    var s = 0
    for i in range(m):
        ex[i] = Int32(s)
        s += Int(f[i])
    return s


def _table_host(src: FPtr, m: Int, is_max: Bool) -> List[Float32]:
    var lv = xi_levels(m)
    var t = List[Float32](length=lv * m if m > 0 else 1, fill=Float32(0))
    var pt = _fpp(t)
    for i in range(m):
        xi_st0_cell(src, pt, i)
    for k in range(1, lv):
        for i in range(m):
            xi_st_cell(pt, m, k, is_max, i)
    return t^


def _mark_host(mut on: List[Int32], mut jmp: List[Int32], m: Int):
    """The device's pointer doubling over m + 1 nodes (node m the sentinel)."""
    var lv = xi_levels(m + 1)
    var j2 = List[Int32](length=m + 1, fill=Int32(0))
    for lvl in range(lv):
        var po = _ipp(on)
        var pj = _ipp(jmp)
        for i in range(m + 1):
            xi_mark_cell(po, pj, i)
        if lvl + 1 < lv:
            var pd = _ipp(j2)
            for i in range(m + 1):
                xi_jump_cell(pj, pd, i)
            for i in range(m + 1):
                jmp[i] = j2[i]


def optics_xi_host(
    ordering: IPtr, reach: FPtr, pred: IPtr, n: Int, xc: Float32, min_samples: Int,
    min_cluster_size: Int, pc: Bool, labels: IPtr,
) -> List[Int32]:
    """The host column: the device's phases (`optics_xi_device`), each a loop
    over the same cells; returns the clusters (start, end) flattened."""
    var m1 = n + 1
    var plot = List[Float32](length=m1, fill=Float32(0))
    var pos = List[Int32](length=m1, fill=Int32(0))
    for i in range(m1):
        xi_plot_cell(reach, ordering, n, _fpp(plot), _ipp(pos), i)
    var f5 = List[Int32](length=XB_N * n + 1, fill=Int32(0))
    for i in range(m1):
        xi_flag_cell(_fpp(plot), n, xc, _ipp(f5), i)
    var ex5 = List[Int32]()
    _ = _scan_host(_ipp(f5), XB_N * n + 1, ex5)
    var p5 = List[Int32](length=XB_N * n + 1, fill=Int32(0))
    for i in range(XB_N * n):
        xi_scatter_cell(_ipp(f5), _ipp(ex5), _ipp(p5), n, i)
    var b2 = List[Int32](length=2 * n + 1, fill=Int32(0))
    for i in range(m1):
        xi_brk_cell(_ipp(f5), _ipp(ex5), _ipp(p5), n, min_samples, _ipp(b2), i)
    var exb = List[Int32]()
    _ = _scan_host(_ipp(b2), 2 * n + 1, exb)
    var pb = List[Int32](length=2 * n + 1, fill=Int32(0))
    for i in range(2 * n):
        xi_scatter_cell(_ipp(b2), _ipp(exb), _ipp(pb), n, i)
    var end_ = List[Int32](length=m1, fill=Int32(0))
    var jmp = List[Int32](length=m1, fill=Int32(0))
    var on = List[Int32](length=m1, fill=Int32(0))
    for i in range(m1):
        xi_region_cell(_ipp(f5), _ipp(ex5), _ipp(p5), _ipp(exb), _ipp(pb), n, _ipp(end_), _ipp(jmp), _ipp(on), i)
    _mark_host(on, jmp, n)
    var exo = List[Int32]()
    var ne = _scan_host(_ipp(on), n, exo)
    var ev = List[Int32](length=m1, fill=Int32(0))
    for i in range(n):
        xi_scatter_cell(_ipp(on), _ipp(exo), _ipp(ev), n, i)
    var pmax = _table_host(_fpp(plot), m1, True)
    var pmin = _table_host(_fpp(plot), m1, False)
    var w = List[Float32](length=n if n > 0 else 1, fill=Float32(-1))
    for i in range(n):
        xi_w_cell(pred, ordering, _ipp(pos), _fpp(w), i)
    var wmax = _table_host(_fpp(w), n, True)
    var mib = List[Float32](length=m1, fill=Float32(0))
    var isd = List[Int32](length=m1, fill=Int32(0))
    for e in range(ne):
        xi_event_cell(_ipp(ev), _ipp(end_), _ipp(f5), n, _fpp(pmax), _fpp(mib), _ipp(isd), e)
    var exd = List[Int32]()
    var nd = _scan_host(_ipp(isd), n, exd)
    var nu = ne - nd
    var dl = List[Int32](length=m1, fill=Int32(0))
    var ul = List[Int32](length=m1, fill=Int32(0))
    for e in range(ne):
        xi_split_cell(_ipp(isd), _ipp(exd), _ipp(dl), _ipp(ul), e)
    var mibt = _table_host(_fpp(mib), ne if ne > 0 else 1, True)
    var cl = List[Int32]()
    var rw = List[Int32]()
    if nd > 0 and nu > 0:
        var rows = XI_PAIR_CAP // nd
        if rows < 1:
            rows = 1
        var r0 = 0
        while r0 < nu:
            var rr = rows if r0 + rows <= nu else nu - r0
            var m = rr * nd
            var flag = List[Int32](length=m, fill=Int32(0))
            var cs_ = List[Int32](length=m, fill=Int32(0))
            var ce_ = List[Int32](length=m, fill=Int32(0))
            for t in range(m):
                xi_pair_cell(
                    _ipp(ul), _ipp(dl), _ipp(ev), _ipp(end_), _fpp(plot), _fpp(pmin), _fpp(wmax), n,
                    _fpp(mibt), ne if ne > 0 else 1, xc, min_cluster_size, pc, nd, r0,
                    _ipp(flag), _ipp(cs_), _ipp(ce_), t,
                )
            var ex = List[Int32]()
            var cnt = _scan_host(_ipp(flag), m, ex)
            var off = len(rw)
            cl.resize(2 * (off + cnt), Int32(0))
            rw.resize(off + cnt, Int32(0))
            for t in range(m):
                xi_pair_out_cell(_ipp(flag), _ipp(ex), _ipp(cs_), _ipp(ce_), r0, nd, off, _ipp(cl), _ipp(rw), t)
            r0 += rr
    var c = len(rw)
    if c > 0:
        var ff = List[Int32](length=c, fill=Int32(0))
        var fsf = List[Float32](length=c, fill=Float32(-1))
        for k in range(c):
            xi_first_cell(_ipp(rw), _ipp(ff), _fpp(fsf), k)
        var exf = List[Int32]()
        var ng = _scan_host(_ipp(ff), c, exf)
        var fs_ = List[Int32](length=c, fill=Int32(0))
        var fe_ = List[Int32](length=c, fill=Int32(0))
        for k in range(c):
            xi_first_out_cell(_ipp(ff), _ipp(exf), _ipp(cl), _ipp(fs_), _ipp(fe_), _fpp(fsf), k)
        var fsmax = _table_host(_fpp(fsf), c, True)
        var ngv = List[Int32](length=1, fill=Int32(ng))
        var gj = List[Int32](length=c + 1, fill=Int32(0))
        var gon = List[Int32](length=c + 1, fill=Int32(0))
        for g in range(c + 1):
            xi_gnext_cell(_ipp(fe_), _fpp(fsmax), c, _ipp(ngv), _ipp(gj), _ipp(gon), g)
        _mark_host(gon, gj, c)
        var exa = List[Int32]()
        var na = _scan_host(_ipp(gon), c, exa)
        var as_ = List[Int32](length=c, fill=Int32(0))
        var ae_ = List[Int32](length=c, fill=Int32(0))
        for g in range(c):
            xi_acc_out_cell(_ipp(gon), _ipp(exa), _ipp(fs_), _ipp(fe_), _ipp(as_), _ipp(ae_), g)
        var nav = List[Int32](length=1, fill=Int32(na))
        for q in range(n):
            xi_label_cell(ordering, _ipp(as_), _ipp(ae_), _ipp(nav), labels, q)
    else:
        var z = List[Int32](length=1, fill=Int32(0))
        for q in range(n):
            xi_label_cell(ordering, _ipp(z), _ipp(z), _ipp(z), labels, q)
    return cl^
