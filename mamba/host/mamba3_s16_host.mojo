# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
# SHIPS: compiled into the Mamba host binding through mamba/host/gen/mamba3_backward.mojo; product code.
"""The Mamba-3 S16 q/k/v backward on the host, at CPU speed (lane neural-cpu,
2026-09-28).

`mamba/impl/modules/mamba3_backward.mojo::mamba3_s16_qkv_backward_kernel` is
one GPU thread per output cell: a `d_q`/`d_k` cell (row, n) walks p over
HEADDIM and, inside, the chunk's earlier (for dq) or later (for dk) tokens; a
`d_v` cell (row, p) walks the later tokens, each through a serial dot product
over D_STATE. `tools/mamba_host_gen.py` substitutes this function for that
launch in the generated host file. For EVERY output cell it performs the
kernel's operations, on the same operands, in the same order:

  dq[n] = ftz(identical_mul_add(dy_i, ftz(identical_mul(ftz(identical_mul(k[tj,n], lv)), v[tj,p])), dq))
          p ascending, then j ascending, from +0.0
  dk[n] = ftz(identical_mul_add(dy[ti,p], ftz(identical_mul(ftz(identical_mul(q[ti,n], lv2)), v[token,p])), dk))
          p ascending, then i ascending, from +0.0
  dv[p] = ftz(identical_mul_add(dy[ti,p], ftz(identical_mul(dot_i, lv_i)), dv))
          i ascending, from +0.0; dot_i = the serial chain over n of
          ftz(identical_mul_add(q[ti,n], k[token,n], dot)) from +0.0

What changes, none of which moves a bit:
  - the D_STATE cells of one row's dq (and dk) advance together, one SIMD
    lane per n, and the HEADDIM cells of its dv one lane per p (each lane is
    its own cell's chain);
  - `dot_i` depends on the row and i but not on p, so it is computed once per
    (row, i) instead of once per (row, i, p), and likewise the first product
    of the dq and dk terms once per (row, j, n) and (row, i, n): the same
    operation on the same operands is the same value;
  - rows (batch, token, head) are split over host tasks
    (`core/host_parallel.mojo`); every output cell belongs to one row.

Under any tier other than IDENTICAL the lanes are not used and every cell
runs the scalar statement. The proof is the Mamba-3 lanes' CPU == GPU
agreement (the GPU runs the kernel itself) and the sabotage
`mamba/checks/sabotage/mamba3_s16_host_dk_descending.patch`.
"""
from std.math import max, min

from checks.numerics import ftz, identical_exp, identical_mul, identical_mul_add, identical_mul_add_simd
from core.host_lanes import F32V, HOST_FW, ftz_lanes, lanes_are_identical, pinned_mul_lanes
from core.host_parallel import host_parallelize
from core.host_predict_threads import host_predict_task_count
from mamba.checks.mamba3_fixture import M3_D_STATE, M3_HEADDIM

comptime _P = MutPointer[Float32, MutAnyOrigin]


def _s16_row(
    row: Int, d_q: _P, d_k: _P, d_v: _P, d_y: _P, q: _P, k: _P, v: _P, seg_l: _P,
    l: Int, nh: Int, qs: Int,
):
    """Every d_q, d_k (D_STATE cells) and d_v (HEADDIM cells) output of one
    (batch, token, head) row."""
    var h = row % nh
    var token = (row // nh) % l
    var bb = row // (nh * l)
    var chunk = token // qs
    var inner = token % qs
    var nc = (l + qs - 1) // qs
    var seg_base = ((bb * nc + chunk) * nh + h) * qs
    var qrow = row * M3_D_STATE
    # ---- d_q and d_k, one lane per n ----------------------------------
    # `ftz(identical_mul(k[tj, n], lv_j))` (dq) and `ftz(identical_mul(q[ti, n],
    # lv2_i))` (dk) do not depend on p: each is computed ONCE per (j, n) or
    # (i, n) into `tq` / `tk` and read by all HEADDIM p steps (the same
    # operation on the same operands is the same value).
    var n0 = 0
    comptime if lanes_are_identical:
        var tq = List[Float32](length=max(inner, 1) * M3_D_STATE, fill=Float32(0.0))
        var tk = List[Float32](length=qs * M3_D_STATE, fill=Float32(0.0))
        var tqp = tq.unsafe_ptr()
        var tkp = tk.unsafe_ptr()
        for j in range(inner):
            var tj = chunk * qs + j
            if tj < l:
                var lv = F32V(ftz(seg_l.unsafe_load((seg_base + inner) * qs + j)))
                var nn = 0
                while nn + HOST_FW <= M3_D_STATE:
                    var kv = ftz_lanes(k.unsafe_load[width=HOST_FW](((bb * l + tj) * nh + h) * M3_D_STATE + nn))
                    tqp.unsafe_store(j * M3_D_STATE + nn, ftz_lanes(pinned_mul_lanes(kv, lv)))
                    nn += HOST_FW
        for i in range(inner + 1, qs):
            var ti = chunk * qs + i
            if ti < l:
                var lv2 = F32V(ftz(seg_l.unsafe_load((seg_base + i) * qs + inner)))
                var nn = 0
                while nn + HOST_FW <= M3_D_STATE:
                    var qv = ftz_lanes(q.unsafe_load[width=HOST_FW](((bb * l + ti) * nh + h) * M3_D_STATE + nn))
                    tkp.unsafe_store(i * M3_D_STATE + nn, ftz_lanes(pinned_mul_lanes(qv, lv2)))
                    nn += HOST_FW
        while n0 + HOST_FW <= M3_D_STATE:
            var dq = F32V(0.0)
            var dk = F32V(0.0)
            for p in range(M3_HEADDIM):
                var dy_i = F32V(ftz(d_y.unsafe_load(((bb * l + token) * nh + h) * M3_HEADDIM + p)))
                for j in range(inner):
                    var tj = chunk * qs + j
                    if tj < l:
                        var vv = F32V(ftz(v.unsafe_load(((bb * l + tj) * nh + h) * M3_HEADDIM + p)))
                        var t = ftz_lanes(pinned_mul_lanes(tqp.unsafe_load[width=HOST_FW](j * M3_D_STATE + n0), vv))
                        dq = ftz_lanes(identical_mul_add_simd[HOST_FW](dy_i, t, dq))
                var vv2 = F32V(ftz(v.unsafe_load(((bb * l + token) * nh + h) * M3_HEADDIM + p)))
                for i in range(inner + 1, qs):
                    var ti = chunk * qs + i
                    if ti < l:
                        var dy = F32V(ftz(d_y.unsafe_load(((bb * l + ti) * nh + h) * M3_HEADDIM + p)))
                        var t2 = ftz_lanes(pinned_mul_lanes(tkp.unsafe_load[width=HOST_FW](i * M3_D_STATE + n0), vv2))
                        dk = ftz_lanes(identical_mul_add_simd[HOST_FW](dy, t2, dk))
            d_q.unsafe_store(qrow + n0, dq)
            d_k.unsafe_store(qrow + n0, dk)
            n0 += HOST_FW
        _ = tq^
        _ = tk^
    for n in range(n0, M3_D_STATE):
        var dq_s = Float32(0.0)
        var dk_s = Float32(0.0)
        for p in range(M3_HEADDIM):
            var dy_i = ftz(d_y.unsafe_load(((bb * l + token) * nh + h) * M3_HEADDIM + p))
            for j in range(inner):
                var tj = chunk * qs + j
                if tj < l:
                    var lv = ftz(seg_l.unsafe_load((seg_base + inner) * qs + j))
                    var kv = ftz(k.unsafe_load(((bb * l + tj) * nh + h) * M3_D_STATE + n))
                    var vv = ftz(v.unsafe_load(((bb * l + tj) * nh + h) * M3_HEADDIM + p))
                    dq_s = ftz(identical_mul_add(dy_i, ftz(identical_mul(ftz(identical_mul(kv, lv)), vv)), dq_s))
            for i in range(inner + 1, qs):
                var ti = chunk * qs + i
                if ti < l:
                    var dy = ftz(d_y.unsafe_load(((bb * l + ti) * nh + h) * M3_HEADDIM + p))
                    var lv2 = ftz(seg_l.unsafe_load((seg_base + i) * qs + inner))
                    var qv = ftz(q.unsafe_load(((bb * l + ti) * nh + h) * M3_D_STATE + n))
                    var vv2 = ftz(v.unsafe_load(((bb * l + token) * nh + h) * M3_HEADDIM + p))
                    dk_s = ftz(identical_mul_add(dy, ftz(identical_mul(ftz(identical_mul(qv, lv2)), vv2)), dk_s))
        d_q.unsafe_store(qrow + n, dq_s)
        d_k.unsafe_store(qrow + n, dk_s)
    # ---- d_v, one lane per p; dot_i once per i --------------------------
    var vrow = row * M3_HEADDIM
    var dots = List[Float32](length=qs + 1, fill=Float32(0.0))
    var lvs = List[Float32](length=qs + 1, fill=Float32(0.0))
    for i in range(inner + 1, qs):
        var ti = chunk * qs + i
        if ti < l:
            var dot = Float32(0.0)
            for n in range(M3_D_STATE):
                dot = ftz(identical_mul_add(
                    ftz(q.unsafe_load(((bb * l + ti) * nh + h) * M3_D_STATE + n)),
                    ftz(k.unsafe_load(((bb * l + token) * nh + h) * M3_D_STATE + n)), dot))
            dots[i] = dot
            lvs[i] = ftz(seg_l.unsafe_load((seg_base + i) * qs + inner))
    var p0 = 0
    comptime if lanes_are_identical:
        while p0 + HOST_FW <= M3_HEADDIM:
            var dv = F32V(0.0)
            for i in range(inner + 1, qs):
                var ti = chunk * qs + i
                if ti < l:
                    var dyv = ftz_lanes(d_y.unsafe_load[width=HOST_FW](((bb * l + ti) * nh + h) * M3_HEADDIM + p0))
                    var w = ftz(identical_mul(dots[i], lvs[i]))
                    dv = ftz_lanes(identical_mul_add_simd[HOST_FW](dyv, F32V(w), dv))
            d_v.unsafe_store(vrow + p0, dv)
            p0 += HOST_FW
    for p in range(p0, M3_HEADDIM):
        var dv_s = Float32(0.0)
        for i in range(inner + 1, qs):
            var ti = chunk * qs + i
            if ti < l:
                dv_s = ftz(identical_mul_add(
                    ftz(d_y.unsafe_load(((bb * l + ti) * nh + h) * M3_HEADDIM + p)),
                    ftz(identical_mul(dots[i], lvs[i])), dv_s))
        d_v.unsafe_store(vrow + p, dv_s)


def mamba3_s16_qkv_backward_host(
    d_q: _P, d_k: _P, d_v: _P, d_y: _P, q: _P, k: _P, v: _P, seg_l: _P,
    b_in: Int32, l_in: Int32, nh_in: Int32, qsize_in: Int32,
    grid_dim: Tuple[Int, Int, Int], block_dim: Tuple[Int, Int, Int],
):
    """The launch `mamba3_s16_qkv_backward_kernel` makes, on the host; the
    grid and block are the device launch's and are not read (every output
    cell of the kernel's two ranges is written, as the kernel writes it)."""
    var l = Int(l_in)
    var nh = Int(nh_in)
    var qs = Int(qsize_in)
    var rows = Int(b_in) * l * nh
    if rows <= 0:
        return
    var tasks = host_predict_task_count(rows)
    var chunk = (rows + tasks - 1) // tasks

    def _rows(t: Int) {imm d_q, imm d_k, imm d_v, imm d_y, imm q, imm k, imm v, imm seg_l, imm l, imm nh, imm qs, imm rows, imm chunk}:
        for row in range(t * chunk, min((t + 1) * chunk, rows)):
            _s16_row(row, d_q, d_k, d_v, d_y, q, k, v, seg_l, l, nh, qs)

    if tasks <= 1:
        _rows(0)
    else:
        host_parallelize(_rows, tasks)


# ===========================================================================
# S17: the direct readout adjoint and the descending chunk-state carry
# (`mamba3_s17_reverse_state_kernel`), on the host.
# ===========================================================================
#
# Per output cell (bb, h, p, n), the kernel's statements, unchanged:
#
#   carry = +0.0
#   for c descending:
#       direct = +0.0
#       for i ascending (t = c*qs + i < l):
#           direct = ftz(identical_mul_add(ftz(dy[t, p]), ftz(identical_mul(ftz(q[t, n]), ev[c, i])), direct))
#       d_state_direct[c, ..] = direct
#       total = ftz(direct + ftz(identical_mul(carry, ftz(identical_exp(last[c])))))
#       d_state_total[c, ..] = total; carry = total
#   d_initial[..] = carry
#
# with ev[c, i] = ftz(identical_exp(ftz(dacs[c, i]))) and last[c] =
# ftz(dacs[c, qs - 1]). What changes: ev, the chunk decays and the product
# `ftz(identical_mul(ftz(q[t, n]), ev[t]))` depend on (bb, h, t, n) but not on
# p, so each is computed once per unit instead of once per p; the D_STATE
# cells of one (bb, h, p) advance together as SIMD lanes over n; units of
# (bb, h, a block of p) split over host tasks.

comptime _S17_PBLOCK = 16


def _s17_unit(
    unit: Int, d_direct: _P, d_total: _P, d_initial: _P, d_y: _P, q: _P, dacs: _P,
    l: Int, nh: Int, qs: Int,
):
    var pblocks = (M3_HEADDIM + _S17_PBLOCK - 1) // _S17_PBLOCK
    var pb = unit % pblocks
    var g = unit // pblocks
    var h = g % nh
    var bb = g // nh
    var nc = (l + qs - 1) // qs
    var p_lo = pb * _S17_PBLOCK
    var p_hi = min(p_lo + _S17_PBLOCK, M3_HEADDIM)
    # ev per token and the chunk decays, once per unit.
    var ev = List[Float32](length=max(nc * qs, 1), fill=Float32(0.0))
    var dec = List[Float32](length=max(nc, 1), fill=Float32(0.0))
    for c in range(nc):
        for i in range(qs):
            if c * qs + i < l:
                ev[c * qs + i] = ftz(identical_exp(ftz(dacs.unsafe_load(((bb * nh + h) * nc + c) * qs + i))))
        var last = ftz(dacs.unsafe_load(((bb * nh + h) * nc + c) * qs + (qs - 1)))
        dec[c] = ftz(identical_exp(last))
    # qe[t, n] = ftz(identical_mul(ftz(q[t, n]), ev[t])), once per unit.
    var qe = List[Float32](length=max(l, 1) * M3_D_STATE, fill=Float32(0.0))
    var qep = qe.unsafe_ptr()
    for t in range(l):
        var qbase = ((bb * l + t) * nh + h) * M3_D_STATE
        var e = ev[t]
        var n0 = 0
        comptime if lanes_are_identical:
            while n0 + HOST_FW <= M3_D_STATE:
                qep.unsafe_store(t * M3_D_STATE + n0, ftz_lanes(pinned_mul_lanes(
                    ftz_lanes(q.unsafe_load[width=HOST_FW](qbase + n0)), F32V(e))))
                n0 += HOST_FW
        for n in range(n0, M3_D_STATE):
            qep.unsafe_store(t * M3_D_STATE + n, ftz(identical_mul(ftz(q.unsafe_load(qbase + n)), e)))
    for p in range(p_lo, p_hi):
        var n0 = 0
        comptime if lanes_are_identical:
            while n0 + HOST_FW <= M3_D_STATE:
                var carry = F32V(0.0)
                for rev in range(nc):
                    var c = nc - 1 - rev
                    var direct = F32V(0.0)
                    for i in range(qs):
                        var t = c * qs + i
                        if t < l:
                            var dy = F32V(ftz(d_y.unsafe_load(((bb * l + t) * nh + h) * M3_HEADDIM + p)))
                            direct = ftz_lanes(identical_mul_add_simd[HOST_FW](
                                dy, qep.unsafe_load[width=HOST_FW](t * M3_D_STATE + n0), direct))
                    var idx = (((bb * nc + c) * nh + h) * M3_HEADDIM + p) * M3_D_STATE + n0
                    d_direct.unsafe_store(idx, direct)
                    var total = ftz_lanes(direct + ftz_lanes(pinned_mul_lanes(carry, F32V(dec[c]))))
                    d_total.unsafe_store(idx, total)
                    carry = total
                d_initial.unsafe_store(((bb * nh + h) * M3_HEADDIM + p) * M3_D_STATE + n0, carry)
                n0 += HOST_FW
        for n in range(n0, M3_D_STATE):
            var carry_s = Float32(0.0)
            for rev in range(nc):
                var c = nc - 1 - rev
                var direct_s = Float32(0.0)
                for i in range(qs):
                    var t = c * qs + i
                    if t < l:
                        var dy_s = ftz(d_y.unsafe_load(((bb * l + t) * nh + h) * M3_HEADDIM + p))
                        direct_s = ftz(identical_mul_add(dy_s, qep.unsafe_load(t * M3_D_STATE + n), direct_s))
                var idx_s = (((bb * nc + c) * nh + h) * M3_HEADDIM + p) * M3_D_STATE + n
                d_direct.unsafe_store(idx_s, direct_s)
                var total_s = ftz(direct_s + ftz(identical_mul(carry_s, dec[c])))
                d_total.unsafe_store(idx_s, total_s)
                carry_s = total_s
            d_initial.unsafe_store(((bb * nh + h) * M3_HEADDIM + p) * M3_D_STATE + n, carry_s)


def mamba3_s17_reverse_state_host(
    d_state_direct: _P, d_state_total: _P, d_initial: _P, d_y: _P, q: _P, dacs: _P,
    b_in: Int32, l_in: Int32, nh_in: Int32, qs_in: Int32,
    grid_dim: Tuple[Int, Int, Int], block_dim: Tuple[Int, Int, Int],
):
    """The launch `mamba3_s17_reverse_state_kernel` makes, on the host (see
    the section note); the grid and block are not read."""
    var l = Int(l_in)
    var nh = Int(nh_in)
    var qs = Int(qs_in)
    var pblocks = (M3_HEADDIM + _S17_PBLOCK - 1) // _S17_PBLOCK
    var units = Int(b_in) * nh * pblocks
    if units <= 0:
        return
    var tasks = host_predict_task_count(units)
    var chunk = (units + tasks - 1) // tasks

    def _units(t: Int) {imm d_state_direct, imm d_state_total, imm d_initial, imm d_y, imm q, imm dacs, imm l, imm nh, imm qs, imm units, imm chunk}:
        for u in range(t * chunk, min((t + 1) * chunk, units)):
            _s17_unit(u, d_state_direct, d_state_total, d_initial, d_y, q, dacs, l, nh, qs)

    if tasks <= 1:
        _units(0)
    else:
        host_parallelize(_units, tasks)
