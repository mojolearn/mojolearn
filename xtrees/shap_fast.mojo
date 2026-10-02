# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""KernelExplainer and PermutationExplainer on the device (lane/apple-fast-shap,
2026-10-02; FAST + Apple only, default OFF).

`-D MOJOLEARN_SHAP_KERNEL_DEV` (SHAP_KERNEL_DEV): KernelExplainer's per-row
host work goes to the device. On the M3 Ultra (Istella, d = 220, 100
background rows, 2048 coalitions, 100 explained rows) the row loop spent
its 144 ms per row on the host: the coalition draws (one Python
Fisher-Yates of 220 per draw, `u.tolist()` of 1.4M doubles), `mask_expand`
(45M words per row, a serial loop) and `kernel_solve`'s normal equations
(m q^2 = 1e8 multiply-adds per row in a serial float64 loop); shap-cpu takes
77 ms per row. Here:

  draws    one thread per draw: the subset size from the cdf and the
           Fisher-Yates permutation from the SAME counter-RNG words as
           `x_trees_uniform` (`ops.draw`, the top 53 bits times 2^-53), the
           float64 products and compares in software float64
           (checks/soft_f64.mojo), so the draw is the host's draw bit for
           bit; the mask row and its hash. Then one thread per draw finds
           the first earlier identical draw (the `used` dict). The host keeps
           only the sequential bookkeeping (which draws are new, the paired
           complements, the weights), a loop over at most 4 x samples_left
           ints, and the mask rows stay on the device for `gather`.
  gather   the fixed rows and the accepted draws (and complements) into the
           m x d mask matrix, returned to the host once.
  expand   the synthetic matrix, one thread per word (`perm_device`'s
           pattern), into a device buffer kept for the process (the same
           size every row) and copied out once.
  gram     the weighted normal equations' integer part: the weights of
           `_masks` take at most a few distinct values (one per enumerated
           subset size, and multiplicity x one scale for the sampled rows),
           so A[r, c] = sum_g w_g N_g[r, c] with N_g the INTEGER count
           matrices of the groups; one thread per (r, c) cell sums the m
           coalitions. The host (float64) forms b (m q terms), A from the
           counts, and runs `kernel_solve`'s elimination unchanged. The
           values differ from `kernel_solve`'s by the association of the
           float64 sums only.

`-D MOJOLEARN_SHAP_PERM_CACHE` (SHAP_PERM_CACHE): `perm_synthetic`'s four
device buffers (388 MB per row at Istella) are created once for the process
and reused by every row of the same size, instead of allocated, first
touched and freed per row."""
from std.ffi import _Global
from std.gpu import block_dim, block_idx, grid_dim, thread_idx
from std.sys.compile import is_defined
from std.sys.info import has_apple_gpu_accelerator
from max.gpu.host import DeviceBuffer, DeviceContext
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_FAST, identical_mul64
from checks.soft_f64 import sf64_from_int, sf64_lt, sf64_mul, sf64_to_int
from xtrees.ops import draw, stream_base
from xtrees.shap import F32P, I32P

comptime _FAST_APPLE = GLOBAL_NUMERIC_MODE == NUMERIC_FAST and has_apple_gpu_accelerator()
comptime SHAP_KERNEL_DEV = _FAST_APPLE and is_defined["MOJOLEARN_SHAP_KERNEL_DEV"]()
comptime SHAP_PERM_CACHE = _FAST_APPLE and is_defined["MOJOLEARN_SHAP_PERM_CACHE"]()

comptime FAST_TPB = 256
#: the distinct weight values `gram` carries per cell (more: the caller
#: runs `kernel_solve`)
comptime SHAP_GRAM_GROUPS = 16
#: 2^-53 as a binary64 word (`ops.unit`'s scale, exact)
comptime _TWO_M53 = UInt64(0x3CA0000000000000)
comptime U64P = MutPointer[UInt64, MutAnyOrigin]
comptime HF32 = MutPointer[Float32, MutUntrackedOrigin]
comptime HF64 = MutPointer[Float64, MutUntrackedOrigin]
comptime HI32 = MutPointer[Int32, MutUntrackedOrigin]


struct _FastSlots(Defaultable, Movable):
    """ONE process-lifetime DeviceContext (xtrees/shap_device.mojo's
    pattern) and the buffers kept across rows: the draws' mask rows
    (between `draws` and `gather`) and the synthetic matrices."""
    var ctx: Optional[DeviceContext]
    var mk: Optional[DeviceBuffer[DType.int32]]
    var mk_len: Int
    var syn: Optional[DeviceBuffer[DType.float32]]
    var syn_len: Int
    var perm_res: Optional[DeviceBuffer[DType.float32]]
    var perm_res_len: Int
    var perm_x: Optional[DeviceBuffer[DType.float32]]
    var perm_x_len: Int
    var perm_bg: Optional[DeviceBuffer[DType.float32]]
    var perm_bg_len: Int
    var perm_inv: Optional[DeviceBuffer[DType.int32]]
    var perm_inv_len: Int

    def __init__(out self):
        self.ctx = Optional[DeviceContext]()
        self.mk = Optional[DeviceBuffer[DType.int32]]()
        self.mk_len = 0
        self.syn = Optional[DeviceBuffer[DType.float32]]()
        self.syn_len = 0
        self.perm_res = Optional[DeviceBuffer[DType.float32]]()
        self.perm_res_len = 0
        self.perm_x = Optional[DeviceBuffer[DType.float32]]()
        self.perm_x_len = 0
        self.perm_bg = Optional[DeviceBuffer[DType.float32]]()
        self.perm_bg_len = 0
        self.perm_inv = Optional[DeviceBuffer[DType.int32]]()
        self.perm_inv_len = 0


comptime X_TREES_SHAP_FAST = _Global[StorageType=_FastSlots, name="MojoXTreesShapFastSlots", init_fn=_FastSlots.__init__]


def _ctx() raises -> DeviceContext:
    var slot = X_TREES_SHAP_FAST.get_or_create_ptr()
    if not slot[].ctx:
        slot[].ctx = DeviceContext()
    return slot[].ctx.value().copy()


@always_inline
def _uid() -> Int:
    return Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)


def _grid(units: Int) -> Int:
    return max(1, (units + FAST_TPB - 1) // FAST_TPB)


@always_inline
def _unit_word(r: UInt64) -> UInt64:
    """`ops.unit(r)` as a binary64 word: the top 53 bits times 2^-53."""
    return sf64_mul(sf64_from_int(Int(r >> 11)), _TWO_M53)


# ------------------------------------------------------------- the draws
def draws_kernel(base: UInt64, n_draw: Int32, m_in: Int32, n_cdf: Int32, num_full: Int32, cdf: U64P, perm: I32P,
                 mk: I32P, size: I32P, hsh: U64P):
    """Draw p: `KernelExplainer._masks`'s sampling loop body on the
    uniforms u[p (1 + M) ..]: c = u[0] cdf[-1], the first cdf entry above
    c (else the last), size = that + num_full + 1; perm = range(M) shuffled
    by j = int(u[1 + i] (i + 1)) for i = M - 1 .. 1; the mask of
    perm[:size]; an FNV-1a hash of the row."""
    var p = _uid()
    if p >= Int(n_draw):
        return
    var M = Int(m_in)
    var nc = Int(n_cdf)
    var k0 = p * (1 + M)
    var c = sf64_mul(_unit_word(draw(base, k0)), cdf[unsafe_offset=nc - 1])
    var ind = nc - 1
    for i in range(nc):
        if sf64_lt(c, cdf[unsafe_offset=i]):
            ind = i
            break
    var sz = ind + Int(num_full) + 1
    size[unsafe_offset=p] = Int32(sz)
    var pb = p * M
    for f in range(M):
        perm[unsafe_offset=pb + f] = Int32(f)
    var i = M - 1
    while i > 0:
        var j = sf64_to_int(sf64_mul(_unit_word(draw(base, k0 + 1 + i)), sf64_from_int(i + 1)))
        var t = perm[unsafe_offset=pb + i]
        perm[unsafe_offset=pb + i] = perm[unsafe_offset=pb + j]
        perm[unsafe_offset=pb + j] = t
        i -= 1
    for f in range(M):
        mk[unsafe_offset=pb + f] = 0
    for q in range(min(sz, M)):
        mk[unsafe_offset=pb + Int(perm[unsafe_offset=pb + q])] = 1
    var h = UInt64(0xCBF29CE484222325)
    for f in range(M):
        h = (h ^ UInt64(Int(mk[unsafe_offset=pb + f]))) * UInt64(0x100000001B3)
    hsh[unsafe_offset=p] = h


def dup_kernel(n_draw: Int32, m_in: Int32, mk: I32P, size: I32P, hsh: U64P, dup: I32P):
    """dup[p] = the first earlier draw with the same mask row, else -1."""
    var p = _uid()
    if p >= Int(n_draw):
        return
    var M = Int(m_in)
    var res = -1
    for q in range(p):
        if size[unsafe_offset=q] != size[unsafe_offset=p] or hsh[unsafe_offset=q] != hsh[unsafe_offset=p]:
            continue
        var same = True
        for f in range(M):
            if mk[unsafe_offset=q * M + f] != mk[unsafe_offset=p * M + f]:
                same = False
                break
        if same:
            res = q
            break
    dup[unsafe_offset=p] = Int32(res)


def kernel_draws(cdf: HF64, size: HI32, dup: HI32, n_draw: Int, M: Int, n_cdf: Int, num_full: Int, seed: Int,
                 stream: Int) raises:
    """`draws_kernel` then `dup_kernel` over n_draw draws of stream
    (seed, stream); size and dup (int32 n_draw) come back, the mask rows
    stay in the process slot for `kernel_gather`."""
    if n_draw <= 0 or M <= 0 or n_cdf <= 0:
        raise Error("x_trees_kernel_draws: needs draws, features and a cdf")
    var ctx = _ctx()
    var slot = X_TREES_SHAP_FAST.get_or_create_ptr()
    var words = n_draw * M
    if not slot[].mk or slot[].mk_len != words:
        slot[].mk = Optional(ctx.enqueue_create_buffer[DType.int32](words))
        slot[].mk_len = words
    var dcdf = ctx.enqueue_create_buffer[DType.uint64](n_cdf)
    ctx.enqueue_copy(dst_buf=dcdf, src_ptr=cdf.bitcast[UInt64]())
    var dperm = ctx.enqueue_create_buffer[DType.int32](words)
    var dsize = ctx.enqueue_create_buffer[DType.int32](n_draw)
    var dhsh = ctx.enqueue_create_buffer[DType.uint64](n_draw)
    var ddup = ctx.enqueue_create_buffer[DType.int32](n_draw)
    var base = stream_base(seed, stream)
    ctx.enqueue_function[draws_kernel](
        base, Int32(n_draw), Int32(M), Int32(n_cdf), Int32(num_full), dcdf.unsafe_ptr(), dperm.unsafe_ptr(),
        slot[].mk.value().unsafe_ptr(), dsize.unsafe_ptr(), dhsh.unsafe_ptr(),
        grid_dim=_grid(n_draw), block_dim=FAST_TPB)
    ctx.enqueue_function[dup_kernel](
        Int32(n_draw), Int32(M), slot[].mk.value().unsafe_ptr(), dsize.unsafe_ptr(), dhsh.unsafe_ptr(),
        ddup.unsafe_ptr(), grid_dim=_grid(n_draw), block_dim=FAST_TPB)
    ctx.enqueue_copy(dst_ptr=size, src_buf=dsize)
    ctx.enqueue_copy(dst_ptr=dup, src_buf=ddup)
    ctx.synchronize()
    _ = dcdf^
    _ = dperm^
    _ = dsize^
    _ = dhsh^
    _ = ddup^


def gather_kernel(total: Int32, m_in: Int32, nfixed: Int32, fixed: I32P, sel: I32P, neg: I32P, mk: I32P,
                  out: I32P):
    """Word e = s * M + f of the m x M mask matrix: a fixed row as given,
    a sampled row s >= nfixed from draw sel[s - nfixed], complemented when
    neg[s - nfixed]."""
    var e = _uid()
    if e >= Int(total):
        return
    var M = Int(m_in)
    var s = e // M
    var f = e - s * M
    var nf = Int(nfixed)
    if s < nf:
        out[unsafe_offset=e] = fixed[unsafe_offset=e]
        return
    var v = mk[unsafe_offset=Int(sel[unsafe_offset=s - nf]) * M + f]
    out[unsafe_offset=e] = (1 - v) if neg[unsafe_offset=s - nf] != 0 else v


def kernel_gather(fixed: HI32, sel: HI32, neg: HI32, out: HI32, nfixed: Int, n_sel: Int, M: Int) raises:
    """out (int32 (nfixed + n_sel) x M) = the fixed rows then the selected
    draws of the last `kernel_draws`."""
    var m = nfixed + n_sel
    if m <= 0 or M <= 0:
        raise Error("x_trees_kernel_gather: needs rows and features")
    var ctx = _ctx()
    var slot = X_TREES_SHAP_FAST.get_or_create_ptr()
    if not slot[].mk:
        if n_sel > 0:
            raise Error("x_trees_kernel_gather: no draws kept")
        slot[].mk = Optional(ctx.enqueue_create_buffer[DType.int32](1))
        slot[].mk_len = 1
    var dfixed = ctx.enqueue_create_buffer[DType.int32](max(1, nfixed * M))
    if nfixed > 0:
        ctx.enqueue_copy(dst_buf=dfixed, src_ptr=fixed)
    var dsel = ctx.enqueue_create_buffer[DType.int32](max(1, n_sel))
    var dneg = ctx.enqueue_create_buffer[DType.int32](max(1, n_sel))
    if n_sel > 0:
        ctx.enqueue_copy(dst_buf=dsel, src_ptr=sel)
        ctx.enqueue_copy(dst_buf=dneg, src_ptr=neg)
    var dout = ctx.enqueue_create_buffer[DType.int32](m * M)
    ctx.enqueue_function[gather_kernel](
        Int32(m * M), Int32(M), Int32(nfixed), dfixed.unsafe_ptr(), dsel.unsafe_ptr(), dneg.unsafe_ptr(),
        slot[].mk.value().unsafe_ptr(), dout.unsafe_ptr(), grid_dim=_grid(m * M), block_dim=FAST_TPB)
    ctx.enqueue_copy(dst_ptr=out, src_buf=dout)
    ctx.synchronize()
    _ = dfixed^
    _ = dsel^
    _ = dneg^
    _ = dout^


# ----------------------------------------------------- the synthetic rows
def mask_expand_kernel(res: F32P, x: F32P, bg: F32P, masks: I32P, nb_in: Int32, d_in: Int32, total_in: Int64):
    """`shap.mask_expand`, one thread per word: res[(s nb + r) d + f] =
    x[f] if masks[s, f] else bg[r, f]. A copy, no arithmetic."""
    var nb = Int(nb_in)
    var d = Int(d_in)
    var e = _uid()
    var stride = Int(grid_dim.x) * Int(block_dim.x)
    while e < Int(total_in):
        var f = e % d
        var q = e // d
        var r = q % nb
        var s = q // nb
        res[unsafe_offset=e] = x[unsafe_offset=f] if masks[unsafe_offset=s * d + f] != 0 else bg[unsafe_offset=r * d + f]
        e += stride


def mask_expand_device(x: HF32, bg: HF32, masks: HI32, res: HF32, nb: Int, d: Int, m: Int) raises:
    """`shap.mask_expand` on the device; the synthetic buffer is kept for
    the process and reused by every call of the same size."""
    var total = m * nb * d
    if total <= 0:
        return
    var ctx = _ctx()
    var slot = X_TREES_SHAP_FAST.get_or_create_ptr()
    if not slot[].syn or slot[].syn_len != total:
        slot[].syn = Optional(ctx.enqueue_create_buffer[DType.float32](total))
        slot[].syn_len = total
    var dx = ctx.enqueue_create_buffer[DType.float32](d)
    var dbg = ctx.enqueue_create_buffer[DType.float32](nb * d)
    var dmasks = ctx.enqueue_create_buffer[DType.int32](m * d)
    ctx.enqueue_copy(dst_buf=dx, src_ptr=x)
    ctx.enqueue_copy(dst_buf=dbg, src_ptr=bg)
    ctx.enqueue_copy(dst_buf=dmasks, src_ptr=masks)
    var blocks = min((total + FAST_TPB - 1) // FAST_TPB, 65535 * 16)
    ctx.enqueue_function[mask_expand_kernel](
        slot[].syn.value().unsafe_ptr(), dx.unsafe_ptr(), dbg.unsafe_ptr(), dmasks.unsafe_ptr(), Int32(nb), Int32(d),
        Int64(total), grid_dim=(blocks, 1, 1), block_dim=(FAST_TPB, 1, 1))
    ctx.enqueue_copy(dst_ptr=res, src_buf=slot[].syn.value())
    ctx.synchronize()
    _ = dx^
    _ = dbg^
    _ = dmasks^


# ------------------------------------------------- the normal equations
def gram_kernel(cells: Int32, q_in: Int32, d_in: Int32, m_in: Int32, n_grp: Int32, masks: I32P, grp: I32P,
                cnt: I32P):
    """Cell (r, c) of the q x q count matrices: cnt[(g q + r) q + c] = the
    sum over the coalitions s of group g of (masks[s, r] - masks[s, q])
    (masks[s, c] - masks[s, q]), the integer factor of `kernel_solve`'s
    a[r q + c] (q = d - 1, the eliminated last feature)."""
    var e = _uid()
    if e >= Int(cells):
        return
    var q = Int(q_in)
    var d = Int(d_in)
    var r = e // q
    var c = e - r * q
    var acc = InlineArray[Int32, SHAP_GRAM_GROUPS](fill=Int32(0))
    for s in range(Int(m_in)):
        var b = s * d
        var last = masks[unsafe_offset=b + q]
        var er = masks[unsafe_offset=b + r] - last
        if er == 0:
            continue
        var ec = masks[unsafe_offset=b + c] - last
        var g = Int(grp[unsafe_offset=s])
        acc[g] = acc[g] + er * ec
    for g in range(Int(n_grp)):
        cnt[unsafe_offset=(g * q + r) * q + c] = acc[g]


def kernel_solve_device(masks: HI32, w: HF64, m: Int, d: Int, ey: HF64, k: Int, fx: HF64, fnull: HF64, phi: HF64,
                        grp: HI32, wval: HF64, n_grp: Int) raises:
    """`shap.kernel_solve` with the normal equations' integer part on the
    device: grp[s] (0 .. n_grp - 1) is coalition s's weight group and
    wval[g] its weight, w[s] == wval[grp[s]]. The host forms b in float64 as
    `kernel_solve` does (its s order), A from the counts, and runs the same
    elimination."""
    if d < 1 or m < 1:
        raise Error("x_trees_kernel_solve_dev: needs coalitions and features")
    if n_grp < 1 or n_grp > SHAP_GRAM_GROUPS:
        raise Error("x_trees_kernel_solve_dev: 1 .. " + String(SHAP_GRAM_GROUPS) + " weight groups")
    var q = d - 1
    var ctx = _ctx()
    var cells = q * q
    var cnt = ctx.enqueue_create_host_buffer[DType.int32](max(1, n_grp * cells))
    if q >= 1:
        var dmasks = ctx.enqueue_create_buffer[DType.int32](m * d)
        var dgrp = ctx.enqueue_create_buffer[DType.int32](m)
        var dcnt = ctx.enqueue_create_buffer[DType.int32](n_grp * cells)
        ctx.enqueue_copy(dst_buf=dmasks, src_ptr=masks)
        ctx.enqueue_copy(dst_buf=dgrp, src_ptr=grp)
        ctx.enqueue_function[gram_kernel](
            Int32(cells), Int32(q), Int32(d), Int32(m), Int32(n_grp), dmasks.unsafe_ptr(), dgrp.unsafe_ptr(),
            dcnt.unsafe_ptr(), grid_dim=_grid(cells), block_dim=FAST_TPB)
        ctx.enqueue_copy(dst_ptr=cnt.unsafe_ptr(), src_buf=dcnt)
        ctx.synchronize()
        _ = dmasks^
        _ = dgrp^
        _ = dcnt^
    for j in range(k):
        var total = fx[unsafe_offset=j] - fnull[unsafe_offset=j]
        if d == 1:
            phi[unsafe_offset=j] = total
            continue
        var a = List[Float64](length=q * q, fill=0.0)
        var bv = List[Float64](length=q, fill=0.0)
        for s in range(m):
            var last = Float64(Int(masks[unsafe_offset=s * d + q]))
            var y2 = (ey[unsafe_offset=s * k + j] - fnull[unsafe_offset=j]) - identical_mul64(last, total)
            var ws = w[unsafe_offset=s]
            for r in range(q):
                var er = Float64(Int(masks[unsafe_offset=s * d + r])) - last
                if er == 0:
                    continue
                bv[r] = bv[r] + identical_mul64(identical_mul64(ws, er), y2)
        for g in range(n_grp):
            var wg = wval[unsafe_offset=g]
            for r in range(q):
                for c in range(q):
                    var n = Int(cnt.unsafe_ptr()[unsafe_offset=(g * q + r) * q + c])
                    if n != 0:
                        a[r * q + c] = a[r * q + c] + identical_mul64(wg, Float64(n))
        # Gaussian elimination, partial pivoting (`kernel_solve`'s)
        var perm = List[Int](length=q, fill=0)
        for r in range(q):
            perm[r] = r
        for col in range(q):
            var piv = col
            var best = abs(a[perm[col] * q + col])
            for r in range(col + 1, q):
                var v = abs(a[perm[r] * q + col])
                if v > best:
                    best = v
                    piv = r
            var t = perm[col]
            perm[col] = perm[piv]
            perm[piv] = t
            var pr = perm[col]
            var pv = a[pr * q + col]
            if pv == 0:
                continue
            for r in range(col + 1, q):
                var rr = perm[r]
                var f = a[rr * q + col] / pv
                if f == 0:
                    continue
                for c in range(col, q):
                    a[rr * q + c] = a[rr * q + c] - identical_mul64(f, a[pr * q + c])
                bv[rr] = bv[rr] - identical_mul64(f, bv[pr])
        var sol = List[Float64](length=q, fill=0.0)
        var r = q - 1
        while r >= 0:
            var pr = perm[r]
            var acc = bv[pr]
            for c in range(r + 1, q):
                acc = acc - identical_mul64(a[pr * q + c], sol[c])
            var pv = a[pr * q + r]
            sol[r] = acc / pv if pv != 0 else 0.0
            r -= 1
        var ssum: Float64 = 0.0
        for c in range(q):
            phi[unsafe_offset=c * k + j] = sol[c]
            ssum = ssum + sol[c]
        phi[unsafe_offset=q * k + j] = total - ssum
    _ = cnt^


# ------------------------------------------- permutation SHAP's buffers
def perm_synth_cached_kernel(res: F32P, x: F32P, bg: F32P, inv: I32P, nb_in: Int32, d_in: Int32, total_in: Int64):
    """`perm_device.perm_synth_kernel`, word for word."""
    var nb = Int(nb_in)
    var d = Int(d_in)
    var span = 2 * d + 1
    var e = _uid()
    var stride = Int(grid_dim.x) * Int(block_dim.x)
    while e < Int(total_in):
        var f = e % d
        var q = e // d
        var r = q % nb
        var s = q // nb
        var p = s // span
        var o = s - p * span
        var pos = Int(inv[unsafe_offset=p * d + f])
        var on = pos < o if o <= d else pos >= o - d
        res[unsafe_offset=e] = x[unsafe_offset=f] if on else bg[unsafe_offset=r * d + f]
        e += stride


def perm_synthetic_cached(x: HF32, bg: HF32, inv: HI32, res: HF32, nb: Int, d: Int, n_perm: Int) raises:
    """`perm_device.perm_synthetic` with its four device buffers kept for
    the process (SHAP_PERM_CACHE): the same words into `res`."""
    var m = n_perm * (2 * d + 1)
    var total = m * nb * d
    if total <= 0:
        return
    var ctx = _ctx()
    var slot = X_TREES_SHAP_FAST.get_or_create_ptr()
    if not slot[].perm_res or slot[].perm_res_len != total:
        slot[].perm_res = Optional(ctx.enqueue_create_buffer[DType.float32](total))
        slot[].perm_res_len = total
    if not slot[].perm_x or slot[].perm_x_len != d:
        slot[].perm_x = Optional(ctx.enqueue_create_buffer[DType.float32](d))
        slot[].perm_x_len = d
    if not slot[].perm_bg or slot[].perm_bg_len != nb * d:
        slot[].perm_bg = Optional(ctx.enqueue_create_buffer[DType.float32](nb * d))
        slot[].perm_bg_len = nb * d
    if not slot[].perm_inv or slot[].perm_inv_len != n_perm * d:
        slot[].perm_inv = Optional(ctx.enqueue_create_buffer[DType.int32](n_perm * d))
        slot[].perm_inv_len = n_perm * d
    ctx.enqueue_copy(dst_buf=slot[].perm_x.value(), src_ptr=x)
    ctx.enqueue_copy(dst_buf=slot[].perm_bg.value(), src_ptr=bg)
    ctx.enqueue_copy(dst_buf=slot[].perm_inv.value(), src_ptr=inv)
    var blocks = min((total + FAST_TPB - 1) // FAST_TPB, 65535 * 16)
    ctx.enqueue_function[perm_synth_cached_kernel](
        slot[].perm_res.value().unsafe_ptr(), slot[].perm_x.value().unsafe_ptr(), slot[].perm_bg.value().unsafe_ptr(),
        slot[].perm_inv.value().unsafe_ptr(), Int32(nb), Int32(d), Int64(total),
        grid_dim=(blocks, 1, 1), block_dim=(FAST_TPB, 1, 1))
    ctx.enqueue_copy(dst_ptr=res, src_buf=slot[].perm_res.value())
    ctx.synchronize()
