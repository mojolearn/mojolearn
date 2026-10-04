# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The model-agnostic explainers on the device (lane cgr2-metrics-shap,
2026-10-03): xtrees/agnostic.mojo's units, one thread per unit, every stage
one launch on one stream, a chunk of rows per call. The caller's model runs
between the two calls of each explainer (`*_synth`, then `*_solve` /
`*_values`); the second call rebuilds the masks or permutations from the
same draws instead of keeping them, so nothing waits on the host between
the calls but the model. The host column (xtrees/agnostic_host.mojo) runs
the same units in the same stage order: the same words."""
from std.gpu import block_dim, block_idx, grid_dim, thread_idx
from max.gpu.host import DeviceBuffer, DeviceContext
from std.ffi import _Global
from std.sys.compile import is_defined
from std.sys.info import has_apple_gpu_accelerator
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_FAST, NUMERIC_IDENTICAL
from core.neural_context import process_ctx
from xtrees.agnostic import (
    F32P, I32P, U64P, I64P, kshap_mask_unit, kshap_synth_unit, bg_mean_unit, logit_unit, kshap_gram_unit,
    kshap_rhs_unit, kshap_pivot_unit, kshap_elim_unit, kshap_back_unit, fx_unit, pshap_perm_unit,
    pshap_synth_unit, pshap_marginal_unit, pshap_dcount_unit, scan_step_unit, pshap_dindex_unit,
    pshap_dsynth_unit, pshap_dmap_unit, bg_mean_mapped_unit,
)

comptime AGN_TPB = 128
#: lane apple-fast-gap-kapprox2 (2026-10-03), the FAST + Apple default
#: (IDENTICAL and the other columns are unchanged; -D
#: MOJOLEARN_KSHAP_FAST_BATCH_OFF restores the per-chunk path):
#: MOJOLEARN_KSHAP_FAST_BATCH  the explainers' synthetic matrix lives in one
#:   pooled device buffer (no per-chunk 100-400 MB allocation), the Python
#:   glue reuses one host buffer for it across chunks (no fresh pages per
#:   row), and KernelExplainer's solve runs once over many rows after every
#:   chunk's background means (`kshap_means` + `kshap_solve_ey`): the 2 (d-1)
#:   pivot/elimination launches per ROW become 2 (d-1) per solve batch.
#:   Bit-inert (the same units in the same order per row). M3 A/B
#:   kap2-kshap-batch-istella: kernel-shap istella 27,011 -> 15,325 ms
#:   (-43.3%), rel_error_vs_exact 4.378e-09 both arms.
comptime _AGN_FAST_APPLE = GLOBAL_NUMERIC_MODE == NUMERIC_FAST and has_apple_gpu_accelerator()
comptime KSHAP_FAST_BATCH = _AGN_FAST_APPLE and not is_defined["MOJOLEARN_KSHAP_FAST_BATCH_OFF"]()
#: Default FAST + Apple; MOJOLEARN_PSHAP_DELTA_OFF restores full synthesis.
#: M3 w2-pdelta-pshap-istella: 28,222.8 -> 14,788.6 ms (-47.6%).
#: w2-pdelta-quality: linear/tanh/two-output phi byte-identical, identical
#: additivity errors; model rows 2,713,500 -> 2,215,900. Measured source
#: 20bbdc37256bd00ef09fc97970f79f6967fa4f2a, no arithmetic changes here.
#: Device counts/compaction/mapping, same background mean/marginal order;
#: callback retains the existing chunked, row-independent model contract.
comptime PSHAP_DELTA = _AGN_FAST_APPLE and not is_defined["MOJOLEARN_PSHAP_DELTA_OFF"]()
#: SHAP_PERM_CACHE (FAST + Apple, on top of PSHAP_DELTA; OPT-IN until its
#: A/B passes: `-D MOJOLEARN_SHAP_PERM_CACHE`): `pshap_dsynth` keeps its
#: chunk's `_Delta` (permutations, uploaded x and background, counts, the
#: prefix sum, the varying-row index and its total) for the process, and the
#: matching `pshap_dvalues` (same addresses, counts, seed and row0: the next
#: call of the same chunk, after the caller's model) takes it instead of
#: rebuilding it: per chunk one permutation draw, two uploads, the count
#: launch, ceil(log2 G) scan launches, the index launch and one host wait
#: fewer. Same buffers, same kernels after: no bit moves. Source
#: lane/apple-fast-shap@13343dd51 (commit 2f0029ac5), recovered 2026-10-04.
#: Prior: never compiled on M3 (the branch failed to parse at
#: xtrees/api.mojo:715, a parameter named `out`, and :811); never timed;
#: board context permutation-shap istella ratio 1.20. Fixed: the old
#: candidate cached the four buffers of `perm_device.perm_synthetic`, a
#: per-row host-permutation path main has since replaced with PSHAP_DELTA
#: (and pooled the synthetic buffer under KSHAP_FAST_BATCH), so the same
#: idea, caching the per-chunk device state, is re-aimed at the work
#: PSHAP_DELTA still does twice per chunk.
comptime SHAP_PERM_CACHE = PSHAP_DELTA and is_defined["MOJOLEARN_SHAP_PERM_CACHE"]()
comptime AGN_MAX_BLOCKS = 65535 * 16
comptime _CTX = "MojoXTreesAgnosticIdentical" if GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL else "MojoXTreesAgnosticFast"


def _ctx() raises -> DeviceContext:
    return process_ctx[_CTX]()


@always_inline
def _blocks(total: Int) -> Int:
    return max(1, min((total + AGN_TPB - 1) // AGN_TPB, AGN_MAX_BLOCKS))


@always_inline
def _t0() -> Int:
    return Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)


@always_inline
def _stride() -> Int:
    return Int(grid_dim.x) * Int(block_dim.x)


def mask_kernel(total: Int64, M: Int32, m: Int32, nfixed: Int64, nfull: Int32, npaired: Int32, L: Int32,
                seed: Int64, row0: Int64, size_off: I64P, size_w: U64P, cdf: U64P, wrand: UInt64, perm: I32P,
                masks: I32P, w: U64P):
    var t = _t0()
    while t < Int(total):
        kshap_mask_unit(t, Int(M), Int(m), Int(nfixed), Int(nfull), Int(npaired), Int(L), Int(seed), Int(row0),
                        size_off, size_w, cdf, wrand, perm, masks, w)
        t += _stride()


def ksynth_kernel(total: Int64, nb: Int32, d: Int32, m: Int32, x: F32P, bg: F32P, masks: I32P, syn: F32P):
    var t = _t0()
    while t < Int(total):
        kshap_synth_unit(t, Int(nb), Int(d), Int(m), x, bg, masks, syn)
        t += _stride()


def mean_kernel(total: Int64, nb: Int32, k: Int32, y: F32P, res: U64P):
    var t = _t0()
    while t < Int(total):
        bg_mean_unit(t, Int(nb), Int(k), y, res)
        t += _stride()


def logit_kernel(total: Int64, x: U64P):
    var t = _t0()
    while t < Int(total):
        logit_unit(t, x)
        t += _stride()


def fx_kernel(total: Int64, y: F32P, res: U64P):
    var t = _t0()
    while t < Int(total):
        fx_unit(t, y, res)
        t += _stride()


def gram_kernel(total: Int64, d: Int32, m: Int32, masks: I32P, w: U64P, A: U64P, perm0: I32P):
    var t = _t0()
    while t < Int(total):
        kshap_gram_unit(t, Int(d), Int(m), masks, w, A, perm0)
        t += _stride()


def rhs_kernel(total: Int64, d: Int32, m: Int32, k: Int32, masks: I32P, w: U64P, ey: U64P, fx: U64P,
               fnull: U64P, B: U64P):
    var t = _t0()
    while t < Int(total):
        kshap_rhs_unit(t, Int(d), Int(m), Int(k), masks, w, ey, fx, fnull, B)
        t += _stride()


def pivot_kernel(total: Int64, q: Int32, col: Int32, A: U64P, pin: I32P, pout: I32P):
    var t = _t0()
    while t < Int(total):
        kshap_pivot_unit(t, Int(q), Int(col), A, pin, pout)
        t += _stride()


def elim_kernel(total: Int64, q: Int32, k: Int32, col: Int32, A: U64P, B: U64P, perm: I32P):
    var t = _t0()
    while t < Int(total):
        kshap_elim_unit(t, Int(q), Int(k), Int(col), A, B, perm)
        t += _stride()


def back_kernel(total: Int64, d: Int32, k: Int32, A: U64P, B: U64P, perm: I32P, fx: U64P, fnull: U64P,
                sol: U64P, phi: U64P):
    var t = _t0()
    while t < Int(total):
        kshap_back_unit(t, Int(d), Int(k), A, B, perm, fx, fnull, sol, phi)
        t += _stride()


def perm_kernel(total: Int64, d: Int32, np: Int32, seed: Int64, row0: Int64, perm: I32P, inv: I32P):
    var t = _t0()
    while t < Int(total):
        pshap_perm_unit(t, Int(d), Int(np), Int(seed), Int(row0), perm, inv)
        t += _stride()


def psynth_kernel(total: Int64, nb: Int32, d: Int32, np: Int32, x: F32P, bg: F32P, inv: I32P, syn: F32P):
    var t = _t0()
    while t < Int(total):
        pshap_synth_unit(t, Int(nb), Int(d), Int(np), x, bg, inv, syn)
        t += _stride()


def marginal_kernel(total: Int64, d: Int32, k: Int32, np: Int32, inv: I32P, ey: U64P, phi: U64P):
    var t = _t0()
    while t < Int(total):
        pshap_marginal_unit(t, Int(d), Int(k), Int(np), inv, ey, phi)
        t += _stride()


def dcount_kernel(total: Int64, nb: Int32, d: Int32, np: Int32, x: F32P, bg: F32P, perm: I32P, cnt: I64P):
    var t = _t0()
    while t < Int(total):
        pshap_dcount_unit(t, Int(nb), Int(d), Int(np), x, bg, perm, cnt)
        t += _stride()


def scan_kernel(total: Int64, off: Int64, src: I64P, dst: I64P):
    var t = _t0()
    while t < Int(total):
        scan_step_unit(t, Int(off), src, dst)
        t += _stride()


def dindex_kernel(total: Int64, nb: Int32, d: Int32, np: Int32, x: F32P, bg: F32P, perm: I32P, incl: I64P,
                  cnt: I64P, idx: I64P, src: I64P):
    var t = _t0()
    while t < Int(total):
        pshap_dindex_unit(t, Int(nb), Int(d), Int(np), x, bg, perm, incl, cnt, idx, src)
        t += _stride()


def dsynth_kernel(total: Int64, nb: Int32, d: Int32, np: Int32, x: F32P, bg: F32P, inv: I32P, src: I64P,
                  syn: F32P):
    var t = _t0()
    while t < Int(total):
        pshap_dsynth_unit(t, Int(nb), Int(d), Int(np), x, bg, inv, src, syn)
        t += _stride()


def dmap_kernel(total: Int64, nb: Int32, idx: I64P, mp: I64P):
    var t = _t0()
    while t < Int(total):
        pshap_dmap_unit(t, Int(nb), idx, mp)
        t += _stride()


def mmean_kernel(total: Int64, nb: Int32, k: Int32, y: F32P, mp: I64P, res: U64P):
    var t = _t0()
    while t < Int(total):
        bg_mean_mapped_unit(t, Int(nb), Int(k), y, mp, res)
        t += _stride()


def _up_f32(ctx: DeviceContext, addr: Int, n: Int) raises -> DeviceBuffer[DType.float32]:
    var b = ctx.enqueue_create_buffer[DType.float32](max(n, 1))
    if n > 0:
        ctx.enqueue_copy(dst_buf=b.create_sub_buffer[DType.float32](0, n), src_ptr=F32P(unsafe_from_address=addr))
    return b^


def _up_u64(ctx: DeviceContext, addr: Int, n: Int) raises -> DeviceBuffer[DType.uint64]:
    var b = ctx.enqueue_create_buffer[DType.uint64](max(n, 1))
    if n > 0:
        ctx.enqueue_copy(dst_buf=b.create_sub_buffer[DType.uint64](0, n), src_ptr=U64P(unsafe_from_address=addr))
    return b^


def _up_i64(ctx: DeviceContext, addr: Int, n: Int) raises -> DeviceBuffer[DType.int64]:
    var b = ctx.enqueue_create_buffer[DType.int64](max(n, 1))
    if n > 0:
        ctx.enqueue_copy(dst_buf=b.create_sub_buffer[DType.int64](0, n), src_ptr=I64P(unsafe_from_address=addr))
    return b^


struct _Masks(Movable):
    """A chunk's coalition masks and weights, on the device."""
    var masks: DeviceBuffer[DType.int32]
    var w: DeviceBuffer[DType.uint64]
    var perm: DeviceBuffer[DType.int32]
    var so: DeviceBuffer[DType.int64]
    var sw: DeviceBuffer[DType.uint64]
    var cdf: DeviceBuffer[DType.uint64]

    def __init__(out self, ctx: DeviceContext, size_off: Int, size_w: Int, cdf: Int, R: Int, d: Int, m: Int,
                 nfixed: Int, nfull: Int, npaired: Int, L: Int, seed: Int, row0: Int, wrand: UInt64) raises:
        self.so = _up_i64(ctx, size_off, nfull + 1)
        self.sw = _up_u64(ctx, size_w, nfull)
        self.cdf = _up_u64(ctx, cdf, L)
        self.masks = ctx.enqueue_create_buffer[DType.int32](max(R * m * d, 1))
        self.perm = ctx.enqueue_create_buffer[DType.int32](max(R * m * d, 1))
        self.w = ctx.enqueue_create_buffer[DType.uint64](max(R * m, 1))
        var total = R * m
        if total > 0:
            ctx.enqueue_function[mask_kernel](
                Int64(total), Int32(d), Int32(m), Int64(nfixed), Int32(nfull), Int32(npaired), Int32(L),
                Int64(seed), Int64(row0), self.so.unsafe_ptr(), self.sw.unsafe_ptr(), self.cdf.unsafe_ptr(),
                wrand, self.perm.unsafe_ptr(), self.masks.unsafe_ptr(), self.w.unsafe_ptr(),
                grid_dim=_blocks(total), block_dim=AGN_TPB,
            )


struct _AgnPool(Defaultable, Movable):
    """The process's pooled synthetic-matrix device buffer
    (MOJOLEARN_KSHAP_FAST_BATCH): grown on demand, never shrunk."""
    var syn: Optional[DeviceBuffer[DType.float32]]
    var cap: Int

    def __init__(out self):
        self.syn = Optional[DeviceBuffer[DType.float32]]()
        self.cap = 0


comptime AGN_POOL = _Global[StorageType=_AgnPool, name="MojoXTreesAgnosticSynPoolFast", init_fn=_AgnPool.__init__]


def _pool_syn(ctx: DeviceContext, total: Int) raises -> F32P:
    """A device pointer to >= total floats of the pooled synthetic buffer.
    Every user synchronizes before returning, so a regrow never frees a
    buffer in flight."""
    var slot = AGN_POOL.get_or_create_ptr()
    if slot[].cap < total:
        slot[].syn = Optional[DeviceBuffer[DType.float32]]()
        slot[].syn = ctx.enqueue_create_buffer[DType.float32](total)
        slot[].cap = total
    return F32P(unsafe_from_address=Int(slot[].syn.value().unsafe_ptr()))


def _pool_down(ctx: DeviceContext, dst: Int, total: Int) raises:
    """The pooled buffer's first `total` floats to the host address dst."""
    var slot = AGN_POOL.get_or_create_ptr()
    ctx.enqueue_copy(dst_ptr=F32P(unsafe_from_address=dst),
                     src_buf=slot[].syn.value().create_sub_buffer[DType.float32](0, total))


def kshap_synth(x: Int, bg: Int, size_off: Int, size_w: Int, cdf: Int, syn: Int, R: Int, nb: Int, d: Int, m: Int,
                nfixed: Int, nfull: Int, npaired: Int, L: Int, seed: Int, row0: Int, wrand: UInt64) raises:
    """The chunk's synthetic rows ((row, sample, background row) x d)."""
    var total = R * m * nb * d
    if total <= 0:
        return
    var ctx = _ctx()
    var mk = _Masks(ctx, size_off, size_w, cdf, R, d, m, nfixed, nfull, npaired, L, seed, row0, wrand)
    var dx = _up_f32(ctx, x, R * d)
    var dbg = _up_f32(ctx, bg, nb * d)
    comptime if KSHAP_FAST_BATCH:
        var ps = _pool_syn(ctx, total)
        ctx.enqueue_function[ksynth_kernel](
            Int64(total), Int32(nb), Int32(d), Int32(m), dx.unsafe_ptr(), dbg.unsafe_ptr(), mk.masks.unsafe_ptr(),
            ps, grid_dim=_blocks(total), block_dim=AGN_TPB,
        )
        _pool_down(ctx, syn, total)
        ctx.synchronize()
    else:
        var dsyn = ctx.enqueue_create_buffer[DType.float32](total)
        ctx.enqueue_function[ksynth_kernel](
            Int64(total), Int32(nb), Int32(d), Int32(m), dx.unsafe_ptr(), dbg.unsafe_ptr(), mk.masks.unsafe_ptr(),
            dsyn.unsafe_ptr(), grid_dim=_blocks(total), block_dim=AGN_TPB,
        )
        ctx.enqueue_copy(dst_ptr=F32P(unsafe_from_address=syn), src_buf=dsyn)
        ctx.synchronize()
        _ = dsyn^
    _ = mk^
    _ = dx^
    _ = dbg^


def _ksolve_core(ctx: DeviceContext, mut mk: _Masks, ey: U64P, dfx: U64P, dnull: U64P, phi: Int, R: Int, d: Int,
                 k: Int, m: Int) raises:
    """From the chunk's (linked) background means ey (R x m x k) and fx (R x
    k) on the device: the normal equations, the elimination, the back
    substitution, phi (R x d x k binary64) to the host address phi;
    synchronizes."""
    var q = d - 1
    var dA = ctx.enqueue_create_buffer[DType.uint64](max(R * q * q, 1))
    var dB = ctx.enqueue_create_buffer[DType.uint64](max(R * q * k, 1))
    var p0 = ctx.enqueue_create_buffer[DType.int32](max(R * q, 1))
    var p1 = ctx.enqueue_create_buffer[DType.int32](max(R * q, 1))
    var sol = ctx.enqueue_create_buffer[DType.uint64](max(R * k * q, 1))
    var dphi = ctx.enqueue_create_buffer[DType.uint64](R * d * k)
    if q > 0:
        ctx.enqueue_function[gram_kernel](
            Int64(R * q * q), Int32(d), Int32(m), mk.masks.unsafe_ptr(), mk.w.unsafe_ptr(), dA.unsafe_ptr(),
            p0.unsafe_ptr(), grid_dim=_blocks(R * q * q), block_dim=AGN_TPB,
        )
        ctx.enqueue_function[rhs_kernel](
            Int64(R * q * k), Int32(d), Int32(m), Int32(k), mk.masks.unsafe_ptr(), mk.w.unsafe_ptr(),
            ey, dfx, dnull, dB.unsafe_ptr(), grid_dim=_blocks(R * q * k), block_dim=AGN_TPB,
        )
    var pa = p0.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]()
    var pb = p1.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]()
    var in0 = True
    for col in range(q):
        var pin = pa if in0 else pb
        var pout = pb if in0 else pa
        ctx.enqueue_function[pivot_kernel](Int64(R), Int32(q), Int32(col), dA.unsafe_ptr(), pin, pout,
                                           grid_dim=_blocks(R), block_dim=AGN_TPB)
        var nr = q - 1 - col
        var units = R * nr * (nr + k)
        if units > 0:
            ctx.enqueue_function[elim_kernel](Int64(units), Int32(q), Int32(k), Int32(col), dA.unsafe_ptr(),
                                              dB.unsafe_ptr(), pout, grid_dim=_blocks(units), block_dim=AGN_TPB)
        in0 = not in0
    var pfin = pa if in0 else pb
    ctx.enqueue_function[back_kernel](
        Int64(R * k), Int32(d), Int32(k), dA.unsafe_ptr(), dB.unsafe_ptr(), pfin, dfx,
        dnull, sol.unsafe_ptr(), dphi.unsafe_ptr(), grid_dim=_blocks(R * k), block_dim=AGN_TPB,
    )
    ctx.enqueue_copy(dst_ptr=U64P(unsafe_from_address=phi), src_buf=dphi)
    ctx.synchronize()
    _ = dA^
    _ = dB^
    _ = p0^
    _ = p1^
    _ = sol^
    _ = dphi^


def _fx_dev(ctx: DeviceContext, fx: Int, R: Int, k: Int, link: Bool,
            mut dfx32: DeviceBuffer[DType.float32], mut dfx: DeviceBuffer[DType.uint64]) raises:
    """The rows' own outputs (Float32 R x k at host address fx) as linked
    binary64 on the device."""
    ctx.enqueue_copy(dst_buf=dfx32.create_sub_buffer[DType.float32](0, R * k),
                     src_ptr=F32P(unsafe_from_address=fx))
    ctx.enqueue_function[fx_kernel](Int64(R * k), dfx32.unsafe_ptr(), dfx.unsafe_ptr(), grid_dim=_blocks(R * k),
                                    block_dim=AGN_TPB)
    if link:
        ctx.enqueue_function[logit_kernel](Int64(R * k), dfx.unsafe_ptr(), grid_dim=_blocks(R * k), block_dim=AGN_TPB)


def kshap_solve(yout: Int, fx: Int, fnull: Int, size_off: Int, size_w: Int, cdf: Int, phi: Int, R: Int, nb: Int,
                d: Int, k: Int, m: Int, nfixed: Int, nfull: Int, npaired: Int, L: Int, seed: Int, row0: Int,
                wrand: UInt64, link: Bool) raises:
    """phi (R x d x k binary64) of the chunk from the model outputs `out` of
    its synthetic rows, `fx` of its own rows (Float32 R x k) and the linked
    background value `fnull` (binary64 k)."""
    if R <= 0:
        return
    var ctx = _ctx()
    var mk = _Masks(ctx, size_off, size_w, cdf, R, d, m, nfixed, nfull, npaired, L, seed, row0, wrand)
    var dout = _up_f32(ctx, yout, R * m * nb * k)
    var ey = ctx.enqueue_create_buffer[DType.uint64](max(R * m * k, 1))
    if R * m * k > 0:
        ctx.enqueue_function[mean_kernel](
            Int64(R * m * k), Int32(nb), Int32(k), dout.unsafe_ptr(), ey.unsafe_ptr(),
            grid_dim=_blocks(R * m * k), block_dim=AGN_TPB,
        )
        if link:
            ctx.enqueue_function[logit_kernel](Int64(R * m * k), ey.unsafe_ptr(), grid_dim=_blocks(R * m * k),
                                               block_dim=AGN_TPB)
    var dfx32 = ctx.enqueue_create_buffer[DType.float32](max(R * k, 1))
    var dfx = ctx.enqueue_create_buffer[DType.uint64](max(R * k, 1))
    _fx_dev(ctx, fx, R, k, link, dfx32, dfx)
    var dnull = _up_u64(ctx, fnull, k)
    _ksolve_core(ctx, mk, ey.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin](),
                 dfx.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin](),
                 dnull.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin](), phi, R, d, k, m)
    _ = mk^
    _ = dout^
    _ = ey^
    _ = dfx32^
    _ = dfx^
    _ = dnull^


def kshap_means(yout: Int, ey: Int, R: Int, nb: Int, k: Int, m: Int, link: Bool) raises:
    """MOJOLEARN_KSHAP_FAST_BATCH: a chunk's (linked) background means ey
    (binary64 R x m x k, to the host address ey) from the model outputs of
    its synthetic rows; `kshap_solve`'s first stage, the same words."""
    var total = R * m * k
    if total <= 0:
        return
    var ctx = _ctx()
    var dout = _up_f32(ctx, yout, R * m * nb * k)
    var dey = ctx.enqueue_create_buffer[DType.uint64](total)
    ctx.enqueue_function[mean_kernel](Int64(total), Int32(nb), Int32(k), dout.unsafe_ptr(), dey.unsafe_ptr(),
                                      grid_dim=_blocks(total), block_dim=AGN_TPB)
    if link:
        ctx.enqueue_function[logit_kernel](Int64(total), dey.unsafe_ptr(), grid_dim=_blocks(total), block_dim=AGN_TPB)
    ctx.enqueue_copy(dst_ptr=U64P(unsafe_from_address=ey), src_buf=dey)
    ctx.synchronize()
    _ = dout^
    _ = dey^


def kshap_solve_ey(ey: Int, fx: Int, fnull: Int, size_off: Int, size_w: Int, cdf: Int, phi: Int, R: Int, d: Int,
                   k: Int, m: Int, nfixed: Int, nfull: Int, npaired: Int, L: Int, seed: Int, row0: Int,
                   wrand: UInt64, link: Bool) raises:
    """MOJOLEARN_KSHAP_FAST_BATCH: `kshap_solve` from the rows' background
    means (`kshap_means`, binary64 R x m x k at host address ey) over a
    batch of R rows starting at row0: one pivot/elimination sweep for all
    of them."""
    if R <= 0:
        return
    var ctx = _ctx()
    var mk = _Masks(ctx, size_off, size_w, cdf, R, d, m, nfixed, nfull, npaired, L, seed, row0, wrand)
    var dey = _up_u64(ctx, ey, R * m * k)
    var dfx32 = ctx.enqueue_create_buffer[DType.float32](max(R * k, 1))
    var dfx = ctx.enqueue_create_buffer[DType.uint64](max(R * k, 1))
    _fx_dev(ctx, fx, R, k, link, dfx32, dfx)
    var dnull = _up_u64(ctx, fnull, k)
    _ksolve_core(ctx, mk, dey.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin](),
                 dfx.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin](),
                 dnull.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin](), phi, R, d, k, m)
    _ = mk^
    _ = dey^
    _ = dfx32^
    _ = dfx^
    _ = dnull^


def _perms(ctx: DeviceContext, R: Int, d: Int, np: Int, seed: Int, row0: Int,
           mut perm: DeviceBuffer[DType.int32], mut inv: DeviceBuffer[DType.int32]) raises:
    ctx.enqueue_function[perm_kernel](
        Int64(R * np), Int32(d), Int32(np), Int64(seed), Int64(row0), perm.unsafe_ptr(), inv.unsafe_ptr(),
        grid_dim=_blocks(R * np), block_dim=AGN_TPB,
    )


def pshap_synth(x: Int, bg: Int, syn: Int, R: Int, nb: Int, d: Int, np: Int, seed: Int, row0: Int) raises:
    """The chunk's synthetic rows ((row, permutation, coalition, background
    row) x d)."""
    var total = R * np * (2 * d + 1) * nb * d
    if total <= 0:
        return
    var ctx = _ctx()
    var perm = ctx.enqueue_create_buffer[DType.int32](R * np * d)
    var inv = ctx.enqueue_create_buffer[DType.int32](R * np * d)
    _perms(ctx, R, d, np, seed, row0, perm, inv)
    var dx = _up_f32(ctx, x, R * d)
    var dbg = _up_f32(ctx, bg, nb * d)
    comptime if KSHAP_FAST_BATCH:
        var ps = _pool_syn(ctx, total)
        ctx.enqueue_function[psynth_kernel](
            Int64(total), Int32(nb), Int32(d), Int32(np), dx.unsafe_ptr(), dbg.unsafe_ptr(), inv.unsafe_ptr(),
            ps, grid_dim=_blocks(total), block_dim=AGN_TPB,
        )
        _pool_down(ctx, syn, total)
        ctx.synchronize()
    else:
        var dsyn = ctx.enqueue_create_buffer[DType.float32](total)
        ctx.enqueue_function[psynth_kernel](
            Int64(total), Int32(nb), Int32(d), Int32(np), dx.unsafe_ptr(), dbg.unsafe_ptr(), inv.unsafe_ptr(),
            dsyn.unsafe_ptr(), grid_dim=_blocks(total), block_dim=AGN_TPB,
        )
        ctx.enqueue_copy(dst_ptr=F32P(unsafe_from_address=syn), src_buf=dsyn)
        ctx.synchronize()
        _ = dsyn^
    _ = perm^
    _ = inv^
    _ = dx^
    _ = dbg^


def pshap_values(yout: Int, phi: Int, R: Int, nb: Int, d: Int, k: Int, np: Int, seed: Int, row0: Int) raises:
    """phi (R x d x k binary64) from the model outputs of the chunk's
    synthetic rows."""
    var mm = np * (2 * d + 1)
    if R * d * k <= 0 or mm <= 0:
        return
    var ctx = _ctx()
    var perm = ctx.enqueue_create_buffer[DType.int32](R * np * d)
    var inv = ctx.enqueue_create_buffer[DType.int32](R * np * d)
    _perms(ctx, R, d, np, seed, row0, perm, inv)
    var dout = _up_f32(ctx, yout, R * mm * nb * k)
    var ey = ctx.enqueue_create_buffer[DType.uint64](R * mm * k)
    ctx.enqueue_function[mean_kernel](
        Int64(R * mm * k), Int32(nb), Int32(k), dout.unsafe_ptr(), ey.unsafe_ptr(),
        grid_dim=_blocks(R * mm * k), block_dim=AGN_TPB,
    )
    var dphi = ctx.enqueue_create_buffer[DType.uint64](R * d * k)
    ctx.enqueue_function[marginal_kernel](
        Int64(R * d * k), Int32(d), Int32(k), Int32(np), inv.unsafe_ptr(), ey.unsafe_ptr(), dphi.unsafe_ptr(),
        grid_dim=_blocks(R * d * k), block_dim=AGN_TPB,
    )
    ctx.enqueue_copy(dst_ptr=U64P(unsafe_from_address=phi), src_buf=dphi)
    ctx.synchronize()
    _ = perm^
    _ = inv^
    _ = dout^
    _ = ey^
    _ = dphi^


def bg_mean(y: Int, res: Int, m: Int, nb: Int, k: Int) raises:
    """`block_mean` on the device: res (binary64 m x k) = the means over nb
    consecutive rows of y (Float32 (m nb) x k)."""
    if m * k <= 0:
        return
    var ctx = _ctx()
    var dy = _up_f32(ctx, y, m * nb * k)
    var dr = ctx.enqueue_create_buffer[DType.uint64](m * k)
    ctx.enqueue_function[mean_kernel](Int64(m * k), Int32(nb), Int32(k), dy.unsafe_ptr(), dr.unsafe_ptr(),
                                      grid_dim=_blocks(m * k), block_dim=AGN_TPB)
    ctx.enqueue_copy(dst_ptr=U64P(unsafe_from_address=res), src_buf=dr)
    ctx.synchronize()
    _ = dy^
    _ = dr^


struct _Delta(Movable):
    """MOJOLEARN_PSHAP_DELTA: a chunk's permutations and its varying-row
    index on the device. idx[(group, r)] = the compact row, or -1 when the
    row repeats its predecessor's; src[compact row] = (group, r); `total`
    = the compact rows (one small download, the only sync before the
    synthetic rows are sized)."""
    var perm: DeviceBuffer[DType.int32]
    var inv: DeviceBuffer[DType.int32]
    var dx: DeviceBuffer[DType.float32]
    var dbg: DeviceBuffer[DType.float32]
    var cnt: DeviceBuffer[DType.int64]
    var sa: DeviceBuffer[DType.int64]
    var sb: DeviceBuffer[DType.int64]
    var idx: DeviceBuffer[DType.int64]
    var src: DeviceBuffer[DType.int64]
    var total: Int

    def __init__(out self, ctx: DeviceContext, x: Int, bg: Int, tot: Int, R: Int, nb: Int, d: Int, np: Int,
                 seed: Int, row0: Int) raises:
        var G = R * np * (2 * d + 1)
        var perm = ctx.enqueue_create_buffer[DType.int32](R * np * d)
        var inv = ctx.enqueue_create_buffer[DType.int32](R * np * d)
        _perms(ctx, R, d, np, seed, row0, perm, inv)
        var dx = _up_f32(ctx, x, R * d)
        var dbg = _up_f32(ctx, bg, nb * d)
        var cnt = ctx.enqueue_create_buffer[DType.int64](G)
        var sa = ctx.enqueue_create_buffer[DType.int64](G)
        var sb = ctx.enqueue_create_buffer[DType.int64](G)
        var idx = ctx.enqueue_create_buffer[DType.int64](G * nb)
        var src = ctx.enqueue_create_buffer[DType.int64](G * nb)
        ctx.enqueue_function[dcount_kernel](
            Int64(G), Int32(nb), Int32(d), Int32(np), dx.unsafe_ptr(), dbg.unsafe_ptr(), perm.unsafe_ptr(),
            cnt.unsafe_ptr(), grid_dim=_blocks(G), block_dim=AGN_TPB,
        )
        # inclusive prefix sum of the group counts: ceil(log2 G) parallel
        # steps between two buffers (the first launch copies the counts)
        var pa = sa.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]()
        var pb = sb.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]()
        ctx.enqueue_function[scan_kernel](Int64(G), Int64(G), cnt.unsafe_ptr(), pa, grid_dim=_blocks(G),
                                          block_dim=AGN_TPB)
        var in_a = True
        var off = 1
        while off < G:
            var pin = pa if in_a else pb
            var pout = pb if in_a else pa
            ctx.enqueue_function[scan_kernel](Int64(G), Int64(off), pin, pout, grid_dim=_blocks(G),
                                              block_dim=AGN_TPB)
            in_a = not in_a
            off *= 2
        var incl = pa if in_a else pb
        ctx.enqueue_function[dindex_kernel](
            Int64(G * nb), Int32(nb), Int32(d), Int32(np), dx.unsafe_ptr(), dbg.unsafe_ptr(), perm.unsafe_ptr(),
            incl, cnt.unsafe_ptr(), idx.unsafe_ptr(), src.unsafe_ptr(), grid_dim=_blocks(G * nb), block_dim=AGN_TPB,
        )
        if in_a:
            ctx.enqueue_copy(dst_ptr=I64P(unsafe_from_address=tot), src_buf=sa.create_sub_buffer[DType.int64](G - 1, 1))
        else:
            ctx.enqueue_copy(dst_ptr=I64P(unsafe_from_address=tot), src_buf=sb.create_sub_buffer[DType.int64](G - 1, 1))
        ctx.synchronize()
        self.total = Int(I64P(unsafe_from_address=tot)[0])
        self.perm = perm^
        self.inv = inv^
        self.dx = dx^
        self.dbg = dbg^
        self.cnt = cnt^
        self.sa = sa^
        self.sb = sb^
        self.idx = idx^
        self.src = src^


struct _DeltaSlot(Defaultable, Movable):
    """SHAP_PERM_CACHE: the last `pshap_dsynth` chunk's `_Delta` and its key
    (x, bg, tot, R, nb, d, np, seed, row0)."""
    var dl: Optional[_Delta]
    var key: List[Int]

    def __init__(out self):
        self.dl = Optional[_Delta]()
        self.key = List[Int]()


comptime AGN_DELTA_SLOT = _Global[StorageType=_DeltaSlot, name="MojoXTreesAgnosticDeltaFast", init_fn=_DeltaSlot.__init__]


def _delta_key(x: Int, bg: Int, tot: Int, R: Int, nb: Int, d: Int, np: Int, seed: Int, row0: Int) -> List[Int]:
    return [x, bg, tot, R, nb, d, np, seed, row0]


def _delta_take(ctx: DeviceContext, x: Int, bg: Int, tot: Int, R: Int, nb: Int, d: Int, np: Int, seed: Int,
                row0: Int) raises -> _Delta:
    """SHAP_PERM_CACHE: the kept `_Delta` when its key matches (the slot is
    emptied either way), else a fresh one."""
    var slot = AGN_DELTA_SLOT.get_or_create_ptr()
    var hit = False
    if slot[].dl:
        hit = slot[].key == _delta_key(x, bg, tot, R, nb, d, np, seed, row0)
    if hit:
        slot[].key = List[Int]()
        return slot[].dl.take()
    slot[].dl = Optional[_Delta]()
    slot[].key = List[Int]()
    return _Delta(ctx, x, bg, tot, R, nb, d, np, seed, row0)


def pshap_dsynth(x: Int, bg: Int, syn: Int, tot: Int, R: Int, nb: Int, d: Int, np: Int, seed: Int,
                 row0: Int) raises:
    """MOJOLEARN_PSHAP_DELTA: the chunk's VARYING synthetic rows, in full
    (row, permutation, coalition, background row) order, to the host
    address syn (room for every row); their count to the host Int64 at
    tot."""
    if R * np * (2 * d + 1) * nb * d <= 0:
        return
    var ctx = _ctx()
    var dl = _Delta(ctx, x, bg, tot, R, nb, d, np, seed, row0)
    var total = dl.total * d
    if total > 0:
        var ps = _pool_syn(ctx, total)
        ctx.enqueue_function[dsynth_kernel](
            Int64(total), Int32(nb), Int32(d), Int32(np), dl.dx.unsafe_ptr(), dl.dbg.unsafe_ptr(),
            dl.inv.unsafe_ptr(), dl.src.unsafe_ptr(), ps, grid_dim=_blocks(total), block_dim=AGN_TPB,
        )
        _pool_down(ctx, syn, total)
        ctx.synchronize()
    comptime if SHAP_PERM_CACHE:
        var slot = AGN_DELTA_SLOT.get_or_create_ptr()
        slot[].dl = Optional[_Delta](dl^)
        slot[].key = _delta_key(x, bg, tot, R, nb, d, np, seed, row0)
    else:
        _ = dl^


def pshap_dvalues(x: Int, bg: Int, yout: Int, phi: Int, tot: Int, R: Int, nb: Int, d: Int, k: Int, np: Int,
                  seed: Int, row0: Int) raises:
    """MOJOLEARN_PSHAP_DELTA: phi (R x d x k binary64) from the model
    outputs of the chunk's varying rows (`pshap_dsynth`'s order): every
    (coalition, background row) reads its latest varying predecessor's
    output, then `pshap_values`' means and marginals."""
    var mm = np * (2 * d + 1)
    if R * d * k <= 0 or mm <= 0:
        return
    var ctx = _ctx()
    var dl: _Delta
    comptime if SHAP_PERM_CACHE:
        dl = _delta_take(ctx, x, bg, tot, R, nb, d, np, seed, row0)
    else:
        dl = _Delta(ctx, x, bg, tot, R, nb, d, np, seed, row0)
    var G = R * mm
    var mp = ctx.enqueue_create_buffer[DType.int64](G * nb)
    ctx.enqueue_function[dmap_kernel](Int64(G * nb), Int32(nb), dl.idx.unsafe_ptr(), mp.unsafe_ptr(),
                                      grid_dim=_blocks(G * nb), block_dim=AGN_TPB)
    var dout = _up_f32(ctx, yout, dl.total * k)
    var ey = ctx.enqueue_create_buffer[DType.uint64](G * k)
    ctx.enqueue_function[mmean_kernel](
        Int64(G * k), Int32(nb), Int32(k), dout.unsafe_ptr(), mp.unsafe_ptr(), ey.unsafe_ptr(),
        grid_dim=_blocks(G * k), block_dim=AGN_TPB,
    )
    var dphi = ctx.enqueue_create_buffer[DType.uint64](R * d * k)
    ctx.enqueue_function[marginal_kernel](
        Int64(R * d * k), Int32(d), Int32(k), Int32(np), dl.inv.unsafe_ptr(), ey.unsafe_ptr(), dphi.unsafe_ptr(),
        grid_dim=_blocks(R * d * k), block_dim=AGN_TPB,
    )
    ctx.enqueue_copy(dst_ptr=U64P(unsafe_from_address=phi), src_buf=dphi)
    ctx.synchronize()
    _ = dl^
    _ = mp^
    _ = dout^
    _ = ey^
    _ = dphi^
