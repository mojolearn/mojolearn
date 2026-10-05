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
    pshap_synth_unit, pshap_marginal_unit, agn_model_unit, pshap_dcount_unit, scan_step_unit, pshap_dindex_unit,
    pshap_dsynth_unit, pshap_dmap_unit, bg_mean_mapped_unit,
)
from std.memory import bitcast

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
#: lane idn-shap-pca (2026-10-04), the IDENTICAL default on NVIDIA, AMD and
#: Apple (-D MOJOLEARN_AGN_IDN_SYN_POOL_OFF restores the per-chunk buffers):
#: MOJOLEARN_AGN_IDN_SYN_POOL  the explainers' synthetic matrix is built in
#:   the one pooled device buffer (no 100-400 MB device allocation per
#:   chunk; Permutation SHAP on a 220-feature table is one chunk a row) and
#:   the Python glue hands every chunk the same host buffer (no fresh pages
#:   per chunk), the FAST + Apple MOJOLEARN_KSHAP_FAST_BATCH buffer handling.
#:   Bit-inert: the same units write the same words to the same offsets; the
#:   host column builds the matrix in the caller's buffer either way.
comptime AGN_IDN_SYN_POOL = (
    GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL and not (is_defined["MOJOLEARN_AGN_IDN_SYN_POOL_OFF"]() or is_defined["MOJOLEARN_IDN_ALL_OFF"]())
)
comptime _AGN_SYN_POOL = KSHAP_FAST_BATCH or AGN_IDN_SYN_POOL
#: lane fam2-forests (2026-10-04), the IDENTICAL default on NVIDIA, AMD and
#: Apple (-D MOJOLEARN_IDN_SHAP_DEVICE_MODEL_OFF restores the model callback;
#: also off under MOJOLEARN_IDN_ALL_OFF):
#: MOJOLEARN_IDN_SHAP_DEVICE_MODEL  Kernel and Permutation SHAP over one of
#:   this library's flat forests (RandomForest*, ExtraTrees*, DecisionTree*
#:   predicting through the strict increasing-tree kernel) evaluate the
#:   model on the device, inside the explainer's own stream: the forest and
#:   the background are uploaded once per `shap_values` call (`model_load`),
#:   then each chunk is masks/permutations -> `model_kernel` (one thread per
#:   synthetic row, reading x or the background per node, never building the
#:   synthetic matrix) -> background means -> solve / marginals. Before, per
#:   chunk: build the (R m nb) x d synthetic matrix, download it, call the
#:   model's predict (its own upload, kernel, download), upload the outputs.
#:   Bit-inert: `agn_model_unit` is the forest's IDENTICAL predict
#:   arithmetic over the same compared values. The host column and every
#:   other model keep the callback.
comptime AGN_IDN_DEVICE_MODEL = (
    GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL
    and not (is_defined["MOJOLEARN_IDN_SHAP_DEVICE_MODEL_OFF"]() or is_defined["MOJOLEARN_IDN_ALL_OFF"]())
)
#: Default FAST + Apple; MOJOLEARN_PSHAP_DELTA_OFF restores full synthesis.
#: M3 w2-pdelta-pshap-istella: 28,222.8 -> 14,788.6 ms (-47.6%).
#: w2-pdelta-quality: linear/tanh/two-output phi byte-identical, identical
#: additivity errors; model rows 2,713,500 -> 2,215,900. Measured source
#: 20bbdc37256bd00ef09fc97970f79f6967fa4f2a, no arithmetic changes here.
#: Device counts/compaction/mapping, same background mean/marginal order;
#: callback retains the existing chunked, row-independent model contract.
comptime PSHAP_DELTA = _AGN_FAST_APPLE and not is_defined["MOJOLEARN_PSHAP_DELTA_OFF"]()
#: SHAP_PERM_CACHE (FAST + Apple DEFAULT, on top of PSHAP_DELTA; rollback
#: `-D MOJOLEARN_SHAP_PERM_CACHE_OFF`). M3 afc_ab_def, full board size, 1 run
#: per arm, 2026-10-04 (ab1 d51f4b4bf): permutation-shap taxi 102.77 -> 85.42
#: ms, istella 15229.5 -> 14991.0 ms; rel_error_vs_exact identical.
#: `pshap_dsynth` keeps its
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
comptime SHAP_PERM_CACHE = PSHAP_DELTA and not is_defined["MOJOLEARN_SHAP_PERM_CACHE_OFF"]()
#: KSHAP_FAST_OVERLAP (lane apple-fast-s-shap, 2026-10-04; READY-AB, opt-in
#: `-D MOJOLEARN_KSHAP_FAST_OVERLAP`, needs KSHAP_FAST_BATCH): KernelExplainer
#: istella spends ~153 ms a row (15,325 ms / 100 rows) where shap-cpu spends
#: ~77: a row's chunk is 2,048 coalitions x 100 background rows x 220 = 45M
#: floats (180 MB), and `kshap_synth` downloads it with a raw host-pointer
#: copy (~3 GB/s on Apple, ~60 ms) and waits, then the caller's model reads
#: it, then the next chunk starts: the download and the model never overlap.
#: Here `kshap_synth_async` enqueues the NEXT chunk's masks, synthetic rows
#: and download into the caller's other host buffer without waiting, the
#: caller runs the model on the current chunk meanwhile, and
#: `kshap_synth_wait` (or the next call on the stream) waits. The chunk's
#: device buffers wait in KS_PEND until then. The same kernels on the same
#: words in the same order: no bit moves (phi byte-identical expected).
comptime KSHAP_FAST_OVERLAP = KSHAP_FAST_BATCH and is_defined["MOJOLEARN_KSHAP_FAST_OVERLAP"]()
#: PSHAP_FAST_OVERLAP (lane apple-fast-s-shap, 2026-10-04; READY-AB, opt-in
#: `-D MOJOLEARN_PSHAP_FAST_OVERLAP`, needs SHAP_PERM_CACHE): the same overlap
#: for PermutationExplainer istella (~150 ms a row vs shap-cpu's ~123): each
#: row's varying synthetic rows (up to 441,000 x 220 floats, 388 MB) come
#: down by a raw host-pointer copy (~3 GB/s) with the model idle. Here
#: `pshap_dsynth_async` builds chunk r + 1's `_Delta` (its one count sync,
#: with the stream otherwise idle), enqueues its compaction and download into
#: the caller's other host buffer and returns; the model runs on chunk r
#: meanwhile; chunk r's `pshap_dvalues` takes chunk r's cached `_Delta`
#: (SHAP_PERM_CACHE's slot), and `pshap_dsynth_wait` then moves chunk r + 1's
#: into that slot. The same kernels on the same words: no bit moves.
comptime PSHAP_FAST_OVERLAP = SHAP_PERM_CACHE and is_defined["MOJOLEARN_PSHAP_FAST_OVERLAP"]()
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
    (MOJOLEARN_KSHAP_FAST_BATCH, MOJOLEARN_AGN_IDN_SYN_POOL): grown on
    demand, never shrunk while an explanation runs. One pool per tier (its
    buffer belongs to that tier's device context). IDENTICAL frees it when
    the explanation ends (`pool_release`)."""
    var syn: Optional[DeviceBuffer[DType.float32]]
    var cap: Int

    def __init__(out self):
        self.syn = Optional[DeviceBuffer[DType.float32]]()
        self.cap = 0


comptime _AGN_POOL_NAME = (
    "MojoXTreesAgnosticSynPoolIdentical" if GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL
    else "MojoXTreesAgnosticSynPoolFast"
)
comptime AGN_POOL = _Global[StorageType=_AgnPool, name=_AGN_POOL_NAME, init_fn=_AgnPool.__init__]


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


def pool_release() raises:
    """Lane idn-all (the review of idn-shap-pca): frees the pooled synthetic
    buffer. The IDENTICAL explainers call it when a `shap_values` call ends
    (python/mojolearn/_expansion_trees.py), so the pool lives for one
    explanation, reused by its chunks, and never holds the largest chunk for
    the life of the process. Every pool user synchronizes before returning,
    so nothing in flight reads the buffer. The FAST pool is left as it is."""
    comptime if AGN_IDN_SYN_POOL:
        var slot = AGN_POOL.get_or_create_ptr()
        slot[].syn = Optional[DeviceBuffer[DType.float32]]()
        slot[].cap = 0


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
    comptime if _AGN_SYN_POOL:
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


struct _KsPend(Defaultable, Movable):
    """KSHAP_FAST_OVERLAP: the one chunk whose synthetic rows are still in
    flight (its masks and uploads stay alive until `kshap_synth_wait`)."""
    var mk: Optional[_Masks]
    var dx: Optional[DeviceBuffer[DType.float32]]
    var dbg: Optional[DeviceBuffer[DType.float32]]

    def __init__(out self):
        self.mk = Optional[_Masks]()
        self.dx = Optional[DeviceBuffer[DType.float32]]()
        self.dbg = Optional[DeviceBuffer[DType.float32]]()


comptime KS_PEND = _Global[StorageType=_KsPend, name="MojoXTreesKshapPendingFast", init_fn=_KsPend.__init__]


def kshap_synth_wait() raises:
    """KSHAP_FAST_OVERLAP: wait for the chunk `kshap_synth_async` left in
    flight (its rows are then in the caller's host buffer) and release its
    device buffers. A no-op with nothing in flight."""
    var slot = KS_PEND.get_or_create_ptr()
    if slot[].mk:
        _ctx().synchronize()
        slot[].mk = Optional[_Masks]()
        slot[].dx = Optional[DeviceBuffer[DType.float32]]()
        slot[].dbg = Optional[DeviceBuffer[DType.float32]]()


def kshap_synth_async(x: Int, bg: Int, size_off: Int, size_w: Int, cdf: Int, syn: Int, R: Int, nb: Int, d: Int,
                      m: Int, nfixed: Int, nfull: Int, npaired: Int, L: Int, seed: Int, row0: Int,
                      wrand: UInt64) raises:
    """KSHAP_FAST_OVERLAP: `kshap_synth`'s launches and download (the same
    kernels, the pooled device buffer), enqueued WITHOUT the final wait: the
    rows reach the host address syn by `kshap_synth_wait`. Until then the
    caller must not read or free syn, and keeps x, bg and the tables alive
    and unchanged (the uploads are enqueued on the same stream)."""
    var total = R * m * nb * d
    kshap_synth_wait()              # one chunk in flight: the pooled buffer may regrow
    if total <= 0:
        return
    var ctx = _ctx()
    var mk = _Masks(ctx, size_off, size_w, cdf, R, d, m, nfixed, nfull, npaired, L, seed, row0, wrand)
    var dx = _up_f32(ctx, x, R * d)
    var dbg = _up_f32(ctx, bg, nb * d)
    var ps = _pool_syn(ctx, total)
    ctx.enqueue_function[ksynth_kernel](
        Int64(total), Int32(nb), Int32(d), Int32(m), dx.unsafe_ptr(), dbg.unsafe_ptr(), mk.masks.unsafe_ptr(),
        ps, grid_dim=_blocks(total), block_dim=AGN_TPB,
    )
    _pool_down(ctx, syn, total)
    var slot = KS_PEND.get_or_create_ptr()
    slot[].mk = Optional[_Masks](mk^)
    slot[].dx = Optional[DeviceBuffer[DType.float32]](dx^)
    slot[].dbg = Optional[DeviceBuffer[DType.float32]](dbg^)


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
    comptime if _AGN_SYN_POOL:
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


# ------------------------------------------- MOJOLEARN_IDN_SHAP_DEVICE_MODEL
def model_kernel[RF_INPUT: Bool, PERM: Bool](total: Int64, nb: Int32, d: Int32, mc: Int32, k: Int32, trees: Int32,
                                             off: I32P, col: I32P, thr: F32P, left: I32P, leaf: F32P, x: F32P,
                                             bg: F32P, sel: I32P, y: F32P):
    """One thread per synthetic row: the forest on it (`agn_model_unit`)."""
    var t = _t0()
    while t < Int(total):
        agn_model_unit[RF_INPUT, PERM](t, Int(nb), Int(d), Int(mc), Int(k), Int(trees), off, col, thr, left, leaf,
                                       x, bg, sel, y)
        t += _stride()


def nonfinite_kernel(total: Int64, v: F32P, flag: I32P):
    """flag[0] = 1 when any of v's first `total` values is inf or NaN
    (core/forest_inference.mojo `forest_nonfinite_kernel`'s predicate; every
    writer stores the same 1, so the flag has no order)."""
    var t = _t0()
    while t < Int(total):
        if (bitcast[DType.uint32](v[t]) & UInt32(0x7f800000)) == UInt32(0x7f800000):
            flag[0] = Int32(1)
        t += _stride()


def model_check_kernel(trees: Int32, nodes: Int32, d: Int32, off: I32P, col: I32P, left: I32P, flag: I32P):
    """`model_load`'s bounds walk (lane/review-fixes). Index t < trees: tree
    t's offsets (flag[0] = 1 unless 0 <= off[t] < off[t + 1] <= nodes).
    Index trees + i: node i, its tree found by a binary search over the
    offsets (in bounds whatever they hold; its answer counts only when
    flag[0] stays 0); flag[1] = 1 when a split node's local child or feature
    is out of range. Every writer stores the same 1, so no order."""
    var nt = Int(trees)
    var nn = Int(nodes)
    var t = _t0()
    while t < nt + nn:
        if t < nt:
            var base = Int(off[t])
            var end = Int(off[t + 1])
            if base < 0 or end <= base or end > nn:
                flag[0] = Int32(1)
        else:
            var i = t - nt
            var lo = 0
            var hi = nt - 1
            while lo < hi:
                var mid = (lo + hi + 1) // 2
                if Int(off[mid]) <= i:
                    lo = mid
                else:
                    hi = mid - 1
            var base = Int(off[lo])
            var end = Int(off[lo + 1])
            var child = Int(left[i])
            if child != -1:
                var f = Int(col[i])
                if child < 0 or child + 1 >= end - base or f < 0 or f >= Int(d):
                    flag[1] = Int32(1)
        t += _stride()


def _up_i32(ctx: DeviceContext, addr: Int, n: Int) raises -> DeviceBuffer[DType.int32]:
    var b = ctx.enqueue_create_buffer[DType.int32](max(n, 1))
    if n > 0:
        ctx.enqueue_copy(dst_buf=b.create_sub_buffer[DType.int32](0, n), src_ptr=I32P(unsafe_from_address=addr))
    return b^


struct _AgnModel(Defaultable, Movable):
    """The explained forest and the background on the device, for one
    `shap_values` call (`model_load` .. `model_release`). One per tier (its
    buffers belong to that tier's device context)."""
    var off: Optional[DeviceBuffer[DType.int32]]
    var col: Optional[DeviceBuffer[DType.int32]]
    var thr: Optional[DeviceBuffer[DType.float32]]
    var left: Optional[DeviceBuffer[DType.int32]]
    var leaf: Optional[DeviceBuffer[DType.float32]]
    var bg: Optional[DeviceBuffer[DType.float32]]
    var trees: Int
    var k: Int
    var d: Int
    var nb: Int
    var rf_input: Bool
    var loaded: Bool

    def __init__(out self):
        self.off = Optional[DeviceBuffer[DType.int32]]()
        self.col = Optional[DeviceBuffer[DType.int32]]()
        self.thr = Optional[DeviceBuffer[DType.float32]]()
        self.left = Optional[DeviceBuffer[DType.int32]]()
        self.leaf = Optional[DeviceBuffer[DType.float32]]()
        self.bg = Optional[DeviceBuffer[DType.float32]]()
        self.trees = 0
        self.k = 0
        self.d = 0
        self.nb = 0
        self.rf_input = False
        self.loaded = False


comptime _AGN_MODEL_NAME = (
    "MojoXTreesAgnosticModelIdentical" if GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL
    else "MojoXTreesAgnosticModelFast"
)
comptime AGN_MODEL = _Global[StorageType=_AgnModel, name=_AGN_MODEL_NAME, init_fn=_AgnModel.__init__]


def model_release() raises:
    """Frees the device forest and background (the explainers call it when
    `shap_values` ends). Every user synchronizes before returning, so
    nothing in flight reads the buffers."""
    var slot = AGN_MODEL.get_or_create_ptr()
    slot[].off = Optional[DeviceBuffer[DType.int32]]()
    slot[].col = Optional[DeviceBuffer[DType.int32]]()
    slot[].thr = Optional[DeviceBuffer[DType.float32]]()
    slot[].left = Optional[DeviceBuffer[DType.int32]]()
    slot[].leaf = Optional[DeviceBuffer[DType.float32]]()
    slot[].bg = Optional[DeviceBuffer[DType.float32]]()
    slot[].loaded = False


def model_load(off: Int, col: Int, thr: Int, left: Int, leaf: Int, bg: Int, trees: Int, nodes: Int, k: Int, d: Int,
               nb: Int, rf_input: Bool) raises:
    """Uploads a flat forest (offsets Int32 trees + 1, feature ids Int32
    nodes, thresholds Float32 nodes, local left children Int32 nodes, leaf
    values Float32 nodes x k) and the background (Float32 nb x d). The
    caller hands the snapshot the forest's own predict validated (acyclic,
    finite); `model_check_kernel` re-checks on the device only what keeps
    a kernel in bounds. Synchronizes, so the host arrays need
    not outlive the call."""
    if trees < 1 or nodes < 1 or k < 1 or d < 1 or nb < 1:
        raise Error("x_trees_agn_model_load: needs trees, nodes, outputs, features and background rows")
    if nodes > 2147483647 // k or nb > 2147483647 // d:
        raise Error("x_trees_agn_model_load: element counts exceed Int32 range")
    var po = I32P(unsafe_from_address=off)
    if Int(po[0]) != 0 or Int(po[trees]) != nodes:
        raise Error("x_trees_agn_model_load: offsets must cover all nodes")
    model_release()
    var ctx = _ctx()
    var slot = AGN_MODEL.get_or_create_ptr()
    slot[].off = _up_i32(ctx, off, trees + 1)
    slot[].col = _up_i32(ctx, col, nodes)
    slot[].thr = _up_f32(ctx, thr, nodes)
    slot[].left = _up_i32(ctx, left, nodes)
    slot[].leaf = _up_f32(ctx, leaf, nodes * k)
    slot[].bg = _up_f32(ctx, bg, nb * d)
    # lane/review-fixes: the bounds walk over every tree and node runs on
    # the device on the uploaded forest (was a serial host loop per call):
    # one thread per tree and per node, two flags read back.
    var dflag = ctx.enqueue_create_buffer[DType.int32](2)
    var hflag = ctx.enqueue_create_host_buffer[DType.int32](2)
    ctx.enqueue_memset(dflag, Int32(0))
    ctx.enqueue_function[model_check_kernel](
        Int32(trees), Int32(nodes), Int32(d),
        I32P(unsafe_from_address=Int(slot[].off.value().unsafe_ptr())),
        I32P(unsafe_from_address=Int(slot[].col.value().unsafe_ptr())),
        I32P(unsafe_from_address=Int(slot[].left.value().unsafe_ptr())),
        dflag.unsafe_ptr(),
        grid_dim=_blocks(trees + nodes), block_dim=AGN_TPB,
    )
    ctx.enqueue_copy(dst_ptr=hflag.unsafe_ptr(), src_buf=dflag)
    ctx.synchronize()
    var bad_off = hflag.unsafe_ptr().unsafe_load(0) != Int32(0)
    var bad_idx = hflag.unsafe_ptr().unsafe_load(1) != Int32(0)
    _ = dflag^
    _ = hflag^
    if bad_off:
        model_release()
        raise Error("x_trees_agn_model_load: offsets must be strictly increasing")
    if bad_idx:
        model_release()
        raise Error("x_trees_agn_model_load: feature/child index out of bounds")
    slot[].trees = trees
    slot[].k = k
    slot[].d = d
    slot[].nb = nb
    slot[].rf_input = rf_input
    slot[].loaded = True


def _model_need(nb: Int, d: Int, k: Int, who: String) raises:
    """Refuses a call that does not match the loaded model's shapes."""
    var slot = AGN_MODEL.get_or_create_ptr()
    if not slot[].loaded:
        raise Error(who + ": no model loaded (x_trees_agn_model_load)")
    if slot[].nb != nb or slot[].d != d or slot[].k != k:
        raise Error(who + ": background rows/features/outputs differ from the loaded model")


def _model_launch(ctx: DeviceContext, perm: Bool, total: Int, d: Int, mc: Int, x: F32P, sel: I32P, y: F32P,
                  flag: I32P) raises:
    """The loaded forest on `total` synthetic rows of a chunk into y (total
    x k), then the finiteness scan of y (the forest predict's output
    check) into flag."""
    if total <= 0:
        return
    var slot = AGN_MODEL.get_or_create_ptr()
    var nb = slot[].nb
    var k = slot[].k
    var trees = slot[].trees
    var off = I32P(unsafe_from_address=Int(slot[].off.value().unsafe_ptr()))
    var col = I32P(unsafe_from_address=Int(slot[].col.value().unsafe_ptr()))
    var thr = F32P(unsafe_from_address=Int(slot[].thr.value().unsafe_ptr()))
    var left = I32P(unsafe_from_address=Int(slot[].left.value().unsafe_ptr()))
    var leaf = F32P(unsafe_from_address=Int(slot[].leaf.value().unsafe_ptr()))
    var bg = F32P(unsafe_from_address=Int(slot[].bg.value().unsafe_ptr()))
    if slot[].rf_input:
        if perm:
            ctx.enqueue_function[model_kernel[True, True]](
                Int64(total), Int32(nb), Int32(d), Int32(mc), Int32(k), Int32(trees), off, col, thr, left, leaf,
                x, bg, sel, y, grid_dim=_blocks(total), block_dim=AGN_TPB,
            )
        else:
            ctx.enqueue_function[model_kernel[True, False]](
                Int64(total), Int32(nb), Int32(d), Int32(mc), Int32(k), Int32(trees), off, col, thr, left, leaf,
                x, bg, sel, y, grid_dim=_blocks(total), block_dim=AGN_TPB,
            )
    else:
        if perm:
            ctx.enqueue_function[model_kernel[False, True]](
                Int64(total), Int32(nb), Int32(d), Int32(mc), Int32(k), Int32(trees), off, col, thr, left, leaf,
                x, bg, sel, y, grid_dim=_blocks(total), block_dim=AGN_TPB,
            )
        else:
            ctx.enqueue_function[model_kernel[False, False]](
                Int64(total), Int32(nb), Int32(d), Int32(mc), Int32(k), Int32(trees), off, col, thr, left, leaf,
                x, bg, sel, y, grid_dim=_blocks(total), block_dim=AGN_TPB,
            )
    ctx.enqueue_function[nonfinite_kernel](Int64(total * k), y, flag, grid_dim=_blocks(total * k),
                                           block_dim=AGN_TPB)


def kshap_solve_model(x: Int, fx: Int, fnull: Int, size_off: Int, size_w: Int, cdf: Int, phi: Int, R: Int, nb: Int,
                      d: Int, k: Int, m: Int, nfixed: Int, nfull: Int, npaired: Int, L: Int, seed: Int, row0: Int,
                      wrand: UInt64, link: Bool) raises:
    """`kshap_synth` + the model + `kshap_solve` for a chunk, on the device:
    phi (R x d x k binary64, to the host address phi) from the rows x
    (Float32 R x d), their own outputs fx (Float32 R x k) and the linked
    background value fnull (binary64 k), over the loaded forest and
    background. Refuses non-finite rows or outputs as the forest's predict
    does."""
    if R <= 0:
        return
    _model_need(nb, d, k, "x_trees_kshap_solve_model")
    var ctx = _ctx()
    var mk = _Masks(ctx, size_off, size_w, cdf, R, d, m, nfixed, nfull, npaired, L, seed, row0, wrand)
    var dx = _up_f32(ctx, x, R * d)
    var flag = ctx.enqueue_create_buffer[DType.int32](1)
    ctx.enqueue_memset(flag, Int32(0))
    var pflag = flag.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]()
    var pdx = dx.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]()
    ctx.enqueue_function[nonfinite_kernel](Int64(R * d), pdx, pflag, grid_dim=_blocks(R * d), block_dim=AGN_TPB)
    var total = R * m * nb
    var dout = ctx.enqueue_create_buffer[DType.float32](max(total * k, 1))
    var ey = ctx.enqueue_create_buffer[DType.uint64](max(R * m * k, 1))
    if total > 0:
        _model_launch(ctx, False, total, d, m, pdx, mk.masks.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin](),
                      dout.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin](), pflag)
        ctx.enqueue_function[mean_kernel](
            Int64(R * m * k), Int32(nb), Int32(k), dout.unsafe_ptr(), ey.unsafe_ptr(),
            grid_dim=_blocks(R * m * k), block_dim=AGN_TPB,
        )
        if link:
            ctx.enqueue_function[logit_kernel](Int64(R * m * k), ey.unsafe_ptr(), grid_dim=_blocks(R * m * k),
                                               block_dim=AGN_TPB)
    var hflag = ctx.enqueue_create_host_buffer[DType.int32](1)
    ctx.enqueue_copy(dst_ptr=hflag.unsafe_ptr(), src_buf=flag)
    var dfx32 = ctx.enqueue_create_buffer[DType.float32](max(R * k, 1))
    var dfx = ctx.enqueue_create_buffer[DType.uint64](max(R * k, 1))
    _fx_dev(ctx, fx, R, k, link, dfx32, dfx)
    var dnull = _up_u64(ctx, fnull, k)
    _ksolve_core(ctx, mk, ey.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin](),
                 dfx.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin](),
                 dnull.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin](), phi, R, d, k, m)
    var bad = hflag.unsafe_ptr().unsafe_load(0) != Int32(0)
    _ = hflag^
    _ = flag^
    _ = mk^
    _ = dx^
    _ = dout^
    _ = ey^
    _ = dfx32^
    _ = dfx^
    _ = dnull^
    if bad:
        raise Error("resident forest requires finite Float32 values")


def pshap_values_model(x: Int, phi: Int, R: Int, nb: Int, d: Int, k: Int, np: Int, seed: Int,
                       row0: Int) raises:
    """`pshap_synth` + the model + `pshap_values` for a chunk, on the device:
    phi (R x d x k binary64, to the host address phi) from the rows x
    (Float32 R x d), over the loaded forest and background. Refuses
    non-finite rows or outputs as the forest's predict does."""
    var mm = np * (2 * d + 1)
    if R * d * k <= 0 or mm <= 0:
        return
    _model_need(nb, d, k, "x_trees_pshap_values_model")
    var ctx = _ctx()
    var perm = ctx.enqueue_create_buffer[DType.int32](R * np * d)
    var inv = ctx.enqueue_create_buffer[DType.int32](R * np * d)
    _perms(ctx, R, d, np, seed, row0, perm, inv)
    var dx = _up_f32(ctx, x, R * d)
    var flag = ctx.enqueue_create_buffer[DType.int32](1)
    ctx.enqueue_memset(flag, Int32(0))
    var pflag = flag.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]()
    var pdx = dx.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]()
    ctx.enqueue_function[nonfinite_kernel](Int64(R * d), pdx, pflag, grid_dim=_blocks(R * d), block_dim=AGN_TPB)
    var total = R * mm * nb
    var dout = ctx.enqueue_create_buffer[DType.float32](total * k)
    _model_launch(ctx, True, total, d, np, pdx, inv.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin](),
                  dout.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin](), pflag)
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
    var hflag = ctx.enqueue_create_host_buffer[DType.int32](1)
    ctx.enqueue_copy(dst_ptr=hflag.unsafe_ptr(), src_buf=flag)
    ctx.enqueue_copy(dst_ptr=U64P(unsafe_from_address=phi), src_buf=dphi)
    ctx.synchronize()
    var bad = hflag.unsafe_ptr().unsafe_load(0) != Int32(0)
    _ = hflag^
    _ = flag^
    _ = perm^
    _ = inv^
    _ = dx^
    _ = dout^
    _ = ey^
    _ = dphi^
    if bad:
        raise Error("resident forest requires finite Float32 values")


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


struct _DeltaPend(Defaultable, Movable):
    """PSHAP_FAST_OVERLAP: the chunk whose compacted rows are in flight, and
    its SHAP_PERM_CACHE key."""
    var dl: Optional[_Delta]
    var key: List[Int]

    def __init__(out self):
        self.dl = Optional[_Delta]()
        self.key = List[Int]()


comptime AGN_DELTA_PEND = _Global[StorageType=_DeltaPend, name="MojoXTreesAgnosticDeltaPendFast",
                                  init_fn=_DeltaPend.__init__]


def pshap_dsynth_wait() raises:
    """PSHAP_FAST_OVERLAP: wait for the chunk `pshap_dsynth_async` left in
    flight (its rows are then at the caller's host address) and move its
    `_Delta` into SHAP_PERM_CACHE's slot for its `pshap_dvalues`. A no-op
    with nothing in flight."""
    var pend = AGN_DELTA_PEND.get_or_create_ptr()
    if pend[].dl:
        _ctx().synchronize()
        var slot = AGN_DELTA_SLOT.get_or_create_ptr()
        slot[].dl = Optional[_Delta](pend[].dl.take())
        slot[].key = pend[].key.copy()
        pend[].dl = Optional[_Delta]()
        pend[].key = List[Int]()


def pshap_dsynth_async(x: Int, bg: Int, syn: Int, tot: Int, R: Int, nb: Int, d: Int, np: Int, seed: Int,
                       row0: Int) raises:
    """PSHAP_FAST_OVERLAP: `pshap_dsynth` whose compaction and download are
    enqueued WITHOUT the final wait (the count at tot is final on return;
    the rows reach syn by `pshap_dsynth_wait`, or by the next call that
    synchronizes the stream). Leaves SHAP_PERM_CACHE's slot alone (it may
    hold the previous chunk's `_Delta` for its `pshap_dvalues`). Until the
    wait the caller must not read or free syn, and keeps x and bg alive and
    unchanged."""
    if R * np * (2 * d + 1) * nb * d <= 0:
        return
    var pend = AGN_DELTA_PEND.get_or_create_ptr()
    if pend[].dl:
        raise Error("x_trees_pshap_dsynth_async: a chunk is already in flight")
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
    pend[].dl = Optional[_Delta](dl^)
    pend[].key = _delta_key(x, bg, tot, R, nb, d, np, seed, row0)


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
