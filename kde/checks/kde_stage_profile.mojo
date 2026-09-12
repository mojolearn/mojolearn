# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Where the time of one IDENTICAL `score_samples` goes, stage by stage.

A timing tool, not a gate: it replays `kde/estimator.mojo::
kde_score_samples_host` and the staged device path of
`kde_score_samples_device` with a `synchronize()` after every stage, on a
synthetic fixture of the classical lane's KDE shapes (100,000 fit rows,
2,000 queries, d = 220 for the Istella-S block and d = 11 for the taxi
block), and prints one `KDE-PROFILE` line per stage per repetition plus
an FNV-1a hash of the scores so two builds can be compared bit for bit.

Run on the box that owns the GPU (never on the Mac):

    pixi run mojo run -I . -D MOJOLEARN_NUMERIC_IDENTICAL=1 kde/checks/kde_stage_profile.mojo
"""

from std.memory import bitcast
from std.time import perf_counter_ns

from max.gpu.host import DeviceBuffer, DeviceContext

from bindings.hostptr import copy_f32, f32_ptr
from checks.numerics import ftz, identical_log, numeric_mode_name
from core.identity_trace import IdentityTrace
from kde.estimator import kde_score_samples_host, kde_score_samples_host_ptr
from kde.impl.distance.distance import pairwise_distance
from kde.impl.distance.distance_ops import DIST_L2_SQRT_UNEXPANDED
from kde.impl.neighbors.kernel_density import (
    KDE_FUSED_ID_K,
    KDE_FUSED_ID_TPB,
    KDE_KERNEL_GAUSSIAN,
    KDE_TILED_CELL,
    KDE_TILED_CHUNK_ROWS,
    KDE_TILED_Q_TPB,
    _launch_fused_sum,
    kde_fused_sum_kernel,
    kde_lse_serial_sum_kernel,
    kde_lse_terms_kernel,
    kde_rowmax_reduce_kernel,
    kde_score_samples_device,
    kde_score_samples_fused_identical,
    kde_score_samples_tiled_identical,
    kde_tiled_logk_kernel,
    kde_validate_data,
    kde_validate_data_ptr,
    log_kernel_matrix_kernel,
    log_kernel_norm,
    logsumexp_kernel,
    normalize_scores_kernel,
)


def _fixture(n: Int, d: Int, seed: UInt64) -> List[Float32]:
    var out = List[Float32](length=n * d, fill=Float32(0))
    var s = seed
    for i in range(n * d):
        s = s * UInt64(6364136223846793005) + UInt64(1442695040888963407)
        var u = Float32(Int((s >> UInt64(40)) & UInt64(0xFFFFFF))) / Float32(16777216.0)
        out[i] = u * Float32(3.4) - Float32(1.7)
    return out^


def _hash(v: List[Float32]) -> UInt64:
    var h = UInt64(0xCBF29CE484222325)
    for i in range(len(v)):
        var b = bitcast[DType.uint32](v[i])
        for k in range(4):
            h = (h ^ UInt64((b >> UInt32(8 * k)) & UInt32(0xFF))) * UInt64(0x100000001B3)
    return h


def _ms(t0: Int, t1: Int) -> Float64:
    return Float64(t1 - t0) / 1e6


def _read_scores(ctx: DeviceContext, mut buf: DeviceBuffer[DType.float32], n: Int) raises -> List[Float32]:
    var host = ctx.enqueue_create_host_buffer[DType.float32](n)
    ctx.enqueue_copy(dst_ptr=host.unsafe_ptr(), src_buf=buf)
    ctx.synchronize()
    var out = List[Float32]()
    for i in range(n):
        out.append(host.unsafe_ptr().unsafe_load(i))
    _ = host^
    return out^


def _say(shape: String, rep: Int, stage: String, ms: Float64):
    print("KDE-PROFILE shape=" + shape + " rep=" + String(rep) + " stage=" + stage + " ms=" + String(ms))


def _upload(ctx: DeviceContext, values: List[Float32]) raises -> DeviceBuffer[DType.float32]:
    var n = len(values)
    var buf = ctx.enqueue_create_buffer[DType.float32](n)
    var host = ctx.enqueue_create_host_buffer[DType.float32](n)
    copy_f32(values.unsafe_ptr(), host.unsafe_ptr(), n)
    ctx.enqueue_copy(dst_buf=buf, src_ptr=host.unsafe_ptr())
    ctx.synchronize()
    _ = host^
    return buf^


def _profile(n_train: Int, n_query: Int, d: Int, reps: Int) raises:
    var shape = String(n_train) + "x" + String(n_query) + "x" + String(d)
    var train = _fixture(n_train, d, UInt64(7))
    var query = _fixture(n_query, d, UInt64(11))
    # Scott's rule on this fixture's size, as the lane's prep writes it.
    var h = Float32(0.95)
    var kernel = KDE_KERNEL_GAUSSIAN
    var metric = DIST_L2_SQRT_UNEXPANDED
    var cells = n_query * n_train
    var none = List[Float32]()

    # The whole host entry, as the binding calls it.
    for rep in range(reps):
        var t0 = perf_counter_ns()
        var out = kde_score_samples_host(
            train, n_train, query, n_query, d, h, "gaussian", "euclidean", none, False,
        )
        var t1 = perf_counter_ns()
        _say(shape, rep, "host_entry_total", _ms(t0, t1))
        print("KDE-PROFILE shape=" + shape + " rep=" + String(rep) + " hash_host_entry=" + String(_hash(out)))

    # DEVIATION 2660: the binding's entry now, over the caller's memory.
    for rep in range(reps):
        var outp = List[Float32](length=n_query, fill=Float32(0))
        var tp = f32_ptr(Int(train.unsafe_ptr()))
        var qp = f32_ptr(Int(query.unsafe_ptr()))
        var op = f32_ptr(Int(outp.unsafe_ptr()))
        var t0 = perf_counter_ns()
        kde_score_samples_host_ptr(
            tp, n_train, qp, n_query, d, h, "gaussian", "euclidean", none, False, op,
        )
        var t1 = perf_counter_ns()
        _say(shape, rep, "host_ptr_entry_total", _ms(t0, t1))
        print("KDE-PROFILE shape=" + shape + " rep=" + String(rep) + " hash_host_ptr_entry=" + String(_hash(outp)))
        t0 = perf_counter_ns()
        kde_validate_data_ptr(tp, n_train, d, metric, "train")
        kde_validate_data_ptr(qp, n_query, d, metric, "query")
        t1 = perf_counter_ns()
        _say(shape, rep, "host_validate_ptr", _ms(t0, t1))

    for rep in range(reps):
        var t0 = perf_counter_ns()
        var tr = train.copy()
        var qu = query.copy()
        var t1 = perf_counter_ns()
        _say(shape, rep, "host_list_copy", _ms(t0, t1))
        t0 = perf_counter_ns()
        kde_validate_data(tr, n_train, d, metric, "train")
        kde_validate_data(qu, n_query, d, metric, "query")
        t1 = perf_counter_ns()
        _say(shape, rep, "host_validate", _ms(t0, t1))
        t0 = perf_counter_ns()
        var ctx = DeviceContext()
        t1 = perf_counter_ns()
        _say(shape, rep, "device_context", _ms(t0, t1))
        t0 = perf_counter_ns()
        var dtrain = _upload(ctx, tr)
        t1 = perf_counter_ns()
        _say(shape, rep, "upload_train", _ms(t0, t1))
        t0 = perf_counter_ns()
        var dquery = _upload(ctx, qu)
        var one = List[Float32]()
        one.append(Float32(1.0))
        var dweights = _upload(ctx, one)
        var dout = ctx.enqueue_create_buffer[DType.float32](n_query)
        ctx.synchronize()
        t1 = perf_counter_ns()
        _say(shape, rep, "upload_query_weights_out", _ms(t0, t1))

        t0 = perf_counter_ns()
        var dist = ctx.enqueue_create_buffer[DType.float32](cells)
        var logk = ctx.enqueue_create_buffer[DType.float32](cells)
        var lse = ctx.enqueue_create_buffer[DType.float32](n_query)
        var rowmax = ctx.enqueue_create_buffer[DType.float32](n_query)
        ctx.synchronize()
        t1 = perf_counter_ns()
        _say(shape, rep, "alloc_two_matrices", _ms(t0, t1))

        t0 = perf_counter_ns()
        pairwise_distance(ctx, dist, dquery, dtrain, n_query, n_train, d, metric, Float32(2.0), 256)
        ctx.synchronize()
        t1 = perf_counter_ns()
        _say(shape, rep, "dev_pairwise_distance", _ms(t0, t1))

        t0 = perf_counter_ns()
        ctx.enqueue_function[log_kernel_matrix_kernel](
            logk.unsafe_ptr(), dist.unsafe_ptr(), Int32(cells), h, Int32(kernel),
            grid_dim=((cells + 255) // 256, 1, 1), block_dim=(256, 1, 1),
        )
        ctx.synchronize()
        t1 = perf_counter_ns()
        _say(shape, rep, "dev_log_kernel_matrix", _ms(t0, t1))

        t0 = perf_counter_ns()
        ctx.enqueue_function[logsumexp_kernel](
            logk.unsafe_ptr(), lse.unsafe_ptr(), rowmax.unsafe_ptr(), Int32(n_query), Int32(n_train),
            grid_dim=((n_query + 127) // 128, 1, 1), block_dim=(128, 1, 1),
        )
        ctx.synchronize()
        t1 = perf_counter_ns()
        _say(shape, rep, "dev_logsumexp_tpb128", _ms(t0, t1))

        t0 = perf_counter_ns()
        var log_sw = ftz(identical_log(Float32(n_train)))
        var norm = log_kernel_norm(kernel, h, d)
        ctx.enqueue_function[normalize_scores_kernel](
            dout.unsafe_ptr(), lse.unsafe_ptr(), Int32(n_query), log_sw, norm,
            grid_dim=((n_query + 255) // 256, 1, 1), block_dim=(256, 1, 1),
        )
        ctx.synchronize()
        t1 = perf_counter_ns()
        _say(shape, rep, "host_norm_and_dev_normalize", _ms(t0, t1))

        t0 = perf_counter_ns()
        var host = ctx.enqueue_create_host_buffer[DType.float32](n_query)
        ctx.enqueue_copy(dst_ptr=host.unsafe_ptr(), src_buf=dout)
        ctx.synchronize()
        var res = List[Float32]()
        for i in range(n_query):
            res.append(host.unsafe_ptr().unsafe_load(i))
        t1 = perf_counter_ns()
        _say(shape, rep, "download", _ms(t0, t1))
        print("KDE-PROFILE shape=" + shape + " rep=" + String(rep) + " hash_staged_replay=" + String(_hash(res)))

        # Scheduling sweeps on the same buffers: the per-row fold's block
        # width, and the per-cell distance's block width.
        var tpbs: List[Int] = [16, 32, 64, 512]
        for t in tpbs:
            var tpb = t
            t0 = perf_counter_ns()
            ctx.enqueue_function[logsumexp_kernel](
                logk.unsafe_ptr(), lse.unsafe_ptr(), rowmax.unsafe_ptr(), Int32(n_query), Int32(n_train),
                grid_dim=((n_query + tpb - 1) // tpb, 1, 1), block_dim=(tpb, 1, 1),
            )
            ctx.synchronize()
            t1 = perf_counter_ns()
            _say(shape, rep, "dev_logsumexp_tpb" + String(tpb), _ms(t0, t1))
        t0 = perf_counter_ns()
        pairwise_distance(ctx, dist, dquery, dtrain, n_query, n_train, d, metric, Float32(2.0), 1024)
        ctx.synchronize()
        t1 = perf_counter_ns()
        _say(shape, rep, "dev_pairwise_distance_tpb1024", _ms(t0, t1))
        t0 = perf_counter_ns()
        pairwise_distance(ctx, dist, dquery, dtrain, n_query, n_train, d, metric, Float32(2.0), 64)
        ctx.synchronize()
        t1 = perf_counter_ns()
        _say(shape, rep, "dev_pairwise_distance_tpb64", _ms(t0, t1))

        # The device entry on already-uploaded buffers (no host staging),
        # staged forced, then the entry's own dispatch.
        var trace = IdentityTrace.disabled()
        t0 = perf_counter_ns()
        kde_score_samples_device(
            ctx, dtrain, dquery, dweights, False, Float32(n_train), n_train, n_query, d,
            h, kernel, metric, dout, trace, 256, 128, Float32(2.0), True,
        )
        ctx.synchronize()
        t1 = perf_counter_ns()
        _say(shape, rep, "device_entry_staged_forced", _ms(t0, t1))
        var staged_scores = _read_scores(ctx, dout, n_query)
        print("KDE-PROFILE shape=" + shape + " rep=" + String(rep) + " hash_device_staged=" + String(_hash(staged_scores)))
        t0 = perf_counter_ns()
        kde_score_samples_device(
            ctx, dtrain, dquery, dweights, False, Float32(n_train), n_train, n_query, d,
            h, kernel, metric, dout, trace,
        )
        ctx.synchronize()
        t1 = perf_counter_ns()
        _say(shape, rep, "device_entry_dispatch", _ms(t0, t1))
        var entry_scores = _read_scores(ctx, dout, n_query)
        print("KDE-PROFILE shape=" + shape + " rep=" + String(rep) + " hash_device_dispatch=" + String(_hash(entry_scores)))

        # DEVIATION 2625's scheduling sweep: query block width x chunk rows.
        var qtpbs: List[Int] = [128, 256, 512]
        var chunks: List[Int] = [256, 1024, 4096]
        for qt in qtpbs:
            for ch in chunks:
                var qtpb = qt
                var chunk = ch
                t0 = perf_counter_ns()
                kde_score_samples_tiled_identical(
                    ctx, dtrain, dquery, dweights, False, Float32(n_train), n_train, n_query, d,
                    h, kernel, metric, dout, 256, 128, qtpb, chunk,
                )
                ctx.synchronize()
                t1 = perf_counter_ns()
                _say(shape, rep, "tiled_q" + String(qtpb) + "_chunk" + String(chunk), _ms(t0, t1))
                var ts = _read_scores(ctx, dout, n_query)
                var same = _hash(ts) == _hash(staged_scores)
                print("KDE-PROFILE shape=" + shape + " rep=" + String(rep) + " tiled_q" + String(qtpb) + "_chunk" + String(chunk) + "_equals_staged=" + String(same))
        var lse_tpbs: List[Int] = [16, 64, 128]
        for lt in lse_tpbs:
            var ltpb = lt
            t0 = perf_counter_ns()
            kde_score_samples_tiled_identical(
                ctx, dtrain, dquery, dweights, False, Float32(n_train), n_train, n_query, d,
                h, kernel, metric, dout, 1024, ltpb, 256, 1024,
            )
            ctx.synchronize()
            t1 = perf_counter_ns()
            _say(shape, rep, "tiled_elem1024_lse" + String(ltpb), _ms(t0, t1))

        # ===================================================================
        # DEVIATION 2690. Where the tiled path's device milliseconds go, one
        # kernel at a time, and what the fused pass costs instead. Each stage
        # is launched on its own and drained, so the entry's total is
        # ATTRIBUTED rather than assumed.
        # ===================================================================
        var n_chunks = (n_train + KDE_TILED_CHUNK_ROWS - 1) // KDE_TILED_CHUNK_ROWS
        var part = ctx.enqueue_create_buffer[DType.float32](n_query * n_chunks)
        ctx.synchronize()
        var qgrid = (n_query + KDE_TILED_Q_TPB - 1) // KDE_TILED_Q_TPB
        # The row-max pass: this kernel with its matrix write elided. The
        # difference against the two writes below IS the write.
        t0 = perf_counter_ns()
        ctx.enqueue_function[kde_tiled_logk_kernel](
            logk.unsafe_ptr(), part.unsafe_ptr(), dquery.unsafe_ptr(), dtrain.unsafe_ptr(),
            dweights.unsafe_ptr(), Int32(n_query), Int32(n_train), Int32(d),
            Int32(n_chunks), Int32(KDE_TILED_CHUNK_ROWS), Int32(0), h, Int32(kernel),
            Int32(metric), Int32(0), Int32(0),
            grid_dim=(qgrid, n_chunks, 1), block_dim=(KDE_TILED_Q_TPB, 1, 1),
        )
        ctx.synchronize()
        t1 = perf_counter_ns()
        _say(shape, rep, "stage_maxpass_no_store", _ms(t0, t1))
        t0 = perf_counter_ns()
        ctx.enqueue_function[kde_rowmax_reduce_kernel](
            rowmax.unsafe_ptr(), part.unsafe_ptr(), Int32(n_query), Int32(n_chunks),
            grid_dim=((n_query + 127) // 128, 1, 1), block_dim=(128, 1, 1),
        )
        ctx.synchronize()
        t1 = perf_counter_ns()
        _say(shape, rep, "stage_rowmax_reduce", _ms(t0, t1))
        # DEVIATION 2691: the three matrix stages in BOTH layouts, each on
        # the matrix its own write just produced. `_qmajor` is `q * n_train
        # + j` (cuML's, and ours before this deviation), `_tmajor` is
        # `j * n_query + q`. Only the addresses differ.
        var layouts: List[Int] = [0, 1]
        for lay in layouts:
            var tr = lay
            var tag = "_qmajor" if tr == 0 else "_tmajor"
            t0 = perf_counter_ns()
            ctx.enqueue_function[kde_tiled_logk_kernel](
                logk.unsafe_ptr(), part.unsafe_ptr(), dquery.unsafe_ptr(), dtrain.unsafe_ptr(),
                dweights.unsafe_ptr(), Int32(n_query), Int32(n_train), Int32(d),
                Int32(n_chunks), Int32(KDE_TILED_CHUNK_ROWS), Int32(0), h, Int32(kernel),
                Int32(metric), Int32(1), Int32(tr),
                grid_dim=(qgrid, n_chunks, 1), block_dim=(KDE_TILED_Q_TPB, 1, 1),
            )
            ctx.synchronize()
            t1 = perf_counter_ns()
            _say(shape, rep, "stage_logk_matrix_write" + tag, _ms(t0, t1))
            t0 = perf_counter_ns()
            ctx.enqueue_function[kde_lse_terms_kernel](
                logk.unsafe_ptr(), rowmax.unsafe_ptr(), Int32(n_query), Int32(n_train), Int32(tr),
                grid_dim=((cells + 255) // 256, 1, 1), block_dim=(256, 1, 1),
            )
            ctx.synchronize()
            t1 = perf_counter_ns()
            _say(shape, rep, "stage_lse_terms" + tag, _ms(t0, t1))
            t0 = perf_counter_ns()
            ctx.enqueue_function[kde_lse_serial_sum_kernel](
                logk.unsafe_ptr(), rowmax.unsafe_ptr(), lse.unsafe_ptr(),
                Int32(n_query), Int32(n_train), Int32(tr),
                grid_dim=((n_query + 127) // 128, 1, 1), block_dim=(128, 1, 1),
            )
            ctx.synchronize()
            t1 = perf_counter_ns()
            _say(shape, rep, "stage_lse_serial_sum" + tag, _ms(t0, t1))
        # And the whole tiled entry in both layouts, default schedule, each
        # hashed against the staged path.
        for lay2 in layouts:
            var tr2 = lay2
            var tag2 = "_qmajor" if tr2 == 0 else "_tmajor"
            t0 = perf_counter_ns()
            kde_score_samples_tiled_identical(
                ctx, dtrain, dquery, dweights, False, Float32(n_train), n_train, n_query, d,
                h, kernel, metric, dout, 256, 128, KDE_TILED_Q_TPB,
                KDE_TILED_CHUNK_ROWS, tr2 == 1,
            )
            ctx.synchronize()
            t1 = perf_counter_ns()
            _say(shape, rep, "tiled_entry" + tag2, _ms(t0, t1))
            var ts2 = _read_scores(ctx, dout, n_query)
            print(
                "KDE-PROFILE shape=" + shape + " rep=" + String(rep)
                + " tiled_entry" + tag2 + "_equals_staged="
                + String(_hash(ts2) == _hash(staged_scores))
            )
        # The fused sum pass ALONE, on the row max just computed: the
        # recompute-and-fold that replaces the three matrix passes above.
        var helpers_k4 = KDE_TILED_CELL // 4
        var fgrid_k4 = (n_query + (KDE_FUSED_ID_TPB // helpers_k4) - 1) // (KDE_FUSED_ID_TPB // helpers_k4)
        t0 = perf_counter_ns()
        _launch_fused_sum[4](
            ctx, lse, rowmax, dquery, dtrain, dweights, n_query, n_train, d,
            False, h, kernel, metric, fgrid_k4, KDE_FUSED_ID_TPB,
        )
        ctx.synchronize()
        t1 = perf_counter_ns()
        _say(shape, rep, "stage_fused_sumpass_k4", _ms(t0, t1))
        var helpers_k8 = KDE_TILED_CELL // 8
        var fgrid_k8 = (n_query + (KDE_FUSED_ID_TPB // helpers_k8) - 1) // (KDE_FUSED_ID_TPB // helpers_k8)
        t0 = perf_counter_ns()
        _launch_fused_sum[8](
            ctx, lse, rowmax, dquery, dtrain, dweights, n_query, n_train, d,
            False, h, kernel, metric, fgrid_k8, KDE_FUSED_ID_TPB,
        )
        ctx.synchronize()
        t1 = perf_counter_ns()
        _say(shape, rep, "stage_fused_sumpass_k8", _ms(t0, t1))

        # The fused entry, over its schedule sweep. Every one of them must
        # hash EQUAL to the staged path: a schedule is not arithmetic.
        var fused_ks: List[Int] = [1, 2, 4, 8, 16]
        var fused_tpbs: List[Int] = [128, 256, 512]
        for a in fused_ks:
            for b in fused_tpbs:
                var kc = a
                var tp = b
                var hlp = KDE_TILED_CELL // kc
                if tp % hlp != 0:
                    continue
                if tp // hlp > 64:
                    continue
                t0 = perf_counter_ns()
                kde_score_samples_fused_identical(
                    ctx, dtrain, dquery, dweights, False, Float32(n_train), n_train, n_query, d,
                    h, kernel, metric, dout, 256, 128, KDE_TILED_Q_TPB,
                    KDE_TILED_CHUNK_ROWS, kc, tp,
                )
                ctx.synchronize()
                t1 = perf_counter_ns()
                _say(shape, rep, "fused_k" + String(kc) + "_tpb" + String(tp), _ms(t0, t1))
                var fs = _read_scores(ctx, dout, n_query)
                print(
                    "KDE-PROFILE shape=" + shape + " rep=" + String(rep)
                    + " fused_k" + String(kc) + "_tpb" + String(tp)
                    + "_equals_staged=" + String(_hash(fs) == _hash(staged_scores))
                )
        _ = part^
        _ = host^
        _ = dist^
        _ = logk^
        _ = lse^
        _ = rowmax^
        _ = dtrain^
        _ = dquery^
        _ = dweights^
        _ = dout^
        _ = ctx^


def main() raises:
    print("== kde/checks/kde_stage_profile.mojo [" + numeric_mode_name() + "] ==")
    _profile(100000, 2000, 220, 3)
    _profile(100000, 2000, 11, 3)
