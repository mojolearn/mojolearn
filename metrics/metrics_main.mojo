"""Every ported metric on one hashed fixture, with an identity card.

    tools/with_build_lock.sh     pixi run mojo run -I . metrics/metrics_main.mojo
    tools/with_identical_mode.sh pixi run mojo run -I . metrics/metrics_main.mojo
    MOJOLEARN_IDENTITY_TRACE=/tmp/metrics.card tools/with_identical_mode.sh pixi run mojo run -I . metrics/metrics_main.mojo

The card (`core/identity_trace.mojo`) carries one stage per metric -- the
INPUT bytes first, then the integer products a metric rests on (the
contingency matrix, the rand pair counts, the trustworthiness rank sum),
then every returned value by its bits -- so a cross-vendor leg diffs stage
by stage with `tools/identity_trace_diff.py` and a difference has an
address: "the contingency matrices agree and the MI bits do not" is a
different finding from "the matrices differ". Tags are unique and carry
no launch parameter.

Not a port: cuML ships one backend and needs no card. This driver is a
CONSTRUCTION plus one Apple device's run; no second vendor has run it.
"""

from max.gpu.host import DeviceBuffer, DeviceContext

from core.identity_trace import IdentityTrace
from metrics.mojo_only.device_io import download_f32, upload_f32, upload_i32
from metrics.mojo_only.fixtures import (
    bits32,
    bits64,
    hashed_floats,
    hashed_pdf,
    hashed_points,
    labels_true_pred,
    u01,
)
from metrics.ported.metrics.accuracy_score import accuracy_score_py
from metrics.ported.metrics.adjusted_rand_index import adjusted_rand_index
from metrics.ported.metrics.completeness_score import completeness_score
from metrics.ported.metrics.entropy import entropy
from metrics.ported.metrics.homogeneity_score import homogeneity_score
from metrics.ported.metrics.kl_divergence import kl_divergence
from metrics.ported.metrics.mutual_info_score import mutual_info_score
from metrics.ported.metrics.r2_score import r2_score_py
from metrics.ported.metrics.rand_index import rand_index
from metrics.ported.metrics.silhouette_score_batched_float import (
    silhouette_score,
)
from metrics.ported.metrics.trustworthiness import trustworthiness_score_traced
from metrics.ported.metrics.v_measure import v_measure
from metrics.ported.stats.detail.mutual_info_score import (
    contingency_matrix_host,
)
from metrics.ported.stats.detail.rand_index import rand_index_counts
from mojo_only.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL


comptime IDENTICAL = GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL
comptime N_LABELS_ROWS = 2053
comptime N_TRUE = 6
comptime N_PRED = 5
comptime N_FLOAT = 2053
comptime N_SIL = 521
comptime D_SIL = 4
comptime K_SIL = 5
comptime N_TRUST = 301
comptime M_TRUST = 6
comptime D_TRUST = 2
comptime K_TRUST = 5


def _mode_name() -> String:
    comptime if IDENTICAL:
        return String("IDENTICAL")
    return String("FAST")


def _rec64(mut trace: IdentityTrace, tag: String, v: Float64) raises:
    var one = List[Float64]()
    one.append(v)
    trace.record_host(tag, one.unsafe_ptr(), 1)
    _ = one^


def _rec_i64(mut trace: IdentityTrace, tag: String, v: Int64) raises:
    var one = List[Int64]()
    one.append(v)
    trace.record_host(tag, one.unsafe_ptr(), 1)
    _ = one^


def _show64(name: String, v: Float64):
    print("  " + name + " = " + String(v) + "  " + bits64(v))


def _show32(name: String, v: Float32):
    print("  " + name + " = " + String(v) + "  " + bits32(v))


def main() raises:
    print("== metrics/metrics_main.mojo [" + _mode_name() + "] ==")
    var ctx = DeviceContext()
    var trace = IdentityTrace()
    trace.header(
        "metrics card, mode " + _mode_name() + ", labels n=" + String(N_LABELS_ROWS)
        + " true=" + String(N_TRUE) + " pred=" + String(N_PRED)
        + ", float n=" + String(N_FLOAT)
        + ", silhouette n=" + String(N_SIL) + " d=" + String(D_SIL) + " k=" + String(K_SIL)
        + ", trust n=" + String(N_TRUST) + " m=" + String(M_TRUST) + " d=" + String(D_TRUST)
        + " k=" + String(K_TRUST)
    )

    # ---- Group A: the label metrics -------------------------------------
    var lp = labels_true_pred(N_LABELS_ROWS, N_TRUE, N_PRED, 0.66, 4099)
    var yt_h = lp[0].copy()
    var yp_h = lp[1].copy()
    trace.record_host("metrics.input.y_true", yt_h.unsafe_ptr(), N_LABELS_ROWS)
    trace.record_host("metrics.input.y_pred", yp_h.unsafe_ptr(), N_LABELS_ROWS)
    var yt = upload_i32(ctx, yt_h)
    var yp = upload_i32(ctx, yp_h)
    var lo = Int32(0)
    var hi = Int32(N_TRUE - 1)
    var cmat = contingency_matrix_host(ctx, yt, yp, N_LABELS_ROWS, lo, hi)
    trace.record_host("metrics.contingency", cmat.unsafe_ptr(), N_TRUE * N_TRUE)
    var ab = rand_index_counts(ctx, yt, yp, N_LABELS_ROWS)
    _rec_i64(trace, "metrics.rand.a", ab[0])
    _rec_i64(trace, "metrics.rand.b", ab[1])

    var acc = accuracy_score_py(ctx, yt, yp, N_LABELS_ROWS)
    trace.record_scalar_f32("metrics.accuracy_score", acc)
    _show32("accuracy_score", acc)
    var ri = rand_index(ctx, yt, yp, N_LABELS_ROWS)
    _rec64(trace, "metrics.rand_index", ri)
    _show64("rand_index", ri)
    var ari = adjusted_rand_index(ctx, yt, yp, N_LABELS_ROWS)
    _rec64(trace, "metrics.adjusted_rand_index", ari)
    _show64("adjusted_rand_index", ari)
    var h_true = entropy(ctx, yt, N_LABELS_ROWS, lo, hi)
    _rec64(trace, "metrics.entropy", h_true)
    _show64("entropy(y_true)", h_true)
    var mi = mutual_info_score(ctx, yt, yp, N_LABELS_ROWS, lo, hi)
    _rec64(trace, "metrics.mutual_info_score", mi)
    _show64("mutual_info_score", mi)
    var hom = homogeneity_score(ctx, yt, yp, N_LABELS_ROWS, lo, hi)
    _rec64(trace, "metrics.homogeneity_score", hom)
    _show64("homogeneity_score", hom)
    var com = completeness_score(ctx, yt, yp, N_LABELS_ROWS, lo, hi)
    _rec64(trace, "metrics.completeness_score", com)
    _show64("completeness_score", com)
    var vm = v_measure(ctx, yt, yp, N_LABELS_ROWS, lo, hi)
    _rec64(trace, "metrics.v_measure", vm)
    _show64("v_measure", vm)

    # ---- Group B: r2 and KL --------------------------------------------
    var y_h = hashed_floats(N_FLOAT, 17, -6, 6)
    var res = hashed_floats(N_FLOAT, 18, -6, 5)
    var yhat_h = List[Float32]()
    for i in range(N_FLOAT):
        yhat_h.append(y_h[i] + res[i])
    trace.record_host("metrics.input.y", y_h.unsafe_ptr(), N_FLOAT)
    trace.record_host("metrics.input.y_hat", yhat_h.unsafe_ptr(), N_FLOAT)
    var dy = upload_f32(ctx, y_h)
    var dyh = upload_f32(ctx, yhat_h)
    var r2 = r2_score_py(ctx, dy, dyh, N_FLOAT)
    trace.record_scalar_f32("metrics.r2_score", r2)
    _show32("r2_score", r2)
    var p_h = hashed_pdf(N_FLOAT, 19, 101)
    var q_h = hashed_pdf(N_FLOAT, 20, 0)
    trace.record_host("metrics.input.p", p_h.unsafe_ptr(), N_FLOAT)
    trace.record_host("metrics.input.q", q_h.unsafe_ptr(), N_FLOAT)
    var dp = upload_f32(ctx, p_h)
    var dq = upload_f32(ctx, q_h)
    var kl = kl_divergence(ctx, dp, dq, N_FLOAT)
    trace.record_scalar_f32("metrics.kl_divergence", kl)
    _show32("kl_divergence", kl)

    # ---- Group C: silhouette -------------------------------------------
    var pts = hashed_points(N_SIL, D_SIL, K_SIL, 23)
    var x_h = pts[0].copy()
    var lab_h = pts[1].copy()
    trace.record_host("metrics.input.X", x_h.unsafe_ptr(), N_SIL * D_SIL)
    trace.record_host("metrics.input.labels", lab_h.unsafe_ptr(), N_SIL)
    var dx = upload_f32(ctx, x_h)
    var dl = upload_i32(ctx, lab_h)
    var ds = ctx.enqueue_create_buffer[DType.float32](N_SIL)
    var sil = silhouette_score(ctx, dx, N_SIL, D_SIL, dl, K_SIL, ds)
    trace.record_device[DType.float32](ctx, "metrics.silhouette_samples", ds, N_SIL)
    trace.record_scalar_f32("metrics.silhouette_score", sil)
    _show32("silhouette_score", sil)
    var samples = download_f32(ctx, ds, N_SIL)
    _show32("silhouette_samples[0]", samples[0])

    # ---- Group D: trustworthiness ---------------------------------------
    var tp = hashed_points(N_TRUST, M_TRUST, 4, 29)
    var tx = tp[0].copy()
    var temb = List[Float32]()
    for i in range(N_TRUST):
        for q in range(D_TRUST):
            temb.append(tx[i * M_TRUST + q] + Float32((u01(i, q, 38) - 0.5) * 0.8))
    trace.record_host("metrics.input.trust_X", tx.unsafe_ptr(), N_TRUST * M_TRUST)
    trace.record_host("metrics.input.trust_X_embedded", temb.unsafe_ptr(), N_TRUST * D_TRUST)
    # ONE traced call: the k-NN's `knn.*` stages, `trust.emb_ind` and
    # `trust.rank_sum` land in this card through the same `seq`.
    var t = trustworthiness_score_traced(ctx, trace, tx, temb, N_TRUST, M_TRUST, D_TRUST, K_TRUST)
    _rec64(trace, "metrics.trustworthiness", t)
    _show64("trustworthiness", t)

    if trace.enabled:
        print("card written: " + trace.path + " (" + String(trace.seq) + " stages)")
    else:
        print("no card (set MOJOLEARN_IDENTITY_TRACE=<path> to emit one)")
    _ = yt_h^
    _ = yp_h^
    _ = cmat^
    _ = y_h^
    _ = yhat_h^
    _ = p_h^
    _ = q_h^
    _ = x_h^
    _ = lab_h^
    _ = tx^
    _ = temb^
    print("== done [" + _mode_name() + "] ==")
