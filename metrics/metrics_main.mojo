# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Every ported metric on one hashed fixture, with an identity card.

    tools/with_build_lock.sh     pixi run mojo run -I . metrics/metrics_main.mojo
    tools/with_identical_mode.sh pixi run mojo run -I . metrics/metrics_main.mojo
    MOJOLEARN_IDENTITY_TRACE=/tmp/metrics.card tools/with_identical_mode.sh pixi run mojo run -I . metrics/metrics_main.mojo

The card (`core/identity_trace.mojo`) carries, per metric, the INPUT bytes
first, then the integer products a metric rests on (the contingency matrix,
the rand pair counts, the trustworthiness rank sum), then THE INTERMEDIATES
A FINAL SCORE WOULD OTHERWISE ABSORB, then every returned value by its bits
-- so a cross-vendor leg diffs stage by stage with `tools/identity_trace_
diff.py` and a difference has an address: "the contingency matrices agree
and the MI bits do not" is a different finding from "the matrices differ".
Tags are unique and carry no launch parameter.

WHY THE INTERMEDIATES ARE NOT OPTIONAL. A final scalar is a LOSSY hash of
the arithmetic that produced it. `r2 = 1 - sse/ssto` absorbs a last-bit
move in either sum whenever `sse << ssto` -- this lane MEASURED that on
the 4099-row fixture (`derived/stats/detail/scores.mojo::r2_score_parts`)
and gated the sums in the checks while the card recorded only the ratio.
An output-only card is blind in exactly the way three other lanes measured
on 2026-08-23 (NOVELTY_NOTES 13, 14, 15: a fold sabotage that moves 13 of
16 stages with the output bit-identical). Every stage added here is a value
the metric already computed and threw away.

WHAT MAY NOT BE RECORDED HERE. A stage must be a decision or a value the
ALGORITHM owns, never one the SCHEDULER owns. These cards are asserted
LAUNCH-INVARIANT (`regression_metrics_check.mojo`: two block widths x two
grid shapes, same bits), so a stage carrying a block width, a grid shape,
an occupancy or a core count would differ between two legal launches BY
CONSTRUCTION and would break the property the invariance gate exists to
prove. That is `core/identity_trace.mojo` rule 3 read forwards.

Not a port: cuML ships one backend and needs no card. This driver is a
CONSTRUCTION plus one Apple device's run; no second vendor has run it.
"""

from max.gpu.host import DeviceBuffer, DeviceContext

from core.identity_trace import IdentityTrace
from metrics.original.device_io import download_f32, upload_f32, upload_i32
from metrics.original.fixtures import (
    bits32,
    bits64,
    hashed_floats,
    hashed_pdf,
    hashed_points,
    labels_true_pred,
    u01,
)
from metrics.derived.metrics.accuracy_score import accuracy_score_py
from metrics.derived.metrics.adjusted_rand_index import (
    adjusted_rand_index_traced,
)
from metrics.derived.metrics.completeness_score import completeness_score
from metrics.derived.metrics.entropy import entropy_traced
from metrics.derived.metrics.homogeneity_score import homogeneity_score
from metrics.derived.metrics.kl_divergence import kl_divergence_traced
from metrics.derived.metrics.mutual_info_score import mutual_info_score_traced
from metrics.derived.metrics.r2_score import r2_score_py_parts_traced
from metrics.derived.metrics.rand_index import rand_index
from metrics.derived.metrics.silhouette_score_batched_float import (
    silhouette_score,
)
from metrics.derived.metrics.trustworthiness import trustworthiness_score_traced
from metrics.derived.metrics.v_measure import v_measure
from metrics.derived.stats.detail.mutual_info_score import (
    contingency_matrix_host,
)
from metrics.derived.stats.detail.rand_index import rand_index_counts
from original.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL, numeric_mode_name


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
    """The build's tier, from the ONE definition of it.

    Delegates to `numeric_mode_name()` since 2026-08-29; see the note
    on that function. A local two-way IDENTICAL-or-FAST answers "FAST"
    for a DETERMINISTIC build, which mislabels every line the driver
    prints.
    """
    return numeric_mode_name()


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
    # The driver's own read of the matrix. `mutual_info_score_traced` and
    # `adjusted_rand_index_traced` each record the matrix THEIR call
    # consumed (`metrics.mi.contingency`, `metrics.mi_swapped.contingency`,
    # `metrics.ari.contingency`), so this stage is no longer standing in
    # for theirs: the four agreeing is a real internal-consistency check of
    # one device kernel run four times, and ARI's is built at ARI's OWN
    # label range, which need not be this (lo, hi) at all.
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
    # ARI is a DIFFERENCE OF LARGE NUMBERS OVER A DIFFERENCE OF LARGE
    # NUMBERS; `metrics.ari.pair_sums` is the three exact integers those
    # differences are formed from, and `metrics.ari.uniq` is the early-
    # return decision, recorded whether or not it fires.
    var ari = adjusted_rand_index_traced(ctx, trace, yt, yp, N_LABELS_ROWS)
    _rec64(trace, "metrics.adjusted_rand_index", ari)
    _show64("adjusted_rand_index", ari)
    var h_true = entropy_traced(
        ctx, trace, yt, N_LABELS_ROWS, lo, hi, String("metrics.entropy")
    )
    _rec64(trace, "metrics.entropy", h_true)
    _show64("entropy(y_true)", h_true)
    var mi = mutual_info_score_traced(
        ctx, trace, yt, yp, N_LABELS_ROWS, lo, hi, String("metrics.mi")
    )
    _rec64(trace, "metrics.mutual_info_score", mi)
    _show64("mutual_info_score", mi)

    # ===================================================================
    # COMPLETENESS'S TWO OPERANDS, WHICH NO STAGE CARRIED.
    # ===================================================================
    # `homogeneity = MI(true, pred) / H(true)` and `completeness =
    # MI(pred, true) / H(pred)` (RAFT computes completeness as homogeneity
    # with the arguments swapped, and `v_measure` calls it a second time).
    # Homogeneity's two operands were already on the card above. THE
    # SWAPPED PAIR WAS NOT, and it is not a copy of the unswapped pair:
    # MI(pred, true) folds the TRANSPOSED contingency matrix, so the host's
    # serial ascending walk visits the cells in a different sequence and
    # the two are the same quantity in exact arithmetic and not
    # necessarily the same Float32 bits. H(y_pred) is a different
    # histogram over a different label array.
    #
    # With these two recorded, EVERY OPERAND of the three remaining host
    # epilogues is a recorded stage: homogeneity and completeness are one
    # division each, v_measure is two multiplies, an add and a division on
    # their results, all correctly rounded on any host. That is why
    # `homogeneity_score`, `completeness_score` and `v_measure` below are
    # called UNTRACED: tracing them would record MI and entropy four more
    # times under four more prefixes and localize nothing new. Their
    # returned scalars are recorded, so a card whose operands agree and
    # whose homogeneity does not has found something real -- in the
    # duplicate computation, which is the one thing left unrecorded here.
    var h_pred = entropy_traced(
        ctx, trace, yp, N_LABELS_ROWS, lo, hi, String("metrics.entropy_pred")
    )
    _rec64(trace, "metrics.entropy_pred", h_pred)
    _show64("entropy(y_pred)", h_pred)
    var mi_sw = mutual_info_score_traced(
        ctx, trace, yp, yt, N_LABELS_ROWS, lo, hi, String("metrics.mi_swapped")
    )
    _rec64(trace, "metrics.mutual_info_score_swapped", mi_sw)
    _show64("mutual_info_score(y_pred, y_true)", mi_sw)

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
    # ONE call; `r2_parts[3]` IS `r2_score_py`'s return, bit for bit, from
    # the same launch (r2_score.mojo::r2_score_py_parts). The three sums are
    # recorded because the ratio ABSORBS them: `1 - sse/ssto` with `sse <<
    # ssto` rounds a last-bit move in either sum away, MEASURED on this very
    # fixture (scores.mojo::r2_score_parts). They are also the PRE-VALUES of
    # `r2_epilogue`'s two washers -- the `ssto == 0` force_finite branch and
    # `canonicalize_nan` (DEVIATION 657) -- so a divergence that either
    # washer maps onto one recorded r2 still has a stage of its own.
    var r2_parts = r2_score_py_parts_traced(ctx, trace, dy, dyh, N_FLOAT)
    trace.record_scalar_f32("metrics.r2.y_bar", r2_parts[0])
    trace.record_scalar_f32("metrics.r2.sse", r2_parts[1])
    trace.record_scalar_f32("metrics.r2.ssto", r2_parts[2])
    var r2 = r2_parts[3]
    trace.record_scalar_f32("metrics.r2_score", r2)
    _show32("r2.y_bar", r2_parts[0])
    _show32("r2.sse", r2_parts[1])
    _show32("r2.ssto", r2_parts[2])
    _show32("r2_score", r2)
    var p_h = hashed_pdf(N_FLOAT, 19, 101)
    var q_h = hashed_pdf(N_FLOAT, 20, 0)
    trace.record_host("metrics.input.p", p_h.unsafe_ptr(), N_FLOAT)
    trace.record_host("metrics.input.q", q_h.unsafe_ptr(), N_FLOAT)
    var dp = upload_f32(ctx, p_h)
    var dq = upload_f32(ctx, q_h)
    # The traced entry records `metrics.kl.partials` (every per-term log
    # product, folded once per chunk) and `metrics.kl.sum_raw` (the fold
    # BEFORE `canonicalize_nan` washes a NaN to one payload). Between the
    # recorded p, q and the recorded answer there was nothing at all.
    var kl = kl_divergence_traced(ctx, trace, dp, dq, N_FLOAT)
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
