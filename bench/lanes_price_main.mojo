"""What IDENTICAL costs on the six frozen lanes: ONE driver, one lane per run.

    MOJOLEARN_LANES_PRICE_LANE=kde \\
        tools/with_build_lock.sh     pixi run mojo run -I . bench/lanes_price_main.mojo
    MOJOLEARN_LANES_PRICE_LANE=kde \\
        tools/with_identical_mode.sh pixi run mojo run -I . bench/lanes_price_main.mojo

Driven by `tools/lanes_price.sh`, which builds this file ONCE per mode and
then ALTERNATES the two binaries inside one window (F I F I ...). Run by
hand, it prices one lane in whichever mode it was compiled in.

THE LANES, AND THE ENTRY EACH ROUND CALLS (the same call the lane's own
`*_main.mojo` makes; read that file for the fixture's provenance):

    cd        solver/cd_main.mojo        `cd_fit_traced` on `fixture_planted_sparse(n, d, 610)`
    kde       kde/kde_main.mojo          `score_samples` on `train_fixture/query_fixture/weight_fixture`
    linkage   hierarchy/linkage_main.mojo `single_linkage` on `FIX_BLOBS_DUPS`
    svm       svm/svc_main.mojo (card)   `_run_device` (svc_fit + two svc_predict) on `F2.xor`
    metrics   metrics/metrics_main.mojo  every ported metric on the same hashed fixtures
    gemm      bench/gemm_card_main.mojo  `identical_gemm` on one `bench/gemm_shapes.mojo` row

Every fixture builder is IMPORTED from the lane; none is re-spelled here,
with two exceptions that are transcriptions and say so below: the metrics
lane's SIZES (comptime constants inside `metrics_main.mojo`, a file that
carries a `main`) and the gemm card's bit-assembled `_exact` fill (same
reason; `bench/gemm_price_main.mojo` made the same choice).

WHAT ONE ROUND IS. One fit (or score pass) from the lane's public entry on a
fixture built ONCE before the loop and re-initialized where the entry reads
its own output (the cd coefficients start at zero every round). The clock is
the host `perf_counter_ns` around the call with a `ctx.synchronize()` on both
sides, so the number is wall seconds for the whole entry including its host
work and its launches, not device time. An UNTIMED WARM-UP round precedes the
timed ones and is printed as round `warmup` (its seconds are printed too, so
a reader can see what the first call paid, and it is never in the table).

A PRICE RUN IS ALSO A HASH RUN. Every round prints the FNV-1a64 of the
output bytes (the SAME function `core/identity_trace.mojo` uses for the
stage cards, byte at a time, little endian, so a lane's single-output hash
here equals its card's final-stage hash where the card records that buffer
whole). If the hash moves between rounds IN ONE PROCESS that is printed as a
finding: under IDENTICAL it is a contract violation, under FAST it is a
report of a non-deterministic arm. `tools/lanes_price.sh` repeats the
comparison across processes and across the two modes.

THE MODE IS READ FROM THE COMPTIME CONSTANT (`_mode_name()`), never from the
environment or from the flag that was passed. The header and every
`LPRICE` line carry it, and the driver ABORTS a leg whose label disagrees
with the mode it asked for. Three mislabeled measurements were caught by
that witness on 2026-08-23.

LINE FORMAT, seven whitespace fields, parsed by `tools/lanes_price.sh`:

    LPRICE <lane> <mode> <round|warmup> <size> <seconds> <hash16>

Environment:
    MOJOLEARN_LANES_PRICE_LANE    cd | kde | linkage | svm | metrics | gemm (required)
    MOJOLEARN_LANES_PRICE_ROUNDS  timed rounds in this process (default 5; the
                                  script sets 1 and alternates processes instead)
    MOJOLEARN_LANES_PRICE_SMOKE   1 -> tiny sizes, for proving the build, the
                                  witness and the hash; never for a number
    MOJOLEARN_LANES_PRICE_GEMM_SHAPE  row of bench/gemm_shapes.mojo (default 6,
                                  kmeans.dist.4096x64x64, the E3 table's shape)
    MOJOLEARN_LANES_PRICE_SVM_FIXTURE index into svm `all_fixtures()` (default 1, F2.xor)

No pixi task. No number printed by this file is a measurement until
`bench/LANES_PRICE.md`'s clean-window procedure produced it.
"""

from std.memory import bitcast
from std.os import getenv
from std.time import perf_counter_ns

from max.gpu.host import DeviceBuffer, DeviceContext, HostBuffer

from core.identity_trace import FNV_OFFSET, FNV_PRIME, IdentityTrace, _hex16
from mojo_only.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL, numeric_mode_name

# ---- cd --------------------------------------------------------------------
from solver.mojo_only.cd_oracle import fixture_planted_sparse
from solver.ported.solver.cd import CdLaunch, cd_fit_traced
from solver.ported.solvers.params import LOSS_SQRD_LOSS

# ---- kde -------------------------------------------------------------------
from kde.mojo_only.kde_fixture import query_fixture, train_fixture, weight_fixture
from kde.ported.kde.kde import score_samples
from kde.ported.neighbors.kernel_density import (
    host_sum_weights,
    kde_fit_validate,
    kde_validate_data,
    kernel_from_name,
    metric_from_name,
)

# ---- linkage ---------------------------------------------------------------
from hierarchy.mojo_only.linkage_oracle import (
    FIX_BLOBS_DUPS,
    FIX_DUPS,
    build_fixture,
    fixture_d,
    fixture_n,
    fixture_n_clusters,
    fixture_name,
)
from hierarchy.ported.cluster.detail.connectivities import DISTANCE_L2_SQRT_EXPANDED
from hierarchy.ported.hierarchy.linkage import single_linkage

# ---- svm -------------------------------------------------------------------
from svm.mojo_only.svc_check import Fixture, _run_device, all_fixtures

# ---- metrics ---------------------------------------------------------------
from metrics.mojo_only.device_io import upload_f32, upload_i32
from metrics.mojo_only.fixtures import (
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
from metrics.ported.metrics.silhouette_score_batched_float import silhouette_score
from metrics.ported.metrics.trustworthiness import trustworthiness_score_traced
from metrics.ported.metrics.v_measure import v_measure

# ---- gemm ------------------------------------------------------------------
from bench.gemm_shapes import (
    GEMM_SHAPE_COUNT,
    gemm_shape_k,
    gemm_shape_m,
    gemm_shape_n,
    gemm_shape_name,
    gemm_shape_op,
)
from bench.gemm_shapes import OP_NN as TBL_OP_NN
from bench.gemm_shapes import OP_NT as TBL_OP_NT
from bench.gemm_shapes import OP_TN as TBL_OP_TN
from gemm.mojo_only.gemm_identical import identical_gemm
from gemm.mojo_only.gemm_oracle import OP_NN as ORACLE_OP_NN
from gemm.mojo_only.gemm_oracle import OP_NT as ORACLE_OP_NT
from gemm.mojo_only.gemm_oracle import OP_TN as ORACLE_OP_TN


def _mode_name() -> String:
    """The build's tier, from the ONE definition of it.

    Delegates to `numeric_mode_name()` since 2026-08-29. This used to
    be a local two-way `IDENTICAL`-or-`FAST`, written when there were
    two tiers, and it answered "FAST" for a DETERMINISTIC build -- so
    a driver run under the middle tier printed the wrong arm onto
    every line it produced. A correctly-labelled measurement of the
    wrong arm is the failure this tree has been bitten by repeatedly,
    and forty-four copies of a mode label is how it happens.
    """
    return numeric_mode_name()


def _env_int(name: String, default: Int) raises -> Int:
    var s = String(getenv(name))
    if s == "":
        return default
    return Int(atol(s))


# ============================================================================
# The hash: FNV-1a64 over bytes, little endian, the card's function.
# ============================================================================


@always_inline
def _fold_word(h: UInt64, v: UInt64, nbytes: Int) -> UInt64:
    """Fold the low `nbytes` bytes of `v`, least significant first. Byte at
    a time and in memory order on a little-endian machine, so this is the
    same function as `core/identity_trace.mojo::fnv1a64_bytes` applied to
    the value's storage."""
    var out = h
    for i in range(nbytes):
        out = (out ^ ((v >> UInt64(8 * i)) & UInt64(0xFF))) * FNV_PRIME
    return out


def _fold_f32(h: UInt64, v: Float32) -> UInt64:
    return _fold_word(h, UInt64(bitcast[DType.uint32](v)), 4)


def _fold_f64(h: UInt64, v: Float64) -> UInt64:
    return _fold_word(h, bitcast[DType.uint64](v), 8)


def _fold_i32(h: UInt64, v: Int32) -> UInt64:
    # `[[mojo-int-widening-sign-extends]]`: mask after the widen.
    return _fold_word(h, UInt64(Int(v)) & UInt64(0xFFFFFFFF), 4)


def _fold_f32_list(h: UInt64, xs: List[Float32]) -> UInt64:
    var out = h
    for i in range(len(xs)):
        out = _fold_f32(out, xs[i])
    return out


def _hash_device_f32(
    ctx: DeviceContext, h: UInt64, buf: DeviceBuffer[DType.float32], n: Int
) raises -> UInt64:
    var host = ctx.enqueue_create_host_buffer[DType.float32](n)
    if n == len(buf):
        ctx.enqueue_copy(dst_ptr=host.unsafe_ptr(), src_buf=buf)
    else:
        var view = buf.create_sub_buffer[DType.float32](0, n)
        ctx.enqueue_copy(dst_ptr=host.unsafe_ptr(), src_buf=view)
    ctx.synchronize()
    var out = h
    for i in range(n):
        out = _fold_f32(out, host.unsafe_ptr().unsafe_load(i))
    _ = host^
    return out


def _hash_device_i32(
    ctx: DeviceContext, h: UInt64, buf: DeviceBuffer[DType.int32], n: Int
) raises -> UInt64:
    var host = ctx.enqueue_create_host_buffer[DType.int32](n)
    if n == len(buf):
        ctx.enqueue_copy(dst_ptr=host.unsafe_ptr(), src_buf=buf)
    else:
        var view = buf.create_sub_buffer[DType.int32](0, n)
        ctx.enqueue_copy(dst_ptr=host.unsafe_ptr(), src_buf=view)
    ctx.synchronize()
    var out = h
    for i in range(n):
        out = _fold_i32(out, host.unsafe_ptr().unsafe_load(i))
    _ = host^
    return out


# ============================================================================
# The line, and the in-process hash-stability finding.
# ============================================================================


struct Ledger(Movable):
    """What one process saw: every round's hash, so the finding is printed
    once at the end with the rounds that disagreed named."""

    var lane: String
    var size: String
    var hashes: List[UInt64]
    var labels: List[String]

    def __init__(out self, lane: String, size: String):
        self.lane = lane
        self.size = size
        self.hashes = List[UInt64]()
        self.labels = List[String]()

    def emit(mut self, label: String, ns: Int, h: UInt64):
        var secs = Float64(ns) / 1.0e9
        print(
            "LPRICE " + self.lane + " " + _mode_name() + " " + label + " "
            + self.size + " " + String(secs) + " " + _hex16(h)
        )
        self.hashes.append(h)
        self.labels.append(label)

    def verdict(self) raises:
        """The hash must not move between rounds in one process: same
        binary, same bytes in, same mode. Under IDENTICAL a move is a
        contract violation (a reduction read something other than the
        problem shape); under FAST it is a report that the arm is not
        deterministic on this box, which is a finding and not a bug."""
        if len(self.hashes) == 0:
            raise Error("lanes_price: no rounds ran")
        var first = self.hashes[0]
        var moved = String("")
        for i in range(1, len(self.hashes)):
            if self.hashes[i] != first:
                moved += " " + self.labels[i] + "=" + _hex16(self.hashes[i])
        if moved == "":
            print(
                "HASH-STABLE " + self.lane + " " + _mode_name() + " "
                + String(len(self.hashes)) + " rounds all " + _hex16(first)
            )
            return
        var msg = (
            "HASH-MOVED " + self.lane + " " + _mode_name() + " round "
            + self.labels[0] + "=" + _hex16(first) + " vs" + moved
        )
        print(msg)
        comptime if GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL:
            raise Error(
                "lanes_price: " + msg
                + " -- under IDENTICAL the output bytes may not move between"
                " two fits of one fixture in one process. FINDING, not noise."
            )
        print(
            "   (FAST: recorded, not a failure -- the arm is not deterministic"
            " on this box in this mode; the IDENTICAL leg is where it asserts)"
        )


# ============================================================================
# cd
# ============================================================================


def run_cd(ctx: DeviceContext, smoke: Bool, rounds: Int) raises:
    """`solver/cd_main.mojo`'s fit: Lasso, alpha 0.01, fit_intercept, 1000
    epochs, tol 1e-3, no shuffle, on `fixture_planted_sparse(n, d, 610)`.
    Shipped size 2048 x 16 (the main's defaults); smoke 256 x 4."""
    var n = 256 if smoke else 2048
    var d = 4 if smoke else 16
    var alpha = Float32(0.01)
    var l1_ratio = Float32(1.0)
    var epochs = 1000
    var tol = Float32(1.0e-3)
    var fx = fixture_planted_sparse(n, d, 610)
    var x = ctx.enqueue_create_buffer[DType.float32](n * d)
    var y = ctx.enqueue_create_buffer[DType.float32](n)
    var coef = ctx.enqueue_create_buffer[DType.float32](d)
    var resid = ctx.enqueue_create_buffer[DType.float32](n)
    var hx = fx[0].copy()
    var hy = fx[1].copy()
    ctx.enqueue_copy(dst_buf=x, src_ptr=hx.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=y, src_ptr=hy.unsafe_ptr())
    ctx.synchronize()
    var ledger = Ledger("cd", String(n) + "x" + String(d))
    for r in range(rounds + 1):
        # The entry reads `coef` as its starting point: zero it every round
        # so each round is the SAME fit, not a continuation. And `cdFit`
        # MUTATES `x` and `labels` IN PLACE under fit_intercept (centered,
        # then un-centered by `postProcessData`, which does not restore the
        # bits exactly -- `solver/ported/solver/cd.mojo::cd_fit` says so),
        # so both are re-uploaded every round: the first smoke run of this
        # harness printed a warm-up hash that differed from every later
        # round's, which was the un-restored input, not the kernel.
        ctx.enqueue_copy(dst_buf=x, src_ptr=hx.unsafe_ptr())
        ctx.enqueue_copy(dst_buf=y, src_ptr=hy.unsafe_ptr())
        ctx.enqueue_memset(coef, Float32(0.0))
        ctx.synchronize()
        var trace = IdentityTrace.disabled()
        var t0 = perf_counter_ns()
        var res = cd_fit_traced(
            ctx, x, n, d, y, coef, True, epochs, LOSS_SQRD_LOSS, alpha,
            l1_ratio, False, tol, False, trace, "cd", CdLaunch.default(), resid,
            True,
        )
        ctx.synchronize()
        var t1 = perf_counter_ns()
        var h = _hash_device_f32(ctx, FNV_OFFSET, coef, d)
        h = _fold_f32(h, res[1])
        h = _fold_word(h, UInt64(res[0]), 4)
        ledger.emit("warmup" if r == 0 else String(r), t1 - t0, h)
    ledger.verdict()
    _ = hx^
    _ = hy^
    _ = x^
    _ = y^
    _ = coef^
    _ = resid^


# ============================================================================
# kde
# ============================================================================


def run_kde(ctx: DeviceContext, smoke: Bool, rounds: Int) raises:
    """`kde/kde_main.mojo`'s score: gaussian, euclidean, weighted, h = 2.75,
    on `train_fixture(n_train, d, 1)` / `query_fixture(..., 1)` /
    `weight_fixture(n_train, 1)`. Shipped 1024 train x 256 query x 8;
    smoke 128 x 32 x 8. The timed call is `score_samples` (what
    `kde_score_samples_host` calls after validating and uploading); the
    validation and the upload happen once, before the loop, exactly as the
    host entry does them."""
    var n_train = 128 if smoke else 1024
    var n_query = 32 if smoke else 256
    var d = 8
    var bandwidth = Float32(2.75)
    var train = train_fixture(n_train, d, 1)
    var query = query_fixture(train, n_train, n_query, d, 1)
    var weights = weight_fixture(n_train, 1)
    var k = kernel_from_name("gaussian")
    var m = metric_from_name("euclidean")
    kde_fit_validate(n_train, d, bandwidth, k, m, weights, True)
    kde_validate_data(train, n_train, d, m, "train")
    kde_validate_data(query, n_query, d, m, "query")
    var sum_w = host_sum_weights(weights)
    var dtrain = ctx.enqueue_create_buffer[DType.float32](n_train * d)
    var dquery = ctx.enqueue_create_buffer[DType.float32](n_query * d)
    var dweights = ctx.enqueue_create_buffer[DType.float32](n_train)
    var dout = ctx.enqueue_create_buffer[DType.float32](n_query)
    ctx.enqueue_copy(dst_buf=dtrain, src_ptr=train.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=dquery, src_ptr=query.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=dweights, src_ptr=weights.unsafe_ptr())
    ctx.synchronize()
    var ledger = Ledger(
        "kde", String(n_train) + "x" + String(n_query) + "x" + String(d)
    )
    for r in range(rounds + 1):
        ctx.enqueue_memset(dout, Float32(0.0))
        ctx.synchronize()
        var trace = IdentityTrace.disabled()
        var t0 = perf_counter_ns()
        score_samples(
            ctx, dquery, dtrain, dweights, True, dout, n_query, n_train, d,
            bandwidth, sum_w, k, m, Float32(2.0), trace,
        )
        ctx.synchronize()
        var t1 = perf_counter_ns()
        var h = _hash_device_f32(ctx, FNV_OFFSET, dout, n_query)
        ledger.emit("warmup" if r == 0 else String(r), t1 - t0, h)
    ledger.verdict()
    _ = train^
    _ = query^
    _ = weights^
    _ = dtrain^
    _ = dquery^
    _ = dweights^
    _ = dout^


# ============================================================================
# linkage
# ============================================================================


def run_linkage(ctx: DeviceContext, smoke: Bool, rounds: Int) raises:
    """`hierarchy/linkage_main.mojo`'s fit: `single_linkage`, pairwise
    connectivity, L2SqrtExpanded, on `FIX_BLOBS_DUPS` (102 x 5, 3 clusters;
    the card fixture). Smoke: `FIX_DUPS` (48 x 2, 4 clusters). The output
    hashed is the dendrogram `children` then the flat `labels`."""
    var fix = FIX_DUPS if smoke else FIX_BLOBS_DUPS
    var m = fixture_n(fix)
    var d = fixture_d(fix)
    var n_clusters = fixture_n_clusters(fix)
    var hx = build_fixture(ctx, fix)
    var x = ctx.enqueue_create_buffer[DType.float32](m * d)
    ctx.enqueue_copy(dst_buf=x, src_ptr=hx.unsafe_ptr())
    ctx.synchronize()
    var children = ctx.enqueue_create_buffer[DType.int32]((m - 1) * 2)
    var labels = ctx.enqueue_create_buffer[DType.int32](m)
    var ledger = Ledger(
        "linkage", fixture_name(fix) + "." + String(m) + "x" + String(d)
    )
    for r in range(rounds + 1):
        ctx.enqueue_memset(children, Int32(-7))
        ctx.enqueue_memset(labels, Int32(-7))
        ctx.synchronize()
        var t0 = perf_counter_ns()
        var out = single_linkage(
            ctx, x, m, d, n_clusters, DISTANCE_L2_SQRT_EXPANDED, children, labels
        )
        ctx.synchronize()
        var t1 = perf_counter_ns()
        var h = _hash_device_i32(ctx, FNV_OFFSET, children, (m - 1) * 2)
        h = _hash_device_i32(ctx, h, labels, m)
        h = _fold_word(h, UInt64(out.n_boruvka_rounds), 4)
        ledger.emit("warmup" if r == 0 else String(r), t1 - t0, h)
    ledger.verdict()
    _ = hx^
    _ = x^
    _ = children^
    _ = labels^


# ============================================================================
# svm
# ============================================================================


def run_svm(ctx: DeviceContext, smoke: Bool, rounds: Int) raises:
    """`svm/svc_main.mojo`'s card fit: `_run_device` (svc_fit, then
    svc_predict twice: decision function and classes, on the training rows
    plus 37 hashed queries) on `F2.xor` (240 x 2, rbf gamma 0.5, C 10),
    with the card's launch parameters (block_solve_threads 0, tile limit
    1 << 30, predict buffer 200 MiB, no scratch pad, no poison).
    `MOJOLEARN_LANES_PRICE_SVM_FIXTURE` picks another row of
    `all_fixtures()`. The svm lane has no smaller fixture than F2.xor, so
    SMOKE runs the same fixture. The output hashed is the decision function
    (n + 37 floats), the predicted classes, `n_support` and `b`."""
    var which = _env_int("MOJOLEARN_LANES_PRICE_SVM_FIXTURE", 1)
    var fixtures = all_fixtures()
    if which < 0 or which >= len(fixtures):
        raise Error(
            "lanes_price: MOJOLEARN_LANES_PRICE_SVM_FIXTURE out of range 0.."
            + String(len(fixtures) - 1)
        )
    var ledger = Ledger(
        "svm",
        fixtures[which].name + "." + String(fixtures[which].n) + "x"
        + String(fixtures[which].k),
    )
    for r in range(rounds + 1):
        var card = IdentityTrace.disabled()
        var t0 = perf_counter_ns()
        var run = _run_device(
            ctx, fixtures[which], 0, 1 << 30, 200.0, 0, Float32(0.0), card
        )
        ctx.synchronize()
        var t1 = perf_counter_ns()
        var h = _fold_f32_list(FNV_OFFSET, run.decision)
        h = _fold_f32_list(h, run.classes)
        h = _fold_word(h, UInt64(run.model.n_support), 4)
        h = _fold_f32(h, run.model.b)
        ledger.emit("warmup" if r == 0 else String(r), t1 - t0, h)
    ledger.verdict()
    _ = fixtures^


# ============================================================================
# metrics
# ============================================================================

#: TRANSCRIBED from `metrics/metrics_main.mojo` (comptime constants in a
#: file that carries a `main`; same choice `bench/gemm_price_main.mojo` made
#: about `gemm_card_main.mojo`). If that file's sizes move, these must.
comptime MET_N_LABELS_ROWS = 2053
comptime MET_N_TRUE = 6
comptime MET_N_PRED = 5
comptime MET_N_FLOAT = 2053
comptime MET_N_SIL = 521
comptime MET_D_SIL = 4
comptime MET_K_SIL = 5
comptime MET_N_TRUST = 301
comptime MET_M_TRUST = 6
comptime MET_D_TRUST = 2
comptime MET_K_TRUST = 5


def run_metrics(ctx: DeviceContext, smoke: Bool, rounds: Int) raises:
    """`metrics/metrics_main.mojo`'s score pass: every ported metric on the
    same hashed fixtures (same builders, same salts: 4099 / 17,18 / 19,20 /
    23 / 29,38), timed as ONE pass because the lane ships them as one card.
    Shipped sizes are the main's; smoke divides the row counts by eight.
    The output hashed is every returned value by its bits plus the
    silhouette samples buffer, in the main's order."""
    var n_lab = 257 if smoke else MET_N_LABELS_ROWS
    var n_flt = 257 if smoke else MET_N_FLOAT
    var n_sil = 67 if smoke else MET_N_SIL
    var n_tru = 61 if smoke else MET_N_TRUST
    var lp = labels_true_pred(n_lab, MET_N_TRUE, MET_N_PRED, 0.66, 4099)
    var yt_h = lp[0].copy()
    var yp_h = lp[1].copy()
    var yt = upload_i32(ctx, yt_h)
    var yp = upload_i32(ctx, yp_h)
    var lo = Int32(0)
    var hi = Int32(MET_N_TRUE - 1)
    var y_h = hashed_floats(n_flt, 17, -6, 6)
    var res = hashed_floats(n_flt, 18, -6, 5)
    var yhat_h = List[Float32]()
    for i in range(n_flt):
        yhat_h.append(y_h[i] + res[i])
    var dy = upload_f32(ctx, y_h)
    var dyh = upload_f32(ctx, yhat_h)
    var p_h = hashed_pdf(n_flt, 19, 101)
    var q_h = hashed_pdf(n_flt, 20, 0)
    var dp = upload_f32(ctx, p_h)
    var dq = upload_f32(ctx, q_h)
    var pts = hashed_points(n_sil, MET_D_SIL, MET_K_SIL, 23)
    var x_h = pts[0].copy()
    var lab_h = pts[1].copy()
    var dx = upload_f32(ctx, x_h)
    var dl = upload_i32(ctx, lab_h)
    var ds = ctx.enqueue_create_buffer[DType.float32](n_sil)
    var tp = hashed_points(n_tru, MET_M_TRUST, 4, 29)
    var tx = tp[0].copy()
    var temb = List[Float32]()
    for i in range(n_tru):
        for q in range(MET_D_TRUST):
            temb.append(
                tx[i * MET_M_TRUST + q] + Float32((u01(i, q, 38) - 0.5) * 0.8)
            )
    ctx.synchronize()
    var ledger = Ledger(
        "metrics",
        "lab" + String(n_lab) + ".flt" + String(n_flt) + ".sil" + String(n_sil)
        + "x" + String(MET_D_SIL) + ".tru" + String(n_tru) + "x"
        + String(MET_M_TRUST),
    )
    for r in range(rounds + 1):
        ctx.enqueue_memset(ds, Float32(0.0))
        ctx.synchronize()
        var trace = IdentityTrace.disabled()
        var t0 = perf_counter_ns()
        var acc = accuracy_score_py(ctx, yt, yp, n_lab)
        var ri = rand_index(ctx, yt, yp, n_lab)
        var ari = adjusted_rand_index(ctx, yt, yp, n_lab)
        var h_true = entropy(ctx, yt, n_lab, lo, hi)
        var mi = mutual_info_score(ctx, yt, yp, n_lab, lo, hi)
        var hom = homogeneity_score(ctx, yt, yp, n_lab, lo, hi)
        var com = completeness_score(ctx, yt, yp, n_lab, lo, hi)
        var vm = v_measure(ctx, yt, yp, n_lab, lo, hi)
        var r2 = r2_score_py(ctx, dy, dyh, n_flt)
        var kl = kl_divergence(ctx, dp, dq, n_flt)
        var sil = silhouette_score(ctx, dx, n_sil, MET_D_SIL, dl, MET_K_SIL, ds)
        var t = trustworthiness_score_traced(
            ctx, trace, tx, temb, n_tru, MET_M_TRUST, MET_D_TRUST, MET_K_TRUST
        )
        ctx.synchronize()
        var t1 = perf_counter_ns()
        var h = _fold_f32(FNV_OFFSET, acc)
        h = _fold_f64(h, ri)
        h = _fold_f64(h, ari)
        h = _fold_f64(h, h_true)
        h = _fold_f64(h, mi)
        h = _fold_f64(h, hom)
        h = _fold_f64(h, com)
        h = _fold_f64(h, vm)
        h = _fold_f32(h, r2)
        h = _fold_f32(h, kl)
        h = _hash_device_f32(ctx, h, ds, n_sil)
        h = _fold_f32(h, sil)
        h = _fold_f64(h, t)
        ledger.emit("warmup" if r == 0 else String(r), t1 - t0, h)
    ledger.verdict()
    _ = yt_h^
    _ = yp_h^
    _ = y_h^
    _ = yhat_h^
    _ = p_h^
    _ = q_h^
    _ = x_h^
    _ = lab_h^
    _ = tx^
    _ = temb^
    _ = yt^
    _ = yp^
    _ = dy^
    _ = dyh^
    _ = dp^
    _ = dq^
    _ = dx^
    _ = dl^
    _ = ds^


# ============================================================================
# gemm
# ============================================================================


def _gemm_mix(i: Int, salt: Int) -> UInt64:
    """TRANSCRIBED from `bench/gemm_card_main.mojo::_mix` (splitmix64)."""
    var z = (
        UInt64(i + 1) * 0x9E3779B97F4A7C15
        + UInt64(salt + 1) * 0xBF58476D1CE4E5B9
    )
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9
    z = (z ^ (z >> 27)) * 0x94D049BB133111EB
    return z ^ (z >> 31)


def _gemm_exact(i: Int, salt: Int) -> Float32:
    """TRANSCRIBED from `bench/gemm_card_main.mojo::_exact`: `<int below
    2^21> / 2^20`, exact in Float32 on every backend, no host float chain."""
    var num = Int(_gemm_mix(i, salt) % 2097151) - 1048575
    return Float32(num) / Float32(1048576.0)


def _gemm_oracle_op(tbl: Int) -> Int:
    """`bench/gemm_shapes.mojo` numbers NT=0, TN=1, NN=2; the kernel takes
    `gemm_oracle.mojo`'s NN=0, NT=1, TN=2. Same map as
    `bench/gemm_card_main.mojo::_oracle_op`, for the same reason."""
    if tbl == TBL_OP_NT:
        return ORACLE_OP_NT
    if tbl == TBL_OP_TN:
        return ORACLE_OP_TN
    return ORACLE_OP_NN


def run_gemm(ctx: DeviceContext, smoke: Bool, rounds: Int) raises:
    """The gemm lane's host-visible entry `identical_gemm` on one row of
    `bench/gemm_shapes.mojo`, the table with provenance (no shape is invented
    here). Default row 6, `kmeans.dist.4096x64x64` NT, which is the shape
    E3_RESULTS.md priced as the standalone gemm; smoke row 4,
    `pca.transform.8192x4x4`. `MOJOLEARN_LANES_PRICE_GEMM_SHAPE` selects.
    Under FAST the same kernel runs with its pins compiled away, which is
    what the two builds of every other lane do too."""
    var row = _env_int("MOJOLEARN_LANES_PRICE_GEMM_SHAPE", 4 if smoke else 6)
    if row < 0 or row >= GEMM_SHAPE_COUNT:
        raise Error(
            "lanes_price: MOJOLEARN_LANES_PRICE_GEMM_SHAPE out of range 0.."
            + String(GEMM_SHAPE_COUNT - 1)
        )
    var m = gemm_shape_m(row)
    var n = gemm_shape_n(row)
    var k = gemm_shape_k(row)
    var op = _gemm_oracle_op(gemm_shape_op(row))
    var ha = ctx.enqueue_create_host_buffer[DType.float32](m * k)
    var hb = ctx.enqueue_create_host_buffer[DType.float32](k * n)
    ctx.synchronize()
    for i in range(m * k):
        ha.unsafe_ptr().unsafe_store(i, _gemm_exact(i, 11))
    for i in range(k * n):
        hb.unsafe_ptr().unsafe_store(i, _gemm_exact(i, 13))
    var a = ctx.enqueue_create_buffer[DType.float32](m * k)
    var b = ctx.enqueue_create_buffer[DType.float32](k * n)
    var c = ctx.enqueue_create_buffer[DType.float32](m * n)
    ctx.enqueue_copy(dst_buf=a, src_ptr=ha.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=b, src_ptr=hb.unsafe_ptr())
    ctx.synchronize()
    var ledger = Ledger(
        "gemm",
        gemm_shape_name(row) + "." + String(m) + "x" + String(n) + "x" + String(k),
    )
    for r in range(rounds + 1):
        ctx.enqueue_memset(c, Float32(-987654.0))
        ctx.synchronize()
        var t0 = perf_counter_ns()
        identical_gemm(ctx, c, a, b, m, n, k, op)
        ctx.synchronize()
        var t1 = perf_counter_ns()
        var h = _hash_device_f32(ctx, FNV_OFFSET, c, m * n)
        ledger.emit("warmup" if r == 0 else String(r), t1 - t0, h)
    ledger.verdict()
    _ = ha^
    _ = hb^
    _ = a^
    _ = b^
    _ = c^


# ============================================================================
# main
# ============================================================================


def main() raises:
    var mode = _mode_name()
    var lane = String(getenv("MOJOLEARN_LANES_PRICE_LANE"))
    var rounds = _env_int("MOJOLEARN_LANES_PRICE_ROUNDS", 5)
    var smoke = String(getenv("MOJOLEARN_LANES_PRICE_SMOKE")) == "1"
    print(
        "== bench/lanes_price_main.mojo [" + mode + "] lane=" + lane
        + " rounds=" + String(rounds) + " smoke=" + String(smoke) + " =="
    )
    if rounds < 1:
        raise Error("lanes_price: MOJOLEARN_LANES_PRICE_ROUNDS must be >= 1")
    var ctx = DeviceContext()
    if lane == "cd":
        run_cd(ctx, smoke, rounds)
    elif lane == "kde":
        run_kde(ctx, smoke, rounds)
    elif lane == "linkage":
        run_linkage(ctx, smoke, rounds)
    elif lane == "svm":
        run_svm(ctx, smoke, rounds)
    elif lane == "metrics":
        run_metrics(ctx, smoke, rounds)
    elif lane == "gemm":
        run_gemm(ctx, smoke, rounds)
    else:
        raise Error(
            "lanes_price: MOJOLEARN_LANES_PRICE_LANE must be one of"
            " cd kde linkage svm metrics gemm; got '" + lane + "'"
        )
    print("== done [" + mode + "] lane=" + lane + " ==")
