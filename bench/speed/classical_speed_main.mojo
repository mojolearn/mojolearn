"""How fast our FAST path is on the classical lanes: ONE driver, ONE lane per run.

    MOJOLEARN_SPEED_LANE=kmeans \\
        tools/with_build_lock.sh pixi run mojo run -I . bench/speed/classical_speed_main.mojo

This is the FAST arm and only the FAST arm. It is built WITHOUT
`-D MOJOLEARN_NUMERIC_IDENTICAL=1`, which is what an NVIDIA user who
installed us would get, and it is the arm the cuML / cuVS / cuSOLVER
comparison in `tools/speed_cuml_arm.py` is against. `bench/LANES_PRICE.md`
is the OTHER question (what conforming costs) and no number from that file
belongs beside a number from this one.

WHY THE MODE IS STILL PRINTED, AND WHERE IT COMES FROM. `_mode_name()` reads
the COMPTIME constant `GLOBAL_NUMERIC_MODE`, never the environment and never
the flag that was passed. Three mislabeled measurements were caught by that
witness on 2026-08-23 and the header carries it so a run that was
accidentally built under the identical wrapper is visible as `mode=IDENTICAL`
in the output rather than invisible in the shell history.

THE LANES, AND THE ENTRY EACH ROUND CALLS. Every fixture builder is IMPORTED
from the lane it belongs to; none is re-spelled here. Where a size lives
inside a file that carries its own `main` it is TRANSCRIBED and says so at
the site, which is the same choice `bench/lanes_price_main.mojo` made for the
metrics lane and `bench/gemm_price_main.mojo` made for the gemm card.

    kmeans      cluster/ported/cluster/kmeans::fit         bench/bench_main.mojo's shape
    dbscan      dbscan/ported/dbscan/dbscan::dbscan_fit_impl
    pca         decomposition/ported/linalg/detail/pca::pca_fit
    ols         glm/ported/linalg/detail/lstsq::lstsq_eig
    knn         neighbors/ported/.../knn_brute_force::brute_force_knn_impl
    cd          solver/ported/solver/cd::cd_fit_traced     Lasso
    kde         kde/ported/kde/kde::score_samples
    linkage     hierarchy/ported/hierarchy/linkage::single_linkage
    svm         svm/ported/svm/svc_impl::svc_fit           FIT ONLY, see below
    metrics     metrics/ported/metrics/*                   eleven metrics, one pass
    ivf         ivf/estimator::ivf_flat_build_and_search_host
    hdbscan     hdbscan/ported/hdbscan/runner::fit_hdbscan
    cholesky    cholesky/mojo_only/potrf::potrf_lower + trsm::cho_solve
    gmm         mixture/estimator::gaussian_mixture_fit
    gp          gaussian_process/estimator::gpr_fit_host + gpr_predict_host
    krr         kernel_methods/estimator::kernel_ridge_fit_host + predict
    nystroem    kernel_methods/estimator::nystroem_fit_host + transform
    rbfsampler  kernel_methods/estimator::rbf_sampler_fit_host + transform
    resample    resample/estimator::bootstrap_host          the bootstrap
    spectral    spectral/ported/.../spectral::fit_predict_dataset
    holtwinters holtwinters/estimator::holtwinters_fit_host_traced
    kpss        tsa/ported/tsa/stationarity::kpss_test

WHAT ONE ROUND IS. One fit (or one score pass) through the lane's public
entry on a fixture built ONCE, before the loop, and re-initialized where the
entry reads its own output. The clock is the host `perf_counter_ns` around
the call with a `ctx.synchronize()` on BOTH sides, because on this stack an
unsynchronized timing measures enqueue rate and nothing else. One UNTIMED
warm-up round precedes the timed ones and is printed as `FSPEED-WARMUP` so a
reader can see what the first call paid; it is never in the table.

WHAT IS INSIDE THE TIMED REGION AND WHAT IS NOT, STATED RATHER THAN BURIED.
For the lanes whose entry takes DEVICE BUFFERS (kmeans, dbscan, pca, ols,
knn, cd, kde, linkage, cholesky, kpss) the host-to-device upload happens once
before the loop and the timed region is compute only. SIX LANES HAVE NO SUCH
ENTRY: `ivf`, `gmm`, `gp`, `krr`, `nystroem`, `rbfsampler`, `holtwinters`,
`resample`, `spectral`, `svm` and `hdbscan` take HOST lists, so their timed
region includes whatever upload and download the entry itself performs, and
four of them (`gmm`, `gp`, `krr`, `nystroem`, `rbfsampler`, and
`holtwinters`'s untraced form) construct their own `DeviceContext` inside the
call. That context construction is real cost a caller pays and it is NOT
compute. Every one of those lanes says so in its own docstring and
`bench/speed/CLASSICAL_SPEED.md` repeats the list, because a reader comparing
a 3 ms `gmm` against a 3 ms `cuml` arm has to know that most of our 3 ms is a
context.

THE SHIPPED SIZE IS THE DEFAULT AND SOME SHIPPED SIZES ARE TINY. Five lanes
(kmeans, dbscan, pca, ols, knn) run `bench/bench_main.mojo`'s shapes, which
are millions of rows. The rest run the fixture their own lane ships, and
several of those fixtures are CORRECTNESS fixtures of a few dozen rows --
`gp` is 12 x 3, `gmm` is 24 x 2, `hdbscan` is 96 x 4, `cholesky` is 64 x 64.
At those sizes the number is per-call FIXED COST, not throughput, and it
answers a different question from the one the big lanes answer. It is still
the measurement: a house rule forbids picking, dropping, deferring or tuning
a benchmark dataset by whether it flatters us, and inventing a bigger
fixture for a lane whose fixture builder has no size knob would be inventing
a dataset. `size=` on every line says which regime the number came from and
`MOJOLEARN_SPEED_SIZE=smoke` shrinks only the five that have a knob.

THE SVM LANE TIMES `svc_fit` AND NOT `_run_device`. `bench/lanes_price_main.
mojo`'s svm lane times a fit plus TWO predicts over `n + 37` queries, which
is the right shape for pricing identity and the wrong shape for racing
`cuml.svm.SVC.fit`. This lane times the fit alone so the opponent is
`SVC(...).fit(X, y)` and nothing else. THE TWO NUMBERS ARE NOT COMPARABLE
WITH EACH OTHER; do not put them in one table.

THE OTHER SIDE GETS THE SAME BYTES. For the five `bench_main.mojo` lanes the
opponent regenerates the data from the same splitmix64 mixer, exactly as
`bench/bench_sklearn.py` already does, and `tools/speed_cuml_arm.py` IMPORTS
`u01` from that file rather than re-spelling it. For every other lane the
fixture is a Mojo builder with no Python twin, so re-spelling it in numpy
would be inventing a second fixture that drifts. Instead:

    MOJOLEARN_SPEED_DUMP=/tmp/speedfix MOJOLEARN_SPEED_LANE=kde \\
        pixi run mojo run -I . bench/speed/classical_speed_main.mojo

writes `/tmp/speedfix/kde.fixture`: every input array as one hex word per
line (float32 by its BITS, because `String(Float32)` does not round trip in
this toolchain) plus the lane's scalar parameters. `tools/speed_cuml_arm.py`
reads that file. The two sides then hold IDENTICAL BYTES by construction and
not by two transcriptions agreeing. Dump once, race after.

THE OUTPUT CONTRACT. Four line kinds, whitespace delimited, and nothing else
appears on those lines:

    FSPEED-HEADER family=classical lane=<l> arm=ours mode=<FAST|IDENTICAL> \\
        device=<name> rounds=<n> size=<shipped|smoke>
    FSPEED-WARMUP lane=<l> arm=ours shape=<tag> ms=<float>
    FSPEED lane=<l> arm=ours shape=<tag> round=<i> ms=<float> hash=<16 hex>
    FSPEED-NOTE lane=<l> arm=ours hash moved across rounds: <h1> <h2>

`hash` is the FNV-1a64 of the output bytes, byte at a time and little endian
-- the same function `core/identity_trace.mojo::fnv1a64_bytes` uses for the
stage cards, so a lane's hash here is comparable with its card's final stage
where the card records that buffer whole. UNDER FAST THE HASH IS ALLOWED TO
MOVE between rounds. If it does, the note is printed and the run CONTINUES:
that is a report about a non-deterministic arm, which is what FAST is, and it
is not a failure. The IDENTICAL leg is where a moving hash is a violation and
this file is not that leg.

Environment:
    MOJOLEARN_SPEED_LANE    required; one of the lane names above
    MOJOLEARN_SPEED_ROUNDS  timed rounds (default 5)
    MOJOLEARN_SPEED_SIZE    `shipped` (default) or `smoke`
    MOJOLEARN_SPEED_DUMP    directory; writes <lane>.fixture and exits normally

ONE FILE, ONE COMPILE UNIT, AND WHAT THAT COSTS. One lane per PROCESS keeps
a lane's RUN-TIME failure from taking the others down, which is what the
rented hour needs. It does not keep a lane's COMPILE failure from taking them
down, because Mojo compiles every import in this file whether the selected
lane uses it or not, and most of these lanes have never been compiled for
CUDA at all. The per-lane import blocks below are separated by banners for
exactly that reason: if the build fails inside one lane, comment out that
lane's import block and its `run_*` function and its `elif` arm, rebuild, and
the other twenty-one still run. Record the amputation as a REFUSED lane in
the results rather than dropping it silently.
"""

from std.memory import bitcast
from std.os import getenv
from std.time import perf_counter_ns

from max.gpu.host import DeviceBuffer, DeviceContext, HostBuffer

from core.identity_trace import FNV_OFFSET, FNV_PRIME, IdentityTrace, _hex16
from mojo_only.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL

# ---- kmeans ----------------------------------------------------------------
from cluster.ported.cluster.kmeans import fit as kmeans_fit_api
from cluster.ported.cluster.kmeans_params import (
    INIT_ARRAY,
    KMeansParams,
    METRIC_L2_EXPANDED,
)
from mojo_only.fixed_point import choose_scale

# ---- dbscan ----------------------------------------------------------------
from dbscan.ported.dbscan.dbscan import dbscan_fit_impl

# ---- pca -------------------------------------------------------------------
from decomposition.ported.linalg.detail.pca import pca_fit

# ---- ols -------------------------------------------------------------------
from glm.ported.linalg.detail.lstsq import lstsq_eig

# ---- knn -------------------------------------------------------------------
from neighbors.ported.neighbors.detail.knn_brute_force import (
    brute_force_knn_impl,
    compute_norms,
)

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
    fixture_as_list,
    fixture_d,
    fixture_n,
    fixture_n_clusters,
    fixture_name,
)
from hierarchy.ported.cluster.detail.connectivities import DISTANCE_L2_SQRT_EXPANDED
from hierarchy.ported.hierarchy.linkage import single_linkage

# ---- svm -------------------------------------------------------------------
from svm.mojo_only.svc_check import all_fixtures
from svm.ported.svm.smosolver import SmoTrace
from svm.ported.svm.svc_impl import svc_fit

# ---- metrics ---------------------------------------------------------------
from metrics.mojo_only.device_io import upload_f32 as met_upload_f32
from metrics.mojo_only.device_io import upload_i32 as met_upload_i32
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
from metrics.ported.metrics.silhouette_score_batched_float import silhouette_score
from metrics.ported.metrics.trustworthiness import trustworthiness_score_traced
from metrics.ported.metrics.v_measure import v_measure

# ---- ivf -------------------------------------------------------------------
from ivf.estimator import ivf_flat_build_and_search_host
from ivf.mojo_only.ivf_fixture import ivf_index_fixture, ivf_query_fixture

# ---- hdbscan ---------------------------------------------------------------
from hdbscan.mojo_only.hdbscan_fixture import (
    HFIX_BLOBS,
    hfixture_as_list,
    hfixture_d,
    hfixture_min_cluster_size,
    hfixture_min_samples,
    hfixture_n,
    hfixture_name,
)
from hdbscan.ported.hdbscan.detail.select import CLUSTER_SELECTION_EOM
from hdbscan.ported.hdbscan.runner import (
    GRAPH_BUILD_BRUTE_FORCE_KNN,
    HDBSCANParams,
    fit_hdbscan,
)

# ---- cholesky --------------------------------------------------------------
from cholesky.mojo_only.cholesky_fixture import (
    FIX_RBF,
    chol_fixture,
    chol_fixture_n,
    chol_fixture_name,
    chol_rhs_fixture,
)
from cholesky.mojo_only.potrf import (
    CHOL_ELEM_TPB,
    CHOL_NB_PINNED,
    CHOL_PANEL_TPB,
    add_jitter,
    chol_jitter_pinned,
    chol_logdet,
    chol_workspace_floats,
    potrf_lower,
)
from cholesky.mojo_only.trsm import CHOL_SOLVE_TPB, cho_solve

# ---- gmm -------------------------------------------------------------------
from mixture.estimator import (
    COV_FULL,
    INIT_KMEANS,
    GmmParams,
    gaussian_mixture_fit,
)
from mixture.mojo_only.gmm_fixture import (
    FIX_SEPARATED,
    gmm_fixture,
    gmm_fixture_d,
    gmm_fixture_k,
    gmm_fixture_n,
    gmm_fixture_name,
)

# ---- gp --------------------------------------------------------------------
from gaussian_process.estimator import gpr_fit_host, gpr_predict_host
from gaussian_process.mojo_only.gp_fixture import (
    GP_FIX_ARD,
    gp_fixture_alpha,
    gp_fixture_d,
    gp_fixture_kernel,
    gp_fixture_n,
    gp_fixture_n_star,
    gp_fixture_name,
    gp_fixture_x,
    gp_fixture_x_star,
    gp_fixture_y,
)

# ---- krr / nystroem / rbfsampler -------------------------------------------
from kernel_methods.estimator import (
    kernel_ridge_fit_host,
    kernel_ridge_predict_host,
    nystroem_fit_host,
    nystroem_transform_host,
    rbf_sampler_fit_host,
    rbf_sampler_transform_host,
)
from kernel_methods.mojo_only.km_fixture import (
    FIX_KM_RBF,
    km_fixture_d,
    km_fixture_n,
    km_fixture_name,
    km_fixture_query,
    km_fixture_x,
    km_fixture_y,
)
from kernel_methods.mojo_only.kernel_matrix import KM_KERNEL_RBF
from svm.ported.svm.svm_parameter import KernelParams

# ---- resample --------------------------------------------------------------
from resample.estimator import bootstrap_host
from resample.mojo_only.intervals import ALT_TWO_SIDED, METHOD_PERCENTILE
from resample.mojo_only.resample_fixture import (
    FIX_HASHED,
    build_sample,
    fixture_d as resample_fixture_d,
    fixture_n as resample_fixture_n,
)
from resample.mojo_only.statistics import STAT_MEAN

# ---- spectral --------------------------------------------------------------
from spectral.mojo_only.spectral_fixture import blobs_fixture
from spectral.ported.cuvs.cluster.detail.spectral import (
    SpectralClusteringParams,
    fit_predict_dataset,
)

# ---- holtwinters -----------------------------------------------------------
from holtwinters.estimator import holtwinters_fit_host_traced
from holtwinters.mojo_only.hw_fixture import hw_fixture, spec_additive
from holtwinters.ported.holtwinters.runner import HW_DEFAULT_EPS

# ---- kpss ------------------------------------------------------------------
from tsa.mojo_only.fixtures import kpss_fixture
from tsa.mojo_only.fixtures import upload_f32 as tsa_upload_f32
from tsa.ported.tsa.stationarity import kpss_test


# ============================================================================
# The mode witness, the environment, and the two small string helpers.
# ============================================================================


def _mode_name() -> String:
    """The mode this binary COMPILED in, read from the comptime constant.

    NOT from `MOJOLEARN_NUMERIC_IDENTICAL`, NOT from the wrapper script, NOT
    from anything a shell could have got wrong. Copied from
    `bench/lanes_price_main.mojo::_mode_name`, which is where the witness was
    introduced after three measurements were labeled with the mode the
    operator MEANT rather than the one that ran.
    """
    comptime if GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL:
        return String("IDENTICAL")
    return String("FAST")


def _env_int(name: String, default: Int) raises -> Int:
    var s = String(getenv(name))
    if s == "":
        return default
    return Int(atol(s))


def _no_spaces(s: String) -> String:
    """A device name with its spaces turned into underscores.

    `device=` is one whitespace-delimited field and `ctx.name()` returns
    things like `Apple M4` and `NVIDIA H100 80GB HBM3`. A parser splitting on
    whitespace would read the second word as a new field, so the spaces go.
    """
    var out = String("")
    for i in range(s.byte_length()):
        var c = String(s[byte=i])
        out += "_" if c == " " else c
    return out


comptime HEX_DIGITS = "0123456789abcdef"


def _hex8(u: UInt32) -> String:
    """Eight lowercase hex digits, most significant first."""
    var out = String("")
    for i in range(8):
        var nib = Int((u >> UInt32(28 - 4 * i)) & UInt32(0xF))
        out += String(HEX_DIGITS[byte=nib])
    return out


# ============================================================================
# The hash: FNV-1a64 over bytes, little endian, the identity card's function.
#
# TRANSCRIBED from `bench/lanes_price_main.mojo`, which transcribed it from
# `core/identity_trace.mojo::fnv1a64_bytes`. It is spelled out rather than
# imported because `lanes_price_main.mojo` carries a `main` and importing a
# second `main` into this compile unit is a linker question nobody needs to
# answer at 2 a.m. on a rented box. If that file's fold changes, this must.
# ============================================================================


@always_inline
def _fold_word(h: UInt64, v: UInt64, nbytes: Int) -> UInt64:
    """Fold the low `nbytes` bytes of `v`, least significant byte first, which
    is byte order in memory on a little-endian machine and therefore the same
    function as `fnv1a64_bytes` applied to the value's storage."""
    var out = h
    for i in range(nbytes):
        out = (out ^ ((v >> UInt64(8 * i)) & UInt64(0xFF))) * FNV_PRIME
    return out


def _fold_f32(h: UInt64, v: Float32) -> UInt64:
    return _fold_word(h, UInt64(bitcast[DType.uint32](v)), 4)


def _fold_f64(h: UInt64, v: Float64) -> UInt64:
    return _fold_word(h, bitcast[DType.uint64](v), 8)


def _fold_i32(h: UInt64, v: Int32) -> UInt64:
    # `[[mojo-int-widening-sign-extends]]`: mask AFTER the widen or every
    # negative label folds eight bytes of sign instead of four bytes of value.
    return _fold_word(h, UInt64(Int(v)) & UInt64(0xFFFFFFFF), 4)


def _fold_u32(h: UInt64, v: UInt32) -> UInt64:
    return _fold_word(h, UInt64(v), 4)


def _fold_f32_list(h: UInt64, xs: List[Float32]) -> UInt64:
    var out = h
    for i in range(len(xs)):
        out = _fold_f32(out, xs[i])
    return out


def _fold_f64_list(h: UInt64, xs: List[Float64]) -> UInt64:
    var out = h
    for i in range(len(xs)):
        out = _fold_f64(out, xs[i])
    return out


def _fold_i32_list(h: UInt64, xs: List[Int32]) -> UInt64:
    var out = h
    for i in range(len(xs)):
        out = _fold_i32(out, xs[i])
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


def _hash_device_u32(
    ctx: DeviceContext, h: UInt64, buf: DeviceBuffer[DType.uint32], n: Int
) raises -> UInt64:
    var host = ctx.enqueue_create_host_buffer[DType.uint32](n)
    if n == len(buf):
        ctx.enqueue_copy(dst_ptr=host.unsafe_ptr(), src_buf=buf)
    else:
        var view = buf.create_sub_buffer[DType.uint32](0, n)
        ctx.enqueue_copy(dst_ptr=host.unsafe_ptr(), src_buf=view)
    ctx.synchronize()
    var out = h
    for i in range(n):
        out = _fold_u32(out, host.unsafe_ptr().unsafe_load(i))
    _ = host^
    return out


# ============================================================================
# The fixture dump.
#
# WHY THIS EXISTS. The opponent has to run on the SAME BYTES. For the five
# lanes that came from `bench/bench_main.mojo` that is already solved: their
# data is a splitmix64 recurrence and `bench/bench_sklearn.py` has the
# vectorized twin, proven bit-identical. Every other lane's fixture is a Mojo
# builder with no Python counterpart, and writing one would be writing a
# SECOND fixture that agrees today and drifts on the first edit. So the Mojo
# side, which owns the fixture, writes the bytes out and the Python side reads
# them. One source, no transcription, and a `diff` can prove it.
#
# FLOATS GO OUT AS BITS. `[[mojo-string-float-roundtrip]]`: `String(Float32)`
# does not round trip in this toolchain, so a decimal dump would hand the
# opponent a DIFFERENT dataset while looking correct.
# ============================================================================


comptime DUMP_CHUNK = 2048
"""Values per file append. One `String` per array would be fine at these
sizes and is not fine at the kde lane's eleven thousand; chunking keeps the
concatenation linear."""


struct Dump(Movable):
    """The lane's inputs, as hex words, for `tools/speed_cuml_arm.py`.

    Off unless `MOJOLEARN_SPEED_DUMP` names a directory. The file is
    `<dir>/<lane>.fixture` and its grammar is three line kinds:

        PARAM     <name> <decimal integer or plain token>
        PARAMHEX  <name> <8 hex digits>          a float32 by its BITS
        ARRAY     <name> <f32|i32> <count>       then `count` hex-word lines
    """

    var path: String
    var on: Bool

    def __init__(out self, lane: String) raises:
        var dir = String(getenv("MOJOLEARN_SPEED_DUMP"))
        self.on = dir != ""
        self.path = String("")
        if self.on:
            self.path = dir + "/" + lane + ".fixture"
            with open(self.path, "w") as fh:
                fh.write("# mojolearn-speed-fixture v1 lane=" + lane + "\n")

    def param(self, name: String, value: String) raises:
        if not self.on:
            return
        with open(self.path, "a") as fh:
            fh.write("PARAM " + name + " " + value + "\n")

    def param_int(self, name: String, value: Int) raises:
        self.param(name, String(value))

    def param_f32(self, name: String, v: Float32) raises:
        if not self.on:
            return
        with open(self.path, "a") as fh:
            fh.write(
                "PARAMHEX " + name + " " + _hex8(bitcast[DType.uint32](v)) + "\n"
            )

    def f32(self, name: String, xs: List[Float32]) raises:
        if not self.on:
            return
        with open(self.path, "a") as fh:
            fh.write("ARRAY " + name + " f32 " + String(len(xs)) + "\n")
        var chunk = String("")
        var held = 0
        for i in range(len(xs)):
            chunk += _hex8(bitcast[DType.uint32](xs[i])) + "\n"
            held += 1
            if held == DUMP_CHUNK:
                with open(self.path, "a") as fh:
                    fh.write(chunk)
                chunk = String("")
                held = 0
        if held > 0:
            with open(self.path, "a") as fh:
                fh.write(chunk)

    def i32(self, name: String, xs: List[Int32]) raises:
        if not self.on:
            return
        with open(self.path, "a") as fh:
            fh.write("ARRAY " + name + " i32 " + String(len(xs)) + "\n")
        var chunk = String("")
        var held = 0
        for i in range(len(xs)):
            chunk += _hex8(bitcast[DType.uint32](xs[i])) + "\n"
            held += 1
            if held == DUMP_CHUNK:
                with open(self.path, "a") as fh:
                    fh.write(chunk)
                chunk = String("")
                held = 0
        if held > 0:
            with open(self.path, "a") as fh:
                fh.write(chunk)

    def done(self) raises:
        if self.on:
            print("wrote fixture dump " + self.path)


# ============================================================================
# The emitter: the four line kinds, and the in-process hash-stability report.
# ============================================================================


struct Emitter(Movable):
    """One lane's output lines, plus the record of every round's hash.

    The hash report is a REPORT here and not an assertion. Under FAST a
    reduction whose block count depends on occupancy can legitimately return
    different bits on two calls with the same input, and this file exists to
    time the FAST arm. `bench/lanes_price_main.mojo`'s `Ledger.verdict` raises
    on a move under IDENTICAL; that is the right behavior THERE and the wrong
    behavior here, so this one prints and returns.
    """

    var lane: String
    var arm: String
    var shape: String
    var size: String
    var hashes: List[UInt64]

    def __init__(out self, lane: String, shape: String, size: String):
        self.lane = lane
        self.arm = String("ours")
        self.shape = shape
        self.size = size
        self.hashes = List[UInt64]()

    def header(self, device: String, rounds: Int):
        print(
            "FSPEED-HEADER family=classical lane=" + self.lane + " arm="
            + self.arm + " mode=" + _mode_name() + " device=" + device
            + " rounds=" + String(rounds) + " size=" + self.size
        )

    def warmup(self, ns: Int):
        print(
            "FSPEED-WARMUP lane=" + self.lane + " arm=" + self.arm + " shape="
            + self.shape + " ms=" + String(Float64(ns) / 1.0e6)
        )

    def emit(mut self, idx: Int, ns: Int, h: UInt64):
        # `idx` and not `round`: `round` is a builtin and shadowing it inside
        # a struct method is the kind of thing that compiles here and stops
        # compiling on the next toolchain.
        print(
            "FSPEED lane=" + self.lane + " arm=" + self.arm + " shape="
            + self.shape + " round=" + String(idx) + " ms="
            + String(Float64(ns) / 1.0e6) + " hash=" + _hex16(h)
        )
        self.hashes.append(h)

    def note(self):
        """One `FSPEED-NOTE` if any timed round's hash differs from the first.

        Names the two hashes and nothing more: the orchestrator's table wants
        to know THAT the arm is non-deterministic on this box, and the
        identity lanes are where WHICH stage moved gets answered.
        """
        if len(self.hashes) < 2:
            return
        var first = self.hashes[0]
        for i in range(1, len(self.hashes)):
            if self.hashes[i] != first:
                print(
                    "FSPEED-NOTE lane=" + self.lane + " arm=" + self.arm
                    + " hash moved across rounds: " + _hex16(first) + " "
                    + _hex16(self.hashes[i])
                )
                return


def _refuse(lane: String, arm: String, reason: String):
    print("FSPEED-REFUSED lane=" + lane + " arm=" + arm + " reason=" + reason)


# ============================================================================
# kmeans -- bench/bench_main.mojo's shape and its fixture, both transcribed.
#
# TRANSCRIBED from `bench/bench_main.mojo` (a file that carries a `main`): the
# shapes 4,000,000 x 32 at k = 64 for 20 iterations, the `_u01` mixer, the
# `* 10.0` scaling, the `c * 7919` centroid seed and the `KMeansParams`
# settings. `bench/bench_sklearn.py` already holds the Python twin of exactly
# those, which is why this lane does not dump: the opponent regenerates.
# If `bench_main.mojo`'s shape moves, this must move with it.
# ============================================================================


def _u01(row: Int, k: Int, salt: Int) -> Float64:
    """TRANSCRIBED from `bench/bench_main.mojo::_u01`, the splitmix64 mixer
    `bench/bench_sklearn.py::u01` vectorizes. The two sides get the same
    dataset because they run the same recurrence, not because two people
    wrote down the same intention."""
    var z = (
        UInt64(row) * 0x9E3779B97F4A7C15
        + UInt64(k + 1) * 0xBF58476D1CE4E5B9
        + UInt64(salt + 1) * 0x94D049BB133111EB
    )
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9
    z = (z ^ (z >> 27)) * 0x94D049BB133111EB
    z = z ^ (z >> 31)
    return Float64(z >> 11) * (1.0 / 9007199254740992.0)


def run_kmeans(ctx: DeviceContext, smoke: Bool, rounds: Int, size: String) raises:
    """`fit` with `init = INIT_ARRAY`, `n_init = 1`, `max_iter = 20`,
    `tol = 1e-7`, metric L2Expanded, on 4,000,000 x 32 at k = 64.

    THE CENTROIDS ARE RE-UPLOADED EVERY ROUND. `INIT_ARRAY` means the entry
    READS `km_c` as its starting point and writes the answer back over it, so
    without the re-upload round two would be a continuation of round one's fit
    and every round would time a different problem.
    """
    var rows = 40000 if smoke else 4000000
    var cols = 32
    var k = 64
    var iters = 20
    var x = ctx.enqueue_create_buffer[DType.float32](rows * cols)
    var w = ctx.enqueue_create_buffer[DType.float32](rows)
    var c = ctx.enqueue_create_buffer[DType.float32](k * cols)
    var labels = ctx.enqueue_create_buffer[DType.uint32](rows)
    ctx.synchronize()
    var hx = ctx.enqueue_create_host_buffer[DType.float32](rows * cols)
    var hw = ctx.enqueue_create_host_buffer[DType.float32](rows)
    for i in range(rows):
        hw.unsafe_ptr().unsafe_store(i, Float32(1.0))
        for f in range(cols):
            hx.unsafe_ptr().unsafe_store(
                i * cols + f, Float32(_u01(i, f, 0) * 10.0)
            )
    var hc = ctx.enqueue_create_host_buffer[DType.float32](k * cols)
    for cc in range(k):
        for f in range(cols):
            hc.unsafe_ptr().unsafe_store(
                cc * cols + f, Float32(_u01(cc * 7919, f, 5) * 10.0)
            )
    ctx.enqueue_copy(dst_buf=x, src_ptr=hx.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=w, src_ptr=hw.unsafe_ptr())
    ctx.synchronize()
    var sum_scale = Float32(choose_scale(Float64(rows) * 10.0))
    var wt_scale = Float32(choose_scale(Float64(rows)))
    var p = KMeansParams.default()
    p.n_clusters = k
    p.init = INIT_ARRAY
    p.metric = METRIC_L2_EXPANDED
    p.n_init = 1
    p.max_iter = iters
    p.tol = 0.0000001

    var em = Emitter(
        "kmeans",
        String(rows) + "x" + String(cols) + "k" + String(k) + "i" + String(iters),
        size,
    )
    em.header(_no_spaces(ctx.name()), rounds)
    for r in range(rounds + 1):
        ctx.enqueue_copy(dst_buf=c, src_ptr=hc.unsafe_ptr())
        ctx.synchronize()
        var t0 = perf_counter_ns()
        _ = kmeans_fit_api(
            ctx, x, w, c, labels, p, rows, cols, sum_scale, wt_scale
        )
        ctx.synchronize()
        var t1 = perf_counter_ns()
        var h = _hash_device_f32(ctx, FNV_OFFSET, c, k * cols)
        h = _hash_device_u32(ctx, h, labels, rows)
        if r == 0:
            em.warmup(t1 - t0)
        else:
            em.emit(r, t1 - t0, h)
    em.note()
    _ = hx^
    _ = hw^
    _ = hc^
    _ = x^
    _ = w^
    _ = c^
    _ = labels^


# ============================================================================
# dbscan
# ============================================================================


def run_dbscan(ctx: DeviceContext, smoke: Bool, rounds: Int, size: String) raises:
    """`dbscan_fit_impl` at eps 0.35, min_samples 5, on 4,000 x 16.

    TRANSCRIBED from `bench/bench_main.mojo`, including the row count, which
    that file explains: the workspace is sized INSIDE the entry from the
    device's memory, exactly as cuML's `dbscanFitImpl` does it, so 4,000 is
    the shipped shape and not a memory ceiling this harness imposed.
    """
    var rows = 512 if smoke else 4000
    var cols = 16
    var x = ctx.enqueue_create_buffer[DType.float32](rows * cols)
    var labels = ctx.enqueue_create_buffer[DType.int32](rows)
    ctx.synchronize()
    var hx = ctx.enqueue_create_host_buffer[DType.float32](rows * cols)
    for i in range(rows):
        for f in range(cols):
            hx.unsafe_ptr().unsafe_store(
                i * cols + f, Float32(_u01(i, f, 4) * 2.0)
            )
    ctx.enqueue_copy(dst_buf=x, src_ptr=hx.unsafe_ptr())
    ctx.synchronize()
    var em = Emitter("dbscan", String(rows) + "x" + String(cols), size)
    em.header(_no_spaces(ctx.name()), rounds)
    for r in range(rounds + 1):
        ctx.enqueue_memset(labels, Int32(-7))
        ctx.synchronize()
        var t0 = perf_counter_ns()
        _ = dbscan_fit_impl(ctx, x, labels, rows, cols, 0.35, 5)
        ctx.synchronize()
        var t1 = perf_counter_ns()
        var h = _hash_device_i32(ctx, FNV_OFFSET, labels, rows)
        if r == 0:
            em.warmup(t1 - t0)
        else:
            em.emit(r, t1 - t0, h)
    em.note()
    _ = hx^
    _ = x^
    _ = labels^


# ============================================================================
# pca
# ============================================================================


def run_pca(ctx: DeviceContext, smoke: Bool, rounds: Int, size: String) raises:
    """`pca_fit` at 8 components on 4,000,000 x 32, the covariance route.

    The opponent asks cuML for the covariance eigendecomposition BY NAME
    (`svd_solver="covariance_eigh"`) and falls back to `full` only if the
    installed build has no such solver, in which case it prints a note saying
    the ratio is now algorithm plus device rather than device alone. `auto` is
    used on neither side: a comparison that depends on somebody's heuristic
    staying put stops being one without telling you.

    The hash is the components, which `PCAResult` returns as `Float64`.
    """
    var rows = 40000 if smoke else 4000000
    var cols = 32
    var comp = 8
    var x = ctx.enqueue_create_buffer[DType.float32](rows * cols)
    var xa = ctx.enqueue_create_buffer[DType.float32](rows * cols)
    var xa2 = ctx.enqueue_create_buffer[DType.float32](rows * cols)
    var mu = ctx.enqueue_create_buffer[DType.float32](cols)
    var cov = ctx.enqueue_create_buffer[DType.float32](cols * cols)
    ctx.synchronize()
    var hx = ctx.enqueue_create_host_buffer[DType.float32](rows * cols)
    for i in range(rows):
        for f in range(cols):
            hx.unsafe_ptr().unsafe_store(
                i * cols + f, Float32(_u01(i, f, 3) * 4.0)
            )
    ctx.enqueue_copy(dst_buf=x, src_ptr=hx.unsafe_ptr())
    ctx.synchronize()
    var em = Emitter(
        "pca", String(rows) + "x" + String(cols) + "c" + String(comp), size
    )
    em.header(_no_spaces(ctx.name()), rounds)
    for r in range(rounds + 1):
        var t0 = perf_counter_ns()
        var res = pca_fit(ctx, x, xa, xa2, mu, cov, rows, cols, comp)
        ctx.synchronize()
        var t1 = perf_counter_ns()
        var h = _fold_f64_list(FNV_OFFSET, res.components)
        h = _fold_f64_list(h, res.explained_var)
        h = _fold_f64_list(h, res.singular_vals)
        if r == 0:
            em.warmup(t1 - t0)
        else:
            em.emit(r, t1 - t0, h)
    em.note()
    _ = hx^
    _ = x^
    _ = xa^
    _ = xa2^
    _ = mu^
    _ = cov^


# ============================================================================
# ols
# ============================================================================


def run_ols(ctx: DeviceContext, smoke: Bool, rounds: Int, size: String) raises:
    """`lstsq_eig` on 4,000,000 x 32: the normal equations, eigen route.

    The opponent is `cuml.LinearRegression(algorithm="eig", fit_intercept=
    False)`, which is the SAME algorithm class. cuML's default `algorithm` is
    `"eig"` for a tall matrix, so this is also what a user gets; it is passed
    explicitly on both sides anyway, because a comparison that depends on a
    heuristic staying put is a comparison that silently stops being one.
    """
    var rows = 40000 if smoke else 4000000
    var cols = 32
    var a = ctx.enqueue_create_buffer[DType.float32](rows * cols)
    var b = ctx.enqueue_create_buffer[DType.float32](rows)
    var w = ctx.enqueue_create_buffer[DType.float32](cols)
    var cov = ctx.enqueue_create_buffer[DType.float32](cols * cols)
    var q = ctx.enqueue_create_buffer[DType.float32](cols * cols)
    var qs = ctx.enqueue_create_buffer[DType.float32](cols * cols)
    var s = ctx.enqueue_create_buffer[DType.float32](cols)
    var ab = ctx.enqueue_create_buffer[DType.float32](cols)
    var inv = ctx.enqueue_create_buffer[DType.float32](cols * cols)
    var aa = ctx.enqueue_create_buffer[DType.float32](rows * cols)
    var aa2 = ctx.enqueue_create_buffer[DType.float32](rows * cols)
    ctx.synchronize()
    var ha = ctx.enqueue_create_host_buffer[DType.float32](rows * cols)
    var hb = ctx.enqueue_create_host_buffer[DType.float32](rows)
    for i in range(rows):
        var t = 0.0
        for f in range(cols):
            var v = _u01(i, f, 6) - 0.5
            ha.unsafe_ptr().unsafe_store(i * cols + f, Float32(v))
            t += v * (1.0 + 0.1 * Float64(f))
        hb.unsafe_ptr().unsafe_store(i, Float32(t))
    ctx.enqueue_copy(dst_buf=a, src_ptr=ha.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=b, src_ptr=hb.unsafe_ptr())
    ctx.synchronize()
    var em = Emitter("ols", String(rows) + "x" + String(cols), size)
    em.header(_no_spaces(ctx.name()), rounds)
    for r in range(rounds + 1):
        # `lstsq_eig` centers and un-centers `a` and `b` in place on some
        # arms; re-uploading both makes every round the SAME problem. The cd
        # lane's harness learned this the expensive way (see
        # `bench/lanes_price_main.mojo::run_cd`) and it costs nothing here.
        ctx.enqueue_copy(dst_buf=a, src_ptr=ha.unsafe_ptr())
        ctx.enqueue_copy(dst_buf=b, src_ptr=hb.unsafe_ptr())
        ctx.enqueue_memset(w, Float32(0.0))
        ctx.synchronize()
        var t0 = perf_counter_ns()
        lstsq_eig(ctx, a, b, w, cov, q, qs, s, ab, inv, aa, aa2, rows, cols)
        ctx.synchronize()
        var t1 = perf_counter_ns()
        var h = _hash_device_f32(ctx, FNV_OFFSET, w, cols)
        if r == 0:
            em.warmup(t1 - t0)
        else:
            em.emit(r, t1 - t0, h)
    em.note()
    _ = ha^
    _ = hb^
    _ = a^
    _ = b^
    _ = w^
    _ = cov^
    _ = q^
    _ = qs^
    _ = s^
    _ = ab^
    _ = inv^
    _ = aa^
    _ = aa2^


# ============================================================================
# knn
# ============================================================================


def run_knn(ctx: DeviceContext, smoke: Bool, rounds: Int, size: String) raises:
    """Brute-force k-NN, k = 10, 4,000 queries against a 400,000 x 32 index.

    TRANSCRIBED from `bench/bench_main.mojo`, INCLUDING the two `compute_norms`
    calls, which are inside the timed region there and are inside it here:
    they are part of what a search costs and cuML's `NearestNeighbors` pays
    the same price inside its own `kneighbors`.

    THEIR DISPATCH decides which kernel runs (`knn_brute_force.cuh:443`): at
    k = 10 <= 64 on row-major L2 it is `fusedL2Knn`, which only exists on a
    32-lane column. On a 64-wide wavefront the entry refuses and takes the
    tiled arm; that is a real column difference and not a harness choice.
    """
    var index = 40000 if smoke else 400000
    var queries = 400 if smoke else 4000
    var cols = 32
    var k = 10
    var tile = 256
    var idx = ctx.enqueue_create_buffer[DType.float32](index * cols)
    var qr = ctx.enqueue_create_buffer[DType.float32](queries * cols)
    var inorm = ctx.enqueue_create_buffer[DType.float32](index)
    var qnorm = ctx.enqueue_create_buffer[DType.float32](queries)
    var dist = ctx.enqueue_create_buffer[DType.float32](tile * index)
    var bl = index // 8
    var bv = ctx.enqueue_create_buffer[DType.float32](tile * 2 * bl)
    var bi = ctx.enqueue_create_buffer[DType.uint32](tile * 2 * bl)
    var od = ctx.enqueue_create_buffer[DType.float32](queries * k)
    var oi = ctx.enqueue_create_buffer[DType.uint32](queries * k)
    var oi32 = ctx.enqueue_create_buffer[DType.int32](queries * k)
    ctx.synchronize()
    var hi = ctx.enqueue_create_host_buffer[DType.float32](index * cols)
    for i in range(index):
        for f in range(cols):
            hi.unsafe_ptr().unsafe_store(i * cols + f, Float32(_u01(i, f, 1)))
    ctx.enqueue_copy(dst_buf=idx, src_ptr=hi.unsafe_ptr())
    var hq = ctx.enqueue_create_host_buffer[DType.float32](queries * cols)
    for i in range(queries):
        for f in range(cols):
            hq.unsafe_ptr().unsafe_store(i * cols + f, Float32(_u01(i, f, 2)))
    ctx.enqueue_copy(dst_buf=qr, src_ptr=hq.unsafe_ptr())
    ctx.synchronize()
    var em = Emitter(
        "knn",
        String(index) + "x" + String(cols) + "q" + String(queries) + "k"
        + String(k),
        size,
    )
    em.header(_no_spaces(ctx.name()), rounds)
    for r in range(rounds + 1):
        var t0 = perf_counter_ns()
        compute_norms(ctx, idx, inorm, index, cols, False)
        compute_norms(ctx, qr, qnorm, queries, cols, False)
        brute_force_knn_impl(
            ctx, qr, qnorm, idx, inorm, dist, bv, bi, od, oi, oi32,
            queries, index, cols, k, tile, bl, False,
        )
        ctx.synchronize()
        var t1 = perf_counter_ns()
        var h = _hash_device_f32(ctx, FNV_OFFSET, od, queries * k)
        h = _hash_device_u32(ctx, h, oi, queries * k)
        if r == 0:
            em.warmup(t1 - t0)
        else:
            em.emit(r, t1 - t0, h)
    em.note()
    _ = hi^
    _ = hq^
    _ = idx^
    _ = qr^
    _ = inorm^
    _ = qnorm^
    _ = dist^
    _ = bv^
    _ = bi^
    _ = od^
    _ = oi^
    _ = oi32^


# ============================================================================
# cd -- Lasso by coordinate descent
# ============================================================================


def run_cd(ctx: DeviceContext, smoke: Bool, rounds: Int, size: String) raises:
    """`cd_fit_traced`: Lasso, alpha 0.01, l1_ratio 1, fit_intercept, 1000
    epochs, tol 1e-3, no shuffle, on `fixture_planted_sparse(n, d, 610)`.

    Shipped 2,048 x 16, which is `solver/cd_main.mojo`'s default; smoke
    256 x 4. Same lane, same parameters and the same re-upload discipline as
    `bench/lanes_price_main.mojo::run_cd`, whose comment explains why `x` and
    `y` are re-uploaded every round: `cd_fit` MUTATES both in place under
    `fit_intercept` and `postProcessData` does not restore the bits exactly,
    so without the re-upload the warm-up hash and the round hashes differ for
    a reason that has nothing to do with the kernel.
    """
    var n = 256 if smoke else 2048
    var d = 4 if smoke else 16
    var alpha = Float32(0.01)
    var l1_ratio = Float32(1.0)
    var epochs = 1000
    var tol = Float32(1.0e-3)
    var fx = fixture_planted_sparse(n, d, 610)
    var hx = fx[0].copy()
    var hy = fx[1].copy()

    var dump = Dump("cd")
    dump.param_int("n", n)
    dump.param_int("d", d)
    dump.param_f32("alpha", alpha)
    dump.param_f32("l1_ratio", l1_ratio)
    dump.param_int("max_iter", epochs)
    dump.param_f32("tol", tol)
    dump.param_int("fit_intercept", 1)
    dump.f32("x", hx)
    dump.f32("y", hy)
    dump.done()

    var x = ctx.enqueue_create_buffer[DType.float32](n * d)
    var y = ctx.enqueue_create_buffer[DType.float32](n)
    var coef = ctx.enqueue_create_buffer[DType.float32](d)
    var resid = ctx.enqueue_create_buffer[DType.float32](n)
    ctx.synchronize()
    var em = Emitter("cd", String(n) + "x" + String(d), size)
    em.header(_no_spaces(ctx.name()), rounds)
    for r in range(rounds + 1):
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
        if r == 0:
            em.warmup(t1 - t0)
        else:
            em.emit(r, t1 - t0, h)
    em.note()
    _ = hx^
    _ = hy^
    _ = x^
    _ = y^
    _ = coef^
    _ = resid^


# ============================================================================
# kde
# ============================================================================


def run_kde(ctx: DeviceContext, smoke: Bool, rounds: Int, size: String) raises:
    """`score_samples`: gaussian kernel, euclidean metric, weighted,
    bandwidth 2.75, on `kde/kde_main.mojo`'s fixtures.

    Shipped 1,024 train x 256 query x 8 features; smoke 128 x 32 x 8. The
    validation and the upload happen once, before the loop, exactly as
    `kde_score_samples_host` does them, so the timed region is the score pass.
    """
    var n_train = 128 if smoke else 1024
    var n_query = 32 if smoke else 256
    var d = 8
    var bandwidth = Float32(2.75)
    var train = train_fixture(n_train, d, 1)
    var query = query_fixture(train, n_train, n_query, d, 1)
    var weights = weight_fixture(n_train, 1)
    var kern = kernel_from_name("gaussian")
    var metric = metric_from_name("euclidean")
    kde_fit_validate(n_train, d, bandwidth, kern, metric, weights, True)
    kde_validate_data(train, n_train, d, metric, "train")
    kde_validate_data(query, n_query, d, metric, "query")
    var sum_w = host_sum_weights(weights)

    var dump = Dump("kde")
    dump.param_int("n_train", n_train)
    dump.param_int("n_query", n_query)
    dump.param_int("d", d)
    dump.param_f32("bandwidth", bandwidth)
    dump.param("kernel", "gaussian")
    dump.param("metric", "euclidean")
    dump.f32("train", train)
    dump.f32("query", query)
    dump.f32("weights", weights)
    dump.done()

    var dtrain = ctx.enqueue_create_buffer[DType.float32](n_train * d)
    var dquery = ctx.enqueue_create_buffer[DType.float32](n_query * d)
    var dweights = ctx.enqueue_create_buffer[DType.float32](n_train)
    var dout = ctx.enqueue_create_buffer[DType.float32](n_query)
    ctx.enqueue_copy(dst_buf=dtrain, src_ptr=train.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=dquery, src_ptr=query.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=dweights, src_ptr=weights.unsafe_ptr())
    ctx.synchronize()
    var em = Emitter(
        "kde",
        String(n_train) + "x" + String(n_query) + "x" + String(d),
        size,
    )
    em.header(_no_spaces(ctx.name()), rounds)
    for r in range(rounds + 1):
        ctx.enqueue_memset(dout, Float32(0.0))
        ctx.synchronize()
        var trace = IdentityTrace.disabled()
        var t0 = perf_counter_ns()
        score_samples(
            ctx, dquery, dtrain, dweights, True, dout, n_query, n_train, d,
            bandwidth, sum_w, kern, metric, Float32(2.0), trace,
        )
        ctx.synchronize()
        var t1 = perf_counter_ns()
        var h = _hash_device_f32(ctx, FNV_OFFSET, dout, n_query)
        if r == 0:
            em.warmup(t1 - t0)
        else:
            em.emit(r, t1 - t0, h)
    em.note()
    _ = train^
    _ = query^
    _ = weights^
    _ = dtrain^
    _ = dquery^
    _ = dweights^
    _ = dout^


# ============================================================================
# linkage -- single-linkage agglomerative clustering
# ============================================================================


def run_linkage(ctx: DeviceContext, smoke: Bool, rounds: Int, size: String) raises:
    """`single_linkage` with pairwise connectivity and L2SqrtExpanded on
    `FIX_BLOBS_DUPS` (102 x 5, 3 clusters), the card fixture; smoke is
    `FIX_DUPS` (48 x 2, 4 clusters).

    102 ROWS IS A FIXED-COST MEASUREMENT. The fixture is the one the lane
    ships and its builder takes no size, so this is what the shipped size IS.
    Read the number as what one small agglomerative fit costs end to end, not
    as a throughput.
    """
    var fix = FIX_DUPS if smoke else FIX_BLOBS_DUPS
    var m = fixture_n(fix)
    var d = fixture_d(fix)
    var n_clusters = fixture_n_clusters(fix)
    var hx = build_fixture(ctx, fix)

    var dump = Dump("linkage")
    dump.param_int("n", m)
    dump.param_int("d", d)
    dump.param_int("n_clusters", n_clusters)
    dump.param("fixture", fixture_name(fix))
    # `build_fixture` returns a `HostBuffer` and the dump takes a `List`;
    # `fixture_as_list` is the SAME `fixture_value` loop in the same file, so
    # this is one fixture in two containers and not two fixtures.
    dump.f32("x", fixture_as_list(fix))
    dump.done()

    var x = ctx.enqueue_create_buffer[DType.float32](m * d)
    ctx.enqueue_copy(dst_buf=x, src_ptr=hx.unsafe_ptr())
    ctx.synchronize()
    var children = ctx.enqueue_create_buffer[DType.int32]((m - 1) * 2)
    var labels = ctx.enqueue_create_buffer[DType.int32](m)
    var em = Emitter(
        "linkage", fixture_name(fix) + "." + String(m) + "x" + String(d), size
    )
    em.header(_no_spaces(ctx.name()), rounds)
    for r in range(rounds + 1):
        ctx.enqueue_memset(children, Int32(-7))
        ctx.enqueue_memset(labels, Int32(-7))
        ctx.synchronize()
        var t0 = perf_counter_ns()
        var res = single_linkage(
            ctx, x, m, d, n_clusters, DISTANCE_L2_SQRT_EXPANDED, children, labels
        )
        ctx.synchronize()
        var t1 = perf_counter_ns()
        var h = _hash_device_i32(ctx, FNV_OFFSET, children, (m - 1) * 2)
        h = _hash_device_i32(ctx, h, labels, m)
        h = _fold_word(h, UInt64(res.n_boruvka_rounds), 4)
        if r == 0:
            em.warmup(t1 - t0)
        else:
            em.emit(r, t1 - t0, h)
    em.note()
    _ = hx^
    _ = x^
    _ = children^
    _ = labels^


# ============================================================================
# svm -- C-SVC, FIT ONLY
# ============================================================================


def run_svm(ctx: DeviceContext, smoke: Bool, rounds: Int, size: String) raises:
    """`svc_fit` on `F2.xor` (240 x 2, RBF gamma 0.5, C 10, tol 1e-3).

    FIT ONLY, AND NOT `_run_device`. `bench/lanes_price_main.mojo`'s svm lane
    times a fit plus two predicts over `n + 37` queries because that is what
    its identity card covers. The opponent here is `cuml.svm.SVC(...).fit`,
    so the timed region is the fit and only the fit. The two numbers are
    about different work and must not appear in one table.

    THE TIMED REGION INCLUDES THE UPLOAD. `svc_fit` takes HOST lists and
    uploads them itself; there is no device-buffer entry to call instead. At
    240 x 2 the upload is negligible, and cuML's `fit` takes a host or device
    array and pays the same kind of cost, so the arms are shaped alike.
    """
    # `Fixture` is `Movable` and NOT `Copyable`, so `var fx = fixtures[1]`
    # does not compile. Every use goes through the subscript, exactly as
    # `bench/lanes_price_main.mojo::run_svm` does it and for the same reason.
    var fixtures = all_fixtures()
    var which = 1

    var dump = Dump("svm")
    dump.param("fixture", fixtures[which].name)
    dump.param_int("n", fixtures[which].n)
    dump.param_int("d", fixtures[which].k)
    dump.param("C", String(fixtures[which].param.C))
    dump.param("gamma", String(fixtures[which].kp.gamma))
    dump.param("tol", String(fixtures[which].param.tol))
    dump.param("kernel", "rbf")
    dump.param_int("nochange_steps", fixtures[which].param.nochange_steps)
    dump.f32("x", fixtures[which].x)
    dump.f32("y", fixtures[which].labels)
    dump.done()

    var em = Emitter(
        "svm",
        fixtures[which].name + "." + String(fixtures[which].n) + "x"
        + String(fixtures[which].k),
        size,
    )
    em.header(_no_spaces(ctx.name()), rounds)
    for r in range(rounds + 1):
        var card = IdentityTrace.disabled()
        var smo = SmoTrace()
        var t0 = perf_counter_ns()
        var model = svc_fit(
            ctx, fixtures[which].x, fixtures[which].labels,
            fixtures[which].n, fixtures[which].k, fixtures[which].param,
            fixtures[which].kp, card, smo,
            False, 1 << 30, 0, False, 0, Float32(0.0),
        )
        ctx.synchronize()
        var t1 = perf_counter_ns()
        var h = _fold_f32_list(FNV_OFFSET, model.dual_coefs)
        h = _fold_i32_list(h, model.support_idx)
        h = _fold_f32(h, model.b)
        h = _fold_word(h, UInt64(model.n_support), 4)
        if r == 0:
            em.warmup(t1 - t0)
        else:
            em.emit(r, t1 - t0, h)
        _ = model^
        _ = smo^
    em.note()
    _ = fixtures^


# ============================================================================
# metrics -- eleven metrics, one pass
#
# TRANSCRIBED from `metrics/metrics_main.mojo` (comptime constants in a file
# that carries a `main`; the same choice `bench/lanes_price_main.mojo` made
# and for the same reason). If that file's sizes move, these must.
#
# ELEVEN AND NOT TWELVE. `bench/lanes_price_main.mojo` also times
# `rand_index`. cuML ships `adjusted_rand_score` and does NOT ship a plain
# Rand index, so timing it here would make our pass do work the opponent's
# pass cannot do and the ratio would be about that difference. It is dropped
# from the TIMED PASS, not from the lane: `metrics/metrics_main.mojo` still
# gates it. Do not compare this lane's number with the lanes_price one.
# ============================================================================

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


def run_metrics(ctx: DeviceContext, smoke: Bool, rounds: Int, size: String) raises:
    """Every metric cuML also ships, on the fixtures `metrics/metrics_main.
    mojo` builds, timed as ONE pass because the lane ships them as one card.

    Same builders and the same salts (4099 / 17,18 / 19,20 / 23 / 29,38).
    Shipped sizes are the main's; smoke shrinks the row counts.
    """
    var n_lab = 257 if smoke else MET_N_LABELS_ROWS
    var n_flt = 257 if smoke else MET_N_FLOAT
    var n_sil = 67 if smoke else MET_N_SIL
    var n_tru = 61 if smoke else MET_N_TRUST
    var lp = labels_true_pred(n_lab, MET_N_TRUE, MET_N_PRED, 0.66, 4099)
    var yt_h = lp[0].copy()
    var yp_h = lp[1].copy()
    var lo = Int32(0)
    var hi = Int32(MET_N_TRUE - 1)
    var y_h = hashed_floats(n_flt, 17, -6, 6)
    var res_h = hashed_floats(n_flt, 18, -6, 5)
    var yhat_h = List[Float32]()
    for i in range(n_flt):
        yhat_h.append(y_h[i] + res_h[i])
    var p_h = hashed_pdf(n_flt, 19, 101)
    var q_h = hashed_pdf(n_flt, 20, 0)
    var pts = hashed_points(n_sil, MET_D_SIL, MET_K_SIL, 23)
    var x_h = pts[0].copy()
    var lab_h = pts[1].copy()
    var tp = hashed_points(n_tru, MET_M_TRUST, 4, 29)
    var tx = tp[0].copy()
    var temb = List[Float32]()
    for i in range(n_tru):
        for qq in range(MET_D_TRUST):
            temb.append(
                tx[i * MET_M_TRUST + qq] + Float32((u01(i, qq, 38) - 0.5) * 0.8)
            )

    var dump = Dump("metrics")
    dump.param_int("n_labels", n_lab)
    dump.param_int("n_float", n_flt)
    dump.param_int("n_sil", n_sil)
    dump.param_int("d_sil", MET_D_SIL)
    dump.param_int("k_sil", MET_K_SIL)
    dump.param_int("n_trust", n_tru)
    dump.param_int("m_trust", MET_M_TRUST)
    dump.param_int("d_trust", MET_D_TRUST)
    dump.param_int("k_trust", MET_K_TRUST)
    dump.i32("y_true", yt_h)
    dump.i32("y_pred", yp_h)
    dump.f32("y", y_h)
    dump.f32("y_hat", yhat_h)
    dump.f32("p", p_h)
    dump.f32("q", q_h)
    dump.f32("sil_x", x_h)
    dump.i32("sil_labels", lab_h)
    dump.f32("trust_x", tx)
    dump.f32("trust_emb", temb)
    dump.done()

    var yt = met_upload_i32(ctx, yt_h)
    var yp = met_upload_i32(ctx, yp_h)
    var dy = met_upload_f32(ctx, y_h)
    var dyh = met_upload_f32(ctx, yhat_h)
    var dp = met_upload_f32(ctx, p_h)
    var dq = met_upload_f32(ctx, q_h)
    var dx = met_upload_f32(ctx, x_h)
    var dl = met_upload_i32(ctx, lab_h)
    var ds = ctx.enqueue_create_buffer[DType.float32](n_sil)
    ctx.synchronize()
    var em = Emitter(
        "metrics",
        "lab" + String(n_lab) + ".flt" + String(n_flt) + ".sil" + String(n_sil)
        + "x" + String(MET_D_SIL) + ".tru" + String(n_tru) + "x"
        + String(MET_M_TRUST),
        size,
    )
    em.header(_no_spaces(ctx.name()), rounds)
    for r in range(rounds + 1):
        ctx.enqueue_memset(ds, Float32(0.0))
        ctx.synchronize()
        var trace = IdentityTrace.disabled()
        var t0 = perf_counter_ns()
        var acc = accuracy_score_py(ctx, yt, yp, n_lab)
        var ari = adjusted_rand_index(ctx, yt, yp, n_lab)
        var h_true = entropy(ctx, yt, n_lab, lo, hi)
        var mi = mutual_info_score(ctx, yt, yp, n_lab, lo, hi)
        var hom = homogeneity_score(ctx, yt, yp, n_lab, lo, hi)
        var com = completeness_score(ctx, yt, yp, n_lab, lo, hi)
        var vm = v_measure(ctx, yt, yp, n_lab, lo, hi)
        var r2 = r2_score_py(ctx, dy, dyh, n_flt)
        var kl = kl_divergence(ctx, dp, dq, n_flt)
        var sil = silhouette_score(ctx, dx, n_sil, MET_D_SIL, dl, MET_K_SIL, ds)
        var tw = trustworthiness_score_traced(
            ctx, trace, tx, temb, n_tru, MET_M_TRUST, MET_D_TRUST, MET_K_TRUST
        )
        ctx.synchronize()
        var t1 = perf_counter_ns()
        var h = _fold_f32(FNV_OFFSET, acc)
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
        h = _fold_f64(h, tw)
        if r == 0:
            em.warmup(t1 - t0)
        else:
            em.emit(r, t1 - t0, h)
    em.note()
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
# ivf -- IVF-Flat build plus search
# ============================================================================


def run_ivf(ctx: DeviceContext, smoke: Bool, rounds: Int, size: String) raises:
    """`ivf_flat_build_and_search_host`: ONE build and ONE search, together.

    `ivf/ivf_main.mojo`'s shape: 512 index rows x 8 features, 64 queries,
    8 lists, 3 probes, k = 8, 20 coarse k-means iterations, L2Expanded
    (SQUARED distances), seed 0. Smoke halves the index and the query set.

    A BUILD AND A SEARCH TOGETHER IS THE UNIT ON PURPOSE. The lane's own
    policy 3 says a build plus a search is one card because the trace's
    sequence may not restart, and it is also the unit the cuVS opponent
    measures: `ivf_flat.build` then `ivf_flat.search`, both inside the clock.

    THE TIMED REGION INCLUDES THE UPLOAD AND THE DOWNLOAD. This entry takes
    host lists and returns host lists; there is no device-buffer surface. The
    cuVS arm is given host numpy for the same reason, so both sides pay it.
    """
    var n_rows = 256 if smoke else 512
    var n_queries = 32 if smoke else 64
    var dim = 8
    var n_lists = 8
    var n_probes = 3
    var k = 8
    var kmeans_iters = 20
    var index = ivf_index_fixture(n_rows, dim, 1)
    var queries = ivf_query_fixture(index, n_rows, n_queries, dim, 1)

    var dump = Dump("ivf")
    dump.param_int("n_rows", n_rows)
    dump.param_int("n_queries", n_queries)
    dump.param_int("dim", dim)
    dump.param_int("n_lists", n_lists)
    dump.param_int("n_probes", n_probes)
    dump.param_int("k", k)
    dump.param_int("kmeans_n_iters", kmeans_iters)
    dump.param("metric", "sqeuclidean")
    dump.param_int("seed", 0)
    dump.f32("index", index)
    dump.f32("queries", queries)
    dump.done()

    var em = Emitter(
        "ivf",
        String(n_rows) + "x" + String(dim) + "q" + String(n_queries) + "L"
        + String(n_lists) + "p" + String(n_probes) + "k" + String(k),
        size,
    )
    em.header(_no_spaces(ctx.name()), rounds)
    for r in range(rounds + 1):
        var t0 = perf_counter_ns()
        var res = ivf_flat_build_and_search_host(
            ctx, index, n_rows, dim, n_lists, queries, n_queries, k, n_probes,
            kmeans_iters,
        )
        ctx.synchronize()
        var t1 = perf_counter_ns()
        var h = _fold_f32_list(FNV_OFFSET, res.distances)
        for i in range(len(res.indices)):
            h = _fold_u32(h, res.indices[i])
        if r == 0:
            em.warmup(t1 - t0)
        else:
            em.emit(r, t1 - t0, h)
        _ = res^
    em.note()
    _ = index^
    _ = queries^


# ============================================================================
# hdbscan
# ============================================================================


def run_hdbscan(ctx: DeviceContext, smoke: Bool, rounds: Int, size: String) raises:
    """`fit_hdbscan` on `blobs96` (96 x 4), min_samples 5, min_cluster_size 5,
    L2SqrtExpanded, dense mutual reachability, Excess of Mass.

    96 ROWS IS THE SHIPPED SIZE AND IT IS A FIXED-COST MEASUREMENT. The dense
    mutual-reachability arm (DEVIATION 1600) materializes an `m x m` matrix,
    the fixture builder takes no size, and inventing a bigger one to make the
    number look like throughput would be inventing a dataset. Read this as
    what one small HDBSCAN fit costs, and read `cuml.HDBSCAN`'s number the
    same way: at 96 rows both sides are mostly launch latency.
    """
    var fix = HFIX_BLOBS
    var m = hfixture_n(fix)
    var d = hfixture_d(fix)
    var x_host = hfixture_as_list(fix)
    var min_samples = hfixture_min_samples(fix)
    var min_cluster_size = hfixture_min_cluster_size(fix)

    var dump = Dump("hdbscan")
    dump.param("fixture", hfixture_name(fix))
    dump.param_int("n", m)
    dump.param_int("d", d)
    dump.param_int("min_samples", min_samples)
    dump.param_int("min_cluster_size", min_cluster_size)
    dump.param("metric", "euclidean")
    dump.param("cluster_selection_method", "eom")
    dump.f32("x", x_host)
    dump.done()

    var x = ctx.enqueue_create_buffer[DType.float32](m * d)
    ctx.synchronize()
    ctx.enqueue_copy(dst_buf=x, src_ptr=x_host.unsafe_ptr())
    ctx.synchronize()
    var params = HDBSCANParams(
        min_samples, min_cluster_size, 0, Float32(0.0), False, Float32(1.0),
        CLUSTER_SELECTION_EOM, GRAPH_BUILD_BRUTE_FORCE_KNN,
    )
    var em = Emitter(
        "hdbscan",
        hfixture_name(fix) + "." + String(m) + "x" + String(d),
        size,
    )
    em.header(_no_spaces(ctx.name()), rounds)
    for r in range(rounds + 1):
        var trace = IdentityTrace.disabled()
        var t0 = perf_counter_ns()
        var res = fit_hdbscan(
            ctx, trace, x_host, x, m, d, DISTANCE_L2_SQRT_EXPANDED, params
        )
        ctx.synchronize()
        var t1 = perf_counter_ns()
        var h = _fold_i32_list(FNV_OFFSET, res.labels)
        h = _fold_word(h, UInt64(res.n_clusters), 4)
        h = _fold_word(h, UInt64(res.n_outliers), 4)
        if r == 0:
            em.warmup(t1 - t0)
        else:
            em.emit(r, t1 - t0, h)
        _ = res^
    em.note()
    _ = x_host^
    _ = x^


# ============================================================================
# cholesky
# ============================================================================


def run_cholesky(ctx: DeviceContext, smoke: Bool, rounds: Int, size: String) raises:
    """`add_jitter` then `potrf_lower` then `chol_logdet` then `cho_solve`:
    the whole factor-and-solve, on `cholesky/cholesky_main.mojo`'s default
    `RBF` fixture (64 x 64 SPD) with the profile's pinned ridge and 4
    right-hand sides.

    64 x 64 IS THE SHIPPED SIZE AND IT IS A FIXED-COST MEASUREMENT. The
    fixture builder takes a fixture id, not an `n`. Against
    `torch.linalg.cholesky` on cuSOLVER this races two launch paths, which is
    a real thing a user experiences and is not a FLOPs comparison.

    THE JITTER IS APPLIED INSIDE THE TIMED REGION and so is the solve, because
    the opponent's `torch.linalg.cholesky(A + jitter*I)` plus
    `torch.cholesky_solve` is the same four steps. `da` is re-uploaded every
    round: `potrf_lower` factors IN PLACE, so round two would otherwise
    factor a factor.
    """
    var which = FIX_RBF
    var n = chol_fixture_n(which)
    var nrhs = 4
    var jitter = chol_jitter_pinned()
    var a_host = chol_fixture(which, 0)
    var x_planted = chol_rhs_fixture(n, nrhs, 0)
    # B = (A + jitter I) X on the host, exactly as `cholesky_main.mojo` forms
    # it, so the solve's answer is the planted X back again.
    var aj = a_host.copy()
    for i in range(n):
        aj[i * n + i] = aj[i * n + i] + jitter
    var b_host = List[Float32]()
    for i in range(n):
        for j in range(nrhs):
            var acc = Float32(0.0)
            for kk in range(n):
                acc = acc + aj[i * n + kk] * x_planted[kk * nrhs + j]
            b_host.append(acc)

    var dump = Dump("cholesky")
    dump.param("fixture", chol_fixture_name(which))
    dump.param_int("n", n)
    dump.param_int("nrhs", nrhs)
    dump.param_f32("jitter", jitter)
    dump.f32("a", a_host)
    dump.f32("b", b_host)
    dump.done()

    var da = ctx.enqueue_create_buffer[DType.float32](n * n)
    var db = ctx.enqueue_create_buffer[DType.float32](n * nrhs)
    var ws = ctx.enqueue_create_buffer[DType.float32](
        chol_workspace_floats(n, CHOL_NB_PINNED)
    )
    var dwork = ctx.enqueue_create_buffer[DType.float32](n + 1)
    ctx.synchronize()
    var em = Emitter(
        "cholesky",
        chol_fixture_name(which) + "." + String(n) + "x" + String(n) + "r"
        + String(nrhs),
        size,
    )
    em.header(_no_spaces(ctx.name()), rounds)
    for r in range(rounds + 1):
        ctx.enqueue_copy(dst_buf=da, src_ptr=a_host.unsafe_ptr())
        ctx.enqueue_copy(dst_buf=db, src_ptr=b_host.unsafe_ptr())
        ctx.synchronize()
        var trace = IdentityTrace.disabled()
        var t0 = perf_counter_ns()
        add_jitter(ctx, da, n, jitter, CHOL_ELEM_TPB)
        var run = potrf_lower(
            ctx, da, ws, n, trace, CHOL_NB_PINNED, CHOL_PANEL_TPB, CHOL_ELEM_TPB
        )
        var logdet = chol_logdet(ctx, da, dwork, n, trace, CHOL_ELEM_TPB)
        cho_solve(ctx, da, db, n, nrhs, trace, CHOL_SOLVE_TPB)
        ctx.synchronize()
        var t1 = perf_counter_ns()
        if run.info != 0:
            _refuse(
                "cholesky", "ours",
                "the fixture did not factor: LAPACK info=" + String(run.info),
            )
            return
        var h = _hash_device_f32(ctx, FNV_OFFSET, da, n * n)
        h = _hash_device_f32(ctx, h, db, n * nrhs)
        h = _fold_f32(h, logdet)
        if r == 0:
            em.warmup(t1 - t0)
        else:
            em.emit(r, t1 - t0, h)
    em.note()
    _ = a_host^
    _ = b_host^
    _ = aj^
    _ = x_planted^
    _ = da^
    _ = db^
    _ = ws^
    _ = dwork^


# ============================================================================
# gmm -- Gaussian mixture, EM
# ============================================================================


def run_gmm(ctx: DeviceContext, smoke: Bool, rounds: Int, size: String) raises:
    """`gaussian_mixture_fit` on `FIX_SEPARATED` (24 x 2, K = 3), covariance
    type `full`, `init_params = kmeans`, `max_iter = 50`, scikit-learn's
    `tol = 1e-3` and `reg_covar = 1e-6`, `n_init = 1`, seed 0.

    TWO THINGS ARE INSIDE THE TIMED REGION THAT A READER WILL NOT EXPECT.
    First, `gaussian_mixture_fit` takes a host list and constructs its OWN
    `DeviceContext` (`mixture/estimator.mojo:704`), so every round pays a
    context construction. Second, at 24 rows the whole fit is a few dozen
    launches over almost no data, so the number is dominated by both. That is
    what the entry costs a caller today; it is a finding about the lane's
    surface and not a defect in this harness.

    `ctx` is still passed in and still synchronized so the shape matches the
    other lanes, but the fit does not use it.
    """
    var which = FIX_SEPARATED
    var n = gmm_fixture_n(which)
    var d = gmm_fixture_d(which)
    var ncomp = gmm_fixture_k(which)
    var x = gmm_fixture(which)
    var params = GmmParams.default()
    params.n_components = ncomp
    params.covariance_type = COV_FULL
    params.max_iter = 50
    params.init_params = INIT_KMEANS
    params.random_state = UInt64(0)

    var dump = Dump("gmm")
    dump.param("fixture", gmm_fixture_name(which))
    dump.param_int("n", n)
    dump.param_int("d", d)
    dump.param_int("n_components", ncomp)
    dump.param_int("max_iter", params.max_iter)
    dump.param_f32("tol", params.tol)
    dump.param_f32("reg_covar", params.reg_covar)
    dump.param("covariance_type", "full")
    dump.param("init_params", "kmeans")
    dump.param_int("random_state", 0)
    dump.f32("x", x)
    dump.done()

    var em = Emitter(
        "gmm",
        gmm_fixture_name(which) + "." + String(n) + "x" + String(d) + "K"
        + String(ncomp),
        size,
    )
    em.header(_no_spaces(ctx.name()), rounds)
    for r in range(rounds + 1):
        var t0 = perf_counter_ns()
        var model = gaussian_mixture_fit(x, n, d, params)
        ctx.synchronize()
        var t1 = perf_counter_ns()
        var h = _fold_f32_list(FNV_OFFSET, model.weights)
        h = _fold_f32_list(h, model.means)
        h = _fold_f32_list(h, model.covariances)
        h = _fold_word(h, UInt64(model.n_iter), 4)
        if r == 0:
            em.warmup(t1 - t0)
        else:
            em.emit(r, t1 - t0, h)
        _ = model^
    em.note()
    _ = x^


# ============================================================================
# gp -- Gaussian-process regression
# ============================================================================


def run_gp(ctx: DeviceContext, smoke: Bool, rounds: Int, size: String) raises:
    """`gpr_fit_host` then `gpr_predict_host(return_std = True)` on the `ard`
    fixture: 12 training rows x 3 features, 6 test rows, an ARD RBF kernel
    with length scales [0.5, 1.0, 7.0], no hyperparameter optimizer.

    12 TRAINING ROWS. This is the fixture the lane ships and its builder takes
    a fixture id, not an `n`. There is no NVIDIA GPU Gaussian process to race
    at any size (RAPIDS ships none), so the opponent is scikit-learn on the
    CPU and the honest reading is "what does one tiny GPR fit plus predict
    cost end to end on each device". Both arms construct a context, both pay
    launch latency, and neither number is a throughput.

    `gpr_fit_host` and `gpr_predict_host` construct their own `DeviceContext`,
    like the gmm lane. Same caveat, same reason.
    """
    var which = GP_FIX_ARD
    var n = gp_fixture_n(which)
    var d = gp_fixture_d(which)
    var ns = gp_fixture_n_star(which)
    var x = gp_fixture_x(which, 0)
    var y = gp_fixture_y(which, 0)
    var xs = gp_fixture_x_star(which, 0)
    var spec = gp_fixture_kernel(which)
    var alpha = gp_fixture_alpha(which)

    var dump = Dump("gp")
    dump.param("fixture", gp_fixture_name(which))
    dump.param_int("n_train", n)
    dump.param_int("n_star", ns)
    dump.param_int("d", d)
    dump.param_f32("alpha", alpha)
    dump.param("kernel", "rbf_ard")
    dump.f32("length_scale", spec.length_scales)
    dump.f32("x", x)
    dump.f32("y", y)
    dump.f32("x_star", xs)
    dump.done()

    var em = Emitter(
        "gp",
        gp_fixture_name(which) + "." + String(n) + "x" + String(d) + "s"
        + String(ns),
        size,
    )
    em.header(_no_spaces(ctx.name()), rounds)
    for r in range(rounds + 1):
        var t0 = perf_counter_ns()
        var model = gpr_fit_host(x, n, d, y, spec, alpha)
        var pred = gpr_predict_host(model, xs, ns, True)
        ctx.synchronize()
        var t1 = perf_counter_ns()
        if model.info != 0:
            _refuse(
                "gp", "ours",
                "the kernel matrix did not factor: info=" + String(model.info),
            )
            return
        var h = _fold_f32_list(FNV_OFFSET, pred.mean)
        h = _fold_f32_list(h, pred.std)
        h = _fold_f32(h, model.logdet)
        if r == 0:
            em.warmup(t1 - t0)
        else:
            em.emit(r, t1 - t0, h)
        _ = pred^
        _ = model^
    em.note()
    _ = x^
    _ = y^
    _ = xs^


# ============================================================================
# krr / nystroem / rbfsampler -- the kernel_methods lane, three separate lanes
#
# THREE LANES AND NOT ONE. `kernel_methods/kernel_methods_main.mojo` runs all
# three estimators in one driver because they share a card. They are split
# here because their OPPONENTS are different: kernel ridge has a real cuML GPU
# counterpart and the other two do not, so folding them into one timing would
# average a GPU comparison with a CPU one.
#
# The shared fixture is `FIX_KM_RBF` (16 x 5), the shared kernel is RBF with
# gamma 0.25, alpha 0.5, `n_components = 8`, seed 20260825, three query rows.
# TRANSCRIBED from `kernel_methods_main.mojo`, which is a file with a `main`.
# ============================================================================

comptime KM_GAMMA = 0.25
comptime KM_ALPHA = Float32(0.5)
comptime KM_COMPONENTS = 8
comptime KM_SEED: UInt64 = 20260825
comptime KM_N_QUERY = 3


def _km_dump(lane: String, which: Int, n: Int, d: Int) raises:
    var dump = Dump(lane)
    dump.param("fixture", km_fixture_name(which))
    dump.param_int("n", n)
    dump.param_int("d", d)
    dump.param_int("n_query", KM_N_QUERY)
    dump.param_int("n_components", KM_COMPONENTS)
    dump.param("gamma", String(KM_GAMMA))
    dump.param_f32("alpha", KM_ALPHA)
    dump.param("kernel", "rbf")
    dump.param("seed", String(KM_SEED))
    dump.f32("x", km_fixture_x(which, 0))
    dump.f32("y", km_fixture_y(which, 1, 0))
    dump.f32("x_query", km_fixture_query(which, KM_N_QUERY, 0))
    dump.done()


def run_krr(ctx: DeviceContext, smoke: Bool, rounds: Int, size: String) raises:
    """`kernel_ridge_fit_host` then `kernel_ridge_predict_host`: RBF kernel
    ridge, gamma 0.25, alpha 0.5, on 16 x 5 with three query rows.

    The opponent is `cuml.kernel_ridge.KernelRidge`, which is a real RAPIDS
    GPU estimator and is the same algorithm: form K, ridge it, solve, then
    K(X_new, X_fit) . dual_coef. Same caveats as gmm about the internal
    context and about 16 rows being a fixed-cost measurement.
    """
    var which = FIX_KM_RBF
    var n = km_fixture_n(which)
    var d = km_fixture_d(which)
    _km_dump("krr", which, n, d)
    var x = km_fixture_x(which, 0)
    var y = km_fixture_y(which, 1, 0)
    var xq = km_fixture_query(which, KM_N_QUERY, 0)
    var kp = KernelParams(KM_KERNEL_RBF, 3, KM_GAMMA, 1.0)
    var em = Emitter(
        "krr", km_fixture_name(which) + "." + String(n) + "x" + String(d), size
    )
    em.header(_no_spaces(ctx.name()), rounds)
    for r in range(rounds + 1):
        var trace = IdentityTrace.disabled()
        var t0 = perf_counter_ns()
        var model = kernel_ridge_fit_host(x, y, n, d, 1, kp, KM_ALPHA, trace)
        var pred = kernel_ridge_predict_host(model, xq, KM_N_QUERY, trace)
        ctx.synchronize()
        var t1 = perf_counter_ns()
        var h = _fold_f32_list(FNV_OFFSET, model.dual_coef)
        h = _fold_f32_list(h, pred)
        if r == 0:
            em.warmup(t1 - t0)
        else:
            em.emit(r, t1 - t0, h)
        _ = pred^
        _ = model^
    em.note()
    _ = x^
    _ = y^
    _ = xq^


def run_nystroem(ctx: DeviceContext, smoke: Bool, rounds: Int, size: String) raises:
    """`nystroem_fit_host` then `nystroem_transform_host`: RBF Nystroem at 8
    components on 16 x 5, three query rows, seed 20260825.

    NO RAPIDS COUNTERPART. cuML ships no Nystroem, so the opponent is
    `sklearn.kernel_approximation.Nystroem` on the CPU and the arm is labeled
    `sklearn-cpu`. THE BASIS SAMPLE WILL NOT MATCH: ours permutes with a
    pinned Philox stream and theirs uses numpy's `RandomState.permutation`, so
    the two fit different basis rows and the hashes cannot be compared. That
    is a note in the markdown, not a reason to skip the timing: the WORK is
    the same (sample q rows, form the q x q kernel, eigendecompose it, scale,
    then a cross kernel and a matmul).
    """
    var which = FIX_KM_RBF
    var n = km_fixture_n(which)
    var d = km_fixture_d(which)
    _km_dump("nystroem", which, n, d)
    var x = km_fixture_x(which, 0)
    var xq = km_fixture_query(which, KM_N_QUERY, 0)
    var kp = KernelParams(KM_KERNEL_RBF, 3, KM_GAMMA, 1.0)
    var em = Emitter(
        "nystroem",
        km_fixture_name(which) + "." + String(n) + "x" + String(d) + "q"
        + String(KM_COMPONENTS),
        size,
    )
    em.header(_no_spaces(ctx.name()), rounds)
    for r in range(rounds + 1):
        var trace = IdentityTrace.disabled()
        var t0 = perf_counter_ns()
        var model = nystroem_fit_host(
            x, n, d, kp, KM_COMPONENTS, KM_SEED, trace
        )
        var emb = nystroem_transform_host(model, xq, KM_N_QUERY, trace)
        ctx.synchronize()
        var t1 = perf_counter_ns()
        var h = _fold_f32_list(FNV_OFFSET, model.eigenvalues)
        h = _fold_f32_list(h, emb)
        if r == 0:
            em.warmup(t1 - t0)
        else:
            em.emit(r, t1 - t0, h)
        _ = emb^
        _ = model^
    em.note()
    _ = x^
    _ = xq^


def run_rbfsampler(
    ctx: DeviceContext, smoke: Bool, rounds: Int, size: String
) raises:
    """`rbf_sampler_fit_host` then `rbf_sampler_transform_host`: random
    Fourier features, 8 components, gamma 0.25, seed 20260825, three rows.

    NO RAPIDS COUNTERPART. cuML ships no RBFSampler, so the opponent is
    `sklearn.kernel_approximation.RBFSampler` on the CPU, labeled
    `sklearn-cpu`. The DRAWS differ between the two (a pinned Philox stream
    against numpy's `RandomState.normal`) so the hashes are incomparable and
    the arm reports `hash=-` on the Python side.

    THIS IS THE SMALLEST LANE HERE. `fit` reads only `n_features`,
    `n_components` and the seed -- it does not look at X, and neither does
    theirs -- so the timed work is one draw plus one 3 x 5 by 5 x 8 matmul
    and a cosine. Whatever this number is, it is a launch-latency number.
    """
    var which = FIX_KM_RBF
    var n = km_fixture_n(which)
    var d = km_fixture_d(which)
    _km_dump("rbfsampler", which, n, d)
    var xq = km_fixture_query(which, KM_N_QUERY, 0)
    var em = Emitter(
        "rbfsampler",
        String(KM_N_QUERY) + "x" + String(d) + "q" + String(KM_COMPONENTS),
        size,
    )
    em.header(_no_spaces(ctx.name()), rounds)
    for r in range(rounds + 1):
        var trace = IdentityTrace.disabled()
        var t0 = perf_counter_ns()
        var model = rbf_sampler_fit_host(
            d, KM_COMPONENTS, Float32(KM_GAMMA), KM_SEED, trace
        )
        var z = rbf_sampler_transform_host(model, xq, KM_N_QUERY, trace)
        ctx.synchronize()
        var t1 = perf_counter_ns()
        var h = _fold_f32_list(FNV_OFFSET, z)
        if r == 0:
            em.warmup(t1 - t0)
        else:
            em.emit(r, t1 - t0, h)
        _ = z^
        _ = model^
    em.note()
    _ = xq^


# ============================================================================
# resample -- the bootstrap
# ============================================================================


def run_resample(ctx: DeviceContext, smoke: Bool, rounds: Int, size: String) raises:
    """`bootstrap_host`: the percentile bootstrap of the MEAN, 10,000
    resamples, confidence 0.95, two-sided, seed 20260825, on `FIX_HASHED`
    (200 x 2). BCa diagnostics OFF.

    TRANSCRIBED from `resample/resample_main.mojo`: its `RESAMPLE_MAIN_SEED`,
    its `RESAMPLE_MAIN_CONFIDENCE`, its default statistic and method and its
    default `n_resamples`. `with_bca_diagnostics` is FALSE here and TRUE in
    that driver, because SciPy's `method="percentile"` computes no jackknife
    and an arm that computed one would be doing extra work the opponent does
    not do.

    THIS IS ONE OF THE FEW SMALL-FIXTURE LANES WITH REAL WORK IN IT: 10,000
    resamples of 200 rows is two million draws, so the number is about the
    kernel and not only about the launch. There is no RAPIDS bootstrap; the
    opponent is `scipy.stats.bootstrap` on the CPU, labeled `scipy-cpu`.
    """
    var fix = FIX_HASHED
    var n = resample_fixture_n(fix)
    var d = resample_fixture_d(fix)
    var n_resamples = 1000 if smoke else 10000
    var seed = UInt64(20260825)
    var confidence = Float32(0.95)
    var x = build_sample(fix)

    var dump = Dump("resample")
    dump.param_int("n", n)
    dump.param_int("d", d)
    dump.param_int("n_resamples", n_resamples)
    dump.param("seed", String(seed))
    dump.param_f32("confidence_level", confidence)
    dump.param("statistic", "mean")
    dump.param("method", "percentile")
    dump.param("alternative", "two-sided")
    dump.f32("x", x)
    dump.done()

    var em = Emitter(
        "resample",
        String(n) + "x" + String(d) + "r" + String(n_resamples),
        size,
    )
    em.header(_no_spaces(ctx.name()), rounds)
    for r in range(rounds + 1):
        var t0 = perf_counter_ns()
        var res = bootstrap_host(
            x, n, d, STAT_MEAN, n_resamples, seed, METHOD_PERCENTILE,
            confidence, ALT_TWO_SIDED, Float32(0.5), 0, 256, False,
        )
        ctx.synchronize()
        var t1 = perf_counter_ns()
        var h = _fold_f32(FNV_OFFSET, res.point_estimate)
        h = _fold_f32(h, res.standard_error)
        h = _fold_f32_list(h, res.sorted_distribution)
        if r == 0:
            em.warmup(t1 - t0)
        else:
            em.emit(r, t1 - t0, h)
        _ = res^
    em.note()
    _ = x^


# ============================================================================
# spectral -- spectral clustering from a dataset
# ============================================================================


def run_spectral(ctx: DeviceContext, smoke: Bool, rounds: Int, size: String) raises:
    """`fit_predict_dataset`, rung 2: 48 blob rows x 4 features -> a 10-NN
    connectivity graph -> a normalized-Laplacian embedding -> k-means at
    3 clusters. `n_components = 3`, `n_init = 10`, tolerance 1e-5, seed 7.

    TRANSCRIBED from `spectral/spectral_main.mojo`'s `MOJOLEARN_SPECTRAL_
    CLUSTER=1` branch, including the `^ 0xB10B` on the blob seed and the
    hardcoded 48 rows, which is `16 * 3`.

    RAPIDS SHIPS NO SPECTRAL CLUSTERING ESTIMATOR. The opponent is
    `sklearn.cluster.SpectralClustering(affinity="nearest_neighbors")` on the
    CPU. Both sides build a kNN affinity, take a Lanczos eigendecomposition of
    the normalized Laplacian and run k-means on the embedding, so the pipeline
    matches even though the eigensolvers are two Lanczos implementations
    (ours restarts; theirs is ARPACK). Labels are a permutation of each other
    at best, so the hashes are not comparable and the Python arm says so.
    """
    var seed = 7
    var n_per_blob = 16
    var n_blobs = 3
    var d = 4
    var rows = n_per_blob * n_blobs
    var bl = blobs_fixture(n_per_blob, n_blobs, d, UInt64(seed) ^ UInt64(0xB10B))
    var data = bl[0].copy()

    var dump = Dump("spectral")
    dump.param_int("n", rows)
    dump.param_int("d", d)
    dump.param_int("n_clusters", 3)
    dump.param_int("n_components", 3)
    dump.param_int("n_init", 10)
    dump.param_int("n_neighbors", 10)
    dump.param_f32("tolerance", Float32(1.0e-5))
    dump.param_int("seed", seed)
    dump.f32("x", data)
    dump.done()

    var cfg = SpectralClusteringParams(
        n_clusters=3, n_components=3, n_init=10, n_neighbors=10,
        tolerance=Float32(1.0e-5), seed=UInt64(seed),
    )
    var em = Emitter("spectral", String(rows) + "x" + String(d) + "c3", size)
    em.header(_no_spaces(ctx.name()), rounds)
    for r in range(rounds + 1):
        var labels = List[Int32]()
        var emb = List[Float32]()
        var trace = IdentityTrace.disabled()
        var t0 = perf_counter_ns()
        fit_predict_dataset(ctx, cfg, data, rows, d, labels, emb, trace)
        ctx.synchronize()
        var t1 = perf_counter_ns()
        var h = _fold_i32_list(FNV_OFFSET, labels)
        h = _fold_f32_list(h, emb)
        if r == 0:
            em.warmup(t1 - t0)
        else:
            em.emit(r, t1 - t0, h)
        _ = labels^
        _ = emb^
    em.note()
    _ = data^


# ============================================================================
# holtwinters
#
# THE ONLY LANE HERE WITH NO `*_main.mojo` OF ITS OWN. Its sizes are
# TRANSCRIBED from `holtwinters/mojo_only/hw_check.mojo` (N = 72, FREQ = 12,
# BATCH = 7), which is a file that carries a `main`. Same rule and same
# reason as the metrics sizes above: if that file's constants move, these
# must move with them.
# ============================================================================

comptime HW_N = 72
comptime HW_FREQ = 12
comptime HW_BATCH = 7
comptime HW_START_PERIODS = 2
comptime HW_SALT = 2


def run_holtwinters(
    ctx: DeviceContext, smoke: Bool, rounds: Int, size: String
) raises:
    """`holtwinters_fit_host_traced`: additive Holt-Winters over 7 series of
    72 observations at frequency 12, `start_periods = 2`, the lane's default
    eps.

    The TRACED form is called rather than `holtwinters_fit_host` so the
    `DeviceContext` is the harness's and not one constructed per round; the
    trace passed in is DISABLED, so no stage is recorded and no queue is
    drained for the instrument. That makes this the one host-list lane whose
    number is not inflated by a context construction, and it is worth saying
    because the gmm, gp and kernel_methods lanes have no such entry.

    The opponent is `cuml.ExponentialSmoothing`, a real RAPIDS GPU estimator
    with the same `ts_num` / `seasonal_periods` / `start_periods` surface.
    """
    var data = hw_fixture(spec_additive(), HW_N, HW_BATCH, HW_FREQ, HW_SALT)

    var dump = Dump("holtwinters")
    dump.param_int("n", HW_N)
    dump.param_int("batch_size", HW_BATCH)
    dump.param_int("frequency", HW_FREQ)
    dump.param_int("start_periods", HW_START_PERIODS)
    dump.param("seasonal", "additive")
    dump.param_f32("eps", HW_DEFAULT_EPS)
    dump.param("layout", "series-major (batch_size x n)")
    dump.f32("y", data)
    dump.done()

    var em = Emitter(
        "holtwinters",
        String(HW_BATCH) + "x" + String(HW_N) + "f" + String(HW_FREQ),
        size,
    )
    em.header(_no_spaces(ctx.name()), rounds)
    for r in range(rounds + 1):
        var trace = IdentityTrace.disabled()
        var t0 = perf_counter_ns()
        var fitted = holtwinters_fit_host_traced(
            ctx, data, HW_N, HW_BATCH, HW_FREQ, HW_START_PERIODS, "additive",
            HW_DEFAULT_EPS, trace,
        )
        ctx.synchronize()
        var t1 = perf_counter_ns()
        var h = _fold_f32_list(FNV_OFFSET, fitted.level)
        h = _fold_f32_list(h, fitted.trend)
        h = _fold_f32_list(h, fitted.season)
        h = _fold_f32_list(h, fitted.sse)
        if r == 0:
            em.warmup(t1 - t0)
        else:
            em.emit(r, t1 - t0, h)
        _ = fitted^
    em.note()
    _ = data^


# ============================================================================
# kpss -- the stationarity test
# ============================================================================


comptime KPSS_N_OBS = 520
comptime KPSS_SALT = 1


def run_kpss(ctx: DeviceContext, smoke: Bool, rounds: Int, size: String) raises:
    """`kpss_test` at `(d, D, s) = (1, 0, 0)` and a 0.05 threshold, on
    `tsa/tsa_main.mojo`'s eight-series batch of 520 observations.

    TRANSCRIBED from `tsa/tsa_main.mojo`: `N_OBS = 520`, `SALT = 1`, the same
    `(1, 0, 0)` and the same threshold. `select_d` is NOT in the timed region;
    it runs `kpss_test` up to three more times and is a different question.

    The opponent is cuML's own `kpss_test` where `cuml.tsa.stationarity`
    exposes one, and `statsmodels.tsa.stattools.kpss` per series on the CPU
    where it does not. Whichever ran is in the arm label.
    """
    var f = kpss_fixture(KPSS_N_OBS, KPSS_SALT)
    var batch = f.batch_size
    var n_obs = f.n_obs

    var dump = Dump("kpss")
    dump.param_int("n_obs", n_obs)
    dump.param_int("batch_size", batch)
    dump.param_int("d", 1)
    dump.param_int("D", 0)
    dump.param_int("s", 0)
    dump.param("pval_threshold", "0.05")
    dump.param("layout", "series-major (batch_size x n_obs)")
    dump.f32("y", f.y)
    dump.done()

    var em = Emitter("kpss", String(batch) + "x" + String(n_obs), size)
    em.header(_no_spaces(ctx.name()), rounds)
    for r in range(rounds + 1):
        # Re-uploaded every round: `kpss_test` takes `d_y` as `mut` and the
        # differencing arm may write through it, so a shared buffer would make
        # round two a test of round one's output.
        var y = tsa_upload_f32(ctx, f.y)
        ctx.synchronize()
        var t0 = perf_counter_ns()
        var res = kpss_test(ctx, y, batch, n_obs, 1, 0, 0, 0.05)
        ctx.synchronize()
        var t1 = perf_counter_ns()
        var h = _hash_device_f32(ctx, FNV_OFFSET, res.scratch.stat, batch)
        if r == 0:
            em.warmup(t1 - t0)
        else:
            em.emit(r, t1 - t0, h)
        _ = res^
        _ = y^
    em.note()
    _ = f^


# ============================================================================
# main
# ============================================================================


def main() raises:
    var lane = String(getenv("MOJOLEARN_SPEED_LANE"))
    var rounds = _env_int("MOJOLEARN_SPEED_ROUNDS", 5)
    var size_env = String(getenv("MOJOLEARN_SPEED_SIZE"))
    if size_env == "":
        size_env = String("shipped")
    if size_env != "shipped" and size_env != "smoke":
        raise Error(
            "classical_speed: MOJOLEARN_SPEED_SIZE must be `shipped` or"
            " `smoke`; got '" + size_env + "'"
        )
    var smoke = size_env == "smoke"
    if rounds < 1:
        raise Error("classical_speed: MOJOLEARN_SPEED_ROUNDS must be >= 1")

    var ctx = DeviceContext()

    if lane == "kmeans":
        run_kmeans(ctx, smoke, rounds, size_env)
    elif lane == "dbscan":
        run_dbscan(ctx, smoke, rounds, size_env)
    elif lane == "pca":
        run_pca(ctx, smoke, rounds, size_env)
    elif lane == "ols":
        run_ols(ctx, smoke, rounds, size_env)
    elif lane == "knn":
        run_knn(ctx, smoke, rounds, size_env)
    elif lane == "cd":
        run_cd(ctx, smoke, rounds, size_env)
    elif lane == "kde":
        run_kde(ctx, smoke, rounds, size_env)
    elif lane == "linkage":
        run_linkage(ctx, smoke, rounds, size_env)
    elif lane == "svm":
        run_svm(ctx, smoke, rounds, size_env)
    elif lane == "metrics":
        run_metrics(ctx, smoke, rounds, size_env)
    elif lane == "ivf":
        run_ivf(ctx, smoke, rounds, size_env)
    elif lane == "hdbscan":
        run_hdbscan(ctx, smoke, rounds, size_env)
    elif lane == "cholesky":
        run_cholesky(ctx, smoke, rounds, size_env)
    elif lane == "gmm":
        run_gmm(ctx, smoke, rounds, size_env)
    elif lane == "gp":
        run_gp(ctx, smoke, rounds, size_env)
    elif lane == "krr":
        run_krr(ctx, smoke, rounds, size_env)
    elif lane == "nystroem":
        run_nystroem(ctx, smoke, rounds, size_env)
    elif lane == "rbfsampler":
        run_rbfsampler(ctx, smoke, rounds, size_env)
    elif lane == "resample":
        run_resample(ctx, smoke, rounds, size_env)
    elif lane == "spectral":
        run_spectral(ctx, smoke, rounds, size_env)
    elif lane == "holtwinters":
        run_holtwinters(ctx, smoke, rounds, size_env)
    elif lane == "kpss":
        run_kpss(ctx, smoke, rounds, size_env)
    else:
        raise Error(
            "classical_speed: MOJOLEARN_SPEED_LANE must be one of kmeans"
            " dbscan pca ols knn cd kde linkage svm metrics ivf hdbscan"
            " cholesky gmm gp krr nystroem rbfsampler resample spectral"
            " holtwinters kpss; got '" + lane + "'"
        )
