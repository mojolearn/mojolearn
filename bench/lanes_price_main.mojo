# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""What IDENTICAL costs on the frozen lanes: ONE driver, one lane per run.

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

AND SIX MORE LANES, THE ARMS SECTION 7 OF THE PAPER PRICES, ported here on
2026-09-02 from the two Apple-only mains so the same question can be asked
on NVIDIA and AMD. Those two mains are driven by
`tools/price_unsupervised_identity.sh` and `tools/price_linalg_identity.sh`,
neither of which has any remote-leg wiring, so every number the paper prints
for them is a ONE-BOX number. This driver is the one
`tools/diag/identity_cost_leg.sh` rides onto a rented box, which is the
whole reason for the port:

    kmeans    bench/identity_price_main.mojo:92   `kmeans_fit_main`, the Lloyd loop end to end
    knn       bench/identity_price_main.mojo:150  `brute_force_knn_impl`, AUTO by default, TILED on request
    dbscan    bench/identity_price_main.mojo:195  `dbscan_fit_impl`, brute-force eps neighbourhood
    gram      bench/linalg_price_main.mojo:94     `gemm_tn`, the Gram shape `A^T A`
    nt        bench/linalg_price_main.mojo:124    `gemm_nt`, the N-T product
    gemv      bench/linalg_price_main.mojo:151    `gemv_n`, OLS's step 6

WHY each of those six is worth pricing is argued in the two source files'
docstrings and is not re-argued here; what each lane function below records
is where its fixture came from, line by line, so a reader can check the port
against the original without leaving the file.

AND THREE TREE LANES, ADDED THE SAME DAY, BECAUSE THERE WERE NONE. The
twelve lanes above cover clustering, neighbours, density, linkage, svm,
metrics and linear algebra and NOT ONE OF THEM FITS A TREE. Decision trees
are the largest certified family in this project -- 109 of 180
bitwise-identical training configurations -- and they are the workloads the
performance section leads with, so until today the one question this harness
exists to answer had never been asked of them:

    gbdt   checks/estimation_bench.mojo:76-81  `gbdt.train.train`, symmetric
                                               (oblivious) boosting, Logloss
    rf     ensemble/bench/rf_bench.mojo:319    `fit_forest[ClsObj]`, the
                                               random forest
    et     extratrees/bench/fit_once.mojo:153  `train_forest_classification_device`,
                                               extremely randomized trees

These three DIFFER FROM EVERY LANE ABOVE IN ONE WAY THAT MATTERS TO THE SIZE
TABLE. [[mojolearn-large-data-timing-floor]] is a hard rule -- tree
performance is never measured or decided below 1,000,000 rows -- so their
defaults are 1,000,000 rows and not the 2048, 4096 or 20000 rows the six
Section 7 lanes default to. A 20000-row tree fit is not a small price, it is
an inadmissible one, and the 50,000-row symmetric window that was caught and
killed on 2026-09-01 (`bench/results/ab_large_2026-09-01/RESULTS.md:3-7`,
nothing from it votes anywhere) is why the rule is a rule. Each lane
therefore takes TWO knobs, rows AND trees, and it is the TREE count that was
dialed to bring one fit into the seconds this harness wants. Each lane
function says which of its numbers came from a measured window and which
were reasoned from one.

They also build their own fixtures rather than importing one. Every fixture
above is either imported from the lane it prices or transcribed from a price
main that already existed; there is no tree price main to port, and the tree
benches that do exist read HIGGS or epsilon from `~/.cache/mojolearn`, which
a rented box has not staged. So all three fixtures are built in the lane out
of this file's own `_price_u01`, with an axis-aligned label whose reason is
argued at `_tree_hits3` and is a TIMING reason, not an accuracy one.

THE PROTOCOL IS THIS FILE'S, NOT THE TWO SOURCES'. They disagree with each
other and both disagree with this harness, so neither is carried over.
`bench/identity_price_main.mojo` runs `REPEATS = 3` (`:63`) with NO untimed
warm-up and prints one line per repeat, so its first line pays every
kernel's first launch and lands in the same median as the other two.
`bench/linalg_price_main.mojo` takes ONE untimed warm-up per arm and then
AVERAGES its three timed reps into a SINGLE printed sample (`:104-117`), so
its band cannot be read per round at all. Every lane here does what cd and
kde already do: one untimed warm-up round, then ROUNDS individually timed
and individually hashed rounds, which is the shape `tools/lanes_price.sh`
pairs mode against mode and turns into a band. A port that kept its source's
protocol would produce a row that cannot be read beside the rows above it.

THE SIZE KNOB, AND WHY THE SIX NEW LANES HAVE ONE AND THE SIX OLD ONES DO
NOT. Every fixture in this file was built on an Apple M4 and sized for it.
On a datacenter GPU an Apple-sized fixture measures the launch and not the
arm. Measured 2026-09-02 at one commit on an M4 and an H100, 5 rounds: the
cd lane took 1.3 ms on the H100 against 89 ms on the M4, and the gemm lane
took 100 MICROSECONDS against 9.2 ms, and at 100 us that gemm lane's
per-round ratio band came out 0.205 .. 0.773 -- a band that straddles 1.0
and spends most of its width below it. Four of six H100 lanes and five of
six Apple lanes straddled 1.0 that day. A FIXTURE THAT CANNOT SEPARATE THE
TWO ARMS IS A FIXTURE THAT IS TOO SMALL, NOT A RESULT, and that
100-microsecond gemm is the recorded example of it. So each of the six new
lanes reads its size from the environment, with the DEFAULT set to the size
the published Apple numbers were taken at (so those stay reproducible) and a
larger step named beside it for a datacenter box:

    lane    variable                            default (Apple)  datacenter step
    kmeans  MOJOLEARN_LANES_PRICE_KMEANS_ROWS   100000           2000000
    knn     MOJOLEARN_LANES_PRICE_KNN_INDEX     20000            400000
    dbscan  MOJOLEARN_LANES_PRICE_DBSCAN_ROWS   20000            100000
    gram    MOJOLEARN_LANES_PRICE_GRAM_ROWS     1000000          8000000
    nt      MOJOLEARN_LANES_PRICE_NT_ROWS       4096             262144
    gemv    MOJOLEARN_LANES_PRICE_GEMV_DIM      128              8192

and the three tree lanes, whose row default is a FLOOR rather than a
published Apple size, take a tree count beside it:

    lane    variable                            default (Apple)  datacenter step
    gbdt    MOJOLEARN_LANES_PRICE_GBDT_ROWS     1000000          2000000
            MOJOLEARN_LANES_PRICE_GBDT_TREES    20               60
    rf      MOJOLEARN_LANES_PRICE_RF_ROWS       1000000          2000000
            MOJOLEARN_LANES_PRICE_RF_TREES      4                16
    et      MOJOLEARN_LANES_PRICE_ET_ROWS       1000000          2000000
            MOJOLEARN_LANES_PRICE_ET_TREES      20               60

THE TREE ROW DEFAULTS ARE NOT NEGOTIABLE DOWNWARD and they are not a fixture
that was sized for an M4. 1,000,000 is [[mojolearn-large-data-timing-floor]]'s
floor and the smaller sizes the twelve lanes above use would be meaningless
here for a reason that is stronger than "too small to separate": a tree fit
below the floor is a DIFFERENT PROBLEM, not a cheaper one. Under the floor
the per-tree fixed costs (quantization, border building, pool setup, the
per-level launch tax) dominate the per-row work, so the measurement prices
the harness's arrival rather than the arm. `bench/results/PERF_2026-08-20_fixed-cost.md`
puts the symmetric arm at 9.43 ms fixed plus 18.5 us per 1000 rows, which at
20000 rows is 96% fixed cost and at 1,000,000 rows is 34%. Every default in
the tree rows is taken from a window this tree has already run at the floor:
et's 28 x 20 trees x depth 12 at 1M and 2M is verbatim
`bench/results/ab_large_2026-09-01/RESULTS.md:14-16`, gbdt's 100 columns at
depth 6 is `checks/estimation_bench.mojo:60` with that window's patched
`N_ROWS`, and rf's 50 columns at depth 12 and 128 bins is
`ensemble/bench/rf_bench.mojo:62-64` at that window's patched row count. Only
the TREE counts were dialed, and only downward, and each lane says by how
much and against which measured seconds.

The right-hand column is a STARTING POINT, not a measurement: it is the step
to try first, and if its band still straddles 1.0 the answer is a bigger
fixture and not a rerun. Two of the six are not invented here --
`bench/scaling_main.mojo:180` already sweeps k-NN at 400,000 index rows with
the same `buf_len = max(n_index // 8, k)` formula this lane uses (`:50`),
and `:185` sweeps brute-force DBSCAN at 200,000 rows. DBSCAN's step stops
below that because the brute arm's cost is QUADRATIC in the rows: that
file's `_dbscan_rbc_only_at` docstring puts brute at 800,000 near four and a
half minutes PER REPEAT on the laptop, and this harness runs one warm-up
plus ROUNDS rounds in each of two modes.

THE SIX ORIGINAL LANES TAKE NO SIZE KNOB AND MUST NOT GAIN ONE HERE. An
MI325X price run was taken at their spellings and their sizes
(`bench/LANES_PRICE.md`, 2026-08-31, sha `035493e1`), and a size that moves
is a row that can no longer be compared with it.

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
    MOJOLEARN_LANES_PRICE_LANE    cd | kde | linkage | svm | metrics | gemm |
                                  kmeans | knn | dbscan | gram | nt | gemv |
                                  gbdt | rf | et (required)
    MOJOLEARN_LANES_PRICE_ROUNDS  timed rounds in this process (default 5; the
                                  script sets 1 and alternates processes instead)
    MOJOLEARN_LANES_PRICE_SMOKE   1 -> tiny sizes, for proving the build, the
                                  witness and the hash; NEVER FOR A NUMBER,
                                  and on the three tree lanes never for a
                                  number twice over: a SMOKE tree fit is 4096
                                  rows, which is below the 1,000,000-row floor
                                  those lanes exist under, so it is not a
                                  small price but an inadmissible one
    MOJOLEARN_LANES_PRICE_GEMM_SHAPE  row of bench/gemm_shapes.mojo (default 6,
                                  kmeans.dist.4096x64x64, the E3 table's shape)
    MOJOLEARN_LANES_PRICE_SVM_FIXTURE index into svm `all_fixtures()` (default 1, F2.xor)
    MOJOLEARN_LANES_PRICE_KMEANS_ROWS n rows for the kmeans lane (default 100000)
    MOJOLEARN_LANES_PRICE_KNN_INDEX   index rows for the knn lane (default 20000)
    MOJOLEARN_LANES_PRICE_KNN_METHOD  auto | tiled for the knn lane (default auto)
    MOJOLEARN_LANES_PRICE_DBSCAN_ROWS n rows for the dbscan lane (default 20000)
    MOJOLEARN_LANES_PRICE_GRAM_ROWS   k rows for the gram lane (default 1000000)
    MOJOLEARN_LANES_PRICE_NT_ROWS     m rows for the nt lane (default 4096)
    MOJOLEARN_LANES_PRICE_GEMV_DIM    the square m = k for the gemv lane (default 128)
    MOJOLEARN_LANES_PRICE_GBDT_ROWS   n rows for the gbdt lane (default 1000000)
    MOJOLEARN_LANES_PRICE_GBDT_TREES  n_estimators for the gbdt lane (default 20)
    MOJOLEARN_LANES_PRICE_RF_ROWS     n rows for the rf lane (default 1000000)
    MOJOLEARN_LANES_PRICE_RF_TREES    n_trees for the rf lane (default 4)
    MOJOLEARN_LANES_PRICE_ET_ROWS     n rows for the et lane (default 1000000)
    MOJOLEARN_LANES_PRICE_ET_TREES    n_trees for the et lane (default 20)

No pixi task. No number printed by this file is a measurement until
`bench/LANES_PRICE.md`'s clean-window procedure produced it.
"""

from std.memory import bitcast
from std.os import getenv
from std.time import perf_counter_ns

from max.gpu.host import DeviceBuffer, DeviceContext, HostBuffer

from core.identity_trace import FNV_OFFSET, FNV_PRIME, IdentityTrace, _hex16
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL, numeric_mode_name

# ---- cd --------------------------------------------------------------------
from solver.checks.cd_oracle import fixture_planted_sparse
from solver.impl.solver.cd import CdLaunch, cd_fit_traced
from solver.impl.solvers.params import LOSS_SQRD_LOSS

# ---- kde -------------------------------------------------------------------
from kde.checks.kde_fixture import query_fixture, train_fixture, weight_fixture
from kde.impl.kde.kde import score_samples
from kde.impl.neighbors.kernel_density import (
    host_sum_weights,
    kde_fit_validate,
    kde_validate_data,
    kernel_from_name,
    metric_from_name,
)

# ---- linkage ---------------------------------------------------------------
from hierarchy.checks.linkage_oracle import (
    FIX_BLOBS_DUPS,
    FIX_DUPS,
    build_fixture,
    fixture_d,
    fixture_n,
    fixture_n_clusters,
    fixture_name,
)
from hierarchy.impl.cluster.detail.connectivities import DISTANCE_L2_SQRT_EXPANDED
from hierarchy.impl.hierarchy.linkage import single_linkage

# ---- svm -------------------------------------------------------------------
from svm.checks.svc_check import Fixture, _run_device, all_fixtures

# ---- metrics ---------------------------------------------------------------
from metrics.checks.device_io import upload_f32, upload_i32
from metrics.checks.fixtures import (
    hashed_floats,
    hashed_pdf,
    hashed_points,
    labels_true_pred,
    u01,
)
from metrics.impl.metrics.accuracy_score import accuracy_score_py
from metrics.impl.metrics.adjusted_rand_index import adjusted_rand_index
from metrics.impl.metrics.completeness_score import completeness_score
from metrics.impl.metrics.entropy import entropy
from metrics.impl.metrics.homogeneity_score import homogeneity_score
from metrics.impl.metrics.kl_divergence import kl_divergence
from metrics.impl.metrics.mutual_info_score import mutual_info_score
from metrics.impl.metrics.r2_score import r2_score_py
from metrics.impl.metrics.rand_index import rand_index
from metrics.impl.metrics.silhouette_score_batched_float import silhouette_score
from metrics.impl.metrics.trustworthiness import trustworthiness_score_traced
from metrics.impl.metrics.v_measure import v_measure

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
from gemm.checks.gemm_identical import identical_gemm
from gemm.checks.gemm_oracle import OP_NN as ORACLE_OP_NN
from gemm.checks.gemm_oracle import OP_NT as ORACLE_OP_NT
from gemm.checks.gemm_oracle import OP_TN as ORACLE_OP_TN

# ---- kmeans / knn / dbscan (bench/identity_price_main.mojo's arms) ----------
from cluster.impl.cluster.detail.kmeans import kmeans_fit_main
from cluster.impl.cluster.kmeans_params import INIT_ARRAY, KMeansParams
from dbscan.impl.dbscan.dbscan import dbscan_fit_impl
from dbscan.impl.dbscan.runner import EPS_NN_BRUTE_FORCE
from neighbors.impl.neighbors.detail.knn_brute_force import (
    KNN_METHOD_AUTO,
    KNN_METHOD_TILED,
    brute_force_knn_impl,
    compute_norms,
)

# ---- gram / nt / gemv (bench/linalg_price_main.mojo's arms) -----------------
from core.gemm import gemm_nt, gemm_tn, gemv_n

# ---- gbdt / rf / et (the tree families; no price main existed to port) -----
# `DecisionTreeParams` is spelled once under `ensemble/` and once under
# `extratrees/` and the two are DIFFERENT structs (the extratrees one carries
# `max_leaf_nodes`, DEVIATION 466, and its `__init__` defaults `max_depth` to
# 16). They are aliased apart here so neither lane can be handed the other
# lane's parameter block by an import that merely looks right.
from ensemble.decisiontree.batched_levelalgo.bins import ClassificationBin
from ensemble.decisiontree.batched_levelalgo.objectives import (
    ClassificationObjectiveFunction,
)
from ensemble.decisiontree.decisiontree import GINI as RF_GINI
from ensemble.decisiontree.decisiontree import (
    DecisionTreeParams as RfTreeParams,
)
from ensemble.randomforest import RF_params, fit_forest
from extratrees.estimator import (
    MAX_FEATURES_SQRT,
    count_to_ratio,
    resolve_max_features,
)
from extratrees.impl.decisiontree.batched_levelalgo.builder import (
    train_forest_classification_device,
    upload_dataset,
)
from extratrees.impl.decisiontree.decisiontree import (
    DecisionTreeParams as EtTreeParams,
)
from extratrees.impl.randomforest.randomforest import class_ids_for
from gbdt.train import train as gbdt_train


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


def _fold_u32(h: UInt64, v: UInt32) -> UInt64:
    # Same masking as `_fold_i32` and for the same reason. An unsigned
    # source cannot sign-extend, but the mask costs nothing and the two
    # folds then read identically, so neither can drift from the other.
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


def _hash_device_u32(
    ctx: DeviceContext, h: UInt64, buf: DeviceBuffer[DType.uint32], n: Int
) raises -> UInt64:
    """The `uint32` sibling. k-means labels and k-NN neighbour indices are
    `uint32` buffers, and they are OUTPUTS, so they belong in the hash."""
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
        # bits exactly -- `solver/impl/solver/cd.mojo::cd_fit` says so),
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
# kmeans, knn, dbscan -- the three arms of bench/identity_price_main.mojo
# ============================================================================


def _price_u01(row: Int, k: Int, salt: Int) -> Float32:
    """TRANSCRIBED from `bench/identity_price_main.mojo:76-85`, byte for byte.

    Named `_price_u01` and not `_u01` because `u01` is already imported into
    this file from `metrics/checks/fixtures.mojo` and the two are DIFFERENT
    generators. The three fixtures below are the fixtures the published
    Apple numbers were taken on only if these bits are that file's bits, so
    the mixer is transcribed rather than approximated -- the same choice,
    for the same reason, that `_gemm_mix` above records.
    """
    var z = (
        UInt64(row) * 0x9E3779B97F4A7C15
        + UInt64(k + 1) * 0xBF58476D1CE4E5B9
        + UInt64(salt + 1) * 0x94D049BB133111EB
    )
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9
    z = (z ^ (z >> 27)) * 0x94D049BB133111EB
    z = z ^ (z >> 31)
    return Float32(Int((z >> 40) & UInt64(0xFFFF))) / Float32(65536.0)


def run_kmeans(ctx: DeviceContext, smoke: Bool, rounds: Int) raises:
    """`bench/identity_price_main.mojo:92`'s `kmeans.fit` arm: the Lloyd loop
    end to end through `kmeans_fit_main`, `INIT_ARRAY`, `n_init` 1,
    `max_iter` 10, `tol` 1e-12, at d 32 and k 16, on rows
    `Float32(i % k) * 3 + _price_u01(i, f, 1)` with unit weights and
    centroids seeded from `Float32(j) * 3 + _price_u01(j, f, 77)`. The
    fixture and the parameters are transcribed from `:93-133`.

    `tol` IS 1e-12 AND THAT IS THE ARM (`:128-133` says why): it is the
    smallest tolerance `validate()` accepts, so both modes run all ten
    iterations. A fit that stops on a different iteration in the two modes
    is not a price, it is a different amount of work.

    Rows come from `MOJOLEARN_LANES_PRICE_KMEANS_ROWS`, default 100000 (the
    Apple size, so the published number stays reproducible), 2000000 the
    datacenter step. d and k are deliberately NOT knobs: the fixture's
    cluster structure IS `i % k` at a spacing of 3, so k moves the fixture
    and not merely its size, and 32 x 16 is the aspect the Apple number was
    taken at.
    """
    var n = _env_int(
        "MOJOLEARN_LANES_PRICE_KMEANS_ROWS", 2048 if smoke else 100000
    )
    var d = 32
    var k = 16

    var hx = ctx.enqueue_create_host_buffer[DType.float32](n * d)
    var hw = ctx.enqueue_create_host_buffer[DType.float32](n)
    var hc = ctx.enqueue_create_host_buffer[DType.float32](k * d)
    ctx.synchronize()
    for i in range(n):
        hw.unsafe_ptr().unsafe_store(i, Float32(1.0))
        for f in range(d):
            hx.unsafe_ptr().unsafe_store(
                i * d + f,
                Float32(i % k) * Float32(3.0) + _price_u01(i, f, 1),
            )
    for j in range(k):
        for f in range(d):
            hc.unsafe_ptr().unsafe_store(
                j * d + f, Float32(j) * Float32(3.0) + _price_u01(j, f, 77)
            )

    var x = ctx.enqueue_create_buffer[DType.float32](n * d)
    var w = ctx.enqueue_create_buffer[DType.float32](n)
    var cent = ctx.enqueue_create_buffer[DType.float32](k * d)
    var labels = ctx.enqueue_create_buffer[DType.uint32](n)
    ctx.synchronize()
    ctx.enqueue_copy(dst_buf=x, src_ptr=hx.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=w, src_ptr=hw.unsafe_ptr())
    ctx.synchronize()

    var params = KMeansParams.default()
    params.n_clusters = k
    params.init = INIT_ARRAY
    params.n_init = 1
    params.max_iter = 10
    params.tol = 1.0e-12

    var ledger = Ledger(
        "kmeans", String(n) + "x" + String(d) + ".k" + String(k)
    )
    for r in range(rounds + 1):
        # The fit READS `cent` as its starting set under `INIT_ARRAY` and
        # overwrites it with the best restart
        # (`cluster/impl/cluster/detail/kmeans.mojo:922`'s docstring says
        # both), so the seed is re-uploaded every round and every round is
        # therefore the SAME fit rather than a continuation. That is the
        # only re-initialization the upstream arm does between its repeats
        # (`bench/identity_price_main.mojo:136-137`), and `x` and `w` are
        # left alone here for the same reason: if the fit mutated them the
        # way `cdFit` mutates its inputs, the in-process hash verdict below
        # would print HASH-MOVED, which is exactly how the cd lane's
        # in-place mutation was found.
        ctx.enqueue_copy(dst_buf=cent, src_ptr=hc.unsafe_ptr())
        ctx.enqueue_memset(labels, UInt32(0xFFFFFFFF))
        ctx.synchronize()
        var t0 = perf_counter_ns()
        var res = kmeans_fit_main(
            ctx, x, w, cent, labels, params, n, d,
            Float32(4096.0), Float32(4096.0),
        )
        ctx.synchronize()
        var t1 = perf_counter_ns()
        var h = _hash_device_f32(ctx, FNV_OFFSET, cent, k * d)
        h = _hash_device_u32(ctx, h, labels, n)
        h = _fold_f64(h, res.inertia)
        h = _fold_word(h, UInt64(res.n_iter), 4)
        ledger.emit("warmup" if r == 0 else String(r), t1 - t0, h)
    ledger.verdict()
    _ = hx^
    _ = hw^
    _ = hc^
    _ = x^
    _ = w^
    _ = cent^
    _ = labels^


def run_knn(ctx: DeviceContext, smoke: Bool, rounds: Int) raises:
    """`bench/identity_price_main.mojo:150`'s k-NN arms: `brute_force_knn_impl`
    on `_price_u01(i, f, 3)` index rows against `_price_u01(i, f, 5)`
    queries, d 32, k 10, query tile 256, `buf_len = max(n_index // 8, k)`,
    `is_sqrt` True, row-major on both sides, vendor top-k off. Transcribed
    from `:151-182`, including the two `compute_norms` calls, which happen
    ONCE before the loop there (`:180-182`) and once before the loop here.

    ONE LANE, TWO METHODS, because they are two arms of ONE entry and the
    upstream file prices both and says why at `:19-43`:
    `MOJOLEARN_LANES_PRICE_KNN_METHOD` is `auto` (the default; the shipped
    dispatch, which under IDENTICAL is pinned to the tiled arm on every
    column by DEVIATION 509, so at `k <= 64` this arm prices a KERNEL SWAP)
    or `tiled` (the same arm asked for explicitly, which is also what
    `k > 64` must take). The method is part of the ledger's size field, so
    the two can never be averaged into one row by the driver script.

    Index rows come from `MOJOLEARN_LANES_PRICE_KNN_INDEX`, default 20000
    (the Apple size), 400000 the datacenter step -- a size
    `bench/scaling_main.mojo:180` already sweeps through this same entry
    with this same `buf_len` formula (`:50`), so it is a shape this tree has
    run and not one invented for a bigger number. The query count stays at
    the upstream 1000: the distance work is `n_queries * n_index` and the
    tiled arm processes 256 queries at a time, so the index axis is the one
    that grows the arm rather than the number of tiles.
    """
    var method_name = String(getenv("MOJOLEARN_LANES_PRICE_KNN_METHOD"))
    if method_name == "":
        method_name = String("auto")
    var method = KNN_METHOD_AUTO
    if method_name == "tiled":
        method = KNN_METHOD_TILED
    elif method_name != "auto":
        raise Error(
            "lanes_price: MOJOLEARN_LANES_PRICE_KNN_METHOD must be 'auto' or"
            " 'tiled'; got '" + method_name + "'"
        )

    var n_index = _env_int(
        "MOJOLEARN_LANES_PRICE_KNN_INDEX", 2048 if smoke else 20000
    )
    var n_queries = 1000
    var d = 32
    var k = 10
    var tile = 256
    var buf_len = max(n_index // 8, k)

    var index = ctx.enqueue_create_buffer[DType.float32](n_index * d)
    var queries = ctx.enqueue_create_buffer[DType.float32](n_queries * d)
    var inorm = ctx.enqueue_create_buffer[DType.float32](n_index)
    var qnorm = ctx.enqueue_create_buffer[DType.float32](n_queries)
    var dist = ctx.enqueue_create_buffer[DType.float32](tile * n_index)
    var bv = ctx.enqueue_create_buffer[DType.float32](tile * 2 * buf_len)
    var bi = ctx.enqueue_create_buffer[DType.uint32](tile * 2 * buf_len)
    var od = ctx.enqueue_create_buffer[DType.float32](n_queries * k)
    var oi = ctx.enqueue_create_buffer[DType.uint32](n_queries * k)
    var oi32 = ctx.enqueue_create_buffer[DType.int32](n_queries * k)
    var hi = ctx.enqueue_create_host_buffer[DType.float32](n_index * d)
    var hq = ctx.enqueue_create_host_buffer[DType.float32](n_queries * d)
    ctx.synchronize()
    for i in range(n_index):
        for f in range(d):
            hi.unsafe_ptr().unsafe_store(i * d + f, _price_u01(i, f, 3))
    for i in range(n_queries):
        for f in range(d):
            hq.unsafe_ptr().unsafe_store(i * d + f, _price_u01(i, f, 5))
    ctx.enqueue_copy(dst_buf=index, src_ptr=hi.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=queries, src_ptr=hq.unsafe_ptr())
    ctx.synchronize()
    compute_norms(ctx, index, inorm, n_index, d, False)
    compute_norms(ctx, queries, qnorm, n_queries, d, False)
    ctx.synchronize()

    var ledger = Ledger(
        "knn",
        method_name + "." + String(n_index) + "x" + String(n_queries) + "x"
        + String(d) + ".k" + String(k),
    )
    for r in range(rounds + 1):
        # Poisoned before the clock starts, this file's convention (see
        # `run_linkage`): if the entry ever left an output slot unwritten,
        # the poison would be in the hash instead of whatever the previous
        # round left there. The poison is the same value every round, so it
        # cannot be what a HASH-MOVED verdict is reporting.
        #
        # THE POISON IS LARGE AND POSITIVE, unlike the gemm lane's
        # `-987654.0`, and the sign is the point. This entry selects the k
        # SMALLEST distances, and its distances are non-negative, so a
        # positive poison LOSES every comparison it could take part in. A
        # negative one would win, and if any arm of this dispatch ever
        # merged into `out_dist` rather than overwriting it, the harness
        # would have quietly changed the answer it is timing. The index
        # poison is the largest `uint32` for the same reason: the tiled
        # arm's composite key breaks ties toward the LOWEST index
        # (DEVIATION 500), so this one loses there too.
        ctx.enqueue_memset(od, Float32(987654.0))
        ctx.enqueue_memset(oi, UInt32(0xFFFFFFFF))
        ctx.synchronize()
        var t0 = perf_counter_ns()
        brute_force_knn_impl(
            ctx, queries, qnorm, index, inorm, dist, bv, bi, od, oi, oi32,
            n_queries, n_index, d, k, tile, buf_len, True, False, True, True,
            method,
        )
        ctx.synchronize()
        var t1 = perf_counter_ns()
        var h = _hash_device_f32(ctx, FNV_OFFSET, od, n_queries * k)
        h = _hash_device_u32(ctx, h, oi, n_queries * k)
        ledger.emit("warmup" if r == 0 else String(r), t1 - t0, h)
    ledger.verdict()
    _ = hi^
    _ = hq^
    _ = index^
    _ = queries^
    _ = inorm^
    _ = qnorm^
    _ = dist^
    _ = bv^
    _ = bi^
    _ = od^
    _ = oi^
    _ = oi32^


def run_dbscan(ctx: DeviceContext, smoke: Bool, rounds: Int) raises:
    """`bench/identity_price_main.mojo:195`'s `dbscan.fit` arm: the
    brute-force eps neighbourhood plus the propagation, `eps` 1.2,
    `min_pts` 8, device-chosen batch, 200 iterations, `EPS_NN_BRUTE_FORCE`,
    on `Float32(i % 6) * 5 + _price_u01(i, f, 9)` rows at d 8. Fixture and
    call transcribed from `:196-217`.

    Rows come from `MOJOLEARN_LANES_PRICE_DBSCAN_ROWS`, default 20000 (the
    Apple size), 100000 the datacenter step. THE ROWS ARE THE EXPENSIVE
    AXIS HERE IN A WAY THEY ARE NOT IN THE OTHER LANES: the brute arm is
    quadratic in them, and `bench/scaling_main.mojo`'s
    `_dbscan_rbc_only_at` docstring puts brute at 800,000 near four and a
    half minutes per repeat on the laptop. The fixture keeps six clusters
    at a spacing of 5 at every size, so more rows means a denser blob and
    not a different problem; d stays at 8 for the same reason k stays at 16
    in the kmeans lane.
    """
    var n = _env_int(
        "MOJOLEARN_LANES_PRICE_DBSCAN_ROWS", 2048 if smoke else 20000
    )
    var d = 8
    var hx = ctx.enqueue_create_host_buffer[DType.float32](n * d)
    ctx.synchronize()
    for i in range(n):
        for f in range(d):
            hx.unsafe_ptr().unsafe_store(
                i * d + f, Float32(i % 6) * Float32(5.0) + _price_u01(i, f, 9)
            )
    var x = ctx.enqueue_create_buffer[DType.float32](n * d)
    var labels = ctx.enqueue_create_buffer[DType.int32](n)
    ctx.synchronize()
    ctx.enqueue_copy(dst_buf=x, src_ptr=hx.unsafe_ptr())
    ctx.synchronize()

    var ledger = Ledger("dbscan", String(n) + "x" + String(d))
    for r in range(rounds + 1):
        # `-7` is the linkage lane's poison and is not a label DBSCAN can
        # produce (`-1` is its noise marker), so an unwritten row is visible
        # in the hash rather than inherited from the round before.
        ctx.enqueue_memset(labels, Int32(-7))
        ctx.synchronize()
        var t0 = perf_counter_ns()
        var n_clusters = dbscan_fit_impl(
            ctx, x, labels, n, d, 1.2, 8, 0, 200, EPS_NN_BRUTE_FORCE, False
        )
        ctx.synchronize()
        var t1 = perf_counter_ns()
        var h = _hash_device_i32(ctx, FNV_OFFSET, labels, n)
        h = _fold_word(h, UInt64(n_clusters), 4)
        ledger.emit("warmup" if r == 0 else String(r), t1 - t0, h)
    ledger.verdict()
    _ = hx^
    _ = x^
    _ = labels^


# ============================================================================
# gram, nt, gemv -- the three arms of bench/linalg_price_main.mojo
# ============================================================================


def _linalg_val(i: Int, salt: Int) -> Float32:
    """TRANSCRIBED from `bench/linalg_price_main.mojo:76-77`: an integer in
    `[-1000000, 1000000]` scaled by 1e-6.

    `_gemm_mix` above IS that file's `_mix` (`:66-73`) -- the same splitmix64
    constants in the same order -- so it is reused rather than spelled a
    third time, and the three linalg fixtures below carry that file's bits.
    """
    var num = Int(_gemm_mix(i, salt) % 2000001) - 1000000
    return Float32(num) * Float32(1.0e-6)


def _fill_linalg(
    ctx: DeviceContext, mut buf: DeviceBuffer[DType.float32], n: Int, salt: Int
) raises:
    """`bench/linalg_price_main.mojo:80-87`'s `_fill`, with ONE change.

    That file lets its staging host buffer die at its last use, which is the
    `enqueue_copy` -- and the copy is enqueued, not finished, so the buffer
    can be freed while the DMA is still reading it
    (`[[mojo-buffer-freed-at-last-use]]`). Held past the `synchronize` here,
    which is what `_hash_device_f32` above already does with its own staging
    buffer. Nothing else about the fill moves.
    """
    var h = ctx.enqueue_create_host_buffer[DType.float32](n)
    ctx.synchronize()
    for i in range(n):
        h.unsafe_ptr().unsafe_store(i, _linalg_val(i, salt))
    ctx.enqueue_copy(dst_buf=buf, src_ptr=h.unsafe_ptr())
    ctx.synchronize()
    _ = h^


def run_gram(ctx: DeviceContext, smoke: Bool, rounds: Int) raises:
    """`bench/linalg_price_main.mojo:94`'s `gram.32x32x1M` arm: `gemm_tn` at
    the shipped PCA/OLS Gram aspect, 32 features by however many rows.
    Transcribed from `:96-102`, salt 11.

    WHAT THIS ARM PRICES IS NOT THE SAME THING ON EVERY COLUMN, and the
    upstream docstring's reading is the APPLE one. On Apple both modes take
    `core/gram_splitk.mojo`'s kernel and IDENTICAL pins its partition, so
    the arm prices a PIN: same code, two partitions. On NVIDIA and AMD
    `gram_splitk_applies` returns False at EVERY shape under FAST -- its
    first test is a comptime one, and that function's docstring gives the
    reason (MAX's own split-K is reachable off Apple, so the vendor rule
    resumes) -- while under IDENTICAL the same predicate answers True
    wherever the kernel's own capacity allows, which 32 x 32 does. So off
    Apple this arm prices a REPLACEMENT of the vendor matmul by the pinned
    kernel. Both are real prices of identity; they are not the same number
    and must not be read as one.

    Rows come from `MOJOLEARN_LANES_PRICE_GRAM_ROWS`, default 1000000 (the
    Apple size, the `32x32x1M` in the upstream arm's name), 8000000 the
    datacenter step. m and n stay 32: `gram_splitk_applies` REFUSES under
    IDENTICAL above `GRAM_MAX_COLS` or the register budget, and 32 is the
    shipped aspect, so the row count is the only axis that may move. The
    memory is 384 bytes per row across `x`, `xt` and `xt2`, so the
    datacenter step wants about 3.1 GB of device memory for this lane.
    """
    var m = 32
    var k = _env_int(
        "MOJOLEARN_LANES_PRICE_GRAM_ROWS", 4096 if smoke else 1000000
    )
    var x = ctx.enqueue_create_buffer[DType.float32](k * m)
    var z = ctx.enqueue_create_buffer[DType.float32](m * m)
    var xt = ctx.enqueue_create_buffer[DType.float32](k * m)
    var xt2 = ctx.enqueue_create_buffer[DType.float32](k * m)
    _fill_linalg(ctx, x, k * m, 11)

    var ledger = Ledger("gram", String(m) + "x" + String(m) + "x" + String(k))
    for r in range(rounds + 1):
        ctx.enqueue_memset(z, Float32(-987654.0))
        ctx.synchronize()
        var t0 = perf_counter_ns()
        gemm_tn(ctx, z, x, xt, xt2, m, m, k)
        ctx.synchronize()
        var t1 = perf_counter_ns()
        var h = _hash_device_f32(ctx, FNV_OFFSET, z, m * m)
        ledger.emit("warmup" if r == 0 else String(r), t1 - t0, h)
    ledger.verdict()
    _ = x^
    _ = z^
    _ = xt^
    _ = xt2^


def run_nt(ctx: DeviceContext, smoke: Bool, rounds: Int) raises:
    """`bench/linalg_price_main.mojo:124`'s `nt.4096x64x64` arm: `gemm_nt`,
    the N-T product, at n 64 and k 64. Transcribed from `:126-133`, salts 21
    and 22.

    THE EXPENSIVE ONE BY CONSTRUCTION, and the upstream docstring says why:
    under FAST this is MAX's tuned matmul and under IDENTICAL it is
    `pinned_gemm_nt_kernel`, one thread per output cell (DEVIATION 526). So
    the arm prices the REPLACEMENT of a closed vendor library rather than a
    change of rounding.

    Rows come from `MOJOLEARN_LANES_PRICE_NT_ROWS`, default 4096 (the Apple
    size, the `4096` in the upstream arm's name), 262144 the datacenter
    step. n and k stay 64 because 64 is the feature width of the k-NN
    distance step this shape comes from, and because the pinned kernel's
    thread count is `m * n`: the row axis is what gives a datacenter GPU
    enough cells to fill itself.
    """
    var m = _env_int("MOJOLEARN_LANES_PRICE_NT_ROWS", 256 if smoke else 4096)
    var n = 64
    var k = 64
    var x = ctx.enqueue_create_buffer[DType.float32](m * k)
    var y = ctx.enqueue_create_buffer[DType.float32](n * k)
    var z = ctx.enqueue_create_buffer[DType.float32](m * n)
    _fill_linalg(ctx, x, m * k, 21)
    _fill_linalg(ctx, y, n * k, 22)

    var ledger = Ledger("nt", String(m) + "x" + String(n) + "x" + String(k))
    for r in range(rounds + 1):
        ctx.enqueue_memset(z, Float32(-987654.0))
        ctx.synchronize()
        var t0 = perf_counter_ns()
        gemm_nt(ctx, z, x, y, m, n, k)
        ctx.synchronize()
        var t1 = perf_counter_ns()
        var h = _hash_device_f32(ctx, FNV_OFFSET, z, m * n)
        ledger.emit("warmup" if r == 0 else String(r), t1 - t0, h)
    ledger.verdict()
    _ = x^
    _ = y^
    _ = z^


def run_gemv(ctx: DeviceContext, smoke: Bool, rounds: Int) raises:
    """`bench/linalg_price_main.mojo:151`'s `gemv.128x128` arm: `gemv_n`,
    OLS's step 6, on a square `m x k`. Transcribed from `:153-159`, salts 31
    and 32.

    Priced although it is tiny, and the upstream docstring gives the reason
    to keep it that way: "it is small so it cannot matter" is an argument,
    and this harness exists to replace arguments with seconds. Under
    IDENTICAL the arm is `pinned_gemv_n_kernel`, which gives each output row
    to ONE thread so k is walked ascending with no cross-lane fold at all
    (`core/gemm.mojo:794-802`, DEVIATION 526).

    The square dimension comes from `MOJOLEARN_LANES_PRICE_GEMV_DIM`,
    default 128 (the Apple size, the `128x128` in the upstream arm's name),
    8192 the datacenter step. ONE knob sets both m and k because the call
    site is square -- `w <- covA Ab` at the feature count -- and because the
    two axes buy different things in the pinned arm: m is its whole thread
    count and k is the length of each thread's walk, so moving only one of
    them prices half the kernel.
    """
    var dim = _env_int("MOJOLEARN_LANES_PRICE_GEMV_DIM", 32 if smoke else 128)
    var m = dim
    var k = dim
    var x = ctx.enqueue_create_buffer[DType.float32](m * k)
    var y = ctx.enqueue_create_buffer[DType.float32](k)
    var z = ctx.enqueue_create_buffer[DType.float32](m)
    _fill_linalg(ctx, x, m * k, 31)
    _fill_linalg(ctx, y, k, 32)

    var ledger = Ledger("gemv", String(m) + "x" + String(k))
    for r in range(rounds + 1):
        ctx.enqueue_memset(z, Float32(-987654.0))
        ctx.synchronize()
        var t0 = perf_counter_ns()
        gemv_n(ctx, z, x, y, m, k)
        ctx.synchronize()
        var t1 = perf_counter_ns()
        var h = _hash_device_f32(ctx, FNV_OFFSET, z, m)
        ledger.emit("warmup" if r == 0 else String(r), t1 - t0, h)
    ledger.verdict()
    _ = x^
    _ = y^
    _ = z^


# ============================================================================
# gbdt, rf, et -- the three tree families, added 2026-09-02
# ============================================================================

#: `fit_forest`'s objective, spelled exactly as
#: `ensemble/bench/rf_bench.mojo:57` spells it (`comptime ClsObj =
#: ClassificationObjectiveFunction[DT, LT, ClassificationBin]` at DT float32,
#: LT int32). The forest loop is generic over the objective, so this alias is
#: what picks the CLASSIFICATION arm.
comptime RF_CLS_OBJ = ClassificationObjectiveFunction[
    DType.float32, DType.int32, ClassificationBin
]


def _tree_hits3(row: Int, salt: Int) -> Int:
    """How many of the first three features of `row` sit above their midpoint.

    THE LABEL IS AXIS-ALIGNED ON PURPOSE, and the argument is
    `ensemble/bench/rf_bench.mojo:277-289`'s, which is worth repeating here
    because it is a TIMING argument and not only an accuracy one. That file's
    first fixture was twelve diagonal bands; held-out accuracy came out at the
    class baseline for BOTH arms, so neither implementation learned anything --
    and the two arms were building trees of very different SIZES while learning
    nothing. A price is a ratio of two fits, so a fixture with no signal does
    not merely make the accuracy vacuous, it lets the two modes build different
    amounts of tree and then reports the difference as a price. A rule three
    levels of any of these trees can represent exactly cannot do that.

    `_price_u01` is a pure function of `(row, feature, salt)`, so recomputing
    the three columns here gives the SAME BITS the fill loop stored, with no
    second pass over the fixture.
    """
    var hits = 0
    for f in range(3):
        if _price_u01(row, f, salt) >= Float32(0.5):
            hits += 1
    return hits


def run_gbdt(ctx: DeviceContext, smoke: Bool, rounds: Int) raises:
    """`checks/estimation_bench.mojo:76-81`'s fit: `gbdt.train.train` at
    `border_count` 254, `learning_rate` 0.1, `l2_leaf_reg` 1.0, Logloss with
    `leaf_estimation_iterations` 10 (their `catboost_options.cpp:157-164`
    default), depth 6, on 100 raw float32 columns. `grow_policy` is passed
    EXPLICITLY as `SymmetricTree` although that is already the default
    (`gbdt/train.mojo:511`): this lane is named for the growth policy, and a
    lane whose identity depends on a default nobody restates is a lane that
    silently changes meaning when the default does.

    WHY 100 FEATURES AT DEPTH 6 AND NOT SOMETHING ROUNDER. That is the shape
    the symmetric lane's own 1M window ran (`N_FEATURES = 100`,
    `checks/estimation_bench.mojo:60`, with `N_ROWS` patched to 1000000 and
    then 2000000 -- `bench/results/ab_large_2026-09-01/RESULTS.md:10-13`), and
    the d6 cells there are the only per-tree symmetric costs this tree has
    MEASURED at the floor. The cell this lane's arm actually sits in is d6
    ll10 -- 273.2 and 257.5 ms/tree at HEAD, `RESULTS.md:57` -- which at the
    default 20 trees predicts a fit near 5.3 s on the M4, in the band this
    harness wants. The d6 RMSE cell beside it (170.3 / 166.4, `:56`) is the
    same fit without the estimation stage, quoted only so a reader can see how
    much of this lane's seconds the Newton walker is.

    WHAT THE ROUND INCLUDES, said plainly because it dilutes the ratio.
    `train` is the RAW-FLOATS door: it builds the borders and quantizes into
    the compressed index before the boosting loop starts
    (`_build_cindex_from_floats`, `gbdt/train.mojo:155`), and all of that is
    inside the clock. It is inside the clock in BOTH modes, so the comparison
    is still like for like, but the identical pins live in the boosting loop
    and the quantization is a constant added to both arms, which pulls the
    ratio toward 1.0. The tighter arm exists -- `fit`
    (`gbdt/methods/doc_parallel_boosting.mojo:2369`) takes an already-built
    compressed index -- and is the lane to add if this row turns out not to
    separate. It is NOT taken here because building that index means
    transcribing `build_layout` plus the binarize kernel launch
    (`bench/interleaved/catboost_interleaved.mojo:56-90`) into a harness that
    is supposed to mirror call sites rather than re-spell them.

    Rows come from `MOJOLEARN_LANES_PRICE_GBDT_ROWS`, default 1000000, with
    2000000 the datacenter step; trees from `MOJOLEARN_LANES_PRICE_GBDT_TREES`,
    default 20, 60 the datacenter step. THE ROW DEFAULT IS A FLOOR AND NOT A
    TUNING CHOICE: [[mojolearn-large-data-timing-floor]] forbids measuring or
    deciding tree performance below 1,000,000 rows, so the 2048- and
    20000-row defaults the kmeans, knn and dbscan lanes above use would be
    inadmissible here even though they are the published sizes there. A
    50,000-row symmetric window was caught and killed on 2026-09-01 for
    exactly this and nothing from it votes anywhere (`RESULTS.md:3-7`). The
    TREE count is the axis that was moved to fit the fit into a few seconds
    instead, because a boosted round is very nearly a fixed cost repeated --
    the per-tree ms/tree figure quoted above is what makes that a subtraction
    the lane already trusts -- whereas the row count changes what the kernels
    are doing.

    MEMORY. The fixture is a HOST `List[Float32]` of `n * 100` because that is
    what `train` takes: 400 MB at the default and 800 MB at the datacenter
    step, before the device copy of the compressed index. The list is
    PRE-SIZED and index-assigned rather than appended to, and filled
    feature-major so the stores are contiguous; `checks/estimation_bench.mojo`
    appends, which is fine at 50,000 rows and is not fine at a million.
    """
    var n = _env_int(
        "MOJOLEARN_LANES_PRICE_GBDT_ROWS", 4096 if smoke else 1000000
    )
    var trees = _env_int(
        "MOJOLEARN_LANES_PRICE_GBDT_TREES", 2 if smoke else 20
    )
    var d = 100
    var depth = 6

    var x = List[Float32](length=n * d, fill=Float32(0.0))
    var y = List[Float32](length=n, fill=Float32(0.0))
    for f in range(d):
        for i in range(n):
            x[f * n + i] = _price_u01(i, f, 41)
    for i in range(n):
        y[i] = Float32(1.0) if _tree_hits3(i, 41) >= 2 else Float32(0.0)

    var ledger = Ledger(
        "gbdt",
        String(n) + "x" + String(d) + ".t" + String(trees) + ".d"
        + String(depth),
    )
    for r in range(rounds + 1):
        # Nothing to re-initialize before the clock. `train` takes its two
        # host lists by value and returns a FRESH `TrainedModel` every call,
        # so a round cannot continue the round before it the way the cd lane's
        # `coef` or the kmeans lane's `cent` can. If that ever stops being
        # true the in-process verdict below is what says so.
        var t0 = perf_counter_ns()
        var tm = gbdt_train(
            ctx, x, y, n, d,
            border_count=254, n_estimators=trees, max_depth=depth,
            learning_rate=Float32(0.1), l2_leaf_reg=Float32(1.0),
            loss=String("Logloss"), leaf_estimation_iterations=10,
            grow_policy=String("SymmetricTree"),
        )
        ctx.synchronize()
        var t1 = perf_counter_ns()
        # THE MODEL, NOT A SUMMARY. An oblivious tree IS its split list plus
        # its leaf vector (`gbdt/models/oblivious_model.mojo:113-141`), and
        # both are host lists by the time `train` returns, so the whole
        # trained state is reachable with no device readback at all. The
        # per-iteration learn curve goes in after it because a fit that
        # produced the same trees by a different route is a finding this lane
        # should print rather than hide.
        var h = _fold_f64(FNV_OFFSET, tm.model.bias)
        h = _fold_word(h, UInt64(len(tm.model.weak_models)), 4)
        for t in range(len(tm.model.weak_models)):
            for s in range(len(tm.model.weak_models[t].structure.splits)):
                h = _fold_i32(
                    h, tm.model.weak_models[t].structure.splits[s].feature_id
                )
                h = _fold_i32(
                    h, tm.model.weak_models[t].structure.splits[s].bin_idx
                )
                h = _fold_i32(
                    h, tm.model.weak_models[t].structure.splits[s].split_type
                )
            for v in range(len(tm.model.weak_models[t].leaf_values)):
                h = _fold_f32(h, tm.model.weak_models[t].leaf_values[v])
        for i in range(len(tm.losses)):
            h = _fold_f64(h, tm.losses[i])
        ledger.emit("warmup" if r == 0 else String(r), t1 - t0, h)
        _ = tm^
    ledger.verdict()
    _ = x^
    _ = y^


def run_rf(ctx: DeviceContext, smoke: Bool, rounds: Int) raises:
    """`ensemble/bench/rf_bench.mojo:319-321`'s fit: `fit_forest[ClsObj]` with
    that file's `_params` block (`:107-129`) -- bootstrap on, `max_samples`
    1.0, seed 20260821, `n_streams` 4, depth 12, `max_features` 1.0,
    `max_n_bins` 128, gini, `max_batch_size` 4096 -- on 50 raw float32
    columns and four classes, column-major on the device.

    `n_streams` STAYS AT 4 AND IS NOT A KNOB. It is their Python default
    (`randomforestclassifier.py:94`) and the pipelined forest loop's output is
    bit-identical at any value since DEVIATION 117, so moving it would change
    the seconds without changing the answer -- which is the one thing a price
    lane must not let a size knob do.

    Rows come from `MOJOLEARN_LANES_PRICE_RF_ROWS`, default 1000000, with
    2000000 the datacenter step; trees from `MOJOLEARN_LANES_PRICE_RF_TREES`,
    default 4, 16 the datacenter step. The 1,000,000 floor is
    [[mojolearn-large-data-timing-floor]]'s and is argued in `run_gbdt` above;
    the in-tree arms of `rf_bench.mojo` are `rf@100000` and `rf@500000`
    (`:661`, `:663`) and BOTH are below it, which is why this lane does not
    inherit its row count from them. It inherits the 1M/2M shape the RF lane
    actually measured at, which was a harness-only patch of that same file
    (`bench/results/ab_large_2026-09-01/RESULTS.md:17-19`).

    WHY FOUR TREES WHEN THAT WINDOW RAN THIRTY. Because that window's fits
    took 22.3 to 26.4 SECONDS each (`RESULTS.md:28-35`, 30 trees) and
    `tools/lanes_price.sh` at its default ROUNDS runs twenty fits (five legs
    per mode, a warm-up and a timed round in each), which is eight minutes of
    fitting in this one lane before either build. A forest is `n_trees`
    independent trees with one shared quantile pass in front of them, so the
    tree count is very nearly a repetition axis: four of them still take about
    3 s at 1,000,000 rows on the M4 by that same window's per-tree figure, and
    the whole lane comes back inside a minute. The ROWS are what may not be
    cut, and they are not cut.
    """
    var n = _env_int(
        "MOJOLEARN_LANES_PRICE_RF_ROWS", 4096 if smoke else 1000000
    )
    var trees = _env_int("MOJOLEARN_LANES_PRICE_RF_TREES", 2 if smoke else 4)
    var d = 50
    var n_classes = 4
    var depth = 12
    var n_bins = 128

    var hx = ctx.enqueue_create_host_buffer[DType.float32](n * d)
    var hy = ctx.enqueue_create_host_buffer[DType.int32](n)
    ctx.synchronize()
    # Filled FEATURE-MAJOR although the layout is column-major either way:
    # `rf_bench.mojo` walks rows outside and columns inside, which strides
    # every store across a 200 MB buffer. `_price_u01(i, f, .)` does not care
    # which order it is asked in, so the bits are the same and the fill is
    # contiguous.
    for f in range(d):
        for i in range(n):
            hx.unsafe_ptr().unsafe_store(f * n + i, _price_u01(i, f, 43))
    for i in range(n):
        hy.unsafe_ptr().unsafe_store(i, Int32(_tree_hits3(i, 43)))

    var dx = ctx.enqueue_create_buffer[DType.float32](n * d)
    var dy = ctx.enqueue_create_buffer[DType.int32](n)
    # A ONE-ELEMENT PLACEHOLDER, which is what the entry wants. Unweighted
    # fits read their weights from `sample_weight_host` and never from this
    # buffer; `rf_bench.mojo:312` allocates exactly one element for the same
    # reason, and the binding does the same and says so
    # (`bindings/_mojolearn_rf.mojo:322-325`).
    var dsw = ctx.enqueue_create_buffer[DType.float32](1)
    ctx.enqueue_copy(dst_buf=dx, src_ptr=hx.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=dy, src_ptr=hy.unsafe_ptr())
    ctx.synchronize()

    var ledger = Ledger(
        "rf",
        String(n) + "x" + String(d) + ".t" + String(trees) + ".d"
        + String(depth) + ".b" + String(n_bins),
    )
    for r in range(rounds + 1):
        # Rebuilt every round rather than hoisted, because `fit_forest` takes
        # it `mut` (`ensemble/randomforest.mojo:2699`) and a round that
        # inherited a mutated parameter block would be a different fit wearing
        # the same label. This is the same reasoning the kmeans lane applies
        # to its `cent` seed, and it costs nothing here: the block is a
        # handful of scalars built on the host outside the clock.
        var p = RF_params(
            n_trees=Int32(trees),
            bootstrap=True,
            max_samples=Float32(1.0),
            seed=UInt64(20260821),
            n_streams=Int32(4),
            tree_params=RfTreeParams(
                max_depth=Int32(depth),
                max_leaves=Int32(-1),
                max_features=Float32(1.0),
                max_n_bins=Int32(n_bins),
                min_samples_leaf=Int32(1),
                min_samples_split=Int32(2),
                split_criterion=RF_GINI,
                min_impurity_decrease=Float32(0.0),
                max_batch_size=Int32(4096),
            ),
        )
        var t0 = perf_counter_ns()
        var forest = fit_forest[RF_CLS_OBJ](
            ctx, dx, dy, dsw, n, d, n_classes, p
        )
        ctx.synchronize()
        var t1 = perf_counter_ns()
        # DEVIATION 401's two identity checkpoints per tree, in its field
        # order (`ensemble/randomforest.mojo:2645-2687`): the STRUCTURE
        # (column, threshold bits, metric bits, left child, instance count)
        # and then `vector_leaf`. Field by field and never as raw struct
        # bytes, because the struct has padding; that is identity_trace's
        # rule and the reason the fold is spelled out rather than memcpy'd.
        # Every one of those is a HOST list by the time the fit returns, so
        # this whole hash costs no device readback.
        #
        # `train_time` IS DELIBERATELY NOT IN IT. It is a member of this
        # lane's `TreeMetaDataNode` (`ensemble/decisiontree/decisiontree.mojo`)
        # and it is a WALL CLOCK, so folding it in would make every round
        # disagree with every other round and the harness would report
        # HASH-MOVED -- a false contract violation under IDENTICAL, raised
        # by the harness's own instrument rather than by the arm.
        var h = FNV_OFFSET
        for t in range(len(forest.trees)):
            h = _fold_i32(h, forest.trees[t].depth_counter)
            h = _fold_i32(h, forest.trees[t].leaf_counter)
            h = _fold_word(h, UInt64(len(forest.trees[t].sparsetree)), 4)
            for i in range(len(forest.trees[t].sparsetree)):
                h = _fold_i32(h, forest.trees[t].sparsetree[i].ColumnId())
                h = _fold_f32(h, forest.trees[t].sparsetree[i].QueryValue())
                h = _fold_f32(h, forest.trees[t].sparsetree[i].BestMetric())
                # `LeftChildId()` hands back an Int64 and -1 IS the leaf test,
                # so the widen sign-extends to all ones
                # (`[[mojo-int-widening-sign-extends]]`). Masked to 32 bits
                # AFTER the widen, which is what `_fold_i32` does internally
                # and what `_record_tree` does at its own call site.
                h = _fold_word(
                    h,
                    UInt64(Int(forest.trees[t].sparsetree[i].LeftChildId()))
                    & UInt64(0xFFFFFFFF),
                    4,
                )
                h = _fold_i32(h, forest.trees[t].sparsetree[i].InstanceCount())
            for i in range(len(forest.trees[t].vector_leaf)):
                h = _fold_f32(h, forest.trees[t].vector_leaf[i])
        ledger.emit("warmup" if r == 0 else String(r), t1 - t0, h)
        _ = forest^
    ledger.verdict()
    _ = hx^
    _ = hy^
    _ = dx^
    _ = dy^
    _ = dsw^


def run_et(ctx: DeviceContext, smoke: Bool, rounds: Int) raises:
    """`extratrees/bench/fit_once.mojo:153-155`'s fit:
    `train_forest_classification_device` on a `DeviceDataset` uploaded ONCE
    before the loop (DEVIATION 184 is the reason that upload is a separate
    entry at all), `max_features` sqrt, seed 0x0F17, depth 12, 28 columns, two
    classes.

    THE UPLOAD IS OUTSIDE THE CLOCK AND THAT IS DELIBERATE. The estimator door
    `fit_extra_trees_classifier_device` (`extratrees/estimator.mojo:529`) is
    the shipped public entry, but it uploads the matrix on every call, so a
    round taken through it would time a 112 MB host-to-device copy alongside
    the fit in both modes. `fit_once.mojo` is also the driver the lane's own
    1M and 2M runs went through, so this is the call whose seconds the tree
    already has numbers for.

    Rows come from `MOJOLEARN_LANES_PRICE_ET_ROWS`, default 1000000, with
    2000000 the datacenter step; trees from `MOJOLEARN_LANES_PRICE_ET_TREES`,
    default 20, 60 the datacenter step. The floor is
    [[mojolearn-large-data-timing-floor]]'s 1,000,000 rows, argued in
    `run_gbdt` above. UNLIKE the other two tree lanes this one keeps the tree
    count its measured window used: 28 features, 20 trees, depth 12, sqrt at
    1M and 2M rows is exactly the ET arm of
    `bench/results/ab_large_2026-09-01/RESULTS.md:14-16`, and it came out at
    5.9 to 6.3 s per fit at 1M and 12.1 to 12.2 s at 2M (`:74-75`). Nothing
    had to be dialed back to reach a few seconds, so nothing was.

    The published ET profile shape is 100 trees at depth 16 (11.8 s per fit on
    the M4, `extratrees/DEVIATIONS.md:4182`) and it is NOT the default here.
    `tools/lanes_price.sh` at its default ROUNDS runs five legs per mode with
    a warm-up and a timed round in each, which is twenty fits: about four
    minutes of fitting at the profile shape against about two at this one, for
    the same kernels run fewer times. Two minutes is worth having and four is
    not worth spending, and neither number is the reason the rows stay at a
    million.
    """
    var n = _env_int(
        "MOJOLEARN_LANES_PRICE_ET_ROWS", 4096 if smoke else 1000000
    )
    var trees = _env_int("MOJOLEARN_LANES_PRICE_ET_TREES", 2 if smoke else 20)
    var d = 28
    var n_classes = 2
    var depth = 12

    var xcol = List[Float32](length=n * d, fill=Float32(0.0))
    var lf = List[Float32](length=n, fill=Float32(0.0))
    for f in range(d):
        for i in range(n):
            xcol[f * n + i] = _price_u01(i, f, 47)
    for i in range(n):
        lf[i] = Float32(1.0) if _tree_hits3(i, 47) >= 2 else Float32(0.0)
    # `class_ids_for` is the hoisted float-to-int truncation the device
    # trainer requires (DEVIATION 186); `fit_once.mojo:113` calls it in the
    # same place, before the upload and outside every clock.
    var ids = class_ids_for(lf, Int32(n), Int32(n_classes))
    var dev = upload_dataset(
        ctx, xcol, ids, Int32(n), Int32(d), Int32(n_classes)
    )

    var params = EtTreeParams()
    params.max_depth = Int32(depth)
    # `fit_once.mojo:122-125`, with `max_features_code("sqrt", n_feat)` folded
    # to the constant it returns (`extratrees/bench/bench_data.mojo:153-154`)
    # so this file does not import a helper whose only job is parsing an
    # argv string. The ratio, not the count, is what the struct holds, and
    # `count_to_ratio` is what lands it in the middle of the bucket that
    # truncates back (`extratrees/estimator.mojo:204-222`).
    params.max_features = count_to_ratio(
        resolve_max_features(MAX_FEATURES_SQRT, 0.0, d), d
    )
    var tree_ids = List[Int32]()
    for t in range(trees):
        tree_ids.append(Int32(t))
    var seed = UInt64(0x0F17)

    var ledger = Ledger(
        "et",
        String(n) + "x" + String(d) + ".t" + String(trees) + ".d"
        + String(depth) + ".sqrt",
    )
    for r in range(rounds + 1):
        var t0 = perf_counter_ns()
        # `dataset` is taken `mut` (`builder.mojo:3570`) even though the
        # matrix is immutable, so the dataset is NOT re-uploaded between
        # rounds and every round therefore reads whatever the round before it
        # left. That is the same bet the kmeans lane takes on `x` and `w`, and
        # the same witness catches it: if a round mutated the data the
        # in-process verdict below would print HASH-MOVED, which is exactly
        # how the cd lane's in-place mutation was found.
        var forest = train_forest_classification_device(
            ctx, dev, params, tree_ids, seed
        )
        ctx.synchronize()
        var t1 = perf_counter_ns()
        # The SAME fold as the rf lane's, in the same field order, and it is
        # written out a second time rather than shared because the two lanes'
        # `TreeMetaDataNode` and `SparseTreeNode` are DIFFERENT STRUCTS in
        # different packages (`extratrees/impl/decisiontree/flatnode.mojo:313`
        # against `ensemble/decisiontree/decisiontree.mojo:424`) that happen
        # to agree on their accessors. A shared helper would have to be
        # generic over both, which would assert a relationship between two
        # ports that this file has no business asserting.
        #
        # This lane's node type has NO `train_time`, so the rf lane's
        # exclusion has nothing to exclude here; the field list is otherwise
        # identical and is `extratrees/bench/batched_ab.mojo:52-68`'s
        # `forest_digest` plus the metric.
        var h = FNV_OFFSET
        for t in range(len(forest)):
            h = _fold_i32(h, forest[t].depth_counter)
            h = _fold_i32(h, forest[t].leaf_counter)
            h = _fold_word(h, UInt64(forest[t].num_nodes()), 4)
            for i in range(forest[t].num_nodes()):
                h = _fold_i32(h, forest[t].sparsetree[i].ColumnId())
                h = _fold_f32(h, forest[t].sparsetree[i].QueryValue())
                h = _fold_f32(h, forest[t].sparsetree[i].BestMetric())
                h = _fold_word(
                    h,
                    UInt64(Int(forest[t].sparsetree[i].LeftChildId()))
                    & UInt64(0xFFFFFFFF),
                    4,
                )
                h = _fold_i32(h, forest[t].sparsetree[i].InstanceCount())
            for i in range(len(forest[t].vector_leaf)):
                h = _fold_f32(h, forest[t].vector_leaf[i])
        ledger.emit("warmup" if r == 0 else String(r), t1 - t0, h)
        _ = forest^
    ledger.verdict()
    _ = xcol^
    _ = lf^
    _ = ids^
    _ = tree_ids^
    _ = dev^


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
    elif lane == "kmeans":
        run_kmeans(ctx, smoke, rounds)
    elif lane == "knn":
        run_knn(ctx, smoke, rounds)
    elif lane == "dbscan":
        run_dbscan(ctx, smoke, rounds)
    elif lane == "gram":
        run_gram(ctx, smoke, rounds)
    elif lane == "nt":
        run_nt(ctx, smoke, rounds)
    elif lane == "gemv":
        run_gemv(ctx, smoke, rounds)
    elif lane == "gbdt":
        run_gbdt(ctx, smoke, rounds)
    elif lane == "rf":
        run_rf(ctx, smoke, rounds)
    elif lane == "et":
        run_et(ctx, smoke, rounds)
    else:
        raise Error(
            "lanes_price: MOJOLEARN_LANES_PRICE_LANE must be one of"
            " cd kde linkage svm metrics gemm kmeans knn dbscan gram nt gemv"
            " gbdt rf et; got '" + lane + "'"
        )
    print("== done [" + mode + "] lane=" + lane + " ==")
