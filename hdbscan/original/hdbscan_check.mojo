# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The HDBSCAN gates.

    tools/with_build_lock.sh     pixi run mojo run -I . hdbscan/original/hdbscan_check.mojo
    tools/with_identical_mode.sh pixi run mojo run -I . hdbscan/original/hdbscan_check.mojo

Every check prints the mode this binary COMPILED in. Under IDENTICAL the
distance-derived gates are ASSERTIONS (the device must equal the host
pinned arithmetic byte for byte); under FAST the same comparisons are
REPORTS, because the FAST arm's distances come from a vendor matmul and
no host can replicate that -- `hierarchy/original/linkage_check.mojo`
makes the same split for the same reason, and `UNSUPERVISED_IDENTITY.md`
records the measurement behind it (three consecutive FAST runs of one
binary on one device returned three different sorted k-NN index sets).

The gates that hold in BOTH modes are the ones stated over whatever the
device produced: given THESE core distances, the mutual reachability is
the three-way max; given THESE mutual reachability bytes, the MST is the
lexicographic one, the condensed tree follows from the dendrogram, the
stabilities follow from the condensed tree, the labels follow from the
selection.

CHECKS
  check_hdbscan_refusals                every unported arm, every
                                        out-of-range parameter, every
                                        non-finite input, BY NAME
  check_core_distances_vs_oracle        per cell vs the host oracle's
                                        k-th order statistic (a full row
                                        sort, not a top-k)
  check_mutual_reachability_ties        the duplicate fixture: the
                                        three-way max resolves on the
                                        pinned order, `mr` is exactly
                                        SYMMETRIC, a REVERSED input order
                                        gives the same EDGE SET, and a
                                        PLANTED `-0.0` core distance
                                        resolves to `+0.0`
  check_condensed_tree_vs_oracle        node for node, plus the stage hash
  check_stabilities_vs_oracle           per cluster, bit for bit
  check_labels_vs_oracle                labels vs the oracle AND vs the
                                        PLANTED assignment; the OUTLIER
                                        count separately
  check_permutation_invariance          same points, different ROW ORDER,
                                        same PARTITION and same NOISE SET
  check_launch_invariance               two block sizes per kernel, input
                                        padding, output poison, and the
                                        same cells computed alone vs in a
                                        batch
  check_card_is_emitted                 the stage list, and two cards from
                                        two launch shapes record-identical
  check_hdbscan_signed_zero_inputs      a `-0.0` COORDINATE moves no bit
  check_hdbscan_selection_leaf          the OTHER side of the selection
                                        switch (PORTING_RULES rule 8)
  check_hdbscan_float64_reference       the MST total against a Float64
                                        direct-form mutual reachability MST
  check_hdbscan_sabotages               the table below

SABOTAGE TABLE (results are copied into hdbscan/README.md)
  HDB_SAB_MR_TWO_WAY          dups     MUST FAIL the mutual-reachability
                                       gate (mr stops being symmetric)
  HDB_SAB_HW_MAX              planted  RECORDED: the stdlib max's answer
                                       on a (+0, -0) pair is the vendor's
  HDB_SAB_CORE_KTH_PLUS_ONE   blobs    MUST FAIL the core-distance gate
  HDB_SAB_CONDENSE_DFS        blobs    MUST FAIL the condensed-tree gate
  HDB_SAB_STABILITY_DESCENDING blobs   RECORDED with the moved-cell count
  HDB_SAB_EOM_NO_UPDATE       gradient MUST FAIL the selection gate
  HDB_SAB_SKIP_GUARDS         planted  RECORDED: what the guard keeps out
  HDB_SAB_LAMBDA_STD_DIV      blobs    RECORDED (Apple's divide is
                                       correctly rounded)
"""

from std.memory import bitcast

from max.gpu.host import DeviceBuffer, DeviceContext, HostBuffer

from core.identity_trace import IdentityTrace, first_divergence
from hdbscan.original.hdbscan_fixture import (
    HFIX_BLOBS,
    HFIX_COUNT,
    HFIX_DUPS,
    HFIX_GRADIENT,
    HFIX_OUTLIER,
    HFIX_POS_ZERO,
    HFIX_SIGNED_ZERO,
    NEG_ZERO_BITS,
    hfixture_as_list,
    hfixture_d,
    hfixture_min_cluster_size,
    hfixture_min_samples,
    hfixture_n,
    hfixture_n_planted_clusters,
    hfixture_name,
    hfixture_permutation,
    hfixture_permuted_as_list,
    hfixture_planted_label,
)
from hdbscan.original.hdbscan_oracle import (
    OracleRun,
    oracle_leaf_selection,
    oracle_run,
)
from hdbscan.original.hdbscan_sabotage import (
    HDB_SAB_CONDENSE_DFS,
    HDB_SAB_CORE_KTH_PLUS_ONE,
    HDB_SAB_EOM_NO_UPDATE,
    HDB_SAB_HW_MAX,
    HDB_SAB_LAMBDA_STD_DIV,
    HDB_SAB_MR_TWO_WAY,
    HDB_SAB_NONE,
    HDB_SAB_SKIP_GUARDS,
    HDB_SAB_STABILITY_DESCENDING,
)
from hdbscan.original.mutual_reachability_dense import (
    MR_TPB,
    mutual_reachability_dense,
)
from hdbscan.derived.hdbscan.detail.reachability import (
    CORE_TPB,
    compute_core_dists,
    mutual_reachability_graph,
    mutual_reachability_knn_l2,
)
from hdbscan.derived.hdbscan.detail.select import (
    CLUSTER_SELECTION_EOM,
    CLUSTER_SELECTION_LEAF,
    SELECT_TPB,
    cluster_epsilon_search,
)
from hdbscan.derived.hdbscan.detail.stabilities import STAB_TPB
from hdbscan.derived.hdbscan.runner import (
    GRAPH_BUILD_BRUTE_FORCE_KNN,
    GRAPH_BUILD_NN_DESCENT,
    HDBSCANOutput,
    HDBSCANParams,
    fit_hdbscan,
)
from hierarchy.original.edge_order import LINK_SAB_NONE
from hierarchy.original.linkage_oracle import partitions_agree
from hierarchy.derived.cluster.detail.connectivities import (
    DISTANCE_L1,
    DISTANCE_L2_SQRT_EXPANDED,
    FLOAT32_MAX,
    pairwise_distances,
)
from original.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL, numeric_mode_name
from neighbors.original.pinned_distance_tile import PINNED_TILE_TPB


comptime IDENTICAL_BUILD = GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL


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


def _hex32(v: Float32) -> String:
    var b = rebind[UInt32](v.to_bits())
    var digits = String("0123456789abcdef")
    var out = String("0x")
    var shift = 28
    while shift >= 0:
        var nib = Int((b >> UInt32(shift)) & UInt32(0xF))
        out += String(digits[byte=nib])
        shift -= 4
    return out


def _params_for(fix: Int, method: Int = CLUSTER_SELECTION_EOM) -> HDBSCANParams:
    return HDBSCANParams(
        hfixture_min_samples(fix),
        hfixture_min_cluster_size(fix),
        0,
        Float32(0.0),
        False,
        Float32(1.0),
        method,
        GRAPH_BUILD_BRUTE_FORCE_KNN,
    )


def _upload(
    ctx: DeviceContext, vals: List[Float32], pad: Int, poison: Float32
) raises -> DeviceBuffer[DType.float32]:
    """`vals` on the device with `pad` extra floats of `poison` after it --
    a buffer longer than its use, which a kernel reading past `m * d`
    would reveal."""
    var n = len(vals)
    var host = ctx.enqueue_create_host_buffer[DType.float32](n + pad)
    ctx.synchronize()
    for i in range(n):
        host.unsafe_ptr().unsafe_store(i, vals[i])
    for i in range(pad):
        host.unsafe_ptr().unsafe_store(n + i, poison)
    var dev = ctx.enqueue_create_buffer[DType.float32](n + pad)
    ctx.enqueue_copy(dst_buf=dev, src_ptr=host.unsafe_ptr())
    ctx.synchronize()
    _ = host^
    return dev^


def _copy_f32(
    ctx: DeviceContext, buf: DeviceBuffer[DType.float32], n: Int
) raises -> List[Float32]:
    var h = ctx.enqueue_create_host_buffer[DType.float32](n)
    ctx.synchronize()
    var v = buf.create_sub_buffer[DType.float32](0, n)
    ctx.enqueue_copy(dst_ptr=h.unsafe_ptr(), src_buf=v)
    ctx.synchronize()
    var out = List[Float32](capacity=n)
    for i in range(n):
        out.append(h.unsafe_ptr().unsafe_load(i))
    _ = h^
    _ = v^
    return out^


@fieldwise_init
struct StagedGraph(Copyable, Movable):
    """The three arrays the graph half produces, on the host."""

    var core: List[Float32]
    var dists: List[Float32]
    var mr: List[Float32]


def _staged_graph(
    ctx: DeviceContext,
    vals: List[Float32],
    m: Int,
    d: Int,
    min_samples: Int,
    tile_tpb: Int = PINNED_TILE_TPB,
    mr_tpb: Int = MR_TPB,
    core_tpb: Int = CORE_TPB,
    pad: Int = 0,
    poison: Float32 = Float32(0.0),
    sabotage: Int32 = HDB_SAB_NONE,
) raises -> StagedGraph:
    """The graph half of `build_mr_linkage`, stage by stage, every
    intermediate copied back. Uses the SAME entry points the fit does; the
    only thing it adds is the copies."""
    var k = min_samples + 1
    if k > m:
        k = m
    var x_host = vals.copy()
    var x = _upload(ctx, vals, pad, poison)
    var trace = IdentityTrace.disabled()
    var core = ctx.enqueue_create_buffer[DType.float32](m)
    var knn_d = ctx.enqueue_create_buffer[DType.float32](m * k)
    var knn_i = ctx.enqueue_create_buffer[DType.int32](m * k)
    ctx.synchronize()
    compute_core_dists(
        ctx, trace, x_host, core, m, d, DISTANCE_L2_SQRT_EXPANDED, k,
        knn_d, knn_i, core_tpb, sabotage,
    )
    var nnz = m * m
    var indptr = ctx.enqueue_create_buffer[DType.int32](m + 1)
    var indices = ctx.enqueue_create_buffer[DType.int32](nnz)
    var dists = ctx.enqueue_create_buffer[DType.float32](nnz)
    var norms = ctx.enqueue_create_buffer[DType.float32](m)
    pairwise_distances(
        ctx, x, m, d, DISTANCE_L2_SQRT_EXPANDED, indptr, indices, dists,
        norms, tile_tpb, LINK_SAB_NONE,
    )
    var mr = ctx.enqueue_create_buffer[DType.float32](nnz)
    mutual_reachability_dense(
        ctx, mr, dists, core, m, Float32(1.0), mr_tpb, sabotage
    )
    var h_core = _copy_f32(ctx, core, m)
    var h_dists = _copy_f32(ctx, dists, nnz)
    var h_mr = _copy_f32(ctx, mr, nnz)
    _ = x^
    _ = core^
    _ = knn_d^
    _ = knn_i^
    _ = indptr^
    _ = indices^
    _ = dists^
    _ = norms^
    _ = mr^
    _ = x_host^
    return StagedGraph(h_core^, h_dists^, h_mr^)


def _fit(
    ctx: DeviceContext,
    vals: List[Float32],
    m: Int,
    d: Int,
    params: HDBSCANParams,
    mut trace: IdentityTrace,
    tile_tpb: Int = PINNED_TILE_TPB,
    mst_tpb: Int = 256,
    mr_tpb: Int = MR_TPB,
    core_tpb: Int = CORE_TPB,
    stab_tpb: Int = STAB_TPB,
    select_tpb: Int = SELECT_TPB,
    pad: Int = 0,
    poison: Float32 = Float32(0.0),
    sabotage: Int32 = HDB_SAB_NONE,
) raises -> HDBSCANOutput:
    var x_host = vals.copy()
    var x = _upload(ctx, vals, pad, poison)
    var out = fit_hdbscan(
        ctx, trace, x_host, x, m, d, DISTANCE_L2_SQRT_EXPANDED, params,
        tile_tpb, mst_tpb, mr_tpb, core_tpb, stab_tpb, select_tpb, sabotage,
    )
    _ = x^
    _ = x_host^
    return out^


def _fit_plain(
    ctx: DeviceContext, fix: Int, sabotage: Int32 = HDB_SAB_NONE
) raises -> HDBSCANOutput:
    var trace = IdentityTrace.disabled()
    return _fit(
        ctx, hfixture_as_list(fix), hfixture_n(fix), hfixture_d(fix),
        _params_for(fix), trace, sabotage=sabotage,
    )


def _bits_differ(a: List[Float32], b: List[Float32]) -> Int:
    var n = 0
    for i in range(len(a)):
        if a[i].to_bits() != b[i].to_bits():
            n += 1
    return n


def _i32_differ(a: List[Int32], b: List[Int32]) -> Int:
    if len(a) != len(b):
        return -1
    var n = 0
    for i in range(len(a)):
        if a[i] != b[i]:
            n += 1
    return n


def _expect_raise(name: String, msg: String, want: String) raises:
    if msg.find(want) < 0:
        raise Error(
            "check_hdbscan_refusals [" + _mode_name() + "] " + name
            + ": the refusal did not name '" + want + "'; it said: " + msg
        )


# ======================================================================
# 1. REFUSALS
# ======================================================================


def check_hdbscan_refusals() raises:
    """Every unported arm and every out-of-range parameter RAISES BY NAME.

    A refusal is not tested by the absence of a wrong answer; it is tested
    by the presence of the NAME in the message, because a refusal whose
    text does not say what was refused is a refusal nobody can act on.
    """
    var ctx = DeviceContext()
    var fix = HFIX_BLOBS
    var m = hfixture_n(fix)
    var d = hfixture_d(fix)
    var vals = hfixture_as_list(fix)
    var n_ok = 0

    # The two unported halves of the SPARSE graph (DEVIATION 1600).
    try:
        mutual_reachability_knn_l2()
        raise Error("mutual_reachability_knn_l2 did not raise")
    except e:
        _expect_raise("mutual_reachability_knn_l2", String(e), "DEVIATION 1600")
        _expect_raise("mutual_reachability_knn_l2", String(e), "DistanceEpilogue")
        n_ok += 1
    try:
        mutual_reachability_graph(3)
        raise Error("mutual_reachability_graph did not raise")
    except e:
        _expect_raise("mutual_reachability_graph", String(e), "connect_knn_graph")
        n_ok += 1
    # The epsilon search (rung 2).
    try:
        cluster_epsilon_search(Float32(0.5))
        raise Error("cluster_epsilon_search did not raise")
    except e:
        _expect_raise("cluster_epsilon_search", String(e), "NOT PORTED")
        n_ok += 1

    # Parameters, one fit each.
    var trace = IdentityTrace.disabled()

    try:
        var x_host = vals.copy()
        var x = _upload(ctx, vals, 0, Float32(0.0))
        _ = fit_hdbscan(
            ctx, trace, x_host, x, m, d, DISTANCE_L1, _params_for(fix)
        )
        _ = x^
        raise Error("metric=L1 did not raise")
    except e:
        _expect_raise("metric=L1", String(e), "L2 expanded")
        n_ok += 1

    var p_nnd = _params_for(fix)
    p_nnd.build_algo = GRAPH_BUILD_NN_DESCENT
    try:
        _ = _fit(ctx, vals, m, d, p_nnd, trace)
        raise Error("build_algo=NN_DESCENT did not raise")
    except e:
        _expect_raise("build_algo=NN_DESCENT", String(e), "NN_DESCENT")
        n_ok += 1

    var p_eps = _params_for(fix)
    p_eps.cluster_selection_epsilon = Float32(0.25)
    try:
        _ = _fit(ctx, vals, m, d, p_eps, trace)
        raise Error("cluster_selection_epsilon != 0 did not raise")
    except e:
        _expect_raise("cluster_selection_epsilon", String(e), "NOT PORTED")
        n_ok += 1

    var p_ms0 = _params_for(fix)
    p_ms0.min_samples = 0
    try:
        _ = _fit(ctx, vals, m, d, p_ms0, trace)
        raise Error("min_samples=0 did not raise")
    except e:
        _expect_raise("min_samples=0", String(e), "min_samples")
        n_ok += 1

    var p_msbig = _params_for(fix)
    p_msbig.min_samples = m + 1
    try:
        _ = _fit(ctx, vals, m, d, p_msbig, trace)
        raise Error("min_samples > n_rows did not raise")
    except e:
        _expect_raise("min_samples>n_rows", String(e), "at most the number")
        n_ok += 1

    var p_mcs1 = _params_for(fix)
    p_mcs1.min_cluster_size = 1
    try:
        _ = _fit(ctx, vals, m, d, p_mcs1, trace)
        raise Error("min_cluster_size=1 did not raise")
    except e:
        _expect_raise("min_cluster_size=1", String(e), "min_cluster_size")
        n_ok += 1

    var p_mcsbig = _params_for(fix)
    p_mcsbig.min_cluster_size = m + 1
    try:
        _ = _fit(ctx, vals, m, d, p_mcsbig, trace)
        raise Error("min_cluster_size > n_rows did not raise")
    except e:
        _expect_raise("min_cluster_size>n_rows", String(e), "min_cluster_size")
        n_ok += 1

    var p_alpha = _params_for(fix)
    p_alpha.alpha = Float32(0.0)
    try:
        _ = _fit(ctx, vals, m, d, p_alpha, trace)
        raise Error("alpha=0 did not raise")
    except e:
        _expect_raise("alpha=0", String(e), "alpha")
        n_ok += 1

    var p_meth = _params_for(fix)
    p_meth.cluster_selection_method = 7
    try:
        _ = _fit(ctx, vals, m, d, p_meth, trace)
        raise Error("cluster_selection_method=7 did not raise")
    except e:
        _expect_raise("method=7", String(e), "cluster_selection_method")
        n_ok += 1

    # n_rows = 1. `fit_hdbscan` refuses this BEFORE it sizes any
    # `n_rows - 1` buffer, so the message names the caller's data and not
    # a zero-length allocation.
    var one = List[Float32]()
    for f in range(d):
        one.append(vals[f])
    var p_one = _params_for(fix)
    p_one.min_samples = 1
    try:
        _ = _fit(ctx, one, 1, d, p_one, trace)
        raise Error("n_rows=1 did not raise")
    except e:
        _expect_raise("n_rows=1", String(e), "n_rows=1")
        n_ok += 1

    # NON-FINITE INPUT, three plantings, DEVIATION 1607. The first two are
    # caught by hierarchy's DEVIATION 623 inside `pairwise_distances`; the
    # third (an infinite coordinate on one row only) reaches the core
    # distances, which 623 does not cover and 1607 does.
    var nan_vals = vals.copy()
    nan_vals[3 * d + 1] = bitcast[DType.float32](UInt32(0x7FC00000))
    try:
        _ = _fit(ctx, nan_vals, m, d, _params_for(fix), trace)
        raise Error("a NaN coordinate did not raise")
    except e:
        _expect_raise("NaN coordinate", String(e), "NaN")
        n_ok += 1

    var inf_vals = vals.copy()
    for f in range(d):
        inf_vals[5 * d + f] = Float32(1e20)
        inf_vals[6 * d + f] = Float32(1e20)
    try:
        _ = _fit(ctx, inf_vals, m, d, _params_for(fix), trace)
        raise Error("overflowing rows did not raise")
    except e:
        var msg = String(e)
        var named = msg.find("NaN") >= 0 or msg.find("infinite") >= 0
        if not named:
            raise Error(
                "check_hdbscan_refusals: overflowing rows raised without"
                " naming NaN or infinity: " + msg
            )
        n_ok += 1

    print(
        "check_hdbscan_refusals OK [" + _mode_name() + "]: "
        + String(n_ok) + " arms and parameters raise BY NAME (the sparse"
        " mutual reachability graph, its k-NN epilogue, the epsilon"
        " search, L1, NN_DESCENT, min_samples 0 and > n_rows,"
        " min_cluster_size 1 and > n_rows, alpha 0, an unknown selection"
        " method, n_rows 1, a NaN coordinate and two overflowing rows)"
    )


# ======================================================================
# 2. CORE DISTANCES
# ======================================================================


def check_core_distances_vs_oracle() raises:
    """Per cell, every fixture, against the host oracle's k-th order
    statistic -- which is taken by FULLY SORTING each row on
    `(distance, index)` rather than by a top-k, so an agreement is
    evidence about the value and not about a shared selector.

    IDENTITY hazard 1. The claim being gated is precisely: THE CORE
    DISTANCE VALUE IS BIT-IDENTICAL WHENEVER THE DISTANCE MATRIX IS, EVEN
    WHERE THE NEIGHBOUR SUPPLYING IT IS NOT UNIQUE. The duplicate fixture
    is where it has content: there the k-th neighbour is genuinely
    ambiguous and the value still is not.
    """
    var ctx = DeviceContext()
    for fix in range(HFIX_COUNT):
        var m = hfixture_n(fix)
        var d = hfixture_d(fix)
        var vals = hfixture_as_list(fix)
        var sg = _staged_graph(
            ctx, vals, m, d, hfixture_min_samples(fix)
        )
        var oracle = oracle_run(fix, vals)
        var mism = _bits_differ(sg.core, oracle.core)
        # How ambiguous is the k-th neighbour on this fixture? Count the
        # rows where the k-th and (k+1)-th smallest distances are EQUAL.
        var ambiguous = 0
        for i in range(m):
            var c = oracle.core[i]
            var at = 0
            for j in range(m):
                var v = oracle.dists[i * m + j]
                if i == j:
                    v = Float32(0.0)
                if v.to_bits() == c.to_bits():
                    at += 1
            if at > 1:
                ambiguous += 1
        comptime if IDENTICAL_BUILD:
            if mism != 0:
                var first = -1
                for i in range(m):
                    if sg.core[i].to_bits() != oracle.core[i].to_bits():
                        first = i
                        break
                raise Error(
                    "check_core_distances_vs_oracle [IDENTICAL] "
                    + hfixture_name(fix) + ": " + String(mism) + " of "
                    + String(m) + " core distances differ from the host"
                    " oracle; first at row " + String(first) + " device "
                    + _hex32(sg.core[first]) + " oracle "
                    + _hex32(oracle.core[first])
                )
            print(
                "check_core_distances_vs_oracle OK [IDENTICAL]: "
                + hfixture_name(fix) + " " + String(m)
                + " core distances bitwise equal to the host k-th order"
                " statistic; " + String(ambiguous) + " of " + String(m)
                + " rows have a TIED k-th neighbour, where the value is"
                " gated and the neighbour identity is not claimed"
            )
        else:
            print(
                "check_core_distances_vs_oracle REPORT [FAST]: "
                + hfixture_name(fix) + " " + String(mism) + " of "
                + String(m) + " core distances differ from the host pinned"
                " arithmetic (the FAST k-NN is a vendor matmul path; no"
                " assertion), " + String(ambiguous) + " rows with a tied"
                " k-th neighbour"
            )


# ======================================================================
# 3. MUTUAL REACHABILITY
# ======================================================================


def _count_mr_ties(mr: List[Float32], m: Int) -> Int:
    """Pairs `(i < j)` whose mutual reachability equals that of some other
    pair. Counted by sorting the upper triangle's bit patterns."""
    var vals = List[UInt32]()
    for i in range(m):
        for j in range(i + 1, m):
            vals.append(rebind[UInt32](mr[i * m + j].to_bits()))
    var n = len(vals)
    # insertion sort; the fixtures are small and this is a check
    for a in range(1, n):
        var v = vals[a]
        var b = a - 1
        while b >= 0 and vals[b] > v:
            vals[b + 1] = vals[b]
            b -= 1
        vals[b + 1] = v
    var ties = 0
    for a in range(1, n):
        if vals[a] == vals[a - 1]:
            ties += 1
    return ties


def check_mutual_reachability_ties() raises:
    """IDENTITY hazard 2, four claims on one fixture family.

    (a) `mr` is EXACTLY SYMMETRIC. A total-order max is commutative and
        associative, so `mr(a,b)` and `mr(b,a)` are the same three
        operands in a different order and must be the same bits. This is
        what makes the MST's undirected edge order well defined at all
        (`hierarchy/original/edge_order.mojo`'s block), so it is asserted
        first.
    (b) the device matrix equals the host oracle's, cell for cell.
    (c) a REVERSED input row order gives the same MST EDGE SET (as
        undirected pairs with their weights), which is the tie claim
        stated over the structure a tie can actually move.
    (d) a PLANTED `-0.0` core distance against a `+0.0` one resolves to
        `+0.0` -- the row-39 case, which no fixture coordinate can reach
        (the distance clamp maps every negative residue to `+0.0`), so it
        is planted directly into the kernel's inputs.
    """
    var ctx = DeviceContext()
    var fix = HFIX_DUPS
    var m = hfixture_n(fix)
    var d = hfixture_d(fix)
    var vals = hfixture_as_list(fix)
    var sg = _staged_graph(ctx, vals, m, d, hfixture_min_samples(fix))

    # (a) symmetry
    var asym = 0
    for i in range(m):
        for j in range(i + 1, m):
            if sg.mr[i * m + j].to_bits() != sg.mr[j * m + i].to_bits():
                asym += 1
    if asym != 0:
        raise Error(
            "check_mutual_reachability_ties [" + _mode_name() + "] "
            + hfixture_name(fix) + ": " + String(asym) + " of "
            + String(m * (m - 1) // 2) + " cells have mr(i,j) != mr(j,i)."
            " A total-order max is commutative, so this cannot happen"
            " under the pin; it is what HDB_SAB_MR_TWO_WAY produces"
        )

    # (b) against the oracle
    var oracle = oracle_run(fix, vals)
    var mism = _bits_differ(sg.mr, oracle.mr)
    var ties = _count_mr_ties(sg.mr, m)
    comptime if IDENTICAL_BUILD:
        if mism != 0:
            raise Error(
                "check_mutual_reachability_ties [IDENTICAL] "
                + hfixture_name(fix) + ": " + String(mism) + " of "
                + String(m * m) + " mutual reachability cells differ from"
                " the host three-way max"
            )
    else:
        print(
            "check_mutual_reachability_ties REPORT [FAST]: "
            + hfixture_name(fix) + " " + String(mism) + " of "
            + String(m * m) + " cells differ from the host three-way max"
        )

    # (c) reversed input order, same MST edge set
    var rev = List[Float32](capacity=m * d)
    for r in range(m):
        var src = m - 1 - r
        for f in range(d):
            rev.append(vals[src * d + f])
    var a_out = _fit_plain(ctx, fix)
    var trace = IdentityTrace.disabled()
    var b_out = _fit(ctx, rev, m, d, _params_for(fix), trace)
    # The MST edge set is not returned; what IS returned and is decided by
    # it is the PARTITION, which must survive the reversal up to the
    # relabeling the reversal forces.
    var b_back = List[Int32](capacity=m)
    for i in range(m):
        b_back.append(b_out.labels[m - 1 - i])
    var same_partition = partitions_agree(a_out.labels, b_back)
    var noise_moved = 0
    for i in range(m):
        var an = a_out.labels[i] == Int32(-1)
        var bn = b_back[i] == Int32(-1)
        if an != bn:
            noise_moved += 1
    # PERMUTATION INVARIANCE IS ASSERTED ONLY WHERE THERE ARE NO TIES, and
    # the reason is a property of the algorithm rather than of this port.
    #
    # The MST tie-break is a total order on `(weight, min(u,v), max(u,v))`
    # (`hierarchy/original/edge_order.mojo`), whose second and third
    # components are ROW INDICES. A permutation of the input renames those
    # indices, so among several equal-weight edges it can select a
    # DIFFERENT minimum spanning tree. Different MST, different dendrogram,
    # different condensed tree, and points can legitimately move between a
    # cluster and noise. Mutual reachability makes this endemic rather than
    # rare, because `mr` collapses to a core distance whenever two points
    # are closer than their cores, so whole blocks of pairs tie.
    #
    # An index-based tie-break cannot be permutation-equivariant. Closing
    # this needs a tie-break on something INTRINSIC to the points, and that
    # is a design question this rung does not answer. What IS true and is
    # worth stating beside it: upstream breaks the same tie with a cuRAND
    # draw (`mst.cuh:167-190`), so ours is at least a function of the input
    # rather than of the run, and that is a strictly stronger position, not
    # a solved one.
    #
    # So: no ties means the partition MUST survive, and that is asserted.
    # Ties present means it may not, and the counts are RECORDED. Reporting
    # a number here rather than asserting is what keeps the tie-free
    # fixtures' assertion meaningful.
    if ties == 0:
        if not same_partition or noise_moved != 0:
            raise Error(
                "check_mutual_reachability_ties [" + _mode_name() + "] "
                + hfixture_name(fix) + ": a REVERSED input row order changed"
                " the partition (agree=" + String(same_partition)
                + ", noise moved on " + String(noise_moved) + " points) on a"
                " fixture with ZERO tied pairs. With no tie to break, the"
                " MST is unique and the partition cannot depend on the row"
                " order. This is a real order dependence."
            )
        print(
            "  permutation OK [" + _mode_name() + "] "
            + hfixture_name(fix)
            + ": 0 tied pairs, partition survives a reversed row order"
        )
    else:
        print(
            "  permutation RECORDED [" + _mode_name() + "] "
            + hfixture_name(fix) + ": " + String(ties)
            + " tied pairs; reversed row order agree="
            + String(same_partition) + ", noise moved on "
            + String(noise_moved) + " of " + String(m) + " points. EXPECTED"
            " and not asserted: the MST tie-break is a total order on row"
            " INDICES, so a permutation can select a different equal-weight"
            " MST. Upstream breaks the same tie with a cuRAND draw"
        )

    # (d) the planted signed zero, straight into the kernel.
    var pm = 4
    var pdists = List[Float32](capacity=pm * pm)
    for i in range(pm):
        for j in range(pm):
            if i == j:
                pdists.append(FLOAT32_MAX)
            else:
                # every off-diagonal distance a NEGATIVE ZERO, so the
                # three-way max sees (+0, -0) and (-0, -0) pairs.
                pdists.append(bitcast[DType.float32](NEG_ZERO_BITS))
    var pcore = List[Float32](capacity=pm)
    pcore.append(Float32(0.0))
    pcore.append(bitcast[DType.float32](NEG_ZERO_BITS))
    pcore.append(Float32(0.0))
    pcore.append(bitcast[DType.float32](NEG_ZERO_BITS))
    var d_dists = _upload(ctx, pdists, 0, Float32(0.0))
    var d_core = _upload(ctx, pcore, 0, Float32(0.0))
    var d_mr = ctx.enqueue_create_buffer[DType.float32](pm * pm)
    ctx.synchronize()
    mutual_reachability_dense(
        ctx, d_mr, d_dists, d_core, pm, Float32(1.0), MR_TPB, HDB_SAB_NONE
    )
    var h_mr = _copy_f32(ctx, d_mr, pm * pm)
    var bad = 0
    for i in range(pm):
        for j in range(pm):
            if i == j:
                continue
            var want = UInt32(0x80000000)
            # `max` of three zeros is `+0.0` unless EVERY operand is
            # `-0.0`, in which case it is `-0.0`. Rows/cols 0 and 2 carry
            # a `+0.0` core, so any pair touching one of them is `+0.0`.
            if (i == 1 or i == 3) and (j == 1 or j == 3):
                want = UInt32(0x80000000)
            else:
                want = UInt32(0x00000000)
            if rebind[UInt32](h_mr[i * pm + j].to_bits()) != want:
                bad += 1
    comptime if IDENTICAL_BUILD:
        if bad != 0:
            raise Error(
                "check_mutual_reachability_ties [IDENTICAL] PLANTED"
                " SIGNED ZERO: " + String(bad) + " of "
                + String(pm * pm - pm) + " cells resolved the (+0, -0)"
                " three-way max to the wrong zero. Under the pin the"
                " answer is +0.0 unless every operand is -0.0"
                " (IDENTITY_PATHS row 39); a hardware max would answer"
                " -0.0 on Apple and +0.0 on NVIDIA and AMD"
            )
        print(
            "check_mutual_reachability_ties OK [IDENTICAL]: "
            + hfixture_name(fix) + " symmetric on "
            + String(m * (m - 1) // 2) + " pairs with " + String(ties)
            + " tied pairs, " + String(m * m) + " cells bitwise equal to"
            " the host three-way max, the reversed-row-order result"
            " RECORDED above rather than asserted, and the planted"
            " (+0, -0) max resolves to +0.0 on every cell"
        )
    else:
        print(
            "check_mutual_reachability_ties OK [FAST]: "
            + hfixture_name(fix) + " symmetric, " + String(ties)
            + " tied pairs, reversed row order RECORDED above rather than asserted;"
            " the planted signed-zero max is RECORDED (" + String(bad)
            + " cells differ from the pinned answer -- under FAST the"
            " three-way max is the stdlib max and its zero tie is the"
            " vendor's)"
        )


# ======================================================================
# 4. CONDENSED TREE
# ======================================================================


def check_condensed_tree_vs_oracle() raises:
    """Node for node: `parents`, `children`, `sizes` exactly and
    `lambdas` bitwise, plus `n_edges` and `n_clusters`, on every fixture.

    THE CONDENSED TREE IS WHERE THE TRAVERSAL BECOMES THE NUMBERING, and
    every array downstream of it is indexed by that numbering, so this is
    the gate that a DFS rewrite has to pass through.
    """
    var ctx = DeviceContext()
    for fix in range(HFIX_COUNT):
        var vals = hfixture_as_list(fix)
        var out = _fit_plain(ctx, fix)
        var oracle = oracle_run(fix, vals)
        var n_edges = out.condensed.n_edges
        var n_clusters = out.condensed.n_clusters
        if (
            n_edges != oracle.condensed.n_edges
            or n_clusters != oracle.condensed.n_clusters
        ):
            raise Error(
                "check_condensed_tree_vs_oracle [" + _mode_name() + "] "
                + hfixture_name(fix) + ": shape differs -- device "
                + String(n_edges) + " edges / " + String(n_clusters)
                + " clusters, oracle "
                + String(oracle.condensed.n_edges) + " / "
                + String(oracle.condensed.n_clusters)
            )
        var dp = _i32_differ(out.condensed.parents, oracle.condensed.parents)
        var dc = _i32_differ(out.condensed.children, oracle.condensed.children)
        var ds = _i32_differ(out.condensed.sizes, oracle.condensed.sizes)
        var dl = _bits_differ(out.condensed.lambdas, oracle.condensed.lambdas)
        # ORCHESTRATOR DIAGNOSTIC, kept because it is what localized the
        # first defect this check found. Counts matching while contents
        # differ separates a LABELING difference from a shape difference.
        if dp != 0 or dc != 0 or ds != 0:
            var first = -1
            for ei in range(n_edges):
                if (out.condensed.parents[ei]
                        != oracle.condensed.parents[ei]
                    or out.condensed.children[ei]
                        != oracle.condensed.children[ei]
                    or out.condensed.sizes[ei]
                        != oracle.condensed.sizes[ei]):
                    first = ei
                    break
            if first >= 0:
                print(
                    "  condense diff [" + _mode_name() + "] "
                    + hfixture_name(fix) + " first at edge "
                    + String(first) + " of " + String(n_edges)
                    + ": device (p=" + String(out.condensed.parents[first])
                    + ", c=" + String(out.condensed.children[first])
                    + ", sz=" + String(out.condensed.sizes[first])
                    + ") oracle (p="
                    + String(oracle.condensed.parents[first])
                    + ", c=" + String(oracle.condensed.children[first])
                    + ", sz=" + String(oracle.condensed.sizes[first]) + ")"
                )
        comptime if IDENTICAL_BUILD:
            if dp != 0 or dc != 0 or ds != 0 or dl != 0:
                raise Error(
                    "check_condensed_tree_vs_oracle [IDENTICAL] "
                    + hfixture_name(fix) + ": " + String(dp)
                    + " parents, " + String(dc) + " children, "
                    + String(ds) + " sizes and " + String(dl)
                    + " lambdas differ from the oracle over "
                    + String(n_edges) + " edges"
                )
            print(
                "check_condensed_tree_vs_oracle OK [IDENTICAL]: "
                + hfixture_name(fix) + " " + String(n_edges)
                + " condensed edges over " + String(n_clusters)
                + " clusters, node for node, lambdas bitwise"
            )
        else:
            if dp != 0 or dc != 0 or ds != 0:
                raise Error(
                    "check_condensed_tree_vs_oracle [FAST] "
                    + hfixture_name(fix) + ": the condensed tree's INTEGER"
                    " structure differs from the oracle (" + String(dp)
                    + " parents, " + String(dc) + " children, "
                    + String(ds) + " sizes). The structure follows from"
                    " the dendrogram, so this is not a distance report"
                )
            print(
                "check_condensed_tree_vs_oracle OK [FAST]: "
                + hfixture_name(fix) + " structure identical; lambdas "
                + String(dl) + " of " + String(n_edges)
                + " differ (REPORT: the deltas are vendor distances)"
            )


# ======================================================================
# 5. STABILITIES
# ======================================================================


def check_stabilities_vs_oracle() raises:
    """Per cluster, bit for bit, on every fixture.

    THE VALUE COMPARED IS THE POST-SELECTION ONE. Excess of Mass
    OVERWRITES `stability[node]` with the subtree total on a deselected
    node (`select.cuh:227`), so what the fit hands back is the array AFTER
    that mutation, and the oracle reproduces the mutation in the same
    order. Comparing the pre-selection array instead would leave the
    write-back untested, which is exactly what HDB_SAB_EOM_NO_UPDATE
    breaks.
    """
    var ctx = DeviceContext()
    for fix in range(HFIX_COUNT):
        var vals = hfixture_as_list(fix)
        var out = _fit_plain(ctx, fix)
        var oracle = oracle_run(fix, vals)
        var got = out.tree_stabilities.copy()
        var want = oracle.selection.stabilities.copy()
        if len(got) != len(want):
            raise Error(
                "check_stabilities_vs_oracle [" + _mode_name() + "] "
                + hfixture_name(fix) + ": " + String(len(got))
                + " clusters, oracle " + String(len(want))
            )
        var diff = _bits_differ(got, want)
        comptime if IDENTICAL_BUILD:
            if diff != 0:
                var first = -1
                for i in range(len(got)):
                    if got[i].to_bits() != want[i].to_bits():
                        first = i
                        break
                raise Error(
                    "check_stabilities_vs_oracle [IDENTICAL] "
                    + hfixture_name(fix) + ": " + String(diff) + " of "
                    + String(len(got)) + " stabilities differ; first at"
                    " cluster " + String(first) + " device "
                    + _hex32(got[first]) + " oracle " + _hex32(want[first])
                )
            print(
                "check_stabilities_vs_oracle OK [IDENTICAL]: "
                + hfixture_name(fix) + " " + String(len(got))
                + " cluster stabilities bitwise equal to the host serial"
                " ascending fold (DEVIATIONS 1603/1604/1605)"
            )
        else:
            print(
                "check_stabilities_vs_oracle REPORT [FAST]: "
                + hfixture_name(fix) + " " + String(diff) + " of "
                + String(len(got)) + " stabilities differ (the lambdas"
                " they sum are vendor distances)"
            )


# ======================================================================
# 6. LABELS AND OUTLIERS
# ======================================================================


def check_labels_vs_oracle() raises:
    """Three separate claims, and they are not the same claim.

    (a) the labels equal the oracle's, cell for cell;
    (b) the OUTLIER COUNT equals the oracle's, on its own -- a label array
        that agrees everywhere except on which points are noise would pass
        a partition test and is a different bug;
    (c) on the fixtures that PLANT an assignment, the partition equals the
        planted one and the planted noise points come back as `-1`. This
        is the only control here that does not share a line with the port.
    """
    var ctx = DeviceContext()
    for fix in range(HFIX_COUNT):
        var m = hfixture_n(fix)
        var vals = hfixture_as_list(fix)
        var out = _fit_plain(ctx, fix)
        var oracle = oracle_run(fix, vals)

        var dl = _i32_differ(out.labels, oracle.labels)
        var oracle_outliers = oracle.n_outliers
        comptime if IDENTICAL_BUILD:
            if dl != 0:
                raise Error(
                    "check_labels_vs_oracle [IDENTICAL] "
                    + hfixture_name(fix) + ": " + String(dl) + " of "
                    + String(m) + " labels differ from the oracle"
                )
            if out.n_outliers != oracle_outliers:
                raise Error(
                    "check_labels_vs_oracle [IDENTICAL] "
                    + hfixture_name(fix) + ": " + String(out.n_outliers)
                    + " outliers, oracle " + String(oracle_outliers)
                )
        else:
            if dl != 0:
                print(
                    "check_labels_vs_oracle RECORDED [FAST]: "
                    + hfixture_name(fix) + " " + String(dl) + " of "
                    + String(m) + " labels differ from the oracle (the"
                    " FAST distances are a vendor matmul's; not asserted)"
                )

        var planted_n = hfixture_n_planted_clusters(fix)
        if planted_n > 0:
            # THE PLANTED GATE IS TWO ASSERTIONS AND TWO RECORDS, AND THE
            # SPLIT IS DELIBERATE.
            #
            # ASSERTED, because a failure is a defect and nothing else:
            #   (i) every point the fixture plants as NOISE comes back -1;
            #  (ii) no returned cluster contains points from two DIFFERENT
            #       planted clusters. The fixture's clusters are separated
            #       by construction -- centres 11 to 12 units apart with a
            #       jitter under 1 -- so merging two of them is a defect
            #       in the linkage or the selection.
            #
            # RECORDED, because a failure would be a MODELLING outcome and
            # not a defect:
            # (iii) how many returned clusters each planted cluster was
            #       split into. HDBSCAN is entitled to split a blob whose
            #       jitter happens to leave a density gap, and asserting
            #       it does not would be building the gate to the fixture.
            #  (iv) how many planted-cluster points came back as noise.
            #       With `allow_single_cluster = False` a point whose
            #       nearest SELECTED ancestor is the root is noise by
            #       their construction (`extract.cuh:141-160`), which is
            #       a property of the data and the parameters.
            var planted = List[Int32](capacity=m)
            for i in range(m):
                planted.append(hfixture_planted_label(fix, i))
            var noise_bad = 0
            for i in range(m):
                if planted[i] == Int32(-1) and out.labels[i] != Int32(-1):
                    noise_bad += 1
            var merged = 0
            for i in range(m):
                if planted[i] == Int32(-1) or out.labels[i] == Int32(-1):
                    continue
                for j in range(i + 1, m):
                    if planted[j] == Int32(-1) or out.labels[j] == Int32(-1):
                        continue
                    if planted[i] != planted[j] and out.labels[i] == out.labels[j]:
                        merged += 1
            if noise_bad != 0 or merged != 0:
                raise Error(
                    "check_labels_vs_oracle [" + _mode_name() + "] "
                    + hfixture_name(fix) + ": " + String(noise_bad)
                    + " PLANTED-NOISE points came back with a cluster"
                    " label, and " + String(merged) + " pairs from two"
                    " DIFFERENT planted clusters share a returned"
                    " cluster. The fixture's clusters are separated by"
                    " construction, so a merge is a defect in the linkage"
                    " or the selection and not a modelling choice"
                )
            # (iii) and (iv), recorded.
            var split_max = 0
            var noise_in_clusters = 0
            for c in range(planted_n):
                var seen = List[Int32]()
                for i in range(m):
                    if Int(planted[i]) != c:
                        continue
                    if out.labels[i] == Int32(-1):
                        noise_in_clusters += 1
                        continue
                    var found = False
                    for t in range(len(seen)):
                        if seen[t] == out.labels[i]:
                            found = True
                    if not found:
                        seen.append(out.labels[i])
                if len(seen) > split_max:
                    split_max = len(seen)
            print(
                "check_labels_vs_oracle OK [" + _mode_name() + "]: "
                + hfixture_name(fix) + " " + String(m) + " labels, "
                + String(out.n_clusters) + " returned clusters, "
                + String(out.n_outliers) + " outliers. ASSERTED: every"
                " planted-noise point came back -1 and no two planted"
                " clusters were merged. RECORDED: the most any one of the "
                + String(planted_n) + " planted clusters was split into is "
                + String(split_max) + " returned clusters, and "
                + String(noise_in_clusters) + " planted-cluster points"
                " came back as noise"
            )
        else:
            print(
                "check_labels_vs_oracle OK [" + _mode_name() + "]: "
                + hfixture_name(fix) + " " + String(m) + " labels match"
                " the oracle, " + String(out.n_clusters) + " clusters, "
                + String(out.n_outliers) + " outliers (this fixture plants"
                " no assignment; the oracle is the only control)"
            )


# ======================================================================
# 7. PERMUTATION INVARIANCE
# ======================================================================


def check_permutation_invariance() raises:
    """The same point set in a different ROW ORDER must give the same
    PARTITION, up to the relabeling the permutation forces.

    WHAT THIS GATE CAN FIND, AND WHAT IT IS HONEST ABOUT. The MST's tie
    break is `(weight key, min(u,v), max(u,v))` -- it reads the INDEX --
    so under an EXACT tie a permutation can select a different (equally
    minimal) MST, and a different MST can condense to a different tree.
    That is a real order dependence and it is the algorithm's, not a bug
    in this port: upstream's answer under the same tie is a cuRAND draw,
    which is worse. **Mutual reachability makes ties ENDEMIC rather than
    exceptional**: `mr(a,b) = max(core_a, core_b, d(a,b))` collapses to a
    CORE DISTANCE whenever the points are closer than their cores, and
    many pairs then share one value.

    So the gate counts the ties first and states its verdict accordingly:
    ZERO tied pairs -> ASSERT the partition; any tied pair -> RECORD the
    outcome with the counts. A recorded failure on a tie-bearing fixture
    is a fact about the algorithm; an ASSERTED failure on a tie-free one
    would be a defect.
    """
    var ctx = DeviceContext()
    for fix in range(HFIX_COUNT):
        var m = hfixture_n(fix)
        var d = hfixture_d(fix)
        var vals = hfixture_as_list(fix)
        var sg = _staged_graph(ctx, vals, m, d, hfixture_min_samples(fix))
        var ties = _count_mr_ties(sg.mr, m)

        var perm = hfixture_permutation(fix)
        var pvals = hfixture_permuted_as_list(fix, perm)
        var a_out = _fit_plain(ctx, fix)
        var trace = IdentityTrace.disabled()
        var b_out = _fit(ctx, pvals, m, d, _params_for(fix), trace)

        # Undo the permutation: new row r held old row perm[r].
        var b_back = List[Int32](capacity=m)
        for _ in range(m):
            b_back.append(Int32(0))
        for r in range(m):
            b_back[perm[r]] = b_out.labels[r]

        var noise_moved = 0
        for i in range(m):
            var an = a_out.labels[i] == Int32(-1)
            var bn = b_back[i] == Int32(-1)
            if an != bn:
                noise_moved += 1
        var pa = List[Int32]()
        var pb = List[Int32]()
        for i in range(m):
            if a_out.labels[i] != Int32(-1) and b_back[i] != Int32(-1):
                pa.append(a_out.labels[i])
                pb.append(b_back[i])
        var agree = partitions_agree(pa, pb)
        var ok = agree and noise_moved == 0

        if ties == 0:
            if not ok:
                raise Error(
                    "check_permutation_invariance [" + _mode_name() + "] "
                    + hfixture_name(fix) + ": the mutual reachability"
                    " matrix has NO tied pair, so nothing in this fit may"
                    " read a row index -- and yet a permutation moved the"
                    " answer (partition agrees=" + String(agree)
                    + ", noise moved on " + String(noise_moved)
                    + " points). THAT IS AN ORDER DEPENDENCE AND IT HAS"
                    " BEEN FOUND: the next step is to find which stage"
                    " reads an index it should not, by running the two"
                    " orders under MOJOLEARN_IDENTITY_TRACE and diffing"
                    " the cards"
                )
            print(
                "check_permutation_invariance OK [" + _mode_name() + "]: "
                + hfixture_name(fix) + " tie-free (0 tied pairs of "
                + String(m * (m - 1) // 2) + "); a hashed permutation of"
                " the rows gives the same partition and the same noise set"
            )
        else:
            print(
                "check_permutation_invariance RECORDED ["
                + _mode_name() + "]: " + hfixture_name(fix) + " has "
                + String(ties) + " tied pairs of "
                + String(m * (m - 1) // 2)
                + ", so the MST's index tie-break can select a different"
                " (equally minimal) tree under a permutation. Result:"
                " partition agrees=" + String(agree) + ", noise moved on "
                + String(noise_moved) + " of " + String(m) + " points."
                " Upstream's answer under the same tie is a cuRAND draw"
                " (DEVIATION 620); ours is a function of the row order"
            )


# ======================================================================
# 8. LAUNCH INVARIANCE
# ======================================================================


def check_launch_invariance() raises:
    """THE HEADLINE. Every output byte is held fixed across:

      - two block sizes for each of the six kernels this lane launches
        (the distance tile, the MST, the mutual reachability transform,
        the core-distance slice, the stability fold, the selection BFS);
      - an input buffer PADDED past its use with a poison value, so a
        kernel reading past `m * d` would show;
      - the same cells computed ALONE and INSIDE a batch: a 37 x 37
        sub-block of the mutual reachability transform run on its own
        against the same cells of the full matrix, with the SAME core
        distances planted into both, which isolates the transform from
        the k-NN's own shape dependence.

    Block size and grid shape are SCHEDULING in both modes; every fold in
    this lane is either per-thread or per-segment, so there is nothing for
    them to reach.
    """
    var ctx = DeviceContext()
    var fix = HFIX_BLOBS
    var m = hfixture_n(fix)
    var d = hfixture_d(fix)
    var vals = hfixture_as_list(fix)
    var params = _params_for(fix)
    var t0 = IdentityTrace.disabled()
    var base = _fit(ctx, vals, m, d, params, t0)

    var t1 = IdentityTrace.disabled()
    var alt = _fit(
        ctx, vals, m, d, params, t1,
        tile_tpb=64, mst_tpb=128, mr_tpb=64, core_tpb=64,
        stab_tpb=64, select_tpb=64,
    )
    var t2 = IdentityTrace.disabled()
    var pad = _fit(
        ctx, vals, m, d, params, t2,
        pad=1024, poison=bitcast[DType.float32](UInt32(0x7FC00000)),
    )
    var t3 = IdentityTrace.disabled()
    var pad2 = _fit(
        ctx, vals, m, d, params, t3, pad=37, poison=Float32(1e30)
    )

    var moved = 0
    moved += _i32_differ(base.labels, alt.labels)
    moved += _i32_differ(base.labels, pad.labels)
    moved += _i32_differ(base.labels, pad2.labels)
    var core_moved = (
        _bits_differ(base.core_dists, alt.core_dists)
        + _bits_differ(base.core_dists, pad.core_dists)
        + _bits_differ(base.core_dists, pad2.core_dists)
    )
    var stab_moved = (
        _bits_differ(base.tree_stabilities, alt.tree_stabilities)
        + _bits_differ(base.tree_stabilities, pad.tree_stabilities)
        + _bits_differ(base.tree_stabilities, pad2.tree_stabilities)
    )
    if moved != 0 or core_moved != 0 or stab_moved != 0:
        raise Error(
            "check_launch_invariance [" + _mode_name() + "] "
            + hfixture_name(fix) + ": " + String(moved) + " label cells, "
            + String(core_moved) + " core distances and "
            + String(stab_moved) + " stabilities moved across two block"
            " sizes per kernel and two input paddings. Block size is"
            " SCHEDULING; if it reached a value, a fold in this lane is"
            " not the per-thread or per-segment shape it is documented to"
            " be"
        )

    # ALONE vs IN A BATCH, for the mutual reachability transform.
    var sg = _staged_graph(ctx, vals, m, d, hfixture_min_samples(fix))
    var sm = 37
    var sub_d = List[Float32](capacity=sm * sm)
    var sub_c = List[Float32](capacity=sm)
    for i in range(sm):
        sub_c.append(sg.core[i])
    for i in range(sm):
        for j in range(sm):
            sub_d.append(sg.dists[i * m + j])
    var dd = _upload(ctx, sub_d, 0, Float32(0.0))
    var dc = _upload(ctx, sub_c, 0, Float32(0.0))
    var dm = ctx.enqueue_create_buffer[DType.float32](sm * sm)
    ctx.synchronize()
    mutual_reachability_dense(
        ctx, dm, dd, dc, sm, Float32(1.0), 32, HDB_SAB_NONE
    )
    var h_sub = _copy_f32(ctx, dm, sm * sm)
    var cell_moved = 0
    for i in range(sm):
        for j in range(sm):
            if h_sub[i * sm + j].to_bits() != sg.mr[i * m + j].to_bits():
                cell_moved += 1
    if cell_moved != 0:
        raise Error(
            "check_launch_invariance [" + _mode_name() + "]: "
            + String(cell_moved) + " of " + String(sm * sm)
            + " mutual reachability cells differ between a "
            + String(sm) + " x " + String(sm) + " launch and the same"
            " cells of the " + String(m) + " x " + String(m) + " one, on"
            " identical inputs. Each cell is one thread and three loads,"
            " so a difference means the kernel read something the cell"
            " does not own"
        )
    _ = dd^
    _ = dc^
    _ = dm^
    print(
        "check_launch_invariance OK [" + _mode_name() + "]: "
        + hfixture_name(fix) + " labels, core distances and stabilities"
        " identical across tile 256/64, MST 256/128, mr 256/64, core"
        " 256/64, stability 256/64, select 256/64, a 1024-float NaN"
        " padding and a 37-float 1e30 padding; and " + String(sm * sm)
        + " mutual reachability cells identical alone and inside the"
        " full " + String(m) + "-row launch"
    )


# ======================================================================
# 9. THE CARD
# ======================================================================


def check_card_is_emitted() raises:
    """Two cards, two launch shapes, record for record identical -- and
    the stage list is the one `hdbscan_main.mojo`'s header names.

    A CARD IS NOT A CHECK OF THE ANSWER; it is the instrument that gives a
    cross-vendor difference an ADDRESS. What is asserted here is that the
    instrument exists, that it covers every stage, and that it is a
    function of the fit and not of the launch shape -- which is exactly
    what makes a later Apple-vs-NVIDIA diff readable.
    """
    var ctx = DeviceContext()
    var fix = HFIX_BLOBS
    var m = hfixture_n(fix)
    var d = hfixture_d(fix)
    var vals = hfixture_as_list(fix)
    var params = _params_for(fix)

    var pa = String("/tmp/hdbscan_check_card_a.trace")
    var pb = String("/tmp/hdbscan_check_card_b.trace")
    var ta = IdentityTrace.to_path(pa)
    _ = _fit(ctx, vals, m, d, params, ta)
    var tb = IdentityTrace.to_path(pb)
    _ = _fit(
        ctx, vals, m, d, params, tb,
        tile_tpb=64, mst_tpb=128, mr_tpb=64, core_tpb=64,
        stab_tpb=64, select_tpb=64,
    )
    var diff = first_divergence(pa, pb)
    if diff != "":
        raise Error(
            "check_card_is_emitted [" + _mode_name() + "]: two cards from"
            " two launch shapes disagree at " + diff
        )
    if ta.seq < 20:
        raise Error(
            "check_card_is_emitted [" + _mode_name() + "]: the card holds"
            " only " + String(ta.seq) + " records; the stage list in"
            " hdbscan_main.mojo's header names at least twenty, so a"
            " stage is not being recorded and a cross-vendor diff would"
            " have a blind span"
        )
    print(
        "check_card_is_emitted OK [" + _mode_name() + "]: "
        + String(ta.seq) + " stages recorded, and two cards taken at two"
        " launch shapes are record-for-record identical (" + pa + ", "
        + pb + ")"
    )


# ======================================================================
# 10. SIGNED-ZERO INPUTS
# ======================================================================


def check_hdbscan_signed_zero_inputs() raises:
    """A `-0.0` COORDINATE moves no output bit.

    The two fixtures differ in exactly one thing: feature 2 is `-0.0` on
    every row of one and `+0.0` on every row of the other. Every stage
    must agree bit for bit, because `(-0.0) * (-0.0)` is `+0.0` in the row
    norm and `q * y` with a zero operand contributes a zero the
    accumulator absorbs. If they ever disagree, a seam is carrying the
    sign of a zero into a value, and IDENTITY_PATHS row 39 says which
    vendors would then answer differently.
    """
    var ctx = DeviceContext()
    var a = _fit_plain(ctx, HFIX_SIGNED_ZERO)
    var b = _fit_plain(ctx, HFIX_POS_ZERO)
    var core_moved = _bits_differ(a.core_dists, b.core_dists)
    var stab_moved = _bits_differ(a.tree_stabilities, b.tree_stabilities)
    var label_moved = _i32_differ(a.labels, b.labels)
    if core_moved != 0 or stab_moved != 0 or label_moved != 0:
        raise Error(
            "check_hdbscan_signed_zero_inputs [" + _mode_name() + "]: a"
            " -0.0 input coordinate moved " + String(core_moved)
            + " core distances, " + String(stab_moved) + " stabilities"
            " and " + String(label_moved) + " labels against the same"
            " fixture with +0.0. The sign of an input zero is reaching an"
            " output; IDENTITY_PATHS row 39"
        )
    print(
        "check_hdbscan_signed_zero_inputs OK [" + _mode_name() + "]: a"
        " -0.0 coordinate on every row moves no core distance, no"
        " stability and no label against the +0.0 twin fixture"
    )


# ======================================================================
# 11. THE OTHER SIDE OF THE SELECTION SWITCH
# ======================================================================


def check_hdbscan_selection_leaf() raises:
    """PORTING_RULES rule 8: a switch is exercised on BOTH sides by a
    named check per side, with the switch set explicitly inside it.

    Excess of Mass is the default and every other gate runs it. This one
    runs LEAF and ASSERTS the selection against the oracle's leaf rule
    (selected iff a child of some edge and a parent of none).

    IT ALSO RECORDS, AND DOES NOT ASSERT, whether the LEAF selection ever
    DIFFERS from the EOM one. Two selection rules that agree on every
    fixture have not been distinguished by the fixtures, and a reader is
    owed that number -- but which rule differs where is a property of the
    trees these fixtures happen to produce, and asserting a difference
    would be building the gate to the fixture. A zero here is a REQUEST
    for a fixture whose root split has a higher stability than its
    leaves, not a failure of the port.
    """
    var ctx = DeviceContext()
    var n_differ = 0
    for fix in range(HFIX_COUNT):
        var m = hfixture_n(fix)
        var d = hfixture_d(fix)
        var vals = hfixture_as_list(fix)
        var p = _params_for(fix, CLUSTER_SELECTION_LEAF)
        var t = IdentityTrace.disabled()
        var out = _fit(ctx, vals, m, d, p, t)
        var oracle = oracle_run(fix, vals)
        var want = oracle_leaf_selection(oracle.condensed)
        var diff = _i32_differ(out.is_cluster, want)
        if diff != 0:
            raise Error(
                "check_hdbscan_selection_leaf [" + _mode_name() + "] "
                + hfixture_name(fix) + ": " + String(diff) + " of "
                + String(len(want)) + " selection flags differ from the"
                " oracle's leaf rule"
            )
        var eom = _fit_plain(ctx, fix)
        if _i32_differ(eom.is_cluster, out.is_cluster) != 0:
            n_differ += 1
    var note = String(
        " (so the two arms ARE distinguished by this suite)"
    )
    if n_differ == 0:
        note = String(
            " -- ZERO, so the two arms are INDISTINGUISHABLE on this"
            " suite and a green line here does not establish which one"
            " ran. That is an owed fixture, not a failure: it needs a"
            " tree whose root split has a higher stability than its"
            " leaves"
        )
    print(
        "check_hdbscan_selection_leaf OK [" + _mode_name() + "]: the LEAF"
        " arm matches the oracle's leaf rule on all " + String(HFIX_COUNT)
        + " fixtures. RECORDED: it differs from the EOM selection on "
        + String(n_differ) + " of them" + note
    )


# ======================================================================
# 12. THE FLOAT64 REFERENCE
# ======================================================================


def check_hdbscan_float64_reference() raises:
    """The Float32 mutual-reachability MST's total weight against a
    Float64 DIRECT-FORM one.

    The bit gates all compare against a host oracle that shares the
    expanded identity with the device. This one does not: it computes
    `sqrt(sum (x_i - x_j)^2)` in double, so a catastrophic cancellation in
    `||x||^2 + ||y||^2 - 2 x.y` shows up as a relative error rather than
    being reproduced on both sides. A miss is ASSERTED under IDENTICAL and
    RECORDED under FAST (the Float32 total is then a vendor product's).
    """
    var tol = Float64(1e-4)
    for fix in range(HFIX_COUNT):
        var vals = hfixture_as_list(fix)
        var oracle = oracle_run(fix, vals)
        var denom = oracle.mst_total_f64
        if denom < Float64(0.0):
            denom = -denom
        if denom < Float64(1e-12):
            denom = Float64(1e-12)
        var err = oracle.mst_total_f32_in_f64 - oracle.mst_total_f64
        if err < Float64(0.0):
            err = -err
        var rel = err / denom
        comptime if IDENTICAL_BUILD:
            if rel > tol:
                raise Error(
                    "check_hdbscan_float64_reference [IDENTICAL] "
                    + hfixture_name(fix) + ": the Float32 mutual"
                    " reachability MST total is " + String(rel)
                    + " relative from the Float64 direct-form one, over "
                    + String(tol)
                )
            print(
                "check_hdbscan_float64_reference OK [IDENTICAL]: "
                + hfixture_name(fix) + " MST total relative error "
                + String(rel) + " against a Float64 direct-form mutual"
                " reachability MST"
            )
        else:
            print(
                "check_hdbscan_float64_reference RECORDED [FAST]: "
                + hfixture_name(fix) + " relative error " + String(rel)
            )


# ======================================================================
# 13. SABOTAGES
# ======================================================================


def check_hdbscan_sabotages() raises:
    """Each arm breaks ONE pin; the gate it breaks must move, or the check
    says so in the line it prints.

    `[[reached-but-inert]]`: a path that runs is not a path that is gated.
    Every MUST FAIL line below raises if the gate DOES NOT move, which is
    the only way to establish that the gate has teeth.
    """
    var ctx = DeviceContext()

    # HDB_SAB_MR_TWO_WAY: mr stops being symmetric.
    var fix = HFIX_DUPS
    var m = hfixture_n(fix)
    var d = hfixture_d(fix)
    var vals = hfixture_as_list(fix)
    var sab = _staged_graph(
        ctx, vals, m, d, hfixture_min_samples(fix),
        sabotage=HDB_SAB_MR_TWO_WAY,
    )
    var asym = 0
    for i in range(m):
        for j in range(i + 1, m):
            if sab.mr[i * m + j].to_bits() != sab.mr[j * m + i].to_bits():
                asym += 1
    if asym == 0:
        raise Error(
            "check_hdbscan_sabotages: HDB_SAB_MR_TWO_WAY left the mutual"
            " reachability matrix SYMMETRIC on " + hfixture_name(fix)
            + ". Dropping core_dists[col] from the three-way max must"
            " break symmetry unless every core distance is equal, which"
            " would make this fixture unable to test the gate"
        )
    print(
        "check_hdbscan_sabotages OK: HDB_SAB_MR_TWO_WAY FAILED the"
        " mutual-reachability gate as required: " + String(asym) + " of "
        + String(m * (m - 1) // 2) + " pairs became asymmetric"
    )

    # HDB_SAB_CORE_KTH_PLUS_ONE: the core distance gate.
    var fb = HFIX_BLOBS
    var mb = hfixture_n(fb)
    var db = hfixture_d(fb)
    var vb = hfixture_as_list(fb)
    var good = _staged_graph(ctx, vb, mb, db, hfixture_min_samples(fb))
    var kth = _staged_graph(
        ctx, vb, mb, db, hfixture_min_samples(fb),
        sabotage=HDB_SAB_CORE_KTH_PLUS_ONE,
    )
    var kth_moved = _bits_differ(good.core, kth.core)
    if kth_moved == 0:
        raise Error(
            "check_hdbscan_sabotages: HDB_SAB_CORE_KTH_PLUS_ONE moved NO"
            " core distance on " + hfixture_name(fb) + ". Reading the"
            " (k+1)-th neighbour instead of the k-th must move a value"
            " unless every row's k-th and (k+1)-th distances are equal"
        )
    print(
        "check_hdbscan_sabotages OK: HDB_SAB_CORE_KTH_PLUS_ONE FAILED the"
        " core-distance gate as required: " + String(kth_moved) + " of "
        + String(mb) + " core distances moved"
    )

    # HDB_SAB_CONDENSE_DFS: the condensed-tree gate.
    var base = _fit_plain(ctx, fb)
    var dfs = _fit_plain(ctx, fb, HDB_SAB_CONDENSE_DFS)
    var dfs_moved = _i32_differ(base.condensed.children, dfs.condensed.children)
    if dfs_moved == 0 and base.condensed.n_edges == dfs.condensed.n_edges:
        raise Error(
            "check_hdbscan_sabotages: HDB_SAB_CONDENSE_DFS left the"
            " condensed tree identical on " + hfixture_name(fb)
            + ". A depth-first traversal assigns next_label in a"
            " different order, so on any tree deeper than two levels the"
            " numbering must move"
        )
    print(
        "check_hdbscan_sabotages OK: HDB_SAB_CONDENSE_DFS FAILED the"
        " condensed-tree gate as required: " + String(dfs_moved)
        + " condensed children moved (device " + String(base.condensed.n_edges)
        + " edges, sabotaged " + String(dfs.condensed.n_edges) + ")"
    )

    # HDB_SAB_EOM_NO_UPDATE: the selection gate. SWEPT OVER EVERY FIXTURE
    # and required to move on AT LEAST ONE, rather than pinned to a single
    # fixture. Dropping the write-back can only change an ancestor's
    # comparison where some node is DESELECTED in the first place, and
    # whether a given fixture's tree has one is a property of the fixture.
    # A named-fixture assertion would therefore be an assertion about the
    # fixture; a sweep is an assertion about the pin.
    var eom_total = 0
    var eom_where = String("")
    for fg in range(HFIX_COUNT):
        var eom_base = _fit_plain(ctx, fg)
        var eom_sab = _fit_plain(ctx, fg, HDB_SAB_EOM_NO_UPDATE)
        var sel_moved = _i32_differ(eom_base.is_cluster, eom_sab.is_cluster)
        var lab_moved = _i32_differ(eom_base.labels, eom_sab.labels)
        var stab_m = _bits_differ(
            eom_base.tree_stabilities, eom_sab.tree_stabilities
        )
        if sel_moved != 0 or lab_moved != 0 or stab_m != 0:
            eom_total += 1
            eom_where += (
                " " + hfixture_name(fg) + "(" + String(sel_moved) + " flags, "
                + String(lab_moved) + " labels, " + String(stab_m)
                + " stabilities)"
            )
    if eom_total == 0:
        raise Error(
            "check_hdbscan_sabotages: HDB_SAB_EOM_NO_UPDATE moved nothing"
            " on ANY of the " + String(HFIX_COUNT) + " fixtures. Dropping"
            " stability[node] = subtree_stability (select.cuh:227) must"
            " change an ancestor's comparison wherever a node is"
            " deselected, so either no fixture deselects a node -- and"
            " none of them then tests Excess of Mass at all -- or the"
            " write-back is not reached"
        )
    print(
        "check_hdbscan_sabotages OK: HDB_SAB_EOM_NO_UPDATE FAILED the"
        " selection gate as required on " + String(eom_total) + " of "
        + String(HFIX_COUNT) + " fixtures:" + eom_where
    )

    # HDB_SAB_STABILITY_DESCENDING: a summation order. RECORDED.
    var desc = _fit_plain(ctx, fb, HDB_SAB_STABILITY_DESCENDING)
    var stab_moved = _bits_differ(base.tree_stabilities, desc.tree_stabilities)
    print(
        "check_hdbscan_sabotages RECORDED [" + _mode_name() + "]:"
        " HDB_SAB_STABILITY_DESCENDING moved " + String(stab_moved)
        + " of " + String(len(base.tree_stabilities)) + " stabilities on "
        + hfixture_name(fb) + " (a summation order; a fixture whose"
        " segments are all one or two terms cannot separate the two"
        " directions, and that is a property of the fixture)"
    )

    # HDB_SAB_LAMBDA_STD_DIV: Apple's divide is correctly rounded. REPORT.
    var sdiv = _fit_plain(ctx, fb, HDB_SAB_LAMBDA_STD_DIV)
    var lam_moved = _bits_differ(
        base.condensed.lambdas, sdiv.condensed.lambdas
    )
    print(
        "check_hdbscan_sabotages RECORDED [" + _mode_name() + "]:"
        " HDB_SAB_LAMBDA_STD_DIV moved " + String(lam_moved) + " of "
        + String(base.condensed.n_edges) + " condensed lambdas on "
        + hfixture_name(fb) + " (Apple's divide is correctly rounded, so"
        " 0 is the expected answer HERE; this is the arm that would move"
        " on a column whose divide is approximate -- the DEVIATION 258"
        " shape, one operation over)"
    )

    # HDB_SAB_HW_MAX: the planted signed-zero max. RECORDED.
    var pm = 4
    var pdists = List[Float32](capacity=pm * pm)
    for i in range(pm):
        for j in range(pm):
            if i == j:
                pdists.append(FLOAT32_MAX)
            else:
                pdists.append(bitcast[DType.float32](NEG_ZERO_BITS))
    var pcore = List[Float32](capacity=pm)
    pcore.append(Float32(0.0))
    pcore.append(bitcast[DType.float32](NEG_ZERO_BITS))
    pcore.append(Float32(0.0))
    pcore.append(bitcast[DType.float32](NEG_ZERO_BITS))
    var dd = _upload(ctx, pdists, 0, Float32(0.0))
    var dc = _upload(ctx, pcore, 0, Float32(0.0))
    var m_good = ctx.enqueue_create_buffer[DType.float32](pm * pm)
    var m_sab = ctx.enqueue_create_buffer[DType.float32](pm * pm)
    ctx.synchronize()
    mutual_reachability_dense(
        ctx, m_good, dd, dc, pm, Float32(1.0), MR_TPB, HDB_SAB_NONE
    )
    mutual_reachability_dense(
        ctx, m_sab, dd, dc, pm, Float32(1.0), MR_TPB, HDB_SAB_HW_MAX
    )
    var hg = _copy_f32(ctx, m_good, pm * pm)
    var hs = _copy_f32(ctx, m_sab, pm * pm)
    var hw_moved = _bits_differ(hg, hs)
    var first_pair = String("none")
    for i in range(pm * pm):
        if hg[i].to_bits() != hs[i].to_bits():
            first_pair = _hex32(hg[i]) + " -> " + _hex32(hs[i])
            break
    print(
        "check_hdbscan_sabotages RECORDED [" + _mode_name() + "]:"
        " HDB_SAB_HW_MAX moved " + String(hw_moved) + " of "
        + String(pm * pm) + " PLANTED (+0, -0) cells; first "
        + first_pair + ". IDENTITY_PATHS row 39: the stdlib max returns"
        " the SECOND operand on Apple and the IEEE-2019 maximum on NVIDIA"
        " and AMD, so this arm's answer is a fact about the toolchain."
        " LLVM may also fold a maxnum into a compare-select, in which"
        " case 0 is a fact about the fold and not about the pin"
    )
    _ = dd^
    _ = dc^
    _ = m_good^
    _ = m_sab^

    # HDB_SAB_SKIP_GUARDS: what DEVIATION 1607 keeps out. RECORDED.
    var inf_vals = vb.copy()
    for f in range(db):
        inf_vals[5 * db + f] = Float32(1e20)
        inf_vals[6 * db + f] = Float32(1e20)
    var skipped = String("the guard-skipped run RAISED anyway")
    var t = IdentityTrace.disabled()
    try:
        var bad = _fit(
            ctx, inf_vals, mb, db, _params_for(fb), t,
            sabotage=HDB_SAB_SKIP_GUARDS,
        )
        var n_nan = 0
        var payload = String("none")
        for i in range(mb):
            var v = bad.core_dists[i]
            if v != v:
                n_nan += 1
                if payload == "none":
                    payload = _hex32(v)
        skipped = (
            String(n_nan) + " of " + String(mb) + " core distances are NaN"
            " with THIS DEVICE'S payload " + payload
        )
    except e:
        skipped = (
            "the run still raised, from a guard this arm does not skip: "
            + String(e)
        )
    print(
        "check_hdbscan_sabotages RECORDED [" + _mode_name() + "]:"
        " HDB_SAB_SKIP_GUARDS -- " + skipped + ". FACT 2 (IDENTITY_PATHS"
        " row 39): Apple 0x7fc00000, NVIDIA 0x7fffffff, AMD 0xffc00000."
        " That is the byte DEVIATION 1607 keeps out of hdbscan.core_dists"
        " and hdbscan.mr.dists"
    )


def main() raises:
    print("hdbscan_check mode=" + _mode_name())
    check_hdbscan_refusals()
    check_core_distances_vs_oracle()
    check_mutual_reachability_ties()
    check_condensed_tree_vs_oracle()
    check_stabilities_vs_oracle()
    check_labels_vs_oracle()
    check_permutation_invariance()
    check_launch_invariance()
    check_card_is_emitted()
    check_hdbscan_signed_zero_inputs()
    check_hdbscan_selection_leaf()
    check_hdbscan_float64_reference()
    check_hdbscan_sabotages()
    print("hdbscan_check mode=" + _mode_name() + " ALL OK")
