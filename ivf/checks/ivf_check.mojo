# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The IVF-FLAT gates.

    tools/with_build_lock.sh     pixi run mojo run -I . ivf/checks/ivf_check.mojo
    tools/with_identical_mode.sh pixi run mojo run -I . ivf/checks/ivf_check.mojo

Every check prints the mode this binary COMPILED in. Under `IDENTICAL` the
device-vs-oracle and device-vs-brute-force comparisons are ASSERTIONS (the
distances come from `pinned_distance_tile.mojo`, one thread per cell, and
the selection from the composite-key selector, so a host replay is the same
arithmetic in the same order). Under `FAST` those same comparisons are
REPORTS: the distances are MAX's matmul, whose tile shape and k-split are
per-shape and per-vendor, and the selector is RAFT's, whose tie back-fill
is an atomic arrival order. Everything structural -- the refusals, the
permutation invariants, the carry, the empty list, the tie rules, the card
-- asserts in BOTH modes, because none of it is a float question.

CHECKS, in the order `main` runs them

  check_nprobe_equals_nlists_is_brute_force   THE HEADLINE, and the first
                                              one written. n_probe ==
                                              n_lists against
                                              `neighbors/`'s own tiled
                                              brute force, distances AND
                                              indices, bit for bit
  check_ivf_refusals                          IVF-PQ, HNSW, CAGRA, an
                                              unsupported metric, n_probe >
                                              n_lists, n_lists > n_rows,
                                              non-finite input, the three
                                              unported index_params, and
                                              k > SELECT_BLOCK: each
                                              RAISES BY NAME
  check_list_layout_and_index_carry           the layout is a PERMUTATION
                                              and the carry survives it;
                                              SABOTAGED with the within-
                                              list position
  check_empty_list                            a list that comes out empty
                                              corrupts nothing; SABOTAGED
                                              with a chunk scan that counts
                                              it as one
  check_assignment_ties                       an exactly equidistant point
                                              goes to the LOWER list id,
                                              and reversing the centroid
                                              order does not move which
                                              list it is; the same rule on
                                              the coarse probe side,
                                              SABOTAGED
  check_quantizer_is_reproducible             HAZARD 2: the centroid set is
                                              the same bytes twice before
                                              any search result is called
                                              reproducible
  check_search_vs_oracle                      per query, per neighbour, bit
                                              for bit under IDENTICAL, on
                                              the hashed, duplicate and
                                              signed-zero fixtures
  check_recall_is_reported                    recall at several n_probes,
                                              printed as a REPORT
  check_launch_invariance                     two threads-per-block
                                              choices, a padded workspace,
                                              and one query alone versus
                                              inside a batch
  check_card_is_emitted                       the sixteen stages, and two
                                              runs record-identical

SABOTAGES: `ivf/checks/sabotage_layout.mojo` carries five arms and
`hierarchy/checks/sabotage_tile.mojo::sabotage_distance_tile_kernel` is
IMPORTED for the sixth rather than copied a third time. The table of what
each one did belongs in `ivf/README.md` and is filled in by whoever RUNS
this file. The five layout arms were driven on the Apple M4 on 2026-08-25;
their per-arm output is not yet transcribed into that table, and a sabotage
table written from expectation instead of from output is the exact thing
`[[reached-but-inert]]` is about.
"""

from std.memory import bitcast

from max.gpu.host import DeviceBuffer, DeviceContext

from cluster.impl.cluster.detail.kmeans_common import metric_is_sqrt
from cluster.impl.cluster.kmeans import predict
from cluster.impl.cluster.kmeans_params import (
    INIT_KMEANS_PLUS_PLUS,
    KMeansParams,
    METRIC_COSINE_EXPANDED,
    METRIC_L2_EXPANDED,
)
from core.identity_trace import IdentityTrace, first_divergence, read_trace_lines
from ivf.checks.ivf_fixture import (
    ivf_duplicate_fixture,
    ivf_equidistant_centers,
    ivf_equidistant_centers_reversed,
    ivf_equidistant_points,
    ivf_index_fixture,
    ivf_planted_centers,
    ivf_planted_labels,
    ivf_query_fixture,
    ivf_query_source_row,
    ivf_signed_zero_fixture,
)
from ivf.checks.ivf_oracle import (
    OracleSearchResult,
    oracle_brute_force,
    oracle_expanded_distance,
    oracle_ivf_search,
    oracle_row_norms,
    oracle_select_k,
    recall_against,
    reference_assignment_f64,
    reference_distance_f64,
    reference_nearest_f64,
)
from ivf.checks.list_layout import (
    ListLayout,
    build_list_layout,
    gather_candidate_indices,
    merge_probed_lists,
)
from ivf.checks.sabotage_layout import (
    IVF_SAB_CARRY_POSITION,
    IVF_SAB_EMPTY_COUNTS_ONE,
    IVF_SAB_LIST_ARRIVAL_ORDER,
    IVF_SAB_MERGE_PROBE_ORDER,
    IVF_SAB_PROBE_TIE_HIGH,
    sabotage_build_list_layout,
    sabotage_calc_chunk_indices,
    sabotage_gather_candidate_indices,
    sabotage_merge_probed_lists,
    sabotage_name,
    sabotage_sort_probe_slots,
)
from ivf.impl.neighbors.ivf_common import calc_chunk_indices
from ivf.impl.neighbors.ivf_flat.ivf_flat_build import (
    compute_row_norms,
    download_f32,
    ivf_flat_build,
    upload_f32,
)
from ivf.impl.neighbors.ivf_flat.ivf_flat_index import (
    IvfFlatIndex,
    IvfFlatIndexParams,
    IvfFlatSearchParams,
    ivf_metric_from_name,
    ivf_index_params_validate,
    ivf_refuse_algorithm,
    ivf_search_params_validate,
    ivf_validate_data,
)
from ivf.impl.neighbors.ivf_flat.ivf_flat_search import (
    IVF_EXPAND_TPB,
    IvfSearchResult,
    ivf_flat_search_traced,
    sort_slots_by_distance_then_index,
)
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL, numeric_mode_name
from neighbors.estimator import knn_search
from neighbors.checks.pinned_distance_tile import PINNED_TILE_TPB
from neighbors.impl.matrix.detail.select_radix import SELECT_BLOCK
from neighbors.impl.neighbors.detail.knn_brute_force import KNN_METHOD_TILED


comptime IDENTICAL = GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL

comptime N_ROWS = 320
"""Larger than `SELECT_BLOCK` ON PURPOSE, so `check_ivf_refusals` can ask
for `k = SELECT_BLOCK + 1` and reach the SELECTOR'S refusal rather than the
candidate-count one that would fire first on a smaller index. A refusal
check that reaches a different refusal than the one it names is a check
about the wrong sentence."""

comptime N_QUERIES = 24
comptime DIM = 6
comptime N_LISTS = 6
comptime K = 5

comptime SCRATCH = "/tmp"
"""Where the per-check cards go. `/tmp` so the check runs on any box."""


def _mode_name() -> String:
    """The build's tier, from the ONE definition of it.

    Delegates to `numeric_mode_name()` since 2026-08-29; see the note
    on that function. A local two-way IDENTICAL-or-FAST answers "FAST"
    for a DETERMINISTIC build, which mislabels every line the driver
    prints.
    """
    return numeric_mode_name()


def _hex32(v: Float32) -> String:
    comptime DIGITS = "0123456789abcdef"
    var u = bitcast[DType.uint32](v)
    var out = String("0x")
    for i in range(8):
        var nib = Int((u >> UInt32(28 - 4 * i)) & UInt32(0xF))
        out += String(DIGITS[byte=nib])
    return out


@always_inline
def _same_bits(a: Float32, b: Float32) -> Bool:
    """Bitwise equality, which is NOT `a == b`: `+0.0 == -0.0` compares
    true and this lane has a fixture built out of that pair (row 39)."""
    return bitcast[DType.uint32](a) == bitcast[DType.uint32](b)


def _build_index(
    ctx: DeviceContext,
    x: List[Float32],
    n_rows: Int,
    dim: Int,
    n_lists: Int,
    seed: UInt64,
    metric: Int = METRIC_L2_EXPANDED,
) raises -> IvfFlatIndex:
    """A real fitted index, with the trace explicitly OFF.

    `IdentityTrace.disabled()` and not `IdentityTrace()`: a check whose
    behaviour depends on whether the operator happens to have
    `MOJOLEARN_IDENTITY_TRACE` exported is a check that passes or fails for
    reasons outside itself. `core/identity_trace.mojo` says exactly that
    where it defines `disabled()`.
    """
    var params = IvfFlatIndexParams.default()
    params.n_lists = n_lists
    params.kmeans_n_iters = 20
    params.kmeans_trainset_fraction = Float64(1.0)
    params.metric = metric
    params.seed = seed
    var trace = IdentityTrace.disabled()
    return ivf_flat_build(ctx, trace, params, x, n_rows, dim)


def _plant_index(
    ctx: DeviceContext,
    x: List[Float32],
    labels: List[UInt32],
    centers: List[Float32],
    n_rows: Int,
    dim: Int,
    n_lists: Int,
    metric: Int = METRIC_L2_EXPANDED,
) raises -> IvfFlatIndex:
    """An index whose LAYOUT is planted rather than fitted.

    For the degenerate cases the quantizer cannot be relied on to produce
    -- an empty list, a chosen carry -- because a check that only runs when
    k-means happens to cooperate cannot state that it ran. The centres are
    still real centres and the centre norms still come off
    `core/row_norms.mojo`, so the search path below is the shipped one.
    """
    var layout = build_list_layout(labels, x, n_rows, dim, n_lists)
    var dc = upload_f32(ctx, centers)
    var dn = ctx.enqueue_create_buffer[DType.float32](n_lists)
    ctx.synchronize()
    compute_row_norms(ctx, dc, dn, n_lists, dim, metric_is_sqrt(metric))
    ctx.synchronize()
    var norms = download_f32(ctx, dn, n_lists)
    _ = dc^
    _ = dn^
    return IvfFlatIndex(
        n_lists,
        dim,
        n_rows,
        metric,
        centers.copy(),
        norms^,
        layout.offsets.copy(),
        layout.list_indices.copy(),
        layout.list_data.copy(),
        labels.copy(),
    )


def _search(
    ctx: DeviceContext,
    index: IvfFlatIndex,
    queries: List[Float32],
    n_queries: Int,
    k: Int,
    n_probes: Int,
    tile_tpb: Int = PINNED_TILE_TPB,
    expand_tpb: Int = IVF_EXPAND_TPB,
) raises -> IvfSearchResult:
    var sp = IvfFlatSearchParams(n_probes)
    var trace = IdentityTrace.disabled()
    return ivf_flat_search_traced(
        ctx, trace, index, sp, queries, n_queries, k, tile_tpb, expand_tpb
    )


def _knn_reference(
    ctx: DeviceContext,
    x: List[Float32],
    n_rows: Int,
    queries: List[Float32],
    n_queries: Int,
    dim: Int,
    k: Int,
) raises -> OracleSearchResult:
    """`neighbors/estimator.mojo::knn_search`, THE TILED ARM BY NAME.

    Not `KNN_METHOD_AUTO`. Under `IDENTICAL` AUTO already pins to tiled
    (DEVIATION 509), but under `FAST` it chooses by shape, and a reference
    whose arm depends on how many queries this check happens to pass is a
    reference that tests a different thing in the two modes.
    `PORTING_RULES.md` rule 8: the harness names the kernel it ran.

    `return_sqrt=False` because the index under test carries
    `METRIC_L2_EXPANDED`, which is SQUARED distances on both sides.
    """
    var hx = ctx.enqueue_create_host_buffer[DType.float32](n_rows * dim)
    var hq = ctx.enqueue_create_host_buffer[DType.float32](n_queries * dim)
    var hd = ctx.enqueue_create_host_buffer[DType.float32](n_queries * k)
    var hi = ctx.enqueue_create_host_buffer[DType.uint32](n_queries * k)
    ctx.synchronize()
    for i in range(n_rows * dim):
        hx.unsafe_ptr().unsafe_store(i, x[i])
    for i in range(n_queries * dim):
        hq.unsafe_ptr().unsafe_store(i, queries[i])

    var used_tile = knn_search(
        ctx,
        hx.unsafe_ptr(),
        n_rows,
        hq.unsafe_ptr(),
        n_queries,
        dim,
        k,
        hd.unsafe_ptr(),
        hi.unsafe_ptr(),
        False,
        n_queries,
        KNN_METHOD_TILED,
    )
    _ = used_tile

    var out_d = List[Float32]()
    var out_i = List[UInt32]()
    var counts = List[Int32]()
    for i in range(n_queries * k):
        out_d.append(hd.unsafe_ptr().unsafe_load(i))
        out_i.append(hi.unsafe_ptr().unsafe_load(i))
    for _ in range(n_queries):
        counts.append(Int32(n_rows))
    _ = hx^
    _ = hq^
    _ = hd^
    _ = hi^
    return OracleSearchResult(out_d^, out_i^, counts^)


# =====================================================================
# THE HEADLINE GATE. Written first, for the reason the brief gives: it is
# the strongest statement this lane can make, and everything else is a
# statement about how it is reached.
# =====================================================================


def check_nprobe_equals_nlists_is_brute_force() raises:
    """`n_probe == n_lists` is EXACTLY the brute force, both arrays.

    WHY IT CAN BE BIT FOR BIT AND NOT MERELY CLOSE, in four steps, each of
    which is a property some other check gates:

      1. every list is probed, so the candidate set is the whole index;
      2. `merge_probed_lists` emits it ASCENDING BY ORIGINAL INDEX
         (DEVIATION 1786), so the candidate row's position `p` holds the
         distance to original row `p` -- the same row `knn_search`'s
         distance row holds at position `p`;
      3. the candidate norms are `row_norm_kernel` over the PERMUTED matrix
         read back through the carry, and that kernel is one block per row,
         so they are the same floats `knn_search` computed over the
         unpermuted one;
      4. the distances go through the same kernel and the selection through
         the same selector at the same `buf_len`.

    So the two computations are not similar, they are the same one, and any
    difference is a defect rather than a rounding. That is why this is an
    ASSERTION under `IDENTICAL`. Under `FAST` step 4 stops holding -- the
    vendor matmul is called at `m = 1` here and at `m = n_queries` there,
    and a matmul is entitled to a different k-split per shape -- so the
    same comparison is a REPORT and the line says so.
    """
    var ctx = DeviceContext()
    var x = ivf_index_fixture(N_ROWS, DIM, 3)
    var q = ivf_query_fixture(x, N_ROWS, N_QUERIES, DIM, 3)
    var index = _build_index(ctx, x, N_ROWS, DIM, N_LISTS, UInt64(7))

    var got = _search(ctx, index, q, N_QUERIES, K, N_LISTS)
    var want = _knn_reference(ctx, x, N_ROWS, q, N_QUERIES, DIM, K)

    var moved_d = 0
    var moved_i = 0
    var first = String("")
    for i in range(N_QUERIES * K):
        if not _same_bits(got.distances[i], want.distances[i]):
            moved_d += 1
            if first == "":
                first = (
                    "slot " + String(i) + " ivf " + _hex32(got.distances[i])
                    + " vs brute " + _hex32(want.distances[i])
                )
        if got.indices[i] != want.indices[i]:
            moved_i += 1
            if first == "":
                first = (
                    "slot " + String(i) + " ivf idx "
                    + String(got.indices[i]) + " vs brute idx "
                    + String(want.indices[i])
                )
    for qi in range(N_QUERIES):
        if Int(got.n_candidates[qi]) != N_ROWS:
            raise Error(
                "check_nprobe_equals_nlists_is_brute_force: query "
                + String(qi)
                + " scored "
                + String(got.n_candidates[qi])
                + " candidates at n_probe == n_lists, expected the whole"
                " index ("
                + String(N_ROWS)
                + "). The reduction cannot hold if the candidate set is not"
                " the index."
            )

    comptime if IDENTICAL:
        if moved_d != 0 or moved_i != 0:
            raise Error(
                "check_nprobe_equals_nlists_is_brute_force FAILED: "
                + String(moved_d)
                + " distances and "
                + String(moved_i)
                + " indices differ from the tiled brute force at n_probe =="
                " n_lists. First: "
                + first
            )
        print(
            "check_nprobe_equals_nlists_is_brute_force OK [IDENTICAL]: "
            + String(N_QUERIES * K)
            + " slots, distances AND indices bit for bit against"
            " neighbors/ knn_search's TILED arm; every query scored all "
            + String(N_ROWS)
            + " candidates"
        )
    else:
        print(
            "check_nprobe_equals_nlists_is_brute_force REPORT [FAST]: "
            + String(moved_d)
            + " of "
            + String(N_QUERIES * K)
            + " distances and "
            + String(moved_i)
            + " indices differ from the tiled brute force. A REPORT and not"
            " an assertion: the FAST distances come from MAX's matmul,"
            " called at m=1 here and m=n_queries there, and the FAST"
            " selector resolves a tie by atomic arrival."
        )
    _ = index^

    # DEVIATION 1946: the context dies LAST, after every value built on it.
    # Mojo frees at LAST USE, so without this the buffer releases above run
    # against a context that is already gone. On sm_89 the next GPU call in
    # the process then never returns (GPU idle, host threads in futex wait);
    # Apple and AMD do not show it, which is how it stayed latent here.
    _ = ctx^


# =====================================================================
# REFUSALS
# =====================================================================


def check_ivf_refusals() raises:
    var refused = 0

    var algos: List[String] = [
        "ivf_pq", "hnsw", "cagra", "ivf_sq", "ivf_rabitq", "scann",
        "vamana", "nn_descent",
    ]
    for ai in range(len(algos)):
        var name = algos[ai].copy()
        var raised = False
        try:
            ivf_refuse_algorithm(name)
        except e:
            raised = True
            refused += 1
            print("  refused  algorithm=" + name + ": " + String(e))
        if not raised:
            raise Error(
                "check_ivf_refusals: algorithm '" + name + "' did NOT raise"
            )
    # ivf_flat itself must NOT raise, or the refusal list is a wall.
    ivf_refuse_algorithm(String("ivf_flat"))

    var metrics: List[String] = ["cosine", "inner_product", "hamming"]
    for mi in range(len(metrics)):
        var name = metrics[mi].copy()
        var raised = False
        try:
            _ = ivf_metric_from_name(name)
        except e:
            raised = True
            refused += 1
            print("  refused  metric=" + name + ": " + String(e))
        if not raised:
            raise Error(
                "check_ivf_refusals: metric '" + name + "' did NOT raise"
            )
    _ = ivf_metric_from_name(String("l2_expanded"))
    _ = ivf_metric_from_name(String("l2_sqrt_expanded"))

    # The index_params refusals, one per parameter, each from a params
    # struct that differs from the default in exactly one field.
    var base = IvfFlatIndexParams.default()
    base.n_lists = N_LISTS
    base.kmeans_trainset_fraction = Float64(1.0)

    var cases = List[String]()
    var params_list = List[IvfFlatIndexParams]()

    var p_metric = base
    p_metric.metric = METRIC_COSINE_EXPANDED
    cases.append(String("metric=CosineExpanded"))
    params_list.append(p_metric)

    var p_frac = base
    p_frac.kmeans_trainset_fraction = Float64(0.5)
    cases.append(String("kmeans_trainset_fraction=0.5 (DEVIATION 1781)"))
    params_list.append(p_frac)

    var p_adapt = base
    p_adapt.adaptive_centers = True
    cases.append(String("adaptive_centers=True"))
    params_list.append(p_adapt)

    var p_cons = base
    p_cons.conservative_memory_allocation = True
    cases.append(String("conservative_memory_allocation=True"))
    params_list.append(p_cons)

    var p_add = base
    p_add.add_data_on_build = False
    cases.append(String("add_data_on_build=False"))
    params_list.append(p_add)

    var p_lists = base
    p_lists.n_lists = N_ROWS + 1
    cases.append(String("n_lists > n_rows (their :404)"))
    params_list.append(p_lists)

    for i in range(len(cases)):
        var raised = False
        try:
            ivf_index_params_validate(params_list[i], N_ROWS, DIM)
        except e:
            raised = True
            refused += 1
            print("  refused  " + cases[i] + ": " + String(e))
        if not raised:
            raise Error(
                "check_ivf_refusals: " + cases[i] + " did NOT raise"
            )
    # and the default shape validates
    ivf_index_params_validate(base, N_ROWS, DIM)

    # n_probes: zero, and greater than n_lists (theirs CLAMPS, ours refuses)
    var probe_cases = List[Int]()
    probe_cases.append(0)
    probe_cases.append(N_LISTS + 1)
    for i in range(len(probe_cases)):
        var raised = False
        try:
            ivf_search_params_validate(
                IvfFlatSearchParams(probe_cases[i]), N_LISTS, N_QUERIES, K
            )
        except e:
            raised = True
            refused += 1
            print(
                "  refused  n_probes=" + String(probe_cases[i]) + ": "
                + String(e)
            )
        if not raised:
            raise Error(
                "check_ivf_refusals: n_probes=" + String(probe_cases[i])
                + " did NOT raise"
            )
    ivf_search_params_validate(
        IvfFlatSearchParams(N_LISTS), N_LISTS, N_QUERIES, K
    )

    # Non-finite and over-magnitude input.
    var bad = ivf_index_fixture(4, 2, 5)
    bad[3] = bitcast[DType.float32](UInt32(0x7FC00000))
    var raised_nan = False
    try:
        ivf_validate_data(bad, 4, 2, String("dataset"))
    except e:
        raised_nan = True
        refused += 1
        print("  refused  NaN in the dataset: " + String(e))
    if not raised_nan:
        raise Error("check_ivf_refusals: a NaN in the dataset did NOT raise")

    var big = ivf_index_fixture(4, 2, 5)
    big[1] = bitcast[DType.float32](UInt32(0x7F800000))
    var raised_inf = False
    try:
        ivf_validate_data(big, 4, 2, String("dataset"))
    except e:
        raised_inf = True
        refused += 1
        print("  refused  +inf in the dataset: " + String(e))
    if not raised_inf:
        raise Error("check_ivf_refusals: a +inf in the dataset did NOT raise")

    # k > SELECT_BLOCK, at the search boundary.
    var ctx = DeviceContext()
    var x = ivf_index_fixture(N_ROWS, DIM, 5)
    var q = ivf_query_fixture(x, N_ROWS, 2, DIM, 5)
    var index = _build_index(ctx, x, N_ROWS, DIM, N_LISTS, UInt64(1))
    var raised_k = False
    try:
        _ = _search(ctx, index, q, 2, SELECT_BLOCK + 1, N_LISTS)
    except e:
        raised_k = True
        refused += 1
        print(
            "  refused  k = SELECT_BLOCK + 1 = "
            + String(SELECT_BLOCK + 1)
            + ": "
            + String(e)
        )
    if not raised_k:
        raise Error(
            "check_ivf_refusals: k > SELECT_BLOCK did NOT raise"
        )

    # A probe set too small to supply k (DEVIATION 1794), PLANTED so it is
    # reached rather than hoped for. Row 0 sits alone in list 0, whose
    # centre IS row 0; the query IS row 0, so the coarse selection at
    # n_probes = 1 probes list 0 and finds one candidate against k = 5.
    var tiny_labels = List[UInt32]()
    tiny_labels.append(UInt32(0))
    for _ in range(N_ROWS - 1):
        tiny_labels.append(UInt32(1))
    var tiny_centers = List[Float32]()
    for f in range(DIM):
        tiny_centers.append(x[f])
    for f in range(DIM):
        tiny_centers.append(x[DIM + f])
    # THE CHECK REFUSES ITSELF IF THE COARSE SELECTION IS NOT FORCED.
    var q0 = List[Float32]()
    for f in range(DIM):
        q0.append(x[f])
    var dc0 = reference_distance_f64(q0, 0, tiny_centers, 0, DIM, False)
    var dc1 = reference_distance_f64(q0, 0, tiny_centers, 1, DIM, False)
    if dc0 >= dc1:
        raise Error(
            "check_ivf_refusals: the planted short-list fixture does not"
            " force list 0 (" + String(dc0) + " vs " + String(dc1)
            + "), so DEVIATION 1794's refusal would not be reached"
        )
    var tiny = _plant_index(
        ctx, x, tiny_labels, tiny_centers, N_ROWS, DIM, 2
    )
    var raised_small = False
    try:
        _ = _search(ctx, tiny, q0, 1, K, 1)
    except e:
        raised_small = True
        refused += 1
        print("  refused  probed lists hold fewer than k: " + String(e))
    if not raised_small:
        raise Error(
            "check_ivf_refusals: a single probed list holding one vector"
            " supplied k = " + String(K) + " neighbours without raising"
            " (DEVIATION 1794)"
        )
    _ = tiny^

    print(
        "check_ivf_refusals OK ["
        + _mode_name()
        + "]: "
        + String(refused)
        + " refusals by name; ivf_flat, both ported metrics and the default"
        " parameter set all resolve"
    )
    _ = index^

    # DEVIATION 1946: the context dies LAST, after every value built on it.
    # Mojo frees at LAST USE, so without this the buffer releases above run
    # against a context that is already gone. On sm_89 the next GPU call in
    # the process then never returns (GPU idle, host threads in futex wait);
    # Apple and AMD do not show it, which is how it stayed latent here.
    _ = ctx^


# =====================================================================
# THE LAYOUT AND THE CARRY
# =====================================================================


def check_list_layout_and_index_carry() raises:
    """The layout is a PERMUTATION, and the ORIGINAL index survives it.

    Three claims, then the sabotage.

      (a) `list_offsets` partitions `[0, n_rows)`: it ascends, starts at 0,
          ends at `n_rows`.
      (b) `list_indices` is a PERMUTATION of `[0, n_rows)` -- every row
          appears exactly once -- and it ASCENDS within every list
          (DEVIATION 1783).
      (c) `list_data[slot]` is `x[list_indices[slot]]` BIT FOR BIT. This is
          the carry: a permutation changes nothing only if the thing that
          travels with the row is the row's own name.

    THE SABOTAGE. `IVF_SAB_CARRY_POSITION` carries the WITHIN-LIST
    POSITION instead. The distances are untouched by it -- the same
    candidates are scored in the same order -- so the only thing that moves
    is WHICH ROWS THE ANSWER NAMES, which is why this bug survives a
    distance-based review. The check selects the same k candidates through
    the oracle and maps them through both carries; the two must disagree,
    and the good one must equal the fitted search's answer.
    """
    var ctx = DeviceContext()
    var x = ivf_index_fixture(N_ROWS, DIM, 9)
    var q = ivf_query_fixture(x, N_ROWS, N_QUERIES, DIM, 9)
    var index = _build_index(ctx, x, N_ROWS, DIM, N_LISTS, UInt64(11))

    # (a)
    if Int(index.list_offsets[0]) != 0:
        raise Error("check_list_layout_and_index_carry: offsets[0] != 0")
    if Int(index.list_offsets[N_LISTS]) != N_ROWS:
        raise Error(
            "check_list_layout_and_index_carry: offsets[n_lists] = "
            + String(index.list_offsets[N_LISTS])
            + ", expected n_rows = "
            + String(N_ROWS)
        )
    for l in range(N_LISTS):
        if index.list_offsets[l + 1] < index.list_offsets[l]:
            raise Error(
                "check_list_layout_and_index_carry: offsets descend at list"
                " " + String(l)
            )

    # (b)
    var seen = List[Int]()
    for _ in range(N_ROWS):
        seen.append(0)
    for s in range(N_ROWS):
        var orig = Int(index.list_indices[s])
        if orig < 0 or orig >= N_ROWS:
            raise Error(
                "check_list_layout_and_index_carry: slot "
                + String(s)
                + " carries "
                + String(orig)
            )
        seen[orig] += 1
    for r in range(N_ROWS):
        if seen[r] != 1:
            raise Error(
                "check_list_layout_and_index_carry: original row "
                + String(r)
                + " appears "
                + String(seen[r])
                + " times in the layout; it is not a permutation"
            )
    for l in range(N_LISTS):
        var s = Int(index.list_offsets[l]) + 1
        while s < Int(index.list_offsets[l + 1]):
            if index.list_indices[s] <= index.list_indices[s - 1]:
                raise Error(
                    "check_list_layout_and_index_carry: list "
                    + String(l)
                    + " does not ascend in the carried index at slot "
                    + String(s)
                    + " (DEVIATION 1783)"
                )
            s += 1

    # (c)
    for s in range(N_ROWS):
        var orig = Int(index.list_indices[s])
        for f in range(DIM):
            if not _same_bits(
                index.list_data[s * DIM + f], x[orig * DIM + f]
            ):
                raise Error(
                    "check_list_layout_and_index_carry: slot "
                    + String(s)
                    + " column "
                    + String(f)
                    + " does not carry row "
                    + String(orig)
                    + "'s bits"
                )

    # THE SABOTAGE, on one query's candidate set.
    var got = _search(ctx, index, q, N_QUERIES, K, 2)
    var layout = ListLayout(
        N_LISTS,
        N_ROWS,
        DIM,
        index.list_offsets.copy(),
        index.list_indices.copy(),
        index.list_data.copy(),
    )
    var qn = oracle_row_norms(q, N_QUERIES, DIM, False)
    var cn = oracle_row_norms(index.centers, N_LISTS, DIM, False)
    var ln = oracle_row_norms(index.list_data, N_ROWS, DIM, False)

    var crow = List[Float32]()
    var cids = List[UInt32]()
    for l in range(N_LISTS):
        crow.append(
            oracle_expanded_distance(
                q, 0, index.centers, l, DIM, qn[0], cn[l], False
            )
        )
        cids.append(UInt32(l))
    var probes_sel = oracle_select_k(crow, cids, 2)
    var probe_ids = List[UInt32]()
    for p in range(2):
        probe_ids.append(cids[Int(probes_sel[p])])

    var slots = merge_probed_lists(layout, probe_ids, 2)
    var good = gather_candidate_indices(layout, slots)
    var bad = sabotage_gather_candidate_indices(
        layout, slots, IVF_SAB_CARRY_POSITION
    )

    var crow2 = List[Float32]()
    for c in range(len(slots)):
        crow2.append(
            oracle_expanded_distance(
                q, 0, index.list_data, Int(slots[c]), DIM, qn[0],
                ln[Int(slots[c])], False,
            )
        )
    var sel = oracle_select_k(crow2, good, K)

    var differ = 0
    for i in range(K):
        if good[Int(sel[i])] != bad[Int(sel[i])]:
            differ += 1
    if differ == 0:
        raise Error(
            "check_list_layout_and_index_carry: the IVF_SAB_CARRY_POSITION"
            " sabotage did NOT move the answer on any of the "
            + String(K)
            + " selected slots. A sabotage that changes nothing proves the"
            " path is not reached ([[reached-but-inert]]); pick a fixture"
            " where a candidate's within-list position and its original"
            " index differ."
        )

    comptime if IDENTICAL:
        for i in range(K):
            if got.indices[i] != good[Int(sel[i])]:
                raise Error(
                    "check_list_layout_and_index_carry: slot "
                    + String(i)
                    + " of the device search names row "
                    + String(got.indices[i])
                    + " where the good carry names "
                    + String(good[Int(sel[i])])
                )

    print(
        "check_list_layout_and_index_carry OK ["
        + _mode_name()
        + "]: offsets partition "
        + String(N_ROWS)
        + " rows, the carry is a permutation and ascends within every list,"
        " list_data matches x[carry] bit for bit, and"
        " IVF_SAB_CARRY_POSITION moves "
        + String(differ)
        + " of "
        + String(K)
        + " returned identities"
    )
    _ = index^

    # DEVIATION 1946: the context dies LAST, after every value built on it.
    # Mojo frees at LAST USE, so without this the buffer releases above run
    # against a context that is already gone. On sm_89 the next GPU call in
    # the process then never returns (GPU idle, host threads in futex wait);
    # Apple and AMD do not show it, which is how it stayed latent here.
    _ = ctx^


# =====================================================================
# THE DEGENERATE LIST
# =====================================================================


def check_empty_list() raises:
    """An empty list corrupts nothing, and the sabotage shows it could.

    The layout is PLANTED (`ivf_planted_labels`) so the empty list is a
    fact of the fixture and not something k-means might or might not have
    produced. Every list is probed, so the empty one IS probed, and the
    answer must still be the whole index's brute force.

    THE SABOTAGE. `IVF_SAB_EMPTY_COUNTS_ONE` gives the empty list a size of
    one in the chunk scan, so `n_samples` overcounts and the selector is
    handed a row longer than the one the distance kernel filled. The check
    only compares the COUNTS, because the corrupted read is uninitialized
    memory and a check that asserted on its VALUE would be asserting on
    garbage.
    """
    var ctx = DeviceContext()
    var x = ivf_index_fixture(N_ROWS, DIM, 13)
    var q = ivf_query_fixture(x, N_ROWS, N_QUERIES, DIM, 13)
    var labels = ivf_planted_labels(N_ROWS, N_LISTS, 2)
    var centers = ivf_planted_centers(N_LISTS, DIM, 13)
    var index = _plant_index(
        ctx, x, labels, centers, N_ROWS, DIM, N_LISTS
    )

    if index.list_size(2) != 0:
        raise Error(
            "check_empty_list: list 2 holds "
            + String(index.list_size(2))
            + " vectors; the fixture was supposed to leave it empty"
        )
    for l in range(N_LISTS):
        if l != 2 and index.list_size(l) == 0:
            raise Error(
                "check_empty_list: list "
                + String(l)
                + " is ALSO empty, so 'one empty list' is not what this"
                " fixture plants"
            )

    var got = _search(ctx, index, q, N_QUERIES, K, N_LISTS)
    var want = _knn_reference(ctx, x, N_ROWS, q, N_QUERIES, DIM, K)
    var moved = 0
    for i in range(N_QUERIES * K):
        if not _same_bits(got.distances[i], want.distances[i]):
            moved += 1
        if got.indices[i] != want.indices[i]:
            moved += 1
    for qi in range(N_QUERIES):
        if Int(got.n_candidates[qi]) != N_ROWS:
            raise Error(
                "check_empty_list: query "
                + String(qi)
                + " counted "
                + String(got.n_candidates[qi])
                + " candidates with an empty list among the probes;"
                " expected "
                + String(N_ROWS)
            )

    # The sabotage, on the chunk scan alone.
    var list_sizes = List[Int32]()
    for l in range(N_LISTS):
        list_sizes.append(Int32(index.list_size(l)))
    var all_lists = List[UInt32]()
    for l in range(N_LISTS):
        all_lists.append(UInt32(l))
    var good_chunks = calc_chunk_indices(list_sizes, all_lists, N_LISTS)
    var bad_chunks = sabotage_calc_chunk_indices(
        list_sizes, all_lists, N_LISTS, IVF_SAB_EMPTY_COUNTS_ONE
    )
    if Int(good_chunks[N_LISTS - 1]) != N_ROWS:
        raise Error(
            "check_empty_list: the chunk scan totalled "
            + String(good_chunks[N_LISTS - 1])
            + ", expected "
            + String(N_ROWS)
        )
    if Int(bad_chunks[N_LISTS - 1]) != N_ROWS + 1:
        raise Error(
            "check_empty_list: IVF_SAB_EMPTY_COUNTS_ONE did NOT overcount;"
            " it totalled "
            + String(bad_chunks[N_LISTS - 1])
            + " where the good scan gives "
            + String(N_ROWS)
            + ". A sabotage that changes nothing proves nothing."
        )

    comptime if IDENTICAL:
        if moved != 0:
            raise Error(
                "check_empty_list FAILED: "
                + String(moved)
                + " of "
                + String(2 * N_QUERIES * K)
                + " values moved against the brute force with one empty"
                " list probed"
            )
        print(
            "check_empty_list OK [IDENTICAL]: list 2 empty, all "
            + String(N_LISTS)
            + " probed, answer bit-identical to brute force;"
            " IVF_SAB_EMPTY_COUNTS_ONE overcounts to "
            + String(bad_chunks[N_LISTS - 1])
        )
    else:
        print(
            "check_empty_list OK [FAST]: list 2 empty, all "
            + String(N_LISTS)
            + " probed, candidate counts exact,"
            " IVF_SAB_EMPTY_COUNTS_ONE overcounts to "
            + String(bad_chunks[N_LISTS - 1])
            + "; the "
            + String(moved)
            + " value differences against brute force are a REPORT under"
            " FAST (see check_nprobe_equals_nlists_is_brute_force)"
        )
    _ = index^

    # DEVIATION 1946: the context dies LAST, after every value built on it.
    # Mojo frees at LAST USE, so without this the buffer releases above run
    # against a context that is already gone. On sm_89 the next GPU call in
    # the process then never returns (GPU idle, host threads in futex wait);
    # Apple and AMD do not show it, which is how it stayed latent here.
    _ = ctx^


# =====================================================================
# THE TIE RULES
# =====================================================================


def check_assignment_ties() raises:
    """An exactly equidistant point goes to the LOWER LIST ID, on both
    sides of the algorithm, and reversing the centroid order proves the
    rule reads the ID.

    PART 1, THE ASSIGNMENT (hazard 1, DEVIATION 1789). The fixture's point
    is exactly equidistant from the two centres -- every coordinate a power
    of two, so the two distances are EQUAL IN FLOAT and the check refuses
    itself if the Float64 reference says otherwise. `cluster/`'s
    `predict` must label it 0. With the centres REVERSED the geometry is
    unchanged and the ids have swapped, so a rule that reads the id still
    says 0 (now naming the other centre) while a rule that reads arrival
    order, a pointer or the fold shape has no reason to.

    THE TIE RULE IS NOT IMPLEMENTED HERE. It is `raft::argmin_op`'s
    `(value, key)` order, carried by `cluster/impl/distance/
    fused_distance_nn/simt_kernel.mojo:537`, and this check GATES it. If it
    ever fails, the defect is in `cluster/` and this lane's index is what
    noticed.

    PART 2, THE PROBE ORDER (DEVIATION 1788). The same tie on the query
    side: a query equidistant from two centroids must probe the lower list
    first, which comes from the coarse selector's composite key rather than
    from an argmin. SABOTAGED with `sabotage_sort_probe_slots`, whose only
    difference is `>=` where the real sort has `<=`.
    """
    var ctx = DeviceContext()
    var d = 4
    var n = 8
    var centers = ivf_equidistant_centers(d)
    var rev = ivf_equidistant_centers_reversed(d)
    var pts = ivf_equidistant_points(n, d, 0)

    # THE CHECK REFUSES ITSELF IF THE TIE IS NOT A TIE.
    var d0 = reference_distance_f64(pts, 0, centers, 0, d, False)
    var d1 = reference_distance_f64(pts, 0, centers, 1, d, False)
    if d0 != d1:
        raise Error(
            "check_assignment_ties: the fixture is not a tie ("
            + String(d0)
            + " vs "
            + String(d1)
            + "). Every gate below would be testing the non-tie path."
        )

    var ref_labels = reference_assignment_f64(pts, n, centers, 2, d)
    for r in range(n):
        if ref_labels[r] != UInt32(0):
            raise Error(
                "check_assignment_ties: the Float64 reference put an exactly"
                " equidistant point in list "
                + String(ref_labels[r])
                + ", not 0. The reference's own tie rule is wrong."
            )

    var got_fwd = _device_assignment(ctx, pts, n, centers, 2, d)
    var got_rev = _device_assignment(ctx, pts, n, rev, 2, d)
    for r in range(n):
        if got_fwd[r] != UInt32(0):
            raise Error(
                "check_assignment_ties FAILED: device assignment put row "
                + String(r)
                + " in list "
                + String(got_fwd[r])
                + ", not the LOWER id 0 (IDENTITY_PATHS row 22,"
                " simt_kernel.mojo:537)"
            )
        if got_rev[r] != UInt32(0):
            raise Error(
                "check_assignment_ties FAILED: with the centroids REVERSED,"
                " row "
                + String(r)
                + " went to list "
                + String(got_rev[r])
                + ". The rule must still name the lower id, which now names"
                " the other centre."
            )

    # PART 2: the probe order.
    var probe_d = List[Float32]()
    var probe_i = List[UInt32]()
    probe_d.append(Float32(1.25))
    probe_d.append(Float32(1.25))
    probe_i.append(UInt32(0))
    probe_i.append(UInt32(1))
    var good_d = probe_d.copy()
    var good_i = probe_i.copy()
    sort_slots_by_distance_then_index(good_d, good_i, 0, 2)
    var bad_d = probe_d.copy()
    var bad_i = probe_i.copy()
    sabotage_sort_probe_slots(bad_d, bad_i, 0, 2)
    if good_i[0] != UInt32(0):
        raise Error(
            "check_assignment_ties FAILED: the probe order broke a tie"
            " toward list " + String(good_i[0]) + ", not the lower id 0"
        )
    if bad_i[0] != UInt32(1):
        raise Error(
            "check_assignment_ties: IVF_SAB_PROBE_TIE_HIGH did NOT move the"
            " probe order; a sabotage that changes nothing proves nothing"
        )

    print(
        "check_assignment_ties OK ["
        + _mode_name()
        + "]: "
        + String(n)
        + " exactly equidistant points to list 0 in both centroid orders;"
        " the probe tie goes to the lower id and "
        + sabotage_name(IVF_SAB_PROBE_TIE_HIGH)
        + " flips it"
    )


def _device_assignment(
    ctx: DeviceContext,
    x: List[Float32],
    n_rows: Int,
    centers: List[Float32],
    n_lists: Int,
    dim: Int,
) raises -> List[UInt32]:
    """`cluster/impl/cluster/kmeans.mojo::predict` against given centres.

    The build's assignment step, called directly so the tie question can be
    asked without a k-means fit in the way. `x_norm` is computed here for
    the reason `cluster/estimator.mojo` records: `predict` does not compute
    it and passing it uninitialized MERGES CLUSTERS.
    """
    var kp = KMeansParams.default()
    kp.n_clusters = n_lists
    kp.init = INIT_KMEANS_PLUS_PLUS
    kp.metric = METRIC_L2_EXPANDED
    var dx = upload_f32(ctx, x)
    var dc = upload_f32(ctx, centers)
    var dl = ctx.enqueue_create_buffer[DType.uint32](n_rows)
    var dn = ctx.enqueue_create_buffer[DType.float32](n_rows)
    var dmin = ctx.enqueue_create_buffer[DType.float32](n_rows)
    ctx.synchronize()
    compute_row_norms(ctx, dx, dn, n_rows, dim, False)
    ctx.synchronize()
    predict(ctx, dx, dn, dc, dl, dmin, kp, n_rows, dim)
    ctx.synchronize()
    var host = ctx.enqueue_create_host_buffer[DType.uint32](n_rows)
    ctx.enqueue_copy(dst_ptr=host.unsafe_ptr(), src_buf=dl)
    ctx.synchronize()
    var out = List[UInt32]()
    for r in range(n_rows):
        out.append(host.unsafe_ptr().unsafe_load(r))
    _ = host^
    _ = dx^
    _ = dc^
    _ = dl^
    _ = dn^
    _ = dmin^
    return out^


# =====================================================================
# HAZARD 2: THE CENTROIDS BEFORE ANYTHING DOWNSTREAM
# =====================================================================


def check_quantizer_is_reproducible() raises:
    """The centroid set is the same bytes twice, BEFORE any search result
    is called reproducible.

    HAZARD 2, and the order of these two claims is the whole point. List
    membership is a NUMERIC decision -- which vectors are in a list is
    which vectors the top-k sums over -- so the search result inherits
    every property the centroid set has and none it does not. A lane that
    gated its search and not its quantizer would be claiming
    reproducibility for a computation whose first stage it never looked at.

    WHAT THIS DOES AND DOES NOT ESTABLISH. It is TWO RUNS ON ONE DEVICE, in
    ONE process, in ONE mode. It cannot see contraction (IDENTITY_PATHS row
    9), the denormal policy (row 10) or the device transcendentals (row
    12), because those need a second backend. The cross-vendor statement
    for this k-means is `archive/research/UNSUPERVISED_IDENTITY.md`'s -- Apple, NVIDIA and
    AMD produce one distinct answer under IDENTICAL -- and this lane
    inherits exactly that and no more.
    """
    var ctx = DeviceContext()
    var x = ivf_index_fixture(N_ROWS, DIM, 17)
    var a = _build_index(ctx, x, N_ROWS, DIM, N_LISTS, UInt64(23))
    var b = _build_index(ctx, x, N_ROWS, DIM, N_LISTS, UInt64(23))

    var moved = 0
    for i in range(N_LISTS * DIM):
        if not _same_bits(a.centers[i], b.centers[i]):
            moved += 1
    var norm_moved = 0
    for l in range(N_LISTS):
        if not _same_bits(a.center_norms[l], b.center_norms[l]):
            norm_moved += 1
    var label_moved = 0
    for r in range(N_ROWS):
        if a.labels[r] != b.labels[r]:
            label_moved += 1
    var layout_moved = 0
    for s in range(N_ROWS):
        if a.list_indices[s] != b.list_indices[s]:
            layout_moved += 1

    if moved != 0 or norm_moved != 0 or label_moved != 0 or layout_moved != 0:
        raise Error(
            "check_quantizer_is_reproducible FAILED: two builds of one"
            " dataset at one seed differ -- "
            + String(moved)
            + " centroid values, "
            + String(norm_moved)
            + " centroid norms, "
            + String(label_moved)
            + " labels, "
            + String(layout_moved)
            + " layout slots. Nothing downstream of this can be called"
            " reproducible."
        )
    print(
        "check_quantizer_is_reproducible OK ["
        + _mode_name()
        + "]: two builds, same "
        + String(N_LISTS * DIM)
        + " centroid bytes, same "
        + String(N_LISTS)
        + " norms, same "
        + String(N_ROWS)
        + " labels and layout slots. ONE DEVICE, ONE PROCESS: rows 9, 10"
        " and 12 cannot fail this and E1 is not shortened by it."
    )
    _ = a^
    _ = b^

    # DEVIATION 1946: the context dies LAST, after every value built on it.
    # Mojo frees at LAST USE, so without this the buffer releases above run
    # against a context that is already gone. On sm_89 the next GPU call in
    # the process then never returns (GPU idle, host threads in futex wait);
    # Apple and AMD do not show it, which is how it stayed latent here.
    _ = ctx^


# =====================================================================
# THE ORACLE
# =====================================================================


def check_search_vs_oracle() raises:
    """Per query, per neighbour, bit for bit under IDENTICAL.

    Three fixtures, because the tie class is the interesting one and the
    hashed fixture does not reach it:

      hashed      no exact ties; the plain path
      duplicate   every distance attained by THREE original indices, so the
                  key's index half decides every slot
      signedzero  rows differing ONLY in the sign of a zero -- numerically
                  identical points that no distance comparison can order
                  (IDENTITY_PATHS row 39)

    The known-by-construction claim rides along: every EVEN query row is an
    exact copy of an index row, so that row must be its own first
    neighbour, and the Float64 reference says which row that is
    independently of any of this lane's arithmetic.
    """
    var ctx = DeviceContext()
    var names: List[String] = ["hashed", "duplicate", "signedzero"]
    var total_slots = 0
    var total_moved = 0

    for fi in range(3):
        var x: List[Float32]
        if fi == 0:
            x = ivf_index_fixture(N_ROWS, DIM, 29)
        elif fi == 1:
            x = ivf_duplicate_fixture(N_ROWS, DIM, 29)
        else:
            x = ivf_signed_zero_fixture(N_ROWS, DIM, 29)
        var q = ivf_query_fixture(x, N_ROWS, N_QUERIES, DIM, 29)
        var index = _build_index(ctx, x, N_ROWS, DIM, N_LISTS, UInt64(31))
        var n_probes = 3
        var got = _search(ctx, index, q, N_QUERIES, K, n_probes)
        var want = oracle_ivf_search(
            index.centers,
            N_LISTS,
            index.list_offsets,
            index.list_indices,
            index.list_data,
            N_ROWS,
            q,
            N_QUERIES,
            DIM,
            K,
            n_probes,
            False,
        )
        var moved = 0
        var first = String("")
        for i in range(N_QUERIES * K):
            var bad = False
            if not _same_bits(got.distances[i], want.distances[i]):
                bad = True
            if got.indices[i] != want.indices[i]:
                bad = True
            if bad:
                moved += 1
                if first == "":
                    first = (
                        "slot " + String(i) + " device ("
                        + _hex32(got.distances[i]) + ", "
                        + String(got.indices[i]) + ") oracle ("
                        + _hex32(want.distances[i]) + ", "
                        + String(want.indices[i]) + ")"
                    )
        for qi in range(N_QUERIES):
            if got.n_candidates[qi] != want.n_candidates[qi]:
                raise Error(
                    "check_search_vs_oracle ["
                    + names[fi]
                    + "]: query "
                    + String(qi)
                    + " scored "
                    + String(got.n_candidates[qi])
                    + " candidates, the oracle "
                    + String(want.n_candidates[qi])
                    + ". The two probe sets differ, which is a coarse-"
                    "selection difference and not a distance one."
                )
        total_slots += N_QUERIES * K
        total_moved += moved

        # THE KNOWN-BY-CONSTRUCTION CLAIM, from Float64 and from the
        # fixture's own expression, on the fixture that has no duplicates.
        if fi == 0:
            var truth = reference_nearest_f64(
                x, N_ROWS, q, N_QUERIES, DIM, 1, False
            )
            var wrong = 0
            for qi in range(N_QUERIES):
                var src = ivf_query_source_row(qi, N_ROWS)
                if src < 0:
                    continue
                if Int(truth[qi]) != src:
                    raise Error(
                        "check_search_vs_oracle: the Float64 reference says"
                        " query "
                        + String(qi)
                        + "'s nearest row is "
                        + String(truth[qi])
                        + ", the fixture built it as a copy of row "
                        + String(src)
                        + ". The fixture and the reference disagree."
                    )
                if Int(got.indices[qi * K]) != src:
                    wrong += 1
            if wrong != 0:
                print(
                    "  NOTE ["
                    + names[fi]
                    + "]: "
                    + String(wrong)
                    + " of the copied queries did not return their own row"
                    " first at n_probes="
                    + String(n_probes)
                    + ". That is IVF being APPROXIMATE -- the query's own"
                    " list was not probed -- and it is a REPORT, not a"
                    " failure. check_recall_is_reported is where it belongs."
                )

        comptime if IDENTICAL:
            if moved != 0:
                raise Error(
                    "check_search_vs_oracle FAILED ["
                    + names[fi]
                    + "]: "
                    + String(moved)
                    + " of "
                    + String(N_QUERIES * K)
                    + " slots differ from the serial float32 oracle. First: "
                    + first
                )
        _ = index^

        # DEVIATION 1946: the context dies LAST, after every value built on it.
        # Mojo frees at LAST USE, so without this the buffer releases above run
        # against a context that is already gone. On sm_89 the next GPU call in
        # the process then never returns (GPU idle, host threads in futex wait);
        # Apple and AMD do not show it, which is how it stayed latent here.
        _ = ctx^

    comptime if IDENTICAL:
        print(
            "check_search_vs_oracle OK [IDENTICAL]: "
            + String(total_slots)
            + " slots over hashed / duplicate / signed-zero fixtures, every"
            " distance AND index bit for bit against the serial oracle"
        )
    else:
        print(
            "check_search_vs_oracle REPORT [FAST]: "
            + String(total_moved)
            + " of "
            + String(total_slots)
            + " slots differ from the serial oracle. A REPORT: under FAST"
            " the distances are MAX's matmul and the selector resolves a"
            " tie by atomic arrival, and no host can replay either."
        )


# =====================================================================
# RECALL: A REPORT, NEVER AN ASSERTION
# =====================================================================


def check_recall_is_reported() raises:
    """Recall against brute force at several `n_probes`, as a REPORT.

    IVF IS APPROXIMATE. A recall below one at `n_probes < n_lists` is the
    algorithm working exactly as designed, so a threshold here would be a
    number nobody measured pretending to be a property, and this repository
    has a rule about that (`[[no-dataset-cherry-picking]]`'s neighbour:
    never build a gate to a number). The ONE assertion in this check is at
    `n_probes == n_lists`, where recall must be exactly 1.0 -- and that is
    not a recall claim at all, it is
    `check_nprobe_equals_nlists_is_brute_force` restated as a set.
    """
    var ctx = DeviceContext()
    var x = ivf_index_fixture(N_ROWS, DIM, 37)
    var q = ivf_query_fixture(x, N_ROWS, N_QUERIES, DIM, 37)
    var index = _build_index(ctx, x, N_ROWS, DIM, N_LISTS, UInt64(41))
    var truth = oracle_brute_force(
        x, N_ROWS, q, N_QUERIES, DIM, K, False
    )

    var probes: List[Int] = [1, 2, 3, 4, N_LISTS]
    for pi in range(len(probes)):
        var n_probes = probes[pi]
        # DEVIATION 1794 IS A LEGITIMATE OUTCOME AT A LOW n_probes and is
        # reported rather than swallowed: probing one list of an unbalanced
        # quantizer can leave a query with fewer than k candidates, which
        # is the refusal firing correctly and not a recall of zero. A check
        # that turned it into a zero would publish a number for a
        # computation that never ran.
        try:
            var got = _search(ctx, index, q, N_QUERIES, K, n_probes)
            var r = recall_against(
                got.indices, truth.indices, N_QUERIES, K
            )
            var scored = 0
            for qi in range(N_QUERIES):
                scored += Int(got.n_candidates[qi])
            print(
                "  REPORT recall@k="
                + String(K)
                + " n_probes="
                + String(n_probes)
                + " of "
                + String(N_LISTS)
                + " = "
                + String(r)
                + "  ("
                + String(scored)
                + " candidate cells scored of "
                + String(N_QUERIES * N_ROWS)
                + " a brute force would score).  THIS LINE IS A REPORT, not"
                " an assertion, and the cell count is not a speedup."
            )
            if n_probes == N_LISTS and r != Float64(1.0):
                raise Error(
                    "check_recall_is_reported: recall at n_probes =="
                    " n_lists is "
                    + String(r)
                    + ", not 1.0. That is not a recall failure, it is"
                    " check_nprobe_equals_nlists_is_brute_force failing as"
                    " a set."
                )
        except e:
            if n_probes == N_LISTS:
                raise Error(
                    "check_recall_is_reported: the n_probes == n_lists run"
                    " refused, which cannot be DEVIATION 1794 (every"
                    " candidate is probed). " + String(e)
                )
            print(
                "  REPORT recall@k="
                + String(K)
                + " n_probes="
                + String(n_probes)
                + " NOT COMPUTED, the search refused -- "
                + String(e)
            )

    print(
        "check_recall_is_reported OK ["
        + _mode_name()
        + "]: "
        + String(len(probes))
        + " n_probes values REPORTED; the only assertion is recall == 1.0"
        " at n_probes == n_lists"
    )
    _ = index^

    # DEVIATION 1946: the context dies LAST, after every value built on it.
    # Mojo frees at LAST USE, so without this the buffer releases above run
    # against a context that is already gone. On sm_89 the next GPU call in
    # the process then never returns (GPU idle, host threads in futex wait);
    # Apple and AMD do not show it, which is how it stayed latent here.
    _ = ctx^


# =====================================================================
# LAUNCH INVARIANCE
# =====================================================================


def check_launch_invariance() raises:
    """The answer does not move across two threads-per-block choices, a
    padded workspace, or a query answered alone versus inside a batch.

    THREADS PER BLOCK ARE SCHEDULING (`checks/numerics.mojo`'s
    `NumericMode`), so `tile_tpb` and `expand_tpb` may move and the answer
    may not. The NUMERIC choices -- the accumulator width, the feature-axis
    order, `SELECT_BLOCK`, the candidate merge's order, `n_probes` itself
    -- are pinned and are not varied here, because varying them is a
    different computation and `ivf/README.md` says which is which.

    THE PADDED ARM IS THE WORKSPACE, AND IT IS NOT AS STRONG AS A POISON.
    The candidate buffers are allocated once at `n_rows` and used at
    `n_cand`, so any `n_probes < n_lists` run already reads a buffer whose
    tail holds the previous query's values -- which is also why the
    alone-versus-batch arm doubles as the poison arm: the alone run gets
    freshly allocated buffers and the batch run gets dirty ones. What this
    does NOT do is choose the poison, so a value that happened to be
    benign would not be caught. A poison parameter belongs on the search
    entry and is named in `ivf/README.md`'s WHAT IS OWED rather than added
    here, because a test knob threaded into a production signature is a
    branch the shipped path can take.
    """
    var ctx = DeviceContext()
    var x = ivf_index_fixture(N_ROWS, DIM, 43)
    var q = ivf_query_fixture(x, N_ROWS, N_QUERIES, DIM, 43)
    var index = _build_index(ctx, x, N_ROWS, DIM, N_LISTS, UInt64(47))
    var n_probes = 3

    var base = _search(ctx, index, q, N_QUERIES, K, n_probes)
    var alt_tile = _search(
        ctx, index, q, N_QUERIES, K, n_probes, 64, IVF_EXPAND_TPB
    )
    var alt_expand = _search(
        ctx, index, q, N_QUERIES, K, n_probes, PINNED_TILE_TPB, 64
    )

    var moved_tile = 0
    var moved_expand = 0
    for i in range(N_QUERIES * K):
        if not _same_bits(base.distances[i], alt_tile.distances[i]):
            moved_tile += 1
        if base.indices[i] != alt_tile.indices[i]:
            moved_tile += 1
        if not _same_bits(base.distances[i], alt_expand.distances[i]):
            moved_expand += 1
        if base.indices[i] != alt_expand.indices[i]:
            moved_expand += 1

    # ALONE VERSUS INSIDE A BATCH. Query 5's answer computed on its own,
    # against the same query's slots inside the full batch.
    var one = List[Float32]()
    for f in range(DIM):
        one.append(q[5 * DIM + f])
    var alone = _search(ctx, index, one, 1, K, n_probes)
    var moved_batch = 0
    for i in range(K):
        if not _same_bits(alone.distances[i], base.distances[5 * K + i]):
            moved_batch += 1
        if alone.indices[i] != base.indices[5 * K + i]:
            moved_batch += 1
    if Int(alone.n_candidates[0]) != Int(base.n_candidates[5]):
        raise Error(
            "check_launch_invariance: query 5 scored "
            + String(alone.n_candidates[0])
            + " candidates alone and "
            + String(base.n_candidates[5])
            + " inside the batch. The candidate set is a function of the"
            " query and the index and of nothing else."
        )

    comptime if IDENTICAL:
        if moved_tile != 0 or moved_expand != 0 or moved_batch != 0:
            raise Error(
                "check_launch_invariance FAILED: tile_tpb moved "
                + String(moved_tile)
                + " values, expand_tpb moved "
                + String(moved_expand)
                + ", alone-vs-batch moved "
                + String(moved_batch)
                + ". Threads per block and batch composition are SCHEDULING"
                " and may not reach an output."
            )
        print(
            "check_launch_invariance OK [IDENTICAL]: "
            + String(N_QUERIES * K)
            + " slots unmoved across tile_tpb 256/64, expand_tpb 256/64,"
            " a workspace padded by "
            + String(N_ROWS - Int(base.n_candidates[5]))
            + " cells, and query 5 alone versus inside "
            + String(N_QUERIES)
        )
    else:
        print(
            "check_launch_invariance REPORT [FAST]: tile_tpb moved "
            + String(moved_tile)
            + ", expand_tpb moved "
            + String(moved_expand)
            + ", alone-vs-batch moved "
            + String(moved_batch)
            + " of "
            + String(2 * N_QUERIES * K)
            + ". A REPORT under FAST: the vendor matmul is entitled to a"
            " different k-split per shape, so the batch arm in particular"
            " is expected to move."
        )
    _ = index^

    # DEVIATION 1946: the context dies LAST, after every value built on it.
    # Mojo frees at LAST USE, so without this the buffer releases above run
    # against a context that is already gone. On sm_89 the next GPU call in
    # the process then never returns (GPU idle, host threads in futex wait);
    # Apple and AMD do not show it, which is how it stayed latent here.
    _ = ctx^


# =====================================================================
# THE CARD
# =====================================================================


def check_card_is_emitted() raises:
    """The sixteen `ivf.*` stages exist, in order, and two runs agree.

    A card is the instrument every cross-vendor claim about this lane will
    be made with, so the gate is that it EXISTS AND IS STABLE, not that its
    hashes are any particular value -- a hash asserted against a literal is
    a hash that has to be edited every time the fixture changes, which is
    how a card gate becomes a card rubber stamp.

    `IdentityTrace.to_path` and not `IdentityTrace()`: a check that reads
    `MOJOLEARN_IDENTITY_TRACE` behaves differently depending on the
    operator's shell.
    """
    var ctx = DeviceContext()
    var x = ivf_index_fixture(64, DIM, 53)
    var q = ivf_query_fixture(x, 64, 8, DIM, 53)
    var params = IvfFlatIndexParams.default()
    params.n_lists = 4
    params.kmeans_n_iters = 10
    params.kmeans_trainset_fraction = Float64(1.0)
    params.metric = METRIC_L2_EXPANDED
    params.seed = UInt64(59)
    var sp = IvfFlatSearchParams(2)

    var paths: List[String] = [
        String(SCRATCH) + "/ivf_check.card.a",
        String(SCRATCH) + "/ivf_check.card.b",
    ]
    for pi in range(2):
        var trace = IdentityTrace.to_path(paths[pi])
        var index = ivf_flat_build(ctx, trace, params, x, 64, DIM)
        _ = ivf_flat_search_traced(ctx, trace, index, sp, q, 8, 4)
        _ = index^

        # DEVIATION 1946: the context dies LAST, after every value built on it.
        # Mojo frees at LAST USE, so without this the buffer releases above run
        # against a context that is already gone. On sm_89 the next GPU call in
        # the process then never returns (GPU idle, host threads in futex wait);
        # Apple and AMD do not show it, which is how it stayed latent here.
        _ = ctx^

    var lines = read_trace_lines(paths[0])
    var expected: List[String] = [
        "ivf.centers", "ivf.center_norms", "ivf.assign", "ivf.list_offsets",
        "ivf.list_indices", "ivf.list_data", "ivf.query_norm",
        "ivf.coarse_dist", "ivf.probe_dist", "ivf.probe_lists",
        "ivf.cand_counts", "ivf.cand_idx", "ivf.cand_dist", "ivf.out_dist",
        "ivf.out_idx",
    ]
    var found = 0
    var quantizer = 0
    for i in range(len(lines)):
        var line = lines[i]
        if line.find("\tivf.quantizer.") >= 0:
            quantizer += 1
        for e in range(len(expected)):
            if line.find(String("\t") + expected[e] + "\t") >= 0:
                found += 1
    if found != len(expected):
        raise Error(
            "check_card_is_emitted: the card carries "
            + String(found)
            + " of the "
            + String(len(expected))
            + " expected ivf.* stages"
        )
    if quantizer == 0:
        raise Error(
            "check_card_is_emitted: the card carries NO ivf.quantizer.*"
            " stage, so the coarse fit wrote nothing into it and DEVIATION"
            " 1795's one-card property is not in force"
        )

    var div = first_divergence(paths[0], paths[1])
    if div != "":
        raise Error(
            "check_card_is_emitted: two runs of one build+search produced"
            " different cards. First divergence: " + div
        )

    print(
        "check_card_is_emitted OK ["
        + _mode_name()
        + "]: "
        + String(len(lines))
        + " records, "
        + String(len(expected))
        + " ivf.* stages plus "
        + String(quantizer)
        + " ivf.quantizer.* stages, two runs record-identical"
    )


# =====================================================================
# THE REMAINING SABOTAGE ARMS
# =====================================================================


def check_ivf_sabotages() raises:
    """The two arms the other checks do not already drive.

    `IVF_SAB_CARRY_POSITION` is driven by
    `check_list_layout_and_index_carry`, `IVF_SAB_EMPTY_COUNTS_ONE` by
    `check_empty_list`, and `IVF_SAB_PROBE_TIE_HIGH` by
    `check_assignment_ties`. The two here are the layout's within-list
    order and the candidate merge's order, which are the two halves of
    DEVIATIONS 1783 and 1786 and which nothing else reaches.

    Both are gated on a STRUCTURAL property -- does the candidate order
    still ascend in the carried original index -- rather than on the final
    answer, and that is deliberate. The final answer moves only where an
    exact tie exists, so an answer-level gate would be silently satisfied
    on any fixture without one; the ordering property is violated the
    moment either arm is on, on every fixture.
    """
    # NO DEVICE WORK HERE. Both sabotages are host-side orderings and the
    # property they break is structural, so this check needs no context and
    # takes none: a check that opens a `DeviceContext` it does not use is a
    # check that fails on a box for a reason unrelated to what it tests.
    var x = ivf_duplicate_fixture(N_ROWS, DIM, 61)
    # ROUND-ROBIN LABELS, so lists 0 and 1 INTERLEAVE in the original index
    # (row 0 to list 0, row 1 to list 1, row 2 to list 0, ...). Two lists
    # holding disjoint ascending ranges would make the probe-order sabotage
    # inert, which is what the raise at the bottom of this check refuses.
    var labels = ivf_planted_labels(N_ROWS, N_LISTS, N_LISTS - 1)

    var good_layout = build_list_layout(labels, x, N_ROWS, DIM, N_LISTS)
    var bad_layout = sabotage_build_list_layout(
        labels, x, N_ROWS, DIM, N_LISTS, IVF_SAB_LIST_ARRIVAL_ORDER
    )

    var probes = List[UInt32]()
    probes.append(UInt32(0))
    probes.append(UInt32(1))

    var good_slots = merge_probed_lists(good_layout, probes, 2)
    var good_ids = gather_candidate_indices(good_layout, good_slots)
    for c in range(1, len(good_ids)):
        if good_ids[c] <= good_ids[c - 1]:
            raise Error(
                "check_ivf_sabotages: the PRODUCTION merge does not ascend"
                " at candidate " + String(c) + " (DEVIATION 1786)"
            )

    var arrival_slots = merge_probed_lists(bad_layout, probes, 2)
    var arrival_ids = gather_candidate_indices(bad_layout, arrival_slots)
    var arrival_break = 0
    for c in range(1, len(arrival_ids)):
        if arrival_ids[c] <= arrival_ids[c - 1]:
            arrival_break += 1
    if arrival_break == 0:
        raise Error(
            "check_ivf_sabotages: "
            + sabotage_name(IVF_SAB_LIST_ARRIVAL_ORDER)
            + " left the candidate order ascending. The merge assumes each"
            " list is sorted; a sabotage that changes nothing proves the"
            " assumption is not load-bearing on this fixture."
        )

    var probe_slots = sabotage_merge_probed_lists(
        good_layout, probes, 2, IVF_SAB_MERGE_PROBE_ORDER
    )
    var probe_ids = gather_candidate_indices(good_layout, probe_slots)
    var probe_break = 0
    for c in range(1, len(probe_ids)):
        if probe_ids[c] <= probe_ids[c - 1]:
            probe_break += 1
    if probe_break == 0:
        raise Error(
            "check_ivf_sabotages: "
            + sabotage_name(IVF_SAB_MERGE_PROBE_ORDER)
            + " left the candidate order ascending, so this fixture's two"
            " probed lists happen to hold disjoint ascending ranges. Pick"
            " lists whose members interleave."
        )

    print(
        "check_ivf_sabotages OK ["
        + _mode_name()
        + "]: the production merge ascends over "
        + String(len(good_ids))
        + " candidates; "
        + sabotage_name(IVF_SAB_LIST_ARRIVAL_ORDER)
        + " breaks it at "
        + String(arrival_break)
        + " positions and "
        + sabotage_name(IVF_SAB_MERGE_PROBE_ORDER)
        + " at "
        + String(probe_break)
    )


def main() raises:
    print("ivf_check mode=" + _mode_name())
    check_nprobe_equals_nlists_is_brute_force()
    check_ivf_refusals()
    check_list_layout_and_index_carry()
    check_empty_list()
    check_assignment_ties()
    check_quantizer_is_reproducible()
    check_search_vs_oracle()
    check_recall_is_reported()
    check_launch_invariance()
    check_card_is_emitted()
    check_ivf_sabotages()
    print("ivf_check mode=" + _mode_name() + " ALL OK")
