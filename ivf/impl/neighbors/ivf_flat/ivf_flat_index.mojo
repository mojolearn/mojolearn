# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""IVF-FLAT's parameters, its index, and every refusal by name.

PORT OF `cuvs/include/cuvs/neighbors/ivf_flat.hpp` (`index_params` :28-66,
`search_params` :76-82, `index` :137-274) and
`cuvs/src/neighbors/ivf_flat_index.cpp` (the constructor's
`check_consistency` at :206-215) at cuVS `6ba2ce2`. Partial. Do not improve.

WHAT AN IVF-FLAT INDEX IS, IN THEIR WORDS AND IN OURS
------------------------------------------------------
Theirs is `n_lists` centroids plus `n_lists` separately allocated lists,
each holding its members' vectors in INTERLEAVED GROUPS of
`kIndexGroupSize = 32` (`ivf_flat.hpp:25`, `list_spec::make_list_extents`
at `:110-115`) so that `ivfflat_interleaved_scan` can issue vectorized
loads of `veclen` elements per lane. Ours is the same centroids plus ONE
CSR-shaped triple -- offsets, carried original indices, and the permuted
vectors -- because the layout exists to feed a scan, and the scan we run
is not theirs. **DEVIATION 1782**, and `ivf/README.md` carries the whole
reason.

THE FIELD THAT IS THE WHOLE LANE
--------------------------------
`list_indices`. Their `build_index_kernel` writes
`list_index[inlist_id] = source_ix` (`ivf_flat_build.cuh:135`), so the
ORIGINAL row id travels with the vector into the layout and their
`postprocess_neighbors` reads it back out (`ivf_common.cuh:133`). Ours does
the same thing and then does the thing theirs cannot: it puts that original
id into the SELECTION KEY, which is what makes `n_probe == n_lists` reduce
to brute force bit for bit. **DEVIATION 1784**, and IVF's classic identity
bug is the version of this file where `list_indices` holds a position
within a list instead. `check_index_carry` sabotages exactly that.

WHAT IS REFUSED HERE, AND WHY IT IS REFUSED HERE
-------------------------------------------------
Every unported neighbour of IVF-FLAT in the cuVS tree raises BY NAME from
`ivf_refuse_algorithm` rather than by absence, because absence is a silent
answer and this repository has paid for those. HNSW is refused
PERMANENTLY (`ivf/README.md`, and `cpp/include/cuvs/neighbors/hnsw.hpp` is
the header a future reader will find and wonder about); IVF-PQ, IVF-SQ,
IVF-RaBitQ, CAGRA, ScaNN, Vamana and NN-Descent are refused as UNPORTED,
which is a different sentence and `ivf/NOT_IMPLEMENTED.tsv` says which is which.
"""

from std.memory import bitcast

from cluster.impl.cluster.kmeans_params import (
    METRIC_COSINE_EXPANDED,
    METRIC_L2_EXPANDED,
    METRIC_L2_SQRT_EXPANDED,
)


comptime IVF_GROUP_SIZE = 32
"""`ivf_flat.hpp:25`'s `kIndexGroupSize`. Recorded, NOT USED: it is the
interleaved layout's group width and DEVIATION 1782 does not lay data out
that way. It is here so a reader diffing against their header finds the
constant and this sentence rather than an unexplained absence."""

comptime IVF_DEFAULT_N_LISTS = 1024
comptime IVF_DEFAULT_KMEANS_N_ITERS = 20
comptime IVF_DEFAULT_KMEANS_TRAINSET_FRACTION = Float64(0.5)
comptime IVF_DEFAULT_N_PROBES = 20
"""`ivf_flat.hpp:30,32,34,78`, unchanged."""


@fieldwise_init
struct IvfFlatIndexParams(Copyable, ImplicitlyCopyable, Movable):
    """`ivf_flat::index_params`, `ivf_flat.hpp:28-66`.

    Every field theirs has is here, including the three this port REFUSES,
    because a parameter that silently does not exist is a parameter a
    caller assumes is honoured. `ivf_index_params_validate` names each one.
    """

    var n_lists: Int
    var kmeans_n_iters: Int
    var kmeans_trainset_fraction: Float64
    var adaptive_centers: Bool
    var conservative_memory_allocation: Bool
    var add_data_on_build: Bool
    var metric: Int
    var seed: UInt64
    """NOT THEIRS AT THIS LEVEL. cuVS's `index_params` carries no seed; the
    seed reaches their quantizer through `kmeans::balanced_params`. Ours
    reaches `KMeansParams.seed`, and it is surfaced because k-means++ draws
    from it and a reproducible index needs the number written down."""

    @staticmethod
    def default() -> Self:
        return Self(
            IVF_DEFAULT_N_LISTS,
            IVF_DEFAULT_KMEANS_N_ITERS,
            IVF_DEFAULT_KMEANS_TRAINSET_FRACTION,
            False,
            False,
            True,
            METRIC_L2_EXPANDED,
            UInt64(0),
        )


@fieldwise_init
struct IvfFlatSearchParams(Copyable, ImplicitlyCopyable, Movable):
    """`ivf_flat::search_params`, `ivf_flat.hpp:76-82`.

    `n_probes` IS A NUMERIC PARAMETER (DEVIATION 1787). It decides which
    vectors are summed over, so it is part of the answer and not a tuning
    knob: it goes in the identity card's header, `ivf/README.md`'s identity
    table has a row for it, and two runs at different `n_probes` are two
    different computations that must not be compared as if they were one.

    `metric_udf` (`:80`) is not ported: it is a JIT string compiled into
    their scan.
    """

    var n_probes: Int

    @staticmethod
    def default() -> Self:
        return Self(IVF_DEFAULT_N_PROBES)


struct IvfFlatIndex(Movable):
    """`ivf_flat::index`, `ivf_flat.hpp:137-274`, CSR instead of their lists.

    HOST-RESIDENT BETWEEN BUILD AND SEARCH, **DEVIATION 1804**. Theirs lives
    on the device from `build` to the last `search`; ours is host `List`s
    that the search uploads. That is a real departure from
    `PORTING_RULES.md` rule 2 and it is named rather than hidden: it costs
    an upload per search call and it changes no bit, because the upload is
    a copy. Closure condition: hold the `DeviceBuffer`s in this struct and
    give the estimator an explicit `DeviceContext` lifetime, which is a
    surface change and not a numeric one. Not done, because this lane has
    run nothing and a memory-residency choice made without a measurement is
    the kind of invention `PORTING_RULES.md` 0c is about.
    """

    var n_lists: Int
    var dim: Int
    var n_rows: Int
    var metric: Int

    var centers: List[Float32]
    """`index::centers()`, `[n_lists, dim]`, row-major. Produced by the
    ported k-means; see `ivf_flat_build.mojo` for which entry point."""

    var center_norms: List[Float32]
    """`index::center_norms()`, `[n_lists]`. SQUARED on `L2Expanded` and
    `L2SqrtExpanded` alike, because the expanded form wants the square and
    the root is taken at the end -- their `:117-118` comment on the k-NN
    side, and `core/row_norms.mojo` is the kernel in both places."""

    var list_offsets: List[Int32]
    """`[n_lists + 1]`, the CSR row pointer. `list_offsets[l+1] -
    list_offsets[l]` is `index::list_sizes()(l)`. An EMPTY list is
    `list_offsets[l+1] == list_offsets[l]` and is a normal state, not an
    error: `check_empty_list` gates that it corrupts nothing."""

    var list_indices: List[UInt32]
    """`[n_rows]`, the ORIGINAL row id of each stored vector, ASCENDING
    within every list (DEVIATIONS 1783/1784). Their
    `list_index[inlist_id] = source_ix` with the arrival order removed."""

    var list_data: List[Float32]
    """`[n_rows, dim]`, the vectors in list order. A PERMUTATION of the
    input, and a permutation changes nothing only because `list_indices`
    travels beside it."""

    var labels: List[UInt32]
    """`[n_rows]`, the assignment. Not a field of theirs -- their labels are
    workspace inside `extend` -- and kept because it is a card stage
    (`ivf.assign`) and because `check_assignment_ties` reads it."""

    def __init__(
        out self,
        n_lists: Int,
        dim: Int,
        n_rows: Int,
        metric: Int,
        var centers: List[Float32],
        var center_norms: List[Float32],
        var list_offsets: List[Int32],
        var list_indices: List[UInt32],
        var list_data: List[Float32],
        var labels: List[UInt32],
    ):
        self.n_lists = n_lists
        self.dim = dim
        self.n_rows = n_rows
        self.metric = metric
        self.centers = centers^
        self.center_norms = center_norms^
        self.list_offsets = list_offsets^
        self.list_indices = list_indices^
        self.list_data = list_data^
        self.labels = labels^

    def list_size(self, list_id: Int) -> Int:
        """`index::list_sizes()(list_id)`."""
        return Int(self.list_offsets[list_id + 1]) - Int(
            self.list_offsets[list_id]
        )

    def size(self) -> Int:
        """`index::size()`, `ivf_flat_index.cpp:133`."""
        return self.n_rows


def ivf_metric_name(metric: Int) -> String:
    if metric == METRIC_L2_EXPANDED:
        return String("L2Expanded")
    if metric == METRIC_L2_SQRT_EXPANDED:
        return String("L2SqrtExpanded")
    if metric == METRIC_COSINE_EXPANDED:
        return String("CosineExpanded")
    return String("<unknown metric ") + String(metric) + ">"


def ivf_metric_from_name(name: String) raises -> Int:
    """The two metrics this port carries, by their cuVS names.

    `search_impl`'s switch (`ivf_flat_search.cuh:109-146`) has three arms:
    L2Expanded / L2SqrtExpanded share one (`alpha=-2, beta=1` over the
    query norms), CosineExpanded is its own, and `default` is the
    inner-product fall-through. Only the first arm is ported, so cosine and
    inner product raise here rather than run a different arm's arithmetic.
    """
    if name == "l2_expanded" or name == "euclidean" or name == "sqeuclidean":
        return METRIC_L2_EXPANDED
    if name == "l2_sqrt_expanded" or name == "l2":
        return METRIC_L2_SQRT_EXPANDED
    if name == "cosine" or name == "cosine_expanded":
        raise Error(
            "ivf_flat: metric 'cosine' is CosineExpanded, which is"
            " ivf_flat_search.cuh:130-141's own arm (a per-cell divide by"
            " the norm product) and is NOT PORTED. See ivf/NOT_IMPLEMENTED.tsv."
        )
    if name == "inner_product" or name == "ip":
        raise Error(
            "ivf_flat: metric 'inner_product' takes ivf_flat_search.cuh"
            ":142-145's default arm (alpha=1, beta=0, select_max) and is"
            " NOT PORTED. See ivf/NOT_IMPLEMENTED.tsv."
        )
    raise Error(
        "ivf_flat: unknown metric '"
        + name
        + "'. This port carries l2_expanded and l2_sqrt_expanded."
    )


def ivf_refuse_algorithm(name: String) raises:
    """Every neighbour of IVF-FLAT in the cuVS tree, refused BY NAME.

    Scope is IVF-FLAT ONLY. This function exists so that a caller who asks
    for one of the others gets a sentence naming the thing and its status
    instead of an attribute error four layers down.
    """
    if name == "hnsw":
        raise Error(
            "hnsw: REFUSED PERMANENTLY in this repository. cuVS's"
            " cpp/include/cuvs/neighbors/hnsw.hpp dispatches to the hnswlib"
            " CPU graph; it is not a GPU algorithm and PORTING_RULES.md"
            " 0b-ii says there is no CPU path here. This is a refusal, not"
            " an unported item, and it does not become one later."
        )
    if name == "ivf_pq" or name == "ivfpq":
        raise Error(
            "ivf_pq: NOT PORTED. cuvs/src/neighbors/ivf_pq/ is a different"
            " algorithm (product quantization, a codebook per subspace and"
            " a LUT scan), out of this lane's scope by its brief. See"
            " ivf/NOT_IMPLEMENTED.tsv."
        )
    if name == "cagra":
        raise Error(
            "cagra: NOT PORTED. cuvs/src/neighbors/cagra.cuh is a graph"
            " index, out of this lane's scope by its brief. See"
            " ivf/NOT_IMPLEMENTED.tsv."
        )
    if name == "ivf_sq":
        raise Error(
            "ivf_sq: NOT PORTED (cuvs/src/neighbors/ivf_sq/, scalar"
            " quantization over an IVF index). See ivf/NOT_IMPLEMENTED.tsv."
        )
    if name == "ivf_rabitq":
        raise Error(
            "ivf_rabitq: NOT PORTED (cuvs/src/neighbors/ivf_rabitq/). See"
            " ivf/NOT_IMPLEMENTED.tsv."
        )
    if name == "scann":
        raise Error("scann: NOT PORTED. See ivf/NOT_IMPLEMENTED.tsv.")
    if name == "vamana":
        raise Error("vamana: NOT PORTED. See ivf/NOT_IMPLEMENTED.tsv.")
    if name == "nn_descent":
        raise Error("nn_descent: NOT PORTED. See ivf/NOT_IMPLEMENTED.tsv.")
    if name == "ivf_flat":
        return
    raise Error(
        "ivf_flat: unknown algorithm '"
        + name
        + "'. This lane implements ivf_flat and refuses the rest by name."
    )


comptime IVF_MAGNITUDE_BOUND = Float32(9.2233720368547758e18)
"""2^63. The same bound `kde/impl/neighbors/kernel_density.mojo` uses for
the same reason: an expanded L2 forms `||x||^2`, so a coordinate at or above
2^63 overflows the square to `+inf` and every downstream stage then carries
a vendor-payload NaN rather than a number (IDENTITY_PATHS row 39, FACT 2)."""


@always_inline
def ivf_is_finite(v: Float32) -> Bool:
    """NaN and both infinities, by bits. `v != v` catches NaN and an
    exponent of `0xFF` with a zero mantissa is an infinity; testing the
    exponent field catches both in one read and does not depend on how a
    backend orders a comparison against NaN."""
    var u = bitcast[DType.uint32](v)
    return ((u >> UInt32(23)) & UInt32(0xFF)) != UInt32(0xFF)


def ivf_validate_data(
    values: List[Float32], n_rows: Int, dim: Int, where: String
) raises:
    """DEVIATION 1793's finiteness half, refused BEFORE any upload.

    cuVS's `build` checks `n_rows > 0 && dim > 0` and `n_rows >=
    n_lists` (`ivf_flat_build.cuh:403-405`) and nothing about the VALUES.
    We check the values because every stage of this lane is a recorded card
    stage and a computed NaN carries the vendor's payload, so one non-finite
    input turns a cross-vendor comparison into a comparison of NaN payloads.
    Same rule, same wording, same reason as `kde_validate_data`.
    """
    if len(values) != n_rows * dim:
        raise Error(
            "ivf_flat: "
            + where
            + " has "
            + String(len(values))
            + " values, expected n_rows * dim = "
            + String(n_rows * dim)
        )
    for i in range(n_rows * dim):
        var v = values[i]
        if not ivf_is_finite(v):
            raise Error(
                "ivf_flat: "
                + where
                + " value at flat position "
                + String(i)
                + " (row "
                + String(i // dim)
                + ", column "
                + String(i % dim)
                + ") is not finite. Every stage here is a card stage and a"
                " computed NaN carries the vendor's payload"
                " (IDENTITY_PATHS row 39)."
            )
        if abs(v) >= IVF_MAGNITUDE_BOUND:
            raise Error(
                "ivf_flat: "
                + where
                + " value at flat position "
                + String(i)
                + " has magnitude at or above 2^63; the expanded L2 squares"
                " it and the square overflows to +inf."
            )


def ivf_index_params_validate(
    params: IvfFlatIndexParams, n_rows: Int, dim: Int
) raises:
    """Their `RAFT_EXPECTS` block (`ivf_flat_build.cuh:403-407`) plus the
    three parameters this port refuses.

    Theirs, unchanged:
        n_rows > 0 && dim > 0                      (:403)
        n_rows >= params.n_lists                   (:404)
        metric != CosineExpanded || dim > 1        (:405)

    Ours, added, each with the sentence saying why:
        n_lists >= 1
        kmeans_n_iters >= 1
        kmeans_trainset_fraction == 1.0            DEVIATION 1781
        adaptive_centers == False                  DEVIATION 1793
        conservative_memory_allocation == False    DEVIATION 1782
        add_data_on_build == True                  DEVIATION 1793
    """
    if n_rows <= 0 or dim <= 0:
        raise Error(
            "ivf_flat build: empty dataset (n_rows="
            + String(n_rows)
            + ", dim="
            + String(dim)
            + "), their RAFT_EXPECTS at ivf_flat_build.cuh:403"
        )
    if params.n_lists < 1:
        raise Error(
            "ivf_flat build: n_lists must be at least 1, got "
            + String(params.n_lists)
        )
    if n_rows < params.n_lists:
        raise Error(
            "ivf_flat build: number of rows ("
            + String(n_rows)
            + ") can't be less than n_lists ("
            + String(params.n_lists)
            + "), their RAFT_EXPECTS at ivf_flat_build.cuh:404"
        )
    if params.kmeans_n_iters < 1:
        raise Error(
            "ivf_flat build: kmeans_n_iters must be at least 1, got "
            + String(params.kmeans_n_iters)
        )
    if params.metric != METRIC_L2_EXPANDED and (
        params.metric != METRIC_L2_SQRT_EXPANDED
    ):
        raise Error(
            "ivf_flat build: metric "
            + ivf_metric_name(params.metric)
            + " is not ported; this port carries L2Expanded and"
            " L2SqrtExpanded (ivf_flat_search.cuh:110-129's arm)."
        )
    if params.kmeans_trainset_fraction != Float64(1.0):
        # DEVIATION 1781. Their trainset is a STRIDED SUBSAMPLE whose row
        # count is `n_rows / max(1, n_rows / max(fraction * n_rows,
        # n_lists))` (`ivf_flat_build.cuh:416-418`) -- a float product
        # truncated to an integer, which is IDENTITY_PATHS row 18's class
        # exactly: near an integer boundary the truncation flips WHICH rows
        # train the quantizer, and the quantizer decides list membership,
        # and list membership decides the summation set. Their own comment
        # on the line below it is "TODO: a proper sampling".
        raise Error(
            "ivf_flat build: kmeans_trainset_fraction = "
            + String(params.kmeans_trainset_fraction)
            + " is REFUSED (DEVIATION 1781). Their trainset row count"
            " (ivf_flat_build.cuh:416-418) is a truncated float product,"
            " so near an integer boundary two hosts train the quantizer on"
            " DIFFERENT ROWS and the whole index differs. Only 1.0 (the"
            " whole dataset, no truncation) is accepted. Closure: port"
            " their stride as exact integer arithmetic, the way"
            " IDENTITY_PATHS row 18 closed compute_max_features."
        )
    if params.adaptive_centers:
        raise Error(
            "ivf_flat build: adaptive_centers=True is NOT PORTED"
            " (DEVIATION 1793). Their own header says it makes the"
            " centroids depend on the ORDER new data is added in"
            " (ivf_flat.hpp:41-46); that is an order dependence in the"
            " index itself, and extend() is not ported either."
        )
    if params.conservative_memory_allocation:
        raise Error(
            "ivf_flat build: conservative_memory_allocation is NOT PORTED"
            " (DEVIATION 1782). It selects `align_max` for their"
            " per-list allocation (ivf_flat.hpp:96-100); this port lays the"
            " lists out CSR and has no per-list allocation to align."
        )
    if not params.add_data_on_build:
        raise Error(
            "ivf_flat build: add_data_on_build=False is NOT PORTED"
            " (DEVIATION 1793). It leaves the index trained and empty for a"
            " later extend(), and extend() is not ported. See"
            " ivf/NOT_IMPLEMENTED.tsv."
        )


def ivf_search_params_validate(
    sp: IvfFlatSearchParams, n_lists: Int, n_queries: Int, k: Int
) raises:
    """Their `RAFT_EXPECTS` at `ivf_flat_search.cuh:329-331` plus the two
    shape refusals this port owes its own selector.

    THEIRS CLAMPS AND OURS REFUSES, and that is DEVIATION 1793's second
    half. `search_with_filtering` does `n_probes = std::min(params.n_probes,
    index.n_lists())` (`:331`), so asking for more probes than there are
    lists silently becomes "probe everything". Because `n_probes` is a
    NUMERIC parameter here (DEVIATION 1787), a silent clamp would mean two
    callers who wrote different numbers get one answer and neither can tell
    from the card which computation ran. So it raises.
    """
    if sp.n_probes <= 0:
        raise Error(
            "ivf_flat search: n_probes (number of clusters to probe) must"
            " be positive, their RAFT_EXPECTS at ivf_flat_search.cuh:329"
        )
    if sp.n_probes > n_lists:
        raise Error(
            "ivf_flat search: n_probes ("
            + String(sp.n_probes)
            + ") exceeds n_lists ("
            + String(n_lists)
            + "). Theirs clamps at ivf_flat_search.cuh:331; this port"
            " REFUSES, because n_probes is a numeric parameter"
            " (DEVIATION 1787) and a clamped one is a card that does not"
            " say which computation ran."
        )
    if n_queries <= 0:
        raise Error(
            "ivf_flat search: n_queries must be positive, got "
            + String(n_queries)
        )
    if k <= 0:
        raise Error("ivf_flat search: k must be positive, got " + String(k))
