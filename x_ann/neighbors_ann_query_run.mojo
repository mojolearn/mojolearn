# SPDX-License-Identifier: Apache-2.0
"""ANN search door RUN: the checks on typed values, the search plan (fitted
array schema, parameter words and address order) and the filter preparation.

`bindings/neighbors_ann_query_native.mojo` is the UNPACK/PACK side: it reads
the estimator's fields once, admits the queries and the fitted arrays, owns
the resident handle and publishes the result, and calls these functions at
the point the public method made each check, so refusal order and wording are
unchanged. Nothing here imports Python.
"""
from std.memory import MutPointer
from x_ann.neighbors_ann_run import rabitq_code_words

comptime ANN_SEARCH_PQ = 0
comptime ANN_SEARCH_SQ = 1
comptime ANN_SEARCH_RABITQ = 2
comptime ANN_SEARCH_CAGRA = 3
comptime _ANN_Q_I32P = MutPointer[Int32, MutUntrackedOrigin]
comptime _ANN_Q_U8P = MutPointer[UInt8, MutUntrackedOrigin]


def ann_search_owner(kind: Int) raises -> String:
    """The public class name of search `kind` (refuses an unknown kind)."""
    if kind == ANN_SEARCH_PQ:
        return "IVFPQIndex"
    if kind == ANN_SEARCH_SQ:
        return "IVFSQIndex"
    if kind == ANN_SEARCH_RABITQ:
        return "IVFRaBitQIndex"
    if kind == ANN_SEARCH_CAGRA:
        return "CagraIndex"
    raise Error("RuntimeError: unknown native ANN search kind")


def ann_search_entry(kind: Int) -> String:
    """The per-call search export of `kind` (no resident handle)."""
    if kind == ANN_SEARCH_PQ:
        return "x_ann_ivf_pq_search"
    if kind == ANN_SEARCH_SQ:
        return "x_ann_ivf_sq_search"
    if kind == ANN_SEARCH_RABITQ:
        return "x_ann_ivf_rabitq_search"
    return "x_ann_cagra_search"


def ann_check_search_dims(n: Int, dim: Int, k: Int) raises:
    if n < 1 or dim < 1 or k < 1:
        raise Error("ValueError: ANN rows, features and n_neighbors must be positive")


def ann_check_query_features(owner: String, query_features: Int, dim: Int) raises:
    if query_features != dim:
        raise Error("ValueError: mojolearn " + owner + ": queries have " + String(query_features)
                    + " features, the index has " + String(dim))


def cagra_check_search(graph_degree: Int, length: Int, width: Int, iterations: Int, seeds: Int) raises:
    if graph_degree < 1 or length < 1 or width < 1 or iterations < 0 or seeds < 0:
        raise Error("ValueError: invalid CagraIndex search configuration")


def ivf_check_search(n: Int, n_lists: Int, probes: Int) raises:
    if n_lists < 1 or n_lists > n or probes < 1 or probes > n_lists:
        raise Error("ValueError: ANN n_lists must be in [1, n_rows] and n_probes in [1, n_lists]")


def pq_check_search(dim: Int, pq_dim: Int, bits: Int, length: Int) raises:
    if pq_dim < 1 or pq_dim > dim or bits < 1 or bits > 8 or length != (dim + pq_dim - 1) // pq_dim:
        raise Error("ValueError: invalid fitted IVFPQIndex dimensions")


def pq_check_code_dims(n: Int, pq_dim: Int) raises:
    if n < 1 or pq_dim < 1:
        raise Error("ValueError: invalid fitted IVFPQIndex code dimensions")


def cagra_seed_word(seed_bytes: List[Int]) raises -> Int:
    """The search seed from the admitted exact integer's signed big-endian
    bytes: refuses a negative value, then folds the bytes modulo 2^31 - 2
    (int31 modular arithmetic, any seed size) and adds one."""
    if len(seed_bytes) > 0 and seed_bytes[0] >= 128:
        raise Error("ValueError: mojolearn CagraIndex: random_state must be None or >= 0")
    var word = 0
    for value in seed_bytes:  # cpu-route: seed-word admission over exact integer bytes
        word = (word * 256 + value) % 2147483646
    return word + 1


# ---------------------------------------------------------------- stored arrays


def ann_check_stored_shape(name: String, shape: List[Int], expected: List[Int]) raises:
    """A fitted array's rank and shape against the fitted dimensions."""
    if len(shape) != len(expected):
        raise Error("ValueError: invalid fitted " + name + " rank")
    for axis in range(len(shape)):  # small-loop(saved array rank)
        if shape[axis] != expected[axis]:
            raise Error("ValueError: invalid fitted " + name + " shape")


def ann_check_stored_strides(name: String, shape: List[Int], strides: List[Int], non_int_axis: Int) raises:
    """C-contiguous 32-bit storage, walked from the last axis. `non_int_axis`
    is the last axis whose stride was not an exact int (-1 for none); it is
    refused when the walk reaches it, as the inline check was."""
    var step = 4
    for offset in range(len(shape)):  # small-loop(saved array rank)
        var axis = len(shape) - offset - 1
        if axis == non_int_axis:
            raise Error("TypeError: fitted array strides must be exact native ints")
        if shape[axis] > 1 and strides[axis] != step:
            raise Error("ValueError: fitted " + name + " requires C-contiguous storage")
        step *= shape[axis]


# ---------------------------------------------------------------- filter


def ann_mask_refusal(owner: String, n: Int) -> String:
    return ("ValueError: mojolearn " + owner + ": filter must be a boolean array of shape (" + String(n)
            + ",), one flag per indexed row")


def ann_mask_fill_ones(dest_address: Int, n: Int):
    """The all-rows filter: one int32 flag per indexed row."""
    var dest = _ANN_Q_I32P(unsafe_from_address=dest_address)
    for i in range(n):  # cpu-route: explicit filter preparation, no numerical algorithm
        dest[i] = 1


def ann_mask_from_bytes(dest_address: Int, source_address: Int, n: Int, stride: Int):
    """A strided one-byte boolean view as int32 flags."""
    var dest = _ANN_Q_I32P(unsafe_from_address=dest_address)
    var source = _ANN_Q_U8P(unsafe_from_address=source_address)
    for i in range(n):  # cpu-route: explicit strided boolean-to-int32 filter preparation
        dest[i] = 1 if source[i * stride] != 0 else 0


# ---------------------------------------------------------------- search plan


@fieldwise_init
struct AnnSearchRequest(Copyable, Movable):
    """The admitted fitted dimensions and search settings of one search."""
    var kind: Int
    var n: Int
    var dim: Int
    var k: Int
    var m: Int
    var status_word: Int
    var filtered: Bool
    var device_codes: Bool
    var n_lists: Int
    var probes: Int
    var pq_dim: Int
    var pq_bits: Int
    var pq_len: Int
    var seed: Int
    var graph_degree: Int
    var itopk: Int
    var width: Int
    var iterations: Int
    var seeds: Int


@fieldwise_init
struct AnnSearchPlan(Copyable, Movable):
    """The fitted-array schema to admit (name, expected shape, float or int
    storage), the parameter words of the prepare, resident search and
    per-call search exports, and the candidate count per query."""
    var names: List[String]
    var shapes: List[List[Int]]
    var floating: List[Bool]
    var prepare_params: List[Int]
    var search_params: List[Int]
    var full_params: List[Int]
    var selected: Int


def _want(mut plan: AnnSearchPlan, name: String, var shape: List[Int], is_float: Bool):
    plan.names.append(name)
    plan.shapes.append(shape^)
    plan.floating.append(is_float)


def ann_search_plan(r: AnnSearchRequest) -> AnnSearchPlan:
    """The schema and parameter words, in the order the exports take them
    (bindings/_mojolearn_x_ann.mojo)."""
    var plan = AnnSearchPlan(List[String](), List[List[Int]](), List[Bool](), List[Int](), List[Int](),
                             List[Int](), r.k)
    if r.kind == ANN_SEARCH_CAGRA:
        var iterations = r.iterations if r.iterations != 0 else r.itopk
        var seeds = min(r.seeds if r.seeds != 0 else r.itopk + r.width * r.graph_degree, r.n)
        plan.selected = r.itopk if r.filtered else r.k
        _want(plan, "dataset_", [r.n, r.dim], True)
        _want(plan, "graph_", [r.n, r.graph_degree], False)
        plan.prepare_params = [r.n, r.dim, r.graph_degree]
        plan.search_params = [r.m, plan.selected, r.itopk, r.width, iterations, seeds, r.seed, r.status_word]
        plan.full_params = [r.n, r.dim, r.graph_degree, r.m, plan.selected, r.itopk, r.width, iterations, seeds,
                            r.seed, r.status_word]
        return plan^
    _want(plan, "centers_", [r.n_lists, r.dim], True)
    _want(plan, "list_offsets_", [r.n_lists + 1], False)
    _want(plan, "list_indices_", [r.n], False)
    if r.kind == ANN_SEARCH_PQ:
        _want(plan, "codebooks_", [r.pq_dim, 1 << r.pq_bits, r.pq_len], True)
        if not r.device_codes:
            _want(plan, "codes_", [r.n, r.pq_dim], False)
        plan.prepare_params = [r.n, r.dim, r.n_lists, r.pq_dim, r.pq_bits]
        plan.full_params = [r.n, r.dim, r.n_lists, r.pq_dim, r.pq_bits, r.m, r.k, r.probes, r.status_word]
    elif r.kind == ANN_SEARCH_SQ:
        _want(plan, "sq_vmin_", [r.dim], True)
        _want(plan, "sq_delta_", [r.dim], True)
        _want(plan, "codes_", [r.n, r.dim], False)
        plan.prepare_params = [r.n, r.dim, r.n_lists]
        plan.full_params = [r.n, r.dim, r.n_lists, r.m, r.k, r.probes, r.status_word]
    else:
        _want(plan, "codes_", [r.n, rabitq_code_words(r.dim)], False)
        _want(plan, "norms_", [r.n], True)
        _want(plan, "ip_factors_", [r.n], True)
        plan.prepare_params = [r.n, r.dim, r.n_lists, r.seed]
        plan.full_params = [r.n, r.dim, r.n_lists, r.seed, r.m, r.k, r.probes, r.status_word]
    plan.search_params = [r.m, r.k, r.probes, r.status_word]
    return plan^


def ann_handle_addresses(kind: Int, queries: Int, distances: Int, indices: Int, candidates: Int,
                         mask: Int, has_filter: Bool) -> List[Int]:
    """`x_ann_index_search`'s addresses: queries, distances, indices, then
    (IVF kinds) candidates and an explicit filter."""
    var addrs: List[Int] = [queries, distances, indices]
    if kind != ANN_SEARCH_CAGRA:
        addrs.append(candidates)
        if has_filter:
            addrs.append(mask)
    return addrs^


def ann_full_addresses(kind: Int, stored: List[Int], mask: Int, queries: Int, distances: Int, indices: Int,
                       candidates: Int) -> List[Int]:
    """The per-call search export's addresses: the fitted arrays, the filter
    (SQ, RaBitQ), queries, distances, indices, candidates (IVF kinds), the
    filter (PQ)."""
    var addrs = stored.copy()
    if kind == ANN_SEARCH_SQ or kind == ANN_SEARCH_RABITQ:
        addrs.append(mask)
    addrs.append(queries)
    addrs.append(distances)
    addrs.append(indices)
    if kind != ANN_SEARCH_CAGRA:
        addrs.append(candidates)
    if kind == ANN_SEARCH_PQ:
        addrs.append(mask)
    return addrs^


def ann_filter_params(m: Int, n: Int, selected: Int, k: Int) -> List[Int]:
    """`x_ann_filter_topk`'s parameter words."""
    return [m, n, selected, k]
