# SPDX-License-Identifier: Apache-2.0
"""ANN fit and refine door RUN: the checks on typed values, the fitted-array
size plans and the candidate narrowing.

`bindings/neighbors_ann_native.mojo` is the UNPACK/PACK side: it reads the
estimator's fields once, refuses what it cannot convert, allocates and
publishes the fitted arrays, and calls these functions at the point the public
method made each check, so refusal order and wording are unchanged. Nothing
here imports Python.
"""
from std.memory import unsafe_memcpy, MutPointer
from core.admitted_ids import admitted_ids_i32

comptime _ANN_F32P = MutPointer[Float32, MutUntrackedOrigin]
comptime _ANN_I32P = MutPointer[Int32, MutUntrackedOrigin]


# ---------------------------------------------------------------- arrays


def ann_check_dimension(count: Int) raises:
    """`Array`'s refusal of a negative dimension (`_buffer.empty((count,))`)."""
    if count < 0:
        raise Error("mojolearn: negative dimension in shape (" + String(count) + ",)")


def ann_shape_size(shape: List[Int]) -> Int:
    """The element count of a C-order shape."""
    var size = 1
    for axis in shape:  # small-loop(array rank metadata)
        size *= axis
    return size


def ann_c_strides(shape: List[Int], item: Int) -> List[Int]:
    """The C-order byte strides of `shape` for `item`-byte elements."""
    var rank = len(shape)
    var steps = List[Int](length=rank, fill=0)
    var step = item
    for k in range(rank):  # small-loop(array rank metadata)
        var axis = rank - 1 - k
        steps[axis] = step
        step *= shape[axis]
    return steps^


# ---------------------------------------------------------------- IVFIndex


@fieldwise_init
struct IvfFitPlan(Copyable, Movable):
    """Element counts of the IVF-Flat fitted arrays."""
    var centers: Int
    var norms: Int
    var offsets: Int
    var indices: Int
    var data: Int


def ivf_metric_code(code: Float64, metric_repr: String) raises -> Int:
    """`IVFIndex._metric_code` for an exact int or float: 0 (L2Expanded) or
    1 (L2SqrtExpanded)."""
    if code != 0.0 and code != 1.0:
        raise Error("mojolearn IVFIndex: the metric codes are 0 (L2Expanded) and 1 (L2SqrtExpanded), got "
                    + metric_repr)
    return Int(code)


def ivf_check_n_lists(n_lists: Int) raises:
    if n_lists < 1:
        raise Error("mojolearn IVFIndex: n_lists must be at least 1, got " + String(n_lists))


def ivf_fit_plan(n: Int, dim: Int, n_lists: Int) -> IvfFitPlan:
    """The fitted array sizes (the door's allocation refuses a negative one)."""
    return IvfFitPlan(n_lists * dim, n_lists, n_lists + 1, n, n * dim)


# ---------------------------------------------------------------- IVFPQIndex


@fieldwise_init
struct PqFitPlan(Copyable, Movable):
    """IVF-PQ derived sizes: the sub-vector length, the codebook size and the
    element counts of the fitted arrays."""
    var pq_len: Int
    var n_codes: Int
    var centers: Int
    var offsets: Int
    var indices: Int
    var codebooks: Int
    var codes: Int


def pq_fit_plan(n: Int, dim: Int, n_lists: Int, pq_dim: Int, pq_bits: Int) raises -> PqFitPlan:
    """`IVFPQIndex.fit`'s parameter refusals (pq_dim, pq_bits, n_lists, in
    that order) and its sizes."""
    if not (1 <= pq_dim and pq_dim <= dim):
        raise Error("mojolearn IVFPQIndex: pq_dim must be in [1, " + String(dim) + "], got " + String(pq_dim))
    if not (1 <= pq_bits and pq_bits <= 8):
        raise Error("mojolearn IVFPQIndex: pq_bits must be in [1, 8], got " + String(pq_bits))
    if not (1 <= n_lists and n_lists <= n):
        raise Error("mojolearn IVFPQIndex: n_lists must be in [1, " + String(n) + "], got " + String(n_lists))
    # `-(-dim // pq_dim)`: the ceiling of dim / pq_dim (both positive here).
    var pq_len = (dim + pq_dim - 1) // pq_dim
    var n_codes = 1 << pq_bits
    return PqFitPlan(pq_len, n_codes, n_lists * dim, n_lists + 1, n, pq_dim * n_codes * pq_len, n * pq_dim)


# ---------------------------------------------------------------- IVFSQIndex


def ann_check_n_lists(owner: String, n: Int, n_lists: Int) raises:
    """`n_lists` in [1, n] (IVFPQIndex, IVFSQIndex, IVFRaBitQIndex)."""
    if not (1 <= n_lists and n_lists <= n):
        raise Error("mojolearn " + owner + ": n_lists must be in [1, " + String(n) + "], got " + String(n_lists))


def sq_check_train_per_list(train_per_list: Int) raises:
    if train_per_list < 1:
        raise Error("mojolearn IVFSQIndex: max_train_points_per_cluster must be positive or None, "
                    + "got " + String(train_per_list))


# ---------------------------------------------------------------- IVFRaBitQIndex


def rabitq_padded_dim(dim: Int) -> Int:
    """The power-of-two ceiling of the feature count (1 for dim <= 1)."""
    var padded = 1
    while padded < dim:  # small-loop(D: the power-of-two ceiling of the feature count, at most 63 doublings)
        padded *= 2
    return padded


def rabitq_code_words(dim: Int) -> Int:
    """32-bit code words per row over the padded feature count."""
    return (rabitq_padded_dim(dim) + 31) // 32


# ---------------------------------------------------------------- CagraIndex


def cagra_degrees(n: Int, intermediate_graph_degree: Int, graph_degree: Int) -> Tuple[Int, Int]:
    """`min(intermediate_graph_degree, n - 1)`, then `min(graph_degree, kdeg)`."""
    var kdeg = min(intermediate_graph_degree, n - 1)
    var deg = min(graph_degree, kdeg)
    return (kdeg, deg)


def cagra_copy_dataset(dst_address: Int, src_address: Int, count: Int):
    """The owned snapshot of X the index keeps: `count` float32 words."""
    unsafe_memcpy(dest=_ANN_F32P(unsafe_from_address=dst_address),
                  src=_ANN_F32P(unsafe_from_address=src_address), count=count)


# ---------------------------------------------------------------- refine


def refine_metric_root(metric: String, metric_repr: String) raises -> Bool:
    """'sqeuclidean' or 'euclidean' (the binding takes the roots for the
    latter); True when the roots are taken."""
    if metric != "sqeuclidean" and metric != "euclidean":
        raise Error("mojolearn refine: metric must be 'sqeuclidean' or 'euclidean' "
                    + "(inner product is not implemented), got " + metric_repr)
    return metric == "euclidean"


def refine_check_features(query_features: Int, d: Int) raises:
    if query_features != d:
        raise Error("mojolearn refine: queries have " + String(query_features) + " features, the dataset has " + String(d))


def refine_candidates_refusal(m: Int) -> String:
    return "mojolearn refine: candidates must be an integer array of shape (" + String(m) + ", k0)"


def refine_candidates_k0(shape: List[Int], m: Int) raises -> Int:
    """The candidates' column count; refuses a rank or row count other than
    (m, k0)."""
    if len(shape) != 2 or shape[0] != m:
        raise Error(refine_candidates_refusal(m))
    return shape[1]


def refine_check_candidate_kind(integer: Bool, m: Int) raises:
    if not integer:
        raise Error(refine_candidates_refusal(m))


def refine_narrow_ids(address: Int, s0: Int, s1: Int, m: Int, k0: Int, itemsize: Int, signed: Bool,
                      swap: Bool, dst_address: Int) raises:
    """The (m, k0) candidate view narrowed to C-order int32 words at
    `dst_address`. The kernel skips every id outside [0, n)
    (refine_core.refine_cell); an id beyond int32 is refused here."""
    if m * k0 <= 0:
        return
    var fits = admitted_ids_i32(address, s0, s1, m, k0, itemsize, signed, swap,
                                _ANN_I32P(unsafe_from_address=dst_address))
    if not fits:
        raise Error("mojolearn refine: a candidate index exceeds int32")


def refine_check_status(status: Int) raises:
    """The dataset's and queries' finite status word (one refusal for either)."""
    if status != 0:
        raise Error("Input dataset or queries contains NaN or infinity (refine).")
