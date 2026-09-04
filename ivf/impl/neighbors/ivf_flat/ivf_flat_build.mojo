# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""IVF-FLAT's build: train the coarse quantizer, assign, lay the lists out.

PORT OF `cuvs/src/neighbors/ivf_flat/ivf_flat_build.cuh` at cuVS `6ba2ce2`:
`build` (`:390-444`) and the `extend`-on-build path it takes (`:180-345`),
reduced to the one call `build` makes with `add_data_on_build = true` and
`adaptive_centers = false`.

THEIR THREE STEPS, AND WHICH OF OURS IS WHICH
----------------------------------------------

| theirs | line | ours |
|---|---|---|
| train the quantizer on a strided subsample | `:414-437` | the WHOLE dataset (DEVIATION 1781) through the ported k-means |
| `kmeans::predict` the labels, in batches | `:222-224` | `cluster/impl/cluster/kmeans.mojo::predict`, one call |
| `build_index_kernel` scatters into the lists | `:317-325` | `ivf/checks/list_layout.mojo::build_list_layout` (DEVIATIONS 1782/1783) |

**THE COARSE QUANTIZER IS NOT THEIR QUANTIZER, AND THAT IS DEVIATION 1780.**
`build` at `:432-436` fills `cuvs::cluster::kmeans::balanced_params` and
calls `cuvs::cluster::kmeans::fit`, which dispatches to KMEANS-BALANCED --
a hierarchical, balanced-cluster-size quantizer with its own mesocluster
recursion. This tree has no port of it (`cluster/DERIVATION_MAP.tsv` mirrors
`cuvs::cluster::kmeans`, the Lloyd/k-means++ estimator, and nothing else),
so this build trains the ported Lloyd k-means instead. That is a departure
from `PORTING_RULES.md` 0b-i -- their dispatch goes somewhere we do not
have -- and it is stated at the top of `ivf/README.md` and in
`ivf/NOT_IMPLEMENTED.tsv` rather than buried. Two consequences a reader must
carry:

  - **list sizes are not balanced.** Balanced k-means exists to keep them
    even, which is what makes their scan's per-list work uniform. Ours
    inherits Lloyd's list-size distribution, empty lists included.
  - **the identity status of the coarse centroids is the k-means lane's,
    not this lane's.** `archive/research/UNSUPERVISED_IDENTITY.md` is the file that says
    what it is, and `ivf/README.md` quotes it rather than restating it.

WHICH K-MEANS ENTRY POINT, AND WHY THAT ONE
---------------------------------------------
`cluster/impl/cluster/detail/kmeans.mojo::kmeans_fit_main_traced`, with
`tag_prefix = "ivf.quantizer."`.

NOT `cluster/estimator.mojo::kmeans_fit` and not
`cluster/impl/cluster/kmeans.mojo::fit`, and the reason is the CARD.
Both of those construct their own `IdentityTrace()` internally
(`detail/kmeans.mojo:953`), which reads `MOJOLEARN_IDENTITY_TRACE` and
appends a SECOND record numbered `seq 0` into the file this lane is already
writing. `tools/identity_trace_diff.py` refuses a file whose sequence
numbers restart, so an IVF card built that way would be unreadable -- the
exact defect DEVIATION 518 fixed for k-means|| and DEVIATION 544 for the
k-NN classifier. `kmeans_fit_main_traced` is the sanctioned re-entry: it
takes the caller's trace and prefixes every tag it writes. **DEVIATION
1795.**

The host-side work `cluster/estimator.mojo` does around that call -- the
fixed-point scale from the data, the weight bound, `row_norm_kernel` for
`x_norm` -- is done here for the same reasons, and its policy notes 1, 2
and 3 are the reading. Policy 3 in particular: `fit` leaves `labels`
holding the assignment from BEFORE the last centroid update, so the list
membership has to come from a FRESH `predict` against the final centroids
or every list is one iteration stale. That is not a tidiness point here the
way it is there -- a stale membership is a stale summation set.
"""

from max.gpu.host import DeviceBuffer, DeviceContext

from cluster.impl.cluster.detail.kmeans import kmeans_fit_main_traced
from cluster.impl.cluster.detail.kmeans_common import metric_is_sqrt
from cluster.impl.cluster.kmeans import predict
from cluster.impl.cluster.kmeans_params import (
    INIT_KMEANS_PLUS_PLUS,
    KMeansParams,
)
from core.identity_trace import IdentityTrace
from core.row_norms import NORM_TPB, row_norm_kernel
from ivf.checks.list_layout import build_list_layout
from ivf.impl.neighbors.ivf_flat.ivf_flat_index import (
    IvfFlatIndex,
    IvfFlatIndexParams,
    ivf_index_params_validate,
    ivf_metric_name,
    ivf_validate_data,
)
from checks.fixed_point import choose_scale


def upload_f32(
    ctx: DeviceContext, values: List[Float32]
) raises -> DeviceBuffer[DType.float32]:
    """Host list to device buffer, through a runtime host buffer.

    The second hop is `neighbors/estimator.mojo`'s: `archive/plans/UNWIRED.md:31` records
    that an arbitrary host pointer is not interchangeable with one from
    `enqueue_create_host_buffer` on this stack, and that the failure is
    SILENT.
    """
    var n = len(values)
    if n == 0:
        raise Error("upload_f32: refusing to upload an empty list")
    var buf = ctx.enqueue_create_buffer[DType.float32](n)
    var host = ctx.enqueue_create_host_buffer[DType.float32](n)
    for i in range(n):
        host.unsafe_ptr().unsafe_store(i, values[i])
    ctx.enqueue_copy(dst_buf=buf, src_ptr=host.unsafe_ptr())
    ctx.synchronize()
    _ = host^
    return buf^


def download_f32(
    ctx: DeviceContext, mut buf: DeviceBuffer[DType.float32], n: Int
) raises -> List[Float32]:
    var host = ctx.enqueue_create_host_buffer[DType.float32](n)
    ctx.enqueue_copy(dst_ptr=host.unsafe_ptr(), src_buf=buf)
    ctx.synchronize()
    var out = List[Float32]()
    for i in range(n):
        out.append(host.unsafe_ptr().unsafe_load(i))
    _ = host^
    return out^


def download_u32(
    ctx: DeviceContext, mut buf: DeviceBuffer[DType.uint32], n: Int
) raises -> List[UInt32]:
    var host = ctx.enqueue_create_host_buffer[DType.uint32](n)
    ctx.enqueue_copy(dst_ptr=host.unsafe_ptr(), src_buf=buf)
    ctx.synchronize()
    var out = List[UInt32]()
    for i in range(n):
        out.append(host.unsafe_ptr().unsafe_load(i))
    _ = host^
    return out^


def compute_row_norms(
    ctx: DeviceContext,
    mut a: DeviceBuffer[DType.float32],
    mut a_norm: DeviceBuffer[DType.float32],
    n_rows: Int,
    n_features: Int,
    take_sqrt: Bool,
) raises:
    """`core/row_norms.mojo::row_norm_kernel`, one block per row.

    THE SAME LAUNCH `neighbors/.../knn_brute_force.mojo::compute_norms`
    MAKES, spelled here only because importing across two ported trees for
    a four-line launch is a dependency with no payoff. The KERNEL is the
    k-NN lane's and is not re-implemented; `NORM_TPB` is read from the
    kernel matrix, which is where every block size in this tree lives.

    `take_sqrt` follows the metric, which is their comment at
    `knn_brute_force.cuh:117-118` and the same flag `cluster/` carries: an
    expanded L2 wants the SQUARED norm on both sides and takes the root at
    the very end.
    """
    ctx.enqueue_function[row_norm_kernel](
        a_norm.unsafe_ptr(),
        a.unsafe_ptr(),
        Int32(n_features),
        Int32(1 if take_sqrt else 0),
        grid_dim=(n_rows, 1, 1),
        block_dim=(NORM_TPB, 1, 1),
    )


def plan_quantizer_scale(
    x: List[Float32], n_rows: Int, dim: Int
) raises -> Float64:
    """The fixed-point multiplier for the centroid accumulation.

    `cluster/estimator.mojo::plan_sum_scale`'s arithmetic, over a `List`
    rather than a raw pointer, because this lane holds its data as a list
    and that entry takes a `MutPointer[Float32, MutUntrackedOrigin]`. The
    RULE is theirs and is not re-decided here: `choose_scale` bounds a
    partial sum over any subset of rows, the centroid accumulation forms
    one such sum per feature, so the binding constraint is the worst
    column, and the row count is passed because
    `checks/fixed_point.mojo:55-70` records that stating it buys a scale
    4x finer than the blanket three-bit headroom.

    A shared entry taking a `List` belongs in `cluster/estimator.mojo` and
    would delete this function; that file is another lane's, so it is named
    in `ivf/README.md`'s WHAT IS OWED rather than edited.
    """
    var worst = Float64(0.0)
    for f in range(dim):
        var column = Float64(0.0)
        for r in range(n_rows):
            column += Float64(abs(x[r * dim + f]))
        if column > worst:
            worst = column
    return choose_scale(worst, n_rows)


def ivf_flat_build(
    ctx: DeviceContext,
    mut trace: IdentityTrace,
    params: IvfFlatIndexParams,
    x: List[Float32],
    n_rows: Int,
    dim: Int,
) raises -> IvfFlatIndex:
    """`ivf_flat::build`, `ivf_flat_build.cuh:390-444`.

    Row-major `x` of `n_rows x dim` float32 on the host. Returns the index:
    centroids, centroid norms, and the CSR lists with the original row id
    carried beside every stored vector.

    STAGES RECORDED (the tags, in order):

        ivf.quantizer.*     every stage of the coarse k-means fit, written
                            by `kmeans_fit_main_traced` under this prefix
        ivf.centers         the coarse centroids, [n_lists, dim]
        ivf.center_norms    their squared norms, [n_lists]
        ivf.assign          the assignment against the FINAL centroids
        ivf.list_offsets    the CSR row pointer, [n_lists + 1]
        ivf.list_indices    the carried ORIGINAL row ids, [n_rows]
        ivf.list_data       the permuted vectors, [n_rows, dim]

    `ivf.list_indices` and `ivf.list_data` are recorded as SEPARATE stages
    on purpose, and the separation is the diagnosis exactly the way
    `knn.out_dist` / `knn.out_idx` is: two runs whose `list_data` agrees and
    whose `list_indices` does not have permuted the layout without moving
    the carry, which is the shape of the classic IVF bug, and it is
    invisible in any comparison of the vectors alone.
    """
    ivf_index_params_validate(params, n_rows, dim)
    ivf_validate_data(x, n_rows, dim, "dataset")

    var n_lists = params.n_lists
    var take_sqrt = metric_is_sqrt(params.metric)

    if trace.enabled:
        trace.header(
            String("ivf_flat build: n_rows=")
            + String(n_rows)
            + " dim="
            + String(dim)
            + " n_lists="
            + String(n_lists)
            + " metric="
            + ivf_metric_name(params.metric)
            + " kmeans_n_iters="
            + String(params.kmeans_n_iters)
            + " kmeans_trainset_fraction="
            + String(params.kmeans_trainset_fraction)
            + " seed="
            + String(params.seed)
        )

    var sum_scale = plan_quantizer_scale(x, n_rows, dim)
    # Unit weights, so the weight bound is exactly `n_rows`
    # (`cluster/estimator.mojo`'s note on why the supplied case is summed
    # instead). IVF has no per-row weight: their `build` passes none.
    var weight_scale = choose_scale(Float64(n_rows), n_rows)

    var dx = upload_f32(ctx, x)
    var weights = ctx.enqueue_create_buffer[DType.float32](n_rows)
    weights.enqueue_fill(Float32(1.0))
    var centroids = ctx.enqueue_create_buffer[DType.float32](n_lists * dim)
    var labels = ctx.enqueue_create_buffer[DType.uint32](n_rows)
    var x_norm = ctx.enqueue_create_buffer[DType.float32](n_rows)
    var min_dist = ctx.enqueue_create_buffer[DType.float32](n_rows)
    var center_norm = ctx.enqueue_create_buffer[DType.float32](n_lists)
    ctx.synchronize()

    # `x_norm` MUST EXIST BEFORE `predict`, and `predict` does not compute
    # it -- `cluster/estimator.mojo` records that passing it uninitialized
    # MERGES CLUSTERS, measured on the first run of
    # `check_kmeans_fit_recovers_planted`.
    compute_row_norms(ctx, dx, x_norm, n_rows, dim, take_sqrt)
    ctx.synchronize()

    # `kmeans_n_iters` IS THEIR `max_iter`, `ivf_flat_build.cuh:433`, and
    # `n_init = 1` is cuVS's own default (`kmeans.hpp:28-121`). The
    # tolerance stays `KMeansParams.default()`'s 1e-4, because their
    # `balanced_params` carries no tolerance at all and inventing one would
    # be an improvement.
    var kp = KMeansParams.default()
    kp.n_clusters = n_lists
    kp.init = INIT_KMEANS_PLUS_PLUS
    kp.metric = params.metric
    kp.max_iter = params.kmeans_n_iters
    kp.seed = params.seed
    kp.n_init = 1

    var fit = kmeans_fit_main_traced(
        ctx,
        dx,
        weights,
        centroids,
        labels,
        kp,
        n_rows,
        dim,
        Float32(sum_scale),
        Float32(weight_scale),
        trace,
        String("ivf.quantizer."),
    )
    _ = fit.n_iter

    if trace.enabled:
        trace.record_device(ctx, "ivf.centers", centroids, n_lists * dim)

    compute_row_norms(ctx, centroids, center_norm, n_lists, dim, take_sqrt)
    ctx.synchronize()
    if trace.enabled:
        trace.record_device(ctx, "ivf.center_norms", center_norm, n_lists)

    # THE FRESH ASSIGNMENT. `cluster/estimator.mojo` policy 3: `fit` leaves
    # `labels` holding the assignment from BEFORE the final centroid
    # update. For k-means that is an off-by-one-iteration bug in the
    # returned labels; here it would be an off-by-one-iteration INDEX,
    # because list membership is the summation set of every later top-k.
    #
    # THE TIE RULE COMES FROM THIS CALL AND IS NOT RE-DECIDED HERE
    # (DEVIATION 1789). The assignment argmin carries `raft::argmin_op`'s
    # `(value, key)` total order in both k-means arms (IDENTITY_PATHS row
    # 22; `cluster/impl/distance/fused_distance_nn/simt_kernel.mojo:537`
    # is the compare, `d < val[i] or (d == val[i] and col < key[i])`), so a
    # point equidistant from two centroids goes to the LOWER LIST ID.
    # `check_assignment_ties` gates that; it does not implement it.
    predict(
        ctx, dx, x_norm, centroids, labels, min_dist, kp, n_rows, dim
    )
    ctx.synchronize()
    if trace.enabled:
        trace.record_device(ctx, "ivf.assign", labels, n_rows)

    var host_centers = download_f32(ctx, centroids, n_lists * dim)
    var host_center_norms = download_f32(ctx, center_norm, n_lists)
    var host_labels = download_u32(ctx, labels, n_rows)

    var layout = build_list_layout(host_labels, x, n_rows, dim, n_lists)

    if trace.enabled:
        trace.record_list_i32("ivf.list_offsets", layout.offsets)
        var carried = List[Int32]()
        for i in range(n_rows):
            carried.append(Int32(layout.list_indices[i]))
        trace.record_list_i32("ivf.list_indices", carried)
        trace.record_list_f32("ivf.list_data", layout.list_data)

    _ = dx^
    _ = weights^
    _ = centroids^
    _ = labels^
    _ = x_norm^
    _ = min_dist^
    _ = center_norm^

    return IvfFlatIndex(
        n_lists,
        dim,
        n_rows,
        params.metric,
        host_centers^,
        host_center_norms^,
        layout.offsets.copy(),
        layout.list_indices.copy(),
        layout.list_data.copy(),
        host_labels^,
    )
