# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""IVF-FLAT's build: train the coarse quantizer, assign, lay the lists out.

Reference: `cuvs/src/neighbors/ivf_flat/ivf_flat_build.cuh` (cuVS `6ba2ce2`):
`build` (`:390-444`) and the `extend`-on-build path it takes (`:180-345`),
reduced to the one call `build` makes with `add_data_on_build = true` and
`adaptive_centers = false`.

THE REFERENCE'S THREE STEPS AND THIS IMPLEMENTATION'S
-----------------------------------------------------

| reference | line | here |
|---|---|---|
| train the quantizer on a strided subsample | `:414-437` | the WHOLE dataset (DEVIATION 1781) through the implemented k-means |
| `kmeans::predict` the labels, in batches | `:222-224` | `cluster/impl/kmeans.mojo::predict`, one call |
| `build_index_kernel` scatters into the lists | `:317-325` | `ivf/checks/list_layout.mojo::build_list_layout` (DEVIATIONS 1782/1783) |

**THE COARSE QUANTIZER IS NOT THE REFERENCE QUANTIZER, AND THAT IS DEVIATION 1780.**
`build` at `:432-436` fills `cuvs::cluster::kmeans::balanced_params` and
calls `cuvs::cluster::kmeans::fit`, which dispatches to KMEANS-BALANCED --
a hierarchical, balanced-cluster-size quantizer with its own mesocluster
recursion. This tree has no implementation of it,
so this build trains the implemented Lloyd k-means instead. That is a departure
from `CONTRIBUTING.md` (Algorithms and references) -- the reference dispatch goes somewhere this tree does not
have -- and it is stated at the top of `ivf/README.md` and in
`ivf/NOT_IMPLEMENTED.tsv` rather than buried. Two consequences a reader must
carry:

  - **list sizes are not balanced.** Balanced k-means exists to keep them
    even, which is what makes the reference scan's per-list work uniform. This build
    inherits Lloyd's list-size distribution, empty lists included.
  - **the identity status of the coarse centroids is the k-means lane's,
    not this lane's.** `IDENTITY_PATHS.md` is the file that says
    what it is, and `ivf/README.md` quotes it rather than restating it.

WHICH K-MEANS ENTRY POINT, AND WHY THAT ONE
---------------------------------------------
`cluster/impl/detail/kmeans.mojo::kmeans_fit_main_traced`, with
`tag_prefix = "ivf.quantizer."`.

NOT `cluster/estimator.mojo::kmeans_fit` and not
`cluster/impl/kmeans.mojo::fit`, and the reason is the CARD.
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

# DEVIATION 2486: bulk host staging; stream/lifetime boundaries unchanged.
from bindings.hostptr import copy_f32
from max.gpu.host import DeviceBuffer, DeviceContext

from cluster.impl.detail.kmeans import kmeans_fit_main_traced
from cluster.impl.kmeans import predict
from cluster.impl.kmeans_params import (
    INIT_ARRAY,
    INIT_KMEANS_PLUS_PLUS,
    KMeansParams,
)
from core.identity_trace import IdentityTrace
from core.row_norms import NORM_TPB, row_norm_kernel
from ivf.checks.list_layout import ListLayout, build_list_layout, extend_list_layout
from ivf.impl.neighbors.ivf_flat.ivf_group_device import ivf_extend_layout_device, ivf_list_layout_device
from ivf.impl.neighbors.ivf_flat.ivf_flat_index import (
    IVF_MAGNITUDE_BOUND,
    IvfFlatIndex,
    IvfFlatIndexParams,
    ivf_index_params_validate,
    ivf_metric_name,
    ivf_validate_data,
)
from std.memory import bitcast
from std.gpu import block_dim, block_idx, thread_idx
from checks.fixed_point import choose_scale
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_FAST
from std.sys.compile import is_defined
from std.sys.info import has_apple_gpu_accelerator
from std.memory import memcpy
from x_ann.stage_timer import AnnStages
from x_ann.switches import ANN3_COARSE_SEED, ANN3_HOST_PASSES, ANN3_TRAINSET_COPY
from x_ann.kpp_seed import kpp_seed
from x_ann.kpp_seed_device import kpp_seed_device

comptime IVF_FAST_TRAINSET = (
    GLOBAL_NUMERIC_MODE == NUMERIC_FAST
    and has_apple_gpu_accelerator()
    and not is_defined["MOJOLEARN_IVF_FAST_TRAINSET_OFF"]()
)
"""FAST on Apple: the coarse quantizer trains on at most
`IVF_FAST_ROWS_PER_LIST` rows per list, a seeded uniform sample (FAISS's
rule; cuVS trains on `kmeans_trainset_fraction` of the rows). Every row is
still assigned to the trained centroids."""
comptime IVF_FAST_ROWS_PER_LIST = 256

comptime IVF_FAST_SEED = (
    ANN3_COARSE_SEED
    and GLOBAL_NUMERIC_MODE == NUMERIC_FAST
    and has_apple_gpu_accelerator()
)
"""FAST on Apple, DEFAULT (lane ann-apple3; `-D MOJOLEARN_ANN3_COARSE_SEED_OFF` reverts):
the coarse quantizer is seeded by `x_ann/kpp_seed.mojo` (host k-means++ over
a stride sample of its training rows) and cluster/'s k-means starts from
those seeds (`INIT_ARRAY`). An untraced build only."""

comptime IVF_FAST_SEED_DEVICE = IVF_FAST_SEED and not is_defined["MOJOLEARN_IVF_FAST_SEED_DEVICE_OFF"]()
"""Lane apple-fast-fastonly2, FAST + Apple DEFAULT (`-D
MOJOLEARN_IVF_FAST_SEED_DEVICE_OFF` reverts to the host k-means++). M3 A/Bs:
fastonly2-5-ivf-pq-istella-m3 (no seed -> seed+device) 5976 -> 5253 ms
(-12.1%), recall_at_10 .5995 -> .6071; fastonly2-3-ivf-pq-istella-m3 (host
seed -> device seed) 5516 -> 5243 ms (-4.9%), recall .6017 -> .6071. The
IVF_FAST_SEED k-means++ runs on the device (`x_ann/kpp_seed_device.mojo`: a
distance-update grid plus a fold/selection launch per seed, same HostRng
stream) over the training rows already on the device, instead of on one host
core. FAST bits may move (Float32 blocked sums); recall check paired."""


comptime IVF_FAST_RANDOM_INIT = (
    GLOBAL_NUMERIC_MODE == NUMERIC_FAST
    and has_apple_gpu_accelerator()
    and not is_defined["MOJOLEARN_IVF_COARSE_RANDOM_INIT_OFF"]()
)
#: DEFAULT (FAST + Apple) since 2026-10-04: VSEARCH_ALL A/B on the M3 (afc_ab_def, full board size, 1 run per arm,
#: 2026-10-04, source ad265a028; x_ann/vsearch_fast.mojo has every row): 9/9
#: IVF-family rows faster, recall_at_10 identical. ACCEPT;
#: rollback `-D MOJOLEARN_IVF_COARSE_RANDOM_INIT_OFF` (k-means|| seeding).
"""FAST on Apple, DEFAULT since 2026-10-04 (lane af-vsearch, 2026-10-03;
rollback `-D MOJOLEARN_IVF_COARSE_RANDOM_INIT_OFF`): the coarse quantizer starts from n_lists
DISTINCT training rows drawn with the fit's seed (FAISS's `Clustering`
rule) and handed to cluster/'s k-means as INIT_ARRAY, so its scalable
k-means|| seeding (8 rounds of every-row-to-candidate distances, each with
host readbacks and waits, then the candidates' own Lloyd loop) does not run.
Untraced builds only; an INIT_ARRAY set by `IVF_FAST_SEED` wins. The rows are
gathered on the device (`ivf_gather_rows_kernel`, 2026-10-04); only the
seeded ids are made on the host. Moves FAST
bits (another start): paired recall check."""

comptime IVF_COARSE_FAISS_INIT = IVF_FAST_RANDOM_INIT and is_defined["MOJOLEARN_IVF_COARSE_FAISS_INIT"]()
"""FAST QUALITY FIX candidate (lane apple-fast-q-misc, 2026-10-04), REVERTED
the same day: opt-in `-D MOJOLEARN_IVF_COARSE_FAISS_INIT` (the old
`-D MOJOLEARN_IVF_COARSE_INIT_QOLD` is harmless). OUTCOME (M3 afc_ab_def,
full board size, 1 run per arm, tag rab5-ivfinit): ivf-sq recall_at_10
istella 0.60895 -> 0.51065, taxi 0.83325 -> 0.76685 (worse); ivf-filter /
ivf-pq istella +0.01, taxi -0.001 to -0.002; time -3% to -21%. The quality
loss on ivf-sq decides: IVF_FAST_SEED's k-means++ start is the default
again. What it did: the coarse quantizer starts from
`IVF_FAST_RANDOM_INIT`'s n_lists distinct seeded training rows (FAISS's
`Clustering` rule, the board opponent's) and the `IVF_FAST_SEED` k-means++
seeding is skipped. Before this, `IVF_FAST_SEED` set INIT_ARRAY first, so
the promoted random init never ran (the VSEARCH_ALL A/B's "recall identical"
is that). D^2 seeding on heavy-tailed data spends lists on outlier rows."""
#: Board quality audit 2026-10-04: ivf-sq taxi recall_at_10 FAST 0.83325 vs
#: faiss-cpu 0.857025 (FAST 0.8.34, before IVF_FAST_SEED became the default at
#: f2e71e548: 0.902025; IDENTICAL 0.934975); ivf-filter istella FAST 0.801 vs
#: faiss-cpu 0.84195. Paired recall A/B owed (READY-AB).

comptime IVF_FAST_DEVICE_VALIDATE = (
    GLOBAL_NUMERIC_MODE == NUMERIC_FAST
    and has_apple_gpu_accelerator()
    and not is_defined["MOJOLEARN_IVF_DEVICE_VALIDATE_OFF"]()
)
#: DEFAULT (FAST + Apple) since 2026-10-04: VSEARCH_ALL A/B on the M3 (afc_ab_def, full board size, 1 run per arm,
#: 2026-10-04, source ad265a028; x_ann/vsearch_fast.mojo has every row): 9/9
#: IVF-family rows faster, recall_at_10 identical. ACCEPT;
#: rollback `-D MOJOLEARN_IVF_DEVICE_VALIDATE_OFF` (host word scan).
"""FAST on Apple, DEFAULT since 2026-10-04 (lane af-vsearch; rollback
`-D MOJOLEARN_IVF_DEVICE_VALIDATE_OFF`):
`ivf_validate_data`'s word scan of the n x dim dataset (one host pass over
352 MB on Istella) runs on the device after the upload (`ivf_refused_words_kernel`,
one flag word read back); a refused dataset then runs the host scan, which
raises its own message naming the first offender. The same words accepted
and refused: no bit moves."""
comptime IVF_VALIDATE_BLOCKS = 1024
comptime IVF_VALIDATE_TPB = 256


def ivf_refused_words_kernel(
    x: MutPointer[Float32, MutAnyOrigin], count: Int32, flag: MutPointer[Int32, MutAnyOrigin]
):
    """`ivf_validate_data`'s test per word (`IVF_FAST_DEVICE_VALIDATE`): a word
    whose low 31 bits are at or above 2^63's (NaN, an infinity, |v| >= 2^63)
    sets the flag. A grid-stride loop over IVF_VALIDATE_BLOCKS blocks; every
    writer stores the same 1."""
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var step = Int(block_dim.x) * IVF_VALIDATE_BLOCKS
    var bound = bitcast[DType.uint32](IVF_MAGNITUDE_BOUND)
    var bad = False
    while i < Int(count):
        if (bitcast[DType.uint32](x.unsafe_load(i)) & UInt32(0x7FFFFFFF)) >= bound:
            bad = True
        i += step
    if bad:
        flag.unsafe_store(0, Int32(1))


comptime IVF_GATHER_TPB = 256


def ivf_gather_rows_kernel(
    src: MutPointer[Float32, MutAnyOrigin], rows: MutPointer[Int32, MutAnyOrigin], dim: Int32,
    dst: MutPointer[Float32, MutAnyOrigin],
):
    """`IVF_FAST_RANDOM_INIT`'s start: row `rows[l]` of the row-major `src`
    copied to row `l` of `dst`, one block per destination row, the threads
    striding over the columns. A copy: the same words as the host gather."""
    var l = Int(block_idx.x)
    var d = Int(dim)
    var r = Int(rows.unsafe_load(l))
    var c = Int(thread_idx.x)
    while c < d:
        dst.unsafe_store(l * d + c, src.unsafe_load(r * d + c))
        c += Int(block_dim.x)


def ivf_trainset_rows(n_rows: Int, n_train: Int, seed: UInt64) -> List[Int]:
    """`n_train` distinct row ids, ascending, from a seeded partial
    Fisher-Yates over `0 .. n_rows` (splitmix64)."""
    var perm = List[Int](capacity=n_rows)
    for i in range(n_rows):
        perm.append(i)
    var st = seed ^ UInt64(0x9E3779B97F4A7C15)
    for i in range(n_train):
        st += UInt64(0x9E3779B97F4A7C15)
        var z = st
        z = (z ^ (z >> 30)) * UInt64(0xBF58476D1CE4E5B9)
        z = (z ^ (z >> 27)) * UInt64(0x94D049BB133111EB)
        z = z ^ (z >> 31)
        var j = i + Int(z % UInt64(n_rows - i))
        var t = perm[i]
        perm[i] = perm[j]
        perm[j] = t
    var out = List[Int](capacity=n_train)
    for i in range(n_train):
        out.append(perm[i])
    sort(out)
    return out^


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
    comptime if ANN3_HOST_PASSES:
        # Every column (lane neural-pass35, 2026-10-01; Apple only since
        # lane ann-apple3): the copy reads the caller's list and is drained
        # before the return. The pinned hop below pinned a buffer the size
        # of the dataset per upload and copied into it first: on the L40S
        # the build's 352 MB upload read 752 ms (0.47 GB/s) that way
        # (bench/results/ivf-stages-l40s-20261001.log). The raw host-pointer
        # upload is the one the resident optimizer step and the x_decomp
        # kit use on all three vendors with the same words.
        # lane ann-apple3, Apple only, behind `ANN3_HOST_PASSES`: the copy
        # reads the caller's list (the same words, one host pass fewer;
        # `x_ann/io.mojo` has uploaded this way since lane ann-apple2). The
        # copy is drained before the return, while the list is alive.
        ctx.enqueue_copy(dst_buf=buf, src_ptr=values.unsafe_ptr())
        ctx.synchronize()
    else:
        var host = ctx.enqueue_create_host_buffer[DType.float32](n)
        copy_f32(values.unsafe_ptr(), host.unsafe_ptr(), n)
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
    # lane ann-apple3, behind `ANN3_HOST_PASSES`: one memcpy of the staged
    # words (one append each otherwise)
    comptime if ANN3_HOST_PASSES:
        var moved = List[Float32](length=n, fill=Float32(0.0))
        if n > 0:
            memcpy(dest=moved.unsafe_ptr(), src=host.unsafe_ptr(), count=n)
        _ = host^
        return moved^
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
    comptime if ANN3_HOST_PASSES:
        var moved = List[UInt32](length=n, fill=UInt32(0))
        if n > 0:
            memcpy(dest=moved.unsafe_ptr(), src=host.unsafe_ptr(), count=n)
        _ = host^
        return moved^
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
) raises:
    """`core/row_norms.mojo::row_norm_kernel`, one block per row, SQUARED.

    THE SAME LAUNCH `neighbors/.../knn_brute_force.mojo::compute_norms`
    MAKES, spelled here only because importing across two implemented trees for
    a four-line launch is a dependency with no payoff. The KERNEL is the
    k-NN lane's and is not re-implemented; `NORM_TPB` is read from the
    kernel matrix, which is where every block size in this tree lives.

    ALWAYS THE SQUARED NORM, AND THERE IS NO FLAG TO ASK FOR ANYTHING ELSE.
    Both metrics this lane carries are expanded L2, and the expanded form
    `||q||^2 + ||y||^2 - 2 q.y` wants the square on both sides whether or
    not the root is taken later: their build takes the centre norms with
    `raft::linalg::norm<L2Norm>` and no `sqrt_op` for every metric except
    CosineExpanded (`ivf_flat_build.cuh:351-357`), and their search takes
    the query norms the same way under `L2Expanded` and `L2SqrtExpanded`
    alike (`ivf_flat_search.cuh:110-118`).

    FIXED 2026-09-14. This launch used to take a `take_sqrt` flag that
    every caller filled from `metric_is_sqrt(metric)`. `metric_is_sqrt` is
    k-means's REDUCTION flag (`kmeans_common.cuh:444`, whether the min
    distance is rooted), not a norm flag, and it is true for
    L2SqrtExpanded. Under that metric every norm was rooted, the expanded
    distance became `||q|| + ||y|| - 2 q.y`, which is negative for most
    pairs and clamped to 0, and the search returned all-zero distances and
    ids that are not the nearest rows (seen on the Apple M4 through the
    Python door). The flag is gone rather than defaulted so nothing can
    pass it again.
    """
    ctx.enqueue_function[row_norm_kernel](
        a_norm.unsafe_ptr(),
        a.unsafe_ptr(),
        Int32(n_features),
        Int32(0),
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
    # lane ann-apple3, behind `ANN3_HOST_PASSES`: one pass over the rows
    # with one running sum per column. Each column still adds its rows in
    # ascending order, so every column sum, and the largest of them, is the
    # same Float64.
    comptime if ANN3_HOST_PASSES:
        var columns = List[Float64](length=dim, fill=Float64(0.0))
        var xp = x.unsafe_ptr()
        for r in range(n_rows):
            var b = r * dim
            for f in range(dim):
                columns[f] += Float64(abs(xp.unsafe_load(b + f)))
        var largest = Float64(0.0)
        for f in range(dim):
            if columns[f] > largest:
                largest = columns[f]
        return choose_scale(largest, n_rows)
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
    with_list_data: Bool = True,
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

    `with_list_data = False` (lane ann-apple3; the x_ann indexes, which
    never read the permuted vectors) returns, under `ANN3_HOST_PASSES`, an
    index whose `list_data` is EMPTY; every other field is the same. A
    traced build always lays the vectors out, so the card does not change.

    `ivf.list_indices` and `ivf.list_data` are recorded as SEPARATE stages
    on purpose, and the separation is the diagnosis exactly the way
    `knn.out_dist` / `knn.out_idx` is: two runs whose `list_data` agrees and
    whose `list_indices` does not have permuted the layout without moving
    the carry, which is the shape of the classic IVF bug, and it is
    invisible in any comparison of the vectors alone.
    """
    # lane ann-apple3: MOJOLEARN_ANN_STAGES=1 prints this build's phases
    # (off: no sync, no print)
    var st = AnnStages("ivf_flat_build")
    ivf_index_params_validate(params, n_rows, dim)
    comptime if IVF_FAST_DEVICE_VALIDATE:
        # lane af-vsearch: the extent refusal now (it raises before any scan);
        # the word test runs on the device after the upload below
        if len(x) != n_rows * dim:
            ivf_validate_data(x, n_rows, dim, "dataset")
    else:
        ivf_validate_data(x, n_rows, dim, "dataset")
    st.host("validate")

    var n_lists = params.n_lists

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

    # the quantizer's training rows: all of them, or (FAST) a sample
    var n_train = n_rows
    comptime if IVF_FAST_TRAINSET:
        if n_rows > IVF_FAST_ROWS_PER_LIST * n_lists:
            n_train = IVF_FAST_ROWS_PER_LIST * n_lists
    var xt = List[Float32]()
    if n_train < n_rows:
        var rows = ivf_trainset_rows(n_rows, n_train, UInt64(params.seed))
        comptime if ANN3_TRAINSET_COPY:
            # lane ann-apple3, OPT-IN: one memcpy per sampled row
            xt = List[Float32](length=n_train * dim, fill=Float32(0.0))
            for r in range(n_train):
                memcpy(dest=xt.unsafe_ptr() + r * dim, src=x.unsafe_ptr() + rows[r] * dim, count=dim)
        else:
            xt = List[Float32](capacity=n_train * dim)
            for r in range(n_train):
                var b = rows[r] * dim
                for c in range(dim):
                    xt.append(x[b + c])

    var sum_scale: Float64
    if n_train < n_rows:
        sum_scale = plan_quantizer_scale(xt, n_train, dim)
    else:
        sum_scale = plan_quantizer_scale(x, n_rows, dim)
    # Unit weights, so the weight bound is exactly `n_train`
    # (`cluster/estimator.mojo`'s note on why the supplied case is summed
    # instead). IVF has no per-row weight: their `build` passes none.
    var weight_scale = choose_scale(Float64(n_train), n_train)
    st.host("trainset_scale")

    var dx = upload_f32(ctx, x)
    comptime if IVF_FAST_DEVICE_VALIDATE:
        var dflag = ctx.enqueue_create_buffer[DType.int32](1)
        dflag.enqueue_fill(Int32(0))
        ctx.enqueue_function[ivf_refused_words_kernel](
            dx.unsafe_ptr(), Int32(n_rows * dim), dflag.unsafe_ptr(),
            grid_dim=(IVF_VALIDATE_BLOCKS, 1, 1), block_dim=(IVF_VALIDATE_TPB, 1, 1),
        )
        var hflag = ctx.enqueue_create_host_buffer[DType.int32](1)
        ctx.enqueue_copy(dst_ptr=hflag.unsafe_ptr(), src_buf=dflag)
        ctx.synchronize()
        if hflag.unsafe_ptr().unsafe_load(0) != Int32(0):
            # the host scan names the first offender with its own message
            ivf_validate_data(x, n_rows, dim, "dataset")
        _ = hflag^
        _ = dflag^
        st.host("device_validate")
    if n_train == n_rows:
        xt.append(Float32(0.0))
    var dxt = upload_f32(ctx, xt)
    var weights = ctx.enqueue_create_buffer[DType.float32](n_train)
    weights.enqueue_fill(Float32(1.0))
    var centroids = ctx.enqueue_create_buffer[DType.float32](n_lists * dim)
    var labels = ctx.enqueue_create_buffer[DType.uint32](n_rows)
    var x_norm = ctx.enqueue_create_buffer[DType.float32](n_rows)
    var min_dist = ctx.enqueue_create_buffer[DType.float32](n_rows)
    var center_norm = ctx.enqueue_create_buffer[DType.float32](n_lists)
    ctx.synchronize()
    st.host("upload")

    # `x_norm` MUST EXIST BEFORE `predict`, and `predict` does not compute
    # it -- `cluster/estimator.mojo` records that passing it uninitialized
    # MERGES CLUSTERS, measured on the first run of
    # `check_kmeans_fit_recovers_planted`.
    compute_row_norms(ctx, dx, x_norm, n_rows, dim)
    ctx.synchronize()
    st.host("row_norms")

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
    # KMEANS_LAZY_SHIFT (FAST on Apple default, 2026-10-04) is scoped to the
    # callers that ask for it: the coarse quantizer does (and its k-means||
    # recluster inherits it); a no-op in every other mode
    kp.lazy_shift = True

    comptime if IVF_FAST_SEED and not IVF_COARSE_FAISS_INIT:
        if not trace.enabled and n_train >= n_lists:
            comptime if IVF_FAST_SEED_DEVICE:
                if n_train < n_rows:
                    kpp_seed_device(ctx, dxt, n_train, dim, n_lists, params.seed, 8, centroids)
                else:
                    kpp_seed_device(ctx, dx, n_rows, dim, n_lists, params.seed, 8, centroids)
            else:
                var seeds = List[Float32](length=n_lists * dim, fill=Float32(0.0))
                if n_train < n_rows:
                    kpp_seed(xt, n_train, dim, n_lists, params.seed, 8, seeds)
                else:
                    kpp_seed(x, n_rows, dim, n_lists, params.seed, 8, seeds)
                ctx.enqueue_copy(dst_buf=centroids, src_ptr=seeds.unsafe_ptr())
                ctx.synchronize()
                _ = seeds^
            kp.init = INIT_ARRAY
            st.host("seed")

    comptime if IVF_FAST_RANDOM_INIT:
        # lane af-vsearch: FAISS's coarse start, n_lists distinct training rows
        # (a seeded draw) as INIT_ARRAY; the k-means|| rounds do not run. The
        # row ids come from the seeded generator (`ivf_trainset_rows`, the
        # same seed and index stream as the measured ad265a028 arm, so the
        # same rows); the rows are gathered ON THE DEVICE from the training
        # rows already uploaded (dxt, or dx when every row trains) straight
        # into `centroids` by `ivf_gather_rows_kernel`. Only the n_lists ids
        # cross to the device (2026-10-04: the host gather is gone).
        if not trace.enabled and kp.init == INIT_KMEANS_PLUS_PLUS and n_train >= n_lists:
            var pick = ivf_trainset_rows(n_train, n_lists, UInt64(params.seed) ^ UInt64(0xD1B54A32D192ED03))
            var pick32 = List[Int32](capacity=n_lists)
            for l in range(n_lists):
                pick32.append(Int32(pick[l]))
            var drows = ctx.enqueue_create_buffer[DType.int32](n_lists)
            ctx.enqueue_copy(dst_buf=drows, src_ptr=pick32.unsafe_ptr())
            if n_train < n_rows:
                ctx.enqueue_function[ivf_gather_rows_kernel](
                    dxt.unsafe_ptr(), drows.unsafe_ptr(), Int32(dim), centroids.unsafe_ptr(),
                    grid_dim=(n_lists, 1, 1), block_dim=(IVF_GATHER_TPB, 1, 1),
                )
            else:
                ctx.enqueue_function[ivf_gather_rows_kernel](
                    dx.unsafe_ptr(), drows.unsafe_ptr(), Int32(dim), centroids.unsafe_ptr(),
                    grid_dim=(n_lists, 1, 1), block_dim=(IVF_GATHER_TPB, 1, 1),
                )
            # drained while the id list is alive (the copy reads it)
            ctx.synchronize()
            _ = drows^
            _ = pick32^
            _ = pick^
            kp.init = INIT_ARRAY
            st.host("random_init")

    if n_train < n_rows:
        _ = kmeans_fit_main_traced(
            ctx,
            dxt,
            weights,
            centroids,
            labels,
            kp,
            n_train,
            dim,
            Float32(sum_scale),
            Float32(weight_scale),
            trace,
            String("ivf.quantizer."),
        )
    else:
        _ = kmeans_fit_main_traced(
            ctx,
            dx,
            weights,
            centroids,
            labels,
            kp,
            n_train,
            dim,
            Float32(sum_scale),
            Float32(weight_scale),
            trace,
            String("ivf.quantizer."),
        )

    st.mark(ctx, "kmeans")
    if trace.enabled:
        trace.record_device(ctx, "ivf.centers", centroids, n_lists * dim)

    compute_row_norms(ctx, centroids, center_norm, n_lists, dim)
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
    st.host("predict")
    if trace.enabled:
        trace.record_device(ctx, "ivf.assign", labels, n_rows)

    var host_centers = download_f32(ctx, centroids, n_lists * dim)
    var host_center_norms = download_f32(ctx, center_norm, n_lists)
    var host_labels = download_u32(ctx, labels, n_rows)
    st.host("download")

    var lay_data = True
    comptime if ANN3_HOST_PASSES:
        lay_data = with_list_data or trace.enabled
    # lane cgr4-download-loop: the CSR is built on the device (stable radix
    # sort by label, histogram, scan, gather); the host pass is kept only to
    # name a bad label's first row (an error path)
    var dev_layout = ivf_list_layout_device(ctx, labels, dx, n_rows, dim, n_lists, lay_data)
    if n_rows > 0 and Int(dev_layout[3]) >= n_lists:
        _ = build_list_layout(host_labels, x, n_rows, dim, n_lists, with_data=False)
        raise Error("build_list_layout: a label lies outside [0, n_lists)")
    var layout = ListLayout(
        n_lists, n_rows, dim, dev_layout[0].copy(), dev_layout[1].copy(), dev_layout[2].copy()
    )
    _ = dev_layout^
    st.host("layout")

    if trace.enabled:
        trace.record_list_i32("ivf.list_offsets", layout.offsets)
        var carried = List[Int32]()
        for i in range(n_rows):
            carried.append(Int32(layout.list_indices[i]))
        trace.record_list_i32("ivf.list_indices", carried)
        trace.record_list_f32("ivf.list_data", layout.list_data)

    _ = dx^
    _ = dxt^
    _ = weights^
    _ = centroids^
    _ = labels^
    _ = x_norm^
    _ = min_dist^
    _ = center_norm^

    # lane ann-apple3, behind `ANN3_HOST_PASSES`: the layout's three lists
    # move into the index (copied otherwise, the n_rows x dim vectors among
    # them)
    var out_offsets = List[Int32]()
    var out_indices = List[UInt32]()
    var out_data = List[Float32]()
    comptime if ANN3_HOST_PASSES:
        swap(out_offsets, layout.offsets)
        swap(out_indices, layout.list_indices)
        swap(out_data, layout.list_data)
    else:
        out_offsets = layout.offsets.copy()
        out_indices = layout.list_indices.copy()
        out_data = layout.list_data.copy()
    _ = layout^
    st.host("index")

    return IvfFlatIndex(
        n_lists,
        dim,
        n_rows,
        params.metric,
        host_centers^,
        host_center_norms^,
        out_offsets^,
        out_indices^,
        out_data^,
        host_labels^,
    )


def ivf_flat_extend(
    ctx: DeviceContext,
    index: IvfFlatIndex,
    new_x: List[Float32],
    n_new: Int,
) raises -> IvfFlatIndex:
    """`ivf_flat::extend`, `ivf_flat_build.cuh:180-345`, with
    `adaptive_centers = false` (the only arm implemented; refused by name in
    `ivf_index_params_validate`) (lane/inference-embedding-ivf-cholesky,
    2026-09-15).

    Theirs predicts a label for every new vector with `kmeans::predict`
    against the original centroids, adds the histogram of the new labels to
    the list sizes, resizes the lists and inserts each vector with the id the
    caller passes. Here: the build's own `predict` launch (the same
    `KMeansParams` the build uses, so the same fused argmin and its
    `(distance, list id)` total order, DEVIATION 1789) over the squared row
    norms, then `extend_list_layout`, which appends each new row to its list
    under the id `n_rows + j`. The ids are not a parameter: an id chosen by
    the caller would need a merge to keep each list ascending, and a
    sequential id keeps the index a function of the rows alone. The centres
    and their norms do not move.
    """
    ivf_validate_data(new_x, n_new, index.dim, "extension rows")
    var dim = index.dim
    var n_lists = index.n_lists
    var kp = KMeansParams.default()
    kp.n_clusters = n_lists
    kp.init = INIT_KMEANS_PLUS_PLUS
    kp.metric = index.metric
    kp.n_init = 1

    var dx = upload_f32(ctx, new_x)
    var x_norm = ctx.enqueue_create_buffer[DType.float32](n_new)
    var centroids = upload_f32(ctx, index.centers)
    var labels = ctx.enqueue_create_buffer[DType.uint32](n_new)
    var min_dist = ctx.enqueue_create_buffer[DType.float32](n_new)
    ctx.synchronize()
    compute_row_norms(ctx, dx, x_norm, n_new, dim)
    ctx.synchronize()
    predict(ctx, dx, x_norm, centroids, labels, min_dist, kp, n_new, dim)
    ctx.synchronize()
    var new_labels = download_u32(ctx, labels, n_new)
    _ = dx^
    _ = x_norm^
    _ = centroids^
    _ = min_dist^

    # the extended layout on the device (lane cgr5-owed): the same offsets,
    # ids and rows `extend_list_layout` builds on the host column
    var lay = ivf_extend_layout_device(
        ctx, index.list_offsets, index.list_indices, index.list_data, index.n_rows,
        dim, n_lists, labels, new_x, n_new,
    )
    _ = labels^
    var all_labels = index.labels.copy()
    for j in range(n_new):
        all_labels.append(new_labels[j])
    return IvfFlatIndex(
        n_lists, dim, index.n_rows + n_new, index.metric, index.centers.copy(),
        index.center_norms.copy(), lay[0].copy(), lay[1].copy(),
        lay[2].copy(), all_labels^,
    )
