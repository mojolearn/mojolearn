# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
# Derivative work: the upstream file and its pinned commit are recorded in this lane's DERIVATION_MAP.tsv and in this file's own docstring. See NOTICE.
"""Isolation Forest: the model, `fit`, `score_samples`, `predict`.

MIRRORS `cpp/include/cuml/ensemble/isolation_forest.hpp` (`IF_params`
`:27-34`, `IsolationForestModel` `:38-50`, the `fit` / `score_samples` /
`predict` entries `:114-200`), `cpp/src/isolation_forest/isolation_forest.cuh`
(`compute_global_max_nodes_per_tree` `:29-44`, `IsolationForest::
error_checking` `:50-56`, `::fit` `:70-140`, `::compute_path_lengths`
`:143-162`, `::compute_anomaly_scores` `:165-184`) and
`cpp/src/isolation_forest/isolation_forest.cu` (`compute_c_normalization`
`:27-36`, `score_samples` `:161-177`, `predict` `:201-224`), all at
rapidsai/cuml v26.08.00. Float32 only (their `IsolationForestD` arm is the
same code at T = double; no float64 on device here).

The pair of Thrust transforms (`std::pow(T(2), -path_length / c_n)` and
`score > threshold ? 1 : -1`) are two one-thread-per-row kernels here
(`anomaly_score_kernel`, `predict_labels_kernel`), which is what
`thrust::transform` launches. `get_compact_trees` / Treelite export are
NOT ported (NOT_IMPLEMENTED.tsv).

================= DEVIATION BLOCK =================
DEVIATION 680. NON-FINITE INPUTS ARE REFUSED BY NAME ON THE HOST. Theirs:
`error_checking` asserts `n_rows > 0`, `n_cols > 0`, `n_estimators > 0`
and that the pointer is a device pointer (`isolation_forest.cuh:50-56`),
and nothing rejects NaN or infinity (cuML's Python `check_inputs` does not
either). A NaN row reaches the builder, where `val < candidate_min` is
false for it and a node whose min/max straddle it splits on a threshold
that may itself be NaN; a NaN query traverses by `NaN < thr == false`
(always right). Under IDENTICAL the certified stages (`if.treeNNN.
structure.thr`, `if.pathlen`, `if.scores`) may not carry a computed NaN
(IDENTITY_PATHS row 39: the payload is the vendor's). Ours: `fit` and
`score_samples` scan the host input before any upload and raise naming
the argument (`X`, `X_query`) and the first offending (row, column) in
scikit-learn's wording ("contains NaN" / "contains infinity"). Measured by
`check_if_refusals` (README). The brief's clause (5).

DEVIATION 681. THE SCORE'S `pow(2, y)` IS `identical_pow(2, y)`. Theirs:
`std::pow(T(2), -path_length / c_n)` in a Thrust functor (CUDA libm pow,
not correctly rounded, vendor bits). Ours: `identical_pow(Float32(2), y)`
(`original/numerics.mojo`, row 12's seam: IDENTICAL is the portable
`exp(p * log(x))` through fma-only polynomials, FAST is the stdlib `**`).
`c_n = T(model->c_normalization)` is the double `compute_c_normalization`
rounded to float32 exactly as theirs; the double itself is
`identical_log64` under IDENTICAL (row 18's class: a host libm log). The
README's std-exp sabotage moves scores device-vs-oracle, so the seam is
reached.

DEVIATION 684. `max_depth` AUTO IS AN INTEGER `ceil(log2(n))`. Theirs:
`static_cast<int>(std::ceil(std::log2(static_cast<double>(n_sampled_rows))))`
then `max(., 1)` (`isolation_forest.cuh:86-89`). Ours: the smallest k
with `2^k >= n` (bit arithmetic). Same integer for every n >= 1 when
`log2` is exact at powers of two (every libm is; glibc's `log2(8.0)` is
3.0), and ours cannot be off by one on a platform whose `log2` is not.
Spelling only; gated by `check_if_refusals` over n = 1..4097.
===================================================
"""

from std.math import log2
from std.memory import bitcast
from std.sys import is_defined

from std.gpu import block_dim, block_idx, thread_idx
from max.gpu.host import DeviceBuffer, DeviceContext

from core.identity_trace import IdentityTrace
from isolation_forest.derived.curand.curand_kernel import (
    XorwowTables,
    build_xorwow_tables,
    curand,
    curand_init,
    curandStateXORWOW,
    XORWOW_TABLE_WORDS,
)
from isolation_forest.derived.isolation_forest.isolation_tree_builder import (
    IF_BUILD_TPB,
    IF_DECISION_WORDS,
    IF_PATH_TPB,
    IF_RNG_STATE_WORDS,
    IF_SCRATCH_WORDS_PER_NODE,
    IF_STACK_WORDS,
    build_isolation_trees_global_kernel,
    compute_path_lengths_global_kernel,
)
#: DIAGNOSTIC (2026-08-29, the RTX 4090 hang): flushed prints on either side
#: of the two launches so a hang is placed at "enqueue" (MAX compiles the
#: kernel there) or at "synchronize" (the kernel is resident). A no-op in
#: every build that does not name it. See isolation_tree_builder.mojo.
comptime DIAG_TRACE = is_defined["MOJOLEARN_IF_DIAG_TRACE"]()

from original.numerics import (
    GLOBAL_NUMERIC_MODE,
    NUMERIC_IDENTICAL,
    ftz,
    identical_log64,
    identical_pow,
)


# ---------------------------------------------------------------------------
# isolation_forest.hpp:27-34
# ---------------------------------------------------------------------------


@fieldwise_init
struct IF_params(Copyable, Movable):
    """`struct IF_params` with their defaults."""

    var n_estimators: Int
    var max_samples: Int
    var max_depth: Int
    var max_features: Int
    var bootstrap: Bool
    var seed: UInt64

    @staticmethod
    def default() -> Self:
        return Self(100, 256, -1, -1, False, 0)


@fieldwise_init
struct IFLaunchKnobs(Copyable, Movable):
    """SCHEDULING knobs (never numeric), here so the gates can vary them:
    `build_tpb` (theirs 128), `path_tpb` (theirs 256), `pad` extra
    elements on every scratch and node buffer, `poison` the value every
    buffer is filled with before use (a launch-invariance gate runs two
    poisons and two paddings and asks for the same bytes)."""

    var build_tpb: Int
    var path_tpb: Int
    var pad: Int
    var poison: Float32

    @staticmethod
    def default() -> Self:
        return Self(IF_BUILD_TPB, IF_PATH_TPB, 0, Float32(0.0))


# ---------------------------------------------------------------------------
# isolation_forest.hpp:38-50 (+ DEVIATION 685's four node arrays)
# ---------------------------------------------------------------------------


struct IsolationForestModel(Movable):
    """`IsolationForestModel<float>`. The node storage is DEVIATION 685's
    four arrays; `tree_n_nodes_host` is a host copy of
    `global_tree_n_nodes` (read once after `fit`) so the card can hash
    each tree's USED nodes and nothing else."""

    var params: IF_params
    var n_features: Int
    var n_features_per_tree: Int
    var n_samples_per_tree: Int
    var c_normalization: Float64
    var max_nodes_per_tree: Int
    var has_feature_indices: Bool
    var global_feature_indices: DeviceBuffer[DType.int32]
    var node_feature: DeviceBuffer[DType.int32]
    var node_threshold: DeviceBuffer[DType.float32]
    var node_left: DeviceBuffer[DType.int32]
    var node_right: DeviceBuffer[DType.int32]
    var global_tree_offsets: DeviceBuffer[DType.int32]
    var global_tree_n_nodes: DeviceBuffer[DType.int32]
    var global_tree_max_depth: DeviceBuffer[DType.int32]
    var tree_n_nodes_host: List[Int32]
    var tree_max_depth_host: List[Int32]
    var fitted: Bool

    def __init__(out self, ctx: DeviceContext) raises:
        self.params = IF_params.default()
        self.n_features = 0
        self.n_features_per_tree = 0
        self.n_samples_per_tree = 0
        self.c_normalization = 0.0
        self.max_nodes_per_tree = 0
        self.has_feature_indices = False
        self.global_feature_indices = ctx.enqueue_create_buffer[DType.int32](1)
        self.node_feature = ctx.enqueue_create_buffer[DType.int32](1)
        self.node_threshold = ctx.enqueue_create_buffer[DType.float32](1)
        self.node_left = ctx.enqueue_create_buffer[DType.int32](1)
        self.node_right = ctx.enqueue_create_buffer[DType.int32](1)
        self.global_tree_offsets = ctx.enqueue_create_buffer[DType.int32](1)
        self.global_tree_n_nodes = ctx.enqueue_create_buffer[DType.int32](1)
        self.global_tree_max_depth = ctx.enqueue_create_buffer[DType.int32](1)
        self.tree_n_nodes_host = List[Int32]()
        self.tree_max_depth_host = List[Int32]()
        self.fitted = False
        ctx.synchronize()


# ---------------------------------------------------------------------------
# isolation_forest.cu:27-36
# ---------------------------------------------------------------------------


def compute_c_normalization(n: Int) -> Float64:
    """`compute_c_normalization(int n)`: host double, `2H(n-1) - 2(n-1)/n`
    with `H(n-1) = log(n-1) + euler_mascheroni`. The `log` is
    `identical_log64` under IDENTICAL (DEVIATION 681)."""
    if n <= 1:
        return 0.0
    if n == 2:
        return 1.0
    var euler_mascheroni = Float64(0.5772156649015329)
    var harmonic_n_minus_1 = identical_log64(Float64(n - 1)) + euler_mascheroni
    return 2.0 * harmonic_n_minus_1 - 2.0 * Float64(n - 1) / Float64(n)


# ---------------------------------------------------------------------------
# isolation_forest.cuh:29-44
# ---------------------------------------------------------------------------


def compute_global_max_nodes_per_tree(max_depth: Int, max_samples: Int) raises -> Int:
    if max_depth < 0:
        raise Error(
            "max_depth must be non-negative, got " + String(max_depth)
        )
    if max_samples <= 0:
        raise Error(
            "max_samples must be positive, got " + String(max_samples)
        )
    var sample_bound = 2 * max_samples - 1
    var depth_bound = sample_bound
    if max_depth < 30:
        depth_bound = (1 << (max_depth + 1)) - 1
    var bounded = sample_bound if sample_bound < depth_bound else depth_bound
    if bounded > 2147483647:
        raise Error(
            "Global-memory Isolation Forest node capacity exceeds int range."
        )
    return bounded


def ceil_log2_int(n: Int) -> Int:
    """DEVIATION 684: the smallest k >= 0 with 2^k >= n (n >= 1)."""
    var k = 0
    while (1 << k) < n:
        k += 1
    return k


# ---------------------------------------------------------------------------
# DEVIATION 680: the host finite scan.
# ---------------------------------------------------------------------------


def check_finite_by_name(
    name: String, x: List[Float32], n_rows: Int, n_cols: Int
) raises:
    """Raise naming `name` and the first non-finite cell. `x` is read as
    `n_rows * n_cols` cells in either layout; the (row, col) printed
    assumes row-major and is labelled `flat` as well."""
    if len(x) < n_rows * n_cols:
        raise Error(
            name
            + " holds "
            + String(len(x))
            + " values, expected n_rows * n_cols = "
            + String(n_rows * n_cols)
        )
    for i in range(n_rows * n_cols):
        var v = x[i]
        var bits = bitcast[DType.uint32](v)
        if (bits & 0x7F800000) == 0x7F800000:
            var what = String("infinity")
            if (bits & 0x007FFFFF) != 0:
                what = String("NaN")
            raise Error(
                "Input "
                + name
                + " contains "
                + what
                + " at flat index "
                + String(i)
                + " (row "
                + String(i // n_cols)
                + ", column "
                + String(i % n_cols)
                + " row-major); Isolation Forest does not accept non-finite"
                + " values (DEVIATION 680)"
            )


# ---------------------------------------------------------------------------
# Host <-> device helpers (padding and poison are IFLaunchKnobs')
# ---------------------------------------------------------------------------


def _upload_f32(
    ctx: DeviceContext, values: List[Float32], n: Int, pad: Int, poison: Float32
) raises -> DeviceBuffer[DType.float32]:
    var buf = ctx.enqueue_create_buffer[DType.float32](n + pad)
    var host = ctx.enqueue_create_host_buffer[DType.float32](n + pad)
    for i in range(n + pad):
        host.unsafe_ptr().unsafe_store(i, poison)
    # DEVIATION 1942, row 10: `_upload_f32` is the ONE seam every feature
    # value crosses onto the device (the fit input, the query input for
    # path lengths and predict). Flushed here, under the pin, so the
    # per-node min/max, the `val < threshold` partition, the
    # `threshold = max_val` fallback and the traversal all read the same
    # value on a flush-to-zero backend and on a denormal-honoring one.
    # `ftz` is a comptime no-op under FAST.
    for i in range(n):
        host.unsafe_ptr().unsafe_store(i, ftz(values[i]))
    ctx.enqueue_copy(dst_buf=buf, src_ptr=host.unsafe_ptr())
    ctx.synchronize()
    _ = host^
    return buf^


def _upload_u32(
    ctx: DeviceContext, values: List[UInt32], n: Int
) raises -> DeviceBuffer[DType.uint32]:
    var buf = ctx.enqueue_create_buffer[DType.uint32](n)
    var host = ctx.enqueue_create_host_buffer[DType.uint32](n)
    for i in range(n):
        host.unsafe_ptr().unsafe_store(i, values[i])
    ctx.enqueue_copy(dst_buf=buf, src_ptr=host.unsafe_ptr())
    ctx.synchronize()
    _ = host^
    return buf^


def _poisoned_f32(
    ctx: DeviceContext, n: Int, pad: Int, poison: Float32
) raises -> DeviceBuffer[DType.float32]:
    var buf = ctx.enqueue_create_buffer[DType.float32](n + pad)
    buf.enqueue_fill(poison)
    ctx.synchronize()
    return buf^


def _poisoned_i32(
    ctx: DeviceContext, n: Int, pad: Int, poison: Float32
) raises -> DeviceBuffer[DType.int32]:
    var buf = ctx.enqueue_create_buffer[DType.int32](n + pad)
    buf.enqueue_fill(Int32(bitcast[DType.int32](poison)))
    ctx.synchronize()
    return buf^


def _poisoned_i64(
    ctx: DeviceContext, n: Int, pad: Int, poison: Float32
) raises -> DeviceBuffer[DType.int64]:
    var buf = ctx.enqueue_create_buffer[DType.int64](n + pad)
    buf.enqueue_fill(Int64(bitcast[DType.int32](poison)))
    ctx.synchronize()
    return buf^


def read_f32(
    ctx: DeviceContext, buf: DeviceBuffer[DType.float32], n: Int
) raises -> List[Float32]:
    var h = ctx.enqueue_create_host_buffer[DType.float32](len(buf))
    ctx.enqueue_copy(dst_ptr=h.unsafe_ptr(), src_buf=buf)
    ctx.synchronize()
    var out = List[Float32]()
    for i in range(n):
        out.append(h.unsafe_ptr().unsafe_load(i))
    _ = h^
    return out^


def read_i32(
    ctx: DeviceContext, buf: DeviceBuffer[DType.int32], n: Int
) raises -> List[Int32]:
    var h = ctx.enqueue_create_host_buffer[DType.int32](len(buf))
    ctx.enqueue_copy(dst_ptr=h.unsafe_ptr(), src_buf=buf)
    ctx.synchronize()
    var out = List[Int32]()
    for i in range(n):
        out.append(h.unsafe_ptr().unsafe_load(i))
    _ = h^
    return out^


def read_i64(
    ctx: DeviceContext, buf: DeviceBuffer[DType.int64], n: Int
) raises -> List[Int64]:
    var h = ctx.enqueue_create_host_buffer[DType.int64](len(buf))
    ctx.enqueue_copy(dst_ptr=h.unsafe_ptr(), src_buf=buf)
    ctx.synchronize()
    var out = List[Int64]()
    for i in range(n):
        out.append(h.unsafe_ptr().unsafe_load(i))
    _ = h^
    return out^




def _mp_f32(buf: DeviceBuffer[DType.float32]) -> MutPointer[Float32, MutAnyOrigin]:
    """A kernel-argument pointer from a buffer bound immutably (the model's
    node arrays are READ by the scorer; the origin cast says so)."""
    return buf.unsafe_ptr().unsafe_mut_cast[True]().unsafe_origin_cast[MutAnyOrigin]()


def _mp_i32(buf: DeviceBuffer[DType.int32]) -> MutPointer[Int32, MutAnyOrigin]:
    return buf.unsafe_ptr().unsafe_mut_cast[True]().unsafe_origin_cast[MutAnyOrigin]()


def _mp_u32(buf: DeviceBuffer[DType.uint32]) -> MutPointer[UInt32, MutAnyOrigin]:
    return buf.unsafe_ptr().unsafe_mut_cast[True]().unsafe_origin_cast[MutAnyOrigin]()


def _mp_i64(buf: DeviceBuffer[DType.int64]) -> MutPointer[Int64, MutAnyOrigin]:
    return buf.unsafe_ptr().unsafe_mut_cast[True]().unsafe_origin_cast[MutAnyOrigin]()


# ---------------------------------------------------------------------------
# The XORWOW tables on the device (DEVIATION 683), built once per context use.
# ---------------------------------------------------------------------------


struct XorwowDeviceTables(Movable):
    var sequence: DeviceBuffer[DType.uint32]
    var offset: DeviceBuffer[DType.uint32]
    var host: XorwowTables

    def __init__(out self, ctx: DeviceContext) raises:
        self.host = build_xorwow_tables()
        self.sequence = _upload_u32(ctx, self.host.sequence, XORWOW_TABLE_WORDS)
        self.offset = _upload_u32(ctx, self.host.offset, XORWOW_TABLE_WORDS)


# ---------------------------------------------------------------------------
# The Thrust transforms of score_samples / predict as kernels.
# ---------------------------------------------------------------------------


def anomaly_score_kernel(
    path_lengths: MutPointer[Float32, MutAnyOrigin],
    scores: MutPointer[Float32, MutAnyOrigin],
    n_in: Int32,
    c_n: Float32,
):
    """`[c_n] __device__(T path_length) { return std::pow(T(2),
    -path_length / c_n); }` (`isolation_forest.cuh:177-181`); DEVIATION
    681's `identical_pow`."""
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i >= Int(n_in):
        return
    var y = ftz(-path_lengths.unsafe_load(i) / c_n)
    scores.unsafe_store(i, ftz(identical_pow(Float32(2.0), y)))


def predict_labels_kernel(
    scores: MutPointer[Float32, MutAnyOrigin],
    predictions: MutPointer[Int32, MutAnyOrigin],
    n_in: Int32,
    threshold: Float32,
):
    """`[threshold] __device__(float score) { return score > threshold ? 1
    : -1; }` (`isolation_forest.cu:212-217`): 1 = anomaly, -1 = normal."""
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i >= Int(n_in):
        return
    var s = scores.unsafe_load(i)
    predictions.unsafe_store(i, Int32(1) if s > threshold else Int32(-1))


# ---------------------------------------------------------------------------
# isolation_forest.cuh:46-185: class IsolationForest<T>
# ---------------------------------------------------------------------------


struct IsolationForest(Movable):
    """`IsolationForest<float>`: the params and the three methods."""

    var params: IF_params

    def __init__(out self, cfg_params: IF_params):
        self.params = cfg_params.copy()

    def error_checking(self, n_rows: Int, n_cols: Int) raises:
        """`:50-56` plus DEVIATION 680's finite scan (done by `fit`, which
        holds the host data). The `is_dev_ptr` assert has no object: the
        input is a host List that `fit` uploads itself."""
        if n_rows <= 0:
            raise Error("Invalid n_rows " + String(n_rows))
        if n_cols <= 0:
            raise Error("Invalid n_cols " + String(n_cols))
        if self.params.n_estimators <= 0:
            raise Error(
                "n_estimators must be > 0, got " + String(self.params.n_estimators)
            )

    def fit(
        self,
        ctx: DeviceContext,
        input_colmajor: List[Float32],
        n_rows: Int,
        n_cols: Int,
        mut model: IsolationForestModel,
        mut trace: IdentityTrace,
        knobs: IFLaunchKnobs = IFLaunchKnobs.default(),
    ) raises:
        """`IsolationForest::fit` (`:70-140`): resolve `n_sampled_rows`,
        `max_depth`, `n_sampled_features`, `c_normalization`,
        `max_nodes_per_tree`; allocate; `build_isolation_forest_global`;
        sync. `input_colmajor` is column-major `n_rows x n_cols` as theirs
        (`isolation_forest.hpp:118`)."""
        self.error_checking(n_rows, n_cols)
        check_finite_by_name("X", input_colmajor, n_rows, n_cols)

        var n_sampled_rows = self.params.max_samples
        if n_rows < n_sampled_rows:
            n_sampled_rows = n_rows
        if n_sampled_rows <= 0:
            raise Error(
                "max_samples must be positive, got " + String(self.params.max_samples)
            )
        var max_depth = self.params.max_depth
        if max_depth <= 0:
            max_depth = ceil_log2_int(n_sampled_rows)
            if max_depth < 1:
                max_depth = 1

        var n_sampled_features = self.params.max_features
        if n_sampled_features <= 0 or n_sampled_features > n_cols:
            n_sampled_features = n_cols

        model.params = self.params.copy()
        model.n_features = n_cols
        model.n_features_per_tree = n_sampled_features
        model.n_samples_per_tree = n_sampled_rows
        model.c_normalization = compute_c_normalization(n_sampled_rows)
        model.max_nodes_per_tree = compute_global_max_nodes_per_tree(
            max_depth, n_sampled_rows
        )
        var n_trees = self.params.n_estimators
        var total_nodes = n_trees * model.max_nodes_per_tree
        if total_nodes > 2147483647:
            raise Error("Isolation Forest node offsets exceed int range.")
        model.has_feature_indices = n_sampled_features < n_cols

        var pad = knobs.pad
        var poison = knobs.poison
        model.node_feature = _poisoned_i32(ctx, total_nodes, pad, poison)
        model.node_threshold = _poisoned_f32(ctx, total_nodes, pad, poison)
        model.node_left = _poisoned_i32(ctx, total_nodes, pad, poison)
        model.node_right = _poisoned_i32(ctx, total_nodes, pad, poison)
        model.global_tree_offsets = _poisoned_i32(ctx, n_trees, pad, poison)
        model.global_tree_n_nodes = _poisoned_i32(ctx, n_trees, pad, poison)
        model.global_tree_max_depth = _poisoned_i32(ctx, n_trees, pad, poison)
        # Theirs allocates the feature-index buffer only when sampling
        # features; ours always (a null pointer has no portable spelling as
        # a kernel argument), and the kernel reads it only when
        # `has_feature_indices`.
        model.global_feature_indices = _poisoned_i32(
            ctx, n_trees * n_sampled_features, pad, poison
        )

        # build_isolation_forest_global (isolation_tree_builder.cuh:377-420)
        var data = _upload_f32(ctx, input_colmajor, n_rows * n_cols, pad, poison)
        var subsample_buffer = _poisoned_f32(
            ctx, n_trees * n_sampled_rows * n_sampled_features, pad, poison
        )
        var sample_indices = _poisoned_i64(ctx, n_trees * n_sampled_rows, pad, poison)
        var work_indices = _poisoned_i32(ctx, n_trees * n_sampled_rows, pad, poison)
        # One scratch buffer per tree carved into three disjoint slices
        # (stack, per-node decisions, final RNG state). ONE kernel argument:
        # Metal caps a kernel at 31 and this one stands at 25.
        var scratch_stride = (
            model.max_nodes_per_tree * IF_SCRATCH_WORDS_PER_NODE
            + IF_RNG_STATE_WORDS
        )
        var stack = _poisoned_i32(ctx, n_trees * scratch_stride, pad, poison)
        var tables = XorwowDeviceTables(ctx)

        # The card's RNG probe: the first 16 draws of tree 0's state, on the
        # host through the same port (a pure function of (seed, 0)).
        if trace.enabled:
            var st = curandStateXORWOW.zero()
            curand_init(
                self.params.seed,
                UInt64(0),
                UInt64(0),
                st,
                tables.host.sequence.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin](),
                tables.host.offset.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin](),
            )
            var probe = List[Int32]()
            for _ in range(16):
                probe.append(Int32(bitcast[DType.int32](curand(st))))
            trace.record_list_i32("if.rng.probe", probe)

        comptime if DIAG_TRACE:
            print("if_diag: build kernel enqueue begin", flush=True)
        ctx.enqueue_function[build_isolation_trees_global_kernel](
            data.unsafe_ptr(),
            Int64(n_rows),
            Int32(n_cols),
            Int32(n_trees),
            Int32(n_sampled_rows),
            Int32(n_sampled_features),
            Int32(max_depth),
            Int32(model.max_nodes_per_tree),
            Int32(1) if self.params.bootstrap else Int32(0),
            self.params.seed,
            Int32(1) if model.has_feature_indices else Int32(0),
            model.global_feature_indices.unsafe_ptr(),
            model.node_feature.unsafe_ptr(),
            model.node_threshold.unsafe_ptr(),
            model.node_left.unsafe_ptr(),
            model.node_right.unsafe_ptr(),
            model.global_tree_offsets.unsafe_ptr(),
            model.global_tree_n_nodes.unsafe_ptr(),
            model.global_tree_max_depth.unsafe_ptr(),
            subsample_buffer.unsafe_ptr(),
            sample_indices.unsafe_ptr(),
            work_indices.unsafe_ptr(),
            stack.unsafe_ptr(),
            tables.sequence.unsafe_ptr(),
            tables.offset.unsafe_ptr(),
            grid_dim=(n_trees, 1, 1),
            block_dim=(knobs.build_tpb, 1, 1),
        )
        comptime if DIAG_TRACE:
            print("if_diag: build kernel enqueue returned, synchronize begin", flush=True)
        ctx.synchronize()
        comptime if DIAG_TRACE:
            print("if_diag: build kernel synchronize returned", flush=True)

        model.tree_n_nodes_host = read_i32(ctx, model.global_tree_n_nodes, n_trees)
        model.tree_max_depth_host = read_i32(ctx, model.global_tree_max_depth, n_trees)
        model.fitted = True

        if trace.enabled:
            var rows = read_i64(ctx, sample_indices, n_trees * n_sampled_rows)
            var feats = read_i32(ctx, model.global_feature_indices, n_trees * n_sampled_features)
            var h_scratch = read_i32(ctx, stack, n_trees * scratch_stride)
            var h_feat = read_i32(ctx, model.node_feature, total_nodes)
            var h_thr = read_f32(ctx, model.node_threshold, total_nodes)
            var h_left = read_i32(ctx, model.node_left, total_nodes)
            var h_right = read_i32(ctx, model.node_right, total_nodes)
            for t in range(n_trees):
                var tag = "if.tree" + _tree_tag(t)
                var n_used = Int(model.tree_n_nodes_host[t])
                var off = t * model.max_nodes_per_tree
                var rows_t = List[Int32]()
                for i in range(n_sampled_rows):
                    rows_t.append(Int32(rows[t * n_sampled_rows + i]))
                trace.record_list_i32(tag + ".rows", rows_t)
                if model.has_feature_indices:
                    var f_t = List[Int32]()
                    for i in range(n_sampled_features):
                        f_t.append(feats[t * n_sampled_features + i])
                    trace.record_list_i32(tag + ".features", f_t)
                var meta = List[Int32]()
                meta.append(model.tree_n_nodes_host[t])
                meta.append(model.tree_max_depth_host[t])
                trace.record_list_i32(tag + ".meta", meta)
                var sf = List[Int32]()
                var sthr = List[Float32]()
                var sl = List[Int32]()
                var sr = List[Int32]()
                for i in range(n_used):
                    sf.append(h_feat[off + i])
                    sthr.append(h_thr[off + i])
                    sl.append(h_left[off + i])
                    sr.append(h_right[off + i])
                trace.record_list_i32(tag + ".structure.feat", sf)
                trace.record_list_f32(tag + ".structure.thr", sthr)
                trace.record_list_i32(tag + ".structure.left", sl)
                trace.record_list_i32(tag + ".structure.right", sr)
                # CARD_GAPS.md items 1-4 for this lane. `structure.thr` is a
                # hash taken on the far side of two washers: the mul-add
                # ABSORBS a divergence in either bound whenever (max - min)
                # is small against |min|, and the repartition fallback
                # OVERWRITES the drawn threshold with max_val. These three
                # stages record the decisions themselves.
                var dbase = t * scratch_stride + (
                    model.max_nodes_per_tree * IF_STACK_WORDS
                )
                var bounds = List[Float32]()
                var choice = List[Int32]()
                for i in range(n_used):
                    var w = dbase + IF_DECISION_WORDS * i
                    bounds.append(bitcast[DType.float32](h_scratch[w + 0]))
                    bounds.append(bitcast[DType.float32](h_scratch[w + 1]))
                    bounds.append(bitcast[DType.float32](h_scratch[w + 2]))
                    choice.append(h_scratch[w + 3])
                    choice.append(h_scratch[w + 4])
                    choice.append(h_scratch[w + 5])
                trace.record_list_f32(tag + ".split.bounds", bounds)
                trace.record_list_i32(tag + ".split.choice", choice)
                var rng_final = List[Int32]()
                var rbase = t * scratch_stride + (
                    model.max_nodes_per_tree * IF_SCRATCH_WORDS_PER_NODE
                )
                for i in range(IF_RNG_STATE_WORDS):
                    rng_final.append(h_scratch[rbase + i])
                trace.record_list_i32(tag + ".rng.final", rng_final)

        _ = data^
        _ = subsample_buffer^
        _ = sample_indices^
        _ = work_indices^
        _ = stack^
        _ = tables^

    def compute_path_lengths(
        self,
        ctx: DeviceContext,
        model: IsolationForestModel,
        input_rowmajor: DeviceBuffer[DType.float32],
        n_rows: Int,
        n_cols: Int,
        mut avg_path_lengths: DeviceBuffer[DType.float32],
        path_tpb: Int,
    ) raises:
        """`IsolationForest::compute_path_lengths` (`:143-162`): `threads =
        256`, one thread per row-major sample."""
        var threads = path_tpb
        var blocks = (n_rows + threads - 1) // threads
        comptime if DIAG_TRACE:
            print("if_diag: path kernel enqueue begin", flush=True)
        ctx.enqueue_function[compute_path_lengths_global_kernel](
            _mp_f32(input_rowmajor),
            Int64(n_rows),
            Int32(n_cols),
            _mp_i32(model.node_feature),
            _mp_f32(model.node_threshold),
            _mp_i32(model.node_left),
            _mp_i32(model.node_right),
            _mp_i32(model.global_tree_offsets),
            Int32(self.params.n_estimators),
            avg_path_lengths.unsafe_ptr(),
            grid_dim=(blocks, 1, 1),
            block_dim=(threads, 1, 1),
        )
        comptime if DIAG_TRACE:
            print("if_diag: path kernel enqueue returned, synchronize begin", flush=True)
        ctx.synchronize()
        comptime if DIAG_TRACE:
            print("if_diag: path kernel synchronize returned", flush=True)

    def compute_anomaly_scores(
        self,
        ctx: DeviceContext,
        model: IsolationForestModel,
        avg_path_lengths: DeviceBuffer[DType.float32],
        n_rows: Int,
        mut scores: DeviceBuffer[DType.float32],
        path_tpb: Int,
    ) raises:
        """`IsolationForest::compute_anomaly_scores` (`:165-184`): `c_n =
        T(model->c_normalization)`; `c_n <= 0` fills 0.5, else the pow
        transform (DEVIATION 681)."""
        var c_n = Float32(model.c_normalization)
        if c_n <= Float32(0.0):
            scores.enqueue_fill(Float32(0.5))
        else:
            var blocks = (n_rows + path_tpb - 1) // path_tpb
            ctx.enqueue_function[anomaly_score_kernel](
                _mp_f32(avg_path_lengths),
                scores.unsafe_ptr(),
                Int32(n_rows),
                c_n,
                grid_dim=(blocks, 1, 1),
                block_dim=(path_tpb, 1, 1),
            )
        ctx.synchronize()


def _tree_tag(t: Int) -> String:
    var s = String(t)
    while s.byte_length() < 3:
        s = "0" + s
    return s


# ---------------------------------------------------------------------------
# isolation_forest.cu:161-224: the C entries
# ---------------------------------------------------------------------------


def fit(
    ctx: DeviceContext,
    mut forest: IsolationForestModel,
    input_colmajor: List[Float32],
    n_rows: Int,
    n_cols: Int,
    params: IF_params,
    mut trace: IdentityTrace,
    knobs: IFLaunchKnobs = IFLaunchKnobs.default(),
) raises:
    """`ML::fit(handle, IsolationForestF*, const float*, n_rows, n_cols,
    params)` (`isolation_forest.cu:119-130`)."""
    var if_model = IsolationForest(params)
    if_model.fit(ctx, input_colmajor, n_rows, n_cols, forest, trace, knobs)


def _score_samples_device(
    ctx: DeviceContext,
    forest: IsolationForestModel,
    input_rowmajor: List[Float32],
    n_rows: Int,
    n_cols: Int,
    mut trace: IdentityTrace,
    knobs: IFLaunchKnobs,
) raises -> DeviceBuffer[DType.float32]:
    """The device half of `ML::score_samples` (`isolation_forest.cu:
    161-177`): path lengths, then scores, both left on the device (so
    `predict` thresholds them there as theirs does)."""
    if not forest.fitted:
        raise Error("Model has not been fitted. Call fit() first.")
    if n_rows <= 0:
        raise Error("Invalid n_rows " + String(n_rows))
    if n_cols != forest.n_features:
        raise Error(
            "X_query has "
            + String(n_cols)
            + " features, the model was fitted with "
            + String(forest.n_features)
        )
    check_finite_by_name("X_query", input_rowmajor, n_rows, n_cols)
    var if_model = IsolationForest(forest.params)
    var data = _upload_f32(ctx, input_rowmajor, n_rows * n_cols, knobs.pad, knobs.poison)
    var avg_path_lengths = _poisoned_f32(ctx, n_rows, knobs.pad, knobs.poison)
    var scores = _poisoned_f32(ctx, n_rows, knobs.pad, knobs.poison)
    if_model.compute_path_lengths(
        ctx, forest, data, n_rows, n_cols, avg_path_lengths, knobs.path_tpb
    )
    trace.record_device[DType.float32](ctx, "if.pathlen", avg_path_lengths, n_rows)
    if_model.compute_anomaly_scores(
        ctx, forest, avg_path_lengths, n_rows, scores, knobs.path_tpb
    )
    trace.record_device[DType.float32](ctx, "if.scores", scores, n_rows)
    _ = data^
    _ = avg_path_lengths^
    return scores^


def score_samples(
    ctx: DeviceContext,
    forest: IsolationForestModel,
    input_rowmajor: List[Float32],
    n_rows: Int,
    n_cols: Int,
    mut trace: IdentityTrace,
    knobs: IFLaunchKnobs = IFLaunchKnobs.default(),
) raises -> List[Float32]:
    """`ML::score_samples` (`isolation_forest.cu:161-177`), PAPER
    convention (1 = anomaly, 0.5 = normal). The Python layer negates
    (`isolation_forest.pyx:959`). The card records `if.pathlen` and
    `if.scores`."""
    var scores = _score_samples_device(
        ctx, forest, input_rowmajor, n_rows, n_cols, trace, knobs
    )
    var out = read_f32(ctx, scores, n_rows)
    _ = scores^
    return out^


def path_lengths(
    ctx: DeviceContext,
    forest: IsolationForestModel,
    input_rowmajor: List[Float32],
    n_rows: Int,
    n_cols: Int,
    knobs: IFLaunchKnobs = IFLaunchKnobs.default(),
) raises -> List[Float32]:
    """The average path lengths alone (their `compute_path_lengths`
    output, what the Treelite export predicts)."""
    if not forest.fitted:
        raise Error("Model has not been fitted. Call fit() first.")
    check_finite_by_name("X_query", input_rowmajor, n_rows, n_cols)
    var if_model = IsolationForest(forest.params)
    var data = _upload_f32(ctx, input_rowmajor, n_rows * n_cols, knobs.pad, knobs.poison)
    var avg_path_lengths = _poisoned_f32(ctx, n_rows, knobs.pad, knobs.poison)
    if_model.compute_path_lengths(
        ctx, forest, data, n_rows, n_cols, avg_path_lengths, knobs.path_tpb
    )
    var out = read_f32(ctx, avg_path_lengths, n_rows)
    _ = data^
    _ = avg_path_lengths^
    return out^


def predict(
    ctx: DeviceContext,
    forest: IsolationForestModel,
    input_rowmajor: List[Float32],
    n_rows: Int,
    n_cols: Int,
    threshold: Float32 = Float32(0.5),
    knobs: IFLaunchKnobs = IFLaunchKnobs.default(),
) raises -> List[Int32]:
    """`ML::predict` (`isolation_forest.cu:201-224`): scores, then `score
    > threshold ? 1 : -1` (1 = anomaly). The Python layer negates."""
    var trace = IdentityTrace.disabled()
    var scores = _score_samples_device(
        ctx, forest, input_rowmajor, n_rows, n_cols, trace, knobs
    )
    var preds = _poisoned_i32(ctx, n_rows, knobs.pad, knobs.poison)
    var blocks = (n_rows + knobs.path_tpb - 1) // knobs.path_tpb
    ctx.enqueue_function[predict_labels_kernel](
        _mp_f32(scores),
        preds.unsafe_ptr(),
        Int32(n_rows),
        threshold,
        grid_dim=(blocks, 1, 1),
        block_dim=(knobs.path_tpb, 1, 1),
    )
    ctx.synchronize()
    var out = read_i32(ctx, preds, n_rows)
    _ = scores^
    _ = preds^
    return out^
