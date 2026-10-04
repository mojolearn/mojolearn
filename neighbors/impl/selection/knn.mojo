# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The k-NN classifier's vote and the k-NN regressor's mean, over a finished
neighbour search.

Reference: `cuml/cpp/src_prims/selection/knn.cuh` (cuML `00094f7`,
branch-25.08): `get_lbls` (`:40`), `class_probs_kernel` (`:52`),
`class_vote_kernel` (`:74`), `regress_avg_kernel` (`:112`), `class_probs`
(`:158`), `knn_classify` (`:233`), `knn_regress` (`:310`). Behavior matches
except where a DEVIATION BLOCK says so.

This is `MLCommon::Selection`. The `ML::` entry points a caller reaches
(`knn.cu:328-389`, which compute the unique-label sets and then call here)
are in `neighbors/impl/knn/knn.mojo`, one directory over, mirroring
`src/knn/` beside `src_prims/selection/` exactly as cuML lays them out.

WHAT THE THREE KERNELS ARE, IN ONE LINE EACH
--------------------------------------------
- `class_probs_kernel`: one thread per query row; for each of the `k`
  neighbour slots IN SLOT ORDER, `out[row, label] += 1/k`. A SERIAL float
  fold of `k` terms per (row, class).
- `class_vote_kernel`: one thread per row; argmax over the classes with a
  STRICT `>` against `cur_max = -1.0`, so a tie between classes goes to the
  LOWEST CLASS INDEX in the sorted unique-label array -- their docstring at
  `:208-210` says so and it is what scikit-learn's `mode` does too. The
  output is the ORIGINAL label value, `unique_labels[cur_label]`.
- `regress_avg_kernel`: one thread per row; `pred += y[neighbour]` over the
  `k` slots in slot order, then `pred / k`. The same serial fold.

THE IDENTITY CONTRACT (IDENTITY_PATHS row 32, DEVIATION 542)
------------------------------------------------------------
Both folds are SERIAL PER ROW IN SLOT ORDER in cuML already -- no block
reduction, no atomic, no shuffle -- so their summation ORDER is fixed by
construction and is the same on every backend in BOTH modes. What is NOT
fixed by construction is the two things IDENTITY_PATHS rows 9 and 10 name:
contraction (there is no multiply-add here, so row 9 has nothing to pin)
and DENORMAL POLICY (row 10: a regression target can be denormal, and a
sum of targets can pass through a denormal, which Metal flushes and CUDA
does not). So under IDENTICAL every float seam in these kernels goes
through `ftz`: the loaded target, each partial sum, the quotient, and
`n_neigh_inv`. Under FAST `ftz` compiles away and the kernels are their
literal text. THE SAME LOOP RUNS IN BOTH MODES; the only thing the mode
changes is the flush, which is bit-inert on an FTZ backend (Apple) and
aligns a denormal-honoring one (NVIDIA, AMD) to it. That is why a FAST and
an IDENTICAL run of this file agree bit for bit on this Mac, and why that
agreement is NOT evidence the pin is unreached (`numerics.mojo`'s
`identical_mul_add` docstring, the same lesson).

The SLOT ORDER itself comes from `knn_search`'s host sort: `(distance,
index)` ascending, a total order over the returned set, so under IDENTICAL
(where the SET is the lowest-index tie set, row 11) the fold's operand
sequence is a pure function of the data.

`1.0f / n_neighbors` is one IEEE division, correctly rounded on normals on
every column measured (row 10's division clause); `k` is an integer so it
is exact on every backend, and the probabilities are multiples of it. A
class with `j` votes holds the serial sum of `j` copies of `1/k`, which is
the same float for the same `j` on every row -- so a tie in COUNT is an
exact tie in FLOAT and `class_vote_kernel`'s lowest-index rule decides it
with no rounding in the way. Two DIFFERENT counts differ by at least
`1/k - (k+1) * ulp`, never reordered by rounding for any `k < 2^23`.

WHAT IS NOT HERE
----------------
- `precomp_lbls = true` (the MNMG reduction arm, where `knn_indices` is
  unused because `y` already holds per-row labels): `NOT_IMPLEMENTED.tsv`.
- `class_vote_kernel`'s `label_cache` shared-memory copy of the unique
  labels (`:86-93`): a SCHEDULING choice (where the array is read from),
  not a numeric one; the labels are read from global memory here.
- The round-robin `get_next_usable_stream` per output column: Metal has no
  streams (`metal-hardware-gaps`); the outputs run in sequence on the one
  queue. Scheduling, not numeric.
"""

from std.gpu import block_dim, block_idx, thread_idx
from max.gpu.host import DeviceBuffer, DeviceContext

from core.identity_trace import IdentityTrace
from checks.numerics import ftz
from std.sys.compile import is_defined
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
from neighbors.impl.label.classlabels import make_monotonic, map_label_kernel, LABEL_TPB
from neighbors.impl.selection.distance_weights import (
    weighted_class_probs_kernel,
    weighted_regress_avg_kernel,
    distance_weights_kernel,
)


#: lane/fam-neighbors (2026-10-04), IDENTICAL on every vendor: the vote
#: relabels the training labels with ONE launch over the unique set the
#: caller already holds on the device, instead of `make_monotonic` (a
#: second device sort + unique of all n_index labels, two n_index-sized
#: copies, a host round trip of the set, five drains) and a subtract-one
#: pass, on every predict. Integer only; the tally reads the same column
#: indices, so no output bit moves and the host column is untouched.
#: -D MOJOLEARN_IDN_KNN_DIRECT_RELABEL_OFF (or MOJOLEARN_IDN_ALL_OFF)
#: restores the old sequence.
comptime KNN_IDN_DIRECT_RELABEL = GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL and not (
    is_defined["MOJOLEARN_IDN_KNN_DIRECT_RELABEL_OFF"]() or is_defined["MOJOLEARN_IDN_ALL_OFF"]()
)

#: lane/fam2-neighbors (2026-10-04), IDENTICAL on every vendor, default ON:
#: weights='distance' computes the weights in a kernel
#: (`distance_weights_kernel`, `host_distance_weights`' statements, one
#: thread per query row). Before, the vote copied the n_queries x k
#: distances into a host list element by element, ran the rule on the host,
#: copied the result into a pinned buffer element by element and uploaded
#: it. No bit moves; the host column keeps `host_distance_weights`.
#: cpu3-neighbors (2026-10-04): the kernel is now the only GPU route, in
#: every mode and when traced (the trace records the device weights); the
#: host loop and this define's off arm are gone from the vote. The constant
#: stays for callers that still read it.
comptime KNN_IDN_DEVICE_WEIGHTS = GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL and not (
    is_defined["MOJOLEARN_IDN_KNN_DEVICE_WEIGHTS_OFF"]() or is_defined["MOJOLEARN_IDN_ALL_OFF"]()
)

comptime KNN_TPB_X = 32
"""`template <int TPB_X = 32>` on all three launchers. A scheduling number;
the kernels are one-thread-per-row and read nothing across threads."""


@always_inline
def get_lbls(
    labels: MutPointer[Int32, MutAnyOrigin],
    knn_indices: MutPointer[UInt32, MutAnyOrigin],
    idx: Int,
) -> Int32:
    """`get_lbls<precomp_lbls=false>` (`:40-49`): `labels[knn_indices[idx]]`.

    `knn_indices` is `uint32` here where theirs is `int64_t`: it is what
    `knn_search` returns. A type at the boundary, not an arithmetic
    deviation; it carries no number.
    """
    var neighbor_idx = Int(knn_indices.unsafe_load(idx))
    return labels.unsafe_load(neighbor_idx)


@always_inline
def get_lbls_f32(
    labels: MutPointer[Float32, MutAnyOrigin],
    knn_indices: MutPointer[UInt32, MutAnyOrigin],
    idx: Int,
) -> Float32:
    """`get_lbls<precomp_lbls=false, float>` for the regressor."""
    var neighbor_idx = Int(knn_indices.unsafe_load(idx))
    return labels.unsafe_load(neighbor_idx)


def class_probs_kernel(
    out_p: MutPointer[Float32, MutAnyOrigin],
    knn_indices: MutPointer[UInt32, MutAnyOrigin],
    labels: MutPointer[Int32, MutAnyOrigin],
    n_uniq_labels_in: Int32,
    n_samples_in: Int32,
    n_neighbors_in: Int32,
):
    """`class_probs_kernel<float, false>` (`:52-71`).

    `out` is `n_samples x n_uniq_labels`, zeroed by the caller
    (`cudaMemsetAsync`, `:174`). `labels` is the MONOTONIC 0-based
    relabelling, so `out_label` is a column index.
    """
    var row = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var n_neighbors = Int(n_neighbors_in)
    var i = row * n_neighbors
    var n_neigh_inv = ftz(Float32(1.0) / Float32(n_neighbors))
    if row >= Int(n_samples_in):
        return
    var n_uniq = Int(n_uniq_labels_in)
    for j in range(n_neighbors):
        var out_label = Int(get_lbls(labels, knn_indices, i + j))
        var out_idx = row * n_uniq + out_label
        # THE SERIAL FOLD, one (row, class) cell at a time, in slot order.
        # Row 10 flush at the seam; the operand from `out` was written by
        # this thread through the same flush.
        out_p.unsafe_store(
            out_idx, ftz(out_p.unsafe_load(out_idx) + n_neigh_inv)
        )


def class_vote_kernel(
    out_p: MutPointer[Int32, MutAnyOrigin],
    class_proba: MutPointer[Float32, MutAnyOrigin],
    unique_labels: MutPointer[Int32, MutAnyOrigin],
    n_uniq_labels_in: Int32,
    n_samples_in: Int32,
    n_outputs_in: Int32,
    output_offset_in: Int32,
):
    """`class_vote_kernel<int>` (`:74-109`), without the `label_cache` arm.

    `cur_max = -1.0`, strict `>`: the FIRST maximal class wins, which is
    the lowest index in the sorted unique-label array. Output is the
    original label value, placed at `out[row * n_outputs + output_offset]`
    (multi-output layout, row-major over outputs).
    """
    var row = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var n_uniq = Int(n_uniq_labels_in)
    var i = row * n_uniq
    if row >= Int(n_samples_in):
        return
    var cur_max = Float32(-1.0)
    var cur_label = -1
    for j in range(n_uniq):
        var cur_proba = class_proba.unsafe_load(i + j)
        if cur_proba > cur_max:
            cur_max = cur_proba
            cur_label = j
    var val = unique_labels.unsafe_load(cur_label)
    out_p.unsafe_store(row * Int(n_outputs_in) + Int(output_offset_in), val)


def regress_avg_kernel(
    out_p: MutPointer[Float32, MutAnyOrigin],
    knn_indices: MutPointer[UInt32, MutAnyOrigin],
    labels: MutPointer[Float32, MutAnyOrigin],
    n_samples_in: Int32,
    n_neighbors_in: Int32,
    n_outputs_in: Int32,
    output_offset_in: Int32,
):
    """`regress_avg_kernel<float, false>` (`:112-131`).

    `pred` is the serial sum over the `k` slots in slot order, then divided
    by `(float) n_neighbors`. Every seam flushed under IDENTICAL (row 10),
    including the LOADED target: Metal flushes a denormal operand before
    the add and CUDA does not, so an unflushed load is a cross-vendor split
    on a fixture with a denormal target.
    """
    var row = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var n_neighbors = Int(n_neighbors_in)
    var i = row * n_neighbors
    if row >= Int(n_samples_in):
        return
    var pred = Float32(0.0)
    for j in range(n_neighbors):
        pred = ftz(pred + ftz(get_lbls_f32(labels, knn_indices, i + j)))
    out_p.unsafe_store(
        row * Int(n_outputs_in) + Int(output_offset_in),
        ftz(pred / Float32(n_neighbors)),
    )


def _subtract_one_kernel(
    buf: MutPointer[Int32, MutAnyOrigin],
    n_in: Int32,
):
    """`raft::linalg::unaryOp<int>(..., [](int input){ return input - 1; })`
    at `:195-200`. The generic is stood in for by the one lambda it is
    instantiated with here."""
    var tid = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if tid < Int(n_in):
        buf.unsafe_store(tid, buf.unsafe_load(tid) - Int32(1))


def _clf_tag(stage: StringSlice, i: Int, n_outputs: Int) -> String:
    """`knn_clf.<stage>` for a single output; `knn_clf.o<i>.<stage>` when
    there are several, because a tag must be unique within a trace
    (`core/identity_trace.mojo`'s uniqueness invariant) and must name a
    position in the algorithm, never a machine property (rule 2)."""
    if n_outputs == 1:
        return String("knn_clf.") + String(stage)
    return String("knn_clf.o") + String(i) + "." + String(stage)


# ===========================================================================
# DEVIATION BLOCK 556 (2026-09-01): AN ARM cuML DOES NOT HAVE
# THEIRS: `class_probs`, `knn_classify` and `knn_regress` weight every
#   neighbour equally. `weights='distance'` is REFUSED in the Python layer
#   above them (`kneighbors_classifier.pyx:191-193`, `kneighbors_regressor
#   .pyx:188-190`, "Only uniform weighting strategy is supported
#   currently"), so no C++ entry, no kernel and no template parameter for
#   it exists anywhere in cuML or cuVS to cite as a reference.
# OURS: the three functions below take an OPTIONAL per-(query, slot)
#   weight buffer as trailing defaulted parameters. When `has_weights` is
#   False -- which is every existing caller, unedited -- the enqueued
#   kernel and every argument to it are unchanged, so cuML's arm is byte
#   for byte what it was. When it is True the kernel is
#   `neighbors/impl/selection/distance_weights.mojo`'s, whose semantics
#   are scikit-learn's `_get_weights` / `_classification.py:404-410` /
#   `_regression.py` and whose file says so at length.
# REASON: "the reference does not have it" stopped being a reason to refuse
#   on 2026-09-01. A refusal is now legitimate only when the thing is
#   genuinely impossible here (a closed vendor library with no portable
#   path) or when refusing IS the right behaviour for the input. Neither
#   applies: the definition is four lines of arithmetic and this library
#   already goes beyond cuML in several directions.
# WHERE IT LIVES: the KERNELS are in their own file because they have no
#   reference to sit beside; the PARAMETER is here because CONTRIBUTING.md
#   rule 4 wants a departure in the file it departs from, so a reader
#   diffing this file against `knn.cuh` sees it.
# ===========================================================================


def class_probs(
    ctx: DeviceContext,
    mut trace: IdentityTrace,
    mut outs: List[DeviceBuffer[DType.float32]],
    mut knn_indices: DeviceBuffer[DType.uint32],
    mut y: List[DeviceBuffer[DType.int32]],
    n_index_rows: Int,
    n_query_rows: Int,
    k: Int,
    mut uniq_labels: List[DeviceBuffer[DType.int32]],
    n_unique: List[Int],
    mut weights: DeviceBuffer[DType.float32],
    has_weights: Bool = False,
) raises:
    """`class_probs<32, false>` (`:158-205`), one output column at a time.

    `weights` / `has_weights`: DEVIATION 556 above. `weights` is
    `n_query_rows * k` floats in the SAME slot order as `knn_indices` and
    is read only when `has_weights`; pass any buffer otherwise (Mojo's
    launch refuses a null pointer argument).

    `out[i]` is `n_query_rows * n_unique[i]` floats and is zeroed here.
    `uniq_labels[i]` holds `n_unique[i]` sorted labels (from
    `getUniquelabels`, `knn.cu:344`).

    The `y_tmp = [y[i]; uniq_labels[i]]` append at `:185-192` is kept: their
    comment says it exists so `make_monotonic` cannot miss a label absent
    from this partition of `y`, which on one GPU cannot happen, and copying
    the guard costs one `n_unique`-sized copy.

    Records `knn_clf.votes` (the tally buffer, `out[i]`) per output, with
    the `.o<i>` infix only when there is more than one output, so a single-
    output trace carries the plain tags `tools/e2u_matrix_fit.py` and the
    checks look for.
    """
    for i in range(len(y)):
        var n_unique_labels = n_unique[i]
        var cur_size = n_query_rows * n_unique_labels
        # ------------------------------------------------------------------
        # DEVIATION BLOCK 543: THE TALLY BUFFER IS BOUNDS-CHECKED
        # THEIRS: `out[i]` is sized by the CALLER from `len(cp.unique(y))`
        #   (`kneighbors_classifier.pyx:317-322`) and `n_unique[i]` is
        #   recomputed HERE from `getUniquelabels` (`knn.cu:379`); nothing
        #   compares the two and `class_probs_kernel` writes
        #   `row * n_uniq_labels + out_label` into whatever was allocated.
        # OURS: raise before the memset if the buffer is short. Same answer
        #   whenever the two sets agree, which is always for one `y`; the
        #   guard turns a silent overrun into a named error when they do
        #   not (a caller that passed a different `y` than it sized by).
        # REASON: a device write past a buffer is not a wrong answer, it is
        #   a corrupted process, and `neighbors/estimator.mojo` policy 7
        #   relies on this to check the wrapper's set against the implementation's.
        # ------------------------------------------------------------------
        if len(outs[i]) < cur_size:
            raise Error(
                "class_probs: output "
                + String(i)
                + " holds "
                + String(len(outs[i]))
                + " floats, the tally needs "
                + String(cur_size)
                + " ("
                + String(n_query_rows)
                + " rows x "
                + String(n_unique_labels)
                + " classes)"
            )
        ctx.enqueue_memset(outs[i], Float32(0.0))
        ctx.synchronize()

        var grid = (n_query_rows + KNN_TPB_X - 1) // KNN_TPB_X

        var y_normalized = ctx.enqueue_create_buffer[DType.int32](
            n_index_rows + n_unique_labels
        )
        comptime if KNN_IDN_DIRECT_RELABEL:
            # lane/fam-neighbors: `uniq_labels[i]` IS `getUniquelabels(y[i])`
            # (`_unique_label_sets`), so the set `make_monotonic` would
            # sort out of [y; uniq] again is the one already on the device.
            # One `map_label_kernel` launch, zero-based, writes the same
            # column index the map-then-subtract-one pair wrote.
            ctx.enqueue_memset(y_normalized, Int32(0))
            ctx.enqueue_function[map_label_kernel](
                uniq_labels[i].unsafe_ptr(),
                Int32(n_unique_labels),
                y[i].unsafe_ptr(),
                y_normalized.unsafe_ptr(),
                Int32(n_index_rows),
                Int32(1),
                grid_dim=((n_index_rows + LABEL_TPB - 1) // LABEL_TPB, 1, 1),
                block_dim=(LABEL_TPB, 1, 1),
            )
        else:
            var y_tmp = ctx.enqueue_create_buffer[DType.int32](
                n_index_rows + n_unique_labels
            )
            ctx.synchronize()
            # `raft::update_device(y_tmp, y[i], n_index_rows)` then
            # `update_device(y_tmp + n_index_rows, uniq_labels[i], n_unique)`:
            # device-to-device copies into the two halves.
            ctx.enqueue_copy(
                dst_buf=y_tmp.create_sub_buffer[DType.int32](0, n_index_rows),
                src_buf=y[i],
            )
            ctx.enqueue_copy(
                dst_buf=y_tmp.create_sub_buffer[DType.int32](
                    n_index_rows, n_unique_labels
                ),
                src_buf=uniq_labels[i],
            )
            ctx.synchronize()

            make_monotonic(
                ctx, y_normalized, y_tmp, n_index_rows + n_unique_labels, False
            )
            ctx.enqueue_function[_subtract_one_kernel](
                y_normalized.unsafe_ptr(),
                Int32(n_index_rows),
                grid_dim=((n_index_rows + 255) // 256, 1, 1),
                block_dim=(256, 1, 1),
            )
            ctx.synchronize()
            _ = y_tmp^

        if has_weights:
            ctx.enqueue_function[weighted_class_probs_kernel](
                outs[i].unsafe_ptr(),
                knn_indices.unsafe_ptr(),
                y_normalized.unsafe_ptr(),
                weights.unsafe_ptr(),
                Int32(n_unique_labels),
                Int32(n_query_rows),
                Int32(k),
                grid_dim=(grid, 1, 1),
                block_dim=(KNN_TPB_X, 1, 1),
            )
        else:
            ctx.enqueue_function[class_probs_kernel](
                outs[i].unsafe_ptr(),
                knn_indices.unsafe_ptr(),
                y_normalized.unsafe_ptr(),
                Int32(n_unique_labels),
                Int32(n_query_rows),
                Int32(k),
                grid_dim=(grid, 1, 1),
                block_dim=(KNN_TPB_X, 1, 1),
            )
        ctx.synchronize()

        if trace.enabled:
            trace.record_device(
                ctx, _clf_tag("votes", i, len(y)), outs[i], cur_size
            )
        _ = y_normalized^


def knn_classify(
    ctx: DeviceContext,
    mut trace: IdentityTrace,
    mut out_buf: DeviceBuffer[DType.int32],
    mut knn_indices: DeviceBuffer[DType.uint32],
    mut y: List[DeviceBuffer[DType.int32]],
    n_index_rows: Int,
    n_query_rows: Int,
    k: Int,
    mut uniq_labels: List[DeviceBuffer[DType.int32]],
    n_unique: List[Int],
    mut weights: DeviceBuffer[DType.float32],
    has_weights: Bool = False,
) raises:
    """`knn_classify<32, false>` (`:233-284`): `class_probs` into temporary
    per-output tallies, then `class_vote_kernel` per output into `out`,
    which is `n_query_rows x y.size()` row-major.

    Records `knn_clf.labels` (the whole `out`) once, after every output has
    voted.
    """
    var probs = List[DeviceBuffer[DType.float32]]()
    for i in range(len(n_unique)):
        probs.append(
            ctx.enqueue_create_buffer[DType.float32](
                n_query_rows * n_unique[i]
            )
        )
    ctx.synchronize()

    class_probs(
        ctx,
        trace,
        probs,
        knn_indices,
        y,
        n_index_rows,
        n_query_rows,
        k,
        uniq_labels,
        n_unique,
        weights,
        has_weights,
    )

    # `class_vote_kernel` IS UNCHANGED ON BOTH ARMS. Its argmax with a
    # strict `>` against `cur_max = -1.0` is scikit-learn's `weighted_mode`
    # tie rule too -- lowest class index in the sorted unique set wins --
    # so the weighted arm needs no second vote kernel, only a differently
    # filled tally.
    var grid = (n_query_rows + KNN_TPB_X - 1) // KNN_TPB_X
    for i in range(len(y)):
        ctx.enqueue_function[class_vote_kernel](
            out_buf.unsafe_ptr(),
            probs[i].unsafe_ptr(),
            uniq_labels[i].unsafe_ptr(),
            Int32(n_unique[i]),
            Int32(n_query_rows),
            Int32(len(y)),
            Int32(i),
            grid_dim=(grid, 1, 1),
            block_dim=(KNN_TPB_X, 1, 1),
        )
        ctx.synchronize()

    if trace.enabled:
        trace.record_device(
            ctx, "knn_clf.labels", out_buf, n_query_rows * len(y)
        )
    _ = probs^


def knn_regress(
    ctx: DeviceContext,
    mut trace: IdentityTrace,
    mut out_buf: DeviceBuffer[DType.float32],
    mut knn_indices: DeviceBuffer[DType.uint32],
    mut y: List[DeviceBuffer[DType.float32]],
    n_index_rows: Int,
    n_query_rows: Int,
    k: Int,
    mut weights: DeviceBuffer[DType.float32],
    has_weights: Bool = False,
) raises:
    """`knn_regress<float, 32, false>` (`:310-331`): one `regress_avg_kernel`
    per output column into `out`, `n_query_rows x y.size()` row-major.

    Records `knn_reg.pred` (the whole `out`) once, after the last output.

    `weights` / `has_weights`: DEVIATION 556.
    """
    var grid = (n_query_rows + KNN_TPB_X - 1) // KNN_TPB_X
    for i in range(len(y)):
        if has_weights:
            ctx.enqueue_function[weighted_regress_avg_kernel](
                out_buf.unsafe_ptr(),
                knn_indices.unsafe_ptr(),
                y[i].unsafe_ptr(),
                weights.unsafe_ptr(),
                Int32(n_query_rows),
                Int32(k),
                Int32(len(y)),
                Int32(i),
                grid_dim=(grid, 1, 1),
                block_dim=(KNN_TPB_X, 1, 1),
            )
        else:
            ctx.enqueue_function[regress_avg_kernel](
                out_buf.unsafe_ptr(),
                knn_indices.unsafe_ptr(),
                y[i].unsafe_ptr(),
                Int32(n_query_rows),
                Int32(k),
                Int32(len(y)),
                Int32(i),
                grid_dim=(grid, 1, 1),
                block_dim=(KNN_TPB_X, 1, 1),
            )
        # `handle.sync_stream(stream)` at `:328`, theirs, per output.
        ctx.synchronize()
    if trace.enabled:
        trace.record_device(
            ctx, "knn_reg.pred", out_buf, n_query_rows * len(y)
        )


def device_distance_weights(
    ctx: DeviceContext,
    dist_ptr: MutPointer[Float32, MutUntrackedOrigin],
    mut d_w: DeviceBuffer[DType.float32],
    n_queries: Int,
    k: Int,
) raises:
    """KNN_IDN_DEVICE_WEIGHTS: `host_distance_weights` over the n_queries x k
    sorted distances at `dist_ptr`, written to `d_w` on the device. One
    upload, one launch, one 4-byte flag read (the rule's two refusals)."""
    if n_queries <= 0 or k <= 0:
        raise Error(
            "knn weights='distance': n_queries and k must be positive, got "
            + String(n_queries) + ", " + String(k)
        )
    var d_dist = ctx.enqueue_create_buffer[DType.float32](n_queries * k)
    ctx.enqueue_copy(dst_buf=d_dist, src_ptr=dist_ptr)
    var d_flag = ctx.enqueue_create_buffer[DType.int32](1)
    ctx.enqueue_memset(d_flag, Int32(0))
    ctx.enqueue_function[distance_weights_kernel](
        d_dist.unsafe_ptr(), d_w.unsafe_ptr(), d_flag.unsafe_ptr(),
        Int32(n_queries), Int32(k),
        grid_dim=(n_queries + KNN_TPB_X - 1) // KNN_TPB_X, block_dim=KNN_TPB_X,
    )
    var h_flag = List[Int32](length=1, fill=Int32(0))
    ctx.enqueue_copy(dst_ptr=h_flag.unsafe_ptr(), src_buf=d_flag)
    ctx.synchronize()
    var got = Int(h_flag[0])
    _ = h_flag^
    _ = d_flag^
    _ = d_dist^
    if got == 1:
        raise Error(
            "knn weights='distance': a query row has a NEGATIVE distance, which no"
            " implemented metric can produce; refusing rather than weighting it"
        )
    if got != 0:
        raise Error(
            "knn weights='distance': every neighbour of a query row has 1/d that"
            " underflows float32, so the row's normalizer is zero on an FTZ column"
            " and not on a denormal-honoring one (DEVIATION 555)"
        )


def index_range_kernel(
    idx: MutPointer[UInt32, MutAnyOrigin],
    flag: MutPointer[Int32, MutAnyOrigin],
    count: Int,
    n_index: Int,
):
    """cpu3-neighbors: the precomputed-neighbours bounds check as a kernel,
    one thread per slot. A slot naming a row outside the reference data sets
    flag[0] to 1 (every writer stores the same word, so the race is benign);
    the caller zeroes it and raises on a nonzero value. Integer compares
    only, no float work, so no column can differ."""
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i >= count:
        return
    if UInt64(idx.unsafe_load(i)) >= UInt64(n_index):
        flag.unsafe_store(0, Int32(1))


def device_check_index_range(
    ctx: DeviceContext,
    mut d_idx: DeviceBuffer[DType.uint32],
    count: Int,
    n_index: Int,
) raises:
    """Refuses a precomputed neighbour index outside [0, n_index) on the
    device: one launch over the resident index buffer and one 4-byte flag
    read, instead of a host walk over the n_queries x k indices."""
    if count <= 0:
        return
    var d_flag = ctx.enqueue_create_buffer[DType.int32](1)
    ctx.enqueue_memset(d_flag, Int32(0))
    ctx.enqueue_function[index_range_kernel](
        d_idx.unsafe_ptr(), d_flag.unsafe_ptr(), count, n_index,
        grid_dim=(count + KNN_TPB_X - 1) // KNN_TPB_X, block_dim=KNN_TPB_X,
    )
    var h_flag = List[Int32](length=1, fill=Int32(0))
    ctx.enqueue_copy(dst_ptr=h_flag.unsafe_ptr(), src_buf=d_flag)
    ctx.synchronize()
    var got = Int(h_flag[0])
    _ = h_flag^
    _ = d_flag^
    if got != 0:
        raise Error("precomputed neighbor index outside reference data")
