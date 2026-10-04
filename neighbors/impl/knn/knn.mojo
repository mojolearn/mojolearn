# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""`ML::knn_classify`, `ML::knn_regress`, `ML::knn_class_proba`: the entry
points cuML's Python calls after `kneighbors`.

Reference: `cuml/cpp/src/knn/knn.cu:328-389` (cuML `00094f7`, branch-25.08).
The rest of `knn.cu` -- `brute_force_knn`,
`approx_knn_build_index`, `approx_knn_search` -- is cuVS's `brute_force` /
`ivf_*` behind a cuML facade; the brute-force half of it is
`neighbors/impl/detail/knn_brute_force.mojo` and the approximate
half is OUT OF SCOPE (`NOT_IMPLEMENTED.tsv`).

Each of the three does the same two things: compute the sorted unique-label
set per output column with `raft::label::getUniquelabels` (`:344`, `:379`),
and call the `MLCommon::Selection` primitive of the same name
(`neighbors/impl/selection/knn.mojo`). The regressor has no label set to
compute and is one call.

`knn_indices` is the `uint32` buffer `knn_search` returns (`get_lbls` in
`selection/knn.mojo`), uploaded by the caller; theirs is `int64_t*`.

Records `knn_clf.uniq_labels` (or `.o<i>.uniq_labels` when there are
several outputs): the sorted unique label set, int32, the one stage here
that is neither a vote nor a mean and that every downstream class index is
relative to.
"""

from max.gpu.host import DeviceBuffer, DeviceContext

from core.identity_trace import IdentityTrace
from neighbors.impl.label.classlabels import getUniquelabels
from neighbors.impl.selection.knn import (
    _clf_tag,
    class_probs,
)
from neighbors.impl.selection.knn import knn_classify as selection_knn_classify
from neighbors.impl.selection.knn import knn_regress as selection_knn_regress
from std.sys.compile import is_defined
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL

#: lane/fam2-neighbors (2026-10-04), IDENTICAL on every vendor, default ON:
#: a resident index keeps the training labels and their sorted unique sets
#: on the device after the first vote (`KnnVoteCache`, owned by
#: neighbors/resident_index.mojo's entry). Before, every predict uploaded
#: n_index labels per output and sorted them on the device for the unique
#: set. The same buffers feed the same vote kernels: no bit moves, the host
#: column is untouched. A traced call takes the uncached route (its records).
#: -D MOJOLEARN_IDN_KNN_VOTE_CACHE_OFF (or MOJOLEARN_IDN_ALL_OFF) off.
comptime KNN_IDN_VOTE_CACHE = GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL and not (
    is_defined["MOJOLEARN_IDN_KNN_VOTE_CACHE_OFF"]() or is_defined["MOJOLEARN_IDN_ALL_OFF"]()
)


struct KnnVoteCache(Movable):
    """The classifier vote's inputs that depend on the fit alone: the
    labels per output on the device, their sorted unique sets (device and
    host). Keyed by the label array's address and shape: one fit owns one
    resident handle and one label array, so a key match is the same labels;
    a different key rebuilds."""

    var key: Int
    var n_index: Int
    var n_outputs: Int
    var y: List[DeviceBuffer[DType.int32]]
    var uniq: List[DeviceBuffer[DType.int32]]
    var n_unique: List[Int]
    var uniq_host: List[List[Int32]]

    def __init__(out self):
        self.key = 0
        self.n_index = 0
        self.n_outputs = 0
        self.y = List[DeviceBuffer[DType.int32]]()
        self.uniq = List[DeviceBuffer[DType.int32]]()
        self.n_unique = List[Int]()
        self.uniq_host = List[List[Int32]]()

    def ensure(
        mut self,
        ctx: DeviceContext,
        mut trace: IdentityTrace,
        y_ptr: MutPointer[Int32, MutUntrackedOrigin],
        n_index: Int,
        n_outputs: Int,
    ) raises:
        """Upload the labels and build the unique sets unless this cache
        already holds them for (y_ptr, n_index, n_outputs)."""
        var key = Int(y_ptr)
        if self.key == key and self.n_index == n_index and self.n_outputs == n_outputs and len(self.y) == n_outputs:
            return
        self.key = 0
        self.y = List[DeviceBuffer[DType.int32]]()
        self.uniq = List[DeviceBuffer[DType.int32]]()
        self.n_unique = List[Int]()
        self.uniq_host = List[List[Int32]]()
        for i in range(n_outputs):
            var buf = ctx.enqueue_create_buffer[DType.int32](n_index)
            ctx.enqueue_copy(dst_buf=buf, src_ptr=y_ptr.unsafe_offset(i * n_index))
            self.y.append(buf^)
        ctx.synchronize()
        _unique_label_sets(ctx, trace, self.y, n_index, self.uniq, self.n_unique, self.uniq_host)
        self.key = key
        self.n_index = n_index
        self.n_outputs = n_outputs

    def uniq_copy(self) -> List[List[Int32]]:
        var out = List[List[Int32]]()
        for i in range(len(self.uniq_host)):
            out.append(self.uniq_host[i].copy())
        return out^

    def class_proba(
        mut self,
        ctx: DeviceContext,
        mut trace: IdentityTrace,
        mut outs: List[DeviceBuffer[DType.float32]],
        mut knn_indices: DeviceBuffer[DType.uint32],
        n_query_rows: Int,
        k: Int,
        mut weights: DeviceBuffer[DType.float32],
        has_weights: Bool,
    ) raises -> List[List[Int32]]:
        """`knn_class_proba` over the cached labels and sets."""
        class_probs(
            ctx, trace, outs, knn_indices, self.y, self.n_index, n_query_rows, k,
            self.uniq, self.n_unique, weights, has_weights,
        )
        return self.uniq_copy()

    def classify(
        mut self,
        ctx: DeviceContext,
        mut trace: IdentityTrace,
        mut out_buf: DeviceBuffer[DType.int32],
        mut knn_indices: DeviceBuffer[DType.uint32],
        n_query_rows: Int,
        k: Int,
        mut weights: DeviceBuffer[DType.float32],
        has_weights: Bool,
    ) raises -> List[List[Int32]]:
        """`knn_classify` over the cached labels and sets."""
        selection_knn_classify(
            ctx, trace, out_buf, knn_indices, self.y, self.n_index, n_query_rows, k,
            self.uniq, self.n_unique, weights, has_weights,
        )
        return self.uniq_copy()


comptime KnnVoteCachePointer = Optional[MutPointer[KnnVoteCache, MutAnyOrigin]]


def _unique_label_sets(
    ctx: DeviceContext,
    mut trace: IdentityTrace,
    mut y: List[DeviceBuffer[DType.int32]],
    n_index_rows: Int,
    mut uniq_labels: List[DeviceBuffer[DType.int32]],
    mut n_unique: List[Int],
    mut uniq_host: List[List[Int32]],
) raises:
    """`knn.cu:338-346` / `:373-381`: `getUniquelabels` per output column,
    the result held on the device where `class_probs` and `class_vote`
    read it, and on the host for the caller (`uniq_host`)."""
    for i in range(len(y)):
        var uniq = getUniquelabels(ctx, y[i], n_index_rows)
        var n = len(uniq)
        var dev = ctx.enqueue_create_buffer[DType.int32](n)
        var host = ctx.enqueue_create_host_buffer[DType.int32](n)
        ctx.synchronize()
        for j in range(n):
            host.unsafe_ptr().unsafe_store(j, uniq[j])
        ctx.enqueue_copy(dst_buf=dev, src_ptr=host.unsafe_ptr())
        ctx.synchronize()
        if trace.enabled:
            trace.record_host(
                _clf_tag("uniq_labels", i, len(y)), host.unsafe_ptr(), n
            )
        uniq_labels.append(dev^)
        n_unique.append(n)
        uniq_host.append(uniq^)
        _ = host^


def knn_classify(
    ctx: DeviceContext,
    mut trace: IdentityTrace,
    mut out_buf: DeviceBuffer[DType.int32],
    mut knn_indices: DeviceBuffer[DType.uint32],
    mut y: List[DeviceBuffer[DType.int32]],
    n_index_rows: Int,
    n_query_rows: Int,
    k: Int,
    mut weights: DeviceBuffer[DType.float32],
    has_weights: Bool = False,
) raises -> List[List[Int32]]:
    """`ML::knn_classify` (`knn.cu:328-350`). Returns the sorted unique
    label set per output, so the caller can check it against the class set
    it sized its own buffers by (the pyx sizes `proba` by
    `len(self._classes)`; the C++ recomputes; the two must agree or one of
    them is wrong). Their function returns void; the set is the one thing
    a caller here cannot otherwise see (DEVIATION 544, `neighbors/
    estimator.mojo`)."""
    var uniq_labels = List[DeviceBuffer[DType.int32]]()
    var n_unique = List[Int]()
    var uniq_host = List[List[Int32]]()
    _unique_label_sets(
        ctx, trace, y, n_index_rows, uniq_labels, n_unique, uniq_host
    )
    selection_knn_classify(
        ctx,
        trace,
        out_buf,
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
    _ = uniq_labels^
    return uniq_host^


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
    """`ML::knn_regress` (`knn.cu:352-361`): one call through.

    `weights` / `has_weights` are DEVIATION 556; see
    `neighbors/impl/selection/knn.mojo`'s block and
    `neighbors/impl/selection/distance_weights.mojo`."""
    selection_knn_regress(
        ctx, trace, out_buf, knn_indices, y, n_index_rows, n_query_rows, k,
        weights, has_weights,
    )


def knn_class_proba(
    ctx: DeviceContext,
    mut trace: IdentityTrace,
    mut outs: List[DeviceBuffer[DType.float32]],
    mut knn_indices: DeviceBuffer[DType.uint32],
    mut y: List[DeviceBuffer[DType.int32]],
    n_index_rows: Int,
    n_query_rows: Int,
    k: Int,
    mut weights: DeviceBuffer[DType.float32],
    has_weights: Bool = False,
) raises -> List[List[Int32]]:
    """`ML::knn_class_proba` (`knn.cu:363-389`): the unique sets, then
    `class_probs` straight into the caller's per-output buffers. `out[i]`
    must hold `n_query_rows * n_unique[i]` floats; the caller sized it
    from its own class set and the returned sets let it check.

    Records `knn_clf.proba` per output AFTER the tally (the tally itself
    is `knn_clf.votes`, recorded inside `class_probs`; here the two are
    the same buffer and the pair says so -- in `knn_classify` the tally is
    a temporary and only `labels` is caller-visible)."""
    var uniq_labels = List[DeviceBuffer[DType.int32]]()
    var n_unique = List[Int]()
    var uniq_host = List[List[Int32]]()
    _unique_label_sets(
        ctx, trace, y, n_index_rows, uniq_labels, n_unique, uniq_host
    )
    class_probs(
        ctx,
        trace,
        outs,
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
    if trace.enabled:
        for i in range(len(outs)):
            trace.record_device(
                ctx,
                _clf_tag("proba", i, len(outs)),
                outs[i],
                n_query_rows * n_unique[i],
            )
    _ = uniq_labels^
    return uniq_host^
