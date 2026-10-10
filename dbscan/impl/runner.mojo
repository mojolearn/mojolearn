# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The DBSCAN driver: neighborhood, core points, CSR, label propagation.

Reference: `cuml/cpp/src/dbscan/runner.cuh::run` (cuML `00094f7`). Partial
(single GPU).

THE REFERENCE STRUCTURE, WHICH IS TWO LOOPS OVER THE BATCHES AND NOT ONE
------------------------------------------------------------------------
    loop 1, batches n-1 .. 0 (REVERSED):
        VertexDeg   -> adj (boolean, batch x N) and vd (degrees, batch + 1)
        read vd[n_points] back to the host: the batch's edge count
        maxklen[i] = max(vd[0 .. n_points))    (RBC only, `runner.cuh:289`)
        CorePoints  -> core[i + start] = vd[i] >= min_pts
    allocate adj_graph for the LARGEST batch
    loop 2, batches 0 .. n-1:
        VertexDeg again, EXCEPT for batch 0; the RBC arm takes the ONE-PASS
        max_k form when loop 1's bound fits the spare room (`algo.cuh:119`)
        AdjGraph    -> exclusive scan of vd, then adj_to_csr
        weak_cc_batched -> a labelling of THIS batch's sub-graph, over all N
        MergeLabels -> fold it into the running labelling, except for batch 0
    final_relabel   -> monotonic 0..k-1
    relabelForSkl   -> MAX_LABEL becomes -1, everything else loses one

The reference comment at `runner.cuh:245-246` explains the reversal and is
quoted rather than paraphrased:

    // 1. Compute the part owned by this worker (reversed order of batches to
    // keep the batch 0 in memory)

so that loop 2's first iteration finds batch 0's `adj` and `vd` already
resident and skips one neighborhood pass. Two passes over the data are
unavoidable and are NOT a defect of the implementation: the core mask over the WHOLE
dataset has to exist before any batch is labelled, because `weak_cc`'s
`filter_op` reads `core[j]` for neighbours `j` in every other batch.

The one thing worth understanding before changing anything here is WHERE the
core-point restriction is applied. It is not in the graph. The CSR contains
every edge, including edges out of border points, and the restriction lives
in the labeler's `filter_op` (`runner.cuh:384`). Moving it earlier looks like
an optimization and quietly changes the answer.

WHAT THE PREVIOUS VERSION OF THIS FILE DID INSTEAD, AND WHAT IT COST
---------------------------------------------------------------------
It ran `gemm_nt` (MAX's matmul) into an `m x N` float32 distance buffer, then
`expand_distances_kernel` over that buffer, then a third kernel that read it
back to threshold it -- three kernels and 16 bytes of memory traffic per
pair where `epsUnexpL2SqNeighborhood` is one kernel and one byte. It also
kept ONE `weak_cc` over a CSR built from every row of the dataset, so the
memory that batching the adjacency saved came straight back in `col_ind`,
and it never relabelled, so its output did not match cuML's or sklearn's.
Those were three separate departures from `runner.cuh` and all three are
gone.

DEVIATION BLOCK 29: `need_ja_compute` ON THE WEIGHTED BALL-COVER ARM
--------------------------------------------------------------------
`runner.cuh:257` is

    bool need_ja_compute = sparse_rbc_mode && ((i == 0) || (sample_weight != nullptr));

and the second disjunct is the entire reason the weighted ball-cover arm
works. Loop 1 normally asks the ball cover only to COUNT: it fills `ia` and
`vd` and emits no columns, because the integer degree is all the core-point
test needs. A WEIGHTED degree needs the neighbour IDS, so with weights on,
every batch of loop 1 must also FILL `ja`. That is their line, implemented.

WHERE OURS DIFFERS, AND IT IS THE SAME PLACE DEVIATION 39 ALREADY DIFFERS.
Theirs fills into the resizable `adj_graph` and grows it later
(`rmm::device_uvector::resize` preserves contents); `DeviceBuffer` has no
growing resize, so ours fills into a per-batch scratch sized to that batch's
own edge count, accumulates the weights out of it, and drops it. The cost is
ONE extra fill of batch 0 per fit -- batch 0 is filled here for its weights
and again into `col_ind` once `col_ind` is sized, where theirs reuses the
first fill. It is a duplicated pass over one batch, not a duplicated answer:
`rbc_eps_nn_query_fill` is a pure function of the index and the query rows,
so both fills write the same columns. Named rather than hidden because it is
a real per-fit cost that only the weighted arm pays.

THE OTHER PLACE THE WEIGHT COULD HAVE GONE, AND WHY IT DID NOT. Nothing but
the core-point test reads the weight. The neighborhood, the adjacency, the
CSR, `weak_cc`, `MergeLabels`, `final_relabel` and `relabelForSkl` are
byte-for-byte the unweighted path, which is what makes "uniform weights of 1
reproduce the unweighted labels" a real gate rather than a tautology: the
two runs differ in one buffer and one kernel, and if the weighted degree is
wrong that gate is what would say so. IT HAS NOT SAID ANYTHING YET:
`check_dbscan_uniform_weight_matches_unweighted` first compiled and passed on 2026-09-01, having never compiled before that (the cure was `-O1`, not this lane's source)
(`DeadArgumentElimination surveyUse failed`, an LLVM pass assertion) and the
gate-side workaround on top of `dfb47fc9` is unverified.

NOT IMPLEMENTED, and named in `dbscan/NOT_IMPLEMENTED.tsv`: the multi-GPU arms
(`CorePoints::exchange`, `MergeLabels::tree_reduction`) and the
`core_indices` output (`runner.cuh:419-442`, a `thrust::copy_if` stream
compaction of the core mask). `sample_weight` sat on this list until
2026-09-01 and is now above. The two-loop `max_k` dispatch
(`runner.cuh:257`, `:289`, `:327`, `:335`) briefly sat on this list and is
now IMPLEMENTED below: loop 2 reuses batch 0's CSR from loop 1 and takes the
one-pass arm for the rest whenever `algo.cuh:119`'s spare guard admits it.
"""

from std.gpu import block_dim, block_idx, thread_idx
from std.time import perf_counter_ns
from std.sys.compile import is_defined
from std.sys.info import has_apple_gpu_accelerator
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_FAST, NUMERIC_IDENTICAL
from checks.kernel_matrix import TARGET_COLUMN, COLUMN_NVIDIA, COLUMN_AMD
from max.gpu.host import DeviceBuffer, DeviceContext

from dbscan.impl.adjgraph.algo import (
    adj_graph_run,
    scan_blocks_needed,
)
from core.identity_trace import IdentityTrace
from dbscan.impl.corepoints.compute import (
    core_points_compute,
    core_points_compute_weighted,
)
from dbscan.impl.mergelabels.runner import merge_labels_run
# TOMBSTONE: MOJOLEARN_DBSCAN_FAST_DENSEBALL (DROPPED-slower) deleted 2026-10-09 by lane/owed-deletions-D3; code recoverable at b639a2bd2.
# Tried: dbscan with no edge list: dense-ball cliques, early-exit core counts and union-find over landmark pairs (dbscan/impl/denseball.mojo); dbscan istella A 247,185 ms, B timed out (rab3-denseball).
# Restore: git apply experiments/removed/MOJOLEARN_DBSCAN_FAST_DENSEBALL.patch; record in docs/TOMBSTONES.md.
from dbscan.impl.vertexdeg.algo import (
    weighted_vertex_deg_csr,
    weighted_vertex_deg_dense,
)
from dbscan.impl.neighbors.epsilon_neighborhood import (
    DBSCAN_METRIC_L2,
    dbscan_metric_name,
)
from dbscan.impl.label.classlabels import make_monotonic
from dbscan.impl.sparse.detail.csr import (
    DBSCAN_CC_FLAG_CELLS,
    MAX_LABEL,
    weak_cc_batched,
)
from neighbors.impl.ball_cover.ball_cover import (
    rbc_build_index,
    rbc_n_landmarks,
)
from dbscan.impl.multi_gpu import (
    vertex_deg_dispatch, rbc_eps_nn_query_count,
    rbc_eps_nn_query_fill, rbc_eps_nn_query_max_k,
)
from neighbors.impl.ball_cover.scan import (
    rbc_exclusive_scan_launch,
    RBC_SCAN_TPB,
    rbc_max_reduce_launch,
)
from neighbors.impl.ball_cover.registers import (
    IDN_DBSCAN_ADJ_BITMAP,
    rbc_eps_bitmap_words,
    rbc_eps_pass_count_bitmap,
    rbc_eps_pass_fill_bitmap,
)


#: `EpsNnMethod` (`cuml/cpp/include/cuml/cluster/dbscan.hpp:32`). Their enum,
#: their order, and their DEFAULT: `BRUTE_FORCE` is the value every public
#: signature in `dbscan.hpp` carries (`:74`, `:88`, `:103`, `:117`) and the
#: value cuML's Python layer passes unless the user asks for `'rbc'`
#: (`dbscan.pyx:371-372`, `algorithm='brute'` at `:300`).
#:
#: DEVIATION 35: WE DEFAULT TO RBC AND THEY DEFAULT TO BRUTE_FORCE.
#:
#: Measured on an M4, 8 features, eps 0.30, arms interleaved inside the repeat
#: loop, medians of 3 -- OURS AGAINST OURS, which is the comparison that
#: decides a default:
#:
#:     n          brute ms    rbc ms   speedup
#:     4,000           8.1       8.1     1.00x   (indistinguishable)
#:     16,000         77.5      28.7     2.70x
#:     50,000        808.8     231.7     3.49x
#:     100,000     4,093.6     323.1    12.67x
#:     200,000    17,243.6     626.3    27.53x
#:
#: **RBC wins at every measured size and loses at none.** There is no n at
#: which BRUTE_FORCE is the better choice for a user on this hardware.
#:
#: This does not change any answer. `check_dbscan_rbc_matches_brute` compares
#: the two labellings POINT FOR POINT -- not up to permutation, because
#: `final_relabel` + `relabelForSkl` (`runner.cuh:410-416`) make the ids
#: canonical -- and they are identical. So the flip cannot change a user's
#: output, only their wait, which is why it needs no further justification.
#:
#: HOW FAR THE DEPARTURE ACTUALLY GOES, STATED HONESTLY. Their DESIGN is kept
#: whole: their index, their two eps kernels, their batch structure, their
#: `if` at `algo.cuh:226`. What changed is which side of that `if` is taken by
#: default. But this is NOT merely "cuML picked the other default for their
#: hardware", and an earlier version of this note claimed it was.
#:
#: **AT OUR EXACT PARAMETERS cuML NEVER REACHES RBC AT ALL.** `runner.cuh:143-150`
#: is a `constexpr` downgrade, not a runtime one:
#:
#:     if constexpr (std::is_same_v<Type_f, double> || std::is_same_v<Index_, int32_t>) {
#:       if (sparse_rbc_mode) { sparse_rbc_mode = false; ... }
#:     }
#:
#: and `runner.cuh:235` builds the index only `if constexpr (float && int64_t)`.
#: **This implementation is int32-label.** So a caller who asks cuML for `algorithm='rbc'`
#: on an int32-label build gets BRUTE_FORCE with a warning, every time. Their
#: dispatch sends our parameters to brute force and to nothing else; the RBC
#: arm we implemented is one their dispatch would not hand us.
#:
#: That does not make the default wrong -- `check_dbscan_rbc_matches_brute`
#: compares the two labellings POINT FOR POINT and the measurement above is
#: 27x -- but it does mean the ONLY support for it is that measurement plus
#: that equality check. There is no "we are following their dispatch" here.
#: Reverting is this one constant.
#:
#: The one restriction that survives as a genuine cost-free match is the
#: METRIC: `runner.cuh:152-156` downgrades anything but
#: L2Sqrt{Expanded,Unexpanded}, and L2 is all this implementation does.
#: FAST on Apple: the ball-cover arm keeps each batch's neighbour counts
#: from loop 1 and loop 2 scans them instead of re-running the count pass,
#: so a fit walks the dataset twice instead of three times. The counts are
#: the same kernel's output on the same rows, so the CSR is the same.
#: fam-cluster (2026-10-04): IDENTICAL on the NVIDIA and AMD columns keeps
#: the counts too (it was Apple only), so a fit the edge cap splits into
#: several batches does not re-run the count pass in loop 2. Same counts, so
#: the same CSR. `-D MOJOLEARN_IDN_DBSCAN_KEEP_COUNTS_OFF=1` re-counts there.
comptime IDN_DBSCAN_KEEP_COUNTS = (
    GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL
    and not has_apple_gpu_accelerator()
    and not (
        is_defined["MOJOLEARN_IDN_DBSCAN_KEEP_COUNTS_OFF"]()
        or is_defined["MOJOLEARN_IDN_ALL_OFF"]()
    )
)
comptime DBSCAN_RBC_KEEP_COUNTS = (
    (
        (
            (GLOBAL_NUMERIC_MODE == NUMERIC_FAST or GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL)
            and has_apple_gpu_accelerator()
        )
        or IDN_DBSCAN_KEEP_COUNTS
    )
    and not is_defined["MOJOLEARN_DBSCAN_RBC_KEEP_COUNTS_OFF"]()
)

#: fam-cluster (2026-10-04), IDENTICAL: on the ball-cover arm loop 1 no
#: longer reads back two scalars nothing uses. `vd[n_points]` is overwritten
#: by the exact 64-bit count (`adjlen_here = nnz1`), and the per-batch
#: maximum degree feeds only `rbc_dbscan_take_one_pass`, which returns False
#: unconditionally. Three drains and one launch per batch; no value any
#: kernel reads changes. `-D MOJOLEARN_IDN_DBSCAN_RBC_DEAD_READS_OFF=1`
#: restores them.
comptime IDN_DBSCAN_RBC_DEAD_READS = (
    GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL
    and not (
        is_defined["MOJOLEARN_IDN_DBSCAN_RBC_DEAD_READS_OFF"]()
        or is_defined["MOJOLEARN_IDN_ALL_OFF"]()
    )
)

#: fam2-cluster (2026-10-04), IDENTICAL, every column: the border pass
#: decides on the device which batches hold a labelled non-core row. One
#: small launch per batch writes one Int32 cell and `n_batches` cells come
#: back, in place of downloading the core mask and the labels (5 bytes per
#: row) and walking them on the host. Same predicate on the same values, so
#: the same batches take the pass.
#: cpu3-neighbors (2026-10-04): the device decision is now the only route in
#: every mode; the host walk and this define's off arm are gone from the fit.
comptime IDN_DBSCAN_BORDER_NEEDS_DEVICE = (
    GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL
    and not (
        is_defined["MOJOLEARN_IDN_DBSCAN_BORDER_NEEDS_DEVICE_OFF"]()
        or is_defined["MOJOLEARN_IDN_ALL_OFF"]()
    )
)

comptime EPS_NN_BRUTE_FORCE = 0
comptime EPS_NN_RBC = 1


comptime TPB = 256


# TOMBSTONE: MOJOLEARN_IDN_DBSCAN_BATCH_SAMPLE (DROPPED-noise) deleted 2026-10-10 by lane/postmerge-act-7; code recoverable at ca8ea1f8d.
# Tried: D4, comptime IDN_DBSCAN_BATCH_SAMPLE, DS_SAMPLE and ds_gather_rows_kernel (the sampled-degree range plan).
# Restore: git apply experiments/removed/MOJOLEARN_IDN_DBSCAN_BATCH_SAMPLE.patch; record in docs/TOMBSTONES.md.


#: fg-tsne-dbscan D2 (with D1, see `neighbors/impl/ball_cover/registers.mojo`,
#: IDN_DBSCAN_ADJ_BITMAP): loop 1 counts each range with the bit-writing
#: count and, when the range fits the cap, fills its columns from the bits at
#: once into a KEPT copy (columns + offsets); loop 2, batch 0's fill and the
#: border pass copy a kept range's CSR instead of running another distance
#: pass. Memory: the bit matrix is `rbc_eps_bitmap_words(n, landmarks) * 8`
#: bytes a query row (n / 8 + 8 x landmarks), so a range holds at most
#: DB_BM_SHARE of the device's free memory over that (the cap below); the
#: kept columns total at most `edge_cap` edges (the one-batch plan's column
#: budget, 4 bytes an edge) and a range past it is filled as before. The
#: columns are the D1 predicate's, canonicalized as the fill's: the same CSR.
#: Bits none. Weighted fits keep today's loop.
comptime DB_BM_SHARE_PCT = 25


def _kept_find(kept_start: List[Int], start: Int) -> Int:
    """The kept-range slot holding row range `start`, or -1."""
    for k in range(len(kept_start)):  # small-loop(kept_start: one entry per kept row range, at most the batch count): finds a plan entry, no row data
        if kept_start[k] == start:
            return k
    return -1


def rbc_take_one_pass(
    batch_size: Int,
    n_rows: Int,
    ja_capacity: Int,
    n_points: Int,
    max_k: Int,
) -> Bool:
    """`vertexdeg/algo.cuh:119-122`: which arm the second batch loop takes.

        int64_t spare_elemets_per_row =
          data.max_k > 0 ? (batch_size * data.N - data.ja->capacity()) / n : 0;
        if (data.max_k > 0 && data.max_k < spare_elemets_per_row) { ... }

    Their guard, verbatim (their misspelling too). It is a MEMORY test on
    the `n x max_k` scratch the one-pass kernel needs (`registers.cuh:1431`),
    not a correctness test: `batch_size * N` is the dense worst case the
    runner budgeted for, and `ja_capacity` is what the CSR columns already
    claim -- `maxadjlen` by the time loop 2 runs (`runner.cuh:317`).

    A named host function rather than two inline lines so the checks can
    assert which arm a fixture routes to with the SAME arithmetic the runner
    uses (`dbscan_check.mojo::check_dbscan_rbc_two_loop_arms`), per
    CONTRIBUTING.md (Non-default paths): a parameter that selects a kernel is a parameter the
    checks enumerate.
    """
    if max_k <= 0:
        return False
    var spare = (batch_size * n_rows - ja_capacity) // n_points
    return max_k < spare


def rbc_dbscan_take_one_pass(
    batch_size: Int,
    n_rows: Int,
    ja_capacity: Int,
    n_points: Int,
    max_k: Int,
) -> Bool:
    """Whether DBSCAN may currently use the reference max-k shortcut.

    The reference memory predicate remains in `rbc_take_one_pass`, and the
    max-k kernel remains independently checked.  It is not safe to compose
    the two in DBSCAN on Metal yet: the max-k and count kernels disagreed at
    an epsilon boundary in the 400k x 32 reproducer.  Returning false makes
    DBSCAN use count+fill, whose count half is the identical kernel loop 1
    used to decide core points.
    """
    _ = batch_size
    _ = n_rows
    _ = ja_capacity
    _ = n_points
    _ = max_k
    return False


def relabel_for_skl_kernel(
    labels: MutPointer[Int32, MutAnyOrigin],
    n_in: Int32,
):
    """`relabelForSkl` (`runner.cuh:59`), copied.

        1. Turn any labels matching MAX_LABEL into -1
        2. Subtract 1 from all other labels.
    """
    var tid = Int(thread_idx.x) + Int(block_dim.x) * Int(block_idx.x)
    if tid < Int(n_in):
        if labels.unsafe_load(tid) == MAX_LABEL:
            labels.unsafe_store(tid, Int32(-1))
        else:
            labels.unsafe_store(tid, labels.unsafe_load(tid) - Int32(1))


def border_needs_kernel(
    needs: MutPointer[Int32, MutAnyOrigin],
    core: MutPointer[UInt8, MutAnyOrigin],
    labels: MutPointer[Int32, MutAnyOrigin],
    cell_in: Int32,
    start_in: Int32,
    n_points_in: Int32,
):
    """`needs[cell] = 1` when rows `start .. start + n_points` hold a
    non-core row with a label (`IDN_DBSCAN_BORDER_NEEDS_DEVICE`). Every
    writer stores the same 1, so the cell does not depend on thread order.
    """
    var tid = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if tid < Int(n_points_in):
        var r = tid + Int(start_in)
        if core.unsafe_load(r) == 0 and labels.unsafe_load(r) != MAX_LABEL:
            needs.unsafe_store(Int(cell_in), Int32(1))


def border_pull_kernel(
    labels: MutPointer[Int32, MutAnyOrigin],
    row_ind: MutPointer[Int32, MutAnyOrigin],
    col_ind: MutPointer[Int32, MutAnyOrigin],
    core: MutPointer[UInt8, MutAnyOrigin],
    start_vertex_id_in: Int32,
    batch_size_in: Int32,
    n_in: Int32,
):
    """DEVIATION 5130, the border pass: every NON-core row of the batch takes
    the smallest label among its CORE neighbours (`MAX_LABEL`, noise, when it
    has none). Reads core labels only and writes non-core labels only, so no
    thread reads what another writes, and the result is independent of the
    launch order."""
    var tid = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var global_id = tid + Int(start_vertex_id_in)
    if tid >= Int(batch_size_in) or global_id >= Int(n_in):
        return
    if core.unsafe_load(global_id) != 0:
        return
    var best = MAX_LABEL
    var start = Int(row_ind.unsafe_load(tid))
    var end = Int(row_ind.unsafe_load(tid + 1))
    for j in range(start, end):
        var jj = Int(col_ind.unsafe_load(j))
        if core.unsafe_load(jj) != 0:
            var lj = labels.unsafe_load(jj)
            if lj < best:
                best = lj
    labels.unsafe_store(global_id, best)


def dbscan_fit(
    ctx: DeviceContext,
    mut x: DeviceBuffer[DType.float32],
    mut adj: DeviceBuffer[DType.uint8],
    mut vd: DeviceBuffer[DType.int32],
    mut core: DeviceBuffer[DType.uint8],
    mut ex_scan: DeviceBuffer[DType.int32],
    mut labels: DeviceBuffer[DType.int32],
    mut labels_temp: DeviceBuffer[DType.int32],
    mut work_buffer: DeviceBuffer[DType.int32],
    mut block_sums: DeviceBuffer[DType.int32],
    mut sample_weight: DeviceBuffer[DType.float32],
    mut wght_sum: DeviceBuffer[DType.float32],
    n_rows: Int,
    n_features: Int,
    eps: Float64,
    min_pts: Int,
    batch_size: Int = 0,
    max_iterations: Int = 200,
    eps_nn_method: Int = EPS_NN_RBC,
    phase_timing: Bool = False,
    metric: Int = DBSCAN_METRIC_L2,
    has_weights: Bool = False,
    edge_cap: Int = Int(MAX_LABEL),
    n_batches_out_addr: Int = 0,
) raises -> Int:
    """`Dbscan::run`, single node. Returns the total propagation passes.

    `edge_cap` is the most edges one batch's CSR may hold: `MAX_LABEL`, the
    int32 CSR's bound, and nothing else in production. A ball-cover batch
    whose EXACT edge count exceeds it is split in two and re-counted until
    every batch fits (loop 1 below). A check passes a small cap to force the
    split on a small fixture. `n_batches_out_addr`, when nonzero, is the
    address of one host Int that receives the batch count loop 1 settled on.

    Workspace, sized as `runner.cuh:169-177` sizes theirs:

        adj          bool  [N * batch_size]
        core         bool  [N]
        vd           Index [batch_size + 1]
        ex_scan      Index [batch_size + 1]
        labels       Index [N]         (output)
        labels_temp  Index [N]
        work_buffer  Index [N]
        block_sums   Index [scan_blocks_needed(N) + 1]
        wght_sum     Type_f [batch_size]   ONLY when `has_weights`

`wght_sum` is theirs and is sized exactly as `runner.cuh:176-177` sizes it
(`sample_weight != nullptr ? alignTo(sizeof(Type_f) * batch_size) : 0`).
`sample_weight` is N long and is the CALLER'S array, not workspace, which is
also theirs (it is a `const Type_f*` parameter of `Dbscan::run`, `:120`).
When `has_weights` is False both are one-element placeholders and neither is
read: Mojo has no null `DeviceBuffer`, so the `sample_weight != nullptr`
their code branches on is this Bool.

    `block_sums` is OURS and replaces two things of theirs: thrust's internal
    scan scratch, and `row_counters` (`runner.cuh:218`), which their
    `adj_to_csr` uses as a per-row atomic cursor and our block-prefix-sum
    compaction does not need.

    `adj_graph` (the CSR column indices) is NOT a parameter, which is also
    theirs: `runner.cuh:230` declares it as a local `rmm::device_uvector` of
    length 0 and resizes it to the largest batch's edge count at `:317`,
    after loop 1 has measured them. It is allocated here at the same point
    for the same reason.

    `batch_size = 0` means one batch over the whole dataset.

    `phase_timing` is the implementation of the instrumentation cuML
    hangs on this function: their `verbosity` parameter gates a
    `CUML_LOG_DEBUG("- Batch %d / %ld ...")` per batch per loop, and every
    phase sits in an nvtx range (`Trace::Dbscan::VertexDeg` :255/:330,
    `CorePoints` :299, `AdjGraph` :355, `WeakCC` :373, `MergeLabels` :397,
    `FinalRelabel` :411). Metal has no nvtx consumer, so the ranges print as
    wall-clock lines instead, one per phase per batch:

        PHASE plan n_rows <N> batch <b> n_batches <nb> method <rbc|brute>
        PHASE mask.vertexdeg batch <i>/<n> <ms>      loop 1, includes the
                                                     vd[n] readback, as their
                                                     range does (:255-296)
        PHASE mask.corepoints batch <i>/<n> <ms>
        PHASE label.vertexdeg batch <i>/<n> <ms>     loop 2, batches > 0
                                                     (batch 0 is resident
                                                     from loop 1); the rbc
                                                     arm is one max_k pass,
                                                     or count + fill when
                                                     the bound does not fit
        PHASE label.adjgraph batch <i>/<n> <ms>      brute arm only, as :355
        PHASE label.weak_cc batch <i>/<n> <ms> passes <p>
        PHASE label.merge_labels batch <i>/<n> <ms>  batches > 0, as :389
        PHASE final_relabel batch 1/1 <ms>

    `<i>` is 1-based, as their "- Batch %d" prints `i + 1`. Every phase
    already ends on a `ctx.synchronize()`, so the timestamps add no sync
    that the implementation does not already perform. Off (the default), nothing
    prints and nothing is measured.
    """
    var batch = batch_size if batch_size > 0 else n_rows
    if batch > n_rows:
        batch = n_rows
    var n_batches = (n_rows + batch - 1) // batch

    # `DBSCAN_CC_FLAG_CELLS` is 1 unless `IDN_DBSCAN_CC_GATED` keeps the
    # convergence cells of a whole chunk of passes on the device.
    var d_flag = ctx.enqueue_create_buffer[DType.int32](DBSCAN_CC_FLAG_CELLS)
    var h_flag = ctx.enqueue_create_host_buffer[DType.int32](
        DBSCAN_CC_FLAG_CELLS
    )
    var h_adjlen = ctx.enqueue_create_host_buffer[DType.int32](1)
    ctx.synchronize()

    # --- the RBC arm's fallbacks, `runner.cuh:139-201` --------------------
    # Theirs DOWNGRADES rather than refusing, and logs. Each of these is a
    # real condition in their file and none is ours:
    #   :143-150  double precision OR int32 labels        -> BRUTE_FORCE
    #   :152-156  any metric but L2Sqrt{Expanded,Unexpanded} -> BRUTE_FORCE
    #   :194-200  D > MAX_LABEL / N (the index cannot be addressed)
    #
    # THE FIRST ONE FIRES FOR US AND IS DELIBERATELY NOT COPIED. `Index_` here
    # is Int32, which is the exact case `:143` disables RBC for, so a faithful
    # copy of that guard would make `EPS_NN_RBC` dead code and pin every fit
    # to the n^2 arm. We keep the RBC arm reachable anyway; see DEVIATION 35
    # at the top of this file for the measurement that is its only support,
    # and for why "their dispatch takes this path" is NOT among the reasons.
    # The metric guard cannot fire because L2 is all this implementation does, and the
    # third is copied verbatim below.
    var sparse_rbc_mode = eps_nn_method == EPS_NN_RBC
    if sparse_rbc_mode and n_features > Int(MAX_LABEL) // n_rows:
        sparse_rbc_mode = False

    # THE L1 ARM IS BRUTE FORCE ONLY, AND THIS RAISES RATHER THAN
    # DOWNGRADING. `runner.cuh:152-156` DOWNGRADES an unsupported metric to
    # L2Sqrt and logs a warning; copying that here would answer a Manhattan
    # query with Euclidean neighborhoods, which is the "accepted and
    # ignored" failure this repository refuses by house rule.
    #
    # THE REASON IS SCOPE, NOT IMPOSSIBILITY, and the distinction matters
    # because a refusal that hides a doable feature is how a library stops
    # growing. The ball cover's pruning is metric-generic in principle: it
    # rests on the triangle inequality, which L1 satisfies. What is L2 in
    # this tree is the IMPLEMENTATION -- `neighbors/impl/
    # ball_cover/common.mojo::eps_dist_sq` and the three landmark bounds at
    # `ball_cover/registers.mojo:280`, `:422`, `:547` all compute Euclidean
    # distance, and the index's radii are built from it. An L1 ball cover is
    # a real and reachable piece of work; it belongs to the `neighbors/`
    # lane, not to this one, and until it lands the honest answer to
    # `metric='manhattan', algorithm='rbc'` is a refusal that says which arm
    # does serve it.
    if sparse_rbc_mode and metric != DBSCAN_METRIC_L2:
        raise Error(
            "dbscan: metric='" + dbscan_metric_name(metric) + "' is served by"
            " the BRUTE_FORCE arm only. The ball cover's landmark radii and"
            " its triangle-inequality bounds are computed as Euclidean"
            " distances in neighbors/impl/ball_cover/ (common.mojo"
            " eps_dist_sq; registers.mojo:280, :422, :547), so an L1 query"
            " needs an L1 index that lane has not built yet. Pass"
            " algorithm='brute'."
        )

    # `runner.cuh:181-186`: `ASSERT(N * batch_size <
    # static_cast<std::size_t>(MAX_LABEL), "An overflow occurred with the
    # current choice of precision ...")`. Theirs is unconditional and cannot
    # bind on their RBC path, because RBC requires int64 labels (`:143-150`)
    # and 2^63 / N never caps a real batch. Ours is int32-label with RBC
    # reachable (DEVIATION 35), so the assert is scoped to the arm whose
    # dense `N * batch_size` adjacency is real; on the RBC arm the honest
    # int32 bound is the EDGE COUNT, met by splitting the batch at the query below, and copying
    # the assert unconditionally would re-impose the very clamp
    # `dbscan.cuh:71` gates off for RBC.
    if not sparse_rbc_mode and n_rows * batch >= Int(MAX_LABEL):
        raise Error(
            "An overflow occurred with the current choice of precision and"
            " the number of samples. (Max allowed batch size is "
            + String(Int(MAX_LABEL) // n_rows)
            + ", but was "
            + String(batch)
            + ")."
        )

    if phase_timing:
        var method_name = String("brute")
        if sparse_rbc_mode:
            method_name = String("rbc")
        print(
            "PHASE plan n_rows " + String(n_rows) + " batch " + String(batch)
            + " n_batches " + String(n_batches) + " method " + method_name
            + " metric " + dbscan_metric_name(metric)
            + " weighted " + String(has_weights)
        )

    # `runner.cuh:231-241`: build the index ONCE, before the batch loop, not
    # per batch. `rbc_build_index` is `cuvs::neighbors::ball_cover::build`.
    var n_landmarks = rbc_n_landmarks(n_rows) if sparse_rbc_mode else 1
    var rbc_r = ctx.enqueue_create_buffer[DType.float32](
        n_landmarks * n_features
    )
    var rbc_xr = ctx.enqueue_create_buffer[DType.float32](
        (n_rows * n_features) if sparse_rbc_mode else 1
    )
    var rbc_lm = ctx.enqueue_create_buffer[DType.int32](n_landmarks)
    var rbc_sc = ctx.enqueue_create_buffer[DType.int32](n_rows)
    var rbc_sd = ctx.enqueue_create_buffer[DType.float32](n_rows)
    var rbc_ne = ctx.enqueue_create_buffer[DType.int32](n_rows)
    var rbc_nd = ctx.enqueue_create_buffer[DType.float32](n_rows)
    var rbc_ip = ctx.enqueue_create_buffer[DType.int32](n_landmarks + 1)
    var rbc_c1 = ctx.enqueue_create_buffer[DType.int32](n_rows)
    var rbc_d1 = ctx.enqueue_create_buffer[DType.float32](n_rows)
    var rbc_rad = ctx.enqueue_create_buffer[DType.float32](n_landmarks)
    var rbc_cnt = ctx.enqueue_create_buffer[DType.int32](n_landmarks)
    # One int of device scratch: loop 1's max-reduce result, then the max_k
    # query's `actual_max` readback (`registers.cuh:1453`).
    var rbc_mk_scratch = ctx.enqueue_create_buffer[DType.int32](1)
    var keep_counts = False
    comptime if DBSCAN_RBC_KEEP_COUNTS:
        keep_counts = sparse_rbc_mode
    var vd_all = ctx.enqueue_create_buffer[DType.int32](
        n_rows if keep_counts else 1
    )
    ctx.synchronize()

    if sparse_rbc_mode:
        rbc_build_index(
            ctx, x, rbc_r, rbc_xr, rbc_lm, rbc_sc, rbc_sd, rbc_ne, rbc_nd,
            rbc_ip, rbc_c1, rbc_d1, rbc_rad, rbc_cnt,
            n_rows, n_features, n_landmarks,
        )
        ctx.synchronize()

    # TOMBSTONE: MOJOLEARN_DBSCAN_FAST_DENSEBALL (DROPPED-slower) deleted 2026-10-09 by lane/owed-deletions-D3; code recoverable at b639a2bd2.
    # Tried: the dense-ball route ahead of the edge-list loops here.
    # Restore: git apply experiments/removed/MOJOLEARN_DBSCAN_FAST_DENSEBALL.patch; record in docs/TOMBSTONES.md.

    # THE RADIUS, NOT ITS SQUARE. `algo.cuh:227` hands `data.eps` to `eps_nn`
    # while the brute-force arm one line later gets `eps2`. The query kernel
    # squares it once, internally. Passing the squared value here silently
    # widens every neighborhood to eps^2.
    var eps_radius = Float32(eps)

    # --- loop 1: the mask. REVERSED, so batch 0 stays resident -----------
    #
    # ADAPTIVE SPLITTING (lane dbscan-int64, 2026-09-29). The batch size
    # above is a MEMORY estimate; it does not know how many edges a batch
    # will have. On an L40S, taxi 4.1M x 16 at eps 3 / min_samples 2 (the
    # cuML benchmark's values), one batch of the memory-sized ball-cover arm
    # had about 2.5e9 edges, past the int32 CSR, and the fit used to REFUSE
    # (with the wrapped count in its message; a count past 2^32 could wrap
    # back positive and pass silently). The count pass now returns the exact
    # 64-bit count, and a batch over `edge_cap` is split in two halves that
    # are counted again, until every batch fits. cuML meets the same bound by
    # widening the index to int64 (`runner.cuh:143-150`); this implementation
    # keeps the int32 CSR and narrows the batch instead.
    #
    # WHY THIS CANNOT MOVE A LABEL. Nothing downstream reads the batch
    # boundaries as data: the core mask is per row, each batch's CSR is its
    # rows' full neighbour lists, `merge_labels` folds the per-batch
    # components, and the border pass (DEVIATION 5130) recomputes every
    # border label from the final core labels. Uneven batches are the same
    # argument as even ones; `dbscan/checks/dbscan_edge_split_check.mojo`
    # gates it bitwise against the one-batch fit.
    #
    # MEMORY. Only ONE batch's columns are ever resident (`col_ind` below is
    # the LARGEST batch's edge count, `runner.cuh:317`), never the whole
    # graph, so splitting is also what keeps the columns within
    # `edge_cap * 4` bytes (8.6 GB at the int32 bound).
    #
    # THE ORDER IS PRESERVED. A stack of pending row ranges, highest first:
    # a split pushes its lower half and then its upper half, so the ranges
    # are still visited in DESCENDING row order and the last range counted
    # starts at row 0 -- the batch 0 whose `ex_scan` the fill after this
    # loop relies on being resident.
    var plan_start = List[Int]()
    var plan_rows = List[Int]()
    var batchadjlen = List[Int]()
    var maxklen = List[Int]()
    var pend_start = List[Int]()
    var pend_rows = List[Int]()
    # TOMBSTONE: MOJOLEARN_IDN_DBSCAN_BATCH_SAMPLE (DROPPED-noise) deleted 2026-10-10 by lane/postmerge-act-7; code recoverable at ca8ea1f8d.
    # Tried: D4, the first ranges sized from a 1,024-row sampled degree (edge_cap / (2 x mean degree)).
    # Restore: git apply experiments/removed/MOJOLEARN_IDN_DBSCAN_BATCH_SAMPLE.patch; record in docs/TOMBSTONES.md.
    # fg-tsne-dbscan D2: the bit matrix (IDN_DBSCAN_ADJ_BITMAP) caps a range
    # at DB_BM_SHARE_PCT of the free memory over its bytes per query row
    var use_bm = False
    var bm_words = 1
    comptime if IDN_DBSCAN_ADJ_BITMAP:
        if sparse_rbc_mode and not has_weights:
            use_bm = True
            bm_words = rbc_eps_bitmap_words(n_rows, n_landmarks)
            var bm_free = Int(ctx.get_memory_info()[0])
            var bm_rows = (bm_free * DB_BM_SHARE_PCT // 100) // (bm_words * 8)
            if bm_rows < 1:
                bm_rows = 1
            if bm_rows < batch:
                batch = bm_rows
                n_batches = (n_rows + batch - 1) // batch
            if phase_timing:
                print(
                    "PHASE plan.bitmap words " + String(bm_words) + " batch "
                    + String(batch) + " n_batches " + String(n_batches)
                )
    var bm_buf = ctx.enqueue_create_buffer[DType.uint64]((batch * bm_words) if use_bm else 1)
    var kept_start = List[Int]()
    var kept_ia = List[DeviceBuffer[DType.int32]]()
    var kept_ja = List[DeviceBuffer[DType.int32]]()
    var kept_nnz = List[Int]()
    var kept_total = 0
    for b0 in range(n_batches):  # small-loop(n_batches: one pending range per row batch): builds the batch plan, a launch-argument list
        pend_start.append(b0 * batch)
        pend_rows.append(min(n_rows - b0 * batch, batch))
    var n_splits = 0

    while len(pend_start) > 0:
        var start_vertex_id = pend_start.pop()
        var n_points = pend_rows.pop()
        # The uniform batch this range came from; phase labels only.
        var i = start_vertex_id // batch
        var nnz1 = 0
        var maxk_here = 0
        var t_vd1 = perf_counter_ns()

        if sparse_rbc_mode:
            # `algo.cuh:137-144`, the `max_k == 0` arm: fill `ia` and `vd`,
            # emit no columns. `runner.cuh:262` passes the literal 0 here,
            # so this is their first-loop call verbatim.
            # THE QUERY IS THE BATCH, NOT THE DATASET. `algo.cuh:132`,
            # `:143` and `:161` all build the query view as
            # `data.x + start_vertex_id * k, n, k` -- an offset of
            # `start_vertex_id` ROWS. Handing the whole `x` here makes every
            # batch re-query rows 0..n_points and the batched fit disagrees
            # with the unbatched one, which is exactly what
            # `check_dbscan_batching_agrees` caught: 412 of 612 labels.
            var qb1 = x.create_sub_buffer[DType.float32](
                start_vertex_id * n_features, n_points * n_features
            )
            if use_bm:
                # D2: the count also writes the range's bit matrix
                nnz1 = rbc_eps_pass_count_bitmap(
                    ctx, rbc_xr, qb1, rbc_r, rbc_ip, rbc_c1, rbc_d1, rbc_rad,
                    ex_scan, vd, bm_buf, n_points, n_features, n_landmarks,
                    eps_radius,
                )
            else:
                nnz1 = rbc_eps_nn_query_count(
                    ctx, rbc_xr, qb1, rbc_r, rbc_ip, rbc_c1, rbc_d1, rbc_rad,
                    ex_scan, vd, n_points, n_features, n_landmarks, eps_radius,
                )
            # WHY cuML REQUIRES int64 ON THIS PATH, AND WHAT OURS DOES INSTEAD.
            #
            # `runner.cuh:143-150` refuses RBC for `Index_ == int32_t`, and
            # `:235` builds the index only under `float && int64_t`: the CSR
            # this query emits is indexed by the EDGE COUNT, and a dense
            # neighbourhood at large n runs past 2^31 long before it runs out
            # of memory. Ours keeps the int32 CSR and SPLITS the batch (see
            # the block above loop 1). `nnz1` is exact (summed in 64-bit by
            # the count pass), so this test cannot be fooled by a wrap; the
            # `ex_scan` and `vd` of a rejected range are overwritten by the
            # next count and nothing else has read them.
            if nnz1 < 0:
                raise Error(
                    "dbscan: the ball-cover count returned " + String(nnz1)
                    + " edges for rows " + String(start_vertex_id) + ".."
                    + String(start_vertex_id + n_points)
                    + "; an exact count is never negative"
                )
            if nnz1 > edge_cap:
                if n_points < 2:
                    raise Error(
                        "dbscan: row " + String(start_vertex_id) + " alone has "
                        + String(nnz1) + " neighbours, more than the "
                        + String(edge_cap) + " edges one int32 CSR batch"
                        " holds (" + String(nnz1 * 4) + " bytes of column"
                        " ids), so no split of the query rows can fit it"
                    )
                var lo = n_points // 2
                pend_start.append(start_vertex_id)
                pend_rows.append(lo)
                pend_start.append(start_vertex_id + lo)
                pend_rows.append(n_points - lo)
                n_splits += 1
                if phase_timing:
                    print(
                        "PHASE mask.split rows " + String(start_vertex_id)
                        + "+" + String(n_points) + " edges " + String(nnz1)
                        + " cap " + String(edge_cap) + " "
                        + String(Float64(perf_counter_ns() - t_vd1) / 1.0e6)
                    )
                continue
            # D2: the range's columns from its bits, kept for loop 2
            if use_bm and kept_total + nnz1 <= edge_cap:
                var kja = ctx.enqueue_create_buffer[DType.int32](nnz1 if nnz1 > 0 else 1)
                var kia = ctx.enqueue_create_buffer[DType.int32](n_points + 1)
                if nnz1 > 0:
                    rbc_eps_pass_fill_bitmap(
                        ctx, bm_buf, rbc_ip, rbc_c1, ex_scan, kja, n_points,
                        n_landmarks, nnz1,
                    )
                ctx.enqueue_copy(
                    dst_buf=kia,
                    src_buf=ex_scan.create_sub_buffer[DType.int32](0, n_points + 1),
                )
                ctx.synchronize()
                kept_start.append(start_vertex_id)
                kept_ia.append(kia^)
                kept_ja.append(kja^)
                kept_nnz.append(nnz1)
                kept_total += nnz1
            # DEVIATION 29: `need_ja_compute = sparse_rbc_mode && ((i == 0)
            # || (sample_weight != nullptr))`, `runner.cuh:257`. The count
            # pass above emitted `ia` and `vd` and NO columns, and a
            # weighted degree needs the neighbour ids, so with weights on
            # every batch of loop 1 fills too. The scratch is this batch's
            # own edge count; see the deviation block for why it is not
            # `col_ind` (which is not sized until loop 1 has finished).
            if has_weights:
                var ja_len1 = nnz1 if nnz1 > 0 else 1
                var ja1 = ctx.enqueue_create_buffer[DType.int32](ja_len1)
                ctx.synchronize()
                var qbw = x.create_sub_buffer[DType.float32](
                    start_vertex_id * n_features, n_points * n_features
                )
                rbc_eps_nn_query_fill(
                    ctx, rbc_xr, qbw, rbc_r, rbc_ip, rbc_c1, rbc_d1, rbc_rad,
                    ex_scan, ja1, n_points, n_features, n_landmarks,
                    eps_radius,
                )
                ctx.synchronize()
                weighted_vertex_deg_csr(
                    ctx, wght_sum, ex_scan, ja1, sample_weight, n_points
                )
                ctx.synchronize()
                _ = ja1^
        else:
            vertex_deg_dispatch(
                ctx, adj, vd, x, start_vertex_id, n_points, n_rows,
                n_features, eps, metric,
            )
            ctx.synchronize()
            # `algo.cuh:243-254`, the dense half of their sample-weight arm.
            # It reads the `adj` the neighborhood kernel just wrote, so it
            # sits after the synchronize and before the degree readback.
            if has_weights:
                weighted_vertex_deg_dense(
                    ctx, wght_sum, adj, sample_weight, n_points, n_rows
                )
        ctx.synchronize()

        # `raft::update_host(&curradjlen, vd + n_points, 1, stream)`
        # (`runner.cuh:281`): the neighborhood kernel put the batch's total
        # edge count in the last element of `vd`.
        if keep_counts:
            ctx.enqueue_copy(
                dst_buf=vd_all.create_sub_buffer[DType.int32](
                    start_vertex_id, n_points
                ),
                src_buf=vd.create_sub_buffer[DType.int32](0, n_points),
            )
        var skip_dead_reads = False
        comptime if IDN_DBSCAN_RBC_DEAD_READS:
            skip_dead_reads = sparse_rbc_mode
        var adjlen_here = nnz1
        if not skip_dead_reads:
            var vd_last = vd.create_sub_buffer[DType.int32](n_points, 1)
            ctx.enqueue_copy(dst_ptr=h_adjlen.unsafe_ptr(), src_buf=vd_last)
            ctx.synchronize()
            # The ball-cover arm keeps the EXACT count: `vd[n_points]` is the
            # int32 scan's tail, equal to it only because the split above has
            # already brought it under `edge_cap`.
            adjlen_here = Int(h_adjlen.unsafe_ptr().unsafe_load(0))
            if sparse_rbc_mode:
                adjlen_here = nnz1

        # `runner.cuh:287-293`: `maxklen.at(i) = thrust::reduce(vd, vd +
        # n_points, 0, maximum{})` -- the longest row in this batch, measured
        # while the degrees are resident so loop 2 can take the one-pass
        # form. The reduce runs on the DEVICE as thrust's does; only the
        # scalar comes back. It sits inside the mask.vertexdeg window below
        # exactly as it sits inside their nvtx VertexDeg range (:255-296).
        if sparse_rbc_mode and not skip_dead_reads:
            rbc_max_reduce_launch(
                ctx, rbc_mk_scratch, vd, n_points
            )
            ctx.synchronize()
            ctx.enqueue_copy(
                dst_ptr=h_adjlen.unsafe_ptr(), src_buf=rbc_mk_scratch
            )
            ctx.synchronize()
            maxk_here = Int(h_adjlen.unsafe_ptr().unsafe_load(0))
        if phase_timing:
            print(
                "PHASE mask.vertexdeg batch " + String(i + 1) + "/"
                + String(n_batches) + " "
                + String(Float64(perf_counter_ns() - t_vd1) / 1.0e6)
            )

        # `runner.cuh:300-306`: the ONE place `sample_weight` changes the
        # answer. Their ternary is two instantiations of one template; ours
        # is two functions in the file that template lives in.
        var t_cp = perf_counter_ns()
        if has_weights:
            core_points_compute_weighted(
                ctx, wght_sum, core, min_pts, start_vertex_id, n_points
            )
        else:
            core_points_compute(
                ctx, vd, core, min_pts, start_vertex_id, n_points
            )
        ctx.synchronize()
        if phase_timing:
            print(
                "PHASE mask.corepoints batch " + String(i + 1) + "/"
                + String(n_batches) + " "
                + String(Float64(perf_counter_ns() - t_cp) / 1.0e6)
            )
        plan_start.append(start_vertex_id)
        plan_rows.append(n_points)
        batchadjlen.append(adjlen_here)
        maxklen.append(maxk_here)

    # Loop 1 accepted the ranges in DESCENDING row order; every later loop
    # indexes them ascending, batch 0 first, as `runner.cuh` does.
    plan_start.reverse()
    plan_rows.reverse()
    batchadjlen.reverse()
    maxklen.reverse()
    n_batches = len(plan_start)
    if plan_start[0] != 0:
        raise Error("dbscan: loop 1's plan does not start at row 0")
    if phase_timing and n_splits > 0:
        print(
            "PHASE plan.split splits " + String(n_splits) + " n_batches "
            + String(n_batches) + " edge_cap " + String(edge_cap)
        )
    if n_batches_out_addr != 0:
        MutPointer[Int, MutUntrackedOrigin](
            unsafe_from_address=n_batches_out_addr
        ).unsafe_store(0, n_batches)

    # `Index_ maxadjlen = *std::max_element(...); adj_graph.resize(maxadjlen)`
    var maxadjlen = 1
    for b in range(n_batches):  # small-loop(n_batches: one adjacency count per row batch): max of the plan's per-batch counts, shapes one buffer
        if batchadjlen[b] > maxadjlen:
            maxadjlen = batchadjlen[b]
    var col_ind = ctx.enqueue_create_buffer[DType.int32](maxadjlen)
    ctx.synchronize()

    # `need_ja_compute = sparse_rbc_mode && ((i == 0) || sample_weight)`,
    # `runner.cuh:257`: batch 0 is the one batch whose COLUMNS loop 1 also
    # produces, so loop 2 can skip its neighborhood pass (`:327`).
    #
    # DEVIATION 39: theirs fills during loop 1 into `adj_graph`
    # sized to batch 0's own edge count (`algo.cuh:150`) and then GROWS it
    # to `maxadjlen` at `runner.cuh:317` -- `rmm::device_uvector::resize`
    # preserves contents when growing. `DeviceBuffer` has no growing resize,
    # so ours sizes `col_ind` first and runs batch 0's fill immediately
    # after, against the `ex_scan` and `vd` that loop 1's last, reversed
    # iteration left resident. Same single fill of batch 0, and the device
    # state at loop 2's entry is identical byte for byte.
    var rbc_tmp_len = 1
    if sparse_rbc_mode:
        var np0 = plan_rows[0]
        var k0 = _kept_find(kept_start, 0)
        if k0 >= 0:
            # D2: batch 0's kept columns (its offsets are resident)
            if kept_nnz[k0] > 0:
                ctx.enqueue_copy(
                    dst_buf=col_ind.create_sub_buffer[DType.int32](0, kept_nnz[k0]),
                    src_buf=kept_ja[k0].create_sub_buffer[DType.int32](0, kept_nnz[k0]),
                )
        else:
            var qb0 = x.create_sub_buffer[DType.float32](0, np0 * n_features)
            rbc_eps_nn_query_fill(
                ctx, rbc_xr, qb0, rbc_r, rbc_ip, rbc_c1, rbc_d1, rbc_rad,
                ex_scan, col_ind, np0, n_features, n_landmarks, eps_radius,
            )
        ctx.synchronize()

        # `registers.cuh:1431` allocates the `n x max_k` scratch inside
        # each max_k call; `maxklen` is fully known here, so ours is one
        # buffer at the largest size any one-pass batch will ask for.
        for b1 in range(1, n_batches):  # small-loop(n_batches: one plan entry per row batch): sizes the max-k scratch from the plan, no row data
            var np_b = plan_rows[b1]
            if np_b <= 0:
                break
            # The max-k kernel duplicates the count kernel's distance loop.
            # On Metal the two separately compiled loops can disagree by one
            # at the epsilon boundary (observed at 400k x 32: loop 1 found a
            # maximum degree of 1, the max-k loop found 0).  A larger scratch
            # allocation cannot repair two different neighbourhoods.  Keep
            # the reference one-pass implementation available and tested, but
            # do not dispatch it from DBSCAN until both paths are bitwise the
            # same predicate.  The two-pass arm calls the SAME count kernel
            # used by loop 1, then fills from its CSR offsets.
            if rbc_dbscan_take_one_pass(
                batch, n_rows, maxadjlen, np_b, maxklen[b1]
            ):
                if np_b * maxklen[b1] > rbc_tmp_len:
                    rbc_tmp_len = np_b * maxklen[b1]
    var rbc_tmp = ctx.enqueue_create_buffer[DType.int32](rbc_tmp_len)
    ctx.synchronize()

    # --- loop 2: the labelling -------------------------------------------
    var passes = 0
    for b2 in range(n_batches):
        var start2 = plan_start[b2]
        var n_points2 = plan_rows[b2]
        if n_points2 <= 0:
            break

        # i == 0 -> adj and vd for batch 0 already in memory
        var t_vd2 = perf_counter_ns()
        if sparse_rbc_mode:
            # The query EMITS CSR, so `ex_scan` is `adj_ia` and `col_ind` is
            # `adj_ja`. `algo.cuh` has no `adj_to_csr` in this branch at all,
            # which is why `AdjGraph::run` is skipped below: their own
            # `runner.cuh:355` guards it with `if (!sparse_rbc_mode)`.
            # Running both would scan the degrees twice.
            #
            # `if (i > 0)`, `runner.cuh:327` -- "i==0 -> adj and vd for
            # batch 0 already in memory". Batch 0's `ia` is in `ex_scan`
            # from loop 1's last, reversed iteration and its `ja` was
            # filled into `col_ind` the moment `col_ind` was sized, so
            # batch 0 runs NO neighborhood pass here. Every other batch is
            # ONE pass when loop 1's bound fits the spare room, and the
            # two-pass form otherwise. Two walks over the dataset per fit,
            # not three.
            if b2 > 0:
                var qb2 = x.create_sub_buffer[DType.float32](
                    start2 * n_features, n_points2 * n_features
                )
                if rbc_dbscan_take_one_pass(
                    batch, n_rows, maxadjlen, n_points2, maxklen[b2]
                ):
                    # `runner.cuh:335` passes `maxklen.at(i)`. Their `vd`
                    # argument is `nullptr` here (`:337`) and cuVS reads
                    # the degrees off `adj_ia` instead
                    # (`registers.cuh:1429`); ours hands the same `vd`
                    # buffer, and nothing after this point reads it -- the
                    # core mask came from loop 1.
                    var actual = rbc_eps_nn_query_max_k(
                        ctx, rbc_xr, qb2, rbc_r, rbc_ip, rbc_c1, rbc_d1,
                        rbc_rad, ex_scan, col_ind, vd, rbc_tmp,
                        rbc_mk_scratch, n_points2, n_features,
                        n_landmarks, eps_radius, maxklen[b2],
                    )
                    # `ASSERT(max_k == data.max_k, "given maximum rowsize
                    # was not sufficient")`, `algo.cuh:135`. An EQUALITY:
                    # the bound was measured on these exact rows in loop
                    # 1, so it cannot be exceeded, and a mismatch means
                    # the CSR in `col_ind` is truncated garbage.
                    if actual != maxklen[b2]:
                        raise Error(
                            "dbscan rbc: batch " + String(b2)
                            + " was bounded at " + String(maxklen[b2])
                            + " columns by loop 1 and came back with "
                            + String(actual)
                            + "; given maximum rowsize was not sufficient"
                        )
                elif _kept_find(kept_start, start2) >= 0:
                    # D2: the range's kept CSR, no distance pass
                    var k2 = _kept_find(kept_start, start2)
                    ctx.enqueue_copy(
                        dst_buf=ex_scan.create_sub_buffer[DType.int32](0, n_points2 + 1),
                        src_buf=kept_ia[k2],
                    )
                    if kept_nnz[k2] > 0:
                        ctx.enqueue_copy(
                            dst_buf=col_ind.create_sub_buffer[DType.int32](0, kept_nnz[k2]),
                            src_buf=kept_ja[k2].create_sub_buffer[DType.int32](0, kept_nnz[k2]),
                        )
                    ctx.synchronize()
                else:
                    # `algo.cuh:137-163`, the two-pass arm loop 2 falls
                    # back to when the bound does not fit the spare room.
                    if keep_counts:
                        ctx.enqueue_copy(
                            dst_buf=vd.create_sub_buffer[DType.int32](
                                0, n_points2
                            ),
                            src_buf=vd_all.create_sub_buffer[DType.int32](
                                start2, n_points2
                            ),
                        )
                        rbc_exclusive_scan_launch(ctx, ex_scan, vd, n_points2)
                        ctx.enqueue_copy(
                            dst_buf=vd.create_sub_buffer[DType.int32](
                                n_points2, 1
                            ),
                            src_buf=ex_scan.create_sub_buffer[DType.int32](
                                n_points2, 1
                            ),
                        )
                    else:
                        var _nnz2 = rbc_eps_nn_query_count(
                            ctx, rbc_xr, qb2, rbc_r, rbc_ip, rbc_c1, rbc_d1,
                            rbc_rad, ex_scan, vd, n_points2, n_features,
                            n_landmarks, eps_radius,
                        )
                    ctx.synchronize()
                    var qb3 = x.create_sub_buffer[DType.float32](
                        start2 * n_features, n_points2 * n_features
                    )
                    rbc_eps_nn_query_fill(
                        ctx, rbc_xr, qb3, rbc_r, rbc_ip, rbc_c1, rbc_d1,
                        rbc_rad, ex_scan, col_ind, n_points2, n_features,
                        n_landmarks, eps_radius,
                    )
                    ctx.synchronize()
                if phase_timing:
                    print(
                        "PHASE label.vertexdeg batch " + String(b2 + 1)
                        + "/" + String(n_batches) + " "
                        + String(Float64(perf_counter_ns() - t_vd2) / 1.0e6)
                    )
        else:
            if b2 > 0:
                # No weighted pass here: loop 2 rebuilds `adj` only to build
                # the CSR the labeller walks, and the core mask it consults
                # was finished by loop 1 over every batch. Recomputing
                # `wght_sum` would write the same numbers and be read by
                # nothing.
                vertex_deg_dispatch(
                    ctx, adj, vd, x, start2, n_points2, n_rows, n_features,
                    eps, metric,
                )
                ctx.synchronize()
                if phase_timing:
                    print(
                        "PHASE label.vertexdeg batch " + String(b2 + 1) + "/"
                        + String(n_batches) + " "
                        + String(Float64(perf_counter_ns() - t_vd2) / 1.0e6)
                    )

            var t_ag = perf_counter_ns()
            adj_graph_run(
                ctx, adj, vd, ex_scan, col_ind, block_sums, n_points2, n_rows
            )
            ctx.synchronize()
            if phase_timing:
                print(
                    "PHASE label.adjgraph batch " + String(b2 + 1) + "/"
                    + String(n_batches) + " "
                    + String(Float64(perf_counter_ns() - t_ag) / 1.0e6)
                )

        # Their ternary `i == 0 ? labels : labels_temp` is written out: a
        # pointer-valued conditional picks the wrong branch in this Mojo
        # and, besides, buffers are not pointers here anyway. The
        # merge is a separate `if (i > 0)` in theirs too (`runner.cuh:389`).
        var t_cc = perf_counter_ns()
        var batch_passes: Int
        if b2 == 0:
            batch_passes = weak_cc_batched(
                ctx, labels, ex_scan, col_ind, core, d_flag, h_flag,
                n_rows, start2, n_points2, max_iterations,
            )
        else:
            batch_passes = weak_cc_batched(
                ctx, labels_temp, ex_scan, col_ind, core, d_flag, h_flag,
                n_rows, start2, n_points2, max_iterations,
            )
        passes += batch_passes
        if phase_timing:
            print(
                "PHASE label.weak_cc batch " + String(b2 + 1) + "/"
                + String(n_batches) + " "
                + String(Float64(perf_counter_ns() - t_cc) / 1.0e6)
                + " passes " + String(batch_passes)
            )

        if b2 > 0:
            # The labels_temp array contains the labelling for the
            # neighborhood graph of the current batch. This needs to be
            # merged with the labelling created by the previous batches.
            # Using the labelling from the previous batches as initial value
            # for weak_cc_batched and skipping the merge step would lead to
            # incorrect results as described in #3094.
            var t_ml = perf_counter_ns()
            merge_labels_run(
                ctx, labels, labels_temp, core, work_buffer, d_flag, h_flag,
                n_rows, max_iterations,
            )
            if phase_timing:
                print(
                    "PHASE label.merge_labels batch " + String(b2 + 1) + "/"
                    + String(n_batches) + " "
                    + String(Float64(perf_counter_ns() - t_ml) / 1.0e6)
                )

    # --- the border pass (DEVIATION 5130, lane cluster-apple2) -------------
    # A BATCHED fit's border labels depended on the batch count. After the
    # merges every CORE label is the fixed point of the whole graph (the
    # merge unions the per-batch components over core points), but a border
    # point's label was the smallest RAW label among its core neighbours in
    # each per-batch labelling, resolved through `R` only afterwards: the
    # raw minimum can come from a component whose resolved label is not the
    # smallest, and `reassign` cannot undo that. One batch has no such step
    # (its border label is the minimum over the final labels of its core
    # neighbours), so the 48 GB M4 Pro's two-batch taxi 100k fit gave labels
    # no one-batch column gives. This pass recomputes every border label of
    # a batched fit from the final core labels, which is the one-batch
    # answer by construction; a one-batch fit skips it (bits unchanged).
    # The last batch's CSR is still resident; every other batch's is
    # rebuilt with the same count and fill loop 2 used.
    if n_batches > 1:
        var t_bp = perf_counter_ns()
        # A batch needs the pass only if it holds a non-core row with a
        # label: a non-core row still at MAX_LABEL after the merges has no
        # core neighbour (its own batch pulls from every core neighbour, the
        # core mask being global), so it is noise in every batching.
        # cpu3-neighbors: the device decision is the only route (every mode
        # and column); the host walk over the downloaded core mask and
        # labels is gone from the GPU fit.
        var d_needs = ctx.enqueue_create_buffer[DType.int32](n_batches)
        var h_needs = ctx.enqueue_create_host_buffer[DType.int32](n_batches)
        ctx.enqueue_memset(d_needs, Int32(0))
        for nb2 in range(n_batches):
            var np_n = plan_rows[nb2]
            if np_n > 0:
                ctx.enqueue_function[border_needs_kernel](
                    d_needs.unsafe_ptr(), core.unsafe_ptr(),
                    labels.unsafe_ptr(), Int32(nb2),
                    Int32(plan_start[nb2]), Int32(np_n),
                    grid_dim=((np_n + TPB - 1) // TPB, 1, 1),
                    block_dim=(TPB, 1, 1),
                )
        ctx.enqueue_copy(dst_ptr=h_needs.unsafe_ptr(), src_buf=d_needs)
        ctx.synchronize()
        var bb = n_batches - 1
        while bb >= 0:
            var start_b = plan_start[bb]
            var np_b = plan_rows[bb]
            var needs = h_needs.unsafe_ptr().unsafe_load(bb) != Int32(0)
            if not needs:
                bb -= 1
                continue
            if np_b > 0 and bb < n_batches - 1:
                var kb = _kept_find(kept_start, start_b)
                if sparse_rbc_mode and kb >= 0:
                    # D2: the range's kept CSR, no distance pass
                    ctx.enqueue_copy(
                        dst_buf=ex_scan.create_sub_buffer[DType.int32](0, np_b + 1),
                        src_buf=kept_ia[kb],
                    )
                    if kept_nnz[kb] > 0:
                        ctx.enqueue_copy(
                            dst_buf=col_ind.create_sub_buffer[DType.int32](0, kept_nnz[kb]),
                            src_buf=kept_ja[kb].create_sub_buffer[DType.int32](0, kept_nnz[kb]),
                        )
                    ctx.synchronize()
                elif sparse_rbc_mode:
                    var qbb = x.create_sub_buffer[DType.float32](
                        start_b * n_features, np_b * n_features
                    )
                    if keep_counts:
                        ctx.enqueue_copy(
                            dst_buf=vd.create_sub_buffer[DType.int32](0, np_b),
                            src_buf=vd_all.create_sub_buffer[DType.int32](
                                start_b, np_b
                            ),
                        )
                        rbc_exclusive_scan_launch(ctx, ex_scan, vd, np_b)
                    else:
                        var _nnzb = rbc_eps_nn_query_count(
                            ctx, rbc_xr, qbb, rbc_r, rbc_ip, rbc_c1, rbc_d1,
                            rbc_rad, ex_scan, vd, np_b, n_features,
                            n_landmarks, eps_radius,
                        )
                    ctx.synchronize()
                    rbc_eps_nn_query_fill(
                        ctx, rbc_xr, qbb, rbc_r, rbc_ip, rbc_c1, rbc_d1,
                        rbc_rad, ex_scan, col_ind, np_b, n_features,
                        n_landmarks, eps_radius,
                    )
                    ctx.synchronize()
                else:
                    vertex_deg_dispatch(
                        ctx, adj, vd, x, start_b, np_b, n_rows, n_features,
                        eps, metric,
                    )
                    ctx.synchronize()
                    adj_graph_run(
                        ctx, adj, vd, ex_scan, col_ind, block_sums, np_b,
                        n_rows,
                    )
                    ctx.synchronize()
            if np_b > 0:
                ctx.enqueue_function[border_pull_kernel](
                    labels.unsafe_ptr(), ex_scan.unsafe_ptr(),
                    col_ind.unsafe_ptr(), core.unsafe_ptr(), Int32(start_b),
                    Int32(np_b), Int32(n_rows),
                    grid_dim=((np_b + TPB - 1) // TPB, 1, 1),
                    block_dim=(TPB, 1, 1),
                )
                ctx.synchronize()
            bb -= 1
        _ = d_needs^
        _ = h_needs^
        if phase_timing:
            print(
                "PHASE border_pass batches " + String(n_batches) + " "
                + String(Float64(perf_counter_ns() - t_bp) / 1.0e6)
            )

    # --- THE STAGE HASHES (`core/identity_trace.mojo`) --------------------
    # THREE RECORDS, AND THE CHOICE OF THREE IS THE WHOLE POINT.
    #
    # A tag must name a position in the ALGORITHM and never a property of
    # the machine (rule 2 in that file), and DBSCAN's per-batch stages fail
    # that test outright: `max_mbytes_per_batch = 0` -- the DEFAULT, and
    # cuML's -- derives the batch count from the DEVICE'S FREE MEMORY
    # (`dbscan.mojo:151`), so `batch03.core` exists on one machine and not
    # on another and the differ would align two disjoint tag sets.
    #
    # These three exist on every machine at every batch count:
    #
    #   dbscan.core            the core mask over all N, complete once the
    #                          neighbourhood loop has finished. A
    #                          divergence HERE is the float distance and
    #                          the eps compare -- the only float
    #                          arithmetic in DBSCAN.
    #   dbscan.labels.merged   the propagation's fixed point, before the
    #                          ids are renumbered. A divergence here with
    #                          `dbscan.core` agreeing is the propagation,
    #                          which by construction should not have one.
    #   dbscan.labels.final    what the caller gets.
    #
    # The batch-count invariance the omitted per-batch records would have
    # tested is gated directly instead, by
    # `check_dbscan_batch_count_invariance`.
    _dbscan_finish(
        ctx, labels, core, work_buffer, block_sums, n_rows, n_features, eps,
        min_pts, n_batches, metric, has_weights, phase_timing,
    )
    return passes


def _dbscan_finish(
    ctx: DeviceContext,
    mut labels: DeviceBuffer[DType.int32],
    mut core: DeviceBuffer[DType.uint8],
    mut work_buffer: DeviceBuffer[DType.int32],
    mut block_sums: DeviceBuffer[DType.int32],
    n_rows: Int,
    n_features: Int,
    eps: Float64,
    min_pts: Int,
    n_batches: Int,
    metric: Int,
    has_weights: Bool,
    phase_timing: Bool,
) raises:
    """The identity trace and `final_relabel` + `relabelForSkl`, shared by
    the reference route (and, until 2026-10-09, the FAST dense-ball route)."""
    var trace = IdentityTrace()
    if trace.enabled:
        trace.header(
            String("dbscan n=") + String(n_rows) + " d="
            + String(n_features) + " eps=" + String(eps)
            + " min_pts=" + String(min_pts) + " batches="
            + String(n_batches) + " metric=" + dbscan_metric_name(metric)
            + " weighted=" + String(has_weights)
        )
        trace.record_device(ctx, "dbscan.core", core, n_rows)
        trace.record_device(ctx, "dbscan.labels.merged", labels, n_rows)

    # --- final relabel (`runner.cuh:410-416`) -----------------------------
    # `if (algo_ccl == 2) final_relabel(labels, N, stream);` and cuML's own
    # `dbscanFitImpl` hardcodes `algo_ccl = 2` (`dbscan.cuh:122`), so this is
    # not optional in their dispatch.
    var t_fr = perf_counter_ns()
    var rank = ctx.enqueue_create_buffer[DType.int32](n_rows + 1)
    ctx.synchronize()
    make_monotonic(ctx, labels, work_buffer, rank, block_sums, n_rows)
    ctx.enqueue_function[relabel_for_skl_kernel](
        labels.unsafe_ptr(),
        Int32(n_rows),
        grid_dim=((n_rows + TPB - 1) // TPB, 1, 1),
        block_dim=(TPB, 1, 1),
    )
    ctx.synchronize()
    if phase_timing:
        print(
            "PHASE final_relabel batch 1/1 "
            + String(Float64(perf_counter_ns() - t_fr) / 1.0e6)
        )
    if trace.enabled:
        trace.record_device(ctx, "dbscan.labels.final", labels, n_rows)

