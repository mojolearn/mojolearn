# SPDX-License-Identifier: Apache-2.0
"""THE CLUSTER LABEL-PAIR METRICS TABLE (lane mf-cluster5, 2026-11-23): the
entries of the metrics binding `_mojolearn_metrics` (device column,
metrics/cluster_entry_device.mojo) and of its CPU column
`_mojolearn_metrics_host` (metrics/host/cluster_entry_host.mojo) for
`rand_score`, `adjusted_rand_score`, `mutual_info_score`,
`homogeneity_score`, `completeness_score`, `v_measure_score`,
`fowlkes_mallows_score` and `homogeneity_completeness_v_measure`, as ONE
`NativeDescriptorCall` per image dispatching on its operation id. Each image
runs the SAME body its Python export runs (the export now calls it too), so
the GPU image keeps its device kernels and the host image its oracle loops:
two routes of one entry, no kernel swapped.

Callers: the public doors (bindings/mcls_cluster_door.mojo: UNPACK reads the
labels and the image's key `metrics_cluster_key = (mode, vendor)`, then the
typed RUN below with the GIL released) and the search scorers' RUN
(core/msel_scorer_run.mojo, over the entry its UNPACK resolved once per
admission). Publication and lookup are metadata operations, modeled on
core/neighbors_entry_table_umap.mojo (the same core/entry_table.mojo slots,
one per tier and vendor: 0 cpu, 1 metal, 2 cuda, 3 hip).

Request layout of every operation (a: addresses, p: Int words, f: Float64):
  a [labels_true, labels_pred, result (float64 words: 1, or 3 for CLUSTER_OP_HCV)]
  p [n, lower_class_range, upper_class_range]   (rand / adjusted rand: raw
                                                 labels; the range words unread)
  f [beta]                                       (v_measure / hcv; unread otherwise)
The entry answers the number of float64 words it wrote. No Python.
"""
from std.memory import unsafe_memcpy
from core.entry_table import EntryRef, entry_table_publish, entry_table_resolve, entry_ref_call
from core.native_descriptor_abi import NativeDescriptorCall
from core.labels_score_encode import score_encode_signed

comptime CLUSTER_OPS_ABI = "mojolearn.metrics.cluster.v1"
comptime CLUSTER_OP_RAND = 0
comptime CLUSTER_OP_ADJUSTED_RAND = 1
comptime CLUSTER_OP_MUTUAL_INFO = 2
comptime CLUSTER_OP_HOMOGENEITY = 3
comptime CLUSTER_OP_COMPLETENESS = 4
comptime CLUSTER_OP_V_MEASURE = 5
comptime CLUSTER_OP_FOWLKES_MALLOWS = 6
comptime CLUSTER_OP_HCV = 7
comptime CLUSTER_OPERATION_COUNT = 8

comptime ClusterWordPtr = MutPointer[Int, MutUntrackedOrigin]
comptime ClusterFloatPtr = MutPointer[Float64, MutUntrackedOrigin]


def cluster_ops_check_spans(operation: Int, aa: Int, na: Int, pp: Int, np: Int, ff: Int, nf: Int,
                            result: Int) raises:
    """Refuse a request whose spans do not match the one layout before any
    word is read."""
    if not result:
        raise Error("metrics cluster ops: null result word")
    if operation < 0 or operation >= CLUSTER_OPERATION_COUNT:
        raise Error("metrics cluster ops: unknown operation " + String(operation))
    if na != 3 or np != 3 or nf != 1 or not aa or not pp or not ff:
        raise Error("metrics cluster ops: request spans differ from operation " + String(operation))


def cluster_ops_publish(mode: Int, vendor: Int, publisher: Int, entry: NativeDescriptorCall) raises:
    """The image's publication (its module init), one slot per tier and vendor."""
    var schemas: List[String] = [String(CLUSTER_OPS_ABI)]
    var operations: List[Int] = [CLUSTER_OPERATION_COUNT]
    var entries: List[NativeDescriptorCall] = [entry]
    if mode == 1 and vendor == 0:
        entry_table_publish["MojolearnEntriesMetricsClusterIdenticalCPUV1"](mode, vendor, publisher, schemas, operations, entries)
    elif mode == 0 and vendor == 0:
        entry_table_publish["MojolearnEntriesMetricsClusterFastCPUV1"](mode, vendor, publisher, schemas, operations, entries)
    elif mode == 1 and vendor == 1:
        entry_table_publish["MojolearnEntriesMetricsClusterIdenticalMetalV1"](mode, vendor, publisher, schemas, operations, entries)
    elif mode == 0 and vendor == 1:
        entry_table_publish["MojolearnEntriesMetricsClusterFastMetalV1"](mode, vendor, publisher, schemas, operations, entries)
    elif mode == 1 and vendor == 2:
        entry_table_publish["MojolearnEntriesMetricsClusterIdenticalCUDAV1"](mode, vendor, publisher, schemas, operations, entries)
    elif mode == 0 and vendor == 2:
        entry_table_publish["MojolearnEntriesMetricsClusterFastCUDAV1"](mode, vendor, publisher, schemas, operations, entries)
    elif mode == 1 and vendor == 3:
        entry_table_publish["MojolearnEntriesMetricsClusterIdenticalHIPV1"](mode, vendor, publisher, schemas, operations, entries)
    elif mode == 0 and vendor == 3:
        entry_table_publish["MojolearnEntriesMetricsClusterFastHIPV1"](mode, vendor, publisher, schemas, operations, entries)
    else:
        raise Error("native metrics cluster ops: unsupported execution key")
    _ = schemas
    _ = operations
    _ = entries


def cluster_ops_entry(mode: Int, vendor: Int) raises -> EntryRef:
    """UNPACK: the image's entry, resolved ONCE (a door call, a search
    admission); every refusal of the table happens here."""
    var schema = String(CLUSTER_OPS_ABI)
    comptime C = CLUSTER_OPERATION_COUNT
    if mode == 1 and vendor == 0:
        return entry_table_resolve["MojolearnEntriesMetricsClusterIdenticalCPUV1"](mode, vendor, 0, schema, C)
    if mode == 0 and vendor == 0:
        return entry_table_resolve["MojolearnEntriesMetricsClusterFastCPUV1"](mode, vendor, 0, schema, C)
    if mode == 1 and vendor == 1:
        return entry_table_resolve["MojolearnEntriesMetricsClusterIdenticalMetalV1"](mode, vendor, 0, schema, C)
    if mode == 0 and vendor == 1:
        return entry_table_resolve["MojolearnEntriesMetricsClusterFastMetalV1"](mode, vendor, 0, schema, C)
    if mode == 1 and vendor == 2:
        return entry_table_resolve["MojolearnEntriesMetricsClusterIdenticalCUDAV1"](mode, vendor, 0, schema, C)
    if mode == 0 and vendor == 2:
        return entry_table_resolve["MojolearnEntriesMetricsClusterFastCUDAV1"](mode, vendor, 0, schema, C)
    if mode == 1 and vendor == 3:
        return entry_table_resolve["MojolearnEntriesMetricsClusterIdenticalHIPV1"](mode, vendor, 0, schema, C)
    if mode == 0 and vendor == 3:
        return entry_table_resolve["MojolearnEntriesMetricsClusterFastHIPV1"](mode, vendor, 0, schema, C)
    raise Error("native metrics cluster ops: unsupported execution key")


# ---------------------------------------------------------------- the typed RUN


def cluster_ops_call(entry: EntryRef, operation: Int, truth: Int, pred: Int, n: Int, lower: Int, upper: Int,
                     beta: Float64, dst: Int) raises -> Int:
    """One call of the image's entry over int32 label words the caller keeps
    alive; `dst` receives the float64 answer(s)."""
    var a: Array[Int, 3] = [truth, pred, dst]
    var p: Array[Int, 3] = [n, lower, upper]
    var f: Array[Float64, 1] = [beta]
    return entry_ref_call(entry, operation, a, p, f)


def cluster_union_codes(truth: Int, pred: Int, n: Int, mut codes: List[Int32]) raises -> Int:
    """`_prepare_cluster_labels` (was bindings/mcls_cluster_door.mojo
    `_cluster_codes`): both clusterings' n int32 labels, concatenated, mapped
    onto 0 .. k-1 over the sorted union of their values by THE ONE label
    encoder (core/labels_score_encode.mojo `score_encode_signed`, which the
    door called on the same words). `codes` gets 2n words: truth's, then
    pred's. Returns k."""
    var total = 2 * n
    # Uninitialized, as the door's buffers were: every word is written below
    # (the class table: only its first k words, and only k is read).
    var both = List[Int32](unsafe_uninit_length=max(total, 1))
    if n > 0:
        unsafe_memcpy(dest=both.unsafe_ptr(), src=MutPointer[Int32, MutUntrackedOrigin](unsafe_from_address=truth),
                      count=n)
        unsafe_memcpy(dest=both.unsafe_ptr() + n, src=MutPointer[Int32, MutUntrackedOrigin](unsafe_from_address=pred),
                      count=n)
    var classes = List[Int32](unsafe_uninit_length=max(total, 1))
    codes = List[Int32](unsafe_uninit_length=max(total, 1))
    var k = score_encode_signed(Int(both.unsafe_ptr()), 4, total, Int(classes.unsafe_ptr()), Int(codes.unsafe_ptr()))
    _ = len(both)  # read by the encoder above
    _ = len(classes)
    return k


def cluster_pair_run(entry: EntryRef, operation: Int, truth: Int, pred: Int, n: Int, beta: Float64) raises -> Float64:
    """THE TYPED RUN of one cluster label-pair metric (CLUSTER_OP_RAND ..
    CLUSTER_OP_FOWLKES_MALLOWS) over n int32 labels of each clustering (raw
    labels, or class codes in their labels' order: the same answer). The
    door's order: rand and adjusted rand read the raw labels; the others the
    union codes with the range [0, k-1]. Refusals are the image body's."""
    var out = List[Float64](length=3, fill=0.0)
    if operation == CLUSTER_OP_RAND or operation == CLUSTER_OP_ADJUSTED_RAND:
        _ = cluster_ops_call(entry, operation, truth, pred, n, 0, 0, beta, Int(out.unsafe_ptr()))
        return out[0]
    if operation == CLUSTER_OP_HCV:
        raise Error("RuntimeError: cluster_pair_run answers one value; use cluster_hcv_run")
    var codes = List[Int32]()
    var k = cluster_union_codes(truth, pred, n, codes)
    var base = Int(codes.unsafe_ptr())
    _ = cluster_ops_call(entry, operation, base, base + 4 * n, n, 0, k - 1, beta, Int(out.unsafe_ptr()))
    _ = len(codes)  # read by the entry above
    return out[0]


def cluster_hcv_run(entry: EntryRef, truth: Int, pred: Int, n: Int, beta: Float64) raises -> Tuple[Float64, Float64, Float64]:
    """`homogeneity_completeness_v_measure`: the union codes once, one call
    (the image's fused body), three values."""
    var out = List[Float64](length=3, fill=0.0)
    var codes = List[Int32]()
    var k = cluster_union_codes(truth, pred, n, codes)
    var base = Int(codes.unsafe_ptr())
    _ = cluster_ops_call(entry, CLUSTER_OP_HCV, base, base + 4 * n, n, 0, k - 1, beta, Int(out.unsafe_ptr()))
    _ = len(codes)
    return (out[0], out[1], out[2])
