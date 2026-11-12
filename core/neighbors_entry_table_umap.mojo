# SPDX-License-Identifier: Apache-2.0
"""The UMAP operations table (lane calltable-cn-umap, 2026-11-11): the UMAP
entries of the metrics binding `_mojolearn_metrics` (device,
umap/ops_entry_device.mojo) and of its CPU column `_mojolearn_metrics_host`
(umap/ops_entry_host.mojo) as one `NativeDescriptorCall` per image,
dispatching on its operation id. UMAP's owner
(bindings/mani_umap_native.mojo) calls it with plain values instead of the
bindings' Python `umap_fit_transform` / `umap_fit_transform_ex` /
`umap_transform` exports, and compares the key's mode with the requested
mode instead of calling `umap_numeric_mode`.

Modeled on core/neighbors_entry_table.mojo: the same core/entry_table.mojo
publication and lookup, separate slot names per tier and vendor, and its
key struct and vendor words (0 cpu, 1 metal, 2 cuda, 3 hip). An image
records its key as module attribute `umap_native_key = (mode, vendor)` when
it publishes; the owner reads that attribute at UNPACK. Publication and
lookup are metadata operations with no independent numerical work to split.

Request layouts (a: addresses, p: Int words, f: Float64 words). Every float
word is rounded to Float32 by the entry, the same `Float32(Float64(py=...))`
rounding the Python exports apply, so the bits are unchanged.
  UMAP_OP_FIT        a [x, embedding]
                     p [n, d, n_neighbors, n_components, n_epochs, seed (, negative_sample_rate)]
                     f [min_dist, spread, set_op_mix_ratio, local_connectivity (, learning_rate,
                        repulsion_strength)]                                   -> n_components
                     (the export's 10 / 13 words: 6 + 4 or 7 + 6)
  UMAP_OP_FIT_EX     a [x, embedding, initial (0: spectral), target (0: none)]
                     p [n, d, n_neighbors, n_components, n_epochs, seed, negative_sample_rate, metric,
                        target_kind, target_dims, target_n_neighbors]
                     f [min_dist, spread, set_op_mix_ratio, local_connectivity, learning_rate,
                        repulsion_strength, metric_arg, a, b, target_weight]  -> n_components
  UMAP_OP_TRANSFORM  a [training, fitted embedding, queries, output]
                     p [n_train, n_queries, n_features, n_neighbors, n_components, n_epochs, seed
                        (, negative_sample_rate (, metric))]
                     f [min_dist, spread, set_op_mix_ratio, local_connectivity (, learning_rate,
                        repulsion_strength (, metric_arg, a, b))]            -> n_components
                     (the export's 11 / 14 / 18 words: 7 + 4, 8 + 6 or 9 + 9)
"""
from core.entry_table import entry_table_publish, entry_table_read
from core.native_descriptor_abi import NativeDescriptorCall
from core.native_descriptor_call import native_descriptor_call
from core.neighbors_entry_table import NeighborsEntryKey

comptime UMAP_OPS_ABI = "mojolearn.umap.ops.v1"
comptime UMAP_OP_FIT = 0
comptime UMAP_OP_FIT_EX = 1
comptime UMAP_OP_TRANSFORM = 2
comptime UMAP_OPERATION_COUNT = 3

#: The transform's optional tails: the eight legacy words only, plus
#: learning_rate / repulsion_strength / negative_sample_rate, plus the
#: metric, metric_arg, a and b.
comptime UMAP_TAIL_LEGACY = 0
comptime UMAP_TAIL_RATES = 1
comptime UMAP_TAIL_EXTRAS = 2


def umap_ops_check_layout(operation: Int, na: Int, np: Int, nf: Int) raises:
    """Refuse a request whose word counts do not match the operation's
    layout before any word is read."""
    if operation == UMAP_OP_FIT:
        if na == 2 and ((np == 6 and nf == 4) or (np == 7 and nf == 6)):
            return
    elif operation == UMAP_OP_FIT_EX:
        if na == 4 and np == 11 and nf == 10:
            return
    elif operation == UMAP_OP_TRANSFORM:
        if na == 4 and ((np == 7 and nf == 4) or (np == 8 and nf == 6) or (np == 9 and nf == 9)):
            return
    else:
        raise Error("umap ops: unknown operation " + String(operation))
    raise Error("umap ops: request spans differ from operation " + String(operation))


def umap_ops_check_spans(operation: Int, aa: Int, na: Int, pp: Int, np: Int,
                         ff: Int, nf: Int, result: Int) raises:
    if not result:
        raise Error("umap ops: null result word")
    if (na > 0 and not aa) or (np > 0 and not pp) or (nf > 0 and not ff) or na < 0 or np < 0 or nf < 0:
        raise Error("umap ops: null or invalid request span")
    umap_ops_check_layout(operation, na, np, nf)


def umap_ops_publish(key: NeighborsEntryKey, publisher: Int, entry: NativeDescriptorCall) raises:
    var schemas: List[String] = [String(UMAP_OPS_ABI)]
    var operations: List[Int] = [UMAP_OPERATION_COUNT]
    var entries: List[NativeDescriptorCall] = [entry]
    if key.mode == 1 and key.vendor == 0:
        entry_table_publish["MojolearnEntriesUmapOpsIdenticalCPUV1"](key.mode, key.vendor, publisher, schemas, operations, entries)
    elif key.mode == 0 and key.vendor == 0:
        entry_table_publish["MojolearnEntriesUmapOpsFastCPUV1"](key.mode, key.vendor, publisher, schemas, operations, entries)
    elif key.mode == 1 and key.vendor == 1:
        entry_table_publish["MojolearnEntriesUmapOpsIdenticalMetalV1"](key.mode, key.vendor, publisher, schemas, operations, entries)
    elif key.mode == 0 and key.vendor == 1:
        entry_table_publish["MojolearnEntriesUmapOpsFastMetalV1"](key.mode, key.vendor, publisher, schemas, operations, entries)
    elif key.mode == 1 and key.vendor == 2:
        entry_table_publish["MojolearnEntriesUmapOpsIdenticalCUDAV1"](key.mode, key.vendor, publisher, schemas, operations, entries)
    elif key.mode == 0 and key.vendor == 2:
        entry_table_publish["MojolearnEntriesUmapOpsFastCUDAV1"](key.mode, key.vendor, publisher, schemas, operations, entries)
    elif key.mode == 1 and key.vendor == 3:
        entry_table_publish["MojolearnEntriesUmapOpsIdenticalHIPV1"](key.mode, key.vendor, publisher, schemas, operations, entries)
    elif key.mode == 0 and key.vendor == 3:
        entry_table_publish["MojolearnEntriesUmapOpsFastHIPV1"](key.mode, key.vendor, publisher, schemas, operations, entries)
    else:
        raise Error("native umap ops: unsupported execution key")
    _ = schemas
    _ = operations
    _ = entries


def umap_ops_entry(key: NeighborsEntryKey) raises -> NativeDescriptorCall:
    var schema = String(UMAP_OPS_ABI)
    if key.mode == 1 and key.vendor == 0:
        return entry_table_read["MojolearnEntriesUmapOpsIdenticalCPUV1"](key.mode, key.vendor, 0, schema, UMAP_OPERATION_COUNT)
    if key.mode == 0 and key.vendor == 0:
        return entry_table_read["MojolearnEntriesUmapOpsFastCPUV1"](key.mode, key.vendor, 0, schema, UMAP_OPERATION_COUNT)
    if key.mode == 1 and key.vendor == 1:
        return entry_table_read["MojolearnEntriesUmapOpsIdenticalMetalV1"](key.mode, key.vendor, 0, schema, UMAP_OPERATION_COUNT)
    if key.mode == 0 and key.vendor == 1:
        return entry_table_read["MojolearnEntriesUmapOpsFastMetalV1"](key.mode, key.vendor, 0, schema, UMAP_OPERATION_COUNT)
    if key.mode == 1 and key.vendor == 2:
        return entry_table_read["MojolearnEntriesUmapOpsIdenticalCUDAV1"](key.mode, key.vendor, 0, schema, UMAP_OPERATION_COUNT)
    if key.mode == 0 and key.vendor == 2:
        return entry_table_read["MojolearnEntriesUmapOpsFastCUDAV1"](key.mode, key.vendor, 0, schema, UMAP_OPERATION_COUNT)
    if key.mode == 1 and key.vendor == 3:
        return entry_table_read["MojolearnEntriesUmapOpsIdenticalHIPV1"](key.mode, key.vendor, 0, schema, UMAP_OPERATION_COUNT)
    if key.mode == 0 and key.vendor == 3:
        return entry_table_read["MojolearnEntriesUmapOpsFastHIPV1"](key.mode, key.vendor, 0, schema, UMAP_OPERATION_COUNT)
    raise Error("native umap ops: unsupported execution key")


# ---------------------------------------------------------------- typed calls


@fieldwise_init
struct UmapConfigWords(Copyable, Movable):
    """UMAP's validated configuration as plain words: the owner's `config`
    (11 values) and `extras` (4 values) of
    bindings/umap_parameters_native.mojo `umap_parameters_fields`, read once
    at UNPACK."""
    var n_neighbors: Int
    var n_components: Int
    var n_epochs: Int
    var min_dist: Float64
    var spread: Float64
    var set_op_mix_ratio: Float64
    var local_connectivity: Float64
    var seed: Int
    var learning_rate: Float64
    var repulsion_strength: Float64
    var negative_sample_rate: Int
    var metric: Int
    var metric_arg: Float64
    var curve_a: Float64
    var curve_b: Float64

    def legacy_rates(self) -> Bool:
        """`config[8:] == [1.0, 1.0, 5]`: the rates the legacy layout
        leaves out."""
        return self.learning_rate == 1.0 and self.repulsion_strength == 1.0 and self.negative_sample_rate == 5

    def default_extras(self) -> Bool:
        """`extras == (-1, 2.0, 0.0, 0.0)`."""
        return self.metric == -1 and self.metric_arg == 2.0 and self.curve_a == 0.0 and self.curve_b == 0.0


@fieldwise_init
struct UmapTargetWords(Copyable, Movable):
    """The supervised target (`kind` 0: none; its address 0)."""
    var kind: Int
    var dims: Int
    var n_neighbors: Int
    var weight: Float64
    var address: Int

    @staticmethod
    def none() -> UmapTargetWords:
        return UmapTargetWords(0, 1, 0, 0.5, 0)


def _base_words(mut ints: List[Int], mut floats: List[Float64], cfg: UmapConfigWords):
    ints.append(cfg.n_neighbors)
    ints.append(cfg.n_components)
    ints.append(cfg.n_epochs)
    ints.append(cfg.seed)
    floats.append(cfg.min_dist)
    floats.append(cfg.spread)
    floats.append(cfg.set_op_mix_ratio)
    floats.append(cfg.local_connectivity)


def umap_fit_call(entry: NativeDescriptorCall, x: Int, embedding: Int, n: Int, d: Int, cfg: UmapConfigWords,
                  rates: Bool) raises -> Int:
    """The legacy fit (`umap_fit_transform`): `rates` False sends the eight
    legacy words only (the export's 10-word form)."""
    var addresses: List[Int] = [x, embedding]
    var ints: List[Int] = [n, d]
    var floats = List[Float64]()
    _base_words(ints, floats, cfg)
    if rates:
        ints.append(cfg.negative_sample_rate)
        floats.append(cfg.learning_rate)
        floats.append(cfg.repulsion_strength)
    return native_descriptor_call(entry, UMAP_OP_FIT, addresses, ints, floats)


def umap_fit_ex_call(entry: NativeDescriptorCall, x: Int, embedding: Int, initial: Int, n: Int, d: Int,
                     cfg: UmapConfigWords, target: UmapTargetWords) raises -> Int:
    var addresses: List[Int] = [x, embedding, initial, target.address]
    var ints: List[Int] = [n, d]
    var floats = List[Float64]()
    _base_words(ints, floats, cfg)
    ints.append(cfg.negative_sample_rate)
    ints.append(cfg.metric)
    ints.append(target.kind)
    ints.append(target.dims)
    ints.append(target.n_neighbors)
    floats.append(cfg.learning_rate)
    floats.append(cfg.repulsion_strength)
    floats.append(cfg.metric_arg)
    floats.append(cfg.curve_a)
    floats.append(cfg.curve_b)
    floats.append(target.weight)
    return native_descriptor_call(entry, UMAP_OP_FIT_EX, addresses, ints, floats)


def umap_transform_call(entry: NativeDescriptorCall, training: Int, fitted: Int, queries: Int, dst: Int,
                        n_train: Int, n_queries: Int, d: Int, cfg: UmapConfigWords, tail: Int) raises -> Int:
    """`tail`: UMAP_TAIL_LEGACY, UMAP_TAIL_RATES or UMAP_TAIL_EXTRAS (the
    export's 11, 14 or 18 words)."""
    var addresses: List[Int] = [training, fitted, queries, dst]
    var ints: List[Int] = [n_train, n_queries, d]
    var floats = List[Float64]()
    _base_words(ints, floats, cfg)
    if tail >= UMAP_TAIL_RATES:
        ints.append(cfg.negative_sample_rate)
        floats.append(cfg.learning_rate)
        floats.append(cfg.repulsion_strength)
    if tail >= UMAP_TAIL_EXTRAS:
        ints.append(cfg.metric)
        floats.append(cfg.metric_arg)
        floats.append(cfg.curve_a)
        floats.append(cfg.curve_b)
    return native_descriptor_call(entry, UMAP_OP_TRANSFORM, addresses, ints, floats)
