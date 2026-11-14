# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The ET forest digest of one small fixture, for the three depth-wise arms.

Lane fix-wave2-leftovers (2026-11-14): the ET early-exit device loop
(`-D MOJOLEARN_IDN_ET_DEVICE_LOOP_EARLY_EXIT`) had no equality check. The ET
route is a compile-time choice (no host-queue hook in the product builder,
unlike RF's `force_host_queue`), so the comparison spans three binaries, one
per arm, each printing the same `ET_FOREST_DIGEST` line:

    default                                  the host queue (`_et_run_depthwise_host`)
    -D MOJOLEARN_IDN_ET_DEVICE_LOOP          the plain device loop (B arm)
    -D MOJOLEARN_IDN_ET_DEVICE_LOOP_EARLY_EXIT  the early-exit device loop

`checks/native_tests/et_device_loop_early_exit_gpu.sh` (pixi task
`check-native-et-device-loop-early-exit-gpu`) runs all three and requires one
digest. The digest covers every node word (column, threshold bits, metric
bits, left child, instance count) and every leaf value bit of every tree.

The fixture makes the DEVIATION 205 rescue matter and the early exit fire:
12 columns, `max_features` 0.25 (k = 3), only columns 0-3 informative and
columns 4-11 constant, so about one node in four draws only constant columns
and is rescued; `max_batch_size` 4 keeps batches small, so many batches have
no node to rescue, which is where the early exit skips the rescue.
"""

from max.gpu.host import DeviceContext

from extratrees.impl.decisiontree.decisiontree import DecisionTreeParams
from extratrees.impl.decisiontree.flatnode import TreeMetaDataNode
from extratrees.impl.decisiontree.batched_levelalgo.builder import (
    ET_DEVICE_LOOP,
    ET_DEVICE_LOOP_EARLY_EXIT,
    train_forest_classification_device,
    upload_dataset,
)
from extratrees.impl.randomforest.randomforest import class_ids_for_host

comptime N_ROWS = 1200
comptime N_COLS = 12
comptime N_INFORMATIVE = 4
comptime N_CLASSES = 3
comptime N_TREES = 6
comptime SEED = UInt64(0xE7_5EED)


def _hash01(i: Int, j: Int) -> Float32:
    """A hashed value in [0, 1), distinct per (row, column)."""
    var h = UInt32(i * 2654435761 + j * 40503 + 17)
    h ^= h >> 15
    h = h * UInt32(2246822519)
    h ^= h >> 13
    return Float32(Int(h % UInt32(1000))) / Float32(1000.0)


def _fnv(h: UInt64, word: UInt64) -> UInt64:
    """FNV-1a over the eight bytes of `word`."""
    var x = h
    for k in range(8):  # small-loop(the eight bytes of one word)
        x ^= (word >> UInt64(8 * k)) & UInt64(0xFF)
        x *= UInt64(0x100000001B3)
    return x


def _arm() -> String:
    comptime if ET_DEVICE_LOOP_EARLY_EXIT:
        return String("early-exit device loop")
    elif ET_DEVICE_LOOP:
        return String("plain device loop")
    else:
        return String("host queue")


def run_checks() raises:
    var ctx = DeviceContext()
    # Column-major X: columns 0-3 hashed, 4-11 constant; the class is two
    # thresholds on the informative columns plus a hashed 15% flip.
    var x = List[Float32](length=N_ROWS * N_COLS, fill=Float32(1.0))
    var y = List[Float32](length=N_ROWS, fill=Float32(0.0))
    for i in range(N_ROWS):  # fixture setup on the host, not product code
        for j in range(N_INFORMATIVE):
            x[j * N_ROWS + i] = _hash01(i, j)
        var cls = 0
        if _hash01(i, 0) + Float32(0.3) * _hash01(i, 2) > Float32(0.6):
            cls += 1
        if _hash01(i, 1) - Float32(0.2) * _hash01(i, 3) > Float32(0.5):
            cls = 2
        if _hash01(i, 99) < Float32(0.15):
            cls = (cls + 1) % N_CLASSES
        y[i] = Float32(cls)
    var class_ids = class_ids_for_host(y, Int32(N_ROWS), Int32(N_CLASSES))
    var dev = upload_dataset(ctx, x, class_ids, Int32(N_ROWS), Int32(N_COLS), Int32(N_CLASSES))
    var p = DecisionTreeParams()
    p.max_depth = 7
    p.max_features = Float32(0.25)
    p.max_batch_size = 4
    var ids = List[Int32]()
    for t in range(N_TREES):  # fixture setup: the forest's tree ids
        ids.append(Int32(t))
    var forest = train_forest_classification_device(ctx, dev, p, ids, SEED)
    if len(forest) != N_TREES:
        raise Error("et_device_loop_early_exit: " + String(len(forest)) + " trees, want " + String(N_TREES))

    var h = UInt64(0xCBF29CE484222325)
    var splits = 0
    var nodes = 0
    for t in range(len(forest)):  # check driver: over the forest's trees
        ref tree = forest[t]
        h = _fnv(h, UInt64(tree.num_nodes()))
        for i in range(tree.num_nodes()):  # check driver: over the tree's nodes
            ref n = tree.sparsetree[i]
            nodes += 1
            if n.colid >= Int32(0) and n.left_child_id >= Int32(0):
                splits += 1
                if n.colid >= Int32(N_INFORMATIVE):
                    raise Error("et_device_loop_early_exit: a split landed on a constant column: fixture is wrong")
            h = _fnv(h, UInt64(Int(n.colid) & 0xFFFFFFFF))
            h = _fnv(h, UInt64(Int(n.quesval.to_bits())))
            h = _fnv(h, UInt64(Int(n.best_metric_val.to_bits())))
            h = _fnv(h, UInt64(Int(n.left_child_id) & 0xFFFFFFFF))
            h = _fnv(h, UInt64(Int(n.instance_count) & 0xFFFFFFFF))
        h = _fnv(h, UInt64(len(tree.vector_leaf)))
        for i in range(len(tree.vector_leaf)):  # check driver: over the tree's leaf values
            h = _fnv(h, UInt64(Int(tree.vector_leaf[i].to_bits())))
    if splits < 8 * N_TREES:
        raise Error("et_device_loop_early_exit: fixture too small: " + String(splits) + " splits")
    _ = dev^
    print("et_device_loop_early_exit: arm", _arm(), ",", N_TREES, "trees,", nodes, "nodes,", splits, "splits")
    print("ET_FOREST_DIGEST", hex(h))
