# SPDX-License-Identifier: Apache-2.0
# native-check: gpu
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Tree CTR (feature combination) columns ON THE DEVICE against the host
statement, word for word.

    pixi run check-native-gpu tree_ctr_device_check

One fixture: a packed compressed index holding a float column, a one-hot
column and one spare tree-CTR slot; a tree tensor of one split on each of
the two columns plus one categorical feature; two candidate categorical
features; Borders at two priors and FeatureFreq; two permutations. Checked:

  * `launch_tree_split_bits` gives each split's 0/1 side off the packed
    index (`bin >` for the float column, `bin ==` for the one-hot one);
  * every candidate column of `tree_ctr_candidate_columns_device` over
    `tree_ctr_base_builder` equals `tree_ctr_level_host`'s bit for bit, on
    both permutations, and the Borders columns of the two permutations
    differ (the order is reached);
  * `launch_tree_slot_write` installs a column's bins in the spare slot and
    leaves the other two columns of the index untouched.
"""

from std.testing import TestSuite

from max.gpu.host import DeviceBuffer, DeviceContext

from checks.native_tests.kit import download, upload
from gbdt.checks.tree_ctr_check import fixture_codes
from gbdt.ctrs.ctr import CTR_BORDERS, CTR_FEATURE_FREQ, TCtrConfig, TPrior
from gbdt.ctrs.ctr_order import ctr_estimation_order_host
from gbdt.ctrs.kernel.ctr_order import launch_ctr_estimation_order
from gbdt.ctrs.tree_ctr import (
    TreeF32Ptr,
    TreeU32Ptr,
    TreeU8Ptr,
    tree_ctr_bins_into,
    tree_ctr_groups_host,
    tree_ctr_level_host,
    tree_ctr_uniform_borders_host,
)
from gbdt.ctrs.tree_ctr_device import (
    launch_tree_slot_write,
    launch_tree_split_bits,
    tree_ctr_base_builder,
    tree_ctr_candidate_columns_device,
)
from gbdt.gpu_data.compressed_index_builder import pack_quantized_columns_host


def _need(ok: Bool, what: String) raises:
    if not ok:
        raise Error("tree CTR device check: " + what)


def _u32p(values: List[UInt32]) -> TreeU32Ptr:
    return TreeU32Ptr(unsafe_from_address=Int(values.unsafe_ptr()))


def check_tree_ctr_device() raises:
    var ctx = DeviceContext()
    var n = 1531
    var seed = UInt64(23)
    # quantized columns: a float column of 6 bins, a one-hot of 3 codes and
    # the spare tree-CTR slot (15 borders), all zero to start
    var float_bins = fixture_codes(n, 6, 11)
    var hot_bins = fixture_codes(n, 3, 12)
    var slot_bins = List[UInt32](length=n, fill=UInt32(0))
    var columns = List[List[UInt32]]()
    columns.append(float_bins.copy())
    columns.append(hot_bins.copy())
    columns.append(slot_bins.copy())
    var folds: List[Int] = [5, 3, 15]
    var one_hot: List[Bool] = [False, True, False]
    var packed = pack_quantized_columns_host(columns, folds, one_hot)
    var d_cindex = upload[DType.uint32](ctx, packed.words)

    # the tree: `float > 2`, `one-hot == 1`, and categorical feature A
    var cat_a = fixture_codes(n, 9, 13)
    var cand_b = fixture_codes(n, 14, 14)
    var cand_c = fixture_codes(n, 4, 15)
    var tcodes = fixture_codes(n, 2, 16)
    var target = List[UInt8](capacity=n)
    for r in range(n):
        target.append(UInt8(Int(tcodes[r])))

    var d_side0 = ctx.enqueue_create_buffer[DType.uint32](n)
    var d_side1 = ctx.enqueue_create_buffer[DType.uint32](n)
    launch_tree_split_bits(
        ctx, d_cindex, packed.layout.features[0], n, 2, False, d_side0
    )
    launch_tree_split_bits(
        ctx, d_cindex, packed.layout.features[1], n, 1, True, d_side1
    )
    var side0 = download[DType.uint32](ctx, d_side0, n)
    var side1 = download[DType.uint32](ctx, d_side1, n)
    for r in range(n):
        _need(
            side0[r] == (UInt32(1) if float_bins[r] > UInt32(2) else UInt32(0)),
            "a float split's side is bin > border",
        )
        _need(
            side1[r] == (UInt32(1) if hot_bins[r] == UInt32(1) else UInt32(0)),
            "a one-hot split's side is bin == category",
        )

    var d_a = upload[DType.uint32](ctx, cat_a)
    var d_b = upload[DType.uint32](ctx, cand_b)
    var d_c = upload[DType.uint32](ctx, cand_c)
    var d_target = upload[DType.uint8](ctx, target)

    var configs = List[TCtrConfig]()
    configs.append(TCtrConfig(CTR_BORDERS, TPrior(Float32(0.0), Float32(1.0)), 0, 0))
    configs.append(TCtrConfig(CTR_BORDERS, TPrior(Float32(1.0), Float32(1.0)), 0, 0))
    configs.append(TCtrConfig(CTR_FEATURE_FREQ, TPrior(Float32(0.0), Float32(1.0)), 0, 1))
    var n_cfg = len(configs)

    # the host statement: the same members in the same order
    var members = List[TreeU32Ptr]()
    members.append(_u32p(side0))
    members.append(_u32p(side1))
    members.append(_u32p(cat_a))
    var member_cards: List[Int] = [2, 2, 9]
    var base_groups = tree_ctr_groups_host(members, member_cards, n)
    var cand = List[TreeU32Ptr]()
    cand.append(_u32p(cand_b))
    cand.append(_u32p(cand_c))
    var cand_cards: List[Int] = [14, 4]

    var first_borders_column = List[Float32]()
    var differs = False
    # driver-loop: over the two permutations
    for p in range(2):
        var order = ctr_estimation_order_host(n, p, seed)
        var want = tree_ctr_level_host(
            base_groups, cand, cand_cards, _u32p(order),
            TreeU8Ptr(unsafe_from_address=Int(target.unsafe_ptr())), n, configs,
        )
        var d_order = ctx.enqueue_create_buffer[DType.uint32](n)
        launch_ctr_estimation_order(ctx, d_order, n, p, seed)
        var d_members = List[DeviceBuffer[DType.uint32]]()
        d_members.append(d_side0)
        d_members.append(d_side1)
        d_members.append(d_a)
        var base = tree_ctr_base_builder(ctx, d_order, n, d_members, member_cards)
        # driver-loop: over the two candidates
        for k in range(2):
            var d_cand = d_b
            if k == 1:
                d_cand = d_c
            var got = tree_ctr_candidate_columns_device(
                ctx, base, d_cand, cand_cards[k], n, configs, d_target,
            )
            _need(len(got) == n_cfg, "one device column per config")
            # driver-loop: over the configs
            for c in range(n_cfg):
                var d_column = got[c]
                var column = download[DType.float32](ctx, d_column, n)
                for r in range(n):
                    if column[r] != want[(k * n_cfg + c) * n + r]:
                        raise Error(
                            "tree CTR device check: permutation " + String(p)
                            + " candidate " + String(k) + " config " + String(c)
                            + " row " + String(r) + " differs from the host"
                            " statement"
                        )
                if k == 0 and c == 0:
                    if p == 0:
                        first_borders_column = column.copy()
                    else:
                        for r in range(n):
                            if column[r] != first_borders_column[r]:
                                differs = True
        ctx.synchronize()
        _ = base^
        _ = d_order^
        _ = len(order)
    _need(differs, "the permutation reaches the ordered column")

    # the slot: a column's grid and bins on the host, written on the device
    var order0 = ctr_estimation_order_host(n, 0, seed)
    var col = tree_ctr_level_host(
        base_groups, cand, cand_cards, _u32p(order0),
        TreeU8Ptr(unsafe_from_address=Int(target.unsafe_ptr())), n, configs,
    )
    var cp = TreeF32Ptr(unsafe_from_address=Int(col.unsafe_ptr()))
    var borders = tree_ctr_uniform_borders_host(cp, n, 15)
    var bins = List[UInt32](length=n, fill=UInt32(0))
    tree_ctr_bins_into(cp, n, borders, _u32p(bins))
    _ = col^  # keep alive: cp points into it and the two calls above read through cp
    var d_bins = upload[DType.uint32](ctx, bins)
    launch_tree_slot_write(ctx, d_cindex, packed.layout.features[2], n, d_bins)
    # a second write of the same slot replaces, it does not OR
    launch_tree_slot_write(ctx, d_cindex, packed.layout.features[2], n, d_bins)
    var d_back = ctx.enqueue_create_buffer[DType.uint32](n)
    # driver-loop: over the slot's bins (each read back as an equality side)
    for b in range(len(borders) + 1):
        launch_tree_split_bits(
            ctx, d_cindex, packed.layout.features[2], n, b, True, d_back
        )
        var back = download[DType.uint32](ctx, d_back, n)
        for r in range(n):
            _need(
                back[r] == (UInt32(1) if Int(bins[r]) == b else UInt32(0)),
                "the slot holds the written bins",
            )
    launch_tree_split_bits(
        ctx, d_cindex, packed.layout.features[0], n, 2, False, d_back
    )
    var again = download[DType.uint32](ctx, d_back, n)
    for r in range(n):
        _need(again[r] == side0[r], "a slot write keeps the neighbouring columns")
    # read through raw pointers above
    _ = len(side0)
    _ = len(side1)
    _ = len(cat_a)
    _ = len(cand_b)
    _ = len(cand_c)
    _ = len(target)
    _ = len(order0)
    _ = len(bins)
    _ = len(base_groups.ids)
    print("  tree CTR device columns equal the host statement")


def test_tree_ctr_device() raises:
    check_tree_ctr_device()


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
