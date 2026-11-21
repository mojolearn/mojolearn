# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Native check of the tree-CTR host statement (`gbdt/ctrs/tree_ctr.mojo`):
the tensor and its candidates, the groups against a pairwise comparison of
the member tuples, the ordered Borders and the FeatureFreq columns against
an O(n^2) tally, the level driver against the single-column entries, and
the grid. Host only; `gbdt/checks/tree_ctr_check_main.mojo` runs it."""

from gbdt.ctrs.ctr import CTR_BORDERS, CTR_FEATURE_FREQ, TCtrConfig, TPrior
from gbdt.ctrs.ctr_order import ctr_estimation_order_host
from gbdt.ctrs.tree_ctr import (
    TREE_COL_FLOAT,
    TREE_COL_ONE_HOT,
    TREE_COL_SIMPLE_CTR,
    TreeCtrGroups,
    TreeCtrPlan,
    TreeCtrSplit,
    TreeCtrTensor,
    TreeF32Ptr,
    TreeU32Ptr,
    TreeU8Ptr,
    tree_ctr_bins_into,
    tree_ctr_groups_host,
    tree_ctr_level_host,
    tree_ctr_uniform_borders_host,
)


def _need(ok: Bool, what: String) raises:
    if not ok:
        raise Error("tree CTR check: " + what)


def _u32p(values: List[UInt32]) -> TreeU32Ptr:
    return TreeU32Ptr(unsafe_from_address=Int(values.unsafe_ptr()))


def fixture_codes(n: Int, cardinality: Int, salt: Int) -> List[UInt32]:
    """Deterministic codes below `cardinality`, every code present."""
    var out = List[UInt32](capacity=n)
    var state = UInt64(salt) * UInt64(2654435761) + UInt64(12345)
    for r in range(n):
        state = state * UInt64(6364136223846793005) + UInt64(1442695040888963407)
        var code = Int((state >> 33) % UInt64(cardinality))
        if r < cardinality:
            code = r
        out.append(UInt32(code))
    return out^


def tensors() raises:
    var a = TreeCtrTensor()
    a.add_cat(5)
    a.add_split(TreeCtrSplit(2, 3, False))
    a.add_cat(1)
    a.add_split(TreeCtrSplit(0, 7, True))
    var b = TreeCtrTensor()
    b.add_split(TreeCtrSplit(0, 7, True))
    b.add_cat(1)
    b.add_split(TreeCtrSplit(2, 3, False))
    b.add_cat(5)
    b.add_cat(5)
    b.add_split(TreeCtrSplit(2, 3, False))
    _need(a.same(b), "a tensor is its members, not its build order")
    _need(a.cats[0] == 1 and a.cats[1] == 5, "categorical members ascend")
    _need(a.splits[0].feature == 0 and a.splits[1].feature == 2, "splits ascend")
    _need(a.complexity() == 3, "two categorical members and all splits as one")
    var c = TreeCtrTensor()
    c.add_split(TreeCtrSplit(4, 0, False))
    c.add_split(TreeCtrSplit(4, 1, False))
    c.add_split(TreeCtrSplit(6, 0, False))
    _need(c.complexity() == 1, "any number of splits counts one")
    c.add_tensor(a)
    _need(len(c.cats) == 2 and len(c.splits) == 5, "add_tensor is the union")

    # columns: 0 float, 1 one-hot, 2 and 3 the simple CTRs of feature 7,
    # 4 the simple CTR of feature 9
    var kinds: List[Int] = [
        TREE_COL_FLOAT, TREE_COL_ONE_HOT, TREE_COL_SIMPLE_CTR,
        TREE_COL_SIMPLE_CTR, TREE_COL_SIMPLE_CTR,
    ]
    var sources: List[Int] = [-1, -1, 7, 7, 9]
    var cats: List[Int] = [7, 9, 11]
    var cards: List[Int] = [5, 6, 4]
    var plan = TreeCtrPlan(kinds.copy(), sources.copy(), cats.copy(), cards.copy(), 3)
    var t = TreeCtrTensor()
    _need(len(plan.candidates(t)) == 0, "no candidate from an empty tree")
    plan.track_split(t, 0, 4)
    var c1 = plan.candidates(t)
    _need(len(c1) == 3, "a split-only tensor is extended by every feature")
    plan.track_split(t, 3, 2)
    _need(t.has_cat(7) and len(t.splits) == 1, "a CTR split joins as its feature")
    var c2 = plan.candidates(t)
    _need(len(c2) == 2 and c2[0] == 1 and c2[1] == 2, "members are not candidates")
    plan.track_split(t, 4, 0)
    _need(t.complexity() == 3, "splits one, two features")
    _need(len(plan.candidates(t)) == 0, "no candidate at max_ctr_complexity")
    var off = TreeCtrPlan(kinds.copy(), sources.copy(), cats.copy(), cards.copy(), 1)
    var t1 = TreeCtrTensor()
    off.track_split(t1, 1, 0)
    _need(t1.splits[0].take_bin, "a one-hot split is an equality side")
    _need(len(off.candidates(t1)) == 0, "max_ctr_complexity 1 builds no combination")


def groups_and_columns() raises:
    var n = 257
    var a = fixture_codes(n, 5, 1)
    var b = fixture_codes(n, 7, 2)
    var side = fixture_codes(n, 2, 3)
    var extra = fixture_codes(n, 4, 4)
    var target = List[UInt8](capacity=n)
    var tcodes = fixture_codes(n, 2, 5)
    for r in range(n):
        target.append(UInt8(Int(tcodes[r])))

    var members = List[TreeU32Ptr]()
    members.append(_u32p(side))
    members.append(_u32p(a))
    var member_cards: List[Int] = [2, 5]
    var base = tree_ctr_groups_host(members, member_cards, n)
    for r in range(n):
        for s in range(r):
            var same_tuple = side[r] == side[s] and a[r] == a[s]
            var same_group = base.ids[r] == base.ids[s]
            if same_tuple != same_group:
                raise Error("tree CTR check: groups disagree with the member tuples")
    var seen = List[Bool](length=base.count, fill=False)
    for r in range(n):
        _need(Int(base.ids[r]) < base.count, "a group id is below the group count")
        seen[Int(base.ids[r])] = True
    for g in range(base.count):
        _need(seen[g], "group ids are dense")

    var configs = List[TCtrConfig]()
    configs.append(TCtrConfig(CTR_BORDERS, TPrior(Float32(0.0), Float32(1.0)), 0, 0))
    configs.append(TCtrConfig(CTR_BORDERS, TPrior(Float32(0.5), Float32(1.0)), 0, 0))
    configs.append(TCtrConfig(CTR_FEATURE_FREQ, TPrior(Float32(0.0), Float32(1.0)), 0, 1))
    var order = ctr_estimation_order_host(n, 1, UInt64(17))
    var position = List[Int](length=n, fill=0)
    for i in range(n):
        position[Int(order[i])] = i

    var cand = List[TreeU32Ptr]()
    cand.append(_u32p(b))
    cand.append(_u32p(extra))
    var cand_cards: List[Int] = [7, 4]
    var flat = tree_ctr_level_host(
        base, cand, cand_cards, _u32p(order),
        TreeU8Ptr(unsafe_from_address=Int(target.unsafe_ptr())), n, configs,
    )
    _need(len(flat) == 2 * 3 * n, "one column per candidate and config")
    for k in range(2):
        for r in range(n):
            var before = 0
            var hits = 0
            var count = 0
            for s in range(n):
                var same = side[r] == side[s] and a[r] == a[s]
                if k == 0:
                    same = same and b[r] == b[s]
                else:
                    same = same and extra[r] == extra[s]
                if not same:
                    continue
                count += 1
                if position[s] < position[r]:
                    before += 1
                    if target[s] > UInt8(0):
                        hits += 1
            var want0 = (Float32(hits) + Float32(0.0)) / (Float32(before) + Float32(1.0))
            var want1 = (Float32(hits) + Float32(0.5)) / (Float32(before) + Float32(1.0))
            var want2 = (Float32(count) + Float32(0.0)) / (Float32(n) + Float32(1.0))
            _need(flat[(k * 3 + 0) * n + r] == want0, "ordered Borders, prior 0")
            _need(flat[(k * 3 + 1) * n + r] == want1, "ordered Borders, prior 0.5")
            _need(flat[(k * 3 + 2) * n + r] == want2, "FeatureFreq")

    # the grid: Uniform over the extremes, strict comparison, constant -> 0.5
    var fp = TreeF32Ptr(unsafe_from_address=Int(flat.unsafe_ptr()))
    var borders = tree_ctr_uniform_borders_host(fp, n, 15)
    _need(len(borders) >= 1 and len(borders) <= 15, "at most the asked borders")
    for i in range(1, len(borders)):
        _need(borders[i - 1] < borders[i], "borders ascend")
    var bins = List[UInt32](length=n, fill=UInt32(0))
    tree_ctr_bins_into(fp, n, borders, _u32p(bins))
    for r in range(n):
        var want = 0
        for i in range(len(borders)):
            if flat[r] > borders[i]:
                want += 1
        _need(Int(bins[r]) == want, "a bin is the borders strictly exceeded")
    var flat_const = List[Float32](length=9, fill=Float32(0.25))
    var cb = tree_ctr_uniform_borders_host(
        TreeF32Ptr(unsafe_from_address=Int(flat_const.unsafe_ptr())), 9, 15
    )
    _need(len(cb) == 1 and cb[0] == Float32(0.5), "a constant column takes 0.5")
    # the lists above are read through raw pointers
    _ = len(a)
    _ = len(b)
    _ = len(side)
    _ = len(extra)
    _ = len(target)
    _ = len(order)
    _ = len(bins)


def run_checks() raises:
    tensors()
    groups_and_columns()
