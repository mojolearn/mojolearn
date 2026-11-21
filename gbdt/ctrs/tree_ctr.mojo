# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Tree CTRs (feature combinations): the tensor a tree carries, the
candidates one level mints from it, and the host statement of their columns.

A tree CTR is the target statistic of a COMBINATION: the categorical
features and the binary splits already in the oblivious tree, crossed with
one more categorical feature. After every accepted level the searcher

  1. folds the winning split into the tree's tensor (`TreeCtrPlan.track_split`
     for a float, one-hot or simple-CTR column, `TreeCtrTensor.add_tensor`
     for a split on an earlier tree-CTR column): a float or one-hot split
     joins as its 0/1 side, a CTR split as the categorical features (and
     splits) its own tensor names;
  2. asks `TreeCtrPlan.candidates` which categorical features may extend it
     (complexity below `max_ctr_complexity`, the feature not already in);
  3. computes, per candidate, one column per tree-CTR config over the rows
     grouped by the extended tensor: Borders in the ordered way (each row
     sees the rows of its group that precede it in the permutation's learn
     order), FeatureFreq as the group's share of the pool.

Complexity is the reference's: categorical members plus ONE for all the
binary splits together.

THE TWO COLUMNS. The device computes the columns with the bin builder and
the calcers the simple CTRs already use (`gbdt/ctrs/tree_ctr_device.mojo`);
this file is the host column's statement of the same numbers and is what
`gbdt/checks/tree_ctr_check.mojo` and the device check compare against.
Every running sum is a count held in Float32 (exact below 2^24 rows of one
group) and the one rounding is the final Float32 divide, as in
`gbdt/host/gbdt_oracle_ctr.mojo`:

    FeatureFreq   (count(g) + prior_num) / (n + prior_denom)
    Borders       (before_hits(r) + prior_num) / (before(r) + prior_denom)

GROUPS. A group id is the rank of the row's member values among the distinct
member-value tuples present, built one member at a time
(`tree_ctr_refine_host`: key = code * groups + group, then the rank of the
key, so the member added last is the most significant). That is the device
builder's segment numbering: each member is a stable sort of all rows by its
code, so the segments come out ordered by (new code, old segment). Only
group EQUALITY reaches the columns; the shared numbering is what lets a
model table built on either side index the same entries.

PARALLEL PLAN. The candidates of one level are independent units:
`tree_ctr_level_host` runs one task per candidate across the host's workers,
each writing its own slice of the flat result. Inside a task the passes are
serial (the ordered pass is a running count along the permutation: serial by
nature within one group set). The base grouping, computed once per level,
splits its row passes across the workers; its distinct-key sort is the
standard library's serial sort.

GPU-free: no device import, so the host column and the bindings import it.
"""

from std.builtin.sort import sort
from core.host_parallel import host_parallelize
from core.host_predict_threads import (
    host_predict_chunk,
    host_predict_task_count_work,
)
from gbdt.ctrs.ctr import CTR_BORDERS, CTR_FEATURE_FREQ, TCtrConfig
from gbdt.ctrs.ctr_binarization import uniform_borders_from_range

#: what a search-layout column contributes to the tree's tensor when a
#: split lands on it
comptime TREE_COL_FLOAT = 0  # its 0/1 side, bin > border
comptime TREE_COL_ONE_HOT = 1  # its 0/1 side, bin == category
comptime TREE_COL_SIMPLE_CTR = 2  # its categorical feature
comptime TREE_COL_TENSOR = 3  # its whole tensor (the caller holds it)

comptime TreeU32Ptr = MutPointer[UInt32, MutUntrackedOrigin]
comptime TreeU64Ptr = MutPointer[UInt64, MutUntrackedOrigin]
comptime TreeF32Ptr = MutPointer[Float32, MutUntrackedOrigin]
comptime TreeU8Ptr = MutPointer[UInt8, MutUntrackedOrigin]


@fieldwise_init
struct TreeCtrSplit(Copyable, ImplicitlyCopyable, Movable):
    """One binary split of the tree as a tensor member: search-layout
    column, bin, and whether the side is equality (one-hot) or `bin >`."""

    var feature: Int
    var bin: Int
    var take_bin: Bool


def _split_less(a: TreeCtrSplit, b: TreeCtrSplit) -> Bool:
    if a.feature != b.feature:
        return a.feature < b.feature
    if a.bin != b.bin:
        return a.bin < b.bin
    return (not a.take_bin) and b.take_bin


def _split_same(a: TreeCtrSplit, b: TreeCtrSplit) -> Bool:
    return a.feature == b.feature and a.bin == b.bin and a.take_bin == b.take_bin


struct TreeCtrTensor(Copyable, Movable):
    """The tree's tensor: categorical features (raw input feature ids) and
    binary splits, each ascending and unique, so two tensors naming the
    same members are equal however they were built."""

    var cats: List[Int]
    var splits: List[TreeCtrSplit]

    def __init__(out self):
        self.cats = List[Int]()
        self.splits = List[TreeCtrSplit]()

    def add_cat(mut self, feature: Int):
        var at = 0
        while at < len(self.cats) and self.cats[at] < feature:
            at += 1
        if at < len(self.cats) and self.cats[at] == feature:
            return
        self.cats.append(feature)
        var j = len(self.cats) - 1
        while j > at:
            self.cats[j] = self.cats[j - 1]
            j -= 1
        self.cats[at] = feature

    def add_split(mut self, split: TreeCtrSplit):
        var at = 0
        while at < len(self.splits) and _split_less(self.splits[at], split):
            at += 1
        if at < len(self.splits) and _split_same(self.splits[at], split):
            return
        self.splits.append(split)
        var j = len(self.splits) - 1
        while j > at:
            self.splits[j] = self.splits[j - 1]
            j -= 1
        self.splits[at] = split

    def add_tensor(mut self, other: Self):
        """The union: a member the two share appears once."""
        for i in range(len(other.cats)):
            self.add_cat(other.cats[i])
        for i in range(len(other.splits)):
            self.add_split(other.splits[i])

    def has_cat(self, feature: Int) -> Bool:
        for i in range(len(self.cats)):
            if self.cats[i] == feature:
                return True
        return False

    def is_empty(self) -> Bool:
        return len(self.cats) == 0 and len(self.splits) == 0

    def complexity(self) -> Int:
        """Categorical members plus one for all the splits together."""
        return len(self.cats) + (1 if len(self.splits) > 0 else 0)

    def same(self, other: Self) -> Bool:
        if len(self.cats) != len(other.cats):
            return False
        if len(self.splits) != len(other.splits):
            return False
        for i in range(len(self.cats)):
            if self.cats[i] != other.cats[i]:
                return False
        for i in range(len(self.splits)):
            if not _split_same(self.splits[i], other.splits[i]):
                return False
        return True


struct TreeCtrPlan(Copyable, Movable):
    """What a fit knows before its first tree: what each search-layout
    column is, which categorical features feed CTRs (ascending raw feature
    ids) with their cardinalities, and `max_ctr_complexity`."""

    var column_kind: List[Int]
    #: the raw categorical feature of a TREE_COL_SIMPLE_CTR column, else -1
    var column_cat: List[Int]
    var cat_features: List[Int]
    var cat_cardinality: List[Int]
    var max_complexity: Int

    def __init__(
        out self,
        var column_kind: List[Int],
        var column_cat: List[Int],
        var cat_features: List[Int],
        var cat_cardinality: List[Int],
        max_complexity: Int,
    ) raises:
        if len(column_kind) != len(column_cat):
            raise Error("tree CTR plan: one kind and one source per column")
        if len(cat_features) != len(cat_cardinality):
            raise Error("tree CTR plan: one cardinality per categorical feature")
        for i in range(1, len(cat_features)):
            if cat_features[i - 1] >= cat_features[i]:
                raise Error("tree CTR plan: categorical features must ascend")
        if max_complexity < 1:
            raise Error("max_ctr_complexity must be at least 1")
        self.column_kind = column_kind^
        self.column_cat = column_cat^
        self.cat_features = cat_features^
        self.cat_cardinality = cat_cardinality^
        self.max_complexity = max_complexity

    def enabled(self) -> Bool:
        """Tree CTRs exist when a categorical feature feeds CTRs and
        combinations of more than one member are allowed."""
        return len(self.cat_features) > 0 and self.max_complexity > 1

    def track_split(
        self, mut tensor: TreeCtrTensor, column: Int, bin: Int
    ) raises:
        """Fold an accepted split on a float, one-hot or simple-CTR column
        into the tree's tensor. A split on a tree-CTR column is the
        caller's `tensor.add_tensor(that column's tensor)`."""
        if column < 0 or column >= len(self.column_kind):
            raise Error("tree CTR plan: split column out of range")
        var kind = self.column_kind[column]
        if kind == TREE_COL_FLOAT:
            tensor.add_split(TreeCtrSplit(column, bin, False))
        elif kind == TREE_COL_ONE_HOT:
            tensor.add_split(TreeCtrSplit(column, bin, True))
        elif kind == TREE_COL_SIMPLE_CTR:
            tensor.add_cat(self.column_cat[column])
        else:
            raise Error(
                "tree CTR plan: a split on a tree-CTR column joins through"
                " its tensor"
            )

    def candidates(self, base: TreeCtrTensor) -> List[Int]:
        """The ordinals (into `cat_features`) of the categorical features
        that extend `base` at this level, ascending. None for an empty base
        (a single feature is the simple CTR the fit already holds) and none
        once the base has reached `max_ctr_complexity`."""
        var out = List[Int]()
        if not self.enabled() or base.is_empty():
            return out^
        if base.complexity() >= self.max_complexity:
            return out^
        for k in range(len(self.cat_features)):
            if not base.has_cat(self.cat_features[k]):
                out.append(k)
        return out^


struct TreeCtrGroups(Copyable, Movable):
    """Every row's group id under a tensor, and how many groups there are."""

    var ids: List[UInt32]
    var count: Int

    def __init__(out self, var ids: List[UInt32], count: Int):
        self.ids = ids^
        self.count = count

    @staticmethod
    def whole(n_rows: Int) -> Self:
        """One group holding every row: the empty tensor."""
        return Self(List[UInt32](length=n_rows, fill=UInt32(0)), 1)


@always_inline
def _rank_of(sorted_keys: TreeU64Ptr, count: Int, key: UInt64) -> Int:
    var lo = 0
    var hi = count
    while lo < hi:
        var mid = (lo + hi) // 2
        if sorted_keys[mid] < key:
            lo = mid + 1
        else:
            hi = mid
    return lo


def tree_ctr_refine_host(
    base_ids: TreeU32Ptr,
    base_count: Int,
    codes: TreeU32Ptr,
    cardinality: Int,
    n_rows: Int,
    parallel: Bool,
) -> TreeCtrGroups:
    """Split every group of the base by one more member: the new id is the
    rank of `code * base_count + base` among the keys present (the new
    member most significant, the device builder's segment order). `codes[r]`
    is below `cardinality` (a dense categorical code, or a split's 0/1 side
    with cardinality 2). `parallel` splits the row passes across the
    workers (the caller is not itself inside a task); the distinct-key sort
    is serial either way."""
    var keys = List[UInt64](unsafe_uninit_length=max(n_rows, 1))
    var ids = List[UInt32](unsafe_uninit_length=max(n_rows, 1))
    var kp = TreeU64Ptr(unsafe_from_address=Int(keys.unsafe_ptr()))
    var ip = TreeU32Ptr(unsafe_from_address=Int(ids.unsafe_ptr()))
    var radix = UInt64(max(base_count, 1))
    var units = 1
    if parallel:
        units = max(1, host_predict_task_count_work(n_rows, n_rows * 2))
    var chunk = host_predict_chunk(n_rows, units)

    def fill(unit: Int) {imm chunk, imm n_rows, imm kp, imm base_ids, imm codes, imm radix}:
        for r in range(unit * chunk, min((unit + 1) * chunk, n_rows)):
            kp[r] = UInt64(codes[r]) * radix + UInt64(base_ids[r])

    host_parallelize(fill, units)

    var distinct = List[UInt64](unsafe_uninit_length=max(n_rows, 1))
    var dp = TreeU64Ptr(unsafe_from_address=Int(distinct.unsafe_ptr()))
    for r in range(n_rows):
        dp[r] = kp[r]
    var span = Span(unsafe_ptr=dp, length=n_rows)
    sort(span)
    var count = 0
    for r in range(n_rows):
        if r == 0 or dp[r] != dp[r - 1]:
            dp[count] = dp[r]
            count += 1
    var levels = 1
    var remaining = count
    while remaining > 1:
        levels += 1
        remaining = (remaining + 1) // 2
    var rank_units = 1
    if parallel:
        rank_units = max(1, host_predict_task_count_work(n_rows, n_rows * levels))
    var rank_chunk = host_predict_chunk(n_rows, rank_units)

    def rank(unit: Int) {imm rank_chunk, imm n_rows, imm kp, imm dp, imm ip, imm count}:
        for r in range(unit * rank_chunk, min((unit + 1) * rank_chunk, n_rows)):
            ip[r] = UInt32(_rank_of(dp, count, kp[r]))

    host_parallelize(rank, rank_units)
    # the scratch lists are read through raw pointers above
    _ = len(keys)
    _ = len(distinct)
    _ = cardinality
    if n_rows <= 0:
        return TreeCtrGroups(List[UInt32](), 0)
    return TreeCtrGroups(ids^, count)


def tree_ctr_groups_host(
    member_codes: List[TreeU32Ptr],
    member_cards: List[Int],
    n_rows: Int,
) raises -> TreeCtrGroups:
    """The groups of a tensor from its members' per-row codes (each
    categorical member's dense codes, each split's 0/1 side), one refine
    per member, row passes across the workers."""
    if len(member_codes) != len(member_cards):
        raise Error("tree CTR groups: one cardinality per member")
    var groups = TreeCtrGroups.whole(n_rows)
    for m in range(len(member_codes)):
        if member_cards[m] < 2:
            raise Error("tree CTR groups: a member needs at least two values")
        var bp = TreeU32Ptr(unsafe_from_address=Int(groups.ids.unsafe_ptr()))
        var next = tree_ctr_refine_host(
            bp, groups.count, member_codes[m], member_cards[m], n_rows, True
        )
        _ = len(groups.ids)
        groups = next^
    return groups^


def tree_ctr_borders_into(
    ids: TreeU32Ptr,
    group_count: Int,
    order: TreeU32Ptr,
    binarized_target: TreeU8Ptr,
    n_rows: Int,
    prior_num: Float32,
    prior_denom: Float32,
    param_id: Int,
    dst: TreeF32Ptr,
):
    """One Borders column, ordered: walking the permutation's learn order,
    `dst[row] = (hits + prior_num) / (before + prior_denom)` over the rows
    of `row`'s group already walked, a hit being a binarized target above
    `param_id`. Serial by nature: a running count along the order."""
    var before = List[Float32](length=max(group_count, 1), fill=Float32(0.0))
    var hits = List[Float32](length=max(group_count, 1), fill=Float32(0.0))
    var param = UInt32(param_id)
    for i in range(n_rows):
        var row = Int(order[i])
        var g = Int(ids[row])
        dst[row] = (hits[g] + prior_num) / (before[g] + prior_denom)
        before[g] += Float32(1.0)
        if UInt32(binarized_target[row]) > param:
            hits[g] += Float32(1.0)


def tree_ctr_freq_into(
    ids: TreeU32Ptr,
    group_count: Int,
    n_rows: Int,
    prior_num: Float32,
    prior_denom: Float32,
    dst: TreeF32Ptr,
):
    """One FeatureFreq column: `(rows in the group + prior_num) /
    (n + prior_denom)`, the integer count converted to Float32 once."""
    var counts = List[Int](length=max(group_count, 1), fill=0)
    for r in range(n_rows):
        counts[Int(ids[r])] += 1
    var total = Float32(n_rows)
    for r in range(n_rows):
        dst[r] = (Float32(counts[Int(ids[r])]) + prior_num) / (
            total + prior_denom
        )


def tree_ctr_check_configs(configs: List[TCtrConfig]) raises:
    """The tree-CTR config types this implementation computes."""
    for c in range(len(configs)):
        if configs[c].ctr_type != CTR_BORDERS and (
            configs[c].ctr_type != CTR_FEATURE_FREQ
        ):
            raise Error(
                "tree CTRs are computed for Borders and FeatureFreq configs"
                " only"
            )


def tree_ctr_level_host(
    base: TreeCtrGroups,
    candidate_codes: List[TreeU32Ptr],
    candidate_cards: List[Int],
    order: TreeU32Ptr,
    binarized_target: TreeU8Ptr,
    n_rows: Int,
    configs: List[TCtrConfig],
) raises -> List[Float32]:
    """Every candidate's columns for one level and one permutation, flat:
    candidate `k`, config `c`, row `r` at `(k * len(configs) + c) * n_rows
    + r`. `base` is the tree tensor's groups, `candidate_codes[k]` the dense
    codes of the categorical feature extending it. One task per candidate
    across the workers. `order` and `binarized_target` are read only when a
    Borders config is present."""
    tree_ctr_check_configs(configs)
    var n_cand = len(candidate_codes)
    if len(candidate_cards) != n_cand:
        raise Error("tree CTR level: one cardinality per candidate")
    for k in range(n_cand):
        if candidate_cards[k] < 2:
            raise Error("tree CTR level: a candidate needs at least two categories")
    var n_cfg = len(configs)
    var out = List[Float32](
        length=max(n_cand * n_cfg * n_rows, 1), fill=Float32(0.0)
    )
    if n_cand == 0 or n_cfg == 0 or n_rows <= 0:
        return out^
    var op = TreeF32Ptr(unsafe_from_address=Int(out.unsafe_ptr()))
    var bp = TreeU32Ptr(unsafe_from_address=Int(base.ids.unsafe_ptr()))
    var base_count = base.count
    # the launch words of every config, read by the tasks through pointers
    var cfg_type = List[Int](capacity=n_cfg)
    var cfg_param = List[Int](capacity=n_cfg)
    var cfg_num = List[Float32](capacity=n_cfg)
    var cfg_den = List[Float32](capacity=n_cfg)
    for c in range(n_cfg):
        cfg_type.append(configs[c].ctr_type)
        cfg_param.append(configs[c].param_id)
        cfg_num.append(configs[c].numerator_shift())
        cfg_den.append(configs[c].denumerator_shift())
    var tp = MutPointer[Int, MutUntrackedOrigin](
        unsafe_from_address=Int(cfg_type.unsafe_ptr())
    )
    var pp = MutPointer[Int, MutUntrackedOrigin](
        unsafe_from_address=Int(cfg_param.unsafe_ptr())
    )
    var np = TreeF32Ptr(unsafe_from_address=Int(cfg_num.unsafe_ptr()))
    var dnp = TreeF32Ptr(unsafe_from_address=Int(cfg_den.unsafe_ptr()))
    var codes_p = MutPointer[TreeU32Ptr, MutUntrackedOrigin](
        unsafe_from_address=Int(candidate_codes.unsafe_ptr())
    )
    var cards_p = MutPointer[Int, MutUntrackedOrigin](
        unsafe_from_address=Int(candidate_cards.unsafe_ptr())
    )

    def one(k: Int) {imm op, imm bp, imm base_count, imm tp, imm pp, imm np, imm dnp, imm codes_p, imm cards_p, imm order, imm binarized_target, imm n_rows, imm n_cfg}:
        var groups = tree_ctr_refine_host(
            bp, base_count, codes_p[k], cards_p[k], n_rows, False
        )
        var gp = TreeU32Ptr(unsafe_from_address=Int(groups.ids.unsafe_ptr()))
        for c in range(n_cfg):
            var dst = op + (k * n_cfg + c) * n_rows
            if tp[c] == CTR_BORDERS:
                tree_ctr_borders_into(
                    gp, groups.count, order, binarized_target, n_rows,
                    np[c], dnp[c], pp[c], dst,
                )
            else:
                tree_ctr_freq_into(
                    gp, groups.count, n_rows, np[c], dnp[c], dst
                )
        _ = len(groups.ids)

    host_parallelize(one, n_cand)
    # read through raw pointers by the tasks
    _ = len(cfg_type)
    _ = len(cfg_param)
    _ = len(cfg_num)
    _ = len(cfg_den)
    _ = len(base.ids)
    return out^


def tree_ctr_uniform_borders_host(
    values: TreeF32Ptr, n_rows: Int, border_count: Int
) -> List[Float32]:
    """A tree-CTR column's grid: Uniform over the column's extremes
    (`uniform_borders_from_range`, the arithmetic the device build feeds
    with its reduced extremes), a constant column taking the single 0.5."""
    var lo = values[0]
    var hi = values[0]
    for r in range(1, n_rows):
        var v = values[r]
        if v < lo:
            lo = v
        if v > hi:
            hi = v
    var borders = uniform_borders_from_range(lo, hi, border_count)
    if len(borders) == 0:
        borders.append(Float32(0.5))
    return borders^


def tree_ctr_bins_into(
    values: TreeF32Ptr,
    n_rows: Int,
    borders: List[Float32],
    dst: TreeU32Ptr,
):
    """`dst[r]` = the number of borders `values[r]` strictly exceeds, rows
    across the workers."""
    var nb = len(borders)
    var bp = TreeF32Ptr(unsafe_from_address=Int(borders.unsafe_ptr()))
    var units = max(1, host_predict_task_count_work(n_rows, n_rows * max(nb, 1)))
    var chunk = host_predict_chunk(n_rows, units)

    def fill(unit: Int) {imm chunk, imm n_rows, imm values, imm bp, imm nb, imm dst}:
        for r in range(unit * chunk, min((unit + 1) * chunk, n_rows)):
            var bin = 0
            for b in range(nb):
                if values[r] > bp[b]:
                    bin += 1
            dst[r] = UInt32(bin)

    host_parallelize(fill, units)
