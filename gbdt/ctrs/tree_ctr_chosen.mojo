# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""What the trees of a tree-CTR fit chose, and the model section built from
it: the ONE statement both columns use.

The device fit (`gbdt/methods/tree_ctr_search.mojo`) and the host column
(`gbdt/host/gbdt_oracle_tree_ctr.mojo`) each grow their trees their own way
(launches there, host passes here) and each record the same three things:

  a MAP ENTRY per combination first chosen (`TreeCtrMapEntry`): its members
  in the order the fit added them, its map, the learn pool's counts per
  group;
  an ACCEPTED entry per model column (`TreeCtrAccepted`): a map, a config,
  the estimation order, the borders it won under;
  per tree, the accepted entry of each persist slot it used.

`tree_ctr_members` names a combination's members, `tree_ctr_assemble` turns
the record into the model's section. Both are integer bookkeeping over model
structure (maps, columns, members): no row is touched here, so there is
nothing to split across workers, and the two columns arrive at the same
section because they call the same code.

GPU-free: no device import.
"""

from gbdt.ctrs.ctr import TCtrConfig
from gbdt.ctrs.tree_ctr import TreeCtrSplit, TreeCtrTensor
from gbdt.models.combination_map import (
    TCombinationCtrs,
    TCombinationMap,
    TCombinationMember,
    combination_cat_member,
    combination_ctr_tables_from_counts,
    combination_float_member,
    combination_one_hot_member,
)


struct TreeCtrMapEntry(Copyable, Movable):
    """One combination the trees chose: its members as the fit added them
    (the base tensor's categorical features, then its splits, then the
    candidate's categorical feature), its map, and the learn pool's counts
    per group (and per group and target class when a Borders config
    exists)."""

    var cats: List[Int]
    var splits: List[TreeCtrSplit]
    var candidate: Int
    var cmap: TCombinationMap
    var counts: List[Int]
    var histogram: List[Int]

    def __init__(
        out self,
        var cats: List[Int],
        var splits: List[TreeCtrSplit],
        candidate: Int,
        var cmap: TCombinationMap,
        var counts: List[Int],
        var histogram: List[Int],
    ):
        self.cats = cats^
        self.splits = splits^
        self.candidate = candidate
        self.cmap = cmap^
        self.counts = counts^
        self.histogram = histogram^

    def has_members(self, base: TreeCtrTensor, candidate: Int) -> Bool:
        if self.candidate != candidate:
            return False
        if len(self.cats) != len(base.cats):
            return False
        if len(self.splits) != len(base.splits):
            return False
        for i in range(len(self.cats)):  # small-loop(cats: members of one combination): compares member ids
            if self.cats[i] != base.cats[i]:
                return False
        for i in range(len(self.splits)):  # small-loop(splits: members of one combination): compares member ids
            ref a = self.splits[i]
            ref b = base.splits[i]
            if a.feature != b.feature or a.bin != b.bin or (
                a.take_bin != b.take_bin
            ):
                return False
        return True


struct TreeCtrAccepted(Copyable, Movable):
    """One tree-CTR model column: a map, a config, the estimation order its
    training column was taken in (0 for a permutation-independent config)
    and the borders it won under."""

    var map_ordinal: Int
    var config: Int
    var order: Int
    var borders: List[Float32]

    def __init__(
        out self,
        map_ordinal: Int,
        config: Int,
        order: Int,
        var borders: List[Float32],
    ):
        self.map_ordinal = map_ordinal
        self.config = config
        self.order = order
        self.borders = borders^


struct TreeCtrModelSection(Movable):
    """What `tree_ctr_assemble` hands the fit: the model's tree-CTR section,
    the borders of its columns in column order, and each accepted entry's
    model column."""

    var ctrs: TCombinationCtrs
    var borders: List[List[Float32]]
    var column_of: List[Int]

    def __init__(
        out self,
        var ctrs: TCombinationCtrs,
        var borders: List[List[Float32]],
        var column_of: List[Int],
    ):
        self.ctrs = ctrs^
        self.borders = borders^
        self.column_of = column_of^


def tree_ctr_find_map(
    maps: List[TreeCtrMapEntry], base: TreeCtrTensor, candidate: Int
) -> Int:
    """The entry of (base tensor, candidate raw feature), or -1."""
    for m in range(len(maps)):  # small-loop(maps: combinations the trees chose): finds one combination
        if maps[m].has_members(base, candidate):
            return m
    return -1


def tree_ctr_find_accepted(
    accepted: List[TreeCtrAccepted], map_ordinal: Int, config: Int, order: Int
) -> Int:
    """The model column entry of (map, config, order), or -1."""
    for i in range(len(accepted)):  # small-loop(accepted: tree-CTR model columns): finds one column
        ref e = accepted[i]
        if e.map_ordinal == map_ordinal and e.config == config and (
            e.order == order
        ):
            return i
    return -1


def tree_ctr_members(
    base: TreeCtrTensor,
    candidate: Int,
    column_raw: List[Int],
    column_borders: List[List[Float32]],
    column_nan_side: List[Int],
) raises -> List[TCombinationMember]:
    """The members of the combination (base tensor, candidate raw feature)
    in the order every builder adds them: the tensor's categorical features
    ascending, its splits ascending (a one-hot split as its category, a
    float split as the border VALUE and the side a NaN takes), then the
    candidate. `column_raw[c]` is the raw input feature layout column `c`
    reads (-1 for a CTR column)."""
    var members = List[TCombinationMember]()
    # driver-loop: over the tensor's categorical members
    for i in range(len(base.cats)):
        members.append(combination_cat_member(base.cats[i]))
    # driver-loop: over the tensor's split members
    for i in range(len(base.splits)):
        var split = base.splits[i]
        var column = split.feature
        if column < 0 or column >= len(column_raw):
            raise Error("tree CTR fit: a split member outside the base columns")
        var raw = column_raw[column]
        if raw < 0:
            raise Error("tree CTR fit: a split member on a CTR column")
        if split.take_bin:
            members.append(combination_one_hot_member(raw, split.bin))
        else:
            if split.bin < 0 or split.bin >= len(column_borders[column]):
                raise Error("tree CTR fit: a split member past its borders")
            members.append(
                combination_float_member(
                    raw, column_borders[column][split.bin],
                    column_nan_side[column],
                )
            )
    members.append(combination_cat_member(candidate))
    return members^


def tree_ctr_assemble(
    maps: List[TreeCtrMapEntry],
    accepted: List[TreeCtrAccepted],
    configs: List[TCtrConfig],
    n_base: Int,
    target_classes_count: Int,
    n_rows: Int,
) raises -> TreeCtrModelSection:
    """The model's tree-CTR section: the maps in the order they were first
    chosen, each followed by its columns, which fill the model's columns
    from `n_base` on."""
    var ctrs = TCombinationCtrs()
    var borders = List[List[Float32]]()
    var column_of = List[Int](length=len(accepted), fill=-1)
    var next = n_base
    # driver-loop: over the combinations the trees chose (model structure)
    for m in range(len(maps)):
        var cfgs = List[TCtrConfig]()
        var which = List[Int]()
        for a in range(len(accepted)):  # small-loop(accepted: tree-CTR model columns): groups the columns by map
            if accepted[a].map_ordinal == m:
                cfgs.append(configs[accepted[a].config])
                which.append(a)
        if len(which) == 0:
            raise Error("tree CTR fit: a combination without a column")
        ref entry = maps[m]
        var tables = combination_ctr_tables_from_counts(
            entry.counts, entry.histogram, entry.cmap.groups(), cfgs,
            target_classes_count, m, next, n_rows,
        )
        if len(tables) != len(which):
            raise Error("tree CTR fit: one table per chosen column")
        for i in range(len(which)):  # small-loop(which: columns of one map): records each column's place
            column_of[which[i]] = next + i
            borders.append(accepted[which[i]].borders.copy())
        next += len(which)
        ctrs.add(entry.cmap.copy(), tables^)
    return TreeCtrModelSection(ctrs^, borders^, column_of^)
