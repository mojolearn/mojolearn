# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Tree CTRs (feature combinations) on the host column: the state a host fit
carries and what it does between the levels of one oblivious tree.

THE SAME ALGORITHM AS THE DEVICE FIT, statement for statement
(`gbdt/methods/tree_ctr_search.mojo`; read its module note first). The
layout is pinned the same way (the base columns, `max_depth` persist slots,
one candidate slot per (CTR-bearing categorical feature, tree-CTR config),
every slot at the grid's fold capacity), so the histogram cells, the winner
and its tie-break are the device's. What a launch does there, a row pass
across the host's workers does here:

    device                                   host (this file)
    launch_tree_slot_clear / write / copy    _slot_clear / _slot_write / _slot_copy
    launch_tree_split_bits                   _split_bits
    tree_ctr_base_builder                    tree_ctr_groups_host
    tree_ctr_candidate_columns_device        tree_ctr_level_host
    launch_tree_median_words                 _level_words (the same record,
                                             `tree_ctr_median_pair_into`)
    tree_ctr_median_from_words / Uniform     the same two functions
    launch_tree_ctr_bins                     tree_ctr_bins_into
    code_histogram_kernel                    _group_tally
    launch_combination_tuples                combination_map_from_groups
    the model's predict launches (eval)      expand_combination_columns

and what the trees chose is recorded and assembled by the shared
`gbdt/ctrs/tree_ctr_chosen.mojo`, so the model section is the device's.

Bits: integer counts held in Float32 and one Float32 divide per CTR value,
order statistics of an exact sort, comparisons for the bins and the split
sides. Nothing depends on a reduction order or on how the rows were split
across workers, so the columns, grids, bins, maps and tables are the
device's on every vendor.

PARALLEL PLAN. Rows are the independent units of every pass here (slot
writes, split sides, bins, the map's tuples, the tally), split across the
host's workers by `host_parallelize` with the worker count from
`host_predict_task_count_work`. The candidates of a level are the units of
`tree_ctr_level_host` (one task each) and the candidate columns the units of
`_level_words` (one sort each). The permutations are the units of a won
Borders column's recompute. `_group_tally` holds one dense tally per worker
and takes fewer workers when the tallies together would pass four words per
row. Serial by nature: the levels of one tree (a level's candidates depend
on the split before it), the running count along a permutation inside one
ordered column, and the sort inside one column. The loops over members,
candidates, configs and permutations are loops over model structure.

HOST ONLY: no device import. `gbdt/host/gbdt_oracle.mojo::gbdt_host_boost`
drives it (`_tree_ctr_levels_host`), `gbdt/host/gbdt_oracle_ctr.mojo` builds
it and writes the model.
"""

from std.builtin.sort import sort
from core.host_parallel import host_parallelize
from core.host_predict_threads import (
    host_predict_chunk,
    host_predict_task_count_work,
)
from gbdt.ctrs.ctr import CTR_BORDERS, CTR_FEATURE_FREQ, TCtrConfig
from gbdt.ctrs.ctr_binarization import uniform_borders_from_range
from gbdt.ctrs.tree_ctr import (
    TreeCtrGroups,
    TreeCtrPlan,
    TreeCtrTensor,
    TreeF32Any,
    TreeF32Ptr,
    TreeU32Ptr,
    TreeU8Ptr,
    tree_ctr_bins_into,
    tree_ctr_borders_into,
    tree_ctr_check_configs,
    tree_ctr_groups_host,
    tree_ctr_level_host,
    tree_ctr_median_from_words,
    tree_ctr_median_pair_into,
    tree_ctr_median_words,
    tree_ctr_refine_host,
)
from gbdt.ctrs.tree_ctr_chosen import (
    TreeCtrAccepted,
    TreeCtrMapEntry,
    TreeCtrModelSection,
    tree_ctr_assemble,
    tree_ctr_find_accepted,
    tree_ctr_find_map,
    tree_ctr_members,
)
from gbdt.gpu_data.compressed_index_builder import CompressedIndexLayout
from gbdt.models.combination_map import (
    CombF32Ptr,
    CombU32Ptr,
    TCombinationCtrs,
    combination_ctr_tables_from_counts,
    combination_map_from_groups,
    expand_combination_columns,
)

comptime TreeIntPtr = MutPointer[Int, MutUntrackedOrigin]


def _u32_ptr(values: List[UInt32]) -> TreeU32Ptr:
    return TreeU32Ptr(unsafe_from_address=Int(values.unsafe_ptr()))


def _f32_ptr(values: List[Float32]) -> TreeF32Ptr:
    return TreeF32Ptr(unsafe_from_address=Int(values.unsafe_ptr()))


def _slot_write(
    cindex: TreeU32Ptr,
    base: Int,
    mask: UInt32,
    shift: UInt32,
    bins: TreeU32Ptr,
    n_rows: Int,
):
    """Per row, one feature's bits of its compressed-index word replaced by
    `bins[row]` (`tree_slot_write_kernel`). One row owns one word, so the
    tasks never meet; the neighbouring features of the word are kept."""
    var units = max(1, host_predict_task_count_work(n_rows, n_rows))
    var chunk = host_predict_chunk(n_rows, units)
    var field = mask << shift

    def rows(unit: Int) {imm chunk, imm n_rows, imm cindex, imm base, imm mask, imm shift, imm field, imm bins}:
        for r in range(unit * chunk, min((unit + 1) * chunk, n_rows)):
            cindex[base + r] = (cindex[base + r] & ~field) | (
                (bins[r] & mask) << shift
            )

    host_parallelize(rows, units)


def _slot_clear(
    cindex: TreeU32Ptr, base: Int, mask: UInt32, shift: UInt32, n_rows: Int
):
    """Per row, one feature's bits set to bin 0 (`tree_slot_clear_kernel`)."""
    var units = max(1, host_predict_task_count_work(n_rows, n_rows))
    var chunk = host_predict_chunk(n_rows, units)
    var field = mask << shift

    def rows(unit: Int) {imm chunk, imm n_rows, imm cindex, imm base, imm field}:
        for r in range(unit * chunk, min((unit + 1) * chunk, n_rows)):
            cindex[base + r] = cindex[base + r] & ~field

    host_parallelize(rows, units)


def _slot_copy(
    src: TreeU32Ptr,
    src_base: Int,
    src_mask: UInt32,
    src_shift: UInt32,
    dst: TreeU32Ptr,
    dst_base: Int,
    dst_mask: UInt32,
    dst_shift: UInt32,
    n_rows: Int,
):
    """Per row, one feature's bin read off one index and written into a
    feature of the same or another index (`tree_slot_copy_kernel`)."""
    var units = max(1, host_predict_task_count_work(n_rows, n_rows))
    var chunk = host_predict_chunk(n_rows, units)
    var field = dst_mask << dst_shift

    def rows(unit: Int) {imm chunk, imm n_rows, imm src, imm src_base, imm src_mask, imm src_shift, imm dst, imm dst_base, imm dst_mask, imm dst_shift, imm field}:
        for r in range(unit * chunk, min((unit + 1) * chunk, n_rows)):
            var bin = (src[src_base + r] >> src_shift) & src_mask
            dst[dst_base + r] = (dst[dst_base + r] & ~field) | (
                (bin & dst_mask) << dst_shift
            )

    host_parallelize(rows, units)


def _split_bits(
    cindex: TreeU32Ptr,
    base: Int,
    mask: UInt32,
    shift: UInt32,
    bin: UInt32,
    take_bin: Bool,
    dst: TreeU32Ptr,
    n_rows: Int,
):
    """Per row, the 0/1 side of one binary split read off the compressed
    index (`tree_split_bits_kernel`): `bin == split bin` for a one-hot
    split, `bin > split bin` otherwise."""
    var units = max(1, host_predict_task_count_work(n_rows, n_rows))
    var chunk = host_predict_chunk(n_rows, units)

    def rows(unit: Int) {imm chunk, imm n_rows, imm cindex, imm base, imm mask, imm shift, imm bin, imm take_bin, imm dst}:
        for r in range(unit * chunk, min((unit + 1) * chunk, n_rows)):
            var value = (cindex[base + r] >> shift) & mask
            var side = UInt32(0)
            if take_bin:
                if value == bin:
                    side = UInt32(1)
            else:
                if value > bin:
                    side = UInt32(1)
            dst[r] = side

    host_parallelize(rows, units)


def _level_words(
    columns: TreeF32Ptr,
    n_columns: Int,
    n_rows: Int,
    asked: List[Int],
    words_per_column: Int,
    mut words: List[Float32],
):
    """Every candidate column's words record for one level
    (`launch_tree_median_words`): the column's minimum and maximum, then for
    each of `asked[c]` borders the pair `tree_ctr_median_pair_into` reads
    off the column sorted ascending. `asked` is per config (column `u` is
    config `u % len(asked)`); 0 asks for the two extremes only and takes no
    sort. One task per column; the sort inside a column is serial by
    nature."""
    var n_cfg = len(asked)
    var ap = TreeIntPtr(unsafe_from_address=Int(asked.unsafe_ptr()))
    var wp = TreeF32Any(unsafe_from_address=Int(words.unsafe_ptr()))

    def one(u: Int) {imm columns, imm n_rows, imm ap, imm n_cfg, imm wp, imm words_per_column}:
        var col = columns + u * n_rows
        var base = u * words_per_column
        var want = ap[u % n_cfg]
        if want <= 0:
            var lo = col[0]
            var hi = col[0]
            for r in range(1, n_rows):
                var v = col[r]
                if v < lo:
                    lo = v
                if v > hi:
                    hi = v
            wp.unsafe_store(base, lo)
            wp.unsafe_store(base + 1, hi)
        else:
            var sorted = List[Float32](unsafe_uninit_length=max(n_rows, 1))
            var sp = TreeF32Ptr(unsafe_from_address=Int(sorted.unsafe_ptr()))
            for r in range(n_rows):
                sp[r] = col[r]
            var span = Span(unsafe_ptr=sp, length=n_rows)
            sort(span)
            wp.unsafe_store(base, sp[0])
            wp.unsafe_store(base + 1, sp[n_rows - 1])
            var sa = TreeF32Any(unsafe_from_address=Int(sorted.unsafe_ptr()))
            for i in range(want):
                tree_ctr_median_pair_into(sa, n_rows, want, i, wp, base)
            _ = len(sorted)

    host_parallelize(one, n_columns)
    _ = len(asked)
    _ = len(words)


def _group_tally(
    ids: TreeU32Ptr,
    n_groups: Int,
    binarized_target: TreeU8Ptr,
    n_rows: Int,
    classes: Int,
    with_histogram: Bool,
    mut counts: List[Int],
    mut histogram: List[Int],
):
    """The learn pool's rows per group, and per group and target class when
    `with_histogram` (`code_histogram_kernel`'s two tallies). Integer
    counts, so the split of the rows across workers does not reach them.
    Each worker tallies its rows into its own dense record and the records
    are added per group; the worker count is cut so that the records
    together stay within four words per row."""
    var width = 1 + (classes if with_histogram else 0)
    var per_task = max(n_groups, 1) * width
    var units = max(1, host_predict_task_count_work(n_rows, n_rows))
    var cap = max(1, (4 * n_rows) // per_task)
    if units > cap:
        units = cap
    var chunk = host_predict_chunk(n_rows, units)
    var scratch = List[Int](length=units * per_task, fill=0)
    var sp = TreeIntPtr(unsafe_from_address=Int(scratch.unsafe_ptr()))

    def rows(unit: Int) {imm chunk, imm n_rows, imm ids, imm binarized_target, imm sp, imm per_task, imm width, imm with_histogram}:
        var base = unit * per_task
        for r in range(unit * chunk, min((unit + 1) * chunk, n_rows)):
            var at = base + Int(ids[r]) * width
            sp[at] = sp[at] + 1
            if with_histogram:
                var cell = at + 1 + Int(binarized_target[r])
                sp[cell] = sp[cell] + 1

    host_parallelize(rows, units)

    counts.clear()
    counts.resize(max(n_groups, 1), 0)
    histogram.clear()
    if with_histogram:
        histogram.resize(max(n_groups, 1) * classes, 0)
    var cp = TreeIntPtr(unsafe_from_address=Int(counts.unsafe_ptr()))
    # an empty histogram is never written (`with_histogram` guards it)
    var hp = cp
    if with_histogram:
        hp = TreeIntPtr(unsafe_from_address=Int(histogram.unsafe_ptr()))
    var g_units = max(1, host_predict_task_count_work(n_groups, units * per_task))
    var g_chunk = host_predict_chunk(n_groups, g_units)

    def merge(unit: Int) {imm g_chunk, imm n_groups, imm units, imm per_task, imm width, imm sp, imm cp, imm hp, imm classes, imm with_histogram}:
        for g in range(unit * g_chunk, min((unit + 1) * g_chunk, n_groups)):
            for u in range(units):
                var at = u * per_task + g * width
                cp[g] = cp[g] + sp[at]
                if with_histogram:
                    for k in range(classes):
                        hp[g * classes + k] = hp[g * classes + k] + sp[at + 1 + k]

    host_parallelize(merge, g_units)
    _ = len(scratch)
    _ = len(counts)
    _ = len(histogram)


struct GbdtTreeCtrHost(Movable):
    """A host fit's tree-CTR state: `TreeCtrFit`'s fields with host lists
    where it holds device buffers. The compressed indexes are the boost
    loop's; every method that writes a slot takes them `mut`."""

    var plan: TreeCtrPlan
    var configs: List[TCtrConfig]
    #: borders asked of each config's grid
    var border_counts: List[Int]
    #: fold capacity of every slot (the largest `border_counts` entry)
    var fold_cap: Int
    var n_rows: Int
    #: base columns of the layout; persist slot j is column `n_base + j` and
    #: candidate slot s column `n_base + max_depth + s`
    var n_base: Int
    var max_depth: Int
    #: every layout column's word column, mask and shift
    var f_offset: List[Int]
    var f_mask: List[UInt32]
    var f_shift: List[UInt32]
    #: per base column: the raw input feature a float or one-hot column
    #: reads (-1 for a CTR column), its model borders and the side a NaN
    #: takes under a `bin >` split
    var column_raw: List[Int]
    var column_borders: List[List[Float32]]
    var column_nan_side: List[Int]
    #: dense codes of every CTR-bearing categorical feature, in
    #: `plan.cat_features` order
    var cat_codes: List[List[UInt32]]
    #: one CTR estimation order per permutation (or one for the whole fit)
    var orders: List[List[UInt32]]
    var btarget: List[UInt8]
    var target_classes_count: Int
    var has_borders: Bool
    # the held-out arm: the raw eval input, column-major
    var test_rows: Int
    var n_raw: Int
    var test_x: List[Float32]
    # the search index's slots that hold something other than bin 0, per
    # permutation
    var dirty_persist: List[List[Bool]]
    var dirty_cand: List[List[Bool]]
    #: set by any slot write on the search index; the level loop turns it
    #: into a full histogram rebuild and clears it
    var learn_changed: Bool
    # the level's candidates: the base tensor, its groups and its split
    # members' sides
    var level_base: TreeCtrTensor
    var level_groups: TreeCtrGroups
    var level_bits: List[List[UInt32]]
    var slot_live: List[Bool]
    var slot_borders: List[List[Float32]]
    # the tree being grown
    var tensor: TreeCtrTensor
    var persist_tensor: List[TreeCtrTensor]
    var persist_accept: List[Int]
    # what the trees chose
    var maps: List[TreeCtrMapEntry]
    var accepted: List[TreeCtrAccepted]
    #: per accepted entry, its bins on the test rows
    var test_bins: List[List[UInt32]]
    #: per tree, the accepted entry of each persist slot it used
    var tree_persist: List[List[Int]]

    def __init__(
        out self,
        var plan: TreeCtrPlan,
        var configs: List[TCtrConfig],
        var border_counts: List[Int],
        n_rows: Int,
        n_base: Int,
        max_depth: Int,
        layout: CompressedIndexLayout,
        var column_raw: List[Int],
        var column_borders: List[List[Float32]],
        var column_nan_side: List[Int],
        var cat_codes: List[List[UInt32]],
        var orders: List[List[UInt32]],
        var btarget: List[UInt8],
        target_classes_count: Int,
        permutation_count: Int,
        test_rows: Int,
        n_raw: Int,
        var test_x: List[Float32],
    ) raises:
        tree_ctr_check_configs(configs)
        if not plan.enabled() or len(configs) == 0:
            raise Error("tree CTR fit: no categorical feature or no config")
        if len(border_counts) != len(configs):
            raise Error("tree CTR fit: one border count per config")
        if len(cat_codes) != len(plan.cat_features):
            raise Error("tree CTR fit: one code column per categorical feature")
        if len(plan.column_kind) != n_base or len(column_raw) != n_base or (
            len(column_borders) != n_base or len(column_nan_side) != n_base
        ):
            raise Error("tree CTR fit: the base column plan is incomplete")
        if len(orders) < 1 or permutation_count < 1 or max_depth < 1:
            raise Error("tree CTR fit: needs an order, a permutation and a level")
        if n_rows < 1:
            raise Error("tree CTR fit: no rows")
        var n_slots = len(plan.cat_features) * len(configs)
        if len(layout.features) != n_base + max_depth + n_slots:
            raise Error(
                "tree CTR fit: the layout holds " + String(len(layout.features))
                + " columns for " + String(n_base) + " base columns, "
                + String(max_depth) + " persist slots and " + String(n_slots)
                + " candidate slots"
            )
        var cap = 1
        var borders_seen = False
        for c in range(len(configs)):  # small-loop(configs: tree-CTR configs): reads the handful of grids
            if border_counts[c] < 1:
                raise Error("tree CTR fit: a grid needs at least one border")
            if border_counts[c] > cap:
                cap = border_counts[c]
            if configs[c].ctr_type == CTR_BORDERS:
                borders_seen = True
        if borders_seen and target_classes_count < 2:
            raise Error(
                "a Borders combination CTR is a statistic of the target and"
                " needs the fit's binarized target"
            )
        if borders_seen and len(orders) != permutation_count:
            raise Error(
                "a Borders combination CTR needs one estimation order per"
                " permutation"
            )
        if borders_seen and len(btarget) != n_rows:
            raise Error("tree CTR fit: one binarized target per row")
        # driver-loop: over the estimation orders (permutations)
        for p in range(len(orders)):
            if len(orders[p]) != n_rows:
                raise Error("tree CTR fit: an estimation order of the wrong length")
        # driver-loop: over the CTR-bearing categorical features
        for k in range(len(cat_codes)):
            if len(cat_codes[k]) != n_rows:
                raise Error("tree CTR fit: a code column of the wrong length")
        var f_offset = List[Int]()
        var f_mask = List[UInt32]()
        var f_shift = List[UInt32]()
        # driver-loop: over the layout's columns
        for f in range(len(layout.features)):
            f_offset.append(Int(layout.features[f].offset))
            f_mask.append(layout.features[f].mask)
            f_shift.append(layout.features[f].shift)
        var dirty_persist = List[List[Bool]]()
        var dirty_cand = List[List[Bool]]()
        for _ in range(permutation_count):  # small-loop(permutation_count: permutations): one flag list per permutation
            dirty_persist.append(List[Bool](length=max_depth, fill=False))
            dirty_cand.append(List[Bool](length=n_slots, fill=False))
        var slot_borders = List[List[Float32]]()
        for _ in range(n_slots):  # small-loop(n_slots: candidate slots): one empty grid per slot
            slot_borders.append(List[Float32]())
        # the tasks read the target through a pointer: never an empty list
        if len(btarget) == 0:
            btarget.append(UInt8(0))
        self.fold_cap = cap
        self.has_borders = borders_seen
        self.n_rows = n_rows
        self.n_base = n_base
        self.max_depth = max_depth
        self.target_classes_count = target_classes_count
        self.test_rows = test_rows
        self.n_raw = n_raw
        self.f_offset = f_offset^
        self.f_mask = f_mask^
        self.f_shift = f_shift^
        self.dirty_persist = dirty_persist^
        self.dirty_cand = dirty_cand^
        self.learn_changed = False
        self.level_base = TreeCtrTensor()
        self.level_groups = TreeCtrGroups.whole(n_rows)
        self.level_bits = List[List[UInt32]]()
        self.slot_live = List[Bool](length=n_slots, fill=False)
        self.slot_borders = slot_borders^
        self.tensor = TreeCtrTensor()
        self.persist_tensor = List[TreeCtrTensor]()
        self.persist_accept = List[Int]()
        self.maps = List[TreeCtrMapEntry]()
        self.accepted = List[TreeCtrAccepted]()
        self.test_bins = List[List[UInt32]]()
        self.tree_persist = List[List[Int]]()
        self.plan = plan^
        self.configs = configs^
        self.border_counts = border_counts^
        self.column_raw = column_raw^
        self.column_borders = column_borders^
        self.column_nan_side = column_nan_side^
        self.cat_codes = cat_codes^
        self.orders = orders^
        self.btarget = btarget^
        self.test_x = test_x^

    def slot_count(self) -> Int:
        return len(self.plan.cat_features) * len(self.configs)

    def persist_column(self, j: Int) -> Int:
        return self.n_base + j

    def candidate_column(self, slot: Int) -> Int:
        return self.n_base + self.max_depth + slot

    def order_of(self, permutation: Int) -> Int:
        """The estimation order a permutation's columns are taken in."""
        if len(self.orders) == 1:
            return 0
        return permutation

    def _base(self, column: Int) -> Int:
        """Where a layout column's words start in a learn index."""
        return self.f_offset[column] * self.n_rows

    def begin_tree(mut self, learn_p: Int, mut cindex: List[UInt32]):
        """A new tree on permutation `learn_p`'s index: every slot an
        earlier tree left holding a column goes back to bin 0, and the
        tree's tensor starts empty."""
        var cp = _u32_ptr(cindex)
        # driver-loop: over the persist slots (tree levels)
        for j in range(self.max_depth):
            if self.dirty_persist[learn_p][j]:
                var column = self.persist_column(j)
                _slot_clear(
                    cp, self._base(column), self.f_mask[column],
                    self.f_shift[column], self.n_rows,
                )
                self.dirty_persist[learn_p][j] = False
        # driver-loop: over the candidate slots (features x configs)
        for s in range(self.slot_count()):
            if self.dirty_cand[learn_p][s]:
                var column = self.candidate_column(s)
                _slot_clear(
                    cp, self._base(column), self.f_mask[column],
                    self.f_shift[column], self.n_rows,
                )
                self.dirty_cand[learn_p][s] = False
            self.slot_live[s] = False
        _ = len(cindex)
        self.learn_changed = True
        self.tensor = TreeCtrTensor()
        self.level_base = TreeCtrTensor()
        self.persist_tensor.clear()
        self.persist_accept.clear()

    def end_tree(mut self):
        """Record which accepted entry each persist slot of the finished
        tree holds (`rewrite_splits` reads it by tree index)."""
        self.tree_persist.append(self.persist_accept.copy())

    def _cat_ordinal(self, feature: Int) raises -> Int:
        for k in range(len(self.plan.cat_features)):  # small-loop(cat_features: CTR-bearing categorical features): finds one feature's ordinal
            if self.plan.cat_features[k] == feature:
                return k
        raise Error(
            "tree CTR fit: categorical feature " + String(feature)
            + " feeds no CTR"
        )

    def _member_pointers(
        self, mut codes: List[TreeU32Ptr], mut cards: List[Int]
    ) raises:
        """The level's base tensor's members as per-row code columns, in the
        order every builder adds them: the categorical features ascending,
        then the splits ascending (their sides are in `level_bits`)."""
        # driver-loop: over the tensor's categorical members
        for i in range(len(self.level_base.cats)):
            var k = self._cat_ordinal(self.level_base.cats[i])
            codes.append(_u32_ptr(self.cat_codes[k]))
            cards.append(self.plan.cat_cardinality[k])
        if len(self.level_base.splits) != len(self.level_bits):
            raise Error("tree CTR fit: the split sides of another level")
        # driver-loop: over the tensor's split members
        for i in range(len(self.level_base.splits)):
            codes.append(_u32_ptr(self.level_bits[i]))
            cards.append(2)

    def mint_level(mut self, learn_p: Int, mut cindex: List[UInt32]) raises:
        """The next level's candidates into the candidate slots of the
        search index: one column per (candidate feature, config) over the
        rows grouped by the tree's tensor extended by that feature, binned
        under its own grid. A slot no candidate holds any more goes back to
        bin 0."""
        var n = self.n_rows
        var n_cfg = len(self.configs)
        var n_slots = self.slot_count()
        var cands = self.plan.candidates(self.tensor)
        self.level_base = self.tensor.copy()
        var live = List[Bool](length=n_slots, fill=False)
        var cp = _u32_ptr(cindex)
        if len(cands) > 0:
            # every split member's 0/1 side, off the search index (float and
            # one-hot columns are the same on every permutation)
            self.level_bits.clear()
            # driver-loop: over the tensor's split members
            for i in range(len(self.level_base.splits)):
                var split = self.level_base.splits[i]
                var bits = List[UInt32](length=n, fill=UInt32(0))
                _split_bits(
                    cp, self._base(split.feature), self.f_mask[split.feature],
                    self.f_shift[split.feature], UInt32(split.bin),
                    split.take_bin, _u32_ptr(bits), n,
                )
                self.level_bits.append(bits^)
            var codes = List[TreeU32Ptr]()
            var cards = List[Int]()
            self._member_pointers(codes, cards)
            self.level_groups = tree_ctr_groups_host(codes, cards, n)
            var cand_codes = List[TreeU32Ptr]()
            var cand_cards = List[Int]()
            # driver-loop: over the level's candidate features
            for i in range(len(cands)):
                cand_codes.append(_u32_ptr(self.cat_codes[cands[i]]))
                cand_cards.append(self.plan.cat_cardinality[cands[i]])
            var columns = tree_ctr_level_host(
                self.level_groups, cand_codes, cand_cards,
                _u32_ptr(self.orders[self.order_of(learn_p)]),
                TreeU8Ptr(unsafe_from_address=Int(self.btarget.unsafe_ptr())),
                n, self.configs,
            )
            var colp = _f32_ptr(columns)
            # Median needs every border's pair; Uniform reads the record's
            # two extremes only
            var asked = List[Int]()
            for c in range(n_cfg):  # small-loop(configs: tree-CTR configs): one word count per config
                if self.configs[c].ctr_type == CTR_FEATURE_FREQ:
                    asked.append(self.border_counts[c])
                else:
                    asked.append(0)
            var n_columns = len(cands) * n_cfg
            var wps = tree_ctr_median_words(self.fold_cap)
            var words = List[Float32](length=n_columns * wps, fill=Float32(0.0))
            _level_words(colp, n_columns, n, asked, wps, words)
            var bins = List[UInt32](length=n, fill=UInt32(0))
            # driver-loop: over the level's candidate columns
            for u in range(n_columns):
                var c = u % n_cfg
                var slot = cands[u // n_cfg] * n_cfg + c
                var at = u * wps
                var grid = List[Float32]()
                if self.configs[c].ctr_type == CTR_FEATURE_FREQ:
                    var record = List[Float32](capacity=wps)
                    for w in range(tree_ctr_median_words(self.border_counts[c])):  # small-loop(words: order statistics of one grid, at most 2 + 2 * 255): one column's record
                        record.append(words[at + w])
                    grid = tree_ctr_median_from_words(
                        record, 0, self.border_counts[c]
                    )
                else:
                    grid = uniform_borders_from_range(
                        words[at], words[at + 1], self.border_counts[c]
                    )
                    if len(grid) == 0:
                        grid.append(Float32(0.5))
                if len(grid) > self.fold_cap:
                    raise Error("tree CTR fit: a grid exceeds its slot")
                tree_ctr_bins_into(colp + u * n, n, grid, _u32_ptr(bins))
                var column = self.candidate_column(slot)
                _slot_write(
                    cp, self._base(column), self.f_mask[column],
                    self.f_shift[column], _u32_ptr(bins), n,
                )
                self.slot_borders[slot] = grid^
                live[slot] = True
                self.dirty_cand[learn_p][slot] = True
                self.learn_changed = True
            # read through raw pointers above
            _ = len(columns)
            _ = len(bins)
            _ = len(codes)
            _ = len(cand_codes)
        # driver-loop: over the candidate slots (features x configs)
        for s in range(n_slots):
            if not live[s] and self.dirty_cand[learn_p][s]:
                var column = self.candidate_column(s)
                _slot_clear(
                    cp, self._base(column), self.f_mask[column],
                    self.f_shift[column], n,
                )
                self.dirty_cand[learn_p][s] = False
                self.learn_changed = True
            self.slot_live[s] = live[s]
        _ = len(cindex)

    def _new_map(mut self, k: Int, joined: TreeCtrGroups) raises -> Int:
        """The map entry of (the level's base tensor, candidate feature
        ordinal `k`), built the first time that combination is chosen, from
        the rows' groups under its members (`joined`): every group's member
        tuple and the learn pool's counts."""
        var n = self.n_rows
        var feature = self.plan.cat_features[k]
        var n_groups = joined.count
        if n_groups < 1 or n_groups > n:
            raise Error(
                "tree CTR fit: a combination of " + String(n_groups)
                + " groups over " + String(n) + " rows"
            )
        var member_codes = List[CombU32Ptr]()
        # driver-loop: over the tensor's categorical members
        for i in range(len(self.level_base.cats)):
            var ck = self._cat_ordinal(self.level_base.cats[i])
            member_codes.append(
                CombU32Ptr(unsafe_from_address=Int(self.cat_codes[ck].unsafe_ptr()))
            )
        # driver-loop: over the tensor's split members
        for i in range(len(self.level_base.splits)):
            member_codes.append(
                CombU32Ptr(unsafe_from_address=Int(self.level_bits[i].unsafe_ptr()))
            )
        member_codes.append(
            CombU32Ptr(unsafe_from_address=Int(self.cat_codes[k].unsafe_ptr()))
        )
        var members = tree_ctr_members(
            self.level_base, feature, self.column_raw, self.column_borders,
            self.column_nan_side,
        )
        var cmap = combination_map_from_groups(
            members^, member_codes,
            CombU32Ptr(unsafe_from_address=Int(joined.ids.unsafe_ptr())),
            n_groups, n,
        )
        var counts = List[Int]()
        var histogram = List[Int]()
        _group_tally(
            _u32_ptr(joined.ids), n_groups,
            TreeU8Ptr(unsafe_from_address=Int(self.btarget.unsafe_ptr())),
            n, self.target_classes_count, self.has_borders, counts, histogram,
        )
        _ = len(member_codes)
        self.maps.append(
            TreeCtrMapEntry(
                self.level_base.cats.copy(), self.level_base.splits.copy(),
                feature, cmap^, counts^, histogram^,
            )
        )
        return len(self.maps) - 1

    def _test_bins_for(
        self, map_ordinal: Int, config: Int, slot: Int
    ) raises -> List[UInt32]:
        """A chosen column on the held-out rows: the derived code of every
        row under the map (an unseen tuple takes the prior), the table's
        value, the bin under the column's borders: the model's own predict
        statement over the raw eval input."""
        var one = List[TCtrConfig]()
        one.append(self.configs[config])
        var cmap = self.maps[map_ordinal].cmap.copy()
        var tables = combination_ctr_tables_from_counts(
            self.maps[map_ordinal].counts, self.maps[map_ordinal].histogram,
            cmap.groups(), one, self.target_classes_count, 0, 0, self.n_rows,
        )
        var ctrs = TCombinationCtrs()
        ctrs.add(cmap^, tables^)
        var values = List[Float32](length=self.test_rows, fill=Float32(0.0))
        expand_combination_columns(
            ctrs,
            CombF32Ptr(unsafe_from_address=Int(self.test_x.unsafe_ptr())),
            self.test_rows, self.n_raw, False, 1,
            CombF32Ptr(unsafe_from_address=Int(values.unsafe_ptr())),
        )
        var bins = List[UInt32](length=self.test_rows, fill=UInt32(0))
        tree_ctr_bins_into(
            _f32_ptr(values), self.test_rows, self.slot_borders[slot],
            _u32_ptr(bins),
        )
        _ = len(values)
        return bins^

    def take_winner(
        mut self,
        learn_p: Int,
        mut cindexes: List[List[UInt32]],
        mut test_cindex: List[UInt32],
        feature: Int,
        bin: Int,
    ) raises -> Int:
        """The accepted winner of a level (layout column `feature`, bin
        `bin`) into the tree's tensor. Returns the column the tree records
        for the split: the winner's own for a base column or a persist
        slot; for a won candidate, the tree's next persist slot, which this
        call fills on every permutation's index and on the test index. -1
        when the winner sits on a slot that holds no column (bin 0 for every
        row: no split), which ends the tree as a level without an
        admissible split does."""
        var n = self.n_rows
        if feature < self.n_base:
            var grown = self.tensor.copy()
            self.plan.track_split(grown, feature, bin)
            self.tensor = grown^
            return feature
        if feature < self.n_base + self.max_depth:
            var held = feature - self.n_base
            if held >= len(self.persist_tensor):
                return -1
            var other = self.persist_tensor[held].copy()
            self.tensor.add_tensor(other)
            return feature
        var slot = feature - self.n_base - self.max_depth
        if slot >= self.slot_count() or not self.slot_live[slot]:
            return -1
        if bin < 0 or bin >= len(self.slot_borders[slot]):
            return -1
        var n_cfg = len(self.configs)
        var k = slot // n_cfg
        var c = slot % n_cfg
        var dependent = self.configs[c].ctr_type == CTR_BORDERS
        var order = self.order_of(learn_p) if dependent else 0
        var raw_feature = self.plan.cat_features[k]

        # the rows' groups under (base tensor, candidate): the map's ids
        # and the groups of every other permutation's ordered column
        var joined = tree_ctr_refine_host(
            _u32_ptr(self.level_groups.ids), self.level_groups.count,
            _u32_ptr(self.cat_codes[k]), self.plan.cat_cardinality[k], n, True,
        )
        var m = tree_ctr_find_map(self.maps, self.level_base, raw_feature)
        if m < 0:
            m = self._new_map(k, joined)
        var a = tree_ctr_find_accepted(self.accepted, m, c, order)
        if a < 0:
            self.accepted.append(
                TreeCtrAccepted(m, c, order, self.slot_borders[slot].copy())
            )
            a = len(self.accepted) - 1
            if self.test_rows > 0:
                self.test_bins.append(self._test_bins_for(m, c, slot))

        var j = len(self.persist_tensor)
        if j >= self.max_depth:
            raise Error("tree CTR fit: more persist columns than tree levels")
        var persist = self.persist_column(j)
        var source = self.candidate_column(slot)
        var p_base = self._base(persist)
        var p_mask = self.f_mask[persist]
        var p_shift = self.f_shift[persist]
        var s_base = self._base(source)
        var s_mask = self.f_mask[source]
        var s_shift = self.f_shift[source]
        var learn = _u32_ptr(cindexes[learn_p])
        # the search index first: the other permutations copy from its
        # candidate slot, which stays as it is until the next mint
        _slot_copy(
            learn, s_base, s_mask, s_shift, learn, p_base, p_mask, p_shift, n
        )
        self.dirty_persist[learn_p][j] = True
        self.learn_changed = True
        # the permutations whose own ordered statistic the column needs
        var recompute = List[Int]()
        # driver-loop: over the permutations
        for p in range(len(cindexes)):
            if p == learn_p:
                continue
            if dependent and self.order_of(p) != order:
                recompute.append(p)
            else:
                _slot_copy(
                    learn, s_base, s_mask, s_shift, _u32_ptr(cindexes[p]),
                    p_base, p_mask, p_shift, n,
                )
            self.dirty_persist[p][j] = True
        if len(recompute) > 0:
            # one ordered column per permutation, the permutations across
            # the workers; each is binned under the search permutation's
            # borders
            var n_re = len(recompute)
            var flat = List[Float32](length=n_re * n, fill=Float32(0.0))
            var fp = _f32_ptr(flat)
            var order_list = List[TreeU32Ptr]()
            for i in range(n_re):  # small-loop(recompute: permutations): one order pointer per permutation
                order_list.append(_u32_ptr(self.orders[self.order_of(recompute[i])]))
            var op = MutPointer[TreeU32Ptr, MutUntrackedOrigin](
                unsafe_from_address=Int(order_list.unsafe_ptr())
            )
            var jp = _u32_ptr(joined.ids)
            var j_count = joined.count
            var bt = TreeU8Ptr(unsafe_from_address=Int(self.btarget.unsafe_ptr()))
            var num = self.configs[c].numerator_shift()
            var den = self.configs[c].denumerator_shift()
            var param = self.configs[c].param_id

            def ordered(i: Int) {imm jp, imm j_count, imm op, imm bt, imm n, imm num, imm den, imm param, imm fp}:
                tree_ctr_borders_into(
                    jp, j_count, op[i], bt, n, num, den, param, fp + i * n
                )

            host_parallelize(ordered, n_re)
            var bins = List[UInt32](length=n, fill=UInt32(0))
            # driver-loop: over the recomputed permutations
            for i in range(n_re):
                tree_ctr_bins_into(
                    fp + i * n, n, self.slot_borders[slot], _u32_ptr(bins)
                )
                _slot_write(
                    _u32_ptr(cindexes[recompute[i]]), p_base, p_mask, p_shift,
                    _u32_ptr(bins), n,
                )
            _ = len(flat)
            _ = len(order_list)
            _ = len(bins)
        if self.test_rows > 0:
            _slot_write(
                _u32_ptr(test_cindex), self.f_offset[persist] * self.test_rows,
                p_mask, p_shift, _u32_ptr(self.test_bins[a]), self.test_rows,
            )
        _ = len(joined.ids)
        _ = len(cindexes)
        _ = len(test_cindex)

        var joined_tensor = self.level_base.copy()
        joined_tensor.add_cat(raw_feature)
        self.tensor.add_tensor(joined_tensor)
        self.persist_tensor.append(joined_tensor^)
        self.persist_accept.append(a)
        return persist

    def assemble(self) raises -> TreeCtrModelSection:
        """The model's tree-CTR section (`tree_ctr_assemble`, the device
        fit's statement)."""
        return tree_ctr_assemble(
            self.maps, self.accepted, self.configs, self.n_base,
            self.target_classes_count, self.n_rows,
        )

    def rewrite_splits(
        self,
        tree_split_offsets: List[Int],
        mut split_features: List[Int],
        column_of: List[Int],
    ) raises:
        """Every tree's persist-slot splits renumbered to their model
        columns (the fit grew them on layout slots that the next tree
        reuses). `tree_split_offsets` and `split_features` are the flat
        host model's."""
        var n_trees = len(tree_split_offsets) - 1
        if n_trees > len(self.tree_persist):
            raise Error("tree CTR fit: a tree without its persist record")
        # driver-loop: over the trees and their splits (model structure)
        for t in range(n_trees):
            for i in range(tree_split_offsets[t], tree_split_offsets[t + 1]):  # small-loop(splits: tree depth): renumbers one tree's splits
                var feature = split_features[i]
                if feature < self.n_base:
                    continue
                var j = feature - self.n_base
                if j >= len(self.tree_persist[t]):
                    raise Error("tree CTR fit: a split on an unheld slot")
                split_features[i] = column_of[self.tree_persist[t][j]]
