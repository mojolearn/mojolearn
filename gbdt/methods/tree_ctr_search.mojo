# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The fit arm of tree CTRs (feature combinations): one oblivious tree grown
level by level, the candidates of each level minted from the splits already
in the tree.

WHO TAKES THIS ARM. Only a fit with tree CTRs on (`max_ctr_complexity` above
1 and a categorical feature that feeds CTRs). Every other fit keeps the
fused per-tree call (`run_tree_layout_traced`), launch for launch.

THE LAYOUT IS PINNED FOR THE FIT. After the base columns it holds
`max_depth` PERSIST slots and one CANDIDATE slot per (CTR-bearing
categorical feature, tree-CTR config), every slot at the tree-CTR grid's
fold capacity. A slot nobody holds is bin 0 for every row and offers no
split.

ONE LEVEL.

  1. `search_synchronized_symmetric_level`: histograms over the whole
     layout, the scores, the winner, ONE drain, the gates. The candidates of
     the level are ordinary columns of that one launch set: no host loop
     over candidates or rows.
  2. `TreeCtrFit.take_winner`: a base column joins the tree's tensor
     (`TreeCtrPlan.track_split`); a persist slot joins through its own
     tensor; a won candidate slot is copied into the tree's next persist
     slot on every permutation's index and on the test index, its
     combination map and counts are read home once (first time that
     combination is chosen), and the split is rewritten to the persist slot.
  3. `enqueue_post_winner`: the partition, as the synchronized driver does.
  4. `TreeCtrFit.mint_level`: the next level's candidates on the search
     permutation: the rows ordered by the tree's tensor once
     (`tree_ctr_base_builder`), one column set per candidate
     (`tree_ctr_candidate_columns_device`), each column's order statistics
     into one words buffer read home ONCE for the level, the grids on the
     host (Median for FeatureFreq, Uniform over the extremes for Borders),
     then bins and slot writes on the device.

Then the leaf tail (`finish_synchronized_symmetric_tree`): the fused path's
own leaf kernels.

PERMUTATIONS. The search runs on the tree's learn permutation. A won
FeatureFreq column is the same on every permutation and is copied; a won
Borders column is recomputed on each other permutation in that
permutation's order and binned with the search permutation's borders, so
every permutation's estimation reads its own ordered statistic.

THE MODEL. A chosen combination is a model column: its map
(`gbdt/models/combination_map.mojo`), a table per config from counts tallied
on the device over the builder's per-row group ids, and the borders of the
grid it won under. `TreeCtrFit.assemble` builds the section after the fit
and `rewrite_model` renumbers the trees' persist-slot splits to the model
columns. The same (members, config, order) chosen again reuses its column.

PARALLEL PLAN. Every launch is one thread per row (or the builders' and
calcers' own grids). The host loops are over model structure: levels,
permutations, the tensor's members, the candidates and configs of a level.
Serial by nature: the levels of one tree (a level's candidates depend on
the split before it) and the one drain per level that reads its winner.

Bits: integer counts held in Float32 and one Float32 divide per CTR value,
order statistics from an exact sort, comparisons for the bins; nothing here
depends on a reduction order, so the columns, grids and bins are the same on
every vendor and in the host column (`gbdt/ctrs/tree_ctr.mojo`).
"""

from max.gpu.host import DeviceBuffer, DeviceContext, HostBuffer

from core.device_plan import DevicePlan
from core.device_zero import enqueue_fill
from core.identity_trace import IdentityTrace
from core.transfer import transfer_download_list
from gbdt.ctrs.ctr import CTR_BORDERS, CTR_FEATURE_FREQ, TCtrConfig
from gbdt.ctrs.ctr_binarization import uniform_borders_from_range
from gbdt.ctrs.fast_prep import HIST_BLOCK, _read_counts_wide, code_histogram_kernel
from gbdt.ctrs.tree_ctr import (
    TreeCtrPlan,
    TreeCtrTensor,
    tree_ctr_check_configs,
    tree_ctr_median_from_words,
    tree_ctr_median_words,
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
from gbdt.ctrs.tree_ctr_device import (
    TreeMedianScratch,
    launch_tree_ctr_bins,
    launch_tree_group_count,
    launch_tree_median_words,
    launch_tree_slot_clear,
    launch_tree_slot_copy,
    launch_tree_slot_write,
    launch_tree_split_bits,
    tree_ctr_base_builder,
    tree_ctr_candidate_columns_device,
)
from gbdt.data.permutation import TRandom
from gbdt.gpu_data.compressed_index_builder import CompressedIndexLayout
from gbdt.methods.greedy_subsets_searcher.depthwise_stage_times import (
    StageTimes,
)
from gbdt.methods.greedy_subsets_searcher.greedy_search_helper import (
    TSynchronizedSymmetricLevelState,
    begin_synchronized_symmetric_tree,
    finish_synchronized_symmetric_tree,
    new_synchronized_symmetric_state,
    search_synchronized_symmetric_level,
)
from gbdt.models.combination_map import (
    TCombinationCtrs,
    TCombinationMap,
    combination_ctr_tables_from_counts,
)
from gbdt.models.kernel.combination_code import (
    ResidentCombinationPlan,
    enqueue_combination_columns,
    launch_combination_tuples,
)
from gbdt.models.kernel.resident_ctr import CTR_NO_BAD
from gbdt.models.oblivious_model import (
    BIN_SPLIT_TAKE_GREATER,
    TAdditiveModel,
    TBinarySplit,
)


struct TreeCtrFit(Movable):
    """A fit's tree-CTR state: what `train` knows before the first tree, the
    resident buffers the levels reuse, and what the trees chose."""

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
    var layout: CompressedIndexLayout
    #: per base column: the raw input feature a float or one-hot column
    #: reads (-1 for a CTR column), its model borders and the side a NaN
    #: takes under a `bin >` split
    var column_raw: List[Int]
    var column_borders: List[List[Float32]]
    var column_nan_side: List[Int]
    #: dense codes of every CTR-bearing categorical feature, in
    #: `plan.cat_features` order
    var d_cat_codes: List[DeviceBuffer[DType.uint32]]
    #: one CTR estimation order per permutation (one identity-free order
    #: when the fit has no permutation-dependent CTR)
    var d_orders: List[DeviceBuffer[DType.uint32]]
    var d_btarget: DeviceBuffer[DType.uint8]
    var target_classes_count: Int
    var has_borders: Bool
    # the held-out arm: the raw eval input, column-major, and its index
    var test_rows: Int
    var n_raw: Int
    var d_test_x: DeviceBuffer[DType.float32]
    var d_test_cindex: DeviceBuffer[DType.uint32]
    var d_test_vals: DeviceBuffer[DType.float32]
    # scratch, allocated once
    var d_split_bits: List[DeviceBuffer[DType.uint32]]
    var d_bins_tmp: DeviceBuffer[DType.uint32]
    var d_one: DeviceBuffer[DType.uint32]
    var median: TreeMedianScratch
    var words_per_slot: Int
    var d_words: DeviceBuffer[DType.float32]
    var h_words: HostBuffer[DType.float32]
    var d_borders: DeviceBuffer[DType.float32]
    var h_borders: HostBuffer[DType.float32]
    # the search index's slots that hold something other than bin 0, per
    # permutation
    var dirty_persist: List[List[Bool]]
    var dirty_cand: List[List[Bool]]
    #: set by any slot write on the search index; the driver turns it into
    #: a full histogram rebuild and clears it
    var learn_changed: Bool
    #: the permutation whose index the level state holds (-1 before the
    #: first tree)
    var state_perm: Int
    # the level's candidates
    var level_base: TreeCtrTensor
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
    var d_test_bins: List[DeviceBuffer[DType.uint32]]
    #: per tree, the accepted entry of each persist slot it used
    var tree_persist: List[List[Int]]

    def __init__(
        out self,
        ctx: DeviceContext,
        var plan: TreeCtrPlan,
        var configs: List[TCtrConfig],
        var border_counts: List[Int],
        n_rows: Int,
        n_base: Int,
        max_depth: Int,
        var layout: CompressedIndexLayout,
        var column_raw: List[Int],
        var column_borders: List[List[Float32]],
        var column_nan_side: List[Int],
        var d_cat_codes: List[DeviceBuffer[DType.uint32]],
        var d_orders: List[DeviceBuffer[DType.uint32]],
        var d_btarget: DeviceBuffer[DType.uint8],
        target_classes_count: Int,
        permutation_count: Int,
        test_rows: Int,
        n_raw: Int,
        var d_test_x: DeviceBuffer[DType.float32],
        var d_test_cindex: DeviceBuffer[DType.uint32],
    ) raises:
        tree_ctr_check_configs(configs)
        if not plan.enabled() or len(configs) == 0:
            raise Error("tree CTR fit: no categorical feature or no config")
        if len(border_counts) != len(configs):
            raise Error("tree CTR fit: one border count per config")
        if len(d_cat_codes) != len(plan.cat_features):
            raise Error("tree CTR fit: one code column per categorical feature")
        if len(plan.column_kind) != n_base or len(column_raw) != n_base or (
            len(column_borders) != n_base or len(column_nan_side) != n_base
        ):
            raise Error("tree CTR fit: the base column plan is incomplete")
        if len(d_orders) < 1 or permutation_count < 1 or max_depth < 1:
            raise Error("tree CTR fit: needs an order, a permutation and a level")
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
        if borders_seen and len(d_orders) != permutation_count:
            raise Error(
                "a Borders combination CTR needs one estimation order per"
                " permutation"
            )
        var split_bits = List[DeviceBuffer[DType.uint32]]()
        for _ in range(max_depth):  # small-loop(max_depth: tree levels): one scratch column per possible split member
            split_bits.append(
                ctx.enqueue_create_buffer[DType.uint32](max(n_rows, 1))
            )
        var dirty_persist = List[List[Bool]]()
        var dirty_cand = List[List[Bool]]()
        for _ in range(permutation_count):  # small-loop(permutation_count: permutations): one flag list per permutation
            dirty_persist.append(List[Bool](length=max_depth, fill=False))
            dirty_cand.append(List[Bool](length=n_slots, fill=False))
        var slot_borders = List[List[Float32]]()
        for _ in range(n_slots):  # small-loop(n_slots: candidate slots): one empty grid per slot
            slot_borders.append(List[Float32]())
        var words_per_slot = tree_ctr_median_words(cap)
        self.fold_cap = cap
        self.has_borders = borders_seen
        self.n_rows = n_rows
        self.n_base = n_base
        self.max_depth = max_depth
        self.target_classes_count = target_classes_count
        self.test_rows = test_rows
        self.n_raw = n_raw
        self.words_per_slot = words_per_slot
        self.d_split_bits = split_bits^
        self.d_bins_tmp = ctx.enqueue_create_buffer[DType.uint32](max(n_rows, 1))
        self.d_one = ctx.enqueue_create_buffer[DType.uint32](1)
        self.median = TreeMedianScratch(ctx, n_rows)
        self.d_words = ctx.enqueue_create_buffer[DType.float32](
            n_slots * words_per_slot
        )
        self.h_words = ctx.enqueue_create_host_buffer[DType.float32](
            n_slots * words_per_slot
        )
        self.d_borders = ctx.enqueue_create_buffer[DType.float32](n_slots * cap)
        self.h_borders = ctx.enqueue_create_host_buffer[DType.float32](
            n_slots * cap
        )
        self.d_test_vals = ctx.enqueue_create_buffer[DType.float32](
            max(test_rows, 1)
        )
        self.dirty_persist = dirty_persist^
        self.dirty_cand = dirty_cand^
        self.learn_changed = False
        self.state_perm = -1
        self.level_base = TreeCtrTensor()
        self.slot_live = List[Bool](length=n_slots, fill=False)
        self.slot_borders = slot_borders^
        self.tensor = TreeCtrTensor()
        self.persist_tensor = List[TreeCtrTensor]()
        self.persist_accept = List[Int]()
        self.maps = List[TreeCtrMapEntry]()
        self.accepted = List[TreeCtrAccepted]()
        self.d_test_bins = List[DeviceBuffer[DType.uint32]]()
        self.tree_persist = List[List[Int]]()
        self.plan = plan^
        self.configs = configs^
        self.border_counts = border_counts^
        self.layout = layout^
        self.column_raw = column_raw^
        self.column_borders = column_borders^
        self.column_nan_side = column_nan_side^
        self.d_cat_codes = d_cat_codes^
        self.d_orders = d_orders^
        self.d_btarget = d_btarget^
        self.d_test_x = d_test_x^
        self.d_test_cindex = d_test_cindex^

    def slot_count(self) -> Int:
        return len(self.plan.cat_features) * len(self.configs)

    def persist_column(self, j: Int) -> Int:
        return self.n_base + j

    def candidate_column(self, slot: Int) -> Int:
        return self.n_base + self.max_depth + slot

    def order_of(self, permutation: Int) -> Int:
        """The estimation order a permutation's columns are taken in."""
        if len(self.d_orders) == 1:
            return 0
        return permutation

    def begin_tree(
        mut self,
        ctx: DeviceContext,
        learn_p: Int,
        mut cindex: DeviceBuffer[DType.uint32],
    ) raises:
        """A new tree on permutation `learn_p`'s index: every slot an
        earlier tree left holding a column goes back to bin 0, and the
        tree's tensor starts empty."""
        # driver-loop: over the persist slots (tree levels)
        for j in range(self.max_depth):
            if self.dirty_persist[learn_p][j]:
                launch_tree_slot_clear(
                    ctx, cindex, self.layout.features[self.persist_column(j)],
                    self.n_rows,
                )
                self.dirty_persist[learn_p][j] = False
        # driver-loop: over the candidate slots (features x configs)
        for s in range(self.slot_count()):
            if self.dirty_cand[learn_p][s]:
                launch_tree_slot_clear(
                    ctx, cindex, self.layout.features[self.candidate_column(s)],
                    self.n_rows,
                )
                self.dirty_cand[learn_p][s] = False
            self.slot_live[s] = False
        self.learn_changed = True
        self.tensor = TreeCtrTensor()
        self.level_base = TreeCtrTensor()
        self.persist_tensor.clear()
        self.persist_accept.clear()

    def end_tree(mut self):
        """Record which accepted entry each persist slot of the finished
        tree holds (`rewrite_model` reads it by tree index)."""
        self.tree_persist.append(self.persist_accept.copy())

    def _cat_ordinal(self, feature: Int) raises -> Int:
        for k in range(len(self.plan.cat_features)):  # small-loop(cat_features: CTR-bearing categorical features): finds one feature's ordinal
            if self.plan.cat_features[k] == feature:
                return k
        raise Error(
            "tree CTR fit: categorical feature " + String(feature)
            + " feeds no CTR"
        )

    def _member_handles(
        self,
        base: TreeCtrTensor,
        mut codes: List[DeviceBuffer[DType.uint32]],
        mut cards: List[Int],
    ) raises:
        """The base tensor's members as device code columns, in the order
        every builder adds them: the categorical features ascending, then
        the splits ascending (their sides are in `d_split_bits`, written by
        `_launch_split_sides` for this same tensor)."""
        # driver-loop: over the tensor's categorical members
        for i in range(len(base.cats)):
            var k = self._cat_ordinal(base.cats[i])
            codes.append(self.d_cat_codes[k].copy())
            cards.append(self.plan.cat_cardinality[k])
        if len(base.splits) > len(self.d_split_bits):
            raise Error("tree CTR fit: more split members than tree levels")
        # driver-loop: over the tensor's split members
        for i in range(len(base.splits)):
            codes.append(self.d_split_bits[i].copy())
            cards.append(2)

    def _launch_split_sides(
        mut self,
        ctx: DeviceContext,
        mut cindex: DeviceBuffer[DType.uint32],
    ) raises:
        """Every split member of the level's base tensor as its 0/1 side per
        row, off the compressed index (float and one-hot columns are the
        same on every permutation)."""
        if len(self.level_base.splits) > len(self.d_split_bits):
            raise Error("tree CTR fit: more split members than tree levels")
        # driver-loop: over the tensor's split members
        for i in range(len(self.level_base.splits)):
            var split = self.level_base.splits[i]
            var bits = self.d_split_bits[i].copy()
            launch_tree_split_bits(
                ctx, cindex, self.layout.features[split.feature], self.n_rows,
                split.bin, split.take_bin, bits,
            )

    def mint_level(
        mut self,
        ctx: DeviceContext,
        learn_p: Int,
        mut cindex: DeviceBuffer[DType.uint32],
    ) raises:
        """The next level's candidates into the candidate slots of the
        search index: one column per (candidate feature, config) over the
        rows grouped by the tree's tensor extended by that feature, binned
        under its own grid. A slot no candidate holds any more goes back to
        bin 0. One readback (the order statistics of every column of the
        level)."""
        var n_cfg = len(self.configs)
        var n_slots = self.slot_count()
        var cands = self.plan.candidates(self.tensor)
        self.level_base = self.tensor.copy()
        var live = List[Bool](length=n_slots, fill=False)
        var columns = List[DeviceBuffer[DType.float32]]()
        var column_slot = List[Int]()
        if len(cands) > 0:
            self._launch_split_sides(ctx, cindex)
            var codes = List[DeviceBuffer[DType.uint32]]()
            var cards = List[Int]()
            self._member_handles(self.level_base, codes, cards)
            var base = tree_ctr_base_builder(
                ctx, self.d_orders[self.order_of(learn_p)], self.n_rows,
                codes, cards,
            )
            # driver-loop: over the level's candidate features
            for i in range(len(cands)):
                var k = cands[i]
                var cols = tree_ctr_candidate_columns_device(
                    ctx, base, self.d_cat_codes[k],
                    self.plan.cat_cardinality[k], self.n_rows, self.configs,
                    self.d_btarget,
                )
                # driver-loop: over the tree-CTR configs
                for c in range(n_cfg):
                    var slot = k * n_cfg + c
                    var col = cols[c].copy()
                    # Median needs every border's pair; Uniform reads the
                    # record's two extremes only
                    var asked = 1
                    if self.configs[c].ctr_type == CTR_FEATURE_FREQ:
                        asked = self.border_counts[c]
                    launch_tree_median_words(
                        ctx, col, self.n_rows, asked, self.median,
                        self.d_words, slot * self.words_per_slot,
                    )
                    live[slot] = True
                    columns.append(col^)
                    column_slot.append(slot)
            # ONE readback for the level
            ctx.enqueue_copy(
                dst_ptr=self.h_words.unsafe_ptr(), src_buf=self.d_words
            )
            ctx.synchronize()
            _ = base^  # past the drain
            # the grids, on the host: a handful of words per column
            # driver-loop: over the level's candidate columns
            for i in range(len(column_slot)):
                var slot = column_slot[i]
                var c = slot % n_cfg
                var at = slot * self.words_per_slot
                var grid = List[Float32]()
                if self.configs[c].ctr_type == CTR_FEATURE_FREQ:
                    var record = List[Float32](capacity=self.words_per_slot)
                    for w in range(tree_ctr_median_words(self.border_counts[c])):  # small-loop(words: order statistics of one grid, at most 2 + 2 * 255): one column's record
                        record.append(self.h_words.unsafe_ptr().unsafe_load(at + w))
                    grid = tree_ctr_median_from_words(
                        record, 0, self.border_counts[c]
                    )
                else:
                    grid = uniform_borders_from_range(
                        self.h_words.unsafe_ptr().unsafe_load(at),
                        self.h_words.unsafe_ptr().unsafe_load(at + 1),
                        self.border_counts[c],
                    )
                    if len(grid) == 0:
                        grid.append(Float32(0.5))
                if len(grid) > self.fold_cap:
                    raise Error("tree CTR fit: a grid exceeds its slot")
                for b in range(len(grid)):  # small-loop(grid: borders of one CTR grid, at most 255): stages one grid for the upload
                    self.h_borders.unsafe_ptr().unsafe_store(
                        slot * self.fold_cap + b, grid[b]
                    )
                self.slot_borders[slot] = grid^
            ctx.enqueue_copy(
                dst_buf=self.d_borders, src_ptr=self.h_borders.unsafe_ptr()
            )
            # bins and slot writes, enqueued; the next level's histograms
            # read them in queue order
            # driver-loop: over the level's candidate columns
            for i in range(len(column_slot)):
                var slot = column_slot[i]
                var col = columns[i].copy()
                launch_tree_ctr_bins(
                    ctx, col, self.n_rows, self.d_borders,
                    slot * self.fold_cap, len(self.slot_borders[slot]),
                    self.d_bins_tmp,
                )
                launch_tree_slot_write(
                    ctx, cindex,
                    self.layout.features[self.candidate_column(slot)],
                    self.n_rows, self.d_bins_tmp,
                )
                self.dirty_cand[learn_p][slot] = True
                self.learn_changed = True
            # the columns are read by the launches just enqueued: drain
            # before their handles go
            ctx.synchronize()
            _ = len(columns)
        # driver-loop: over the candidate slots (features x configs)
        for s in range(n_slots):
            if not live[s] and self.dirty_cand[learn_p][s]:
                launch_tree_slot_clear(
                    ctx, cindex, self.layout.features[self.candidate_column(s)],
                    self.n_rows,
                )
                self.dirty_cand[learn_p][s] = False
                self.learn_changed = True
            self.slot_live[s] = live[s]

    def _map_for(
        mut self,
        ctx: DeviceContext,
        k: Int,
    ) raises -> Int:
        """The map entry of (the level's base tensor, candidate feature
        ordinal `k`), built the first time that combination is chosen: the
        rows grouped by its members, the group count, every group's member
        tuple and the learn pool's counts, read home once."""
        var feature = self.plan.cat_features[k]
        var found = tree_ctr_find_map(self.maps, self.level_base, feature)
        if found >= 0:
            return found
        var codes = List[DeviceBuffer[DType.uint32]]()
        var cards = List[Int]()
        self._member_handles(self.level_base, codes, cards)
        # group ids do not depend on the order the rows are walked in
        var builder = tree_ctr_base_builder(
            ctx, self.d_orders[0], self.n_rows, codes, cards
        )
        builder.add_cat_feature_bins_device(
            ctx, self.d_cat_codes[k], self.plan.cat_cardinality[k]
        )
        launch_tree_group_count(ctx, builder, self.d_one)
        var h_one = transfer_download_list[DType.uint32](ctx, self.d_one, 1)
        var n_groups = Int(h_one[0])
        if n_groups < 1 or n_groups > self.n_rows:
            raise Error(
                "tree CTR fit: a combination of " + String(n_groups)
                + " groups over " + String(self.n_rows) + " rows"
            )
        codes.append(self.d_cat_codes[k].copy())
        var n_members = len(codes)
        var d_tuples = ctx.enqueue_create_buffer[DType.uint32](
            n_groups * n_members
        )
        var d_ids = builder.current_bins.copy()
        # driver-loop: over the combination's members
        for j in range(n_members):
            var member = codes[j].copy()
            launch_combination_tuples(
                ctx, member, d_ids, self.n_rows, n_members, j, n_groups,
                d_tuples,
            )
        var classes = self.target_classes_count if self.has_borders else 1
        var d_counts = ctx.enqueue_create_buffer[DType.uint32](n_groups)
        enqueue_fill(ctx, d_counts, UInt32(0))
        var d_hist = ctx.enqueue_create_buffer[DType.uint32](n_groups * classes)
        enqueue_fill(ctx, d_hist, UInt32(0))
        ctx.enqueue_function[code_histogram_kernel](
            d_ids.unsafe_ptr(),
            Int32(self.n_rows),
            self.d_btarget.unsafe_ptr(),
            Int32(1) if self.has_borders else Int32(0),
            Int32(classes),
            d_counts.unsafe_ptr(),
            d_hist.unsafe_ptr(),
            grid_dim=((self.n_rows + HIST_BLOCK - 1) // HIST_BLOCK, 1, 1),
            block_dim=(HIST_BLOCK, 1, 1),
        )
        var tuples = transfer_download_list[DType.uint32](
            ctx, d_tuples, n_groups * n_members
        )
        var counts = List[Int]()
        _ = _read_counts_wide(ctx, d_counts, n_groups, False, counts)
        var histogram = List[Int]()
        if self.has_borders:
            _ = _read_counts_wide(
                ctx, d_hist, n_groups * classes, False, histogram
            )
        _ = builder^  # past the drains
        _ = d_ids^
        _ = d_hist^
        _ = len(codes)

        # the one statement of the member list, shared with the host column
        var members = tree_ctr_members(
            self.level_base, feature, self.column_raw, self.column_borders,
            self.column_nan_side,
        )
        var cmap = TCombinationMap(members^, tuples^)
        self.maps.append(
            TreeCtrMapEntry(
                self.level_base.cats.copy(), self.level_base.splits.copy(),
                feature, cmap^, counts^, histogram^,
            )
        )
        return len(self.maps) - 1

    def _test_bins_for(
        mut self,
        ctx: DeviceContext,
        map_ordinal: Int,
        config: Int,
        slot: Int,
    ) raises -> DeviceBuffer[DType.uint32]:
        """A chosen column on the held-out rows: the derived code of every
        row under the map (an unseen tuple takes the prior), the table's
        value, the bin under the column's borders. The model's own predict
        launches, over the raw eval input resident on the device."""
        var one = List[TCtrConfig]()
        one.append(self.configs[config])
        var cmap = self.maps[map_ordinal].cmap.copy()
        var tables = combination_ctr_tables_from_counts(
            self.maps[map_ordinal].counts, self.maps[map_ordinal].histogram,
            cmap.groups(), one, self.target_classes_count, 0, 0, self.n_rows,
        )
        var ctrs = TCombinationCtrs()
        ctrs.add(cmap^, tables^)
        var plan = ResidentCombinationPlan(ctx, ctrs, 1, self.n_raw)
        var words = 1 + max(self.n_raw, 1)
        var d_bad = ctx.enqueue_create_buffer[DType.int32](words)
        enqueue_fill(ctx, d_bad, CTR_NO_BAD)
        enqueue_combination_columns(
            ctx, plan, self.d_test_x, self.test_rows, False, self.d_test_vals,
            d_bad,
        )
        var d_bins = ctx.enqueue_create_buffer[DType.uint32](self.test_rows)
        launch_tree_ctr_bins(
            ctx, self.d_test_vals, self.test_rows, self.d_borders,
            slot * self.fold_cap, len(self.slot_borders[slot]), d_bins,
        )
        var bad = transfer_download_list[DType.int32](ctx, d_bad, words)
        _ = plan^  # past the drain
        if bad[0] != CTR_NO_BAD:
            var f = Int(bad[0])
            raise Error(
                "eval_set: categorical feature " + String(f) + " row "
                + String(Int(bad[1 + f])) + " is not a dense category code"
            )
        return d_bins^

    def take_winner(
        mut self,
        ctx: DeviceContext,
        learn_p: Int,
        cindexes: List[DeviceBuffer[DType.uint32]],
        mut out_splits: List[TBinarySplit],
    ) raises -> Bool:
        """The accepted winner of a level (the last entry of `out_splits`)
        into the tree's tensor. A won candidate becomes the tree's next
        persist column on every permutation's index and on the test index,
        and the split is rewritten to it. False when the winner sits on a
        slot that holds no column (bin 0 for every row: no split), which
        ends the tree as a level without an admissible split does; the
        caller drops the split."""
        var won = out_splits[len(out_splits) - 1]
        var feature = Int(won.feature_id)
        var bin = Int(won.bin_idx)
        if feature < self.n_base:
            self.plan.track_split(self.tensor, feature, bin)
            return True
        if feature < self.n_base + self.max_depth:
            var j = feature - self.n_base
            if j >= len(self.persist_tensor):
                return False
            self.tensor.add_tensor(self.persist_tensor[j])
            return True
        var slot = feature - self.n_base - self.max_depth
        if slot >= self.slot_count() or not self.slot_live[slot]:
            return False
        if bin < 0 or bin >= len(self.slot_borders[slot]):
            return False
        var n_cfg = len(self.configs)
        var k = slot // n_cfg
        var c = slot % n_cfg
        var dependent = self.configs[c].ctr_type == CTR_BORDERS
        var order = self.order_of(learn_p) if dependent else 0

        var m = self._map_for(ctx, k)
        var a = tree_ctr_find_accepted(self.accepted, m, c, order)
        if a < 0:
            self.accepted.append(
                TreeCtrAccepted(m, c, order, self.slot_borders[slot].copy())
            )
            a = len(self.accepted) - 1
            if self.test_rows > 0:
                self.d_test_bins.append(self._test_bins_for(ctx, m, c, slot))

        var j = len(self.persist_tensor)
        if j >= self.max_depth:
            raise Error("tree CTR fit: more persist columns than tree levels")
        var persist = self.layout.features[self.persist_column(j)]
        var source = self.layout.features[self.candidate_column(slot)]
        var learn = cindexes[learn_p].copy()
        var learn_dst = cindexes[learn_p].copy()
        # the search index first: the other permutations copy from its
        # candidate slot, which stays as it is until the next mint
        launch_tree_slot_copy(
            ctx, learn, source, learn_dst, persist, self.n_rows
        )
        self.dirty_persist[learn_p][j] = True
        self.learn_changed = True
        var base_codes = List[DeviceBuffer[DType.uint32]]()
        var base_cards = List[Int]()
        if dependent and len(cindexes) > 1:
            self._member_handles(self.level_base, base_codes, base_cards)
        var one = List[TCtrConfig]()
        one.append(self.configs[c])
        # driver-loop: over the permutations
        for p in range(len(cindexes)):
            if p == learn_p:
                continue
            var other = cindexes[p].copy()
            if dependent and self.order_of(p) != order:
                # this permutation's own ordered statistic, binned under
                # the search permutation's borders
                var base = tree_ctr_base_builder(
                    ctx, self.d_orders[self.order_of(p)], self.n_rows,
                    base_codes, base_cards,
                )
                var cols = tree_ctr_candidate_columns_device(
                    ctx, base, self.d_cat_codes[k],
                    self.plan.cat_cardinality[k], self.n_rows, one,
                    self.d_btarget,
                )
                var col = cols[0].copy()
                launch_tree_ctr_bins(
                    ctx, col, self.n_rows, self.d_borders,
                    slot * self.fold_cap, len(self.slot_borders[slot]),
                    self.d_bins_tmp,
                )
                launch_tree_slot_write(
                    ctx, other, persist, self.n_rows, self.d_bins_tmp
                )
                ctx.synchronize()
                _ = col^  # past the drain
                _ = base^
            else:
                launch_tree_slot_copy(
                    ctx, learn, source, other, persist, self.n_rows
                )
            self.dirty_persist[p][j] = True
        if self.test_rows > 0:
            var test_bins = self.d_test_bins[a].copy()
            launch_tree_slot_write(
                ctx, self.d_test_cindex, persist, self.test_rows, test_bins
            )
            # the write reads `test_bins` through this handle
            ctx.synchronize()
            _ = test_bins^

        out_splits[len(out_splits) - 1] = TBinarySplit(
            Int32(self.persist_column(j)),
            won.bin_idx,
            Int32(BIN_SPLIT_TAKE_GREATER),
        )
        var joined = self.level_base.copy()
        joined.add_cat(self.plan.cat_features[k])
        self.tensor.add_tensor(joined)
        self.persist_tensor.append(joined^)
        self.persist_accept.append(a)
        return True

    def assemble(self) raises -> TreeCtrModelSection:
        """The model's tree-CTR section: the maps in the order they were
        first chosen, each followed by its columns, which fill the model's
        columns from `n_base` on."""
        return tree_ctr_assemble(
            self.maps, self.accepted, self.configs, self.n_base,
            self.target_classes_count, self.n_rows,
        )

    def rewrite_model(
        self, mut model: TAdditiveModel, column_of: List[Int]
    ) raises:
        """Every tree's persist-slot splits renumbered to their model
        columns (the fit grew them on layout slots that the next tree
        reuses)."""
        if len(model.weak_models) > len(self.tree_persist):
            raise Error("tree CTR fit: a tree without its persist record")
        # driver-loop: over the trees and their splits (model structure)
        for t in range(len(model.weak_models)):
            for i in range(len(model.weak_models[t].structure.splits)):  # small-loop(splits: tree depth): renumbers one tree's splits
                var split = model.weak_models[t].structure.splits[i]
                var feature = Int(split.feature_id)
                if feature < self.n_base:
                    continue
                var j = feature - self.n_base
                if j >= len(self.tree_persist[t]):
                    raise Error("tree CTR fit: a split on an unheld slot")
                model.weak_models[t].structure.splits[i] = TBinarySplit(
                    Int32(column_of[self.tree_persist[t][j]]),
                    split.bin_idx,
                    split.split_type,
                )


def run_tree_ctr_tree(
    mut fit: TreeCtrFit,
    mut states: List[TSynchronizedSymmetricLevelState],
    device_plan: DevicePlan,
    ctx: DeviceContext,
    learn_p: Int,
    cindexes: List[DeviceBuffer[DType.uint32]],
    approx_dim: Int,
    mut stats: DeviceBuffer[DType.float32],
    mut row_index: DeviceBuffer[DType.uint32],
    mut cursor: DeviceBuffer[DType.float32],
    mut out_splits: List[TBinarySplit],
    mut out_leaf_values: List[Float32],
    mut out_leaf_offsets: List[Int],
    mut trace: IdentityTrace,
    mut times: StageTimes,
    tree_tag: String,
    export_offsets: Bool,
    use_subtraction: Bool,
    apply_to_cursor: Bool,
    learning_rate: Float32,
    l2_leaf_reg: Float32,
    score_function: Int,
    multiclass_optimization: Bool,
    var mags_dev: Optional[DeviceBuffer[DType.float32]],
    random_strength: Float32,
    random_seed: UInt64,
) raises -> List[Int]:
    """One oblivious tree with tree CTRs, on permutation `learn_p`'s index
    (`cindexes` holds every permutation's, in permutation order). The
    arguments after `cindexes` are `run_tree_layout_traced`'s, with the same
    meaning; the return value and the three `out_` lists are its too.
    `states` is the fit's pool of one level state (empty before the first
    tree). The level loop is serial by nature (see the module note); the
    work inside a level is device launches over all rows and all
    candidates."""
    if learn_p < 0 or learn_p >= len(cindexes):
        raise Error("tree CTR tree: the learn permutation has no index")
    if len(cindexes) != len(fit.dirty_persist):
        raise Error("tree CTR tree: one index per permutation of the fit")
    var cindex = cindexes[learn_p].copy()
    if len(states) == 0:
        states.append(
            new_synchronized_symmetric_state(
                device_plan, ctx, fit.layout, cindex, fit.n_rows,
                1 + approx_dim, fit.max_depth,
            )
        )
    elif fit.state_perm != learn_p:
        # another permutation's index under the same pinned layout (the
        # call drains the readers of the old handle first)
        states[0].replace_active_cindex(ctx, cindex.copy())
    fit.state_perm = learn_p
    fit.begin_tree(ctx, learn_p, cindex)
    var score_std_dev = begin_synchronized_symmetric_tree(
        states[0], ctx, stats, mags_dev^, random_strength,
        multiclass_optimization,
    )
    # their `TGpuAwareRandom`, one draw per level before its launches, as
    # the fused path draws it
    var level_rand = TRandom(random_seed)
    for depth in range(fit.max_depth):
        var level_seed = level_rand.next_uniform_l()
        if fit.learn_changed:
            states[0].cindex_changed_since_histogram = True
            fit.learn_changed = False
        if not search_synchronized_symmetric_level(
            states[0], ctx, row_index, stats, out_splits, score_function,
            multiclass_optimization, l2_leaf_reg, score_std_dev, level_seed,
            use_subtraction, trace, times, tree_tag,
        ):
            break
        if not fit.take_winner(ctx, learn_p, cindexes, out_splits):
            _ = out_splits.pop()
            break
        _ = states[0].enqueue_post_winner(ctx, row_index, stats)
        if depth + 1 < fit.max_depth:
            fit.mint_level(ctx, learn_p, cindex)
    var sizes = finish_synchronized_symmetric_tree(
        states[0], ctx, stats, row_index, cursor, apply_to_cursor,
        export_offsets, learning_rate, l2_leaf_reg, out_leaf_values,
        out_leaf_offsets, trace, tree_tag,
    )
    fit.end_tree()
    return sizes^
