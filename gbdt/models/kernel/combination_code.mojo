# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Tree-CTR (feature combination) model columns on the device.

The statement is `gbdt/models/combination_map.mojo`. Three launches:

  * `combination_code_kernel`: one thread per row, the row's derived code
    under one combination map (`combination_row_code`, the function the host
    column runs: member values off the raw input, then the lexicographic
    binary search over the map's tuples; an unseen tuple takes the code one
    past the table).
  * `combination_value_kernel`: one thread per row, one tree-CTR model
    column = the simple CTR table value at the derived code
    (`resident_ctr._simple_value`, the simple CTR's device statement,
    unchanged; the code one past the table is its unseen-category arm, the
    prior).
  * `combination_tuples_kernel` (fit side): one thread per learn row writes
    one member's value into the row's group slot of the tuple list. The
    rows of a group hold the same value, so the threads never disagree on a
    word and no atomic is needed.

`ResidentCombinationPlan` is the model's maps and tables flattened once per
resident model; `enqueue_combination_columns` is the per-predict call: one
code launch per map, then one value launch per table of that map, on the
caller's queue, reading the raw input the simple expansion already uploaded
and writing straight into the model's column-major matrix. Nothing reads a
row on the host.

Parallel: every launch is one thread per row. The loops here are over maps
and tables (model structure).

Bits: the code is comparisons and integer arithmetic, the same on every
vendor and in the host column; the value is the simple CTR's correctly
rounded statement.
"""

from max.gpu import block_dim, block_idx, grid_dim, thread_idx
from max.gpu.host import DeviceBuffer, DeviceContext

from core.transfer import transfer_upload_list
from gbdt.models.combination_map import (
    COMB_MEMBER_WORDS,
    CombF32Ptr,
    CombI64Ptr,
    CombU32Ptr,
    TCombinationCtrs,
    combination_row_code,
)
from gbdt.models.kernel.resident_ctr import (
    CTR_BLOCK,
    ResidentCtrPlan,
    SIMPLE_F,
    SIMPLE_I,
    _flag_bad,
    _grid,
    _simple_value,
    _upload_f32,
    _upload_i64,
)


def combination_code_kernel(
    x: CombF32Ptr,
    n_rows_in: Int32,
    n_raw_in: Int32,
    row_major: Int32,
    members_i: CombI64Ptr,
    members_f: CombF32Ptr,
    member_off_in: Int32,
    m_in: Int32,
    tuples: CombU32Ptr,
    tuple_off_in: Int32,
    g_in: Int32,
    codes: CombU32Ptr,
    bad: MutPointer[Int32, MutAnyOrigin],
):
    """`codes[r]` = row `r`'s derived code under one map (grid-stride). A
    row whose categorical input is not a dense category code is recorded in
    `bad` (lowest feature, that feature's lowest row) and takes the unseen
    code."""
    var n = Int(n_rows_in)
    var n_raw = Int(n_raw_in)
    var m = Int(m_in)
    var g = Int(g_in)
    var mi = members_i + Int(member_off_in) * COMB_MEMBER_WORDS
    var mf = members_f + Int(member_off_in)
    var tp = tuples + Int(tuple_off_in)
    var r = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var stride = Int(block_dim.x) * Int(grid_dim.x)
    while r < n:
        var code = combination_row_code(
            x, n, n_raw, row_major, r, mi, mf, m, tp, g
        )
        if code < 0:
            _flag_bad(bad, -(code + 1), r)
            code = g
        codes[r] = UInt32(code)
        r += stride


def combination_value_kernel(
    codes: CombU32Ptr,
    n_rows_in: Int32,
    meta_i: CombI64Ptr,
    meta_f: CombF32Ptr,
    counts: CombI64Ptr,
    t_in: Int32,
    column_in: Int32,
    dst: CombF32Ptr,
):
    """`dst[column * n + r]` = table `t`'s value at `codes[r]`
    (grid-stride)."""
    var n = Int(n_rows_in)
    var t = Int(t_in)
    var base = Int(column_in) * n
    var r = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var stride = Int(block_dim.x) * Int(grid_dim.x)
    while r < n:
        dst[base + r] = _simple_value(meta_i, meta_f, counts, t, Int(codes[r]))
        r += stride


def combination_tuples_kernel(
    member_codes: CombU32Ptr,
    group_ids: CombU32Ptr,
    n_rows_in: Int32,
    m_in: Int32,
    j_in: Int32,
    g_in: Int32,
    tuples: CombU32Ptr,
):
    """`tuples[group_ids[r] * m + j] = member_codes[r]` for every learn row
    (grid-stride): member `j`'s word of every group's tuple."""
    var n = Int(n_rows_in)
    var m = Int(m_in)
    var j = Int(j_in)
    var g = Int(g_in)
    var r = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var stride = Int(block_dim.x) * Int(grid_dim.x)
    while r < n:
        var gid = Int(group_ids[r])
        if gid < g:
            tuples[gid * m + j] = member_codes[r]
        r += stride


def launch_combination_tuples(
    ctx: DeviceContext,
    mut member_codes: DeviceBuffer[DType.uint32],
    mut group_ids: DeviceBuffer[DType.uint32],
    n_rows: Int,
    n_members: Int,
    member: Int,
    n_groups: Int,
    mut tuples: DeviceBuffer[DType.uint32],
) raises:
    """Fit side: write member `member`'s word of every group's tuple into
    `tuples` (`n_groups * n_members` words). `member_codes[r]` is the
    member's value for learn row `r` (dense code, or a split's side) and
    `group_ids[r]` the row's group (the bin builder's per-row segment id).
    One launch per member; the caller reads `tuples` home once for the map
    (`TCombinationMap(members, tuples)` checks the order)."""
    if n_rows <= 0:
        return
    if len(tuples) < n_groups * n_members:
        raise Error("combination tuples: the tuple buffer is too small")
    ctx.enqueue_function[combination_tuples_kernel](
        member_codes.unsafe_ptr(),
        group_ids.unsafe_ptr(),
        Int32(n_rows),
        Int32(n_members),
        Int32(member),
        Int32(n_groups),
        tuples.unsafe_ptr(),
        grid_dim=_grid(n_rows),
        block_dim=CTR_BLOCK,
    )


struct ResidentCombinationPlan(Movable):
    """A model's combination maps and tree-CTR tables flattened for the
    kernels above, plus the per-call code buffer of the current row count."""

    var n_raw: Int
    var n_cols: Int
    #: per map: member count, first member, first tuple word, group count
    var map_members: List[Int]
    var map_member_off: List[Int]
    var map_tuple_off: List[Int]
    var map_groups: List[Int]
    #: per table: its map and its model column
    var table_map: List[Int]
    var table_column: List[Int]
    var d_members_i: DeviceBuffer[DType.int64]
    var d_members_f: DeviceBuffer[DType.float32]
    var d_tuples: DeviceBuffer[DType.uint32]
    #: the tables in `resident_ctr`'s simple layout (`SIMPLE_I` / `SIMPLE_F`)
    var d_meta_i: DeviceBuffer[DType.int64]
    var d_meta_f: DeviceBuffer[DType.float32]
    var d_counts: DeviceBuffer[DType.int64]
    var rows: Int
    var d_codes: Optional[DeviceBuffer[DType.uint32]]

    def __init__(
        out self,
        ctx: DeviceContext,
        ctrs: TCombinationCtrs,
        n_columns: Int,
        n_raw: Int,
    ) raises:
        """`n_columns` is the whole model's column count and `n_raw` the raw
        input's; the section is checked here, once, so no kernel meets a
        table it cannot read."""
        ctrs.check(n_raw, n_columns)
        if ctrs.is_empty():
            raise Error("a combination plan needs at least one table")
        self.n_raw = n_raw
        self.n_cols = n_columns
        self.rows = 0
        self.d_codes = None
        self.map_members = List[Int]()
        self.map_member_off = List[Int]()
        self.map_tuple_off = List[Int]()
        self.map_groups = List[Int]()
        self.table_map = List[Int]()
        self.table_column = List[Int]()
        var members_i = List[Int]()
        var members_f = List[Float32]()
        var tuples = List[UInt32]()
        var n_members = 0
        # driver-loop: over the model's combination maps (model structure)
        for i in range(len(ctrs.maps)):  # small-loop(maps: combination maps of one model): packs model metadata once per model
            ref cmap = ctrs.maps[i]
            var words = cmap.flat_words()
            var borders = cmap.flat_borders()
            self.map_members.append(len(cmap.members))
            self.map_member_off.append(n_members)
            self.map_tuple_off.append(len(tuples))
            self.map_groups.append(cmap.groups())
            # driver-loop: over one map's member words
            for k in range(len(words)):  # small-loop(words: member words of one map): packs model metadata once per model
                members_i.append(Int(words[k]))
            members_f.extend(borders^)
            tuples.extend(cmap.tuples.copy())
            n_members += len(cmap.members)
        if len(tuples) > 2147483647:
            raise Error("the combination maps hold more tuple words than a kernel offset can address")
        var mi = List[Int]()
        var mf = List[Float32]()
        var counts = List[Int]()
        # driver-loop: over the model's tree-CTR tables (model structure)
        for t in range(len(ctrs.tables)):  # small-loop(tables: tree-CTR model columns): packs model metadata once per model
            ref tab = ctrs.tables[t]
            self.table_map.append(tab.source_feature)
            self.table_column.append(tab.column)
            mi.append(tab.ctr_type)
            mi.append(tab.target_classes_count)
            mi.append(tab.target_border_idx)
            mi.append(len(counts))
            mi.append(len(tab.counts))
            mi.append(tab.counter_denominator)
            mf.append(tab.prior_num)
            mf.append(tab.prior_denom)
            mf.append(tab.shift)
            mf.append(tab.scale)
            counts.extend(tab.counts.copy())
        if len(mi) != SIMPLE_I * len(ctrs.tables) or len(mf) != SIMPLE_F * len(ctrs.tables):
            raise Error("combination plan: the simple table layout changed")
        self.d_members_i = _upload_i64(ctx, members_i)
        self.d_members_f = _upload_f32(ctx, members_f)
        self.d_tuples = transfer_upload_list[DType.uint32](
            ctx, tuples, len(tuples), len(tuples)
        )
        self.d_meta_i = _upload_i64(ctx, mi)
        self.d_meta_f = _upload_f32(ctx, mf)
        self.d_counts = _upload_i64(ctx, counts)
        ctx.synchronize()
        # the uploads copy from these lists' storage
        _ = len(members_i)
        _ = len(members_f)
        _ = len(tuples)
        _ = len(mi)
        _ = len(mf)
        _ = len(counts)

    def __deinit__(deinit self):
        # every buffer before the context it was created on (DEVIATION
        # 1946): the owner releases the plan before its context
        _ = self.d_codes^
        _ = self.d_counts^
        _ = self.d_meta_f^
        _ = self.d_meta_i^
        _ = self.d_tuples^
        _ = self.d_members_f^
        _ = self.d_members_i^


def enqueue_combination_columns(
    ctx: DeviceContext,
    mut plan: ResidentCombinationPlan,
    mut d_x: DeviceBuffer[DType.float32],
    n_rows: Int,
    row_major: Bool,
    mut d_out: DeviceBuffer[DType.float32],
    mut d_bad: DeviceBuffer[DType.int32],
) raises:
    """The model's tree-CTR columns into `d_out` (`n_cols * n_rows`,
    column-major), ENQUEUED: `d_x` is the raw input on the device as the
    caller handed it over (`n_rows * n_raw`), `d_bad` the simple
    expansion's refusal words (`1 + n_raw`, filled by the caller before its
    own launches and read home after them)."""
    if n_rows <= 0:
        return
    if len(d_out) < plan.n_cols * n_rows:
        raise Error("combination columns: the model matrix is too small")
    if plan.rows != n_rows:
        plan.d_codes = None
        plan.rows = 0
        plan.d_codes = ctx.enqueue_create_buffer[DType.uint32](n_rows)
        plan.rows = n_rows
    var rm = Int32(1) if row_major else Int32(0)
    var d_codes = plan.d_codes.value().unsafe_ptr()
    # driver-loop: over the model's combination maps (model structure)
    for i in range(len(plan.map_members)):
        ctx.enqueue_function[combination_code_kernel](
            d_x.unsafe_ptr(), Int32(n_rows), Int32(plan.n_raw), rm,
            plan.d_members_i.unsafe_ptr(), plan.d_members_f.unsafe_ptr(),
            Int32(plan.map_member_off[i]), Int32(plan.map_members[i]),
            plan.d_tuples.unsafe_ptr(), Int32(plan.map_tuple_off[i]),
            Int32(plan.map_groups[i]), d_codes, d_bad.unsafe_ptr(),
            grid_dim=_grid(n_rows), block_dim=CTR_BLOCK,
        )
        # driver-loop: over the model's tree-CTR tables (those of this map; the launches after a map's code launch read its codes in queue order)
        for t in range(len(plan.table_map)):
            if plan.table_map[t] != i:
                continue
            ctx.enqueue_function[combination_value_kernel](
                d_codes, Int32(n_rows), plan.d_meta_i.unsafe_ptr(),
                plan.d_meta_f.unsafe_ptr(), plan.d_counts.unsafe_ptr(),
                Int32(t), Int32(plan.table_column[t]), d_out.unsafe_ptr(),
                grid_dim=_grid(n_rows), block_dim=CTR_BLOCK,
            )


def enqueue_combination_columns_resident(
    ctx: DeviceContext,
    mut plan: ResidentCombinationPlan,
    mut simple: ResidentCtrPlan,
    n_rows: Int,
    row_major: Bool,
    mut d_out: DeviceBuffer[DType.float32],
) raises:
    """The resident predict's call: the tree-CTR columns after the simple
    expansion on the same queue (`enqueue_resident_ctr_expand(...,
    read_bad=False)`, which uploaded the raw input into `simple.d_in` and
    filled `simple.d_bad`), then the ONE copy home of the refusal words both
    wrote. `resident_ctr_raise_if_bad` reads them after the caller's drain."""
    if simple.rows != n_rows or simple.n_raw != plan.n_raw:
        raise Error("combination columns: the simple expansion did not stage this call")
    enqueue_combination_columns(
        ctx, plan, simple.d_in.value(), n_rows, row_major, d_out, simple.d_bad
    )
    ctx.enqueue_copy(dst_ptr=simple.h_bad.unsafe_ptr(), src_buf=simple.d_bad)
