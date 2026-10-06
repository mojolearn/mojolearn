# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""GPU TreeSHAP (lane gap-treeshap, 2026-10-02): the units of
xtrees/shap.mojo, one thread per unit, every stage one launch on one stream.

  prepare   parent+mark (node), depth (node), cover (tree x background row),
            slot (tree), expected-value share (tree x output), expected-value
            fold (output); cover, ev and the meta words come back once.
  values    parent+mark, slot; then per row chunk: the tree unit (tree x row)
            and the fold unit (feature x output x row); phi comes back once.

A row chunk holds as many rows as fit BUF_BYTES of (row, tree, slot, output)
cells and at most UNITS_MAX (row, tree) units, so no single launch walks an
unbounded grid (Apple aborts long command buffers). The chunking moves no
bit: every unit's chain is independent of it (xtrees/shap.mojo)."""
from std.ffi import _Global
from std.sys.compile import is_defined
from std.sys.info import has_apple_gpu_accelerator
from std.gpu import block_idx, block_dim, thread_idx
from max.gpu.host import DeviceBuffer, DeviceContext
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_FAST, NUMERIC_IDENTICAL
from core.device_zero import enqueue_fill
from xtrees.shap import (
    F32P, I32P, SHAP_META_BAD, SHAP_META_DEPTH, SHAP_META_SLOTS, SHAP_META_WORDS,
    shap_parent_unit, shap_depth_unit, shap_cover_unit, shap_slot_unit, shap_ev_part_unit, shap_ev_fold_unit,
    shap_tree_unit, shap_fold_unit, shap_table_unit, shap_table_row_unit,
)
from xtrees.shap_tab import U64P, shap_tab_rank_unit, shap_tab_need_unit, shap_tab_row_unit

comptime TPB = 128
comptime BUF_BYTES = 128 * 1024 * 1024
comptime UNITS_MAX = 1 << 20

# lane/idn-gates (2026-10-04): also the IDENTICAL default on every vendor
# (the table's terms are `shap_tree_unit`'s statements on the same operands,
# added in the same order); -D MOJOLEARN_IDN_GATES_OFF (or the _OFF below)
# restores the per-row units in IDENTICAL.
comptime SHAP_TABLE = (
    (
        (GLOBAL_NUMERIC_MODE == NUMERIC_FAST and has_apple_gpu_accelerator())
        or (GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL and not (is_defined["MOJOLEARN_IDN_GATES_OFF"]() or is_defined["MOJOLEARN_IDN_ALL_OFF"]()))
    )
    and not is_defined["MOJOLEARN_TREESHAP_FAST_TABLE_OFF"]()
)
"""FAST + Apple DEFAULT (lane fix-treeshap; M3 A/B ab-tshap-table-taxi
21.4 -> 11.8 ms, ab-tshap-table-istella 71.6 -> 25.1 ms, additivity error
identical; `-D MOJOLEARN_TREESHAP_FAST_TABLE_OFF` restores the per-row
units): the leaf table of
xtrees/shap.mojo (`shap_table_unit`, `shap_table_row_unit`). The per-row
work drops from a path rebuild plus `extend_path` and every unwound sum per
leaf (cubic in the path length) to one path walk and n table reads per
leaf. Taken when the compiled width is 8 (at most 7 merged features per
path, 128 patterns) and the table fits TABLE_BYTES; otherwise the
per-row units run. Same terms, same cells, same order: the same bits."""
comptime TABLE_BYTES = 256 * 1024 * 1024
#: SHAP_TREE_TAB (FAST + Apple, on top of SHAP_TABLE; the default since
#: 2026-10-04, rollback `-D MOJOLEARN_SHAP_TREE_TAB_OFF`): the leaf table's row unit finds
#: each leaf's pattern from ONE decision word per (row, tree) (one x read
#: and compare per internal node) instead of walking every leaf's root path
#: (xtrees/shap_tab.mojo). Source lane/apple-fast-shap@13343dd51 (commit
#: 73ceaa451), recovered 2026-10-04. Prior: never compiled on M3 (the
#: branch failed to parse at xtrees/api.mojo:715, a parameter named `out`,
#: and :811); the "21.4 ms vs board 25.1" lead in LOST_CANDIDATES is the
#: pre-table taxi number of ab-tshap-table-taxi, not a measurement of this.
#: Fixed: the old branch tabulated per leaf and pattern itself; main's
#: SHAP_TABLE now does that, so only the decision-word row unit is kept,
#: reading main's table, dead flags and leaf features. A tree with more
#: than 64 internal nodes (the word's width) takes main's row unit inside
#: the same launch: no host wait, no shape window. Same adds, same cells,
#: same order as SHAP_TABLE's row unit: the same bits.
#: OUTCOME (M3 afc_ab_def, full board size, 1 run per arm, 2026-10-04, tag
#: rab1d-shaptab): tree-shap istella 24.75 -> 19.57 ms (-21.0%), taxi 11.81 ->
#: 9.69 ms (-17.9%); max_additivity_error and output digest identical. KEEP:
#: the FAST + Apple default; the old -D name is harmless.
comptime SHAP_TREE_TAB = SHAP_TABLE and not is_defined["MOJOLEARN_SHAP_TREE_TAB_OFF"]()
# Bounded adjacent rows in one thread reuse the same tree metadata; no new
# pipeline/state overlap or attribution fold. Full SHAP caller decides speed.
# F13 SHAP M3 2026-10-06 MEASURED; keep OFF. Actual x_trees public binding,
# depth3/c2,depth6/c3,depth9/c4skew; cold B/A1.0682/1.1598/1.7360,
# repeat1.0757/1.4591/1.6713. All quality equal, paired reach2 each; one warmup
# plus one score. Regression. ab-20261006/repairs-shap-3e19f734e/F13/shap.
# Original wrong-binding failure retained; affected artifacts built fe1864e2c.
comptime SHAP_FAST_ROW_PAIR = GLOBAL_NUMERIC_MODE == NUMERIC_FAST and has_apple_gpu_accelerator() and is_defined["MOJOLEARN_SHAP_FAST_ROW_PAIR"]()
struct ShapPairAudit(Defaultable, Movable):
    var calls: Int
    def __init__(out self):
        self.calls = 0
comptime SHAP_PAIR_STATE = _Global[StorageType=ShapPairAudit, name="ShapPairAudit", init_fn=ShapPairAudit.__init__]

def shap_pair_count() raises -> Int:
    return SHAP_PAIR_STATE.get_or_create_ptr()[].calls

comptime TABLE_ACC = 64


struct _ShapContext(Defaultable, Movable):
    """ONE process-lifetime DeviceContext (the x_ann/x_metrics pattern: a
    context per call exhausts Metal's per-process command queues)."""
    var ctx: Optional[DeviceContext]

    def __init__(out self):
        self.ctx = Optional[DeviceContext]()


comptime _CTX_NAME = "MojoXTreesShapContextIdentical" if GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL else "MojoXTreesShapContextFast"
comptime X_TREES_SHAP_CONTEXT = _Global[StorageType=_ShapContext, name=_CTX_NAME, init_fn=_ShapContext.__init__]


def _ctx() raises -> DeviceContext:
    var slot = X_TREES_SHAP_CONTEXT.get_or_create_ptr()
    if not slot[].ctx:
        slot[].ctx = DeviceContext()
    return slot[].ctx.value().copy()


@always_inline
def _uid() -> Int:
    return Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)


def _grid(units: Int) -> Int:
    return (units + TPB - 1) // TPB


def parent_kernel(units: Int32, offsets: I32P, n_trees: Int32, colid: I32P, left: I32P, d: Int32, parent: I32P,
                  mark: I32P, meta: I32P):
    var u = _uid()
    if u < Int(units):
        shap_parent_unit(u, offsets, Int(n_trees), colid, left, Int(d), parent, mark, meta)


def depth_kernel(units: Int32, offsets: I32P, n_trees: Int32, parent: I32P, meta: I32P):
    var u = _uid()
    if u < Int(units):
        shap_depth_unit(u, offsets, Int(n_trees), parent, meta)


def cover_kernel(units: Int32, nb: Int32, offsets: I32P, colid: I32P, quesval: F32P, left: I32P, bg: F32P,
                 d: Int32, cover: I32P, meta: I32P):
    var u = _uid()
    if u < Int(units):
        shap_cover_unit(u, Int(nb), offsets, colid, quesval, left, bg, Int(d), cover, meta)


def slot_kernel(units: Int32, d: Int32, slot: I32P, meta: I32P):
    var u = _uid()
    if u < Int(units):
        shap_slot_unit(u, Int(d), slot, meta)


def ev_part_kernel(units: Int32, k: Int32, offsets: I32P, left: I32P, leaves: F32P, cover: I32P, tscale: F32P,
                   part: F32P):
    var u = _uid()
    if u < Int(units):
        shap_ev_part_unit(u, Int(k), offsets, left, leaves, cover, tscale, part)


def ev_fold_kernel(units: Int32, n_trees: Int32, k: Int32, part: F32P, ev: F32P):
    var u = _uid()
    if u < Int(units):
        shap_ev_fold_unit(u, Int(n_trees), Int(k), part, ev)


def tree_kernel[W: Int](units: Int32, rows: Int32, d: Int32, k: Int32, slots: Int32, offsets: I32P, colid: I32P,
                        quesval: F32P, left: I32P, leaves: F32P, parent: I32P, cover: I32P, tscale: F32P,
                        slot: I32P, x: F32P, r0: Int32, buf: F32P, meta: I32P):
    """x is the whole input; the chunk's rows start at r0 (offset here, on
    the typed pointer: a pointer rebuilt from an integer misses on Metal)."""
    var group = _uid()
    comptime if SHAP_FAST_ROW_PAIR:
        var pairs = (Int(rows) + 1) // 2
        var tree = group // pairs
        var row = (group % pairs) * 2
        if tree < Int(units) // Int(rows):
            for offset in range(2):
                if row + offset < Int(rows):
                    var u = tree * Int(rows) + row + offset
                    shap_tree_unit[W](u, Int(rows), Int(d), Int(k), Int(slots), offsets, colid, quesval, left, leaves, parent,
                                      cover, tscale, slot, x.unsafe_offset(Int(r0) * Int(d)), buf, meta)
    else:
        var u = _uid()
        if u < Int(units):
            shap_tree_unit[W](u, Int(rows), Int(d), Int(k), Int(slots), offsets, colid, quesval, left, leaves, parent,
                              cover, tscale, slot, x.unsafe_offset(Int(r0) * Int(d)), buf, meta)


def table_kernel[W: Int](units: Int32, m: Int32, nm: Int32, offsets: I32P, n_trees: Int32, colid: I32P, left: I32P,
                         parent: I32P, cover: I32P, tscale: F32P, leaf_mf: I32P, leaf_n: I32P, table: F32P,
                         dead: I32P, meta: I32P):
    var u = _uid()
    if u < Int(units):
        shap_table_unit[W](u, Int(m), Int(nm), offsets, Int(n_trees), colid, left, parent, cover, tscale, leaf_mf,
                           leaf_n, table, dead, meta)


def table_row_kernel[ACC: Int](units: Int32, rows: Int32, d: Int32, k: Int32, slots: Int32, m: Int32, nm: Int32,
                               offsets: I32P, colid: I32P, quesval: F32P, left: I32P, leaves: F32P, parent: I32P,
                               slot: I32P, leaf_mf: I32P, leaf_n: I32P, table: F32P, dead: I32P, x: F32P,
                               r0: Int32, buf: F32P, meta: I32P):
    var group = _uid()
    comptime if SHAP_FAST_ROW_PAIR:
        var pairs = (Int(rows) + 1) // 2
        var tree = group // pairs
        var row = (group % pairs) * 2
        if tree < Int(units) // Int(rows):
            for offset in range(2):
                if row + offset < Int(rows):
                    var u = tree * Int(rows) + row + offset
                    shap_table_row_unit[ACC](u, Int(rows), Int(d), Int(k), Int(slots), Int(m), Int(nm), offsets, colid, quesval,
                                             left, leaves, parent, slot, leaf_mf, leaf_n, table, dead, x.unsafe_offset(Int(r0) * Int(d)), buf,
                                             meta)
    else:
        var u = _uid()
        if u < Int(units):
            shap_table_row_unit[ACC](u, Int(rows), Int(d), Int(k), Int(slots), Int(m), Int(nm), offsets, colid, quesval,
                                     left, leaves, parent, slot, leaf_mf, leaf_n, table, dead, x.unsafe_offset(Int(r0) * Int(d)), buf,
                                     meta)


def tab_rank_kernel(units: Int32, offsets: I32P, left: I32P, rank: I32P, nint: I32P):
    var u = _uid()
    if u < Int(units):
        shap_tab_rank_unit(u, offsets, left, rank, nint)


def tab_need_kernel(units: Int32, nm: Int32, offsets: I32P, n_trees: Int32, colid: I32P, left: I32P, parent: I32P,
                    rank: I32P, leaf_mf: I32P, leaf_n: I32P, nint: I32P, needl: U64P, needr: U64P):
    var u = _uid()
    if u < Int(units):
        shap_tab_need_unit(u, Int(nm), offsets, Int(n_trees), colid, left, parent, rank, leaf_mf, leaf_n, nint,
                           needl, needr)


def tab_row_kernel[ACC: Int](units: Int32, rows: Int32, d: Int32, k: Int32, slots: Int32, m: Int32, nm: Int32,
                             offsets: I32P, colid: I32P, quesval: F32P, left: I32P, leaves: F32P, parent: I32P,
                             slot: I32P, leaf_mf: I32P, leaf_n: I32P, table: F32P, dead: I32P, nint: I32P,
                             needl: U64P, needr: U64P, x: F32P, r0: Int32, buf: F32P, meta: I32P):
    var group = _uid()
    comptime if SHAP_FAST_ROW_PAIR:
        var pairs = (Int(rows) + 1) // 2
        var tree = group // pairs
        var row = (group % pairs) * 2
        if tree < Int(units) // Int(rows):
            for offset in range(2):
                if row + offset < Int(rows):
                    var u = tree * Int(rows) + row + offset
                    shap_tab_row_unit[ACC](u, Int(rows), Int(d), Int(k), Int(slots), Int(m), Int(nm), offsets, colid, quesval,
                                           left, leaves, parent, slot, leaf_mf, leaf_n, table, dead, nint, needl, needr,
                                           x.unsafe_offset(Int(r0) * Int(d)), buf, meta)
    else:
        var u = _uid()
        if u < Int(units):
            shap_tab_row_unit[ACC](u, Int(rows), Int(d), Int(k), Int(slots), Int(m), Int(nm), offsets, colid, quesval,
                                   left, leaves, parent, slot, leaf_mf, leaf_n, table, dead, nint, needl, needr,
                                   x.unsafe_offset(Int(r0) * Int(d)), buf, meta)


def fold_kernel(units: Int32, r0: Int32, rows: Int32, n_trees: Int32, d: Int32, k: Int32, slots: Int32, slot: I32P,
                buf: F32P, phi: F32P):
    var u = _uid()
    if u < Int(units):
        shap_fold_unit(u, Int(r0), Int(rows), Int(n_trees), Int(d), Int(k), Int(slots), slot, buf, phi)


def _up_i32(ctx: DeviceContext, addr: Int, n: Int) raises -> DeviceBuffer[DType.int32]:
    var b = ctx.enqueue_create_buffer[DType.int32](n if n > 0 else 1)
    if n > 0:
        ctx.enqueue_copy(dst_buf=b, src_ptr=I32P(unsafe_from_address=addr))
    return b^


def _up_f32(ctx: DeviceContext, addr: Int, n: Int) raises -> DeviceBuffer[DType.float32]:
    var b = ctx.enqueue_create_buffer[DType.float32](n if n > 0 else 1)
    if n > 0:
        ctx.enqueue_copy(dst_buf=b, src_ptr=F32P(unsafe_from_address=addr))
    return b^


struct _Forest(Movable):
    var offsets: DeviceBuffer[DType.int32]
    var colid: DeviceBuffer[DType.int32]
    var quesval: DeviceBuffer[DType.float32]
    var left: DeviceBuffer[DType.int32]
    var leaves: DeviceBuffer[DType.float32]
    var tscale: DeviceBuffer[DType.float32]
    var parent: DeviceBuffer[DType.int32]
    var slot: DeviceBuffer[DType.int32]
    var meta: DeviceBuffer[DType.int32]

    def __init__(out self, ctx: DeviceContext, forest: List[Int], tscale: Int, d: Int, n_trees: Int, k: Int,
                 n_nodes: Int) raises:
        """Uploads the forest and runs parent+mark and slot."""
        self.offsets = _up_i32(ctx, forest[0], n_trees + 1)
        self.colid = _up_i32(ctx, forest[1], n_nodes)
        self.quesval = _up_f32(ctx, forest[2], n_nodes)
        self.left = _up_i32(ctx, forest[3], n_nodes)
        self.leaves = _up_f32(ctx, forest[4], n_nodes * k)
        self.tscale = _up_f32(ctx, tscale, n_trees)
        self.parent = ctx.enqueue_create_buffer[DType.int32](n_nodes)
        self.slot = ctx.enqueue_create_buffer[DType.int32](n_trees * d)
        self.meta = ctx.enqueue_create_buffer[DType.int32](SHAP_META_WORDS)
        enqueue_fill(ctx, self.parent, Int32(-1))
        enqueue_fill(ctx, self.slot, Int32(0))
        enqueue_fill(ctx, self.meta, Int32(0))
        ctx.enqueue_function[parent_kernel](
            Int32(n_nodes), self.offsets.unsafe_ptr(), Int32(n_trees), self.colid.unsafe_ptr(),
            self.left.unsafe_ptr(), Int32(d), self.parent.unsafe_ptr(), self.slot.unsafe_ptr(), self.meta.unsafe_ptr(),
            grid_dim=_grid(n_nodes), block_dim=TPB)
        ctx.enqueue_function[slot_kernel](
            Int32(n_trees), Int32(d), self.slot.unsafe_ptr(), self.meta.unsafe_ptr(),
            grid_dim=_grid(n_trees), block_dim=TPB)


def _check_meta(ctx: DeviceContext, meta: DeviceBuffer[DType.int32], out_addr: Int) raises:
    """Downloads the meta words (into out_addr when nonzero) and refuses a
    malformed forest."""
    var h = ctx.enqueue_create_host_buffer[DType.int32](SHAP_META_WORDS)
    ctx.enqueue_copy(dst_ptr=h.unsafe_ptr(), src_buf=meta)
    ctx.synchronize()
    var bad = Int(h.unsafe_ptr()[unsafe_offset=SHAP_META_BAD])
    if bad != 0:
        raise Error("x_trees tree_shap: malformed tree (reason " + String(bad) + ", xtrees/shap.mojo meta words)")
    if out_addr != 0:
        var o = I32P(unsafe_from_address=out_addr)
        for i in range(SHAP_META_WORDS):
            o[unsafe_offset=i] = h.unsafe_ptr()[unsafe_offset=i]
    _ = h^


def shap_prepare(forest: List[Int], tscale: Int, bg: Int, cover_out: Int, ev: Int, meta_out: Int, nb: Int, d: Int,
                 n_trees: Int, k: Int, n_nodes: Int) raises:
    """cover_out (Int32 per node) = background counts; ev (Float32 k, the
    caller's init) += the expected value; meta_out = [widest slot count,
    deepest leaf depth, 0]."""
    var ctx = _ctx()
    var fo = _Forest(ctx, forest, tscale, d, n_trees, k, n_nodes)
    var dbg = _up_f32(ctx, bg, nb * d)
    var dev = _up_f32(ctx, ev, k)
    var cover = ctx.enqueue_create_buffer[DType.int32](n_nodes)
    var part = ctx.enqueue_create_buffer[DType.float32](n_trees * k)
    enqueue_fill(ctx, cover, Int32(0))
    ctx.enqueue_function[depth_kernel](
        Int32(n_nodes), fo.offsets.unsafe_ptr(), Int32(n_trees), fo.parent.unsafe_ptr(), fo.meta.unsafe_ptr(),
        grid_dim=_grid(n_nodes), block_dim=TPB)
    var cu = n_trees * nb
    ctx.enqueue_function[cover_kernel](
        Int32(cu), Int32(nb), fo.offsets.unsafe_ptr(), fo.colid.unsafe_ptr(), fo.quesval.unsafe_ptr(),
        fo.left.unsafe_ptr(), dbg.unsafe_ptr(), Int32(d), cover.unsafe_ptr(), fo.meta.unsafe_ptr(),
        grid_dim=_grid(cu), block_dim=TPB)
    ctx.enqueue_function[ev_part_kernel](
        Int32(n_trees * k), Int32(k), fo.offsets.unsafe_ptr(), fo.left.unsafe_ptr(), fo.leaves.unsafe_ptr(),
        cover.unsafe_ptr(), fo.tscale.unsafe_ptr(), part.unsafe_ptr(),
        grid_dim=_grid(n_trees * k), block_dim=TPB)
    ctx.enqueue_function[ev_fold_kernel](
        Int32(k), Int32(n_trees), Int32(k), part.unsafe_ptr(), dev.unsafe_ptr(),
        grid_dim=_grid(k), block_dim=TPB)
    ctx.enqueue_copy(dst_ptr=I32P(unsafe_from_address=cover_out), src_buf=cover)
    ctx.enqueue_copy(dst_ptr=F32P(unsafe_from_address=ev), src_buf=dev)
    _check_meta(ctx, fo.meta, meta_out)
    _ = dbg^
    _ = part^
    _ = fo^


def _rows_per_chunk(n: Int, n_trees: Int, slots: Int, k: Int) -> Int:
    var per_row = n_trees * max(slots, 1) * k * 4
    var r = max(1, min(n, BUF_BYTES // per_row))
    return max(1, min(r, UNITS_MAX // n_trees))


def tree_shap_values(forest: List[Int], tscale: Int, cover_in: Int, x: Int, phi: Int, n: Int, d: Int, n_trees: Int,
                     k: Int, n_nodes: Int, slots: Int, width: Int) raises:
    """phi (Float32 n x d x k) = the TreeSHAP values of the n rows of x."""
    var ctx = _ctx()
    var fo = _Forest(ctx, forest, tscale, d, n_trees, k, n_nodes)
    var cover = _up_i32(ctx, cover_in, n_nodes)
    var dx = _up_f32(ctx, x, n * d)
    var dphi = ctx.enqueue_create_buffer[DType.float32](n * d * k)
    var rows = _rows_per_chunk(n, n_trees, slots, k)
    var sl = max(slots, 1)
    var buf = ctx.enqueue_create_buffer[DType.float32](rows * n_trees * sl * k)
    # the leaf table (SHAP_TABLE): NM elements, M = 2^NM patterns per node
    var nm = width - 1
    var tm = 1 << nm
    var use_table = False
    comptime if SHAP_TABLE:
        use_table = width == 8 and n_nodes * tm * (nm + 1) * 4 <= TABLE_BYTES
    var tsz = n_nodes * tm * nm if use_table else 1
    var table = ctx.enqueue_create_buffer[DType.float32](tsz)
    var dead = ctx.enqueue_create_buffer[DType.int32](n_nodes * tm if use_table else 1)
    var leaf_mf = ctx.enqueue_create_buffer[DType.int32](n_nodes * nm if use_table else 1)
    var leaf_n = ctx.enqueue_create_buffer[DType.int32](n_nodes if use_table else 1)
    comptime if SHAP_TABLE:
        if use_table:
            var tu0 = n_nodes * tm
            ctx.enqueue_function[table_kernel[8]](
                Int32(tu0), Int32(tm), Int32(nm), fo.offsets.unsafe_ptr(), Int32(n_trees), fo.colid.unsafe_ptr(),
                fo.left.unsafe_ptr(), fo.parent.unsafe_ptr(), cover.unsafe_ptr(), fo.tscale.unsafe_ptr(),
                leaf_mf.unsafe_ptr(), leaf_n.unsafe_ptr(), table.unsafe_ptr(), dead.unsafe_ptr(),
                fo.meta.unsafe_ptr(), grid_dim=_grid(tu0), block_dim=TPB)
    # SHAP_TREE_TAB: ranks, internal counts and per-element decision masks
    var tt = False
    comptime if SHAP_TREE_TAB:
        tt = use_table
    var rank = ctx.enqueue_create_buffer[DType.int32](n_nodes if tt else 1)
    var nint = ctx.enqueue_create_buffer[DType.int32](n_trees if tt else 1)
    var needl = ctx.enqueue_create_buffer[DType.uint64](n_nodes * nm if tt else 1)
    var needr = ctx.enqueue_create_buffer[DType.uint64](n_nodes * nm if tt else 1)
    comptime if SHAP_TREE_TAB:
        if tt:
            ctx.enqueue_function[tab_rank_kernel](
                Int32(n_trees), fo.offsets.unsafe_ptr(), fo.left.unsafe_ptr(), rank.unsafe_ptr(), nint.unsafe_ptr(),
                grid_dim=_grid(n_trees), block_dim=TPB)
            ctx.enqueue_function[tab_need_kernel](
                Int32(n_nodes), Int32(nm), fo.offsets.unsafe_ptr(), Int32(n_trees), fo.colid.unsafe_ptr(),
                fo.left.unsafe_ptr(), fo.parent.unsafe_ptr(), rank.unsafe_ptr(), leaf_mf.unsafe_ptr(),
                leaf_n.unsafe_ptr(), nint.unsafe_ptr(), needl.unsafe_ptr(), needr.unsafe_ptr(),
                grid_dim=_grid(n_nodes), block_dim=TPB)
    var r0 = 0
    while r0 < n:
        var rc = min(rows, n - r0)
        var tu = n_trees * rc
        var row_units = n_trees * ((rc + 1) // 2) if SHAP_FAST_ROW_PAIR else tu
        comptime if SHAP_FAST_ROW_PAIR:
            SHAP_PAIR_STATE.get_or_create_ptr()[].calls += 1
        var ran = False
        comptime if SHAP_TABLE:
            if tt:
                ran = True
                comptime if SHAP_TREE_TAB:
                    if sl * k <= TABLE_ACC:
                        ctx.enqueue_function[tab_row_kernel[TABLE_ACC]](
                            Int32(tu), Int32(rc), Int32(d), Int32(k), Int32(sl), Int32(tm), Int32(nm),
                            fo.offsets.unsafe_ptr(), fo.colid.unsafe_ptr(), fo.quesval.unsafe_ptr(),
                            fo.left.unsafe_ptr(), fo.leaves.unsafe_ptr(), fo.parent.unsafe_ptr(), fo.slot.unsafe_ptr(),
                            leaf_mf.unsafe_ptr(), leaf_n.unsafe_ptr(), table.unsafe_ptr(), dead.unsafe_ptr(),
                            nint.unsafe_ptr(), needl.unsafe_ptr(), needr.unsafe_ptr(), dx.unsafe_ptr(), Int32(r0),
                            buf.unsafe_ptr(), fo.meta.unsafe_ptr(), grid_dim=_grid(row_units), block_dim=TPB)
                    else:
                        ctx.enqueue_function[tab_row_kernel[0]](
                            Int32(tu), Int32(rc), Int32(d), Int32(k), Int32(sl), Int32(tm), Int32(nm),
                            fo.offsets.unsafe_ptr(), fo.colid.unsafe_ptr(), fo.quesval.unsafe_ptr(),
                            fo.left.unsafe_ptr(), fo.leaves.unsafe_ptr(), fo.parent.unsafe_ptr(), fo.slot.unsafe_ptr(),
                            leaf_mf.unsafe_ptr(), leaf_n.unsafe_ptr(), table.unsafe_ptr(), dead.unsafe_ptr(),
                            nint.unsafe_ptr(), needl.unsafe_ptr(), needr.unsafe_ptr(), dx.unsafe_ptr(), Int32(r0),
                            buf.unsafe_ptr(), fo.meta.unsafe_ptr(), grid_dim=_grid(row_units), block_dim=TPB)
            elif use_table:
                ran = True
                if sl * k <= TABLE_ACC:
                    ctx.enqueue_function[table_row_kernel[TABLE_ACC]](
                        Int32(tu), Int32(rc), Int32(d), Int32(k), Int32(sl), Int32(tm), Int32(nm),
                        fo.offsets.unsafe_ptr(), fo.colid.unsafe_ptr(), fo.quesval.unsafe_ptr(), fo.left.unsafe_ptr(),
                        fo.leaves.unsafe_ptr(), fo.parent.unsafe_ptr(), fo.slot.unsafe_ptr(), leaf_mf.unsafe_ptr(),
                        leaf_n.unsafe_ptr(), table.unsafe_ptr(), dead.unsafe_ptr(), dx.unsafe_ptr(), Int32(r0),
                        buf.unsafe_ptr(), fo.meta.unsafe_ptr(), grid_dim=_grid(row_units), block_dim=TPB)
                else:
                    ctx.enqueue_function[table_row_kernel[0]](
                        Int32(tu), Int32(rc), Int32(d), Int32(k), Int32(sl), Int32(tm), Int32(nm),
                        fo.offsets.unsafe_ptr(), fo.colid.unsafe_ptr(), fo.quesval.unsafe_ptr(), fo.left.unsafe_ptr(),
                        fo.leaves.unsafe_ptr(), fo.parent.unsafe_ptr(), fo.slot.unsafe_ptr(), leaf_mf.unsafe_ptr(),
                        leaf_n.unsafe_ptr(), table.unsafe_ptr(), dead.unsafe_ptr(), dx.unsafe_ptr(), Int32(r0),
                        buf.unsafe_ptr(), fo.meta.unsafe_ptr(), grid_dim=_grid(row_units), block_dim=TPB)
        comptime for wi in range(6):
            comptime W = 8 << wi
            if not ran and width == W:
                ctx.enqueue_function[tree_kernel[W]](
                    Int32(tu), Int32(rc), Int32(d), Int32(k), Int32(sl), fo.offsets.unsafe_ptr(),
                    fo.colid.unsafe_ptr(), fo.quesval.unsafe_ptr(), fo.left.unsafe_ptr(), fo.leaves.unsafe_ptr(),
                    fo.parent.unsafe_ptr(), cover.unsafe_ptr(), fo.tscale.unsafe_ptr(), fo.slot.unsafe_ptr(),
                    dx.unsafe_ptr(), Int32(r0), buf.unsafe_ptr(), fo.meta.unsafe_ptr(), grid_dim=_grid(row_units),
                    block_dim=TPB)
        var fu = d * k * rc
        ctx.enqueue_function[fold_kernel](
            Int32(fu), Int32(r0), Int32(rc), Int32(n_trees), Int32(d), Int32(k), Int32(sl), fo.slot.unsafe_ptr(),
            buf.unsafe_ptr(), dphi.unsafe_ptr(), grid_dim=_grid(fu), block_dim=TPB)
        r0 += rc
    ctx.enqueue_copy(dst_ptr=F32P(unsafe_from_address=phi), src_buf=dphi)
    _check_meta(ctx, fo.meta, 0)
    _ = rank^
    _ = nint^
    _ = needl^
    _ = needr^
    _ = table^
    _ = dead^
    _ = leaf_mf^
    _ = leaf_n^
    _ = buf^
    _ = dx^
    _ = cover^
    _ = fo^
