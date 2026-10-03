# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The device-resident parsed GBDT model (lane/gbdt-resident-predict,
2026-09-17, DEVIATION 2980).

`gbdt/estimator.mojo`'s header (policy choice 1) says the model crosses
the CPython boundary as its own text and that this costs a parse per
`predict` call; it names the fix, "a cached parse keyed on the text", and
asks that it be built when someone measures the cost. lane/infer-speed-
trees measured it: on an RTX 4090 the parse of a 1000-tree Logloss model
is 34 ms of a 100 ms `predict` over 1,000,000 taxi rows, and the rest of
that call is a `DeviceContext` per call, three host copies of the input
matrix, one pinned staging ring per feature and the pack and upload of
every tree's split records and leaf values.

This file is that cached parse. The FIRST `predict` or `predict_proba`
after a fit or a load hands the text to `gbdt_resident_prepare`, which
parses it once, packs the oblivious ensemble once, uploads the pack and
every feature's border list once, and keeps them in a process-wide
registry under an integer handle the Python instance holds
(`GradientBoosting._resident`). Every later call goes through
`gbdt_resident_predict` with the handle. The registry is
`core/forest_inference_model.mojo`'s FOREST-RESIDENT-1 shape and
`neighbors/resident_index.mojo`'s (DEVIATION 2921): one `_Global` per
numeric tier, monotonically increasing handles, the calling extension
holding the GIL across prepare, predict and release, and every buffer of
an entry destroyed before the context that created it (DEVIATION 1946).

WHAT DOES NOT CHANGE, AND WHY NO BIT CAN MOVE. The device work of a
resident call is the device work of `gbdt/train.mojo::predict_floats` and
`predict_multi_floats`, statement for statement:

  * the compressed index is built by the same `binarize_float_feature_kernel`
    launch per bordered feature, with the same `feature_offset`, mask, shift
    and border list, over the same float32 values after the same NaN
    substitution (`nan_substitution` per `nan_treatment`, the same refusal
    text for a NaN on an `AsIs` column); features without borders are
    skipped exactly as `_build_cindex_from_floats` skips them;
  * the cursor is filled with `Float32(model.bias)` and every oblivious
    tree is applied in tree order with the same grid. FAST and DETERMINISTIC
    use the same `compute_bins_and_add_kernel` launch per tree; IDENTICAL
    groups at most four consecutive trees while retaining the same ordered
    float32 additions and split walk
    (`gbdt/methods/doc_parallel_boosting.mojo::predict`); the split records
    and leaf values it reads are the bytes that function packs, packed once
    here instead of once per call. The one per-call word is
    `feature_offset * n_rows`, an integer product the pack multiplies in on
    the host exactly as `predict` does;
  * a non-symmetric ensemble (DEVIATION 259) is applied by that same
    `predict` function, called here with the resident compressed index, so
    its per-tree path is untouched and it still gains the cached parse;
  * the transforms are the same statements over the same readback:
    `multiclass_probabilities`'s loop body per row (the max seeded at zero
    for the pinned class, `identical_exp64`, one double division per
    class), `one_vs_all_probabilities`'s element, and for the Logloss and
    CrossEntropy pair `1 / (1 + identical_exp64(-r))` and `1 - p` in
    double, the two statements of `gbdt_sigmoid_pair` (DEVIATION 2902)
    over the exact widening of the float32 raw value; every one is per
    row, and the rows fan out to host threads.

What moves out of the call: the parse, the `DeviceContext`, the pack and
its six uploads, the border uploads, and the per-feature staging ring
with its drain per revolution. The input may arrive ROW-MAJOR (the C-order
float32 array a caller usually holds, DEVIATION 2637's shape for the
forests): the staging pass writes the column-major pinned buffer straight
from it, so the per-call native transpose the column-major contract cost
(`_buffer.as_f32_colmajor`, DEVIATION 2472) is gone. The staging pass and
the Logloss pair fan out to host threads (`MOJOLEARN_CPU_THREADS`, the
forests' `host_worker_count`); staging moves bytes and substitutes NaNs
and the pair is per row, so no bit depends on the thread count. What is retained across calls of the same
row count, exact size as FOREST-IO-REUSE-1 retains it: one pinned host
staging buffer for the input, the device input, the compressed index,
the cursor and the pinned readback; a call with a different row count
releases the set and allocates the new size. The whole input goes up in
ONE copy and the whole cursor comes back in one, and the call waits on
the stream ONCE, at the readback.

The old per-call entries (`gbdt_predict`, `gbdt_predict_multi`) stay for
older bindings and for the verifier; the Python layer takes this door
only where the loaded binding exports it.

NEGATIVE CONTROL. `-D MOJOLEARN_GBDT_RESIDENT_SABOTAGE=1` (default off)
adds 1.0 to row 0 of plane 0 of the readback before any transform, so a
build with it moves every infer and proba cell that goes through this
door and nothing that does not; the identity gate's sabotage column is
built with it.
"""
from std.ffi import _Global
from std.memory import memcpy
from std.sys.compile import is_defined

from max.gpu.host import DeviceBuffer, DeviceContext, HostBuffer
from core.neural_context import process_ctx
from checks.numerics import GLOBAL_NUMERIC_MODE as _DEVCTX_MODE, NUMERIC_IDENTICAL as _DEVCTX_IDENTICAL

#: This binding's ONE process-lifetime DeviceContext (core/neural_context.mojo,
#: lane/devctx-lifetime): a context per call exhausts Metal command queues.
comptime _DEVCTX_SLOT = "MojoGbdtContextIdentical" if _DEVCTX_MODE == _DEVCTX_IDENTICAL else "MojoGbdtContextFast"


from checks.numerics import (
    GLOBAL_NUMERIC_MODE,
    NUMERIC_IDENTICAL,
    identical_exp64,
)
from core.device_zero import enqueue_fill
from gbdt.data.quantization import NAN_TREATMENT_AS_IS, nan_substitution
from gbdt.gpu_data.compressed_index_builder import (
    CompressedIndexLayout,
    build_layout,
)
from gbdt.gpu_data.kernel.binarize import (
    BINARIZE_BLOCK_SIZE,
    BINARIZE_DOCS_PER_THREAD,
    binarize_float_feature_kernel,
)
from gbdt.methods.doc_parallel_boosting import model_approx_dim, predict
from gbdt.models.ctr_value_table import expand_raw_columns
from gbdt.models.kernel.add_bin_values import (
    compute_bins_and_add_four_kernel,
    compute_bins_and_add_kernel,
)
from gbdt.models.model_text import load_model_text
from gbdt.models.oblivious_model import BIN_SPLIT_TAKE_BIN
from gbdt.train import TrainedModel, model_input_features
from gbdt.models.kernel.resident_link import (
    LINK_BLOCK,
    STAGE_BLOCK,
    STAGE_NO_BAD,
    STAGE_SKIP,
    resident_link_kernel,
    resident_stage_kernel,
)


#: `gbdt_resident_predict`'s modes. The first three are
#: `gbdt/estimator.mojo`'s `PREDICT_RAW`, `PREDICT_SOFTMAX` and
#: `PREDICT_SIGMOID`, the same numbers; the fourth is the Logloss and
#: CrossEntropy `predict_proba` pair written as float64 `[1 - p, p]` per row
#: (`gbdt_sigmoid_pair`'s two statements, DEVIATION 2902), one-dimensional
#: models only.
comptime RESIDENT_RAW = 0
comptime RESIDENT_SOFTMAX = 1
comptime RESIDENT_SIGMOID = 2
comptime RESIDENT_SIGMOID_PAIR = 3
#: FAST classifier codes. Binary classification uses the unchanged raw cursor;
#: multiclass modes reproduce the public Float32 probability cells before
#: first-max selection so probability-rounding ties remain exact.
comptime RESIDENT_CLASSES_BINARY = 4
comptime RESIDENT_CLASSES_PINNED = 5
comptime RESIDENT_CLASSES_OVA = 6

#: One float32 slab per model column holds `[count, border_0, ...]`, the
#: layout `_build_cindex_from_floats` stages per feature into a 256-float
#: buffer; the kernel reads the count from slot 0.
comptime BORDER_SLAB = 256

comptime RESIDENT_SABOTAGE = is_defined["MOJOLEARN_GBDT_RESIDENT_SABOTAGE"]()

#: a staging or pair task takes at least this many rows, so a small
#: fixture does not fan out
comptime STAGE_MIN_ROWS_PER_TASK = 4096

#: The identity gate's control for `gbdt_sigmoid_pair`, honored on the
#: resident pair path too so that column keeps its reach over the GPU
#: proba cells (`bindings/_mojolearn_gbdt.mojo::GBDT_PAIR_SABOTAGE`).
comptime PAIR_SABOTAGE = is_defined["MOJOLEARN_FOREST_HOST_SABOTAGE"]()


struct ResidentGbdtModel(Movable):
    """One parsed model, its packed ensemble and its border lists on the
    device, plus the exact-size per-row workspace of the last call."""

    var ctx: DeviceContext
    var tm: TrainedModel
    var layout: CompressedIndexLayout
    var approx_dim: Int
    #: RAW input columns a call must hand over (`model_input_features`)
    var n_input_features: Int
    #: model columns after CTR expansion, `len(tm.fold_counts)`
    var n_columns: Int
    var needs_expansion: Bool
    var d_borders: DeviceBuffer[DType.float32]
    var oblivious: Bool
    var total_levels: Int
    var split_capacity: Int
    var total_leaves: Int
    #: `cf.offset` per level; the per-call word is this times `n_rows`
    var off_base: List[UInt32]
    var h_off: HostBuffer[DType.uint32]
    var d_off: DeviceBuffer[DType.uint32]
    var d_shift: DeviceBuffer[DType.uint32]
    var d_mask: DeviceBuffer[DType.uint32]
    var d_bin: DeviceBuffer[DType.uint32]
    var d_eq: DeviceBuffer[DType.uint8]
    var d_vals: DeviceBuffer[DType.float32]
    var workspace_rows: Int
    var h_x: Optional[HostBuffer[DType.float32]]
    var d_x: Optional[DeviceBuffer[DType.float32]]
    var d_cindex: Optional[DeviceBuffer[DType.uint32]]
    var d_cursor: Optional[DeviceBuffer[DType.float32]]
    var h_out: Optional[HostBuffer[DType.float32]]
    #: lane hr2-gbdt-host: the device staging + link workspace (raw input
    #: as handed over, per-column NaN treatment, the refusal word, and the
    #: three output planes the link writes in the caller's layout)
    var d_raw: Optional[DeviceBuffer[DType.float32]]
    var d_treat: Optional[DeviceBuffer[DType.int32]]
    var d_bad: Optional[DeviceBuffer[DType.int32]]
    var h_bad: Optional[HostBuffer[DType.int32]]
    var d_out_f32: Optional[DeviceBuffer[DType.float32]]
    var d_out_u64: Optional[DeviceBuffer[DType.uint64]]
    var d_out_i64: Optional[DeviceBuffer[DType.int64]]

    def __init__(out self, text: String) raises:
        var tm = load_model_text(text)
        # the refusals `predict_floats` makes before any device work, made
        # once here so a model this door cannot apply is refused at prepare
        if (
            len(tm.tensor_ctr_registry.features) != 0
            and len(tm.ctr_tables) != 0
        ):
            raise Error(
                "combined simple-CTR and tensor-CTR model apply needs a"
                " composed column plan and is not wired yet"
            )
        if tm.ctr_column_count != len(tm.ctr_tables):
            raise Error(
                "predict_floats cannot apply a model with "
                + String(tm.ctr_column_count)
                + " CTR columns and "
                + String(len(tm.ctr_tables))
                + " CTR tables: a CTR value is a statistic of the LEARN"
                " pool, and scoring a new row needs the final CTR tables"
                " their model file carries (ctr_data.hash_map in"
                " save_model(format='json')). Refused rather than scored"
                " against a grid the rows were never mapped onto"
            )
        var approx_dim = model_approx_dim(tm.model)
        var n_input_features = model_input_features(tm)
        var n_columns = len(tm.fold_counts)
        if len(tm.borders) != n_columns:
            raise Error(
                "fold_counts has "
                + String(n_columns)
                + " entries for "
                + String(len(tm.borders))
                + " feature border lists"
            )
        # `_build_cindex_from_floats` lays the index out from the fold
        # counts alone; `predict` lays it out with the one-hot flags, which
        # only set the predicate each level is checked against. The
        # offsets, masks and shifts agree, and this one layout serves both.
        var layout = build_layout(tm.fold_counts, tm.one_hot)
        var ctx = process_ctx[_DEVCTX_SLOT]()

        # the border slabs, uploaded once
        var h_borders = ctx.enqueue_create_host_buffer[DType.float32](
            n_columns * BORDER_SLAB
        )
        var hb = h_borders.unsafe_ptr()
        for f in range(n_columns):
            if len(tm.borders[f]) >= BORDER_SLAB:
                raise Error(
                    "feature " + String(f) + " has "
                    + String(len(tm.borders[f]))
                    + " borders; the staging slab holds "
                    + String(BORDER_SLAB - 1)
                )
            hb.unsafe_store(f * BORDER_SLAB, Float32(len(tm.borders[f])))
            for b in range(len(tm.borders[f])):
                hb.unsafe_store(f * BORDER_SLAB + 1 + b, tm.borders[f][b])
        var d_borders = ctx.enqueue_create_buffer[DType.float32](
            n_columns * BORDER_SLAB
        )
        ctx.enqueue_copy(dst_buf=d_borders, src_ptr=hb)

        # the oblivious pack, `predict`'s, once
        var oblivious = tm.model.is_oblivious()
        var total_levels = 0
        var total_leaves = 0
        if oblivious:
            for t in range(tm.model.size()):
                total_levels += tm.model.weak_models[t].structure.get_depth()
                total_leaves += (
                    (1 << tm.model.weak_models[t].structure.get_depth())
                    * approx_dim
                )
        var split_capacity = max(total_levels, 1)
        var leaf_capacity = max(total_leaves, 1)
        var off_base = List[UInt32]()
        var h_off = ctx.enqueue_create_host_buffer[DType.uint32](split_capacity)
        var h_shift = ctx.enqueue_create_host_buffer[DType.uint32](
            split_capacity
        )
        var h_mask = ctx.enqueue_create_host_buffer[DType.uint32](split_capacity)
        var h_bin = ctx.enqueue_create_host_buffer[DType.uint32](split_capacity)
        var h_eq = ctx.enqueue_create_host_buffer[DType.uint8](split_capacity)
        var h_vals = ctx.enqueue_create_host_buffer[DType.float32](
            leaf_capacity
        )
        h_off.unsafe_ptr().unsafe_store(0, UInt32(0))
        h_shift.unsafe_ptr().unsafe_store(0, UInt32(0))
        h_mask.unsafe_ptr().unsafe_store(0, UInt32(0))
        h_bin.unsafe_ptr().unsafe_store(0, UInt32(0))
        h_eq.unsafe_ptr().unsafe_store(0, UInt8(0))
        h_vals.unsafe_ptr().unsafe_store(0, Float32(0.0))
        if oblivious:
            var lvl = 0
            var leaf = 0
            for t in range(tm.model.size()):
                ref weak = tm.model.weak_models[t]
                var depth = weak.structure.get_depth()
                for level in range(depth):
                    ref cf = layout.features[
                        Int(weak.structure.splits[level].feature_id)
                    ]
                    off_base.append(cf.offset)
                    h_shift.unsafe_ptr().unsafe_store(lvl, cf.shift)
                    h_mask.unsafe_ptr().unsafe_store(lvl, cf.mask)
                    h_bin.unsafe_ptr().unsafe_store(
                        lvl, UInt32(Int(weak.structure.splits[level].bin_idx))
                    )
                    # the predicate comes off the model and the layout only
                    # confirms it (`predict`'s check, same text)
                    var take_bin = (
                        Int(weak.structure.splits[level].split_type)
                        == BIN_SPLIT_TAKE_BIN
                    )
                    if take_bin != cf.one_hot_feature:
                        raise Error(
                            "tree " + String(t) + " level " + String(level)
                            + " is a "
                            + String("TakeBin" if take_bin else "TakeGreater")
                            + " split on feature "
                            + String(Int(weak.structure.splits[level].feature_id))
                            + ", which the layout says is "
                            + String(
                                "one-hot" if cf.one_hot_feature else "ordered"
                            )
                        )
                    h_eq.unsafe_ptr().unsafe_store(
                        lvl, UInt8(1) if take_bin else UInt8(0)
                    )
                    lvl += 1
                var n_values = (1 << depth) * approx_dim
                for i in range(n_values):
                    var v = Float32(0.0)
                    if i < len(weak.leaf_values):
                        v = weak.leaf_values[i]
                    h_vals.unsafe_ptr().unsafe_store(leaf + i, v)
                leaf += n_values
        var d_off = ctx.enqueue_create_buffer[DType.uint32](split_capacity)
        var d_shift = ctx.enqueue_create_buffer[DType.uint32](split_capacity)
        var d_mask = ctx.enqueue_create_buffer[DType.uint32](split_capacity)
        var d_bin = ctx.enqueue_create_buffer[DType.uint32](split_capacity)
        var d_eq = ctx.enqueue_create_buffer[DType.uint8](split_capacity)
        var d_vals = ctx.enqueue_create_buffer[DType.float32](leaf_capacity)
        ctx.enqueue_copy(dst_buf=d_shift, src_ptr=h_shift.unsafe_ptr())
        ctx.enqueue_copy(dst_buf=d_mask, src_ptr=h_mask.unsafe_ptr())
        ctx.enqueue_copy(dst_buf=d_bin, src_ptr=h_bin.unsafe_ptr())
        ctx.enqueue_copy(dst_buf=d_eq, src_ptr=h_eq.unsafe_ptr())
        ctx.enqueue_copy(dst_buf=d_vals, src_ptr=h_vals.unsafe_ptr())
        ctx.synchronize()
        _ = h_borders^
        _ = h_shift^
        _ = h_mask^
        _ = h_bin^
        _ = h_eq^
        _ = h_vals^

        self.approx_dim = approx_dim
        self.n_input_features = n_input_features
        self.n_columns = n_columns
        self.needs_expansion = (
            len(tm.tensor_ctr_registry.features) != 0
            or len(tm.ctr_tables) != 0
        )
        self.oblivious = oblivious
        self.total_levels = total_levels
        self.split_capacity = split_capacity
        self.total_leaves = total_leaves
        self.off_base = off_base^
        self.layout = layout^
        self.tm = tm^
        self.d_borders = d_borders^
        self.h_off = h_off^
        self.d_off = d_off^
        self.d_shift = d_shift^
        self.d_mask = d_mask^
        self.d_bin = d_bin^
        self.d_eq = d_eq^
        self.d_vals = d_vals^
        self.workspace_rows = 0
        self.h_x = Optional[HostBuffer[DType.float32]]()
        self.d_x = Optional[DeviceBuffer[DType.float32]]()
        self.d_cindex = Optional[DeviceBuffer[DType.uint32]]()
        self.d_cursor = Optional[DeviceBuffer[DType.float32]]()
        self.h_out = Optional[HostBuffer[DType.float32]]()
        self.d_raw = Optional[DeviceBuffer[DType.float32]]()
        self.d_treat = Optional[DeviceBuffer[DType.int32]]()
        self.d_bad = Optional[DeviceBuffer[DType.int32]]()
        self.h_bad = Optional[HostBuffer[DType.int32]]()
        self.d_out_f32 = Optional[DeviceBuffer[DType.float32]]()
        self.d_out_u64 = Optional[DeviceBuffer[DType.uint64]]()
        self.d_out_i64 = Optional[DeviceBuffer[DType.int64]]()
        self.ctx = ctx^

    def __deinit__(deinit self):
        # every buffer before the context it was created on (DEVIATION 1946)
        _ = self.d_out_i64^
        _ = self.d_out_u64^
        _ = self.d_out_f32^
        _ = self.h_bad^
        _ = self.d_bad^
        _ = self.d_treat^
        _ = self.d_raw^
        _ = self.h_out^
        _ = self.d_cursor^
        _ = self.d_cindex^
        _ = self.d_x^
        _ = self.h_x^
        _ = self.d_vals^
        _ = self.d_eq^
        _ = self.d_bin^
        _ = self.d_mask^
        _ = self.d_shift^
        _ = self.d_off^
        _ = self.h_off^
        _ = self.d_borders^
        # DEVIATION 3010 (DEVIATION 2520's drain): the frees enqueued by the
        # releases above must complete before the context is destroyed, or
        # the runtime allocator's lock is left held and the next context's
        # first allocation never returns. Host-side drain; no output bit.
        try:
            self.ctx.synchronize()
        except:
            pass
        _ = self.ctx^

    def _prepare_workspace(mut self, n_rows: Int) raises:
        """The exact-size per-row set, retained while the row count
        repeats (FOREST-IO-REUSE-1's rule: no high-water cache, a new size
        releases the old set first)."""
        if self.workspace_rows == n_rows:
            return
        self.d_out_i64 = None
        self.d_out_u64 = None
        self.d_out_f32 = None
        self.d_raw = None
        self.h_out = None
        self.d_cursor = None
        self.d_cindex = None
        self.d_x = None
        self.h_x = None
        self.workspace_rows = 0
        try:
            self.d_x = self.ctx.enqueue_create_buffer[DType.float32](
                n_rows * self.n_columns
            )
            self.d_cindex = self.ctx.enqueue_create_buffer[DType.uint32](
                n_rows * self.layout.columns
            )
            self.d_cursor = self.ctx.enqueue_create_buffer[DType.float32](
                self.approx_dim * n_rows
            )
            self.d_raw = self.ctx.enqueue_create_buffer[DType.float32](
                n_rows * self.n_columns
            )
            self.d_out_f32 = self.ctx.enqueue_create_buffer[
                DType.float32
            ]((self.approx_dim + 1) * n_rows)
            self.d_out_u64 = self.ctx.enqueue_create_buffer[DType.uint64](
                2 * n_rows
            )
            self.d_out_i64 = self.ctx.enqueue_create_buffer[DType.int64](
                n_rows
            )
            if not self.d_treat:
                var ht = self.ctx.enqueue_create_host_buffer[DType.int32](
                    max(self.n_columns, 1)
                )
                self.ctx.synchronize()
                var htp = ht.unsafe_ptr()
                for f in range(self.n_columns):
                    var t = Int32(STAGE_SKIP)
                    if len(self.tm.borders[f]) != 0:
                        t = Int32(NAN_TREATMENT_AS_IS)
                        if len(self.tm.nan_treatment) == self.n_columns:
                            t = Int32(self.tm.nan_treatment[f])
                    htp.unsafe_store(f, t)
                var dt = self.ctx.enqueue_create_buffer[DType.int32](
                    max(self.n_columns, 1)
                )
                self.ctx.enqueue_copy(dst_buf=dt, src_buf=ht)
                self.d_treat = dt^
                self.d_bad = self.ctx.enqueue_create_buffer[DType.int32](1)
                self.h_bad = self.ctx.enqueue_create_host_buffer[
                    DType.int32
                ](1)
                self.ctx.synchronize()
                _ = ht^
        except e:
            self.ctx.synchronize()
            self.d_out_i64 = None
            self.d_out_u64 = None
            self.d_out_f32 = None
            self.d_raw = None
            self.h_out = None
            self.d_cursor = None
            self.d_cindex = None
            self.d_x = None
            self.h_x = None
            raise e
        self.workspace_rows = n_rows

    def _apply(mut self, n_rows: Int) raises:
        """`predict`'s device work over the resident pack: the cursor
        seeded at the bias, one `compute_bins_and_add_kernel` per tree in
        tree order, no drain between trees. A non-symmetric ensemble takes
        `predict` itself."""
        ref ctx = self.ctx
        if not self.oblivious:
            predict(
                self.tm.model, ctx, n_rows, self.tm.fold_counts,
                self.d_cindex.value(), self.d_cursor.value(),
                one_hot=self.tm.one_hot,
            )
            return
        enqueue_fill(ctx, self.d_cursor.value(), Float32(self.tm.model.bias))
        if self.tm.model.size() == 0:
            return
        var ho = self.h_off.unsafe_ptr()
        if self.total_levels == 0:
            ho.unsafe_store(0, UInt32(0))
        for lvl in range(self.total_levels):
            ho.unsafe_store(lvl, self.off_base[lvl] * UInt32(n_rows))
        ctx.enqueue_copy(dst_buf=self.d_off, src_ptr=ho)
        var wide = (n_rows + 255) // 256
        if wide > 1024:
            wide = 1024
        var lvl = 0
        var leaf = 0
        comptime if GLOBAL_NUMERIC_MODE != NUMERIC_IDENTICAL:
            for t0 in range(self.tm.model.size()):
                ref weak = self.tm.model.weak_models[t0]
                var depth = weak.structure.get_depth()
                var split_offset = lvl if depth > 0 else 0
                ctx.enqueue_function[compute_bins_and_add_kernel](
                    self.d_cindex.value().unsafe_ptr(),
                    self.d_off.unsafe_ptr() + split_offset,
                    self.d_shift.unsafe_ptr() + split_offset,
                    self.d_mask.unsafe_ptr() + split_offset,
                    self.d_bin.unsafe_ptr() + split_offset,
                    self.d_eq.unsafe_ptr() + split_offset,
                    Int32(depth), self.d_vals.unsafe_ptr() + leaf,
                    Int32(n_rows), self.d_cursor.value().unsafe_ptr(),
                    Int32(self.approx_dim), Int32(n_rows),
                    grid_dim=(wide, self.approx_dim, 1),
                    block_dim=(256, 1, 1),
                )
                lvl += depth
                leaf += (1 << depth) * self.approx_dim
            return
        var t = 0
        while t < self.tm.model.size():
            var count = min(4, self.tm.model.size() - t)
            var d0 = self.tm.model.weak_models[t].structure.get_depth()
            var d1 = self.tm.model.weak_models[t + 1].structure.get_depth() if count > 1 else 0
            var d2 = self.tm.model.weak_models[t + 2].structure.get_depth() if count > 2 else 0
            var d3 = self.tm.model.weak_models[t + 3].structure.get_depth() if count > 3 else 0
            var split_offset = lvl if d0 > 0 else 0
            ctx.enqueue_function[compute_bins_and_add_four_kernel](
                self.d_cindex.value().unsafe_ptr(),
                self.d_off.unsafe_ptr() + split_offset,
                self.d_shift.unsafe_ptr() + split_offset,
                self.d_mask.unsafe_ptr() + split_offset,
                self.d_bin.unsafe_ptr() + split_offset,
                self.d_eq.unsafe_ptr() + split_offset,
                Int32(d0), Int32(d1), Int32(d2), Int32(d3), Int32(count),
                self.d_vals.unsafe_ptr() + leaf,
                Int32(n_rows),
                self.d_cursor.value().unsafe_ptr(),
                Int32(self.approx_dim),
                Int32(n_rows),
                grid_dim=(wide, self.approx_dim, 1),
                block_dim=(256, 1, 1),
            )
            lvl += d0 + d1 + d2 + d3
            leaf += (
                (1 << d0) + (1 << d1 if count > 1 else 0)
                + (1 << d2 if count > 2 else 0)
                + (1 << d3 if count > 3 else 0)
            ) * self.approx_dim
            t += count

    def _predict_device(
        mut self,
        src: MutPointer[Float32, MutUntrackedOrigin],
        n_rows: Int,
        row_major: Bool,
        out_f32: MutPointer[Float32, MutUntrackedOrigin],
        out_f64: MutPointer[Float64, MutUntrackedOrigin],
        out_i64: MutPointer[Int64, MutUntrackedOrigin],
        mode: Int,
    ) raises -> Int:
        """Lane hr2-gbdt-host: the whole call on the device. The input
        goes up as handed over, `resident_stage_kernel` substitutes and
        lays it out, the binarize launches and `_apply` run as before,
        `resident_link_kernel` writes the caller's output, and ONE copy
        brings it into the caller's array; the refusal word rides the
        same drain. The mode refusals are made before any device work."""
        var dim = self.approx_dim
        var width: Int
        if mode >= RESIDENT_CLASSES_BINARY and mode <= RESIDENT_CLASSES_OVA:
            if mode == RESIDENT_CLASSES_BINARY and dim != 1:
                raise Error("binary class prediction requires one model dimension")
            width = 1
        elif mode == RESIDENT_RAW:
            width = dim
        elif mode == RESIDENT_SIGMOID_PAIR:
            if dim != 1:
                raise Error(
                    "gbdt_resident_predict: the sigmoid pair is the"
                    " one-dimensional Logloss / CrossEntropy transform; this"
                    " model has dim " + String(dim)
                )
            width = 2
        else:
            if dim < 2:
                raise Error(
                    "gbdt_predict_multi: a probability mode needs a"
                    " multi-dimensional model; this one has dim " + String(dim)
                    + ". A two-class problem's link is the sigmoid, which"
                    " Logloss's own predict_proba applies."
                )
            if mode != RESIDENT_SOFTMAX and mode != RESIDENT_SIGMOID:
                raise Error("gbdt_resident_predict: unknown mode " + String(mode))
            width = dim + 1 if mode == RESIDENT_SOFTMAX else dim
        ref ctx = self.ctx
        var n_cols = self.n_columns
        try:
            enqueue_fill(ctx, self.d_bad.value(), STAGE_NO_BAD)
            enqueue_fill(ctx, self.d_cindex.value(), UInt32(0))
            if n_rows * n_cols > 0:
                ctx.enqueue_copy(
                    dst_buf=self.d_raw.value(),
                    src_ptr=rebind[UnsafePointer[Float32, MutAnyOrigin]](src),
                )
                var total = n_rows * n_cols
                var blocks = min((total + STAGE_BLOCK - 1) // STAGE_BLOCK, 65535)
                ctx.enqueue_function[resident_stage_kernel](
                    self.d_raw.value().unsafe_ptr(),
                    self.d_x.value().unsafe_ptr(),
                    self.d_treat.value().unsafe_ptr(),
                    self.d_bad.value().unsafe_ptr(),
                    Int32(n_rows), Int32(n_cols),
                    Int32(1 if row_major else 0),
                    grid_dim=blocks, block_dim=STAGE_BLOCK,
                )
            comptime BIN_GRID = BINARIZE_BLOCK_SIZE * BINARIZE_DOCS_PER_THREAD
            for f in range(n_cols):
                if len(self.tm.borders[f]) == 0:
                    continue
                ref cf = self.layout.features[f]
                ctx.enqueue_function[binarize_float_feature_kernel](
                    Int32(Int(cf.offset) * n_rows), cf.mask, cf.shift,
                    self.d_x.value().unsafe_ptr() + f * n_rows, Int32(n_rows),
                    self.d_borders.unsafe_ptr() + f * BORDER_SLAB,
                    self.d_cindex.value().unsafe_ptr(),
                    grid_dim=(n_rows + BIN_GRID - 1) // BIN_GRID,
                    block_dim=(BINARIZE_BLOCK_SIZE, 1, 1),
                )
            self._apply(n_rows)
            var link_blocks = min((n_rows + LINK_BLOCK - 1) // LINK_BLOCK, 65535)
            ctx.enqueue_function[resident_link_kernel](
                self.d_cursor.value().unsafe_ptr(),
                self.d_out_f32.value().unsafe_ptr(),
                self.d_out_u64.value().unsafe_ptr(),
                self.d_out_i64.value().unsafe_ptr(),
                Int32(n_rows), Int32(dim), Int32(mode),
                Int32(1 if RESIDENT_SABOTAGE else 0),
                Int32(1 if PAIR_SABOTAGE else 0),
                grid_dim=link_blocks, block_dim=LINK_BLOCK,
            )
            ctx.enqueue_copy(
                dst_buf=self.h_bad.value(), src_buf=self.d_bad.value()
            )
            if mode >= RESIDENT_CLASSES_BINARY and mode <= RESIDENT_CLASSES_OVA:
                var v = self.d_out_i64.value().create_sub_buffer[DType.int64](
                    0, n_rows
                )
                ctx.enqueue_copy(
                    dst_ptr=rebind[UnsafePointer[Int64, MutAnyOrigin]](out_i64),
                    src_buf=v,
                )
                ctx.synchronize()
            elif mode == RESIDENT_SIGMOID_PAIR:
                var v2 = self.d_out_u64.value().create_sub_buffer[DType.uint64](
                    0, 2 * n_rows
                )
                ctx.enqueue_copy(
                    dst_ptr=rebind[UnsafePointer[UInt64, MutAnyOrigin]](
                        out_f64.bitcast[UInt64]()
                    ),
                    src_buf=v2,
                )
                ctx.synchronize()
            else:
                var v3 = self.d_out_f32.value().create_sub_buffer[
                    DType.float32
                ](0, width * n_rows)
                ctx.enqueue_copy(
                    dst_ptr=rebind[UnsafePointer[Float32, MutAnyOrigin]](out_f32),
                    src_buf=v3,
                )
                ctx.synchronize()
        except e:
            ctx.synchronize()
            raise e
        var first = Int(self.h_bad.value().unsafe_ptr().unsafe_load(0))
        if first != Int(STAGE_NO_BAD):
            raise Error(
                "There are NaNs in feature number " + String(first)
                + " but there were no NaNs in the learn dataset"
            )
        return width

    def predict_into(
        mut self,
        x: MutPointer[Float32, MutUntrackedOrigin],
        n_rows: Int,
        out_f32: MutPointer[Float32, MutUntrackedOrigin],
        out_f64: MutPointer[Float64, MutUntrackedOrigin],
        out_i64: MutPointer[Int64, MutUntrackedOrigin],
        mode: Int,
        row_major: Bool = False,
    ) raises -> Int:
        """One call: stage, upload once, quantize, apply, read back once,
        transform on the host. `x` holds `n_rows` rows of
        `n_input_features` raw columns, COLUMN-MAJOR unless `row_major`.
        Returns the width written per row. Raw/softmax/sigmoid write float32,
        `RESIDENT_SIGMOID_PAIR` writes float64, and the three class modes
        write int64 codes."""
        if n_rows <= 0:
            raise Error("gbdt_resident_predict: n_rows must be positive")
        var expanded = List[Float32]()
        var src = x
        var staged_row_major = row_major
        if self.needs_expansion:
            # their CalcCtrs ahead of the quantizer, the same host
            # functions `predict_floats` calls over the same column-major
            # copy; a row-major input is transposed into that copy
            var n_x = n_rows * self.n_input_features
            var xs = List[Float32]()
            xs.resize(n_x, Float32(0.0))
            if row_major:
                var nf = self.n_input_features
                for r in range(n_rows):
                    for f in range(nf):
                        xs[f * n_rows + r] = x.unsafe_load(r * nf + f)
            else:
                memcpy(dest=xs.unsafe_ptr(), src=x, count=n_x)
            staged_row_major = False
            if len(self.tm.tensor_ctr_registry.features) != 0:
                expanded = self.tm.tensor_ctr_registry.expand_for_model_apply(
                    xs, n_rows, self.tm.borders, self.tm.one_hot
                )
            else:
                expanded = expand_raw_columns(
                    self.tm.ctr_tables, self.n_columns, xs, n_rows
                )
            if len(expanded) != n_rows * self.n_columns:
                raise Error("x_colmajor size mismatch")
            src = rebind[MutPointer[Float32, MutUntrackedOrigin]](
                expanded.unsafe_ptr()
            )
        self._prepare_workspace(n_rows)
        # lane hr2-gbdt-host's device staging and link are the only route
        # (cpu-gpu-cleanup t-gbdt: the `MOJOLEARN_HR2_OLD_LINK` host staging
        # and host link across the thread pool are deleted)
        var width = self._predict_device(
            src, n_rows, staged_row_major, out_f32, out_f64, out_i64, mode
        )
        _ = len(expanded)
        return width


struct GbdtModelRegistry(Movable):
    var entries: Dict[Int, ResidentGbdtModel]
    var next_id: Int

    def __init__(out self):
        self.entries = Dict[Int, ResidentGbdtModel]()
        self.next_id = 1


comptime GBDT_MODEL_REGISTRY = _Global[
    StorageType=GbdtModelRegistry,
    name=(
        "MojoGbdtResidentModelIdentical" if GLOBAL_NUMERIC_MODE == 1 else
        "MojoGbdtResidentModelDeterministic" if GLOBAL_NUMERIC_MODE == 2 else
        "MojoGbdtResidentModelFast"
    ),
    init_fn=GbdtModelRegistry.__init__,
]


def gbdt_resident_prepare(text: String) raises -> Int:
    """Parse, pack and upload once; the handle every later call names."""
    var state = GBDT_MODEL_REGISTRY.get_or_create_ptr()
    var entry = ResidentGbdtModel(text)
    if state[].next_id == 9223372036854775807:
        raise Error("resident GBDT model handle space exhausted")
    var handle = state[].next_id
    state[].next_id += 1
    state[].entries[handle] = entry^
    return handle


def gbdt_resident_release(handle: Int) raises:
    """Drop the device copy; a released handle is refused by every later
    call."""
    var state = GBDT_MODEL_REGISTRY.get_or_create_ptr()
    if handle not in state[].entries:
        raise Error("unknown or released resident GBDT model handle")
    var released = state[].entries.pop(handle)
    _ = released^


def gbdt_resident_info(handle: Int) raises -> List[Int]:
    """`[approx_dim, n_input_features, workspace_rows, oblivious, n_trees]`
    of a live handle, for the wrapper and the harness."""
    var state = GBDT_MODEL_REGISTRY.get_or_create_ptr()
    if handle not in state[].entries:
        raise Error("unknown or released resident GBDT model handle")
    ref entry = state[].entries[handle]
    var out = List[Int]()
    out.append(entry.approx_dim)
    out.append(entry.n_input_features)
    out.append(entry.workspace_rows)
    out.append(1 if entry.oblivious else 0)
    out.append(entry.tm.model.size())
    return out^


def gbdt_resident_predict(
    handle: Int,
    x: MutPointer[Float32, MutUntrackedOrigin],
    n_rows: Int,
    out_f32: MutPointer[Float32, MutUntrackedOrigin],
    out_f64: MutPointer[Float64, MutUntrackedOrigin],
    out_i64: MutPointer[Int64, MutUntrackedOrigin],
    mode: Int,
    row_major: Bool = False,
) raises -> Int:
    """`ResidentGbdtModel.predict_into` over the handle's model. Returns
    the width written per row."""
    var state = GBDT_MODEL_REGISTRY.get_or_create_ptr()
    if handle not in state[].entries:
        raise Error("unknown or released resident GBDT model handle")
    return state[].entries[handle].predict_into(
        x, n_rows, out_f32, out_f64, out_i64, mode, row_major
    )
