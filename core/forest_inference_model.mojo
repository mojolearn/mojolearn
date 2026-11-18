# SPDX-License-Identifier: Apache-2.0
"""Owned GPU inference snapshots, shared by RF/ET binding registries.

nvForest26.08 cef3a50d caches an owning inference model; cuML26.08
randomforest_common.pyx:675-693 retains that model across predictions.
FOREST-RESIDENT-1: explicit monotonically increasing registry IDs replace the
Cython owning-object boundary. Python holds the GIL across these operations.
Snapshots own model buffers and a context; release drops buffers before context.
Only input validation/upload and result readback recur per prediction.
"""
from std.sys.compile import is_defined
from std.memory import bitcast
from std.time import perf_counter_ns
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_FAST, NUMERIC_IDENTICAL
from checks.kernel_matrix import TARGET_COLUMN, COLUMN_APPLE, COLUMN_NVIDIA, COLUMN_AMD
from max.gpu.host import DeviceContext, DeviceBuffer, HostBuffer
from core.forest_inference import (
    _forest_shape_checks, forest_pack_device, launch_forest_inference, launch_forest_argmax, FOREST_PACKED_NODES,
    device_all_finite,
)
from core.forest_inference_pool import PooledForest, forest_device_count
from core.neural_context import process_ctx
from core.admitted_finite_status import enqueue_finite_status
# TOMBSTONE: MOJOLEARN_TREES_T34_CHUNK_FOLD (opt-in arm, no A/B recorded) deleted 2026-11-15 by lane/expdel2-04-base; code recoverable at 08081353e1.
# Restore: git apply experiments/removed/MOJOLEARN_TREES_T34_CHUNK_FOLD.expdel2-04-base.patch; record in docs/TOMBSTONES.md.
# TOMBSTONE: MOJOLEARN_TREES_T38_FUSED_LABELS (opt-in arm, no A/B recorded) deleted 2026-11-15 by lane/expdel2-04-base; code recoverable at 1caa34cfb3.
# Restore: git apply experiments/removed/MOJOLEARN_TREES_T38_FUSED_LABELS.expdel2-04-base.patch; record in docs/TOMBSTONES.md.
# TOMBSTONE: MOJOLEARN_TREES_T37_WORKSPACE (opt-in arm, no A/B recorded) deleted 2026-11-15 by lane/expdel2-04-base; code recoverable at 9d8b51b0a6.
# Restore: git apply experiments/removed/MOJOLEARN_TREES_T37_WORKSPACE.expdel2-04-base.patch; record in docs/TOMBSTONES.md.
from core.shared_slot import SharedSlot


#: The resident path checks its input and output for non-finite values on
#: the device where they land (`device_all_finite`, core/forest_inference);
#: the host scans of DEVIATION 2962 are gone (cpu-gpu-cleanup); the upload
#: reads the caller's rows. `MOJOLEARN_FOREST_PROFILE=1` prints one
#: line per call with the nanoseconds of each stage, the stream drained
#: between stages so the numbers attribute; it is a diagnostic build, never
#: a timed arm.
# TOMBSTONE: MOJOLEARN_FOREST_PINNED_STAGE (opt-in arm, no A/B recorded) deleted 2026-11-15 by lane/expdel2-04-base; code recoverable at 28a3c1be2b.
# Restore: git apply experiments/removed/MOJOLEARN_FOREST_PINNED_STAGE.patch; record in docs/TOMBSTONES.md.
comptime FOREST_PROFILE = is_defined["MOJOLEARN_FOREST_PROFILE"]()

#: gap-fails2 (2026-10-02): every resident snapshot of one registry (RF or
#: ET, per numeric tier) runs on ONE process-lifetime context
#: (`process_ctx`) and borrows ONE input/output workspace (`ForestIOWorkspace`,
#: exact size, released with the registry's last snapshot) instead of a
#: context and an X-sized device workspace per snapshot. AdaBoost keeps 50
#: one-tree members alive: on istella (1,000,000 x 220, 880 MB) the 0.8.34
#: NVIDIA board held 47 X-sized workspaces, 41 GB, and the next fit ran the
#: L40S out of memory. Calls are synchronous under the GIL, so one workspace
#: serves every snapshot; same kernels, same launches, no bit moves.
# TOMBSTONE: MOJOLEARN_FOREST_PER_MODEL_IO (opt-in arm, no A/B recorded) deleted 2026-11-15 by lane/expdel2-04-base; code recoverable at c9c333f2d7.
# Restore: git apply experiments/removed/MOJOLEARN_FOREST_PER_MODEL_IO.patch; record in docs/TOMBSTONES.md.
def forest_ordered_resident_policy[
    column: Int, mode: Int, forced: Bool, disabled: Bool
]() -> Bool:
    """Strict ordered resident defaults on every supported GPU vendor."""
    return not disabled and (
        ((mode == NUMERIC_FAST or mode == NUMERIC_IDENTICAL) and (
            column == COLUMN_APPLE
            or column == COLUMN_NVIDIA
            or column == COLUMN_AMD
        ))
        or (mode == NUMERIC_IDENTICAL and forced)
    )


#: Retain the resident model/workspaces while launching the strict
#: increasing-tree kernel. H100 full-buffer qualification promotes this for
#: Apple, NVIDIA and AMD in FAST/IDENTICAL.
#: Since 2026-09-22 this is the DEFAULT of a per-snapshot choice
#: (`ResidentForest.ordered`, forest_prepare_gpu's optional 4th param): an
#: IDENTICAL caller names it, strict for `auto` (the sequential bits) and
#: the 32-grove graph for an explicit `parallel_groves` (the recorded fold,
#: which the CPU host groves engine also computes). As a bare compiled
#: default it moved recorded IDENTICAL parallel_groves cells on every GPU.
# TOMBSTONE: MOJOLEARN_FOREST_ORDERED_RESIDENT (opt-in arm, no A/B recorded) deleted 2026-11-15 by lane/expdel2-04-base; code recoverable at 39d9cbb242.
# Restore: git apply experiments/removed/MOJOLEARN_FOREST_ORDERED_RESIDENT.patch; record in docs/TOMBSTONES.md.
comptime FOREST_ORDERED_RESIDENT = forest_ordered_resident_policy[
    TARGET_COLUMN,
    GLOBAL_NUMERIC_MODE,
    False,
    False,
]()
# TOMBSTONE: MOJOLEARN_FOREST_ORDERED_RESIDENT_OFF (old path, no A/B recorded beside the default) deleted 2026-11-15 by lane/expdel2-04-base; code recoverable at 057d3cc56b.
# Restore: git apply experiments/removed/MOJOLEARN_FOREST_ORDERED_RESIDENT_OFF.patch; record in docs/TOMBSTONES.md.


struct ForestIOWorkspace(Defaultable, Movable):
    """The input/output workspace every shared-context resident snapshot of
    one registry borrows. Exact size:
    a call of another size releases and reallocates, as a snapshot's own
    workspace did (FOREST-IO-REUSE-1)."""
    var input: Optional[DeviceBuffer[DType.float32]]
    var output: Optional[DeviceBuffer[DType.float32]]
    var label: Optional[DeviceBuffer[DType.int32]]
    var input_len: Int
    var output_len: Int
    var label_len: Int
    #: The staged input (lane perf-sweep-trees, 2026-11-05): while
    #: `staged_len` > 0, the first `staged_len` values of `input` are the
    #: caller's host rows at `staged_addr`, uploaded and scanned finite once
    #: by `stage`. An ensemble's public call stages its X before its members
    #: predict and unstages after; each member's resident prediction on those
    #: same rows then skips its own upload and input scan. Before it every
    #: member copied X to the device again (AdaBoost's 50 one-tree members,
    #: 50 uploads of the same rows). Same kernels, same inputs: no bit moves.
    var staged_addr: Int
    var staged_len: Int

    def __init__(out self):
        self.input = Optional[DeviceBuffer[DType.float32]]()
        self.output = Optional[DeviceBuffer[DType.float32]]()
        self.label = Optional[DeviceBuffer[DType.int32]]()
        self.input_len = 0
        self.output_len = 0
        self.label_len = 0
        self.staged_addr = 0
        self.staged_len = 0

    def release(mut self):
        self.input = None
        self.output = None
        self.label = None
        self.input_len = 0
        self.output_len = 0
        self.label_len = 0
        self.staged_addr = 0
        self.staged_len = 0

    def _input_fits(self, n_in: Int) -> Bool:
        return self.input_len == n_in

    def is_staged(self, addr: Int, n: Int) -> Bool:
        """True when the input buffer holds the staged rows `addr`, `n`."""
        return n > 0 and self.staged_len == n and self.staged_addr == addr and self.input_len >= n

    def unstage(mut self):
        self.staged_addr = 0
        self.staged_len = 0

    def stage_input(mut self, ctx: DeviceContext, x: MutPointer[Float32, MutAnyOrigin], n: Int) raises -> Bool:
        """Upload `n` host values at `x` into the input buffer and scan them
        on the device; True when they are staged. Non-finite rows are not
        staged: each member then uploads and refuses them itself, with its
        own words."""
        self.unstage()
        if n <= 0:
            return False
        if not self._input_fits(n):
            self.input = None
            self.input_len = 0
            try:
                self.input = ctx.enqueue_create_buffer[DType.float32](n)
            except e:
                ctx.synchronize()
                self.input = None
                raise e
            self.input_len = n
        try:
            ctx.enqueue_copy(dst_buf=self.input.value().create_sub_buffer[DType.float32](0, n), src_ptr=x)
            if not device_all_finite(ctx, self.input.value(), n):
                ctx.synchronize()
                return False
            ctx.synchronize()
        except e:
            ctx.synchronize()
            raise e
        self.staged_addr = Int(x)
        self.staged_len = n
        return True

    def prepare(mut self, ctx: DeviceContext, n_in: Int, n_out: Int) raises:
        if self.input_len == n_in and self.output_len == n_out:
            return
        if self.staged_len > 0 and self.staged_len == n_in and self._input_fits(n_in):
            # Keep the staged rows: only the output buffer is resized.
            self.output = None
            self.output_len = 0
            try:
                self.output = ctx.enqueue_create_buffer[DType.float32](n_out)
            except e:
                ctx.synchronize()
                self.output = None
                raise e
            self.output_len = n_out
            return
        self.release()
        try:
            self.input = ctx.enqueue_create_buffer[DType.float32](n_in)
            self.output = ctx.enqueue_create_buffer[DType.float32](n_out)
        except e:
            ctx.synchronize()
            self.release()
            raise e
        self.input_len = n_in
        self.output_len = n_out

    def prepare_label(mut self, ctx: DeviceContext, rows: Int) raises:
        if self.label_len == rows:
            return
        self.label = None
        self.label_len = 0
        self.label = ctx.enqueue_create_buffer[DType.int32](rows)
        self.label_len = rows


comptime RF_IO = SharedSlot[ForestIOWorkspace, ("MojolearnRFResidentIOIdenticalV1" if GLOBAL_NUMERIC_MODE == 1 else "MojolearnRFResidentIOFastV1")]
comptime ET_IO = SharedSlot[ForestIOWorkspace, ("MojolearnETResidentIOIdenticalV1" if GLOBAL_NUMERIC_MODE == 1 else "MojolearnETResidentIOFastV1")]
comptime RF_CTX_SLOT = (
    "MojolearnRFResidentContextIdenticalV3" if GLOBAL_NUMERIC_MODE == 1 else
    "MojolearnRFResidentContextFastV3")
comptime ET_CTX_SLOT = (
    "MojolearnETResidentContextIdenticalV3" if GLOBAL_NUMERIC_MODE == 1 else
    "MojolearnETResidentContextFastV3")


struct ResidentForest(Movable):
    var pool: Optional[PooledForest]
    var ctx: Optional[DeviceContext]
    var offsets: Optional[DeviceBuffer[DType.int32]]
    var columns: Optional[DeviceBuffer[DType.int32]]
    var thresholds: Optional[DeviceBuffer[DType.float32]]
    var left: Optional[DeviceBuffer[DType.int32]]
    var leaves: Optional[DeviceBuffer[DType.float32]]
    var input_workspace: Optional[DeviceBuffer[DType.float32]]
    var output_workspace: Optional[DeviceBuffer[DType.float32]]
    var label_workspace: Optional[DeviceBuffer[DType.int32]]
    #: Lane flag-trees-infer-iforest (2026-11-15): the two-word finite status
    #: (word 0 the input, word 1 the output) and its pinned host word pair,
    #: held from the first predict. `_predict_into_buffers` created both on
    #: every call (25bb542958, 199f170c73): one device and one PINNED host
    #: allocation per predict, and the pinned free at the end of each call.
    var status_dev: Optional[DeviceBuffer[DType.int32]]
    var status_host: Optional[HostBuffer[DType.int32]]
    var workspace_rows: Int
    var features: Int
    var outputs: Int
    var trees: Int
    #: True launches the strict increasing-tree kernel (the sequential
    #: route's arithmetic); False the 32-grove graph. Chosen per snapshot so
    #: an IDENTICAL `parallel_groves` model keeps the grove fold it was
    #: recorded with (and that the CPU host groves engine reproduces) while
    #: IDENTICAL `auto` keeps the sequential bits on a resident snapshot.
    var ordered: Bool
    #: True: `ctx` is the registry's process context and the I/O workspace
    #: is the registry's `ForestIOWorkspace`.
    var shared: Bool

    def __init__(out self, offsets: List[Int32], columns: List[Int32],
        thresholds: List[Float32], left: List[Int32], leaves: List[Float32],
        features: Int, outputs: Int, ordered: Bool = FOREST_ORDERED_RESIDENT,
        shared_ctx: Optional[DeviceContext] = None) raises:
        var empty = List[Float32]()
        self.shared = False
        # constant-time shape checks here; the per-node checks run on the
        # device with the upload (`forest_pack_device` / `forest_validate_device`)
        _forest_shape_checks(offsets, columns, thresholds, left, leaves, empty, 0, features, outputs)
        self.pool = Optional[PooledForest]()
        self.input_workspace = Optional[DeviceBuffer[DType.float32]]()
        self.output_workspace = Optional[DeviceBuffer[DType.float32]]()
        self.label_workspace = Optional[DeviceBuffer[DType.int32]]()
        self.status_dev = Optional[DeviceBuffer[DType.int32]]()
        self.status_host = Optional[HostBuffer[DType.int32]]()
        self.workspace_rows = 0
        self.features = features
        self.outputs = outputs
        self.trees = len(offsets) - 1
        self.ordered = ordered
        self.ctx = Optional[DeviceContext]()
        self.offsets = Optional[DeviceBuffer[DType.int32]]()
        self.columns = Optional[DeviceBuffer[DType.int32]]()
        self.thresholds = Optional[DeviceBuffer[DType.float32]]()
        self.left = Optional[DeviceBuffer[DType.int32]]()
        self.leaves = Optional[DeviceBuffer[DType.float32]]()
        var device_count = forest_device_count()
        if len(columns) > 2147483647 // 4:
            raise Error("packed forest node word count exceeds Int32")
        # A sharded pool combines per-device tree partitions and therefore
        # cannot express one global increasing-tree fold.  The experimental
        # ordered arm stays on one device so its arithmetic graph is exact.
        if device_count > 1 and not ordered:
            # the whole model is validated on owner 0's device before it is
            # partitioned by grove there (`PooledForest`, lane cpu4-misc)
            self.pool = PooledForest(offsets, columns, thresholds, left, leaves,
                features, outputs, device_count)
            return
        # DEVIATION BLOCK FOREST-PACKED-1 (experimental, no speed claim):
        # nvForest cef3a50d detail/node.hpp:81-175 packs node fields; builder
        # detail/decision_forest_builder.hpp:135-149 stores only leaf vectors.
        # Keep our sibling node order/local child IDs and raw <= policy;
        # do not adopt depth-first offsets or converted thresholds here.
        # Archive arrays are unchanged. Packing runs once per resident snapshot.
        # Packing runs once per resident snapshot, on the device
        # (`forest_pack_device`, lane cpu3-core: no host loop over nodes).
        if shared_ctx:
            self.ctx = shared_ctx.value().copy()
            self.shared = True
        else:
            self.ctx = DeviceContext()
        try:
            self.offsets = self.ctx.value().enqueue_create_buffer[DType.int32](len(offsets))
            self.ctx.value().enqueue_copy(dst_buf=self.offsets.value(), src_ptr=offsets.unsafe_ptr())
            forest_pack_device(
                self.ctx.value(), offsets, columns, thresholds, left, leaves,
                features, outputs, self.columns, self.leaves,
            )
            # Unused ABI operands; avoid retaining original SoA buffers.
            self.thresholds = self.ctx.value().enqueue_create_buffer[DType.float32](1)
            self.left = self.ctx.value().enqueue_create_buffer[DType.int32](1)
            self.ctx.value().synchronize()
            _ = len(offsets)
            _ = len(columns)
            _ = len(thresholds)
            _ = len(left)
            _ = len(leaves)
        except e:
            self.close()
            raise e

    def __deinit__(deinit self):
        # Predict/prepare are synchronous; destroy GPU operands before context
        # even when an upload/allocation exception bypasses explicit release.
        _ = self.pool^
        _ = self.status_host^
        _ = self.status_dev^
        _ = self.output_workspace^
        _ = self.label_workspace^
        _ = self.input_workspace^
        _ = self.leaves^
        _ = self.left^
        _ = self.thresholds^
        _ = self.columns^
        _ = self.offsets^
        # DEVIATION 2520's drain (see `close` below): the frees the releases
        # above enqueued must complete before the context is destroyed.
        if self.ctx:
            try:
                self.ctx.value().synchronize()
            except:
                pass
        _ = self.ctx^

    def close(mut self) raises:
        self.pool = None
        if self.ctx:
            self.ctx.value().synchronize()
        self.status_host = None
        self.status_dev = None
        self.output_workspace = None
        self.label_workspace = None
        self.input_workspace = None
        self.workspace_rows = 0
        self.leaves = None
        self.left = None
        self.thresholds = None
        self.columns = None
        self.offsets = None
        # DEVIATION 3010: the drain of DEVIATION 2520, which
        # `bindings/_mojolearn_byte_lm.mojo` has carried since 2026-09-11,
        # applied to the resident forest. The `synchronize()` above drains
        # the last PREDICTION; the ten releases between it and here enqueue
        # the snapshot's and the workspace pair's buffer FREES on the same
        # stream, and destroying the context with those frees in flight left
        # the MAX runtime allocator's lock held on an RTX 4090 (sm_89). The
        # next `DeviceContext()` in the process then blocked forever inside
        # its first `enqueue_create_buffer`, which is
        # `ResidentForest.__init__` for the second model: release then
        # prepare hung, two live snapshots did not. Host-side drain only; no
        # kernel, no arithmetic, no output bit.
        if self.ctx:
            self.ctx.value().synchronize()
        self.ctx = None

    def predict[RF_INPUT: Bool](mut self, x: List[Float32], rows: Int,
        features: Int, outputs: Int) raises -> List[Float32]:
        if features != self.features or outputs != self.outputs:
            raise Error("resident forest dimensions differ from prepared snapshot")
        if rows < 0 or rows > 2147483647 // features or rows > 2147483647 // outputs:
            raise Error("resident forest prediction dimensions exceed Int32")
        if len(x) != rows * features:
            raise Error("resident forest input shape mismatch")
        if rows == 0:
            return List[Float32]()
        if self.pool:
            var result = List[Float32](length=rows * outputs, fill=Float32(0.0))
            self.pool.value().predict_into[RF_INPUT](
                rebind[MutPointer[Float32, MutAnyOrigin]](x.unsafe_ptr()),
                result.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin](), rows)
            _ = len(x)
            return result^
        var dx = self.ctx.value().enqueue_create_buffer[DType.float32](len(x))
        var dout = self.ctx.value().enqueue_create_buffer[DType.float32](rows * outputs)
        var out_ok = True
        # the answer lands straight in the result list (no host copy loop)
        var result = List[Float32](length=rows * outputs, fill=Float32(0.0))
        try:
            self.ctx.value().enqueue_copy(dst_buf=dx, src_ptr=x.unsafe_ptr())
            if not device_all_finite(self.ctx.value(), dx, rows * features):
                raise Error("resident forest requires finite Float32 values")
            if self.ordered:
                launch_forest_inference[RF_INPUT, False, FOREST_PACKED_NODES](
                    self.ctx.value(), self.offsets.value(), self.columns.value(),
                    self.thresholds.value(), self.left.value(), self.leaves.value(),
                    dx, dout, rows, features, outputs, self.trees,
                )
            else:
                launch_forest_inference[RF_INPUT, True, FOREST_PACKED_NODES](
                    self.ctx.value(), self.offsets.value(), self.columns.value(),
                    self.thresholds.value(), self.left.value(), self.leaves.value(),
                    dx, dout, rows, features, outputs, self.trees,
                )
            out_ok = device_all_finite(self.ctx.value(), dout, rows * outputs)
            self.ctx.value().enqueue_copy(dst_ptr=result.unsafe_ptr(), src_buf=dout)
            self.ctx.value().synchronize()
        except e:
            # A launch may already reference the temporary buffers or X.
            self.ctx.value().synchronize()
            raise e
        if not out_ok:
            raise Error("resident forest requires finite Float32 values")
        _ = len(x)
        _ = dx^
        _ = dout^
        return result^

    def predict_into[RF_INPUT: Bool](mut self,
        x: MutPointer[Float32, MutAnyOrigin], output: MutPointer[Float32, MutAnyOrigin],
        rows: Int, features: Int, outputs: Int,
        reuse_io: Bool = False) raises:
        """Borrowed synchronous I/O; caller retains buffers and holds the GIL.

        nvForest26.08 forest_model.hpp:284-308 wraps borrowed I/O pointers.
        Here host input still uploads, but no intermediate host Lists are made.
        Output may be written before a nonfinite-input or nonfinite-result
        error is raised: both tests read back with the output under one
        synchronize (`_predict_into_buffers`).
        """
        if features != self.features or outputs != self.outputs:
            raise Error("resident forest dimensions differ from prepared snapshot")
        if rows < 0 or rows > 2147483647 // features or rows > 2147483647 // outputs:
            raise Error("resident forest prediction dimensions exceed Int32")
        if rows == 0:
            return
        if self.pool:
            self.pool.value().predict_into[RF_INPUT](x, output, rows)
            return
        self._ensure_status()
        if reuse_io and self.shared:
            var io = RF_IO.get_or_create_ptr()
            comptime if not RF_INPUT:
                io = ET_IO.get_or_create_ptr()
            io[].prepare(self.ctx.value(), rows * features, rows * outputs)
            var uploaded = io[].is_staged(Int(x), rows * features)
            if not uploaded:
                # Other rows overwrite the input buffer: nothing stays staged.
                io[].unstage()
            _predict_into_buffers[RF_INPUT](self.ctx.value(), self.offsets.value(),
                self.columns.value(), self.thresholds.value(), self.left.value(),
                self.leaves.value(), x, output, rows, features, outputs, self.trees,
                io[].input.value(), io[].output.value(), x, False,
                self.ordered, uploaded, self.status_dev.value(), self.status_host.value())
        elif reuse_io:
            self.prepare_workspace(rows)
            _predict_into_buffers[RF_INPUT](self.ctx.value(), self.offsets.value(),
                self.columns.value(), self.thresholds.value(), self.left.value(),
                self.leaves.value(), x, output, rows, features, outputs, self.trees,
                self.input_workspace.value(), self.output_workspace.value(), x, False,
                self.ordered, False, self.status_dev.value(), self.status_host.value())
        else:
            var dx = self.ctx.value().enqueue_create_buffer[DType.float32](rows * features)
            var dout = self.ctx.value().enqueue_create_buffer[DType.float32](rows * outputs)
            _predict_into_buffers[RF_INPUT](self.ctx.value(), self.offsets.value(),
                self.columns.value(), self.thresholds.value(), self.left.value(),
                self.leaves.value(), x, output, rows, features, outputs, self.trees, dx, dout, x, False,
                self.ordered, False, self.status_dev.value(), self.status_host.value())
            _ = dx^
            _ = dout^

    def _ensure_status(mut self) raises:
        """The held two-word finite status pair (`status_dev`,
        `status_host`), created at the first predict of this snapshot; every
        later call reuses it (`_predict_into_buffers` fills word 0 and 1
        before it reads them back)."""
        if self.status_dev and self.status_host:
            return
        self.status_dev = self.ctx.value().enqueue_create_buffer[DType.int32](2)
        self.status_host = self.ctx.value().enqueue_create_host_buffer[DType.int32](2)

    def prepare_workspace(mut self, rows: Int) raises:
        # DEVIATION BLOCK FOREST-IO-REUSE-1:
        # nvForest cef3a50d forest_model.hpp:284-308 borrows caller-owned GPU
        # buffers; our host-buffer boundary requires host/device copies. Retain one
        # exact-size pair, avoiding two device allocations on equal-size calls.
        # No high-water cache: resizing releases the previous pair. Public
        # parallel_groves selects reuse after the September 10 large-data gate:
        # CUDA IDENTICAL RF/HIGGS throughput 21.485 -> 21.094 ms/call;
        # ET/HIGGS single calls 25.926 -> 24.510 ms. See forest_io_reuse results.
        # Calls are synchronous and the binding holds the GIL throughout.
        if self.workspace_rows == rows:
            return
        self.input_workspace = None
        self.output_workspace = None
        self.label_workspace = None
        self.workspace_rows = 0
        try:
            self.input_workspace = self.ctx.value().enqueue_create_buffer[DType.float32](rows * self.features)
            self.output_workspace = self.ctx.value().enqueue_create_buffer[DType.float32](rows * self.outputs)
        except e:
            self.ctx.value().synchronize()
            self.input_workspace = None
            self.output_workspace = None
            self.label_workspace = None
            raise e
        self.workspace_rows = rows

    def _labels_on[RF_INPUT: Bool](mut self, mut dx: DeviceBuffer[DType.float32],
        mut dout: DeviceBuffer[DType.float32], mut dlab: DeviceBuffer[DType.int32],
        x: MutPointer[Float32, MutAnyOrigin], output: MutPointer[Int32, MutAnyOrigin],
        rows: Int, features: Int, outputs: Int, uploaded: Bool = False) raises:
        """`predict_labels`' device leg on a given workspace. `uploaded`:
        `dx` already holds these rows, scanned finite (the staged input)."""
        if not uploaded:
            self.ctx.value().enqueue_copy(dst_buf=dx.create_sub_buffer[DType.float32](0, rows*features), src_ptr=x)
            if not device_all_finite(self.ctx.value(), dx, rows * features):
                raise Error("resident forest requires finite Float32 values")
        if self.ordered:
            launch_forest_inference[RF_INPUT, False, FOREST_PACKED_NODES](
                self.ctx.value(), self.offsets.value(), self.columns.value(),
                self.thresholds.value(), self.left.value(), self.leaves.value(),
                dx, dout, rows, features, outputs, self.trees,
            )
        else:
            launch_forest_inference[RF_INPUT, True, FOREST_PACKED_NODES](
                self.ctx.value(), self.offsets.value(), self.columns.value(),
                self.thresholds.value(), self.left.value(), self.leaves.value(),
                dx, dout, rows, features, outputs, self.trees,
            )
        if not device_all_finite(self.ctx.value(), dout, rows * outputs):
            raise Error("resident forest requires finite Float32 values")
        launch_forest_argmax(self.ctx.value(), dout, dlab, rows, outputs)
        self.ctx.value().enqueue_copy(dst_ptr=output, src_buf=dlab.create_sub_buffer[DType.int32](0, rows))
        self.ctx.value().synchronize()

    def predict_labels[RF_INPUT: Bool](mut self,
        x: MutPointer[Float32, MutAnyOrigin], output: MutPointer[Int32, MutAnyOrigin],
        rows: Int, features: Int, outputs: Int) raises:
        """FAST classifier boundary: keep vote rows on device and return codes."""
        if features != self.features or outputs != self.outputs or outputs < 2:
            raise Error("resident forest dimensions differ from prepared snapshot")
        if rows < 0 or rows > 2147483647 // features or rows > 2147483647 // outputs:
            raise Error("resident forest prediction dimensions exceed Int32")
        if self.pool:
            self.pool.value().predict_labels_into[RF_INPUT](x, output, rows)
            return
        if rows == 0:
            return
        if self.shared:
            var io = RF_IO.get_or_create_ptr()
            comptime if not RF_INPUT:
                io = ET_IO.get_or_create_ptr()
            io[].prepare(self.ctx.value(), rows * features, rows * outputs)
            io[].prepare_label(self.ctx.value(), rows)
            var uploaded = io[].is_staged(Int(x), rows * features)
            if not uploaded:
                # Other rows overwrite the input buffer: nothing stays staged.
                io[].unstage()
            try:
                self._labels_on[RF_INPUT](io[].input.value(), io[].output.value(),
                                          io[].label.value(), x, output, rows, features, outputs, uploaded)
            except e:
                self.ctx.value().synchronize()
                raise e
        else:
            self.prepare_workspace(rows)
            if not self.label_workspace:
                self.label_workspace = self.ctx.value().enqueue_create_buffer[DType.int32](rows)
            # Move the three workspaces out while the model borrows its other
            # buffers. Restore ownership after the synchronous operation on both
            # paths; no second mutable borrow through self reaches _labels_on.
            var dx = self.input_workspace.take()
            var dout = self.output_workspace.take()
            var dlab = self.label_workspace.take()
            try:
                self._labels_on[RF_INPUT](dx, dout, dlab, x, output, rows, features, outputs)
            except e:
                self.ctx.value().synchronize()
                self.input_workspace = dx^
                self.output_workspace = dout^
                self.label_workspace = dlab^
                raise e
            self.input_workspace = dx^
            self.output_workspace = dout^
            self.label_workspace = dlab^


def _predict_into_buffers[RF_INPUT: Bool](ctx: DeviceContext,
    mut offsets: DeviceBuffer[DType.int32], mut columns: DeviceBuffer[DType.int32],
    mut thresholds: DeviceBuffer[DType.float32], mut left: DeviceBuffer[DType.int32],
    mut leaves: DeviceBuffer[DType.float32],
    x: MutPointer[Float32, MutAnyOrigin], output: MutPointer[Float32, MutAnyOrigin],
    rows: Int, features: Int, outputs: Int, trees: Int,
    mut dx: DeviceBuffer[DType.float32], mut dout: DeviceBuffer[DType.float32],
    stage: MutPointer[Float32, MutAnyOrigin], staged: Bool, ordered: Bool,
    uploaded: Bool, mut dstat: DeviceBuffer[DType.int32], mut hstat: HostBuffer[DType.int32]) raises:
    """`stage` is the pinned host stage the upload reads when `staged`
    (DEVIATION 2962), else `x` itself; the input scan is the same predicate
    either way. `uploaded`: `dx` already holds these rows, scanned finite
    (the registry's staged input, `ForestIOWorkspace.stage_input`)."""
    var t0 = 0
    var t1 = 0
    var t2 = 0
    var t3 = 0
    var t4 = 0
    comptime if FOREST_PROFILE:
        t0 = Int(perf_counter_ns())
    # the input goes up as it is and is scanned where it lands (the host
    # scan and the staging copy are gone; `stage` is unused)
    _ = staged
    _ = stage
    comptime if FOREST_PROFILE:
        t1 = Int(perf_counter_ns())
    # The input's finite test (lane a7c): the status word of the words just
    # uploaded (core/admitted_finite_status.mojo), copied back with the
    # output under the one synchronize below; the separate scan with its
    # own drain (`device_all_finite`) is gone. The public doors defer their
    # host scan of a borrowed X to this test (trees_public_forest
    # `_public_deferred`). Numerics: none (a predicate on bits).
    var scan_input = not uploaded
    # Lane forests-regression (2026-11-14): word 0 is the input's status,
    # word 1 the output's. The output test was `device_all_finite(ctx, dout,
    # ...)` after the synchronize below: its own flag buffer, pinned host
    # word, launch and a SECOND synchronize per predict. It is now the same
    # predicate (`|bits| >= 0x7F800000`, the exponent of all ones that
    # `forest_nonfinite_kernel` tested) enqueued after the traversal and
    # copied back with the output under the one synchronize. Numerics: none.
    # Lane flag-trees-infer-iforest: `dstat` / `hstat` are the snapshot's
    # held pair (`ResidentForest.status_dev` / `status_host`); this function
    # created both on every call (a device and a pinned host allocation, and
    # the pinned free, per predict). The fill below resets both words each
    # call, as the fresh pair's fill did. Numerics: none.
    try:
        dstat.enqueue_fill(Int32(0))
        if not uploaded:
            ctx.enqueue_copy(dst_buf=dx.create_sub_buffer[DType.float32](0, rows*features), src_ptr=x)
        if scan_input:
            enqueue_finite_status(ctx, dx.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin](), rows * features,
                                  dstat.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin](), 0)
        comptime if FOREST_PROFILE:
            ctx.synchronize()
            t2 = Int(perf_counter_ns())
        if ordered:
            launch_forest_inference[RF_INPUT, False, FOREST_PACKED_NODES](ctx, offsets, columns,
                thresholds, left, leaves, dx, dout, rows, features, outputs, trees)
        else:
            launch_forest_inference[RF_INPUT, True, FOREST_PACKED_NODES](ctx, offsets, columns,
                thresholds, left, leaves, dx, dout, rows, features, outputs, trees)
        comptime if FOREST_PROFILE:
            ctx.synchronize()
            t3 = Int(perf_counter_ns())
        enqueue_finite_status(ctx, dout.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin](), rows * outputs,
                              dstat.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin](), 1)
        ctx.enqueue_copy(dst_ptr=output, src_buf=dout.create_sub_buffer[DType.float32](0, rows*outputs))
        ctx.enqueue_copy(dst_ptr=hstat.unsafe_ptr(), src_buf=dstat)
        ctx.synchronize()
        if scan_input and hstat.unsafe_ptr().unsafe_load(0) != 0:
            raise Error("resident forest requires finite Float32 values")
        if hstat.unsafe_ptr().unsafe_load(1) != 0:
            raise Error("resident forest requires finite Float32 values")
    except e:
        ctx.synchronize()
        raise e
    comptime if FOREST_PROFILE:
        t4 = Int(perf_counter_ns())
    comptime if FOREST_PROFILE:
        var t5 = Int(perf_counter_ns())
        print("FOREST_PROFILE rows", rows, "features", features, "outputs", outputs,
              "staged", staged, "input_scan_ns", t1 - t0, "upload_ns", t2 - t1,
              "kernel_ns", t3 - t2, "readback_ns", t4 - t3, "output_scan_ns", t5 - t4)



struct ForestRegistry(Defaultable, Movable):
    var entries: Dict[Int, ResidentForest]
    var next_id: Int

    def __init__(out self):
        self.entries = Dict[Int, ResidentForest]()
        self.next_id = 1


# One registry per RF_INPUT specialization avoids RF/ET policy confusion.
comptime RF_REGISTRY = SharedSlot[ForestRegistry, ("MojolearnRFResidentForestIdenticalV1" if GLOBAL_NUMERIC_MODE == 1 else "MojolearnRFResidentForestFastV1")]
comptime ET_REGISTRY = SharedSlot[ForestRegistry, ("MojolearnETResidentForestIdenticalV1" if GLOBAL_NUMERIC_MODE == 1 else "MojolearnETResidentForestFastV1")]


def resident_prepare[RF_INPUT: Bool](offsets: List[Int32], columns: List[Int32],
    thresholds: List[Float32], left: List[Int32], leaves: List[Float32],
    features: Int, outputs: Int, ordered: Bool = FOREST_ORDERED_RESIDENT) raises -> Int:
    var state = RF_REGISTRY.get_or_create_ptr()
    comptime if not RF_INPUT:
        state = ET_REGISTRY.get_or_create_ptr()
    var shared = Optional[DeviceContext]()
    comptime if RF_INPUT:
        shared = process_ctx[RF_CTX_SLOT]()
    else:
        shared = process_ctx[ET_CTX_SLOT]()
    var model = ResidentForest(offsets, columns, thresholds, left, leaves, features, outputs,
                               ordered, shared)
    if state[].next_id == 9223372036854775807:
        model.close()
        raise Error("resident forest handle space exhausted")
    var handle = state[].next_id
    state[].next_id += 1
    state[].entries[handle] = model^
    return handle


def resident_predict[RF_INPUT: Bool](handle: Int, x: List[Float32], rows: Int,
    features: Int, outputs: Int) raises -> List[Float32]:
    var state = RF_REGISTRY.get_or_create_ptr()
    comptime if not RF_INPUT:
        state = ET_REGISTRY.get_or_create_ptr()
    if handle not in state[].entries:
        raise Error("unknown or released resident forest handle")
    return state[].entries[handle].predict[RF_INPUT](x, rows, features, outputs)


def resident_release[RF_INPUT: Bool](handle: Int) raises:
    var state = RF_REGISTRY.get_or_create_ptr()
    comptime if not RF_INPUT:
        state = ET_REGISTRY.get_or_create_ptr()
    if handle not in state[].entries:
        raise Error("unknown or released resident forest handle")
    state[].entries[handle].close()
    var released = state[].entries.pop(handle)
    _ = released^
    if len(state[].entries) == 0:
        # the registry's last snapshot: the shared workspace goes with it
        var io = RF_IO.get_or_create_ptr()
        comptime if not RF_INPUT:
            io = ET_IO.get_or_create_ptr()
        io[].release()


def resident_predict_into[RF_INPUT: Bool](handle: Int,
    x: MutPointer[Float32, MutAnyOrigin], output: MutPointer[Float32, MutAnyOrigin],
    rows: Int, features: Int, outputs: Int, reuse_io: Bool = False) raises:
    var state = RF_REGISTRY.get_or_create_ptr()
    comptime if not RF_INPUT:
        state = ET_REGISTRY.get_or_create_ptr()
    if handle not in state[].entries:
        raise Error("unknown or released resident forest handle")
    state[].entries[handle].predict_into[RF_INPUT](x, output, rows, features, outputs, reuse_io)


def resident_predict_labels[RF_INPUT: Bool](handle: Int,
    x: MutPointer[Float32, MutAnyOrigin], output: MutPointer[Int32, MutAnyOrigin],
    rows: Int, features: Int, outputs: Int) raises:
    var state = RF_REGISTRY.get_or_create_ptr()
    comptime if not RF_INPUT:
        state = ET_REGISTRY.get_or_create_ptr()
    if handle not in state[].entries:
        raise Error("unknown or released resident forest handle")
    state[].entries[handle].predict_labels[RF_INPUT](x, output, rows, features, outputs)


def resident_stage_input[RF_INPUT: Bool](x: MutPointer[Float32, MutAnyOrigin], n: Int) raises -> Bool:
    """Stage `n` host values at `x` as the registry's shared input (see
    `ForestIOWorkspace.staged_len`); False where nothing is staged: several
    devices (pooled snapshots upload per device), no live snapshot yet, or
    rows that are not all finite. The caller unstages with
    `resident_unstage_input`."""
    if forest_device_count() != 1:
        return False
    var state = RF_REGISTRY.get_or_create_ptr()
    comptime if not RF_INPUT:
        state = ET_REGISTRY.get_or_create_ptr()
    if len(state[].entries) == 0:
        return False
    var io = RF_IO.get_or_create_ptr()
    comptime if not RF_INPUT:
        io = ET_IO.get_or_create_ptr()
    # The context every shared snapshot of this registry runs on.
    comptime if RF_INPUT:
        return io[].stage_input(process_ctx[RF_CTX_SLOT](), x, n)
    else:
        return io[].stage_input(process_ctx[ET_CTX_SLOT](), x, n)


def resident_unstage_input[RF_INPUT: Bool]() raises:
    """Drop the staged-input mark; the buffer stays for the next call."""
    var io = RF_IO.get_or_create_ptr()
    comptime if not RF_INPUT:
        io = ET_IO.get_or_create_ptr()
    io[].unstage()
