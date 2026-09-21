# SPDX-License-Identifier: Apache-2.0
"""Owned GPU inference snapshots, shared by RF/ET binding registries.

nvForest26.08 cef3a50d caches an owning inference model; cuML26.08
randomforest_common.pyx:675-693 retains that model across predictions.
FOREST-RESIDENT-1: explicit monotonically increasing registry IDs replace the
Cython owning-object boundary. Python holds the GIL across these operations.
Snapshots own model buffers and a context; release drops buffers before context.
Only input validation/upload and result readback recur per prediction.
"""
from std.ffi import _Global
from std.sys.compile import is_defined
from std.memory import bitcast
from std.time import perf_counter_ns
from max.algorithm import sync_parallelize
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_FAST, NUMERIC_IDENTICAL
from checks.kernel_matrix import TARGET_COLUMN, COLUMN_APPLE, COLUMN_NVIDIA, COLUMN_AMD
from max.gpu.host import DeviceContext, DeviceBuffer, HostBuffer
from core.forest_inference import validate_flat_forest, require_finite, launch_forest_inference, launch_forest_argmax, FOREST_PACKED_NODES
from core.forest_inference_pool import PooledForest, forest_device_count


#: DEVIATION 2962 (lane/forest-groves-cpu-and-speed, 2026-09-17): the two
#: per-call host scans of the resident path (input finiteness before the
#: upload, output finiteness after the readback) run 16 lanes wide in chunks
#: across the host pool instead of one scalar compare per element. The
#: verdict is the same predicate on the same bits (exponent field all ones);
#: no arithmetic. With `MOJOLEARN_FOREST_PINNED_STAGE=1` the input scan is
#: fused with a copy into a pinned host stage retained beside the device
#: workspace pair (FOREST-IO-REUSE-1's lifetime), and the upload reads the
#: stage, so the caller's pageable rows cross to the device from pinned
#: memory. Both are host glue; the kernel, its 32-group topology, the
#: reduction order and the FTZ rules are untouched. `MOJOLEARN_FOREST_PROFILE=1`
#: prints one line per call with the nanoseconds of each stage, the
#: stream drained between stages so the numbers attribute; it is a
#: diagnostic build, never a timed arm.
comptime FOREST_PINNED_STAGE = is_defined["MOJOLEARN_FOREST_PINNED_STAGE"]()
comptime FOREST_PROFILE = is_defined["MOJOLEARN_FOREST_PROFILE"]()
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
#: Apple, NVIDIA and AMD in FAST/IDENTICAL. `_OFF` restores the former
#: sequential IDENTICAL AUTO policy and resident FAST 32-grove graph; the
#: positive define remains an experiment switch for other IDENTICAL columns.
comptime FOREST_ORDERED_RESIDENT = forest_ordered_resident_policy[
    TARGET_COLUMN,
    GLOBAL_NUMERIC_MODE,
    is_defined["MOJOLEARN_FOREST_ORDERED_RESIDENT"](),
    is_defined["MOJOLEARN_FOREST_ORDERED_RESIDENT_OFF"](),
]()
comptime FOREST_CHECK_W = 16
comptime FOREST_CHECK_CHUNK = 1 << 20
comptime FOREST_CHECK_SERIAL = 1 << 16


@always_inline
def _nonfinite_lanes(v: SIMD[DType.float32, FOREST_CHECK_W]) -> Int:
    var e = bitcast[DType.uint32](v) & SIMD[DType.uint32, FOREST_CHECK_W](0x7f800000)
    return Int(e.eq(SIMD[DType.uint32, FOREST_CHECK_W](0x7f800000)).cast[DType.uint32]().reduce_add())


def scan_finite_f32(src: MutPointer[Float32, MutAnyOrigin], dst: MutPointer[Float32, MutAnyOrigin],
                    n: Int, copy: Bool) -> Bool:
    """DEVIATION 2962: True when every one of the `n` floats at `src` is
    finite; with `copy`, also moves them to `dst` (a pure byte move) in the
    same pass. Chunks across the host pool above `FOREST_CHECK_SERIAL`."""
    if n <= 0:
        return True
    var chunks = (n + FOREST_CHECK_CHUNK - 1) // FOREST_CHECK_CHUNK
    var flags = List[Int](length=chunks, fill=0)
    var fp = flags.unsafe_ptr()

    def _chunk(k: Int) {imm src, imm dst, imm n, imm copy, imm fp}:
        var i0 = k * FOREST_CHECK_CHUNK
        var i1 = i0 + FOREST_CHECK_CHUNK
        if i1 > n:
            i1 = n
        var bad = 0
        var i = i0
        while i + FOREST_CHECK_W <= i1:
            var v = src.unsafe_load[width=FOREST_CHECK_W](i)
            if copy:
                dst.unsafe_store[width=FOREST_CHECK_W](i, v)
            bad += _nonfinite_lanes(v)
            i += FOREST_CHECK_W
        while i < i1:
            var s = src.unsafe_load(i)
            if copy:
                dst.unsafe_store(i, s)
            if (bitcast[DType.uint32](s) & UInt32(0x7f800000)) == UInt32(0x7f800000):
                bad += 1
            i += 1
        if bad > 0:
            fp.unsafe_store(k, 1)

    if chunks == 1 or n < FOREST_CHECK_SERIAL:
        for k in range(chunks):
            _chunk(k)
    else:
        sync_parallelize(_chunk, chunks)
    _ = len(flags)
    for k in range(chunks):
        if flags[k] != 0:
            return False
    return True


def require_finite_pointer(values: MutPointer[Float32, MutAnyOrigin], count: Int) raises:
    if not scan_finite_f32(values, values, count, False):
        raise Error("resident forest requires finite Float32 values")


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
    var input_stage: Optional[HostBuffer[DType.float32]]
    var workspace_rows: Int
    var features: Int
    var outputs: Int
    var trees: Int

    def __init__(out self, offsets: List[Int32], columns: List[Int32],
        thresholds: List[Float32], left: List[Int32], leaves: List[Float32],
        features: Int, outputs: Int) raises:
        var empty = List[Float32]()
        validate_flat_forest(offsets, columns, thresholds, left, leaves, empty, 0, features, outputs)
        self.pool = Optional[PooledForest]()
        self.input_workspace = Optional[DeviceBuffer[DType.float32]]()
        self.output_workspace = Optional[DeviceBuffer[DType.float32]]()
        self.label_workspace = Optional[DeviceBuffer[DType.int32]]()
        self.input_stage = Optional[HostBuffer[DType.float32]]()
        self.workspace_rows = 0
        self.features = features
        self.outputs = outputs
        self.trees = len(offsets) - 1
        self.ctx = Optional[DeviceContext]()
        self.offsets = Optional[DeviceBuffer[DType.int32]]()
        self.columns = Optional[DeviceBuffer[DType.int32]]()
        self.thresholds = Optional[DeviceBuffer[DType.float32]]()
        self.left = Optional[DeviceBuffer[DType.int32]]()
        self.leaves = Optional[DeviceBuffer[DType.float32]]()
        var device_count = forest_device_count()
        comptime if FOREST_PACKED_NODES:
            if len(columns) > 2147483647 // 4:
                raise Error("packed forest node word count exceeds Int32")
        # A sharded pool combines per-device tree partitions and therefore
        # cannot express one global increasing-tree fold.  The experimental
        # ordered arm stays on one device so its arithmetic graph is exact.
        if device_count > 1 and not FOREST_ORDERED_RESIDENT:
            self.pool = PooledForest(offsets, columns, thresholds, left, leaves,
                features, outputs, device_count)
            return
        # DEVIATION BLOCK FOREST-PACKED-1 (experimental, no speed claim):
        # nvForest cef3a50d detail/node.hpp:81-175 packs node fields; builder
        # detail/decision_forest_builder.hpp:135-149 stores only leaf vectors.
        # Keep our sibling node order/local child IDs and raw <= policy;
        # do not adopt depth-first offsets or converted thresholds here.
        # Archive arrays are unchanged. Packing runs once per resident snapshot.
        var packed_nodes = List[Int32]()
        var compact_leaves = List[Float32]()
        comptime if FOREST_PACKED_NODES:
            if len(columns) > 2147483647 // 4:
                raise Error("packed forest node word count exceeds Int32")
            for node in range(len(columns)):
                var payload = bitcast[DType.int32](thresholds[node])
                if left[node] == -1:
                    payload = Int32(len(compact_leaves) // outputs)
                    for c in range(outputs):
                        compact_leaves.append(leaves[node * outputs + c])
                packed_nodes.append(payload)
                packed_nodes.append(left[node])
                packed_nodes.append(columns[node])
                packed_nodes.append(0)
        self.ctx = DeviceContext()
        try:
            self.offsets = self.ctx.value().enqueue_create_buffer[DType.int32](len(offsets))
            self.ctx.value().enqueue_copy(dst_buf=self.offsets.value(), src_ptr=offsets.unsafe_ptr())
            comptime if FOREST_PACKED_NODES:
                self.columns = self.ctx.value().enqueue_create_buffer[DType.int32](len(packed_nodes))
                # Unused ABI operands; avoid retaining original SoA buffers.
                self.thresholds = self.ctx.value().enqueue_create_buffer[DType.float32](1)
                self.left = self.ctx.value().enqueue_create_buffer[DType.int32](1)
                self.leaves = self.ctx.value().enqueue_create_buffer[DType.float32](len(compact_leaves))
                self.ctx.value().enqueue_copy(dst_buf=self.columns.value(), src_ptr=packed_nodes.unsafe_ptr())
                self.ctx.value().enqueue_copy(dst_buf=self.leaves.value(), src_ptr=compact_leaves.unsafe_ptr())
            else:
                self.columns = self.ctx.value().enqueue_create_buffer[DType.int32](len(columns))
                self.thresholds = self.ctx.value().enqueue_create_buffer[DType.float32](len(thresholds))
                self.left = self.ctx.value().enqueue_create_buffer[DType.int32](len(left))
                self.leaves = self.ctx.value().enqueue_create_buffer[DType.float32](len(leaves))
                self.ctx.value().enqueue_copy(dst_buf=self.columns.value(), src_ptr=columns.unsafe_ptr())
                self.ctx.value().enqueue_copy(dst_buf=self.thresholds.value(), src_ptr=thresholds.unsafe_ptr())
                self.ctx.value().enqueue_copy(dst_buf=self.left.value(), src_ptr=left.unsafe_ptr())
                self.ctx.value().enqueue_copy(dst_buf=self.leaves.value(), src_ptr=leaves.unsafe_ptr())
            self.ctx.value().synchronize()
            _ = len(packed_nodes)
            _ = len(compact_leaves)
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
        _ = self.output_workspace^
        _ = self.label_workspace^
        _ = self.input_workspace^
        _ = self.input_stage^
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
        self.output_workspace = None
        self.label_workspace = None
        self.input_workspace = None
        self.input_stage = None
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
        require_finite(x)
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
        var hout = self.ctx.value().enqueue_create_host_buffer[DType.float32](rows * outputs)
        try:
            self.ctx.value().enqueue_copy(dst_buf=dx, src_ptr=x.unsafe_ptr())
            launch_forest_inference[RF_INPUT, not FOREST_ORDERED_RESIDENT, FOREST_PACKED_NODES](
                self.ctx.value(), self.offsets.value(), self.columns.value(),
                self.thresholds.value(), self.left.value(), self.leaves.value(),
                dx, dout, rows, features, outputs, self.trees,
            )
            self.ctx.value().enqueue_copy(dst_ptr=hout.unsafe_ptr(), src_buf=dout)
            self.ctx.value().synchronize()
        except e:
            # A launch may already reference the temporary buffers or X.
            self.ctx.value().synchronize()
            raise e
        var result = List[Float32](capacity=rows * outputs)
        for i in range(rows * outputs):
            result.append(hout.unsafe_ptr().unsafe_load(i))
        require_finite(result)
        _ = len(x)
        _ = dx^
        _ = dout^
        _ = hout^
        return result^

    def predict_into[RF_INPUT: Bool](mut self,
        x: MutPointer[Float32, MutAnyOrigin], output: MutPointer[Float32, MutAnyOrigin],
        rows: Int, features: Int, outputs: Int,
        reuse_io: Bool = False) raises:
        """Borrowed synchronous I/O; caller retains buffers and holds the GIL.

        nvForest26.08 forest_model.hpp:284-308 wraps borrowed I/O pointers.
        Here host input still uploads, but no intermediate host Lists are made.
        Output may be written before a nonfinite-result error is raised.
        """
        if features != self.features or outputs != self.outputs:
            raise Error("resident forest dimensions differ from prepared snapshot")
        if rows < 0 or rows > 2147483647 // features or rows > 2147483647 // outputs:
            raise Error("resident forest prediction dimensions exceed Int32")
        if rows == 0:
            require_finite_pointer(x, 0)
            return
        if self.pool:
            require_finite_pointer(x, rows * features)
            self.pool.value().predict_into[RF_INPUT](x, output, rows)
            return
        if reuse_io:
            self.prepare_workspace(rows)
            var stage = x
            var staged = False
            comptime if FOREST_PINNED_STAGE:
                stage = self.input_stage.value().unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]()
                staged = True
            _predict_into_buffers[RF_INPUT](self.ctx.value(), self.offsets.value(),
                self.columns.value(), self.thresholds.value(), self.left.value(),
                self.leaves.value(), x, output, rows, features, outputs, self.trees,
                self.input_workspace.value(), self.output_workspace.value(), stage, staged)
        else:
            var dx = self.ctx.value().enqueue_create_buffer[DType.float32](rows * features)
            var dout = self.ctx.value().enqueue_create_buffer[DType.float32](rows * outputs)
            _predict_into_buffers[RF_INPUT](self.ctx.value(), self.offsets.value(),
                self.columns.value(), self.thresholds.value(), self.left.value(),
                self.leaves.value(), x, output, rows, features, outputs, self.trees, dx, dout, x, False)
            _ = dx^
            _ = dout^

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
        self.input_stage = None
        self.workspace_rows = 0
        try:
            self.input_workspace = self.ctx.value().enqueue_create_buffer[DType.float32](rows * self.features)
            self.output_workspace = self.ctx.value().enqueue_create_buffer[DType.float32](rows * self.outputs)
            comptime if FOREST_PINNED_STAGE:
                # DEVIATION 2962: the pinned input stage, the same lifetime
                # as the device pair it feeds.
                self.input_stage = self.ctx.value().enqueue_create_host_buffer[DType.float32](rows * self.features)
                self.ctx.value().synchronize()
        except e:
            self.ctx.value().synchronize()
            self.input_workspace = None
            self.output_workspace = None
            self.label_workspace = None
            self.input_stage = None
            raise e
        self.workspace_rows = rows

    def predict_labels[RF_INPUT: Bool](mut self,
        x: MutPointer[Float32, MutAnyOrigin], output: MutPointer[Int32, MutAnyOrigin],
        rows: Int, features: Int, outputs: Int) raises:
        """FAST classifier boundary: keep vote rows on device and return codes."""
        if features != self.features or outputs != self.outputs or outputs < 2:
            raise Error("resident forest dimensions differ from prepared snapshot")
        if rows < 0 or rows > 2147483647 // features or rows > 2147483647 // outputs:
            raise Error("resident forest prediction dimensions exceed Int32")
        if self.pool:
            require_finite_pointer(x, rows * features)
            var votes = List[Float32](length=rows * outputs, fill=Float32(0))
            self.pool.value().predict_into[RF_INPUT](
                x, votes.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin](), rows)
            for row in range(rows):
                var best = 0
                var best_value = votes[row * outputs]
                for c in range(1, outputs):
                    if votes[row * outputs + c] > best_value:
                        best = c
                        best_value = votes[row * outputs + c]
                output.unsafe_store(row, Int32(best))
            _ = votes^
            return
        if rows == 0:
            require_finite_pointer(x, 0)
            return
        self.prepare_workspace(rows)
        require_finite_pointer(x, rows * features)
        try:
            if not self.label_workspace:
                self.label_workspace = self.ctx.value().enqueue_create_buffer[DType.int32](rows)
            self.ctx.value().enqueue_copy(dst_buf=self.input_workspace.value(), src_ptr=x)
            launch_forest_inference[RF_INPUT, not FOREST_ORDERED_RESIDENT, FOREST_PACKED_NODES](
                self.ctx.value(), self.offsets.value(), self.columns.value(),
                self.thresholds.value(), self.left.value(), self.leaves.value(),
                self.input_workspace.value(), self.output_workspace.value(), rows,
                features, outputs, self.trees,
            )
            launch_forest_argmax(self.ctx.value(), self.output_workspace.value(),
                                 self.label_workspace.value(), rows, outputs)
            self.ctx.value().enqueue_copy(dst_ptr=output, src_buf=self.label_workspace.value())
            self.ctx.value().synchronize()
        except e:
            self.ctx.value().synchronize()
            raise e
        for row in range(rows):
            if output.unsafe_load(row) < 0:
                raise Error("resident forest requires finite Float32 values")


def _predict_into_buffers[RF_INPUT: Bool](ctx: DeviceContext,
    mut offsets: DeviceBuffer[DType.int32], mut columns: DeviceBuffer[DType.int32],
    mut thresholds: DeviceBuffer[DType.float32], mut left: DeviceBuffer[DType.int32],
    mut leaves: DeviceBuffer[DType.float32],
    x: MutPointer[Float32, MutAnyOrigin], output: MutPointer[Float32, MutAnyOrigin],
    rows: Int, features: Int, outputs: Int, trees: Int,
    mut dx: DeviceBuffer[DType.float32], mut dout: DeviceBuffer[DType.float32],
    stage: MutPointer[Float32, MutAnyOrigin], staged: Bool) raises:
    """`stage` is the pinned host stage the upload reads when `staged`
    (DEVIATION 2962), else `x` itself; the input scan is the same predicate
    either way."""
    var t0 = 0
    var t1 = 0
    var t2 = 0
    var t3 = 0
    var t4 = 0
    comptime if FOREST_PROFILE:
        t0 = Int(perf_counter_ns())
    if not scan_finite_f32(x, stage, rows * features, staged):
        raise Error("resident forest requires finite Float32 values")
    comptime if FOREST_PROFILE:
        t1 = Int(perf_counter_ns())
    try:
        ctx.enqueue_copy(dst_buf=dx, src_ptr=stage)
        comptime if FOREST_PROFILE:
            ctx.synchronize()
            t2 = Int(perf_counter_ns())
        launch_forest_inference[RF_INPUT, not FOREST_ORDERED_RESIDENT, FOREST_PACKED_NODES](ctx, offsets, columns,
            thresholds, left, leaves, dx, dout, rows, features, outputs, trees)
        comptime if FOREST_PROFILE:
            ctx.synchronize()
            t3 = Int(perf_counter_ns())
        ctx.enqueue_copy(dst_ptr=output, src_buf=dout)
        ctx.synchronize()
    except e:
        ctx.synchronize()
        raise e
    comptime if FOREST_PROFILE:
        t4 = Int(perf_counter_ns())
    require_finite_pointer(output, rows * outputs)
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
comptime RF_REGISTRY = _Global[StorageType=ForestRegistry,
    name=("MojoRFResidentForestIdentical" if GLOBAL_NUMERIC_MODE == 1 else
          "MojoRFResidentForestDeterministic" if GLOBAL_NUMERIC_MODE == 2 else
          "MojoRFResidentForestFast"), init_fn=ForestRegistry.__init__]
comptime ET_REGISTRY = _Global[StorageType=ForestRegistry,
    name=("MojoETResidentForestIdentical" if GLOBAL_NUMERIC_MODE == 1 else
          "MojoETResidentForestDeterministic" if GLOBAL_NUMERIC_MODE == 2 else
          "MojoETResidentForestFast"), init_fn=ForestRegistry.__init__]


def resident_prepare[RF_INPUT: Bool](offsets: List[Int32], columns: List[Int32],
    thresholds: List[Float32], left: List[Int32], leaves: List[Float32],
    features: Int, outputs: Int) raises -> Int:
    var state = RF_REGISTRY.get_or_create_ptr()
    comptime if not RF_INPUT:
        state = ET_REGISTRY.get_or_create_ptr()
    var model = ResidentForest(offsets, columns, thresholds, left, leaves, features, outputs)
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
