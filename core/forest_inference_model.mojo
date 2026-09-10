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
from std.memory import bitcast
from checks.numerics import GLOBAL_NUMERIC_MODE
from max.gpu.host import DeviceContext, DeviceBuffer
from core.forest_inference import validate_flat_forest, require_finite, launch_forest_inference


def require_finite_pointer(values: MutPointer[Float32, MutAnyOrigin], count: Int) raises:
    for i in range(count):
        if (bitcast[DType.uint32](values.unsafe_load(i)) & UInt32(0x7f800000)) == UInt32(0x7f800000):
            raise Error("resident forest requires finite Float32 values")


struct ResidentForest(Movable):
    var ctx: Optional[DeviceContext]
    var offsets: Optional[DeviceBuffer[DType.int32]]
    var columns: Optional[DeviceBuffer[DType.int32]]
    var thresholds: Optional[DeviceBuffer[DType.float32]]
    var left: Optional[DeviceBuffer[DType.int32]]
    var leaves: Optional[DeviceBuffer[DType.float32]]
    var input_workspace: Optional[DeviceBuffer[DType.float32]]
    var output_workspace: Optional[DeviceBuffer[DType.float32]]
    var workspace_rows: Int
    var features: Int
    var outputs: Int
    var trees: Int

    def __init__(out self, offsets: List[Int32], columns: List[Int32],
        thresholds: List[Float32], left: List[Int32], leaves: List[Float32],
        features: Int, outputs: Int) raises:
        var empty = List[Float32]()
        validate_flat_forest(offsets, columns, thresholds, left, leaves, empty, 0, features, outputs)
        self.input_workspace = Optional[DeviceBuffer[DType.float32]]()
        self.output_workspace = Optional[DeviceBuffer[DType.float32]]()
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
        self.ctx = DeviceContext()
        try:
            self.offsets = self.ctx.value().enqueue_create_buffer[DType.int32](len(offsets))
            self.columns = self.ctx.value().enqueue_create_buffer[DType.int32](len(columns))
            self.thresholds = self.ctx.value().enqueue_create_buffer[DType.float32](len(thresholds))
            self.left = self.ctx.value().enqueue_create_buffer[DType.int32](len(left))
            self.leaves = self.ctx.value().enqueue_create_buffer[DType.float32](len(leaves))
            self.ctx.value().enqueue_copy(dst_buf=self.offsets.value(), src_ptr=offsets.unsafe_ptr())
            self.ctx.value().enqueue_copy(dst_buf=self.columns.value(), src_ptr=columns.unsafe_ptr())
            self.ctx.value().enqueue_copy(dst_buf=self.thresholds.value(), src_ptr=thresholds.unsafe_ptr())
            self.ctx.value().enqueue_copy(dst_buf=self.left.value(), src_ptr=left.unsafe_ptr())
            self.ctx.value().enqueue_copy(dst_buf=self.leaves.value(), src_ptr=leaves.unsafe_ptr())
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
        _ = self.output_workspace^
        _ = self.input_workspace^
        _ = self.leaves^
        _ = self.left^
        _ = self.thresholds^
        _ = self.columns^
        _ = self.offsets^
        _ = self.ctx^

    def close(mut self) raises:
        if self.ctx:
            self.ctx.value().synchronize()
        self.output_workspace = None
        self.input_workspace = None
        self.workspace_rows = 0
        self.leaves = None
        self.left = None
        self.thresholds = None
        self.columns = None
        self.offsets = None
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
        var dx = self.ctx.value().enqueue_create_buffer[DType.float32](len(x))
        var dout = self.ctx.value().enqueue_create_buffer[DType.float32](rows * outputs)
        var hout = self.ctx.value().enqueue_create_host_buffer[DType.float32](rows * outputs)
        try:
            self.ctx.value().enqueue_copy(dst_buf=dx, src_ptr=x.unsafe_ptr())
            launch_forest_inference[RF_INPUT, True](
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
        require_finite_pointer(x, rows * features)
        if rows == 0:
            return
        if reuse_io:
            self.prepare_workspace(rows)
            _predict_into_buffers[RF_INPUT](self.ctx.value(), self.offsets.value(),
                self.columns.value(), self.thresholds.value(), self.left.value(),
                self.leaves.value(), x, output, rows, features, outputs, self.trees,
                self.input_workspace.value(), self.output_workspace.value())
        else:
            var dx = self.ctx.value().enqueue_create_buffer[DType.float32](rows * features)
            var dout = self.ctx.value().enqueue_create_buffer[DType.float32](rows * outputs)
            _predict_into_buffers[RF_INPUT](self.ctx.value(), self.offsets.value(),
                self.columns.value(), self.thresholds.value(), self.left.value(),
                self.leaves.value(), x, output, rows, features, outputs, self.trees, dx, dout)
            _ = dx^
            _ = dout^

    def prepare_workspace(mut self, rows: Int) raises:
        # DEVIATION BLOCK FOREST-IO-REUSE-1:
        # nvForest cef3a50d forest_model.hpp:284-308 borrows caller-owned GPU
        # buffers; our NumPy boundary requires host/device copies. Retain one
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
        self.workspace_rows = 0
        try:
            self.input_workspace = self.ctx.value().enqueue_create_buffer[DType.float32](rows * self.features)
            self.output_workspace = self.ctx.value().enqueue_create_buffer[DType.float32](rows * self.outputs)
        except e:
            self.ctx.value().synchronize()
            self.input_workspace = None
            self.output_workspace = None
            raise e
        self.workspace_rows = rows


def _predict_into_buffers[RF_INPUT: Bool](ctx: DeviceContext,
    mut offsets: DeviceBuffer[DType.int32], mut columns: DeviceBuffer[DType.int32],
    mut thresholds: DeviceBuffer[DType.float32], mut left: DeviceBuffer[DType.int32],
    mut leaves: DeviceBuffer[DType.float32],
    x: MutPointer[Float32, MutAnyOrigin], output: MutPointer[Float32, MutAnyOrigin],
    rows: Int, features: Int, outputs: Int, trees: Int,
    mut dx: DeviceBuffer[DType.float32], mut dout: DeviceBuffer[DType.float32]) raises:
    try:
        ctx.enqueue_copy(dst_buf=dx, src_ptr=x)
        launch_forest_inference[RF_INPUT, True](ctx, offsets, columns,
            thresholds, left, leaves, dx, dout, rows, features, outputs, trees)
        ctx.enqueue_copy(dst_ptr=output, src_buf=dout)
        ctx.synchronize()
    except e:
        ctx.synchronize()
        raise e
    require_finite_pointer(output, rows * outputs)



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
