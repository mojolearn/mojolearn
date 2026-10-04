# SPDX-License-Identifier: Apache-2.0
"""Uncompiled opt-in experiment: fixed-capacity, single-context call storage.

No arithmetic lives here. The caller enqueues its EXISTING kernels against
device_input/device_output/scratch on the same in-order context, between
begin() and seal(). Keep this object and context alive through drain().
"""
from max.gpu.host import DeviceBuffer, DeviceContext, HostBuffer
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_FAST
from std.sys.info import has_apple_gpu_accelerator
from std.sys.compile import is_defined

comptime CALLPATH_ENABLED = (GLOBAL_NUMERIC_MODE == NUMERIC_FAST
    and has_apple_gpu_accelerator()
    and is_defined["MOJOLEARN_APPLE_FAST_CALLPATH_CANDIDATES"]())


struct ResidentCallSlot(Movable):
    var context_token: Int
    var device_input: DeviceBuffer[DType.float32]
    var device_output: DeviceBuffer[DType.float32]
    var scratch: DeviceBuffer[DType.float32]
    var index_scratch: DeviceBuffer[DType.int32]
    var host_input: HostBuffer[DType.float32]
    var host_output: HostBuffer[DType.float32]
    var input_count: Int
    var output_count: Int
    # 0 idle; 1 kernels may be enqueued; 2 readback submitted; 3 readable.
    var phase: Int
    var staged: Bool

    def __init__(out self, ctx: DeviceContext, inputs: Int, outputs: Int,
                 scratch_count: Int, index_count: Int) raises:
        comptime if not CALLPATH_ENABLED:
            raise Error("callpath candidates require opt-in FAST on Apple")
        self.context_token = Int(Pointer(to=ctx))
        if inputs < 0 or outputs < 0 or scratch_count < 0 or index_count < 0:
            raise Error("negative resident slot capacity")
        self.device_input = ctx.enqueue_create_buffer[DType.float32](max(inputs, 1))
        self.device_output = ctx.enqueue_create_buffer[DType.float32](max(outputs, 1))
        self.scratch = ctx.enqueue_create_buffer[DType.float32](max(scratch_count, 1))
        self.index_scratch = ctx.enqueue_create_buffer[DType.int32](max(index_count, 1))
        self.host_input = ctx.enqueue_create_host_buffer[DType.float32](max(inputs, 1))
        self.host_output = ctx.enqueue_create_host_buffer[DType.float32](max(outputs, 1))
        self.input_count = inputs
        self.output_count = outputs
        self.phase = 0
        self.staged = False
        # Setup boundary only; no constructor allocations on a warmed call.
        ctx.synchronize()

    def check_context(self, ctx: DeviceContext) raises:
        if self.context_token != Int(Pointer(to=ctx)):
            raise Error("callpath context identity changed")

    def stage(mut self, values: List[Float32]) raises:
        if self.phase != 0 or len(values) != self.input_count:
            raise Error("slot must be idle and input shape must match")
        for i in range(self.input_count):
            self.host_input.unsafe_ptr()[i] = values[i]
        self.staged = True

    def begin(mut self, ctx: DeviceContext) raises:
        self.check_context(ctx)
        if self.phase != 0 or not self.staged:
            raise Error("slot must be idle with staged input")
        # Claim before submission so exception cleanup must drain the slot.
        self.phase = 1
        self.staged = False
        if self.input_count > 0:
            ctx.enqueue_copy(dst_buf=self.device_input, src_ptr=self.host_input.unsafe_ptr())

    def seal(mut self, ctx: DeviceContext) raises:
        self.check_context(ctx)
        if self.phase != 1:
            raise Error("seal requires an open slot")
        if self.output_count > 0:
            ctx.enqueue_copy(dst_ptr=self.host_output.unsafe_ptr(), src_buf=self.device_output)
        self.phase = 2

    def wait(mut self, ctx: DeviceContext) raises:
        self.check_context(ctx)
        if self.phase != 2:
            raise Error("wait requires a sealed slot")
        ctx.synchronize()
        self.phase = 3

    def collect_into(mut self, mut result: List[Float32]) raises:
        """Reuse caller's result allocation; copying does not change FP bits."""
        if self.phase != 3 or len(result) != self.output_count:
            raise Error("result must be ready and output shape must match")
        for i in range(self.output_count):
            result[i] = self.host_output.unsafe_ptr()[i]
        self.phase = 0

    def drain(mut self, ctx: DeviceContext) raises:
        """Abort/release boundary; pending output is intentionally discarded."""
        self.check_context(ctx)
        ctx.synchronize()
        self.phase = 0
        self.staged = False


def wait_pair(ctx: DeviceContext, mut first: ResidentCallSlot,
              mut second: ResidentCallSlot) raises:
    """Two independent calls share one host wait on the SAME context.

    Both must be sealed. This is queue batching, not a claim of concurrent
    Metal execution. Independent slots prevent overwrite before collection.
    """
    first.check_context(ctx)
    second.check_context(ctx)
    if Int(Pointer(to=first)) == Int(Pointer(to=second)):
        raise Error("wait_pair requires distinct slots")
    if first.phase != 2 or second.phase != 2:
        raise Error("both slots must be sealed")
    ctx.synchronize()
    first.phase = 3
    second.phase = 3
