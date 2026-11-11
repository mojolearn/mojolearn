# SPDX-License-Identifier: Apache-2.0
"""GP classifier accuracy over the existing unweighted COUNT program."""
from std.memory import unsafe_memcpy, bitcast
from core.hostptr import f32_ptr
from core.labels_accuracy_plan import LabelAccuracyPlan, label_accuracy_plan
from core.native_descriptor_abi import NativeDescriptorCall
from core.native_descriptor_call import native_descriptor_call
from x_metrics.descriptor_schema import MP_CAPABILITIES, MP_HAS_RANGES, MP_RUN, MP_RUN_OUT, MP_RUN_RANGES


struct GPCAccuracyProgram(Movable):
    var plan: LabelAccuracyPlan
    var arena: List[Float32]
    var inputs: List[Int32]
    var outputs: List[Int32]

    def __init__(out self, rows: Int, classes: Int, truth: Int, prediction: Int) raises:
        self.plan = label_accuracy_plan(rows, classes, False)
        self.arena = List[Float32](length=self.plan.program.size, fill=0.0)
        var address = Int(self.arena.unsafe_ptr())
        # Both original code arrays were copied once into the metrics arena;
        # their int32 words are copied as bits, with no numeric conversion.
        unsafe_memcpy(dest=f32_ptr(address), src=f32_ptr(truth), count=rows)
        unsafe_memcpy(dest=f32_ptr(address) + rows, src=f32_ptr(prediction), count=rows)
        self.inputs = [0, Int32(2 * rows), -1]
        var known = True
        # Keep the original complement of input/scratch ranges, including
        # group/count outputs. Do not shrink transfers to the final fraction.
        self.outputs = self.plan.program.download(known)

    def run(self, entry: NativeDescriptorCall, ranges: Bool, output_ranges: Bool = True) raises -> Float64:
        var address = Int(self.arena.unsafe_ptr())
        var stages = len(self.plan.program.stages) // 16
        var words = Int(self.plan.program.stages.unsafe_ptr())
        var count = len(self.outputs) // 4
        var capabilities = native_descriptor_call(entry, MP_CAPABILITIES, [], [], [])
        try:
            if ranges and capabilities & MP_HAS_RANGES:
                _ = native_descriptor_call(entry, MP_RUN_RANGES,
                    [address, words, Int(self.inputs.unsafe_ptr()), Int(self.outputs.unsafe_ptr())],
                    [self.plan.program.size, stages, 1, count], [])
            elif output_ranges:
                _ = native_descriptor_call(entry, MP_RUN_OUT,
                    [address, words, Int(self.outputs.unsafe_ptr())],
                    [self.plan.program.size, stages, count], [])
            else:
                _ = native_descriptor_call(entry, MP_RUN,
                    [address, words], [self.plan.program.size, stages], [])
            var at = self.plan.result_offset + 8
            var result = Pointer[UInt32, ImmUntrackedOrigin](unsafe_from_address=address)
            return bitcast[DType.float64](UInt64(result[at]) | (UInt64(result[at + 1]) << 32))
        finally:
            _ = self.arena
            _ = self.plan
            _ = self.inputs
            _ = self.outputs
