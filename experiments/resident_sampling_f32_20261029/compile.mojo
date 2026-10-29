from max.gpu.host import DeviceContext
from core.generation_sampling_f32 import sampling_pick_f32
from core.generation_policy_impl import GenerationSamplingScratch, generation_pick_row


def probe(logits: MutPointer[Float32, MutAnyOrigin], order: MutPointer[Int32, MutAnyOrigin],
          merge: MutPointer[Int32, MutAnyOrigin], weights: MutPointer[Float32, MutAnyOrigin],
          output: MutPointer[Int32, MutAnyOrigin]):
    var result = sampling_pick_f32(logits, order, merge, weights,
        4, Float32(1), 0, Float32(0.9), UInt64(7), 0, 3)
    output[0] = Int32(result[0])
    output[1] = Int32(result[1])


def main() raises:
    var logits = List[Float32]()
    for i in range(4):
        logits.append(Float32(i))
    var scratch = GenerationSamplingScratch()
    scratch.use_f32_policy = True
    _ = generation_pick_row(MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=Int(logits.unsafe_ptr())), 4, False, 1, 0, 0.9, 7, 0, 3, scratch)
    var ctx = DeviceContext()
    var scores = ctx.enqueue_create_buffer[DType.float32](4)
    var order = ctx.enqueue_create_buffer[DType.int32](4)
    var merge = ctx.enqueue_create_buffer[DType.int32](4)
    var weights = ctx.enqueue_create_buffer[DType.float32](4)
    var output = ctx.enqueue_create_buffer[DType.int32](2)
    ctx.enqueue_function[probe](scores.unsafe_ptr(), order.unsafe_ptr(), merge.unsafe_ptr(),
        weights.unsafe_ptr(), output.unsafe_ptr(), grid_dim=(1, 1, 1), block_dim=(1, 1, 1))
    ctx.synchronize()
