from max.gpu.host import DeviceContext
from core.generation_sampling_f32 import sampling_pick_f32
from core.generation_policy_impl import GenerationSamplingScratch, generation_pick_row


def probe(logits: MutPointer[Float32, MutAnyOrigin], order: MutPointer[Int32, MutAnyOrigin],
          merge: MutPointer[Int32, MutAnyOrigin], weights: MutPointer[Float32, MutAnyOrigin],
          output: MutPointer[Int32, MutAnyOrigin], seed: UInt64, position: Int64):
    var result = sampling_pick_f32(logits, order, merge, weights,
        64, Float32(0.7), 0, Float32(0.83), seed, 0, Int(position))
    output[0] = Int32(result[0])
    output[1] = Int32(result[1])
    output[2] = order[0]
    output[3] = merge[0]


def main() raises:
    var logits = List[Float32]()
    for i in range(64):
        logits.append(Float32(i) / 10)
    var scratch = GenerationSamplingScratch()
    scratch.use_f32_policy = True
    var expected = generation_pick_row(MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=Int(logits.unsafe_ptr())), 64, False, 0.7, 0, 0.83, 123, 0, 0, scratch)
    var ctx = DeviceContext()
    var scores = ctx.enqueue_create_buffer[DType.float32](64)
    var order = ctx.enqueue_create_buffer[DType.int32](64)
    var merge = ctx.enqueue_create_buffer[DType.int32](64)
    var weights = ctx.enqueue_create_buffer[DType.float32](64)
    var output = ctx.enqueue_create_buffer[DType.int32](4)
    var got = List[Int32](length=4, fill=-999)
    try:
        ctx.enqueue_copy(dst_buf=scores, src_ptr=logits.unsafe_ptr())
        ctx.enqueue_function[probe](scores.unsafe_ptr(), order.unsafe_ptr(), merge.unsafe_ptr(),
            weights.unsafe_ptr(), output.unsafe_ptr(), UInt64(123), Int64(0), grid_dim=(1, 1, 1), block_dim=(1, 1, 1))
        ctx.enqueue_copy(dst_ptr=got.unsafe_ptr(), src_buf=output)
        ctx.synchronize()
        print("expected", expected, "device", got[0], "bad", got[1], "order", got[2], "merge", got[3])
        if Int(got[0]) != expected or got[1] != -1:
            raise Error("shared host/device sampling mismatch")
    finally:
        _ = logits
        _ = got
        _ = scratch
