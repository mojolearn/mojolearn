from max.gpu.host import DeviceContext
from core.generation_sampling_f32 import sampling_pick_f32
from transformer.transformer_binding_impl import _lm_read_sample_kernel
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
    var statuses = List[Int32](length=2, fill=0)
    var device_status = ctx.enqueue_create_buffer[DType.int32](2)
    var next_ids = ctx.enqueue_create_buffer[DType.int32](1)
    try:
        ctx.enqueue_copy(dst_buf=scores, src_ptr=logits.unsafe_ptr())
        ctx.enqueue_copy(dst_buf=device_status, src_ptr=statuses.unsafe_ptr())
        ctx.enqueue_function[_lm_read_sample_kernel](scores.unsafe_ptr(), next_ids.unsafe_ptr(), output.unsafe_ptr(),
            device_status.unsafe_ptr(), order.unsafe_ptr(), merge.unsafe_ptr(), weights.unsafe_ptr(),
            Int32(1), Int32(1), Int32(64), Int32(1), Int32(0), Int32(9),
            Float32(0.7), Int32(0), Float32(0.83), UInt64(123), Int64(0),
            grid_dim=(1, 1, 1), block_dim=(1, 1, 1))
        ctx.enqueue_copy(dst_ptr=statuses.unsafe_ptr(), src_buf=device_status)
        ctx.enqueue_copy(dst_ptr=got.unsafe_ptr(), src_buf=output)
        ctx.synchronize()
        print("expected", expected, "device", got[0], "bad", statuses[1], "finished", statuses[0])
        if Int(got[0]) != expected or statuses[1] != -1:
            raise Error("shared host/device sampling mismatch")
    finally:
        _ = scores^
        _ = order^
        _ = merge^
        _ = weights^
        _ = output^
        _ = next_ids^
        _ = device_status^
        _ = statuses
        _ = logits
        _ = got
        _ = scratch
