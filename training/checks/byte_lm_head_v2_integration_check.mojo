# SPDX-License-Identifier: Apache-2.0
"""End-to-end ByteTrainer gate for the explicit chunked LM-head v2 selector."""
from max.gpu.host import DeviceContext
from std.math import abs
from std.time import perf_counter_ns
from training.byte_lm import ByteTrainer, byte_train_step_resident
from training.chunked_lm_head_v2 import LM_HEAD_V2_CHUNK
from training.byte_lm_config import ByteConfig
from training.checks.optimizer_oracle import OPT_ADAMW, OptimizerConfig
from training.checks.train_loop import download_f32


def params(config: ByteConfig) raises -> List[Float32]:
    var out = List[Float32]()
    var offsets = config.offsets()
    for j in range(config.n_tensors()):
        var norm = j > 0 and j < config.n_tensors() - 1 and ((j - 1) % 9 == 0 or (j - 1) % 9 == 5)
        for i in range(offsets[j], offsets[j + 1]):
            var x = Float32((i * 17 + j * 13) % 101 - 50) / Float32(5000.0)
            out.append(x + Float32(1.0) if norm else x)
    return out^


def ids(config: ByteConfig) -> List[Int32]:
    var out = List[Int32]()
    for i in range(config.batch * (config.length + 1)):
        out.append(Int32((i * 127 + 11) % config.vocab_size))
    return out^


def run(ctx: DeviceContext, config: ByteConfig) raises -> Tuple[Float32, List[Float32], Int, Int]:
    var p = params(config)
    var z = List[Float32](length=len(p), fill=Float32(0.0))
    var flags = List[Bool](length=config.n_tensors(), fill=False)
    var opt = OptimizerConfig(OPT_ADAMW, Float32(1e-3), Float32(0.9),
        Float32(0.999), Float32(1e-8), Float32(0.0), Float32(0.0),
        Float32(0.0), False, Float32(0.0))
    var tr = ByteTrainer(ctx, p, z, z.copy(), flags, 0, opt, config)
    var head_cells = len(tr.buffers.logits) + len(tr.buffers.ce_expo) + len(tr.buffers.ce_ones) + len(tr.buffers.ce_ws) + len(tr.buffers.head_ws) + len(tr.buffers.head_bwd_ws)
    if config.chunked_lm_head_v2:
        if len(tr.buffers.logits) != config.batch * config.length * min(config.vocab_size, LM_HEAD_V2_CHUNK) or len(tr.buffers.ce_expo) != 1 or len(tr.buffers.ce_dlogits) != 1:
            raise Error("chunked LM-head v2 retained a full V1 tensor")
    var start = Int(perf_counter_ns())
    var result = byte_train_step_resident(ctx, tr, ids(config))
    var elapsed = Int(perf_counter_ns()) - start
    var state = download_f32(ctx, tr.buffers.param, config.n_total())
    return (result.loss, state^, elapsed, head_cells)


def main() raises:
    var ctx = DeviceContext()
    var v1 = ByteConfig(1, 8, 16, 2, 1, 8, 24, 1, 513, False)
    var v2 = ByteConfig(1, 8, 16, 2, 1, 8, 24, 1, 513, True)
    var a = run(ctx, v2)
    var b = run(ctx, v2)
    if a[0].to_bits() != b[0].to_bits():
        raise Error("chunked LM-head v2 repeated loss moved")
    for i in range(len(a[1])):
        if a[1][i].to_bits() != b[1][i].to_bits():
            raise Error("chunked LM-head v2 repeated update moved at " + String(i))
    var base = run(ctx, v1)
    if abs(a[0] - base[0]) > Float32(2e-5):
        raise Error("chunked LM-head v2 loss quality differs from V1")
    print("v1_ns", base[2], "v2_ns", a[2])
    print("v1_logits_cells", v1.batch * v1.length * v1.vocab_size,
          "v2_logits_cells", v2.batch * v2.length * min(v2.vocab_size, LM_HEAD_V2_CHUNK))
    print("v1_head_cells", base[3], "v2_head_cells", a[3])
    var large1 = ByteConfig(1, 64, 64, 8, 2, 8, 128, 1, 8192, False)
    var large2 = ByteConfig(1, 64, 64, 8, 2, 8, 128, 1, 8192, True)
    var la = run(ctx, large1)
    var lb = run(ctx, large2)
    if abs(la[0] - lb[0]) > Float32(2e-5):
        raise Error("large chunked LM-head v2 loss quality differs from V1")
    print("large_v1_ns", la[2], "large_v2_ns", lb[2])
    print("large_v1_head_cells", la[3], "large_v2_head_cells", lb[3])
    print("BYTE_LM_HEAD_V2_INTEGRATION_OK")
