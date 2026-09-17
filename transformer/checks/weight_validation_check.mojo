# SPDX-License-Identifier: Apache-2.0
"""Host-list weight construction, batch counters and explicit device revalidation."""
from std.memory import bitcast
from std.sys.compile import is_defined
from max.gpu.host import DeviceContext
from core.step_phase import step_counts_now
from transformer.impl.llama.modeling_llama import LlamaDims, LlamaDeviceWeights


def main() raises:
    comptime assert is_defined["MOJOLEARN_STEP_PHASE_TIMERS"](), "counters required"
    var ctx = DeviceContext()
    var dims = LlamaDims(32, 2, 1, 16, 64)
    var norm = List[Float32](length=32, fill=Float32(1))
    var square = List[Float32](length=1024, fill=Float32(0.125))
    var kv = List[Float32](length=512, fill=Float32(0.25))
    var mlp = List[Float32](length=2048, fill=Float32(0.0625))
    var w = LlamaDeviceWeights(ctx, dims, Float32(0.00001), norm, norm,
        square, kv, kv, square, mlp, mlp, mlp)
    var before = step_counts_now()
    w._validate_finite(ctx)
    var after = step_counts_now()
    if after.syncs - before.syncs != 1 or after.d2h - before.d2h != 1:
        raise Error("weight validation must have one wait and one result copy")
    if after.device_allocs - before.device_allocs != 1 or after.host_allocs - before.host_allocs != 1:
        raise Error("weight validation must share one device and one host scratch allocation")
    if after.launches - before.launches != 9:
        raise Error("all nine original scans must run")
    w.w_q.enqueue_fill(bitcast[DType.float32](UInt32(0x7FC01234)))
    var refused = False
    try:
        w._validate_finite(ctx)
    except e:
        refused = String(e) == "llama: NaN in q_proj.weight at flat index 0 REFUSED (row 39: NaN payloads are vendor-shaped; no stage may record one)"
    if not refused:
        raise Error("edited device weights must be revalidated")
    w.w_q.enqueue_fill(Float32(0.125))
    w._validate_finite(ctx)
    _ = w^
    _ = ctx^
    print("PASS host-list constructor; nine scans, one wait/copy, two allocations; mutated device weight refused and recovery passed")
