# SPDX-License-Identifier: Apache-2.0
"""Fused finite-status snapshot operator and rollback/refusal contract.
The candidate never mutates optimizer state: callers gate commit on the
returned field/index. Kernel outputs ordered integer contributions that can
also be emitted directly by an update kernel while values are live.
Opt-in OOP Adam now emits status while words are live and the full
byte-LM step validates it before completion, preserving existing rollback."""
from std.gpu import block_idx, thread_idx
from std.sys.compile import is_defined
from std.memory import bitcast
from max.gpu.host import DeviceBuffer, DeviceContext
from training.checks.optimizer_check import clause_f
from training.checks.step_glue_check import clause_update,clause_step,clause_refusal,clause_rollback,_new_trainer,_set_arm,_opt_cfg,_ids
from training.byte_lm import ByteConfig,byte_train_step_resident
comptime TILE = 128

def status_contributions(fields: MutPointer[Float32, MutAnyOrigin], n: Int32, contributions: MutPointer[Int32, MutAnyOrigin]):
    var task = Int(block_idx.x) * TILE + Int(thread_idx.x)
    var lo = task * 32
    var hi = min(lo + 32, Int(n))
    var first = Int32(n)
    for i in range(lo, hi):
        if (bitcast[DType.uint32](fields[i]) & UInt32(0x7f800000)) == UInt32(0x7f800000):
            first = Int32(i)
            break
    contributions[task] = first

def merge_status(parts: MutPointer[Int32, MutAnyOrigin], tasks: Int32, output: MutPointer[Int32, MutAnyOrigin]):
    # Independent fixed 32-contribution tiles; final tiny scalar read is
    # error reporting only, never a data arithmetic runtime fallback.
    var tile = Int(block_idx.x) * TILE + Int(thread_idx.x)
    var lo = tile * 32
    if lo >= Int(tasks):
        return
    var first = Int32(2147483647)
    for i in range(lo, min(lo + 32, Int(tasks))):
        first = min(first, parts[i])
    output[tile] = first

def check(ctx: DeviceContext, count: Int, bad: Int) raises:
    var host = ctx.enqueue_create_host_buffer[DType.float32](count)
    for i in range(count):
        host[i] = Float32(i % 13) * Float32(0.125)
    if bad >= 0:
        host[bad] = bitcast[DType.float32](UInt32(0x7f800001))
    var data = ctx.enqueue_create_buffer[DType.float32](count)
    ctx.enqueue_copy(dst_buf=data, src_buf=host)
    var tasks = (count + 31) // 32
    # Rounded launch storage ensures inactive threads can write sentinels.
    var allocated = ((tasks + TILE - 1) // TILE) * TILE
    var parts = ctx.enqueue_create_buffer[DType.int32](allocated)
    ctx.enqueue_function[status_contributions](data.unsafe_ptr(), Int32(count), parts.unsafe_ptr(), grid_dim=(allocated // TILE,1,1), block_dim=(TILE,1,1))
    var current = tasks
    while current > 1:
        var next_count = (current + 31) // 32
        var merged = ctx.enqueue_create_buffer[DType.int32](next_count)
        ctx.enqueue_function[merge_status](parts.unsafe_ptr(), Int32(current), merged.unsafe_ptr(), grid_dim=((next_count + TILE - 1)//TILE,1,1), block_dim=(TILE,1,1))
        ctx.synchronize()
        parts = merged^
        current = next_count
    var result = ctx.enqueue_create_host_buffer[DType.int32](1)
    ctx.enqueue_copy(dst_ptr=result.unsafe_ptr(), src_buf=parts.create_sub_buffer[DType.int32](0,1))
    ctx.synchronize()
    var want = bad if bad >= 0 else count
    if Int(result[0]) != want:
        raise Error("I10 first offending field/index moved")
    _ = data^; _ = parts^

def check_live_reach(ctx: DeviceContext) raises:
    var cfg = _opt_cfg(Float32(1.0e-3),Float32(0.9),Float32(0.999),Float32(0.01))
    for arm in [String("shipped"),String("noshadow")]:
        _set_arm(arm,False)
        var tr = _new_trainer(ctx,cfg)
        var config = ByteConfig()
        for step in range(2):
            _ = byte_train_step_resident(ctx,tr,_ids(config,step))
        var expected = 0
        # NEVER RUN — PENDING MEASUREMENT
        comptime if is_defined["MOJOLEARN_TRAIN_LIVE_STATUS"]():
            if arm==String("noshadow"):
                expected = 2
        if tr.live_status_steps!=expected:
            raise Error("I10 live-status candidate did not reach expected full steps")
    _set_arm(String(""),False)

# NEVER RUN — PENDING MEASUREMENT
def main() raises:
    var ctx = DeviceContext()
    for n in [31, 33, 4099]:
        check(ctx,n,-1)
        check(ctx,n,0)
        check(ctx,n,n-1)
    check_live_reach(ctx)
    clause_f(ctx)
    var failures = clause_update(ctx)
    failures += clause_step(ctx)
    failures += clause_refusal(ctx)
    failures += clause_rollback(ctx)
    if failures!=0:
        raise Error("I10 live-status full-step/rollback gates failed")
    print("I10 PASS fused_snapshot=9 live_update complete_step refusal_rollback")
