# SPDX-License-Identifier: Apache-2.0
"""Cloud-only complete model-pool vs single-device ordered-replay bits."""
from std.os import getenv
from std.memory import bitcast
from training.byte_lm_config import ByteConfig
from training.byte_lm_model_pool import ByteModelPool
from training.byte_lm_parallel import ByteParallelTrainer
from training.checks.byte_lm_layer_pool_check import same
from training.checks.train_loop import download_f32
from training.checks.optimizer_oracle import OptimizerConfig, OPT_ADAMW


def check(shape: ByteConfig, devices: List[Int], logical: Int) raises:
    var n = shape.n_total()
    var o = shape.offsets()
    var p = List[Float32](length=n,fill=Float32(0))
    var m = List[Float32](length=n,fill=Float32(0))
    var v = List[Float32](length=n,fill=Float32(0))
    for i in range(n):
        p[i] = Float32((i*37)%127-63)/Float32(4096)
        m[i] = Float32(i%17-8)/Float32(8192)
        v[i] = Float32(i%13+1)/Float32(16384)
    for layer in range(shape.n_layers):
        for j in range(9):
            if j == 0 or j == 5:
                for i in range(o[1+9*layer+j],o[2+9*layer+j]):
                    p[i] = Float32(1)+Float32(i%7)/Float32(128)
    var flags = List[Bool](length=shape.n_tensors(),fill=True)
    var opt = OptimizerConfig(OPT_ADAMW,0.001,0.9,0.999,1e-8,0.07,0,0,False,0)
    var reference = ByteParallelTrainer()
    reference.open([0],logical,p,m,v,flags,7,opt,shape,False)
    var pool = ByteModelPool()
    pool.open(devices,logical,p,m,v,flags,7,opt,shape)
    var counts = List[Int](length=len(devices),fill=0)
    for i in range(len(pool.chunks)):
        counts[pool.chunks[i].owner] += len(pool.chunks[i].p)
    var total = 0
    for count in counts:
        total += count
        if count <= 0 or (len(devices)>1 and count >= n):
            raise Error("model parameter storage is not partitioned")
    if total != n:
        raise Error("model parameter ownership sum")
    for step in range(3):
        var shards = List[List[Int32]]()
        for shard in range(logical):
            var ids = List[Int32]()
            for i in range(shape.batch*(shape.length+1)):
                ids.append(Int32((i*17+shard*13+step*7)%shape.vocab_size))
            shards.append(ids^)
        var loss_one = reference.step(shards)
        var loss_pool = pool.step(shards)
        same(loss_one,loss_pool,"loss")
        same(download_f32(reference.contexts[0],reference.trainers[0].buffers.param,n),pool.export_values(0),"parameters")
        same(download_f32(reference.contexts[0],reference.trainers[0].buffers.m_state,n),pool.export_values(1),"first moments")
        same(download_f32(reference.contexts[0],reference.trainers[0].buffers.v_state,n),pool.export_values(2),"second moments")
        same(download_f32(reference.contexts[0],reference.trainers[0].buffers.grad,n),pool.export_values(3),"gradient")
        if pool.completed != reference.trainers[0].completed_steps:
            raise Error("completed step mismatch")
        for i in range(len(flags)):
            if pool.flags[i] != reference.trainers[0].buffers.buf_initialized[i]:
                raise Error("optimizer flags mismatch")
        # Reject invalid final microbatch before any state changes.
        var before = pool.export_values(0)
        shards[logical-1][0] = Int32(shape.vocab_size)
        var refused = False
        try:
            _ = pool.step(shards)
        except:
            refused = True
        if not refused:
            raise Error("invalid final tokens admitted")
        same(before,pool.export_values(0),"invalid tokens atomicity")
        # A caller that cannot publish a successful step can roll it back.
        pool.rollback()
        reference.rollback()
        same(download_f32(reference.contexts[0],reference.trainers[0].buffers.param,n),pool.export_values(0),"rollback parameters")
        same(download_f32(reference.contexts[0],reference.trainers[0].buffers.m_state,n),pool.export_values(1),"rollback first moments")
        same(download_f32(reference.contexts[0],reference.trainers[0].buffers.v_state,n),pool.export_values(2),"rollback second moments")
        shards[logical-1][0] = Int32(((logical-1)*13+step*7)%shape.vocab_size)
        same(reference.step(shards),pool.step(shards),"replay after rollback")
        same(download_f32(reference.contexts[0],reference.trainers[0].buffers.param,n),pool.export_values(0),"replay parameters")
    print("PASS model pool",shape.n_layers,shape.d_model,shape.length,len(devices),logical,"owned",total)
    _ = pool^
    _ = reference^


def main() raises:
    if String(getenv("RUNPOD_POD_ID")) == "":
        raise Error("cloud host required")
    check(ByteConfig(batch=2,length=7,d_model=16,n_heads=2,n_kv=1,head_dim=8,intermediate=24,n_layers=1),[0,1],1)
    check(ByteConfig(batch=2,length=7,d_model=16,n_heads=2,n_kv=1,head_dim=8,intermediate=24,n_layers=2),[0],3)
    check(ByteConfig(batch=2,length=7,d_model=16,n_heads=2,n_kv=1,head_dim=8,intermediate=24,n_layers=2),[0,1],3)
    check(ByteConfig(batch=2,length=7,d_model=24,n_heads=3,n_kv=1,head_dim=8,intermediate=40,n_layers=3),[1,0],5)
    check(ByteConfig(batch=1,length=33,d_model=32,n_heads=4,n_kv=2,head_dim=8,intermediate=64,n_layers=4),[0,1],1)
