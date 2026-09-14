# SPDX-License-Identifier: Apache-2.0
"""Cloud-only layer ownership, activation/cotangent and gradient-bit gate."""
from std.os import getenv
from std.memory import bitcast
from max.gpu.host import DeviceContext, DeviceBuffer
from core.identity_trace import IdentityTrace
from training.byte_lm import ByteTrainer, _pack_block
from training.byte_lm_layer_pool import ByteLayerPool
from training.byte_lm_config import ByteConfig
from training.byte_lm_parallel import _ordered_add_kernel
from training.checks.optimizer_oracle import OptimizerConfig, OPT_ADAMW
from training.checks.train_loop import _upload, _zeros, _copy_into, download_f32
from transformer.impl.llama.modeling_llama import llama_decoder_layer_forward
from transformer.checks.transformer_backward import llama_decoder_layer_backward_device


def same(a: List[Float32], b: List[Float32], label: String) raises:
    if len(a) != len(b):
        raise Error(label+": length mismatch")
    for i in range(len(a)):
        if bitcast[DType.uint32](a[i]) != bitcast[DType.uint32](b[i]):
            raise Error(label+": differing bits at "+String(i))


def check(shape: ByteConfig, devices: List[Int], logical: Int) raises:
    var n = shape.n_total()
    var o = shape.offsets()
    var p = List[Float32](length=n,fill=Float32(0))
    for i in range(n):
        p[i] = Float32((i*37)%127-63)/Float32(4096)
    for layer in range(shape.n_layers):
        for j in range(9):
            if j == 0 or j == 5:
                for i in range(o[1+9*layer+j],o[2+9*layer+j]):
                    p[i] = Float32(1)+Float32(i%7)/Float32(128)
    var zero = List[Float32](length=n,fill=Float32(0))
    var flags = List[Bool](length=shape.n_tensors(),fill=False)
    var ctx = DeviceContext(device_id=0)
    var reference = ByteTrainer(ctx,p,zero,zero,flags,0,OptimizerConfig(OPT_ADAMW,0.001,0.9,0.999,1e-8,0.01,0,0,False,0),shape)
    var pool = ByteLayerPool()
    pool.open(devices,p,shape)
    var m = shape.batch*shape.length*shape.d_model
    var x = List[Float32](length=m,fill=Float32(0))
    var dy = List[Float32](length=m,fill=Float32(0))
    var y_pool = _zeros(ctx,m)
    var dx_pool = _zeros(ctx,m)
    var total = _zeros(ctx,n)
    var trace = IdentityTrace.disabled()
    # Ownership counts are derived from actual resident weight buffers, not
    # theoretical registry partitions. No device stores all decoder layers.
    var counts = List[Int](length=len(devices),fill=0)
    for i in range(len(pool.layers)):
        var owner = pool.layers[i].owner
        counts[owner] += len(pool.layers[i].weights.norm1_w)+len(pool.layers[i].weights.w_q)+len(pool.layers[i].weights.w_k)+len(pool.layers[i].weights.w_v)+len(pool.layers[i].weights.w_o)+len(pool.layers[i].weights.norm2_w)+len(pool.layers[i].weights.w_gate)+len(pool.layers[i].weights.w_up)+len(pool.layers[i].weights.w_down)
    var owned = 0
    for count in counts:
        if count <= 0 or (len(devices)>1 and count >= o[shape.n_tensors()-1]-o[1]):
            raise Error("layer weights are not disjoint")
        owned += count
    if owned != o[shape.n_tensors()-1]-o[1]:
        raise Error("layer ownership sum mismatch")
    for shard in range(logical):
        for i in range(m):
            x[i] = Float32((i*29+shard*13)%101-50)/Float32(64)
            dy[i] = Float32((i*17+shard*7)%73-36)/Float32(1024)
        var xd = _upload(ctx,x)
        var dyd = _upload(ctx,dy)
        pool.forward_into(ctx,xd,y_pool)
        for i in range(shape.n_layers):
            var st = reference.forward.pop(i)
            reference.prefill_cache.s = 0
            if i == 0:
                llama_decoder_layer_forward(ctx,st,reference.prefill_cache,reference.rope,
                    reference.weights[i],xd,shape.batch,shape.length,0,trace,"reference")
            else:
                llama_decoder_layer_forward(ctx,st,reference.prefill_cache,reference.rope,
                    reference.weights[i],reference.forward[i-1].residual2,shape.batch,shape.length,0,trace,"reference")
            ctx.synchronize()
            same(download_f32(ctx,st.residual2,m),download_f32(pool.contexts[pool.layers[i].owner],pool.layers[i].forward.residual2,m),"layer forward")
            reference.forward.insert(i,st^)
        same(download_f32(ctx,reference.forward[shape.n_layers-1].residual2,m),download_f32(ctx,y_pool,m),"forward transfer")
        pool.backward_into(ctx,dyd,dx_pool)
        for i in range(shape.n_layers-1,-1,-1):
            var st = reference.forward.pop(i)
            var bw = reference.backward.pop(i)
            # Copy reference inputs/cotangents only; preserve each original
            # decoder kernel and its full shape, just as ByteTrainer does.
            if i == 0:
                _copy_into(ctx,reference.buffers.x,xd,0,0,m)
            else:
                _copy_into(ctx,reference.buffers.x,reference.forward[i-1].residual2,0,0,m)
            if i == shape.n_layers-1:
                _copy_into(ctx,reference.buffers.d_h,dyd,0,0,m)
            else:
                _copy_into(ctx,reference.buffers.d_h,reference.backward[i].d_x,0,0,m)
            ctx.synchronize()
            llama_decoder_layer_backward_device(ctx,bw,st,reference.weights[i],
                reference.rope.cos,reference.rope.sin,reference.buffers.x,reference.buffers.d_h,
                shape.batch,shape.length,0,trace,"reference")
            ctx.synchronize()
            same(download_f32(ctx,bw.d_x,m),download_f32(pool.contexts[pool.layers[i].owner],pool.layers[i].backward.d_x,m),"layer backward")
            _pack_block(ctx,reference.buffers,bw,i)
            reference.forward.insert(i,st^)
            reference.backward.insert(i,bw^)
        same(download_f32(ctx,reference.backward[0].d_x,m),download_f32(ctx,dx_pool,m),"backward transfer")
        pool.accumulate(shard)
        if shard == 0:
            _copy_into(ctx,total,reference.buffers.grad,0,0,n)
        else:
            ctx.enqueue_function[_ordered_add_kernel](total,reference.buffers.grad,Int32(n),grid_dim=(n+255)//256,block_dim=256)
        ctx.synchronize()
        var all = download_f32(ctx,total,n)
        var expected = List[Float32]()
        for j in range(o[1],o[shape.n_tensors()-1]):
            expected.append(all[j])
        same(expected,pool.export_gradients(),"canonical ordered gradients")
        # Reusing a gradient must refuse without destroying the valid sum.
        var refused = False
        try:
            pool.accumulate(shard+1)
        except:
            refused = True
        if not refused:
            raise Error("duplicate accumulation admitted")
        same(expected,pool.export_gradients(),"refusal atomicity")
    print("PASS layer pool",shape.n_layers,shape.d_model,shape.length,len(devices),logical,"owned",owned)
    _ = pool^
    _ = reference^
    _ = y_pool^
    _ = dx_pool^
    _ = total^
    ctx.synchronize()


def main() raises:
    if String(getenv("RUNPOD_POD_ID")) == "":
        raise Error("cloud host required")
    check(ByteConfig(batch=2,length=7,d_model=16,n_heads=2,n_kv=1,head_dim=8,intermediate=24,n_layers=2),[0],3)
    check(ByteConfig(batch=2,length=7,d_model=16,n_heads=2,n_kv=1,head_dim=8,intermediate=24,n_layers=2),[0,1],3)
    check(ByteConfig(batch=2,length=7,d_model=24,n_heads=3,n_kv=1,head_dim=8,intermediate=40,n_layers=3),[1,0],5)
    check(ByteConfig(batch=1,length=33,d_model=32,n_heads=4,n_kv=2,head_dim=8,intermediate=64,n_layers=4),[0,1],2)
