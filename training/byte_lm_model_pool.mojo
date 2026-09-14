# SPDX-License-Identifier: Apache-2.0
"""Layer-partitioned byte LM, with canonical ordered microbatch gradients.

One model spans the device contexts. Decoder weights, moments, gradients and
rollback state have one owner. Embedding/head live on the first device. The
initial schedule is sequential; pooling capacity does not imply speedup.
"""
from max.gpu.host import DeviceContext, DeviceBuffer
from core.device_scan import DeviceScanScratch
from training.byte_lm import (
    byte_validate_state, byte_validate_optimizer, byte_validate_tokens,
    _require_profile, _byte_validate_allocations, _require_device_finite,
    _require_finite, byte_glue_update_launch, _FAULT_NAN, _FAULT_INF, _FAULT_MINUS_ONE,
)
from training.byte_lm_optimizer_pool import pool_maybe_fault
from training.byte_lm_config import ByteConfig
from training.byte_lm_layer_pool import ByteLayerPool
from training.byte_lm_pooled_head import BytePooledHead
from training.byte_lm_parallel import _ordered_add_kernel
from training.checks.train_loop import _zeros, _upload, _copy_into, download_f32
from training.checks.optimizer import OPT_RECORD_INTERMEDIATES
from training.checks.optimizer_oracle import OptimizerConfig
from embedding.checks.embedding_oracle import EmbConfig
from embedding.checks.embedding_identical import identical_embedding_forward_into, identical_embedding_backward_into
from gemm.checks.gemm_oracle import OP_NT
from gemm.checks.gemm_identical import identical_gemm_into
from gemm.checks.gemm_backward import identical_gemm_backward_a_into, identical_gemm_backward_b_into
from training.checks.loss import identical_ce_forward_into, identical_ce_backward_into
from training.checks.loss_oracle import CeConfig


def _host_range(values: List[Float32], first: Int, n: Int) -> List[Float32]:
    var result = List[Float32]()
    for i in range(first,first+n):
        result.append(values[i])
    return result^


struct ByteModelChunk(Movable):
    var owner: Int
    var first: Int
    var p: DeviceBuffer[DType.float32]
    var m: DeviceBuffer[DType.float32]
    var v: DeviceBuffer[DType.float32]
    var g: DeviceBuffer[DType.float32]
    var shadow_p: DeviceBuffer[DType.float32]
    var shadow_m: DeviceBuffer[DType.float32]
    var shadow_v: DeviceBuffer[DType.float32]
    var scratch: DeviceBuffer[DType.float32]
    var scratch2: DeviceBuffer[DType.float32]
    var scan: DeviceScanScratch

    def __init__(out self, ctx: DeviceContext, owner: Int, first: Int, n: Int,
                 p: List[Float32], m: List[Float32], v: List[Float32]) raises:
        self.owner = owner
        self.first = first
        self.p = _upload(ctx,_host_range(p,first,n))
        self.m = _upload(ctx,_host_range(m,first,n))
        self.v = _upload(ctx,_host_range(v,first,n))
        self.g = _zeros(ctx,n)
        self.shadow_p = _zeros(ctx,n)
        self.shadow_m = _zeros(ctx,n)
        self.shadow_v = _zeros(ctx,n)
        self.scratch = _zeros(ctx,1)
        self.scratch2 = _zeros(ctx,1)
        self.scan = DeviceScanScratch(ctx)
        ctx.synchronize()

    def validate(mut self, ctx: DeviceContext) raises:
        var n = len(self.p)
        _require_device_finite(ctx,self.scan,self.p,n,"parameters")
        _require_device_finite(ctx,self.scan,self.m,n,"first moments")
        _require_device_finite(ctx,self.scan,self.v,n,"second moments")
        if self.scan.first_negative(ctx,self.v,n) >= 0:
            raise Error("byte model pool: negative second moment")

    def fold(mut self, ctx: DeviceContext, mut source: DeviceBuffer[DType.float32], index: Int) raises:
        var n = len(self.g)
        _require_device_finite(ctx,self.scan,source,n,"gradients")
        if index == 0:
            _copy_into(ctx,self.g,source,0,0,n)
        else:
            ctx.enqueue_function[_ordered_add_kernel](self.g,source,Int32(n),grid_dim=(n+255)//256,block_dim=256)
        ctx.synchronize()
        _require_device_finite(ctx,self.scan,self.g,n,"summed gradients")

    def snapshot(mut self, ctx: DeviceContext) raises:
        self.validate(ctx)
        var n = len(self.p)
        _copy_into(ctx,self.shadow_p,self.p,0,0,n)
        _copy_into(ctx,self.shadow_m,self.m,0,0,n)
        _copy_into(ctx,self.shadow_v,self.v,0,0,n)
        ctx.synchronize()

    def restore(mut self, ctx: DeviceContext) raises:
        var n = len(self.p)
        _copy_into(ctx,self.p,self.shadow_p,0,0,n)
        _copy_into(ctx,self.m,self.shadow_m,0,0,n)
        _copy_into(ctx,self.v,self.shadow_v,0,0,n)
        ctx.synchronize()
        self.validate(ctx)


struct ByteModelPool(Movable, Writable):
    var layers: ByteLayerPool
    var head_context: Optional[DeviceContext]
    var head: Optional[BytePooledHead]
    var chunks: List[ByteModelChunk]
    var config: ByteConfig
    var optimizer: OptimizerConfig
    var flags: List[Bool]
    var completed: Int
    var logical_shards: Int
    var shadow_valid: Bool
    var shadow_step: Int
    var gradient_step: Int
    var busy: Bool
    var usable: Bool

    def __init__(out self):
        self.layers = ByteLayerPool()
        self.head_context = Optional[DeviceContext]()
        self.head = Optional[BytePooledHead]()
        self.chunks = List[ByteModelChunk]()
        self.config = ByteConfig()
        self.optimizer = OptimizerConfig(1,0.001,0.9,0.999,1e-8,0,0,0,False,0)
        self.flags = List[Bool]()
        self.completed = 0
        self.logical_shards = 0
        self.shadow_valid = False
        self.shadow_step = -1
        self.gradient_step = -1
        self.busy = False
        self.usable = False

    def __deinit__(deinit self):
        _ = self.head^
        _ = self.chunks^
        _ = self.head_context^
        _ = self.layers^

    def write_to(self, mut writer: Some[Writer]):
        writer.write("ByteModelPool")

    def write_repr_to(self, mut writer: Some[Writer]):
        writer.write("ByteModelPool")

    def close(mut self) raises:
        if self.busy:
            raise Error("byte model pool: busy")
        self.usable = False
        self.head = None
        self.chunks = List[ByteModelChunk]()
        if self.head_context:
            self.head_context.value().synchronize()
        self.head_context = None
        self.layers = ByteLayerPool()

    def open(mut self, devices: List[Int], shards: Int, p: List[Float32],
             m: List[Float32], v: List[Float32], flags: List[Bool],
             completed: Int, optimizer: OptimizerConfig, shape: ByteConfig) raises:
        if self.busy or len(self.chunks) != 0 or len(self.layers.contexts) != 0:
            raise Error("byte model pool: already open")
        _require_profile()
        _byte_validate_allocations(shape)
        byte_validate_state(p,m,v,flags,completed,shape)
        byte_validate_optimizer(optimizer)
        if shards < 1 or shards > 1024:
            raise Error("byte model pool: logical shard count must be in [1,1024]")
        comptime if OPT_RECORD_INTERMEDIATES:
            raise Error("byte model pool: recorded optimizer intermediates unsupported")
        self.config = shape.copy()
        self.optimizer = optimizer.copy()
        self.flags = flags.copy()
        self.completed = completed
        self.logical_shards = shards
        self.layers.open(devices,p,shape,reserve_head_device=True)
        self.head_context = DeviceContext(device_id=devices[0])
        self.head = BytePooledHead(self.head_context.value(),shape)
        var o = shape.offsets()
        self.chunks.append(ByteModelChunk(self.layers.contexts[0],0,0,o[1],p,m,v))
        for layer in range(shape.n_layers):
            var owner = self.layers.layers[layer].owner
            var first = o[1+9*layer]
            self.chunks.append(ByteModelChunk(self.layers.contexts[owner],owner,first,o[10+9*layer]-first,p,m,v))
        var last = o[shape.n_tensors()-1]
        self.chunks.append(ByteModelChunk(self.layers.contexts[0],0,last,shape.n_total()-last,p,m,v))
        self.usable = True

    def require_open(self) raises:
        if not self.usable or self.busy:
            raise Error("byte model pool: closed, busy or failed; restore an export")

    def refresh_weights(mut self) raises:
        ref ctx = self.layers.contexts[0]
        ref h = self.head.value()
        var n = self.config.vocab_size*self.config.d_model
        _copy_into(ctx,h.emb_w,self.chunks[0].p,0,0,n)
        _copy_into(ctx,h.lm_w,self.chunks[len(self.chunks)-1].p,0,0,n)
        ctx.synchronize()
        for i in range(self.config.n_layers):
            ref layer = self.layers.layers[i]
            ref source = self.chunks[i+1].p
            ref c = self.layers.contexts[layer.owner]
            var o = layer.offsets.copy()
            _copy_into(c,layer.weights.norm1_w,source,0,o[0],o[1]-o[0])
            _copy_into(c,layer.weights.w_q,source,0,o[1],o[2]-o[1])
            _copy_into(c,layer.weights.w_k,source,0,o[2],o[3]-o[2])
            _copy_into(c,layer.weights.w_v,source,0,o[3],o[4]-o[3])
            _copy_into(c,layer.weights.w_o,source,0,o[4],o[5]-o[4])
            _copy_into(c,layer.weights.norm2_w,source,0,o[5],o[6]-o[5])
            _copy_into(c,layer.weights.w_gate,source,0,o[6],o[7]-o[6])
            _copy_into(c,layer.weights.w_up,source,0,o[7],o[8]-o[7])
            _copy_into(c,layer.weights.w_down,source,0,o[8],o[9]-o[8])
            c.synchronize()

    def gradient(mut self, ids: List[Int32], logical: Int) raises -> Float32:
        var config = self.config.copy()
        var M = config.batch*config.length
        # The contexts stay owned by layers throughout both graph calls.
        # A persistent separate head context on the same device avoids borrowing
        # the layer pool itself across its mutating forward/backward methods.
        ref ctx = self.head_context.value()
        ref h = self.head.value()
        var hi = ctx.enqueue_create_host_buffer[DType.int32](M)
        var ht = ctx.enqueue_create_host_buffer[DType.int32](M)
        ctx.synchronize()
        for b in range(config.batch):
            for l in range(config.length):
                hi.unsafe_ptr().unsafe_store(b*config.length+l,ids[b*(config.length+1)+l])
                ht.unsafe_ptr().unsafe_store(b*config.length+l,ids[b*(config.length+1)+l+1])
        ctx.enqueue_copy(dst_buf=h.ids,src_ptr=hi.unsafe_ptr())
        ctx.enqueue_copy(dst_buf=h.targets,src_ptr=ht.unsafe_ptr())
        ctx.synchronize()
        _ = hi^
        _ = ht^
        var emb = EmbConfig.llama(config.vocab_size,config.d_model)
        var ce = CeConfig.causal_lm(config.vocab_size)
        identical_embedding_forward_into(ctx,h.x,h.emb_w,h.ids,M,emb)
        ctx.synchronize()
        self.layers.forward_into(ctx,h.x,h.final_hidden)
        identical_gemm_into(ctx,h.logits,h.final_hidden,h.lm_w,h.head_ws,M,config.vocab_size,config.d_model,OP_NT)
        ctx.synchronize()
        identical_ce_forward_into(ctx,h.ce_max,h.ce_shift,h.ce_expo,h.ce_denom,h.ce_logdenom,
            h.ce_logp_target,h.ce_nll,h.ce_logp,h.ce_logp_sum,h.ce_smooth,h.ce_row,
            h.ce_total,h.ce_loss,h.logits,h.targets,h.ce_ones,h.ce_ws,M,M,ce)
        ctx.synchronize()
        var loss = download_f32(ctx,h.ce_loss,1)
        _require_finite(loss,"loss")
        identical_ce_backward_into(ctx,h.ce_weights,h.ce_dlogits,h.ce_expo,h.ce_denom,h.ce_logp,h.targets,M,M,ce)
        ctx.synchronize()
        identical_gemm_backward_a_into(ctx,h.d_h,h.ce_dlogits,h.lm_w,h.head_bwd_ws,M,config.vocab_size,config.d_model,OP_NT)
        identical_gemm_backward_b_into(ctx,h.dw_lm,h.ce_dlogits,h.final_hidden,h.head_bwd_ws,M,config.vocab_size,config.d_model,OP_NT)
        ctx.synchronize()
        # x is no longer needed: the layer owner saved its input activation.
        self.layers.backward_into(ctx,h.d_h,h.x)
        identical_embedding_backward_into(ctx,h.dw_emb,h.x,h.ids,h.emb_counts,h.emb_run_begin,h.emb_perm,M,emb)
        ctx.synchronize()
        self.chunks[0].fold(ctx,h.dw_emb,logical)
        self.chunks[len(self.chunks)-1].fold(ctx,h.dw_lm,logical)
        self.layers.accumulate(logical)
        return loss[0]

    def rollback(mut self) raises:
        var lost = not self.layers.usable
        if self.shadow_valid:
            for i in range(len(self.chunks)):
                try:
                    self.chunks[i].restore(self.layers.contexts[self.chunks[i].owner])
                except:
                    lost = True
            self.completed = self.shadow_step
        self.shadow_valid = False
        self.gradient_step = -1
        if lost:
            self.usable = False
            raise Error("byte model pool: recovery failed; restore canonical state")

    def step(mut self, shards: List[List[Int32]]) raises -> List[Float32]:
        self.require_open()
        if len(shards) != self.logical_shards or self.completed >= 999999:
            raise Error("byte model pool: logical shard count or step bound")
        for i in range(len(shards)):
            byte_validate_tokens(shards[i],self.config)
        self.busy = True
        self.shadow_valid = False
        self.gradient_step = -1
        var losses = List[Float32]()
        try:
            self.refresh_weights()
            for i in range(len(shards)):
                losses.append(self.gradient(shards[i],i))
            for i in range(self.config.n_layers):
                var owner = self.chunks[i+1].owner
                _copy_into(self.layers.contexts[owner],self.chunks[i+1].g,self.layers.layers[i].total,0,0,len(self.chunks[i+1].g))
                self.layers.contexts[owner].synchronize()
            for i in range(len(self.chunks)):
                ref chunk = self.chunks[i]
                ref ctx = self.layers.contexts[chunk.owner]
                pool_maybe_fault(ctx,chunk.g,"grad_nonfinite",0,_FAULT_NAN,chunk.first)
                _require_device_finite(ctx,chunk.scan,chunk.g,len(chunk.g),"summed gradients")
            for i in range(len(self.chunks)):
                self.chunks[i].snapshot(self.layers.contexts[self.chunks[i].owner])
            self.shadow_step = self.completed
            self.shadow_valid = True
            for i in range(len(self.chunks)):
                ref chunk = self.chunks[i]
                ref ctx = self.layers.contexts[chunk.owner]
                pool_maybe_fault(ctx,chunk.m,"opt_refuse",min(5,len(chunk.m)-1),_FAULT_NAN,chunk.first)
                chunk.validate(ctx)
                byte_glue_update_launch(ctx,chunk.p,chunk.g,chunk.m,chunk.v,
                    chunk.shadow_p,chunk.shadow_m,chunk.shadow_v,chunk.scratch,chunk.scratch2,
                    len(chunk.p),self.optimizer,self.completed+1,False)
                pool_maybe_fault(ctx,chunk.v,"after_nonfinite",min(3,len(chunk.v)-1),_FAULT_INF,chunk.first)
                pool_maybe_fault(ctx,chunk.v,"after_negative",min(3,len(chunk.v)-1),_FAULT_MINUS_ONE,chunk.first)
                chunk.validate(ctx)
            self.completed += 1
            self.gradient_step = self.completed
        except error:
            self.busy = False
            self.rollback()
            raise error
        self.busy = False
        return losses^

    def export_values(mut self, slot: Int) raises -> List[Float32]:
        self.require_open()
        if slot < 0 or slot > 3 or (slot == 3 and self.gradient_step != self.completed):
            raise Error("byte model pool: invalid export slot or no committed gradient")
        var result = List[Float32]()
        for i in range(len(self.chunks)):
            ref chunk = self.chunks[i]
            ref ctx = self.layers.contexts[chunk.owner]
            chunk.validate(ctx)
            var values = List[Float32]()
            if slot == 0:
                values = download_f32(ctx,chunk.p,len(chunk.p))
            elif slot == 1:
                values = download_f32(ctx,chunk.m,len(chunk.m))
            elif slot == 2:
                values = download_f32(ctx,chunk.v,len(chunk.v))
            else:
                values = download_f32(ctx,chunk.g,len(chunk.g))
            for value in values:
                result.append(value)
        return result^
