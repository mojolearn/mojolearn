# SPDX-License-Identifier: Apache-2.0
"""One-GPU ordered replay with host state and one decoder layer resident.

All arithmetic reuses the original GPU kernels. Host arrays store canonical
state, saved activations and gradient sums. Backward recomputes one layer's
forward stages. This trades transfers/recomputation for device capacity.
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

from core.identity_trace import IdentityTrace
from training.byte_lm import byte_dims
from training.byte_lm_layer_pool import ByteOwnedLayer
from training.byte_lm_model_pool import ByteModelChunk, _host_range
from transformer.impl.llama.modeling_llama import LlamaRopeTable, LlamaKVCache, llama_decoder_layer_forward
from transformer.checks.transformer_backward import llama_decoder_layer_backward_device


def _write_range(mut target: List[Float32], first: Int, source: List[Float32]):
    for i in range(len(source)):
        target[first+i] = source[i]


def _load_range(ctx: DeviceContext, mut target: DeviceBuffer[DType.float32],
                values: List[Float32], first: Int) raises:
    var source = _upload(ctx,_host_range(values,first,len(target)))
    _copy_into(ctx,target,source,0,0,len(target))
    ctx.synchronize()


def _fold_host(ctx: DeviceContext, mut total: List[Float32], first: Int,
               mut source: DeviceBuffer[DType.float32], logical: Int) raises:
    var n = len(source)
    var scan = DeviceScanScratch(ctx)
    _require_device_finite(ctx,scan,source,n,"offloaded gradients")
    if logical == 0:
        _write_range(total,first,download_f32(ctx,source,n))
    else:
        var previous = _upload(ctx,_host_range(total,first,n))
        ctx.enqueue_function[_ordered_add_kernel](previous,source,Int32(n),grid_dim=(n+255)//256,block_dim=256)
        ctx.synchronize()
        _require_device_finite(ctx,scan,previous,n,"offloaded gradient sum")
        _write_range(total,first,download_f32(ctx,previous,n))


def _replay_layer(ctx: DeviceContext, p: List[Float32], shape: ByteConfig,
                  index: Int, input: List[Float32], mut rope: LlamaRopeTable,
                  mut cache: LlamaKVCache) raises -> List[Float32]:
    # Function scope destroys every layer allocation before the next layer.
    var layer = ByteOwnedLayer(ctx,0,index,p,shape)
    _load_range(ctx,layer.input,input,0)
    var trace = IdentityTrace.disabled()
    cache.s = 0
    llama_decoder_layer_forward(ctx,layer.forward,cache,rope,layer.weights,layer.input,
        shape.batch,shape.length,0,trace,String("byte.block")+String(index)+".forward")
    ctx.synchronize()
    return download_f32(ctx,layer.forward.residual2,len(input))


def _replay_backward(ctx: DeviceContext, p: List[Float32], shape: ByteConfig,
                     index: Int, input: List[Float32], cotangent: List[Float32],
                     mut rope: LlamaRopeTable, mut cache: LlamaKVCache,
                     mut total: List[Float32], logical: Int) raises -> List[Float32]:
    var layer = ByteOwnedLayer(ctx,0,index,p,shape)
    _load_range(ctx,layer.input,input,0)
    _load_range(ctx,layer.cotangent,cotangent,0)
    var trace = IdentityTrace.disabled()
    cache.s = 0
    # Recompute original forward stages from the exact saved input/weights.
    llama_decoder_layer_forward(ctx,layer.forward,cache,rope,layer.weights,layer.input,
        shape.batch,shape.length,0,trace,String("byte.block")+String(index)+".forward")
    ctx.synchronize()
    llama_decoder_layer_backward_device(ctx,layer.backward,layer.forward,layer.weights,
        rope.cos,rope.sin,layer.input,layer.cotangent,shape.batch,shape.length,0,
        trace,String("byte.block")+String(index)+".backward")
    layer.pack(ctx)
    _fold_host(ctx,total,layer.first,layer.gradient,logical)
    return download_f32(ctx,layer.backward.d_x,len(input))


def _replay_update(ctx: DeviceContext, first: Int, n: Int,
                   p: List[Float32], m: List[Float32], v: List[Float32], g: List[Float32],
                   mut next_p: List[Float32], mut next_m: List[Float32], mut next_v: List[Float32],
                   opt: OptimizerConfig, step: Int) raises:
    var chunk = ByteModelChunk(ctx,0,first,n,p,m,v)
    _load_range(ctx,chunk.g,g,first)
    pool_maybe_fault(ctx,chunk.g,"grad_nonfinite",0,_FAULT_NAN,first)
    _require_device_finite(ctx,chunk.scan,chunk.g,n,"offloaded summed gradients")
    pool_maybe_fault(ctx,chunk.m,"opt_refuse",min(5,n-1),_FAULT_NAN,first)
    chunk.validate(ctx)
    byte_glue_update_launch(ctx,chunk.p,chunk.g,chunk.m,chunk.v,
        chunk.shadow_p,chunk.shadow_m,chunk.shadow_v,chunk.scratch,chunk.scratch2,
        n,opt,step,False)
    pool_maybe_fault(ctx,chunk.v,"after_nonfinite",min(3,n-1),_FAULT_INF,first)
    pool_maybe_fault(ctx,chunk.v,"after_negative",min(3,n-1),_FAULT_MINUS_ONE,first)
    chunk.validate(ctx)
    _write_range(next_p,first,download_f32(ctx,chunk.p,n))
    _write_range(next_m,first,download_f32(ctx,chunk.m,n))
    _write_range(next_v,first,download_f32(ctx,chunk.v,n))


struct ByteOffloadedReplay(Movable, Writable):
    var context: Optional[DeviceContext]
    var head: Optional[BytePooledHead]
    var rope: Optional[LlamaRopeTable]
    var cache: Optional[LlamaKVCache]
    var p: List[Float32]
    var m: List[Float32]
    var v: List[Float32]
    var g: List[Float32]
    var shadow_p: List[Float32]
    var shadow_m: List[Float32]
    var shadow_v: List[Float32]
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
        self.context = None
        self.head = None
        self.rope = None
        self.cache = None
        self.p = List[Float32]()
        self.m = List[Float32]()
        self.v = List[Float32]()
        self.g = List[Float32]()
        self.shadow_p = List[Float32]()
        self.shadow_m = List[Float32]()
        self.shadow_v = List[Float32]()
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
        _ = self.rope^
        _ = self.cache^
        _ = self.context^

    def write_to(self, mut writer: Some[Writer]):
        writer.write("ByteOffloadedReplay")

    def write_repr_to(self, mut writer: Some[Writer]):
        writer.write("ByteOffloadedReplay")

    def require_open(self) raises:
        if not self.usable or self.busy:
            raise Error("byte offloaded replay: closed, busy or lost")

    def close(mut self) raises:
        if self.busy:
            raise Error("byte offloaded replay: busy")
        self.usable = False
        self.head = None
        self.rope = None
        self.cache = None
        if self.context:
            self.context.value().synchronize()
        self.context = None
        self.p = List[Float32]()
        self.m = List[Float32]()
        self.v = List[Float32]()
        self.g = List[Float32]()
        self.shadow_p = List[Float32]()
        self.shadow_m = List[Float32]()
        self.shadow_v = List[Float32]()
        self.shadow_valid = False

    def open(mut self, devices: List[Int], shards: Int, p: List[Float32],
             m: List[Float32], v: List[Float32], flags: List[Bool],
             completed: Int, optimizer: OptimizerConfig, shape: ByteConfig) raises:
        if self.context or self.busy:
            raise Error("byte offloaded replay: already open")
        _require_profile()
        _byte_validate_allocations(shape)
        byte_validate_state(p,m,v,flags,completed,shape)
        byte_validate_optimizer(optimizer)
        if len(devices) != 1 or devices[0] < 0 or shards < 1 or shards > 1024:
            raise Error("byte offloaded replay: one device and logical count in [1,1024] required")
        comptime if OPT_RECORD_INTERMEDIATES:
            raise Error("byte offloaded replay: recorded optimizer intermediates unsupported")
        self.config = shape.copy()
        self.optimizer = optimizer.copy()
        self.flags = flags.copy()
        self.completed = completed
        self.logical_shards = shards
        self.p = p.copy()
        self.m = m.copy()
        self.v = v.copy()
        self.g = List[Float32](length=shape.n_total(),fill=Float32(0))
        self.context = DeviceContext(device_id=devices[0])
        self.head = BytePooledHead(self.context.value(),shape)
        self.rope = LlamaRopeTable(self.context.value(),byte_dims(shape),Float32(10000),shape.length)
        self.cache = LlamaKVCache(self.context.value(),shape.batch,byte_dims(shape),shape.length)
        self.usable = True

    def gradient(mut self, ids: List[Int32], logical: Int) raises -> Float32:
        var config = self.config.copy()
        var M = config.batch*config.length
        ref ctx = self.context.value()
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
        var inputs = List[List[Float32]]()
        var hidden = download_f32(ctx,h.x,M*config.d_model)
        for i in range(config.n_layers):
            inputs.append(hidden.copy())
            hidden = _replay_layer(ctx,self.p,config,i,inputs[i],self.rope.value(),self.cache.value())
        _load_range(ctx,h.final_hidden,hidden,0)
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
        var cotangent = download_f32(ctx,h.d_h,M*config.d_model)
        for i in range(config.n_layers-1,-1,-1):
            cotangent = _replay_backward(ctx,self.p,config,i,inputs[i],cotangent,
                self.rope.value(),self.cache.value(),self.g,logical)
        _load_range(ctx,h.x,cotangent,0)
        identical_embedding_backward_into(ctx,h.dw_emb,h.x,h.ids,h.emb_counts,h.emb_run_begin,h.emb_perm,M,emb)
        ctx.synchronize()
        _fold_host(ctx,self.g,0,h.dw_emb,logical)
        var o = config.offsets()
        _fold_host(ctx,self.g,o[len(o)-2],h.dw_lm,logical)
        return loss[0]

    def rollback(mut self) raises:
        if self.shadow_valid:
            self.p = self.shadow_p^
            self.m = self.shadow_m^
            self.v = self.shadow_v^
            self.shadow_p = List[Float32]()
            self.shadow_m = List[Float32]()
            self.shadow_v = List[Float32]()
            self.completed = self.shadow_step
        self.shadow_valid = False
        self.gradient_step = -1

    def step(mut self, shards: List[List[Int32]]) raises -> List[Float32]:
        self.require_open()
        if len(shards) != self.logical_shards or self.completed >= 999999:
            raise Error("byte offloaded replay: logical shard count or step bound")
        for i in range(len(shards)):
            byte_validate_tokens(shards[i],self.config)
        self.busy = True
        self.shadow_valid = False
        self.shadow_p = List[Float32]()
        self.shadow_m = List[Float32]()
        self.shadow_v = List[Float32]()
        self.gradient_step = -1
        var losses = List[Float32]()
        try:
            var o = self.config.offsets()
            _load_range(self.context.value(),self.head.value().emb_w,self.p,0)
            _load_range(self.context.value(),self.head.value().lm_w,self.p,o[len(o)-2])
            for i in range(len(shards)):
                losses.append(self.gradient(shards[i],i))
            # Stage all updates on the host. No canonical state is changed until
            # every chunk has completed and passed its original device scans.
            var next_p = self.p.copy()
            var next_m = self.m.copy()
            var next_v = self.v.copy()
            for i in range(self.config.n_layers+2):
                var first = 0 if i == 0 else o[1+9*(i-1)]
                var end = o[1] if i == 0 else (o[len(o)-1] if i == self.config.n_layers+1 else o[1+9*i])
                _replay_update(self.context.value(),first,end-first,self.p,self.m,self.v,self.g,
                    next_p,next_m,next_v,self.optimizer,self.completed+1)
            self.shadow_p = self.p^
            self.shadow_m = self.m^
            self.shadow_v = self.v^
            self.p = next_p^
            self.m = next_m^
            self.v = next_v^
            self.shadow_step = self.completed
            self.shadow_valid = True
            self.completed += 1
            self.gradient_step = self.completed
        except error:
            self.busy = False
            self.rollback()
            # GPU work is ephemeral. A failed context is detected by synchronize;
            # host state still exists, but this session must be reconstructed.
            try:
                self.context.value().synchronize()
            except:
                self.usable = False
            raise error
        self.busy = False
        return losses^

    def export_values(mut self, slot: Int) raises -> List[Float32]:
        self.require_open()
        if slot < 0 or slot > 3 or (slot == 3 and self.gradient_step != self.completed):
            raise Error("byte offloaded replay: invalid export slot or no committed gradient")
        if slot == 0:
            return self.p.copy()
        elif slot == 1:
            return self.m.copy()
        elif slot == 2:
            return self.v.copy()
        return self.g.copy()
