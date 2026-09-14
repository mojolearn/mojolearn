# SPDX-License-Identifier: Apache-2.0
"""Layer-owned byte-LM forward/backward storage and ordered gradient sums.

Internal model-pooling component. Each decoder layer exists on one device;
only activations/cotangents cross ownership boundaries. The caller owns the
embedding/head and the eventual whole-model optimizer transaction. This is
not a complete training driver or a throughput claim.
"""
from max.gpu.host import DeviceContext, DeviceBuffer
from core.identity_trace import IdentityTrace
from core.device_scan import DeviceScanScratch
from training.byte_lm import (
    byte_dims, _block_weights, _require_profile, _byte_validate_allocations,
    _require_finite, _require_device_finite,
)
from training.byte_lm_config import ByteConfig
from training.byte_lm_parallel import _ordered_add_kernel
from training.checks.train_loop import _zeros, _copy_into, download_f32
from transformer.impl.llama.modeling_llama import (
    LlamaDeviceWeights, LlamaDeviceStages, LlamaRopeTable, LlamaKVCache,
    llama_decoder_layer_forward,
)
from transformer.checks.transformer_backward import (
    LlamaBackwardStages, llama_decoder_layer_backward_device,
)


struct ByteOwnedLayer(Movable):
    var owner: Int
    var first: Int
    var offsets: List[Int]
    var weights: LlamaDeviceWeights
    var forward: LlamaDeviceStages
    var backward: LlamaBackwardStages
    var input: DeviceBuffer[DType.float32]
    var cotangent: DeviceBuffer[DType.float32]
    var gradient: DeviceBuffer[DType.float32]
    var total: DeviceBuffer[DType.float32]
    var scan: DeviceScanScratch

    def __init__(out self, ctx: DeviceContext, owner: Int, layer: Int,
                 p: List[Float32], shape: ByteConfig) raises:
        self.owner = owner
        var o = shape.offsets()
        var base = 1 + 9 * layer
        self.first = o[base]
        self.offsets = List[Int]()
        for j in range(10):
            self.offsets.append(o[base+j] - self.first)
        self.weights = _block_weights(ctx, p, layer, shape)
        self.forward = LlamaDeviceStages(ctx, shape.batch, shape.length,
            shape.length, byte_dims(shape), lean=True)
        self.backward = LlamaBackwardStages(ctx, shape.batch, shape.length,
            shape.length, byte_dims(shape), lean=True)
        self.input = _zeros(ctx, shape.batch*shape.length*shape.d_model)
        self.cotangent = _zeros(ctx, shape.batch*shape.length*shape.d_model)
        self.gradient = _zeros(ctx, self.offsets[9])
        self.total = _zeros(ctx, self.offsets[9])
        self.scan = DeviceScanScratch(ctx)
        ctx.synchronize()

    def pack(mut self, ctx: DeviceContext) raises:
        var o = self.offsets.copy()
        _copy_into(ctx, self.gradient, self.backward.dw_norm1, o[0], 0, o[1]-o[0])
        _copy_into(ctx, self.gradient, self.backward.dw_q, o[1], 0, o[2]-o[1])
        _copy_into(ctx, self.gradient, self.backward.dw_k, o[2], 0, o[3]-o[2])
        _copy_into(ctx, self.gradient, self.backward.dw_v, o[3], 0, o[4]-o[3])
        _copy_into(ctx, self.gradient, self.backward.dw_o, o[4], 0, o[5]-o[4])
        _copy_into(ctx, self.gradient, self.backward.dw_norm2, o[5], 0, o[6]-o[5])
        _copy_into(ctx, self.gradient, self.backward.dw_gate, o[6], 0, o[7]-o[6])
        _copy_into(ctx, self.gradient, self.backward.dw_up, o[7], 0, o[8]-o[7])
        _copy_into(ctx, self.gradient, self.backward.dw_down, o[8], 0, o[9]-o[8])
        ctx.synchronize()
        _require_device_finite(ctx, self.scan, self.gradient, o[9], "layer gradients")


struct ByteLayerPool(Movable):
    var contexts: List[DeviceContext]
    var layers: List[ByteOwnedLayer]
    var ropes: List[LlamaRopeTable]
    var caches: List[LlamaKVCache]
    var config: ByteConfig
    var forward_ready: Bool
    var gradients_ready: Bool
    var accumulated: Int
    var usable: Bool

    def __init__(out self):
        self.contexts = List[DeviceContext]()
        self.layers = List[ByteOwnedLayer]()
        self.ropes = List[LlamaRopeTable]()
        self.caches = List[LlamaKVCache]()
        self.config = ByteConfig()
        self.forward_ready = False
        self.gradients_ready = False
        self.accumulated = 0
        self.usable = False

    def __deinit__(deinit self):
        _ = self.layers^
        _ = self.ropes^
        _ = self.caches^
        for i in range(len(self.contexts)):
            try:
                self.contexts[i].synchronize()
            except:
                pass
        _ = self.contexts^

    def open(mut self, devices: List[Int], p: List[Float32], shape: ByteConfig) raises:
        if len(self.contexts) != 0:
            raise Error("byte layer pool: already open")
        _require_profile()
        _byte_validate_allocations(shape)
        if len(p) != shape.n_total():
            raise Error("byte layer pool: canonical parameter length required")
        _require_finite(p, "parameters")
        if len(devices) < 1 or len(devices) > min(64,shape.n_layers):
            raise Error("byte layer pool: require 1 <= devices <= min(64,layers)")
        for i in range(len(devices)):
            if devices[i] < 0:
                raise Error("byte layer pool: negative device index")
            for j in range(i):
                if devices[i] == devices[j]:
                    raise Error("byte layer pool: duplicate device index")
        self.config = shape.copy()
        for i in range(len(devices)):
            self.contexts.append(DeviceContext(device_id=devices[i]))
            self.ropes.append(LlamaRopeTable(self.contexts[i], byte_dims(shape), Float32(10000), shape.length))
            self.caches.append(LlamaKVCache(self.contexts[i], shape.batch, byte_dims(shape), shape.length))
        for layer in range(shape.n_layers):
            var owner = layer * len(devices) // shape.n_layers
            self.layers.append(ByteOwnedLayer(self.contexts[owner], owner, layer, p, shape))
        self.usable = True

    def require_open(self) raises:
        if not self.usable:
            raise Error("byte layer pool: closed or failed; reconstruct from canonical state")

    def forward_into(mut self, source_ctx: DeviceContext,
                     mut input: DeviceBuffer[DType.float32],
                     mut output: DeviceBuffer[DType.float32]) raises:
        self.require_open()
        var n = self.config.batch*self.config.length*self.config.d_model
        if len(input) != n or len(output) != n:
            raise Error("byte layer pool: activation length mismatch")
        self.forward_ready = False
        self.gradients_ready = False
        try:
            source_ctx.synchronize()
            input.enqueue_copy_to(self.layers[0].input)
            source_ctx.synchronize()
            var trace = IdentityTrace.disabled()
            for i in range(len(self.layers)):
                var layer = self.layers.pop(i)
                var owner = layer.owner
                self.contexts[owner].synchronize()
                self.caches[owner].s = 0
                llama_decoder_layer_forward(self.contexts[owner], layer.forward,
                    self.caches[owner], self.ropes[owner], layer.weights, layer.input,
                    self.config.batch, self.config.length, 0, trace,
                    String("byte.block")+String(i)+".forward")
                self.contexts[owner].synchronize()
                if i < len(self.layers):
                    layer.forward.residual2.enqueue_copy_to(self.layers[i].input)
                else:
                    layer.forward.residual2.enqueue_copy_to(output)
                self.contexts[owner].synchronize()
                self.layers.insert(i, layer^)
            source_ctx.synchronize()
            self.forward_ready = True
        except error:
            self.usable = False
            raise error

    def backward_into(mut self, source_ctx: DeviceContext,
                      mut cotangent: DeviceBuffer[DType.float32],
                      mut d_input: DeviceBuffer[DType.float32]) raises:
        self.require_open()
        var n = self.config.batch*self.config.length*self.config.d_model
        if not self.forward_ready:
            raise Error("byte layer pool: backward requires a fresh forward")
        if len(cotangent) != n or len(d_input) != n:
            raise Error("byte layer pool: cotangent length mismatch")
        self.forward_ready = False
        try:
            source_ctx.synchronize()
            cotangent.enqueue_copy_to(self.layers[len(self.layers)-1].cotangent)
            source_ctx.synchronize()
            var trace = IdentityTrace.disabled()
            for i in range(len(self.layers)-1,-1,-1):
                var layer = self.layers.pop(i)
                var owner = layer.owner
                self.contexts[owner].synchronize()
                llama_decoder_layer_backward_device(self.contexts[owner], layer.backward,
                    layer.forward, layer.weights, self.ropes[owner].cos, self.ropes[owner].sin,
                    layer.input, layer.cotangent, self.config.batch, self.config.length, 0,
                    trace, String("byte.block")+String(i)+".backward")
                layer.pack(self.contexts[owner])
                if i > 0:
                    layer.backward.d_x.enqueue_copy_to(self.layers[i-1].cotangent)
                else:
                    layer.backward.d_x.enqueue_copy_to(d_input)
                self.contexts[owner].synchronize()
                self.layers.insert(i, layer^)
            source_ctx.synchronize()
            self.gradients_ready = True
        except error:
            self.usable = False
            raise error

    def accumulate(mut self, logical_index: Int) raises:
        self.require_open()
        if not self.gradients_ready or logical_index < 0 or logical_index >= 1024:
            raise Error("byte layer pool: fresh gradients and legal logical index required")
        if logical_index != 0 and logical_index != self.accumulated:
            raise Error("byte layer pool: logical gradients must be added in order")
        try:
            for i in range(len(self.layers)):
                var layer = self.layers.pop(i)
                var owner = layer.owner
                var n = len(layer.gradient)
                if logical_index == 0:
                    _copy_into(self.contexts[owner], layer.total, layer.gradient, 0, 0, n)
                else:
                    self.contexts[owner].enqueue_function[_ordered_add_kernel](
                        layer.total, layer.gradient, Int32(n),
                        grid_dim=(n+255)//256, block_dim=256)
                self.contexts[owner].synchronize()
                _require_device_finite(self.contexts[owner], layer.scan, layer.total, n, "layer gradient sum")
                self.layers.insert(i, layer^)
            self.accumulated = logical_index+1
            self.gradients_ready = False
        except error:
            self.usable = False
            raise error

    def export_gradients(mut self) raises -> List[Float32]:
        self.require_open()
        if self.accumulated == 0:
            raise Error("byte layer pool: no accumulated gradients")
        var result = List[Float32]()
        for i in range(len(self.layers)):
            var owner = self.layers[i].owner
            var values = download_f32(self.contexts[owner], self.layers[i].total, len(self.layers[i].total))
            for value in values:
                result.append(value)
        return result^
