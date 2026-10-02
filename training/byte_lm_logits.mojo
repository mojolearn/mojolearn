# SPDX-License-Identifier: Apache-2.0
"""Forward-only device logits for the byte LM (DEVIATION 2658).

THE TRAINING FORWARD'S KERNELS, AT ANY SHAPE. `training/byte_lm.mojo::
_byte_forward_loss` launches `identical_embedding_forward_into`, one
`llama_decoder_layer_forward` per block (a fresh prefill from absolute
position 0) and the head `identical_gemm_into(..., OP_NT)`, then the loss.
This file launches the same three kernels on the same weights, with the RoPE
table sized from the configured length exactly as the trainer sizes it, and
nothing after the head.

The block stages and the KV cache are allocated for the call's own
`[batch, length]`, because `llama_decoder_layer_forward` refuses stages built
for any other shape. So every shape computes what the CPU reference path
(`training/byte_lm_host.mojo::byte_host_logits`) computes for that shape, with
no padding and no dependence on the training batch.

Nothing here writes the parameters, the moments, the flags or the step. A
resident session's weight views are refreshed from its flat parameters the
way the loss forward refreshes them.

Every refusal is on the host before any device work, except the one only the
device can answer: a non-finite output logit is refused, because a computed
NaN's payload is vendor-shaped (IDENTITY_PATHS row 39) and so can never be
part of an identity claim. The CPU class returns such logits; this path does
not.

A PREDICTION UNTIL GATED. `tools/byte_lm_gpu_logits_sweep.py` compares these
logits byte for byte with the CPU reference path on each GPU.
"""
from std.memory import bitcast
from std.sys.compile import is_defined
from max.gpu.host import DeviceBuffer, DeviceContext
from core.identity_trace import IdentityTrace
from training.byte_lm_config import ByteConfig
from training.byte_lm import (
    ByteTrainer,
    _bind_emb_head,
    _block_weights,
    _byte_check_gemm,
    _byte_recover,
    _require_profile,
    _unpack_block,
    byte_dims,
)
from training.checks.train_loop import _copy_into, _upload, _zeros, _zeros_i32, download_f32, download_f32_into, download_f32_into_scanned
from core.device_arena import arena_begin, arena_end, arena_release
from gemm.checks.gemm_identical import identical_gemm_into, identical_gemm_workspace_max_floats
from gemm.checks.gemm_oracle import OP_NT
from embedding.checks.embedding_identical import identical_embedding_forward_into
from embedding.checks.embedding_oracle import EmbConfig
from std.time import perf_counter_ns
from transformer.impl.llama.modeling_llama import (
    timing_on,
    timing_tick,
    LlamaDeviceStages,
    LlamaDeviceWeights,
    LlamaKVCache,
    LlamaRopeTable,
    llama_decoder_layer_forward,
    residual_next_norm_fusion_enabled,
)


comptime BYTE_LOGITS_MAX_BATCH = 1024
comptime BYTE_LOGITS_MAX_CELLS = 268435456


def byte_logits_validate(inputs: List[Int32], batch: Int, length: Int, config: ByteConfig) raises:
    """Host-only admission of one logits call: profile, shape bounds, span,
    ids, and the GEMM workspaces the call's shape reaches."""
    _require_profile()
    config.validate()
    if batch < 1 or batch > BYTE_LOGITS_MAX_BATCH:
        raise Error("byte LM logits: batch must be in [1, " + String(BYTE_LOGITS_MAX_BATCH) + "]")
    if length < 1 or length > config.length:
        raise Error("byte LM logits: length must be in [1, " + String(config.length) + "]")
    var m = batch * length
    if m > BYTE_LOGITS_MAX_CELLS // config.vocab_size:
        raise Error("byte LM logits: batch * length * vocab exceeds the admitted span")
    if len(inputs) != m:
        raise Error("byte LM logits: ids must hold batch * length tokens")
    for t in range(m):
        var v = Int(inputs[t])
        if v < 0 or v >= config.vocab_size:
            raise Error("byte LM logits: token id outside [0, vocab) at " + String(t))
    var widths: List[Int] = [config.d_model, config.n_kv * config.head_dim,
                             config.intermediate, config.vocab_size]
    for width in widths:
        _byte_check_gemm(m, width, config.d_model)
    _byte_check_gemm(m, config.d_model, config.intermediate)
    _byte_check_gemm(length, length, config.head_dim)
    _byte_check_gemm(length, config.head_dim, length)


def byte_logits_validate_params(params: List[Float32], config: ByteConfig) raises:
    """The stateless call's parameters: the configured count, every value
    finite (tested by bits)."""
    if len(params) != config.n_total():
        raise Error("byte LM logits: expected " + String(config.n_total()) + " parameters, got "
                    + String(len(params)))
    for i in range(len(params)):
        if (bitcast[DType.uint32](params[i]) & UInt32(0x7F800000)) == UInt32(0x7F800000):
            raise Error("byte LM logits: non-finite parameter at flat index " + String(i))


struct ByteLogitsScratch(Movable):
    """The call-shaped device buffers of one logits forward (DEVIATION 2942,
    lane/infer-speed-neural, 2026-09-17): the ids, the embedding output, one
    KV cache reset per block, one stage struct per block, the logits and the
    head GEMM workspace, all sized for `[batch, length]`. A resident session
    keeps one of these across calls and rebuilds it when the shape changes;
    the stateless call builds one per call. Reusing stages across calls is
    `ByteTrainer.forward`'s own pattern (its per-layer stages serve every
    training step), and every buffer here is written whole before it is
    read: the ids by the upload, `x` by the embedding, the cache from
    `s = 0`, the stages by the block, the logits by the head GEMM, whose
    workspace is scratch the GEMM writes before reading, as the trainer's
    own retained workspaces are. Nothing about what is computed changes;
    only when the buffers are allocated."""
    var batch: Int
    var length: Int
    var ids: DeviceBuffer[DType.int32]
    var x: DeviceBuffer[DType.float32]
    var cache: LlamaKVCache
    var stages: List[LlamaDeviceStages]
    var logits: DeviceBuffer[DType.float32]
    var head_ws: DeviceBuffer[DType.float32]
    var arena_id: Int

    def __init__(out self, ctx: DeviceContext, batch: Int, length: Int, config: ByteConfig) raises:
        var m = batch * length
        var dims = byte_dims(config)
        self.batch = batch
        self.length = length
        # lane/neural-pass43: the scratch's buffers carved from arena chunks
        self.arena_id = arena_begin()
        self.ids = _zeros_i32(ctx, m)
        self.x = _zeros(ctx, m * config.d_model)
        self.cache = LlamaKVCache(ctx, batch, dims, length)
        self.stages = List[LlamaDeviceStages]()
        for _ in range(config.n_layers):
            self.stages.append(LlamaDeviceStages(ctx, batch, length, length, dims, lean=True))
        self.logits = _zeros(ctx, m * config.vocab_size)
        self.head_ws = _zeros(ctx, identical_gemm_workspace_max_floats(m, config.vocab_size, config.d_model))
        arena_end(self.arena_id)

    def __deinit__(deinit self):
        try:
            arena_release(self.arena_id)
        except:
            pass

    def fits(self, batch: Int, length: Int) -> Bool:
        return self.batch == batch and self.length == length


def _logits_enqueue(
    ctx: DeviceContext,
    mut weights: List[LlamaDeviceWeights],
    mut emb_w: DeviceBuffer[DType.float32],
    mut lm_w: DeviceBuffer[DType.float32],
    mut rope: LlamaRopeTable,
    mut sc: ByteLogitsScratch,
    inputs: List[Int32],
    batch: Int,
    length: Int,
    config: ByteConfig,
) raises:
    """Embedding, the blocks and the head, as `_byte_forward_loss` launches
    them, into the scratch's call-shaped buffers; the logits are left in
    `sc.logits` `[batch * length, vocab]`, row-major, NOT downloaded (the
    two callers below download them differently). No completion wait here.

    ONE completion wait, the download's (DEVIATION 2942). The ids copy, the
    embedding, every block and the head GEMM are enqueued on the same
    in-order context, so the waits that used to sit between them ordered
    nothing; `inputs` is borrowed for the whole call, past that wait. The
    kernels, their order and their operands are exactly the previous ones."""
    var m = batch * length
    var dm = config.d_model
    var vocab = config.vocab_size
    if not sc.fits(batch, length):
        raise Error("byte LM logits: the scratch was built for another shape")
    if len(inputs) != m:
        raise Error("byte LM logits: ids must hold batch * length tokens")

    var ton = timing_on()
    var tk = Int(perf_counter_ns())
    ctx.enqueue_copy(dst_buf=sc.ids, src_ptr=inputs.unsafe_ptr())
    identical_embedding_forward_into(ctx, sc.x, emb_w, sc.ids, m, EmbConfig.llama(vocab, dm))
    timing_tick(ctx, ton, tk, "logits.ids_and_embedding")

    # One cache for every block, reset to a fresh prefill before each, as the
    # trainer's `prefill_cache` is; stages sized for this call's shape.
    var trace = IdentityTrace.disabled()
    for layer in range(config.n_layers):
        var st = sc.stages.pop(layer)
        # After the pop, the NEXT block's stages are at index `layer`, not `layer + 1`.
        sc.cache.s = 0
        var prefix = String("byte.logits.block") + String(layer) + ".forward"
        var norm1_ready = layer > 0 and residual_next_norm_fusion_enabled(
            m, weights[layer].opts.norm_kind, weights[layer].opts.norm_bias
        )
        var fuse_next = layer + 1 < config.n_layers and residual_next_norm_fusion_enabled(
            m, weights[layer + 1].opts.norm_kind,
            weights[layer + 1].opts.norm_bias,
        )
        if layer == 0:
            if fuse_next:
                llama_decoder_layer_forward(ctx, st, sc.cache, rope, weights[layer], sc.x,
                    batch, length, 0, trace, prefix, norm1_ready=norm1_ready,
                    next_norm_sumsq=Optional(sc.stages[layer].norm1_sumsq.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]()),
                    next_norm_out=Optional(sc.stages[layer].norm1_out.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]()),
                    next_norm_weight=Optional(weights[layer + 1].norm1_w.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]()),
                    next_norm_eps=Optional(weights[layer + 1].eps), forward_only=True)
            else:
                llama_decoder_layer_forward(ctx, st, sc.cache, rope, weights[layer], sc.x,
                    batch, length, 0, trace, prefix, norm1_ready=norm1_ready, forward_only=True)
        else:
            if fuse_next:
                llama_decoder_layer_forward(ctx, st, sc.cache, rope, weights[layer], sc.stages[layer - 1].residual2,
                    batch, length, 0, trace, prefix, norm1_ready=norm1_ready,
                    next_norm_sumsq=Optional(sc.stages[layer].norm1_sumsq.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]()),
                    next_norm_out=Optional(sc.stages[layer].norm1_out.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]()),
                    next_norm_weight=Optional(weights[layer + 1].norm1_w.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]()),
                    next_norm_eps=Optional(weights[layer + 1].eps), forward_only=True)
            else:
                llama_decoder_layer_forward(ctx, st, sc.cache, rope, weights[layer], sc.stages[layer - 1].residual2,
                    batch, length, 0, trace, prefix, norm1_ready=norm1_ready, forward_only=True)
        sc.stages.insert(layer, st^)
    timing_tick(ctx, ton, tk, "logits.layers")
    identical_gemm_into(ctx, sc.logits, sc.stages[config.n_layers - 1].residual2, lm_w, sc.head_ws,
        m, vocab, dm, OP_NT)
    timing_tick(ctx, ton, tk, "logits.head_gemm")
    _ = trace


def _logits_forward(
    ctx: DeviceContext,
    mut weights: List[LlamaDeviceWeights],
    mut emb_w: DeviceBuffer[DType.float32],
    mut lm_w: DeviceBuffer[DType.float32],
    mut rope: LlamaRopeTable,
    mut sc: ByteLogitsScratch,
    inputs: List[Int32],
    batch: Int,
    length: Int,
    config: ByteConfig,
) raises -> List[Float32]:
    """`_logits_enqueue`, then the logits downloaded into a List (the
    original return shape, kept for callers that want an owned List). The
    bindings take `_logits_forward_into` instead, which skips this path's
    pinned host buffer, element loop and second copy."""
    var m = batch * length
    var vocab = config.vocab_size
    _logits_enqueue(ctx, weights, emb_w, lm_w, rope, sc, inputs, batch, length, config)
    var out = download_f32(ctx, sc.logits, m * vocab)
    comptime if is_defined["MOJOLEARN_BYTE_LM_LOGITS_SABOTAGE"]():
        # The negative control for this lane's sweeps: one output bit moved
        # after the arithmetic, so a passing sweep on this build would be a
        # sweep that compares nothing. Never in a shipped binary.
        out[0] = bitcast[DType.float32](bitcast[DType.uint32](out[0]) ^ UInt32(1))
    for i in range(len(out)):
        if (bitcast[DType.uint32](out[i]) & UInt32(0x7F800000)) == UInt32(0x7F800000):
            raise Error("byte LM logits: non-finite logit at flat index " + String(i)
                        + " REFUSED (NaN payloads are vendor-shaped, IDENTITY_PATHS row 39)")
    return out^


def _refuse_nonfinite_logits(destination: MutPointer[Float32, MutUntrackedOrigin], n: Int) raises:
    """The same refusal `_logits_forward` makes over its List, over the
    caller's memory, eight lanes at a time: the exponent-all-ones test on
    the bits (an infinity or a NaN), and the FIRST offending flat index in
    the message, found by the scalar tail once a vector has one. No
    floating-point arithmetic, so no bit is judged differently."""
    comptime W = 8
    var exp = SIMD[DType.uint32, W](0x7F800000)
    var i = 0
    var body = n - n % W
    while i < body:
        var bits = bitcast[DType.uint32, W](destination.unsafe_load[width=W](i)) & exp
        var hit = bits.eq(exp).select(SIMD[DType.uint32, W](1), SIMD[DType.uint32, W](0))
        if hit.reduce_or() != UInt32(0):
            break
        i += W
    while i < n:
        if (bitcast[DType.uint32](destination.unsafe_load(i)) & UInt32(0x7F800000)) == UInt32(0x7F800000):
            _raise_nonfinite_logit(i)
        i += 1


def _raise_nonfinite_logit(i: Int) raises:
    raise Error("byte LM logits: non-finite logit at flat index " + String(i)
                + " REFUSED (NaN payloads are vendor-shaped, IDENTITY_PATHS row 39)")


def _logits_forward_into(
    ctx: DeviceContext,
    mut weights: List[LlamaDeviceWeights],
    mut emb_w: DeviceBuffer[DType.float32],
    mut lm_w: DeviceBuffer[DType.float32],
    mut rope: LlamaRopeTable,
    mut sc: ByteLogitsScratch,
    inputs: List[Int32],
    batch: Int,
    length: Int,
    config: ByteConfig,
    destination: MutPointer[Float32, MutUntrackedOrigin],
) raises:
    """`_logits_enqueue`, then ONE device-to-host copy straight into `destination`
    (`batch * length * vocab` floats the caller owns and keeps alive for
    the whole call), then the non-finite refusal over that memory.

    lane/neural-net-experiment (2026-09-30): the bindings used
    `_logits_forward` and then copied its List into the caller's array. At
    the board's LM shape (L2048, V8192: 64 MiB of logits) that was a 64 MiB
    PINNED host allocation per call, a device-to-host copy into it, a
    16.8M-element append loop into a List, the scan, a SIMD copy into the
    caller's array and the pinned free -- all host work after the GPU was
    done, and the reason `lm-forward` cost three `lm-train-step`s on the
    L40S (147 ms against 46 ms; the training step returns 4 bytes). This is
    DEVIATION 3120's fix (`download_f32_into`) applied to the logits: the
    same bytes land in the same places, one transfer, no List.

    The sabotage define moves the same bit it moved before (flat index 0,
    after the copy, before the scan)."""
    var m = batch * length
    var vocab = config.vocab_size
    _logits_enqueue(ctx, weights, emb_w, lm_w, rope, sc, inputs, batch, length, config)
    var ton = timing_on()
    var tk = Int(perf_counter_ns())
    var bad = download_f32_into_scanned(ctx, sc.logits, m * vocab, destination, True)
    timing_tick(ctx, ton, tk, "logits.download")
    comptime if is_defined["MOJOLEARN_BYTE_LM_LOGITS_SABOTAGE"]():
        destination.unsafe_store(0, bitcast[DType.float32](bitcast[DType.uint32](destination.unsafe_load(0)) ^ UInt32(1)))
    if bad >= 0:
        _raise_nonfinite_logit(bad)
    timing_tick(ctx, ton, tk, "logits.scan")


def byte_logits_from_params(ctx: DeviceContext, params: List[Float32], inputs: List[Int32],
                            batch: Int, length: Int, config: ByteConfig) raises -> List[Float32]:
    """The stateless call: upload the weights, embedding and head from host
    parameters in registry order (`_block_weights` and the loss forward's
    embedding and head slices), then `_logits_forward`."""
    byte_logits_validate(inputs, batch, length, config)
    byte_logits_validate_params(params, config)
    var offsets = config.offsets()
    var vd = config.vocab_size * config.d_model
    var weights = List[LlamaDeviceWeights]()
    for layer in range(config.n_layers):
        weights.append(_block_weights(ctx, params, layer, config))
    var emb_host = List[Float32](capacity=vd)
    var head_host = List[Float32](capacity=vd)
    var head_base = offsets[config.n_tensors() - 1]
    for i in range(vd):
        emb_host.append(params[i])
        head_host.append(params[head_base + i])
    var emb_w = _upload(ctx, emb_host)
    var lm_w = _upload(ctx, head_host)
    var rope = LlamaRopeTable(ctx, byte_dims(config), Float32(10000), config.length)
    var sc = ByteLogitsScratch(ctx, batch, length, config)
    var out = _logits_forward(ctx, weights, emb_w, lm_w, rope, sc, inputs, batch, length, config)
    ctx.synchronize()
    _ = sc^
    return out^


def byte_logits_from_params_into(ctx: DeviceContext, params: List[Float32], inputs: List[Int32],
                                 batch: Int, length: Int, config: ByteConfig,
                                 destination: MutPointer[Float32, MutUntrackedOrigin]) raises:
    """`byte_logits_from_params` writing straight into the caller's memory
    (`_logits_forward_into`): the same uploads, kernels and order."""
    byte_logits_validate(inputs, batch, length, config)
    byte_logits_validate_params(params, config)
    var offsets = config.offsets()
    var vd = config.vocab_size * config.d_model
    var weights = List[LlamaDeviceWeights]()
    for layer in range(config.n_layers):
        weights.append(_block_weights(ctx, params, layer, config))
    var emb_host = List[Float32](capacity=vd)
    var head_host = List[Float32](capacity=vd)
    var head_base = offsets[config.n_tensors() - 1]
    for i in range(vd):
        emb_host.append(params[i])
        head_host.append(params[head_base + i])
    var emb_w = _upload(ctx, emb_host)
    var lm_w = _upload(ctx, head_host)
    var rope = LlamaRopeTable(ctx, byte_dims(config), Float32(10000), config.length)
    var sc = ByteLogitsScratch(ctx, batch, length, config)
    _logits_forward_into(ctx, weights, emb_w, lm_w, rope, sc, inputs, batch, length, config, destination)
    ctx.synchronize()
    _ = sc^


def byte_logits_resident(ctx: DeviceContext, mut tr: ByteTrainer, inputs: List[Int32],
                         batch: Int, length: Int,
                         mut scratch: Optional[ByteLogitsScratch]) raises -> List[Float32]:
    """The resident call: refresh the trainer's weight views, embedding and
    head from its flat device parameters exactly as `_byte_forward_loss`
    does, then `_logits_forward` on `scratch`, which the session keeps
    across calls and which is rebuilt here when the call's shape differs
    (DEVIATION 2942). The failure convention is `byte_eval_loss_resident`'s:
    nothing was written, so a failure re-scans the state once
    (`_byte_recover`) before the trainer is healthy again."""
    var config = tr.config.copy()
    byte_logits_validate(inputs, batch, length, config)
    if not tr.healthy:
        raise Error("byte LM: session lost; restore a retained export")
    var rebuild = True
    if scratch:
        rebuild = not scratch.value().fits(batch, length)
    if rebuild:
        scratch = None
        scratch = ByteLogitsScratch(ctx, batch, length, config)
    tr.healthy = False
    tr.shadow_valid = False
    var out = List[Float32]()
    var failed = False
    var message = String("")
    try:
        for layer in range(config.n_layers):
            _unpack_block(ctx, tr.buffers, tr.weights[layer], layer)
        _bind_emb_head(ctx, tr.buffers, config)
        out = _logits_forward(ctx, tr.weights, tr.buffers.emb_w, tr.buffers.lm_w, tr.rope,
            scratch.value(), inputs, batch, length, config)
        ctx.synchronize()
    except error:
        failed = True
        message = String(error)
    if failed:
        _byte_recover(ctx, tr, message)
    tr.healthy = True
    return out^


def byte_logits_resident_into(ctx: DeviceContext, mut tr: ByteTrainer, inputs: List[Int32],
                              batch: Int, length: Int,
                              mut scratch: Optional[ByteLogitsScratch],
                              destination: MutPointer[Float32, MutUntrackedOrigin]) raises:
    """`byte_logits_resident` writing straight into the caller's memory
    (`_logits_forward_into`); the same refresh, scratch rule and failure
    convention. On a raise the caller's memory holds an unspecified prefix
    of the copy, as `download_f32_into` says; the session binding does not
    publish a failed call."""
    var config = tr.config.copy()
    byte_logits_validate(inputs, batch, length, config)
    if not tr.healthy:
        raise Error("byte LM: session lost; restore a retained export")
    var rebuild = True
    if scratch:
        rebuild = not scratch.value().fits(batch, length)
    if rebuild:
        scratch = None
        scratch = ByteLogitsScratch(ctx, batch, length, config)
    tr.healthy = False
    tr.shadow_valid = False
    var failed = False
    var message = String("")
    try:
        for layer in range(config.n_layers):
            _unpack_block(ctx, tr.buffers, tr.weights[layer], layer)
        _bind_emb_head(ctx, tr.buffers, config)
        _logits_forward_into(ctx, tr.weights, tr.buffers.emb_w, tr.buffers.lm_w, tr.rope,
            scratch.value(), inputs, batch, length, config, destination)
        ctx.synchronize()
    except error:
        failed = True
        message = String(error)
    if failed:
        _byte_recover(ctx, tr, message)
    tr.healthy = True
