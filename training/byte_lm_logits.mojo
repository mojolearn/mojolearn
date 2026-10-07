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
from std.memory import bitcast, stack_allocation
from std.gpu import block_idx, thread_idx
from max.gpu.memory import AddressSpace
from max.gpu.sync import barrier
from std.sys.compile import is_defined
from max.gpu.host import DeviceBuffer, DeviceContext
from core.identity_trace import IdentityTrace
from training.byte_lm_config import ByteConfig
from training.byte_lm import (
    ByteTrainer,
    _bind_emb_head,
    _block_weights,
    _block_weights_dev,
    _param_dev_copy,
    _byte_check_gemm,
    _byte_recover,
    _require_profile,
    _unpack_block,
    byte_dims,
)
from training.checks.train_loop import _copy_into, _upload, _zeros, _zeros_i32, download_f32, download_f32_into, download_f32_into_scanned
from core.device_arena import arena_begin, arena_end, arena_release
from core.device_scan import device_first_nonfinite, device_first_token_oob
from gemm.neural_dispatch import identical_gemm_into, identical_gemm_workspace_max_floats
from gemm.experiments.neural_switches import ROLE_HEAD
from gemm.experiments.neural_streaming import NN16
from gemm.contract import OP_NT
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
    # cpu3-seq: the token-range refusal moved to the device
    # (`_logits_enqueue`, `device_first_token_oob` on the uploaded ids, same
    # message and first index), before the embedding gather reads them.
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
    # cpu3-seq: the finite refusal runs on the device after the one upload
    # (`_logits_params_upload`: same first flat index, same message).


def _logits_params_upload(ctx: DeviceContext, params: List[Float32], config: ByteConfig) raises -> DeviceBuffer[DType.float32]:
    """cpu3-seq: the flat parameters in ONE upload, then the non-finite
    refusal as a device scan (one partials readback and the wait that also
    completes the upload, so `params` may be released after). Every slice
    the stateless logits call needs is then copied device to device."""
    var n = config.n_total()
    if len(params) != n:
        raise Error("byte LM logits: expected " + String(n) + " parameters, got "
                    + String(len(params)))
    var flat = ctx.enqueue_create_buffer[DType.float32](n)
    ctx.enqueue_copy(dst_buf=flat, src_ptr=params.unsafe_ptr())
    var bad = device_first_nonfinite(ctx, flat, n)
    if bad >= 0:
        raise Error("byte LM logits: non-finite parameter at flat index " + String(bad))
    return flat^


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
        var head_cells = identical_gemm_workspace_max_floats(m, config.vocab_size, config.d_model)
        comptime if NN16 and not is_defined["MOJOLEARN_IDN_NEURAL_GEMM_CONTROL"]():
            # Every consumed GEMM scratch cell has an in-order producer. This
            # removes the real cold owner clear; geometry/allocation bounds
            # remain those of the selected dispatcher, on every shape/vendor.
            self.head_ws = ctx.enqueue_create_buffer[DType.float32](head_cells)
        else:
            self.head_ws = _zeros(ctx, head_cells)
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
    # cpu3-seq: the token-range refusal on the uploaded ids (one scan, one
    # partials readback), before the embedding gather reads them.
    var bad_id = device_first_token_oob(ctx, sc.ids, m, vocab)
    if bad_id >= 0:
        raise Error("byte LM logits: token id outside [0, vocab) at " + String(bad_id))
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
            m, dm, weights[layer].opts.norm_kind, weights[layer].opts.norm_bias
        )
        var fuse_next = layer + 1 < config.n_layers and residual_next_norm_fusion_enabled(
            m, dm, weights[layer + 1].opts.norm_kind,
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
    identical_gemm_into[ROLE=ROLE_HEAD](ctx, sc.logits, sc.stages[config.n_layers - 1].residual2, lm_w, sc.head_ws,
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
    # cpu3-seq: the non-finite refusal is the device scan `_logits_forward_into`
    # runs (same first flat index, same message), before the download; no
    # host walk over the m * vocab logits.
    var bad = device_first_nonfinite(ctx, sc.logits, m * vocab)
    if bad >= 0:
        _raise_nonfinite_logit(bad)
    var out = download_f32(ctx, sc.logits, m * vocab)
    comptime if is_defined["MOJOLEARN_BYTE_LM_LOGITS_SABOTAGE"]():
        # The negative control for this lane's sweeps: one output bit moved
        # after the arithmetic, so a passing sweep on this build would be a
        # sweep that compares nothing. Never in a shipped binary.
        out[0] = bitcast[DType.float32](bitcast[DType.uint32](out[0]) ^ UInt32(1))
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
    """`_logits_enqueue`, then the device non-finite refusal, then ONE
    device-to-host copy straight into `destination` (`batch * length * vocab`
    floats the caller owns and keeps alive for the whole call).

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
    after the copy; since cpu2-l11-neural the scan runs before the copy)."""
    var m = batch * length
    var vocab = config.vocab_size
    _logits_enqueue(ctx, weights, emb_w, lm_w, rope, sc, inputs, batch, length, config)
    var ton = timing_on()
    var tk = Int(perf_counter_ns())
    # cpu2-l11-neural (2026-10-04): the non-finite refusal is a DEVICE scan
    # (`device_first_nonfinite`: one launch, one partials copy, the same
    # first flat index the host walk returned), run before the download, so
    # no host loop reads the m * vocab logits. The download is unscanned.
    var bad = device_first_nonfinite(ctx, sc.logits, m * vocab)
    timing_tick(ctx, ton, tk, "logits.scan")
    if bad >= 0:
        _raise_nonfinite_logit(bad)
    _ = download_f32_into_scanned(ctx, sc.logits, m * vocab, destination, False)
    timing_tick(ctx, ton, tk, "logits.download")
    comptime if is_defined["MOJOLEARN_BYTE_LM_LOGITS_SABOTAGE"]():
        destination.unsafe_store(0, bitcast[DType.float32](bitcast[DType.uint32](destination.unsafe_load(0)) ^ UInt32(1)))


def byte_logits_from_params(ctx: DeviceContext, params: List[Float32], inputs: List[Int32],
                            batch: Int, length: Int, config: ByteConfig) raises -> List[Float32]:
    """The stateless call: upload the weights, embedding and head from host
    parameters in registry order (`_block_weights` and the loss forward's
    embedding and head slices), then `_logits_forward`."""
    byte_logits_validate(inputs, batch, length, config)
    byte_logits_validate_params(params, config)
    var offsets = config.offsets()
    var vd = config.vocab_size * config.d_model
    # cpu3-seq: one upload of the flat parameters, the finite refusal on the
    # device, and every block/embedding/head slice copied device to device
    # (no host slice loops; same values in the same buffers as before).
    var flat = _logits_params_upload(ctx, params, config)
    var weights = List[LlamaDeviceWeights]()
    for layer in range(config.n_layers):
        weights.append(_block_weights_dev(ctx, flat, layer, config))
    var head_base = offsets[config.n_tensors() - 1]
    var emb_w = _param_dev_copy(ctx, flat, 0, vd)
    var lm_w = _param_dev_copy(ctx, flat, head_base, vd)
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
    # cpu3-seq: one upload of the flat parameters, the finite refusal on the
    # device, and every block/embedding/head slice copied device to device
    # (no host slice loops; same values in the same buffers as before).
    var flat = _logits_params_upload(ctx, params, config)
    var weights = List[LlamaDeviceWeights]()
    for layer in range(config.n_layers):
        weights.append(_block_weights_dev(ctx, flat, layer, config))
    var head_base = offsets[config.n_tensors() - 1]
    var emb_w = _param_dev_copy(ctx, flat, 0, vd)
    var lm_w = _param_dev_copy(ctx, flat, head_base, vd)
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


# ===========================================================================
# cpu3-seq (2026-10-04): GREEDY NEXT BYTE ON THE DEVICE
# ===========================================================================

comptime BYTE_NEXT_TPB = 256
"""Threads per row of the next-byte argmax (a halving tree over them)."""


@always_inline
def _next_key(v: Float32) -> UInt32:
    """An order key whose unsigned order is the float order `>` for every
    non-NaN value, by BITS (contract row 49: Metal flushes compare
    operands, so a subnormal logit is never compared as a float here):
    sign set -> all bits flipped, sign clear -> sign bit set, and -0.0
    keyed as +0.0 (they compare equal). The logits reaching this kernel
    passed the device non-finite refusal, so no NaN is keyed."""
    var bits = bitcast[DType.uint32](v)
    if bits == UInt32(0x80000000):
        bits = UInt32(0)
    if (bits & UInt32(0x80000000)) != UInt32(0):
        return ~bits
    return bits | UInt32(0x80000000)


def byte_next_byte_kernel(
    picked_out: MutPointer[Int32, MutAnyOrigin],
    logits: MutPointer[Float32, MutAnyOrigin],
    length_in: Int32,
    vocab_in: Int32,
):
    """One block per row `b`: the greedy next byte, the argmax over `vocab`
    of the row's LAST position `logits[(b * length + length - 1) * vocab:]`,
    ties to the LOWEST byte value. That is the host `argmax_rows_f32` rule
    (strict `>` scanning from index 0), as an order-free reduction: the
    larger key wins, an equal key keeps the smaller index, so every vendor
    and every block size gives the same byte. Integer keys, no float
    arithmetic: no bit can differ."""
    var keys = stack_allocation[
        BYTE_NEXT_TPB, Scalar[DType.uint32], address_space = AddressSpace.SHARED
    ]()
    var idxs = stack_allocation[
        BYTE_NEXT_TPB, Scalar[DType.int32], address_space = AddressSpace.SHARED
    ]()
    var tid = Int(thread_idx.x)
    var b = Int(block_idx.x)
    var length = Int(length_in)
    var vocab = Int(vocab_in)
    var base = (b * length + length - 1) * vocab
    var best_k = UInt32(0)
    var best_i = Int32(2147483647)
    var j = tid
    while j < vocab:
        var k = _next_key(logits.unsafe_load(base + j))
        if best_i == Int32(2147483647) or k > best_k:
            best_k = k
            best_i = Int32(j)
        j += BYTE_NEXT_TPB
    keys.unsafe_store(tid, best_k)
    idxs.unsafe_store(tid, best_i)
    barrier()
    var active = BYTE_NEXT_TPB // 2
    while active > 0:
        if tid < active:
            var ok = keys.unsafe_load(tid + active)
            var oi = idxs.unsafe_load(tid + active)
            var mk = keys.unsafe_load(tid)
            var mi = idxs.unsafe_load(tid)
            if oi != Int32(2147483647) and (
                mi == Int32(2147483647) or ok > mk or (ok == mk and oi < mi)
            ):
                keys.unsafe_store(tid, ok)
                idxs.unsafe_store(tid, oi)
        barrier()
        active = active // 2
    if tid == 0:
        var r = idxs.unsafe_load(0)
        picked_out.unsafe_store(b, Int32(0) if r == Int32(2147483647) else r)


def byte_next_bytes_resident_into(ctx: DeviceContext, mut tr: ByteTrainer, inputs: List[Int32],
                                  batch: Int, length: Int,
                                  mut scratch: Optional[ByteLogitsScratch],
                                  destination: MutPointer[Int32, MutUntrackedOrigin]) raises:
    """cpu3-seq: `byte_logits_resident_into`'s forward and refusals, then the
    greedy next byte of every row on the device (`byte_next_byte_kernel`)
    and ONE download of `batch` int32 into `destination`. The logits never
    leave the device (the Python glue downloaded `batch * length * vocab`
    floats and ran the argmax on the host). Same failure convention."""
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
        ref sc = scratch.value()
        var m = batch * length
        var vocab = config.vocab_size
        _logits_enqueue(ctx, tr.weights, tr.buffers.emb_w, tr.buffers.lm_w, tr.rope,
            sc, inputs, batch, length, config)
        var bad = device_first_nonfinite(ctx, sc.logits, m * vocab)
        if bad >= 0:
            _raise_nonfinite_logit(bad)
        var picked = ctx.enqueue_create_buffer[DType.int32](batch)
        ctx.enqueue_function[byte_next_byte_kernel](
            picked.unsafe_ptr(), sc.logits.unsafe_ptr(), Int32(length), Int32(vocab),
            grid_dim=(batch, 1, 1), block_dim=(BYTE_NEXT_TPB, 1, 1),
        )
        ctx.enqueue_copy(dst_ptr=destination, src_buf=picked)
        ctx.synchronize()
        _ = picked^
    except error:
        failed = True
        message = String(error)
    if failed:
        _byte_recover(ctx, tr, message)
    tr.healthy = True
