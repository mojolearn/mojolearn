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
from max.gpu.host import DeviceBuffer, DeviceContext
from core.identity_trace import IdentityTrace
from training.byte_lm_config import ByteConfig
from training.byte_lm import (
    ByteTrainer,
    _block_weights,
    _byte_check_gemm,
    _byte_recover,
    _require_profile,
    _unpack_block,
    byte_dims,
)
from training.checks.train_loop import _copy_into, _upload, _zeros, _zeros_i32, download_f32
from gemm.checks.gemm_identical import identical_gemm_into, identical_gemm_workspace_max_floats
from gemm.checks.gemm_oracle import OP_NT
from embedding.checks.embedding_identical import identical_embedding_forward_into
from embedding.checks.embedding_oracle import EmbConfig
from transformer.impl.llama.modeling_llama import (
    LlamaDeviceStages,
    LlamaDeviceWeights,
    LlamaKVCache,
    LlamaRopeTable,
    llama_decoder_layer_forward,
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


def _logits_forward(
    ctx: DeviceContext,
    mut weights: List[LlamaDeviceWeights],
    mut emb_w: DeviceBuffer[DType.float32],
    mut lm_w: DeviceBuffer[DType.float32],
    mut rope: LlamaRopeTable,
    inputs: List[Int32],
    batch: Int,
    length: Int,
    config: ByteConfig,
) raises -> List[Float32]:
    """Embedding, the blocks and the head, as `_byte_forward_loss` launches
    them, into call-shaped buffers; returns the downloaded logits
    `[batch * length, vocab]`, row-major."""
    var m = batch * length
    var dm = config.d_model
    var vocab = config.vocab_size
    var dims = byte_dims(config)

    var staging = ctx.enqueue_create_host_buffer[DType.int32](m)
    ctx.synchronize()
    for i in range(m):
        staging.unsafe_ptr().unsafe_store(i, inputs[i])
    var ids = _zeros_i32(ctx, m)
    ctx.enqueue_copy(dst_buf=ids, src_ptr=staging.unsafe_ptr())
    ctx.synchronize()
    _ = staging

    var x = _zeros(ctx, m * dm)
    identical_embedding_forward_into(ctx, x, emb_w, ids, m, EmbConfig.llama(vocab, dm))
    ctx.synchronize()

    # One cache for every block, reset to a fresh prefill before each, as the
    # trainer's `prefill_cache` is; stages sized for this call's shape.
    var cache = LlamaKVCache(ctx, batch, dims, length)
    var trace = IdentityTrace.disabled()
    var stages = List[LlamaDeviceStages]()
    for layer in range(config.n_layers):
        var st = LlamaDeviceStages(ctx, batch, length, length, dims, lean=True)
        cache.s = 0
        var prefix = String("byte.logits.block") + String(layer) + ".forward"
        if layer == 0:
            llama_decoder_layer_forward(ctx, st, cache, rope, weights[layer], x,
                batch, length, 0, trace, prefix)
        else:
            llama_decoder_layer_forward(ctx, st, cache, rope, weights[layer], stages[layer - 1].residual2,
                batch, length, 0, trace, prefix)
        ctx.synchronize()
        stages.append(st^)

    var logits = _zeros(ctx, m * vocab)
    var head_ws = _zeros(ctx, identical_gemm_workspace_max_floats(m, vocab, dm))
    identical_gemm_into(ctx, logits, stages[config.n_layers - 1].residual2, lm_w, head_ws,
        m, vocab, dm, OP_NT)
    ctx.synchronize()
    var out = download_f32(ctx, logits, m * vocab)
    _ = trace
    for i in range(len(out)):
        if (bitcast[DType.uint32](out[i]) & UInt32(0x7F800000)) == UInt32(0x7F800000):
            raise Error("byte LM logits: non-finite logit at flat index " + String(i)
                        + " REFUSED (NaN payloads are vendor-shaped, IDENTITY_PATHS row 39)")
    return out^


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
    ctx.synchronize()
    var out = _logits_forward(ctx, weights, emb_w, lm_w, rope, inputs, batch, length, config)
    ctx.synchronize()
    return out^


def byte_logits_resident(ctx: DeviceContext, mut tr: ByteTrainer, inputs: List[Int32],
                         batch: Int, length: Int) raises -> List[Float32]:
    """The resident call: refresh the trainer's weight views, embedding and
    head from its flat device parameters exactly as `_byte_forward_loss`
    does, then `_logits_forward`. The failure convention is
    `byte_eval_loss_resident`'s: nothing was written, so a failure re-scans
    the state once (`_byte_recover`) before the trainer is healthy again."""
    var config = tr.config.copy()
    byte_logits_validate(inputs, batch, length, config)
    if not tr.healthy:
        raise Error("byte LM: session lost; restore a retained export")
    tr.healthy = False
    tr.shadow_valid = False
    var out = List[Float32]()
    var failed = False
    var message = String("")
    try:
        for layer in range(config.n_layers):
            _unpack_block(ctx, tr.buffers, tr.weights[layer], layer)
        _copy_into(ctx, tr.buffers.emb_w, tr.buffers.param, 0, 0, config.vocab_size * config.d_model)
        _copy_into(ctx, tr.buffers.lm_w, tr.buffers.param, 0,
            tr.buffers.offsets[config.n_tensors() - 1], config.vocab_size * config.d_model)
        ctx.synchronize()
        out = _logits_forward(ctx, tr.weights, tr.buffers.emb_w, tr.buffers.lm_w, tr.rope,
            inputs, batch, length, config)
        ctx.synchronize()
    except error:
        failed = True
        message = String(error)
    if failed:
        _byte_recover(ctx, tr, message)
    tr.healthy = True
    return out^
