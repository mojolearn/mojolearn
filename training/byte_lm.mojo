# SPDX-License-Identifier: Apache-2.0
"""FP32 decoder LM orchestration with runtime shapes.

Runtime shapes are compile/host checked; numerical qualification is separate.

Runtime shapes; default B2/L32/DM32/H4/KV2/FF64, 34,944 parameters.
Vocabulary, layer count and the flat parameter registry follow ByteConfig.
Layer construction/forward order follows transformers/models/llama/modeling_llama.py:354-356,402-412.
The existing no-final-norm, untied-head architecture is preserved.
Every numerical operation is an existing embedding/GEMM/Llama/loss/AdamW op.
Caller supplies actual parameters, moments, flags, completed step and token IDs.
No generated data, hidden initialization, tokenizer, performance or learning claim.
"""
from std.memory import bitcast
from std.time import perf_counter_ns
from max.gpu.host import DeviceBuffer, DeviceContext
from core.identity_trace import IdentityTrace
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
from checks.vendor import COMPILED_VENDOR
from training.checkpoint import Checkpoint
from training.byte_lm_config import ByteConfig
from training.checks.train_loop import (
    _zeros, _zeros_i32, _upload, _ones, _copy_into, download_f32,
)
from gemm.checks.gemm_identical import (
    ANY_SABOTAGE as GEMM_SABOTAGE, identical_gemm_into,
    identical_gemm_workspace_max_floats,
)
from gemm.checks.gemm_backward import (
    ANY_BWD_SABOTAGE as GEMM_BWD_SABOTAGE, identical_gemm_backward_a_into,
    identical_gemm_backward_b_into, identical_gemm_backward_workspace_max_floats,
)
from gemm.checks.gemm_oracle import OP_NT, OP_NN
from embedding.checks.embedding_identical import (
    ANY_EMB_SABOTAGE, emb_run_scratch_ints, identical_embedding_forward_into,
    identical_embedding_backward_into,
)
from embedding.checks.embedding_oracle import EmbConfig
from training.checks.loss import (
    ANY_LOSS_SABOTAGE, identical_ce_forward_into, identical_ce_backward_into,
    identical_ce_ones_floats, identical_ce_workspace_max_floats,
)
from training.checks.loss_oracle import REDUCTION_MEAN, CeConfig
from training.checks.optimizer import (
    ANY_SABOTAGE as OPT_SABOTAGE, OPT_RECORD_INTERMEDIATES, SAB_CHUNKS, identical_optimizer_step,
    identical_optimizer_workspace_floats,
)
from training.checks.optimizer_oracle import OPT_ADAMW, OptimizerConfig
from transformer.checks.transformer_backward import (
    BWD_ANY_SABOTAGE, LlamaBackwardStages, llama_decoder_layer_backward_device,
)
from transformer.impl.llama.modeling_llama import (
    BLOCK_ANY_SABOTAGE, LlamaDims, LlamaDeviceWeights, LlamaDeviceStages,
    LlamaRopeTable, LlamaKVCache, llama_decoder_layer_forward,
    timing_on, timing_tick,
)

comptime BYTE_B = 2
comptime BYTE_L = 32
comptime BYTE_DM = 32
comptime BYTE_V = 256
comptime BYTE_J = 20
comptime BYTE_N_TOTAL = 34944
comptime BYTE_PROFILE = "mojolearn.byte-lm.b2-l32-d32-h4-kv2-ff64-v256-blocks2.fp32.v1"


def byte_dims(config: ByteConfig = ByteConfig()) raises -> LlamaDims:
    config.validate()
    return LlamaDims(config.d_model, config.n_heads, config.n_kv,
                     config.head_dim, config.intermediate)


def byte_param_count(j: Int, config: ByteConfig = ByteConfig()) raises -> Int:
    return config.param_count(j)


def byte_param_name(j: Int, config: ByteConfig = ByteConfig()) raises -> String:
    if j < 0 or j >= config.n_tensors():
        raise Error("byte LM: parameter index out of range")
    if j == 0:
        return "embed"
    if j == config.n_tensors() - 1:
        return "lm_head"
    var names = List[String]()
    names.append("norm1_w")
    names.append("w_q")
    names.append("w_k")
    names.append("w_v")
    names.append("w_o")
    names.append("norm2_w")
    names.append("w_gate")
    names.append("w_up")
    names.append("w_down")
    return String("block") + String((j - 1) // 9) + "." + names[(j - 1) % 9]


def byte_offsets(config: ByteConfig = ByteConfig()) raises -> List[Int]:
    return config.offsets()


def byte_names(config: ByteConfig = ByteConfig()) raises -> List[String]:
    var result = List[String]()
    for j in range(config.n_tensors()):
        result.append(byte_param_name(j, config))
    return result^


def timing_bytes(on: Bool, name: String, n_bytes: Int):
    """DEVIATION 2499: the byte count of a transfer phase, on the same
    `timing` line shape the phase timers use (`timing <name> <n> bytes`),
    so a reader can put bandwidth beside conversion. Prints only under
    `MOJOLEARN_TRANSFORMER_TIMING=1`; costs nothing otherwise."""
    if not on:
        return
    print("timing " + name + " " + String(n_bytes) + " bytes")


def _require_finite(values: List[Float32], name: String) raises:
    for i in range(len(values)):
        if (bitcast[DType.uint32](values[i]) & UInt32(0x7F800000)) == UInt32(0x7F800000):
            raise Error("byte LM: nonfinite " + name + " at " + String(i))


def byte_validate_state(param: List[Float32], m: List[Float32], v: List[Float32],
                        flags: List[Bool], completed: Int, config: ByteConfig = ByteConfig()) raises:
    config.validate()
    var n_total = config.n_total()
    if len(param) != n_total or len(m) != n_total or len(v) != n_total or len(flags) != config.n_tensors():
        raise Error("byte LM: state length differs from canonical registry")
    if completed < 0 or completed >= 1000000:
        raise Error("byte LM: completed step must be in [0,1000000)")
    _require_finite(param, "parameters")
    _require_finite(m, "first moments")
    _require_finite(v, "second moments")
    for i in range(n_total):
        if v[i] < Float32(0):
            raise Error("byte LM: negative second moment")


def byte_validate_optimizer(cfg: OptimizerConfig) raises:
    var scalars = List[Float32]()
    scalars.append(cfg.lr)
    scalars.append(cfg.beta1)
    scalars.append(cfg.beta2)
    scalars.append(cfg.eps)
    scalars.append(cfg.weight_decay)
    scalars.append(cfg.momentum)
    scalars.append(cfg.dampening)
    scalars.append(cfg.max_norm)
    _require_finite(scalars, "optimizer configuration")
    if (cfg.kind != OPT_ADAMW or cfg.lr <= 0 or cfg.eps <= 0
        or cfg.beta1 < 0 or cfg.beta1 >= 1 or cfg.beta2 < 0 or cfg.beta2 >= 1
        or cfg.weight_decay < 0 or cfg.max_norm != 0 or cfg.momentum != 0
        or cfg.dampening != 0 or cfg.nesterov):
        raise Error("byte LM: requires positive-lr AdamW, finite legal betas/eps, no clipping/SGD options")


def byte_validate_tokens(ids: List[Int32], config: ByteConfig = ByteConfig()) raises:
    config.validate()
    if len(ids) != config.batch * (config.length + 1):
        raise Error("byte LM: token count differs from row-major [batch,length+1]")
    for i in range(len(ids)):
        if ids[i] < 0 or ids[i] >= Int32(config.vocab_size):
            raise Error("byte LM: token ID outside configured vocabulary")


def _require_profile() raises:
    comptime if GLOBAL_NUMERIC_MODE != NUMERIC_IDENTICAL:
        raise Error("byte LM: training requires IDENTICAL")
    comptime if (GEMM_SABOTAGE or GEMM_BWD_SABOTAGE or ANY_EMB_SABOTAGE
                 or ANY_LOSS_SABOTAGE or OPT_SABOTAGE or BWD_ANY_SABOTAGE
                 or BLOCK_ANY_SABOTAGE):
        raise Error("byte LM: numerical sabotage build refused")


def _byte_check_workspace(n: Int) raises:
    if n < 0 or n > 2147483647:
        raise Error("byte LM: workspace exceeds int32 indexing")


def _byte_check_gemm(m: Int, n: Int, k: Int) raises:
    _byte_check_workspace(identical_gemm_workspace_max_floats(m, n, k))
    _byte_check_workspace(identical_gemm_backward_workspace_max_floats(OP_NT, m, n, k, False))
    _byte_check_workspace(identical_gemm_backward_workspace_max_floats(OP_NN, m, n, k, False))


def _byte_validate_allocations(config: ByteConfig) raises:
    # Pure host sizing; no DeviceContext or allocation. Use the dispatcher's
    # exact workspace functions rather than assuming output-sized scratch.
    config.validate()
    var m = config.batch * config.length
    var widths: List[Int] = [config.d_model, config.n_kv * config.head_dim,
                            config.intermediate, config.vocab_size]
    for width in widths:
        _byte_check_gemm(m, width, config.d_model)
    _byte_check_gemm(m, config.d_model, config.intermediate)
    _byte_check_gemm(config.length, config.length, config.head_dim)
    _byte_check_gemm(config.length, config.head_dim, config.length)
    _byte_check_gemm(1, config.d_model, m)
    _byte_check_workspace(identical_ce_workspace_max_floats(m, config.vocab_size, REDUCTION_MEAN))
    _byte_check_workspace(identical_ce_ones_floats(m, config.vocab_size))
    _byte_check_workspace(identical_optimizer_workspace_floats(config.offsets()))


struct ByteBuffers(Movable):
    """Configured buffers; flat arrays are authoritative, weights are copies."""
    var config: ByteConfig
    var n_total: Int
    var offsets: List[Int]

    var param: DeviceBuffer[DType.float32]
    var grad: DeviceBuffer[DType.float32]
    var m_state: DeviceBuffer[DType.float32]
    var v_state: DeviceBuffer[DType.float32]
    var denom_out: DeviceBuffer[DType.float32]
    var q_out: DeviceBuffer[DType.float32]
    var sumsq: DeviceBuffer[DType.float32]
    var norms: DeviceBuffer[DType.float32]
    var total_cell: DeviceBuffer[DType.float32]
    var out2: DeviceBuffer[DType.float32]
    var opt_ws: DeviceBuffer[DType.float32]
    var sab_partials: DeviceBuffer[DType.float32]

    var emb_w: DeviceBuffer[DType.float32]  # [V, d_model]
    var lm_w: DeviceBuffer[DType.float32]  # [V, d_model]
    var dw_emb: DeviceBuffer[DType.float32]  # [V, d_model]
    var dw_lm: DeviceBuffer[DType.float32]  # [V, d_model]

    var ids: DeviceBuffer[DType.int32]  # [M]
    var targets: DeviceBuffer[DType.int32]  # [M]

    var x: DeviceBuffer[DType.float32]  # [M, d_model]  block input
    var logits: DeviceBuffer[DType.float32]  # [M, V]
    var d_h: DeviceBuffer[DType.float32]  # [M, d_model]

    var ce_max: DeviceBuffer[DType.float32]
    var ce_shift: DeviceBuffer[DType.float32]
    var ce_expo: DeviceBuffer[DType.float32]
    var ce_denom: DeviceBuffer[DType.float32]
    var ce_logdenom: DeviceBuffer[DType.float32]
    var ce_logp_target: DeviceBuffer[DType.float32]
    var ce_nll: DeviceBuffer[DType.float32]
    var ce_logp: DeviceBuffer[DType.float32]
    var ce_logp_sum: DeviceBuffer[DType.float32]
    var ce_smooth: DeviceBuffer[DType.float32]
    var ce_row: DeviceBuffer[DType.float32]
    var ce_total: DeviceBuffer[DType.float32]
    var ce_loss: DeviceBuffer[DType.float32]
    var ce_weights: DeviceBuffer[DType.float32]
    var ce_dlogits: DeviceBuffer[DType.float32]
    var ce_ones: DeviceBuffer[DType.float32]
    var ce_ws: DeviceBuffer[DType.float32]

    var head_ws: DeviceBuffer[DType.float32]
    var head_bwd_ws: DeviceBuffer[DType.float32]

    var emb_counts: DeviceBuffer[DType.int32]
    var emb_run_begin: DeviceBuffer[DType.int32]
    var emb_perm: DeviceBuffer[DType.int32]

    var buf_initialized: List[Bool]

    def __init__(out self, ctx: DeviceContext, initial_params: List[Float32],
                 initial_m: List[Float32], initial_v: List[Float32], flags: List[Bool],
                 config: ByteConfig = ByteConfig()) raises:
        _byte_validate_allocations(config)
        self.config = config.copy()
        var M = config.batch * config.length
        var DM = config.d_model
        var V = config.vocab_size

        self.offsets = byte_offsets(config)
        self.n_total = config.n_total()
        var n = self.n_total

        self.param = _upload(ctx, initial_params)
        self.grad = _zeros(ctx, n)
        self.m_state = _upload(ctx, initial_m)
        self.v_state = _upload(ctx, initial_v)
        # Recording kernels write one intermediate per parameter.
        var record_n = 1
        comptime if OPT_RECORD_INTERMEDIATES:
            record_n = n
        self.denom_out = _zeros(ctx, record_n)
        self.q_out = _zeros(ctx, record_n)
        self.sumsq = _zeros(ctx, config.n_tensors())
        self.norms = _zeros(ctx, config.n_tensors())
        self.total_cell = _zeros(ctx, 1)
        self.out2 = _zeros(ctx, 2)
        self.opt_ws = _zeros(
            ctx, identical_optimizer_workspace_floats(self.offsets)
        )
        self.sab_partials = _zeros(ctx, SAB_CHUNKS)

        self.emb_w = _zeros(ctx, V * DM)
        self.lm_w = _zeros(ctx, V * DM)
        self.dw_emb = _zeros(ctx, V * DM)
        self.dw_lm = _zeros(ctx, V * DM)

        self.ids = _zeros_i32(ctx, M)
        self.targets = _zeros_i32(ctx, M)

        self.x = _zeros(ctx, M * DM)
        self.logits = _zeros(ctx, M * V)
        self.d_h = _zeros(ctx, M * DM)

        self.ce_max = _zeros(ctx, M)
        self.ce_shift = _zeros(ctx, M * V)
        self.ce_expo = _zeros(ctx, M * V)
        self.ce_denom = _zeros(ctx, M)
        self.ce_logdenom = _zeros(ctx, M)
        self.ce_logp_target = _zeros(ctx, M)
        self.ce_nll = _zeros(ctx, M)
        self.ce_logp = _zeros(ctx, 1)
        self.ce_logp_sum = _zeros(ctx, 1)
        self.ce_smooth = _zeros(ctx, 1)
        self.ce_row = _zeros(ctx, M)
        self.ce_total = _zeros(ctx, 1)
        self.ce_loss = _zeros(ctx, 1)
        self.ce_weights = _zeros(ctx, M * V)
        self.ce_dlogits = _zeros(ctx, M * V)
        self.ce_ones = _ones(ctx, identical_ce_ones_floats(M, V))
        self.ce_ws = _zeros(
            ctx, identical_ce_workspace_max_floats(M, V, REDUCTION_MEAN)
        )

        self.head_ws = _zeros(
            ctx, identical_gemm_workspace_max_floats(M, V, DM)
        )
        self.head_bwd_ws = _zeros(
            ctx,
            identical_gemm_backward_workspace_max_floats(
                OP_NT, M, V, DM, False
            ),
        )

        var scratch = emb_run_scratch_ints(V, M)
        if scratch != V + (V + 1) + M:
            raise Error(
                String("byte LM: emb_run_scratch_ints says ")
                + String(scratch)
                + " ints and counts+run_begin+perm is "
                + String(V + (V + 1) + M)
                + ". The embedding lane changed its run structure and this"
                + " harness would hand it three buffers of the wrong size."
            )
        self.emb_counts = _zeros_i32(ctx, V)
        self.emb_run_begin = _zeros_i32(ctx, V + 1)
        self.emb_perm = _zeros_i32(ctx, M)

        self.buf_initialized = flags.copy()




def _param_slice(values: List[Float32], j: Int, config: ByteConfig) raises -> List[Float32]:
    var offsets = byte_offsets(config)
    var result = List[Float32]()
    for i in range(offsets[j], offsets[j + 1]):
        result.append(values[i])
    return result^


def _block_weights(ctx: DeviceContext, params: List[Float32], block: Int, config: ByteConfig) raises -> LlamaDeviceWeights:
    var base = 1 + 9 * block
    return LlamaDeviceWeights(ctx, byte_dims(config), Float32(1e-6),
        _param_slice(params, base, config), _param_slice(params, base + 5, config),
        _param_slice(params, base + 1, config), _param_slice(params, base + 2, config),
        _param_slice(params, base + 3, config), _param_slice(params, base + 4, config),
        _param_slice(params, base + 6, config), _param_slice(params, base + 7, config),
        _param_slice(params, base + 8, config))


def _unpack_block(ctx: DeviceContext, mut tb: ByteBuffers, mut w: LlamaDeviceWeights, block: Int) raises:
    var base = 1 + 9 * block
    var o = tb.offsets.copy()
    _copy_into(ctx, w.norm1_w, tb.param, 0, o[base], o[base + 1] - o[base])
    _copy_into(ctx, w.w_q, tb.param, 0, o[base + 1], o[base + 2] - o[base + 1])
    _copy_into(ctx, w.w_k, tb.param, 0, o[base + 2], o[base + 3] - o[base + 2])
    _copy_into(ctx, w.w_v, tb.param, 0, o[base + 3], o[base + 4] - o[base + 3])
    _copy_into(ctx, w.w_o, tb.param, 0, o[base + 4], o[base + 5] - o[base + 4])
    _copy_into(ctx, w.norm2_w, tb.param, 0, o[base + 5], o[base + 6] - o[base + 5])
    _copy_into(ctx, w.w_gate, tb.param, 0, o[base + 6], o[base + 7] - o[base + 6])
    _copy_into(ctx, w.w_up, tb.param, 0, o[base + 7], o[base + 8] - o[base + 7])
    _copy_into(ctx, w.w_down, tb.param, 0, o[base + 8], o[base + 9] - o[base + 8])
    ctx.synchronize()


def _pack_block(ctx: DeviceContext, mut tb: ByteBuffers, mut bst: LlamaBackwardStages, block: Int) raises:
    var base = 1 + 9 * block
    var o = tb.offsets.copy()
    _copy_into(ctx, tb.grad, bst.dw_norm1, o[base], 0, o[base + 1] - o[base])
    _copy_into(ctx, tb.grad, bst.dw_q, o[base + 1], 0, o[base + 2] - o[base + 1])
    _copy_into(ctx, tb.grad, bst.dw_k, o[base + 2], 0, o[base + 3] - o[base + 2])
    _copy_into(ctx, tb.grad, bst.dw_v, o[base + 3], 0, o[base + 4] - o[base + 3])
    _copy_into(ctx, tb.grad, bst.dw_o, o[base + 4], 0, o[base + 5] - o[base + 4])
    _copy_into(ctx, tb.grad, bst.dw_norm2, o[base + 5], 0, o[base + 6] - o[base + 5])
    _copy_into(ctx, tb.grad, bst.dw_gate, o[base + 6], 0, o[base + 7] - o[base + 6])
    _copy_into(ctx, tb.grad, bst.dw_up, o[base + 7], 0, o[base + 8] - o[base + 7])
    _copy_into(ctx, tb.grad, bst.dw_down, o[base + 8], 0, o[base + 9] - o[base + 8])
    ctx.synchronize()


struct ByteTrainer(Movable):
    """Owned device state. Use only with the same DeviceContext that created it.

    Do not mutate fields externally. A failed numerical step poisons this object;
    reconstruct from the last successful retained state before continuing.
    """
    var config: ByteConfig
    var buffers: ByteBuffers
    var weights: List[LlamaDeviceWeights]
    var rope: LlamaRopeTable
    var prefill_cache: LlamaKVCache
    var forward: List[LlamaDeviceStages]
    var backward: List[LlamaBackwardStages]
    var optimizer: OptimizerConfig
    var completed_steps: Int
    var healthy: Bool

    def __init__(out self, ctx: DeviceContext, initial_params: List[Float32],
                 initial_m: List[Float32], initial_v: List[Float32],
                 flags: List[Bool], completed_steps: Int, optimizer: OptimizerConfig,
                 config: ByteConfig = ByteConfig()) raises:
        # All supplied host state/configuration admitted before first allocation.
        _require_profile()
        _byte_validate_allocations(config)
        self.config = config.copy()
        byte_validate_state(initial_params, initial_m, initial_v, flags, completed_steps, config)
        byte_validate_optimizer(optimizer)
        self.optimizer = optimizer.copy()
        self.completed_steps = completed_steps
        self.healthy = True
        self.buffers = ByteBuffers(ctx, initial_params, initial_m, initial_v, flags, config)
        self.weights = List[LlamaDeviceWeights]()
        self.forward = List[LlamaDeviceStages]()
        self.backward = List[LlamaBackwardStages]()
        self.rope = LlamaRopeTable(ctx, byte_dims(config), Float32(10000), config.length)
        self.prefill_cache = LlamaKVCache(ctx, config.batch, byte_dims(config), config.length)
        for layer in range(config.n_layers):
            self.weights.append(_block_weights(ctx, initial_params, layer, config))
            # The trace-disabled trainer uses the existing fused attention path.
            # Allocate its quadratic stages lazily; eager/diagnostic fallback
            # still grows them through ensure_*_attention_capacity.
            self.forward.append(LlamaDeviceStages(ctx, config.batch, config.length, config.length, byte_dims(config), lean=True))
            self.backward.append(LlamaBackwardStages(ctx, config.batch, config.length, config.length, byte_dims(config), lean=True))
        ctx.synchronize()


@fieldwise_init
struct ByteStepCapture(Movable):
    """Host-owned full step. Gradients/loss are BEFORE the update.

    Loss is the mean of batch*length next-byte cross-entropies: inputs
    [:,:length], targets [:,1:length+1]. No ignored positions, padding, label smoothing or batch splitting.
    All arrays use canonical byte_offsets()/byte_names() order; FP32 raw bits
    must be retained, never converted through decimal text by an adapter.
    """
    var token_ids: List[Int32]
    var before_params: List[Float32]
    var before_m: List[Float32]
    var before_v: List[Float32]
    var before_flags: List[Bool]
    var gradients: List[Float32]
    var after_params: List[Float32]
    var after_m: List[Float32]
    var after_v: List[Float32]
    var after_flags: List[Bool]
    var loss: Float32
    var completed_steps: Int
    var optimizer: OptimizerConfig
    var profile: String
    var numeric_mode: String
    var vendor: String


def byte_train_step(ctx: DeviceContext, mut trainer: ByteTrainer,
                    token_ids: List[Int32]) raises -> ByteStepCapture:
    """One supplied-batch step; reuses every existing numerical primitive."""
    _require_profile()
    var config = trainer.config.copy()
    byte_validate_tokens(token_ids, config)
    byte_validate_optimizer(trainer.optimizer)
    if not trainer.healthy:
        raise Error("byte LM: previous step failed; restore a retained checkpoint")
    if trainer.completed_steps < 0 or trainer.completed_steps >= 999999:
        raise Error("byte LM: step bound reached")
    # Device-state validation requires readback. It precedes all numerical work
    # and all writes; downloads themselves are the only device operations here.
    # DEVIATION 2499: step-phase timers, IDENTICAL-only instrumentation
    # behind MOJOLEARN_TRANSFORMER_TIMING=1 (the switch the block timers
    # use). Every `timing_tick` synchronizes the context before it reads
    # the clock; on a phase that already ends with its own wait or a host
    # scan the extra wait is a no-op, and where a phase was asynchronous
    # the tick's wait exists ONLY under the switch. Off, nothing here runs.
    var ton = timing_on()
    var tk = Int(perf_counter_ns())
    var before_p = download_f32(ctx, trainer.buffers.param, config.n_total())
    var before_m = download_f32(ctx, trainer.buffers.m_state, config.n_total())
    var before_v = download_f32(ctx, trainer.buffers.v_state, config.n_total())
    var before_flags = trainer.buffers.buf_initialized.copy()
    # download_f32 waits inside; the three Lists are complete on the host.
    timing_tick(ctx, ton, tk, "step.mirror_download_before")
    timing_bytes(ton, "step.mirror_download_before_bytes", 3 * config.n_total() * 4)
    byte_validate_state(before_p, before_m, before_v, before_flags, trainer.completed_steps, config)
    timing_tick(ctx, ton, tk, "step.validate_before")
    trainer.healthy = False
    var capture = _byte_step_admitted(ctx, trainer, token_ids, before_p, before_m, before_v, before_flags)
    trainer.healthy = True
    return capture^


def _byte_forward_loss(ctx: DeviceContext, mut tr: ByteTrainer,
                       ids: List[Int32], mut trace: IdentityTrace) raises -> Float32:
    """Shared forward only. Authoritative params/m/v/flags/t are read-only."""
    var config = tr.config.copy()
    var M = config.batch * config.length
    var ton = timing_on()
    var tk = Int(perf_counter_ns())
    var inputs = List[Int32]()
    var targets = List[Int32]()
    for b in range(config.batch):
        for l in range(config.length):
            inputs.append(ids[b * (config.length + 1) + l])
            targets.append(ids[b * (config.length + 1) + l + 1])
    var hi = ctx.enqueue_create_host_buffer[DType.int32](M)
    var ht = ctx.enqueue_create_host_buffer[DType.int32](M)
    ctx.synchronize()
    for i in range(M):
        hi.unsafe_ptr().unsafe_store(i, inputs[i])
        ht.unsafe_ptr().unsafe_store(i, targets[i])
    ctx.enqueue_copy(dst_buf=tr.buffers.ids, src_ptr=hi.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=tr.buffers.targets, src_ptr=ht.unsafe_ptr())
    ctx.synchronize()
    _ = hi
    _ = ht
    # The wait above completes both uploads: host split, staging, H2D.
    timing_tick(ctx, ton, tk, "step.upload_inputs")
    timing_bytes(ton, "step.upload_inputs_bytes", 2 * M * 4)
    for layer in range(config.n_layers):
        _unpack_block(ctx, tr.buffers, tr.weights[layer], layer)
    _copy_into(ctx, tr.buffers.emb_w, tr.buffers.param, 0, 0, config.vocab_size * config.d_model)
    _copy_into(ctx, tr.buffers.lm_w, tr.buffers.param, 0, tr.buffers.offsets[config.n_tensors() - 1], config.vocab_size * config.d_model)
    ctx.synchronize()
    # Device-to-device: every parameter byte copied once (blocks, emb, head).
    timing_tick(ctx, ton, tk, "step.unpack_weights")
    timing_bytes(ton, "step.unpack_weights_bytes", config.n_total() * 4)
    var emb = EmbConfig.llama(config.vocab_size, config.d_model)
    var ce = CeConfig.causal_lm(config.vocab_size)
    identical_embedding_forward_into(ctx, tr.buffers.x, tr.buffers.emb_w, tr.buffers.ids, M, emb)
    ctx.synchronize()
    timing_tick(ctx, ton, tk, "step.embedding_forward")
    for layer in range(config.n_layers):
        # Move the current stages out while borrowing the preceding residual.
        # No extra activation copy; restore canonical layer order after the call.
        var stages = tr.forward.pop(layer)
        # Upstream modeling_llama.py:402-412 visits independent decoder layers.
        # Training always starts a full prefill: s=0 makes kv_append_kernel
        # read only fresh K/V. Reuse storage, never another layer's history.
        # Backward reads the per-layer stages.k_cache/v_cache, not this scratch.
        tr.prefill_cache.s = 0
        var prefix = String("byte.block") + String(layer) + ".forward"
        if layer == 0:
            llama_decoder_layer_forward(ctx, stages, tr.prefill_cache, tr.rope, tr.weights[layer],
                tr.buffers.x, config.batch, config.length, 0, trace, prefix)
        else:
            llama_decoder_layer_forward(ctx, stages, tr.prefill_cache, tr.rope, tr.weights[layer],
                tr.forward[layer - 1].residual2, config.batch, config.length, 0, trace, prefix)
        ctx.synchronize()
        tr.forward.insert(layer, stages^)
    # The blocks print their own `block.*` / `attn.*` lines; this envelope
    # is the whole forward loop including the per-layer waits and the
    # stage pop/insert bookkeeping, so the block sum can be checked.
    timing_tick(ctx, ton, tk, "envelope.blocks_forward")
    identical_gemm_into(ctx, tr.buffers.logits, tr.forward[config.n_layers - 1].residual2,
        tr.buffers.lm_w, tr.buffers.head_ws, M, config.vocab_size, config.d_model, OP_NT)
    ctx.synchronize()
    timing_tick(ctx, ton, tk, "step.head_forward")
    # `identical_ce_forward_into` prints `step.ce_refuse_download` (its
    # first statement: the full logits download and host scan) and then
    # `step.ce_forward` (L1-L13, waited on under the switch only), both from
    # its own clock; this clock is re-read after the call's wait.
    identical_ce_forward_into(ctx, tr.buffers.ce_max, tr.buffers.ce_shift,
        tr.buffers.ce_expo, tr.buffers.ce_denom, tr.buffers.ce_logdenom,
        tr.buffers.ce_logp_target, tr.buffers.ce_nll, tr.buffers.ce_logp,
        tr.buffers.ce_logp_sum, tr.buffers.ce_smooth, tr.buffers.ce_row,
        tr.buffers.ce_total, tr.buffers.ce_loss, tr.buffers.logits,
        tr.buffers.targets, tr.buffers.ce_ones, tr.buffers.ce_ws, M, M, ce)
    ctx.synchronize()
    if ton:
        tk = Int(perf_counter_ns())
    var losses = download_f32(ctx, tr.buffers.ce_loss, 1)
    _require_finite(losses, "loss")
    timing_tick(ctx, ton, tk, "step.loss_download")
    timing_bytes(ton, "step.loss_download_bytes", 4)
    return losses[0]


def _byte_step_admitted(ctx: DeviceContext, mut tr: ByteTrainer,
                        ids: List[Int32], before_p: List[Float32],
                        before_m: List[Float32], before_v: List[Float32],
                        before_flags: List[Bool]) raises -> ByteStepCapture:
    var config = tr.config.copy()
    var M = config.batch * config.length
    var emb = EmbConfig.llama(config.vocab_size, config.d_model)
    var ce = CeConfig.causal_lm(config.vocab_size)
    var trace = IdentityTrace.disabled()
    var loss = _byte_forward_loss(ctx, tr, ids, trace)
    # DEVIATION 2499 step-phase timers; see byte_train_step. The forward's
    # own clock ended at `step.loss_download`; this one starts here.
    var ton = timing_on()
    var tk = Int(perf_counter_ns())
    identical_ce_backward_into(ctx, tr.buffers.ce_weights, tr.buffers.ce_dlogits,
        tr.buffers.ce_expo, tr.buffers.ce_denom, tr.buffers.ce_logp,
        tr.buffers.targets, M, M, ce)
    ctx.synchronize()
    timing_tick(ctx, ton, tk, "step.ce_backward")
    identical_gemm_backward_a_into(ctx, tr.buffers.d_h, tr.buffers.ce_dlogits,
        tr.buffers.lm_w, tr.buffers.head_bwd_ws, M, config.vocab_size, config.d_model, OP_NT)
    # dA and dB share one wait below; the tick between them waits ONLY
    # under the switch (a timed step is not a sample, and the two GEMMs are
    # queued on one in-order context either way).
    timing_tick(ctx, ton, tk, "step.head_backward_da")
    identical_gemm_backward_b_into(ctx, tr.buffers.dw_lm, tr.buffers.ce_dlogits,
        tr.forward[config.n_layers - 1].residual2, tr.buffers.head_bwd_ws, M, config.vocab_size, config.d_model, OP_NT)
    ctx.synchronize()
    timing_tick(ctx, ton, tk, "step.head_backward_db")
    # Keep inter-layer cotangents on device, as the upstream tensor graph
    # does (transformers/models/llama/modeling_llama.py:402-412). Each
    # backward call synchronizes before its borrowed buffers are reinserted.
    for layer in range(config.n_layers - 1, -1, -1):
        var stages = tr.forward.pop(layer)
        var backward = tr.backward.pop(layer)
        var prefix = String("byte.block") + String(layer) + ".backward"
        if layer == config.n_layers - 1:
            if layer == 0:
                llama_decoder_layer_backward_device(ctx, backward, stages, tr.weights[layer],
                    tr.rope.cos, tr.rope.sin, tr.buffers.x, tr.buffers.d_h,
                    config.batch, config.length, 0, trace, prefix)
            else:
                llama_decoder_layer_backward_device(ctx, backward, stages, tr.weights[layer],
                    tr.rope.cos, tr.rope.sin, tr.forward[layer - 1].residual2, tr.buffers.d_h,
                    config.batch, config.length, 0, trace, prefix)
        elif layer == 0:
            llama_decoder_layer_backward_device(ctx, backward, stages, tr.weights[layer],
                tr.rope.cos, tr.rope.sin, tr.buffers.x, tr.backward[layer].d_x,
                config.batch, config.length, 0, trace, prefix)
        else:
            llama_decoder_layer_backward_device(ctx, backward, stages, tr.weights[layer],
                tr.rope.cos, tr.rope.sin, tr.forward[layer - 1].residual2, tr.backward[layer].d_x,
                config.batch, config.length, 0, trace, prefix)
        ctx.synchronize()
        tr.backward.insert(layer, backward^)
        tr.forward.insert(layer, stages^)
    # Envelope of the whole backward loop (the blocks print `bwd.*`).
    timing_tick(ctx, ton, tk, "envelope.blocks_backward")
    identical_embedding_backward_into(ctx, tr.buffers.dw_emb, tr.backward[0].d_x,
        tr.buffers.ids, tr.buffers.emb_counts, tr.buffers.emb_run_begin,
        tr.buffers.emb_perm, M, emb)
    ctx.synchronize()
    timing_tick(ctx, ton, tk, "step.embedding_backward")
    for layer in range(config.n_layers):
        _pack_block(ctx, tr.buffers, tr.backward[layer], layer)
    _copy_into(ctx, tr.buffers.grad, tr.buffers.dw_emb, 0, 0, config.vocab_size * config.d_model)
    _copy_into(ctx, tr.buffers.grad, tr.buffers.dw_lm, tr.buffers.offsets[config.n_tensors() - 1], 0, config.vocab_size * config.d_model)
    ctx.synchronize()
    # Device-to-device: every gradient byte copied once into `grad`.
    timing_tick(ctx, ton, tk, "step.pack_grads")
    timing_bytes(ton, "step.pack_grads_bytes", config.n_total() * 4)
    # Capture actual raw pre-update gradients before ANY optimizer writes.
    var grads = download_f32(ctx, tr.buffers.grad, config.n_total())
    timing_tick(ctx, ton, tk, "step.mirror_download_grads")
    timing_bytes(ton, "step.mirror_download_grads_bytes", config.n_total() * 4)
    _require_finite(grads, "gradients")
    timing_tick(ctx, ton, tk, "step.validate_grads")
    var next_step = tr.completed_steps + 1
    # `identical_optimizer_step` prints `step.opt_refuse_download` (its
    # first statement: param, grad, m, v downloaded and scanned on the host)
    # and `step.optimizer` (clip, scalars, update; it waits before it
    # returns), both from its own clock; this clock is re-read after it.
    identical_optimizer_step(ctx, tr.buffers.param, tr.buffers.grad,
        tr.buffers.m_state, tr.buffers.v_state, tr.buffers.denom_out,
        tr.buffers.q_out, tr.buffers.sumsq, tr.buffers.norms,
        tr.buffers.total_cell, tr.buffers.out2, tr.buffers.opt_ws,
        tr.buffers.sab_partials, tr.buffers.buf_initialized, tr.buffers.offsets,
        tr.optimizer, next_step)
    if ton:
        tk = Int(perf_counter_ns())
    var after_p = download_f32(ctx, tr.buffers.param, config.n_total())
    var after_m = download_f32(ctx, tr.buffers.m_state, config.n_total())
    var after_v = download_f32(ctx, tr.buffers.v_state, config.n_total())
    var after_flags = tr.buffers.buf_initialized.copy()
    timing_tick(ctx, ton, tk, "step.mirror_download_after")
    timing_bytes(ton, "step.mirror_download_after_bytes", 3 * config.n_total() * 4)
    byte_validate_state(after_p, after_m, after_v, after_flags, next_step, config)
    ctx.synchronize()
    timing_tick(ctx, ton, tk, "step.validate_after")
    tr.completed_steps = next_step
    # Explicit last uses retain all async operands through completion.
    _ = trace
    var capture = ByteStepCapture(ids.copy(), before_p.copy(), before_m.copy(),
        before_v.copy(), before_flags.copy(), grads^, after_p^, after_m^,
        after_v^, after_flags^, loss, next_step, tr.optimizer.copy(),
        config.profile(), "identical", String(COMPILED_VENDOR))
    # Host-only: the second copy of before_p/m/v (3n floats) and the ids.
    timing_tick(ctx, ton, tk, "step.capture_copy")
    timing_bytes(ton, "step.capture_copy_bytes", 3 * config.n_total() * 4 + len(ids) * 4)
    return capture^


def byte_checkpoint(capture: ByteStepCapture, seed: UInt64,
                    steps_planned: Int, config: ByteConfig = ByteConfig()) raises -> Checkpoint:
    """Encode the successful post-step state using the unchanged v1 codec.

    Seed is descriptive data-schedule metadata supplied by caller, not hidden
    initialization. A required sidecar binds profile, raw corpus bytes, token
    schedule/cursor, numeric mode, optimizer bits, source and checkpoint hash.
    The codec alone cannot admit a resume of this configured architecture.
    """
    if (capture.profile != config.profile() or capture.numeric_mode != "identical"
        or steps_planned < capture.completed_steps or steps_planned >= 1000000):
        raise Error("byte LM: checkpoint profile/planned steps mismatch")
    byte_validate_state(capture.after_params, capture.after_m, capture.after_v,
        capture.after_flags, capture.completed_steps, config)
    byte_validate_optimizer(capture.optimizer)
    var ck = Checkpoint()
    ck.param = capture.after_params.copy()
    ck.m_state = capture.after_m.copy()
    ck.v_state = capture.after_v.copy()
    ck.buf_initialized = capture.after_flags.copy()
    ck.offsets = byte_offsets(config)
    ck.names = byte_names(config)
    ck.t = capture.completed_steps
    ck.seed = seed
    ck.opt_kind = capture.optimizer.kind
    ck.lr = capture.optimizer.lr
    ck.beta1 = capture.optimizer.beta1
    ck.beta2 = capture.optimizer.beta2
    ck.eps = capture.optimizer.eps
    ck.weight_decay = capture.optimizer.weight_decay
    ck.momentum = capture.optimizer.momentum
    ck.dampening = capture.optimizer.dampening
    ck.nesterov = capture.optimizer.nesterov
    ck.max_norm = capture.optimizer.max_norm
    ck.steps_planned = steps_planned
    ck.arm = 0
    ck.validate()
    return ck^


def byte_eval_loss(ctx: DeviceContext, mut trainer: ByteTrainer,
                   token_ids: List[Int32]) raises -> Float32:
    """Forward-only mean next-byte loss. No backward or optimizer invocation.

    Authoritative parameters/moments/flags/completed_steps are unchanged.
    Scratch, weight views and the temporary health guard may change. Root must
    retain independent before/after state evidence when qualifying this promise.
    """
    _require_profile()
    var config = trainer.config.copy()
    byte_validate_tokens(token_ids, config)
    byte_validate_optimizer(trainer.optimizer)
    if not trainer.healthy:
        raise Error("byte LM: previous operation failed; restore retained state")
    var p = download_f32(ctx, trainer.buffers.param, config.n_total())
    var m = download_f32(ctx, trainer.buffers.m_state, config.n_total())
    var v = download_f32(ctx, trainer.buffers.v_state, config.n_total())
    byte_validate_state(p, m, v, trainer.buffers.buf_initialized, trainer.completed_steps, config)
    trainer.healthy = False
    var trace = IdentityTrace.disabled()
    var loss = _byte_forward_loss(ctx, trainer, token_ids, trace)
    ctx.synchronize()
    _ = trace
    trainer.healthy = True
    return loss
