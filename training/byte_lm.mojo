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
from std.os import getenv
from std.sys.compile import is_defined
from std.time import perf_counter_ns
from max.gpu.host import DeviceBuffer, DeviceContext
# DEVIATION 2630: the step phase timers and counters (core/step_phase.mojo;
# compiled only under -D MOJOLEARN_STEP_PHASE_TIMERS=1).
from core.step_phase import (
    StepPhaseClock,
    step_count_device_alloc,
    step_count_h2d,
    step_count_host_alloc,
    step_count_launch,
    step_count_sync,
)
# DEVIATIONS 2646 and 2647: the step glue update arms (core/step_glue.mojo).
# Their path is compiled under -D MOJOLEARN_STEP_GLUE_TRIAL=1, and since
# DEVIATION 2649 also on a shipped build whose column default carries an
# update bit (`STEP_GLUE_SHIPPED_UPDATE`, NVIDIA only).
from core.step_glue import (
    STEP_GLUE_NOSHADOW,
    STEP_GLUE_OPTSKIP,
    STEP_GLUE_SHIPPED_UPDATE,
    STEP_GLUE_TRIAL,
    STEP_GLUE_UPDATE_BITS,
    step_glue_arm_from_env,
    step_glue_blocks,
)
from core.device_scan import DeviceScanScratch, device_first_token_oob
from core.identity_trace import IdentityTrace
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
from training.neural_identical_experiments import (
    IDN_LM_PARAM_VIEWS, IDN_LM_OWNED_TOKENS, IDN_TRAIN_NO_DECODE_CACHE,
    IDN_LM_VIEWS_ARM, IDN_LM_RESIDENT_TOKENS_ARM,
)
from checks.vendor import COMPILED_VENDOR
from training.byte_lm_afn import (
    AFN_LM_HEAD_FUSE, AFN_LM_NOSYNC, AFN_LM_PARAM_VIEWS, BYTE_LM_FAST_APPLE,
    afn_ce_fused, afn_reset, afn_status_view, afn_step_finish, afn_upload_ids,
)
from training.checkpoint import Checkpoint
from training.byte_lm_config import ByteConfig
from training.checks.train_loop import (
    _zeros, _zeros_i32, _upload, _ones, _copy_into, download_f32,
)
from training.byte_lm_block_copy import byte_block_copy
from gemm.neural_dispatch import (
    ANY_SABOTAGE as GEMM_SABOTAGE, identical_gemm_into,
    identical_gemm_workspace_max_floats,
)
from gemm.experiments.neural_switches import ROLE_HEAD
from gemm.neural_backward import (
    ANY_BWD_SABOTAGE as GEMM_BWD_SABOTAGE, identical_gemm_backward_a_into,
    identical_gemm_backward_b_into, identical_gemm_backward_workspace_max_floats,
)
from gemm.contract import OP_NT, OP_NN
from embedding.checks.embedding_identical import (
    ANY_EMB_SABOTAGE, EMB_AUTO_SORT_MIN_CELLS, emb_run_scratch_ints,
    identical_embedding_forward_into, identical_embedding_backward_into,
    identical_embedding_forward_prerefused_into,
    identical_embedding_backward_prerefused_into,
)
from embedding.owned_runs import NN50_OWNED_RUNS, OwnedEmbeddingRuns
from training.neural_ab_lifetime import NN62_LIFETIME_ARENA, NeuralLifetimeArena, NeuralLiveRange
from embedding.checks.embedding_oracle import EmbConfig, emb_refuse_shape
from embedding.checks.embedding_sort import PLAN_SCAN, PLAN_SORT
from training.checks.loss import (
    ANY_LOSS_SABOTAGE, identical_ce_forward_into, identical_ce_backward_into,
    identical_ce_ones_floats, identical_ce_workspace_max_floats,
)
from training.checks.loss_contract import REDUCTION_MEAN, CeConfig
from training.chunked_lm_head_v2 import (
    LM_HEAD_V2_CHUNK,
    chunked_lm_head_v2_gemm_forward_into, chunked_lm_head_v2_gemm_backward_into,
    chunked_lm_head_v2_gemm_workspace_floats,
)
from training.checks.optimizer import (
    ANY_SABOTAGE as OPT_SABOTAGE, OPT_RECORD_INTERMEDIATES, SAB_CHUNKS, identical_optimizer_step,
    identical_optimizer_workspace_floats,
    # DEVIATIONS 2646 and 2647: the pieces the glue update path launches.
    OPT_TPB, _grid_for as _opt_grid_for, adam_update_kernel, adam_update_oop_kernel, adam_update_oop_status_kernel,
    device_step_scalars, opt_refuse_device_inputs,
)
from training.neural_ab_optimizer import NN55_BLOCK_STATUS, NN56_GROUPED_ADAM, NN_ADAM_TPB, nn_adam_scalar_table, nn_grouped_adam_into
from training.checks.optimizer_contract import OPT_ADAMW, OPT_SGD, OptimizerConfig
from checks.kernel_matrix import COLUMN_APPLE, TARGET_COLUMN, byte_lm_release_eager_for
from core.device_arena import arena_begin, arena_end, arena_release
from transformer.impl.llama.fused_attention import ATTN_EXACT_TAIL_GUARD, ATTN_TAIL_GUARD_SABOTAGE, FUSED_CORNER, ATTN_REPAIR_MASKED_TAIL, ATTN_REPAIR_SAB_Z, ATTN_REPAIR_SAB_DQ, attention_estash_memory_grant
from transformer.checks.transformer_backward import (
    BWD_ANY_SABOTAGE, LlamaBackwardStages, llama_decoder_layer_backward_device,
)
from transformer.impl.llama.fused_attention import (
    ATTN_BWD_KV_CORNER_GUARD, ATTN_NO_BWD_CORNER, ATTN_STICKY,
)
from transformer.impl.llama.modeling_llama import (
    BLOCK_ANY_SABOTAGE, LlamaDims, LlamaDeviceWeights, LlamaDeviceStages,
    LlamaRopeTable, LlamaKVCache, llama_decoder_layer_forward,
    residual_next_norm_fusion_enabled,
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
    for j in range(config.n_tensors()):  # small-loop(n_tensors: parameter tensors, 9 per layer plus 2): name strings only, no data
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


comptime BYTE_LM_FAULT_INJECT = is_defined["MOJOLEARN_BYTE_LM_FAULT_INJECT"]()
"""DEVIATION 2514, step 7: the failed-update controls of design gate G4.
Compiled ONLY when the gate binary is built with
`-D MOJOLEARN_BYTE_LM_FAULT_INJECT=1`; a production build contains no
fault code and `byte_lm_fault_inject_available()` reports False. Under the
flag, `MOJOLEARN_BYTE_LM_FAULT=<name>` in the environment plants one bad
element at one named point of the step so that each refusal, and the
rollback behind it, can be exercised by name:

    loss_nonfinite   NaN into `ce_loss[0]` after L13, before the download
    grad_nonfinite   NaN into `grad[0]` after `pack_grads`
    opt_refuse       NaN into `m_state[5]` AFTER the shadow copy
    after_nonfinite  +inf into `v_state[3]` after the update kernel
    after_negative   -1.0 into `v_state[3]` after the update kernel

This is not a numerical sabotage arm: nothing here changes an arithmetic
path, it writes a value the profile must refuse. `_require_profile` does
not refuse the flag for that reason; the gate reads the availability
witness and skips the controls on a build without it."""


comptime BYTE_LM_STICKY_EAGER = is_defined["MOJOLEARN_BYTE_LM_STICKY_EAGER"]()

comptime BYTE_LM_RELEASE_EAGER = (
    is_defined["MOJOLEARN_BYTE_LM_RELEASE_EAGER"]() or byte_lm_release_eager_for[TARGET_COLUMN]()
) and not is_defined["MOJOLEARN_BYTE_LM_RETAIN_EAGER"]()

comptime BYTE_LM_CE_UNALIASED = is_defined["MOJOLEARN_BYTE_LM_CE_UNALIASED"]()
"""DEVIATION 3011: build the five `[M, V]` cross-entropy buffers as five
SEPARATE allocations, the way they were before the LM step memory work
aliased them.

THIS DEFINE EXISTS TO BE THE OTHER ARM. Aliasing is a storage decision and
the whole claim about it is that it moves no bit, so the claim is only worth
something if the unaliased spelling can still be BUILT and RUN beside it on
the same fixture. `[[mojotrees-switches-must-flip]]`: a flip nobody can flip
is prose. The shipped build defines nothing and gets the aliased buffers,
which are 3 * M * V * 4 bytes smaller."""


def byte_lm_ce_aliased() -> Bool:
    """False in a build carrying `-D MOJOLEARN_BYTE_LM_CE_UNALIASED=1`.

    An A/B that cannot tell its two arms apart is not an A/B: without this
    witness, building the same arm twice and comparing it with itself reads
    exactly like a passed identity gate."""
    comptime if BYTE_LM_CE_UNALIASED:
        return False
    return True


def byte_lm_attn_sticky_fallback() -> Bool:
    """DEVIATION 3110: TRUE only in a build carrying
    `-D MOJOLEARN_ATTN_STICKY=1`, which stops relaunching the fused attention
    kernels for a layer that has already refused. The default is FALSE and
    relaunches, because the latch measured 3.6% SLOWER.

    The A/B that claims the latch moves no bit reads this from INSIDE the
    process that loaded the binding. A `.so` digest cannot answer it: the
    define gates a single runtime branch on a comptime constant, so the two
    builds differ by about one byte and are the same size."""
    comptime if not ATTN_STICKY:
        return False
    return True


def byte_lm_attn_kv_corner_guard() -> Bool:
    """Whether the exact masked-tail predicate guards dk/dv negative zeros.

    Enabled by the qualified replay column or either explicit guard define.
    Read this inside the loaded binding to distinguish experimental arms.
    """
    comptime if ATTN_BWD_KV_CORNER_GUARD:
        return True
    return False


def byte_lm_attn_bwd_corner_refuses() -> Bool:
    """DEVIATION 3112: False in the MEASUREMENT build carrying
    `-D MOJOLEARN_ATTN_NO_BWD_CORNER=1`, whose backward does not refuse on a
    corner. That build is never shipped and its only use is the bit
    comparison against the refusing arm."""
    comptime if ATTN_NO_BWD_CORNER:
        return False
    return True


def byte_lm_fault_inject_available() -> Bool:
    """True only in a build compiled with the G4 fault-injection flag."""
    comptime if BYTE_LM_FAULT_INJECT:
        return True
    return False


def _fault_plant(ctx: DeviceContext, mut buf: DeviceBuffer[DType.float32],
                 idx: Int, bits: UInt32) raises:
    """Write ONE element, by bits, into a device buffer (one 4 B H2D copy
    through a pinned scalar). Reached only through `_maybe_fault`."""
    step_count_host_alloc()
    var h = ctx.enqueue_create_host_buffer[DType.float32](1)
    step_count_sync()
    ctx.synchronize()
    h.unsafe_ptr().unsafe_store(0, bitcast[DType.float32](bits))
    var view = buf.create_sub_buffer[DType.float32](idx, 1)
    step_count_h2d()
    ctx.enqueue_copy(dst_buf=view, src_ptr=h.unsafe_ptr())
    step_count_sync()
    ctx.synchronize()
    _ = view^
    _ = h^


def _maybe_fault(ctx: DeviceContext, mut buf: DeviceBuffer[DType.float32],
                 name: String, idx: Int, bits: UInt32) raises:
    """One injection site. Under the flag: plant `bits` at `buf[idx]` when
    `MOJOLEARN_BYTE_LM_FAULT` names this site. Without the flag the body is
    not compiled and the call is empty."""
    comptime if BYTE_LM_FAULT_INJECT:
        if String(getenv("MOJOLEARN_BYTE_LM_FAULT")) == name:
            _fault_plant(ctx, buf, idx, bits)


comptime _FAULT_NAN: UInt32 = 0x7FC00000
comptime _FAULT_INF: UInt32 = 0x7F800000
comptime _FAULT_MINUS_ONE: UInt32 = 0xBF800000


def _require_finite(values: List[Float32], name: String) raises:
    for i in range(len(values)):
        if (bitcast[DType.uint32](values[i]) & UInt32(0x7F800000)) == UInt32(0x7F800000):
            raise Error("byte LM: nonfinite " + name + " at " + String(i))


def _require_device_finite(ctx: DeviceContext, mut scan: DeviceScanScratch,
                           mut buf: DeviceBuffer[DType.float32], n: Int, name: String) raises:
    """`_require_finite` over a device buffer: one scan, no download, the
    SAME message with the index the scan returns (the smallest, as the host
    loop reported it)."""
    var bad = scan.first_nonfinite(ctx, buf, n)
    if bad >= 0:
        raise Error("byte LM: nonfinite " + name + " at " + String(bad))


def _byte_validate_state_shape(param: List[Float32], m: List[Float32], v: List[Float32],
                               flags: List[Bool], completed: Int, config: ByteConfig = ByteConfig()) raises:
    """The scalar half of `byte_validate_state`: the config, the four
    lengths and the step bound, no per-parameter walk. Lane box-run-2-fix
    (2026-10-05): ByteTrainer's open admits the host state with this before
    any allocation and runs the per-parameter predicates on the device after
    the upload (`validate_device_state`); it used to walk every parameter,
    moment and flag on the host (`byte_validate_state`) at open."""
    config.validate()
    var n_total = config.n_total()
    if len(param) != n_total or len(m) != n_total or len(v) != n_total or len(flags) != config.n_tensors():
        raise Error("byte LM: state length differs from canonical registry")
    if completed < 0 or completed >= 1000000:
        raise Error("byte LM: completed step must be in [0,1000000)")


def byte_validate_state(param: List[Float32], m: List[Float32], v: List[Float32],
                        flags: List[Bool], completed: Int, config: ByteConfig = ByteConfig()) raises:
    _byte_validate_state_shape(param, m, v, flags, completed, config)
    var n_total = config.n_total()
    _require_finite(param, "parameters")
    _require_finite(m, "first moments")
    _require_finite(v, "second moments")
    for i in range(n_total):
        if v[i] < Float32(0):
            raise Error("byte LM: negative second moment")


def byte_validate_device_state(ctx: DeviceContext, mut scan: DeviceScanScratch,
                               mut param: DeviceBuffer[DType.float32],
                               mut m: DeviceBuffer[DType.float32],
                               mut v: DeviceBuffer[DType.float32],
                               flags: List[Bool], completed: Int,
                               config: ByteConfig = ByteConfig()) raises:
    """`byte_validate_state` over the DEVICE state (DEVIATION 2514, design
    2.3): the same predicates, the same names, the same order, with no
    download. Lengths and the step bound are host scalars; finite
    parameters, first moments and second moments are three
    `device_first_nonfinite` scans raising the host message with the index
    the scan returns; `v[i] < 0` is one `device_first_negative` scan
    raising the host message, which carries no index. The negativity
    predicate is by bits and equals `v < 0` for every non-NaN float, and
    NaN was refused by the finite scan one line earlier, exactly as the
    host loops ordered it. Nothing here is a fold that feeds an output:
    each scan writes only its own integer partials."""
    config.validate()
    var n_total = config.n_total()
    if len(param) != n_total or len(m) != n_total or len(v) != n_total or len(flags) != config.n_tensors():
        raise Error("byte LM: state length differs from canonical registry")
    if completed < 0 or completed >= 1000000:
        raise Error("byte LM: completed step must be in [0,1000000)")
    _require_device_finite(ctx, scan, param, n_total, "parameters")
    _require_device_finite(ctx, scan, m, n_total, "first moments")
    _require_device_finite(ctx, scan, v, n_total, "second moments")
    if scan.first_negative(ctx, v, n_total) >= 0:
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
    # cpu3-seq: the per-token range refusal runs on the device after the
    # upload (`byte_require_tokens_device`, same message), before the
    # embedding gather reads the ids. Only the shape is checked here.


def byte_require_tokens_device(ctx: DeviceContext, mut ids_dev: DeviceBuffer[DType.int32],
                               mut targets_dev: DeviceBuffer[DType.int32],
                               config: ByteConfig) raises:
    """cpu3-seq: the token-range refusal of `byte_validate_tokens` on the
    uploaded inputs and targets (`[b, 0:L]` and `[b, 1:L+1]` cover every
    token of the row-major `[batch, length+1]` ids). Two device scans, one
    partials readback and one wait each; the first wait also completes the
    uploads queued before it on the same context."""
    var m = config.batch * config.length
    if device_first_token_oob(ctx, ids_dev, m, config.vocab_size) >= 0:
        raise Error("byte LM: token ID outside configured vocabulary")
    if device_first_token_oob(ctx, targets_dev, m, config.vocab_size) >= 0:
        raise Error("byte LM: token ID outside configured vocabulary")


def _require_profile() raises:
    # afn-lm (2026-10-03): a FAST build on Apple is admitted too; IDENTICAL
    # reads the same condition as before (BYTE_LM_FAST_APPLE is False there).
    comptime if GLOBAL_NUMERIC_MODE != NUMERIC_IDENTICAL and not BYTE_LM_FAST_APPLE:
        raise Error("byte LM: training requires IDENTICAL")
    comptime if (GEMM_SABOTAGE or GEMM_BWD_SABOTAGE or ANY_EMB_SABOTAGE
                 or ANY_LOSS_SABOTAGE or OPT_SABOTAGE or BWD_ANY_SABOTAGE
                 or BLOCK_ANY_SABOTAGE or ATTN_TAIL_GUARD_SABOTAGE or ATTN_REPAIR_SAB_Z or ATTN_REPAIR_SAB_DQ):
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
                            config.intermediate]
    if not config.chunked_lm_head_v2:
        widths.append(config.vocab_size)
    else:
        _byte_check_gemm(m, min(config.vocab_size, LM_HEAD_V2_CHUNK), config.d_model)
        _byte_check_workspace(chunked_lm_head_v2_gemm_workspace_floats(m,config.vocab_size,config.d_model))
    for width in widths:
        _byte_check_gemm(m, width, config.d_model)
    _byte_check_gemm(m, config.d_model, config.intermediate)
    _byte_check_gemm(config.length, config.length, config.head_dim)
    _byte_check_gemm(config.length, config.head_dim, config.length)
    _byte_check_gemm(1, config.d_model, m)
    if not config.chunked_lm_head_v2:
        _byte_check_workspace(identical_ce_workspace_max_floats(m, config.vocab_size, REDUCTION_MEAN))
        _byte_check_workspace(identical_ce_ones_floats(m, config.vocab_size))
    _byte_check_workspace(identical_optimizer_workspace_floats(config.offsets()))


struct ByteBuffers(Movable):
    """Configured buffers; flat arrays are authoritative, weights are copies."""
    var neural_scratch: List[NeuralLifetimeArena]
    var config: ByteConfig
    var n_total: Int
    var optimizer_first: Int
    var optimizer_count: Int
    var optimizer_pooled: Bool
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

    # DEVIATION 2514 (design 4.2): the shadow of `param`, `m_state`,
    # `v_state` and `buf_initialized` taken immediately before the in-place
    # update kernel, so a failure after it can be rolled back on the device.
    # `adam_update_kernel` updates in place; a second buffer set would be a
    # new kernel spelling in the optimizer lane's file, so this is a copy.
    var shadow_p: DeviceBuffer[DType.float32]
    var shadow_m: DeviceBuffer[DType.float32]
    var shadow_v: DeviceBuffer[DType.float32]
    var flags_before: List[Bool]

    def __init__(out self, ctx: DeviceContext, initial_params: List[Float32],
                 initial_m: List[Float32], initial_v: List[Float32], flags: List[Bool],
                 config: ByteConfig = ByteConfig(), optimizer_first: Int = 0,
                 optimizer_count: Int = -1) raises:
        _byte_validate_allocations(config)
        self.config = config.copy()
        var M = config.batch * config.length
        var DM = config.d_model
        var V = config.vocab_size

        self.offsets = byte_offsets(config)
        self.n_total = config.n_total()
        var n = self.n_total

        self.optimizer_pooled = optimizer_count >= 0
        self.optimizer_first = optimizer_first
        self.optimizer_count = n if optimizer_count < 0 else optimizer_count
        var owned = self.optimizer_count
        if optimizer_first < 0 or owned < 1 or optimizer_first > n - owned:
            raise Error("byte LM: invalid optimizer ownership range")
        self.param = _upload(ctx, initial_params)
        self.grad = _zeros(ctx, n)
        if self.optimizer_pooled:
            # cpu3-seq: the owned moment range goes up straight from the
            # caller's Lists (one DMA each), no host slice copy.
            if len(initial_m) < optimizer_first + owned or len(initial_v) < optimizer_first + owned:
                raise Error("byte LM: optimizer moments shorter than the owned range")
            self.m_state = ctx.enqueue_create_buffer[DType.float32](owned)
            self.v_state = ctx.enqueue_create_buffer[DType.float32](owned)
            step_count_h2d()
            ctx.enqueue_copy(dst_buf=self.m_state, src_ptr=initial_m.unsafe_ptr() + optimizer_first)
            step_count_h2d()
            ctx.enqueue_copy(dst_buf=self.v_state, src_ptr=initial_v.unsafe_ptr() + optimizer_first)
            step_count_sync()
            ctx.synchronize()
        else:
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
        self.neural_scratch = List[NeuralLifetimeArena]()
        var opt_cells = identical_optimizer_workspace_floats(self.offsets)
        var ce_cells = 1 if config.chunked_lm_head_v2 else identical_ce_workspace_max_floats(M, V, REDUCTION_MEAN)
        comptime if NN62_LIFETIME_ARENA:
            # CE workspace is dead when forward enqueues its final fold (phase
            # 0); clipping starts after all backward work (phase 2). Neither is
            # retained for replay. Both use the same ordered context, and every
            # scratch read follows its producer. The owner retains the slab
            # across forward, backward, update and rollback; no async alias
            # crosses contexts. Fresh steps repeat this fixed phase order.
            var ranges = List[NeuralLiveRange]()
            ranges.append(NeuralLiveRange(ce_cells, 0, 0))
            ranges.append(NeuralLiveRange(opt_cells, 2, 2))
            self.neural_scratch.append(NeuralLifetimeArena(ctx, ranges, 4 * max(ce_cells, opt_cells)))
            self.opt_ws = self.neural_scratch[0].view(1, 2)
        else:
            self.opt_ws = _zeros(ctx, opt_cells)
        self.sab_partials = _zeros(ctx, SAB_CHUNKS)

        # Views of the flat parameters (`_bind_emb_head`), not copies.
        comptime if BYTE_LM_EMB_HEAD_VIEWS:
            self.emb_w = self.param.create_sub_buffer[DType.float32](0, V * DM)
            self.lm_w = self.param.create_sub_buffer[DType.float32](
                self.offsets[config.n_tensors() - 1], V * DM
            )
        else:
            self.emb_w = _zeros(ctx, V * DM)
            self.lm_w = _zeros(ctx, V * DM)
        self.dw_emb = _zeros(ctx, V * DM)
        self.dw_lm = _zeros(ctx, V * DM)

        self.ids = _zeros_i32(ctx, M)
        self.targets = _zeros_i32(ctx, M)

        self.x = _zeros(ctx, M * DM)
        self.logits = _zeros(ctx, M * min(V, LM_HEAD_V2_CHUNK) if config.chunked_lm_head_v2 else M * V)
        self.d_h = _zeros(ctx, M * DM)

        self.ce_max = _zeros(ctx, M)
        # DEVIATION 3011: `ce_shift` IS `logits` and `ce_weights` and `ce_dlogits`
        # ARE `ce_expo`. Five `[M, V]` allocations become two, which is
        # 3 * M * V * 4 bytes -- 1,178 MiB at B1/L2048/V50257 and eight
        # times that at the batch 4 operating point
        # lane/lm-training-shakedown measured on 2026-09-17.
        #
        # A STORAGE DECISION WITH THE SAME OPERANDS, NOT A FOLD CHANGE. Each
        # of the three kernels involved owns one CELL and reads only that
        # cell of its input, so writing the result back over the input is
        # the same arithmetic on the same bytes in the same thread:
        #   `ce_shift_exp_kernel` loads `logits[cell]` into a register,
        #       subtracts `max[row]`, THEN stores `shift[cell]` and
        #       `expo[cell]` (loss.mojo:660-668);
        #   `ce_weights_kernel` divides `expo[cell]` by `denom[row]` and
        #       stores `weights[cell]` (:1020-1032);
        #   `ce_dlogits_kernel` subtracts a host constant from
        #       `weights[cell]`, divides, stores `dlogits[cell]` (:1034).
        # No kernel reads a neighbour cell, no kernel folds over `V` here
        # (L4's denom GEMM and L12's row fold read `expo` and `ce_row`,
        # both before L14), and the two backward kernels are enqueued back
        # to back on one in-order context, so `expo` is dead at L14 and
        # `weights` is dead at L16.
        #
        # WHAT IS DEAD WHEN. `logits` is read by the refusal scan (first
        # statement of the CE forward) and by L2/L3; after L3 the clean
        # path never reads it again -- L6/L7 read `shift[row*V + y]` and
        # `logdenom`, and their `logits`/`expo`/`denom` arguments feed the
        # SAB_NLL_* arms only. Those arms cannot reach a trainer:
        # `_require_profile` (:297) refuses a build carrying any of them by
        # name, so the brief's flagged level-2 risk is closed by a check
        # and not by a promise. The loss lane's own gate fixtures pass
        # five distinct buffers and are untouched by this.
        if config.chunked_lm_head_v2:
            self.ce_shift = _zeros(ctx, 1)
        else:
            comptime if BYTE_LM_CE_UNALIASED:
                self.ce_shift = _zeros(ctx, M * V)
            else:
                self.ce_shift = self.logits.create_sub_buffer[DType.float32](0, M * V)
        self.ce_expo = _zeros(ctx, 1 if config.chunked_lm_head_v2 else M * V)
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
        # DEVIATION 3011: both are `ce_expo` (see `ce_shift` above).
        if config.chunked_lm_head_v2:
            self.ce_weights = _zeros(ctx, 1)
            self.ce_dlogits = _zeros(ctx, 1)
        else:
            comptime if BYTE_LM_CE_UNALIASED:
                self.ce_weights = _zeros(ctx, M * V)
                self.ce_dlogits = _zeros(ctx, M * V)
            else:
                self.ce_weights = self.ce_expo.create_sub_buffer[DType.float32](0, M * V)
                self.ce_dlogits = self.ce_expo.create_sub_buffer[DType.float32](0, M * V)
        self.ce_ones = _ones(ctx, 1 if config.chunked_lm_head_v2 else identical_ce_ones_floats(M, V))
        comptime if NN62_LIFETIME_ARENA:
            self.ce_ws = self.neural_scratch[0].view(0, 0)
        else:
            self.ce_ws = _zeros(ctx, ce_cells)

        var head_scratch = (chunked_lm_head_v2_gemm_workspace_floats(M,V,DM) if config.chunked_lm_head_v2
            else identical_gemm_workspace_max_floats(M,V,DM))
        self.head_ws = _zeros(ctx, head_scratch)
        self.head_bwd_ws = _zeros(ctx, 1 if config.chunked_lm_head_v2 else identical_gemm_backward_workspace_max_floats(OP_NT, M, V, DM, False))

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
        self.shadow_p = _zeros(ctx, owned)
        self.shadow_m = _zeros(ctx, owned)
        self.shadow_v = _zeros(ctx, owned)
        self.flags_before = flags.copy()



def _param_upload(ctx: DeviceContext, values: List[Float32], offsets: List[Int], j: Int) raises -> DeviceBuffer[DType.float32]:
    """cpu3-seq: parameter tensor `j` uploaded straight from its offset in
    the flat parameter List (one DMA), no host slice copy. The caller waits
    before `values` may be released."""
    var lo = offsets[j]
    var n = offsets[j + 1] - lo
    if lo < 0 or n < 0 or lo + n > len(values):
        raise Error("byte LM: parameter tensor outside the flat parameter list")
    if n < 1:
        return _zeros(ctx, 1)
    var buf = ctx.enqueue_create_buffer[DType.float32](n)
    step_count_h2d()
    ctx.enqueue_copy(dst_buf=buf, src_ptr=values.unsafe_ptr() + lo)
    return buf^


def _block_weights(ctx: DeviceContext, params: List[Float32], block: Int, config: ByteConfig) raises -> LlamaDeviceWeights:
    var base = 1 + 9 * block
    var o = byte_offsets(config)
    var w = LlamaDeviceWeights(ctx, byte_dims(config), Float32(1e-6),
        _param_upload(ctx, params, o, base), _param_upload(ctx, params, o, base + 5),
        _param_upload(ctx, params, o, base + 1), _param_upload(ctx, params, o, base + 2),
        _param_upload(ctx, params, o, base + 3), _param_upload(ctx, params, o, base + 4),
        _param_upload(ctx, params, o, base + 6), _param_upload(ctx, params, o, base + 7),
        _param_upload(ctx, params, o, base + 8))
    step_count_sync()
    ctx.synchronize()
    return w^


def _param_dev_copy(ctx: DeviceContext, mut flat: DeviceBuffer[DType.float32], lo: Int, n: Int) raises -> DeviceBuffer[DType.float32]:
    """cpu3-seq: `flat[lo:lo+n]` copied device to device into its own buffer
    (no host slice). Enqueue only; the in-order context orders its readers."""
    if lo < 0 or n < 0 or lo + n > len(flat):
        raise Error("byte LM: parameter tensor outside the flat device parameters")
    if n < 1:
        return _zeros(ctx, 1)
    var buf = ctx.enqueue_create_buffer[DType.float32](n)
    var view = flat.create_sub_buffer[DType.float32](lo, n)
    ctx.enqueue_copy(dst_buf=buf, src_buf=view)
    _ = view^
    return buf^


def _block_weights_dev(ctx: DeviceContext, mut flat: DeviceBuffer[DType.float32], block: Int, config: ByteConfig) raises -> LlamaDeviceWeights:
    """`_block_weights` from the flat parameters already on the device: the
    nine tensors in the same constructor order, copied device to device."""
    var base = 1 + 9 * block
    var o = byte_offsets(config)
    return LlamaDeviceWeights(ctx, byte_dims(config), Float32(1e-6),
        _param_dev_copy(ctx, flat, o[base], o[base + 1] - o[base]),
        _param_dev_copy(ctx, flat, o[base + 5], o[base + 6] - o[base + 5]),
        _param_dev_copy(ctx, flat, o[base + 1], o[base + 2] - o[base + 1]),
        _param_dev_copy(ctx, flat, o[base + 2], o[base + 3] - o[base + 2]),
        _param_dev_copy(ctx, flat, o[base + 3], o[base + 4] - o[base + 3]),
        _param_dev_copy(ctx, flat, o[base + 4], o[base + 5] - o[base + 4]),
        _param_dev_copy(ctx, flat, o[base + 6], o[base + 7] - o[base + 6]),
        _param_dev_copy(ctx, flat, o[base + 7], o[base + 8] - o[base + 7]),
        _param_dev_copy(ctx, flat, o[base + 8], o[base + 9] - o[base + 8]))


def _block_offsets(o: List[Int], base: Int) raises -> List[Int]:
    """The ten flat-buffer offsets that bound one block's nine tensors."""
    var offs = List[Int]()
    for j in range(10):
        offs.append(o[base + j])
    return offs^


#: lane/apple-identical-neural (2026-09-26): the embedding and head weights
#: are VIEWS of the flat parameter buffer (`create_sub_buffer`), re-bound at
#: the top of every forward because the out-of-place optimizer swaps the
#: `param` handle with `shadow_p` each step. The two per-step copies they
#: replace moved 2 x V x d_model floats and nothing else: no float operation,
#: so the bits cannot move. What it buys is MEMORY: two V x d_model
#: allocations (309 MB at GPT-3 small) and their per-step traffic. It is NOT
#: a measured step-time win: the 430-530 ms first seen in `step.unpack_weights`
#: was the first submission after host-side exports paging the working set
#: back in (1.4 ms in consecutive steps), and consecutive-step A/B on the M4
#: shows no difference (1.686 vs 1.711 s, 1.969 vs 1.992 s).
#: Apple only for now: Apple's step glue keeps `param` in place (no swap);
#: NVIDIA's out-of-place arm swaps it every step, which the re-bind handles,
#: but that column has not run this yet.
#: `-D MOJOLEARN_BYTE_LM_COPY_EMB_HEAD` restores the copies.
# NN60: two independently selectable, default-OFF IDENTICAL view arms.
# Existing sub-buffer APIs supply the mechanism; no new vendor capability is
# assumed. These source drafts have no compile/identity/full-model A/B evidence.
# Rebind from the CURRENT owner each call, including after optimizer swaps.
# L11 (2026-10-07): NN60's arms are arms 1 and 2 of MOJOLEARN_IDN_LM_VIEWS
# and NN51 is arm 2 of MOJOLEARN_IDN_LM_RESIDENT_TOKENS
# (training/neural_identical_experiments.mojo).
comptime NN60_EMB_HEAD_VIEWS = (
    GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL
    and IDN_LM_VIEWS_ARM == 2
    and not is_defined["MOJOLEARN_IDN_ALL_OFF"]()
)
comptime NN51_RESIDENT_TOKEN_VALIDATION = (
    GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL
    and IDN_LM_RESIDENT_TOKENS_ARM == 2
    and not is_defined["MOJOLEARN_IDN_ALL_OFF"]()
)
comptime NN60_BLOCK_VIEWS = (
    GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL
    and IDN_LM_VIEWS_ARM == 1
    and not is_defined["MOJOLEARN_IDN_ALL_OFF"]()
)
comptime BYTE_LM_BLOCK_VIEWS = AFN_LM_PARAM_VIEWS or NN60_BLOCK_VIEWS or IDN_LM_PARAM_VIEWS
comptime BYTE_LM_EMB_HEAD_VIEWS = (
    (TARGET_COLUMN == COLUMN_APPLE or NN60_EMB_HEAD_VIEWS or IDN_LM_PARAM_VIEWS)
    and not is_defined["MOJOLEARN_BYTE_LM_COPY_EMB_HEAD"]()
)

# NI27 source candidate, not a promotion of the Apple FAST experiment.
# Only aliases change; every gradient writer keeps the same arithmetic.
# Both parameter and gradient views are rebound before each use, including
# after the transactional optimizer swaps param with shadow_p. Unverified.


def _bind_emb_head(ctx: DeviceContext, mut tb: ByteBuffers, config: ByteConfig) raises:
    """`emb_w` and `lm_w` read the CURRENT flat parameters: fresh views of
    `param` (after any handle swap), or the two copies under the revert arm."""
    var n = config.vocab_size * config.d_model
    comptime if BYTE_LM_EMB_HEAD_VIEWS:
        tb.emb_w = tb.param.create_sub_buffer[DType.float32](0, n)
        tb.lm_w = tb.param.create_sub_buffer[DType.float32](tb.offsets[config.n_tensors() - 1], n)
    else:
        _copy_into(ctx, tb.emb_w, tb.param, 0, 0, n)
        _copy_into(ctx, tb.lm_w, tb.param, 0, tb.offsets[config.n_tensors() - 1], n)


def _unpack_block(ctx: DeviceContext, mut tb: ByteBuffers, mut w: LlamaDeviceWeights, block: Int) raises:
    """Flat parameters -> the block's nine tensors, in ONE launch.

    Was nine `_copy_into` launches and one `ctx.synchronize()`. The nine
    became one through `byte_block_copy` (commit 1895b0287, which carries
    its own Metal receipt); the wait went because it enforced nothing.
    Every caller queues further work on
    the SAME in-order context and reads no host memory in between, and each
    one already waits after its loop over the blocks (`byte_lm.mojo`
    forward and gradient, `byte_lm_logits.mojo`). A wait that only delays
    the host is a Metal round trip for nothing.
    """
    var base = 1 + 9 * block
    comptime if BYTE_LM_BLOCK_VIEWS:
        # FAST Apple or opt-in NN60 IDENTICAL: the nine weights become
        # views of the CURRENT flat `param` (re-bound every call, as
        # `_bind_emb_head` does), so no launch and no separate allocation.
        ref vo = tb.offsets
        w.norm1_w = tb.param.create_sub_buffer[DType.float32](vo[base], vo[base + 1] - vo[base])
        w.w_q = tb.param.create_sub_buffer[DType.float32](vo[base + 1], vo[base + 2] - vo[base + 1])
        w.w_k = tb.param.create_sub_buffer[DType.float32](vo[base + 2], vo[base + 3] - vo[base + 2])
        w.w_v = tb.param.create_sub_buffer[DType.float32](vo[base + 3], vo[base + 4] - vo[base + 3])
        w.w_o = tb.param.create_sub_buffer[DType.float32](vo[base + 4], vo[base + 5] - vo[base + 4])
        w.norm2_w = tb.param.create_sub_buffer[DType.float32](vo[base + 5], vo[base + 6] - vo[base + 5])
        w.w_gate = tb.param.create_sub_buffer[DType.float32](vo[base + 6], vo[base + 7] - vo[base + 6])
        w.w_up = tb.param.create_sub_buffer[DType.float32](vo[base + 7], vo[base + 8] - vo[base + 7])
        w.w_down = tb.param.create_sub_buffer[DType.float32](vo[base + 8], vo[base + 9] - vo[base + 8])
        return
    var o = tb.offsets.copy()
    step_count_launch()
    byte_block_copy[False](ctx, tb.param,
        w.norm1_w, w.w_q, w.w_k, w.w_v, w.w_o,
        w.norm2_w, w.w_gate, w.w_up, w.w_down,
        _block_offsets(o, base))


def _pack_block(ctx: DeviceContext, mut tb: ByteBuffers, mut bst: LlamaBackwardStages, block: Int) raises:
    """The block's nine gradients -> the flat `grad`, in ONE launch.

    The mirror of `_unpack_block`, same order, same reasoning about the
    removed wait. `byte_lm_layer_pool_check.mojo` was the one caller that
    read the device right after without a wait of its own, and it now
    carries that wait at its own call site.
    """
    var base = 1 + 9 * block
    var o = tb.offsets.copy()
    step_count_launch()
    byte_block_copy[True](ctx, tb.grad,
        bst.dw_norm1, bst.dw_q, bst.dw_k, bst.dw_v, bst.dw_o,
        bst.dw_norm2, bst.dw_gate, bst.dw_up, bst.dw_down,
        _block_offsets(o, base))


def _afn_bind_grad_views(mut tb: ByteBuffers, mut bst: LlamaBackwardStages, block: Int) raises:
    """FAST Apple / NN60 IDENTICAL: the block's nine weight
    gradients become views of the flat `grad` at the offsets `_pack_block`
    copies them to. Every one is written whole, once, by the block backward
    (a GEMM or the norm weight GEMM), so the flat buffer ends holding what
    the pack would have copied, and the pack launch goes."""
    var base = 1 + 9 * block
    ref o = tb.offsets
    bst.dw_norm1 = tb.grad.create_sub_buffer[DType.float32](o[base], o[base + 1] - o[base])
    bst.dw_q = tb.grad.create_sub_buffer[DType.float32](o[base + 1], o[base + 2] - o[base + 1])
    bst.dw_k = tb.grad.create_sub_buffer[DType.float32](o[base + 2], o[base + 3] - o[base + 2])
    bst.dw_v = tb.grad.create_sub_buffer[DType.float32](o[base + 3], o[base + 4] - o[base + 3])
    bst.dw_o = tb.grad.create_sub_buffer[DType.float32](o[base + 4], o[base + 5] - o[base + 4])
    bst.dw_norm2 = tb.grad.create_sub_buffer[DType.float32](o[base + 5], o[base + 6] - o[base + 5])
    bst.dw_gate = tb.grad.create_sub_buffer[DType.float32](o[base + 6], o[base + 7] - o[base + 6])
    bst.dw_up = tb.grad.create_sub_buffer[DType.float32](o[base + 7], o[base + 8] - o[base + 7])
    bst.dw_down = tb.grad.create_sub_buffer[DType.float32](o[base + 8], o[base + 9] - o[base + 8])


struct ByteTrainer(Movable):
    """Owned device state. Use only with the same DeviceContext that created it.

    Do not mutate fields externally. On the stateless path (`byte_train_step`)
    a failed numerical step poisons this object; reconstruct from the last
    successful retained state before continuing. On the resident path
    (`byte_train_step_resident`, DEVIATION 2514) a failed step is rolled
    back on the device from the shadow taken before the update, and
    `healthy` stays False ONLY for a lost context (a rollback whose re-scan
    did not answer); such a session is unrecoverable and its steps since
    the last export are gone.

    `grad_step` is the step whose gradient `buffers.grad` holds, or -1
    after open, rollback and any failure; `byte_lm_session_export_gradients`
    refuses unless it equals `completed_steps`.
    """
    var config: ByteConfig
    var arena_id: Int
    var buffers: ByteBuffers
    var weights: List[LlamaDeviceWeights]
    var rope: LlamaRopeTable
    var prefill_cache: LlamaKVCache
    var forward: List[LlamaDeviceStages]
    var backward: List[LlamaBackwardStages]
    var optimizer: OptimizerConfig
    var completed_steps: Int
    var healthy: Bool
    var scan: DeviceScanScratch
    var shadow_valid: Bool
    var shadow_step: Int
    var released_eager_cells: Int
    var grad_step: Int
    # Diagnostic reach witness; counts successful live-status optimizer checks.
    var embedding_runs: List[OwnedEmbeddingRuns]
    var live_status_steps: Int

    def __init__(out self, ctx: DeviceContext, initial_params: List[Float32],
                 initial_m: List[Float32], initial_v: List[Float32],
                 flags: List[Bool], completed_steps: Int, optimizer: OptimizerConfig,
                 config: ByteConfig = ByteConfig(), optimizer_first: Int = 0,
                 optimizer_count: Int = -1) raises:
        # All supplied host state/configuration admitted before first allocation.
        _require_profile()
        _byte_validate_allocations(config)
        self.config = config.copy()
        # lane box-run-2-fix (2026-10-05): only the scalar checks here (config,
        # lengths, step bound); the finite / negative-moment predicates run on
        # the device after the upload (end of this body). Was
        # `byte_validate_state`, a host walk over every parameter and moment.
        _byte_validate_state_shape(initial_params, initial_m, initial_v, flags, completed_steps, config)
        byte_validate_optimizer(optimizer)
        self.optimizer = optimizer.copy()
        self.completed_steps = completed_steps
        self.healthy = True
        self.shadow_valid = False
        self.shadow_step = -1
        self.released_eager_cells = 0
        self.grad_step = -1
        self.embedding_runs = List[OwnedEmbeddingRuns]()
        self.live_status_steps = 0
        # lane/neural-pass43: the session's buffers carved from arena chunks
        # (core/device_arena.mojo) between here and `arena_end` below.
        self.arena_id = arena_begin()
        self.scan = DeviceScanScratch(ctx)
        self.buffers = ByteBuffers(ctx, initial_params, initial_m, initial_v, flags, config, optimizer_first, optimizer_count)
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
        arena_end(self.arena_id)
        step_count_sync()
        ctx.synchronize()
        # lane box-run-2-fix: the state's per-parameter predicates (finite
        # parameters, first and second moments, no negative second moment;
        # the host messages, same order) as device scans over the uploaded
        # buffers, before any step reads them. A pooled optimizer scans its
        # owned moment slice, the only one this session uploads.
        self.validate_device_state(ctx, completed_steps)
        # lane/neural-apple (2026-09-28): on a gated Apple build, the estash
        # attention word runs only when this trainer's kept stashes fit the
        # free device memory (a schedule choice; the bits are the same).
        _ = attention_estash_memory_grant(
            ctx, config.n_layers, config.batch, config.length, config.n_heads, config.length,
            2 * 4 * config.batch * config.length * config.vocab_size,
        )

    def validate_device_state(mut self, ctx: DeviceContext, completed: Int) raises:
        """`byte_validate_device_state` over this trainer's buffers with the
        given step (the binding calls it before an export; the step body
        calls it with the NEXT step after the update)."""
        if self.buffers.optimizer_pooled:
            var n = self.buffers.optimizer_count
            if completed < 0 or completed >= 1000000:
                raise Error("byte LM: completed step must be in [0,1000000)")
            if len(self.buffers.m_state) != n or len(self.buffers.v_state) != n:
                raise Error("byte LM: optimizer ownership length mismatch")
            _require_device_finite(ctx, self.scan, self.buffers.param, self.config.n_total(), "parameters")
            _require_device_finite(ctx, self.scan, self.buffers.m_state, n, "first moments")
            _require_device_finite(ctx, self.scan, self.buffers.v_state, n, "second moments")
            if self.scan.first_negative(ctx, self.buffers.v_state, n) >= 0:
                raise Error("byte LM: negative second moment")
            return
        byte_validate_device_state(ctx, self.scan, self.buffers.param, self.buffers.m_state,
            self.buffers.v_state, self.buffers.buf_initialized, completed, self.config)

    def __deinit__(deinit self):
        """The session's arena chunks go back to the pool (not freed: the
        views in this struct's fields die right after this body)."""
        try:
            arena_release(self.arena_id)
        except:
            pass


def byte_attention_eager_cells(tr: ByteTrainer) raises -> List[Int]:
    """DEVIATION 3010: THE QUADRATIC ATTENTION STAGES THIS TRAINER IS
    ACTUALLY HOLDING RIGHT NOW, so that a session which grew them can be
    told apart from one that never did WITHOUT a second run.

    Returns, in order:

        0  forward eager cells     sum over layers of len(scores) +
                                   len(masked) + len(weights) + len(sbh)
        1  forward aexp cells      sum over layers of len(aexp)
        2  backward eager cells    sum over layers of len(d_attn_weights) +
                                   len(d_attn_masked) + len(d_attn_scores) +
                                   len(d_qk_cell) + len(head_c)
        3  layers grown forward    layers whose len(scores) > 1
        4  layers grown backward   layers whose len(d_attn_weights) > 1
        5  layers with a full aexp layers whose len(aexp) > 1

    Then forward/backward stage-list lengths and one
    (forward status, backward status, materialized) triple per paired layer; exact-tail-guard flag; release flag; released eager cells;
    sticky-routing flag; one prefer-eager flag per layer; replay flag;
    one actual estash repair-site bitmask per layer (1 zdot, 2 dQ).
    The binding prepends its original four session fields. The two list
    lengths precede all variable-length fields; Python reads min(lengths).

    `aexp` IS REPORTED APART FROM THE OTHER THREE ON PURPOSE. Two
    different mechanisms grow it and they mean opposite things: the eager
    fallback grows all four together
    (`ensure_attention_stage_capacity`, modeling_llama.mojo), while a
    build carrying the DEVIATION 2652 exp stash grows `aexp` ALONE on a
    fused call that refused nothing. Summing them would read a healthy
    stash as a fallback.

    WHY THIS EXISTS. The LM step memory study found that these ten arrays are allocated at ONE element and
    grow on demand, that the growth is data dependent per layer and per
    step (`regime_product_ok`, `regime_finite`, or a `FUSED_CORNER` hit),
    and that under the legacy policy they never leave the session. So
    the device footprint of a long run is not a property of its shape: a
    session can double its device memory at some step nobody chose, and
    every capacity number taken in the first few steps is then wrong for
    the rest of the run. Nothing reported that growth, so a run that had
    it and a run that did not looked the same from outside.

    NO ARITHMETIC AND NO DEVICE WORK. Counts use `len()` of owned buffers;
    path fields are saved host metadata. Nothing is
    launched, downloaded or synchronized, and no step path calls this.
    It is read between steps.
    """
    var fwd_eager = 0
    var fwd_aexp = 0
    var bwd_eager = 0
    var grown_fwd = 0
    var grown_bwd = 0
    var grown_aexp = 0
    # DEVIATION 3110: BOUND BY THE LIST, NOT BY `n_layers`. The backward
    # loop `pop`s each layer's stages out of `tr.forward` and `tr.backward`
    # and reinserts them (`:1169-1170`), so a step that raises inside a
    # backward call leaves the lists SHORT. Reading `n_layers` entries out of
    # an 11-entry list then aborts the process with an out-of-bounds assert
    # instead of reporting anything, which is how the batch-4 arm of the
    # 2026-09-18 H100 leg died at step ~320: not an OOM, this. The counts are
    # reported so a short list is VISIBLE rather than fatal.
    var n_fwd = len(tr.forward)
    var n_bwd = len(tr.backward)
    var n = n_fwd if n_fwd < n_bwd else n_bwd
    for layer in range(n):  # small-loop(n: layers): per layer eager cell counts, no data
        fwd_eager += len(tr.forward[layer].scores)
        fwd_eager += len(tr.forward[layer].masked)
        fwd_eager += len(tr.forward[layer].weights)
        fwd_eager += len(tr.forward[layer].sbh)
        fwd_aexp += len(tr.forward[layer].aexp)
        bwd_eager += len(tr.backward[layer].d_attn_weights)
        bwd_eager += len(tr.backward[layer].d_attn_masked)
        bwd_eager += len(tr.backward[layer].d_attn_scores)
        bwd_eager += len(tr.backward[layer].d_qk_cell)
        bwd_eager += len(tr.backward[layer].head_c)
        if len(tr.forward[layer].scores) > 1:
            grown_fwd += 1
        if len(tr.backward[layer].d_attn_weights) > 1:
            grown_bwd += 1
        if len(tr.forward[layer].aexp) > 1:
            grown_aexp += 1
    var out = List[Int]()
    out.append(fwd_eager)
    out.append(fwd_aexp)
    out.append(bwd_eager)
    out.append(grown_fwd)
    out.append(grown_bwd)
    out.append(grown_aexp)
    out.append(n_fwd)
    out.append(n_bwd)
    # Triples in layer order: actual launch statuses and current materialization.
    for layer in range(n):  # small-loop(n: layers): per layer eager cell counts, no data
        out.append(tr.forward[layer].attn_forward_status)
        out.append(tr.backward[layer].attn_backward_status)
        out.append(Int(tr.forward[layer].attn_materialized))
    out.append(Int(ATTN_EXACT_TAIL_GUARD))
    out.append(Int(BYTE_LM_RELEASE_EAGER))
    out.append(tr.released_eager_cells)
    out.append(Int(BYTE_LM_STICKY_EAGER))
    for layer in range(n):  # small-loop(n: layers): per layer eager cell counts, no data
        out.append(Int(tr.forward[layer].attn_prefer_eager))
    out.append(Int(ATTN_REPAIR_MASKED_TAIL))
    for layer in range(n):  # small-loop(n: layers): per layer eager cell counts, no data
        out.append(tr.backward[layer].attn_repaired)
    return out^


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
    _byte_admit_tokens(ctx, trainer, token_ids)
    timing_tick(ctx, ton, tk, "step.validate_before")
    trainer.healthy = False
    trainer.shadow_valid = False
    # DEVIATION 2514 step 4: the ONE step body, shared with the resident
    # path. The stateless capture is the same body with the mirrors around
    # it; the optimizer does not write `grad` (no clipping in this profile),
    # so the pre-update gradient is still in `grad` after the update and is
    # downloaded here, byte for byte what the pre-2514 download returned.
    var loss = _byte_step_device(ctx, trainer, token_ids)
    if ton:
        tk = Int(perf_counter_ns())
    var grads = download_f32(ctx, trainer.buffers.grad, config.n_total())
    timing_tick(ctx, ton, tk, "step.mirror_download_grads")
    timing_bytes(ton, "step.mirror_download_grads_bytes", config.n_total() * 4)
    var after_p = download_f32(ctx, trainer.buffers.param, config.n_total())
    var after_m = download_f32(ctx, trainer.buffers.m_state, config.n_total())
    var after_v = download_f32(ctx, trainer.buffers.v_state, config.n_total())
    var after_flags = trainer.buffers.buf_initialized.copy()
    timing_tick(ctx, ton, tk, "step.mirror_download_after")
    timing_bytes(ton, "step.mirror_download_after_bytes", 3 * config.n_total() * 4)
    trainer.healthy = True
    var capture = ByteStepCapture(token_ids.copy(), before_p^, before_m^,
        before_v^, before_flags^, grads^, after_p^, after_m^,
        after_v^, after_flags^, loss, trainer.completed_steps, trainer.optimizer.copy(),
        config.profile(), "identical", String(COMPILED_VENDOR))
    # Host-only: the ids copy into the capture (the mirrors are moved).
    timing_tick(ctx, ton, tk, "step.capture_copy")
    timing_bytes(ton, "step.capture_copy_bytes", len(token_ids) * 4)
    return capture^


@fieldwise_init
struct ByteLeanResult(Movable):
    """The resident step's result (DEVIATION 2514, design 2.1): the loss,
    the completed step and the momentum flags, nothing else. The gradient
    and the state stay on the device until exported."""
    var loss: Float32
    var completed_steps: Int
    var flags: List[Bool]


def _byte_admit_tokens(ctx: DeviceContext, mut tr: ByteTrainer, ids: List[Int32]) raises:
    """cpu3-seq: the token admission of the non-resident step and eval,
    BEFORE they mark the trainer unhealthy (as the host walk did): the ids
    go up into the scratch input/target buffers and are range-checked on
    the device. Writes only those scratch buffers, which the forward
    rewrites with the same bytes."""
    var config = tr.config.copy()
    afn_upload_ids(ctx, tr.buffers.ids, tr.buffers.targets, ids, config.batch, config.length)
    byte_require_tokens_device(ctx, tr.buffers.ids, tr.buffers.targets, config)


def _byte_recover(ctx: DeviceContext, mut tr: ByteTrainer, message: String) raises:
    """The resident step's failure path (design 4.2 item 5). ALWAYS raises.

    After the shadow point: roll the device state back to the shadow and
    re-raise the step's own message; if the rollback's re-scan does not
    answer, the context is not answering and the message says the session
    is lost (`healthy` stays False). Before the shadow point: nothing was
    written (design 4.1, the first five signals), but a context-level
    failure is indistinguishable from a refusal by its message alone, so
    the state is re-scanned once to prove the context answers before the
    trainer is marked healthy again."""
    if tr.shadow_valid:
        try:
            _ = byte_rollback(ctx, tr)
        except lost:
            raise Error(String(lost) + " (step failure: " + message + ")")
        raise Error(message)
    try:
        tr.validate_device_state(ctx, tr.completed_steps)
    except lost:
        raise Error("byte LM: session lost: " + String(lost) + " (step failure: " + message + ")")
    tr.healthy = True
    raise Error(message)


def byte_train_step_resident(ctx: DeviceContext, mut trainer: ByteTrainer,
                             token_ids: List[Int32]) raises -> ByteLeanResult:
    """One step on device-resident state (DEVIATION 2514). No mirror is
    downloaded before or after: the state was validated when it was last
    written (admission scan before upload, the previous step's device
    `validate_after`, the rollback re-scan) and no write path exists
    between steps (design 2.3), so `validate_before` is dropped here and
    `validate_after` runs as device scans inside the shared body. A raise
    after the shadow point is rolled back before it propagates."""
    _require_profile()
    var config = trainer.config.copy()
    byte_validate_tokens(token_ids, config)
    byte_validate_optimizer(trainer.optimizer)
    if not trainer.healthy:
        raise Error("byte LM: session lost; restore a retained export")
    if trainer.completed_steps < 0 or trainer.completed_steps >= 999999:
        raise Error("byte LM: step bound reached")
    trainer.healthy = False
    trainer.shadow_valid = False
    trainer.grad_step = -1
    var loss = Float32(0)
    var failed = False
    var message = String("")
    try:
        loss = _byte_step_device(ctx, trainer, token_ids)
    except error:
        failed = True
        message = String(error)
    if failed:
        _byte_recover(ctx, trainer, message)
    trainer.healthy = True
    return ByteLeanResult(loss, trainer.completed_steps, trainer.buffers.buf_initialized.copy())


def byte_rollback(ctx: DeviceContext, mut tr: ByteTrainer) raises -> Bool:
    """Restore `param`, `m_state`, `v_state` and the flags from the shadow
    taken at this step's shadow point (design 4.2 item 4): three D2D
    copies (a copy, not a handle swap, so per-layer views stay valid), the
    flags from `flags_before`, `completed_steps` back to `shadow_step`,
    `grad_step` to -1, then `byte_validate_device_state` on the restored
    buffers so a dead context is detected HERE and not one step later.
    Returns False, touching nothing, when there is no shadow to restore
    (no step reached the shadow point, or it was already rolled back);
    True after a restore. Raises "byte LM: session lost ..." and leaves
    `healthy` False if the re-scan raises."""
    if tr.buffers.optimizer_pooled:
        raise Error("byte LM: pooled optimizer requires its group rollback")
    if not tr.shadow_valid:
        return False
    var config = tr.config.copy()
    var n = config.n_total()
    tr.shadow_valid = False
    tr.healthy = False
    _copy_into(ctx, tr.buffers.param, tr.buffers.shadow_p, 0, 0, n)
    _copy_into(ctx, tr.buffers.m_state, tr.buffers.shadow_m, 0, 0, n)
    _copy_into(ctx, tr.buffers.v_state, tr.buffers.shadow_v, 0, 0, n)
    step_count_sync()
    ctx.synchronize()
    tr.buffers.buf_initialized = tr.buffers.flags_before.copy()
    tr.completed_steps = tr.shadow_step
    tr.grad_step = -1
    try:
        tr.validate_device_state(ctx, tr.completed_steps)
    except lost:
        raise Error("byte LM: session lost: rollback re-scan failed: " + String(lost))
    tr.healthy = True
    return True


def _byte_forward_loss[deferred: Bool = False](ctx: DeviceContext, mut tr: ByteTrainer,
                       ids: List[Int32], mut trace: IdentityTrace) raises -> Float32:
    """Shared forward only. Authoritative params/m/v/flags/t are read-only.

    `deferred` (afn-lm NOSYNC, FAST + Apple only; False everywhere else):
    the ids go up from `ids` itself with no staging and no wait, and the
    loss stays on the device (returns 0; the step's one wait reads it).
    The caller keeps `ids` alive past that wait."""
    var config = tr.config.copy()
    var M = config.batch * config.length
    var ton = timing_on()
    var tk = Int(perf_counter_ns())
    # cpu3-seq: no host split or staging loop on any path. The rows of `ids`
    # go straight into the device input/target buffers (two DMA copies per
    # row, the same bytes the old host split produced), then the token-range
    # refusal runs on the device (`byte_require_tokens_device`, the message
    # `byte_validate_tokens` raised) before the embedding gather reads them;
    # its wait also completes the uploads, so `ids` may be released after.
    # Under AFN NOSYNC this adds the scan's wait to the step (the host walk
    # it replaces was data-sized CPU work on the GPU route).
    afn_upload_ids(ctx, tr.buffers.ids, tr.buffers.targets, ids, config.batch, config.length)
    byte_require_tokens_device(ctx, tr.buffers.ids, tr.buffers.targets, config)
    timing_tick(ctx, ton, tk, "step.upload_inputs")
    timing_bytes(ton, "step.upload_inputs_bytes", 2 * M * 4)
    for layer in range(config.n_layers):
        _unpack_block(ctx, tr.buffers, tr.weights[layer], layer)
    _bind_emb_head(ctx, tr.buffers, config)
    # No wait: the embedding forward below is the next thing queued on this
    # same in-order context and reads no host memory.
    # A host round trip costs about a dozen kernel launches on Metal.
    # Device-to-device: every parameter byte copied once (blocks, emb, head).
    timing_tick(ctx, ton, tk, "step.unpack_weights")
    var head_param_cells = 2 * config.vocab_size * config.d_model
    var unpack_cells = 0 if BYTE_LM_BLOCK_VIEWS else config.n_total() - head_param_cells
    if not BYTE_LM_EMB_HEAD_VIEWS:
        unpack_cells += head_param_cells
    timing_bytes(ton, "step.unpack_weights_bytes", unpack_cells * 4)
    var emb = EmbConfig.llama(config.vocab_size, config.d_model)
    var ce = CeConfig.causal_lm(config.vocab_size)
    comptime if NN51_RESIDENT_TOKEN_VALIDATION or IDN_LM_OWNED_TOKENS:
        # byte_require_tokens_device above completed the range admission on
        # these owned IDs. No mutation occurs between that check and gather.
        # Source draft only: all-vendor identity/full-step qualification pending.
        emb_refuse_shape(emb, M)
        identical_embedding_forward_prerefused_into(ctx, tr.buffers.x, tr.buffers.emb_w, tr.buffers.ids, M, emb)
    else:
        identical_embedding_forward_into(ctx, tr.buffers.x, tr.buffers.emb_w, tr.buffers.ids, M, emb)
    # No wait: the block forward loop below queues onto this same in-order context.
    # A host round trip costs about a dozen kernel launches on Metal.
    timing_tick(ctx, ton, tk, "step.embedding_forward")
    for layer in range(config.n_layers):
        # Move the current stages out while borrowing the preceding residual.
        # No extra activation copy; restore canonical layer order after the call.
        var stages = tr.forward.pop(layer)
        # After the pop, the NEXT block's stages are at index `layer`, not `layer + 1`.
        # The reference modeling_llama.py:402-412 visits independent decoder layers.
        # Training always starts a full prefill: s=0 makes kv_append_kernel
        # read only fresh K/V. Reuse storage, never another layer's history.
        # Backward reads the per-layer stages.k_cache/v_cache, not this scratch.
        tr.prefill_cache.s = 0
        var prefix = String("byte.block") + String(layer) + ".forward"
        var norm1_ready = layer > 0 and residual_next_norm_fusion_enabled(
            M, config.d_model, tr.weights[layer].opts.norm_kind, tr.weights[layer].opts.norm_bias
        )
        var fuse_next = layer + 1 < config.n_layers and residual_next_norm_fusion_enabled(
            M, config.d_model, tr.weights[layer + 1].opts.norm_kind,
            tr.weights[layer + 1].opts.norm_bias,
        )
        if layer == 0:
            if fuse_next:
                llama_decoder_layer_forward(ctx, stages, tr.prefill_cache, tr.rope, tr.weights[layer],
                    tr.buffers.x, config.batch, config.length, 0, trace, prefix,
                    norm1_ready=norm1_ready,
                    next_norm_sumsq=Optional(tr.forward[layer].norm1_sumsq.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]()),
                    next_norm_out=Optional(tr.forward[layer].norm1_out.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]()),
                    next_norm_weight=Optional(tr.weights[layer + 1].norm1_w.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]()),
                    next_norm_eps=Optional(tr.weights[layer + 1].eps),
                    retain_kv_cache=not IDN_TRAIN_NO_DECODE_CACHE)
            else:
                llama_decoder_layer_forward(ctx, stages, tr.prefill_cache, tr.rope, tr.weights[layer],
                    tr.buffers.x, config.batch, config.length, 0, trace, prefix,
                    norm1_ready=norm1_ready, retain_kv_cache=not IDN_TRAIN_NO_DECODE_CACHE)
        else:
            if fuse_next:
                llama_decoder_layer_forward(ctx, stages, tr.prefill_cache, tr.rope, tr.weights[layer],
                    tr.forward[layer - 1].residual2, config.batch, config.length, 0, trace, prefix,
                    norm1_ready=norm1_ready,
                    next_norm_sumsq=Optional(tr.forward[layer].norm1_sumsq.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]()),
                    next_norm_out=Optional(tr.forward[layer].norm1_out.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]()),
                    next_norm_weight=Optional(tr.weights[layer + 1].norm1_w.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]()),
                    next_norm_eps=Optional(tr.weights[layer + 1].eps),
                    retain_kv_cache=not IDN_TRAIN_NO_DECODE_CACHE)
            else:
                llama_decoder_layer_forward(ctx, stages, tr.prefill_cache, tr.rope, tr.weights[layer],
                    tr.forward[layer - 1].residual2, config.batch, config.length, 0, trace, prefix,
                    norm1_ready=norm1_ready, retain_kv_cache=not IDN_TRAIN_NO_DECODE_CACHE)
        if _byte_layer_sync():
            step_count_sync()
            ctx.synchronize()
        comptime if BYTE_LM_RELEASE_EAGER:
            tr.released_eager_cells += _byte_release_forward_scratch(ctx, stages)
        tr.forward.insert(layer, stages^)
    # The blocks print their own `block.*` / `attn.*` lines; this envelope
    # is the whole forward loop including the per-layer waits and the
    # stage pop/insert bookkeeping, so the block sum can be checked.
    timing_tick(ctx, ton, tk, "envelope.blocks_forward")
    # DEVIATION 2630: the head GEMM's call-kind line (core/step_phase.mojo),
    # compiled only under -D MOJOLEARN_STEP_PHASE_TIMERS=1.
    var pg = StepPhaseClock(ctx)
    if config.chunked_lm_head_v2:
        chunked_lm_head_v2_gemm_forward_into(
            ctx, tr.buffers.ce_loss, tr.buffers.ce_max, tr.buffers.ce_denom,
            tr.buffers.ce_row, tr.buffers.logits, tr.buffers.head_ws,
            tr.forward[config.n_layers - 1].residual2,
            tr.buffers.lm_w, tr.buffers.targets, M, config.vocab_size,
            config.d_model,
        )
    else:
        identical_gemm_into[ROLE=ROLE_HEAD](ctx, tr.buffers.logits, tr.forward[config.n_layers - 1].residual2,
            tr.buffers.lm_w, tr.buffers.head_ws, M, config.vocab_size, config.d_model, OP_NT)
    # No wait: the cross entropy forward below queues onto this same in-order context.
    # A host round trip costs about a dozen kernel launches on Metal.
    pg.tick(ctx, "gemm.head_fwd")
    timing_tick(ctx, ton, tk, "step.head_forward")
    # `identical_ce_forward_into` prints `step.ce_refuse_download` (its
    # first statement: the full logits download and host scan) and then
    # `step.ce_forward` (L1-L13, waited on under the switch only), both from
    # its own clock; this clock is re-read after the call's wait.
    comptime if AFN_LM_HEAD_FUSE:
        # afn-lm HEAD_FUSE (FAST + Apple only): softmax, CE, mean loss and
        # dlogits in one row kernel; the CE backward below is skipped.
        if not config.chunked_lm_head_v2:
            var afn_status = afn_status_view(tr.scan)
            afn_ce_fused(ctx, tr.buffers.ce_loss, tr.buffers.ce_dlogits,
                tr.buffers.logits, tr.buffers.targets, afn_status, M,
                config.vocab_size)
    if not config.chunked_lm_head_v2 and not AFN_LM_HEAD_FUSE:
        identical_ce_forward_into(ctx, tr.buffers.ce_max, tr.buffers.ce_shift,
            tr.buffers.ce_expo, tr.buffers.ce_denom, tr.buffers.ce_logdenom,
            tr.buffers.ce_logp_target, tr.buffers.ce_nll, tr.buffers.ce_logp,
            tr.buffers.ce_logp_sum, tr.buffers.ce_smooth, tr.buffers.ce_row,
            tr.buffers.ce_total, tr.buffers.ce_loss, tr.buffers.logits,
            tr.buffers.targets, tr.buffers.ce_ones, tr.buffers.ce_ws, M, M, ce, targets_prerefused=NN51_RESIDENT_TOKEN_VALIDATION)
    # No wait: `download_f32` below waits for the loss itself.
    # A host round trip costs about a dozen kernel launches on Metal.
    comptime if AFN_LM_NOSYNC and deferred:
        _maybe_fault(ctx, tr.buffers.ce_loss, "loss_nonfinite", 0, _FAULT_NAN)
        return Float32(0.0)
    if ton:
        tk = Int(perf_counter_ns())
    _maybe_fault(ctx, tr.buffers.ce_loss, "loss_nonfinite", 0, _FAULT_NAN)
    var losses = download_f32(ctx, tr.buffers.ce_loss, 1)
    _require_finite(losses, "loss")
    timing_tick(ctx, ton, tk, "step.loss_download")
    timing_bytes(ton, "step.loss_download_bytes", 4)
    return losses[0]


def _byte_step_device(ctx: DeviceContext, mut tr: ByteTrainer,
                      ids: List[Int32]) raises -> Float32:
    """Single-device composition; the numerical operation order is unchanged."""
    comptime if AFN_LM_NOSYNC:
        return _afn_byte_step_nosync(ctx, tr, ids)
    var loss = byte_gradient_device(ctx, tr, ids)
    byte_update_device(ctx, tr)
    return loss


def _afn_byte_step_nosync(ctx: DeviceContext, mut tr: ByteTrainer,
                          ids: List[Int32]) raises -> Float32:
    """afn-lm NOSYNC (FAST + Apple only): the resident step with ONE host
    wait. The gradient half runs deferred (no staged upload, no loss
    download); the update is the shipped shadow copy and the shipped Adam
    kernel with the shipped scalars, minus the entry scans and the wait;
    every check of the synchronous step (loss, gradients, then
    parameters, moments and v >= 0 after the update) is one status launch
    read back with the loss at the one wait. A failing check raises AFTER
    the shadow point, so `_byte_recover` rolls the update back: the state
    a failure leaves is the synchronous step's. `ids` is borrowed for the
    whole call, past the wait, so its upload never reads freed memory."""
    if tr.buffers.optimizer_pooled:
        raise Error("byte LM: pooled optimizer requires its group update")
    if tr.optimizer.kind == OPT_SGD or tr.optimizer.max_norm > Float32(0.0):
        raise Error("byte LM: NOSYNC update takes Adam or AdamW without clipping")
    var config = tr.config.copy()
    var n = config.n_total()
    var next_step = tr.completed_steps + 1
    if next_step >= 1000000:
        raise Error("byte LM: completed step must be in [0,1000000)")
    var ton = timing_on()
    var tk = Int(perf_counter_ns())
    var status = afn_status_view(tr.scan)
    afn_reset(ctx, status, tr.buffers.ce_loss, True, False)
    _ = byte_gradient_device[True](ctx, tr, ids)
    _maybe_fault(ctx, tr.buffers.grad, "grad_nonfinite", 0, _FAULT_NAN)
    _copy_into(ctx, tr.buffers.shadow_p, tr.buffers.param, 0, 0, n)
    _copy_into(ctx, tr.buffers.shadow_m, tr.buffers.m_state, 0, 0, n)
    _copy_into(ctx, tr.buffers.shadow_v, tr.buffers.v_state, 0, 0, n)
    tr.buffers.flags_before = tr.buffers.buf_initialized.copy()
    tr.shadow_step = tr.completed_steps
    tr.shadow_valid = True
    _maybe_fault(ctx, tr.buffers.m_state, "opt_refuse", 5, _FAULT_NAN)
    ref cfg = tr.optimizer
    var sc = device_step_scalars(cfg, next_step)
    var is_adamw = Int32(1 if cfg.kind == OPT_ADAMW else 0)
    step_count_launch()
    ctx.enqueue_function[adam_update_kernel](
        tr.buffers.param.unsafe_ptr(),
        tr.buffers.grad.unsafe_ptr(),
        tr.buffers.m_state.unsafe_ptr(),
        tr.buffers.v_state.unsafe_ptr(),
        tr.buffers.denom_out.unsafe_ptr(),
        tr.buffers.q_out.unsafe_ptr(),
        Int32(n),
        is_adamw,
        cfg.beta1,
        cfg.beta2,
        cfg.eps,
        cfg.weight_decay,
        sc.c1,
        sc.c2,
        sc.step_size,
        sc.rt_bc2,
        sc.decay_mul,
        cfg.lr,
        sc.bc1,
        sc.bc2,
        grid_dim=(_opt_grid_for(n), 1, 1),
        block_dim=(OPT_TPB, 1, 1),
    )
    _maybe_fault(ctx, tr.buffers.v_state, "after_nonfinite", 3, _FAULT_INF)
    _maybe_fault(ctx, tr.buffers.v_state, "after_negative", 3, _FAULT_MINUS_ONE)
    var loss = afn_step_finish(ctx, tr.scan, tr.buffers.ce_loss, tr.buffers.grad,
        tr.buffers.param, tr.buffers.m_state, tr.buffers.v_state, n)
    timing_tick(ctx, ton, tk, "step.afn_nosync_step")
    _ = status^
    _ = ids
    tr.completed_steps = next_step
    tr.grad_step = next_step
    return loss


def _byte_layer_sync() -> Bool:
    """lane/neural-net-experiment (2026-09-30): whether the step waits on
    the host after EVERY layer's forward and backward (sixteen waits a step
    at the board's depth, before the fused attention's own regime reads).
    `MOJOLEARN_BYTE_LM_LAYER_SYNC=0` skips them: every launch sits on the
    one in-order context, the next layer's kernels are enqueued behind this
    layer's, and the eager-scratch release only drops buffer handles whose
    frees are themselves enqueued (DEVIATION 2520), so no bit moves; what
    moves is the host idling per layer. Default ON on NVIDIA and AMD (the
    measured behaviour; the L40S toggle sweep read 1.00 either way) and OFF
    on the Apple column (lane/neural-pass43, 2026-10-01: a Metal wait with
    a readback costs about 0.2 ms on the M4 and more on the M3 Ultra;
    `MOJOLEARN_BYTE_LM_LAYER_SYNC=1` restores them); the step's `step.*`
    timing ticks read the same either way."""
    var v = String(getenv("MOJOLEARN_BYTE_LM_LAYER_SYNC"))
    comptime if TARGET_COLUMN == COLUMN_APPLE:
        return v == "1"
    return v != "0"


def _byte_release_forward_scratch(ctx: DeviceContext,
                                  mut fwd: LlamaDeviceStages) raises -> Int:
    """After forward completion/trace: backward reads weights, not these
    score/mask/gather buffers. Keep weights and aexp valid until backward.
    This bounds the eager forward peak when several layers prefer eager."""
    var released = 0
    if len(fwd.scores) > 1:
        released += len(fwd.scores) - 1
        step_count_device_alloc()
        fwd.scores = ctx.enqueue_create_buffer[DType.float32](1)
    if len(fwd.masked) > 1:
        released += len(fwd.masked) - 1
        step_count_device_alloc()
        fwd.masked = ctx.enqueue_create_buffer[DType.float32](1)
    if len(fwd.sbh) > 1:
        released += len(fwd.sbh) - 1
        step_count_device_alloc()
        fwd.sbh = ctx.enqueue_create_buffer[DType.float32](1)
    return released


def _byte_release_eager(ctx: DeviceContext, mut fwd: LlamaDeviceStages,
                        mut bwd: LlamaBackwardStages) raises -> Int:
    """Called only AFTER the layer backward completion fence. These nine
    arrays are dead until the next eager call's ensure-capacity writes them.
    Keep aexp (the independent forward exp stash) and every dw/dx tensor.
    Return released cells, excluding the replacement one-cell placeholders.
    This is buffer lifetime only, with no kernel or arithmetic changes."""
    var released = 0
    if len(fwd.scores) > 1:
        released += len(fwd.scores) - 1
        step_count_device_alloc()
        fwd.scores = ctx.enqueue_create_buffer[DType.float32](1)
    if len(fwd.masked) > 1:
        released += len(fwd.masked) - 1
        step_count_device_alloc()
        fwd.masked = ctx.enqueue_create_buffer[DType.float32](1)
    if len(fwd.weights) > 1:
        released += len(fwd.weights) - 1
        step_count_device_alloc()
        fwd.weights = ctx.enqueue_create_buffer[DType.float32](1)
    if len(fwd.sbh) > 1:
        released += len(fwd.sbh) - 1
        step_count_device_alloc()
        fwd.sbh = ctx.enqueue_create_buffer[DType.float32](1)
    if len(bwd.d_attn_weights) > 1:
        released += len(bwd.d_attn_weights) - 1
        step_count_device_alloc()
        bwd.d_attn_weights = ctx.enqueue_create_buffer[DType.float32](1)
    if len(bwd.d_attn_masked) > 1:
        released += len(bwd.d_attn_masked) - 1
        step_count_device_alloc()
        bwd.d_attn_masked = ctx.enqueue_create_buffer[DType.float32](1)
    if len(bwd.d_attn_scores) > 1:
        released += len(bwd.d_attn_scores) - 1
        step_count_device_alloc()
        bwd.d_attn_scores = ctx.enqueue_create_buffer[DType.float32](1)
    if len(bwd.d_qk_cell) > 1:
        released += len(bwd.d_qk_cell) - 1
        step_count_device_alloc()
        bwd.d_qk_cell = ctx.enqueue_create_buffer[DType.float32](1)
    if len(bwd.head_c) > 1:
        released += len(bwd.head_c) - 1
        step_count_device_alloc()
        bwd.head_c = ctx.enqueue_create_buffer[DType.float32](1)
    if released > 0:
        fwd.attn_materialized = False
        fwd.attn_estash_cells = 0
    return released


def byte_gradient_device[deferred: Bool = False](ctx: DeviceContext, mut tr: ByteTrainer,
                         ids: List[Int32]) raises -> Float32:
    """Internal gradient half. Caller owns admission and transaction recovery."""
    tr.released_eager_cells = 0
    var config = tr.config.copy()
    var M = config.batch * config.length
    var emb = EmbConfig.llama(config.vocab_size, config.d_model)
    var ce = CeConfig.causal_lm(config.vocab_size)
    var trace = IdentityTrace.disabled()
    var loss = _byte_forward_loss[deferred](ctx, tr, ids, trace)
    # DEVIATION 2499 step-phase timers; see byte_train_step. The forward's
    # own clock ended at `step.loss_download`; this one starts here.
    var ton = timing_on()
    var tk = Int(perf_counter_ns())
    if config.chunked_lm_head_v2:
        chunked_lm_head_v2_gemm_backward_into(
            ctx, tr.buffers.d_h, tr.buffers.dw_lm, tr.buffers.logits,
            tr.buffers.head_ws,
            tr.forward[config.n_layers - 1].residual2, tr.buffers.lm_w,
            tr.buffers.targets, tr.buffers.ce_max, tr.buffers.ce_denom,
            M, config.vocab_size, config.d_model,
        )
    elif not AFN_LM_HEAD_FUSE:
        identical_ce_backward_into(ctx, tr.buffers.ce_weights, tr.buffers.ce_dlogits,
            tr.buffers.ce_expo, tr.buffers.ce_denom, tr.buffers.ce_logp,
            tr.buffers.targets, M, M, ce)
    # No wait: the two head backward GEMMs below queue onto this same in-order context.
    # A host round trip costs about a dozen kernel launches on Metal.
    timing_tick(ctx, ton, tk, "step.ce_backward")
    # DEVIATION 2630: the head GEMMs' call-kind lines (core/step_phase.mojo),
    # compiled only under -D MOJOLEARN_STEP_PHASE_TIMERS=1. Its dA tick
    # waits only there, like the `step.head_backward_da` tick below.
    var pg = StepPhaseClock(ctx)
    if not config.chunked_lm_head_v2:
        identical_gemm_backward_a_into[ROLE_HEAD](ctx, tr.buffers.d_h, tr.buffers.ce_dlogits,
            tr.buffers.lm_w, tr.buffers.head_bwd_ws, M, config.vocab_size, config.d_model, OP_NT)
    pg.tick(ctx, "gemm.head_dA")
    # dA and dB share one wait below; the tick between them waits ONLY
    # under the switch (a timed step is not a sample, and the two GEMMs are
    # queued on one in-order context either way).
    timing_tick(ctx, ton, tk, "step.head_backward_da")
    if not config.chunked_lm_head_v2:
        identical_gemm_backward_b_into(ctx, tr.buffers.dw_lm, tr.buffers.ce_dlogits,
            tr.forward[config.n_layers - 1].residual2, tr.buffers.head_bwd_ws, M, config.vocab_size, config.d_model, OP_NT)
    # No wait: the block backward loop below queues onto this same in-order context.
    # A host round trip costs about a dozen kernel launches on Metal.
    pg.tick(ctx, "gemm.head_dB")
    timing_tick(ctx, ton, tk, "step.head_backward_db")
    comptime if BYTE_LM_BLOCK_VIEWS:
        # The weight gradients land in the current flat `grad` owner.
        for layer in range(config.n_layers):
            _afn_bind_grad_views(tr.buffers, tr.backward[layer], layer)
    # Keep inter-layer cotangents on device, as the reference tensor graph
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
        if _byte_layer_sync():
            step_count_sync()
            ctx.synchronize()
        comptime if BYTE_LM_STICKY_EAGER:
            # A policy decision from an observed refusal, not a prediction of
            # a numerical corner. Both directions use the existing eager
            # kernels BEFORE attempting fused on subsequent auto calls.
            if stages.attn_forward_status == FUSED_CORNER or backward.attn_backward_status == FUSED_CORNER:
                stages.attn_prefer_eager = True
        comptime if BYTE_LM_RELEASE_EAGER:
            tr.released_eager_cells += _byte_release_eager(ctx, stages, backward)
        tr.backward.insert(layer, backward^)
        tr.forward.insert(layer, stages^)
    # Envelope of the whole backward loop (the blocks print `bwd.*`).
    timing_tick(ctx, ton, tk, "envelope.blocks_backward")
    # PLAN_SCAN performs 2 * vocab * positions integer probes before the
    # identical row fold; PLAN_SORT is a total-key sort over the positions
    # alone and builds the same ascending-position runs (contract 6 clause (d):
    # bit-identical on every vendor and the host column, so the pick moves no
    # bit). The scan cost grows with V * M, so the pick keys on V * M with the
    # embedding lane's own EMB_AUTO_SORT_MIN_CELLS (2^26), not on a token
    # count (the old `M >= 16384` sat one doubling below the one measured
    # GPT-3 row and ignored V). Tier-independent here, as before. Needs
    # neighbor-shape validation (V 256 and 50,257; M 4,096 to 65,536).
    var emb_plan = (
        PLAN_SORT if config.vocab_size * M >= EMB_AUTO_SORT_MIN_CELLS
        else PLAN_SCAN
    )
    comptime if NN50_OWNED_RUNS:
        var reuse = False
        if len(tr.embedding_runs) == 1:
            reuse = tr.embedding_runs[0].matches(ctx, tr.buffers.ids, M, emb)
        if not reuse:
            if len(tr.embedding_runs) == 1:
                _ = tr.embedding_runs.pop()
            tr.embedding_runs.append(OwnedEmbeddingRuns(ctx, tr.buffers.ids, M, emb))
        tr.embedding_runs[0].backward_into(ctx, tr.buffers.dw_emb, tr.backward[0].d_x, emb)
    elif NN51_RESIDENT_TOKEN_VALIDATION or IDN_LM_OWNED_TOKENS:
        # The immediately preceding forward in this gradient operation owned
        # and validated the same IDs. This skips a redundant ID download only.
        identical_embedding_backward_prerefused_into(ctx, tr.buffers.dw_emb, tr.backward[0].d_x,
            tr.buffers.ids, tr.buffers.emb_counts, tr.buffers.emb_run_begin,
            tr.buffers.emb_perm, M, emb, emb_plan)
    else:
        identical_embedding_backward_into(ctx, tr.buffers.dw_emb, tr.backward[0].d_x,
            tr.buffers.ids, tr.buffers.emb_counts, tr.buffers.emb_run_begin,
            tr.buffers.emb_perm, M, emb, emb_plan)
    # No wait: the pack loop below queues onto this same in-order context.
    # A host round trip costs about a dozen kernel launches on Metal.
    timing_tick(ctx, ton, tk, "step.embedding_backward")
    for layer in range(0 if BYTE_LM_BLOCK_VIEWS else config.n_layers):
        _pack_block(ctx, tr.buffers, tr.backward[layer], layer)
    _copy_into(ctx, tr.buffers.grad, tr.buffers.dw_emb, 0, 0, config.vocab_size * config.d_model)
    _copy_into(ctx, tr.buffers.grad, tr.buffers.dw_lm, tr.buffers.offsets[config.n_tensors() - 1], 0, config.vocab_size * config.d_model)
    # No wait: the gradient scan in `byte_update_device` waits for its own answer.
    # A host round trip costs about a dozen kernel launches on Metal.
    # Device-to-device: every gradient byte copied once into `grad`.
    timing_tick(ctx, ton, tk, "step.pack_grads")
    var packed_cells = 2 * config.vocab_size * config.d_model if BYTE_LM_BLOCK_VIEWS else config.n_total()
    timing_bytes(ton, "step.pack_grads_bytes", packed_cells * 4)
    _ = trace
    return loss


def byte_update_device(ctx: DeviceContext, mut tr: ByteTrainer) raises:
    """Internal update half, including gradient scan, shadow and validation."""
    if tr.buffers.optimizer_pooled:
        raise Error("byte LM: pooled optimizer requires its group update")
    var config = tr.config.copy()
    var ton = timing_on()
    var tk = Int(perf_counter_ns())
    var n = config.n_total()
    _maybe_fault(ctx, tr.buffers.grad, "grad_nonfinite", 0, _FAULT_NAN)
    # The pre-update gradient scan, on the device (was `_require_finite`
    # over a downloaded mirror): same message, same first index.
    _require_device_finite(ctx, tr.scan, tr.buffers.grad, n, "gradients")
    timing_tick(ctx, ton, tk, "step.validate_grads_scan")
    timing_bytes(ton, "step.validate_grads_scan_bytes", n * 4)
    # Host arithmetic on a field nothing below writes before the update, so
    # computing it here (it was computed after the shadow copy) is the same
    # value on both paths.
    var next_step = tr.completed_steps + 1
    # DEVIATIONS 2646 and 2647: a trial build under an arm carrying
    # `optskip` or `noshadow`, or (DEVIATION 2649) a shipped build whose
    # column default carries one, takes `_byte_glue_update` INSTEAD of the
    # shadow copy and `identical_optimizer_step`. On every other build
    # `glue_update` is the constant False and the block below is the shipped
    # path, unchanged.
    var glue_update = False
    var live_validated = False
    comptime if NN56_GROUPED_ADAM and not BYTE_LM_FAULT_INJECT:
        glue_update = True
        live_validated = _byte_neural_group_update(ctx, tr, next_step)
    elif STEP_GLUE_TRIAL or STEP_GLUE_SHIPPED_UPDATE:
        var glue_arm = step_glue_arm_from_env()
        if (glue_arm & STEP_GLUE_UPDATE_BITS) != 0:
            glue_update = True
            live_validated = _byte_glue_update(ctx, tr, next_step, glue_arm)
    if not glue_update:
        # THE SHADOW POINT (design 4.2 item 2): everything before this line
        # left param/m/v untouched; the update kernel below writes them in
        # place, so their bytes and the flags are shadowed here first.
        _copy_into(ctx, tr.buffers.shadow_p, tr.buffers.param, 0, 0, n)
        _copy_into(ctx, tr.buffers.shadow_m, tr.buffers.m_state, 0, 0, n)
        _copy_into(ctx, tr.buffers.shadow_v, tr.buffers.v_state, 0, 0, n)
        # No wait: `identical_optimizer_step` below queues onto this same in-order context.
        # A host round trip costs about a dozen kernel launches on Metal.
        tr.buffers.flags_before = tr.buffers.buf_initialized.copy()
        tr.shadow_step = tr.completed_steps
        tr.shadow_valid = True
        timing_tick(ctx, ton, tk, "step.shadow_copy")
        timing_bytes(ton, "step.shadow_copy_bytes", 3 * n * 4)
        _maybe_fault(ctx, tr.buffers.m_state, "opt_refuse", 5, _FAULT_NAN)
        # `identical_optimizer_step` prints `step.opt_refuse_scan` (its first
        # statement: param, grad, m, v scanned on the device) and
        # `step.optimizer` (clip, scalars, update; it waits before it
        # returns), both from its own clock; this clock is re-read after it.
        identical_optimizer_step(ctx, tr.buffers.param, tr.buffers.grad,
            tr.buffers.m_state, tr.buffers.v_state, tr.buffers.denom_out,
            tr.buffers.q_out, tr.buffers.sumsq, tr.buffers.norms,
            tr.buffers.total_cell, tr.buffers.out2, tr.buffers.opt_ws,
            tr.buffers.sab_partials, tr.buffers.buf_initialized, tr.buffers.offsets,
            tr.optimizer, next_step)
    if ton:
        tk = Int(perf_counter_ns())
    _maybe_fault(ctx, tr.buffers.v_state, "after_nonfinite", 3, _FAULT_INF)
    _maybe_fault(ctx, tr.buffers.v_state, "after_negative", 3, _FAULT_MINUS_ONE)
    # validate_after as device scans (design 2.3): finite param, m, v and
    # `v >= 0`, by name, with no download.
    if not live_validated:
        tr.validate_device_state(ctx, next_step)
    timing_tick(ctx, ton, tk, "step.validate_after_scan")
    timing_bytes(ton, "step.validate_after_scan_bytes", 16 if live_validated else 4 * n * 4)
    tr.completed_steps = next_step
    tr.grad_step = next_step


def byte_glue_update_launch(
    ctx: DeviceContext,
    mut param: DeviceBuffer[DType.float32],
    mut grad: DeviceBuffer[DType.float32],
    mut m_state: DeviceBuffer[DType.float32],
    mut v_state: DeviceBuffer[DType.float32],
    mut p_out: DeviceBuffer[DType.float32],
    mut m_out: DeviceBuffer[DType.float32],
    mut v_out: DeviceBuffer[DType.float32],
    mut denom_out: DeviceBuffer[DType.float32],
    mut q_out: DeviceBuffer[DType.float32],
    n: Int,
    cfg: OptimizerConfig,
    t: Int,
    out_of_place: Bool,
) raises:
    """DEVIATIONS 2646 and 2647 (reached from `_byte_glue_update`, and from
    `training/checks/step_glue_check.mojo` under
    -D MOJOLEARN_STEP_GLUE_TRIAL=1; since DEVIATION 2649 also from a shipped
    build whose column default carries an update bit, which on NVIDIA it
    does): the Adam update of
    `identical_optimizer_step` WITHOUT its entry scans and without a clip,
    one launch over `n` elements, then a wait.

    `out_of_place` False launches the SHIPPED `adam_update_kernel` with the
    arguments `identical_optimizer_step` passes it (the same scalars from
    `device_step_scalars`, the same `_grid_for` geometry at `OPT_TPB`), in
    place; `p_out`, `m_out` and `v_out` are not touched. True launches
    `adam_update_oop_kernel`, which reads `param`, `grad`, `m_state`,
    `v_state` and writes `p_out`, `m_out`, `v_out` (brief section 4.3), at
    `OPT_TPB` threads per block over `step_glue_blocks(n, OPT_TPB)` blocks
    (the ceiling; the floor only under the reach sabotage).

    Refused, because this path does not carry them: SGD (a different kernel
    with per-tensor flags) and clipping (`max_norm > 0`). The byte LM admits
    neither (`byte_validate_optimizer`). Recorded intermediates are refused
    at compile time on the trial build."""
    comptime if STEP_GLUE_TRIAL or STEP_GLUE_SHIPPED_UPDATE:
        comptime assert not OPT_RECORD_INTERMEDIATES, (
            "the step glue update path does not record denom/q; build the"
            " glue trial without MOJOLEARN_OPT_RECORD"
        )
    if cfg.kind == OPT_SGD:
        raise Error("byte LM glue update: Adam or AdamW only")
    if cfg.max_norm > Float32(0.0):
        raise Error("byte LM glue update: gradient clipping is not on this path")
    if n <= 0:
        return
    var sc = device_step_scalars(cfg, t)
    var is_adamw = Int32(0)
    if cfg.kind == OPT_ADAMW:
        is_adamw = Int32(1)
    if out_of_place:
        step_count_launch()
        ctx.enqueue_function[adam_update_oop_kernel](
            p_out.unsafe_ptr(),
            m_out.unsafe_ptr(),
            v_out.unsafe_ptr(),
            param.unsafe_ptr(),
            grad.unsafe_ptr(),
            m_state.unsafe_ptr(),
            v_state.unsafe_ptr(),
            Int32(n),
            is_adamw,
            cfg.beta1,
            cfg.beta2,
            cfg.eps,
            cfg.weight_decay,
            sc.c1,
            sc.c2,
            sc.step_size,
            sc.rt_bc2,
            sc.decay_mul,
            grid_dim=(step_glue_blocks(n, OPT_TPB), 1, 1),
            block_dim=(OPT_TPB, 1, 1),
        )
    else:
        step_count_launch()
        ctx.enqueue_function[adam_update_kernel](
            param.unsafe_ptr(),
            grad.unsafe_ptr(),
            m_state.unsafe_ptr(),
            v_state.unsafe_ptr(),
            denom_out.unsafe_ptr(),
            q_out.unsafe_ptr(),
            Int32(n),
            is_adamw,
            cfg.beta1,
            cfg.beta2,
            cfg.eps,
            cfg.weight_decay,
            sc.c1,
            sc.c2,
            sc.step_size,
            sc.rt_bc2,
            sc.decay_mul,
            cfg.lr,
            sc.bc1,
            sc.bc2,
            grid_dim=(_opt_grid_for(n), 1, 1),
            block_dim=(OPT_TPB, 1, 1),
        )
    step_count_sync()
    ctx.synchronize()
    # `[[mojo-buffer-freed-at-last-use]]`: every buffer is the caller's and
    # outlives the wait above.
    _ = param
    _ = grad
    _ = m_state
    _ = v_state
    _ = p_out
    _ = m_out
    _ = v_out
    _ = denom_out
    _ = q_out



def _byte_neural_group_update(ctx: DeviceContext, mut tr: ByteTrainer, next_step: Int) raises -> Bool:
    """NN56 actual byte-LM transactional update; NN55 replaces output scans.

    The tensor registry defines groups. This model has one admitted AdamW
    config, repeated in metadata, while the kernel supports distinct configs.
    Gradients and all carried state retain their existing entry scans. Updates
    are tentative until the normal finish/rollback boundary accepts them.
    """
    var n = tr.config.n_total()
    opt_refuse_device_inputs(ctx, tr.buffers.param, tr.buffers.grad,
        tr.buffers.m_state, tr.buffers.v_state, tr.buffers.offsets, tr.optimizer)
    # byte_validate_optimizer currently refuses clipping and non-AdamW options;
    # this native arm inherits that admission rather than silently dropping it.
    var offsets = List[Int32]()
    var configs = List[OptimizerConfig]()
    var steps = List[Int]()
    var kinds = List[Int32]()
    var groups = len(tr.buffers.offsets) - 1
    var largest = 0
    for j in range(groups):  # small-loop(groups: optimizer parameter groups): builds per-group metadata, not data
        offsets.append(Int32(tr.buffers.offsets[j]))
        largest = max(largest, tr.buffers.offsets[j + 1] - tr.buffers.offsets[j])
        configs.append(tr.optimizer.copy())
        steps.append(next_step)
        kinds.append(Int32(1))
    offsets.append(Int32(n))
    var d_table = nn_adam_scalar_table(ctx, configs, steps)
    var d_offsets = ctx.enqueue_create_buffer[DType.int32](groups + 1)
    var d_kinds = ctx.enqueue_create_buffer[DType.int32](groups)
    ctx.enqueue_copy(dst_buf=d_offsets, src_ptr=offsets.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=d_kinds, src_ptr=kinds.unsafe_ptr())
    var parts = ctx.enqueue_create_buffer[DType.int32](4 * groups * ((largest + NN_ADAM_TPB - 1) // NN_ADAM_TPB))
    var status = ctx.enqueue_create_buffer[DType.int32](4)
    tr.buffers.flags_before = tr.buffers.buf_initialized.copy()
    tr.shadow_step = tr.completed_steps
    nn_grouped_adam_into[NN55_BLOCK_STATUS](ctx, tr.buffers.shadow_p, tr.buffers.shadow_m,
        tr.buffers.shadow_v, tr.buffers.param, tr.buffers.grad, tr.buffers.m_state,
        tr.buffers.v_state, d_offsets, d_kinds, d_table, parts, status, n, groups, largest)
    ctx.synchronize()
    # The old state remains in the three original handles until the launch has
    # finished. After swapping, existing recovery owns exactly that old state.
    swap(tr.buffers.param, tr.buffers.shadow_p)
    swap(tr.buffers.m_state, tr.buffers.shadow_m)
    swap(tr.buffers.v_state, tr.buffers.shadow_v)
    tr.shadow_valid = True
    comptime if NN55_BLOCK_STATUS:
        _byte_live_status_finish(ctx, status, n, next_step)
        tr.live_status_steps += 1
    _ = offsets
    _ = kinds
    _ = d_offsets^
    _ = d_kinds^
    _ = d_table^
    _ = parts^
    _ = status^
    return NN55_BLOCK_STATUS


def _byte_live_status_launch(ctx: DeviceContext,mut tr: ByteTrainer,mut status: DeviceBuffer[DType.int32],next_step: Int) raises:
    var n=tr.config.n_total()
    var cfg=tr.optimizer.copy()
    var sc=device_step_scalars(cfg,next_step)
    status.enqueue_fill(Int32(n))
    step_count_launch()
    ctx.enqueue_function[adam_update_oop_status_kernel](
        status.unsafe_ptr(),tr.buffers.shadow_p.unsafe_ptr(),tr.buffers.shadow_m.unsafe_ptr(),tr.buffers.shadow_v.unsafe_ptr(),tr.buffers.param.unsafe_ptr(),tr.buffers.grad.unsafe_ptr(),tr.buffers.m_state.unsafe_ptr(),tr.buffers.v_state.unsafe_ptr(),Int32(n),Int32(1) if cfg.kind==OPT_ADAMW else Int32(0),cfg.beta1,cfg.beta2,cfg.eps,cfg.weight_decay,sc.c1,sc.c2,sc.step_size,sc.rt_bc2,sc.decay_mul,
        grid_dim=(step_glue_blocks(n,OPT_TPB),1,1),block_dim=(OPT_TPB,1,1),
    )
    # Handles are swapped by _byte_glue_update before status refusal so the
    # existing rollback always sees the new state and its pre-update shadow.


def _byte_live_status_finish(ctx: DeviceContext,mut status: DeviceBuffer[DType.int32],n: Int,completed: Int) raises:
    if completed<0 or completed>=1000000:
        raise Error("byte LM: completed step must be in [0,1000000)")
    var host=ctx.enqueue_create_host_buffer[DType.int32](4)
    ctx.enqueue_copy(dst_ptr=host.unsafe_ptr(),src_buf=status)
    step_count_sync()
    ctx.synchronize()
    var names: List[String]=["parameters","first moments","second moments"]
    for field in range(3):
        if host[field]<Int32(n):
            raise Error("byte LM: nonfinite "+names[field]+" at "+String(host[field]))
    if host[3]<Int32(n):
        raise Error("byte LM: negative second moment")

def _byte_glue_update(ctx: DeviceContext, mut tr: ByteTrainer, next_step: Int, arm: Int) raises -> Bool:
    """DEVIATIONS 2646 and 2647: the shadow point and the update of
    `_byte_step_device` under a glue arm carrying `optskip` and/or
    `noshadow` (brief sections 4.2, 4.3, 5.2, 5.3). Reached from a trial
    build, and since DEVIATION 2649 from a shipped build whose column
    default carries an update bit (NVIDIA today, no other column). Ends with the new state
    in `param`, `m_state`, `v_state`, the pre-update state in `shadow_*`,
    `flags_before` and `shadow_step` set and `shadow_valid` True, exactly
    the invariant the shipped shadow copy plus `identical_optimizer_step`
    leave, so `validate_after`, `_byte_recover` and `byte_rollback` run
    unchanged after it.

    Without `noshadow`: the shipped shadow copy, then (unless `optskip`) the
    shipped entry scans, then the shipped in-place kernel. With `noshadow`:
    (unless `optskip`) the entry scans with `shadow_valid` still False, then
    the out-of-place kernel into `shadow_*`, the wait, three handle swaps,
    and only then `shadow_valid = True`, so a raise before the swaps leaves
    the state untouched and takes `_byte_recover`'s no-shadow branch.

    The fault-injection build is refused at compile time: its `opt_refuse`
    site writes `m_state` between the shadow copy and the optimizer, which
    is the one write the `optskip` argument excludes (brief section 5.2)."""
    comptime if STEP_GLUE_TRIAL or STEP_GLUE_SHIPPED_UPDATE:
        comptime assert not BYTE_LM_FAULT_INJECT, (
            "the step glue update path and MOJOLEARN_BYTE_LM_FAULT_INJECT are"
            " not combined: the G4 fault sites assume the shipped update order"
            ". Since"
            " DEVIATION 2649 this refusal also covers a shipped build whose"
            " column default carries an update bit, which is the build the"
            " fault-inject harness would otherwise reach unguarded."
        )
    var n = tr.config.n_total()
    var ton = timing_on()
    var tk = Int(perf_counter_ns())
    var out_of_place = (arm & STEP_GLUE_NOSHADOW) != 0
    if not out_of_place:
        # The shipped shadow point, as `_byte_step_device` spells it.
        _copy_into(ctx, tr.buffers.shadow_p, tr.buffers.param, 0, 0, n)
        _copy_into(ctx, tr.buffers.shadow_m, tr.buffers.m_state, 0, 0, n)
        _copy_into(ctx, tr.buffers.shadow_v, tr.buffers.v_state, 0, 0, n)
        step_count_sync()
        ctx.synchronize()
        tr.buffers.flags_before = tr.buffers.buf_initialized.copy()
        tr.shadow_step = tr.completed_steps
        tr.shadow_valid = True
        timing_tick(ctx, ton, tk, "step.shadow_copy")
        timing_bytes(ton, "step.shadow_copy_bytes", 3 * n * 4)
    if (arm & STEP_GLUE_OPTSKIP) == 0:
        opt_refuse_device_inputs(ctx, tr.buffers.param, tr.buffers.grad,
            tr.buffers.m_state, tr.buffers.v_state, tr.buffers.offsets, tr.optimizer)
        timing_tick(ctx, ton, tk, "step.opt_refuse_scan")
    if out_of_place:
        tr.buffers.flags_before = tr.buffers.buf_initialized.copy()
        tr.shadow_step = tr.completed_steps
    var use_live_status = False
    # I10 sourcecbcc8dcd3303 (2026-10-06) L40S scoped WIN: live/base0.960,
    # 1.601/1.667ms for one resident byte-training step (34944parameters,
    # B2,L32,d_model32,two blocks). Driver selects noshadow/OOP Adam without
    # fault injection, admitting this distinct status path. One same-process
    # warmup/score; accepted identity reused, no additional validation.
    # Matching MI325X3.066/3.087ms gives live/base0.993 (near-neutral small gain).
    # Both paired receipts share frozen source, per-vendor machine and case;
    # full application qualification remains pending. This single synthetic
    # step does not justify promotion: live status stays default OFF.
    # Evidence: experiments/performance_ideas/measurements/20261006/index.json,
    # I10 AMD/NVIDIA; retained per-arm binary hashes and raw capture paths.
    comptime if (GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL
                and is_defined["MOJOLEARN_TRAIN_LIVE_STATUS"]()
                and not is_defined["MOJOLEARN_IDN_ALL_OFF"]()):
        # This mechanism changes scheduling only on the existing admitted
        # OOP Adam path. Fault builds keep the required post-fault scan;
        # the path already refuses fault injection before it changes state.
        use_live_status = out_of_place and not BYTE_LM_FAULT_INJECT
    var live = List[DeviceBuffer[DType.int32]]()
    if use_live_status:
        live.append(ctx.enqueue_create_buffer[DType.int32](4))
        _byte_live_status_launch(ctx,tr,live[0],next_step)
    else:
        byte_glue_update_launch(ctx, tr.buffers.param, tr.buffers.grad,
            tr.buffers.m_state, tr.buffers.v_state, tr.buffers.shadow_p,
            tr.buffers.shadow_m, tr.buffers.shadow_v, tr.buffers.denom_out,
            tr.buffers.q_out, n, tr.optimizer, next_step, out_of_place)
    if out_of_place:
        # The new state is in `shadow_*` and the pre-update state is still in
        # `param`, `m_state`, `v_state`; swap the handles so every later
        # reader sees what the shipped path leaves (brief section 4.3).
        swap(tr.buffers.param, tr.buffers.shadow_p)
        swap(tr.buffers.m_state, tr.buffers.shadow_m)
        swap(tr.buffers.v_state, tr.buffers.shadow_v)
        tr.shadow_valid = True
    if use_live_status:
        _byte_live_status_finish(ctx,live[0],n,next_step)
        tr.live_status_steps += 1
    timing_tick(ctx, ton, tk, "step.optimizer")
    _ = live^
    return use_live_status


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
    _byte_admit_tokens(ctx, trainer, token_ids)
    trainer.healthy = False
    var trace = IdentityTrace.disabled()
    var loss = _byte_forward_loss(ctx, trainer, token_ids, trace)
    step_count_sync()
    ctx.synchronize()
    _ = trace
    trainer.healthy = True
    return loss


def byte_eval_loss_resident(ctx: DeviceContext, mut trainer: ByteTrainer,
                            token_ids: List[Int32]) raises -> Float32:
    """`byte_eval_loss` without the 3n download and host validation
    (DEVIATION 2514, design 2.3, same argument as the resident step: the
    state was validated when last written and no write path exists between
    calls). `_byte_forward_loss` has no write path to param/m/v (gate G1
    exports before and after an evaluation to show it). The CE refusal
    still scans the logits and the loss is still checked finite. A raise
    here leaves the state untouched; the trainer is re-scanned once to
    prove the context answers before it is marked healthy again."""
    _require_profile()
    var config = trainer.config.copy()
    byte_validate_tokens(token_ids, config)
    byte_validate_optimizer(trainer.optimizer)
    if not trainer.healthy:
        raise Error("byte LM: session lost; restore a retained export")
    trainer.healthy = False
    trainer.shadow_valid = False
    var trace = IdentityTrace.disabled()
    var loss = Float32(0)
    var failed = False
    var message = String("")
    try:
        loss = _byte_forward_loss(ctx, trainer, token_ids, trace)
        step_count_sync()
        ctx.synchronize()
    except error:
        failed = True
        message = String(error)
    _ = trace
    if failed:
        _byte_recover(ctx, trainer, message)
    trainer.healthy = True
    return loss
