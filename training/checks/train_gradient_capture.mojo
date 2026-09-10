# SPDX-License-Identifier: Apache-2.0
"""Root-only single-step raw training capture; no new numerical kernels.

Set MOJOLEARN_TRAIN_CAPTURE_OUTPUT to a NEW directory, and
MOJOLEARN_TRAIN_EXPECT_VENDOR to cuda or hip. Optional
MOJOLEARN_TRAIN_CAPTURE_IDS is a JSON file containing exactly 18 token IDs.
Run from repository root with Python available. No timing claims.
"""
from std.memory import bitcast
from std.python import Python, PythonObject
from max.gpu.host import DeviceContext
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
from checks.vendor import COMPILED_VENDOR
from core.identity_trace import IdentityTrace
from embedding.checks.embedding_identical import ANY_EMB_SABOTAGE
from gemm.checks.gemm_identical import ANY_SABOTAGE as GEMM_SABOTAGE
from gemm.checks.gemm_backward import ANY_BWD_SABOTAGE as GEMM_BWD_SABOTAGE
from training.checks.loss import ANY_LOSS_SABOTAGE
from training.checks.optimizer import ANY_SABOTAGE as OPT_SABOTAGE
from transformer.checks.transformer_backward import BWD_ANY_SABOTAGE
from transformer.impl.llama.modeling_llama import BLOCK_ANY_SABOTAGE
from training.checks.checkpoint_check import device_weights
from training.checks.train_loop import (
    ARM_NONE, SEED_BASE, TRAIN_B, TRAIN_L, TRAIN_ROPE_POSITIONS,
    TRAIN_ROPE_THETA, TRAIN_D_MODEL, TRAIN_VOCAB, TRAIN_N_HEADS, TRAIN_N_KV,
    TRAIN_HEAD_DIM, TRAIN_INTERMEDIATE, TRAIN_RMS_EPS, TRAIN_J,
    param_id_name, param_id_count, TrainBuffers, TrainConfig, download_f32, env_str,
    env_u64, train_batch_ids, train_dims, train_step, unpack_params,
)
from transformer.checks.transformer_backward import LlamaBackwardStages
from transformer.impl.llama.modeling_llama import (
    LlamaDeviceStages, LlamaRopeTable,
)


def retain(writer: PythonObject, name: String, values: List[Float32]) raises:
    # Decimal INTEGER bits, never decimal floats: the Python bridge preserves
    # every FP32 cell and writes explicitly little-endian raw bytes.
    var words = String("")
    for i in range(len(values)):
        if i != 0:
            words += ","
        words += String(bitcast[DType.uint32](values[i]))
    _ = writer.add_bits(name, words)


def main() raises:
    comptime if (TRAIN_B != 2 or TRAIN_L != 8 or TRAIN_D_MODEL != 32
                 or TRAIN_VOCAB != 64 or TRAIN_N_HEADS != 4 or TRAIN_N_KV != 2
                 or TRAIN_HEAD_DIM != 8 or TRAIN_INTERMEDIATE != 64
                 or TRAIN_J != 11 or TRAIN_ROPE_POSITIONS != 64):
        raise Error("training capture fixed profile changed")
    if TRAIN_RMS_EPS != Float32(1e-6) or TRAIN_ROPE_THETA != Float32(10000):
        raise Error("training capture RMS/RoPE profile changed")
    comptime if GLOBAL_NUMERIC_MODE != NUMERIC_IDENTICAL:
        raise Error("training capture requires IDENTICAL")
    comptime if (ANY_EMB_SABOTAGE or GEMM_SABOTAGE or GEMM_BWD_SABOTAGE
                 or ANY_LOSS_SABOTAGE or OPT_SABOTAGE
                 or BWD_ANY_SABOTAGE or BLOCK_ANY_SABOTAGE):
        raise Error("training capture refuses numerical sabotage builds")
    var vendor = String(COMPILED_VENDOR)
    if (vendor != "cuda" and vendor != "hip") or vendor != env_str("MOJOLEARN_TRAIN_EXPECT_VENDOR"):
        raise Error("training capture requires expected remote cuda/hip; no Apple execution")
    var sysmod = Python.import_module("sys")
    _ = sysmod.path.insert(0, ".")
    var helper = Python.import_module("tools.transformer_training_gradient_oracle")
    var seed = env_u64("MOJOLEARN_TRAIN_SEED", SEED_BASE)
    var writer = helper.CaptureWriter(env_str("MOJOLEARN_TRAIN_CAPTURE_OUTPUT"), vendor, String(seed))
    for j in range(TRAIN_J):
        _ = writer.require_registry(j, param_id_name(j), param_id_count(j))
    var ids = train_batch_ids(seed, 1)
    var supplied = env_str("MOJOLEARN_TRAIN_CAPTURE_IDS")
    if supplied != "":
        var loaded = helper.read_ids(supplied)
        for i in range(18):
            ids[i] = Int32(Int(py=loaded[i]))
    var id_words = String("")
    for i in range(len(ids)):
        if i != 0:
            id_words += ","
        id_words += String(ids[i])
    _ = writer.add_ids(id_words, supplied != "")
    var cfg = TrainConfig.for_arm(1, seed, ARM_NONE)
    var ctx = DeviceContext()
    var tb = TrainBuffers(ctx, seed)
    retain(writer, "initial_params", download_f32(ctx, tb.param, tb.n_total))
    retain(writer, "initial_m", download_f32(ctx, tb.m_state, tb.n_total))
    retain(writer, "initial_v", download_f32(ctx, tb.v_state, tb.n_total))
    var w = device_weights(ctx, seed)
    var rope = LlamaRopeTable(ctx, train_dims(), TRAIN_ROPE_THETA, TRAIN_ROPE_POSITIONS)
    var stages = LlamaDeviceStages(ctx, TRAIN_B, TRAIN_L, TRAIN_L, train_dims())
    var bst = LlamaBackwardStages(ctx, TRAIN_B, TRAIN_L, TRAIN_L, train_dims())
    var trace = IdentityTrace.disabled()
    unpack_params(ctx, tb, w)
    var loss = train_step(ctx, tb, w, rope, stages, bst, trace, cfg, ids, 1)
    var loss_cells = List[Float32]()
    loss_cells.append(loss)
    retain(writer, "loss", loss_cells)
    # max_norm=0: optimizer does not clip or mutate packed raw gradients.
    retain(writer, "gradients", download_f32(ctx, tb.grad, tb.n_total))
    retain(writer, "updated_params", download_f32(ctx, tb.param, tb.n_total))
    retain(writer, "updated_m", download_f32(ctx, tb.m_state, tb.n_total))
    retain(writer, "updated_v", download_f32(ctx, tb.v_state, tb.n_total))
    ctx.synchronize()
    _ = writer.finish()
    _ = tb^
    _ = w^
    _ = rope^
    _ = stages^
    _ = bst^
    _ = trace^
    print("TRAIN_GRADIENT_CAPTURE_COMPLETE: root must retain guard exit status")
