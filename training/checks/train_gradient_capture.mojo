# SPDX-License-Identifier: Apache-2.0
"""Root-only single-step raw training capture; no new numerical kernels.

Set MOJOLEARN_TRAIN_CAPTURE_OUTPUT to a NEW directory, and
MOJOLEARN_TRAIN_EXPECT_VENDOR to cuda or hip. Optional
MOJOLEARN_TRAIN_CAPTURE_IDS is a JSON file containing exactly 18 token IDs.
No timing claims.

The capture imports no Python. It creates the output directory (refusing an
existing one), writes each array as little-endian raw FP32 words
(`<name>.f32`), the token IDs as little-endian int32 (`token_ids.i32`) and a
plain `native-capture.txt` (vendor, seed, supplied flag, parameter registry).
`tools/transformer_training_gradient_oracle.py seal <dir>` then checks counts,
finiteness, the registry and the IDs and writes `capture.json`, the manifest
the oracle reads.
"""
from std.memory import bitcast
from std.os import mkdir
from max.gpu.host import DeviceContext
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
from checks.vendor import COMPILED_VENDOR
from core.identity_trace import IdentityTrace
from embedding.checks.embedding_identical import ANY_EMB_SABOTAGE
from gemm.neural_dispatch import ANY_SABOTAGE as GEMM_SABOTAGE
from gemm.neural_backward import ANY_BWD_SABOTAGE as GEMM_BWD_SABOTAGE
from training.checks.loss import ANY_LOSS_SABOTAGE
from training.checks.optimizer import ANY_SABOTAGE as OPT_SABOTAGE
from autograd.kernels.transformer import BWD_ANY_SABOTAGE
from transformer.impl.llama.modeling_llama import BLOCK_ANY_SABOTAGE
from training.checks.checkpoint_check import device_weights
from training.checks.train_loop import (
    ARM_NONE, SEED_BASE, TRAIN_B, TRAIN_L, TRAIN_ROPE_POSITIONS,
    TRAIN_ROPE_THETA, TRAIN_D_MODEL, TRAIN_VOCAB, TRAIN_N_HEADS, TRAIN_N_KV,
    TRAIN_HEAD_DIM, TRAIN_INTERMEDIATE, TRAIN_RMS_EPS, TRAIN_J,
    param_id_name, param_id_count, TrainBuffers, TrainConfig, download_f32, env_str,
    env_u64, train_batch_ids, train_dims, train_step, unpack_params,
)
from autograd.kernels.transformer import LlamaBackwardStages
from transformer.impl.llama.modeling_llama import (
    LlamaDeviceStages, LlamaRopeTable,
)


def _put_u32_le(mut dst: List[UInt8], word: UInt32):
    dst.append(UInt8(word & UInt32(0xFF)))
    dst.append(UInt8((word >> 8) & UInt32(0xFF)))
    dst.append(UInt8((word >> 16) & UInt32(0xFF)))
    dst.append(UInt8((word >> 24) & UInt32(0xFF)))


def _write_raw(path: String, raw: List[UInt8]) raises:
    with open(path, "w") as fh:
        fh.write_bytes(Span(raw))


def retain(dir: String, name: String, values: List[Float32]) raises:
    """Every FP32 cell as its exact bits, explicitly little-endian."""
    var raw = List[UInt8](capacity=len(values) * 4)
    for i in range(len(values)):
        _put_u32_le(raw, bitcast[DType.uint32](values[i]))
    _write_raw(dir + "/" + name + ".f32", raw)


def _is_space(c: Int) -> Bool:
    return c == 32 or c == 9 or c == 10 or c == 13


def read_capture_ids(path: String) raises -> List[Int32]:
    """A JSON file holding a flat list of exactly 18 integer IDs in [0, 64).

    Bounded to 4096 bytes. Only `[`, `]`, `,`, ASCII digits and whitespace
    are admitted, so a float, a sign, a nested list or any other JSON value
    is refused."""
    var text = String("")
    with open(path, "r") as fh:
        text = fh.read()
    var b = text.as_bytes()
    if len(b) > 4096:
        raise Error("token fixture exceeds 4096-byte bound")
    var ids = List[Int32]()
    var i = 0
    while i < len(b) and _is_space(Int(b[i])):
        i += 1
    if i >= len(b) or Int(b[i]) != 91:
        raise Error("expected flat list of exactly 18 integer IDs in [0,64)")
    i += 1
    var closed = False
    while i < len(b):
        while i < len(b) and _is_space(Int(b[i])):
            i += 1
        var start = i
        var value = 0
        while i < len(b) and Int(b[i]) >= 48 and Int(b[i]) <= 57:
            value = value * 10 + (Int(b[i]) - 48)
            if value >= 64:
                raise Error("expected flat list of exactly 18 integer IDs in [0,64)")
            i += 1
        if i == start:
            raise Error("expected flat list of exactly 18 integer IDs in [0,64)")
        ids.append(Int32(value))
        while i < len(b) and _is_space(Int(b[i])):
            i += 1
        if i < len(b) and Int(b[i]) == 44:
            i += 1
            continue
        if i < len(b) and Int(b[i]) == 93:
            closed = True
            i += 1
        break
    while i < len(b) and _is_space(Int(b[i])):
        i += 1
    if not closed or i != len(b) or len(ids) != 18:
        raise Error("expected flat list of exactly 18 integer IDs in [0,64)")
    return ids^


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
    var seed = env_u64("MOJOLEARN_TRAIN_SEED", SEED_BASE)
    var out_dir = env_str("MOJOLEARN_TRAIN_CAPTURE_OUTPUT")
    if out_dir == "":
        raise Error("training capture requires MOJOLEARN_TRAIN_CAPTURE_OUTPUT")
    var ids = train_batch_ids(seed, 1)
    var supplied = env_str("MOJOLEARN_TRAIN_CAPTURE_IDS")
    if supplied != "":
        var loaded = read_capture_ids(supplied)
        for i in range(18):
            ids[i] = loaded[i]
    if len(ids) != 18:
        raise Error("training capture expects 18 token IDs")
    # A NEW directory only (mkdir refuses an existing one or a missing parent).
    mkdir(out_dir)
    var native = String("schema mojolearn.training.gradient-capture.native.v1\n")
    native += "vendor " + vendor + "\n"
    native += "seed " + String(seed) + "\n"
    native += "supplied " + String(1 if supplied != "" else 0) + "\n"
    for j in range(TRAIN_J):
        native += ("param " + String(j) + " " + param_id_name(j) + " "
                   + String(param_id_count(j)) + "\n")
    var id_raw = List[UInt8](capacity=len(ids) * 4)
    for i in range(len(ids)):
        _put_u32_le(id_raw, bitcast[DType.uint32](ids[i]))
    _write_raw(out_dir + "/token_ids.i32", id_raw)
    var cfg = TrainConfig.for_arm(1, seed, ARM_NONE)
    var ctx = DeviceContext()
    var tb = TrainBuffers(ctx, seed)
    retain(out_dir, "initial_params", download_f32(ctx, tb.param, tb.n_total))
    retain(out_dir, "initial_m", download_f32(ctx, tb.m_state, tb.n_total))
    retain(out_dir, "initial_v", download_f32(ctx, tb.v_state, tb.n_total))
    var w = device_weights(ctx, seed)
    var rope = LlamaRopeTable(ctx, train_dims(), TRAIN_ROPE_THETA, TRAIN_ROPE_POSITIONS)
    var stages = LlamaDeviceStages(ctx, TRAIN_B, TRAIN_L, TRAIN_L, train_dims())
    var bst = LlamaBackwardStages(ctx, TRAIN_B, TRAIN_L, TRAIN_L, train_dims())
    var trace = IdentityTrace.disabled()
    unpack_params(ctx, tb, w)
    var loss = train_step(ctx, tb, w, rope, stages, bst, trace, cfg, ids, 1)
    var loss_cells = List[Float32]()
    loss_cells.append(loss)
    retain(out_dir, "loss", loss_cells)
    # max_norm=0: optimizer does not clip or mutate packed raw gradients.
    retain(out_dir, "gradients", download_f32(ctx, tb.grad, tb.n_total))
    retain(out_dir, "updated_params", download_f32(ctx, tb.param, tb.n_total))
    retain(out_dir, "updated_m", download_f32(ctx, tb.m_state, tb.n_total))
    retain(out_dir, "updated_v", download_f32(ctx, tb.v_state, tb.n_total))
    ctx.synchronize()
    # Written last: its presence says every array above is complete.
    with open(out_dir + "/native-capture.txt", "w") as fh:
        fh.write(native)
    _ = tb^
    _ = w^
    _ = rope^
    _ = stages^
    _ = bst^
    _ = trace^
    print("TRAIN_GRADIENT_CAPTURE_COMPLETE: root must retain guard exit status")
