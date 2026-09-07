# SPDX-License-Identifier: Apache-2.0
"""Root-only fixed-profile NVIDIA/AMD checkpoint continuation driver.

Separate executable: checkpoint_check.mojo's existing default gates are unchanged.
MOJOLEARN_TRAIN_RESUME_ACTION: head8, continuous16, or resume16.
MOJOLEARN_TRAIN_RESUME_OUTPUT: new checkpoint output path (receipt adds .json).
MOJOLEARN_TRAIN_RESUME_INPUT: a foreign head8 checkpoint, only for resume16.
MOJOLEARN_TRAIN_EXPECT_VENDOR: cuda or hip, verified from the compiled binary.
MOJOLEARN_TRAIN_SEED: existing decimal seed spelling; default SEED_BASE.
The descriptor always plans 16 steps, including a checkpoint written at step 8.
"""
from std.memory import bitcast
from std.python import Python
from max.gpu.host import DeviceContext
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
from checks.vendor import COMPILED_VENDOR
from core.identity_trace import FNV_OFFSET, IdentityTrace, fnv1a64_bytes
from embedding.checks.embedding_identical import ANY_EMB_SABOTAGE
from gemm.checks.gemm_identical import ANY_SABOTAGE as GEMM_SABOTAGE
from gemm.checks.gemm_backward import ANY_BWD_SABOTAGE as GEMM_BWD_SABOTAGE
from training.checks.loss import ANY_LOSS_SABOTAGE
from training.checks.optimizer import ANY_SABOTAGE as OPT_SABOTAGE
from transformer.checks.transformer_backward import BWD_ANY_SABOTAGE, LlamaBackwardStages
from transformer.impl.transformers.models.llama.modeling_llama import (
    BLOCK_ANY_SABOTAGE, LlamaDeviceStages, LlamaRopeTable,
)
from training.checkpoint import Checkpoint, load_checkpoint, save_checkpoint
from training.checks.checkpoint_check import (
    checkpoint_of, device_weights, read_file_bytes, restore_into, train_names,
)
from training.checks.train_loop import (
    ARM_NONE, SEED_BASE, TRAIN_B, TRAIN_L, TRAIN_ROPE_POSITIONS,
    TRAIN_ROPE_THETA, TrainBuffers, TrainConfig, digest_of_lists, env_str,
    env_u64, hex16, train_batch_ids, train_dims, train_offsets, train_step,
    unpack_params,
)


def _bits(x: Float32) -> UInt32:
    return bitcast[DType.uint32](x)


def _require_finite(values: List[Float32], name: String) raises:
    for i in range(len(values)):
        if (_bits(values[i]) & UInt32(0x7F800000)) == UInt32(0x7F800000):
            raise Error("resume: nonfinite " + name + " at " + String(i))


def _require_descriptor(ck: Checkpoint, cfg: TrainConfig, completed: Int) raises:
    ck.validate()
    var opt = cfg.optimizer()
    if (ck.t != completed or ck.seed != cfg.seed or ck.steps_planned != 16
        or ck.arm != ARM_NONE or ck.opt_kind != opt.kind
        or _bits(ck.lr) != _bits(opt.lr)
        or _bits(ck.beta1) != _bits(opt.beta1)
        or _bits(ck.beta2) != _bits(opt.beta2)
        or _bits(ck.eps) != _bits(opt.eps)
        or _bits(ck.weight_decay) != _bits(opt.weight_decay)
        or _bits(ck.momentum) != _bits(opt.momentum)
        or _bits(ck.dampening) != _bits(opt.dampening)
        or ck.nesterov != opt.nesterov
        or _bits(ck.max_norm) != _bits(opt.max_norm)):
        raise Error("resume: checkpoint step/seed/optimizer descriptor differs from fixed 16-step run")
    _require_finite(ck.param, "param")
    _require_finite(ck.m_state, "m_state")
    _require_finite(ck.v_state, "v_state")


def main() raises:
    comptime if GLOBAL_NUMERIC_MODE != NUMERIC_IDENTICAL:
        raise Error("resume: only NUMERIC_IDENTICAL is allowed")
    comptime if (ANY_EMB_SABOTAGE or GEMM_SABOTAGE or GEMM_BWD_SABOTAGE
                 or ANY_LOSS_SABOTAGE or OPT_SABOTAGE
                 or BWD_ANY_SABOTAGE or BLOCK_ANY_SABOTAGE):
        raise Error("resume: a numerical sabotage is compiled in")
    var vendor = String(COMPILED_VENDOR)
    var expected = env_str("MOJOLEARN_TRAIN_EXPECT_VENDOR")
    if (expected != "cuda" and expected != "hip") or vendor != expected:
        raise Error("resume: expected cuda/hip must match compiled vendor; Apple execution is excluded")
    var action = env_str("MOJOLEARN_TRAIN_RESUME_ACTION")
    if action != "head8" and action != "continuous16" and action != "resume16":
        raise Error("resume: action must be head8, continuous16, or resume16")
    var output = env_str("MOJOLEARN_TRAIN_RESUME_OUTPUT")
    var input_path = env_str("MOJOLEARN_TRAIN_RESUME_INPUT")
    if output == "" or output == input_path:
        raise Error("resume: output must be a new path distinct from input")
    if (action == "resume16" and input_path == "") or (action != "resume16" and input_path != ""):
        raise Error("resume: input is required exactly for resume16")
    var seed = env_u64("MOJOLEARN_TRAIN_SEED", SEED_BASE)
    var cfg = TrainConfig.for_arm(16, seed, ARM_NONE)
    # The helper opens the incoming inode once and bounds its size BEFORE
    # reading it, then seals an immutable memfd. Existing codec and hash reads
    # use that same capture. Outputs are O_EXCL-reserved before GPU work and
    # written only through owned descriptors, never re-opened caller paths.
    var sysmod = Python.import_module("sys")
    _ = sysmod.path.insert(0, ".")
    var file_helper = Python.import_module("tools.training_cross_vendor_resume")
    var files = file_helper.capture_native_paths(input_path, output)
    var output_sink = String("/proc/self/fd/") + String(Int(py=files.output_fd))
    var receipt_sink = String("/proc/self/fd/") + String(Int(py=files.receipt_fd))
    var first = 1
    var last = 16
    if action == "head8":
        last = 8
    var initial = Checkpoint()
    var input_hash = String("")
    if action == "resume16":
        # Verify codec hash/layout and every descriptor bit BEFORE device work.
        var captured_input = String("/proc/self/fd/") + String(Int(py=files.input_fd))
        initial = load_checkpoint(captured_input, train_offsets(), train_names())
        _require_descriptor(initial, cfg, 8)
        var raw = read_file_bytes(captured_input)
        input_hash = hex16(fnv1a64_bytes(FNV_OFFSET, raw.unsafe_ptr(), len(raw)))
        _ = raw^
        first = 9

    var ctx = DeviceContext()
    var tb = TrainBuffers(ctx, cfg.seed)
    var w = device_weights(ctx, cfg.seed)
    var rope = LlamaRopeTable(ctx, train_dims(), TRAIN_ROPE_THETA, TRAIN_ROPE_POSITIONS)
    var stages = LlamaDeviceStages(ctx, TRAIN_B, TRAIN_L, TRAIN_L, train_dims())
    var bst = LlamaBackwardStages(ctx, TRAIN_B, TRAIN_L, TRAIN_L, train_dims())
    var trace = IdentityTrace.disabled()
    if action == "resume16":
        restore_into(ctx, tb, initial)
    var records = String("")
    for step in range(first, last + 1):
        var ids = train_batch_ids(cfg.seed, step)
        unpack_params(ctx, tb, w)
        var loss = train_step(ctx, tb, w, rope, stages, bst, trace, cfg, ids, step)
        if (_bits(loss) & UInt32(0x7F800000)) == UInt32(0x7F800000):
            raise Error("resume: nonfinite loss at step " + String(step))
        var snapshot = checkpoint_of(ctx, tb, step, cfg)
        _require_descriptor(snapshot, cfg, step)
        var digest = digest_of_lists(snapshot.param, snapshot.m_state, snapshot.v_state, snapshot.offsets)
        if step != first:
            records += ","
        records += (String("{\"step\":") + String(step)
                    + ",\"loss_bits\":" + String(_bits(loss))
                    + ",\"state_h_all\":\"" + hex16(digest.h_all) + "\"}")
        _ = snapshot^
        _ = ids^
    ctx.synchronize()
    var final = checkpoint_of(ctx, tb, last, cfg)
    _require_descriptor(final, cfg, last)
    var hashes = save_checkpoint(output_sink, final)
    # The receipt is published only after the checkpoint is complete.
    with open(receipt_sink, "w") as fh:
        fh.write(String("{\"schema\":\"mojolearn.training.resume-receipt.v1\",\"status\":\"COMPLETE\",")
                 + "\"numeric_mode\":\"identical\",\"vendor\":\"" + vendor
                 + "\",\"action\":\"" + action
                 + "\",\"seed_hex\":\"" + hex16(seed)
                 + "\",\"schedule\":\"train_batch_ids.splitmix64.v1\",\"planned_steps\":16,"
                 + "\"first_step\":" + String(first) + ",\"completed_steps\":" + String(last)
                 + ",\"input_bytes_fnv1a64\":\"" + input_hash
                 + "\",\"checkpoint_h_all\":\"" + hashes.h_all_hex()
                 + "\",\"checkpoint_h_file\":\"" + hashes.h_file_hex()
                 + "\",\"checkpoint_bytes\":" + String(hashes.bytes)
                 + ",\"steps\":[" + records + "]}\n")
    _ = files.finish()
    print("TRAINING RESUME COMPLETE " + action + " " + vendor + " " + hashes.h_all_hex())
    _ = final^
    _ = initial^
    _ = bst^
    _ = stages^
    _ = rope^
    _ = w^
    _ = tb^
    _ = ctx^
    _ = files^
