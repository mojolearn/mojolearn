#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The neural family of tools/bench_board.py: THE WHEEL'S PUBLIC PYTHON API
against torch on the same box, interleaved round by round, quality beside
every time.

    python3 tools/bench_board_neural.py race --lane lm-train-step --shape full \\
        --arms ours,torch-eager-fp32,torch-compile-bf16 --rounds 5 --out DIR --work DIR \\
        --ours-python PY --theirs-python PY

It speaks tools/classical_two_datasets.py's protocol (it reuses its `Worker`):
one persistent worker process per arm, a warm-up round then `--rounds` timed
rounds, the arm order rotated every round, and one race JSON whose shape
tools/bench_board.py's `classical_cells` already reads. This module imports
only the standard library at import time, so the board reads its tables
(`LANES`, `opponents`, `NOT_PLANNED`, ...) without torch or numpy.

THE LANES (each is a public mojolearn entry point, installed from the wheel)
----------------------------------------------------------------------------
GPU lanes (the class runs on the box's GPU):
  lm-train-step        LanguageModelTrainer(resident=True, step_result='lean')
                       .train_step(ids): forward, mean CE, backward, AdamW.
  lm-forward           LanguageModelTrainer(resident=True).logits(ids).
  gemm                 mojolearn.linalg.matmul(a, b), fp32.
  transformer-forward  TransformerBlock(weights, n_heads, n_kv_heads,
                       head_dim).forward(x).
  mamba1-forward       Mamba1Block(weights).forward(x).
  mamba2-forward       Mamba2Block(weights).forward(x).
  mamba3-forward       Mamba3Block(weights).forward(x).
  samba-train-step     SambaStack(config, weights).train_step(inputs, targets):
                       forward, mean CE, full backward, AdamW (no clip).
  samba-forward        SambaStack(config, weights).forward(inputs): logits.
  mlp-train-step       SmallMLPTrainer(w1, b1, w2, b2).train_step(X, y):
                       8-16-3 ReLU, mean CE, AdamW.
CPU lanes (the *Inference classes run on the host binding, on the CPU):
  transformer-infer    TransformerBlockInference(...).forward(x).
  mamba1-infer / mamba2-infer / mamba3-infer
                       Mamba{1,2,3}BlockInference(weights).forward(x).
  samba-infer          SambaInference(config, weights).forward(inputs).
  mlp-infer            MLPInference(w1, b1, w2, b2).predict_logits(X).
The Mamba blocks and TransformerBlock expose a backward (a VJP), not a
training step; they have no optimizer, so they race forward only.

THE OPPONENT: torch, at every fast setting it supports on the box
-----------------------------------------------------------------
Our IDENTICAL arm is raced against the opponent's FASTEST supported setting
(bench/OPPONENT_REFERENCE.md), so every lane carries one torch arm per
setting of tools/torch_lm_step_opponent.py's COLUMNS, the precision in the
arm name:

  torch-eager-fp32     eager, float32, TF32 off (that file's THE ROW).
  torch-eager-tf32     eager, float32 with TF32 ON (NVIDIA CUDA only: TF32 is
                       an NVIDIA tensor-core matmul mode; on ROCm and MPS the
                       flag does nothing, so the arm is not planned there).
  torch-compile-fp32   torch.compile (inductor, default mode), TF32 off.
  torch-compile-tf32   compile with TF32 on (NVIDIA CUDA only).
  torch-eager-bf16     bf16 MIXED PRECISION: torch.autocast(device, bfloat16)
                       around the forward (and the loss); parameters,
                       gradients and AdamW state stay float32. The worker
                       probes autocast on the device (the twin's
                       `probe_autocast`) and refuses the arm by name when it
                       does not work there.
  torch-compile-bf16   compile inside the same autocast.
A CPU lane's torch arms run on the CPU and are named `torch-cpu-<setting>`
(fp32 and bf16, eager and compiled; TF32 does not exist there). An arm that
fails on the box (compile on MPS, bf16 on an old GPU) refuses BY NAME in its
cell; nothing falls back to another setting or device. Every bf16 and TF32
arm is ANOTHER PRECISION than ours; its quality columns (loss or output
difference against ours) show how far.

The torch models (every one is an existing repo twin; nothing new invented):
  lm-*          tools/torch_lm_step_opponent.py build_model (SDPA; on the
                torch.cuda API a probed backend is pinned for the fp32/tf32
                columns, torch's own pick for bf16).
  gemm          a @ b.
  transformer   tools/speed_torch_seq.py LlamaEager.block with sdpa=True
                (torch's scaled_dot_product_attention, causal mask).
  mamba1        mamba/corpus/gen_corpus.py block_forward: the PURE-PYTORCH
                reference (mamba_ssm's selective_scan_ref, a per-token Python
                loop). NOT a fused deployment kernel: mamba-ssm's CUDA
                kernels are not installed by the board. A per-token loop is
                not a torch.compile target, so mamba1 has no compile arms.
  mamba2        gen_corpus.py m2_forward: the pure-PyTorch chunked SSD
                reference (mamba_ssm ssd_minimal / HF mamba2_chunk_scan),
                not the Triton kernels.
  mamba3        gen_corpus.py m3_forward: the pure-PyTorch Mamba-3 SISO
                reference, not the fused kernels.
  mamba1/2/3, NVIDIA, arms mamba-ssm-fp32 / mamba-ssm-tf32: mamba_ssm's own
                fused kernels (Block + Mamba / Mamba2 / Mamba3 SISO, the
                deployment path; MAMBA_SSM_ARMS below), our weights loaded.
  samba         embedding, then per layer m3_forward or LlamaEager.block
                (SDPA), final RMSNorm (eps 1e-5), tied head, mean CE,
                torch.optim.AdamW: the stack composed from those two twins.
  mlp           F.linear, ReLU, F.linear, mean CE, torch.optim.AdamW.
gen_corpus.py sets torch's deterministic switch and one thread at import; the
worker turns both back (tools/speed_torch_seq.py DEVIATION 1856) and records
it. The mamba references create tensors without a device, so they run inside
`with torch.device(dev)`.

SAME INPUTS, BYTE FOR BYTE
--------------------------
The conductor writes ONE input file per race and every arm reads it (seed 7):
  * LM: parameters default_rng(7).normal(0, .02) float32, +1 on every norm;
    batches of the byte stream (the installed mojolearn package's .py
    sources, sorted and concatenated; sha256 recorded), step k row b =
    stream[(k*B*L + b*L) % (n - L - 1) : + L + 1]. Samba reads the same
    stream the same way.
  * transformer: every weight normal(0, .02), +1 on the two norms; x
    standard normal.
  * mamba1/2/3: every weight uniform in mamba/corpus/gen_corpus.py's default
    range for that tensor (default_ranges / m2_default_ranges /
    m3_default_ranges), x uniform(-2, 2).
  * samba: SambaStack's initializer rules on numpy's generator: embedding
    normal(0, .02), every projection uniform(+-1/sqrt(fan_in)), dt_bias
    uniform(-4, -2), norms, D and the B/C biases ones. Our worker refuses
    unless mojolearn.SambaConfig's registry equals the conductor's.
  * mlp: weights uniform(+-1/sqrt(fan_in)); X standard normal, labels
    integers 0..2.
  * gemm: A [m, k] and B [k, n] standard normal.

THE CLOCK (both sides): host inputs to the device, the call, the result
(loss or output) back on the host, synchronized. Training state stays on the
device between steps on both sides; round r is step r + 1.

QUALITY (the conductor, float64 NumPy)
--------------------------------------
  train lanes     loss_first_step, loss_last_step (same init, same batches),
                  loss_first_abs_diff_vs_ours, loss_last_abs_diff_vs_ours.
  forward lanes   max_abs_diff_vs_ours, max_rel_diff_vs_ours (max |o - ours|
                  / max |ours|); the logit lanes add mean_nll; gemm adds
                  max_rel_err_vs_fp64.

SHAPES (`--shape`): see SHAPES below; `full` is the board, `small` a smoke.
The CPU lanes cap the sequence length at 512 in `full`.

IDENTICAL ONLY. The neural surface builds `identical` only; an `ours` worker
refuses to start under any other MOJOLEARN_NUMERIC_MODE.
"""
import argparse
import glob
import hashlib
import importlib.util
import json
import os
import platform
import shlex
import statistics
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)

SEED = 7

#: AdamW on every train lane, every arm, passed EXPLICITLY (never a library
#: default): ours' LanguageModelTrainer / SambaStack / SmallMLPTrainer and
#: torch.optim.AdamW. Samba runs without clipping on both (max_norm=None).
ADAMW = {"lr": 1e-3, "betas": (0.9, 0.999), "eps": 1e-8, "weight_decay": 0.01}
#: TransformerBlock / LlamaEager: RMSNorm eps and RoPE base, passed explicitly
#: to ours and read back from tools/speed_torch_seq.py on the torch side.
BLOCK_NORM_EPS, BLOCK_ROPE_THETA = 1e-6, 10000.0
#: SambaStack's final RMSNorm eps and dropout, passed explicitly; the torch
#: stack twin uses the same eps and has no dropout.
SAMBA_NORM_EPS, SAMBA_DROPOUT = 1e-5, 0.0

# ---------------------------------------------------------------------------
# The tables the board reads (standard library only)
# ---------------------------------------------------------------------------

LANES = ("lm-train-step", "lm-forward", "gemm",
         "transformer-forward", "transformer-infer",
         "mamba1-forward", "mamba1-infer", "mamba2-forward", "mamba2-infer",
         "mamba3-forward", "mamba3-infer",
         "samba-train-step", "samba-forward", "samba-infer",
         "mlp-train-step", "mlp-infer",
         # 2026-09-29: the byte LM's CPU inference and CPU training step, and the
         # bf16 and int8 GEMM profiles (SmallByteLanguageModelTrainer IS
         # LanguageModelTrainer, raced by lm-train-step and lm-forward)
         "lm-infer", "lm-host-train-step", "gemm-bf16", "gemm-int8")
#: The model each lane runs.
MODEL_OF = {"lm-train-step": "lm", "lm-forward": "lm", "gemm": "gemm",
            "transformer-forward": "transformer", "transformer-infer": "transformer",
            "mamba1-forward": "mamba1", "mamba1-infer": "mamba1",
            "mamba2-forward": "mamba2", "mamba2-infer": "mamba2",
            "mamba3-forward": "mamba3", "mamba3-infer": "mamba3",
            "samba-train-step": "samba", "samba-forward": "samba", "samba-infer": "samba",
            "mlp-train-step": "mlp", "mlp-infer": "mlp",
            "lm-infer": "lm", "lm-host-train-step": "lm", "gemm-bf16": "gemm", "gemm-int8": "gemm"}
TRAIN_LANES = ("lm-train-step", "samba-train-step", "mlp-train-step", "lm-host-train-step")
#: Where OUR class runs: the *Inference classes are the host binding.
DEVICE_OF = {lane: ("cpu" if lane.endswith("-infer") or lane == "lm-host-train-step" else "gpu")
             for lane in LANES}
#: The data each lane reads (the board's `dataset` column).
DATA_OF = {lane: ("bytes" if MODEL_OF[lane] in ("lm", "samba") else "gaussian") for lane in LANES}

#: torch settings, in tools/torch_lm_step_opponent.py COLUMNS order.
TORCH_SETTINGS = ("eager-fp32", "eager-tf32", "compile-fp32", "compile-tf32",
                  "eager-bf16", "compile-bf16")
#: The GPU settings planned per vendor. TF32 is an NVIDIA CUDA matmul mode:
#: on ROCm torch accepts the flag and does nothing (torch_lm_step_opponent.py
#: writes NOT APPLICABLE), and MPS has no such mode.
GPU_SETTINGS = {
    "nvidia": TORCH_SETTINGS,
    "amd": ("eager-fp32", "compile-fp32", "eager-bf16", "compile-bf16"),
    "apple": ("eager-fp32", "compile-fp32", "eager-bf16", "compile-bf16"),
}
CPU_SETTINGS = ("eager-fp32", "compile-fp32", "eager-bf16", "compile-bf16")
#: Lanes whose torch twin is not a compile target (a per-token Python loop).
NO_COMPILE = ("mamba1-forward", "mamba1-infer")
VENDORS = ("apple", "nvidia", "amd")

#: THE STRONGEST MAMBA OPPONENT (Andrew, 2026-09-29: "we should be using the
#: strongest opponent always"): the mamba_ssm package's own fused kernels,
#: beside the pure-PyTorch reference arms. One block is mamba_ssm's
#: modules/block.py Block (fused Triton add+RMSNorm, the mixer, residual in
#: fp32) around its mixer: mamba_simple.Mamba on the fast path (causal_conv1d
#: + mamba_inner_fn, the selective-scan CUDA kernel), mamba2.Mamba2 on the
#: memory-efficient path (mamba_split_conv1d_scan_combined, the SSD Triton
#: kernels), mamba3.Mamba3 SISO (mamba3_siso_combined, Triton). Weights, input
#: and dtype (float32) are ours, loaded with load_state_dict(strict=True) and
#: read back bit for bit. Two settings, the precision in the arm name:
#:   mamba-ssm-fp32   TF32 off in torch and TRITON_F32_DEFAULT=ieee, so every
#:                    fp32 tl.dot in the Triton kernels is IEEE fp32 (ours'
#:                    precision).
#:   mamba-ssm-tf32   TF32 on in torch and TRITON_F32_DEFAULT=tf32 (Triton's
#:                    own default for fp32 tl.dot: what an out-of-the-box
#:                    mamba_ssm install runs). ANOTHER PRECISION than ours.
#: Pinned to the state-spaces/mamba commit mamba/corpus/gen_corpus.py cites
#: (tools/bench_board.py MAMBA_SSM_*): PyPI's mamba-ssm 2.3.2.post1 is an
#: OLDER Mamba-3 (A = -softplus(dd_A)); this commit's Mamba3 is the
#: heavy-tail A our Mamba3Block and the reference implement.
MAMBA_SSM_ARMS = ("mamba-ssm-fp32", "mamba-ssm-tf32")
MAMBA_SSM_LANES = ("mamba1-forward", "mamba2-forward", "mamba3-forward")
#: mamba_ssm's CUDA extensions and Triton kernels are built and raced on
#: NVIDIA only (see NOT_PLANNED for AMD and Apple).
MAMBA_SSM_VENDORS = ("nvidia",)

ARMS = ("ours", "ours-cpu") + tuple("torch-" + s for s in TORCH_SETTINGS) \
    + tuple("torch-cpu-" + s for s in CPU_SETTINGS) + ("torch-eager-int8", "torch-compile-int8") \
    + MAMBA_SSM_ARMS
#: gemm-int8's torch settings: torch._int_mm (int8 x int8 -> int32), no autocast, no TF32
INT8_COLUMNS = {"eager_int8": dict(tf32=False, compile=False, autocast=None),
                "compile_int8": dict(tf32=False, compile=True, autocast=None)}

#: What is left off the plan per vendor, named (the board prints it).
NOT_PLANNED = {
    v: (["torch-eager-tf32 / torch-compile-tf32: TF32 is an NVIDIA CUDA tensor-core matmul "
         "mode; torch on %s accepts the flag and changes nothing"
         % {"apple": "MPS", "amd": "ROCm"}[v]] if v != "nvidia" else [])
    + ["torch-compile-* on mamba1-forward and mamba1-infer: the only torch Mamba-1 twin is "
       "the pure-PyTorch reference scan, a per-token Python loop that torch.compile would "
       "unroll L times; mamba1 races the eager arms only",
       "gemm-bf16 races torch's bf16 settings only (bf16 operands, fp32 accumulate)"]
    + (["gemm-int8: torch._int_mm is a CUDA kernel; torch on %s has no int8 matmul, so ours "
        "races alone" % {"apple": "MPS", "amd": "ROCm"}[v]] if v != "nvidia" else [])
    + ({"apple": ["mamba-ssm-* on mamba*-forward: mamba_ssm's kernels are CUDA and Triton "
                  "(no Metal build exists); the Mamba lanes race the torch reference arms"],
        "amd": ["mamba-ssm-* on mamba*-forward: mamba_ssm publishes no ROCm wheel, and its "
                "gfx942 source build (setup.py's HIP path, causal-conv1d's and the Mamba-3 "
                "Triton kernels on ROCm) has not been built and checked on the board's AMD box; "
                "the Mamba lanes race the torch reference arms there"]}.get(v, [])
       + ["mamba-ssm bf16: our Mamba blocks are float32, so mamba_ssm races in float32 (and "
          "its TF32 setting); torch's bf16 arms carry the lower precision"])
    for v in VENDORS}

#: The board's "Not covered" lines for the neural family.
NOT_COVERED = [
    "The Mamba torch-* arms are the repo's pure-PyTorch references (mamba/corpus/gen_corpus.py: "
    "mamba_ssm's selective_scan_ref for Mamba-1, the chunked SSD reference for Mamba-2, the "
    "SISO reference for Mamba-3), not deployment kernels. On NVIDIA the mamba-ssm-* arms are "
    "mamba_ssm's own fused CUDA/Triton kernels (the deployment path); on AMD and Apple the "
    "Mamba lanes race the references only (NOT_PLANNED says why).",
    "The blocks' backward (the Mamba and TransformerBlock VJPs), their decode `step`, ragged "
    "`lengths` and the carried-state forward are public and not raced; only a zero-state "
    "forward is.",
    "SmallMLPTrainer.predict_logits (the GPU forward of the 8-16-3 MLP) is not raced; "
    "MLPInference (its CPU forward) and the training step are.",
    "The GPT-3-small target shape is not on the board (the LM lanes use the smaller control "
    "shape so one shape runs on every box, a 16 GB Mac included).",
]

# ---------------------------------------------------------------------------
# Shapes
# ---------------------------------------------------------------------------

#: [batch, length, d_model, n_heads, n_kv, head_dim, intermediate, n_layers, vocab]
LM_SHAPES = {
    "full": [1, 2048, 384, 6, 6, 64, 1024, 8, 8192],   # torch_lm_step_opponent.SHAPES['control']
    "small": [2, 64, 64, 4, 2, 16, 128, 2, 256],
}
LM_FIELDS = ("batch", "length", "d_model", "n_heads", "n_kv", "head_dim",
             "intermediate", "n_layers", "vocab_size")
GEMM_SHAPES = {"full": (4096, 4096, 4096), "small": (256, 256, 256)}   # m, n, k
BLOCK_SHAPES = {
    # the LM control shape's block
    "transformer": {"full": dict(batch=1, length=2048, d_model=384, n_heads=6, n_kv=6,
                                 head_dim=64, intermediate=1024),
                    "small": dict(batch=2, length=64, d_model=64, n_heads=4, n_kv=2,
                                  head_dim=16, intermediate=128)},
    "mamba1": {"full": dict(batch=1, length=2048, d_model=384),
               "small": dict(batch=2, length=64, d_model=16)},
    "mamba2": {"full": dict(batch=1, length=2048, d_model=384),
               "small": dict(batch=2, length=64, d_model=64)},
    "mamba3": {"full": dict(batch=1, length=2048, d_model=384),
               "small": dict(batch=2, length=64, d_model=64)},
}
SAMBA_SHAPES = {
    "full": dict(batch=2, length=512, vocab=256, d_model=384,
                 layers=("mamba3", "attention", "mamba3", "attention"),
                 n_heads=6, intermediate=1024),
    "small": dict(batch=2, length=64, vocab=256, d_model=64, layers=("mamba3", "attention"),
                  n_heads=2, intermediate=128),
}
MLP_SHAPES = {"full": dict(batch=256), "small": dict(batch=32)}
#: The CPU lanes' sequence cap in `full` (the host binding on a laptop CPU).
CPU_LENGTH_CAP = 512

#: The Mamba blocks' weight names, in each class's _W_NAMES order (our
#: worker constructs the block from them; the class refuses a wrong name or
#: shape by name).
MAMBA_NAMES = {
    "mamba1": ("norm.weight", "in_proj.weight", "conv1d.weight", "conv1d.bias",
               "x_proj.weight", "dt_proj.weight", "dt_proj.bias", "A_log", "D",
               "out_proj.weight"),
    "mamba2": ("block_norm.weight", "in_proj.weight", "conv1d.weight", "conv1d.bias",
               "dt_bias", "A_log", "D", "norm.weight", "out_proj.weight"),
    "mamba3": ("block_norm.weight", "in_proj.weight", "dt_bias", "B_norm.weight",
               "C_norm.weight", "B_bias", "C_bias", "D", "out_proj.weight"),
}
TRANSFORMER_NAMES = ("input_layernorm.weight", "post_attention_layernorm.weight",
                     "q_proj.weight", "k_proj.weight", "v_proj.weight", "o_proj.weight",
                     "gate_proj.weight", "up_proj.weight", "down_proj.weight")
#: our TransformerBlock names -> tools/speed_torch_seq.py LlamaEager's names
LLAMA_NAME = {"input_layernorm.weight": "norm1.weight",
              "post_attention_layernorm.weight": "norm2.weight"}
MLP_NAMES = ("weight1", "bias1", "weight2", "bias2")
MLP_DIMS = ((16, 8), (16,), (3, 16), (3,))


def opponents(vendor, lane):
    """The torch arms planned for (vendor, lane), in COLUMNS order."""
    if lane == "gemm-int8":
        return ("torch-eager-int8", "torch-compile-int8") if vendor == "nvidia" else ()
    if lane == "gemm-bf16":
        return tuple("torch-" + s for s in GPU_SETTINGS[vendor] if s.endswith("bf16"))
    if DEVICE_OF[lane] == "cpu":
        arms = ["torch-cpu-" + s for s in CPU_SETTINGS]
    else:
        arms = ["torch-" + s for s in GPU_SETTINGS[vendor]]
    if lane in NO_COMPILE:
        arms = [a for a in arms if "-compile-" not in a]
    if vendor in MAMBA_SSM_VENDORS and lane in MAMBA_SSM_LANES:
        arms += list(MAMBA_SSM_ARMS)
    return tuple(arms)


def arm_setting(arm):
    """'torch-cpu-eager-bf16' -> ('cpu', 'eager-bf16'); 'torch-compile-fp32' ->
    ('gpu', 'compile-fp32')."""
    if arm.startswith("torch-cpu-"):
        return "cpu", arm[len("torch-cpu-"):]
    if arm.startswith("torch-"):
        return "gpu", arm[len("torch-"):]
    raise ValueError("not a torch arm: %r" % arm)


def precision_text(setting):
    return {"int8": "int8 operands, int32 accumulate (torch._int_mm)",
            "fp32": "float32, TF32 off",
            "tf32": "float32 matmuls in TF32 (10-bit mantissa tensor cores)",
            "bf16": "bf16 autocast mixed precision (parameters, gradients and optimizer "
                    "state float32)"}[setting.split("-")[1]]


def lm_dims(lane, shape):
    """LM_SHAPES[shape], the length capped at CPU_LENGTH_CAP on a CPU lane (full)."""
    dims = list(LM_SHAPES[shape])
    if DEVICE_OF[lane] == "cpu" and shape == "full":
        dims[1] = min(dims[1], CPU_LENGTH_CAP)
    return dims


def _dims_of(lane, shape):
    model = MODEL_OF[lane]
    if model == "lm":
        return dict(zip(LM_FIELDS, lm_dims(lane, shape)))
    if model == "gemm":
        m, n, k = GEMM_SHAPES[shape]
        return {"m": m, "n": n, "k": k}
    if model == "samba":
        d = dict(SAMBA_SHAPES[shape])
    elif model == "mlp":
        d = dict(MLP_SHAPES[shape])
    else:
        d = dict(BLOCK_SHAPES[model][shape])
    if DEVICE_OF[lane] == "cpu" and "length" in d and shape == "full":
        d["length"] = min(d["length"], CPU_LENGTH_CAP)
    return d


def shape_record(lane, shape):
    """What the board shows as this race's shape, with the dimensions named."""
    d = _dims_of(lane, shape)
    model = MODEL_OF[lane]
    rec = dict(d, name=shape)
    if model == "lm":
        rec["label"] = "B%d L%d DM%d H%d KV%d HD%d FF%d layers%d V%d" % tuple(lm_dims(lane, shape))
    elif model == "gemm":
        rec["label"] = "%dx%dx%d" % (d["m"], d["n"], d["k"])
    elif model == "transformer":
        rec["label"] = "B%d L%d DM%d H%d KV%d HD%d FF%d" % (
            d["batch"], d["length"], d["d_model"], d["n_heads"], d["n_kv"], d["head_dim"],
            d["intermediate"])
    elif model == "samba":
        rec["layers"] = list(d["layers"])
        rec["label"] = "B%d L%d DM%d V%d H%d FF%d layers %s" % (
            d["batch"], d["length"], d["d_model"], d["vocab"], d["n_heads"], d["intermediate"],
            "+".join(d["layers"]))
    elif model == "mlp":
        rec["label"] = "rows%d 8-16-3" % d["batch"]
    else:
        rec["label"] = "B%d L%d DM%d" % (d["batch"], d["length"], d["d_model"])
    return rec


def shape_text(lane, shape):
    return shape_record(lane, shape)["label"] + (" (smoke)" if shape == "small" else "")


#: The board's per-lane settings text (what each side calls).
LANE_TEXT = {
    "lm-train-step": ("mojolearn.LanguageModelTrainer(resident=True, step_result='lean').train_step(ids)",
                      "tools/torch_lm_step_opponent.py build_model; zero_grad; forward + mean CE; "
                      "backward; torch.optim.AdamW step; loss.item()"),
    "lm-forward": ("mojolearn.LanguageModelTrainer(resident=True).logits(ids)",
                   "no_grad forward of the same twin to logits; logits.cpu()"),
    "gemm": ("mojolearn.linalg.matmul(a, b)", "a.to(dev) @ b.to(dev), .cpu()"),
    "gemm-bf16": ("mojolearn.linalg.matmul_bf16(a_bf16, b_bf16) (bf16 bits made before the clock)",
                  "a_bf16.to(dev) @ b_bf16.to(dev) under the setting, .float().cpu()"),
    "gemm-int8": ("mojolearn.linalg.matmul_int8((a_codes, 0), (b_codes, 0)) = a @ b.T exactly",
                  "torch._int_mm(a.to(dev), b.to(dev).t()), .float().cpu()"),
    "lm-infer": ("mojolearn.LanguageModelInference(parameters, shape=cfg).logits(ids) (CPU)",
                 "no_grad forward of the tools/torch_lm_step_opponent.py twin on the CPU"),
    "lm-host-train-step": ("mojolearn.LanguageModelHostTrainer(parameters, shape=cfg).train_step(ids) (CPU)",
                           "the same twin on the CPU; zero_grad; forward + mean CE; backward; "
                           "torch.optim.AdamW step; loss.item()"),
    "transformer-forward": ("mojolearn.TransformerBlock(weights, n_heads, n_kv_heads, head_dim).forward(x)",
                            "tools/speed_torch_seq.py LlamaEager.block(sdpa=True)"),
    "transformer-infer": ("mojolearn.TransformerBlockInference(...).forward(x) (CPU host binding)",
                          "LlamaEager.block(sdpa=True) on the CPU"),
    "mamba1-forward": ("mojolearn.Mamba1Block(weights).forward(x)",
                       "mamba/corpus/gen_corpus.py block_forward (pure-PyTorch reference scan)"),
    "mamba1-infer": ("mojolearn.Mamba1BlockInference(weights).forward(x) (CPU host binding)",
                     "gen_corpus.py block_forward on the CPU"),
    "mamba2-forward": ("mojolearn.Mamba2Block(weights).forward(x)",
                       "mamba/corpus/gen_corpus.py m2_forward (pure-PyTorch chunked SSD reference)"),
    "mamba2-infer": ("mojolearn.Mamba2BlockInference(weights).forward(x) (CPU host binding)",
                     "gen_corpus.py m2_forward on the CPU"),
    "mamba3-forward": ("mojolearn.Mamba3Block(weights).forward(x)",
                       "mamba/corpus/gen_corpus.py m3_forward (pure-PyTorch SISO reference)"),
    "mamba3-infer": ("mojolearn.Mamba3BlockInference(weights).forward(x) (CPU host binding)",
                     "gen_corpus.py m3_forward on the CPU"),
    "samba-train-step": ("mojolearn.SambaStack(config, weights).train_step(inputs, targets)",
                         "embedding + m3_forward / LlamaEager.block layers + RMSNorm + tied head; "
                         "mean CE; backward; torch.optim.AdamW step; loss.item()"),
    "samba-forward": ("mojolearn.SambaStack(config, weights).forward(inputs)",
                      "no_grad forward of the same stack twin to logits"),
    "samba-infer": ("mojolearn.SambaInference(config, weights).forward(inputs) (CPU host binding)",
                    "the same stack twin on the CPU"),
    "mlp-train-step": ("mojolearn.SmallMLPTrainer(w1, b1, w2, b2).train_step(X, y)",
                       "F.linear, ReLU, F.linear; mean CE; backward; torch.optim.AdamW step"),
    "mlp-infer": ("mojolearn.MLPInference(w1, b1, w2, b2).predict_logits(X) (CPU host binding)",
                  "F.linear, ReLU, F.linear on the CPU"),
}


def lane_settings(lane):
    """The board's settings block for one lane (bench_board.race_settings)."""
    ours, theirs = LANE_TEXT[lane]
    s = {"ours_call": ours, "torch_call": theirs, "ours_device": DEVICE_OF[lane],
         "clock": "host inputs to the device, the call, the result back on the host, "
                  "synchronized" + ("; training state device-resident on both sides; round r "
                                    "is step r+1" if lane in TRAIN_LANES else "")}
    s["seed_rule"] = ("seed %d: every parameter and input of every arm comes from the conductor's "
                 "default_rng(%d) file; torch arms also call torch.manual_seed(%d); ours' neural "
                 "classes take no seed argument (nothing in them draws)" % (SEED, SEED, SEED))
    s["explicit_params"] = _lane_params(lane)
    if lane in TRAIN_LANES:
        s["optimizer"] = ("AdamW lr 1e-3, betas (0.9, 0.999), eps 1e-8, weight decay 0.01, no "
                          "amsgrad, passed explicitly on both" + (", no clipping (max_norm=None)"
                                                                   if MODEL_OF[lane] == "samba" else ""))
        s["quality"] = ("loss_first_step, loss_last_step (same init, same batches), "
                        "loss_first_abs_diff_vs_ours, loss_last_abs_diff_vs_ours")
    else:
        s["quality"] = "max_abs_diff_vs_ours, max_rel_diff_vs_ours" + (
            ", mean_nll" if lane in ("lm-forward", "samba-forward", "samba-infer") else "") + (
            ", max_rel_err_vs_fp64" if MODEL_OF[lane] == "gemm" else "")
    s["torch_settings"] = {a: precision_text(arm_setting(a)[1]) for a in
                           ["torch-" + x for x in TORCH_SETTINGS]}
    return s


def now_utc():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _load(name, alias=None):
    path = os.path.join(HERE, name + ".py")
    spec = importlib.util.spec_from_file_location(alias or ("bbn_" + name), path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _load_mamba_corpus():
    """mamba/corpus/gen_corpus.py under its own alias (tools/speed_torch_seq.py
    _load_module's reason). It needs torch and numpy."""
    alias = "mojolearn_mamba_corpus"
    if alias in sys.modules:
        return sys.modules[alias]
    path = os.path.join(REPO, "mamba", "corpus", "gen_corpus.py")
    spec = importlib.util.spec_from_file_location(alias, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[alias] = mod
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# Inputs (the conductor, once per race)
# ---------------------------------------------------------------------------

def byte_stream():
    """The installed mojolearn package's .py sources, sorted, concatenated.
    find_spec locates the package WITHOUT importing it (no binding loads in
    the conductor)."""
    spec = importlib.util.find_spec("mojolearn")
    roots = list(spec.submodule_search_locations) if spec and spec.submodule_search_locations else []
    if not roots:
        raise SystemExit("bench_board_neural: mojolearn is not installed in %s" % sys.executable)
    root = roots[0]
    files = sorted(glob.glob(os.path.join(root, "**", "*.py"), recursive=True))
    raw = b"".join(open(f, "rb").read() for f in files)
    return raw, {"source": "installed mojolearn package .py sources, sorted by path, concatenated",
                 "package_dir": root, "files": len(files), "bytes": len(raw), "sha256": _sha(raw)}


def byte_batches(steps, bsz, length):
    import numpy as np
    raw, stream = byte_stream()
    if len(raw) < length + 2:
        raise SystemExit("bench_board_neural: byte stream shorter than one row")
    modulus = len(raw) - length - 1
    buf = np.frombuffer(raw, dtype=np.uint8)
    batches = np.empty((steps, bsz, length + 1), dtype=np.int32)
    for k in range(steps):
        for b in range(bsz):
            start = (k * bsz * length + b * length) % modulus
            batches[k, b] = buf[start:start + length + 1]
    return batches, stream


def lm_registry(dims):
    twin = _load("torch_lm_step_opponent")
    return twin.registry(dims)


def samba_registry(d):
    """SambaConfig.registry() for this shape, transcribed (python/mojolearn/
    _samba_impl.py block_shapes); our worker REFUSES unless the installed
    class answers the same list."""
    dm, vocab = d["d_model"], d["vocab"]
    out = [("embed.weight", (vocab, dm))]
    for i, kind in enumerate(d["layers"]):
        if kind == "mamba3":
            di = 2 * dm
            nh = di // 64
            dip = 2 * di + 2 * 128 + 3 * nh + 32
            shapes = {"block_norm.weight": (dm,), "in_proj.weight": (dip, dm), "dt_bias": (nh,),
                      "B_norm.weight": (128,), "C_norm.weight": (128,), "B_bias": (nh, 128),
                      "C_bias": (nh, 128), "D": (nh,), "out_proj.weight": (dm, di)}
            names = MAMBA_NAMES["mamba3"]
        else:
            hd = dm // d["n_heads"]
            qw = kw = d["n_heads"] * hd
            it = d["intermediate"]
            shapes = {"input_layernorm.weight": (dm,), "post_attention_layernorm.weight": (dm,),
                      "q_proj.weight": (qw, dm), "k_proj.weight": (kw, dm),
                      "v_proj.weight": (kw, dm), "o_proj.weight": (dm, qw),
                      "gate_proj.weight": (it, dm), "up_proj.weight": (it, dm),
                      "down_proj.weight": (dm, it)}
            names = TRANSFORMER_NAMES
        out.extend(("layers.%d.%s" % (i, n), shapes[n]) for n in names)
    out.append(("norm_f.weight", (dm,)))
    return out


def samba_init(rng, name, shape):
    """SambaStack's _init_tensor rules on numpy's generator."""
    import numpy as np
    last = name.split(".")[-1]
    if name in ("embed.weight", "lm_head.weight"):
        return rng.normal(0, .02, shape).astype(np.float32)
    if last == "weight" and len(shape) == 2:
        bound = 1.0 / np.sqrt(shape[1])
        return rng.uniform(-bound, bound, shape).astype(np.float32)
    if last == "dt_bias":
        return rng.uniform(-4.0, -2.0, shape).astype(np.float32)
    return np.ones(shape, dtype=np.float32)


def mamba_weight_spec(model, d):
    """[(name, shape, (lo, hi))] from mamba/corpus/gen_corpus.py's shape and
    range functions, plus x's."""
    corpus = _load_mamba_corpus()
    dm, B, L = d["d_model"], d["batch"], d["length"]
    if model == "mamba1":
        di, r = 2 * dm, -(-dm // 16)
        shapes = corpus.shapes_for(dm, di, r, 16, 4, B, L)
        ranges = corpus.default_ranges(dm, di, r, 16, 4)
    elif model == "mamba2":
        shapes, ranges = corpus.m2_shapes_for(dm, B, L), corpus.m2_default_ranges(dm)
    else:
        shapes, ranges = corpus.m3_shapes_for(dm, B, L), corpus.m3_default_ranges(dm)
    return [(n, tuple(shapes[n]), ranges[n]) for n in MAMBA_NAMES[model]], \
        (tuple(shapes["x"]), ranges["x"])


def make_inputs(lane, shape, steps, path):
    """Write the race's single input file (.npz) and return its record."""
    import numpy as np
    rng = np.random.default_rng(SEED)
    model = MODEL_OF[lane]
    d = _dims_of(lane, shape)
    rec = {"lane": lane, "shape": shape_record(lane, shape), "seed": SEED}
    arrays = {}
    if model == "gemm" and lane == "gemm-int8":
        arrays["a"] = rng.integers(-127, 128, (d["m"], d["k"])).astype(np.int8)
        arrays["b"] = rng.integers(-127, 128, (d["n"], d["k"])).astype(np.int8)
        rec["inputs"] = "A [m,k], B [n,k] = default_rng(%d).integers(-127, 128) int8; C = A B^T" % SEED
    elif model == "gemm":
        arrays["a"] = rng.standard_normal((d["m"], d["k"])).astype(np.float32)
        arrays["b"] = rng.standard_normal((d["k"], d["n"])).astype(np.float32)
        rec["inputs"] = "A [m,k], B [k,n] = default_rng(%d).standard_normal float32" % SEED
    elif model == "lm":
        dims = lm_dims(lane, shape)
        shapes = lm_registry(dims)
        n_total = sum(int(np.prod(s)) for _, s in shapes)
        flat = rng.normal(0, .02, n_total).astype(np.float32)
        off = 0
        for name, s in shapes:
            size = int(np.prod(s))
            if "norm" in name:
                flat[off:off + size] += np.float32(1)
            off += size
        if dims[8] < 256:
            raise SystemExit("bench_board_neural: byte ids need vocab >= 256")
        arrays["init"] = flat
        arrays["batches"], rec["stream"] = byte_batches(steps, dims[0], dims[1])
        rec.update(parameters=n_total, init="default_rng(%d).normal(0, .02) float32, +1 on norms" % SEED,
                   schedule="step k row b: stream[(k*B*L + b*L) % (n - L - 1) : +L+1]; "
                            "inputs [:, :L], targets [:, 1:]")
    elif model == "transformer":
        dm, hd, it = d["d_model"], d["head_dim"], d["intermediate"]
        qw, kw = d["n_heads"] * hd, d["n_kv"] * hd
        shapes = {"input_layernorm.weight": (dm,), "post_attention_layernorm.weight": (dm,),
                  "q_proj.weight": (qw, dm), "k_proj.weight": (kw, dm), "v_proj.weight": (kw, dm),
                  "o_proj.weight": (dm, qw), "gate_proj.weight": (it, dm),
                  "up_proj.weight": (it, dm), "down_proj.weight": (dm, it)}
        for n in TRANSFORMER_NAMES:
            w = rng.normal(0, .02, shapes[n]).astype(np.float32)
            if "layernorm" in n:
                w += np.float32(1)
            arrays["w:" + n] = w
        arrays["x"] = rng.standard_normal((d["batch"], d["length"], dm)).astype(np.float32)
        rec["inputs"] = "weights default_rng(%d).normal(0, .02) (+1 on norms), x standard normal" % SEED
    elif model in MAMBA_NAMES:
        spec, (xshape, xrange) = mamba_weight_spec(model, d)
        for n, s, (lo, hi) in spec:
            arrays["w:" + n] = rng.uniform(lo, hi, s).astype(np.float32)
        arrays["x"] = rng.uniform(xrange[0], xrange[1], xshape).astype(np.float32)
        rec["inputs"] = ("every tensor default_rng(%d).uniform over mamba/corpus/gen_corpus.py's "
                         "default range for it" % SEED)
        rec["ranges"] = {n: list(r) for n, _, r in spec}
    elif model == "samba":
        reg = samba_registry(d)
        for n, s in reg:
            arrays["w:" + n] = samba_init(rng, n, s)
        arrays["batches"], rec["stream"] = byte_batches(steps, d["batch"], d["length"])
        rec.update(parameters=sum(int(np.prod(s)) for _, s in reg),
                   init="SambaStack._init_tensor rules on default_rng(%d)" % SEED,
                   schedule="the LM lanes' byte schedule")
    elif model == "mlp":
        for n, s in zip(MLP_NAMES, MLP_DIMS):
            bound = 1.0 / np.sqrt(8 if n.endswith("1") else 16)
            arrays["w:" + n] = rng.uniform(-bound, bound, s).astype(np.float32)
        arrays["X"] = rng.standard_normal((steps, d["batch"], 8)).astype(np.float32)
        arrays["y"] = rng.integers(0, 3, (steps, d["batch"])).astype(np.int32)
        rec["inputs"] = "weights uniform(+-1/sqrt(fan_in)); X standard normal; labels 0..2"
    np.savez(path, **arrays)
    rec["sha256"] = {k: _sha(np.ascontiguousarray(v).tobytes())[:16] for k, v in sorted(arrays.items())}
    rec["steps"] = steps
    return rec


def _weights(data):
    return {k[2:]: data[k] for k in data if k.startswith("w:")}


# ---------------------------------------------------------------------------
# Workers: ours (one process per arm)
# ---------------------------------------------------------------------------

def _ours_module():
    want = os.environ.get("MOJOLEARN_NUMERIC_MODE", "identical").strip().lower()
    if want != "identical":
        raise RuntimeError("REFUSED: the neural surface is IDENTICAL only; this worker was started "
                           "under MOJOLEARN_NUMERIC_MODE=%s" % want)
    import mojolearn
    return mojolearn


def _ours_info(ml, module_path, mode_used, device="gpu"):
    info = {"library": "mojolearn", "version": getattr(ml, "__version__", "unknown"),
            "numeric_mode_env": os.environ.get("MOJOLEARN_NUMERIC_MODE"),
            "numeric_mode_used": mode_used, "device": device,
            "module_path": module_path, "pre_clock_fit": False,
            "input_home": "host"}
    try:
        info["vendor_used"] = ml.vendor()
    except Exception as exc:  # noqa: BLE001
        info["vendor_used"] = "unavailable (%r)" % (exc,)
    if mode_used != "identical":
        raise RuntimeError("ours is not IDENTICAL: read back %r" % (mode_used,))
    # an ours-cpu worker: the wheel must have loaded its CPU set (refuses by name)
    info.update(_load("bench_board_probe").ours_cpu_check(ml))
    return info


def _mode_of(obj, ml):
    """The tier read back from the binary the object holds, else the process's."""
    fn = getattr(obj, "numeric_mode_used", None)
    if callable(fn):
        try:
            return fn()
        except Exception:  # noqa: BLE001  (an *Inference class holds the host binding)
            pass
    return ml.numeric_mode()


def _lane_params(lane):
    """The tuning parameters this lane sets explicitly on EVERY arm (the
    board's parameter check compares them by their canonical names)."""
    model = MODEL_OF[lane]
    out = {}
    if lane in TRAIN_LANES:
        out.update(lr=ADAMW["lr"], betas=list(ADAMW["betas"]), eps=ADAMW["eps"],
                   weight_decay=ADAMW["weight_decay"], amsgrad=False)
    elif model == "transformer":
        out.update(eps=BLOCK_NORM_EPS, rope_theta=BLOCK_ROPE_THETA)
    elif model == "samba":
        out.update(eps=SAMBA_NORM_EPS, rope_theta=BLOCK_ROPE_THETA)
    if model == "samba":
        out.update(dropout=SAMBA_DROPOUT, max_grad_norm=None if lane in TRAIN_LANES else "n/a")
    return out


def _ours_record(lane, config_readback=None):
    """Ours' parameter record for tools/bench_board_params.py: the values
    passed to the constructor (a declared dict; our neural classes take no
    seed: every parameter and input comes from the conductor's seed-7 file)."""
    rec = {"__library__": "mojolearn"}
    rec.update(_lane_params(lane))
    if config_readback is not None:
        rec["config_readback"] = json.dumps(config_readback, sort_keys=True, default=str)
    return rec


class Ours:
    """Common shape of an `ours` runner: call / sync / outputs / digest."""
    out = None

    def sync(self):
        pass        # every public call returns host results (the binding syncs first)

    def outputs(self):
        return {"y": self.np.asarray(self.out)}

    def digest(self):
        return _sha(self.np.ascontiguousarray(self.np.asarray(self.out)).data)[:16]


class OursLM(Ours):
    """lm-train-step and lm-forward through LanguageModelTrainer."""

    def __init__(self, lane, shape, data):
        import numpy as np
        self.np = np
        ml = _ours_module()
        dims = lm_dims(lane, shape)
        cfg = ml.LanguageModelConfig(*dims)
        twin = lm_registry(dims)
        ours = [(e["name"], tuple(e["shape"])) for e in ml.LanguageModelTrainer.parameter_registry(cfg)]
        if ours != [(n, tuple(s)) for n, s in twin]:
            raise RuntimeError("REFUSED: our parameter registry differs from the torch twin's")
        self.lane = lane
        self.batches = data["batches"]
        if lane in ("lm-infer", "lm-host-train-step"):
            flat = np.ascontiguousarray(data["init"])
            if lane == "lm-infer":
                self.model = ml.LanguageModelInference(flat, shape=cfg)
            else:
                self.model = ml.LanguageModelHostTrainer(flat, shape=cfg, lr=ADAMW["lr"],
                                                         betas=ADAMW["betas"], eps=ADAMW["eps"],
                                                         weight_decay=ADAMW["weight_decay"])
            self.info = _ours_info(ml, getattr(sys.modules[type(self.model).__module__], "__file__",
                                               None), ml.numeric_mode(), "cpu")
            self.info.update(call=LANE_TEXT[lane][0], cls=type(self.model).__name__)
            self.k = 0
            self.losses = []
            self.ids = np.ascontiguousarray(self.batches[0][:, :-1])
            self.record = _ours_record(lane)
            return
        self.trainer = ml.LanguageModelTrainer(
            np.ascontiguousarray(data["init"]), shape=cfg, resident=True, step_result="lean",
            data_schedule={"fixture": "tools/bench_board_neural.py", "seed": SEED,
                           "batches": "installed mojolearn .py sources, board schedule"},
            lr=ADAMW["lr"], betas=ADAMW["betas"], eps=ADAMW["eps"],
            weight_decay=ADAMW["weight_decay"])
        meta = self.trainer.run_metadata()
        mode = "identical" if int(meta.get("native_numeric_mode", -1)) == 1 else \
            "native_numeric_mode=%s" % meta.get("native_numeric_mode")
        self.info = _ours_info(ml, meta.get("binding_file"), mode)
        self.info.update(profile=meta.get("native_profile"), native_vendor=meta.get("native_vendor"),
                         call=LANE_TEXT[lane][0], config=json.dumps(meta.get("config"), sort_keys=True))
        self.k = 0
        self.losses = []
        self.ids = np.ascontiguousarray(self.batches[0][:, :-1])
        self.record = _ours_record(lane, config_readback=meta.get("config"))

    def call(self):
        np = self.np
        if self.lane == "lm-train-step":
            res = self.trainer.train_step(np.ascontiguousarray(self.batches[self.k]))
            self.losses.append(float(res["loss"]))
            self.k += 1
        elif self.lane == "lm-host-train-step":
            bits = self.model.train_step(np.ascontiguousarray(self.batches[self.k]))
            self.losses.append(float(np.array([int(bits)], dtype=np.uint32).view(np.float32)[0]))
            self.k += 1
        elif self.lane == "lm-infer":
            self.out = self.model.logits(self.ids)
        else:
            self.out = self.trainer.logits(self.ids)

    def outputs(self):
        if self.lane in TRAIN_LANES:
            return {"losses": self.np.array(self.losses, dtype=self.np.float64)}
        return {"y": self.np.asarray(self.out)}

    def digest(self):
        return None if self.lane in TRAIN_LANES else Ours.digest(self)


class OursGEMM(Ours):
    def __init__(self, lane, shape, data):
        import numpy as np
        self.np = np
        ml = _ours_module()
        import mojolearn.linalg as linalg
        self.linalg = linalg
        self.lane = lane
        self.a, self.b = np.ascontiguousarray(data["a"]), np.ascontiguousarray(data["b"])
        if lane == "gemm-bf16":           # the bf16 bits, made before the clock
            self.a, self.b = linalg.to_bf16(self.a), linalg.to_bf16(self.b)
        elif lane == "gemm-int8":         # the codes with exponent 0: C = A B^T exactly
            self.a = (self.a, np.zeros(self.a.shape[0], dtype=np.int32))
            self.b = (self.b, np.zeros(self.b.shape[0], dtype=np.int32))
        self.info = _ours_info(ml, getattr(linalg, "__file__", None), linalg.numeric_mode())
        self.info.update(profile=linalg.PROFILE, call=LANE_TEXT[lane][0])
        self.record = _ours_record(lane)

    def call(self):
        fn = {"gemm": self.linalg.matmul, "gemm-bf16": self.linalg.matmul_bf16,
              "gemm-int8": self.linalg.matmul_int8}[self.lane]
        self.out = fn(self.a, self.b)


class OursBlock(Ours):
    """transformer-* and mamba*-*: one block's zero-state forward."""

    def __init__(self, lane, shape, data):
        import numpy as np
        self.np = np
        ml = _ours_module()
        model, cpu = MODEL_OF[lane], DEVICE_OF[lane] == "cpu"
        d = _dims_of(lane, shape)
        w = {k: np.ascontiguousarray(v) for k, v in _weights(data).items()}
        if model == "transformer":
            cls = ml.TransformerBlockInference if cpu else ml.TransformerBlock
            self.block = cls(w, n_heads=d["n_heads"], n_kv_heads=d["n_kv"], head_dim=d["head_dim"],
                             norm_eps=BLOCK_NORM_EPS, rope_theta=BLOCK_ROPE_THETA)
        else:
            name = {"mamba1": "Mamba1Block", "mamba2": "Mamba2Block", "mamba3": "Mamba3Block"}[model]
            cls = getattr(ml, name + ("Inference" if cpu else ""))
            self.block = cls(w)
        self.x = np.ascontiguousarray(data["x"])
        self.info = _ours_info(ml, getattr(sys.modules[cls.__module__], "__file__", None),
                               _mode_of(self.block, ml), "cpu" if cpu else "gpu")
        self.info.update(call=LANE_TEXT[lane][0], cls=cls.__name__)
        self.record = _ours_record(lane)
        if lane in MAMBA_SSM_LANES:
            # the block's structural constants, read from the INSTALLED class's
            # module (and the block), compared with what the mamba-ssm arms'
            # constructed modules hold (tools/bench_board_params.py ssm_*)
            self.record.update(ours_ssm_record(model, sys.modules[cls.__module__], self.block))

    def call(self):
        self.out = self.block.forward(self.x)


def _floats(v):
    return None if v is None else [float(x) for x in v]


def ours_ssm_record(model, mod, block):
    """Ours' Mamba structural constants under the canonical ssm_* names:
    the profile constants of the installed mojolearn._mamba_impl and the
    block's own attributes."""
    if model == "mamba1":
        return {"ssm_d_state": mod._M1_D_STATE, "ssm_d_conv": mod._M1_D_CONV,
                "ssm_expand": mod._M1_EXPAND, "ssm_dt_rank": getattr(block, "dt_rank", None)}
    if model == "mamba2":
        return {"ssm_d_state": mod._M2_D_STATE, "ssm_d_conv": mod._M2_D_CONV,
                "ssm_expand": mod._M2_EXPAND, "ssm_headdim": mod._M2_HEADDIM,
                "ssm_ngroups": mod._M2_NGROUPS, "ssm_chunk_size": mod._M2_CHUNK_SIZE,
                "ssm_dt_limit": _floats(getattr(block, "dt_limit", None))}
    return {"ssm_d_state": mod._M3_D_STATE, "ssm_expand": mod._M3_EXPAND,
            "ssm_headdim": mod._M3_HEADDIM, "ssm_ngroups": mod._M3_NGROUPS,
            "ssm_chunk_size": mod._M3_CHUNK_SIZE, "ssm_rope_angles": mod._M3_NUM_ROPE_ANGLES}


class OursSamba(Ours):
    def __init__(self, lane, shape, data):
        import numpy as np
        self.np = np
        ml = _ours_module()
        d = _dims_of(lane, shape)
        cfg = ml.SambaConfig(d["vocab"], d["d_model"], d["layers"], n_heads=d["n_heads"],
                             intermediate=d["intermediate"], norm_eps=SAMBA_NORM_EPS,
                             dropout=SAMBA_DROPOUT)
        ours = [(n, tuple(s)) for n, s in cfg.registry()]
        if ours != [(n, tuple(s)) for n, s in samba_registry(d)]:
            raise RuntimeError("REFUSED: mojolearn.SambaConfig's registry differs from the "
                               "conductor's (tools/bench_board_neural.py samba_registry)")
        w = {k: np.ascontiguousarray(v) for k, v in _weights(data).items()}
        self.lane = lane
        if lane == "samba-infer":
            self.model = ml.SambaInference(cfg, w)
            mode, dev = ml.numeric_mode(), "cpu"
        else:
            self.model = ml.SambaStack(cfg, weights=w, lr=ADAMW["lr"], betas=ADAMW["betas"],
                                       eps=ADAMW["eps"], weight_decay=ADAMW["weight_decay"],
                                       max_norm=None)
            mode, dev = _mode_of((getattr(self.model, "_blocks", None) or [None])[0], ml), "gpu"
        self.info = _ours_info(ml, getattr(sys.modules[type(self.model).__module__], "__file__", None),
                               mode, dev)
        self.info.update(call=LANE_TEXT[lane][0], config=json.dumps(cfg.to_dict(), sort_keys=True))
        self.record = _ours_record(lane, config_readback=cfg.to_dict())
        self.batches = data["batches"]
        self.ids = np.ascontiguousarray(self.batches[0][:, :-1])
        self.k = 0
        self.losses = []

    def call(self):
        np = self.np
        if self.lane == "samba-train-step":
            b = self.batches[self.k]
            res = self.model.train_step(np.ascontiguousarray(b[:, :-1]), np.ascontiguousarray(b[:, 1:]))
            self.losses.append(float(res["loss"]))
            self.k += 1
        else:
            self.out = self.model.forward(self.ids)

    def outputs(self):
        if self.lane == "samba-train-step":
            return {"losses": self.np.array(self.losses, dtype=self.np.float64)}
        return {"y": self.np.asarray(self.out)}

    def digest(self):
        return None if self.lane == "samba-train-step" else Ours.digest(self)


class OursMLP(Ours):
    def __init__(self, lane, shape, data):
        import numpy as np
        self.np = np
        ml = _ours_module()
        w = [np.ascontiguousarray(data["w:" + n]) for n in MLP_NAMES]
        self.lane = lane
        self.X, self.y = data["X"], data["y"]
        if lane == "mlp-infer":
            self.model = ml.MLPInference(*w)
            mode, dev = ml.numeric_mode(), "cpu"
        else:
            self.model = ml.SmallMLPTrainer(*w, data_schedule={"fixture": "tools/bench_board_neural.py",
                                                               "seed": SEED},
                                            lr=ADAMW["lr"], betas=ADAMW["betas"], eps=ADAMW["eps"],
                                            weight_decay=ADAMW["weight_decay"])
            mode, dev = ml.numeric_mode(), "gpu"
        self.info = _ours_info(ml, getattr(sys.modules[type(self.model).__module__], "__file__", None),
                               mode, dev)
        self.info.update(call=LANE_TEXT[lane][0])
        self.record = _ours_record(lane)
        self.x0 = np.ascontiguousarray(self.X[0])
        self.k = 0
        self.losses = []

    def call(self):
        np = self.np
        if self.lane == "mlp-train-step":
            res = self.model.train_step(np.ascontiguousarray(self.X[self.k]),
                                        np.ascontiguousarray(self.y[self.k]))
            self.losses.append(float(res["loss"]))
            self.k += 1
        else:
            self.out = self.model.predict_logits(self.x0)

    def outputs(self):
        if self.lane == "mlp-train-step":
            return {"losses": self.np.array(self.losses, dtype=self.np.float64)}
        return {"y": self.np.asarray(self.out)}

    def digest(self):
        return None if self.lane == "mlp-train-step" else Ours.digest(self)


OURS = {"lm": OursLM, "gemm": OursGEMM, "transformer": OursBlock, "mamba1": OursBlock,
        "mamba2": OursBlock, "mamba3": OursBlock, "samba": OursSamba, "mlp": OursMLP}


# ---------------------------------------------------------------------------
# Workers: torch (one process per arm, one setting each)
# ---------------------------------------------------------------------------

def _torch_device():
    import torch
    if torch.cuda.is_available():
        return torch, torch.device("cuda"), "cuda", torch.cuda.get_device_name(0), torch.cuda.synchronize
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return torch, torch.device("mps"), "mps", "Apple MPS", torch.mps.synchronize
    raise RuntimeError("REFUSED: torch sees no CUDA, ROCm or MPS device; no GPU arm on this box "
                       "(never a CPU fallback)")


def _cpu_name():
    try:
        if sys.platform == "darwin":
            import subprocess
            return subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True,
                                  text=True, timeout=10).stdout.strip() or platform.processor()
        with open("/proc/cpuinfo") as fh:
            for line in fh:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except Exception:  # noqa: BLE001
        pass
    return platform.processor() or "cpu"


def _rms(torch, x, w, eps):
    return w * (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps))


class TorchArm:
    """One torch setting on one lane. The lane builder gives `fn(*device
    inputs) -> loss or output` (compiled when the setting says so); the clock
    covers host inputs to the device, fn under the setting's autocast, the
    result back on the host."""

    def __init__(self, lane, shape, data, arm):
        import numpy as np
        self.np, self.lane, self.shape, self.data = np, lane, shape, data
        where, setting = arm_setting(arm)
        if where != DEVICE_OF[lane]:
            raise RuntimeError("REFUSED: arm %s runs on the %s and lane %s's class on the %s"
                               % (arm, where, lane, DEVICE_OF[lane]))
        twin = _load("torch_lm_step_opponent")
        self.twin = twin
        col = INT8_COLUMNS.get(setting.replace("-", "_")) or twin.COLUMNS[setting.replace("-", "_")]
        if where == "cpu":
            import torch
            dev, kind, name, sync = torch.device("cpu"), "cpu", _cpu_name(), (lambda: None)
        else:
            torch, dev, kind, name, sync = _torch_device()
        self.torch, self.dev, self.kind, self._sync = torch, dev, kind, sync
        # the board's seed (rule: same seed on every arm). Every weight and
        # input comes from the conductor's seed-7 file; nothing here draws.
        torch.manual_seed(SEED)
        hip = getattr(torch.version, "hip", None)
        if col["tf32"] and not (kind == "cuda" and not hip):
            raise RuntimeError("REFUSED: %s: TF32 is an NVIDIA CUDA matmul mode; torch on %s "
                               "accepts the flag and changes nothing" % (arm, "ROCm" if hip else kind))
        precision = twin.set_precision(torch, col["tf32"])
        self.autocast = None
        probe = None
        if col["autocast"]:
            probe = twin.probe_autocast(torch, dev, col["autocast"], sync)
            if not probe.get("applicable"):
                raise RuntimeError("REFUSED: %s: torch.autocast(%r, %s) does not work on this box: %s"
                                   % (arm, dev.type, col["autocast"], json.dumps(probe)[:600]))
            self.autocast = getattr(torch, col["autocast"])
        self.compile = bool(col["compile"])
        self.info = {"library": "torch", "version": torch.__version__,
                     "torch_version_cuda": torch.version.cuda, "torch_version_hip": hip,
                     "torch_backend": kind, "device": where, "device_name": name,
                     "setting": setting, "precision": precision_text(setting),
                     "compile": "torch.compile (inductor, default mode)" if self.compile else "eager",
                     "mode": "torch %s, %s" % ("compile" if self.compile else "eager",
                                               precision_text(setting)),
                     "precision_readback": precision, "autocast_probe": probe,
                     "column": setting.replace("-", "_") + " (tools/torch_lm_step_opponent.py COLUMNS)",
                     "torch_threads": torch.get_num_threads(),
                     "pre_clock_fit": False, "input_home": "host", "call": LANE_TEXT[lane][1]}
        self.k = 0
        self.losses = []
        self.out = None
        self.train = lane in TRAIN_LANES
        getattr(self, "_build_" + MODEL_OF[lane])()
        if (twin.LR, tuple(twin.BETAS), twin.ADAM_EPS, twin.WEIGHT_DECAY) != (
                ADAMW["lr"], tuple(ADAMW["betas"]), ADAMW["eps"], ADAMW["weight_decay"]):
            raise RuntimeError("REFUSED: tools/torch_lm_step_opponent.py's AdamW constants %r "
                               "differ from the board's %r" % (
                                   (twin.LR, twin.BETAS, twin.ADAM_EPS, twin.WEIGHT_DECAY), ADAMW))
        rec = {"__library__": "torch", "seed": SEED}
        if self.train:
            self.opt = torch.optim.AdamW(self.module.parameters(), lr=ADAMW["lr"], betas=ADAMW["betas"],
                                         eps=ADAMW["eps"], weight_decay=ADAMW["weight_decay"],
                                         amsgrad=False)
            self.info["optimizer"] = "torch.optim.AdamW lr %g betas %s eps %g wd %g (torch's default impl)" % (
                ADAMW["lr"], ADAMW["betas"], ADAMW["eps"], ADAMW["weight_decay"])
            # read back from the optimizer itself
            back = _load("bench_board_params").arm_record(self.opt)["params"]
            rec.update(back)
            if "betas" in rec:
                rec["betas"] = list(rec["betas"])
        else:
            rec.update(self.block_consts)
        if MODEL_OF[lane] == "samba":
            rec.update(dropout=SAMBA_DROPOUT, max_grad_norm=None if self.train else "n/a")
        self.record = rec
        self.fn = torch.compile(self.module) if self.compile else self.module
        sync()

    # -- the device context (the mamba references allocate without a device)
    def _ctx(self):
        import contextlib
        torch = self.torch
        stack = contextlib.ExitStack()
        if self.uses_default_device:
            stack.enter_context(torch.device(self.dev))
        if self.autocast is not None:
            stack.enter_context(torch.autocast(device_type=self.dev.type, dtype=self.autocast))
        if not self.train:
            stack.enter_context(torch.no_grad())
        return stack

    uses_default_device = False
    #: the forward twin's constants the lane compares (norm eps, RoPE base)
    block_consts = {}

    def _module(self, params, forward):
        """An nn.Module holding `params` (name -> tensor, registry order) whose
        forward is forward(dict of parameters, *inputs)."""
        torch = self.torch
        names = list(params)

        class Twin(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.plist = torch.nn.ParameterList(
                    [torch.nn.Parameter(params[n], requires_grad=True) for n in names])

            def forward(self, *inputs):
                return forward(dict(zip(names, self.plist)), *inputs)

        return Twin()

    def _t(self, a):
        return self.torch.from_numpy(self.np.ascontiguousarray(a))

    def _dev_params(self):
        return {k: self._t(v).to(self.dev) for k, v in _weights(self.data).items()}

    # -- lane builders -----------------------------------------------------
    def _build_gemm(self):
        torch = self.torch
        self.host = [self._t(self.data["a"]), self._t(self.data["b"])]
        if self.lane == "gemm-bf16":
            self.host = [h.to(torch.bfloat16) for h in self.host]
        if self.lane == "gemm-int8":
            if self.kind != "cuda" or getattr(torch.version, "hip", None):
                raise RuntimeError("REFUSED: torch._int_mm is a CUDA kernel; not on %s" % self.kind)
            self.module = self._module({}, lambda p, a, b: torch._int_mm(a, b.t()))
            return
        self.module = self._module({}, lambda p, a, b: a @ b)

    def _build_lm(self):
        torch, np, twin = self.torch, self.np, self.twin
        dims = lm_dims(self.lane, self.shape)
        flat = torch.from_numpy(np.ascontiguousarray(self.data["init"]))
        model = twin.build_model(torch, dims, twin.registry(dims), flat, self.dev)
        n = sum(p.numel() for p in model.parameters())
        if n != flat.numel():
            raise RuntimeError("REFUSED: torch model has %d parameters, the input %d" % (n, flat.numel()))
        if self.kind == "cuda":
            self.info["sdpa"] = twin.choose_sdpa_backend(
                torch, "auto", self.dev, dims, self._sync,
                self.info["setting"].split("-")[1] == "bf16" and "bfloat16" or None)
        else:
            self.info["sdpa"] = {"backend": "torch_default",
                                 "selection": "%s: torch's own dispatch (no backend switches exist)"
                                              % self.kind}
        self.batches = self.data["batches"]
        if self.train:
            self.module = model
        else:
            model.eval()

            class Logits(torch.nn.Module):
                def __init__(self):
                    super().__init__()
                    self.m = model

                def forward(self, ids):
                    return self.m(ids, return_logits=True)

            self.module = Logits()
            self.host = [torch.from_numpy(np.ascontiguousarray(self.batches[0][:, :-1]).astype(np.int64))]

    def _llama(self, W, d, length):
        """tools/speed_torch_seq.py's LlamaEager over tensors named as ours."""
        seq = _load_speed_torch_seq(self.torch, self.info)
        if (float(seq.RMS_EPS), float(seq.ROPE_THETA)) != (BLOCK_NORM_EPS, BLOCK_ROPE_THETA):
            raise RuntimeError("REFUSED: tools/speed_torch_seq.py's RMS_EPS %r / ROPE_THETA %r differ "
                               "from ours' norm_eps %r / rope_theta %r" % (
                                   seq.RMS_EPS, seq.ROPE_THETA, BLOCK_NORM_EPS, BLOCK_ROPE_THETA))
        self.block_consts = {"eps": float(seq.RMS_EPS), "rope_theta": float(seq.ROPE_THETA)}
        cfg = dict(n_heads=d["n_heads"], n_kv=d["n_kv"], head_dim=d["head_dim"],
                   intermediate=d["intermediate"], d_model=d["d_model"], ctx=0, l=length)
        return seq.LlamaEager(self.torch, self.dev, cfg, {LLAMA_NAME.get(k, k): v for k, v in W.items()},
                              self.torch.float32)

    def _build_transformer(self):
        d = _dims_of(self.lane, self.shape)
        B, L, dm = d["batch"], d["length"], d["d_model"]
        params = self._dev_params()
        blk = self._llama(params, d, L)

        def forward(p, x):
            blk.W = {LLAMA_NAME.get(k, k): v for k, v in p.items()}
            return blk.block(x.reshape(B * L, dm), None, B, L, sdpa=True)[0].reshape(B, L, dm)

        self.module = self._module(params, forward)
        self.host = [self._t(self.data["x"])]
        self.info["twin"] = "tools/speed_torch_seq.py LlamaEager.block(sdpa=True)"

    def _mamba_call(self, model):
        corpus = _load_mamba_corpus_restoring(self.torch, self.info)
        f32 = self.torch.float32
        if model == "mamba1":
            return lambda p, x: corpus.block_forward(p, x, f32)["block.out"]
        if model == "mamba2":
            return lambda p, x: corpus.m2_forward(p, x, f32)["residual.out"].reshape(x.shape)
        return lambda p, x: corpus.m3_forward(p, x, f32)["residual.out"].reshape(x.shape)

    def _build_mamba(self):
        model = MODEL_OF[self.lane]
        call = self._mamba_call(model)
        self.uses_default_device = True
        self.module = self._module(self._dev_params(), call)
        self.host = [self._t(self.data["x"])]
        self.info["twin"] = {"mamba1": "mamba/corpus/gen_corpus.py block_forward (selective_scan_ref)",
                             "mamba2": "mamba/corpus/gen_corpus.py m2_forward",
                             "mamba3": "mamba/corpus/gen_corpus.py m3_forward"}[model]

    _build_mamba1 = _build_mamba2 = _build_mamba3 = _build_mamba

    def _build_samba(self):
        torch = self.torch
        F = torch.nn.functional
        d = _dims_of(self.lane, self.shape)
        B, L, dm, V = d["batch"], d["length"], d["d_model"], d["vocab"]
        m3 = self._mamba_call("mamba3")
        params = self._dev_params()
        hd = dm // d["n_heads"]
        attn = {}
        for i, kind in enumerate(d["layers"]):
            if kind == "attention":
                pre = "layers.%d." % i
                attn[i] = self._llama({k[len(pre):]: v for k, v in params.items() if k.startswith(pre)},
                                      dict(d, n_kv=d["n_heads"], head_dim=hd), L)
        eps = float(torch.tensor(SAMBA_NORM_EPS, dtype=torch.float32))
        # the final norm's eps as passed to ours (the blocks' eps is read back in _llama)
        self.block_consts = dict(self.block_consts, eps=SAMBA_NORM_EPS)
        train = self.train

        def forward(p, ids, targets=None):
            x = F.embedding(ids, p["embed.weight"])
            for i, kind in enumerate(d["layers"]):
                pre = "layers.%d." % i
                w = {k[len(pre):]: v for k, v in p.items() if k.startswith(pre)}
                if kind == "mamba3":
                    x = m3(w, x)
                else:
                    blk = attn[i]
                    blk.W = {LLAMA_NAME.get(k, k): v for k, v in w.items()}
                    x = blk.block(x.reshape(B * L, dm), None, B, L, sdpa=True)[0].reshape(B, L, dm)
            h = _rms(torch, x, p["norm_f.weight"], eps)
            logits = F.linear(h, p["embed.weight"])
            if not train:
                return logits
            return F.cross_entropy(logits.reshape(B * L, V).float(), targets.reshape(B * L),
                                   reduction="mean")

        self.uses_default_device = True
        self.module = self._module(params, forward)
        self.batches = self.data["batches"]
        if not train:
            self.host = [torch.from_numpy(self.np.ascontiguousarray(self.batches[0][:, :-1]).astype(self.np.int64))]
        self.info["twin"] = ("embedding; per layer mamba/corpus/gen_corpus.py m3_forward or "
                             "tools/speed_torch_seq.py LlamaEager.block(sdpa=True); RMSNorm eps "
                             "1e-5; tied head; mean CE")

    def _build_mlp(self):
        torch = self.torch
        F = torch.nn.functional
        train = self.train

        def forward(p, x, y=None):
            logits = F.linear(F.relu(F.linear(x, p["weight1"], p["bias1"])), p["weight2"], p["bias2"])
            if not train:
                return logits
            return F.cross_entropy(logits.float(), y, reduction="mean")

        self.module = self._module(self._dev_params(), forward)
        if not train:
            self.host = [self._t(self.data["X"][0])]
        self.info["twin"] = "F.linear, ReLU, F.linear (mean CE, AdamW for the step)"

    # -- the protocol ------------------------------------------------------
    def _train_inputs(self):
        np, torch = self.np, self.torch
        model = MODEL_OF[self.lane]
        if model == "lm":
            return [torch.from_numpy(self.batches[self.k].astype(np.int64))]
        if model == "samba":
            b = self.batches[self.k].astype(np.int64)
            return [torch.from_numpy(np.ascontiguousarray(b[:, :-1])),
                    torch.from_numpy(np.ascontiguousarray(b[:, 1:]))]
        return [self._t(self.data["X"][self.k]), torch.from_numpy(self.data["y"][self.k].astype(np.int64))]

    def call(self):
        if self.train:
            inputs = [t.to(self.dev) for t in self._train_inputs()]
            self.opt.zero_grad(set_to_none=True)
            with self._ctx():
                loss = self.fn(*inputs)
            loss.backward()
            self.opt.step()
            self.losses.append(float(loss.item()))
            self.k += 1
        else:
            inputs = [t.to(self.dev) for t in self.host]
            with self._ctx():
                out = self.fn(*inputs)
            self.out = out.float().cpu().numpy()

    def sync(self):
        self._sync()

    def outputs(self):
        if self.train:
            return {"losses": self.np.array(self.losses, dtype=self.np.float64)}
        return {"y": self.out}

    def digest(self):
        return None if self.train else _sha(self.np.ascontiguousarray(self.out).data)[:16]


def _restore_torch(torch, threads, info):
    """gen_corpus.py switches torch to deterministic algorithms and one thread
    at import; an opponent runs at torch's defaults (tools/speed_torch_seq.py
    DEVIATION 1856). Read back and recorded."""
    torch.use_deterministic_algorithms(False)
    torch.set_num_threads(threads)
    info["corpus_import_switches_restored"] = {
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "num_threads": torch.get_num_threads()}


def _load_mamba_corpus_restoring(torch, info):
    threads = torch.get_num_threads()
    mod = _load_mamba_corpus()
    _restore_torch(torch, threads, info)
    return mod


def _load_speed_torch_seq(torch, info):
    """tools/speed_torch_seq.py (it loads the mamba corpus at import and sets
    OMP/OPENBLAS/MKL thread variables when unset); the switches are put back."""
    if "bbn_speed_torch_seq" in sys.modules:
        _restore_torch(torch, torch.get_num_threads(), info)
        return sys.modules["bbn_speed_torch_seq"]
    threads = torch.get_num_threads()
    env = {k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")}
    mod = _load("speed_torch_seq", "bbn_speed_torch_seq")
    sys.modules["bbn_speed_torch_seq"] = mod
    for k, v in env.items():
        if v is None:
            os.environ.pop(k, None)
    _restore_torch(torch, threads, info)
    return mod


# ---------------------------------------------------------------------------
# Workers: mamba_ssm's fused kernels (MAMBA_SSM_ARMS)
# ---------------------------------------------------------------------------

#: The mamba_ssm modules each model runs, and our weight names -> the Block's
#: state-dict names (block.norm = the block RMSNorm, block.mixer = the mixer).
MAMBA_SSM_MIXER = {"mamba1": "mamba_ssm.modules.mamba_simple.Mamba",
                   "mamba2": "mamba_ssm.modules.mamba2.Mamba2",
                   "mamba3": "mamba_ssm.modules.mamba3.Mamba3"}
MAMBA_SSM_BLOCK_NORM = {"mamba1": "norm.weight", "mamba2": "block_norm.weight",
                        "mamba3": "block_norm.weight"}


def mamba_ssm_state_name(model, name):
    """Our weight name -> the mamba_ssm Block's state-dict name."""
    return "norm.weight" if name == MAMBA_SSM_BLOCK_NORM[model] else "mixer." + name


def mamba_ssm_setting(arm):
    """'mamba-ssm-fp32' -> {'tf32': False, 'triton_f32_default': 'ieee'}."""
    if arm not in MAMBA_SSM_ARMS:
        raise ValueError("not a mamba-ssm arm: %r" % arm)
    tf32 = arm.endswith("-tf32")
    return {"tf32": tf32, "triton_f32_default": "tf32" if tf32 else "ieee"}


def mamba_ssm_config(model, corpus, dm):
    """The mixer's constructor arguments (every one EXPLICIT, never the
    library default) and the block norm's eps, from mamba/corpus/gen_corpus.py:
    the same constants our blocks and the torch references use."""
    if model == "mamba1":
        return dict(d_state=corpus.D_STATE, d_conv=corpus.D_CONV, expand=corpus.EXPAND,
                    dt_rank=-(-dm // 16), conv_bias=True, bias=False,
                    use_fast_path=True), corpus.EPS
    if model == "mamba2":
        return dict(d_state=corpus.M2_D_STATE, d_conv=corpus.M2_D_CONV, expand=corpus.M2_EXPAND,
                    headdim=corpus.M2_HEADDIM, ngroups=corpus.M2_NGROUPS, D_has_hdim=False,
                    rmsnorm=True, norm_before_gate=False, dt_limit=(0.0, float("inf")),
                    bias=False, conv_bias=True, chunk_size=corpus.M2_CHUNK,
                    use_mem_eff_path=True), corpus.M2_EPS
    return dict(d_state=corpus.M3_D_STATE, expand=corpus.M3_EXPAND, headdim=corpus.M3_HEADDIM,
                ngroups=corpus.M3_NGROUPS, rope_fraction=0.5, A_floor=corpus.M3_A_FLOOR,
                is_outproj_norm=False, is_mimo=False,
                chunk_size=corpus.M3_CHUNK), corpus.M3_EPS


def mamba_ssm_readback(model, block):
    """The constructed Block's structural values under the canonical ssm_*
    names (compared with ours' by tools/bench_board_params.py) and the
    norms' eps (checked against the corpus here)."""
    m = block.mixer
    rec = {"ssm_d_state": int(m.d_state), "ssm_expand": int(m.expand)}
    eps = {"block_norm_eps": float(block.norm.eps)}
    if model == "mamba1":
        rec.update(ssm_d_conv=int(m.d_conv), ssm_dt_rank=int(m.dt_rank))
    elif model == "mamba2":
        rec.update(ssm_d_conv=int(m.d_conv), ssm_headdim=int(m.headdim),
                   ssm_ngroups=int(m.ngroups), ssm_chunk_size=int(m.chunk_size),
                   ssm_dt_limit=[float(v) for v in m.dt_limit])
        eps["gated_norm_eps"] = float(m.norm.eps)
    else:
        rec.update(ssm_headdim=int(m.headdim), ssm_ngroups=int(m.num_bc_heads),
                   ssm_chunk_size=int(m.chunk_size), ssm_rope_angles=int(m.num_rope_angles))
        eps.update(B_norm_eps=float(m.B_norm.eps), C_norm_eps=float(m.C_norm.eps))
    return rec, eps


def _mamba_ssm_source():
    """(version text, pip's direct_url record) of the installed mamba_ssm: a
    git build's version carries the commit (the board pins one)."""
    import importlib.metadata as md
    import mamba_ssm
    ver = str(getattr(mamba_ssm, "__version__", "unknown"))
    url = None
    for dist in ("mamba_ssm", "mamba-ssm"):
        try:
            raw = md.distribution(dist).read_text("direct_url.json")
        except md.PackageNotFoundError:
            continue
        if raw:
            url = json.loads(raw)
            break
    commit = ((url or {}).get("vcs_info") or {}).get("commit_id")
    return (ver + "+g" + commit[:12] if commit else ver), url


def _dist_version(name):
    import importlib.metadata as md
    try:
        return md.version(name)
    except md.PackageNotFoundError:
        return None


class MambaSsmArm:
    """mamba_ssm's fused kernels on one Mamba forward lane: Block(norm =
    Triton RMSNorm, mixer, fused_add_norm, residual in fp32); the output is
    mixer(norm(x)) + x, the block's residual add (the next layer's fused
    add-norm does it inside a mamba_ssm model). The clock: host x to the
    device, the block, the output back on the host, synchronized (the torch
    arms' clock)."""

    def __init__(self, lane, shape, data, arm):
        import numpy as np
        self.np, self.lane = np, lane
        if lane not in MAMBA_SSM_LANES:
            raise RuntimeError("REFUSED: %s races the Mamba forward lanes only (%s), not %s"
                               % (arm, ", ".join(MAMBA_SSM_LANES), lane))
        setting = mamba_ssm_setting(arm)
        # the Triton fp32 dot precision, set before any kernel compiles
        os.environ["TRITON_F32_DEFAULT"] = setting["triton_f32_default"]
        torch, dev, kind, name, sync = _torch_device()
        hip = getattr(torch.version, "hip", None)
        if kind != "cuda" or hip:
            raise RuntimeError("REFUSED: %s: mamba_ssm's CUDA extensions and Triton kernels are "
                               "raced on NVIDIA CUDA only; this torch is %s" % (arm, "ROCm" if hip else kind))
        self.torch, self.dev, self._sync = torch, dev, sync
        twin = _load("torch_lm_step_opponent")
        precision = twin.set_precision(torch, setting["tf32"])
        torch.manual_seed(SEED)
        info = {}
        corpus = _load_mamba_corpus_restoring(torch, info)
        try:
            import mamba_ssm  # noqa: F401
            from functools import partial
            from mamba_ssm.modules.block import Block
            from mamba_ssm.ops.triton.layer_norm import RMSNorm
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError("REFUSED: %s: mamba_ssm does not import in %s: %r"
                               % (arm, sys.executable, exc))
        model = MODEL_OF[lane]
        d = _dims_of(lane, shape)
        cfg, eps = mamba_ssm_config(model, corpus, d["d_model"])
        mod_name, cls_name = MAMBA_SSM_MIXER[model].rsplit(".", 1)
        mixer_mod = importlib.import_module(mod_name)
        mixer_cls = getattr(mixer_mod, cls_name)
        # THE FUSED PATH OR NOTHING: an arm that would fall back to a slower
        # path (no causal_conv1d) refuses by name instead
        fused = self._fused_path(model, mixer_mod, arm)
        f32 = torch.float32
        block = Block(d["d_model"], partial(mixer_cls, device=dev, dtype=f32, **cfg), torch.nn.Identity,
                      norm_cls=partial(RMSNorm, eps=eps, device=dev, dtype=f32),
                      fused_add_norm=True, residual_in_fp32=True).to(dev)
        block.eval()
        W = _weights(data)
        sd = {}
        for k, v in W.items():
            t = torch.from_numpy(np.ascontiguousarray(v))
            if model == "mamba3" and k in ("B_bias", "C_bias"):
                t = t.reshape(t.shape[0], 1, t.shape[1])      # [H, N] -> [H, mimo_rank 1, N]
            sd[mamba_ssm_state_name(model, k)] = t
        block.load_state_dict(sd, strict=True)
        for k, v in sd.items():            # every parameter holds our bits
            got = block.state_dict()[k].detach().cpu()
            if got.dtype != f32 or not torch.equal(got.reshape(v.shape), v):
                raise RuntimeError("REFUSED: %s: %s did not load our float32 bits" % (arm, k))
        readback, eps_back = mamba_ssm_readback(model, block)
        for k, v in eps_back.items():
            if v != eps:
                raise RuntimeError("REFUSED: %s: %s is %r, ours and the reference use %r"
                                   % (arm, k, v, eps))
        self.block = block
        self.host = [torch.from_numpy(np.ascontiguousarray(data["x"]))]
        version, direct_url = _mamba_ssm_source()
        try:
            import triton
            triton_ver = triton.__version__
            f32_default = getattr(getattr(getattr(triton, "knobs", None), "language", None),
                                  "fp32_default", None)
        except Exception:  # noqa: BLE001
            triton_ver, f32_default = None, None
        if f32_default is not None and f32_default != setting["triton_f32_default"]:
            raise RuntimeError("REFUSED: %s: Triton's fp32 default reads back %r, not %r"
                               % (arm, f32_default, setting["triton_f32_default"]))
        self.info = dict(info)
        self.info.update({
            "library": "mamba-ssm", "version": version, "device": "gpu", "device_name": name,
            "torch_version": torch.__version__, "torch_version_cuda": torch.version.cuda,
            "torch_backend": kind, "setting": arm[len("mamba-ssm-"):],
            "precision": ("float32 weights and activations; TF32 %s in torch; Triton fp32 tl.dot "
                          "input_precision %s" % ("ON" if setting["tf32"] else "off",
                                                  setting["triton_f32_default"])),
            "mode": "mamba_ssm fused kernels, eager, %s" % arm[len("mamba-ssm-"):],
            "precision_readback": precision, "triton_version": triton_ver,
            "triton_f32_default": f32_default if f32_default is not None
            else os.environ.get("TRITON_F32_DEFAULT"),
            "causal_conv1d_version": _dist_version("causal_conv1d") or _dist_version("causal-conv1d"),
            "mamba_ssm_direct_url": direct_url, "mamba_ssm_file": getattr(mamba_ssm, "__file__", None),
            "mixer": MAMBA_SSM_MIXER[model], "mixer_config": json.dumps(cfg, sort_keys=True, default=str),
            "block": "mamba_ssm.modules.block.Block(norm=Triton RMSNorm eps %g, fused_add_norm=True, "
                     "residual_in_fp32=True, mlp=Identity); out = mixer(norm(x)) + x" % eps,
            "fused_path": fused, "norm_eps_readback": eps_back, "config_readback": readback,
            "weights": "ours, load_state_dict(strict=True), read back bit for bit",
            "compile": "eager (mamba_ssm's own kernels)", "pre_clock_fit": False,
            "input_home": "host",
            "call": "mamba_ssm Block(%s).forward(x) on %s; h + residual; .cpu()"
                    % (MAMBA_SSM_MIXER[model].rsplit(".", 1)[1], dev)})
        self.record = {"__library__": "mamba-ssm", "seed": SEED}
        self.record.update(readback)
        self.out = None
        sync()

    @staticmethod
    def _fused_path(model, mixer_mod, arm):
        if model == "mamba3":
            return "mamba3_siso_combined (Triton)"
        if getattr(mixer_mod, "causal_conv1d_fn", None) is None:
            raise RuntimeError("REFUSED: %s: causal_conv1d does not import, so mamba_ssm would "
                               "leave its fused path; install the board's pinned causal-conv1d"
                               % arm)
        if model == "mamba1":
            return "mamba_inner_fn (causal_conv1d + selective_scan CUDA kernel, fused)"
        return "mamba_split_conv1d_scan_combined (causal_conv1d + SSD Triton kernels, gated RMSNorm)"

    def call(self):
        torch = self.torch
        x = self.host[0].to(self.dev)
        with torch.no_grad():
            h, r = self.block(x)
            out = h + r
        self.out = out.float().cpu().numpy()

    def sync(self):
        self._sync()

    def outputs(self):
        return {"y": self.out}

    def digest(self):
        return _sha(self.np.ascontiguousarray(self.out).data)[:16]


def build_runner(lane, arm, shape, data):
    if arm in ("ours", "ours-cpu"):
        return OURS[MODEL_OF[lane]](lane, shape, data)
    if arm in MAMBA_SSM_ARMS:
        return MambaSsmArm(lane, shape, data, arm)
    return TorchArm(lane, shape, data, arm)


def worker(args):
    # fd 1 is the protocol; everything a library prints goes to the log.
    proto = os.fdopen(os.dup(1), "w", buffering=1)
    os.dup2(2, 1)
    sys.stdout = sys.stderr

    def say(obj):
        proto.write(json.dumps(obj, sort_keys=True, default=str) + "\n")
        proto.flush()

    import numpy as np
    try:
        with np.load(args.data) as z:
            data = {k: z[k] for k in z.files}
        runner = build_runner(args.lane, args.arm, args.shape, data)
    except (Exception, SystemExit) as exc:  # noqa: BLE001  (a twin's refuse() exits)
        import traceback
        traceback.print_exc()
        say({"event": "error", "stage": "ready", "error": repr(exc)[:2000]})
        return 1
    if isinstance(runner.info, dict):   # the arm's own library version and GPU (the store's key)
        runner.info.update(_load("bench_board_probe").library_identity(runner.info))
    say({"event": "ready", "info": runner.info, "pid": os.getpid(),
         "params_record": getattr(runner, "record", None)})
    # peak memory per round, reset and read OUTSIDE the clock
    mem = _load("bench_board_probe").MemProbe((runner.info or {}).get("device", "gpu"),
        library=(runner.info or {}).get("library") or "?")
    for line in sys.stdin:
        parts = line.split()
        if not parts:
            continue
        if parts[0] == "round":
            r = int(parts[1])
            try:
                mem.start()
                t0 = time.perf_counter()
                runner.call()
                runner.sync()
                ms = (time.perf_counter() - t0) * 1000.0
                m = mem.stop()
                digest = runner.digest()
            except Exception as exc:  # noqa: BLE001
                import traceback
                traceback.print_exc()
                where = runner.info.get("torch_backend") or runner.info.get("device")
                say({"event": "error", "stage": "round %d" % r,
                     "error": "REFUSED: %s on %s failed in round %d%s: %s" % (
                         args.arm, where, r, " (compile happens here)"
                         if r == 0 and "compile" in args.arm else "", repr(exc)[:2000])})
                return 1
            say({"event": "round", "round": r, "ms": ms, "digest": digest, "mem": m})
        elif parts[0] == "save":
            try:
                path = parts[1]
                tmp = path + ".tmp.npz"
                np.savez(tmp, **runner.outputs())
                os.replace(tmp, path)
                say({"event": "saved", "path": path, "info": runner.info})
            except Exception as exc:  # noqa: BLE001
                say({"event": "error", "stage": "save", "error": repr(exc)})
                return 1
        elif parts[0] == "quit":
            say({"event": "bye"})
            return 0
    return 0


# ---------------------------------------------------------------------------
# Quality (the conductor, float64)
# ---------------------------------------------------------------------------

def _mean_nll(np, logits, targets):
    lg = logits.astype(np.float64)
    mx = lg.max(axis=-1, keepdims=True)
    lse = mx[..., 0] + np.log(np.exp(lg - mx).sum(axis=-1))
    picked = np.take_along_axis(lg, targets[..., None], axis=-1)[..., 0]
    return float((lse - picked).mean())


def quality(lane, data, outs):
    import numpy as np
    q = {}
    if lane in TRAIN_LANES:
        for arm, o in outs.items():
            losses = [float(x) for x in o["losses"]]
            q[arm] = {"loss_first_step": losses[0], "loss_last_step": losses[-1], "steps": len(losses)}
        ref = q.get("ours")
        for arm in q:
            if arm != "ours" and ref is not None:
                q[arm]["loss_first_abs_diff_vs_ours"] = abs(q[arm]["loss_first_step"] - ref["loss_first_step"])
                q[arm]["loss_last_abs_diff_vs_ours"] = abs(q[arm]["loss_last_step"] - ref["loss_last_step"])
        return q
    for arm, o in outs.items():
        q[arm] = {}
    if lane in ("lm-forward", "lm-infer", "samba-forward", "samba-infer"):
        targets = data["batches"][0][:, 1:].astype(np.int64)
        for arm, o in outs.items():
            q[arm]["mean_nll"] = _mean_nll(np, o["y"], targets)
    if MODEL_OF[lane] == "gemm":
        a64, b64 = data["a"].astype(np.float64), data["b"].astype(np.float64)
        if lane == "gemm-bf16":           # the bf16-rounded operands (round to nearest even)
            def bf16(x):
                u = x.astype(np.float32).view(np.uint32).astype(np.uint64)
                u = ((u + 0x7FFF + ((u >> 16) & 1)) >> 16) << 16
                return u.astype(np.uint32).view(np.float32).astype(np.float64)
            a64, b64 = bf16(data["a"]), bf16(data["b"])
        c64 = a64 @ b64.T if lane == "gemm-int8" else a64 @ b64
        scale = float(np.abs(c64).max()) or 1.0
        for arm, o in outs.items():
            q[arm]["max_rel_err_vs_fp64"] = float(np.abs(o["y"].astype(np.float64) - c64).max()) / scale
    ref = outs.get("ours", {}).get("y")
    if ref is not None:
        r64 = ref.astype(np.float64)
        scale = float(np.abs(r64).max()) or 1.0
        for arm, o in outs.items():
            if arm == "ours":
                continue
            if o["y"].shape != ref.shape:
                q[arm]["shape_mismatch_vs_ours"] = "%s vs %s" % (list(o["y"].shape), list(ref.shape))
                continue
            diff = float(np.abs(o["y"].astype(np.float64) - r64).max())
            q[arm]["max_abs_diff_vs_ours"] = diff
            q[arm]["max_rel_diff_vs_ours"] = diff / scale
    return q


# ---------------------------------------------------------------------------
# race: the conductor for one lane
# ---------------------------------------------------------------------------

def _worker_env(arm):
    ctd = _load("classical_two_datasets")
    env = dict(os.environ)
    for k in ctd.THREAD_ENV:
        env.pop(k, None)
    if arm in ("ours", "ours-cpu"):
        env["MOJOLEARN_NUMERIC_MODE"] = "identical"
        if arm == "ours-cpu":
            _load("bench_board_probe").ours_cpu_env(env)
        if os.environ.get("MOJOLEARN_BENCH_INSTALLED", "0").strip() in ("", "0"):
            tree = os.path.join(REPO, "python")
            env["PYTHONPATH"] = tree + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    return env


SPAN = {"input_home": "host", "pre_clock_fit": False,
        "inside_clock": "host inputs to the device, the call, the result back on the host, synchronized"}


def race(args):
    import numpy as np
    ctd = _load("classical_two_datasets")
    lane, shape = args.lane, args.shape
    arms = [a for a in args.arms.split(",") if a]
    for a in arms:
        if a not in ARMS:
            raise SystemExit("no arm %r for lane %r" % (a, lane))
    os.makedirs(args.out, exist_ok=True)
    os.makedirs(args.work, exist_ok=True)
    tag = "%s-%s" % (lane, DATA_OF[lane])
    data_path = os.path.join(args.work, "neural-%s-%s.npz" % (lane, shape))
    inputs = make_inputs(lane, shape, args.rounds + 1, data_path)
    srec = shape_record(lane, shape)
    result = {"lane": lane, "dataset": DATA_OF[lane], "shape": srec["label"], "shape_record": srec,
              "inputs": inputs, "arms": {}, "rounds_requested": args.rounds, "started": now_utc(),
              "script": "tools/bench_board_neural.py", "ours_device": DEVICE_OF[lane],
              "torch_settings": {a: precision_text(arm_setting(a)[1]) for a in arms
                                 if a.startswith("torch-")},
              "commit": os.environ.get("MOJOLEARN_REPO_COMMIT", "unknown")}
    workers = {}
    for arm in arms:
        py = args.ours_python if arm in ("ours", "ours-cpu") else args.theirs_python
        cmd = shlex.split(py) + [os.path.abspath(__file__), "worker", "--arm", arm,
                                 "--lane", lane, "--shape", shape, "--data", data_path]
        workers[arm] = ctd.Worker(arm, cmd, _worker_env(arm),
                                  os.path.join(args.out, "%s-%s.log" % (tag, arm)), REPO)
        result["arms"][arm] = {"command": cmd, "warmup_ms": None, "ms": [], "digests": [],
                               "mem": [], "status": "ok"}
    for arm, w in workers.items():
        msg = w.read(args.ready_seconds)
        if msg is None or msg.get("event") != "ready":
            w.kill("not_ready", msg)
            result["arms"][arm].update(status="not_ready", error=msg)
            print("NEURAL-REFUSED lane=%s arm=%s stage=ready detail=%s" % (lane, arm, json.dumps(msg)),
                  flush=True)
            continue
        w.info = msg["info"]
        result["arms"][arm]["info"] = msg["info"]
        result["arms"][arm]["params_record"] = msg.get("params_record")
    # THE PARAMETER CHECK (tools/bench_board_params.py), before the first
    # timed round: same seed, same tuning parameters on every arm, read back
    # from what each worker constructed. A refusal fails the race by name.
    if getattr(args, "params_only", False):
        return _load("classical_two_datasets").params_only_exit(
            result, workers, arms, "neural/" + lane, "neural",
            os.path.join(args.out, "%s.params.json" % lane))
    BP = _load("bench_board_params")
    records = {a: result["arms"][a]["params_record"] for a in arms
               if workers[a].alive and result["arms"][a].get("params_record")}
    try:
        result["params_check"] = BP.enforce("neural/" + lane, records, family="neural")
    except BP.ParamsRefused as exc:
        result["params_check"] = BP.check("neural/" + lane, records, family="neural")
        result["params_refused"] = str(exc)
        for arm in arms:
            if workers[arm].alive:
                workers[arm].kill("params_refused", None)
            if result["arms"][arm]["status"] == "ok":
                result["arms"][arm].update(status="params_refused", error=str(exc)[:2000])
        result["finished"] = now_utc()
        out_json = os.path.join(args.out, "%s.json" % tag)
        with open(out_json + ".tmp", "w") as fh:
            json.dump(result, fh, indent=2, sort_keys=True, default=str)
        os.replace(out_json + ".tmp", out_json)
        print("NEURAL-PARAMS-REFUSED lane=%s %s" % (lane, str(exc)[:2000]), flush=True)
        return 3
    for r in range(args.rounds + 1):
        live = [a for a in arms if workers[a].alive]
        if not live:
            break
        shift = r % len(live)
        for arm in live[shift:] + live[:shift]:
            w = workers[arm]
            w.send("round %d" % r)
            msg = w.read(args.warmup_seconds if r == 0 else args.round_seconds)
            if msg is None or msg.get("event") != "round":
                status = "timeout" if msg is None else "error"
                w.kill(status, msg)
                result["arms"][arm].update(status=status, error=msg, failed_round=r)
                print("NEURAL-REFUSED lane=%s arm=%s stage=round%d detail=%s"
                      % (lane, arm, r, json.dumps(msg)), flush=True)
                continue
            if r == 0:
                result["arms"][arm]["warmup_ms"] = msg["ms"]
            else:
                result["arms"][arm]["ms"].append(msg["ms"])
            result["arms"][arm]["digests"].append(msg["digest"])
            result["arms"][arm]["mem"].append(msg.get("mem"))
            print("NEURAL-ROUND lane=%s arm=%s round=%d ms=%.3f digest=%s"
                  % (lane, arm, r, msg["ms"], msg["digest"]), flush=True)
    outs = {}
    for arm in arms:
        w = workers[arm]
        if w.alive and len(result["arms"][arm]["ms"]) == args.rounds:
            path = os.path.join(args.work, "%s-%s.npz" % (tag, arm))
            w.send("save %s" % path)
            msg = w.read(args.round_seconds)
            if msg is not None and msg.get("event") == "saved":
                with np.load(path) as z:
                    outs[arm] = {k: z[k] for k in z.files}
                os.remove(path)
            else:
                result["arms"][arm].update(status="save_failed", error=msg)
        w.close()
    try:
        with np.load(data_path) as z:
            data = {k: z[k] for k in z.files}
        result["quality"] = quality(lane, data, outs)
        # our CPU tier against our GPU IDENTICAL, bit for bit (the promise):
        # the outputs on forward lanes, every step's loss on train lanes
        if "ours-cpu" in outs and "ours" in outs:
            result["quality"].setdefault("ours-cpu", {})["bits_equal_vs_ours_identical"] = \
                _load("bench_board_probe").bits_equal(outs["ours-cpu"], outs["ours"])
    except Exception as exc:  # noqa: BLE001
        result["quality"] = {"error": repr(exc)}
    try:
        os.remove(data_path)
    except OSError:
        pass
    for arm in arms:
        a = result["arms"][arm]
        ok = a["status"] == "ok" and len(a["ms"]) == args.rounds
        a["median_ms"] = statistics.median(a["ms"]) if ok else None
        timed = [d for d in a["digests"][1:] if d is not None]
        # a training step changes the state every round (no repeat to compare),
        # and one timed round has nothing to compare against either
        a["digest_stable"] = (len(set(timed)) == 1) if ok and len(timed) >= 2 else None
        a["span"] = dict(SPAN)
        print("NEURAL lane=%s arm=%s status=%s median_ms=%s quality=%s"
              % (lane, arm, a["status"], a["median_ms"],
                 json.dumps(result["quality"].get(arm, {}), sort_keys=True)), flush=True)
    result["finished"] = now_utc()
    out_json = os.path.join(args.out, "%s.json" % tag)
    with open(out_json + ".tmp", "w") as fh:
        json.dump(result, fh, indent=2, sort_keys=True, default=str)
    os.replace(out_json + ".tmp", out_json)
    failed = [a for a in arms if result["arms"][a]["status"] != "ok"]
    return 1 if failed and len(failed) == len(arms) else 0


def build_parser():
    p = argparse.ArgumentParser(prog="bench_board_neural", description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("race")
    r.add_argument("--lane", required=True, choices=LANES)
    r.add_argument("--shape", default="full", choices=sorted(LM_SHAPES))
    r.add_argument("--arms", default="ours,torch-eager-fp32")
    r.add_argument("--rounds", type=int, default=5)
    r.add_argument("--out", required=True)
    r.add_argument("--work", required=True)
    r.add_argument("--ours-python", default=sys.executable)
    r.add_argument("--theirs-python", default=sys.executable)
    r.add_argument("--ready-seconds", type=int, default=1800)
    r.add_argument("--params-only", action="store_true",
                   help="construct every arm, read its parameters back, write <out>/<tag>.params.json "
                        "and stop before the warm-up (the board's opponent-store lookup)")
    r.add_argument("--warmup-seconds", type=int, default=1800)
    r.add_argument("--round-seconds", type=int, default=1800)
    w = sub.add_parser("worker")
    w.add_argument("--arm", required=True, choices=ARMS)
    w.add_argument("--lane", required=True, choices=LANES)
    w.add_argument("--shape", required=True, choices=sorted(LM_SHAPES))
    w.add_argument("--data", required=True)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.cmd == "worker":
        return worker(args)
    if args.rounds < 1:
        raise SystemExit("--rounds must be >= 1")
    return race(args)


if __name__ == "__main__":
    sys.exit(main())
