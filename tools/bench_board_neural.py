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
GPU inference lanes (2026-10-07, lane/neural-gpu-infer-rows: the GPU
inference/decode rows; the *-infer rows below are the CPU host binding and
are never raced, so GPU decode work had no row before these):
  transformer-decode   TransformerBlock(...).decode_session(state).step(x_t),
                       one token per call over the whole sequence, zero
                       state (load_state) per round.
  mamba1-decode        Mamba1Block(weights).decode_session(state).step(x_t).
  mamba2-decode / mamba3-decode
                       Mamba{2,3}Block(weights).step(x_t, state) (their
                       resident sessions are private, so the public
                       per-call step), fresh allocate_state per round.
  samba-decode         SambaStack(config, weights).step(ids_t, state).
  mlp-predict          SmallMLPTrainer(w1, b1, w2, b2).predict_logits(X).
CPU lanes (the *Inference classes run on the host binding, on the CPU):
  transformer-infer    TransformerBlockInference(...).forward(x).
  mamba1-infer / mamba2-infer / mamba3-infer
                       Mamba{1,2,3}BlockInference(weights).forward(x).
  samba-infer          SambaInference(config, weights).forward(inputs).
  mlp-infer            MLPInference(w1, b1, w2, b2).predict_logits(X).
GPU fit lanes (2026-10-19, lane ag-bench: the trainers whose backward runs
through the autograd tape; one round is ONE WHOLE FIT, the held-out
prediction runs after the clock and is scored by the conductor):
  mlp-fit              MLPClassifier(hidden_layer_sizes=(256,), solver='adam',
                       batch_size=512, max_iter=1, alpha=0.0, ...).fit(X, y)
                       on HIGGS (tools/sequence_speed.py case_mlp's recipe:
                       1M rows x 28 standardized; one logit, binary log loss).
  lstm-fit / gru-fit / rnn-fit
                       {LSTM,GRU,RNN}Regressor(hidden_size=64, num_layers=1,
                       batch_size=512, max_epochs=1, optimizer='adam')
                       .fit(X, y) on HIGGS rows cut into 16-step sequences
                       (case_recurrent's recipe; the target is the label of
                       each sequence's last row, MSE).
  tape-user-model      mojolearn.Sequential: Linear(28, 64), ReLU,
                       Linear(64, 64), ReLU, Linear(64, 1), BCE with logits,
                       Adam: a model the user describes, trained through the
                       tape (the model API of lane ag-pyapi; the arm refuses
                       by name while that class is absent).
  sequential-fit       (2026-10-19, lane ag-board-twins) mojolearn.Sequential:
                       linear(128), relu, linear(64), relu, linear(1), BCE
                       with logits, Adam; fit(X, y) and predict_proba(X_test),
                       BOTH inside the clock, on a seeded synthetic binary set
                       (no download: 32 standard-normal features, the label
                       the sign of a fixed teacher plus noise; FIT_ROWS rows).
  Their opponents are the same networks in torch.nn (nn.Sequential of
  nn.Linear/nn.ReLU, nn.LSTM / nn.GRU / nn.RNN with an nn.Linear head on the
  last hidden state) trained by torch.optim.Adam with the same sizes, batch
  size, epoch count, learning rate, betas, eps and seed, every training row
  moved to the device inside the clock (TorchFitArm), at every torch setting
  of the box (TORCH_SETTINGS); on Apple also MLX (`mlx-fp32`: mlx.nn and
  mlx.optimizers.Adam with bias correction, MlxFitArm). On Apple the fit
  lanes race `ours-fast` only: identical is never timed on Apple.
  Quality beside every time (held-out rows, float64 in the conductor):
  accuracy, log loss and AUC for the classifiers, RMSE and AUC of the score
  for the recurrent regressors (tools/speed_gbdt_arm.py's metric helpers).
GPU composition lane (2026-10-19, lane ag-board-twins):
  ops-session-step     one training step composed with the public operation
                       session, mojolearn.ops.Session(record=True): linear,
                       relu, linear, softmax cross-entropy, backward, every
                       gradient read back with grad(), the SGD update in place
                       on the device with the session's optimizer (step) and
                       the updated parameters back on the host. Every round is the same
                       step from the same fixture weights (nothing carries
                       over), so an identical arm's digest repeats. The
                       opponent is the same step in torch autograd (leaf
                       tensors with requires_grad, F.cross_entropy, backward,
                       torch.optim.SGD), the same weights, batch and lr, at
                       every torch setting of the box. Quality: the loss, each
                       gradient's and updated parameter's max difference from
                       ours, and every arm's gradient digests.
GPU captured-program lane (2026-10-19, lane ag-program):
  program-run          one inference forward of a captured model,
                       mojolearn.ops.Program.run: the batch to the device,
                       relu(linear), linear, softmax, the probabilities back
                       on the host, in one call (Session.capture of the same
                       composition, the weights resident on the device as the
                       program's parameters, before the clock). The worker
                       checks once, before the clock, that the program's
                       output words equal the eager session's; the A/B
                       operation `program_eager` times the eager session (one
                       call per operation) for the comparison. The opponent is
                       the same forward in torch under no_grad (weights on the
                       device before the clock), at every torch setting of the
                       box. Quality: every arm's max error from a float64
                       forward and its difference from ours.
Inference lanes (2026-10-19, lane ag-inference-board), weights resident before the clock:
  causal-lm-generate   mojolearn.models.CausalLM.load(<checkpoint dir>,
                       weight_format='float32', numeric_profile='fp32_v1')
                       .generate(prompt_ids, N), greedy, on the published
                       mojolearn-162m checkpoint (huggingface.co/mojolearn-ai/
                       mojolearn-162m, its byte-LM format: no final norm);
  causal-lm-generate-smollm2  the same on HuggingFaceTB/SmolLM2-135M (a public
                       Llama checkpoint the loader supports). The clock is
                       generate(prompt, N): the prompt ids to the device, the
                       prefill, N - 1 cached decode steps, the ids back on the
                       host; prefill_ms is a separate generate(prompt, 1) timed
                       the same way, decode_ms_per_token their difference over
                       N - 1. The opponent is transformers' LlamaForCausalLM on
                       the same checkpoint (model.norm an Identity for the
                       162M), generate(do_sample=False, num_beams=1, no stop
                       token), eager fp32 / tf32 / bf16 weights. Quality: the
                       new ids equal to ours or the first divergence index.
  sequential-predict   mojolearn.Sequential (the sequential-fit network) fitted
                       once in the untimed warm-up round, then
                       predict_proba(X_test) alone in the clock; the opponent
                       the same torch.nn network fitted the same way, its
                       held-out sigmoid in the clock. Quality: held-out
                       accuracy, log loss and AUC of each arm's own fit.
  Every inference arm also runs one UNTIMED memory pass after each round (the
  same call): the device bytes the model holds (read around its load) plus the
  pass's transient peak, sampled in process (KFD per-process VRAM on AMD, NVML
  per-process or device used on NVIDIA), and torch.cuda.max_memory_allocated
  for torch.
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

FAIRNESS FIX (2026-10-13 audit): a compile arm gets its warm-up (the compile)
outside the timed rounds with its own longer deadline (--compile-warmup-seconds)
and its warm-up time is recorded as `compile_warmup_ms`; if it still times out
the race has no opponent (the other torch arms are recorded as context:
`not_opponent`, board mode "context", no ratio) instead of silently promoting
the eager arm, as it did on AMD samba-train-step. Results raced before this
change were wrong for these lanes; apply extra scrutiny.

The torch models (FAIRNESS FIX 2026-10-13 audit: library classes, not harness
twins; the harness-written ByteDecoder / LlamaEager twins and the pure-PyTorch
Mamba references were the opponents before; see "library opponents" below):
  lm-*          transformers LlamaForCausalLM at ours' architecture (untied
                head, final norm Identity, SDPA; on the torch.cuda API a
                probed SDPA backend is pinned for the fp32/tf32 columns,
                torch's own pick for bf16).
  gemm          a @ b.
  transformer   transformers LlamaDecoderLayer (SDPA, causal) with
                LlamaRotaryEmbedding; the decode lane uses its DynamicCache.
  mamba1/2/3, arms mamba-ssm-fp32 (NVIDIA, AMD) / mamba-ssm-tf32 (NVIDIA):
                mamba_ssm's own fused kernels (Block + Mamba / Mamba2 /
                Mamba3 SISO), our weights loaded. The torch-* arms below are
                the pure-PyTorch references and are REFUSED by name.
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
  samba         torch Embedding, then per layer mamba_ssm's Mamba3 Block or
                transformers' LlamaDecoderLayer, F.rms_norm (eps 1e-5), tied
                head, mean CE, torch.optim.AdamW; refused where mamba_ssm
                does not import (Apple, the CPU).
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

OUR TWO ARMS. `ours` is the IDENTICAL tier (every vendor; the worker starts
under MOJOLEARN_NUMERIC_MODE=identical and refuses any other). `ours-fast` is
the Apple FAST neural tier (2026-10-03): the same public classes under
MOJOLEARN_NUMERIC_MODE=fast, the FAST binaries the wheel ships on Apple; its
worker refuses to start under any other mode and refuses a binary that reads
back another tier. Quality columns of `ours-fast` are against `ours` like an
opponent's (loss differences on the train lanes, max abs / rel output
difference on the forward lanes). Neither arm runs on a CPU lane. The A/B tool
tools/afn_ab.sh races `ours-fast` alone (`--arms ours-fast`) and keeps each
arm's outputs (`--keep-outputs`) for its own judge.
"""
import argparse
import glob
import hashlib
import importlib.util
import json
import os
import platform
import shlex
import signal
import statistics
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)

SEED = 7

#: AdamW on every train lane, every arm, passed EXPLICITLY (never a library
#: default): ours' LanguageModelTrainer / SambaStack / SmallMLPTrainer and
#: torch.optim.AdamW. Samba runs without clipping on both (max_norm=None).
ADAMW = {"lr": 1e-3, "betas": (0.9, 0.999), "eps": 1e-8, "weight_decay": 0.01}
#: The AdamW flags that change the update and that ours' trainers fix by implementation (preflight-smoke,
#: 2026-10-14). Ours' record carries them; the torch arm passes the same values explicitly and its read-back
#: is compared. Source-reviewed: python/mojolearn/_training_impl.py AdamW (_KIND_ADAMW: decoupled decay
#: p*(1-lr*wd); amsgrad unsupported, raises TypeError; maximize default False, and the trainers
#: (_mlp_impl.SmallMLPTrainer, LanguageModelTrainer, SambaStack) never pass it).
OURS_ADAMW_SEMANTICS = {"amsgrad": False, "maximize": False, "decoupled_weight_decay": True}
#: TransformerBlock / LlamaEager: RMSNorm eps and RoPE base, passed explicitly
#: to ours and read back from tools/speed_torch_seq.py on the torch side.
BLOCK_NORM_EPS, BLOCK_ROPE_THETA = 1e-6, 10000.0
#: SambaStack's final RMSNorm eps and dropout, passed explicitly; the torch
#: stack twin uses the same eps and has no dropout.
SAMBA_NORM_EPS, SAMBA_DROPOUT = 1e-5, 0.0
#: samba-train-step's max_grad_norm on both arms: gradient clipping off (see _lane_params)
SAMBA_CLIP_OFF = "off"

# ---------------------------------------------------------------------------
# The tables the board reads (standard library only)
# ---------------------------------------------------------------------------

LANES = ("lm-train-step", "lm-forward", "gemm",
         "transformer-forward", "transformer-infer",
         "mamba1-forward", "mamba1-infer", "mamba2-forward", "mamba2-infer",
         "mamba3-forward", "mamba3-infer",
         "samba-train-step", "samba-forward", "samba-infer",
         "mlp-train-step", "mlp-infer",
         # 2026-10-07: GPU inference rows (the *-infer rows are the CPU
         # binding and never race; these are the GPU decode/predict routes)
         "transformer-decode", "mamba1-decode", "mamba2-decode", "mamba3-decode",
         "samba-decode", "mlp-predict",
         # 2026-09-29: the byte LM's CPU inference and CPU training step, and the
         # bf16 and int8 GEMM profiles (SmallByteLanguageModelTrainer IS
         # LanguageModelTrainer, raced by lm-train-step and lm-forward)
         "lm-infer", "lm-host-train-step", "gemm-bf16", "gemm-int8")
#: 2026-10-19 (lane ag-bench): whole fits of the tape-trained models on HIGGS (see the docstring)
FIT_LANES = ("mlp-fit", "lstm-fit", "gru-fit", "rnn-fit", "tape-user-model",
             # 2026-10-19 (lane ag-board-twins): Sequential fit + predict_proba on a synthetic set
             "sequential-fit",
             # 2026-10-19 (lane ag-inference-board): the same Sequential fitted untimed, predict_proba timed alone
             "sequential-predict")
#: fit lanes whose fit is untimed preparation (the warm-up round) and whose clock is the prediction alone
PREDICT_LANES = ("sequential-predict",)
#: fit lanes whose data is the conductor's seeded synthetic set (no HIGGS download)
SYNTH_FIT_LANES = ("sequential-fit", "sequential-predict")
#: fit lanes whose held-out predict_proba runs INSIDE the clock on every arm
FIT_PREDICT_IN_CLOCK = ("sequential-fit",)
#: 2026-10-19 (lane ag-board-twins): one recorded training step through mojolearn.ops.Session
OPS_LANES = ("ops-session-step",)
#: 2026-10-19 (lane ag-program): one inference forward through a captured mojolearn.ops.Program
PROGRAM_LANES = ("program-run",)
LANES = LANES + FIT_LANES + OPS_LANES + PROGRAM_LANES
#: 2026-10-19 (lane ag-inference-board): greedy generation from a real pretrained checkpoint, lane -> checkpoint
#: (LM_CHECKPOINTS); the checkpoint is the lane's dataset column
GEN_LANES = {"causal-lm-generate": "mojolearn-162m", "causal-lm-generate-smollm2": "smollm2-135m"}
#: the inference lanes: an untimed memory pass after every round (DeviceMemory)
INFER_LANES = tuple(GEN_LANES) + PREDICT_LANES
LANES = LANES + FIT_LANES + OPS_LANES + tuple(GEN_LANES)
#: The model each lane runs.
MODEL_OF = {"lm-train-step": "lm", "lm-forward": "lm", "gemm": "gemm",
            "transformer-forward": "transformer", "transformer-infer": "transformer",
            "mamba1-forward": "mamba1", "mamba1-infer": "mamba1",
            "mamba2-forward": "mamba2", "mamba2-infer": "mamba2",
            "mamba3-forward": "mamba3", "mamba3-infer": "mamba3",
            "samba-train-step": "samba", "samba-forward": "samba", "samba-infer": "samba",
            "mlp-train-step": "mlp", "mlp-infer": "mlp",
            "lm-infer": "lm", "lm-host-train-step": "lm", "gemm-bf16": "gemm", "gemm-int8": "gemm",
            "transformer-decode": "transformer", "mamba1-decode": "mamba1",
            "mamba2-decode": "mamba2", "mamba3-decode": "mamba3", "samba-decode": "samba",
            "mlp-predict": "mlp",
            "mlp-fit": "mlpfit", "lstm-fit": "rnnfit", "gru-fit": "rnnfit", "rnn-fit": "rnnfit",
            "tape-user-model": "tapefit", "sequential-fit": "tapefit", "ops-session-step": "opsstep",
            "program-run": "program"}
            "sequential-predict": "tapefit", "causal-lm-generate": "causallm",
            "causal-lm-generate-smollm2": "causallm"}
#: The recurrent cell of each recurrent fit lane.
FIT_CELL = {"lstm-fit": "lstm", "gru-fit": "gru", "rnn-fit": "rnn"}
#: GPU incremental-decode lanes: one public step call per token over the
#: lane's whole sequence, from a zero state each round.
DECODE_LANES = ("transformer-decode", "mamba1-decode", "mamba2-decode", "mamba3-decode",
                "samba-decode")
#: Decode lanes with a torch decode twin in this harness. The Mamba
#: references (mamba/corpus/gen_corpus.py) are full-sequence scans with no
#: carried-state step, and Samba stacks Mamba-3, so those lanes race ours alone.
DECODE_TORCH_TWIN = ("transformer-decode",)
TRAIN_LANES = ("lm-train-step", "samba-train-step", "mlp-train-step", "lm-host-train-step")
#: Whole-fit lanes (2026-10-19, lane ag-bench): FIT_LANES above. Every arm's round is one fit
#: from a fresh seeded initialization; nothing carries over between rounds, so the per-round
#: digest of an identical arm repeats (digest_stable is meaningful).
FIT_TASK = {"mlp-fit": "classification", "tape-user-model": "classification", "sequential-fit": "classification",
            "sequential-predict": "classification",
            "lstm-fit": "regression", "gru-fit": "regression", "rnn-fit": "regression"}
#: Where OUR class runs: the *Inference classes are the host binding.
DEVICE_OF = {lane: ("cpu" if lane.endswith("-infer") or lane == "lm-host-train-step" else "gpu")
             for lane in LANES}
#: The data each lane reads (the board's `dataset` column).
DATA_OF = {lane: (GEN_LANES[lane] if lane in GEN_LANES else "synthetic" if lane in SYNTH_FIT_LANES else "higgs" if lane in FIT_LANES else "bytes" if MODEL_OF[lane] in ("lm", "samba")
                  else "gaussian") for lane in LANES}

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
NO_COMPILE = ("mamba1-forward", "mamba1-infer",
              # a per-token Python decode loop whose cache length grows every
              # step: torch.compile would recompile per length
              "transformer-decode",
              # 2026-10-19 (lane ag-inference-board): transformers' generate is the same growing-cache loop
              "causal-lm-generate", "causal-lm-generate-smollm2")
VENDORS = ("apple", "nvidia", "amd")

# FAIRNESS FIX (2026-10-13 audit): the Mamba lanes' only opponents on main were the repo's
# pure-PyTorch references (a reference implementation is not an opponent). mamba_ssm's own
# fused kernels (lane/mamba-opp facde18cc, Andrew 2026-09-29 "we should be using the strongest
# opponent always") come onto main as the Mamba opponent: on NVIDIA as built there, and on AMD
# from mamba_ssm's ROCm/HIP source build (tools/bench_board.py MAMBA_SSM_BUILD["amd"]; never
# built or run on the board's AMD box: unverified at runtime). Apple has no mamba_ssm build
# (CUDA/HIP and Triton only), so the Mamba lanes there have no opponent. One block is
# mamba_ssm's modules/block.py Block (fused Triton add+RMSNorm, the mixer, residual in fp32)
# around its mixer: mamba_simple.Mamba on the fast path (causal_conv1d + mamba_inner_fn, the
# selective-scan kernel), mamba2.Mamba2 on the memory-efficient path
# (mamba_split_conv1d_scan_combined, the SSD Triton kernels), mamba3.Mamba3 SISO
# (mamba3_siso_combined, Triton). Weights, input and dtype (float32) are ours, loaded with
# load_state_dict(strict=True) and read back bit for bit. Settings, the precision in the name:
#   mamba-ssm-fp32   TF32 off in torch and TRITON_F32_DEFAULT=ieee, so every fp32 tl.dot in
#                    the Triton kernels is IEEE fp32 (ours' precision).
#   mamba-ssm-tf32   TF32 on in torch and TRITON_F32_DEFAULT=tf32 (Triton's own default, what an
#                    out-of-the-box install runs); NVIDIA only (TF32 is a CUDA tensor-core mode).
# Pinned to the state-spaces/mamba commit mamba/corpus/gen_corpus.py cites (PyPI 2.3.2.post1 is
# an older Mamba-3). Results raced before this change were wrong for this lane; apply extra
# scrutiny.
MAMBA_SSM_ARMS = ("mamba-ssm-fp32", "mamba-ssm-tf32")
MAMBA_SSM_LANES = ("mamba1-forward", "mamba2-forward", "mamba3-forward")
MAMBA_SSM_VENDOR_ARMS = {"nvidia": MAMBA_SSM_ARMS, "amd": ("mamba-ssm-fp32",)}
MAMBA_SSM_VENDORS = tuple(MAMBA_SSM_VENDOR_ARMS)

#: Our arms: `ours` IDENTICAL, `ours-fast` the Apple FAST neural tier.
OUR_ARMS = ("ours", "ours-fast")
OUR_MODE = {"ours": "identical", "ours-fast": "fast"}
#: 2026-10-19 (lane ag-bench): MLX on Apple, the fit lanes' second Apple opponent (MlxFitArm)
MLX_ARMS = ("mlx-fp32",)
ARMS = OUR_ARMS + tuple("torch-" + s for s in TORCH_SETTINGS) \
    + tuple("torch-cpu-" + s for s in CPU_SETTINGS) + ("torch-eager-int8", "torch-compile-int8") \
    + MAMBA_SSM_ARMS + MLX_ARMS
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
       "gemm-bf16 races torch's bf16 settings only (bf16 operands, fp32 accumulate)",
       "mamba1-decode, mamba2-decode, mamba3-decode and samba-decode race ours alone: the "
       "repo's torch Mamba references are full-sequence scans with no carried-state decode "
       "step (a torch decode twin is not written yet)",
       "torch-compile-* on transformer-decode: the twin is a per-token loop over a growing "
       "KV cache; transformer-decode races the eager arms only"]
    + (["gemm-int8: torch._int_mm is a CUDA kernel; torch on %s has no int8 matmul, so ours "
        "races alone" % {"apple": "MPS", "amd": "ROCm"}[v]] if v != "nvidia" else [])
    # FAIRNESS FIX (2026-10-13 audit): where mamba_ssm's kernels do not exist the Mamba and Samba
    # lanes have no opponent (the pure-PyTorch references are refused, not raced). Results raced
    # before this change were wrong for this lane; apply extra scrutiny.
    + ({"apple": ["mamba-ssm-* on mamba*-forward, and the Samba torch arms: mamba_ssm's kernels "
                  "are CUDA/HIP and Triton (no Metal build exists), so the Mamba and Samba lanes "
                  "have no opponent on Apple (the torch references are refused)"],
        "amd": ["mamba-ssm-tf32: TF32 is an NVIDIA CUDA mode; mamba_ssm races in float32 "
                "(mamba-ssm-fp32, from its ROCm/HIP source build) on AMD"]}.get(v, [])
       + ["mamba-ssm bf16: our Mamba blocks are float32, so mamba_ssm races in float32 (and its "
          "TF32 setting on NVIDIA)"]
       # 2026-10-19 (lane ag-bench): the fit lanes' MLX arm and Apple's tier
       + (["mlx-fp32 on the fit lanes: MLX runs on Apple silicon only"] if v != "apple" else
          ["ours (identical) on the fit lanes: identical is never timed on Apple; the fit lanes race "
           "ours-fast there"])
       # 2026-10-19 (lane ag-inference-board)
       + ["torch-compile-* on causal-lm-generate*: transformers' generate is a per-token loop over a growing KV "
          "cache; the generate lanes race the eager arms only",
          "mlx-fp32 on sequential-predict: no MLX prediction twin is written; it races the torch arms"])
    for v in VENDORS}

#: The board's "Not covered" lines for the neural family.
NOT_COVERED = [
    # FAIRNESS FIX (2026-10-13 audit): this line used to present the pure-PyTorch Mamba
    # references as the Mamba opponents. Those torch arms are refused by name
    # (REFERENCE_TWIN_LANES); mamba_ssm's fused kernels are the opponent where they are built
    # (MAMBA_SSM_VENDOR_ARMS), and the Samba torch arms run mamba_ssm's Mamba3 layers.
    # Results raced before this change were wrong for this lane; apply extra scrutiny.
    "The Mamba lanes' torch-* arms would be the repo's pure-PyTorch references "
    "(mamba/corpus/gen_corpus.py: mamba_ssm's selective_scan_ref for Mamba-1, the chunked SSD "
    "reference for Mamba-2, the SISO reference for Mamba-3), not mamba-ssm's fused CUDA/Triton "
    "kernels; a reference implementation is not an opponent, so they are refused by name. The "
    "Mamba opponent is mamba_ssm's own fused kernels (mamba-ssm-fp32 on NVIDIA and AMD, "
    "mamba-ssm-tf32 on NVIDIA), and the Samba torch arms run mamba_ssm's Mamba3 layers with "
    "transformers' LlamaDecoderLayer attention; where mamba_ssm is not built (Apple) those "
    "lanes have no opponent.",
    "The blocks' backward (the Mamba and TransformerBlock VJPs), ragged `lengths` and a "
    "prefill followed by decode are public and not raced; the zero-state forward (*-forward) "
    "and the zero-state token-by-token decode (*-decode) are.",
    "The byte LM has no incremental decode on any route: LanguageModelTrainer (GPU) and "
    "LanguageModelInference (CPU) expose full-sequence logits only (lm-forward on the GPU), "
    "no KV-cache state or step, so there is no lm-decode row.",
    "The *-infer and lm-host-train-step rows are the CPU host binding and are never raced; "
    "their GPU twins are the *-forward, *-decode and mlp-predict rows.",
    "The GPT-3-small target shape is not on the board (the LM lanes use the smaller control "
    "shape so one shape runs on every box, a 16 GB Mac included).",
    # 2026-10-19 (lane ag-bench)
    "The fit lanes time one whole fit (one epoch at the recipe's batch size); the held-out "
    "prediction after it is not timed, and multi-epoch fits, early stopping and the classifier "
    "variants of the recurrent models are not raced.",
    # 2026-10-19 (lane ag-board-twins)
    # 2026-10-19 (lane ag-inference-board)
    "causal-lm-generate* time greedy generate at one prompt shape (batch 1); batched or sampled decoding, a "
    "stop token, a tokenizer in the clock and the low-bit weight formats are not raced; the memory figures come "
    "from an untimed pass after each round.",
    "sequential-fit times the fit and the held-out predict_proba together; ops-session-step times one "
    "recorded step (no multi-step loop, no optimizer state); the session's other operations are timed per "
    "operation by tools/ag_opbench.py, not on the board.",
    # 2026-10-19 (lane ag-program)
    "program-run times one Program.run of a two-layer forward (inference only); the eager session's time "
    "for the same forward is the A/B operation program_eager, not a board arm.",
]

# FAIRNESS FIX (2026-10-13 audit): the torch twins of these lanes are pure-PyTorch REFERENCE
# implementations (a per-token Python loop for Mamba-1, the chunked reference for Mamba-2/3,
# and the Samba stack built on the Mamba-3 reference), which this file's own docstring calls
# "not a deployment kernel"; they were raced and reported as the opponent. A reference
# implementation is not an opponent (an opponent is a library's own implementation through
# its public API). The torch arms of these lanes are now refused by name at race time
# (race(): no worker is started). mamba_ssm's fused kernels (MAMBA_SSM_VENDOR_ARMS, from
# lane/mamba-opp facde18cc) are the opponent where they are built. Results raced before this
# change were wrong for this lane; apply extra scrutiny.
REFERENCE_TWIN_LANES = {
    lane: ("not an opponent: the torch twin of %s is a pure-PyTorch reference implementation "
           "(mamba/corpus/gen_corpus.py%s), not a library kernel; mamba_ssm's fused kernels are "
           "the opponent only where they run (2026-10-13 fairness audit)"
           % (lane, " m3_forward inside the Samba stack" if lane.startswith("samba") else ""))
    # samba-train-step / samba-forward are not here: their torch arms now run mamba_ssm's Mamba3
    # layers (TorchArm._build_samba) and refuse by name where mamba_ssm does not import. The CPU
    # lanes stay: mamba_ssm's kernels need a GPU, so a CPU torch twin is always the reference.
    for lane in ("mamba1-forward", "mamba2-forward", "mamba3-forward", "mamba1-infer",
                 "mamba2-infer", "mamba3-infer", "samba-infer")}

# FAIRNESS FIX (2026-10-13 audit): when a compile arm timed out, the race silently promoted the
# eager arm to the opponent (AMD samba-train-step: both compile arms timed out and eager became
# the opponent, while compile took 1/3.8-1/3.4 of eager's time on NVIDIA). Now a compile arm that
# times out marks every other torch arm of its device class `not_opponent` (board mode
# "context", no ratio): torch's fastest setting is unmeasured, so the race has no opponent rather
# than another one. Results raced before this change were wrong for this lane; apply extra
# scrutiny.
PRECISIONS = ("fp32", "tf32", "bf16", "int8")
OPPONENT_RULE = ("a compile arm that times out leaves the race without an opponent (torch's "
                 "fastest setting is unmeasured) instead of promoting another arm (2026-10-13 "
                 "fairness audit)")


def arm_precision(arm):
    """'torch-compile-bf16' -> 'bf16'; 'mamba-ssm-tf32' -> 'tf32'; None when the name has none."""
    last = str(arm).rsplit("-", 1)[-1]
    return last if last in PRECISIONS else None


def mark_not_opponents(lane, arms_result):
    """Set `not_opponent` (the reason) on every completed non-mojolearn arm of a race result whose
    device class (GPU torch or CPU torch) has a compile arm that timed out: torch's fastest
    setting is then unmeasured and no other setting is promoted to the opponent.
    bench_board.classical_cells records such a cell as mode "context" with no ratio. Returns
    {arm: reason} for the arms it marked."""
    marked = {}

    def _ours(arm):
        return str(arm).startswith("ours")

    def _device_class(arm):
        return "cpu" if str(arm).startswith("torch-cpu-") else "gpu"
    for t, ta in arms_result.items():
        if _ours(t) or "-compile-" not in str(t) or ta.get("status") != "timeout":
            continue
        for arm, a in arms_result.items():
            if arm == t or _ours(arm) or a.get("not_opponent") or _device_class(arm) != _device_class(t):
                continue
            a["not_opponent"] = marked[arm] = (
                "%s timed out, so torch's fastest setting on %s is unmeasured; no other setting is "
                "promoted to the opponent" % (t, lane))
    return marked


def is_compile_arm(arm):
    """A torch.compile arm (torch-compile-*, torch-cpu-compile-*)."""
    return "-compile-" in str(arm)


def warmup_deadline(arm, args):
    """FAIRNESS FIX (2026-10-13 audit): the warm-up round's deadline. A compile arm compiles in
    its untimed warm-up and gets --compile-warmup-seconds (default 8 x the round deadline: 4 h on
    the full shape), not the eager arms' --warmup-seconds: samba-train-step's compile arms timed
    out at 1800 s and the eager arm silently became the opponent. Results raced before this
    change were wrong for this lane; apply extra scrutiny."""
    base = getattr(args, "warmup_seconds", 1800)
    if not is_compile_arm(arm):
        return base
    cw = getattr(args, "compile_warmup_seconds", None)
    return max(base, cw if cw else 8 * getattr(args, "round_seconds", base))


class RefusedArm:
    """FAIRNESS FIX (2026-10-13 audit): the stand-in for an arm a race refuses before starting
    its worker (a reference twin, a torch arm without ours' exported init): never alive, reads
    nothing, so every loop of the conductor skips it and its status stays "refused"."""
    alive = False
    info = None

    def read(self, *_a, **_k):
        return None

    def send(self, *_a, **_k):
        pass

    def kill(self, *_a, **_k):
        pass

    def close(self, *_a, **_k):
        pass

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
#: THE FIT LANES' RECIPE (2026-10-19, lane ag-bench), every value explicit and the same on every arm.
#: Rows: tools/sequence_speed.py's 1M HIGGS training rows (`full`) and a held-out block; `small` is the
#: smoke. The recurrent lanes cut consecutive rows into FIT_SEQ_LEN-step sequences, so they fit on
#: train // FIT_SEQ_LEN sequences (sequence_speed.case_recurrent).
FIT_ROWS = {"full": dict(train=1_000_000, test=100_000), "small": dict(train=4096, test=1024)}
FIT_SEQ_LEN = 16
#: HIGGS's feature count, for the settings text only (every arm reads its width from the data)
HIGGS_FEATURES = 28
FIT_BATCH, FIT_EPOCHS = 512, 1
FIT_MLP_HIDDEN = (256,)
FIT_RNN_HIDDEN = 64
FIT_TAPE_HIDDEN = (64, 64)
#: sequential-fit (2026-10-19, lane ag-board-twins): the synthetic set's width and the network's hidden
#: widths. Sizes a user's small tabular classifier has; no rule anywhere keys on them.
SEQFIT_FEATURES = 32
SEQFIT_HIDDEN = (128, 64)
#: the hidden widths of every feed-forward fit lane
FIT_HIDDEN = {"mlp-fit": FIT_MLP_HIDDEN, "tape-user-model": FIT_TAPE_HIDDEN, "sequential-fit": SEQFIT_HIDDEN,
              "sequential-predict": SEQFIT_HIDDEN}
#: causal-lm-generate (2026-10-19, lane ag-inference-board): prompt rows, prompt tokens and new tokens, `full` the
#: board and `small` the smoke. One row and a 128-token prompt is a chat-style single request; no rule keys on them.
GEN_SHAPES = {"full": dict(batch=1, prompt=128, new_tokens=128), "small": dict(batch=1, prompt=8, new_tokens=4)}
#: the checkpoints, fetched once by the conductor (huggingface_hub) into $MOJOLEARN_BOARD_LM_CHECKPOINTS/<key>
#: before any clock; the resolved commit is recorded with every race
LM_CHECKPOINTS = {
    "mojolearn-162m": dict(repo="mojolearn-ai/mojolearn-162m", revision="arxiv-v1",
                           files=("config.json", "model.safetensors"), fmt="mojolearn.byte-lm"),
    "smollm2-135m": dict(repo="HuggingFaceTB/SmolLM2-135M", revision="main",
                         files=("config.json", "model.safetensors", "generation_config.json"), fmt="llama"),
}
LM_CKPT_ENV = "MOJOLEARN_BOARD_LM_CHECKPOINTS"
LM_CKPT_DEFAULT = "~/mojolearn-data/hf-checkpoints"
#: the greedy rule both arms name (CausalLM: strict > scan from index 0; torch.argmax: the first maximal index)
GEN_DECODING = "greedy argmax, ties to the lowest id, KV cache, no stop token (exactly new_tokens per row)"
#: ops-session-step (2026-10-19, lane ag-board-twins): batch rows, input width, hidden width and classes of
#: the one step, `full` the board and `small` the smoke; power-of-two sizes of a typical MLP step, no rule keys
#: on them. SGD on every arm, every value explicit.
OPS_SHAPES = {"full": dict(batch=4096, d_in=256, hidden=512, classes=16),
              "small": dict(batch=64, d_in=16, hidden=32, classes=4)}
OPS_SGD = {"lr": 0.05, "momentum": 0.0, "weight_decay": 0.0}
#: the step's parameters, in this order on every arm (weights (inputs, outputs), biases (1, outputs))
OPS_PARAMS = ("w1", "b1", "w2", "b2")
#: Adam on every fit lane and arm (no weight decay, no amsgrad): ours' MLPClassifier (beta_1, beta_2,
#: epsilon), the recurrent estimators (optimizer_options), torch.optim.Adam and mlx.optimizers.Adam.
FIT_ADAM = {"lr": 1e-3, "betas": (0.9, 0.999), "eps": 1e-8, "weight_decay": 0.0}
#: Where the conductor reads HIGGS: --higgs, else this variable, else the first default that exists
#: (tools/sequence_speed.py's defaults).
HIGGS_ENV = "MOJOLEARN_BENCH_HIGGS"
HIGGS_DEFAULTS = ("~/data/higgs_speed.npz", "~/datasets/gbm-bench/higgs/higgs_speed.npz")
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
    if lane in DECODE_LANES and lane not in DECODE_TORCH_TWIN:
        return ()
    if lane in PREDICT_LANES:
        # 2026-10-19 (lane ag-inference-board): torch at every GPU setting of the box (no MLX twin yet)
        return tuple("torch-" + s for s in GPU_SETTINGS[vendor])
    if lane in FIT_LANES:
        # 2026-10-19 (lane ag-bench): torch at every GPU setting of the box, and MLX on Apple
        return tuple("torch-" + s for s in GPU_SETTINGS[vendor]) + (MLX_ARMS if vendor == "apple" else ())
    if DEVICE_OF[lane] == "cpu":
        arms = ["torch-cpu-" + s for s in CPU_SETTINGS]
    else:
        arms = ["torch-" + s for s in GPU_SETTINGS[vendor]]
    if lane in NO_COMPILE:
        arms = [a for a in arms if "-compile-" not in a]
    # FAIRNESS FIX (2026-10-13 audit): mamba_ssm's fused kernels are the Mamba opponent where
    # they are built (NVIDIA, AMD); the torch reference arms stay planned and are refused by name
    # at race time. Results raced before this change were wrong for this lane; apply extra
    # scrutiny.
    if lane in MAMBA_SSM_LANES:
        arms += list(MAMBA_SSM_VENDOR_ARMS.get(vendor, ()))
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
    elif model == "opsstep" or model == "program":
        return dict(OPS_SHAPES[shape])
    elif model == "causallm":
        return dict(GEN_SHAPES[shape], checkpoint=GEN_LANES[lane])
    elif lane in FIT_LANES:
        d = dict(FIT_ROWS[shape], batch=FIT_BATCH, epochs=FIT_EPOCHS)
        if lane in FIT_CELL:
            d.update(length=FIT_SEQ_LEN, hidden=FIT_RNN_HIDDEN)
        if lane in SYNTH_FIT_LANES:
            d.update(features=SEQFIT_FEATURES)
        return d
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
    elif lane in FIT_LANES:
        if lane in FIT_CELL:
            net = "%s h%d T%d" % (FIT_CELL[lane], FIT_RNN_HIDDEN, FIT_SEQ_LEN)
        else:
            net = "%d-%s-1" % (fit_d_in(lane), "-".join(str(h) for h in FIT_HIDDEN[lane]))
        rec["label"] = "%s train %d test %d %s b%d %d epoch adam" % (
            "synthetic" if lane in SYNTH_FIT_LANES else "HIGGS", d["train"], d["test"], net, d["batch"],
            d["epochs"]) + (" + predict_proba" if lane in FIT_PREDICT_IN_CLOCK else
                            ", predict_proba %d rows timed" % d["test"] if lane in PREDICT_LANES else "")
    elif model == "causallm":
        rec["label"] = "%s B%d prompt %d + %d greedy tokens" % (d["checkpoint"], d["batch"], d["prompt"],
                                                                d["new_tokens"])
    elif model == "opsstep":
        rec["label"] = "B%d %d-%d-%d linear relu linear CE sgd" % (d["batch"], d["d_in"], d["hidden"], d["classes"])
    elif model == "program":
        rec["label"] = "B%d %d-%d-%d linear relu linear softmax, one Program.run" % (
            d["batch"], d["d_in"], d["hidden"], d["classes"])
    else:
        rec["label"] = "B%d L%d DM%d" % (d["batch"], d["length"], d["d_model"])
    return rec


def shape_text(lane, shape):
    return shape_record(lane, shape)["label"] + (" (smoke)" if shape == "small" else "")


#: The board's per-lane settings text (what each side calls).
# FAIRNESS FIX (2026-10-13 audit): the LM, transformer and Samba torch calls name transformers' and
# mamba_ssm's own classes (they named the harness's twins); the Mamba torch calls name the refused
# references and the mamba-ssm arms. The changed text is part of the lane settings, so no cell
# stored under the old twins is reused. Results raced before this change were wrong for these
# lanes; apply extra scrutiny.
LANE_TEXT = {
    "lm-train-step": ("mojolearn.LanguageModelTrainer(resident=True, step_result='lean').train_step(ids)",
                      "transformers LlamaForCausalLM (ours' architecture: untied head, no final "
                      "norm, sdpa), ours' weights; zero_grad; forward + mean CE; backward; "
                      "torch.optim.AdamW step; loss.item()"),
    "lm-forward": ("mojolearn.LanguageModelTrainer(resident=True).logits(ids)",
                   "no_grad forward of the same transformers LlamaForCausalLM to logits; logits.cpu()"),
    "gemm": ("mojolearn.linalg.matmul(a, b)", "a.to(dev) @ b.to(dev), .cpu()"),
    "gemm-bf16": ("mojolearn.linalg.matmul_bf16(a_bf16, b_bf16) (bf16 bits made before the clock)",
                  "a_bf16.to(dev) @ b_bf16.to(dev) under the setting, .float().cpu()"),
    "gemm-int8": ("mojolearn.linalg.matmul_int8((a_codes, 0), (b_codes, 0)) = a @ b.T exactly",
                  "torch._int_mm(a.to(dev), b.to(dev).t()), .float().cpu()"),
    "lm-infer": ("mojolearn.LanguageModelInference(parameters, shape=cfg).logits(ids) (CPU)",
                 "no_grad forward of the same transformers LlamaForCausalLM on the CPU"),
    "lm-host-train-step": ("mojolearn.LanguageModelHostTrainer(parameters, shape=cfg).train_step(ids) (CPU)",
                           "the same transformers LlamaForCausalLM on the CPU; zero_grad; forward + "
                           "mean CE; backward; torch.optim.AdamW step; loss.item()"),
    "transformer-forward": ("mojolearn.TransformerBlock(weights, n_heads, n_kv_heads, head_dim).forward(x)",
                            "transformers LlamaDecoderLayer (sdpa, LlamaRotaryEmbedding), ours' weights"),
    "transformer-infer": ("mojolearn.TransformerBlockInference(...).forward(x) (CPU host binding)",
                          "transformers LlamaDecoderLayer on the CPU"),
    "mamba1-forward": ("mojolearn.Mamba1Block(weights).forward(x)",
                       "mamba-ssm-*: mamba_ssm Block(Mamba, fused fast path), ours' weights; the "
                       "torch-* reference arms (gen_corpus.py block_forward) are refused"),
    "mamba1-infer": ("mojolearn.Mamba1BlockInference(weights).forward(x) (CPU host binding)",
                     "gen_corpus.py block_forward on the CPU"),
    "mamba2-forward": ("mojolearn.Mamba2Block(weights).forward(x)",
                       "mamba-ssm-*: mamba_ssm Block(Mamba2, mem-eff SSD Triton path), ours' weights; "
                       "the torch-* reference arms (gen_corpus.py m2_forward) are refused"),
    "mamba2-infer": ("mojolearn.Mamba2BlockInference(weights).forward(x) (CPU host binding)",
                     "gen_corpus.py m2_forward on the CPU"),
    "mamba3-forward": ("mojolearn.Mamba3Block(weights).forward(x)",
                       "mamba-ssm-*: mamba_ssm Block(Mamba3 SISO Triton), ours' weights; the torch-* "
                       "reference arms (gen_corpus.py m3_forward) are refused"),
    "mamba3-infer": ("mojolearn.Mamba3BlockInference(weights).forward(x) (CPU host binding)",
                     "gen_corpus.py m3_forward on the CPU"),
    "samba-train-step": ("mojolearn.SambaStack(config, weights).train_step(inputs, targets)",
                         "torch Embedding + mamba_ssm Mamba3 Block / transformers LlamaDecoderLayer "
                         "layers + F.rms_norm + tied head; mean CE; backward; torch.optim.AdamW "
                         "step; loss.item()"),
    "samba-forward": ("mojolearn.SambaStack(config, weights).forward(inputs)",
                      "no_grad forward of the same mamba_ssm + transformers stack to logits"),
    "samba-infer": ("mojolearn.SambaInference(config, weights).forward(inputs) (CPU host binding)",
                    "the same stack twin on the CPU"),
    "mlp-train-step": ("mojolearn.SmallMLPTrainer(w1, b1, w2, b2).train_step(X, y)",
                       "F.linear, ReLU, F.linear; mean CE; backward; torch.optim.AdamW step"),
    "mlp-infer": ("mojolearn.MLPInference(w1, b1, w2, b2).predict_logits(X) (CPU host binding)",
                  "F.linear, ReLU, F.linear on the CPU"),
    "transformer-decode": ("mojolearn.TransformerBlock(weights, n_heads, n_kv_heads, head_dim)"
                           ".decode_session(state): state.cached_tokens = 0; load_state(); "
                           "step(x[:, t:t+1]) for t < L (resident GPU decode, one token per call)",
                           "transformers LlamaDecoderLayer over its own DynamicCache, one token per "
                           "call at position t (RoPE from LlamaRotaryEmbedding); x uploaded once, "
                           "outputs .cpu() once"),
    "mamba1-decode": ("mojolearn.Mamba1Block(weights).decode_session(state): load_state(); "
                      "step(x[:, t:t+1]) for t < L (resident GPU decode)", "none (ours alone)"),
    "mamba2-decode": ("mojolearn.Mamba2Block(weights): state = allocate_state(B); "
                      "step(x[:, t:t+1], state) for t < L", "none (ours alone)"),
    "mamba3-decode": ("mojolearn.Mamba3Block(weights): state = allocate_state(B); "
                      "step(x[:, t:t+1], state) for t < L", "none (ours alone)"),
    "samba-decode": ("mojolearn.SambaStack(config, weights): state = allocate_state(B, L); "
                     "step(ids[:, t], state) for t < L -> logits (B, L, vocab)", "none (ours alone)"),
    "mlp-predict": ("mojolearn.SmallMLPTrainer(w1, b1, w2, b2).predict_logits(X) (GPU)",
                    "F.linear, ReLU, F.linear on the GPU; .cpu()"),
    # 2026-10-19 (lane ag-bench): whole fits through the tape
    "mlp-fit": ("mojolearn.MLPClassifier(hidden_layer_sizes=(256,), activation='relu', solver='adam', "
                "alpha=0.0, batch_size=512, learning_rate_init=1e-3, max_iter=1, shuffle=True, "
                "random_state=7, tol=0.0, n_iter_no_change=1000).fit(X, y) (the tape path, the MLP's "
                "default)",
                "torch.nn.Sequential(Linear(28, 256), ReLU, Linear(256, 1)), Glorot-uniform init, "
                "X and y moved to the device, one epoch of torch.randperm minibatches of 512: zero_grad, "
                "BCE with logits, backward, torch.optim.Adam step; losses .cpu() (sklearn's Adam folds "
                "the bias correction into the step size, torch adds eps after it: same algorithm, eps "
                "placement differs)"),
    "lstm-fit": ("mojolearn.LSTMRegressor(hidden_size=64, num_layers=1, optimizer='adam', "
                 "learning_rate=1e-3, batch_size=512, max_epochs=1, shuffle=True, random_state=7).fit(X, y)",
                 "torch.nn.LSTM(28, 64, batch_first=True) + nn.Linear(64, 1) on the last hidden state, "
                 "MSE, torch.optim.Adam, one epoch of minibatches of 512, inputs moved inside the clock"),
    "gru-fit": ("mojolearn.GRURegressor(hidden_size=64, num_layers=1, optimizer='adam', "
                "learning_rate=1e-3, batch_size=512, max_epochs=1, shuffle=True, random_state=7).fit(X, y)",
                "torch.nn.GRU(28, 64, batch_first=True) + nn.Linear(64, 1) on the last hidden state, "
                "MSE, torch.optim.Adam, one epoch of minibatches of 512, inputs moved inside the clock"),
    "rnn-fit": ("mojolearn.RNNRegressor(hidden_size=64, num_layers=1, nonlinearity='tanh', optimizer='adam', "
                "learning_rate=1e-3, batch_size=512, max_epochs=1, shuffle=True, random_state=7).fit(X, y)",
                "torch.nn.RNN(28, 64, nonlinearity='tanh', batch_first=True) + nn.Linear(64, 1) on the last "
                "hidden state, MSE, torch.optim.Adam, one epoch of minibatches of 512, inputs moved inside "
                "the clock"),
    # ag-board-twins (2026-10-19): the calls the arm makes (Sequential.linear takes the output width)
    "tape-user-model": ("mojolearn.Sequential().linear(64).relu().linear(64).relu().linear(1); "
                        "compile(loss='bce_with_logits', optimizer='adam', lr=1e-3, beta1=0.9, beta2=0.999, "
                        "eps=1e-8, weight_decay=0.0, schedule='constant', tol=0.0, n_iter_no_change=0); "
                        "fit(X, y, epochs=1, batch_size=512, shuffle=True, seed=7)",
                        "the same torch.nn.Sequential, Glorot-uniform init, BCE with logits, "
                        "torch.optim.Adam, one epoch of minibatches of 512, inputs moved inside the clock"),
    # 2026-10-19 (lane ag-board-twins)
    "sequential-fit": ("mojolearn.Sequential().linear(128).relu().linear(64).relu().linear(1); "
                       "compile(loss='bce_with_logits', optimizer='adam', lr=1e-3, beta1=0.9, beta2=0.999, "
                       "eps=1e-8, weight_decay=0.0, schedule='constant', tol=0.0, n_iter_no_change=0); "
                       "fit(X, y, epochs=1, batch_size=512, shuffle=True, seed=7); predict_proba(X_test)",
                       "torch.nn.Sequential(Linear(32, 128), ReLU, Linear(128, 64), ReLU, Linear(64, 1)), "
                       "Glorot-uniform init, X and y moved to the device, one epoch of torch.randperm minibatches "
                       "of 512: zero_grad, BCE with logits, backward, torch.optim.Adam step; then the held-out "
                       "rows to the device, sigmoid of the logits, .cpu(), inside the clock"),
    "ops-session-step": ("mojolearn.ops.Session(record=True): tensor(X), tensor(y_codes), tensor(w1, b1, w2, b2, "
                         "requires_grad=True); relu(linear(x, w1, b1)); linear(h, w2, b2); cross_entropy(logits, "
                         "codes); backward(loss); scalar(loss); grad(p) for each parameter; step(optimizer('sgd', "
                         "lr), [w1, b1, w2, b2]): the SGD update p - lr * grad in place on the device from the "
                         "backward's gradients (no gradient re-upload); numpy(p) for each updated parameter",
                         "the same tensors .to(dev) as leaves with requires_grad; relu(x @ w1 + b1) @ w2 + b2; "
                         "F.cross_entropy(logits, y); backward; loss.item(); each .grad .cpu(); "
                         "torch.optim.SGD(lr, momentum=0, weight_decay=0).step(); each updated parameter .cpu()"),
    "program-run": ("mojolearn.ops.Session(): tensor(X), tensor(w1, b1, w2, b2); softmax(linear(relu(linear(x, w1, "
                    "b1)), w2, b2)); capture(inputs=[x], outputs=[p]) before the clock; Program.run(X) inside it: "
                    "X to the device, every captured operation, the probabilities back on the host, one call",
                    "the weights .to(dev) before the clock; under torch.no_grad: X .to(dev), softmax(relu(x @ w1 "
                    "+ b1) @ w2 + b2), .cpu()"),
    # 2026-10-19 (lane ag-inference-board)
    "sequential-predict": ("the sequential-fit network and fit (untimed, the warm-up round); then "
                           "predict_proba(X_test) inside the clock",
                           "the same torch.nn.Sequential fitted the same way (untimed, the warm-up round); then "
                           "X_test to the device, sigmoid of the (compiled, on a compile arm) forward under no_grad, "
                           ".cpu(), inside the clock"),
}
_GEN_OURS = ("mojolearn.models.CausalLM.load(dir, weight_format='float32', numeric_profile='fp32_v1', device='gpu') "
             "once (untimed, the warm-up round); generate(prompt_ids, 1) [prefill] and generate(prompt_ids, N) "
             "[the clock]: greedy, the prompt ids from the host, the ids back on the host")
_GEN_TORCH = ("transformers LlamaForCausalLM on the same checkpoint%s, .eval(), .to(dev) once (untimed); "
              "generate(input_ids, attention_mask, GenerationConfig(max_new_tokens=1 [prefill] or N [the clock], "
              "do_sample=False, num_beams=1, use_cache=True, eos_token_id=None)) under inference_mode, the ids "
              "moved to the device and back inside the clock; bf16 arms hold the weights in bfloat16")
LANE_TEXT.update({
    "causal-lm-generate": (_GEN_OURS, _GEN_TORCH % (" (the byte-LM tensors renamed to Llama's, loaded strict and "
                                                    "read back bit for bit; model.norm an Identity: the 162M has no "
                                                    "final norm)")),
    "causal-lm-generate-smollm2": (_GEN_OURS, _GEN_TORCH % " (from_pretrained, float32)"),
})


def lane_settings(lane):
    """The board's settings block for one lane (bench_board.race_settings)."""
    ours, theirs = LANE_TEXT[lane]
    s = {"ours_call": ours, "torch_call": theirs, "ours_device": DEVICE_OF[lane],
         "clock": "host inputs to the device, the call, the result back on the host, "
                  "synchronized" + ("; training state device-resident on both sides; round r "
                                    "is step r+1" if lane in TRAIN_LANES else "")
                  + ("; one round = a zero-state reset and L single-token decode calls, every "
                     "token's output back on the host (ours per call); weights resident "
                     "before the clock on both sides" if lane in DECODE_LANES else "")}
    s["seed_rule"] = ("seed %d: every parameter and input of every arm comes from the conductor's "
                 "default_rng(%d) file; torch arms also call torch.manual_seed(%d); ours' neural "
                 "classes take no seed argument (nothing in them draws)" % (SEED, SEED, SEED))
    s["explicit_params"] = _lane_params(lane)
    if lane in PROGRAM_LANES:
        # 2026-10-19 (lane ag-program): one round = one inference forward of the captured program
        s["clock"] = PROGRAM_SPAN["inside_clock"]
        s["seed_rule"] = ("seed %d: every weight and input of every arm comes from the conductor's "
                          "default_rng(%d) file; torch arms also call torch.manual_seed(%d); nothing in the "
                          "forward draws" % (SEED, SEED, SEED))
    if lane in GEN_LANES:
        # 2026-10-19 (lane ag-inference-board): one round = generate(prompt, 1) then generate(prompt, N)
        s["clock"] = GEN_SPAN["inside_clock"]
        s["seed_rule"] = ("seed %d: the prompt ids come from the conductor's default_rng(%d) file; greedy decoding "
                          "draws nothing" % (SEED, SEED))
        s["quality"] = ("tokens_equal_ours, first_divergence_index (new tokens), prefill_token_equal_ours, "
                        "new_ids_digest of every arm")
        s["memory"] = MEMORY_RULE
        s["torch_settings"] = {a: GEN_PRECISION.get(arm_setting(a)[1], precision_text(arm_setting(a)[1]))
                               for a in ["torch-" + x for x in TORCH_SETTINGS if x.startswith("eager")]}
        return s
    if lane in OPS_LANES:
        # 2026-10-19 (lane ag-board-twins): one round = the same one recorded step
        s["clock"] = OPS_SPAN["inside_clock"]
        s["seed_rule"] = ("seed %d: every weight, input and label of every arm comes from the conductor's "
                          "default_rng(%d) file; torch arms also call torch.manual_seed(%d); nothing in the step "
                          "draws" % (SEED, SEED, SEED))
        s["optimizer"] = ("SGD lr %g, momentum %g, weight decay %g, passed explicitly on every arm"
                          % (OPS_SGD["lr"], OPS_SGD["momentum"], OPS_SGD["weight_decay"]))
        s["quality"] = ("loss, loss_abs_diff_vs_ours, grad_max_abs_diff_vs_ours and param_max_abs_diff_vs_ours "
                        "per tensor, grad_digests of every arm")
        s["torch_settings"] = {a: precision_text(arm_setting(a)[1]) for a in
                               ["torch-" + x for x in TORCH_SETTINGS]}
        return s
    if lane in FIT_LANES:
        # 2026-10-19 (lane ag-bench): one round = one whole fit
        s["clock"] = fit_span(lane)["inside_clock"]
        s["seed_rule"] = ("seed %d on every arm: ours random_state=%d (its initialization and shuffle draw "
                          "in Mojo), torch.manual_seed(%d) before each fit, mx.random.seed(%d) on MLX; the "
                          "draws differ between libraries (a different random start, not a different "
                          "setting)" % (SEED, SEED, SEED, SEED))
        s["optimizer"] = ("Adam lr %g, betas %s, eps %g, no weight decay, no amsgrad, passed explicitly "
                          "on every arm" % (FIT_ADAM["lr"], FIT_ADAM["betas"], FIT_ADAM["eps"]))
        s["quality"] = ("held-out accuracy, log_loss, auc" if FIT_TASK[lane] == "classification"
                        else "held-out rmse, auc of the score") + (
            ", train_loss_last_step; *_diff_vs_ours for every other arm")
        if lane in PREDICT_LANES:
            s["memory"] = MEMORY_RULE
        return s
    if lane in TRAIN_LANES:
        s["optimizer"] = ("AdamW lr 1e-3, betas (0.9, 0.999), eps 1e-8, weight decay 0.01, no "
                          "amsgrad, passed explicitly on both" + (", no clipping (max_norm=None)"
                                                                   if MODEL_OF[lane] == "samba" else ""))
        s["quality"] = ("loss_first_step, loss_last_step (same init, same batches), "
                        "loss_first_abs_diff_vs_ours, loss_last_abs_diff_vs_ours")
    else:
        s["quality"] = "max_abs_diff_vs_ours, max_rel_diff_vs_ours" + (
            ", mean_nll" if lane in ("lm-forward", "samba-forward", "samba-infer",
                                     "samba-decode") else "") + (
            ", max_rel_err_vs_fp64" if MODEL_OF[lane] == "gemm" else "")
        if lane in FORWARD_REFERENCE_LANES:
            s["quality"] += ", untimed Mojo CPU host-reference finite/shape check and atol=1e-6 + rtol=1e-5 * abs(reference)"
    s["torch_settings"] = {a: precision_text(arm_setting(a)[1]) for a in
                           ["torch-" + x for x in TORCH_SETTINGS]}
    # FAIRNESS FIX (2026-10-13 audit): the reference-twin refusal is part of the lane's
    # settings, so the opponent store's settings key changes and no reference cell stored
    # before is reused as an opponent. Results raced before this change were wrong for this
    # lane; apply extra scrutiny.
    if lane in REFERENCE_TWIN_LANES:
        s["opponent_refused"] = REFERENCE_TWIN_LANES[lane]
    return s


def now_utc():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def outputs_digest(outs):
    """sha256 (first 16 hex) over a race's saved outputs {name: array}: every name in sorted order with its
    dtype, shape and C-order bytes. Recorded per arm as `outputs_digest` and printed as `NEURAL-DIGEST`,
    so the training lanes, whose per-round `digest` is None (their state changes every round), still carry
    one output hash per race for the NVIDIA == AMD identity check (tools/six_lane_grid_lq.py). Hashed after
    the timed rounds, from the arrays the race already saved; never inside a clock."""
    import numpy as np
    if not outs:
        return None
    h = hashlib.sha256()
    for name in sorted(outs):
        arr = np.ascontiguousarray(np.asarray(outs[name]))
        h.update(("%s|%s|%s;" % (name, arr.dtype.str, ",".join(str(int(s)) for s in arr.shape))).encode())
        h.update(arr.tobytes())
    return h.hexdigest()[:16]


def _load(name, alias=None):
    path = os.path.join(HERE, name + ".py")
    alias = alias or ("bbn_" + name)
    spec = importlib.util.spec_from_file_location(alias, path)
    mod = importlib.util.module_from_spec(spec)
    # Dynamo resolves a model class through its defining module's name. Match
    # normal Python imports before execution, including for the LM logits twin.
    previous = sys.modules.get(alias)
    sys.modules[alias] = mod
    try:
        spec.loader.exec_module(mod)
    except BaseException:
        if previous is None:
            sys.modules.pop(alias, None)
        else:
            sys.modules[alias] = previous
        raise
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
    source = "installed mojolearn package .py sources, sorted by path, concatenated"
    if not roots:
        # Opponent-only environments need the byte fixture, not our GPU binding.
        # Use this checkout's frozen sources and retain their hash/provenance.
        checkout = os.path.join(REPO, "python", "mojolearn")
        if not os.path.isfile(os.path.join(checkout, "__init__.py")):
            raise SystemExit("bench_board_neural: no installed or checkout mojolearn byte corpus in %s" % sys.executable)
        roots = [checkout]
        source = "repository mojolearn package .py sources, sorted by path, concatenated"
    root = roots[0]
    files = sorted(glob.glob(os.path.join(root, "**", "*.py"), recursive=True))
    raw = b"".join(open(f, "rb").read() for f in files)
    return raw, {"source": source,
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


def make_inputs(lane, shape, steps, path, *, mamba_spec=None):
    """Write the race's single input file (.npz) and return its record.

    smoke-fixes (2026-10-15; NVIDIA smoke-0845 neural/gemm/gaussian/shape=full STARTED at 5 s, check 6.79 s: the
    4096^3 GEMM's 128 MB of inputs were generated, written and hashed here, then read back and hashed again by
    every arm, all inside the recipe's timer): inside the campaign smoke's worker the file stage_inputs wrote
    before the recipe's timer (the same function, seed, lane, shape and steps, so the same bytes) is linked (or
    copied) to `path` and its record returned. A timed race (no smoke worker) writes the file here as always."""
    SI = smoke_staging()
    if SI is not None and mamba_spec is None:
        got = SI.STAGED.entries.get(_stage_key(lane, shape, steps)) or SI.STAGED.pinned.get(
            _stage_key(lane, shape, steps))
        if got is not None:
            staged_path, stamp, rec = got[0]
            try:
                if SI.file_stamp(staged_path) == stamp:
                    _place_file(staged_path, path)
                    return json.loads(json.dumps(rec))
            except OSError:
                pass                          # gone: write it here, as a timed race does
    return _write_inputs(lane, shape, steps, path, mamba_spec=mamba_spec)


def smoke_staging():
    """tools/smoke_inproc.py when this process is the campaign smoke's worker or one of its arm servers (they
    serve the --params-only parameter probe only), else None: a timed race or worker gets None here without
    importing anything, so it writes, reads and hashes its inputs exactly as before."""
    if os.environ.get("MOJOLEARN_SMOKE_IN_PROCESS") != str(os.getpid()):
        return None
    if HERE not in sys.path:
        sys.path.insert(0, HERE)
    import smoke_inproc
    return smoke_inproc if smoke_inproc.in_process() else None


def _stage_key(lane, shape, steps):
    return ("neural-inputs", lane, shape, int(steps), SEED)


def _place_file(src, dst):
    """dst holds src's bytes: a hard link where the file system allows it, else a copy."""
    import shutil
    if os.path.lexists(dst):
        os.remove(dst)
    try:
        os.link(src, dst)
    except OSError:
        shutil.copyfile(src, dst)


def _content_key(path):
    """An npz's members as its zip directory names them (name, size, CRC-32): the key of its exact bytes, read
    without reading the arrays (a link or copy of the same file has the same key)."""
    import zipfile
    with zipfile.ZipFile(path) as zf:
        return ("neural-npz",) + tuple(sorted((i.filename, i.file_size, i.CRC) for i in zf.infolist()))


def read_inputs(path):
    """{name: array} of a race's input file, what `{k: z[k] for k in np.load(path).files}` gives. Inside the
    smoke worker or its arm servers (smoke-fixes 2026-10-15, gemm STARTED at 5 s in this read, once per arm) the
    arrays stage_inputs made are returned when the file's zip directory names the same bytes, else the file is
    read once per process: every arm then shares the same arrays (and sha256_array hashes them once)."""
    import numpy as np

    def read():
        with np.load(path) as z:
            return {k: z[k] for k in z.files}
    SI = smoke_staging()
    if SI is None:
        return read()
    return dict(SI.STAGED.get(_content_key(path), read))


def stage_inputs(lane, dataset, data):
    """The smoke worker's stage-data step for one neural recipe (tools/smoke_worker.py, also in each arm server),
    outside every recipe's timer: the race's input file written once under data["dir"] (the same make_inputs
    body, seed, lane, shape and steps the recipe's race uses), its arrays kept for the parameter probes' reads
    and its record for make_inputs. A no-op outside the smoke worker. `dataset` is the lane's own (DATA_OF)."""
    SI = smoke_staging()
    if SI is None:
        return {"status": "not in the smoke worker"}
    import numpy as np
    shape, steps = data["shape"], int(data["steps"])
    os.makedirs(data["dir"], exist_ok=True)
    # one file per process: the worker and its arm servers each stage their own, never rewriting another's
    stem = os.path.join(data["dir"], "neural-%s-%s-steps%d." % (lane, shape, steps))
    for old in glob.glob(glob.escape(stem) + "*.npz"):     # an earlier smoke's process (gone): its file goes
        try:
            pid = int(old[len(stem):-len(".npz")])
            os.kill(pid, 0)
        except (ValueError, ProcessLookupError):
            try:
                os.remove(old)
            except OSError:
                pass
        except OSError:
            pass                                          # alive under another user: leave it
    path = "%s%d.npz" % (stem, os.getpid())
    rec = _write_inputs(lane, shape, steps, path)
    with np.load(path) as z:
        arrays = {k: z[k] for k in z.files}
    SI.STAGED.get(_content_key(path), lambda: arrays)
    key = _stage_key(lane, shape, steps)
    SI.STAGED.entries.pop(key, None)
    SI.STAGED.get(key, lambda: (path, SI.file_stamp(path), rec), nbytes=0)
    return {"status": "ok", "staged": ["neural %s %s steps=%d" % (lane, shape, steps)], "arrays": sorted(arrays),
            "file": path}


def _write_inputs(lane, shape, steps, path, *, mamba_spec=None):
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
        spec, (xshape, xrange) = (mamba_spec or mamba_weight_spec)(model, d)
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
    elif model == "opsstep" or model == "program":
        # program-run reads the same weights and batch (its labels unused)
        arrays.update(ops_arrays(shape, rng, rec))
    elif model == "causallm":
        arrays.update(gen_arrays(lane, shape, rng, rec))
    elif lane in FIT_LANES:
        arrays.update(fit_arrays(lane, shape, rec))
    np.savez(path, **arrays)
    rec["sha256"] = {k: _sha(np.ascontiguousarray(v).tobytes())[:16] for k, v in sorted(arrays.items())}
    rec["steps"] = steps
    return rec


def _weights(data):
    return {k[2:]: data[k] for k in data if k.startswith("w:")}


def higgs_path():
    """The HIGGS npz the fit lanes read (HIGGS_ENV, else the first of HIGGS_DEFAULTS that exists)."""
    got = os.environ.get(HIGGS_ENV, "").strip()
    if got:
        return os.path.expanduser(got)
    for cand in HIGGS_DEFAULTS:
        if os.path.exists(os.path.expanduser(cand)):
            return os.path.expanduser(cand)
    raise SystemExit("bench_board_neural: the fit lanes read HIGGS (gbm-bench/higgs/higgs_speed.npz from R2, "
                     "docs/REMOTE_DATA_R2.md); pass --higgs or set %s (looked in %s)"
                     % (HIGGS_ENV, ", ".join(HIGGS_DEFAULTS)))


def _npz_pick(z, stem):
    """z[stem_train] / z[stem_test]-style lookup as tools/sequence_speed.py load() does."""
    keys = list(z.keys())
    if stem in keys:
        return stem
    low = stem[0].lower()
    hits = [k for k in keys if k.lower().startswith(low) and k.lower().endswith(stem.split("_")[1])]
    return hits[0] if hits else None


def fit_arrays(lane, shape, rec):
    """The fit lanes' input arrays from HIGGS: X_train / y_train (the first FIT_ROWS[shape]["train"] rows
    of the training split, the rows tools/sequence_speed.py fits on) and X_test / y_test (the first
    FIT_ROWS[shape]["test"] rows of the npz's test split, else the training rows right after the fitted
    ones). Columns standardized with the fitted rows' float64 mean and std, stored float32. Labels 0/1 as
    float32 (the classifiers' class labels, the recurrent regressors' target). The recurrent lanes cut
    consecutive rows into FIT_SEQ_LEN-step sequences, the target the label of each sequence's last row
    (sequence_speed.case_recurrent). Bench-side preparation before every clock; no arm computes it."""
    if lane in SYNTH_FIT_LANES:
        return synthetic_fit_arrays(shape, rec)
    import numpy as np
    want = FIT_ROWS[shape]
    path = higgs_path()
    with np.load(path) as z:
        kx, ky = _npz_pick(z, "X_train"), _npz_pick(z, "y_train")
        if kx is None or ky is None:
            raise SystemExit("bench_board_neural: %s holds no X_train / y_train (keys %s)" % (path, list(z.keys())))
        X, y = z[kx], z[ky]
        kxt, kyt = _npz_pick(z, "X_test"), _npz_pick(z, "y_test")
        need = want["train"] + (0 if kxt is not None and kyt is not None else want["test"])
        if len(X) < need:
            raise SystemExit("bench_board_neural: %s has %d training rows; the %s fit lanes need %d"
                             % (path, len(X), shape, need))
        Xtr, ytr = X[:want["train"]], y[:want["train"]]
        if kxt is not None and kyt is not None and len(z[kxt]) >= want["test"]:
            Xte, yte = z[kxt][:want["test"]], z[kyt][:want["test"]]
            held = "%s[:%d], %s[:%d]" % (kxt, want["test"], kyt, want["test"])
        else:
            Xte, yte = X[want["train"]:want["train"] + want["test"]], y[want["train"]:want["train"] + want["test"]]
            held = "%s[%d:%d] (the npz has no test split)" % (kx, want["train"], want["train"] + want["test"])
    mu = Xtr.astype(np.float64).mean(0)
    sd = Xtr.astype(np.float64).std(0)
    sd[sd == 0] = 1.0
    out = {}
    for name, a, lab in (("train", Xtr, ytr), ("test", Xte, yte)):
        xs = ((a.astype(np.float64) - mu) / sd).astype(np.float32)
        lab = (np.asarray(lab, dtype=np.float64).ravel() > 0.5).astype(np.float32)
        if lane in FIT_CELL:
            n = len(xs) // FIT_SEQ_LEN
            xs = xs[:n * FIT_SEQ_LEN].reshape(n, FIT_SEQ_LEN, xs.shape[1])
            lab = lab[FIT_SEQ_LEN - 1:n * FIT_SEQ_LEN:FIT_SEQ_LEN]
        out["X_" + name] = np.ascontiguousarray(xs, dtype=np.float32)
        out["y_" + name] = np.ascontiguousarray(lab, dtype=np.float32)
    with open(path, "rb") as fh:
        head = fh.read(1 << 20)
    rec.update(higgs={"path": path, "bytes": os.path.getsize(path), "sha256_first_mib": _sha(head)[:16],
                      "train": "%s[:%d], %s[:%d]" % (kx, want["train"], ky, want["train"]), "held_out": held},
               inputs=("HIGGS, standardized with the fitted rows' float64 mean/std; labels 0/1"
                       + ("; %d-step sequences of consecutive rows, target the last row's label" % FIT_SEQ_LEN
                          if lane in FIT_CELL else "")),
               shapes={k: list(v.shape) for k, v in out.items()})
    return out


def synthetic_fit_arrays(shape, rec):
    """sequential-fit's data (2026-10-19, lane ag-board-twins): FIT_ROWS[shape] training rows then the
    held-out rows of one seeded synthetic binary set, no download. X is default_rng(SEED) standard normal
    float32 with SEQFIT_FEATURES columns (already standardized); the label is 1 where a fixed float64 teacher
    z = sum_j X[:, j] w[j] / sqrt(features) + 0.5 X[:, 0] X[:, 1] + 0.3 noise is positive, so a linear model
    is not enough and a small MLP reaches a high AUC. The teacher sums column by column in one fixed order
    (elementwise products, no BLAS), so the labels are the same bytes on every box. Bench-side preparation
    before every clock; no arm computes it."""
    import numpy as np
    want = FIT_ROWS[shape]
    n = want["train"] + want["test"]
    rng = np.random.default_rng(SEED)
    X = rng.standard_normal((n, SEQFIT_FEATURES), dtype=np.float32)
    w = rng.standard_normal(SEQFIT_FEATURES)
    noise = rng.standard_normal(n)
    z = np.zeros(n, dtype=np.float64)
    for j in range(SEQFIT_FEATURES):          # one fixed order, elementwise: the same bits on every box
        z += X[:, j].astype(np.float64) * w[j]
    z = z / np.sqrt(np.float64(SEQFIT_FEATURES)) + 0.5 * (X[:, 0].astype(np.float64) * X[:, 1].astype(np.float64)) \
        + 0.3 * noise
    y = (z > 0).astype(np.float32)
    ntr = want["train"]
    out = {"X_train": np.ascontiguousarray(X[:ntr]), "y_train": np.ascontiguousarray(y[:ntr]),
           "X_test": np.ascontiguousarray(X[ntr:]), "y_test": np.ascontiguousarray(y[ntr:])}
    rec.update(inputs=("synthetic binary set: default_rng(%d) X standard normal (%d features), label 1 where "
                       "X w / sqrt(%d) + 0.5 X0 X1 + 0.3 noise > 0 (w, noise standard normal from the same "
                       "generator; the teacher summed column by column, no BLAS)"
                       % (SEED, SEQFIT_FEATURES, SEQFIT_FEATURES)),
               positive_rate_train=float(y[:ntr].mean()),
               shapes={k: list(v.shape) for k, v in out.items()})
    return out


def lm_checkpoint(key):
    """The local directory of checkpoint `key` (LM_CHECKPOINTS) and its record: $MOJOLEARN_BOARD_LM_CHECKPOINTS/<key>
    (default ~/mojolearn-data/hf-checkpoints/<key>), fetched once with huggingface_hub.snapshot_download at the
    pinned revision when a file is missing, the resolved commit kept beside it (.board_revision.json). The
    conductor does this before any worker starts; no arm downloads."""
    spec = LM_CHECKPOINTS[key]
    root = os.path.expanduser(os.environ.get(LM_CKPT_ENV, "").strip() or LM_CKPT_DEFAULT)
    local = os.path.join(root, key)
    marker = os.path.join(local, ".board_revision.json")
    missing = [f for f in spec["files"] if not os.path.exists(os.path.join(local, f))]
    if missing:
        try:
            from huggingface_hub import HfApi, snapshot_download
        except ImportError as exc:
            raise SystemExit("bench_board_neural: %s needs %s (%s) at %s; it is missing %s and huggingface_hub "
                             "does not import (%r)" % (key, spec["repo"], spec["revision"], local, missing, exc))
        os.makedirs(local, exist_ok=True)
        snapshot_download(repo_id=spec["repo"], revision=spec["revision"], local_dir=local,
                          allow_patterns=list(spec["files"]))
        sha = None
        try:
            sha = HfApi().model_info(spec["repo"], revision=spec["revision"]).sha
        except Exception:  # noqa: BLE001  (recorded as unknown, never guessed)
            pass
        with open(marker, "w") as fh:
            json.dump({"repo": spec["repo"], "revision": spec["revision"], "commit": sha, "fetched": now_utc()}, fh)
    got = {}
    try:
        with open(marker) as fh:
            got = json.load(fh)
    except (OSError, ValueError):
        pass
    still = [f for f in spec["files"] if not os.path.exists(os.path.join(local, f))]
    if still:
        raise SystemExit("bench_board_neural: %s: %s still lacks %s" % (key, local, still))
    return local, {"checkpoint": key, "repo": spec["repo"], "revision": spec["revision"],
                   "commit": got.get("commit"), "dir": local, "format": spec["fmt"],
                   "files_sha256": {f: _file_sha256(os.path.join(local, f)) for f in spec["files"]}}


def _file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def gen_arrays(lane, shape, rng, rec):
    """causal-lm-generate's inputs (2026-10-19, lane ag-inference-board): the checkpoint directory (fetched here,
    before any clock) and the prompt ids (batch, prompt) int32 drawn from the race's generator over the
    checkpoint's whole vocabulary, so every arm reads the same ids; no tokenizer runs."""
    import numpy as np
    d = GEN_SHAPES[shape]
    local, ck = lm_checkpoint(GEN_LANES[lane])
    with open(os.path.join(local, "config.json")) as fh:
        vocab = int(json.load(fh)["vocab_size"])
    ids = rng.integers(0, vocab, (d["batch"], d["prompt"])).astype(np.int32)
    rec["inputs"] = ("prompt ids default_rng(%d).integers(0, %d) int32, shape (%d, %d); %d new tokens"
                     % (SEED, vocab, d["batch"], d["prompt"], d["new_tokens"]))
    rec["checkpoint"] = ck
    return {"prompt_ids": ids, "new_tokens": np.asarray(d["new_tokens"], dtype=np.int64),
            "checkpoint_dir": np.asarray(local), "checkpoint_key": np.asarray(GEN_LANES[lane])}


def ops_arrays(shape, rng, rec):
    """ops-session-step's inputs (2026-10-19, lane ag-board-twins), from the race's generator: w1 (d_in,
    hidden), b1 (1, hidden), w2 (hidden, classes), b2 (1, classes) uniform(+-1/sqrt(fan_in)); X (batch, d_in)
    standard normal; labels integers 0..classes-1, stored as int64 `y` (torch's F.cross_entropy) and as the
    float32 codes (batch, 1) `y_codes` that ops.Session.cross_entropy takes, so no arm converts them."""
    import numpy as np
    d = OPS_SHAPES[shape]
    di, h, k, bsz = d["d_in"], d["hidden"], d["classes"], d["batch"]
    out = {}
    for name, s, fan_in in (("w1", (di, h), di), ("b1", (1, h), di), ("w2", (h, k), h), ("b2", (1, k), h)):
        bound = 1.0 / np.sqrt(fan_in)
        out["w:" + name] = rng.uniform(-bound, bound, s).astype(np.float32)
    out["X"] = rng.standard_normal((bsz, di)).astype(np.float32)
    labels = rng.integers(0, k, bsz)
    out["y"] = labels.astype(np.int64)
    out["y_codes"] = labels.astype(np.float32).reshape(bsz, 1)
    rec["inputs"] = ("weights uniform(+-1/sqrt(fan_in)) (weights (inputs, outputs), biases (1, outputs)); X "
                     "standard normal; labels 0..%d (int64 and float32 codes)" % (k - 1))
    return out


# ---------------------------------------------------------------------------
# Workers: ours (one process per arm)
# ---------------------------------------------------------------------------

def _ours_mode():
    """The tier this `ours*` worker was started for: `ours` is identical,
    `ours-fast` fast (the conductor's _worker_env sets the variable per arm)."""
    want = os.environ.get("MOJOLEARN_NUMERIC_MODE", "identical").strip().lower()
    if want not in ("identical", "fast"):
        raise RuntimeError("REFUSED: the neural surface races identical (ours) and fast "
                           "(ours-fast, the Apple tier) only; this worker was started under "
                           "MOJOLEARN_NUMERIC_MODE=%s" % want)
    return want


def _ours_module():
    _ours_mode()
    import mojolearn
    return mojolearn


def _ours_info(ml, module_path, mode_used, device="gpu"):
    want = _ours_mode()
    info = {"library": "mojolearn", "version": getattr(ml, "__version__", "unknown"),
            "numeric_mode_env": os.environ.get("MOJOLEARN_NUMERIC_MODE"),
            "numeric_mode_used": mode_used, "device": device,
            "module_path": module_path, "pre_clock_fit": False,
            "input_home": "host"}
    try:
        info["vendor_used"] = ml.vendor()
    except Exception as exc:  # noqa: BLE001
        info["vendor_used"] = "unavailable (%r)" % (exc,)
    if mode_used != want:
        raise RuntimeError("ours is not %s: read back %r" % (want.upper(), mode_used))
    # The public library reports the API (metal); the board names its column
    # apple. Accept either spelling, while refusing CUDA/HIP/CPU FAST here.
    if want == "fast" and info["vendor_used"] not in ("metal", "apple"):
        raise RuntimeError("REFUSED: ours-fast is the Apple FAST neural tier; this box's vendor "
                           "reads back %r" % (info["vendor_used"],))
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
    if lane in FIT_LANES:
        return fit_params(lane)
    if lane in OPS_LANES:
        return ops_params()
    if lane in PROGRAM_LANES:
        return program_params()
    if lane in GEN_LANES:
        return gen_params(lane)
    if lane in TRAIN_LANES:
        out.update(lr=ADAMW["lr"], betas=list(ADAMW["betas"]), eps=ADAMW["eps"],
                   weight_decay=ADAMW["weight_decay"], amsgrad=False)
    elif model == "transformer":
        out.update(eps=BLOCK_NORM_EPS, rope_theta=BLOCK_ROPE_THETA)
    elif model == "samba":
        out.update(eps=SAMBA_NORM_EPS, rope_theta=BLOCK_ROPE_THETA)
    if model == "samba":
        # smoke-fixes (2026-10-14; neural/samba-train-step: "max_grad_norm is unset on ours and unset on
        # torch-*"): clipping is OFF on both arms, pinned as the explicit value "off" (not None, which the
        # parameter check reads as a library default): ours reads it back from SambaStack.max_norm (None =
        # clipping off, python/mojolearn/_samba_impl.py SambaStack docstring "None turns clipping off"), the
        # torch arm's call() runs no clip_grad_norm_ (TorchArm.call: zero_grad, backward, opt.step).
        out.update(dropout=SAMBA_DROPOUT, max_grad_norm=SAMBA_CLIP_OFF if lane in TRAIN_LANES else "n/a")
    return out


def fit_layers(lane, d_in, hidden, d_out=1):
    """The network of a fit lane as every arm reads it back from what it built: one string per layer."""
    if lane in FIT_CELL:
        return ["%s %d %d bias batch_first" % (FIT_CELL[lane], d_in, int(hidden[0])),
                "linear %d %d bias (last time step)" % (int(hidden[0]), d_out)]
    sizes = [int(d_in)] + [int(h) for h in hidden] + [int(d_out)]
    out = []
    for i in range(len(sizes) - 1):
        out.append("linear %d %d bias" % (sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2:
            out.append("relu")
    return out


def fit_d_in(lane):
    """The input width of a fit lane's data (HIGGS's 28 features, or the synthetic set's)."""
    return SEQFIT_FEATURES if lane in SYNTH_FIT_LANES else HIGGS_FEATURES


def ops_layers(d_in, hidden, classes):
    """ops-session-step's composition as every arm reads it back from the tensors it built."""
    return ["linear %d %d bias" % (int(d_in), int(hidden)), "relu",
            "linear %d %d bias" % (int(hidden), int(classes)), "softmax cross_entropy mean"]


def ops_params(shape="full", **readback):
    """ops-session-step's work settings, the same explicit values on every arm (readback overrides a value
    with what the arm read back from what it built, so a difference shows as a mismatch)."""
    d = OPS_SHAPES[shape]
    out = dict(optimizer="sgd", learning_rate=OPS_SGD["lr"], momentum=OPS_SGD["momentum"],
               weight_decay=OPS_SGD["weight_decay"], nesterov=False, loss="cross_entropy", reduction="mean",
               batch_size=int(d["batch"]), steps=1, layers=ops_layers(d["d_in"], d["hidden"], d["classes"]),
               seed=SEED)
    for k, v in readback.items():
        if k not in out:
            raise KeyError("ops_params: %r is not one of the lane's settings" % (k,))
        out[k] = list(v) if isinstance(v, tuple) else v
    return out


def ops_record(library, implementation, shape="full", **readback):
    """An ops-session-step arm's parameter record for tools/bench_board_params.py."""
    rec = {"__library__": library}
    if implementation is not None:
        rec["__implementation__"] = implementation
    rec.update(ops_params(shape, **readback))
    return rec


def program_layers(d_in, hidden, classes):
    """program-run's forward as every arm reads it back from what it built."""
    return ["linear %d %d bias" % (int(d_in), int(hidden)), "relu",
            "linear %d %d bias" % (int(hidden), int(classes)), "softmax"]


def program_params(shape="full", **readback):
    """program-run's work settings, the same explicit values on every arm (readback overrides a value with
    what the arm read back from what it built)."""
    d = OPS_SHAPES[shape]
    out = dict(mode="inference", batch_size=int(d["batch"]), steps=1,
               layers=program_layers(d["d_in"], d["hidden"], d["classes"]), seed=SEED)
    for k, v in readback.items():
        if k not in out:
            raise KeyError("program_params: %r is not one of the lane's settings" % (k,))
def gen_params(lane, shape="full", **readback):
    """causal-lm-generate's work settings (2026-10-19, lane ag-inference-board), the same explicit values on
    every arm; `readback` overrides a value with what the arm read back from the model it built (its config
    and plan), so a difference shows as a mismatch. The architecture fields are None here (the checkpoint
    decides them) and every arm reads them back."""
    d = GEN_SHAPES[shape]
    spec = LM_CHECKPOINTS[GEN_LANES[lane]]
    out = dict(checkpoint=GEN_LANES[lane], checkpoint_repo=spec["repo"], checkpoint_revision=spec["revision"],
               batch_size=int(d["batch"]), prompt_tokens=int(d["prompt"]), new_tokens=int(d["new_tokens"]),
               decoding=GEN_DECODING, seed=SEED, n_layers=None, d_model=None, n_heads=None, n_kv_heads=None,
               vocab_size=None, final_norm=None, tied_head=None, rope_theta=None, norm_eps=None)
    for k, v in readback.items():
        if k not in out:
            raise KeyError("gen_params: %r is not one of the lane's settings" % (k,))
        out[k] = list(v) if isinstance(v, tuple) else v
    return out


def program_record(library, implementation, shape="full", **readback):
    """A program-run arm's parameter record for tools/bench_board_params.py."""
    rec = {"__library__": library}
    if implementation is not None:
        rec["__implementation__"] = implementation
    rec.update(program_params(shape, **readback))
def gen_record(lane, library, implementation, shape="full", **readback):
    """A causal-lm-generate arm's parameter record for tools/bench_board_params.py."""
    rec = {"__library__": library}
    if implementation is not None:
        rec["__implementation__"] = implementation
    rec.update(gen_params(lane, shape, **readback))
    return rec


def fit_params(lane, d_in=None, **readback):
    """The work settings of a fit lane, the same explicit values on every arm (the board's parameter
    check compares them by these names). `readback` overrides a value with what the arm read back
    from the object it built, so a difference shows as a mismatch, never as the lane's constant."""
    d_in = fit_d_in(lane) if d_in is None else d_in
    hidden = (FIT_RNN_HIDDEN,) if lane in FIT_CELL else FIT_HIDDEN[lane]
    out = dict(optimizer="adam", learning_rate=FIT_ADAM["lr"], betas=list(FIT_ADAM["betas"]),
               eps=FIT_ADAM["eps"], weight_decay=FIT_ADAM["weight_decay"], amsgrad=False, maximize=False,
               batch_size=FIT_BATCH, epochs=FIT_EPOCHS, shuffle=True, layers=fit_layers(lane, d_in, hidden),
               seed=SEED)
    if lane == "mlp-fit":
        # alpha: scikit-learn's L2 term, 0 so that no arm adds a penalty (torch's loss has none)
        out.update(loss="log_loss", alpha=0.0)
    elif lane in FIT_CELL:
        out.update(loss="mse", num_layers=1, timesteps=FIT_SEQ_LEN)
        if lane == "rnn-fit":
            out["nonlinearity"] = "tanh"
    else:
        out.update(loss="bce_with_logits")
    for k, v in readback.items():
        if k not in out:
            raise KeyError("fit_params(%s): %r is not one of the lane's settings" % (lane, k))
        out[k] = list(v) if isinstance(v, tuple) else v
    return out


def fit_record(lane, library, implementation, d_in, **readback):
    """A fit arm's parameter record for tools/bench_board_params.py."""
    rec = {"__library__": library}
    if implementation is not None:
        rec["__implementation__"] = implementation
    rec.update(fit_params(lane, d_in, **readback))
    return rec


def _ours_record(lane, config_readback=None):
    """Ours' parameter record for tools/bench_board_params.py: the values
    passed to the constructor (a declared dict; our neural classes take no
    seed: every parameter and input comes from the conductor's seed-7 file)."""
    rec = {"__library__": "mojolearn"}
    rec.update(_lane_params(lane))
    if lane in TRAIN_LANES:
        for flag, value in OURS_ADAMW_SEMANTICS.items():
            rec.setdefault(flag, value)
    if config_readback is not None:
        rec["config_readback"] = json.dumps(config_readback, sort_keys=True, default=str)
    return rec


#: The architecture fields both arms of a model lane read back and the board compares as one value,
#: `config_readback` (preflight-smoke, 2026-10-14: the torch arm had none, so every LM and Samba race refused
#: with "config_readback missing from torch-* read-back"). Ours reads them from its own config object; the
#: torch arm from the transformers config and modules it actually built. A field one side cannot read back
#: is None there and the race refuses by name.
ARCH_FIELDS = {
    "lm": ("layers", "d_model", "n_heads", "n_kv_heads", "head_dim", "intermediate", "vocab", "tied_head",
           "param_dtype"),
    "samba": ("layers", "d_model", "n_heads", "n_kv_heads", "head_dim", "intermediate", "vocab", "tied_head",
              "norm_eps"),
    # refused-param-mappings (2026-10-15): the MLP lanes compared no setting at all (ours' record was empty,
    # the torch arm's only its seed), so a twin of another width or activation would have matched. Both arms
    # now read back the network they run: Linear(d_in, hidden) -> activation -> Linear(hidden, d_out).
    "mlp": ("d_in", "hidden", "d_out", "activation", "bias"),
}


def mlp_arch(d_in, hidden, d_out, activation, bias):
    """The config_readback of a two-layer MLP lane (ARCH_FIELDS['mlp'])."""
    return arch_readback("mlp", dict(d_in=d_in, hidden=hidden, d_out=d_out, activation=activation, bias=bias))


def arch_readback(model, values):
    """The canonical config_readback JSON of `model` ('lm' or 'samba') from read-back `values`: exactly the
    ARCH_FIELDS keys, ints as ints, floats at float32 precision (both libraries store eps as float32)."""
    import struct
    out = {}
    for key in ARCH_FIELDS[model]:
        v = values.get(key)
        if isinstance(v, bool) or v is None or isinstance(v, str):
            out[key] = v
        elif isinstance(v, (list, tuple)):
            out[key] = [str(x) for x in v]
        elif isinstance(v, float):
            out[key] = repr(struct.unpack("f", struct.pack("f", v))[0])
        else:
            out[key] = int(v)
    return json.dumps(out, sort_keys=True)


def ours_lm_arch(config, profile):
    """Ours' LM architecture from the ByteLanguageModelConfig fields the trainer holds (`config`, a
    ByteLanguageModelConfig.to_dict()) and its native profile name. Ours' parameter registry holds a separate
    lm_head (untied; python/mojolearn/_byte_lm_config.py parameter_names) and the profile names its parameter
    precision."""
    # smoke-fixes (2026-10-14; neural/lm-forward, lm-train-step: "config_readback is '{"d_model": null, ...}'"):
    # run_metadata()['config'] is the OPTIMIZER configuration (kind, lr, beta1, ...; _byte_lm_impl.py
    # _configuration), never the architecture, so every dimension read back None. The architecture is the
    # trainer's held model shape (see ours_lm_held_shape).
    c = config or {}
    prof = str(profile or "")
    return arch_readback("lm", dict(layers=c.get("n_layers"), d_model=c.get("d_model"), n_heads=c.get("n_heads"),
                                    n_kv_heads=c.get("n_kv"), head_dim=c.get("head_dim"),
                                    intermediate=c.get("intermediate"), vocab=c.get("vocab_size"),
                                    tied_head=False,
                                    param_dtype="float32" if ".fp32." in prof else (prof or None)))


def ours_lm_held_shape(trainer):
    """The model dimensions a LanguageModelTrainer actually holds, decoded by the library's own
    `state_shape` (python/mojolearn/_byte_lm_config.py state_shape: the state's 'model_shape', or the
    default ByteLanguageModelConfig when the state carries none; the same decoder its state validation
    uses, _byte_lm_impl.py _validate_state). Reads host metadata only: no device download."""
    import importlib
    mod = importlib.import_module(type(trainer).__module__.rsplit(".", 1)[0] + "._byte_lm_config")
    return mod.state_shape(trainer._state).to_dict()


def ours_samba_arch(config):
    """Ours' Samba architecture from SambaConfig.to_dict() (python/mojolearn/_samba_impl.py)."""
    c = config or {}
    return arch_readback("samba", dict(layers=c.get("layers"), d_model=c.get("d_model"), n_heads=c.get("n_heads"),
                                       n_kv_heads=c.get("n_kv_heads"), head_dim=c.get("head_dim"),
                                       intermediate=c.get("intermediate"), vocab=c.get("vocab"),
                                       tied_head=c.get("tie_embeddings"), norm_eps=c.get("norm_eps")))


def _gemm_record(lane, library, implementation, data):
    """Witness the actual library callable and the shared input values, untimed."""
    # FAIRNESS FIX (2026-10-13 audit): GEMM previously reported no work
    # parameters and no function identity, so an empty comparison could pass.
    import numpy as np
    arrays = {k: np.ascontiguousarray(data[k]) for k in ("a", "b")}
    return dict(__library__=library, __implementation__=implementation, seed=SEED,
                operation="matmul", transpose_b=lane == "gemm-int8",
                operand_value_type={"gemm":"float32", "gemm-bf16":"bfloat16", "gemm-int8":"int8"}[lane],
                input_arrays={k: dict(shape=list(a.shape), dtype=str(a.dtype), sha256=_array_sha(a))
                              for k, a in arrays.items()})


def _array_sha(a):
    """sha256 of a C-contiguous array's bytes; inside the smoke worker hashed once per staged array
    (smoke_inproc.sha256_array: the same digest), not once per arm (smoke-fixes 2026-10-15, gemm)."""
    SI = smoke_staging()
    return _sha(a.tobytes()) if SI is None else SI.sha256_array(a)


def _read_neural_ab_config(path):
    """Read selected source-arm metadata; this never builds or admits a binary."""
    if not path:
        return None
    with open(path, encoding="utf-8") as fh:
        config = json.load(fh)
    if not isinstance(config, dict) or config.get("mode") != "identical":
        raise ValueError("neural A/B config requires an object with mode='identical'")
    if not isinstance(config.get("id"), str) or not isinstance(config.get("arm"), str):
        raise ValueError("neural A/B config requires string id and arm")
    definitions = config.get("compile_defines", [])
    if (not isinstance(definitions, list)
            or any(not isinstance(v, str) or "=" not in v or v.startswith("-D") for v in definitions)):
        raise ValueError("neural A/B compile_defines must be bare NAME=value strings")
    env = config.get("environment", {})
    if (not isinstance(env, dict)
            or any(not isinstance(k, str) or not isinstance(v, str) for k, v in env.items())):
        raise ValueError("neural A/B environment requires string names and values")
    if env.get("MOJOLEARN_NUMERIC_MODE", "identical") != "identical":
        raise ValueError("neural A/B environment cannot override the IDENTICAL mode")
    unset = config.get("environment_unset", [])
    if not isinstance(unset, list) or any(not isinstance(k, str) for k in unset):
        raise ValueError("neural A/B environment_unset requires a list of names")
    if not isinstance(config.get("runtime", {}), dict):
        raise ValueError("neural A/B runtime must be an object")
    return config


_NEURAL_AB_TRANSFORMER_FULL = dict(BLOCK_SHAPES["transformer"]["full"])


def _apply_neural_ab_environment(config):
    BLOCK_SHAPES["transformer"]["full"] = dict(_NEURAL_AB_TRANSFORMER_FULL)
    if config is not None:
        for name in config.get("environment_unset", []):
            os.environ.pop(name, None)
        os.environ.update(config.get("environment", {}))
        os.environ["MOJOLEARN_NUMERIC_MODE"] = "identical"
        runtime = config.get("runtime", {})
        dimensions = runtime.get("transformer_config")
        if dimensions is not None:
            # Explicit saved workload, never a kernel dispatch/board-row rule.
            # The conductor and worker install the same complete dimensions
            # before fixture creation, shape reporting or model construction.
            fields = {"batch", "length", "d_model", "n_heads", "n_kv", "head_dim", "intermediate"}
            if (runtime.get("operation") != "transformer_configured_forward"
                    or not isinstance(dimensions, dict) or set(dimensions) != fields
                    or any(type(v) is not int or v < 1 or v > (1 << 20) for v in dimensions.values())):
                raise ValueError("explicit transformer workload requires all positive dimensions and its own operation")
            if (dimensions["d_model"] != dimensions["n_heads"] * dimensions["head_dim"]
                    or dimensions["n_heads"] % dimensions["n_kv"] or dimensions["head_dim"] % 2):
                raise ValueError("explicit transformer workload has incompatible head dimensions")
            BLOCK_SHAPES["transformer"]["full"] = dict(dimensions)


def _neural_ab_operation(config):
    return (config or {}).get("runtime", {}).get("operation", "model_default")


def _neural_ab_receipt(runner, config, executed=False):
    if config is None:
        return None
    # A supplied define is configuration, not evidence of the loaded binary
    # or of a conditional kernel branch actually running. Native metadata
    # below may prove ownership/retention, never unobserved kernel coverage.
    receipt = {"id": config["id"], "arm": config["arm"],
               "configured": config, "operation": _neural_ab_operation(config),
               "public_operation_completed": bool(executed),
               "compile_defines_status": "requested_not_binary_verified",
               "kernel_reach_status": "unproven_without_native_counter",
               "full_affected_estimator_coverage": "not_established_by_one_workload"}
    metadata = getattr(runner, "neural_ab_runtime", None)
    if metadata is not None:
        receipt["native_or_public_runtime_metadata"] = dict(metadata)
        if metadata.get("operation") == "mamba3_owned_forward":
            # Native counter readback is outside the measured operation.
            receipt["native_or_public_runtime_metadata"]["session_info"] = runner.block.session_info()
    return receipt


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

    def __init__(self, lane, shape, data, ab_runtime=None):
        import numpy as np
        self.np = np
        ml = _ours_module()
        dims = lm_dims(lane, shape)
        cfg = ml.LanguageModelConfig(*dims)
        runtime = ab_runtime or {}
        if runtime.get("chunked_lm_head_v2", False):
            if lane != "lm-train-step":
                raise RuntimeError("REFUSED: chunked_lm_head_v2 operation requires lm-train-step")
            from dataclasses import replace
            cfg = replace(cfg, chunked_lm_head_v2=True)
        self.neural_ab_runtime = {
            "chunked_lm_head_v2_requested": bool(runtime.get("chunked_lm_head_v2", False)),
            "chunked_lm_head_v2_config": bool(getattr(cfg, "chunked_lm_head_v2", False)),
            "kernel_reach": "unproven_without_native_counter"}
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
        mode = {0: "fast", 1: "identical"}.get(
            int(meta.get("native_numeric_mode", -1)),
            "native_numeric_mode=%s" % meta.get("native_numeric_mode"))
        self.info = _ours_info(ml, meta.get("binding_file"), mode)
        self.info.update(profile=meta.get("native_profile"), native_vendor=meta.get("native_vendor"),
                         call=LANE_TEXT[lane][0], config=json.dumps(meta.get("config"), sort_keys=True))
        self.k = 0
        self.losses = []
        self.ids = np.ascontiguousarray(self.batches[0][:, :-1])
        self.record = _ours_record(lane)
        # the canonical architecture read-back (ARCH_FIELDS); the raw config stays in info["config"]
        self.record["config_readback"] = ours_lm_arch(ours_lm_held_shape(self.trainer), meta.get("native_profile"))

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


def fast_bf16_capability():
    """Read the actual installed native capability without submitting GPU work."""
    ml = _ours_module()
    import mojolearn.linalg as linalg
    from mojolearn import _linalg_impl
    binding = _linalg_impl._load()
    getter = getattr(binding, "fast_bf16_available", None)
    compiled_mode = int(binding.linalg_numeric_mode())
    enabled = bool(getter()) if getter is not None else False
    # A binary flag alone is insufficient: an older generic wrapper still
    # calls require_identical. Verify the actual public entrypoint aliases.
    aliases = {name: getattr(linalg, name, None) is getattr(binding, native, None)
               and getattr(binding, native, None) is not None
               for name, native in (("matmul_bf16", "fast_matmul_bf16"),
                                    ("to_bf16", "public_fast_to_bf16"),
                                    ("from_bf16", "public_fast_from_bf16"))}
    supported = compiled_mode == 0 and enabled and all(aliases.values())
    module_path = os.path.abspath(binding.__file__)
    with open(module_path, "rb") as fh:
        binary_sha256 = hashlib.sha256(fh.read()).hexdigest()
    return dict(schema="mojolearn.fast-bf16-capability.v1", supported=supported,
                native_enabled=enabled, compiled_mode=compiled_mode, aliases=aliases,
                version=getattr(ml, "__version__", None), module_path=module_path,
                binary_sha256=binary_sha256, gpu_work_submitted=False,
                reason="native Apple FAST BF16 and generic aliases available" if supported else
                       "installed binding lacks enabled Apple FAST BF16 or its native generic aliases")


class OursGEMM(Ours):
    def __init__(self, lane, shape, data):
        import numpy as np
        self.np = np
        ml = _ours_module()
        import mojolearn.linalg as linalg
        self.linalg = linalg
        self.lane = lane
        mode = linalg.numeric_mode()
        capability = None
        if lane == "gemm-bf16" and mode == "fast":
            capability = fast_bf16_capability()
            if not capability["supported"]:
                raise RuntimeError("UNSUPPORTED: " + capability["reason"])
        self.a, self.b = np.ascontiguousarray(data["a"]), np.ascontiguousarray(data["b"])
        if lane == "gemm-bf16":           # the bf16 bits, made before the clock
            self.a, self.b = linalg.to_bf16(self.a), linalg.to_bf16(self.b)
        elif lane == "gemm-int8":         # the codes with exponent 0: C = A B^T exactly
            self.a = (self.a, np.zeros(self.a.shape[0], dtype=np.int32))
            self.b = (self.b, np.zeros(self.b.shape[0], dtype=np.int32))
        self.identical = mode == "identical"
        self.info = _ours_info(ml, getattr(linalg, "__file__", None), mode)
        self.info.update(profile=linalg.PROFILE, call=LANE_TEXT[lane][0])
        if capability is not None:
            self.info.update(bf16_capability=capability, profile="apple.fast.bf16.fp32-accumulate")
        self.library_call = {"gemm": linalg.matmul, "gemm-bf16": linalg.matmul_bf16,
                             "gemm-int8": linalg.matmul_int8}[lane]
        self.record = _gemm_record(lane, "mojolearn", self.library_call, data)

    def call(self):
        fn = self.library_call
        if self.lane == "gemm":
            # matmul's default identical=True refuses a FAST binary (require_identical);
            # the ours-fast arm asks for the FAST route explicitly, ours keeps the default.
            self.out = fn(self.a, self.b, identical=self.identical)
        else:
            self.out = fn(self.a, self.b)


class OursBlock(Ours):
    """transformer-* and mamba*-*: one block's zero-state forward."""

    def __init__(self, lane, shape, data, ab_runtime=None):
        import numpy as np
        self.np = np
        self.runtime = dict(ab_runtime or {})
        self.operation = self.runtime.get("operation", "model_default")
        self.gradients = None
        self.window_state = None
        self.neural_ab_runtime = {"operation": self.operation, "public_calls_completed": 0}
        ml = _ours_module()
        model, cpu = MODEL_OF[lane], DEVICE_OF[lane] == "cpu"
        d = _dims_of(lane, shape)
        w = {k: np.ascontiguousarray(v) for k, v in _weights(data).items()}
        if model == "transformer":
            window = int(self.runtime.get("transformer_window", 0))
            if window < 0 or (window and self.operation != "transformer_windowed_forward"):
                raise RuntimeError("REFUSED: a positive transformer_window needs the explicit windowed operation")
            cls = ml.TransformerBlockInference if cpu else ml.TransformerBlock
            self.block = cls(w, n_heads=d["n_heads"], n_kv_heads=d["n_kv"], head_dim=d["head_dim"],
                             norm_eps=BLOCK_NORM_EPS, rope_theta=BLOCK_ROPE_THETA, window=window)
            self.neural_ab_runtime["transformer_window"] = window
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
            # FAIRNESS FIX (2026-10-13 audit): the block's structural constants, read from the
            # INSTALLED class's module (and the block), compared with what the mamba-ssm arms'
            # constructed modules hold (tools/bench_board_params.py ssm_*), so the real-kernel
            # opponent is checked to run ours' block. Results raced before this change were
            # wrong for this lane; apply extra scrutiny.
            self.record.update(ours_ssm_record(model, sys.modules[cls.__module__], self.block))
        self.decode_tokens = self.decode_session = self.decode_state = None
        if lane in DECODE_LANES:
            if self.operation != "model_default":
                raise RuntimeError("REFUSED: decode lanes run the model_default operation only")
            batch, length = int(self.x.shape[0]), int(self.x.shape[1])
            # Before the clock: the per-token host inputs (the caller's decode
            # loop feeds one (B, 1, d_model) token per call).
            self.decode_tokens = [np.ascontiguousarray(self.x[:, t:t + 1, :]) for t in range(length)]
            if model == "transformer":
                self.decode_state = self.block.allocate_state(batch, length)
            elif model == "mamba1":
                self.decode_state = self.block.allocate_state(batch)
            if self.decode_state is not None:
                # The public resident session: weights uploaded once, here,
                # like the torch twin's device weights. The state's host
                # buffers stay zero (never synced), so load_state() is the
                # per-round zero reset.
                self.decode_session = self.block.decode_session(self.decode_state)
            self.info["operation_scope"] = ("zero-state token-by-token decode of the whole input, "
                                            "one public step call per token")
        if self.operation == "transformer_forward_vjp_tape":
            if lane != "transformer-forward":
                raise RuntimeError("REFUSED: transformer tape operation requires transformer-forward")
            self.info["call"] = "TransformerBlock.forward_with_tape(x); backward_from_tape(tape, x)"
            self.info["operation_scope"] = "full declared block forward plus VJP; no loss or optimizer"
            self.neural_ab_runtime["cotangent_source"] = "same immutable full input x"
        elif self.operation == "transformer_windowed_forward":
            if lane != "transformer-forward" or window <= 0:
                raise RuntimeError("REFUSED: windowed operation needs transformer-forward and transformer_window > 0")
            self.window_state = self.block.allocate_state(int(self.x.shape[0]), int(self.x.shape[1]))
            self.info["call"] = "TransformerBlock.forward(x, state=zero_position_window_state)"
            self.info["operation_scope"] = "full input prefill plus consumed K/V ring cache; reset logical position per round"
            self.neural_ab_runtime["carried_decode_coverage"] = "not_established_by_prefill"
        elif self.operation == "mamba_forward_vjp":
            if lane not in ("mamba1-forward", "mamba2-forward", "mamba3-forward"):
                raise RuntimeError("REFUSED: Mamba VJP requires a full GPU Mamba block fixture")
            self.info.update(call="MambaBlock.forward + MambaBlock.backward",
                             operation_scope="complete zero-state prefill block forward and all input/weight VJPs; no loss or optimizer")
            self.neural_ab_runtime["cotangent_source"] = "same immutable full input x"
        elif self.operation == "mamba3_owned_forward":
            if lane != "mamba3-forward":
                raise RuntimeError("REFUSED: owned-weight operation requires mamba3-forward")
            install = bool(self.runtime.get("install_owned_weights", False))
            generation = self.block.install_owned_weights() if install else None
            self.neural_ab_runtime.update(owned_weights_requested=install,
                installed_generation=generation, installation_boundary="preparation_in_cold_total")

    def call(self):
        if self.operation == "transformer_forward_vjp_tape":
            self.out, tape = self.block.forward_with_tape(self.x,
                activation_budget_bytes=int(self.runtime.get("activation_budget_bytes", 536870912)),
                minimum_replay_ops_per_byte=int(self.runtime.get("minimum_replay_ops_per_byte", 0)))
            self.neural_ab_runtime.update(retained=bool(tape.retained),
                stage_bytes=int(tape.stage_bytes), profile=str(tape.profile))
            expected = self.runtime.get("expect_retained")
            if expected is not None and bool(tape.retained) != bool(expected):
                tape.close()
                raise RuntimeError("REFUSED: tape retained/replay selection did not reach the declared A/B arm")
            try:
                self.gradients = self.block.backward_from_tape(tape, self.x)
            finally:
                tape.close()
            self.neural_ab_runtime["vjp_completed"] = True
        elif self.operation == "mamba_forward_vjp":
            self.out = self.block.forward(self.x)
            self.gradients = self.block.backward(self.x, self.x)
            self.neural_ab_runtime["vjp_completed"] = True
        elif self.operation == "transformer_windowed_forward":
            # Position is API metadata. Native code reads no previous keys
            # at zero; each repeat processes the same entire saved input.
            self.window_state.cached_tokens = 0
            self.out = self.block.forward(self.x, state=self.window_state)
            self.neural_ab_runtime["cached_tokens"] = int(self.window_state.cached_tokens)
        elif self.decode_session is not None:
            if hasattr(self.decode_state, "cached_tokens"):
                self.decode_state.cached_tokens = 0      # position is API metadata
            self.decode_session.load_state()
            step = self.decode_session.step
            self.out = [step(xt) for xt in self.decode_tokens]
        elif self.decode_tokens is not None:
            state = self.block.allocate_state(int(self.x.shape[0]))
            step = self.block.step
            self.out = [step(xt, state) for xt in self.decode_tokens]
        else:
            self.out = self.block.forward(self.x)
        self.neural_ab_runtime["public_calls_completed"] += 1

    def _y(self):
        np = self.np
        if self.decode_tokens is not None:     # after the clock: (B, 1, D) per token -> (B, L, D)
            return np.concatenate([np.asarray(y).reshape(self.x.shape[0], 1, -1) for y in self.out],
                                  axis=1)
        return np.asarray(self.out)

    def digest(self):
        return _sha(self.np.ascontiguousarray(self._y()).data)[:16]

    def outputs(self):
        out = {"y": self._y()}
        if self.window_state is not None:
            out.update(k_cache=self.np.asarray(self.window_state.k_cache),
                       v_cache=self.np.asarray(self.window_state.v_cache))
        if self.gradients is not None:
            out.update({"grad:" + name: self.np.asarray(value)
                        for name, value in self.gradients.items()})
        return out


class OursSamba(Ours):
    def __init__(self, lane, shape, data, ab_runtime=None):
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
        runtime = ab_runtime or {}
        max_norm = runtime.get("samba_max_norm")
        accumulation_steps = int(runtime.get("samba_accumulation_steps", 1))
        if (max_norm is not None or accumulation_steps != 1) and lane != "samba-train-step":
            raise RuntimeError("REFUSED: Samba optimizer A/B settings require samba-train-step")
        self.neural_ab_runtime = dict(max_norm=max_norm, accumulation_steps=accumulation_steps,
                                     public_calls_completed=0)
        if lane == "samba-infer":
            self.model = ml.SambaInference(cfg, w)
            mode, dev = ml.numeric_mode(), "cpu"
        else:
            self.model = ml.SambaStack(cfg, weights=w, lr=ADAMW["lr"], betas=ADAMW["betas"],
                                       eps=ADAMW["eps"], weight_decay=ADAMW["weight_decay"],
                                       max_norm=max_norm, accumulation_steps=accumulation_steps)
            mode, dev = _mode_of((getattr(self.model, "_blocks", None) or [None])[0], ml), "gpu"
        self.info = _ours_info(ml, getattr(sys.modules[type(self.model).__module__], "__file__", None),
                               mode, dev)
        self.info.update(call=LANE_TEXT[lane][0], config=json.dumps(cfg.to_dict(), sort_keys=True))
        self.record = _ours_record(lane)
        # the canonical architecture read-back (ARCH_FIELDS); the raw config stays in info["config"]
        self.record["config_readback"] = ours_samba_arch(cfg.to_dict())
        if lane == "samba-train-step":
            # read back from the stack it built (None there = clipping off, recorded as SAMBA_CLIP_OFF)
            held = getattr(self.model, "max_norm", "missing")
            self.record.update(max_grad_norm=(SAMBA_CLIP_OFF if held is None else
                                              None if held == "missing" else float(held)),
                               accumulation_steps=accumulation_steps)
        self.batches = data["batches"]
        self.ids = np.ascontiguousarray(self.batches[0][:, :-1])
        self.k = 0
        self.losses = []
        # samba-decode: the per-token ids, made before the clock
        self.decode_ids = ([np.ascontiguousarray(self.ids[:, t]) for t in range(self.ids.shape[1])]
                           if lane == "samba-decode" else None)

    def call(self):
        np = self.np
        if self.lane == "samba-train-step":
            b = self.batches[self.k]
            res = self.model.train_step(np.ascontiguousarray(b[:, :-1]), np.ascontiguousarray(b[:, 1:]))
            self.losses.append(float(res["loss"]))
            self.k += 1
        elif self.decode_ids is not None:
            state = self.model.allocate_state(int(self.ids.shape[0]), int(self.ids.shape[1]))
            step = self.model.step
            self.out = [step(t, state) for t in self.decode_ids]
        else:
            self.out = self.model.forward(self.ids)
        self.neural_ab_runtime["public_calls_completed"] += 1

    def _y(self):
        if self.decode_ids is not None:        # after the clock: (B, V) per token -> (B, L, V)
            return self.np.stack([self.np.asarray(y) for y in self.out], axis=1)
        return self.np.asarray(self.out)

    def outputs(self):
        if self.lane == "samba-train-step":
            return {"losses": self.np.array(self.losses, dtype=self.np.float64)}
        return {"y": self._y()}

    def digest(self):
        return None if self.lane == "samba-train-step" else \
            _sha(self.np.ascontiguousarray(self._y()).data)[:16]


def mlp_state_evidence(state):
    """Untimed, complete public trainer snapshot; no training/inference calls."""
    import numpy as np
    optimizer = state["optimizer"]
    arrays = {"state.weight." + name: np.asarray(state["weights"][name]).copy()
              for name in MLP_NAMES}
    arrays.update({"state.optimizer." + name: np.asarray(optimizer[name]).copy()
                   for name in ("m", "v", "flags")})
    arrays["state.optimizer.step"] = np.asarray([optimizer["step"]], dtype=np.int64)
    metadata = {k: v for k, v in state.items() if k not in ("weights", "optimizer")}
    metadata["optimizer"] = {k: v for k, v in optimizer.items()
                             if k not in ("m", "v", "flags", "step")}
    raw = json.dumps(metadata, sort_keys=True, separators=(",", ":"),
                     allow_nan=False).encode("utf-8")
    arrays["state.metadata_json"] = np.frombuffer(raw, dtype=np.uint8).copy()
    return arrays


def mlp_state_digest(arrays):
    """SHA256 over named, typed, shaped state bytes (not an NPZ container)."""
    import numpy as np
    h = hashlib.sha256(b"mojolearn.board.mlp-state.v1\0")
    for name in sorted(arrays):
        array = np.ascontiguousarray(arrays[name])
        if array.dtype.hasobject or not np.isfinite(array).all():
            raise ValueError("invalid MLP state array: " + name)
        header = json.dumps([name, array.dtype.str, list(array.shape)],
                            separators=(",", ":")).encode("ascii")
        payload = array.tobytes(order="C")
        for part in (header, payload):
            h.update(len(part).to_bytes(8, "little"))
            h.update(part)
    return h.hexdigest()


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
        # refused-param-mappings (2026-10-15): ours' network, from the weights its constructor accepted (it
        # refuses any shape but its fixed (16, 8), (16,), (3, 16), (3,): python/mojolearn/_mlp_impl.py _SHAPES)
        # and its one activation, ReLU (the class has no other: _mlp_impl.py "Linear(8, 16) -> ReLU ->
        # Linear(16, 3)"); compared with the torch twin's modules (_build_mlp).
        self.record["config_readback"] = mlp_arch(int(w[0].shape[1]), int(w[0].shape[0]), int(w[2].shape[0]),
                                                  "relu", True)
        self.x0 = np.ascontiguousarray(self.X[0])
        self.k = 0
        self.losses = []
        self._state_evidence = None

    def call(self):
        np = self.np
        if self.lane == "mlp-train-step":
            res = self.model.train_step(np.ascontiguousarray(self.X[self.k]),
                                        np.ascontiguousarray(self.y[self.k]))
            self.losses.append(float(res["loss"]))
            self.k += 1
            self._state_evidence = None
        else:
            self.out = self.model.predict_logits(self.x0)

    def outputs(self):
        if self.lane == "mlp-train-step":
            return {"losses": self.np.array(self.losses, dtype=self.np.float64),
                    **self._snapshot()}
        return {"y": self.np.asarray(self.out)}

    def _snapshot(self):
        if self._state_evidence is None:
            self._state_evidence = mlp_state_evidence(self.model.state_dict())
        return self._state_evidence

    def digest(self):
        # Worker invokes this only AFTER stopping the round timer and memory probe.
        return (mlp_state_digest(self._snapshot()) if self.lane == "mlp-train-step"
                else Ours.digest(self))


class OursMLPSessions(Ours):
    """Full saved MLP inputs through the explicit native session operation."""
    def __init__(self, lane, shape, data, runtime):
        import numpy as np
        from mojolearn import training
        if lane != "mlp-forward":
            raise RuntimeError("REFUSED: MLP sessions require the GPU mlp-forward lane")
        self.np, self.training = np, training
        self.weights = [np.ascontiguousarray(data["w:" + name]) for name in MLP_NAMES]
        self.x = np.ascontiguousarray(data["X"][0])
        count = int(runtime.get("session_count", 4))
        if count < 1 or count > len(self.x):
            raise ValueError("session_count must be in [1, full input rows]")
        # Metadata-only partition, all original rows exactly once in order.
        # Native Mojo owns packing, projections, activation and scattering.
        edges = [j * len(self.x) // count for j in range(count + 1)]
        self.sessions = [self.x[edges[j]:edges[j + 1]] for j in range(count)]
        ml = _ours_module()
        self.info = _ours_info(ml, training.__file__, ml.numeric_mode(), "gpu")
        self.info.update(call="mojolearn.training.mlp_inference_sessions", operation_scope="complete stateless MLP forward over every saved input row")
        self.record = _ours_record(lane)
        self.neural_ab_runtime = dict(operation="mlp_inference_sessions", session_rows=[len(x) for x in self.sessions],
                                     public_calls_completed=0, input_rows=len(self.x))

    def call(self):
        self.out = self.training.mlp_inference_sessions(self.sessions, *self.weights, numeric_mode="identical")
        self.neural_ab_runtime["public_calls_completed"] += 1


class OursResidualDropout(Ours):
    """An explicit layer plus VJP; never labeled a full transformer step."""
    def __init__(self, lane, shape, data, runtime):
        import numpy as np
        from mojolearn import training
        if lane != "transformer-forward":
            raise RuntimeError("REFUSED: residual/dropout uses the complete transformer-forward input fixture")
        self.np, self.training = np, training
        self.x = np.ascontiguousarray(data["x"])
        self.residual = self.x.copy()
        self.dy = self.x.copy()
        self.settings = dict(p=float(runtime.get("dropout_probability", 0.1)),
                             seed=int(runtime.get("seed", SEED)), stream=int(runtime.get("stream", 0)),
                             offset=int(runtime.get("offset", 0)))
        ml = _ours_module()
        self.info = _ours_info(ml, training.__file__, ml.numeric_mode(), "gpu")
        self.info.update(call="residual_dropout + residual_dropout_backward",
                         operation_scope="complete explicit residual/dropout layer and VJP; not a transformer block or model")
        self.record = {"__library__": "mojolearn", **self.settings}
        self.neural_ab_runtime = dict(operation="residual_dropout_forward_vjp", public_calls_completed=0,
                                     full_input_shape=list(self.x.shape), cotangent_source="immutable copy of full input")

    def call(self):
        self.out = self.training.residual_dropout(self.x, self.residual, **self.settings)
        self.dx, self.dr = self.training.residual_dropout_backward(self.dy, **self.settings)
        self.neural_ab_runtime["public_calls_completed"] += 1

    def outputs(self):
        return {"y": self.np.asarray(self.out), "d_values": self.np.asarray(self.dx), "d_residual": self.np.asarray(self.dr)}


# ---------------------------------------------------------------------------
# The fit lanes (2026-10-19, lane ag-bench): one round = one whole fit
# ---------------------------------------------------------------------------

#: tape-user-model's loss, by its name in the model API (lane ag-pyapi)
TAPE_LOSS = "bce_with_logits"
#: rows per prediction chunk on the opponents' held-out pass (after the clock)
FIT_PREDICT_CHUNK = 65536


def _ours_sequential(ml, lane):
    """A Sequential fit lane's network through the public mojolearn.Sequential: linear(h), relu, ...,
    linear(1) (Sequential.linear takes the output width; the input width comes from X at fit), compiled with
    BCE with logits and Adam, every work setting passed explicitly. Refuses by name when this mojolearn has
    no Sequential. Returns (model, its read-back: the layers it recorded and the options it accepted).

    ag-board-twins (2026-10-19): the earlier helper (lane ag-bench, written while the class was absent)
    called add("linear", fan_in, fan_out), which Sequential.add reads as units=fan_in, bias=fan_out, and
    compile(betas=...), which Sequential.compile does not take (beta1, beta2): tape-user-model refused on
    every box. This one speaks the class's own signatures (python/mojolearn/_x_sequence_model.py)."""
    Seq = getattr(ml, "Sequential", None)
    if Seq is None:
        raise ImportError("REFUSED: %s races mojolearn.Sequential, the model-description API over the autograd "
                          "tape; this mojolearn (%s, %s) has no Sequential"
                          % (lane, getattr(ml, "__version__", "?"), getattr(ml, "__file__", "?")))
    model = Seq()
    for h in FIT_HIDDEN[lane]:
        model.linear(int(h))
        model.relu()
    model.linear(1)
    accepted = dict(loss=TAPE_LOSS, optimizer="adam", lr=FIT_ADAM["lr"], beta1=FIT_ADAM["betas"][0],
                    beta2=FIT_ADAM["betas"][1], eps=FIT_ADAM["eps"], weight_decay=FIT_ADAM["weight_decay"],
                    schedule="constant", tol=0.0, n_iter_no_change=0)
    try:
        model.compile(**accepted)
    except TypeError as exc:
        raise RuntimeError("REFUSED: %s: mojolearn.Sequential.compile does not take the lane's explicit Adam "
                           "settings %s (%r); every work setting is passed on every arm" % (lane, sorted(accepted), exc))
    return model, accepted


def _sequential_readback(model, accepted, d_in):
    """fit_params read-back of a compiled Sequential: the layers from its public `layers` list (widths chained
    from the data's d_in), the optimizer settings from what compile recorded (its option table when it keeps
    one, else the values it accepted)."""
    layers, width = [], int(d_in)
    for kind, a0, a1, _f in model.layers:
        if kind == "linear":
            layers.append("linear %d %d %s" % (width, int(a0), "bias" if a1 else "nobias"))
            width = int(a0)
        else:
            layers.append(kind)
    opts = getattr(model, "_options", None) or {}
    adam = opts.get("opt") == 1 if "opt" in opts else accepted["optimizer"] == "adam"
    return dict(optimizer="adam" if adam else "other (code %r)" % (opts.get("opt"),),
                learning_rate=opts.get("lr", accepted["lr"]),
                betas=[opts.get("f1", accepted["beta1"]), opts.get("f2", accepted["beta2"])],
                eps=opts.get("eps", accepted["eps"]), weight_decay=opts.get("weight_decay", accepted["weight_decay"]),
                loss=opts.get("loss", accepted["loss"]), layers=layers)


class OursFit(Ours):
    """The fit lanes through the public estimators: MLPClassifier (the tape path is the MLP's default),
    LSTM/GRU/RNNRegressor (the tape path, their only backward since 2026-10-19) and
    mojolearn.Sequential. call() is one whole fit from random_state=SEED; outputs() predicts the
    held-out rows after the clock."""

    def __init__(self, lane, shape, data):
        import numpy as np
        self.np, self.lane = np, lane
        ml = _ours_module()
        self.ml = ml
        self.Xtr = np.ascontiguousarray(data["X_train"], dtype=np.float32)
        self.Xte = np.ascontiguousarray(data["X_test"], dtype=np.float32)
        ytr = np.asarray(data["y_train"])
        # the classifier takes class labels, the regressors the float target (prepared before the clock)
        self.ytr = (np.ascontiguousarray((ytr > 0.5).astype(np.int64)) if FIT_TASK[lane] == "classification"
                    and MODEL_OF[lane] != "tapefit" else np.ascontiguousarray(ytr, dtype=np.float32))
        d_in = int(self.Xtr.shape[-1])
        self.model = None
        self._score = None
        if MODEL_OF[lane] == "tapefit":
            probe, accepted = _ours_sequential(ml, lane)
            cls = type(probe)
            readback = _sequential_readback(probe, accepted, d_in)
        elif lane == "mlp-fit":
            cls = ml.MLPClassifier
            self.kwargs = dict(hidden_layer_sizes=FIT_MLP_HIDDEN, activation="relu", solver="adam", alpha=0.0,
                               batch_size=FIT_BATCH, learning_rate="constant", learning_rate_init=FIT_ADAM["lr"],
                               max_iter=FIT_EPOCHS, shuffle=True, random_state=SEED, tol=0.0,
                               n_iter_no_change=1000, beta_1=FIT_ADAM["betas"][0], beta_2=FIT_ADAM["betas"][1],
                               epsilon=FIT_ADAM["eps"])
            probe = cls(**self.kwargs)
            g = probe.get_params()
            readback = dict(optimizer=g["solver"], learning_rate=g["learning_rate_init"],
                            betas=[g["beta_1"], g["beta_2"]], eps=g["epsilon"], batch_size=g["batch_size"],
                            epochs=g["max_iter"], shuffle=g["shuffle"], seed=g["random_state"], alpha=g["alpha"],
                            layers=fit_layers(lane, d_in, list(g["hidden_layer_sizes"])))
            if g["activation"] != "relu":
                readback["layers"] = [l.replace("relu", g["activation"]) for l in readback["layers"]]
        else:
            cls = {"lstm-fit": ml.LSTMRegressor, "gru-fit": ml.GRURegressor, "rnn-fit": ml.RNNRegressor}[lane]
            self.kwargs = dict(hidden_size=FIT_RNN_HIDDEN, num_layers=1, optimizer="adam",
                               learning_rate=FIT_ADAM["lr"], batch_size=FIT_BATCH, max_epochs=FIT_EPOCHS,
                               shuffle=True, random_state=SEED,
                               optimizer_options=dict(betas=tuple(FIT_ADAM["betas"]), eps=FIT_ADAM["eps"],
                                                      weight_decay=FIT_ADAM["weight_decay"]))
            if lane == "rnn-fit":
                self.kwargs["nonlinearity"] = "tanh"
            probe = cls(**self.kwargs)
            g = probe.get_params()
            opts = g.get("optimizer_options") or {}
            readback = dict(optimizer=g["optimizer"], learning_rate=g["learning_rate"],
                            betas=list(opts.get("betas", (None, None))), eps=opts.get("eps"),
                            weight_decay=opts.get("weight_decay"), batch_size=g["batch_size"],
                            epochs=g["max_epochs"], shuffle=g["shuffle"], seed=g["random_state"],
                            num_layers=g["num_layers"], timesteps=int(self.Xtr.shape[1]),
                            layers=fit_layers(lane, d_in, [g["hidden_size"]]))
            if lane == "rnn-fit":
                readback["nonlinearity"] = g["nonlinearity"]
        self.cls = cls
        self.info = _ours_info(ml, getattr(sys.modules[cls.__module__], "__file__", None), ml.numeric_mode(), "gpu")
        self.info.update(call=LANE_TEXT[lane][0], fit_rows=int(self.Xtr.shape[0]), held_out_rows=int(self.Xte.shape[0]),
                         training_path=("the autograd tape (MLPClassifier's default since 9542fadae)" if lane == "mlp-fit"
                                        else "the autograd tape (mojolearn.Sequential)" if MODEL_OF[lane] == "tapefit"
                                        else "the autograd tape (the recurrent trainer's only backward since 2026-10-19)"))
        self.record = fit_record(lane, "mojolearn", cls, d_in, **readback)

    def call(self):
        import warnings
        if MODEL_OF[self.lane] == "tapefit":
            m, _ = _ours_sequential(self.ml, self.lane)
            m.fit(self.Xtr, self.ytr, epochs=FIT_EPOCHS, batch_size=FIT_BATCH, shuffle=True, seed=SEED)
        else:
            m = self.cls(**self.kwargs)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")      # one epoch: scikit-learn's ConvergenceWarning
                m.fit(self.Xtr, self.ytr)
        self.model = m
        self._score = None
        if self.lane in FIT_PREDICT_IN_CLOCK:
            self._predict()               # sequential-fit: predict_proba of the held-out rows inside the clock

    def _predict(self):
        np = self.np
        if self._score is None:
            m = self.model
            if self.lane == "mlp-fit":
                p = np.asarray(m.predict_proba(self.Xte))[:, 1]
            elif MODEL_OF[self.lane] == "tapefit":
                p = np.asarray(m.predict_proba(self.Xte))
                if p.ndim == 2 and p.shape[1] == 2:
                    p = p[:, 1]
            else:
                p = np.asarray(m.predict(self.Xte))
            self._score = np.ascontiguousarray(np.asarray(p, dtype=np.float32).ravel())
        return self._score

    def _loss_curve(self):
        curve = getattr(self.model, "loss_curve_", None)     # a list (MLP) or an array (recurrent)
        return self.np.asarray([] if curve is None else curve, dtype=self.np.float64).ravel()

    def outputs(self):
        return {"score": self._predict(), "loss_curve": self._loss_curve()}

    def digest(self):
        np = self.np
        parts = [self._predict()]
        m = self.model
        if self.lane == "mlp-fit":
            parts += [np.asarray(a) for a in m.coefs_] + [np.asarray(a) for a in m.intercepts_]
        elif self.lane in FIT_CELL or MODEL_OF[self.lane] == "tapefit":
            parts.append(np.asarray(m.params_))
        h = hashlib.sha256()
        for a in parts:
            h.update(np.ascontiguousarray(a).tobytes())
        return h.hexdigest()[:16]


# ---------------------------------------------------------------------------
# ops-session-step (2026-10-19, lane ag-board-twins): one recorded step through mojolearn.ops.Session
# ---------------------------------------------------------------------------

def _ops_outputs(np, loss, grads, updated):
    """One ops-session-step round's results, the same keys on every arm: the loss, every gradient (g:<name>)
    and every updated parameter (p:<name>), float32 host arrays, OPS_PARAMS order."""
    out = {"loss": np.asarray([loss], dtype=np.float64)}
    for name, g, p in zip(OPS_PARAMS, grads, updated):
        out["g:" + name] = np.ascontiguousarray(g, dtype=np.float32)
        out["p:" + name] = np.ascontiguousarray(p, dtype=np.float32)
    return out


def _ops_digest(np, grads):
    """The round digest of ops-session-step: sha256 of every gradient's float32 bytes, OPS_PARAMS order."""
    h = hashlib.sha256()
    for g in grads:
        h.update(np.ascontiguousarray(g, dtype=np.float32).tobytes())
    return h.hexdigest()[:16]


class OursOpsStep(Ours):
    """ops-session-step through the public operation session, mojolearn.ops.Session(record=True): the batch,
    the codes and the four parameters (requires_grad=True) uploaded, relu(linear(x, w1, b1)), linear(h, w2,
    b2), cross_entropy, backward, the loss and every gradient read back, then the SGD update p - lr * grad in
    place on the device by the session's optimizer (Session.step: one update launch per parameter from the
    backward's own gradients, no re-upload) and every updated parameter read back. One session and one
    optimizer per worker: each round frees the previous round's uploads (their optimizer state goes with them)
    and resets the session first (inside the clock), so every round is the same step from the same fixture
    weights."""

    def __init__(self, lane, shape, data):
        import importlib
        import numpy as np
        self.np, self.lane, self.shape = np, lane, shape
        ml = _ours_module()
        self.ml = ml
        try:
            ops = importlib.import_module("mojolearn.ops")
        except ImportError as exc:
            raise ImportError("REFUSED: ops-session-step races mojolearn.ops.Session; this mojolearn (%s, %s) "
                              "has no ops module (%r)" % (getattr(ml, "__version__", "?"),
                                                          getattr(ml, "__file__", "?"), exc))
        w = _weights(data)
        self.p0 = [np.ascontiguousarray(w[n], dtype=np.float32) for n in OPS_PARAMS]
        self.X = np.ascontiguousarray(data["X"], dtype=np.float32)
        self.codes = np.ascontiguousarray(data["y_codes"], dtype=np.float32)
        self.lr = float(OPS_SGD["lr"])
        self.s = ops.Session(record=True)
        if not self.s.record:
            raise RuntimeError("REFUSED: ops-session-step: Session(record=True) reads back record=%r"
                               % (self.s.record,))
        if not hasattr(self.s, "step"):
            raise ImportError("REFUSED: ops-session-step updates with Session.step; this mojolearn (%s) has "
                              "no session optimizer" % (getattr(ml, "__version__", "?"),))
        # torch.optim.SGD(lr, momentum=0, weight_decay=0): the same rule (OPS_SGD)
        self.opt = self.s.optimizer("sgd", lr=self.lr, momentum=float(OPS_SGD["momentum"]),
                                    weight_decay=float(OPS_SGD["weight_decay"]))
        self._uploads = []
        self.loss, self.grads, self.updated = None, None, None
        d_in, hidden = (int(v) for v in self.p0[0].shape)
        classes = int(self.p0[2].shape[1])
        self.info = _ours_info(ml, getattr(sys.modules[ops.Session.__module__], "__file__", None),
                               ml.numeric_mode(), "gpu")
        self.info.update(call=LANE_TEXT[lane][0], session="one Session(record=True) per worker; free + reset "
                         "at the start of every round, inside the clock",
                         update="SGD in place on the device: Session.step(optimizer('sgd', lr), params)")
        self.record = ops_record("mojolearn", ops.Session, shape, batch_size=int(self.X.shape[0]),
                                 layers=ops_layers(d_in, hidden, classes))

    def call(self):
        s = self.s
        for h in self._uploads:          # driver-loop: over the previous round's uploaded tensors (ten handles)
            s.free(h)
        s.reset()
        x = s.tensor(self.X)
        codes = s.tensor(self.codes)
        ps = [s.tensor(p, requires_grad=True) for p in self.p0]   # driver-loop: over the four parameters
        uploads = [x, codes] + ps
        hidden = s.relu(s.linear(x, ps[0], ps[1]))
        logits = s.linear(hidden, ps[2], ps[3])
        loss = s.cross_entropy(logits, codes)
        s.backward(loss)
        self.loss = s.scalar(loss)
        self.grads = [s.grad(p) for p in ps]                       # driver-loop: over the four parameters
        s.step(self.opt, ps)                                       # in place on the device, no re-upload
        self.updated = [s.numpy(p) for p in ps]                    # driver-loop: over the four parameters
        self._uploads = uploads

    def outputs(self):
        return _ops_outputs(self.np, self.loss, self.grads, self.updated)

    def digest(self):
        return _ops_digest(self.np, self.grads)


# ---------------------------------------------------------------------------
# program-run (2026-10-19, lane ag-program): one inference forward through a captured mojolearn.ops.Program
# ---------------------------------------------------------------------------

class OursProgramRun(Ours):
    """program-run through mojolearn.ops: one Session() per worker, the four weights uploaded once, the forward
    softmax(linear(relu(linear(x, w1, b1)), w2, b2)) composed once and captured (Session.capture, inputs [x],
    outputs [p]) before the clock. A round is Program.run(X): the batch to the device, every captured operation,
    the probabilities back on the host, one call. Before the clock the worker checks that the program's output
    words equal the eager composition's (REFUSED otherwise). With the A/B operation `program_eager` a round is
    the eager composition instead (one call per operation, the same weights), reset first."""

    def __init__(self, lane, shape, data, eager=False):
        import importlib
        import numpy as np
        self.np, self.lane, self.shape, self.eager = np, lane, shape, bool(eager)
        ml = _ours_module()
        self.ml = ml
        try:
            ops = importlib.import_module("mojolearn.ops")
        except ImportError as exc:
            raise ImportError("REFUSED: program-run races mojolearn.ops.Program; this mojolearn (%s, %s) has no "
                              "ops module (%r)" % (getattr(ml, "__version__", "?"), getattr(ml, "__file__", "?"), exc))
        if not hasattr(ops.Session, "capture"):
            raise ImportError("REFUSED: program-run races Session.capture; this mojolearn (%s) has none"
                              % (getattr(ml, "__version__", "?"),))
        w = _weights(data)
        self.p0 = [np.ascontiguousarray(w[n], dtype=np.float32) for n in OPS_PARAMS]
        self.X = np.ascontiguousarray(data["X"], dtype=np.float32)
        self.s = ops.Session()
        self.ws = [self.s.tensor(p) for p in self.p0]               # driver-loop: over the four parameters
        x = self.s.tensor(self.X)
        prob = self._forward(x)
        eager_words = self.s.numpy(prob)
        self.program = self.s.capture(inputs=[x], outputs=[prob])
        self.s.reset()
        first = self.program.run(self.X)
        if not np.array_equal(first.view(np.uint32), eager_words.view(np.uint32)):
            raise RuntimeError("REFUSED: program-run: Program.run's output words differ from the eager session's")
        self._x = x
        self.out = None
        d_in, hidden = (int(v) for v in self.p0[0].shape)
        classes = int(self.p0[2].shape[1])
        self.info = _ours_info(ml, getattr(sys.modules[ops.Session.__module__], "__file__", None),
                               ml.numeric_mode(), "gpu")
        self.info.update(call=LANE_TEXT[lane][0], words_equal_eager=True,
                         timed=("the eager session, one call per operation (A/B operation program_eager)"
                                if self.eager else "Program.run, one call"),
                         session="one Session() per worker; weights uploaded and the forward captured before "
                                 "the clock")
        self.record = program_record("mojolearn", ops.Program, shape, batch_size=int(self.X.shape[0]),
                                     layers=program_layers(d_in, hidden, classes))

    def _forward(self, x):
        s, (w1, b1, w2, b2) = self.s, self.ws
        return s.softmax(s.linear(s.linear(x, w1, b1, activation="relu"), w2, b2))

    def call(self):
        if self.eager:
            s = self.s
            s.free(self._x)
            s.reset()
            self._x = s.tensor(self.X)
            self.out = s.numpy(self._forward(self._x))
        else:
            self.out = self.program.run(self.X)
# The inference lanes (2026-10-19, lane ag-inference-board): device memory, CausalLM generate, Sequential predict
# ---------------------------------------------------------------------------

class DeviceMemory:
    """The inference lanes' device-memory counters, read in this process with no subprocess (a fork between
    rounds made the next timed AMD fit pay ~100 ms, bench_board_probe.MemProbe). AMD: the KFD's per-process VRAM
    bytes (/sys/class/kfd/kfd/proc/<pid>/vram_*), and the device's used bytes (sum of /sys/class/drm/card*/device/
    mem_info_vram_used). NVIDIA: NVML through ctypes, the per-process used bytes when NVML lists this pid (inside
    a container's pid namespace it does not) and the device's used bytes (nvmlDeviceGetMemoryInfo) of the first
    visible GPU. Apple: no device counter (unified memory). read() -> {"process": bytes or None, "device": bytes
    or None}."""

    def __init__(self):
        self.vendor = (os.environ.get("MOJOLEARN_BOARD_VENDOR") or
                       ("apple" if sys.platform == "darwin" else
                        "amd" if os.path.isdir("/sys/class/kfd/kfd/proc") else "nvidia"))
        self._nvml, self._handle, self.error = None, None, None
        if self.vendor == "nvidia":
            try:
                self._open_nvml()
            except Exception as exc:  # noqa: BLE001  (memory never breaks a timing)
                self.error = "NVML unavailable: %r" % (exc,)

    def _open_nvml(self):
        import ctypes
        lib = ctypes.CDLL("libnvidia-ml.so.1")
        if lib.nvmlInit_v2() != 0:
            raise RuntimeError("nvmlInit_v2 failed")
        handle = ctypes.c_void_p()
        first = (os.environ.get("CUDA_VISIBLE_DEVICES") or "0").split(",")[0].strip()
        if first.isdigit():
            rc = lib.nvmlDeviceGetHandleByIndex_v2(ctypes.c_uint(int(first)), ctypes.byref(handle))
        else:
            rc = lib.nvmlDeviceGetHandleByUUID(ctypes.c_char_p(first.encode()), ctypes.byref(handle))
        if rc != 0:
            raise RuntimeError("NVML has no device %r (rc %d)" % (first, rc))
        self._nvml, self._handle = lib, handle

    def method(self):
        if self.vendor == "amd":
            return ("KFD /sys/class/kfd/kfd/proc/<pid>/vram_* (this process) and /sys/class/drm/card*/device/"
                    "mem_info_vram_used (the device)")
        if self.vendor == "nvidia":
            return self.error or ("NVML nvmlDeviceGetComputeRunningProcesses usedGpuMemory for this pid (None "
                                  "inside a container's pid namespace) and nvmlDeviceGetMemoryInfo used (the "
                                  "device), via ctypes")
        return "no device counter on Apple (unified memory); see the round's peak_host_mb"

    def _read_nvidia(self):
        import ctypes
        out = {"process": None, "device": None}
        if self._nvml is None:
            return out

        class _Mem(ctypes.Structure):
            _fields_ = [("total", ctypes.c_ulonglong), ("free", ctypes.c_ulonglong), ("used", ctypes.c_ulonglong)]

        class _Proc(ctypes.Structure):
            _fields_ = [("pid", ctypes.c_uint), ("usedGpuMemory", ctypes.c_ulonglong),
                        ("gpuInstanceId", ctypes.c_uint), ("computeInstanceId", ctypes.c_uint)]
        m = _Mem()
        if self._nvml.nvmlDeviceGetMemoryInfo(self._handle, ctypes.byref(m)) == 0:
            out["device"] = int(m.used)
        fn = getattr(self._nvml, "nvmlDeviceGetComputeRunningProcesses_v3", None) or \
            getattr(self._nvml, "nvmlDeviceGetComputeRunningProcesses_v2", None)
        if fn is not None:
            count = ctypes.c_uint(64)
            procs = (_Proc * 64)()
            if fn(self._handle, ctypes.byref(count), procs) == 0:
                pid = os.getpid()
                for i in range(min(int(count.value), 64)):     # driver-loop: over the GPU's compute processes
                    if procs[i].pid == pid and procs[i].usedGpuMemory < (1 << 63):
                        out["process"] = int(procs[i].usedGpuMemory)
        return out

    def _read_amd(self):
        out = {"process": None, "device": None}
        kfd = "/sys/class/kfd/kfd/proc/%d" % os.getpid()
        try:
            names = [n for n in os.listdir(kfd) if n.startswith("vram_")]
        except OSError:
            names = []
        if names:
            used = 0
            for n in names:                                    # driver-loop: over this process's GPUs
                with open(os.path.join(kfd, n)) as fh:
                    used += int(fh.read().strip() or 0)
            out["process"] = used
        dev = 0
        found = False
        for path in glob.glob("/sys/class/drm/card*/device/mem_info_vram_used"):   # driver-loop: over the GPUs
            try:
                with open(path) as fh:
                    dev += int(fh.read().strip() or 0)
                found = True
            except (OSError, ValueError):
                pass
        out["device"] = dev if found else None
        return out

    def read(self):
        try:
            if self.vendor == "nvidia":
                return self._read_nvidia()
            if self.vendor == "amd":
                return self._read_amd()
        except Exception:  # noqa: BLE001
            pass
        return {"process": None, "device": None}


def _mem_delta(before, after):
    return {k: (after[k] - before[k] if isinstance(after.get(k), int) and isinstance(before.get(k), int) else None)
            for k in ("process", "device")}


def memory_pass(dm, resident, fn, torch=None):
    """One UNTIMED call of `fn` (the round's call again, its results discarded) with a 1 ms sampler thread on
    `dm`: {resident_bytes (the load's delta, per counter), before, peak, peak_bytes (resident + the pass's
    transient, per counter), process_peak_bytes (the absolute per-process peak, context included, where the
    vendor exposes one), torch allocator figures for a torch arm, method}."""
    import threading
    stop = threading.Event()
    before = dm.read()
    peak = dict(before)

    def sample():
        while not stop.is_set():
            r = dm.read()
            for k, v in r.items():
                if isinstance(v, int) and (peak.get(k) is None or v > peak[k]):
                    peak[k] = v
            stop.wait(0.001)
    alloc0 = None
    if torch is not None:
        try:
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            alloc0 = int(torch.cuda.memory_allocated())
        except Exception:  # noqa: BLE001  (MPS: no allocator peak)
            torch = None
    t = threading.Thread(target=sample, daemon=True)
    t.start()
    try:
        fn()
    finally:
        stop.set()
        t.join()
    last = dm.read()
    for k, v in last.items():
        if isinstance(v, int) and (peak.get(k) is None or v > peak[k]):
            peak[k] = v
    out = {"method": dm.method(), "resident_bytes": dict(resident or {}), "before": before, "peak": peak,
           "peak_bytes": {}, "process_peak_bytes": peak.get("process")}
    for k in ("process", "device"):
        r, b, p = (resident or {}).get(k), before.get(k), peak.get(k)
        out["peak_bytes"][k] = (r + (p - b)) if isinstance(r, int) and isinstance(b, int) and isinstance(p, int) \
            else None
    if torch is not None:
        try:
            out["torch_allocated_before_bytes"] = alloc0
            out["torch_max_allocated_bytes"] = int(torch.cuda.max_memory_allocated())
            out["torch_reserved_bytes"] = int(torch.cuda.memory_reserved())
        except Exception:  # noqa: BLE001
            pass
    return out


def _gen_inputs(np, data):
    ids = np.ascontiguousarray(data["prompt_ids"], dtype=np.int32)
    return ids, int(np.asarray(data["new_tokens"]).ravel()[0]), str(np.asarray(data["checkpoint_dir"])), \
        str(np.asarray(data["checkpoint_key"]))


def _gen_timing(t0, t1, t2, n_new):
    prefill, gen = (t1 - t0) * 1000.0, (t2 - t1) * 1000.0
    return {"prefill_ms": prefill, "generate_ms": gen, "new_tokens": n_new,
            "decode_ms_per_token": ((gen - prefill) / (n_new - 1)) if n_new > 1 else None,
            "clock_ms": gen}


class OursGenerate(Ours):
    """causal-lm-generate through mojolearn.models.CausalLM: the checkpoint loaded once in the untimed warm-up
    round (its device bytes read around the load), then every round generate(ids, 1) (prefill_ms) and
    generate(ids, N) (the clock), both from the host ids, greedy. The architecture is read back from the
    loader's own plan of the checkpoint's config.json at ready (no device work before the warm-up)."""

    def __init__(self, lane, shape, data):
        import numpy as np
        self.np, self.lane, self.shape = np, lane, shape
        self.dm = DeviceMemory()
        ml = _ours_module()
        self.ml = ml
        try:
            from mojolearn.models import CausalLM, HFConfig, plan_for
        except ImportError as exc:
            raise ImportError("REFUSED: %s races mojolearn.models.CausalLM; this mojolearn (%s) has no models "
                              "module (%r)" % (lane, getattr(ml, "__version__", "?"), exc))
        self.CausalLM = CausalLM
        self.ids, self.n_new, self.ckpt_dir, self.key = _gen_inputs(np, data)
        plan = plan_for(HFConfig.from_json(self.ckpt_dir))     # refuses an unsupported checkpoint by name
        self.lm, self.resident, self.first, self.out, self.timing = None, None, None, None, None
        self.info = _ours_info(ml, getattr(sys.modules[CausalLM.__module__], "__file__", None), ml.numeric_mode(),
                               "gpu")
        self.info.update(call=LANE_TEXT[lane][0], checkpoint_dir=self.ckpt_dir, model_type=plan.model_type,
                         device_memory=self.dm.method(),
                         load="CausalLM.load in the untimed warm-up round (round 0)")
        self.record = gen_record(lane, "mojolearn", CausalLM, shape, batch_size=int(self.ids.shape[0]),
                                 prompt_tokens=int(self.ids.shape[1]), new_tokens=self.n_new,
                                 n_layers=int(plan.n_layers), d_model=int(plan.d_model), n_heads=int(plan.n_heads),
                                 n_kv_heads=int(plan.n_kv_heads), vocab_size=int(plan.vocab_size),
                                 final_norm=plan.norm_name is not None, tied_head=plan.head_name is None,
                                 rope_theta=float(plan.block_options["rope_theta"]),
                                 norm_eps=float(plan.block_options["norm_eps"]))

    def _build(self):
        before = self.dm.read()
        self.lm = self.CausalLM.load(self.ckpt_dir, weight_format="float32", numeric_profile="fp32_v1", device="gpu")
        self.resident = _mem_delta(before, self.dm.read())
        self.info.update(model=repr(self.lm), numeric_profile_used=self.lm.numeric_profile,
                         weight_format_used=self.lm.weight_format)

    def _gen(self, n):
        return self.np.asarray(self.lm.generate(self.ids, n), dtype=self.np.int32)

    def call(self):
        if self.lm is None:
            self._build()                   # the warm-up round: load + upload, untimed (its ms is warmup_ms)
        t0 = time.perf_counter()
        first = self._gen(1)
        t1 = time.perf_counter()
        out = self._gen(self.n_new)
        t2 = time.perf_counter()
        self.first, self.out = first, out
        self.timing = _gen_timing(t0, t1, t2, self.n_new)

    def round_extra(self):
        return dict(self.timing or {})

    def memory_pass(self):
        return memory_pass(self.dm, self.resident, lambda: self._gen(self.n_new))

    def outputs(self):
        return {"ids": self.out, "first": self.first}

    def digest(self):
        return _sha(self.np.ascontiguousarray(self.out).tobytes())[:16]


class OursPredict(OursFit):
    """sequential-predict: OursFit's Sequential (the sequential-fit recipe) fitted once in the untimed warm-up
    round (its device bytes read around the fit), then every round predict_proba(X_test) alone in the clock."""

    def __init__(self, lane, shape, data):
        self.dm = DeviceMemory()
        OursFit.__init__(self, lane, shape, data)
        self._fitted, self.resident = False, None
        self.info.update(call=LANE_TEXT[lane][0], device_memory=self.dm.method(),
                         fit="one fit in the untimed warm-up round (round 0); the clock is predict_proba alone")

    def call(self):
        if not self._fitted:
            before = self.dm.read()
            OursFit.call(self)              # the fit only: this lane is not in FIT_PREDICT_IN_CLOCK
            self.resident = _mem_delta(before, self.dm.read())
            self._fitted = True
        self._score = None
        self._predict()

    def memory_pass(self):
        return memory_pass(self.dm, self.resident, lambda: self.model.predict_proba(self.Xte))


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
        self.arm = arm
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
            # preflight-smoke (2026-10-14): every math flag pinned to ours' value where this torch takes it;
            # capturable/differentiable explicitly off (implementation-only, recorded as exceptions)
            import inspect
            try:
                sig = inspect.signature(torch.optim.AdamW).parameters
            except (TypeError, ValueError):
                sig = {}
            pinned = {k: v for k, v in OURS_ADAMW_SEMANTICS.items() if k in sig}
            pinned.update({k: False for k in ("capturable", "differentiable") if k in sig})
            self.opt = torch.optim.AdamW(self.module.parameters(), lr=ADAMW["lr"], betas=ADAMW["betas"],
                                         eps=ADAMW["eps"], weight_decay=ADAMW["weight_decay"], **pinned)
            self.info["optimizer"] = "torch.optim.AdamW lr %g betas %s eps %g wd %g (torch's default impl)" % (
                ADAMW["lr"], ADAMW["betas"], ADAMW["eps"], ADAMW["weight_decay"])
            # read back from the optimizer itself
            back = _load("bench_board_params").arm_record(self.opt)["params"]
            rec.update(back)
            if "betas" in rec:
                rec["betas"] = list(rec["betas"])
        else:
            rec.update(self.block_consts)
        # preflight-smoke (2026-10-14): the library class this arm actually built (transformers' Llama, mamba_ssm's
        # Block) and its architecture read-back, so the parameter check compares them with ours'
        if getattr(self, "library_impl", None) is not None:
            rec["__implementation__"] = self.library_impl
        if getattr(self, "config_readback", None) is not None:
            rec["config_readback"] = self.config_readback
        if MODEL_OF[lane] == "gemm":
            rec = _gemm_record(lane, "torch", self.library_call, data)
        if MODEL_OF[lane] == "samba":
            # this arm's call() never clips (no clip_grad_norm_): clipping off, the same explicit value as ours
            rec.update(dropout=SAMBA_DROPOUT, max_grad_norm=SAMBA_CLIP_OFF if self.train else "n/a")
            if lane == "samba-train-step":
                # smoke-fixes (2026-10-14; samba-train-step refused with "accumulation_steps missing from
                # torch-* read-back"): this arm's call() runs zero_grad, ONE backward and opt.step per
                # call, so it accumulates over exactly 1 micro-batch by construction (see call() below).
                # Ours reads back its own value; an A/B that sets ours to n != 1 shows as a mismatch.
                rec.update(accumulation_steps=1)
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
            self.library_call = torch._int_mm
            self.module = self._module({}, lambda p, a, b: self.library_call(a, b.t()))
            return
        self.library_call = torch.matmul
        self.module = self._module({}, lambda p, a, b: self.library_call(a, b))

    def _build_lm(self):
        # FAIRNESS FIX (2026-10-13 audit): this arm ran tools/torch_lm_step_opponent.py's
        # ByteDecoder, a decoder this harness wrote from torch ops (hand-written RMSNorm, RoPE and
        # SwiGLU around SDPA): code our harness wrote is not an opponent. It now runs transformers'
        # own LlamaForCausalLM configured to ours' exact architecture (hf_llama_config: untied
        # head, SwiGLU, no biases, RMSNorm eps and RoPE base read back), its final norm replaced by
        # Identity because ours' byte LM applies none before the head, with ours' initial weights
        # (the same flat file, mapped by hf_lm_state_name, read back bit for bit) and the same
        # batches; the loss is torch's mean cross-entropy on the next bytes, as before. Results
        # raced before this change were wrong for this lane; apply extra scrutiny.
        torch, np, twin = self.torch, self.np, self.twin
        F = torch.nn.functional
        dims = lm_dims(self.lane, self.shape)
        b, l, dm, h, kv, hd, ff, layers, vocab = dims
        flat = torch.from_numpy(np.ascontiguousarray(self.data["init"]))
        shapes = twin.registry(dims)
        total = 0
        for _, shp in shapes:
            n = 1
            for v in shp:
                n *= int(v)
            total += n
        if total != flat.numel():
            raise RuntimeError("REFUSED: the LM registry has %d parameters, the input %d" % (total, flat.numel()))
        transformers, cfg = hf_llama_config(self.arm, vocab=vocab, d_model=dm, n_heads=h, n_kv=kv,
                                            head_dim=hd, intermediate=ff, layers=layers, max_pos=l,
                                            eps=twin.RMS_EPS, theta=twin.ROPE_THETA)
        from transformers import LlamaForCausalLM
        torch.manual_seed(SEED)
        if PARAMS_PROBE:
            # smoke-fixes r16 (2026-10-14; neural lm-forward / lm-train-step: each torch arm of the parameter
            # probe built and randomly initialized the whole LlamaForCausalLM (21M parameters) on the CPU and
            # copied it to the GPU, for a read-back of its config, class and structure): the probe builds it on
            # torch's meta device, the same class from the same config, so the same modules, shapes, dtype and
            # head tying (transformers allocates every parameter from the config alone), and is never moved.
            # The rotary read-back below then reads a CPU instance of the model's own rotary class built from
            # the model's own config (LlamaModel.__init__ builds model.rotary_emb exactly so). Every timed
            # worker builds, loads and moves the model exactly as before.
            with torch.device("meta"):
                model = LlamaForCausalLM(cfg)
        else:
            model = LlamaForCausalLM(cfg)
        model.model.norm = torch.nn.Identity()       # ours' LM has no final norm before the head
        # smoke-fixes (2026-10-14; AMD neural/lm-forward, lm-train-step STARTED: the parameter probe was cut off
        # inside the weight copy (state[...] = flat[...].clone()) and hf_load_exact's bit-for-bit torch.equal
        # read-back of every weight). The parameter read-back (config_readback, the optimizer record, the
        # library class) does not depend on the weight values, so the smoke's parameter probe (PARAMS_PROBE)
        # builds the model and skips the weight copy and its equality check; every timed run (a worker
        # subprocess, never PARAMS_PROBE) still loads ours' weights and checks them bit for bit.
        if PARAMS_PROBE:
            self.info["weights"] = "not loaded: parameter probe only (timed runs load and check them)"
        else:
            state, off = {}, 0
            for name, shp in shapes:
                n = 1
                for v in shp:
                    n *= int(v)
                state[hf_lm_state_name(name)] = flat[off:off + n].reshape(shp).clone()
                off += n
            hf_load_exact(torch, model, state, self.arm)
        if PARAMS_PROBE:
            rotary = type(model.model.rotary_emb)(config=model.config)
        else:
            model = model.to(self.dev)
            rotary = model.model.rotary_emb
        theta = hf_check_rotary(torch, rotary, hd, twin.ROPE_THETA, self.arm)
        self.library_impl = LlamaForCausalLM
        mc = model.config
        self.config_readback = arch_readback("lm", dict(
            layers=len(model.model.layers), d_model=mc.hidden_size, n_heads=mc.num_attention_heads,
            n_kv_heads=mc.num_key_value_heads, head_dim=getattr(mc, "head_dim", None),
            intermediate=mc.intermediate_size, vocab=model.lm_head.out_features,
            tied_head=model.lm_head.weight is model.model.embed_tokens.weight,
            param_dtype=str(next(model.parameters()).dtype).replace("torch.", "")))
        attn = getattr(model.config, "_attn_implementation", None)
        if attn != "sdpa":
            raise RuntimeError("REFUSED: %s: transformers' attention implementation reads back %r, not "
                               "sdpa" % (self.arm, attn))
        if self.kind == "cuda":
            self.info["sdpa"] = twin.choose_sdpa_backend(
                torch, "auto", self.dev, dims, self._sync,
                self.info["setting"].split("-")[1] == "bf16" and "bfloat16" or None)
        else:
            self.info["sdpa"] = {"backend": "torch_default",
                                 "selection": "%s: torch's own dispatch (no backend switches exist)"
                                              % self.kind}
        self.info["twin"] = ("transformers %s LlamaForCausalLM (attn sdpa, rms_norm_eps %g, RoPE base %g "
                             "read back, untied head, final norm Identity: ours has none), ours' "
                             "initial weights" % (transformers.__version__, cfg.rms_norm_eps, theta))
        self.info["transformers_version"] = transformers.__version__
        self.batches = self.data["batches"]
        if self.train:
            model.train()

            class Step(torch.nn.Module):
                def __init__(self):
                    super().__init__()
                    self.m = model

                def forward(self, ids):
                    logits = self.m(input_ids=ids[:, :-1], use_cache=False).logits
                    return F.cross_entropy(logits.reshape(b * l, vocab).float(),
                                           ids[:, 1:].reshape(b * l), reduction="mean")

            self.module = Step()
        else:
            model.eval()

            class Logits(torch.nn.Module):
                def __init__(self):
                    super().__init__()
                    self.m = model

                def forward(self, ids):
                    return self.m(input_ids=ids, use_cache=False).logits

            self.module = Logits()
            self.host = [torch.from_numpy(np.ascontiguousarray(self.batches[0][:, :-1]).astype(np.int64))]

    def _hf_layer(self, W, d, length, eps, theta):
        """transformers' own LlamaDecoderLayer holding ours' block weights W (our names, numpy),
        with RoPE cos/sin for positions 0..length-1 from its LlamaRotaryEmbedding, precomputed on
        the device before the clock (as the previous twin precomputed its tables). FAIRNESS FIX
        (2026-10-13 audit): replaces tools/speed_torch_seq.py's LlamaEager, harness code. Results
        raced before this change were wrong for this lane; apply extra scrutiny."""
        torch, np = self.torch, self.np
        transformers, cfg = hf_llama_config(self.arm, vocab=1, d_model=d["d_model"],
                                            n_heads=d["n_heads"], n_kv=d["n_kv"],
                                            head_dim=d["head_dim"], intermediate=d["intermediate"],
                                            layers=1, max_pos=length, eps=eps, theta=theta)
        from transformers.models.llama.modeling_llama import LlamaDecoderLayer, LlamaRotaryEmbedding
        torch.manual_seed(SEED)
        layer = LlamaDecoderLayer(cfg, layer_idx=0)
        self.library_impl = LlamaDecoderLayer
        state = {HF_LAYER_NAME[k]: torch.from_numpy(np.ascontiguousarray(v)) for k, v in W.items()}
        hf_load_exact(torch, layer, state, self.arm)
        layer = layer.to(self.dev)
        rotary = LlamaRotaryEmbedding(config=cfg).to(self.dev)
        back = hf_check_rotary(torch, rotary, d["head_dim"], theta, self.arm)
        pos = torch.arange(length, device=self.dev)[None, :]
        with torch.no_grad():
            cos, sin = rotary(torch.zeros((1, length, d["d_model"]), dtype=torch.float32,
                                          device=self.dev), pos)
        self.block_consts = {"eps": float(cfg.rms_norm_eps), "rope_theta": back}
        self.info["transformers_version"] = transformers.__version__
        return layer, cfg, cos, sin, pos

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
        # FAIRNESS FIX (2026-10-13 audit): this arm ran tools/speed_torch_seq.py's LlamaEager (harness
        # code from torch ops), forward and per-token decode. It now runs transformers' own
        # LlamaDecoderLayer with ours' weights (_hf_layer), SDPA attention, and for the decode lane
        # the layer's own KV cache (transformers DynamicCache), one token per call. Results raced
        # before this change were wrong for this lane; apply extra scrutiny.
        torch = self.torch
        d = _dims_of(self.lane, self.shape)
        L = d["length"]
        layer, cfg, cos, sin, pos = self._hf_layer(_weights(self.data), d, L, BLOCK_NORM_EPS,
                                                   BLOCK_ROPE_THETA)
        layer.eval()
        self.host = [self._t(self.data["x"])]
        if self.lane == "transformer-decode":
            import inspect
            from transformers import DynamicCache
            try:
                params = inspect.signature(layer.forward).parameters
            except (TypeError, ValueError):
                params = {}
            cache_kw = ("past_key_value" if "past_key_value" in params and "past_key_values" not in params
                        else "past_key_values")
            try:
                inspect.signature(DynamicCache).bind(config=cfg)
                new_cache = lambda: DynamicCache(config=cfg)  # noqa: E731
            except TypeError:
                new_cache = DynamicCache

            class Decode(torch.nn.Module):
                def __init__(self):
                    super().__init__()
                    self.layer = layer

                def forward(self, x):
                    cache = new_cache()
                    outs = []
                    for t in range(L):
                        outs.append(hf_layer_call(self.layer, x[:, t:t + 1, :], cos[:, t:t + 1],
                                                  sin[:, t:t + 1], pos[:, t:t + 1], use_cache=True,
                                                  cache_position=pos[0, t:t + 1], **{cache_kw: cache}))
                    return torch.cat(outs, dim=1)

            self.module = Decode()
            self.info["twin"] = ("transformers %s LlamaDecoderLayer, one token per call over its "
                                 "DynamicCache (%s=)" % (self.info["transformers_version"], cache_kw))
            return

        class Block(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.layer = layer

            def forward(self, x):
                return hf_layer_call(self.layer, x, cos, sin, pos)

        self.module = Block()
        self.info["twin"] = ("transformers %s LlamaDecoderLayer (attn sdpa, rms_norm_eps %g, RoPE base "
                             "read back), ours' weights" % (self.info["transformers_version"],
                                                            cfg.rms_norm_eps))

    def _mamba_call(self, model):
        corpus = _load_mamba_corpus_restoring(self.torch, self.info)
        f32 = self.torch.float32
        if model == "mamba1":
            return lambda p, x: corpus.block_forward(p, x, f32)["block.out"]
        if model == "mamba2":
            return lambda p, x: corpus.m2_forward(p, x, f32)["residual.out"].reshape(x.shape)
        return lambda p, x: corpus.m3_forward(p, x, f32)["residual.out"].reshape(x.shape)

    def _build_mamba(self):
        # FAIRNESS FIX (2026-10-13 audit): the only torch Mamba twin is the pure-PyTorch reference,
        # which is not an opponent; race() refuses these arms before a worker starts, and the
        # worker refuses too (a direct `worker` call). mamba_ssm's kernels are MambaSsmArm. Results
        # raced before this change were wrong for this lane; apply extra scrutiny.
        # SCORER GAPS (2026-10-15): the one exception is the untimed quality reference
        # (torch_reference_outputs), which is never an arm or an opponent.
        if self.lane in REFERENCE_TWIN_LANES and os.environ.get(QUALITY_REFERENCE_ENV) != "1":
            raise RuntimeError("REFUSED: " + REFERENCE_TWIN_LANES[self.lane])
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
        # FAIRNESS FIX (2026-10-13 audit): this stack ran the pure-PyTorch Mamba-3 reference
        # (gen_corpus.py m3_forward) and the harness's LlamaEager attention blocks: a reference and
        # harness code are not an opponent. It now runs mamba_ssm's own Mamba3 Block (the SISO
        # Triton kernels; refused by name where mamba_ssm does not import, e.g. Apple, or a CPU
        # lane) and transformers' own LlamaDecoderLayer for the attention layers, ours' weights in
        # every one (read back bit for bit), torch's Embedding, F.rms_norm (eps 1e-5) and the tied
        # head, mean CE, torch.optim.AdamW. UNVERIFIED AT RUNTIME: mamba_ssm Mamba3's backward
        # (samba-train-step), its behaviour under autocast and torch.compile, and the ROCm build.
        # Results raced before this change were wrong for this lane; apply extra scrutiny.
        torch, np = self.torch, self.np
        F = torch.nn.functional
        if self.kind != "cuda":
            raise RuntimeError("REFUSED: %s on %s: mamba_ssm's Mamba3 kernels run on CUDA or ROCm/HIP "
                               "only; there is no library Mamba-3 on this device and the pure-PyTorch "
                               "reference is not an opponent" % (self.arm, self.kind))
        d = _dims_of(self.lane, self.shape)
        B, L, dm, V = d["batch"], d["length"], d["d_model"], d["vocab"]
        # the Mamba3 Triton kernels' fp32 tl.dot precision follows the arm's setting (Triton's own
        # default is TF32 on NVIDIA): IEEE on the fp32 and bf16 arms, TF32 on the tf32 arms, set
        # before any kernel compiles, as the mamba-ssm-* arms do
        tf32 = self.info["setting"].endswith("tf32")
        os.environ["TRITON_F32_DEFAULT"] = "tf32" if tf32 else "ieee"
        self.info["triton_f32_default_set"] = os.environ["TRITON_F32_DEFAULT"]
        corpus = _load_mamba_corpus_restoring(torch, self.info)
        W = _weights(self.data)
        hd = dm // d["n_heads"]
        layers, kinds, rope, ssm_back, attn_cfg, ssm_cls = torch.nn.ModuleList(), [], None, None, None, None
        for i, kind in enumerate(d["layers"]):
            pre = "layers.%d." % i
            w = {k[len(pre):]: v for k, v in W.items() if k.startswith(pre)}
            if kind == "mamba3":
                blk, ssm_back, _eps_back, fused, _cfg, _eps = mamba_ssm_block(
                    torch, self.dev, "mamba3", corpus, dm, w, self.arm)
                layers.append(blk)
                ssm_cls = ssm_cls or type(blk)
                self.info["mamba3_fused_path"] = fused
            else:
                layer, _cfg, cos, sin, pos = self._hf_layer(
                    w, dict(d, n_kv=d["n_heads"], head_dim=hd), L, BLOCK_NORM_EPS, BLOCK_ROPE_THETA)
                attn_cfg = _cfg
                layers.append(layer)
                rope = (cos, sin, pos)
            kinds.append(kind)
        embed = torch.nn.Embedding(V, dm)
        hf_load_exact(torch, embed, {"weight": torch.from_numpy(np.ascontiguousarray(W["embed.weight"]))},
                      self.arm)
        embed = embed.to(self.dev)
        norm_w = torch.nn.Parameter(torch.from_numpy(np.ascontiguousarray(W["norm_f.weight"])).to(self.dev))
        eps = float(torch.tensor(SAMBA_NORM_EPS, dtype=torch.float32))
        # the final norm's eps as passed to ours; the attention layers' eps and RoPE base read back
        self.block_consts = dict(self.block_consts, eps=SAMBA_NORM_EPS)
        self.info.update(_mamba_ssm_info(torch, self.arm,
                                         {"triton_f32_default": os.environ["TRITON_F32_DEFAULT"]}))
        train = self.train

        class Stack(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.embed, self.layers, self.norm_w = embed, layers, norm_w

            def forward(self, ids, targets=None):
                x = self.embed(ids)
                for kind, m in zip(kinds, self.layers):
                    if kind == "mamba3":
                        h, r = m(x)
                        x = h + r
                    else:
                        x = hf_layer_call(m, x, rope[0], rope[1], rope[2])
                logits = F.linear(F.rms_norm(x, (dm,), self.norm_w, eps), self.embed.weight)
                if not train:
                    return logits
                return F.cross_entropy(logits.reshape(B * L, V).float(), targets.reshape(B * L),
                                       reduction="mean")

        self.module = Stack()
        self.module.train(train)
        # the library classes in the stack: mamba_ssm's Block where present (else transformers' Llama layer)
        if ssm_cls is not None:
            self.library_impl = ssm_cls
        self.config_readback = arch_readback("samba", dict(
            layers=kinds, d_model=embed.embedding_dim, vocab=embed.num_embeddings,
            n_heads=getattr(attn_cfg, "num_attention_heads", None) if attn_cfg is not None else d["n_heads"],
            n_kv_heads=getattr(attn_cfg, "num_key_value_heads", None) if attn_cfg is not None else None,
            head_dim=getattr(attn_cfg, "head_dim", None) if attn_cfg is not None else None,
            intermediate=getattr(attn_cfg, "intermediate_size", None) if attn_cfg is not None else None,
            tied_head=True,          # Stack.forward computes the logits with self.embed.weight
            norm_eps=eps))
        self.batches = self.data["batches"]
        if not train:
            self.host = [torch.from_numpy(np.ascontiguousarray(self.batches[0][:, :-1]).astype(np.int64))]
        self.info["twin"] = ("torch Embedding; per layer mamba_ssm Mamba3 Block (%s) or transformers %s "
                             "LlamaDecoderLayer; F.rms_norm eps 1e-5; tied head; mean CE"
                             % (self.info.get("mamba3_fused_path"), self.info.get("transformers_version")))
        if ssm_back is not None:
            self.info["mamba3_config_readback"] = ssm_back

    def _build_mlp(self):
        torch = self.torch
        F = torch.nn.functional
        train = self.train

        # smoke-fixes (2026-10-14; mlp-predict, mlp-train-step refused with "missing real library implementation
        # read-back: torch-*"): the twin was F.linear calls in a harness closure, so its implementation read back
        # as bench_board_neural. It is now torch's own modules, torch.nn.Sequential(Linear, ReLU, Linear), holding
        # ours' weights (nn.Linear computes x @ W.T + b with W (out, in), the same F.linear as before: same work,
        # same math), and the arm reads back the class it built.
        p = self._dev_params()
        nn = torch.nn
        seq = nn.Sequential(nn.Linear(p["weight1"].shape[1], p["weight1"].shape[0], device=self.dev,
                                      dtype=p["weight1"].dtype),
                            nn.ReLU(),
                            nn.Linear(p["weight2"].shape[1], p["weight2"].shape[0], device=self.dev,
                                      dtype=p["weight2"].dtype))
        seq[0].weight, seq[0].bias = nn.Parameter(p["weight1"]), nn.Parameter(p["bias1"])
        seq[2].weight, seq[2].bias = nn.Parameter(p["weight2"]), nn.Parameter(p["bias2"])

        class MLP(nn.Module):
            def __init__(self):
                super().__init__()
                self.net = seq

            def forward(self, x, y=None):
                logits = self.net(x)
                if not train:
                    return logits
                return F.cross_entropy(logits.float(), y, reduction="mean")

        self.module = MLP()
        self.library_impl = type(seq)
        # refused-param-mappings (2026-10-15): the network this arm built, read from its own modules
        self.config_readback = mlp_arch(seq[0].in_features, seq[0].out_features, seq[2].out_features,
                                        type(seq[1]).__name__.lower(),
                                        seq[0].bias is not None and seq[2].bias is not None)
        if not train:
            self.host = [self._t(self.data["X"][0])]
        self.info["twin"] = "torch.nn.Sequential(Linear, ReLU, Linear) (mean CE, AdamW for the step)"

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


class TorchFitArm:
    """The fit lanes' torch opponent at one setting (2026-10-19, lane ag-bench): the same network in
    torch.nn, torch.optim.Adam, the lane's batch size and epoch count. The clock covers what ours'
    fit() covers: a fresh seeded initialization, the training rows to the device, the shuffle, every
    minibatch step (zero_grad, forward under the setting's autocast, loss, backward, Adam) and the loss
    curve back on the host. The held-out prediction runs after the clock (outputs()). The module is
    built (and compiled) once and re-initialized in place each round, so a compile arm compiles in its
    untimed warm-up fit only."""

    def __init__(self, lane, shape, data, arm):
        import math
        import numpy as np
        self.np, self.lane, self.shape, self.arm = np, lane, shape, arm
        where, setting = arm_setting(arm)
        if where != "gpu":
            raise RuntimeError("REFUSED: %s: the fit lanes race torch on the GPU only" % arm)
        twin = _load("torch_lm_step_opponent")
        col = twin.COLUMNS[setting.replace("-", "_")]
        torch, dev, kind, name, sync = _torch_device()
        self.torch, self.dev, self.kind, self._sync = torch, dev, kind, sync
        torch.manual_seed(SEED)
        hip = getattr(torch.version, "hip", None)
        if col["tf32"] and not (kind == "cuda" and not hip):
            raise RuntimeError("REFUSED: %s: TF32 is an NVIDIA CUDA matmul mode; torch on %s "
                               "accepts the flag and changes nothing" % (arm, "ROCm" if hip else kind))
        precision = twin.set_precision(torch, col["tf32"])
        self.autocast, probe = None, None
        if col["autocast"]:
            probe = twin.probe_autocast(torch, dev, col["autocast"], sync)
            if not probe.get("applicable"):
                raise RuntimeError("REFUSED: %s: torch.autocast(%r, %s) does not work on this box: %s"
                                   % (arm, dev.type, col["autocast"], json.dumps(probe)[:600]))
            self.autocast = getattr(torch, col["autocast"])
        self.compile = bool(col["compile"])
        # host tensors (no copy): moved to the device inside every timed fit
        self.Xtr = torch.from_numpy(np.ascontiguousarray(data["X_train"], dtype=np.float32))
        self.ytr = torch.from_numpy(np.ascontiguousarray(data["y_train"], dtype=np.float32))
        self.Xte = torch.from_numpy(np.ascontiguousarray(data["X_test"], dtype=np.float32))
        d_in = int(self.Xtr.shape[-1])
        nn, F = torch.nn, torch.nn.functional
        recurrent = lane in FIT_CELL
        task = FIT_TASK[lane]
        if recurrent:
            cls = {"lstm": nn.LSTM, "gru": nn.GRU, "rnn": nn.RNN}[FIT_CELL[lane]]
            extra = {"nonlinearity": "tanh"} if lane == "rnn-fit" else {}
            body = cls(d_in, FIT_RNN_HIDDEN, num_layers=1, bias=True, batch_first=True, device=dev, **extra)
            head = nn.Linear(FIT_RNN_HIDDEN, 1, device=dev)
            impl = type(body)
            self.linears = []
        else:
            sizes = [d_in] + list(FIT_HIDDEN[lane]) + [1]
            mods = []
            for i in range(len(sizes) - 1):
                mods.append(nn.Linear(sizes[i], sizes[i + 1], device=dev))
                if i < len(sizes) - 2:
                    mods.append(nn.ReLU())
            body, head = nn.Sequential(*mods), None
            impl = type(body)
            self.linears = [m for m in mods if isinstance(m, nn.Linear)]

        class FitNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.body = body
                self.head = head

            def forward(self, x, y=None):
                if recurrent:
                    z = self.head(self.body(x)[0][:, -1]).squeeze(-1)
                else:
                    z = self.body(x).squeeze(-1)
                if y is None:
                    return z
                if task == "classification":
                    return F.binary_cross_entropy_with_logits(z.float(), y, reduction="mean")
                return F.mse_loss(z.float(), y, reduction="mean")

        self.module = FitNet()
        self.body = body
        self.head = head
        self.recurrent = recurrent
        self.task = task
        self.fn = torch.compile(self.module) if self.compile else self.module
        # every Adam flag explicit where this torch takes it (fused/foreach left to torch: implementation)
        import inspect
        try:
            sig = inspect.signature(torch.optim.Adam).parameters
        except (TypeError, ValueError):
            sig = {}
        self.opt_kwargs = dict(lr=FIT_ADAM["lr"], betas=tuple(FIT_ADAM["betas"]), eps=FIT_ADAM["eps"],
                               weight_decay=FIT_ADAM["weight_decay"])
        self.opt_kwargs.update({k: False for k in ("amsgrad", "maximize", "capturable", "differentiable")
                                if k in sig})
        back = torch.optim.Adam(self.module.parameters(), **self.opt_kwargs).defaults
        try:
            torch.randperm(4, device=dev)
            self.perm_where = "torch.randperm on the device"
        except Exception:  # noqa: BLE001  (a backend without a device randperm)
            self.perm_where = "torch.randperm on the host, moved to the device inside the clock"
        if recurrent:
            layers = fit_layers(lane, d_in, [body.hidden_size])
            extra_rb = dict(num_layers=body.num_layers, timesteps=int(self.Xtr.shape[1]))
            if lane == "rnn-fit":
                extra_rb["nonlinearity"] = body.nonlinearity
        else:
            layers = []
            for m in body:
                layers.append("linear %d %d %s" % (m.in_features, m.out_features, "bias" if m.bias is not None
                                                   else "nobias") if isinstance(m, nn.Linear)
                              else type(m).__name__.lower())
            extra_rb = {}
        self.record = fit_record(lane, "torch", impl, d_in, optimizer="adam", learning_rate=back["lr"],
                                 betas=list(back["betas"]), eps=back["eps"], weight_decay=back["weight_decay"],
                                 amsgrad=back.get("amsgrad", False), maximize=back.get("maximize", False),
                                 layers=layers, **extra_rb)
        self.info = {"library": "torch", "version": torch.__version__,
                     "torch_version_cuda": torch.version.cuda, "torch_version_hip": hip,
                     "torch_backend": kind, "device": "gpu", "device_name": name,
                     "setting": setting, "precision": precision_text(setting),
                     "compile": "torch.compile (inductor, default mode)" if self.compile else "eager",
                     "mode": "torch %s, %s" % ("compile" if self.compile else "eager", precision_text(setting)),
                     "precision_readback": precision, "autocast_probe": probe,
                     "column": setting.replace("-", "_") + " (tools/torch_lm_step_opponent.py COLUMNS)",
                     "torch_threads": torch.get_num_threads(), "pre_clock_fit": False, "input_home": "host",
                     "call": LANE_TEXT[lane][1], "shuffle": self.perm_where,
                     "init": ("torch's own reset_parameters (U(+-1/sqrt(hidden)) on the cell, nn.Linear's "
                              "default on the head: the same bound)" if recurrent else
                              "Glorot uniform sqrt(6/(fan_in+fan_out)) on every weight and bias (scikit-learn's "
                              "bounds, ours' rule)"),
                     "fit_rows": int(self.Xtr.shape[0]), "held_out_rows": int(self.Xte.shape[0])}
        self._glorot = lambda m: math.sqrt(6.0 / (m.in_features + m.out_features))
        self.losses = None
        self._score = None
        sync()

    def _ctx(self, grad=True):
        import contextlib
        torch = self.torch
        stack = contextlib.ExitStack()
        if self.autocast is not None:
            stack.enter_context(torch.autocast(device_type=self.dev.type, dtype=self.autocast))
        if not grad:
            stack.enter_context(torch.no_grad())
        return stack

    def _reinit(self):
        torch = self.torch
        with torch.no_grad():
            if self.recurrent:
                self.body.reset_parameters()
                self.head.reset_parameters()
            else:
                for m in self.linears:
                    b = self._glorot(m)
                    m.weight.uniform_(-b, b)
                    m.bias.uniform_(-b, b)

    def _perm(self, n):
        torch = self.torch
        if self.perm_where.endswith("on the device"):
            return torch.randperm(n, device=self.dev)
        return torch.randperm(n).to(self.dev)

    def call(self):
        np, torch = self.np, self.torch
        torch.manual_seed(SEED)
        self._reinit()
        opt = torch.optim.Adam(self.module.parameters(), **self.opt_kwargs)
        X, y = self.Xtr.to(self.dev), self.ytr.to(self.dev)
        n = int(X.shape[0])
        bs = max(1, min(FIT_BATCH, n))
        losses = []
        for _epoch in range(FIT_EPOCHS):
            perm = self._perm(n)
            for s in range(0, n, bs):
                idx = perm[s:s + bs]
                opt.zero_grad(set_to_none=True)
                with self._ctx():
                    loss = self.fn(X[idx], y[idx])
                loss.backward()
                opt.step()
                losses.append(loss.detach())
        self.losses = torch.stack(losses).float().cpu().numpy().astype(np.float64)
        self._score = None
        if self.lane in FIT_PREDICT_IN_CLOCK:
            self._predict()               # sequential-fit: the held-out probabilities inside the clock

    def sync(self):
        self._sync()

    def _predict(self):
        torch = self.torch
        if self._score is None:
            parts = []
            with self._ctx(grad=False):
                X = self.Xte.to(self.dev)
                for s in range(0, int(X.shape[0]), FIT_PREDICT_CHUNK):
                    z = self.module(X[s:s + FIT_PREDICT_CHUNK]).float()
                    parts.append(torch.sigmoid(z) if self.task == "classification" else z)
            self._score = self.np.ascontiguousarray(torch.cat(parts).cpu().numpy().astype(self.np.float32))
        return self._score

    def outputs(self):
        return {"score": self._predict(), "loss_curve": self.losses}

    def digest(self):
        return _sha(self.np.ascontiguousarray(self._predict()).data)[:16]


class TorchOpsStepArm:
    """ops-session-step's torch opponent at one setting (2026-10-19, lane ag-board-twins): the same step in
    torch autograd. The clock covers what ours' covers: the batch, labels and four parameters to the device
    (fresh leaf tensors with requires_grad every round), relu(x @ w1 + b1) @ w2 + b2 under the setting's
    autocast, F.cross_entropy (mean), backward, the loss and every gradient back on the host,
    torch.optim.SGD(lr, momentum=0, weight_decay=0).step() and every updated parameter back on the host. A
    compile arm compiles the forward + loss function in its untimed warm-up round."""

    def __init__(self, lane, shape, data, arm):
        import inspect
        import numpy as np
        self.np, self.lane, self.shape, self.arm = np, lane, shape, arm
        where, setting = arm_setting(arm)
        if where != "gpu":
            raise RuntimeError("REFUSED: %s: ops-session-step races torch on the GPU only" % arm)
        twin = _load("torch_lm_step_opponent")
        col = twin.COLUMNS[setting.replace("-", "_")]
        torch, dev, kind, name, sync = _torch_device()
        self.torch, self.dev, self.kind, self._sync = torch, dev, kind, sync
        torch.manual_seed(SEED)
        hip = getattr(torch.version, "hip", None)
        if col["tf32"] and not (kind == "cuda" and not hip):
            raise RuntimeError("REFUSED: %s: TF32 is an NVIDIA CUDA matmul mode; torch on %s "
                               "accepts the flag and changes nothing" % (arm, "ROCm" if hip else kind))
        precision = twin.set_precision(torch, col["tf32"])
        self.autocast, probe = None, None
        if col["autocast"]:
            probe = twin.probe_autocast(torch, dev, col["autocast"], sync)
            if not probe.get("applicable"):
                raise RuntimeError("REFUSED: %s: torch.autocast(%r, %s) does not work on this box: %s"
                                   % (arm, dev.type, col["autocast"], json.dumps(probe)[:600]))
            self.autocast = getattr(torch, col["autocast"])
        self.compile = bool(col["compile"])
        # host tensors (no copy): moved to the device inside every timed round
        self.p0 = [torch.from_numpy(np.ascontiguousarray(data["w:" + n], dtype=np.float32)) for n in OPS_PARAMS]
        self.X = torch.from_numpy(np.ascontiguousarray(data["X"], dtype=np.float32))
        self.y = torch.from_numpy(np.ascontiguousarray(data["y"], dtype=np.int64))
        F = torch.nn.functional

        def step_loss(x, y, w1, b1, w2, b2):
            h = torch.relu(x @ w1 + b1)
            return F.cross_entropy((h @ w2 + b2).float(), y, reduction="mean")
        self.fn = torch.compile(step_loss) if self.compile else step_loss
        try:
            sig = inspect.signature(torch.optim.SGD).parameters
        except (TypeError, ValueError):
            sig = {}
        self.opt_kwargs = dict(lr=OPS_SGD["lr"], momentum=OPS_SGD["momentum"], weight_decay=OPS_SGD["weight_decay"])
        self.opt_kwargs.update({k: v for k, v in (("dampening", 0.0), ("nesterov", False), ("maximize", False),
                                                   ("differentiable", False)) if k in sig})
        back = torch.optim.SGD([torch.zeros(1, requires_grad=True)], **self.opt_kwargs).defaults
        d_in, hidden = (int(v) for v in self.p0[0].shape)
        classes = int(self.p0[2].shape[1])
        self.record = ops_record("torch", torch.optim.SGD, shape, learning_rate=back["lr"],
                                 momentum=back["momentum"], weight_decay=back["weight_decay"],
                                 nesterov=back.get("nesterov", False), batch_size=int(self.X.shape[0]),
                                 layers=ops_layers(d_in, hidden, classes))
        self.info = {"library": "torch", "version": torch.__version__,
                     "torch_version_cuda": torch.version.cuda, "torch_version_hip": hip,
                     "torch_backend": kind, "device": "gpu", "device_name": name,
                     "setting": setting, "precision": precision_text(setting),
                     "compile": "torch.compile (inductor, default mode)" if self.compile else "eager",
                     "mode": "torch %s, %s" % ("compile" if self.compile else "eager", precision_text(setting)),
                     "precision_readback": precision, "autocast_probe": probe,
                     "column": setting.replace("-", "_") + " (tools/torch_lm_step_opponent.py COLUMNS)",
                     "torch_threads": torch.get_num_threads(), "pre_clock_fit": False, "input_home": "host",
                     "call": LANE_TEXT[lane][1]}
        self.loss, self.grads, self.updated = None, None, None
        sync()

    def _ctx(self):
        import contextlib
        stack = contextlib.ExitStack()
        if self.autocast is not None:
            stack.enter_context(self.torch.autocast(device_type=self.dev.type, dtype=self.autocast))
        return stack

    def call(self):
        torch = self.torch
        torch.manual_seed(SEED)
        x, y = self.X.to(self.dev), self.y.to(self.dev)
        ps = [p.to(self.dev, copy=True).requires_grad_(True) for p in self.p0]
        opt = torch.optim.SGD(ps, **self.opt_kwargs)
        with self._ctx():
            loss = self.fn(x, y, *ps)
        loss.backward()
        self.loss = float(loss.item())
        self.grads = [p.grad.detach().float().cpu().numpy() for p in ps]
        opt.step()
        self.updated = [p.detach().float().cpu().numpy() for p in ps]

    def sync(self):
        self._sync()

    def outputs(self):
        return _ops_outputs(self.np, self.loss, self.grads, self.updated)

    def digest(self):
        return _ops_digest(self.np, self.grads)


class TorchProgramArm:
    """program-run's torch opponent at one setting (2026-10-19, lane ag-program): the same forward under
    torch.no_grad. The four weights are on the device before the clock (ours' are the program's resident
    parameters); a round moves the batch to the device, runs softmax(relu(x @ w1 + b1) @ w2 + b2) under the
    setting's autocast and brings the probabilities back on the host. A compile arm compiles the forward in its
    untimed warm-up round."""
#: the byte-LM tensor names -> transformers' Llama names (the 162M's export, research/gpt3-small-162m)
BYTE_LM_HF_NAME = {"norm1_w": "input_layernorm.weight", "w_q": "self_attn.q_proj.weight",
                   "w_k": "self_attn.k_proj.weight", "w_v": "self_attn.v_proj.weight",
                   "w_o": "self_attn.o_proj.weight", "norm2_w": "post_attention_layernorm.weight",
                   "w_gate": "mlp.gate_proj.weight", "w_up": "mlp.up_proj.weight", "w_down": "mlp.down_proj.weight"}


def byte_lm_hf_state(state):
    """{byte-LM name: tensor} -> {transformers LlamaForCausalLM name: tensor} (names only, no arithmetic)."""
    out = {}
    for k, v in state.items():
        if k == "embed":
            out["model.embed_tokens.weight"] = v
        elif k == "lm_head":
            out["lm_head.weight"] = v
        else:
            blk, name = k.split(".", 1)
            out["model.layers.%d.%s" % (int(blk[len("block"):]), BYTE_LM_HF_NAME[name])] = v
    return out


class TorchGenerateArm:
    """causal-lm-generate's opponent at one eager torch setting (2026-10-19, lane ag-inference-board):
    transformers' LlamaForCausalLM on the same checkpoint, built on the host at ready and moved to the device in
    the untimed warm-up round (its device bytes read around the move), then every round generate(ids, 1)
    (prefill_ms) and generate(ids, N) (the clock) with an explicit GenerationConfig (greedy, KV cache, no stop
    token), the ids from the host and back. The 162M's byte-LM tensors are renamed to Llama's, loaded
    strict=True and read back bit for bit, with model.norm an Identity (it has no final norm). bf16 arms hold the
    weights in bfloat16 (GEN_PRECISION). No compile arm: generate is a growing-cache Python loop."""

    def __init__(self, lane, shape, data, arm):
        import numpy as np
        self.np, self.lane, self.shape, self.arm = np, lane, shape, arm
        where, setting = arm_setting(arm)
        if where != "gpu":
            raise RuntimeError("REFUSED: %s: program-run races torch on the GPU only" % arm)
        twin = _load("torch_lm_step_opponent")
        col = twin.COLUMNS[setting.replace("-", "_")]
        self.dm = DeviceMemory()
        where, setting = arm_setting(arm)
        if where != "gpu":
            raise RuntimeError("REFUSED: %s: %s races torch on the GPU only" % (arm, lane))
        twin = _load("torch_lm_step_opponent")
        col = twin.COLUMNS[setting.replace("-", "_")]
        if col["compile"]:
            raise RuntimeError("REFUSED: %s: transformers' generate is a per-token loop over a growing KV cache; "
                               "%s races the eager arms only" % (arm, lane))
        torch, dev, kind, name, sync = _torch_device()
        self.torch, self.dev, self.kind, self._sync = torch, dev, kind, sync
        torch.manual_seed(SEED)
        hip = getattr(torch.version, "hip", None)
        if col["tf32"] and not (kind == "cuda" and not hip):
            raise RuntimeError("REFUSED: %s: TF32 is an NVIDIA CUDA matmul mode; torch on %s "
                               "accepts the flag and changes nothing" % (arm, "ROCm" if hip else kind))
        precision = twin.set_precision(torch, col["tf32"])
        self.autocast, probe = None, None
        if col["autocast"]:
            probe = twin.probe_autocast(torch, dev, col["autocast"], sync)
            if not probe.get("applicable"):
                raise RuntimeError("REFUSED: %s: torch.autocast(%r, %s) does not work on this box: %s"
                                   % (arm, dev.type, col["autocast"], json.dumps(probe)[:600]))
            self.autocast = getattr(torch, col["autocast"])
        self.compile = bool(col["compile"])
        self.ws = [torch.from_numpy(np.ascontiguousarray(data["w:" + n], dtype=np.float32)).to(dev)
                   for n in OPS_PARAMS]
        self.X = torch.from_numpy(np.ascontiguousarray(data["X"], dtype=np.float32))

        def forward(x, w1, b1, w2, b2):
            h = torch.relu(x @ w1 + b1)
            return torch.softmax((h @ w2 + b2).float(), dim=-1)
        self.fn = torch.compile(forward) if self.compile else forward
        d_in, hidden = (int(v) for v in self.ws[0].shape)
        classes = int(self.ws[2].shape[1])
        self.record = program_record("torch", torch.nn.functional.softmax, shape, batch_size=int(self.X.shape[0]),
                                     layers=program_layers(d_in, hidden, classes))
        self.info = {"library": "torch", "version": torch.__version__,
                     "torch_version_cuda": torch.version.cuda, "torch_version_hip": hip,
                     "torch_backend": kind, "device": "gpu", "device_name": name,
                     "setting": setting, "precision": precision_text(setting),
                     "compile": "torch.compile (inductor, default mode)" if self.compile else "eager",
                     "mode": "torch %s, %s" % ("compile" if self.compile else "eager", precision_text(setting)),
                     "precision_readback": precision, "autocast_probe": probe,
                     "column": setting.replace("-", "_") + " (tools/torch_lm_step_opponent.py COLUMNS)",
                     "torch_threads": torch.get_num_threads(), "pre_clock_fit": False, "input_home": "host",
                     "call": LANE_TEXT[lane][1]}
        self.out = None
        sync()

    def call(self):
        import contextlib
        torch = self.torch
        with torch.no_grad(), contextlib.ExitStack() as stack:
            if self.autocast is not None:
                stack.enter_context(torch.autocast(device_type=self.dev.type, dtype=self.autocast))
            y = self.fn(self.X.to(self.dev), *self.ws)
        self.out = y.float().cpu().numpy()
        self.bf16 = bool(col["autocast"])
        transformers, LlamaConfig, modeling_llama = _transformers(arm)
        from transformers import GenerationConfig
        self.ids_np, self.n_new, self.ckpt_dir, self.key = _gen_inputs(np, data)
        self.ids = torch.from_numpy(self.ids_np.astype(np.int64))
        self.mask = torch.ones_like(self.ids)
        with open(os.path.join(self.ckpt_dir, "config.json")) as fh:
            raw = json.load(fh)
        byte_lm = raw.get("format") == LM_CHECKPOINTS["mojolearn-162m"]["fmt"]
        if byte_lm:
            _, cfg = hf_llama_config(arm, vocab=raw["vocab_size"], d_model=raw["d_model"], n_heads=raw["n_heads"],
                                     n_kv=raw["n_kv"], head_dim=raw["head_dim"], intermediate=raw["intermediate"],
                                     layers=raw["n_layers"], max_pos=raw["max_position_embeddings"],
                                     eps=raw["rms_norm_eps"], theta=raw["rope_theta"])
            cfg.use_cache = True
            model = modeling_llama.LlamaForCausalLM(cfg)
            model.model.norm = torch.nn.Identity()      # the 162M has no final norm
            from safetensors.torch import load_file
            state = byte_lm_hf_state(load_file(os.path.join(self.ckpt_dir, "model.safetensors")))
            hf_load_exact(torch, model, state, arm)
            theta, eps, final_norm = float(raw["rope_theta"]), float(raw["rms_norm_eps"]), False
        else:
            try:
                model = modeling_llama.LlamaForCausalLM.from_pretrained(self.ckpt_dir, dtype=torch.float32,
                                                                        attn_implementation="sdpa")
            except TypeError:
                model = modeling_llama.LlamaForCausalLM.from_pretrained(self.ckpt_dir, torch_dtype=torch.float32,
                                                                        attn_implementation="sdpa")
            cfg = model.config
            theta = float(getattr(cfg, "rope_theta", None) or
                          (getattr(cfg, "rope_parameters", None) or {}).get("rope_theta"))
            eps, final_norm = float(cfg.rms_norm_eps), True
        model.eval()
        rotary = getattr(model.model, "rotary_emb", None)
        head_dim = int(getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads)
        if rotary is not None:
            hf_check_rotary(torch, rotary, head_dim, theta, arm)
        if self.bf16:
            model = model.to(torch.bfloat16)
        self.model = model
        self._on_device = False
        self.resident = None
        self.gc = {n: GenerationConfig(max_new_tokens=int(n), do_sample=False, num_beams=1, use_cache=True,
                                       eos_token_id=None, bos_token_id=None, pad_token_id=0)
                   for n in (1, self.n_new)}
        model.generation_config = self.gc[self.n_new]   # no checkpoint default (an eos, a sampler) reaches generate
        tied = bool(getattr(cfg, "tie_word_embeddings", False))
        self.record = gen_record(lane, "transformers", modeling_llama.LlamaForCausalLM, shape,
                                 batch_size=int(self.ids.shape[0]), prompt_tokens=int(self.ids.shape[1]),
                                 new_tokens=self.n_new, n_layers=int(cfg.num_hidden_layers),
                                 d_model=int(cfg.hidden_size), n_heads=int(cfg.num_attention_heads),
                                 n_kv_heads=int(cfg.num_key_value_heads), vocab_size=int(cfg.vocab_size),
                                 final_norm=final_norm, tied_head=tied, rope_theta=theta, norm_eps=eps)
        self.first, self.out, self.timing = None, None, None
        self.info = {"library": "torch", "version": torch.__version__, "transformers_version": transformers.__version__,
                     "torch_version_cuda": torch.version.cuda, "torch_version_hip": hip,
                     "torch_backend": kind, "device": "gpu", "device_name": name,
                     "setting": setting, "precision": GEN_PRECISION.get(setting, precision_text(setting)),
                     "compile": "eager", "mode": "torch eager, %s" % GEN_PRECISION.get(setting, setting),
                     "precision_readback": precision, "column": setting.replace("-", "_"),
                     "torch_threads": torch.get_num_threads(), "pre_clock_fit": False, "input_home": "host",
                     "call": LANE_TEXT[lane][1], "checkpoint_dir": self.ckpt_dir, "device_memory": self.dm.method(),
                     "attn_implementation": getattr(cfg, "_attn_implementation", None),
                     "load": "built on the host at ready; .to(device) in the untimed warm-up round (round 0)"}

    def _gen(self, n):
        torch = self.torch
        with torch.inference_mode():
            out = self.model.generate(input_ids=self.ids.to(self.dev), attention_mask=self.mask.to(self.dev),
                                      generation_config=self.gc[n])
            got = out.cpu().numpy().astype(self.np.int32)
        want = (int(self.ids.shape[0]), int(self.ids.shape[1]) + int(n))
        if tuple(got.shape) != want:
            raise RuntimeError("REFUSED: %s: generate returned shape %s, not %s (a stop token or a cap ended it)"
                               % (self.arm, list(got.shape), list(want)))
        return got

    def call(self):
        if not self._on_device:
            self._sync()
            before = self.dm.read()
            self.model.to(self.dev)
            self._sync()
            self.resident = _mem_delta(before, self.dm.read())
            self._on_device = True
        t0 = time.perf_counter()
        first = self._gen(1)
        t1 = time.perf_counter()
        out = self._gen(self.n_new)
        t2 = time.perf_counter()
        self.first, self.out = first, out
        self.timing = _gen_timing(t0, t1, t2, self.n_new)

    def round_extra(self):
        return dict(self.timing or {})

    def memory_pass(self):
        return memory_pass(self.dm, self.resident, lambda: self._gen(self.n_new),
                           torch=self.torch if self.kind == "cuda" else None)

    def sync(self):
        self._sync()

    def outputs(self):
        return {"y": self.np.asarray(self.out)}

    def digest(self):
        return _sha(self.np.ascontiguousarray(self.np.asarray(self.out)).data)[:16]
        return {"ids": self.out, "first": self.first}

    def digest(self):
        return _sha(self.np.ascontiguousarray(self.out).tobytes())[:16]


class TorchPredictArm(TorchFitArm):
    """sequential-predict's opponent (2026-10-19, lane ag-inference-board): TorchFitArm's network fitted once in
    the untimed warm-up round (its device bytes read around the fit), then every round the held-out rows to the
    device, the forward (the compiled module on a compile arm) under no_grad and the setting's autocast, the
    sigmoid back on the host, in the clock."""

    def __init__(self, lane, shape, data, arm):
        self.dm = DeviceMemory()
        TorchFitArm.__init__(self, lane, shape, data, arm)
        self._fitted, self.resident = False, None
        self.info.update(call=LANE_TEXT[lane][1], device_memory=self.dm.method(),
                         fit="one fit in the untimed warm-up round (round 0); the clock is the prediction alone")

    def _predict(self):
        torch = self.torch
        if self._score is None:
            parts = []
            with self._ctx(grad=False):
                X = self.Xte.to(self.dev)
                for s in range(0, int(X.shape[0]), FIT_PREDICT_CHUNK):     # driver-loop: over row chunks
                    parts.append(torch.sigmoid(self.fn(X[s:s + FIT_PREDICT_CHUNK]).float()))
            self._score = self.np.ascontiguousarray(torch.cat(parts).cpu().numpy().astype(self.np.float32))
        return self._score

    def call(self):
        if not self._fitted:
            self._sync()
            before = self.dm.read()
            TorchFitArm.call(self)          # the fit only: this lane is not in FIT_PREDICT_IN_CLOCK
            self._sync()
            self.resident = _mem_delta(before, self.dm.read())
            self._fitted = True
        self._score = None
        self._predict()

    def memory_pass(self):
        def again():
            self._score, keep = None, self._score
            try:
                self._predict()
            finally:
                self._score = keep
        return memory_pass(self.dm, self.resident, again, torch=self.torch if self.kind == "cuda" else None)


class MlxFitArm:
    """The fit lanes' MLX opponent on Apple silicon (2026-10-19, lane ag-bench): mlx.nn (Linear/ReLU,
    LSTM/GRU/RNN with a Linear head on the last hidden state), mlx.optimizers.Adam with bias correction,
    nn.value_and_grad, mx.eval after every step (MLX's own training-loop pattern), float32. Same clock as
    TorchFitArm: a fresh seeded model, the training rows into MLX arrays, the shuffle, every step and the
    loss curve back on the host; the held-out prediction after the clock."""

    def __init__(self, lane, shape, data, arm):
        import numpy as np
        if arm != "mlx-fp32":
            raise RuntimeError("REFUSED: no MLX arm %r (mlx-fp32 only)" % arm)
        if sys.platform != "darwin" or platform.machine() != "arm64":
            raise RuntimeError("REFUSED: %s: MLX runs on Apple silicon only" % arm)
        try:
            import mlx.core as mx
            import mlx.nn as mnn
            import mlx.optimizers as mopt
        except ImportError as exc:
            raise RuntimeError("REFUSED: %s: mlx is not installed in this environment (%r)" % (arm, exc))
        self.np, self.mx, self.mnn, self.mopt = np, mx, mnn, mopt
        self.lane, self.arm = lane, arm
        self.Xtr = np.ascontiguousarray(data["X_train"], dtype=np.float32)
        self.ytr = np.ascontiguousarray(data["y_train"], dtype=np.float32)
        self.Xte = np.ascontiguousarray(data["X_test"], dtype=np.float32)
        self.d_in = int(self.Xtr.shape[-1])
        self.recurrent = lane in FIT_CELL
        self.task = FIT_TASK[lane]
        mx.random.seed(SEED)
        model = self._build()
        opt = self._optimizer()
        layers = (fit_layers(lane, self.d_in, [FIT_RNN_HIDDEN]) if self.recurrent else
                  ["linear %d %d bias" % (m.weight.shape[1], m.weight.shape[0]) if isinstance(m, mnn.Linear)
                   else type(m).__name__.lower() for m in model.body.layers])
        extra = dict(num_layers=1, timesteps=int(self.Xtr.shape[1])) if self.recurrent else {}
        if lane == "rnn-fit":
            extra["nonlinearity"] = "tanh"
        lr = float(np.asarray(opt.learning_rate))
        self.record = fit_record(lane, "mlx", type(model.body), self.d_in, optimizer="adam", learning_rate=lr,
                                 betas=[float(b) for b in opt.betas], eps=float(opt.eps), weight_decay=0.0,
                                 layers=layers, **extra)
        self.info = {"library": "mlx", "version": getattr(mx, "__version__", "unknown"), "device": "gpu",
                     "device_name": str(mx.default_device()), "setting": "fp32",
                     "precision": "float32", "mode": "mlx eager, float32", "pre_clock_fit": False,
                     "input_home": "host", "call": "mlx.nn twin of: " + LANE_TEXT[lane][1],
                     "shuffle": "mx.random.permutation", "init": (
                         "mlx's own init (U(+-1/sqrt(hidden)) on the cell and the head)" if self.recurrent
                         else "Glorot uniform sqrt(6/(fan_in+fan_out)) on every weight and bias"),
                     "fit_rows": int(self.Xtr.shape[0]), "held_out_rows": int(self.Xte.shape[0])}
        self.model, self.losses, self._score = None, None, None

    def _build(self):
        import math
        mx, mnn = self.mx, self.mnn
        recurrent, task = self.recurrent, self.task
        if recurrent:
            cls = {"lstm": mnn.LSTM, "gru": mnn.GRU, "rnn": mnn.RNN}[FIT_CELL[self.lane]]
            body, head = cls(self.d_in, FIT_RNN_HIDDEN, bias=True), mnn.Linear(FIT_RNN_HIDDEN, 1)
        else:
            sizes = [self.d_in] + list(FIT_HIDDEN[self.lane]) + [1]
            mods = []
            for i in range(len(sizes) - 1):
                lin = mnn.Linear(sizes[i], sizes[i + 1])
                b = math.sqrt(6.0 / (sizes[i] + sizes[i + 1]))
                lin.weight = mx.random.uniform(-b, b, lin.weight.shape)
                lin.bias = mx.random.uniform(-b, b, lin.bias.shape)
                mods.append(lin)
                if i < len(sizes) - 2:
                    mods.append(mnn.ReLU())
            body, head = mnn.Sequential(*mods), None

        class FitNet(mnn.Module):
            def __init__(self):
                super().__init__()
                self.body = body
                if head is not None:
                    self.head = head

            def __call__(self, x):
                if recurrent:
                    out = self.body(x)
                    h = out[0] if isinstance(out, tuple) else out
                    return self.head(h[:, -1, :]).squeeze(-1)
                return self.body(x).squeeze(-1)

        self._loss = ((lambda m, x, y: mnn.losses.binary_cross_entropy(m(x), y, with_logits=True,
                                                                        reduction="mean"))
                      if task == "classification" else
                      (lambda m, x, y: mnn.losses.mse_loss(m(x), y, reduction="mean")))
        return FitNet()

    def _optimizer(self):
        try:
            return self.mopt.Adam(learning_rate=FIT_ADAM["lr"], betas=list(FIT_ADAM["betas"]), eps=FIT_ADAM["eps"],
                                  bias_correction=True)
        except TypeError as exc:
            raise RuntimeError("REFUSED: %s: this mlx's Adam has no bias_correction (ours and torch correct "
                               "the bias; a different update) (%r)" % (self.arm, exc))

    def call(self):
        np, mx, mnn = self.np, self.mx, self.mnn
        mx.random.seed(SEED)
        model = self._build()
        opt = self._optimizer()
        step = mnn.value_and_grad(model, self._loss)
        X, y = mx.array(self.Xtr), mx.array(self.ytr)
        n = int(X.shape[0])
        bs = max(1, min(FIT_BATCH, n))
        losses = []
        for _epoch in range(FIT_EPOCHS):
            perm = mx.random.permutation(n)
            for s in range(0, n, bs):
                idx = perm[s:s + bs]
                loss, grads = step(model, X[idx], y[idx])
                opt.update(model, grads)
                mx.eval(model.parameters(), opt.state, loss)
                losses.append(loss)
        self.losses = np.asarray(mx.stack(losses), dtype=np.float64)
        self.model, self._score = model, None
        if self.lane in FIT_PREDICT_IN_CLOCK:
            self._predict()               # sequential-fit: the held-out probabilities inside the clock

    def sync(self):
        self.mx.synchronize() if hasattr(self.mx, "synchronize") else None

    def _predict(self):
        np, mx = self.np, self.mx
        if self._score is None:
            parts = []
            for s in range(0, len(self.Xte), FIT_PREDICT_CHUNK):
                z = self.model(mx.array(self.Xte[s:s + FIT_PREDICT_CHUNK]))
                parts.append(np.asarray(mx.sigmoid(z) if self.task == "classification" else z, dtype=np.float32))
            self._score = np.ascontiguousarray(np.concatenate(parts))
        return self._score

    def outputs(self):
        return {"score": self._predict(), "loss_curve": self.losses}

    def digest(self):
        return _sha(self.np.ascontiguousarray(self._predict()).data)[:16]


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
# FAIRNESS FIX (2026-10-13 audit): library opponents in place of harness-written twins
# ---------------------------------------------------------------------------
# The LM and transformer torch twins were code this harness wrote from torch ops
# (tools/torch_lm_step_opponent.py ByteDecoder, tools/speed_torch_seq.py LlamaEager), and the
# Mamba twins pure-PyTorch references: neither is a library's own implementation. They are
# replaced by transformers' own Llama classes (LlamaForCausalLM, LlamaDecoderLayer,
# LlamaRotaryEmbedding, DynamicCache) configured to ours' exact architecture, and mamba_ssm's own
# fused blocks, every one loaded with ours' initial weights and read back bit for bit. The torch
# settings (eager/compile x fp32/tf32/bf16 autocast) are unchanged. transformers is pinned in
# tools/bench_board.py NEURAL_PINS (TRANSFORMERS_PIN); mamba_ssm in MAMBA_SSM_BUILD. NONE OF THIS
# HAS RUN: the transformers 5.17.0 constructor/forward signatures used below (and mamba_ssm's
# ROCm build, Mamba3 backward and autocast/compile behaviour inside the Samba stack) are
# unverified at runtime; every mismatch refuses the arm by name instead of falling back.
# Results raced before this change were wrong for these lanes; apply extra scrutiny.
TRANSFORMERS_PIN = "transformers==5.17.0"

#: The mamba_ssm modules each model runs, and our weight names -> the Block's state-dict names
#: (block.norm = the block RMSNorm, block.mixer = the mixer).
MAMBA_SSM_MIXER = {"mamba1": "mamba_ssm.modules.mamba_simple.Mamba",
                   "mamba2": "mamba_ssm.modules.mamba2.Mamba2",
                   "mamba3": "mamba_ssm.modules.mamba3.Mamba3"}
MAMBA_SSM_BLOCK_NORM = {"mamba1": "norm.weight", "mamba2": "block_norm.weight",
                        "mamba3": "block_norm.weight"}


def _floats(v):
    return None if v is None else [float(x) for x in v]


def ours_ssm_record(model, mod, block):
    """Ours' Mamba structural constants under the canonical ssm_* names: the profile constants
    of the installed mojolearn._mamba_impl and the block's own attributes."""
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
    """The mixer's constructor arguments (every one EXPLICIT, never the library default) and the
    block norm's eps, from mamba/corpus/gen_corpus.py: the constants our blocks use."""
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
    """The constructed Block's structural values under the canonical ssm_* names (compared with
    ours' by tools/bench_board_params.py) and the norms' eps (checked against the corpus)."""
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
    """(version text, pip's direct_url record) of the installed mamba_ssm: a git build's version
    carries the commit (the board pins one)."""
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


def _mamba_ssm_fused_path(model, mixer_mod, arm):
    """THE FUSED PATH OR NOTHING: an arm that would leave mamba_ssm's fused path (no
    causal_conv1d) refuses by name instead."""
    if model == "mamba3":
        return "mamba3_siso_combined (Triton)"
    if getattr(mixer_mod, "causal_conv1d_fn", None) is None:
        raise RuntimeError("REFUSED: %s: causal_conv1d does not import, so mamba_ssm would leave "
                           "its fused path; install the board's pinned causal-conv1d" % arm)
    if model == "mamba1":
        return "mamba_inner_fn (causal_conv1d + selective_scan kernel, fused)"
    return "mamba_split_conv1d_scan_combined (causal_conv1d + SSD Triton kernels, gated RMSNorm)"


def mamba_ssm_block(torch, dev, model, corpus, dm, weights, arm):
    """mamba_ssm's own Block(norm = Triton RMSNorm, mixer, fused_add_norm, residual in fp32)
    holding ours' float32 weights `weights` (our names), read back bit for bit. Returns (block,
    structural read-back, eps read-back, fused path, eps). Raises REFUSED when mamba_ssm does
    not import on this box (never a reference fallback)."""
    import numpy as np
    try:
        from functools import partial
        from mamba_ssm.modules.block import Block
        from mamba_ssm.ops.triton.layer_norm import RMSNorm
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError("REFUSED: %s: mamba_ssm does not import in %s (%r); there is no library "
                           "Mamba kernel on this box and the pure-PyTorch reference is not an "
                           "opponent" % (arm, sys.executable, exc))
    cfg, eps = mamba_ssm_config(model, corpus, dm)
    mod_name, cls_name = MAMBA_SSM_MIXER[model].rsplit(".", 1)
    mixer_mod = importlib.import_module(mod_name)
    mixer_cls = getattr(mixer_mod, cls_name)
    fused = _mamba_ssm_fused_path(model, mixer_mod, arm)
    f32 = torch.float32
    block = Block(dm, partial(mixer_cls, device=dev, dtype=f32, **cfg), torch.nn.Identity,
                  norm_cls=partial(RMSNorm, eps=eps, device=dev, dtype=f32),
                  fused_add_norm=True, residual_in_fp32=True).to(dev)
    sd = {}
    for k, v in weights.items():
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
            raise RuntimeError("REFUSED: %s: %s is %r, ours uses %r" % (arm, k, v, eps))
    return block, readback, eps_back, fused, cfg, eps


def _mamba_ssm_info(torch, arm, setting):
    import mamba_ssm
    version, direct_url = _mamba_ssm_source()
    try:
        import triton
        triton_ver = triton.__version__
        f32_default = getattr(getattr(getattr(triton, "knobs", None), "language", None),
                              "fp32_default", None)
    except Exception:  # noqa: BLE001
        triton_ver, f32_default = None, None
    if f32_default is not None and setting is not None and f32_default != setting["triton_f32_default"]:
        raise RuntimeError("REFUSED: %s: Triton's fp32 default reads back %r, not %r"
                           % (arm, f32_default, setting["triton_f32_default"]))
    return {"mamba_ssm_version": version, "mamba_ssm_direct_url": direct_url,
            "mamba_ssm_file": getattr(mamba_ssm, "__file__", None), "triton_version": triton_ver,
            "triton_f32_default": f32_default if f32_default is not None
            else os.environ.get("TRITON_F32_DEFAULT"),
            "causal_conv1d_version": _dist_version("causal_conv1d") or _dist_version("causal-conv1d"),
            "torch_version_hip": getattr(torch.version, "hip", None)}


class MambaSsmArm:
    """mamba_ssm's fused kernels on one Mamba forward lane: Block(norm = Triton RMSNorm, mixer,
    fused_add_norm, residual in fp32); the output is mixer(norm(x)) + x, the block's residual
    add. The clock: host x to the device, the block, the output back on the host, synchronized
    (the torch arms' clock). FAIRNESS FIX (2026-10-13 audit): brought onto main from
    lane/mamba-opp facde18cc (NVIDIA) and opened to AMD's ROCm/HIP build (float32 only there;
    unverified at runtime). Results raced before this change were wrong for this lane; apply
    extra scrutiny."""

    def __init__(self, lane, shape, data, arm):
        if lane not in MAMBA_SSM_LANES:
            raise RuntimeError("REFUSED: %s races the Mamba forward lanes only (%s), not %s"
                               % (arm, ", ".join(MAMBA_SSM_LANES), lane))
        import numpy as np
        self.np, self.lane = np, lane
        setting = mamba_ssm_setting(arm)
        # the Triton fp32 dot precision, set before any kernel compiles
        os.environ["TRITON_F32_DEFAULT"] = setting["triton_f32_default"]
        torch, dev, kind, name, sync = _torch_device()
        hip = getattr(torch.version, "hip", None)
        if kind != "cuda":
            raise RuntimeError("REFUSED: %s: mamba_ssm's kernels run on CUDA or ROCm/HIP only; "
                               "this torch is %s" % (arm, kind))
        if hip and setting["tf32"]:
            raise RuntimeError("REFUSED: %s: TF32 is an NVIDIA CUDA mode; on ROCm mamba_ssm races "
                               "mamba-ssm-fp32 only" % arm)
        self.torch, self.dev, self._sync = torch, dev, sync
        twin = _load("torch_lm_step_opponent")
        precision = twin.set_precision(torch, setting["tf32"])
        torch.manual_seed(SEED)
        info = {}
        corpus = _load_mamba_corpus_restoring(torch, info)
        model = MODEL_OF[lane]
        d = _dims_of(lane, shape)
        block, readback, eps_back, fused, cfg, eps = mamba_ssm_block(
            torch, dev, model, corpus, d["d_model"], _weights(data), arm)
        block.eval()
        self.block = block
        self.host = [torch.from_numpy(np.ascontiguousarray(data["x"]))]
        self.info = dict(info)
        self.info.update(_mamba_ssm_info(torch, arm, setting))
        self.info.update({
            "library": "mamba-ssm", "version": self.info["mamba_ssm_version"], "device": "gpu",
            "device_name": name, "torch_version": torch.__version__,
            "torch_version_cuda": torch.version.cuda, "torch_backend": "rocm" if hip else kind,
            "setting": arm[len("mamba-ssm-"):],
            "precision": ("float32 weights and activations; TF32 %s in torch; Triton fp32 tl.dot "
                          "input_precision %s" % ("ON" if setting["tf32"] else "off",
                                                  setting["triton_f32_default"])),
            "mode": "mamba_ssm fused kernels, eager, %s" % arm[len("mamba-ssm-"):],
            "precision_readback": precision, "mixer": MAMBA_SSM_MIXER[model],
            "mixer_config": json.dumps(cfg, sort_keys=True, default=str),
            "block": "mamba_ssm.modules.block.Block(norm=Triton RMSNorm eps %g, fused_add_norm=True, "
                     "residual_in_fp32=True, mlp=Identity); out = mixer(norm(x)) + x" % eps,
            "fused_path": fused, "norm_eps_readback": eps_back, "config_readback": readback,
            "weights": "ours, load_state_dict(strict=True), read back bit for bit",
            "compile": "eager (mamba_ssm's own kernels)", "pre_clock_fit": False,
            "input_home": "host",
            "call": "mamba_ssm Block(%s).forward(x) on %s; h + residual; .cpu()"
                    % (MAMBA_SSM_MIXER[model].rsplit(".", 1)[1], dev)})
        # preflight-smoke (2026-10-14): the library class constructed (mamba_ssm.modules.block.Block), so the
        # preflight's real-implementation check reads it back
        self.record = {"__library__": "mamba-ssm", "seed": SEED, "__implementation__": type(block)}
        self.record.update(readback)
        self.out = None
        sync()

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


#: ours' TransformerBlock / SambaStack attention names -> transformers' LlamaDecoderLayer names
HF_LAYER_NAME = {"input_layernorm.weight": "input_layernorm.weight",
                 "post_attention_layernorm.weight": "post_attention_layernorm.weight",
                 "q_proj.weight": "self_attn.q_proj.weight", "k_proj.weight": "self_attn.k_proj.weight",
                 "v_proj.weight": "self_attn.v_proj.weight", "o_proj.weight": "self_attn.o_proj.weight",
                 "gate_proj.weight": "mlp.gate_proj.weight", "up_proj.weight": "mlp.up_proj.weight",
                 "down_proj.weight": "mlp.down_proj.weight"}
#: tools/torch_lm_step_opponent.py registry names (the flat init's order) -> LlamaForCausalLM's
HF_LM_BLOCK_NAME = {"norm1_w": "input_layernorm.weight", "w_q": "self_attn.q_proj.weight",
                    "w_k": "self_attn.k_proj.weight", "w_v": "self_attn.v_proj.weight",
                    "w_o": "self_attn.o_proj.weight", "norm2_w": "post_attention_layernorm.weight",
                    "w_gate": "mlp.gate_proj.weight", "w_up": "mlp.up_proj.weight",
                    "w_down": "mlp.down_proj.weight"}


def hf_lm_state_name(name):
    """tools/torch_lm_step_opponent.py registry name -> LlamaForCausalLM state-dict name."""
    if name == "embed":
        return "model.embed_tokens.weight"
    if name == "lm_head":
        return "lm_head.weight"
    block, part = name.split(".", 1)
    return "model.layers.%d.%s" % (int(block[len("block"):]), HF_LM_BLOCK_NAME[part])


def _transformers(arm):
    try:
        import transformers
        from transformers import LlamaConfig
        from transformers.models.llama import modeling_llama
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError("REFUSED: %s: transformers does not import in %s (%r); the board pins %s "
                           "(tools/bench_board.py NEURAL_PINS) and the harness's own twin is not an "
                           "opponent" % (arm, sys.executable, exc, TRANSFORMERS_PIN))
    return transformers, LlamaConfig, modeling_llama


def hf_llama_config(arm, *, vocab, d_model, n_heads, n_kv, head_dim, intermediate, layers,
                    max_pos, eps, theta):
    """A LlamaConfig of ours' exact architecture, every field explicit (never a library
    default): SwiGLU MLP, no biases, RMSNorm eps, default RoPE at base `theta`, SDPA attention,
    no dropout, untied head (the LM sets tying itself)."""
    transformers, LlamaConfig, _ = _transformers(arm)
    kw = dict(vocab_size=int(vocab), hidden_size=int(d_model), intermediate_size=int(intermediate),
              num_hidden_layers=int(layers), num_attention_heads=int(n_heads),
              num_key_value_heads=int(n_kv), head_dim=int(head_dim), hidden_act="silu",
              max_position_embeddings=int(max_pos), rms_norm_eps=float(eps), rope_theta=float(theta),
              attention_bias=False, mlp_bias=False, attention_dropout=0.0,
              tie_word_embeddings=False, use_cache=False, initializer_range=0.02,
              attn_implementation="sdpa")
    cfg = LlamaConfig(**kw)
    if hasattr(cfg, "rope_parameters"):          # transformers' newer RoPE field
        try:
            cfg.rope_parameters = {"rope_type": "default", "rope_theta": float(theta)}
        except Exception:  # noqa: BLE001  (read back below through the rotary inv_freq)
            pass
    return transformers, cfg


def hf_check_rotary(torch, rotary, head_dim, theta, arm):
    """transformers' rotary inv_freq must be ours' default RoPE at base `theta` (float32), or the
    arm refuses: this is the read-back of the RoPE base whatever the config field is called."""
    import numpy as np
    inv = rotary.inv_freq.detach().float().cpu().numpy()
    want = (1.0 / (np.float32(theta) ** (np.arange(0, head_dim, 2, dtype=np.float32)
                                         / np.float32(head_dim)))).astype(np.float32)
    if inv.shape != want.shape or not np.allclose(inv, want, rtol=1e-6, atol=0):
        raise RuntimeError("REFUSED: %s: transformers' rotary inv_freq is not the default RoPE at "
                           "base %g (max rel diff %s)" % (arm, theta, float(np.max(np.abs(inv - want) / want))
                                                          if inv.shape == want.shape else inv.shape))
    scaling = float(getattr(rotary, "attention_scaling", 1.0))
    if scaling != 1.0:
        raise RuntimeError("REFUSED: %s: transformers' rotary attention_scaling is %r, not 1" % (arm, scaling))
    return float(theta)


def hf_load_exact(torch, module, state, arm):
    """load_state_dict(strict=True) of `state` ({hf name: tensor}) into `module`, read back bit for
    bit (float32)."""
    module.load_state_dict(state, strict=True)
    got = module.state_dict()
    for k, v in state.items():
        g = got[k].detach().cpu()
        if g.dtype != torch.float32 or tuple(g.shape) != tuple(v.shape) or not torch.equal(g, v.cpu()):
            raise RuntimeError("REFUSED: %s: transformers module did not load our float32 bits for %s"
                               % (arm, k))


def hf_layer_call(layer, x, cos, sin, position_ids, **extra):
    """LlamaDecoderLayer(x) with precomputed RoPE cos/sin; returns the hidden states whatever the
    installed version returns (a tensor, or a tuple whose first item is the hidden states)."""
    out = layer(x, attention_mask=None, position_ids=position_ids, position_embeddings=(cos, sin),
                **extra)
    return out[0] if isinstance(out, (tuple, list)) else out


def build_runner(lane, arm, shape, data, ab_config=None):
    operation = _neural_ab_operation(ab_config)
    supported = ("model_default", "transformer_configured_forward", "transformer_windowed_forward", "transformer_forward_vjp_tape", "mamba3_owned_forward", "lm_chunked_head_v2",
                 "residual_dropout_forward_vjp", "mlp_inference_sessions", "samba_training_config", "mamba_forward_vjp",
                 "program_eager")
    if operation not in supported:
        raise RuntimeError("REFUSED: no actual public workload implementation for neural A/B operation " + str(operation))
    if arm in OUR_ARMS:
        if os.environ.get("MOJOLEARN_NUMERIC_MODE", "identical").strip().lower() != OUR_MODE[arm]:
            raise RuntimeError("REFUSED: arm %s runs under MOJOLEARN_NUMERIC_MODE=%s, not %r"
                               % (arm, OUR_MODE[arm], os.environ.get("MOJOLEARN_NUMERIC_MODE")))
        model = MODEL_OF[lane]
        runtime = (ab_config or {}).get("runtime", {})
        if lane in GEN_LANES or lane in PREDICT_LANES:   # 2026-10-19 (lane ag-inference-board)
            if operation != "model_default":
                raise RuntimeError("REFUSED: the inference lanes have no experimental operation")
            return OursGenerate(lane, shape, data) if lane in GEN_LANES else OursPredict(lane, shape, data)
        if lane in FIT_LANES:
            if operation != "model_default":
                raise RuntimeError("REFUSED: the fit lanes have no experimental operation")
            return OursFit(lane, shape, data)
        if lane in OPS_LANES:
            if operation != "model_default":
                raise RuntimeError("REFUSED: ops-session-step has no experimental operation")
            return OursOpsStep(lane, shape, data)
        if lane in PROGRAM_LANES:
            if operation not in ("model_default", "program_eager"):
                raise RuntimeError("REFUSED: program-run's operations are model_default and program_eager")
            return OursProgramRun(lane, shape, data, eager=operation == "program_eager")
        if operation == "program_eager":
            raise RuntimeError("REFUSED: program_eager requires program-run")
        if operation == "residual_dropout_forward_vjp":
            return OursResidualDropout(lane, shape, data, runtime)
        if operation == "mlp_inference_sessions":
            return OursMLPSessions(lane, shape, data, runtime)
        if operation == "samba_training_config" and lane != "samba-train-step":
            raise RuntimeError("REFUSED: Samba A/B optimizer operation requires samba-train-step")
        if operation == "transformer_configured_forward" and (lane != "transformer-forward" or shape != "full" or not runtime.get("transformer_config")):
            raise RuntimeError("REFUSED: explicit transformer configuration requires its full transformer-forward recipe")
        if operation == "lm_chunked_head_v2" and (lane != "lm-train-step" or not runtime.get("chunked_lm_head_v2")):
            raise RuntimeError("REFUSED: lm_chunked_head_v2 requires lm-train-step and its config flag")
        if operation == "transformer_windowed_forward" and (lane != "transformer-forward" or int(runtime.get("transformer_window", 0)) <= 0):
            raise RuntimeError("REFUSED: windowed operation requires transformer-forward and a positive configured window")
        if operation == "transformer_forward_vjp_tape" and model != "transformer":
            raise RuntimeError("REFUSED: tape operation requires a transformer block")
        if operation == "mamba3_owned_forward" and model != "mamba3":
            raise RuntimeError("REFUSED: owned-weight operation requires Mamba3")
        if operation == "mamba_forward_vjp" and model not in ("mamba1", "mamba2", "mamba3"):
            raise RuntimeError("REFUSED: Mamba VJP operation requires a Mamba block")
        if model == "lm":
            return OursLM(lane, shape, data, ab_runtime=runtime)
        if model == "samba":
            return OursSamba(lane, shape, data, ab_runtime=runtime)
        if model in ("transformer", "mamba1", "mamba2", "mamba3"):
            return OursBlock(lane, shape, data, ab_runtime=runtime)
        return OURS[model](lane, shape, data)
    if operation != "model_default":
        raise RuntimeError("REFUSED: experimental operation has no matched opponent implementation")
    if lane in GEN_LANES or lane in PREDICT_LANES:   # 2026-10-19 (lane ag-inference-board): inference
        if arm in MLX_ARMS or arm in MAMBA_SSM_ARMS:
            raise RuntimeError("REFUSED: %s does not race %s (torch arms only)" % (arm, lane))
        return TorchGenerateArm(lane, shape, data, arm) if lane in GEN_LANES else TorchPredictArm(lane, shape, data, arm)
    if lane in FIT_LANES:                # 2026-10-19 (lane ag-bench): whole fits
        if arm in MLX_ARMS:
            return MlxFitArm(lane, shape, data, arm)
        return TorchFitArm(lane, shape, data, arm)
    if arm in MLX_ARMS:
        raise RuntimeError("REFUSED: %s races the fit lanes only" % arm)
    if lane in OPS_LANES:                # 2026-10-19 (lane ag-board-twins): one recorded step
        if arm in MAMBA_SSM_ARMS:
            raise RuntimeError("REFUSED: %s races the Mamba lanes only" % arm)
        return TorchOpsStepArm(lane, shape, data, arm)
    if lane in PROGRAM_LANES:            # 2026-10-19 (lane ag-program): one inference forward
        if arm in MAMBA_SSM_ARMS:
            raise RuntimeError("REFUSED: %s races the Mamba lanes only" % arm)
        return TorchProgramArm(lane, shape, data, arm)
    if arm in MAMBA_SSM_ARMS:            # FAIRNESS FIX (2026-10-13 audit): the real Mamba kernels
        return MambaSsmArm(lane, shape, data, arm)
    return TorchArm(lane, shape, data, arm)


def _prepare_worker(args):
    """Construct the arm and retain the preparation context used by round receipts."""
    try:
        ab_config = _read_neural_ab_config(getattr(args, "neural_ab_config", None))
        _apply_neural_ab_environment(ab_config)
        import numpy as np
        data = read_inputs(args.data)
        preparation_start = time.perf_counter()
        runner = build_runner(args.lane, args.arm, args.shape, data, ab_config=ab_config)
        runner.sync()
        preparation_ms = (time.perf_counter() - preparation_start) * 1000.0
    except (Exception, SystemExit) as exc:  # noqa: BLE001  (a twin's refuse() exits)
        import traceback
        traceback.print_exc()
        return None, {"event": "error", "stage": "ready", "error": repr(exc)[:2000]}, None
    if isinstance(runner.info, dict):   # the arm's own library version and GPU (the store's key)
        runner.info.update(_load("bench_board_probe").library_identity(runner.info))
    record = getattr(runner, "record", None)
    if record is not None:
        # Resolve callable identity BEFORE JSON transport; default=str would
        # erase the actual library function and leave an unusable repr.
        record = _load("bench_board_params").arm_record(record)
    return (runner, {"event": "ready", "info": runner.info, "pid": os.getpid(),
                     "params_record": record,
                     "neural_ab": _neural_ab_receipt(runner, ab_config)},
            {"ab_config": ab_config, "preparation_ms": preparation_ms})


def worker_ready(args):
    """The worker's ready stage, shared with the smoke's parameter-only probe.

    Return (runner, message), with runner None on preparation failure.
    """
    runner, message, _ = _prepare_worker(args)
    return runner, message


#: True only inside probe_ready (the smoke's parameter probe, in process or in its arm server): the arm is
#: built for its parameter read-back only and is never warmed up or timed (tools/smoke_inproc.py).
PARAMS_PROBE = False


def probe_ready(argv):
    """The smoke's in-process parameter probe (tools/smoke_inproc.py): a worker's argv, its ready message."""
    global PARAMS_PROBE
    previous, PARAMS_PROBE = PARAMS_PROBE, True
    try:
        return worker_ready(build_parser().parse_args(argv))[1]
    finally:
        PARAMS_PROBE = previous


def worker(args):
    import numpy as np
    # fd 1 is the protocol; everything a library prints goes to the log.
    proto = os.fdopen(os.dup(1), "w", buffering=1)
    os.dup2(2, 1)
    sys.stdout = sys.stderr

    def say(obj):
        proto.write(json.dumps(obj, sort_keys=True, default=str) + "\n")
        proto.flush()

    # The ready probe returns only its public protocol pair. Timed workers also
    # need its local configuration and preparation clock; neither is a global.
    runner, msg, preparation = _prepare_worker(args)
    say(msg)
    if runner is None:
        return 1
    ab_config = preparation["ab_config"]
    preparation_ms = preparation["preparation_ms"]
    # peak memory per round, reset and read OUTSIDE the clock
    mem = _load("bench_board_probe").MemProbe((runner.info or {}).get("device", "gpu"),
        library=(runner.info or {}).get("library") or "?", arm=args.arm)
    for line in sys.stdin:
        parts = line.split()
        if not parts:
            continue
        if parts[0] == "round":
            r = int(parts[1])
            try:
                mem.start()
                # COMPILE-IN-CLOCK (Andrew 2026-10-14): torch.compile counters around the scored
                # call, read outside its clock. No extra warm-up call here: a neural round consumes
                # the next input and advances the trained state, so the warm-up stays the one
                # identical round-0 step and a compile in a scored step is recorded and flagged.
                guard = _load("bench_board_compile_guard")
                c0 = guard.read()
                t0 = time.perf_counter()
                runner.call()
                runner.sync()
                ms = (time.perf_counter() - t0) * 1000.0
                compiles_now = guard.compiles(c0, guard.read())
                outputs = runner.outputs()
                runner.sync()
                operation_ms = (time.perf_counter() - t0) * 1000.0
                m = mem.stop()
                digest = runner.digest()
                # 2026-10-19 (lane ag-inference-board): an inference runner's own sub-clocks (prefill, generate)
                # and its UNTIMED memory pass, both read after the round's clock
                extra = runner.round_extra() if hasattr(runner, "round_extra") else None
                if extra and extra.get("clock_ms") is not None:
                    ms = float(extra["clock_ms"])     # the lane's clock: generate(prompt, N) alone
                device_memory = None
                if hasattr(runner, "memory_pass"):
                    try:
                        device_memory = runner.memory_pass()
                    except Exception as exc:  # noqa: BLE001  (memory never breaks a timing; recorded by name)
                        device_memory = {"error": "memory pass failed: %r" % (exc,)}
                from bench_board_state import scored_receipt
                state_receipt = scored_receipt(outputs, runner) if r > 0 else None
                round_message = {"event": "round", "round": r, "ms": ms, "digest": digest, "mem": m,
                    "extra": extra, "device_memory": device_memory,
                    "state_receipt": state_receipt, "compiles": compiles_now,
                    "neural_ab": _neural_ab_receipt(runner, ab_config, executed=True),
                    "operation": {"ms": operation_ms, "preparation_ms": preparation_ms if r == 0 else 0.0,
                                  "scope": (_neural_ab_operation(ab_config) if ab_config else "training-step-or-forward-consume"),
                                  "dataset_load_included": False,
                                  "cold_total_ms": preparation_ms + operation_ms if r == 0 else None}}
            except Exception as exc:  # noqa: BLE001
                import traceback
                traceback.print_exc()
                where = runner.info.get("torch_backend") or runner.info.get("device")
                say({"event": "error", "stage": "round %d" % r,
                     "error": "REFUSED: %s on %s failed in round %d%s: %s" % (
                         args.arm, where, r, " (compile happens here)"
                         if r == 0 and "compile" in args.arm else "", repr(exc)[:2000])})
                return 1
            say(round_message)
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


# Same fp32 forward tolerance used by the Mamba and transformer surface gates.
# This checks saved benchmark output, never a timed opponent or a GPU rerun.
FORWARD_REFERENCE_LANES = ("transformer-forward", "mamba1-forward", "mamba2-forward", "mamba3-forward",
                           # zero-state decode token by token is the same sequence as one
                           # fresh forward (tools/step_vs_full_check.py), so the decode rows
                           # carry the same untimed host-reference check
                           "transformer-decode", "mamba1-decode", "mamba2-decode",
                           "mamba3-decode")
FORWARD_REFERENCE_ATOL = 1e-6
FORWARD_REFERENCE_RTOL = 1e-5


def _forward_reference(lane, shape, data):
    """(host inference class, the reference forward of data["x"] as float64): the untimed own-host reference."""
    import numpy as np
    ml = _ours_module()
    weights = {k: np.ascontiguousarray(v) for k, v in _weights(data).items()}
    model = MODEL_OF[lane]
    kwargs = {"numeric_mode": "identical"}
    if model == "transformer":
        d = _dims_of(lane, shape)
        cls = ml.TransformerBlockInference
        kwargs.update(n_heads=d["n_heads"], n_kv_heads=d["n_kv"], head_dim=d["head_dim"],
                      norm_eps=BLOCK_NORM_EPS, rope_theta=BLOCK_ROPE_THETA)
    else:
        cls = getattr(ml, {"mamba1": "Mamba1BlockInference", "mamba2": "Mamba2BlockInference",
                           "mamba3": "Mamba3BlockInference"}[model])
    block = cls(weights, **kwargs)
    # The public inference classes dispatch only to the shipped Mojo CPU host
    # binding. Their fresh forward uses exactly the race's weights and inputs.
    return cls, np.asarray(block.forward(np.ascontiguousarray(data["x"])), dtype=np.float64)


def forward_host_quality(lane, shape, data, outs):
    import numpy as np
    own = {arm: out for arm, out in outs.items() if arm in ("ours", "ours-fast")}
    if not own:
        return {}
    cls, ref = _forward_reference(lane, shape, data)
    result = {}
    for arm, out in own.items():
        got = np.asarray(out["y"], dtype=np.float64)
        matching_shape = got.shape == ref.shape
        finite = bool(np.isfinite(got).all() and np.isfinite(ref).all())
        q = {"host_reference": cls.__name__, "host_reference_atol": FORWARD_REFERENCE_ATOL,
             "host_reference_rtol": FORWARD_REFERENCE_RTOL, "host_reference_finite": finite,
             "host_reference_passed": False}
        if not matching_shape:
            q["shape_mismatch_vs_host"] = "%s vs %s" % (list(got.shape), list(ref.shape))
        elif finite and ref.size:
            error = np.abs(got - ref)
            bound = FORWARD_REFERENCE_ATOL + FORWARD_REFERENCE_RTOL * np.abs(ref)
            q.update(max_abs_diff_vs_host=float(error.max()),
                     max_rel_diff_vs_host=float(error.max()) / (float(np.abs(ref).max()) or 1.0),
                     host_reference_tolerance_ratio=float((error / bound).max()),
                     host_reference_passed=bool(np.all(error <= bound)))
        result[arm] = q
    return result


# SCORER GAPS (lane quality-scorers, 2026-10-15). The forward lanes' quality columns were each arm's
# distance to OURS (max_abs_diff_vs_ours), which is only recorded when ours and the opponent share one
# race, and the board races ours alone: transformer-forward and mamba1/2/3-forward had no metric on
# ours or on the torch arms. Every arm is now also measured against the eager FP32 torch twin's
# output on the same weights and inputs (rel_fro_vs_torch_eager_fp32, declared in
# tools/board_quality.py; the stored torch-eager-fp32 cell reads 0 there). On the Mamba lanes that
# twin is the pure-PyTorch reference implementation: refused as an OPPONENT (REFERENCE_TWIN_LANES),
# it is exactly what a quality reference should be.
TORCH_REFERENCE_ARM = "torch-eager-fp32"
#: set only in the untimed reference worker's environment: lets the pure-PyTorch Mamba twin run as the reference
QUALITY_REFERENCE_ENV = "MOJOLEARN_BOARD_QUALITY_REFERENCE"
TORCH_REFERENCE_LANES = ("transformer-forward", "mamba1-forward", "mamba2-forward", "mamba3-forward")


def torch_reference_quality(outs, reference):
    """{arm: {max_rel_diff_vs_torch_eager_fp32, rel_fro_vs_torch_eager_fp32}} for every arm whose y
    matches the reference's shape (the same two metrics as tools/bench_board_algos.py's layer lanes)."""
    import numpy as np
    r = np.asarray(reference["y"], dtype=np.float64)
    if not np.isfinite(r).all():
        raise ValueError("nonfinite torch reference output")
    rn = float(np.linalg.norm(r)) or 1e-30
    den = np.maximum(np.abs(r), 1e-6)
    q = {}
    for arm, o in outs.items():
        if "y" not in o:
            continue
        y = np.asarray(o["y"], dtype=np.float64)
        if y.shape != r.shape:
            q[arm] = {"shape_mismatch_vs_torch_eager_fp32": "%s vs %s" % (list(y.shape), list(r.shape))}
            continue
        if not np.isfinite(y).all():
            q[arm] = {"nonfinite_output": True}
            continue
        q[arm] = {"max_rel_diff_vs_torch_eager_fp32": float(np.max(np.abs(y - r) / den)),
                  "rel_fro_vs_torch_eager_fp32": float(np.linalg.norm(y - r)) / rn}
    return q


def torch_reference_outputs(args, lane, shape, data_path, tag):
    """Run the eager FP32 torch twin UNTIMED, after every timed worker has stopped, when a race of ours
    holds no such arm: one worker, the same protocol and inputs, one forward (round 0, whose time is
    discarded), its saved output. Returns (receipt, outputs or None); never raises."""
    import numpy as np
    arm = TORCH_REFERENCE_ARM
    receipt = {"role": "untimed independent quality reference; never a timed race arm", "arm": arm,
               "lane": lane}
    w = None
    try:
        ctd = _load("classical_two_datasets")
        cmd = shlex.split(args.theirs_python) + [os.path.abspath(__file__), "worker", "--arm", arm,
                                                 "--lane", lane, "--shape", shape, "--data", data_path]
        env = _worker_env(arm)
        env[QUALITY_REFERENCE_ENV] = "1"
        w = ctd.start_worker(arm, cmd, env,
                             os.path.join(args.out, "%s-%s-reference.log" % (tag, arm)), REPO)
        msg = _worker_message(w, args.ready_seconds, "reference ready")
        if msg.get("event") != "ready":
            raise RuntimeError("reference worker not ready: %s" % json.dumps(msg, default=str)[:400])
        receipt["info"] = msg.get("info")
        msg = _worker_message(w, warmup_deadline(arm, args), "reference forward", "round 0")
        if msg.get("event") != "round":
            raise RuntimeError("reference forward failed: %s" % json.dumps(msg, default=str)[:400])
        path = os.path.join(args.out, "%s-%s.reference.npz" % (tag, arm))
        msg = _worker_message(w, args.round_seconds, "reference save", "save %s" % path)
        if msg.get("event") != "saved":
            raise RuntimeError("reference save failed: %s" % json.dumps(msg, default=str)[:400])
        with np.load(path) as z:
            out = {k: z[k] for k in z.files}
        with open(path, "rb") as fh:
            receipt.update(status="ok", output=path, output_sha256=hashlib.sha256(fh.read()).hexdigest())
        return receipt, out
    except Exception as exc:  # noqa: BLE001  (no torch, no device, a refused twin)
        receipt.update(status="failed", error=repr(exc)[:600])
        return receipt, None
    finally:
        if w is not None:
            try:
                w.close()
            except Exception:  # noqa: BLE001
                pass


SCORER_REFUSED = "quality scorer unavailable: "
#: the input arrays each scorer reads (quality() below)
_NLL_LANES = ("lm-forward", "lm-infer", "samba-forward", "samba-infer", "samba-decode")


class ScorerUnavailable(RuntimeError):
    """This race's quality scorer cannot run: the race is refused before any timed work."""


def scorer_preflight(lane, shape, data_path, arms, operation):
    """THE SCORER PREFLIGHT (lane harness-evidence, 2026-10-14), before any worker starts: the input arrays the
    scorer reads are in the race's input file, and the own-host forward reference (FORWARD_REFERENCE_LANES, the
    scorer of ours' forward) imports and runs on a tiny slice (the first sequence of the race's own x and weights).
    Returns its receipt; raises ScorerUnavailable naming the failed check."""
    import numpy as np
    receipt = {"lane": lane, "shape": shape, "checks": []}
    if operation != "model_default":
        receipt["status"] = "not-applicable (experimental operation: its quality is pending by design)"
        return receipt
    with np.load(data_path) as z:
        keys = set(z.files)
        need = []
        if lane in FORWARD_REFERENCE_LANES:
            need.append("x")
        if lane in _NLL_LANES:
            need.append("batches")
        if MODEL_OF.get(lane) == "gemm":
            need += ["a", "b"]
        if lane in FIT_LANES:
            need += ["X_test", "y_test"]
        if lane in OPS_LANES:
            need += ["X", "y", "y_codes"]
        if lane in PROGRAM_LANES:
            need += ["X"] + ["w:" + n for n in OPS_PARAMS]
        if lane in GEN_LANES:
            need += ["prompt_ids", "new_tokens", "checkpoint_dir"]
        absent = [k for k in need if k not in keys]
        receipt["checks"].append({"check": "reference data", "need": need, "absent": absent,
                                  "status": "failed" if absent else "ok"})
        if absent:
            raise ScorerUnavailable("reference data: the race inputs hold no %s" % ", ".join(absent))
    if lane in FORWARD_REFERENCE_LANES and any(a in OUR_ARMS for a in arms):
        # in a child process, so this conductor loads no mojolearn binding while the workers are timed
        import subprocess
        check = {"check": "own host forward reference (one sequence)"}
        try:
            if os.environ.get("MOJOLEARN_SMOKE_IN_PROCESS") == str(os.getpid()):
                # the campaign smoke's worker process already holds every binding and times nothing, so the
                # child's reason (no binding in a conductor that times workers) does not apply: the same
                # check, as a function call
                got = scorer_preflight_reference(lane, shape, data_path)
                if got.get("status") != "ok":
                    raise RuntimeError(got.get("error") or "no reference")
            else:
                done = subprocess.run([sys.executable, os.path.abspath(__file__), "scorer-preflight-child", lane,
                                       shape, data_path], capture_output=True, text=True, timeout=1800)
                last = (done.stdout.strip().splitlines() or ["{}"])[-1]
                got = json.loads(last) if last.startswith("{") else {}
                if done.returncode or got.get("status") != "ok":
                    tail = (done.stderr.strip().splitlines() or ["no output"])[-1]
                    raise RuntimeError(got.get("error") or tail[:600])
            check.update(status="ok", reference=got.get("reference"), shape=got.get("shape"))
        except Exception as exc:  # noqa: BLE001
            check.update(status="failed", detail=str(exc)[:600])
            receipt["checks"].append(check)
            raise ScorerUnavailable("own host forward reference: %s" % str(exc)[:600])
        receipt["checks"].append(check)
    receipt["status"] = "ok"
    return receipt


def scorer_preflight_reference(lane, shape, data_path):
    """The forward reference on the first sequence of the race's own inputs: {"status": "ok", "reference",
    "shape"} or {"status": "failed", "error"}."""
    import numpy as np
    try:
        with np.load(data_path) as z:
            tiny = {k: z[k] for k in z.files if k.startswith("w:")}
            tiny["x"] = np.ascontiguousarray(z["x"][:1])
        cls, ref = _forward_reference(lane, shape, tiny)
        if not ref.size or not np.isfinite(ref).all():
            raise ValueError("the host reference forward of one sequence is empty or nonfinite")
        return {"status": "ok", "reference": cls.__name__, "shape": list(ref.shape)}
    except Exception as exc:  # noqa: BLE001
        return {"status": "failed", "error": repr(exc)[:600]}


def scorer_preflight_child(lane, shape, data_path):
    """scorer_preflight_reference in a child process; one JSON line on stdout."""
    got = scorer_preflight_reference(lane, shape, data_path)
    print(json.dumps(got), flush=True)
    return 0 if got.get("status") == "ok" else 1


def mark_forward_quality_failures(result):
    """A missing/failed host reference cannot leave a timed own arm marked ok."""
    if result["lane"] not in FORWARD_REFERENCE_LANES:
        return
    for arm in ("ours", "ours-fast"):
        row = result["arms"].get(arm)
        if row and row.get("status") == "ok" and not result.get("quality", {}).get(arm, {}).get("host_reference_passed"):
            row.update(status="quality_failed", error={"host_reference": result.get("quality", {}).get(arm,
                       result.get("quality", {}).get("error", "missing saved output or host reference"))})


def fit_quality(lane, data, outs):
    """The fit lanes' quality (2026-10-19, lane ag-bench), float64, on the held-out rows, with
    tools/speed_gbdt_arm.py's metric helpers: accuracy (score > 0.5), log_loss and auc for the
    classifiers (the score is P(y = 1)); rmse and the auc of the score for the regressors (the target
    is the 0/1 label). Every other arm also gets its difference from ours' value."""
    import numpy as np
    M = _load("speed_gbdt_arm")
    y = np.asarray(data["y_test"], dtype=np.float64).ravel()
    q = {}
    for arm, o in outs.items():
        row = q[arm] = {"held_out_rows": int(y.size)}
        score = np.asarray(o.get("score"), dtype=np.float64).ravel()
        if score.shape != y.shape:
            row["shape_mismatch"] = "%s vs %s" % (list(score.shape), list(y.shape))
            continue
        if not np.isfinite(score).all():
            row["nonfinite_scores"] = int((~np.isfinite(score)).sum())
            continue
        if FIT_TASK[lane] == "classification":
            row.update(accuracy=M.accuracy(y > 0.5, score > 0.5), log_loss=M.logloss(y, score),
                       auc=M.auc(y, score))
        else:
            row.update(rmse=M.rmse(y, score), auc=M.auc(y, score))
        curve = np.asarray(o.get("loss_curve") if o.get("loss_curve") is not None else [], dtype=np.float64).ravel()
        if curve.size:
            row.update(train_loss_last_step=float(curve[-1]), loss_curve_length=int(curve.size),
                       train_loss_finite=bool(np.isfinite(curve).all()))
    ref = q.get("ours") or q.get("ours-fast")
    for arm, row in q.items():
        if ref is None or row is ref:
            continue
        for k in ("accuracy", "log_loss", "auc", "rmse"):
            if k in row and k in ref:
                row[k + "_diff_vs_ours"] = row[k] - ref[k]
    return q


def ops_quality(outs):
    """ops-session-step's quality (2026-10-19, lane ag-board-twins), float64: every arm's loss and gradient
    digests (sha256 of each gradient's float32 bytes), and every other arm's loss difference and max abs / rel
    difference of each gradient and updated parameter from ours'."""
    import numpy as np
    q = {}
    for arm, o in outs.items():
        loss = float(np.asarray(o["loss"], dtype=np.float64).ravel()[0])
        q[arm] = {"loss": loss, "loss_finite": bool(np.isfinite(loss)),
                  "grads_finite": bool(all(np.isfinite(o["g:" + n]).all() for n in OPS_PARAMS)),
                  "grad_digests": {n: _sha(np.ascontiguousarray(o["g:" + n], dtype=np.float32).tobytes())[:16]
                                   for n in OPS_PARAMS}}
    ref_arm = "ours" if "ours" in outs else ("ours-fast" if "ours-fast" in outs else None)
    if ref_arm is None:
        return q
    ref = outs[ref_arm]
    for arm, o in outs.items():
        if arm == ref_arm:
            continue
        row = q[arm]
        row["loss_abs_diff_vs_ours"] = abs(row["loss"] - q[ref_arm]["loss"])
        row["grad_digests_equal_ours"] = row["grad_digests"] == q[ref_arm]["grad_digests"]
        for n in OPS_PARAMS:
            for key, pre in (("grad", "g:"), ("param", "p:")):
                a, b = np.asarray(o[pre + n], dtype=np.float64), np.asarray(ref[pre + n], dtype=np.float64)
                if a.shape != b.shape:
                    row["%s_shape_mismatch_%s" % (key, n)] = "%s vs %s" % (list(a.shape), list(b.shape))
                    continue
                diff = float(np.abs(a - b).max())
                row["%s_max_abs_diff_vs_ours_%s" % (key, n)] = diff
                row["%s_max_rel_diff_vs_ours_%s" % (key, n)] = diff / (float(np.abs(b).max()) or 1.0)
    return q


def program_quality(data, outs):
    """program-run's quality (2026-10-19, lane ag-program), float64: every arm's max abs error of its
    probabilities from a float64 forward of the same weights and batch, its output digest, and every other
    arm's max abs / rel difference from ours'."""
    import numpy as np
    x = np.asarray(data["X"], dtype=np.float64)
    w1, b1, w2, b2 = (np.asarray(data["w:" + n], dtype=np.float64) for n in OPS_PARAMS)
    z = np.maximum(x @ w1 + b1, 0.0) @ w2 + b2
    z = np.exp(z - z.max(axis=1, keepdims=True))
    ref64 = z / z.sum(axis=1, keepdims=True)
    q = {}
    for arm, o in outs.items():
        y = np.asarray(o["y"], dtype=np.float64)
        row = {"output_digest": _sha(np.ascontiguousarray(o["y"], dtype=np.float32).tobytes())[:16],
               "finite": bool(np.isfinite(y).all())}
        if y.shape == ref64.shape:
            row["max_abs_err_vs_fp64"] = float(np.abs(y - ref64).max())
        else:
            row["shape_mismatch_vs_fp64"] = "%s vs %s" % (list(y.shape), list(ref64.shape))
        q[arm] = row
    ref_arm = "ours" if "ours" in outs else ("ours-fast" if "ours-fast" in outs else None)
    if ref_arm is None:
        return q
    r = np.asarray(outs[ref_arm]["y"], dtype=np.float64)
    for arm, o in outs.items():
        if arm == ref_arm:
            continue
        y = np.asarray(o["y"], dtype=np.float64)
        if y.shape != r.shape:
            q[arm]["shape_mismatch_vs_ours"] = "%s vs %s" % (list(y.shape), list(r.shape))
            continue
        diff = float(np.abs(y - r).max())
        q[arm]["max_abs_diff_vs_ours"] = diff
        q[arm]["max_rel_diff_vs_ours"] = diff / (float(np.abs(r).max()) or 1.0)
        q[arm]["digest_equal_ours"] = q[arm]["output_digest"] == q[ref_arm]["output_digest"]
def gen_quality(data, outs):
    """causal-lm-generate's quality (2026-10-19, lane ag-inference-board): every arm's new-ids digest, and every
    other arm's agreement with ours: tokens_equal_ours (all new ids), first_divergence_index (the first new-token
    position where any row differs, None when equal), matching_prefix_tokens, prefill_token_equal_ours (the one
    token of generate(prompt, 1))."""
    import numpy as np
    L = int(np.asarray(data["prompt_ids"]).shape[1])
    q = {}
    for arm, o in outs.items():
        ids = np.asarray(o["ids"])
        q[arm] = {"new_tokens": int(ids.shape[1] - L), "new_ids_digest": _sha(np.ascontiguousarray(
            ids[:, L:], dtype=np.int32).tobytes())[:16],
            "prompt_echoed": bool(np.array_equal(ids[:, :L], np.asarray(data["prompt_ids"])))}
    ref_arm = "ours" if "ours" in outs else ("ours-fast" if "ours-fast" in outs else None)
    if ref_arm is None:
        return q
    ref, ref_first = np.asarray(outs[ref_arm]["ids"])[:, L:], np.asarray(outs[ref_arm]["first"])[:, L:]
    for arm, o in outs.items():
        if arm == ref_arm:
            continue
        new, first = np.asarray(o["ids"])[:, L:], np.asarray(o["first"])[:, L:]
        row = q[arm]
        if new.shape != ref.shape:
            row["shape_mismatch_vs_ours"] = "%s vs %s" % (list(new.shape), list(ref.shape))
            continue
        cols = np.nonzero((new != ref).any(axis=0))[0]
        row["tokens_equal_ours"] = bool(cols.size == 0)
        row["first_divergence_index"] = None if cols.size == 0 else int(cols[0])
        row["matching_prefix_tokens"] = int(new.shape[1] if cols.size == 0 else cols[0])
        row["prefill_token_equal_ours"] = bool(first.shape == ref_first.shape and np.array_equal(first, ref_first))
    return q


def quality(lane, data, outs, *, shape="full"):
    import numpy as np
    q = {}
    if lane in GEN_LANES:
        return gen_quality(data, outs)
    if lane in FIT_LANES:
        return fit_quality(lane, data, outs)
    if lane in OPS_LANES:
        return ops_quality(outs)
    if lane in PROGRAM_LANES:
        return program_quality(data, outs)
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
    if lane in FORWARD_REFERENCE_LANES:
        q.update(forward_host_quality(lane, shape, data, outs))
    if lane in ("lm-forward", "lm-infer", "samba-forward", "samba-infer", "samba-decode"):
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
    if lane in TORCH_REFERENCE_LANES and TORCH_REFERENCE_ARM in outs:
        for arm, extra in torch_reference_quality(outs, outs[TORCH_REFERENCE_ARM]).items():
            if arm != TORCH_REFERENCE_ARM:        # its own distance is 0 by definition
                q[arm].update(extra)
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
    ctd.apply_cpu_quota(env)
    if arm in OUR_ARMS:
        env["MOJOLEARN_NUMERIC_MODE"] = OUR_MODE[arm]     # ours identical, ours-fast fast
        if os.environ.get("MOJOLEARN_BENCH_INSTALLED", "0").strip() in ("", "0"):
            tree = os.path.join(REPO, "python")
            env["PYTHONPATH"] = tree + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    return env


SPAN = {"input_home": "host", "pre_clock_fit": False,
        "inside_clock": "host inputs to the device, the call, the result back on the host, synchronized"}
APPLE_FIT_IDENTICAL_REFUSED = ("refused: identical is never timed on Apple (CLAUDE.md: Apple is for same-bits "
                               "checks only); the fit lanes race ours-fast on Apple")
#: the fit lanes' comparability span (2026-10-19, lane ag-bench), the same on every arm
FIT_SPAN = {"input_home": "host", "pre_clock_fit": False, "upload_ms_untimed": None,
            "inside_clock": "one whole fit: a fresh seeded initialization, the training rows to the device, "
                            "the shuffle, every minibatch step (forward, loss, backward, Adam), the loss curve "
                            "back on the host, synchronized; the held-out prediction is after the clock"}
#: sequential-fit's span (2026-10-19, lane ag-board-twins): the fit AND the held-out predict_proba
FIT_PREDICT_SPAN = dict(FIT_SPAN, inside_clock=(
    "one whole fit and the held-out prediction: a fresh seeded initialization, the training rows to the device, "
    "the shuffle, every minibatch step (forward, loss, backward, Adam), the loss curve back on the host, then the "
    "held-out rows to the device and their class probabilities back on the host, synchronized"))
#: ops-session-step's span (2026-10-19, lane ag-board-twins), the same on every arm
OPS_SPAN = {"input_home": "host", "pre_clock_fit": False, "upload_ms_untimed": None,
            "inside_clock": "one training step: the batch, labels and parameters from the host to the device, "
                            "forward, softmax cross-entropy, backward, the loss and every gradient back on the "
                            "host, the SGD update on the device, every updated parameter back on the host, "
                            "synchronized"}
#: program-run's span (2026-10-19, lane ag-program), the same on every arm: the weights are resident on the
#: device before the clock on both sides (ours' as the program's parameters)
PROGRAM_SPAN = {"input_home": "host", "pre_clock_fit": False, "upload_ms_untimed": None,
                "inside_clock": "one inference forward: the batch from the host to the device, linear, relu, "
                                "linear, softmax, the probabilities back on the host, synchronized; the weights "
                                "on the device before the clock"}


#: sequential-predict's span (2026-10-19, lane ag-inference-board): the fit is untimed, the prediction alone timed
PREDICT_SPAN = {"input_home": "host", "pre_clock_fit": True, "upload_ms_untimed": None,
                "inside_clock": "the held-out rows from the host to the device, the forward, the class probabilities "
                                "back on the host, synchronized; the fit (the sequential-fit recipe) is the untimed "
                                "warm-up round"}
#: causal-lm-generate's span (2026-10-19, lane ag-inference-board), the same on every arm
GEN_SPAN = {"input_home": "host", "pre_clock_fit": False, "upload_ms_untimed": None,
            "weights_resident_before_clock": True,
            "inside_clock": "generate(prompt, N): the prompt ids from the host to the device, the prefill, N - 1 "
                            "greedy decode steps over the KV cache, the N new ids back on the host, synchronized; "
                            "prefill_ms is a separate generate(prompt, 1) timed the same way just before; the "
                            "checkpoint load and weight upload are the untimed warm-up round"}
#: the torch generate arms' precision (bf16 arms hold the weights in bfloat16, how torch serves in bf16)
GEN_PRECISION = {"eager-fp32": "float32 weights and math, TF32 off",
                 "eager-tf32": "float32 weights, matmuls in TF32",
                 "eager-bf16": "bfloat16 weights and activations (model.to(torch.bfloat16))"}
MEMORY_RULE = ("one untimed memory pass after every round (the same call): resident_bytes = device bytes added by "
               "the model's load (or fit), read before and after it in the warm-up round; peak_bytes = "
               "resident_bytes + the pass's transient peak (sampled every ms in process: KFD per-process VRAM on "
               "AMD, NVML per-process used bytes or the device's used bytes on NVIDIA; a sampled peak can miss a "
               "spike shorter than the interval); torch arms also torch.cuda.max_memory_allocated over the pass; "
               "Apple has no device counter (unified memory: the round's peak_host_mb)")


def fit_span(lane):
    """The comparability span of a fit lane."""
    if lane in PREDICT_LANES:
        return dict(PREDICT_SPAN)
    return dict(FIT_PREDICT_SPAN if lane in FIT_PREDICT_IN_CLOCK else FIT_SPAN)


def _worker_message(worker, seconds, stage, command=None):
    """Keep protocol EOF/exit distinct from a deadline, including child diagnostics."""
    send_error = None
    if command is not None:
        try:
            worker.send(command)
        except OSError as exc:
            send_error = repr(exc)
    # A broken pipe can follow a final error message: drain it before classifying
    # the process failure. The shared reader retains its existing None interface.
    message = worker.read(min(seconds, 0.2) if send_error else seconds)
    if message is not None:
        return message
    failure = dict(getattr(worker, "read_failure", None) or {"reason": "unknown"})
    proc = getattr(worker, "proc", None)
    returncode = proc.poll() if proc is not None else None
    if proc is not None and returncode is None and (failure["reason"] == "eof" or send_error):
        # A closed pipe can reach the reader just before waitpid observes exit.
        # Bound this reap grace; an alive child that closed stdout is still EOF,
        # never a claim that the operation consumed its full allowed time.
        try:
            returncode = proc.wait(timeout=0.2)
        except subprocess.TimeoutExpired:
            pass
    reason = ("worker_exited" if returncode is not None else "worker_io_error" if send_error
              else "timeout" if failure["reason"] == "deadline" else "worker_eof"
              if failure["reason"] == "eof" else "worker_protocol_failure")
    signal_number = -returncode if returncode is not None and returncode < 0 else None
    try:
        signal_name = signal.Signals(signal_number).name if signal_number else None
    except ValueError:
        signal_name = "signal_%d" % signal_number  # e.g. unnamed POSIX realtime signals
    return {"event": "error", "stage": stage, "reason": reason,
            "error": "worker returned no protocol message: " + reason,
            "returncode": returncode, "signal": signal_number,
            "signal_name": signal_name,
            "read_failure": failure, "send_error": send_error,
            "worker_log": getattr(getattr(worker, "log", None), "name", None)}


def _stop_failed_worker(worker, status, message, row):
    """Retain observation before cleanup and distinguish a controller's SIGKILL."""
    proc = getattr(worker, "proc", None)
    before = proc.poll() if proc is not None else None
    if proc is not None and before is None and message.get("event") == "error" and not message.get("reason"):
        # A structured worker error is followed by exit(1). Give that normal
        # teardown the same short reap grace before recording forced cleanup.
        try:
            before = proc.wait(timeout=0.2)
        except subprocess.TimeoutExpired:
            pass
    worker.kill(status, message)
    if proc is not None:
        row["worker_process"] = {"returncode_after_cleanup": proc.poll(),
                                 "returncode_before_cleanup": before,
                                 "controller_cleanup_requested": True}


def race(args):
    ab_config = _read_neural_ab_config(getattr(args, "neural_ab_config", None))
    _apply_neural_ab_environment(ab_config)
    operation = _neural_ab_operation(ab_config)
    import numpy as np
    ctd = _load("classical_two_datasets")
    lane, shape = args.lane, args.shape
    arms = [a for a in args.arms.split(",") if a]
    if ab_config and any(a == "ours-fast" for a in arms):
        raise SystemExit("neural A/B config is IDENTICAL-only; ours-fast is incompatible")
    if operation != "model_default" and any(a != "ours" for a in arms):
        raise SystemExit("experimental public operation requires --arms ours; matched opponent operation is not implemented")
    probe = _load("bench_board_probe")
    probe.refuse_our_cpu_arms(arms, "bench_board_neural")
    if DEVICE_OF.get(lane) == "cpu" and any(a.startswith("ours") for a in arms):
        # ours IS the host binding on this lane (a *-infer or host train lane);
        # ours-fast is refused here too: the Apple FAST neural tier is GPU only
        raise SystemExit("bench_board_neural: refused %s on %s (it runs on the CPU, in no numeric "
                         "mode): %s" % (",".join(a for a in arms if a.startswith("ours")), lane,
                                        probe.OUR_CPU_RULE))
    for a in arms:
        if a not in ARMS:
            raise SystemExit("no arm %r for lane %r" % (a, lane))
    os.makedirs(args.out, exist_ok=True)
    os.makedirs(args.work, exist_ok=True)
    frozen_config = None
    if ab_config is not None:
        frozen_config = os.path.abspath(os.path.join(args.out, "neural-ab-config.json"))
        with open(frozen_config, "w", encoding="utf-8") as fh:
            json.dump(ab_config, fh, indent=2, sort_keys=True)
            fh.write("\n")
    tag = "%s-%s" % (lane, DATA_OF[lane])
    if operation != "model_default":
        tag += "-" + operation
    data_path = os.path.join(args.work, "neural-%s-%s.npz" % (lane, shape))
    if getattr(args, "higgs", None):
        os.environ[HIGGS_ENV] = os.path.abspath(os.path.expanduser(args.higgs))   # fit_arrays reads it
    if getattr(args, "lm_checkpoints", None):
        os.environ[LM_CKPT_ENV] = os.path.abspath(os.path.expanduser(args.lm_checkpoints))   # lm_checkpoint reads it
    inputs = make_inputs(lane, shape, args.rounds + 1, data_path)
    srec = shape_record(lane, shape)
    result = {"lane": lane, "dataset": DATA_OF[lane], "shape": srec["label"], "shape_record": srec,
              "inputs": inputs, "arms": {}, "rounds_requested": args.rounds, "started": now_utc(),
              "script": "tools/bench_board_neural.py", "ours_device": DEVICE_OF[lane],
              "torch_settings": {a: precision_text(arm_setting(a)[1]) for a in arms
                                 if a.startswith("torch-")},
              "commit": os.environ.get("MOJOLEARN_REPO_COMMIT", "unknown")}
    result["neural_ab_config"] = ab_config
    result["operation"] = operation
    result["comparison_scope"] = ("declared public experimental operation; not the ordinary lane boundary"
                                  if operation != "model_default" else "ordinary lane boundary")
    # THE SCORER PREFLIGHT (lane harness-evidence, 2026-10-14): refuse before any timed work when the quality
    # scorer cannot run
    if not getattr(args, "params_only", False):
        try:
            result["scorer_preflight"] = scorer_preflight(lane, shape, data_path, arms, operation)
        except ScorerUnavailable as exc:
            why = SCORER_REFUSED + str(exc)
            result["scorer_preflight"] = {"lane": lane, "status": "refused", "reason": why}
            result["quality"] = {"error": why, "scorer": "failed"}
            for arm in arms:
                result["arms"][arm] = {"command": None, "warmup_ms": None, "ms": [], "digests": [],
                                       "mem": [], "status": "REFUSED", "error": why, "median_ms": None}
                print("NEURAL-REFUSED lane=%s arm=%s stage=scorer-preflight detail=%s"
                      % (lane, arm, json.dumps(why)), flush=True)
            result["finished"] = now_utc()
            out_json = os.path.join(args.out, "%s.json" % tag)
            with open(out_json + ".tmp", "w") as fh:
                json.dump(result, fh, indent=2, sort_keys=True, default=str)
            os.replace(out_json + ".tmp", out_json)
            return 1
    elif getattr(args, "scorer_preflight", False):
        # preflight smoke (2026-10-14): the same check, recorded, and the parameter probe still runs
        try:
            result["scorer_preflight"] = scorer_preflight(lane, shape, data_path, arms, operation)
        except ScorerUnavailable as exc:
            result["scorer_preflight"] = {"lane": lane, "status": "refused", "reason": SCORER_REFUSED + str(exc)}
    workers = {}
    # FAIRNESS FIX (2026-10-13 audit): a torch arm on a lane whose only torch twin is a
    # pure-PyTorch reference was raced and reported as the opponent. It is now refused by name
    # here (no worker starts; the cell reads REFUSED with the reason). Results raced before
    # this change were wrong for this lane; apply extra scrutiny.
    refused_before = {a: REFERENCE_TWIN_LANES[lane] for a in arms
                      if lane in REFERENCE_TWIN_LANES and a.startswith("torch-")}
    if lane in FIT_LANES and sys.platform == "darwin" and "ours" in arms:
        # 2026-10-19 (lane ag-bench): identical is never timed on Apple; the fit lanes race ours-fast there
        refused_before["ours"] = APPLE_FIT_IDENTICAL_REFUSED
    for arm in arms:
        if arm in refused_before:
            workers[arm] = RefusedArm()
            result["arms"][arm] = {"command": None, "warmup_ms": None, "ms": [], "digests": [],
                                   "mem": [], "status": "refused", "error": refused_before[arm]}
            print("NEURAL-REFUSED lane=%s arm=%s stage=plan detail=%s"
                  % (lane, arm, json.dumps(refused_before[arm])), flush=True)
            continue
        py = args.ours_python if arm in OUR_ARMS else args.theirs_python
        cmd = shlex.split(py) + [os.path.abspath(__file__), "worker", "--arm", arm,
                                 "--lane", lane, "--shape", shape, "--data", data_path]
        if frozen_config is not None:
            cmd += ["--neural-ab-config", frozen_config]
        workers[arm] = ctd.start_worker(arm, cmd, _worker_env(arm),
                                        os.path.join(args.out, "%s-%s.log" % (tag, arm)), REPO,
                                        params_only=getattr(args, "params_only", False))
        result["arms"][arm] = {"command": cmd, "warmup_ms": None, "ms": [], "digests": [],
                               "mem": [], "status": "ok"}
    for arm, w in workers.items():
        if arm in refused_before:
            continue
        msg = _worker_message(w, args.ready_seconds, "ready")
        if msg.get("event") != "ready":
            row = result["arms"][arm]
            row.update(status="not_ready", error=msg)
            _stop_failed_worker(w, "not_ready", msg, row)
            print("NEURAL-REFUSED lane=%s arm=%s stage=ready detail=%s" % (lane, arm, json.dumps(msg)),
                  flush=True)
            continue
        w.info = msg["info"]
        result["arms"][arm]["info"] = msg["info"]
        result["arms"][arm]["params_record"] = msg.get("params_record")
        result["arms"][arm]["neural_ab_preparation"] = msg.get("neural_ab")
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
            msg = _worker_message(w, warmup_deadline(arm, args) if r == 0 else args.round_seconds,
                                  "round %d" % r, "round %d" % r)
            if msg.get("event") != "round":
                status = "timeout" if msg.get("reason") == "timeout" else "error"
                row = result["arms"][arm]
                row.update(status=status, error=msg, failed_round=r)
                _stop_failed_worker(w, status, msg, row)
                if r == 0 and status == "timeout":
                    result["arms"][arm]["warmup_deadline_s"] = warmup_deadline(arm, args)
                print("NEURAL-REFUSED lane=%s arm=%s stage=round%d detail=%s"
                      % (lane, arm, r, json.dumps(msg)), flush=True)
                continue
            if r == 0:
                result["arms"][arm]["warmup_ms"] = msg["ms"]
                result["arms"][arm]["warmup_compiles"] = msg.get("compiles")
                if is_compile_arm(arm):
                    # FAIRNESS FIX (2026-10-13 audit): the compile (torch.compile's first call)
                    # is this untimed warm-up round; its time is recorded on its own. Results
                    # raced before this change were wrong for this lane; apply extra scrutiny.
                    result["arms"][arm]["compile_warmup_ms"] = msg["ms"]
                    result["arms"][arm]["warmup_deadline_s"] = warmup_deadline(arm, args)
            else:
                result["arms"][arm]["ms"].append(msg["ms"])
                # COMPILE-IN-CLOCK (Andrew 2026-10-14): compiles inside the scored rounds, summed
                rounds_c = result["arms"][arm].setdefault("compile_rounds", [])
                rounds_c.append(msg.get("compiles"))
                result["arms"][arm]["recompiles_in_timed_run"] = _load("bench_board_compile_guard").summarize(rounds_c)
            result["arms"][arm]["digests"].append(msg["digest"])
            if msg.get("neural_ab") is not None:
                result["arms"][arm].setdefault("neural_ab_rounds", []).append(dict(msg["neural_ab"], round=r, warmup=r == 0))
            if msg.get("operation") is not None:
                result["arms"][arm].setdefault("operations", []).append(dict(msg["operation"], round=r, warmup=r == 0))
            if msg.get("state_receipt") is not None:
                result["arms"][arm].setdefault("state_receipts", []).append(msg["state_receipt"])
            result["arms"][arm]["mem"].append(msg.get("mem"))
            if msg.get("extra") is not None:
                result["arms"][arm].setdefault("round_extra", []).append(dict(msg["extra"], round=r, warmup=r == 0))
            if msg.get("device_memory") is not None:
                result["arms"][arm].setdefault("device_memory", []).append(dict(msg["device_memory"], round=r,
                                                                               warmup=r == 0))
            print("NEURAL-ROUND lane=%s arm=%s round=%d ms=%.3f digest=%s"
                  % (lane, arm, r, msg["ms"], msg["digest"]), flush=True)
    outs = {}
    for arm in arms:
        w = workers[arm]
        if w.alive and len(result["arms"][arm]["ms"]) == args.rounds:
            path = os.path.join(args.work, "%s-%s.npz" % (tag, arm))
            msg = _worker_message(w, args.round_seconds, "save", "save %s" % path)
            if msg.get("event") == "saved":
                with np.load(path) as z:
                    outs[arm] = {k: z[k] for k in z.files}
                result["arms"][arm]["outputs_digest"] = outputs_digest(outs[arm])
                print("NEURAL-DIGEST lane=%s arm=%s digest=%s"
                      % (lane, arm, result["arms"][arm]["outputs_digest"]), flush=True)
                if (getattr(args, "keep_outputs", False) or lane == "mlp-train-step" or lane in FORWARD_REFERENCE_LANES
                        or lane in FIT_LANES or lane in OPS_LANES or lane in PROGRAM_LANES):
                        or lane in FIT_LANES or lane in OPS_LANES or lane in GEN_LANES):
                    # Retain the measured trainer state for independent identity checks.
                    # tools/afn_ab.sh's judge compares two builds' outputs
                    keep = os.path.join(args.out, "%s-%s.outputs.npz" % (tag, arm))
                    os.replace(path, keep)
                    result["arms"][arm]["outputs_npz"] = keep
                else:
                    os.remove(path)
            else:
                row = result["arms"][arm]
                row.update(status="save_failed", error=msg)
                _stop_failed_worker(w, "save_failed", msg, row)
        w.close()
    try:
        with np.load(data_path) as z:
            data = {k: z[k] for k in z.files}
        if operation == "transformer_windowed_forward":
            # The ordinary reference uses window=0 and cannot qualify this
            # explicitly different model configuration or its saved ring.
            result["quality"] = {"status": "pending_windowed_forward_and_cache_reference"}
            for row in result["arms"].values():
                if row.get("status") == "ok":
                    row["status"] = "unqualified_experimental_operation"
        elif operation == "residual_dropout_forward_vjp":
            # The ordinary reference computes a transformer block, not this
            # explicit layer. Preserve outputs for its own future admission.
            result["quality"] = {"status": "pending_layer_forward_and_vjp_reference"}
            for row in result["arms"].values():
                if row.get("status") == "ok":
                    row["status"] = "unqualified_experimental_operation"
        else:
            scored = outs
            if (lane in TORCH_REFERENCE_LANES and TORCH_REFERENCE_ARM not in outs
                    and any(a in OUR_ARMS for a in outs)):
                # SCORER GAPS (2026-10-15): the untimed torch reference, after the timed workers stopped
                result["torch_quality_reference"], reference = torch_reference_outputs(
                    args, lane, args.shape, data_path, tag)
                if reference is not None:
                    scored = dict(outs, **{TORCH_REFERENCE_ARM: reference})
            result["quality"] = quality(lane, data, scored, shape=args.shape)
            if scored is not outs:
                result["quality"].pop(TORCH_REFERENCE_ARM, None)    # the reference is not a raced arm
        if operation in ("transformer_forward_vjp_tape", "mamba_forward_vjp"):
            result["experimental_quality"] = {
                "forward": "ordinary host forward comparator only",
                "vjp": "not_qualified_by_forward_quality",
                "full_model_training": "not_measured_by_block_forward_vjp"}
    except Exception as exc:  # noqa: BLE001
        # the scorer itself crashed: "quality not scored", never a quality failure of an arm
        result["quality"] = {"error": repr(exc), "scorer": "failed"}
    mark_forward_quality_failures(result)
    try:
        if getattr(args, "keep_outputs", False) or lane in FORWARD_REFERENCE_LANES:
            keep = os.path.join(args.out, "%s.inputs.npz" % tag)
            os.replace(data_path, keep)
            result["inputs_npz"] = keep
        else:
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
        a["span"] = (fit_span(lane) if lane in FIT_LANES else dict(OPS_SPAN) if lane in OPS_LANES
                     else dict(PROGRAM_SPAN) if lane in PROGRAM_LANES else dict(SPAN))
                     else dict(GEN_SPAN) if lane in GEN_LANES else dict(SPAN))
        if lane in INFER_LANES:
            a["inference"] = inference_summary(a) if ok else None
        print("NEURAL lane=%s arm=%s status=%s median_ms=%s quality=%s"
              % (lane, arm, a["status"], a["median_ms"],
                 json.dumps(result["quality"].get(arm, {}), sort_keys=True)), flush=True)
    # FAIRNESS FIX (2026-10-13 audit): when a compile arm timed out, the other torch arms were
    # recorded as ordinary opponents (the eager arm silently became the opponent). They now
    # carry `not_opponent` (shown as context, never the opponent). Results raced before this
    # change were wrong for this lane; apply extra scrutiny.
    result["opponent_rule"] = OPPONENT_RULE
    for arm, why in sorted(mark_not_opponents(lane, result["arms"]).items()):
        print("NEURAL-NOT-OPPONENT lane=%s arm=%s %s" % (lane, arm, why), flush=True)
    result["finished"] = now_utc()
    out_json = os.path.join(args.out, "%s.json" % tag)
    with open(out_json + ".tmp", "w") as fh:
        json.dump(result, fh, indent=2, sort_keys=True, default=str)
    os.replace(out_json + ".tmp", out_json)
    failed = [a for a in arms if result["arms"][a]["status"] != "ok"]
    return 1 if failed and len(failed) == len(arms) else 0


def inference_summary(a):
    """An inference arm's headline figures from its timed rounds (round > 0) and every memory pass: the median
    prefill_ms, generate_ms and decode_ms_per_token (causal-lm-generate), the largest peak_bytes and
    process_peak_bytes per counter, the resident bytes and torch's allocator peak."""
    timed = [x for x in a.get("round_extra", []) if not x.get("warmup")]
    out = {}
    for k in ("prefill_ms", "generate_ms", "decode_ms_per_token"):
        v = [x[k] for x in timed if x.get(k) is not None]
        out[k] = statistics.median(v) if v else None
    if timed:
        out["new_tokens"] = timed[0].get("new_tokens")
    passes = a.get("device_memory", [])

    def most(get):
        v = [get(p) for p in passes]
        v = [x for x in v if isinstance(x, int)]
        return max(v) if v else None
    for k in ("process", "device"):
        out["peak_bytes_" + k] = most(lambda p, k=k: (p.get("peak_bytes") or {}).get(k))
        out["resident_bytes_" + k] = most(lambda p, k=k: (p.get("resident_bytes") or {}).get(k))
    out["process_peak_bytes"] = most(lambda p: p.get("process_peak_bytes"))
    out["torch_max_allocated_bytes"] = most(lambda p: p.get("torch_max_allocated_bytes"))
    out["memory_method"] = passes[0].get("method") if passes else None
    return out


def build_parser():
    p = argparse.ArgumentParser(prog="bench_board_neural", description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("race")
    r.add_argument("--lane", required=True, choices=LANES)
    r.add_argument("--shape", default="full", choices=sorted(LM_SHAPES))
    r.add_argument("--arms", default="ours,torch-eager-fp32")
    r.add_argument("--neural-ab-config", help="selected source-arm JSON; records requested flags without building or admitting the binary")
    # One seed, one scored run per arm by default (Andrew 2026-10-14): alpha, no noise fighting; just get measurements.
    r.add_argument("--rounds", type=int, default=1)
    r.add_argument("--out", required=True)
    r.add_argument("--work", required=True)
    r.add_argument("--ours-python", default=sys.executable)
    r.add_argument("--theirs-python", default=sys.executable)
    r.add_argument("--ready-seconds", type=int, default=1800)
    r.add_argument("--params-only", action="store_true",
                   help="construct every arm, read its parameters back, write <out>/<tag>.params.json "
                        "and stop before the warm-up (the board's opponent-store lookup)")
    r.add_argument("--scorer-preflight", action="store_true",
                   help="with --params-only: run the quality-scorer preflight first and record its receipt "
                        "(a refusal included) in the params JSON (bench_board.py --preflight-scorer)")
    r.add_argument("--warmup-seconds", type=int, default=1800)
    # FAIRNESS FIX (2026-10-13 audit): a compile arm's own warm-up deadline (warmup_deadline).
    # Results raced before this change were wrong for this lane; apply extra scrutiny.
    r.add_argument("--compile-warmup-seconds", type=int, default=None,
                   help="warm-up deadline of a torch.compile arm (its compile is the untimed "
                        "warm-up round); default 8 x --round-seconds")
    r.add_argument("--round-seconds", type=int, default=1800)
    r.add_argument("--higgs", default=None,
                   help="the HIGGS npz the fit lanes read (default: $%s, else %s)"
                        % (HIGGS_ENV, " or ".join(HIGGS_DEFAULTS)))
    r.add_argument("--lm-checkpoints", default=None,
                   help="the directory holding the causal-lm-generate checkpoints, one folder per key (default: "
                        "$%s, else %s; a missing one is fetched there with huggingface_hub)"
                        % (LM_CKPT_ENV, LM_CKPT_DEFAULT))
    r.add_argument("--keep-outputs", action="store_true",
                   help="keep every arm's saved outputs as <out>/<lane>-<data>-<arm>.outputs.npz "
                        "(y for a forward lane, losses for a train lane): tools/afn_ab.sh's judge "
                        "compares two builds from these")
    w = sub.add_parser("worker")
    w.add_argument("--arm", required=True, choices=ARMS)
    w.add_argument("--lane", required=True, choices=LANES)
    w.add_argument("--shape", required=True, choices=sorted(LM_SHAPES))
    w.add_argument("--neural-ab-config", help="frozen selected source-arm JSON from conductor")
    w.add_argument("--data", required=True)
    return p


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if argv == ["capability-fast-bf16"]:
        print(json.dumps(fast_bf16_capability()), flush=True)
        return 0
    if argv[:1] == ["scorer-preflight-child"]:      # scorer_preflight's child (lane, shape, inputs npz)
        return scorer_preflight_child(*argv[1:4])
    args = build_parser().parse_args(argv)
    if args.cmd == "worker":
        return worker(args)
    if args.rounds < 1:
        raise SystemExit("--rounds must be >= 1")
    return race(args)


if __name__ == "__main__":
    sys.exit(main())
