#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""THE OPPONENT ARM for the byte-level language-model speed lanes: a PyTorch
twin of the fixed two-block byte LM, on the same GPU, and the SHARED DATA
RECIPE that `bench/speed/byte_lm_speed_arm.py` (our arm) imports so both
sides consume identical bytes.

    python3 tools/speed_torch_byte_lm.py --lane lm-train --arm torch
    python3 tools/speed_torch_byte_lm.py --lane lm-train --arm torch-deterministic
    python3 tools/speed_torch_byte_lm.py --lane lm-train --arm torch-fast
    python3 tools/speed_torch_byte_lm.py --lane lm-train --arm torch-compiled
    python3 tools/speed_torch_byte_lm.py --lane lm-train --arm torch-compiled-deterministic
    python3 tools/speed_torch_byte_lm.py --lane lm-infer --arm torch --rounds 20

AUTHORED, NOT EXECUTED. No model, benchmark, build or install was run to
produce this file; the first run on a rented box is a BUILD and its output
must be read as such. Importing this module imports numpy only; torch is
imported inside `main` so the recipe half can be consumed by our arm in a
process that never loads torch.

THE QUESTION THESE LANES ANSWER
================================
How much does our IDENTICAL byte-LM trainer -- the one whose 128-step run is
bitwise identical on NVIDIA CUDA, AMD HIP and Apple Metal -- cost against
what a PyTorch user would run for the EXACT same model, on the same H100,
for training (128 steps, lane `lm-train`) and for a forward/evaluate (lane
`lm-infer`). Torch is measured as it ships (arm `torch`), in its documented
deterministic configuration (arm `torch-deterministic`), on its strongest
precision-relaxed default path (arm `torch-fast`), and on its strongest
FULL-FP32 path, compiled, in both its default form (arm `torch-compiled`)
and its documented deterministic form (arm `torch-compiled-deterministic`),
so the price of torch's determinism, the price of ours, the speed torch
reaches when it is allowed to stop being FP32, and the speed it reaches
when it stays FP32 but stops being eager all sit in one table. The last two
arms exist because eager FP32 torch is not torch's strongest FP32
configuration on a launch-bound program, and "faster than PyTorch" is not a
fair sentence until torch's strongest matched-precision configuration is in
the same table (DEVIATIONS 2220, 2221).

THIS IS A LATENCY COMPARISON, NOT A THROUGHPUT ONE (DEVIATION 2191)
====================================================================
The shape is fixed and tiny: batch 2, context 32, d_model 32, 4 heads over
head dim 8, 2 KV heads, FFN 64, vocabulary 256, two blocks, 34,944 FP32
parameters. On an H100 every kernel in it is launch-bound; the numbers are
per-step latency of a small program, and nothing here says anything about
either side at a size that fills the device. Every run prints that as an
`FSPEED-NOTE` so a table cannot lose the caveat.

THE MODEL DEFINITION IS `tools/byte_lm_gradient_oracle.py::reference`
======================================================================
That function is the FP64 PyTorch transcription of the native forward and
loss (rotary base 10000 over head dim 8, GQA with 2 KV heads repeated to 4,
causal mask, RMSNorm with eps 9.999999974752427e-7, SiLU-gated FFN, mean
cross-entropy over the 64 targets; no biases, dropout, final norm or tied
head). `torch_forward` below is that function ported to FP32 on the GPU
with the 20 parameters as leaf tensors. Two spellings differ from the
oracle on purpose and are numbered:

  * DEVIATION 2195: the rotary cos/sin tables are computed in FP64 and
    rounded ONCE to FP32, rather than computed in FP32, so the table holds
    the nearest FP32 to the true angle. The oracle keeps them in FP64.
  * DEVIATION 2204: `repeat_interleave(2, dim=1)` on K and V is spelled as
    `expand` + `reshape`, which yields the same head order
    [kv0, kv0, kv1, kv1] and the same values, but back-propagates through a
    sum over the expanded dimension instead of through `index_add_`. The
    docs list `torch.repeat_interleave()` as deterministic-when-
    differentiated anyway; the respelling removes one indexing backward
    from the hot loop on every torch arm and is NOT what makes the
    deterministic arm pass or fail.

THE DATA RECIPE IS `tools/byte_lm_real_text_capture.py`, REPRODUCED
====================================================================
Every constant of the pinned run is transcribed from that file with its
line, and the initializer is reproduced bit for bit rather than imported:
that file is root-guard-only, imports `mojolearn` in `main`, and lives in a
checkout this arm must not depend on. The pieces:

  * corpus: `training/corpus/tinyshakespeare/input.txt`, 1,115,394 bytes,
    SHA256 `CORPUS_SHA` (lines 25-26, 56); the manifest beside it is
    checked field by field exactly as `read_corpus` does (lines 58-70);
  * training batches: `TRAIN_SCHEDULE` (line 27), implemented by
    `train_ids` (lines 78-84): step s zero-based, row b in {0,1} reads
    bytes[(s*64 + b*32) % 65504 : start+33], the first 32 are inputs and
    the last 32 are the targets, shifted one byte;
  * held-out batches: `VALIDATION_STARTS = range(65536, 66048, 64)`, eight
    starts (line 28), `heldout_ids` (lines 87-90): row b reads
    bytes[start + b*32 : start + b*32 + 33];
  * held-out loss: `evaluate_heldout` (lines 207-231) takes the FP32 mean
    loss of each of the eight batches and reports
    `math.fsum(losses) / 8` -- "math.fsum of eight FP32 batch means / 8;
    512 targets" (line 229). `heldout_mean` is that line;
  * initializer: `INIT_ID` (line 29), `initialize` (lines 110-125), an
    integer avalanche hash of the flat index, top byte centered on 128 and
    divided by 1024, then every `norm1_w`/`norm2_w` tensor set to 1.0.
    `initialize` below is that loop, character for character;
  * optimizer: `SmallByteLanguageModelTrainer(..., lr=.003, betas=(.9,
    .999), eps=1e-8, weight_decay=.01)` (lines 348-349). See DEVIATION
    2190 at `LR` below: the pinned run used lr .003, not the trainer's
    1e-3 default.

THE TORCH ARMS, AND THE SENTENCES THEY REST ON
===============================================
`torch` is torch as it ships: nothing about determinism or TF32 is touched,
and the TF32 switches are READ and printed in a note so the reader knows
what the default was on the day (DEVIATION 2201; `allow_tf32` for matmul
has shipped False since 1.12 and `set_float32_matmul_precision` defaults
to "highest", so the default arm is genuine FP32 GEMM, but it is reported,
not assumed). Eager attention (matmul, mask, softmax, matmul), the oracle's
spelling; `torch.optim.AdamW(foreach=False, fused=False)`, the plain
single-tensor algorithm.

`torch-deterministic` is torch's own documented recipe (DEVIATION 2197),
applied in this order: `CUBLAS_WORKSPACE_CONFIG=:4096:8` placed in
`os.environ` BEFORE `import torch` (which is why `main` parses argv before
importing), then `torch.use_deterministic_algorithms(True)`,
`torch.backends.cudnn.deterministic = True`,
`torch.backends.cudnn.benchmark = False`. The sentences relied on, fetched
2026-09-07:

  * https://docs.pytorch.org/docs/stable/notes/randomness.html (which is
    the 2.14 page today): "torch.use_deterministic_algorithms() lets you
    configure PyTorch to use deterministic algorithms instead of
    nondeterministic ones where available, and to throw an error if an
    operation is known to be nondeterministic (and without a deterministic
    alternative)." and, on cuDNN: "Disabling the benchmarking feature with
    torch.backends.cudnn.benchmark = False causes cuDNN to deterministically
    select an algorithm, possibly at the cost of reduced performance." and
    "that algorithm itself may be nondeterministic, unless either
    torch.use_deterministic_algorithms(True) or
    torch.backends.cudnn.deterministic = True is set."
  * https://docs.pytorch.org/docs/2.8/notes/randomness.html: "Furthermore,
    if you are using CUDA tensors, and your CUDA version is 10.2 or greater,
    you should set the environment variable CUBLAS_WORKSPACE_CONFIG
    according to CUDA documentation:
    https://docs.nvidia.com/cuda/cublas/index.html#results-reproducibility"
    THE STABLE (2.14) PAGE NO LONGER CARRIES THAT SENTENCE; it is quoted
    from the 2.8 page, and the variable is still set here because the
    cuBLAS document it points at still names it and because setting it is
    harmless on a build that no longer needs it.
  * https://docs.pytorch.org/docs/2.14/generated/torch.use_deterministic_algorithms.html:
    "When enabled, operations will use deterministic algorithms when
    available, and if only nondeterministic algorithms are available they
    will throw a RuntimeError when called." The same page lists
    "torch.nn.NLLLoss when called on a CUDA tensor" among the operations
    that "will throw a RuntimeError when mode=True".

That last line matters: `F.cross_entropy` IS log-softmax plus NLLLoss, so
the deterministic arm is EXPECTED to refuse the model as the oracle spells
it. Per the brief, a RuntimeError saying an op has no deterministic
implementation is printed as `FSPEED-REFUSED` with the first line of the
error and the process exits 0 (DEVIATION 2203; any other failure also
prints a refusal but exits 1). That refusal is a finding: torch's
documented deterministic configuration cannot train this model as written.
`torch-deterministic-gatherloss` (DEVIATION 2205, also reachable as
`--arm torch-deterministic --loss gather`) is the respelling
`-(log_softmax.gather(target)).mean()`, whose ops are all on the documented
deterministic list; it is its own arm name so the two spellings can never
share a row, and it is not run unless asked.

`torch-fast` (DEVIATION 2206) is PyTorch's strongest precision-relaxed
default path on an H100, and it is NOT FP32; its header says
`mode=TF32` so no table can read it beside the FP32 arms without the label
(`bench/speed/README.md`: "TF32 and other reduced-precision paths must be
named"). It differs from `torch` in exactly four ways, each named:

  * `torch.set_float32_matmul_precision("high")`,
    `torch.backends.cuda.matmul.allow_tf32 = True` and
    `torch.backends.cudnn.allow_tf32 = True`: every FP32 GEMM may run on
    TF32 tensor cores, ten explicit mantissa bits instead of twenty-three.
    `tools/speed_torch_seq.py` measured the same GEMM at about 5x between
    the two settings on an H100, which is why this is a separate arm and
    not a footnote.
  * attention through `torch.nn.functional.scaled_dot_product_attention(q,
    k, v, is_causal=True)` (DEVIATION 2207), with K and V expanded from 2
    to 4 heads by the same expand + reshape as the eager path, so the
    function computed is the reference's (the SDPA page's own equivalent
    listing is `attn_weight = softmax(q @ k^T * 1/sqrt(E) + causal bias)
    @ v`, which is what the eager path spells with `scale = 1/sqrt(8)`).
    The BACKEND IS AUTO-SELECTED BY TORCH AND IS NOT NAMED HERE; the
    enabled backends are printed in a note, which backend actually ran is
    not observable from Python without a profiler and is reported as
    unverified. What the docs say (fetched 2026-09-07 from
    https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html,
    the 2.14 page): "Scaled dot product attention attempts to
    automatically select the most optimal implementation based on the
    inputs." "Due to the nature of fusing floating point operations, the
    output of this function may be different depending on what backend
    kernel is chosen." "In some circumstances when given tensors on a CUDA
    device and using CuDNN, this operator may select a nondeterministic
    algorithm to increase performance. If this is undesirable, you can try
    to make the operation deterministic (potentially at a performance
    cost) by setting torch.backends.cudnn.deterministic = True." And the
    Reproducibility page's SDPA table: SDPBackend.MATH is "Deterministic"
    forward and backward, FLASH_ATTENTION, EFFICIENT_ATTENTION and
    CUDNN_ATTENTION are "Deterministic" forward and "Non-deterministic"
    backward ("The backward pass uses non-deterministic atomic operations
    by default."), and "Bitwise matching numerics across different SDPA
    backends are not guaranteed, even for the same inputs and dtype." So
    `torch-fast`'s per-round parameter hash is EXPECTED to be free to move
    and is printed as a finding, not a defect.
  * forward + loss compiled with
    `torch.compile(mode="max-autotune-no-cudagraphs")` (DEVIATION 2208).
    The compile happens on the FIRST CALL, so the warm-up (at least one is
    forced for this arm) contains it and the `FSPEED-WARMUP` line for this
    arm is expected to be seconds, not milliseconds; the timed rounds
    re-use the compiled graph. If the compile raises at the first call the
    arm falls back to the uncompiled function, prints an `FSPEED-NOTE`
    saying so, and continues, because a `torch-fast` row without
    `torch.compile` is still the strongest eager path and worth having;
    the note is what keeps the two from being confused. The optimizer step
    is not compiled.
  * `torch.optim.AdamW(fused=True)` (DEVIATION 2209), torch's fused CUDA
    kernel for the whole parameter list.

`torch-compiled` (DEVIATION 2220) is torch's strongest configuration AT
MATCHED PRECISION: full FP32, no TF32 anywhere, and everything else that
`torch-fast` turns on. It exists because the eager `torch` arm measured
SLOWER than our identical-mode trainer on the H100 (2.20 s against 1.09 s
for 128 steps), and a reader is right to object that eager dispatch on a
34,944-parameter model is torch paying per-op launch cost, not torch at its
strongest; the FP32 sentence is only fair against this arm. It is
`torch-fast` with the precision switches inverted, each stated even though
it is the shipped default, and READ BACK after configuration into the
header note (DEVIATION 2222), so the row carries its own witness that no
TF32 path was open:

  * `torch.set_float32_matmul_precision("highest")`,
    `torch.backends.cuda.matmul.allow_tf32 = False`,
    `torch.backends.cudnn.allow_tf32 = False`. The sentence relied on,
    fetched 2026-09-08 from
    https://docs.pytorch.org/docs/2.4/generated/torch.set_float32_matmul_precision.html
    (the page for the torch that ran, 2.4.1+cu124 on the
    runpod/pytorch:2.4.0-py3.11-cuda12.4.1 image): "highest": "float32
    matrix multiplications use the float32 datatype (24 mantissa bits with
    23 bits explicitly stored) for internal computations." and, on the
    switch as a whole: "This does not change the output dtype of float32
    matrix multiplications, it controls how the internal computation of
    the matrix multiplication is performed."
  * attention through `scaled_dot_product_attention(q, k, v,
    is_causal=True)` exactly as `torch-fast` (DEVIATION 2207; backend
    auto-selected, enabled backends printed in their own `FSPEED-NOTE`,
    DEVIATION 2227, which backend ran is not observable from Python).
  * forward + loss compiled with
    `torch.compile(mode="max-autotune-no-cudagraphs")`, the same mode
    string and the same first-call compile and non-silent fallback as
    `torch-fast` (DEVIATION 2208). From
    https://docs.pytorch.org/docs/2.4/generated/torch.compile.html, fetched
    2026-09-08: torch.compile "Optimizes given model/function using
    TorchDynamo and specified backend."; "max-autotune" "is a mode that
    leverages Triton based matrix multiplications and convolutions It
    enables CUDA graphs by default."; "max-autotune-no-cudagraphs" "is a
    mode similar to "max-autotune" but without CUDA graphs". No CUDA graphs
    because the per-step token copy and the uncompiled optimizer step sit
    between compiled calls, as on `torch-fast`.
  * `torch.optim.AdamW(fused=True)` (DEVIATION 2209).

Its header says `mode=FP32-COMPILED` (DEVIATION 2228). Its per-round
parameter hash is free to move for the same reason `torch-fast`'s is: the
SDPA backward "uses non-deterministic atomic operations by default" on
every backend but MATH, and nothing about determinism is touched on this
arm; movement is printed as a finding.

`torch-compiled-deterministic` (DEVIATION 2221) is `torch-compiled` under
the FULL `torch-deterministic` recipe, applied FIRST and in the same order
(`CUBLAS_WORKSPACE_CONFIG=:4096:8` before `import torch`, then
`torch.use_deterministic_algorithms(True)`,
`torch.backends.cudnn.deterministic = True`,
`torch.backends.cudnn.benchmark = False`), then the three FP32 switches
above, then SDPA + compile + fused AdamW. Its header says
`mode=FP32-COMPILED-DETERMINISTIC`. The sentences relied on, fetched
2026-09-08 from the 2.4 pages:

  * https://docs.pytorch.org/docs/2.4/generated/torch.use_deterministic_algorithms.html:
    "Sets whether PyTorch operations must use "deterministic" algorithms.
    That is, algorithms which, given the same input, and when run on the
    same software and hardware, always produce the same output. When
    enabled, operations will use deterministic algorithms when available,
    and if only nondeterministic algorithms are available they will throw
    a RuntimeError when called." The same page: "A handful of CUDA
    operations are nondeterministic if the CUDA version is 10.2 or greater,
    unless the environment variable CUBLAS_WORKSPACE_CONFIG=:4096:8 or
    CUBLAS_WORKSPACE_CONFIG=:16:8 is set." and it lists "torch.nn.NLLLoss
    when called on a CUDA tensor" among the operations that throw.
  * https://docs.pytorch.org/docs/2.4/generated/torch.nn.functional.scaled_dot_product_attention.html:
    "In some circumstances when given tensors on a CUDA device and using
    CuDNN, this operator may select a nondeterministic algorithm to
    increase performance. If this is undesirable, you can try to make the
    operation deterministic (potentially at a performance cost) by setting
    torch.backends.cudnn.deterministic = True." The 2.4 page names the
    backends as "FlashAttention-2", "Memory-Efficient Attention" and the
    "PyTorch C++ implementation" (MATH); it carries NO per-backend
    determinism table. THE 2.4 REPRODUCIBILITY PAGE
    (https://docs.pytorch.org/docs/2.4/notes/randomness.html) CARRIES NO
    SDPA TABLE EITHER; the per-backend table quoted under `torch-fast`
    above (MATH deterministic both ways, FLASH/EFFICIENT/CUDNN
    "Non-deterministic" backward, "The backward pass uses non-deterministic
    atomic operations by default.") is from the STABLE (2.14) page and is
    what a reader of the current docs will find. Which backend torch 2.4
    selects under `use_deterministic_algorithms(True)`, and whether it
    swaps to a deterministic backward or refuses, is NOT stated on any 2.4
    page fetched; it is what the run finds out, and the run prints it.

What the deterministic recipe does to a COMPILED forward is not written
down anywhere fetched: `torch.use_deterministic_algorithms` documents ATen
operators, and the kernels `torch.compile` generates are not on that list
either way. So this arm's documented promise is weaker than
`torch-deterministic`'s, and its hash is EXPECTED to repeat but movement
is a finding, not a defect. Three consequences are wired in:

  * a RuntimeError of the "does not have a deterministic implementation"
    kind raised INSIDE the compiled forward is NOT re-raised as the
    refusal on this arm (on `torch-deterministic` it is, DEVIATION 2203):
    it is caught by the same non-silent compile fallback as any other
    first-call failure, the `FSPEED-NOTE` names the exception text, and
    the uncompiled forward runs (DEVIATION 2223). If the uncompiled
    forward then refuses too, THAT refusal propagates and is printed as
    `FSPEED-REFUSED`, exit 0, exactly as on `torch-deterministic`; the
    note before it says whether the compiled graph refused as well or only
    eager did. That distinction is the point: `torch.compile` decomposes
    `cross_entropy` into its own kernels and may well run the model that
    eager deterministic torch refuses (the NLLLoss CUDA refusal is an ATen
    kernel's check), and a table needs to know which of the two happened.
  * the same for a raise inside the compiled BACKWARD (DEVIATION 2224):
    `loss.backward()` on a compiled forward runs the AOTAutograd-compiled
    backward graph, which is where an SDPA backend's backward would
    refuse. `Forward.backward` wraps it; on this arm a raise there drops
    the partial gradients (`.grad = None` on every leaf), recomputes the
    loss uncompiled, back-propagates that, prints the note with the
    exception text, and stays uncompiled for the rest of the process. On
    every other arm `Forward.backward` IS `loss.backward()` and a raise
    propagates as before.
  * `--loss gather` renames it `torch-compiled-deterministic-gatherloss`
    by the rule `torch-deterministic` uses (DEVIATION 2225); it is not run
    unless asked.

Init, batches, lr, hashing and ACC metrics are the `torch` arm's, unchanged.

WHAT IS AND IS NOT INSIDE THE TIMER (DEVIATIONS 2193, 2194)
============================================================
`lm-train`: one round is one FRESH 128-step run from the same initial
parameters, timed end to end between two device synchronizations. Per-step
times inside the run come from CUDA events recorded around each step and
read after the final synchronize (DEVIATION 2193), so no host sync is
added to the loop and the end-to-end number is the honest one; our arm's
`train_step` returns host arrays and is complete on return, so its per-step
time is host wall-clock. The host-to-device copy of each step's token batch
is INSIDE the timer on both sides (DEVIATION 2194), because our surface
takes a numpy array and copies it in, and a torch user's loop copies too;
our surface additionally validates and copies its full state every step and
hands back the gradients, and that cost stays in for the reason
`bench/speed/forest_speed_arm.py` gives for its transpose: a user of the
surface pays it. Rebuilding the parameters and the optimizer is outside the
timer on both sides; the held-out evaluations are outside the timer.

`lm-infer`: one round is one forward (loss only, no gradient) over the
first held-out batch (`VALIDATION_STARTS[0]`, start 65536) on the INITIAL
parameters (DEVIATION 2199: both sides can build them from `INIT_ID` with
no training, so the batch and the weights are identical by construction).

HASHES (DEVIATION 2192)
========================
`lm-train`'s round hash is SHA256 of the final flat FP32 parameters in
registry order, little-endian; `lm-infer`'s is SHA256 of the FP32 loss
bits. The `FSPEED` line carries the first 16 hex digits, which is the
contract's field width (`tools/speed_gbdt_arm.py::hash_predictions` does
the same truncation); the full digest is printed in an `FSPEED-NOTE` beside
it. Torch's per-round hashes are what show whether torch repeats; the
`torch` and `torch-fast` arms may move between rounds and that is a
finding, not an error.

OUTPUT CONTRACT (the `FSPEED-*` family, `bench/speed/README.md`)
=================================================================
    FSPEED-HEADER family=lm lane=<lane> arm=<arm> mode=<mode> device=<d> rounds=<n> size=shipped
    FSPEED-WARMUP lane=<l> arm=<a> shape=<tag> ms=<f>
    FSPEED lane=<l> arm=<a> shape=<tag> round=<i> ms=<f> hash=<16 hex>
    FSPEED-ACC lane=<l> arm=<a> metric=<m> value=<f>
    FSPEED-REFUSED lane=<l> arm=<a> reason=<one line>
    FSPEED-NOTE lane=<l> arm=<a> <one line>

Shape tags: `bytelm-2x33-steps128` (lm-train), `bytelm-2x33-forward`
(lm-infer). Mode labels (DEVIATION 2198): `torch` prints `mode=FAST`, as
`tools/speed_torch_seq.py` does for torch; `torch-deterministic` and
`torch-deterministic-gatherloss` print `mode=DETERMINISTIC`; `torch-fast`
prints `mode=TF32`; `torch-compiled` prints `mode=FP32-COMPILED` and
`torch-compiled-deterministic` prints `mode=FP32-COMPILED-DETERMINISTIC`
(DEVIATION 2228; the label is informational on a vendor arm, the parser
`tools/identity_grid_json.py` classifies a vendor arm only by whether its
NAME contains `-deterministic`, which both new names honor); our arm prints
the tier `mojolearn.numeric_mode()` reads back after import.
"""

import argparse
import hashlib
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import time

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

FAMILY = "lm"
LANES = ("lm-train", "lm-infer")
SHAPE_TAGS = {"lm-train": "bytelm-2x33-steps128", "lm-infer": "bytelm-2x33-forward"}
#: Timed rounds per lane when neither `--rounds` nor MOJOLEARN_SPEED_ROUNDS
#: says otherwise. A training round is a full 128-step run, so three; a
#: forward is microseconds of work, so twenty.
DEFAULT_ROUNDS = {"lm-train": 3, "lm-infer": 20}
ARMS = ("torch", "torch-deterministic", "torch-deterministic-gatherloss", "torch-fast",
        "torch-compiled", "torch-compiled-deterministic")
#: DEVIATION 2226. The base arms that run torch's documented deterministic
#: recipe (and therefore need CUBLAS_WORKSPACE_CONFIG in the environment
#: BEFORE `import torch`), and the base arms whose forward is
#: `torch.compile`d (and therefore need at least one warm-up for the
#: compile to land in). `main` keys its pre-import arrangement and its
#: warm-up floor on these sets, so a new arm cannot silently miss either.
DETERMINISTIC_BASE_ARMS = ("torch-deterministic", "torch-compiled-deterministic")
COMPILED_BASE_ARMS = ("torch-fast", "torch-compiled", "torch-compiled-deterministic")
COMPILE_MODE = "max-autotune-no-cudagraphs"      # DEVIATION 2208, shared by every compiled arm
SIZE = "shipped"

# --------------------------------------------------------------------------
# The pinned run, transcribed. Every constant names the line it came from.
# --------------------------------------------------------------------------

#: tools/byte_lm_real_text_capture.py:24 and python/mojolearn/_byte_lm_impl.py:22.
PROFILE = "mojolearn.byte-lm.b2-l32-d32-h4-kv2-ff64-v256-blocks2.fp32.v1"
#: tools/byte_lm_real_text_capture.py:25-26 and :56.
CORPUS_RELPATH = os.path.join("training", "corpus", "tinyshakespeare", "input.txt")
CORPUS_SHA = "86c4e6aa9db7c042ec79f339dcb96d42b0075e16b8fc2e86bf0ca57e2dc565ed"
CORPUS_BYTES = 1115394
#: tools/byte_lm_real_text_capture.py:27.
TRAIN_SCHEDULE = ("step s zero-based: row b reads bytes[(s*64+b*32) % 65504 : start+33]; "
                  "targets shifted one byte")
#: tools/byte_lm_real_text_capture.py:28.
VALIDATION_STARTS = list(range(65536, 66048, 64))
#: tools/byte_lm_real_text_capture.py:29.
INIT_ID = "u32-avalanche-index-xor-42595445-top8-centered128-div1024-norm1.v1"
#: tools/byte_lm_real_text_capture.py:66 and :317 (`planned_steps=128`,
#: `--steps` choices (1, 128)); the 128-step run is the one with the
#: learning gate.
STEPS = 128
#: python/mojolearn/_byte_lm_impl.py:32 (`_N = 34944`), also
#: tools/byte_lm_real_text_capture.py:106 and tools/byte_lm_gradient_oracle.py:27.
N_PARAMETERS = 34944
#: tools/byte_lm_gradient_oracle.py:118: "Epsilon is the exact scalar that
#: FP32 native configuration holds."
RMS_EPS = 9.999999974752427e-7
#: tools/byte_lm_gradient_oracle.py:104 (`10000.0 ** ...`).
ROPE_BASE = 10000.0

#: DEVIATION 2190. The pinned 128-step run that produced held-out
#: 5.5413 -> 2.8436 was constructed at
#: tools/byte_lm_real_text_capture.py:348-349 with
#: `lr=.003, betas=(.9, .999), eps=1e-8, weight_decay=.01`, and its
#: admission check at :351-353 pins `lr=float(np.float32(.003))`. The
#: trainer's own default (`python/mojolearn/_byte_lm_impl.py:265`) is
#: `lr=1e-3`, and the brief for this file also said 1e-3. Both arms here
#: use .003 so the run being timed is the run whose result is quoted; the
#: value is one constant so the orchestrator can move it, and the header
#: note prints it so no table can quote the wrong one.
LR = .003
BETAS = (.9, .999)
EPS = 1e-8
WEIGHT_DECAY = .01

#: The 20-tensor registry, python/mojolearn/_byte_lm_impl.py:23-31 (and
#: tools/byte_lm_gradient_oracle.py:20-26, tools/byte_lm_real_text_capture.py:94-100).
#: DEVIATION 2202: re-spelled here so this file imports nothing from
#: `mojolearn`; our arm cross-checks it against
#: `SmallByteLanguageModelTrainer.parameter_registry()` and refuses on a
#: mismatch.
_BLOCK_SHAPES = (("norm1_w", (32,)), ("w_q", (32, 32)), ("w_k", (16, 32)),
                 ("w_v", (16, 32)), ("w_o", (32, 32)), ("norm2_w", (32,)),
                 ("w_gate", (64, 32)), ("w_up", (64, 32)), ("w_down", (32, 64)))
SHAPES = [("embed", (256, 32))]
for _block in range(2):
    SHAPES.extend(("block%d.%s" % (_block, name), shape) for name, shape in _BLOCK_SHAPES)
SHAPES.append(("lm_head", (256, 32)))
NAMES = tuple(name for name, _ in SHAPES)


def registry():
    """`[{name, shape, offset, count}]`, the same list
    tools/byte_lm_real_text_capture.py:101-107 builds and validates."""
    entries, offset = [], 0
    for name, shape in SHAPES:
        count = math.prod(shape)
        entries.append(dict(name=name, shape=list(shape), offset=offset, count=count))
        offset += count
    if offset != N_PARAMETERS:
        raise ValueError("registry does not sum to 34944")
    return entries


def initialize(reg=None):
    """`INIT_ID`, reproduced from tools/byte_lm_real_text_capture.py:110-125
    with the loop body copied verbatim. Pure Python integers, so every bit
    is exact; the only FP32 rounding is the final store of a dyadic value
    with at most 8 significant bits, which is exact.

    Returns a fresh float32[34944]. Not vectorized on purpose: 34,944
    iterations cost tens of milliseconds and a uint32 numpy transcription
    would be a second implementation of the same hash."""
    reg = registry() if reg is None else reg
    values = np.empty(N_PARAMETERS, dtype=np.float32)
    for i in range(N_PARAMETERS):
        h = (i + 1) ^ 0x42595445
        h = (h ^ (h >> 16)) & 0xffffffff
        h = (h * 0x85ebca6b) & 0xffffffff
        h = (h ^ (h >> 13)) & 0xffffffff
        h = (h * 0xc2b2ae35) & 0xffffffff
        h = (h ^ (h >> 16)) & 0xffffffff
        values[i] = ((h >> 24) - 128) / 1024.0
    for entry in reg:
        if entry["name"].endswith(("norm1_w", "norm2_w")):
            values[entry["offset"]:entry["offset"] + entry["count"]] = 1.0
    return values


def sha256_hex(raw):
    return hashlib.sha256(raw).hexdigest()


def read_corpus(root=REPO):
    """tools/byte_lm_real_text_capture.py:55-75, including the manifest
    checks (:58-70), so a corpus that is not the pinned one refuses here
    instead of producing a plausible number for different bytes.

    Returns (raw bytes, manifest dict, manifest raw bytes)."""
    path = os.path.join(root, CORPUS_RELPATH)
    manifest_path = os.path.join(os.path.dirname(path), "manifest.json")
    with open(manifest_path, "rb") as stream:
        manifest_raw = stream.read(65537)
    if len(manifest_raw) > 65536:
        raise ValueError("corpus manifest exceeds bound")
    manifest = json.loads(manifest_raw)
    expected = dict(schema="mojolearn.byte-lm.corpus.v1", sha256=CORPUS_SHA,
                    bytes=CORPUS_BYTES, train_range=[0, 65536], validation_range=[65536, 73728],
                    vocabulary=256, batch=2, context=32, planned_steps=STEPS,
                    train_batch_schedule=TRAIN_SCHEDULE, validation_batch_starts=VALIDATION_STARTS)
    if any(manifest.get(k) != v for k, v in expected.items()):
        raise ValueError("pinned corpus/schedule manifest differs")
    if manifest.get("learning_gate", {}).get("heldout_mean_loss_ratio_max") != .9:
        raise ValueError("predeclared learning threshold changed")
    with open(path, "rb") as stream:
        raw = stream.read(CORPUS_BYTES + 1)
    if len(raw) != CORPUS_BYTES or sha256_hex(raw) != CORPUS_SHA:
        raise ValueError("pinned corpus length/SHA mismatch")
    return raw, manifest, manifest_raw


def train_ids(raw, step):
    """tools/byte_lm_real_text_capture.py:78-84: int32[2,33] for step `step`."""
    rows = []
    for b in range(2):
        start = (step * 64 + b * 32) % 65504
        rows.append(np.frombuffer(raw[start:start + 33], dtype=np.uint8).astype(np.int32))
    return np.stack(rows)


def heldout_ids(raw, start):
    """tools/byte_lm_real_text_capture.py:87-90: int32[2,33] at `start`."""
    return np.stack([np.frombuffer(raw[start + b * 32:start + b * 32 + 33],
                                   dtype=np.uint8).astype(np.int32) for b in range(2)])


def train_batches(raw):
    """All 128 training batches, materialized once, outside every timer."""
    return [train_ids(raw, step) for step in range(STEPS)]


def heldout_batches(raw):
    return [heldout_ids(raw, start) for start in VALIDATION_STARTS]


def heldout_mean(losses):
    """tools/byte_lm_real_text_capture.py:228-229: `math.fsum(losses) / 8`
    over the eight FP32 batch means. `losses` must be the eight FP32 batch
    losses as Python floats, in `VALIDATION_STARTS` order."""
    if len(losses) != len(VALIDATION_STARTS):
        raise ValueError("held-out mean needs exactly eight batch losses")
    return math.fsum(losses) / 8


def data_schedule(raw, manifest_raw, initial):
    """The bounded JSON descriptor tools/byte_lm_real_text_capture.py:335-342
    hands to `SmallByteLanguageModelTrainer(data_schedule=...)`. Metadata
    only; it reaches the checkpoint bytes, so it is reproduced exactly."""
    full_schedule = b"".join(np.asarray(train_ids(raw, s), dtype="<i4", order="C").tobytes()
                             for s in range(STEPS))
    heldout_schedule = b"".join(np.asarray(heldout_ids(raw, s), dtype="<i4", order="C").tobytes()
                                for s in VALIDATION_STARTS)
    return dict(schema="mojolearn.byte-lm.real-text-schedule.v1", corpus_sha256=CORPUS_SHA,
                corpus_manifest_sha256=sha256_hex(manifest_raw),
                train_schedule_sha256=sha256_hex(full_schedule),
                heldout_schedule_sha256=sha256_hex(heldout_schedule), planned_steps=STEPS,
                initialization=INIT_ID,
                initial_parameters_sha256=sha256_hex(flat_bytes(initial)))


def flat_bytes(values):
    """Little-endian FP32 bytes of a flat float32[34944]; the hash input."""
    a = np.asarray(values)
    if a.dtype != np.float32 or a.shape != (N_PARAMETERS,):
        raise TypeError("expected float32[34944]")
    return np.asarray(a, dtype="<f4", order="C").tobytes()


def loss_bytes(value):
    """The FP32 bits of one loss, for the lm-infer hash; refuses a value
    that is not exactly representable so a double sneaks in nowhere."""
    f = np.float32(value)
    if not np.isfinite(f) or float(f) != float(value):
        raise ValueError("loss is not an exact finite FP32 value")
    return np.asarray([f], dtype="<f4").tobytes()


def optimizer_scalars():
    """DEVIATION 2196. The trainer rounds every optimizer scalar to FP32
    at construction (python/mojolearn/_byte_lm_impl.py:80-103, `_float32`)
    and the pinned run's admission check compares against
    `float(np.float32(.003))` and friends (tools/byte_lm_real_text_capture.py:351-353).
    The torch twin receives the SAME rounded values, so both sides hold
    literally the same lr/beta/eps/decay numbers. Torch's AdamW then does
    its scalar arithmetic in Python double, which is one of the reasons the
    twin is a twin and not an identity claim."""
    return dict(lr=float(np.float32(LR)), betas=(float(np.float32(BETAS[0])), float(np.float32(BETAS[1]))),
                eps=float(np.float32(EPS)), weight_decay=float(np.float32(WEIGHT_DECAY)))


# --------------------------------------------------------------------------
# The output contract. Same spellings as tools/speed_gbdt_arm.py:227-263 and
# tools/speed_torch_seq.py:729-734.
# --------------------------------------------------------------------------

def one_line(s, limit=240):
    return " ".join(str(s).split())[:limit]


def emit_header(lane, arm, mode, device, n_rounds):
    print("FSPEED-HEADER family=%s lane=%s arm=%s mode=%s device=%s rounds=%d size=%s"
          % (FAMILY, lane, arm, mode, device, n_rounds, SIZE))
    sys.stdout.flush()


def emit_warmup(lane, arm, shape, ms):
    print("FSPEED-WARMUP lane=%s arm=%s shape=%s ms=%.3f" % (lane, arm, shape, ms))
    sys.stdout.flush()


def emit_round(lane, arm, shape, index, ms, digest):
    print("FSPEED lane=%s arm=%s shape=%s round=%d ms=%.3f hash=%s"
          % (lane, arm, shape, index, ms, digest or "-"))
    sys.stdout.flush()


def emit_acc(lane, arm, metric, value):
    print("FSPEED-ACC lane=%s arm=%s metric=%s value=%.6f" % (lane, arm, metric, value))
    sys.stdout.flush()


def emit_refused(lane, arm, reason):
    print("FSPEED-REFUSED lane=%s arm=%s reason=%s" % (lane, arm, one_line(reason)))
    sys.stdout.flush()


def emit_note(lane, arm, text):
    print("FSPEED-NOTE lane=%s arm=%s %s" % (lane, arm, one_line(text, 400)))
    sys.stdout.flush()


def emit_latency_note(lane, arm):
    """DEVIATION 2191, on the card and not only in the docstring."""
    emit_note(lane, arm, "fixed tiny shape B2/L32/DM32/H4/KV2/FF64/V256 (34944 FP32 "
                         "parameters): this is a LATENCY comparison of a launch-bound "
                         "program, not a throughput comparison; no number here "
                         "describes either side at a size that fills the device")


def emit_hash_note(lane, arm, index, label, digest):
    emit_note(lane, arm, "round=%d %s_sha256=%s" % (index, label, digest))


def round_count(args_rounds, lane):
    """`--rounds`, else MOJOLEARN_SPEED_ROUNDS, else the lane's default."""
    if args_rounds is not None:
        n = int(args_rounds)
    else:
        env = os.environ.get("MOJOLEARN_SPEED_ROUNDS", "").strip()
        n = int(env) if env else DEFAULT_ROUNDS[lane]
    if n < 1:
        raise SystemExit("rounds must be at least 1")
    return n


def device_string():
    """tools/speed_gbdt_arm.py:206-224, copied: `nvidia-smi` first, the host
    platform as the fallback, so a line is never emitted without a device."""
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=20, check=False)
        name = out.stdout.strip().splitlines()
        if out.returncode == 0 and name:
            return name[0].strip().replace(" ", "_")
    except (OSError, subprocess.SubprocessError):
        pass
    return (platform.system() + "_" + platform.machine()).replace(" ", "_")


def build_parser(prog, with_arm):
    p = argparse.ArgumentParser(prog=prog)
    p.add_argument("--lane", required=True, choices=LANES)
    p.add_argument("--rounds", type=int, default=None,
                   help="timed rounds; MOJOLEARN_SPEED_ROUNDS if unset, else %s"
                        % ", ".join("%s=%d" % kv for kv in DEFAULT_ROUNDS.items()))
    p.add_argument("--warmups", type=int, default=1,
                   help="untimed rounds of the same shape before the timed ones "
                        "(DEVIATION 2200: for lm-train a warm-up is a full "
                        "128-step run, so the warm-up line is the same shape "
                        "as the rounds and is never a different program; "
                        "the compiled arms torch-fast, torch-compiled and "
                        "torch-compiled-deterministic force at least one "
                        "because the compile lands on the first call)")
    if with_arm:
        p.add_argument("--arm", required=True, choices=ARMS)
        p.add_argument("--loss", default="cross_entropy", choices=("cross_entropy", "gather"),
                       help="DEVIATION 2205: `cross_entropy` is the oracle's spelling "
                            "(F.cross_entropy) and the default; `gather` respells the "
                            "loss as -(log_softmax.gather(target)).mean(), whose ops are "
                            "on torch's documented deterministic list, and renames the "
                            "arm to <arm>-gatherloss. `--arm torch-deterministic-gatherloss` "
                            "is the same thing as an explicit arm name")
    return p


def resolve_arm(arm, loss):
    """(base arm, loss spelling, reported arm name). The explicit
    `torch-deterministic-gatherloss` arm and `torch-deterministic --loss
    gather` are one row and get one name (DEVIATION 2205/2209).

    DEVIATION 2225: the explicit-name rule is the suffix, not the one
    spelling, so `torch-compiled-deterministic --loss gather` is named
    `torch-compiled-deterministic-gatherloss` and that name, should it ever
    be added to `ARMS`, resolves back to its base by the same rule. For
    `torch-deterministic-gatherloss` the result is unchanged. Every other
    name is its own base arm."""
    if arm.endswith("-gatherloss"):
        return arm[:-len("-gatherloss")], "gather", arm
    if loss == "gather":
        return arm, "gather", arm + "-gatherloss"
    return arm, "cross_entropy", arm


# --------------------------------------------------------------------------
# The torch twin. Nothing below is imported at module import time.
# --------------------------------------------------------------------------

def torch_parameters(torch, initial, device):
    """The 20 leaf tensors, FP32, on `device`, in registry order. Sliced from
    the same flat float32[34944] our arm hands its trainer."""
    weights = {}
    for entry in registry():
        start, end = entry["offset"], entry["offset"] + entry["count"]
        block = np.ascontiguousarray(initial[start:end].reshape(entry["shape"]), dtype=np.float32)
        weights[entry["name"]] = torch.tensor(block, dtype=torch.float32, device=device,
                                              requires_grad=True)
    return weights


def torch_flat(torch, weights):
    """The 20 tensors back to one flat float32[34944] on the host, registry
    order, for the parameter hash."""
    parts = [weights[entry["name"]].detach().reshape(-1) for entry in registry()]
    return torch.cat(parts).to(torch.float32).cpu().numpy().astype(np.float32, copy=False)


def torch_tables(torch, device):
    """Rotary cos/sin (DEVIATION 2195: FP64 then one rounding to FP32) and the
    causal mask; tools/byte_lm_gradient_oracle.py:104-108."""
    frequency = ROPE_BASE ** (-torch.arange(0, 8, 2, dtype=torch.float64) / 8)
    angle = torch.arange(32, dtype=torch.float64)[:, None] * frequency
    angle = torch.cat((angle, angle), dim=-1)[None, None]
    cosine = angle.cos().to(torch.float32).to(device)
    sine = angle.sin().to(torch.float32).to(device)
    mask = torch.ones((32, 32), dtype=torch.bool, device=device).triu(1)
    return cosine, sine, mask


def torch_forward(torch, F, weights, tokens, tables, loss_spelling="cross_entropy", attention="eager"):
    """tools/byte_lm_gradient_oracle.py:103-137 in FP32. `tokens` is a
    long[2,33] on the device; returns the scalar mean loss over 64 targets.
    `attention` is `eager` (the oracle's matmul/mask/softmax/matmul) or
    `sdpa` (DEVIATION 2207: `torch-fast`, `torch-compiled` and
    `torch-compiled-deterministic`)."""
    cosine, sine, mask = tables
    h = F.embedding(tokens[:, :-1], weights["embed"])

    def rotate(a):
        half = torch.cat((-a[..., 4:], a[..., :4]), dim=-1)
        return a * cosine + half * sine

    for block in range(2):
        prefix = "block%d." % block

        def linear(x, name):
            return F.linear(x, weights[prefix + name])

        def norm(x, name):
            return x * torch.rsqrt(x.square().mean(-1, keepdim=True) + RMS_EPS) * weights[prefix + name]

        z = norm(h, "norm1_w")
        q = linear(z, "w_q").reshape(2, 32, 4, 8).transpose(1, 2)
        k = linear(z, "w_k").reshape(2, 32, 2, 8).transpose(1, 2)
        v = linear(z, "w_v").reshape(2, 32, 2, 8).transpose(1, 2)
        q, k = rotate(q), rotate(k)
        # DEVIATION 2204: `repeat_interleave(2, dim=1)` as expand + reshape;
        # [B, 2, L, 8] -> [B, 2, 1, L, 8] -> [B, 2, 2, L, 8] -> [B, 4, L, 8],
        # head order kv0, kv0, kv1, kv1, the same as repeat_interleave. The
        # same expansion feeds SDPA, so GQA is handled outside the kernel
        # exactly as the reference does it and `enable_gqa` is not used.
        k = k[:, :, None].expand(2, 2, 2, 32, 8).reshape(2, 4, 32, 8)
        v = v[:, :, None].expand(2, 2, 2, 32, 8).reshape(2, 4, 32, 8)
        if attention == "sdpa":
            # DEVIATION 2207: default scale is 1/sqrt(E) = 1/sqrt(8), the
            # eager path's `/ math.sqrt(8)`; is_causal=True is the eager
            # path's `triu(1)` mask filled with -inf. Backend auto-selected.
            attended = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        else:
            scores = q @ k.transpose(-1, -2) / math.sqrt(8)
            probability = scores.masked_fill(mask, -torch.inf).softmax(-1)
            attended = probability @ v
        attended = attended.transpose(1, 2).reshape(2, 32, 32)
        residual = h + linear(attended, "w_o")
        z = norm(residual, "norm2_w")
        gate = linear(z, "w_gate")
        activated = gate * gate.sigmoid()
        h = residual + linear(activated * linear(z, "w_up"), "w_down")
    # No final RMSNorm, bias, tied head, dropout or cache carried between
    # steps (tools/byte_lm_gradient_oracle.py:135).
    logits = F.linear(h, weights["lm_head"]).reshape(64, 256)
    targets = tokens[:, 1:].reshape(64)
    if loss_spelling == "cross_entropy":
        return F.cross_entropy(logits, targets, reduction="mean")
    # DEVIATION 2205, opt-in only.
    log_probability = F.log_softmax(logits, dim=-1)
    return -(log_probability.gather(1, targets[:, None]).squeeze(1)).mean()


def is_nondeterministic_refusal(exc):
    """The RuntimeError torch raises under use_deterministic_algorithms(True)
    for an op without a deterministic implementation. Matched on the
    documented wording ("does not have a deterministic implementation") and
    on the setting's name, so an unrelated CUDA error is not swallowed."""
    text = str(exc)
    return isinstance(exc, RuntimeError) and (
        "does not have a deterministic implementation" in text
        or "use_deterministic_algorithms" in text)


def first_line(exc):
    return one_line(str(exc).strip().splitlines()[0] if str(exc).strip() else exc.__class__.__name__)


def tf32_switches(torch):
    return dict(matmul_allow_tf32=getattr(getattr(torch.backends.cuda, "matmul", None), "allow_tf32", None),
                cudnn_allow_tf32=getattr(torch.backends.cudnn, "allow_tf32", None),
                float32_matmul_precision=getattr(torch, "get_float32_matmul_precision", lambda: None)())


def sdpa_backends(torch):
    """Which SDPA backends torch has ENABLED. Not which one ran."""
    cuda = torch.backends.cuda
    out = {}
    for label, probe in (("flash", "flash_sdp_enabled"), ("mem_efficient", "mem_efficient_sdp_enabled"),
                         ("math", "math_sdp_enabled"), ("cudnn", "cudnn_sdp_enabled")):
        fn = getattr(cuda, probe, None)
        try:
            out[label] = None if fn is None else bool(fn())
        except Exception:                           # noqa: BLE001
            out[label] = None
    return out


def apply_deterministic_recipe(torch):
    """DEVIATION 2197, torch's documented deterministic recipe in its
    documented order, shared by `torch-deterministic` and
    `torch-compiled-deterministic` (DEVIATION 2221) so the two arms cannot
    drift apart in what "deterministic" means. Returns the applied-switches
    text for the header note."""
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        raise RuntimeError("CUBLAS_WORKSPACE_CONFIG was not set before torch was imported")
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    return ("CUBLAS_WORKSPACE_CONFIG=:4096:8 (pre-import) "
            "use_deterministic_algorithms(True) cudnn.deterministic=True cudnn.benchmark=False")


def apply_full_fp32(torch):
    """DEVIATION 2222. The three precision switches stated explicitly at
    their shipped-default values, so the FP32 claim of the compiled arms
    rests on a line in this file and not on the day's defaults; the
    values are read back by `tf32_switches` afterwards and printed, so the
    row also carries the witness. Returns the applied-switches text."""
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    return "set_float32_matmul_precision('highest') cuda.matmul.allow_tf32=False cudnn.allow_tf32=False"


def compiled_settings_text(torch):
    """The SDPA + compile + fused-AdamW part of the header note, shared by
    every compiled arm so the three describe the same three switches in
    the same words."""
    return ("attention=scaled_dot_product_attention(is_causal=True, backend auto-selected, "
            "enabled=%s) compile=torch.compile(mode=%r) optimizer=AdamW(fused=True)"
            % (sdpa_backends(torch), COMPILE_MODE))


def configure_arm(torch, base_arm):
    """Apply the base arm's documented configuration and return (mode label,
    description for the header note, arm settings dict). DEVIATIONS 2197,
    2198, 2201, 2206, 2220, 2221, 2228. `settings["deterministic"]` is
    what `Forward` keys its refusal handling on (DEVIATIONS 2223, 2224)."""
    settings = dict(attention="eager", compile=False, fused=False, deterministic=False)
    if base_arm == "torch-deterministic":
        applied = apply_deterministic_recipe(torch)
        settings["deterministic"] = True
        mode = "DETERMINISTIC"
    elif base_arm == "torch-compiled":
        # DEVIATION 2220. FP32 at full precision, then everything
        # `torch-fast` turns on that is not a precision switch.
        applied_fp32 = apply_full_fp32(torch)
        settings = dict(attention="sdpa", compile=True, fused=True, deterministic=False)
        mode = "FP32-COMPILED"
        applied = ("%s %s; deterministic_algorithms=%s untouched"
                   % (applied_fp32, compiled_settings_text(torch),
                      torch.are_deterministic_algorithms_enabled()))
    elif base_arm == "torch-compiled-deterministic":
        # DEVIATION 2221. The deterministic recipe FIRST, in its documented
        # order, then the FP32 switches, then SDPA + compile + fused AdamW.
        applied_det = apply_deterministic_recipe(torch)
        applied_fp32 = apply_full_fp32(torch)
        settings = dict(attention="sdpa", compile=True, fused=True, deterministic=True)
        mode = "FP32-COMPILED-DETERMINISTIC"
        applied = "%s %s %s" % (applied_det, applied_fp32, compiled_settings_text(torch))
    elif base_arm == "torch-fast":
        # DEVIATION 2206. NOT FP32. Every switch named, in the order applied.
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        settings = dict(attention="sdpa", compile=True, fused=True)
        mode = "TF32"
        applied = ("set_float32_matmul_precision('high') cuda.matmul.allow_tf32=True cudnn.allow_tf32=True "
                   "attention=scaled_dot_product_attention(is_causal=True, backend auto-selected, "
                   "enabled=%s) compile=torch.compile(mode='max-autotune-no-cudagraphs') "
                   "optimizer=AdamW(fused=True); deterministic_algorithms=%s untouched"
                   % (sdpa_backends(torch), torch.are_deterministic_algorithms_enabled()))
    else:
        mode = "FAST"
        applied = ("torch default configuration; deterministic_algorithms=%s cudnn.deterministic=%s "
                   "cudnn.benchmark=%s, none of them changed by this arm"
                   % (torch.are_deterministic_algorithms_enabled(), torch.backends.cudnn.deterministic,
                      torch.backends.cudnn.benchmark))
    description = "%s; tf32 switches as found after configuration: %s" % (applied, tf32_switches(torch))
    return mode, description, settings


class Forward:
    """The forward+loss callable for one arm: `fn(tensors, tokens)` over the
    20 tensors IN REGISTRY ORDER (a list, not the dict, so a fresh
    parameter set per round hits the same compiled graph; DEVIATION 2208).

    For the compiled arms (`torch-fast`, `torch-compiled`,
    `torch-compiled-deterministic`) the callable is `torch.compile`d and the
    compile lands on the FIRST CALL, inside the warm-up. If that first call
    raises, the arm falls back to the eager callable for the rest of the
    process and says so in an `FSPEED-NOTE`; the fallback is NOT silent.

    A deterministic-mode refusal (`is_nondeterministic_refusal`) raised by
    the compiled call is re-raised unchanged on every arm except the
    deterministic compiled one, where it takes the same non-silent fallback
    with the exception text in the note and the uncompiled forward is then
    given its own chance to refuse (DEVIATION 2223). `backward` does the
    same for the compiled backward graph (DEVIATION 2224)."""

    def __init__(self, torch, F, tables, loss_spelling, settings, lane, arm):
        self.torch, self.lane, self.arm = torch, lane, arm
        self.compiled_active = False
        self.fallback_reason = None
        self.deterministic = bool(settings.get("deterministic", False))

        def eager(tensors, tokens):
            return torch_forward(torch, F, dict(zip(NAMES, tensors)), tokens, tables,
                                 loss_spelling, settings["attention"])

        self.eager = eager
        self.fn = eager
        if settings["compile"]:
            try:
                self.fn = torch.compile(eager, mode=COMPILE_MODE)
                self.compiled_active = True
            except Exception as exc:                # noqa: BLE001
                self._fallback("torch.compile() raised at construction: %s: %s"
                               % (exc.__class__.__name__, first_line(exc)))

    def _fallback(self, reason):
        self.fn = self.eager
        self.compiled_active = False
        self.fallback_reason = reason
        emit_note(self.lane, self.arm, "COMPILE FALLBACK to the uncompiled forward: %s" % reason)

    def _refusal_text(self, where, exc):
        """DEVIATION 2223/2224: the note names the exception text, and says
        it was the deterministic mode that refused, so a reader can tell a
        compiled-graph refusal from an unrelated compile failure."""
        return ("%s raised a deterministic-mode refusal %s: %s; the uncompiled forward "
                "gets its own chance to refuse next" % (where, exc.__class__.__name__, first_line(exc)))

    def __call__(self, tensors, tokens):
        if not self.compiled_active:
            return self.eager(tensors, tokens)
        try:
            return self.fn(tensors, tokens)
        except Exception as exc:                    # noqa: BLE001
            if is_nondeterministic_refusal(exc):
                if not self.deterministic:
                    raise
                # DEVIATION 2223: on the deterministic compiled arm the
                # refusal inside the compiled graph is a fallback, not the
                # verdict; eager's own refusal, if any, is the verdict.
                self._fallback(self._refusal_text("first compiled call", exc))
                return self.eager(tensors, tokens)
            self._fallback("first compiled call raised %s: %s" % (exc.__class__.__name__, first_line(exc)))
            return self.eager(tensors, tokens)

    def backward(self, loss, tensors, tokens):
        """`loss.backward()`, verbatim, on every arm and every call where
        the compiled graph is not active. DEVIATION 2224: on the
        deterministic compiled arm, while the compiled graph IS active, a
        deterministic-mode refusal raised by the compiled backward is
        caught, the partial gradients that backward may have written are
        dropped (`.grad = None` on every leaf, which is what the loop's
        `zero_grad(set_to_none=True)` had left them as), the loss is
        recomputed by the uncompiled forward and back-propagated, and the
        fallback note names the exception text. Any other exception, on
        any arm, propagates exactly as `loss.backward()` would have."""
        if not (self.compiled_active and self.deterministic):
            loss.backward()
            return
        try:
            loss.backward()
        except Exception as exc:                    # noqa: BLE001
            if not is_nondeterministic_refusal(exc):
                raise
            self._fallback(self._refusal_text("first compiled backward", exc))
            for tensor in tensors:
                tensor.grad = None
            self.eager(tensors, tokens).backward()


def make_optimizer(torch, tensors, settings):
    """DEVIATION 2196: the float32-rounded scalars. `torch`/`torch-deterministic`:
    foreach/fused OFF, the plain single-tensor AdamW (decoupled decay).
    `torch-fast`, `torch-compiled`, `torch-compiled-deterministic`:
    DEVIATION 2209, `fused=True`."""
    scalars = optimizer_scalars()
    if settings["fused"]:
        return torch.optim.AdamW(tensors, lr=scalars["lr"], betas=scalars["betas"], eps=scalars["eps"],
                                 weight_decay=scalars["weight_decay"], fused=True)
    return torch.optim.AdamW(tensors, lr=scalars["lr"], betas=scalars["betas"], eps=scalars["eps"],
                             weight_decay=scalars["weight_decay"], foreach=False, fused=False)


def ordered(weights):
    return [weights[name] for name in NAMES]


def run_train_lane(torch, lane, arm, forward, settings, initial, batches, heldout, device, n_rounds, warmups):
    """Lane `lm-train`: `n_rounds` fresh 128-step runs, each timed end to
    end between two `torch.cuda.synchronize()` calls, plus the held-out
    losses before and after (outside the timer) and the per-step medians."""
    shape = SHAPE_TAGS[lane]
    host_batches = [np.ascontiguousarray(b, dtype=np.int64) for b in batches]
    heldout_dev = [torch.from_numpy(np.ascontiguousarray(b, dtype=np.int64)).to(device) for b in heldout]

    def fresh():
        weights = torch_parameters(torch, initial, device)
        tensors = ordered(weights)
        return weights, tensors, make_optimizer(torch, tensors, settings)

    def evaluate(tensors):
        losses = []
        with torch.no_grad():
            for tokens in heldout_dev:
                losses.append(float(forward(tensors, tokens).detach().to(torch.float32).cpu()))
        torch.cuda.synchronize()
        return heldout_mean(losses)

    def one_run(tensors, optimizer):
        """Returns (end-to-end ms, list of per-step ms from CUDA events)."""
        starts = [torch.cuda.Event(enable_timing=True) for _ in range(STEPS)]
        ends = [torch.cuda.Event(enable_timing=True) for _ in range(STEPS)]
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for step in range(STEPS):
            starts[step].record()
            tokens = torch.from_numpy(host_batches[step]).to(device, non_blocking=False)  # DEVIATION 2194
            optimizer.zero_grad(set_to_none=True)
            loss = forward(tensors, tokens)
            forward.backward(loss, tensors, tokens)     # `loss.backward()` on every arm but DEVIATION 2224
            optimizer.step()
            ends[step].record()
        torch.cuda.synchronize()
        total_ms = (time.perf_counter() - t0) * 1000.0
        step_ms = [starts[i].elapsed_time(ends[i]) for i in range(STEPS)]   # DEVIATION 2193
        return total_ms, step_ms

    _, initial_tensors, _ = fresh()
    emit_acc(lane, arm, "heldout_loss_initial", evaluate(initial_tensors))
    del initial_tensors

    for _ in range(warmups):
        _, tensors, optimizer = fresh()
        ms, _ = one_run(tensors, optimizer)
        emit_warmup(lane, arm, shape, ms)
    if settings["compile"]:
        emit_note(lane, arm, "compiled_forward_active=%s after warm-up%s"
                  % (forward.compiled_active, "" if forward.fallback_reason is None
                     else "; fallback: " + forward.fallback_reason))

    all_step_ms, finals, hashes = [], [], []
    for index in range(1, n_rounds + 1):
        weights, tensors, optimizer = fresh()
        ms, step_ms = one_run(tensors, optimizer)
        flat = torch_flat(torch, weights)
        if not np.isfinite(flat).all():
            raise RuntimeError("round %d produced nonfinite parameters" % index)
        digest = sha256_hex(flat_bytes(flat))
        hashes.append(digest)
        emit_round(lane, arm, shape, index, ms, digest[:16])
        emit_hash_note(lane, arm, index, "final_parameters", digest)
        emit_note(lane, arm, "round=%d step_ms_median=%.4f step_ms_min=%.4f step_ms_max=%.4f"
                  % (index, statistics.median(step_ms), min(step_ms), max(step_ms)))
        all_step_ms.extend(step_ms)
        finals.append(evaluate(tensors))
    if len(set(hashes)) > 1:
        emit_note(lane, arm, "hash moved across rounds: %s %s"
                  % (hashes[0][:16], next(h for h in hashes if h != hashes[0])[:16]))
    else:
        emit_note(lane, arm, "hash repeated across %d rounds" % len(hashes))
    emit_acc(lane, arm, "heldout_loss_final", finals[-1])
    if len(set(finals)) > 1:
        emit_note(lane, arm, "heldout_loss_final moved across rounds: %r" % (finals,))
    emit_acc(lane, arm, "step_ms_median", statistics.median(all_step_ms))


def run_infer_lane(torch, lane, arm, forward, settings, initial, heldout, device, n_rounds, warmups):
    """Lane `lm-infer`: one forward over `VALIDATION_STARTS[0]` on the
    initial parameters per round (DEVIATION 2199), no gradient."""
    shape = SHAPE_TAGS[lane]
    tensors = ordered(torch_parameters(torch, initial, device))
    host = np.ascontiguousarray(heldout[0], dtype=np.int64)

    def one_forward():
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            tokens = torch.from_numpy(host).to(device, non_blocking=False)   # DEVIATION 2194
            loss = forward(tensors, tokens)
            value = float(loss.to(torch.float32).cpu())
        torch.cuda.synchronize()
        return (time.perf_counter() - t0) * 1000.0, value

    for _ in range(warmups):
        ms, _ = one_forward()
        emit_warmup(lane, arm, shape, ms)
    if settings["compile"]:
        emit_note(lane, arm, "compiled_forward_active=%s after warm-up%s"
                  % (forward.compiled_active, "" if forward.fallback_reason is None
                     else "; fallback: " + forward.fallback_reason))
    hashes, last = [], None
    for index in range(1, n_rounds + 1):
        ms, value = one_forward()
        digest = sha256_hex(loss_bytes(value))
        hashes.append(digest)
        last = value
        emit_round(lane, arm, shape, index, ms, digest[:16])
    if len(set(hashes)) > 1:
        emit_note(lane, arm, "hash moved across rounds: %s %s"
                  % (hashes[0][:16], next(h for h in hashes if h != hashes[0])[:16]))
    else:
        emit_note(lane, arm, "hash repeated across %d rounds" % len(hashes))
    emit_acc(lane, arm, "forward_loss", last)
    emit_note(lane, arm, "forward_loss_fp32_bits=0x%08x forward_loss_sha256=%s batch_start=%d"
              % (int(np.asarray([np.float32(last)]).view(np.uint32)[0]), hashes[-1], VALIDATION_STARTS[0]))


def main(argv=None):
    args = build_parser("speed_torch_byte_lm", with_arm=True).parse_args(argv)
    lane = args.lane
    base_arm, loss_spelling, arm = resolve_arm(args.arm, args.loss)
    n_rounds = round_count(args.rounds, lane)
    warmups = args.warmups
    if warmups < 0:
        raise SystemExit("warmups must be nonnegative")
    if base_arm in COMPILED_BASE_ARMS and warmups < 1:
        warmups = 1        # DEVIATION 2208/2226: the compile must land in a warm-up

    # DEVIATION 2197/2226: the workspace variable must precede `import torch`
    # on EVERY arm whose base runs the deterministic recipe.
    if base_arm in DETERMINISTIC_BASE_ARMS:
        if "torch" in sys.modules:
            emit_refused(lane, arm, "torch was imported before CUBLAS_WORKSPACE_CONFIG could be set; "
                                    "run this file as its own process")
            return 1
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    for thread_env in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(thread_env, "2")   # tools/speed_torch_seq.py:116-117

    try:
        import torch
        import torch.nn.functional as F
    except ImportError as exc:
        emit_refused(lane, arm, "torch is not importable: %s" % exc)
        return 1
    if not torch.cuda.is_available():
        # tools/speed_torch_seq.py::require_accelerator, minus MPS: this
        # comparison is the H100 one and a CPU forward would be a perfectly
        # good number for the wrong device.
        emit_refused(lane, arm, "no CUDA/HIP device visible to torch %s; not timing on CPU" % torch.__version__)
        return 1
    device = torch.device("cuda")
    device_name = torch.cuda.get_device_name(0).replace(" ", "_")
    hip = getattr(torch.version, "hip", None)
    build = ("ROCm " + str(hip)) if hip else ("CUDA " + str(getattr(torch.version, "cuda", "?")))

    try:
        mode, description, settings = configure_arm(torch, base_arm)
    except Exception as exc:                        # noqa: BLE001
        emit_refused(lane, arm, "%s: %s" % (exc.__class__.__name__, first_line(exc)))
        return 1

    try:
        raw, _manifest, _manifest_raw = read_corpus()
        initial = initialize()
    except Exception as exc:                        # noqa: BLE001
        emit_refused(lane, arm, "recipe: %s: %s" % (exc.__class__.__name__, first_line(exc)))
        return 1
    batches = train_batches(raw)
    heldout = heldout_batches(raw)
    scalars = optimizer_scalars()

    emit_header(lane, arm, mode, device_name, n_rounds)
    emit_note(lane, arm, "torch=%s build=%s %s" % (torch.__version__, build, description))
    emit_note(lane, arm, "model=%s definition=tools/byte_lm_gradient_oracle.py::reference ported to FP32 "
                         "leaf tensors; loss_spelling=%s attention=%s"
              % (PROFILE, loss_spelling, settings["attention"]))
    emit_note(lane, arm, "recipe init=%s corpus_sha256=%s initial_parameters_sha256=%s steps=%d "
                         "optimizer=AdamW(lr=%r,betas=%r,eps=%r,weight_decay=%r,%s)"
              % (INIT_ID, CORPUS_SHA, sha256_hex(flat_bytes(initial)), STEPS,
                 scalars["lr"], scalars["betas"], scalars["eps"], scalars["weight_decay"],
                 "fused=True" if settings["fused"] else "foreach=False,fused=False"))
    emit_latency_note(lane, arm)
    if base_arm == "torch-fast":
        emit_note(lane, arm, "NOT FP32: TF32 GEMM, auto-selected SDPA backend (which one ran is "
                             "UNVERIFIED from Python), torch.compile, fused AdamW; hash movement "
                             "across rounds is expected here and is a finding, not a defect")
    if base_arm in ("torch-compiled", "torch-compiled-deterministic"):
        # DEVIATION 2220/2221: the FP32 claim, with its witness, on the card.
        switches = tf32_switches(torch)
        emit_note(lane, arm, "FULL FP32: float32_matmul_precision=%s cuda.matmul.allow_tf32=%s "
                             "cudnn.allow_tf32=%s as read back after configuration; "
                             "torch.compile(mode=%r) + auto-selected SDPA backend (which one ran is "
                             "UNVERIFIED from Python) + fused AdamW; this is torch's strongest "
                             "configuration at matched precision, %s"
                  % (switches["float32_matmul_precision"], switches["matmul_allow_tf32"],
                     switches["cudnn_allow_tf32"], COMPILE_MODE,
                     "deterministic algorithms untouched, so hash movement across rounds is "
                     "expected and is a finding, not a defect" if base_arm == "torch-compiled" else
                     "under torch's documented deterministic recipe; the hash is expected to "
                     "repeat and movement is a finding, since torch.compile's generated kernels "
                     "are not on use_deterministic_algorithms' documented operator list"))
        if any(v not in (False, "highest") for v in switches.values()):
            emit_note(lane, arm, "WARNING: a precision switch did not read back closed after "
                                 "configuration: %s; do not quote this run as FP32 until that is "
                                 "explained" % switches)
        # DEVIATION 2227: which SDPA backends torch has ENABLED, on its own
        # line, for both compiled FP32 arms; not which one ran.
        emit_note(lane, arm, "sdpa_backends_enabled=%s deterministic_algorithms=%s "
                             "cudnn.deterministic=%s cudnn.benchmark=%s"
                  % (sdpa_backends(torch), torch.are_deterministic_algorithms_enabled(),
                     torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark))
    if hip:
        emit_note(lane, arm, "this is a ROCm build; the device string is torch's and the "
                             "CUBLAS_WORKSPACE_CONFIG sentence is written for CUDA")

    try:
        tables = torch_tables(torch, device)
        forward = Forward(torch, F, tables, loss_spelling, settings, lane, arm)
        if lane == "lm-train":
            run_train_lane(torch, lane, arm, forward, settings, initial, batches, heldout, device,
                           n_rounds, warmups)
        else:
            run_infer_lane(torch, lane, arm, forward, settings, initial, heldout, device, n_rounds, warmups)
    except Exception as exc:                        # noqa: BLE001
        if is_nondeterministic_refusal(exc):
            # DEVIATION 2203: the documented deterministic configuration
            # cannot run this model as spelled. A finding, exit 0.
            emit_refused(lane, arm, "torch deterministic mode: %s" % first_line(exc))
            return 0
        emit_refused(lane, arm, "%s: %s" % (exc.__class__.__name__, first_line(exc)))
        return 1
    print("FSPEED-DONE lane=%s arm=%s" % (lane, arm))
    return 0


if __name__ == "__main__":
    sys.exit(main())
