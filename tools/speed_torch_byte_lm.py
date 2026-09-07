#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""THE OPPONENT ARM for the byte-level language-model speed lanes: a PyTorch
twin of the fixed two-block byte LM, on the same GPU, and the SHARED DATA
RECIPE that `bench/speed/byte_lm_speed_arm.py` (our arm) imports so both
sides consume identical bytes.

    python3 tools/speed_torch_byte_lm.py --lane lm-train --arm torch
    python3 tools/speed_torch_byte_lm.py --lane lm-train --arm torch-deterministic
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
`lm-infer`). Torch is measured twice: as it ships (arm `torch`) and in its
own documented deterministic configuration (arm `torch-deterministic`), so
the price of torch's determinism and the price of ours sit in one table.

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
    from the hot loop on both torch arms and is NOT what makes the
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

THE TWO TORCH ARMS, AND THE SENTENCES THEY REST ON (DEVIATION 2197)
====================================================================
`torch` is torch as it ships: nothing about determinism or TF32 is touched,
and the TF32 switches are READ and printed in a note so the reader knows
what the default was on the day (DEVIATION 2201; `allow_tf32` for matmul
has shipped False since 1.12 and `set_float32_matmul_precision` defaults
to "highest", so the default arm is genuine FP32 GEMM, but it is reported,
not assumed).

`torch-deterministic` is torch's own documented recipe, applied in this
order: `CUBLAS_WORKSPACE_CONFIG=:4096:8` placed in `os.environ` BEFORE
`import torch` (which is why `main` parses argv before importing), then
`torch.use_deterministic_algorithms(True)`,
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
`--loss gather` (DEVIATION 2205) is the OPT-IN respelling `-(log_softmax
.gather(target)).mean()`, whose ops are all on the documented deterministic
list; it changes the arm name to `<arm>-gatherloss` so the two spellings can
never share a row. It is not run unless asked.

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
`torch` arm may move between rounds and that is a finding, not an error.

OUTPUT CONTRACT (the `FSPEED-*` family, `bench/speed/README.md`)
=================================================================
    FSPEED-HEADER family=lm lane=<lane> arm=<arm> mode=<mode> device=<d> rounds=<n> size=shipped
    FSPEED-WARMUP lane=<l> arm=<a> shape=<tag> ms=<f>
    FSPEED lane=<l> arm=<a> shape=<tag> round=<i> ms=<f> hash=<16 hex>
    FSPEED-ACC lane=<l> arm=<a> metric=<m> value=<f>
    FSPEED-REFUSED lane=<l> arm=<a> reason=<one line>
    FSPEED-NOTE lane=<l> arm=<a> <one line>

Shape tags: `bytelm-2x33-steps128` (lm-train), `bytelm-2x33-forward`
(lm-infer). Mode labels (DEVIATION 2198): the `torch` arm prints
`mode=FAST`, as `tools/speed_torch_seq.py` does for torch; the
`torch-deterministic` arm prints `mode=DETERMINISTIC`; our arm prints the
tier `mojolearn.numeric_mode()` read back after import.
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
ARMS = ("torch", "torch-deterministic")
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
                        "as the rounds and is never a different program)")
    if with_arm:
        p.add_argument("--arm", required=True, choices=ARMS)
        p.add_argument("--loss", default="cross_entropy", choices=("cross_entropy", "gather"),
                       help="DEVIATION 2205: `cross_entropy` is the oracle's spelling "
                            "(F.cross_entropy) and the default; `gather` respells the "
                            "loss as -(log_softmax.gather(target)).mean(), whose ops are "
                            "on torch's documented deterministic list, and renames the "
                            "arm to <arm>-gatherloss")
    return p


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


def torch_forward(torch, F, weights, tokens, tables, loss_spelling="cross_entropy"):
    """tools/byte_lm_gradient_oracle.py:103-137 in FP32. `tokens` is a
    long[2,33] on the device; returns the scalar mean loss over 64 targets."""
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
        # head order kv0, kv0, kv1, kv1, the same as repeat_interleave.
        k = k[:, :, None].expand(2, 2, 2, 32, 8).reshape(2, 4, 32, 8)
        v = v[:, :, None].expand(2, 2, 2, 32, 8).reshape(2, 4, 32, 8)
        scores = q @ k.transpose(-1, -2) / math.sqrt(8)
        probability = scores.masked_fill(mask, -torch.inf).softmax(-1)
        attended = (probability @ v).transpose(1, 2).reshape(2, 32, 32)
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


def configure_arm(torch, arm):
    """Apply the arm's documented configuration and return the mode label
    and a description for the header note (DEVIATION 2197, 2198, 2201)."""
    tf32 = dict(matmul_allow_tf32=getattr(getattr(torch.backends.cuda, "matmul", None), "allow_tf32", None),
                cudnn_allow_tf32=getattr(torch.backends.cudnn, "allow_tf32", None),
                float32_matmul_precision=getattr(torch, "get_float32_matmul_precision", lambda: None)())
    if arm == "torch-deterministic":
        if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
            raise RuntimeError("CUBLAS_WORKSPACE_CONFIG was not set before torch was imported")
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        mode = "DETERMINISTIC"
        applied = ("CUBLAS_WORKSPACE_CONFIG=:4096:8 (pre-import) "
                   "use_deterministic_algorithms(True) cudnn.deterministic=True cudnn.benchmark=False")
    else:
        mode = "FAST"
        applied = ("torch default configuration; deterministic_algorithms=%s cudnn.deterministic=%s "
                   "cudnn.benchmark=%s, none of them changed by this arm"
                   % (torch.are_deterministic_algorithms_enabled(), torch.backends.cudnn.deterministic,
                      torch.backends.cudnn.benchmark))
    description = "%s; tf32 switches as found: %s" % (applied, tf32)
    return mode, description


def run_train_lane(torch, F, lane, arm, initial, batches, heldout, device, n_rounds, warmups, loss_spelling):
    """Lane `lm-train`: `n_rounds` fresh 128-step runs, each timed end to
    end between two `torch.cuda.synchronize()` calls, plus the held-out
    losses before and after (outside the timer) and the per-step medians."""
    shape = SHAPE_TAGS[lane]
    tables = torch_tables(torch, device)
    scalars = optimizer_scalars()
    host_batches = [np.ascontiguousarray(b, dtype=np.int64) for b in batches]
    heldout_dev = [torch.from_numpy(np.ascontiguousarray(b, dtype=np.int64)).to(device) for b in heldout]

    def fresh():
        weights = torch_parameters(torch, initial, device)
        # DEVIATION 2196: the float32-rounded scalars; foreach/fused OFF so
        # this is the plain single-tensor AdamW algorithm, decoupled decay.
        optimizer = torch.optim.AdamW([weights[e["name"]] for e in registry()],
                                      lr=scalars["lr"], betas=scalars["betas"], eps=scalars["eps"],
                                      weight_decay=scalars["weight_decay"], foreach=False, fused=False)
        return weights, optimizer

    def evaluate(weights):
        losses = []
        with torch.no_grad():
            for tokens in heldout_dev:
                losses.append(float(torch_forward(torch, F, weights, tokens, tables, loss_spelling)
                                    .detach().to(torch.float32).cpu()))
        torch.cuda.synchronize()
        return heldout_mean(losses)

    def one_run(weights, optimizer):
        """Returns (end-to-end ms, list of per-step ms from CUDA events)."""
        starts = [torch.cuda.Event(enable_timing=True) for _ in range(STEPS)]
        ends = [torch.cuda.Event(enable_timing=True) for _ in range(STEPS)]
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for step in range(STEPS):
            starts[step].record()
            tokens = torch.from_numpy(host_batches[step]).to(device, non_blocking=False)  # DEVIATION 2194
            optimizer.zero_grad(set_to_none=True)
            loss = torch_forward(torch, F, weights, tokens, tables, loss_spelling)
            loss.backward()
            optimizer.step()
            ends[step].record()
        torch.cuda.synchronize()
        total_ms = (time.perf_counter() - t0) * 1000.0
        step_ms = [starts[i].elapsed_time(ends[i]) for i in range(STEPS)]   # DEVIATION 2193
        return total_ms, step_ms

    initial_weights, _ = fresh()
    emit_acc(lane, arm, "heldout_loss_initial", evaluate(initial_weights))
    del initial_weights

    for _ in range(warmups):
        weights, optimizer = fresh()
        ms, _ = one_run(weights, optimizer)
        emit_warmup(lane, arm, shape, ms)

    all_step_ms, finals, hashes = [], [], []
    for index in range(1, n_rounds + 1):
        weights, optimizer = fresh()
        ms, step_ms = one_run(weights, optimizer)
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
        finals.append(evaluate(weights))
    if len(set(hashes)) > 1:
        emit_note(lane, arm, "hash moved across rounds: %s %s"
                  % (hashes[0][:16], next(h for h in hashes if h != hashes[0])[:16]))
    else:
        emit_note(lane, arm, "hash repeated across %d rounds" % len(hashes))
    emit_acc(lane, arm, "heldout_loss_final", finals[-1])
    if len(set(finals)) > 1:
        emit_note(lane, arm, "heldout_loss_final moved across rounds: %r" % (finals,))
    emit_acc(lane, arm, "step_ms_median", statistics.median(all_step_ms))


def run_infer_lane(torch, F, lane, arm, initial, heldout, device, n_rounds, warmups, loss_spelling):
    """Lane `lm-infer`: one forward over `VALIDATION_STARTS[0]` on the
    initial parameters per round (DEVIATION 2199), no gradient."""
    shape = SHAPE_TAGS[lane]
    tables = torch_tables(torch, device)
    weights = torch_parameters(torch, initial, device)
    host = np.ascontiguousarray(heldout[0], dtype=np.int64)

    def one_forward():
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            tokens = torch.from_numpy(host).to(device, non_blocking=False)   # DEVIATION 2194
            loss = torch_forward(torch, F, weights, tokens, tables, loss_spelling)
            value = float(loss.to(torch.float32).cpu())
        torch.cuda.synchronize()
        return (time.perf_counter() - t0) * 1000.0, value

    for _ in range(warmups):
        ms, _ = one_forward()
        emit_warmup(lane, arm, shape, ms)
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
    lane, arm = args.lane, args.arm
    if args.loss == "gather":
        arm = arm + "-gatherloss"      # DEVIATION 2205
    n_rounds = round_count(args.rounds, lane)
    if args.warmups < 0:
        raise SystemExit("warmups must be nonnegative")

    # DEVIATION 2197: the workspace variable must precede `import torch`.
    if args.arm == "torch-deterministic":
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
        mode, description = configure_arm(torch, args.arm)
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

    emit_header(lane, arm, mode, device_name, n_rounds)
    emit_note(lane, arm, "torch=%s build=%s %s" % (torch.__version__, build, description))
    emit_note(lane, arm, "model=%s definition=tools/byte_lm_gradient_oracle.py::reference ported to FP32 "
                         "leaf tensors; loss_spelling=%s" % (PROFILE, args.loss))
    emit_note(lane, arm, "recipe init=%s corpus_sha256=%s initial_parameters_sha256=%s steps=%d "
                         "optimizer=AdamW(lr=%r,betas=%r,eps=%r,weight_decay=%r,foreach=False,fused=False)"
              % (INIT_ID, CORPUS_SHA, sha256_hex(flat_bytes(initial)), STEPS,
                 optimizer_scalars()["lr"], optimizer_scalars()["betas"], optimizer_scalars()["eps"],
                 optimizer_scalars()["weight_decay"]))
    emit_latency_note(lane, arm)
    if hip:
        emit_note(lane, arm, "this is a ROCm build; the device string is torch's and the "
                             "CUBLAS_WORKSPACE_CONFIG sentence is written for CUDA")

    try:
        if lane == "lm-train":
            run_train_lane(torch, F, lane, arm, initial, batches, heldout, device, n_rounds, args.warmups, args.loss)
        else:
            run_infer_lane(torch, F, lane, arm, initial, heldout, device, n_rounds, args.warmups, args.loss)
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
