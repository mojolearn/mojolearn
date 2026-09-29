# Low-bit profiles on the matrix units: plan (2026-09-29)

Branch `lane/lowbit-units`, forked from main at the 0.8.25 release tip.
Evidence: `~/mojolearn-evidence/lowbit-units/`.

## The goal (Andrew, 2026-09-29)

1. Bitwise identical results across Apple, NVIDIA and AMD. The new results do
   NOT have to equal what `fp32.v1` produces today.
2. Quality kept. A profile that is under 1% worse may ship behind a flag.
3. Training speed AND inference speed both improve.
4. Every such change is a flag. The default does not move.

## What already exists (read before writing anything)

| Piece | Where | State |
|---|---|---|
| `bf16f32.v1`, `int8i32.v1` | `gemm/IDENTICAL_LOWBIT_CONTRACT.md` | gated on three vendors at nine shapes |
| int8 on the NVIDIA and AMD integer matrix units | `gemm/checks/gemm_int8_mma.mojo` | bit-equal to the flat kernel and the host oracle |
| Apple float matrix unit in IDENTICAL GEMM | `gemm/checks/gemm_identical.mojo` (PLAN_APPLE_MMA) | block admission, exact fallback |
| Model leg, SmolLM2-360M | `bench/model/`, `tools/model_leg/` | written, never run |
| Packed weights in the blocks | `python/mojolearn/lowbit.py` | blocks MATERIALIZE to fp32, so no speed today |

Not done anywhere: a timing of any low-bit plan, a quality number for any
low-bit profile on a real model, a low-bit training path.

## The flag

A named numeric profile, never a switch inside `fp32.v1` (its contract
forbids one).

- Python: `gemm_profile=` on the model classes and `CausalLM.load`, with a
  process default `mojolearn.set_gemm_profile(...)`. It extends the existing
  `weight_format=` and `numeric_mode=` selectors.
- Default: `"fp32.v1"`. Nothing changes for a caller who passes nothing.
- A checkpoint records the profile it was trained under. Loading it under
  another profile is refused by name.
- A profile ships only after it passes all three gates below. A profile that
  costs any quality says so in its docstring with the measured number.

## The three gates, each with a number

| Gate | Passes when |
|---|---|
| Identity | same output bits on Apple, NVIDIA and AMD; full logits, not only tokens; sabotage arm fails |
| Quality | relative change in held-out perplexity under 1% against `fp32.v1` |
| Speed | end-to-end time improves, with quantize, pack and convert costs counted; judged on NVIDIA, Apple and AMD (Andrew, 2026-09-29: first "just the bitwise idenity" on the AMD box, then "ok can we start timing on the amd") |

## Candidates

| Profile | Operands | Unit | Quality risk |
|---|---|---|---|
| `bf16f32.v1` fused in the blocks | bf16 weights, fp32 activations | none, memory traffic only | small |
| `int8i32.v1` | int8 weights and activations | integer unit (NVIDIA, AMD), float unit exact chunks (Apple) | real |
| `int16` from int8 pieces | 15-bit codes, per-row power-of-two scale | same units, several products | unknown |
| mixed `int16` weights, `int8` activations | | same units, two products | unknown |
| native bf16 float unit | | vendor unit | comparison only, no identity claim |

Exact sums do not make the model lossless. Rounding values into codes loses
information, and that loss is what the quality gate measures.

## Order of work

1. QUALITY FIRST, no new kernels. Simulate every candidate's arithmetic in a
   reference implementation on SmolLM2-360M (inference) and the byte LM
   (training). Drop what misses 1%. Lane `lane/lowbit-quality`.
2. TIME WHAT EXISTS on three vendors at training and decode shapes, with the
   conversion costs. Probe the Apple exact-chunk idea on the M2 and on a
   second Apple generation. Lane `lane/lowbit-units`.
3. Integrate the winner into inference. Run the model leg.
4. Training experiment: quantized forward products first, backward second,
   fp32 master weights and the pinned update. Judge by quality reached.
5. Attention products. Ship the flag.

## Machines (orchestrator provisions, lanes never do)

| Column | Box | How |
|---|---|---|
| NVIDIA | shared pod `nvc3`, H100 | `tools/nvidia_central.sh` |
| NVIDIA, quality runs | shared pod `nvc1`, 2x RTX 4090 | `tools/nvidia_central.sh` |
| AMD | `do-amd`, MI325X; identity only for part of 2026-09-29, TIMED since Andrew's "ok can we start timing on the amd": a steward speed job, warm cache, run once untimed and then time | `tools/apple_steward.py submit --target do-amd` |
| Apple M2 | `m2pro` | `tools/apple_steward.py submit --target m2pro` |
| Apple M3 Ultra | `m3ultra-b`, ready since 2026-09-29 ~03:15Z (the brief) | `tools/apple_steward.py submit --target m3ultra-b` |

## Rules that bind every lane here

`~/mojolearn-evidence/lane_common_rules.md`, plus: no builds, tests or
benchmarks on the laptop; commit and push at every step; data from R2 only;
no duration estimates; never claim we are faster; a speed number is ours
over theirs at the same shape on the same box.

## Before anything merges to main (2026-09-29)

One candidate is left: `numeric_profile="fixed15_v1"`, the GEMM
`mojolearn.identical.gemm.int15i64.v1`. int8, the int8-attention mix and
native bf16 on the units are dropped. The merge is in TWO STAGES, and each
stage merges only when every row of its table reads done.

### Stage 1: inference, behind the flag

| Gate | What closes it | State on 2026-09-29 |
|---|---|---|
| GEMM identity | same bits on H100, M3 Ultra, M2 Pro, MI325X, every plan, arms seen failing | DONE, 178 cases |
| GEMM in the repo's verification harness | lane `gemm-int15`, the harness's own verdict on each box | in progress |
| Blocks compute under the profile | projections and attention products run the GEMM on codes, no fp32 materialization | not started |
| Whole-model identity | full logits of SmolLM2-360M equal on every box, a sabotage arm seen failing | not started |
| CPU path | the host binding computes the profile, equal to the devices (a CPU-only install must run it) | host oracle exists, binding not started |
| Quality, inference | whole configuration under 1 percent with margin on two texts | DONE, -0.0015% and +0.0055% |
| Quality, a task evaluation | one task set, staged in R2 | owed |
| Speed, complete operation | under fp32.v1 at the decode rows AND the training-size rows, H100 and M3 Ultra, two runs a box, conversions counted | provisional: 1.08 and 0.77 on the H100 with four int8 products |
| Speed, AMD | one run of record before any sentence states a speed for AMD | owed |
| The default does not move | every fp32.v1 card and digest on main unchanged; default checkpoints byte-equal | to be checked at the merge |
| The repo's merge check | ONE consolidation branch, ONE light check of what changed: M2 Pro, CPU, one NVIDIA run; `test_host_surface`; bindings that import a changed module rebuilt | at the merge |
| Words | contract section, CHANGELOG entry, the flag's docstring with the measured numbers | contract section written |

### Stage 2: training, behind the same flag

| Gate | What closes it | State |
|---|---|---|
| Backward products | the GEMM in the orientations the backward pass needs (the profile is OP_NT only today) | not started |
| Training-step identity | a training step's weights equal on every box, and a resumed checkpoint continues to the same bits | not started |
| Quality, training | five seeds, forward and backward, inside the baseline's spread and under 1 percent | running |
| Gradients | the zero codes of the head's gradient (58.8% at 15 bits) shown harmless, or the pinned stochastic rounding added under a new profile name | running |
| Speed, training | the complete step, weights and gradients converted every step | not started |
| A checkpoint carries its profile | written and refused by name across profiles | flag side written |

### What can merge earlier, because it moves no bit and no default

The aligned load in the int8 unit kernel, the parallel quantizer, the timing
harness and its results, and the `dev_pod` sync fallback. Each through the
same one light check.
