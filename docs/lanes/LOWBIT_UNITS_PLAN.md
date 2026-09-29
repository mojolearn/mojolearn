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
| Speed | end-to-end time improves, with quantize, pack and convert costs counted; judged on NVIDIA and on Apple only (Andrew, 2026-09-29: the AMD box is checked for bitwise identity and is not timed) |

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
| AMD | `do-amd`, MI325X, IDENTITY ONLY: every arm once for its output hash and the gate verdicts, no timing and no vendor arm (Andrew, 2026-09-29) | `tools/apple_steward.py submit --target do-amd` |
| Apple M2 | `m2pro` | `tools/apple_steward.py submit --target m2pro` |
| Apple M3 Ultra | host held, no instance | waits on Andrew lifting the two locks |

## Rules that bind every lane here

`~/mojolearn-evidence/lane_common_rules.md`, plus: no builds, tests or
benchmarks on the laptop; commit and push at every step; data from R2 only;
no duration estimates; never claim we are faster; a speed number is ours
over theirs at the same shape on the same box.
