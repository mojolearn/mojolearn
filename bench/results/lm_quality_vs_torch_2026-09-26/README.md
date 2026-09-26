# Byte LM training quality, mojolearn 0.8.19 against torch, 2026-09-26

Question. Does the published wheel's IDENTICAL byte LM trainer learn as well
as torch from the same start, and did the recent speed work leave its quality
intact?

Setup. `pip install mojolearn==0.8.19` (the published wheel, no source
build) with `MOJOLEARN_NUMERIC_MODE=identical`, resident session, lean step.
The opponent is torch eager float32 with TF32 off (the `eager_fp32` row of
`tools/torch_lm_step_opponent.py`, SDPA backend `efficient` on both vendors).
Both sides start from the same parameters (numpy `default_rng(93261)`,
normal(0, 0.02), +1 on norms; init sha256 0e36d8d0 on every run) and read the
same batches (ids sha256 equal on all 300 steps). AdamW lr 1e-3, betas (0.9,
0.999), eps 1e-8, weight decay 0.01, constant rate, no warmup, no clipping.

Shape. The GPT-3 Small target shape (162,147,840 parameters, 12 layers, d
768, 12 heads, FF 2048, vocabulary 50,257), batch 1, length 2048, 300 steps
on enwik8 (bytes 0 to 614,401). The target shape fits easily, 300 steps took
under two minutes per side. Held-out loss is the mean next-byte cross entropy
over 32 rows of 2049 bytes spread over the enwik8 validation range (bytes 90M
to 95M), which training never reads.

Tool. `tools/lm_quality_vs_torch.py` (ours, torch, compare) and the box body
`tools/lm_quality_vs_torch_leg.sh`, commit e6f681d71.

## Training loss (ours is one column, its bits are the same on both vendors)

| step | mojolearn (H100 and MI300X) | torch H100 | torch MI300X |
|---|---|---|---|
| 1 | 10.903326 | 10.903326 | 10.903326 |
| 10 | 27.208422 | 27.205992 | 27.201729 |
| 50 | 6.770453 | 5.789887 | 7.556121 |
| 100 | 3.607079 | 3.507805 | 3.382118 |
| 150 | 3.037967 | 2.976823 | 3.060077 |
| 200 | 2.795862 | 2.835634 | 2.956689 |
| 250 | 2.602077 | 2.675054 | 2.664823 |
| 300 | 2.221399 | 2.277791 | 2.236105 |
| mean of steps 251 to 300 | 2.230286 | 2.269878 | 2.274896 |

Differences, ours minus torch.

| | against torch H100 | against torch MI300X |
|---|---|---|
| first step whose loss bits differ | 2 | 2 |
| max absolute difference | 8.830746 (step 41, 52.1 percent) | 4.932355 (step 47, 49.7 percent) |
| final absolute difference, step 300 | -0.056392 (-2.48 percent) | -0.014707 (-0.66 percent) |
| mean of the last 50 steps | -0.039591 (-1.74 percent) | -0.044609 (-1.96 percent) |

The large mid-run gaps come from the recipe, not from either library. With
batch 1, a constant 1e-3 rate and no warmup, every run spikes to about 57.9
at step 3 and stays unstable until about step 60. In that stretch a last-bit
difference at step 2 grows into a different trajectory. Torch against torch
shows the same size. Torch H100 and torch MI300X differ by up to 5.807177 at
step 41 (52.1 percent) and by 0.041685 at step 300. All three runs settle to
the same level after the unstable stretch.

## Held-out loss (32 validation rows, 65,536 targets)

| | mojolearn | torch H100 | torch MI300X |
|---|---|---|---|
| before training | 11.047020 | 11.047019 | 11.047019 |
| after 300 steps, own evaluator | 2.650262 | 2.660241 | 2.659080 |
| after 300 steps, torch evaluator | 2.650261 | 2.660241 | 2.659080 |

Ours ends 0.37 percent below torch H100 and 0.33 percent below torch MI300X on
held-out bytes. Our final parameters loaded into the torch model and scored
by torch give 2.650261 against our own 2.650262 (difference 3.2e-7), so the
two loss definitions agree and the comparison is on one scale.

## mojolearn H100 against mojolearn MI300X

BITWISE IDENTICAL. The float32 loss bits are equal on all 300 steps, the
held-out loss bits are equal on every row before and after training, and the
final parameters, AdamW m and v and flags hash the same (parameters sha256
772a3124eeae59b3...). The bindings are the wheel's `cuda/sm_90a/identical`
and `hip/gfx942/identical` builds (`runtime` in each `ours.json`).

## torch H100 against torch MI300X

Not identical. Loss bits agree on step 1 only; they differ from step 2 on,
by up to 5.807177 (step 41) and by 0.041685 at step 300. Held-out after
training 2.660241 against 2.659080.

## Verdict

No quality gap. From the same init and batches the IDENTICAL trainer's final
training loss sits within the spread between two torch runs on two vendors,
its last 50 steps average 1.7 to 2.0 percent below torch, and its held-out
loss is 0.33 to 0.37 percent below torch. One seed and one corpus, 300 steps.

## Boxes, cost, teardown

| vendor | provider | box | GPU | software | on the clock | cost |
|---|---|---|---|---|---|---|
| NVIDIA | RunPod | pod xpvxyutz8bl22u | H100 80GB HBM3, driver 580.126.09 | Python 3.11, torch 2.4.1+cu124 | 05:54:15Z to 06:02:31Z | about $0.48 at $3.49/h |
| AMD | RunPod | pod 5ipwpk4cncy044 | MI300X, ROCm 6.4.1 image | Python 3.10 (wheel), torch 2.6.0+rocm6.4.1 in a Python 3.12 venv | 06:25:25Z to 06:38:44Z | about $0.53 at $2.39/h |

Both pods were deleted by the runner (HTTP 204) and verified gone (HTTP 404),
`*/teardown.txt`. Hot Aisle was tried first for AMD and showed no MI300X
stock on any spec for 30 minutes, so nothing was created there (balance
unchanged at $48.28). DigitalOcean was not used. The AMD body's exit 1 is
the `torch-python` item of the pin step (the ROCm image has no torch on
PATH); the pinned torch then installed and ran (`amd/torch-pin/status.tsv`).
Each `torch-pin/` directory also holds a 9-step `eager_fp32` timing row as a
by-product of installing the pinned torch.

## Files

`{nvidia,amd}/ours.json` and `torch.json` hold every step's loss as a float
and as float32 hex, the batch ids sha256, the held-out rows, the init sha256
and (ours) the final state hashes and runtime. `compare.log` is each box's
own table, `cross_vendor.txt` the table with both vendors.
