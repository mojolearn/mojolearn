# Byte LM inference on the CPU

`mojolearn.LanguageModelInference` runs the byte-level decoder language model's
forward pass on a CPU with no GPU. It loads the parameters of a checkpoint
written by `LanguageModelTrainer.export_checkpoint` and returns logits, the
mean next-byte loss, or greedy next bytes.

```python
import mojolearn
model = mojolearn.LanguageModelInference.from_checkpoint("final.checkpoint.json")
logits = model.logits(ids)          # int32 [batch, length] -> float32 [batch, length, 256]
bits = model.loss_bits(batch_ids)   # int32 [2, 33] -> IEEE-754 bits of the mean loss
```

## What it computes and why it can match the GPUs

There is no new arithmetic. The CPU forward calls the host FP32 oracles that
the Metal, CUDA and HIP kernels are gated against bit for bit, in the order
`training/byte_lm.mojo::_byte_forward_loss` launches those kernels.

| stage | CPU (this path) | GPU kernel it mirrors |
|---|---|---|
| embedding | `emb_forward_oracle` | `identical_embedding_forward_into` |
| each block | `transformer_block_oracle`, fresh cache, prefill at 0 | `llama_decoder_layer_forward` |
| head | `gemm_oracle(OP_NT)` | `identical_gemm_into(..., OP_NT)` |
| loss | `ce_forward_oracle(causal_lm)` | `identical_ce_forward_into` |

That composition is a prediction until the gate below runs on a CPU. A CPU is
certified only by its own row in the table at the end of this file.

The threaded path (DEVIATIONS 2616 and 2640) computes the same numbers through
host kernels in `training/byte_lm_host_kernels.mojo`. Each kernel performs the
operations its oracle performs, on the same operands, in the same order for
every output value, and changes only what surrounds that arithmetic. There is no
allocation per GEMM cell. Operands are flushed and packed once per call. The
cells of a GEMM row advance together as SIMD lanes, with the flush deferred
behind an exact fallback. Per-element seams run as lanes, and the maxima whose
fold shape the contracts leave free run as a branch-free total-order fold. That
is an argument and not a qualification; the gate and the two checks below are
the qualification.

## Scope

- One model family, the byte LM profile
  `mojolearn.byte-lm.b2-l32-d32-h4-kv2-ff64-v256-blocks2.fp32.v1`.
- FP32, forward only. No training, no backward pass, no optimizer on the CPU.
- The reference path is the oracles as written, single-threaded scalar loops.
  `threaded=True` runs the kernels described above and splits batch rows
  across at most `threads` threads (`None` is one per physical core);
  `threaded=True, threads=1` is the kernels on one core. No float crosses a
  thread and no fold changes order, and the gate requires both paths to
  reproduce the capture bytes and each other's logits.
- Every GPU estimator, block and trainer still requires a GPU. On a CPU-only
  install they raise by name on use (DEVIATION 2615).

## The gate

`tools/byte_lm_host_gate.py` (DEVIATION 2613) loads parameters from the retained
three-vendor capture
`bench/results/resume/2026-09-07-root-byte-lm-three-vendor/`, checks every file
against the SHA-256 its own manifest recorded, and requires the CPU loss to
reproduce the recorded 4 loss bytes for

- 8 held-out batches before training (initial parameters),
- 8 held-out batches after 128 steps (final parameters),
- the training batches of the selected steps (parameters before each step).

The capture is Apple's, and its `comparison.json` records that CUDA and HIP
matched it on every one of these bytes, so a match is a match against all three.
The gate also records SHA-256 of fixed logits probes so CPUs can be compared at
full `[batch, length, vocab]` resolution.

The negative control (DEVIATION 2612) is a build with
`-D MOJOLEARN_BYTE_LM_HOST_SABOTAGE=1`, which folds the head product in reverse
order, inside the GEMM kernel on the threaded path. The gate run on that build
must find a mismatch on both paths.

Two Mojo programs check the kernels where the captures cannot reach.
`training/checks/byte_lm_host_kernels_check.mojo` compares every kernel with
its oracle by bits. It covers GEMM shapes with scalar tails, eight-chain groups
and two or three leaves, rows split across calls, and the reversed fold. It
exercises the deferred flush's fallback both forced and triggered by planted
subnormal accumulators, with a self test that the plants would expose a missing
flush. It checks the free-shape maximum against both scalar folds on 20000
planted rows (signed zeros, subnormals, infinities, quiet and signaling NaNs),
and RMS norm, RoPE, whole blocks at head_dim 8 and 6 and the loss at vocab 256
and 300. `training/checks/byte_lm_host_exp_check.mojo` compares the lane-wise
exponential and SiLU with the scalar seams on all 4294967296 Float32 bit
patterns. Its first run found the lanes returning signaling NaNs quieted where
the scalar returns them unchanged. No model input reaches a signaling NaN, but
the lanes now pass NaNs through by an integer select and match on every pattern.

## Running it

The default is GitHub's free standard runners. `.github/workflows/byte-lm-cpu-gate.yml`
runs both builds, both gates and the fake-binding plumbing tests on ARM64 Linux
(Azure Cobalt 100), Apple M1 macOS and five x86-64 Linux draws, on every push
to `main` or the lane branches that touches the CPU path, and uploads the
reports. GitHub assigns the x86-64 host, so Intel and AMD rows come only from a
report that names the CPU. A row enters the table below only after its uploaded
reports are read. The workflow also builds both Mojo checks with the binding's
CPU target and runs them on every runner. Locally, from the repository root,
after `bindings/build_byte_lm_host.sh`:

```sh
pixi run mojo build -D MOJOLEARN_NUMERIC_IDENTICAL=1 -I . training/checks/byte_lm_host_kernels_check.mojo -o kernels_check && ./kernels_check
pixi run mojo build -D MOJOLEARN_NUMERIC_IDENTICAL=1 -I . training/checks/byte_lm_host_exp_check.mojo -o exp_check && ./exp_check
python tools/byte_lm_host_gate.py --steps all --threads 3
```

A rented CPU-only droplet (DEVIATION 2614) runs the same body. It is the
fallback, not the default: do not overlap it with another lane's GPU legs.

```sh
MOJOLEARN_GEMM_LEG_EXTRA=tools/byte_lm_host_leg.sh \
MOJOLEARN_DO_EXTRA_UPLOAD=/absolute/path/byte_lm_host_capture_payload.tgz \
bash tools/do_extra_leg.sh cpu-intel --minutes 45
```

`cpu-amd` is the AMD twin. The upload is the capture subset `git archive`
excludes, packed at its repository paths; the body unpacks it.

## Speed

Apple M4, bare metal, through the Python surface, median of repeated calls, one
thread unless the row says three. Every build in the table produced the same
logits SHA-256 at `[1, 1]`, `[1, 32]`, `[2, 32]`, `[8, 32]` and `[32, 32]` and
the same loss bits. The reference row is from the session's first and quietest
window. Each kernel row is from an interleaved A/B run against the build that
followed it, in one window per pair. The Mac was shared with other work, so
compare rows within this table and not with other machines.

| path | `[1, 1]` logits | `[2, 32]` logits | `[32, 32]` logits | `[2, 33]` loss |
|---|---|---|---|---|
| reference path, the oracles as written | 0.26 ms | 13.3 ms | 214 ms | 13.5 ms |
| kernels at `6349278a` | 0.12 ms | 0.87 ms | 12.3 ms | 1.13 ms |
| kernels at `185ad160` | 0.042 ms | 0.57 ms | 8.7 ms | 0.78 ms |
| kernels at `450addf1` | 0.041 ms | 0.456 ms | 6.9 ms | 0.535 ms |
| kernels at `450addf1`, three threads | 0.041 ms | 0.269 ms | 2.72 ms | 0.359 ms |

What moved the numbers, each measured by interleaved A/B with bits unchanged.
Removing per-cell allocation and advancing GEMM cells as lanes came first. Then
register accumulators, a deferred flush (17% at `[32, 32]`), setup straight
from parameter offsets (a one-token call from 0.12 to 0.045 ms), lane-wise
exponential and SiLU, and the total-order maximum (about 20%). Measured and
dropped: sixteen accumulator chains instead of eight (18% slower), and
splitting one sequence's head product across threads (slower at `[1, 32]`).

PyTorch 2.13 on the same parameters (eager FP32 with
`scaled_dot_product_attention`, `torch.set_num_threads`) computes the same
function, with a largest logit difference of 2.9e-6 and identical argmax, and
makes no cross-vendor bit identity claim. In one window with the kernels at
`450addf1`, on a loaded Mac, PyTorch took 0.30 ms at `[2, 32]` and 3.4 ms at
`[32, 32]` on one thread against the kernels' 0.80 ms and 11.9 ms, and 0.35 ms
and 2.4 ms on three threads against 0.41 ms and 4.2 ms. PyTorch is faster on
this model on this CPU. The kernels keep every fold order the contracts pin,
which a reordering BLAS does not; how much of the difference that accounts for
is not measured.

Evidence `bench/results/local/2026-09-11_1618-apple-m4-byte-lm-cpu-speed`
holds the gate reports at `450addf1` with `--threads 3`, the user path through
`from_checkpoint` on both paths, both Mojo check outputs, every A/B row, the
same-window PyTorch rows, a profile and the benchmark scripts.

## Certified CPUs

| CPU | host | build | loss bytes equal | sabotage caught | evidence |
|---|---|---|---|---|---|
| Intel x86_64, DigitalOcean Premium Intel (model name masked by QEMU; AVX2, FMA), 4 vCPU | Ubuntu 24.04, Linux 6.8, Python 3.12.3 | `--target-cpu x86-64-v3`, Mojo 1.0.0, commit `146becba` (reference path only) | 33 of 33 (16 held-out, 17 training steps) | yes, 9 of 33 differ | `bench/results/e1g/2026-09-11_163928-cpu-intel-byte-lm-host` |
| AMD x86_64, DigitalOcean Premium AMD (model name masked by QEMU; family 23 model 49, Zen 2; AVX2, FMA), 4 vCPU | Ubuntu 24.04, Python 3.12.3 | `--target-cpu x86-64-v3`, Mojo 1.0.0, commit `da4fe0c6` | 33 of 33 on the reference AND threaded paths | yes, 9 of 33 on both paths | `bench/results/e1g/2026-09-11_165210-cpu-amd-byte-lm-host-threads` |
| AMD x86_64, EPYC 9V45 (Zen 5, family 26 model 2; AVX2, FMA, AVX-512 present but not targeted), GitHub `ubuntu-24.04` | Ubuntu 24.04, Python 3.12 | `--target-cpu x86-64-v3`, Mojo 1.0.0, commit `5e10863b` | 33 of 33 on the reference AND threaded paths | yes, 9 of 33 on both paths | `bench/results/gh-actions/2026-09-11_165309-byte-lm-cpu-gate-run34624545221/byte-lm-cpu-gate-ubuntu-24.04` |
| Intel x86_64, Xeon 6973P-C (Granite Rapids, family 6 model 173; AVX2, FMA, AVX-512 present but not targeted), GitHub `ubuntu-22.04` | Ubuntu 22.04, Linux 6.8, Python 3.12.14 | `--target-cpu x86-64-v3`, Mojo 1.0.0, commit `205b22fa` (main merged) | 33 of 33 on the reference AND threaded paths; plumbing tests 8 passed | yes, 9 of 33 on both paths | `bench/results/gh-actions/2026-09-11_1716-byte-lm-cpu-gate-run34626867783/byte-lm-cpu-gate-x86-e` |
| Intel x86_64, Xeon Platinum 8573C (Emerald Rapids, family 6 model 207; AVX2, FMA, AVX-512 present but not targeted), GitHub `ubuntu-24.04` and `ubuntu-22.04` | Ubuntu 24.04 and 22.04, Python 3.12.14 | `--target-cpu x86-64-v3`, Mojo 1.0.0, commit `155f6195` | 33 of 33 on the reference AND threaded paths in two draws; plumbing tests 8 passed | yes, 9 of 33 on both paths | `bench/results/gh-actions/2026-09-11_1722-byte-lm-cpu-gate-run34627322685/byte-lm-cpu-gate-x86-b` and `x86-d` |
| Intel x86_64, Xeon Platinum 8370C (Ice Lake, family 6 model 106; AVX2, FMA, AVX-512 present but not targeted), GitHub `ubuntu-24.04` | Ubuntu 24.04, Python 3.12.14 | `--target-cpu x86-64-v3`, Mojo 1.0.0, commit `b06cd442` (main) | 33 of 33 on the reference AND threaded paths; plumbing tests 8 passed | yes, 9 of 33 on both paths | `bench/results/gh-actions/2026-09-11_1727-byte-lm-cpu-gate-run34627708675/byte-lm-cpu-gate-x86-b` |
| AMD x86_64, EPYC 7763 (Zen 3, family 25 model 1; AVX2, FMA), GitHub `ubuntu-24.04` and `ubuntu-22.04` | Ubuntu 24.04 and 22.04 | `--target-cpu x86-64-v3`, Mojo 1.0.0, commit `205b22fa` | 33 of 33 on both paths in six draws; plumbing tests 8 passed | yes, 9 of 33 on both paths | runs 34626868143 (x86-b, c, e) and 34626867783 (x86-a, b, d) |
| AMD x86_64, EPYC 9V74 (Zen 4, family 25 model 17), GitHub `ubuntu-24.04` and `ubuntu-22.04` | Ubuntu 24.04 and 22.04 | `--target-cpu x86-64-v3`, Mojo 1.0.0, commit `205b22fa` | 33 of 33 on both paths in three draws; plumbing tests 8 passed | yes, 9 of 33 on both paths | runs 34626868143 (x86-a, d) and 34626867783 (x86-c) |
| ARM64, Azure Cobalt 100 (Arm Neoverse N2, implementer 0x41 part 0xd49; ASIMD, SVE, SVE2), GitHub `ubuntu-24.04-arm` | Ubuntu 24.04, Python 3.12 | no CPU flag (aarch64 default), Mojo 1.0.0, commit `5e10863b` | 33 of 33 on both paths | yes, 9 of 33 on both paths | `.../byte-lm-cpu-gate-ubuntu-24.04-arm` |
| Apple M1 (virtual), GitHub `macos-15` | macOS 15, Python 3.12 | `--target-cpu apple-m1`, Mojo 1.0.0, commit `5e10863b` | 33 of 33 on both paths | yes, 9 of 33 on both paths | `.../byte-lm-cpu-gate-macos-15` |
| Apple M4, bare metal, 10 cores | macOS, Python 3.13.15 (pixi) | `--target-cpu apple-m1`, `mojo build -j 1`, Mojo 1.0.0, commit `68be1c79` (main) | **144 of 144** (all 128 training steps and 16 held-out batches) on both paths; the full user path through `from_checkpoint` passes | yes, 65 of 144 on both paths | `bench/results/local/2026-09-11_1436-apple-m4-byte-lm-cpu-certify` |

The GitHub rows are run
[34624545221](https://github.com/mojolearn/mojolearn/actions/runs/34624545221)
on virtual machines; each runner built its own binding. The Apple row is a
virtualized M1, not bare metal.

**Logits agree across every certified CPU at full resolution.** SHA-256 of the
float32 logits on the final parameters, identical on the Intel droplet, the
AMD Zen 2 droplet, the AMD EPYC 9V45, the Neoverse N2 and the Apple M1, and on
the threaded path wherever it ran: held-out batch 00 `[2, 32]`
`2e2408f47392b4a1b69a791cf4672cdf6e49fcc0f576e9bcd6b7018f0ffaa6c4`, its first
token alone `[1, 1]`
`8f46027a73524437c25bc9988e309fdc3f37a4210447a56577f2ed1a1401e839`, and its
first row `[1, 32]` (GitHub rows) `f54cc3905b65d434...`.

Operational timing, not a benchmark and not comparable across rows (different
machines and load): one `[2, 32]` forward plus the loss took 29 to 56 ms on the
Intel droplet and 8 to 24 ms on the GitHub runners through the Python surface.

The runs on commit `205b22fa` (main merged into the lane) repeat the Apple M1
row, and all ten x86-64 draws pass. In both of those runs the ARM64 job stopped
in its "Runner facts" step before building: the step shell is `bash -eo
pipefail` and ARM64 `/proc/cpuinfo` has no `model name` line. That is a
workflow defect, fixed in commit `155f6195`, and not a result for ARM64 either
way. The rerun on `155f6195` (run
[34627322685](https://github.com/mojolearn/mojolearn/actions/runs/34627322685),
evidence `bench/results/gh-actions/2026-09-11_1722-byte-lm-cpu-gate-run34627322685`)
is green on all seven runners, ARM64 included: 33 of 33 on both paths, the
control caught, plumbing tests 8 passed, the same logits hash.

The run on main `b06cd442` (run
[34627708675](https://github.com/mojolearn/mojolearn/actions/runs/34627708675))
is green on all seven runners, with an Intel Xeon Platinum 8370C among its
x86-64 draws.

**Bare-metal Apple M4, every step.** On main `68be1c79`, built on the M4 with
one compiler job at the lowest priority and gated under `taskpolicy -c
background`, the CPU binding reproduces all 144 recorded loss bytes (every one
of the 128 training steps, not every 8th, and the 16 held-out batches) on the
reference and threaded paths, and the reversed-fold control changes 65 of them
on both. `certify_user_path.py` then takes the path a user takes:
`LanguageModelInference.from_checkpoint` on the run's `final.checkpoint.json`
yields exactly the step-128 parameters, all 8 held-out losses match their
recorded bytes on both paths, their `math.fsum` mean equals the recorded
2.8436418771743774 exactly, the logits hash is `2e2408f4...` on both paths, the
package imports through the CPU-only path (`vendor()` is `cpu`), and an
out-of-range token, an over-length input and a wrong loss shape are refused.
The out-of-range token is refused by the native binding as a bare `Exception`
rather than a `ValueError`.

Not measured here: a Qualcomm CPU (no rentable cloud offers one), an installed
wheel (the CPU binding is not packaged yet), and a binary built on one CPU and
run on another. The Intel
droplet leg for commit `5e10863b`
(`bench/results/e1g/2026-09-11_165526-cpu-intel-byte-lm-host-threads`) never
built: cloud-init held the apt lock, `build-essential` did not install, and the
link step found no C compiler. That is an infrastructure failure with no
numerical result in either direction; the leg body waits for cloud-init. The
threaded path on Intel is measured instead on the Xeon 6973P-C row above.

## Threaded path kernels by commit (DEVIATION 2640)

The table above certified the threaded path as it was before DEVIATION 2640,
one task per batch row running the oracles. The kernels are certified
separately, by commit.

The lane commits `6349278a`, `185ad160` and `450addf1` name these kernels
DEVIATION 2624. Main's DEVIATION 2624 is the pointwise histogram fix, which
landed first, so the kernels are DEVIATION 2640 from the merge on.

| commit | where | gate | kernel check | exhaustive exp and SiLU | evidence |
|---|---|---|---|---|---|
| `6349278a` | GitHub run 34640825397: Neoverse N2, Apple M1 (virtual), EPYC 7763 in four draws, EPYC 9V74 | 33 of 33 on both paths; reversed fold caught, 9 of 33 on both | not yet in CI | not yet in CI | `bench/results/gh-actions/2026-09-11_1948-byte-lm-cpu-gate-run34640825397` |
| `185ad160` | GitHub run 34643134242: Neoverse N2, Apple M1 (virtual), EPYC 7763 in three draws, EPYC 9V74 in two | 33 of 33 on both paths; 9 of 33 caught on both | PASS | 0 of 4294967296 differ, at 8 lanes on x86-64 and 4 on ARM64 | `bench/results/gh-actions/2026-09-11_2014-byte-lm-cpu-gate-run34643134242` |
| `450addf1` | Apple M4, bare metal, `--threads 3` | 144 of 144 on both paths; 65 of 144 caught on both; user path through `from_checkpoint` PASS | PASS | 0 of 4294967296 differ | `bench/results/local/2026-09-11_1618-apple-m4-byte-lm-cpu-speed` |
| `450addf1` | GitHub run 34643527802: Neoverse N2, Apple M1 (virtual), EPYC 9V74 in three draws, EPYC 9V45, EPYC 7763 | 33 of 33 on both paths; 9 of 33 caught on both; plumbing tests 9 passed | PASS | 0 of 4294967296 differ, at 8 lanes on x86-64 and 4 on ARM64 | `bench/results/gh-actions/2026-09-11_2018-byte-lm-cpu-gate-run34643527802` |

Not measured with the kernels: any Intel CPU (no Intel host was drawn in
these runs) and the DigitalOcean droplets.
