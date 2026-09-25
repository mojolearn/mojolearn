# Release 0.8.19 proofs: the merged source on an MI300X and an H100, 2026-09-25

Branch `lane/release-0819`, from origin/main 5e33d2db3 (0.8.18) with
`origin/lane/nvidia-step-time` (bc051265b) and
`origin/lane/amd-step-time-2-proof` (72cb886ed) merged. The two merges had
no textual conflict. One fix on top (6528c625e): both lanes added the same
`from std.gpu.primitives.warp import shuffle_xor` line to
`gemm/checks/gemm_identical.mojo`; the second copy is removed. Every default
is as each lane left it: NVIDIA GEMM window admission and the 128x64 kpack
tile under a 512 bound (NVIDIA rows), attention forward and dq launch bounds
(NVIDIA 1024 and 768, every other column 1024), the one-block embedding
run-start scan (every column), and the AMD matrix-core GEMM exact admission
(AMD only). The AMD attention matrix-core trial kernels
(`MOJOLEARN_ATTN_DQ_MFMA`, `MOJOLEARN_ATTN_DKDV_MFMA`,
`MOJOLEARN_ATTN_FWD_MFMA`) are off and no build here turned them on.

Both legs shipped commit dc38e4c64 (the merge plus the leg bodies) and built
every binary on the box from that source, IDENTICAL mode, Mojo 1.0.0
(ed45d567). Bodies: `tools/release_0819/amd_leg.sh`,
`tools/release_0819/nvidia_leg.sh`. Inputs: the published T3 checkpoints
(`ckpt_00000100.blm` sha256 80cd2126a89ba6d8..., `ckpt_00001998.blm`
b8d980905332fa84...), recipe sha256 9f7f695b9a0bae17..., the pinned
FineWeb-Edu stream, and the H100 run's A/1 and A/2 chain files (read only;
nothing of the T3 run, its box or its R2 prefix was written or touched).

## Verdicts

| check | MI300X (Hot Aisle, gfx942) | H100 80GB HBM3 (RunPod, sm_90a) |
|---|---|---|
| replay 101..103 from ckpt 100 against the H100 A/1 chain | PASS, every line equal | PASS, every line equal |
| replay 1999..2000 from ckpt 1998 against the H100 A/2 chain | PASS, every line equal | PASS, every line equal |
| GEMM A/B hashes at the T3 shapes | 64 of 64 lines (ordinary, tiny, mixed, skew, border, sparse) byte-identical to the VALU kernels (`-D MOJOLEARN_GEMM_NO_MFMA=1`); every rehash equal | 36 of 36 (ordinary, tiny, mixed) equal to 0.8.18's NVIDIA GEMM (`-D MOJOLEARN_GEMM_NO_WINDOW_ADMIT=1 -D MOJOLEARN_GEMM_NO_KPACK_NARROW=1`); 36 of 36 more (skew, border, sparse) equal; the admission sabotage DIFFERS on 12 lines (the check can fail) |
| the same hashes across vendors | the 64 MI300X lines appear, hash for hash, among the H100's 72 | |
| `gemm_device_check` | all green, 8 gates | green (the runner's gate, before the body) |
| `gemm_backward_check` | all green, 10 gates | all green, 10 gates |
| `gemm_workspace_check` | PASS, 9 GEMMs, 4,608 cells bitwise equal to the host oracle | PASS, the same |
| lean B4 witnesses (3 steps) | equal to the known digests | equal to the known digests |
| 201 GEMM-reaching non-par lanes vs the shipped reference table | 181 VERIFIED, 6,813 cell parts IDENTICAL, **0 DIVERGENT**, 0 OWED, 20 REFUSED | 181 VERIFIED, 6,813 IDENTICAL, **0 DIVERGENT**, 0 OWED, 20 REFUSED |

No DISAGREE, no DIVERGENT cell and no moved hash anywhere in either leg.

The 20 refused lanes are the same list on both boxes and the same list as
the AMD pass-2 leg and the NVIDIA lane's legs 3 and 4: bindings or parts
these legs do not build (byte-lm-host-infer, byte-lm-host-infer-threaded,
byte-lm-host-train, gbdt-catboost-defaults, gbdt-multiclass-defaults,
gbdt-stochastic-arms, gp-normalize-y, language-model-config,
metrics-classification, ols-weighted, rf-clf, rf-clf-balanced-parallel,
rf-clf-entropy-log2-noboot, rf-reg-poisson, rf-score-weighted, samba,
samba-bf16w, samba-int8w, samba-untied-dropout-accum,
saved-model-host-infer). None is a disagreement.

### Replayed digests (identical on both vendors, equal to the H100 chain)

| step | state | gradient | loss | lr bits |
|---|---|---|---|---|
| 101 | abc8b816b5c3fb15 | 25830bfc2016dc14 | 6.9004 | 397e2cc1 |
| 102 | a9421f91b947f82c | 94c40a6e3d5ec5aa | 6.8585 | 39805880 |
| 103 | fcdb48b8ab51f2ef | 19a43804ef4065de | 6.8373 | 39819a9f |
| 1999 | dcb05e4e668a81e1 | 6170c58b93c1ec4a | 3.5280 | 39e5f6c0 |
| 2000 | 0e39ed2bfe9bcbae | 7c100f6927db84d3 | 3.5538 | 39e5e0ce |

Every replayed step's line (state, gradient, the 64 shard losses, the
learning-rate bits) was checked by `tools/lm_segment.py run --expect-chain`.
Lean B4 witnesses on both: parameters 5516ffe5f550 / 77477af42588 /
4e439a8a9751, loss 676298dabb30 / afc46227a372 / 34fe4c49b0dd.

## Seconds an optimizer step (T3 shape: 64 shards of 8,192 tokens, batch 4, length 2048, 162,147,840 parameters)

| box | steady steps | s a step | lean B4 shard | first step (carries setup) | host hashing (outside the step) |
|---|---|---|---|---|---|
| MI300X, merged | 102, 103, 2000 | 29.12, 29.08, 29.12 | 0.453 s | 37.42, 37.45 | about 7.0 s |
| H100, merged | 102, 103, 2000 | 30.66, 30.66, 30.79 | 0.481 s | 39.89, 39.77 | 7.1 to 7.6 s |

For reference, from the lanes' own legs (other VMs and pods): 0.8.18 took
32.3 s a step on an MI300X (AMD first pass, leg 7) and 38.83 s on an H100
(NVIDIA lane, leg 1); the AMD pass-2 admission build took 29.7 s, the NVIDIA
lane's head 30.67 s. These are per-box measurements of each vendor against
its own earlier build, not a comparison between vendors.

## Apple

Compile only, and partial. The RunPod CPU pod refused three creates ("no
longer any instances available", 8 and 16 vCPU, cpu3c/cpu5c/cpu3g/cpu5g/
cpu3m/cpu5m; no pod was created, nothing billed), so the check ran on the
H100 pod's host CPU after its replays, while the lane verification ran:
`tools/release_0819/apple_compile_probe.mojo` built with
`-D MOJOLEARN_COLUMN_APPLE` and emitted LLVM (AIR, `air64-apple-macosx`,
target `apple-m4`) for the three changed kernels every column compiles:
`emb_run_begin_block_kernel` (6,342 bytes of IR),
`fused_attn_forward_r2_kernel[64, 32, True, True, False]` (635,372) and
`fused_bwd_dq_tiled_pf_kernel[64]` (451,615). All three compiled
(`nvidia-2026-09-25_095629-runpod-h100/remote/nv-step-time/apple/`).

Not shown by this check:
- The two attention kernels' IR carries the bound as `"nvvm.maxntid"="1024"`.
  The metadata key comes from the host's default accelerator (this host is an
  H100); the AMD first pass's CPU-pod probe of the same decorator for
  apple-m4 printed `"air.max_work_group_size"`. The spelling a Mac build
  emits is therefore not proven here. The value, 1024, is the Metal
  threadgroup maximum.
- A Metal library. The GEMM A/B harness built for `--target-accelerator
  apple-m4` fails on Linux with "Metal Compiler failed to compile metallib"
  (no Metal toolchain off a Mac); that is the environment, not the source.
- Apple bits. No Apple GPU ran. The Apple column is owed on the Mac (one
  Metal job at a time), at least the embedding binding and the byte LM step.

## Costs

| leg | box | window (UTC) | rate | cost |
|---|---|---|---|---|
| AMD | Hot Aisle 1x MI300X (13 core), VM enc1-gpuvm002 | 09:56:46 to 10:32:22 (35.6 min), verified gone (404) | $2.99 an hour | about $1.77 |
| NVIDIA | RunPod H100 80GB HBM3, pod 4ao17wbwuf67u3 | 09:56:57 to 10:35:19 (38.4 min), verified gone (404) | $3.49 an hour | about $2.23 |
| Apple | RunPod CPU pod | not created (no stock) | | $0 |

About $4.00 in all. The Hot Aisle balance read $25.57 before and $48.92
after (a top-up landed during the leg, so the difference is not this leg's
cost).

## Files

- `amd-2026-09-25_095629-hotaisle-mi300x/`: the Hot Aisle runner record
  (`leg.txt`, `runner.console.log`, `teardown.txt`) and
  `remote/amd-step-time/` (`session.txt` is the running record; `ab/`,
  `lean-merged/`, `replay-merged-*`, `gemm_*_check.log`, `verify/`,
  `builds/`).
- `nvidia-2026-09-25_095629-runpod-h100/`: the RunPod runner record
  (`leg.txt`, `runner.console.log`, `teardown.txt`, the runner's card diff
  against the 2026-09-22 Apple card passed with `--local-card`, which is not
  a proof here) and `remote/nv-step-time/` (`session.txt`, `ab/`,
  `lean-merged/`, `replay-merged-*.clean.log`, `gemm_*_check.log`,
  `verify/`, `apple/`).
