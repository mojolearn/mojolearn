# AMD step time at the GPT-3 Small shape, 2026-09-24

Branch `lane/amd-step-time`. The problem: one optimizer step of the T3 run
(162,147,840 parameters, batch 4, length 2048, vocabulary 50,257, K=64 shards
of 8,192 tokens) took 139 s on one MI325X against 39.9 s on one H100. The
lane's target: under 60 s on AMD with identical bits.

Measured on Hot Aisle 1x MI300X VMs (gfx942, 304 CUs, ROCm 6.4.1 container
`rocm/dev-ubuntu-22.04:6.4.1-complete`, 13 cores), two legs through
`tools/hotaisle_leg.sh` (60-minute cap, dead-men, verified delete), every
binding built from this branch's source on the box with
`MOJOLEARN_GPU_ARCHS=gfx942`, `MOJOLEARN_TARGET_COLUMN=amd`, IDENTICAL.

## Result

| binding | what changed | s per optimizer step (steady) | chain replay against the H100 chain |
|---|---|---|---|
| baseline | main's GEMM kernels (`-D MOJOLEARN_GEMM_NO_LAUNCH_BOUND=1`) | 141.0 | PASS, steps 101 to 102 from ckpt 100 |
| branch | GEMM launch bound | 62.7 | PASS, steps 101 to 103 from ckpt 100 and 1999 to 2000 from ckpt 1998 |
| leafsplit | GEMM launch bound + AMD leaf-split dispatch | 57.5 | PASS, steps 101 to 103 from ckpt 100 |
| ftz (leg 4) | + `ftz` spelled as one class compare in AMD device code | 52.6 | PASS, steps 101 to 103 and 1999 to 2000 |
| bswz (leg 5) | + NVIDIA's attention block map `_bswz` as AMD's default | 49.1 | PASS, steps 101 to 103 |
| mfma (leg 6) | + the TUNED GEMM calls on the matrix cores | 39.5 | PASS, steps 101 to 103 and 1999 to 2000 |
| mfma groups (leg 7, the branch head) | + the matrix-core group launch (leaf groups of at least 4) | **32.3** | PASS, steps 101 to 103 and 1999 to 2000 |

Same VM, same leg (leg 2), `tools/lm_segment.py run --no-checkpoints
--expect-chain`, the published T3 checkpoints (sha256 80cd2126... and
b8d98090..., equal to `bench/results/lm_t3_2026-09-23/ledger.json`), the
pinned FineWeb-Edu stream, recipe sha256 9f7f695b.... The first step of each
replay carries setup (66 to 72 s). Host hashing adds 7.0 s a step on this
13-core VM (1.6 s on the DigitalOcean MI325X host in T1), outside the step
seconds above. The 141.0 s baseline on this MI300X matches the 138.5 to 141.9
s T1 measured on the MI325X.

Every replayed step's line equals the H100 run's (A/1 and A/2 chains): state
digest, gradient digest, the 64 shard losses and the learning-rate bits. The
state digests at steps 101, 102, 103 are abc8b816b5c3fb15, a9421f91b947f82c,
fcdb48b8ab51f2ef on all three bindings; at 1999 and 2000 dcb05e4e668a81e1 and
0e39ed2bfe9bcbae.

The ftz, bswz and mfma rows ran on later VMs of the same host type (legs 4,
5 and 6); the lean B4 step on the same VM as its predecessor: 0.896 -> 0.819
s (ftz), 0.818 -> 0.763 s (bswz, trial build against itself), 0.763 -> 0.617
s (mfma), 0.617 -> 0.504 s (mfma groups, leg 7). The H100 takes 39.9 s a step
on one GPU (T1); the MI300X now takes 32.3 s. Every lean step wrote
the same six witnesses (loss, gradients, parameters, m, v, flags) on every
build of every leg (`lean-*/result.json`).

## What was wrong: the GEMM kernels spilled their registers on gfx942

Neither step GEMM kernel declared a launch bound, so the gfx942 backend
compiled them for 1,024-thread blocks (the default flat work-group size) and
budgeted 128 VGPRs a lane. Both kernels launch 256 threads and hold a 128x128
output tile, 64 accumulators a thread. The emitted code spilled
(`asm/*/resources.txt`, `tools/amd_codegen/probe_step_gemm.mojo`):

| kernel | `.vgpr_spill_count` before | after |
|---|---|---|
| `identical_gemm_tuned_kernel` 128x128 | 630 | 0 |
| `identical_gemm_kpack_kernel` all leaves | 396 | 0 |
| `identical_gemm_kpack_kernel` group (ksplit) | 390 | 0 |

The fix is `@__llvm_metadata(MAX_THREADS_PER_BLOCK_METADATA=...)` at the real
launch size, 256 (`GEMM_LAUNCH_BOUND` in `gemm/checks/gemm_identical.mojo`,
on the tuned, kpack, arm and ksplit kernels). Register allocation only: no
operation, operand, rounding or order changes. It explains the 2026-09-18
itemization's "flat 2.72 TFLOP/s whatever the shape": a per-product-step cost
from scratch traffic, not the seam.

The second change is a schedule. With the spills gone, the `ksplit_leaf`
geometry (the 128x128 group kernel over the finest leaf groups the workspace
cap allows, then the group fold; DEVIATION 2591's arm, unchanged) beat the
shipped AMD dispatch on the calls it takes. `lib_gemm_leaf_split_for` in
`checks/kernel_matrix.mojo` makes it the AMD default (every other column
False; `-D MOJOLEARN_GEMM_NO_LEAF_SPLIT=1` reverts). A group size reaches no
leaf boundary and no tree level.

## Per kernel, one shard (timed B4 step, random-init weights)

`tools/lm_step_memory_probe.py --shape 4 2048 768 12 12 64 2048 12 50257
--component-timing` on a timers build (`-D MOJOLEARN_STEP_PHASE_TIMERS=1
-D MOJOLEARN_ATTN_PHASE_TIMERS=1`), folded by
`tools/amd_step_timing_summary.py`; ms per shard (the second timed shard; the
timers synchronize every phase, so the envelope is a little above the untimed
step). Untimed lean B4 step: 0.990 s (launch bound), 0.895 s (leaf split).

| component | calls a shard | baseline | launch bound | + leaf split |
|---|---|---|---|---|
| whole shard (envelope) | 1 | 2204.6 | 988.0 | 903.3 |
| GEMM, all calls (norm dW included) | 279 | 1891.6 | 678.2 | 594.8 |
| of which head dA (8192x768, k=50257) | 1 | 248.9 | 80.2 | 58.1 |
| head forward (8192x50257, k=768) | 1 | 170.3 | 56.4 | 56.6 |
| head dB (50257x768, k=8192) | 1 | 169.4 | 55.1 | 54.6 |
| gate/up dA | 24 | 241.3 | 77.0 | 62.2 |
| gate/up forward | 24 | 192.9 | 62.0 | 61.8 |
| gate/up dB | 24 | 125.0 | 70.5 | 62.9 |
| q/k/v/o dA | 48 | 193.6 | 61.7 | 49.4 |
| q/k/v/o forward | 48 | 171.4 | 59.6 | 46.8 |
| q/k/v/o dB | 48 | 87.2 | 49.1 | 47.8 |
| down forward / dA / dB | 12 each | 119.6 / 108.9 / 61.8 | 38.3 / 30.9 / 35.8 | 30.7 / 30.9 / 31.5 |
| attention forward (`fwd_r2`) | 12 | 72.7 | 73.4 | 72.6 |
| attention dk/dv (`kvgrid_dkdv_pf`) | 12 | 71.1 | 67.9 | 68.0 |
| attention dq (`dq_tiled_pf`) | 12 | 65.7 | 66.2 | 65.2 |
| attention zdot (`zdot_estash_dres_pf`) | 12 | 34.2 | 34.8 | 34.0 |
| cross entropy forward + backward + scans | 1 | 16.9 | 16.8 | 16.8 |
| embedding backward | 1 | 8.7 | 8.5 | 8.8 |
| RMSNorm forward/backward kernels | 48 | 12.7 | 12.4 | 13.1 |

Baseline and launch bound: leg 1 VM; leaf split: leg 2 VM (same host).

## The bit proofs

1. GEMM A/B at the T3 shapes (`bench/gemm_excp_ab_main.mojo`, the twelve
   call kinds of a shard through the shipped entry points, three operand
   kinds including one that drives subnormal products): the launch-bound
   build's output hashes equal the baseline's on all 28 cases (leg 1,
   `ab/base.hashes`, `ab/lb.hashes`); every trial arm and the leaf-split
   build equal the branch on all 12 ordinary cases (leg 2, `ab/*.hashes`).
2. The chain replays above (PASS, H100 chain lines).
3. The identity lanes on AMD against the shipped reference table, leaf-split
   binding (leg 2, `verify-leafsplit.log`): byte-lm, byte-lm-resident,
   transformer, embedding, mlp, cross-entropy-arms, training-primitives,
   grad-accumulation, optim-adam-clip, gemm-pinned, gemm-transposed,
   gemm-bf16, gemm-int8, ols, ridge, pca, tsvd, logistic, kmeans: 19 lanes
   VERIFIED, 573 cell parts IDENTICAL, 0 DIVERGENT, 0 OWED;
   language-model-config REFUSED (it needs the host binding, not built).
   The gemm-* and core lanes ran on base/linalg bindings with the launch
   bound; the others on bindings with both changes.

4. Leg 3 (a fresh VM, every device binding built from the branch head, the
   runner's gates on): `gemm_device_check` all green (8 gates: oracle
   agreement, launch and batch invariance, the default dispatch, 5 shapes
   bit-identical to FLAT and the old plan); `gemm_backward_check` all green
   (10 gates); `gemm_workspace_check` PASS (9 GEMMs, 4,608 cells bitwise equal
   to the host oracle; it needs `-D MOJOLEARN_STEP_PHASE_TIMERS=1`); the AMD
   card built (`amd.card`). Then `python -m mojolearn verify` over the 201
   non-par lanes that reach `gemm/checks/gemm_identical.mojo`, in chunks of
   25 (`verify/chunk*.log`): **181 lanes VERIFIED, 6,813 cell parts
   IDENTICAL, 0 DIVERGENT, 0 OWED; 20 REFUSED**, every refusal a binding this
   leg did not build (the byte LM, tokenizer and saved-model HOST bindings,
   samba, the random forest family, arima, and a few gbdt/gp/metrics/ols
   variants), none a disagreement: arima, byte-lm-host-infer,
   byte-lm-host-infer-threaded, byte-lm-host-train, gbdt-catboost-defaults,
   gbdt-multiclass-defaults, gbdt-stochastic-arms, gp-normalize-y,
   language-model-config, metrics-classification, ols-weighted, rf-clf,
   rf-clf-balanced-parallel, rf-clf-entropy-log2-noboot, rf-reg-poisson,
   rf-score-weighted, samba, samba-bf16w, samba-int8w,
   samba-untied-dropout-accum, saved-model-host-infer, tokenizer (the
   per-chunk summaries count 20; the list is the union of names the reports
   print).

## Two more changes, both bit-for-bit the same function

- `checks/numerics.mojo::ftz` in AMD device code: one `v_cmp_class_f32`
  (the two subnormal classes) and a select of the signed zero, the spelling
  the GEMM seam already used, instead of the integer test. Same word for every
  input. It is in every IDENTICAL kernel (attention chains, norms, loss, the
  GEMM fold), which is why it moved the whole step: 57.5 -> 52.6 s. Leg 4
  re-ran the 201 GEMM-reaching lanes on this default: 181 VERIFIED, 0
  DIVERGENT, the same 20 refused. `-D MOJOLEARN_FTZ_NO_CLASS=1` reverts.
- AMD's attention default arm takes `_bswz` (DEVIATION 2900, NVIDIA's
  default since 2026-09-17): a bijection over `block_idx.x` that hands the
  causal tiles out heaviest first. 52.6 -> 49.1 s.

## Where the time goes now (rocprofv3, leg 5, one lean B4 step on the ftz build)

`prof/lean_kernel_stats.csv` (two steps traced, per step below): the ksplit
group GEMM kernel 439 ms (253 launches), the three head GEMMs not split 55 +
53 + 8 ms, the group folds 19 ms, the four attention kernels 54 + 53 + 51 +
28 ms, everything else about 30 ms. Counters on three GEMM calls
(`prof/pmc-*`): the group kernel issues about 8.4 VALU instructions per
product and a wave's VALU issue is about 0.7 instructions a cycle, at one wave
per SIMD.

## The matrix cores (legs 5 to 7): 49.1 -> 39.5 -> 32.3 s

The contract step is `ftz(fma_rn(a, b, acc))`, one product per step. Two
device facts make it a matrix-core instruction plus one VALU product:

1. `v_mfma_f32_32x32x1f32` has K = 1, so every output it writes is one
   product plus its accumulator. It returned exactly the VALU `fma_rn` on
   8,652,800 elements (subnormal results included), in IEEE mode and with the
   wave's MODE flush set (`gemm/checks/amd_mfma_probe.mojo`).
2. With the MODE f32 FP_DENORM field at 2, a VALU product by one returned
   `ftz(x)` exactly on 8,388,608 elements (signed zero included) while the
   MFMA kept its unflushed IEEE result; at 3 (the default) the product does
   not flush, and at 0 or 1 the MFMA flushes too
   (`gemm/checks/amd_mfma_probe2.mojo`, `MFMA2_MODE` lines). The class test
   the staging flush uses reads the same under every mode.

`identical_gemm_mfma_kernel` (`gemm/checks/gemm_identical.mojo`, row
`lib_gemm_mfma_for`, AMD only, `-D MOJOLEARN_GEMM_NO_MFMA=1` reverts) keeps
the tuned kernel's staging, windows, pages, barriers, leaf partial, local fold
stack and output seam; per `p` ascending each wave issues two MFMA steps
(its 64x64 quarter of the 128x128 tile) and multiplies both 32-float
accumulators by `one`, a kernel argument equal to 1.0 so the compiler cannot
fold the product; MODE is set once at kernel entry. The output layout was read
off the device (`MFMA_LAYOUT`). Small outputs split their leaves over the
grid (the tuned kernel's SPLIT mode, the same fold kernel); the long-k weight
gradients stay on the leaf split, which priced faster for them.

Proofs: 28 of 28 T3-shape GEMM hashes identical to the VALU kernels, the kind
that forces subnormal intermediates included (`ab/leg6-*.hashes`); the lean
B4 witnesses identical; the replays PASS at 39.5 s a step; on a rebuild of
every device binding with this default: `gemm_device_check` (8 gates),
`gemm_backward_check` (10 gates) and `gemm_workspace_check` (4,608 cells
against the host oracle) green, and the 201 GEMM-reaching lanes 181
VERIFIED, 0 DIVERGENT, the same 20 refused (`verify6/`).

Leg 7 gave the kernel the packed kernel's GROUP mode (a block walks a
power-of-two group of leaves aligned at leaf 0, folds them on its local stack
and stores the group's node unflushed; `_ksplit_fold_launch` folds the
nodes), with the group size from the leaf split's rule and a floor of 4
leaves (`GEMM_MFMA_MIN_GROUP_LEAVES`, priced at 1, 2, 4 and 16). Every build's
28 hashes (floors 1, 2, 4, 16 and no groups) equal the VALU build's; the
weight gradients went from 2.6 to 1.4 ms and head dA from 43.8 to 30.4 ms;
the step from 39.5 to 32.3 s with both replays PASS; on a rebuild of every
binding: device, backward and workspace checks green and the 201 lanes 181
VERIFIED, 0 DIVERGENT (`verify7/`).

Per call (ms, MI300X, one of each at the T3 shape, before -> after): proj
forward 0.99 -> 0.53, proj dA 0.97 -> 0.53, proj dB 0.95 -> 0.51, gate/up
forward 2.56 -> 1.32, gate/up dA 2.66 -> 1.73, down forward 2.62 -> 1.73,
down dA 2.66 -> 1.42, head forward 54.9 -> 28.6, head dA 56.5 -> 43.8, head
dB 54.1 -> 31.4; the weight gradients with 64 leaves stayed on the leaf split
(2.6 ms).

## Tried and not taken (all bit-identical where they ran)

- EXCP seam (bare FMA, the wave's sticky TRAPSTS exception bits as the
  subnormal detector): the bits are never recorded on this device, not even
  for an FMA consuming a subnormal (`probe.log`, TRAPSTS reads 0x80000000 on
  every case). Dead; the code was replaced.
- DETECT seam (bare packed FMA plus a per-thread class test, exact recompute
  where it fires; `lib_gemm_detect_seam_for`, trial define): the hot loop drops
  from 3.6 to 1.5 VALU slots per product step, bits equal (24 of 24), but it is
  slower than the plain seam once the spills are gone (proj forward 1.26 ->
  1.64 ms) and its recompute makes operand words near 2^-110 cost 4x to 16x.
  So the loop is not issue-bound at one wave per SIMD. Off by default.
- The same launch bound on the seven attention kernels: slower (forward
  73.4 -> 77.4, dq 66.2 -> 74.1 ms). Not applied.
- The packed (kpack) body in the leaf split instead of the ksplit body: bits
  equal, but slower (lean 0.763 -> 0.816 s, 49.1 -> 52.4 s a step, replay
  PASS). Trial define `MOJOLEARN_GEMM_LEAF_SPLIT_KPACK_BODY`.
- Attention arms `_kvgrid_r64` (2.4 to 2.7 s lean), `_estash` without
  `_dres` (equal), `_fgrid_r64` and `_kvsplit` (not valid with this word).
- One LDS page, the TUNED 64x64 plan and the packed body for AMD's k=768
  calls, and every GEMM step arm (`half`, `half_ks16`, `quarter`, `lfold`,
  `kpack`, `kpack_wide`, `kpack_hg`, `ksplit`, `kfoldv`): 207.7 to 581.0 ms
  for one of each call against 212.9 (branch) and 189.4 (`ksplit_leaf`,
  taken).
- `rocdl.waves_per_eu` cannot be spelled through `__llvm_metadata` (the
  lowering wants an integer attribute; scalars and arrays are refused), so the
  kernels still run one wave per SIMD (260 to 290 VGPRs with AGPRs).

## The remaining gap to the H100, kernel by kernel

At 32.3 s a step the MI300X is ahead of one H100's 39.9 s (0.504 s a shard
lean against about 0.62); two H100s take 19.5 s. The paragraphs below describe the 57.5 s state, kept
for the record; since then the ftz and bswz changes took about 0.13 s a shard
from attention, the norms and the GEMM fold, and the matrix cores about 0.15
s a shard from the GEMM. What is left on AMD per shard (leg 5 trace, before
the matrix cores): attention about 0.19 s of kernel time (its chains still
flush on the VALU; the same MFMA-plus-product spelling applies to its dot
chains and is the next lever), the remaining VALU GEMMs (the 64-leaf weight
gradients), then launches, scans and copies. GEMM is 595 ms of it (6.07 TFLOP, 10.2 TFLOP/s); attention
about 250 ms; everything else about 60 ms. On the H100 the 2026-09-17 B1
itemization (`history_h100_b1_breakdown_2026-09-17.tsv`) put GEMM at 58
percent and attention at 29 percent of the step. What is left on AMD:

- GEMM: every product step still carries the post-round class flush (1 FMA
  per two products packed, then a compare, a mask and a select per product,
  3.6 VALU slots against NVIDIA's 2), and the kernels run at one wave per
  SIMD (260 to 290 VGPRs, 33 to 41 KB of LDS a block), so latency is not
  hidden. The DETECT experiment shows fewer VALU slots alone do not help; the
  next lever is occupancy (a register tile that fits 256 VGPRs, or a way to
  set waves-per-EU) and then the seam.
- Attention: 250 ms a shard, flat through both changes; its chains use the
  software flush (`_step`, 8 issue slots a product on AMD). Not attempted
  beyond the launch bound.
- The per-step host hashing on the 13-core Hot Aisle VM (7.0 s) is outside
  the step seconds; on the DigitalOcean MI325X host it is 1.6 s.

## Owed

- NVIDIA re-proof before any release: `GEMM_LAUNCH_BOUND` is in shared
  source and emits `.maxntid 256` on NVIDIA (the kernels' real launch size;
  the leaf split and every other change is AMD-only). NVIDIA bits and speed
  are not re-measured in this lane (no NVIDIA rental).
- The MI325X confirmation of the step time (this lane measured on MI300X).
- The 20 refused lanes of leg 3 (their bindings), and the 59 par-* drivers
  (two devices), on AMD.
- A release (0.8.18) carrying the new bindings.

## Costs

Leg 1 $2.24 (48 min), leg 2 $2.19 (47 min), leg 3 $1.90 (41 min), leg 4
$1.75, leg 5 $1.59, leg 6 $2.05, leg 7 $1.64; Hot Aisle balance $44.65 ->
$30.60 ($14.05 in all).

## Files

`RESUME.md` (the lane's running state), `kernel_map.md` (written before any
rental), `history_*_breakdown_*.tsv` (the two B1 itemizations recovered from
git history), `lanes_gemm_nonpar.txt`, `legs/<stamp>-hotaisle-mi300x-leg*/`
(each leg's runner record and `remote/amd-step-time/`: `session.txt`,
`probe.log`, `asm/`, `ab/`, `item-*.summary.tsv`, `lean-*/`, `replay-*.log`,
`verify-*.log`). Tools: `tools/amd_step_time_leg{1,2,3}.sh`,
`tools/amd_step_time_session.sh`, `tools/amd_step_time_urls.py`,
`tools/amd_step_timing_summary.py`, `tools/amd_codegen/probe_step_gemm.mojo`,
`tools/amd_codegen/lb_test.mojo`, `gemm/checks/amd_excp_probe.mojo`,
`bench/gemm_excp_ab_main.mojo`.
