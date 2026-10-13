# Neural gaps read (Fable, read-only, 2026-10-09; main @ 444f0b96e)

Board facts read from bench/results/bench_board/main-nvidia-l40s/board.json (ours cells: n0526 main@42d1e42c6 for the
optimizers, n0666 main@432d6e8ff for the layers; torch cells copied from opponents-*-20261006). NV and AMD ours
digests agree on every optimizer lane (identity MATCH recorded in each cell). No box, build or run was touched.

## 1. Optimizers: is the lead real, and why no quality

### What each arm does (tools/bench_board_algos.py)
- Workload: one fp32 parameter of 2^24 = 16,777,216 values, 10 seed-7 gradients (N(0,1) x 0.01), 10 steps
  (`_build_optim` :4064-4070; lane notes :812-813). Hyperparameters explicit and equal on both arms (`_OPT_HYPER`
  :791-803; lr=1e-3 :811).
- Ours (:4072-4102): grads uploaded to the device BEFORE the clock (`to_device(grads[i])`, :4086-4088,
  input_home=device); inside the clock: `p = to_dev(p0)` (one 64 MB upload), `cls([p], **hyper)` (state slots
  opened and zero-filled on the device, sequence/opt_resident.mojo:816-832), then 10 x `opt.step([g])`. Each step is
  ONE Python call -> `optimizer_resident_step_dev` (python/mojolearn/_x_sequence_optim.py:266-281) -> ONE fused
  element-wise launch `OP_OPT` over 16.7M elements at TPB=128 (sequence/opt_resident.mojo:834-843;
  sequence/exec_device.mojo:88, :916-922) + one `ex.sync()` (:844). The bias-correction scalars run on the host
  (`opt_scalars`, :824-826: 3 floats, glue). Kernel branch for NAdam: sequence/ops.mojo:895-909 (lerp, fma, sqrt,
  div, two fma updates: reads p,g,m,v, writes p,m,v = 7 x 64 MB per step).
- Torch (:4103-4125): `torch.optim.X([p], **hyper)` with NO foreach/fused argument (:4120), so torch's default:
  foreach on CUDA, never the fused kernel (torch has fused only for Adam/AdamW/SGD/Adagrad; NAdam, Adamax,
  RMSprop, Adafactor have none). Inside its clock: `torch.from_numpy(p0).to(dev)` (the same 64 MB upload), the
  optimizer construction, 10 steps (state lazily allocated in step 1, as ours), `torch.cuda.synchronize()`.
  Gradients resident before the clock (`g_dev`, :4111). Comparability span stored on the torch cells:
  input_home=device, inside_clock="10 optimizer steps" (board cell).
- Same steps, same parameter, same hyperparameters, both clocks hold the p0 upload and the state allocation. No arm
  does less work. The structural difference is fusion: ours is one pass (7 x 64 MB = 448 MB/step, 4.5 GB per 10
  steps, ~5 ms at the L40S's 864 GB/s + ~3-6 ms for the 64 MB pageable upload + 10 syncs = the 13 ms observed);
  torch's foreach NAdam is ~10 element-wise kernels per step (lerp_, mul_, addcmul_, div, sqrt, add, two addcdiv),
  ~1.5-2 GB/step, 15-20 GB per 10 steps = the 20.8 ms observed. AMD confirms the bandwidth picture (nadam 4.1 ms on
  the MI325X's 5.3 TB/s). Verdict: the 1.3-1.6x lead on nadam/adamax/adagrad/rmsprop is real and expected; it is a
  memory-bound fused update against an unfused reference, not a measurement artifact.
- Adafactor (ours 17.3 vs torch 15.3): ours is slower because a step is ~8 launches (sequence/pyapi.mojo:851-935:
  OP_AF_ALPHA with n=1, a blocked sum-of-squares, OP_SCALE, OP_AF_VEC over n, OP_AF_DENOM with n=1, the apply),
  two of them single-thread launches that serialize the stream; torch's vector path is a similar count of
  foreach ops. Idea: `-D MOJOLEARN_IDN_AF_VEC_FUSED` fold alpha/denom into the first and last element-wise pass
  (the 1-thread launches become block-0 epilogues, same fold order) -> ~17 -> ~10 ms; no bits change if the
  reductions keep the blocked order. Opus-carriable.

### Why "Own host reference failed"
- Quality for an optimizer cell is `rel_fro_vs_torch_eager_fp32` computed ONLY from a torch-eager-fp32 output of the
  same race (:5241-5244). Opponents are never re-raced (stored), so `outs` has no torch arm, quality stays `{}`,
  and `bench_board_host_quality.enrich` runs the own-host reference (:5542-5545).
- `enrich` (tools/bench_board_host_quality.py:40-74) spawns `python bench_board_host_quality.py <lane>` with
  MOJOLEARN_VENDOR=cpu, MOJOLEARN_NUMERIC_MODE=identical, stdout to `host.log`; a non-zero exit becomes the string
  "Own host reference failed; see .../host.log" (:58-59). The subprocess needs the lane's HOST binding
  (`_mojolearn_x_sequence_host.so`, `_mojolearn_x_cnn_host.so`, `_mojolearn_embedding_host.so`; it refuses any
  non-host .so, :88-89) and `MOJOLEARN_VENDOR=cpu` import (python/mojolearn/_backend.py:1270-1312).
- On the box a RACE of main builds only the affected bindings: `affected_builds.py origin/main origin/main` is empty,
  so `builds=build_x_linear` (~/mojolearn-evidence/lq/box_job.sh:19-37). Device bindings a race lacks are built by
  the retry loop, which greps the RACE LOG for `needs python/mojolearn/..._mojolearn_X.so` or `no host binding covers
  _mojolearn_X` (overlay_race_job2.sh:72-82). The host reference's ImportError goes to host.log, never to the race
  log, so no `_host` binding is ever built for a main race and every x_sequence / x_cnn / embedding lane fails the
  same way (all 17 cells above carry the message; the earlier main@de2b2b739 run n0316 too). On the Mac, where host
  bindings exist, the same code path succeeded (full-identical-20261005/apple-fast/f867-fast-smoke/work/
  adam-synthetic-host-quality/: receipt.json + host.npz present, host.log empty). The box's host.log is the proof:
  `lq log nv n0526 host` (not pulled here).
- Fix (test tooling, allowed): (a) overlay_race_job2.sh / box_job.sh build `build_<lane>_host` for every RACE lane
  (ALL_BUILDS.txt lists them), or make `enrich` append `tail -1 host.log` to the quality error so the retry loop
  sees "no host binding covers"; (b) cheaper and stronger for these lanes: compare against the STORED opponent
  output. For embedding the stored torch hash already equals ours (2753e61a6dde36e3 on both arms, sha256 of the
  outputs :4911-4915): bit-identical, quality proven without any reference. For the optimizers the opponents board
  saved `saved_outputs` npz on its box; carrying the torch-eager-fp32 `param` into the opponent store (or its
  hash plus a tolerance class) lets `quality()` run offline. Until one of these lands, the optimizer speed cells are
  honest but unqualified.

## 2. Layers

Shapes: layernorm x (16384, 1024) :774-778; conv2d x (64, 64, 56, 56) C=OC=64 3x3 :1059-1062; resnet-block same x
:1110-1114; moe x (8192, 1024), E=8, F=2816, top-2, forward only :779-788; embedding ids (64, 512) over 32768 x
1024 :1125-1133; batchnorm2d / dropout2d x (64, 64, 56, 56) :1097-1103. Clock: forward + backward(dy) (:1477), dy
from the fixture, read back outside the clock. torch twins: eager/compile x fp32/tf32/bf16 (autocast :2909, tf32
flags :2859-2860); dropout2d has no bf16 arm (:1350-1351), embedding fp32 only (:1352-1353).

### layernorm: 50.6 ms vs 3.45 (14.7x). Two pathologies, both certain from code
1. THE BOARD RACES THE CLASS, NOT THE DEVICE-I/O ENTRY. `ours=("LayerNorm", "layer_norm_forward")` (:774) and
   `_ours_class` returns the FIRST name that resolves (:2392-2404); `mojolearn.LayerNorm` is exported
   (python/mojolearn/_expansion_sequence.py:20, :39), so the lane takes the generic layer path (:3905-3975), not the
   `layer_norm_forward` branch that lane gap-neural-io gave device tensors (:3841-3870). `LayerNorm` has no
   `_device_io` (python/mojolearn/_x_sequence_norm.py:120-150), so the harness hands host arrays and `_run`
   (:68-95) does per fit: `np.zeros` y 67 MB, upload x 67 MB, download y 67 MB; backward: `np.zeros` dx 67 MB,
   upload x and dy 67 MB each, download dx 67 MB (:83-91). Six 67 MB bus crossings and two page-faulted 67 MB
   zeros inside the clock. Evidence: the forward-only infer of the same cell is 20.5-27 ms (nv-grid-logs n0666
   ALGOS-ROUND lines), i.e. one forward already costs 6x torch's forward+backward.
2. THE KERNEL IS ONE THREAD PER ROW. `op_ln_fwd` (sequence/layernorm.mojo:67-82) walks the 1024-float row three
   times serially (sum, variance, write; `_stats` :54-64), `op_ln_bwd_x` twice (:84-115); launched with n = M =
   16384 threads at TPB=128 (exec_device.mojo:88, :916-922) = 128 blocks for 142 SMs, adjacent threads 4 KB apart
   (no coalescing). The backward entry recomputes the forward to get mean/rstd (layernorm_dev.mojo:58-61;
   pyapi.mojo:1215-1229). Only the dweight/dbias fold is wide (blocked: S = ln_fold_rows(M) = 128 splits x 1024
   columns, :117-171). Stage table per fit: launches 1 (fwd) + 1 (fwd again) + 1 (bwd_x) + 2 (bwd_w) = 5; syncs 2
   (one per entry); bytes ~8 row passes x 67 MB uncoalesced; copies inside clock: 6 x 67 MB + 2 zero fills (path 1).
   Ideas:
   - `MOJOLEARN_SEQ_LN_BOARD_DEVICE_IO` (really: reorder the lane tuple to ("layer_norm_forward", "LayerNorm") at
     :774, or give `LayerNorm` `_device_io` + `to_device` and let `_build_layer` pick the sequence `to_device`
     for sequence classes). No bits. Expected 50 -> ~25 ms (removes ~400 MB of copies and the page faults). Opus.
   - `MOJOLEARN_IDN_LN_ROW_WARP`: 32 lanes per row, lane j folds elements j, j+32, ... ascending from +0.0, then a
     5-level fixed-lane-order tree; spelled identically on NVIDIA, AMD (32 logical lanes of the wave64) and the host
     twin. Bits change once (fold order; allowed). Kernel 20 ms -> well under 1 ms. Fable (bit-changing fold + host
     twin + NV/AMD ID check).
   - `MOJOLEARN_IDN_LN_KEEP_STATS`: keep mean/rstd (2 x M floats) from the forward for the backward instead of
     re-running fwd_core. No bits. -1 forward pass. Opus.
   - `MOJOLEARN_SEQ_LN_ROW_BLOCK`: block-per-row (128 threads, shared-memory tree) for D >= 4096; same tree spelling
     as ROW_WARP. Fable. Not needed for the board shape.
   - Single-pass E[x^2]-mean^2: changes numerics (cancellation); not for IDENTICAL; FAST-only candidate at best.

### conv2d: 9.66 vs 2.30 fp32 / 1.33 bf16; resnet-block: 20.96 vs 5.27 / 2.85
- Path: C k k = 576 > DC_MAXK = 32 so no direct conv (x_cnn/device.mojo:337, :318-336); forward = im2col into a
  cached 462 MB workspace (`ws(ctx, 4, rows*ckk)`, :1414, :1454-1462, rows = 200704) + the identical fp32 GEMM
  (`device_gemm` :780-800 -> `identical_gemm_into`, profile fp32.v1, allow_vendor=False: no TF32) + conv_out/bias;
  backward = dout rows, dW as OP_TN split-k fold kernels (:1443-1488), dX GEMM + col2im. x, y, dy, dx resident
  (`_forward_t`/`_backward_t` handles, python/mojolearn/_expansion_cnn.py:548-616); weights (147 KB) up per call.
  FLOPs: 14.8 GFLOP fwd, ~44 GFLOP fwd+bwd -> 4.6 TFLOP/s at 9.66 ms; cuDNN fp32 (also CUDA cores, no TF32 in
  that arm) reaches ~19 TFLOP/s at 2.30 ms. Stage table: ~2 im2col/col2im passes (462 MB write + read each),
  3 GEMMs, ~6 small launches, 1-2 syncs; no host copies. Dominant: GEMM kernel efficiency (4x), then im2col traffic
  (~1 ms). resnet-block = 2 conv fwd+bwd (2 x 9.66 = 19.3 of 20.96) + BN (~7 launches fwd :2088-2107, aux readback
  sync :2111) + ReLU/add map2 launches (BasicBlock._forward_dev, _expansion_cnn.py:2160-2240): BN fuse can save
  ~1 ms of 21; conv is the lever.
- Switches already in main, default OFF, NEVER MEASURED: `MOJOLEARN_NI12_IMPLICIT_CONV` (x_cnn/device.mojo:89,
  :392-433: 64x64 tile, K slab 16, 256 threads, implicit im2col, same (c,kh,kw) k order -> same bits),
  `MOJOLEARN_IDN_CNN_BN_FUSE` (:94-102, same words). Their A/B jobs failed in BOTH arms: n0529/a0876 (RACE,
  "ModuleNotFoundError" at stage=ready: the ours arm imported torch before 432d6e8ff) and gaps2.gemm.A/B n0570,
  n0571, a0885, a0886 (CMD rc=1, GRIDBB-DONE races=3 ok=0). Re-queue them first on current main.
- Ideas beyond: `MOJOLEARN_IDN_GEMM_PLAN_N64`: a plan for tall-skinny n = 64 (128x64 tile, 8x4 micro-tile; the
  contract says the fold is a pure function of k so plan changes keep bits, docs/plans/gaps-2026-10-08.md:68-86).
  Opus if it stays inside `identical_gemm_with_plan`'s leaf structure. `MOJOLEARN_IDN_CONV_WGRAD_TILED`: the dW
  OP_TN over 200704 rows as the GEMM's split plan with wider leaves (bits change once). Fable. Floor: under fp32
  IDENTICAL the tf32/bf16 twins (1.77 / 1.33 ms) are tensor-core work and out of reach; the fp32 twin (2.30) is not.

### moe: 32.5 vs 12.0 fp32 / 4.3 bf16 (forward only)
- Path: python glue allocates four `np.zeros` per call and uploads X (32 MB) / downloads y (32 MB) inside the clock
  (python/mojolearn/_x_sequence_moe.py:104-107, :117-121; no `to_device` in this file: lane gap-neural-io did not
  reach MoE). Kernel: routing + device grouping (exec_device.mojo:700-750, MOE_DEVGROUP) then `moe_reg` register
  tiles 64x64x16, 256 threads, 4x4 scalar fp32 micro-tiles (sequence/moe_reg.mojo:57-75): 283 GFLOP in ~25 ms of
  the 32.5 = ~11 TFLOP/s; cuBLAS SGEMM per expert ~24 TFLOP/s. Stage table: 1 route + 3-4 grouping launches +
  2 product launches; 1 sync; 64 MB host copies + 4 page-faulted zeros (~5-8 ms).
- Switch in main, default OFF, unmeasured (same failed jobs): `MOJOLEARN_IDN_MOE_GROUPED_GEMM`
  (sequence/moe_grouped_fold.mojo:31-35; bits change once, NV+AMD+host twin together).
- Ideas: `MOJOLEARN_SEQ_MOE_DEVICE_IO` (x, y as SequenceDeviceTensor like layernorm's dev entry; no bits; -5 to
  -8 ms; Opus). Re-queue GROUPED_GEMM A/B. `MOJOLEARN_IDN_MOE_REGTILE_128`: 128x128 tile, 8x8 micro-tile; the
  per-cell chain stays ascending over k so bits hold; Opus/Fable medium.

### embedding: 2.65 vs 1.88; OUTPUT BITS EQUAL TORCH (hash 2753e61a6dde36e3 both arms)
- Path: `forward(ids, device=True)` and `backward(ids, dy)` with dy and the (V, d) gradient resident
  (:3882-3900; python/mojolearn/embedding.py:270-345). Per call the owned-runs cache verifies the ids on the
  device and reads a status word back with `ctx.synchronize()` (embedding/owned_runs.mojo:92-104), the range
  refusal does the same (:86-90, `emb_refuse_ids` bindings/_mojolearn_embedding.mojo:1001-1003); backward is
  PLAN_SORT: seed/zero fill of the 134 MB dW (`emb_seed_kernel`), pad row, counts, stable radix sort of the 32768
  keys (several launches), segmented fold (embedding/checks/embedding_identical.mojo:1214-1290;
  embedding_sort.mojo:1-14). Dominant: fixed per-call latency (2+ host round trips, ~10 small launches), not bytes.
- Ideas: `MOJOLEARN_IDN_EMB_IDS_CHECK_ASYNC` (fold the ids-equal and range checks into one launch whose status is
  read at the end of the call, one sync instead of two; no bits), reuse the sorted runs when the ids handle is the
  same resident tensor (no bits). Expected 2.65 -> ~2.0. Opus. (The `MOJOLEARN_IDN_EMB_UNIQUE_ROWS` switch exists,
  its A/B gaps2.nio.B also rc=1.)

### batchnorm2d: 1.64 vs 1.33; dropout2d: 1.36 vs 0.98
- batchnorm forward: blk_sum, blk_mean, blk_sq, blk_var (two-pass variance, blocked fold), apply, running update
  (x_cnn/device.mojo:2088-2107) + aux (2 + 7C floats) read back with `ctx.synchronize()` (:2111); backward ~5
  launches + sync. Traffic ~450 MB (0.5 ms at peak) -> the 1.64 ms is ~12 launches, 2 syncs and glue. Ideas:
  `MOJOLEARN_IDN_BN_AUX_RESIDENT` (aux stays on the device, downloaded on access like Dropout's `mask_`; no bits;
  Opus), `MOJOLEARN_IDN_BN_LASTBLOCK_FOLD` (mean/var finalize in the last-arriving block, partials folded in
  index order: same bits; Fable small). Expected 1.64 -> ~1.2.
- dropout2d: y = `DeviceTensor._new` -> `x_cnn_res_alloc`, ZERO FILLED (device.mojo:1161-1170), one
  `dropout_channel_kernel` launch (:2362); backward another zero-filled dx + 1 launch; freed tensors `res_free`
  synchronize (:42-45). Traffic ~200 MB -> the 1.36 ms is two 51 MB zero fills, syncs on free, glue. Ideas:
  `MOJOLEARN_XCNN_RES_NEW_NOZERO` for outputs every element of which the op writes (no bits),
  `MOJOLEARN_XCNN_RES_FREE_ASYNC` (return to pool without the sync; sync on reuse). Opus. 1.36 -> ~1.0.

## 3. BF16 (Andrew 2026-10-09: vendor bf16 matrix units are NOT identical across NVIDIA and AMD by decision;
## that is IDENTICAL's identity tax; tensor cores are FAST-mode only. No probe is designed here.)

### (2) What runs today, with file:line
- Numeric profile default `fp32_v1` for inference AND training (python/mojolearn/_numeric_profile.py:115-120; env
  MOJOLEARN_NUMERIC_PROFILE :126). Transformer surface fp32 only, "not BF16, FP16, FP8" (python/mojolearn/
  _transformer_impl.py:9-10, :56, :364); Mamba fp32, "a certified bf16 profile is SHIP LATER" (_mamba_impl.py:46,
  :164); checkpoints FP32 only (training/checkpoint.mojo:122). The neural layers and optimizers hold float32 only
  (SequenceDeviceTensor.dtype, python/mojolearn/_x_sequence_device.py:48; x_cnn resident arrays are 4-byte words,
  x_cnn/device.mojo:1161). Packed bf16 weights are MATERIALIZED to fp32 before a block runs (python/mojolearn/
  lowbit.py:10-16, :25): at runtime every model is fp32.v1 on widened weights.

### (1) bf16f32.v1 vs fp32.v1: cost, gain, end to end
- What it is: fp32.v1's fixed-order fma chain on exactly widened bf16 operands (gemm/IDENTICAL_LOWBIT_CONTRACT.md:
  28-33; L-1 widen is a shift, L-2 narrow is RNE after ftz, :48-49). RIGHT operand (weights) only; the left operand
  is float32 (gemm/checks/gemm_lowbit.mojo:380-398). Two plans: fused (widen as loaded, m*n <= BF16W_FUSED_MAX_CELLS,
  :225-245) and widen (a `bf16_widen` launch to a float32 image, then `identical_gemm_into`, :349-366).
- Cost today: UNPRICED on NVIDIA/AMD. Every stored price row for the bf16 arms is LOWBIT-NOT-RUN ("not asked for
  (MOJOLEARN_LOWBIT_PRICE_ARMS)", bench/results/lowbit_amd_tuned/2026-09-29/mi325x/*/lowbit.tsv); only fp32.v1 rows
  carry numbers. Structurally: the widen plan is strictly slower than fp32.v1 (one extra pass over B); the fused plan
  does the same number of fmas on half the B bytes, so compute-bound GEMMs (conv2d, moe, resnet) gain ~0 and only
  memory-bound GEMV (m = 1 decode, weight stream) approaches 2x. The gain today is bytes on disk and in memory.
- End to end on the layers/models: NO. Nothing accepts bf16 activations; attention, layernorm, batchnorm, dropout,
  embedding and the optimizers are fp32 in and out. A "bf16 storage everywhere" IDENTICAL path (bf16 activations
  and grads in memory, every kernel widening on load and narrowing RNE on store, fp32 master params and state in
  the optimizers) would be identical across vendors by construction (IEEE fp32 + RNE, no units) and would halve the
  traffic of the bandwidth-bound layers: layernorm/dropout/batchnorm move ~50-70 MB per tensor pass, embedding's
  dW 134 MB, the optimizers 448 MB per step (p, g, m, v read; p, m, v written) -> ~224 MB with bf16 grads and a bf16
  model copy and fp32 master/state (the usual mixed-precision layout keeps m, v fp32, so the realistic saving is
  the g read and a p copy write: ~25-35%). So: speed on the bandwidth-bound layers, YES, up to ~2x on the kernel
  part, but (a) only after the kernels stop being launch/latency-bound (layernorm's thread-per-row, dropout's
  zero fills, batchnorm's syncs: Section 2), because halving bytes does nothing to a kernel at 5% of bandwidth;
  (b) the optimizer quality column must then compare against a torch bf16 AMP reference, not fp32. Size: a bf16
  tensor type in the sequence/x_cnn/embedding device layers + widen/narrow seams in every kernel (S each, ~15
  kernels), optimizer bf16 copy launch (S), contract clauses and host twins (M): an L, multi-lane program, with
  quality gates against torch bf16 twins (the board holds them). Not recommended before the fp32 fixes land: today
  the bandwidth-bound layers are nowhere near bandwidth.

### (3) The FAST-mode tensor-core path for these layers (FAST = speed + no quality loss; no identity)
- What exists: Apple only. MoE simdgroup MMA (`MOJOLEARN_MOE_FAST_MMA`, sequence/moe_mma.mojo:1-50, FAST+Apple,
  default OFF), the Apple FAST GEMM on bf16 bits (gemm/afn_apple_fast.mojo:8-28, `AFN_GEMM_BF16_MMA`), direct conv
  FAST on Apple (x_cnn/device.mojo:318-336). On NVIDIA/AMD the only matrix-unit kernel in the tree is the int8 IMMA/
  MFMA GEMM (gemm/checks/gemm_int8_mma.mojo:1-30, exact integers), which gives the intrinsic plumbing
  (`llvm.nvvm.mma.*`, `llvm.amdgcn.mfma.*`) a bf16 kernel would reuse. There is no cuBLAS/rocBLAS route (contract:
  "NO REFERENCE FILE ... cuBLASLt, a closed", gemm/checks/gemm_identical.mojo:10; `allow_vendor` in
  gemm/neural_dispatch.mojo:157-242 selects between our own plans). FAST neural on NVIDIA/AMD does not exist yet:
  MOE_REGTILE's FAST clause is Apple-gated (sequence/moe_reg.mojo:57-63) and `_CLASSICAL_FAST` is the classical set
  (python/mojolearn/_backend.py:359-392).
- Size by layer (a `-D MOJOLEARN_<AREA>_FAST_<NAME>`, default off, A/B on nv and amd, quality = rel_fro vs torch
  eager fp32 no worse than the torch bf16 twin's):
  - conv2d / resnet-block (gap 9.66 -> 1.33, 21.0 -> 2.85): a bf16 (or tf32) implicit-GEMM conv on mma.sync
    m16n8k16 bf16 / MFMA 16x16x16 bf16 with fp32 accumulate, built on NI12's A-tile gather (x_cnn/device.mojo:
    392-433) and the int8 MMA kernel's fragment code; forward + dX + dW (three GEMM shapes). M-L, Fable: a new
    kernel family plus the three conv GEMM variants. Expected: within ~1.5-2x of cuDNN bf16 (2-2.5 ms conv,
    ~5 ms resnet) after tuning; BN fuse (IDN_CNN_BN_FUSE words) applies as is.
  - moe (32.5 -> 4.34): the Apple MMA design (64 pairs x BN outputs per block, k-major slabs, moe_mma.mojo:11-24)
    re-spelled on the NVIDIA/AMD intrinsics with bf16 operands, fp32 accumulate; grouped by expert as DEVGROUP
    already does. M, Fable. Expected ~6-9 ms (cuBLAS bf16 per expert reaches 4.3).
  - layernorm, embedding, batchnorm2d, dropout2d: NO tensor-core content. Their FAST path is the same kernel work
    as IDENTICAL (coalesced row-parallel reductions, fewer launches, no syncs, resident tensors) plus optional bf16
    I/O; nothing in them is an identity tax. S-M each, Opus-carriable.
  - optimizers: no matmul; nothing to gain from units (board: "torch-*-tf32: no matmul in an optimizer step",
    tools/bench_board_algos.py:1426-1427).
- Order inside FAST: conv/resnet first (largest absolute tax, one kernel family serves both), moe second.

## 4. Headline ranking: ours (IDENTICAL fp32) against torch's bf16 arm, closable vs identity tax
Closable = what IDENTICAL can take back without vendor matrix units (fixed-order fp32 or bf16f32 storage: bandwidth,
fusion, launch count, resident tensors). Tax = the remaining gap once our kernel runs at CUDA-core fp32 parity with
torch's fp32 arm (the fp32 twin is the fair IDENTICAL target; bf16 minus fp32 on the torch side is the tensor-core
tax, since none of these layers lose quality in bf16 for the customer). Estimates from Section 2's stage tables.

| lane | ours | torch bf16 | gap | closable in IDENTICAL (how) | irreducible tax |
|---|---|---|---|---|---|
| layernorm | 50.6 | 1.87 | 27x | ~48 of 48.7 ms: board path uses host arrays (6 x 67 MB + 2 zero fills) and a thread-per-row kernel at 128 blocks; device I/O + warp-per-row fixed-order fold -> ~2-3 ms (torch fp32 3.45; no matmul) | none: LayerNorm has no tensor-core content; bf16's 1.87 vs fp32's 3.45 is bf16 I/O bytes, reachable with bf16f32 storage later |
| moe | 32.5 | 4.34 | 7.5x | ~20 of 28 ms: device I/O (-5 to -8), grouped identical GEMM / 128x128 register tiles toward cuBLAS SGEMM parity (~12 ms, torch fp32 12.0) | ~8 ms (12.0 fp32 -> 4.3 bf16: tensor cores) |
| resnet-block | 21.0 | 2.85 | 7.4x | ~15 of 18 ms: conv GEMM efficiency 4.6 -> ~15-19 TFLOP/s (fp32 twin 5.27), BN fuse, im2col removal (NI12) | ~2.4 ms (5.27 fp32 -> 2.85 bf16) |
| conv2d | 9.66 | 1.33 | 7.3x | ~7.4 of 8.3 ms: same GEMM work (fp32 twin 2.30) | ~1 ms (2.30 -> 1.33) |
| embedding | 2.65 | 1.88 | 1.4x | ~0.6 of 0.8 ms: one async ids/range check instead of two host syncs, reuse of sorted runs, fewer small launches (fp32 twin 1.88 = bf16: gather has no matmul) | none |
| batchnorm2d | 1.64 | 1.18 | 1.4x | ~0.4 of 0.46 ms: aux resident (no sync), last-block fold, fewer launches (fp32 twin 1.33) | ~0.15 ms (1.33 -> 1.18 is bf16 bytes, not units) |
| dropout2d | 1.36 | 0.98 | 1.4x | ~0.35 of 0.38 ms: no zero fill of fully written outputs, async pool free (fp32 twin 0.98) | none |
| optimizers | 10.7-13.9 | n/a (fp32 only) | ours ahead 1.25-1.6x; adafactor behind 1.13x | adafactor: fuse the 1-thread alpha/denom launches | none (no matmul) |

Reading: ~90% of the summed gap (118 ms of ~130 ms over the seven layer cells) is closable inside IDENTICAL; the
tensor-core tax is ~11-12 ms, concentrated in moe (8), resnet (2.4), conv (1). Do layernorm first (largest, zero
tax, Opus for the I/O half, Fable for the fold), then the conv GEMM (serves conv2d and resnet-block), then moe.

## Owed / not provable from the laptop
- The box host.log for any failed cell (`lq log nv n0526 host`, `lq log nv n0666 host`) to confirm the missing
  `_host` binding mechanism; the kernel-only split of layernorm's 20 ms forward (a CMD timing of `layer_norm` vs
  `layer_norm_dev` would separate copies from kernel); the three gap-gemm-layers switches' A/B (never completed).
- No tensor-core identity probe is proposed (Andrew 2026-10-09: the units are the identity tax; FAST only).
