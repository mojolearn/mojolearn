# cnn: progress

Lane 8 (docs/lanes/ALGORITHM_EXPANSION_BRIEFS.md). Pass 1: build + sanity +
`tools/algos_lane_check.sh` AGREE (CPU == NVIDIA bitwise), PENDING lanes.

## Design (applies to every algorithm below)

- `x_cnn/ops.mojo`: ONE source of element functions (one output element per
  call, Int32 parameter block). The device (`x_cnn/device.mojo`) launches
  them one thread per element; the host (`x_cnn/host/ops_host.mojo`) calls
  them in a loop. Every contraction is the pinned GEMM:
  `identical_gemm[allow_vendor=False]` on the device, `gemm_oracle` on the
  host (mojolearn.identical.gemm.fp32.v1).
- Conv = im2col (a copy) + GEMM NT; backward dW = GEMM TN over the N*OH*OW
  rows (DEVIATION 5701, the pinned weight-gradient order), db = GEMM TN
  against a ones vector, dcols = GEMM NN, col2im = a GATHER per input pixel
  in (kh, kw) ascending order (DEVIATION 5700, no atomics).
- Bindings: `bindings/_mojolearn_x_cnn{,_host}.mojo`, same export names.
- Sanity: `python/mojolearn/tests/test_x_cnn_*.py` (float64 NumPy reference
  of the PyTorch semantics; `test_reference_matches_torch` checks that
  reference against torch, run with the pod's system python which has torch).
  The pod's pixi default env has no pytest: run the test modules with a
  small runner (see the evidence dir).
- Sabotage used for vacuity (not a pass-1 requirement):
  `~/mojolearn-evidence/algos-cnn/sab_col2im_device.patch` (device col2im in
  reversed kh order): AGREE, DISAGREE, AGREE on x-cnn-conv2d.
- Trainer path (softmax cross entropy, SGD, linear/bias adds, conv output):
  every stored value passes `canon()` (DEVIATION 5705, Clause B): a NaN is
  0x7FC00000 on every column; a softmax row whose max is +inf takes the
  limit (equal split over the +inf logits) instead of inf - inf. Found by
  the `wide` fixture: the trainer diverges to 1e33 and predict_proba read
  NaN 0x7FFFFFFF (NVIDIA) vs 0xFFC00000 (x86), DISAGREE; AGREE after.
  GEMM outputs downloaded directly (dW, dx) are NOT canonicalized: pass 2.
- Torch cross-checks (two-stage: the pixi env has the bindings, the system
  python has torch 2.4.1 + torchvision 0.19.1):
  CNNClassifier 9 SGD steps (momentum 0.9, wd 1e-3) vs torch.optim.SGD:
  max loss diff 2.4e-7, max weight diff 1.5e-8, proba 6e-8.
  BasicBlock (identity and downsampling) vs torchvision BasicBlock, train
  mode: y 9.5e-7, dx 7e-7, dW1 2.1e-5 (scale 63).
  Scripts: /root/trainer_stage{1,2}.py, /root/blk{1,2}.py on the pod.
- PyG is not on the pod: GCN/SAGE sanity is against a float64 NumPy restatement
  of gcn_norm / add_remaining_self_loops / mean aggregation.
- Gotcha: a device entry must `_ = ctx^` AFTER dropping its buffers; a
  DeviceContext destroyed before its buffers hangs the next call.

## Merged

| algorithm | commit | lane check |
|---|---|---|
| Conv2d / Conv1d forward + backward | 0db3c1f17 | x-cnn-conv2d, x-cnn-conv1d: AGREE on 9 fixtures (RTX 4090 vs EPYC 75F3); sabotage DISAGREE then AGREE |
| MaxPool2d/AvgPool2d (+1d) forward + backward | 54309b183 | x-cnn-pool: AGREE on 9 fixtures (RTX 4090 vs EPYC 75F3) |
| CNNClassifier (the small CNN trainer: conv, relu, max pool, linear, softmax CE, SGD) | see git log | x-cnn-trainer: AGREE on 9 fixtures (after the Clause B fix for `wide`) |
| BatchNorm2d / BatchNorm1d (training + eval, running stats, backward) | see git log | x-cnn-batchnorm: AGREE on 9 fixtures |
| Dropout2d (Philox channel mask) | see git log | x-cnn-dropout2d: AGREE on 9 fixtures |
| AdaptiveAvgPool2d / AdaptiveMaxPool2d (global pooling) | see git log | x-cnn-globalpool: AGREE on 9 fixtures |
| ResNet BasicBlock (torchvision) | see git log | x-cnn-resnet-block: AGREE on 9 fixtures; vs torchvision y 9.5e-7, dx 7e-7 |
| GCNConv (PyG) | see git log | x-cnn-gcn: AGREE on 9 fixtures |
| SAGEConv (PyG, mean/sum) | see git log | x-cnn-sage: AGREE on 9 fixtures |

## Pass 2

| step | commit | result |
|---|---|---|
| per-seam proof, NVIDIA + CPU: oracle `x_cnn/checks/oracle.mojo`, driver `x_cnn/checks/seams_check.mojo`, one sabotage arm per seam (`x_cnn/checks/sabotage/seam_57xx_*.patch`, listed in `tools/identity_lanes/cnn.checks`), DEVIATIONs 5700-5715, IDENTITY_PATHS rows 170-179, card stages `x_cnn.<seam>` | f8601ae3b (+ 0e2798963 for 5711-5715) | every arm FAILs with the driver's own `FAIL 57xx ... cells differ` message (checked in the log, not just the exit code) |
| lane-check proof hole fixed (a seam arm bites only if its driver built, ran and failed; BROKEN ARM otherwise) | 3084ca09c | old bad 5709 arm -> BROKEN ARM; fixed arm bites |
| one process-lifetime DeviceContext (`cnn_ctx`); seam 5709 arm fixed; end-to-end steward arm `e2e_host_output_bit.patch` (every CPU output word's low bit flipped) | dd3d15068 | the first M2 Pro run failed with "Failed to create Metal command queue" (a context per call) |

## Option parity (x_cnn/NOT_IMPLEMENTED.tsv), commit 0e2798963

| option | lane | torch check |
|---|---|---|
| Conv padding_mode reflect/replicate/circular (explicit pad + gather backward, 5711), padding 'same'/'valid', groups | x-cnn-conv-options | float64 reference == torch to 1e-10 |
| MaxPool/AvgPool ceil_mode, divisor_override; adaptive pooling to non-dividing sizes (5712) | x-cnn-pool-options | == torch (avg backward to 1e-7: torch rounds through float32 there) |
| BatchNorm momentum=None, track_running_stats=False | x-cnn-bn-options | 1e-10 |
| SAGEConv aggr='max' (ties split, 5713), normalize=True (5714) | x-cnn-gnn-options | == scatter_reduce amax, except torch 2.4 also counts the zero-initialized output as a tie when the max is exactly 0.0 (not carried: a reference quirk) |
| CNNClassifier optimizer adam/adamw (5715), SGD dampening/nesterov | x-cnn-trainer-options | 9 steps vs torch.optim: loss 2.4e-7, weights 4.5e-8 |
| SAGEConv project=True (ba1eaeffe; lane evaluation-order fix e6d526e51) | x-cnn-gnn-options parts prj, prjx | vs torch float64: y 8e-8, dx 1.9e-7 |

Gate for the merge of dd3d15068 + 0e2798963: `algos_lane_check.sh <15 x-cnn lanes> --pass 2 --sabotage x_cnn/checks/sabotage/e2e_host_output_bit.patch` on the RTX 4090 pod, then `tools/test_lane_select.py` (registries changed). Result (RTX 4090 pod, 2026-09-27): all 16 seam arms FAIL under their patch with the driver's own FAIL and PASS after reversal; all 15 lanes AGREE clean, DISAGREE under the end-to-end arm, AGREE restored (`RESULT: PASS`); test_lane_select `OK: 0 failure(s)`. PASSED: do not re-run.

DONE (pass 2 item 2): steward request 1790534359123-cnn-6bb49c8220 on main 6bb49c822: m2pro PASS, do-amd PASS (15 lanes + the e2e arm). Was: (SUBMITTED for main 6bb49c82205f6af4f101594e3c44f08e1dfa3657, 15 lanes, the e2e arm: check `python3 tools/apple_steward.py status`) (submit with `--sabotage x_cnn/checks/sabotage/e2e_host_output_bit.patch`); there is no `cnn-amd` dev box, do-amd is the AMD column.

Option parity gate (SAGEConv project=True), RTX 4090 pod, 2026-09-27:
`algos_lane_check.sh <15 x-cnn lanes> --pass 2 --sabotage x_cnn/checks/sabotage/e2e_host_output_bit.patch`
RESULT: PASS (every seam arm bites; 15 lanes AGREE, DISAGREE under the e2e arm, AGREE restored).
Existing bits unchanged: x-cnn-gnn-options parts mx/mxn/mnn + infer on all 9 fixtures,
and x-cnn-sage train/infer, equal to the 02b63f107 and 25b570476 lane-check JSONs (36/36 equal).
test_x_cnn_gnn 0 failures; test_host_surface 196 passed; test_lane_select `OK: 0 failure(s)`.
PASSED: do not re-run. OPTION PARITY PHASE DONE (every NOT_IMPLEMENTED.tsv row is implemented,
carried, or refused by name).

Steward verdicts collected 2026-09-27 (phase d session): 1790534359123-cnn-6bb49c8220 PASS
(m2pro PASS, do-amd PASS; m3ultra queued, not gating). 1790537166338-cnn-18f9b4d233
(x-cnn-gnn-options, x-cnn-sage, e2e arm) was still QUEUED on m2pro, m3ultra and do-amd:
collect it next session (`apple_steward.py status`); a FAIL there is fixed at the root first.
The old FAIL 1790528141908-cnn-25b570476e is superseded by 1790534359123 (the context and e2e-arm fixes).

## Phase 3 (charter; directive 1(d)): FAST speed (2026-09-27)

FAST tier: `MOJOLEARN_NUMERIC_MODE=fast sh bindings/build_x_cnn.sh` builds (python/mojolearn/_mojolearn_x_cnn.so);
the host twin is IDENTICAL-only by design (build_host_family).

MEASURED FIRST (RTX 4090 pod, stage timers): the kernels were NOT the cost. Conv2d N256 64->64 32x32
forward: kernels 2.9 ms (im2col 1.7, pinned GEMM NT 0.85 = ~23 TFLOP/s fp32, conv_out 0.24) against
49.6 ms per call; the rest was host copies (five per array: read_f32, upload_f32's copy, pinned staging,
an element-by-element append, copy_f32) and, in the trainer, three layer calls per conv block each
moving the full activation both ways. So the FAST win is plumbing and fusion, and both are copies: they
move no bit in either tier, which is why they are in both tiers.
- DEVIATION 5716: `x_cnn/device.mojo` `*_into` entries take the caller's host addresses; one H2D and one
  D2H per array, one synchronize per entry. List forms stay for the seam check.
- DEVIATION 5717: `x_cnn_conv_block_forward/backward` (GPU + host twin, `_surface_cnn.py` exports):
  CNNClassifier's Conv2d->ReLU->MaxPool2d block in one call each way; the backward recomputes the conv
  output from the input it uploads anyway (the forward's kernels on the forward's inputs); the first
  block skips dx (col2im + NN GEMM).
- Tried and REVERTED: pinned staging buffer + 8-thread memcpy for large downloads (no gain, fit slower).
- Not taken: the vendor matmul route (TF32 on NVIDIA, DEVIATION 1885: a precision cut, a quality loss for
  the layer API). A FAST-only GEMM/implicit-GEMM conv has a measured ceiling of ~1-2 ms per trainer step
  (block kernels 0.95 + 2.3 ms of a 35 ms step): not worth it until the transfers are gone (phase e).

Before -> after, same pod, same script (/root/bench_cnn.py; FAST; IDENTICAL within noise of it):
| shape | before fwd / bwd | after fwd / bwd |
|---|---|---|
| Conv2d N256 3->64 32x32 | 25.7 / 24.0 ms | 12.1 / 9.5 ms |
| Conv2d N256 64->64 32x32 | 48.0 / 72.8 ms | 31.0 / 44.9 ms |
| Conv2d N256 64->128 16x16 | 20.0 / 28.3 ms | 10.4 / 14.0 ms |
| CNNClassifier fit 2048x3x32x32, (32,64), batch 256, 1 epoch | 1579.9 ms | 283.1 ms (5.6x) |

Quality rule (paired, 5 seeds x 2 conv shapes + 5 seeds x 2 seeded datasets, /root/fastq.py +
/root/torchq.py): FAST outputs before == after BIT FOR BIT (170/170 arrays), IDENTICAL before == after
(170/170), IDENTICAL GPU == CPU host (170/170). vs float64 torch: conv max rel err FAST 2.7e-7 / 4.0e-7
(= IDENTICAL); trainer test acc FAST 0.9684 / 0.9922 (= IDENTICAL, same bits). No quality change.

Gate: `algos_lane_check.sh <15 x-cnn lanes> --pass 2 --sabotage x_cnn/checks/sabotage/e2e_host_output_bit.patch`
on the RTX 4090 pod (tree 07d66ad83): RESULT: PASS (all 16 seam arms FAIL under their patch and PASS after reversal; 15 lanes AGREE, DISAGREE under the e2e arm, AGREE restored). test_lane_select (surface fragment changed): `OK: 0 failure(s)`. test_host_surface: 196 passed. PASSED: do not re-run.
MERGED to main fad716a02 (gate 0000b: pod checks passed, steward verdicts post-merge).
Stewards: identity 1790540597511-cnn-07d66ad836 (same x_cnn content as the merge) (15 lanes + e2e arm; m2pro, m3ultra, do-amd) SUBMITTED.
Apple FAST speed (m3ultra, before/after, same inline timing cmd): 1790540566963-speed-cnn-6226c84178 (before),
1790540569623-speed-cnn-07d66ad836 (after) SUBMITTED: read `apple_steward.py status` / the verdict stdout
(lines `XCNN-SPEED`). AMD FAST speed: OWED (`apple_steward.py submit` has no `--target do-amd` on main yet; no cnn-amd box). Submit the same inline command with `--kind speed --target do-amd` for 6226c8417 and the merge commit once it exists.

## Phase 4 (charter; directive 1(e)): IDENTICAL speed (2026-09-27), MERGED main 3330b7963

MEASURED FIRST (RTX 4090 stage timers, /root/p4 on the pod): after phase 3 the kernels were still not the
cost. The block backward at N256 64x64x32x32: 9.2 ms of kernels, 12.3 ms FREEING its eighteen device
buffers, 1.2 ms allocating them, plus pageable uploads; `identical_gemm` allocated, synchronized and freed
a workspace per call (0.3 ms of a 0.05 ms TN GEMM); and CNNClassifier moved every activation across PCIe
two or three times per step. DEVIATION 5718 (x_cnn/README.md), plumbing and execution plans only:
- workspace: every entry's device buffers are views of per-slot buffers cached for the process
  (`ws`, grown on demand with a synchronize before a slot is replaced).
- the pinned GEMM through `identical_gemm_into` on a cached workspace (the shipped plan); the small-output
  OP_TN weight/bias gradients (m*n <= 65536, any k) name SPLIT 32x32 / SPLIT 64x64 (forced-plan sweep at
  the lane's shapes: every plan bit-identical, these 2-7x faster than the dispatcher's SPLIT 16x16 plus
  NVIDIA's long-k group rule). No floor on k, so the lane fixtures take this path on every column.
- resident arrays (`x_cnn_res_alloc/free/upload/download/gather` and the `_r` entries): CNNClassifier's
  fit and predict keep weights, optimizer state, activations and gradients on the device; the whole X
  (<= 1 GiB) and labels resident, each batch gathered on the device (a 4-byte word copy); each block's
  backward reads its forward's saved im2col matrix and conv output. CPU twin: resident = host allocation,
  `_r` entries = its ordinary ones.
- test_x_cnn_repeat.py (the 2026-09-27 CHECK NOW directive): every entry large/small/large in one
  process, resident == address bits, the gather. GPU identical, GPU fast, CPU: 0 failures.

Bits (all on the RTX 4090 pod, before = main 025c7a921's bindings):
- /root/q2.py (every touched entry: pools, grouped and reflect conv, BN, BasicBlock, the trainer under
  sgd/adam/adamw/nesterov/dampening/no-pool/no-conv/k5/big): 165/165 arrays equal old GPU vs new GPU,
  old CPU vs new CPU, and GPU == CPU.
- /root/fastq.py (the phase-3 quality set, big shapes that reach the forced plans): 170/170 equal to
  q_new_identical.npz.
- Lane-check output hashes (all 15 lanes, CPU and GPU JSONs) of the gate below equal the phase-3 run
  20260927T202035 in every field but the binding/source digests.

Gate (tree 1dfafaeb1, before a README-only commit and the origin/main merge):
`algos_lane_check.sh <15 x-cnn lanes> --pass 2 --sabotage x_cnn/checks/sabotage/e2e_host_output_bit.patch`
RESULT: PASS (16 seam arms FAIL under their patch and PASS after reversal; 15 lanes AGREE, DISAGREE under
the e2e arm, AGREE restored). test_lane_select `OK: 0 failure(s)`. test_host_surface 196 passed (200 on
the merged tree). PASSED: do not re-run. (The e2e patch's context needed the host binding's new import
moved below `PythonModuleBuilder`; the first gate stopped there, with every seam arm biting.)

NVIDIA IDENTICAL speed, RTX 4090, same pod, same script (median; two runs each, before -> after), script
= the `--cmd` below (synthetic CIFAR-shaped tensors: timing only, no dataset):
| shape | before | after |
|---|---|---|
| Conv2d N256 3->64 32x32 fwd / bwd | 12.1-16.1 / 10.0-11.3 ms | 9.9-11.3 / 5.7-6.0 ms |
| Conv2d N256 64->64 32x32 fwd / bwd | 31.4-33.4 / 45.7-54.6 ms | 17.8-18.1 / 24.6-24.7 ms |
| Conv2d N256 64->128 16x16 fwd / bwd | 10.5-12.0 / 14.4-16.2 ms | 7.5-7.7 / 7.4-7.6 ms |
| CNNClassifier fit 2048x3x32x32 (32,64), batch 256, 1 epoch | 286-312 ms | 27-31 ms (~10x) |
| CNNClassifier fit 8192 rows | 1100-1246 ms | 86-101 ms (~12x) |
| predict_proba 2048 / 8192 | 84-94 / 358-388 ms | 20 / 61-71 ms (~4.5-5.5x) |
The layer API (Conv2d.forward/backward on host arrays) is now PCIe bound (pageable copies of the input and
the output); the trainer is kernel bound (block backward ~1 ms, forward ~0.45 ms per step).

AMD / Apple speed (steward speed jobs, `--builds bindings/build_x_cnn.sh`, `--cmd` = the bench inline:
~/mojolearn-evidence/algos-cnn/speed_cmd.sh, script bench_p4.py; also q2.py and cmplc.py, the bit checks; on
the pod they are /root/p4/*):
- do-amd IDENTICAL before 025c7a921: 1790549425969-speed-cnn-025c7a9214; after 1dfafaeb1:
  1790549426913-speed-cnn-1dfafaeb16 (held in do-amd's queue at session end).
- do-amd FAST (owed from phase 3) before 6226c8417: 1790549424046-speed-cnn-6226c84178; after fad716a02:
  1790549425139-speed-cnn-fad716a02a (held).
- Apple m4pro-a IDENTICAL before: 1790549427934-speed-cnn-025c7a9214; after: 1790549429480-speed-cnn-1dfafaeb16 (queued).
- Identity (post-merge, 15 lanes + e2e arm): 1790540597512-cnn-3330b79639 (m2pro, m3ultra-b, m4pro-b, do-amd).
Read them with `apple_steward.py status` and the verdict stdout (`XCNN-SPEED` lines). A FAIL is fixed at the
root before phase 5. Still queued from phase 3: 1790540566963 / 1790540569623 (m3ultra FAST before/after).

## Steward results collected 2026-09-27 (phase 5 session)

- Identity 1790540597512-cnn-3330b79639 (post-merge phase 4, 15 lanes + e2e arm): m2pro PASS, m3ultra-b PASS,
  m4pro-b PASS; do-amd WORKING at session end (it sat in do-amd's queue/held while another lane's
  `do_amd_steward.sh update` waited for the running requests). Its predecessor 1790540597511 (same x_cnn
  content) is do-amd PASS. Next session: read `apple_steward.py status`; a FAIL is fixed at the root first.
- Apple FAST speed, phase 3 (m3ultra, before 6226c8417 -> after 07d66ad83; median, N256 CIFAR shapes):
  | shape | before fwd / bwd | after fwd / bwd |
  |---|---|---|
  | Conv2d N256 3->64 32x32 | 126.7 / 26.6 ms | 36.4 / 18.8 ms |
  | Conv2d N256 64->64 32x32 | 171.6 / 216.9 ms | 83.7 / 121.3 ms |
  | Conv2d N256 64->128 16x16 | 62.7 / 58.3 ms | 28.9 / 36.3 ms |
  | CNNClassifier fit 2048 rows, 1 epoch | 3619.7 ms | 511.7 ms (7.1x) |
- STILL QUEUED (read them next session, `XCNN-SPEED` lines in the verdict stdout): do-amd IDENTICAL before/after
  1790549425969 / 1790549426913, do-amd FAST before/after 1790549424046 / 1790549425139 (all four released
  from queue/held, `do-amd: queue`); m4pro-a IDENTICAL before/after 1790549427934 / 1790549429480 (`queue`).

## Phase 5 (charter; directive 1(f)): CPU speed (branch lane/algos-cnn)

BLOCKED 2026-09-27 ~00:00Z: the RunPod account balance is negative. Every RunPod pod was deleted (the `cnn`
pod y2ezm0b96e7znv included, stale state cleared with `dev_pod.sh down cnn`), and `dev_pod.sh up` is refused
("balance too low"). Orchestrator: no renting until Andrew tops up. So the code below is written, and it
COMPILES (local one-core `mojo build` through tools/mac_slot.py only, binding + check + every sabotage
arm), but NOTHING HAS RUN. It is not merged. The next session needs a pod first.

Tiers: the host binding is IDENTICAL-only by design (build_host_family refuses FAST), and the CPU route
serves both tiers with the IDENTICAL bits, so one CPU speed-up covers FAST and IDENTICAL.

What the commit does (DEVIATION 5719, x_cnn/README.md):
- `x_cnn/host/gemm_host.mojo` (new): gemm_oracle's cells bit for bit. Operands flushed once (a [k x n]
  right operand with no subnormal is read in place: parallel scan); the n cells of an output row advance as
  SIMD lanes (8 vectors per group) down each leaf with the flush DEFERRED (a tracker of `|bits| - 1`
  catches any nonzero-subnormal raw result, and that group reruns flushing every step; zeros are not
  suspects, unlike the byte LM's version); the balanced tree evaluated as a binary counter (same
  additions, same operands, same order: the argument is in the file); rows split across tasks, or for
  few rows and many leaves (dW: m = OC, k = N*OH*OW) aligned power-of-two LEAF chunks, each chunk a
  subtree of the same tree. The host sabotage define still routes through gemm_oracle itself.
- `x_cnn/host/ops_host.mojo`: `run` splits element loops into contiguous tasks (MOJOLEARN_CPU_THREADS,
  core/host_predict_threads.mojo; at least 16384 elements per task); new pointer cores
  conv2d_forward_into / conv2d_backward_into / conv_block_forward_into / conv_block_backward_into /
  linear_forward_into / linear_backward_into with uninitialized scratch; the List entries stay as the seam
  check's doors. The conv block backward skips dcols + col2im when dx is not wanted.
- `bindings/_mojolearn_x_cnn_host.mojo`: gemm, conv2d fwd/bwd, conv block fwd/bwd (+ `_r`), linear
  fwd/bwd (+ `_r`) read the caller's arrays in place and write outputs in place (no read_f32 / copy_f32);
  `out_f32` is their output seam (a no-op), which the end-to-end arm patches. The other entries (pools,
  BN, dropout, pad, adaptive, graph ops, softmax, sgd, adam) still copy through Lists: convert them next,
  after measuring.
- Proof driver `x_cnn/checks/gemm_host_check.mojo`: NN/NT/TN x 8 shapes (k 1, 7, 128, 129, 896, 2560,
  131077; n reaching the group, vector and scalar tails; n = 1) x fixtures mixed/special(NaN, +-inf)/tiny
  x schedules (1/3/7 tasks, rows and leaves split, default), each against gemm_oracle; `tiny` first shows it
  separates the flushed chain from the unflushed. Arms (tools/identity_lanes/cnn.checks):
  seam_5719_fold_finish, seam_5719_no_redo, seam_5719_chunk_span. Rewritten for the new code:
  seam_5701_dw_serial.patch (dW as one serial chain) and e2e_host_output_bit.patch (copy_f32 AND out_f32).
- `_surface_cnn.py` host_modules gains x_cnn/host/gemm_host.mojo.

### Phase 5 session 2 (2026-09-28, pod `cnn-cpu` py0t44redboxli, RTX 4090 + 2x EPYC 7282, CFS quota 13.6 CPUs)

Steward results collected: identity 1790540597512-cnn-3330b79639 PASS on m2pro, m3ultra-b, m4pro-b AND do-amd.
Apple m4pro-a IDENTICAL speed, phase 4 (before 025c7a921 -> after 1dfafaeb1, N256 CIFAR shapes, median):
| shape | before fwd / bwd | after fwd / bwd |
|---|---|---|
| Conv2d N256 3->64 32x32 | 42.5 / 24.5 ms | 37.5 / 18.9 ms |
| Conv2d N256 64->64 32x32 | 124.8 / 215.5 ms | 106.0 / 193.1 ms |
| Conv2d N256 64->128 16x16 | 40.4 / 57.2 ms | 35.2 / 49.0 ms |
| CNNClassifier fit 2048 / 8192 rows, 1 epoch | 859.9 / 3430.9 ms | 470.4 / 1840.7 ms |
| predict_proba 2048 / 8192 | 314.4 / 1301.6 ms | 203.7 / 851.9 ms |
STILL QUEUED on do-amd (not collectable this session): 1790549424046, 1790549425139 (FAST before/after),
1790549425969, 1790549426913 (IDENTICAL before/after). Read `apple_steward.py status` next session.

Owed items done:
1. gemm_host_check (413680dd1) at MOJOLEARN_CPU_THREADS=1, 3 and unset: PASS (432 cases equal gemm_oracle).
2. Bits (script ~/mojolearn-evidence/algos-cnn/q5.py = q2.py plus conv shapes reaching the leaf-chunk
   split and every SIMD tail, and a CIFAR-shaped fit under sgd and adam; 197 arrays; cmpnpz.py): before =
   main's x_cnn (pod /root/before), CPU route (MOJOLEARN_VENDOR=cpu). before-CPU == before-GPU == after-CPU
   at threads 1, 3 and default == after-GPU: 197/197 every pair. Re-run after EACH later commit below
   (3ff35c935, f2623f7fb, 68c4a76a4): 197/197 at 1, 3 and default every time.
   test_x_cnn_repeat.py: 0 failures on CPU and GPU.
3. Commits after 413680dd1 (all DEVIATION 5719, x_cnn/README.md):
   - 3ff35c935: pools, ReLU/add/mul, softmax CE, SGD, Adam, BatchNorm, Dropout2d host entries read and
     write the caller's arrays (pointer cores `*_into` in ops_host; the List doors call them).
   - f2623f7fb: MEASURED FIRST: x_cnn_gemm alone ran the conv-forward GEMM in 228 ms (21 GFLOP/s, one
     core) inside a 1387 ms conv forward: the rest was integer division in the element functions' index
     decoding. im2col, conv output layout + bias, dout rows and col2im now run as host row/plane loops
     (same words; col2im's (kh, kw) gather order with the stride tests tabulated; CP_REV honored).
   - 68c4a76a4: the CPU twin keeps the fit's saved arrays (im2col matrix, conv output) like the GPU
     binding, so the block backward stops recomputing the conv forward; max pool forward/backward host
     loops (same comparisons, same gather order, PP_REV honored). seam_5701_dw_serial.patch regenerated
     for the new context.
   - 77aa9bc5d (NOT in the gate below): gemm_host 4/2/1-vector deferred-flush groups for n < 64 (the
     small convs' OC ran one flushed chain per vector at ~6 GFLOP/s); gemm_host_check gains n = 32, 27,
     56, 16; seam_5719_no_redo removes every group's fallback.

CPU timing (bench_p5.py, N64 conv shapes, fit 1024 rows (32,64) batch 256 1 epoch, CPU route, median):
| shape | main (serial) t=1 | 413680dd1 t=1 / t=12 | 68c4a76a4 t=1 / t=12 |
|---|---|---|---|
| Conv2d N64 3->64 32x32 fwd / bwd | 665 / 2313 ms | 150 / 268 ; 28 / 50 ms | 21 / 175 ; 8 / 51 ms |
| Conv2d N64 64->64 32x32 fwd / bwd | 13938 / 62691 ms | 1427 / 1883 ; 118 / 246 ms | 294 / 752 ; 51 / 140 ms |
| Conv2d N64 64->128 16x16 fwd / bwd | 6991 / 16886 ms | 376 / 601 ; 49 / 71 ms | 117 / 285 ; 26 / 49 ms |
| CNNClassifier fit 1024 rows | 214674 ms | 18850 ; 2564 ms | 5380 ; 1109 ms (194x at 12 threads) |
| predict_proba 1024 | 40266 ms | 6013 ; 602 ms | 2112 ; 326 ms (124x) |
(main at t=12 equals main at t=1: its host path is serial.) Logs: ~/mojolearn-evidence/algos-cnn/bench_p5_run{1..4}.log.

GATE_PLACEHOLDER

## Earlier next list (history)


1. The gate above PASSED and was merged. Steward: `tools/cloudmac.sh push m2pro <sha>` and `python3 tools/apple_steward.py
   submit --lane cnn --commit <sha> --verify-lanes <15 x-cnn lanes>
   --sabotage x_cnn/checks/sabotage/e2e_host_output_bit.patch`; poll
   `apple_steward.py status` for m2pro AND do-amd PASS; fix anything they find.
2. SAGEConv project=True (the last option row).
3. Speed (PASS 2 item 3): IDENTICAL and FAST on NVIDIA, AMD (do-amd or a
   cnn-amd box), Apple (`apple_steward.py submit --kind speed`) and the CPU
   host path, at a realistic CNN shape (e.g. N 256, 3x32x32, 64 channels,
   R2 data only). Obvious first targets: per-call upload/download (keep
   tensors resident across a trainer step), the FAST tier's conv (allow the
   vendor GEMM route and a tiled direct conv), the host loops (threads).
