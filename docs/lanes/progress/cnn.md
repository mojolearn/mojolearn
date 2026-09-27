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

## Next: phase 4: IDENTICAL speed (its own session)

1. Collect: 1790537166338 (identity), 1790540597511 (identity, this phase), the two m3ultra speed jobs;
   fix any FAIL at the root first. AMD FAST speed once the steward has an AMD speed kind.
2. Phase 4 (IDENTICAL speed): same bits, faster. The cost is now transfers and allocations, not kernels: device-resident
   tensors across a trainer step (weights, optimizer state and activations on the device; only the batch
   up and the loss down), cached device buffers instead of per-call allocation, and the conv im2col
   buffer (604 MB at N256 C64) replaced by a tiled im2col-in-GEMM staging with the pinned fold order.
   Re-prove bitwise on every column.
3. Then phase 5 (CPU speed).

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
