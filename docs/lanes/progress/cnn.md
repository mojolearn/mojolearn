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
| still NOT IMPLEMENTED: SAGEConv project=True | | |

Gate for the merge of dd3d15068 + 0e2798963: `algos_lane_check.sh <15 x-cnn lanes> --pass 2 --sabotage x_cnn/checks/sabotage/e2e_host_output_bit.patch` on the RTX 4090 pod, then `tools/test_lane_select.py` (registries changed). Result (RTX 4090 pod, 2026-09-27): all 16 seam arms FAIL under their patch with the driver's own FAIL and PASS after reversal; all 15 lanes AGREE clean, DISAGREE under the end-to-end arm, AGREE restored (`RESULT: PASS`); test_lane_select `OK: 0 failure(s)`. PASSED: do not re-run.

OWED (pass 2 item 2): M2 Pro steward PASS and do-amd PASS on the merged commit (SUBMITTED for main 6bb49c82205f6af4f101594e3c44f08e1dfa3657, 15 lanes, the e2e arm: check `python3 tools/apple_steward.py status`) (submit with `--sabotage x_cnn/checks/sabotage/e2e_host_output_bit.patch`); there is no `cnn-amd` dev box, do-amd is the AMD column.

## Next

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
