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

## Next

PASS 1 is complete once the rows above are merged. Then PASS 2 per the
plan's CURRENT DIRECTIVES: AMD box (`tools/dev_pod.sh up cnn 240 --vendor amd`),
per-seam `.checks` drivers + sabotage patches (seams: col2im gather 5700,
weight-grad GEMM TN 5701, BN folds 5702, Dropout2d Philox mask 5703,
SpMM row folds 5704, NaN canon 5705, maxpool tie, avgpool divisor), option
parity (x_cnn/NOT_IMPLEMENTED.tsv), then speed.
