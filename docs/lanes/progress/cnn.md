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
- Gotcha: a device entry must `_ = ctx^` AFTER dropping its buffers; a
  DeviceContext destroyed before its buffers hangs the next call.

## Merged

| algorithm | commit | lane check |
|---|---|---|
| Conv2d / Conv1d forward + backward | (this commit) | x-cnn-conv2d, x-cnn-conv1d: AGREE on 9 fixtures (RTX 4090 vs EPYC 75F3); sabotage DISAGREE then AGREE |

## Next

MaxPool2d/AvgPool2d (+1d), CNN trainer, BatchNorm, Dropout2d, global pooling,
ResNet BasicBlock, GCN, GraphSAGE. Then PASS 2 (plan's last section).
