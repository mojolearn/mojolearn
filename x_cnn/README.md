# x_cnn: the CNN lane (convolution, pooling, normalization, the CNN trainer, graph convolution)

Lane 8 of docs/lanes/ALGORITHM_EXPANSION_BRIEFS.md. Public classes (door
`python/mojolearn/_expansion_cnn.py`): Conv1d, Conv2d, MaxPool1d/2d,
AvgPool1d/2d, AdaptiveAvgPool2d, AdaptiveMaxPool2d, BatchNorm1d/2d,
Dropout2d, BasicBlock, CNNClassifier, GCNConv, SAGEConv.

## Contract

- `ops.mojo` holds one source of element functions; `device.mojo` launches
  them one thread per element, `host/ops_host.mojo` loops over them. Every
  contraction is mojolearn.identical.gemm.fp32.v1 (`identical_gemm` without
  the vendor route on the device, `gemm_oracle` on the host).
- IDENTICAL: CPU == every GPU vendor, bit for bit. FAST: the same kernels
  (no bit promise). Both tiers take the caller's host addresses straight to
  and from the device (DEVIATION 5716) and run CNNClassifier's conv block
  (Conv2d -> ReLU -> MaxPool2d) as one binding call each way
  (`x_cnn_conv_block_*`, DEVIATION 5717); both are copies/fusions that move no
  bit in either tier.
- Seams and their DEVIATIONs (IDENTITY_PATHS.md rows 170-179):
  5700 col2im gather order, 5701 weight gradient on the pinned GEMM,
  5702 BatchNorm folds, 5703 Dropout2d Philox mask, 5704 SpMM row folds,
  5705 NaN canon and the softmax +inf limit, 5706 max-pool tie,
  5707 avg-pool division, 5708 softmax exp-sum order, 5709 SGD without FMA,
  5710 GCN normalization order, 5711 pad backward gather order,
  5712 adaptive average pooling backward gather order, 5713 SAGE max
  backward (ties split, targets ascending), 5714 row L2 normalize fold,
  5715 Adam/AdamW without FMA.
- Check: `tools/with_identical_mode.sh pixi run mojo run -I . x_cnn/checks/seams_check.mojo`;
  sabotage arms `checks/sabotage/seam_57xx_*.patch`; the lane check
  `tools/algos_lane_check.sh <x-cnn lanes> --pass 2`.
- Options not carried: `NOT_IMPLEMENTED.tsv`.
