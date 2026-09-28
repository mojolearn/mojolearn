# x_cnn: the CNN lane (convolution, pooling, normalization, the CNN trainer, graph convolution)

Lane 8 of docs/lanes/ALGORITHM_EXPANSION_BRIEFS.md. Public classes (door
`python/mojolearn/_expansion_cnn.py`): Conv1d, Conv2d, MaxPool1d/2d,
AvgPool1d/2d, AdaptiveAvgPool2d, AdaptiveMaxPool2d, BatchNorm1d/2d,
Dropout2d, BasicBlock, CNNClassifier, GCNConv, SAGEConv.

## Contract

- `ops.mojo` holds one source of element functions; `device.mojo` launches
  them one thread per element, `host/ops_host.mojo` loops over them. Every
  contraction is mojolearn.identical.gemm.fp32.v1 (`identical_gemm` without
  the vendor route on the device, `gemm_oracle`'s cells on the host through
  `host/gemm_host.mojo`, DEVIATION 5719).
- IDENTICAL: CPU == every GPU vendor, bit for bit. FAST: the same kernels
  (no bit promise). Both tiers take the caller's host addresses straight to
  and from the device (DEVIATION 5716) and run CNNClassifier's conv block
  (Conv2d -> ReLU -> MaxPool2d) as one binding call each way
  (`x_cnn_conv_block_*`, DEVIATION 5717); both are copies/fusions that move no
  bit in either tier.
- DEVIATION 5718 (IDENTICAL speed): an entry's device buffers are views of
  per-slot workspace buffers cached for the process (no allocate/free per
  call); the pinned GEMM runs through `identical_gemm_into` on a cached
  workspace (the shipped plan), except the small-output OP_TN weight and bias
  gradients, which name a SPLIT plan (every plan gives the same bits); and
  CNNClassifier's fit and predict keep weights, optimizer state, activations
  and gradients in RESIDENT arrays (`x_cnn_res_*` and the `_r` entries: device
  addresses on the GPU binding, host allocations on the CPU twin, whose `_r`
  entries are its ordinary ones). The fit keeps the whole X (up to 1 GiB) and
  its labels resident and gathers each batch on the binding's side
  (`x_cnn_res_gather`, a 4-byte word copy), and each block's backward reads
  the im2col matrix and conv output its forward saved instead of recomputing
  them (the same kernels on the same inputs made them). Plumbing and
  execution plans only: the same kernels on the same values in the same
  order.
- DEVIATION 5719 (CPU speed, phase 5): the host twin's contractions run in
  `host/gemm_host.mojo`, `gemm_oracle`'s cells bit for bit: operands flushed
  once (a subnormal-free right operand read in place), the cells of an output
  row advanced together as SIMD lanes down each leaf with the flush deferred
  (a nonzero subnormal raw result sends the group back through the
  flush-every-step chain), the balanced tree evaluated as a binary counter
  (the same additions, same operands, same order), rows split across tasks,
  or for few rows and many leaves, aligned power-of-two leaf chunks (each
  chunk a subtree of the same tree). Element loops (`run`) split into
  contiguous tasks: an element function computes one output element, the
  device's own contract. The conv layout loops (im2col, the conv output's
  NCHW layout and bias, the gradient's row layout, col2im) and max pooling
  forward and backward run as host row/plane loops with the indices advanced
  incrementally instead of decoded by integer division per element (the
  division was several times the GEMM's cost): the same words, and col2im
  and the pool backward keep their (kh, kw) gather order (CP_REV/PP_REV
  honored for the host sabotage build) with the stride tests tabulated once
  per call. Every host entry (gemm, conv, conv block, linear, pools, ReLU,
  add, mul, softmax cross entropy, SGD, Adam, BatchNorm, Dropout2d) reads
  the caller's arrays in place and writes its outputs in place (no List
  copies in or out; `out_f32` is the output seam the end-to-end sabotage
  patches); the pad, adaptive pooling and graph entries still copy. In a
  fit the CPU twin keeps each block's im2col matrix and conv output in the
  saved arrays the fit already allocates, as the GPU binding does, and the
  block backward reads them instead of recomputing them. Thread count: MOJOLEARN_CPU_THREADS
  (`core/host_predict_threads.mojo`); it moves no bit. Check:
  `x_cnn/checks/gemm_host_check.mojo` (every op, one leaf and many, both
  splits, 1/3/7 tasks, planted subnormals, NaN, infinities; arms
  `seam_5719_*.patch`).
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
