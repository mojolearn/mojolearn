# SVM

Dense FP32 binary C-SVC and epsilon-SVR derived from cuML's SMO solver and cuVS kernel matrices. `NOT_IMPLEMENTED.tsv` defines unsupported
multiclass, sparse, kernel, and parameter combinations. Unsupported behavior must fail clearly
rather than silently selecting a different algorithm.

## Verify

```bash
pixi run check-svm
pixi run check-svm-oracle
```

Identity depends on working-set selection, tie handling, kernel evaluation, and reduction order.
Performance changes must preserve those choices in IDENTICAL mode.

## The SMO's schedule (DEVIATIONS 2491 and 2492, 2026-09-10)

Measured on the M4 at HIGGS 50,000 x 28 (156 outer iterations, 71,153
inner) with `MOJOLEARN_STAGE_TIMES=1`, which `svm/impl/smosolver.mojo` now
honors: the block solve was 2.73 s and the full kernel tile 3.58 s of a
7.53 s fit. Two changes, one per phase:

- **2491** `svm/checks/pinned_argreduce.mojo::block_argext`: the three
  arg-reductions of the block solve fold with `shuffle_xor` butterflies
  inside a warp and one threadgroup exchange across warps (two barriers
  instead of twelve), and return the winning thread beside the pair so the
  ballot that recovered it is gone. Same total order on (value, key) with
  unique keys, so the same element wins under any topology or lane width;
  IDENTICAL takes it too. Block solve 2.73 s to 0.55 s.
- **2492** `svm/impl/distance/kernel_matrices.mojo::rbf_fused_tile`: FAST
  only. The RBF tile is one kernel (a row of `b` per thread in registers,
  rows of `a` through shared memory, `exp(-gamma(na + nb - 2 dot))`
  written once) instead of a k = n_features GEMM plus an epilogue. Register
  rows to 64 features; wider inputs keep the GEMM path. IDENTICAL keeps
  `identical_gemm_into` and the pinned epilogue. Full tile 3.58 s to 0.39 s.

Fit 7.5 s to 2.2 s under the stage clock; `svc_main.mojo` 44/44 in FAST,
43/44 in IDENTICAL where the one failure (`svr_device_matches_oracle`,
ws sequence at outer iteration 0) is pre-existing on main at deb01bcf and
untouched by these changes. Numbers against scikit-learn:
`bench/results/svm_fast_2026-09-10/`. The next phase by size is
`select_ws` (a 32-pass one-bit radix sort, about 130 launches per outer
iteration, 0.58 s of the 2.2 s).
