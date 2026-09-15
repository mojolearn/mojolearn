# KernelRidge, Nystroem, RBFSampler and Cholesky rows — two H100s

RunPod pod `6yabsuveugdl0g`, two NVIDIA H100 80GB HBM3 (sm_90a), 2026-09-14
22:21-22:26Z. Source: the `git archive` of commit `d12a1597f`
(`commit.txt`, lane/multigpu-kernel-methods, which contains
lane/multigpu-cholesky); box and Mac source SHA256 agree. Body `body.sh`;
GPU work ran serially. Design: `docs/multi_gpu/kernel_methods.md` and
`docs/multi_gpu/cholesky.md`.

## Result (`out/gate.txt`)

- `tools/parallel_kernel_methods_check.py`: `PASS 32 kernel method
  configurations and 8 refusals` (`out/public.json`). KernelRidge dual
  coefficients and predictions equal one device for linear, rbf, poly and
  sigmoid at 1, 37, 129 and 515 rows (1 to 3 targets); at 515 rows the sigmoid
  kernel matrix is not positive definite at alpha 0.5 and both paths refuse
  with the same sentence (info 259). Nystroem components, indices,
  normalization, eigenvalues and transforms equal one device for the four
  kernels at 37, 129 and 515 rows. RBFSampler transforms split into 1, 128,
  1000 and 4096-row shards across the two GPUs equal the one-call transform
  (up to 4097 rows, 257 components). Refusals: laplacian, wrong type, FAST
  mode, missing y, bad targets (no publication), unfitted sampler, shape,
  shard size.
- The same check against a kernel-methods binding built with
  `-D MOJOLEARN_CHOLESKY_PARALLEL_SABOTAGE=1` FAILS at its fifth KernelRidge
  fit: `KernelRidge outcome differs, 37, linear` (one device fits, the
  sabotaged two-device factorization refuses at info 35).
- `training/checks/cholesky_parallel_check.mojo`: 40 PASS lines at this
  commit, as in `../cholesky-rows-h100/`.
- The same Cholesky gate built with the sabotage define and run with
  `MOJOLEARN_CHOLESKY_CHECK_FACTOR_ONLY=1` FAILS at the first trailing
  update that has two owners: `factor trace differs n65-r1: 3
  chol.panel000.trailing f32 4225 e6f52b1305e9b1d4 VS 3 chol.panel000.trailing
  f32 4225 6e00708d1c4c7366`. This is the trailing-row partition failing on its
  own (the n=1 to 33 cases before it have at most one trailing row or none).
- `kernel_methods/checks/km_check.mojo`, the unchanged single-device gate:
  `kernel_methods: 18 checks OK [IDENTICAL]`, 13 sabotage arms driven.

## The first leg

`failed-first-leg/` is pod `y5zst4rxktyajt` at `f3dde665a` (22:11Z). Its
check passed 26 configurations and then stopped because the one-device
KernelRidge sigmoid fit at 515 rows refused by name; the check did not yet
compare refusals. No configuration reported a bit difference. The check now
compares outcomes.

No speed or capacity claim is made.
