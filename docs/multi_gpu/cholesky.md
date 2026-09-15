# Cholesky: operation-level rows and right-hand sides

Design note for `mojolearn.parallel_classical.fit_cholesky` and
`solve_cholesky`, and for `MOJOLEARN_CHOLESKY_DEVICE_COUNT` inside
`cholesky/checks/potrf.mojo::potrf_lower` and
`cholesky/checks/trsm.mojo::cho_solve`. IDENTICAL mode only; a multi-device
factor or solution must have the exact bits of the one-device one.

## The factorization

`potrf_lower` is blocked and right-looking with the pinned panel width
`CHOL_NB_PINNED = 32` (DEVIATION 1630). For each panel it factors the
`nb x nb` diagonal block, reads `info` back, solves the panel
`L21 = A21 L11^-T` one thread per trailing row, forms `G = L21 L21^T` with
`identical_gemm_into(OP_NT)` and subtracts the lower triangle of `G` from
`A22`.

Can the trailing update be partitioned without changing bits? The part that
cannot is the order of panels. Every trailing cell accumulates one running
subtraction per earlier panel, in panel order, and the next panel's pivot
depends on all of them. That order is sequential by construction and stays
sequential. What can be partitioned is one panel's product. An output cell
`G[i][j]` contracts row `i` and row `j` of `L21` over the panel width `w`,
and the gemm fp32.v1 fold of a cell is a function of `k = w` only (at
`w <= 32` it is one ascending leaf). It does not depend on how many output
rows share a launch or which plan the dispatcher picks for the launch shape.
So contiguous output-row ranges of `G` run on owners that hold their rows of
`L21` (left operand) and all of `L21` (right operand), and the rows are
copied back as bytes into `G`. The subtraction, the panel factor, the panel
solve and the `info` decision stay on the root.

## The solve

`cho_solve` runs `trsm_lower_kernel` then `trsm_upper_kernel`, one thread
per right-hand-side column. A column's substitution is sequential in its rows
(row `i` reads every solved row before it), so a single column is not
partitioned and a one-column solve runs on one device. Whole columns are
independent: each owner receives the factor and its columns, runs both
original kernels, and its columns are copied back into their original
positions after each stage. The factor, the columns and the results move
through host memory and each owner's own context, not device to device: the
device-to-device form diverged on two MI300X for every factor above 1 MiB, in
the columns owned by device 1 (`bench/results/multi_gpu/2026-09-14/
cholesky-mi300x-diag/`); the cause is not identified. The `chol.solve.forward` and `chol.solve.back`
card stages are recorded on the root after each gather, so a traced
multi-device solve writes the same card.

## Consumers

Because the switch lives inside `potrf_lower` and `cho_solve`, any caller in
a process with the variable set gets the same partition: `Cholesky`,
`KernelRidge` (`kernel_methods/impl/kernel_ridge/kernel_ridge.mojo`) and the
GaussianMixture precision Cholesky. The pool does not set it globally; the
`cholesky_fit` and `cholesky_solve` worker operations set it for their own
call only, so the already-qualified GP and GaussianMixture drivers keep
their root factorization path.

## Gates

- `training/checks/cholesky_parallel_check.mojo`: shapes 1 to 513 across the
  panel boundaries, 1 to 7 right-hand sides, pinned ridge and no ridge;
  equal identity traces for the factor (every panel stage) and the solve,
  equal factor, `info`, `nb` and `logdet` bits; two non-positive-definite
  matrices fail at the same `info` with the same partial factor. A check-only
  build, `-D MOJOLEARN_CHOLESKY_PARALLEL_SABOTAGE=1`, reads later owners'
  rows and columns one position early and must fail.
- `tools/parallel_cholesky_check.py`: public factor and solve against one
  device, failed factor refusal, refusals.

The root keeps the full matrix and the panel workspace, and every panel
creates owner contexts and copies `L21`. No speed or capacity claim is made.
