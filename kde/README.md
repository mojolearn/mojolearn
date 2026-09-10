# Kernel density estimation

GPU KernelDensity derived from cuML.

`NOT_IMPLEMENTED.tsv` lists kernels, metrics, and options that are deliberately
refused. `estimator.mojo` is the public Mojo surface.

## Verify

```bash
pixi run check-kde
```

Changes should test normalization, metric dispatch, bandwidth edge cases, and reduction order.
FAST-mode approximations must remain explicit and must not leak into IDENTICAL mode.

## FAST score pass (DEVIATION 2490, 2026-09-10)

The FAST tier scores queries in one fused kernel: distance, log-kernel,
log-weight and an online log-sum-exp per query thread, training rows
streamed through shared memory. No `n_query x n_train` matrix exists, so
memory is O(n_query + n_train) and 100k x 100k scores in about half a
second on an M4 where the staged path could not allocate. The staged
path (`kde.dists`, `kde.logk`, `kde.rowmax` stages, serial ascending fold)
is what IDENTICAL and DETERMINISTIC run, and what FAST runs whenever an
identity trace is requested, so the two arms compare in one process
(`python/mojolearn/tests/test_kde_fused_fast.py`). Register rows cover
`d <= 64`; wider rows take a feature-chunked kernel. Numbers and the
timing protocol: `bench/results/kde_fast_2026-09-10/`.
