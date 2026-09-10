# Experimental RF binned column tiles

The public RF builder already implements the shared histogram algorithm from
cuML v26.08.00 (`265b9da6a0e75dbef071a3168398b993a5ff6f0e`,
`cpp/src/decisiontree/batched-levelalgo/kernels/builder_kernels_impl.cuh:285-355`).
This is not a replacement of a global-only histogram implementation.

The opt-in `MOJOLEARN_RF_HIST_COLUMNS2` and `MOJOLEARN_RF_HIST_COLUMNS4`
compile definitions group two or four sampled columns per CTA. Row indices
and labels are gathered once per row. Each feature retains its own histogram,
actual quantile-bin count and existing objective `IncrementHistogram` and
`BinT.AtomicAdd` operations. Integer/fixed-point overflow behavior, bootstrap
RNG, sampled-label/weight addressing, bins and split tie order are unchanged.
The global output retains the original column-batch stride; the split kernel
and tree-growth schedule are unchanged.

Shared storage uses the first fitting 2/4/8/16 KiB tier, aligned down to whole
BinT slots. This binned-only kernel does not copy quantiles, so its complete
required shared storage is `tile * max_bins * num_outputs * sizeof(BinT)`.
The existing shared-memory configuration still decides whether shared storage
is allowed before this specialization. Searching, global, oversized tiled
histograms and replicated histogram combinations fall back to the reference.
Callers omitting the new optional host `num_outputs` also retain the reference.
Both flags are OFF by default on every vendor and numeric mode.

The direct oracle in `ensemble/checks/rf_perf_candidates_check.mojo` compares
all 288 cells for tile2 fine and tile4 coarse/tail workloads against a separate
host tally. Six ragged nodes, mixed feature bin counts and nonzero column
start exercise addressing. Constant-bin sabotage changes 134 cells, proving
that the tiled kernel is actually reached. Existing shared/global, replicated
histogram and bootstrap sorting checks also pass.

Full-model gates and raw local timings are saved in
`bench/results/rf_column_tiles_2026-09-09/`. The repeatable timing command is:

```sh
tools/with_build_lock.sh python3 ensemble/bench/rf_column_tiles_bench.py --mode fast --rows 262145
```

The benchmark uses a synthetic binary target, 28 column-major features,
20 trees, depth limit 12 and 128 bins. It is HIGGS-shaped, not HIGGS data.
Fit timing excludes generation/upload and model hashing. Variants run in
forward/reverse order; each process has five canary warmups and drops its
first fit. Gross timing validity requires canary max/min <=1.1; improvements
smaller than observed canary variation are not established speedups.
No CUDA/AMD performance claim is supported by the local Apple M4 runs.

Validation completed on local Apple M4: FAST, DETERMINISTIC and IDENTICAL
reference/tile2/tile4 builds agree on all 11 complete-model and prediction
fingerprints per build (99 checks total). Cases include 2/5/17 classes,
regression, sample weights, bootstrap on/off, max-leaves cap, 256-bin upper
binned boundary and 257-bin searching fallback. Launch logs assert tiled
reach for eligible cases and no tiled launches for oversize/search fallbacks.

The public FAST tile4 AOT extension also passes classifier/regressor fits
and repeat prediction checks; the trace contains 88 actual tiled launches.
`ensemble/checks/rf_column_tiles_public.sh` builds that diagnostic extension,
sets the development runtime loader path, executes public fits, and restores
the previously installed RF module. Other public numeric modes and other
vendors were not AOT-tested here; their native tests are the evidence above.

No timing result establishes a speedup. The initial four-class 65k nominal
3% gain was below 6.2% canary variation. Larger binary 262145×28 runs preserved
all model fingerprints but were invalidated: FAST canary spread 1.184,
IDENTICAL 1.699, versus the final 1.1 limit. Raw timings are retained for
investigation; defaults remain OFF on every vendor and mode.

September10 bounded follow-up: two attempts each for FAST and IDENTICAL,
with full-fit prewarmups and shared-lock serialization, again failed the
unchanged1.1 canary limit. All120 measured full-model fingerprints matched.
No default changed and no speedup was certified. Raw samples, binary/source
hashes and reproduction commands are in
`bench/results/rf_column_tiles_2026-09-10/README.md`.
