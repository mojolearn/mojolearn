# Extra Trees shared class counts — Apple M4, 2026-09-09

The normal Apple classification dispatcher now uses shared integer counts
for 5–32 classes (accumulator widths 8/16/32). Binary/3-/4-class classification
keeps its private-count kernel, as does regression. NVIDIA and AMD retain
private counts by default and can explicitly opt in for validation.

Each (node, feature, workload-block) still tests exactly one ET random
threshold. Thirty-two logical threads share a left/total class-count shard.
The shards merge into the same packed global integer arrays. This borrows
cuML's shared-count/global-merge organization without replacing ET's exact
random thresholds with RF quantile bins. The RNG, threshold comparison,
row traversal, integer totals, gain calculation, tie rules and leaf
arithmetic are unchanged. Shared atomics replace two class-wide block
reductions per class, substantially reducing barriers and private scratch.

Overrides:

- `-D MOJOLEARN_ET_SHARED_CLASS_COUNTS=1`: opt in on every accelerator,
  including binary classification.
- `-D MOJOLEARN_ET_NO_SHARED_CLASS_COUNTS=1`: select the original private
  reference; this wins when both overrides are present.
- `MOJOLEARN_EXTRA_DEFINES` now passes these through `bindings/build_trees.sh`.

The private binding exports `trees_numeric_mode()` and
`trees_shared_counts_mask()`. Mask bits 0/1/2/3 mean accumulator widths 4/8/16/32.
All three locally built public artifacts reported mask 14 and their actual
compiled numeric mode (FAST 0, DETERMINISTIC 2, IDENTICAL 1).

## Validation

`extratrees/tools/check_shared_score.sh` compares the forced-private baseline
(mask 0) against forced-shared (mask 15) in all three numeric modes:

- 27 complete-forest/probability comparisons per mode, covering class counts
  1/2/4/5/8/9/16/17/32, bootstrap, best-first, Gini and entropy;
- 210 score-kernel oracle cells per mode across classification and regression,
  plus the existing separating sabotage arms;
- Actual predicted-probability bits on all fixture rows (first 2048 rows for
  optional large timing workloads), in addition to every forest node/leaf bit.

The default public FAST, DETERMINISTIC and IDENTICAL bindings were rebuilt
in this isolated worktree and passed `binding_dispatch_smoke.py`, including
all nine class boundaries, CPU/device probability equality and normalization,
regression agreement, and native mode/policy readbacks. Each mode also passed
the 45-cell batched forest gate across both objectives. A build with both
force-on and force-off flags reported mask 0 and matched all 27 reference
forests, checking override precedence.

Build scripts' automatic import smoke was skipped because this isolated
worktree borrows non-ET binary dependencies via symlinks; the explicit public
smoke above launched the real ET kernels in all three modes afterward.
No original-worktree binaries were modified.

## Timing

The `timing/` directory is the initial exploration. It includes noisy FAST
binary and 9-class cases and is not the final default-selection evidence.
`timing-final/` measures the final shared kernels against explicit force-off
on the enabled class ranges. Each case runs baseline/candidate/candidate/
baseline, one warmup and three measured fits per process. Both build and
benchmark locks span the complete window. Timing covers native ET device
fit plus synchronization; data generation, full-model fingerprints and
prediction verification are outside the timer. This is not Python
end-to-end fit timing. Every timed result requires matching exact model and
probability fingerprints and compiled mode/policy readbacks.

Reproduce after the gate builds the six binaries:

```
tools/with_build_lock.sh env MOJOLEARN_ET_BENCH_MULTICLASS_ONLY=1 python3 extratrees/bench/shared_score_ab.py OUTPUT
```

Only Apple M4 was measured. No paid resources were started and no remote
training pod was used. Other-vendor speed and execution remain unverified.

## Final stable timing cells

All cells below use 65,536 rows, 13 features, 8 trees and depth 8. Ratios are native
full-fit medians, with exact model/probability fingerprints required.

| Mode | Classes | Private reference ms | Shared counts ms | Speedup |
|---|---:|---:|---:|---:|
| FAST | 5 | 63.323 | 40.593 | 1.560x |
| FAST | 9 | 87.134 | 42.088 | 2.070x |
| FAST | 17 | 205.586 | 48.784 | 4.214x |
| FAST | 32 | 303.678 | 44.057 | 6.893x |
| IDENTICAL | 5 | 62.819 | 41.078 | 1.529x |
| IDENTICAL | 9 | 87.623 | 41.751 | 2.099x |
| IDENTICAL | 32 | 312.617 | 43.587 | 7.172x |

The final IDENTICAL 17-class cell failed the drift criterion, including its
single bounded follow-up (11.4% reference drift), and is not quoted. Its
initial pre-policy candidate run was stable at 3.875x, and all final-code
correctness checks pass. The final IDENTICAL 32-class row above comes from
`timing-confirm/`, with five warmups per process; the remaining rows use
one warmup. No further timing attempts were made.

Compiler logs with diagnostic whitespace are stored as `.log.gz`; their raw contents are preserved.
