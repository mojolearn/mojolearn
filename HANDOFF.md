# Where this stands, 2026-08-19

Read `PORTING_RULES.md` first, then this. `VENDOR_LIBS.md` is the substitution
ledger. `UNWIRED.md` is everything ported but unreached.

## The rule that governs everything

**Assume OUR code is broken; CatBoost is right** (PORTING_RULES 0b). Earned:
this session found ~12 real defects, every one by reading their source, none
by reasoning about ours. Both "optimisations" we had invented were worse than
what they replaced.

## State: green

`probe_main.mojo` passes end to end. Boosting reaches 0.61 mse on the
synthetic fixture (mean baseline 66.46), model round-trips to ~9e-07, mixed
tree 16/16 non-empty at depth 4 and 56/64 at depth 6, all histogram slice
checks 0 wrong, permuted-id checks clean on both load paths.

Timing, 800k x 100 one-byte depth 6: **~109 ms/tree**. CatBoost CPU on the
same shape is 30.1 ms. Earlier numbers in git history are NOT comparable --
several were taken while the histogram was silently empty.

## CLOSED: the half-byte replication bug, and what CatBoost does

Half-byte replication is BACK ON, matching CatBoost
(`hist_half_byte.cu:80-81`), and boosting is at 0.61 mse.

**Root cause, and it was ours, not the kernel's.** CatBoost sums replicated
partials with `atomicAdd(dst + fold, val)` on a FLOAT
(`hist_half_byte.cu:45-51`) -- no scaling, no range limit. Metal has no float
atomic, so we quantize into an Int32 accumulator by `fixed_scale`
(`mojo_only/fixed_point.mojo`). That scale comes from `choose_scale(mag)`,
where `mag` must be a sum of absolute values that BOUNDS every partial the
device forms.

The boosting loop was passing a magnitude that did not bound them. With
`SCALE_LIMIT = 2^28 - 1` and a magnitude near 4, the scale came out at 2^26,
and a weight count of 550 times 2^26 overflows Int32. The histogram came back
as `-1.5e-08, -3.0e-08, ...`, which is exactly `-1 / 2^26` accumulating.

`doc_parallel_boosting.mojo` now computes the true sum of magnitudes for both
planes before each tree, and the scale bounds the partials. Restoring
replication after that is safe and is what CatBoost does.

**The lesson worth keeping:** a fixed-point substitution for a float atomic
carries a RANGE CONTRACT the original does not have. Every caller that feeds
`choose_scale` is part of that contract. Grep for `choose_scale` before
touching any accumulator.

**Still true and still a gap:** every half-byte check runs at
`grid_dim=(1,1,1)`, so the multi-block path is exercised only by the boosting
check, indirectly. A check comparing a REPLICATED half-byte histogram against
a host tally is still missing.

## Method that works, and the one that does not

Bisect. Do not reason. This bug cost two wrong guesses (the Cosine calcer,
the GatherInplace fast path) before a worktree bisect found it in minutes:

1. `git worktree add <dir> <good-commit>` and run the failing check
2. per-file or per-flag flips inside the worktree to localise
3. print the intermediate VALUES (splits, then score, then histogram) --
   the histogram print is what actually found it

## Unverified

The `gpu_lib` control-plane rewrite landed but has NOT been compiled. It
reports: `NOT_PORTED.md` claimed 47 headers where there are 57; `WaitSubmit`
was draining the device where CatBoost only waits for submission
(`gpu_single_worker.cpp:107-109`); `is_local()` returned the opposite answer
(`device_id.h`, local is `HostId == 0`); `TCommand` carried two fields with
no counterpart. It also added `gpu_profiler.mojo` and `mapping.mojo`, and
says `mojo_only/gpu_lib_worker_check.mojo` will now fail because it was
asserting our old over-syncing.

## Known-weak checks

- `boosting_check` asserts monotonicity and beating the mean. BOTH still
  passed at 14.79 mse, a 20x regression. It needs an absolute floor.
- `mojo_only/boosting_hist_check.mojo` is BROKEN -- it fails at known-good
  commits. Do not trust it; fix or delete it.
- Every half-byte check is single-block (see above).
