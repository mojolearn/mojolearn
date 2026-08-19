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

## THE OPEN BUG, highest priority

**Half-byte replication is DISABLED and must not be re-enabled without a
check.** `greedy_search_helper.mojo`, the two `half_byte_hist*` launches, use
`grid_dim=(groups, ...)` where CatBoost uses `groups * replicas`
(`hist_half_byte.cu:80-81`).

CatBoost DOES replicate this kernel and sums partials with
`atomicAdd(dst + fold, val)` when `blockCount > 1` (`hist_half_byte.cu:45-51`).
Our fixed-point Int32 stand-in for that atomic is WRONG in this kernel --
though the SAME stand-in is correct in `hist_binary.mojo`, which is the clue.

Evidence, depth-0 weight histogram of the boosting fixture:

    correct   550, 1104, 1651, 2237, 2812, 3308, 3783, 4328
    with rep  -1.5e-08, -3.0e-08, -4.5e-08, -6.0e-08, ...

`part_stats` was identical, so the histogram alone dies. Boosting goes
0.61 -> 14.79 mse.

**Why no check caught it: every half-byte check runs at `grid_dim=(1,1,1)`.**
The multi-block path of that kernel has never been executed by any test.

Next step: diff `hist_half_byte.mojo`'s `acc_i32` address arithmetic against
`hist_binary.mojo`'s, which works. Their writebacks differ -- binary decodes
a combination (`hist_binary.cu:47-56`), half-byte reads `smem[fid + 8*fold]`
(`hist_half_byte.cu:33-43`) -- so the acc offset almost certainly needs to
mirror the half-byte writeback, not the binary one. THEN write a check that
compares a REPLICATED half-byte histogram against a host tally.

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
