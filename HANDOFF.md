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
(`hist_half_byte.cu:45-51`) -- no scaling, no range limit. We quantize into
an Int32 accumulator by `fixed_scale` (`checks/fixed_point.mojo`). That
scale comes from `choose_scale(mag)`, where `mag` must be a sum of absolute
values that BOUNDS every partial the device forms.

The boosting loop was passing a magnitude that did not bound them. `fit`
passed `0.0 / 0.0`, and the inlined derivation fell back to `mag = 1.0`,
which is the OPPOSITE of what `choose_scale` returns for a zero magnitude.
That put the scale at its ceiling:

    scale = 2^28 - 1 = 268435455        the largest the type admits
    4 blocks x INT32_MAX = 8589934588   as Int32, exactly -4
    -4 / 268435455 = -1.4901e-08
    prefix scan -> -8, -12, -16 -> -2.98e-08, -4.47e-08, -5.96e-08

Those four numbers were sitting in the dead histogram and name the cause
outright. The kernel summed its partials correctly; every block saturated at
INT32_MAX and the wrap did the rest. Both inlined derivations are now gone in
favour of `choose_scale`, and `fit` passes the real `sum|w|` and `sum|der|`.

An earlier version of this document read the same failure as `-1 / 2^26` from
a magnitude near 4. That was wrong, and the arithmetic above replaces it.

It also explains why the L2 and Cosine calcers agreed to every digit while
this was live. Both were ranking the same garbage. Leaf values come from
`compute_partition_stats`, which never reads the histogram, so only the
SPLITS were wrong, which is exactly how a monotone, mean-beating model hid a
20x regression.

`doc_parallel_boosting.mojo` now computes the true sum of magnitudes for both
planes before each tree, and the scale bounds the partials. Restoring
replication after that is safe and is what CatBoost does.

**The lesson worth keeping:** a fixed-point substitution for a float atomic
carries a RANGE CONTRACT the original does not have. Every caller that feeds
`choose_scale` is part of that contract. Grep for `choose_scale` before
touching any accumulator.

**SUPERSEDED, and this is the important part.** The premise under all of the
above was that Metal has no float `atomicAdd`. It was FALSE. Probed on the
M4, 1024 threads each adding 1.0 through `Atomic.fetch_add` give exactly
1024.0. The FAST arm now takes CatBoost's float atomic on every vendor, which
is what they ship, and the fixed-point Int32 accumulator is what IDENTICAL
selects. Nothing is forced by a vendor.

So the range contract above is no longer live in the default build. Keep the
lesson anyway, because IDENTICAL still pays it.

`checks/replicated_half_byte_check.mojo` compares a REPLICATED half-byte
histogram against a host tally at three arms, one block, many blocks, and a
SABOTAGE arm. The sabotage's expectation FOLLOWS THE BUILD: under fixed point
an unbounded scale must move cells, under the float atomic it must move none,
because the scale is dead input there. Asserting movement in both is how that
check failed on a correct kernel for exactly one commit.

## Method that works, and the one that does not

Bisect. Do not reason. This bug cost two wrong guesses (the Cosine calcer,
the GatherInplace fast path) before a worktree bisect found it in minutes:

1. `git worktree add <dir> <good-commit>` and run the failing check
2. per-file or per-flag flips inside the worktree to localise
3. print the intermediate VALUES (splits, then score, then histogram) --
   the histogram print is what actually found it

## The control plane, now COMPILED

`gbdt/gpu_lib/` had never been compiled at all. It is compiled now and both
checks pass, as `pixi run check-gpu-lib` and `pixi run check-gpu-lib-worker`.
They are separate tasks rather than part of `probe_main` because each carries
its own `main()`.

ONE defect surfaced in that first compile, and it was in the check rather
than in the port: `String(e)[byte=0:48]` on an error message 38 bytes long.
A fixed slice width is an assertion about the message, and the message was
the thing under test.

Fixed along the way: the header count is 57, not the 47 an earlier note
claimed (55 `.h` plus 2 `.cuh`, verified group by group in `NOT_PORTED.md`);
`WaitSubmit` was draining the device where CatBoost only waits for submission
(`gpu_single_worker.cpp:107-109`); `is_local()` returned the opposite answer
(`device_id.h`, local is `HostId == 0`); `stream_synchronize` was missing
outright (`single_device.h:342-347`), so a barrier left sibling streams
marked ACTIVE and they were charged again at the next drain; `run` took its
command BY VALUE, so `RequestStream`'s write-back landed in a temporary; and
a `StopWorker` arriving through the queue skipped the leak checks entirely.

Still unverified: nothing in `gpu_lib` is REACHED by the tree code. It
compiles and it self-checks, and no histogram launch goes through it.

## Known-weak checks

- FIXED. `boosting_check` asserted monotonicity and beating the mean, and
  BOTH still passed at 14.79 mse, a 20x regression. It now asserts an
  absolute level, final loss at or under 2% of the mean baseline, correct is
  about 1% and the broken run was 22.3%, AND that most splits land on the
  three features the target is built from against thirteen noise features.
  The loss bar detects, the feature bar localises.
- FIXED. `checks/boosting_hist_check.mojo` failed at known-good commits,
  and the assertion was fine while the FIXTURE was not. It built four
  asynchronous `enqueue_copy` calls out of ONE host staging buffer and
  mutated it between them, so the partition offset and the leaf id list
  raced a store meant only for the size. It demanded a histogram it had
  never asked the device to build. It also planted a constant weight plane,
  bins that never reached the top bucket, and it was imported by nothing, so
  it had never run in the harness at all. It is now the two-feature-group
  half-byte check, per cell against a host tally, with a sabotage arm that
  zeroes every column past the first and requires group 1 to move while
  group 0 stays bit-identical.
- FIXED. Every half-byte check used to be single-block. See the CLOSED note
  above.
