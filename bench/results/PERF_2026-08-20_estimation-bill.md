# What Newton leaves COST per tree: the Logloss estimation bill, priced

2026-08-20 night, M4 base, 10 GPU cores, AC power, run under
`tools/with_build_lock.sh`. Harness: `pixi run estimation-bench`
(`mojo_only/estimation_bench.mojo`), new in this commit.

## Why this was measured

gbm-bench runs Logloss on five of its six datasets, and the fixed-cost
decomposition (`PERF_2026-08-20_fixed-cost.md`) says fixed per-tree cost is
what decides every row below ~436k rows. The Logloss path runs a stage RMSE
never touches: CatBoost's Newton walker at THEIR default of ten iterations
(`catboost_options.cpp:157-164`), and every iteration ends in a blocking
readback, which is the expensive syllable on Metal. Nobody had priced it.

## Method

Same-arm subtraction, alternated arms, one process, one context: per arm and
rep, T(2 trees) and T(22 trees) back to back; (T22 - T2)/20 is ms/tree with
quantization, borders, pool setup and kernel compilation cancelled by the
difference. Arms alternate INSIDE each rep because this box drifts 1.7x in
twenty minutes (same file as above). 50k x 100, 254 borders, depth 6,
lr 0.1, l2 1 -- the gbm-bench shared params at the fixed-cost fixture's
scale, at depth 6 (the fixed-cost fixture's depth) and depth 8 (the depth
gbm-bench PINS). Medians of three reps; two full windows were run and both
are shown, because the box drifts and one window quoted alone would be a
number with a mood:

    arm                        depth 6          depth 6 (2nd)   depth 8
    ------------------------   --------------   -------------   -------
    RMSE (no estimation)       11.71            9.24            17.88
    Logloss, iterations 1       9.66            10.12           16.46
    Logloss, iterations 10     14.18            15.41           25.54

    one Newton iteration        0.50            0.59            1.01  ms
    the ten-iteration bill      4.5             5.3             9.1   ms/tree

The per-iteration figure is the durable number: it comes from same-arm
pairs and repeats within every window (depth 6: 4.2/5.5/4.5 and
0.3/5.3/5.8 ms for the 9-step gap; depth 8: 10.3/4.0/12.5 -- rep spread is
real, medians are quoted). The RMSE column is NOISY across windows (9.1 to
13.2 at depth 6), so the "entry fee" read (ll1 minus rmse, -2.0 to +0.9)
says only that entering the estimation stage costs about what leaving the
RMSE leaf-value path saves, plus or minus the box's mood. Do not quote the
entry fee; quote the iteration price.

**The iteration price DOUBLES from depth 6 to depth 8** (0.5-0.6 -> 1.0
ms), which the sync alone cannot explain -- a synchronize does not know the
tree depth. The part that grows is leaf-slot-proportional work inside the
eval: `compute_partition_stats` launches its grid at `n_leaf_slots` in y
(256 slots at depth 8 against 64), and MoveTo's `add_model_value_kernel`
likewise. So the bill is a sync FLOOR of ~0.4 ms plus a slot-proportional
part of comparable size at 256 leaves. That split matters: the floor is
unfixable (below), the slot term is ordinary kernel work and would shrink
only if their grid shape does -- theirs is the same shape
(`update_part_props.cu:209-215`), so there is nothing to mirror away.

## Where the half-millisecond goes

One walker iteration is: MoveTo (1 h2d copy + 1 launch), then the fused
eval (1 cross-entropy launch + 2 partition-stats launches + 2 d2h copies +
**1 blocking synchronize**). At the enqueue price the fixed-cost file
measured (~18 us each), the seven enqueues are ~0.13 ms; at depth 6 the
remaining ~0.4 ms is dominated by the synchronize round trip, and at depth
8 a slot-proportional kernel term of similar size joins it (previous
section). The sync count equals the iteration count, exactly one per eval,
and the two readbacks are already batched ahead of the single sync
(`pointwise_oracle.mojo`, `write_value_and_first_derivatives`).

## The negative result, recorded so nobody builds it

**A device-resident walker is DEAD on this target, by hardware, not by
taste.** The sequential dependency is real -- iteration k+1's direction
needs iteration k's gradient, so the sync per iteration can only go away if
the direction/backtracking logic moves onto the device. But the walker's
arithmetic is float64 BY THEIR DESIGN (`descent_helpers.cpp:128-204`:
FunctionValue, gradient, `G/(H + 1e-20)` all double), and Metal has no
float64. A float32 device walker would move bits in every leaf value,
which the replay and oracle gates would rightly fail. Speculative
enqueue-ahead is equally dead: there is nothing to speculate with until the
gradient is on the host.

What remains shaveable without moving a bit: enqueue count only -- merging
the two d2h copies into one buffer, folding MoveTo's shift copy into a
kernel constant. Upper bound ~0.13 ms of the 0.50, call it ~1 ms/tree
across ten iterations, PRICED HERE AND DECLINED for now: it buys back at
most a quarter of a bill whose floor is the sync, and the oracle's buffer
layout is currently byte-shared with the checks that gate it.

## What it means for the scoreboard

gbm-bench pins depth 8, so THE row that matters is: at 50k rows our
Logloss-at-defaults tree costs **25.5 ms against 17.9 for RMSE** -- the
estimation stage is a ~9 ms/tree, ~43% surcharge on the fixed cost, and
every fixed-cost crossover moves right by roughly that much on five of the
six gbm-bench datasets. `fraud` (285k rows) sat below the crossover before
the surcharge and sits further below it now; `bosch` (1.18M) and `higgs`
(11M) stay slope-dominated and stay expected wins.

Two things this table CANNOT say, so it does not: our depth-8 slope was
not measured here (one row count only -- 50k -- so no a + b*rows fit), and
their Logloss CPU cost was not measured anywhere yet. Their walker runs
the same ten iterations over every document on the HOST, so their Logloss
arm gets slower too, possibly by more than ours. Both numbers can only
come from the user's interleaved window; nothing here ran a CatBoost
process.

**The bill is not a knob.** Ten iterations is CatBoost's default and
gbm-bench pins defaults; `leaf_estimation_iterations=1` is 4.5 ms/tree
faster and is CHEATING under the same-settings rule. This file exists so
the 14 ms is understood, not so it gets "fixed" by turning their knob.
