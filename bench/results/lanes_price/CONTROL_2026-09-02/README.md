# The control that exonerated the code, 2026-09-02

A price run on this box read the Gram product at 1.00x and the standalone
N-T matrix product at 0.99x, against the 1.22x and 5.69x published in
August. The first reading was that the tax had been engineered away. THAT
READING WAS WRONG, and so was the second one (that August therefore stood).
The experiment that settles it is the one every timing surprise in this repo
is owed: RUN THE OLD CODE ON THE NEW DAY.

## The three runs, same box, same afternoon

| arm | Aug 23 published | Aug code, run today | HEAD, run today |
|---|---|---|---|
| gemv.128x128 | 0.07 / 0.08 ms | 2.34 / 2.66 ms | 2.67 / 2.35 ms |
| gram.32x32x1M | 5.27 / 6.41 ms | 26.33 / 25.67 ms | 23.99 / 24.02 ms |
| nt.4096x64x64 | 0.09 / 0.54 ms | 2.65 / 3.67 ms | 3.66 / 3.64 ms |

`aug_code_today.log` is `tools/price_linalg_identity.sh` at commit
`bbee2c9b`, built in a detached worktree. `head_code_today.log` is the same
script at HEAD. The two agree with each other and neither agrees with what
the same script recorded in August.

The timed path did not change: the only diff between those two commits in
`bench/linalg_price_main.mojo` is the mode-label function delegating to
`numeric_mode_name()`, plus an import rename. Nothing inside the clock moved.

## What this establishes

THE LIBRARY DID NOT REGRESS. Old code and new code perform the same on this
box today. Any conclusion of the form "X got slower since August" drawn from
a same-day comparison against an August number is unsupported unless this
control runs first.

THE AUGUST FIGURES ARE RETRACTED, NOT PROMOTED. This control cannot promote
them: those absolute times have never been reproduced on any box, and the run
that produced them recorded no memory state, so the conditions behind them
cannot be established. The box was carrying 7.5 GB of swap on an 8 GB device
while reporting a clean 0.90 load average.

RATIOS ARE NOT PROTECTED BY ALTERNATION. Alternating FAST and IDENTICAL
inside one window cancels drift that SCALES both arms. It does not cancel a
FIXED per-launch cost, which lands in both arms equally and pulls every ratio
toward 1.0, hardest on the arm whose true time is smallest. The arithmetic
closes on the table above: (6.41+21)/(5.27+21) = 1.04 against 0.98 measured
for gram, and (0.54+2.56)/(0.09+2.56) = 1.17 against 1.38 for nt.

So a contended box UNDERSTATES the identity tax. It is not a conservative
estimate of it, and the two Apple readings do not bound each other. That
disqualifies the 2026-09-02 lanes run too, whose Gram arm read 34 ms for the
shape August read at 5.27 ms.

## Where the number comes from instead

A single-tenant rented GPU: no laptop governor, no swap, no browser, one job
on the device. `tools/e2_remote_leg.sh`'s `lanes-price` check runs nine lanes
on an MI325X and an H100. `tools/lanes_price.sh` now records memory pressure
at both ends of every run, because the concurrency gate and the load average
both read clean on the swapping box -- load average does not see paging.
