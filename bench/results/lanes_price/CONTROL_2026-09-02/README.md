# The control that exonerated the code, 2026-09-02

A price run on this box read the linear-algebra identity tax at 1.00x for
the Gram product and 0.99x for the standalone N-T matrix product, against
1.22x and 5.69x in `results/identity-cost-unsupervised-m4-2026-08-23.json`.
The first reading of that was that the tax had been engineered away by the
eleven commits that touched the identical GEMM path since. THAT READING WAS
WRONG, and the experiment that shows it is the one every timing surprise in
this repo is owed: RUN THE OLD CODE ON THE NEW DAY.

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

## What this establishes, and what it forbids

THE LIBRARY DID NOT REGRESS. Old code and new code perform the same on this
box today. Any conclusion of the form "X got slower since August" drawn from
a same-day comparison against an August number is unsupported unless this
control runs first.

RATIOS ARE NOT PROTECTED BY ALTERNATION when the arm is small. Alternating
FAST and IDENTICAL inside one window cancels drift that scales both arms.
It does NOT cancel a fixed per-launch cost, which lands in both arms
equally and therefore pulls every ratio toward 1.0 -- hardest on the arms
whose true time is smallest. `nt` at 0.09 ms of real work under about 2.6 ms
of overhead reads 1.38x on the very code that read 5.69x when the overhead
was absent. A ratio taken on a degraded box is not a conservative estimate
of the ratio; it is a different number.

So the August figures stand as the identity cost, and a re-measurement is
owed on a quiet box. `sysctl vm.swapusage` and the load average belong in
the record of any price run.
