# The identity tax is measured on ONE vendor

**Status 2026-09-03: measured on AMD MI325X. Apple is the column that is
owed, and that is the opposite of what this file used to say.**

The identity tax is priced by `tools/lanes_price.sh` on rented single-tenant
GPUs. The MI325X column landed 2026-09-03 at sha `9d4aabb`: nine lanes, three
alternated rounds, zero swap, nothing else on the device, trees at 1,000,000
rows.

The Apple figures this file was built around are RETRACTED. They were taken on
a 16 GB laptop carrying 7.5 GB of swap and have never been reproduced -- the
same source rebuilt in a worktree measured the Gram arm at 26.33 ms against
the 5.27 ms published. A contended box adds a fixed per-launch cost to BOTH
arms, which pulls every ratio toward 1.0 and hardest on the smallest arm, so
it UNDERSTATES the tax. Evidence: `bench/results/lanes_price/CONTROL_2026-09-02/`.

So a rented GPU is the LESS confounded column, not the more. `lanes_price.sh`
records memory pressure at both ends of a run, because the concurrency gate
and the load average both read clean on the swapping box.

**Owed:** an Apple column on a quiet box, and an H100 column.

## Why one box is not enough here, specifically

The tax is not one number. It is `ftz`, a pinned `fma`, `portable_sqrtf` and
the fold pins, and what each costs depends on what the hardware gives you for
free. Apple's `fmin`/`fmax` and its sqrt already agree with the pinned forms,
which is why row 10 exists at all; NVIDIA's approximate PTX sqrt does not,
which is what DEVIATION 550 is about. So the Apple number is plausibly the
CHEAPEST of the three and quoting it as "the" cost is the direction that
flatters us.

`[[one-box-verdict-is-not-three]]` and `[[identity-is-not-free]]` both
already forbid generalizing this. The measurement is what is missing.

## What closes it, exactly

`tools/lanes_price.sh` is portable: a shell script that builds the FAST and
the IDENTICAL binary, then alternates two ready binaries with no compile
inside the window. It needs no Mac-specific anything. So it rides a leg as a
phase-9 diag, the same way `packaging/linux/leg_diag.sh` and
`tools/diag/rbc_551_leg.sh` do:

    MOJOLEARN_E1_PHASES=9 MOJOLEARN_P9_ONLY_DIAG=1 \
    MOJOLEARN_P9_DIAG=tools/lanes_price.sh \
    bash tools/e2_remote_leg.sh amd <token>          # MI325X

    MOJOLEARN_GEMM_LEG_E1_PHASES=9 MOJOLEARN_GEMM_LEG_P9_ONLY_DIAG=1 \
    MOJOLEARN_GEMM_LEG_P9_DIAG=tools/lanes_price.sh \
    sh tools/gemm_remote_leg.sh nvidia --payload phase8 --rent   # H100

Two legs, six lanes each, at ONE shared commit with the Apple run repeated at
that same commit so all three columns are comparable. The Apple number in the
table above is at `26eb8ba` and the round would need re-taking at whatever sha
the legs use.

## What must not happen

Do not report a three-vendor cost by pairing the Apple price run with the
NVIDIA and AMD SPEED boards. Those are FAST on both sides and record
`ours_headers_identical=0`; they answer a different question. A tier
comparison has to alternate two binaries in one window on one box, which is
the whole design of `lanes_price.sh`.
