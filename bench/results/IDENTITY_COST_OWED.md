# The identity tax is measured on ONE vendor

**Status 2026-09-03. THIS FILE'S 2026-08-31 HEADLINE WAS WRONG IN BOTH
DIRECTIONS AT ONCE, and the corrections point opposite ways.**

It said "Apple only" and it said "No price run has ever executed on either
[H100 or MI325X]". By then a clean-window price run HAD executed on an
MI325X (2026-08-31, 5 alternated rounds, sha `035493e1`,
`bench/LANES_PRICE.md`), and an H100 run followed on 2026-09-02. So the
rented columns were not missing.

And the Apple column it treated as the solid one has since been RETRACTED.
Those figures were taken on a 16 GB laptop carrying 7.5 GB of swap, and no
run has ever reproduced them: August's own source, rebuilt in a detached
worktree, measured the Gram arm at 26.33 ms against the 5.27 ms published,
and HEAD measured the same. The code did not regress; the box was
contended. See `bench/results/lanes_price/CONTROL_2026-09-02/`.

**So the sentence to carry away is the reverse of this file's title.** The
one-vendor problem was never the shape of the gap. The gap is that a price
taken on a shared laptop understates the tax -- a fixed per-launch cost
lands in BOTH arms and drags every ratio toward 1.0, hardest on the
smallest arm -- so the LEAST confounded column is the rented single-tenant
GPU, not the machine on the desk. `tools/lanes_price.sh` now records
memory pressure at both ends of every run, because the concurrency gate
and the load average both read clean on the swapping box: load average
does not see paging.

**What is still genuinely owed:** an Apple column taken on a quiet box, and
the three tree lanes (`gbdt`, `rf`, `et`), which had never been priced on
any column until 2026-09-03.

The 2026-08-31 text follows unchanged, including its two wrong sentences,
because what it got wrong is the point.

## What exists

`bench/results/lanes_price/2026-08-28_163353_26eb8ba/env.txt`:

    host MacBook-Air-1-terrabyte
    cpu  Apple M4
    rounds 5
    lanes cd kde linkage svm metrics gemm

Six lanes, FAST against IDENTICAL, two binaries built once and ALTERNATED
`F I F I F I` so a thermal drift cannot be read as a tier difference. That is
a good measurement and it is on one box.

Other documents mention H100 and MI325X near the price discussion. Those are
PROSE. No price run has ever executed on either.

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
