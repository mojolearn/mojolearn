# HANDOFF: the dbscan `DeadArgumentElimination` compiler crash (2026-09-09)

Lane branch `lane/dbscan-compiler-crash`, based on main at 71d2ba71.

## Commits

- `27059d23 parent 71d2ba71` -- the fix, the reduced repro, the `-O1`
  workaround removed from `pixi.toml` (`check-dbscan`),
  `tools/check_unsupervised_identity.sh` and `tools/e1_unsupervised.sh`,
  the 2026-09-01 "the cure was -O1" paragraphs corrected in place
  (`dbscan/dbscan_main.mojo`, `dbscan/checks/dbscan_check.mojo` block
  comment and the three candidate sites, `IDENTITY_PATHS.md` row 64), and
  the draft upstream report `docs/lanes/MODULAR_ISSUE_dead_arg_elim.md`
  (NOT filed).
- the commit carrying this file.

A note on two files the brief named: `PORTING.md` does not exist at
71d2ba71 and `dbscan/README.md` at that commit no longer carries the
"-O1 is the cure" paragraphs (both were rewritten by the Sep 9 lane
merges). The surviving record of the -O1 cure was `IDENTITY_PATHS.md` row
64, `dbscan/dbscan_main.mojo`, the check file and `pixi.toml`, and all
four are corrected. The historical text remains in commit c0922140.

## The triggering construct, and the fix

`dbscan/checks/dbscan_check.mojo`, `_host_weighted_degree_strided` (the
host oracle both the fold-pin gate and the degree-oracle gate call):

    while k < len(cols):          # len() of a borrowed List argument,
        acc = ftz(acc + ...)      # re-read every iteration ...
        k += width                # ... with a RUNTIME Int argument as the step

That pair, and only that pair, makes Mojo 1.0.0 (ed45d567) abort at -O2
and above with `DeadArgumentElimination surveyUse failed. UNREACHABLE
executed!`. The fix is one line, `var n = len(cols)` before the loop
(`while k < n`). `cols` is a read-only argument, so the bound is
loop-invariant and the arithmetic is unchanged in value and in order.

The reduced repro is `dbscan/checks/compiler_repro_dead_arg_elim.mojo`
(host only, no GPU, 20 lines, asserts at -O2 and -O3, builds at -O1 and
-O0). Its header lists the ladder.

Why the 2026-09-01 record saw "two independent triggers": one entry point
per gate on the L40S shows the fold gate ALONE asserts (6 s) and the
degree-oracle gate ALONE asserts (10 s), while the uniform-weight gate,
the duplicate gate and the 12 other gates each build clean at -O3. The two
that assert are exactly the two callers of that function. None of the
four 2026-09-01 rewrites touched that loop, which is why all four failed.

## Evidence (NVIDIA L40S RunPod pod 4dlcwf8k5zabuj, driver 595.91.07, Mojo 1.0.0 ed45d567, 128 cores)

Reproduce (unfixed 71d2ba71 source; `dbscan/` and `checks/` are
byte-identical between f3d76e8d and 71d2ba71):

    full_O3    rc=134 secs=30  DeadArgumentElimination surveyUse failed
    full_O1    rc=0   secs=36  clean
    fold_O3    rc=134 secs=6   asserts    (fold gate alone)
    oracle_O3  rc=134 secs=10  asserts    (degree-oracle gate alone)
    uniform_O3 rc=0   secs=19  clean
    dup_O3     rc=0   secs=20  clean
    none_O3    rc=0   secs=41  clean      (the 12 others, no weighted gate)

Reduction ladder, 34 single-variable files, all `mojo build -O3`
(`r18` is the committed repro):

    asserts: r1 (fold gate verbatim), r2 (no matrix), r3 (no ftz), r5 (strided
             partials only), r6 (one call site), r7 (no inner List), r8 (no
             cols indirection), r10 (@no_inline), r11 (split index), r12
             (List[Int32]), r13 (unsafe_ptr load), r14 (owned arg), r15
             (runtime-derived width), r16 (bare strided while), r18 (ONE List
             arg, bare strided while), r19 (List[Int]), r22 (both Int), r27
             (local cols list)
    builds:  r4 (halving fold only), r17 (for range with stride), r20 (fixed
             stride), r21 (unit-stride for), r24 (@always_inline), r28 (len()
             hoisted), r29 (Int bound argument), r30 (r18 + hoist = THE FIX),
             r31 (strided range), r32 (comptime stride), r33 (unit stride, len
             re-read), r34 (stride not an argument)
    parse errors, not evidence: r9/r23 (`fn` is removed in Mojo 1.0), r25
             (Span), r26 (UnsafePointer) -- spelling, not tried further

Fix verified (fixed source in `/root/ml2`, unfixed in `/root/ml` as the
control), `verify/summary.txt`:

    build fast_O3    rc=0    secs=75  clean
    build fast_O1    rc=0    secs=73  clean
    build id_O3      rc=0    secs=73  clean     (-D MOJOLEARN_NUMERIC_IDENTICAL=1)
    build id_O1      rc=0    secs=70  clean
    build repro_O3   rc=134  secs=6   DeadArgumentElimination surveyUse failed
    build repro_O2   rc=134           DeadArgumentElimination surveyUse failed
    build repro_O1   rc=0    secs=7   clean      prints "1.0000151 1.000015"
    build unfixed_O3 rc=134  secs=67  DeadArgumentElimination surveyUse failed
    run fast_O3  rc=0 checks=17 OK=17
    run fast_O1  rc=0 checks=17 OK=17
    run id_O3    rc=0 checks=17 OK=17
    run id_O1    rc=0 checks=17 OK=17
    fast: O3 vs O1 byte-identical after dropping mbind lines (3248 bytes, 17 lines)
    id:   O3 vs O1 byte-identical after dropping mbind lines (3248 bytes, 17 lines)

The "mbind lines" are tcmalloc's `Warning: Unable to mbind memory`
stderr line, one or two per process on this NUMA box, prefixed with the
process id; they are allocator noise, not check output, and are the ONLY
difference between the raw -O3 and -O1 outputs.

The literal tasks, as the RUN OWED spells them:

    pixi run check-dbscan                               rc=0 checks=17 OK=17
    sh tools/with_identical_mode.sh pixi run check-dbscan   rc=0 checks=17 OK=17
    sh tools/check_unsupervised_identity.sh   IDENTICAL pass rc=0 (all six files,
                                              dbscan_main at the default level
                                              through the gate itself);
                                              FAST pass rc=1, see below

The fold gate's printed line, both levels, both modes:
`check_dbscan_weighted_fold_is_pinned OK: WVD_TPB = 128 ... 1.0000151 at
128, 1.000015 at 64, 1.0 left to right`, which is the Apple M4's number
from c0922140.

## A finding for ANOTHER lane (not caused here, not fixed here)

`tools/check_unsupervised_identity.sh`'s FAST pass on the L40S fails at
`neighbors/knn_main.mojo`: `Unhandled exception ... 1 of 512 returned
neighbors are not in the true k-nearest set`. The gate's FAST loop stops
at the first failure, so `dbscan_main` did not run through the gate in
FAST there (it did run directly, `pixi run check-dbscan`, 17/17). This
lane's diff touches nothing under `neighbors/`; the kNN lane owns it. It
may be a pre-existing NVIDIA FAST-arm tie or ordering issue; nobody
measured whether it reproduces at 71d2ba71 without this lane's change
(it cannot depend on it), nor on the M4.

## RUN OWED on the Apple M4 (orchestrator, one at a time)

    pixi run check-dbscan
        expect 17 `check_... OK` lines and the same fold-pin numbers as
        above; this is the default level now, no -O1 anywhere
    tools/with_identical_mode.sh pixi run check-unsupervised-identity
    pixi run check-unsupervised-identity
        expect "unsupervised identity: both modes green." (the L40S FAST
        pass failed on knn_main, see above; if the M4 does the same it is
        the kNN lane's)
    optional, the repro itself:
    pixi run mojo build -O3 -I . dbscan/checks/compiler_repro_dead_arg_elim.mojo
        expect the assertion (it is the compiler's, not ours); at -O1 it
        prints "1.0000151 1.000015"

## Unfinished, and why

- The Apple M4 has not run the fixed file at the default level; the crash
  was first seen there and the fix is verified on Linux x86-64 only. The
  construct is host-only and target-independent, and the same compiler
  build is pinned on both, so the expectation is a green, but that is a
  RUN OWED, not a result.
- The upstream report is a draft; Andrew files it or not.
- The three-vendor leg for the weighted NUMBERS (H100, MI325X) that
  c0922140 owed is still owed; this lane measured build/run/diff on one
  NVIDIA box only, and the weighted gates passing there is a first
  NVIDIA green for those four, not the leg.
- `bench/results/runpod_leases/4dlcwf8k5zabuj.lease` was written by the
  guard on this Mac and is untracked; the pod is terminated (see below),
  so `tools/runpod_guard.sh reap` will find nothing to do.

## Pod

Created 17:24Z, armed 60 min via `tools/runpod_guard.sh arm`, terminated
by `DELETE /v1/pods/4dlcwf8k5zabuj` at the end of the session and
verified absent from the listing; no `samba-*` pod was touched.
