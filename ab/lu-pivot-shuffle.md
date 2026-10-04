# LU pivot synchronization candidate, 2026-10-04

Base64cb1211fc4e33a9962c4595f07607b74b9a816e; kernel source checkpoint
f876cca63. Binding x_decomp, define MOJOLEARN_LU_FAST_PIVOT_SHUFFLE, default
off; gated by existing LU_FAST_STEP1 (FAST+Apple). No DBUF or TSLU candidate
code imported. M2 compilation and M3 numerical validation are owed.

Why: source counts for board n8192 predict8192 panel-step launches out of9935
factorization launches, versus472 MMA updates. Each step performs two shared
256-thread argmax reductions and a pivot-read barrier, about19 block barriers.
The preceding DBUF experiment passed exact quality but gave only factor~3%
and no solve improvement in one sample; it remains held. This experiment
targets synchronization instead of repeating matrix tiling.

Implementation preserves directed comparison topology. Full reductions keep
shared128/64/32 levels, then first warp folds16/8/4/2/1 using shuffle_xor of
value and row, updating only lane<offset with unmodified _lfs_better. The
initial fold of <=32 block partials skips only upper invalid-row inputs:
_lfs_better(ov, oi<0, cv, ci) always returnsFalse, including cv=NaN. Larger
initial folds use the complete old implementation. The shared pivot read's
barrier stays, preventing scratch reuse before every thread has read ri[0].
Thread0 publishes final partials after its own exact warp fold; other warps
no longer consume that final result, so no final block barrier is needed.

Expected board step barriers19->6. This is a structural count, not speed
measurement. Panel updates/division, row swaps, pivot ties, zero-pivot info,
MMA order, tile sizes, launch geometry and per-column ordering remain intact.
Same compare tree avoids assuming NaN comparisons are associative. No new
host numerical work and no serial/global single-block pivot implementation.

The helper executes actual public lu_factor/lu_solve on each compiled arm;
it is not a Python emulation of the reduction. It reuses all eight prior
LU-MMA cases (board8192, odd tails, plain1000/plain2051, singular zero700)
and adds tie65 and tie769. Ties span lanes31/32 and block rows256/512,
respectively; the lower row must be first pivot. Both arms must have exactly
equal LU/pivot/solution bytes, equal info, finite outputs, and each float64
factor/solve residual <=A without tolerance. A compiled reach accessor must
report0 for A and1 for B. Original quality helper and historical thresholds
are unchanged. Fixtures/readbacks/oracles are harness verification only.

Manager intake: x_decomp A empty, B MOJOLEARN_LU_FAST_PIVOT_SHUFFLE. After
M2 compilation and exact-source verified staging, serial M3 commands:
  tools/lu_pivot_shuffle_pair.py quality SOURCE w2-lu-pivot-shuffle-q-20261004
  tools/lu_pivot_shuffle_pair.py timing SOURCE w2-lu-pivot-shuffle-q-20261004 w2-lu-pivot-shuffle-t-20261004
Run through the M3 board Python with FAST Apple environment. Timing requires
matching source/binary/fixture PASS receipt; output directories are exclusive.
The binding is one of runner's standard prebuilt families. Helper restores
original installed x_decomp after each pair and verifies binary hashes.

Timing protocol inherited from DBUF but this is a genuine new candidate:
one lu_factor call and one solve(A,B) call per arm on exact board8192 input,
fixed factor-then-solve order within each arm process. Full first read included,
no warmup or repetition. Cold/warm ordering recorded; no old scored arm is
replayed or relabeled. Custom call/read results do not automatically overwrite
old board timings. No defaults, rollback promotion, or board updates here.
