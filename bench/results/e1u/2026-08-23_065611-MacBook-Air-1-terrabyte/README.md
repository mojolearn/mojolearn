# E1 Phase 3u — the APPLE reference leg for rows 19-26

`sh tools/e1_unsupervised.sh` on the M4, under `NUMERIC_IDENTICAL`, at the
commit in `commit.txt`. This is the leg an AMD or NVIDIA box diffs against;
it exists so the second machine only has to produce its own directory.

**Mode read back from each run, not assumed from the flip** (DEVIATION 514):
all three cards report `mode IDENTICAL`. Zero `E1U-FINDING` lines.

## Column

    column 1 (APPLE)   lane_width 32

## Vendor characterization — row 10's precondition, run FIRST

    ieee arith check OK: correctly rounded ON NORMALS, with FLUSH-TO-ZERO
      on denormal operands, intermediates and results
    ftz model: reproduces 53041 of 53041 div/sqrt divergences bit-for-bit
    contraction: a*b+c is FUSED -- read off the 1629 patterns that SEPARATE
      the two spellings, never the tie-dominated totals

That last line is the corrected verdict (IDENTITY_PATHS row 9). The
tie-dominated totals in the same output still read `unfused 1046394 fused
0`, and they are exactly the artifact the correction is about: **read the
separating count, never the totals.** An AMD leg reporting "UNFUSED" off
the totals is reporting nothing.

Row 12's certificate line, which must be the SAME NUMBER on every vendor:

    translog device hash = 8705486125800438413

## The cards

| arm | stages | inputs | outputs |
|---|---|---|---|
| kmeans | 77 | x `14006717752511810141`, centroids `2727533609010192784` | centroids `1636775608130736985`, labels `9543594618727214305` |
| knn | 6 | index `7900684798707350708`, queries `17351274754253287356` | distances `9794987834769335813`, indices `16793617586120664034` |
| dbscan | 3 | x `8706409177216062216` | labels `14389807861238588709` |

Compare `input.*` FIRST. The fixtures are integer-exact functions of a
constant seed (every coordinate is a small integer over a power of two, so
it is exact in Float32 on any backend); if the input hashes differ, the two
machines did not fit the same bytes and nothing below is meaningful.

The k-NN `output.indices` hash is the interesting one: its fixture plants a
43-member tie class and puts half the queries exactly on the planted point,
so the answer is decided by the selector's tie rule rather than by the
distances. Under FAST that same fixture returned three different values in
three consecutive runs on this machine
(`bench/results/column_invariance/2026-08-23_063334/RESULT.md`).

## Provenance, verified rather than asserted

This leg was produced in the main checkout, which at the time carried other
lanes' UNCOMMITTED edits to `mojo_only/numerics.mojo`,
`mojo_only/kernel_matrix.mojo` and `core/gemm.mojo` -- all three on the
unsupervised path. `commit.txt` would then have been a claim about a tree
that did not exist, which is exactly the thing E1's first precondition
("same commit on both sides") is about.

So it was checked instead of argued. A detached worktree at `2432e07`
(`git worktree add --detach`), its own pixi environment, its own
`numerics.mojo`, and the same driver:

    kmeans  cards BYTE-IDENTICAL
    knn     cards BYTE-IDENTICAL
    dbscan  cards BYTE-IDENTICAL

So the other lanes' in-flight work does not reach these bits, and this
directory is what the commit produces. The result is a measurement with a
date on it, not a property: **re-run the worktree comparison before
trusting any future leg taken from a dirty checkout.**

### The worktree is the better instrument, and supersedes the lock here

DEVIATION 514 put the shared build lock around the mode flip because
`numerics.mojo` is one file and the sessions are many. A detached worktree
is strictly stronger for an identity leg: it has its OWN `numerics.mojo`, so
a parallel session's flip cannot reach it at all, and its HEAD is a commit
rather than a working tree. The lock is still needed for gates that must run
in the main checkout (`check-unsupervised-identity`, which is about the
shipped tree); anything that produces a CARD should come off a worktree.
