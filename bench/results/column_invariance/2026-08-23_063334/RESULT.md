# Column invariance, 2026-08-23 — three vendor columns, one M4

`pixi run check-column-invariance`. k-means, k-NN and DBSCAN under
`NUMERIC_IDENTICAL`, compiled against `COLUMN_APPLE`, `COLUMN_NVIDIA` and
`COLUMN_AMD`, twice each. **Eighteen runs, one card and one output hash.**

Every run reports the mode it was COMPILED against (`mode IDENTICAL` in each
`.run` file) rather than the mode the harness intended, because
`mojo_only/numerics.mojo` is a shared file and a parallel session's flip
window would otherwise produce a correctly-labelled measurement of the wrong
arm. See DEVIATION 514.

## What this is

`TARGET_COLUMN` is a comptime choice, so `-D MOJOLEARN_COLUMN_AMD=1`
compiles this source against AMD's block sizes, 64-wide lane width, LDS
budget and 110-core occupancy on the attached Apple device. The vendor's
CONSTANTS are exercised; the vendor's ARITHMETIC is not. Rows 9, 10 and 12
(contraction, denormals, device transcendentals) are properties of the
backend and cannot fail here. Rows 8, 21, 22 and 23 largely can.

## The teeth: the same comparison with the pins off

Both sides verified `mode FAST` in their own output.

| comparison | cards | sorted distances | sorted indices |
|---|---|---|---|
| FAST, APPLE vs AMD column | **diverge** at `knn.out_dist` | identical | **differ** |
| IDENTICAL, APPLE vs AMD vs NVIDIA | identical | identical | identical |

Same distances, different neighbours: the signature of IDENTITY_PATHS row
11, at the column level.

And the run-to-run half, one column, one binary, one fixture:

    FAST      run 1  output.indices    413741041996730565
              run 2  output.indices  11933401367436097206
              run 3  output.indices  12727715687201500419

    IDENTICAL runs 1-3 x {APPLE, AMD}  all  16793617586120664034

Three consecutive runs, three answers. Row 11 was written as a refusal about
cross-VENDOR reproducibility; on this fixture the FAST default is not
reproducible across RUNS. `updateSortedWarpQ`'s mutex merge resolves an
equidistant tie by arrival order.

## What it does not claim

Not a cross-vendor measurement. E1 is still owed: same commit, real MI300X,
`E1_RUNBOOK.md` Phase 3u. Run `check-ieee-arith` first on that box — the
2026-08-23 AMD leg recorded "a\*b+c is UNFUSED on this backend" from the
counting arm later shown to be an artifact, so AMD's real contraction
behaviour is unmeasured.
