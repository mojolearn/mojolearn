# Spectral's CPU column at the published 512-row size

`lane/saved-model-reference-gaps` shrank the `spectral` fixture to 512 rows
(`LANE_REVISIONS["spectral"] = "rows-512-1"`, `f4e589395`), retook the NVIDIA
and Apple columns at that size, and left this owed:

> `spectral`'s x86 CPU identity column at the 512-row size. Its Apple column
> was retaken here; the CPU one was not, and the 2026-09-15 one is superseded.

This is that column, taken on the M4's arm64 CPU host route rather than on
x86. Nine fixtures, two repeats, one core.

    PYTHONPATH=<worktree>/python MOJOLEARN_NUMERIC_MODE=identical \
      python tools/identity_break.py --lanes spectral,spectral-precomputed --repeats 2

    cells=18 stable=18 moved=0 refused=0
    infer: stable=18   model: stable=18   batch: stable=18

## The result

**All 72 cell parts equal the reference already carried by the NVIDIA A100 and
Apple Metal columns, bit for bit. 72 agreeing, 0 disagreeing, 0 without a
reference.** Compared against a table rebuilt from the records on main, whose
`spectral` and `spectral-precomputed` refs come from
`2026-09-16_predict-nvidia` (`nvidia-a100-sm_80.json`,
`apple-m4-metal.spectral-512.json`).

So `spectral` goes from ONE column at the published size to three, and
`spectral-precomputed` likewise on every fixture rather than on two.

## Both sabotage arms were SEEN TO FIRE

**Host build.** `bindings/build_metrics_host.sh` with
`MOJOLEARN_BUILD_EXTRA_DEFINES="-D MOJOLEARN_HOST_SABOTAGE=1"` into a separate
directory, swapped in by byte copy and swapped back the same way. The run's own
metadata reads the binding back as sabotaged
(`host.families._mojolearn_metrics_host.sabotage: true`), which is the check
that the arm used the binary it meant to.

**70 of 72 cell parts MOVED.** The two that did not are named rather than
rounded away: `spectral-precomputed`'s `batch` on `denormal` and
`denormal_ftz`. That part hashes `predict` over 64 held-out affinity rows, and
its output is integer cluster LABELS; on those two fixtures the perturbed
affinities did not change which cluster wins. The `infer` cell of the same two
fixtures DID move, so the arithmetic changed and only the argmax survived. This
is the same absorption `lane/stateful-cpu-decoding` documented for a one-ULP
move into an all-zero state.

**Batch.** `MOJOLEARN_IDENTITY_BATCH_SABOTAGE=1`, exit 1,
`batch: batch_moved=18` -- all eighteen, the two the host arm absorbed
included, each naming the row and both bit patterns:

    BATCH_MOVED spectral-precomputed/denormal: BATCH_MOVED:predict:row 0 of 64
      alone:output 0 element 0: whole 0x00000001 vs alone 0x00000000

## What is still owed

The AMD column, at the next release record. AMD is left alone by instruction.
