# Attention-v1 packed dy/dscore scratch — rejected

## Question

Could the production-v1 backward path store only visible causal/window cells
of `dy_st`, then overwrite those cells with dscore, while retaining the shipped
full-layout exponent stash and exact arithmetic?

## Prototype and exactness

The opt-in prototype used the existing stable packed-estash cell mapping in
three places only: the zdot writer, dQ's dy-load/dscore-store, and dK's dscore
load.  The default full-layout path was otherwise unchanged.

On Apple M4, `transformer_attention_arms_check` passed all 15 cases and all 25
arms.  Context, row maximum, denominator, zdot, dQ, dK, and dV were bitwise
identical, including tail/window cases and sabotage reachability.

## Allocation result

For B1/H12/L2048 causal attention, the full dy/dscore allocation is
201,326,592 bytes.  Packing visible cells reduces it to 100,712,448 bytes, a
100,614,144-byte saving.  A window of 512 removes approximately 75% of this
scratch.

No existing linear input is a safe replacement allocation: `dctx` remains live
through dV, Q remains live through dK, and V remains live through dQ's masked
tail/corner replay.  Each is also only about 6 MiB at the target geometry, far
smaller than the quadratic scratch.

## End-to-end forward+backward timing

Times are Apple M4 medians in milliseconds.  Each cell has 20 measured samples
(two warmups, five rounds, ordered full/packed/packed/full).

| Shape | Full dy/dscore | Packed dy/dscore | Ratio |
|---|---:|---:|---:|
| L1024 causal | 49.9025 | 85.9900 | 1.72x slower |
| L2048 causal | 197.1325 | 328.9495 | 1.67x slower |
| L2048 window512 | 84.2840 | 151.4795 | 1.80x slower |

## Disposition

Rejected and fully reverted.  Packed address calculation (`_visible_prefix`
and packed cell mapping) sits in the hot dQ/dK staging loops, and its cost
overwhelms the bandwidth/allocation benefit.  The exactness result does not
justify vendor rental because the local end-to-end regression is large and
consistent.  No cloud resource was created.
