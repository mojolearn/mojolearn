# par-scaler and par-scaler-minmax: the arm was always firing (2026-09-17)

The UNFIXED side is committed one directory over, in `a-linear-neighbors/`:
under the preprocessing family's sabotage build `par-scaler` raised
`transform_scaler and plain transform differ: 2847 bytes of 16384` on `base`
and 2845 on `ties`, and the cell read **REFUSED** while the clean cell read
STABLE. The negative control was working. The lane swallowed it, and a refusal
is not a catch, so the matrix left the lane at `declared`.

With the disagreement raised as `NumericalMismatch` carrying the parts:

| cell | clean | sabotage |
|---|---|---|
| `par-scaler/base` | STABLE `c4cc661617a4b9a7` | DIVERGENT `20fb67a0e5b04ad2` |
| `par-scaler/ties` | STABLE `1ecadb87b13bb827` | DIVERGENT `da4acec3c37a4c16` |
| `par-scaler-minmax/base` | STABLE `646b46cf73ead58e` | DIVERGENT `b7c8442880f1de0a` |
| `par-scaler-minmax/ties` | STABLE `fcaf2c976dae67d9` | STABLE `f1d79cc1bb919200` |

`par-scaler-minmax/ties` is the interesting one: there the two transforms still
AGREE with each other under the sabotage build, so nothing raises, and the
value moved anyway. That cell would have counted before the repair; the other
three would not.

**The clean cell did not move**, which is the whole point of doing it this way
rather than by adding a part. `diff.clean-unmoved.txt` diffs this run's clean
column against the `a-linear-neighbors` clean column taken BEFORE the repair:
`summary: IDENTICAL=2`, `summary (infer/model): IDENTICAL=4`,
`summary (batch): IDENTICAL=2`. No committed record of either lane is
invalidated and no `LANE_REVISIONS` entry is needed.

One 8 vCPU pod, DELETE verified, $0.0129.
