# RF sequential accumulator reuse: rejected

The IDENTICAL sequential `RandomForest.predict` and `predict_proba` paths
allocated their `num_outputs`-cell row accumulator inside the row loop.  A
candidate allocated it once per public call and reset the same cells before
each row.  Tree order, class order, division, argmax, and traversal were
unchanged.

Apple M4 qualification used the public IDENTICAL binding, 12 trees, and four
timed `predict` calls after fitting.  Times below are milliseconds in run
order; inputs and fitted configurations were reconstructed identically for
the baseline and candidate builds.

| seed/classes/features/depth/rows | baseline | reused accumulator |
|---|---|---|
| 7/2/8/6/1,000,000 | 149.302, 145.650, 144.985, 144.905 | 149.525, 151.328, 148.743, 147.410 |
| 19/3/16/8/1,000,000 | 184.980, 183.563, 186.772, 179.405 | 185.832, 180.818, 185.096, 181.059 |
| 41/8/32/10/250,000 | 96.782, 94.917, 94.644, 95.184 | 95.895, 99.321, 97.136, 95.352 |

Complete prediction and probability buffers were bitwise identical:

| case | prediction SHA-256 | probability SHA-256 |
|---|---|---|
| 7/2/8/6/1M | `51b83fdcda030998a2f951134b745860e3b1512d6c91a5cc743bb56d77ab7396` | `afb659e23049aa72873bbf3f6ebeb9d882dfcc1c06e7d32adb8155a8bb5e19c9` |
| 19/3/16/8/1M | `287639d5265c3da42a4b6732525b0e6132b5e6dae6458a2306a18dd2554a17ca` | `0e9b23394106659190a12d29b47373a9b91a825f20d6212bc49173f12f637f01` |
| 41/8/32/10/250k | `ad8bb1c50bf636395f8e9b24784565b1ff958662c3342e5ad7ca7df82b4c4429` | `2ffd95ed27b4ad2a7c9040f0cc00fe1042f9b4c1bc547cf81a9ea1dd1f50886a` |

`ensemble/checks/predict_check.mojo` also passed with the candidate.  The
candidate was neutral to slower and is fully reverted.  The likely reason is
that the compiler already scalarizes this tiny row-local list, while explicit
reuse introduces stores across loop iterations.  Do not promote or retry this
shape without profiler evidence that allocation survives optimization.
