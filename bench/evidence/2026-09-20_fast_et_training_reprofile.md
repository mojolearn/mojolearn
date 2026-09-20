# FAST Extra Trees million-row training reprofile

Scope: Apple M4 portable device trainer, deterministic synthetic binary
classification, 1,000,000 rows x 28 Float32 features, eight trees, depth 12,
`max_features=sqrt`, seed 1.  No cloud resource was provisioned.  Random
Forest uses a different histogram trainer, so no Extra Trees result below is
claimed for RF.

## Current profile

The existing `fit_once` phase clock reported an 873.005 ms unclocked fit and
a 1,053.321 ms clocked fit (1.207x serialization distortion).  The clocked
phase shares were:

| phase | ms | share |
|---|---:|---:|
| score | 337.997 | 32.3% |
| range | 243.947 | 23.3% |
| partition | 147.911 | 14.1% |
| stage + feature sampler | 111.535 | 10.6% |
| candidate/reduce/readback | 110.130 | 10.5% |
| remaining host/leaf/setup | 96.322 | 9.2% |

This confirms the repeated range and score scans remain the first target;
host queues are not currently the dominant cost.

## Rejected search tiling

The existing `MOJOLEARN_ET_SEARCH_RPT_64` measurement arm folds 64 rows per
thread, amortizing blocks in range and score.  Four balanced process groups,
three fits each, produced six usable timings per arm.  Median whole-fit time:

| arm | median ms | relative |
|---|---:|---:|
| shipped, one row/thread | 759.033 | 1.000 |
| 64 rows/thread | 817.256 | 1.077 |

All twelve forests had 38,016 nodes and digest
`be3843c01f31bb9e`; each arm reproduced itself exactly.  The candidate is
7.7% slower and remains an opt-in measurement arm.

## Rejected single-writer publish

`MOJOLEARN_ET_PLAIN_SINGLE_PUBLISH` replaces atomic range/score publication
only where a node-feature cell has exactly one writer.  The same balanced
six-fit comparison gave:

| arm | median ms | relative |
|---|---:|---:|
| shipped | 711.283 | 1.000 |
| single-writer plain publish | 757.548 | 1.065 |

Every forest again had 38,016 nodes and digest `be3843c01f31bb9e`.  It is
6.5% slower on this large Apple fixture and remains off by default.

Both failures are scheduling-only and quality-neutral on this fixture, but
neither should be promoted.  A next Extra Trees attempt should change the
range/score memory traffic rather than merely reducing their block count or
atomic form.  RF should be independently profiled in its histogram trainer.
