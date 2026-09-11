# ExtraTrees on the Apple M4, 2026-09-11: phase split at HIGGS 1M (FAST)

`extratrees/bench/fit_once.mojo` with `phases` (its `PhaseClock`, one
drain per phase, so the clocked total runs about 6% over the unclocked
fit), HIGGS first 1,000,000 rows x 28, 100 trees, depth 16, sqrt features,
classification; FAST tier, MAX 26.5. `fit_once.higgs1m.phases.log` is the
run below; `speed.fast.et.stage.log` is the harness's whole-fit number the
same night (11,639 ms, hash 2c192f6b12dbb6c5, the H100 hash).

| phase | ms | share |
|---|---:|---:|
| stage + feature sampler | 1,981 | 17% |
| range pass (init, range, decode, nonconst) | 3,001 | 26% |
| score pass (init, score, finalize) | 3,808 | 33% |
| partition (4 kernels) | 2,447 | 21% |
| candidate, reduce, splits readback | 169 | 1.4% |
| leaf pass | 80 | 0.7% |
| host (split records, pop, push) | 111 | 1.0% |
| unclocked fit | 11,006 | |

A Metal System Trace of one fit (not committed, 27 MB) agrees: 8.7 s of
GPU compute in 4,923 launches, 183 launches over 10 ms carrying 4.9 s,
2.9 s of GPU idle. ExtraTrees has none of the RandomForest builder's
host tax (its frontier batches span trees, DEVIATION 211, and it issues no
device memsets): the M4 time is kernel time in the range, score and
partition passes. That is a kernel lane, not tonight's pattern.
