baseline fp32.v1: seeds [0, 1, 2, 3, 4], val loss at step 4000 mean 1.47656, noise floor 0.00850 nats (0.853% of perplexity), range 0.02116

The change is the mean over the seeds of (arm minus the baseline of the same seed) at step 4000, as a relative change of validation perplexity. The interval is that mean plus and minus t(0.975, seeds - 1) standard errors; the seeds are resampled; it bounds the run-to-run error at this shape on this corpus and says nothing about another shape or text.

| profile | products | mode | seeds | change at step 4000 | interval | change at step 6000 | interval | noise floor | inside noise | steps to baseline final | gradient cosine against fp32 (min) | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| int15-both+attn (F1) | projections + attention | forward | 5 | +0.298% (+0.00298 nats) | -0.287% to +0.887% | -0.354% (-0.00355 nats) | -1.389% to +0.691% | 0.853% | yes | median 4100 | 1.000000 | PASS |
| int15-both+attn (F1) | projections + attention | forward + backward | 5 | +0.146% (+0.00146 nats) | -0.275% to +0.569% | -0.024% (-0.00024 nats) | -0.619% to +0.575% | 0.853% | yes | median 4100 | 0.999997 | PASS |
