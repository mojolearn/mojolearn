baseline fp32.v1: seeds [0, 1, 2, 3, 4], val loss at step 4000 mean 1.47656, noise floor 0.00850 nats (0.853% of perplexity), range 0.02116

The change is the mean over the seeds of (arm minus the baseline of the same seed) at step 4000, as a relative change of validation perplexity. The interval is that mean plus and minus t(0.975, seeds - 1) standard errors; the seeds are resampled; it bounds the run-to-run error at this shape on this corpus and says nothing about another shape or text.

| profile | products | mode | seeds | change at step 4000 | interval | change at step 6000 | interval | noise floor | inside noise | steps to baseline final | gradient cosine against fp32 (min) | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| F1-pv32 (the one that would ship) | projections + attention | forward | 5 | -0.102% (-0.00102 nats) | -0.661% to +0.460% | +0.006% (+0.00006 nats) | -0.470% to +0.484% | 0.853% | yes | median 4000 | 1.000000 | PASS |
| F1-pv32 (the one that would ship) | projections + attention | forward + backward | 5 | -0.297% (-0.00297 nats) | -0.698% to +0.106% | -0.344% (-0.00345 nats) | -1.312% to +0.633% | 0.853% | yes | median 4000 | 0.999997 | PASS |
