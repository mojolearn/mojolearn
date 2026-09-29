baseline fp32.v1: seeds [0, 1, 2, 3, 4], val loss at step 4000 mean 1.47656, noise floor 0.00850 nats (0.853% of perplexity), range 0.02116

The change is the mean over the seeds of (arm minus the baseline of the same seed) at step 4000, as a relative change of validation perplexity. The interval is that mean plus and minus t(0.975, seeds - 1) standard errors; the seeds are resampled; it bounds the run-to-run error at this shape on this corpus and says nothing about another shape or text.

| profile | products | mode | seeds | change at step 4000 | interval | change at step 6000 | interval | noise floor | inside noise | steps to baseline final | gradient cosine against fp32 (min) | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| bf16f32.v1 | projections | forward | 3 | -0.203% (-0.00203 nats) | -1.274% to +0.879% | -0.629% (-0.00631 nats) | -2.590% to +1.371% | 0.853% | yes | median 4000 | 0.999912 | PASS |
| bf16-both | projections | forward | 3 | +0.043% (+0.00043 nats) | -0.768% to +0.861% | -0.401% (-0.00402 nats) | -2.353% to +1.590% | 0.853% | yes | median 4100 | 0.999888 | PASS |
| bf16-both | projections | forward + backward | 3 | +0.001% (+0.00001 nats) | -0.616% to +0.621% | -0.276% (-0.00276 nats) | -1.924% to +1.400% | 0.853% | yes | median 4100 | 0.999889 | PASS |
| int8i32.v1 | projections | forward | 3 | -0.282% (-0.00282 nats) | -2.610% to +2.102% | -0.727% (-0.00729 nats) | -3.246% to +1.858% | 0.853% | yes | median 4000 | 0.997629 | UNDERPOWERED; dropped 2026-09-29, Andrew (not offered by the flag) |
| int8i32.v1 | projections | forward + backward | 3 | +3175.255% (+3.48898 nats) | -75.920% to +445385.156% | +2597.140% (+3.29478 nats) | -55.561% to +163597.505% | 0.853% | no | 3 of 3 seeds not within 6000 | 0.854333 | UNDERPOWERED; dropped 2026-09-29, Andrew (not offered by the flag) |
| int15-both | projections | forward | 3 | -0.229% (-0.00229 nats) | -0.877% to +0.424% | -0.283% (-0.00283 nats) | -1.864% to +1.325% | 0.853% | yes | median 4000 | 1.000000 | PASS |
| int15-both | projections | forward + backward | 3 | -0.184% (-0.00184 nats) | -1.400% to +1.046% | -0.184% (-0.00184 nats) | -0.860% to +0.497% | 0.853% | yes | median 4000 | 0.999997 | UNDERPOWERED |
| int15w-int8a | projections | forward | 3 | -0.143% (-0.00143 nats) | -0.402% to +0.116% | -0.112% (-0.00112 nats) | -1.322% to +1.112% | 0.853% | yes | median 4000 | 0.998554 | PASS; dropped 2026-09-29, Andrew (not offered by the flag) |
| int10-both | projections | forward | 3 | -0.159% (-0.00159 nats) | -1.359% to +1.055% | -0.427% (-0.00428 nats) | -1.332% to +0.486% | 0.853% | yes | median 4000 | 0.999829 | UNDERPOWERED |
| int10-both | projections | forward + backward | 3 | +44.294% (+0.36668 nats) | +7.841% to +93.069% | +63.643% (+0.49252 nats) | +21.799% to +119.862% | 0.853% | no | 3 of 3 seeds not within 6000 | 0.985409 | MISS |
| int12-both | projections | forward | 3 | +0.025% (+0.00025 nats) | -0.671% to +0.726% | -0.342% (-0.00343 nats) | -2.201% to +1.553% | 0.853% | yes | median 4300 | 0.999989 | PASS |
| int12-both | projections | forward + backward | 3 | +0.158% (+0.00158 nats) | -0.698% to +1.021% | +0.474% (+0.00473 nats) | -1.962% to +2.970% | 0.853% | yes | median 4000 | 0.999840 | UNDERPOWERED |
| int8w-int15a | projections | forward | 3 | -0.599% (-0.00600 nats) | -2.650% to +1.496% | -0.801% (-0.00804 nats) | -3.274% to +1.735% | 0.853% | yes | median 4000 | 0.998346 | UNDERPOWERED; dropped 2026-09-29, Andrew (not offered by the flag) |
| bf16-both+attn | projections + attention | forward + backward | 1 | +0.412% (+0.00411 nats) | +nan% to +nan% | - | - | 0.853% | yes | median 4200 | 0.999914 | UNDERPOWERED |
| int15-both+attn (F1) | projections + attention | forward | 5 | +0.298% (+0.00298 nats) | -0.287% to +0.887% | -0.354% (-0.00355 nats) | -1.389% to +0.691% | 0.853% | yes | median 4100 | 1.000000 | PASS |
| int15-both+attn (F1) | projections + attention | forward + backward | 1 | -0.366% (-0.00366 nats) | +nan% to +nan% | - | - | 0.853% | yes | median 3700 | 0.999998 | UNDERPOWERED |
