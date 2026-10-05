Cases: 178 (7 of them the simulation's).

| plan | h100 | m3ultra | m2pro | mi325x |
|---|---|---|---|---|
| host oracle (the reference) | 178 cases | 178 cases | 178 cases | 178 cases |
| FLAT kernel, codes, Int64 sum | 156 of 156 equal | 156 of 156 equal | 156 of 156 equal | 156 of 156 equal |
| PIECES kernel, planes, three Int32 sums | 156 of 156 equal | 156 of 156 equal | 156 of 156 equal | 156 of 156 equal |
| MMA, integer matrix unit, four products | 156 of 156 equal | does not run here | does not run here | 156 of 156 equal |
| entry point for codes | 156 of 156 equal | 156 of 156 equal | 156 of 156 equal | 156 of 156 equal |
| entry point for planes | 156 of 156 equal | 156 of 156 equal | 156 of 156 equal | 156 of 156 equal |
| float32 in: device quantizer, split, product | 22 of 22 equal | 22 of 22 equal | 22 of 22 equal | 22 of 22 equal |
| float32 in: parallel quantizer, product | no line | no line | no line | no line |
| the PyTorch simulation's exported product | 7 of 7 equal | 7 of 7 equal | 7 of 7 equal | 7 of 7 equal |

Host oracles across boxes (h100, m3ultra, m2pro, mi325x): THE SAME DIGEST on every case.

| phase (expected) | h100 | m3ultra | m2pro | mi325x |
|---|---|---|---|---|
| int15 (must pass) | PASSED, as expected | PASSED, as expected | PASSED, as expected | PASSED, as expected |
| int15-force-flat (must pass) | PASSED, as expected | PASSED, as expected | PASSED, as expected | PASSED, as expected |
| int15-sabotage (must fail) | FAILED, as expected | FAILED, as expected | FAILED, as expected | FAILED, as expected |
| int15-host-sabotage (must fail) | FAILED, as expected | FAILED, as expected | FAILED, as expected | FAILED, as expected |
| int15-piece-sabotage (must fail) | FAILED, as expected | FAILED, as expected | FAILED, as expected | FAILED, as expected |
| sim (must pass) | PASSED, as expected | PASSED, as expected | PASSED, as expected | PASSED, as expected |
| sim-host-sabotage (must fail) | FAILED, as expected | FAILED, as expected | FAILED, as expected | FAILED, as expected |
| sim-convert-sabotage (must fail) | FAILED, as expected | FAILED, as expected | FAILED, as expected | FAILED, as expected |
| sim-device-sabotage (must fail) | FAILED, as expected | FAILED, as expected | FAILED, as expected | FAILED, as expected |

h100: 12 gates seen failing under the arm that must fail them
m3ultra: 12 gates seen failing under the arm that must fail them
m2pro: 12 gates seen failing under the arm that must fail them
mi325x: 12 gates seen failing under the arm that must fail them

verdict=IDENTICAL on every box that ran: every plan's digest is the host oracle's, the host oracles agree across boxes, and every arm failed the gates it must
