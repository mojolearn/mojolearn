# The fifteen-bit GEMM beside fp32.v1: the complete operation

Profile `mojolearn.identical.gemm.int15i64.v1`. Harness `bench/gemm_int15_price_main.mojo`. Every number is one box's median over the timed calls of one run; `over` is that time over fp32.v1's at the same row in the same run, so above 1 the fifteen-bit operation took longer.

## M3 Ultra, run 2 (1790655096221, ae56d81c3)

Column `apple`. The product's plan on this box: no integer matrix unit; the flat kernel on codes and the pieces kernel on planes, one thread per cell, and the FLOAT matrix unit in exact chunks (two products carried every 8 steps, four products carried every 512). Each cell names the plan that took the least time at that row; the float-unit plans are not dispatched yet.

### The Apple float unit: the two forms beside the flat kernel, the product alone

| row | fp32.v1 ms | flat ms | over | two products ms | over | four products ms | over |
|---|---:|---:|---:|---:|---:|---:|---:|
| qkv.t1 | 0.9750 | 0.7960 | 0.816 | 1.5680 | 1.608 | 1.7030 | 1.747 |
| qkv.t8 | 0.5760 | 1.2350 | 2.144 | 1.6210 | 2.814 | 1.7920 | 3.111 |
| qkv.t512 | 3.3740 | 39.9430 | 11.838 | 17.1210 | 5.074 | 11.5210 | 3.415 |
| qkv.t512.bwd_dx | 3.3950 | 40.0120 | 11.786 | 17.1870 | 5.062 | 11.5130 | 3.391 |
| qkv.t512.bwd_dw | 2.9020 | 40.1090 | 13.821 | 15.4050 | 5.308 | 10.1650 | 3.503 |
| mlp_up.t1 | 1.0420 | 0.8210 | 0.788 | 1.6100 | 1.545 | 1.7470 | 1.677 |
| mlp_up.t8 | 1.0380 | 2.6970 | 2.598 | 1.5830 | 1.525 | 1.7620 | 1.697 |
| mlp_up.t512 | 10.4220 | 137.82 | 13.224 | 58.6360 | 5.626 | 35.5750 | 3.413 |
| mlp_up.t512.bwd_dx | 11.5260 | 139.21 | 12.078 | 61.2590 | 5.315 | 40.5650 | 3.519 |
| mlp_up.t512.bwd_dw | 9.1580 | 139.68 | 15.252 | 51.7910 | 5.655 | 34.0570 | 3.719 |
| mlp_down.t1 | 2.8860 | 2.0750 | 0.719 | 4.8970 | 1.697 | 5.5190 | 1.912 |
| mlp_down.t8 | 1.3840 | 3.6900 | 2.666 | 4.9110 | 3.548 | 5.5370 | 4.001 |
| mlp_down.t512 | 11.5060 | 139.03 | 12.083 | 61.2600 | 5.324 | 41.5280 | 3.609 |
| mlp_down.t512.bwd_dx | 10.4670 | 137.83 | 13.168 | 56.1470 | 5.364 | 34.7820 | 3.323 |
| mlp_down.t512.bwd_dw | 9.1970 | 139.55 | 15.174 | 53.7770 | 5.847 | 34.1170 | 3.710 |
| lm_head.t1 | 4.0960 | 3.0710 | 0.750 | 8.7130 | 2.127 | 9.8860 | 2.414 |
| lm_head.t8 | 5.6610 | 20.6100 | 3.641 | 8.8430 | 1.562 | 10.0720 | 1.779 |
| lm_head.t512 | 11.5760 | 154.32 | 13.331 | 62.6690 | 5.414 | 39.3270 | 3.397 |
| lm_head.t512.bwd_dx | 17.6030 | refused | refused | refused | refused | refused | refused |
| lm_head.t512.bwd_dw | 10.1950 | 156.30 | 15.331 | 58.1300 | 5.702 | 37.9770 | 3.725 |

### Inference: one call, weights packed once

| row | m x n x k | fp32.v1 ms | product alone ms | over | activations to planes ms | complete call ms | over | complete call, row quantizer ms | over |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| qkv.t1 | 1 x 4096 x 4096 | 0.9750 | 0.7960 (flat) | 0.816 | 0.2950 | 0.7870 (codes) | 0.807 | 1.7350 | 1.779 |
| qkv.t8 | 8 x 4096 x 4096 | 0.5760 | 1.2350 (flat) | 2.144 | 0.3010 | 1.3170 (codes) | 2.286 | 2.9690 | 5.155 |
| qkv.t512 | 512 x 4096 x 4096 | 3.3740 | 11.5210 (apple.four) | 3.415 | 0.3770 | 11.6890 (apple.four) | 3.464 | 79.1570 | 23.461 |
| mlp_up.t1 | 1 x 14336 x 4096 | 1.0420 | 0.8210 (flat) | 0.788 | 0.2680 | 0.8170 (codes) | 0.784 | 1.7630 | 1.692 |
| mlp_up.t8 | 8 x 14336 x 4096 | 1.0380 | 1.5830 (apple.two) | 1.525 | 0.2660 | 1.6450 (apple.two) | 1.585 | 5.1570 | 4.968 |
| mlp_up.t512 | 512 x 14336 x 4096 | 10.4220 | 35.5750 (apple.four) | 3.413 | 0.3810 | 35.0630 (apple.four) | 3.364 | 257.53 | 24.710 |
| mlp_down.t1 | 1 x 4096 x 14336 | 2.8860 | 2.0750 (flat) | 0.719 | 0.3070 | 2.1330 (codes) | 0.739 | 5.4960 | 1.904 |
| mlp_down.t8 | 8 x 4096 x 14336 | 1.3840 | 3.6900 (flat) | 2.666 | 0.2920 | 3.7760 (codes) | 2.728 | 9.2910 | 6.713 |
| mlp_down.t512 | 512 x 4096 x 14336 | 11.5060 | 41.5280 (apple.four) | 3.609 | 0.7140 | 41.0460 (apple.four) | 3.567 | 263.06 | 22.863 |
| lm_head.t1 | 1 x 128256 x 4096 | 4.0960 | 3.0710 (flat) | 0.750 | 0.2980 | 3.1310 (codes) | 0.764 | 6.5740 | 1.605 |
| lm_head.t8 | 8 x 128256 x 4096 | 5.6610 | 8.8430 (apple.two) | 1.562 | 0.3010 | 8.9600 (apple.two) | 1.583 | 42.2240 | 7.459 |
| lm_head.t512 | 512 x 16032 x 4096 (capped) | 11.5760 | 39.3270 (apple.four) | 3.397 | 0.3810 | 39.1890 (apple.four) | 3.385 | 289.05 | 24.970 |

### Training shapes: one product, both operands converted per call

| row | m x n x k | fp32.v1 ms | product alone ms | over | A to planes ms | B to planes ms | complete product ms | over | complete, row quantizer ms | over |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| qkv.t512 | 512 x 4096 x 4096 | 3.3740 | 11.5210 (apple.four) | 3.415 | 0.3770 | 1.2640 | 12.7000 (apple.four) | 3.764 | 80.6770 | 23.911 |
| qkv.t512.bwd_dx | 512 x 4096 x 4096 | 3.3950 | 11.5130 (apple.four) | 3.391 | 0.3840 | 1.3170 | 12.6910 (apple.four) | 3.738 | 82.7090 | 24.362 |
| qkv.t512.bwd_dw | 4096 x 4096 x 512 | 2.9020 | 10.1650 (apple.four) | 3.503 | 0.4360 | 0.4320 | 10.6200 (apple.four) | 3.660 | 77.3220 | 26.644 |
| mlp_up.t512 | 512 x 14336 x 4096 | 10.4220 | 35.5750 (apple.four) | 3.413 | 0.3810 | 3.9050 | 38.8600 (apple.four) | 3.729 | 260.01 | 24.948 |
| mlp_up.t512.bwd_dx | 512 x 4096 x 14336 | 11.5260 | 40.5650 (apple.four) | 3.519 | 0.6530 | 5.1170 | 45.1720 (apple.four) | 3.919 | 277.42 | 24.069 |
| mlp_up.t512.bwd_dw | 14336 x 4096 x 512 | 9.1580 | 34.0570 (apple.four) | 3.719 | 0.7630 | 0.4450 | 34.8480 (apple.four) | 3.805 | 284.36 | 31.050 |
| mlp_down.t512 | 512 x 4096 x 14336 | 11.5060 | 41.5280 (apple.four) | 3.609 | 0.7140 | 3.7730 | 44.8030 (apple.four) | 3.894 | 269.88 | 23.456 |
| mlp_down.t512.bwd_dx | 512 x 14336 x 4096 | 10.4670 | 34.7820 (apple.four) | 3.323 | 0.3720 | 4.0590 | 38.9670 (apple.four) | 3.723 | 262.42 | 25.071 |
| mlp_down.t512.bwd_dw | 4096 x 14336 x 512 | 9.1970 | 34.1170 (apple.four) | 3.710 | 0.4330 | 0.7530 | 34.9760 (apple.four) | 3.803 | 275.58 | 29.965 |
| lm_head.t512 | 512 x 16032 x 4096 (capped) | 11.5760 | 39.3270 (apple.four) | 3.397 | 0.3810 | 4.3400 | 43.0850 (apple.four) | 3.722 | 292.46 | 25.265 |
| lm_head.t512.bwd_dx | 512 x 512 x 128256 (capped) | 17.6030 | refused | refused | refused | refused | refused | refused | refused | refused |
| lm_head.t512.bwd_dw | 16032 x 4096 x 512 (capped) | 10.1950 | 37.9770 (apple.four) | 3.725 | 0.8170 | 0.4410 | 38.9920 (apple.four) | 3.825 | 315.75 | 30.971 |

### Three products of one layer, each timed alone and added

The forward product, the input gradient and the weight gradient of one layer. Each line is the sum of three operations measured one at a time. It is not an integrated training step: no optimizer, no activation, no norm and no memory traffic between the products is in it.

| layer | fp32.v1, three products ms | fifteen-bit, three complete products ms | over |
|---|---:|---:|---:|
| qkv.t512 | 9.6710 | 36.0110 | 3.724 |
| mlp_up.t512 | 31.1060 | 118.88 | 3.822 |
| mlp_down.t512 | 31.1700 | 118.75 | 3.810 |
| lm_head.t512 | 39.3740 | refused (lm_head.t512.bwd_dx: k above 65536) | refused |

