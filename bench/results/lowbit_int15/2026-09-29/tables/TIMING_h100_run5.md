# The fifteen-bit GEMM beside fp32.v1: the complete operation

Profile `mojolearn.identical.gemm.int15i64.v1`. Harness `bench/gemm_int15_price_main.mojo`. Every number is one box's median over the timed calls of one run; `over` is that time over fp32.v1's at the same row in the same run, so above 1 the fifteen-bit operation took longer.

## H100, run 5 (nvc3-0028, 99ae7bb66): the run of record, alternation in two blocks

Column `nvidia`. The product's plan on this box: the integer matrix unit, four products per k-tile.

### The tuned plan beside the reference unit plan

Reference: one warp one tile, fragments read from device memory, the epilogue in the kernel. Tuned: lane/lowbit-mma-speed's four products with one staging, then the epilogue as a launch of its own. Complete = the operands to planes by the parallel quantizer, then the product (inference: the left operand; training shapes: both).

| row | fp32.v1 ms | reference product ms | over | tuned product ms | over | reference complete ms | over | tuned complete ms | over |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| qkv.t1 (inference) | 0.0715 | 0.2257 | 3.157 | 0.0903 | 1.263 | 0.2138 | 2.990 | 0.1602 | 2.241 |
| qkv.t8 (inference) | 0.1020 | 0.2175 | 2.132 | 0.0961 | 0.942 | 0.2125 | 2.083 | 0.1722 | 1.688 |
| qkv.t512 (both operands converted) | 1.0324 | 1.2167 | 1.179 | 0.4342 | 0.421 | 1.4429 | 1.398 | 0.6761 | 0.655 |
| qkv.t512.bwd_dx (both operands converted) | 1.0173 | 1.1943 | 1.174 | 0.4335 | 0.426 | 1.4728 | 1.448 | 0.7035 | 0.692 |
| qkv.t512.bwd_dw (both operands converted) | 0.8823 | 1.0537 | 1.194 | 0.7670 | 0.869 | 1.1751 | 1.332 | 0.8629 | 0.978 |
| mlp_up.t1 (inference) | 0.1802 | 0.2189 | 1.215 | 0.1403 | 0.779 | 0.2445 | 1.357 | 0.1613 | 0.895 |
| mlp_up.t8 (inference) | 0.2725 | 0.2176 | 0.799 | 0.1558 | 0.572 | 0.2502 | 0.918 | 0.1863 | 0.684 |
| mlp_up.t512 (both operands converted) | 3.4145 | 3.6611 | 1.072 | 1.6316 | 0.478 | 4.3579 | 1.276 | 2.2924 | 0.671 |
| mlp_up.t512.bwd_dx (both operands converted) | 3.3626 | 4.2725 | 1.271 | 1.3756 | 0.409 | 5.3732 | 1.598 | 2.4937 | 0.742 |
| mlp_up.t512.bwd_dw (both operands converted) | 2.9800 | 3.6175 | 1.214 | 2.4828 | 0.833 | 3.7861 | 1.271 | 2.6760 | 0.898 |
| mlp_down.t1 (inference) | 0.1693 | 0.7503 | 4.432 | 0.4417 | 2.609 | 0.7793 | 4.603 | 0.4552 | 2.689 |
| mlp_down.t8 (inference) | 0.2883 | 0.7275 | 2.523 | 0.4541 | 1.575 | 0.7744 | 2.686 | 0.4965 | 1.722 |
| mlp_down.t512 (both operands converted) | 3.4251 | 4.2410 | 1.238 | 1.3772 | 0.402 | 4.9243 | 1.438 | 2.0551 | 0.600 |
| mlp_down.t512.bwd_dx (both operands converted) | 3.4121 | 3.7244 | 1.092 | 1.6424 | 0.481 | 4.3751 | 1.282 | 2.3174 | 0.679 |
| mlp_down.t512.bwd_dw (both operands converted) | 3.1127 | 3.6001 | 1.157 | 2.7006 | 0.868 | 3.7693 | 1.211 | 2.8891 | 0.928 |
| lm_head.t1 (inference) | 1.2450 | 1.0345 | 0.831 | 1.0921 | 0.877 | 1.0589 | 0.851 | 1.1134 | 0.894 |
| lm_head.t8 (inference) | 2.3660 | 1.0455 | 0.442 | 1.2161 | 0.514 | 1.0848 | 0.458 | 1.2500 | 0.528 |
| lm_head.t512 (both operands converted) | 4.0654 | 4.4525 | 1.095 | 1.8781 | 0.462 | 5.3217 | 1.309 | 2.6789 | 0.659 |
| lm_head.t512.bwd_dx (both operands converted) | 3.8219 | refused | refused | refused | refused | refused | refused | refused | refused |
| lm_head.t512.bwd_dw (both operands converted) | 3.3384 | 4.0378 | 1.210 | 2.7607 | 0.827 | 4.2159 | 1.263 | 2.9528 | 0.884 |

#### Inference at the training rows, the tuned plan

| row | fp32.v1 ms | reference complete call ms | over | tuned complete call ms | over |
|---|---:|---:|---:|---:|---:|
| qkv.t512 | 1.0324 | 1.2716 | 1.232 | 0.4993 | 0.484 |
| mlp_up.t512 | 3.4145 | 3.7483 | 1.098 | 1.6924 | 0.496 |
| mlp_down.t512 | 3.4251 | 4.3383 | 1.267 | 1.4638 | 0.427 |
| lm_head.t512 | 4.0654 | 4.5903 | 1.129 | 1.9440 | 0.478 |

#### Three products of one layer, each timed alone and added, the tuned plan

| layer | fp32.v1 ms | reference plan ms | over | tuned plan ms | over |
|---|---:|---:|---:|---:|---:|
| qkv.t512 | 2.9320 | 4.0908 | 1.395 | 2.2425 | 0.765 |
| mlp_up.t512 | 9.7571 | 13.5172 | 1.385 | 7.4621 | 0.765 |
| mlp_down.t512 | 9.9499 | 13.0687 | 1.313 | 7.2616 | 0.730 |
| lm_head.t512 | 11.2257 | refused (k above 65536) | refused | refused (k above 65536) | refused |

### Inference: one call, weights packed once

| row | m x n x k | fp32.v1 ms | product alone ms | over | activations to planes ms | complete call ms | over | complete call, row quantizer ms | over |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| qkv.t1 | 1 x 4096 x 4096 | 0.0715 | 0.2257 (mma) | 3.157 | 0.0268 | 0.2138 (planes) | 2.990 | 0.7220 | 10.098 |
| qkv.t8 | 8 x 4096 x 4096 | 0.1020 | 0.2175 (mma) | 2.132 | 0.0347 | 0.2125 (planes) | 2.083 | 0.8248 | 8.086 |
| qkv.t512 | 512 x 4096 x 4096 | 1.0324 | 1.2167 (mma) | 1.179 | 0.0645 | 1.2716 (planes) | 1.232 | 3.6672 | 3.552 |
| mlp_up.t1 | 1 x 14336 x 4096 | 0.1802 | 0.2189 (mma) | 1.215 | 0.0261 | 0.2445 (planes) | 1.357 | 0.7188 | 3.989 |
| mlp_up.t8 | 8 x 14336 x 4096 | 0.2725 | 0.2176 (mma) | 0.799 | 0.0345 | 0.2502 (planes) | 0.918 | 0.8310 | 3.050 |
| mlp_up.t512 | 512 x 14336 x 4096 | 3.4145 | 3.6611 (mma) | 1.072 | 0.0659 | 3.7483 (planes) | 1.098 | 9.2665 | 2.714 |
| mlp_down.t1 | 1 x 4096 x 14336 | 0.1693 | 0.7503 (mma) | 4.432 | 0.0301 | 0.7793 (planes) | 4.603 | 2.4972 | 14.750 |
| mlp_down.t8 | 8 x 4096 x 14336 | 0.2883 | 0.7275 (mma) | 2.523 | 0.0538 | 0.7744 (planes) | 2.686 | 3.1511 | 10.930 |
| mlp_down.t512 | 512 x 4096 x 14336 | 3.4251 | 4.2410 (mma) | 1.238 | 0.0957 | 4.3383 (planes) | 1.267 | 14.5249 | 4.241 |
| lm_head.t1 | 1 x 128256 x 4096 | 1.2450 | 1.0345 (mma) | 0.831 | 0.0261 | 1.0589 (planes) | 0.851 | 1.4682 | 1.179 |
| lm_head.t8 | 8 x 128256 x 4096 | 2.3660 | 1.0455 (mma) | 0.442 | 0.0376 | 1.0848 (planes) | 0.458 | 2.0255 | 0.856 |
| lm_head.t512 | 512 x 16032 x 4096 (capped) | 4.0654 | 4.4525 (mma) | 1.095 | 0.0696 | 4.5903 (planes) | 1.129 | 9.3458 | 2.299 |

### Training shapes: one product, both operands converted per call

| row | m x n x k | fp32.v1 ms | product alone ms | over | A to planes ms | B to planes ms | complete product ms | over | complete, row quantizer ms | over |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| qkv.t512 | 512 x 4096 x 4096 | 1.0324 | 1.2167 (mma) | 1.179 | 0.0645 | 0.1831 | 1.4429 (planes) | 1.398 | 6.2002 | 6.006 |
| qkv.t512.bwd_dx | 512 x 4096 x 4096 | 1.0173 | 1.1943 (mma) | 1.174 | 0.0640 | 0.2075 | 1.4728 (planes) | 1.448 | 6.3416 | 6.234 |
| qkv.t512.bwd_dw | 4096 x 4096 x 512 | 0.8823 | 1.0537 (mma) | 1.194 | 0.0470 | 0.0468 | 1.1751 (planes) | 1.332 | 1.4537 | 1.648 |
| mlp_up.t512 | 512 x 14336 x 4096 | 3.4145 | 3.6611 (mma) | 1.072 | 0.0659 | 0.6073 | 4.3579 (planes) | 1.276 | 11.1922 | 3.278 |
| mlp_up.t512.bwd_dx | 512 x 4096 x 14336 | 3.3626 | 4.2725 (mma) | 1.271 | 0.0955 | 1.0404 | 5.3732 (planes) | 1.598 | 23.8590 | 7.095 |
| mlp_up.t512.bwd_dw | 14336 x 4096 x 512 | 2.9800 | 3.6175 (mma) | 1.214 | 0.1123 | 0.0482 | 3.7861 (planes) | 1.271 | 4.1066 | 1.378 |
| mlp_down.t512 | 512 x 4096 x 14336 | 3.4251 | 4.2410 (mma) | 1.238 | 0.0957 | 0.6002 | 4.9243 (planes) | 1.438 | 24.6943 | 7.210 |
| mlp_down.t512.bwd_dx | 512 x 14336 x 4096 | 3.4121 | 3.7244 (mma) | 1.092 | 0.0661 | 0.6204 | 4.3751 (planes) | 1.282 | 12.1110 | 3.549 |
| mlp_down.t512.bwd_dw | 4096 x 14336 x 512 | 3.1127 | 3.6001 (mma) | 1.157 | 0.0473 | 0.1127 | 3.7693 (planes) | 1.211 | 4.0953 | 1.316 |
| lm_head.t512 | 512 x 16032 x 4096 (capped) | 4.0654 | 4.4525 (mma) | 1.095 | 0.0696 | 0.7370 | 5.3217 (planes) | 1.309 | 11.7859 | 2.899 |
| lm_head.t512.bwd_dx | 512 x 512 x 128256 (capped) | 3.8219 | refused | refused | refused | refused | refused | refused | refused | refused |
| lm_head.t512.bwd_dw | 16032 x 4096 x 512 (capped) | 3.3384 | 4.0378 (mma) | 1.210 | 0.1237 | 0.0481 | 4.2159 (planes) | 1.263 | 4.5583 | 1.365 |

### Three products of one layer, each timed alone and added

The forward product, the input gradient and the weight gradient of one layer. Each line is the sum of three operations measured one at a time. It is not an integrated training step: no optimizer, no activation, no norm and no memory traffic between the products is in it.

| layer | fp32.v1, three products ms | fifteen-bit, three complete products ms | over |
|---|---:|---:|---:|
| qkv.t512 | 2.9320 | 4.0908 | 1.395 |
| mlp_up.t512 | 9.7571 | 13.5172 | 1.385 |
| mlp_down.t512 | 9.9499 | 13.0687 | 1.313 |
| lm_head.t512 | 11.2257 | refused (lm_head.t512.bwd_dx: k above 65536) | refused |

