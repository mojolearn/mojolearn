# The fifteen-bit GEMM beside fp32.v1: the complete operation

Profile `mojolearn.identical.gemm.int15i64.v1`. Harness `bench/gemm_int15_price_main.mojo`. Every number is one box's median over the timed calls of one run; `over` is that time over fp32.v1's at the same row in the same run, so above 1 the fifteen-bit operation took longer.

## H100, run 6 (nvc3-0034, 71db5bb26)

Column `nvidia`. The product's plan on this box: the integer matrix unit, four products per k-tile.

### The tuned plan beside the reference unit plan

Reference: one warp one tile, fragments read from device memory, the epilogue in the kernel. Tuned: lane/lowbit-mma-speed's four products with one staging, then the epilogue as a launch of its own. Complete = the operands to planes by the parallel quantizer, then the product (inference: the left operand; training shapes: both).

| row | fp32.v1 ms | reference product ms | over | tuned product ms | over | reference complete ms | over | tuned complete ms | over |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| qkv.t1 (inference) | 0.0676 | 0.2217 | 3.280 | 0.0888 | 1.314 | 0.2045 | 3.025 | 0.1568 | 2.320 |
| qkv.t8 (inference) | 0.1010 | 0.2154 | 2.133 | 0.0908 | 0.899 | 0.2073 | 2.052 | 0.1673 | 1.656 |
| qkv.t512 (both operands converted) | 1.0314 | 1.2124 | 1.175 | 0.3834 | 0.372 | 1.4425 | 1.399 | 0.6261 | 0.607 |
| qkv.t512.bwd_dx (both operands converted) | 1.0139 | 1.1970 | 1.181 | 0.3860 | 0.381 | 1.4688 | 1.449 | 0.6554 | 0.646 |
| qkv.t512.bwd_dw (both operands converted) | 0.8809 | 1.0498 | 1.192 | 0.4104 | 0.466 | 1.1724 | 1.331 | 0.5079 | 0.577 |
| mlp_up.t1 (inference) | 0.1791 | 0.2180 | 1.217 | 0.1368 | 0.764 | 0.2414 | 1.348 | 0.1583 | 0.884 |
| mlp_up.t8 (inference) | 0.2707 | 0.2161 | 0.798 | 0.1508 | 0.557 | 0.2482 | 0.917 | 0.1803 | 0.666 |
| mlp_up.t512 (both operands converted) | 3.4303 | 3.6865 | 1.075 | 1.4973 | 0.436 | 4.3493 | 1.268 | 2.1783 | 0.635 |
| mlp_up.t512.bwd_dx (both operands converted) | 3.3430 | 4.2088 | 1.259 | 1.3243 | 0.396 | 5.3781 | 1.609 | 2.5020 | 0.748 |
| mlp_up.t512.bwd_dw (both operands converted) | 2.9816 | 3.6161 | 1.213 | 1.3536 | 0.454 | 3.7850 | 1.269 | 1.5100 | 0.506 |
| mlp_down.t1 (inference) | 0.1678 | 0.7509 | 4.475 | 0.4408 | 2.627 | 0.7794 | 4.645 | 0.4489 | 2.675 |
| mlp_down.t8 (inference) | 0.2885 | 0.7261 | 2.517 | 0.4493 | 1.557 | 0.7748 | 2.686 | 0.4929 | 1.708 |
| mlp_down.t512 (both operands converted) | 3.3921 | 4.2407 | 1.250 | 1.3239 | 0.390 | 4.9168 | 1.449 | 2.0050 | 0.591 |
| mlp_down.t512.bwd_dx (both operands converted) | 3.3907 | 3.7197 | 1.097 | 1.5017 | 0.443 | 4.4012 | 1.298 | 2.1774 | 0.642 |
| mlp_down.t512.bwd_dw (both operands converted) | 3.1062 | 3.5976 | 1.158 | 1.4430 | 0.465 | 3.7658 | 1.212 | 1.6126 | 0.519 |
| lm_head.t1 (inference) | 1.2439 | 1.0295 | 0.828 | 1.0904 | 0.877 | 1.0582 | 0.851 | 1.1158 | 0.897 |
| lm_head.t8 (inference) | 2.3611 | 1.0486 | 0.444 | 1.2068 | 0.511 | 1.0739 | 0.455 | 1.2392 | 0.525 |
| lm_head.t512 (both operands converted) | 3.7860 | 4.1996 | 1.109 | 1.7053 | 0.450 | 4.9690 | 1.312 | 2.4406 | 0.645 |
| lm_head.t512.bwd_dx (both operands converted) | 3.8247 | refused | refused | refused | refused | refused | refused | refused | refused |
| lm_head.t512.bwd_dw (both operands converted) | 3.3422 | 4.0369 | 1.208 | 1.4902 | 0.446 | 4.2134 | 1.261 | 1.6675 | 0.499 |

#### Inference at the training rows, the tuned plan

| row | fp32.v1 ms | reference complete call ms | over | tuned complete call ms | over |
|---|---:|---:|---:|---:|---:|
| qkv.t512 | 1.0314 | 1.2725 | 1.234 | 0.4500 | 0.436 |
| mlp_up.t512 | 3.4303 | 3.7743 | 1.100 | 1.5533 | 0.453 |
| mlp_down.t512 | 3.3921 | 4.2908 | 1.265 | 1.4140 | 0.417 |
| lm_head.t512 | 3.7860 | 4.2812 | 1.131 | 1.7656 | 0.466 |

#### Three products of one layer, each timed alone and added, the tuned plan

| layer | fp32.v1 ms | reference plan ms | over | tuned plan ms | over |
|---|---:|---:|---:|---:|---:|
| qkv.t512 | 2.9262 | 4.0837 | 1.396 | 1.7894 | 0.612 |
| mlp_up.t512 | 9.7549 | 13.5124 | 1.385 | 6.1903 | 0.635 |
| mlp_down.t512 | 9.8890 | 13.0838 | 1.323 | 5.7950 | 0.586 |
| lm_head.t512 | 10.9529 | refused (k above 65536) | refused | refused (k above 65536) | refused |

### The epilogue fold: fused beside two-launch, one lever

Fused: the sums kernel calls the epilogue at its store, one launch, no sums in device memory. Two-launch: the sums stored (12 bytes per cell), then the epilogue launch.
`fused/two` is the fused time over the two-launch time in this run; below 1 the fold took less time.

| row | fp32.v1 ms | product two-launch ms | product fused ms | fused/two | complete two-launch ms | over fp32.v1 | complete fused ms | over fp32.v1 | fused/two |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| qkv.t1 (inference) | 0.0676 | 0.0904 | 0.0888 | 0.982 | 0.1116 | 1.651 | 0.1568 | 2.320 | 1.405 |
| qkv.t8 (inference) | 0.1010 | 0.0863 | 0.0908 | 1.052 | 0.1273 | 1.260 | 0.1673 | 1.656 | 1.314 |
| qkv.t512 (inference) | 1.0314 | 0.4528 | 0.3834 | 0.847 | 0.4995 | 0.484 | 0.4500 | 0.436 | 0.901 |
| qkv.t512.bwd_dx (both operands) | 1.0139 | 0.4525 | 0.3860 | 0.853 | 0.7049 | 0.695 | 0.6554 | 0.646 | 0.930 |
| qkv.t512.bwd_dw (both operands) | 0.8809 | 0.7736 | 0.4104 | 0.531 | 0.8973 | 1.019 | 0.5079 | 0.577 | 0.566 |
| mlp_up.t1 (inference) | 0.1791 | 0.1395 | 0.1368 | 0.981 | 0.1542 | 0.861 | 0.1583 | 0.884 | 1.027 |
| mlp_up.t8 (inference) | 0.2707 | 0.1534 | 0.1508 | 0.983 | 0.1795 | 0.663 | 0.1803 | 0.666 | 1.004 |
| mlp_up.t512 (inference) | 3.4303 | 1.6595 | 1.4973 | 0.902 | 1.7186 | 0.501 | 1.5533 | 0.453 | 0.904 |
| mlp_up.t512.bwd_dx (both operands) | 3.3430 | 1.3810 | 1.3243 | 0.959 | 2.5558 | 0.765 | 2.5020 | 0.748 | 0.979 |
| mlp_up.t512.bwd_dw (both operands) | 2.9816 | 2.5232 | 1.3536 | 0.536 | 2.7042 | 0.907 | 1.5100 | 0.506 | 0.558 |
| mlp_down.t1 (inference) | 0.1678 | 0.4423 | 0.4408 | 0.997 | 0.4434 | 2.642 | 0.4489 | 2.675 | 1.012 |
| mlp_down.t8 (inference) | 0.2885 | 0.4568 | 0.4493 | 0.984 | 0.4929 | 1.708 | 0.4929 | 1.708 | 1.000 |
| mlp_down.t512 (inference) | 3.3921 | 1.3782 | 1.3239 | 0.961 | 1.4637 | 0.432 | 1.4140 | 0.417 | 0.966 |
| mlp_down.t512.bwd_dx (both operands) | 3.3907 | 1.6459 | 1.5017 | 0.912 | 2.3192 | 0.684 | 2.1774 | 0.642 | 0.939 |
| mlp_down.t512.bwd_dw (both operands) | 3.1062 | 2.7267 | 1.4430 | 0.529 | 2.9039 | 0.935 | 1.6126 | 0.519 | 0.555 |
| lm_head.t1 (inference) | 1.2439 | 1.0944 | 1.0904 | 0.996 | 1.1142 | 0.896 | 1.1158 | 0.897 | 1.001 |
| lm_head.t8 (inference) | 2.3611 | 1.2134 | 1.2068 | 0.995 | 1.2337 | 0.523 | 1.2392 | 0.525 | 1.004 |
| lm_head.t512 (inference) | 3.7860 | 1.8320 | 1.7053 | 0.931 | 1.8941 | 0.500 | 1.7656 | 0.466 | 0.932 |
| lm_head.t512.bwd_dx | 3.8247 | refused | refused | | refused | | refused | | |
| lm_head.t512.bwd_dw (both operands) | 3.3422 | 2.7984 | 1.4902 | 0.533 | 2.9710 | 0.889 | 1.6675 | 0.499 | 0.561 |

### Inference: one call, weights packed once

| row | m x n x k | fp32.v1 ms | product alone ms | over | activations to planes ms | complete call ms | over | complete call, row quantizer ms | over |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| qkv.t1 | 1 x 4096 x 4096 | 0.0676 | 0.2217 (mma) | 3.280 | 0.0247 | 0.2045 (planes) | 3.025 | 0.7217 | 10.676 |
| qkv.t8 | 8 x 4096 x 4096 | 0.1010 | 0.2154 (mma) | 2.133 | 0.0336 | 0.2073 (planes) | 2.052 | 0.8223 | 8.142 |
| qkv.t512 | 512 x 4096 x 4096 | 1.0314 | 1.2124 (mma) | 1.175 | 0.0636 | 1.2725 (planes) | 1.234 | 3.6817 | 3.570 |
| mlp_up.t1 | 1 x 14336 x 4096 | 0.1791 | 0.2180 (mma) | 1.217 | 0.0247 | 0.2414 (planes) | 1.348 | 0.7177 | 4.007 |
| mlp_up.t8 | 8 x 14336 x 4096 | 0.2707 | 0.2161 (mma) | 0.798 | 0.0335 | 0.2482 (planes) | 0.917 | 0.8218 | 3.036 |
| mlp_up.t512 | 512 x 14336 x 4096 | 3.4303 | 3.6865 (mma) | 1.075 | 0.0650 | 3.7743 (planes) | 1.100 | 9.3272 | 2.719 |
| mlp_down.t1 | 1 x 4096 x 14336 | 0.1678 | 0.7509 (mma) | 4.475 | 0.0297 | 0.7794 (planes) | 4.645 | 2.4863 | 14.817 |
| mlp_down.t8 | 8 x 4096 x 14336 | 0.2885 | 0.7261 (mma) | 2.517 | 0.0532 | 0.7748 (planes) | 2.686 | 3.1361 | 10.870 |
| mlp_down.t512 | 512 x 4096 x 14336 | 3.3921 | 4.2407 (mma) | 1.250 | 0.0947 | 4.2908 (planes) | 1.265 | 14.6859 | 4.329 |
| lm_head.t1 | 1 x 128256 x 4096 | 1.2439 | 1.0295 (mma) | 0.828 | 0.0253 | 1.0582 (planes) | 0.851 | 1.4660 | 1.179 |
| lm_head.t8 | 8 x 128256 x 4096 | 2.3611 | 1.0486 (mma) | 0.444 | 0.0359 | 1.0739 (planes) | 0.455 | 1.7384 | 0.736 |
| lm_head.t512 | 512 x 16032 x 4096 (capped) | 3.7860 | 4.1996 (mma) | 1.109 | 0.0643 | 4.2812 (planes) | 1.131 | 9.7882 | 2.585 |

### Training shapes: one product, both operands converted per call

| row | m x n x k | fp32.v1 ms | product alone ms | over | A to planes ms | B to planes ms | complete product ms | over | complete, row quantizer ms | over |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| qkv.t512 | 512 x 4096 x 4096 | 1.0314 | 1.2124 (mma) | 1.175 | 0.0636 | 0.1825 | 1.4425 (planes) | 1.399 | 6.1933 | 6.005 |
| qkv.t512.bwd_dx | 512 x 4096 x 4096 | 1.0139 | 1.1970 (mma) | 1.181 | 0.0636 | 0.2073 | 1.4688 (planes) | 1.449 | 6.3433 | 6.256 |
| qkv.t512.bwd_dw | 4096 x 4096 x 512 | 0.8809 | 1.0498 (mma) | 1.192 | 0.0466 | 0.0464 | 1.1724 (planes) | 1.331 | 1.4538 | 1.650 |
| mlp_up.t512 | 512 x 14336 x 4096 | 3.4303 | 3.6865 (mma) | 1.075 | 0.0650 | 0.6085 | 4.3493 (planes) | 1.268 | 11.3189 | 3.300 |
| mlp_up.t512.bwd_dx | 512 x 4096 x 14336 | 3.3430 | 4.2088 (mma) | 1.259 | 0.0951 | 1.0865 | 5.3781 (planes) | 1.609 | 24.2988 | 7.269 |
| mlp_up.t512.bwd_dw | 14336 x 4096 x 512 | 2.9816 | 3.6161 (mma) | 1.213 | 0.1112 | 0.0473 | 3.7850 (planes) | 1.269 | 4.1067 | 1.377 |
| mlp_down.t512 | 512 x 4096 x 14336 | 3.3921 | 4.2407 (mma) | 1.250 | 0.0947 | 0.5985 | 4.9168 (planes) | 1.449 | 24.8442 | 7.324 |
| mlp_down.t512.bwd_dx | 512 x 14336 x 4096 | 3.3907 | 3.7197 (mma) | 1.097 | 0.0657 | 0.6179 | 4.4012 (planes) | 1.298 | 13.0924 | 3.861 |
| mlp_down.t512.bwd_dw | 4096 x 14336 x 512 | 3.1062 | 3.5976 (mma) | 1.158 | 0.0467 | 0.1111 | 3.7658 (planes) | 1.212 | 4.0925 | 1.318 |
| lm_head.t512 | 512 x 16032 x 4096 (capped) | 3.7860 | 4.1996 (mma) | 1.109 | 0.0643 | 0.6782 | 4.9690 (planes) | 1.312 | 12.2336 | 3.231 |
| lm_head.t512.bwd_dx | 512 x 512 x 128256 (capped) | 3.8247 | refused | refused | refused | refused | refused | refused | refused | refused |
| lm_head.t512.bwd_dw | 16032 x 4096 x 512 (capped) | 3.3422 | 4.0369 (mma) | 1.208 | 0.1223 | 0.0476 | 4.2134 (planes) | 1.261 | 4.5634 | 1.365 |

### Three products of one layer, each timed alone and added

The forward product, the input gradient and the weight gradient of one layer. Each line is the sum of three operations measured one at a time. It is not an integrated training step: no optimizer, no activation, no norm and no memory traffic between the products is in it.

| layer | fp32.v1, three products ms | fifteen-bit, three complete products ms | over |
|---|---:|---:|---:|
| qkv.t512 | 2.9262 | 4.0837 | 1.396 |
| mlp_up.t512 | 9.7549 | 13.5124 | 1.385 |
| mlp_down.t512 | 9.8890 | 13.0838 | 1.323 |
| lm_head.t512 | 10.9529 | refused (lm_head.t512.bwd_dx: k above 65536) | refused |

