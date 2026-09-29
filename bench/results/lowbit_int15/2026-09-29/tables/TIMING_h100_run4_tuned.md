# The fifteen-bit GEMM beside fp32.v1: the complete operation

Profile `mojolearn.identical.gemm.int15i64.v1`. Harness `bench/gemm_int15_price_main.mojo`. Every number is one box's median over the timed calls of one run; `over` is that time over fp32.v1's at the same row in the same run, so above 1 the fifteen-bit operation took longer.

## H100, run 4 (nvc3-0026, 985162978): the tuned plan beside the reference unit plan

Column `nvidia`. The product's plan on this box: the integer matrix unit, four products per k-tile.

### The tuned plan beside the reference unit plan

Reference: one warp one tile, fragments read from device memory, the epilogue in the kernel. Tuned: lane/lowbit-mma-speed's four products with one staging, then the epilogue as a launch of its own. Complete = the operands to planes by the parallel quantizer, then the product (inference: the left operand; training shapes: both).

| row | fp32.v1 ms | reference product ms | over | tuned product ms | over | reference complete ms | over | tuned complete ms | over |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| qkv.t1 (inference) | 0.0700 | 0.2230 | 3.186 | 0.0859 | 1.227 | 0.2000 | 2.857 | 0.1600 | 2.286 |
| qkv.t8 (inference) | 0.1021 | 0.2175 | 2.130 | 0.0857 | 0.839 | 0.2095 | 2.052 | 0.1709 | 1.674 |
| qkv.t512 (both operands converted) | 1.0332 | 1.2135 | 1.175 | 0.4325 | 0.419 | 1.4448 | 1.398 | 0.6727 | 0.651 |
| qkv.t512.bwd_dx (both operands converted) | 1.0219 | 1.2103 | 1.184 | 0.4355 | 0.426 | 1.4733 | 1.442 | 0.7050 | 0.690 |
| qkv.t512.bwd_dw (both operands converted) | 0.8883 | 1.0525 | 1.185 | 0.7676 | 0.864 | 1.1745 | 1.322 | 0.8677 | 0.977 |
| mlp_up.t1 (inference) | 0.1824 | 0.2185 | 1.198 | 0.1402 | 0.769 | 0.2417 | 1.325 | 0.1615 | 0.885 |
| mlp_up.t8 (inference) | 0.2751 | 0.2163 | 0.786 | 0.1544 | 0.561 | 0.2481 | 0.902 | 0.1850 | 0.672 |
| mlp_up.t512 (both operands converted) | 6.1283 | 5.5208 | 0.901 | 1.9218 | 0.314 | 4.7480 | 0.775 | 2.6586 | 0.434 |
| mlp_up.t512.bwd_dx (both operands converted) | 3.6653 | 4.5572 | 1.243 | 1.4173 | 0.387 | 5.4467 | 1.486 | 2.5312 | 0.691 |
| mlp_up.t512.bwd_dw (both operands converted) | 2.9871 | 3.5865 | 1.201 | 2.5068 | 0.839 | 3.7557 | 1.257 | 2.6604 | 0.891 |
| mlp_down.t1 (inference) | 0.1698 | 0.7466 | 4.397 | 0.4427 | 2.607 | 0.7760 | 4.570 | 0.4544 | 2.676 |
| mlp_down.t8 (inference) | 0.2896 | 0.7232 | 2.497 | 0.4527 | 1.563 | 0.7715 | 2.664 | 0.4962 | 1.713 |
| mlp_down.t512 (both operands converted) | 3.7229 | 4.4830 | 1.204 | 1.4113 | 0.379 | 4.9610 | 1.333 | 2.0808 | 0.559 |
| mlp_down.t512.bwd_dx (both operands converted) | 6.1267 | 5.1397 | 0.839 | 1.9391 | 0.316 | 4.9457 | 0.807 | 2.7264 | 0.445 |
| mlp_down.t512.bwd_dw (both operands converted) | 3.1182 | 3.5718 | 1.145 | 2.6940 | 0.864 | 3.7420 | 1.200 | 2.8908 | 0.927 |
| lm_head.t1 (inference) | 1.2152 | 1.0165 | 0.836 | 1.0460 | 0.861 | 1.1680 | 0.961 | 1.0551 | 0.868 |
| lm_head.t8 (inference) | 2.7705 | 1.3741 | 0.496 | 1.3251 | 0.478 | 1.0634 | 0.384 | 1.2362 | 0.446 |
| lm_head.t512 (both operands converted) | 5.5734 | 5.7396 | 1.030 | 2.1246 | 0.381 | 5.2872 | 0.949 | 2.8118 | 0.505 |
| lm_head.t512.bwd_dx (both operands converted) | 3.8350 | refused | refused | refused | refused | refused | refused | refused | refused |
| lm_head.t512.bwd_dw (both operands converted) | 3.3524 | 4.0079 | 1.196 | 2.7675 | 0.826 | 4.1891 | 1.250 | 2.9395 | 0.877 |

#### Inference at the training rows, the tuned plan

| row | fp32.v1 ms | reference complete call ms | over | tuned complete call ms | over |
|---|---:|---:|---:|---:|---:|
| qkv.t512 | 1.0332 | 1.2532 | 1.213 | 0.4988 | 0.483 |
| mlp_up.t512 | 6.1283 | 4.4138 | 0.720 | 1.9077 | 0.311 |
| mlp_down.t512 | 3.7229 | 4.4016 | 1.182 | 1.4793 | 0.397 |
| lm_head.t512 | 5.5734 | 4.8793 | 0.875 | 2.2009 | 0.395 |

#### Three products of one layer, each timed alone and added, the tuned plan

| layer | fp32.v1 ms | reference plan ms | over | tuned plan ms | over |
|---|---:|---:|---:|---:|---:|
| qkv.t512 | 2.9434 | 4.0926 | 1.390 | 2.2454 | 0.763 |
| mlp_up.t512 | 12.7807 | 13.9504 | 1.092 | 7.8502 | 0.614 |
| mlp_down.t512 | 12.9678 | 13.6487 | 1.053 | 7.6980 | 0.594 |
| lm_head.t512 | 12.7608 | refused (k above 65536) | refused | refused (k above 65536) | refused |

### Inference: one call, weights packed once

| row | m x n x k | fp32.v1 ms | product alone ms | over | activations to planes ms | complete call ms | over | complete call, row quantizer ms | over |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| qkv.t1 | 1 x 4096 x 4096 | 0.0700 | 0.2230 (mma) | 3.186 | 0.0260 | 0.2000 (planes) | 2.857 | 0.7235 | 10.336 |
| qkv.t8 | 8 x 4096 x 4096 | 0.1021 | 0.2175 (mma) | 2.130 | 0.0347 | 0.2095 (planes) | 2.052 | 0.8238 | 8.069 |
| qkv.t512 | 512 x 4096 x 4096 | 1.0332 | 1.2135 (mma) | 1.175 | 0.0643 | 1.2532 (planes) | 1.213 | 3.6615 | 3.544 |
| mlp_up.t1 | 1 x 14336 x 4096 | 0.1824 | 0.2185 (mma) | 1.198 | 0.0251 | 0.2417 (planes) | 1.325 | 0.7137 | 3.913 |
| mlp_up.t8 | 8 x 14336 x 4096 | 0.2751 | 0.2163 (mma) | 0.786 | 0.0341 | 0.2481 (planes) | 0.902 | 0.8202 | 2.981 |
| mlp_up.t512 | 512 x 14336 x 4096 | 6.1283 | 5.5208 (mma) | 0.901 | 0.0847 | 4.4025 (codes) | 0.718 | 6.9016 | 1.126 |
| mlp_down.t1 | 1 x 4096 x 14336 | 0.1698 | 0.7466 (mma) | 4.397 | 0.0303 | 0.7760 (planes) | 4.570 | 2.4838 | 14.628 |
| mlp_down.t8 | 8 x 4096 x 14336 | 0.2896 | 0.7232 (mma) | 2.497 | 0.0538 | 0.7715 (planes) | 2.664 | 3.1358 | 10.828 |
| mlp_down.t512 | 512 x 4096 x 14336 | 3.7229 | 4.4830 (mma) | 1.204 | 0.1003 | 4.4016 (planes) | 1.182 | 14.3192 | 3.846 |
| lm_head.t1 | 1 x 128256 x 4096 | 1.2152 | 1.0165 (mma) | 0.836 | 0.0254 | 1.1680 (planes) | 0.961 | 1.6566 | 1.363 |
| lm_head.t8 | 8 x 128256 x 4096 | 2.7705 | 1.3741 (mma) | 0.496 | 0.0421 | 1.0634 (planes) | 0.384 | 1.8097 | 0.653 |
| lm_head.t512 | 512 x 16032 x 4096 (capped) | 5.5734 | 5.7396 (mma) | 1.030 | 0.0864 | 4.8793 (planes) | 0.875 | 7.7088 | 1.383 |

### Training shapes: one product, both operands converted per call

| row | m x n x k | fp32.v1 ms | product alone ms | over | A to planes ms | B to planes ms | complete product ms | over | complete, row quantizer ms | over |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| qkv.t512 | 512 x 4096 x 4096 | 1.0332 | 1.2135 (mma) | 1.175 | 0.0643 | 0.1823 | 1.4448 (planes) | 1.398 | 6.1817 | 5.983 |
| qkv.t512.bwd_dx | 512 x 4096 x 4096 | 1.0219 | 1.2103 (mma) | 1.184 | 0.0640 | 0.2078 | 1.4733 (planes) | 1.442 | 6.3336 | 6.198 |
| qkv.t512.bwd_dw | 4096 x 4096 x 512 | 0.8883 | 1.0525 (mma) | 1.185 | 0.0476 | 0.0473 | 1.1745 (planes) | 1.322 | 1.5238 | 1.715 |
| mlp_up.t512 | 512 x 14336 x 4096 | 6.1283 | 5.5208 (mma) | 0.901 | 0.0847 | 0.7630 | 4.7480 (planes) | 0.775 | 9.3766 | 1.530 |
| mlp_up.t512.bwd_dx | 512 x 4096 x 14336 | 3.6653 | 4.5572 (mma) | 1.243 | 0.1001 | 1.0620 | 5.4467 (planes) | 1.486 | 23.8347 | 6.503 |
| mlp_up.t512.bwd_dw | 14336 x 4096 x 512 | 2.9871 | 3.5865 (mma) | 1.201 | 0.1118 | 0.0491 | 3.7557 (planes) | 1.257 | 4.0995 | 1.372 |
| mlp_down.t512 | 512 x 4096 x 14336 | 3.7229 | 4.4830 (mma) | 1.204 | 0.1003 | 0.6119 | 4.9610 (planes) | 1.333 | 24.2494 | 6.514 |
| mlp_down.t512.bwd_dx | 512 x 14336 x 4096 | 6.1267 | 5.1397 (mma) | 0.839 | 0.0866 | 0.7253 | 4.9457 (planes) | 0.807 | 10.1244 | 1.653 |
| mlp_down.t512.bwd_dw | 4096 x 14336 x 512 | 3.1182 | 3.5718 (mma) | 1.145 | 0.0480 | 0.1115 | 3.7420 (planes) | 1.200 | 4.0772 | 1.308 |
| lm_head.t512 | 512 x 16032 x 4096 (capped) | 5.5734 | 5.7396 (mma) | 1.030 | 0.0864 | 0.9181 | 5.2872 (planes) | 0.949 | 9.9624 | 1.787 |
| lm_head.t512.bwd_dx | 512 x 512 x 128256 (capped) | 3.8350 | refused | refused | refused | refused | refused | refused | refused | refused |
| lm_head.t512.bwd_dw | 16032 x 4096 x 512 (capped) | 3.3524 | 4.0079 (mma) | 1.196 | 0.1232 | 0.0487 | 4.1891 (planes) | 1.250 | 4.5495 | 1.357 |

### Three products of one layer, each timed alone and added

The forward product, the input gradient and the weight gradient of one layer. Each line is the sum of three operations measured one at a time. It is not an integrated training step: no optimizer, no activation, no norm and no memory traffic between the products is in it.

| layer | fp32.v1, three products ms | fifteen-bit, three complete products ms | over |
|---|---:|---:|---:|
| qkv.t512 | 2.9434 | 4.0926 | 1.390 |
| mlp_up.t512 | 12.7807 | 13.9504 | 1.092 |
| mlp_down.t512 | 12.9678 | 13.6487 | 1.053 |
| lm_head.t512 | 12.7608 | refused (lm_head.t512.bwd_dx: k above 65536) | refused |

