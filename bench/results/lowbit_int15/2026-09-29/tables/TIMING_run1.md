# The fifteen-bit GEMM beside fp32.v1: the complete operation

Profile `mojolearn.identical.gemm.int15i64.v1`. Harness `bench/gemm_int15_price_main.mojo`. Every number is one box's median over the timed calls of one run; `over` is that time over fp32.v1's at the same row in the same run, so above 1 the fifteen-bit operation took longer.

## H100, run 1 (nvc3-0017, 77598eeb7; fragment loads with no alignment stated)

Column `nvidia`. The product's plan on this box: the integer matrix unit, four products per k-tile.

### Inference: one call, weights packed once

| row | m x n x k | fp32.v1 ms | product alone ms | over | activations to planes ms | complete call ms | over | complete call, row quantizer ms | over |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| qkv.t1 | 1 x 4096 x 4096 | 0.0716 | 0.3343 | 4.669 | 0.0266 | 0.4147 | 5.792 | 0.8884 | 12.408 |
| qkv.t8 | 8 x 4096 x 4096 | 0.1032 | 0.3343 | 3.239 | 0.0341 | 0.4478 | 4.339 | 1.0194 | 9.878 |
| qkv.t512 | 512 x 4096 x 4096 | 1.0326 | 2.8384 | 2.749 | 0.0645 | 2.8937 | 2.802 | 5.3061 | 5.139 |
| mlp_up.t1 | 1 x 14336 x 4096 | 0.1813 | 0.3752 | 2.069 | 0.0252 | 0.4004 | 2.208 | 0.8693 | 4.795 |
| mlp_up.t8 | 8 x 14336 x 4096 | 0.2756 | 0.4139 | 1.502 | 0.0350 | 0.4445 | 1.613 | 1.0149 | 3.683 |
| mlp_up.t512 | 512 x 14336 x 4096 | 3.4095 | 13.9276 | 4.085 | 0.0865 | 11.6497 | 3.417 | 13.2465 | 3.885 |
| mlp_down.t1 | 1 x 4096 x 14336 | 0.1721 | 1.3292 | 7.723 | 0.0310 | 1.3667 | 7.941 | 3.0695 | 17.836 |
| mlp_down.t8 | 8 x 4096 x 14336 | 0.2933 | 1.3875 | 4.731 | 0.0539 | 1.4836 | 5.058 | 3.8323 | 13.066 |
| mlp_down.t512 | 512 x 4096 x 14336 | 3.4058 | 11.1343 | 3.269 | 0.1011 | 10.2577 | 3.012 | 20.0565 | 5.889 |
| lm_head.t1 | 1 x 128256 x 4096 | 1.2043 | 1.7879 | 1.485 | 0.0255 | 1.8135 | 1.506 | 2.2825 | 1.895 |
| lm_head.t8 | 8 x 128256 x 4096 | 2.1881 | 2.8407 | 1.298 | 0.0424 | 2.5127 | 1.148 | 3.1532 | 1.441 |
| lm_head.t512 | 512 x 16032 x 4096 (capped) | 3.7968 | 15.9507 | 4.201 | 0.0879 | 12.2982 | 3.239 | 14.2175 | 3.745 |

### Training: one product of a step, both operands converted per call

| row | m x n x k | fp32.v1 ms | product alone ms | over | A to planes ms | B to planes ms | complete product ms | over | complete, row quantizer ms | over |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| qkv.t512 | 512 x 4096 x 4096 | 1.0326 | 2.8384 | 2.749 | 0.0645 | 0.1837 | 3.0663 | 2.969 | 7.8209 | 7.574 |
| qkv.t512.bwd_dx | 512 x 4096 x 4096 | 1.0184 | 2.8429 | 2.792 | 0.0651 | 0.2048 | 3.0899 | 3.034 | 7.9717 | 7.828 |
| qkv.t512.bwd_dw | 4096 x 4096 x 512 | 0.8872 | 2.6459 | 2.982 | 0.0489 | 0.0488 | 2.7483 | 3.098 | 3.1197 | 3.516 |
| mlp_up.t512 | 512 x 14336 x 4096 | 3.4095 | 13.9276 | 4.085 | 0.0865 | 0.7896 | 10.9194 | 3.203 | 15.2663 | 4.478 |
| mlp_up.t512.bwd_dx | 512 x 4096 x 14336 | 3.3542 | 10.7703 | 3.211 | 0.1002 | 0.9327 | 11.0779 | 3.303 | 29.2794 | 8.729 |
| mlp_up.t512.bwd_dw | 14336 x 4096 x 512 | 2.9681 | 9.1490 | 3.082 | 0.1126 | 0.0491 | 9.3034 | 3.134 | 9.6608 | 3.255 |
| mlp_down.t512 | 512 x 4096 x 14336 | 3.4058 | 11.1343 | 3.269 | 0.1011 | 0.6282 | 10.6300 | 3.121 | 29.8516 | 8.765 |
| mlp_down.t512.bwd_dx | 512 x 14336 x 4096 | 3.4260 | 14.9042 | 4.350 | 0.0878 | 0.6937 | 11.7965 | 3.443 | 16.0417 | 4.682 |
| mlp_down.t512.bwd_dw | 4096 x 14336 x 512 | 3.1161 | 9.1464 | 2.935 | 0.0492 | 0.1122 | 9.3155 | 2.989 | 9.6529 | 3.098 |
| lm_head.t512 | 512 x 16032 x 4096 (capped) | 3.7968 | 15.9507 | 4.201 | 0.0879 | 0.7962 | 11.9581 | 3.150 | 16.3769 | 4.313 |
| lm_head.t512.bwd_dx | 512 x 512 x 128256 (capped) | 3.7888 | refused | refused | refused | refused | refused | refused | refused | refused |
| lm_head.t512.bwd_dw | 16032 x 4096 x 512 (capped) | 3.3558 | 10.1303 | 3.019 | 0.1225 | 0.0483 | 10.3101 | 3.072 | 10.6783 | 3.182 |

### Training: a layer's step, the forward product and the two backward products ADDED

Each line is the sum of three operations measured one at a time, not one measured operation.

| layer | fp32.v1, three products ms | fifteen-bit, three complete products ms | over |
|---|---:|---:|---:|
| qkv.t512 | 2.9382 | 8.9045 | 3.031 |
| mlp_up.t512 | 9.7318 | 31.3007 | 3.216 |
| mlp_down.t512 | 9.9479 | 31.7420 | 3.191 |
| lm_head.t512 | 10.9414 | refused (lm_head.t512.bwd_dx: k above 65536) | refused |

## MI325X, run 1 (1790653523354, 77598eeb7; fragment loads with no alignment stated)

Column `amd`. The product's plan on this box: the integer matrix unit, four products per k-tile.

### Inference: one call, weights packed once

| row | m x n x k | fp32.v1 ms | product alone ms | over | activations to planes ms | complete call ms | over | complete call, row quantizer ms | over |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| qkv.t1 | 1 x 4096 x 4096 | 0.0644 | 0.0682 | 1.059 | 0.0452 | 0.1038 | 1.612 | 1.3732 | 21.323 |
| qkv.t8 | 8 x 4096 x 4096 | 0.0746 | 0.0723 | 0.969 | 0.0627 | 0.1248 | 1.673 | 1.4735 | 19.752 |
| qkv.t512 | 512 x 4096 x 4096 | 0.9665 | 0.6258 | 0.647 | 0.0936 | 0.6400 | 0.662 | 4.5017 | 4.658 |
| mlp_up.t1 | 1 x 14336 x 4096 | 0.1694 | 0.0890 | 0.525 | 0.0431 | 0.1366 | 0.806 | 1.4800 | 8.737 |
| mlp_up.t8 | 8 x 14336 x 4096 | 0.1751 | 0.0957 | 0.547 | 0.0605 | 0.1761 | 1.006 | 1.5439 | 8.817 |
| mlp_up.t512 | 512 x 14336 x 4096 | 2.5449 | 1.8107 | 0.712 | 0.0907 | 1.9940 | 0.784 | 5.7417 | 2.256 |
| mlp_down.t1 | 1 x 4096 x 14336 | 0.1604 | 0.2034 | 1.268 | 0.0512 | 0.2813 | 1.754 | 4.8822 | 30.438 |
| mlp_down.t8 | 8 x 4096 x 14336 | 0.1972 | 0.2047 | 1.038 | 0.0838 | 0.3311 | 1.679 | 5.3084 | 26.919 |
| mlp_down.t512 | 512 x 4096 x 14336 | 2.6490 | 1.7276 | 0.652 | 0.1140 | 2.0296 | 0.766 | 15.4217 | 5.822 |
| lm_head.t1 | 1 x 128256 x 4096 | 1.3293 | 0.5694 | 0.428 | 0.0438 | 0.6331 | 0.476 | 1.9451 | 1.463 |
| lm_head.t8 | 8 x 128256 x 4096 | 1.2723 | 0.6733 | 0.529 | 0.0616 | 0.7791 | 0.612 | 2.0937 | 1.646 |
| lm_head.t512 | 512 x 16032 x 4096 (capped) | 2.8863 | 2.0605 | 0.714 | 0.0926 | 2.2532 | 0.781 | 6.0042 | 2.080 |

### Training: one product of a step, both operands converted per call

| row | m x n x k | fp32.v1 ms | product alone ms | over | A to planes ms | B to planes ms | complete product ms | over | complete, row quantizer ms | over |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| qkv.t512 | 512 x 4096 x 4096 | 0.9665 | 0.6258 | 0.647 | 0.0936 | 0.1933 | 0.8310 | 0.860 | 8.4273 | 8.719 |
| qkv.t512.bwd_dx | 512 x 4096 x 4096 | 0.9619 | 0.5199 | 0.540 | 0.0909 | 0.1873 | 0.7816 | 0.813 | 8.2407 | 8.567 |
| qkv.t512.bwd_dw | 4096 x 4096 x 512 | 1.0254 | 0.4643 | 0.453 | 0.1004 | 0.0998 | 0.6370 | 0.621 | 1.3187 | 1.286 |
| mlp_up.t512 | 512 x 14336 x 4096 | 2.5449 | 1.8107 | 0.712 | 0.0907 | 1.3272 | 3.3060 | 1.299 | 10.1247 | 3.978 |
| mlp_up.t512.bwd_dx | 512 x 4096 x 14336 | 2.6754 | 1.7248 | 0.645 | 0.1149 | 1.1999 | 3.1631 | 1.182 | 34.3126 | 12.825 |
| mlp_up.t512.bwd_dw | 14336 x 4096 x 512 | 2.3863 | 1.5756 | 0.660 | 0.1326 | 0.0969 | 1.7830 | 0.747 | 2.5797 | 1.081 |
| mlp_down.t512 | 512 x 4096 x 14336 | 2.6490 | 1.7276 | 0.652 | 0.1140 | 1.3526 | 3.3790 | 1.276 | 29.8250 | 11.259 |
| mlp_down.t512.bwd_dx | 512 x 14336 x 4096 | 2.6013 | 1.8129 | 0.697 | 0.0913 | 0.4618 | 2.5711 | 0.988 | 12.1090 | 4.655 |
| mlp_down.t512.bwd_dw | 4096 x 14336 x 512 | 2.3546 | 1.5664 | 0.665 | 0.0994 | 0.1364 | 1.7846 | 0.758 | 2.5551 | 1.085 |
| lm_head.t512 | 512 x 16032 x 4096 (capped) | 2.8863 | 2.0605 | 0.714 | 0.0926 | 1.4929 | 3.7486 | 1.299 | 10.4402 | 3.617 |
| lm_head.t512.bwd_dx | 512 x 512 x 128256 (capped) | 3.0972 | refused | refused | refused | refused | refused | refused | refused | refused |
| lm_head.t512.bwd_dw | 16032 x 4096 x 512 (capped) | 2.7402 | 1.7569 | 0.641 | 0.1305 | 0.0978 | 1.9694 | 0.719 | 2.7411 | 1.000 |

### Training: a layer's step, the forward product and the two backward products ADDED

Each line is the sum of three operations measured one at a time, not one measured operation.

| layer | fp32.v1, three products ms | fifteen-bit, three complete products ms | over |
|---|---:|---:|---:|
| qkv.t512 | 2.9538 | 2.2496 | 0.762 |
| mlp_up.t512 | 7.6066 | 8.2521 | 1.085 |
| mlp_down.t512 | 7.6049 | 7.7347 | 1.017 |
| lm_head.t512 | 8.7237 | refused (lm_head.t512.bwd_dx: k above 65536) | refused |

## M3 Ultra, run 1 (1790653519274, 77598eeb7)

Column `apple`. The product's plan on this box: no integer matrix unit; the flat kernel on codes and the pieces kernel on planes, one thread per cell.

### Inference: one call, weights packed once

| row | m x n x k | fp32.v1 ms | product alone ms | over | activations to planes ms | complete call ms | over | complete call, row quantizer ms | over |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| qkv.t1 | 1 x 4096 x 4096 | 1.0680 | 0.8120 | 0.760 | 0.2980 | 0.7860 | 0.736 | 1.6980 | 1.590 |
| qkv.t8 | 8 x 4096 x 4096 | 0.5760 | 1.2360 | 2.146 | 0.2980 | 1.2970 | 2.252 | 2.9260 | 5.080 |
| qkv.t512 | 512 x 4096 x 4096 | 3.3910 | 40.0910 | 11.823 | 0.3770 | 40.0270 | 11.804 | 78.9880 | 23.293 |
| mlp_up.t1 | 1 x 14336 x 4096 | 1.0340 | 0.7580 | 0.733 | 0.3010 | 0.8260 | 0.799 | 1.7030 | 1.647 |
| mlp_up.t8 | 8 x 14336 x 4096 | 1.0580 | 2.7410 | 2.591 | 0.3010 | 2.8040 | 2.650 | 4.9630 | 4.691 |
| mlp_up.t512 | 512 x 14336 x 4096 | 10.4240 | 137.88 | 13.227 | 0.3770 | 137.95 | 13.234 | 264.65 | 25.388 |
| mlp_down.t1 | 1 x 4096 x 14336 | 2.8940 | 2.0700 | 0.715 | 0.3080 | 2.1300 | 0.736 | 5.4820 | 1.894 |
| mlp_down.t8 | 8 x 4096 x 14336 | 1.3930 | 3.6960 | 2.653 | 0.2910 | 3.7770 | 2.711 | 9.2310 | 6.627 |
| mlp_down.t512 | 512 x 4096 x 14336 | 11.5060 | 139.15 | 12.094 | 0.6380 | 139.59 | 12.132 | 261.95 | 22.766 |
| lm_head.t1 | 1 x 128256 x 4096 | 4.0820 | 3.0510 | 0.747 | 0.2590 | 3.1250 | 0.766 | 6.5770 | 1.611 |
| lm_head.t8 | 8 x 128256 x 4096 | 5.6400 | 20.6300 | 3.658 | 0.2880 | 20.5520 | 3.644 | 42.3180 | 7.503 |
| lm_head.t512 | 512 x 16032 x 4096 (capped) | 11.5850 | 154.12 | 13.303 | 0.3790 | 154.06 | 13.298 | 288.86 | 24.934 |

### Training: one product of a step, both operands converted per call

| row | m x n x k | fp32.v1 ms | product alone ms | over | A to planes ms | B to planes ms | complete product ms | over | complete, row quantizer ms | over |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| qkv.t512 | 512 x 4096 x 4096 | 3.3910 | 40.0910 | 11.823 | 0.3770 | 1.2650 | 41.2640 | 12.169 | 80.8230 | 23.835 |
| qkv.t512.bwd_dx | 512 x 4096 x 4096 | 3.3800 | 40.0600 | 11.852 | 0.3780 | 1.3590 | 41.1070 | 12.162 | 83.0240 | 24.563 |
| qkv.t512.bwd_dw | 4096 x 4096 x 512 | 2.9170 | 40.1410 | 13.761 | 0.4510 | 0.4480 | 40.5940 | 13.916 | 77.2300 | 26.476 |
| mlp_up.t512 | 512 x 14336 x 4096 | 10.4240 | 137.88 | 13.227 | 0.3770 | 3.8970 | 141.65 | 13.589 | 267.44 | 25.656 |
| mlp_up.t512.bwd_dx | 512 x 4096 x 14336 | 11.4930 | 139.25 | 12.116 | 0.6470 | 5.0500 | 144.23 | 12.549 | 275.65 | 23.984 |
| mlp_up.t512.bwd_dw | 14336 x 4096 x 512 | 9.1720 | 139.67 | 15.228 | 0.7620 | 0.4470 | 140.47 | 15.315 | 283.20 | 30.877 |
| mlp_down.t512 | 512 x 4096 x 14336 | 11.5060 | 139.15 | 12.094 | 0.6380 | 3.7770 | 143.07 | 12.434 | 268.51 | 23.337 |
| mlp_down.t512.bwd_dx | 512 x 14336 x 4096 | 10.4650 | 137.84 | 13.172 | 0.3730 | 4.0670 | 141.47 | 13.518 | 269.33 | 25.736 |
| mlp_down.t512.bwd_dw | 4096 x 14336 x 512 | 9.1900 | 139.52 | 15.182 | 0.4540 | 0.7610 | 140.31 | 15.267 | 275.27 | 29.954 |
| lm_head.t512 | 512 x 16032 x 4096 (capped) | 11.5850 | 154.12 | 13.303 | 0.3790 | 4.3340 | 158.41 | 13.673 | 292.36 | 25.236 |
| lm_head.t512.bwd_dx | 512 x 512 x 128256 (capped) | 17.5330 | refused | refused | refused | refused | refused | refused | refused | refused |
| lm_head.t512.bwd_dw | 16032 x 4096 x 512 (capped) | 10.1810 | 156.27 | 15.349 | 0.8120 | 0.4430 | 157.12 | 15.433 | 315.88 | 31.026 |

### Training: a layer's step, the forward product and the two backward products ADDED

Each line is the sum of three operations measured one at a time, not one measured operation.

| layer | fp32.v1, three products ms | fifteen-bit, three complete products ms | over |
|---|---:|---:|---:|
| qkv.t512 | 9.6880 | 122.97 | 12.693 |
| mlp_up.t512 | 31.0890 | 426.35 | 13.714 |
| mlp_down.t512 | 31.1610 | 424.84 | 13.634 |
| lm_head.t512 | 39.2990 | refused (lm_head.t512.bwd_dx: k above 65536) | refused |

