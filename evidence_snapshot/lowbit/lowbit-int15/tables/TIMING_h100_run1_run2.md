## H100 run 1 (nvc3-0017, 77598eeb7, fragment loads with no alignment stated) beside H100 run 2 (nvc3-0023, 42bb85725, alignment stated)

`over` is the fifteen-bit time over fp32.v1's at the same row IN THE SAME RUN.

A = H100 run 1 (nvc3-0017, 77598eeb7, fragment loads with no alignment stated). B = H100 run 2 (nvc3-0023, 42bb85725, alignment stated).

### Inference: one call, weights packed once (complete = activations to planes, the product)

| row | A fp32.v1 ms | A product alone ms | over | A complete ms | over | B fp32.v1 ms | B product alone ms | over | B complete ms | over |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| qkv.t1 | 0.0716 | 0.3343 | 4.669 | 0.4147 | 5.792 | 0.0708 | 0.2086 | 2.946 | 0.2506 | 3.540 |
| qkv.t8 | 0.1032 | 0.3343 | 3.239 | 0.4478 | 4.339 | 0.1033 | 0.1961 | 1.898 | 0.2529 | 2.448 |
| qkv.t512 | 1.0326 | 2.8384 | 2.749 | 2.8937 | 2.802 | 1.0368 | 1.2145 | 1.171 | 1.2540 | 1.209 |
| mlp_up.t1 | 0.1813 | 0.3752 | 2.069 | 0.4004 | 2.208 | 0.1794 | 0.2208 | 1.231 | 0.2476 | 1.380 |
| mlp_up.t8 | 0.2756 | 0.4139 | 1.502 | 0.4445 | 1.613 | 0.2734 | 0.2191 | 0.801 | 0.2537 | 0.928 |
| mlp_up.t512 | 3.4095 | 13.9276 | 4.085 | 11.6497 | 3.417 | 3.5682 | 6.0206 | 1.687 | 4.8621 | 1.363 |
| mlp_down.t1 | 0.1721 | 1.3292 | 7.723 | 1.3667 | 7.941 | 0.1715 | 0.7517 | 4.383 | 0.7856 | 4.581 |
| mlp_down.t8 | 0.2933 | 1.3875 | 4.731 | 1.4836 | 5.058 | 0.2897 | 0.7280 | 2.513 | 0.7782 | 2.686 |
| mlp_down.t512 | 3.4058 | 11.1343 | 3.269 | 10.2577 | 3.012 | 3.4189 | 5.0878 | 1.488 | 4.5559 | 1.333 |
| lm_head.t1 | 1.2043 | 1.7879 | 1.485 | 1.8135 | 1.506 | 1.2107 | 1.0330 | 0.853 | 0.9946 | 0.822 |
| lm_head.t8 | 2.1881 | 2.8407 | 1.298 | 2.5127 | 1.148 | 2.2336 | 1.3872 | 0.621 | 1.0975 | 0.491 |
| lm_head.t512 | 3.7968 | 15.9507 | 4.201 | 12.2982 | 3.239 | 3.9645 | 7.7217 | 1.948 | 5.7415 | 1.448 |

### Training shapes: one product, both operands converted per call

| row | A fp32.v1 ms | A product alone ms | over | A complete ms | over | B fp32.v1 ms | B product alone ms | over | B complete ms | over |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| qkv.t512 | 1.0326 | 2.8384 | 2.749 | 3.0663 | 2.969 | 1.0368 | 1.2145 | 1.171 | 1.4462 | 1.395 |
| qkv.t512.bwd_dx | 1.0184 | 2.8429 | 2.792 | 3.0899 | 3.034 | 1.0185 | 1.2125 | 1.190 | 1.4689 | 1.442 |
| qkv.t512.bwd_dw | 0.8872 | 2.6459 | 2.982 | 2.7483 | 3.098 | 0.8825 | 1.0556 | 1.196 | 1.1556 | 1.309 |
| mlp_up.t512 | 3.4095 | 13.9276 | 4.085 | 10.9194 | 3.203 | 3.5682 | 6.0206 | 1.687 | 4.8958 | 1.372 |
| mlp_up.t512.bwd_dx | 3.3542 | 10.7703 | 3.211 | 11.0779 | 3.303 | 3.3652 | 4.4927 | 1.335 | 5.3526 | 1.591 |
| mlp_up.t512.bwd_dw | 2.9681 | 9.1490 | 3.082 | 9.3034 | 3.134 | 2.9693 | 3.6322 | 1.223 | 3.7813 | 1.273 |
| mlp_down.t512 | 3.4058 | 11.1343 | 3.269 | 10.6300 | 3.121 | 3.4189 | 5.0878 | 1.488 | 5.0150 | 1.467 |
| mlp_down.t512.bwd_dx | 3.4260 | 14.9042 | 4.350 | 11.7965 | 3.443 | 3.7635 | 5.8397 | 1.552 | 5.6415 | 1.499 |
| mlp_down.t512.bwd_dw | 3.1161 | 9.1464 | 2.935 | 9.3155 | 2.989 | 3.1156 | 3.5825 | 1.150 | 3.7400 | 1.200 |
| lm_head.t512 | 3.7968 | 15.9507 | 4.201 | 11.9581 | 3.150 | 3.9645 | 7.7217 | 1.948 | 5.5125 | 1.390 |
| lm_head.t512.bwd_dx | 3.7888 | refused | refused | refused | refused | 3.8224 | refused | refused | refused | refused |
| lm_head.t512.bwd_dw | 3.3558 | 10.1303 | 3.019 | 10.3101 | 3.072 | 3.3540 | 4.0103 | 1.196 | 4.1863 | 1.248 |

### Three products of one layer, each timed alone and added

The forward product, the input gradient and the weight gradient of one layer, each a complete product (both operands converted), measured one at a time and ADDED. Not an integrated training step.

| layer | A fp32.v1 ms | A fifteen-bit ms | over | B fp32.v1 ms | B fifteen-bit ms | over |
|---|---:|---:|---:|---:|---:|---:|
| qkv.t512 | 2.9382 | 8.9045 | 3.031 | 2.9378 | 4.0707 | 1.386 |
| mlp_up.t512 | 9.7318 | 31.3007 | 3.216 | 9.9027 | 14.0297 | 1.417 |
| mlp_down.t512 | 9.9479 | 31.7420 | 3.191 | 10.2980 | 14.3965 | 1.398 |
| lm_head.t512 | 10.9414 | refused (k above 65536) | refused | 11.1409 | refused (k above 65536) | refused |

