## H100 run 2 (nvc3-0023) beside H100 run 3 (nvc3-0024)

`over` is the fifteen-bit time over fp32.v1's at the same row IN THE SAME RUN.

A = H100 run 2 (nvc3-0023). B = H100 run 3 (nvc3-0024).

### Inference: one call, weights packed once (complete = activations to planes, the product)

| row | A fp32.v1 ms | A product alone ms | over | A complete ms | over | B fp32.v1 ms | B product alone ms | over | B complete ms | over |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| qkv.t1 | 0.0708 | 0.2086 | 2.946 | 0.2506 | 3.540 | 0.0687 | 0.2073 | 3.017 | 0.2490 | 3.624 |
| qkv.t8 | 0.1033 | 0.1961 | 1.898 | 0.2529 | 2.448 | 0.1009 | 0.1929 | 1.912 | 0.2519 | 2.497 |
| qkv.t512 | 1.0368 | 1.2145 | 1.171 | 1.2540 | 1.209 | 1.0874 | 1.2241 | 1.126 | 1.2652 | 1.164 |
| mlp_up.t1 | 0.1794 | 0.2208 | 1.231 | 0.2476 | 1.380 | 0.1808 | 0.2206 | 1.220 | 0.2466 | 1.364 |
| mlp_up.t8 | 0.2734 | 0.2191 | 0.801 | 0.2537 | 0.928 | 0.2731 | 0.2171 | 0.795 | 0.2520 | 0.923 |
| mlp_up.t512 | 3.5682 | 6.0206 | 1.687 | 4.8621 | 1.363 | 3.5351 | 5.0173 | 1.419 | 4.5970 | 1.300 |
| mlp_down.t1 | 0.1715 | 0.7517 | 4.383 | 0.7856 | 4.581 | 0.1750 | 0.7517 | 4.295 | 0.7842 | 4.481 |
| mlp_down.t8 | 0.2897 | 0.7280 | 2.513 | 0.7782 | 2.686 | 0.2897 | 0.7259 | 2.506 | 0.7783 | 2.687 |
| mlp_down.t512 | 3.4189 | 5.0878 | 1.488 | 4.5559 | 1.333 | 3.4030 | 4.7043 | 1.382 | 4.5600 | 1.340 |
| lm_head.t1 | 1.2107 | 1.0330 | 0.853 | 0.9946 | 0.822 | 1.2108 | 1.1278 | 0.931 | 0.9947 | 0.822 |
| lm_head.t8 | 2.2336 | 1.3872 | 0.621 | 1.0975 | 0.491 | 2.2119 | 1.4095 | 0.637 | 1.1091 | 0.501 |
| lm_head.t512 | 3.9645 | 7.7217 | 1.948 | 5.7415 | 1.448 | 3.9377 | 7.4785 | 1.899 | 5.2913 | 1.344 |

### Training shapes: one product, both operands converted per call

| row | A fp32.v1 ms | A product alone ms | over | A complete ms | over | B fp32.v1 ms | B product alone ms | over | B complete ms | over |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| qkv.t512 | 1.0368 | 1.2145 | 1.171 | 1.4462 | 1.395 | 1.0874 | 1.2241 | 1.126 | 1.5286 | 1.406 |
| qkv.t512.bwd_dx | 1.0185 | 1.2125 | 1.190 | 1.4689 | 1.442 | 1.0204 | 1.1967 | 1.173 | 1.4521 | 1.423 |
| qkv.t512.bwd_dw | 0.8825 | 1.0556 | 1.196 | 1.1556 | 1.309 | 0.8833 | 1.0534 | 1.193 | 1.1527 | 1.305 |
| mlp_up.t512 | 3.5682 | 6.0206 | 1.687 | 4.8958 | 1.372 | 3.5351 | 5.0173 | 1.419 | 4.9010 | 1.386 |
| mlp_up.t512.bwd_dx | 3.3652 | 4.4927 | 1.335 | 5.3526 | 1.591 | 3.3537 | 4.4792 | 1.336 | 5.3515 | 1.596 |
| mlp_up.t512.bwd_dw | 2.9693 | 3.6322 | 1.223 | 3.7813 | 1.273 | 2.9656 | 3.6298 | 1.224 | 3.7802 | 1.275 |
| mlp_down.t512 | 3.4189 | 5.0878 | 1.488 | 5.0150 | 1.467 | 3.4030 | 4.7043 | 1.382 | 5.0164 | 1.474 |
| mlp_down.t512.bwd_dx | 3.7635 | 5.8397 | 1.552 | 5.6415 | 1.499 | 3.5785 | 5.2276 | 1.461 | 5.0457 | 1.410 |
| mlp_down.t512.bwd_dw | 3.1156 | 3.5825 | 1.150 | 3.7400 | 1.200 | 3.1124 | 3.5752 | 1.149 | 3.7406 | 1.202 |
| lm_head.t512 | 3.9645 | 7.7217 | 1.948 | 5.5125 | 1.390 | 3.9377 | 7.4785 | 1.899 | 5.6540 | 1.436 |
| lm_head.t512.bwd_dx | 3.8224 | refused | refused | refused | refused | 3.8074 | refused | refused | refused | refused |
| lm_head.t512.bwd_dw | 3.3540 | 4.0103 | 1.196 | 4.1863 | 1.248 | 3.3525 | 4.0105 | 1.196 | 4.1846 | 1.248 |

### Three products of one layer, each timed alone and added

The forward product, the input gradient and the weight gradient of one layer, each a complete product (both operands converted), measured one at a time and ADDED. Not an integrated training step.

| layer | A fp32.v1 ms | A fifteen-bit ms | over | B fp32.v1 ms | B fifteen-bit ms | over |
|---|---:|---:|---:|---:|---:|---:|
| qkv.t512 | 2.9378 | 4.0707 | 1.386 | 2.9911 | 4.1334 | 1.382 |
| mlp_up.t512 | 9.9027 | 14.0297 | 1.417 | 9.8544 | 14.0327 | 1.424 |
| mlp_down.t512 | 10.2980 | 14.3965 | 1.398 | 10.0939 | 13.8027 | 1.367 |
| lm_head.t512 | 11.1409 | refused (k above 65536) | refused | 11.0976 | refused (k above 65536) | refused |

