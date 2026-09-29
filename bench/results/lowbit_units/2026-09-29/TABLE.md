# Low-bit GEMM plans: times, rates and digests

Written by `tools/lowbit_units/table.py` from the `LOWBIT` lines of
`bench/gemm_lowbit_price_main.mojo`. Every time is one box's time on one run: one call and one
synchronize per sample, the median of the timed calls, the minimum beside it. Every ratio is a time
over a time at the same shape on the same box; above 1 the numerator took longer. Rates are
G MAC/s (multiply-accumulates, `m n k`) for a product and G elem/s for a conversion.

## Summary: the plan each profile's dispatcher runs

`over` is the arm's median over fp32.v1's at the same shape on the same box. `+q` adds the per-call
quantization of the activations (`convert.int8.quantize.a`), a sum of two medians. The Apple probe
is not in any dispatcher.

### h100

| shape | m x n x k | fp32.v1 ms | bf16f32.v1 ms | over | int8i32.v1 ms | over | +q over | Apple probe ms | over | +q over |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| llama8b.qkv.t1 | 1 x 4096 x 4096 | 0.0707 | 0.7907 | 11.184 | 0.1573 | 2.225 | 9.158 | not run | |  |
| llama8b.qkv.t8 | 8 x 4096 x 4096 | 0.1008 | 0.1505 | 1.493 | 0.1478 | 1.466 | 7.409 | not run | |  |
| llama8b.qkv.t512 | 512 x 4096 x 4096 | 1.0328 | 1.0903 | 1.056 | 1.4361 | 1.390 | 3.938 | not run | |  |
| llama8b.mlp_up.t1 | 1 x 14336 x 4096 | 0.1904 | 0.7996 | 4.200 | 0.1933 | 1.015 | 3.650 | not run | |  |
| llama8b.mlp_up.t8 | 8 x 14336 x 4096 | 0.2722 | 0.4527 | 1.663 | 0.2103 | 0.773 | 3.025 | not run | |  |
| llama8b.mlp_up.t512 | 512 x 14336 x 4096 | 3.3374 | 3.5206 | 1.055 | 4.6300 | 1.387 | 2.179 | not run | |  |
| llama8b.mlp_down.t1 | 1 x 4096 x 14336 | 0.1737 | 2.7585 | 15.881 | 0.6498 | 3.741 | 13.738 | not run | |  |
| llama8b.mlp_down.t8 | 8 x 4096 x 14336 | 0.2868 | 0.4656 | 1.623 | 0.6992 | 2.438 | 10.880 | not run | |  |
| llama8b.mlp_down.t512 | 512 x 4096 x 14336 | 3.3603 | 3.5507 | 1.057 | 5.5780 | 1.660 | 4.631 | not run | |  |
| llama8b.lm_head.t1 | 1 x 128256 x 4096 | 1.2066 | 2.8032 | 2.323 | 0.9531 | 0.790 | 1.206 | not run | |  |
| llama8b.lm_head.t8 | 8 x 128256 x 4096 | 2.1553 | 3.7610 | 1.745 | 1.3711 | 0.636 | 0.919 | not run | |  |
| llama8b.lm_head.t512 | 512 x 16032 x 4096 (CAPPED) | 3.7224 | 3.9206 | 1.053 | 5.1254 | 1.377 | 2.056 | not run | |  |

### mi325x: not timed (identity only)

### m2pro

| shape | m x n x k | fp32.v1 ms | bf16f32.v1 ms | over | int8i32.v1 ms | over | +q over | Apple probe ms | over | +q over |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| llama8b.qkv.t1 | 1 x 4096 x 4096 | 2.0650 | 1.8670 | 0.904 | 1.7920 | 0.868 | 1.738 | 1.2530 | 0.607 | 1.477 |
| llama8b.qkv.t8 | 8 x 4096 x 4096 | 1.5080 | 2.0470 | 1.357 | 9.7300 | 6.452 | 7.367 | 0.8230 | 0.546 | 1.461 |
| llama8b.qkv.t512 | 512 x 4096 x 4096 | 17.8000 | 18.2540 | 1.026 | 515.1990 | 28.944 | 29.141 | 8.5930 | 0.483 | 0.680 |
| llama8b.mlp_up.t1 | 1 x 14336 x 4096 | 5.5180 | 5.3520 | 0.970 | 5.1680 | 0.937 | 1.206 | 2.1880 | 0.397 | 0.666 |
| llama8b.mlp_up.t8 | 8 x 14336 x 4096 | 4.1090 | 5.9930 | 1.459 | 29.4200 | 7.160 | 7.492 | 2.0040 | 0.488 | 0.820 |
| llama8b.mlp_up.t512 | 512 x 14336 x 4096 | 60.4090 | 62.3960 | 1.033 | 1799.2350 | 29.784 | 29.842 | 27.9870 | 0.463 | 0.521 |
| llama8b.mlp_down.t1 | 1 x 4096 x 14336 | 5.5890 | 5.1340 | 0.919 | 4.4290 | 0.792 | 1.556 | 2.4550 | 0.439 | 1.203 |
| llama8b.mlp_down.t8 | 8 x 4096 x 14336 | 4.9750 | 6.8790 | 1.383 | 33.9330 | 6.821 | 7.706 | 2.4370 | 0.490 | 1.375 |
| llama8b.mlp_down.t512 | 512 x 4096 x 14336 | 65.7240 | 66.8690 | 1.017 | 1828.9050 | 27.827 | 28.010 | 29.9820 | 0.456 | 0.639 |
| llama8b.lm_head.t1 | 1 x 128256 x 4096 | 33.1620 | 50.1550 | 1.512 | 32.6210 | 0.984 | 1.025 | 11.9000 | 0.359 | 0.400 |
| llama8b.lm_head.t8 | 8 x 128256 x 4096 | 33.2310 | 50.3410 | 1.515 | 252.5700 | 7.600 | 7.642 | 11.8680 | 0.357 | 0.398 |
| llama8b.lm_head.t512 | 512 x 16032 x 4096 (CAPPED) | 67.9340 | 69.7870 | 1.027 | 2011.2770 | 29.606 | 29.658 | 31.3130 | 0.461 | 0.512 |

## h100 (column nvidia)

Vendor comparison: cuBLAS on NVIDIA H100 NVL, torch 2.13.0+cu129, CUDA 12.9, median of 10. COMPARISON ONLY, no digest, no identity claim.

### llama8b.qkv.t1: m=1 n=4096 k=4096 (FULL)

| arm | median ms | min ms | rate | arm over fp32.v1 | note |
|---|---:|---:|---:|---:|---|
| fp32.v1 | 0.0707 | 0.0704 | 237.3149 GMAC/s | 1.000 | plan=13 |
| bf16f32.v1.fused | 0.7907 | 0.7879 | 21.2177 GMAC/s | 11.184 | dispatched |
| bf16f32.v1.widen | 0.1209 | 0.1187 | 138.7785 GMAC/s | 1.710 | not-dispatched(cells<=16384) |
| int8i32.v1.flat | 0.6653 | 0.6644 | 25.2187 GMAC/s | 9.410 | not-dispatched(the-column-has-the-unit) |
| int8i32.v1.mma | 0.1573 | 0.1563 | 106.6269 GMAC/s | 2.225 | dispatched |
| int8i32.v1.applechunk | not run | | | | the column does not have the unit |
| convert.int8.quantize.a | 0.4902 | 0.4899 | 0.0084 Gelem/s | 6.934 | per-call |
| convert.int8.pack.b | 2.6692 | 2.6663 | 6.2855 Gelem/s | 37.754 | once-per-weight |
| convert.bf16.pack.b | 0.0626 | 0.0623 | 268.1994 Gelem/s | 0.885 | once-per-weight |
| convert.bf16.widen.b | 0.0616 | 0.0614 | 272.2513 Gelem/s | 0.871 | per-call-when-materialized |
| convert.int8.dequantize.b | 0.0778 | 0.0774 | 215.7703 Gelem/s | 1.100 | per-call-when-materialized |
| DERIVED int8i32.v1.flat + quantize.a | 1.1555 | | | 16.344 | sum of two medians |
| DERIVED int8i32.v1.mma + quantize.a | 0.6475 | | | 9.158 | sum of two medians |
| vendor strict (COMPARISON ONLY) | 0.0402 | 0.0392 | 417.4058 GMAC/s | | fp32.v1 over this: 1.759 |
| vendor tf32 (COMPARISON ONLY) | 0.0403 | 0.0388 | 416.4505 GMAC/s | | fp32.v1 over this: 1.755 |
| vendor bf16 (COMPARISON ONLY) | 0.0282 | 0.0268 | 594.0739 GMAC/s | | fp32.v1 over this: 2.503 |

### llama8b.qkv.t8: m=8 n=4096 k=4096 (FULL)

| arm | median ms | min ms | rate | arm over fp32.v1 | note |
|---|---:|---:|---:|---:|---|
| fp32.v1 | 0.1008 | 0.0997 | 1331.0761 GMAC/s | 1.000 | plan=14 |
| bf16f32.v1.fused | 0.7813 | 0.7811 | 171.7972 GMAC/s | 7.751 | not-dispatched(cells>16384) |
| bf16f32.v1.widen | 0.1505 | 0.1499 | 891.8299 GMAC/s | 1.493 | dispatched |
| int8i32.v1.flat | 0.6647 | 0.6628 | 201.9177 GMAC/s | 6.594 | not-dispatched(the-column-has-the-unit) |
| int8i32.v1.mma | 0.1478 | 0.1473 | 908.0238 GMAC/s | 1.466 | dispatched |
| int8i32.v1.applechunk | not run | | | | the column does not have the unit |
| convert.int8.quantize.a | 0.5990 | 0.5978 | 0.0547 Gelem/s | 5.942 | per-call |
| convert.int8.pack.b | 2.4738 | 2.4683 | 6.7819 Gelem/s | 24.542 | once-per-weight |
| convert.bf16.pack.b | 0.0622 | 0.0620 | 269.6045 Gelem/s | 0.617 | once-per-weight |
| convert.bf16.widen.b | 0.0612 | 0.0609 | 274.2764 Gelem/s | 0.607 | per-call-when-materialized |
| convert.int8.dequantize.b | 0.0780 | 0.0775 | 215.0126 Gelem/s | 0.774 | per-call-when-materialized |
| DERIVED int8i32.v1.flat + quantize.a | 1.2637 | | | 12.537 | sum of two medians |
| DERIVED int8i32.v1.mma + quantize.a | 0.7468 | | | 7.409 | sum of two medians |
| vendor strict (COMPARISON ONLY) | 0.0651 | 0.0629 | 2061.1587 GMAC/s | | fp32.v1 over this: 1.548 |
| vendor tf32 (COMPARISON ONLY) | 0.0443 | 0.0434 | 3032.6632 GMAC/s | | fp32.v1 over this: 2.278 |
| vendor bf16 (COMPARISON ONLY) | 0.0264 | 0.0258 | 5089.9814 GMAC/s | | fp32.v1 over this: 3.823 |

### llama8b.qkv.t512: m=512 n=4096 k=4096 (FULL)

| arm | median ms | min ms | rate | arm over fp32.v1 | note |
|---|---:|---:|---:|---:|---|
| fp32.v1 | 1.0328 | 1.0308 | 8317.1729 GMAC/s | 1.000 | plan=10 |
| bf16f32.v1.fused | 39.4281 | 39.3688 | 217.8633 GMAC/s | 38.176 | not-dispatched(cells>16384) |
| bf16f32.v1.widen | 1.0903 | 1.0873 | 7878.3755 GMAC/s | 1.056 | dispatched |
| int8i32.v1.flat | 39.1874 | 39.1752 | 219.2012 GMAC/s | 37.943 | not-dispatched(the-column-has-the-unit) |
| int8i32.v1.mma | 1.4361 | 1.4256 | 5981.2780 GMAC/s | 1.390 | dispatched |
| int8i32.v1.applechunk | not run | | | | the column does not have the unit |
| convert.int8.quantize.a | 2.6307 | 2.6184 | 0.7972 Gelem/s | 2.547 | per-call |
| convert.int8.pack.b | 2.4754 | 2.4688 | 6.7775 Gelem/s | 2.397 | once-per-weight |
| convert.bf16.pack.b | 0.0646 | 0.0628 | 259.6771 Gelem/s | 0.063 | once-per-weight |
| convert.bf16.widen.b | 0.0611 | 0.0605 | 274.6536 Gelem/s | 0.059 | per-call-when-materialized |
| convert.int8.dequantize.b | 0.0784 | 0.0779 | 213.9023 Gelem/s | 0.076 | per-call-when-materialized |
| DERIVED int8i32.v1.flat + quantize.a | 41.8181 | | | 40.490 | sum of two medians |
| DERIVED int8i32.v1.mma + quantize.a | 4.0668 | | | 3.938 | sum of two medians |
| vendor strict (COMPARISON ONLY) | 0.4199 | 0.4169 | 20457.8307 GMAC/s | | fp32.v1 over this: 2.460 |
| vendor tf32 (COMPARISON ONLY) | 0.0789 | 0.0766 | 108876.5970 GMAC/s | | fp32.v1 over this: 13.091 |
| vendor bf16 (COMPARISON ONLY) | 0.0419 | 0.0410 | 205070.9156 GMAC/s | | fp32.v1 over this: 24.656 |

### llama8b.mlp_up.t1: m=1 n=14336 k=4096 (FULL)

| arm | median ms | min ms | rate | arm over fp32.v1 | note |
|---|---:|---:|---:|---:|---|
| fp32.v1 | 0.1904 | 0.1902 | 308.3675 GMAC/s | 1.000 | plan=13 |
| bf16f32.v1.fused | 0.7996 | 0.7927 | 73.4403 GMAC/s | 4.200 | dispatched |
| bf16f32.v1.widen | 0.3681 | 0.3654 | 159.5343 GMAC/s | 1.933 | not-dispatched(cells<=16384) |
| int8i32.v1.flat | 0.6655 | 0.6646 | 88.2367 GMAC/s | 3.495 | not-dispatched(the-column-has-the-unit) |
| int8i32.v1.mma | 0.1933 | 0.1918 | 303.7024 GMAC/s | 1.015 | dispatched |
| int8i32.v1.applechunk | not run | | | | the column does not have the unit |
| convert.int8.quantize.a | 0.5016 | 0.4999 | 0.0082 Gelem/s | 2.634 | per-call |
| convert.int8.pack.b | 2.4783 | 2.4735 | 23.6938 Gelem/s | 13.016 | once-per-weight |
| convert.bf16.pack.b | 0.2011 | 0.1996 | 291.9779 Gelem/s | 1.056 | once-per-weight |
| convert.bf16.widen.b | 0.1897 | 0.1895 | 309.5476 Gelem/s | 0.996 | per-call-when-materialized |
| convert.int8.dequantize.b | 0.2464 | 0.2456 | 238.3417 Gelem/s | 1.294 | per-call-when-materialized |
| DERIVED int8i32.v1.flat + quantize.a | 1.1671 | | | 6.130 | sum of two medians |
| DERIVED int8i32.v1.mma + quantize.a | 0.6949 | | | 3.650 | sum of two medians |
| vendor strict (COMPARISON ONLY) | 0.0873 | 0.0859 | 672.3367 GMAC/s | | fp32.v1 over this: 2.180 |
| vendor tf32 (COMPARISON ONLY) | 0.0876 | 0.0862 | 670.0290 GMAC/s | | fp32.v1 over this: 2.173 |
| vendor bf16 (COMPARISON ONLY) | 0.0518 | 0.0502 | 1133.8979 GMAC/s | | fp32.v1 over this: 3.677 |

### llama8b.mlp_up.t8: m=8 n=14336 k=4096 (FULL)

| arm | median ms | min ms | rate | arm over fp32.v1 | note |
|---|---:|---:|---:|---:|---|
| fp32.v1 | 0.2722 | 0.2713 | 1725.8544 GMAC/s | 1.000 | plan=14 |
| bf16f32.v1.fused | 2.5325 | 2.5309 | 185.4900 GMAC/s | 9.304 | not-dispatched(cells>16384) |
| bf16f32.v1.widen | 0.4527 | 0.4519 | 1037.6368 GMAC/s | 1.663 | dispatched |
| int8i32.v1.flat | 2.5044 | 2.5039 | 187.5731 GMAC/s | 9.201 | not-dispatched(the-column-has-the-unit) |
| int8i32.v1.mma | 0.2103 | 0.2095 | 2233.6011 GMAC/s | 0.773 | dispatched |
| int8i32.v1.applechunk | not run | | | | the column does not have the unit |
| convert.int8.quantize.a | 0.6130 | 0.6127 | 0.0535 Gelem/s | 2.252 | per-call |
| convert.int8.pack.b | 2.4833 | 2.4807 | 23.6459 Gelem/s | 9.123 | once-per-weight |
| convert.bf16.pack.b | 0.1979 | 0.1972 | 296.6673 Gelem/s | 0.727 | once-per-weight |
| convert.bf16.widen.b | 0.1879 | 0.1873 | 312.5047 Gelem/s | 0.690 | per-call-when-materialized |
| convert.int8.dequantize.b | 0.2460 | 0.2454 | 238.7235 Gelem/s | 0.904 | per-call-when-materialized |
| DERIVED int8i32.v1.flat + quantize.a | 3.1174 | | | 11.453 | sum of two medians |
| DERIVED int8i32.v1.mma + quantize.a | 0.8233 | | | 3.025 | sum of two medians |
| vendor strict (COMPARISON ONLY) | 0.1529 | 0.1500 | 3073.1649 GMAC/s | | fp32.v1 over this: 1.781 |
| vendor tf32 (COMPARISON ONLY) | 0.0986 | 0.0974 | 4766.1642 GMAC/s | | fp32.v1 over this: 2.762 |
| vendor bf16 (COMPARISON ONLY) | 0.0516 | 0.0504 | 9105.7363 GMAC/s | | fp32.v1 over this: 5.276 |

### llama8b.mlp_up.t512: m=512 n=14336 k=4096 (FULL)

| arm | median ms | min ms | rate | arm over fp32.v1 | note |
|---|---:|---:|---:|---:|---|
| fp32.v1 | 3.3374 | 3.3351 | 9008.4005 GMAC/s | 1.000 | plan=10 |
| bf16f32.v1.fused | 139.4871 | 138.9407 | 215.5380 GMAC/s | 41.795 | not-dispatched(cells>16384) |
| bf16f32.v1.widen | 3.5206 | 3.5162 | 8539.6604 GMAC/s | 1.055 | dispatched |
| int8i32.v1.flat | 135.6760 | 135.6479 | 221.5925 GMAC/s | 40.653 | not-dispatched(the-column-has-the-unit) |
| int8i32.v1.mma | 4.6300 | 4.6258 | 6493.5398 GMAC/s | 1.387 | dispatched |
| int8i32.v1.applechunk | not run | | | | the column does not have the unit |
| convert.int8.quantize.a | 2.6426 | 2.6395 | 0.7936 Gelem/s | 0.792 | per-call |
| convert.int8.pack.b | 2.4761 | 2.4706 | 23.7145 Gelem/s | 0.742 | once-per-weight |
| convert.bf16.pack.b | 0.1964 | 0.1957 | 298.9723 Gelem/s | 0.059 | once-per-weight |
| convert.bf16.widen.b | 0.1870 | 0.1863 | 313.9718 Gelem/s | 0.056 | per-call-when-materialized |
| convert.int8.dequantize.b | 0.2446 | 0.2439 | 240.0959 Gelem/s | 0.073 | per-call-when-materialized |
| DERIVED int8i32.v1.flat + quantize.a | 138.3186 | | | 41.445 | sum of two medians |
| DERIVED int8i32.v1.mma + quantize.a | 7.2726 | | | 2.179 | sum of two medians |
| vendor strict (COMPARISON ONLY) | 1.6535 | 1.6473 | 18182.5823 GMAC/s | | fp32.v1 over this: 2.018 |
| vendor tf32 (COMPARISON ONLY) | 0.2461 | 0.2429 | 122180.9757 GMAC/s | | fp32.v1 over this: 13.563 |
| vendor bf16 (COMPARISON ONLY) | 0.1391 | 0.1378 | 216105.2492 GMAC/s | | fp32.v1 over this: 23.989 |

### llama8b.mlp_down.t1: m=1 n=4096 k=14336 (FULL)

| arm | median ms | min ms | rate | arm over fp32.v1 | note |
|---|---:|---:|---:|---:|---|
| fp32.v1 | 0.1737 | 0.1730 | 338.0867 GMAC/s | 1.000 | plan=13 |
| bf16f32.v1.fused | 2.7585 | 2.7547 | 21.2872 GMAC/s | 15.881 | dispatched |
| bf16f32.v1.widen | 0.3521 | 0.3508 | 166.7753 GMAC/s | 2.027 | not-dispatched(cells<=16384) |
| int8i32.v1.flat | 2.3453 | 2.3389 | 25.0372 GMAC/s | 13.502 | not-dispatched(the-column-has-the-unit) |
| int8i32.v1.mma | 0.6498 | 0.6497 | 90.3676 GMAC/s | 3.741 | dispatched |
| int8i32.v1.applechunk | not run | | | | the column does not have the unit |
| convert.int8.quantize.a | 1.7365 | 1.7302 | 0.0083 Gelem/s | 9.997 | per-call |
| convert.int8.pack.b | 9.9971 | 9.9633 | 5.8738 Gelem/s | 57.554 | once-per-weight |
| convert.bf16.pack.b | 0.1978 | 0.1977 | 296.8758 Gelem/s | 1.139 | once-per-weight |
| convert.bf16.widen.b | 0.1869 | 0.1863 | 314.2356 Gelem/s | 1.076 | per-call-when-materialized |
| convert.int8.dequantize.b | 0.2434 | 0.2422 | 241.2480 Gelem/s | 1.401 | per-call-when-materialized |
| DERIVED int8i32.v1.flat + quantize.a | 4.0818 | | | 23.499 | sum of two medians |
| DERIVED int8i32.v1.mma + quantize.a | 2.3863 | | | 13.738 | sum of two medians |
| vendor strict (COMPARISON ONLY) | 0.0821 | 0.0807 | 715.0476 GMAC/s | | fp32.v1 over this: 2.115 |
| vendor tf32 (COMPARISON ONLY) | 0.0814 | 0.0806 | 721.7224 GMAC/s | | fp32.v1 over this: 2.135 |
| vendor bf16 (COMPARISON ONLY) | 0.0552 | 0.0538 | 1064.7167 GMAC/s | | fp32.v1 over this: 3.150 |

### llama8b.mlp_down.t8: m=8 n=4096 k=14336 (FULL)

| arm | median ms | min ms | rate | arm over fp32.v1 | note |
|---|---:|---:|---:|---:|---|
| fp32.v1 | 0.2868 | 0.2857 | 1637.8002 GMAC/s | 1.000 | plan=14 |
| bf16f32.v1.fused | 2.6961 | 2.6918 | 174.2353 GMAC/s | 9.401 | not-dispatched(cells>16384) |
| bf16f32.v1.widen | 0.4656 | 0.4649 | 1008.8719 GMAC/s | 1.623 | dispatched |
| int8i32.v1.flat | 2.3383 | 2.3372 | 200.8966 GMAC/s | 8.153 | not-dispatched(the-column-has-the-unit) |
| int8i32.v1.mma | 0.6992 | 0.6911 | 671.8507 GMAC/s | 2.438 | dispatched |
| int8i32.v1.applechunk | not run | | | | the column does not have the unit |
| convert.int8.quantize.a | 2.4213 | 2.4205 | 0.0474 Gelem/s | 8.442 | per-call |
| convert.int8.pack.b | 9.9767 | 9.9525 | 5.8857 Gelem/s | 34.786 | once-per-weight |
| convert.bf16.pack.b | 0.1975 | 0.1965 | 297.2560 Gelem/s | 0.689 | once-per-weight |
| convert.bf16.widen.b | 0.1867 | 0.1864 | 314.4577 Gelem/s | 0.651 | per-call-when-materialized |
| convert.int8.dequantize.b | 0.2436 | 0.2428 | 241.0094 Gelem/s | 0.849 | per-call-when-materialized |
| DERIVED int8i32.v1.flat + quantize.a | 4.7596 | | | 16.596 | sum of two medians |
| DERIVED int8i32.v1.mma + quantize.a | 3.1205 | | | 10.880 | sum of two medians |
| vendor strict (COMPARISON ONLY) | 0.1954 | 0.1938 | 2404.4960 GMAC/s | | fp32.v1 over this: 1.468 |
| vendor tf32 (COMPARISON ONLY) | 0.0945 | 0.0928 | 4972.7718 GMAC/s | | fp32.v1 over this: 3.036 |
| vendor bf16 (COMPARISON ONLY) | 0.0552 | 0.0543 | 8504.2345 GMAC/s | | fp32.v1 over this: 5.192 |

### llama8b.mlp_down.t512: m=512 n=4096 k=14336 (FULL)

| arm | median ms | min ms | rate | arm over fp32.v1 | note |
|---|---:|---:|---:|---:|---|
| fp32.v1 | 3.3603 | 3.3576 | 8946.9272 GMAC/s | 1.000 | plan=10 |
| bf16f32.v1.fused | 136.4673 | 136.4099 | 220.3076 GMAC/s | 40.612 | not-dispatched(cells>16384) |
| bf16f32.v1.widen | 3.5507 | 3.5452 | 8467.3468 GMAC/s | 1.057 | dispatched |
| int8i32.v1.flat | 137.9775 | 137.9547 | 217.8962 GMAC/s | 41.061 | not-dispatched(the-column-has-the-unit) |
| int8i32.v1.mma | 5.5780 | 5.5365 | 5389.8411 GMAC/s | 1.660 | dispatched |
| int8i32.v1.applechunk | not run | | | | the column does not have the unit |
| convert.int8.quantize.a | 9.9819 | 9.9540 | 0.7353 Gelem/s | 2.971 | per-call |
| convert.int8.pack.b | 9.9797 | 9.9596 | 5.8840 Gelem/s | 2.970 | once-per-weight |
| convert.bf16.pack.b | 0.2009 | 0.1989 | 292.2613 Gelem/s | 0.060 | once-per-weight |
| convert.bf16.widen.b | 0.1883 | 0.1869 | 311.9055 Gelem/s | 0.056 | per-call-when-materialized |
| convert.int8.dequantize.b | 0.2468 | 0.2449 | 237.8908 Gelem/s | 0.073 | per-call-when-materialized |
| DERIVED int8i32.v1.flat + quantize.a | 147.9594 | | | 44.032 | sum of two medians |
| DERIVED int8i32.v1.mma + quantize.a | 15.5599 | | | 4.631 | sum of two medians |
| vendor strict (COMPARISON ONLY) | 1.3886 | 1.3847 | 21650.9266 GMAC/s | | fp32.v1 over this: 2.420 |
| vendor tf32 (COMPARISON ONLY) | 0.2827 | 0.2755 | 106358.4228 GMAC/s | | fp32.v1 over this: 11.888 |
| vendor bf16 (COMPARISON ONLY) | 0.1372 | 0.1352 | 219079.4976 GMAC/s | | fp32.v1 over this: 24.486 |

### llama8b.lm_head.t1: m=1 n=128256 k=4096 (FULL)

| arm | median ms | min ms | rate | arm over fp32.v1 | note |
|---|---:|---:|---:|---:|---|
| fp32.v1 | 1.2066 | 1.2038 | 435.3819 GMAC/s | 1.000 | plan=13 |
| bf16f32.v1.fused | 2.5838 | 2.5759 | 203.3168 GMAC/s | 2.141 | not-dispatched(cells>16384) |
| bf16f32.v1.widen | 2.8032 | 2.7992 | 187.4064 GMAC/s | 2.323 | dispatched |
| int8i32.v1.flat | 2.4982 | 2.4941 | 210.2869 GMAC/s | 2.070 | not-dispatched(the-column-has-the-unit) |
| int8i32.v1.mma | 0.9531 | 0.9509 | 551.1740 GMAC/s | 0.790 | dispatched |
| int8i32.v1.applechunk | not run | | | | the column does not have the unit |
| convert.int8.quantize.a | 0.5025 | 0.5023 | 0.0082 Gelem/s | 0.416 | per-call |
| convert.int8.pack.b | 12.2065 | 12.1071 | 43.0375 Gelem/s | 10.116 | once-per-weight |
| convert.bf16.pack.b | 1.6701 | 1.6680 | 314.5583 Gelem/s | 1.384 | once-per-weight |
| convert.bf16.widen.b | 1.6085 | 1.6070 | 326.6102 Gelem/s | 1.333 | per-call-when-materialized |
| convert.int8.dequantize.b | 2.0993 | 2.0957 | 250.2426 Gelem/s | 1.740 | per-call-when-materialized |
| DERIVED int8i32.v1.flat + quantize.a | 3.0007 | | | 2.487 | sum of two medians |
| DERIVED int8i32.v1.mma + quantize.a | 1.4556 | | | 1.206 | sum of two medians |
| vendor strict (COMPARISON ONLY) | 0.7224 | 0.7116 | 727.2135 GMAC/s | | fp32.v1 over this: 1.670 |
| vendor tf32 (COMPARISON ONLY) | 0.7112 | 0.7008 | 738.7011 GMAC/s | | fp32.v1 over this: 1.697 |
| vendor bf16 (COMPARISON ONLY) | 0.3226 | 0.3170 | 1628.6954 GMAC/s | | fp32.v1 over this: 3.741 |

### llama8b.lm_head.t8: m=8 n=128256 k=4096 (FULL)

| arm | median ms | min ms | rate | arm over fp32.v1 | note |
|---|---:|---:|---:|---:|---|
| fp32.v1 | 2.1553 | 2.1538 | 1949.9285 GMAC/s | 1.000 | plan=14 |
| bf16f32.v1.fused | 19.8922 | 19.8131 | 211.2734 GMAC/s | 9.229 | not-dispatched(cells>16384) |
| bf16f32.v1.widen | 3.7610 | 3.7553 | 1117.4378 GMAC/s | 1.745 | dispatched |
| int8i32.v1.flat | 19.2560 | 19.2511 | 218.2542 GMAC/s | 8.934 | not-dispatched(the-column-has-the-unit) |
| int8i32.v1.mma | 1.3711 | 1.3660 | 3065.1597 GMAC/s | 0.636 | dispatched |
| int8i32.v1.applechunk | not run | | | | the column does not have the unit |
| convert.int8.quantize.a | 0.6099 | 0.6078 | 0.0537 Gelem/s | 0.283 | per-call |
| convert.int8.pack.b | 12.1895 | 12.1404 | 43.0976 Gelem/s | 5.656 | once-per-weight |
| convert.bf16.pack.b | 1.6709 | 1.6678 | 314.4079 Gelem/s | 0.775 | once-per-weight |
| convert.bf16.widen.b | 1.6090 | 1.6084 | 326.4978 Gelem/s | 0.747 | per-call-when-materialized |
| convert.int8.dequantize.b | 2.0977 | 2.0952 | 250.4323 Gelem/s | 0.973 | per-call-when-materialized |
| DERIVED int8i32.v1.flat + quantize.a | 19.8659 | | | 9.217 | sum of two medians |
| DERIVED int8i32.v1.mma + quantize.a | 1.9810 | | | 0.919 | sum of two medians |
| vendor strict (COMPARISON ONLY) | 1.0847 | 1.0779 | 3874.4800 GMAC/s | | fp32.v1 over this: 1.987 |
| vendor tf32 (COMPARISON ONLY) | 0.8190 | 0.8141 | 5131.5807 GMAC/s | | fp32.v1 over this: 2.632 |
| vendor bf16 (COMPARISON ONLY) | 0.3205 | 0.3191 | 13110.9921 GMAC/s | | fp32.v1 over this: 6.724 |

### llama8b.lm_head.t512: m=512 n=16032 k=4096 (CAPPED)

| arm | median ms | min ms | rate | arm over fp32.v1 | note |
|---|---:|---:|---:|---:|---|
| fp32.v1 | 3.7224 | 3.7173 | 9032.2869 GMAC/s | 1.000 | plan=10 |
| bf16f32.v1.fused | 156.5508 | 156.4679 | 214.7644 GMAC/s | 42.056 | not-dispatched(cells>16384) |
| bf16f32.v1.widen | 3.9206 | 3.9181 | 8575.6636 GMAC/s | 1.053 | dispatched |
| int8i32.v1.flat | 151.8644 | 151.8112 | 221.3919 GMAC/s | 40.797 | not-dispatched(the-column-has-the-unit) |
| int8i32.v1.mma | 5.1254 | 5.1231 | 6559.8156 GMAC/s | 1.377 | dispatched |
| int8i32.v1.applechunk | not run | | | | the column does not have the unit |
| convert.int8.quantize.a | 2.5282 | 2.5201 | 0.8295 Gelem/s | 0.679 | per-call |
| convert.int8.pack.b | 2.6335 | 2.6234 | 24.9354 Gelem/s | 0.707 | once-per-weight |
| convert.bf16.pack.b | 0.2196 | 0.2180 | 298.9705 Gelem/s | 0.059 | once-per-weight |
| convert.bf16.widen.b | 0.2084 | 0.2079 | 315.1556 Gelem/s | 0.056 | per-call-when-materialized |
| convert.int8.dequantize.b | 0.2715 | 0.2705 | 241.8570 Gelem/s | 0.073 | per-call-when-materialized |
| DERIVED int8i32.v1.flat + quantize.a | 154.3926 | | | 41.477 | sum of two medians |
| DERIVED int8i32.v1.mma + quantize.a | 7.6536 | | | 2.056 | sum of two medians |
| vendor strict (COMPARISON ONLY) | 15.4044 | 13.9532 | | n/a | the vendor ran the whole row (n=128256), ours ran capped: no ratio |
| vendor tf32 (COMPARISON ONLY) | 2.3269 | 2.2688 | | n/a | the vendor ran the whole row (n=128256), ours ran capped: no ratio |
| vendor bf16 (COMPARISON ONLY) | 1.2070 | 1.2008 | | n/a | the vendor ran the whole row (n=128256), ours ran capped: no ratio |

## mi325x (column amd)

NOT TIMED (IDENTITY ONLY). This box is judged on bitwise identity (Andrew, 2026-09-29); the speed
gate is judged on NVIDIA and on Apple. Its digests are in the last section.

### llama8b.qkv.t1: m=1 n=4096 k=4096 (FULL)

| arm | median ms | min ms | rate | arm over fp32.v1 | note |
|---|---:|---:|---:|---:|---|
| fp32.v1 | not timed (identity only) | not timed (identity only) | not timed (identity only) | not timed (identity only) | plan=13 |
| bf16f32.v1.fused | not timed (identity only) | not timed (identity only) | not timed (identity only) | not timed (identity only) | dispatched |
| bf16f32.v1.widen | not timed (identity only) | not timed (identity only) | not timed (identity only) | not timed (identity only) | not-dispatched(cells<=16384) |
| int8i32.v1.flat | not timed (identity only) | not timed (identity only) | not timed (identity only) | not timed (identity only) | not-dispatched(the-column-has-the-unit) |
| int8i32.v1.mma | not timed (identity only) | not timed (identity only) | not timed (identity only) | not timed (identity only) | dispatched |
| int8i32.v1.applechunk | not run | | | | the column does not have the unit |
| convert.int8.quantize.a | not timed (identity only) | not timed (identity only) | not timed (identity only) | not timed (identity only) | per-call |
| convert.int8.pack.b | not timed (identity only) | not timed (identity only) | not timed (identity only) | not timed (identity only) | once-per-weight |
| convert.bf16.pack.b | not timed (identity only) | not timed (identity only) | not timed (identity only) | not timed (identity only) | once-per-weight |
| convert.bf16.widen.b | not timed (identity only) | not timed (identity only) | not timed (identity only) | not timed (identity only) | per-call-when-materialized |
| convert.int8.dequantize.b | not timed (identity only) | not timed (identity only) | not timed (identity only) | not timed (identity only) | per-call-when-materialized |

### llama8b.qkv.t8: m=8 n=4096 k=4096 (FULL)

| arm | median ms | min ms | rate | arm over fp32.v1 | note |
|---|---:|---:|---:|---:|---|
| fp32.v1 | not timed (identity only) | not timed (identity only) | not timed (identity only) | not timed (identity only) | plan=14 |
| bf16f32.v1.fused | not timed (identity only) | not timed (identity only) | not timed (identity only) | not timed (identity only) | not-dispatched(cells>16384) |
| bf16f32.v1.widen | not timed (identity only) | not timed (identity only) | not timed (identity only) | not timed (identity only) | dispatched |
| int8i32.v1.flat | not timed (identity only) | not timed (identity only) | not timed (identity only) | not timed (identity only) | not-dispatched(the-column-has-the-unit) |
| int8i32.v1.mma | not timed (identity only) | not timed (identity only) | not timed (identity only) | not timed (identity only) | dispatched |
| int8i32.v1.applechunk | not run | | | | the column does not have the unit |
| convert.int8.quantize.a | not timed (identity only) | not timed (identity only) | not timed (identity only) | not timed (identity only) | per-call |
| convert.int8.pack.b | not timed (identity only) | not timed (identity only) | not timed (identity only) | not timed (identity only) | once-per-weight |
| convert.bf16.pack.b | not timed (identity only) | not timed (identity only) | not timed (identity only) | not timed (identity only) | once-per-weight |
| convert.bf16.widen.b | not timed (identity only) | not timed (identity only) | not timed (identity only) | not timed (identity only) | per-call-when-materialized |
| convert.int8.dequantize.b | not timed (identity only) | not timed (identity only) | not timed (identity only) | not timed (identity only) | per-call-when-materialized |

### llama8b.qkv.t512: m=512 n=4096 k=4096 (FULL)

| arm | median ms | min ms | rate | arm over fp32.v1 | note |
|---|---:|---:|---:|---:|---|
| fp32.v1 | not timed (identity only) | not timed (identity only) | not timed (identity only) | not timed (identity only) | plan=10 |
| bf16f32.v1.fused | not timed (identity only) | not timed (identity only) | not timed (identity only) | not timed (identity only) | not-dispatched(cells>16384) |
| bf16f32.v1.widen | not timed (identity only) | not timed (identity only) | not timed (identity only) | not timed (identity only) | dispatched |
| int8i32.v1.flat | not timed (identity only) | not timed (identity only) | not timed (identity only) | not timed (identity only) | not-dispatched(the-column-has-the-unit) |
| int8i32.v1.mma | not timed (identity only) | not timed (identity only) | not timed (identity only) | not timed (identity only) | dispatched |
| int8i32.v1.applechunk | not run | | | | the column does not have the unit |
| convert.int8.quantize.a | not timed (identity only) | not timed (identity only) | not timed (identity only) | not timed (identity only) | per-call |
| convert.int8.pack.b | not timed (identity only) | not timed (identity only) | not timed (identity only) | not timed (identity only) | once-per-weight |
| convert.bf16.pack.b | not timed (identity only) | not timed (identity only) | not timed (identity only) | not timed (identity only) | once-per-weight |
| convert.bf16.widen.b | not timed (identity only) | not timed (identity only) | not timed (identity only) | not timed (identity only) | per-call-when-materialized |
| convert.int8.dequantize.b | not timed (identity only) | not timed (identity only) | not timed (identity only) | not timed (identity only) | per-call-when-materialized |

### llama8b.mlp_up.t1: m=1 n=14336 k=4096 (FULL)

| arm | median ms | min ms | rate | arm over fp32.v1 | note |
|---|---:|---:|---:|---:|---|
| fp32.v1 | not timed (identity only) | not timed (identity only) | not timed (identity only) | not timed (identity only) | plan=13 |
| bf16f32.v1.fused | not timed (identity only) | not timed (identity only) | not timed (identity only) | not timed (identity only) | dispatched |
| bf16f32.v1.widen | not timed (identity only) | not timed (identity only) | not timed (identity only) | not timed (identity only) | not-dispatched(cells<=16384) |
| int8i32.v1.flat | not timed (identity only) | not timed (identity only) | not timed (identity only) | not timed (identity only) | not-dispatched(the-column-has-the-unit) |
| int8i32.v1.mma | not timed (identity only) | not timed (identity only) | not timed (identity only) | not timed (identity only) | dispatched |
| int8i32.v1.applechunk | not run | | | | the column does not have the unit |
| convert.int8.quantize.a | not timed (identity only) | not timed (identity only) | not timed (identity only) | not timed (identity only) | per-call |
| convert.int8.pack.b | not timed (identity only) | not timed (identity only) | not timed (identity only) | not timed (identity only) | once-per-weight |
| convert.bf16.pack.b | not timed (identity only) | not timed (identity only) | not timed (identity only) | not timed (identity only) | once-per-weight |
| convert.bf16.widen.b | not timed (identity only) | not timed (identity only) | not timed (identity only) | not timed (identity only) | per-call-when-materialized |
| convert.int8.dequantize.b | not timed (identity only) | not timed (identity only) | not timed (identity only) | not timed (identity only) | per-call-when-materialized |

### llama8b.mlp_up.t8: m=8 n=14336 k=4096 (FULL)

| arm | median ms | min ms | rate | arm over fp32.v1 | note |
|---|---:|---:|---:|---:|---|
| fp32.v1 | not timed (identity only) | not timed (identity only) | not timed (identity only) | not timed (identity only) | plan=14 |
| bf16f32.v1.fused | not timed (identity only) | not timed (identity only) | not timed (identity only) | not timed (identity only) | not-dispatched(cells>16384) |
| bf16f32.v1.widen | not timed (identity only) | not timed (identity only) | not timed (identity only) | not timed (identity only) | dispatched |
| int8i32.v1.flat | not timed (identity only) | not timed (identity only) | not timed (identity only) | not timed (identity only) | not-dispatched(the-column-has-the-unit) |
| int8i32.v1.mma | not timed (identity only) | not timed (identity only) | not timed (identity only) | not timed (identity only) | dispatched |
| int8i32.v1.applechunk | not run | | | | the column does not have the unit |
| convert.int8.quantize.a | not timed (identity only) | not timed (identity only) | not timed (identity only) | not timed (identity only) | per-call |
| convert.int8.pack.b | not timed (identity only) | not timed (identity only) | not timed (identity only) | not timed (identity only) | once-per-weight |
| convert.bf16.pack.b | not timed (identity only) | not timed (identity only) | not timed (identity only) | not timed (identity only) | once-per-weight |
| convert.bf16.widen.b | not timed (identity only) | not timed (identity only) | not timed (identity only) | not timed (identity only) | per-call-when-materialized |
| convert.int8.dequantize.b | not timed (identity only) | not timed (identity only) | not timed (identity only) | not timed (identity only) | per-call-when-materialized |

### llama8b.mlp_up.t512: m=512 n=14336 k=4096 (FULL)

| arm | median ms | min ms | rate | arm over fp32.v1 | note |
|---|---:|---:|---:|---:|---|
| fp32.v1 | not timed (identity only) | not timed (identity only) | not timed (identity only) | not timed (identity only) | plan=10 |
| bf16f32.v1.fused | not timed (identity only) | not timed (identity only) | not timed (identity only) | not timed (identity only) | not-dispatched(cells>16384) |
| bf16f32.v1.widen | not timed (identity only) | not timed (identity only) | not timed (identity only) | not timed (identity only) | dispatched |
| int8i32.v1.flat | not timed (identity only) | not timed (identity only) | not timed (identity only) | not timed (identity only) | not-dispatched(the-column-has-the-unit) |
| int8i32.v1.mma | not timed (identity only) | not timed (identity only) | not timed (identity only) | not timed (identity only) | dispatched |
| int8i32.v1.applechunk | not run | | | | the column does not have the unit |
| convert.int8.quantize.a | not timed (identity only) | not timed (identity only) | not timed (identity only) | not timed (identity only) | per-call |
| convert.int8.pack.b | not timed (identity only) | not timed (identity only) | not timed (identity only) | not timed (identity only) | once-per-weight |
| convert.bf16.pack.b | not timed (identity only) | not timed (identity only) | not timed (identity only) | not timed (identity only) | once-per-weight |
| convert.bf16.widen.b | not timed (identity only) | not timed (identity only) | not timed (identity only) | not timed (identity only) | per-call-when-materialized |
| convert.int8.dequantize.b | not timed (identity only) | not timed (identity only) | not timed (identity only) | not timed (identity only) | per-call-when-materialized |

### llama8b.mlp_down.t1: m=1 n=4096 k=14336 (FULL)

| arm | median ms | min ms | rate | arm over fp32.v1 | note |
|---|---:|---:|---:|---:|---|
| fp32.v1 | not timed (identity only) | not timed (identity only) | not timed (identity only) | not timed (identity only) | plan=13 |
| bf16f32.v1.fused | not timed (identity only) | not timed (identity only) | not timed (identity only) | not timed (identity only) | dispatched |
| bf16f32.v1.widen | not timed (identity only) | not timed (identity only) | not timed (identity only) | not timed (identity only) | not-dispatched(cells<=16384) |
| int8i32.v1.flat | not timed (identity only) | not timed (identity only) | not timed (identity only) | not timed (identity only) | not-dispatched(the-column-has-the-unit) |
| int8i32.v1.mma | not timed (identity only) | not timed (identity only) | not timed (identity only) | not timed (identity only) | dispatched |
| int8i32.v1.applechunk | not run | | | | the column does not have the unit |
| convert.int8.quantize.a | not timed (identity only) | not timed (identity only) | not timed (identity only) | not timed (identity only) | per-call |
| convert.int8.pack.b | not timed (identity only) | not timed (identity only) | not timed (identity only) | not timed (identity only) | once-per-weight |
| convert.bf16.pack.b | not timed (identity only) | not timed (identity only) | not timed (identity only) | not timed (identity only) | once-per-weight |
| convert.bf16.widen.b | not timed (identity only) | not timed (identity only) | not timed (identity only) | not timed (identity only) | per-call-when-materialized |
| convert.int8.dequantize.b | not timed (identity only) | not timed (identity only) | not timed (identity only) | not timed (identity only) | per-call-when-materialized |

### llama8b.mlp_down.t8: m=8 n=4096 k=14336 (FULL)

| arm | median ms | min ms | rate | arm over fp32.v1 | note |
|---|---:|---:|---:|---:|---|
| fp32.v1 | not timed (identity only) | not timed (identity only) | not timed (identity only) | not timed (identity only) | plan=14 |
| bf16f32.v1.fused | not timed (identity only) | not timed (identity only) | not timed (identity only) | not timed (identity only) | not-dispatched(cells>16384) |
| bf16f32.v1.widen | not timed (identity only) | not timed (identity only) | not timed (identity only) | not timed (identity only) | dispatched |
| int8i32.v1.flat | not timed (identity only) | not timed (identity only) | not timed (identity only) | not timed (identity only) | not-dispatched(the-column-has-the-unit) |
| int8i32.v1.mma | not timed (identity only) | not timed (identity only) | not timed (identity only) | not timed (identity only) | dispatched |
| int8i32.v1.applechunk | not run | | | | the column does not have the unit |
| convert.int8.quantize.a | not timed (identity only) | not timed (identity only) | not timed (identity only) | not timed (identity only) | per-call |
| convert.int8.pack.b | not timed (identity only) | not timed (identity only) | not timed (identity only) | not timed (identity only) | once-per-weight |
| convert.bf16.pack.b | not timed (identity only) | not timed (identity only) | not timed (identity only) | not timed (identity only) | once-per-weight |
| convert.bf16.widen.b | not timed (identity only) | not timed (identity only) | not timed (identity only) | not timed (identity only) | per-call-when-materialized |
| convert.int8.dequantize.b | not timed (identity only) | not timed (identity only) | not timed (identity only) | not timed (identity only) | per-call-when-materialized |

### llama8b.mlp_down.t512: m=512 n=4096 k=14336 (FULL)

| arm | median ms | min ms | rate | arm over fp32.v1 | note |
|---|---:|---:|---:|---:|---|
| fp32.v1 | not timed (identity only) | not timed (identity only) | not timed (identity only) | not timed (identity only) | plan=10 |
| bf16f32.v1.fused | not timed (identity only) | not timed (identity only) | not timed (identity only) | not timed (identity only) | not-dispatched(cells>16384) |
| bf16f32.v1.widen | not timed (identity only) | not timed (identity only) | not timed (identity only) | not timed (identity only) | dispatched |
| int8i32.v1.flat | not timed (identity only) | not timed (identity only) | not timed (identity only) | not timed (identity only) | not-dispatched(the-column-has-the-unit) |
| int8i32.v1.mma | not timed (identity only) | not timed (identity only) | not timed (identity only) | not timed (identity only) | dispatched |
| int8i32.v1.applechunk | not run | | | | the column does not have the unit |
| convert.int8.quantize.a | not timed (identity only) | not timed (identity only) | not timed (identity only) | not timed (identity only) | per-call |
| convert.int8.pack.b | not timed (identity only) | not timed (identity only) | not timed (identity only) | not timed (identity only) | once-per-weight |
| convert.bf16.pack.b | not timed (identity only) | not timed (identity only) | not timed (identity only) | not timed (identity only) | once-per-weight |
| convert.bf16.widen.b | not timed (identity only) | not timed (identity only) | not timed (identity only) | not timed (identity only) | per-call-when-materialized |
| convert.int8.dequantize.b | not timed (identity only) | not timed (identity only) | not timed (identity only) | not timed (identity only) | per-call-when-materialized |

### llama8b.lm_head.t1: m=1 n=128256 k=4096 (FULL)

| arm | median ms | min ms | rate | arm over fp32.v1 | note |
|---|---:|---:|---:|---:|---|
| fp32.v1 | not timed (identity only) | not timed (identity only) | not timed (identity only) | not timed (identity only) | plan=13 |
| bf16f32.v1.fused | not timed (identity only) | not timed (identity only) | not timed (identity only) | not timed (identity only) | not-dispatched(cells>16384) |
| bf16f32.v1.widen | not timed (identity only) | not timed (identity only) | not timed (identity only) | not timed (identity only) | dispatched |
| int8i32.v1.flat | not timed (identity only) | not timed (identity only) | not timed (identity only) | not timed (identity only) | not-dispatched(the-column-has-the-unit) |
| int8i32.v1.mma | not timed (identity only) | not timed (identity only) | not timed (identity only) | not timed (identity only) | dispatched |
| int8i32.v1.applechunk | not run | | | | the column does not have the unit |
| convert.int8.quantize.a | not timed (identity only) | not timed (identity only) | not timed (identity only) | not timed (identity only) | per-call |
| convert.int8.pack.b | not timed (identity only) | not timed (identity only) | not timed (identity only) | not timed (identity only) | once-per-weight |
| convert.bf16.pack.b | not timed (identity only) | not timed (identity only) | not timed (identity only) | not timed (identity only) | once-per-weight |
| convert.bf16.widen.b | not timed (identity only) | not timed (identity only) | not timed (identity only) | not timed (identity only) | per-call-when-materialized |
| convert.int8.dequantize.b | not timed (identity only) | not timed (identity only) | not timed (identity only) | not timed (identity only) | per-call-when-materialized |

### llama8b.lm_head.t8: m=8 n=128256 k=4096 (FULL)

| arm | median ms | min ms | rate | arm over fp32.v1 | note |
|---|---:|---:|---:|---:|---|
| fp32.v1 | not timed (identity only) | not timed (identity only) | not timed (identity only) | not timed (identity only) | plan=14 |
| bf16f32.v1.fused | not timed (identity only) | not timed (identity only) | not timed (identity only) | not timed (identity only) | not-dispatched(cells>16384) |
| bf16f32.v1.widen | not timed (identity only) | not timed (identity only) | not timed (identity only) | not timed (identity only) | dispatched |
| int8i32.v1.flat | not timed (identity only) | not timed (identity only) | not timed (identity only) | not timed (identity only) | not-dispatched(the-column-has-the-unit) |
| int8i32.v1.mma | not timed (identity only) | not timed (identity only) | not timed (identity only) | not timed (identity only) | dispatched |
| int8i32.v1.applechunk | not run | | | | the column does not have the unit |
| convert.int8.quantize.a | not timed (identity only) | not timed (identity only) | not timed (identity only) | not timed (identity only) | per-call |
| convert.int8.pack.b | not timed (identity only) | not timed (identity only) | not timed (identity only) | not timed (identity only) | once-per-weight |
| convert.bf16.pack.b | not timed (identity only) | not timed (identity only) | not timed (identity only) | not timed (identity only) | once-per-weight |
| convert.bf16.widen.b | not timed (identity only) | not timed (identity only) | not timed (identity only) | not timed (identity only) | per-call-when-materialized |
| convert.int8.dequantize.b | not timed (identity only) | not timed (identity only) | not timed (identity only) | not timed (identity only) | per-call-when-materialized |

### llama8b.lm_head.t512: m=512 n=16032 k=4096 (CAPPED)

| arm | median ms | min ms | rate | arm over fp32.v1 | note |
|---|---:|---:|---:|---:|---|
| fp32.v1 | not timed (identity only) | not timed (identity only) | not timed (identity only) | not timed (identity only) | plan=10 |
| bf16f32.v1.fused | not timed (identity only) | not timed (identity only) | not timed (identity only) | not timed (identity only) | not-dispatched(cells>16384) |
| bf16f32.v1.widen | not timed (identity only) | not timed (identity only) | not timed (identity only) | not timed (identity only) | dispatched |
| int8i32.v1.flat | not timed (identity only) | not timed (identity only) | not timed (identity only) | not timed (identity only) | not-dispatched(the-column-has-the-unit) |
| int8i32.v1.mma | not timed (identity only) | not timed (identity only) | not timed (identity only) | not timed (identity only) | dispatched |
| int8i32.v1.applechunk | not run | | | | the column does not have the unit |
| convert.int8.quantize.a | not timed (identity only) | not timed (identity only) | not timed (identity only) | not timed (identity only) | per-call |
| convert.int8.pack.b | not timed (identity only) | not timed (identity only) | not timed (identity only) | not timed (identity only) | once-per-weight |
| convert.bf16.pack.b | not timed (identity only) | not timed (identity only) | not timed (identity only) | not timed (identity only) | once-per-weight |
| convert.bf16.widen.b | not timed (identity only) | not timed (identity only) | not timed (identity only) | not timed (identity only) | per-call-when-materialized |
| convert.int8.dequantize.b | not timed (identity only) | not timed (identity only) | not timed (identity only) | not timed (identity only) | per-call-when-materialized |

## m2pro (column apple)

Vendor comparison: MPS/MPSGraph on Apple GPU, torch 2.13.0, torch 2.13.0, median of 10. COMPARISON ONLY, no digest, no identity claim.

### llama8b.qkv.t1: m=1 n=4096 k=4096 (FULL)

| arm | median ms | min ms | rate | arm over fp32.v1 | note |
|---|---:|---:|---:|---:|---|
| fp32.v1 | 2.0650 | 1.5250 | 8.1246 GMAC/s | 1.000 | plan=0 |
| bf16f32.v1.fused | 1.8670 | 1.4320 | 8.9862 GMAC/s | 0.904 | dispatched |
| bf16f32.v1.widen | 2.6480 | 2.1760 | 6.3358 GMAC/s | 1.282 | not-dispatched(cells<=16384) |
| int8i32.v1.flat | 1.7920 | 1.3420 | 9.3623 GMAC/s | 0.868 | dispatched |
| int8i32.v1.mma | not run | | | | the column does not have the unit |
| int8i32.v1.applechunk | 1.2530 | 0.7800 | 13.3896 GMAC/s | 0.607 | probe,geometry=1 |
| convert.int8.quantize.a | 1.7970 | 1.3170 | 0.0023 Gelem/s | 0.870 | per-call |
| convert.int8.pack.b | 4.3320 | 3.8210 | 3.8729 Gelem/s | 2.098 | once-per-weight |
| convert.bf16.pack.b | 1.5580 | 1.0610 | 10.7684 Gelem/s | 0.754 | once-per-weight |
| convert.bf16.widen.b | 1.1710 | 0.6980 | 14.3273 Gelem/s | 0.567 | per-call-when-materialized |
| convert.int8.dequantize.b | 3.7150 | 3.6800 | 4.5161 Gelem/s | 1.799 | per-call-when-materialized |
| DERIVED int8i32.v1.flat + quantize.a | 3.5890 | | | 1.738 | sum of two medians |
| DERIVED int8i32.v1.applechunk + quantize.a | 3.0500 | | | 1.477 | sum of two medians |
| vendor strict (COMPARISON ONLY) | 0.9346 | 0.8752 | 17.9508 GMAC/s | | fp32.v1 over this: 2.209 |
| vendor bf16 (COMPARISON ONLY) | 0.4074 | 0.3919 | 41.1837 GMAC/s | | fp32.v1 over this: 5.069 |

### llama8b.qkv.t8: m=8 n=4096 k=4096 (FULL)

| arm | median ms | min ms | rate | arm over fp32.v1 | note |
|---|---:|---:|---:|---:|---|
| fp32.v1 | 1.5080 | 1.4970 | 89.0038 GMAC/s | 1.000 | plan=20 |
| bf16f32.v1.fused | 9.4290 | 9.4200 | 14.2346 GMAC/s | 6.253 | not-dispatched(cells>16384) |
| bf16f32.v1.widen | 2.0470 | 2.0250 | 65.5680 GMAC/s | 1.357 | dispatched |
| int8i32.v1.flat | 9.7300 | 9.6760 | 13.7942 GMAC/s | 6.452 | dispatched |
| int8i32.v1.mma | not run | | | | the column does not have the unit |
| int8i32.v1.applechunk | 0.8230 | 0.8130 | 163.0835 GMAC/s | 0.546 | probe,geometry=1 |
| convert.int8.quantize.a | 1.3800 | 1.3530 | 0.0237 Gelem/s | 0.915 | per-call |
| convert.int8.pack.b | 3.7900 | 3.7660 | 4.4267 Gelem/s | 2.513 | once-per-weight |
| convert.bf16.pack.b | 1.0570 | 1.0290 | 15.8725 Gelem/s | 0.701 | once-per-weight |
| convert.bf16.widen.b | 0.7420 | 0.7040 | 22.6108 Gelem/s | 0.492 | per-call-when-materialized |
| convert.int8.dequantize.b | 3.6940 | 3.6880 | 4.5417 Gelem/s | 2.450 | per-call-when-materialized |
| DERIVED int8i32.v1.flat + quantize.a | 11.1100 | | | 7.367 | sum of two medians |
| DERIVED int8i32.v1.applechunk + quantize.a | 2.2030 | | | 1.461 | sum of two medians |
| vendor strict (COMPARISON ONLY) | 1.9590 | 0.7800 | 68.5134 GMAC/s | | fp32.v1 over this: 0.770 |
| vendor bf16 (COMPARISON ONLY) | 1.1821 | 0.9248 | 113.5413 GMAC/s | | fp32.v1 over this: 1.276 |

### llama8b.qkv.t512: m=512 n=4096 k=4096 (FULL)

| arm | median ms | min ms | rate | arm over fp32.v1 | note |
|---|---:|---:|---:|---:|---|
| fp32.v1 | 17.8000 | 17.5710 | 482.5806 GMAC/s | 1.000 | plan=20 |
| bf16f32.v1.fused | 513.5380 | 512.9090 | 16.7270 GMAC/s | 28.850 | not-dispatched(cells>16384) |
| bf16f32.v1.widen | 18.2540 | 18.2300 | 470.5782 GMAC/s | 1.026 | dispatched |
| int8i32.v1.flat | 515.1990 | 514.8010 | 16.6730 GMAC/s | 28.944 | dispatched |
| int8i32.v1.mma | not run | | | | the column does not have the unit |
| int8i32.v1.applechunk | 8.5930 | 8.5740 | 999.6433 GMAC/s | 0.483 | probe,geometry=0 |
| convert.int8.quantize.a | 3.5060 | 3.4490 | 0.5982 Gelem/s | 0.197 | per-call |
| convert.int8.pack.b | 3.8630 | 3.7960 | 4.3431 Gelem/s | 0.217 | once-per-weight |
| convert.bf16.pack.b | 1.0750 | 1.0560 | 15.6067 Gelem/s | 0.060 | once-per-weight |
| convert.bf16.widen.b | 0.7760 | 0.7270 | 21.6201 Gelem/s | 0.044 | per-call-when-materialized |
| convert.int8.dequantize.b | 3.7230 | 3.6900 | 4.5064 Gelem/s | 0.209 | per-call-when-materialized |
| DERIVED int8i32.v1.flat + quantize.a | 518.7050 | | | 29.141 | sum of two medians |
| DERIVED int8i32.v1.applechunk + quantize.a | 12.0990 | | | 0.680 | sum of two medians |
| vendor strict (COMPARISON ONLY) | 3.6036 | 3.2415 | 2383.7340 GMAC/s | | fp32.v1 over this: 4.940 |
| vendor bf16 (COMPARISON ONLY) | 5.7360 | 5.6643 | 1497.5479 GMAC/s | | fp32.v1 over this: 3.103 |

### llama8b.mlp_up.t1: m=1 n=14336 k=4096 (FULL)

| arm | median ms | min ms | rate | arm over fp32.v1 | note |
|---|---:|---:|---:|---:|---|
| fp32.v1 | 5.5180 | 5.2780 | 10.6416 GMAC/s | 1.000 | plan=0 |
| bf16f32.v1.fused | 5.3520 | 5.2240 | 10.9716 GMAC/s | 0.970 | dispatched |
| bf16f32.v1.widen | 7.2990 | 7.1760 | 8.0450 GMAC/s | 1.323 | not-dispatched(cells<=16384) |
| int8i32.v1.flat | 5.1680 | 4.9740 | 11.3623 GMAC/s | 0.937 | dispatched |
| int8i32.v1.mma | not run | | | | the column does not have the unit |
| int8i32.v1.applechunk | 2.1880 | 2.1370 | 26.8374 GMAC/s | 0.397 | probe,geometry=1 |
| convert.int8.quantize.a | 1.4890 | 1.3160 | 0.0028 Gelem/s | 0.270 | per-call |
| convert.int8.pack.b | 13.8330 | 13.6930 | 4.2449 Gelem/s | 2.507 | once-per-weight |
| convert.bf16.pack.b | 3.2810 | 3.1100 | 17.8971 Gelem/s | 0.595 | once-per-weight |
| convert.bf16.widen.b | 2.2730 | 2.1650 | 25.8338 Gelem/s | 0.412 | per-call-when-materialized |
| convert.int8.dequantize.b | 14.1760 | 14.0850 | 4.1422 Gelem/s | 2.569 | per-call-when-materialized |
| DERIVED int8i32.v1.flat + quantize.a | 6.6570 | | | 1.206 | sum of two medians |
| DERIVED int8i32.v1.applechunk + quantize.a | 3.6770 | | | 0.666 | sum of two medians |
| vendor strict (COMPARISON ONLY) | 1.9838 | 1.5546 | 29.6000 GMAC/s | | fp32.v1 over this: 2.782 |
| vendor bf16 (COMPARISON ONLY) | 0.8888 | 0.8740 | 66.0691 GMAC/s | | fp32.v1 over this: 6.209 |

### llama8b.mlp_up.t8: m=8 n=14336 k=4096 (FULL)

| arm | median ms | min ms | rate | arm over fp32.v1 | note |
|---|---:|---:|---:|---:|---|
| fp32.v1 | 4.1090 | 4.0970 | 114.3252 GMAC/s | 1.000 | plan=20 |
| bf16f32.v1.fused | 30.4230 | 29.7830 | 15.4410 GMAC/s | 7.404 | not-dispatched(cells>16384) |
| bf16f32.v1.widen | 5.9930 | 5.9530 | 78.3851 GMAC/s | 1.459 | dispatched |
| int8i32.v1.flat | 29.4200 | 29.3770 | 15.9674 GMAC/s | 7.160 | dispatched |
| int8i32.v1.mma | not run | | | | the column does not have the unit |
| int8i32.v1.applechunk | 2.0040 | 1.9960 | 234.4122 GMAC/s | 0.488 | probe,geometry=1 |
| convert.int8.quantize.a | 1.3640 | 1.3300 | 0.0240 Gelem/s | 0.332 | per-call |
| convert.int8.pack.b | 13.6580 | 13.6470 | 4.2993 Gelem/s | 3.324 | once-per-weight |
| convert.bf16.pack.b | 3.1010 | 3.0680 | 18.9359 Gelem/s | 0.755 | once-per-weight |
| convert.bf16.widen.b | 2.1100 | 2.1070 | 27.8295 Gelem/s | 0.514 | per-call-when-materialized |
| convert.int8.dequantize.b | 14.0080 | 13.9950 | 4.1919 Gelem/s | 3.409 | per-call-when-materialized |
| DERIVED int8i32.v1.flat + quantize.a | 30.7840 | | | 7.492 | sum of two medians |
| DERIVED int8i32.v1.applechunk + quantize.a | 3.3680 | | | 0.820 | sum of two medians |
| vendor strict (COMPARISON ONLY) | 2.0834 | 2.0551 | 225.4835 GMAC/s | | fp32.v1 over this: 1.972 |
| vendor bf16 (COMPARISON ONLY) | 1.5344 | 1.3872 | 306.1461 GMAC/s | | fp32.v1 over this: 2.678 |

### llama8b.mlp_up.t512: m=512 n=14336 k=4096 (FULL)

| arm | median ms | min ms | rate | arm over fp32.v1 | note |
|---|---:|---:|---:|---:|---|
| fp32.v1 | 60.4090 | 60.2990 | 497.6870 GMAC/s | 1.000 | plan=20 |
| bf16f32.v1.fused | 1793.9800 | 1792.4270 | 16.7587 GMAC/s | 29.697 | not-dispatched(cells>16384) |
| bf16f32.v1.widen | 62.3960 | 62.3070 | 481.8381 GMAC/s | 1.033 | dispatched |
| int8i32.v1.flat | 1799.2350 | 1798.4680 | 16.7098 GMAC/s | 29.784 | dispatched |
| int8i32.v1.mma | not run | | | | the column does not have the unit |
| int8i32.v1.applechunk | 27.9870 | 27.9530 | 1074.2406 GMAC/s | 0.463 | probe,geometry=0 |
| convert.int8.quantize.a | 3.4980 | 3.4830 | 0.5995 Gelem/s | 0.058 | per-call |
| convert.int8.pack.b | 13.7180 | 13.7010 | 4.2805 Gelem/s | 0.227 | once-per-weight |
| convert.bf16.pack.b | 3.1230 | 3.1150 | 18.8025 Gelem/s | 0.052 | once-per-weight |
| convert.bf16.widen.b | 2.1800 | 2.1050 | 26.9359 Gelem/s | 0.036 | per-call-when-materialized |
| convert.int8.dequantize.b | 14.0420 | 14.0140 | 4.1818 Gelem/s | 0.232 | per-call-when-materialized |
| DERIVED int8i32.v1.flat + quantize.a | 1802.7330 | | | 29.842 | sum of two medians |
| DERIVED int8i32.v1.applechunk + quantize.a | 31.4850 | | | 0.521 | sum of two medians |
| vendor strict (COMPARISON ONLY) | 10.4036 | 10.3260 | 2889.8418 GMAC/s | | fp32.v1 over this: 5.807 |
| vendor bf16 (COMPARISON ONLY) | 18.2992 | 18.2628 | 1642.9547 GMAC/s | | fp32.v1 over this: 3.301 |

### llama8b.mlp_down.t1: m=1 n=4096 k=14336 (FULL)

| arm | median ms | min ms | rate | arm over fp32.v1 | note |
|---|---:|---:|---:|---:|---|
| fp32.v1 | 5.5890 | 5.5620 | 10.5064 GMAC/s | 1.000 | plan=0 |
| bf16f32.v1.fused | 5.1340 | 5.1220 | 11.4375 GMAC/s | 0.919 | dispatched |
| bf16f32.v1.widen | 7.4730 | 7.4460 | 7.8577 GMAC/s | 1.337 | not-dispatched(cells<=16384) |
| int8i32.v1.flat | 4.4290 | 4.4220 | 13.2581 GMAC/s | 0.792 | dispatched |
| int8i32.v1.mma | not run | | | | the column does not have the unit |
| int8i32.v1.applechunk | 2.4550 | 2.4120 | 23.9186 GMAC/s | 0.439 | probe,geometry=1 |
| convert.int8.quantize.a | 4.2670 | 4.2250 | 0.0034 Gelem/s | 0.763 | per-call |
| convert.int8.pack.b | 13.8610 | 13.8420 | 4.2364 Gelem/s | 2.480 | once-per-weight |
| convert.bf16.pack.b | 2.9830 | 2.9750 | 19.6850 Gelem/s | 0.534 | once-per-weight |
| convert.bf16.widen.b | 2.1490 | 2.1300 | 27.3245 Gelem/s | 0.385 | per-call-when-materialized |
| convert.int8.dequantize.b | 13.2340 | 13.2070 | 4.4371 Gelem/s | 2.368 | per-call-when-materialized |
| DERIVED int8i32.v1.flat + quantize.a | 8.6960 | | | 1.556 | sum of two medians |
| DERIVED int8i32.v1.applechunk + quantize.a | 6.7220 | | | 1.203 | sum of two medians |
| vendor strict (COMPARISON ONLY) | 1.8613 | 1.4713 | 31.5485 GMAC/s | | fp32.v1 over this: 3.003 |
| vendor bf16 (COMPARISON ONLY) | 1.2683 | 0.8672 | 46.2995 GMAC/s | | fp32.v1 over this: 4.407 |

### llama8b.mlp_down.t8: m=8 n=4096 k=14336 (FULL)

| arm | median ms | min ms | rate | arm over fp32.v1 | note |
|---|---:|---:|---:|---:|---|
| fp32.v1 | 4.9750 | 4.9720 | 94.4245 GMAC/s | 1.000 | plan=20 |
| bf16f32.v1.fused | 34.4620 | 34.3610 | 13.6313 GMAC/s | 6.927 | not-dispatched(cells>16384) |
| bf16f32.v1.widen | 6.8790 | 6.8670 | 68.2893 GMAC/s | 1.383 | dispatched |
| int8i32.v1.flat | 33.9330 | 33.9080 | 13.8438 GMAC/s | 6.821 | dispatched |
| int8i32.v1.mma | not run | | | | the column does not have the unit |
| int8i32.v1.applechunk | 2.4370 | 2.4170 | 192.7624 GMAC/s | 0.490 | probe,geometry=1 |
| convert.int8.quantize.a | 4.4030 | 4.3990 | 0.0260 Gelem/s | 0.885 | per-call |
| convert.int8.pack.b | 13.9820 | 13.9250 | 4.1997 Gelem/s | 2.810 | once-per-weight |
| convert.bf16.pack.b | 2.9870 | 2.9860 | 19.6586 Gelem/s | 0.600 | once-per-weight |
| convert.bf16.widen.b | 2.1080 | 2.0960 | 27.8559 Gelem/s | 0.424 | per-call-when-materialized |
| convert.int8.dequantize.b | 13.1830 | 13.1620 | 4.4542 Gelem/s | 2.650 | per-call-when-materialized |
| DERIVED int8i32.v1.flat + quantize.a | 38.3360 | | | 7.706 | sum of two medians |
| DERIVED int8i32.v1.applechunk + quantize.a | 6.8400 | | | 1.375 | sum of two medians |
| vendor strict (COMPARISON ONLY) | 2.6761 | 2.1763 | 175.5409 GMAC/s | | fp32.v1 over this: 1.859 |
| vendor bf16 (COMPARISON ONLY) | 1.5571 | 1.1390 | 301.6976 GMAC/s | | fp32.v1 over this: 3.195 |

### llama8b.mlp_down.t512: m=512 n=4096 k=14336 (FULL)

| arm | median ms | min ms | rate | arm over fp32.v1 | note |
|---|---:|---:|---:|---:|---|
| fp32.v1 | 65.7240 | 65.3940 | 457.4398 GMAC/s | 1.000 | plan=20 |
| bf16f32.v1.fused | 1935.9580 | 1933.1860 | 15.5297 GMAC/s | 29.456 | not-dispatched(cells>16384) |
| bf16f32.v1.widen | 66.8690 | 66.7580 | 449.6070 GMAC/s | 1.017 | dispatched |
| int8i32.v1.flat | 1828.9050 | 1828.7460 | 16.4387 GMAC/s | 27.827 | dispatched |
| int8i32.v1.mma | not run | | | | the column does not have the unit |
| int8i32.v1.applechunk | 29.9820 | 29.7930 | 1002.7607 GMAC/s | 0.456 | probe,geometry=0 |
| convert.int8.quantize.a | 12.0160 | 11.9830 | 0.6109 Gelem/s | 0.183 | per-call |
| convert.int8.pack.b | 13.8880 | 13.8010 | 4.2281 Gelem/s | 0.211 | once-per-weight |
| convert.bf16.pack.b | 3.0210 | 2.9630 | 19.4374 Gelem/s | 0.046 | once-per-weight |
| convert.bf16.widen.b | 2.1960 | 2.1290 | 26.7396 Gelem/s | 0.033 | per-call-when-materialized |
| convert.int8.dequantize.b | 13.2060 | 13.1960 | 4.4465 Gelem/s | 0.201 | per-call-when-materialized |
| DERIVED int8i32.v1.flat + quantize.a | 1840.9210 | | | 28.010 | sum of two medians |
| DERIVED int8i32.v1.applechunk + quantize.a | 41.9980 | | | 0.639 | sum of two medians |
| vendor strict (COMPARISON ONLY) | 11.1146 | 10.9513 | 2704.9791 GMAC/s | | fp32.v1 over this: 5.913 |
| vendor bf16 (COMPARISON ONLY) | 19.2347 | 19.2038 | 1563.0513 GMAC/s | | fp32.v1 over this: 3.417 |

### llama8b.lm_head.t1: m=1 n=128256 k=4096 (FULL)

| arm | median ms | min ms | rate | arm over fp32.v1 | note |
|---|---:|---:|---:|---:|---|
| fp32.v1 | 33.1620 | 33.0460 | 15.8415 GMAC/s | 1.000 | plan=0 |
| bf16f32.v1.fused | 32.8990 | 32.5840 | 15.9682 GMAC/s | 0.992 | not-dispatched(cells>16384) |
| bf16f32.v1.widen | 50.1550 | 50.0170 | 10.4743 GMAC/s | 1.512 | dispatched |
| int8i32.v1.flat | 32.6210 | 32.5470 | 16.1042 GMAC/s | 0.984 | dispatched |
| int8i32.v1.mma | not run | | | | the column does not have the unit |
| int8i32.v1.applechunk | 11.9000 | 11.8300 | 44.1459 GMAC/s | 0.359 | probe,geometry=1 |
| convert.int8.quantize.a | 1.3700 | 1.3540 | 0.0030 Gelem/s | 0.041 | per-call |
| convert.int8.pack.b | 96.3670 | 96.0540 | 5.4514 Gelem/s | 2.906 | once-per-weight |
| convert.bf16.pack.b | 24.8420 | 24.8330 | 21.1471 Gelem/s | 0.749 | once-per-weight |
| convert.bf16.widen.b | 17.2820 | 17.2720 | 30.3979 Gelem/s | 0.521 | per-call-when-materialized |
| convert.int8.dequantize.b | 147.0130 | 147.0070 | 3.5734 Gelem/s | 4.433 | per-call-when-materialized |
| DERIVED int8i32.v1.flat + quantize.a | 33.9910 | | | 1.025 | sum of two medians |
| DERIVED int8i32.v1.applechunk + quantize.a | 13.2700 | | | 0.400 | sum of two medians |
| vendor strict (COMPARISON ONLY) | 12.8482 | 12.8187 | 40.8879 GMAC/s | | fp32.v1 over this: 2.581 |
| vendor bf16 (COMPARISON ONLY) | 6.1005 | 6.0663 | 86.1134 GMAC/s | | fp32.v1 over this: 5.436 |

### llama8b.lm_head.t8: m=8 n=128256 k=4096 (FULL)

| arm | median ms | min ms | rate | arm over fp32.v1 | note |
|---|---:|---:|---:|---:|---|
| fp32.v1 | 33.2310 | 33.1870 | 126.4690 GMAC/s | 1.000 | plan=20 |
| bf16f32.v1.fused | 252.1600 | 250.8770 | 16.6668 GMAC/s | 7.588 | not-dispatched(cells>16384) |
| bf16f32.v1.widen | 50.3410 | 50.2410 | 83.4845 GMAC/s | 1.515 | dispatched |
| int8i32.v1.flat | 252.5700 | 251.8790 | 16.6397 GMAC/s | 7.600 | dispatched |
| int8i32.v1.mma | not run | | | | the column does not have the unit |
| int8i32.v1.applechunk | 11.8680 | 11.8500 | 354.1197 GMAC/s | 0.357 | probe,geometry=1 |
| convert.int8.quantize.a | 1.3730 | 1.3670 | 0.0239 Gelem/s | 0.041 | per-call |
| convert.int8.pack.b | 96.3660 | 96.2370 | 5.4515 Gelem/s | 2.900 | once-per-weight |
| convert.bf16.pack.b | 24.8690 | 24.8300 | 21.1242 Gelem/s | 0.748 | once-per-weight |
| convert.bf16.widen.b | 17.3730 | 17.3160 | 30.2387 Gelem/s | 0.523 | per-call-when-materialized |
| convert.int8.dequantize.b | 146.9730 | 146.9500 | 3.5744 Gelem/s | 4.423 | per-call-when-materialized |
| DERIVED int8i32.v1.flat + quantize.a | 253.9430 | | | 7.642 | sum of two medians |
| DERIVED int8i32.v1.applechunk + quantize.a | 13.2410 | | | 0.398 | sum of two medians |
| vendor strict (COMPARISON ONLY) | 17.0554 | 17.0454 | 246.4137 GMAC/s | | fp32.v1 over this: 1.948 |
| vendor bf16 (COMPARISON ONLY) | 8.3481 | 8.3207 | 503.4321 GMAC/s | | fp32.v1 over this: 3.981 |

### llama8b.lm_head.t512: m=512 n=16032 k=4096 (CAPPED)

| arm | median ms | min ms | rate | arm over fp32.v1 | note |
|---|---:|---:|---:|---:|---|
| fp32.v1 | 67.9340 | 67.7210 | 494.9148 GMAC/s | 1.000 | plan=20 |
| bf16f32.v1.fused | 2009.9180 | 2008.9810 | 16.7278 GMAC/s | 29.586 | not-dispatched(cells>16384) |
| bf16f32.v1.widen | 69.7870 | 69.6610 | 481.7737 GMAC/s | 1.027 | dispatched |
| int8i32.v1.flat | 2011.2770 | 2011.0160 | 16.7165 GMAC/s | 29.606 | dispatched |
| int8i32.v1.mma | not run | | | | the column does not have the unit |
| int8i32.v1.applechunk | 31.3130 | 31.1150 | 1073.7247 GMAC/s | 0.461 | probe,geometry=0 |
| convert.int8.quantize.a | 3.5000 | 3.4920 | 0.5992 Gelem/s | 0.052 | per-call |
| convert.int8.pack.b | 13.8740 | 13.8560 | 4.7331 Gelem/s | 0.204 | once-per-weight |
| convert.bf16.pack.b | 3.4510 | 3.4080 | 19.0284 Gelem/s | 0.051 | once-per-weight |
| convert.bf16.widen.b | 2.3670 | 2.3360 | 27.7427 Gelem/s | 0.035 | per-call-when-materialized |
| convert.int8.dequantize.b | 15.7850 | 15.7600 | 4.1601 Gelem/s | 0.232 | per-call-when-materialized |
| DERIVED int8i32.v1.flat + quantize.a | 2014.7770 | | | 29.658 | sum of two medians |
| DERIVED int8i32.v1.applechunk + quantize.a | 34.8130 | | | 0.512 | sum of two medians |
| vendor strict (COMPARISON ONLY) | 122.5769 | 122.5294 | | n/a | the vendor ran the whole row (n=128256), ours ran capped: no ratio |
| vendor bf16 (COMPARISON ONLY) | 199.8692 | 199.7611 | | n/a | the vendor ran the whole row (n=128256), ours ran capped: no ratio |

## Where a low-bit plan costs more time than fp32.v1

| box | shape | arm | arm ms | fp32.v1 ms | arm over fp32.v1 | note |
|---|---|---|---:|---:|---:|---|
| h100 | llama8b.qkv.t1 | bf16f32.v1.fused | 0.7907 | 0.0707 | 11.184 | dispatched |
| h100 | llama8b.qkv.t1 | bf16f32.v1.widen | 0.1209 | 0.0707 | 1.710 | not-dispatched(cells<=16384) |
| h100 | llama8b.qkv.t1 | int8i32.v1.flat | 0.6653 | 0.0707 | 9.410 | not-dispatched(the-column-has-the-unit) |
| h100 | llama8b.qkv.t1 | int8i32.v1.mma | 0.1573 | 0.0707 | 2.225 | dispatched |
| h100 | llama8b.qkv.t1 | int8i32.v1.flat + quantize.a (DERIVED) | 1.1555 | 0.0707 | 16.344 | not-dispatched(the-column-has-the-unit) |
| h100 | llama8b.qkv.t1 | int8i32.v1.mma + quantize.a (DERIVED) | 0.6475 | 0.0707 | 9.158 | dispatched |
| h100 | llama8b.qkv.t8 | bf16f32.v1.fused | 0.7813 | 0.1008 | 7.751 | not-dispatched(cells>16384) |
| h100 | llama8b.qkv.t8 | bf16f32.v1.widen | 0.1505 | 0.1008 | 1.493 | dispatched |
| h100 | llama8b.qkv.t8 | int8i32.v1.flat | 0.6647 | 0.1008 | 6.594 | not-dispatched(the-column-has-the-unit) |
| h100 | llama8b.qkv.t8 | int8i32.v1.mma | 0.1478 | 0.1008 | 1.466 | dispatched |
| h100 | llama8b.qkv.t8 | int8i32.v1.flat + quantize.a (DERIVED) | 1.2637 | 0.1008 | 12.537 | not-dispatched(the-column-has-the-unit) |
| h100 | llama8b.qkv.t8 | int8i32.v1.mma + quantize.a (DERIVED) | 0.7468 | 0.1008 | 7.409 | dispatched |
| h100 | llama8b.qkv.t512 | bf16f32.v1.fused | 39.4281 | 1.0328 | 38.176 | not-dispatched(cells>16384) |
| h100 | llama8b.qkv.t512 | bf16f32.v1.widen | 1.0903 | 1.0328 | 1.056 | dispatched |
| h100 | llama8b.qkv.t512 | int8i32.v1.flat | 39.1874 | 1.0328 | 37.943 | not-dispatched(the-column-has-the-unit) |
| h100 | llama8b.qkv.t512 | int8i32.v1.mma | 1.4361 | 1.0328 | 1.390 | dispatched |
| h100 | llama8b.qkv.t512 | int8i32.v1.flat + quantize.a (DERIVED) | 41.8181 | 1.0328 | 40.490 | not-dispatched(the-column-has-the-unit) |
| h100 | llama8b.qkv.t512 | int8i32.v1.mma + quantize.a (DERIVED) | 4.0668 | 1.0328 | 3.938 | dispatched |
| h100 | llama8b.mlp_up.t1 | bf16f32.v1.fused | 0.7996 | 0.1904 | 4.200 | dispatched |
| h100 | llama8b.mlp_up.t1 | bf16f32.v1.widen | 0.3681 | 0.1904 | 1.933 | not-dispatched(cells<=16384) |
| h100 | llama8b.mlp_up.t1 | int8i32.v1.flat | 0.6655 | 0.1904 | 3.495 | not-dispatched(the-column-has-the-unit) |
| h100 | llama8b.mlp_up.t1 | int8i32.v1.mma | 0.1933 | 0.1904 | 1.015 | dispatched |
| h100 | llama8b.mlp_up.t1 | int8i32.v1.flat + quantize.a (DERIVED) | 1.1671 | 0.1904 | 6.130 | not-dispatched(the-column-has-the-unit) |
| h100 | llama8b.mlp_up.t1 | int8i32.v1.mma + quantize.a (DERIVED) | 0.6949 | 0.1904 | 3.650 | dispatched |
| h100 | llama8b.mlp_up.t8 | bf16f32.v1.fused | 2.5325 | 0.2722 | 9.304 | not-dispatched(cells>16384) |
| h100 | llama8b.mlp_up.t8 | bf16f32.v1.widen | 0.4527 | 0.2722 | 1.663 | dispatched |
| h100 | llama8b.mlp_up.t8 | int8i32.v1.flat | 2.5044 | 0.2722 | 9.201 | not-dispatched(the-column-has-the-unit) |
| h100 | llama8b.mlp_up.t8 | int8i32.v1.flat + quantize.a (DERIVED) | 3.1174 | 0.2722 | 11.453 | not-dispatched(the-column-has-the-unit) |
| h100 | llama8b.mlp_up.t8 | int8i32.v1.mma + quantize.a (DERIVED) | 0.8233 | 0.2722 | 3.025 | dispatched |
| h100 | llama8b.mlp_up.t512 | bf16f32.v1.fused | 139.4871 | 3.3374 | 41.795 | not-dispatched(cells>16384) |
| h100 | llama8b.mlp_up.t512 | bf16f32.v1.widen | 3.5206 | 3.3374 | 1.055 | dispatched |
| h100 | llama8b.mlp_up.t512 | int8i32.v1.flat | 135.6760 | 3.3374 | 40.653 | not-dispatched(the-column-has-the-unit) |
| h100 | llama8b.mlp_up.t512 | int8i32.v1.mma | 4.6300 | 3.3374 | 1.387 | dispatched |
| h100 | llama8b.mlp_up.t512 | int8i32.v1.flat + quantize.a (DERIVED) | 138.3186 | 3.3374 | 41.445 | not-dispatched(the-column-has-the-unit) |
| h100 | llama8b.mlp_up.t512 | int8i32.v1.mma + quantize.a (DERIVED) | 7.2726 | 3.3374 | 2.179 | dispatched |
| h100 | llama8b.mlp_down.t1 | bf16f32.v1.fused | 2.7585 | 0.1737 | 15.881 | dispatched |
| h100 | llama8b.mlp_down.t1 | bf16f32.v1.widen | 0.3521 | 0.1737 | 2.027 | not-dispatched(cells<=16384) |
| h100 | llama8b.mlp_down.t1 | int8i32.v1.flat | 2.3453 | 0.1737 | 13.502 | not-dispatched(the-column-has-the-unit) |
| h100 | llama8b.mlp_down.t1 | int8i32.v1.mma | 0.6498 | 0.1737 | 3.741 | dispatched |
| h100 | llama8b.mlp_down.t1 | int8i32.v1.flat + quantize.a (DERIVED) | 4.0818 | 0.1737 | 23.499 | not-dispatched(the-column-has-the-unit) |
| h100 | llama8b.mlp_down.t1 | int8i32.v1.mma + quantize.a (DERIVED) | 2.3863 | 0.1737 | 13.738 | dispatched |
| h100 | llama8b.mlp_down.t8 | bf16f32.v1.fused | 2.6961 | 0.2868 | 9.401 | not-dispatched(cells>16384) |
| h100 | llama8b.mlp_down.t8 | bf16f32.v1.widen | 0.4656 | 0.2868 | 1.623 | dispatched |
| h100 | llama8b.mlp_down.t8 | int8i32.v1.flat | 2.3383 | 0.2868 | 8.153 | not-dispatched(the-column-has-the-unit) |
| h100 | llama8b.mlp_down.t8 | int8i32.v1.mma | 0.6992 | 0.2868 | 2.438 | dispatched |
| h100 | llama8b.mlp_down.t8 | int8i32.v1.flat + quantize.a (DERIVED) | 4.7596 | 0.2868 | 16.596 | not-dispatched(the-column-has-the-unit) |
| h100 | llama8b.mlp_down.t8 | int8i32.v1.mma + quantize.a (DERIVED) | 3.1205 | 0.2868 | 10.880 | dispatched |
| h100 | llama8b.mlp_down.t512 | bf16f32.v1.fused | 136.4673 | 3.3603 | 40.612 | not-dispatched(cells>16384) |
| h100 | llama8b.mlp_down.t512 | bf16f32.v1.widen | 3.5507 | 3.3603 | 1.057 | dispatched |
| h100 | llama8b.mlp_down.t512 | int8i32.v1.flat | 137.9775 | 3.3603 | 41.061 | not-dispatched(the-column-has-the-unit) |
| h100 | llama8b.mlp_down.t512 | int8i32.v1.mma | 5.5780 | 3.3603 | 1.660 | dispatched |
| h100 | llama8b.mlp_down.t512 | int8i32.v1.flat + quantize.a (DERIVED) | 147.9594 | 3.3603 | 44.032 | not-dispatched(the-column-has-the-unit) |
| h100 | llama8b.mlp_down.t512 | int8i32.v1.mma + quantize.a (DERIVED) | 15.5599 | 3.3603 | 4.631 | dispatched |
| h100 | llama8b.lm_head.t1 | bf16f32.v1.fused | 2.5838 | 1.2066 | 2.141 | not-dispatched(cells>16384) |
| h100 | llama8b.lm_head.t1 | bf16f32.v1.widen | 2.8032 | 1.2066 | 2.323 | dispatched |
| h100 | llama8b.lm_head.t1 | int8i32.v1.flat | 2.4982 | 1.2066 | 2.070 | not-dispatched(the-column-has-the-unit) |
| h100 | llama8b.lm_head.t1 | int8i32.v1.flat + quantize.a (DERIVED) | 3.0007 | 1.2066 | 2.487 | not-dispatched(the-column-has-the-unit) |
| h100 | llama8b.lm_head.t1 | int8i32.v1.mma + quantize.a (DERIVED) | 1.4556 | 1.2066 | 1.206 | dispatched |
| h100 | llama8b.lm_head.t8 | bf16f32.v1.fused | 19.8922 | 2.1553 | 9.229 | not-dispatched(cells>16384) |
| h100 | llama8b.lm_head.t8 | bf16f32.v1.widen | 3.7610 | 2.1553 | 1.745 | dispatched |
| h100 | llama8b.lm_head.t8 | int8i32.v1.flat | 19.2560 | 2.1553 | 8.934 | not-dispatched(the-column-has-the-unit) |
| h100 | llama8b.lm_head.t8 | int8i32.v1.flat + quantize.a (DERIVED) | 19.8659 | 2.1553 | 9.217 | not-dispatched(the-column-has-the-unit) |
| h100 | llama8b.lm_head.t512 | bf16f32.v1.fused | 156.5508 | 3.7224 | 42.056 | not-dispatched(cells>16384) |
| h100 | llama8b.lm_head.t512 | bf16f32.v1.widen | 3.9206 | 3.7224 | 1.053 | dispatched |
| h100 | llama8b.lm_head.t512 | int8i32.v1.flat | 151.8644 | 3.7224 | 40.797 | not-dispatched(the-column-has-the-unit) |
| h100 | llama8b.lm_head.t512 | int8i32.v1.mma | 5.1254 | 3.7224 | 1.377 | dispatched |
| h100 | llama8b.lm_head.t512 | int8i32.v1.flat + quantize.a (DERIVED) | 154.3926 | 3.7224 | 41.477 | not-dispatched(the-column-has-the-unit) |
| h100 | llama8b.lm_head.t512 | int8i32.v1.mma + quantize.a (DERIVED) | 7.6536 | 3.7224 | 2.056 | dispatched |
| m2pro | llama8b.qkv.t1 | bf16f32.v1.widen | 2.6480 | 2.0650 | 1.282 | not-dispatched(cells<=16384) |
| m2pro | llama8b.qkv.t1 | int8i32.v1.flat + quantize.a (DERIVED) | 3.5890 | 2.0650 | 1.738 | dispatched |
| m2pro | llama8b.qkv.t1 | int8i32.v1.applechunk + quantize.a (DERIVED) | 3.0500 | 2.0650 | 1.477 | probe,geometry=1 |
| m2pro | llama8b.qkv.t8 | bf16f32.v1.fused | 9.4290 | 1.5080 | 6.253 | not-dispatched(cells>16384) |
| m2pro | llama8b.qkv.t8 | bf16f32.v1.widen | 2.0470 | 1.5080 | 1.357 | dispatched |
| m2pro | llama8b.qkv.t8 | int8i32.v1.flat | 9.7300 | 1.5080 | 6.452 | dispatched |
| m2pro | llama8b.qkv.t8 | int8i32.v1.flat + quantize.a (DERIVED) | 11.1100 | 1.5080 | 7.367 | dispatched |
| m2pro | llama8b.qkv.t8 | int8i32.v1.applechunk + quantize.a (DERIVED) | 2.2030 | 1.5080 | 1.461 | probe,geometry=1 |
| m2pro | llama8b.qkv.t512 | bf16f32.v1.fused | 513.5380 | 17.8000 | 28.850 | not-dispatched(cells>16384) |
| m2pro | llama8b.qkv.t512 | bf16f32.v1.widen | 18.2540 | 17.8000 | 1.026 | dispatched |
| m2pro | llama8b.qkv.t512 | int8i32.v1.flat | 515.1990 | 17.8000 | 28.944 | dispatched |
| m2pro | llama8b.qkv.t512 | int8i32.v1.flat + quantize.a (DERIVED) | 518.7050 | 17.8000 | 29.141 | dispatched |
| m2pro | llama8b.mlp_up.t1 | bf16f32.v1.widen | 7.2990 | 5.5180 | 1.323 | not-dispatched(cells<=16384) |
| m2pro | llama8b.mlp_up.t1 | int8i32.v1.flat + quantize.a (DERIVED) | 6.6570 | 5.5180 | 1.206 | dispatched |
| m2pro | llama8b.mlp_up.t8 | bf16f32.v1.fused | 30.4230 | 4.1090 | 7.404 | not-dispatched(cells>16384) |
| m2pro | llama8b.mlp_up.t8 | bf16f32.v1.widen | 5.9930 | 4.1090 | 1.459 | dispatched |
| m2pro | llama8b.mlp_up.t8 | int8i32.v1.flat | 29.4200 | 4.1090 | 7.160 | dispatched |
| m2pro | llama8b.mlp_up.t8 | int8i32.v1.flat + quantize.a (DERIVED) | 30.7840 | 4.1090 | 7.492 | dispatched |
| m2pro | llama8b.mlp_up.t512 | bf16f32.v1.fused | 1793.9800 | 60.4090 | 29.697 | not-dispatched(cells>16384) |
| m2pro | llama8b.mlp_up.t512 | bf16f32.v1.widen | 62.3960 | 60.4090 | 1.033 | dispatched |
| m2pro | llama8b.mlp_up.t512 | int8i32.v1.flat | 1799.2350 | 60.4090 | 29.784 | dispatched |
| m2pro | llama8b.mlp_up.t512 | int8i32.v1.flat + quantize.a (DERIVED) | 1802.7330 | 60.4090 | 29.842 | dispatched |
| m2pro | llama8b.mlp_down.t1 | bf16f32.v1.widen | 7.4730 | 5.5890 | 1.337 | not-dispatched(cells<=16384) |
| m2pro | llama8b.mlp_down.t1 | int8i32.v1.flat + quantize.a (DERIVED) | 8.6960 | 5.5890 | 1.556 | dispatched |
| m2pro | llama8b.mlp_down.t1 | int8i32.v1.applechunk + quantize.a (DERIVED) | 6.7220 | 5.5890 | 1.203 | probe,geometry=1 |
| m2pro | llama8b.mlp_down.t8 | bf16f32.v1.fused | 34.4620 | 4.9750 | 6.927 | not-dispatched(cells>16384) |
| m2pro | llama8b.mlp_down.t8 | bf16f32.v1.widen | 6.8790 | 4.9750 | 1.383 | dispatched |
| m2pro | llama8b.mlp_down.t8 | int8i32.v1.flat | 33.9330 | 4.9750 | 6.821 | dispatched |
| m2pro | llama8b.mlp_down.t8 | int8i32.v1.flat + quantize.a (DERIVED) | 38.3360 | 4.9750 | 7.706 | dispatched |
| m2pro | llama8b.mlp_down.t8 | int8i32.v1.applechunk + quantize.a (DERIVED) | 6.8400 | 4.9750 | 1.375 | probe,geometry=1 |
| m2pro | llama8b.mlp_down.t512 | bf16f32.v1.fused | 1935.9580 | 65.7240 | 29.456 | not-dispatched(cells>16384) |
| m2pro | llama8b.mlp_down.t512 | bf16f32.v1.widen | 66.8690 | 65.7240 | 1.017 | dispatched |
| m2pro | llama8b.mlp_down.t512 | int8i32.v1.flat | 1828.9050 | 65.7240 | 27.827 | dispatched |
| m2pro | llama8b.mlp_down.t512 | int8i32.v1.flat + quantize.a (DERIVED) | 1840.9210 | 65.7240 | 28.010 | dispatched |
| m2pro | llama8b.lm_head.t1 | bf16f32.v1.widen | 50.1550 | 33.1620 | 1.512 | dispatched |
| m2pro | llama8b.lm_head.t1 | int8i32.v1.flat + quantize.a (DERIVED) | 33.9910 | 33.1620 | 1.025 | dispatched |
| m2pro | llama8b.lm_head.t8 | bf16f32.v1.fused | 252.1600 | 33.2310 | 7.588 | not-dispatched(cells>16384) |
| m2pro | llama8b.lm_head.t8 | bf16f32.v1.widen | 50.3410 | 33.2310 | 1.515 | dispatched |
| m2pro | llama8b.lm_head.t8 | int8i32.v1.flat | 252.5700 | 33.2310 | 7.600 | dispatched |
| m2pro | llama8b.lm_head.t8 | int8i32.v1.flat + quantize.a (DERIVED) | 253.9430 | 33.2310 | 7.642 | dispatched |
| m2pro | llama8b.lm_head.t512 | bf16f32.v1.fused | 2009.9180 | 67.9340 | 29.586 | not-dispatched(cells>16384) |
| m2pro | llama8b.lm_head.t512 | bf16f32.v1.widen | 69.7870 | 67.9340 | 1.027 | dispatched |
| m2pro | llama8b.lm_head.t512 | int8i32.v1.flat | 2011.2770 | 67.9340 | 29.606 | dispatched |
| m2pro | llama8b.lm_head.t512 | int8i32.v1.flat + quantize.a (DERIVED) | 2014.7770 | 67.9340 | 29.658 | dispatched |

## Digests across the boxes

AGREE: every box that ran the arm at the same extents printed the same digest, and at least two did.
ONE BOX: nothing to compare, which is not a pass.

| shape | arm | verdict | h100 | mi325x | m2pro |
|---|---|---|---|---|---|
| llama8b.qkv.t1 | fp32.v1 | AGREE | 0x920cdd138650a8bd | 0x920cdd138650a8bd | 0x920cdd138650a8bd |
| llama8b.qkv.t1 | bf16f32.v1.fused | AGREE | 0xcd39e9d6e6d7179a | 0xcd39e9d6e6d7179a | 0xcd39e9d6e6d7179a |
| llama8b.qkv.t1 | bf16f32.v1.widen | AGREE | 0xcd39e9d6e6d7179a | 0xcd39e9d6e6d7179a | 0xcd39e9d6e6d7179a |
| llama8b.qkv.t1 | int8i32.v1.flat | AGREE | 0x8a9227b93e430acd | 0x8a9227b93e430acd | 0x8a9227b93e430acd |
| llama8b.qkv.t1 | int8i32.v1.mma | AGREE | 0x8a9227b93e430acd | 0x8a9227b93e430acd | not run |
| llama8b.qkv.t1 | int8i32.v1.applechunk | ONE BOX | not run | not run | 0x8a9227b93e430acd |
| llama8b.qkv.t1 | convert.int8.quantize.a | AGREE | 0x54279cbda1444726 | 0x54279cbda1444726 | 0x54279cbda1444726 |
| llama8b.qkv.t1 | convert.int8.pack.b | AGREE | 0x305b1a502296caf6 | 0x305b1a502296caf6 | 0x305b1a502296caf6 |
| llama8b.qkv.t1 | convert.bf16.pack.b | AGREE | 0x24efed638bb6b96b | 0x24efed638bb6b96b | 0x24efed638bb6b96b |
| llama8b.qkv.t1 | convert.bf16.widen.b | AGREE | 0xd2c2d96392622325 | 0xd2c2d96392622325 | 0xd2c2d96392622325 |
| llama8b.qkv.t1 | convert.int8.dequantize.b | AGREE | 0x2fd9ee536fd82325 | 0x2fd9ee536fd82325 | 0x2fd9ee536fd82325 |
| llama8b.qkv.t8 | fp32.v1 | AGREE | 0xd5677d8a13939709 | 0xd5677d8a13939709 | 0xd5677d8a13939709 |
| llama8b.qkv.t8 | bf16f32.v1.fused | AGREE | 0xf36fe1a8828cefd9 | 0xf36fe1a8828cefd9 | 0xf36fe1a8828cefd9 |
| llama8b.qkv.t8 | bf16f32.v1.widen | AGREE | 0xf36fe1a8828cefd9 | 0xf36fe1a8828cefd9 | 0xf36fe1a8828cefd9 |
| llama8b.qkv.t8 | int8i32.v1.flat | AGREE | 0x484bd20aabe66ee5 | 0x484bd20aabe66ee5 | 0x484bd20aabe66ee5 |
| llama8b.qkv.t8 | int8i32.v1.mma | AGREE | 0x484bd20aabe66ee5 | 0x484bd20aabe66ee5 | not run |
| llama8b.qkv.t8 | int8i32.v1.applechunk | ONE BOX | not run | not run | 0x484bd20aabe66ee5 |
| llama8b.qkv.t8 | convert.int8.quantize.a | AGREE | 0xbd0b5aed2b83cb45 | 0xbd0b5aed2b83cb45 | 0xbd0b5aed2b83cb45 |
| llama8b.qkv.t8 | convert.int8.pack.b | AGREE | 0xa8ce17326c01345 | 0xa8ce17326c01345 | 0xa8ce17326c01345 |
| llama8b.qkv.t8 | convert.bf16.pack.b | AGREE | 0xdccc5a093a080c52 | 0xdccc5a093a080c52 | 0xdccc5a093a080c52 |
| llama8b.qkv.t8 | convert.bf16.widen.b | AGREE | 0x944a5da41b4d2325 | 0x944a5da41b4d2325 | 0x944a5da41b4d2325 |
| llama8b.qkv.t8 | convert.int8.dequantize.b | AGREE | 0x24697157b3da2325 | 0x24697157b3da2325 | 0x24697157b3da2325 |
| llama8b.qkv.t512 | fp32.v1 | AGREE | 0x67d3d940546b834c | 0x67d3d940546b834c | 0x67d3d940546b834c |
| llama8b.qkv.t512 | bf16f32.v1.fused | AGREE | 0x4abd79a5b10221e5 | 0x4abd79a5b10221e5 | 0x4abd79a5b10221e5 |
| llama8b.qkv.t512 | bf16f32.v1.widen | AGREE | 0x4abd79a5b10221e5 | 0x4abd79a5b10221e5 | 0x4abd79a5b10221e5 |
| llama8b.qkv.t512 | int8i32.v1.flat | AGREE | 0xa5f2ce1f1b9a0ebd | 0xa5f2ce1f1b9a0ebd | 0xa5f2ce1f1b9a0ebd |
| llama8b.qkv.t512 | int8i32.v1.mma | AGREE | 0xa5f2ce1f1b9a0ebd | 0xa5f2ce1f1b9a0ebd | not run |
| llama8b.qkv.t512 | int8i32.v1.applechunk | ONE BOX | not run | not run | 0xa5f2ce1f1b9a0ebd |
| llama8b.qkv.t512 | convert.int8.quantize.a | AGREE | 0xe8bb3ad105a4ae26 | 0xe8bb3ad105a4ae26 | 0xe8bb3ad105a4ae26 |
| llama8b.qkv.t512 | convert.int8.pack.b | AGREE | 0xd5ea46fdcb251db4 | 0xd5ea46fdcb251db4 | 0xd5ea46fdcb251db4 |
| llama8b.qkv.t512 | convert.bf16.pack.b | AGREE | 0xaef2da473a8e2a10 | 0xaef2da473a8e2a10 | 0xaef2da473a8e2a10 |
| llama8b.qkv.t512 | convert.bf16.widen.b | AGREE | 0xc8e5243510392325 | 0xc8e5243510392325 | 0xc8e5243510392325 |
| llama8b.qkv.t512 | convert.int8.dequantize.b | AGREE | 0x1798a32eac742325 | 0x1798a32eac742325 | 0x1798a32eac742325 |
| llama8b.mlp_up.t1 | fp32.v1 | AGREE | 0x4965ea6af2ff5e61 | 0x4965ea6af2ff5e61 | 0x4965ea6af2ff5e61 |
| llama8b.mlp_up.t1 | bf16f32.v1.fused | AGREE | 0x5e3f5ff35a448024 | 0x5e3f5ff35a448024 | 0x5e3f5ff35a448024 |
| llama8b.mlp_up.t1 | bf16f32.v1.widen | AGREE | 0x5e3f5ff35a448024 | 0x5e3f5ff35a448024 | 0x5e3f5ff35a448024 |
| llama8b.mlp_up.t1 | int8i32.v1.flat | AGREE | 0xff302668595b02d | 0xff302668595b02d | 0xff302668595b02d |
| llama8b.mlp_up.t1 | int8i32.v1.mma | AGREE | 0xff302668595b02d | 0xff302668595b02d | not run |
| llama8b.mlp_up.t1 | int8i32.v1.applechunk | ONE BOX | not run | not run | 0xff302668595b02d |
| llama8b.mlp_up.t1 | convert.int8.quantize.a | AGREE | 0x1343c750c1ded02 | 0x1343c750c1ded02 | 0x1343c750c1ded02 |
| llama8b.mlp_up.t1 | convert.int8.pack.b | AGREE | 0x9dd5dd0e7f10b4d0 | 0x9dd5dd0e7f10b4d0 | 0x9dd5dd0e7f10b4d0 |
| llama8b.mlp_up.t1 | convert.bf16.pack.b | AGREE | 0xed618e2333b13294 | 0xed618e2333b13294 | 0xed618e2333b13294 |
| llama8b.mlp_up.t1 | convert.bf16.widen.b | AGREE | 0x573b3a44b4012325 | 0x573b3a44b4012325 | 0x573b3a44b4012325 |
| llama8b.mlp_up.t1 | convert.int8.dequantize.b | AGREE | 0x5a7e8639220c2325 | 0x5a7e8639220c2325 | 0x5a7e8639220c2325 |
| llama8b.mlp_up.t8 | fp32.v1 | AGREE | 0x7b30465d325906e | 0x7b30465d325906e | 0x7b30465d325906e |
| llama8b.mlp_up.t8 | bf16f32.v1.fused | AGREE | 0x1531c10dcd152579 | 0x1531c10dcd152579 | 0x1531c10dcd152579 |
| llama8b.mlp_up.t8 | bf16f32.v1.widen | AGREE | 0x1531c10dcd152579 | 0x1531c10dcd152579 | 0x1531c10dcd152579 |
| llama8b.mlp_up.t8 | int8i32.v1.flat | AGREE | 0xbd4b3a783f9d9a0d | 0xbd4b3a783f9d9a0d | 0xbd4b3a783f9d9a0d |
| llama8b.mlp_up.t8 | int8i32.v1.mma | AGREE | 0xbd4b3a783f9d9a0d | 0xbd4b3a783f9d9a0d | not run |
| llama8b.mlp_up.t8 | int8i32.v1.applechunk | ONE BOX | not run | not run | 0xbd4b3a783f9d9a0d |
| llama8b.mlp_up.t8 | convert.int8.quantize.a | AGREE | 0x5e46321f760ef85d | 0x5e46321f760ef85d | 0x5e46321f760ef85d |
| llama8b.mlp_up.t8 | convert.int8.pack.b | AGREE | 0xa9e079afde494e3d | 0xa9e079afde494e3d | 0xa9e079afde494e3d |
| llama8b.mlp_up.t8 | convert.bf16.pack.b | AGREE | 0xc5111a6224412c3f | 0xc5111a6224412c3f | 0xc5111a6224412c3f |
| llama8b.mlp_up.t8 | convert.bf16.widen.b | AGREE | 0x6a679200ae02325 | 0x6a679200ae02325 | 0x6a679200ae02325 |
| llama8b.mlp_up.t8 | convert.int8.dequantize.b | AGREE | 0x89939dca58a02325 | 0x89939dca58a02325 | 0x89939dca58a02325 |
| llama8b.mlp_up.t512 | fp32.v1 | AGREE | 0x27a166783c45f9f0 | 0x27a166783c45f9f0 | 0x27a166783c45f9f0 |
| llama8b.mlp_up.t512 | bf16f32.v1.fused | AGREE | 0x5b29318ad80be465 | 0x5b29318ad80be465 | 0x5b29318ad80be465 |
| llama8b.mlp_up.t512 | bf16f32.v1.widen | AGREE | 0x5b29318ad80be465 | 0x5b29318ad80be465 | 0x5b29318ad80be465 |
| llama8b.mlp_up.t512 | int8i32.v1.flat | AGREE | 0x220ff5fe96f5ef55 | 0x220ff5fe96f5ef55 | 0x220ff5fe96f5ef55 |
| llama8b.mlp_up.t512 | int8i32.v1.mma | AGREE | 0x220ff5fe96f5ef55 | 0x220ff5fe96f5ef55 | not run |
| llama8b.mlp_up.t512 | int8i32.v1.applechunk | ONE BOX | not run | not run | 0x220ff5fe96f5ef55 |
| llama8b.mlp_up.t512 | convert.int8.quantize.a | AGREE | 0x8c503fb082856006 | 0x8c503fb082856006 | 0x8c503fb082856006 |
| llama8b.mlp_up.t512 | convert.int8.pack.b | AGREE | 0xf8c1486c6aaba4c1 | 0xf8c1486c6aaba4c1 | 0xf8c1486c6aaba4c1 |
| llama8b.mlp_up.t512 | convert.bf16.pack.b | AGREE | 0x1c04d570a3d23380 | 0x1c04d570a3d23380 | 0x1c04d570a3d23380 |
| llama8b.mlp_up.t512 | convert.bf16.widen.b | AGREE | 0x5c9d5dc97a412325 | 0x5c9d5dc97a412325 | 0x5c9d5dc97a412325 |
| llama8b.mlp_up.t512 | convert.int8.dequantize.b | AGREE | 0xb584f740e2d62325 | 0xb584f740e2d62325 | 0xb584f740e2d62325 |
| llama8b.mlp_down.t1 | fp32.v1 | AGREE | 0x4efb8e54c724fa9 | 0x4efb8e54c724fa9 | 0x4efb8e54c724fa9 |
| llama8b.mlp_down.t1 | bf16f32.v1.fused | AGREE | 0xd044aebb510dc9 | 0xd044aebb510dc9 | 0xd044aebb510dc9 |
| llama8b.mlp_down.t1 | bf16f32.v1.widen | AGREE | 0xd044aebb510dc9 | 0xd044aebb510dc9 | 0xd044aebb510dc9 |
| llama8b.mlp_down.t1 | int8i32.v1.flat | AGREE | 0x1ffdde3ffffb72c9 | 0x1ffdde3ffffb72c9 | 0x1ffdde3ffffb72c9 |
| llama8b.mlp_down.t1 | int8i32.v1.mma | AGREE | 0x1ffdde3ffffb72c9 | 0x1ffdde3ffffb72c9 | not run |
| llama8b.mlp_down.t1 | int8i32.v1.applechunk | ONE BOX | not run | not run | 0x1ffdde3ffffb72c9 |
| llama8b.mlp_down.t1 | convert.int8.quantize.a | AGREE | 0x160d734878c0f458 | 0x160d734878c0f458 | 0x160d734878c0f458 |
| llama8b.mlp_down.t1 | convert.int8.pack.b | AGREE | 0x31274090cc79ba2f | 0x31274090cc79ba2f | 0x31274090cc79ba2f |
| llama8b.mlp_down.t1 | convert.bf16.pack.b | AGREE | 0x9dba12f4fe0c9e9b | 0x9dba12f4fe0c9e9b | 0x9dba12f4fe0c9e9b |
| llama8b.mlp_down.t1 | convert.bf16.widen.b | AGREE | 0x45255120c9d82325 | 0x45255120c9d82325 | 0x45255120c9d82325 |
| llama8b.mlp_down.t1 | convert.int8.dequantize.b | AGREE | 0x8e72f1c5a3402325 | 0x8e72f1c5a3402325 | 0x8e72f1c5a3402325 |
| llama8b.mlp_down.t8 | fp32.v1 | AGREE | 0xfb21004541ae3e8e | 0xfb21004541ae3e8e | 0xfb21004541ae3e8e |
| llama8b.mlp_down.t8 | bf16f32.v1.fused | AGREE | 0xcbe807d3ae13612f | 0xcbe807d3ae13612f | 0xcbe807d3ae13612f |
| llama8b.mlp_down.t8 | bf16f32.v1.widen | AGREE | 0xcbe807d3ae13612f | 0xcbe807d3ae13612f | 0xcbe807d3ae13612f |
| llama8b.mlp_down.t8 | int8i32.v1.flat | AGREE | 0x694aafd520fdadad | 0x694aafd520fdadad | 0x694aafd520fdadad |
| llama8b.mlp_down.t8 | int8i32.v1.mma | AGREE | 0x694aafd520fdadad | 0x694aafd520fdadad | not run |
| llama8b.mlp_down.t8 | int8i32.v1.applechunk | ONE BOX | not run | not run | 0x694aafd520fdadad |
| llama8b.mlp_down.t8 | convert.int8.quantize.a | AGREE | 0xbb4f2a7232392bfe | 0xbb4f2a7232392bfe | 0xbb4f2a7232392bfe |
| llama8b.mlp_down.t8 | convert.int8.pack.b | AGREE | 0x4b12dcda2ab9acd4 | 0x4b12dcda2ab9acd4 | 0x4b12dcda2ab9acd4 |
| llama8b.mlp_down.t8 | convert.bf16.pack.b | AGREE | 0xfbacba127cf6177e | 0xfbacba127cf6177e | 0xfbacba127cf6177e |
| llama8b.mlp_down.t8 | convert.bf16.widen.b | AGREE | 0xcbcb5933d8352325 | 0xcbcb5933d8352325 | 0xcbcb5933d8352325 |
| llama8b.mlp_down.t8 | convert.int8.dequantize.b | AGREE | 0x9dcad54d7e462325 | 0x9dcad54d7e462325 | 0x9dcad54d7e462325 |
| llama8b.mlp_down.t512 | fp32.v1 | AGREE | 0x4743c97cb961339e | 0x4743c97cb961339e | 0x4743c97cb961339e |
| llama8b.mlp_down.t512 | bf16f32.v1.fused | AGREE | 0x508d64e0e41d3350 | 0x508d64e0e41d3350 | 0x508d64e0e41d3350 |
| llama8b.mlp_down.t512 | bf16f32.v1.widen | AGREE | 0x508d64e0e41d3350 | 0x508d64e0e41d3350 | 0x508d64e0e41d3350 |
| llama8b.mlp_down.t512 | int8i32.v1.flat | AGREE | 0x6120545eacb068ad | 0x6120545eacb068ad | 0x6120545eacb068ad |
| llama8b.mlp_down.t512 | int8i32.v1.mma | AGREE | 0x6120545eacb068ad | 0x6120545eacb068ad | not run |
| llama8b.mlp_down.t512 | int8i32.v1.applechunk | ONE BOX | not run | not run | 0x6120545eacb068ad |
| llama8b.mlp_down.t512 | convert.int8.quantize.a | AGREE | 0x903f6ff84c6082 | 0x903f6ff84c6082 | 0x903f6ff84c6082 |
| llama8b.mlp_down.t512 | convert.int8.pack.b | AGREE | 0xa046020a81d05746 | 0xa046020a81d05746 | 0xa046020a81d05746 |
| llama8b.mlp_down.t512 | convert.bf16.pack.b | AGREE | 0xd1b066f4e6a61fed | 0xd1b066f4e6a61fed | 0xd1b066f4e6a61fed |
| llama8b.mlp_down.t512 | convert.bf16.widen.b | AGREE | 0xf0fcf217ba5c2325 | 0xf0fcf217ba5c2325 | 0xf0fcf217ba5c2325 |
| llama8b.mlp_down.t512 | convert.int8.dequantize.b | AGREE | 0xd972f24f17be2325 | 0xd972f24f17be2325 | 0xd972f24f17be2325 |
| llama8b.lm_head.t1 | fp32.v1 | AGREE | 0xcdc134cbce391535 | 0xcdc134cbce391535 | 0xcdc134cbce391535 |
| llama8b.lm_head.t1 | bf16f32.v1.fused | AGREE | 0x23f2e7e17341274a | 0x23f2e7e17341274a | 0x23f2e7e17341274a |
| llama8b.lm_head.t1 | bf16f32.v1.widen | AGREE | 0x23f2e7e17341274a | 0x23f2e7e17341274a | 0x23f2e7e17341274a |
| llama8b.lm_head.t1 | int8i32.v1.flat | AGREE | 0xe0222b2f9de835ed | 0xe0222b2f9de835ed | 0xe0222b2f9de835ed |
| llama8b.lm_head.t1 | int8i32.v1.mma | AGREE | 0xe0222b2f9de835ed | 0xe0222b2f9de835ed | not run |
| llama8b.lm_head.t1 | int8i32.v1.applechunk | ONE BOX | not run | not run | 0xe0222b2f9de835ed |
| llama8b.lm_head.t1 | convert.int8.quantize.a | AGREE | 0x67d2e298b047bf2b | 0x67d2e298b047bf2b | 0x67d2e298b047bf2b |
| llama8b.lm_head.t1 | convert.int8.pack.b | AGREE | 0x1c75800d0b1054bd | 0x1c75800d0b1054bd | 0x1c75800d0b1054bd |
| llama8b.lm_head.t1 | convert.bf16.pack.b | AGREE | 0xa3f53235cc152516 | 0xa3f53235cc152516 | 0xa3f53235cc152516 |
| llama8b.lm_head.t1 | convert.bf16.widen.b | AGREE | 0x63450213e8e12325 | 0x63450213e8e12325 | 0x63450213e8e12325 |
| llama8b.lm_head.t1 | convert.int8.dequantize.b | AGREE | 0x23115689185e2325 | 0x23115689185e2325 | 0x23115689185e2325 |
| llama8b.lm_head.t8 | fp32.v1 | AGREE | 0x3ebab59d0da21444 | 0x3ebab59d0da21444 | 0x3ebab59d0da21444 |
| llama8b.lm_head.t8 | bf16f32.v1.fused | AGREE | 0x82bad7b4e5c4c263 | 0x82bad7b4e5c4c263 | 0x82bad7b4e5c4c263 |
| llama8b.lm_head.t8 | bf16f32.v1.widen | AGREE | 0x82bad7b4e5c4c263 | 0x82bad7b4e5c4c263 | 0x82bad7b4e5c4c263 |
| llama8b.lm_head.t8 | int8i32.v1.flat | AGREE | 0x8151fd05ff287a5d | 0x8151fd05ff287a5d | 0x8151fd05ff287a5d |
| llama8b.lm_head.t8 | int8i32.v1.mma | AGREE | 0x8151fd05ff287a5d | 0x8151fd05ff287a5d | not run |
| llama8b.lm_head.t8 | int8i32.v1.applechunk | ONE BOX | not run | not run | 0x8151fd05ff287a5d |
| llama8b.lm_head.t8 | convert.int8.quantize.a | AGREE | 0x443be8ad16d189a3 | 0x443be8ad16d189a3 | 0x443be8ad16d189a3 |
| llama8b.lm_head.t8 | convert.int8.pack.b | AGREE | 0x8e01af3f075298bb | 0x8e01af3f075298bb | 0x8e01af3f075298bb |
| llama8b.lm_head.t8 | convert.bf16.pack.b | AGREE | 0xbdac0264c773e6f1 | 0xbdac0264c773e6f1 | 0xbdac0264c773e6f1 |
| llama8b.lm_head.t8 | convert.bf16.widen.b | AGREE | 0x4e388bd358a22325 | 0x4e388bd358a22325 | 0x4e388bd358a22325 |
| llama8b.lm_head.t8 | convert.int8.dequantize.b | AGREE | 0xb8835a7fc2342325 | 0xb8835a7fc2342325 | 0xb8835a7fc2342325 |
| llama8b.lm_head.t512 | fp32.v1 | AGREE | 0x84fd143bb625cf70 | 0x84fd143bb625cf70 | 0x84fd143bb625cf70 |
| llama8b.lm_head.t512 | bf16f32.v1.fused | AGREE | 0xa75052ecc1e78b46 | 0xa75052ecc1e78b46 | 0xa75052ecc1e78b46 |
| llama8b.lm_head.t512 | bf16f32.v1.widen | AGREE | 0xa75052ecc1e78b46 | 0xa75052ecc1e78b46 | 0xa75052ecc1e78b46 |
| llama8b.lm_head.t512 | int8i32.v1.flat | AGREE | 0xaae4f3d3c69453cd | 0xaae4f3d3c69453cd | 0xaae4f3d3c69453cd |
| llama8b.lm_head.t512 | int8i32.v1.mma | AGREE | 0xaae4f3d3c69453cd | 0xaae4f3d3c69453cd | not run |
| llama8b.lm_head.t512 | int8i32.v1.applechunk | ONE BOX | not run | not run | 0xaae4f3d3c69453cd |
| llama8b.lm_head.t512 | convert.int8.quantize.a | AGREE | 0xfd89ee79d025dda7 | 0xfd89ee79d025dda7 | 0xfd89ee79d025dda7 |
| llama8b.lm_head.t512 | convert.int8.pack.b | AGREE | 0xcc187b17bc36366d | 0xcc187b17bc36366d | 0xcc187b17bc36366d |
| llama8b.lm_head.t512 | convert.bf16.pack.b | AGREE | 0xaabc0c6ba2a73e85 | 0xaabc0c6ba2a73e85 | 0xaabc0c6ba2a73e85 |
| llama8b.lm_head.t512 | convert.bf16.widen.b | AGREE | 0xbd6a89a89fc02325 | 0xbd6a89a89fc02325 | 0xbd6a89a89fc02325 |
| llama8b.lm_head.t512 | convert.int8.dequantize.b | AGREE | 0xf6a61b0da9142325 | 0xf6a61b0da9142325 | 0xf6a61b0da9142325 |

| arm | shapes | AGREE | DISAGREE | ONE BOX | not comparable |
|---|---:|---:|---:|---:|---:|
| fp32.v1 | 12 | 12 | 0 | 0 | 0 |
| bf16f32.v1.fused | 12 | 12 | 0 | 0 | 0 |
| bf16f32.v1.widen | 12 | 12 | 0 | 0 | 0 |
| int8i32.v1.flat | 12 | 12 | 0 | 0 | 0 |
| int8i32.v1.mma | 12 | 12 | 0 | 0 | 0 |
| int8i32.v1.applechunk | 12 | 0 | 0 | 12 | 0 |
| convert.int8.quantize.a | 12 | 12 | 0 | 0 | 0 |
| convert.int8.pack.b | 12 | 12 | 0 | 0 | 0 |
| convert.bf16.pack.b | 12 | 12 | 0 | 0 | 0 |
| convert.bf16.widen.b | 12 | 12 | 0 | 0 | 0 |
| convert.int8.dequantize.b | 12 | 12 | 0 | 0 | 0 |

### One profile, every plan, every box

AGREE: every digest any plan of the profile printed on any box at the shape is the same, and at
least two boxes ran a plan of it.

| profile | plans | shapes | AGREE | DISAGREE | ONE BOX |
|---|---|---:|---:|---:|---:|
| fp32.v1 | fp32.v1 | 12 | 12 | 0 | 0 |
| bf16f32.v1 | bf16f32.v1.fused, bf16f32.v1.widen | 12 | 12 | 0 | 0 |
| int8i32.v1 | int8i32.v1.flat, int8i32.v1.mma, int8i32.v1.applechunk | 12 | 12 | 0 | 0 |

