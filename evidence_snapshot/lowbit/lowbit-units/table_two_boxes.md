# Low-bit GEMM plans: times, rates and digests

Written by `tools/lowbit_units/table.py` from the `LOWBIT` lines of
`bench/gemm_lowbit_price_main.mojo`. Every time is one box's time on one run: one call and one
synchronize per sample, the median of the timed calls, the minimum beside it. Every ratio is a time
over a time at the same shape on the same box; above 1 the numerator took longer. Rates are
G MAC/s (multiply-accumulates, `m n k`) for a product and G elem/s for a conversion.

## h100 (column nvidia)

Vendor comparison: cuBLAS on NVIDIA H100 NVL, torch 2.13.0+cu129, CUDA 12.9, median of 10. COMPARISON ONLY, no digest, no identity claim.

### llama8b.qkv.t1: m=1 n=4096 k=4096 (FULL)

| arm | median ms | min ms | rate | arm over fp32.v1 | note |
|---|---:|---:|---:|---:|---|
| fp32.v1 | 0.0704 | 0.0698 | 238.3804 GMAC/s | 1.000 | plan=13 |
| bf16f32.v1.fused | 0.7874 | 0.7857 | 21.3080 GMAC/s | 11.185 | dispatched |
| bf16f32.v1.widen | 0.1210 | 0.1203 | 138.7028 GMAC/s | 1.719 | not-dispatched(cells<=16384) |
| int8i32.v1.flat | 0.6643 | 0.6635 | 25.2540 GMAC/s | 9.436 | not-dispatched(the-column-has-the-unit) |
| int8i32.v1.mma | 0.1574 | 0.1555 | 106.6086 GMAC/s | 2.236 | dispatched |
| int8i32.v1.applechunk | not run | | | | the column does not have the unit |
| convert.int8.quantize.a | 0.4879 | 0.4854 | 0.0084 Gelem/s | 6.930 | per-call |
| convert.int8.pack.b | 2.6618 | 2.6567 | 6.3029 Gelem/s | 37.810 | once-per-weight |
| convert.bf16.pack.b | 0.0634 | 0.0628 | 264.4955 Gelem/s | 0.901 | once-per-weight |
| convert.bf16.widen.b | 0.0614 | 0.0612 | 273.4271 Gelem/s | 0.872 | per-call-when-materialized |
| convert.int8.dequantize.b | 0.0776 | 0.0772 | 216.2207 Gelem/s | 1.102 | per-call-when-materialized |
| DERIVED int8i32.v1.flat + quantize.a | 1.1522 | | | 16.366 | sum of two medians |
| DERIVED int8i32.v1.mma + quantize.a | 0.6453 | | | 9.166 | sum of two medians |
| vendor strict (COMPARISON ONLY) | 0.0402 | 0.0392 | 417.4058 GMAC/s | | fp32.v1 over this: 1.752 |
| vendor tf32 (COMPARISON ONLY) | 0.0403 | 0.0388 | 416.4505 GMAC/s | | fp32.v1 over this: 1.747 |
| vendor bf16 (COMPARISON ONLY) | 0.0282 | 0.0268 | 594.0739 GMAC/s | | fp32.v1 over this: 2.493 |

### llama8b.qkv.t8: m=8 n=4096 k=4096 (FULL)

| arm | median ms | min ms | rate | arm over fp32.v1 | note |
|---|---:|---:|---:|---:|---|
| fp32.v1 | 0.1006 | 0.0999 | 1334.8091 GMAC/s | 1.000 | plan=14 |
| bf16f32.v1.fused | 0.7795 | 0.7781 | 172.1736 GMAC/s | 7.749 | not-dispatched(cells>16384) |
| bf16f32.v1.widen | 0.1507 | 0.1496 | 890.4454 GMAC/s | 1.498 | dispatched |
| int8i32.v1.flat | 0.6628 | 0.6623 | 202.5145 GMAC/s | 6.588 | not-dispatched(the-column-has-the-unit) |
| int8i32.v1.mma | 0.1471 | 0.1466 | 912.5553 GMAC/s | 1.462 | dispatched |
| int8i32.v1.applechunk | not run | | | | the column does not have the unit |
| convert.int8.quantize.a | 0.5959 | 0.5946 | 0.0550 Gelem/s | 5.923 | per-call |
| convert.int8.pack.b | 2.4702 | 2.4665 | 6.7918 Gelem/s | 24.555 | once-per-weight |
| convert.bf16.pack.b | 0.0622 | 0.0620 | 269.6175 Gelem/s | 0.618 | once-per-weight |
| convert.bf16.widen.b | 0.0612 | 0.0606 | 274.0435 Gelem/s | 0.608 | per-call-when-materialized |
| convert.int8.dequantize.b | 0.0774 | 0.0772 | 216.6787 Gelem/s | 0.769 | per-call-when-materialized |
| DERIVED int8i32.v1.flat + quantize.a | 1.2587 | | | 12.512 | sum of two medians |
| DERIVED int8i32.v1.mma + quantize.a | 0.7430 | | | 7.386 | sum of two medians |
| vendor strict (COMPARISON ONLY) | 0.0651 | 0.0629 | 2061.1587 GMAC/s | | fp32.v1 over this: 1.545 |
| vendor tf32 (COMPARISON ONLY) | 0.0443 | 0.0434 | 3032.6632 GMAC/s | | fp32.v1 over this: 2.273 |
| vendor bf16 (COMPARISON ONLY) | 0.0264 | 0.0258 | 5089.9814 GMAC/s | | fp32.v1 over this: 3.815 |

### llama8b.qkv.t512: m=512 n=4096 k=4096 (FULL)

| arm | median ms | min ms | rate | arm over fp32.v1 | note |
|---|---:|---:|---:|---:|---|
| fp32.v1 | 1.0341 | 1.0281 | 8306.9018 GMAC/s | 1.000 | plan=10 |
| bf16f32.v1.fused | 39.4126 | 39.2847 | 217.9491 GMAC/s | 38.113 | not-dispatched(cells>16384) |
| bf16f32.v1.widen | 1.0941 | 1.0892 | 7851.3933 GMAC/s | 1.058 | dispatched |
| int8i32.v1.flat | 39.1804 | 39.1717 | 219.2404 GMAC/s | 37.888 | not-dispatched(the-column-has-the-unit) |
| int8i32.v1.mma | 1.4211 | 1.3435 | 6044.4440 GMAC/s | 1.374 | dispatched |
| int8i32.v1.applechunk | not run | | | | the column does not have the unit |
| convert.int8.quantize.a | 2.4705 | 2.4535 | 0.8489 Gelem/s | 2.389 | per-call |
| convert.int8.pack.b | 2.4759 | 2.4686 | 6.7761 Gelem/s | 2.394 | once-per-weight |
| convert.bf16.pack.b | 0.0647 | 0.0633 | 259.1556 Gelem/s | 0.063 | once-per-weight |
| convert.bf16.widen.b | 0.0610 | 0.0601 | 275.2168 Gelem/s | 0.059 | per-call-when-materialized |
| convert.int8.dequantize.b | 0.0782 | 0.0774 | 214.4300 Gelem/s | 0.076 | per-call-when-materialized |
| DERIVED int8i32.v1.flat + quantize.a | 41.6509 | | | 40.277 | sum of two medians |
| DERIVED int8i32.v1.mma + quantize.a | 3.8916 | | | 3.763 | sum of two medians |
| vendor strict (COMPARISON ONLY) | 0.4199 | 0.4169 | 20457.8307 GMAC/s | | fp32.v1 over this: 2.463 |
| vendor tf32 (COMPARISON ONLY) | 0.0789 | 0.0766 | 108876.5970 GMAC/s | | fp32.v1 over this: 13.107 |
| vendor bf16 (COMPARISON ONLY) | 0.0419 | 0.0410 | 205070.9156 GMAC/s | | fp32.v1 over this: 24.687 |

### llama8b.mlp_up.t1: m=1 n=14336 k=4096 (FULL)

| arm | median ms | min ms | rate | arm over fp32.v1 | note |
|---|---:|---:|---:|---:|---|
| fp32.v1 | 0.1872 | 0.1859 | 313.6900 GMAC/s | 1.000 | plan=13 |
| bf16f32.v1.fused | 0.7926 | 0.7900 | 74.0874 GMAC/s | 4.234 | dispatched |
| bf16f32.v1.widen | 0.3646 | 0.3642 | 161.0566 GMAC/s | 1.948 | not-dispatched(cells<=16384) |
| int8i32.v1.flat | 0.6645 | 0.6631 | 88.3682 GMAC/s | 3.550 | not-dispatched(the-column-has-the-unit) |
| int8i32.v1.mma | 0.1926 | 0.1921 | 304.8819 GMAC/s | 1.029 | dispatched |
| int8i32.v1.applechunk | not run | | | | the column does not have the unit |
| convert.int8.quantize.a | 0.5005 | 0.4990 | 0.0082 Gelem/s | 2.674 | per-call |
| convert.int8.pack.b | 2.4833 | 2.4797 | 23.6465 Gelem/s | 13.265 | once-per-weight |
| convert.bf16.pack.b | 0.1976 | 0.1972 | 297.2049 Gelem/s | 1.056 | once-per-weight |
| convert.bf16.widen.b | 0.1871 | 0.1867 | 313.8979 Gelem/s | 0.999 | per-call-when-materialized |
| convert.int8.dequantize.b | 0.2445 | 0.2435 | 240.1686 Gelem/s | 1.306 | per-call-when-materialized |
| DERIVED int8i32.v1.flat + quantize.a | 1.1650 | | | 6.223 | sum of two medians |
| DERIVED int8i32.v1.mma + quantize.a | 0.6931 | | | 3.702 | sum of two medians |
| vendor strict (COMPARISON ONLY) | 0.0873 | 0.0859 | 672.3367 GMAC/s | | fp32.v1 over this: 2.143 |
| vendor tf32 (COMPARISON ONLY) | 0.0876 | 0.0862 | 670.0290 GMAC/s | | fp32.v1 over this: 2.136 |
| vendor bf16 (COMPARISON ONLY) | 0.0518 | 0.0502 | 1133.8979 GMAC/s | | fp32.v1 over this: 3.615 |

### llama8b.mlp_up.t8: m=8 n=14336 k=4096 (FULL)

| arm | median ms | min ms | rate | arm over fp32.v1 | note |
|---|---:|---:|---:|---:|---|
| fp32.v1 | 0.2713 | 0.2705 | 1731.6884 GMAC/s | 1.000 | plan=14 |
| bf16f32.v1.fused | 2.5292 | 2.5274 | 185.7362 GMAC/s | 9.323 | not-dispatched(cells>16384) |
| bf16f32.v1.widen | 0.4528 | 0.4508 | 1037.4306 GMAC/s | 1.669 | dispatched |
| int8i32.v1.flat | 2.5016 | 2.4963 | 187.7881 GMAC/s | 9.221 | not-dispatched(the-column-has-the-unit) |
| int8i32.v1.mma | 0.2085 | 0.2070 | 2252.5260 GMAC/s | 0.769 | dispatched |
| int8i32.v1.applechunk | not run | | | | the column does not have the unit |
| convert.int8.quantize.a | 0.6109 | 0.6091 | 0.0536 Gelem/s | 2.252 | per-call |
| convert.int8.pack.b | 2.4844 | 2.4754 | 23.6351 Gelem/s | 9.157 | once-per-weight |
| convert.bf16.pack.b | 0.1981 | 0.1971 | 296.3933 Gelem/s | 0.730 | once-per-weight |
| convert.bf16.widen.b | 0.1870 | 0.1868 | 314.0809 Gelem/s | 0.689 | per-call-when-materialized |
| convert.int8.dequantize.b | 0.2447 | 0.2441 | 239.9919 Gelem/s | 0.902 | per-call-when-materialized |
| DERIVED int8i32.v1.flat + quantize.a | 3.1125 | | | 11.473 | sum of two medians |
| DERIVED int8i32.v1.mma + quantize.a | 0.8194 | | | 3.020 | sum of two medians |
| vendor strict (COMPARISON ONLY) | 0.1529 | 0.1500 | 3073.1649 GMAC/s | | fp32.v1 over this: 1.775 |
| vendor tf32 (COMPARISON ONLY) | 0.0986 | 0.0974 | 4766.1642 GMAC/s | | fp32.v1 over this: 2.753 |
| vendor bf16 (COMPARISON ONLY) | 0.0516 | 0.0504 | 9105.7363 GMAC/s | | fp32.v1 over this: 5.259 |

### llama8b.mlp_up.t512: m=512 n=14336 k=4096 (FULL)

| arm | median ms | min ms | rate | arm over fp32.v1 | note |
|---|---:|---:|---:|---:|---|
| fp32.v1 | 3.3454 | 3.3345 | 8986.7671 GMAC/s | 1.000 | plan=10 |
| bf16f32.v1.fused | 139.1256 | 138.9246 | 216.0981 GMAC/s | 41.587 | not-dispatched(cells>16384) |
| bf16f32.v1.widen | 3.5246 | 3.5166 | 8529.8673 GMAC/s | 1.054 | dispatched |
| int8i32.v1.flat | 135.6717 | 135.6553 | 221.5994 GMAC/s | 40.555 | not-dispatched(the-column-has-the-unit) |
| int8i32.v1.mma | 4.6375 | 4.6308 | 6483.0296 GMAC/s | 1.386 | dispatched |
| int8i32.v1.applechunk | not run | | | | the column does not have the unit |
| convert.int8.quantize.a | 2.4703 | 2.4648 | 0.8489 Gelem/s | 0.738 | per-call |
| convert.int8.pack.b | 2.4789 | 2.4750 | 23.6876 Gelem/s | 0.741 | once-per-weight |
| convert.bf16.pack.b | 0.1988 | 0.1979 | 295.4018 Gelem/s | 0.059 | once-per-weight |
| convert.bf16.widen.b | 0.1873 | 0.1870 | 313.4723 Gelem/s | 0.056 | per-call-when-materialized |
| convert.int8.dequantize.b | 0.2460 | 0.2443 | 238.6537 Gelem/s | 0.074 | per-call-when-materialized |
| DERIVED int8i32.v1.flat + quantize.a | 138.1420 | | | 41.293 | sum of two medians |
| DERIVED int8i32.v1.mma + quantize.a | 7.1078 | | | 2.125 | sum of two medians |
| vendor strict (COMPARISON ONLY) | 1.6535 | 1.6473 | 18182.5823 GMAC/s | | fp32.v1 over this: 2.023 |
| vendor tf32 (COMPARISON ONLY) | 0.2461 | 0.2429 | 122180.9757 GMAC/s | | fp32.v1 over this: 13.595 |
| vendor bf16 (COMPARISON ONLY) | 0.1391 | 0.1378 | 216105.2492 GMAC/s | | fp32.v1 over this: 24.047 |

### llama8b.mlp_down.t1: m=1 n=4096 k=14336 (FULL)

| arm | median ms | min ms | rate | arm over fp32.v1 | note |
|---|---:|---:|---:|---:|---|
| fp32.v1 | 0.1782 | 0.1776 | 329.4319 GMAC/s | 1.000 | plan=13 |
| bf16f32.v1.fused | 2.7594 | 2.7587 | 21.2804 GMAC/s | 15.485 | dispatched |
| bf16f32.v1.widen | 0.3558 | 0.3539 | 165.0386 GMAC/s | 1.997 | not-dispatched(cells<=16384) |
| int8i32.v1.flat | 2.3432 | 2.3414 | 25.0602 GMAC/s | 13.149 | not-dispatched(the-column-has-the-unit) |
| int8i32.v1.mma | 0.6522 | 0.6512 | 90.0287 GMAC/s | 3.660 | dispatched |
| int8i32.v1.applechunk | not run | | | | the column does not have the unit |
| convert.int8.quantize.a | 1.7350 | 1.7323 | 0.0083 Gelem/s | 9.736 | per-call |
| convert.int8.pack.b | 9.9669 | 9.9567 | 5.8915 Gelem/s | 55.931 | once-per-weight |
| convert.bf16.pack.b | 0.2013 | 0.2002 | 291.7530 Gelem/s | 1.130 | once-per-weight |
| convert.bf16.widen.b | 0.1892 | 0.1891 | 310.4379 Gelem/s | 1.062 | per-call-when-materialized |
| convert.int8.dequantize.b | 0.2466 | 0.2457 | 238.1204 Gelem/s | 1.384 | per-call-when-materialized |
| DERIVED int8i32.v1.flat + quantize.a | 4.0782 | | | 22.886 | sum of two medians |
| DERIVED int8i32.v1.mma + quantize.a | 2.3872 | | | 13.396 | sum of two medians |
| vendor strict (COMPARISON ONLY) | 0.0821 | 0.0807 | 715.0476 GMAC/s | | fp32.v1 over this: 2.170 |
| vendor tf32 (COMPARISON ONLY) | 0.0814 | 0.0806 | 721.7224 GMAC/s | | fp32.v1 over this: 2.190 |
| vendor bf16 (COMPARISON ONLY) | 0.0552 | 0.0538 | 1064.7167 GMAC/s | | fp32.v1 over this: 3.231 |

### llama8b.mlp_down.t8: m=8 n=4096 k=14336 (FULL)

| arm | median ms | min ms | rate | arm over fp32.v1 | note |
|---|---:|---:|---:|---:|---|
| fp32.v1 | 0.2877 | 0.2866 | 1632.6205 GMAC/s | 1.000 | plan=14 |
| bf16f32.v1.fused | 2.7010 | 2.6993 | 173.9219 GMAC/s | 9.388 | not-dispatched(cells>16384) |
| bf16f32.v1.widen | 0.4663 | 0.4650 | 1007.3467 GMAC/s | 1.621 | dispatched |
| int8i32.v1.flat | 2.3437 | 2.3429 | 200.4366 GMAC/s | 8.146 | not-dispatched(the-column-has-the-unit) |
| int8i32.v1.mma | 0.7006 | 0.6937 | 670.5445 GMAC/s | 2.435 | dispatched |
| int8i32.v1.applechunk | not run | | | | the column does not have the unit |
| convert.int8.quantize.a | 2.3796 | 2.3787 | 0.0482 Gelem/s | 8.271 | per-call |
| convert.int8.pack.b | 10.0034 | 9.9585 | 5.8700 Gelem/s | 34.770 | once-per-weight |
| convert.bf16.pack.b | 0.1989 | 0.1980 | 295.1627 Gelem/s | 0.691 | once-per-weight |
| convert.bf16.widen.b | 0.1874 | 0.1866 | 313.2917 Gelem/s | 0.651 | per-call-when-materialized |
| convert.int8.dequantize.b | 0.2445 | 0.2440 | 240.1745 Gelem/s | 0.850 | per-call-when-materialized |
| DERIVED int8i32.v1.flat + quantize.a | 4.7233 | | | 16.417 | sum of two medians |
| DERIVED int8i32.v1.mma + quantize.a | 3.0802 | | | 10.706 | sum of two medians |
| vendor strict (COMPARISON ONLY) | 0.1954 | 0.1938 | 2404.4960 GMAC/s | | fp32.v1 over this: 1.473 |
| vendor tf32 (COMPARISON ONLY) | 0.0945 | 0.0928 | 4972.7718 GMAC/s | | fp32.v1 over this: 3.046 |
| vendor bf16 (COMPARISON ONLY) | 0.0552 | 0.0543 | 8504.2345 GMAC/s | | fp32.v1 over this: 5.208 |

### llama8b.mlp_down.t512: m=512 n=4096 k=14336 (FULL)

| arm | median ms | min ms | rate | arm over fp32.v1 | note |
|---|---:|---:|---:|---:|---|
| fp32.v1 | 3.3604 | 3.3510 | 8946.7435 GMAC/s | 1.000 | plan=10 |
| bf16f32.v1.fused | 136.6104 | 136.3958 | 220.0767 GMAC/s | 40.653 | not-dispatched(cells>16384) |
| bf16f32.v1.widen | 3.5585 | 3.5523 | 8448.7725 GMAC/s | 1.059 | dispatched |
| int8i32.v1.flat | 137.9490 | 137.9182 | 217.9412 GMAC/s | 41.051 | not-dispatched(the-column-has-the-unit) |
| int8i32.v1.mma | 5.5686 | 5.5455 | 5398.9975 GMAC/s | 1.657 | dispatched |
| int8i32.v1.applechunk | not run | | | | the column does not have the unit |
| convert.int8.quantize.a | 9.9605 | 9.9569 | 0.7369 Gelem/s | 2.964 | per-call |
| convert.int8.pack.b | 9.9703 | 9.9647 | 5.8895 Gelem/s | 2.967 | once-per-weight |
| convert.bf16.pack.b | 0.1987 | 0.1983 | 295.4686 Gelem/s | 0.059 | once-per-weight |
| convert.bf16.widen.b | 0.1874 | 0.1864 | 313.3652 Gelem/s | 0.056 | per-call-when-materialized |
| convert.int8.dequantize.b | 0.2463 | 0.2448 | 238.3775 Gelem/s | 0.073 | per-call-when-materialized |
| DERIVED int8i32.v1.flat + quantize.a | 147.9095 | | | 44.015 | sum of two medians |
| DERIVED int8i32.v1.mma + quantize.a | 15.5291 | | | 4.621 | sum of two medians |
| vendor strict (COMPARISON ONLY) | 1.3886 | 1.3847 | 21650.9266 GMAC/s | | fp32.v1 over this: 2.420 |
| vendor tf32 (COMPARISON ONLY) | 0.2827 | 0.2755 | 106358.4228 GMAC/s | | fp32.v1 over this: 11.888 |
| vendor bf16 (COMPARISON ONLY) | 0.1372 | 0.1352 | 219079.4976 GMAC/s | | fp32.v1 over this: 24.487 |

### llama8b.lm_head.t1: m=1 n=128256 k=4096 (FULL)

| arm | median ms | min ms | rate | arm over fp32.v1 | note |
|---|---:|---:|---:|---:|---|
| fp32.v1 | 1.2058 | 1.2040 | 435.6819 GMAC/s | 1.000 | plan=13 |
| bf16f32.v1.fused | 2.5785 | 2.5643 | 203.7356 GMAC/s | 2.138 | not-dispatched(cells>16384) |
| bf16f32.v1.widen | 2.8046 | 2.8025 | 187.3102 GMAC/s | 2.326 | dispatched |
| int8i32.v1.flat | 2.4953 | 2.4943 | 210.5303 GMAC/s | 2.069 | not-dispatched(the-column-has-the-unit) |
| int8i32.v1.mma | 0.9591 | 0.9357 | 547.7528 GMAC/s | 0.795 | dispatched |
| int8i32.v1.applechunk | not run | | | | the column does not have the unit |
| convert.int8.quantize.a | 0.5021 | 0.5016 | 0.0082 Gelem/s | 0.416 | per-call |
| convert.int8.pack.b | 12.1600 | 12.1466 | 43.2021 Gelem/s | 10.085 | once-per-weight |
| convert.bf16.pack.b | 1.6690 | 1.6679 | 314.7600 Gelem/s | 1.384 | once-per-weight |
| convert.bf16.widen.b | 1.6068 | 1.6062 | 326.9393 Gelem/s | 1.333 | per-call-when-materialized |
| convert.int8.dequantize.b | 2.0987 | 2.0968 | 250.3211 Gelem/s | 1.741 | per-call-when-materialized |
| DERIVED int8i32.v1.flat + quantize.a | 2.9974 | | | 2.486 | sum of two medians |
| DERIVED int8i32.v1.mma + quantize.a | 1.4612 | | | 1.212 | sum of two medians |
| vendor strict (COMPARISON ONLY) | 0.7224 | 0.7116 | 727.2135 GMAC/s | | fp32.v1 over this: 1.669 |
| vendor tf32 (COMPARISON ONLY) | 0.7112 | 0.7008 | 738.7011 GMAC/s | | fp32.v1 over this: 1.696 |
| vendor bf16 (COMPARISON ONLY) | 0.3226 | 0.3170 | 1628.6954 GMAC/s | | fp32.v1 over this: 3.738 |

### llama8b.lm_head.t8: m=8 n=128256 k=4096 (FULL)

| arm | median ms | min ms | rate | arm over fp32.v1 | note |
|---|---:|---:|---:|---:|---|
| fp32.v1 | 2.1537 | 2.1517 | 1951.4170 GMAC/s | 1.000 | plan=14 |
| bf16f32.v1.fused | 19.4619 | 19.3580 | 215.9441 GMAC/s | 9.036 | not-dispatched(cells>16384) |
| bf16f32.v1.widen | 3.7583 | 3.7562 | 1118.2427 GMAC/s | 1.745 | dispatched |
| int8i32.v1.flat | 19.2647 | 19.2469 | 218.1550 GMAC/s | 8.945 | not-dispatched(the-column-has-the-unit) |
| int8i32.v1.mma | 1.3619 | 1.3231 | 3085.8405 GMAC/s | 0.632 | dispatched |
| int8i32.v1.applechunk | not run | | | | the column does not have the unit |
| convert.int8.quantize.a | 0.6090 | 0.6067 | 0.0538 Gelem/s | 0.283 | per-call |
| convert.int8.pack.b | 12.1535 | 12.0863 | 43.2250 Gelem/s | 5.643 | once-per-weight |
| convert.bf16.pack.b | 1.6692 | 1.6681 | 314.7185 Gelem/s | 0.775 | once-per-weight |
| convert.bf16.widen.b | 1.6089 | 1.6086 | 326.5132 Gelem/s | 0.747 | per-call-when-materialized |
| convert.int8.dequantize.b | 2.0973 | 2.0955 | 250.4876 Gelem/s | 0.974 | per-call-when-materialized |
| DERIVED int8i32.v1.flat + quantize.a | 19.8737 | | | 9.228 | sum of two medians |
| DERIVED int8i32.v1.mma + quantize.a | 1.9709 | | | 0.915 | sum of two medians |
| vendor strict (COMPARISON ONLY) | 1.0847 | 1.0779 | 3874.4800 GMAC/s | | fp32.v1 over this: 1.986 |
| vendor tf32 (COMPARISON ONLY) | 0.8190 | 0.8141 | 5131.5807 GMAC/s | | fp32.v1 over this: 2.630 |
| vendor bf16 (COMPARISON ONLY) | 0.3205 | 0.3191 | 13110.9921 GMAC/s | | fp32.v1 over this: 6.719 |

### llama8b.lm_head.t512: m=512 n=16032 k=4096 (CAPPED)

| arm | median ms | min ms | rate | arm over fp32.v1 | note |
|---|---:|---:|---:|---:|---|
| fp32.v1 | 3.7236 | 3.7170 | 9029.2184 GMAC/s | 1.000 | plan=10 |
| bf16f32.v1.fused | 156.5389 | 156.3681 | 214.7807 GMAC/s | 42.040 | not-dispatched(cells>16384) |
| bf16f32.v1.widen | 3.9218 | 3.9172 | 8573.0614 GMAC/s | 1.053 | dispatched |
| int8i32.v1.flat | 151.8930 | 151.8256 | 221.3501 GMAC/s | 40.792 | not-dispatched(the-column-has-the-unit) |
| int8i32.v1.mma | 5.1220 | 5.1191 | 6564.1187 GMAC/s | 1.376 | dispatched |
| int8i32.v1.applechunk | not run | | | | the column does not have the unit |
| convert.int8.quantize.a | 2.5305 | 2.5272 | 0.8287 Gelem/s | 0.680 | per-call |
| convert.int8.pack.b | 2.5458 | 2.5362 | 25.7940 Gelem/s | 0.684 | once-per-weight |
| convert.bf16.pack.b | 0.2200 | 0.2182 | 298.5437 Gelem/s | 0.059 | once-per-weight |
| convert.bf16.widen.b | 0.2086 | 0.2081 | 314.7281 Gelem/s | 0.056 | per-call-when-materialized |
| convert.int8.dequantize.b | 0.2723 | 0.2714 | 241.1695 Gelem/s | 0.073 | per-call-when-materialized |
| DERIVED int8i32.v1.flat + quantize.a | 154.4235 | | | 41.472 | sum of two medians |
| DERIVED int8i32.v1.mma + quantize.a | 7.6525 | | | 2.055 | sum of two medians |
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

## Where a low-bit plan costs more time than fp32.v1

| box | shape | arm | arm ms | fp32.v1 ms | arm over fp32.v1 | note |
|---|---|---|---:|---:|---:|---|
| h100 | llama8b.qkv.t1 | bf16f32.v1.fused | 0.7874 | 0.0704 | 11.185 | dispatched |
| h100 | llama8b.qkv.t1 | bf16f32.v1.widen | 0.1210 | 0.0704 | 1.719 | not-dispatched(cells<=16384) |
| h100 | llama8b.qkv.t1 | int8i32.v1.flat | 0.6643 | 0.0704 | 9.436 | not-dispatched(the-column-has-the-unit) |
| h100 | llama8b.qkv.t1 | int8i32.v1.mma | 0.1574 | 0.0704 | 2.236 | dispatched |
| h100 | llama8b.qkv.t1 | int8i32.v1.flat + quantize.a (DERIVED) | 1.1522 | 0.0704 | 16.366 | not-dispatched(the-column-has-the-unit) |
| h100 | llama8b.qkv.t1 | int8i32.v1.mma + quantize.a (DERIVED) | 0.6453 | 0.0704 | 9.166 | dispatched |
| h100 | llama8b.qkv.t8 | bf16f32.v1.fused | 0.7795 | 0.1006 | 7.749 | not-dispatched(cells>16384) |
| h100 | llama8b.qkv.t8 | bf16f32.v1.widen | 0.1507 | 0.1006 | 1.498 | dispatched |
| h100 | llama8b.qkv.t8 | int8i32.v1.flat | 0.6628 | 0.1006 | 6.588 | not-dispatched(the-column-has-the-unit) |
| h100 | llama8b.qkv.t8 | int8i32.v1.mma | 0.1471 | 0.1006 | 1.462 | dispatched |
| h100 | llama8b.qkv.t8 | int8i32.v1.flat + quantize.a (DERIVED) | 1.2587 | 0.1006 | 12.512 | not-dispatched(the-column-has-the-unit) |
| h100 | llama8b.qkv.t8 | int8i32.v1.mma + quantize.a (DERIVED) | 0.7430 | 0.1006 | 7.386 | dispatched |
| h100 | llama8b.qkv.t512 | bf16f32.v1.fused | 39.4126 | 1.0341 | 38.113 | not-dispatched(cells>16384) |
| h100 | llama8b.qkv.t512 | bf16f32.v1.widen | 1.0941 | 1.0341 | 1.058 | dispatched |
| h100 | llama8b.qkv.t512 | int8i32.v1.flat | 39.1804 | 1.0341 | 37.888 | not-dispatched(the-column-has-the-unit) |
| h100 | llama8b.qkv.t512 | int8i32.v1.mma | 1.4211 | 1.0341 | 1.374 | dispatched |
| h100 | llama8b.qkv.t512 | int8i32.v1.flat + quantize.a (DERIVED) | 41.6509 | 1.0341 | 40.277 | not-dispatched(the-column-has-the-unit) |
| h100 | llama8b.qkv.t512 | int8i32.v1.mma + quantize.a (DERIVED) | 3.8916 | 1.0341 | 3.763 | dispatched |
| h100 | llama8b.mlp_up.t1 | bf16f32.v1.fused | 0.7926 | 0.1872 | 4.234 | dispatched |
| h100 | llama8b.mlp_up.t1 | bf16f32.v1.widen | 0.3646 | 0.1872 | 1.948 | not-dispatched(cells<=16384) |
| h100 | llama8b.mlp_up.t1 | int8i32.v1.flat | 0.6645 | 0.1872 | 3.550 | not-dispatched(the-column-has-the-unit) |
| h100 | llama8b.mlp_up.t1 | int8i32.v1.mma | 0.1926 | 0.1872 | 1.029 | dispatched |
| h100 | llama8b.mlp_up.t1 | int8i32.v1.flat + quantize.a (DERIVED) | 1.1650 | 0.1872 | 6.223 | not-dispatched(the-column-has-the-unit) |
| h100 | llama8b.mlp_up.t1 | int8i32.v1.mma + quantize.a (DERIVED) | 0.6931 | 0.1872 | 3.702 | dispatched |
| h100 | llama8b.mlp_up.t8 | bf16f32.v1.fused | 2.5292 | 0.2713 | 9.323 | not-dispatched(cells>16384) |
| h100 | llama8b.mlp_up.t8 | bf16f32.v1.widen | 0.4528 | 0.2713 | 1.669 | dispatched |
| h100 | llama8b.mlp_up.t8 | int8i32.v1.flat | 2.5016 | 0.2713 | 9.221 | not-dispatched(the-column-has-the-unit) |
| h100 | llama8b.mlp_up.t8 | int8i32.v1.flat + quantize.a (DERIVED) | 3.1125 | 0.2713 | 11.473 | not-dispatched(the-column-has-the-unit) |
| h100 | llama8b.mlp_up.t8 | int8i32.v1.mma + quantize.a (DERIVED) | 0.8194 | 0.2713 | 3.020 | dispatched |
| h100 | llama8b.mlp_up.t512 | bf16f32.v1.fused | 139.1256 | 3.3454 | 41.587 | not-dispatched(cells>16384) |
| h100 | llama8b.mlp_up.t512 | bf16f32.v1.widen | 3.5246 | 3.3454 | 1.054 | dispatched |
| h100 | llama8b.mlp_up.t512 | int8i32.v1.flat | 135.6717 | 3.3454 | 40.555 | not-dispatched(the-column-has-the-unit) |
| h100 | llama8b.mlp_up.t512 | int8i32.v1.mma | 4.6375 | 3.3454 | 1.386 | dispatched |
| h100 | llama8b.mlp_up.t512 | int8i32.v1.flat + quantize.a (DERIVED) | 138.1420 | 3.3454 | 41.293 | not-dispatched(the-column-has-the-unit) |
| h100 | llama8b.mlp_up.t512 | int8i32.v1.mma + quantize.a (DERIVED) | 7.1078 | 3.3454 | 2.125 | dispatched |
| h100 | llama8b.mlp_down.t1 | bf16f32.v1.fused | 2.7594 | 0.1782 | 15.485 | dispatched |
| h100 | llama8b.mlp_down.t1 | bf16f32.v1.widen | 0.3558 | 0.1782 | 1.997 | not-dispatched(cells<=16384) |
| h100 | llama8b.mlp_down.t1 | int8i32.v1.flat | 2.3432 | 0.1782 | 13.149 | not-dispatched(the-column-has-the-unit) |
| h100 | llama8b.mlp_down.t1 | int8i32.v1.mma | 0.6522 | 0.1782 | 3.660 | dispatched |
| h100 | llama8b.mlp_down.t1 | int8i32.v1.flat + quantize.a (DERIVED) | 4.0782 | 0.1782 | 22.886 | not-dispatched(the-column-has-the-unit) |
| h100 | llama8b.mlp_down.t1 | int8i32.v1.mma + quantize.a (DERIVED) | 2.3872 | 0.1782 | 13.396 | dispatched |
| h100 | llama8b.mlp_down.t8 | bf16f32.v1.fused | 2.7010 | 0.2877 | 9.388 | not-dispatched(cells>16384) |
| h100 | llama8b.mlp_down.t8 | bf16f32.v1.widen | 0.4663 | 0.2877 | 1.621 | dispatched |
| h100 | llama8b.mlp_down.t8 | int8i32.v1.flat | 2.3437 | 0.2877 | 8.146 | not-dispatched(the-column-has-the-unit) |
| h100 | llama8b.mlp_down.t8 | int8i32.v1.mma | 0.7006 | 0.2877 | 2.435 | dispatched |
| h100 | llama8b.mlp_down.t8 | int8i32.v1.flat + quantize.a (DERIVED) | 4.7233 | 0.2877 | 16.417 | not-dispatched(the-column-has-the-unit) |
| h100 | llama8b.mlp_down.t8 | int8i32.v1.mma + quantize.a (DERIVED) | 3.0802 | 0.2877 | 10.706 | dispatched |
| h100 | llama8b.mlp_down.t512 | bf16f32.v1.fused | 136.6104 | 3.3604 | 40.653 | not-dispatched(cells>16384) |
| h100 | llama8b.mlp_down.t512 | bf16f32.v1.widen | 3.5585 | 3.3604 | 1.059 | dispatched |
| h100 | llama8b.mlp_down.t512 | int8i32.v1.flat | 137.9490 | 3.3604 | 41.051 | not-dispatched(the-column-has-the-unit) |
| h100 | llama8b.mlp_down.t512 | int8i32.v1.mma | 5.5686 | 3.3604 | 1.657 | dispatched |
| h100 | llama8b.mlp_down.t512 | int8i32.v1.flat + quantize.a (DERIVED) | 147.9095 | 3.3604 | 44.015 | not-dispatched(the-column-has-the-unit) |
| h100 | llama8b.mlp_down.t512 | int8i32.v1.mma + quantize.a (DERIVED) | 15.5291 | 3.3604 | 4.621 | dispatched |
| h100 | llama8b.lm_head.t1 | bf16f32.v1.fused | 2.5785 | 1.2058 | 2.138 | not-dispatched(cells>16384) |
| h100 | llama8b.lm_head.t1 | bf16f32.v1.widen | 2.8046 | 1.2058 | 2.326 | dispatched |
| h100 | llama8b.lm_head.t1 | int8i32.v1.flat | 2.4953 | 1.2058 | 2.069 | not-dispatched(the-column-has-the-unit) |
| h100 | llama8b.lm_head.t1 | int8i32.v1.flat + quantize.a (DERIVED) | 2.9974 | 1.2058 | 2.486 | not-dispatched(the-column-has-the-unit) |
| h100 | llama8b.lm_head.t1 | int8i32.v1.mma + quantize.a (DERIVED) | 1.4612 | 1.2058 | 1.212 | dispatched |
| h100 | llama8b.lm_head.t8 | bf16f32.v1.fused | 19.4619 | 2.1537 | 9.036 | not-dispatched(cells>16384) |
| h100 | llama8b.lm_head.t8 | bf16f32.v1.widen | 3.7583 | 2.1537 | 1.745 | dispatched |
| h100 | llama8b.lm_head.t8 | int8i32.v1.flat | 19.2647 | 2.1537 | 8.945 | not-dispatched(the-column-has-the-unit) |
| h100 | llama8b.lm_head.t8 | int8i32.v1.flat + quantize.a (DERIVED) | 19.8737 | 2.1537 | 9.228 | not-dispatched(the-column-has-the-unit) |
| h100 | llama8b.lm_head.t512 | bf16f32.v1.fused | 156.5389 | 3.7236 | 42.040 | not-dispatched(cells>16384) |
| h100 | llama8b.lm_head.t512 | bf16f32.v1.widen | 3.9218 | 3.7236 | 1.053 | dispatched |
| h100 | llama8b.lm_head.t512 | int8i32.v1.flat | 151.8930 | 3.7236 | 40.792 | not-dispatched(the-column-has-the-unit) |
| h100 | llama8b.lm_head.t512 | int8i32.v1.mma | 5.1220 | 3.7236 | 1.376 | dispatched |
| h100 | llama8b.lm_head.t512 | int8i32.v1.flat + quantize.a (DERIVED) | 154.4235 | 3.7236 | 41.472 | not-dispatched(the-column-has-the-unit) |
| h100 | llama8b.lm_head.t512 | int8i32.v1.mma + quantize.a (DERIVED) | 7.6525 | 3.7236 | 2.055 | dispatched |

## Digests across the boxes

AGREE: every box that ran the arm at the same extents printed the same digest, and at least two did.
ONE BOX: nothing to compare, which is not a pass.

| shape | arm | verdict | h100 | mi325x |
|---|---|---|---|---|
| llama8b.qkv.t1 | fp32.v1 | AGREE | 0x920cdd138650a8bd | 0x920cdd138650a8bd |
| llama8b.qkv.t1 | bf16f32.v1.fused | AGREE | 0xcd39e9d6e6d7179a | 0xcd39e9d6e6d7179a |
| llama8b.qkv.t1 | bf16f32.v1.widen | AGREE | 0xcd39e9d6e6d7179a | 0xcd39e9d6e6d7179a |
| llama8b.qkv.t1 | int8i32.v1.flat | AGREE | 0x8a9227b93e430acd | 0x8a9227b93e430acd |
| llama8b.qkv.t1 | int8i32.v1.mma | AGREE | 0x8a9227b93e430acd | 0x8a9227b93e430acd |
| llama8b.qkv.t1 | convert.int8.quantize.a | AGREE | 0x54279cbda1444726 | 0x54279cbda1444726 |
| llama8b.qkv.t1 | convert.int8.pack.b | AGREE | 0x305b1a502296caf6 | 0x305b1a502296caf6 |
| llama8b.qkv.t1 | convert.bf16.pack.b | AGREE | 0x24efed638bb6b96b | 0x24efed638bb6b96b |
| llama8b.qkv.t1 | convert.bf16.widen.b | AGREE | 0xd2c2d96392622325 | 0xd2c2d96392622325 |
| llama8b.qkv.t1 | convert.int8.dequantize.b | AGREE | 0x2fd9ee536fd82325 | 0x2fd9ee536fd82325 |
| llama8b.qkv.t8 | fp32.v1 | AGREE | 0xd5677d8a13939709 | 0xd5677d8a13939709 |
| llama8b.qkv.t8 | bf16f32.v1.fused | AGREE | 0xf36fe1a8828cefd9 | 0xf36fe1a8828cefd9 |
| llama8b.qkv.t8 | bf16f32.v1.widen | AGREE | 0xf36fe1a8828cefd9 | 0xf36fe1a8828cefd9 |
| llama8b.qkv.t8 | int8i32.v1.flat | AGREE | 0x484bd20aabe66ee5 | 0x484bd20aabe66ee5 |
| llama8b.qkv.t8 | int8i32.v1.mma | AGREE | 0x484bd20aabe66ee5 | 0x484bd20aabe66ee5 |
| llama8b.qkv.t8 | convert.int8.quantize.a | AGREE | 0xbd0b5aed2b83cb45 | 0xbd0b5aed2b83cb45 |
| llama8b.qkv.t8 | convert.int8.pack.b | AGREE | 0xa8ce17326c01345 | 0xa8ce17326c01345 |
| llama8b.qkv.t8 | convert.bf16.pack.b | AGREE | 0xdccc5a093a080c52 | 0xdccc5a093a080c52 |
| llama8b.qkv.t8 | convert.bf16.widen.b | AGREE | 0x944a5da41b4d2325 | 0x944a5da41b4d2325 |
| llama8b.qkv.t8 | convert.int8.dequantize.b | AGREE | 0x24697157b3da2325 | 0x24697157b3da2325 |
| llama8b.qkv.t512 | fp32.v1 | AGREE | 0x67d3d940546b834c | 0x67d3d940546b834c |
| llama8b.qkv.t512 | bf16f32.v1.fused | AGREE | 0x4abd79a5b10221e5 | 0x4abd79a5b10221e5 |
| llama8b.qkv.t512 | bf16f32.v1.widen | AGREE | 0x4abd79a5b10221e5 | 0x4abd79a5b10221e5 |
| llama8b.qkv.t512 | int8i32.v1.flat | AGREE | 0xa5f2ce1f1b9a0ebd | 0xa5f2ce1f1b9a0ebd |
| llama8b.qkv.t512 | int8i32.v1.mma | AGREE | 0xa5f2ce1f1b9a0ebd | 0xa5f2ce1f1b9a0ebd |
| llama8b.qkv.t512 | convert.int8.quantize.a | AGREE | 0xe8bb3ad105a4ae26 | 0xe8bb3ad105a4ae26 |
| llama8b.qkv.t512 | convert.int8.pack.b | AGREE | 0xd5ea46fdcb251db4 | 0xd5ea46fdcb251db4 |
| llama8b.qkv.t512 | convert.bf16.pack.b | AGREE | 0xaef2da473a8e2a10 | 0xaef2da473a8e2a10 |
| llama8b.qkv.t512 | convert.bf16.widen.b | AGREE | 0xc8e5243510392325 | 0xc8e5243510392325 |
| llama8b.qkv.t512 | convert.int8.dequantize.b | AGREE | 0x1798a32eac742325 | 0x1798a32eac742325 |
| llama8b.mlp_up.t1 | fp32.v1 | AGREE | 0x4965ea6af2ff5e61 | 0x4965ea6af2ff5e61 |
| llama8b.mlp_up.t1 | bf16f32.v1.fused | AGREE | 0x5e3f5ff35a448024 | 0x5e3f5ff35a448024 |
| llama8b.mlp_up.t1 | bf16f32.v1.widen | AGREE | 0x5e3f5ff35a448024 | 0x5e3f5ff35a448024 |
| llama8b.mlp_up.t1 | int8i32.v1.flat | AGREE | 0xff302668595b02d | 0xff302668595b02d |
| llama8b.mlp_up.t1 | int8i32.v1.mma | AGREE | 0xff302668595b02d | 0xff302668595b02d |
| llama8b.mlp_up.t1 | convert.int8.quantize.a | AGREE | 0x1343c750c1ded02 | 0x1343c750c1ded02 |
| llama8b.mlp_up.t1 | convert.int8.pack.b | AGREE | 0x9dd5dd0e7f10b4d0 | 0x9dd5dd0e7f10b4d0 |
| llama8b.mlp_up.t1 | convert.bf16.pack.b | AGREE | 0xed618e2333b13294 | 0xed618e2333b13294 |
| llama8b.mlp_up.t1 | convert.bf16.widen.b | AGREE | 0x573b3a44b4012325 | 0x573b3a44b4012325 |
| llama8b.mlp_up.t1 | convert.int8.dequantize.b | AGREE | 0x5a7e8639220c2325 | 0x5a7e8639220c2325 |
| llama8b.mlp_up.t8 | fp32.v1 | AGREE | 0x7b30465d325906e | 0x7b30465d325906e |
| llama8b.mlp_up.t8 | bf16f32.v1.fused | AGREE | 0x1531c10dcd152579 | 0x1531c10dcd152579 |
| llama8b.mlp_up.t8 | bf16f32.v1.widen | AGREE | 0x1531c10dcd152579 | 0x1531c10dcd152579 |
| llama8b.mlp_up.t8 | int8i32.v1.flat | AGREE | 0xbd4b3a783f9d9a0d | 0xbd4b3a783f9d9a0d |
| llama8b.mlp_up.t8 | int8i32.v1.mma | AGREE | 0xbd4b3a783f9d9a0d | 0xbd4b3a783f9d9a0d |
| llama8b.mlp_up.t8 | convert.int8.quantize.a | AGREE | 0x5e46321f760ef85d | 0x5e46321f760ef85d |
| llama8b.mlp_up.t8 | convert.int8.pack.b | AGREE | 0xa9e079afde494e3d | 0xa9e079afde494e3d |
| llama8b.mlp_up.t8 | convert.bf16.pack.b | AGREE | 0xc5111a6224412c3f | 0xc5111a6224412c3f |
| llama8b.mlp_up.t8 | convert.bf16.widen.b | AGREE | 0x6a679200ae02325 | 0x6a679200ae02325 |
| llama8b.mlp_up.t8 | convert.int8.dequantize.b | AGREE | 0x89939dca58a02325 | 0x89939dca58a02325 |
| llama8b.mlp_up.t512 | fp32.v1 | AGREE | 0x27a166783c45f9f0 | 0x27a166783c45f9f0 |
| llama8b.mlp_up.t512 | bf16f32.v1.fused | AGREE | 0x5b29318ad80be465 | 0x5b29318ad80be465 |
| llama8b.mlp_up.t512 | bf16f32.v1.widen | AGREE | 0x5b29318ad80be465 | 0x5b29318ad80be465 |
| llama8b.mlp_up.t512 | int8i32.v1.flat | AGREE | 0x220ff5fe96f5ef55 | 0x220ff5fe96f5ef55 |
| llama8b.mlp_up.t512 | int8i32.v1.mma | AGREE | 0x220ff5fe96f5ef55 | 0x220ff5fe96f5ef55 |
| llama8b.mlp_up.t512 | convert.int8.quantize.a | AGREE | 0x8c503fb082856006 | 0x8c503fb082856006 |
| llama8b.mlp_up.t512 | convert.int8.pack.b | AGREE | 0xf8c1486c6aaba4c1 | 0xf8c1486c6aaba4c1 |
| llama8b.mlp_up.t512 | convert.bf16.pack.b | AGREE | 0x1c04d570a3d23380 | 0x1c04d570a3d23380 |
| llama8b.mlp_up.t512 | convert.bf16.widen.b | AGREE | 0x5c9d5dc97a412325 | 0x5c9d5dc97a412325 |
| llama8b.mlp_up.t512 | convert.int8.dequantize.b | AGREE | 0xb584f740e2d62325 | 0xb584f740e2d62325 |
| llama8b.mlp_down.t1 | fp32.v1 | AGREE | 0x4efb8e54c724fa9 | 0x4efb8e54c724fa9 |
| llama8b.mlp_down.t1 | bf16f32.v1.fused | AGREE | 0xd044aebb510dc9 | 0xd044aebb510dc9 |
| llama8b.mlp_down.t1 | bf16f32.v1.widen | AGREE | 0xd044aebb510dc9 | 0xd044aebb510dc9 |
| llama8b.mlp_down.t1 | int8i32.v1.flat | AGREE | 0x1ffdde3ffffb72c9 | 0x1ffdde3ffffb72c9 |
| llama8b.mlp_down.t1 | int8i32.v1.mma | AGREE | 0x1ffdde3ffffb72c9 | 0x1ffdde3ffffb72c9 |
| llama8b.mlp_down.t1 | convert.int8.quantize.a | AGREE | 0x160d734878c0f458 | 0x160d734878c0f458 |
| llama8b.mlp_down.t1 | convert.int8.pack.b | AGREE | 0x31274090cc79ba2f | 0x31274090cc79ba2f |
| llama8b.mlp_down.t1 | convert.bf16.pack.b | AGREE | 0x9dba12f4fe0c9e9b | 0x9dba12f4fe0c9e9b |
| llama8b.mlp_down.t1 | convert.bf16.widen.b | AGREE | 0x45255120c9d82325 | 0x45255120c9d82325 |
| llama8b.mlp_down.t1 | convert.int8.dequantize.b | AGREE | 0x8e72f1c5a3402325 | 0x8e72f1c5a3402325 |
| llama8b.mlp_down.t8 | fp32.v1 | AGREE | 0xfb21004541ae3e8e | 0xfb21004541ae3e8e |
| llama8b.mlp_down.t8 | bf16f32.v1.fused | AGREE | 0xcbe807d3ae13612f | 0xcbe807d3ae13612f |
| llama8b.mlp_down.t8 | bf16f32.v1.widen | AGREE | 0xcbe807d3ae13612f | 0xcbe807d3ae13612f |
| llama8b.mlp_down.t8 | int8i32.v1.flat | AGREE | 0x694aafd520fdadad | 0x694aafd520fdadad |
| llama8b.mlp_down.t8 | int8i32.v1.mma | AGREE | 0x694aafd520fdadad | 0x694aafd520fdadad |
| llama8b.mlp_down.t8 | convert.int8.quantize.a | AGREE | 0xbb4f2a7232392bfe | 0xbb4f2a7232392bfe |
| llama8b.mlp_down.t8 | convert.int8.pack.b | AGREE | 0x4b12dcda2ab9acd4 | 0x4b12dcda2ab9acd4 |
| llama8b.mlp_down.t8 | convert.bf16.pack.b | AGREE | 0xfbacba127cf6177e | 0xfbacba127cf6177e |
| llama8b.mlp_down.t8 | convert.bf16.widen.b | AGREE | 0xcbcb5933d8352325 | 0xcbcb5933d8352325 |
| llama8b.mlp_down.t8 | convert.int8.dequantize.b | AGREE | 0x9dcad54d7e462325 | 0x9dcad54d7e462325 |
| llama8b.mlp_down.t512 | fp32.v1 | AGREE | 0x4743c97cb961339e | 0x4743c97cb961339e |
| llama8b.mlp_down.t512 | bf16f32.v1.fused | AGREE | 0x508d64e0e41d3350 | 0x508d64e0e41d3350 |
| llama8b.mlp_down.t512 | bf16f32.v1.widen | AGREE | 0x508d64e0e41d3350 | 0x508d64e0e41d3350 |
| llama8b.mlp_down.t512 | int8i32.v1.flat | AGREE | 0x6120545eacb068ad | 0x6120545eacb068ad |
| llama8b.mlp_down.t512 | int8i32.v1.mma | AGREE | 0x6120545eacb068ad | 0x6120545eacb068ad |
| llama8b.mlp_down.t512 | convert.int8.quantize.a | AGREE | 0x903f6ff84c6082 | 0x903f6ff84c6082 |
| llama8b.mlp_down.t512 | convert.int8.pack.b | AGREE | 0xa046020a81d05746 | 0xa046020a81d05746 |
| llama8b.mlp_down.t512 | convert.bf16.pack.b | AGREE | 0xd1b066f4e6a61fed | 0xd1b066f4e6a61fed |
| llama8b.mlp_down.t512 | convert.bf16.widen.b | AGREE | 0xf0fcf217ba5c2325 | 0xf0fcf217ba5c2325 |
| llama8b.mlp_down.t512 | convert.int8.dequantize.b | AGREE | 0xd972f24f17be2325 | 0xd972f24f17be2325 |
| llama8b.lm_head.t1 | fp32.v1 | AGREE | 0xcdc134cbce391535 | 0xcdc134cbce391535 |
| llama8b.lm_head.t1 | bf16f32.v1.fused | AGREE | 0x23f2e7e17341274a | 0x23f2e7e17341274a |
| llama8b.lm_head.t1 | bf16f32.v1.widen | AGREE | 0x23f2e7e17341274a | 0x23f2e7e17341274a |
| llama8b.lm_head.t1 | int8i32.v1.flat | AGREE | 0xe0222b2f9de835ed | 0xe0222b2f9de835ed |
| llama8b.lm_head.t1 | int8i32.v1.mma | AGREE | 0xe0222b2f9de835ed | 0xe0222b2f9de835ed |
| llama8b.lm_head.t1 | convert.int8.quantize.a | AGREE | 0x67d2e298b047bf2b | 0x67d2e298b047bf2b |
| llama8b.lm_head.t1 | convert.int8.pack.b | AGREE | 0x1c75800d0b1054bd | 0x1c75800d0b1054bd |
| llama8b.lm_head.t1 | convert.bf16.pack.b | AGREE | 0xa3f53235cc152516 | 0xa3f53235cc152516 |
| llama8b.lm_head.t1 | convert.bf16.widen.b | AGREE | 0x63450213e8e12325 | 0x63450213e8e12325 |
| llama8b.lm_head.t1 | convert.int8.dequantize.b | AGREE | 0x23115689185e2325 | 0x23115689185e2325 |
| llama8b.lm_head.t8 | fp32.v1 | AGREE | 0x3ebab59d0da21444 | 0x3ebab59d0da21444 |
| llama8b.lm_head.t8 | bf16f32.v1.fused | AGREE | 0x82bad7b4e5c4c263 | 0x82bad7b4e5c4c263 |
| llama8b.lm_head.t8 | bf16f32.v1.widen | AGREE | 0x82bad7b4e5c4c263 | 0x82bad7b4e5c4c263 |
| llama8b.lm_head.t8 | int8i32.v1.flat | AGREE | 0x8151fd05ff287a5d | 0x8151fd05ff287a5d |
| llama8b.lm_head.t8 | int8i32.v1.mma | AGREE | 0x8151fd05ff287a5d | 0x8151fd05ff287a5d |
| llama8b.lm_head.t8 | convert.int8.quantize.a | AGREE | 0x443be8ad16d189a3 | 0x443be8ad16d189a3 |
| llama8b.lm_head.t8 | convert.int8.pack.b | AGREE | 0x8e01af3f075298bb | 0x8e01af3f075298bb |
| llama8b.lm_head.t8 | convert.bf16.pack.b | AGREE | 0xbdac0264c773e6f1 | 0xbdac0264c773e6f1 |
| llama8b.lm_head.t8 | convert.bf16.widen.b | AGREE | 0x4e388bd358a22325 | 0x4e388bd358a22325 |
| llama8b.lm_head.t8 | convert.int8.dequantize.b | AGREE | 0xb8835a7fc2342325 | 0xb8835a7fc2342325 |
| llama8b.lm_head.t512 | fp32.v1 | AGREE | 0x84fd143bb625cf70 | 0x84fd143bb625cf70 |
| llama8b.lm_head.t512 | bf16f32.v1.fused | AGREE | 0xa75052ecc1e78b46 | 0xa75052ecc1e78b46 |
| llama8b.lm_head.t512 | bf16f32.v1.widen | AGREE | 0xa75052ecc1e78b46 | 0xa75052ecc1e78b46 |
| llama8b.lm_head.t512 | int8i32.v1.flat | AGREE | 0xaae4f3d3c69453cd | 0xaae4f3d3c69453cd |
| llama8b.lm_head.t512 | int8i32.v1.mma | AGREE | 0xaae4f3d3c69453cd | 0xaae4f3d3c69453cd |
| llama8b.lm_head.t512 | convert.int8.quantize.a | AGREE | 0xfd89ee79d025dda7 | 0xfd89ee79d025dda7 |
| llama8b.lm_head.t512 | convert.int8.pack.b | AGREE | 0xcc187b17bc36366d | 0xcc187b17bc36366d |
| llama8b.lm_head.t512 | convert.bf16.pack.b | AGREE | 0xaabc0c6ba2a73e85 | 0xaabc0c6ba2a73e85 |
| llama8b.lm_head.t512 | convert.bf16.widen.b | AGREE | 0xbd6a89a89fc02325 | 0xbd6a89a89fc02325 |
| llama8b.lm_head.t512 | convert.int8.dequantize.b | AGREE | 0xf6a61b0da9142325 | 0xf6a61b0da9142325 |

| arm | shapes | AGREE | DISAGREE | ONE BOX | not comparable |
|---|---:|---:|---:|---:|---:|
| fp32.v1 | 12 | 12 | 0 | 0 | 0 |
| bf16f32.v1.fused | 12 | 12 | 0 | 0 | 0 |
| bf16f32.v1.widen | 12 | 12 | 0 | 0 | 0 |
| int8i32.v1.flat | 12 | 12 | 0 | 0 | 0 |
| int8i32.v1.mma | 12 | 12 | 0 | 0 | 0 |
| convert.int8.quantize.a | 12 | 12 | 0 | 0 | 0 |
| convert.int8.pack.b | 12 | 12 | 0 | 0 | 0 |
| convert.bf16.pack.b | 12 | 12 | 0 | 0 | 0 |
| convert.bf16.widen.b | 12 | 12 | 0 | 0 | 0 |
| convert.int8.dequantize.b | 12 | 12 | 0 | 0 | 0 |

