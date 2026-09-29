# Low-bit GEMM plans: times, rates and digests

Written by `tools/lowbit_units/table.py` from the `LOWBIT` lines of
`bench/gemm_lowbit_price_main.mojo`. Every time is one box's time on one run: one call and one
synchronize per sample, the median of the timed calls, the minimum beside it. Every ratio is a time
over a time at the same shape on the same box; above 1 the numerator took longer. Rates are
G MAC/s (multiply-accumulates, `m n k`) for a product and G elem/s for a conversion.

## INFERENCE: the complete operation against fp32.v1

The weights were packed once. One call pays the conversion of its activations, the product on the plan the profile's dispatcher picks, and the epilogue. bf16f32.v1 keeps its activations float32. Every step of the call is enqueued and waited for ONCE, so each time is measured, not a
sum. `over` is the time over fp32.v1's at the same shape on the same box; above 1 it took longer.
The Apple probe is not in any dispatcher.

### h100

Run 3, nvc3-0010, 2026-09-29T03:07:48Z to 03:09:08Z, lane commit d4948864a. TAKEN ALONE: the binary was built and run once by an earlier job; this lane ran nothing else on the box from the submit to the end of the job. The pod is shared with other lanes, whose work this lane cannot see.

| shape | m x n x k | fp32.v1 ms | bf16f32.v1 ms | over | int8i32.v1 ms | over | Apple probe ms | over |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| llama8b.qkv.t1 | 1 x 4096 x 4096 | 0.0737 | 0.7976 | 10.822 | 0.6420 | 8.711 | not run |  |
| llama8b.qkv.t8 | 8 x 4096 x 4096 | 0.1074 | 0.1529 | 1.424 | 0.7706 | 7.175 | not run |  |
| llama8b.qkv.t512 | 512 x 4096 x 4096 | 1.0377 | 1.0943 | 1.055 | 4.0802 | 3.932 | not run |  |
| llama8b.mlp_up.t1 | 1 x 14336 x 4096 | 0.1884 | 0.7966 | 4.228 | 0.6701 | 3.557 | not run |  |
| llama8b.mlp_up.t8 | 8 x 14336 x 4096 | 0.2757 | 0.4535 | 1.645 | 0.8027 | 2.911 | not run |  |
| llama8b.mlp_up.t512 | 512 x 14336 x 4096 | 3.3462 | 3.5188 | 1.052 | 7.0929 | 2.120 | not run |  |
| llama8b.mlp_down.t1 | 1 x 4096 x 14336 | 0.1773 | 2.7606 | 15.570 | 2.3209 | 13.090 | not run |  |
| llama8b.mlp_down.t8 | 8 x 4096 x 14336 | 0.2918 | 0.4693 | 1.608 | 3.0493 | 10.450 | not run |  |
| llama8b.mlp_down.t512 | 512 x 4096 x 14336 | 3.3696 | 3.5462 | 1.052 | 15.5603 | 4.618 | not run |  |
| llama8b.lm_head.t1 | 1 x 128256 x 4096 | 1.2049 | 2.8132 | 2.335 | 1.4608 | 1.212 | not run |  |
| llama8b.lm_head.t8 | 8 x 128256 x 4096 | 2.1659 | 3.7630 | 1.737 | 1.9728 | 0.911 | not run |  |
| llama8b.lm_head.t512 | 512 x 16032 x 4096 (CAPPED) | 3.7367 | 3.9246 | 1.050 | 7.5827 | 2.029 | not run |  |

### m3ultra

Run 1 on this box, steward request 1790651983032, timing 2026-09-29T03:21:43Z to 03:23:20Z, commit f1e6cb61c. TAKEN ALONE as a steward speed job (the steward times beside no other job), after the gate and the probe's gate had run in the same job; this lane made no contact with the box from 03:19:49Z to 03:40:22Z. The harness was compiled in this run; every arm has an untimed warm-up call before its timed ones.

| shape | m x n x k | fp32.v1 ms | bf16f32.v1 ms | over | int8i32.v1 ms | over | Apple probe ms | over |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| llama8b.qkv.t1 | 1 x 4096 x 4096 | 1.0240 | 0.9520 | 0.930 | 1.5200 | 1.484 | 1.5390 | 1.503 |
| llama8b.qkv.t8 | 8 x 4096 x 4096 | 0.5420 | 1.1190 | 2.065 | 1.9230 | 3.548 | 1.6450 | 3.035 |
| llama8b.qkv.t512 | 512 x 4096 x 4096 | 3.3200 | 3.8350 | 1.155 | 27.7370 | 8.355 | 3.3780 | 1.017 |
| llama8b.mlp_up.t1 | 1 x 14336 x 4096 | 1.0440 | 0.9630 | 0.922 | 1.5540 | 1.489 | 1.6110 | 1.543 |
| llama8b.mlp_up.t8 | 8 x 14336 x 4096 | 1.0200 | 2.9900 | 2.931 | 3.0540 | 2.994 | 1.7060 | 1.673 |
| llama8b.mlp_up.t512 | 512 x 14336 x 4096 | 10.4170 | 12.3630 | 1.187 | 92.3240 | 8.863 | 7.2710 | 0.698 |
| llama8b.mlp_down.t1 | 1 x 4096 x 14336 | 2.8720 | 2.7100 | 0.944 | 4.8740 | 1.697 | 5.0030 | 1.742 |
| llama8b.mlp_down.t8 | 8 x 4096 x 14336 | 1.3680 | 3.3480 | 2.447 | 6.4440 | 4.711 | 5.4550 | 3.988 |
| llama8b.mlp_down.t512 | 512 x 4096 x 14336 | 11.4650 | 13.4130 | 1.170 | 122.2080 | 10.659 | 11.6260 | 1.014 |
| llama8b.lm_head.t1 | 1 x 128256 x 4096 | 4.0720 | 21.4680 | 5.272 | 3.1180 | 0.766 | 3.7660 | 0.925 |
| llama8b.lm_head.t8 | 8 x 128256 x 4096 | 5.7290 | 23.1400 | 4.039 | 14.3620 | 2.507 | 3.9500 | 0.689 |
| llama8b.lm_head.t512 | 512 x 16032 x 4096 (CAPPED) | 11.5820 | 13.7660 | 1.189 | 103.8610 | 8.967 | 7.8480 | 0.678 |

### m2pro

Run 3, steward request 1790651263666, timing 2026-09-29T03:08:10Z to 03:12:37Z, commit d4948864a. TAKEN ALONE as a steward speed job; this lane made no contact with the box from the submit to 03:19:03Z.

| shape | m x n x k | fp32.v1 ms | bf16f32.v1 ms | over | int8i32.v1 ms | over | Apple probe ms | over |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| llama8b.qkv.t1 | 1 x 4096 x 4096 | 1.5170 | 1.4640 | 0.965 | 2.5170 | 1.659 | 1.9640 | 1.295 |
| llama8b.qkv.t8 | 8 x 4096 x 4096 | 1.4930 | 2.0460 | 1.370 | 10.8990 | 7.300 | 2.0120 | 1.348 |
| llama8b.qkv.t512 | 512 x 4096 x 4096 | 17.7380 | 18.1610 | 1.024 | 517.9090 | 29.198 | 11.8310 | 0.667 |
| llama8b.mlp_up.t1 | 1 x 14336 x 4096 | 5.2580 | 5.2180 | 0.992 | 6.1320 | 1.166 | 3.1930 | 0.607 |
| llama8b.mlp_up.t8 | 8 x 14336 x 4096 | 4.1060 | 6.0150 | 1.465 | 30.5820 | 7.448 | 3.2170 | 0.783 |
| llama8b.mlp_up.t512 | 512 x 14336 x 4096 | 60.5230 | 62.3810 | 1.031 | 1802.4750 | 29.782 | 31.1760 | 0.515 |
| llama8b.mlp_down.t1 | 1 x 4096 x 14336 | 5.5350 | 5.1410 | 0.929 | 8.4750 | 1.531 | 6.5430 | 1.182 |
| llama8b.mlp_down.t8 | 8 x 4096 x 14336 | 4.9780 | 6.9110 | 1.388 | 38.1740 | 7.669 | 6.6580 | 1.337 |
| llama8b.mlp_down.t512 | 512 x 4096 x 14336 | 65.5290 | 67.5340 | 1.031 | 1840.8580 | 28.092 | 41.7580 | 0.637 |
| llama8b.lm_head.t1 | 1 x 128256 x 4096 | 33.0710 | 50.1440 | 1.516 | 33.7790 | 1.021 | 13.0130 | 0.393 |
| llama8b.lm_head.t8 | 8 x 128256 x 4096 | 33.2280 | 50.2860 | 1.513 | 253.6220 | 7.633 | 13.0340 | 0.392 |
| llama8b.lm_head.t512 | 512 x 16032 x 4096 (CAPPED) | 67.5780 | 69.6380 | 1.030 | 2014.5080 | 29.810 | 34.5510 | 0.511 |

### mi325x

Run 4 (the run of record), steward request 1790651987201, timing 2026-09-29T03:20:01Z to 03:20:30Z, commit f1e6cb61c. TAKEN ALONE as a steward speed job, from a warm cache: the same binary had been built and run once, untimed, by request 1790651267504. This lane made no contact with the box from 03:19:49Z to 03:40:22Z. ONE GPU shared by every lane.

| shape | m x n x k | fp32.v1 ms | bf16f32.v1 ms | over | int8i32.v1 ms | over | Apple probe ms | over |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| llama8b.qkv.t1 | 1 x 4096 x 4096 | 0.0617 | 1.1249 | 18.232 | 1.3753 | 22.290 | not run |  |
| llama8b.qkv.t8 | 8 x 4096 x 4096 | 0.0775 | 0.1078 | 1.391 | 1.4746 | 19.027 | not run |  |
| llama8b.qkv.t512 | 512 x 4096 x 4096 | 0.9254 | 0.9480 | 1.024 | 4.0606 | 4.388 | not run |  |
| llama8b.mlp_up.t1 | 1 x 14336 x 4096 | 0.1631 | 1.1905 | 7.299 | 1.4112 | 8.652 | not run |  |
| llama8b.mlp_up.t8 | 8 x 14336 x 4096 | 0.1677 | 0.2920 | 1.741 | 1.5008 | 8.949 | not run |  |
| llama8b.mlp_up.t512 | 512 x 14336 x 4096 | 2.3895 | 2.5044 | 1.048 | 4.9238 | 2.061 | not run |  |
| llama8b.mlp_down.t1 | 1 x 4096 x 14336 | 0.1560 | 3.9026 | 25.017 | 4.8812 | 31.290 | not run |  |
| llama8b.mlp_down.t8 | 8 x 4096 x 14336 | 0.1947 | 0.3250 | 1.669 | 5.3521 | 27.489 | not run |  |
| llama8b.mlp_down.t512 | 512 x 4096 x 14336 | 2.5107 | 2.5862 | 1.030 | 14.7519 | 5.876 | not run |  |
| llama8b.lm_head.t1 | 1 x 128256 x 4096 | 1.3122 | 2.5466 | 1.941 | 1.6439 | 1.253 | not run |  |
| llama8b.lm_head.t8 | 8 x 128256 x 4096 | 1.2280 | 2.4984 | 2.035 | 1.9165 | 1.561 | not run |  |
| llama8b.lm_head.t512 | 512 x 16032 x 4096 (CAPPED) | 2.7776 | 2.9081 | 1.047 | 5.0582 | 1.821 | not run |  |

## TRAINING: the complete operation against fp32.v1

The weights change every step, so one step pays their conversion as well as the activations'. ONLY THE FORWARD PRODUCT: int8i32.v1 is OP_NT only and no low-bit backward kernel exists. Every step of the call is enqueued and waited for ONCE, so each time is measured, not a
sum. `over` is the time over fp32.v1's at the same shape on the same box; above 1 it took longer.
The Apple probe is not in any dispatcher.

### h100

Run 3, nvc3-0010, 2026-09-29T03:07:48Z to 03:09:08Z, lane commit d4948864a. TAKEN ALONE: the binary was built and run once by an earlier job; this lane ran nothing else on the box from the submit to the end of the job. The pod is shared with other lanes, whose work this lane cannot see.

| shape | m x n x k | fp32.v1 ms | bf16f32.v1 ms | over | int8i32.v1 ms | over | Apple probe ms | over |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| llama8b.qkv.t1 | 1 x 4096 x 4096 | 0.0737 | 0.8473 | 11.497 | 3.2892 | 44.630 | not run |  |
| llama8b.qkv.t8 | 8 x 4096 x 4096 | 0.1074 | 0.2084 | 1.940 | 3.3994 | 31.652 | not run |  |
| llama8b.qkv.t512 | 512 x 4096 x 4096 | 1.0377 | 1.1520 | 1.110 | 6.7240 | 6.480 | not run |  |
| llama8b.mlp_up.t1 | 1 x 14336 x 4096 | 0.1884 | 0.9845 | 5.226 | 3.3314 | 17.683 | not run |  |
| llama8b.mlp_up.t8 | 8 x 14336 x 4096 | 0.2757 | 0.6398 | 2.321 | 3.3082 | 11.999 | not run |  |
| llama8b.mlp_up.t512 | 512 x 14336 x 4096 | 3.3462 | 3.7078 | 1.108 | 9.6502 | 2.884 | not run |  |
| llama8b.mlp_down.t1 | 1 x 4096 x 14336 | 0.1773 | 2.9441 | 16.605 | 12.2797 | 69.259 | not run |  |
| llama8b.mlp_down.t8 | 8 x 4096 x 14336 | 0.2918 | 0.6561 | 2.248 | 13.0500 | 44.722 | not run |  |
| llama8b.mlp_down.t512 | 512 x 4096 x 14336 | 3.3696 | 3.7332 | 1.108 | 25.5078 | 7.570 | not run |  |
| llama8b.lm_head.t1 | 1 x 128256 x 4096 | 1.2049 | 4.4719 | 3.711 | 13.6141 | 11.299 | not run |  |
| llama8b.lm_head.t8 | 8 x 128256 x 4096 | 2.1659 | 5.4179 | 2.501 | 14.1975 | 6.555 | not run |  |
| llama8b.lm_head.t512 | 512 x 16032 x 4096 (CAPPED) | 3.7367 | 4.1322 | 1.106 | 10.1328 | 2.712 | not run |  |

### m3ultra

Run 1 on this box, steward request 1790651983032, timing 2026-09-29T03:21:43Z to 03:23:20Z, commit f1e6cb61c. TAKEN ALONE as a steward speed job (the steward times beside no other job), after the gate and the probe's gate had run in the same job; this lane made no contact with the box from 03:19:49Z to 03:40:22Z. The harness was compiled in this run; every arm has an untimed warm-up call before its timed ones.

| shape | m x n x k | fp32.v1 ms | bf16f32.v1 ms | over | int8i32.v1 ms | over | Apple probe ms | over |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| llama8b.qkv.t1 | 1 x 4096 x 4096 | 1.0240 | 1.5340 | 1.498 | 2.8160 | 2.750 | 2.8780 | 2.811 |
| llama8b.qkv.t8 | 8 x 4096 x 4096 | 0.5420 | 1.6640 | 3.070 | 3.2140 | 5.930 | 2.9790 | 5.496 |
| llama8b.qkv.t512 | 512 x 4096 x 4096 | 3.3200 | 4.4200 | 1.331 | 29.0150 | 8.739 | 4.6890 | 1.412 |
| llama8b.mlp_up.t1 | 1 x 14336 x 4096 | 1.0440 | 2.9450 | 2.821 | 2.9160 | 2.793 | 2.9750 | 2.850 |
| llama8b.mlp_up.t8 | 8 x 14336 x 4096 | 1.0200 | 4.9190 | 4.823 | 4.3630 | 4.277 | 3.0480 | 2.988 |
| llama8b.mlp_up.t512 | 512 x 14336 x 4096 | 10.4170 | 14.3630 | 1.379 | 93.6940 | 8.994 | 8.5710 | 0.823 |
| llama8b.mlp_down.t1 | 1 x 4096 x 14336 | 2.8720 | 4.6810 | 1.630 | 9.6470 | 3.359 | 9.8710 | 3.437 |
| llama8b.mlp_down.t8 | 8 x 4096 x 14336 | 1.3680 | 5.3270 | 3.894 | 11.3200 | 8.275 | 10.3290 | 7.550 |
| llama8b.mlp_down.t512 | 512 x 4096 x 14336 | 11.4650 | 15.1230 | 1.319 | 126.9410 | 11.072 | 16.5030 | 1.439 |
| llama8b.lm_head.t1 | 1 x 128256 x 4096 | 4.0720 | 38.9350 | 9.562 | 13.2120 | 3.245 | 13.9330 | 3.422 |
| llama8b.lm_head.t8 | 8 x 128256 x 4096 | 5.7290 | 40.6730 | 7.099 | 24.5000 | 4.276 | 13.9350 | 2.432 |
| llama8b.lm_head.t512 | 512 x 16032 x 4096 (CAPPED) | 11.5820 | 15.9530 | 1.377 | 105.4530 | 9.105 | 9.5210 | 0.822 |

### m2pro

Run 3, steward request 1790651263666, timing 2026-09-29T03:08:10Z to 03:12:37Z, commit d4948864a. TAKEN ALONE as a steward speed job; this lane made no contact with the box from the submit to 03:19:03Z.

| shape | m x n x k | fp32.v1 ms | bf16f32.v1 ms | over | int8i32.v1 ms | over | Apple probe ms | over |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| llama8b.qkv.t1 | 1 x 4096 x 4096 | 1.5170 | 2.3450 | 1.546 | 6.1170 | 4.032 | 5.5970 | 3.690 |
| llama8b.qkv.t8 | 8 x 4096 x 4096 | 1.4930 | 2.9140 | 1.952 | 14.5100 | 9.719 | 5.6440 | 3.780 |
| llama8b.qkv.t512 | 512 x 4096 x 4096 | 17.7380 | 19.0250 | 1.073 | 521.6280 | 29.407 | 15.4370 | 0.870 |
| llama8b.mlp_up.t1 | 1 x 14336 x 4096 | 5.2580 | 7.9600 | 1.514 | 19.5900 | 3.726 | 16.6470 | 3.166 |
| llama8b.mlp_up.t8 | 8 x 14336 x 4096 | 4.1060 | 8.7460 | 2.130 | 44.0230 | 10.722 | 16.6700 | 4.060 |
| llama8b.mlp_up.t512 | 512 x 14336 x 4096 | 60.5230 | 65.2750 | 1.079 | 1815.9380 | 30.004 | 44.6290 | 0.737 |
| llama8b.mlp_down.t1 | 1 x 4096 x 14336 | 5.5350 | 7.9670 | 1.439 | 22.0780 | 3.989 | 20.1120 | 3.634 |
| llama8b.mlp_down.t8 | 8 x 4096 x 14336 | 4.9780 | 9.6860 | 1.946 | 51.9610 | 10.438 | 20.4330 | 4.105 |
| llama8b.mlp_down.t512 | 512 x 4096 x 14336 | 65.5290 | 70.1810 | 1.071 | 1854.3810 | 28.299 | 55.3670 | 0.845 |
| llama8b.lm_head.t1 | 1 x 128256 x 4096 | 33.0710 | 74.5810 | 2.255 | 129.7790 | 3.924 | 109.1880 | 3.302 |
| llama8b.lm_head.t8 | 8 x 128256 x 4096 | 33.2280 | 74.7180 | 2.249 | 349.4260 | 10.516 | 109.1960 | 3.286 |
| llama8b.lm_head.t512 | 512 x 16032 x 4096 (CAPPED) | 67.5780 | 72.9270 | 1.079 | 2028.3360 | 30.015 | 48.1080 | 0.712 |

### mi325x

Run 4 (the run of record), steward request 1790651987201, timing 2026-09-29T03:20:01Z to 03:20:30Z, commit f1e6cb61c. TAKEN ALONE as a steward speed job, from a warm cache: the same binary had been built and run once, untimed, by request 1790651267504. This lane made no contact with the box from 03:19:49Z to 03:40:22Z. ONE GPU shared by every lane.

| shape | m x n x k | fp32.v1 ms | bf16f32.v1 ms | over | int8i32.v1 ms | over | Apple probe ms | over |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| llama8b.qkv.t1 | 1 x 4096 x 4096 | 0.0617 | 1.1700 | 18.963 | 5.2064 | 84.382 | not run |  |
| llama8b.qkv.t8 | 8 x 4096 x 4096 | 0.0775 | 0.1395 | 1.800 | 5.2733 | 68.043 | not run |  |
| llama8b.qkv.t512 | 512 x 4096 x 4096 | 0.9254 | 0.9616 | 1.039 | 8.0397 | 8.688 | not run |  |
| llama8b.mlp_up.t1 | 1 x 14336 x 4096 | 0.1631 | 1.2557 | 7.699 | 5.4792 | 33.594 | not run |  |
| llama8b.mlp_up.t8 | 8 x 14336 x 4096 | 0.1677 | 0.4017 | 2.395 | 5.5771 | 33.256 | not run |  |
| llama8b.mlp_up.t512 | 512 x 14336 x 4096 | 2.3895 | 2.6662 | 1.116 | 8.9997 | 3.766 | not run |  |
| llama8b.mlp_down.t1 | 1 x 4096 x 14336 | 0.1560 | 3.9136 | 25.087 | 18.9537 | 121.498 | not run |  |
| llama8b.mlp_down.t8 | 8 x 4096 x 14336 | 0.1947 | 0.4312 | 2.215 | 19.3871 | 99.574 | not run |  |
| llama8b.mlp_down.t512 | 512 x 4096 x 14336 | 2.5107 | 2.7672 | 1.102 | 28.8290 | 11.482 | not run |  |
| llama8b.lm_head.t1 | 1 x 128256 x 4096 | 1.3122 | 3.5467 | 2.703 | 17.1070 | 13.037 | not run |  |
| llama8b.lm_head.t8 | 8 x 128256 x 4096 | 1.2280 | 3.4718 | 2.827 | 17.5921 | 14.326 | not run |  |
| llama8b.lm_head.t512 | 512 x 16032 x 4096 (CAPPED) | 2.7776 | 3.0784 | 1.108 | 9.1749 | 3.303 | not run |  |

## The products alone: the plan each profile's dispatcher runs

`over` is the arm's median over fp32.v1's at the same shape on the same box. `+q` adds the per-call
quantization of the activations (`convert.int8.quantize.a`), a sum of two medians. The Apple probe
is not in any dispatcher.

### h100

| shape | m x n x k | fp32.v1 ms | bf16f32.v1 ms | over | int8i32.v1 ms | over | +q over | Apple probe ms | over | +q over |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| llama8b.qkv.t1 | 1 x 4096 x 4096 | 0.0737 | 0.7918 | 10.744 | 0.1591 | 2.159 | 8.796 | not run | |  |
| llama8b.qkv.t8 | 8 x 4096 x 4096 | 0.1074 | 0.1506 | 1.402 | 0.1486 | 1.384 | 6.987 | not run | |  |
| llama8b.qkv.t512 | 512 x 4096 x 4096 | 1.0377 | 1.1048 | 1.065 | 1.4407 | 1.388 | 3.917 | not run | |  |
| llama8b.mlp_up.t1 | 1 x 14336 x 4096 | 0.1884 | 0.7926 | 4.207 | 0.1914 | 1.016 | 3.679 | not run | |  |
| llama8b.mlp_up.t8 | 8 x 14336 x 4096 | 0.2757 | 0.4534 | 1.645 | 0.2094 | 0.760 | 2.984 | not run | |  |
| llama8b.mlp_up.t512 | 512 x 14336 x 4096 | 3.3462 | 3.5225 | 1.053 | 4.6280 | 1.383 | 2.174 | not run | |  |
| llama8b.mlp_down.t1 | 1 x 4096 x 14336 | 0.1773 | 2.7570 | 15.550 | 0.6532 | 3.684 | 13.472 | not run | |  |
| llama8b.mlp_down.t8 | 8 x 4096 x 14336 | 0.2918 | 0.4674 | 1.602 | 0.6976 | 2.391 | 10.709 | not run | |  |
| llama8b.mlp_down.t512 | 512 x 4096 x 14336 | 3.3696 | 3.5471 | 1.053 | 5.5492 | 1.647 | 4.604 | not run | |  |
| llama8b.lm_head.t1 | 1 x 128256 x 4096 | 1.2049 | 2.8119 | 2.334 | 0.9633 | 0.799 | 1.222 | not run | |  |
| llama8b.lm_head.t8 | 8 x 128256 x 4096 | 2.1659 | 3.7598 | 1.736 | 1.3649 | 0.630 | 0.913 | not run | |  |
| llama8b.lm_head.t512 | 512 x 16032 x 4096 (CAPPED) | 3.7367 | 3.9342 | 1.053 | 5.1285 | 1.372 | 2.048 | not run | |  |

### m3ultra

| shape | m x n x k | fp32.v1 ms | bf16f32.v1 ms | over | int8i32.v1 ms | over | +q over | Apple probe ms | over | +q over |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| llama8b.qkv.t1 | 1 x 4096 x 4096 | 1.0240 | 1.0150 | 0.991 | 0.7490 | 0.731 | 1.862 | 0.7640 | 0.746 | 1.877 |
| llama8b.qkv.t8 | 8 x 4096 x 4096 | 0.5420 | 1.0750 | 1.983 | 0.9800 | 1.808 | 4.039 | 0.7570 | 1.397 | 3.627 |
| llama8b.qkv.t512 | 512 x 4096 x 4096 | 3.3200 | 3.8250 | 1.152 | 26.5880 | 8.008 | 8.437 | 2.2450 | 0.676 | 1.105 |
| llama8b.mlp_up.t1 | 1 x 14336 x 4096 | 1.0440 | 1.0160 | 0.973 | 0.7150 | 0.685 | 1.814 | 0.8180 | 0.784 | 1.913 |
| llama8b.mlp_up.t8 | 8 x 14336 x 4096 | 1.0200 | 2.9720 | 2.914 | 2.1560 | 2.114 | 3.340 | 0.8260 | 0.810 | 2.036 |
| llama8b.mlp_up.t512 | 512 x 14336 x 4096 | 10.4170 | 12.3530 | 1.186 | 91.1190 | 8.747 | 8.886 | 6.0770 | 0.583 | 0.722 |
| llama8b.mlp_down.t1 | 1 x 4096 x 14336 | 2.8720 | 2.7150 | 0.945 | 1.7910 | 0.624 | 1.800 | 1.9240 | 0.670 | 1.846 |
| llama8b.mlp_down.t8 | 8 x 4096 x 14336 | 1.3680 | 3.3250 | 2.431 | 2.9650 | 2.167 | 4.932 | 1.9530 | 1.428 | 4.192 |
| llama8b.mlp_down.t512 | 512 x 4096 x 14336 | 11.4650 | 13.4240 | 1.171 | 117.6940 | 10.266 | 10.681 | 7.1410 | 0.623 | 1.039 |
| llama8b.lm_head.t1 | 1 x 128256 x 4096 | 4.0720 | 21.4810 | 5.275 | 2.1830 | 0.536 | 0.828 | 2.8760 | 0.706 | 0.998 |
| llama8b.lm_head.t8 | 8 x 128256 x 4096 | 5.7290 | 23.0470 | 4.023 | 13.4380 | 2.346 | 2.553 | 2.9300 | 0.511 | 0.719 |
| llama8b.lm_head.t512 | 512 x 16032 x 4096 (CAPPED) | 11.5820 | 13.7680 | 1.189 | 102.7290 | 8.870 | 8.995 | 6.7970 | 0.587 | 0.712 |

### m2pro

| shape | m x n x k | fp32.v1 ms | bf16f32.v1 ms | over | int8i32.v1 ms | over | +q over | Apple probe ms | over | +q over |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| llama8b.qkv.t1 | 1 x 4096 x 4096 | 1.5170 | 1.4690 | 0.968 | 1.3570 | 0.895 | 1.779 | 0.8090 | 0.533 | 1.418 |
| llama8b.qkv.t8 | 8 x 4096 x 4096 | 1.4930 | 2.0760 | 1.390 | 9.7600 | 6.537 | 7.473 | 0.8360 | 0.560 | 1.496 |
| llama8b.qkv.t512 | 512 x 4096 x 4096 | 17.7380 | 18.3010 | 1.032 | 514.8840 | 29.027 | 29.223 | 8.6110 | 0.485 | 0.682 |
| llama8b.mlp_up.t1 | 1 x 14336 x 4096 | 5.2580 | 5.2000 | 0.989 | 4.9860 | 0.948 | 1.201 | 2.0230 | 0.385 | 0.638 |
| llama8b.mlp_up.t8 | 8 x 14336 x 4096 | 4.1060 | 6.0270 | 1.468 | 29.4060 | 7.162 | 7.505 | 2.0320 | 0.495 | 0.839 |
| llama8b.mlp_up.t512 | 512 x 14336 x 4096 | 60.5230 | 62.3180 | 1.030 | 1799.5110 | 29.733 | 29.791 | 28.0180 | 0.463 | 0.521 |
| llama8b.mlp_down.t1 | 1 x 4096 x 14336 | 5.5350 | 5.1380 | 0.928 | 4.4320 | 0.801 | 1.579 | 2.5020 | 0.452 | 1.230 |
| llama8b.mlp_down.t8 | 8 x 4096 x 14336 | 4.9780 | 6.9070 | 1.388 | 33.9800 | 6.826 | 7.722 | 2.4910 | 0.500 | 1.397 |
| llama8b.mlp_down.t512 | 512 x 4096 x 14336 | 65.5290 | 67.2640 | 1.026 | 1829.4860 | 27.919 | 28.102 | 30.0130 | 0.458 | 0.641 |
| llama8b.lm_head.t1 | 1 x 128256 x 4096 | 33.0710 | 50.1380 | 1.516 | 32.6260 | 0.987 | 1.029 | 11.8530 | 0.358 | 0.401 |
| llama8b.lm_head.t8 | 8 x 128256 x 4096 | 33.2280 | 50.2500 | 1.512 | 252.5490 | 7.600 | 7.644 | 11.8550 | 0.357 | 0.400 |
| llama8b.lm_head.t512 | 512 x 16032 x 4096 (CAPPED) | 67.5780 | 69.6350 | 1.030 | 2011.3610 | 29.764 | 29.815 | 31.3530 | 0.464 | 0.515 |

### mi325x

| shape | m x n x k | fp32.v1 ms | bf16f32.v1 ms | over | int8i32.v1 ms | over | +q over | Apple probe ms | over | +q over |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| llama8b.qkv.t1 | 1 x 4096 x 4096 | 0.0617 | 1.1355 | 18.404 | 0.0512 | 0.830 | 22.825 | not run | |  |
| llama8b.qkv.t8 | 8 x 4096 x 4096 | 0.0775 | 0.1067 | 1.377 | 0.0516 | 0.666 | 19.210 | not run | |  |
| llama8b.qkv.t512 | 512 x 4096 x 4096 | 0.9254 | 0.9384 | 1.014 | 0.2669 | 0.288 | 4.243 | not run | |  |
| llama8b.mlp_up.t1 | 1 x 14336 x 4096 | 0.1631 | 1.1869 | 7.277 | 0.0567 | 0.348 | 8.667 | not run | |  |
| llama8b.mlp_up.t8 | 8 x 14336 x 4096 | 0.1677 | 0.2917 | 1.739 | 0.0583 | 0.348 | 8.937 | not run | |  |
| llama8b.mlp_up.t512 | 512 x 14336 x 4096 | 2.3895 | 2.4240 | 1.014 | 0.8773 | 0.367 | 1.954 | not run | |  |
| llama8b.mlp_down.t1 | 1 x 4096 x 14336 | 0.1560 | 3.9476 | 25.305 | 0.1374 | 0.881 | 30.950 | not run | |  |
| llama8b.mlp_down.t8 | 8 x 4096 x 14336 | 0.1947 | 0.3105 | 1.595 | 0.1415 | 0.727 | 27.285 | not run | |  |
| llama8b.mlp_down.t512 | 512 x 4096 x 14336 | 2.5107 | 2.5016 | 0.996 | 0.8422 | 0.335 | 5.651 | not run | |  |
| llama8b.lm_head.t1 | 1 x 128256 x 4096 | 1.3122 | 2.4613 | 1.876 | 0.2862 | 0.218 | 1.260 | not run | |  |
| llama8b.lm_head.t8 | 8 x 128256 x 4096 | 1.2280 | 2.4054 | 1.959 | 0.3323 | 0.271 | 1.459 | not run | |  |
| llama8b.lm_head.t512 | 512 x 16032 x 4096 (CAPPED) | 2.7776 | 2.8174 | 1.014 | 1.0130 | 0.365 | 1.780 | not run | |  |

## h100 (column nvidia)

Run 3, nvc3-0010, 2026-09-29T03:07:48Z to 03:09:08Z, lane commit d4948864a. TAKEN ALONE: the binary was built and run once by an earlier job; this lane ran nothing else on the box from the submit to the end of the job. The pod is shared with other lanes, whose work this lane cannot see.

Vendor comparison: cuBLAS on NVIDIA H100 NVL, torch 2.13.0+cu129, CUDA 12.9, median of 10. COMPARISON ONLY, no digest, no identity claim.

### llama8b.qkv.t1: m=1 n=4096 k=4096 (FULL)

| arm | median ms | min ms | rate | arm over fp32.v1 | note |
|---|---:|---:|---:|---:|---|
| fp32.v1 | 0.0737 | 0.0718 | 227.5340 GMAC/s | 1.000 | plan=13 |
| bf16f32.v1.fused | 0.7918 | 0.7899 | 21.1884 GMAC/s | 10.744 | dispatched |
| bf16f32.v1.widen | 0.1245 | 0.1198 | 134.7871 GMAC/s | 1.689 | not-dispatched(cells<=16384) |
| int8i32.v1.flat | 0.6663 | 0.6645 | 25.1804 GMAC/s | 9.041 | not-dispatched(the-column-has-the-unit) |
| int8i32.v1.mma | 0.1591 | 0.1556 | 105.4607 GMAC/s | 2.159 | dispatched |
| int8i32.v1.applechunk | not run | | | | the column does not have the unit |
| convert.int8.quantize.a | 0.4892 | 0.4881 | 0.0084 Gelem/s | 6.638 | per-call |
| convert.int8.pack.b | 2.6705 | 2.6580 | 6.2824 Gelem/s | 36.235 | once-per-weight |
| convert.bf16.pack.b | 0.0640 | 0.0636 | 262.2956 Gelem/s | 0.868 | once-per-weight |
| convert.bf16.widen.b | 0.0626 | 0.0618 | 267.9510 Gelem/s | 0.849 | per-call-when-materialized |
| convert.int8.dequantize.b | 0.0794 | 0.0788 | 211.2574 Gelem/s | 1.077 | per-call-when-materialized |
| inference.bf16f32.v1 | 0.7976 | 0.7925 | 21.0351 GMAC/s | 10.822 | product |
| inference.int8i32.v1 | 0.6420 | 0.6393 | 26.1316 GMAC/s | 8.711 | quantize.a+product |
| inference.int8i32.v1.applechunk | not run | | | | the column does not have the unit |
| training.bf16f32.v1 | 0.8473 | 0.8423 | 19.8007 GMAC/s | 11.497 | pack.b+product |
| training.int8i32.v1 | 3.2892 | 3.2759 | 5.1007 GMAC/s | 44.630 | quantize.a+pack.b+product |
| training.int8i32.v1.applechunk | not run | | | | the column does not have the unit |
| DERIVED int8i32.v1.flat + quantize.a | 1.1555 | | | 15.678 | sum of two medians |
| DERIVED int8i32.v1.mma + quantize.a | 0.6483 | | | 8.796 | sum of two medians |
| vendor strict (COMPARISON ONLY) | 0.0460 | 0.0453 | 364.6270 GMAC/s | | fp32.v1 over this: 1.602 |
| vendor tf32 (COMPARISON ONLY) | 0.0443 | 0.0438 | 378.3266 GMAC/s | | fp32.v1 over this: 1.662 |
| vendor bf16 (COMPARISON ONLY) | 0.0344 | 0.0333 | 487.2312 GMAC/s | | fp32.v1 over this: 2.140 |

### llama8b.qkv.t8: m=8 n=4096 k=4096 (FULL)

| arm | median ms | min ms | rate | arm over fp32.v1 | note |
|---|---:|---:|---:|---:|---|
| fp32.v1 | 0.1074 | 0.1036 | 1249.9323 GMAC/s | 1.000 | plan=14 |
| bf16f32.v1.fused | 0.7850 | 0.7816 | 170.9787 GMAC/s | 7.309 | not-dispatched(cells>16384) |
| bf16f32.v1.widen | 0.1506 | 0.1492 | 891.3857 GMAC/s | 1.402 | dispatched |
| int8i32.v1.flat | 0.6652 | 0.6641 | 201.7826 GMAC/s | 6.194 | not-dispatched(the-column-has-the-unit) |
| int8i32.v1.mma | 0.1486 | 0.1474 | 903.3729 GMAC/s | 1.384 | dispatched |
| int8i32.v1.applechunk | not run | | | | the column does not have the unit |
| convert.int8.quantize.a | 0.6018 | 0.5989 | 0.0545 Gelem/s | 5.603 | per-call |
| convert.int8.pack.b | 2.4819 | 2.4809 | 6.7599 Gelem/s | 23.109 | once-per-weight |
| convert.bf16.pack.b | 0.0669 | 0.0658 | 250.6981 Gelem/s | 0.623 | once-per-weight |
| convert.bf16.widen.b | 0.0648 | 0.0619 | 258.8118 Gelem/s | 0.603 | per-call-when-materialized |
| convert.int8.dequantize.b | 0.0807 | 0.0774 | 207.9889 Gelem/s | 0.751 | per-call-when-materialized |
| inference.bf16f32.v1 | 0.1529 | 0.1508 | 877.7219 GMAC/s | 1.424 | product |
| inference.int8i32.v1 | 0.7706 | 0.7679 | 174.1739 GMAC/s | 7.175 | quantize.a+product |
| inference.int8i32.v1.applechunk | not run | | | | the column does not have the unit |
| training.bf16f32.v1 | 0.2084 | 0.2070 | 644.1596 GMAC/s | 1.940 | pack.b+product |
| training.int8i32.v1 | 3.3994 | 3.3924 | 39.4827 GMAC/s | 31.652 | quantize.a+pack.b+product |
| training.int8i32.v1.applechunk | not run | | | | the column does not have the unit |
| DERIVED int8i32.v1.flat + quantize.a | 1.2670 | | | 11.797 | sum of two medians |
| DERIVED int8i32.v1.mma + quantize.a | 0.7504 | | | 6.987 | sum of two medians |
| vendor strict (COMPARISON ONLY) | 0.0692 | 0.0680 | 1938.9606 GMAC/s | | fp32.v1 over this: 1.552 |
| vendor tf32 (COMPARISON ONLY) | 0.0526 | 0.0514 | 2551.6371 GMAC/s | | fp32.v1 over this: 2.042 |
| vendor bf16 (COMPARISON ONLY) | 0.0268 | 0.0263 | 5006.5202 GMAC/s | | fp32.v1 over this: 4.006 |

### llama8b.qkv.t512: m=512 n=4096 k=4096 (FULL)

| arm | median ms | min ms | rate | arm over fp32.v1 | note |
|---|---:|---:|---:|---:|---|
| fp32.v1 | 1.0377 | 1.0367 | 8277.5003 GMAC/s | 1.000 | plan=10 |
| bf16f32.v1.fused | 39.4083 | 39.3668 | 217.9725 GMAC/s | 37.977 | not-dispatched(cells>16384) |
| bf16f32.v1.widen | 1.1048 | 1.0949 | 7775.3008 GMAC/s | 1.065 | dispatched |
| int8i32.v1.flat | 39.1923 | 39.1891 | 219.1738 GMAC/s | 37.768 | not-dispatched(the-column-has-the-unit) |
| int8i32.v1.mma | 1.4407 | 1.4183 | 5962.4664 GMAC/s | 1.388 | dispatched |
| int8i32.v1.applechunk | not run | | | | the column does not have the unit |
| convert.int8.quantize.a | 2.6236 | 2.6179 | 0.7993 Gelem/s | 2.528 | per-call |
| convert.int8.pack.b | 2.4763 | 2.4717 | 6.7752 Gelem/s | 2.386 | once-per-weight |
| convert.bf16.pack.b | 0.0639 | 0.0623 | 262.6447 Gelem/s | 0.062 | once-per-weight |
| convert.bf16.widen.b | 0.0636 | 0.0615 | 263.7305 Gelem/s | 0.061 | per-call-when-materialized |
| convert.int8.dequantize.b | 0.0808 | 0.0790 | 207.7442 Gelem/s | 0.078 | per-call-when-materialized |
| inference.bf16f32.v1 | 1.0943 | 1.0889 | 7850.0228 GMAC/s | 1.055 | product |
| inference.int8i32.v1 | 4.0802 | 4.0790 | 2105.2889 GMAC/s | 3.932 | quantize.a+product |
| inference.int8i32.v1.applechunk | not run | | | | the column does not have the unit |
| training.bf16f32.v1 | 1.1520 | 1.1458 | 7456.5663 GMAC/s | 1.110 | pack.b+product |
| training.int8i32.v1 | 6.7240 | 6.7133 | 1277.4964 GMAC/s | 6.480 | quantize.a+pack.b+product |
| training.int8i32.v1.applechunk | not run | | | | the column does not have the unit |
| DERIVED int8i32.v1.flat + quantize.a | 41.8159 | | | 40.297 | sum of two medians |
| DERIVED int8i32.v1.mma + quantize.a | 4.0643 | | | 3.917 | sum of two medians |
| vendor strict (COMPARISON ONLY) | 0.4194 | 0.4162 | 20480.1120 GMAC/s | | fp32.v1 over this: 2.474 |
| vendor tf32 (COMPARISON ONLY) | 0.0790 | 0.0759 | 108741.1743 GMAC/s | | fp32.v1 over this: 13.136 |
| vendor bf16 (COMPARISON ONLY) | 0.0428 | 0.0414 | 200883.6528 GMAC/s | | fp32.v1 over this: 24.268 |

### llama8b.mlp_up.t1: m=1 n=14336 k=4096 (FULL)

| arm | median ms | min ms | rate | arm over fp32.v1 | note |
|---|---:|---:|---:|---:|---|
| fp32.v1 | 0.1884 | 0.1824 | 311.6241 GMAC/s | 1.000 | plan=13 |
| bf16f32.v1.fused | 0.7926 | 0.7903 | 74.0889 GMAC/s | 4.207 | dispatched |
| bf16f32.v1.widen | 0.3677 | 0.3659 | 159.7161 GMAC/s | 1.952 | not-dispatched(cells<=16384) |
| int8i32.v1.flat | 0.6687 | 0.6646 | 87.8164 GMAC/s | 3.549 | not-dispatched(the-column-has-the-unit) |
| int8i32.v1.mma | 0.1914 | 0.1898 | 306.8704 GMAC/s | 1.016 | dispatched |
| int8i32.v1.applechunk | not run | | | | the column does not have the unit |
| convert.int8.quantize.a | 0.5017 | 0.5007 | 0.0082 Gelem/s | 2.663 | per-call |
| convert.int8.pack.b | 2.4907 | 2.4863 | 23.5757 Gelem/s | 13.220 | once-per-weight |
| convert.bf16.pack.b | 0.1988 | 0.1974 | 295.3260 Gelem/s | 1.055 | once-per-weight |
| convert.bf16.widen.b | 0.1892 | 0.1883 | 310.3706 Gelem/s | 1.004 | per-call-when-materialized |
| convert.int8.dequantize.b | 0.2460 | 0.2456 | 238.7051 Gelem/s | 1.306 | per-call-when-materialized |
| inference.bf16f32.v1 | 0.7966 | 0.7963 | 73.7169 GMAC/s | 4.228 | product |
| inference.int8i32.v1 | 0.6701 | 0.6679 | 87.6271 GMAC/s | 3.557 | quantize.a+product |
| inference.int8i32.v1.applechunk | not run | | | | the column does not have the unit |
| training.bf16f32.v1 | 0.9845 | 0.9800 | 59.6441 GMAC/s | 5.226 | pack.b+product |
| training.int8i32.v1 | 3.3314 | 3.3164 | 17.6260 GMAC/s | 17.683 | quantize.a+pack.b+product |
| training.int8i32.v1.applechunk | not run | | | | the column does not have the unit |
| DERIVED int8i32.v1.flat + quantize.a | 1.1704 | | | 6.212 | sum of two medians |
| DERIVED int8i32.v1.mma + quantize.a | 0.6931 | | | 3.679 | sum of two medians |
| vendor strict (COMPARISON ONLY) | 0.0889 | 0.0875 | 660.2620 GMAC/s | | fp32.v1 over this: 2.118 |
| vendor tf32 (COMPARISON ONLY) | 0.0882 | 0.0856 | 666.0863 GMAC/s | | fp32.v1 over this: 2.137 |
| vendor bf16 (COMPARISON ONLY) | 0.0516 | 0.0507 | 1138.2581 GMAC/s | | fp32.v1 over this: 3.652 |

### llama8b.mlp_up.t8: m=8 n=14336 k=4096 (FULL)

| arm | median ms | min ms | rate | arm over fp32.v1 | note |
|---|---:|---:|---:|---:|---|
| fp32.v1 | 0.2757 | 0.2740 | 1704.1481 GMAC/s | 1.000 | plan=14 |
| bf16f32.v1.fused | 2.5309 | 2.5293 | 185.6071 GMAC/s | 9.180 | not-dispatched(cells>16384) |
| bf16f32.v1.widen | 0.4534 | 0.4532 | 1036.1126 GMAC/s | 1.645 | dispatched |
| int8i32.v1.flat | 2.5062 | 2.5057 | 187.4435 GMAC/s | 9.090 | not-dispatched(the-column-has-the-unit) |
| int8i32.v1.mma | 0.2094 | 0.2089 | 2243.1897 GMAC/s | 0.760 | dispatched |
| int8i32.v1.applechunk | not run | | | | the column does not have the unit |
| convert.int8.quantize.a | 0.6134 | 0.6130 | 0.0534 Gelem/s | 2.225 | per-call |
| convert.int8.pack.b | 2.4844 | 2.4830 | 23.6351 Gelem/s | 9.011 | once-per-weight |
| convert.bf16.pack.b | 0.1984 | 0.1973 | 296.0272 Gelem/s | 0.720 | once-per-weight |
| convert.bf16.widen.b | 0.1876 | 0.1871 | 313.0295 Gelem/s | 0.680 | per-call-when-materialized |
| convert.int8.dequantize.b | 0.2447 | 0.2440 | 239.9517 Gelem/s | 0.888 | per-call-when-materialized |
| inference.bf16f32.v1 | 0.4535 | 0.4519 | 1035.8430 GMAC/s | 1.645 | product |
| inference.int8i32.v1 | 0.8027 | 0.8021 | 585.1932 GMAC/s | 2.911 | quantize.a+product |
| inference.int8i32.v1.applechunk | not run | | | | the column does not have the unit |
| training.bf16f32.v1 | 0.6398 | 0.6388 | 734.2545 GMAC/s | 2.321 | pack.b+product |
| training.int8i32.v1 | 3.3082 | 3.3008 | 141.9977 GMAC/s | 11.999 | quantize.a+pack.b+product |
| training.int8i32.v1.applechunk | not run | | | | the column does not have the unit |
| DERIVED int8i32.v1.flat + quantize.a | 3.1196 | | | 11.315 | sum of two medians |
| DERIVED int8i32.v1.mma + quantize.a | 0.8228 | | | 2.984 | sum of two medians |
| vendor strict (COMPARISON ONLY) | 0.1527 | 0.1516 | 3077.2521 GMAC/s | | fp32.v1 over this: 1.806 |
| vendor tf32 (COMPARISON ONLY) | 0.0994 | 0.0985 | 4727.2802 GMAC/s | | fp32.v1 over this: 2.774 |
| vendor bf16 (COMPARISON ONLY) | 0.0531 | 0.0524 | 8844.6783 GMAC/s | | fp32.v1 over this: 5.191 |

### llama8b.mlp_up.t512: m=512 n=14336 k=4096 (FULL)

| arm | median ms | min ms | rate | arm over fp32.v1 | note |
|---|---:|---:|---:|---:|---|
| fp32.v1 | 3.3462 | 3.3395 | 8984.7099 GMAC/s | 1.000 | plan=10 |
| bf16f32.v1.fused | 139.2825 | 138.9768 | 215.8546 GMAC/s | 41.624 | not-dispatched(cells>16384) |
| bf16f32.v1.widen | 3.5225 | 3.5163 | 8535.1535 GMAC/s | 1.053 | dispatched |
| int8i32.v1.flat | 135.6707 | 135.6653 | 221.6011 GMAC/s | 40.545 | not-dispatched(the-column-has-the-unit) |
| int8i32.v1.mma | 4.6280 | 4.6260 | 6496.2674 GMAC/s | 1.383 | dispatched |
| int8i32.v1.applechunk | not run | | | | the column does not have the unit |
| convert.int8.quantize.a | 2.6462 | 2.6391 | 0.7925 Gelem/s | 0.791 | per-call |
| convert.int8.pack.b | 2.4778 | 2.4746 | 23.6988 Gelem/s | 0.740 | once-per-weight |
| convert.bf16.pack.b | 0.1976 | 0.1964 | 297.1207 Gelem/s | 0.059 | once-per-weight |
| convert.bf16.widen.b | 0.1880 | 0.1861 | 312.3219 Gelem/s | 0.056 | per-call-when-materialized |
| convert.int8.dequantize.b | 0.2451 | 0.2428 | 239.6246 Gelem/s | 0.073 | per-call-when-materialized |
| inference.bf16f32.v1 | 3.5188 | 3.5132 | 8544.1234 GMAC/s | 1.052 | product |
| inference.int8i32.v1 | 7.0929 | 7.0703 | 4238.7326 GMAC/s | 2.120 | quantize.a+product |
| inference.int8i32.v1.applechunk | not run | | | | the column does not have the unit |
| training.bf16f32.v1 | 3.7078 | 3.7048 | 8108.5092 GMAC/s | 1.108 | pack.b+product |
| training.int8i32.v1 | 9.6502 | 9.6310 | 3115.4477 GMAC/s | 2.884 | quantize.a+pack.b+product |
| training.int8i32.v1.applechunk | not run | | | | the column does not have the unit |
| DERIVED int8i32.v1.flat + quantize.a | 138.3169 | | | 41.336 | sum of two medians |
| DERIVED int8i32.v1.mma + quantize.a | 7.2742 | | | 2.174 | sum of two medians |
| vendor strict (COMPARISON ONLY) | 1.6540 | 1.6496 | 18176.6648 GMAC/s | | fp32.v1 over this: 2.023 |
| vendor tf32 (COMPARISON ONLY) | 0.2469 | 0.2405 | 121744.3855 GMAC/s | | fp32.v1 over this: 13.550 |
| vendor bf16 (COMPARISON ONLY) | 0.1391 | 0.1378 | 216184.1221 GMAC/s | | fp32.v1 over this: 24.061 |

### llama8b.mlp_down.t1: m=1 n=4096 k=14336 (FULL)

| arm | median ms | min ms | rate | arm over fp32.v1 | note |
|---|---:|---:|---:|---:|---|
| fp32.v1 | 0.1773 | 0.1711 | 331.2737 GMAC/s | 1.000 | plan=13 |
| bf16f32.v1.fused | 2.7570 | 2.7547 | 21.2986 GMAC/s | 15.550 | dispatched |
| bf16f32.v1.widen | 0.3531 | 0.3517 | 166.2860 GMAC/s | 1.992 | not-dispatched(cells<=16384) |
| int8i32.v1.flat | 2.3473 | 2.3404 | 25.0159 GMAC/s | 13.239 | not-dispatched(the-column-has-the-unit) |
| int8i32.v1.mma | 0.6532 | 0.6507 | 89.8984 GMAC/s | 3.684 | dispatched |
| int8i32.v1.applechunk | not run | | | | the column does not have the unit |
| convert.int8.quantize.a | 1.7354 | 1.7313 | 0.0083 Gelem/s | 9.788 | per-call |
| convert.int8.pack.b | 9.9852 | 9.9656 | 5.8807 Gelem/s | 56.318 | once-per-weight |
| convert.bf16.pack.b | 0.1993 | 0.1972 | 294.6281 Gelem/s | 1.124 | once-per-weight |
| convert.bf16.widen.b | 0.1870 | 0.1865 | 314.0641 Gelem/s | 1.055 | per-call-when-materialized |
| convert.int8.dequantize.b | 0.2452 | 0.2436 | 239.5210 Gelem/s | 1.383 | per-call-when-materialized |
| inference.bf16f32.v1 | 2.7606 | 2.7569 | 21.2706 GMAC/s | 15.570 | product |
| inference.int8i32.v1 | 2.3209 | 2.3175 | 25.3007 GMAC/s | 13.090 | quantize.a+product |
| inference.int8i32.v1.applechunk | not run | | | | the column does not have the unit |
| training.bf16f32.v1 | 2.9441 | 2.9401 | 19.9453 GMAC/s | 16.605 | pack.b+product |
| training.int8i32.v1 | 12.2797 | 12.2718 | 4.7819 GMAC/s | 69.259 | quantize.a+pack.b+product |
| training.int8i32.v1.applechunk | not run | | | | the column does not have the unit |
| DERIVED int8i32.v1.flat + quantize.a | 4.0827 | | | 23.027 | sum of two medians |
| DERIVED int8i32.v1.mma + quantize.a | 2.3886 | | | 13.472 | sum of two medians |
| vendor strict (COMPARISON ONLY) | 0.0822 | 0.0813 | 714.2174 GMAC/s | | fp32.v1 over this: 2.157 |
| vendor tf32 (COMPARISON ONLY) | 0.0820 | 0.0807 | 716.2783 GMAC/s | | fp32.v1 over this: 2.163 |
| vendor bf16 (COMPARISON ONLY) | 0.0557 | 0.0546 | 1053.6761 GMAC/s | | fp32.v1 over this: 3.181 |

### llama8b.mlp_down.t8: m=8 n=4096 k=14336 (FULL)

| arm | median ms | min ms | rate | arm over fp32.v1 | note |
|---|---:|---:|---:|---:|---|
| fp32.v1 | 0.2918 | 0.2900 | 1609.6175 GMAC/s | 1.000 | plan=14 |
| bf16f32.v1.fused | 2.7008 | 2.6968 | 173.9361 GMAC/s | 9.256 | not-dispatched(cells>16384) |
| bf16f32.v1.widen | 0.4674 | 0.4649 | 1005.1052 GMAC/s | 1.602 | dispatched |
| int8i32.v1.flat | 2.3387 | 2.3358 | 200.8640 GMAC/s | 8.015 | not-dispatched(the-column-has-the-unit) |
| int8i32.v1.mma | 0.6976 | 0.6897 | 673.4206 GMAC/s | 2.391 | dispatched |
| int8i32.v1.applechunk | not run | | | | the column does not have the unit |
| convert.int8.quantize.a | 2.4272 | 2.4200 | 0.0473 Gelem/s | 8.318 | per-call |
| convert.int8.pack.b | 9.9917 | 9.9606 | 5.8769 Gelem/s | 34.242 | once-per-weight |
| convert.bf16.pack.b | 0.1987 | 0.1977 | 295.4746 Gelem/s | 0.681 | once-per-weight |
| convert.bf16.widen.b | 0.1870 | 0.1864 | 314.0121 Gelem/s | 0.641 | per-call-when-materialized |
| convert.int8.dequantize.b | 0.2438 | 0.2429 | 240.8394 Gelem/s | 0.836 | per-call-when-materialized |
| inference.bf16f32.v1 | 0.4693 | 0.4681 | 1001.0485 GMAC/s | 1.608 | product |
| inference.int8i32.v1 | 3.0493 | 3.0398 | 154.0573 GMAC/s | 10.450 | quantize.a+product |
| inference.int8i32.v1.applechunk | not run | | | | the column does not have the unit |
| training.bf16f32.v1 | 0.6561 | 0.6550 | 716.0134 GMAC/s | 2.248 | pack.b+product |
| training.int8i32.v1 | 13.0500 | 13.0116 | 35.9972 GMAC/s | 44.722 | quantize.a+pack.b+product |
| training.int8i32.v1.applechunk | not run | | | | the column does not have the unit |
| DERIVED int8i32.v1.flat + quantize.a | 4.7659 | | | 16.333 | sum of two medians |
| DERIVED int8i32.v1.mma + quantize.a | 3.1248 | | | 10.709 | sum of two medians |
| vendor strict (COMPARISON ONLY) | 0.1953 | 0.1947 | 2405.4076 GMAC/s | | fp32.v1 over this: 1.494 |
| vendor tf32 (COMPARISON ONLY) | 0.0940 | 0.0930 | 4997.0592 GMAC/s | | fp32.v1 over this: 3.104 |
| vendor bf16 (COMPARISON ONLY) | 0.0559 | 0.0545 | 8403.2880 GMAC/s | | fp32.v1 over this: 5.220 |

### llama8b.mlp_down.t512: m=512 n=4096 k=14336 (FULL)

| arm | median ms | min ms | rate | arm over fp32.v1 | note |
|---|---:|---:|---:|---:|---|
| fp32.v1 | 3.3696 | 3.3627 | 8922.3243 GMAC/s | 1.000 | plan=10 |
| bf16f32.v1.fused | 136.6683 | 136.4845 | 219.9835 GMAC/s | 40.559 | not-dispatched(cells>16384) |
| bf16f32.v1.widen | 3.5471 | 3.5429 | 8475.7541 GMAC/s | 1.053 | dispatched |
| int8i32.v1.flat | 137.9638 | 137.9026 | 217.9179 GMAC/s | 40.944 | not-dispatched(the-column-has-the-unit) |
| int8i32.v1.mma | 5.5492 | 5.5429 | 5417.8217 GMAC/s | 1.647 | dispatched |
| int8i32.v1.applechunk | not run | | | | the column does not have the unit |
| convert.int8.quantize.a | 9.9645 | 9.9555 | 0.7366 Gelem/s | 2.957 | per-call |
| convert.int8.pack.b | 10.0053 | 9.9670 | 5.8689 Gelem/s | 2.969 | once-per-weight |
| convert.bf16.pack.b | 0.1979 | 0.1975 | 296.6763 Gelem/s | 0.059 | once-per-weight |
| convert.bf16.widen.b | 0.1868 | 0.1862 | 314.3298 Gelem/s | 0.055 | per-call-when-materialized |
| convert.int8.dequantize.b | 0.2435 | 0.2426 | 241.2005 Gelem/s | 0.072 | per-call-when-materialized |
| inference.bf16f32.v1 | 3.5462 | 3.5406 | 8478.0486 GMAC/s | 1.052 | product |
| inference.int8i32.v1 | 15.5603 | 15.4742 | 1932.1518 GMAC/s | 4.618 | quantize.a+product |
| inference.int8i32.v1.applechunk | not run | | | | the column does not have the unit |
| training.bf16f32.v1 | 3.7332 | 3.7294 | 8053.3125 GMAC/s | 1.108 | pack.b+product |
| training.int8i32.v1 | 25.5078 | 25.4380 | 1178.6489 GMAC/s | 7.570 | quantize.a+pack.b+product |
| training.int8i32.v1.applechunk | not run | | | | the column does not have the unit |
| DERIVED int8i32.v1.flat + quantize.a | 147.9283 | | | 43.901 | sum of two medians |
| DERIVED int8i32.v1.mma + quantize.a | 15.5137 | | | 4.604 | sum of two medians |
| vendor strict (COMPARISON ONLY) | 1.3826 | 1.3809 | 21745.6601 GMAC/s | | fp32.v1 over this: 2.437 |
| vendor tf32 (COMPARISON ONLY) | 0.2799 | 0.2745 | 107411.5921 GMAC/s | | fp32.v1 over this: 12.038 |
| vendor bf16 (COMPARISON ONLY) | 0.1374 | 0.1340 | 218844.0967 GMAC/s | | fp32.v1 over this: 24.528 |

### llama8b.lm_head.t1: m=1 n=128256 k=4096 (FULL)

| arm | median ms | min ms | rate | arm over fp32.v1 | note |
|---|---:|---:|---:|---:|---|
| fp32.v1 | 1.2049 | 1.2045 | 435.9962 GMAC/s | 1.000 | plan=13 |
| bf16f32.v1.fused | 2.5968 | 2.5837 | 202.3032 GMAC/s | 2.155 | not-dispatched(cells>16384) |
| bf16f32.v1.widen | 2.8119 | 2.8076 | 186.8239 GMAC/s | 2.334 | dispatched |
| int8i32.v1.flat | 2.5115 | 2.4984 | 209.1759 GMAC/s | 2.084 | not-dispatched(the-column-has-the-unit) |
| int8i32.v1.mma | 0.9633 | 0.9419 | 545.3781 GMAC/s | 0.799 | dispatched |
| int8i32.v1.applechunk | not run | | | | the column does not have the unit |
| convert.int8.quantize.a | 0.5086 | 0.5026 | 0.0081 Gelem/s | 0.422 | per-call |
| convert.int8.pack.b | 12.1727 | 12.0612 | 43.1569 Gelem/s | 10.103 | once-per-weight |
| convert.bf16.pack.b | 1.6761 | 1.6726 | 313.4233 Gelem/s | 1.391 | once-per-weight |
| convert.bf16.widen.b | 1.6152 | 1.6133 | 325.2453 Gelem/s | 1.341 | per-call-when-materialized |
| convert.int8.dequantize.b | 2.1048 | 2.1000 | 249.5942 Gelem/s | 1.747 | per-call-when-materialized |
| inference.bf16f32.v1 | 2.8132 | 2.8054 | 186.7412 GMAC/s | 2.335 | product |
| inference.int8i32.v1 | 1.4608 | 1.4554 | 359.6336 GMAC/s | 1.212 | quantize.a+product |
| inference.int8i32.v1.applechunk | not run | | | | the column does not have the unit |
| training.bf16f32.v1 | 4.4719 | 4.4680 | 117.4741 GMAC/s | 3.711 | pack.b+product |
| training.int8i32.v1 | 13.6141 | 13.5182 | 38.5876 GMAC/s | 11.299 | quantize.a+pack.b+product |
| training.int8i32.v1.applechunk | not run | | | | the column does not have the unit |
| DERIVED int8i32.v1.flat + quantize.a | 3.0201 | | | 2.507 | sum of two medians |
| DERIVED int8i32.v1.mma + quantize.a | 1.4719 | | | 1.222 | sum of two medians |
| vendor strict (COMPARISON ONLY) | 0.7126 | 0.7079 | 737.2177 GMAC/s | | fp32.v1 over this: 1.691 |
| vendor tf32 (COMPARISON ONLY) | 0.7116 | 0.6884 | 738.2921 GMAC/s | | fp32.v1 over this: 1.693 |
| vendor bf16 (COMPARISON ONLY) | 0.3210 | 0.3184 | 1636.4326 GMAC/s | | fp32.v1 over this: 3.753 |

### llama8b.lm_head.t8: m=8 n=128256 k=4096 (FULL)

| arm | median ms | min ms | rate | arm over fp32.v1 | note |
|---|---:|---:|---:|---:|---|
| fp32.v1 | 2.1659 | 2.1627 | 1940.4034 GMAC/s | 1.000 | plan=14 |
| bf16f32.v1.fused | 19.9111 | 19.3711 | 211.0724 GMAC/s | 9.193 | not-dispatched(cells>16384) |
| bf16f32.v1.widen | 3.7598 | 3.7593 | 1117.8010 GMAC/s | 1.736 | dispatched |
| int8i32.v1.flat | 19.2579 | 19.2484 | 218.2323 GMAC/s | 8.891 | not-dispatched(the-column-has-the-unit) |
| int8i32.v1.mma | 1.3649 | 1.3564 | 3079.0716 GMAC/s | 0.630 | dispatched |
| int8i32.v1.applechunk | not run | | | | the column does not have the unit |
| convert.int8.quantize.a | 0.6119 | 0.6091 | 0.0536 Gelem/s | 0.283 | per-call |
| convert.int8.pack.b | 12.1075 | 12.0764 | 43.3892 Gelem/s | 5.590 | once-per-weight |
| convert.bf16.pack.b | 1.6751 | 1.6720 | 313.6156 Gelem/s | 0.773 | once-per-weight |
| convert.bf16.widen.b | 1.6124 | 1.6119 | 325.8168 Gelem/s | 0.744 | per-call-when-materialized |
| convert.int8.dequantize.b | 2.1013 | 2.1008 | 250.0067 Gelem/s | 0.970 | per-call-when-materialized |
| inference.bf16f32.v1 | 3.7630 | 3.7562 | 1116.8588 GMAC/s | 1.737 | product |
| inference.int8i32.v1 | 1.9728 | 1.9625 | 2130.3446 GMAC/s | 0.911 | quantize.a+product |
| inference.int8i32.v1.applechunk | not run | | | | the column does not have the unit |
| training.bf16f32.v1 | 5.4179 | 5.4146 | 775.7077 GMAC/s | 2.501 | pack.b+product |
| training.int8i32.v1 | 14.1975 | 14.1198 | 296.0161 GMAC/s | 6.555 | quantize.a+pack.b+product |
| training.int8i32.v1.applechunk | not run | | | | the column does not have the unit |
| DERIVED int8i32.v1.flat + quantize.a | 19.8698 | | | 9.174 | sum of two medians |
| DERIVED int8i32.v1.mma + quantize.a | 1.9768 | | | 0.913 | sum of two medians |
| vendor strict (COMPARISON ONLY) | 1.0853 | 1.0780 | 3872.2242 GMAC/s | | fp32.v1 over this: 1.996 |
| vendor tf32 (COMPARISON ONLY) | 0.8214 | 0.8176 | 5116.2270 GMAC/s | | fp32.v1 over this: 2.637 |
| vendor bf16 (COMPARISON ONLY) | 0.3227 | 0.3197 | 13022.5131 GMAC/s | | fp32.v1 over this: 6.711 |

### llama8b.lm_head.t512: m=512 n=16032 k=4096 (CAPPED)

| arm | median ms | min ms | rate | arm over fp32.v1 | note |
|---|---:|---:|---:|---:|---|
| fp32.v1 | 3.7367 | 3.7296 | 8997.5813 GMAC/s | 1.000 | plan=10 |
| bf16f32.v1.fused | 156.4376 | 156.2763 | 214.9198 GMAC/s | 41.865 | not-dispatched(cells>16384) |
| bf16f32.v1.widen | 3.9342 | 3.9250 | 8545.9772 GMAC/s | 1.053 | dispatched |
| int8i32.v1.flat | 151.9419 | 151.8072 | 221.2790 GMAC/s | 40.662 | not-dispatched(the-column-has-the-unit) |
| int8i32.v1.mma | 5.1285 | 5.1228 | 6555.8810 GMAC/s | 1.372 | dispatched |
| int8i32.v1.applechunk | not run | | | | the column does not have the unit |
| convert.int8.quantize.a | 2.5252 | 2.5219 | 0.8305 Gelem/s | 0.676 | per-call |
| convert.int8.pack.b | 2.6325 | 2.6258 | 24.9443 Gelem/s | 0.704 | once-per-weight |
| convert.bf16.pack.b | 0.2197 | 0.2182 | 298.8875 Gelem/s | 0.059 | once-per-weight |
| convert.bf16.widen.b | 0.2087 | 0.2082 | 314.5954 Gelem/s | 0.056 | per-call-when-materialized |
| convert.int8.dequantize.b | 0.2734 | 0.2719 | 240.1438 Gelem/s | 0.073 | per-call-when-materialized |
| inference.bf16f32.v1 | 3.9246 | 3.9171 | 8566.7834 GMAC/s | 1.050 | product |
| inference.int8i32.v1 | 7.5827 | 7.5666 | 4433.9690 GMAC/s | 2.029 | quantize.a+product |
| inference.int8i32.v1.applechunk | not run | | | | the column does not have the unit |
| training.bf16f32.v1 | 4.1322 | 4.1293 | 8136.5259 GMAC/s | 1.106 | pack.b+product |
| training.int8i32.v1 | 10.1328 | 10.1145 | 3318.0876 GMAC/s | 2.712 | quantize.a+pack.b+product |
| training.int8i32.v1.applechunk | not run | | | | the column does not have the unit |
| DERIVED int8i32.v1.flat + quantize.a | 154.4671 | | | 41.338 | sum of two medians |
| DERIVED int8i32.v1.mma + quantize.a | 7.6537 | | | 2.048 | sum of two medians |
| vendor strict (COMPARISON ONLY) | 15.5631 | 14.2802 | | n/a | the vendor ran the whole row (n=128256), ours ran capped: no ratio |
| vendor tf32 (COMPARISON ONLY) | 2.3105 | 2.2393 | | n/a | the vendor ran the whole row (n=128256), ours ran capped: no ratio |
| vendor bf16 (COMPARISON ONLY) | 1.2070 | 1.1933 | | n/a | the vendor ran the whole row (n=128256), ours ran capped: no ratio |

## m3ultra (column apple)

Run 1 on this box, steward request 1790651983032, timing 2026-09-29T03:21:43Z to 03:23:20Z, commit f1e6cb61c. TAKEN ALONE as a steward speed job (the steward times beside no other job), after the gate and the probe's gate had run in the same job; this lane made no contact with the box from 03:19:49Z to 03:40:22Z. The harness was compiled in this run; every arm has an untimed warm-up call before its timed ones.

Vendor comparison: MPS/MPSGraph on Apple GPU, torch 2.13.0, torch 2.13.0, median of 10. COMPARISON ONLY, no digest, no identity claim.

### llama8b.qkv.t1: m=1 n=4096 k=4096 (FULL)

| arm | median ms | min ms | rate | arm over fp32.v1 | note |
|---|---:|---:|---:|---:|---|
| fp32.v1 | 1.0240 | 0.9790 | 16.3840 GMAC/s | 1.000 | plan=0 |
| bf16f32.v1.fused | 1.0150 | 0.9590 | 16.5293 GMAC/s | 0.991 | dispatched |
| bf16f32.v1.widen | 1.5800 | 1.5620 | 10.6185 GMAC/s | 1.543 | not-dispatched(cells<=16384) |
| int8i32.v1.flat | 0.7490 | 0.6600 | 22.3995 GMAC/s | 0.731 | dispatched |
| int8i32.v1.mma | not run | | | | the column does not have the unit |
| int8i32.v1.applechunk | 0.7640 | 0.6930 | 21.9597 GMAC/s | 0.746 | probe,geometry=1 |
| convert.int8.quantize.a | 1.1580 | 1.1220 | 0.0035 Gelem/s | 1.131 | per-call |
| convert.int8.pack.b | 1.5550 | 1.5360 | 10.7892 Gelem/s | 1.519 | once-per-weight |
| convert.bf16.pack.b | 0.8700 | 0.7600 | 19.2842 Gelem/s | 0.850 | once-per-weight |
| convert.bf16.widen.b | 0.8710 | 0.8250 | 19.2620 Gelem/s | 0.851 | per-call-when-materialized |
| convert.int8.dequantize.b | 1.1430 | 1.0740 | 14.6782 Gelem/s | 1.116 | per-call-when-materialized |
| inference.bf16f32.v1 | 0.9520 | 0.9270 | 17.6231 GMAC/s | 0.930 | product |
| inference.int8i32.v1 | 1.5200 | 1.5040 | 11.0376 GMAC/s | 1.484 | quantize.a+product |
| inference.int8i32.v1.applechunk | 1.5390 | 1.5290 | 10.9014 GMAC/s | 1.503 | quantize.a+probe |
| training.bf16f32.v1 | 1.5340 | 1.4900 | 10.9369 GMAC/s | 1.498 | pack.b+product |
| training.int8i32.v1 | 2.8160 | 2.8010 | 5.9578 GMAC/s | 2.750 | quantize.a+pack.b+product |
| training.int8i32.v1.applechunk | 2.8780 | 2.8650 | 5.8295 GMAC/s | 2.811 | quantize.a+pack.b+probe |
| DERIVED int8i32.v1.flat + quantize.a | 1.9070 | | | 1.862 | sum of two medians |
| DERIVED int8i32.v1.applechunk + quantize.a | 1.9220 | | | 1.877 | sum of two medians |
| vendor strict (COMPARISON ONLY) | 0.6626 | 0.3682 | 25.3193 GMAC/s | | fp32.v1 over this: 1.545 |
| vendor bf16 (COMPARISON ONLY) | 0.3878 | 0.2775 | 43.2634 GMAC/s | | fp32.v1 over this: 2.641 |

### llama8b.qkv.t8: m=8 n=4096 k=4096 (FULL)

| arm | median ms | min ms | rate | arm over fp32.v1 | note |
|---|---:|---:|---:|---:|---|
| fp32.v1 | 0.5420 | 0.5400 | 247.6342 GMAC/s | 1.000 | plan=20 |
| bf16f32.v1.fused | 1.7130 | 1.6290 | 78.3524 GMAC/s | 3.161 | not-dispatched(cells>16384) |
| bf16f32.v1.widen | 1.0750 | 1.0490 | 124.8537 GMAC/s | 1.983 | dispatched |
| int8i32.v1.flat | 0.9800 | 0.8860 | 136.9569 GMAC/s | 1.808 | dispatched |
| int8i32.v1.mma | not run | | | | the column does not have the unit |
| int8i32.v1.applechunk | 0.7570 | 0.7170 | 177.3022 GMAC/s | 1.397 | probe,geometry=1 |
| convert.int8.quantize.a | 1.2090 | 1.2050 | 0.0271 Gelem/s | 2.231 | per-call |
| convert.int8.pack.b | 1.5460 | 1.5070 | 10.8520 Gelem/s | 2.852 | once-per-weight |
| convert.bf16.pack.b | 0.8460 | 0.7940 | 19.8312 Gelem/s | 1.561 | once-per-weight |
| convert.bf16.widen.b | 0.8580 | 0.7980 | 19.5539 Gelem/s | 1.583 | per-call-when-materialized |
| convert.int8.dequantize.b | 1.1350 | 1.0670 | 14.7817 Gelem/s | 2.094 | per-call-when-materialized |
| inference.bf16f32.v1 | 1.1190 | 1.0510 | 119.9444 GMAC/s | 2.065 | product |
| inference.int8i32.v1 | 1.9230 | 1.8340 | 69.7960 GMAC/s | 3.548 | quantize.a+product |
| inference.int8i32.v1.applechunk | 1.6450 | 1.6140 | 81.5913 GMAC/s | 3.035 | quantize.a+probe |
| training.bf16f32.v1 | 1.6640 | 1.6470 | 80.6597 GMAC/s | 3.070 | pack.b+product |
| training.int8i32.v1 | 3.2140 | 3.0770 | 41.7603 GMAC/s | 5.930 | quantize.a+pack.b+product |
| training.int8i32.v1.applechunk | 2.9790 | 2.9530 | 45.0546 GMAC/s | 5.496 | quantize.a+pack.b+probe |
| DERIVED int8i32.v1.flat + quantize.a | 2.1890 | | | 4.039 | sum of two medians |
| DERIVED int8i32.v1.applechunk + quantize.a | 1.9660 | | | 3.627 | sum of two medians |
| vendor strict (COMPARISON ONLY) | 0.9505 | 0.5983 | 141.2137 GMAC/s | | fp32.v1 over this: 0.570 |
| vendor bf16 (COMPARISON ONLY) | 0.3892 | 0.3339 | 344.8293 GMAC/s | | fp32.v1 over this: 1.392 |

### llama8b.qkv.t512: m=512 n=4096 k=4096 (FULL)

| arm | median ms | min ms | rate | arm over fp32.v1 | note |
|---|---:|---:|---:|---:|---|
| fp32.v1 | 3.3200 | 3.2830 | 2587.3297 GMAC/s | 1.000 | plan=20 |
| bf16f32.v1.fused | 54.7230 | 54.6230 | 156.9712 GMAC/s | 16.483 | not-dispatched(cells>16384) |
| bf16f32.v1.widen | 3.8250 | 3.8060 | 2245.7345 GMAC/s | 1.152 | dispatched |
| int8i32.v1.flat | 26.5880 | 26.5070 | 323.0756 GMAC/s | 8.008 | dispatched |
| int8i32.v1.mma | not run | | | | the column does not have the unit |
| int8i32.v1.applechunk | 2.2450 | 2.2090 | 3826.2515 GMAC/s | 0.676 | probe,geometry=0 |
| convert.int8.quantize.a | 1.4230 | 1.4070 | 1.4738 Gelem/s | 0.429 | per-call |
| convert.int8.pack.b | 1.5490 | 1.5160 | 10.8310 Gelem/s | 0.467 | once-per-weight |
| convert.bf16.pack.b | 0.8000 | 0.7870 | 20.9715 Gelem/s | 0.241 | once-per-weight |
| convert.bf16.widen.b | 0.8600 | 0.8060 | 19.5084 Gelem/s | 0.259 | per-call-when-materialized |
| convert.int8.dequantize.b | 1.0760 | 1.0700 | 15.5922 Gelem/s | 0.324 | per-call-when-materialized |
| inference.bf16f32.v1 | 3.8350 | 3.8100 | 2239.8786 GMAC/s | 1.155 | product |
| inference.int8i32.v1 | 27.7370 | 27.6820 | 309.6923 GMAC/s | 8.355 | quantize.a+product |
| inference.int8i32.v1.applechunk | 3.3780 | 3.3680 | 2542.9054 GMAC/s | 1.017 | quantize.a+probe |
| training.bf16f32.v1 | 4.4200 | 4.4050 | 1943.4241 GMAC/s | 1.331 | pack.b+product |
| training.int8i32.v1 | 29.0150 | 29.0080 | 296.0515 GMAC/s | 8.739 | quantize.a+pack.b+product |
| training.int8i32.v1.applechunk | 4.6890 | 4.6690 | 1831.9332 GMAC/s | 1.412 | quantize.a+pack.b+probe |
| DERIVED int8i32.v1.flat + quantize.a | 28.0110 | | | 8.437 | sum of two medians |
| DERIVED int8i32.v1.applechunk + quantize.a | 3.6680 | | | 1.105 | sum of two medians |
| vendor strict (COMPARISON ONLY) | 1.6037 | 1.5305 | 5356.1556 GMAC/s | | fp32.v1 over this: 2.070 |
| vendor bf16 (COMPARISON ONLY) | 1.7423 | 1.4243 | 4930.2497 GMAC/s | | fp32.v1 over this: 1.906 |

### llama8b.mlp_up.t1: m=1 n=14336 k=4096 (FULL)

| arm | median ms | min ms | rate | arm over fp32.v1 | note |
|---|---:|---:|---:|---:|---|
| fp32.v1 | 1.0440 | 0.9670 | 56.2455 GMAC/s | 1.000 | plan=0 |
| bf16f32.v1.fused | 1.0160 | 0.9990 | 57.7955 GMAC/s | 0.973 | dispatched |
| bf16f32.v1.widen | 2.9430 | 2.9080 | 19.9525 GMAC/s | 2.819 | not-dispatched(cells<=16384) |
| int8i32.v1.flat | 0.7150 | 0.6980 | 82.1262 GMAC/s | 0.685 | dispatched |
| int8i32.v1.mma | not run | | | | the column does not have the unit |
| int8i32.v1.applechunk | 0.8180 | 0.7930 | 71.7852 GMAC/s | 0.784 | probe,geometry=1 |
| convert.int8.quantize.a | 1.1790 | 1.1590 | 0.0035 Gelem/s | 1.129 | per-call |
| convert.int8.pack.b | 1.5760 | 1.5670 | 37.2590 Gelem/s | 1.510 | once-per-weight |
| convert.bf16.pack.b | 2.2360 | 2.1870 | 26.2613 Gelem/s | 2.142 | once-per-weight |
| convert.bf16.widen.b | 2.2340 | 2.1880 | 26.2848 Gelem/s | 2.140 | per-call-when-materialized |
| convert.int8.dequantize.b | 3.2560 | 3.2290 | 18.0345 Gelem/s | 3.119 | per-call-when-materialized |
| inference.bf16f32.v1 | 0.9630 | 0.9430 | 60.9764 GMAC/s | 0.922 | product |
| inference.int8i32.v1 | 1.5540 | 1.5430 | 37.7865 GMAC/s | 1.489 | quantize.a+product |
| inference.int8i32.v1.applechunk | 1.6110 | 1.5990 | 36.4496 GMAC/s | 1.543 | quantize.a+probe |
| training.bf16f32.v1 | 2.9450 | 2.8830 | 19.9390 GMAC/s | 2.821 | pack.b+product |
| training.int8i32.v1 | 2.9160 | 2.8840 | 20.1373 GMAC/s | 2.793 | quantize.a+pack.b+product |
| training.int8i32.v1.applechunk | 2.9750 | 2.9600 | 19.7379 GMAC/s | 2.850 | quantize.a+pack.b+probe |
| DERIVED int8i32.v1.flat + quantize.a | 1.8940 | | | 1.814 | sum of two medians |
| DERIVED int8i32.v1.applechunk + quantize.a | 1.9970 | | | 1.913 | sum of two medians |
| vendor strict (COMPARISON ONLY) | 0.5867 | 0.5210 | 100.0913 GMAC/s | | fp32.v1 over this: 1.780 |
| vendor bf16 (COMPARISON ONLY) | 1.0076 | 0.3954 | 58.2759 GMAC/s | | fp32.v1 over this: 1.036 |

### llama8b.mlp_up.t8: m=8 n=14336 k=4096 (FULL)

| arm | median ms | min ms | rate | arm over fp32.v1 | note |
|---|---:|---:|---:|---:|---|
| fp32.v1 | 1.0200 | 0.9330 | 460.5510 GMAC/s | 1.000 | plan=20 |
| bf16f32.v1.fused | 3.6300 | 3.6100 | 129.4110 GMAC/s | 3.559 | not-dispatched(cells>16384) |
| bf16f32.v1.widen | 2.9720 | 2.8990 | 158.0626 GMAC/s | 2.914 | dispatched |
| int8i32.v1.flat | 2.1560 | 2.0730 | 217.8859 GMAC/s | 2.114 | dispatched |
| int8i32.v1.mma | not run | | | | the column does not have the unit |
| int8i32.v1.applechunk | 0.8260 | 0.7120 | 568.7192 GMAC/s | 0.810 | probe,geometry=1 |
| convert.int8.quantize.a | 1.2510 | 1.2130 | 0.0262 Gelem/s | 1.226 | per-call |
| convert.int8.pack.b | 1.6320 | 1.5790 | 35.9805 Gelem/s | 1.600 | once-per-weight |
| convert.bf16.pack.b | 2.2410 | 2.1980 | 26.2027 Gelem/s | 2.197 | once-per-weight |
| convert.bf16.widen.b | 2.2470 | 2.2140 | 26.1327 Gelem/s | 2.203 | per-call-when-materialized |
| convert.int8.dequantize.b | 3.2630 | 3.2630 | 17.9958 Gelem/s | 3.199 | per-call-when-materialized |
| inference.bf16f32.v1 | 2.9900 | 2.9020 | 157.1111 GMAC/s | 2.931 | product |
| inference.int8i32.v1 | 3.0540 | 3.0120 | 153.8186 GMAC/s | 2.994 | quantize.a+product |
| inference.int8i32.v1.applechunk | 1.7060 | 1.6920 | 275.3588 GMAC/s | 1.673 | quantize.a+probe |
| training.bf16f32.v1 | 4.9190 | 4.8840 | 95.4995 GMAC/s | 4.823 | pack.b+product |
| training.int8i32.v1 | 4.3630 | 4.3000 | 107.6695 GMAC/s | 4.277 | quantize.a+pack.b+product |
| training.int8i32.v1.applechunk | 3.0480 | 2.9940 | 154.1214 GMAC/s | 2.988 | quantize.a+pack.b+probe |
| DERIVED int8i32.v1.flat + quantize.a | 3.4070 | | | 3.340 | sum of two medians |
| DERIVED int8i32.v1.applechunk + quantize.a | 2.0770 | | | 2.036 | sum of two medians |
| vendor strict (COMPARISON ONLY) | 1.3591 | 1.2904 | 345.6515 GMAC/s | | fp32.v1 over this: 0.751 |
| vendor bf16 (COMPARISON ONLY) | 0.5105 | 0.4960 | 920.2756 GMAC/s | | fp32.v1 over this: 1.998 |

### llama8b.mlp_up.t512: m=512 n=14336 k=4096 (FULL)

| arm | median ms | min ms | rate | arm over fp32.v1 | note |
|---|---:|---:|---:|---:|---|
| fp32.v1 | 10.4170 | 10.4020 | 2886.1257 GMAC/s | 1.000 | plan=20 |
| bf16f32.v1.fused | 189.1560 | 188.6940 | 158.9417 GMAC/s | 18.158 | not-dispatched(cells>16384) |
| bf16f32.v1.widen | 12.3530 | 12.3370 | 2433.8032 GMAC/s | 1.186 | dispatched |
| int8i32.v1.flat | 91.1190 | 91.0800 | 329.9506 GMAC/s | 8.747 | dispatched |
| int8i32.v1.mma | not run | | | | the column does not have the unit |
| int8i32.v1.applechunk | 6.0770 | 6.0600 | 4947.3048 GMAC/s | 0.583 | probe,geometry=0 |
| convert.int8.quantize.a | 1.4460 | 1.4180 | 1.4503 Gelem/s | 0.139 | per-call |
| convert.int8.pack.b | 1.6180 | 1.6020 | 36.2919 Gelem/s | 0.155 | once-per-weight |
| convert.bf16.pack.b | 2.2500 | 2.2410 | 26.0979 Gelem/s | 0.216 | once-per-weight |
| convert.bf16.widen.b | 2.2460 | 2.2340 | 26.1444 Gelem/s | 0.216 | per-call-when-materialized |
| convert.int8.dequantize.b | 3.2710 | 3.2600 | 17.9518 Gelem/s | 0.314 | per-call-when-materialized |
| inference.bf16f32.v1 | 12.3630 | 12.3520 | 2431.8346 GMAC/s | 1.187 | product |
| inference.int8i32.v1 | 92.3240 | 92.2620 | 325.6442 GMAC/s | 8.863 | quantize.a+product |
| inference.int8i32.v1.applechunk | 7.2710 | 7.2520 | 4134.8881 GMAC/s | 0.698 | quantize.a+probe |
| training.bf16f32.v1 | 14.3630 | 14.3520 | 2093.2097 GMAC/s | 1.379 | pack.b+product |
| training.int8i32.v1 | 93.6940 | 93.5990 | 320.8826 GMAC/s | 8.994 | quantize.a+pack.b+product |
| training.int8i32.v1.applechunk | 8.5710 | 8.5660 | 3507.7320 GMAC/s | 0.823 | quantize.a+pack.b+probe |
| DERIVED int8i32.v1.flat + quantize.a | 92.5650 | | | 8.886 | sum of two medians |
| DERIVED int8i32.v1.applechunk + quantize.a | 7.5230 | | | 0.722 | sum of two medians |
| vendor strict (COMPARISON ONLY) | 4.0949 | 4.0781 | 7341.9374 GMAC/s | | fp32.v1 over this: 2.544 |
| vendor bf16 (COMPARISON ONLY) | 3.7403 | 3.7321 | 8038.0813 GMAC/s | | fp32.v1 over this: 2.785 |

### llama8b.mlp_down.t1: m=1 n=4096 k=14336 (FULL)

| arm | median ms | min ms | rate | arm over fp32.v1 | note |
|---|---:|---:|---:|---:|---|
| fp32.v1 | 2.8720 | 2.8410 | 20.4458 GMAC/s | 1.000 | plan=0 |
| bf16f32.v1.fused | 2.7150 | 2.6970 | 21.6281 GMAC/s | 0.945 | dispatched |
| bf16f32.v1.widen | 4.7860 | 4.7780 | 12.2692 GMAC/s | 1.666 | not-dispatched(cells<=16384) |
| int8i32.v1.flat | 1.7910 | 1.7800 | 32.7863 GMAC/s | 0.624 | dispatched |
| int8i32.v1.mma | not run | | | | the column does not have the unit |
| int8i32.v1.applechunk | 1.9240 | 1.9140 | 30.5199 GMAC/s | 0.670 | probe,geometry=1 |
| convert.int8.quantize.a | 3.3790 | 3.3240 | 0.0042 Gelem/s | 1.177 | per-call |
| convert.int8.pack.b | 5.0710 | 5.0460 | 11.5796 Gelem/s | 1.766 | once-per-weight |
| convert.bf16.pack.b | 2.2580 | 2.2480 | 26.0054 Gelem/s | 0.786 | once-per-weight |
| convert.bf16.widen.b | 2.2470 | 2.2150 | 26.1327 Gelem/s | 0.782 | per-call-when-materialized |
| convert.int8.dequantize.b | 3.1630 | 3.1410 | 18.5647 Gelem/s | 1.101 | per-call-when-materialized |
| inference.bf16f32.v1 | 2.7100 | 2.6930 | 21.6680 GMAC/s | 0.944 | product |
| inference.int8i32.v1 | 4.8740 | 4.8470 | 12.0477 GMAC/s | 1.697 | quantize.a+product |
| inference.int8i32.v1.applechunk | 5.0030 | 4.9720 | 11.7370 GMAC/s | 1.742 | quantize.a+probe |
| training.bf16f32.v1 | 4.6810 | 4.6650 | 12.5444 GMAC/s | 1.630 | pack.b+product |
| training.int8i32.v1 | 9.6470 | 9.6040 | 6.0869 GMAC/s | 3.359 | quantize.a+pack.b+product |
| training.int8i32.v1.applechunk | 9.8710 | 9.7990 | 5.9488 GMAC/s | 3.437 | quantize.a+pack.b+probe |
| DERIVED int8i32.v1.flat + quantize.a | 5.1700 | | | 1.800 | sum of two medians |
| DERIVED int8i32.v1.applechunk + quantize.a | 5.3030 | | | 1.846 | sum of two medians |
| vendor strict (COMPARISON ONLY) | 0.5458 | 0.5267 | 107.5832 GMAC/s | | fp32.v1 over this: 5.262 |
| vendor bf16 (COMPARISON ONLY) | 0.3942 | 0.3676 | 148.9651 GMAC/s | | fp32.v1 over this: 7.286 |

### llama8b.mlp_down.t8: m=8 n=4096 k=14336 (FULL)

| arm | median ms | min ms | rate | arm over fp32.v1 | note |
|---|---:|---:|---:|---:|---|
| fp32.v1 | 1.3680 | 1.3520 | 343.3933 GMAC/s | 1.000 | plan=20 |
| bf16f32.v1.fused | 5.2120 | 5.1730 | 90.1309 GMAC/s | 3.810 | not-dispatched(cells>16384) |
| bf16f32.v1.widen | 3.3250 | 3.2660 | 141.2818 GMAC/s | 2.431 | dispatched |
| int8i32.v1.flat | 2.9650 | 2.9290 | 158.4358 GMAC/s | 2.167 | dispatched |
| int8i32.v1.mma | not run | | | | the column does not have the unit |
| int8i32.v1.applechunk | 1.9530 | 1.9410 | 240.5336 GMAC/s | 1.428 | probe,geometry=1 |
| convert.int8.quantize.a | 3.7820 | 3.7470 | 0.0303 Gelem/s | 2.765 | per-call |
| convert.int8.pack.b | 5.0650 | 5.0150 | 11.5933 Gelem/s | 3.702 | once-per-weight |
| convert.bf16.pack.b | 2.2510 | 2.2270 | 26.0863 Gelem/s | 1.645 | once-per-weight |
| convert.bf16.widen.b | 2.2430 | 2.1960 | 26.1793 Gelem/s | 1.640 | per-call-when-materialized |
| convert.int8.dequantize.b | 3.1630 | 3.1320 | 18.5647 Gelem/s | 2.312 | per-call-when-materialized |
| inference.bf16f32.v1 | 3.3480 | 3.3330 | 140.3112 GMAC/s | 2.447 | product |
| inference.int8i32.v1 | 6.4440 | 6.4260 | 72.8991 GMAC/s | 4.711 | quantize.a+product |
| inference.int8i32.v1.applechunk | 5.4550 | 5.4370 | 86.1159 GMAC/s | 3.988 | quantize.a+probe |
| training.bf16f32.v1 | 5.3270 | 5.2810 | 88.1851 GMAC/s | 3.894 | pack.b+product |
| training.int8i32.v1 | 11.3200 | 11.2430 | 41.4984 GMAC/s | 8.275 | quantize.a+pack.b+product |
| training.int8i32.v1.applechunk | 10.3290 | 10.2710 | 45.4799 GMAC/s | 7.550 | quantize.a+pack.b+probe |
| DERIVED int8i32.v1.flat + quantize.a | 6.7470 | | | 4.932 | sum of two medians |
| DERIVED int8i32.v1.applechunk + quantize.a | 5.7350 | | | 4.192 | sum of two medians |
| vendor strict (COMPARISON ONLY) | 1.4128 | 0.8782 | 332.4964 GMAC/s | | fp32.v1 over this: 0.968 |
| vendor bf16 (COMPARISON ONLY) | 0.4995 | 0.4083 | 940.4250 GMAC/s | | fp32.v1 over this: 2.739 |

### llama8b.mlp_down.t512: m=512 n=4096 k=14336 (FULL)

| arm | median ms | min ms | rate | arm over fp32.v1 | note |
|---|---:|---:|---:|---:|---|
| fp32.v1 | 11.4650 | 11.1980 | 2622.3089 GMAC/s | 1.000 | plan=20 |
| bf16f32.v1.fused | 191.1210 | 190.8750 | 157.3075 GMAC/s | 16.670 | not-dispatched(cells>16384) |
| bf16f32.v1.widen | 13.4240 | 13.1410 | 2239.6284 GMAC/s | 1.171 | dispatched |
| int8i32.v1.flat | 117.6940 | 117.1610 | 255.4486 GMAC/s | 10.266 | dispatched |
| int8i32.v1.mma | not run | | | | the column does not have the unit |
| int8i32.v1.applechunk | 7.1410 | 7.1350 | 4210.1626 GMAC/s | 0.623 | probe,geometry=0 |
| convert.int8.quantize.a | 4.7690 | 4.7350 | 1.5391 Gelem/s | 0.416 | per-call |
| convert.int8.pack.b | 5.0800 | 5.0630 | 11.5591 Gelem/s | 0.443 | once-per-weight |
| convert.bf16.pack.b | 2.2740 | 2.2400 | 25.8225 Gelem/s | 0.198 | once-per-weight |
| convert.bf16.widen.b | 2.2400 | 2.2270 | 26.2144 Gelem/s | 0.195 | per-call-when-materialized |
| convert.int8.dequantize.b | 3.1610 | 3.1450 | 18.5765 Gelem/s | 0.276 | per-call-when-materialized |
| inference.bf16f32.v1 | 13.4130 | 13.1410 | 2241.4651 GMAC/s | 1.170 | product |
| inference.int8i32.v1 | 122.2080 | 122.1300 | 246.0131 GMAC/s | 10.659 | quantize.a+product |
| inference.int8i32.v1.applechunk | 11.6260 | 11.5880 | 2585.9944 GMAC/s | 1.014 | quantize.a+probe |
| training.bf16f32.v1 | 15.1230 | 15.1040 | 1988.0163 GMAC/s | 1.319 | pack.b+product |
| training.int8i32.v1 | 126.9410 | 126.4750 | 236.8405 GMAC/s | 11.072 | quantize.a+pack.b+product |
| training.int8i32.v1.applechunk | 16.5030 | 16.4490 | 1821.7761 GMAC/s | 1.439 | quantize.a+pack.b+probe |
| DERIVED int8i32.v1.flat + quantize.a | 122.4630 | | | 10.681 | sum of two medians |
| DERIVED int8i32.v1.applechunk + quantize.a | 11.9100 | | | 1.039 | sum of two medians |
| vendor strict (COMPARISON ONLY) | 4.8532 | 4.8349 | 6194.8243 GMAC/s | | fp32.v1 over this: 2.362 |
| vendor bf16 (COMPARISON ONLY) | 4.3217 | 4.2989 | 6956.7533 GMAC/s | | fp32.v1 over this: 2.653 |

### llama8b.lm_head.t1: m=1 n=128256 k=4096 (FULL)

| arm | median ms | min ms | rate | arm over fp32.v1 | note |
|---|---:|---:|---:|---:|---|
| fp32.v1 | 4.0720 | 4.0460 | 129.0119 GMAC/s | 1.000 | plan=0 |
| bf16f32.v1.fused | 4.0230 | 4.0150 | 130.5833 GMAC/s | 0.988 | not-dispatched(cells>16384) |
| bf16f32.v1.widen | 21.4810 | 21.4520 | 24.4559 GMAC/s | 5.275 | dispatched |
| int8i32.v1.flat | 2.1830 | 2.1610 | 240.6489 GMAC/s | 0.536 | dispatched |
| int8i32.v1.mma | not run | | | | the column does not have the unit |
| int8i32.v1.applechunk | 2.8760 | 2.8580 | 182.6622 GMAC/s | 0.706 | probe,geometry=1 |
| convert.int8.quantize.a | 1.1880 | 1.1800 | 0.0034 Gelem/s | 0.292 | per-call |
| convert.int8.pack.b | 10.3390 | 10.2780 | 50.8112 Gelem/s | 2.539 | once-per-weight |
| convert.bf16.pack.b | 17.7460 | 17.7400 | 29.6031 Gelem/s | 4.358 | once-per-weight |
| convert.bf16.widen.b | 17.6730 | 17.6650 | 29.7254 Gelem/s | 4.340 | per-call-when-materialized |
| convert.int8.dequantize.b | 30.5200 | 30.5110 | 17.2129 Gelem/s | 7.495 | per-call-when-materialized |
| inference.bf16f32.v1 | 21.4680 | 21.4630 | 24.4707 GMAC/s | 5.272 | product |
| inference.int8i32.v1 | 3.1180 | 3.0490 | 168.4851 GMAC/s | 0.766 | quantize.a+product |
| inference.int8i32.v1.applechunk | 3.7660 | 3.7450 | 139.4946 GMAC/s | 0.925 | quantize.a+probe |
| training.bf16f32.v1 | 38.9350 | 38.9260 | 13.4927 GMAC/s | 9.562 | pack.b+product |
| training.int8i32.v1 | 13.2120 | 13.1520 | 39.7621 GMAC/s | 3.245 | quantize.a+pack.b+product |
| training.int8i32.v1.applechunk | 13.9330 | 13.7540 | 37.7045 GMAC/s | 3.422 | quantize.a+pack.b+probe |
| DERIVED int8i32.v1.flat + quantize.a | 3.3710 | | | 0.828 | sum of two medians |
| DERIVED int8i32.v1.applechunk + quantize.a | 4.0640 | | | 0.998 | sum of two medians |
| vendor strict (COMPARISON ONLY) | 2.9303 | 2.8775 | 179.2766 GMAC/s | | fp32.v1 over this: 1.390 |
| vendor bf16 (COMPARISON ONLY) | 1.9377 | 1.5345 | 271.1153 GMAC/s | | fp32.v1 over this: 2.101 |

### llama8b.lm_head.t8: m=8 n=128256 k=4096 (FULL)

| arm | median ms | min ms | rate | arm over fp32.v1 | note |
|---|---:|---:|---:|---:|---|
| fp32.v1 | 5.7290 | 5.6690 | 733.5822 GMAC/s | 1.000 | plan=20 |
| bf16f32.v1.fused | 27.3660 | 27.2300 | 153.5735 GMAC/s | 4.777 | not-dispatched(cells>16384) |
| bf16f32.v1.widen | 23.0470 | 23.0160 | 182.3531 GMAC/s | 4.023 | dispatched |
| int8i32.v1.flat | 13.4380 | 13.3840 | 312.7469 GMAC/s | 2.346 | dispatched |
| int8i32.v1.mma | not run | | | | the column does not have the unit |
| int8i32.v1.applechunk | 2.9300 | 2.9160 | 1434.3661 GMAC/s | 0.511 | probe,geometry=1 |
| convert.int8.quantize.a | 1.1910 | 1.1750 | 0.0275 Gelem/s | 0.208 | per-call |
| convert.int8.pack.b | 10.3030 | 10.1710 | 50.9887 Gelem/s | 1.798 | once-per-weight |
| convert.bf16.pack.b | 17.8240 | 17.7480 | 29.4736 Gelem/s | 3.111 | once-per-weight |
| convert.bf16.widen.b | 17.7230 | 17.6700 | 29.6415 Gelem/s | 3.094 | per-call-when-materialized |
| convert.int8.dequantize.b | 30.5390 | 30.5300 | 17.2022 Gelem/s | 5.331 | per-call-when-materialized |
| inference.bf16f32.v1 | 23.1400 | 23.0270 | 181.6203 GMAC/s | 4.039 | product |
| inference.int8i32.v1 | 14.3620 | 14.2660 | 292.6259 GMAC/s | 2.507 | quantize.a+product |
| inference.int8i32.v1.applechunk | 3.9500 | 3.8130 | 1063.9728 GMAC/s | 0.689 | quantize.a+probe |
| training.bf16f32.v1 | 40.6730 | 40.5000 | 103.3288 GMAC/s | 7.099 | pack.b+product |
| training.int8i32.v1 | 24.5000 | 24.4060 | 171.5385 GMAC/s | 4.276 | quantize.a+pack.b+product |
| training.int8i32.v1.applechunk | 13.9350 | 13.8620 | 301.5926 GMAC/s | 2.432 | quantize.a+pack.b+probe |
| DERIVED int8i32.v1.flat + quantize.a | 14.6290 | | | 2.553 | sum of two medians |
| DERIVED int8i32.v1.applechunk + quantize.a | 4.1210 | | | 0.719 | sum of two medians |
| vendor strict (COMPARISON ONLY) | 5.2802 | 5.2510 | 795.9393 GMAC/s | | fp32.v1 over this: 1.085 |
| vendor bf16 (COMPARISON ONLY) | 2.0810 | 1.5751 | 2019.5141 GMAC/s | | fp32.v1 over this: 2.753 |

### llama8b.lm_head.t512: m=512 n=16032 k=4096 (CAPPED)

| arm | median ms | min ms | rate | arm over fp32.v1 | note |
|---|---:|---:|---:|---:|---|
| fp32.v1 | 11.5820 | 11.5640 | 2902.9132 GMAC/s | 1.000 | plan=20 |
| bf16f32.v1.fused | 211.5180 | 210.7490 | 158.9536 GMAC/s | 18.263 | not-dispatched(cells>16384) |
| bf16f32.v1.widen | 13.7680 | 13.7420 | 2442.0062 GMAC/s | 1.189 | dispatched |
| int8i32.v1.flat | 102.7290 | 102.5350 | 327.2838 GMAC/s | 8.870 | dispatched |
| int8i32.v1.mma | not run | | | | the column does not have the unit |
| int8i32.v1.applechunk | 6.7970 | 6.6570 | 4946.5265 GMAC/s | 0.587 | probe,geometry=0 |
| convert.int8.quantize.a | 1.4530 | 1.4190 | 1.4433 Gelem/s | 0.125 | per-call |
| convert.int8.pack.b | 1.8510 | 1.8410 | 35.4765 Gelem/s | 0.160 | once-per-weight |
| convert.bf16.pack.b | 2.4640 | 2.4050 | 26.6506 Gelem/s | 0.213 | once-per-weight |
| convert.bf16.widen.b | 2.4600 | 2.4320 | 26.6939 Gelem/s | 0.212 | per-call-when-materialized |
| convert.int8.dequantize.b | 3.6350 | 3.6030 | 18.0652 Gelem/s | 0.314 | per-call-when-materialized |
| inference.bf16f32.v1 | 13.7660 | 13.7430 | 2442.3610 GMAC/s | 1.189 | product |
| inference.int8i32.v1 | 103.8610 | 103.7820 | 323.7167 GMAC/s | 8.967 | quantize.a+product |
| inference.int8i32.v1.applechunk | 7.8480 | 7.7930 | 4284.0903 GMAC/s | 0.678 | quantize.a+probe |
| training.bf16f32.v1 | 15.9530 | 15.9340 | 2107.5372 GMAC/s | 1.377 | pack.b+product |
| training.int8i32.v1 | 105.4530 | 105.4240 | 318.8296 GMAC/s | 9.105 | quantize.a+pack.b+product |
| training.int8i32.v1.applechunk | 9.5210 | 9.3620 | 3531.3035 GMAC/s | 0.822 | quantize.a+pack.b+probe |
| DERIVED int8i32.v1.flat + quantize.a | 104.1820 | | | 8.995 | sum of two medians |
| DERIVED int8i32.v1.applechunk + quantize.a | 8.2500 | | | 0.712 | sum of two medians |
| vendor strict (COMPARISON ONLY) | 40.3968 | 40.1778 | | n/a | the vendor ran the whole row (n=128256), ours ran capped: no ratio |
| vendor bf16 (COMPARISON ONLY) | 36.5396 | 36.4338 | | n/a | the vendor ran the whole row (n=128256), ours ran capped: no ratio |

## m2pro (column apple)

Run 3, steward request 1790651263666, timing 2026-09-29T03:08:10Z to 03:12:37Z, commit d4948864a. TAKEN ALONE as a steward speed job; this lane made no contact with the box from the submit to 03:19:03Z.

Vendor comparison: MPS/MPSGraph on Apple GPU, torch 2.13.0, torch 2.13.0, median of 10. COMPARISON ONLY, no digest, no identity claim.

### llama8b.qkv.t1: m=1 n=4096 k=4096 (FULL)

| arm | median ms | min ms | rate | arm over fp32.v1 | note |
|---|---:|---:|---:|---:|---|
| fp32.v1 | 1.5170 | 1.5020 | 11.0595 GMAC/s | 1.000 | plan=0 |
| bf16f32.v1.fused | 1.4690 | 1.4420 | 11.4208 GMAC/s | 0.968 | dispatched |
| bf16f32.v1.widen | 2.0390 | 2.0220 | 8.2282 GMAC/s | 1.344 | not-dispatched(cells<=16384) |
| int8i32.v1.flat | 1.3570 | 1.3460 | 12.3635 GMAC/s | 0.895 | dispatched |
| int8i32.v1.mma | not run | | | | the column does not have the unit |
| int8i32.v1.applechunk | 0.8090 | 0.8030 | 20.7382 GMAC/s | 0.533 | probe,geometry=1 |
| convert.int8.quantize.a | 1.3420 | 1.3380 | 0.0031 Gelem/s | 0.885 | per-call |
| convert.int8.pack.b | 3.7870 | 3.7840 | 4.4302 Gelem/s | 2.496 | once-per-weight |
| convert.bf16.pack.b | 1.0410 | 1.0290 | 16.1164 Gelem/s | 0.686 | once-per-weight |
| convert.bf16.widen.b | 0.7160 | 0.7060 | 23.4319 Gelem/s | 0.472 | per-call-when-materialized |
| convert.int8.dequantize.b | 3.6920 | 3.6810 | 4.5442 Gelem/s | 2.434 | per-call-when-materialized |
| inference.bf16f32.v1 | 1.4640 | 1.4560 | 11.4598 GMAC/s | 0.965 | product |
| inference.int8i32.v1 | 2.5170 | 2.5030 | 6.6656 GMAC/s | 1.659 | quantize.a+product |
| inference.int8i32.v1.applechunk | 1.9640 | 1.9550 | 8.5424 GMAC/s | 1.295 | quantize.a+probe |
| training.bf16f32.v1 | 2.3450 | 2.3210 | 7.1545 GMAC/s | 1.546 | pack.b+product |
| training.int8i32.v1 | 6.1170 | 6.0990 | 2.7427 GMAC/s | 4.032 | quantize.a+pack.b+product |
| training.int8i32.v1.applechunk | 5.5970 | 5.5700 | 2.9975 GMAC/s | 3.690 | quantize.a+pack.b+probe |
| DERIVED int8i32.v1.flat + quantize.a | 2.6990 | | | 1.779 | sum of two medians |
| DERIVED int8i32.v1.applechunk + quantize.a | 2.1510 | | | 1.418 | sum of two medians |
| vendor strict (COMPARISON ONLY) | 0.9410 | 0.7645 | 17.8299 GMAC/s | | fp32.v1 over this: 1.612 |
| vendor bf16 (COMPARISON ONLY) | 0.6908 | 0.6790 | 24.2869 GMAC/s | | fp32.v1 over this: 2.196 |

### llama8b.qkv.t8: m=8 n=4096 k=4096 (FULL)

| arm | median ms | min ms | rate | arm over fp32.v1 | note |
|---|---:|---:|---:|---:|---|
| fp32.v1 | 1.4930 | 1.4730 | 89.8980 GMAC/s | 1.000 | plan=20 |
| bf16f32.v1.fused | 9.4700 | 9.3290 | 14.1729 GMAC/s | 6.343 | not-dispatched(cells>16384) |
| bf16f32.v1.widen | 2.0760 | 2.0540 | 64.6521 GMAC/s | 1.390 | dispatched |
| int8i32.v1.flat | 9.7600 | 9.7150 | 13.7518 GMAC/s | 6.537 | dispatched |
| int8i32.v1.mma | not run | | | | the column does not have the unit |
| int8i32.v1.applechunk | 0.8360 | 0.8290 | 160.5475 GMAC/s | 0.560 | probe,geometry=1 |
| convert.int8.quantize.a | 1.3970 | 1.3480 | 0.0235 Gelem/s | 0.936 | per-call |
| convert.int8.pack.b | 3.8150 | 3.7980 | 4.3977 Gelem/s | 2.555 | once-per-weight |
| convert.bf16.pack.b | 1.0560 | 1.0370 | 15.8875 Gelem/s | 0.707 | once-per-weight |
| convert.bf16.widen.b | 0.7180 | 0.7180 | 23.3666 Gelem/s | 0.481 | per-call-when-materialized |
| convert.int8.dequantize.b | 3.6960 | 3.6880 | 4.5393 Gelem/s | 2.476 | per-call-when-materialized |
| inference.bf16f32.v1 | 2.0460 | 1.9970 | 65.6001 GMAC/s | 1.370 | product |
| inference.int8i32.v1 | 10.8990 | 10.8730 | 12.3147 GMAC/s | 7.300 | quantize.a+product |
| inference.int8i32.v1.applechunk | 2.0120 | 2.0090 | 66.7086 GMAC/s | 1.348 | quantize.a+probe |
| training.bf16f32.v1 | 2.9140 | 2.8880 | 46.0596 GMAC/s | 1.952 | pack.b+product |
| training.int8i32.v1 | 14.5100 | 14.4920 | 9.2500 GMAC/s | 9.719 | quantize.a+pack.b+product |
| training.int8i32.v1.applechunk | 5.6440 | 5.6430 | 23.7806 GMAC/s | 3.780 | quantize.a+pack.b+probe |
| DERIVED int8i32.v1.flat + quantize.a | 11.1570 | | | 7.473 | sum of two medians |
| DERIVED int8i32.v1.applechunk + quantize.a | 2.2330 | | | 1.496 | sum of two medians |
| vendor strict (COMPARISON ONLY) | 1.9498 | 1.1585 | 68.8362 GMAC/s | | fp32.v1 over this: 0.766 |
| vendor bf16 (COMPARISON ONLY) | 1.1721 | 1.1673 | 114.5121 GMAC/s | | fp32.v1 over this: 1.274 |

### llama8b.qkv.t512: m=512 n=4096 k=4096 (FULL)

| arm | median ms | min ms | rate | arm over fp32.v1 | note |
|---|---:|---:|---:|---:|---|
| fp32.v1 | 17.7380 | 17.6820 | 484.2674 GMAC/s | 1.000 | plan=20 |
| bf16f32.v1.fused | 513.7060 | 513.4490 | 16.7215 GMAC/s | 28.961 | not-dispatched(cells>16384) |
| bf16f32.v1.widen | 18.3010 | 18.2180 | 469.3697 GMAC/s | 1.032 | dispatched |
| int8i32.v1.flat | 514.8840 | 513.8510 | 16.6832 GMAC/s | 29.027 | dispatched |
| int8i32.v1.mma | not run | | | | the column does not have the unit |
| int8i32.v1.applechunk | 8.6110 | 8.5720 | 997.5537 GMAC/s | 0.485 | probe,geometry=0 |
| convert.int8.quantize.a | 3.4800 | 3.4290 | 0.6026 Gelem/s | 0.196 | per-call |
| convert.int8.pack.b | 3.8480 | 3.8170 | 4.3600 Gelem/s | 0.217 | once-per-weight |
| convert.bf16.pack.b | 1.0670 | 1.0270 | 15.7237 Gelem/s | 0.060 | once-per-weight |
| convert.bf16.widen.b | 0.7270 | 0.7180 | 23.0773 Gelem/s | 0.041 | per-call-when-materialized |
| convert.int8.dequantize.b | 3.7260 | 3.6790 | 4.5027 Gelem/s | 0.210 | per-call-when-materialized |
| inference.bf16f32.v1 | 18.1610 | 18.0720 | 472.9880 GMAC/s | 1.024 | product |
| inference.int8i32.v1 | 517.9090 | 517.2840 | 16.5858 GMAC/s | 29.198 | quantize.a+product |
| inference.int8i32.v1.applechunk | 11.8310 | 11.8220 | 726.0531 GMAC/s | 0.667 | quantize.a+probe |
| training.bf16f32.v1 | 19.0250 | 19.0130 | 451.5077 GMAC/s | 1.073 | pack.b+product |
| training.int8i32.v1 | 521.6280 | 521.2560 | 16.4675 GMAC/s | 29.407 | quantize.a+pack.b+product |
| training.int8i32.v1.applechunk | 15.4370 | 15.4200 | 556.4510 GMAC/s | 0.870 | quantize.a+pack.b+probe |
| DERIVED int8i32.v1.flat + quantize.a | 518.3640 | | | 29.223 | sum of two medians |
| DERIVED int8i32.v1.applechunk + quantize.a | 12.0910 | | | 0.682 | sum of two medians |
| vendor strict (COMPARISON ONLY) | 3.6535 | 3.2435 | 2351.1789 GMAC/s | | fp32.v1 over this: 4.855 |
| vendor bf16 (COMPARISON ONLY) | 5.6663 | 5.6503 | 1515.9712 GMAC/s | | fp32.v1 over this: 3.130 |

### llama8b.mlp_up.t1: m=1 n=14336 k=4096 (FULL)

| arm | median ms | min ms | rate | arm over fp32.v1 | note |
|---|---:|---:|---:|---:|---|
| fp32.v1 | 5.2580 | 5.1950 | 11.1678 GMAC/s | 1.000 | plan=0 |
| bf16f32.v1.fused | 5.2000 | 5.1880 | 11.2924 GMAC/s | 0.989 | dispatched |
| bf16f32.v1.widen | 7.1240 | 7.0660 | 8.2426 GMAC/s | 1.355 | not-dispatched(cells<=16384) |
| int8i32.v1.flat | 4.9860 | 4.9820 | 11.7770 GMAC/s | 0.948 | dispatched |
| int8i32.v1.mma | not run | | | | the column does not have the unit |
| int8i32.v1.applechunk | 2.0230 | 1.9830 | 29.0263 GMAC/s | 0.385 | probe,geometry=1 |
| convert.int8.quantize.a | 1.3290 | 1.3280 | 0.0031 Gelem/s | 0.253 | per-call |
| convert.int8.pack.b | 13.6600 | 13.6470 | 4.2987 Gelem/s | 2.598 | once-per-weight |
| convert.bf16.pack.b | 3.1170 | 3.1090 | 18.8387 Gelem/s | 0.593 | once-per-weight |
| convert.bf16.widen.b | 2.0960 | 2.0720 | 28.0154 Gelem/s | 0.399 | per-call-when-materialized |
| convert.int8.dequantize.b | 14.0100 | 14.0070 | 4.1913 Gelem/s | 2.665 | per-call-when-materialized |
| inference.bf16f32.v1 | 5.2180 | 5.2040 | 11.2534 GMAC/s | 0.992 | product |
| inference.int8i32.v1 | 6.1320 | 6.1290 | 9.5760 GMAC/s | 1.166 | quantize.a+product |
| inference.int8i32.v1.applechunk | 3.1930 | 3.1900 | 18.3903 GMAC/s | 0.607 | quantize.a+probe |
| training.bf16f32.v1 | 7.9600 | 7.9350 | 7.3769 GMAC/s | 1.514 | pack.b+product |
| training.int8i32.v1 | 19.5900 | 19.5500 | 2.9975 GMAC/s | 3.726 | quantize.a+pack.b+product |
| training.int8i32.v1.applechunk | 16.6470 | 16.6180 | 3.5274 GMAC/s | 3.166 | quantize.a+pack.b+probe |
| DERIVED int8i32.v1.flat + quantize.a | 6.3150 | | | 1.201 | sum of two medians |
| DERIVED int8i32.v1.applechunk + quantize.a | 3.3520 | | | 0.638 | sum of two medians |
| vendor strict (COMPARISON ONLY) | 2.0024 | 1.5752 | 29.3253 GMAC/s | | fp32.v1 over this: 2.626 |
| vendor bf16 (COMPARISON ONLY) | 1.1575 | 0.8845 | 50.7293 GMAC/s | | fp32.v1 over this: 4.542 |

### llama8b.mlp_up.t8: m=8 n=14336 k=4096 (FULL)

| arm | median ms | min ms | rate | arm over fp32.v1 | note |
|---|---:|---:|---:|---:|---|
| fp32.v1 | 4.1060 | 4.0330 | 114.4087 GMAC/s | 1.000 | plan=20 |
| bf16f32.v1.fused | 30.2360 | 28.7660 | 15.5365 GMAC/s | 7.364 | not-dispatched(cells>16384) |
| bf16f32.v1.widen | 6.0270 | 6.0090 | 77.9429 GMAC/s | 1.468 | dispatched |
| int8i32.v1.flat | 29.4060 | 29.0390 | 15.9750 GMAC/s | 7.162 | dispatched |
| int8i32.v1.mma | not run | | | | the column does not have the unit |
| int8i32.v1.applechunk | 2.0320 | 2.0250 | 231.1821 GMAC/s | 0.495 | probe,geometry=1 |
| convert.int8.quantize.a | 1.4110 | 1.3950 | 0.0232 Gelem/s | 0.344 | per-call |
| convert.int8.pack.b | 13.6950 | 13.6860 | 4.2877 Gelem/s | 3.335 | once-per-weight |
| convert.bf16.pack.b | 3.1210 | 3.0830 | 18.8146 Gelem/s | 0.760 | once-per-weight |
| convert.bf16.widen.b | 2.1250 | 2.0690 | 27.6331 Gelem/s | 0.518 | per-call-when-materialized |
| convert.int8.dequantize.b | 14.0320 | 14.0300 | 4.1847 Gelem/s | 3.417 | per-call-when-materialized |
| inference.bf16f32.v1 | 6.0150 | 5.9900 | 78.0984 GMAC/s | 1.465 | product |
| inference.int8i32.v1 | 30.5820 | 30.0780 | 15.3607 GMAC/s | 7.448 | quantize.a+product |
| inference.int8i32.v1.applechunk | 3.2170 | 3.1910 | 146.0249 GMAC/s | 0.783 | quantize.a+probe |
| training.bf16f32.v1 | 8.7460 | 8.7010 | 53.7116 GMAC/s | 2.130 | pack.b+product |
| training.int8i32.v1 | 44.0230 | 43.9680 | 10.6708 GMAC/s | 10.722 | quantize.a+pack.b+product |
| training.int8i32.v1.applechunk | 16.6700 | 16.6620 | 28.1801 GMAC/s | 4.060 | quantize.a+pack.b+probe |
| DERIVED int8i32.v1.flat + quantize.a | 30.8170 | | | 7.505 | sum of two medians |
| DERIVED int8i32.v1.applechunk + quantize.a | 3.4430 | | | 0.839 | sum of two medians |
| vendor strict (COMPARISON ONLY) | 2.0945 | 2.0532 | 224.2814 GMAC/s | | fp32.v1 over this: 1.960 |
| vendor bf16 (COMPARISON ONLY) | 1.5861 | 1.0657 | 296.1658 GMAC/s | | fp32.v1 over this: 2.589 |

### llama8b.mlp_up.t512: m=512 n=14336 k=4096 (FULL)

| arm | median ms | min ms | rate | arm over fp32.v1 | note |
|---|---:|---:|---:|---:|---|
| fp32.v1 | 60.5230 | 60.4680 | 496.7495 GMAC/s | 1.000 | plan=20 |
| bf16f32.v1.fused | 1794.7320 | 1793.4150 | 16.7517 GMAC/s | 29.654 | not-dispatched(cells>16384) |
| bf16f32.v1.widen | 62.3180 | 62.2100 | 482.4412 GMAC/s | 1.030 | dispatched |
| int8i32.v1.flat | 1799.5110 | 1799.1320 | 16.7072 GMAC/s | 29.733 | dispatched |
| int8i32.v1.mma | not run | | | | the column does not have the unit |
| int8i32.v1.applechunk | 28.0180 | 27.9960 | 1073.0520 GMAC/s | 0.463 | probe,geometry=0 |
| convert.int8.quantize.a | 3.5070 | 3.4600 | 0.5980 Gelem/s | 0.058 | per-call |
| convert.int8.pack.b | 13.7320 | 13.7080 | 4.2762 Gelem/s | 0.227 | once-per-weight |
| convert.bf16.pack.b | 3.1270 | 3.0970 | 18.7785 Gelem/s | 0.052 | once-per-weight |
| convert.bf16.widen.b | 2.1190 | 2.0770 | 27.7113 Gelem/s | 0.035 | per-call-when-materialized |
| convert.int8.dequantize.b | 14.0410 | 14.0290 | 4.1821 Gelem/s | 0.232 | per-call-when-materialized |
| inference.bf16f32.v1 | 62.3810 | 62.3260 | 481.9540 GMAC/s | 1.031 | product |
| inference.int8i32.v1 | 1802.4750 | 1801.9800 | 16.6797 GMAC/s | 29.782 | quantize.a+product |
| inference.int8i32.v1.applechunk | 31.1760 | 31.1180 | 964.3563 GMAC/s | 0.515 | quantize.a+probe |
| training.bf16f32.v1 | 65.2750 | 65.2290 | 460.5863 GMAC/s | 1.079 | pack.b+product |
| training.int8i32.v1 | 1815.9380 | 1815.8780 | 16.5561 GMAC/s | 30.004 | quantize.a+pack.b+product |
| training.int8i32.v1.applechunk | 44.6290 | 44.6050 | 673.6600 GMAC/s | 0.737 | quantize.a+pack.b+probe |
| DERIVED int8i32.v1.flat + quantize.a | 1803.0180 | | | 29.791 | sum of two medians |
| DERIVED int8i32.v1.applechunk + quantize.a | 31.5250 | | | 0.521 | sum of two medians |
| vendor strict (COMPARISON ONLY) | 10.4447 | 10.3346 | 2878.4691 GMAC/s | | fp32.v1 over this: 5.795 |
| vendor bf16 (COMPARISON ONLY) | 18.3253 | 18.3144 | 1640.6144 GMAC/s | | fp32.v1 over this: 3.303 |

### llama8b.mlp_down.t1: m=1 n=4096 k=14336 (FULL)

| arm | median ms | min ms | rate | arm over fp32.v1 | note |
|---|---:|---:|---:|---:|---|
| fp32.v1 | 5.5350 | 5.4950 | 10.6089 GMAC/s | 1.000 | plan=0 |
| bf16f32.v1.fused | 5.1380 | 5.1020 | 11.4286 GMAC/s | 0.928 | dispatched |
| bf16f32.v1.widen | 7.4530 | 7.4290 | 7.8787 GMAC/s | 1.347 | not-dispatched(cells<=16384) |
| int8i32.v1.flat | 4.4320 | 4.4060 | 13.2492 GMAC/s | 0.801 | dispatched |
| int8i32.v1.mma | not run | | | | the column does not have the unit |
| int8i32.v1.applechunk | 2.5020 | 2.4350 | 23.4693 GMAC/s | 0.452 | probe,geometry=1 |
| convert.int8.quantize.a | 4.3080 | 4.2190 | 0.0033 Gelem/s | 0.778 | per-call |
| convert.int8.pack.b | 13.8480 | 13.8130 | 4.2403 Gelem/s | 2.502 | once-per-weight |
| convert.bf16.pack.b | 2.9840 | 2.9640 | 19.6784 Gelem/s | 0.539 | once-per-weight |
| convert.bf16.widen.b | 2.1270 | 2.0650 | 27.6071 Gelem/s | 0.384 | per-call-when-materialized |
| convert.int8.dequantize.b | 13.2040 | 13.1880 | 4.4472 Gelem/s | 2.386 | per-call-when-materialized |
| inference.bf16f32.v1 | 5.1410 | 5.1110 | 11.4220 GMAC/s | 0.929 | product |
| inference.int8i32.v1 | 8.4750 | 8.4680 | 6.9286 GMAC/s | 1.531 | quantize.a+product |
| inference.int8i32.v1.applechunk | 6.5430 | 6.5110 | 8.9745 GMAC/s | 1.182 | quantize.a+probe |
| training.bf16f32.v1 | 7.9670 | 7.9640 | 7.3704 GMAC/s | 1.439 | pack.b+product |
| training.int8i32.v1 | 22.0780 | 21.9980 | 2.6597 GMAC/s | 3.989 | quantize.a+pack.b+product |
| training.int8i32.v1.applechunk | 20.1120 | 20.0770 | 2.9197 GMAC/s | 3.634 | quantize.a+pack.b+probe |
| DERIVED int8i32.v1.flat + quantize.a | 8.7400 | | | 1.579 | sum of two medians |
| DERIVED int8i32.v1.applechunk + quantize.a | 6.8100 | | | 1.230 | sum of two medians |
| vendor strict (COMPARISON ONLY) | 1.6173 | 1.6099 | 36.3078 GMAC/s | | fp32.v1 over this: 3.422 |
| vendor bf16 (COMPARISON ONLY) | 1.4094 | 1.0588 | 41.6634 GMAC/s | | fp32.v1 over this: 3.927 |

### llama8b.mlp_down.t8: m=8 n=4096 k=14336 (FULL)

| arm | median ms | min ms | rate | arm over fp32.v1 | note |
|---|---:|---:|---:|---:|---|
| fp32.v1 | 4.9780 | 4.9060 | 94.3676 GMAC/s | 1.000 | plan=20 |
| bf16f32.v1.fused | 34.5710 | 34.3430 | 13.5883 GMAC/s | 6.945 | not-dispatched(cells>16384) |
| bf16f32.v1.widen | 6.9070 | 6.8880 | 68.0125 GMAC/s | 1.388 | dispatched |
| int8i32.v1.flat | 33.9800 | 33.9480 | 13.8247 GMAC/s | 6.826 | dispatched |
| int8i32.v1.mma | not run | | | | the column does not have the unit |
| int8i32.v1.applechunk | 2.4910 | 2.4540 | 188.5837 GMAC/s | 0.500 | probe,geometry=1 |
| convert.int8.quantize.a | 4.4610 | 4.4370 | 0.0257 Gelem/s | 0.896 | per-call |
| convert.int8.pack.b | 14.0520 | 13.9540 | 4.1788 Gelem/s | 2.823 | once-per-weight |
| convert.bf16.pack.b | 3.0240 | 3.0080 | 19.4181 Gelem/s | 0.607 | once-per-weight |
| convert.bf16.widen.b | 2.1270 | 2.1180 | 27.6071 Gelem/s | 0.427 | per-call-when-materialized |
| convert.int8.dequantize.b | 13.2000 | 13.1940 | 4.4485 Gelem/s | 2.652 | per-call-when-materialized |
| inference.bf16f32.v1 | 6.9110 | 6.8790 | 67.9731 GMAC/s | 1.388 | product |
| inference.int8i32.v1 | 38.1740 | 38.1500 | 12.3058 GMAC/s | 7.669 | quantize.a+product |
| inference.int8i32.v1.applechunk | 6.6580 | 6.6540 | 70.5560 GMAC/s | 1.337 | quantize.a+probe |
| training.bf16f32.v1 | 9.6860 | 9.6670 | 48.4991 GMAC/s | 1.946 | pack.b+product |
| training.int8i32.v1 | 51.9610 | 51.8650 | 9.0407 GMAC/s | 10.438 | quantize.a+pack.b+product |
| training.int8i32.v1.applechunk | 20.4330 | 20.3540 | 22.9904 GMAC/s | 4.105 | quantize.a+pack.b+probe |
| DERIVED int8i32.v1.flat + quantize.a | 38.4410 | | | 7.722 | sum of two medians |
| DERIVED int8i32.v1.applechunk + quantize.a | 6.9520 | | | 1.397 | sum of two medians |
| vendor strict (COMPARISON ONLY) | 2.5292 | 2.1117 | 185.7333 GMAC/s | | fp32.v1 over this: 1.968 |
| vendor bf16 (COMPARISON ONLY) | 1.5827 | 1.1438 | 296.8050 GMAC/s | | fp32.v1 over this: 3.145 |

### llama8b.mlp_down.t512: m=512 n=4096 k=14336 (FULL)

| arm | median ms | min ms | rate | arm over fp32.v1 | note |
|---|---:|---:|---:|---:|---|
| fp32.v1 | 65.5290 | 64.6390 | 458.8010 GMAC/s | 1.000 | plan=20 |
| bf16f32.v1.fused | 1935.0970 | 1934.4810 | 15.5366 GMAC/s | 29.530 | not-dispatched(cells>16384) |
| bf16f32.v1.widen | 67.2640 | 66.9160 | 446.9667 GMAC/s | 1.026 | dispatched |
| int8i32.v1.flat | 1829.4860 | 1828.6780 | 16.4335 GMAC/s | 27.919 | dispatched |
| int8i32.v1.mma | not run | | | | the column does not have the unit |
| int8i32.v1.applechunk | 30.0130 | 29.9810 | 1001.7250 GMAC/s | 0.458 | probe,geometry=0 |
| convert.int8.quantize.a | 12.0230 | 11.9890 | 0.6105 Gelem/s | 0.183 | per-call |
| convert.int8.pack.b | 13.8820 | 13.8710 | 4.2300 Gelem/s | 0.212 | once-per-weight |
| convert.bf16.pack.b | 2.9620 | 2.9450 | 19.8245 Gelem/s | 0.045 | once-per-weight |
| convert.bf16.widen.b | 2.1240 | 2.0990 | 27.6461 Gelem/s | 0.032 | per-call-when-materialized |
| convert.int8.dequantize.b | 13.1940 | 13.1670 | 4.4505 Gelem/s | 0.201 | per-call-when-materialized |
| inference.bf16f32.v1 | 67.5340 | 66.8820 | 445.1798 GMAC/s | 1.031 | product |
| inference.int8i32.v1 | 1840.8580 | 1840.5670 | 16.3319 GMAC/s | 28.092 | quantize.a+product |
| inference.int8i32.v1.applechunk | 41.7580 | 41.6870 | 719.9763 GMAC/s | 0.637 | quantize.a+probe |
| training.bf16f32.v1 | 70.1810 | 69.7870 | 428.3890 GMAC/s | 1.071 | pack.b+product |
| training.int8i32.v1 | 1854.3810 | 1853.8040 | 16.2128 GMAC/s | 28.299 | quantize.a+pack.b+product |
| training.int8i32.v1.applechunk | 55.3670 | 55.2680 | 543.0089 GMAC/s | 0.845 | quantize.a+pack.b+probe |
| DERIVED int8i32.v1.flat + quantize.a | 1841.5090 | | | 28.102 | sum of two medians |
| DERIVED int8i32.v1.applechunk + quantize.a | 42.0360 | | | 0.641 | sum of two medians |
| vendor strict (COMPARISON ONLY) | 11.2181 | 11.0864 | 2680.0176 GMAC/s | | fp32.v1 over this: 5.841 |
| vendor bf16 (COMPARISON ONLY) | 19.3677 | 19.2410 | 1552.3160 GMAC/s | | fp32.v1 over this: 3.383 |

### llama8b.lm_head.t1: m=1 n=128256 k=4096 (FULL)

| arm | median ms | min ms | rate | arm over fp32.v1 | note |
|---|---:|---:|---:|---:|---|
| fp32.v1 | 33.0710 | 32.9940 | 15.8851 GMAC/s | 1.000 | plan=0 |
| bf16f32.v1.fused | 33.7940 | 32.7900 | 15.5453 GMAC/s | 1.022 | not-dispatched(cells>16384) |
| bf16f32.v1.widen | 50.1380 | 50.0520 | 10.4778 GMAC/s | 1.516 | dispatched |
| int8i32.v1.flat | 32.6260 | 32.5350 | 16.1018 GMAC/s | 0.987 | dispatched |
| int8i32.v1.mma | not run | | | | the column does not have the unit |
| int8i32.v1.applechunk | 11.8530 | 11.8160 | 44.3210 GMAC/s | 0.358 | probe,geometry=1 |
| convert.int8.quantize.a | 1.4120 | 1.3340 | 0.0029 Gelem/s | 0.043 | per-call |
| convert.int8.pack.b | 96.3680 | 96.3010 | 5.4514 Gelem/s | 2.914 | once-per-weight |
| convert.bf16.pack.b | 24.8440 | 24.8200 | 21.1454 Gelem/s | 0.751 | once-per-weight |
| convert.bf16.widen.b | 17.2930 | 17.2740 | 30.3786 Gelem/s | 0.523 | per-call-when-materialized |
| convert.int8.dequantize.b | 147.0100 | 146.9960 | 3.5735 Gelem/s | 4.445 | per-call-when-materialized |
| inference.bf16f32.v1 | 50.1440 | 50.0550 | 10.4766 GMAC/s | 1.516 | product |
| inference.int8i32.v1 | 33.7790 | 33.7100 | 15.5522 GMAC/s | 1.021 | quantize.a+product |
| inference.int8i32.v1.applechunk | 13.0130 | 12.9910 | 40.3701 GMAC/s | 0.393 | quantize.a+probe |
| training.bf16f32.v1 | 74.5810 | 74.4040 | 7.0438 GMAC/s | 2.255 | pack.b+product |
| training.int8i32.v1 | 129.7790 | 129.6230 | 4.0479 GMAC/s | 3.924 | quantize.a+pack.b+product |
| training.int8i32.v1.applechunk | 109.1880 | 108.9680 | 4.8113 GMAC/s | 3.302 | quantize.a+pack.b+probe |
| DERIVED int8i32.v1.flat + quantize.a | 34.0380 | | | 1.029 | sum of two medians |
| DERIVED int8i32.v1.applechunk + quantize.a | 13.2650 | | | 0.401 | sum of two medians |
| vendor strict (COMPARISON ONLY) | 12.8662 | 12.8448 | 40.8308 GMAC/s | | fp32.v1 over this: 2.570 |
| vendor bf16 (COMPARISON ONLY) | 6.0240 | 5.9977 | 87.2076 GMAC/s | | fp32.v1 over this: 5.490 |

### llama8b.lm_head.t8: m=8 n=128256 k=4096 (FULL)

| arm | median ms | min ms | rate | arm over fp32.v1 | note |
|---|---:|---:|---:|---:|---|
| fp32.v1 | 33.2280 | 33.1570 | 126.4805 GMAC/s | 1.000 | plan=20 |
| bf16f32.v1.fused | 252.2930 | 251.9960 | 16.6580 GMAC/s | 7.593 | not-dispatched(cells>16384) |
| bf16f32.v1.widen | 50.2500 | 50.2310 | 83.6357 GMAC/s | 1.512 | dispatched |
| int8i32.v1.flat | 252.5490 | 252.1330 | 16.6411 GMAC/s | 7.600 | dispatched |
| int8i32.v1.mma | not run | | | | the column does not have the unit |
| int8i32.v1.applechunk | 11.8550 | 11.8070 | 354.5080 GMAC/s | 0.357 | probe,geometry=1 |
| convert.int8.quantize.a | 1.4360 | 1.3690 | 0.0228 Gelem/s | 0.043 | per-call |
| convert.int8.pack.b | 96.3540 | 96.0930 | 5.4522 Gelem/s | 2.900 | once-per-weight |
| convert.bf16.pack.b | 24.8460 | 24.8360 | 21.1437 Gelem/s | 0.748 | once-per-weight |
| convert.bf16.widen.b | 17.2920 | 17.2730 | 30.3803 Gelem/s | 0.520 | per-call-when-materialized |
| convert.int8.dequantize.b | 147.0040 | 146.9750 | 3.5736 Gelem/s | 4.424 | per-call-when-materialized |
| inference.bf16f32.v1 | 50.2860 | 50.2640 | 83.5758 GMAC/s | 1.513 | product |
| inference.int8i32.v1 | 253.6220 | 252.9120 | 16.5707 GMAC/s | 7.633 | quantize.a+product |
| inference.int8i32.v1.applechunk | 13.0340 | 13.0190 | 322.4407 GMAC/s | 0.392 | quantize.a+probe |
| training.bf16f32.v1 | 74.7180 | 74.6490 | 56.2474 GMAC/s | 2.249 | pack.b+product |
| training.int8i32.v1 | 349.4260 | 349.3540 | 12.0274 GMAC/s | 10.516 | quantize.a+pack.b+product |
| training.int8i32.v1.applechunk | 109.1960 | 108.9430 | 38.4876 GMAC/s | 3.286 | quantize.a+pack.b+probe |
| DERIVED int8i32.v1.flat + quantize.a | 253.9850 | | | 7.644 | sum of two medians |
| DERIVED int8i32.v1.applechunk + quantize.a | 13.2910 | | | 0.400 | sum of two medians |
| vendor strict (COMPARISON ONLY) | 17.1264 | 17.1180 | 245.3927 GMAC/s | | fp32.v1 over this: 1.940 |
| vendor bf16 (COMPARISON ONLY) | 8.3946 | 8.3750 | 500.6397 GMAC/s | | fp32.v1 over this: 3.958 |

### llama8b.lm_head.t512: m=512 n=16032 k=4096 (CAPPED)

| arm | median ms | min ms | rate | arm over fp32.v1 | note |
|---|---:|---:|---:|---:|---|
| fp32.v1 | 67.5780 | 67.3620 | 497.5220 GMAC/s | 1.000 | plan=20 |
| bf16f32.v1.fused | 2011.1690 | 2008.5220 | 16.7174 GMAC/s | 29.761 | not-dispatched(cells>16384) |
| bf16f32.v1.widen | 69.6350 | 69.5180 | 482.8253 GMAC/s | 1.030 | dispatched |
| int8i32.v1.flat | 2011.3610 | 2010.9250 | 16.7158 GMAC/s | 29.764 | dispatched |
| int8i32.v1.mma | not run | | | | the column does not have the unit |
| int8i32.v1.applechunk | 31.3530 | 31.1510 | 1072.3548 GMAC/s | 0.464 | probe,geometry=0 |
| convert.int8.quantize.a | 3.4500 | 3.4170 | 0.6079 Gelem/s | 0.051 | per-call |
| convert.int8.pack.b | 13.9040 | 13.8570 | 4.7229 Gelem/s | 0.206 | once-per-weight |
| convert.bf16.pack.b | 3.4180 | 3.3880 | 19.2121 Gelem/s | 0.051 | once-per-weight |
| convert.bf16.widen.b | 2.3610 | 2.3380 | 27.8132 Gelem/s | 0.035 | per-call-when-materialized |
| convert.int8.dequantize.b | 15.7890 | 15.7720 | 4.1590 Gelem/s | 0.234 | per-call-when-materialized |
| inference.bf16f32.v1 | 69.6380 | 69.5660 | 482.8045 GMAC/s | 1.030 | product |
| inference.int8i32.v1 | 2014.5080 | 2014.1180 | 16.6897 GMAC/s | 29.810 | quantize.a+product |
| inference.int8i32.v1.applechunk | 34.5510 | 34.3330 | 973.0989 GMAC/s | 0.511 | quantize.a+probe |
| training.bf16f32.v1 | 72.9270 | 72.7700 | 461.0301 GMAC/s | 1.079 | pack.b+product |
| training.int8i32.v1 | 2028.3360 | 2028.0520 | 16.5759 GMAC/s | 30.015 | quantize.a+pack.b+product |
| training.int8i32.v1.applechunk | 48.1080 | 47.9440 | 698.8763 GMAC/s | 0.712 | quantize.a+pack.b+probe |
| DERIVED int8i32.v1.flat + quantize.a | 2014.8110 | | | 29.815 | sum of two medians |
| DERIVED int8i32.v1.applechunk + quantize.a | 34.8030 | | | 0.515 | sum of two medians |
| vendor strict (COMPARISON ONLY) | 122.9330 | 122.7651 | | n/a | the vendor ran the whole row (n=128256), ours ran capped: no ratio |
| vendor bf16 (COMPARISON ONLY) | 199.7351 | 199.6730 | | n/a | the vendor ran the whole row (n=128256), ours ran capped: no ratio |

## mi325x (column amd)

Run 4 (the run of record), steward request 1790651987201, timing 2026-09-29T03:20:01Z to 03:20:30Z, commit f1e6cb61c. TAKEN ALONE as a steward speed job, from a warm cache: the same binary had been built and run once, untimed, by request 1790651267504. This lane made no contact with the box from 03:19:49Z to 03:40:22Z. ONE GPU shared by every lane.

Vendor comparison: hipBLASLt/rocBLAS on AMD Instinct Mi325X VF, torch 2.9.1+rocm6.4, ROCm 6.4.43484-123eb5128, median of 10. COMPARISON ONLY, no digest, no identity claim.

### llama8b.qkv.t1: m=1 n=4096 k=4096 (FULL)

| arm | median ms | min ms | rate | arm over fp32.v1 | note |
|---|---:|---:|---:|---:|---|
| fp32.v1 | 0.0617 | 0.0602 | 271.9601 GMAC/s | 1.000 | plan=13 |
| bf16f32.v1.fused | 1.1355 | 1.1186 | 14.7757 GMAC/s | 18.404 | dispatched |
| bf16f32.v1.widen | 0.0961 | 0.0945 | 174.5263 GMAC/s | 1.558 | not-dispatched(cells<=16384) |
| int8i32.v1.flat | 1.0738 | 1.0514 | 15.6246 GMAC/s | 17.404 | not-dispatched(the-column-has-the-unit) |
| int8i32.v1.mma | 0.0512 | 0.0489 | 327.4242 GMAC/s | 0.830 | dispatched |
| int8i32.v1.applechunk | not run | | | | the column does not have the unit |
| convert.int8.quantize.a | 1.3571 | 1.3439 | 0.0030 Gelem/s | 21.995 | per-call |
| convert.int8.pack.b | 3.8474 | 3.8098 | 4.3607 Gelem/s | 62.357 | once-per-weight |
| convert.bf16.pack.b | 0.0460 | 0.0450 | 364.5636 Gelem/s | 0.746 | once-per-weight |
| convert.bf16.widen.b | 0.0463 | 0.0461 | 362.3589 Gelem/s | 0.750 | per-call-when-materialized |
| convert.int8.dequantize.b | 0.0471 | 0.0460 | 356.5827 Gelem/s | 0.763 | per-call-when-materialized |
| inference.bf16f32.v1 | 1.1249 | 1.1208 | 14.9139 GMAC/s | 18.232 | product |
| inference.int8i32.v1 | 1.3753 | 1.3562 | 12.1991 GMAC/s | 22.290 | quantize.a+product |
| inference.int8i32.v1.applechunk | not run | | | | the column does not have the unit |
| training.bf16f32.v1 | 1.1700 | 1.1561 | 14.3390 GMAC/s | 18.963 | pack.b+product |
| training.int8i32.v1 | 5.2064 | 5.1688 | 3.2224 GMAC/s | 84.382 | quantize.a+pack.b+product |
| training.int8i32.v1.applechunk | not run | | | | the column does not have the unit |
| DERIVED int8i32.v1.flat + quantize.a | 2.4309 | | | 39.399 | sum of two medians |
| DERIVED int8i32.v1.mma + quantize.a | 1.4083 | | | 22.825 | sum of two medians |
| vendor strict (COMPARISON ONLY) | 0.0672 | 0.0664 | 249.7539 GMAC/s | | fp32.v1 over this: 0.918 |
| vendor bf16 (COMPARISON ONLY) | 0.0268 | 0.0264 | 626.1439 GMAC/s | | fp32.v1 over this: 2.303 |

### llama8b.qkv.t8: m=8 n=4096 k=4096 (FULL)

| arm | median ms | min ms | rate | arm over fp32.v1 | note |
|---|---:|---:|---:|---:|---|
| fp32.v1 | 0.0775 | 0.0731 | 1732.7360 GMAC/s | 1.000 | plan=14 |
| bf16f32.v1.fused | 1.1883 | 1.1815 | 112.9487 GMAC/s | 15.333 | not-dispatched(cells>16384) |
| bf16f32.v1.widen | 0.1067 | 0.1055 | 1257.3325 GMAC/s | 1.377 | dispatched |
| int8i32.v1.flat | 1.1356 | 1.1311 | 118.1872 GMAC/s | 14.653 | not-dispatched(the-column-has-the-unit) |
| int8i32.v1.mma | 0.0516 | 0.0511 | 2601.1692 GMAC/s | 0.666 | dispatched |
| int8i32.v1.applechunk | not run | | | | the column does not have the unit |
| convert.int8.quantize.a | 1.4372 | 1.4194 | 0.0228 Gelem/s | 18.545 | per-call |
| convert.int8.pack.b | 3.8437 | 3.7800 | 4.3648 Gelem/s | 49.596 | once-per-weight |
| convert.bf16.pack.b | 0.0470 | 0.0459 | 357.3422 Gelem/s | 0.606 | once-per-weight |
| convert.bf16.widen.b | 0.0472 | 0.0451 | 355.4570 Gelem/s | 0.609 | per-call-when-materialized |
| convert.int8.dequantize.b | 0.0467 | 0.0463 | 359.0245 Gelem/s | 0.603 | per-call-when-materialized |
| inference.bf16f32.v1 | 0.1078 | 0.1063 | 1244.8315 GMAC/s | 1.391 | product |
| inference.int8i32.v1 | 1.4746 | 1.4656 | 91.0213 GMAC/s | 19.027 | quantize.a+product |
| inference.int8i32.v1.applechunk | not run | | | | the column does not have the unit |
| training.bf16f32.v1 | 0.1395 | 0.1394 | 962.0584 GMAC/s | 1.800 | pack.b+product |
| training.int8i32.v1 | 5.2733 | 5.2595 | 25.4521 GMAC/s | 68.043 | quantize.a+pack.b+product |
| training.int8i32.v1.applechunk | not run | | | | the column does not have the unit |
| DERIVED int8i32.v1.flat + quantize.a | 2.5728 | | | 33.197 | sum of two medians |
| DERIVED int8i32.v1.mma + quantize.a | 1.4888 | | | 19.210 | sum of two medians |
| vendor strict (COMPARISON ONLY) | 0.0698 | 0.0690 | 1923.1794 GMAC/s | | fp32.v1 over this: 1.110 |
| vendor bf16 (COMPARISON ONLY) | 0.0267 | 0.0259 | 5023.0253 GMAC/s | | fp32.v1 over this: 2.900 |

### llama8b.qkv.t512: m=512 n=4096 k=4096 (FULL)

| arm | median ms | min ms | rate | arm over fp32.v1 | note |
|---|---:|---:|---:|---:|---|
| fp32.v1 | 0.9254 | 0.9240 | 9282.7428 GMAC/s | 1.000 | plan=10 |
| bf16f32.v1.fused | 41.2363 | 41.1790 | 208.3102 GMAC/s | 44.561 | not-dispatched(cells>16384) |
| bf16f32.v1.widen | 0.9384 | 0.9307 | 9153.9361 GMAC/s | 1.014 | dispatched |
| int8i32.v1.flat | 41.5870 | 41.5626 | 206.5535 GMAC/s | 44.939 | not-dispatched(the-column-has-the-unit) |
| int8i32.v1.mma | 0.2669 | 0.2641 | 32183.0084 GMAC/s | 0.288 | dispatched |
| int8i32.v1.applechunk | not run | | | | the column does not have the unit |
| convert.int8.quantize.a | 3.6599 | 3.6566 | 0.5730 Gelem/s | 3.955 | per-call |
| convert.int8.pack.b | 3.7819 | 3.7787 | 4.4362 Gelem/s | 4.087 | once-per-weight |
| convert.bf16.pack.b | 0.0490 | 0.0474 | 342.3223 Gelem/s | 0.053 | once-per-weight |
| convert.bf16.widen.b | 0.0490 | 0.0486 | 342.1059 Gelem/s | 0.053 | per-call-when-materialized |
| convert.int8.dequantize.b | 0.0494 | 0.0492 | 339.5510 Gelem/s | 0.053 | per-call-when-materialized |
| inference.bf16f32.v1 | 0.9480 | 0.9427 | 9060.9595 GMAC/s | 1.024 | product |
| inference.int8i32.v1 | 4.0606 | 3.9168 | 2115.4353 GMAC/s | 4.388 | quantize.a+product |
| inference.int8i32.v1.applechunk | not run | | | | the column does not have the unit |
| training.bf16f32.v1 | 0.9616 | 0.9513 | 8933.1646 GMAC/s | 1.039 | pack.b+product |
| training.int8i32.v1 | 8.0397 | 7.8813 | 1068.4373 GMAC/s | 8.688 | quantize.a+pack.b+product |
| training.int8i32.v1.applechunk | not run | | | | the column does not have the unit |
| DERIVED int8i32.v1.flat + quantize.a | 45.2469 | | | 48.894 | sum of two medians |
| DERIVED int8i32.v1.mma + quantize.a | 3.9268 | | | 4.243 | sum of two medians |
| vendor strict (COMPARISON ONLY) | 0.2201 | 0.1953 | 39034.6884 GMAC/s | | fp32.v1 over this: 4.205 |
| vendor bf16 (COMPARISON ONLY) | 0.0521 | 0.0512 | 164856.6132 GMAC/s | | fp32.v1 over this: 17.760 |

### llama8b.mlp_up.t1: m=1 n=14336 k=4096 (FULL)

| arm | median ms | min ms | rate | arm over fp32.v1 | note |
|---|---:|---:|---:|---:|---|
| fp32.v1 | 0.1631 | 0.1619 | 359.9621 GMAC/s | 1.000 | plan=13 |
| bf16f32.v1.fused | 1.1869 | 1.1843 | 49.4746 GMAC/s | 7.277 | dispatched |
| bf16f32.v1.widen | 0.2841 | 0.2828 | 206.7040 GMAC/s | 1.742 | not-dispatched(cells<=16384) |
| int8i32.v1.flat | 1.1041 | 1.0982 | 53.1844 GMAC/s | 6.769 | not-dispatched(the-column-has-the-unit) |
| int8i32.v1.mma | 0.0567 | 0.0556 | 1034.7364 GMAC/s | 0.348 | dispatched |
| int8i32.v1.applechunk | not run | | | | the column does not have the unit |
| convert.int8.quantize.a | 1.3569 | 1.3401 | 0.0030 Gelem/s | 8.319 | per-call |
| convert.int8.pack.b | 4.1559 | 4.1214 | 14.1293 Gelem/s | 25.481 | once-per-weight |
| convert.bf16.pack.b | 0.1268 | 0.1260 | 463.2067 Gelem/s | 0.777 | once-per-weight |
| convert.bf16.widen.b | 0.1364 | 0.1339 | 430.5951 Gelem/s | 0.836 | per-call-when-materialized |
| convert.int8.dequantize.b | 0.1492 | 0.1477 | 393.6756 Gelem/s | 0.915 | per-call-when-materialized |
| inference.bf16f32.v1 | 1.1905 | 1.1872 | 49.3246 GMAC/s | 7.299 | product |
| inference.int8i32.v1 | 1.4112 | 1.4013 | 41.6097 GMAC/s | 8.652 | quantize.a+product |
| inference.int8i32.v1.applechunk | not run | | | | the column does not have the unit |
| training.bf16f32.v1 | 1.2557 | 1.2544 | 46.7646 GMAC/s | 7.699 | pack.b+product |
| training.int8i32.v1 | 5.4792 | 5.4255 | 10.7169 GMAC/s | 33.594 | quantize.a+pack.b+product |
| training.int8i32.v1.applechunk | not run | | | | the column does not have the unit |
| DERIVED int8i32.v1.flat + quantize.a | 2.4610 | | | 15.089 | sum of two medians |
| DERIVED int8i32.v1.mma + quantize.a | 1.4136 | | | 8.667 | sum of two medians |
| vendor strict (COMPARISON ONLY) | 0.0845 | 0.0797 | 695.0418 GMAC/s | | fp32.v1 over this: 1.931 |
| vendor bf16 (COMPARISON ONLY) | 0.0534 | 0.0521 | 1099.2289 GMAC/s | | fp32.v1 over this: 3.053 |

### llama8b.mlp_up.t8: m=8 n=14336 k=4096 (FULL)

| arm | median ms | min ms | rate | arm over fp32.v1 | note |
|---|---:|---:|---:|---:|---|
| fp32.v1 | 0.1677 | 0.1658 | 2801.3719 GMAC/s | 1.000 | plan=14 |
| bf16f32.v1.fused | 2.8442 | 2.8216 | 165.1659 GMAC/s | 16.960 | not-dispatched(cells>16384) |
| bf16f32.v1.widen | 0.2917 | 0.2898 | 1610.1582 GMAC/s | 1.739 | dispatched |
| int8i32.v1.flat | 2.8417 | 2.8324 | 165.3084 GMAC/s | 16.945 | not-dispatched(the-column-has-the-unit) |
| int8i32.v1.mma | 0.0583 | 0.0574 | 8063.2003 GMAC/s | 0.348 | dispatched |
| int8i32.v1.applechunk | not run | | | | the column does not have the unit |
| convert.int8.quantize.a | 1.4404 | 1.4312 | 0.0227 Gelem/s | 8.589 | per-call |
| convert.int8.pack.b | 4.1601 | 4.1237 | 14.1149 Gelem/s | 24.807 | once-per-weight |
| convert.bf16.pack.b | 0.1261 | 0.1254 | 465.8120 Gelem/s | 0.752 | once-per-weight |
| convert.bf16.widen.b | 0.1312 | 0.1308 | 447.4606 Gelem/s | 0.782 | per-call-when-materialized |
| convert.int8.dequantize.b | 0.1487 | 0.1472 | 394.8111 Gelem/s | 0.887 | per-call-when-materialized |
| inference.bf16f32.v1 | 0.2920 | 0.2909 | 1609.0497 GMAC/s | 1.741 | product |
| inference.int8i32.v1 | 1.5008 | 1.4954 | 313.0023 GMAC/s | 8.949 | quantize.a+product |
| inference.int8i32.v1.applechunk | not run | | | | the column does not have the unit |
| training.bf16f32.v1 | 0.4017 | 0.3992 | 1169.4379 GMAC/s | 2.395 | pack.b+product |
| training.int8i32.v1 | 5.5771 | 5.5515 | 84.2312 GMAC/s | 33.256 | quantize.a+pack.b+product |
| training.int8i32.v1.applechunk | not run | | | | the column does not have the unit |
| DERIVED int8i32.v1.flat + quantize.a | 4.2821 | | | 25.534 | sum of two medians |
| DERIVED int8i32.v1.mma + quantize.a | 1.4987 | | | 8.937 | sum of two medians |
| vendor strict (COMPARISON ONLY) | 0.0819 | 0.0807 | 5733.0003 GMAC/s | | fp32.v1 over this: 2.047 |
| vendor bf16 (COMPARISON ONLY) | 0.0571 | 0.0528 | 8221.2477 GMAC/s | | fp32.v1 over this: 2.935 |

### llama8b.mlp_up.t512: m=512 n=14336 k=4096 (FULL)

| arm | median ms | min ms | rate | arm over fp32.v1 | note |
|---|---:|---:|---:|---:|---|
| fp32.v1 | 2.3895 | 2.3752 | 12582.0659 GMAC/s | 1.000 | plan=10 |
| bf16f32.v1.fused | 174.3496 | 173.9604 | 172.4396 GMAC/s | 72.965 | not-dispatched(cells>16384) |
| bf16f32.v1.widen | 2.4240 | 2.4223 | 12402.7844 GMAC/s | 1.014 | dispatched |
| int8i32.v1.flat | 151.1207 | 149.6964 | 198.9454 GMAC/s | 63.244 | not-dispatched(the-column-has-the-unit) |
| int8i32.v1.mma | 0.8773 | 0.8759 | 34270.5566 GMAC/s | 0.367 | dispatched |
| int8i32.v1.applechunk | not run | | | | the column does not have the unit |
| convert.int8.quantize.a | 3.7927 | 3.7481 | 0.5529 Gelem/s | 1.587 | per-call |
| convert.int8.pack.b | 4.1662 | 4.1429 | 14.0943 Gelem/s | 1.744 | once-per-weight |
| convert.bf16.pack.b | 0.1270 | 0.1267 | 462.1859 Gelem/s | 0.053 | once-per-weight |
| convert.bf16.widen.b | 0.1357 | 0.1338 | 432.7849 Gelem/s | 0.057 | per-call-when-materialized |
| convert.int8.dequantize.b | 0.1478 | 0.1475 | 397.3787 Gelem/s | 0.062 | per-call-when-materialized |
| inference.bf16f32.v1 | 2.5044 | 2.5020 | 12004.6650 GMAC/s | 1.048 | product |
| inference.int8i32.v1 | 4.9238 | 4.7118 | 6106.0011 GMAC/s | 2.061 | quantize.a+product |
| inference.int8i32.v1.applechunk | not run | | | | the column does not have the unit |
| training.bf16f32.v1 | 2.6662 | 2.6192 | 11276.3386 GMAC/s | 1.116 | pack.b+product |
| training.int8i32.v1 | 8.9997 | 8.9159 | 3340.6467 GMAC/s | 3.766 | quantize.a+pack.b+product |
| training.int8i32.v1.applechunk | not run | | | | the column does not have the unit |
| DERIVED int8i32.v1.flat + quantize.a | 154.9134 | | | 64.831 | sum of two medians |
| DERIVED int8i32.v1.mma + quantize.a | 4.6700 | | | 1.954 | sum of two medians |
| vendor strict (COMPARISON ONLY) | 0.8458 | 0.8032 | 35547.1136 GMAC/s | | fp32.v1 over this: 2.825 |
| vendor bf16 (COMPARISON ONLY) | 0.1084 | 0.0989 | 277364.3529 GMAC/s | | fp32.v1 over this: 22.044 |

### llama8b.mlp_down.t1: m=1 n=4096 k=14336 (FULL)

| arm | median ms | min ms | rate | arm over fp32.v1 | note |
|---|---:|---:|---:|---:|---|
| fp32.v1 | 0.1560 | 0.1549 | 376.3636 GMAC/s | 1.000 | plan=13 |
| bf16f32.v1.fused | 3.9476 | 3.8942 | 14.8750 GMAC/s | 25.305 | dispatched |
| bf16f32.v1.widen | 0.2810 | 0.2796 | 208.9399 GMAC/s | 1.801 | not-dispatched(cells<=16384) |
| int8i32.v1.flat | 3.8100 | 3.7541 | 15.4122 GMAC/s | 24.423 | not-dispatched(the-column-has-the-unit) |
| int8i32.v1.mma | 0.1374 | 0.1363 | 427.4948 GMAC/s | 0.881 | dispatched |
| int8i32.v1.applechunk | not run | | | | the column does not have the unit |
| convert.int8.quantize.a | 4.6908 | 4.6797 | 0.0031 Gelem/s | 30.069 | per-call |
| convert.int8.pack.b | 14.1738 | 14.0770 | 4.1429 Gelem/s | 90.858 | once-per-weight |
| convert.bf16.pack.b | 0.1275 | 0.1269 | 460.4066 Gelem/s | 0.817 | once-per-weight |
| convert.bf16.widen.b | 0.1325 | 0.1321 | 443.3089 Gelem/s | 0.849 | per-call-when-materialized |
| convert.int8.dequantize.b | 0.1493 | 0.1476 | 393.2274 Gelem/s | 0.957 | per-call-when-materialized |
| inference.bf16f32.v1 | 3.9026 | 3.8987 | 15.0466 GMAC/s | 25.017 | product |
| inference.int8i32.v1 | 4.8812 | 4.8317 | 12.0299 GMAC/s | 31.290 | quantize.a+product |
| inference.int8i32.v1.applechunk | not run | | | | the column does not have the unit |
| training.bf16f32.v1 | 3.9136 | 3.8569 | 15.0042 GMAC/s | 25.087 | pack.b+product |
| training.int8i32.v1 | 18.9537 | 18.7923 | 3.0981 GMAC/s | 121.498 | quantize.a+pack.b+product |
| training.int8i32.v1.applechunk | not run | | | | the column does not have the unit |
| DERIVED int8i32.v1.flat + quantize.a | 8.5008 | | | 54.492 | sum of two medians |
| DERIVED int8i32.v1.mma + quantize.a | 4.8282 | | | 30.950 | sum of two medians |
| vendor strict (COMPARISON ONLY) | 0.0748 | 0.0729 | 784.9304 GMAC/s | | fp32.v1 over this: 2.085 |
| vendor bf16 (COMPARISON ONLY) | 0.0426 | 0.0423 | 1379.2195 GMAC/s | | fp32.v1 over this: 3.664 |

### llama8b.mlp_down.t8: m=8 n=4096 k=14336 (FULL)

| arm | median ms | min ms | rate | arm over fp32.v1 | note |
|---|---:|---:|---:|---:|---|
| fp32.v1 | 0.1947 | 0.1906 | 2412.5126 GMAC/s | 1.000 | plan=14 |
| bf16f32.v1.fused | 3.6965 | 3.6292 | 127.0843 GMAC/s | 18.986 | not-dispatched(cells>16384) |
| bf16f32.v1.widen | 0.3105 | 0.3089 | 1513.0772 GMAC/s | 1.595 | dispatched |
| int8i32.v1.flat | 3.7483 | 3.7445 | 125.3254 GMAC/s | 19.252 | not-dispatched(the-column-has-the-unit) |
| int8i32.v1.mma | 0.1415 | 0.1401 | 3320.8354 GMAC/s | 0.727 | dispatched |
| int8i32.v1.applechunk | not run | | | | the column does not have the unit |
| convert.int8.quantize.a | 5.1709 | 5.1059 | 0.0222 Gelem/s | 26.558 | per-call |
| convert.int8.pack.b | 14.1137 | 13.8430 | 4.1605 Gelem/s | 72.489 | once-per-weight |
| convert.bf16.pack.b | 0.1255 | 0.1246 | 467.7451 Gelem/s | 0.645 | once-per-weight |
| convert.bf16.widen.b | 0.1341 | 0.1321 | 437.8874 Gelem/s | 0.689 | per-call-when-materialized |
| convert.int8.dequantize.b | 0.1485 | 0.1473 | 395.2895 Gelem/s | 0.763 | per-call-when-materialized |
| inference.bf16f32.v1 | 0.3250 | 0.3227 | 1445.2927 GMAC/s | 1.669 | product |
| inference.int8i32.v1 | 5.3521 | 5.2635 | 87.7721 GMAC/s | 27.489 | quantize.a+product |
| inference.int8i32.v1.applechunk | not run | | | | the column does not have the unit |
| training.bf16f32.v1 | 0.4312 | 0.4278 | 1089.3058 GMAC/s | 2.215 | pack.b+product |
| training.int8i32.v1 | 19.3871 | 19.3240 | 24.2307 GMAC/s | 99.574 | quantize.a+pack.b+product |
| training.int8i32.v1.applechunk | not run | | | | the column does not have the unit |
| DERIVED int8i32.v1.flat + quantize.a | 8.9192 | | | 45.810 | sum of two medians |
| DERIVED int8i32.v1.mma + quantize.a | 5.3124 | | | 27.285 | sum of two medians |
| vendor strict (COMPARISON ONLY) | 0.0760 | 0.0747 | 6182.7479 GMAC/s | | fp32.v1 over this: 2.563 |
| vendor bf16 (COMPARISON ONLY) | 0.0438 | 0.0432 | 10733.7395 GMAC/s | | fp32.v1 over this: 4.449 |

### llama8b.mlp_down.t512: m=512 n=4096 k=14336 (FULL)

| arm | median ms | min ms | rate | arm over fp32.v1 | note |
|---|---:|---:|---:|---:|---|
| fp32.v1 | 2.5107 | 2.4546 | 11974.5949 GMAC/s | 1.000 | plan=10 |
| bf16f32.v1.fused | 192.6926 | 191.9201 | 156.0245 GMAC/s | 76.749 | not-dispatched(cells>16384) |
| bf16f32.v1.widen | 2.5016 | 2.4830 | 12018.2456 GMAC/s | 0.996 | dispatched |
| int8i32.v1.flat | 139.1355 | 138.8258 | 216.0826 GMAC/s | 55.417 | not-dispatched(the-column-has-the-unit) |
| int8i32.v1.mma | 0.8422 | 0.8415 | 35698.4083 GMAC/s | 0.335 | dispatched |
| int8i32.v1.applechunk | not run | | | | the column does not have the unit |
| convert.int8.quantize.a | 13.3456 | 13.3150 | 0.5500 Gelem/s | 5.315 | per-call |
| convert.int8.pack.b | 13.6201 | 13.6160 | 4.3113 Gelem/s | 5.425 | once-per-weight |
| convert.bf16.pack.b | 0.1285 | 0.1272 | 456.8603 Gelem/s | 0.051 | once-per-weight |
| convert.bf16.widen.b | 0.1344 | 0.1331 | 437.0693 Gelem/s | 0.054 | per-call-when-materialized |
| convert.int8.dequantize.b | 0.1501 | 0.1491 | 391.3379 Gelem/s | 0.060 | per-call-when-materialized |
| inference.bf16f32.v1 | 2.5862 | 2.5802 | 11625.2871 GMAC/s | 1.030 | product |
| inference.int8i32.v1 | 14.7519 | 14.3545 | 2038.0283 GMAC/s | 5.876 | quantize.a+product |
| inference.int8i32.v1.applechunk | not run | | | | the column does not have the unit |
| training.bf16f32.v1 | 2.7672 | 2.7226 | 10864.5215 GMAC/s | 1.102 | pack.b+product |
| training.int8i32.v1 | 28.8290 | 28.5589 | 1042.8644 GMAC/s | 11.482 | quantize.a+pack.b+product |
| training.int8i32.v1.applechunk | not run | | | | the column does not have the unit |
| DERIVED int8i32.v1.flat + quantize.a | 152.4811 | | | 60.733 | sum of two medians |
| DERIVED int8i32.v1.mma + quantize.a | 14.1878 | | | 5.651 | sum of two medians |
| vendor strict (COMPARISON ONLY) | 1.1845 | 1.1572 | 25382.0065 GMAC/s | | fp32.v1 over this: 2.120 |
| vendor bf16 (COMPARISON ONLY) | 0.1275 | 0.1232 | 235849.2942 GMAC/s | | fp32.v1 over this: 19.696 |

### llama8b.lm_head.t1: m=1 n=128256 k=4096 (FULL)

| arm | median ms | min ms | rate | arm over fp32.v1 | note |
|---|---:|---:|---:|---:|---|
| fp32.v1 | 1.3122 | 1.3039 | 400.3583 GMAC/s | 1.000 | plan=13 |
| bf16f32.v1.fused | 5.4303 | 5.4107 | 96.7423 GMAC/s | 4.138 | not-dispatched(cells>16384) |
| bf16f32.v1.widen | 2.4613 | 2.4533 | 213.4426 GMAC/s | 1.876 | dispatched |
| int8i32.v1.flat | 6.4084 | 6.0644 | 81.9758 GMAC/s | 4.884 | not-dispatched(the-column-has-the-unit) |
| int8i32.v1.mma | 0.2862 | 0.2835 | 1835.3075 GMAC/s | 0.218 | dispatched |
| int8i32.v1.applechunk | not run | | | | the column does not have the unit |
| convert.int8.quantize.a | 1.3676 | 1.3425 | 0.0030 Gelem/s | 1.042 | per-call |
| convert.int8.pack.b | 15.6604 | 15.5096 | 33.5455 Gelem/s | 11.934 | once-per-weight |
| convert.bf16.pack.b | 1.0160 | 1.0133 | 517.0493 Gelem/s | 0.774 | once-per-weight |
| convert.bf16.widen.b | 1.1499 | 1.1393 | 456.8359 Gelem/s | 0.876 | per-call-when-materialized |
| convert.int8.dequantize.b | 1.2571 | 1.2500 | 417.8999 Gelem/s | 0.958 | per-call-when-materialized |
| inference.bf16f32.v1 | 2.5466 | 2.5203 | 206.2867 GMAC/s | 1.941 | product |
| inference.int8i32.v1 | 1.6439 | 1.6342 | 319.5585 GMAC/s | 1.253 | quantize.a+product |
| inference.int8i32.v1.applechunk | not run | | | | the column does not have the unit |
| training.bf16f32.v1 | 3.5467 | 3.5365 | 148.1189 GMAC/s | 2.703 | pack.b+product |
| training.int8i32.v1 | 17.1070 | 17.0513 | 30.7088 GMAC/s | 13.037 | quantize.a+pack.b+product |
| training.int8i32.v1.applechunk | not run | | | | the column does not have the unit |
| DERIVED int8i32.v1.flat + quantize.a | 7.7760 | | | 5.926 | sum of two medians |
| DERIVED int8i32.v1.mma + quantize.a | 1.6538 | | | 1.260 | sum of two medians |
| vendor strict (COMPARISON ONLY) | 0.6223 | 0.5672 | 844.2017 GMAC/s | | fp32.v1 over this: 2.109 |
| vendor bf16 (COMPARISON ONLY) | 0.2787 | 0.2750 | 1885.2310 GMAC/s | | fp32.v1 over this: 4.709 |

### llama8b.lm_head.t8: m=8 n=128256 k=4096 (FULL)

| arm | median ms | min ms | rate | arm over fp32.v1 | note |
|---|---:|---:|---:|---:|---|
| fp32.v1 | 1.2280 | 1.2270 | 3422.3686 GMAC/s | 1.000 | plan=14 |
| bf16f32.v1.fused | 40.8262 | 40.1259 | 102.9411 GMAC/s | 33.246 | not-dispatched(cells>16384) |
| bf16f32.v1.widen | 2.4054 | 2.4031 | 1747.2031 GMAC/s | 1.959 | dispatched |
| int8i32.v1.flat | 46.4979 | 45.3977 | 90.3846 GMAC/s | 37.865 | not-dispatched(the-column-has-the-unit) |
| int8i32.v1.mma | 0.3323 | 0.3313 | 12648.0837 GMAC/s | 0.271 | dispatched |
| int8i32.v1.applechunk | not run | | | | the column does not have the unit |
| convert.int8.quantize.a | 1.4594 | 1.4338 | 0.0225 Gelem/s | 1.188 | per-call |
| convert.int8.pack.b | 16.0647 | 15.8689 | 32.7013 Gelem/s | 13.082 | once-per-weight |
| convert.bf16.pack.b | 1.0180 | 1.0156 | 516.0543 Gelem/s | 0.829 | once-per-weight |
| convert.bf16.widen.b | 1.1513 | 1.1495 | 456.3002 Gelem/s | 0.938 | per-call-when-materialized |
| convert.int8.dequantize.b | 1.2664 | 1.2585 | 414.8113 Gelem/s | 1.031 | per-call-when-materialized |
| inference.bf16f32.v1 | 2.4984 | 2.4754 | 1682.1853 GMAC/s | 2.035 | product |
| inference.int8i32.v1 | 1.9165 | 1.8948 | 2192.9285 GMAC/s | 1.561 | quantize.a+product |
| inference.int8i32.v1.applechunk | not run | | | | the column does not have the unit |
| training.bf16f32.v1 | 3.4718 | 3.4434 | 1210.5153 GMAC/s | 2.827 | pack.b+product |
| training.int8i32.v1 | 17.5921 | 17.3529 | 238.8972 GMAC/s | 14.326 | quantize.a+pack.b+product |
| training.int8i32.v1.applechunk | not run | | | | the column does not have the unit |
| DERIVED int8i32.v1.flat + quantize.a | 47.9573 | | | 39.053 | sum of two medians |
| DERIVED int8i32.v1.mma + quantize.a | 1.7917 | | | 1.459 | sum of two medians |
| vendor strict (COMPARISON ONLY) | 0.6488 | 0.5979 | 6477.2553 GMAC/s | | fp32.v1 over this: 1.893 |
| vendor bf16 (COMPARISON ONLY) | 0.2826 | 0.2774 | 14869.4454 GMAC/s | | fp32.v1 over this: 4.345 |

### llama8b.lm_head.t512: m=512 n=16032 k=4096 (CAPPED)

| arm | median ms | min ms | rate | arm over fp32.v1 | note |
|---|---:|---:|---:|---:|---|
| fp32.v1 | 2.7776 | 2.7441 | 12104.5207 GMAC/s | 1.000 | plan=10 |
| bf16f32.v1.fused | 271.0917 | 261.1044 | 124.0227 GMAC/s | 97.599 | not-dispatched(cells>16384) |
| bf16f32.v1.widen | 2.8174 | 2.8086 | 11933.6491 GMAC/s | 1.014 | dispatched |
| int8i32.v1.flat | 250.1423 | 248.9967 | 134.4096 GMAC/s | 90.057 | not-dispatched(the-column-has-the-unit) |
| int8i32.v1.mma | 1.0130 | 1.0072 | 33189.8078 GMAC/s | 0.365 | dispatched |
| int8i32.v1.applechunk | not run | | | | the column does not have the unit |
| convert.int8.quantize.a | 3.9300 | 3.7409 | 0.5336 Gelem/s | 1.415 | per-call |
| convert.int8.pack.b | 4.2702 | 4.2455 | 15.3779 Gelem/s | 1.537 | once-per-weight |
| convert.bf16.pack.b | 0.1431 | 0.1403 | 458.9535 Gelem/s | 0.052 | once-per-weight |
| convert.bf16.widen.b | 0.1499 | 0.1492 | 438.0141 Gelem/s | 0.054 | per-call-when-materialized |
| convert.int8.dequantize.b | 0.1662 | 0.1646 | 395.2062 Gelem/s | 0.060 | per-call-when-materialized |
| inference.bf16f32.v1 | 2.9081 | 2.8681 | 11561.2155 GMAC/s | 1.047 | product |
| inference.int8i32.v1 | 5.0582 | 4.9293 | 6646.9431 GMAC/s | 1.821 | quantize.a+product |
| inference.int8i32.v1.applechunk | not run | | | | the column does not have the unit |
| training.bf16f32.v1 | 3.0784 | 3.0632 | 10921.8612 GMAC/s | 1.108 | pack.b+product |
| training.int8i32.v1 | 9.1749 | 9.0936 | 3664.5071 GMAC/s | 3.303 | quantize.a+pack.b+product |
| training.int8i32.v1.applechunk | not run | | | | the column does not have the unit |
| DERIVED int8i32.v1.flat + quantize.a | 254.0723 | | | 91.472 | sum of two medians |
| DERIVED int8i32.v1.mma + quantize.a | 4.9430 | | | 1.780 | sum of two medians |
| vendor strict (COMPARISON ONLY) | 5.1521 | 4.8552 | | n/a | the vendor ran the whole row (n=128256), ours ran capped: no ratio |
| vendor bf16 (COMPARISON ONLY) | 0.9871 | 0.9436 | | n/a | the vendor ran the whole row (n=128256), ours ran capped: no ratio |

## Where a low-bit plan costs more time than fp32.v1

| box | shape | arm | arm ms | fp32.v1 ms | arm over fp32.v1 | note |
|---|---|---|---:|---:|---:|---|
| h100 | llama8b.qkv.t1 | bf16f32.v1.fused | 0.7918 | 0.0737 | 10.744 | dispatched |
| h100 | llama8b.qkv.t1 | bf16f32.v1.widen | 0.1245 | 0.0737 | 1.689 | not-dispatched(cells<=16384) |
| h100 | llama8b.qkv.t1 | int8i32.v1.flat | 0.6663 | 0.0737 | 9.041 | not-dispatched(the-column-has-the-unit) |
| h100 | llama8b.qkv.t1 | int8i32.v1.mma | 0.1591 | 0.0737 | 2.159 | dispatched |
| h100 | llama8b.qkv.t1 | int8i32.v1.flat + quantize.a (DERIVED) | 1.1555 | 0.0737 | 15.678 | not-dispatched(the-column-has-the-unit) |
| h100 | llama8b.qkv.t1 | int8i32.v1.mma + quantize.a (DERIVED) | 0.6483 | 0.0737 | 8.796 | dispatched |
| h100 | llama8b.qkv.t8 | bf16f32.v1.fused | 0.7850 | 0.1074 | 7.309 | not-dispatched(cells>16384) |
| h100 | llama8b.qkv.t8 | bf16f32.v1.widen | 0.1506 | 0.1074 | 1.402 | dispatched |
| h100 | llama8b.qkv.t8 | int8i32.v1.flat | 0.6652 | 0.1074 | 6.194 | not-dispatched(the-column-has-the-unit) |
| h100 | llama8b.qkv.t8 | int8i32.v1.mma | 0.1486 | 0.1074 | 1.384 | dispatched |
| h100 | llama8b.qkv.t8 | int8i32.v1.flat + quantize.a (DERIVED) | 1.2670 | 0.1074 | 11.797 | not-dispatched(the-column-has-the-unit) |
| h100 | llama8b.qkv.t8 | int8i32.v1.mma + quantize.a (DERIVED) | 0.7504 | 0.1074 | 6.987 | dispatched |
| h100 | llama8b.qkv.t512 | bf16f32.v1.fused | 39.4083 | 1.0377 | 37.977 | not-dispatched(cells>16384) |
| h100 | llama8b.qkv.t512 | bf16f32.v1.widen | 1.1048 | 1.0377 | 1.065 | dispatched |
| h100 | llama8b.qkv.t512 | int8i32.v1.flat | 39.1923 | 1.0377 | 37.768 | not-dispatched(the-column-has-the-unit) |
| h100 | llama8b.qkv.t512 | int8i32.v1.mma | 1.4407 | 1.0377 | 1.388 | dispatched |
| h100 | llama8b.qkv.t512 | int8i32.v1.flat + quantize.a (DERIVED) | 41.8159 | 1.0377 | 40.297 | not-dispatched(the-column-has-the-unit) |
| h100 | llama8b.qkv.t512 | int8i32.v1.mma + quantize.a (DERIVED) | 4.0643 | 1.0377 | 3.917 | dispatched |
| h100 | llama8b.mlp_up.t1 | bf16f32.v1.fused | 0.7926 | 0.1884 | 4.207 | dispatched |
| h100 | llama8b.mlp_up.t1 | bf16f32.v1.widen | 0.3677 | 0.1884 | 1.952 | not-dispatched(cells<=16384) |
| h100 | llama8b.mlp_up.t1 | int8i32.v1.flat | 0.6687 | 0.1884 | 3.549 | not-dispatched(the-column-has-the-unit) |
| h100 | llama8b.mlp_up.t1 | int8i32.v1.mma | 0.1914 | 0.1884 | 1.016 | dispatched |
| h100 | llama8b.mlp_up.t1 | int8i32.v1.flat + quantize.a (DERIVED) | 1.1704 | 0.1884 | 6.212 | not-dispatched(the-column-has-the-unit) |
| h100 | llama8b.mlp_up.t1 | int8i32.v1.mma + quantize.a (DERIVED) | 0.6931 | 0.1884 | 3.679 | dispatched |
| h100 | llama8b.mlp_up.t8 | bf16f32.v1.fused | 2.5309 | 0.2757 | 9.180 | not-dispatched(cells>16384) |
| h100 | llama8b.mlp_up.t8 | bf16f32.v1.widen | 0.4534 | 0.2757 | 1.645 | dispatched |
| h100 | llama8b.mlp_up.t8 | int8i32.v1.flat | 2.5062 | 0.2757 | 9.090 | not-dispatched(the-column-has-the-unit) |
| h100 | llama8b.mlp_up.t8 | int8i32.v1.flat + quantize.a (DERIVED) | 3.1196 | 0.2757 | 11.315 | not-dispatched(the-column-has-the-unit) |
| h100 | llama8b.mlp_up.t8 | int8i32.v1.mma + quantize.a (DERIVED) | 0.8228 | 0.2757 | 2.984 | dispatched |
| h100 | llama8b.mlp_up.t512 | bf16f32.v1.fused | 139.2825 | 3.3462 | 41.624 | not-dispatched(cells>16384) |
| h100 | llama8b.mlp_up.t512 | bf16f32.v1.widen | 3.5225 | 3.3462 | 1.053 | dispatched |
| h100 | llama8b.mlp_up.t512 | int8i32.v1.flat | 135.6707 | 3.3462 | 40.545 | not-dispatched(the-column-has-the-unit) |
| h100 | llama8b.mlp_up.t512 | int8i32.v1.mma | 4.6280 | 3.3462 | 1.383 | dispatched |
| h100 | llama8b.mlp_up.t512 | int8i32.v1.flat + quantize.a (DERIVED) | 138.3169 | 3.3462 | 41.336 | not-dispatched(the-column-has-the-unit) |
| h100 | llama8b.mlp_up.t512 | int8i32.v1.mma + quantize.a (DERIVED) | 7.2742 | 3.3462 | 2.174 | dispatched |
| h100 | llama8b.mlp_down.t1 | bf16f32.v1.fused | 2.7570 | 0.1773 | 15.550 | dispatched |
| h100 | llama8b.mlp_down.t1 | bf16f32.v1.widen | 0.3531 | 0.1773 | 1.992 | not-dispatched(cells<=16384) |
| h100 | llama8b.mlp_down.t1 | int8i32.v1.flat | 2.3473 | 0.1773 | 13.239 | not-dispatched(the-column-has-the-unit) |
| h100 | llama8b.mlp_down.t1 | int8i32.v1.mma | 0.6532 | 0.1773 | 3.684 | dispatched |
| h100 | llama8b.mlp_down.t1 | int8i32.v1.flat + quantize.a (DERIVED) | 4.0827 | 0.1773 | 23.027 | not-dispatched(the-column-has-the-unit) |
| h100 | llama8b.mlp_down.t1 | int8i32.v1.mma + quantize.a (DERIVED) | 2.3886 | 0.1773 | 13.472 | dispatched |
| h100 | llama8b.mlp_down.t8 | bf16f32.v1.fused | 2.7008 | 0.2918 | 9.256 | not-dispatched(cells>16384) |
| h100 | llama8b.mlp_down.t8 | bf16f32.v1.widen | 0.4674 | 0.2918 | 1.602 | dispatched |
| h100 | llama8b.mlp_down.t8 | int8i32.v1.flat | 2.3387 | 0.2918 | 8.015 | not-dispatched(the-column-has-the-unit) |
| h100 | llama8b.mlp_down.t8 | int8i32.v1.mma | 0.6976 | 0.2918 | 2.391 | dispatched |
| h100 | llama8b.mlp_down.t8 | int8i32.v1.flat + quantize.a (DERIVED) | 4.7659 | 0.2918 | 16.333 | not-dispatched(the-column-has-the-unit) |
| h100 | llama8b.mlp_down.t8 | int8i32.v1.mma + quantize.a (DERIVED) | 3.1248 | 0.2918 | 10.709 | dispatched |
| h100 | llama8b.mlp_down.t512 | bf16f32.v1.fused | 136.6683 | 3.3696 | 40.559 | not-dispatched(cells>16384) |
| h100 | llama8b.mlp_down.t512 | bf16f32.v1.widen | 3.5471 | 3.3696 | 1.053 | dispatched |
| h100 | llama8b.mlp_down.t512 | int8i32.v1.flat | 137.9638 | 3.3696 | 40.944 | not-dispatched(the-column-has-the-unit) |
| h100 | llama8b.mlp_down.t512 | int8i32.v1.mma | 5.5492 | 3.3696 | 1.647 | dispatched |
| h100 | llama8b.mlp_down.t512 | int8i32.v1.flat + quantize.a (DERIVED) | 147.9283 | 3.3696 | 43.901 | not-dispatched(the-column-has-the-unit) |
| h100 | llama8b.mlp_down.t512 | int8i32.v1.mma + quantize.a (DERIVED) | 15.5137 | 3.3696 | 4.604 | dispatched |
| h100 | llama8b.lm_head.t1 | bf16f32.v1.fused | 2.5968 | 1.2049 | 2.155 | not-dispatched(cells>16384) |
| h100 | llama8b.lm_head.t1 | bf16f32.v1.widen | 2.8119 | 1.2049 | 2.334 | dispatched |
| h100 | llama8b.lm_head.t1 | int8i32.v1.flat | 2.5115 | 1.2049 | 2.084 | not-dispatched(the-column-has-the-unit) |
| h100 | llama8b.lm_head.t1 | int8i32.v1.flat + quantize.a (DERIVED) | 3.0201 | 1.2049 | 2.507 | not-dispatched(the-column-has-the-unit) |
| h100 | llama8b.lm_head.t1 | int8i32.v1.mma + quantize.a (DERIVED) | 1.4719 | 1.2049 | 1.222 | dispatched |
| h100 | llama8b.lm_head.t8 | bf16f32.v1.fused | 19.9111 | 2.1659 | 9.193 | not-dispatched(cells>16384) |
| h100 | llama8b.lm_head.t8 | bf16f32.v1.widen | 3.7598 | 2.1659 | 1.736 | dispatched |
| h100 | llama8b.lm_head.t8 | int8i32.v1.flat | 19.2579 | 2.1659 | 8.891 | not-dispatched(the-column-has-the-unit) |
| h100 | llama8b.lm_head.t8 | int8i32.v1.flat + quantize.a (DERIVED) | 19.8698 | 2.1659 | 9.174 | not-dispatched(the-column-has-the-unit) |
| h100 | llama8b.lm_head.t512 | bf16f32.v1.fused | 156.4376 | 3.7367 | 41.865 | not-dispatched(cells>16384) |
| h100 | llama8b.lm_head.t512 | bf16f32.v1.widen | 3.9342 | 3.7367 | 1.053 | dispatched |
| h100 | llama8b.lm_head.t512 | int8i32.v1.flat | 151.9419 | 3.7367 | 40.662 | not-dispatched(the-column-has-the-unit) |
| h100 | llama8b.lm_head.t512 | int8i32.v1.mma | 5.1285 | 3.7367 | 1.372 | dispatched |
| h100 | llama8b.lm_head.t512 | int8i32.v1.flat + quantize.a (DERIVED) | 154.4671 | 3.7367 | 41.338 | not-dispatched(the-column-has-the-unit) |
| h100 | llama8b.lm_head.t512 | int8i32.v1.mma + quantize.a (DERIVED) | 7.6537 | 3.7367 | 2.048 | dispatched |
| m3ultra | llama8b.qkv.t1 | bf16f32.v1.widen | 1.5800 | 1.0240 | 1.543 | not-dispatched(cells<=16384) |
| m3ultra | llama8b.qkv.t1 | int8i32.v1.flat + quantize.a (DERIVED) | 1.9070 | 1.0240 | 1.862 | dispatched |
| m3ultra | llama8b.qkv.t1 | int8i32.v1.applechunk + quantize.a (DERIVED) | 1.9220 | 1.0240 | 1.877 | probe,geometry=1 |
| m3ultra | llama8b.qkv.t8 | bf16f32.v1.fused | 1.7130 | 0.5420 | 3.161 | not-dispatched(cells>16384) |
| m3ultra | llama8b.qkv.t8 | bf16f32.v1.widen | 1.0750 | 0.5420 | 1.983 | dispatched |
| m3ultra | llama8b.qkv.t8 | int8i32.v1.flat | 0.9800 | 0.5420 | 1.808 | dispatched |
| m3ultra | llama8b.qkv.t8 | int8i32.v1.applechunk | 0.7570 | 0.5420 | 1.397 | probe,geometry=1 |
| m3ultra | llama8b.qkv.t8 | int8i32.v1.flat + quantize.a (DERIVED) | 2.1890 | 0.5420 | 4.039 | dispatched |
| m3ultra | llama8b.qkv.t8 | int8i32.v1.applechunk + quantize.a (DERIVED) | 1.9660 | 0.5420 | 3.627 | probe,geometry=1 |
| m3ultra | llama8b.qkv.t512 | bf16f32.v1.fused | 54.7230 | 3.3200 | 16.483 | not-dispatched(cells>16384) |
| m3ultra | llama8b.qkv.t512 | bf16f32.v1.widen | 3.8250 | 3.3200 | 1.152 | dispatched |
| m3ultra | llama8b.qkv.t512 | int8i32.v1.flat | 26.5880 | 3.3200 | 8.008 | dispatched |
| m3ultra | llama8b.qkv.t512 | int8i32.v1.flat + quantize.a (DERIVED) | 28.0110 | 3.3200 | 8.437 | dispatched |
| m3ultra | llama8b.qkv.t512 | int8i32.v1.applechunk + quantize.a (DERIVED) | 3.6680 | 3.3200 | 1.105 | probe,geometry=0 |
| m3ultra | llama8b.mlp_up.t1 | bf16f32.v1.widen | 2.9430 | 1.0440 | 2.819 | not-dispatched(cells<=16384) |
| m3ultra | llama8b.mlp_up.t1 | int8i32.v1.flat + quantize.a (DERIVED) | 1.8940 | 1.0440 | 1.814 | dispatched |
| m3ultra | llama8b.mlp_up.t1 | int8i32.v1.applechunk + quantize.a (DERIVED) | 1.9970 | 1.0440 | 1.913 | probe,geometry=1 |
| m3ultra | llama8b.mlp_up.t8 | bf16f32.v1.fused | 3.6300 | 1.0200 | 3.559 | not-dispatched(cells>16384) |
| m3ultra | llama8b.mlp_up.t8 | bf16f32.v1.widen | 2.9720 | 1.0200 | 2.914 | dispatched |
| m3ultra | llama8b.mlp_up.t8 | int8i32.v1.flat | 2.1560 | 1.0200 | 2.114 | dispatched |
| m3ultra | llama8b.mlp_up.t8 | int8i32.v1.flat + quantize.a (DERIVED) | 3.4070 | 1.0200 | 3.340 | dispatched |
| m3ultra | llama8b.mlp_up.t8 | int8i32.v1.applechunk + quantize.a (DERIVED) | 2.0770 | 1.0200 | 2.036 | probe,geometry=1 |
| m3ultra | llama8b.mlp_up.t512 | bf16f32.v1.fused | 189.1560 | 10.4170 | 18.158 | not-dispatched(cells>16384) |
| m3ultra | llama8b.mlp_up.t512 | bf16f32.v1.widen | 12.3530 | 10.4170 | 1.186 | dispatched |
| m3ultra | llama8b.mlp_up.t512 | int8i32.v1.flat | 91.1190 | 10.4170 | 8.747 | dispatched |
| m3ultra | llama8b.mlp_up.t512 | int8i32.v1.flat + quantize.a (DERIVED) | 92.5650 | 10.4170 | 8.886 | dispatched |
| m3ultra | llama8b.mlp_down.t1 | bf16f32.v1.widen | 4.7860 | 2.8720 | 1.666 | not-dispatched(cells<=16384) |
| m3ultra | llama8b.mlp_down.t1 | int8i32.v1.flat + quantize.a (DERIVED) | 5.1700 | 2.8720 | 1.800 | dispatched |
| m3ultra | llama8b.mlp_down.t1 | int8i32.v1.applechunk + quantize.a (DERIVED) | 5.3030 | 2.8720 | 1.846 | probe,geometry=1 |
| m3ultra | llama8b.mlp_down.t8 | bf16f32.v1.fused | 5.2120 | 1.3680 | 3.810 | not-dispatched(cells>16384) |
| m3ultra | llama8b.mlp_down.t8 | bf16f32.v1.widen | 3.3250 | 1.3680 | 2.431 | dispatched |
| m3ultra | llama8b.mlp_down.t8 | int8i32.v1.flat | 2.9650 | 1.3680 | 2.167 | dispatched |
| m3ultra | llama8b.mlp_down.t8 | int8i32.v1.applechunk | 1.9530 | 1.3680 | 1.428 | probe,geometry=1 |
| m3ultra | llama8b.mlp_down.t8 | int8i32.v1.flat + quantize.a (DERIVED) | 6.7470 | 1.3680 | 4.932 | dispatched |
| m3ultra | llama8b.mlp_down.t8 | int8i32.v1.applechunk + quantize.a (DERIVED) | 5.7350 | 1.3680 | 4.192 | probe,geometry=1 |
| m3ultra | llama8b.mlp_down.t512 | bf16f32.v1.fused | 191.1210 | 11.4650 | 16.670 | not-dispatched(cells>16384) |
| m3ultra | llama8b.mlp_down.t512 | bf16f32.v1.widen | 13.4240 | 11.4650 | 1.171 | dispatched |
| m3ultra | llama8b.mlp_down.t512 | int8i32.v1.flat | 117.6940 | 11.4650 | 10.266 | dispatched |
| m3ultra | llama8b.mlp_down.t512 | int8i32.v1.flat + quantize.a (DERIVED) | 122.4630 | 11.4650 | 10.681 | dispatched |
| m3ultra | llama8b.mlp_down.t512 | int8i32.v1.applechunk + quantize.a (DERIVED) | 11.9100 | 11.4650 | 1.039 | probe,geometry=0 |
| m3ultra | llama8b.lm_head.t1 | bf16f32.v1.widen | 21.4810 | 4.0720 | 5.275 | dispatched |
| m3ultra | llama8b.lm_head.t8 | bf16f32.v1.fused | 27.3660 | 5.7290 | 4.777 | not-dispatched(cells>16384) |
| m3ultra | llama8b.lm_head.t8 | bf16f32.v1.widen | 23.0470 | 5.7290 | 4.023 | dispatched |
| m3ultra | llama8b.lm_head.t8 | int8i32.v1.flat | 13.4380 | 5.7290 | 2.346 | dispatched |
| m3ultra | llama8b.lm_head.t8 | int8i32.v1.flat + quantize.a (DERIVED) | 14.6290 | 5.7290 | 2.553 | dispatched |
| m3ultra | llama8b.lm_head.t512 | bf16f32.v1.fused | 211.5180 | 11.5820 | 18.263 | not-dispatched(cells>16384) |
| m3ultra | llama8b.lm_head.t512 | bf16f32.v1.widen | 13.7680 | 11.5820 | 1.189 | dispatched |
| m3ultra | llama8b.lm_head.t512 | int8i32.v1.flat | 102.7290 | 11.5820 | 8.870 | dispatched |
| m3ultra | llama8b.lm_head.t512 | int8i32.v1.flat + quantize.a (DERIVED) | 104.1820 | 11.5820 | 8.995 | dispatched |
| m2pro | llama8b.qkv.t1 | bf16f32.v1.widen | 2.0390 | 1.5170 | 1.344 | not-dispatched(cells<=16384) |
| m2pro | llama8b.qkv.t1 | int8i32.v1.flat + quantize.a (DERIVED) | 2.6990 | 1.5170 | 1.779 | dispatched |
| m2pro | llama8b.qkv.t1 | int8i32.v1.applechunk + quantize.a (DERIVED) | 2.1510 | 1.5170 | 1.418 | probe,geometry=1 |
| m2pro | llama8b.qkv.t8 | bf16f32.v1.fused | 9.4700 | 1.4930 | 6.343 | not-dispatched(cells>16384) |
| m2pro | llama8b.qkv.t8 | bf16f32.v1.widen | 2.0760 | 1.4930 | 1.390 | dispatched |
| m2pro | llama8b.qkv.t8 | int8i32.v1.flat | 9.7600 | 1.4930 | 6.537 | dispatched |
| m2pro | llama8b.qkv.t8 | int8i32.v1.flat + quantize.a (DERIVED) | 11.1570 | 1.4930 | 7.473 | dispatched |
| m2pro | llama8b.qkv.t8 | int8i32.v1.applechunk + quantize.a (DERIVED) | 2.2330 | 1.4930 | 1.496 | probe,geometry=1 |
| m2pro | llama8b.qkv.t512 | bf16f32.v1.fused | 513.7060 | 17.7380 | 28.961 | not-dispatched(cells>16384) |
| m2pro | llama8b.qkv.t512 | bf16f32.v1.widen | 18.3010 | 17.7380 | 1.032 | dispatched |
| m2pro | llama8b.qkv.t512 | int8i32.v1.flat | 514.8840 | 17.7380 | 29.027 | dispatched |
| m2pro | llama8b.qkv.t512 | int8i32.v1.flat + quantize.a (DERIVED) | 518.3640 | 17.7380 | 29.223 | dispatched |
| m2pro | llama8b.mlp_up.t1 | bf16f32.v1.widen | 7.1240 | 5.2580 | 1.355 | not-dispatched(cells<=16384) |
| m2pro | llama8b.mlp_up.t1 | int8i32.v1.flat + quantize.a (DERIVED) | 6.3150 | 5.2580 | 1.201 | dispatched |
| m2pro | llama8b.mlp_up.t8 | bf16f32.v1.fused | 30.2360 | 4.1060 | 7.364 | not-dispatched(cells>16384) |
| m2pro | llama8b.mlp_up.t8 | bf16f32.v1.widen | 6.0270 | 4.1060 | 1.468 | dispatched |
| m2pro | llama8b.mlp_up.t8 | int8i32.v1.flat | 29.4060 | 4.1060 | 7.162 | dispatched |
| m2pro | llama8b.mlp_up.t8 | int8i32.v1.flat + quantize.a (DERIVED) | 30.8170 | 4.1060 | 7.505 | dispatched |
| m2pro | llama8b.mlp_up.t512 | bf16f32.v1.fused | 1794.7320 | 60.5230 | 29.654 | not-dispatched(cells>16384) |
| m2pro | llama8b.mlp_up.t512 | bf16f32.v1.widen | 62.3180 | 60.5230 | 1.030 | dispatched |
| m2pro | llama8b.mlp_up.t512 | int8i32.v1.flat | 1799.5110 | 60.5230 | 29.733 | dispatched |
| m2pro | llama8b.mlp_up.t512 | int8i32.v1.flat + quantize.a (DERIVED) | 1803.0180 | 60.5230 | 29.791 | dispatched |
| m2pro | llama8b.mlp_down.t1 | bf16f32.v1.widen | 7.4530 | 5.5350 | 1.347 | not-dispatched(cells<=16384) |
| m2pro | llama8b.mlp_down.t1 | int8i32.v1.flat + quantize.a (DERIVED) | 8.7400 | 5.5350 | 1.579 | dispatched |
| m2pro | llama8b.mlp_down.t1 | int8i32.v1.applechunk + quantize.a (DERIVED) | 6.8100 | 5.5350 | 1.230 | probe,geometry=1 |
| m2pro | llama8b.mlp_down.t8 | bf16f32.v1.fused | 34.5710 | 4.9780 | 6.945 | not-dispatched(cells>16384) |
| m2pro | llama8b.mlp_down.t8 | bf16f32.v1.widen | 6.9070 | 4.9780 | 1.388 | dispatched |
| m2pro | llama8b.mlp_down.t8 | int8i32.v1.flat | 33.9800 | 4.9780 | 6.826 | dispatched |
| m2pro | llama8b.mlp_down.t8 | int8i32.v1.flat + quantize.a (DERIVED) | 38.4410 | 4.9780 | 7.722 | dispatched |
| m2pro | llama8b.mlp_down.t8 | int8i32.v1.applechunk + quantize.a (DERIVED) | 6.9520 | 4.9780 | 1.397 | probe,geometry=1 |
| m2pro | llama8b.mlp_down.t512 | bf16f32.v1.fused | 1935.0970 | 65.5290 | 29.530 | not-dispatched(cells>16384) |
| m2pro | llama8b.mlp_down.t512 | bf16f32.v1.widen | 67.2640 | 65.5290 | 1.026 | dispatched |
| m2pro | llama8b.mlp_down.t512 | int8i32.v1.flat | 1829.4860 | 65.5290 | 27.919 | dispatched |
| m2pro | llama8b.mlp_down.t512 | int8i32.v1.flat + quantize.a (DERIVED) | 1841.5090 | 65.5290 | 28.102 | dispatched |
| m2pro | llama8b.lm_head.t1 | bf16f32.v1.fused | 33.7940 | 33.0710 | 1.022 | not-dispatched(cells>16384) |
| m2pro | llama8b.lm_head.t1 | bf16f32.v1.widen | 50.1380 | 33.0710 | 1.516 | dispatched |
| m2pro | llama8b.lm_head.t1 | int8i32.v1.flat + quantize.a (DERIVED) | 34.0380 | 33.0710 | 1.029 | dispatched |
| m2pro | llama8b.lm_head.t8 | bf16f32.v1.fused | 252.2930 | 33.2280 | 7.593 | not-dispatched(cells>16384) |
| m2pro | llama8b.lm_head.t8 | bf16f32.v1.widen | 50.2500 | 33.2280 | 1.512 | dispatched |
| m2pro | llama8b.lm_head.t8 | int8i32.v1.flat | 252.5490 | 33.2280 | 7.600 | dispatched |
| m2pro | llama8b.lm_head.t8 | int8i32.v1.flat + quantize.a (DERIVED) | 253.9850 | 33.2280 | 7.644 | dispatched |
| m2pro | llama8b.lm_head.t512 | bf16f32.v1.fused | 2011.1690 | 67.5780 | 29.761 | not-dispatched(cells>16384) |
| m2pro | llama8b.lm_head.t512 | bf16f32.v1.widen | 69.6350 | 67.5780 | 1.030 | dispatched |
| m2pro | llama8b.lm_head.t512 | int8i32.v1.flat | 2011.3610 | 67.5780 | 29.764 | dispatched |
| m2pro | llama8b.lm_head.t512 | int8i32.v1.flat + quantize.a (DERIVED) | 2014.8110 | 67.5780 | 29.815 | dispatched |
| mi325x | llama8b.qkv.t1 | bf16f32.v1.fused | 1.1355 | 0.0617 | 18.404 | dispatched |
| mi325x | llama8b.qkv.t1 | bf16f32.v1.widen | 0.0961 | 0.0617 | 1.558 | not-dispatched(cells<=16384) |
| mi325x | llama8b.qkv.t1 | int8i32.v1.flat | 1.0738 | 0.0617 | 17.404 | not-dispatched(the-column-has-the-unit) |
| mi325x | llama8b.qkv.t1 | int8i32.v1.flat + quantize.a (DERIVED) | 2.4309 | 0.0617 | 39.399 | not-dispatched(the-column-has-the-unit) |
| mi325x | llama8b.qkv.t1 | int8i32.v1.mma + quantize.a (DERIVED) | 1.4083 | 0.0617 | 22.825 | dispatched |
| mi325x | llama8b.qkv.t8 | bf16f32.v1.fused | 1.1883 | 0.0775 | 15.333 | not-dispatched(cells>16384) |
| mi325x | llama8b.qkv.t8 | bf16f32.v1.widen | 0.1067 | 0.0775 | 1.377 | dispatched |
| mi325x | llama8b.qkv.t8 | int8i32.v1.flat | 1.1356 | 0.0775 | 14.653 | not-dispatched(the-column-has-the-unit) |
| mi325x | llama8b.qkv.t8 | int8i32.v1.flat + quantize.a (DERIVED) | 2.5728 | 0.0775 | 33.197 | not-dispatched(the-column-has-the-unit) |
| mi325x | llama8b.qkv.t8 | int8i32.v1.mma + quantize.a (DERIVED) | 1.4888 | 0.0775 | 19.210 | dispatched |
| mi325x | llama8b.qkv.t512 | bf16f32.v1.fused | 41.2363 | 0.9254 | 44.561 | not-dispatched(cells>16384) |
| mi325x | llama8b.qkv.t512 | bf16f32.v1.widen | 0.9384 | 0.9254 | 1.014 | dispatched |
| mi325x | llama8b.qkv.t512 | int8i32.v1.flat | 41.5870 | 0.9254 | 44.939 | not-dispatched(the-column-has-the-unit) |
| mi325x | llama8b.qkv.t512 | int8i32.v1.flat + quantize.a (DERIVED) | 45.2469 | 0.9254 | 48.894 | not-dispatched(the-column-has-the-unit) |
| mi325x | llama8b.qkv.t512 | int8i32.v1.mma + quantize.a (DERIVED) | 3.9268 | 0.9254 | 4.243 | dispatched |
| mi325x | llama8b.mlp_up.t1 | bf16f32.v1.fused | 1.1869 | 0.1631 | 7.277 | dispatched |
| mi325x | llama8b.mlp_up.t1 | bf16f32.v1.widen | 0.2841 | 0.1631 | 1.742 | not-dispatched(cells<=16384) |
| mi325x | llama8b.mlp_up.t1 | int8i32.v1.flat | 1.1041 | 0.1631 | 6.769 | not-dispatched(the-column-has-the-unit) |
| mi325x | llama8b.mlp_up.t1 | int8i32.v1.flat + quantize.a (DERIVED) | 2.4610 | 0.1631 | 15.089 | not-dispatched(the-column-has-the-unit) |
| mi325x | llama8b.mlp_up.t1 | int8i32.v1.mma + quantize.a (DERIVED) | 1.4136 | 0.1631 | 8.667 | dispatched |
| mi325x | llama8b.mlp_up.t8 | bf16f32.v1.fused | 2.8442 | 0.1677 | 16.960 | not-dispatched(cells>16384) |
| mi325x | llama8b.mlp_up.t8 | bf16f32.v1.widen | 0.2917 | 0.1677 | 1.739 | dispatched |
| mi325x | llama8b.mlp_up.t8 | int8i32.v1.flat | 2.8417 | 0.1677 | 16.945 | not-dispatched(the-column-has-the-unit) |
| mi325x | llama8b.mlp_up.t8 | int8i32.v1.flat + quantize.a (DERIVED) | 4.2821 | 0.1677 | 25.534 | not-dispatched(the-column-has-the-unit) |
| mi325x | llama8b.mlp_up.t8 | int8i32.v1.mma + quantize.a (DERIVED) | 1.4987 | 0.1677 | 8.937 | dispatched |
| mi325x | llama8b.mlp_up.t512 | bf16f32.v1.fused | 174.3496 | 2.3895 | 72.965 | not-dispatched(cells>16384) |
| mi325x | llama8b.mlp_up.t512 | bf16f32.v1.widen | 2.4240 | 2.3895 | 1.014 | dispatched |
| mi325x | llama8b.mlp_up.t512 | int8i32.v1.flat | 151.1207 | 2.3895 | 63.244 | not-dispatched(the-column-has-the-unit) |
| mi325x | llama8b.mlp_up.t512 | int8i32.v1.flat + quantize.a (DERIVED) | 154.9134 | 2.3895 | 64.831 | not-dispatched(the-column-has-the-unit) |
| mi325x | llama8b.mlp_up.t512 | int8i32.v1.mma + quantize.a (DERIVED) | 4.6700 | 2.3895 | 1.954 | dispatched |
| mi325x | llama8b.mlp_down.t1 | bf16f32.v1.fused | 3.9476 | 0.1560 | 25.305 | dispatched |
| mi325x | llama8b.mlp_down.t1 | bf16f32.v1.widen | 0.2810 | 0.1560 | 1.801 | not-dispatched(cells<=16384) |
| mi325x | llama8b.mlp_down.t1 | int8i32.v1.flat | 3.8100 | 0.1560 | 24.423 | not-dispatched(the-column-has-the-unit) |
| mi325x | llama8b.mlp_down.t1 | int8i32.v1.flat + quantize.a (DERIVED) | 8.5008 | 0.1560 | 54.492 | not-dispatched(the-column-has-the-unit) |
| mi325x | llama8b.mlp_down.t1 | int8i32.v1.mma + quantize.a (DERIVED) | 4.8282 | 0.1560 | 30.950 | dispatched |
| mi325x | llama8b.mlp_down.t8 | bf16f32.v1.fused | 3.6965 | 0.1947 | 18.986 | not-dispatched(cells>16384) |
| mi325x | llama8b.mlp_down.t8 | bf16f32.v1.widen | 0.3105 | 0.1947 | 1.595 | dispatched |
| mi325x | llama8b.mlp_down.t8 | int8i32.v1.flat | 3.7483 | 0.1947 | 19.252 | not-dispatched(the-column-has-the-unit) |
| mi325x | llama8b.mlp_down.t8 | int8i32.v1.flat + quantize.a (DERIVED) | 8.9192 | 0.1947 | 45.810 | not-dispatched(the-column-has-the-unit) |
| mi325x | llama8b.mlp_down.t8 | int8i32.v1.mma + quantize.a (DERIVED) | 5.3124 | 0.1947 | 27.285 | dispatched |
| mi325x | llama8b.mlp_down.t512 | bf16f32.v1.fused | 192.6926 | 2.5107 | 76.749 | not-dispatched(cells>16384) |
| mi325x | llama8b.mlp_down.t512 | int8i32.v1.flat | 139.1355 | 2.5107 | 55.417 | not-dispatched(the-column-has-the-unit) |
| mi325x | llama8b.mlp_down.t512 | int8i32.v1.flat + quantize.a (DERIVED) | 152.4811 | 2.5107 | 60.733 | not-dispatched(the-column-has-the-unit) |
| mi325x | llama8b.mlp_down.t512 | int8i32.v1.mma + quantize.a (DERIVED) | 14.1878 | 2.5107 | 5.651 | dispatched |
| mi325x | llama8b.lm_head.t1 | bf16f32.v1.fused | 5.4303 | 1.3122 | 4.138 | not-dispatched(cells>16384) |
| mi325x | llama8b.lm_head.t1 | bf16f32.v1.widen | 2.4613 | 1.3122 | 1.876 | dispatched |
| mi325x | llama8b.lm_head.t1 | int8i32.v1.flat | 6.4084 | 1.3122 | 4.884 | not-dispatched(the-column-has-the-unit) |
| mi325x | llama8b.lm_head.t1 | int8i32.v1.flat + quantize.a (DERIVED) | 7.7760 | 1.3122 | 5.926 | not-dispatched(the-column-has-the-unit) |
| mi325x | llama8b.lm_head.t1 | int8i32.v1.mma + quantize.a (DERIVED) | 1.6538 | 1.3122 | 1.260 | dispatched |
| mi325x | llama8b.lm_head.t8 | bf16f32.v1.fused | 40.8262 | 1.2280 | 33.246 | not-dispatched(cells>16384) |
| mi325x | llama8b.lm_head.t8 | bf16f32.v1.widen | 2.4054 | 1.2280 | 1.959 | dispatched |
| mi325x | llama8b.lm_head.t8 | int8i32.v1.flat | 46.4979 | 1.2280 | 37.865 | not-dispatched(the-column-has-the-unit) |
| mi325x | llama8b.lm_head.t8 | int8i32.v1.flat + quantize.a (DERIVED) | 47.9573 | 1.2280 | 39.053 | not-dispatched(the-column-has-the-unit) |
| mi325x | llama8b.lm_head.t8 | int8i32.v1.mma + quantize.a (DERIVED) | 1.7917 | 1.2280 | 1.459 | dispatched |
| mi325x | llama8b.lm_head.t512 | bf16f32.v1.fused | 271.0917 | 2.7776 | 97.599 | not-dispatched(cells>16384) |
| mi325x | llama8b.lm_head.t512 | bf16f32.v1.widen | 2.8174 | 2.7776 | 1.014 | dispatched |
| mi325x | llama8b.lm_head.t512 | int8i32.v1.flat | 250.1423 | 2.7776 | 90.057 | not-dispatched(the-column-has-the-unit) |
| mi325x | llama8b.lm_head.t512 | int8i32.v1.flat + quantize.a (DERIVED) | 254.0723 | 2.7776 | 91.472 | not-dispatched(the-column-has-the-unit) |
| mi325x | llama8b.lm_head.t512 | int8i32.v1.mma + quantize.a (DERIVED) | 4.9430 | 2.7776 | 1.780 | dispatched |

## Digests across the boxes

AGREE: every box that ran the arm at the same extents printed the same digest, and at least two did.
ONE BOX: nothing to compare, which is not a pass.

| shape | arm | verdict | h100 | m3ultra | m2pro | mi325x |
|---|---|---|---|---|---|---|
| llama8b.qkv.t1 | fp32.v1 | AGREE | 0x920cdd138650a8bd | 0x920cdd138650a8bd | 0x920cdd138650a8bd | 0x920cdd138650a8bd |
| llama8b.qkv.t1 | bf16f32.v1.fused | AGREE | 0xcd39e9d6e6d7179a | 0xcd39e9d6e6d7179a | 0xcd39e9d6e6d7179a | 0xcd39e9d6e6d7179a |
| llama8b.qkv.t1 | bf16f32.v1.widen | AGREE | 0xcd39e9d6e6d7179a | 0xcd39e9d6e6d7179a | 0xcd39e9d6e6d7179a | 0xcd39e9d6e6d7179a |
| llama8b.qkv.t1 | int8i32.v1.flat | AGREE | 0x8a9227b93e430acd | 0x8a9227b93e430acd | 0x8a9227b93e430acd | 0x8a9227b93e430acd |
| llama8b.qkv.t1 | int8i32.v1.mma | AGREE | 0x8a9227b93e430acd | not run | not run | 0x8a9227b93e430acd |
| llama8b.qkv.t1 | int8i32.v1.applechunk | AGREE | not run | 0x8a9227b93e430acd | 0x8a9227b93e430acd | not run |
| llama8b.qkv.t1 | convert.int8.quantize.a | AGREE | 0x54279cbda1444726 | 0x54279cbda1444726 | 0x54279cbda1444726 | 0x54279cbda1444726 |
| llama8b.qkv.t1 | convert.int8.pack.b | AGREE | 0x305b1a502296caf6 | 0x305b1a502296caf6 | 0x305b1a502296caf6 | 0x305b1a502296caf6 |
| llama8b.qkv.t1 | convert.bf16.pack.b | AGREE | 0x24efed638bb6b96b | 0x24efed638bb6b96b | 0x24efed638bb6b96b | 0x24efed638bb6b96b |
| llama8b.qkv.t1 | convert.bf16.widen.b | AGREE | 0xd2c2d96392622325 | 0xd2c2d96392622325 | 0xd2c2d96392622325 | 0xd2c2d96392622325 |
| llama8b.qkv.t1 | convert.int8.dequantize.b | AGREE | 0x2fd9ee536fd82325 | 0x2fd9ee536fd82325 | 0x2fd9ee536fd82325 | 0x2fd9ee536fd82325 |
| llama8b.qkv.t1 | inference.bf16f32.v1 | AGREE | 0xcd39e9d6e6d7179a | 0xcd39e9d6e6d7179a | 0xcd39e9d6e6d7179a | 0xcd39e9d6e6d7179a |
| llama8b.qkv.t1 | inference.int8i32.v1 | AGREE | 0x8a9227b93e430acd | 0x8a9227b93e430acd | 0x8a9227b93e430acd | 0x8a9227b93e430acd |
| llama8b.qkv.t1 | inference.int8i32.v1.applechunk | AGREE | not run | 0x8a9227b93e430acd | 0x8a9227b93e430acd | not run |
| llama8b.qkv.t1 | training.bf16f32.v1 | AGREE | 0xcd39e9d6e6d7179a | 0xcd39e9d6e6d7179a | 0xcd39e9d6e6d7179a | 0xcd39e9d6e6d7179a |
| llama8b.qkv.t1 | training.int8i32.v1 | AGREE | 0x8a9227b93e430acd | 0x8a9227b93e430acd | 0x8a9227b93e430acd | 0x8a9227b93e430acd |
| llama8b.qkv.t1 | training.int8i32.v1.applechunk | AGREE | not run | 0x8a9227b93e430acd | 0x8a9227b93e430acd | not run |
| llama8b.qkv.t8 | fp32.v1 | AGREE | 0xd5677d8a13939709 | 0xd5677d8a13939709 | 0xd5677d8a13939709 | 0xd5677d8a13939709 |
| llama8b.qkv.t8 | bf16f32.v1.fused | AGREE | 0xf36fe1a8828cefd9 | 0xf36fe1a8828cefd9 | 0xf36fe1a8828cefd9 | 0xf36fe1a8828cefd9 |
| llama8b.qkv.t8 | bf16f32.v1.widen | AGREE | 0xf36fe1a8828cefd9 | 0xf36fe1a8828cefd9 | 0xf36fe1a8828cefd9 | 0xf36fe1a8828cefd9 |
| llama8b.qkv.t8 | int8i32.v1.flat | AGREE | 0x484bd20aabe66ee5 | 0x484bd20aabe66ee5 | 0x484bd20aabe66ee5 | 0x484bd20aabe66ee5 |
| llama8b.qkv.t8 | int8i32.v1.mma | AGREE | 0x484bd20aabe66ee5 | not run | not run | 0x484bd20aabe66ee5 |
| llama8b.qkv.t8 | int8i32.v1.applechunk | AGREE | not run | 0x484bd20aabe66ee5 | 0x484bd20aabe66ee5 | not run |
| llama8b.qkv.t8 | convert.int8.quantize.a | AGREE | 0xbd0b5aed2b83cb45 | 0xbd0b5aed2b83cb45 | 0xbd0b5aed2b83cb45 | 0xbd0b5aed2b83cb45 |
| llama8b.qkv.t8 | convert.int8.pack.b | AGREE | 0xa8ce17326c01345 | 0xa8ce17326c01345 | 0xa8ce17326c01345 | 0xa8ce17326c01345 |
| llama8b.qkv.t8 | convert.bf16.pack.b | AGREE | 0xdccc5a093a080c52 | 0xdccc5a093a080c52 | 0xdccc5a093a080c52 | 0xdccc5a093a080c52 |
| llama8b.qkv.t8 | convert.bf16.widen.b | AGREE | 0x944a5da41b4d2325 | 0x944a5da41b4d2325 | 0x944a5da41b4d2325 | 0x944a5da41b4d2325 |
| llama8b.qkv.t8 | convert.int8.dequantize.b | AGREE | 0x24697157b3da2325 | 0x24697157b3da2325 | 0x24697157b3da2325 | 0x24697157b3da2325 |
| llama8b.qkv.t8 | inference.bf16f32.v1 | AGREE | 0xf36fe1a8828cefd9 | 0xf36fe1a8828cefd9 | 0xf36fe1a8828cefd9 | 0xf36fe1a8828cefd9 |
| llama8b.qkv.t8 | inference.int8i32.v1 | AGREE | 0x484bd20aabe66ee5 | 0x484bd20aabe66ee5 | 0x484bd20aabe66ee5 | 0x484bd20aabe66ee5 |
| llama8b.qkv.t8 | inference.int8i32.v1.applechunk | AGREE | not run | 0x484bd20aabe66ee5 | 0x484bd20aabe66ee5 | not run |
| llama8b.qkv.t8 | training.bf16f32.v1 | AGREE | 0xf36fe1a8828cefd9 | 0xf36fe1a8828cefd9 | 0xf36fe1a8828cefd9 | 0xf36fe1a8828cefd9 |
| llama8b.qkv.t8 | training.int8i32.v1 | AGREE | 0x484bd20aabe66ee5 | 0x484bd20aabe66ee5 | 0x484bd20aabe66ee5 | 0x484bd20aabe66ee5 |
| llama8b.qkv.t8 | training.int8i32.v1.applechunk | AGREE | not run | 0x484bd20aabe66ee5 | 0x484bd20aabe66ee5 | not run |
| llama8b.qkv.t512 | fp32.v1 | AGREE | 0x67d3d940546b834c | 0x67d3d940546b834c | 0x67d3d940546b834c | 0x67d3d940546b834c |
| llama8b.qkv.t512 | bf16f32.v1.fused | AGREE | 0x4abd79a5b10221e5 | 0x4abd79a5b10221e5 | 0x4abd79a5b10221e5 | 0x4abd79a5b10221e5 |
| llama8b.qkv.t512 | bf16f32.v1.widen | AGREE | 0x4abd79a5b10221e5 | 0x4abd79a5b10221e5 | 0x4abd79a5b10221e5 | 0x4abd79a5b10221e5 |
| llama8b.qkv.t512 | int8i32.v1.flat | AGREE | 0xa5f2ce1f1b9a0ebd | 0xa5f2ce1f1b9a0ebd | 0xa5f2ce1f1b9a0ebd | 0xa5f2ce1f1b9a0ebd |
| llama8b.qkv.t512 | int8i32.v1.mma | AGREE | 0xa5f2ce1f1b9a0ebd | not run | not run | 0xa5f2ce1f1b9a0ebd |
| llama8b.qkv.t512 | int8i32.v1.applechunk | AGREE | not run | 0xa5f2ce1f1b9a0ebd | 0xa5f2ce1f1b9a0ebd | not run |
| llama8b.qkv.t512 | convert.int8.quantize.a | AGREE | 0xe8bb3ad105a4ae26 | 0xe8bb3ad105a4ae26 | 0xe8bb3ad105a4ae26 | 0xe8bb3ad105a4ae26 |
| llama8b.qkv.t512 | convert.int8.pack.b | AGREE | 0xd5ea46fdcb251db4 | 0xd5ea46fdcb251db4 | 0xd5ea46fdcb251db4 | 0xd5ea46fdcb251db4 |
| llama8b.qkv.t512 | convert.bf16.pack.b | AGREE | 0xaef2da473a8e2a10 | 0xaef2da473a8e2a10 | 0xaef2da473a8e2a10 | 0xaef2da473a8e2a10 |
| llama8b.qkv.t512 | convert.bf16.widen.b | AGREE | 0xc8e5243510392325 | 0xc8e5243510392325 | 0xc8e5243510392325 | 0xc8e5243510392325 |
| llama8b.qkv.t512 | convert.int8.dequantize.b | AGREE | 0x1798a32eac742325 | 0x1798a32eac742325 | 0x1798a32eac742325 | 0x1798a32eac742325 |
| llama8b.qkv.t512 | inference.bf16f32.v1 | AGREE | 0x4abd79a5b10221e5 | 0x4abd79a5b10221e5 | 0x4abd79a5b10221e5 | 0x4abd79a5b10221e5 |
| llama8b.qkv.t512 | inference.int8i32.v1 | AGREE | 0xa5f2ce1f1b9a0ebd | 0xa5f2ce1f1b9a0ebd | 0xa5f2ce1f1b9a0ebd | 0xa5f2ce1f1b9a0ebd |
| llama8b.qkv.t512 | inference.int8i32.v1.applechunk | AGREE | not run | 0xa5f2ce1f1b9a0ebd | 0xa5f2ce1f1b9a0ebd | not run |
| llama8b.qkv.t512 | training.bf16f32.v1 | AGREE | 0x4abd79a5b10221e5 | 0x4abd79a5b10221e5 | 0x4abd79a5b10221e5 | 0x4abd79a5b10221e5 |
| llama8b.qkv.t512 | training.int8i32.v1 | AGREE | 0xa5f2ce1f1b9a0ebd | 0xa5f2ce1f1b9a0ebd | 0xa5f2ce1f1b9a0ebd | 0xa5f2ce1f1b9a0ebd |
| llama8b.qkv.t512 | training.int8i32.v1.applechunk | AGREE | not run | 0xa5f2ce1f1b9a0ebd | 0xa5f2ce1f1b9a0ebd | not run |
| llama8b.mlp_up.t1 | fp32.v1 | AGREE | 0x4965ea6af2ff5e61 | 0x4965ea6af2ff5e61 | 0x4965ea6af2ff5e61 | 0x4965ea6af2ff5e61 |
| llama8b.mlp_up.t1 | bf16f32.v1.fused | AGREE | 0x5e3f5ff35a448024 | 0x5e3f5ff35a448024 | 0x5e3f5ff35a448024 | 0x5e3f5ff35a448024 |
| llama8b.mlp_up.t1 | bf16f32.v1.widen | AGREE | 0x5e3f5ff35a448024 | 0x5e3f5ff35a448024 | 0x5e3f5ff35a448024 | 0x5e3f5ff35a448024 |
| llama8b.mlp_up.t1 | int8i32.v1.flat | AGREE | 0xff302668595b02d | 0xff302668595b02d | 0xff302668595b02d | 0xff302668595b02d |
| llama8b.mlp_up.t1 | int8i32.v1.mma | AGREE | 0xff302668595b02d | not run | not run | 0xff302668595b02d |
| llama8b.mlp_up.t1 | int8i32.v1.applechunk | AGREE | not run | 0xff302668595b02d | 0xff302668595b02d | not run |
| llama8b.mlp_up.t1 | convert.int8.quantize.a | AGREE | 0x1343c750c1ded02 | 0x1343c750c1ded02 | 0x1343c750c1ded02 | 0x1343c750c1ded02 |
| llama8b.mlp_up.t1 | convert.int8.pack.b | AGREE | 0x9dd5dd0e7f10b4d0 | 0x9dd5dd0e7f10b4d0 | 0x9dd5dd0e7f10b4d0 | 0x9dd5dd0e7f10b4d0 |
| llama8b.mlp_up.t1 | convert.bf16.pack.b | AGREE | 0xed618e2333b13294 | 0xed618e2333b13294 | 0xed618e2333b13294 | 0xed618e2333b13294 |
| llama8b.mlp_up.t1 | convert.bf16.widen.b | AGREE | 0x573b3a44b4012325 | 0x573b3a44b4012325 | 0x573b3a44b4012325 | 0x573b3a44b4012325 |
| llama8b.mlp_up.t1 | convert.int8.dequantize.b | AGREE | 0x5a7e8639220c2325 | 0x5a7e8639220c2325 | 0x5a7e8639220c2325 | 0x5a7e8639220c2325 |
| llama8b.mlp_up.t1 | inference.bf16f32.v1 | AGREE | 0x5e3f5ff35a448024 | 0x5e3f5ff35a448024 | 0x5e3f5ff35a448024 | 0x5e3f5ff35a448024 |
| llama8b.mlp_up.t1 | inference.int8i32.v1 | AGREE | 0xff302668595b02d | 0xff302668595b02d | 0xff302668595b02d | 0xff302668595b02d |
| llama8b.mlp_up.t1 | inference.int8i32.v1.applechunk | AGREE | not run | 0xff302668595b02d | 0xff302668595b02d | not run |
| llama8b.mlp_up.t1 | training.bf16f32.v1 | AGREE | 0x5e3f5ff35a448024 | 0x5e3f5ff35a448024 | 0x5e3f5ff35a448024 | 0x5e3f5ff35a448024 |
| llama8b.mlp_up.t1 | training.int8i32.v1 | AGREE | 0xff302668595b02d | 0xff302668595b02d | 0xff302668595b02d | 0xff302668595b02d |
| llama8b.mlp_up.t1 | training.int8i32.v1.applechunk | AGREE | not run | 0xff302668595b02d | 0xff302668595b02d | not run |
| llama8b.mlp_up.t8 | fp32.v1 | AGREE | 0x7b30465d325906e | 0x7b30465d325906e | 0x7b30465d325906e | 0x7b30465d325906e |
| llama8b.mlp_up.t8 | bf16f32.v1.fused | AGREE | 0x1531c10dcd152579 | 0x1531c10dcd152579 | 0x1531c10dcd152579 | 0x1531c10dcd152579 |
| llama8b.mlp_up.t8 | bf16f32.v1.widen | AGREE | 0x1531c10dcd152579 | 0x1531c10dcd152579 | 0x1531c10dcd152579 | 0x1531c10dcd152579 |
| llama8b.mlp_up.t8 | int8i32.v1.flat | AGREE | 0xbd4b3a783f9d9a0d | 0xbd4b3a783f9d9a0d | 0xbd4b3a783f9d9a0d | 0xbd4b3a783f9d9a0d |
| llama8b.mlp_up.t8 | int8i32.v1.mma | AGREE | 0xbd4b3a783f9d9a0d | not run | not run | 0xbd4b3a783f9d9a0d |
| llama8b.mlp_up.t8 | int8i32.v1.applechunk | AGREE | not run | 0xbd4b3a783f9d9a0d | 0xbd4b3a783f9d9a0d | not run |
| llama8b.mlp_up.t8 | convert.int8.quantize.a | AGREE | 0x5e46321f760ef85d | 0x5e46321f760ef85d | 0x5e46321f760ef85d | 0x5e46321f760ef85d |
| llama8b.mlp_up.t8 | convert.int8.pack.b | AGREE | 0xa9e079afde494e3d | 0xa9e079afde494e3d | 0xa9e079afde494e3d | 0xa9e079afde494e3d |
| llama8b.mlp_up.t8 | convert.bf16.pack.b | AGREE | 0xc5111a6224412c3f | 0xc5111a6224412c3f | 0xc5111a6224412c3f | 0xc5111a6224412c3f |
| llama8b.mlp_up.t8 | convert.bf16.widen.b | AGREE | 0x6a679200ae02325 | 0x6a679200ae02325 | 0x6a679200ae02325 | 0x6a679200ae02325 |
| llama8b.mlp_up.t8 | convert.int8.dequantize.b | AGREE | 0x89939dca58a02325 | 0x89939dca58a02325 | 0x89939dca58a02325 | 0x89939dca58a02325 |
| llama8b.mlp_up.t8 | inference.bf16f32.v1 | AGREE | 0x1531c10dcd152579 | 0x1531c10dcd152579 | 0x1531c10dcd152579 | 0x1531c10dcd152579 |
| llama8b.mlp_up.t8 | inference.int8i32.v1 | AGREE | 0xbd4b3a783f9d9a0d | 0xbd4b3a783f9d9a0d | 0xbd4b3a783f9d9a0d | 0xbd4b3a783f9d9a0d |
| llama8b.mlp_up.t8 | inference.int8i32.v1.applechunk | AGREE | not run | 0xbd4b3a783f9d9a0d | 0xbd4b3a783f9d9a0d | not run |
| llama8b.mlp_up.t8 | training.bf16f32.v1 | AGREE | 0x1531c10dcd152579 | 0x1531c10dcd152579 | 0x1531c10dcd152579 | 0x1531c10dcd152579 |
| llama8b.mlp_up.t8 | training.int8i32.v1 | AGREE | 0xbd4b3a783f9d9a0d | 0xbd4b3a783f9d9a0d | 0xbd4b3a783f9d9a0d | 0xbd4b3a783f9d9a0d |
| llama8b.mlp_up.t8 | training.int8i32.v1.applechunk | AGREE | not run | 0xbd4b3a783f9d9a0d | 0xbd4b3a783f9d9a0d | not run |
| llama8b.mlp_up.t512 | fp32.v1 | AGREE | 0x27a166783c45f9f0 | 0x27a166783c45f9f0 | 0x27a166783c45f9f0 | 0x27a166783c45f9f0 |
| llama8b.mlp_up.t512 | bf16f32.v1.fused | AGREE | 0x5b29318ad80be465 | 0x5b29318ad80be465 | 0x5b29318ad80be465 | 0x5b29318ad80be465 |
| llama8b.mlp_up.t512 | bf16f32.v1.widen | AGREE | 0x5b29318ad80be465 | 0x5b29318ad80be465 | 0x5b29318ad80be465 | 0x5b29318ad80be465 |
| llama8b.mlp_up.t512 | int8i32.v1.flat | AGREE | 0x220ff5fe96f5ef55 | 0x220ff5fe96f5ef55 | 0x220ff5fe96f5ef55 | 0x220ff5fe96f5ef55 |
| llama8b.mlp_up.t512 | int8i32.v1.mma | AGREE | 0x220ff5fe96f5ef55 | not run | not run | 0x220ff5fe96f5ef55 |
| llama8b.mlp_up.t512 | int8i32.v1.applechunk | AGREE | not run | 0x220ff5fe96f5ef55 | 0x220ff5fe96f5ef55 | not run |
| llama8b.mlp_up.t512 | convert.int8.quantize.a | AGREE | 0x8c503fb082856006 | 0x8c503fb082856006 | 0x8c503fb082856006 | 0x8c503fb082856006 |
| llama8b.mlp_up.t512 | convert.int8.pack.b | AGREE | 0xf8c1486c6aaba4c1 | 0xf8c1486c6aaba4c1 | 0xf8c1486c6aaba4c1 | 0xf8c1486c6aaba4c1 |
| llama8b.mlp_up.t512 | convert.bf16.pack.b | AGREE | 0x1c04d570a3d23380 | 0x1c04d570a3d23380 | 0x1c04d570a3d23380 | 0x1c04d570a3d23380 |
| llama8b.mlp_up.t512 | convert.bf16.widen.b | AGREE | 0x5c9d5dc97a412325 | 0x5c9d5dc97a412325 | 0x5c9d5dc97a412325 | 0x5c9d5dc97a412325 |
| llama8b.mlp_up.t512 | convert.int8.dequantize.b | AGREE | 0xb584f740e2d62325 | 0xb584f740e2d62325 | 0xb584f740e2d62325 | 0xb584f740e2d62325 |
| llama8b.mlp_up.t512 | inference.bf16f32.v1 | AGREE | 0x5b29318ad80be465 | 0x5b29318ad80be465 | 0x5b29318ad80be465 | 0x5b29318ad80be465 |
| llama8b.mlp_up.t512 | inference.int8i32.v1 | AGREE | 0x220ff5fe96f5ef55 | 0x220ff5fe96f5ef55 | 0x220ff5fe96f5ef55 | 0x220ff5fe96f5ef55 |
| llama8b.mlp_up.t512 | inference.int8i32.v1.applechunk | AGREE | not run | 0x220ff5fe96f5ef55 | 0x220ff5fe96f5ef55 | not run |
| llama8b.mlp_up.t512 | training.bf16f32.v1 | AGREE | 0x5b29318ad80be465 | 0x5b29318ad80be465 | 0x5b29318ad80be465 | 0x5b29318ad80be465 |
| llama8b.mlp_up.t512 | training.int8i32.v1 | AGREE | 0x220ff5fe96f5ef55 | 0x220ff5fe96f5ef55 | 0x220ff5fe96f5ef55 | 0x220ff5fe96f5ef55 |
| llama8b.mlp_up.t512 | training.int8i32.v1.applechunk | AGREE | not run | 0x220ff5fe96f5ef55 | 0x220ff5fe96f5ef55 | not run |
| llama8b.mlp_down.t1 | fp32.v1 | AGREE | 0x4efb8e54c724fa9 | 0x4efb8e54c724fa9 | 0x4efb8e54c724fa9 | 0x4efb8e54c724fa9 |
| llama8b.mlp_down.t1 | bf16f32.v1.fused | AGREE | 0xd044aebb510dc9 | 0xd044aebb510dc9 | 0xd044aebb510dc9 | 0xd044aebb510dc9 |
| llama8b.mlp_down.t1 | bf16f32.v1.widen | AGREE | 0xd044aebb510dc9 | 0xd044aebb510dc9 | 0xd044aebb510dc9 | 0xd044aebb510dc9 |
| llama8b.mlp_down.t1 | int8i32.v1.flat | AGREE | 0x1ffdde3ffffb72c9 | 0x1ffdde3ffffb72c9 | 0x1ffdde3ffffb72c9 | 0x1ffdde3ffffb72c9 |
| llama8b.mlp_down.t1 | int8i32.v1.mma | AGREE | 0x1ffdde3ffffb72c9 | not run | not run | 0x1ffdde3ffffb72c9 |
| llama8b.mlp_down.t1 | int8i32.v1.applechunk | AGREE | not run | 0x1ffdde3ffffb72c9 | 0x1ffdde3ffffb72c9 | not run |
| llama8b.mlp_down.t1 | convert.int8.quantize.a | AGREE | 0x160d734878c0f458 | 0x160d734878c0f458 | 0x160d734878c0f458 | 0x160d734878c0f458 |
| llama8b.mlp_down.t1 | convert.int8.pack.b | AGREE | 0x31274090cc79ba2f | 0x31274090cc79ba2f | 0x31274090cc79ba2f | 0x31274090cc79ba2f |
| llama8b.mlp_down.t1 | convert.bf16.pack.b | AGREE | 0x9dba12f4fe0c9e9b | 0x9dba12f4fe0c9e9b | 0x9dba12f4fe0c9e9b | 0x9dba12f4fe0c9e9b |
| llama8b.mlp_down.t1 | convert.bf16.widen.b | AGREE | 0x45255120c9d82325 | 0x45255120c9d82325 | 0x45255120c9d82325 | 0x45255120c9d82325 |
| llama8b.mlp_down.t1 | convert.int8.dequantize.b | AGREE | 0x8e72f1c5a3402325 | 0x8e72f1c5a3402325 | 0x8e72f1c5a3402325 | 0x8e72f1c5a3402325 |
| llama8b.mlp_down.t1 | inference.bf16f32.v1 | AGREE | 0xd044aebb510dc9 | 0xd044aebb510dc9 | 0xd044aebb510dc9 | 0xd044aebb510dc9 |
| llama8b.mlp_down.t1 | inference.int8i32.v1 | AGREE | 0x1ffdde3ffffb72c9 | 0x1ffdde3ffffb72c9 | 0x1ffdde3ffffb72c9 | 0x1ffdde3ffffb72c9 |
| llama8b.mlp_down.t1 | inference.int8i32.v1.applechunk | AGREE | not run | 0x1ffdde3ffffb72c9 | 0x1ffdde3ffffb72c9 | not run |
| llama8b.mlp_down.t1 | training.bf16f32.v1 | AGREE | 0xd044aebb510dc9 | 0xd044aebb510dc9 | 0xd044aebb510dc9 | 0xd044aebb510dc9 |
| llama8b.mlp_down.t1 | training.int8i32.v1 | AGREE | 0x1ffdde3ffffb72c9 | 0x1ffdde3ffffb72c9 | 0x1ffdde3ffffb72c9 | 0x1ffdde3ffffb72c9 |
| llama8b.mlp_down.t1 | training.int8i32.v1.applechunk | AGREE | not run | 0x1ffdde3ffffb72c9 | 0x1ffdde3ffffb72c9 | not run |
| llama8b.mlp_down.t8 | fp32.v1 | AGREE | 0xfb21004541ae3e8e | 0xfb21004541ae3e8e | 0xfb21004541ae3e8e | 0xfb21004541ae3e8e |
| llama8b.mlp_down.t8 | bf16f32.v1.fused | AGREE | 0xcbe807d3ae13612f | 0xcbe807d3ae13612f | 0xcbe807d3ae13612f | 0xcbe807d3ae13612f |
| llama8b.mlp_down.t8 | bf16f32.v1.widen | AGREE | 0xcbe807d3ae13612f | 0xcbe807d3ae13612f | 0xcbe807d3ae13612f | 0xcbe807d3ae13612f |
| llama8b.mlp_down.t8 | int8i32.v1.flat | AGREE | 0x694aafd520fdadad | 0x694aafd520fdadad | 0x694aafd520fdadad | 0x694aafd520fdadad |
| llama8b.mlp_down.t8 | int8i32.v1.mma | AGREE | 0x694aafd520fdadad | not run | not run | 0x694aafd520fdadad |
| llama8b.mlp_down.t8 | int8i32.v1.applechunk | AGREE | not run | 0x694aafd520fdadad | 0x694aafd520fdadad | not run |
| llama8b.mlp_down.t8 | convert.int8.quantize.a | AGREE | 0xbb4f2a7232392bfe | 0xbb4f2a7232392bfe | 0xbb4f2a7232392bfe | 0xbb4f2a7232392bfe |
| llama8b.mlp_down.t8 | convert.int8.pack.b | AGREE | 0x4b12dcda2ab9acd4 | 0x4b12dcda2ab9acd4 | 0x4b12dcda2ab9acd4 | 0x4b12dcda2ab9acd4 |
| llama8b.mlp_down.t8 | convert.bf16.pack.b | AGREE | 0xfbacba127cf6177e | 0xfbacba127cf6177e | 0xfbacba127cf6177e | 0xfbacba127cf6177e |
| llama8b.mlp_down.t8 | convert.bf16.widen.b | AGREE | 0xcbcb5933d8352325 | 0xcbcb5933d8352325 | 0xcbcb5933d8352325 | 0xcbcb5933d8352325 |
| llama8b.mlp_down.t8 | convert.int8.dequantize.b | AGREE | 0x9dcad54d7e462325 | 0x9dcad54d7e462325 | 0x9dcad54d7e462325 | 0x9dcad54d7e462325 |
| llama8b.mlp_down.t8 | inference.bf16f32.v1 | AGREE | 0xcbe807d3ae13612f | 0xcbe807d3ae13612f | 0xcbe807d3ae13612f | 0xcbe807d3ae13612f |
| llama8b.mlp_down.t8 | inference.int8i32.v1 | AGREE | 0x694aafd520fdadad | 0x694aafd520fdadad | 0x694aafd520fdadad | 0x694aafd520fdadad |
| llama8b.mlp_down.t8 | inference.int8i32.v1.applechunk | AGREE | not run | 0x694aafd520fdadad | 0x694aafd520fdadad | not run |
| llama8b.mlp_down.t8 | training.bf16f32.v1 | AGREE | 0xcbe807d3ae13612f | 0xcbe807d3ae13612f | 0xcbe807d3ae13612f | 0xcbe807d3ae13612f |
| llama8b.mlp_down.t8 | training.int8i32.v1 | AGREE | 0x694aafd520fdadad | 0x694aafd520fdadad | 0x694aafd520fdadad | 0x694aafd520fdadad |
| llama8b.mlp_down.t8 | training.int8i32.v1.applechunk | AGREE | not run | 0x694aafd520fdadad | 0x694aafd520fdadad | not run |
| llama8b.mlp_down.t512 | fp32.v1 | AGREE | 0x4743c97cb961339e | 0x4743c97cb961339e | 0x4743c97cb961339e | 0x4743c97cb961339e |
| llama8b.mlp_down.t512 | bf16f32.v1.fused | AGREE | 0x508d64e0e41d3350 | 0x508d64e0e41d3350 | 0x508d64e0e41d3350 | 0x508d64e0e41d3350 |
| llama8b.mlp_down.t512 | bf16f32.v1.widen | AGREE | 0x508d64e0e41d3350 | 0x508d64e0e41d3350 | 0x508d64e0e41d3350 | 0x508d64e0e41d3350 |
| llama8b.mlp_down.t512 | int8i32.v1.flat | AGREE | 0x6120545eacb068ad | 0x6120545eacb068ad | 0x6120545eacb068ad | 0x6120545eacb068ad |
| llama8b.mlp_down.t512 | int8i32.v1.mma | AGREE | 0x6120545eacb068ad | not run | not run | 0x6120545eacb068ad |
| llama8b.mlp_down.t512 | int8i32.v1.applechunk | AGREE | not run | 0x6120545eacb068ad | 0x6120545eacb068ad | not run |
| llama8b.mlp_down.t512 | convert.int8.quantize.a | AGREE | 0x903f6ff84c6082 | 0x903f6ff84c6082 | 0x903f6ff84c6082 | 0x903f6ff84c6082 |
| llama8b.mlp_down.t512 | convert.int8.pack.b | AGREE | 0xa046020a81d05746 | 0xa046020a81d05746 | 0xa046020a81d05746 | 0xa046020a81d05746 |
| llama8b.mlp_down.t512 | convert.bf16.pack.b | AGREE | 0xd1b066f4e6a61fed | 0xd1b066f4e6a61fed | 0xd1b066f4e6a61fed | 0xd1b066f4e6a61fed |
| llama8b.mlp_down.t512 | convert.bf16.widen.b | AGREE | 0xf0fcf217ba5c2325 | 0xf0fcf217ba5c2325 | 0xf0fcf217ba5c2325 | 0xf0fcf217ba5c2325 |
| llama8b.mlp_down.t512 | convert.int8.dequantize.b | AGREE | 0xd972f24f17be2325 | 0xd972f24f17be2325 | 0xd972f24f17be2325 | 0xd972f24f17be2325 |
| llama8b.mlp_down.t512 | inference.bf16f32.v1 | AGREE | 0x508d64e0e41d3350 | 0x508d64e0e41d3350 | 0x508d64e0e41d3350 | 0x508d64e0e41d3350 |
| llama8b.mlp_down.t512 | inference.int8i32.v1 | AGREE | 0x6120545eacb068ad | 0x6120545eacb068ad | 0x6120545eacb068ad | 0x6120545eacb068ad |
| llama8b.mlp_down.t512 | inference.int8i32.v1.applechunk | AGREE | not run | 0x6120545eacb068ad | 0x6120545eacb068ad | not run |
| llama8b.mlp_down.t512 | training.bf16f32.v1 | AGREE | 0x508d64e0e41d3350 | 0x508d64e0e41d3350 | 0x508d64e0e41d3350 | 0x508d64e0e41d3350 |
| llama8b.mlp_down.t512 | training.int8i32.v1 | AGREE | 0x6120545eacb068ad | 0x6120545eacb068ad | 0x6120545eacb068ad | 0x6120545eacb068ad |
| llama8b.mlp_down.t512 | training.int8i32.v1.applechunk | AGREE | not run | 0x6120545eacb068ad | 0x6120545eacb068ad | not run |
| llama8b.lm_head.t1 | fp32.v1 | AGREE | 0xcdc134cbce391535 | 0xcdc134cbce391535 | 0xcdc134cbce391535 | 0xcdc134cbce391535 |
| llama8b.lm_head.t1 | bf16f32.v1.fused | AGREE | 0x23f2e7e17341274a | 0x23f2e7e17341274a | 0x23f2e7e17341274a | 0x23f2e7e17341274a |
| llama8b.lm_head.t1 | bf16f32.v1.widen | AGREE | 0x23f2e7e17341274a | 0x23f2e7e17341274a | 0x23f2e7e17341274a | 0x23f2e7e17341274a |
| llama8b.lm_head.t1 | int8i32.v1.flat | AGREE | 0xe0222b2f9de835ed | 0xe0222b2f9de835ed | 0xe0222b2f9de835ed | 0xe0222b2f9de835ed |
| llama8b.lm_head.t1 | int8i32.v1.mma | AGREE | 0xe0222b2f9de835ed | not run | not run | 0xe0222b2f9de835ed |
| llama8b.lm_head.t1 | int8i32.v1.applechunk | AGREE | not run | 0xe0222b2f9de835ed | 0xe0222b2f9de835ed | not run |
| llama8b.lm_head.t1 | convert.int8.quantize.a | AGREE | 0x67d2e298b047bf2b | 0x67d2e298b047bf2b | 0x67d2e298b047bf2b | 0x67d2e298b047bf2b |
| llama8b.lm_head.t1 | convert.int8.pack.b | AGREE | 0x1c75800d0b1054bd | 0x1c75800d0b1054bd | 0x1c75800d0b1054bd | 0x1c75800d0b1054bd |
| llama8b.lm_head.t1 | convert.bf16.pack.b | AGREE | 0xa3f53235cc152516 | 0xa3f53235cc152516 | 0xa3f53235cc152516 | 0xa3f53235cc152516 |
| llama8b.lm_head.t1 | convert.bf16.widen.b | AGREE | 0x63450213e8e12325 | 0x63450213e8e12325 | 0x63450213e8e12325 | 0x63450213e8e12325 |
| llama8b.lm_head.t1 | convert.int8.dequantize.b | AGREE | 0x23115689185e2325 | 0x23115689185e2325 | 0x23115689185e2325 | 0x23115689185e2325 |
| llama8b.lm_head.t1 | inference.bf16f32.v1 | AGREE | 0x23f2e7e17341274a | 0x23f2e7e17341274a | 0x23f2e7e17341274a | 0x23f2e7e17341274a |
| llama8b.lm_head.t1 | inference.int8i32.v1 | AGREE | 0xe0222b2f9de835ed | 0xe0222b2f9de835ed | 0xe0222b2f9de835ed | 0xe0222b2f9de835ed |
| llama8b.lm_head.t1 | inference.int8i32.v1.applechunk | AGREE | not run | 0xe0222b2f9de835ed | 0xe0222b2f9de835ed | not run |
| llama8b.lm_head.t1 | training.bf16f32.v1 | AGREE | 0x23f2e7e17341274a | 0x23f2e7e17341274a | 0x23f2e7e17341274a | 0x23f2e7e17341274a |
| llama8b.lm_head.t1 | training.int8i32.v1 | AGREE | 0xe0222b2f9de835ed | 0xe0222b2f9de835ed | 0xe0222b2f9de835ed | 0xe0222b2f9de835ed |
| llama8b.lm_head.t1 | training.int8i32.v1.applechunk | AGREE | not run | 0xe0222b2f9de835ed | 0xe0222b2f9de835ed | not run |
| llama8b.lm_head.t8 | fp32.v1 | AGREE | 0x3ebab59d0da21444 | 0x3ebab59d0da21444 | 0x3ebab59d0da21444 | 0x3ebab59d0da21444 |
| llama8b.lm_head.t8 | bf16f32.v1.fused | AGREE | 0x82bad7b4e5c4c263 | 0x82bad7b4e5c4c263 | 0x82bad7b4e5c4c263 | 0x82bad7b4e5c4c263 |
| llama8b.lm_head.t8 | bf16f32.v1.widen | AGREE | 0x82bad7b4e5c4c263 | 0x82bad7b4e5c4c263 | 0x82bad7b4e5c4c263 | 0x82bad7b4e5c4c263 |
| llama8b.lm_head.t8 | int8i32.v1.flat | AGREE | 0x8151fd05ff287a5d | 0x8151fd05ff287a5d | 0x8151fd05ff287a5d | 0x8151fd05ff287a5d |
| llama8b.lm_head.t8 | int8i32.v1.mma | AGREE | 0x8151fd05ff287a5d | not run | not run | 0x8151fd05ff287a5d |
| llama8b.lm_head.t8 | int8i32.v1.applechunk | AGREE | not run | 0x8151fd05ff287a5d | 0x8151fd05ff287a5d | not run |
| llama8b.lm_head.t8 | convert.int8.quantize.a | AGREE | 0x443be8ad16d189a3 | 0x443be8ad16d189a3 | 0x443be8ad16d189a3 | 0x443be8ad16d189a3 |
| llama8b.lm_head.t8 | convert.int8.pack.b | AGREE | 0x8e01af3f075298bb | 0x8e01af3f075298bb | 0x8e01af3f075298bb | 0x8e01af3f075298bb |
| llama8b.lm_head.t8 | convert.bf16.pack.b | AGREE | 0xbdac0264c773e6f1 | 0xbdac0264c773e6f1 | 0xbdac0264c773e6f1 | 0xbdac0264c773e6f1 |
| llama8b.lm_head.t8 | convert.bf16.widen.b | AGREE | 0x4e388bd358a22325 | 0x4e388bd358a22325 | 0x4e388bd358a22325 | 0x4e388bd358a22325 |
| llama8b.lm_head.t8 | convert.int8.dequantize.b | AGREE | 0xb8835a7fc2342325 | 0xb8835a7fc2342325 | 0xb8835a7fc2342325 | 0xb8835a7fc2342325 |
| llama8b.lm_head.t8 | inference.bf16f32.v1 | AGREE | 0x82bad7b4e5c4c263 | 0x82bad7b4e5c4c263 | 0x82bad7b4e5c4c263 | 0x82bad7b4e5c4c263 |
| llama8b.lm_head.t8 | inference.int8i32.v1 | AGREE | 0x8151fd05ff287a5d | 0x8151fd05ff287a5d | 0x8151fd05ff287a5d | 0x8151fd05ff287a5d |
| llama8b.lm_head.t8 | inference.int8i32.v1.applechunk | AGREE | not run | 0x8151fd05ff287a5d | 0x8151fd05ff287a5d | not run |
| llama8b.lm_head.t8 | training.bf16f32.v1 | AGREE | 0x82bad7b4e5c4c263 | 0x82bad7b4e5c4c263 | 0x82bad7b4e5c4c263 | 0x82bad7b4e5c4c263 |
| llama8b.lm_head.t8 | training.int8i32.v1 | AGREE | 0x8151fd05ff287a5d | 0x8151fd05ff287a5d | 0x8151fd05ff287a5d | 0x8151fd05ff287a5d |
| llama8b.lm_head.t8 | training.int8i32.v1.applechunk | AGREE | not run | 0x8151fd05ff287a5d | 0x8151fd05ff287a5d | not run |
| llama8b.lm_head.t512 | fp32.v1 | AGREE | 0x84fd143bb625cf70 | 0x84fd143bb625cf70 | 0x84fd143bb625cf70 | 0x84fd143bb625cf70 |
| llama8b.lm_head.t512 | bf16f32.v1.fused | AGREE | 0xa75052ecc1e78b46 | 0xa75052ecc1e78b46 | 0xa75052ecc1e78b46 | 0xa75052ecc1e78b46 |
| llama8b.lm_head.t512 | bf16f32.v1.widen | AGREE | 0xa75052ecc1e78b46 | 0xa75052ecc1e78b46 | 0xa75052ecc1e78b46 | 0xa75052ecc1e78b46 |
| llama8b.lm_head.t512 | int8i32.v1.flat | AGREE | 0xaae4f3d3c69453cd | 0xaae4f3d3c69453cd | 0xaae4f3d3c69453cd | 0xaae4f3d3c69453cd |
| llama8b.lm_head.t512 | int8i32.v1.mma | AGREE | 0xaae4f3d3c69453cd | not run | not run | 0xaae4f3d3c69453cd |
| llama8b.lm_head.t512 | int8i32.v1.applechunk | AGREE | not run | 0xaae4f3d3c69453cd | 0xaae4f3d3c69453cd | not run |
| llama8b.lm_head.t512 | convert.int8.quantize.a | AGREE | 0xfd89ee79d025dda7 | 0xfd89ee79d025dda7 | 0xfd89ee79d025dda7 | 0xfd89ee79d025dda7 |
| llama8b.lm_head.t512 | convert.int8.pack.b | AGREE | 0xcc187b17bc36366d | 0xcc187b17bc36366d | 0xcc187b17bc36366d | 0xcc187b17bc36366d |
| llama8b.lm_head.t512 | convert.bf16.pack.b | AGREE | 0xaabc0c6ba2a73e85 | 0xaabc0c6ba2a73e85 | 0xaabc0c6ba2a73e85 | 0xaabc0c6ba2a73e85 |
| llama8b.lm_head.t512 | convert.bf16.widen.b | AGREE | 0xbd6a89a89fc02325 | 0xbd6a89a89fc02325 | 0xbd6a89a89fc02325 | 0xbd6a89a89fc02325 |
| llama8b.lm_head.t512 | convert.int8.dequantize.b | AGREE | 0xf6a61b0da9142325 | 0xf6a61b0da9142325 | 0xf6a61b0da9142325 | 0xf6a61b0da9142325 |
| llama8b.lm_head.t512 | inference.bf16f32.v1 | AGREE | 0xa75052ecc1e78b46 | 0xa75052ecc1e78b46 | 0xa75052ecc1e78b46 | 0xa75052ecc1e78b46 |
| llama8b.lm_head.t512 | inference.int8i32.v1 | AGREE | 0xaae4f3d3c69453cd | 0xaae4f3d3c69453cd | 0xaae4f3d3c69453cd | 0xaae4f3d3c69453cd |
| llama8b.lm_head.t512 | inference.int8i32.v1.applechunk | AGREE | not run | 0xaae4f3d3c69453cd | 0xaae4f3d3c69453cd | not run |
| llama8b.lm_head.t512 | training.bf16f32.v1 | AGREE | 0xa75052ecc1e78b46 | 0xa75052ecc1e78b46 | 0xa75052ecc1e78b46 | 0xa75052ecc1e78b46 |
| llama8b.lm_head.t512 | training.int8i32.v1 | AGREE | 0xaae4f3d3c69453cd | 0xaae4f3d3c69453cd | 0xaae4f3d3c69453cd | 0xaae4f3d3c69453cd |
| llama8b.lm_head.t512 | training.int8i32.v1.applechunk | AGREE | not run | 0xaae4f3d3c69453cd | 0xaae4f3d3c69453cd | not run |

| arm | shapes | AGREE | DISAGREE | ONE BOX | not comparable |
|---|---:|---:|---:|---:|---:|
| fp32.v1 | 12 | 12 | 0 | 0 | 0 |
| bf16f32.v1.fused | 12 | 12 | 0 | 0 | 0 |
| bf16f32.v1.widen | 12 | 12 | 0 | 0 | 0 |
| int8i32.v1.flat | 12 | 12 | 0 | 0 | 0 |
| int8i32.v1.mma | 12 | 12 | 0 | 0 | 0 |
| int8i32.v1.applechunk | 12 | 12 | 0 | 0 | 0 |
| convert.int8.quantize.a | 12 | 12 | 0 | 0 | 0 |
| convert.int8.pack.b | 12 | 12 | 0 | 0 | 0 |
| convert.bf16.pack.b | 12 | 12 | 0 | 0 | 0 |
| convert.bf16.widen.b | 12 | 12 | 0 | 0 | 0 |
| convert.int8.dequantize.b | 12 | 12 | 0 | 0 | 0 |
| inference.bf16f32.v1 | 12 | 12 | 0 | 0 | 0 |
| inference.int8i32.v1 | 12 | 12 | 0 | 0 | 0 |
| inference.int8i32.v1.applechunk | 12 | 12 | 0 | 0 | 0 |
| training.bf16f32.v1 | 12 | 12 | 0 | 0 | 0 |
| training.int8i32.v1 | 12 | 12 | 0 | 0 | 0 |
| training.int8i32.v1.applechunk | 12 | 12 | 0 | 0 | 0 |

### One profile, every plan, every box

AGREE: every digest any plan of the profile printed on any box at the shape is the same, and at
least two boxes ran a plan of it.

| profile | plans | shapes | AGREE | DISAGREE | ONE BOX |
|---|---|---:|---:|---:|---:|
| fp32.v1 | fp32.v1 | 12 | 12 | 0 | 0 |
| bf16f32.v1 | bf16f32.v1.fused, bf16f32.v1.widen, inference.bf16f32.v1, training.bf16f32.v1 | 12 | 12 | 0 | 0 |
| int8i32.v1 | int8i32.v1.flat, int8i32.v1.mma, int8i32.v1.applechunk, inference.int8i32.v1, inference.int8i32.v1.applechunk, training.int8i32.v1, training.int8i32.v1.applechunk | 12 | 12 | 0 | 0 |

