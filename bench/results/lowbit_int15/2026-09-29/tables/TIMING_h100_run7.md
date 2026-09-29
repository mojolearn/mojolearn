# The fifteen-bit GEMM beside fp32.v1: the complete operation

Profile `mojolearn.identical.gemm.int15i64.v1`. Harness `bench/gemm_int15_price_main.mojo`. Every number is one box's median over the timed calls of one run; `over` is that time over fp32.v1's at the same row in the same run, so above 1 the fifteen-bit operation took longer.

## H100, run 7 (nvc3-0037, b9db1a41b)

Column `nvidia`. The product's plan on this box: the integer matrix unit, four products per k-tile.

### The tuned plan beside the reference unit plan

Reference: one warp one tile, fragments read from device memory, the epilogue in the kernel. Tuned: lane/lowbit-mma-speed's four products with one staging, then the epilogue as a launch of its own. Complete = the operands to planes by the parallel quantizer, then the product (inference: the left operand; training shapes: both).

| row | fp32.v1 ms | reference product ms | over | tuned product ms | over | reference complete ms | over | tuned complete ms | over |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| qkv.t1 (inference) | 0.0694 | 0.2222 | 3.202 | 0.0506 | 0.729 | 0.2246 | 3.236 | 0.0970 | 1.398 |
| qkv.t8 (inference) | 0.1024 | 0.2169 | 2.118 | 0.0429 | 0.419 | 0.2218 | 2.166 | 0.1043 | 1.019 |
| qkv.t512 (both operands converted) | 1.0313 | 1.1955 | 1.159 | 0.2510 | 0.243 | 1.4428 | 1.399 | 0.4839 | 0.469 |
| qkv.t512.bwd_dx (both operands converted) | 1.0176 | 1.2132 | 1.192 | 0.2518 | 0.247 | 1.4739 | 1.448 | 0.5135 | 0.505 |
| qkv.t512.bwd_dw (both operands converted) | 0.8824 | 1.0515 | 1.192 | 0.3476 | 0.394 | 1.1747 | 1.331 | 0.4476 | 0.507 |
| mlp_up.t1 (inference) | 0.1808 | 0.2191 | 1.212 | 0.0808 | 0.447 | 0.2440 | 1.350 | 0.1140 | 0.631 |
| mlp_up.t8 (inference) | 0.2714 | 0.2163 | 0.797 | 0.0846 | 0.312 | 0.2505 | 0.923 | 0.1243 | 0.458 |
| mlp_up.t512 (both operands converted) | 3.5103 | 3.7956 | 1.081 | 0.9515 | 0.271 | 4.3934 | 1.252 | 1.6365 | 0.466 |
| mlp_up.t512.bwd_dx (both operands converted) | 3.3305 | 4.2006 | 1.261 | 0.7996 | 0.240 | 5.3288 | 1.600 | 1.9363 | 0.581 |
| mlp_up.t512.bwd_dw (both operands converted) | 2.9755 | 3.6163 | 1.215 | 1.1596 | 0.390 | 3.7844 | 1.272 | 1.3334 | 0.448 |
| mlp_down.t1 (inference) | 0.1698 | 0.7497 | 4.415 | 0.2007 | 1.182 | 0.7790 | 4.588 | 0.2387 | 1.406 |
| mlp_down.t8 (inference) | 0.2882 | 0.7272 | 2.523 | 0.1986 | 0.689 | 0.7763 | 2.694 | 0.2541 | 0.882 |
| mlp_down.t512 (both operands converted) | 3.4331 | 4.2938 | 1.251 | 0.8032 | 0.234 | 4.9250 | 1.435 | 1.4812 | 0.431 |
| mlp_down.t512.bwd_dx (both operands converted) | 3.5815 | 3.9538 | 1.104 | 0.9731 | 0.272 | 4.6522 | 1.299 | 1.6631 | 0.464 |
| mlp_down.t512.bwd_dw (both operands converted) | 3.1123 | 3.6021 | 1.157 | 1.2438 | 0.400 | 3.7705 | 1.211 | 1.4136 | 0.454 |
| lm_head.t1 (inference) | 1.2945 | 1.0761 | 0.831 | 0.5759 | 0.445 | 1.1325 | 0.875 | 0.6073 | 0.469 |
| lm_head.t8 (inference) | 2.5144 | 1.0998 | 0.437 | 0.6130 | 0.244 | 1.1302 | 0.449 | 0.6553 | 0.261 |
| lm_head.t512 (both operands converted) | 4.2485 | 4.7572 | 1.120 | 1.1437 | 0.269 | 5.6244 | 1.324 | 1.9716 | 0.464 |
| lm_head.t512.bwd_dx (both operands converted) | 3.8194 | refused | refused | refused | refused | refused | refused | refused | refused |
| lm_head.t512.bwd_dw (both operands converted) | 3.3415 | 4.0411 | 1.209 | 1.3063 | 0.391 | 4.2205 | 1.263 | 1.4888 | 0.446 |

#### Inference at the training rows, the tuned plan

| row | fp32.v1 ms | reference complete call ms | over | tuned complete call ms | over |
|---|---:|---:|---:|---:|---:|
| qkv.t512 | 1.0313 | 1.2740 | 1.235 | 0.3152 | 0.306 |
| mlp_up.t512 | 3.5103 | 3.7511 | 1.069 | 1.0182 | 0.290 |
| mlp_down.t512 | 3.4331 | 4.2948 | 1.251 | 0.8880 | 0.259 |
| lm_head.t512 | 4.2485 | 4.5843 | 1.079 | 1.2124 | 0.285 |

#### Three products of one layer, each timed alone and added, the tuned plan

| layer | fp32.v1 ms | reference plan ms | over | tuned plan ms | over |
|---|---:|---:|---:|---:|---:|
| qkv.t512 | 2.9313 | 4.0914 | 1.396 | 1.4450 | 0.493 |
| mlp_up.t512 | 9.8163 | 13.5066 | 1.376 | 4.9062 | 0.500 |
| mlp_down.t512 | 10.1269 | 13.3477 | 1.318 | 4.5579 | 0.450 |
| lm_head.t512 | 11.4094 | refused (k above 65536) | refused | refused (k above 65536) | refused |

### The epilogue fold: fused beside two-launch, one lever

Fused: the sums kernel calls the epilogue at its store, one launch, no sums in device memory. Two-launch: the sums stored (12 bytes per cell), then the epilogue launch.
`fused/two` is the fused time over the two-launch time in this run; below 1 the fold took less time.

| row | fp32.v1 ms | product two-launch ms | product fused ms | fused/two | complete two-launch ms | over fp32.v1 | complete fused ms | over fp32.v1 | fused/two |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| qkv.t1 (inference) | 0.0694 | 0.0552 | 0.0506 | 0.917 | 0.0773 | 1.114 | 0.0970 | 1.398 | 1.255 |
| qkv.t8 (inference) | 0.1024 | 0.0546 | 0.0429 | 0.786 | 0.0840 | 0.820 | 0.1043 | 1.019 | 1.242 |
| qkv.t512 (inference) | 1.0313 | 0.2954 | 0.2510 | 0.850 | 0.3556 | 0.345 | 0.3152 | 0.306 | 0.886 |
| qkv.t512.bwd_dx (both operands) | 1.0176 | 0.2941 | 0.2518 | 0.856 | 0.5528 | 0.543 | 0.5135 | 0.505 | 0.929 |
| qkv.t512.bwd_dw (both operands) | 0.8824 | 0.6326 | 0.3476 | 0.549 | 0.7565 | 0.857 | 0.4476 | 0.507 | 0.592 |
| mlp_up.t1 (inference) | 0.1808 | 0.0862 | 0.0808 | 0.937 | 0.1100 | 0.608 | 0.1140 | 0.631 | 1.036 |
| mlp_up.t8 (inference) | 0.2714 | 0.0881 | 0.0846 | 0.960 | 0.1205 | 0.444 | 0.1243 | 0.458 | 1.032 |
| mlp_up.t512 (inference) | 3.5103 | 1.0709 | 0.9515 | 0.889 | 1.1219 | 0.320 | 1.0182 | 0.290 | 0.908 |
| mlp_up.t512.bwd_dx (both operands) | 3.3305 | 0.8407 | 0.7996 | 0.951 | 1.9686 | 0.591 | 1.9363 | 0.581 | 0.984 |
| mlp_up.t512.bwd_dw (both operands) | 2.9755 | 2.0708 | 1.1596 | 0.560 | 2.2322 | 0.750 | 1.3334 | 0.448 | 0.597 |
| mlp_down.t1 (inference) | 0.1698 | 0.2032 | 0.2007 | 0.988 | 0.2319 | 1.366 | 0.2387 | 1.406 | 1.029 |
| mlp_down.t8 (inference) | 0.2882 | 0.2004 | 0.1986 | 0.991 | 0.2497 | 0.866 | 0.2541 | 0.882 | 1.018 |
| mlp_down.t512 (inference) | 3.4331 | 0.8365 | 0.8032 | 0.960 | 0.9277 | 0.270 | 0.8880 | 0.259 | 0.957 |
| mlp_down.t512.bwd_dx (both operands) | 3.5815 | 1.0934 | 0.9731 | 0.890 | 1.7833 | 0.498 | 1.6631 | 0.464 | 0.933 |
| mlp_down.t512.bwd_dw (both operands) | 3.1123 | 2.1536 | 1.2438 | 0.578 | 2.3240 | 0.747 | 1.4136 | 0.454 | 0.608 |
| lm_head.t1 (inference) | 1.2945 | 0.5811 | 0.5759 | 0.991 | 0.6095 | 0.471 | 0.6073 | 0.469 | 0.996 |
| lm_head.t8 (inference) | 2.5144 | 0.6284 | 0.6130 | 0.975 | 0.6656 | 0.265 | 0.6553 | 0.261 | 0.985 |
| lm_head.t512 (inference) | 4.2485 | 1.2925 | 1.1437 | 0.885 | 1.3676 | 0.322 | 1.2124 | 0.285 | 0.887 |
| lm_head.t512.bwd_dx | 3.8194 | refused | refused | | refused | | refused | | |
| lm_head.t512.bwd_dw (both operands) | 3.3415 | 2.3052 | 1.3063 | 0.567 | 2.4875 | 0.744 | 1.4888 | 0.446 | 0.599 |

### Inference: one call, weights packed once

| row | m x n x k | fp32.v1 ms | product alone ms | over | activations to planes ms | complete call ms | over | complete call, row quantizer ms | over |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| qkv.t1 | 1 x 4096 x 4096 | 0.0694 | 0.2222 (mma) | 3.202 | 0.0258 | 0.2246 (planes) | 3.236 | 0.7228 | 10.415 |
| qkv.t8 | 8 x 4096 x 4096 | 0.1024 | 0.2169 (mma) | 2.118 | 0.0351 | 0.2218 (planes) | 2.166 | 0.8220 | 8.027 |
| qkv.t512 | 512 x 4096 x 4096 | 1.0313 | 1.1955 (mma) | 1.159 | 0.0651 | 1.2740 (planes) | 1.235 | 3.6621 | 3.551 |
| mlp_up.t1 | 1 x 14336 x 4096 | 0.1808 | 0.2191 (mma) | 1.212 | 0.0251 | 0.2440 (planes) | 1.350 | 0.7174 | 3.968 |
| mlp_up.t8 | 8 x 14336 x 4096 | 0.2714 | 0.2163 (mma) | 0.797 | 0.0339 | 0.2505 (planes) | 0.923 | 0.8232 | 3.033 |
| mlp_up.t512 | 512 x 14336 x 4096 | 3.5103 | 3.7956 (mma) | 1.081 | 0.0679 | 3.7511 (planes) | 1.069 | 9.4899 | 2.703 |
| mlp_down.t1 | 1 x 4096 x 14336 | 0.1698 | 0.7497 (mma) | 4.415 | 0.0300 | 0.7790 (planes) | 4.588 | 2.4874 | 14.649 |
| mlp_down.t8 | 8 x 4096 x 14336 | 0.2882 | 0.7272 (mma) | 2.523 | 0.0537 | 0.7763 (planes) | 2.694 | 3.1363 | 10.882 |
| mlp_down.t512 | 512 x 4096 x 14336 | 3.4331 | 4.2938 (mma) | 1.251 | 0.0959 | 4.2948 (planes) | 1.251 | 14.4979 | 4.223 |
| lm_head.t1 | 1 x 128256 x 4096 | 1.2945 | 1.0761 (mma) | 0.831 | 0.0266 | 1.1325 (planes) | 0.875 | 1.5996 | 1.236 |
| lm_head.t8 | 8 x 128256 x 4096 | 2.5144 | 1.0998 (mma) | 0.437 | 0.0413 | 1.1302 (planes) | 0.449 | 1.7892 | 0.712 |
| lm_head.t512 | 512 x 16032 x 4096 (capped) | 4.2485 | 4.7572 (mma) | 1.120 | 0.0730 | 4.5843 (planes) | 1.079 | 10.3915 | 2.446 |

### Training shapes: one product, both operands converted per call

| row | m x n x k | fp32.v1 ms | product alone ms | over | A to planes ms | B to planes ms | complete product ms | over | complete, row quantizer ms | over |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| qkv.t512 | 512 x 4096 x 4096 | 1.0313 | 1.1955 (mma) | 1.159 | 0.0651 | 0.1830 | 1.4428 (planes) | 1.399 | 6.1820 | 5.994 |
| qkv.t512.bwd_dx | 512 x 4096 x 4096 | 1.0176 | 1.2132 (mma) | 1.192 | 0.0640 | 0.2071 | 1.4739 (planes) | 1.448 | 6.3423 | 6.233 |
| qkv.t512.bwd_dw | 4096 x 4096 x 512 | 0.8824 | 1.0515 (mma) | 1.192 | 0.0467 | 0.0469 | 1.1747 (planes) | 1.331 | 1.4540 | 1.648 |
| mlp_up.t512 | 512 x 14336 x 4096 | 3.5103 | 3.7956 (mma) | 1.081 | 0.0679 | 0.6252 | 4.3934 (planes) | 1.252 | 11.5388 | 3.287 |
| mlp_up.t512.bwd_dx | 512 x 4096 x 14336 | 3.3305 | 4.2006 (mma) | 1.261 | 0.0971 | 1.0457 | 5.3288 (planes) | 1.600 | 24.3636 | 7.315 |
| mlp_up.t512.bwd_dw | 14336 x 4096 x 512 | 2.9755 | 3.6163 (mma) | 1.215 | 0.1116 | 0.0480 | 3.7844 (planes) | 1.272 | 4.1126 | 1.382 |
| mlp_down.t512 | 512 x 4096 x 14336 | 3.4331 | 4.2938 (mma) | 1.251 | 0.0959 | 0.5990 | 4.9250 (planes) | 1.435 | 24.6939 | 7.193 |
| mlp_down.t512.bwd_dx | 512 x 14336 x 4096 | 3.5815 | 3.9538 (mma) | 1.104 | 0.0695 | 0.6286 | 4.6522 (planes) | 1.299 | 12.1735 | 3.399 |
| mlp_down.t512.bwd_dw | 4096 x 14336 x 512 | 3.1123 | 3.6021 (mma) | 1.157 | 0.0471 | 0.1117 | 3.7705 (planes) | 1.211 | 4.0965 | 1.316 |
| lm_head.t512 | 512 x 16032 x 4096 (capped) | 4.2485 | 4.7572 (mma) | 1.120 | 0.0730 | 0.7649 | 5.6244 (planes) | 1.324 | 11.7674 | 2.770 |
| lm_head.t512.bwd_dx | 512 x 512 x 128256 (capped) | 3.8194 | refused | refused | refused | refused | refused | refused | refused | refused |
| lm_head.t512.bwd_dw | 16032 x 4096 x 512 (capped) | 3.3415 | 4.0411 (mma) | 1.209 | 0.1233 | 0.0479 | 4.2205 (planes) | 1.263 | 4.5364 | 1.358 |

### Three products of one layer, each timed alone and added

The forward product, the input gradient and the weight gradient of one layer. Each line is the sum of three operations measured one at a time. It is not an integrated training step: no optimizer, no activation, no norm and no memory traffic between the products is in it.

| layer | fp32.v1, three products ms | fifteen-bit, three complete products ms | over |
|---|---:|---:|---:|
| qkv.t512 | 2.9313 | 4.0914 | 1.396 |
| mlp_up.t512 | 9.8163 | 13.5066 | 1.376 |
| mlp_down.t512 | 10.1269 | 13.3477 | 1.318 |
| lm_head.t512 | 11.4094 | refused (lm_head.t512.bwd_dx: k above 65536) | refused |

