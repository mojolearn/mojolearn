# The FAST path against the vendor, one rented NVIDIA box

Parsed from `bench/results/e1g/2026-08-28_040316-nvidia-speed-gemmseq/remote/logs`, 21 arm logs.

Device(s) reported by the arms themselves: NVIDIA H100 80GB HBM3, NVIDIA_H100_80GB_HBM3

`ratio = median(ours) / median(theirs)`. **Above 1.0 means WE ARE SLOWER.**
The warm-up round is excluded from every statistic and printed on its own,
because torch pays an enormous first call and hiding that makes the median
read as the whole story. `min` is printed beside the median so a reader can
see when the box was busy: the further they are apart, the less the row is worth.

## Every arm, as measured

| lane | arm | shape | n | median ms | min ms | max ms | warmup ms | hash |
|---|---|---|---|---|---|---|---|---|
| attention | ours | lane.b2_l4_d32_kv2 | 5 | 0.284 | 0.283 | 0.286 | 0.291 | 84e4782a78ea4da0 |
| attention | ours | llama8b.decode.t1.ctx512 | 5 | 0.842 | 0.840 | 0.848 | 0.884 | 44c4c4d3b9c621c1 |
| attention | ours | llama8b.prefill.t1 | 5 | 0.713 | 0.701 | 0.727 | 0.815 | e547a9bc7ba84b6a |
| attention | ours | llama8b.prefill.t128 | 5 | 1.372 | 1.364 | 1.387 | 1.491 | 73d4e54a8902f73b |
| attention | ours | llama8b.prefill.t512 | 5 | 2.171 | 2.132 | 2.182 | 2.269 | cd59c6437bc1bfe8 |
| attention | ours | llama8b.prefill.t8 | 5 | 1.314 | 1.310 | 1.319 | 1.502 | 2612f26a20ff1d22 |
| attention | torch-gpu-fp32 | lane.b2_l4_d32_kv2 | 5 | 0.398 | 0.385 | 0.437 | 0.387 | e8e95582b48f83cd |
| attention | torch-gpu-fp32 | llama8b.decode.t1.ctx512 | 5 | 0.379 | 0.371 | 0.395 | 0.370 | 07cb383ee9544db3 |
| attention | torch-gpu-fp32 | llama8b.prefill.t1 | 5 | 0.337 | 0.331 | 0.374 | 0.343 | dc7893ab5281d902 |
| attention | torch-gpu-fp32 | llama8b.prefill.t128 | 5 | 0.590 | 0.542 | 0.644 | 0.513 | - |
| attention | torch-gpu-fp32 | llama8b.prefill.t512 | 5 | 1.425 | 1.386 | 1.442 | 1.391 | - |
| attention | torch-gpu-fp32 | llama8b.prefill.t8 | 5 | 0.448 | 0.402 | 0.466 | 0.408 | 24dff6f6348f8e74 |
| attention | torch-gpu-sdpa-efficient-fp32 | lane.b2_l4_d32_kv2 | 5 | 0.295 | 0.285 | 0.318 | 0.289 | c89e8c280bb9a932 |
| attention | torch-gpu-sdpa-efficient-fp32 | llama8b.decode.t1.ctx512 | 5 | 0.326 | 0.326 | 0.343 | 0.331 | 92c858cfb8cbac82 |
| attention | torch-gpu-sdpa-efficient-fp32 | llama8b.prefill.t1 | 5 | 0.261 | 0.249 | 0.272 | 0.248 | 0108982a62078f7c |
| attention | torch-gpu-sdpa-efficient-fp32 | llama8b.prefill.t128 | 5 | 0.433 | 0.393 | 0.450 | 0.392 | - |
| attention | torch-gpu-sdpa-efficient-fp32 | llama8b.prefill.t512 | 5 | 1.190 | 1.162 | 1.193 | 1.166 | - |
| attention | torch-gpu-sdpa-efficient-fp32 | llama8b.prefill.t8 | 5 | 0.311 | 0.298 | 0.349 | 0.317 | f884bd08cd2be534 |
| attention | torch-gpu-sdpa-math-fp32 | lane.b2_l4_d32_kv2 | 5 | 0.368 | 0.356 | 0.396 | 0.363 | e01cb73a157448ed |
| attention | torch-gpu-sdpa-math-fp32 | llama8b.decode.t1.ctx512 | 5 | 0.312 | 0.299 | 0.316 | 0.304 | 61fe91daf1d8f3d7 |
| attention | torch-gpu-sdpa-math-fp32 | llama8b.prefill.t1 | 5 | 0.282 | 0.275 | 0.292 | 0.283 | dc7893ab5281d902 |
| attention | torch-gpu-sdpa-math-fp32 | llama8b.prefill.t128 | 5 | 0.511 | 0.411 | 0.532 | 0.410 | - |
| attention | torch-gpu-sdpa-math-fp32 | llama8b.prefill.t512 | 5 | 1.272 | 1.250 | 1.273 | 1.252 | - |
| attention | torch-gpu-sdpa-math-fp32 | llama8b.prefill.t8 | 5 | 0.394 | 0.340 | 0.446 | 0.353 | 9bba19a49b80a5ff |
| attention | torch-gpu-tf32 | lane.b2_l4_d32_kv2 | 5 | 0.374 | 0.354 | 0.379 | 0.356 | 550727a9d0db290a |
| attention | torch-gpu-tf32 | llama8b.decode.t1.ctx512 | 5 | 0.367 | 0.361 | 0.373 | 0.382 | 88c632658f1d8421 |
| attention | torch-gpu-tf32 | llama8b.prefill.t1 | 5 | 0.349 | 0.326 | 0.380 | 0.320 | dc7893ab5281d902 |
| attention | torch-gpu-tf32 | llama8b.prefill.t128 | 5 | 0.535 | 0.420 | 0.548 | 0.435 | - |
| attention | torch-gpu-tf32 | llama8b.prefill.t512 | 5 | 0.642 | 0.559 | 0.680 | 0.584 | - |
| attention | torch-gpu-tf32 | llama8b.prefill.t8 | 5 | 0.463 | 0.392 | 0.527 | 0.400 | df95b9a56d17ad9a |
| gemm | cublas-fp32 | gram.128sq.x100003 | 5 | 0.093 | 0.092 | 0.094 | 0.097 | - |
| gemm | cublas-fp32 | gram.32x32x1M | 5 | 0.237 | 0.236 | 0.242 | 0.240 | - |
| gemm | cublas-fp32 | gram.32x32x64K | 5 | 0.037 | 0.037 | 0.047 | 0.040 | - |
| gemm | cublas-fp32 | kmeans.dist.4096x64x64 | 5 | 0.020 | 0.019 | 0.020 | 0.021 | - |
| gemm | cublas-fp32 | llama8b.lm_head.t1 | 5 | 0.675 | 0.672 | 0.883 | 0.682 | - |
| gemm | cublas-fp32 | llama8b.lm_head.t512 | 5 | 10.727 | 10.717 | 10.746 | 10.740 | - |
| gemm | cublas-fp32 | llama8b.lm_head.t8 | 5 | 1.147 | 1.145 | 1.154 | 1.150 | - |
| gemm | cublas-fp32 | llama8b.mlp_down.t1 | 5 | 0.092 | 0.091 | 0.108 | 0.096 | - |
| gemm | cublas-fp32 | llama8b.mlp_down.t512 | 5 | 1.176 | 1.172 | 1.180 | 1.188 | - |
| gemm | cublas-fp32 | llama8b.mlp_down.t8 | 5 | 0.198 | 0.196 | 0.203 | 0.202 | - |
| gemm | cublas-fp32 | llama8b.mlp_up.t1 | 5 | 0.092 | 0.090 | 0.092 | 0.094 | - |
| gemm | cublas-fp32 | llama8b.mlp_up.t512 | 5 | 1.371 | 1.369 | 1.374 | 1.374 | - |
| gemm | cublas-fp32 | llama8b.mlp_up.t8 | 5 | 0.160 | 0.155 | 0.161 | 0.162 | - |
| gemm | cublas-fp32 | llama8b.qkv.t1 | 5 | 0.039 | 0.038 | 0.040 | 0.042 | - |
| gemm | cublas-fp32 | llama8b.qkv.t512 | 5 | 0.369 | 0.366 | 0.374 | 0.369 | - |
| gemm | cublas-fp32 | llama8b.qkv.t8 | 5 | 0.058 | 0.057 | 0.059 | 0.077 | - |
| gemm | cublas-fp32 | ols.predict.gemv.64Kx16 | 5 | 0.018 | 0.017 | 0.018 | 0.019 | - |
| gemm | cublas-fp32 | ols.step1.16x16x64K | 5 | 0.036 | 0.035 | 0.036 | 0.038 | - |
| gemm | cublas-fp32 | pca.transform.8192x4x4 | 5 | 0.019 | 0.018 | 0.022 | 0.024 | - |
| gemm | cublas-fp32 | pca.transform.wide.8192x64x128 | 5 | 0.023 | 0.022 | 0.035 | 0.025 | - |
| gemm | cublas-tf32 | gram.128sq.x100003 | 5 | 0.058 | 0.056 | 0.059 | 0.063 | - |
| gemm | cublas-tf32 | gram.32x32x1M | 5 | 0.108 | 0.108 | 0.109 | 0.114 | - |
| gemm | cublas-tf32 | gram.32x32x64K | 5 | 0.027 | 0.027 | 0.033 | 0.028 | - |
| gemm | cublas-tf32 | kmeans.dist.4096x64x64 | 5 | 0.017 | 0.016 | 0.017 | 0.035 | - |
| gemm | cublas-tf32 | llama8b.lm_head.t1 | 5 | 0.673 | 0.672 | 0.674 | 0.680 | - |
| gemm | cublas-tf32 | llama8b.lm_head.t512 | 5 | 1.357 | 1.357 | 1.361 | 1.370 | - |
| gemm | cublas-tf32 | llama8b.lm_head.t8 | 5 | 0.751 | 0.748 | 0.753 | 0.755 | - |
| gemm | cublas-tf32 | llama8b.mlp_down.t1 | 5 | 0.094 | 0.093 | 0.095 | 0.097 | - |
| gemm | cublas-tf32 | llama8b.mlp_down.t512 | 5 | 0.211 | 0.209 | 0.211 | 0.219 | - |
| gemm | cublas-tf32 | llama8b.mlp_down.t8 | 5 | 0.103 | 0.103 | 0.104 | 0.108 | - |
| gemm | cublas-tf32 | llama8b.mlp_up.t1 | 5 | 0.093 | 0.091 | 0.095 | 0.095 | - |
| gemm | cublas-tf32 | llama8b.mlp_up.t512 | 5 | 0.185 | 0.184 | 0.188 | 0.189 | - |
| gemm | cublas-tf32 | llama8b.mlp_up.t8 | 5 | 0.105 | 0.105 | 0.119 | 0.107 | - |
| gemm | cublas-tf32 | llama8b.qkv.t1 | 5 | 0.040 | 0.039 | 0.041 | 0.042 | - |
| gemm | cublas-tf32 | llama8b.qkv.t512 | 5 | 0.066 | 0.065 | 0.067 | 0.069 | - |
| gemm | cublas-tf32 | llama8b.qkv.t8 | 5 | 0.044 | 0.044 | 0.045 | 0.048 | - |
| gemm | cublas-tf32 | ols.predict.gemv.64Kx16 | 5 | 0.017 | 0.016 | 0.018 | 0.029 | - |
| gemm | cublas-tf32 | ols.step1.16x16x64K | 5 | 0.027 | 0.027 | 0.029 | 0.034 | - |
| gemm | cublas-tf32 | pca.transform.8192x4x4 | 5 | 0.018 | 0.017 | 0.019 | 0.034 | - |
| gemm | cublas-tf32 | pca.transform.wide.8192x64x128 | 5 | 0.019 | 0.019 | 0.021 | 0.022 | - |
| gemm | ours | gram.128sq.x100003 | 5 | 0.142 | 0.141 | 0.143 | 0.208 | 4ffd0f74215ccd72 |
| gemm | ours | gram.32x32x1M | 5 | 0.271 | 0.271 | 0.278 | 101.079 | 569e2f538c8dba79 |
| gemm | ours | gram.32x32x64K | 5 | 0.047 | 0.047 | 0.049 | 0.071 | 73c948814939c999 |
| gemm | ours | kmeans.dist.4096x64x64 | 5 | 0.025 | 0.024 | 0.025 | 0.040 | 852f3968ef903856 |
| gemm | ours | llama8b.lm_head.t512 | 5 | 1.407 | 1.385 | 1.545 | 1.578 | 1703b99d86de912f |
| gemm | ours | llama8b.lm_head.t8 | 5 | 0.775 | 0.774 | 0.808 | 1.358 | 177ba2ff58798563 |
| gemm | ours | llama8b.mlp_down.t1 | 5 | 0.085 | 0.084 | 0.085 | 0.097 | 53d4682d1aa49274 |
| gemm | ours | llama8b.mlp_down.t512 | 5 | 0.232 | 0.228 | 0.245 | 0.292 | ab59d1f4f791549a |
| gemm | ours | llama8b.mlp_down.t8 | 5 | 0.110 | 0.109 | 0.112 | 0.196 | 84f86bdbb850a723 |
| gemm | ours | llama8b.mlp_up.t1 | 5 | 0.083 | 0.083 | 0.085 | 0.097 | 36a00f23caf25b37 |
| gemm | ours | llama8b.mlp_up.t512 | 5 | 0.211 | 0.208 | 0.219 | 0.478 | 67c71ad05c2f0f0b |
| gemm | ours | llama8b.mlp_up.t8 | 5 | 0.115 | 0.114 | 0.121 | 0.226 | a858a777c88ebee7 |
| gemm | ours | llama8b.qkv.t1 | 5 | 0.031 | 0.030 | 0.031 | 0.103 | e1ba240d1ace7e31 |
| gemm | ours | llama8b.qkv.t512 | 5 | 0.090 | 0.084 | 0.105 | 0.513 | 8cfe7f6c1b5804e6 |
| gemm | ours | llama8b.qkv.t8 | 5 | 0.054 | 0.053 | 0.075 | 15.567 | 6abaa8919461da1b |
| gemm | ours | ols.predict.gemv.64Kx16 | 5 | 0.016 | 0.015 | 0.016 | 0.081 | e17121fd170e87d6 |
| gemm | ours | ols.step1.16x16x64K | 5 | 0.045 | 0.044 | 0.046 | 0.065 | e208309014b6097b |
| gemm | ours | pca.transform.8192x4x4 | 5 | 0.021 | 0.021 | 0.022 | 0.029 | b9330556782ece72 |
| gemm | ours | pca.transform.wide.8192x64x128 | 5 | 0.029 | 0.028 | 0.030 | 0.048 | 9c102c3290acdf09 |
| mamba | ours | lane.b2_l4_d8 | 5 | 0.183 | 0.180 | 0.186 | 0.201 | 094451532d3a52e9 |
| mamba | ours | mamba130m.prefill.t1 | 5 | 0.237 | 0.235 | 0.266 | 0.236 | fad7b323c6b9145c |
| mamba | ours | mamba130m.prefill.t128 | 5 | 1.042 | 1.008 | 1.046 | 1.069 | 53f74bccbeeb5197 |
| mamba | ours | mamba130m.prefill.t512 | 5 | 2.470 | 2.311 | 2.533 | 2.475 | 756d0f1dabce09ff |
| mamba | ours | mamba130m.prefill.t8 | 5 | 0.350 | 0.344 | 0.416 | 0.401 | 2fde47a98e094374 |
| mamba | torch-ref-scan-gpu | lane.b2_l4_d8 | 5 | 0.566 | 0.541 | 0.635 | 0.558 | 4eefaa4a619c7219 |
| mamba | torch-ref-scan-gpu | mamba130m.prefill.t1 | 5 | 0.414 | 0.408 | 0.418 | 0.414 | 539eea7de7dd42a6 |
| mamba | torch-ref-scan-gpu | mamba130m.prefill.t128 | 5 | 5.691 | 5.652 | 5.723 | 6.403 | - |
| mamba | torch-ref-scan-gpu | mamba130m.prefill.t512 | 5 | 21.236 | 21.204 | 21.374 | 21.384 | - |
| mamba | torch-ref-scan-gpu | mamba130m.prefill.t8 | 5 | 0.754 | 0.709 | 0.789 | 0.736 | 5cbeb28b0ccd30cb |
| mamba | torch-ref-scan-gpu-tf32 | lane.b2_l4_d8 | 5 | 0.545 | 0.531 | 0.583 | 0.533 | 3621694635e65892 |
| mamba | torch-ref-scan-gpu-tf32 | mamba130m.prefill.t1 | 5 | 0.399 | 0.389 | 0.429 | 0.381 | 539eea7de7dd42a6 |
| mamba | torch-ref-scan-gpu-tf32 | mamba130m.prefill.t128 | 5 | 5.730 | 5.648 | 5.816 | 5.637 | - |
| mamba | torch-ref-scan-gpu-tf32 | mamba130m.prefill.t512 | 5 | 21.276 | 21.244 | 21.435 | 22.821 | - |
| mamba | torch-ref-scan-gpu-tf32 | mamba130m.prefill.t8 | 5 | 0.722 | 0.681 | 0.773 | 0.686 | 178fe924d53298f4 |
| mlp | ours | lane.b2_l4_d32_kv2 | 5 | 0.071 | 0.069 | 0.074 | 0.079 | 69deede6c74a6bb6 |
| mlp | ours | llama8b.decode.t1.ctx512 | 5 | 0.257 | 0.256 | 0.257 | 0.270 | f4030863b88519de |
| mlp | ours | llama8b.prefill.t1 | 5 | 0.255 | 0.254 | 0.256 | 0.283 | 7fdac9fd55be1c0a |
| mlp | ours | llama8b.prefill.t128 | 5 | 0.385 | 0.366 | 0.430 | 0.492 | 2a939d9c97455c5d |
| mlp | ours | llama8b.prefill.t512 | 5 | 0.684 | 0.673 | 0.702 | 0.902 | 1fe0d8abecbf3d6e |
| mlp | ours | llama8b.prefill.t8 | 5 | 0.338 | 0.334 | 0.342 | 0.530 | 451ea95692850783 |
| mlp | torch-gpu-fp32 | lane.b2_l4_d32_kv2 | 5 | 0.101 | 0.097 | 0.113 | 0.116 | 01d13b190e30cb95 |
| mlp | torch-gpu-fp32 | llama8b.decode.t1.ctx512 | 5 | 0.272 | 0.266 | 0.276 | 0.275 | a6524867d4a13e6e |
| mlp | torch-gpu-fp32 | llama8b.prefill.t1 | 5 | 0.273 | 0.271 | 0.277 | 0.279 | c23b389714c357e7 |
| mlp | torch-gpu-fp32 | llama8b.prefill.t128 | 5 | 1.076 | 1.059 | 1.089 | 1.070 | - |
| mlp | torch-gpu-fp32 | llama8b.prefill.t512 | 5 | 4.046 | 4.007 | 4.054 | 4.020 | - |
| mlp | torch-gpu-fp32 | llama8b.prefill.t8 | 5 | 0.513 | 0.498 | 0.520 | 0.501 | 5f4ca2e94c343365 |
| mlp | torch-gpu-tf32 | lane.b2_l4_d32_kv2 | 5 | 0.092 | 0.088 | 0.106 | 0.092 | c286305c554c841d |
| mlp | torch-gpu-tf32 | llama8b.decode.t1.ctx512 | 5 | 0.270 | 0.270 | 0.273 | 0.274 | c1777403df51bc26 |
| mlp | torch-gpu-tf32 | llama8b.prefill.t1 | 5 | 0.271 | 0.270 | 0.271 | 0.271 | c23b389714c357e7 |
| mlp | torch-gpu-tf32 | llama8b.prefill.t128 | 5 | 0.357 | 0.329 | 0.365 | 0.330 | - |
| mlp | torch-gpu-tf32 | llama8b.prefill.t512 | 5 | 0.716 | 0.688 | 0.723 | 0.690 | - |
| mlp | torch-gpu-tf32 | llama8b.prefill.t8 | 5 | 0.306 | 0.286 | 0.333 | 0.298 | 2aff66ca02f232bc |
| rmsnorm | ours | lane.b2_l4_d32_kv2 | 5 | 0.010 | 0.009 | 0.010 | 0.011 | b058a2f6243d396f |
| rmsnorm | ours | llama8b.decode.t1.ctx512 | 5 | 0.282 | 0.282 | 0.283 | 0.398 | 7a1c1841fd45ef80 |
| rmsnorm | ours | llama8b.prefill.t1 | 5 | 0.281 | 0.280 | 0.281 | 0.394 | c65e8b5c4dc7c77f |
| rmsnorm | ours | llama8b.prefill.t128 | 5 | 1.052 | 1.050 | 1.053 | 1.171 | 7908df53e2986305 |
| rmsnorm | ours | llama8b.prefill.t512 | 5 | 1.062 | 1.057 | 1.064 | 1.171 | b7140c9c99fa980b |
| rmsnorm | ours | llama8b.prefill.t8 | 5 | 0.345 | 0.345 | 0.345 | 0.477 | 32bd7d2d012666b5 |
| rmsnorm | torch-gpu-fp32 | lane.b2_l4_d32_kv2 | 5 | 0.053 | 0.046 | 0.057 | 0.053 | 852ff02726038c82 |
| rmsnorm | torch-gpu-fp32 | llama8b.decode.t1.ctx512 | 5 | 0.044 | 0.042 | 0.050 | 0.048 | 409660f2f0359908 |
| rmsnorm | torch-gpu-fp32 | llama8b.prefill.t1 | 5 | 0.049 | 0.044 | 0.050 | 0.050 | 4635a852c2627e50 |
| rmsnorm | torch-gpu-fp32 | llama8b.prefill.t128 | 5 | 0.067 | 0.043 | 0.099 | 0.048 | - |
| rmsnorm | torch-gpu-fp32 | llama8b.prefill.t512 | 5 | 0.084 | 0.048 | 0.110 | 0.051 | - |
| rmsnorm | torch-gpu-fp32 | llama8b.prefill.t8 | 5 | 0.053 | 0.045 | 0.059 | 0.052 | 3dc8fbbfbb167c82 |
| rmsnorm | torch-gpu-tf32 | lane.b2_l4_d32_kv2 | 5 | 0.045 | 0.043 | 0.052 | 0.045 | 852ff02726038c82 |
| rmsnorm | torch-gpu-tf32 | llama8b.decode.t1.ctx512 | 5 | 0.045 | 0.042 | 0.056 | 0.043 | 409660f2f0359908 |
| rmsnorm | torch-gpu-tf32 | llama8b.prefill.t1 | 5 | 0.046 | 0.042 | 0.047 | 0.043 | 4635a852c2627e50 |
| rmsnorm | torch-gpu-tf32 | llama8b.prefill.t128 | 5 | 0.062 | 0.056 | 0.068 | 0.044 | - |
| rmsnorm | torch-gpu-tf32 | llama8b.prefill.t512 | 5 | 0.068 | 0.049 | 0.075 | 0.053 | - |
| rmsnorm | torch-gpu-tf32 | llama8b.prefill.t8 | 5 | 0.048 | 0.042 | 0.053 | 0.043 | 3dc8fbbfbb167c82 |
| selective_scan | ours | lane.b2_l4_d8 | 5 | 0.011 | 0.011 | 0.011 | 0.014 | fb645becd4c7da53 |
| selective_scan | ours | mamba130m.prefill.t1 | 5 | 0.009 | 0.009 | 0.010 | 0.010 | 835f212cd5cc9adf |
| selective_scan | ours | mamba130m.prefill.t128 | 5 | 0.100 | 0.100 | 0.101 | 0.150 | f7c4e81255e852ef |
| selective_scan | ours | mamba130m.prefill.t512 | 5 | 0.375 | 0.374 | 0.376 | 0.541 | 1e23b8c3508db4a8 |
| selective_scan | ours | mamba130m.prefill.t8 | 5 | 0.014 | 0.014 | 0.015 | 0.017 | 3e38a1b6f8e11373 |
| selective_scan | torch-ref-scan-gpu | lane.b2_l4_d8 | 5 | 0.287 | 0.273 | 0.322 | 0.271 | 1109a9d621ae1ff1 |
| selective_scan | torch-ref-scan-gpu | mamba130m.prefill.t1 | 5 | 0.154 | 0.145 | 0.168 | 0.143 | 6ab7df9710724e0a |
| selective_scan | torch-ref-scan-gpu | mamba130m.prefill.t128 | 5 | 5.434 | 5.393 | 5.488 | 5.400 | - |
| selective_scan | torch-ref-scan-gpu | mamba130m.prefill.t512 | 5 | 21.092 | 21.029 | 21.127 | 21.203 | - |
| selective_scan | torch-ref-scan-gpu | mamba130m.prefill.t8 | 5 | 0.464 | 0.438 | 0.491 | 0.457 | 12db7962951ed422 |
| selective_scan | torch-ref-scan-gpu-tf32 | lane.b2_l4_d8 | 5 | 0.268 | 0.266 | 0.288 | 0.270 | f3cb0e7ca459e9e8 |
| selective_scan | torch-ref-scan-gpu-tf32 | mamba130m.prefill.t1 | 5 | 0.142 | 0.139 | 0.147 | 0.145 | 6ab7df9710724e0a |
| selective_scan | torch-ref-scan-gpu-tf32 | mamba130m.prefill.t128 | 5 | 5.468 | 5.366 | 5.479 | 5.362 | - |
| selective_scan | torch-ref-scan-gpu-tf32 | mamba130m.prefill.t512 | 5 | 21.058 | 20.955 | 21.230 | 22.796 | - |
| selective_scan | torch-ref-scan-gpu-tf32 | mamba130m.prefill.t8 | 5 | 0.457 | 0.426 | 0.471 | 0.437 | 3c9a3865311687c2 |
| transformer | ours | lane.b2_l4_d32_kv2 | 5 | 0.396 | 0.394 | 0.401 | 0.409 | 0b2376058aabee05 |
| transformer | ours | llama8b.decode.t1.ctx512 | 5 | 3.594 | 3.582 | 3.612 | 3.680 | aec60cdba019c822 |
| transformer | ours | llama8b.prefill.t1 | 5 | 1.690 | 1.685 | 1.705 | 1.759 | 79188d142b226eaa |
| transformer | ours | llama8b.prefill.t128 | 5 | 5.148 | 4.919 | 10.328 | 5.260 | 2e7fe6554579f94e |
| transformer | ours | llama8b.prefill.t512 | 5 | 8.880 | 8.697 | 10.038 | 18.144 | 3f5bdd0262ad8c75 |
| transformer | ours | llama8b.prefill.t8 | 5 | 2.577 | 2.564 | 2.590 | 2.714 | e4318a399c9c205f |
| transformer | torch-gpu-fp32 | lane.b2_l4_d32_kv2 | 5 | 0.540 | 0.522 | 0.593 | 0.557 | d609221d7268e4c3 |
| transformer | torch-gpu-fp32 | llama8b.decode.t1.ctx512 | 5 | 0.690 | 0.676 | 0.691 | 0.677 | f62a023115027e31 |
| transformer | torch-gpu-fp32 | llama8b.prefill.t1 | 5 | 0.723 | 0.668 | 0.726 | 0.788 | bd44c6cd02b6e51c |
| transformer | torch-gpu-fp32 | llama8b.prefill.t128 | 5 | 1.701 | 1.640 | 1.756 | 1.620 | - |
| transformer | torch-gpu-fp32 | llama8b.prefill.t512 | 5 | 5.542 | 5.454 | 5.552 | 5.459 | - |
| transformer | torch-gpu-fp32 | llama8b.prefill.t8 | 5 | 1.004 | 0.976 | 1.025 | 0.988 | 8187b0382ae2f891 |
| transformer | torch-gpu-sdpa-efficient-fp32 | lane.b2_l4_d32_kv2 | 5 | 0.389 | 0.378 | 0.395 | 0.389 | ffbca2badcffcd66 |
| transformer | torch-gpu-sdpa-efficient-fp32 | llama8b.decode.t1.ctx512 | 5 | 0.695 | 0.630 | 0.790 | 0.648 | d8fbb0fbeec0b7dd |
| transformer | torch-gpu-sdpa-efficient-fp32 | llama8b.prefill.t1 | 5 | 0.588 | 0.570 | 0.592 | 0.571 | 39aeb70c2dd56230 |
| transformer | torch-gpu-sdpa-efficient-fp32 | llama8b.prefill.t128 | 5 | 1.543 | 1.503 | 1.564 | 1.504 | - |
| transformer | torch-gpu-sdpa-efficient-fp32 | llama8b.prefill.t512 | 5 | 5.330 | 5.243 | 5.818 | 5.250 | - |
| transformer | torch-gpu-sdpa-efficient-fp32 | llama8b.prefill.t8 | 5 | 0.854 | 0.825 | 0.859 | 0.824 | b737e94f42d65b66 |
| transformer | torch-gpu-sdpa-math-fp32 | lane.b2_l4_d32_kv2 | 5 | 0.453 | 0.451 | 0.463 | 0.449 | 2e29daa0f9ab306a |
| transformer | torch-gpu-sdpa-math-fp32 | llama8b.decode.t1.ctx512 | 5 | 0.623 | 0.614 | 0.634 | 0.651 | 848e21d1df64d87c |
| transformer | torch-gpu-sdpa-math-fp32 | llama8b.prefill.t1 | 5 | 0.605 | 0.587 | 0.615 | 0.602 | bd44c6cd02b6e51c |
| transformer | torch-gpu-sdpa-math-fp32 | llama8b.prefill.t128 | 5 | 1.602 | 1.528 | 1.607 | 1.518 | - |
| transformer | torch-gpu-sdpa-math-fp32 | llama8b.prefill.t512 | 5 | 5.371 | 5.322 | 5.383 | 5.331 | - |
| transformer | torch-gpu-sdpa-math-fp32 | llama8b.prefill.t8 | 5 | 0.947 | 0.893 | 0.951 | 0.880 | 81a4eb2ae5d6bc88 |
| transformer | torch-gpu-tf32 | lane.b2_l4_d32_kv2 | 5 | 0.504 | 0.495 | 0.514 | 0.513 | 29818408b2158e13 |
| transformer | torch-gpu-tf32 | llama8b.decode.t1.ctx512 | 5 | 0.680 | 0.667 | 0.685 | 0.676 | 1f71bd5f486361ec |
| transformer | torch-gpu-tf32 | llama8b.prefill.t1 | 5 | 0.666 | 0.634 | 0.724 | 0.633 | bd44c6cd02b6e51c |
| transformer | torch-gpu-tf32 | llama8b.prefill.t128 | 5 | 0.897 | 0.824 | 0.913 | 0.799 | - |
| transformer | torch-gpu-tf32 | llama8b.prefill.t512 | 5 | 1.380 | 1.332 | 1.419 | 1.329 | - |
| transformer | torch-gpu-tf32 | llama8b.prefill.t8 | 5 | 0.803 | 0.754 | 0.825 | 0.754 | 6b633a44dbf90a85 |

## Ours against each opponent

| lane | shape | opponent | ours ms | theirs ms | ratio ours/theirs | verdict |
|---|---|---|---|---|---|---|
| attention | lane.b2_l4_d32_kv2 | torch-gpu-fp32 | 0.284 | 0.398 | 0.71 | we are 1.40x FASTER |
| attention | lane.b2_l4_d32_kv2 | torch-gpu-sdpa-efficient-fp32 | 0.284 | 0.295 | 0.96 | we are 1.04x FASTER |
| attention | lane.b2_l4_d32_kv2 | torch-gpu-sdpa-math-fp32 | 0.284 | 0.368 | 0.77 | we are 1.30x FASTER |
| attention | lane.b2_l4_d32_kv2 | torch-gpu-tf32 | 0.284 | 0.374 | 0.76 | we are 1.32x FASTER |
| attention | llama8b.decode.t1.ctx512 | torch-gpu-fp32 | 0.842 | 0.379 | 2.22 | we are 2.22x SLOWER |
| attention | llama8b.decode.t1.ctx512 | torch-gpu-sdpa-efficient-fp32 | 0.842 | 0.326 | 2.58 | we are 2.58x SLOWER |
| attention | llama8b.decode.t1.ctx512 | torch-gpu-sdpa-math-fp32 | 0.842 | 0.312 | 2.70 | we are 2.70x SLOWER |
| attention | llama8b.decode.t1.ctx512 | torch-gpu-tf32 | 0.842 | 0.367 | 2.29 | we are 2.29x SLOWER |
| attention | llama8b.prefill.t1 | torch-gpu-fp32 | 0.713 | 0.337 | 2.12 | we are 2.12x SLOWER |
| attention | llama8b.prefill.t1 | torch-gpu-sdpa-efficient-fp32 | 0.713 | 0.261 | 2.73 | we are 2.73x SLOWER |
| attention | llama8b.prefill.t1 | torch-gpu-sdpa-math-fp32 | 0.713 | 0.282 | 2.53 | we are 2.53x SLOWER |
| attention | llama8b.prefill.t1 | torch-gpu-tf32 | 0.713 | 0.349 | 2.04 | we are 2.04x SLOWER |
| attention | llama8b.prefill.t128 | torch-gpu-fp32 | 1.372 | 0.590 | 2.33 | we are 2.33x SLOWER |
| attention | llama8b.prefill.t128 | torch-gpu-sdpa-efficient-fp32 | 1.372 | 0.433 | 3.17 | we are 3.17x SLOWER |
| attention | llama8b.prefill.t128 | torch-gpu-sdpa-math-fp32 | 1.372 | 0.511 | 2.68 | we are 2.68x SLOWER |
| attention | llama8b.prefill.t128 | torch-gpu-tf32 | 1.372 | 0.535 | 2.56 | we are 2.56x SLOWER |
| attention | llama8b.prefill.t512 | torch-gpu-fp32 | 2.171 | 1.425 | 1.52 | we are 1.52x SLOWER |
| attention | llama8b.prefill.t512 | torch-gpu-sdpa-efficient-fp32 | 2.171 | 1.190 | 1.82 | we are 1.82x SLOWER |
| attention | llama8b.prefill.t512 | torch-gpu-sdpa-math-fp32 | 2.171 | 1.272 | 1.71 | we are 1.71x SLOWER |
| attention | llama8b.prefill.t512 | torch-gpu-tf32 | 2.171 | 0.642 | 3.38 | we are 3.38x SLOWER |
| attention | llama8b.prefill.t8 | torch-gpu-fp32 | 1.314 | 0.448 | 2.93 | we are 2.93x SLOWER |
| attention | llama8b.prefill.t8 | torch-gpu-sdpa-efficient-fp32 | 1.314 | 0.311 | 4.23 | we are 4.23x SLOWER |
| attention | llama8b.prefill.t8 | torch-gpu-sdpa-math-fp32 | 1.314 | 0.394 | 3.34 | we are 3.34x SLOWER |
| attention | llama8b.prefill.t8 | torch-gpu-tf32 | 1.314 | 0.463 | 2.84 | we are 2.84x SLOWER |
| gemm | gram.128sq.x100003 | cublas-fp32 | 0.142 | 0.093 | 1.53 | we are 1.53x SLOWER |
| gemm | gram.128sq.x100003 | cublas-tf32 | 0.142 | 0.058 | 2.47 | we are 2.47x SLOWER |
| gemm | gram.32x32x1M | cublas-fp32 | 0.271 | 0.237 | 1.14 | we are 1.14x SLOWER |
| gemm | gram.32x32x1M | cublas-tf32 | 0.271 | 0.108 | 2.50 | we are 2.50x SLOWER |
| gemm | gram.32x32x64K | cublas-fp32 | 0.047 | 0.037 | 1.27 | we are 1.27x SLOWER |
| gemm | gram.32x32x64K | cublas-tf32 | 0.047 | 0.027 | 1.74 | we are 1.74x SLOWER |
| gemm | kmeans.dist.4096x64x64 | cublas-fp32 | 0.025 | 0.020 | 1.25 | we are 1.25x SLOWER |
| gemm | kmeans.dist.4096x64x64 | cublas-tf32 | 0.025 | 0.017 | 1.45 | we are 1.45x SLOWER |
| gemm | llama8b.lm_head.t512 | cublas-fp32 | 1.407 | 10.727 | 0.13 | we are 7.63x FASTER |
| gemm | llama8b.lm_head.t512 | cublas-tf32 | 1.407 | 1.357 | 1.04 | we are 1.04x SLOWER |
| gemm | llama8b.lm_head.t8 | cublas-fp32 | 0.775 | 1.147 | 0.68 | we are 1.48x FASTER |
| gemm | llama8b.lm_head.t8 | cublas-tf32 | 0.775 | 0.751 | 1.03 | we are 1.03x SLOWER |
| gemm | llama8b.mlp_down.t1 | cublas-fp32 | 0.085 | 0.092 | 0.93 | we are 1.08x FASTER |
| gemm | llama8b.mlp_down.t1 | cublas-tf32 | 0.085 | 0.094 | 0.90 | we are 1.11x FASTER |
| gemm | llama8b.mlp_down.t512 | cublas-fp32 | 0.232 | 1.176 | 0.20 | we are 5.06x FASTER |
| gemm | llama8b.mlp_down.t512 | cublas-tf32 | 0.232 | 0.211 | 1.10 | we are 1.10x SLOWER |
| gemm | llama8b.mlp_down.t8 | cublas-fp32 | 0.110 | 0.198 | 0.55 | we are 1.81x FASTER |
| gemm | llama8b.mlp_down.t8 | cublas-tf32 | 0.110 | 0.103 | 1.07 | we are 1.07x SLOWER |
| gemm | llama8b.mlp_up.t1 | cublas-fp32 | 0.083 | 0.092 | 0.91 | we are 1.10x FASTER |
| gemm | llama8b.mlp_up.t1 | cublas-tf32 | 0.083 | 0.093 | 0.90 | we are 1.11x FASTER |
| gemm | llama8b.mlp_up.t512 | cublas-fp32 | 0.211 | 1.371 | 0.15 | we are 6.51x FASTER |
| gemm | llama8b.mlp_up.t512 | cublas-tf32 | 0.211 | 0.185 | 1.14 | we are 1.14x SLOWER |
| gemm | llama8b.mlp_up.t8 | cublas-fp32 | 0.115 | 0.160 | 0.72 | we are 1.38x FASTER |
| gemm | llama8b.mlp_up.t8 | cublas-tf32 | 0.115 | 0.105 | 1.10 | we are 1.10x SLOWER |
| gemm | llama8b.qkv.t1 | cublas-fp32 | 0.031 | 0.039 | 0.79 | we are 1.27x FASTER |
| gemm | llama8b.qkv.t1 | cublas-tf32 | 0.031 | 0.040 | 0.77 | we are 1.29x FASTER |
| gemm | llama8b.qkv.t512 | cublas-fp32 | 0.090 | 0.369 | 0.24 | we are 4.09x FASTER |
| gemm | llama8b.qkv.t512 | cublas-tf32 | 0.090 | 0.066 | 1.37 | we are 1.37x SLOWER |
| gemm | llama8b.qkv.t8 | cublas-fp32 | 0.054 | 0.058 | 0.93 | we are 1.08x FASTER |
| gemm | llama8b.qkv.t8 | cublas-tf32 | 0.054 | 0.044 | 1.22 | we are 1.22x SLOWER |
| gemm | ols.predict.gemv.64Kx16 | cublas-fp32 | 0.016 | 0.018 | 0.90 | we are 1.12x FASTER |
| gemm | ols.predict.gemv.64Kx16 | cublas-tf32 | 0.016 | 0.017 | 0.91 | we are 1.10x FASTER |
| gemm | ols.step1.16x16x64K | cublas-fp32 | 0.045 | 0.036 | 1.27 | we are 1.27x SLOWER |
| gemm | ols.step1.16x16x64K | cublas-tf32 | 0.045 | 0.027 | 1.65 | we are 1.65x SLOWER |
| gemm | pca.transform.8192x4x4 | cublas-fp32 | 0.021 | 0.019 | 1.11 | we are 1.11x SLOWER |
| gemm | pca.transform.8192x4x4 | cublas-tf32 | 0.021 | 0.018 | 1.16 | we are 1.16x SLOWER |
| gemm | pca.transform.wide.8192x64x128 | cublas-fp32 | 0.029 | 0.023 | 1.28 | we are 1.28x SLOWER |
| gemm | pca.transform.wide.8192x64x128 | cublas-tf32 | 0.029 | 0.019 | 1.51 | we are 1.51x SLOWER |
| mamba | lane.b2_l4_d8 | torch-ref-scan-gpu | 0.183 | 0.566 | 0.32 | we are 3.10x FASTER |
| mamba | lane.b2_l4_d8 | torch-ref-scan-gpu-tf32 | 0.183 | 0.545 | 0.34 | we are 2.98x FASTER |
| mamba | mamba130m.prefill.t1 | torch-ref-scan-gpu | 0.237 | 0.414 | 0.57 | we are 1.75x FASTER |
| mamba | mamba130m.prefill.t1 | torch-ref-scan-gpu-tf32 | 0.237 | 0.399 | 0.59 | we are 1.68x FASTER |
| mamba | mamba130m.prefill.t128 | torch-ref-scan-gpu | 1.042 | 5.691 | 0.18 | we are 5.46x FASTER |
| mamba | mamba130m.prefill.t128 | torch-ref-scan-gpu-tf32 | 1.042 | 5.730 | 0.18 | we are 5.50x FASTER |
| mamba | mamba130m.prefill.t512 | torch-ref-scan-gpu | 2.470 | 21.236 | 0.12 | we are 8.60x FASTER |
| mamba | mamba130m.prefill.t512 | torch-ref-scan-gpu-tf32 | 2.470 | 21.276 | 0.12 | we are 8.62x FASTER |
| mamba | mamba130m.prefill.t8 | torch-ref-scan-gpu | 0.350 | 0.754 | 0.46 | we are 2.16x FASTER |
| mamba | mamba130m.prefill.t8 | torch-ref-scan-gpu-tf32 | 0.350 | 0.722 | 0.48 | we are 2.06x FASTER |
| mlp | lane.b2_l4_d32_kv2 | torch-gpu-fp32 | 0.071 | 0.101 | 0.70 | we are 1.42x FASTER |
| mlp | lane.b2_l4_d32_kv2 | torch-gpu-tf32 | 0.071 | 0.092 | 0.77 | we are 1.29x FASTER |
| mlp | llama8b.decode.t1.ctx512 | torch-gpu-fp32 | 0.257 | 0.272 | 0.94 | we are 1.06x FASTER |
| mlp | llama8b.decode.t1.ctx512 | torch-gpu-tf32 | 0.257 | 0.270 | 0.95 | we are 1.05x FASTER |
| mlp | llama8b.prefill.t1 | torch-gpu-fp32 | 0.255 | 0.273 | 0.94 | we are 1.07x FASTER |
| mlp | llama8b.prefill.t1 | torch-gpu-tf32 | 0.255 | 0.271 | 0.94 | we are 1.06x FASTER |
| mlp | llama8b.prefill.t128 | torch-gpu-fp32 | 0.385 | 1.076 | 0.36 | we are 2.79x FASTER |
| mlp | llama8b.prefill.t128 | torch-gpu-tf32 | 0.385 | 0.357 | 1.08 | we are 1.08x SLOWER |
| mlp | llama8b.prefill.t512 | torch-gpu-fp32 | 0.684 | 4.046 | 0.17 | we are 5.92x FASTER |
| mlp | llama8b.prefill.t512 | torch-gpu-tf32 | 0.684 | 0.716 | 0.96 | we are 1.05x FASTER |
| mlp | llama8b.prefill.t8 | torch-gpu-fp32 | 0.338 | 0.513 | 0.66 | we are 1.51x FASTER |
| mlp | llama8b.prefill.t8 | torch-gpu-tf32 | 0.338 | 0.306 | 1.11 | we are 1.11x SLOWER |
| rmsnorm | lane.b2_l4_d32_kv2 | torch-gpu-fp32 | 0.010 | 0.053 | 0.19 | we are 5.40x FASTER |
| rmsnorm | lane.b2_l4_d32_kv2 | torch-gpu-tf32 | 0.010 | 0.045 | 0.22 | we are 4.64x FASTER |
| rmsnorm | llama8b.decode.t1.ctx512 | torch-gpu-fp32 | 0.282 | 0.044 | 6.36 | we are 6.36x SLOWER |
| rmsnorm | llama8b.decode.t1.ctx512 | torch-gpu-tf32 | 0.282 | 0.045 | 6.28 | we are 6.28x SLOWER |
| rmsnorm | llama8b.prefill.t1 | torch-gpu-fp32 | 0.281 | 0.049 | 5.79 | we are 5.79x SLOWER |
| rmsnorm | llama8b.prefill.t1 | torch-gpu-tf32 | 0.281 | 0.046 | 6.04 | we are 6.04x SLOWER |
| rmsnorm | llama8b.prefill.t128 | torch-gpu-fp32 | 1.052 | 0.067 | 15.58 | we are 15.58x SLOWER |
| rmsnorm | llama8b.prefill.t128 | torch-gpu-tf32 | 1.052 | 0.062 | 16.88 | we are 16.88x SLOWER |
| rmsnorm | llama8b.prefill.t512 | torch-gpu-fp32 | 1.062 | 0.084 | 12.67 | we are 12.67x SLOWER |
| rmsnorm | llama8b.prefill.t512 | torch-gpu-tf32 | 1.062 | 0.068 | 15.67 | we are 15.67x SLOWER |
| rmsnorm | llama8b.prefill.t8 | torch-gpu-fp32 | 0.345 | 0.053 | 6.54 | we are 6.54x SLOWER |
| rmsnorm | llama8b.prefill.t8 | torch-gpu-tf32 | 0.345 | 0.048 | 7.19 | we are 7.19x SLOWER |
| selective_scan | lane.b2_l4_d8 | torch-ref-scan-gpu | 0.011 | 0.287 | 0.04 | we are 26.29x FASTER |
| selective_scan | lane.b2_l4_d8 | torch-ref-scan-gpu-tf32 | 0.011 | 0.268 | 0.04 | we are 24.54x FASTER |
| selective_scan | mamba130m.prefill.t1 | torch-ref-scan-gpu | 0.009 | 0.154 | 0.06 | we are 16.41x FASTER |
| selective_scan | mamba130m.prefill.t1 | torch-ref-scan-gpu-tf32 | 0.009 | 0.142 | 0.07 | we are 15.12x FASTER |
| selective_scan | mamba130m.prefill.t128 | torch-ref-scan-gpu | 0.100 | 5.434 | 0.02 | we are 54.07x FASTER |
| selective_scan | mamba130m.prefill.t128 | torch-ref-scan-gpu-tf32 | 0.100 | 5.468 | 0.02 | we are 54.41x FASTER |
| selective_scan | mamba130m.prefill.t512 | torch-ref-scan-gpu | 0.375 | 21.092 | 0.02 | we are 56.25x FASTER |
| selective_scan | mamba130m.prefill.t512 | torch-ref-scan-gpu-tf32 | 0.375 | 21.058 | 0.02 | we are 56.16x FASTER |
| selective_scan | mamba130m.prefill.t8 | torch-ref-scan-gpu | 0.014 | 0.464 | 0.03 | we are 32.29x FASTER |
| selective_scan | mamba130m.prefill.t8 | torch-ref-scan-gpu-tf32 | 0.014 | 0.457 | 0.03 | we are 31.82x FASTER |
| transformer | lane.b2_l4_d32_kv2 | torch-gpu-fp32 | 0.396 | 0.540 | 0.73 | we are 1.36x FASTER |
| transformer | lane.b2_l4_d32_kv2 | torch-gpu-sdpa-efficient-fp32 | 0.396 | 0.389 | 1.02 | we are 1.02x SLOWER |
| transformer | lane.b2_l4_d32_kv2 | torch-gpu-sdpa-math-fp32 | 0.396 | 0.453 | 0.87 | we are 1.14x FASTER |
| transformer | lane.b2_l4_d32_kv2 | torch-gpu-tf32 | 0.396 | 0.504 | 0.79 | we are 1.27x FASTER |
| transformer | llama8b.decode.t1.ctx512 | torch-gpu-fp32 | 3.594 | 0.690 | 5.21 | we are 5.21x SLOWER |
| transformer | llama8b.decode.t1.ctx512 | torch-gpu-sdpa-efficient-fp32 | 3.594 | 0.695 | 5.17 | we are 5.17x SLOWER |
| transformer | llama8b.decode.t1.ctx512 | torch-gpu-sdpa-math-fp32 | 3.594 | 0.623 | 5.77 | we are 5.77x SLOWER |
| transformer | llama8b.decode.t1.ctx512 | torch-gpu-tf32 | 3.594 | 0.680 | 5.29 | we are 5.29x SLOWER |
| transformer | llama8b.prefill.t1 | torch-gpu-fp32 | 1.690 | 0.723 | 2.34 | we are 2.34x SLOWER |
| transformer | llama8b.prefill.t1 | torch-gpu-sdpa-efficient-fp32 | 1.690 | 0.588 | 2.87 | we are 2.87x SLOWER |
| transformer | llama8b.prefill.t1 | torch-gpu-sdpa-math-fp32 | 1.690 | 0.605 | 2.79 | we are 2.79x SLOWER |
| transformer | llama8b.prefill.t1 | torch-gpu-tf32 | 1.690 | 0.666 | 2.54 | we are 2.54x SLOWER |
| transformer | llama8b.prefill.t128 | torch-gpu-fp32 | 5.148 | 1.701 | 3.03 | we are 3.03x SLOWER |
| transformer | llama8b.prefill.t128 | torch-gpu-sdpa-efficient-fp32 | 5.148 | 1.543 | 3.34 | we are 3.34x SLOWER |
| transformer | llama8b.prefill.t128 | torch-gpu-sdpa-math-fp32 | 5.148 | 1.602 | 3.21 | we are 3.21x SLOWER |
| transformer | llama8b.prefill.t128 | torch-gpu-tf32 | 5.148 | 0.897 | 5.74 | we are 5.74x SLOWER |
| transformer | llama8b.prefill.t512 | torch-gpu-fp32 | 8.880 | 5.542 | 1.60 | we are 1.60x SLOWER |
| transformer | llama8b.prefill.t512 | torch-gpu-sdpa-efficient-fp32 | 8.880 | 5.330 | 1.67 | we are 1.67x SLOWER |
| transformer | llama8b.prefill.t512 | torch-gpu-sdpa-math-fp32 | 8.880 | 5.371 | 1.65 | we are 1.65x SLOWER |
| transformer | llama8b.prefill.t512 | torch-gpu-tf32 | 8.880 | 1.380 | 6.44 | we are 6.44x SLOWER |
| transformer | llama8b.prefill.t8 | torch-gpu-fp32 | 2.577 | 1.004 | 2.57 | we are 2.57x SLOWER |
| transformer | llama8b.prefill.t8 | torch-gpu-sdpa-efficient-fp32 | 2.577 | 0.854 | 3.02 | we are 3.02x SLOWER |
| transformer | llama8b.prefill.t8 | torch-gpu-sdpa-math-fp32 | 2.577 | 0.947 | 2.72 | we are 2.72x SLOWER |
| transformer | llama8b.prefill.t8 | torch-gpu-tf32 | 2.577 | 0.803 | 3.21 | we are 3.21x SLOWER |

## Scale UNKNOWN for every row in this run

No arm in these logs emitted `scale=` in its FSPEED header, so
this run is OLDER than the throughput/fixed-cost declaration and
the split below could not be made. The rows are listed under
FIXED-COST because that is the conservative bucket, but the label
is NOT a measurement here -- some of these lanes are genuinely
large (kmeans ships 4,000,000 x 32) and some are genuinely tiny
(krr ships 16 rows). Check each shape tag by hand before quoting
anything from this file, or re-run so the arms declare it.

## Rows, ranked -- scale UNDECLARED, check each shape by hand

Every row here is a lane whose fixture is small enough that both
arms are dominated by launch and dispatch latency, plus the Python
call overhead the vendor arm pays inside the clock and ours does
not. Read each as WHAT ONE FIT COSTS END TO END on this box.

A ratio here is not wrong, it is UNCLAIMABLE: it does not tell you
whose kernel is faster. Some of these lanes cannot be made bigger
for stated reasons -- `hdbscan`'s dense mutual-reachability arm
materializes an m x m matrix, and inventing a larger fixture to
make the number look like throughput would be inventing a dataset.
Others simply have no size knob yet, and that is owed work.

They are ranked and kept rather than deleted because the fixed
cost is a real thing a user pays on a small problem.

| rank | lane | shape | ours ms | best opponent | their ms | we are |
|---|---|---|---|---|---|---|
| 1 | rmsnorm | llama8b.prefill.t128 | 1.052 | torch-gpu-tf32 | 0.062 | **16.88x SLOWER** |
| 2 | rmsnorm | llama8b.prefill.t512 | 1.062 | torch-gpu-tf32 | 0.068 | **15.67x SLOWER** |
| 3 | rmsnorm | llama8b.prefill.t8 | 0.345 | torch-gpu-tf32 | 0.048 | **7.19x SLOWER** |
| 4 | transformer | llama8b.prefill.t512 | 8.880 | torch-gpu-tf32 | 1.380 | **6.44x SLOWER** |
| 5 | rmsnorm | llama8b.decode.t1.ctx512 | 0.282 | torch-gpu-fp32 | 0.044 | **6.36x SLOWER** |
| 6 | rmsnorm | llama8b.prefill.t1 | 0.281 | torch-gpu-tf32 | 0.046 | **6.04x SLOWER** |
| 7 | transformer | llama8b.decode.t1.ctx512 | 3.594 | torch-gpu-sdpa-math-fp32 | 0.623 | **5.77x SLOWER** |
| 8 | transformer | llama8b.prefill.t128 | 5.148 | torch-gpu-tf32 | 0.897 | **5.74x SLOWER** |
| 9 | attention | llama8b.prefill.t8 | 1.314 | torch-gpu-sdpa-efficient-fp32 | 0.311 | **4.23x SLOWER** |
| 10 | attention | llama8b.prefill.t512 | 2.171 | torch-gpu-tf32 | 0.642 | **3.38x SLOWER** |
| 11 | transformer | llama8b.prefill.t8 | 2.577 | torch-gpu-tf32 | 0.803 | **3.21x SLOWER** |
| 12 | attention | llama8b.prefill.t128 | 1.372 | torch-gpu-sdpa-efficient-fp32 | 0.433 | **3.17x SLOWER** |
| 13 | transformer | llama8b.prefill.t1 | 1.690 | torch-gpu-sdpa-efficient-fp32 | 0.588 | **2.87x SLOWER** |
| 14 | attention | llama8b.prefill.t1 | 0.713 | torch-gpu-sdpa-efficient-fp32 | 0.261 | **2.73x SLOWER** |
| 15 | attention | llama8b.decode.t1.ctx512 | 0.842 | torch-gpu-sdpa-math-fp32 | 0.312 | **2.70x SLOWER** |
| 16 | gemm | gram.32x32x1M | 0.271 | cublas-tf32 | 0.108 | **2.50x SLOWER** |
| 17 | gemm | gram.128sq.x100003 | 0.142 | cublas-tf32 | 0.058 | **2.47x SLOWER** |
| 18 | gemm | gram.32x32x64K | 0.047 | cublas-tf32 | 0.027 | **1.74x SLOWER** |
| 19 | gemm | ols.step1.16x16x64K | 0.045 | cublas-tf32 | 0.027 | **1.65x SLOWER** |
| 20 | gemm | pca.transform.wide.8192x64x128 | 0.029 | cublas-tf32 | 0.019 | **1.51x SLOWER** |
| 21 | gemm | kmeans.dist.4096x64x64 | 0.025 | cublas-tf32 | 0.017 | **1.45x SLOWER** |
| 22 | gemm | llama8b.qkv.t512 | 0.090 | cublas-tf32 | 0.066 | **1.37x SLOWER** |
| 23 | gemm | llama8b.qkv.t8 | 0.054 | cublas-tf32 | 0.044 | **1.22x SLOWER** |
| 24 | gemm | pca.transform.8192x4x4 | 0.021 | cublas-tf32 | 0.018 | **1.16x SLOWER** |
| 25 | gemm | llama8b.mlp_up.t512 | 0.211 | cublas-tf32 | 0.185 | **1.14x SLOWER** |

28 further rows not listed; 53 rows in this table.

53 rows in total have an opponent: 0 throughput, 53 fixed-cost.

## Did the two sides compute the same thing

A speed number for an arm that computes something different from
its opponent is worthless. This is that check, reported and not
gated: a large difference does not fail the run, it disqualifies
the ROW, and the row has to be readable to be disqualified.

| lane | max abs diff | max rel diff | n | source |
|---|---|---|---|---|
| attention | 0.000881135 | 0.075869 | 256 | seq.attention.torch.log |
| attention | 2.47955e-05 | 0.000691507 | 4096 | seq.attention.torch.log |
| attention | 0.108009 | 21.1578 | 32768 | seq.attention.torch.log |
| attention | 0.679304 | 1168.24 | 524288 | seq.attention.torch.log |
| attention | 0.859017 | 1496.45 | 2097152 | seq.attention.torch.log |
| attention | 0.00666761 | 0.304681 | 4096 | seq.attention.torch.log |
| mamba | 2.74181e-06 | 1.37337e-05 | 64 | seq.mamba.torch.log |
| mamba | 5.96046e-08 | 3.14108e-07 | 768 | seq.mamba.torch.log |
| mamba | 3.26335e-06 | 0.00434265 | 6144 | seq.mamba.torch.log |
| mamba | 4.41074e-06 | 0.090943 | 98304 | seq.mamba.torch.log |
| mamba | 4.85778e-06 | 0.341574 | 393216 | seq.mamba.torch.log |
| mlp | 0.000194505 | 0.0539002 | 256 | seq.mlp.torch.log |
| mlp | 4.76837e-05 | 0.00111755 | 4096 | seq.mlp.torch.log |
| mlp | 0.588859 | 26.1478 | 32768 | seq.mlp.torch.log |
| mlp | 3.88814 | 4216.42 | 524288 | seq.mlp.torch.log |
| mlp | 3.98533 | 5861.87 | 2097152 | seq.mlp.torch.log |
| mlp | 0.0250001 | 1.12695 | 4096 | seq.mlp.torch.log |
| rmsnorm | 2.38419e-07 | 1.76607e-07 | 256 | seq.rmsnorm.torch.log |
| rmsnorm | 2.6226e-06 | 1.20776e-06 | 4096 | seq.rmsnorm.torch.log |
| rmsnorm | 2.6226e-06 | 1.10911e-06 | 32768 | seq.rmsnorm.torch.log |
| rmsnorm | 3.33786e-06 | 1.49114e-06 | 524288 | seq.rmsnorm.torch.log |
| rmsnorm | 4.76837e-06 | 2.01201e-06 | 2097152 | seq.rmsnorm.torch.log |
| rmsnorm | 7.15256e-07 | 4.22088e-07 | 4096 | seq.rmsnorm.torch.log |
| selective_scan | 4.61005e-05 | 0.0227862 | 128 | seq.selective_scan.torch.log |
| selective_scan | 5.96046e-08 | 0.000100503 | 1536 | seq.selective_scan.torch.log |
| selective_scan | 0.000123411 | 6.93253 | 12288 | seq.selective_scan.torch.log |
| selective_scan | 0.000177547 | 11.9563 | 196608 | seq.selective_scan.torch.log |
| selective_scan | 0.00022918 | 219.338 | 786432 | seq.selective_scan.torch.log |
| transformer | 0.000887722 | 0.0249994 | 256 | seq.transformer.torch.log |
| transformer | 5.14984e-05 | 0.00300162 | 4096 | seq.transformer.torch.log |
| transformer | 0.635278 | 9.46577 | 32768 | seq.transformer.torch.log |
| transformer | 3.95881 | 949.929 | 524288 | seq.transformer.torch.log |
| transformer | 4.33359 | 14973.6 | 2097152 | seq.transformer.torch.log |
| transformer | 0.0267668 | 8.63169 | 4096 | seq.transformer.torch.log |

## Refused arms, kept rather than dropped

An arm that could not run is a result about this box and this image.
Deleting the row would make the table read as full coverage.

| lane | arm | reason |
|---|---|---|
| gemm | ours | llama8b.lm_head.t1 (m=1 |
| attention | torch-gpu-flash-fp32 | RuntimeError at shape lane.b2_l4_d32_kv2: No available kernel. Aborting execution. |
| attention | torch-gpu-flash-bf16 | RuntimeError at shape lane.b2_l4_d32_kv2: No available kernel. Aborting execution. |
| attention | torch-gpu-flash-fp32 | RuntimeError at shape llama8b.prefill.t1: No available kernel. Aborting execution. |
| attention | torch-gpu-flash-bf16 | RuntimeError at shape llama8b.prefill.t1: No available kernel. Aborting execution. |
| attention | torch-gpu-flash-fp32 | RuntimeError at shape llama8b.prefill.t8: No available kernel. Aborting execution. |
| attention | torch-gpu-flash-bf16 | RuntimeError at shape llama8b.prefill.t8: No available kernel. Aborting execution. |
| attention | torch-gpu-flash-fp32 | RuntimeError at shape llama8b.prefill.t128: No available kernel. Aborting execution. |
| attention | torch-gpu-flash-bf16 | RuntimeError at shape llama8b.prefill.t128: No available kernel. Aborting execution. |
| attention | torch-gpu-flash-fp32 | RuntimeError at shape llama8b.prefill.t512: No available kernel. Aborting execution. |
| attention | torch-gpu-flash-bf16 | RuntimeError at shape llama8b.prefill.t512: No available kernel. Aborting execution. |
| attention | torch-gpu-flash-fp32 | RuntimeError at shape llama8b.decode.t1.ctx512: No available kernel. Aborting execution. |
| attention | torch-gpu-flash-bf16 | RuntimeError at shape llama8b.decode.t1.ctx512: No available kernel. Aborting execution. |
| mamba | mamba-ssm-cuda | mamba_ssm is not importable (No module named 'mamba_ssm'). The only opponent left is the PURE-PYTORCH sequential reference scan, which is NOT what anyone deploys. |
| mamba | mamba-ssm-cuda | mamba_ssm is not importable (No module named 'mamba_ssm'). The only opponent left is the PURE-PYTORCH sequential reference scan, which is NOT what anyone deploys. |
| mamba | mamba-ssm-cuda | mamba_ssm is not importable (No module named 'mamba_ssm'). The only opponent left is the PURE-PYTORCH sequential reference scan, which is NOT what anyone deploys. |
| mamba | mamba-ssm-cuda | mamba_ssm is not importable (No module named 'mamba_ssm'). The only opponent left is the PURE-PYTORCH sequential reference scan, which is NOT what anyone deploys. |
| mamba | mamba-ssm-cuda | mamba_ssm is not importable (No module named 'mamba_ssm'). The only opponent left is the PURE-PYTORCH sequential reference scan, which is NOT what anyone deploys. |
| selective_scan | mamba-ssm-cuda | mamba_ssm is not importable (No module named 'mamba_ssm'). The only opponent left is the PURE-PYTORCH sequential reference scan, which is NOT what anyone deploys. |
| selective_scan | mamba-ssm-cuda | mamba_ssm is not importable (No module named 'mamba_ssm'). The only opponent left is the PURE-PYTORCH sequential reference scan, which is NOT what anyone deploys. |
| selective_scan | mamba-ssm-cuda | mamba_ssm is not importable (No module named 'mamba_ssm'). The only opponent left is the PURE-PYTORCH sequential reference scan, which is NOT what anyone deploys. |
| selective_scan | mamba-ssm-cuda | mamba_ssm is not importable (No module named 'mamba_ssm'). The only opponent left is the PURE-PYTORCH sequential reference scan, which is NOT what anyone deploys. |
| selective_scan | mamba-ssm-cuda | mamba_ssm is not importable (No module named 'mamba_ssm'). The only opponent left is the PURE-PYTORCH sequential reference scan, which is NOT what anyone deploys. |
| transformer | torch-gpu-flash-fp32 | RuntimeError at shape lane.b2_l4_d32_kv2: No available kernel. Aborting execution. |
| transformer | torch-gpu-flash-bf16 | RuntimeError at shape lane.b2_l4_d32_kv2: No available kernel. Aborting execution. |
| transformer | torch-gpu-flash-fp32 | RuntimeError at shape llama8b.prefill.t1: No available kernel. Aborting execution. |
| transformer | torch-gpu-flash-bf16 | RuntimeError at shape llama8b.prefill.t1: No available kernel. Aborting execution. |
| transformer | torch-gpu-flash-fp32 | RuntimeError at shape llama8b.prefill.t8: No available kernel. Aborting execution. |
| transformer | torch-gpu-flash-bf16 | RuntimeError at shape llama8b.prefill.t8: No available kernel. Aborting execution. |
| transformer | torch-gpu-flash-fp32 | RuntimeError at shape llama8b.prefill.t128: No available kernel. Aborting execution. |
| transformer | torch-gpu-flash-bf16 | RuntimeError at shape llama8b.prefill.t128: No available kernel. Aborting execution. |
| transformer | torch-gpu-flash-fp32 | RuntimeError at shape llama8b.prefill.t512: No available kernel. Aborting execution. |
| transformer | torch-gpu-flash-bf16 | RuntimeError at shape llama8b.prefill.t512: No available kernel. Aborting execution. |
| transformer | torch-gpu-flash-fp32 | RuntimeError at shape llama8b.decode.t1.ctx512: No available kernel. Aborting execution. |
| transformer | torch-gpu-flash-bf16 | RuntimeError at shape llama8b.decode.t1.ctx512: No available kernel. Aborting execution. |

## Determinism, reported and not judged

This is the FAST path. A hash that moves between rounds is EXPECTED here
and is recorded, not failed. It is the direct evidence for what the
IDENTICAL mode buys, measured on the same box in the same hour.

No arm's output hash moved across its rounds in this run.

## Notes the arms printed

- `gemm.gemm.cublas.log`: lane=gemm arm=cublas-fp32 library=cuBLAS build=CUDA 12.4 torch=2.4.1+cu124 allow_tf32=False
- `gemm.gemm.cublas.log`: lane=gemm arm=cublas-tf32 library=cuBLAS build=CUDA 12.4 torch=2.4.1+cu124 allow_tf32=True
- `gemm.gemm.ours.log`: lane=gemm arm=ours under FAST core/gemm.mojo calls MAX linalg.matmul, so this arm is Modular's tuned kernel reached through our surface. Its fair opponent is cublas-tf32, not cublas-fp32. See this file's docstring.
- `seq.attention.ours.log`: lane=attention arm=ours entry costs the port pays and the opponent does not: llama_refuse_bad_inputs / mamba_refuse_bad_inputs download every weight per call, and eager_attention_forward synchronizes per (batch, head). See bench/speed/SEQ_SPEED.md.
- `seq.attention.ours.log`: lane=attention arm=ours PRECISION: FAST routes every GEMM through MAX matmul, which on NVIDIA is a TENSOR-CORE (TF32-class, 10 mantissa bits) path -- measured 200 TFLOP/s against cublas-fp32 44.4 and cublas-tf32 207.5 on an H100. The fair vendor opponent for a lane that routes a GEMM is therefore a TF32 arm (torch-gpu-tf32), NOT an fp32 one, and an FSPEED-AGREE line computed against an fp32 reference is measuring THIS CUT and not a defect. Lanes routing no GEMM (rmsnorm) are unaffected and are the control. DEVIATION 1885.
- `seq.attention.ours.log`: lane=attention shape=lane.b2_l4_d32_kv2 tensor=norm1.weight n=32 hash=10553ebe1cacb130
- `seq.attention.ours.log`: lane=attention shape=lane.b2_l4_d32_kv2 tensor=norm2.weight n=32 hash=43392443a35b61f3
- `seq.attention.ours.log`: lane=attention shape=lane.b2_l4_d32_kv2 tensor=q_proj.weight n=1024 hash=b73a1a48e0c0d9f7
- `seq.attention.ours.log`: lane=attention shape=lane.b2_l4_d32_kv2 tensor=k_proj.weight n=1024 hash=de208e4e2a16ff1f
- `seq.attention.ours.log`: lane=attention shape=lane.b2_l4_d32_kv2 tensor=v_proj.weight n=1024 hash=427375797d9ef375
- `seq.attention.ours.log`: lane=attention shape=lane.b2_l4_d32_kv2 tensor=o_proj.weight n=1024 hash=1947e064f026cb25
- `seq.attention.ours.log`: lane=attention shape=lane.b2_l4_d32_kv2 tensor=gate_proj.weight n=2048 hash=284aeae6d05c4f7c
- `seq.attention.ours.log`: lane=attention shape=lane.b2_l4_d32_kv2 tensor=up_proj.weight n=2048 hash=b2fdf8deeea86017
- `seq.attention.ours.log`: lane=attention shape=lane.b2_l4_d32_kv2 tensor=down_proj.weight n=2048 hash=d1d0e3eb84d5f951
- `seq.attention.ours.log`: lane=attention shape=lane.b2_l4_d32_kv2 tensor=x n=256 hash=a55d417a40e61d75
- `seq.attention.ours.log`: lane=attention arm=ours llama_refuse_bad_inputs alone shape=lane.b2_l4_d32_kv2 ms=0.124983 (inside the transformer lane's round, not inside attention/mlp/rmsnorm)
- `seq.attention.ours.log`: lane=attention arm=ours dump shape=lane.b2_l4_d32_kv2 n=256 path=/root/gemm_leg_out/dump/seq.attention.lane.b2_l4_d32_kv2.f32.bin
- `seq.attention.ours.log`: lane=attention shape=llama8b.prefill.t1 tensor=norm1.weight n=4096 hash=2633ac9556859b7d
- `seq.attention.ours.log`: lane=attention shape=llama8b.prefill.t1 tensor=norm2.weight n=4096 hash=03b0bbae422c35f2
- `seq.attention.ours.log`: lane=attention shape=llama8b.prefill.t1 tensor=q_proj.weight n=16777216 hash=65536eef389758f8
- `seq.attention.ours.log`: lane=attention shape=llama8b.prefill.t1 tensor=k_proj.weight n=4194304 hash=2d21f4a6f190e936
- `seq.attention.ours.log`: lane=attention shape=llama8b.prefill.t1 tensor=v_proj.weight n=4194304 hash=2032c9fee9598af5
- `seq.attention.ours.log`: lane=attention shape=llama8b.prefill.t1 tensor=o_proj.weight n=16777216 hash=54a671c58956b474
- `seq.attention.ours.log`: lane=attention shape=llama8b.prefill.t1 tensor=gate_proj.weight n=58720256 hash=22d060eb830e8fa7
- `seq.attention.ours.log`: lane=attention shape=llama8b.prefill.t1 tensor=up_proj.weight n=58720256 hash=6adb4a35fc897af1
- `seq.attention.ours.log`: lane=attention shape=llama8b.prefill.t1 tensor=down_proj.weight n=58720256 hash=623f9a160be08bbb
- `seq.attention.ours.log`: lane=attention shape=llama8b.prefill.t1 tensor=x n=4096 hash=50601e14d24b1558
- `seq.attention.ours.log`: lane=attention arm=ours llama_refuse_bad_inputs alone shape=llama8b.prefill.t1 ms=956.074904 (inside the transformer lane's round, not inside attention/mlp/rmsnorm)
- `seq.attention.ours.log`: lane=attention arm=ours dump shape=llama8b.prefill.t1 n=4096 path=/root/gemm_leg_out/dump/seq.attention.llama8b.prefill.t1.f32.bin
- `seq.attention.ours.log`: lane=attention shape=llama8b.prefill.t8 tensor=norm1.weight n=4096 hash=2554a09c79b9bec3
- `seq.attention.ours.log`: lane=attention shape=llama8b.prefill.t8 tensor=norm2.weight n=4096 hash=54e8bdb50be343e8
- `seq.attention.ours.log`: lane=attention shape=llama8b.prefill.t8 tensor=q_proj.weight n=16777216 hash=7f5e346f0f234ca1
- `seq.attention.ours.log`: lane=attention shape=llama8b.prefill.t8 tensor=k_proj.weight n=4194304 hash=628683891b4ef505
- `seq.attention.ours.log`: lane=attention shape=llama8b.prefill.t8 tensor=v_proj.weight n=4194304 hash=a6e71091674bf598
- `seq.attention.ours.log`: lane=attention shape=llama8b.prefill.t8 tensor=o_proj.weight n=16777216 hash=f322159c55f94428
- `seq.attention.ours.log`: lane=attention shape=llama8b.prefill.t8 tensor=gate_proj.weight n=58720256 hash=e4ae7d5f3c49e108
- `seq.attention.ours.log`: lane=attention shape=llama8b.prefill.t8 tensor=up_proj.weight n=58720256 hash=e1e76c5d8ad14599
- `seq.attention.ours.log`: lane=attention shape=llama8b.prefill.t8 tensor=down_proj.weight n=58720256 hash=cd680ef665764882
- `seq.attention.ours.log`: lane=attention shape=llama8b.prefill.t8 tensor=x n=32768 hash=5fae3cdceecdb9e2
- `seq.attention.ours.log`: lane=attention arm=ours llama_refuse_bad_inputs alone shape=llama8b.prefill.t8 ms=628.995388 (inside the transformer lane's round, not inside attention/mlp/rmsnorm)
- `seq.attention.ours.log`: lane=attention arm=ours dump shape=llama8b.prefill.t8 n=32768 path=/root/gemm_leg_out/dump/seq.attention.llama8b.prefill.t8.f32.bin
- `seq.attention.ours.log`: lane=attention shape=llama8b.prefill.t128 tensor=norm1.weight n=4096 hash=8e82620235b4b085
- `seq.attention.ours.log`: lane=attention shape=llama8b.prefill.t128 tensor=norm2.weight n=4096 hash=c813f5e53899180d
- `seq.attention.ours.log`: lane=attention shape=llama8b.prefill.t128 tensor=q_proj.weight n=16777216 hash=c1de6ff4941afb8e
- `seq.attention.ours.log`: lane=attention shape=llama8b.prefill.t128 tensor=k_proj.weight n=4194304 hash=803e4f6ec477b10e
- `seq.attention.ours.log`: lane=attention shape=llama8b.prefill.t128 tensor=v_proj.weight n=4194304 hash=1d4fa08f9f620c73
- `seq.attention.ours.log`: lane=attention shape=llama8b.prefill.t128 tensor=o_proj.weight n=16777216 hash=27755bc1220fd650
- `seq.attention.ours.log`: lane=attention shape=llama8b.prefill.t128 tensor=gate_proj.weight n=58720256 hash=2cc46a3f987b2a35
- `seq.attention.ours.log`: lane=attention shape=llama8b.prefill.t128 tensor=up_proj.weight n=58720256 hash=8df6060ace203936
- `seq.attention.ours.log`: lane=attention shape=llama8b.prefill.t128 tensor=down_proj.weight n=58720256 hash=8005f6e866ca1053
- `seq.attention.ours.log`: lane=attention shape=llama8b.prefill.t128 tensor=x n=524288 hash=4cfda6cf929d166b
- `seq.attention.ours.log`: lane=attention arm=ours llama_refuse_bad_inputs alone shape=llama8b.prefill.t128 ms=646.834613 (inside the transformer lane's round, not inside attention/mlp/rmsnorm)
- `seq.attention.ours.log`: lane=attention arm=ours dump shape=llama8b.prefill.t128 n=524288 path=/root/gemm_leg_out/dump/seq.attention.llama8b.prefill.t128.f32.bin
- `seq.attention.ours.log`: lane=attention shape=llama8b.prefill.t512 tensor=norm1.weight n=4096 hash=267ae3b42cf4bd1e
- `seq.attention.ours.log`: lane=attention shape=llama8b.prefill.t512 tensor=norm2.weight n=4096 hash=fd8c830453040971
- `seq.attention.ours.log`: lane=attention shape=llama8b.prefill.t512 tensor=q_proj.weight n=16777216 hash=40d0cc6a048d905d
- `seq.attention.ours.log`: lane=attention shape=llama8b.prefill.t512 tensor=k_proj.weight n=4194304 hash=b5af8e64afe64f60
- `seq.attention.ours.log`: lane=attention shape=llama8b.prefill.t512 tensor=v_proj.weight n=4194304 hash=73e1eafa7abcd857
- `seq.attention.ours.log`: lane=attention shape=llama8b.prefill.t512 tensor=o_proj.weight n=16777216 hash=f8a091300d6898c7
- `seq.attention.ours.log`: lane=attention shape=llama8b.prefill.t512 tensor=gate_proj.weight n=58720256 hash=8b7b13c66c2b3235
- `seq.attention.ours.log`: lane=attention shape=llama8b.prefill.t512 tensor=up_proj.weight n=58720256 hash=f61e5f2f2b37b282
- `seq.attention.ours.log`: lane=attention shape=llama8b.prefill.t512 tensor=down_proj.weight n=58720256 hash=35adb0b4c5e445b4
- `seq.attention.ours.log`: lane=attention shape=llama8b.prefill.t512 tensor=x n=2097152 hash=4dea4dee43008247
- `seq.attention.ours.log`: lane=attention arm=ours llama_refuse_bad_inputs alone shape=llama8b.prefill.t512 ms=643.669952 (inside the transformer lane's round, not inside attention/mlp/rmsnorm)
- `seq.attention.ours.log`: lane=attention arm=ours dump shape=llama8b.prefill.t512 n=2097152 path=/root/gemm_leg_out/dump/seq.attention.llama8b.prefill.t512.f32.bin
- `seq.attention.ours.log`: lane=attention shape=llama8b.decode.t1.ctx512 tensor=norm1.weight n=4096 hash=a16c25d484966855
- `seq.attention.ours.log`: lane=attention shape=llama8b.decode.t1.ctx512 tensor=norm2.weight n=4096 hash=fdf0b3b94538d853
- `seq.attention.ours.log`: lane=attention shape=llama8b.decode.t1.ctx512 tensor=q_proj.weight n=16777216 hash=dc126cf683d35c94
- `seq.attention.ours.log`: lane=attention shape=llama8b.decode.t1.ctx512 tensor=k_proj.weight n=4194304 hash=9bb615a45cfbb66d
- `seq.attention.ours.log`: lane=attention shape=llama8b.decode.t1.ctx512 tensor=v_proj.weight n=4194304 hash=b189eaa5691f2cc2
- `seq.attention.ours.log`: lane=attention shape=llama8b.decode.t1.ctx512 tensor=o_proj.weight n=16777216 hash=f460975b0ff8b858
- `seq.attention.ours.log`: lane=attention shape=llama8b.decode.t1.ctx512 tensor=gate_proj.weight n=58720256 hash=4b88f0c6df895117
- `seq.attention.ours.log`: lane=attention shape=llama8b.decode.t1.ctx512 tensor=up_proj.weight n=58720256 hash=0bcff2e323b1e524
- `seq.attention.ours.log`: lane=attention shape=llama8b.decode.t1.ctx512 tensor=down_proj.weight n=58720256 hash=513387fa68b57e5e
- `seq.attention.ours.log`: lane=attention shape=llama8b.decode.t1.ctx512 tensor=x n=4096 hash=a1509d4577b1d620
- `seq.attention.ours.log`: lane=attention shape=llama8b.decode.t1.ctx512 tensor=ctx.x n=2097152 hash=cf5b380dda81451a
- `seq.attention.ours.log`: lane=attention arm=ours llama_refuse_bad_inputs alone shape=llama8b.decode.t1.ctx512 ms=617.965952 (inside the transformer lane's round, not inside attention/mlp/rmsnorm)
- `seq.attention.ours.log`: lane=attention arm=ours dump shape=llama8b.decode.t1.ctx512 n=4096 path=/root/gemm_leg_out/dump/seq.attention.llama8b.decode.t1.ctx512.f32.bin
- `seq.attention.ours.log`: lane=attention mode=FAST
- `seq.attention.torch.log`: lane=attention arm=torch build=CUDA 12.4 torch=2.4.1+cu124 tf32_switches=cuda.matmul.allow_tf32,cudnn.allow_tf32 deterministic=off
- `seq.attention.torch.log`: lane=attention.crosscheck shape=lane.b2_l4_d32_kv2 tensor=norm1.weight n=32 hash=e4ae9bac1e4f2eb0
- `seq.attention.torch.log`: lane=attention.crosscheck shape=lane.b2_l4_d32_kv2 tensor=norm2.weight n=32 hash=fe0abe179e07e573
- `seq.attention.torch.log`: lane=attention.crosscheck shape=lane.b2_l4_d32_kv2 tensor=q_proj.weight n=1024 hash=0d2da18fc0b40d8b
- `seq.attention.torch.log`: lane=attention.crosscheck shape=lane.b2_l4_d32_kv2 tensor=k_proj.weight n=1024 hash=8d6df99dbcd279fb
- `seq.attention.torch.log`: lane=attention.crosscheck shape=lane.b2_l4_d32_kv2 tensor=v_proj.weight n=1024 hash=566369012d3485a1
- `seq.attention.torch.log`: lane=attention.crosscheck shape=lane.b2_l4_d32_kv2 tensor=o_proj.weight n=1024 hash=d879230290d22ae1
- `seq.attention.torch.log`: lane=attention.crosscheck shape=lane.b2_l4_d32_kv2 tensor=gate_proj.weight n=2048 hash=5cd27648cfa7a924
- `seq.attention.torch.log`: lane=attention.crosscheck shape=lane.b2_l4_d32_kv2 tensor=up_proj.weight n=2048 hash=86fa1f885662c17f
- `seq.attention.torch.log`: lane=attention.crosscheck shape=lane.b2_l4_d32_kv2 tensor=down_proj.weight n=2048 hash=62887e5f6fb89109
- `seq.attention.torch.log`: lane=attention.crosscheck shape=lane.b2_l4_d32_kv2 tensor=x n=256 hash=cb5073d347fd2aae
- `seq.attention.torch.log`: lane=attention arm=torch crosscheck LlamaEager vs transformer/corpus block_forward at lane.b2_l4_d32_kv2: max_abs_diff=1.5878e-08
- `seq.attention.torch.log`: lane=attention shape=lane.b2_l4_d32_kv2 tensor=norm1.weight n=32 hash=e4ae9bac1e4f2eb0
- `seq.attention.torch.log`: lane=attention shape=lane.b2_l4_d32_kv2 tensor=norm2.weight n=32 hash=fe0abe179e07e573
- `seq.attention.torch.log`: lane=attention shape=lane.b2_l4_d32_kv2 tensor=q_proj.weight n=1024 hash=0d2da18fc0b40d8b
- `seq.attention.torch.log`: lane=attention shape=lane.b2_l4_d32_kv2 tensor=k_proj.weight n=1024 hash=8d6df99dbcd279fb
- `seq.attention.torch.log`: lane=attention shape=lane.b2_l4_d32_kv2 tensor=v_proj.weight n=1024 hash=566369012d3485a1
- `seq.attention.torch.log`: lane=attention shape=lane.b2_l4_d32_kv2 tensor=o_proj.weight n=1024 hash=d879230290d22ae1
- `seq.attention.torch.log`: lane=attention shape=lane.b2_l4_d32_kv2 tensor=gate_proj.weight n=2048 hash=5cd27648cfa7a924
- `seq.attention.torch.log`: lane=attention shape=lane.b2_l4_d32_kv2 tensor=up_proj.weight n=2048 hash=86fa1f885662c17f
- `seq.attention.torch.log`: lane=attention shape=lane.b2_l4_d32_kv2 tensor=down_proj.weight n=2048 hash=62887e5f6fb89109
- `seq.attention.torch.log`: lane=attention shape=lane.b2_l4_d32_kv2 tensor=x n=256 hash=cb5073d347fd2aae
- `seq.attention.torch.log`: lane=attention arm=torch-gpu-fp32 output witness shape=lane.b2_l4_d32_kv2 n=256 hash=e8e95582b48f83cd
- `seq.attention.torch.log`: lane=attention arm=torch-gpu-tf32 output witness shape=lane.b2_l4_d32_kv2 n=256 hash=550727a9d0db290a
- `seq.attention.torch.log`: lane=attention arm=torch-gpu-sdpa-math-fp32 output witness shape=lane.b2_l4_d32_kv2 n=256 hash=e01cb73a157448ed
- `seq.attention.torch.log`: lane=attention arm=torch-gpu-sdpa-efficient-fp32 output witness shape=lane.b2_l4_d32_kv2 n=256 hash=c89e8c280bb9a932
- `seq.attention.torch.log`: lane=attention shape=llama8b.prefill.t1 tensor=norm1.weight n=4096 hash=30dc9291bcd74c4d
- `seq.attention.torch.log`: lane=attention shape=llama8b.prefill.t1 tensor=norm2.weight n=4096 hash=90de83fb74424482
- `seq.attention.torch.log`: lane=attention shape=llama8b.prefill.t1 tensor=q_proj.weight n=16777216 hash=458f23ff554c3cc3
- `seq.attention.torch.log`: lane=attention shape=llama8b.prefill.t1 tensor=k_proj.weight n=4194304 hash=76706639c3aa7d56
- `seq.attention.torch.log`: lane=attention shape=llama8b.prefill.t1 tensor=v_proj.weight n=4194304 hash=81c62305b8e723d5
- `seq.attention.torch.log`: lane=attention shape=llama8b.prefill.t1 tensor=o_proj.weight n=16777216 hash=cffb8b77180868b3
- `seq.attention.torch.log`: lane=attention shape=llama8b.prefill.t1 tensor=gate_proj.weight n=58720256 hash=4616212f0e15d8ca
- `seq.attention.torch.log`: lane=attention shape=llama8b.prefill.t1 tensor=up_proj.weight n=58720256 hash=3d72973cf9e50898
- `seq.attention.torch.log`: lane=attention shape=llama8b.prefill.t1 tensor=down_proj.weight n=58720256 hash=63d7be8563af230a
- `seq.attention.torch.log`: lane=attention shape=llama8b.prefill.t1 tensor=x n=4096 hash=3d93aac664c66028
- `seq.attention.torch.log`: lane=attention arm=torch-gpu-fp32 output witness shape=llama8b.prefill.t1 n=4096 hash=dc7893ab5281d902
- `seq.attention.torch.log`: lane=attention arm=torch-gpu-tf32 output witness shape=llama8b.prefill.t1 n=4096 hash=dc7893ab5281d902
- `seq.attention.torch.log`: lane=attention arm=torch-gpu-sdpa-math-fp32 output witness shape=llama8b.prefill.t1 n=4096 hash=dc7893ab5281d902
- `seq.attention.torch.log`: lane=attention arm=torch-gpu-sdpa-efficient-fp32 output witness shape=llama8b.prefill.t1 n=4096 hash=0108982a62078f7c
- `seq.attention.torch.log`: lane=attention shape=llama8b.prefill.t8 tensor=norm1.weight n=4096 hash=19770838eb156233
- `seq.attention.torch.log`: lane=attention shape=llama8b.prefill.t8 tensor=norm2.weight n=4096 hash=661f9b4826837778
- `seq.attention.torch.log`: lane=attention shape=llama8b.prefill.t8 tensor=q_proj.weight n=16777216 hash=9ba40b617f6701c6
- `seq.attention.torch.log`: lane=attention shape=llama8b.prefill.t8 tensor=k_proj.weight n=4194304 hash=3fc887e58a86aae5
- `seq.attention.torch.log`: lane=attention shape=llama8b.prefill.t8 tensor=v_proj.weight n=4194304 hash=64cb8d74bed4b578
- `seq.attention.torch.log`: lane=attention shape=llama8b.prefill.t8 tensor=o_proj.weight n=16777216 hash=5abb6368b7c381b7
- `seq.attention.torch.log`: lane=attention shape=llama8b.prefill.t8 tensor=gate_proj.weight n=58720256 hash=40827a19ef224a29
- `seq.attention.torch.log`: lane=attention shape=llama8b.prefill.t8 tensor=up_proj.weight n=58720256 hash=b8d683c2de2a8d40
- `seq.attention.torch.log`: lane=attention shape=llama8b.prefill.t8 tensor=down_proj.weight n=58720256 hash=be5808bd1301911b
- `seq.attention.torch.log`: lane=attention shape=llama8b.prefill.t8 tensor=x n=32768 hash=b393ca1e2e252bc2
- `seq.attention.torch.log`: lane=attention arm=torch-gpu-fp32 output witness shape=llama8b.prefill.t8 n=32768 hash=aa3d909320a0f5a6
- `seq.attention.torch.log`: lane=attention arm=torch-gpu-tf32 output witness shape=llama8b.prefill.t8 n=32768 hash=e7f6184ee11cb9e8
- `seq.attention.torch.log`: lane=attention arm=torch-gpu-sdpa-math-fp32 output witness shape=llama8b.prefill.t8 n=32768 hash=8d996e83fef71af2
- `seq.attention.torch.log`: lane=attention arm=torch-gpu-sdpa-efficient-fp32 output witness shape=llama8b.prefill.t8 n=32768 hash=aa10b5f27dd5ae9f
- `seq.attention.torch.log`: lane=attention shape=llama8b.prefill.t128 tensor=norm1.weight n=4096 hash=2d073c4bf3be4215
- `seq.attention.torch.log`: lane=attention shape=llama8b.prefill.t128 tensor=norm2.weight n=4096 hash=8d13de87ae1db05d
- `seq.attention.torch.log`: lane=attention shape=llama8b.prefill.t128 tensor=q_proj.weight n=16777216 hash=19976a410e812f3d
- `seq.attention.torch.log`: lane=attention shape=llama8b.prefill.t128 tensor=k_proj.weight n=4194304 hash=6eb9a28e49ec40ae
- `seq.attention.torch.log`: lane=attention shape=llama8b.prefill.t128 tensor=v_proj.weight n=4194304 hash=91936c2a28335c93
- `seq.attention.torch.log`: lane=attention shape=llama8b.prefill.t128 tensor=o_proj.weight n=16777216 hash=33f47dd4c2896a57
- `seq.attention.torch.log`: lane=attention shape=llama8b.prefill.t128 tensor=gate_proj.weight n=58720256 hash=00ad980c0db80b74
- `seq.attention.torch.log`: lane=attention shape=llama8b.prefill.t128 tensor=up_proj.weight n=58720256 hash=c2002fe051fc2a67
- `seq.attention.torch.log`: lane=attention shape=llama8b.prefill.t128 tensor=down_proj.weight n=58720256 hash=fe37d1e5eadac30a
- `seq.attention.torch.log`: lane=attention shape=llama8b.prefill.t128 tensor=x n=524288 hash=60e833df1e7657f3
- `seq.attention.torch.log`: lane=attention arm=torch-gpu-fp32 output witness shape=llama8b.prefill.t128 n=524288 hash=e8bfa1bddeaba8ad
- `seq.attention.torch.log`: lane=attention arm=torch-gpu-tf32 output witness shape=llama8b.prefill.t128 n=524288 hash=5f584d5da5cd2a0f
- `seq.attention.torch.log`: lane=attention arm=torch-gpu-sdpa-math-fp32 output witness shape=llama8b.prefill.t128 n=524288 hash=a720598f9544fc67
- `seq.attention.torch.log`: lane=attention arm=torch-gpu-sdpa-efficient-fp32 output witness shape=llama8b.prefill.t128 n=524288 hash=b4c54fc45870ecc8
- `seq.attention.torch.log`: lane=attention shape=llama8b.prefill.t512 tensor=norm1.weight n=4096 hash=f4a0f716372b58ce
- `seq.attention.torch.log`: lane=attention shape=llama8b.prefill.t512 tensor=norm2.weight n=4096 hash=93b3f2cddfd23c01
- `seq.attention.torch.log`: lane=attention shape=llama8b.prefill.t512 tensor=q_proj.weight n=16777216 hash=fe1b3ad1dfd94d86
- `seq.attention.torch.log`: lane=attention shape=llama8b.prefill.t512 tensor=k_proj.weight n=4194304 hash=9f658c4d5c9859c0
- `seq.attention.torch.log`: lane=attention shape=llama8b.prefill.t512 tensor=v_proj.weight n=4194304 hash=286524a647dfa937
- `seq.attention.torch.log`: lane=attention shape=llama8b.prefill.t512 tensor=o_proj.weight n=16777216 hash=7721f27f6a6c4ce8
- `seq.attention.torch.log`: lane=attention shape=llama8b.prefill.t512 tensor=gate_proj.weight n=58720256 hash=630b3a509bc6f674
- `seq.attention.torch.log`: lane=attention shape=llama8b.prefill.t512 tensor=up_proj.weight n=58720256 hash=4cf23bd89c42c8b3
- `seq.attention.torch.log`: lane=attention shape=llama8b.prefill.t512 tensor=down_proj.weight n=58720256 hash=5c6bd78157391281
- `seq.attention.torch.log`: lane=attention shape=llama8b.prefill.t512 tensor=x n=2097152 hash=9ffa2521076bc8c7
- `seq.attention.torch.log`: lane=attention arm=torch-gpu-fp32 output witness shape=llama8b.prefill.t512 n=2097152 hash=c6d68476eaeac14b
- `seq.attention.torch.log`: lane=attention arm=torch-gpu-tf32 output witness shape=llama8b.prefill.t512 n=2097152 hash=139d70bfa794a6a5
- `seq.attention.torch.log`: lane=attention arm=torch-gpu-sdpa-math-fp32 output witness shape=llama8b.prefill.t512 n=2097152 hash=7027b887b773ee7f
- `seq.attention.torch.log`: lane=attention arm=torch-gpu-sdpa-efficient-fp32 output witness shape=llama8b.prefill.t512 n=2097152 hash=d3ed9b12b43c7361
- `seq.attention.torch.log`: lane=attention shape=llama8b.decode.t1.ctx512 tensor=norm1.weight n=4096 hash=939d6972265aa725
- `seq.attention.torch.log`: lane=attention shape=llama8b.decode.t1.ctx512 tensor=norm2.weight n=4096 hash=31804b90e76959c3
- `seq.attention.torch.log`: lane=attention shape=llama8b.decode.t1.ctx512 tensor=q_proj.weight n=16777216 hash=b1b0a1ba4873ca77
- `seq.attention.torch.log`: lane=attention shape=llama8b.decode.t1.ctx512 tensor=k_proj.weight n=4194304 hash=1c16ca3348db5f0d
- `seq.attention.torch.log`: lane=attention shape=llama8b.decode.t1.ctx512 tensor=v_proj.weight n=4194304 hash=237925dc612d1c22
- `seq.attention.torch.log`: lane=attention shape=llama8b.decode.t1.ctx512 tensor=o_proj.weight n=16777216 hash=0ec594792330189f
- `seq.attention.torch.log`: lane=attention shape=llama8b.decode.t1.ctx512 tensor=gate_proj.weight n=58720256 hash=c5c194484347d30a
- `seq.attention.torch.log`: lane=attention shape=llama8b.decode.t1.ctx512 tensor=up_proj.weight n=58720256 hash=9374a3fc8f279dd9
- `seq.attention.torch.log`: lane=attention shape=llama8b.decode.t1.ctx512 tensor=down_proj.weight n=58720256 hash=6e1a2f6cfa111b57
- `seq.attention.torch.log`: lane=attention shape=llama8b.decode.t1.ctx512 tensor=x n=4096 hash=b91b5914caac5a30
- `seq.attention.torch.log`: lane=attention shape=llama8b.decode.t1.ctx512 tensor=ctx.x n=2097152 hash=c54a8d22a1bde29a
- `seq.attention.torch.log`: lane=attention arm=torch-gpu-fp32 output witness shape=llama8b.decode.t1.ctx512 n=4096 hash=07cb383ee9544db3
- `seq.attention.torch.log`: lane=attention arm=torch-gpu-tf32 output witness shape=llama8b.decode.t1.ctx512 n=4096 hash=88c632658f1d8421
- `seq.attention.torch.log`: lane=attention arm=torch-gpu-sdpa-math-fp32 output witness shape=llama8b.decode.t1.ctx512 n=4096 hash=61fe91daf1d8f3d7
- `seq.attention.torch.log`: lane=attention arm=torch-gpu-sdpa-efficient-fp32 output witness shape=llama8b.decode.t1.ctx512 n=4096 hash=92c858cfb8cbac82
- `seq.attention.torch.log`: lane=attention arm=torch
- `seq.mamba.ours.log`: lane=mamba arm=ours entry costs the port pays and the opponent does not: llama_refuse_bad_inputs / mamba_refuse_bad_inputs download every weight per call, and eager_attention_forward synchronizes per (batch, head). See bench/speed/SEQ_SPEED.md.
- `seq.mamba.ours.log`: lane=mamba arm=ours PRECISION: FAST routes every GEMM through MAX matmul, which on NVIDIA is a TENSOR-CORE (TF32-class, 10 mantissa bits) path -- measured 200 TFLOP/s against cublas-fp32 44.4 and cublas-tf32 207.5 on an H100. The fair vendor opponent for a lane that routes a GEMM is therefore a TF32 arm (torch-gpu-tf32), NOT an fp32 one, and an FSPEED-AGREE line computed against an fp32 reference is measuring THIS CUT and not a defect. Lanes routing no GEMM (rmsnorm) are unaffected and are the control. DEVIATION 1885.
- `seq.mamba.ours.log`: lane=mamba shape=lane.b2_l4_d8 tensor=norm.weight n=8 hash=50c65afc06287cd7
- `seq.mamba.ours.log`: lane=mamba shape=lane.b2_l4_d8 tensor=in_proj.weight n=256 hash=e08dc872b261976f
- `seq.mamba.ours.log`: lane=mamba shape=lane.b2_l4_d8 tensor=conv1d.weight n=64 hash=87567a7cd21ae244
- `seq.mamba.ours.log`: lane=mamba shape=lane.b2_l4_d8 tensor=conv1d.bias n=16 hash=fffb5b6163f3b167
- `seq.mamba.ours.log`: lane=mamba shape=lane.b2_l4_d8 tensor=x_proj.weight n=528 hash=e6f1d28bec911569
- `seq.mamba.ours.log`: lane=mamba shape=lane.b2_l4_d8 tensor=dt_proj.weight n=16 hash=ff1a20b85cbe6334
- `seq.mamba.ours.log`: lane=mamba shape=lane.b2_l4_d8 tensor=dt_proj.bias n=16 hash=f9fd9e3769ba990f
- `seq.mamba.ours.log`: lane=mamba shape=lane.b2_l4_d8 tensor=A_log n=256 hash=fd875e94f94076a3
- `seq.mamba.ours.log`: lane=mamba shape=lane.b2_l4_d8 tensor=D n=16 hash=427cb25db82f4bd0
- `seq.mamba.ours.log`: lane=mamba shape=lane.b2_l4_d8 tensor=out_proj.weight n=128 hash=9fce64d8212a8eec
- `seq.mamba.ours.log`: lane=mamba shape=lane.b2_l4_d8 tensor=x n=64 hash=622c7d6b00d2329d
- `seq.mamba.ours.log`: lane=mamba arm=ours mamba_refuse_bad_inputs alone shape=lane.b2_l4_d8 ms=0.039933 (inside the mamba lane's round, not inside selective_scan)
- `seq.mamba.ours.log`: lane=mamba arm=ours dump shape=lane.b2_l4_d8 n=64 path=/root/gemm_leg_out/dump/seq.mamba.lane.b2_l4_d8.f32.bin
- `seq.mamba.ours.log`: lane=mamba shape=mamba130m.prefill.t1 tensor=norm.weight n=768 hash=80ffb6eceac983c1
- `seq.mamba.ours.log`: lane=mamba shape=mamba130m.prefill.t1 tensor=in_proj.weight n=2359296 hash=9a4025f0cad505d9
- `seq.mamba.ours.log`: lane=mamba shape=mamba130m.prefill.t1 tensor=conv1d.weight n=6144 hash=4f37ca5c86596e26
- `seq.mamba.ours.log`: lane=mamba shape=mamba130m.prefill.t1 tensor=conv1d.bias n=1536 hash=3730d3d1a4bd30d5
- `seq.mamba.ours.log`: lane=mamba shape=mamba130m.prefill.t1 tensor=x_proj.weight n=122880 hash=d8becf46fdf2b30f
- `seq.mamba.ours.log`: lane=mamba shape=mamba130m.prefill.t1 tensor=dt_proj.weight n=73728 hash=3333a7a393aaf4c1
- `seq.mamba.ours.log`: lane=mamba shape=mamba130m.prefill.t1 tensor=dt_proj.bias n=1536 hash=670910bd72deb317
- `seq.mamba.ours.log`: lane=mamba shape=mamba130m.prefill.t1 tensor=A_log n=24576 hash=30d9abc8d0f88426
- `seq.mamba.ours.log`: lane=mamba shape=mamba130m.prefill.t1 tensor=D n=1536 hash=4af9cecf78461e69
- `seq.mamba.ours.log`: lane=mamba shape=mamba130m.prefill.t1 tensor=out_proj.weight n=1179648 hash=03bc8590d6d08f9f
- `seq.mamba.ours.log`: lane=mamba shape=mamba130m.prefill.t1 tensor=x n=768 hash=5832c6c66c7945eb
- `seq.mamba.ours.log`: lane=mamba arm=ours mamba_refuse_bad_inputs alone shape=mamba130m.prefill.t1 ms=0.076628 (inside the mamba lane's round, not inside selective_scan)
- `seq.mamba.ours.log`: lane=mamba arm=ours dump shape=mamba130m.prefill.t1 n=768 path=/root/gemm_leg_out/dump/seq.mamba.mamba130m.prefill.t1.f32.bin
- `seq.mamba.ours.log`: lane=mamba shape=mamba130m.prefill.t8 tensor=norm.weight n=768 hash=8ca9a62b03bb0b69
- `seq.mamba.ours.log`: lane=mamba shape=mamba130m.prefill.t8 tensor=in_proj.weight n=2359296 hash=e5e828782fc4c377
- `seq.mamba.ours.log`: lane=mamba shape=mamba130m.prefill.t8 tensor=conv1d.weight n=6144 hash=1fe9d376751939c2
- `seq.mamba.ours.log`: lane=mamba shape=mamba130m.prefill.t8 tensor=conv1d.bias n=1536 hash=284b95c47159d2e2
- `seq.mamba.ours.log`: lane=mamba shape=mamba130m.prefill.t8 tensor=x_proj.weight n=122880 hash=58fc41a3c66083f9
- `seq.mamba.ours.log`: lane=mamba shape=mamba130m.prefill.t8 tensor=dt_proj.weight n=73728 hash=354e3487a0f90c0b
- `seq.mamba.ours.log`: lane=mamba shape=mamba130m.prefill.t8 tensor=dt_proj.bias n=1536 hash=1729f7678a315600
- `seq.mamba.ours.log`: lane=mamba shape=mamba130m.prefill.t8 tensor=A_log n=24576 hash=71d966732193f1fb
- `seq.mamba.ours.log`: lane=mamba shape=mamba130m.prefill.t8 tensor=D n=1536 hash=146d24138daf5ed0
- `seq.mamba.ours.log`: lane=mamba shape=mamba130m.prefill.t8 tensor=out_proj.weight n=1179648 hash=5f7e08294f2017f1
- `seq.mamba.ours.log`: lane=mamba shape=mamba130m.prefill.t8 tensor=x n=6144 hash=aea97b3003677e03
- `seq.mamba.ours.log`: lane=mamba arm=ours mamba_refuse_bad_inputs alone shape=mamba130m.prefill.t8 ms=0.094334 (inside the mamba lane's round, not inside selective_scan)
- `seq.mamba.ours.log`: lane=mamba arm=ours dump shape=mamba130m.prefill.t8 n=6144 path=/root/gemm_leg_out/dump/seq.mamba.mamba130m.prefill.t8.f32.bin
- `seq.mamba.ours.log`: lane=mamba shape=mamba130m.prefill.t128 tensor=norm.weight n=768 hash=9d0c85053f9dbe1b
- `seq.mamba.ours.log`: lane=mamba shape=mamba130m.prefill.t128 tensor=in_proj.weight n=2359296 hash=1a8a102874a8f357
- `seq.mamba.ours.log`: lane=mamba shape=mamba130m.prefill.t128 tensor=conv1d.weight n=6144 hash=0e615d0b95e74f90
- `seq.mamba.ours.log`: lane=mamba shape=mamba130m.prefill.t128 tensor=conv1d.bias n=1536 hash=87a4bee5bec76720
- `seq.mamba.ours.log`: lane=mamba shape=mamba130m.prefill.t128 tensor=x_proj.weight n=122880 hash=24305c34fd48b3a6
- `seq.mamba.ours.log`: lane=mamba shape=mamba130m.prefill.t128 tensor=dt_proj.weight n=73728 hash=6b5d720462192045
- `seq.mamba.ours.log`: lane=mamba shape=mamba130m.prefill.t128 tensor=dt_proj.bias n=1536 hash=68b70160bd63976c
- `seq.mamba.ours.log`: lane=mamba shape=mamba130m.prefill.t128 tensor=A_log n=24576 hash=4d659006cfa457de
- `seq.mamba.ours.log`: lane=mamba shape=mamba130m.prefill.t128 tensor=D n=1536 hash=1f5fdef06a2e6f1e
- `seq.mamba.ours.log`: lane=mamba shape=mamba130m.prefill.t128 tensor=out_proj.weight n=1179648 hash=be795e26211acaff
- `seq.mamba.ours.log`: lane=mamba shape=mamba130m.prefill.t128 tensor=x n=98304 hash=25fed7518eb79bb2
- `seq.mamba.ours.log`: lane=mamba arm=ours mamba_refuse_bad_inputs alone shape=mamba130m.prefill.t128 ms=0.445237 (inside the mamba lane's round, not inside selective_scan)
- `seq.mamba.ours.log`: lane=mamba arm=ours dump shape=mamba130m.prefill.t128 n=98304 path=/root/gemm_leg_out/dump/seq.mamba.mamba130m.prefill.t128.f32.bin
- `seq.mamba.ours.log`: lane=mamba shape=mamba130m.prefill.t512 tensor=norm.weight n=768 hash=8a1f60d0ecc7f985
- `seq.mamba.ours.log`: lane=mamba shape=mamba130m.prefill.t512 tensor=in_proj.weight n=2359296 hash=93cf34e9b609c366
- `seq.mamba.ours.log`: lane=mamba shape=mamba130m.prefill.t512 tensor=conv1d.weight n=6144 hash=f2f7fd124b119079
- `seq.mamba.ours.log`: lane=mamba shape=mamba130m.prefill.t512 tensor=conv1d.bias n=1536 hash=13b6f9e0acc0cdc3
- `seq.mamba.ours.log`: lane=mamba shape=mamba130m.prefill.t512 tensor=x_proj.weight n=122880 hash=858ca82562b215ab
- `seq.mamba.ours.log`: lane=mamba shape=mamba130m.prefill.t512 tensor=dt_proj.weight n=73728 hash=099f48c5ccdb6f44
- `seq.mamba.ours.log`: lane=mamba shape=mamba130m.prefill.t512 tensor=dt_proj.bias n=1536 hash=8bc414b2e6db2757
- `seq.mamba.ours.log`: lane=mamba shape=mamba130m.prefill.t512 tensor=A_log n=24576 hash=0895a62212804fc5
- `seq.mamba.ours.log`: lane=mamba shape=mamba130m.prefill.t512 tensor=D n=1536 hash=d5239bfe63a8472c
- `seq.mamba.ours.log`: lane=mamba shape=mamba130m.prefill.t512 tensor=out_proj.weight n=1179648 hash=b41416f53f8b72c7
- `seq.mamba.ours.log`: lane=mamba shape=mamba130m.prefill.t512 tensor=x n=393216 hash=baa7b8e45c791eef
- `seq.mamba.ours.log`: lane=mamba arm=ours mamba_refuse_bad_inputs alone shape=mamba130m.prefill.t512 ms=0.977212 (inside the mamba lane's round, not inside selective_scan)
- `seq.mamba.ours.log`: lane=mamba arm=ours dump shape=mamba130m.prefill.t512 n=393216 path=/root/gemm_leg_out/dump/seq.mamba.mamba130m.prefill.t512.f32.bin
- `seq.mamba.ours.log`: lane=mamba mode=FAST
- `seq.mamba.torch.log`: lane=mamba arm=torch build=CUDA 12.4 torch=2.4.1+cu124 tf32_switches=cuda.matmul.allow_tf32,cudnn.allow_tf32 deterministic=off
- `seq.mamba.torch.log`: lane=mamba shape=lane.b2_l4_d8 tensor=norm.weight n=8 hash=82d54a938d591baf
- `seq.mamba.torch.log`: lane=mamba shape=lane.b2_l4_d8 tensor=in_proj.weight n=256 hash=f22fc6a25b5a0910
- `seq.mamba.torch.log`: lane=mamba shape=lane.b2_l4_d8 tensor=conv1d.weight n=64 hash=90cc6f6dd561f6e4
- `seq.mamba.torch.log`: lane=mamba shape=lane.b2_l4_d8 tensor=conv1d.bias n=16 hash=edb3d28d0d124257
- `seq.mamba.torch.log`: lane=mamba shape=lane.b2_l4_d8 tensor=x_proj.weight n=528 hash=7e44686404574eaf
- `seq.mamba.torch.log`: lane=mamba shape=lane.b2_l4_d8 tensor=dt_proj.weight n=16 hash=6c1b908f90a2f0e4
- `seq.mamba.torch.log`: lane=mamba shape=lane.b2_l4_d8 tensor=dt_proj.bias n=16 hash=b95295e1de8ff21f
- `seq.mamba.torch.log`: lane=mamba shape=lane.b2_l4_d8 tensor=A_log n=256 hash=3075ef7a49146f3c
- `seq.mamba.torch.log`: lane=mamba shape=lane.b2_l4_d8 tensor=D n=16 hash=0b155ba1ec3dd1e0
- `seq.mamba.torch.log`: lane=mamba shape=lane.b2_l4_d8 tensor=out_proj.weight n=128 hash=bbf7e7dac9bfaacc
- `seq.mamba.torch.log`: lane=mamba shape=lane.b2_l4_d8 tensor=x n=64 hash=8ee57faefcc7733d
- `seq.mamba.torch.log`: lane=mamba arm=torch-ref-scan-gpu output witness shape=lane.b2_l4_d8 n=64 hash=4eefaa4a619c7219
- `seq.mamba.torch.log`: lane=mamba arm=torch-ref-scan-gpu-tf32 output witness shape=lane.b2_l4_d8 n=64 hash=3621694635e65892
- `seq.mamba.torch.log`: lane=mamba shape=mamba130m.prefill.t1 tensor=norm.weight n=768 hash=935bef95533ae1ec
- `seq.mamba.torch.log`: lane=mamba shape=mamba130m.prefill.t1 tensor=in_proj.weight n=2359296 hash=3b23ea144a46a005
- `seq.mamba.torch.log`: lane=mamba shape=mamba130m.prefill.t1 tensor=conv1d.weight n=6144 hash=35f623af5f3ba70e
- `seq.mamba.torch.log`: lane=mamba shape=mamba130m.prefill.t1 tensor=conv1d.bias n=1536 hash=90da985da6eb43d3
- `seq.mamba.torch.log`: lane=mamba shape=mamba130m.prefill.t1 tensor=x_proj.weight n=122880 hash=5f14325f2227fe36
- `seq.mamba.torch.log`: lane=mamba shape=mamba130m.prefill.t1 tensor=dt_proj.weight n=73728 hash=dd4c6a116a702514
- `seq.mamba.torch.log`: lane=mamba shape=mamba130m.prefill.t1 tensor=dt_proj.bias n=1536 hash=d9646bbf3578e7e5
- `seq.mamba.torch.log`: lane=mamba shape=mamba130m.prefill.t1 tensor=A_log n=24576 hash=60f6da4a1075bf66
- `seq.mamba.torch.log`: lane=mamba shape=mamba130m.prefill.t1 tensor=D n=1536 hash=98eff32333cba8fb
- `seq.mamba.torch.log`: lane=mamba shape=mamba130m.prefill.t1 tensor=out_proj.weight n=1179648 hash=056439617b9552d1
- `seq.mamba.torch.log`: lane=mamba shape=mamba130m.prefill.t1 tensor=x n=768 hash=9628d92b7a03fefa
- `seq.mamba.torch.log`: lane=mamba arm=torch-ref-scan-gpu output witness shape=mamba130m.prefill.t1 n=768 hash=539eea7de7dd42a6
- `seq.mamba.torch.log`: lane=mamba arm=torch-ref-scan-gpu-tf32 output witness shape=mamba130m.prefill.t1 n=768 hash=539eea7de7dd42a6
- `seq.mamba.torch.log`: lane=mamba shape=mamba130m.prefill.t8 tensor=norm.weight n=768 hash=29ffa76f7d830538
- `seq.mamba.torch.log`: lane=mamba shape=mamba130m.prefill.t8 tensor=in_proj.weight n=2359296 hash=3cf6cd86f1f4630b
- `seq.mamba.torch.log`: lane=mamba shape=mamba130m.prefill.t8 tensor=conv1d.weight n=6144 hash=ebc04a57b11317ca
- `seq.mamba.torch.log`: lane=mamba shape=mamba130m.prefill.t8 tensor=conv1d.bias n=1536 hash=addcdfea6820daa4
- `seq.mamba.torch.log`: lane=mamba shape=mamba130m.prefill.t8 tensor=x_proj.weight n=122880 hash=e768eba837f92e18
- `seq.mamba.torch.log`: lane=mamba shape=mamba130m.prefill.t8 tensor=dt_proj.weight n=73728 hash=78915298f8d9d5ee
- `seq.mamba.torch.log`: lane=mamba shape=mamba130m.prefill.t8 tensor=dt_proj.bias n=1536 hash=868ca7bfad1eb09a
- `seq.mamba.torch.log`: lane=mamba shape=mamba130m.prefill.t8 tensor=A_log n=24576 hash=713bed8bffae603b
- `seq.mamba.torch.log`: lane=mamba shape=mamba130m.prefill.t8 tensor=D n=1536 hash=451274e066ddc12a
- `seq.mamba.torch.log`: lane=mamba shape=mamba130m.prefill.t8 tensor=out_proj.weight n=1179648 hash=0d6694627e9ed6f7
- `seq.mamba.torch.log`: lane=mamba shape=mamba130m.prefill.t8 tensor=x n=6144 hash=e7528e4ba876320b
- `seq.mamba.torch.log`: lane=mamba arm=torch-ref-scan-gpu output witness shape=mamba130m.prefill.t8 n=6144 hash=5cbeb28b0ccd30cb
- `seq.mamba.torch.log`: lane=mamba arm=torch-ref-scan-gpu-tf32 output witness shape=mamba130m.prefill.t8 n=6144 hash=178fe924d53298f4
- `seq.mamba.torch.log`: lane=mamba shape=mamba130m.prefill.t128 tensor=norm.weight n=768 hash=4aa921f523b6ac42
- `seq.mamba.torch.log`: lane=mamba shape=mamba130m.prefill.t128 tensor=in_proj.weight n=2359296 hash=eac9458ce9da8173
- `seq.mamba.torch.log`: lane=mamba shape=mamba130m.prefill.t128 tensor=conv1d.weight n=6144 hash=59dfbb854fae9768
- `seq.mamba.torch.log`: lane=mamba shape=mamba130m.prefill.t128 tensor=conv1d.bias n=1536 hash=ee97dbf487ffd3ea
- `seq.mamba.torch.log`: lane=mamba shape=mamba130m.prefill.t128 tensor=x_proj.weight n=122880 hash=1110dc85a856bfa3
- `seq.mamba.torch.log`: lane=mamba shape=mamba130m.prefill.t128 tensor=dt_proj.weight n=73728 hash=55e52d9433c01e4c
- `seq.mamba.torch.log`: lane=mamba shape=mamba130m.prefill.t128 tensor=dt_proj.bias n=1536 hash=8d6ed90034b79bb2
- `seq.mamba.torch.log`: lane=mamba shape=mamba130m.prefill.t128 tensor=A_log n=24576 hash=2c09f76fe78d151e
- `seq.mamba.torch.log`: lane=mamba shape=mamba130m.prefill.t128 tensor=D n=1536 hash=7289bf2a82fc7f6c
- `seq.mamba.torch.log`: lane=mamba shape=mamba130m.prefill.t128 tensor=out_proj.weight n=1179648 hash=5241f52ed48743dd
- `seq.mamba.torch.log`: lane=mamba shape=mamba130m.prefill.t128 tensor=x n=98304 hash=da5b40b810dc107b
- `seq.mamba.torch.log`: lane=mamba arm=torch-ref-scan-gpu output witness shape=mamba130m.prefill.t128 n=98304 hash=51636b64b7a5582b
- `seq.mamba.torch.log`: lane=mamba arm=torch-ref-scan-gpu-tf32 output witness shape=mamba130m.prefill.t128 n=98304 hash=57f31b919c203997
- `seq.mamba.torch.log`: lane=mamba shape=mamba130m.prefill.t512 tensor=norm.weight n=768 hash=f1d1236e08c89e88
- `seq.mamba.torch.log`: lane=mamba shape=mamba130m.prefill.t512 tensor=in_proj.weight n=2359296 hash=bb05213b637f1a4a
- `seq.mamba.torch.log`: lane=mamba shape=mamba130m.prefill.t512 tensor=conv1d.weight n=6144 hash=e9c9a84f1a766ce1
- `seq.mamba.torch.log`: lane=mamba shape=mamba130m.prefill.t512 tensor=conv1d.bias n=1536 hash=651606aae79d5f9d
- `seq.mamba.torch.log`: lane=mamba shape=mamba130m.prefill.t512 tensor=x_proj.weight n=122880 hash=c195cf9bcc2f6792
- `seq.mamba.torch.log`: lane=mamba shape=mamba130m.prefill.t512 tensor=dt_proj.weight n=73728 hash=cd355825a97db4bd
- `seq.mamba.torch.log`: lane=mamba shape=mamba130m.prefill.t512 tensor=dt_proj.bias n=1536 hash=1fa1588a6c1c718d
- `seq.mamba.torch.log`: lane=mamba shape=mamba130m.prefill.t512 tensor=A_log n=24576 hash=6e4e811ed61a3e05
- `seq.mamba.torch.log`: lane=mamba shape=mamba130m.prefill.t512 tensor=D n=1536 hash=db420a6dc68a0c5a
- `seq.mamba.torch.log`: lane=mamba shape=mamba130m.prefill.t512 tensor=out_proj.weight n=1179648 hash=f531e2ff03764cb1
- `seq.mamba.torch.log`: lane=mamba shape=mamba130m.prefill.t512 tensor=x n=393216 hash=c83b7a7b649e48d5
- `seq.mamba.torch.log`: lane=mamba arm=torch-ref-scan-gpu output witness shape=mamba130m.prefill.t512 n=393216 hash=0e016b56c8929328
- `seq.mamba.torch.log`: lane=mamba arm=torch-ref-scan-gpu-tf32 output witness shape=mamba130m.prefill.t512 n=393216 hash=48ab46c3bc2d5bee
- `seq.mamba.torch.log`: lane=mamba arm=torch
- `seq.mlp.ours.log`: lane=mlp arm=ours entry costs the port pays and the opponent does not: llama_refuse_bad_inputs / mamba_refuse_bad_inputs download every weight per call, and eager_attention_forward synchronizes per (batch, head). See bench/speed/SEQ_SPEED.md.
- `seq.mlp.ours.log`: lane=mlp arm=ours PRECISION: FAST routes every GEMM through MAX matmul, which on NVIDIA is a TENSOR-CORE (TF32-class, 10 mantissa bits) path -- measured 200 TFLOP/s against cublas-fp32 44.4 and cublas-tf32 207.5 on an H100. The fair vendor opponent for a lane that routes a GEMM is therefore a TF32 arm (torch-gpu-tf32), NOT an fp32 one, and an FSPEED-AGREE line computed against an fp32 reference is measuring THIS CUT and not a defect. Lanes routing no GEMM (rmsnorm) are unaffected and are the control. DEVIATION 1885.
- `seq.mlp.ours.log`: lane=mlp shape=lane.b2_l4_d32_kv2 tensor=norm1.weight n=32 hash=10553ebe1cacb130
- `seq.mlp.ours.log`: lane=mlp shape=lane.b2_l4_d32_kv2 tensor=norm2.weight n=32 hash=43392443a35b61f3
- `seq.mlp.ours.log`: lane=mlp shape=lane.b2_l4_d32_kv2 tensor=q_proj.weight n=1024 hash=b73a1a48e0c0d9f7
- `seq.mlp.ours.log`: lane=mlp shape=lane.b2_l4_d32_kv2 tensor=k_proj.weight n=1024 hash=de208e4e2a16ff1f
- `seq.mlp.ours.log`: lane=mlp shape=lane.b2_l4_d32_kv2 tensor=v_proj.weight n=1024 hash=427375797d9ef375
- `seq.mlp.ours.log`: lane=mlp shape=lane.b2_l4_d32_kv2 tensor=o_proj.weight n=1024 hash=1947e064f026cb25
- `seq.mlp.ours.log`: lane=mlp shape=lane.b2_l4_d32_kv2 tensor=gate_proj.weight n=2048 hash=284aeae6d05c4f7c
- `seq.mlp.ours.log`: lane=mlp shape=lane.b2_l4_d32_kv2 tensor=up_proj.weight n=2048 hash=b2fdf8deeea86017
- `seq.mlp.ours.log`: lane=mlp shape=lane.b2_l4_d32_kv2 tensor=down_proj.weight n=2048 hash=d1d0e3eb84d5f951
- `seq.mlp.ours.log`: lane=mlp shape=lane.b2_l4_d32_kv2 tensor=x n=256 hash=a55d417a40e61d75
- `seq.mlp.ours.log`: lane=mlp arm=ours llama_refuse_bad_inputs alone shape=lane.b2_l4_d32_kv2 ms=0.357329 (inside the transformer lane's round, not inside attention/mlp/rmsnorm)
- `seq.mlp.ours.log`: lane=mlp arm=ours dump shape=lane.b2_l4_d32_kv2 n=256 path=/root/gemm_leg_out/dump/seq.mlp.lane.b2_l4_d32_kv2.f32.bin
- `seq.mlp.ours.log`: lane=mlp shape=llama8b.prefill.t1 tensor=norm1.weight n=4096 hash=2633ac9556859b7d
- `seq.mlp.ours.log`: lane=mlp shape=llama8b.prefill.t1 tensor=norm2.weight n=4096 hash=03b0bbae422c35f2
- `seq.mlp.ours.log`: lane=mlp shape=llama8b.prefill.t1 tensor=q_proj.weight n=16777216 hash=65536eef389758f8
- `seq.mlp.ours.log`: lane=mlp shape=llama8b.prefill.t1 tensor=k_proj.weight n=4194304 hash=2d21f4a6f190e936
- `seq.mlp.ours.log`: lane=mlp shape=llama8b.prefill.t1 tensor=v_proj.weight n=4194304 hash=2032c9fee9598af5
- `seq.mlp.ours.log`: lane=mlp shape=llama8b.prefill.t1 tensor=o_proj.weight n=16777216 hash=54a671c58956b474
- `seq.mlp.ours.log`: lane=mlp shape=llama8b.prefill.t1 tensor=gate_proj.weight n=58720256 hash=22d060eb830e8fa7
- `seq.mlp.ours.log`: lane=mlp shape=llama8b.prefill.t1 tensor=up_proj.weight n=58720256 hash=6adb4a35fc897af1
- `seq.mlp.ours.log`: lane=mlp shape=llama8b.prefill.t1 tensor=down_proj.weight n=58720256 hash=623f9a160be08bbb
- `seq.mlp.ours.log`: lane=mlp shape=llama8b.prefill.t1 tensor=x n=4096 hash=50601e14d24b1558
- `seq.mlp.ours.log`: lane=mlp arm=ours llama_refuse_bad_inputs alone shape=llama8b.prefill.t1 ms=890.916347 (inside the transformer lane's round, not inside attention/mlp/rmsnorm)
- `seq.mlp.ours.log`: lane=mlp arm=ours dump shape=llama8b.prefill.t1 n=4096 path=/root/gemm_leg_out/dump/seq.mlp.llama8b.prefill.t1.f32.bin
- `seq.mlp.ours.log`: lane=mlp shape=llama8b.prefill.t8 tensor=norm1.weight n=4096 hash=2554a09c79b9bec3
- `seq.mlp.ours.log`: lane=mlp shape=llama8b.prefill.t8 tensor=norm2.weight n=4096 hash=54e8bdb50be343e8
- `seq.mlp.ours.log`: lane=mlp shape=llama8b.prefill.t8 tensor=q_proj.weight n=16777216 hash=7f5e346f0f234ca1
- `seq.mlp.ours.log`: lane=mlp shape=llama8b.prefill.t8 tensor=k_proj.weight n=4194304 hash=628683891b4ef505
- `seq.mlp.ours.log`: lane=mlp shape=llama8b.prefill.t8 tensor=v_proj.weight n=4194304 hash=a6e71091674bf598
- `seq.mlp.ours.log`: lane=mlp shape=llama8b.prefill.t8 tensor=o_proj.weight n=16777216 hash=f322159c55f94428
- `seq.mlp.ours.log`: lane=mlp shape=llama8b.prefill.t8 tensor=gate_proj.weight n=58720256 hash=e4ae7d5f3c49e108
- `seq.mlp.ours.log`: lane=mlp shape=llama8b.prefill.t8 tensor=up_proj.weight n=58720256 hash=e1e76c5d8ad14599
- `seq.mlp.ours.log`: lane=mlp shape=llama8b.prefill.t8 tensor=down_proj.weight n=58720256 hash=cd680ef665764882
- `seq.mlp.ours.log`: lane=mlp shape=llama8b.prefill.t8 tensor=x n=32768 hash=5fae3cdceecdb9e2
- `seq.mlp.ours.log`: lane=mlp arm=ours llama_refuse_bad_inputs alone shape=llama8b.prefill.t8 ms=617.765161 (inside the transformer lane's round, not inside attention/mlp/rmsnorm)
- `seq.mlp.ours.log`: lane=mlp arm=ours dump shape=llama8b.prefill.t8 n=32768 path=/root/gemm_leg_out/dump/seq.mlp.llama8b.prefill.t8.f32.bin
- `seq.mlp.ours.log`: lane=mlp shape=llama8b.prefill.t128 tensor=norm1.weight n=4096 hash=8e82620235b4b085
- `seq.mlp.ours.log`: lane=mlp shape=llama8b.prefill.t128 tensor=norm2.weight n=4096 hash=c813f5e53899180d
- `seq.mlp.ours.log`: lane=mlp shape=llama8b.prefill.t128 tensor=q_proj.weight n=16777216 hash=c1de6ff4941afb8e
- `seq.mlp.ours.log`: lane=mlp shape=llama8b.prefill.t128 tensor=k_proj.weight n=4194304 hash=803e4f6ec477b10e
- `seq.mlp.ours.log`: lane=mlp shape=llama8b.prefill.t128 tensor=v_proj.weight n=4194304 hash=1d4fa08f9f620c73
- `seq.mlp.ours.log`: lane=mlp shape=llama8b.prefill.t128 tensor=o_proj.weight n=16777216 hash=27755bc1220fd650
- `seq.mlp.ours.log`: lane=mlp shape=llama8b.prefill.t128 tensor=gate_proj.weight n=58720256 hash=2cc46a3f987b2a35
- `seq.mlp.ours.log`: lane=mlp shape=llama8b.prefill.t128 tensor=up_proj.weight n=58720256 hash=8df6060ace203936
- `seq.mlp.ours.log`: lane=mlp shape=llama8b.prefill.t128 tensor=down_proj.weight n=58720256 hash=8005f6e866ca1053
- `seq.mlp.ours.log`: lane=mlp shape=llama8b.prefill.t128 tensor=x n=524288 hash=4cfda6cf929d166b
- `seq.mlp.ours.log`: lane=mlp arm=ours llama_refuse_bad_inputs alone shape=llama8b.prefill.t128 ms=638.852796 (inside the transformer lane's round, not inside attention/mlp/rmsnorm)
- `seq.mlp.ours.log`: lane=mlp arm=ours dump shape=llama8b.prefill.t128 n=524288 path=/root/gemm_leg_out/dump/seq.mlp.llama8b.prefill.t128.f32.bin
- `seq.mlp.ours.log`: lane=mlp shape=llama8b.prefill.t512 tensor=norm1.weight n=4096 hash=267ae3b42cf4bd1e
- `seq.mlp.ours.log`: lane=mlp shape=llama8b.prefill.t512 tensor=norm2.weight n=4096 hash=fd8c830453040971
- `seq.mlp.ours.log`: lane=mlp shape=llama8b.prefill.t512 tensor=q_proj.weight n=16777216 hash=40d0cc6a048d905d
- `seq.mlp.ours.log`: lane=mlp shape=llama8b.prefill.t512 tensor=k_proj.weight n=4194304 hash=b5af8e64afe64f60
- `seq.mlp.ours.log`: lane=mlp shape=llama8b.prefill.t512 tensor=v_proj.weight n=4194304 hash=73e1eafa7abcd857
- `seq.mlp.ours.log`: lane=mlp shape=llama8b.prefill.t512 tensor=o_proj.weight n=16777216 hash=f8a091300d6898c7
- `seq.mlp.ours.log`: lane=mlp shape=llama8b.prefill.t512 tensor=gate_proj.weight n=58720256 hash=8b7b13c66c2b3235
- `seq.mlp.ours.log`: lane=mlp shape=llama8b.prefill.t512 tensor=up_proj.weight n=58720256 hash=f61e5f2f2b37b282
- `seq.mlp.ours.log`: lane=mlp shape=llama8b.prefill.t512 tensor=down_proj.weight n=58720256 hash=35adb0b4c5e445b4
- `seq.mlp.ours.log`: lane=mlp shape=llama8b.prefill.t512 tensor=x n=2097152 hash=4dea4dee43008247
- `seq.mlp.ours.log`: lane=mlp arm=ours llama_refuse_bad_inputs alone shape=llama8b.prefill.t512 ms=605.629137 (inside the transformer lane's round, not inside attention/mlp/rmsnorm)
- `seq.mlp.ours.log`: lane=mlp arm=ours dump shape=llama8b.prefill.t512 n=2097152 path=/root/gemm_leg_out/dump/seq.mlp.llama8b.prefill.t512.f32.bin
- `seq.mlp.ours.log`: lane=mlp shape=llama8b.decode.t1.ctx512 tensor=norm1.weight n=4096 hash=a16c25d484966855
- `seq.mlp.ours.log`: lane=mlp shape=llama8b.decode.t1.ctx512 tensor=norm2.weight n=4096 hash=fdf0b3b94538d853
- `seq.mlp.ours.log`: lane=mlp shape=llama8b.decode.t1.ctx512 tensor=q_proj.weight n=16777216 hash=dc126cf683d35c94
- `seq.mlp.ours.log`: lane=mlp shape=llama8b.decode.t1.ctx512 tensor=k_proj.weight n=4194304 hash=9bb615a45cfbb66d
- `seq.mlp.ours.log`: lane=mlp shape=llama8b.decode.t1.ctx512 tensor=v_proj.weight n=4194304 hash=b189eaa5691f2cc2
- `seq.mlp.ours.log`: lane=mlp shape=llama8b.decode.t1.ctx512 tensor=o_proj.weight n=16777216 hash=f460975b0ff8b858
- `seq.mlp.ours.log`: lane=mlp shape=llama8b.decode.t1.ctx512 tensor=gate_proj.weight n=58720256 hash=4b88f0c6df895117
- `seq.mlp.ours.log`: lane=mlp shape=llama8b.decode.t1.ctx512 tensor=up_proj.weight n=58720256 hash=0bcff2e323b1e524
- `seq.mlp.ours.log`: lane=mlp shape=llama8b.decode.t1.ctx512 tensor=down_proj.weight n=58720256 hash=513387fa68b57e5e
- `seq.mlp.ours.log`: lane=mlp shape=llama8b.decode.t1.ctx512 tensor=x n=4096 hash=a1509d4577b1d620
- `seq.mlp.ours.log`: lane=mlp shape=llama8b.decode.t1.ctx512 tensor=ctx.x n=2097152 hash=cf5b380dda81451a
- `seq.mlp.ours.log`: lane=mlp arm=ours llama_refuse_bad_inputs alone shape=llama8b.decode.t1.ctx512 ms=624.844175 (inside the transformer lane's round, not inside attention/mlp/rmsnorm)
- `seq.mlp.ours.log`: lane=mlp arm=ours dump shape=llama8b.decode.t1.ctx512 n=4096 path=/root/gemm_leg_out/dump/seq.mlp.llama8b.decode.t1.ctx512.f32.bin
- `seq.mlp.ours.log`: lane=mlp mode=FAST
- `seq.mlp.torch.log`: lane=mlp arm=torch build=CUDA 12.4 torch=2.4.1+cu124 tf32_switches=cuda.matmul.allow_tf32,cudnn.allow_tf32 deterministic=off
- `seq.mlp.torch.log`: lane=mlp.crosscheck shape=lane.b2_l4_d32_kv2 tensor=norm1.weight n=32 hash=e4ae9bac1e4f2eb0
- `seq.mlp.torch.log`: lane=mlp.crosscheck shape=lane.b2_l4_d32_kv2 tensor=norm2.weight n=32 hash=fe0abe179e07e573
- `seq.mlp.torch.log`: lane=mlp.crosscheck shape=lane.b2_l4_d32_kv2 tensor=q_proj.weight n=1024 hash=0d2da18fc0b40d8b
- `seq.mlp.torch.log`: lane=mlp.crosscheck shape=lane.b2_l4_d32_kv2 tensor=k_proj.weight n=1024 hash=8d6df99dbcd279fb
- `seq.mlp.torch.log`: lane=mlp.crosscheck shape=lane.b2_l4_d32_kv2 tensor=v_proj.weight n=1024 hash=566369012d3485a1
- `seq.mlp.torch.log`: lane=mlp.crosscheck shape=lane.b2_l4_d32_kv2 tensor=o_proj.weight n=1024 hash=d879230290d22ae1
- `seq.mlp.torch.log`: lane=mlp.crosscheck shape=lane.b2_l4_d32_kv2 tensor=gate_proj.weight n=2048 hash=5cd27648cfa7a924
- `seq.mlp.torch.log`: lane=mlp.crosscheck shape=lane.b2_l4_d32_kv2 tensor=up_proj.weight n=2048 hash=86fa1f885662c17f
- `seq.mlp.torch.log`: lane=mlp.crosscheck shape=lane.b2_l4_d32_kv2 tensor=down_proj.weight n=2048 hash=62887e5f6fb89109
- `seq.mlp.torch.log`: lane=mlp.crosscheck shape=lane.b2_l4_d32_kv2 tensor=x n=256 hash=cb5073d347fd2aae
- `seq.mlp.torch.log`: lane=mlp arm=torch crosscheck LlamaEager vs transformer/corpus block_forward at lane.b2_l4_d32_kv2: max_abs_diff=1.5878e-08
- `seq.mlp.torch.log`: lane=mlp shape=lane.b2_l4_d32_kv2 tensor=norm1.weight n=32 hash=e4ae9bac1e4f2eb0
- `seq.mlp.torch.log`: lane=mlp shape=lane.b2_l4_d32_kv2 tensor=norm2.weight n=32 hash=fe0abe179e07e573
- `seq.mlp.torch.log`: lane=mlp shape=lane.b2_l4_d32_kv2 tensor=q_proj.weight n=1024 hash=0d2da18fc0b40d8b
- `seq.mlp.torch.log`: lane=mlp shape=lane.b2_l4_d32_kv2 tensor=k_proj.weight n=1024 hash=8d6df99dbcd279fb
- `seq.mlp.torch.log`: lane=mlp shape=lane.b2_l4_d32_kv2 tensor=v_proj.weight n=1024 hash=566369012d3485a1
- `seq.mlp.torch.log`: lane=mlp shape=lane.b2_l4_d32_kv2 tensor=o_proj.weight n=1024 hash=d879230290d22ae1
- `seq.mlp.torch.log`: lane=mlp shape=lane.b2_l4_d32_kv2 tensor=gate_proj.weight n=2048 hash=5cd27648cfa7a924
- `seq.mlp.torch.log`: lane=mlp shape=lane.b2_l4_d32_kv2 tensor=up_proj.weight n=2048 hash=86fa1f885662c17f
- `seq.mlp.torch.log`: lane=mlp shape=lane.b2_l4_d32_kv2 tensor=down_proj.weight n=2048 hash=62887e5f6fb89109
- `seq.mlp.torch.log`: lane=mlp shape=lane.b2_l4_d32_kv2 tensor=x n=256 hash=cb5073d347fd2aae
- `seq.mlp.torch.log`: lane=mlp arm=torch-gpu-fp32 output witness shape=lane.b2_l4_d32_kv2 n=256 hash=01d13b190e30cb95
- `seq.mlp.torch.log`: lane=mlp arm=torch-gpu-tf32 output witness shape=lane.b2_l4_d32_kv2 n=256 hash=c286305c554c841d
- `seq.mlp.torch.log`: lane=mlp shape=llama8b.prefill.t1 tensor=norm1.weight n=4096 hash=30dc9291bcd74c4d
- `seq.mlp.torch.log`: lane=mlp shape=llama8b.prefill.t1 tensor=norm2.weight n=4096 hash=90de83fb74424482
- `seq.mlp.torch.log`: lane=mlp shape=llama8b.prefill.t1 tensor=q_proj.weight n=16777216 hash=458f23ff554c3cc3
- `seq.mlp.torch.log`: lane=mlp shape=llama8b.prefill.t1 tensor=k_proj.weight n=4194304 hash=76706639c3aa7d56
- `seq.mlp.torch.log`: lane=mlp shape=llama8b.prefill.t1 tensor=v_proj.weight n=4194304 hash=81c62305b8e723d5
- `seq.mlp.torch.log`: lane=mlp shape=llama8b.prefill.t1 tensor=o_proj.weight n=16777216 hash=cffb8b77180868b3
- `seq.mlp.torch.log`: lane=mlp shape=llama8b.prefill.t1 tensor=gate_proj.weight n=58720256 hash=4616212f0e15d8ca
- `seq.mlp.torch.log`: lane=mlp shape=llama8b.prefill.t1 tensor=up_proj.weight n=58720256 hash=3d72973cf9e50898
- `seq.mlp.torch.log`: lane=mlp shape=llama8b.prefill.t1 tensor=down_proj.weight n=58720256 hash=63d7be8563af230a
- `seq.mlp.torch.log`: lane=mlp shape=llama8b.prefill.t1 tensor=x n=4096 hash=3d93aac664c66028
- `seq.mlp.torch.log`: lane=mlp arm=torch-gpu-fp32 output witness shape=llama8b.prefill.t1 n=4096 hash=c23b389714c357e7
- `seq.mlp.torch.log`: lane=mlp arm=torch-gpu-tf32 output witness shape=llama8b.prefill.t1 n=4096 hash=c23b389714c357e7
- `seq.mlp.torch.log`: lane=mlp shape=llama8b.prefill.t8 tensor=norm1.weight n=4096 hash=19770838eb156233
- `seq.mlp.torch.log`: lane=mlp shape=llama8b.prefill.t8 tensor=norm2.weight n=4096 hash=661f9b4826837778
- `seq.mlp.torch.log`: lane=mlp shape=llama8b.prefill.t8 tensor=q_proj.weight n=16777216 hash=9ba40b617f6701c6
- `seq.mlp.torch.log`: lane=mlp shape=llama8b.prefill.t8 tensor=k_proj.weight n=4194304 hash=3fc887e58a86aae5
- `seq.mlp.torch.log`: lane=mlp shape=llama8b.prefill.t8 tensor=v_proj.weight n=4194304 hash=64cb8d74bed4b578
- `seq.mlp.torch.log`: lane=mlp shape=llama8b.prefill.t8 tensor=o_proj.weight n=16777216 hash=5abb6368b7c381b7
- `seq.mlp.torch.log`: lane=mlp shape=llama8b.prefill.t8 tensor=gate_proj.weight n=58720256 hash=40827a19ef224a29
- `seq.mlp.torch.log`: lane=mlp shape=llama8b.prefill.t8 tensor=up_proj.weight n=58720256 hash=b8d683c2de2a8d40
- `seq.mlp.torch.log`: lane=mlp shape=llama8b.prefill.t8 tensor=down_proj.weight n=58720256 hash=be5808bd1301911b
- `seq.mlp.torch.log`: lane=mlp shape=llama8b.prefill.t8 tensor=x n=32768 hash=b393ca1e2e252bc2
- `seq.mlp.torch.log`: lane=mlp arm=torch-gpu-fp32 output witness shape=llama8b.prefill.t8 n=32768 hash=a41f4b32330963bb
- `seq.mlp.torch.log`: lane=mlp arm=torch-gpu-tf32 output witness shape=llama8b.prefill.t8 n=32768 hash=0aaf3a93c21aaef1
- `seq.mlp.torch.log`: lane=mlp shape=llama8b.prefill.t128 tensor=norm1.weight n=4096 hash=2d073c4bf3be4215
- `seq.mlp.torch.log`: lane=mlp shape=llama8b.prefill.t128 tensor=norm2.weight n=4096 hash=8d13de87ae1db05d
- `seq.mlp.torch.log`: lane=mlp shape=llama8b.prefill.t128 tensor=q_proj.weight n=16777216 hash=19976a410e812f3d
- `seq.mlp.torch.log`: lane=mlp shape=llama8b.prefill.t128 tensor=k_proj.weight n=4194304 hash=6eb9a28e49ec40ae
- `seq.mlp.torch.log`: lane=mlp shape=llama8b.prefill.t128 tensor=v_proj.weight n=4194304 hash=91936c2a28335c93
- `seq.mlp.torch.log`: lane=mlp shape=llama8b.prefill.t128 tensor=o_proj.weight n=16777216 hash=33f47dd4c2896a57
- `seq.mlp.torch.log`: lane=mlp shape=llama8b.prefill.t128 tensor=gate_proj.weight n=58720256 hash=00ad980c0db80b74
- `seq.mlp.torch.log`: lane=mlp shape=llama8b.prefill.t128 tensor=up_proj.weight n=58720256 hash=c2002fe051fc2a67
- `seq.mlp.torch.log`: lane=mlp shape=llama8b.prefill.t128 tensor=down_proj.weight n=58720256 hash=fe37d1e5eadac30a
- `seq.mlp.torch.log`: lane=mlp shape=llama8b.prefill.t128 tensor=x n=524288 hash=60e833df1e7657f3
- `seq.mlp.torch.log`: lane=mlp arm=torch-gpu-fp32 output witness shape=llama8b.prefill.t128 n=524288 hash=32d12916a6b81487
- `seq.mlp.torch.log`: lane=mlp arm=torch-gpu-tf32 output witness shape=llama8b.prefill.t128 n=524288 hash=a1801b44b9cfb889
- `seq.mlp.torch.log`: lane=mlp shape=llama8b.prefill.t512 tensor=norm1.weight n=4096 hash=f4a0f716372b58ce
- `seq.mlp.torch.log`: lane=mlp shape=llama8b.prefill.t512 tensor=norm2.weight n=4096 hash=93b3f2cddfd23c01
- `seq.mlp.torch.log`: lane=mlp shape=llama8b.prefill.t512 tensor=q_proj.weight n=16777216 hash=fe1b3ad1dfd94d86
- `seq.mlp.torch.log`: lane=mlp shape=llama8b.prefill.t512 tensor=k_proj.weight n=4194304 hash=9f658c4d5c9859c0
- `seq.mlp.torch.log`: lane=mlp shape=llama8b.prefill.t512 tensor=v_proj.weight n=4194304 hash=286524a647dfa937
- `seq.mlp.torch.log`: lane=mlp shape=llama8b.prefill.t512 tensor=o_proj.weight n=16777216 hash=7721f27f6a6c4ce8
- `seq.mlp.torch.log`: lane=mlp shape=llama8b.prefill.t512 tensor=gate_proj.weight n=58720256 hash=630b3a509bc6f674
- `seq.mlp.torch.log`: lane=mlp shape=llama8b.prefill.t512 tensor=up_proj.weight n=58720256 hash=4cf23bd89c42c8b3
- `seq.mlp.torch.log`: lane=mlp shape=llama8b.prefill.t512 tensor=down_proj.weight n=58720256 hash=5c6bd78157391281
- `seq.mlp.torch.log`: lane=mlp shape=llama8b.prefill.t512 tensor=x n=2097152 hash=9ffa2521076bc8c7
- `seq.mlp.torch.log`: lane=mlp arm=torch-gpu-fp32 output witness shape=llama8b.prefill.t512 n=2097152 hash=a81db66015daf839
- `seq.mlp.torch.log`: lane=mlp arm=torch-gpu-tf32 output witness shape=llama8b.prefill.t512 n=2097152 hash=33b43cf0752a70f0
- `seq.mlp.torch.log`: lane=mlp shape=llama8b.decode.t1.ctx512 tensor=norm1.weight n=4096 hash=939d6972265aa725
- `seq.mlp.torch.log`: lane=mlp shape=llama8b.decode.t1.ctx512 tensor=norm2.weight n=4096 hash=31804b90e76959c3
- `seq.mlp.torch.log`: lane=mlp shape=llama8b.decode.t1.ctx512 tensor=q_proj.weight n=16777216 hash=b1b0a1ba4873ca77
- `seq.mlp.torch.log`: lane=mlp shape=llama8b.decode.t1.ctx512 tensor=k_proj.weight n=4194304 hash=1c16ca3348db5f0d
- `seq.mlp.torch.log`: lane=mlp shape=llama8b.decode.t1.ctx512 tensor=v_proj.weight n=4194304 hash=237925dc612d1c22
- `seq.mlp.torch.log`: lane=mlp shape=llama8b.decode.t1.ctx512 tensor=o_proj.weight n=16777216 hash=0ec594792330189f
- `seq.mlp.torch.log`: lane=mlp shape=llama8b.decode.t1.ctx512 tensor=gate_proj.weight n=58720256 hash=c5c194484347d30a
- `seq.mlp.torch.log`: lane=mlp shape=llama8b.decode.t1.ctx512 tensor=up_proj.weight n=58720256 hash=9374a3fc8f279dd9
- `seq.mlp.torch.log`: lane=mlp shape=llama8b.decode.t1.ctx512 tensor=down_proj.weight n=58720256 hash=6e1a2f6cfa111b57
- `seq.mlp.torch.log`: lane=mlp shape=llama8b.decode.t1.ctx512 tensor=x n=4096 hash=b91b5914caac5a30
- `seq.mlp.torch.log`: lane=mlp shape=llama8b.decode.t1.ctx512 tensor=ctx.x n=2097152 hash=c54a8d22a1bde29a
- `seq.mlp.torch.log`: lane=mlp arm=torch-gpu-fp32 output witness shape=llama8b.decode.t1.ctx512 n=4096 hash=a6524867d4a13e6e
- `seq.mlp.torch.log`: lane=mlp arm=torch-gpu-tf32 output witness shape=llama8b.decode.t1.ctx512 n=4096 hash=c1777403df51bc26
- `seq.mlp.torch.log`: lane=mlp arm=torch
- `seq.rmsnorm.ours.log`: lane=rmsnorm arm=ours entry costs the port pays and the opponent does not: llama_refuse_bad_inputs / mamba_refuse_bad_inputs download every weight per call, and eager_attention_forward synchronizes per (batch, head). See bench/speed/SEQ_SPEED.md.
- `seq.rmsnorm.ours.log`: lane=rmsnorm arm=ours PRECISION: FAST routes every GEMM through MAX matmul, which on NVIDIA is a TENSOR-CORE (TF32-class, 10 mantissa bits) path -- measured 200 TFLOP/s against cublas-fp32 44.4 and cublas-tf32 207.5 on an H100. The fair vendor opponent for a lane that routes a GEMM is therefore a TF32 arm (torch-gpu-tf32), NOT an fp32 one, and an FSPEED-AGREE line computed against an fp32 reference is measuring THIS CUT and not a defect. Lanes routing no GEMM (rmsnorm) are unaffected and are the control. DEVIATION 1885.
- `seq.rmsnorm.ours.log`: lane=rmsnorm shape=lane.b2_l4_d32_kv2 tensor=norm1.weight n=32 hash=10553ebe1cacb130
- `seq.rmsnorm.ours.log`: lane=rmsnorm shape=lane.b2_l4_d32_kv2 tensor=norm2.weight n=32 hash=43392443a35b61f3
- `seq.rmsnorm.ours.log`: lane=rmsnorm shape=lane.b2_l4_d32_kv2 tensor=q_proj.weight n=1024 hash=b73a1a48e0c0d9f7
- `seq.rmsnorm.ours.log`: lane=rmsnorm shape=lane.b2_l4_d32_kv2 tensor=k_proj.weight n=1024 hash=de208e4e2a16ff1f
- `seq.rmsnorm.ours.log`: lane=rmsnorm shape=lane.b2_l4_d32_kv2 tensor=v_proj.weight n=1024 hash=427375797d9ef375
- `seq.rmsnorm.ours.log`: lane=rmsnorm shape=lane.b2_l4_d32_kv2 tensor=o_proj.weight n=1024 hash=1947e064f026cb25
- `seq.rmsnorm.ours.log`: lane=rmsnorm shape=lane.b2_l4_d32_kv2 tensor=gate_proj.weight n=2048 hash=284aeae6d05c4f7c
- `seq.rmsnorm.ours.log`: lane=rmsnorm shape=lane.b2_l4_d32_kv2 tensor=up_proj.weight n=2048 hash=b2fdf8deeea86017
- `seq.rmsnorm.ours.log`: lane=rmsnorm shape=lane.b2_l4_d32_kv2 tensor=down_proj.weight n=2048 hash=d1d0e3eb84d5f951
- `seq.rmsnorm.ours.log`: lane=rmsnorm shape=lane.b2_l4_d32_kv2 tensor=x n=256 hash=a55d417a40e61d75
- `seq.rmsnorm.ours.log`: lane=rmsnorm arm=ours llama_refuse_bad_inputs alone shape=lane.b2_l4_d32_kv2 ms=0.130169 (inside the transformer lane's round, not inside attention/mlp/rmsnorm)
- `seq.rmsnorm.ours.log`: lane=rmsnorm arm=ours dump shape=lane.b2_l4_d32_kv2 n=256 path=/root/gemm_leg_out/dump/seq.rmsnorm.lane.b2_l4_d32_kv2.f32.bin
- `seq.rmsnorm.ours.log`: lane=rmsnorm shape=llama8b.prefill.t1 tensor=norm1.weight n=4096 hash=2633ac9556859b7d
- `seq.rmsnorm.ours.log`: lane=rmsnorm shape=llama8b.prefill.t1 tensor=norm2.weight n=4096 hash=03b0bbae422c35f2
- `seq.rmsnorm.ours.log`: lane=rmsnorm shape=llama8b.prefill.t1 tensor=q_proj.weight n=16777216 hash=65536eef389758f8
- `seq.rmsnorm.ours.log`: lane=rmsnorm shape=llama8b.prefill.t1 tensor=k_proj.weight n=4194304 hash=2d21f4a6f190e936
- `seq.rmsnorm.ours.log`: lane=rmsnorm shape=llama8b.prefill.t1 tensor=v_proj.weight n=4194304 hash=2032c9fee9598af5
- `seq.rmsnorm.ours.log`: lane=rmsnorm shape=llama8b.prefill.t1 tensor=o_proj.weight n=16777216 hash=54a671c58956b474
- `seq.rmsnorm.ours.log`: lane=rmsnorm shape=llama8b.prefill.t1 tensor=gate_proj.weight n=58720256 hash=22d060eb830e8fa7
- `seq.rmsnorm.ours.log`: lane=rmsnorm shape=llama8b.prefill.t1 tensor=up_proj.weight n=58720256 hash=6adb4a35fc897af1
- `seq.rmsnorm.ours.log`: lane=rmsnorm shape=llama8b.prefill.t1 tensor=down_proj.weight n=58720256 hash=623f9a160be08bbb
- `seq.rmsnorm.ours.log`: lane=rmsnorm shape=llama8b.prefill.t1 tensor=x n=4096 hash=50601e14d24b1558
- `seq.rmsnorm.ours.log`: lane=rmsnorm arm=ours llama_refuse_bad_inputs alone shape=llama8b.prefill.t1 ms=907.205096 (inside the transformer lane's round, not inside attention/mlp/rmsnorm)
- `seq.rmsnorm.ours.log`: lane=rmsnorm arm=ours dump shape=llama8b.prefill.t1 n=4096 path=/root/gemm_leg_out/dump/seq.rmsnorm.llama8b.prefill.t1.f32.bin
- `seq.rmsnorm.ours.log`: lane=rmsnorm shape=llama8b.prefill.t8 tensor=norm1.weight n=4096 hash=2554a09c79b9bec3
- `seq.rmsnorm.ours.log`: lane=rmsnorm shape=llama8b.prefill.t8 tensor=norm2.weight n=4096 hash=54e8bdb50be343e8
- `seq.rmsnorm.ours.log`: lane=rmsnorm shape=llama8b.prefill.t8 tensor=q_proj.weight n=16777216 hash=7f5e346f0f234ca1
- `seq.rmsnorm.ours.log`: lane=rmsnorm shape=llama8b.prefill.t8 tensor=k_proj.weight n=4194304 hash=628683891b4ef505
- `seq.rmsnorm.ours.log`: lane=rmsnorm shape=llama8b.prefill.t8 tensor=v_proj.weight n=4194304 hash=a6e71091674bf598
- `seq.rmsnorm.ours.log`: lane=rmsnorm shape=llama8b.prefill.t8 tensor=o_proj.weight n=16777216 hash=f322159c55f94428
- `seq.rmsnorm.ours.log`: lane=rmsnorm shape=llama8b.prefill.t8 tensor=gate_proj.weight n=58720256 hash=e4ae7d5f3c49e108
- `seq.rmsnorm.ours.log`: lane=rmsnorm shape=llama8b.prefill.t8 tensor=up_proj.weight n=58720256 hash=e1e76c5d8ad14599
- `seq.rmsnorm.ours.log`: lane=rmsnorm shape=llama8b.prefill.t8 tensor=down_proj.weight n=58720256 hash=cd680ef665764882
- `seq.rmsnorm.ours.log`: lane=rmsnorm shape=llama8b.prefill.t8 tensor=x n=32768 hash=5fae3cdceecdb9e2
- `seq.rmsnorm.ours.log`: lane=rmsnorm arm=ours llama_refuse_bad_inputs alone shape=llama8b.prefill.t8 ms=626.083343 (inside the transformer lane's round, not inside attention/mlp/rmsnorm)
- `seq.rmsnorm.ours.log`: lane=rmsnorm arm=ours dump shape=llama8b.prefill.t8 n=32768 path=/root/gemm_leg_out/dump/seq.rmsnorm.llama8b.prefill.t8.f32.bin
- `seq.rmsnorm.ours.log`: lane=rmsnorm shape=llama8b.prefill.t128 tensor=norm1.weight n=4096 hash=8e82620235b4b085
- `seq.rmsnorm.ours.log`: lane=rmsnorm shape=llama8b.prefill.t128 tensor=norm2.weight n=4096 hash=c813f5e53899180d
- `seq.rmsnorm.ours.log`: lane=rmsnorm shape=llama8b.prefill.t128 tensor=q_proj.weight n=16777216 hash=c1de6ff4941afb8e
- `seq.rmsnorm.ours.log`: lane=rmsnorm shape=llama8b.prefill.t128 tensor=k_proj.weight n=4194304 hash=803e4f6ec477b10e
- `seq.rmsnorm.ours.log`: lane=rmsnorm shape=llama8b.prefill.t128 tensor=v_proj.weight n=4194304 hash=1d4fa08f9f620c73
- `seq.rmsnorm.ours.log`: lane=rmsnorm shape=llama8b.prefill.t128 tensor=o_proj.weight n=16777216 hash=27755bc1220fd650
- `seq.rmsnorm.ours.log`: lane=rmsnorm shape=llama8b.prefill.t128 tensor=gate_proj.weight n=58720256 hash=2cc46a3f987b2a35
- `seq.rmsnorm.ours.log`: lane=rmsnorm shape=llama8b.prefill.t128 tensor=up_proj.weight n=58720256 hash=8df6060ace203936
- `seq.rmsnorm.ours.log`: lane=rmsnorm shape=llama8b.prefill.t128 tensor=down_proj.weight n=58720256 hash=8005f6e866ca1053
- `seq.rmsnorm.ours.log`: lane=rmsnorm shape=llama8b.prefill.t128 tensor=x n=524288 hash=4cfda6cf929d166b
- `seq.rmsnorm.ours.log`: lane=rmsnorm arm=ours llama_refuse_bad_inputs alone shape=llama8b.prefill.t128 ms=597.460882 (inside the transformer lane's round, not inside attention/mlp/rmsnorm)
- `seq.rmsnorm.ours.log`: lane=rmsnorm arm=ours dump shape=llama8b.prefill.t128 n=524288 path=/root/gemm_leg_out/dump/seq.rmsnorm.llama8b.prefill.t128.f32.bin
- `seq.rmsnorm.ours.log`: lane=rmsnorm shape=llama8b.prefill.t512 tensor=norm1.weight n=4096 hash=267ae3b42cf4bd1e
- `seq.rmsnorm.ours.log`: lane=rmsnorm shape=llama8b.prefill.t512 tensor=norm2.weight n=4096 hash=fd8c830453040971
- `seq.rmsnorm.ours.log`: lane=rmsnorm shape=llama8b.prefill.t512 tensor=q_proj.weight n=16777216 hash=40d0cc6a048d905d
- `seq.rmsnorm.ours.log`: lane=rmsnorm shape=llama8b.prefill.t512 tensor=k_proj.weight n=4194304 hash=b5af8e64afe64f60
- `seq.rmsnorm.ours.log`: lane=rmsnorm shape=llama8b.prefill.t512 tensor=v_proj.weight n=4194304 hash=73e1eafa7abcd857
- `seq.rmsnorm.ours.log`: lane=rmsnorm shape=llama8b.prefill.t512 tensor=o_proj.weight n=16777216 hash=f8a091300d6898c7
- `seq.rmsnorm.ours.log`: lane=rmsnorm shape=llama8b.prefill.t512 tensor=gate_proj.weight n=58720256 hash=8b7b13c66c2b3235
- `seq.rmsnorm.ours.log`: lane=rmsnorm shape=llama8b.prefill.t512 tensor=up_proj.weight n=58720256 hash=f61e5f2f2b37b282
- `seq.rmsnorm.ours.log`: lane=rmsnorm shape=llama8b.prefill.t512 tensor=down_proj.weight n=58720256 hash=35adb0b4c5e445b4
- `seq.rmsnorm.ours.log`: lane=rmsnorm shape=llama8b.prefill.t512 tensor=x n=2097152 hash=4dea4dee43008247
- `seq.rmsnorm.ours.log`: lane=rmsnorm arm=ours llama_refuse_bad_inputs alone shape=llama8b.prefill.t512 ms=618.391854 (inside the transformer lane's round, not inside attention/mlp/rmsnorm)
- `seq.rmsnorm.ours.log`: lane=rmsnorm arm=ours dump shape=llama8b.prefill.t512 n=2097152 path=/root/gemm_leg_out/dump/seq.rmsnorm.llama8b.prefill.t512.f32.bin
- `seq.rmsnorm.ours.log`: lane=rmsnorm shape=llama8b.decode.t1.ctx512 tensor=norm1.weight n=4096 hash=a16c25d484966855
- `seq.rmsnorm.ours.log`: lane=rmsnorm shape=llama8b.decode.t1.ctx512 tensor=norm2.weight n=4096 hash=fdf0b3b94538d853
- `seq.rmsnorm.ours.log`: lane=rmsnorm shape=llama8b.decode.t1.ctx512 tensor=q_proj.weight n=16777216 hash=dc126cf683d35c94
- `seq.rmsnorm.ours.log`: lane=rmsnorm shape=llama8b.decode.t1.ctx512 tensor=k_proj.weight n=4194304 hash=9bb615a45cfbb66d
- `seq.rmsnorm.ours.log`: lane=rmsnorm shape=llama8b.decode.t1.ctx512 tensor=v_proj.weight n=4194304 hash=b189eaa5691f2cc2
- `seq.rmsnorm.ours.log`: lane=rmsnorm shape=llama8b.decode.t1.ctx512 tensor=o_proj.weight n=16777216 hash=f460975b0ff8b858
- `seq.rmsnorm.ours.log`: lane=rmsnorm shape=llama8b.decode.t1.ctx512 tensor=gate_proj.weight n=58720256 hash=4b88f0c6df895117
- `seq.rmsnorm.ours.log`: lane=rmsnorm shape=llama8b.decode.t1.ctx512 tensor=up_proj.weight n=58720256 hash=0bcff2e323b1e524
- `seq.rmsnorm.ours.log`: lane=rmsnorm shape=llama8b.decode.t1.ctx512 tensor=down_proj.weight n=58720256 hash=513387fa68b57e5e
- `seq.rmsnorm.ours.log`: lane=rmsnorm shape=llama8b.decode.t1.ctx512 tensor=x n=4096 hash=a1509d4577b1d620
- `seq.rmsnorm.ours.log`: lane=rmsnorm shape=llama8b.decode.t1.ctx512 tensor=ctx.x n=2097152 hash=cf5b380dda81451a
- `seq.rmsnorm.ours.log`: lane=rmsnorm arm=ours llama_refuse_bad_inputs alone shape=llama8b.decode.t1.ctx512 ms=613.179678 (inside the transformer lane's round, not inside attention/mlp/rmsnorm)
- `seq.rmsnorm.ours.log`: lane=rmsnorm arm=ours dump shape=llama8b.decode.t1.ctx512 n=4096 path=/root/gemm_leg_out/dump/seq.rmsnorm.llama8b.decode.t1.ctx512.f32.bin
- `seq.rmsnorm.ours.log`: lane=rmsnorm mode=FAST
- `seq.rmsnorm.torch.log`: lane=rmsnorm arm=torch build=CUDA 12.4 torch=2.4.1+cu124 tf32_switches=cuda.matmul.allow_tf32,cudnn.allow_tf32 deterministic=off
- `seq.rmsnorm.torch.log`: lane=rmsnorm.crosscheck shape=lane.b2_l4_d32_kv2 tensor=norm1.weight n=32 hash=e4ae9bac1e4f2eb0
- `seq.rmsnorm.torch.log`: lane=rmsnorm.crosscheck shape=lane.b2_l4_d32_kv2 tensor=norm2.weight n=32 hash=fe0abe179e07e573
- `seq.rmsnorm.torch.log`: lane=rmsnorm.crosscheck shape=lane.b2_l4_d32_kv2 tensor=q_proj.weight n=1024 hash=0d2da18fc0b40d8b
- `seq.rmsnorm.torch.log`: lane=rmsnorm.crosscheck shape=lane.b2_l4_d32_kv2 tensor=k_proj.weight n=1024 hash=8d6df99dbcd279fb
- `seq.rmsnorm.torch.log`: lane=rmsnorm.crosscheck shape=lane.b2_l4_d32_kv2 tensor=v_proj.weight n=1024 hash=566369012d3485a1
- `seq.rmsnorm.torch.log`: lane=rmsnorm.crosscheck shape=lane.b2_l4_d32_kv2 tensor=o_proj.weight n=1024 hash=d879230290d22ae1
- `seq.rmsnorm.torch.log`: lane=rmsnorm.crosscheck shape=lane.b2_l4_d32_kv2 tensor=gate_proj.weight n=2048 hash=5cd27648cfa7a924
- `seq.rmsnorm.torch.log`: lane=rmsnorm.crosscheck shape=lane.b2_l4_d32_kv2 tensor=up_proj.weight n=2048 hash=86fa1f885662c17f
- `seq.rmsnorm.torch.log`: lane=rmsnorm.crosscheck shape=lane.b2_l4_d32_kv2 tensor=down_proj.weight n=2048 hash=62887e5f6fb89109
- `seq.rmsnorm.torch.log`: lane=rmsnorm.crosscheck shape=lane.b2_l4_d32_kv2 tensor=x n=256 hash=cb5073d347fd2aae
- `seq.rmsnorm.torch.log`: lane=rmsnorm arm=torch crosscheck LlamaEager vs transformer/corpus block_forward at lane.b2_l4_d32_kv2: max_abs_diff=1.5878e-08
- `seq.rmsnorm.torch.log`: lane=rmsnorm shape=lane.b2_l4_d32_kv2 tensor=norm1.weight n=32 hash=e4ae9bac1e4f2eb0
- `seq.rmsnorm.torch.log`: lane=rmsnorm shape=lane.b2_l4_d32_kv2 tensor=norm2.weight n=32 hash=fe0abe179e07e573
- `seq.rmsnorm.torch.log`: lane=rmsnorm shape=lane.b2_l4_d32_kv2 tensor=q_proj.weight n=1024 hash=0d2da18fc0b40d8b
- `seq.rmsnorm.torch.log`: lane=rmsnorm shape=lane.b2_l4_d32_kv2 tensor=k_proj.weight n=1024 hash=8d6df99dbcd279fb
- `seq.rmsnorm.torch.log`: lane=rmsnorm shape=lane.b2_l4_d32_kv2 tensor=v_proj.weight n=1024 hash=566369012d3485a1
- `seq.rmsnorm.torch.log`: lane=rmsnorm shape=lane.b2_l4_d32_kv2 tensor=o_proj.weight n=1024 hash=d879230290d22ae1
- `seq.rmsnorm.torch.log`: lane=rmsnorm shape=lane.b2_l4_d32_kv2 tensor=gate_proj.weight n=2048 hash=5cd27648cfa7a924
- `seq.rmsnorm.torch.log`: lane=rmsnorm shape=lane.b2_l4_d32_kv2 tensor=up_proj.weight n=2048 hash=86fa1f885662c17f
- `seq.rmsnorm.torch.log`: lane=rmsnorm shape=lane.b2_l4_d32_kv2 tensor=down_proj.weight n=2048 hash=62887e5f6fb89109
- `seq.rmsnorm.torch.log`: lane=rmsnorm shape=lane.b2_l4_d32_kv2 tensor=x n=256 hash=cb5073d347fd2aae
- `seq.rmsnorm.torch.log`: lane=rmsnorm arm=torch-gpu-fp32 output witness shape=lane.b2_l4_d32_kv2 n=256 hash=852ff02726038c82
- `seq.rmsnorm.torch.log`: lane=rmsnorm arm=torch-gpu-tf32 output witness shape=lane.b2_l4_d32_kv2 n=256 hash=852ff02726038c82
- `seq.rmsnorm.torch.log`: lane=rmsnorm shape=llama8b.prefill.t1 tensor=norm1.weight n=4096 hash=30dc9291bcd74c4d
- `seq.rmsnorm.torch.log`: lane=rmsnorm shape=llama8b.prefill.t1 tensor=norm2.weight n=4096 hash=90de83fb74424482
- `seq.rmsnorm.torch.log`: lane=rmsnorm shape=llama8b.prefill.t1 tensor=q_proj.weight n=16777216 hash=458f23ff554c3cc3
- `seq.rmsnorm.torch.log`: lane=rmsnorm shape=llama8b.prefill.t1 tensor=k_proj.weight n=4194304 hash=76706639c3aa7d56
- `seq.rmsnorm.torch.log`: lane=rmsnorm shape=llama8b.prefill.t1 tensor=v_proj.weight n=4194304 hash=81c62305b8e723d5
- `seq.rmsnorm.torch.log`: lane=rmsnorm shape=llama8b.prefill.t1 tensor=o_proj.weight n=16777216 hash=cffb8b77180868b3
- `seq.rmsnorm.torch.log`: lane=rmsnorm shape=llama8b.prefill.t1 tensor=gate_proj.weight n=58720256 hash=4616212f0e15d8ca
- `seq.rmsnorm.torch.log`: lane=rmsnorm shape=llama8b.prefill.t1 tensor=up_proj.weight n=58720256 hash=3d72973cf9e50898
- `seq.rmsnorm.torch.log`: lane=rmsnorm shape=llama8b.prefill.t1 tensor=down_proj.weight n=58720256 hash=63d7be8563af230a
- `seq.rmsnorm.torch.log`: lane=rmsnorm shape=llama8b.prefill.t1 tensor=x n=4096 hash=3d93aac664c66028
- `seq.rmsnorm.torch.log`: lane=rmsnorm arm=torch-gpu-fp32 output witness shape=llama8b.prefill.t1 n=4096 hash=4635a852c2627e50
- `seq.rmsnorm.torch.log`: lane=rmsnorm arm=torch-gpu-tf32 output witness shape=llama8b.prefill.t1 n=4096 hash=4635a852c2627e50
- `seq.rmsnorm.torch.log`: lane=rmsnorm shape=llama8b.prefill.t8 tensor=norm1.weight n=4096 hash=19770838eb156233
- `seq.rmsnorm.torch.log`: lane=rmsnorm shape=llama8b.prefill.t8 tensor=norm2.weight n=4096 hash=661f9b4826837778
- `seq.rmsnorm.torch.log`: lane=rmsnorm shape=llama8b.prefill.t8 tensor=q_proj.weight n=16777216 hash=9ba40b617f6701c6
- `seq.rmsnorm.torch.log`: lane=rmsnorm shape=llama8b.prefill.t8 tensor=k_proj.weight n=4194304 hash=3fc887e58a86aae5
- `seq.rmsnorm.torch.log`: lane=rmsnorm shape=llama8b.prefill.t8 tensor=v_proj.weight n=4194304 hash=64cb8d74bed4b578
- `seq.rmsnorm.torch.log`: lane=rmsnorm shape=llama8b.prefill.t8 tensor=o_proj.weight n=16777216 hash=5abb6368b7c381b7
- `seq.rmsnorm.torch.log`: lane=rmsnorm shape=llama8b.prefill.t8 tensor=gate_proj.weight n=58720256 hash=40827a19ef224a29
- `seq.rmsnorm.torch.log`: lane=rmsnorm shape=llama8b.prefill.t8 tensor=up_proj.weight n=58720256 hash=b8d683c2de2a8d40
- `seq.rmsnorm.torch.log`: lane=rmsnorm shape=llama8b.prefill.t8 tensor=down_proj.weight n=58720256 hash=be5808bd1301911b
- `seq.rmsnorm.torch.log`: lane=rmsnorm shape=llama8b.prefill.t8 tensor=x n=32768 hash=b393ca1e2e252bc2
- `seq.rmsnorm.torch.log`: lane=rmsnorm arm=torch-gpu-fp32 output witness shape=llama8b.prefill.t8 n=32768 hash=93a8ec973a2dd36c
- `seq.rmsnorm.torch.log`: lane=rmsnorm arm=torch-gpu-tf32 output witness shape=llama8b.prefill.t8 n=32768 hash=93a8ec973a2dd36c
- `seq.rmsnorm.torch.log`: lane=rmsnorm shape=llama8b.prefill.t128 tensor=norm1.weight n=4096 hash=2d073c4bf3be4215
- `seq.rmsnorm.torch.log`: lane=rmsnorm shape=llama8b.prefill.t128 tensor=norm2.weight n=4096 hash=8d13de87ae1db05d
- `seq.rmsnorm.torch.log`: lane=rmsnorm shape=llama8b.prefill.t128 tensor=q_proj.weight n=16777216 hash=19976a410e812f3d
- `seq.rmsnorm.torch.log`: lane=rmsnorm shape=llama8b.prefill.t128 tensor=k_proj.weight n=4194304 hash=6eb9a28e49ec40ae
- `seq.rmsnorm.torch.log`: lane=rmsnorm shape=llama8b.prefill.t128 tensor=v_proj.weight n=4194304 hash=91936c2a28335c93
- `seq.rmsnorm.torch.log`: lane=rmsnorm shape=llama8b.prefill.t128 tensor=o_proj.weight n=16777216 hash=33f47dd4c2896a57
- `seq.rmsnorm.torch.log`: lane=rmsnorm shape=llama8b.prefill.t128 tensor=gate_proj.weight n=58720256 hash=00ad980c0db80b74
- `seq.rmsnorm.torch.log`: lane=rmsnorm shape=llama8b.prefill.t128 tensor=up_proj.weight n=58720256 hash=c2002fe051fc2a67
- `seq.rmsnorm.torch.log`: lane=rmsnorm shape=llama8b.prefill.t128 tensor=down_proj.weight n=58720256 hash=fe37d1e5eadac30a
- `seq.rmsnorm.torch.log`: lane=rmsnorm shape=llama8b.prefill.t128 tensor=x n=524288 hash=60e833df1e7657f3
- `seq.rmsnorm.torch.log`: lane=rmsnorm arm=torch-gpu-fp32 output witness shape=llama8b.prefill.t128 n=524288 hash=1fe2e438e7ccdc51
- `seq.rmsnorm.torch.log`: lane=rmsnorm arm=torch-gpu-tf32 output witness shape=llama8b.prefill.t128 n=524288 hash=1fe2e438e7ccdc51
- `seq.rmsnorm.torch.log`: lane=rmsnorm shape=llama8b.prefill.t512 tensor=norm1.weight n=4096 hash=f4a0f716372b58ce
- `seq.rmsnorm.torch.log`: lane=rmsnorm shape=llama8b.prefill.t512 tensor=norm2.weight n=4096 hash=93b3f2cddfd23c01
- `seq.rmsnorm.torch.log`: lane=rmsnorm shape=llama8b.prefill.t512 tensor=q_proj.weight n=16777216 hash=fe1b3ad1dfd94d86
- `seq.rmsnorm.torch.log`: lane=rmsnorm shape=llama8b.prefill.t512 tensor=k_proj.weight n=4194304 hash=9f658c4d5c9859c0
- `seq.rmsnorm.torch.log`: lane=rmsnorm shape=llama8b.prefill.t512 tensor=v_proj.weight n=4194304 hash=286524a647dfa937
- `seq.rmsnorm.torch.log`: lane=rmsnorm shape=llama8b.prefill.t512 tensor=o_proj.weight n=16777216 hash=7721f27f6a6c4ce8
- `seq.rmsnorm.torch.log`: lane=rmsnorm shape=llama8b.prefill.t512 tensor=gate_proj.weight n=58720256 hash=630b3a509bc6f674
- `seq.rmsnorm.torch.log`: lane=rmsnorm shape=llama8b.prefill.t512 tensor=up_proj.weight n=58720256 hash=4cf23bd89c42c8b3
- `seq.rmsnorm.torch.log`: lane=rmsnorm shape=llama8b.prefill.t512 tensor=down_proj.weight n=58720256 hash=5c6bd78157391281
- `seq.rmsnorm.torch.log`: lane=rmsnorm shape=llama8b.prefill.t512 tensor=x n=2097152 hash=9ffa2521076bc8c7
- `seq.rmsnorm.torch.log`: lane=rmsnorm arm=torch-gpu-fp32 output witness shape=llama8b.prefill.t512 n=2097152 hash=65303702246b774b
- `seq.rmsnorm.torch.log`: lane=rmsnorm arm=torch-gpu-tf32 output witness shape=llama8b.prefill.t512 n=2097152 hash=65303702246b774b
- `seq.rmsnorm.torch.log`: lane=rmsnorm shape=llama8b.decode.t1.ctx512 tensor=norm1.weight n=4096 hash=939d6972265aa725
- `seq.rmsnorm.torch.log`: lane=rmsnorm shape=llama8b.decode.t1.ctx512 tensor=norm2.weight n=4096 hash=31804b90e76959c3
- `seq.rmsnorm.torch.log`: lane=rmsnorm shape=llama8b.decode.t1.ctx512 tensor=q_proj.weight n=16777216 hash=b1b0a1ba4873ca77
- `seq.rmsnorm.torch.log`: lane=rmsnorm shape=llama8b.decode.t1.ctx512 tensor=k_proj.weight n=4194304 hash=1c16ca3348db5f0d
- `seq.rmsnorm.torch.log`: lane=rmsnorm shape=llama8b.decode.t1.ctx512 tensor=v_proj.weight n=4194304 hash=237925dc612d1c22
- `seq.rmsnorm.torch.log`: lane=rmsnorm shape=llama8b.decode.t1.ctx512 tensor=o_proj.weight n=16777216 hash=0ec594792330189f
- `seq.rmsnorm.torch.log`: lane=rmsnorm shape=llama8b.decode.t1.ctx512 tensor=gate_proj.weight n=58720256 hash=c5c194484347d30a
- `seq.rmsnorm.torch.log`: lane=rmsnorm shape=llama8b.decode.t1.ctx512 tensor=up_proj.weight n=58720256 hash=9374a3fc8f279dd9
- `seq.rmsnorm.torch.log`: lane=rmsnorm shape=llama8b.decode.t1.ctx512 tensor=down_proj.weight n=58720256 hash=6e1a2f6cfa111b57
- `seq.rmsnorm.torch.log`: lane=rmsnorm shape=llama8b.decode.t1.ctx512 tensor=x n=4096 hash=b91b5914caac5a30
- `seq.rmsnorm.torch.log`: lane=rmsnorm shape=llama8b.decode.t1.ctx512 tensor=ctx.x n=2097152 hash=c54a8d22a1bde29a
- `seq.rmsnorm.torch.log`: lane=rmsnorm arm=torch-gpu-fp32 output witness shape=llama8b.decode.t1.ctx512 n=4096 hash=409660f2f0359908
- `seq.rmsnorm.torch.log`: lane=rmsnorm arm=torch-gpu-tf32 output witness shape=llama8b.decode.t1.ctx512 n=4096 hash=409660f2f0359908
- `seq.rmsnorm.torch.log`: lane=rmsnorm arm=torch
- `seq.selective_scan.ours.log`: lane=selective_scan arm=ours entry costs the port pays and the opponent does not: llama_refuse_bad_inputs / mamba_refuse_bad_inputs download every weight per call, and eager_attention_forward synchronizes per (batch, head). See bench/speed/SEQ_SPEED.md.
- `seq.selective_scan.ours.log`: lane=selective_scan arm=ours PRECISION: FAST routes every GEMM through MAX matmul, which on NVIDIA is a TENSOR-CORE (TF32-class, 10 mantissa bits) path -- measured 200 TFLOP/s against cublas-fp32 44.4 and cublas-tf32 207.5 on an H100. The fair vendor opponent for a lane that routes a GEMM is therefore a TF32 arm (torch-gpu-tf32), NOT an fp32 one, and an FSPEED-AGREE line computed against an fp32 reference is measuring THIS CUT and not a defect. Lanes routing no GEMM (rmsnorm) are unaffected and are the control. DEVIATION 1885.
- `seq.selective_scan.ours.log`: lane=selective_scan shape=lane.b2_l4_d8 tensor=norm.weight n=8 hash=50c65afc06287cd7
- `seq.selective_scan.ours.log`: lane=selective_scan shape=lane.b2_l4_d8 tensor=in_proj.weight n=256 hash=e08dc872b261976f
- `seq.selective_scan.ours.log`: lane=selective_scan shape=lane.b2_l4_d8 tensor=conv1d.weight n=64 hash=87567a7cd21ae244
- `seq.selective_scan.ours.log`: lane=selective_scan shape=lane.b2_l4_d8 tensor=conv1d.bias n=16 hash=fffb5b6163f3b167
- `seq.selective_scan.ours.log`: lane=selective_scan shape=lane.b2_l4_d8 tensor=x_proj.weight n=528 hash=e6f1d28bec911569
- `seq.selective_scan.ours.log`: lane=selective_scan shape=lane.b2_l4_d8 tensor=dt_proj.weight n=16 hash=ff1a20b85cbe6334
- `seq.selective_scan.ours.log`: lane=selective_scan shape=lane.b2_l4_d8 tensor=dt_proj.bias n=16 hash=f9fd9e3769ba990f
- `seq.selective_scan.ours.log`: lane=selective_scan shape=lane.b2_l4_d8 tensor=A_log n=256 hash=fd875e94f94076a3
- `seq.selective_scan.ours.log`: lane=selective_scan shape=lane.b2_l4_d8 tensor=D n=16 hash=427cb25db82f4bd0
- `seq.selective_scan.ours.log`: lane=selective_scan shape=lane.b2_l4_d8 tensor=out_proj.weight n=128 hash=9fce64d8212a8eec
- `seq.selective_scan.ours.log`: lane=selective_scan shape=lane.b2_l4_d8 tensor=x n=64 hash=622c7d6b00d2329d
- `seq.selective_scan.ours.log`: lane=selective_scan arm=ours mamba_refuse_bad_inputs alone shape=lane.b2_l4_d8 ms=0.050034 (inside the mamba lane's round, not inside selective_scan)
- `seq.selective_scan.ours.log`: lane=selective_scan arm=ours dump shape=lane.b2_l4_d8 n=128 path=/root/gemm_leg_out/dump/seq.selective_scan.lane.b2_l4_d8.f32.bin
- `seq.selective_scan.ours.log`: lane=selective_scan shape=mamba130m.prefill.t1 tensor=norm.weight n=768 hash=80ffb6eceac983c1
- `seq.selective_scan.ours.log`: lane=selective_scan shape=mamba130m.prefill.t1 tensor=in_proj.weight n=2359296 hash=9a4025f0cad505d9
- `seq.selective_scan.ours.log`: lane=selective_scan shape=mamba130m.prefill.t1 tensor=conv1d.weight n=6144 hash=4f37ca5c86596e26
- `seq.selective_scan.ours.log`: lane=selective_scan shape=mamba130m.prefill.t1 tensor=conv1d.bias n=1536 hash=3730d3d1a4bd30d5
- `seq.selective_scan.ours.log`: lane=selective_scan shape=mamba130m.prefill.t1 tensor=x_proj.weight n=122880 hash=d8becf46fdf2b30f
- `seq.selective_scan.ours.log`: lane=selective_scan shape=mamba130m.prefill.t1 tensor=dt_proj.weight n=73728 hash=3333a7a393aaf4c1
- `seq.selective_scan.ours.log`: lane=selective_scan shape=mamba130m.prefill.t1 tensor=dt_proj.bias n=1536 hash=670910bd72deb317
- `seq.selective_scan.ours.log`: lane=selective_scan shape=mamba130m.prefill.t1 tensor=A_log n=24576 hash=30d9abc8d0f88426
- `seq.selective_scan.ours.log`: lane=selective_scan shape=mamba130m.prefill.t1 tensor=D n=1536 hash=4af9cecf78461e69
- `seq.selective_scan.ours.log`: lane=selective_scan shape=mamba130m.prefill.t1 tensor=out_proj.weight n=1179648 hash=03bc8590d6d08f9f
- `seq.selective_scan.ours.log`: lane=selective_scan shape=mamba130m.prefill.t1 tensor=x n=768 hash=5832c6c66c7945eb
- `seq.selective_scan.ours.log`: lane=selective_scan arm=ours mamba_refuse_bad_inputs alone shape=mamba130m.prefill.t1 ms=0.0865 (inside the mamba lane's round, not inside selective_scan)
- `seq.selective_scan.ours.log`: lane=selective_scan arm=ours dump shape=mamba130m.prefill.t1 n=1536 path=/root/gemm_leg_out/dump/seq.selective_scan.mamba130m.prefill.t1.f32.bin
- `seq.selective_scan.ours.log`: lane=selective_scan shape=mamba130m.prefill.t8 tensor=norm.weight n=768 hash=8ca9a62b03bb0b69
- `seq.selective_scan.ours.log`: lane=selective_scan shape=mamba130m.prefill.t8 tensor=in_proj.weight n=2359296 hash=e5e828782fc4c377
- `seq.selective_scan.ours.log`: lane=selective_scan shape=mamba130m.prefill.t8 tensor=conv1d.weight n=6144 hash=1fe9d376751939c2
- `seq.selective_scan.ours.log`: lane=selective_scan shape=mamba130m.prefill.t8 tensor=conv1d.bias n=1536 hash=284b95c47159d2e2
- `seq.selective_scan.ours.log`: lane=selective_scan shape=mamba130m.prefill.t8 tensor=x_proj.weight n=122880 hash=58fc41a3c66083f9
- `seq.selective_scan.ours.log`: lane=selective_scan shape=mamba130m.prefill.t8 tensor=dt_proj.weight n=73728 hash=354e3487a0f90c0b
- `seq.selective_scan.ours.log`: lane=selective_scan shape=mamba130m.prefill.t8 tensor=dt_proj.bias n=1536 hash=1729f7678a315600
- `seq.selective_scan.ours.log`: lane=selective_scan shape=mamba130m.prefill.t8 tensor=A_log n=24576 hash=71d966732193f1fb
- `seq.selective_scan.ours.log`: lane=selective_scan shape=mamba130m.prefill.t8 tensor=D n=1536 hash=146d24138daf5ed0
- `seq.selective_scan.ours.log`: lane=selective_scan shape=mamba130m.prefill.t8 tensor=out_proj.weight n=1179648 hash=5f7e08294f2017f1
- `seq.selective_scan.ours.log`: lane=selective_scan shape=mamba130m.prefill.t8 tensor=x n=6144 hash=aea97b3003677e03
- `seq.selective_scan.ours.log`: lane=selective_scan arm=ours mamba_refuse_bad_inputs alone shape=mamba130m.prefill.t8 ms=0.101533 (inside the mamba lane's round, not inside selective_scan)
- `seq.selective_scan.ours.log`: lane=selective_scan arm=ours dump shape=mamba130m.prefill.t8 n=12288 path=/root/gemm_leg_out/dump/seq.selective_scan.mamba130m.prefill.t8.f32.bin
- `seq.selective_scan.ours.log`: lane=selective_scan shape=mamba130m.prefill.t128 tensor=norm.weight n=768 hash=9d0c85053f9dbe1b
- `seq.selective_scan.ours.log`: lane=selective_scan shape=mamba130m.prefill.t128 tensor=in_proj.weight n=2359296 hash=1a8a102874a8f357
- `seq.selective_scan.ours.log`: lane=selective_scan shape=mamba130m.prefill.t128 tensor=conv1d.weight n=6144 hash=0e615d0b95e74f90
- `seq.selective_scan.ours.log`: lane=selective_scan shape=mamba130m.prefill.t128 tensor=conv1d.bias n=1536 hash=87a4bee5bec76720
- `seq.selective_scan.ours.log`: lane=selective_scan shape=mamba130m.prefill.t128 tensor=x_proj.weight n=122880 hash=24305c34fd48b3a6
- `seq.selective_scan.ours.log`: lane=selective_scan shape=mamba130m.prefill.t128 tensor=dt_proj.weight n=73728 hash=6b5d720462192045
- `seq.selective_scan.ours.log`: lane=selective_scan shape=mamba130m.prefill.t128 tensor=dt_proj.bias n=1536 hash=68b70160bd63976c
- `seq.selective_scan.ours.log`: lane=selective_scan shape=mamba130m.prefill.t128 tensor=A_log n=24576 hash=4d659006cfa457de
- `seq.selective_scan.ours.log`: lane=selective_scan shape=mamba130m.prefill.t128 tensor=D n=1536 hash=1f5fdef06a2e6f1e
- `seq.selective_scan.ours.log`: lane=selective_scan shape=mamba130m.prefill.t128 tensor=out_proj.weight n=1179648 hash=be795e26211acaff
- `seq.selective_scan.ours.log`: lane=selective_scan shape=mamba130m.prefill.t128 tensor=x n=98304 hash=25fed7518eb79bb2
- `seq.selective_scan.ours.log`: lane=selective_scan arm=ours mamba_refuse_bad_inputs alone shape=mamba130m.prefill.t128 ms=0.284653 (inside the mamba lane's round, not inside selective_scan)
- `seq.selective_scan.ours.log`: lane=selective_scan arm=ours dump shape=mamba130m.prefill.t128 n=196608 path=/root/gemm_leg_out/dump/seq.selective_scan.mamba130m.prefill.t128.f32.bin
- `seq.selective_scan.ours.log`: lane=selective_scan shape=mamba130m.prefill.t512 tensor=norm.weight n=768 hash=8a1f60d0ecc7f985
- `seq.selective_scan.ours.log`: lane=selective_scan shape=mamba130m.prefill.t512 tensor=in_proj.weight n=2359296 hash=93cf34e9b609c366
- `seq.selective_scan.ours.log`: lane=selective_scan shape=mamba130m.prefill.t512 tensor=conv1d.weight n=6144 hash=f2f7fd124b119079
- `seq.selective_scan.ours.log`: lane=selective_scan shape=mamba130m.prefill.t512 tensor=conv1d.bias n=1536 hash=13b6f9e0acc0cdc3
- `seq.selective_scan.ours.log`: lane=selective_scan shape=mamba130m.prefill.t512 tensor=x_proj.weight n=122880 hash=858ca82562b215ab
- `seq.selective_scan.ours.log`: lane=selective_scan shape=mamba130m.prefill.t512 tensor=dt_proj.weight n=73728 hash=099f48c5ccdb6f44
- `seq.selective_scan.ours.log`: lane=selective_scan shape=mamba130m.prefill.t512 tensor=dt_proj.bias n=1536 hash=8bc414b2e6db2757
- `seq.selective_scan.ours.log`: lane=selective_scan shape=mamba130m.prefill.t512 tensor=A_log n=24576 hash=0895a62212804fc5
- `seq.selective_scan.ours.log`: lane=selective_scan shape=mamba130m.prefill.t512 tensor=D n=1536 hash=d5239bfe63a8472c
- `seq.selective_scan.ours.log`: lane=selective_scan shape=mamba130m.prefill.t512 tensor=out_proj.weight n=1179648 hash=b41416f53f8b72c7
- `seq.selective_scan.ours.log`: lane=selective_scan shape=mamba130m.prefill.t512 tensor=x n=393216 hash=baa7b8e45c791eef
- `seq.selective_scan.ours.log`: lane=selective_scan arm=ours mamba_refuse_bad_inputs alone shape=mamba130m.prefill.t512 ms=0.956685 (inside the mamba lane's round, not inside selective_scan)
- `seq.selective_scan.ours.log`: lane=selective_scan arm=ours dump shape=mamba130m.prefill.t512 n=786432 path=/root/gemm_leg_out/dump/seq.selective_scan.mamba130m.prefill.t512.f32.bin
- `seq.selective_scan.ours.log`: lane=selective_scan mode=FAST
- `seq.selective_scan.torch.log`: lane=selective_scan arm=torch build=CUDA 12.4 torch=2.4.1+cu124 tf32_switches=cuda.matmul.allow_tf32,cudnn.allow_tf32 deterministic=off
- `seq.selective_scan.torch.log`: lane=selective_scan shape=lane.b2_l4_d8 tensor=norm.weight n=8 hash=82d54a938d591baf
- `seq.selective_scan.torch.log`: lane=selective_scan shape=lane.b2_l4_d8 tensor=in_proj.weight n=256 hash=f22fc6a25b5a0910
- `seq.selective_scan.torch.log`: lane=selective_scan shape=lane.b2_l4_d8 tensor=conv1d.weight n=64 hash=90cc6f6dd561f6e4
- `seq.selective_scan.torch.log`: lane=selective_scan shape=lane.b2_l4_d8 tensor=conv1d.bias n=16 hash=edb3d28d0d124257
- `seq.selective_scan.torch.log`: lane=selective_scan shape=lane.b2_l4_d8 tensor=x_proj.weight n=528 hash=7e44686404574eaf
- `seq.selective_scan.torch.log`: lane=selective_scan shape=lane.b2_l4_d8 tensor=dt_proj.weight n=16 hash=6c1b908f90a2f0e4
- `seq.selective_scan.torch.log`: lane=selective_scan shape=lane.b2_l4_d8 tensor=dt_proj.bias n=16 hash=b95295e1de8ff21f
- `seq.selective_scan.torch.log`: lane=selective_scan shape=lane.b2_l4_d8 tensor=A_log n=256 hash=3075ef7a49146f3c
- `seq.selective_scan.torch.log`: lane=selective_scan shape=lane.b2_l4_d8 tensor=D n=16 hash=0b155ba1ec3dd1e0
- `seq.selective_scan.torch.log`: lane=selective_scan shape=lane.b2_l4_d8 tensor=out_proj.weight n=128 hash=bbf7e7dac9bfaacc
- `seq.selective_scan.torch.log`: lane=selective_scan shape=lane.b2_l4_d8 tensor=x n=64 hash=8ee57faefcc7733d
- `seq.selective_scan.torch.log`: lane=selective_scan arm=torch-ref-scan-gpu output witness shape=lane.b2_l4_d8 n=128 hash=1109a9d621ae1ff1
- `seq.selective_scan.torch.log`: lane=selective_scan arm=torch-ref-scan-gpu-tf32 output witness shape=lane.b2_l4_d8 n=128 hash=f3cb0e7ca459e9e8
- `seq.selective_scan.torch.log`: lane=selective_scan shape=mamba130m.prefill.t1 tensor=norm.weight n=768 hash=935bef95533ae1ec
- `seq.selective_scan.torch.log`: lane=selective_scan shape=mamba130m.prefill.t1 tensor=in_proj.weight n=2359296 hash=3b23ea144a46a005
- `seq.selective_scan.torch.log`: lane=selective_scan shape=mamba130m.prefill.t1 tensor=conv1d.weight n=6144 hash=35f623af5f3ba70e
- `seq.selective_scan.torch.log`: lane=selective_scan shape=mamba130m.prefill.t1 tensor=conv1d.bias n=1536 hash=90da985da6eb43d3
- `seq.selective_scan.torch.log`: lane=selective_scan shape=mamba130m.prefill.t1 tensor=x_proj.weight n=122880 hash=5f14325f2227fe36
- `seq.selective_scan.torch.log`: lane=selective_scan shape=mamba130m.prefill.t1 tensor=dt_proj.weight n=73728 hash=dd4c6a116a702514
- `seq.selective_scan.torch.log`: lane=selective_scan shape=mamba130m.prefill.t1 tensor=dt_proj.bias n=1536 hash=d9646bbf3578e7e5
- `seq.selective_scan.torch.log`: lane=selective_scan shape=mamba130m.prefill.t1 tensor=A_log n=24576 hash=60f6da4a1075bf66
- `seq.selective_scan.torch.log`: lane=selective_scan shape=mamba130m.prefill.t1 tensor=D n=1536 hash=98eff32333cba8fb
- `seq.selective_scan.torch.log`: lane=selective_scan shape=mamba130m.prefill.t1 tensor=out_proj.weight n=1179648 hash=056439617b9552d1
- `seq.selective_scan.torch.log`: lane=selective_scan shape=mamba130m.prefill.t1 tensor=x n=768 hash=9628d92b7a03fefa
- `seq.selective_scan.torch.log`: lane=selective_scan arm=torch-ref-scan-gpu output witness shape=mamba130m.prefill.t1 n=1536 hash=6ab7df9710724e0a
- `seq.selective_scan.torch.log`: lane=selective_scan arm=torch-ref-scan-gpu-tf32 output witness shape=mamba130m.prefill.t1 n=1536 hash=6ab7df9710724e0a
- `seq.selective_scan.torch.log`: lane=selective_scan shape=mamba130m.prefill.t8 tensor=norm.weight n=768 hash=29ffa76f7d830538
- `seq.selective_scan.torch.log`: lane=selective_scan shape=mamba130m.prefill.t8 tensor=in_proj.weight n=2359296 hash=3cf6cd86f1f4630b
- `seq.selective_scan.torch.log`: lane=selective_scan shape=mamba130m.prefill.t8 tensor=conv1d.weight n=6144 hash=ebc04a57b11317ca
- `seq.selective_scan.torch.log`: lane=selective_scan shape=mamba130m.prefill.t8 tensor=conv1d.bias n=1536 hash=addcdfea6820daa4
- `seq.selective_scan.torch.log`: lane=selective_scan shape=mamba130m.prefill.t8 tensor=x_proj.weight n=122880 hash=e768eba837f92e18
- `seq.selective_scan.torch.log`: lane=selective_scan shape=mamba130m.prefill.t8 tensor=dt_proj.weight n=73728 hash=78915298f8d9d5ee
- `seq.selective_scan.torch.log`: lane=selective_scan shape=mamba130m.prefill.t8 tensor=dt_proj.bias n=1536 hash=868ca7bfad1eb09a
- `seq.selective_scan.torch.log`: lane=selective_scan shape=mamba130m.prefill.t8 tensor=A_log n=24576 hash=713bed8bffae603b
- `seq.selective_scan.torch.log`: lane=selective_scan shape=mamba130m.prefill.t8 tensor=D n=1536 hash=451274e066ddc12a
- `seq.selective_scan.torch.log`: lane=selective_scan shape=mamba130m.prefill.t8 tensor=out_proj.weight n=1179648 hash=0d6694627e9ed6f7
- `seq.selective_scan.torch.log`: lane=selective_scan shape=mamba130m.prefill.t8 tensor=x n=6144 hash=e7528e4ba876320b
- `seq.selective_scan.torch.log`: lane=selective_scan arm=torch-ref-scan-gpu output witness shape=mamba130m.prefill.t8 n=12288 hash=2858e3bfa6b768e2
- `seq.selective_scan.torch.log`: lane=selective_scan arm=torch-ref-scan-gpu-tf32 output witness shape=mamba130m.prefill.t8 n=12288 hash=26a4e1e9135ca0fa
- `seq.selective_scan.torch.log`: lane=selective_scan shape=mamba130m.prefill.t128 tensor=norm.weight n=768 hash=4aa921f523b6ac42
- `seq.selective_scan.torch.log`: lane=selective_scan shape=mamba130m.prefill.t128 tensor=in_proj.weight n=2359296 hash=eac9458ce9da8173
- `seq.selective_scan.torch.log`: lane=selective_scan shape=mamba130m.prefill.t128 tensor=conv1d.weight n=6144 hash=59dfbb854fae9768
- `seq.selective_scan.torch.log`: lane=selective_scan shape=mamba130m.prefill.t128 tensor=conv1d.bias n=1536 hash=ee97dbf487ffd3ea
- `seq.selective_scan.torch.log`: lane=selective_scan shape=mamba130m.prefill.t128 tensor=x_proj.weight n=122880 hash=1110dc85a856bfa3
- `seq.selective_scan.torch.log`: lane=selective_scan shape=mamba130m.prefill.t128 tensor=dt_proj.weight n=73728 hash=55e52d9433c01e4c
- `seq.selective_scan.torch.log`: lane=selective_scan shape=mamba130m.prefill.t128 tensor=dt_proj.bias n=1536 hash=8d6ed90034b79bb2
- `seq.selective_scan.torch.log`: lane=selective_scan shape=mamba130m.prefill.t128 tensor=A_log n=24576 hash=2c09f76fe78d151e
- `seq.selective_scan.torch.log`: lane=selective_scan shape=mamba130m.prefill.t128 tensor=D n=1536 hash=7289bf2a82fc7f6c
- `seq.selective_scan.torch.log`: lane=selective_scan shape=mamba130m.prefill.t128 tensor=out_proj.weight n=1179648 hash=5241f52ed48743dd
- `seq.selective_scan.torch.log`: lane=selective_scan shape=mamba130m.prefill.t128 tensor=x n=98304 hash=da5b40b810dc107b
- `seq.selective_scan.torch.log`: lane=selective_scan arm=torch-ref-scan-gpu output witness shape=mamba130m.prefill.t128 n=196608 hash=a696161c090fb38e
- `seq.selective_scan.torch.log`: lane=selective_scan arm=torch-ref-scan-gpu-tf32 output witness shape=mamba130m.prefill.t128 n=196608 hash=e8efa81d2a691546
- `seq.selective_scan.torch.log`: lane=selective_scan shape=mamba130m.prefill.t512 tensor=norm.weight n=768 hash=f1d1236e08c89e88
- `seq.selective_scan.torch.log`: lane=selective_scan shape=mamba130m.prefill.t512 tensor=in_proj.weight n=2359296 hash=bb05213b637f1a4a
- `seq.selective_scan.torch.log`: lane=selective_scan shape=mamba130m.prefill.t512 tensor=conv1d.weight n=6144 hash=e9c9a84f1a766ce1
- `seq.selective_scan.torch.log`: lane=selective_scan shape=mamba130m.prefill.t512 tensor=conv1d.bias n=1536 hash=651606aae79d5f9d
- `seq.selective_scan.torch.log`: lane=selective_scan shape=mamba130m.prefill.t512 tensor=x_proj.weight n=122880 hash=c195cf9bcc2f6792
- `seq.selective_scan.torch.log`: lane=selective_scan shape=mamba130m.prefill.t512 tensor=dt_proj.weight n=73728 hash=cd355825a97db4bd
- `seq.selective_scan.torch.log`: lane=selective_scan shape=mamba130m.prefill.t512 tensor=dt_proj.bias n=1536 hash=1fa1588a6c1c718d
- `seq.selective_scan.torch.log`: lane=selective_scan shape=mamba130m.prefill.t512 tensor=A_log n=24576 hash=6e4e811ed61a3e05
- `seq.selective_scan.torch.log`: lane=selective_scan shape=mamba130m.prefill.t512 tensor=D n=1536 hash=db420a6dc68a0c5a
- `seq.selective_scan.torch.log`: lane=selective_scan shape=mamba130m.prefill.t512 tensor=out_proj.weight n=1179648 hash=f531e2ff03764cb1
- `seq.selective_scan.torch.log`: lane=selective_scan shape=mamba130m.prefill.t512 tensor=x n=393216 hash=c83b7a7b649e48d5
- `seq.selective_scan.torch.log`: lane=selective_scan arm=torch-ref-scan-gpu output witness shape=mamba130m.prefill.t512 n=786432 hash=36307548f826be13
- `seq.selective_scan.torch.log`: lane=selective_scan arm=torch-ref-scan-gpu-tf32 output witness shape=mamba130m.prefill.t512 n=786432 hash=ffa0dd24c2a21877
- `seq.selective_scan.torch.log`: lane=selective_scan arm=torch
- `seq.transformer.ours.log`: lane=transformer arm=ours entry costs the port pays and the opponent does not: llama_refuse_bad_inputs / mamba_refuse_bad_inputs download every weight per call, and eager_attention_forward synchronizes per (batch, head). See bench/speed/SEQ_SPEED.md.
- `seq.transformer.ours.log`: lane=transformer arm=ours PRECISION: FAST routes every GEMM through MAX matmul, which on NVIDIA is a TENSOR-CORE (TF32-class, 10 mantissa bits) path -- measured 200 TFLOP/s against cublas-fp32 44.4 and cublas-tf32 207.5 on an H100. The fair vendor opponent for a lane that routes a GEMM is therefore a TF32 arm (torch-gpu-tf32), NOT an fp32 one, and an FSPEED-AGREE line computed against an fp32 reference is measuring THIS CUT and not a defect. Lanes routing no GEMM (rmsnorm) are unaffected and are the control. DEVIATION 1885.
- `seq.transformer.ours.log`: lane=transformer shape=lane.b2_l4_d32_kv2 tensor=norm1.weight n=32 hash=10553ebe1cacb130
- `seq.transformer.ours.log`: lane=transformer shape=lane.b2_l4_d32_kv2 tensor=norm2.weight n=32 hash=43392443a35b61f3
- `seq.transformer.ours.log`: lane=transformer shape=lane.b2_l4_d32_kv2 tensor=q_proj.weight n=1024 hash=b73a1a48e0c0d9f7
- `seq.transformer.ours.log`: lane=transformer shape=lane.b2_l4_d32_kv2 tensor=k_proj.weight n=1024 hash=de208e4e2a16ff1f
- `seq.transformer.ours.log`: lane=transformer shape=lane.b2_l4_d32_kv2 tensor=v_proj.weight n=1024 hash=427375797d9ef375
- `seq.transformer.ours.log`: lane=transformer shape=lane.b2_l4_d32_kv2 tensor=o_proj.weight n=1024 hash=1947e064f026cb25
- `seq.transformer.ours.log`: lane=transformer shape=lane.b2_l4_d32_kv2 tensor=gate_proj.weight n=2048 hash=284aeae6d05c4f7c
- `seq.transformer.ours.log`: lane=transformer shape=lane.b2_l4_d32_kv2 tensor=up_proj.weight n=2048 hash=b2fdf8deeea86017
- `seq.transformer.ours.log`: lane=transformer shape=lane.b2_l4_d32_kv2 tensor=down_proj.weight n=2048 hash=d1d0e3eb84d5f951
- `seq.transformer.ours.log`: lane=transformer shape=lane.b2_l4_d32_kv2 tensor=x n=256 hash=a55d417a40e61d75
- `seq.transformer.ours.log`: lane=transformer arm=ours llama_refuse_bad_inputs alone shape=lane.b2_l4_d32_kv2 ms=0.125921 (inside the transformer lane's round, not inside attention/mlp/rmsnorm)
- `seq.transformer.ours.log`: lane=transformer arm=ours dump shape=lane.b2_l4_d32_kv2 n=256 path=/root/gemm_leg_out/dump/seq.transformer.lane.b2_l4_d32_kv2.f32.bin
- `seq.transformer.ours.log`: lane=transformer shape=llama8b.prefill.t1 tensor=norm1.weight n=4096 hash=2633ac9556859b7d
- `seq.transformer.ours.log`: lane=transformer shape=llama8b.prefill.t1 tensor=norm2.weight n=4096 hash=03b0bbae422c35f2
- `seq.transformer.ours.log`: lane=transformer shape=llama8b.prefill.t1 tensor=q_proj.weight n=16777216 hash=65536eef389758f8
- `seq.transformer.ours.log`: lane=transformer shape=llama8b.prefill.t1 tensor=k_proj.weight n=4194304 hash=2d21f4a6f190e936
- `seq.transformer.ours.log`: lane=transformer shape=llama8b.prefill.t1 tensor=v_proj.weight n=4194304 hash=2032c9fee9598af5
- `seq.transformer.ours.log`: lane=transformer shape=llama8b.prefill.t1 tensor=o_proj.weight n=16777216 hash=54a671c58956b474
- `seq.transformer.ours.log`: lane=transformer shape=llama8b.prefill.t1 tensor=gate_proj.weight n=58720256 hash=22d060eb830e8fa7
- `seq.transformer.ours.log`: lane=transformer shape=llama8b.prefill.t1 tensor=up_proj.weight n=58720256 hash=6adb4a35fc897af1
- `seq.transformer.ours.log`: lane=transformer shape=llama8b.prefill.t1 tensor=down_proj.weight n=58720256 hash=623f9a160be08bbb
- `seq.transformer.ours.log`: lane=transformer shape=llama8b.prefill.t1 tensor=x n=4096 hash=50601e14d24b1558
- `seq.transformer.ours.log`: lane=transformer arm=ours llama_refuse_bad_inputs alone shape=llama8b.prefill.t1 ms=964.776748 (inside the transformer lane's round, not inside attention/mlp/rmsnorm)
- `seq.transformer.ours.log`: lane=transformer arm=ours dump shape=llama8b.prefill.t1 n=4096 path=/root/gemm_leg_out/dump/seq.transformer.llama8b.prefill.t1.f32.bin
- `seq.transformer.ours.log`: lane=transformer shape=llama8b.prefill.t8 tensor=norm1.weight n=4096 hash=2554a09c79b9bec3
- `seq.transformer.ours.log`: lane=transformer shape=llama8b.prefill.t8 tensor=norm2.weight n=4096 hash=54e8bdb50be343e8
- `seq.transformer.ours.log`: lane=transformer shape=llama8b.prefill.t8 tensor=q_proj.weight n=16777216 hash=7f5e346f0f234ca1
- `seq.transformer.ours.log`: lane=transformer shape=llama8b.prefill.t8 tensor=k_proj.weight n=4194304 hash=628683891b4ef505
- `seq.transformer.ours.log`: lane=transformer shape=llama8b.prefill.t8 tensor=v_proj.weight n=4194304 hash=a6e71091674bf598
- `seq.transformer.ours.log`: lane=transformer shape=llama8b.prefill.t8 tensor=o_proj.weight n=16777216 hash=f322159c55f94428
- `seq.transformer.ours.log`: lane=transformer shape=llama8b.prefill.t8 tensor=gate_proj.weight n=58720256 hash=e4ae7d5f3c49e108
- `seq.transformer.ours.log`: lane=transformer shape=llama8b.prefill.t8 tensor=up_proj.weight n=58720256 hash=e1e76c5d8ad14599
- `seq.transformer.ours.log`: lane=transformer shape=llama8b.prefill.t8 tensor=down_proj.weight n=58720256 hash=cd680ef665764882
- `seq.transformer.ours.log`: lane=transformer shape=llama8b.prefill.t8 tensor=x n=32768 hash=5fae3cdceecdb9e2
- `seq.transformer.ours.log`: lane=transformer arm=ours llama_refuse_bad_inputs alone shape=llama8b.prefill.t8 ms=649.545216 (inside the transformer lane's round, not inside attention/mlp/rmsnorm)
- `seq.transformer.ours.log`: lane=transformer arm=ours dump shape=llama8b.prefill.t8 n=32768 path=/root/gemm_leg_out/dump/seq.transformer.llama8b.prefill.t8.f32.bin
- `seq.transformer.ours.log`: lane=transformer shape=llama8b.prefill.t128 tensor=norm1.weight n=4096 hash=8e82620235b4b085
- `seq.transformer.ours.log`: lane=transformer shape=llama8b.prefill.t128 tensor=norm2.weight n=4096 hash=c813f5e53899180d
- `seq.transformer.ours.log`: lane=transformer shape=llama8b.prefill.t128 tensor=q_proj.weight n=16777216 hash=c1de6ff4941afb8e
- `seq.transformer.ours.log`: lane=transformer shape=llama8b.prefill.t128 tensor=k_proj.weight n=4194304 hash=803e4f6ec477b10e
- `seq.transformer.ours.log`: lane=transformer shape=llama8b.prefill.t128 tensor=v_proj.weight n=4194304 hash=1d4fa08f9f620c73
- `seq.transformer.ours.log`: lane=transformer shape=llama8b.prefill.t128 tensor=o_proj.weight n=16777216 hash=27755bc1220fd650
- `seq.transformer.ours.log`: lane=transformer shape=llama8b.prefill.t128 tensor=gate_proj.weight n=58720256 hash=2cc46a3f987b2a35
- `seq.transformer.ours.log`: lane=transformer shape=llama8b.prefill.t128 tensor=up_proj.weight n=58720256 hash=8df6060ace203936
- `seq.transformer.ours.log`: lane=transformer shape=llama8b.prefill.t128 tensor=down_proj.weight n=58720256 hash=8005f6e866ca1053
- `seq.transformer.ours.log`: lane=transformer shape=llama8b.prefill.t128 tensor=x n=524288 hash=4cfda6cf929d166b
- `seq.transformer.ours.log`: lane=transformer arm=ours llama_refuse_bad_inputs alone shape=llama8b.prefill.t128 ms=646.154811 (inside the transformer lane's round, not inside attention/mlp/rmsnorm)
- `seq.transformer.ours.log`: lane=transformer arm=ours dump shape=llama8b.prefill.t128 n=524288 path=/root/gemm_leg_out/dump/seq.transformer.llama8b.prefill.t128.f32.bin
- `seq.transformer.ours.log`: lane=transformer shape=llama8b.prefill.t512 tensor=norm1.weight n=4096 hash=267ae3b42cf4bd1e
- `seq.transformer.ours.log`: lane=transformer shape=llama8b.prefill.t512 tensor=norm2.weight n=4096 hash=fd8c830453040971
- `seq.transformer.ours.log`: lane=transformer shape=llama8b.prefill.t512 tensor=q_proj.weight n=16777216 hash=40d0cc6a048d905d
- `seq.transformer.ours.log`: lane=transformer shape=llama8b.prefill.t512 tensor=k_proj.weight n=4194304 hash=b5af8e64afe64f60
- `seq.transformer.ours.log`: lane=transformer shape=llama8b.prefill.t512 tensor=v_proj.weight n=4194304 hash=73e1eafa7abcd857
- `seq.transformer.ours.log`: lane=transformer shape=llama8b.prefill.t512 tensor=o_proj.weight n=16777216 hash=f8a091300d6898c7
- `seq.transformer.ours.log`: lane=transformer shape=llama8b.prefill.t512 tensor=gate_proj.weight n=58720256 hash=8b7b13c66c2b3235
- `seq.transformer.ours.log`: lane=transformer shape=llama8b.prefill.t512 tensor=up_proj.weight n=58720256 hash=f61e5f2f2b37b282
- `seq.transformer.ours.log`: lane=transformer shape=llama8b.prefill.t512 tensor=down_proj.weight n=58720256 hash=35adb0b4c5e445b4
- `seq.transformer.ours.log`: lane=transformer shape=llama8b.prefill.t512 tensor=x n=2097152 hash=4dea4dee43008247
- `seq.transformer.ours.log`: lane=transformer arm=ours llama_refuse_bad_inputs alone shape=llama8b.prefill.t512 ms=617.50587 (inside the transformer lane's round, not inside attention/mlp/rmsnorm)
- `seq.transformer.ours.log`: lane=transformer arm=ours dump shape=llama8b.prefill.t512 n=2097152 path=/root/gemm_leg_out/dump/seq.transformer.llama8b.prefill.t512.f32.bin
- `seq.transformer.ours.log`: lane=transformer shape=llama8b.decode.t1.ctx512 tensor=norm1.weight n=4096 hash=a16c25d484966855
- `seq.transformer.ours.log`: lane=transformer shape=llama8b.decode.t1.ctx512 tensor=norm2.weight n=4096 hash=fdf0b3b94538d853
- `seq.transformer.ours.log`: lane=transformer shape=llama8b.decode.t1.ctx512 tensor=q_proj.weight n=16777216 hash=dc126cf683d35c94
- `seq.transformer.ours.log`: lane=transformer shape=llama8b.decode.t1.ctx512 tensor=k_proj.weight n=4194304 hash=9bb615a45cfbb66d
- `seq.transformer.ours.log`: lane=transformer shape=llama8b.decode.t1.ctx512 tensor=v_proj.weight n=4194304 hash=b189eaa5691f2cc2
- `seq.transformer.ours.log`: lane=transformer shape=llama8b.decode.t1.ctx512 tensor=o_proj.weight n=16777216 hash=f460975b0ff8b858
- `seq.transformer.ours.log`: lane=transformer shape=llama8b.decode.t1.ctx512 tensor=gate_proj.weight n=58720256 hash=4b88f0c6df895117
- `seq.transformer.ours.log`: lane=transformer shape=llama8b.decode.t1.ctx512 tensor=up_proj.weight n=58720256 hash=0bcff2e323b1e524
- `seq.transformer.ours.log`: lane=transformer shape=llama8b.decode.t1.ctx512 tensor=down_proj.weight n=58720256 hash=513387fa68b57e5e
- `seq.transformer.ours.log`: lane=transformer shape=llama8b.decode.t1.ctx512 tensor=x n=4096 hash=a1509d4577b1d620
- `seq.transformer.ours.log`: lane=transformer shape=llama8b.decode.t1.ctx512 tensor=ctx.x n=2097152 hash=cf5b380dda81451a
- `seq.transformer.ours.log`: lane=transformer arm=ours llama_refuse_bad_inputs alone shape=llama8b.decode.t1.ctx512 ms=650.434941 (inside the transformer lane's round, not inside attention/mlp/rmsnorm)
- `seq.transformer.ours.log`: lane=transformer arm=ours dump shape=llama8b.decode.t1.ctx512 n=4096 path=/root/gemm_leg_out/dump/seq.transformer.llama8b.decode.t1.ctx512.f32.bin
- `seq.transformer.ours.log`: lane=transformer mode=FAST
- `seq.transformer.torch.log`: lane=transformer arm=torch build=CUDA 12.4 torch=2.4.1+cu124 tf32_switches=cuda.matmul.allow_tf32,cudnn.allow_tf32 deterministic=off
- `seq.transformer.torch.log`: lane=transformer.crosscheck shape=lane.b2_l4_d32_kv2 tensor=norm1.weight n=32 hash=e4ae9bac1e4f2eb0
- `seq.transformer.torch.log`: lane=transformer.crosscheck shape=lane.b2_l4_d32_kv2 tensor=norm2.weight n=32 hash=fe0abe179e07e573
- `seq.transformer.torch.log`: lane=transformer.crosscheck shape=lane.b2_l4_d32_kv2 tensor=q_proj.weight n=1024 hash=0d2da18fc0b40d8b
- `seq.transformer.torch.log`: lane=transformer.crosscheck shape=lane.b2_l4_d32_kv2 tensor=k_proj.weight n=1024 hash=8d6df99dbcd279fb
- `seq.transformer.torch.log`: lane=transformer.crosscheck shape=lane.b2_l4_d32_kv2 tensor=v_proj.weight n=1024 hash=566369012d3485a1
- `seq.transformer.torch.log`: lane=transformer.crosscheck shape=lane.b2_l4_d32_kv2 tensor=o_proj.weight n=1024 hash=d879230290d22ae1
- `seq.transformer.torch.log`: lane=transformer.crosscheck shape=lane.b2_l4_d32_kv2 tensor=gate_proj.weight n=2048 hash=5cd27648cfa7a924
- `seq.transformer.torch.log`: lane=transformer.crosscheck shape=lane.b2_l4_d32_kv2 tensor=up_proj.weight n=2048 hash=86fa1f885662c17f
- `seq.transformer.torch.log`: lane=transformer.crosscheck shape=lane.b2_l4_d32_kv2 tensor=down_proj.weight n=2048 hash=62887e5f6fb89109
- `seq.transformer.torch.log`: lane=transformer.crosscheck shape=lane.b2_l4_d32_kv2 tensor=x n=256 hash=cb5073d347fd2aae
- `seq.transformer.torch.log`: lane=transformer arm=torch crosscheck LlamaEager vs transformer/corpus block_forward at lane.b2_l4_d32_kv2: max_abs_diff=1.5878e-08
- `seq.transformer.torch.log`: lane=transformer shape=lane.b2_l4_d32_kv2 tensor=norm1.weight n=32 hash=e4ae9bac1e4f2eb0
- `seq.transformer.torch.log`: lane=transformer shape=lane.b2_l4_d32_kv2 tensor=norm2.weight n=32 hash=fe0abe179e07e573
- `seq.transformer.torch.log`: lane=transformer shape=lane.b2_l4_d32_kv2 tensor=q_proj.weight n=1024 hash=0d2da18fc0b40d8b
- `seq.transformer.torch.log`: lane=transformer shape=lane.b2_l4_d32_kv2 tensor=k_proj.weight n=1024 hash=8d6df99dbcd279fb
- `seq.transformer.torch.log`: lane=transformer shape=lane.b2_l4_d32_kv2 tensor=v_proj.weight n=1024 hash=566369012d3485a1
- `seq.transformer.torch.log`: lane=transformer shape=lane.b2_l4_d32_kv2 tensor=o_proj.weight n=1024 hash=d879230290d22ae1
- `seq.transformer.torch.log`: lane=transformer shape=lane.b2_l4_d32_kv2 tensor=gate_proj.weight n=2048 hash=5cd27648cfa7a924
- `seq.transformer.torch.log`: lane=transformer shape=lane.b2_l4_d32_kv2 tensor=up_proj.weight n=2048 hash=86fa1f885662c17f
- `seq.transformer.torch.log`: lane=transformer shape=lane.b2_l4_d32_kv2 tensor=down_proj.weight n=2048 hash=62887e5f6fb89109
- `seq.transformer.torch.log`: lane=transformer shape=lane.b2_l4_d32_kv2 tensor=x n=256 hash=cb5073d347fd2aae
- `seq.transformer.torch.log`: lane=transformer arm=torch-gpu-fp32 output witness shape=lane.b2_l4_d32_kv2 n=256 hash=d609221d7268e4c3
- `seq.transformer.torch.log`: lane=transformer arm=torch-gpu-tf32 output witness shape=lane.b2_l4_d32_kv2 n=256 hash=29818408b2158e13
- `seq.transformer.torch.log`: lane=transformer arm=torch-gpu-sdpa-math-fp32 output witness shape=lane.b2_l4_d32_kv2 n=256 hash=2e29daa0f9ab306a
- `seq.transformer.torch.log`: lane=transformer arm=torch-gpu-sdpa-efficient-fp32 output witness shape=lane.b2_l4_d32_kv2 n=256 hash=ffbca2badcffcd66
- `seq.transformer.torch.log`: lane=transformer shape=llama8b.prefill.t1 tensor=norm1.weight n=4096 hash=30dc9291bcd74c4d
- `seq.transformer.torch.log`: lane=transformer shape=llama8b.prefill.t1 tensor=norm2.weight n=4096 hash=90de83fb74424482
- `seq.transformer.torch.log`: lane=transformer shape=llama8b.prefill.t1 tensor=q_proj.weight n=16777216 hash=458f23ff554c3cc3
- `seq.transformer.torch.log`: lane=transformer shape=llama8b.prefill.t1 tensor=k_proj.weight n=4194304 hash=76706639c3aa7d56
- `seq.transformer.torch.log`: lane=transformer shape=llama8b.prefill.t1 tensor=v_proj.weight n=4194304 hash=81c62305b8e723d5
- `seq.transformer.torch.log`: lane=transformer shape=llama8b.prefill.t1 tensor=o_proj.weight n=16777216 hash=cffb8b77180868b3
- `seq.transformer.torch.log`: lane=transformer shape=llama8b.prefill.t1 tensor=gate_proj.weight n=58720256 hash=4616212f0e15d8ca
- `seq.transformer.torch.log`: lane=transformer shape=llama8b.prefill.t1 tensor=up_proj.weight n=58720256 hash=3d72973cf9e50898
- `seq.transformer.torch.log`: lane=transformer shape=llama8b.prefill.t1 tensor=down_proj.weight n=58720256 hash=63d7be8563af230a
- `seq.transformer.torch.log`: lane=transformer shape=llama8b.prefill.t1 tensor=x n=4096 hash=3d93aac664c66028
- `seq.transformer.torch.log`: lane=transformer arm=torch-gpu-fp32 output witness shape=llama8b.prefill.t1 n=4096 hash=bd44c6cd02b6e51c
- `seq.transformer.torch.log`: lane=transformer arm=torch-gpu-tf32 output witness shape=llama8b.prefill.t1 n=4096 hash=bd44c6cd02b6e51c
- `seq.transformer.torch.log`: lane=transformer arm=torch-gpu-sdpa-math-fp32 output witness shape=llama8b.prefill.t1 n=4096 hash=bd44c6cd02b6e51c
- `seq.transformer.torch.log`: lane=transformer arm=torch-gpu-sdpa-efficient-fp32 output witness shape=llama8b.prefill.t1 n=4096 hash=39aeb70c2dd56230
- `seq.transformer.torch.log`: lane=transformer shape=llama8b.prefill.t8 tensor=norm1.weight n=4096 hash=19770838eb156233
- `seq.transformer.torch.log`: lane=transformer shape=llama8b.prefill.t8 tensor=norm2.weight n=4096 hash=661f9b4826837778
- `seq.transformer.torch.log`: lane=transformer shape=llama8b.prefill.t8 tensor=q_proj.weight n=16777216 hash=9ba40b617f6701c6
- `seq.transformer.torch.log`: lane=transformer shape=llama8b.prefill.t8 tensor=k_proj.weight n=4194304 hash=3fc887e58a86aae5
- `seq.transformer.torch.log`: lane=transformer shape=llama8b.prefill.t8 tensor=v_proj.weight n=4194304 hash=64cb8d74bed4b578
- `seq.transformer.torch.log`: lane=transformer shape=llama8b.prefill.t8 tensor=o_proj.weight n=16777216 hash=5abb6368b7c381b7
- `seq.transformer.torch.log`: lane=transformer shape=llama8b.prefill.t8 tensor=gate_proj.weight n=58720256 hash=40827a19ef224a29
- `seq.transformer.torch.log`: lane=transformer shape=llama8b.prefill.t8 tensor=up_proj.weight n=58720256 hash=b8d683c2de2a8d40
- `seq.transformer.torch.log`: lane=transformer shape=llama8b.prefill.t8 tensor=down_proj.weight n=58720256 hash=be5808bd1301911b
- `seq.transformer.torch.log`: lane=transformer shape=llama8b.prefill.t8 tensor=x n=32768 hash=b393ca1e2e252bc2
- `seq.transformer.torch.log`: lane=transformer arm=torch-gpu-fp32 output witness shape=llama8b.prefill.t8 n=32768 hash=264834725c1d0333
- `seq.transformer.torch.log`: lane=transformer arm=torch-gpu-tf32 output witness shape=llama8b.prefill.t8 n=32768 hash=d6b8ee1c6138f26c
- `seq.transformer.torch.log`: lane=transformer arm=torch-gpu-sdpa-math-fp32 output witness shape=llama8b.prefill.t8 n=32768 hash=c454cab7efffdd60
- `seq.transformer.torch.log`: lane=transformer arm=torch-gpu-sdpa-efficient-fp32 output witness shape=llama8b.prefill.t8 n=32768 hash=096bf1274161501e
- `seq.transformer.torch.log`: lane=transformer shape=llama8b.prefill.t128 tensor=norm1.weight n=4096 hash=2d073c4bf3be4215
- `seq.transformer.torch.log`: lane=transformer shape=llama8b.prefill.t128 tensor=norm2.weight n=4096 hash=8d13de87ae1db05d
- `seq.transformer.torch.log`: lane=transformer shape=llama8b.prefill.t128 tensor=q_proj.weight n=16777216 hash=19976a410e812f3d
- `seq.transformer.torch.log`: lane=transformer shape=llama8b.prefill.t128 tensor=k_proj.weight n=4194304 hash=6eb9a28e49ec40ae
- `seq.transformer.torch.log`: lane=transformer shape=llama8b.prefill.t128 tensor=v_proj.weight n=4194304 hash=91936c2a28335c93
- `seq.transformer.torch.log`: lane=transformer shape=llama8b.prefill.t128 tensor=o_proj.weight n=16777216 hash=33f47dd4c2896a57
- `seq.transformer.torch.log`: lane=transformer shape=llama8b.prefill.t128 tensor=gate_proj.weight n=58720256 hash=00ad980c0db80b74
- `seq.transformer.torch.log`: lane=transformer shape=llama8b.prefill.t128 tensor=up_proj.weight n=58720256 hash=c2002fe051fc2a67
- `seq.transformer.torch.log`: lane=transformer shape=llama8b.prefill.t128 tensor=down_proj.weight n=58720256 hash=fe37d1e5eadac30a
- `seq.transformer.torch.log`: lane=transformer shape=llama8b.prefill.t128 tensor=x n=524288 hash=60e833df1e7657f3
- `seq.transformer.torch.log`: lane=transformer arm=torch-gpu-fp32 output witness shape=llama8b.prefill.t128 n=524288 hash=d38a35a21e9b39da
- `seq.transformer.torch.log`: lane=transformer arm=torch-gpu-tf32 output witness shape=llama8b.prefill.t128 n=524288 hash=d382e30f3c8f8ec6
- `seq.transformer.torch.log`: lane=transformer arm=torch-gpu-sdpa-math-fp32 output witness shape=llama8b.prefill.t128 n=524288 hash=ffd67056365b9cfc
- `seq.transformer.torch.log`: lane=transformer arm=torch-gpu-sdpa-efficient-fp32 output witness shape=llama8b.prefill.t128 n=524288 hash=9c5143d1e9f63dbb
- `seq.transformer.torch.log`: lane=transformer shape=llama8b.prefill.t512 tensor=norm1.weight n=4096 hash=f4a0f716372b58ce
- `seq.transformer.torch.log`: lane=transformer shape=llama8b.prefill.t512 tensor=norm2.weight n=4096 hash=93b3f2cddfd23c01
- `seq.transformer.torch.log`: lane=transformer shape=llama8b.prefill.t512 tensor=q_proj.weight n=16777216 hash=fe1b3ad1dfd94d86
- `seq.transformer.torch.log`: lane=transformer shape=llama8b.prefill.t512 tensor=k_proj.weight n=4194304 hash=9f658c4d5c9859c0
- `seq.transformer.torch.log`: lane=transformer shape=llama8b.prefill.t512 tensor=v_proj.weight n=4194304 hash=286524a647dfa937
- `seq.transformer.torch.log`: lane=transformer shape=llama8b.prefill.t512 tensor=o_proj.weight n=16777216 hash=7721f27f6a6c4ce8
- `seq.transformer.torch.log`: lane=transformer shape=llama8b.prefill.t512 tensor=gate_proj.weight n=58720256 hash=630b3a509bc6f674
- `seq.transformer.torch.log`: lane=transformer shape=llama8b.prefill.t512 tensor=up_proj.weight n=58720256 hash=4cf23bd89c42c8b3
- `seq.transformer.torch.log`: lane=transformer shape=llama8b.prefill.t512 tensor=down_proj.weight n=58720256 hash=5c6bd78157391281
- `seq.transformer.torch.log`: lane=transformer shape=llama8b.prefill.t512 tensor=x n=2097152 hash=9ffa2521076bc8c7
- `seq.transformer.torch.log`: lane=transformer arm=torch-gpu-fp32 output witness shape=llama8b.prefill.t512 n=2097152 hash=b66f01529b3be4c9
- `seq.transformer.torch.log`: lane=transformer arm=torch-gpu-tf32 output witness shape=llama8b.prefill.t512 n=2097152 hash=e9e65e78d26a5bfc
- `seq.transformer.torch.log`: lane=transformer arm=torch-gpu-sdpa-math-fp32 output witness shape=llama8b.prefill.t512 n=2097152 hash=9248658d19e75e10
- `seq.transformer.torch.log`: lane=transformer arm=torch-gpu-sdpa-efficient-fp32 output witness shape=llama8b.prefill.t512 n=2097152 hash=66a31bf32885c80f
- `seq.transformer.torch.log`: lane=transformer shape=llama8b.decode.t1.ctx512 tensor=norm1.weight n=4096 hash=939d6972265aa725
- `seq.transformer.torch.log`: lane=transformer shape=llama8b.decode.t1.ctx512 tensor=norm2.weight n=4096 hash=31804b90e76959c3
- `seq.transformer.torch.log`: lane=transformer shape=llama8b.decode.t1.ctx512 tensor=q_proj.weight n=16777216 hash=b1b0a1ba4873ca77
- `seq.transformer.torch.log`: lane=transformer shape=llama8b.decode.t1.ctx512 tensor=k_proj.weight n=4194304 hash=1c16ca3348db5f0d
- `seq.transformer.torch.log`: lane=transformer shape=llama8b.decode.t1.ctx512 tensor=v_proj.weight n=4194304 hash=237925dc612d1c22
- `seq.transformer.torch.log`: lane=transformer shape=llama8b.decode.t1.ctx512 tensor=o_proj.weight n=16777216 hash=0ec594792330189f
- `seq.transformer.torch.log`: lane=transformer shape=llama8b.decode.t1.ctx512 tensor=gate_proj.weight n=58720256 hash=c5c194484347d30a
- `seq.transformer.torch.log`: lane=transformer shape=llama8b.decode.t1.ctx512 tensor=up_proj.weight n=58720256 hash=9374a3fc8f279dd9
- `seq.transformer.torch.log`: lane=transformer shape=llama8b.decode.t1.ctx512 tensor=down_proj.weight n=58720256 hash=6e1a2f6cfa111b57
- `seq.transformer.torch.log`: lane=transformer shape=llama8b.decode.t1.ctx512 tensor=x n=4096 hash=b91b5914caac5a30
- `seq.transformer.torch.log`: lane=transformer shape=llama8b.decode.t1.ctx512 tensor=ctx.x n=2097152 hash=c54a8d22a1bde29a
- `seq.transformer.torch.log`: lane=transformer arm=torch-gpu-fp32 output witness shape=llama8b.decode.t1.ctx512 n=4096 hash=f62a023115027e31
- `seq.transformer.torch.log`: lane=transformer arm=torch-gpu-tf32 output witness shape=llama8b.decode.t1.ctx512 n=4096 hash=1f71bd5f486361ec
- `seq.transformer.torch.log`: lane=transformer arm=torch-gpu-sdpa-math-fp32 output witness shape=llama8b.decode.t1.ctx512 n=4096 hash=848e21d1df64d87c
- `seq.transformer.torch.log`: lane=transformer arm=torch-gpu-sdpa-efficient-fp32 output witness shape=llama8b.decode.t1.ctx512 n=4096 hash=d8fbb0fbeec0b7dd
- `seq.transformer.torch.log`: lane=transformer arm=torch

