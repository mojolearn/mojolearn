# The FAST path against the vendor, one rented NVIDIA box

Parsed from `bench/results/fast_speed/mac-2026-08-28_131915-gemmseq/logs`, 20 arm logs.

Device(s) reported by the arms themselves: Apple GPU, Apple M4, Apple_M4, Apple_MPS

`ratio = median(ours) / median(theirs)`. **Above 1.0 means WE ARE SLOWER.**
The warm-up round is excluded from every statistic and printed on its own,
because torch pays an enormous first call and hiding that makes the median
read as the whole story. `min` is printed beside the median so a reader can
see when the box was busy: the further they are apart, the less the row is worth.

## Every arm, as measured

| lane | arm | shape | n | median ms | min ms | max ms | warmup ms | hash |
|---|---|---|---|---|---|---|---|---|
| attention | ours | lane.b2_l4_d32_kv2 | 3 | 4.282 | 3.935 | 4.553 | 7.402 | ffea5f20a005c245 |
| attention | ours | llama8b.decode.t1.ctx512 | 3 | 18.832 | 17.198 | 23.284 | 38.831 | 33abfdd6c1472371 |
| attention | ours | llama8b.prefill.t1 | 3 | 16.171 | 14.692 | 23.998 | 35.415 | e547a9bc7ba84b6a |
| attention | ours | llama8b.prefill.t128 | 3 | 27.133 | 26.226 | 31.305 | 55.763 | 1517a52ca382b1fa |
| attention | ours | llama8b.prefill.t512 | 3 | 88.574 | 87.465 | 89.142 | 120.560 | e02620df4d0c41da |
| attention | ours | llama8b.prefill.t8 | 3 | 20.762 | 19.590 | 26.158 | 41.130 | dab24b18ece70e11 |
| attention | torch-gpu-flash-bf16 | lane.b2_l4_d32_kv2 | 3 | 0.557 | 0.555 | 0.559 | 0.584 | d75aaf042d86b26a |
| attention | torch-gpu-flash-bf16 | llama8b.decode.t1.ctx512 | 3 | 2.463 | 1.850 | 2.493 | 1.888 | 07bd27566b6f7b64 |
| attention | torch-gpu-flash-bf16 | llama8b.prefill.t1 | 3 | 1.578 | 1.366 | 1.692 | 1.390 | e03765092a7f565d |
| attention | torch-gpu-flash-bf16 | llama8b.prefill.t128 | 3 | 4.851 | 4.785 | 4.918 | 4.817 | - |
| attention | torch-gpu-flash-bf16 | llama8b.prefill.t512 | 3 | 18.524 | 18.418 | 19.206 | 18.813 | - |
| attention | torch-gpu-flash-bf16 | llama8b.prefill.t8 | 3 | 1.565 | 1.504 | 1.870 | 1.548 | 1149a5e153fe8e58 |
| attention | torch-gpu-flash-fp32 | lane.b2_l4_d32_kv2 | 3 | 0.564 | 0.532 | 1.034 | 0.523 | aceb31ca74fb96da |
| attention | torch-gpu-flash-fp32 | llama8b.decode.t1.ctx512 | 3 | 3.312 | 2.734 | 3.380 | 2.903 | 5c6f4165622bf7e3 |
| attention | torch-gpu-flash-fp32 | llama8b.prefill.t1 | 3 | 2.172 | 2.158 | 2.641 | 2.243 | dcd189bc2aa6895f |
| attention | torch-gpu-flash-fp32 | llama8b.prefill.t128 | 3 | 5.604 | 5.543 | 5.694 | 5.610 | - |
| attention | torch-gpu-flash-fp32 | llama8b.prefill.t512 | 3 | 22.073 | 21.812 | 22.313 | 23.772 | - |
| attention | torch-gpu-flash-fp32 | llama8b.prefill.t8 | 3 | 3.193 | 2.613 | 3.209 | 3.117 | 4dc36d77c2677bf5 |
| attention | torch-gpu-fp32 | lane.b2_l4_d32_kv2 | 3 | 2.302 | 1.416 | 2.336 | 2.219 | 1c464d02303b51dc |
| attention | torch-gpu-fp32 | llama8b.decode.t1.ctx512 | 3 | 4.063 | 3.823 | 4.594 | 4.810 | c5b75965b7575017 |
| attention | torch-gpu-fp32 | llama8b.prefill.t1 | 3 | 4.197 | 3.204 | 4.249 | 3.952 | dcd189bc2aa6895f |
| attention | torch-gpu-fp32 | llama8b.prefill.t128 | 3 | 7.594 | 6.522 | 7.656 | 6.964 | - |
| attention | torch-gpu-fp32 | llama8b.prefill.t512 | 3 | 27.636 | 27.381 | 27.965 | 27.857 | - |
| attention | torch-gpu-fp32 | llama8b.prefill.t8 | 3 | 4.676 | 4.653 | 4.807 | 4.100 | a17485e9a9152030 |
| attention | torch-gpu-sdpa-efficient-fp32 | lane.b2_l4_d32_kv2 | 3 | 0.563 | 0.560 | 1.008 | 0.551 | aceb31ca74fb96da |
| attention | torch-gpu-sdpa-efficient-fp32 | llama8b.decode.t1.ctx512 | 3 | 2.834 | 2.833 | 2.910 | 2.860 | 5c6f4165622bf7e3 |
| attention | torch-gpu-sdpa-efficient-fp32 | llama8b.prefill.t1 | 3 | 2.152 | 2.149 | 2.226 | 2.890 | dcd189bc2aa6895f |
| attention | torch-gpu-sdpa-efficient-fp32 | llama8b.prefill.t128 | 3 | 5.615 | 5.556 | 6.271 | 5.633 | - |
| attention | torch-gpu-sdpa-efficient-fp32 | llama8b.prefill.t512 | 3 | 23.663 | 23.225 | 23.796 | 21.991 | - |
| attention | torch-gpu-sdpa-efficient-fp32 | llama8b.prefill.t8 | 3 | 3.171 | 2.710 | 3.266 | 3.326 | 4dc36d77c2677bf5 |
| attention | torch-gpu-sdpa-math-fp32 | lane.b2_l4_d32_kv2 | 3 | 0.570 | 0.551 | 1.059 | 1.055 | aceb31ca74fb96da |
| attention | torch-gpu-sdpa-math-fp32 | llama8b.decode.t1.ctx512 | 3 | 2.688 | 2.645 | 2.696 | 3.408 | 5c6f4165622bf7e3 |
| attention | torch-gpu-sdpa-math-fp32 | llama8b.prefill.t1 | 3 | 2.199 | 2.165 | 2.203 | 2.225 | dcd189bc2aa6895f |
| attention | torch-gpu-sdpa-math-fp32 | llama8b.prefill.t128 | 3 | 5.612 | 5.542 | 5.632 | 5.620 | - |
| attention | torch-gpu-sdpa-math-fp32 | llama8b.prefill.t512 | 3 | 21.860 | 21.426 | 22.325 | 21.393 | - |
| attention | torch-gpu-sdpa-math-fp32 | llama8b.prefill.t8 | 3 | 3.090 | 2.947 | 3.184 | 2.774 | 4dc36d77c2677bf5 |
| attention | torch-gpu-tf32 | lane.b2_l4_d32_kv2 | 3 | 2.155 | 1.912 | 4.593 | 1.219 | 1c464d02303b51dc |
| attention | torch-gpu-tf32 | llama8b.decode.t1.ctx512 | 3 | 3.648 | 3.300 | 4.425 | 3.593 | c5b75965b7575017 |
| attention | torch-gpu-tf32 | llama8b.prefill.t1 | 3 | 3.358 | 3.352 | 4.142 | 2.900 | dcd189bc2aa6895f |
| attention | torch-gpu-tf32 | llama8b.prefill.t128 | 3 | 6.964 | 6.849 | 7.450 | 6.707 | - |
| attention | torch-gpu-tf32 | llama8b.prefill.t512 | 3 | 27.246 | 26.959 | 28.023 | 27.763 | - |
| attention | torch-gpu-tf32 | llama8b.prefill.t8 | 3 | 3.831 | 3.636 | 4.116 | 4.695 | a17485e9a9152030 |
| gemm | mps-default | gram.128sq.x100003 | 3 | 1.894 | 1.527 | 1.974 | 1.896 | - |
| gemm | mps-default | gram.32x32x1M | 3 | 8.340 | 7.648 | 8.684 | 8.693 | - |
| gemm | mps-default | gram.32x32x64K | 3 | 0.756 | 0.742 | 0.853 | 0.895 | - |
| gemm | mps-default | kmeans.dist.4096x64x64 | 3 | 0.260 | 0.259 | 0.262 | 0.280 | - |
| gemm | mps-default | llama8b.lm_head.t1 | 3 | 22.179 | 21.089 | 22.681 | 22.153 | - |
| gemm | mps-default | llama8b.lm_head.t512 | 3 | 238.973 | 238.895 | 239.213 | 238.611 | - |
| gemm | mps-default | llama8b.lm_head.t8 | 3 | 29.002 | 28.710 | 29.061 | 30.327 | - |
| gemm | mps-default | llama8b.mlp_down.t1 | 3 | 2.763 | 2.601 | 2.865 | 2.562 | - |
| gemm | mps-default | llama8b.mlp_down.t512 | 3 | 22.665 | 22.396 | 23.462 | 22.343 | - |
| gemm | mps-default | llama8b.mlp_down.t8 | 3 | 3.539 | 3.495 | 3.745 | 3.642 | - |
| gemm | mps-default | llama8b.mlp_up.t1 | 3 | 2.976 | 2.932 | 3.030 | 3.730 | - |
| gemm | mps-default | llama8b.mlp_up.t512 | 3 | 22.494 | 21.774 | 22.528 | 22.571 | - |
| gemm | mps-default | llama8b.mlp_up.t8 | 3 | 3.892 | 3.811 | 3.902 | 3.867 | - |
| gemm | mps-default | llama8b.qkv.t1 | 3 | 0.990 | 0.949 | 1.042 | 0.993 | - |
| gemm | mps-default | llama8b.qkv.t512 | 3 | 6.508 | 6.494 | 6.539 | 7.307 | - |
| gemm | mps-default | llama8b.qkv.t8 | 3 | 1.195 | 1.168 | 1.718 | 1.155 | - |
| gemm | mps-default | ols.predict.gemv.64Kx16 | 3 | 0.239 | 0.236 | 0.239 | 0.269 | - |
| gemm | mps-default | ols.step1.16x16x64K | 3 | 0.761 | 0.713 | 0.764 | 0.774 | - |
| gemm | mps-default | pca.transform.8192x4x4 | 3 | 0.220 | 0.204 | 0.222 | 0.268 | - |
| gemm | mps-default | pca.transform.wide.8192x64x128 | 3 | 0.329 | 0.279 | 1.083 | 0.299 | - |
| gemm | ours | gram.128sq.x100003 | 3 | 4.212 | 4.203 | 4.216 | 7.960 | 5d8bedc24cc05ccf |
| gemm | ours | gram.32x32x1M | 3 | 8.041 | 7.088 | 8.869 | 16.322 | ba19a7670161a95d |
| gemm | ours | gram.32x32x64K | 3 | 0.856 | 0.811 | 0.917 | 1.291 | 2e5882d6d4fc4e89 |
| gemm | ours | kmeans.dist.4096x64x64 | 3 | 0.279 | 0.275 | 0.279 | 0.279 | 05c1bddd0f21fbc0 |
| gemm | ours | llama8b.lm_head.t1 | 3 | 24.148 | 21.530 | 25.373 | 39.912 | 82695be4cb214ea0 |
| gemm | ours | llama8b.lm_head.t512 | 3 | 251.424 | 246.162 | 255.487 | 256.567 | 5910de8ba5512131 |
| gemm | ours | llama8b.lm_head.t8 | 3 | 40.071 | 39.694 | 40.728 | 47.683 | 0c92b06a55959232 |
| gemm | ours | llama8b.mlp_down.t1 | 3 | 2.997 | 2.697 | 3.229 | 7.384 | 53d4682d1aa49274 |
| gemm | ours | llama8b.mlp_down.t512 | 3 | 53.140 | 49.810 | 58.571 | 58.255 | 2690fef516f52350 |
| gemm | ours | llama8b.mlp_down.t8 | 3 | 4.853 | 4.771 | 4.958 | 6.677 | fb635d6a0d7d641f |
| gemm | ours | llama8b.mlp_up.t1 | 3 | 3.052 | 2.554 | 3.231 | 7.304 | 36a00f23caf25b37 |
| gemm | ours | llama8b.mlp_up.t512 | 3 | 36.506 | 27.646 | 37.133 | 41.457 | 71eab89d39318fd9 |
| gemm | ours | llama8b.mlp_up.t8 | 3 | 9.404 | 9.020 | 9.752 | 10.731 | 5eb1d46daf724bdb |
| gemm | ours | llama8b.qkv.t1 | 3 | 0.947 | 0.946 | 0.969 | 1.604 | e1ba240d1ace7e31 |
| gemm | ours | llama8b.qkv.t512 | 3 | 17.047 | 16.413 | 19.596 | 15.971 | 5e273b5584e560be |
| gemm | ours | llama8b.qkv.t8 | 3 | 2.067 | 1.911 | 2.157 | 2.309 | 601458c362d976cc |
| gemm | ours | ols.predict.gemv.64Kx16 | 3 | 0.503 | 0.490 | 0.530 | 1.053 | e17121fd170e87d6 |
| gemm | ours | ols.step1.16x16x64K | 3 | 0.705 | 0.689 | 1.079 | 0.962 | f4a40db03107a492 |
| gemm | ours | pca.transform.8192x4x4 | 3 | 0.209 | 0.192 | 0.221 | 0.644 | 59b1a1993889ef60 |
| gemm | ours | pca.transform.wide.8192x64x128 | 3 | 0.431 | 0.425 | 0.504 | 0.472 | ce518fe285e59ce5 |
| mamba | ours | lane.b2_l4_d8 | 3 | 4.480 | 4.152 | 7.250 | 3.920 | 4eefaa4a619c7219 |
| mamba | ours | mamba130m.prefill.t1 | 3 | 4.927 | 4.724 | 7.797 | 5.222 | f40756d484865d3d |
| mamba | ours | mamba130m.prefill.t128 | 3 | 5.019 | 4.669 | 7.235 | 12.032 | ee5cc1df0907b5b0 |
| mamba | ours | mamba130m.prefill.t512 | 3 | 7.749 | 7.434 | 8.769 | 13.232 | 024a1f81a8f44125 |
| mamba | ours | mamba130m.prefill.t8 | 3 | 5.352 | 4.193 | 8.472 | 3.357 | 4aaf08796d3de736 |
| mlp | ours | lane.b2_l4_d32_kv2 | 3 | 0.688 | 0.559 | 0.707 | 0.723 | 998ffb5b6782dc00 |
| mlp | ours | llama8b.decode.t1.ctx512 | 3 | 7.744 | 7.110 | 11.839 | 15.415 | f63262c18aa54a1a |
| mlp | ours | llama8b.prefill.t1 | 3 | 10.972 | 7.625 | 17.566 | 16.431 | d1e147331d423757 |
| mlp | ours | llama8b.prefill.t128 | 3 | 24.744 | 24.597 | 25.730 | 43.098 | 5946f50d21f0ad50 |
| mlp | ours | llama8b.prefill.t512 | 3 | 111.150 | 110.576 | 111.634 | 142.390 | dba7cf1cb61730b7 |
| mlp | ours | llama8b.prefill.t8 | 3 | 13.181 | 13.161 | 13.563 | 40.518 | 7f08a16d1f50ddd3 |
| mlp | torch-gpu-fp32 | lane.b2_l4_d32_kv2 | 3 | 0.537 | 0.497 | 0.551 | 0.588 | 9bbbcfc5f41bf3c2 |
| mlp | torch-gpu-fp32 | llama8b.decode.t1.ctx512 | 3 | 7.374 | 7.158 | 7.556 | 7.088 | 3808aa8e354d41f7 |
| mlp | torch-gpu-fp32 | llama8b.prefill.t1 | 3 | 7.740 | 7.486 | 8.208 | 7.085 | cab71e91e62ef2fb |
| mlp | torch-gpu-fp32 | llama8b.prefill.t128 | 3 | 18.379 | 17.695 | 18.455 | 17.762 | - |
| mlp | torch-gpu-fp32 | llama8b.prefill.t512 | 3 | 71.586 | 71.436 | 71.750 | 71.006 | - |
| mlp | torch-gpu-fp32 | llama8b.prefill.t8 | 3 | 9.335 | 8.875 | 9.964 | 8.864 | 7d9c30e9e58b8fa7 |
| mlp | torch-gpu-tf32 | lane.b2_l4_d32_kv2 | 3 | 0.298 | 0.289 | 0.323 | 0.546 | 9bbbcfc5f41bf3c2 |
| mlp | torch-gpu-tf32 | llama8b.decode.t1.ctx512 | 3 | 7.062 | 6.971 | 7.080 | 7.321 | 3808aa8e354d41f7 |
| mlp | torch-gpu-tf32 | llama8b.prefill.t1 | 3 | 7.015 | 7.006 | 7.109 | 8.267 | cab71e91e62ef2fb |
| mlp | torch-gpu-tf32 | llama8b.prefill.t128 | 3 | 17.985 | 17.610 | 18.486 | 18.275 | - |
| mlp | torch-gpu-tf32 | llama8b.prefill.t512 | 3 | 71.771 | 71.545 | 72.219 | 73.559 | - |
| mlp | torch-gpu-tf32 | llama8b.prefill.t8 | 3 | 9.544 | 9.442 | 10.037 | 9.043 | 7d9c30e9e58b8fa7 |
| rmsnorm | ours | lane.b2_l4_d32_kv2 | 3 | 0.269 | 0.205 | 0.537 | 0.257 | b058a2f6243d396f |
| rmsnorm | ours | llama8b.decode.t1.ctx512 | 3 | 3.277 | 3.044 | 3.519 | 4.326 | 7fe822dc61475447 |
| rmsnorm | ours | llama8b.prefill.t1 | 3 | 3.137 | 3.078 | 6.388 | 3.698 | c65e8b5c4dc7c77f |
| rmsnorm | ours | llama8b.prefill.t128 | 3 | 2.364 | 1.928 | 6.510 | 4.693 | e34b3d46e8a281da |
| rmsnorm | ours | llama8b.prefill.t512 | 3 | 4.144 | 3.970 | 4.243 | 22.521 | 44e3314215974137 |
| rmsnorm | ours | llama8b.prefill.t8 | 3 | 3.178 | 3.128 | 3.490 | 5.258 | e85374980534e30b |
| rmsnorm | torch-gpu-fp32 | lane.b2_l4_d32_kv2 | 3 | 0.494 | 0.476 | 0.506 | 0.532 | 7e3e6a4334c30308 |
| rmsnorm | torch-gpu-fp32 | llama8b.decode.t1.ctx512 | 3 | 0.617 | 0.343 | 0.673 | 0.598 | 409660f2f0359908 |
| rmsnorm | torch-gpu-fp32 | llama8b.prefill.t1 | 3 | 0.584 | 0.512 | 3.703 | 0.729 | 4635a852c2627e50 |
| rmsnorm | torch-gpu-fp32 | llama8b.prefill.t128 | 3 | 1.035 | 0.700 | 1.067 | 0.900 | - |
| rmsnorm | torch-gpu-fp32 | llama8b.prefill.t512 | 3 | 2.070 | 1.211 | 3.718 | 4.198 | - |
| rmsnorm | torch-gpu-fp32 | llama8b.prefill.t8 | 3 | 0.611 | 0.565 | 0.713 | 0.570 | ce88b1bcde6e809d |
| rmsnorm | torch-gpu-tf32 | lane.b2_l4_d32_kv2 | 3 | 0.440 | 0.298 | 0.505 | 0.528 | 7e3e6a4334c30308 |
| rmsnorm | torch-gpu-tf32 | llama8b.decode.t1.ctx512 | 3 | 0.226 | 0.223 | 0.573 | 0.210 | 409660f2f0359908 |
| rmsnorm | torch-gpu-tf32 | llama8b.prefill.t1 | 3 | 0.503 | 0.316 | 0.505 | 0.526 | 4635a852c2627e50 |
| rmsnorm | torch-gpu-tf32 | llama8b.prefill.t128 | 3 | 0.510 | 0.449 | 1.223 | 0.790 | - |
| rmsnorm | torch-gpu-tf32 | llama8b.prefill.t512 | 3 | 1.498 | 1.239 | 1.506 | 1.531 | - |
| rmsnorm | torch-gpu-tf32 | llama8b.prefill.t8 | 3 | 0.528 | 0.526 | 0.591 | 0.603 | ce88b1bcde6e809d |
| selective_scan | ours | lane.b2_l4_d8 | 3 | 0.267 | 0.263 | 0.278 | 2.155 | 51404fb94f6ca943 |
| selective_scan | ours | mamba130m.prefill.t1 | 3 | 0.270 | 0.261 | 0.271 | 0.261 | fbfb6be4b35b48bf |
| selective_scan | ours | mamba130m.prefill.t128 | 3 | 0.877 | 0.545 | 0.885 | 0.544 | a41875537158170d |
| selective_scan | ours | mamba130m.prefill.t512 | 3 | 1.166 | 0.524 | 1.248 | 0.768 | 73ade061baf1002c |
| selective_scan | ours | mamba130m.prefill.t8 | 3 | 0.195 | 0.189 | 0.199 | 0.201 | 322d35bd8918d2b8 |
| transformer | ours | lane.b2_l4_d32_kv2 | 3 | 8.707 | 5.141 | 9.083 | 9.728 | cbf93c1be68cec3a |
| transformer | ours | llama8b.decode.t1.ctx512 | 3 | 33.589 | 33.544 | 39.014 | 59.399 | 2104cd166a4c517c |
| transformer | ours | llama8b.prefill.t1 | 3 | 28.178 | 25.479 | 31.381 | 46.756 | 104018e45e3bf971 |
| transformer | ours | llama8b.prefill.t128 | 3 | 55.527 | 55.062 | 56.909 | 83.589 | 997faa51ef93951c |
| transformer | ours | llama8b.prefill.t512 | 3 | 221.114 | 213.735 | 228.377 | 248.767 | 5ee9ac38ccf0478e |
| transformer | ours | llama8b.prefill.t8 | 3 | 35.603 | 34.782 | 40.648 | 62.207 | d1ef18be1cfaa49d |
| transformer | torch-gpu-flash-bf16 | lane.b2_l4_d32_kv2 | 3 | 1.227 | 1.162 | 1.389 | 1.152 | 33a8ffb39891403a |
| transformer | torch-gpu-flash-bf16 | llama8b.decode.t1.ctx512 | 3 | 5.664 | 5.258 | 6.431 | 5.156 | b393e512937ba972 |
| transformer | torch-gpu-flash-bf16 | llama8b.prefill.t1 | 3 | 5.757 | 5.315 | 6.290 | 5.709 | f992e21881c9d23e |
| transformer | torch-gpu-flash-bf16 | llama8b.prefill.t128 | 3 | 20.300 | 20.148 | 20.418 | 19.506 | - |
| transformer | torch-gpu-flash-bf16 | llama8b.prefill.t512 | 3 | 80.126 | 79.768 | 80.615 | 80.207 | - |
| transformer | torch-gpu-flash-bf16 | llama8b.prefill.t8 | 3 | 5.640 | 5.079 | 6.617 | 5.683 | 3fcf9a3fc451735a |
| transformer | torch-gpu-flash-fp32 | lane.b2_l4_d32_kv2 | 3 | 1.086 | 0.903 | 1.245 | 1.112 | d3262edc343c168f |
| transformer | torch-gpu-flash-fp32 | llama8b.decode.t1.ctx512 | 3 | 9.556 | 9.444 | 10.381 | 9.473 | ec9bd0d9e8bc6284 |
| transformer | torch-gpu-flash-fp32 | llama8b.prefill.t1 | 3 | 10.881 | 10.746 | 11.715 | 10.839 | 64fcadaa649bc195 |
| transformer | torch-gpu-flash-fp32 | llama8b.prefill.t128 | 3 | 24.384 | 23.785 | 24.487 | 24.270 | - |
| transformer | torch-gpu-flash-fp32 | llama8b.prefill.t512 | 3 | 100.280 | 99.371 | 100.437 | 101.000 | - |
| transformer | torch-gpu-flash-fp32 | llama8b.prefill.t8 | 3 | 11.974 | 11.572 | 12.598 | 11.609 | c07cfefa1e45697b |
| transformer | torch-gpu-fp32 | lane.b2_l4_d32_kv2 | 3 | 2.295 | 2.257 | 2.295 | 2.042 | 014d7d70948e1158 |
| transformer | torch-gpu-fp32 | llama8b.decode.t1.ctx512 | 3 | 11.508 | 10.519 | 12.167 | 12.113 | 25d620f56e175aa1 |
| transformer | torch-gpu-fp32 | llama8b.prefill.t1 | 3 | 10.955 | 10.558 | 11.111 | 11.264 | 64fcadaa649bc195 |
| transformer | torch-gpu-fp32 | llama8b.prefill.t128 | 3 | 24.852 | 24.168 | 25.109 | 25.046 | - |
| transformer | torch-gpu-fp32 | llama8b.prefill.t512 | 3 | 101.787 | 101.313 | 102.703 | 102.029 | - |
| transformer | torch-gpu-fp32 | llama8b.prefill.t8 | 3 | 12.724 | 12.649 | 12.754 | 12.557 | 809e8075f0a9ad74 |
| transformer | torch-gpu-sdpa-efficient-fp32 | lane.b2_l4_d32_kv2 | 3 | 1.253 | 1.252 | 1.487 | 0.980 | d3262edc343c168f |
| transformer | torch-gpu-sdpa-efficient-fp32 | llama8b.decode.t1.ctx512 | 3 | 9.464 | 9.434 | 10.611 | 9.729 | ec9bd0d9e8bc6284 |
| transformer | torch-gpu-sdpa-efficient-fp32 | llama8b.prefill.t1 | 3 | 9.225 | 9.166 | 9.248 | 9.393 | 64fcadaa649bc195 |
| transformer | torch-gpu-sdpa-efficient-fp32 | llama8b.prefill.t128 | 3 | 24.482 | 23.764 | 24.631 | 24.490 | - |
| transformer | torch-gpu-sdpa-efficient-fp32 | llama8b.prefill.t512 | 3 | 96.518 | 96.100 | 97.551 | 95.890 | - |
| transformer | torch-gpu-sdpa-efficient-fp32 | llama8b.prefill.t8 | 3 | 11.940 | 11.459 | 12.679 | 11.618 | c07cfefa1e45697b |
| transformer | torch-gpu-sdpa-math-fp32 | lane.b2_l4_d32_kv2 | 3 | 1.354 | 1.275 | 1.477 | 1.479 | d3262edc343c168f |
| transformer | torch-gpu-sdpa-math-fp32 | llama8b.decode.t1.ctx512 | 3 | 9.771 | 9.534 | 9.906 | 10.141 | ec9bd0d9e8bc6284 |
| transformer | torch-gpu-sdpa-math-fp32 | llama8b.prefill.t1 | 3 | 10.001 | 9.090 | 10.609 | 9.379 | 64fcadaa649bc195 |
| transformer | torch-gpu-sdpa-math-fp32 | llama8b.prefill.t128 | 3 | 24.385 | 23.547 | 24.511 | 24.147 | - |
| transformer | torch-gpu-sdpa-math-fp32 | llama8b.prefill.t512 | 3 | 97.549 | 97.387 | 97.602 | 97.571 | - |
| transformer | torch-gpu-sdpa-math-fp32 | llama8b.prefill.t8 | 3 | 12.123 | 12.062 | 12.167 | 11.700 | c07cfefa1e45697b |
| transformer | torch-gpu-tf32 | lane.b2_l4_d32_kv2 | 3 | 2.373 | 2.039 | 2.373 | 4.261 | 014d7d70948e1158 |
| transformer | torch-gpu-tf32 | llama8b.decode.t1.ctx512 | 3 | 10.294 | 10.286 | 11.170 | 10.127 | 25d620f56e175aa1 |
| transformer | torch-gpu-tf32 | llama8b.prefill.t1 | 3 | 9.717 | 9.564 | 10.909 | 9.577 | 64fcadaa649bc195 |
| transformer | torch-gpu-tf32 | llama8b.prefill.t128 | 3 | 24.870 | 24.356 | 24.888 | 25.256 | - |
| transformer | torch-gpu-tf32 | llama8b.prefill.t512 | 3 | 101.400 | 101.238 | 102.319 | 101.185 | - |
| transformer | torch-gpu-tf32 | llama8b.prefill.t8 | 3 | 12.502 | 12.415 | 12.520 | 12.034 | 809e8075f0a9ad74 |

## Ours against each opponent

| lane | shape | opponent | ours ms | theirs ms | ratio ours/theirs | verdict |
|---|---|---|---|---|---|---|
| attention | lane.b2_l4_d32_kv2 | torch-gpu-flash-bf16 | 4.282 | 0.557 | 7.68 | we are 7.68x SLOWER |
| attention | lane.b2_l4_d32_kv2 | torch-gpu-flash-fp32 | 4.282 | 0.564 | 7.59 | we are 7.59x SLOWER |
| attention | lane.b2_l4_d32_kv2 | torch-gpu-fp32 | 4.282 | 2.302 | 1.86 | we are 1.86x SLOWER |
| attention | lane.b2_l4_d32_kv2 | torch-gpu-sdpa-efficient-fp32 | 4.282 | 0.563 | 7.61 | we are 7.61x SLOWER |
| attention | lane.b2_l4_d32_kv2 | torch-gpu-sdpa-math-fp32 | 4.282 | 0.570 | 7.52 | we are 7.52x SLOWER |
| attention | lane.b2_l4_d32_kv2 | torch-gpu-tf32 | 4.282 | 2.155 | 1.99 | we are 1.99x SLOWER |
| attention | llama8b.decode.t1.ctx512 | torch-gpu-flash-bf16 | 18.832 | 2.463 | 7.65 | we are 7.65x SLOWER |
| attention | llama8b.decode.t1.ctx512 | torch-gpu-flash-fp32 | 18.832 | 3.312 | 5.69 | we are 5.69x SLOWER |
| attention | llama8b.decode.t1.ctx512 | torch-gpu-fp32 | 18.832 | 4.063 | 4.64 | we are 4.64x SLOWER |
| attention | llama8b.decode.t1.ctx512 | torch-gpu-sdpa-efficient-fp32 | 18.832 | 2.834 | 6.65 | we are 6.65x SLOWER |
| attention | llama8b.decode.t1.ctx512 | torch-gpu-sdpa-math-fp32 | 18.832 | 2.688 | 7.01 | we are 7.01x SLOWER |
| attention | llama8b.decode.t1.ctx512 | torch-gpu-tf32 | 18.832 | 3.648 | 5.16 | we are 5.16x SLOWER |
| attention | llama8b.prefill.t1 | torch-gpu-flash-bf16 | 16.171 | 1.578 | 10.25 | we are 10.25x SLOWER |
| attention | llama8b.prefill.t1 | torch-gpu-flash-fp32 | 16.171 | 2.172 | 7.45 | we are 7.45x SLOWER |
| attention | llama8b.prefill.t1 | torch-gpu-fp32 | 16.171 | 4.197 | 3.85 | we are 3.85x SLOWER |
| attention | llama8b.prefill.t1 | torch-gpu-sdpa-efficient-fp32 | 16.171 | 2.152 | 7.51 | we are 7.51x SLOWER |
| attention | llama8b.prefill.t1 | torch-gpu-sdpa-math-fp32 | 16.171 | 2.199 | 7.35 | we are 7.35x SLOWER |
| attention | llama8b.prefill.t1 | torch-gpu-tf32 | 16.171 | 3.358 | 4.82 | we are 4.82x SLOWER |
| attention | llama8b.prefill.t128 | torch-gpu-flash-bf16 | 27.133 | 4.851 | 5.59 | we are 5.59x SLOWER |
| attention | llama8b.prefill.t128 | torch-gpu-flash-fp32 | 27.133 | 5.604 | 4.84 | we are 4.84x SLOWER |
| attention | llama8b.prefill.t128 | torch-gpu-fp32 | 27.133 | 7.594 | 3.57 | we are 3.57x SLOWER |
| attention | llama8b.prefill.t128 | torch-gpu-sdpa-efficient-fp32 | 27.133 | 5.615 | 4.83 | we are 4.83x SLOWER |
| attention | llama8b.prefill.t128 | torch-gpu-sdpa-math-fp32 | 27.133 | 5.612 | 4.83 | we are 4.83x SLOWER |
| attention | llama8b.prefill.t128 | torch-gpu-tf32 | 27.133 | 6.964 | 3.90 | we are 3.90x SLOWER |
| attention | llama8b.prefill.t512 | torch-gpu-flash-bf16 | 88.574 | 18.524 | 4.78 | we are 4.78x SLOWER |
| attention | llama8b.prefill.t512 | torch-gpu-flash-fp32 | 88.574 | 22.073 | 4.01 | we are 4.01x SLOWER |
| attention | llama8b.prefill.t512 | torch-gpu-fp32 | 88.574 | 27.636 | 3.21 | we are 3.21x SLOWER |
| attention | llama8b.prefill.t512 | torch-gpu-sdpa-efficient-fp32 | 88.574 | 23.663 | 3.74 | we are 3.74x SLOWER |
| attention | llama8b.prefill.t512 | torch-gpu-sdpa-math-fp32 | 88.574 | 21.860 | 4.05 | we are 4.05x SLOWER |
| attention | llama8b.prefill.t512 | torch-gpu-tf32 | 88.574 | 27.246 | 3.25 | we are 3.25x SLOWER |
| attention | llama8b.prefill.t8 | torch-gpu-flash-bf16 | 20.762 | 1.565 | 13.27 | we are 13.27x SLOWER |
| attention | llama8b.prefill.t8 | torch-gpu-flash-fp32 | 20.762 | 3.193 | 6.50 | we are 6.50x SLOWER |
| attention | llama8b.prefill.t8 | torch-gpu-fp32 | 20.762 | 4.676 | 4.44 | we are 4.44x SLOWER |
| attention | llama8b.prefill.t8 | torch-gpu-sdpa-efficient-fp32 | 20.762 | 3.171 | 6.55 | we are 6.55x SLOWER |
| attention | llama8b.prefill.t8 | torch-gpu-sdpa-math-fp32 | 20.762 | 3.090 | 6.72 | we are 6.72x SLOWER |
| attention | llama8b.prefill.t8 | torch-gpu-tf32 | 20.762 | 3.831 | 5.42 | we are 5.42x SLOWER |
| gemm | gram.128sq.x100003 | mps-default | 4.212 | 1.894 | 2.22 | we are 2.22x SLOWER |
| gemm | gram.32x32x1M | mps-default | 8.041 | 8.340 | 0.96 | we are 1.04x FASTER |
| gemm | gram.32x32x64K | mps-default | 0.856 | 0.756 | 1.13 | we are 1.13x SLOWER |
| gemm | kmeans.dist.4096x64x64 | mps-default | 0.279 | 0.260 | 1.07 | we are 1.07x SLOWER |
| gemm | llama8b.lm_head.t1 | mps-default | 24.148 | 22.179 | 1.09 | we are 1.09x SLOWER |
| gemm | llama8b.lm_head.t512 | mps-default | 251.424 | 238.973 | 1.05 | we are 1.05x SLOWER |
| gemm | llama8b.lm_head.t8 | mps-default | 40.071 | 29.002 | 1.38 | we are 1.38x SLOWER |
| gemm | llama8b.mlp_down.t1 | mps-default | 2.997 | 2.763 | 1.08 | we are 1.08x SLOWER |
| gemm | llama8b.mlp_down.t512 | mps-default | 53.140 | 22.665 | 2.34 | we are 2.34x SLOWER |
| gemm | llama8b.mlp_down.t8 | mps-default | 4.853 | 3.539 | 1.37 | we are 1.37x SLOWER |
| gemm | llama8b.mlp_up.t1 | mps-default | 3.052 | 2.976 | 1.03 | we are 1.03x SLOWER |
| gemm | llama8b.mlp_up.t512 | mps-default | 36.506 | 22.494 | 1.62 | we are 1.62x SLOWER |
| gemm | llama8b.mlp_up.t8 | mps-default | 9.404 | 3.892 | 2.42 | we are 2.42x SLOWER |
| gemm | llama8b.qkv.t1 | mps-default | 0.947 | 0.990 | 0.96 | we are 1.05x FASTER |
| gemm | llama8b.qkv.t512 | mps-default | 17.047 | 6.508 | 2.62 | we are 2.62x SLOWER |
| gemm | llama8b.qkv.t8 | mps-default | 2.067 | 1.195 | 1.73 | we are 1.73x SLOWER |
| gemm | ols.predict.gemv.64Kx16 | mps-default | 0.503 | 0.239 | 2.11 | we are 2.11x SLOWER |
| gemm | ols.step1.16x16x64K | mps-default | 0.705 | 0.761 | 0.93 | we are 1.08x FASTER |
| gemm | pca.transform.8192x4x4 | mps-default | 0.209 | 0.220 | 0.95 | we are 1.05x FASTER |
| gemm | pca.transform.wide.8192x64x128 | mps-default | 0.431 | 0.329 | 1.31 | we are 1.31x SLOWER |
| mamba | lane.b2_l4_d8 | (none ran) | 4.480 | - | - | NO OPPONENT ON THIS BOX |
| mamba | mamba130m.prefill.t1 | (none ran) | 4.927 | - | - | NO OPPONENT ON THIS BOX |
| mamba | mamba130m.prefill.t128 | (none ran) | 5.019 | - | - | NO OPPONENT ON THIS BOX |
| mamba | mamba130m.prefill.t512 | (none ran) | 7.749 | - | - | NO OPPONENT ON THIS BOX |
| mamba | mamba130m.prefill.t8 | (none ran) | 5.352 | - | - | NO OPPONENT ON THIS BOX |
| mlp | lane.b2_l4_d32_kv2 | torch-gpu-fp32 | 0.688 | 0.537 | 1.28 | we are 1.28x SLOWER |
| mlp | lane.b2_l4_d32_kv2 | torch-gpu-tf32 | 0.688 | 0.298 | 2.31 | we are 2.31x SLOWER |
| mlp | llama8b.decode.t1.ctx512 | torch-gpu-fp32 | 7.744 | 7.374 | 1.05 | we are 1.05x SLOWER |
| mlp | llama8b.decode.t1.ctx512 | torch-gpu-tf32 | 7.744 | 7.062 | 1.10 | we are 1.10x SLOWER |
| mlp | llama8b.prefill.t1 | torch-gpu-fp32 | 10.972 | 7.740 | 1.42 | we are 1.42x SLOWER |
| mlp | llama8b.prefill.t1 | torch-gpu-tf32 | 10.972 | 7.015 | 1.56 | we are 1.56x SLOWER |
| mlp | llama8b.prefill.t128 | torch-gpu-fp32 | 24.744 | 18.379 | 1.35 | we are 1.35x SLOWER |
| mlp | llama8b.prefill.t128 | torch-gpu-tf32 | 24.744 | 17.985 | 1.38 | we are 1.38x SLOWER |
| mlp | llama8b.prefill.t512 | torch-gpu-fp32 | 111.150 | 71.586 | 1.55 | we are 1.55x SLOWER |
| mlp | llama8b.prefill.t512 | torch-gpu-tf32 | 111.150 | 71.771 | 1.55 | we are 1.55x SLOWER |
| mlp | llama8b.prefill.t8 | torch-gpu-fp32 | 13.181 | 9.335 | 1.41 | we are 1.41x SLOWER |
| mlp | llama8b.prefill.t8 | torch-gpu-tf32 | 13.181 | 9.544 | 1.38 | we are 1.38x SLOWER |
| rmsnorm | lane.b2_l4_d32_kv2 | torch-gpu-fp32 | 0.269 | 0.494 | 0.54 | we are 1.84x FASTER |
| rmsnorm | lane.b2_l4_d32_kv2 | torch-gpu-tf32 | 0.269 | 0.440 | 0.61 | we are 1.64x FASTER |
| rmsnorm | llama8b.decode.t1.ctx512 | torch-gpu-fp32 | 3.277 | 0.617 | 5.31 | we are 5.31x SLOWER |
| rmsnorm | llama8b.decode.t1.ctx512 | torch-gpu-tf32 | 3.277 | 0.226 | 14.48 | we are 14.48x SLOWER |
| rmsnorm | llama8b.prefill.t1 | torch-gpu-fp32 | 3.137 | 0.584 | 5.37 | we are 5.37x SLOWER |
| rmsnorm | llama8b.prefill.t1 | torch-gpu-tf32 | 3.137 | 0.503 | 6.24 | we are 6.24x SLOWER |
| rmsnorm | llama8b.prefill.t128 | torch-gpu-fp32 | 2.364 | 1.035 | 2.28 | we are 2.28x SLOWER |
| rmsnorm | llama8b.prefill.t128 | torch-gpu-tf32 | 2.364 | 0.510 | 4.64 | we are 4.64x SLOWER |
| rmsnorm | llama8b.prefill.t512 | torch-gpu-fp32 | 4.144 | 2.070 | 2.00 | we are 2.00x SLOWER |
| rmsnorm | llama8b.prefill.t512 | torch-gpu-tf32 | 4.144 | 1.498 | 2.77 | we are 2.77x SLOWER |
| rmsnorm | llama8b.prefill.t8 | torch-gpu-fp32 | 3.178 | 0.611 | 5.20 | we are 5.20x SLOWER |
| rmsnorm | llama8b.prefill.t8 | torch-gpu-tf32 | 3.178 | 0.528 | 6.02 | we are 6.02x SLOWER |
| selective_scan | lane.b2_l4_d8 | (none ran) | 0.267 | - | - | NO OPPONENT ON THIS BOX |
| selective_scan | mamba130m.prefill.t1 | (none ran) | 0.270 | - | - | NO OPPONENT ON THIS BOX |
| selective_scan | mamba130m.prefill.t128 | (none ran) | 0.877 | - | - | NO OPPONENT ON THIS BOX |
| selective_scan | mamba130m.prefill.t512 | (none ran) | 1.166 | - | - | NO OPPONENT ON THIS BOX |
| selective_scan | mamba130m.prefill.t8 | (none ran) | 0.195 | - | - | NO OPPONENT ON THIS BOX |
| transformer | lane.b2_l4_d32_kv2 | torch-gpu-flash-bf16 | 8.707 | 1.227 | 7.09 | we are 7.09x SLOWER |
| transformer | lane.b2_l4_d32_kv2 | torch-gpu-flash-fp32 | 8.707 | 1.086 | 8.02 | we are 8.02x SLOWER |
| transformer | lane.b2_l4_d32_kv2 | torch-gpu-fp32 | 8.707 | 2.295 | 3.79 | we are 3.79x SLOWER |
| transformer | lane.b2_l4_d32_kv2 | torch-gpu-sdpa-efficient-fp32 | 8.707 | 1.253 | 6.95 | we are 6.95x SLOWER |
| transformer | lane.b2_l4_d32_kv2 | torch-gpu-sdpa-math-fp32 | 8.707 | 1.354 | 6.43 | we are 6.43x SLOWER |
| transformer | lane.b2_l4_d32_kv2 | torch-gpu-tf32 | 8.707 | 2.373 | 3.67 | we are 3.67x SLOWER |
| transformer | llama8b.decode.t1.ctx512 | torch-gpu-flash-bf16 | 33.589 | 5.664 | 5.93 | we are 5.93x SLOWER |
| transformer | llama8b.decode.t1.ctx512 | torch-gpu-flash-fp32 | 33.589 | 9.556 | 3.51 | we are 3.51x SLOWER |
| transformer | llama8b.decode.t1.ctx512 | torch-gpu-fp32 | 33.589 | 11.508 | 2.92 | we are 2.92x SLOWER |
| transformer | llama8b.decode.t1.ctx512 | torch-gpu-sdpa-efficient-fp32 | 33.589 | 9.464 | 3.55 | we are 3.55x SLOWER |
| transformer | llama8b.decode.t1.ctx512 | torch-gpu-sdpa-math-fp32 | 33.589 | 9.771 | 3.44 | we are 3.44x SLOWER |
| transformer | llama8b.decode.t1.ctx512 | torch-gpu-tf32 | 33.589 | 10.294 | 3.26 | we are 3.26x SLOWER |
| transformer | llama8b.prefill.t1 | torch-gpu-flash-bf16 | 28.178 | 5.757 | 4.89 | we are 4.89x SLOWER |
| transformer | llama8b.prefill.t1 | torch-gpu-flash-fp32 | 28.178 | 10.881 | 2.59 | we are 2.59x SLOWER |
| transformer | llama8b.prefill.t1 | torch-gpu-fp32 | 28.178 | 10.955 | 2.57 | we are 2.57x SLOWER |
| transformer | llama8b.prefill.t1 | torch-gpu-sdpa-efficient-fp32 | 28.178 | 9.225 | 3.05 | we are 3.05x SLOWER |
| transformer | llama8b.prefill.t1 | torch-gpu-sdpa-math-fp32 | 28.178 | 10.001 | 2.82 | we are 2.82x SLOWER |
| transformer | llama8b.prefill.t1 | torch-gpu-tf32 | 28.178 | 9.717 | 2.90 | we are 2.90x SLOWER |
| transformer | llama8b.prefill.t128 | torch-gpu-flash-bf16 | 55.527 | 20.300 | 2.74 | we are 2.74x SLOWER |
| transformer | llama8b.prefill.t128 | torch-gpu-flash-fp32 | 55.527 | 24.384 | 2.28 | we are 2.28x SLOWER |
| transformer | llama8b.prefill.t128 | torch-gpu-fp32 | 55.527 | 24.852 | 2.23 | we are 2.23x SLOWER |
| transformer | llama8b.prefill.t128 | torch-gpu-sdpa-efficient-fp32 | 55.527 | 24.482 | 2.27 | we are 2.27x SLOWER |
| transformer | llama8b.prefill.t128 | torch-gpu-sdpa-math-fp32 | 55.527 | 24.385 | 2.28 | we are 2.28x SLOWER |
| transformer | llama8b.prefill.t128 | torch-gpu-tf32 | 55.527 | 24.870 | 2.23 | we are 2.23x SLOWER |
| transformer | llama8b.prefill.t512 | torch-gpu-flash-bf16 | 221.114 | 80.126 | 2.76 | we are 2.76x SLOWER |
| transformer | llama8b.prefill.t512 | torch-gpu-flash-fp32 | 221.114 | 100.280 | 2.20 | we are 2.20x SLOWER |
| transformer | llama8b.prefill.t512 | torch-gpu-fp32 | 221.114 | 101.787 | 2.17 | we are 2.17x SLOWER |
| transformer | llama8b.prefill.t512 | torch-gpu-sdpa-efficient-fp32 | 221.114 | 96.518 | 2.29 | we are 2.29x SLOWER |
| transformer | llama8b.prefill.t512 | torch-gpu-sdpa-math-fp32 | 221.114 | 97.549 | 2.27 | we are 2.27x SLOWER |
| transformer | llama8b.prefill.t512 | torch-gpu-tf32 | 221.114 | 101.400 | 2.18 | we are 2.18x SLOWER |
| transformer | llama8b.prefill.t8 | torch-gpu-flash-bf16 | 35.603 | 5.640 | 6.31 | we are 6.31x SLOWER |
| transformer | llama8b.prefill.t8 | torch-gpu-flash-fp32 | 35.603 | 11.974 | 2.97 | we are 2.97x SLOWER |
| transformer | llama8b.prefill.t8 | torch-gpu-fp32 | 35.603 | 12.724 | 2.80 | we are 2.80x SLOWER |
| transformer | llama8b.prefill.t8 | torch-gpu-sdpa-efficient-fp32 | 35.603 | 11.940 | 2.98 | we are 2.98x SLOWER |
| transformer | llama8b.prefill.t8 | torch-gpu-sdpa-math-fp32 | 35.603 | 12.123 | 2.94 | we are 2.94x SLOWER |
| transformer | llama8b.prefill.t8 | torch-gpu-tf32 | 35.603 | 12.502 | 2.85 | we are 2.85x SLOWER |

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
| 1 | rmsnorm | llama8b.decode.t1.ctx512 | 3.277 | torch-gpu-tf32 | 0.226 | **14.48x SLOWER** |
| 2 | attention | llama8b.prefill.t8 | 20.762 | torch-gpu-flash-bf16 | 1.565 | **13.27x SLOWER** |
| 3 | attention | llama8b.prefill.t1 | 16.171 | torch-gpu-flash-bf16 | 1.578 | **10.25x SLOWER** |
| 4 | transformer | lane.b2_l4_d32_kv2 | 8.707 | torch-gpu-flash-fp32 | 1.086 | **8.02x SLOWER** |
| 5 | attention | lane.b2_l4_d32_kv2 | 4.282 | torch-gpu-flash-bf16 | 0.557 | **7.68x SLOWER** |
| 6 | attention | llama8b.decode.t1.ctx512 | 18.832 | torch-gpu-flash-bf16 | 2.463 | **7.65x SLOWER** |
| 7 | transformer | llama8b.prefill.t8 | 35.603 | torch-gpu-flash-bf16 | 5.640 | **6.31x SLOWER** |
| 8 | rmsnorm | llama8b.prefill.t1 | 3.137 | torch-gpu-tf32 | 0.503 | **6.24x SLOWER** |
| 9 | rmsnorm | llama8b.prefill.t8 | 3.178 | torch-gpu-tf32 | 0.528 | **6.02x SLOWER** |
| 10 | transformer | llama8b.decode.t1.ctx512 | 33.589 | torch-gpu-flash-bf16 | 5.664 | **5.93x SLOWER** |
| 11 | attention | llama8b.prefill.t128 | 27.133 | torch-gpu-flash-bf16 | 4.851 | **5.59x SLOWER** |
| 12 | transformer | llama8b.prefill.t1 | 28.178 | torch-gpu-flash-bf16 | 5.757 | **4.89x SLOWER** |
| 13 | attention | llama8b.prefill.t512 | 88.574 | torch-gpu-flash-bf16 | 18.524 | **4.78x SLOWER** |
| 14 | rmsnorm | llama8b.prefill.t128 | 2.364 | torch-gpu-tf32 | 0.510 | **4.64x SLOWER** |
| 15 | rmsnorm | llama8b.prefill.t512 | 4.144 | torch-gpu-tf32 | 1.498 | **2.77x SLOWER** |
| 16 | transformer | llama8b.prefill.t512 | 221.114 | torch-gpu-flash-bf16 | 80.126 | **2.76x SLOWER** |
| 17 | transformer | llama8b.prefill.t128 | 55.527 | torch-gpu-flash-bf16 | 20.300 | **2.74x SLOWER** |
| 18 | gemm | llama8b.qkv.t512 | 17.047 | mps-default | 6.508 | **2.62x SLOWER** |
| 19 | gemm | llama8b.mlp_up.t8 | 9.404 | mps-default | 3.892 | **2.42x SLOWER** |
| 20 | gemm | llama8b.mlp_down.t512 | 53.140 | mps-default | 22.665 | **2.34x SLOWER** |
| 21 | mlp | lane.b2_l4_d32_kv2 | 0.688 | torch-gpu-tf32 | 0.298 | **2.31x SLOWER** |
| 22 | gemm | gram.128sq.x100003 | 4.212 | mps-default | 1.894 | **2.22x SLOWER** |
| 23 | gemm | ols.predict.gemv.64Kx16 | 0.503 | mps-default | 0.239 | **2.11x SLOWER** |
| 24 | gemm | llama8b.qkv.t8 | 2.067 | mps-default | 1.195 | **1.73x SLOWER** |
| 25 | gemm | llama8b.mlp_up.t512 | 36.506 | mps-default | 22.494 | **1.62x SLOWER** |

19 further rows not listed; 44 rows in this table.

44 rows in total have an opponent: 0 throughput, 44 fixed-cost.

## Did the two sides compute the same thing

A speed number for an arm that computes something different from
its opponent is worthless. This is that check, reported and not
gated: a large difference does not fail the run, it disqualifies
the ROW, and the row has to be readable to be disqualified.

| lane | max abs diff | max rel diff | n | source |
|---|---|---|---|---|
| attention | 2.98023e-07 | 2.65266e-05 | 256 | seq.attention.torch.log |
| attention | 2.09808e-05 | 0.00134321 | 4096 | seq.attention.torch.log |
| attention | 0.00086832 | 0.158728 | 32768 | seq.attention.torch.log |
| attention | 0.00238216 | 9.55512 | 524288 | seq.attention.torch.log |
| attention | 0.00815892 | 16.1063 | 2097152 | seq.attention.torch.log |
| attention | 1.85966e-05 | 0.00130415 | 4096 | seq.attention.torch.log |
| mlp | 7.45058e-08 | 1.77597e-05 | 256 | seq.mlp.torch.log |
| mlp | 4.57764e-05 | 0.000961717 | 4096 | seq.mlp.torch.log |
| mlp | 0.00454724 | 0.214703 | 32768 | seq.mlp.torch.log |
| mlp | 0.0115907 | 2.37797 | 524288 | seq.mlp.torch.log |
| mlp | 0.0343227 | 139.175 | 2097152 | seq.mlp.torch.log |
| mlp | 7.62939e-05 | 0.0175585 | 4096 | seq.mlp.torch.log |
| rmsnorm | 4.76837e-07 | 2.23803e-07 | 256 | seq.rmsnorm.torch.log |
| rmsnorm | 2.6226e-06 | 1.20776e-06 | 4096 | seq.rmsnorm.torch.log |
| rmsnorm | 2.14577e-06 | 9.9071e-07 | 32768 | seq.rmsnorm.torch.log |
| rmsnorm | 3.33786e-06 | 1.46874e-06 | 524288 | seq.rmsnorm.torch.log |
| rmsnorm | 4.76837e-06 | 2.01201e-06 | 2097152 | seq.rmsnorm.torch.log |
| rmsnorm | 7.15256e-07 | 3.66111e-07 | 4096 | seq.rmsnorm.torch.log |
| transformer | 3.57628e-07 | 6.73257e-06 | 256 | seq.transformer.torch.log |
| transformer | 5.34058e-05 | 0.00266389 | 4096 | seq.transformer.torch.log |
| transformer | 0.00491047 | 0.0709332 | 32768 | seq.transformer.torch.log |
| transformer | 0.0120764 | 1.89928 | 524288 | seq.transformer.torch.log |
| transformer | 0.0370274 | 84 | 2097152 | seq.transformer.torch.log |
| transformer | 8.01086e-05 | 0.00722394 | 4096 | seq.transformer.torch.log |

## Refused arms, kept rather than dropped

An arm that could not run is a result about this box and this image.
Deleting the row would make the table read as full coverage.

| lane | arm | reason |
|---|---|---|
| mamba | mamba-ssm-cuda | mamba_ssm is not importable (No module named 'mamba_ssm'). The only opponent left is the PURE-PYTORCH sequential reference scan, which is NOT what anyone deploys. |
| mamba | torch-ref-scan-gpu | ModuleNotFoundError at shape lane.b2_l4_d8: No module named 'einops' |
| mamba | torch-ref-scan-gpu-tf32 | ModuleNotFoundError at shape lane.b2_l4_d8: No module named 'einops' |
| mamba | agree | the reference arm did not produce an output at shape lane.b2_l4_d8 |
| mamba | mamba-ssm-cuda | mamba_ssm is not importable (No module named 'mamba_ssm'). The only opponent left is the PURE-PYTORCH sequential reference scan, which is NOT what anyone deploys. |
| mamba | torch-ref-scan-gpu | ModuleNotFoundError at shape mamba130m.prefill.t1: No module named 'einops' |
| mamba | torch-ref-scan-gpu-tf32 | ModuleNotFoundError at shape mamba130m.prefill.t1: No module named 'einops' |
| mamba | agree | the reference arm did not produce an output at shape mamba130m.prefill.t1 |
| mamba | mamba-ssm-cuda | mamba_ssm is not importable (No module named 'mamba_ssm'). The only opponent left is the PURE-PYTORCH sequential reference scan, which is NOT what anyone deploys. |
| mamba | torch-ref-scan-gpu | ModuleNotFoundError at shape mamba130m.prefill.t8: No module named 'einops' |
| mamba | torch-ref-scan-gpu-tf32 | ModuleNotFoundError at shape mamba130m.prefill.t8: No module named 'einops' |
| mamba | agree | the reference arm did not produce an output at shape mamba130m.prefill.t8 |
| mamba | mamba-ssm-cuda | mamba_ssm is not importable (No module named 'mamba_ssm'). The only opponent left is the PURE-PYTORCH sequential reference scan, which is NOT what anyone deploys. |
| mamba | torch-ref-scan-gpu | ModuleNotFoundError at shape mamba130m.prefill.t128: No module named 'einops' |
| mamba | torch-ref-scan-gpu-tf32 | ModuleNotFoundError at shape mamba130m.prefill.t128: No module named 'einops' |
| mamba | agree | the reference arm did not produce an output at shape mamba130m.prefill.t128 |
| mamba | mamba-ssm-cuda | mamba_ssm is not importable (No module named 'mamba_ssm'). The only opponent left is the PURE-PYTORCH sequential reference scan, which is NOT what anyone deploys. |
| mamba | torch-ref-scan-gpu | ModuleNotFoundError at shape mamba130m.prefill.t512: No module named 'einops' |
| mamba | torch-ref-scan-gpu-tf32 | ModuleNotFoundError at shape mamba130m.prefill.t512: No module named 'einops' |
| mamba | agree | the reference arm did not produce an output at shape mamba130m.prefill.t512 |
| selective_scan | mamba-ssm-cuda | mamba_ssm is not importable (No module named 'mamba_ssm'). The only opponent left is the PURE-PYTORCH sequential reference scan, which is NOT what anyone deploys. |
| selective_scan | torch-ref-scan-gpu | ModuleNotFoundError at shape lane.b2_l4_d8: No module named 'einops' |
| selective_scan | torch-ref-scan-gpu-tf32 | ModuleNotFoundError at shape lane.b2_l4_d8: No module named 'einops' |
| selective_scan | agree | the reference arm did not produce an output at shape lane.b2_l4_d8 |
| selective_scan | mamba-ssm-cuda | mamba_ssm is not importable (No module named 'mamba_ssm'). The only opponent left is the PURE-PYTORCH sequential reference scan, which is NOT what anyone deploys. |
| selective_scan | torch-ref-scan-gpu | ModuleNotFoundError at shape mamba130m.prefill.t1: No module named 'einops' |
| selective_scan | torch-ref-scan-gpu-tf32 | ModuleNotFoundError at shape mamba130m.prefill.t1: No module named 'einops' |
| selective_scan | agree | the reference arm did not produce an output at shape mamba130m.prefill.t1 |
| selective_scan | mamba-ssm-cuda | mamba_ssm is not importable (No module named 'mamba_ssm'). The only opponent left is the PURE-PYTORCH sequential reference scan, which is NOT what anyone deploys. |
| selective_scan | torch-ref-scan-gpu | ModuleNotFoundError at shape mamba130m.prefill.t8: No module named 'einops' |
| selective_scan | torch-ref-scan-gpu-tf32 | ModuleNotFoundError at shape mamba130m.prefill.t8: No module named 'einops' |
| selective_scan | agree | the reference arm did not produce an output at shape mamba130m.prefill.t8 |
| selective_scan | mamba-ssm-cuda | mamba_ssm is not importable (No module named 'mamba_ssm'). The only opponent left is the PURE-PYTORCH sequential reference scan, which is NOT what anyone deploys. |
| selective_scan | torch-ref-scan-gpu | ModuleNotFoundError at shape mamba130m.prefill.t128: No module named 'einops' |
| selective_scan | torch-ref-scan-gpu-tf32 | ModuleNotFoundError at shape mamba130m.prefill.t128: No module named 'einops' |
| selective_scan | agree | the reference arm did not produce an output at shape mamba130m.prefill.t128 |
| selective_scan | mamba-ssm-cuda | mamba_ssm is not importable (No module named 'mamba_ssm'). The only opponent left is the PURE-PYTORCH sequential reference scan, which is NOT what anyone deploys. |
| selective_scan | torch-ref-scan-gpu | ModuleNotFoundError at shape mamba130m.prefill.t512: No module named 'einops' |
| selective_scan | torch-ref-scan-gpu-tf32 | ModuleNotFoundError at shape mamba130m.prefill.t512: No module named 'einops' |
| selective_scan | agree | the reference arm did not produce an output at shape mamba130m.prefill.t512 |

## Determinism, reported and not judged

This is the FAST path. A hash that moves between rounds is EXPECTED here
and is recorded, not failed. It is the direct evidence for what the
IDENTICAL mode buys, measured on the same box in the same hour.

No arm's output hash moved across its rounds in this run.

## Notes the arms printed

- `gemm.gemm.cublas.log`: lane=gemm arm=mps-default library=MPS/MPSGraph build=torch 2.13.0 torch=2.13.0 allow_tf32=None
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
- `seq.attention.ours.log`: lane=attention arm=ours llama_refuse_bad_inputs alone shape=lane.b2_l4_d32_kv2 ms=2.399 (inside the transformer lane's round, not inside attention/mlp/rmsnorm)
- `seq.attention.ours.log`: lane=attention arm=ours dump shape=lane.b2_l4_d32_kv2 n=256 path=/Users/andrewhendel/CascadeProjects/mojolearn/bench/results/fast_speed/mac-2026-08-28_131915-gemmseq/dump/seq.attention.lane.b2_l4_d32_kv2.f32.bin
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
- `seq.attention.ours.log`: lane=attention arm=ours llama_refuse_bad_inputs alone shape=llama8b.prefill.t1 ms=1347.317 (inside the transformer lane's round, not inside attention/mlp/rmsnorm)
- `seq.attention.ours.log`: lane=attention arm=ours dump shape=llama8b.prefill.t1 n=4096 path=/Users/andrewhendel/CascadeProjects/mojolearn/bench/results/fast_speed/mac-2026-08-28_131915-gemmseq/dump/seq.attention.llama8b.prefill.t1.f32.bin
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
- `seq.attention.ours.log`: lane=attention arm=ours llama_refuse_bad_inputs alone shape=llama8b.prefill.t8 ms=1320.316 (inside the transformer lane's round, not inside attention/mlp/rmsnorm)
- `seq.attention.ours.log`: lane=attention arm=ours dump shape=llama8b.prefill.t8 n=32768 path=/Users/andrewhendel/CascadeProjects/mojolearn/bench/results/fast_speed/mac-2026-08-28_131915-gemmseq/dump/seq.attention.llama8b.prefill.t8.f32.bin
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
- `seq.attention.ours.log`: lane=attention arm=ours llama_refuse_bad_inputs alone shape=llama8b.prefill.t128 ms=1250.766 (inside the transformer lane's round, not inside attention/mlp/rmsnorm)
- `seq.attention.ours.log`: lane=attention arm=ours dump shape=llama8b.prefill.t128 n=524288 path=/Users/andrewhendel/CascadeProjects/mojolearn/bench/results/fast_speed/mac-2026-08-28_131915-gemmseq/dump/seq.attention.llama8b.prefill.t128.f32.bin
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
- `seq.attention.ours.log`: lane=attention arm=ours llama_refuse_bad_inputs alone shape=llama8b.prefill.t512 ms=1381.226 (inside the transformer lane's round, not inside attention/mlp/rmsnorm)
- `seq.attention.ours.log`: lane=attention arm=ours dump shape=llama8b.prefill.t512 n=2097152 path=/Users/andrewhendel/CascadeProjects/mojolearn/bench/results/fast_speed/mac-2026-08-28_131915-gemmseq/dump/seq.attention.llama8b.prefill.t512.f32.bin
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
- `seq.attention.ours.log`: lane=attention arm=ours llama_refuse_bad_inputs alone shape=llama8b.decode.t1.ctx512 ms=1230.347 (inside the transformer lane's round, not inside attention/mlp/rmsnorm)
- `seq.attention.ours.log`: lane=attention arm=ours dump shape=llama8b.decode.t1.ctx512 n=4096 path=/Users/andrewhendel/CascadeProjects/mojolearn/bench/results/fast_speed/mac-2026-08-28_131915-gemmseq/dump/seq.attention.llama8b.decode.t1.ctx512.f32.bin
- `seq.attention.ours.log`: lane=attention mode=FAST
- `seq.attention.torch.log`: lane=attention arm=torch build=MPS 2.13.0 torch=2.13.0 tf32_switches=cuda.matmul.allow_tf32,cudnn.allow_tf32,cuda.matmul.fp32_precision deterministic=off
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
- `seq.attention.torch.log`: lane=attention arm=torch-gpu-fp32 output witness shape=lane.b2_l4_d32_kv2 n=256 hash=1c464d02303b51dc
- `seq.attention.torch.log`: lane=attention arm=torch-gpu-tf32 output witness shape=lane.b2_l4_d32_kv2 n=256 hash=1c464d02303b51dc
- `seq.attention.torch.log`: lane=attention arm=torch-gpu-sdpa-math-fp32 output witness shape=lane.b2_l4_d32_kv2 n=256 hash=aceb31ca74fb96da
- `seq.attention.torch.log`: lane=attention arm=torch-gpu-sdpa-efficient-fp32 output witness shape=lane.b2_l4_d32_kv2 n=256 hash=aceb31ca74fb96da
- `seq.attention.torch.log`: lane=attention arm=torch-gpu-flash-fp32 output witness shape=lane.b2_l4_d32_kv2 n=256 hash=aceb31ca74fb96da
- `seq.attention.torch.log`: lane=attention arm=torch-gpu-flash-bf16 output witness shape=lane.b2_l4_d32_kv2 n=256 hash=d75aaf042d86b26a
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
- `seq.attention.torch.log`: lane=attention arm=torch-gpu-fp32 output witness shape=llama8b.prefill.t1 n=4096 hash=dcd189bc2aa6895f
- `seq.attention.torch.log`: lane=attention arm=torch-gpu-tf32 output witness shape=llama8b.prefill.t1 n=4096 hash=dcd189bc2aa6895f
- `seq.attention.torch.log`: lane=attention arm=torch-gpu-sdpa-math-fp32 output witness shape=llama8b.prefill.t1 n=4096 hash=dcd189bc2aa6895f
- `seq.attention.torch.log`: lane=attention arm=torch-gpu-sdpa-efficient-fp32 output witness shape=llama8b.prefill.t1 n=4096 hash=dcd189bc2aa6895f
- `seq.attention.torch.log`: lane=attention arm=torch-gpu-flash-fp32 output witness shape=llama8b.prefill.t1 n=4096 hash=dcd189bc2aa6895f
- `seq.attention.torch.log`: lane=attention arm=torch-gpu-flash-bf16 output witness shape=llama8b.prefill.t1 n=4096 hash=e03765092a7f565d
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
- `seq.attention.torch.log`: lane=attention arm=torch-gpu-fp32 output witness shape=llama8b.prefill.t8 n=32768 hash=1aad6e97af7db09e
- `seq.attention.torch.log`: lane=attention arm=torch-gpu-tf32 output witness shape=llama8b.prefill.t8 n=32768 hash=1aad6e97af7db09e
- `seq.attention.torch.log`: lane=attention arm=torch-gpu-sdpa-math-fp32 output witness shape=llama8b.prefill.t8 n=32768 hash=a64e27fcd45dd87e
- `seq.attention.torch.log`: lane=attention arm=torch-gpu-sdpa-efficient-fp32 output witness shape=llama8b.prefill.t8 n=32768 hash=a64e27fcd45dd87e
- `seq.attention.torch.log`: lane=attention arm=torch-gpu-flash-fp32 output witness shape=llama8b.prefill.t8 n=32768 hash=a64e27fcd45dd87e
- `seq.attention.torch.log`: lane=attention arm=torch-gpu-flash-bf16 output witness shape=llama8b.prefill.t8 n=32768 hash=bdfc58e5d720f393
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
- `seq.attention.torch.log`: lane=attention arm=torch-gpu-fp32 output witness shape=llama8b.prefill.t128 n=524288 hash=26444562580ec88e
- `seq.attention.torch.log`: lane=attention arm=torch-gpu-tf32 output witness shape=llama8b.prefill.t128 n=524288 hash=26444562580ec88e
- `seq.attention.torch.log`: lane=attention arm=torch-gpu-sdpa-math-fp32 output witness shape=llama8b.prefill.t128 n=524288 hash=b80fff3efb949aea
- `seq.attention.torch.log`: lane=attention arm=torch-gpu-sdpa-efficient-fp32 output witness shape=llama8b.prefill.t128 n=524288 hash=b80fff3efb949aea
- `seq.attention.torch.log`: lane=attention arm=torch-gpu-flash-fp32 output witness shape=llama8b.prefill.t128 n=524288 hash=b80fff3efb949aea
- `seq.attention.torch.log`: lane=attention arm=torch-gpu-flash-bf16 output witness shape=llama8b.prefill.t128 n=524288 hash=b44146896b5e11b3
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
- `seq.attention.torch.log`: lane=attention arm=torch-gpu-fp32 output witness shape=llama8b.prefill.t512 n=2097152 hash=ce68528bea00b3b3
- `seq.attention.torch.log`: lane=attention arm=torch-gpu-tf32 output witness shape=llama8b.prefill.t512 n=2097152 hash=ce68528bea00b3b3
- `seq.attention.torch.log`: lane=attention arm=torch-gpu-sdpa-math-fp32 output witness shape=llama8b.prefill.t512 n=2097152 hash=56e10bc6fdd2bcdd
- `seq.attention.torch.log`: lane=attention arm=torch-gpu-sdpa-efficient-fp32 output witness shape=llama8b.prefill.t512 n=2097152 hash=56e10bc6fdd2bcdd
- `seq.attention.torch.log`: lane=attention arm=torch-gpu-flash-fp32 output witness shape=llama8b.prefill.t512 n=2097152 hash=56e10bc6fdd2bcdd
- `seq.attention.torch.log`: lane=attention arm=torch-gpu-flash-bf16 output witness shape=llama8b.prefill.t512 n=2097152 hash=b173b4601e46949e
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
- `seq.attention.torch.log`: lane=attention arm=torch-gpu-fp32 output witness shape=llama8b.decode.t1.ctx512 n=4096 hash=c5b75965b7575017
- `seq.attention.torch.log`: lane=attention arm=torch-gpu-tf32 output witness shape=llama8b.decode.t1.ctx512 n=4096 hash=c5b75965b7575017
- `seq.attention.torch.log`: lane=attention arm=torch-gpu-sdpa-math-fp32 output witness shape=llama8b.decode.t1.ctx512 n=4096 hash=5c6f4165622bf7e3
- `seq.attention.torch.log`: lane=attention arm=torch-gpu-sdpa-efficient-fp32 output witness shape=llama8b.decode.t1.ctx512 n=4096 hash=5c6f4165622bf7e3
- `seq.attention.torch.log`: lane=attention arm=torch-gpu-flash-fp32 output witness shape=llama8b.decode.t1.ctx512 n=4096 hash=5c6f4165622bf7e3
- `seq.attention.torch.log`: lane=attention arm=torch-gpu-flash-bf16 output witness shape=llama8b.decode.t1.ctx512 n=4096 hash=07bd27566b6f7b64
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
- `seq.mamba.ours.log`: lane=mamba arm=ours mamba_refuse_bad_inputs alone shape=lane.b2_l4_d8 ms=0.808 (inside the mamba lane's round, not inside selective_scan)
- `seq.mamba.ours.log`: lane=mamba arm=ours dump shape=lane.b2_l4_d8 n=64 path=/Users/andrewhendel/CascadeProjects/mojolearn/bench/results/fast_speed/mac-2026-08-28_131915-gemmseq/dump/seq.mamba.lane.b2_l4_d8.f32.bin
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
- `seq.mamba.ours.log`: lane=mamba arm=ours mamba_refuse_bad_inputs alone shape=mamba130m.prefill.t1 ms=0.848 (inside the mamba lane's round, not inside selective_scan)
- `seq.mamba.ours.log`: lane=mamba arm=ours dump shape=mamba130m.prefill.t1 n=768 path=/Users/andrewhendel/CascadeProjects/mojolearn/bench/results/fast_speed/mac-2026-08-28_131915-gemmseq/dump/seq.mamba.mamba130m.prefill.t1.f32.bin
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
- `seq.mamba.ours.log`: lane=mamba arm=ours mamba_refuse_bad_inputs alone shape=mamba130m.prefill.t8 ms=0.671 (inside the mamba lane's round, not inside selective_scan)
- `seq.mamba.ours.log`: lane=mamba arm=ours dump shape=mamba130m.prefill.t8 n=6144 path=/Users/andrewhendel/CascadeProjects/mojolearn/bench/results/fast_speed/mac-2026-08-28_131915-gemmseq/dump/seq.mamba.mamba130m.prefill.t8.f32.bin
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
- `seq.mamba.ours.log`: lane=mamba arm=ours mamba_refuse_bad_inputs alone shape=mamba130m.prefill.t128 ms=1.191 (inside the mamba lane's round, not inside selective_scan)
- `seq.mamba.ours.log`: lane=mamba arm=ours dump shape=mamba130m.prefill.t128 n=98304 path=/Users/andrewhendel/CascadeProjects/mojolearn/bench/results/fast_speed/mac-2026-08-28_131915-gemmseq/dump/seq.mamba.mamba130m.prefill.t128.f32.bin
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
- `seq.mamba.ours.log`: lane=mamba arm=ours mamba_refuse_bad_inputs alone shape=mamba130m.prefill.t512 ms=2.103 (inside the mamba lane's round, not inside selective_scan)
- `seq.mamba.ours.log`: lane=mamba arm=ours dump shape=mamba130m.prefill.t512 n=393216 path=/Users/andrewhendel/CascadeProjects/mojolearn/bench/results/fast_speed/mac-2026-08-28_131915-gemmseq/dump/seq.mamba.mamba130m.prefill.t512.f32.bin
- `seq.mamba.ours.log`: lane=mamba mode=FAST
- `seq.mamba.torch.log`: lane=mamba arm=torch build=MPS 2.13.0 torch=2.13.0 tf32_switches=cuda.matmul.allow_tf32,cudnn.allow_tf32,cuda.matmul.fp32_precision deterministic=off
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
- `seq.mlp.ours.log`: lane=mlp arm=ours llama_refuse_bad_inputs alone shape=lane.b2_l4_d32_kv2 ms=2.098 (inside the transformer lane's round, not inside attention/mlp/rmsnorm)
- `seq.mlp.ours.log`: lane=mlp arm=ours dump shape=lane.b2_l4_d32_kv2 n=256 path=/Users/andrewhendel/CascadeProjects/mojolearn/bench/results/fast_speed/mac-2026-08-28_131915-gemmseq/dump/seq.mlp.lane.b2_l4_d32_kv2.f32.bin
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
- `seq.mlp.ours.log`: lane=mlp arm=ours llama_refuse_bad_inputs alone shape=llama8b.prefill.t1 ms=1220.822 (inside the transformer lane's round, not inside attention/mlp/rmsnorm)
- `seq.mlp.ours.log`: lane=mlp arm=ours dump shape=llama8b.prefill.t1 n=4096 path=/Users/andrewhendel/CascadeProjects/mojolearn/bench/results/fast_speed/mac-2026-08-28_131915-gemmseq/dump/seq.mlp.llama8b.prefill.t1.f32.bin
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
- `seq.mlp.ours.log`: lane=mlp arm=ours llama_refuse_bad_inputs alone shape=llama8b.prefill.t8 ms=1232.714 (inside the transformer lane's round, not inside attention/mlp/rmsnorm)
- `seq.mlp.ours.log`: lane=mlp arm=ours dump shape=llama8b.prefill.t8 n=32768 path=/Users/andrewhendel/CascadeProjects/mojolearn/bench/results/fast_speed/mac-2026-08-28_131915-gemmseq/dump/seq.mlp.llama8b.prefill.t8.f32.bin
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
- `seq.mlp.ours.log`: lane=mlp arm=ours llama_refuse_bad_inputs alone shape=llama8b.prefill.t128 ms=1255.873 (inside the transformer lane's round, not inside attention/mlp/rmsnorm)
- `seq.mlp.ours.log`: lane=mlp arm=ours dump shape=llama8b.prefill.t128 n=524288 path=/Users/andrewhendel/CascadeProjects/mojolearn/bench/results/fast_speed/mac-2026-08-28_131915-gemmseq/dump/seq.mlp.llama8b.prefill.t128.f32.bin
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
- `seq.mlp.ours.log`: lane=mlp arm=ours llama_refuse_bad_inputs alone shape=llama8b.prefill.t512 ms=1386.602 (inside the transformer lane's round, not inside attention/mlp/rmsnorm)
- `seq.mlp.ours.log`: lane=mlp arm=ours dump shape=llama8b.prefill.t512 n=2097152 path=/Users/andrewhendel/CascadeProjects/mojolearn/bench/results/fast_speed/mac-2026-08-28_131915-gemmseq/dump/seq.mlp.llama8b.prefill.t512.f32.bin
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
- `seq.mlp.ours.log`: lane=mlp arm=ours llama_refuse_bad_inputs alone shape=llama8b.decode.t1.ctx512 ms=1246.309 (inside the transformer lane's round, not inside attention/mlp/rmsnorm)
- `seq.mlp.ours.log`: lane=mlp arm=ours dump shape=llama8b.decode.t1.ctx512 n=4096 path=/Users/andrewhendel/CascadeProjects/mojolearn/bench/results/fast_speed/mac-2026-08-28_131915-gemmseq/dump/seq.mlp.llama8b.decode.t1.ctx512.f32.bin
- `seq.mlp.ours.log`: lane=mlp mode=FAST
- `seq.mlp.torch.log`: lane=mlp arm=torch build=MPS 2.13.0 torch=2.13.0 tf32_switches=cuda.matmul.allow_tf32,cudnn.allow_tf32,cuda.matmul.fp32_precision deterministic=off
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
- `seq.mlp.torch.log`: lane=mlp arm=torch-gpu-fp32 output witness shape=lane.b2_l4_d32_kv2 n=256 hash=9bbbcfc5f41bf3c2
- `seq.mlp.torch.log`: lane=mlp arm=torch-gpu-tf32 output witness shape=lane.b2_l4_d32_kv2 n=256 hash=9bbbcfc5f41bf3c2
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
- `seq.mlp.torch.log`: lane=mlp arm=torch-gpu-fp32 output witness shape=llama8b.prefill.t1 n=4096 hash=cab71e91e62ef2fb
- `seq.mlp.torch.log`: lane=mlp arm=torch-gpu-tf32 output witness shape=llama8b.prefill.t1 n=4096 hash=cab71e91e62ef2fb
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
- `seq.mlp.torch.log`: lane=mlp arm=torch-gpu-fp32 output witness shape=llama8b.prefill.t8 n=32768 hash=4538ab378d143a86
- `seq.mlp.torch.log`: lane=mlp arm=torch-gpu-tf32 output witness shape=llama8b.prefill.t8 n=32768 hash=4538ab378d143a86
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
- `seq.mlp.torch.log`: lane=mlp arm=torch-gpu-fp32 output witness shape=llama8b.prefill.t128 n=524288 hash=0a0bf392f4af8cee
- `seq.mlp.torch.log`: lane=mlp arm=torch-gpu-tf32 output witness shape=llama8b.prefill.t128 n=524288 hash=0a0bf392f4af8cee
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
- `seq.mlp.torch.log`: lane=mlp arm=torch-gpu-fp32 output witness shape=llama8b.prefill.t512 n=2097152 hash=7f0019949b3a3ba8
- `seq.mlp.torch.log`: lane=mlp arm=torch-gpu-tf32 output witness shape=llama8b.prefill.t512 n=2097152 hash=7f0019949b3a3ba8
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
- `seq.mlp.torch.log`: lane=mlp arm=torch-gpu-fp32 output witness shape=llama8b.decode.t1.ctx512 n=4096 hash=3808aa8e354d41f7
- `seq.mlp.torch.log`: lane=mlp arm=torch-gpu-tf32 output witness shape=llama8b.decode.t1.ctx512 n=4096 hash=3808aa8e354d41f7
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
- `seq.rmsnorm.ours.log`: lane=rmsnorm arm=ours llama_refuse_bad_inputs alone shape=lane.b2_l4_d32_kv2 ms=4.134 (inside the transformer lane's round, not inside attention/mlp/rmsnorm)
- `seq.rmsnorm.ours.log`: lane=rmsnorm arm=ours dump shape=lane.b2_l4_d32_kv2 n=256 path=/Users/andrewhendel/CascadeProjects/mojolearn/bench/results/fast_speed/mac-2026-08-28_131915-gemmseq/dump/seq.rmsnorm.lane.b2_l4_d32_kv2.f32.bin
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
- `seq.rmsnorm.ours.log`: lane=rmsnorm arm=ours llama_refuse_bad_inputs alone shape=llama8b.prefill.t1 ms=1231.171 (inside the transformer lane's round, not inside attention/mlp/rmsnorm)
- `seq.rmsnorm.ours.log`: lane=rmsnorm arm=ours dump shape=llama8b.prefill.t1 n=4096 path=/Users/andrewhendel/CascadeProjects/mojolearn/bench/results/fast_speed/mac-2026-08-28_131915-gemmseq/dump/seq.rmsnorm.llama8b.prefill.t1.f32.bin
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
- `seq.rmsnorm.ours.log`: lane=rmsnorm arm=ours llama_refuse_bad_inputs alone shape=llama8b.prefill.t8 ms=1266.098 (inside the transformer lane's round, not inside attention/mlp/rmsnorm)
- `seq.rmsnorm.ours.log`: lane=rmsnorm arm=ours dump shape=llama8b.prefill.t8 n=32768 path=/Users/andrewhendel/CascadeProjects/mojolearn/bench/results/fast_speed/mac-2026-08-28_131915-gemmseq/dump/seq.rmsnorm.llama8b.prefill.t8.f32.bin
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
- `seq.rmsnorm.ours.log`: lane=rmsnorm arm=ours llama_refuse_bad_inputs alone shape=llama8b.prefill.t128 ms=1226.974 (inside the transformer lane's round, not inside attention/mlp/rmsnorm)
- `seq.rmsnorm.ours.log`: lane=rmsnorm arm=ours dump shape=llama8b.prefill.t128 n=524288 path=/Users/andrewhendel/CascadeProjects/mojolearn/bench/results/fast_speed/mac-2026-08-28_131915-gemmseq/dump/seq.rmsnorm.llama8b.prefill.t128.f32.bin
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
- `seq.rmsnorm.ours.log`: lane=rmsnorm arm=ours llama_refuse_bad_inputs alone shape=llama8b.prefill.t512 ms=1471.158 (inside the transformer lane's round, not inside attention/mlp/rmsnorm)
- `seq.rmsnorm.ours.log`: lane=rmsnorm arm=ours dump shape=llama8b.prefill.t512 n=2097152 path=/Users/andrewhendel/CascadeProjects/mojolearn/bench/results/fast_speed/mac-2026-08-28_131915-gemmseq/dump/seq.rmsnorm.llama8b.prefill.t512.f32.bin
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
- `seq.rmsnorm.ours.log`: lane=rmsnorm arm=ours llama_refuse_bad_inputs alone shape=llama8b.decode.t1.ctx512 ms=1228.366 (inside the transformer lane's round, not inside attention/mlp/rmsnorm)
- `seq.rmsnorm.ours.log`: lane=rmsnorm arm=ours dump shape=llama8b.decode.t1.ctx512 n=4096 path=/Users/andrewhendel/CascadeProjects/mojolearn/bench/results/fast_speed/mac-2026-08-28_131915-gemmseq/dump/seq.rmsnorm.llama8b.decode.t1.ctx512.f32.bin
- `seq.rmsnorm.ours.log`: lane=rmsnorm mode=FAST
- `seq.rmsnorm.torch.log`: lane=rmsnorm arm=torch build=MPS 2.13.0 torch=2.13.0 tf32_switches=cuda.matmul.allow_tf32,cudnn.allow_tf32,cuda.matmul.fp32_precision deterministic=off
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
- `seq.rmsnorm.torch.log`: lane=rmsnorm arm=torch-gpu-fp32 output witness shape=lane.b2_l4_d32_kv2 n=256 hash=7e3e6a4334c30308
- `seq.rmsnorm.torch.log`: lane=rmsnorm arm=torch-gpu-tf32 output witness shape=lane.b2_l4_d32_kv2 n=256 hash=7e3e6a4334c30308
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
- `seq.rmsnorm.torch.log`: lane=rmsnorm arm=torch-gpu-fp32 output witness shape=llama8b.prefill.t8 n=32768 hash=1ede88acf968164e
- `seq.rmsnorm.torch.log`: lane=rmsnorm arm=torch-gpu-tf32 output witness shape=llama8b.prefill.t8 n=32768 hash=1ede88acf968164e
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
- `seq.rmsnorm.torch.log`: lane=rmsnorm arm=torch-gpu-fp32 output witness shape=llama8b.prefill.t128 n=524288 hash=f23a79a7d15eb3cf
- `seq.rmsnorm.torch.log`: lane=rmsnorm arm=torch-gpu-tf32 output witness shape=llama8b.prefill.t128 n=524288 hash=f23a79a7d15eb3cf
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
- `seq.rmsnorm.torch.log`: lane=rmsnorm arm=torch-gpu-fp32 output witness shape=llama8b.prefill.t512 n=2097152 hash=2a2412b1cce01005
- `seq.rmsnorm.torch.log`: lane=rmsnorm arm=torch-gpu-tf32 output witness shape=llama8b.prefill.t512 n=2097152 hash=2a2412b1cce01005
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
- `seq.selective_scan.ours.log`: lane=selective_scan arm=ours mamba_refuse_bad_inputs alone shape=lane.b2_l4_d8 ms=1.775 (inside the mamba lane's round, not inside selective_scan)
- `seq.selective_scan.ours.log`: lane=selective_scan arm=ours dump shape=lane.b2_l4_d8 n=128 path=/Users/andrewhendel/CascadeProjects/mojolearn/bench/results/fast_speed/mac-2026-08-28_131915-gemmseq/dump/seq.selective_scan.lane.b2_l4_d8.f32.bin
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
- `seq.selective_scan.ours.log`: lane=selective_scan arm=ours mamba_refuse_bad_inputs alone shape=mamba130m.prefill.t1 ms=0.741 (inside the mamba lane's round, not inside selective_scan)
- `seq.selective_scan.ours.log`: lane=selective_scan arm=ours dump shape=mamba130m.prefill.t1 n=1536 path=/Users/andrewhendel/CascadeProjects/mojolearn/bench/results/fast_speed/mac-2026-08-28_131915-gemmseq/dump/seq.selective_scan.mamba130m.prefill.t1.f32.bin
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
- `seq.selective_scan.ours.log`: lane=selective_scan arm=ours mamba_refuse_bad_inputs alone shape=mamba130m.prefill.t8 ms=0.699 (inside the mamba lane's round, not inside selective_scan)
- `seq.selective_scan.ours.log`: lane=selective_scan arm=ours dump shape=mamba130m.prefill.t8 n=12288 path=/Users/andrewhendel/CascadeProjects/mojolearn/bench/results/fast_speed/mac-2026-08-28_131915-gemmseq/dump/seq.selective_scan.mamba130m.prefill.t8.f32.bin
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
- `seq.selective_scan.ours.log`: lane=selective_scan arm=ours mamba_refuse_bad_inputs alone shape=mamba130m.prefill.t128 ms=1.144 (inside the mamba lane's round, not inside selective_scan)
- `seq.selective_scan.ours.log`: lane=selective_scan arm=ours dump shape=mamba130m.prefill.t128 n=196608 path=/Users/andrewhendel/CascadeProjects/mojolearn/bench/results/fast_speed/mac-2026-08-28_131915-gemmseq/dump/seq.selective_scan.mamba130m.prefill.t128.f32.bin
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
- `seq.selective_scan.ours.log`: lane=selective_scan arm=ours mamba_refuse_bad_inputs alone shape=mamba130m.prefill.t512 ms=2.427 (inside the mamba lane's round, not inside selective_scan)
- `seq.selective_scan.ours.log`: lane=selective_scan arm=ours dump shape=mamba130m.prefill.t512 n=786432 path=/Users/andrewhendel/CascadeProjects/mojolearn/bench/results/fast_speed/mac-2026-08-28_131915-gemmseq/dump/seq.selective_scan.mamba130m.prefill.t512.f32.bin
- `seq.selective_scan.ours.log`: lane=selective_scan mode=FAST
- `seq.selective_scan.torch.log`: lane=selective_scan arm=torch build=MPS 2.13.0 torch=2.13.0 tf32_switches=cuda.matmul.allow_tf32,cudnn.allow_tf32,cuda.matmul.fp32_precision deterministic=off
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
- `seq.transformer.ours.log`: lane=transformer arm=ours llama_refuse_bad_inputs alone shape=lane.b2_l4_d32_kv2 ms=3.6 (inside the transformer lane's round, not inside attention/mlp/rmsnorm)
- `seq.transformer.ours.log`: lane=transformer arm=ours dump shape=lane.b2_l4_d32_kv2 n=256 path=/Users/andrewhendel/CascadeProjects/mojolearn/bench/results/fast_speed/mac-2026-08-28_131915-gemmseq/dump/seq.transformer.lane.b2_l4_d32_kv2.f32.bin
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
- `seq.transformer.ours.log`: lane=transformer arm=ours llama_refuse_bad_inputs alone shape=llama8b.prefill.t1 ms=1236.967 (inside the transformer lane's round, not inside attention/mlp/rmsnorm)
- `seq.transformer.ours.log`: lane=transformer arm=ours dump shape=llama8b.prefill.t1 n=4096 path=/Users/andrewhendel/CascadeProjects/mojolearn/bench/results/fast_speed/mac-2026-08-28_131915-gemmseq/dump/seq.transformer.llama8b.prefill.t1.f32.bin
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
- `seq.transformer.ours.log`: lane=transformer arm=ours llama_refuse_bad_inputs alone shape=llama8b.prefill.t8 ms=1225.834 (inside the transformer lane's round, not inside attention/mlp/rmsnorm)
- `seq.transformer.ours.log`: lane=transformer arm=ours dump shape=llama8b.prefill.t8 n=32768 path=/Users/andrewhendel/CascadeProjects/mojolearn/bench/results/fast_speed/mac-2026-08-28_131915-gemmseq/dump/seq.transformer.llama8b.prefill.t8.f32.bin
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
- `seq.transformer.ours.log`: lane=transformer arm=ours llama_refuse_bad_inputs alone shape=llama8b.prefill.t128 ms=1221.441 (inside the transformer lane's round, not inside attention/mlp/rmsnorm)
- `seq.transformer.ours.log`: lane=transformer arm=ours dump shape=llama8b.prefill.t128 n=524288 path=/Users/andrewhendel/CascadeProjects/mojolearn/bench/results/fast_speed/mac-2026-08-28_131915-gemmseq/dump/seq.transformer.llama8b.prefill.t128.f32.bin
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
- `seq.transformer.ours.log`: lane=transformer arm=ours llama_refuse_bad_inputs alone shape=llama8b.prefill.t512 ms=1242.177 (inside the transformer lane's round, not inside attention/mlp/rmsnorm)
- `seq.transformer.ours.log`: lane=transformer arm=ours dump shape=llama8b.prefill.t512 n=2097152 path=/Users/andrewhendel/CascadeProjects/mojolearn/bench/results/fast_speed/mac-2026-08-28_131915-gemmseq/dump/seq.transformer.llama8b.prefill.t512.f32.bin
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
- `seq.transformer.ours.log`: lane=transformer arm=ours llama_refuse_bad_inputs alone shape=llama8b.decode.t1.ctx512 ms=1229.419 (inside the transformer lane's round, not inside attention/mlp/rmsnorm)
- `seq.transformer.ours.log`: lane=transformer arm=ours dump shape=llama8b.decode.t1.ctx512 n=4096 path=/Users/andrewhendel/CascadeProjects/mojolearn/bench/results/fast_speed/mac-2026-08-28_131915-gemmseq/dump/seq.transformer.llama8b.decode.t1.ctx512.f32.bin
- `seq.transformer.ours.log`: lane=transformer mode=FAST
- `seq.transformer.torch.log`: lane=transformer arm=torch build=MPS 2.13.0 torch=2.13.0 tf32_switches=cuda.matmul.allow_tf32,cudnn.allow_tf32,cuda.matmul.fp32_precision deterministic=off
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
- `seq.transformer.torch.log`: lane=transformer arm=torch-gpu-fp32 output witness shape=lane.b2_l4_d32_kv2 n=256 hash=014d7d70948e1158
- `seq.transformer.torch.log`: lane=transformer arm=torch-gpu-tf32 output witness shape=lane.b2_l4_d32_kv2 n=256 hash=014d7d70948e1158
- `seq.transformer.torch.log`: lane=transformer arm=torch-gpu-sdpa-math-fp32 output witness shape=lane.b2_l4_d32_kv2 n=256 hash=d3262edc343c168f
- `seq.transformer.torch.log`: lane=transformer arm=torch-gpu-sdpa-efficient-fp32 output witness shape=lane.b2_l4_d32_kv2 n=256 hash=d3262edc343c168f
- `seq.transformer.torch.log`: lane=transformer arm=torch-gpu-flash-fp32 output witness shape=lane.b2_l4_d32_kv2 n=256 hash=d3262edc343c168f
- `seq.transformer.torch.log`: lane=transformer arm=torch-gpu-flash-bf16 output witness shape=lane.b2_l4_d32_kv2 n=256 hash=33a8ffb39891403a
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
- `seq.transformer.torch.log`: lane=transformer arm=torch-gpu-fp32 output witness shape=llama8b.prefill.t1 n=4096 hash=64fcadaa649bc195
- `seq.transformer.torch.log`: lane=transformer arm=torch-gpu-tf32 output witness shape=llama8b.prefill.t1 n=4096 hash=64fcadaa649bc195
- `seq.transformer.torch.log`: lane=transformer arm=torch-gpu-sdpa-math-fp32 output witness shape=llama8b.prefill.t1 n=4096 hash=64fcadaa649bc195
- `seq.transformer.torch.log`: lane=transformer arm=torch-gpu-sdpa-efficient-fp32 output witness shape=llama8b.prefill.t1 n=4096 hash=64fcadaa649bc195
- `seq.transformer.torch.log`: lane=transformer arm=torch-gpu-flash-fp32 output witness shape=llama8b.prefill.t1 n=4096 hash=64fcadaa649bc195
- `seq.transformer.torch.log`: lane=transformer arm=torch-gpu-flash-bf16 output witness shape=llama8b.prefill.t1 n=4096 hash=f992e21881c9d23e
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
- `seq.transformer.torch.log`: lane=transformer arm=torch-gpu-fp32 output witness shape=llama8b.prefill.t8 n=32768 hash=4c85341d2984a2af
- `seq.transformer.torch.log`: lane=transformer arm=torch-gpu-tf32 output witness shape=llama8b.prefill.t8 n=32768 hash=4c85341d2984a2af
- `seq.transformer.torch.log`: lane=transformer arm=torch-gpu-sdpa-math-fp32 output witness shape=llama8b.prefill.t8 n=32768 hash=763482c061917ec0
- `seq.transformer.torch.log`: lane=transformer arm=torch-gpu-sdpa-efficient-fp32 output witness shape=llama8b.prefill.t8 n=32768 hash=763482c061917ec0
- `seq.transformer.torch.log`: lane=transformer arm=torch-gpu-flash-fp32 output witness shape=llama8b.prefill.t8 n=32768 hash=763482c061917ec0
- `seq.transformer.torch.log`: lane=transformer arm=torch-gpu-flash-bf16 output witness shape=llama8b.prefill.t8 n=32768 hash=b4617bdcafa4fa21
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
- `seq.transformer.torch.log`: lane=transformer arm=torch-gpu-fp32 output witness shape=llama8b.prefill.t128 n=524288 hash=670a5025170ed0d6
- `seq.transformer.torch.log`: lane=transformer arm=torch-gpu-tf32 output witness shape=llama8b.prefill.t128 n=524288 hash=670a5025170ed0d6
- `seq.transformer.torch.log`: lane=transformer arm=torch-gpu-sdpa-math-fp32 output witness shape=llama8b.prefill.t128 n=524288 hash=3d63317a2ac95666
- `seq.transformer.torch.log`: lane=transformer arm=torch-gpu-sdpa-efficient-fp32 output witness shape=llama8b.prefill.t128 n=524288 hash=3d63317a2ac95666
- `seq.transformer.torch.log`: lane=transformer arm=torch-gpu-flash-fp32 output witness shape=llama8b.prefill.t128 n=524288 hash=3d63317a2ac95666
- `seq.transformer.torch.log`: lane=transformer arm=torch-gpu-flash-bf16 output witness shape=llama8b.prefill.t128 n=524288 hash=8875256d31f6702d
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
- `seq.transformer.torch.log`: lane=transformer arm=torch-gpu-fp32 output witness shape=llama8b.prefill.t512 n=2097152 hash=a7b6679953f06d33
- `seq.transformer.torch.log`: lane=transformer arm=torch-gpu-tf32 output witness shape=llama8b.prefill.t512 n=2097152 hash=a7b6679953f06d33
- `seq.transformer.torch.log`: lane=transformer arm=torch-gpu-sdpa-math-fp32 output witness shape=llama8b.prefill.t512 n=2097152 hash=4aedcc4a0675ec81
- `seq.transformer.torch.log`: lane=transformer arm=torch-gpu-sdpa-efficient-fp32 output witness shape=llama8b.prefill.t512 n=2097152 hash=4aedcc4a0675ec81
- `seq.transformer.torch.log`: lane=transformer arm=torch-gpu-flash-fp32 output witness shape=llama8b.prefill.t512 n=2097152 hash=4aedcc4a0675ec81
- `seq.transformer.torch.log`: lane=transformer arm=torch-gpu-flash-bf16 output witness shape=llama8b.prefill.t512 n=2097152 hash=5c048da1d9966438
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
- `seq.transformer.torch.log`: lane=transformer arm=torch-gpu-fp32 output witness shape=llama8b.decode.t1.ctx512 n=4096 hash=25d620f56e175aa1
- `seq.transformer.torch.log`: lane=transformer arm=torch-gpu-tf32 output witness shape=llama8b.decode.t1.ctx512 n=4096 hash=25d620f56e175aa1
- `seq.transformer.torch.log`: lane=transformer arm=torch-gpu-sdpa-math-fp32 output witness shape=llama8b.decode.t1.ctx512 n=4096 hash=ec9bd0d9e8bc6284
- `seq.transformer.torch.log`: lane=transformer arm=torch-gpu-sdpa-efficient-fp32 output witness shape=llama8b.decode.t1.ctx512 n=4096 hash=ec9bd0d9e8bc6284
- `seq.transformer.torch.log`: lane=transformer arm=torch-gpu-flash-fp32 output witness shape=llama8b.decode.t1.ctx512 n=4096 hash=ec9bd0d9e8bc6284
- `seq.transformer.torch.log`: lane=transformer arm=torch-gpu-flash-bf16 output witness shape=llama8b.decode.t1.ctx512 n=4096 hash=b393e512937ba972
- `seq.transformer.torch.log`: lane=transformer arm=torch

