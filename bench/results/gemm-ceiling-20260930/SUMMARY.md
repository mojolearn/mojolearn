# GEMM ceiling, NVIDIA L40S, 2026-09-30

GPU-resident, operands on the device, only the product timed. Ours: `bench/gemm_int15_price_main.mojo`
(identical mode, sm_89; arms alternated call by call; conversion costs counted in the complete operations).
PyTorch 2.13.0: `x @ y.t()` with CUDA events, median of 30. TFLOPS = 2mnk / median time.
Datasheet peaks (vendor, not measured): FP32 ~92, TF32 ~183, BF16 ~362, INT8 ~733 dense.

| shape (Llama-8B) | m x n x k | FP32 identical | fixed15 product | fixed15 training op | PyTorch FP32 | PyTorch TF32 | PyTorch BF16 |
|---|---|---:|---:|---:|---:|---:|---:|
| qkv prefill | 512x4096x4096 | 20.1 | 67.3 | 24.2 | 31.4 | 102.9 | 186.0 |
| qkv dX | 512x4096x4096 | 20.6 | 68.1 | 26.9 | 30.1 | 98.0 | 175.2 |
| qkv dW | 4096x4096x512 | 24.6 | 45.4 | 38.0 | 31.9 | 80.1 | 136.5 |
| mlp_up prefill | 512x14336x4096 | 19.5 | 18.2 | 12.0 | 29.8 | 59.6 | 133.6 |
| mlp_down prefill | 512x4096x14336 | 22.1 | 40.8 | 19.0 | 31.6 | 78.2 | 152.8 |
| mlp_down dW | 4096x14336x512 | 25.0 | 46.5 | 42.2 | 33.3 | 78.0 | 151.8 |
| lm_head dW (capped) | 16032x4096x512 | 24.9 | 47.5 | 43.0 | 32.2 | 75.8 | 153.8 |
| square (PyTorch only) | 8192^3 | | | | 31.4 | 90.3 | 198.3 |

fixed15 product = the tuned four-unit-product kernel alone on prepared planes; training op = quantize and
split both operands plus the product (best complete arm). Every fixed15 arm of a row has one digest.
Full per-arm table: nvidia-l40s/timed.log; PyTorch: nvidia-l40s/torch.json.
