# CPU column (CUDA_VISIBLE_DEVICES=-1, kit vendor "cpu") check of the Lanczos KernelPCA, 2026-09-30

mojolearn 0.8.31 on the L40S pod with the GPU hidden; Python from lane/kernel-pca-default-on-20260930
(Lanczos route on every column). Same board-stride taxi input as the GPU runs.

| rows | passed | output sha256 | equals NVIDIA L40S | equals Apple M4 |
|---|---|---|---|---|
| 256 | yes | b92023de7a6e... | yes | yes |
| 1000 | yes | 45d3b6071431... | yes | yes |

AMD not run (no MI300X stock on RunPod or Hot Aisle).
