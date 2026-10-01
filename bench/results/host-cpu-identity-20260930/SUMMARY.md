# Host-CPU and AMD-GPU identity divergences, 2026-09-30

**1. SGD-reg and LARS: the harness's input `y` depended on the OpenBLAS thread count, not on the library.**
`X @ w` (OpenBLAS sgemv) gives different bits for different thread counts and kernels (`amd/y_blas_sweep.txt`:
7 distinct `y` hashes at 20000 rows on the MI325X host). The L4 pod (64 visible CPUs) and the 20-CPU MI325X VM fed
the library different `y`. With OpenBLAS made to see 64 CPUs (an LD_PRELOAD shim), the released 0.8.31 CPU path on
the Zen 5 host reproduces the L4 digests exactly (`amd/old_inputs_cpu_view.txt`). Fix (harness only):
`host_independent_xw` (float64, column by column, no BLAS) and an `inputs` digest on every case. The sgd-reg and
lars harness reference digests change (new inputs, not new library bits):
sgd-reg 20000 431ad9f1214c29a2 -> 89cfa8f1827cd043, 1M f07701df3903e412 -> 23f7125a7f0738c6;
lars 200000 356afc301a0d55c1 -> 640cb53e8400f17d, 1M 88fba1aba928096e -> 4f5339c631d9eb2e.

**2. IVF 40000 on AMD: the k-means++ float block scan used the 64-lane wavefront.** Identity traces (`ivf-trace/`):
the first difference is the k-means++ seeding inside the IVF quantizer, pick 890 of 1024. `block.prefix_sum` adds
per hardware warp (32 lanes on Apple and NVIDIA, 64 on AMD). Fix: `pinned_block_prefix_sum`
(`core/pinned_reduce.mojo`), the unchanged library call where `WARP_SIZE == 32` and a shared-memory replay of the
32-lane order elsewhere (IDENTICAL only; FAST keeps the library call). Gate `pixi run check-pinned-scan` passes on the
M4 and the MI325X. The fixed MI325X trace equals the M4 trace at every stage; IVF 40000 on AMD goes
deec079c73e23671 -> 5bf7822421de6cd5 (the reference).

| case (fixed harness) | AMD GPU | AMD CPU | M4 CPU | M4 Metal |
|---|---|---|---|---|
| sgd-reg 20000 / 1M | 89cfa8f1827cd043 / 23f7125a7f0738c6 | same / same | same / same | 20000 same |
| lars 200000 / 1M | 640cb53e8400f17d / 4f5339c631d9eb2e | same / same | same / same | 200000 same |
| sgd-clf 20000 / 1M | a6cb75bf0e2f93da / bcac4b45fd82f48b | same / same | 20000 same | 20000 same |
| lu 1024 / 8192 | b53aeffc5bf56d98 / 2a296d1d25da0a2c | same / same | 1024 same | 1024 same |
| ivf 40000 / 400000 | 5bf7822421de6cd5 / 26a769d35c913838 | 40000 same | not run | released 5bf7822421de6cd5 |

**NVIDIA (L40S pod, released 0.8.32 + IVF rebuilt from this branch; `nvidia/`):** `check-pinned-scan` PASS (warp 32:
the library scan equals the replay). GPU and CPU columns: sgd-reg 89cfa8f1827cd043 / 23f7125a7f0738c6, lars
640cb53e8400f17d / 4f5339c631d9eb2e, sgd-clf a6cb75bf0e2f93da, lu b53aeffc5bf56d98, ivf 5bf7822421de6cd5 /
26a769d35c913838: every digest and every input digest equal to the AMD and M4 columns. All four columns agree.
