# H100 GEMM resources and scalar shared-read experiment

Dedicated RunPod onkqrqw3o3msdf, NVIDIA H100 80GB HBM3, UUID
GPU-1ba08457-69e7-4035-d268-d7898ff2c362, driver 580.126.09.
One-hour automatic termination armed before work. Trees' pod was not used.
Source base b69e8344, Mojo 1.0.0 (ed45d567). No opponent rerun.

## Resource capture

`cuda124-rejected` preserves the expected PTX-version refusal (PTX8.5 versus
assembler8.4). `cuda126` assembles the exact retained
`gemm_swizzle_2026-09-10/h100-current-128.ptx.gz` using ptxas12.6.85, installed
with `python -m pip install --no-deps --target /root/cuda126-tools
nvidia-cuda-nvcc-cu12==12.6.85`; the installer log is retained. cuobjdump is
12.4.127. Tool paths/hashes/versions and exact commands live in each manifest.

`query_occupancy.py` loads the resulting cubin with the CUDA driver API and
queries function resources and occupancy at block256, dynamic shared0. It
launches no kernel. Constants and signatures were read from the installed
CUDA12.4 `cuda.h`. Driver function resources:255 registers/thread,
4144 local bytes/thread,40960 shared bytes. Driver occupancy:1block/SM,
2048 maximum resident threads/SM, theoretical thread occupancy12.5%.
ptxas reports44bytes spill loads/stores. These are static compiler counts,
not measured traffic. The offline cubin is not proven equal to the runtime
JIT result; achieved occupancy remains unmeasured.

## Rejected scalar shared-read candidate

`scalar-shared.patch` forces full windows through the already existing
scalar ascending feature loop when its experimental define is present.
The patch is archived only, not enabled or added to production source.
The two exact compiled source files and executables are gzip-retained with
packed/unpacked hashes. Relative to b69e8344, the probe only additionally
prints both successful output digests. Build from `/root/performance`:

```sh
pixi run mojo build -D MOJOLEARN_NUMERIC_IDENTICAL=1 -I . gemm/checks/gemm_tuned_probe.mojo -o /root/jobs/gemm-baseline
pixi run mojo build -D MOJOLEARN_NUMERIC_IDENTICAL=1 -D MOJOLEARN_GEMM_SCALAR_SHARED_PROBE=1 -I . gemm/checks/gemm_tuned_probe.mojo -o /root/jobs/gemm-scalar
```

`run_prices.sh` is the exact four-process baseline/scalar/scalar/baseline
controller. Each process takes seven rounds after digest warmups, actual
m512 with n/k4096/14336 as printed. Inside each process the probe's two
selectors both use current dispatch; candidate samples are summarized
consistently, and every raw sample remains. Successful full-output digests
also match ACROSS executables for all three shapes; no poison remains.

| Shape | Baseline medians ms (first/last) | Scalar medians ms (two captures) |
|---|---:|---:|
| QKV |1.5007 /1.4998|1.5012 /1.5027|
| MLP up |6.1206 /6.1212|6.0838 /6.0853|
| MLP down |5.3073 /5.3202|5.3258 /5.3264|

QKV is flat; ~0.6% on up and a slight down regression do not justify a
change. REJECTED. No stronger admission/promotion campaign was run.
`scalar-cuda126` comes from the same candidate built with `--emit asm
--target-accelerator sm_90`; its offline result still uses255registers and
4144stack bytes, now48bytes spill stores/loads, and the driver still reports
oneblock/SM. The attempted change did not relieve the occupancy limit.

Before/after full device snapshots, source/binary hashes, raw samples and
summary are in `prices`. Timing is host-synchronized, not CUDA-event time.
The new sample medians do not replace the accepted staged opponent rows.

Rental cleanup: DELETE HTTP204 at09:06:32 EDT, verified absent with GET
HTTP404 at09:06:37. The lease was not extended.

Large PTX and SASS outputs are stored as deterministic gzip files. Original
resource manifests hash the uncompressed bytes; packed-artifacts.json maps
packed and unpacked hashes. Decompress before replaying a manifest check.
