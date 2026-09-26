# IDENTICAL CUDA sets carry machine code: no driver JIT, no higher driver floor (2026-09-26)

## The change

`packaging/linux/build_sets.sh`, after the `.rn` pass, downloads NVIDIA's `cuda_nvcc` 12.5.82
redist archive (pinned by sha256) and runs `packaging/linux/cubin_contract.py patch`: every
IDENTICAL PTX module of the set is compiled with `ptxas 12.5 -arch=<set arch> --fmad=false -O3`,
wrapped in an LZ4-compressed fatbin (`fatbinary --compress-all`) and written over the PTX string
in place. A tiny module whose fatbin is longer than its PTX gets its fatbin in the padding a
converted module left, and the RIP-relative `lea` instructions that reach the string are
repointed (objdump cross-checks each; the old PTX is zeroed). The wheel audit
(`packaging/portable_math/wheel.py`) refuses an IDENTICAL CUDA binary with PTX outside the
JIT-invariant rule, without fatbins, or with fatbins of another architecture.

Why it works unchanged: the Mojo runtime (libAsyncRTMojoBindings, MAX 26.5.0) hands the
embedded bytes to `cuModuleLoadDataEx`, which takes PTX, cubin or fatbin by content. The
bundled nvPTXCompiler is used only in compile-only mode, and the wheel does not ship it, so
today the user's DRIVER compiles our IDENTICAL PTX.

## The driver floor, before and after

| wheel | default (no env) | with `MODULAR_NVPTX_COMPILER_PATH` (MAX's documented older-driver escape) |
|---|---|---|
| 0.8.19 (PTX: sm_89 ISA 8.1, sm_90a ISA 8.5) | >= 580: the MAX runtime refuses below it ("Required: driver version >= 580 (CUDA >= 13.0)") | PTX floor: sm_89 >= 530 (CUDA 12.1), sm_90a >= 555 (CUDA 12.5); ran on 570 |
| fatbins from CUDA 13.0 (zstd) | >= 580 | >= 580: `CUDA_ERROR_INVALID_IMAGE` on 570 (13.x binaries need r580, release notes) |
| **fatbins from CUDA 12.5 (LZ4), shipped** | >= 580 (same runtime check) | 12.x cubins run on any r525+ driver (minor version compatibility); **ran on 570**, same bits |

So the shipped recipe does not raise the floor: 580 by default as today, and the escape hatch
keeps working. With it the user's ptxas is no longer used for IDENTICAL kernels (the fatbins
need no compile); only the env var's presence is needed to pass MAX's version check.
Measured below 570: nothing (RunPod offered no older driver on an sm_89/sm_90 GPU); the
sm_90a 12.5 fatbins were run only on 580 drivers.

## Proof on 0.8.19 (the .rn wheel, sha256 2296577101b1..., converted with the shipped recipe)

Converted wheel sha256 `c5bd934f8f043e538137ec03da1dac0a547a2ea087c4d96ba4dccb0a26fc182d`, built
twice with the same result. Per set: sm_89 1,786 modules: 1,655 fatbin in place, 129 moved, 2 left
PTX; sm_90a 1,786: 1,664 in place, 121 moved, 1 left. The left modules are integer-only check
kernels (1,988 and 2,949 bytes) in `_mojolearn_embedding.so` that no release lane loads.

The 0.8.19 release column from the installed wheel (209 lanes, base/denormal/odd, one fit):

| box (driver) | env | `CUDA_DISABLE_PTX_JIT=1` | JIT cache after the column | vs 0.8.19 Apple + AMD + NVIDIA |
|---|---|---|---|---|
| RTX 4090, sm_89 (580.159.03) | none | exit 0, 148 s | 0 files | IDENTICAL=627, infer/model 984, 0 DIVERGENT |
| H100 80GB HBM3, sm_90a (580.126.09) | none | exit 0, 142 s | 0 files | IDENTICAL=627, infer/model 984, 0 DIVERGENT |
| RTX 2000 Ada, sm_89 (570.195.03) | escape hatch | exit 0, 124 s | 0 files | IDENTICAL=627, infer/model 984, 0 DIVERGENT |

The three columns against each other: IDENTICAL=627, 0 DIVERGENT. On 570 without the env var the
wheel is refused by MAX exactly as the PTX wheel is (`r8-rtx2000-570-driver-matrix.txt`), and
the CUDA 13.0 zstd wheel fails with `CUDA_ERROR_INVALID_IMAGE` even with it.

Negative control (an earlier run, `cuda13-zstd-r2-*`): the PTX wheel under
`CUDA_DISABLE_PTX_JIT=1` fails every kernel load with `CUDA_ERROR_JIT_COMPILATION_DISABLED`; with
the JIT on and a fresh cache it JIT-compiles 794 modules (15.1 MB) per column. The CUDA 13.0 zstd
variant (superseded for the driver floor) passed the same column on a 4090 and an H100
(`cuda13-zstd-*`).

## Sizes

| | 0.8.19 (.rn PTX) | CUDA 12.5 LZ4 (shipped) | CUDA 13.0 zstd (not shipped) |
|---|---|---|---|
| wheel | 100,638,553 B | 96,120,465 B | 95,493,509 B |
| IDENTICAL sm_89 binaries, zipped | 20,080,068 B | 17,525,169 B | 17,252,148 B |
| IDENTICAL sm_90a binaries, zipped | 20,081,224 B | 18,118,042 B | 17,764,103 B |
| device code per set, unzipped | 68.2 MB PTX | 14.0 / 14.3 MB fatbin | 9.1 / 9.7 MB fatbin |

Installed size is unchanged: the rewrite is in place, NUL-padded. A bare cubin would not fit in
place (sm_89: 337 of 1,108 distinct modules), hence a compressed fatbin.

Full logs, columns and both converted wheels: `~/mojolearn-evidence/nvidia-cubin-2026-09-26/`
(r5 and r8: driver 570; r7 and r9: driver 580).
