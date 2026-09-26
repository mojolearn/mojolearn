# IDENTICAL CUDA sets carry machine code: no driver JIT (2026-09-26)

## The change

`packaging/linux/build_sets.sh`, after the `.rn` pass, installs a pinned ptxas and
fatbinary (pip `nvidia-cuda-nvcc==13.0.88`, `--no-deps`) and runs
`packaging/linux/cubin_contract.py patch`: every IDENTICAL PTX module of the set is compiled
with `ptxas -arch=<set arch> --fmad=false -O3`, wrapped in a zstd-compressed fatbin
(`fatbinary --compress-all --compress-mode=size`) and written over the PTX string in place.
A tiny module whose fatbin is longer than its PTX gets its fatbin in the padding a converted
module left, and the RIP-relative `lea` instructions that reach the string are repointed
(objdump cross-checks each; the old PTX is zeroed). The wheel audit
(`packaging/portable_math/wheel.py`) refuses an IDENTICAL CUDA binary with PTX outside the
JIT-invariant rule, without fatbins, or with fatbins of another architecture.

Why it works unchanged: the Mojo runtime (libAsyncRTMojoBindings, MAX 26.5.0) hands the
embedded bytes to `cuModuleLoadDataEx`, which takes PTX, cubin or fatbin by content. The
bundled nvPTXCompiler is used only in compile-only mode, and the wheel does not ship it, so
today the user's DRIVER compiles our IDENTICAL PTX.

## Proof on 0.8.19 (the .rn wheel, sha256 2296577101b1..., converted on the box)

Converted wheel sha256 `47b1ad452eac593c50c33bfe8a718ffcf8a5414821df1e65fe23f73f9de43f82`,
built twice with the same result. Per set (`mkcubin-report.json`):

| set | modules | fatbin in place | moved | left PTX (JIT-invariant) |
|---|---|---|---|---|
| sm_89 | 1,786 | 1,745 | 40 | 1 |
| sm_90a | 1,786 | 1,713 | 72 | 1 |

The left module is the 2,949-byte integer-only `embedding_checks_embedding_ide*` kernel in
`_mojolearn_embedding.so`; its fatbin (5.7 KB) fits no free region of that binary.

The 0.8.19 release column from the installed wheel (209 lanes, base/denormal/odd, one fit):

| box | `CUDA_DISABLE_PTX_JIT=1` | JIT cache after the column | vs 0.8.19 Apple + AMD + NVIDIA |
|---|---|---|---|
| RTX 4090 (sm_89), driver 580 | exit 0, 145 s | 0 files | IDENTICAL=627, infer/model 984, 0 DIVERGENT |
| H100 80GB HBM3 (sm_90a), driver 580.126.09 | exit 0, 141 s | 0 files | IDENTICAL=627, infer/model 984, 0 DIVERGENT |

RTX 4090 against H100: IDENTICAL=627, 0 DIVERGENT. Negative control: the PTX wheel under
`CUDA_DISABLE_PTX_JIT=1` fails every kernel load with `CUDA_ERROR_JIT_COMPILATION_DISABLED`;
with the JIT on and a fresh cache it JIT-compiles 794 modules (15.1 MB) in a 163 s column
(`r2-4090-jit-ab.txt`, an earlier fatbin build that still JIT'd 20 leftovers).

## Sizes

| | before (.rn PTX) | after (fatbin) |
|---|---|---|
| wheel | 100,638,553 B | 95,493,509 B |
| IDENTICAL sm_89 binaries, zipped | 20,080,068 B | 17,252,148 B |
| IDENTICAL sm_90a binaries, zipped | 20,081,224 B | 17,764,103 B |
| embedded device code per set, unzipped | 68.2 MB PTX | 9.1 MB (sm_89) / 9.7 MB (sm_90a) fatbin |

Installed size is unchanged: the rewrite is in place, NUL-padded. The zipped saving is
modest because PTX text compresses well and most of a zipped IDENTICAL binary is host code.
A bare cubin would not have fit in place (sm_89: 337 of 1,108 distinct modules; ELF skeleton
and the long Mojo symbol names), hence the compressed fatbin.

Full logs, columns and the wheel: `~/mojolearn-evidence/nvidia-cubin-2026-09-26/`.
