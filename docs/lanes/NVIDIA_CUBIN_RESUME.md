# NVIDIA cubin lane: resume file

Branch `lane/nvidia-cubin` (never merged by the lane itself). Evidence lives
outside the repository in `~/mojolearn-evidence/nvidia-cubin-2026-09-26/`.

## Goal

Ship precompiled NVIDIA machine code (cubin, `ptxas --fmad=false` from the
already `.rn`-rewritten IDENTICAL PTX) for sm_89 and sm_90a inside the Linux
CUDA sets, so (1) the user's driver never JIT-compiles our IDENTICAL kernels
and (2) the NVIDIA payload shrinks.

## Spend so far

$0.00 (cap $25, RunPod NVIDIA only).

## Step 1, feasibility: what is proven (static reading, no GPU yet)

How a Mojo kernel reaches the GPU (Mojo 1.0.0 / MAX 26.5.0, the pinned
toolchain; source read from modular/modular main
`max/mojo/max/gpu/host/device_context.mojo`, runtime read from the Linux
`max-core-26.5.0-release.conda` `lib/libAsyncRTMojoBindings.so`, which is the
same build the wheel bundles at `mojolearn/.libs/`):

1. `DeviceFunction` compiles the kernel with
   `_emission_kind = "asm" if (_cross_compilation() and nvidia) else "object"`.
   Our release builds pass `--target-accelerator` (MOJOLEARN_GPU_ARCHS), which
   is cross compilation, so the binary embeds PTX text ("asm"). A native build
   with no `--target-accelerator` would emit "object" (a cubin produced at
   build time); that path cannot pin `--fmad=false` (no option reaches ptxas
   except per-call `compile_options`), and our builds need the explicit arch.
2. Either way the bytes go to ONE entry point:
   `AsyncRT_DeviceContext_loadFunction(result, ctx, moduleName, functionName,
   data, dataLen, maxDynamicSharedBytes, debugLevel, optimizationLevel)`.
   Since the same entry point receives cubin in the "object" case, the
   runtime must accept a cubin there.
3. In `libAsyncRTMojoBindings.so` the CUDA device loads modules with
   `cuModuleLoadDataEx` (dlsym'd from libcuda). That driver call detects
   PTX, cubin (ELF) or fatbin by content. There is no ELF-magic or `.version`
   check in the library (searched the disassembly for 0x464c457f and the
   rodata for ".version"/".target": none), consistent with passing the bytes
   through.
4. The bundled nvPTXCompiler path (`libNVPTX.so`, `MODULAR_NVPTX_*`,
   "CompilationDevice ... virtual mode / compile-only mode") is for
   compile-only (no GPU) mode. The wheel does NOT ship `libNVPTX.so`, so on a
   user's box PTX goes to the driver JIT, which is the driver-version
   dependence this lane removes.
5. The embedded PTX is a NUL-terminated string in `.rodata`; its length is a
   compile-time constant in the caller. `cuModuleLoadDataEx` ignores length
   (ELF is self-sized), so a cubin written over the PTX, NUL-padded to the
   same length, is a valid image IF cubin <= PTX in bytes. That is the
   least invasive route: no relink, same as `ptx_contract.py`'s in-place
   rewrite.

Open (needs the rented box): does the unchanged runtime load the patched .so
and produce the same bits; is cubin <= PTX for every module; what does
compression do to the wheel size.

## Exact next step

Rent one RTX 4090 (sm_89) on RunPod with the repo's dead-man + on-pod
watchdog; on it: install the 0.8.19 wheel, `.rn`-patch its sm_89 IDENTICAL
set (`packaging/linux/ptx_contract.py patch`), convert every embedded PTX
module to `ptxas -arch=sm_89 --fmad=false` cubin in place, run a lane, compare.

## Commands / scripts

(filled in as they are written)
