# Ask for Modular: let the MAX runtime reuse the HIP/HSA runtime a process already has

2026-10-26, lane amd-torch-interop-0859. Status: open; mojolearn works around the import order in
`python/mojolearn/_rocm_runtime.py` and documents the setup (docs-site/user_guide/framework_adapters.md).

## What happens

On AMD the MAX runtime opens the HIP and HSA runtimes with `dlopen` by bare name (`libamdhip64.so`,
`libhsa-runtime64.so`) at the first device use. PyTorch's ROCm wheels (torch 2.13.0+rocm7.2,
torchvision 0.28.0+rocm7.2) bundle their own `libamdhip64.so` and `libhsa-runtime64.so` (soname
`libhsa-runtime64.so.1`) in `torch/lib` and map them at `import torch`.

glibc reuses a loaded object for a bare-name `dlopen` only when the name equals one the object was
loaded under, its DT_SONAME, or a file on the search path (LD_LIBRARY_PATH, the caller's RUNPATH, the
ld.so cache) with the same inode. With torch imported first and `/opt/rocm-7.2.3/lib` on the path,
none of those hold, so MAX maps a second HIP and HSA runtime:

- `torch.onnx.export` -> `mojolearn.import_model` -> `Program.run` on device:
  `HIP architecture query failed: Failed to initialize HSA runtime`
  (max/mojo/max/gpu/host/device_context.mojo:3825).
- With two `libamdhip64` copies mapped, the next HIP fat-binary registration (torchvision `_C.so`)
  aborts: `hip.cpp:512 __hipPopCallConfiguration has non-null function pointer 1 despite this being
  the first instance of the library being copies`, stack `__hip_module_ctor` ->
  `__hipRegisterFatBinary` -> `rocprofiler_register_library_api_table`
  (~/mojolearn-evidence/amd-torch-interop/real-model-crash.log). Part of that case was ours: our device
  probe dlopened every `libamdhip64` candidate; fixed on this lane.

Setting `LD_LIBRARY_PATH=<torch/lib>:...` before Python starts makes MAX's bare-name `dlopen` find
torch's files (same inode) and the process keeps one runtime. glibc reads LD_LIBRARY_PATH once at
process start, so a library cannot arrange this after `import torch`.

## The ask

One supported way to point the MAX runtime at a HIP/HSA runtime the process already has, so
`import torch; import max` (or any library built on the MAX runtime) works in either order:

1. Before its own `dlopen`, reuse an already-mapped `libamdhip64` / `libhsa-runtime64`
   (`dlopen(..., RTLD_NOLOAD)` on the sonames `libamdhip64.so.7`, `libamdhip64.so.6`,
   `libhsa-runtime64.so.1`, or `dl_iterate_phdr` for any mapped `libamdhip64*`); or
2. An environment variable naming the HIP and HSA library files to open (for example
   `MODULAR_HIP_LIBRARY_PATH` / `MODULAR_HSA_LIBRARY_PATH`), read at the first device use, documented
   on https://max.modular.com/environment-variables; and
3. Open the versioned sonames (`libamdhip64.so.7`, then `.so.6`) rather than the unversioned
   development symlink, which a runtime-only ROCm install does not carry and which glibc never matches
   against a loaded object's soname.

## Until then (what mojolearn does)

- `import mojolearn` before `import torch`: mojolearn loads, by MAX's bare names, the copies MAX would
  load; torch's libraries then resolve to them by name or soname. One runtime.
- `import torch` first: mojolearn checks with `RTLD_NOLOAD` whether MAX's bare-name `dlopen` reaches
  the loaded copy. If not, `import mojolearn` raises with the loaded path and the two working setups
  (import mojolearn first, or `LD_LIBRARY_PATH` led by `torch/lib`).
- The device probe looks for HIP libraries without loading them.

Box evidence for the mechanism: `tools/amd_torch_interop_diag.sh <venv>` (readelf NEEDED/SONAME/RUNPATH,
the names MAX's libraries carry, LD_DEBUG=libs and the mapped copies for both import orders and both
library paths).
