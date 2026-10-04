# A portable AMD path (decision and prototype)

Status: PROTOTYPE, 2026-10-04, branch `lane/amd-portable`. Nothing here is
IDENTICAL-qualified, and no wheel carries the payload yet.

NVIDIA ships native sm_89/sm_90a code plus an sm_80 PTX build that the driver
compiles on GPUs we carry no native code for (docs/NVIDIA_PTX_IDENTITY.md).
AMD ships gfx942 code objects only, so any other AMD GPU is refused. This
note compares two ways to get a portable AMD form, says what each needs from
our toolchain (Mojo 1.0.0, ed45d567), and records what was measured.

## Decision

**Route A: LLVM bitcode carried in the wheel, compiled on the user's machine
by AMD COMGR.** COMGR (`libamd_comgr`) ships with every ROCm runtime, because
HIP itself depends on it. The compiled code objects are then patched into
copies of our native bindings, and the stock Mojo runtime loads those copies.
Route B (AMDGCN SPIR-V) also ran on the probe kernels. It is not chosen,
because it needs everything A needs plus an IR rewrite (address spaces,
calling convention, triple) that text edits cannot do safely on real kernels.

Both routes depend on one thing Mojo 1.0.0 does not offer: a build mode that
keeps every kernel's IR. The prototype gets the IR from a second, never-shipped
"extraction" build (below). A Modular-side option would remove that step (see
"What would unblock it").

The portable form covers **wave64 CDNA only (gfx90a, gfx942, gfx950)**. Mojo
fixes `WARP_SIZE`, the MFMA choices and other target branches when it emits the
IR. So IR emitted for gfx942 cannot serve wave32 RDNA parts (gfx10/11/12), and
the loader refuses them. RDNA would need its own emission, and its warp
reductions fold in a different order (measured below), so it is a separate
IDENTICAL question.

## Evidence

The tiny probes ran on the MI325X box (`ssh root@162.243.193.137`, gfx942,
ROCm 6.4.0, COMGR 3.0, ROCm LLVM 19). Their output is under
`/root/lq/amd-portable/` (harvested to
`~/mojolearn-evidence/identical-all/amd/lq/amd-portable/`). The summaries are
committed in `bench/results/amd-portable-20261004/`, and the sources in
`tools/amd_portable/`. The four kernels in `pkernels.mojo` are:

- `k_muladd`: a plain `a*b+c`. Mojo marks both operations `contract`.
- `k_pinned`: the IDENTICAL spelling, with the product pinned by inline asm
  `v_mul_f32` and an explicit `llvm.fma`.
- `k_math`: `sqrt(a)/b + exp(c)`, whose lowering is up to the compiler.
- `k_dot`: a fixed serial fold followed by `warp.sum`.

Each kernel ran on 4096 integer-generated inputs, and every output word was
compared.

| Probe | Result |
| --- | --- |
| Mojo emits AMDGPU LLVM IR per kernel (`compile_info`, kinds `llvm`, `llvm-opt`) for gfx942/gfx90a/gfx1100/gfx1201 | yes. Same datalayout and triple on every target; gfx942 and gfx90a IR differ only in `target-cpu` |
| IR across wave sizes | `k_dot` IR for gfx1100/gfx1201 differs from gfx942 (wave32 shuffle tree: 4 vs 6 adds in the ISA) |
| ROCm 6.4 `llc` (LLVM 19) reads Mojo IR as emitted | **no**: `captures(none)`, `nocreateundeforpoison`, `icmp samesign`, `getelementptr inbounds nuw` and `f0x` float literals are LLVM-24 syntax |
| The same IR after `downgrade_ll.py` (syntax-only respellings) | parses; 5 rules covered all 4 kernels |
| IR -> ROCm `llc` + `ld.lld` -> `hipModuleLoadData` on gfx942 | **8/8 MATCH** native Mojo bits (`route_a_summary.txt`) |
| IR -> **COMGR only** (bitcode, and textual IR) -> gfx942 | **8/8 MATCH** (`route_a_comgr_summary.txt`); also builds gfx90a/gfx1100/gfx1201 |
| Float-op ISA histograms, ROCm-built vs Mojo-native, gfx942/gfx90a | identical on all 4 kernels (`isa_hist.txt`). With gfx1100, `k_dot` differs (wave size) and `k_math` has one extra `v_mul_f32` |
| IR -> COMGR for the generic target **gfx9-4-generic** (gfx940/941/942/950), loaded on gfx942 | **4/4 MATCH** (`route_c_generic_summary.txt`); gfx9-generic/gfx10-3/gfx11/gfx12-generic also build |
| Mojo IR -> `spirv64-amd-amdhsa` IR -> `amd-llvm-spirv` -> HIP bundle -> runtime JIT | **4/4 MATCH** for these kernels (`route_b_spirv_summary.txt`); clang's own amdgcnspirv HIP kernel also JITs and matches |
| A Mojo executable whose embedded gfx942 objects are replaced in place by COMGR objects (gfx942 and gfx9-4-generic) | runs on the stock Mojo runtime, **MATCH** (`patch_and_jit_summary.txt`). Stripping `.symtab` crashes the Mojo runtime; dropping only `.comment` is safe |
| `enqueue_function[k, dump_llvm=True]` in a build | embeds the kernel's optimized IR as text even when the launch never runs. It equals `compile_info(..., "llvm-opt")` and can be extracted without running |
| End to end: inject -> extraction build -> `amd_portable_payload.py extract` + `audit` -> runtime `amd_portable.materialize` (COMGR via ctypes) -> patched executable | **MATCH** on gfx942. Materializes for gfx90a and gfx950 (compile only); gfx1100 refused as outside the family |

Cross-target claims are compile-only. We have no gfx90a, gfx950 or RDNA
hardware, so the runtime proof is gfx942 loading the portable form.

## The two routes against our toolchain

**What Mojo can emit.** Per kernel, it emits AMDGPU LLVM IR (unoptimized and
optimized) and ISA, through `std.compile.compile_info`. It also emits IR
through the `dump_llvm` launch parameter. It has no SPIR-V target. Its
supported AMD list is fixed in `std/gpu/host/info.mojo`: gfx90a, gfx942,
gfx950, gfx1030-gfx1201. The LLVM generic targets (`gfx9-generic`,
`gfx11-generic`, ...) are refused by that list, not by LLVM. `mojo build --emit
asm` writes `.amdgcn` assembly sidecars for AMD, never IR. No build flag
embeds IR or bitcode in place of code objects.

**How target-specific the IR is.** The datalayout and triple are shared by
every amdgcn target. Functions carry `target-cpu`. Target branches are
resolved at emit time:

- `WARP_SIZE` and the warp primitives (wave64 `mbcnt.hi`/DPP trees vs wave32);
- MFMA intrinsics (`fused_attention.mojo` calls
  `llvm.amdgcn.mfma.f32.16x16x1f32`, CDNA only);
- inline asm (`v_mul_f32`/`v_mul_f64` pins; valid mnemonics on gfx9-gfx12);
- `llvm.amdgcn.class`, `s.setreg` denormal mode.

So gfx942 IR is portable within wave64 CDNA, not across wave sizes.

**Bits (IDENTICAL).** Within one arm, a runtime-compiled kernel is bit-equal to
native only if the backend makes the same choices: contraction, the lowering
of `llvm.sqrt`/`fdiv`, and denormal mode. Mojo marks every plain float op
`contract`. IDENTICAL code pins products (`pinned_mul_f32`) and fmas
explicitly, so the fusable pairs that remain belong to FAST, which makes no
cross-device claim. The payload manifest inventories fast-math flags per tier,
as the PTX audit does. The user's COMGR is a different LLVM (19 on ROCm 6.4)
from Mojo's (24). On the probe kernels it produced the same float ISA and the
same bits, but that does not prove it for the full library. Like PTX under the
driver JIT, IDENTICAL therefore needs an admission bound to (device gfx,
COMGR/ROCm version, payload manifest).

**What the runtime accepts.** On ROCm 6.4.0:

- `hipModuleLoadData` accepts COMGR-built code objects for the exact target and
  for gfx9-4-generic.
- It accepts an offload bundle that holds amdgcnspirv (JIT through COMGR's
  `TRANSLATE_SPIRV_TO_BC`).
- COMGR's `CODEGEN_BC_TO_RELOCATABLE` reads both bitcode and textual IR.
- Bitcode is read forward only. The payload must therefore be written by the
  OLDEST LLVM we support (ROCm 6.4 = LLVM 19). Mojo's LLVM 24 output reaches
  that LLVM only through the syntax downgrade, which `opt` checks module by
  module.

**Route B costs beyond A.**

- Mojo's IR must be retargeted to `spirv64-amd-amdhsa`. That means
  renumbering address spaces (AMDGPU generic 0 / private 5 / constant 4 become
  SPIR 4 / 0 / 2), changing `amdgpu_kernel` to `spir_kernel`, and moving
  functions to addrspace(4). The probe did this with text edits that cover only
  address spaces 0 and 1. Real kernels need an LLVM pass.
- Buffer fat pointers (p7/p8/p9 in Mojo's datalayout) have no SPIR-V form.
- AMDGCN intrinsics and inline asm pass only through
  `--spirv-allow-unknown-intrinsics` and `+all` extensions.
- Clang documents amdgcnspirv as under active development.
- The SPIR-V would still have to become a code object before Mojo's runtime
  loads it, so B adds steps and removes none.
- B's one advantage is a stable interchange format. A gets most of that by
  writing bitcode with the oldest supported LLVM.

## The prototype (mirrors the PTX design)

- **Build:** `packaging/linux/amd_portable_payload.py`.
  - `inject` adds `dump_llvm=True` to each of the 3444 product
    `.enqueue_function[...]` sites, in an exported copy only; it refuses a git
    checkout.
  - The extraction build of that copy is never shipped or run.
  - `extract` pairs every kernel embedded in each native binding (by its `.kd`
    symbol) with that kernel's IR. It downgrades the syntax, assembles bitcode
    with the pinned old LLVM's `opt`, and writes `hip_portable/cdna-w64/` plus
    `AMD_PORTABLE.json`.
  - The manifest (schema `mojolearn.amd-portable.v1`) records the format, the
    emitted-for target, the wave size, the family targets, the bitcode LLVM,
    the downgrade rule counts, the fast-math inventory per tier, the source
    commit and dirty flag, the per-binding kernel table and all file hashes.
    `identical_qualified` is always `false`.
  - `audit` re-checks every hash and every kernel table.
  - Any kernel without IR, any object holding more than one kernel, or any
    module the old LLVM rejects refuses the build.
- **Runtime:** `python/mojolearn/amd_portable.py`.
  - Verifies the manifest and that the payload was built from the clean
    installed source.
  - Maps the device to its family target, or refuses (RDNA, gfx908, unknown).
  - Compiles each kernel through COMGR (ctypes).
  - Drops `.comment` and patches the object in place into a copy of the
    binding.
  - Caches the copies under `~/.cache/mojolearn/amd-portable/<manifest>/<gfx>/comgr-<ver>/`.
- **Loader:** in `_backend._vendor_base`, a hip `_NoCompatibleNative` goes to
  `_amd_portable_base`. That is the same single case the PTX fallback takes,
  never anything else.
  - FAST and DETERMINISTIC take the route automatically.
  - IDENTICAL refuses, at import and per call (`load_set`).
  - Every refusal is a `GpuPluginError`, and no CPU set is ever substituted.
  - The receipt is `amd_portable_selection_receipt()`.

## What is missing

1. **Moving objects that grow.** An object larger than its old span is refused
   today. Production needs a move with `lea` repointing, which
   `packaging/linux/cubin_contract.py` already does for CUDA fatbins.
2. **Release binding.** A vendor marker hash for the payload (the
   `bundled_ptx` analogue in `gpu_plugins.py`), packing in `pack_wheel.py`, and
   a `build_sets.sh` mode that runs the inject/extraction build and `extract`.
3. **IDENTICAL admission for AMD.** Mirror `ptx_admission.py` with a key of
   (gfx, COMGR version, manifest, reference hashes), plus a `verify
   --qualify-gpu` path on hip. Until it exists, IDENTICAL refuses.
4. **A full-library extraction and run.** Only the probe kernels have been
   carried. The downgrade rule list is checked by `opt` but is not known to be
   complete. Whether every product kernel's recompiled object fits, or needs
   the move, is unmeasured. The verifier through the portable route on gfx942
   is owed.
5. **Hardware beyond gfx942.** gfx90a and gfx950 are compile-only.
6. **RDNA.** Out of scope for the gfx942 payload. It needs a wave32 emission
   and its own identity work.

## What would unblock it (Modular asks)

- A build option that embeds bitcode (or keeps `.ll` sidecars for AMD, as
  `--emit asm` already does for Metal) beside the code objects. That would
  remove the extraction build and `inject`.
- `--target-accelerator gfx9-4-generic` / `gfx11-generic`. LLVM supports
  these, and COV6 (which Mojo already emits) loads them. A gfx9-4-generic build
  would cover gfx942 and gfx950 ahead of time with no runtime compiler.
- A published LLVM revision for Mojo's backend, so that bitcode compatibility
  can be stated instead of tested.

## Fallback alternative: ship more native targets

Mojo compiles natively for gfx90a, gfx950 and gfx1100+. Shipping those sets
is the simplest coverage for named parts. It costs one full build per target
in compile time and wheel size, and gives no help for an unlisted GPU. It is
not a portable path: each target needs its own identity verification on its
own hardware. A wave32 RDNA set also folds warp reductions in a different
order from gfx942.
