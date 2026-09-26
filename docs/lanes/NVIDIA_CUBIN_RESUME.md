# NVIDIA cubin lane: resume file

Branch `lane/nvidia-cubin` (never merged by the lane itself). Evidence lives
outside the repository in `~/mojolearn-evidence/nvidia-cubin-2026-09-26/`.

## Goal

Ship precompiled NVIDIA machine code (cubin, `ptxas --fmad=false` from the
already `.rn`-rewritten IDENTICAL PTX) for sm_89 and sm_90a inside the Linux
CUDA sets, so (1) the user's driver never JIT-compiles our IDENTICAL kernels
and (2) the NVIDIA payload shrinks.

## Spend so far

r1 (4090, pod eq7wr46p6rc9l2, ~4 min, verified gone, box script bug) ~$0.05;
r2 (4090, pod b11cqjkymdxu0l, $0.74/h, 59 min, verified gone) ~$0.73.
r3 (4090, pod nd4pf5m1yj11mk, ~11 min, verified gone) ~$0.14.
r4 (H100 80GB HBM3, pod 0wu9vfceofo6fh, $3.49/h, ~15 min, verified gone) ~$0.90.
Total ~$1.82 of the $25 cap. RunPod NVIDIA only. After r4 the RunPod listing
shows no mojolearn-cubin pod and no dead-man of this lane is running.

## PROVEN on r4 (sm_90a, H100 80GB HBM3, driver 580.126.09)

Same r3 wheel. `CUDA_DISABLE_PTX_JIT=1` column exit 0 (141 s); JIT allowed:
0 JIT-cache files; both vs 0.8.19 Apple + AMD + NVIDIA: IDENTICAL=627,
infer/model 984, 0 DIVERGENT; 4090 vs H100: IDENTICAL=627, 0 DIVERGENT.
Summary files: `bench/results/nvidia_fatbin_2026-09-26/` (README there).

## PROVEN on r3 (sm_89, RTX 4090, driver 580.x): NO JIT AT ALL, same bits

Evidence `~/mojolearn-evidence/nvidia-cubin-2026-09-26/r3-4090/`. Wheel built
on the box by `scripts/box/mkcubin.py` with the lea-moving cubin_contract.py
(commit 8b4b7146a): sha256 47b1ad452eac593c50c33bfe8a718ffcf8a5414821df1e65fe23f73f9de43f82
(a copy is in that directory), REPRODUCIBLE (built twice, same sha256).
sm_89: 1,745 fatbins in place + 40 moved + 1 left PTX; sm_90a: 1,713 + 72 + 1.

- `CUDA_DISABLE_PTX_JIT=1`, full 0.8.19 release column (209 lanes,
  base/denormal/odd): exit 0, 145 s. The PTX wheel under the same variable
  fails every kernel load (r2 negative control).
- JIT allowed, fresh private `CUDA_CACHE_PATH`: 0 files in the JIT cache
  after the whole column (the PTX wheel: 794 modules, 15.1 MB).
- Both columns vs the recorded 0.8.19 Apple + AMD + NVIDIA columns:
  IDENTICAL=627, infer/model IDENTICAL=984, 0 DIVERGENT
  (diff-ref-cubin_nojit.txt, diff-ref-cubin.txt).
- The one left PTX module per set is `embedding_checks_embedding_ide*`
  (2,949 bytes, integer only) in `_mojolearn_embedding.so`: its fatbin
  (5.7 KB; ptxas unrolls it to 15.8 KB SASS) fits no free region of that
  binary (largest PTX span 3.9 KB). It is JIT-invariant and no release lane
  loads it (the JIT-disabled column passed). The audit admits exactly this
  class (<= 4,096 bytes, no approx op, every float op rounding-pinned) and
  reports it as `jit_invariant_ptx`.

## PROVEN on r2 (sm_89, RTX 4090): the runtime loads the fatbins unchanged, same bits

Evidence `~/mojolearn-evidence/nvidia-cubin-2026-09-26/r2-4090/` (all.log,
col_*.json/log, mkcubin.json, diff-*.txt). Wheel: the .rn 0.8.19 wheel
(sha256 2296577101b1...) with both IDENTICAL sets converted on the box by
`box/mkcubin.py` (scratchpad copy in the evidence dir's parent notes).

- NEGATIVE CONTROL: the PTX (.rn) wheel with `CUDA_DISABLE_PTX_JIT=1` fails
  every kernel load: `CUDA_ERROR_JIT_COMPILATION_DISABLED` (col_neg_rn.log).
  So the driver JIT really is what compiles the shipped PTX today.
- The fatbin wheel, full 0.8.19 release column (209 lanes, base/denormal/odd,
  one fit): exit 0 in 145 s; diffed against the recorded 0.8.19 Apple, AMD and
  NVIDIA columns: IDENTICAL=627, infer/model IDENTICAL=984, 0 DIVERGENT
  (diff-ref-cubin.txt), the same counts as the .rn proof of 2026-09-25.
- Same box A/B, fresh private CUDA_CACHE_PATH each: the PTX wheel JIT-compiled
  794 modules (15.1 MB of JIT cache) in a 163 s column; the fatbin wheel 20
  modules (150 KB), 145 s. rn vs fatbin columns: IDENTICAL=627, 0 DIVERGENT.
- NOT YET "NO JIT AT ALL": the fatbin wheel under `CUDA_DISABLE_PTX_JIT=1`
  refused every cell (col_cubin_nojit.log): the 20 JIT'd modules are the
  leftover PTX kernels (fatbin bigger than the PTX text), and at least one of
  them (e.g. `core_device_liveness_write_can*`, `core_multi_gpu_copy_scalar*`)
  runs in every fit. Their arithmetic is JIT-invariant (integer / .rn only),
  so the bits cannot move, but the JIT still happens for them.

## Measured on r2 (RTX 4090, driver 580.126.20, ptxas/fatbinary 13.0.88 from pip nvidia-cuda-nvcc==13.0.*)

- A BARE CUBIN DOES NOT FIT in place: sm_89, 1,108 distinct modules: cubins
  19.1 MB vs PTX 25.0 MB, only 337 fit (worst 6.6x, the ELF skeleton + long
  Mojo symbol names). Over the set's 1,786 embedded modules 677 fit.
- A FATBIN WITH THE CUBIN ZSTD-COMPRESSED (`fatbinary --compress-all
  --compress-mode=size`) totals 4.2 MB and fits 1,087 of 1,108. The 21 that
  do not are 765..2,949-byte kernels (fills, scans, copies, a few with only
  add.rn / div.rn / rcp.rn / abs / max / setp): JIT-invariant by construction.
  Over the embedded modules: sm_89 1,745 converted + 41 left PTX; sm_90a
  1,713 + 73; audit 0 errors.
- ELF/fatbin headers (ptxas 13.0 = CUDA ELF ABI 8, OSABI 0x41): cubin
  e_flags carry the SM in bits 8..15 and do NOT distinguish sm_90 from sm_90a;
  the fatbin entry does (u32 at +28 = SM, flag 0x100000 at +40 = 'a').
- Sizes (wheel, zip): whole wheel 100,638,553 -> 95,366,527 bytes; IDENTICAL
  sm_89 compressed 20,080,068 -> 17,206,280, sm_90a 20,081,224 -> 17,683,000.
  The PTX already zipped well; most of the compressed identical tier is the
  host code. Installed (uncompressed) size is unchanged (in place, NUL-padded).

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

Both arches proven. What is left (none needs a GPU to start):
1. Review + merge lane/nvidia-cubin (Andrew; never merged by the lane).
2. The first real release build after merge runs the new build_sets.sh block on
   the NVIDIA build boxes (needs `python3 -m pip` and `objdump` there; both were
   on the RunPod pytorch image). cubin_contract.py is a LINUX_BUILDERS /
   LINUX_SET_BUILDERS input, so every CUDA set rebuilds (no reuse). Watch
   `cubin.jsonl` for `unplaced` rows with jit_invariant false (the build refuses).
3. Optional: the FAST and deterministic CUDA tiers still ship PTX (driver JIT);
   the same pass with fmad left on would remove their JIT too (not done: out of
   the IDENTICAL scope).
4. Optional: the one JIT-invariant leftover per set could be removed by shrinking
   that check kernel's unrolling at the source.

## Commands / scripts

All scripts: `~/mojolearn-evidence/nvidia-cubin-2026-09-26/scripts/`.
- `rent.sh <outdir> [gpu...]` rents one RunPod pod (dead-man before create,
  on-pod watchdog via tools/runpod_guard.sh, uploads box/ and a wheel:
  `WHEEL=<whl> WHEEL_DIR=cubin` for a converted wheel), then HOLDS until
  `<outdir>/STOP` exists, then deletes and verifies gone.
- `box/setup.sh`: pinned ptxas 13.0.88, converts the .rn wheel twice
  (`box/mkcubin.py`), installs a venv. `box/all.sh`: JIT-disabled column +
  JIT-cache count via `box/col.sh`. Launch on the box with
  `setsid bash /root/cb/box/all.sh > /root/cb/all.log 2>&1 < /dev/null & disown`
  (a plain nohup over ssh did not survive).
- Diff on the Mac: `python3 tools/identity_break.py --diff
  ~/mojolearn-evidence/release-check/69a519c1522d/metal/column.json
  ~/mojolearn-evidence/release/0.8.19/69a519c1522d/column-amd/column-hip.json
  ~/mojolearn-evidence/release/0.8.19/69a519c1522d/smoke-linux/column-cuda.json <col.json>`.
- Tests: `packaging/linux/test_cubin_contract.py` (in tools/test_wheel_audit.sh).
