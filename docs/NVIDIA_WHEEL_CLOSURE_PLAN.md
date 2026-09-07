# NVIDIA wheel closure plan

Source/artifact inspection only, September 7, 2026. No builds, tests, model
execution, measurements, rentals or publication were performed for this plan.
All future execution belongs to root/main, serially on remote Linux. No Apple
execution and no subagent tests or measurements.

## Why 0.6.0 has Mac and AMD binaries but no NVIDIA binary

This is an artifact-delivery gap, not evidence that NVIDIA cannot execute
MojoLearn. The published alpha route overlaid Python modules on existing Mac
and AMD wheels and preserved their native/runtime bytes. It did not build or
add a CUDA set. The historical complete NVIDIA candidate had failed installed
checks; later targeted native fixes were never assembled and qualified as one
final combined Linux wheel.

The retained publication record is
`bench/results/releases/2026-09-06-alpha-api/README.md`, with actual index
readback in `pypi-0.6.0-verification.json` in that directory. This audit reads
that retained evidence; it does not perform a new index query.

Published candidates are in that directory's `release-0.6.0-retry2/`:

| Artifact | SHA256 |
| --- | --- |
| `mojolearn-0.6.0-py3-none-manylinux_2_35_x86_64.whl` | `beeb02d2d41bb64421957bfbcd937860ed00c0aed5618840454e322f11a372b0` |
| `mojolearn-0.6.0-py3-none-macosx_11_0_arm64.whl` | `a44889df2990289658a232cab6f3b734268e4e274785b84039024025920847a7` |

The Linux wheel's actual `mojolearn-0.6.0.dist-info/ALPHA_PROVENANCE.json`
identifies base wheel SHA
`7c5f9af825cbcbd74a293adc75ad15670a30a179d3a9a8a8cfd993cd9476f7be`,
HIP/gfx942 native directories and unchanged inherited runtime bytes. Its
current numerical qualification is explicitly **not inherited**. The new
`_mojolearn_byte_lm` extension is absent. Python trainer exports do not create
missing native operations. The release's metadata-header failure was repaired
before publication; it is unrelated to the NVIDIA numerical failures.

## Exact evidence boundaries

All paths below are relative to the repository root.

| Evidence | What it establishes; what remains |
| --- | --- |
| `bench/results/resume/2026-09-06-installed-gap-closure/README.md` | Frozen `eb835021dcd79a59a7e8f78c754a75db3c1fea83`: all 45 extensions built on each vendor. AMD's 24 installed jobs passed. CUDA/sm_89 candidate SHA `805970b2bc44a002194cee66cbe378996f7e4a483aa3b6ca18a4f6da06060663` had 19 passes, two Mamba accuracy failures and three Transformer stalls. These failures remain failures. |
| Same directory, `nvidia-installed/sequence-followup-env-fixed/result.json` | Context lifetime fix removes Transformer stalls; IDENTICAL Mamba/Transformer pass. Full-FP32 Mamba projections remove block-output misses, but non-IDENTICAL Mamba3 state and Transformer accuracy failures remained. Six-binding overlay, not a final wheel. |
| Same directory, `nvidia-installed/transformer-fp32-followup/result.json` | Full-FP32 matrix products pass FAST/DETERMINISTIC Transformer checks. IDENTICAL was not rebuilt after the final dispatch changes. Corrections committed at `3963a0fc`; refreshed AMD qualification also needed for changed non-IDENTICAL paths. |
| Same directory, `nvidia-installed/mode-persistence-followup/provenance.json` | Serialization wrapper overlay at `29a8c848` passes ordered mode-persistence checks. Modified installed Python bytes are not the frozen wheel. |
| `bench/results/resume/2026-09-06-root-feature-nvidia/README.md`, `run3/` | Snapshot `4a271ae6d719c79e2e871f4776b74c9a12f380ee`: Mamba3 FAST dt-softplus repair passes the targeted FAST/IDENTICAL fixtures; 26/27 jobs pass, external cuML loader fails. This does not qualify DETERMINISTIC or a new wheel. Subsequent lane results in this README remain scoped to their own sources and binaries. |
| `bench/results/resume/2026-09-06-root-byte-lm-cross-vendor/nv-to-amd-resume128-comparison.json` and `amd-to-nv-resume128-comparison.json` | Both directions are `QUALIFIED_BOUNDED_CAPTURE`: fixed two-block, 34,944-parameter FP32 real-text model, 128 steps, actual foreign step-64 checkpoint bytes, all raw state, effective missing-moments controls and independently guarded first-step FP64 oracles. Held-out loss is 5.5412986278533936 → 2.8436418771743774. Numerical inventory is frozen at `d921eade`; this separate extension is not in the published wheel and does not qualify a later build, arbitrary shapes, Metal or speed. |

Old 24-job qualification snapshots omitted nested Mamba2/3 corpus hashes.
`tools/check_linux_release_qualification.py` now requires those fingerprints;
the old records cannot be relabeled as fresh final-release admission.

## Main-only sequential closure jobs

1. **Freeze one release source and explicit alpha availability contract.**
   Preserve the user's authorization to expose alpha functionality without
   pretending it has a complete numerical certificate. Inventory actual symbols
   needed by public GEMM, UMAP, Mamba/Transformer, loss/optimizers, MLP, PCA and
   byte-LM wrappers. Freeze wrapper/native compatibility, runtime/toolchain and
   qualification sources before any build. Record known unavailable operations
   explicitly. Do not silently substitute the old AMD native set under new
   wrappers and claim current compatibility.

2. **Resolve the separate byte-LM packaging inventory.**
   `python/mojolearn/_backend.py` knows `_mojolearn_byte_lm`, but
   `packaging/linux/build_sets.sh` and `packaging/linux/pack_wheel.py` still
   enumerate 15 standard extensions per mode, excluding it. If the new alpha
   ships the callable trainer, extend build/staging/runtime/RECORD and admission
   inventories with an explicit CUDA/HIP IDENTICAL-only optional extension;
   do not fabricate FAST/DETERMINISTIC support or make a 16-per-mode claim.
   `bindings/build_byte_lm.sh` is its separate build entrypoint. Add installed
   trainer coverage in addition to the existing 24 jobs, including actual
   bindings/symbols for MLP. If omitted, retain the native-unavailable statement.

3. **Preflight and build CUDA on RunPod, then HIP, never concurrently.**
   Root first reconciles the existing provider inventory/lease ownership.
   Use a pinned source fetch and a fresh evidence directory. One compiler at a
   time, at most two compiler workers, at most three CPU cores and bounded
   BLAS/OpenMP threads; retain memory/VRAM guards, shrinking work deadlines,
   fetch reserve, cancellation forwarding and independent deletion watchdogs.
   `tools/linux_surface_qualification.sh build OUT` is the full-set entrypoint;
   `build-tier OUT MODE` produces partial sets only. The September 7 source
   patch narrows affinity to at most two inherited cores, verifies child
   readback and overrides inherited compiler/BLAS/job limits. Its two Linux
   resource-cap tests are authored but not yet executed; root must validate
   this before using the new qualification entrypoint. Also impose a bounded parent
   affinity. The wrapper
   `packaging/linux/leg.sh nvidia` currently defaults to DigitalOcean; an
   approved RunPod campaign must explicitly select
   `MOJOLEARN_LINUX_LEG_NVIDIA_VIA=runpod`, or use root's guarded RunPod
   controller. Do not run the historical wrapper blindly. Record actual
   architecture/vendor readbacks and all output hashes. Fetch and verify each
   leg, then delete and verify absence before the next rental.

4. **Assemble one Linux wheel containing both vendor sets.**
   Use `packaging/linux/pack_wheel.py --set CUDA_SET --set HIP_SET --out OUT`
   on remote Linux, with matching source/version/runtime closure. Exact set
   paths come from the retained build manifests under `sets/cuda` and
   `sets/hip`; do not copy source-tree extensions over installed files.
   Do not publish two same-tag Linux wheels expecting pip to detect GPU vendor.
   The loader selects supported architectures inside the one wheel.

5. **Repair and freeze final bytes before installed checks.**
   Apply the manifest exclusions and audit/repair logic in
   `packaging/linux/audit.sh` on remote Linux, not its historical Mac Docker
   route. Retain actual manylinux determination, CPU ISA checks, dependencies,
   driver-library exclusions, RECORD and runtime readbacks. Preserve original
   failed empty-directory/RECORD artifacts if the issue recurs; normalize
   transparently before computing the final SHA. Any later overlay, repair or
   metadata change creates a different artifact requiring new installed checks.

6. **Install and exercise these exact final bytes on NVIDIA, then AMD.**
   For each vendor, root runs the guarded existing entrypoint:
   `tools/linux_surface_qualification.sh qualify WHEEL SHA256 cuda|hip OUT BUILD_PROVENANCE_JSON`.
   Retain all eight surfaces × three modes, all binding hashes, source/corpus
   fingerprints, UMAP quality raw arrays, OrderedRMSE bytes and failures.
   Use clean environments and prevent repository imports; add bounded installed
   checks for newly advertised trainer/PCA symbols absent from the old matrix.
   Run no speed comparisons as part of this packaging gate. The old successful
   source fixtures guide coverage but cannot replace installed-artifact checks.

7. **Admit the declared scope and stage, without changing historical claims.**
   The full stable dual-vendor gate is
   `tools/check_linux_release_qualification.py WHEEL --qualification-root DIR --source-root SOURCE`;
   it requires both vendor directories for the same final SHA and recomputes
   retained UMAP/Ordered identity. Layout and required evidence are in
   `packaging/linux/RELEASE_QUALIFICATION.md`. The separately authorized alpha
   route in `docs/PYPI_RELEASE.md`, `packaging/alpha_overlay.py` and
   `packaging/verify_alpha_artifacts.py` permits exposure without a full stable
   certificate; its file-only admission must never be reported as numerical
   qualification. Preserve failed/unrun installed cells in the alpha manifest
   and availability notes rather than silently lowering existing tolerances.
   Stage a new version's exact files through the existing alpha Trusted
   Publisher route only when root reaches publication. Do not overwrite 0.6.0
   files or dispatch the normal Apple build workflow. Verify index hashes and
   installed public-file behavior separately, then update NVIDIA availability.

Completion means a delivered, explicitly scoped CUDA/HIP Linux artifact with
retained exact-byte installation evidence. It does not extend the historical
three-vendor matrix, the tiny-LM certificate, or any external performance claim.
