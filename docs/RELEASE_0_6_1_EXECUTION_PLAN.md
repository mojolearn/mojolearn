# 0.6.1 combined Linux execution plan

The 0.6.1 number in this file's name was never published: the profile it
describes, now named `release-linux3`, ships the version in
`python/mojolearn/_version.py` (0.7.0 at the time of writing), and no script
on the release path compares against a version literal any more (DEVIATION
2290; `release-0.6.1` remains accepted as a deprecated alias of the profile).

Source-only implementation, no executions or publication by the author.
Root/main alone runs all jobs, tests, builds and measurements; subagents never
run them. Preserve PyPI 0.6.0 immutable files and all historical failures.

Root follow-up: bounded packer, three-architecture admission, legacy and alpha
publication fixtures passed on September 7 against the then-declared 0.6.1
metadata; the same fixtures now derive the version from `_version.py`. These
are file checks, not native/runtime qualification or publication.
[Retained checks](../bench/results/releases/2026-09-07-combined-linux-preparation/README.md).

The requested payload is one Linux wheel containing CUDA sm_89, CUDA sm_90
and HIP gfx942, with FAST, DETERMINISTIC and IDENTICAL for each. The existing
architecture-aware loader chooses a compatible set; the wheel tag does not
choose a GPU vendor. An architecture string/readback proves compiled coverage,
not successful execution on that architecture.

## Implemented source contract

`packaging/linux/pack_wheel.py --profile release-linux3` now requires exactly
those three sets and three corresponding complete build proofs, all from the
same full source commit and identical current source inventory. Binary hashes
must equal the proof's exact 45 standard outputs plus one IDENTICAL-only
byte-LM extension per architecture: 46 per architecture, 138 in the combined
wheel. Unexpected native files are
refused rather than silently omitted. Existing artifacts cannot be overwritten.

The RECORD-covered `mojolearn-<version>.dist-info/LINUX_PAYLOAD.json`
(`mojolearn-0.7.0.dist-info/LINUX_PAYLOAD.json` for the version in
`python/mojolearn/_version.py` at the time of writing) records every
extension, runtime and Python hash, per-architecture proof/source hashes,
runtime layout and source inventory. Its runtime coverage is explicitly
pending until separate exact-wheel installed evidence exists. It does not
certify numerical behavior. File fixtures are authored in
`packaging/linux/test_release_061_inventory.py`; root must run them.

Byte-LM is now required in this profile for CUDA sm_89/sm_90 and HIP gfx942,
strictly in IDENTICAL. Its manifest explicitly marks FAST and DETERMINISTIC
unsupported. Generic historical packing still expects the legacy45. The
coordinated build/stage and admission implementations must be frozen together;
the installed byte-LM step/checkpoint job is additional to the original24,
giving25 jobs per architecture. Imports alone do not qualify the native step. MLP also
needs explicit installed symbol/step coverage beyond the old 24 surfaces.

## Sequential root jobs and evidence

1. Freeze the source/version metadata (`python/mojolearn/_version.py`, 0.7.0 at
   the time of writing) and all packaging/qualification changes.
   Run bounded file/host checks in the main thread only. Known corrections
   must include context lifetime/full-FP32 work after `eb835021` and the later
   Mamba3 FAST dt-softplus repair validated at `4a271ae6`, not only `3963a0fc`.
   Retained `bench/results/resume/2026-09-06-root-feature-nvidia/README.md`
   does not establish DETERMINISTIC or final-wheel qualification. Keep that
   mode in the fresh matrix without changing tolerances.

2. Reconcile lease ownership before any rental. Root guards each remote build
   with <=2 CPU affinity cores, one binding build at a time, compiler workers2,
   BLAS/OpenMP1, RSS/VRAM limits, deadline, fetch reserve and independent
   teardown watchdog. `tools/linux_surface_qualification.sh build OUT`
   creates one complete architecture proof.
   Root has also changed `packaging/linux/build_sets.sh` to a two-core
   inherited-affinity cap with child readback; that source change must be in
   the frozen inventory and is not itself an executed build result.
   Build sm_89 and sm_90 serially
   into separate fresh outputs, setting the explicit architecture per build
   and retaining native architecture readback. Cross-compilation, if it
   succeeds, still does not establish runtime coverage on the other GPU.

3. Build gfx942 serially from the same frozen inventory. Fetch/verify proofs
   and binaries before deleting each resource; verify resource absence. Keep
   transport snapshots separate from the numerical/build inventory. Root may
   schedule packing on an existing guarded remote Linux host to avoid a local
   memory-intensive ZIP operation.

4. Pack the exact three retained sets with three proofs:

   ```sh
   python packaging/linux/pack_wheel.py --profile release-linux3 \
     --set "$CUDA89_SET" --set "$CUDA90_SET" --set "$HIP942_SET" \
     --build-proof "$CUDA89_PROOF" --build-proof "$CUDA90_PROOF" \
     --build-proof "$HIP942_PROOF" --out "$NEW_OUTPUT"
   ```

   All variables identify root-retained fresh outputs, not historical overlays.
   Shared runtime bytes are deduplicated only when hashes agree; disagreeing
   CUDA runtime closures are refused. Root measures actual compressed size
   after packing using the emitted `SIZES-<version>-linux.json`; there is no size
   estimate or assumption of PyPI admission here. Audit/repair to the measured
   manylinux floor on remote Linux. Bind the repaired final SHA and preserve
   original payload/repair evidence. Audit changes may change binary hashes;
   any such change needs explicit repaired-byte provenance, never a bypass.

5. The explicit multiarchitecture qualification/admission path is now authored
   but unexecuted. Put the three unchanged proofs in `PROOF_DIRECTORY` as
   `cuda-sm_89.json`, `cuda-sm_90.json`, `hip-gfx942.json`. Root runs:

   ```sh
   tools/linux_surface_qualification.sh qualify-release-linux3 \
     "$FINAL_WHEEL" "$FINAL_SHA256" cuda "$NEW_SM89_OUT" "$PROOF_DIRECTORY" sm_89
   ```

   Run corresponding sm_90 and hip/gfx942 commands on their actual devices.
   Preflight verifies all138 files, complete current source/proofs and the
   embedded payload manifest. Each installed job rejects architecture overrides
   and independently records detected device and selected binding architecture.
   The legacy `qualify` path remains separate. Stage fetched results in
   `qualification/{cuda/sm_89,cuda/sm_90,hip/gfx942}/`, including each run's
   copied `build-provenance.json`, and all three original proofs in
   `qualification/build-proofs/`. Final file-only root admission is:

   ```sh
   python tools/check_linux_release_qualification.py "$FINAL_WHEEL" \
     --qualification-root "$QUALIFICATION_ROOT" --source-root "$SOURCE_ROOT" \
     --profile release-linux3
   ```

   It recomputes all three architecture audits, all24 standard installed records plus
   the IDENTICAL byte-LM step/checkpoint record per
   architecture and both HIP-versus-CUDA UMAP/Ordered comparisons. One CUDA run
   cannot satisfy both runtime columns. Source equality, full payload coverage,
   hashes, nested corpus fingerprints and numerical jobs remain required.

6. Install the same repaired final wheel bytes into clean environments on
   actual sm_89, sm_90 and gfx942 devices. Run all25 installed jobs on each
   claimed runtime architecture, serially with bounded per-job deadlines.
   Retain source/corpus/dependency snapshots, installed native hashes and
   vendor/mode/architecture readbacks, all status rows and raw UMAP/Ordered
   evidence. Prevent checkout shadowing and do not overlay binaries after
   installation. Add checks for newly advertised symbols; alpha exposure
   remains separate from configuration-specific numerical certification.

7. Root validates final-byte receipts and scope, then stages the explicitly
   alpha artifact of the declared version through the user-authorized Trusted Publisher route
   described in `docs/PYPI_RELEASE.md`. If an alpha overlay is used, qualify
   its final bytes rather than borrowing admission from the pre-overlay wheel.
   Do not dispatch an Apple build or invent a new macOS qualification.
   Record PyPI public filenames/hashes and installed public-file readback;
   retain every failure/unrun architecture and avoid general identity claims.

## Rental budget is conditional

Two rentals are feasible only if one NVIDIA rental exposes both required
runtime architectures (or an already retained exact-final-wheel qualification
exists for the other architecture) and one AMD rental supplies gfx942, with
enough time to build, assemble, return the final bytes and test before fetch
reserve. A typical single RTX4090 or single H100 cannot satisfy both runtime
columns. Otherwise a third hardware session, or a clearly unqualified runtime
column under alpha scope, is necessary. Packing cross-compiled binaries on one
host is not a substitute. Root chooses scheduling from actual resources and
remaining lease time; this document authorizes no unattended renewal.

The separately qualified bidirectional tiny byte-LM continuation at frozen
`d921eade` remains scoped to its retained binaries, shapes and raw trajectories;
it does not qualify this wheel or its optional module inventory.

## Prepared-host build entrypoint

`tools/release061_remote_build.sh` is authored for root use on one already
rented/prepared Linux host. It does not install dependencies, rent, transfer,
pack or publish. Prerequisites: existing absolute stdlib Python, `taskset`,
`objdump`, `pixi` on PATH and the locked Pixi **default** environment already
installed (`.pixi/envs/default/bin/{python,mojo}`), plus `patchelf` on PATH.
The helper refuses missing `patchelf` before compilation, preventing the older
build script's implicit installer fallback. It pins the kernel column to the
selected vendor and the Linux CPU target to `x86-64-v3`, overriding inherited
shell settings. Root passed its bounded shell syntax check; no native build
is implied. Root prepares that environment
with the frozen `pixi.lock`; no gbmbench/external benchmark dependencies are
requested by this helper. Native build scripts retain their own normal Pixi
invocations, so any lock/source drift during build remains a hard failure.

Root sets `MOJOLEARN_COMMIT` to the frozen full40 SHA,
`MOJOLEARN_PYTHON` to an existing absolute Python, and
`MOJOLEARN_RELEASE_BUILD_SECONDS` to 120..2400 remaining work seconds **after**
subtracting the rental's fetch/teardown reserve. Then, on the actual device:

```sh
bash tools/release061_remote_build.sh cuda sm_89 /absolute/new-build-evidence
```

Other accepted pairs are `cuda sm_90` and `hip gfx942`. The helper uses the
matching vendor serial guard, 12GiB RSS cap, hard two-core affinity, one binding
compiler invocation at a time and at most two compiler workers. It runs only
the bounded Linux resource-prefix tests, physical-device/source preflight,
full46 build (45 standard plus IDENTICAL byte-LM) and postbuild proof linkage. Jobs share a shrinking deadline,
reserve30 seconds internally, retain command/log/status files, and stop on the
first failure. Cancellation signals the active guard and waits for its child
cleanup. A successful exit is `BUILT_NOT_INSTALLED`, never wheel admission.
Retain `build/build-provenance.json`, complete `build/sets/`, logs and preflight
alongside the controller's teardown evidence before releasing the host.
