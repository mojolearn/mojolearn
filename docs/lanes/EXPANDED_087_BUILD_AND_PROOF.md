# Expanded 0.8.7: build and installed proof path

2026-09-18. This is the expanded source candidate, not the older frozen wheel.
No build, rental, workflow dispatch or publication was started by this audit.

## What must change

Use one new immutable source commit after the intended feature/proof merges.
Both existing builders already include `build_ivf.sh`, neural bindings and the
host-surface manifest. No new native binding family is needed. The distributed
IVF implementation does require its newly exported `ivf_flat_partial_search`
and `ivf_finalize_distances` symbols; copying Python into an old wheel is not a
valid substitute for rebuilding the IVF binding.

The builders now fail `tools/wheel_api_audit.py --require-complete` if any current
public API or Python implementation/verifier bytes are absent or stale. This
includes models and distributed verifier subpackages. Packaging exposure is
not numerical qualification.

The existing `qualify_verifier_wheel.py` installed gate is extended, so both the
macOS release workflow and Linux supplemental release checks automatically run:

1. Fresh isolated wheel install with dependency check and import-origin guard.
2. Public API availability (including parallel CV), both new native IVF symbols
   and the native GBDT fit entry for CV, failing closed.
3. Loaded-model CPU and GPU v2 captures: all seven supported checkpoint families,
   24 cases across FP32/BF16/int8, then exact CPU/GPU comparison of 144 parts.
4. Existing installed coverage, bundled models, batch/property and self-test gates.

Each stage has its existing180-second bound and writes its logs/checkpoint before
continuing. The receipt is now written atomically. `--expected-source-commit`
must match the wheel's packaged `identity_columns/COMMIT`; both release callers
pass their exact checkout/archive commit. Linux native provenance continues to
be checked by the separate existing build-inventory admission gate.

`--scope cpu-only` explicitly runs coverage, bundled models and loaded-model CPU
proof without GPU claims, returning `PASSED_CPU_ONLY`. Default `expanded` refuses
a CPU-only backend. Python `-O` is refused because legacy admission functions
use assertions; child environments also remove `PYTHONOPTIMIZE`. New API/native
checks use explicit exceptions.

## Shortest reproducible campaign

1. Create a clean release worktree at the final integrated source SHA. Record
   `git rev-parse HEAD`; verify `python/pyproject.toml` and `_version.py` both say
   0.8.7. Use this same SHA for the source archive, every build, packaging and
   installed proof. Do not move the pin mid-campaign.
2. On prepared physical Linux builders, run the existing bounded command:

   ```sh
   MOJOLEARN_COMMIT="$SOURCE_SHA" \
   MOJOLEARN_PYTHON=/absolute/existing/python \
   MOJOLEARN_RELEASE_BUILD_SECONDS=2400 MOJOLEARN_BUILD_JOBS=1 \
     bash tools/release061_remote_build.sh cuda sm_89 /absolute/new-output
   ```

   Repeat for `cuda sm_90a` (or the supported `sm_90` spelling) and `hip gfx942`.
   The historical script name does not select a historical release version.
   Parent controllers retain responsibility for lease deadline, fetch and delete.
   The script records source/native inventory, runtime architecture, bindings and
   build provenance, and refuses partial/reused output. The locked Pixi default
   environment and tools must already be prepared.
3. Match Linux linker/CRT provenance across vendors before full AMD compilation.
   `tools/release_ubuntu22_build.sh prepare/run` provides the pinned Ubuntu22.04
   AMD userspace path and checks a small core-host binary against the NVIDIA
   `MOJOLEARN_EXPECT_CORE_HOST_SHA256` first. This avoids paying for a full build
   that the shared host-byte gate later rejects. It is an existing prepared-host
   command, not an instruction to provision another rental automatically.
4. Stage the three architecture sets and their build-provenance JSON files.
   Pack from the pinned worktree using `packaging/linux/pack_wheel.py` with
   `--profile release-linux3`, `--set` for the CUDA/HIP set directories and one
   `--build-proof` for each architecture. Auditwheel repair/runtime exclusions
   and directory normalization produce the **final** wheel SHA. Do not qualify
   a pre-repair candidate and then publish different bytes.
5. Install that exact final wheel on every advertised architecture using
   `tools/release_installed_checks.sh qualify-release-linux3 WHEEL SHA VENDOR OUT
   PROOFS ARCH`. On two-device NVIDIA and AMD qualification hosts, set
   `MOJOLEARN_RELEASE_MULTI_GPU_DEVICES=0,1`; this forwards to the expanded gate.
   One/two/reversed classical layouts, installed cross-validation, and
   split/reversed loaded-LM owners are checked without another wheel install. Guarded archive source is exported by
   the existing supplemental script. Every output directory is new and retained.
6. Assemble the retained per-architecture proofs and run
   `tools/check_linux_release_qualification.py ... --profile release-linux3`.
   Its existing policy requires the three architecture slots and a full column
   for each vendor; declared smoke/known-failure allowances remain explicit.
   **That existing admission is not the complete new multi-GPU release gate.**
   Review the expanded receipts alongside it: missing two-device runs remain
   OWED. Classical driver placement is checked; loaded-LM split/reversed
   numerical checks still owe their own placement trace. Physical kernel traces
   remain separate evidence even after numerical and driver-placement checks pass. No expanded receipt sets
   `release_qualified=True`.
7. Apple uses `packaging/macos/build_release_wheel.sh` with the same source pin
   and `MOJOLEARN_PACKAGE_BYTE_LM=1`; the workflow retains five-interpreter,
   UMAP and installed verifier qualification. The existing instruction to hold
   another full Apple campaign remains in force; this document launches none.

## Workflow inputs and remaining release work

`release-provenance.yml` accepts `publish=none|testpypi|pypi`, optional
`alpha_candidate_tag`, and its required matching `alpha_manifest_sha256`.
The dispatch ref selects the source; there is no separate source-SHA input.
For a build-only review, use `publish=none` at the pinned ref. The normal path
builds on the self-hosted Apple GPU runner; the alpha path consumes prepared,
SHA-bound release assets instead. This report does not create tags/assets or
invoke `release_linux_publish.sh`, which would upload and dispatch.

`publish=none` does **not** run the workflow's full CPU certification job; that
job is conditioned on a publishing request. Successful build-only results are
therefore not publication admission. Expanded wheel work must still finish the
remaining ordinary reference holds, clean CPU architecture certification,
matching installed captures, physical multi-GPU execution evidence and any
numerical boundary debt selected for this candidate. Existing frozen artifacts
and old-source qualification receipts remain historical, not recertification of
new Python/native source.

Validation:13 release-qualifier/shell orchestration tests and8 public CLI tests
pass on the integrated source. No expanded candidate wheel has been built or
installed by this lane; these tests validate gating/orchestration, not hardware.

## Deferred local build: measured budget and exact launch recipe

New user instruction: commit/push progress, merge good work, no new pods; usage
budget is nearly exhausted. **Do not start this build now.** No fresh macOS build
was started during this lane.

The retained log at
`/Users/andrewhendel/mojolearn-evidence/release-087-final/logs/macos-build-full.log`
records61 fresh extension builds (29 GPU tier outputs and32 host families), one
at a time, followed by all five interpreter gates, completed in1175.24seconds
(19.6minutes). That is an observed earlier 0.8.7 build, not a timing guarantee
for the expanded source. A40-minute execution bound provides margin; expanded
loaded-model installed proof adds roughly20seconds on this machine, with other
qualification stages budgeted separately. No90-minute CPU certification matrix
is implied by the wheel build.

When resumed, first choose the final merged source pin including the CV CLI and
portable-model bundle, create a fresh dedicated build worktree, and verify it
contains no native output. The current `pixi.toml`/`pixi.lock` match the retained
release worktree's pinned toolchain at audit time. Prepare default/pkg/test
environments from that lock or deliberately link the verified existing pinned
environments; do not copy any old package `.so` artifacts. The builder needs
`.pixi/envs/default/lib` and the five claimed Python interpreters. Check these
before taking the shared slot.

With `ART` set to a new external artifact directory, create `ART/tmp` first,
then run from the fresh pinned worktree:

```sh
python3 tools/mac_slot.py --timeout 2400 --wait-timeout 600 \
  --timing-json "$ART/mac-slot.json" metal nice -n 19 \
  env -u PYTHONPATH -u PYTHONHOME -u MOJOLEARN_HOST_DIR \
      -u MOJOLEARN_BUILD_EXTRA_DEFINES -u MOJOLEARN_HOST_ALLOW_SABOTAGE \
      -u MACOSX_DEPLOYMENT_TARGET \
      MOJOLEARN_PACKAGE_BYTE_LM=1 MOJOLEARN_BUILD_JOBS=1 \
      MOJOLEARN_COMPILE_JOBS=1 MOJOLEARN_CPU_THREADS=1 \
      OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
      NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
      TMPDIR="$ART/tmp/" \
      bash packaging/macos/build_release_wheel.sh \
      > "$ART/build.log" 2>&1
```

The shared slot avoids overlap with the independent GEMM lane. One compiler and
math-thread settings constrain concurrency; they are not an OS hard RSS/CPU
limit. The full builder has its own fresh-output timestamp, native code,
runtime closure, ISA, packaging and installed-interpreter gates. A timeout keeps
per-extension logs under the external TMPDIR and does not produce a qualified
release. Do not weaken its fresh-build gates to resume with unwitnessed binaries.
Copy the completed wheel/API audit into ART, record SHA256, then run the expanded
installed qualifier with the same expected source SHA when budget allows. No
publication is authorized merely by successful local wheel construction.
