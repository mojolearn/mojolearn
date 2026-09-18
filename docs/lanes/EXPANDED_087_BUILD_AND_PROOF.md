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
2. Public API availability and both new native IVF symbols, failing closed.
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
   One/two/reversed classical layouts and split/reversed loaded-LM owners are
   checked without another wheel install. Guarded archive source is exported by
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
