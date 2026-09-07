# AMD-only Linux release admission plan

Source-only audit, 2026-09-06. No release gates were changed and no tests,
measurements, builds, model code, GPU work, provisioning or publication were
performed for this audit. The latest user instruction is **skip Apple
testing**. Any new hardware execution belongs to the root/main thread on
remote AMD/NVIDIA; subagents must never run it.

Target: publish an explicitly AMD-only Linux 0.6.0 artifact and, if its exact
retained evidence permits, the existing macOS 0.6.0 artifact. NVIDIA remains
source-build-only. This requires an explicit admission path; accepting a HIP
candidate through the current combined-wheel gate would be incorrect.

## What the current source already does

| File | Observed contract | Change needed |
| --- | --- | --- |
| [pack_wheel.py](pack_wheel.py) | `--set` accepts CUDA or HIP sets; despite dual-vendor prose/help, `main()` does not require both vendors. Every supplied architecture must carry 15 bindings in each of three modes, with vendor/architecture readbacks and runtime closure. | Add an explicit release profile and machine-readable advertised-vendor/architecture metadata; update misleading “give both” documentation. A new single-vendor packing algorithm is unnecessary. |
| [audit.sh](audit.sh) | Accepts one or more manifests, derives driver/runtime exclusions, repairs to a measured manylinux tag and checks metadata. | Document/use the one-HIP-manifest case on remote Linux. Preserve real audit/repair/ISA checks and final-byte qualification. Do not infer manylinux compatibility from a filename. |
| [linux_surface_qualification.sh](../../tools/linux_surface_qualification.sh) | Its inline wheel audit already admits a single-vendor candidate, requires the selected vendor and every advertised set to be complete, and records `advertised_vendors`. It runs eight surfaces in three modes: 24 serial jobs. | Bind the declared release profile to actual payload and qualification metadata; retain all 24 jobs, all binding hashes, corpus snapshots and source proofs. Do not turn candidate admission into publication admission. |
| [check_linux_release_qualification.py](../../tools/check_linux_release_qualification.py) | `inspect_wheel()` requires exactly HIP and CUDA. `check_vendor()` requires advertised vendors `['cuda', 'hip']`. `check()` requires both qualification directories and recomputes cross-vendor UMAP/Ordered comparisons. | Add a separate explicit HIP-only branch as described below. Preserve the existing dual-vendor default and all its checks. |
| [_backend.py](../../python/mojolearn/_backend.py) | `_layout()` discovers only present vendor sets, selects matching hardware and refuses absent/unsupported sets before native loading. HIP-only layout already works structurally. | Improve missing-CUDA diagnostics for the declared AMD-only release; retain no-CPU-fallback and no-wrong-vendor-load behavior. Check with authored mocked cases and root-run remote checks. |
| [release-provenance.yml](../../.github/workflows/release-provenance.yml) | Builds and executes Apple qualification on every dispatch; optionally stages a Linux wheel through the dual-vendor checker. Publisher consumes exact wheel digests from the build job. | An evidence-admission route for prebuilt macOS is required to obey “skip Apple testing”; staging HIP-only Linux needs an explicit profile passed to the strengthened checker. Do not dispatch the current build workflow expecting it to skip Apple work. |

## 1. Freeze the release scope and artifact contract

- [ ] Inspect actual index files and hashes before upload; this audit did not
  query PyPI and does not establish that version 0.6.0 is still unpublished.
  An existing same-version filename cannot be overwritten.
- [ ] Define an explicit profile, for example `hip-only`, alongside existing
  `dual-vendor`. Keep `dual-vendor` the admission default so an accidental
  missing CUDA set is still a failure.
- [ ] Put a versioned release-scope manifest inside the Linux wheel, covered
  by RECORD. Bind profile, advertised vendors, exact supported architectures
  and numeric modes to the actual extension inventory. HIP-only means
  `['hip']`, complete FAST/DETERMINISTIC/IDENTICAL sets and **no CUDA extension
  payload**. Do not advertise all AMD architectures from one gfx942 result.
- [ ] Make the CLI-selected profile, embedded manifest, qualification audit
  and release admission report agree. Unknown profiles, undeclared payloads,
  missing modes and attempts to select HIP-only for a combined wheel fail.
- [ ] Document that wheel tags do not encode GPU vendor. The ordinary Linux
  wheel can be selected by pip on NVIDIA; it must then give a clear
  source-build-only explanation before loading incompatible binaries. Do not
  publish HIP-only and combined wheels with indistinguishable selection tags
  and assume pip chooses based on hardware.

## 2. Implement the smallest explicit admission extension

Changes belong in
[tools/check_linux_release_qualification.py](../../tools/check_linux_release_qualification.py)
and its [existing tests](../../tools/test_check_linux_release_qualification.py).

- [ ] Add a profile parameter/CLI option, default `dual-vendor`, and return it
  in a versioned admission report. Share validation code; do not create a
  permissive “skip CUDA” switch.
- [ ] Parameterize the exact required vendor set in `inspect_wheel()` and
  advertised-vendor equality in `check_vendor()`. Preserve RECORD/path
  integrity, complete per-architecture mode sets, Python-source equality,
  repaired tag, build proofs, native inventory equality, exact final wheel
  SHA, isolated installed records and native mode/vendor readback.
- [ ] In `check()`, HIP-only requires `qualification/hip/`, all 24 passing
  installed jobs and current nested Mamba corpus fingerprints. It does not
  require a fabricated CUDA qualification. Unexpected staged evidence must
  not expand the report's admitted vendor scope.
- [ ] Keep `surface.compare(hip,cuda)` and
  `compare_ordered_python.compare(hip,cuda)` mandatory for dual-vendor.
  For HIP-only report these cross-vendor claims as **not applicable to this
  artifact**, never PASSED. Preserve per-vendor UMAP quality and validate the
  retained OrderedRMSE output/provenance independently. If the existing
  comparator combines integrity and cross-vendor checks, factor reusable
  single-vendor integrity checks instead of dropping them wholesale.
- [ ] Emit an unambiguous scope: exact HIP-only wheel, named architecture,
  24 installed jobs and no NVIDIA binary qualification or fresh
  cross-vendor identity certificate. Include wheel/source/qualification
  hashes and admitted sets in the retained report.
- [ ] Author refusal checks for profile mismatch, missing CUDA under the
  default profile, CUDA payload under HIP-only, incomplete HIP modes,
  stale source/proof/wheel hashes, failed/missing jobs, omitted nested
  corpora, malformed quality and modified installed bindings. Root alone
  runs them remotely; this audit only specifies them.

The [existing retained-evidence verifier](../../tools/verify_linux_surface_qualification.py)
and [qualification driver](../../tools/linux_surface_qualification.sh) already
record useful single-vendor evidence. Extend their schema only where the new
manifest requires it. Do not relax their numerical quality thresholds,
surface count or native provenance requirements.

## 3. Packaging, runtime and workflow integration

- [ ] Extend [pack_wheel.py](pack_wheel.py) to emit and validate the declared
  profile. Pack only the qualified HIP architecture directories and their
  required runtime closure; retain architecture/readback validation and
  [extension-inventory consistency](../check_ext_lists.py).
- [ ] Audit/repair on remote AMD Linux, preserving logs and excluded driver
  libraries. Run the existing CPU ISA inspection on the final archive on
  remote Linux. If normalization removes empty ZIP directories, verify
  unchanged payload/RECORD bytes and retain both hashes. Qualify the exact
  final repaired/normalized artifact after all mutations.
- [ ] Update `_backend.py` diagnostics to distinguish “NVIDIA device present
  but this wheel is AMD-only” from “no GPU device found”, with a concrete
  pinned source-build route. Preserve absent-architecture refusals and do
  not use force-vendor overrides as the default recovery instruction.
- [ ] Add an explicit Linux profile input/staging field to the release
  workflow; pass it into final admission and retain the admission JSON as
  a publication artifact. Verify the wheel SHA again immediately before
  recording/uploading digests. A sidecar alone is insufficient evidence.
- [ ] Preserve Trusted Publisher OIDC, index selection, version checks,
  artifact digest verification and publication serialization. Update
  [RELEASE_QUALIFICATION.md](RELEASE_QUALIFICATION.md),
  [PYPI_RELEASE.md](../../docs/PYPI_RELEASE.md),
  [LINUX_WHEEL.md](../../docs/LINUX_WHEEL.md) and current support/release
  wording only when the new contract and actual delivery are evidenced.

## 4. Existing macOS evidence, without Apple testing

The current workflow's `Build the wheel, then install and RUN it` and
`Qualify installed UMAP fit, transform and held-out quality` steps are real
Apple execution. They cannot be used unchanged under the latest instruction.

- [ ] Add a deliberately selected **prebuilt-evidence admission** route,
  preferably a distinct job/workflow, which consumes the exact retained
  macOS wheel and its complete qualification evidence. Keep the normal
  build-and-qualify route intact; do not turn a skipped Apple job into PASS.
- [ ] Reuse the read-only
  [UMAP qualification verifier](../../tools/verify_umap_qualification.py)
  against the original qualification source snapshot. Also admit retained
  full interpreter/mode verification records, package/binding hashes,
  embedded-GPU/ISA results and version metadata; a wheel digest plus a README
  does not replace those records.
- [ ] Bind macOS evidence to its **own exact source/artifact**, separately
  from the current AMD source. If publishing different source revisions under
  one version is selected, retain an explicit per-platform source manifest,
  accurately document feature differences and satisfy metadata/version
  consistency checks. Never compare old macOS evidence to new bytes and
  call it current qualification.
- [ ] If the retained macOS artifact or proof is incomplete/stale for the
  intended release claim, leave that artifact blocked or publish only an
  independently admissible artifact. Do not run Apple tests to repair the
  gap under the current user instruction. No new macOS post-install GPU
  execution is planned; final index hash/readback verification can be
  performed remotely without claiming a new Apple test.

This is an engineering dependency for the requested release, not permission
to weaken admission. A new macOS artifact cannot inherit old qualification.

## 5. Evidence dependencies and known exclusions

The inspected
[installed gap-closure record](../../bench/results/resume/2026-09-06-installed-gap-closure/README.md)
binds 24 passing AMD installed jobs to source
`eb835021dcd79a59a7e8f78c754a75db3c1fea83` and wheel SHA256
`7c5f9af825cbcbd74a293adc75ad15670a30a179d3a9a8a8cfd993cd9476f7be`.
Its native source inventory SHA256 is
`ade965b90496132596d8dda79860a87f472193c14129093c5c3a72f273b8159f`.
Later wrapper/native changes are not qualified by that result. Changing
`packaging/linux/*.py` also changes the native inventory used by current
admission; freeze the new release implementation before recording fresh
build/qualification proofs.

The [Apple follow-up record](../../bench/results/resume/2026-09-06-feature-finish/README.md)
and [roadmap](../../ROADMAP.md) record fifteen interpreter/mode passes for
wheel SHA256
`061d9f47acdce1f6f4d3fa451fc03a7b55ea8909f8e84a318e02723a41739027`.
Retained underlying machine-readable proof must be admitted before reuse;
this source-only audit did not execute a verifier or establish publication.

The frozen NVIDIA candidate retained 19 passes and five failures (two Mamba
accuracy failures and three Transformer stalls). Source overlays improved
them; neither overlays nor a narrow new gradient result qualify a final
NVIDIA wheel. Keep NVIDIA source-build-only until full current-source,
final-byte installed qualification passes. AMD-only release must not erase
these failures or imply they were rerun successfully.

## Root execution order after implementation

1. Freeze profile/schema/runtime/workflow changes and author admission cases
   through source-only parallel work; no subagent execution.
2. Root runs bounded admission checks remotely, then one serial AMD build,
   pack/repair/audit and exact-wheel 24-job installed qualification, with CPU
   and memory caps, deadlines and a cleanup watchdog.
3. Root checks HIP-only refusal on NVIDIA through a bounded safe import
   diagnostic where useful; this is not NVIDIA wheel qualification. Run no
   local Apple tests, builds or model workloads.
4. Admit retained macOS evidence without execution, or retain an explicit
   unresolved macOS delivery gap. Retain reports for each artifact separately.
5. Confirm index availability, publish only admitted exact artifacts through
   OIDC, verify public hashes remotely and update release/support wording
   with evidence. Confirm rental teardown and preserve failed/stopped jobs.

This plan complements [FEATURE_COMPLETION_PLAN.md](../../FEATURE_COMPLETION_PLAN.md)
and does not change the original Mamba2/3, symmetric-tree, UMAP or NVIDIA-only
performance comparison objectives.
