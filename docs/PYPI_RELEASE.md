> Start at `docs/RELEASE_CHECKLIST.md`. This file is background for when a step there refuses.

# Release runbook

## Explicit alpha API exposure

The user-authorized alpha route exposes implemented Python operations before
completion of every numerical certificate. It is separate from the stable
build-and-qualify route below. It normally requires `X.Y.ZaN`; the explicitly
requested `0.6.0` uses `--allow-alpha-final-version` and the manifest
`release_profile: "alpha-api"`, retaining the Alpha classifier.
`packaging/alpha_overlay.py` preserves the base wheel's native/runtime bytes,
overlays current Python modules and the alpha guide, and records both source
and native provenance. Missing native extensions remain unavailable; neither
file presence nor successful imports certify numerical behavior.

Root assembles each candidate from an identified base wheel, verifies the
result with `packaging/verify_alpha_artifacts.py`, and retains compatibility
checks separately. The candidate directory contains exactly its wheels and
`alpha-manifest.json` (`mojolearn.alpha-release.v1`, alpha `version`, and a
`files` mapping from each exact wheel basename to its SHA256).

For publication, stage those exact files as assets of an `alpha-api-*` release
tag. Dispatch the existing `release-provenance.yml` Trusted Publisher workflow
with `alpha_candidate_tag`, the exact `alpha_manifest_sha256`, and `publish`.
The alpha route uses Linux file verification capped at two CPU cores, checks
the hashes again immediately before OIDC upload, and does not start the Apple
build job. Stable publication retains its existing checks. A local candidate
or a workflow source edit is not publication: verify the actual PyPI filenames
and hashes before updating release status.

PyPI 0.6.0 was published and its exact two wheel hashes verified on September
6. Evidence is retained under `bench/results/releases/2026-09-06-alpha-api/`.
The published alpha API uses inherited macOS and AMD binaries; newly authored
byte-LM native code needs a separate build. NVIDIA remains source-build-only.

Releases are built and published by
`.github/workflows/release-provenance.yml`. Publishing uses GitHub Trusted
Publisher OIDC; do not add an API token. The workflow is manual-only and its
build runs on a deliberately started, ephemeral Apple-silicon GPU runner.

The release may contain:

- one `py3-none-macosx_11_0_arm64` wheel, built by the workflow;
- optionally one prebuilt Linux x86-64 wheel carrying CUDA and HIP sets.

There is no sdist. Each wheel contains the requested numeric-mode sets; normal
releases carry FAST, DETERMINISTIC, and IDENTICAL. The extension inventory is
15 per mode and is checked mechanically by `packaging/check_ext_lists.py`.

## 1. Prepare the source state

Update the same `X.Y.Z` in:

- `python/pyproject.toml`
- `python/mojolearn/_version.py`
- `CITATION.cff`, including `date-released`

Update `CHANGELOG.md`, commit the exact release state, and run the release
preflight:

```sh
pixi run probe
bash packaging/release_workflow_test.sh
python packaging/check_ext_lists.py
git diff --check
```

Do not release from a dirty tree or treat a source-tree pass as an
installed-wheel pass.

## 2. Optional Linux artifact

Linux vendor sets must be built on their actual GPU vendors using
`packaging/linux/leg.sh`, combined with `packaging/linux/pack_wheel.py`, and
repaired/audited with `packaging/linux/audit.sh`. Follow each script's help
and fail-closed checks; architecture coverage must be explicit.

Run installed-wheel smoke on supported NVIDIA and AMD targets. One successful
build box does not certify another GPU architecture. Retain the resulting
logs under `bench/results/wheels/`.

Stage exactly one final Linux wheel and its matching `.sha256` sidecar in:

```text
${MOJOLEARN_LINUX_WHEEL_DIR:-$HOME/.mojolearn-linux-wheel}
```

Leave that directory absent or empty for a macOS-only release. The workflow
rejects ambiguous, mismatched, wrongly versioned, or unaudited artifacts.

## 3. Start the one-job runner

On the Apple-silicon release Mac:

```sh
tools/release_runner.sh --dry-run
tools/release_runner.sh
```

The real invocation registers one ephemeral runner and waits in the
foreground. Keep it open. It removes its registration after the job exits.

The machine must have pixi and Python 3.10 through 3.14 available. The workflow
builds on real Metal hardware, checks embedded GPU code and ISA/minimum-OS
requirements, installs the wheel into clean environments, and runs every
claimed interpreter and numeric mode.

For the additional Python 3.12 UMAP qualification, dependencies can be staged
before starting the runner to avoid an index outage after the build:

```sh
python3.12 -m pip download --only-binary=:all: --no-deps \
  --dest "$HOME/.mojolearn-qualification-wheelhouse" 'numpy>=1.24'
MOJOLEARN_QUALIFICATION_WHEELHOUSE="$HOME/.mojolearn-qualification-wheelhouse" \
  tools/release_runner.sh
```

When configured, that gate uses `--no-index` and the supplied wheelhouse;
missing or incompatible dependencies fail installation. It records dependency
wheel hashes, installed versions and `pip check`, while retaining the exact
candidate wheel and all GPU checks. Other build and smoke steps may still
need network access. The standalone qualifier accepts the same directory via
`--wheelhouse`.

## 4. Dispatch

From another terminal, first build without publishing:

```sh
gh workflow run release-provenance.yml --ref <release-commit-or-tag> -f publish=none
gh run watch
```

Inspect the complete job and artifact manifest. A skipped interpreter, absent
mode, missing extension, GPU smoke failure, or digest disagreement blocks the
release.

Then dispatch to TestPyPI when qualification is needed:

```sh
gh workflow run release-provenance.yml --ref <release-commit-or-tag> -f publish=testpypi
gh run watch
```

Install from TestPyPI in a clean environment and run representative public
surfaces in all three modes. Dependencies may need the normal PyPI index as an
extra index.

Finally publish the same source state to PyPI:

```sh
gh workflow run release-provenance.yml --ref <release-commit-or-tag> -f publish=pypi
gh run watch
```

Each dispatch builds a fresh artifact. Its own interpreter smoke, installed
UMAP fit/transform/held-out quality checks and digest gates must pass before
publication. Never label that artifact as byte-equivalent to an earlier
candidate without comparing the digests. The workflow retains the UMAP
qualification manifest and rechecks wheel digests before upload.

Immediately before recording upload digests, the workflow also checks that
the macOS wheel matches the successful UMAP qualification digest, all nine
installed GPU jobs and four setup jobs succeeded, the qualification sources
are unchanged, and the wrapper and per-mode binding hashes match the wheel.
This prevents a replaced candidate from acquiring a new upload digest after
qualification. Retained evidence can be checked without GPU work:

```sh
python3 tools/verify_umap_qualification.py /path/to/candidate.whl \
  --results /path/to/qualification/results.json \
  --source-root . --expected-version 0.6.0
```

This reads the existing evidence and artifact only. It neither reruns the
installed checks nor certifies a different source state or Linux wheel.

## 5. Close the release

Confirm the project page exposes the intended files and hashes, then install
from PyPI into clean environments and run the documented smoke path. Record
the workflow run, wheel hashes, supported architectures, and any unrun column
in the release evidence.

If any artifact is wrong, stop. PyPI files cannot be replaced under the same
version; fix the issue and publish a new version.
