# Release runbook

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

Never rebuild after a successful qualification and silently call the new
artifact equivalent. The workflow records and rechecks wheel digests before
upload.

## 5. Close the release

Confirm the project page exposes the intended files and hashes, then install
from PyPI into clean environments and run the documented smoke path. Record
the workflow run, wheel hashes, supported architectures, and any unrun column
in the release evidence.

If any artifact is wrong, stop. PyPI files cannot be replaced under the same
version; fix the issue and publish a new version.
