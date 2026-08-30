# Releasing `mojolearn` to PyPI

The runbook for `.github/workflows/release-provenance.yml` and
`tools/release_runner.sh`. Terse on purpose. Every command is meant to be
pasted as written, with `X.Y.Z` replaced.

## 1. What a release is here

* Up to TWO wheels under the project name `mojolearn`: the macOS arm64
  wheel, tagged `py3` and `macosx_11_0_arm64`, built by section 3, and from
  0.3.0 a Linux x86_64 wheel carrying both a CUDA and a HIP set, built by
  section 9 on rented boxes and handed to the workflow as a finished
  artifact. Each artifact serves python 3.10 through 3.14 because the
  extensions link no libpython (see `python/setup.py`). A release with no
  Linux wheel staged is macOS-only and that is not an error.
* The wheel carries TEN extensions in THREE numeric tiers. This bullet said
  "five extensions in TWO numeric modes"; both halves were false by 2026-08-29
  and are replaced, not softened. The FAST set lives at
  `python/mojolearn/*.so`, the other two at
  `python/mojolearn/deterministic/*.so` and `python/mojolearn/identical/*.so`.
  All three ship in the one wheel, and which tiers a wheel carries is ONE
  variable, `MOJOLEARN_RELEASE_MODES`, read by `build_release_wheel.sh` and
  `verify_wheel.sh` alike; set them the same for one release or the verifier
  fails a wheel for lacking a tier nobody asked it to build.
* The tier is a PARAMETER, not an install option, and never was an extra.
  `mojolearn.set_numeric_mode(...)` sets the process default in code and
  `Estimator(..., numeric_mode=...)` sets one instance's;
  `MOJOLEARN_NUMERIC_MODE` still sets the STARTING default at import and is
  the oldest spelling, not the only one (`python/mojolearn/_mode.py`,
  `python/mojolearn/_backend.py`). There is one distribution,
  `pip install mojolearn`, with no extras.
* The source targets Metal, CUDA and HIP from one tree, but only the macOS
  wheel is published so far. The Linux wheel (one wheel, CUDA and HIP sets,
  vendor picked at import) is designed and tooled in `docs/LINUX_WHEEL.md`
  and section 9 below is its flow; until it has run, no Linux wheel, and no
  sdist ever (section 8).
* The version is written in THREE places and all three must be bumped
  together, in one commit: `python/pyproject.toml` (`version = "X.Y.Z"`),
  `python/mojolearn/_version.py` (`__version__ = "X.Y.Z"`) and `CITATION.cff`
  (`version`, `date-released`). The workflow's "Version agreement" step
  refuses a build where the first two differ; it does NOT read `CITATION.cff`,
  so that one is on you, and a wrong `date-released` is minted permanently
  into a DOI. The CHANGELOG heading date is a fourth place and it is the one
  most easily left on the day the entry was drafted rather than the day the
  wheel shipped.
* The runner is your M4. GitHub's hosted macOS runner has no usable Apple
  GPU and produces a wheel with no Metal kernels; TestPyPI 0.1.0a2 was that
  wheel. The repo has no permanently registered runners; `tools/release_runner.sh`
  registers one ephemeral runner for exactly one job.

## 2. Preflight checklist

Check each; do not start the build until all hold.

* `git status` is clean for everything the wheel reads. `bindings/`,
  `packaging/`, `python/`, `pixi.toml`, `pixi.lock`, `README.md`, `LICENSE`,
  `NOTICE`, and every Mojo source directory the ten extensions compile.
  Uncommitted edits build into the wheel and are unreproducible afterwards.
* Version bumped in `python/pyproject.toml`, `python/mojolearn/_version.py`,
  and `CITATION.cff` (`version`, `date-released`). One commit, pushed to
  `main`.
* `README.md` at the repository root is what PyPI shows as the long
  description. `packaging/macos/build_release_wheel.sh` copies root
  `LICENSE`, `NOTICE` and `README.md` into `python/` on every build
  (`python/.gitignore` keeps the copies out of the checkout), and
  `python/pyproject.toml` reads `readme = "README.md"`. Read the root README
  as a PyPI visitor would before building.
* `LICENSE` and `NOTICE` exist at the repository root. Both are listed in
  `license-files` in `python/pyproject.toml` and ship in the wheel.
* No secrets anywhere in the tree or in the workflow. The publish job uses a
  Trusted Publisher (OIDC) and holds no API token. `grep -rn "pypi-Ag"
  .github tools packaging python` should print nothing (PyPI API tokens
  begin with that prefix).
* Trusted Publisher configured on BOTH indexes for owner `mojolearn`, repo
  `mojolearn`, workflow file `release-provenance.yml`, environment `pypi`
  (PyPI) and `testpypi` (TestPyPI). Check it; nothing in this repository can.
  * https://pypi.org/manage/project/mojolearn/settings/publishing/
  * https://test.pypi.org/manage/project/mojolearn/settings/publishing/
* GitHub environments `testpypi` and `pypi` exist on the repository
  (https://github.com/mojolearn/mojolearn/settings/environments). The
  workflow header records they were created 2026-08-20.
* The organization setting "Require approval for all outside collaborators"
  is ON at https://github.com/organizations/mojolearn/settings/actions. The
  release runner is a self-hosted machine on a public repo; that setting and
  the workflow_dispatch-only trigger are the two things keeping fork code off
  it.
* `gh auth status` succeeds on the M4 and the account is an admin of the
  repository (the runner registration token endpoint needs it).
* `python3.10` through `python3.14` all resolve on the M4. The workflow runs
  `actions/setup-python` for all five; if the first run reports `SKIP` for
  any interpreter, install that interpreter on the machine so it is on the
  runner account's PATH and re-run. A `SKIP` fails the job by design.

## 3. Build and verify locally

Always first, on the M4, before touching the workflow, and always from a
CLEAN CHECKOUT of the commit being released, never from the shared working
tree. The first clean-tree build on 2026-08-23 found that the per-script
gates in `bindings/build_*.sh` import the whole package and so need every
extension to exist already; the shared tree always had them, a fresh
checkout never does. `build_release_wheel.sh` therefore builds with those
gates off and runs `verify_wheel.sh` itself at the end as the one release
gate.

```sh
git worktree add --detach ../mojolearn-release vX.Y.Z
cd ../mojolearn-release && pixi install -e default -e pkg
./packaging/macos/build_release_wheel.sh      # builds, gates, and verifies
pixi run -e pkg twine check python/dist/*.whl
```

`verify_wheel.sh` installs the wheel into a fresh venv under every
`python3.10` .. `python3.14` it can find and runs `packaging/macos/smoke.py`
ONCE PER SHIPPED TIER per interpreter, and the smoke does real fits
on every estimator family (k-means, k-NN, gradient boosting, random forest,
extra trees, DBSCAN, PCA, truncated SVD, OLS) and asserts that
`mojolearn.numeric_mode()` reads back the mode that was asked for. All five
interpreters must print `PASS` for EVERY shipped tier; a `SKIP` means an interpreter
is missing from the machine and the release workflow will fail on it. Never
use `--no-gpu` for a release.

What `build_release_wheel.sh` refuses, from its own comments. Each is an
exit, not a warning.

* **Stale extension.** `pyproject.toml` globs `*.so`, so any extension not
  rebuilt in this run ships whatever old file sits in the tree. The script
  builds every extension itself before packing.
* **Unclosed dylib set.** `packaging/macos/stage_dylibs.py` walks the full
  transitive closure of MAX runtime dylibs for all extensions in one call,
  stages them under `python/mojolearn/.dylibs`, repoints rpaths, re-signs,
  and fails if anything is left unresolved. On the build machine the
  original rpath still resolves into the pixi environment, so this is the
  only check that can see the problem.
* **minos/tag mismatch.** Every extension's `LC_BUILD_VERSION minos` must
  equal `DEFAULT_MACOS_TARGET` in `python/setup.py`. One wheel carries one
  tag and the tag must be the floor of everything inside.
* **ISA baseline.** `packaging/isa_baseline.py` disassembles each extension
  and refuses bf16, i8mm or SME instructions, which would SIGILL on the
  older Macs the tag invites in.
* **No GPU kernels embedded.** `packaging/macos/check_gpu_embedded.py`
  refuses a binary with no Metal shader code, which is what a machine
  without a usable Apple GPU silently produces.

Expected local output shows `wheel:` and one
`python/dist/mojolearn-X.Y.Z-py3-none-macosx_11_0_arm64.whl`, then FIFTEEN
`PASS` lines -- five interpreters times three tiers -- and `all interpreters
passed` from the verify script. It read "ten" while there were two tiers.
Count them: a wheel that silently shipped one tier short prints ten PASS lines
too, and the difference between those two tens is the whole point of the loop.

## 4. Publish path A (the normal one)

Two terminals on the M4. Nothing here stores a token.

Terminal 1, the runner. It installs the GitHub runner under
`$HOME/.mojolearn-runner` on first use (sha256-verified against the release
notes), registers ONE ephemeral runner with labels
`self-hosted, macos, arm64, metal, real-gpu`, runs `run.sh` in the
foreground, and removes the registration when the single job finishes or
when you Ctrl-C.

```sh
tools/release_runner.sh --dry-run   # first time: installs, registers nothing
tools/release_runner.sh
```

Terminal 2, dispatch and watch. `gh workflow run` targets the default
branch unless you pass `--ref <branch-or-tag>`.

```sh
gh workflow run release-provenance.yml -f publish=testpypi
gh run watch
```

A run with `-f publish=none` builds and verifies without publishing; use it
for a rehearsal. Each dispatch needs its own `tools/release_runner.sh`
because the runner exits after one job.

Test-install from TestPyPI in a clean venv, outside the repository so the
checkout cannot shadow the installed package. The extra index lets numpy
and anything else the wheel depends on resolve from PyPI.

```sh
cd /tmp
python3.12 -m venv tp && tp/bin/pip install --no-cache-dir \
  --index-url https://test.pypi.org/simple/ \
  --extra-index-url https://pypi.org/simple/ \
  "mojolearn==X.Y.Z"
tp/bin/python /Users/andrewhendel/CascadeProjects/mojolearn/packaging/macos/smoke.py
MOJOLEARN_NUMERIC_MODE=deterministic tp/bin/python \
  /Users/andrewhendel/CascadeProjects/mojolearn/packaging/macos/smoke.py
MOJOLEARN_NUMERIC_MODE=identical tp/bin/python \
  /Users/andrewhendel/CascadeProjects/mojolearn/packaging/macos/smoke.py
```

All three invocations must succeed (fast, deterministic and identical sets).
Then repeat the whole dance for PyPI.

```sh
tools/release_runner.sh                                   # terminal 1
gh workflow run release-provenance.yml -f publish=pypi    # terminal 2
gh run watch
```

The publish job prints the sha256 of what it uploaded and the build job's
step summary prints the sha256 of what it built; they must match, and the
job fails if they do not. Compare the PyPI file hash on
https://pypi.org/project/mojolearn/#files against the same value.

## 5. Tag and GitHub release

Only after the PyPI upload is on the index. Tag the exact commit that was
built.

```sh
git tag -a vX.Y.Z -m "mojolearn X.Y.Z"
git push origin vX.Y.Z
gh release create vX.Y.Z --title "mojolearn X.Y.Z" --notes-file <notes> \
  python/dist/mojolearn-X.Y.Z-*.whl
```

Creating the GitHub release is what fires Zenodo. Zenodo's GitHub
integration is enabled for this repository and acts on `release` events
only; pushes and tags alone do nothing. It mints one DOI per release using
`/.zenodo.json` for metadata (author ORCID 0009-0000-9877-3623, also in
`CITATION.cff`). The DOI appears at
https://zenodo.org/account/settings/github/ within minutes. Put the
CONCEPT DOI (the one that resolves to the latest version) into `README.md`
and into `CITATION.cff` as a `doi:` field in a follow-up commit; the
version DOI changes every release, the concept DOI does not.

Pre-releases also mint DOIs. For an rc, tag as `vX.Y.ZrcN`, pass
`--prerelease` to `gh release create`, and only create the GitHub release
when you actually want a DOI for it. A tag with no GitHub release mints
nothing.

## 6. Rollback

PyPI never allows re-uploading a filename or a version, so a bad release
cannot be replaced in place.

1. Yank it in the web UI. Open
   https://pypi.org/manage/project/mojolearn/releases/ and use the yank
   action on that version's page. A yanked release stays downloadable by
   exact pin but is no longer chosen by default. Do the same on TestPyPI if
   it went there.
2. Fix, bump to `X.Y.Z+1` (or the next pre-release), and release again from
   section 2. The Zenodo record of the yanked release stays; say so in the
   next release notes.

## 7. Dispatch without a runner

If `gh workflow run` is issued with no runner registered, the build job
sits queued waiting for one. Start `tools/release_runner.sh` and it is
picked up; or cancel with `gh run cancel <id>`.

## 8. What is deliberately NOT done

* No API tokens in the repo, in the workflow, or in any environment
  secret. Trusted Publisher OIDC only; `id-token: write` exists on exactly
  one job.
* No push-triggered or tag-triggered publishing. `workflow_dispatch` only,
  so a release is always a decision somebody made at a terminal.
* No sdist. There is no source build without the Mojo toolchain and no way
  to make one that works; `python/setup.py` says so at the top.
* No Linux wheel yet. The source builds for Linux from the same tree, and the
  wheel's build, pack, audit and gate scripts exist under `packaging/linux/`
  (section 9), but none has run and no Linux wheel has ever been produced, so
  no Linux classifier is claimed in `python/pyproject.toml`. The classifier is
  added in the release commit that first ships one.
* No permanently registered runner. The self-hosted runner exists for one
  job and is removed.

## 9. The Linux wheel (unrun, docs/LINUX_WHEEL.md)

Every command below runs from the repository root on the M4, and every
command that touches a GPU rents a box. Nothing here runs Mojo, pixi, pytest
or a wheel build on the Mac; the only local work is a pure-Python zip and two
`docker run --cpus 2` containers on a finished wheel. Check
`bench/results/runpod_leases/` and the RunPod pod list before the NVIDIA leg
and the DigitalOcean droplet list before the AMD leg; another session may
own a box, and each provider holds one at a time.

1. Build the CUDA sets and run the on-box gates, on RunPod. Launch from a
   session that will outlive it.

   ```sh
   nohup bash packaging/linux/leg.sh nvidia > bench/results/wheels/nvidia.log 2>&1 &
   disown
   ```

   Lands in `bench/results/wheels/<stamp>-nvidia/`, with `wheels/sets/cuda/`
   (thirty `.so`, `.libs/`, `manifest.json`, `readback.txt`), `wheels/SIZES.txt`,
   `gates/smoke_{fast,deterministic,identical}.json`, `gates/sabotage.json`,
   `gates/nogpu.json` and `SUMMARY.txt`. The raw leg output is under
   `bench/results/e1/<stamp>-runpod-nvidia/` and `bench/results/e1g/`.

2. The same on the MI325X, on DigitalOcean.

   ```sh
   nohup bash packaging/linux/leg.sh amd > bench/results/wheels/amd.log 2>&1 &
   disown
   ```

   Lands in `bench/results/wheels/<stamp>-amd/` with the same files under
   `wheels/sets/hip/`.

3. Pack, on the Mac, pure Python.

   ```sh
   python3 packaging/linux/pack_wheel.py \
     --set bench/results/wheels/<stamp>-nvidia/wheels/sets/cuda \
     --set bench/results/wheels/<stamp>-amd/wheels/sets/hip \
     --check-against python/dist/mojolearn-X.Y.Z-py3-none-macosx_11_0_arm64.whl
   ```

   Writes `python/dist/mojolearn-X.Y.Z-py3-none-linux_x86_64.whl` and
   `python/dist/SIZES-X.Y.Z-linux.json`, and STOPS over 100 MB with the
   numbers printed.

4. Audit and check, in docker, one container at a time.

   ```sh
   bash packaging/linux/audit.sh python/dist/mojolearn-X.Y.Z-py3-none-linux_x86_64.whl \
     bench/results/wheels/<stamp>-nvidia/wheels/sets/cuda/manifest.json \
     bench/results/wheels/<stamp>-amd/wheels/sets/hip/manifest.json
   ```

   Writes `python/dist/audit/show.txt` (the artifact, and the measured tag),
   `repair.txt`, `repaired/mojolearn-X.Y.Z-py3-none-manylinux_<measured>_x86_64.whl`
   and `twine.txt`. The repaired wheel is the one that is uploaded.

5. Hand the repaired wheel to the release workflow. It CANNOT be built in
   CI (steps 1 and 2 rent GPU boxes), so it is staged on the runner's own
   disk, outside any checkout, with a digest sidecar:

   ```sh
   mkdir -p ~/.mojolearn-linux-wheel && rm -f ~/.mojolearn-linux-wheel/*
   cp python/dist/audit/repaired/mojolearn-X.Y.Z-py3-none-manylinux_*_x86_64.whl \
      ~/.mojolearn-linux-wheel/
   ( cd ~/.mojolearn-linux-wheel && shasum -a 256 *.whl > "$(ls *.whl).sha256" )
   ```

   The build job's "Admit a pre-built, pre-audited Linux wheel" step picks it
   up and REFUSES it unless exactly one wheel is staged, the sidecar matches
   its bytes, the version equals the one just built, the tag is a manylinux
   tag (`linux_x86_64` means `audit.sh` never ran) and `twine check` passes.
   With nothing staged the release is macOS-only, which is what every release
   before 0.3.0 was. `MOJOLEARN_LINUX_WHEEL_DIR` in the runner's environment
   overrides the location.

   The publish job then checks the artifact against a `sha256  filename`
   manifest the build job emitted, in BOTH directions: every named wheel is
   present and hashes correctly, and no wheel is present that was not named.
   `bash packaging/release_workflow_test.sh` runs all three of those shell
   blocks on the Mac against fabricated wheels, 15 cases, no network; run it
   after editing any of them.

6. Install-smoke on each vendor, from TestPyPI, in a clean venv on the box.

   ```sh
   MOJOLEARN_WHEEL_VERSION=X.Y.Z nohup bash packaging/linux/leg.sh nvidia install > bench/results/wheels/nvidia-install.log 2>&1 &
   disown
   MOJOLEARN_WHEEL_VERSION=X.Y.Z nohup bash packaging/linux/leg.sh amd install > bench/results/wheels/amd-install.log 2>&1 &
   disown
   ```

   Lands in `bench/results/wheels/<stamp>-<vendor>/` with `install.log`,
   `pip_show.txt`, `gates/smoke_<tier>.json` and `SUMMARY.txt`. Rent the
   NVIDIA install leg on a DIFFERENT GPU model than the build leg
   (`MOJOLEARN_GEMM_LEG_GPU_NVIDIA`); it is the only evidence that a set built
   on one architecture runs on another.

7. Then PyPI, by section 4, with both wheels in `python/dist/`.
