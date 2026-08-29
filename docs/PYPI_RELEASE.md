# Releasing `mojolearn` to PyPI

The runbook for `.github/workflows/release-provenance.yml` and
`tools/release_runner.sh`. Terse on purpose. Every command is meant to be
pasted as written, with `X.Y.Z` replaced.

## 1. What a release is here

* One wheel, project name `mojolearn`, macOS arm64 only, tagged `py3` and
  `macosx_11_0_arm64`. One artifact serves python 3.10 through 3.14 because
  the extensions link no libpython (see `python/setup.py`).
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
  wheel is published. No Linux wheel, no sdist (section 8).
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
  `NOTICE`, and every Mojo source directory the five extensions compile.
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
MOJOLEARN_NUMERIC_MODE=identical tp/bin/python \
  /Users/andrewhendel/CascadeProjects/mojolearn/packaging/macos/smoke.py
```

Both invocations must succeed (FAST and IDENTICAL sets). Then repeat the
whole dance for PyPI.

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
* No Linux wheel yet. The source builds for Linux from the same tree, but no
  Linux wheel has ever been produced, so no Linux classifier is claimed in
  `python/pyproject.toml`.
* No permanently registered runner. The self-hosted runner exists for one
  job and is removed.
