# Release checklist

**Alpha Python/reference patch:** use the [bounded patch path](lanes/RELEASE_PROCESS_ALPHA.md#pythonreference-patches-with-unchanged-native-inputs). Reuse unchanged native binaries, check only affected numerical references, and smoke each exact final wheel. The broader native-build and certification steps below do not apply to every such patch.

Five steps, one finish line: the file is on PyPI and installs. The longer
runbooks (`docs/PYPI_RELEASE.md`, `docs/RELEASE_0_6_1_EXECUTION_PLAN.md`,
`packaging/ALPHA_LINUX_061_PUBLICATION.md`, `docs/RELEASE_0_6_1_RUNPOD_PROFILE7.md`)
are background for when a step refuses. They are not a longer version of
this list.

**Policy (2026-09-09).** Build what we ship, retag it, publish it. Install
the actual wheel and test it on real GPUs only when numerics changed on a
path users call, or when the release is cited as paper evidence; the archive
is then attached to the release. No deviation numbers, admission essays or
evidence READMEs. Plain commit messages and the files the tools write.

Measured on 2026-09-10 (0.8.0, serial compiles): one architecture build was
12 minutes of compile on the MI325X and 19 on the H100 plus about 5 minutes
of rental spin-up, and the three run in parallel; pack, audit and upload are
about 15 minutes; one install-and-test column is about four minutes. The
macOS run on the release Mac was 20 minutes, 8 of them compiling.

Since then the extension builds run four at a time on every builder
(MOJOLEARN_BUILD_JOBS, default 4, each build still capped at two compiler
workers and one BLAS thread, the box affinity at 2 x jobs cores). Set
MOJOLEARN_BUILD_JOBS=1 to reproduce a serial build. Times for the parallel
path are OWED from the next release; record them here when it ships.

## 1. Freeze

`CITATION.cff` version and release date are generated; do not update them by hand.
The date comes from the matching `published YYYY-MM-DD` changelog heading,
not the build clock. An `unreleased YYYY-MM-DD` entry omits the citation date.
If publication moves to another day, update the changelog and regenerate before
freezing the release artifacts. CI rejects stale citation metadata.

Bump `python/mojolearn/_version.py` and `python/pyproject.toml`, add the
CHANGELOG entry with the publication date in UTC, then

```sh
pixi run write-docs-facts   # generates CITATION version/date and marked doc spans
pixi run check-docs-facts   # fails if a pin, a badge or a prose default drifted
```

```sh
python3 packaging/check_ext_lists.py   # every pack/build/smoke list agrees with _backend
python3 packaging/check_ext_lists.py --host   # the host list is read from the manifest everywhere
```

```sh
MOJOLEARN_NUMERIC_MODE=identical pixi run -e test test-python   # 978 pytest tests, about 7 s
pixi run check-python-gates-release                              # explicit bounded release gates
pixi run check-mamba-poison                                      # DEVIATIONS 2712/2713: the four Mamba lanes cold on a NaN-poisoned, guard-banded binding, about 3 min
```

Both need the 17 IDENTICAL bindings built on this Mac (`bindings/build*.sh`).
They run here and not on a hosted runner because a GitHub macOS VM cannot
compile Metal AOT (`.github/workflows/python-tests.yml` header); that workflow
is dispatch-only on the same ephemeral runner as step 6.

The wheel's contents are one rule: the three tree bindings in every tier,
every other binding in `identical/` alone (DEVIATION 2490). The checker holds
the pack and build lists to `_backend._TIERED` and `_MODULES`; a binding that
drifted back into three tiers, or out of the wheel, goes red here.

From 0.8.7 the wheel also carries every host binding
`python/mojolearn/host_surface.py` declares (every family it lists, under
`mojolearn/host/`; ask `--wheel-families` rather than quoting a count, which
has read ten, then fifteen, then sixteen, then thirty-two),
`mojolearn/reference_cards/`, a copy of `tools/identity_trace_diff.py` and
`tools/identity_break.py`, and the three training GPU columns the manifest
names under `mojolearn/identity_columns/<record>/` with a COMMIT witness. The
host list lives in the manifest alone; `--host` above fails any builder,
packer, smoke or admission file that spells one of its own. The columns the
wheel ships are `TRAINING_GPU_COLUMNS` in the manifest, so a new record is a
manifest edit, never a packaging edit.

`_version.py` is the source of truth and the checker holds `pyproject.toml`,
`CITATION.cff` and the newest published CHANGELOG heading to it, so a
half-finished bump goes red here rather than shipping in a README. Commit,
push. Every build below runs from that commit.

## 2. Build the three Linux sets (parallel legs, parallel compiles inside each)

```sh
REF=<40-hex commit>
# AMD gfx942 on a DigitalOcean MI325X. The output directory
# ~/mojolearn-evidence/releases/$REF/hip-gfx942/release-build must not pre-exist.
bash tools/do_release061_leg.sh $REF ~/.mojolearn_do_token --rent
# NVIDIA sm_90a (RunPod H100) and sm_89 (RunPod L40S), started 90 s apart
MOJOLEARN_RUNPOD_KEY_FILE=~/.mojolearn_runpod_key MOJOLEARN_NVIDIA_CAMPAIGN=7 MOJOLEARN_GPU_ARCHS=sm_90a \
  sh tools/gemm_remote_leg.sh nvidia --payload mamba --source-ref $REF --gpu "NVIDIA H100 80GB HBM3" --allow-concurrent --rent --minutes 60
MOJOLEARN_RUNPOD_KEY_FILE=~/.mojolearn_runpod_key MOJOLEARN_NVIDIA_CAMPAIGN=7 MOJOLEARN_GPU_ARCHS=sm_89 \
  sh tools/gemm_remote_leg.sh nvidia --payload mamba --source-ref $REF --gpu "NVIDIA L40S" --allow-concurrent --rent --minutes 60
```

Launch each with `nohup ... &` from a shell that outlives it. Proofs land at
`~/mojolearn-evidence/releases/<source-commit>/hip-gfx942/release-build/` and
`~/mojolearn-evidence/e1g/<stamp>-nvidia-mamba/remote/release-build/`; each
`build/build-provenance.json` names its commit and architecture.
Set `MOJOLEARN_EVIDENCE_ROOT` to use another storage root. Existing explicit
`MOJOLEARN_RELEASE_RESULTS_ROOT` and `MOJOLEARN_GEMM_LEG_OUT` overrides take
precedence. Keep raw output there; commit only the reviewed summary and
provenance needed by the release, with links and hashes for external artifacts.

## 2b. The host bindings, per vendor (the check the CPU gates cannot make)

The host bindings are vendor-neutral and built ON EACH LEG, pinned to
`MOJOLEARN_TARGET_COLUMN=cpu` with no accelerator target. The 0.8.5 freeze
caught two regressions here that no CPU-only workflow can see, a host build
that took the leg's GPU column and refused, and a detected-column read-back
that folded the build box's GPU name into a vendor-neutral binary so the
NVIDIA and AMD copies differed by 43 bytes. On every leg, before packing:

```sh
# every host binding the manifest ships is in the set, read back as cpu, with no device code
python3 python/mojolearn/host_surface.py --wheel-bindings
grep '^host ' <leg>/build/sets/<vendor>/<arch>/readback.txt        # one row per binding, third field cpu
grep '^host ' <leg>/build/sets/<vendor>/<arch>/arch_readback.txt   # one row per binding, NONE-BY-DESIGN
grep '"tier": "host"' <leg>/build/sets/<vendor>/<arch>/manifest.json   # staged with a RUNPATH, in the closure
# the byte compare across the three legs: one digest per binding, the same on sm_89, sm_90a and gfx942
sha256sum <sm89>/build/sets/cuda/sm_89/host/*.so <sm90a>/build/sets/cuda/sm_90a/host/*.so <hip>/build/sets/hip/gfx942/host/*.so | sort
```

`pack_wheel.py` (step 3) refuses the wheel when any binding's digests
differ across legs, and requires every manifest binding in every set; the
lines above say WHICH leg is wrong before the packer says that one is.

## 3. Pack, audit, strip (on the Mac, docker, about 10 minutes)

```sh
python3 packaging/linux/pack_wheel.py --profile release-linux3 \
  --set <sm89>/build/sets/cuda --set <sm90a>/build/sets/cuda --set <hip>/build/sets/hip \
  --build-proof <sm89>/build/build-provenance.json --build-proof <sm90a>/build/build-provenance.json \
  --build-proof <hip>/build/build-provenance.json --out <dist>
bash packaging/linux/audit.sh <dist>/mojolearn-*-linux_x86_64.whl \
  <sm89>/build/sets/cuda/sm_89/manifest.json <sm90a>/build/sets/cuda/sm_90a/manifest.json <hip>/build/sets/hip/gfx942/manifest.json
python3 tools/strip_wheel_dir_entries.py <dist>/audit/repaired/mojolearn-*-manylinux*.whl <dist>/final/<same name> \
  --receipt <dist>/final/dir-entry-strip.json
```

## 4. Install and test on real GPUs (OPTIONAL, never required for a release)

A release is verified by the CPU column and the Apple pass (section 5b). This
section is a diagnostic for a suspected NVIDIA or AMD problem and the release
workflow admits a Linux wheel without it. Copy the three `build-provenance.json` files to a proofs directory
as `cuda-sm_89.json`, `cuda-sm_90a.json`, `hip-gfx942.json`, then:

```sh
export MOJOLEARN_LEG_MODE=qualify MOJOLEARN_QUALIFY_WHEEL=<dist>/final/<wheel> MOJOLEARN_QUALIFY_PROOFS=<proofs>
bash tools/do_release061_leg.sh $REF ~/.mojolearn_do_token --rent                        # gfx942
MOJOLEARN_LEG_GPU=h100 bash tools/do_release061_leg.sh $REF ~/.mojolearn_do_token --rent  # sm_90a
MOJOLEARN_RUNPOD_KEY_FILE=~/.mojolearn_runpod_key MOJOLEARN_NVIDIA_CAMPAIGN=7 MOJOLEARN_GPU_ARCHS=sm_89 \
  sh tools/gemm_remote_leg.sh nvidia --payload mamba --source-ref $REF --gpu "NVIDIA L40S" --allow-concurrent --rent --minutes 60
```

A failing job is a failing release: fix forward, back to step 2.

Each qualification's `*.installed.json` now carries `installed_host_bindings`
(one row per manifest binding, read back through the package's own host
loader, digest equal to the wheel member); `verify_linux_surface_qualification.py`
refuses a record that lacks one. On each vendor's box, from the installed
wheel, the identity command must also pass against the shipped columns:

```sh
python -m venv /tmp/q && /tmp/q/bin/pip install <dist>/final/<wheel> numpy
cd /tmp && MOJOLEARN_NUMERIC_MODE=identical /tmp/q/bin/python -m mojolearn identity --check   # exit 0: harness, 3 columns, witness
cd /tmp && MOJOLEARN_NUMERIC_MODE=identical /tmp/q/bin/python -m mojolearn identity --keep /tmp/q/local.json   # exit 0: IDENTICAL x4 on every cell
```

The second command runs the whole record (47 lanes, 9 fixtures, 2 fits per
cell); keep `local.json` with the leg. On the release Mac the same two
commands run from the venv `verify_wheel.sh` leaves, and `verify_wheel.sh`
itself already loads every host binding and runs `identity --check`.

## 5. Publish Linux (one command)

```sh
bash tools/release_linux_publish.sh <dist>/final/<wheel> alpha-api-<version>-<yyyymmdd> pypi <workdir>
```

Manifest, GitHub release, Trusted Publisher dispatch, watch. If step 4 ran,
pass its three output directories and the proofs directory through the
`MOJOLEARN_QUAL_*` variables named in the script header and the archive is
attached and checked. Use `none` first to see the workflow's own checks
without uploading.

## 5b. Release verification: CPU and the Apple GPU, nothing else

A release is verified by two things (2026-09-19):

```sh
pixi run -e test release-check            # both, on this Mac, nothing rented
# which is:
pixi run -e test cpu-pass                 # the CPU route, 5 local slots, same cells as the Apple pass
pixi run -e test apple-pass               # the Apple GPU: every Metal lane, fitted once, end model, 600 s
```

Both run LOCALLY. No pod, no NVIDIA or AMD box and no R2 staging is part of a
release. Each pass is `base,denormal,odd`, fitted once, end model only.

**Only the lanes the release touched are required, and that is automatic.**
With no selection given, a pass checks the lanes whose sources changed since
the newest `v*` tag (`git describe`). When a changed path cannot be attributed
to lanes (a build script, a shared kernel) the selector widens to every lane by
itself, so it can run too much and never too little. `--all` forces everything.

**It records as it goes and resumes.** Records live in
`~/mojolearn-evidence/release-check/<commit>/<backend>/`, one checkpoint per
finished cell. If a pass is interrupted, killed or runs out of its one-hour
hang guard, run the same command again at the same commit: it prints
`resuming` and fits only what is missing. Measured 2026-09-19 on the M4 CPU
route: 15 lanes x 3 fixtures in 14 s; killed at 6 s with 13 cells on disk,
the rerun completed all 45.

## 6. Publish macOS (on the release Mac, 30 to 60 minutes)

```sh
git tag -a v<version> -m "mojolearn <version>" <commit> && git push origin refs/tags/v<version>
nohup sh tools/release_runner.sh > runner.log 2>&1 &          # one ephemeral runner, one job
gh workflow run release-provenance.yml --ref v<version> -f validation_profile=full -f publish=pypi
gh run watch <run id> --exit-status
```

Leave `~/.mojolearn-linux-wheel` empty: the Linux wheel went out in step 5
and PyPI refuses a second upload of the same filename.

## Finish line

```sh
python3 -m venv /tmp/v && /tmp/v/bin/pip install mojolearn==<version> \
  && /tmp/v/bin/python -c "import mojolearn; print(mojolearn.__version__)"
```

Run it on a Linux GPU box and on a Mac. Both wheels are listed at
https://pypi.org/project/mojolearn/#files.
