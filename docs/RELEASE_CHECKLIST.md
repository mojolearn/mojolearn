# Release checklist

## The one command

```sh
pixi run release <version> --dry-run          # the plan, every command, and what is already done
pixi run release <version>                    # everything up to publication
pixi run release <version> --publish pypi     # ... then publish both wheels, finish line, record
```

`tools/release.py` walks this whole checklist. Write the CHANGELOG entry
(`## <version> (published YYYY-MM-DD)`, UTC) first; the command bumps the two
version files, runs `write-docs-facts`, commits and pushes exactly those files,
freezes HEAD, and then runs: the rehearsal (step 0); the three Linux build legs
in parallel, detached (step 2, the AMD leg no longer waits for NVIDIA); the
macOS wheel with the byte LM and host bindings on, four slots; the macOS smoke
under the Metal lock; `release-check` (step 5b, the guarantee); a wait that
checks every leg's proof for the frozen commit and names any host binding whose
bytes differ across legs; pack, audit and strip under the `pkg` environment
(step 3); the Linux smoke on one rented RTX 4090 (`tools/release_wheel_smoke.sh`);
and, only with `--publish none|testpypi|pypi`, both publishes (step 5), the
`pip install` finish line on this Mac and the
`bench/results/release_verification/<date>_pypi_<v>/` record, committed and
pushed.

It is resumable: rerun the same command after any stop. State lives in
`~/mojolearn-evidence/release/<version>/state.json`, and everything after the
freeze in `<version>/<commit12>/` (legs, wheels, smoke receipts, logs). Each
step checks its own output (a wheel of the frozen commit, a PASSED receipt for
that wheel's sha256, complete release-check records, the file already on PyPI)
and skips when it is there. A running build leg is never relaunched; a failed
one is moved aside, never deleted, and relaunched. A new commit after the freeze
refreezes and starts the post-freeze steps in a fresh directory, unless a wheel
of the frozen commit is already published. `--only STEP[,STEP]` runs some steps,
`--redo STEP` discards a step's record, `--amd-expect-from <NVIDIA release-build
dir>` restores the AMD core-host probe, `--smoke-gpu` picks the Linux smoke GPU.
The Linux build route is one function (`launch_linux_builds`, routes in
`BUILD_BACKENDS`), so the CPU build box route plugs in as `--build-backend`.

The numbered steps below are the same work by hand: the fallback when a step
refuses and needs a person, and the reference for what each step runs.

**Alpha Python/reference patch:** use the [bounded patch path](lanes/RELEASE_PROCESS_ALPHA.md#pythonreference-patches-with-unchanged-native-inputs). Reuse unchanged native binaries, check only affected numerical references, and smoke each exact final wheel. The broader native-build and certification steps below do not apply to every such patch.

## 0. Rehearse (by hand, before anything is rented)

```sh
pixi run release-rehearsal
```

A pre-release rehearsal the releaser runs by hand. It is not a per-merge check
and not a CI job. It runs locally in minutes, rents nothing and compiles
nothing, and prints one PASS or FAIL line per step with a non-zero exit when
any step failed: the wheel audit's tests, the Python suite in IDENTICAL mode
(the one step that may touch Metal, so it waits for the Mac's Metal slot),
the docs facts, both extension-list checks, the wheel's NumPy and platform-math
audit over the package staged as the wheel ships it (with the
`_identity_break.py` and `_identity_trace_diff.py` copies), and the dry runs of
the NVIDIA and AMD release legs. Each of these failed 0.8.12 after boxes were
rented or after a 16-minute macOS compile. `--list` prints the steps; `--only`
reruns some of them.

Five steps, one finish line: the file is on PyPI and installs. The longer
runbook (`docs/PYPI_RELEASE.md`) is background for when a step refuses. They are not a longer version of
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

Since then the Linux extension builds run four at a time (MOJOLEARN_BUILD_JOBS,
default 4, each build capped at two compiler workers and one BLAS thread, the
box affinity at 2 x jobs cores). On the Mac the default is FOUR builds of one
compiler worker, heaviest first (2026-09-21), inside the Apple release budget
of five cores and about 5 GB. The 0.8.13 cold build ran one at a time (2666 s,
2371 s of it compiling); its measured per-extension times put four single
worker builds at about 15 minutes cold. Under `tools/mac_slot.py` pass
`--slots 4`, or mac_slot pins one build at a time:
`python3 tools/mac_slot.py --slots 4 run -- ./packaging/macos/build_release_wheel.sh`.
Unchanged bindings come from a local compile cache (step 6). Set
MOJOLEARN_BUILD_JOBS=1 to reproduce a serial build.

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
pixi run test-wheel-audit                                        # the wheel ships nothing but Python and the Mojo/MAX runtime: no third-party imports, no NumPy, no platform math
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
# AMD gfx942 on a DigitalOcean MI325X, built in the pinned Ubuntu 22.04 container.
# The droplet image is Ubuntu 24.04 (GCC 13); built on the host, every host binding
# differs from the NVIDIA legs' (Ubuntu 22.04, GCC 11) and step 3 refuses the wheel.
# It builds MOJOLEARN_BUILD_JOBS extensions at a time like the NVIDIA legs
# (default 4, 2 x jobs cores, 16 GiB per job) and starts IN PARALLEL with them:
# the core-host probe is skipped by default, because pack_wheel.py (step 3)
# refuses any host binding whose bytes differ across the legs.
# The output directory ~/mojolearn-evidence/releases/$REF/hip-gfx942/release-build
# must not pre-exist.
MOJOLEARN_RELEASE_UBUNTU22=1 bash tools/do_release061_leg.sh $REF ~/.mojolearn_do_token --rent
# Optional, to fail two minutes in rather than at pack time when an NVIDIA leg
# of $REF has already finished: derive the probe's digest from its STAGED set copy
# (<leg>/remote/release-build/build/sets/cuda/<arch>/host/_mojolearn_core_host.so,
# after patchelf). Never hash python/mojolearn/host/ on the pod: that unstaged
# copy differs, and typing it cost 0.8.14 an MI325X rental.
MOJOLEARN_RELEASE_UBUNTU22=1 bash tools/do_release061_leg.sh $REF ~/.mojolearn_do_token --rent \
  --expect-from ~/mojolearn-evidence/e1g/<stamp>-nvidia-mamba
# NVIDIA sm_90a (RunPod H100) and sm_89 (RunPod L40S), started 90 s apart
MOJOLEARN_RUNPOD_KEY_FILE=~/.mojolearn_runpod_key MOJOLEARN_NVIDIA_CAMPAIGN=7 MOJOLEARN_GPU_ARCHS=sm_90a \
  sh tools/gemm_remote_leg.sh nvidia --payload mamba --source-ref $REF --gpu "NVIDIA H100 80GB HBM3" --allow-concurrent --rent --minutes 60
MOJOLEARN_RUNPOD_KEY_FILE=~/.mojolearn_runpod_key MOJOLEARN_NVIDIA_CAMPAIGN=7 MOJOLEARN_GPU_ARCHS=sm_89 \
  sh tools/gemm_remote_leg.sh nvidia --payload mamba --source-ref $REF --gpu "NVIDIA L40S" --allow-concurrent --rent --minutes 60
```

The legs' pre-flights and the packer judge TRACKED source only: an ignored or
untracked file (a generated table, local test output) cannot refuse a launch
or a pack, while an uncommitted edit to a tracked file still does. The build
inventory leaves out `python/mojolearn/tests/`, which never ships, so a
test-only commit after the Linux builds does not force a rebuild; the packer
still refuses any shipped `.py` that differs from the build commit.

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

## 2c. The CPU build route: all three sets on one CPU box, no GPU

```sh
bash tools/release_linux_build.sh $REF          # dry run: nothing rented
bash tools/release_linux_build.sh $REF --rent   # about 17 minutes, about $0.40
```

One RunPod CPU pod (32 vCPU, 128 GB) runs the AMD leg's pinned
`rocm/dev-ubuntu-22.04` image (GCC 11.4, ld 2.38, patchelf 0.17.2) and builds
cuda/sm_90a, cuda/sm_89 and hip/gfx942 one after another at `/root/mojolearn`
through the same `tools/release061_remote_build.sh`, with
`MOJOLEARN_RELEASE_NO_DEVICE=1`: Mojo compiles each set from
`--target-accelerator` alone, the guard (`tools/cpu_build_guard.py`) refuses if
any GPU device node is visible, and the postflight requires the read-back
architecture to equal the requested one. The output is
`~/mojolearn-evidence/releases/$REF/linux-cpu-box/<stamp>/{cuda-sm_90a,cuda-sm_89,hip-gfx942}/release-build`,
the same trees the GPU legs write; the script prints the step 3 pack command.
It replaces the three GPU rentals of section 2 (20 to 35 minutes each).

Proof at d181d9792 (0.8.14) against that release's three GPU-box builds,
sha256 of every `.so` in each set (tiers, `host/`, `.libs/`):
cuda/sm_90a 66 of 66 and cuda/sm_89 66 of 66 byte-identical, with
`readback.txt`, `arch_readback.txt` and the provenance extension and
host-binding digests identical. hip/gfx942 matched 62 of 66: the Mojo
compiler's gfx942 output is not reproducible run to run on any box (with a
cold Mojo cache, `build_tsa.sh` gave 2 binaries in 12 builds with `-j 1`,
`build_mixture.sh` 3 in 5 with `-j 2`; a warm cache hides it), so the MI325X
leg's own bytes were one draw. `packaging/linux/build_sets.sh` compiles AMD
GPU bindings with one worker, which narrows it, and the binding cache below is
what makes a released AMD binary reproducible. Two cold CPU-box builds of
4756f57a9 agreed on 197 of 198 binaries (the one: `identical/_mojolearn_tsa.so`
on gfx942). `manifest.json` differs only in its `set` field, the
staging path, which nothing reads.

The binding cache (`tools/bincache.py`, R2 `bincache/v1/none/<image>/`) is on
by default: a build whose key (source closure, scripts, toolchain, build
variables, OS, path) matches an archive takes its bytes after the archive's
key, fields and every file's sha256 verify. `build-provenance.json` records per
binary whether it was built or taken, its key, and the commit whose build
compiled it. `--no-bincache` builds everything from source.

## 3. Pack, audit, strip (on the Mac, docker, about 10 minutes)

The packer needs `lief`, which only the `pkg` environment has; under the system
`python3` it now re-executes itself there (or refuses up front without pixi).
Pixi tasks run at the repository root, so give absolute paths.

```sh
pixi run -e pkg pack-linux-wheel --profile release-linux3 \
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

For bounded alpha publication, first run the existing expanded smoke on the
**exact final wheel**, on a CUDA host for Linux or on this Mac for macOS.

Linux: one rented RTX 4090 (sm_89, which the wheel carries), nothing shipped but
the wheel and `tools/qualify_verifier_wheel.py`, every transfer bounded, the pod
deleted and verified gone on every exit. Without `--rent` it is a dry run;
`--ssh '<target>'` runs it on a CUDA box you already have up.

```sh
bash tools/release_wheel_smoke.sh <dist>/final/<wheel> --expected-source-commit <40-hex commit> \
  --out <fresh-smoke-dir> --rent                      # writes <fresh-smoke-dir>/results.json
```

macOS: build the wheel (the release profile, byte LM and every host binding,
is the default of `build_release_wheel.sh` since 0.8.14; `MOJOLEARN_PACKAGE_BYTE_LM=0`
is an explicit opt-out that no release uses), then smoke it under the Metal lock:

```sh
python3 tools/mac_slot.py --slots 4 run -- ./packaging/macos/build_release_wheel.sh   # python/dist/*.whl
python3 tools/mac_slot.py metal -- python3 tools/qualify_verifier_wheel.py python/dist/<wheel> \
  --scope expanded --expected-source-commit <40-hex commit> --output <fresh-smoke-dir>
```

Then publish each wheel with its own receipt:

```sh
bash tools/release_linux_publish.sh <final-wheel> \
  alpha-api-<version>-<linux-or-macos>-<yyyymmdd> pypi <workdir> \
  --light-smoke <fresh-smoke-dir>/results.json
```

Run the publisher from the frozen source checkout. The light route is the
default: despite its historical name, the helper takes either platform with
`--light-smoke`, hashes and stages the receipt, checks the light admission
rules, and dispatches the matching platform batch. Publish the two wheels
independently with separate tags and work directories. The full native Linux
certification route is opt-in with `--full`; with neither flag the helper
refuses. No smoke receipt from an older wheel can be reused.

Manifest, GitHub release, Trusted Publisher dispatch, watch. If step 4 ran,
pass its three output directories and the proofs directory through the
`MOJOLEARN_QUAL_*` variables named in the script header and the archive is
attached and checked. Use `none` first to see the workflow's own checks
without uploading.

## 5b. Release verification: CPU and the Apple GPU, nothing else

A release is verified by two things (2026-09-19):

```sh
pixi run -e test release-check            # both AT ONCE, on this Mac, nothing rented
# which is, concurrently (tools/release_check.py):
pixi run -e test apple-pass               # the Apple GPU: Metal lock + one of the 5 Mac slots
MOJOLEARN_CPU_PASS_SLOTS=4 pixi run -e test cpu-pass   # the CPU route on the other 4 slots
```

Both run LOCALLY. No pod, no NVIDIA or AMD box and no R2 staging is part of a
release. Each pass is `base,denormal,odd`, fitted once, end model only.

**Only the lanes the release touched are required, and that is automatic.**
With no selection given, a pass checks the lanes whose sources changed since
the last pass on the same backend that FINISHED on this Mac (complete, same
fixtures, one fit, a clean tree, and itself anchored on a full pass, a tag or
another such pass). With no such record it falls back to the newest `v*` tag,
as before; 0.8.9 to 0.8.11 were cut under `alpha-api-*` tags, which is why the
0.8.12 pass diffed 2,328 paths against v0.8.8. When a changed path cannot be
attributed to lanes the selector widens to every lane by itself, so it can run
too much and never too little. `--all` forces everything. Build scripts, tools
nothing runs, check programs nothing imports, the verifier's own modules and
pixi task edits are attributed now (tools/lane_select.py, 2026-09-21); what
still widens to every lane is a change to `tools/identity_break.py` outside a
lane body, a whole-surface registry, `pixi.lock`, `__main__.py` and anything
the map genuinely cannot place.

**Sharded Metal is opt-in and unproven.** `pixi run -e test apple-pass
--metal-shards N` (N = 2 or 3) splits the lanes over N processes that share the
one GPU under a single Metal slot, to overlap their host-to-device waits.
Concurrent Metal jobs returned NaN and zero outputs on 2026-09-15, so N stays 1
until this has been run once, after a release, on a quiet Mac:

```sh
C=$(git rev-parse --short=12 HEAD); E=~/mojolearn-evidence/metal-shards; mkdir -p $E
for n in 1 2 3; do
  MOJOLEARN_RELEASE_CHECK_DIR=$E/n$n /usr/bin/time -p pixi run -e test \
    python3 tools/verify_lanes.py --apple-pass --all --metal-shards $n 2>&1 | tee $E/n$n.log
done
# every part of every cell, sharded against unsharded, then against the CPU
# column of the same commit (the reference the Apple column must equal; make
# it full first with `pixi run -e test cpu-pass --all` at this commit):
pixi run -e test python3 tools/identity_break.py --diff $E/n1/$C/metal/column.json $E/n2/$C/metal/column.json
pixi run -e test python3 tools/identity_break.py --diff $E/n1/$C/metal/column.json $E/n3/$C/metal/column.json
pixi run -e test python3 tools/identity_break.py --diff \
  ~/mojolearn-evidence/release-check/$C/cpu/column.json $E/n3/$C/metal/column.json
```

Only if all three diffs read zero differences over the full sweep may the
default change, and a sharded record never anchors the next pass (the
`metal_shards` field in its manifest). Record the three wall times here.

**It records as it goes and resumes.** Records live in
`~/mojolearn-evidence/release-check/<commit>/<backend>/`, one checkpoint per
finished cell. If a pass is interrupted, killed or runs out of its one-hour
hang guard, run the same command again at the same commit: it prints
`resuming` and fits only what is missing. Measured 2026-09-19 on the M4 CPU
route: 15 lanes x 3 fixtures in 14 s; killed at 6 s with 13 cells on disk,
the rerun completed all 45.

## 6. Publish macOS through the full workflow route (not the light route)

The light route above builds and smokes the macOS wheel on this Mac and
publishes it with `release_linux_publish.sh --light-smoke`; `pixi run release`
does that. This section is the full-certification alternative, where the
workflow builds the wheel on an ephemeral runner.

The build job compiles four extensions at a time with one compiler worker
each, and reuses from `~/.mojolearn-bincache/macos-release` every binding whose
inputs are unchanged since the last workflow run (tools/bincache.py, "a LOCAL
directory cache"; the log ends with how many were reused). A rerun of a failed
dispatch therefore compiles nothing it compiled before. It was 30 to 60 minutes
serial and uncached.

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
