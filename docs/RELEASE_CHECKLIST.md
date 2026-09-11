# Release checklist

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

Bump `python/mojolearn/_version.py` and `python/pyproject.toml`, add the
CHANGELOG entry, then

```sh
pixi run write-docs-facts   # rewrites the marked spans in README.md
pixi run check-docs-facts   # fails if a pin, a badge or a prose default drifted
```

```sh
python3 packaging/check_ext_lists.py   # every pack/build/smoke list agrees with _backend
```

The wheel's contents are one rule: the three tree bindings in every tier,
every other binding in `identical/` alone (DEVIATION 2490). The checker holds
the pack and build lists to `_backend._TIERED` and `_MODULES`; a binding that
drifted back into three tiers, or out of the wheel, goes red here.

`_version.py` is the source of truth and the checker holds `pyproject.toml`,
`CITATION.cff` and the newest published CHANGELOG heading to it, so a
half-finished bump goes red here rather than shipping in a README. Commit,
push. Every build below runs from that commit.

## 2. Build the three Linux sets (parallel legs, parallel compiles inside each)

```sh
REF=<40-hex commit>
# AMD gfx942 on a DigitalOcean MI325X. The output directory
# bench/results/releases/<release>/hip-gfx942 must not pre-exist; move a stale one aside.
bash tools/do_release061_leg.sh $REF ~/.mojolearn_do_token --rent
# NVIDIA sm_90a (RunPod H100) and sm_89 (RunPod L40S), started 90 s apart
MOJOLEARN_RUNPOD_KEY_FILE=~/.mojolearn_runpod_key MOJOLEARN_NVIDIA_CAMPAIGN=7 MOJOLEARN_GPU_ARCHS=sm_90a \
  sh tools/gemm_remote_leg.sh nvidia --payload mamba --source-ref $REF --gpu "NVIDIA H100 80GB HBM3" --allow-concurrent --rent --minutes 60
MOJOLEARN_RUNPOD_KEY_FILE=~/.mojolearn_runpod_key MOJOLEARN_NVIDIA_CAMPAIGN=7 MOJOLEARN_GPU_ARCHS=sm_89 \
  sh tools/gemm_remote_leg.sh nvidia --payload mamba --source-ref $REF --gpu "NVIDIA L40S" --allow-concurrent --rent --minutes 60
```

Launch each with `nohup ... &` from a shell that outlives it. Proofs land at
`bench/results/releases/<release>/hip-gfx942/release-build/` and
`bench/results/e1g/<stamp>-nvidia-mamba/remote/release-build/`; each
`build/build-provenance.json` names its commit and architecture.

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

## 4. Install and test on real GPUs (conditional, about 5 minutes each, parallel)

Only when numerics changed on a user-facing path or the release is paper
evidence. Copy the three `build-provenance.json` files to a proofs directory
as `cuda-sm_89.json`, `cuda-sm_90a.json`, `hip-gfx942.json`, then:

```sh
export MOJOLEARN_LEG_MODE=qualify MOJOLEARN_QUALIFY_WHEEL=<dist>/final/<wheel> MOJOLEARN_QUALIFY_PROOFS=<proofs>
bash tools/do_release061_leg.sh $REF ~/.mojolearn_do_token --rent                        # gfx942
MOJOLEARN_LEG_GPU=h100 bash tools/do_release061_leg.sh $REF ~/.mojolearn_do_token --rent  # sm_90a
MOJOLEARN_RUNPOD_KEY_FILE=~/.mojolearn_runpod_key MOJOLEARN_NVIDIA_CAMPAIGN=7 MOJOLEARN_GPU_ARCHS=sm_89 \
  sh tools/gemm_remote_leg.sh nvidia --payload mamba --source-ref $REF --gpu "NVIDIA L40S" --allow-concurrent --rent --minutes 60
```

A failing job is a failing release: fix forward, back to step 2.

## 5. Publish Linux (one command)

```sh
bash tools/release_linux_publish.sh <dist>/final/<wheel> alpha-api-<version>-<yyyymmdd> pypi <workdir>
```

Manifest, GitHub release, Trusted Publisher dispatch, watch. If step 4 ran,
pass its three output directories and the proofs directory through the
`MOJOLEARN_QUAL_*` variables named in the script header and the archive is
attached and checked. Use `none` first to see the workflow's own checks
without uploading.

## 6. Publish macOS (on the release Mac, 30 to 60 minutes)

```sh
git tag -a v<version> -m "mojolearn <version>" <commit> && git push origin refs/tags/v<version>
nohup sh tools/release_runner.sh > runner.log 2>&1 &          # one ephemeral runner, one job
gh workflow run release-provenance.yml --ref v<version> -f publish=pypi
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
