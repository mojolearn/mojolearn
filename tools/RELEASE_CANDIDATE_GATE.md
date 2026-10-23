# Qualify candidate wheels before publication

`release.py` now gates the actual files emitted by `linux-pack` before any
publication. Both NVIDIA and AMD gates must pass, including the existing rule
that every judged opponent pair is `MATCHED`. The core and vendor wheels are
local inputs; version 0.8.56 need not exist on PyPI. NVIDIA also supplies its PTX
wheel. Third-party dependencies still install normally.

The candidate installer checks the complete distribution set, exact metadata
version, wheel RECORD contents, wheel ABI/platform tags against the target
interpreter, and `LINUX_PAYLOAD.json` source provenance. The core identity-table
commit must match. Any mojolearn dependency absent from the supplied artifact
set is refused before pip runs. Both installation passes use explicit local
files, and the installed direct-URL hashes must match those files. Candidate
records are labeled `source: candidate`; they never claim a PyPI digest match.
All artifacts must also agree on `release_modes`. Missing metadata retains its
historical dual-mode meaning; the current environment cannot reinterpret it as
an identical-only artifact. The release controller also requires those recorded
modes to equal its selected release modes, including when it reuses a cached
candidate identity.

The gate also needs eight precompiled autograd executables for each vendor.
The orchestrator builds them from the clean candidate source freeze, using the
existing `autograd_smoke.py` commands and a target matching the held GPU:

```sh
python3 tools/gate_autograd_artifacts.py build \
  --vendor nvidia --source-commit FULL_CANDIDATE_SHA --target sm_89 \
  --mojo /root/mojolearn/.pixi/envs/default/bin/mojo --compile-jobs 2 \
  --out /root/results-candidate-autograd/nvidia
python3 tools/gate_autograd_artifacts.py build \
  --vendor amd --source-commit FULL_CANDIDATE_SHA --target gfx942 \
  --mojo /root/mojolearn/.pixi/envs/default/bin/mojo --compile-jobs 8 \
  --out /root/results-candidate-autograd/amd
```

Use the actual held GPU target and supported compiler worker policy; the example
targets are not automatic hardware selection. Run these as orchestrator-owned
builds on appropriate build machines. The commands build no wheel and publish
nothing. `autograd_smoke.py build-commands --vendor VENDOR --mojo MOJO --bin-dir
DIR --target-accelerator TARGET` still prints the underlying commands without
executing them.
An interrupted build can use `--resume` with the same directory, source,
compiler, target and flags. Completed artifact hashes are checked and reused;
old failed logs are retained. Changing any of those inputs requires a fresh
directory, keeping the earlier evidence intact.

Retain each directory's eight executable bytes, logs and `build-manifest.json`
on the Mac and R2. Place those directories below a local artifact root with
`nvidia/` and `amd/` subdirectories, then pass `--gate-autograd-root ROOT` to
`release.py` (default: the release directory's `autograd-bin`). The release
controller verifies the source-matched receipts and uploads the executables
to each gate's `out/autograd-bin`. It never compiles them on a timing GPU.

The local and remote gate directories include the candidate wheel identity.
The remote root is `/root/results-release-gate/`, within retained result paths.
A gate from a published installation or different wheel hashes cannot qualify
these candidate files. Missing binaries, incomplete setup, ABI errors and
unmatched pairs remain failures; publication also checks both gates directly
when a publish step is invoked alone.

For a manually staged candidate on a held box, the explicit interface is:

```sh
python3 tools/gate_smoke_box.py amd 0.8.56 /root/results-candidate-gate/amd all \
  --candidate-wheel /root/candidates/mojolearn_core-VERSION-TAGS.whl \
  --candidate-wheel /root/candidates/mojolearn_amd-VERSION-TAGS.whl \
  --source-commit FULL_CANDIDATE_SHA \
  --autograd-manifest /root/results-candidate-gate/amd/autograd-bin/build-manifest.json
```

Standalone published-version validation remains available by omitting all
candidate options. That mode keeps PyPI provenance and is explicitly labeled
`input_mode: published`; it cannot satisfy the prerelease publication guard.
This document and source changes are not executed qualification evidence.
