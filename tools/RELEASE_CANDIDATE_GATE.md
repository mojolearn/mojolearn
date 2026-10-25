# Qualify candidate wheels before publication

`release.py` now gates the actual files emitted by `linux-pack` before any
publication. Both NVIDIA and AMD gates must pass. Owner clarification on
2026-10-25 separates bounded release verification from performance campaigns:
`release-light-v1` is the publication default. The core and vendor wheels are
local inputs; the candidate version need not exist on PyPI. NVIDIA also supplies its PTX
wheel. Third-party dependencies still install normally.

The light profile completes eleven installed verifier records (nine correctness
commands plus location and dependency checks), all 24 autograd entries, and the
three installed Torch/MAX adapter phases. It installs test/framework dependencies
without the board's opponent environment. The aggregate 1,800-second limit is a
hard ceiling, not an expected duration. Correctness commands have individual
180-second limits and autograd entries 30-second limits. Missing, timed-out,
`STARTED`, skipped or incomplete checks fail. Setup/framework installation and
test execution have separate elapsed-time records; coverage counts and any
unqualified verifier properties remain in the raw reports. An individual debug
stage cannot satisfy the complete profile's budget record.

`--profile performance-campaign-v1` explicitly selects the historical whole-plan
smoke and full-dataset opponent campaign; `release_lq_smoke.py` exposes this as
`--gate-profile performance-campaign-v1`. Historical reports without a profile
keep that campaign contract and still require complete measurements and matched
pairs. Performance ledger rows are optional for publication; complete comparable
measurements remain required for performance claims and default promotions.
The light gate does not replace required within-release identity, physical
multi-GPU or runtime evidence. Development and partial-build evidence retains its
own exact coverage/provenance; packaging checks do not demand repeating it.

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
The orchestrator builds missing/invalidated entries from the clean candidate source freeze, using the
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
The NVIDIA/AMD executables require a Linux x86-64 build host and explicitly use
`--target-cpu x86-64-v3`, matching the wheels' portable host instruction set.
The receipt records that baseline, and admission rejects older unpinned
commands. Metal's command remains unchanged. Compiler worker counts and the
GPU target are independent of this host instruction-set choice.
An interrupted build can use `--resume` with the same directory, source,
compiler, target and flags. Completed artifact hashes are checked and reused;
old failed logs are retained. Changing any of those inputs requires a fresh
directory, keeping the earlier evidence intact.

For a new freeze, add `--reuse-from /path/to/previous/complete/vendor-directory`
to the build command with a fresh `--out`. The builder checks the original
binary hashes, compiler hash/version, CPU/GPU targets, supported compile recipe,
transitive Mojo import closures and toolchain locks. It copies unchanged entries
and compiles only those whose relevant inputs changed. Every copied entry retains
its original source revision and manifest plus an explicit equivalence proof;
`reuse_refused` lists actual invalidation reasons. No Git commit label alone
invalidates a binary. Unresolved imports, missing source/locks, or changed flags
refuse reuse. Source discovery uses committed bytes, excluding untracked files.
If every input is unchanged, gate admission can use the original directory
directly and retain the same equivalence proof without invoking the builder.

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
failed correctness/adapter checks remain failures; publication also checks both gates directly
when a publish step is invoked alone.

For a manually staged candidate on a held box, the explicit interface is:

```sh
python3 tools/gate_smoke_box.py amd VERSION /root/results-candidate-gate/amd all \
  --profile release-light-v1 \
  --candidate-wheel /root/candidates/mojolearn_core-VERSION-TAGS.whl \
  --candidate-wheel /root/candidates/mojolearn_amd-VERSION-TAGS.whl \
  --source-commit FULL_CANDIDATE_SHA \
  --autograd-manifest /root/results-candidate-gate/amd/autograd-bin/build-manifest.json
```

Standalone published-version validation remains available by omitting all
candidate options. That mode keeps PyPI provenance and is explicitly labeled
`input_mode: published`; it cannot satisfy the prerelease publication guard.
This document and source changes are not executed qualification evidence.
