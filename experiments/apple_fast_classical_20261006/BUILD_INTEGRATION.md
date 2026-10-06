# Apple FAST classical paired build integration

This is programmed source integration, **uncompiled, unverified and unmeasured**.
No builder, manifest checker, import checker, test, quality gate, benchmark or
new helper was executed for this delivery. All candidate defaults remain off.

`build_pair.py` connects the 54 lane cards to the supported binding builders.
`build_bindings.json` records the classical package surface and each card's
minimum conservative native consumers. The future builder widens those maps
with transitive local Mojo imports; it never uses source import discovery to
erase a manually identified consumer or claim that a runtime route was reached.

Shared-source coverage includes linear Gram kernels in **both x_linear and
x_prep**, decomposition `DevExec` in **x_decomp, linalg, metrics and
x_neighbors**, exact-neighbor kernels in **core, metrics and hdbscan**, KMeans
in **core, mixture, metrics, x_cluster, ivf and x_ann**, forest inference in
**rf, trees and x_trees**, and TreeSHAP imports in **trees and x_trees**.
Conservative consumers can include compiled but unused specializations; that
does not claim affected workload coverage.

## Future interface

Run only after a separate authorization for compilation, through the existing
cheap Apple M2 queue with `MOJOLEARN_PERFORMANCE_QUEUE_JOB=1`. No queue submission
is part of this source delivery. The worker must already have its supported
pixi environment and `~/mojolearn-evidence/compile_slot.sh` provisioned.

```sh
python3 experiments/apple_fast_classical_20261006/build_pair.py \
  --idea AFCL-L01 \
  --source-sha FROZEN_FULL_COMMIT_SHA \
  --output /absolute/fresh/evidence/AFCL-L01
```

Repeat `--idea` to build a combined baseline/candidate configuration. Add
`--factorial` for all combinations of up to six interacting cards. The existing
G01/G02 route conflict is rejected. Required rollback/opt-in flags apply equally
to both arms, while each selected candidate contributes its own AFCL flag.
Both arm controls are read from the lane manifests, so the native recipe does
not carry a second independently maintained set of define lists.

## Isolation and reuse

- Require a clean checkout and the exact committed source SHA, then create a
  fresh detached build worktree outside the source checkout. Retain it and all
  failure evidence. Never build into the source checkout or an existing arm.
- Build every classical package dependency from this freeze. For each native
  binding, apply every selected card's arm defines if that binding is a declared
  or discovered consumer. The candidate cannot accidentally import an unchanged
  baseline binding containing the same affected kernel.
- Cache within this one invocation by binding, numeric mode and exact defines,
  under a fixed source/compiler/target/lockfile receipt. Unaffected bindings and
  equal prerequisite configurations compile once and are copied into each arm.
  No arbitrary historical artifact is treated as reusable evidence.
- Build the IDENTICAL core once, with no experiment flags, solely for the public
  adapters' existing input-transport helpers. It does not vote on FAST arithmetic
  or claim an IDENTICAL experiment.
- Clear inherited MojoLearn flags and mode/target overrides. Pass exact defines
  through `MOJOLEARN_MOJO_BUILD_FLAGS` only; both other extra-define channels stay
  empty. Enforce each card's absent-define requirements. Record runtime
  environment controls separately from compiler flags.
- Invoke existing `bindings/build*.sh` serially through the machine compile-slot
  semaphore with one compiler job and `MOJOLEARN_SKIP_BUILD_GATE=1`. Do not call
  native probes, import bindings or run device smoke checks. Existing builders'
  static artifact checks remain part of their supported build recipe.
- Stage each arm's complete runtime dylib closure with the existing
  `packaging/macos/stage_dylibs.py`, so a copied package can be used by the later
  Apple timing worker. This is ordinary package relocation/signing, not a new
  compiler mode. Record raw compiler hashes and final staged hashes separately.

The package intentionally excludes neural-only bindings and the mixed sequence
expansion binding: none of these 54 cards targets them. Its Python API source is
copied from the same frozen revision; only the requested classical workloads
are eligible for later execution. It is an experiment package, not a release
wheel or an assertion that all lazy imports in the full library are available.

## Artifact contract

`OUTPUT/paired-build.json` contains `schema`, `ideas`, `source_sha`, `vendor`,
`numeric_mode`, `status`, `toolchain`, `input_source_sha256`, `affected_bindings`,
`source_import_closures`, `builds` and `arms`.

For ordinary paired builds, `arms` has `baseline` and `candidate` entries.
Factorial builds use `configuration_000`, etc., with an explicit `cards` map.
Each arm contains:

- `package_root`: build-machine absolute path to `arms/NAME/python` for
  `PYTHONPATH`; `package_relative` gives the transferable path relative to the
  directory containing `paired-build.json`.
- `defines`, `exclude_defines`, `runtime_environment`, `cards` and `source_sha`.
- `artifacts`, keyed relative to the **mojolearn package directory**, containing
  the exact per-binding define set, source/compiler provenance, final staged
  `sha256`, raw `compiled_sha256`, build exit code and complete log path.
- `runtime_libraries`, keyed relative to the package, with hashes of the staged
  `.dylibs` files; `python_sources_sha256` covers the frozen Python shell.
- `runtime_staging`, including the stage command, log and exit status.

The final success string is `build_complete_unverified_unmeasured`. It means
only that the requested future build/staging process exited successfully; it
does not satisfy model quality, actual path reach, full-workload coverage or
speed acceptance. Interrupted/failed builds retain completed cells and a
`build_failed` top-level receipt. A later run must consume only complete arm
packages, check their exact provenance, and retain all full-workload quality
and timing evidence separately.

`logs/*.log` keeps full process output; `logs/*.receipt.json` records build and
staging results. Inspect exit codes, coverage and receipt status first. Use
bounded `rg` excerpts and short tails for diagnostics, never complete log dumps.
No compilation or execution log exists for this source-only delivery.
