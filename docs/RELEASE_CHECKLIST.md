# Release checklist

## Readiness ledger and evidence reuse

`~/mojolearn-evidence/release/<version>/requirements.json` is the release's
readiness record. `tools/release_requirements.json` declares the required scope;
compile requirements expand by binding, identity requirements by lane, and
performance requirements by explicit full-workload recipe. Every row is
`passed`, `failed`, or `untested`, with its source revision, input fingerprint,
retained evidence and the revision actually tested. Pipeline completion flags
and handoff prose do not qualify a requirement.

`release.py` consumes this ledger before publication and before recording a
completed release. All required pre-publication rows must pass before publishing;
published-wheel checks remain required afterwards. `--preverified` reuses
qualified evidence; it does not bypass these gates. Source audits qualify only
source checks, and compilation does not qualify runtime behavior.

```sh
python3 tools/release_ledger.py sync --ledger "$HOME/mojolearn-evidence/release/0.8.57/requirements.json" --version 0.8.57 --revision integ/0857 --configuration /path/to/qualification-inputs.json
python3 tools/release_ledger.py plan --ledger "$HOME/mojolearn-evidence/release/0.8.57/requirements.json" --json
python3 tools/release_ledger.py check --ledger "$HOME/mojolearn-evidence/release/0.8.57/requirements.json" --stage pre
```

The input configuration is retained in the ledger. It declares `release_modes`,
`build_flags` and `dependency_context`, including an explicit `cpu_target`
(`osx-arm64` or `linux-64`) and full-workload recipes under `workloads` by vendor.
The accepted CPU identity column runs on Linux x86_64, so sync with
`--configuration tools/release_identity_targets.json` (`cpu_target: linux-64`);
`dependency_context` keys merge, so the retained compiler hashes are kept.
Identity units are admitted by `tools/release_identity_evidence.py`: `begin
--vendor <v>` before the column runs (it prints the `LANES=` list), then `admit
--records <dir>... --launch <launch.json>...` once the Apple, NVIDIA, AMD, CPU
and (for NVIDIA units binding the sm_80 set) PTX records exist. A record from
other hardware than the unit's target is refused.
A unit's records carry its begun run's revision. After a refreeze, a unit whose
begun run has the same requirement fingerprint as the current requirement is
admitted from records made at that run's revision; identity is never rerun for
it. The unit report and receipt (`identity_detail.revisions`) record the record
commits, the begun revision, the frozen revision and the fingerprint equality.
A begun run whose inputs changed is refused: close it and begin the unit again
at the frozen source, then record those lanes (`LANES=`) there. Each column of a
lane is taken at the revision its own unit uses (PTX follows the NVIDIA unit),
so one unit's columns may come from different fingerprint-equal revisions;
every slot is still compared bit for bit. Lanes of a record at a revision no
unit uses (for example the changed lanes inside an older full column) are
listed under `ignored` in `identity-admission.json` and never count as a
repeat; two records of one column at the revision a unit uses are a repeat.
The harness fixture check runs once, from a clean checkout of the frozen source.
Recipes name their identity lanes, source files, dataset identity, settings,
timing boundary, phases and environment. Missing mappings remain untested;
discovery of a recipe file does not establish full-dataset coverage.
`release.py --requirements-configuration <file>` imports these explicit inputs
and records the current toolchain and available artifact hashes.

A commit name is provenance, never an invalidation input. The plan lists the
specific changed source, compiler, flags, dependencies or coverage that require
work, and reuses unchanged evidence. Where a shared source file changed but a
frozen dependency analysis proves a lane unaffected, reuse preserves the old
receipt and tested revision with that proof. Unresolved dependency attribution
requires investigation; it does not silently schedule a complete rerun. Missing
or corrupt retained evidence first requires restoration from its archive.

Before launching work, producers call `begin --requirement <id>` on the ledger.
It returns reuse for an unchanged pass and refuses duplicate unfinished runs.
An exceptional rerun of unchanged qualified inputs requires both `--reason`
and retained diagnostic bytes through `--invalidation-evidence`; a new commit
or a free-text request alone cannot force that rerun. The run retains the
diagnostic hashes and all computed input-change reasons.
Completed work is recorded with `record --run-id <id>`, retained evidence,
exit status, coverage counts and `--result-report <typed-receipt.json>`.
`tools/release_requirement_receipts.py` defines the receipt contract: exact
requirement and input identity, per-coverage results and hashed raw evidence.
Every required coverage item must complete successfully for a pass. Legacy
reports without sufficient input provenance require reconciliation; do not
attach today's fingerprint and present them as a new pass. Candidate-gate
producers record these receipts directly. Admission also parses the approved
producer's retained report; a caller-authored receipt and an arbitrary log cannot
establish a pass. Unsupported report schemas remain untested until their producer
adapter is implemented and its actual coverage reconciled. This calls for
evidence admission work, not an automatic rerun of the underlying test.

Every rerun keeps its reason and original attempt. `release.py --redo <step>`
requires `--rerun-reason <specific invalidation or failure>`; a new commit alone
does not justify repeating completed checks. Existing binaries, identity cells
and measurements remain reusable when their relevant inputs match.

### Compile sub-units: runtime libraries, framework adapters, Apple pack bindings

`compile-<vendor>/runtime`, `compile-cpu/runtime` (Linux CPU column),
`compile-<vendor>/adapters/<arch>`, `compile-apple/adapters/apple` and the
compile-apple bindings the Mac packer compiles itself are admitted from retained
producers by `tools/release_compile_subunit_evidence.py`; nothing is rebuilt.

- `linux-release-leg`: one retained Linux release leg, `LEGS/<vendor>-<arch>` beside
  its `.provenance.json` and `.exit`. The hip/gfx942 leg gives `compile-amd/runtime`
  and `compile-amd/adapters/gfx942`; the cuda/sm_89 (host) leg gives
  `compile-nvidia/runtime`, `compile-cpu/runtime` and `compile-nvidia/adapters/sm_89`.
  The PTX leg (sm_80) owns neither and is refused. Checked: the leg built the tested
  revision from a clean, overlay-free tree (every compiled-inventory byte), the
  launch record's runtime identity digest, the Mojo compiler executable and
  toolchain, the release flags, the portable math helper's source, every staged
  `.libs` file, the adapter manifest, invocations (target, CPU, jobs, column),
  Torch pin and source bytes, and every shipped adapter file.
- `apple-pack`: a `tools/apple_retained_groups.py pack` directory with
  `pack-proof.json`. It admits `compile-apple/adapters/apple` and every binding of
  a group the packer compiled in that run (host, and identical groups whose
  retained placement was refused): its bincache key must reproduce from the tested
  revision's tracked bytes, name the admitted checkout and Xcode toolchain and carry
  the binding's flags. Retained placements are `apple-retained-device`'s; cache hits
  are not compilations and are not admitted.

An identity's `image` is the box-image pin from the tested tree (as for the device
bindings); the image the CPU box actually ran is kept as evidence, and the C++
compiler executable is not captured by the producers.

```sh
LEGS=~/mojolearn-evidence/release/$V/<build-sha12>/legs
for LEG in hip-gfx942 cuda-sm_89; do
  python3 tools/release_ledger_admission.py prepare --adapter linux-release-leg --ledger "$LEDGER" --root . \
    --producer leg=$LEGS/$LEG --out ~/mojolearn-evidence/release/$V/admit-subunits-$LEG
  python3 tools/release_ledger_admission.py admit --ledger "$LEDGER" --root . \
    --report ~/mojolearn-evidence/release/$V/admit-subunits-$LEG/report.json
done
PACK=~/mojolearn-evidence/apple-$V-final/pack   # after pack-proof.json exists
python3 tools/release_ledger_admission.py prepare --adapter apple-pack --ledger "$LEDGER" --root . \
  --producer pack=$PACK --out ~/mojolearn-evidence/release/$V/admit-apple-pack
python3 tools/release_ledger_admission.py admit --ledger "$LEDGER" --root . \
  --report ~/mojolearn-evidence/release/$V/admit-apple-pack/report.json
```

### Installed runtime and published producers

Two box producers fill `runtime-<vendor>` and (post
stage) `published-<vendor>`. Each runs against installed wheels in a fresh venv,
writes one report into a NEW output directory, and exits 0 only on a complete
pass. Pull the whole directory home, then admit it on the Mac with
`tools/release_ledger_admission.py` (adapters `installed-runtime` and
`published-install`, both in `tools/release_installed_evidence.py`). Evidence
paths are relative to the directory, so it can move; admission re-derives every
count from the raw files and refuses wheels that are not the ledger's
`artifacts:*` candidate bytes.

Each box needs the candidate wheels, `python3.12` with `venv`/`pip` and PyPI
access (numpy, pytest, and the qualifier's test-only dependencies: pandas,
scipy, torch, scikit-learn, pyarrow). The orchestrator queues them through
`lq ... CMD`; `V`, `SHA` (full commit packaged in the wheels) and `W` (the
candidate wheel directory) are placeholders.

```sh
# runtime-<vendor>: every file of tools/release_runtime_coverage.json, grouped by coverage ID.
# The tests are copied from a checkout of SHA; a git worktree of SHA is enough.
git worktree add --detach ~/src-$SHA $SHA
python3 tools/release_runtime_installed.py run --vendor nvidia --source-root ~/src-$SHA --source-commit $SHA \
  --wheel $W/mojolearn_core-$V-*.whl --wheel $W/mojolearn_nvidia-$V-*.whl --wheel $W/mojolearn_nvidia_ptx-$V-*.whl \
  --python python3.12 --out ~/mojolearn-evidence/release/$V/runtime-nvidia
#   amd:   --vendor amd   --wheel $W/mojolearn_core-$V-*.whl --wheel $W/mojolearn_amd-$V-*.whl
#   cpu:   --vendor cpu   --wheel $W/mojolearn_core-$V-*.whl            (Linux CPU box, cpu_target linux-64)
#   apple: python3 tools/mac_slot.py --wait-timeout 3600 metal -- python3 tools/release_runtime_installed.py run \
#            --vendor apple --wheel $W/mojolearn-$V-*-macosx_*.whl ...  (the M2 Pro)

# published-<vendor>, after publication: PyPI install, published hashes == candidate hashes,
# the eleven installed correctness jobs and the three adapter phases.
python3 tools/release_published_install.py run --vendor nvidia --version $V \
  --candidate-wheel $W/mojolearn_core-$V-*.whl --candidate-wheel $W/mojolearn_nvidia-$V-*.whl \
  --candidate-wheel $W/mojolearn_nvidia_ptx-$V-*.whl --python python3.12 --out ~/mojolearn-evidence/release/$V/published-nvidia
#   apple runs under tools/mac_slot.py ... metal -- with --candidate-wheel $W/mojolearn-$V-*-macosx_*.whl

# On the Mac, for each pulled directory DIR and its adapter:
python3 tools/release_ledger_admission.py prepare --adapter installed-runtime --ledger "$LEDGER" --root . \
  --producer report=DIR/runtime-report.json --out DIR-admission
python3 tools/release_ledger_admission.py admit --ledger "$LEDGER" --root . --report DIR-admission/report.json
#   published-install:  --adapter published-install --producer report=DIR/published-report.json
```

Multi-GPU verification is retired (owner, 2026-10-31): there are no
`multi-gpu-<vendor>` rows, no `verify --par` and no `par-*` identity lanes
(docs/TOMBSTONES.md, MOJOLEARN_PAR_VERIFY).

## Standalone native-only candidate

The optional standalone artifact has a separate strict qualification gate:
`tools/native_only_smoke.py` and the manual `native-only-smoke.yml` workflow.
The workflow builds a frozen commit, runs bounded native cases in a Linux root
filesystem containing no Python interpreter or packages, and retains the actual
runtime archive plus complete logs and provenance. Compilation defaults to two
jobs; each execution has a ten-second limit. Reuse verified unchanged binaries.

Do not ship or describe this candidate as a complete Python-free mojolearn
release until its inventory is covered and `summary.json` records
`release_qualified: true`. Missing algorithms, CPython-bound owners, failed cases
and partial selections prevent that result. A local Mac pass is dependency
screening. Existing public-operation callback audits and cross-vendor numerical
qualification remain required independently. See
[native-only release qualification](plans/NATIVE_ONLY_RELEASE.md).

## Releasing changes already checked before merge

This is the default release (Andrew 2026-10-13: identity and behavior are
checked before merge on NVIDIA, AMD, Apple and CPU; the release only builds,
checks artifact integrity and publishes). `pixi run release <v> --publish pypi`
builds Linux where the source's binding layout fits (merged bindings: RunPod CPU
pods, 2c; per-binding: GitHub's free runners, 2d), skips the GPU wheel smokes, gates the
macOS wheel on `tools/wheel_self_test.py` below, and publishes every package
with `tools/release_linux_publish.sh ... --preverified`. Its rehearsal skips
`python-tests` and the GPU-box leg dry-runs. `--gpu-smoke` restores the old
route: the NVIDIA/AMD wheel smokes, the expanded macOS smoke and light-smoke
receipts.

When the user requests publication using completed checks on merged main,
select `validation_profile=preverified` with prepared wheels and their pinned
`artifact_source_commit`. The manifest records `qualification_reuse` with that
source, `basis: "completed checks on merged main"`, and
`new_runtime_verification: false`. This path checks artifact integrity and
publishes without running CPU certification or new installed GPU/CPU smoke.
Do not manufacture fresh passing test receipts. Use `packaging/macos/build_release_wheel.sh --build-only` to compile and pack
without the Mac builder's automatic post-build runtime tests.

The verifier self-test gate still applies to a preverified macOS wheel: on the
Mac, before dispatching its publication, run

```sh
python3 tools/wheel_self_test.py <prepared macOS wheel> --out <dir> --python python3.12
```

It installs the wheel into a fresh venv and runs
`python -m mojolearn verify --self-test --cpu-threads 3`. Anything but
`WHEEL-SELF-TEST PASSED` (receipt `<dir>/results.json`, keyed to the wheel's
sha256) blocks publication: the bundled reference table does not reproduce
the wheel's own bits. Regenerate it (below) and rebuild.

## Regenerate the reference table whenever a release changes bits

A release that changes any digest (a new fold order, a parallel kernel that
replaces a serial chain, a changed default) must ship a reference table
(`python/mojolearn/verify_reference/table.json`) and portable models
regenerated at its source commit. 0.8.35 did not: `verify --quick` on the PyPI
wheel stopped at the comparator self-test (ols/base `23eecb87d9e84cc7` against
the table's `3d1d7c30b12d9872`). In order, from one pushed commit:

1. On each box (NVIDIA, AMD, the Apple M2, and once with `cpu` for the host
   column), in a checkout with every binding built:
   `tools/record_identity_column.sh <vendor-label> <absolute outdir>`
   (it records only the lanes whose closure changed and reuses the rest;
   `FULL=1` records every lane; shard a full column with `SHARD=i/N` when a
   job has a time limit).
2. On this Mac: `tools/admit_identity_columns.sh --build-host <outdir>...`
   places the records under `bench/results/identity_break/<date>-<REF_TAG>/` (REF_TAG, e.g. 0836),
   regenerates the table, and runs `verify --self-test` and `verify --coverage`
   against it. Commit and push.
3. On one GPU box, at that pushed commit:
   `tools/record_identity_column.sh <gpu-label> <outdir> --models`; then here
   `tools/admit_identity_columns.sh --models <outdir>/models`. Commit and push.

Before step 1, the coverage gate (static, on the Mac): `python3 tools/verify_coverage_audit.py --base <last
release source> --json <file> --markdown docs/plans/VERIFY_COVERAGE_<date>.md` for the endpoints changed
since the release, and again with `--all` (`VERIFY_COVERAGE_ALL_<date>.md`) for the whole public surface.
Both must show native GAP 0 and Python GAP 0; every new lane they name goes into the column records, and every
`coverage-*` arm in `tools/identity_lanes/coverage_sabotage.tsv` not yet in `coverage_sabotage_proven.tsv` is
proven with `tools/coverage_sabotage.py plan --arm <arm>` / `check` (`apply-check` first).

macos-smoke's self-test job fails a wheel whose table is stale; it cannot
regenerate one. Reuse is the default (Andrew 2026-10-31: reuse good old runs,
automatically). `tools/record_identity_column.sh <label> <outdir>` records just
the lanes whose closure moved since the newest good evidence of that device
class. That evidence is the admitted table, the committed and retained records,
the output dir and `MOJOLEARN_IDENTITY_REUSE_RECORDS=dir1:dir2`. Every other
lane is reused. The script prints `REUSED n / RECORD m`, or `NOTHING-CHANGED`.
`FULL=1` is the only way to re-record unchanged lanes. When it prints
`ADMIT-WITH`, pass those record directories to `admit_identity_columns.sh` as
well. `ONLY_CHANGED=table|<sha>` keeps its older table-wide meaning. The admit
is THE identity check of the release (Andrew 2026-10-10: identity runs ONCE; a
mismatch is a bug to fix, never a reason to rerun).

## The one command

```sh
pixi run release <version> --dry-run          # the plan, every command, and what is already done
pixi run release <version> --status           # per pipeline, leg and column: done, failed (log), owed; runs nothing
pixi run release <version>                    # everything up to publication
pixi run release <version> --publish pypi     # ... then publish each wheel as its pipeline passes, finish line, record
```

`tools/release.py` walks this whole checklist. Write the CHANGELOG entry
(`## <version> (published YYYY-MM-DD)`, UTC) first; the command bumps the two
version files, runs `write-docs-facts`, commits and pushes exactly those files,
freezes HEAD, runs the rehearsal (step 0) and the reuse plan, and then runs
FIVE PIPELINES AT ONCE, each publishing as soon as its own gates pass:

- **macos**: macos-build, macos-smoke (its self-test job is the bundled-table
  gate: publish-macos refuses without its PASSED receipt), publish-macos.
- **core-linux**: cross-compile (step 0b, beside the rehearsal; it gates
  linux-builds, the first rental), linux-builds (the legs of step 2, launched together,
  detached), linux-wait, linux-assemble, linux-pack (step 3: the core and both
  plugins plus PTX, 3b), publish-core-linux (last, after both plugins).
- **nvidia-ptx**: publish-nvidia-ptx, before the NVIDIA vendor package that
  requires it. The default preverified route uses its build and artifact checks.
- **nvidia**: gpu-column-nvidia (the expanded smoke, on the
  nv box we hold, the RunPod L40S sm_89), gpu-column-nvidia-hopper (only with
  `--hopper-box <ssh>`, a Hopper box we hold; otherwise SKIPPED), publish-nvidia
  (after publish-nvidia-ptx).
- **amd**: gpu-column-amd (on the amd box we hold, the DigitalOcean MI325X
  gfx942; the expanded smoke too), publish-amd.

  **RELEASES RENT NOTHING (2026-10-08).** The per-architecture rented smoke is
  deprecated. `--smoke-via lq` (the default) runs each column through
  `tools/release_lq_smoke.py`: it copies the final core and both plugins to `/root/release-smoke/<version>/<column>/in`
  on the box (sha256 checked there), queues
  `lq add --front <nv|amd> CMD main rel-<version>-<column>-<stamp> bash tools/release_wheel_smoke.sh ... --local BUILDS=none`
  (next in the box queue, after the running job and ahead of every grid line;
  no binding build, no pixi env), polls `lq results <box> <tag>`, fetches the out
  directory back and is judged exactly as the rented run was (same receipt). A relaunch while the job is queued polls
  it again (`<commit12>/columns/<column>.lq.json`); it never queues twice.
  Hopper (sm_90a) is unsmoked unless a Hopper box is held: publish-nvidia then
  needs only the Ada column, the sm_90a payload publishes on its GitHub build
  receipt, the alpha manifest carries `"smoke": {"sm_90a": "not run (no Hopper box held)"}`
  and the vendor wheel's README says so. `--smoke-via rent` keeps the old
  rented route (`--smoke-gpu`, `--amd-provider`) for an emergency only.
  Both columns run at once as detached legs from the installed core and both
  plugins. ONE IDENTITY CHECK PER RELEASE (Andrew 2026-10-10: identity runs ONCE; a mismatch is a bug to fix, never a reason to rerun):
  the admitted table decided NVIDIA == AMD; the release no longer re-fits the
  changed lanes (release-check, the column pass) or diffs them again
  (linux-joint-diff, the macOS cross-diff).

A failure in one pipeline never blocks or undoes the other. The run ends with
every outcome and exits non-zero unless every package published. The finish line
checks each published platform (`pip install` on this Mac for macOS; `pip
download` of the Linux file with its sha256 compared) and the
`bench/results/release_verification/<date>_pypi_<v>/` record covers what is
published; both run again when the other platform publishes.

**The shipped source and the release tooling are two commits.** The freeze
pins the source commit in `state.json`; it never moves because main moved,
and only `--refreeze` moves it to HEAD (refused once a wheel of it is
published). `tools/release.py` and everything it drives (the leg runners, the
guards, the wheel smoke, the publisher) run from the checkout the command is
run in, normally current main, so a tooling fix lands without a refreeze. The
steps that read the source (the macOS build, the pack, the
rehearsal) run in the source checkout: this one when HEAD
is the source commit, else a detached worktree under
`<version>/source/<commit12>/` (or `--source-checkout`). A build leg's box
unpacks the source archive and then the route overlay
(`tools/release_tooling.py`, `tools/route_overlay_lib.sh`): the tooling copy
of each box-side tool that differs (the remote build driver, the serial
guards, the binding cache), sha256 before and after in
`<leg>/route-overlay.txt`. Nothing in the source inventory is ever overlaid
(Mojo files, `bindings/`, `packaging/`, `python/`, `tokenizer/`, `pixi.*`,
`tools/linux_surface_qualification.sh`), and an overlaid builder enters the
binding identities with the bytes that ran. Every leg, column and record
names both commits (`<legs>/<leg>.provenance.json`,
`column-provenance.json`, the record's `release-provenance.json`).

**Results are keyed to what they built.** Under a new freeze, a completed
build leg of an earlier freeze is taken (not launched) only when its set
identity (`release_reuse.set_identity`: every binding identity of the set,
the host bindings and the runtime closure) and its build tooling digest are
equal, its proof is complete, every binary on disk is the proof's bytes and
its overlay verified; `<legs>/<leg>.reused.json` says built from X, admitted
for Y and why, and the pack names those files in `reuse.json` with that
origin. A GPU column is taken only for a byte-identical wheel (sha256) and the
same lane selection, and is diffed again against this release's references.
Changed relevant inputs require the affected check; the requirements ledger
records the specific invalidation before that work starts. Unreadable evidence
requires restoration from its archive, and uncommitted inputs require a frozen
source record. Neither condition alone authorizes repeating completed work.
Every leg and column PASS is retained in the artifact cache in R2
(`tools/release_ledger.py`, `release-ledger/v1/` in `mojolearn-data`, keyed
by set identity and tooling digest, or wheel sha256, vendor and lane
selection). This cache supplies evidence to the canonical requirements ledger;
it is not a separate readiness decision. The release consults retained results
before launching and admits a PASS only when its evidence is available and
verifies against the requirement's inputs and coverage.

It is resumable: rerun the same command after any stop, and only what is not
done runs. State lives in `~/mojolearn-evidence/release/<version>/state.json`,
and everything after the freeze in `<version>/<commit12>/` (legs, wheels,
smoke receipts, logs). Each step checks its own output (a wheel of the frozen
commit, a PASSED receipt for that wheel's sha256, the file already on PyPI) and skips when it is there. A running
build leg is never relaunched; a failed one is moved aside, never deleted, and
relaunched. `--only STEP[,STEP]` runs some steps, `--redo STEP` discards a
step's record, `--amd-expect-from <NVIDIA release-build dir>` restores the AMD
core-host probe, `--smoke-gpu` picks the Linux smoke GPU. The build legs and
the GPU columns share one launcher (`launch_detached`; build routes in
`BUILD_BACKENDS`, chosen with `--build-backend`, default by the source's layout, 2c). A pipeline is `PIPELINES`'
named builds, checks and one publish, so the Linux pipeline can split into
per-package pipelines without a new scheduler.

**Bindings are rebuilt only when their identity moved (2026-09-23,
`tools/release_reuse.py`).** After the rehearsal the `reuse-plan` step decides,
for every binding of every set (cuda sm_90a, cuda sm_89, hip gfx942, the host
bindings, the runtime closure, and the macOS wheel), REUSE or BUILD. The
identity of a binding is its source closure digest (the same walk
`tools/binding_stamps.py` records, computed from `git archive` of the commit),
the Mojo/MAX packages pixi.lock pins for the target platform, the build flags
the release scripts export for it, the builder scripts, and the pinned box
image (on macOS the Xcode and Metal toolchain instead). REUSE means the exact
identity digest of the last PUBLISHED release (the newest
`bench/results/release_verification/<date>_pypi_<v>/` record, its
`binding-identities.json` when it has one, else that commit's tree); the
published bytes then come from a local copy or PyPI, verified by sha256 against
the record and against the wheel's own payload, and are packed again. Anything
else, including an unreadable pin, builds. A build leg runs only for a set with
a binding to BUILD (the leg builds the whole set as before; the pack still
takes the published bytes for the set's REUSE bindings, which is what freezes
a released gfx942 binary), so a Python-only release rents nothing for builds
and packs from `<version>/<commit12>/reuse/sets/`, while the NVIDIA and AMD
release columns still run from the packed wheel. `LINUX_PAYLOAD.json` records
per binding `built` or `reused` and from which release (`binding_origin`,
`sets`); the macOS build places its reused bindings from the published macOS
wheel and refuses if a byte moves after staging. `--dry-run` prints the full
decision table with a reason per BUILD row; `python3 tools/release_reuse.py
plan --full` prints it on its own.

The numbered steps below are the same work by hand: the fallback when a step
refuses and needs a person, and the reference for what each step runs.

### Bounded followup after a completed qualification run

For an alpha Python/reference patch with unchanged native inputs, retain the
original reports and compare the actual old and new wheel payloads. Admit
unchanged results only when the relevant shipped Python and native bytes are
unchanged and their values still match the final references. Recheck every
lane in the affected family; do not mark the original failed run as passing.
`tools/admit_cluster_identity_repair.py` implements this admission for the
clustering buffer repair, including the complete original lane inventory.

Pack all final wheels with the new source witness, including plugins whose
payload metadata names that source. Run the expanded installed smoke on each
exact final platform/vendor artifact and compare the selected public identity
reports with `tools/compare_installed_identity.py`. These bounded receipts do
not claim full numerical certification: retain the broader original campaign,
reference-admission evidence and native-source proofs alongside them. Any unresolved applicable failure holds publication.

Use the by-hand publication steps below when the orchestrator cannot represent
these reports. Do not synthesize legacy `column.json` files or mark unexecuted
orchestrator steps complete. Preserve the original receipts and their SHA256s,
run the existing exact-artifact admission checks, publish both Linux plugins
before the Linux core, and complete the post-index installation checks.

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

## 0b. Cross-compile on GitHub (the release does this; nothing is rented)

The `cross-compile` step of `pixi run release` dispatches
`.github/workflows/cross-compile-check.yml` on the frozen commit, diffed
against the previous release's source commit (the reuse plan's base), and
waits for it (`gh run watch --exit-status`). The plan job runs
`tools/cross_compile_check.py --list-json` (the same selection as the local
`pixi run cross-compile-check`), and every changed Linux GPU binding x tier x
arch (sm_90a, sm_89, gfx942) compiles on its own free standard runner, in
parallel, with the release flags. The summary job fails on any FAIL, TIMEOUT,
STALLED or MISSING job and names it; the release step then stops the Linux
pipelines by name before linux-builds rents a box (the macOS pipeline goes
on). The workflow is `workflow_dispatch` only and never runs on a merge. By
hand:

```sh
gh workflow run cross-compile-check.yml --ref main -f ref=<previous release commit> -f commit=<sha>
```

`MOJOLEARN_XCC_WORKFLOW_REF` names the branch whose workflow file is
dispatched (default `main`; GitHub dispatches a workflow only once its file is on the default branch). Measured 2026-09-27 on the 0.8.22 diff: 96 jobs
(32 per arch), 1 to 5 minutes each (median 2), compiler peak RSS 2.0 GB, 20 runners at a time, 19 minutes wall, all PASS; the same run with the b055fe72e potrf gate reverted failed by name on gp, kernel_methods and mixture (gfx942, both tiers).

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
pixi run test-wheel-audit                                        # optional NumPy only; owned math helpers, with the tested CPython compensated-sum guard
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
# NVIDIA sm_90a (RunPod H100) and sm_89 (RunPod L40S), started together
MOJOLEARN_RUNPOD_KEY_FILE=~/.mojolearn_runpod_key MOJOLEARN_NVIDIA_CAMPAIGN=7 MOJOLEARN_GPU_ARCHS=sm_90a \
  sh tools/gemm_remote_leg.sh nvidia --payload mamba --source-ref $REF --gpu "NVIDIA H100 80GB HBM3" --allow-concurrent --rent --minutes 60
MOJOLEARN_RUNPOD_KEY_FILE=~/.mojolearn_runpod_key MOJOLEARN_NVIDIA_CAMPAIGN=7 MOJOLEARN_GPU_ARCHS=sm_89 \
  sh tools/gemm_remote_leg.sh nvidia --payload mamba --source-ref $REF --gpu "NVIDIA L40S" --allow-concurrent --rent --minutes 60
```

**The AMD providers and their order (2026-09-25).** Every AMD box is gfx942,
the architecture the wheel ships (MI300X on RunPod and Hot Aisle, MI325X on
DigitalOcean). The AMD BUILD leg rents a DigitalOcean MI325X
(`tools/do_release061_leg.sh`), or a Hot Aisle 1x MI300X
(`tools/hotaisle_release_leg.sh <commit> [--rent] [--expect-from DIR] [--lease 30..60] [--cap USD]`)
when DigitalOcean has a GPU droplet live (its leg refuses a rental then, one
GPU droplet per account) or no token. The Hot Aisle leg runs the same archive,
host preparation, pinned Ubuntu 22.04 container, `release061_remote_build.sh`,
read-backs and provenance, and writes the same `legs/hip-gfx942/release-build/`
tree; it takes only the 1x VM, because `tools/amd_serial_guard.py` requires
exactly one visible render GPU. `pixi run release` decides once per run
(`--amd-build-provider auto|do|hotaisle`, or `MOJOLEARN_AMD_PROVIDER`), and
`linux-wait` walks a DigitalOcean leg that refused on a live droplet to Hot
Aisle. The AMD COLUMN (`release_wheel_smoke.sh --vendor hip`, step 5) walks
`runpod`, `hotaisle`, `do` under `--provider auto`: RunPod once; Hot Aisle when
RunPod has no MI300X (the 1x VM, or the 2x VM pinned to GPU 0 when no 1x is in
stock, `--hotaisle-spec`, `--hotaisle-cap`); DigitalOcean when Hot Aisle refuses
before creating anything (no key, stock, slot, balance or cap). Hot Aisle's
guards are `tools/hotaisle_vm_lib.sh`'s: the whole lease priced live against the
cap (the balance is only recorded), a Mac dead-man before the create, an on-box
watchdog verified from two sessions, DELETE then GET 404 before the slot is
released. `tools/tests/test_hotaisle_release_shim.py` tests both against a
local stand-in API; `bench/results/release_hotaisle_2026-09-25/` is the real
rehearsal.

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

## 2c. The CPU build route: the merged layout's default (`--build-backend cpu-box`)

**Which route builds Linux is decided in ONE place: the LINUX BUILD BACKENDS
table at the top of `tools/release.py`.** In short: the default follows the
source commit's binding layout (`packaging/linux/merged_layout.py enabled` at
that commit). A merged-layout source (MOJOLEARN_MERGED_BINDINGS=1, the default
since 2026-10-20) builds here, on RunPod CPU pods, with or without
`--gpu-smoke`, and `--build-backend github` is refused for it: one merged
compile per tier peaked at 46.4 GB (identical + byte LM; fast 36.6 GB, host
6.9 GB; DO MI325X, 2026-10-20) and a free runner has 15. A per-binding source
keeps GitHub (2d) as its default and this route behind `--gpu-smoke`.

```sh
bash tools/release_linux_build.sh $REF                        # dry run: nothing rented
bash tools/release_linux_build.sh $REF --rent --archs sm_89   # one set, what each release leg runs
```

The pod follows the layout. Merged: 16 vCPU of `cpu5m,cpu3m` (>= 64 GB), one
merged compile at a time (identical + fast together would need 83 GB) under a
64 GiB build cap (`tools/release_linux_cpu_box.sh` sets it through
`tools/release061_remote_build.sh`), every core given to the build step, the
compiler workers left as `packaging/linux/build_sets.sh` sets them, and a lease
of 75 min plus 45 per set. Per-binding: 32 vCPU of `cpu5g,cpu3g`, 16 builds at
a time at 16 GiB each, lease 120. `MOJOLEARN_MERGED_BINDINGS` in the calling
shell picks the layout and is passed to the pod when set.

One RunPod CPU pod runs the AMD leg's pinned
`rocm/dev-ubuntu-22.04` image (GCC 11.4, ld 2.38, patchelf 0.17.2) and builds
cuda/sm_90a, cuda/sm_89 and hip/gfx942 one after another at `/root/mojolearn`
through the same `tools/release061_remote_build.sh`, with
`MOJOLEARN_RELEASE_NO_DEVICE=1`: Mojo compiles each set from
`--target-accelerator` alone, the guard (`tools/cpu_build_guard.py`) refuses if
any GPU device node is visible, and the postflight requires the read-back
architecture to equal the requested one. The output is
`~/mojolearn-evidence/releases/$REF/linux-cpu-box/<stamp>/{cuda-sm_90a,cuda-sm_89,hip-gfx942}/release-build`,
the same trees the GPU legs write; the script prints the step 3 pack command.

Proof at d181d9792 (0.8.14) against that release's three GPU-box builds,
sha256 of every `.so` in each set (tiers, `host/`, `.libs/`):
cuda/sm_90a 66 of 66 and cuda/sm_89 66 of 66 byte-identical, with
`readback.txt`, `arch_readback.txt` and the provenance extension and
host-binding digests identical. hip/gfx942 matched 62 of 66, an open identity defect under investigation
(lane/amd-gfx942-identity), not yet attributed to our code or the compiler:
the gfx942 builds were not reproducible run to run on any box (with a
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

## 2d. The GitHub build route: the default for either layout (`--build-backend github`)

The same CPU build route as 2c, on GitHub's free `ubuntu-24.04` runners (4
vCPU, 15 GB) instead of a rented pod. Nothing is rented and no repository
secret is needed. It is the per-binding layout's default since 2026-09-27 and
the merged layout's since 2026-10-26 (Andrew: compiling never uses paid
rentals); `--build-backend cpu-box` rents a CPU pod and `gpu-legs` GPU boxes
by name.

**The merged layout: one compile per job, swap for the peak.** A merged file
is one `mojo build` of its directory set's root: Mojo does not compile the
members separately into objects that link into the same bytes, so a merged
file cannot be split across jobs without becoming a different artifact. The
plan gives each merged compile of a set (the identical tier, the host set,
the fast tier when the release modes name it) its own shard job, which runs
the cpu box's exact compile (image, path, flags, compiler workers, so the same
bytes and cache keys). When the compile's planning peak
(`release_github_build.py UNIT_PEAK_GIB`, the largest RSS recorded for it:
identical about 48 GiB at 0.8.57 on the cpu box; the host compile was killed
at the runner's 16 GB without swap in run 40272710760, so it is planned at 48
too) exceeds the
runner's 15 GiB, the job first adds a swapfile on the runner's scratch disk
(`gha_release_box.sh swap`, recorded in `GHA/.../swap.txt`); the plan refuses
a compile that would need more than `SWAP_MAX_GIB`. Each shard records its
peak (`memory.txt`): raise a `UNIT_PEAK_GIB` row when a run records more.
`python3 tools/release_github_build.py estimate` prints the jobs, swap and a
lower-bound wall time for the current tree. The workflow takes the release
`modes` (`MOJOLEARN_RELEASE_MODES`, default `identical`), exported to the
route as the cpu box does.

**The host producer.** `host` in the workflow's `sets` (`run --host`) adds a
job that runs the source's `tools/release_host_build.py build` in the same
image from a git checkout of the commit, with the default pixi env and
patchelf 0.17.2, its runtime donor this run's first assembled set or a
retained `<leg>/release-build/build` (`run --host-donor DIR`: the Mac uploads
it to R2 and passes one presigned GET plus its sha256). Its artifact
`host-build` is what `tools/retained_release_composition.py --host-build`
takes; the job adds the host compile's swap like a shard (`docs/plans/RETAINED_DEVICE_HOST_COMPOSITION_20261025.md`).

**The cache without R2.** With no `map_url` (`run --no-bincache`) each shard
restores its hot directory from `actions/cache` (key
`rlb1-<arch>-<unit>-<commit>`, falling back to the newest of the same unit)
and saves back only the archives its own builds used or made. An archive is
named by its content key and placed only after it verifies, so an older
commit's archive is reused exactly when the unit's inputs are unchanged.
With a `map_url` R2 is the cache and `actions/cache` is not touched (gfx942
bytes stay frozen by R2).

```sh
pixi run release <v> --build-backend github          # the three legs, detached
# one set by hand (the leg's own command):
python3 tools/release_github_build.py run --commit $REF --arch sm_89 --out DIR
```

`.github/workflows/release-linux-build.yml` is `workflow_dispatch` only. Per
set: a `plan` job evaluates the SOURCE's own `build_sets.sh` lists (76 builds:
44 GPU bindings over three tiers, 32 host bindings), checks the counts against
`verify_linux_surface_qualification.py`, and splits them into `shards` jobs
(default 6) by measured compile time; each `shard` job runs
`tools/gha_release_box.sh shard` inside the CPU box's pinned
`rocm/dev-ubuntu-22.04` image (by digest, at `/root/mojolearn`, the RunPod
file list, the same route overlay), which is `tools/release_linux_cpu_box.sh`
unchanged with `MOJOLEARN_BINCACHE_SHARD_ONLY` naming its builds (every other
build returns at once; the variable is not a cache-key input), `jobs` builds at
a time (default 2). Each compiled binding lands packed and keyed in the hot
directory, the shard's artifact. One `assemble` job per set puts every shard's
archives in its hot directory and runs the route with no filter: every build
is a verified `hot-hit` (or an R2 `hit`), a build no shard delivered compiles
there, and the output is exactly a build leg's tree. The artifact
`release-<vendor>-<arch>` holds `<vendor>-<arch>/release-build/` (with
`build/build-provenance.json`) and `GHA/` (per-job timings, memory, cache
outcomes, uploads). `run` downloads it into the cpu-box leg layout, verifies
every binary against the proof and every `.libs/` library against the set
manifest, promotes the cache uploads, and names any failed job.

**The binding cache without a secret.** The Mac mints the URL map with its own
`~/.mojolearn_r2` (`tools/bincache.py plan_lines`: presigned GETs of the
`none/runpod-cpu-...` partition, presigned PUTs into disjoint inbox slots per
job), stores it in R2 as one object and passes one presigned GET of it (8 h)
as the input `map_url`. No workflow line interpolates it: each job reads it
from the event file, masks it, and keeps the map in a 0600 file no artifact
carries. After the run the map object is deleted, and an inbox upload is
promoted only when its bytes hash to the archive the job recorded
(`GHA/bincache/hot_sha256.tsv`). The keys name the image as
`runpod-cpu:<digest>`, the CPU box's own declaration: same image, path, route
and OS fields, so the GitHub jobs read and freeze the same partition.
`--no-bincache` builds NVIDIA sets cold; gfx942 refuses to build without the
cache unless `--allow-cold-amd` (its codegen is not reproducible, 2c).

**Registering the workflow.** GitHub dispatches a workflow file only from the
default branch, or by id once registered; `run` dispatches by id. Until this
file is on main, register it once from a throwaway branch whose copy adds
`push: branches: [<that branch>]`, push, cancel the run, delete the branch.

**Proof at c8654671b (0.8.24), 2026-09-27.** Every set against 0.8.24's
released legs, sha256 of every `.so` (tiers, `host/`, `.libs/`), the proof's
`extensions`, `host_extension`, `source_inventory` and `source_sha256`, and
`readback.txt`/`arch_readback.txt`:
all three sets 81 of 81 byte-identical, and every compared proof field and
read-back equal; `manifest.json` differs only in its `set` field, the staging
path, which nothing reads. cuda/sm_89 was compiled entirely on the runners
with no cache (`--no-bincache`, 76 hot-hits: runs 36318515173, 36319821137),
equal to 0.8.24's L40S GPU-box build; cuda/sm_90a compiled its 44 GPU bindings
on the runners and took the 32 host bindings from R2 (run 36319826564), equal
to the H100 build, and its 44 archives were promoted; hip/gfx942 took all 76
from R2 (the 38 bindings 0.8.24's CPU pod froze there plus older ones; runs
36318586673, 36319831210), equal to the released AMD set by construction.

Measured (the three sets dispatched together, 6 shards, 2 builds at a time):
21 minutes wall for all three (1268 s; 1169 to 1267 s for a cold NVIDIA set,
1075 s for gfx942 from the cache). A cold NVIDIA shard job takes 7.5 to 14
minutes, of which about 4 is setup (image pull, source, pixi) and the rest its
builds (the longest single build 206 s, `fast:build_gbdt.sh`); the assemble job
6 to 7 minutes. Peak memory per job: at most 4.2 GiB process-group RSS (the
guard) and 7.3 GiB for the whole container including page cache, of 16 GB; 4
vCPU run 2 builds of 2 compiler workers. Nothing near the 150-minute job cap
or the 20-minute per-binding bound. The weights in
`tools/release_github_build.py` are these runners' measured build times.

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

### 3b. Vendor GPU packages

The Linux release uses three existing projects at one exact version. Ownership
and pins come from `python/mojolearn/gpu_plugins.py`:

| Project | Contents | Exact-version dependencies |
|---|---|---|
| `mojolearn` | Python, host bindings, shared runtime | `mojolearn-nvidia`, `mojolearn-amd` |
| `mojolearn-nvidia` | Native Ada `cuda_native/sm_89/` and Hopper `cuda_native/sm_90a/` (or `sm_90`) | `mojolearn` |
| `mojolearn-amd` | Native AMD `hip_native/gfx942/` | `mojolearn` |

Ordinary `pip install mojolearn` installs both vendor packages automatically.
The loader selects the installed architecture for the detected GPU. Packaging
by vendor does not make machine code portable across GPU generations or add
new supported architectures. The macOS wheel remains separate.

Binary bytes and architecture directory depth are preserved, including relative
RUNPATHs. `test_split_wheels.py` checks that the three wheels partition the
combined package without changing a native member. `test_plugin_upgrade.py`
checks pip's actual uninstall of the prior vendor-directory layout.

```sh
pixi run -e pkg pack-linux-wheel --profile release-split \
  --set <sm89>/build/sets/cuda --set <sm90a>/build/sets/cuda --set <hip>/build/sets/hip \
  --build-proof ... --out <dist>
python3 tools/wheel_api_audit.py --split <dist>/*.whl
python3 tools/gpu_release_projects.py <dist> --require-complete
```

NVIDIA requires both Ada and Hopper sets; a vendor-only pack cannot silently
omit one. Audit every wheel with the core beside the vendor packages. Native
vendor wheels exclude the runtime libraries owned by the core.

The NVIDIA project needs a **250 MiB per-file PyPI allowance**. The packer and
structural audit use this requested NVIDIA budget, while core and AMD retain
100 MiB budgets. These checks do not grant a server-side allowance: confirm the
increase on the target index before uploading a NVIDIA wheel above 100 MiB.
No architecture-specific PyPI project or trusted publisher is required.

Release columns install all three wheels. Ada, Hopper, and AMD still require
separate actual-architecture receipts. NVIDIA publication waits for both Ada
and Hopper; vendors publish before the core. The core upload checks that both
exact vendor versions are available on the selected index. Experimental PTX
remains excluded from publication.

Inventories retain the frozen source commit and build origins. Do not relabel
older artifacts or receipts as a newer main commit. Historical six-wheel `5c`
candidate evidence remains tied to those files and can still be exercised with
its frozen harness; it is not a receipt for these new three-wheel bundles.

#### Index installation and trusted publishers

Test local artifacts before upload, then resolve the actual dependency graph
from TestPyPI before PyPI. `tools/index_install_check.sh <v> <index>` is a dry
run; `--rent` runs the NVIDIA/AMD installed-index checks. The precheck requires
all three projects, exact pins, and non-yanked manylinux wheels. The separate
Hopper qualification column is still required.

| Project | PyPI environment | TestPyPI environment |
|---|---|---|
| `mojolearn` | `pypi` | `testpypi` |
| `mojolearn-nvidia` | `pypi-nvidia` | `testpypi-nvidia` |
| `mojolearn-amd` | `pypi-amd` | `testpypi-amd` |

Keep the existing project publishers for repository `mojolearn/mojolearn`,
workflow `release-provenance.yml`. Each job uploads only its project's files.

#### The NVIDIA PTX slot

Andrew 2026-10-10: PTX is a normal target; no flag. `mojolearn-nvidia`
carries the native sm_89 set AND the PTX set `cuda_ptx/sm_80` as a regular
slot (`gpu_plugins.PTX_ARCH`). The release builds it as the leg `cuda-sm_80`
on every build backend, through the same route as any arch
(`tools/release061_remote_build.sh` gives cuda/sm_80 the `ptx` code format, so
`packaging/linux/build_sets.sh` keeps the rounding-pinned PTX and writes
`PTX_BASELINE.json`), cross-compiles it in `cross-compile-check`, and packs it
into the vendor wheel with its build proof. The vendor marker binds the PTX
manifest by its SHA256. `pack_wheel.py` checks every wheel against PyPI's
100 MiB file limit at pack time and fails with the per-set sizes.

The loader takes a native set when the device's architecture has one and the
PTX set on every other NVIDIA GPU of compute capability 8.0 or newer. Its
identity is its column in the reference table, recorded once on an NVIDIA box
(`tools/record_identity_column.sh nvidia-ptx-<gpu>-sm80 <out>`) and admitted with
the other columns; see [the NVIDIA PTX identity contract](NVIDIA_PTX_IDENTITY.md).

## 4. Install and test on real GPUs (OPTIONAL, never required for a release)

A release is verified by the Apple column (section 5b) and the NVIDIA and AMD
wheel columns (`gpu-column-nvidia`, `gpu-column-amd`). This section is a diagnostic for a suspected NVIDIA or AMD problem and the release
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

The AMD column (`--vendor hip --column <selection>`) runs the same way from the
installed wheel on gfx942. `--provider auto` (the default) rents a RunPod MI300X
once and, when RunPod answers "There are no instances currently available"
(0.8.16), a Hot Aisle MI300X VM, and when Hot Aisle refuses before a create
(section 2, the AMD providers), a DigitalOcean `gpu-mi325x1-256gb` droplet in tor1 or nyc2 instead:
token `~/.mojolearn_do_token`, the shared GPU lock `/tmp/mojolearn-do-gpu.lock`
(held = refused), a Mac dead-man armed before the create, an on-droplet
self-destruct verified after ssh, DELETE then GET 404 before the lock is
released. `--provider runpod|hotaisle|do` pins one; `pixi run release` passes
`--amd-provider`.

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

## 5b. Optional bounded diagnostics on this Mac

These commands are diagnostics, not release identity qualification: they use
three fixtures and omit batch and rlpair probes. They require an explicit
`--diagnostic-only` acknowledgement before execution. For release identity,
reconcile retained evidence first and use `tools/release_identity_run.py` to
produce only missing or invalidated full-protocol records. A new version or
an empty new ledger does not justify repeating existing verification.

```sh
pixi run -e test release-check --diagnostic-only
pixi run -e test release-check --diagnostic-only --cpu-column
# which is, concurrently (tools/release_check.py):
pixi run -e test apple-pass --diagnostic-only
MOJOLEARN_CPU_PASS_SLOTS=4 pixi run -e test cpu-pass --diagnostic-only
```

These partial columns must not be substituted for full release records.

Both refuse to start unless every binding in `python/mojolearn` was built
from this tree's sources (`tools/binding_stamps.py check`: the macOS release
build stamps each binding with its source-closure digest; a missing or stale
stamp names the binding and nothing runs). Build the macOS wheel (step 6's
build script) first, then run the check.

Both run locally, with no pod and no R2 staging. Each pass is
`base,denormal,odd`, fitted once, with batch and rlpair probes omitted.

**Diagnostics select lanes affected by changes.**
With no selection given, a pass checks the lanes whose sources changed since
the last pass on the same backend that FINISHED on this Mac (complete, same
fixtures, one fit, a clean tree, and itself anchored on a full pass, a tag or
another such pass). With no such record it falls back to the newest `v*` tag.
Every cell is fitted once. When it runs, the CPU pass covers the union of its
own and the Apple pass's selection.

**The selector never widens and never guesses (2026-09-22).** Each changed
path is attributed to exactly the lanes whose derived source set reaches it
(kernels, bindings, host oracles, the lane's own code in
`tools/identity_break.py` by call graph, data files the lane reads), is inert
(prose, evidence, tools and check programs no lane runs), or selects every lane
by a NAMED rule that is printed (the pinned toolchain in `pixi.lock` or
`pixi.toml`, a whole-surface registry such as `_backend.py`, harness code every
column runs, the Linux set builder for the NVIDIA and AMD columns). A path it
cannot attribute is printed as `UNATTRIBUTED PATH: <path>` and the pass exits
non-zero with nothing verified, until the mapping in `tools/lane_select.py` is
fixed. `--all` is the explicit full sweep. To see what the default would run,
per backend, with each lane's reason and the cell counts, without running
anything:

```sh
pixi run -e test release-check --plan                       # the diff since each pass's anchor
pixi run -e test release-check --plan --paths=a.mojo,b.py   # a hypothetical change
```

**Sharded Metal is opt-in and unproven.** `pixi run -e test apple-pass --diagnostic-only
--metal-shards N` (N = 2 or 3) splits the lanes over N processes that share the
one GPU under a single Metal slot, to overlap their host-to-device waits.
Concurrent Metal jobs returned NaN and zero outputs on 2026-09-15, so N stays 1
until this has been run once, after a release, on a quiet Mac:

```sh
C=$(git rev-parse --short=12 HEAD); E=~/mojolearn-evidence/metal-shards; mkdir -p $E
for n in 1 2 3; do
  MOJOLEARN_RELEASE_CHECK_DIR=$E/n$n /usr/bin/time -p pixi run -e test \
    python3 tools/verify_lanes.py --apple-pass --diagnostic-only --all --metal-shards $n 2>&1 | tee $E/n$n.log
done
# every part of every cell, sharded against unsharded, then against the CPU
# column of the same commit (the reference the Apple column must equal; make
# it cover those diagnostic lanes with `pixi run -e test cpu-pass --diagnostic-only --all`):
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

## The release gate: smoke and measurement on the boxes (the `gate-nvidia` and `gate-amd` steps)

After each Linux vendor publishes, `pixi run release <version> --publish pypi` runs that vendor's gate on its
held lq box (`tools/release_lq_smoke.py --gate <vendor> <version>`; a box job, never a Mac job, nothing
rented). On the box, `python3 tools/gate_smoke_box.py <vendor> <version> <out> all`:

1. setup: the bench venv from the published PyPI wheels (verified) and every pinned opponent;
2. smoke: the 5 s "does it start" check of EVERY line of the vendor's plan (`--cpu-fallback`: CPU-opponent
   and ours-only lines included);
3. autograd: the autograd smoke entries;
4. adapters: the gate also runs the installed-wheel framework adapter smoke on each vendor (`tools/adapter_gate.py`:
   the Torch and MAX adapters the wheel ships, frameworks installed from the shipped pins in a fresh venv, every
   phase compared byte for byte with the Python API; only PASSED passes, NO_FRAMEWORK fails), on NVIDIA and AMD
   here and on the Mac in the `gate-apple` step (`--device mps`) before `publish-macos`; `MOJOLEARN_RELEASE_ADAPTERS=0`
   skips it with the adapters themselves;
5. measure: one scored run per arm of every plan line, IDENTICAL, one seed, against the peers the plan names
   with the same explicit parameters; opponents already scored on the box are reused from its opponent store,
   only missing ones are scored. Each line's evidence is uploaded to R2 as it finishes, and the lines' boards
   are merged into `measure-*/board/board.json` (put on the rolling main board with
   `tools/main_board_ingest.py --gate-board <box>=<path>`).

The step is done only on verdict PASS (no FAILED smoke line, autograd exit 0, adapter smoke PASSED, every plan line measured, every
opponent pair MATCHED). The
`record` step refuses while a published Linux vendor has no passing gate, so a release cannot skip it. To run
a gate by hand on a held box: `lq add nv CMD main gate-<version>-nvidia 'python3 tools/gate_smoke_box.py nvidia
<version> /root/release-gate/<version>/nvidia/out all'` (and `amd` likewise).

Apples-to-apples is enforced, not reported (Andrew, 2026-10-22): our `identical` mode is the only difference
between a pair. Every opponent pair on the gate board must read MATCHED in its race's parameter check
(`tools/board_pairing.py` `params_state` over the race's `params_check`, read back from each constructed arm).
An opponent whose explicit parameters cannot be read back or differ is refused for that cell with the reason:
the cell shows ours alone, the gate verdict is FAIL, and every refusal is listed in the gate report
(`gate-smoke.json` `measure.pairs`, the verdict reasons, `measure-*/board/pair-check.json`). On every board
(`tools/bench_board.py` save and render, `tools/main_board_ingest.py`) a pair that is not MATCHED shows no
ratio and is never a best opponent: it reads `refused: <reason>`, or `unchecked` for an opponent copied from an
older board that was never re-raced with the check. Copied cells stay visible as `unchecked` until the gate
re-races their line and `--gate-board` replaces them with checked (or refused) cells.
`tools/board_clock_audit.py` exits 1 on any ratio stored from an unchecked or refused pair.

## 7. Publish the docs and the site (docs.mojolearn.org, mojolearn.org)

After the macOS wheel is on PyPI, run the one command (on this Mac; nothing runs a model):

```sh
tools/publish_sites.sh --dry-run   # optional: everything except the deploy
tools/publish_sites.sh
```

It builds from the latest origin/main of both repos in its own worktrees (your checkouts' state does not
matter), documents the latest mojolearn on PyPI, regenerates measurements.json from the origin/main boards,
builds both sites, deploys both with wrangler, and writes a build record (`/build.json` on each site:
version, both commits, UTC time). Release facts are never hand-edited: versions, Python range and GPU
architectures are filled from the release and PyPI on both sites, and the homepage numbers come from the
boards. Only the current release is published; older docs URLs are not kept.


## Verifier-only wheel updates

When only the shipped verifier changes, `packaging/verifier_patch.py --build`
reuses published PyPI wheels and changes only `__main__.py`, `_verify_all.py`,
`_verification_profiles.py` and package version metadata. It preserves every
algorithm implementation, binary, fixture, model, reference and native provenance
member byte-for-byte. A separate `VERIFIER_PATCH.json` records the verifier source
commit and original members. The `verifier-patch` release workflow profile checks
the entire payload against independently downloaded PyPI bases both at admission
and immediately before publication. It does not rerun numerical certification.
Verifier changes still need focused tests; existing numerical receipts describe
the original payload, not a newly executed full run on the repackaged wheel.
