# The release plan

One dry-run decision for a release, with a reason for every scheduled job, enforced in code. The
operating target: change one algorithm, rebuild its actual dependencies, execute its invalidated
checks, reuse everything else, and see exactly why.

    pixi run release 0.8.66 --plan [--plan-json FILE] [--plan-columns nvidia,amd,cpu,ptx,apple]
    python3 tools/release_plan.py show --json <release>/<commit12>/plan.json
    python3 tools/release_plan.py decide --facts plan-facts.json --json plan.json   # the table alone

`--plan` also carries the status report's totals line (`report_totals`, docs/release/STATUS_REPORT.md):
`REPORT <v> passed=.. reused=.. failed=.. missing=.. blocked=.. n/a=.. waived=..  ready=YES|NO`;
`pixi run release <v> --report` writes the full report (one row per ledger requirement and step).

`--plan` runs, rents and publishes nothing. It gathers the facts, applies the decision table
(`tools/release_plan.py decide`, pure) and writes `plan.json` and `plan-facts.json` beside the
release state (`<release root>/<version>/<commit12>/`). It exits 0 when publication is ready, 3
when rows block it.

## What the plan says

| section | one row per | action | reason |
|---|---|---|---|
| `compile` | merged group per platform (`darwin/identical/_mojolearn_native`, ...) | `hit` or `compile` | closure unchanged since the previous release / `closure changed: <files>` / `build inputs changed` / `toolchain pin changed` / new group / the macOS build path moved |
| `identity` | lane x column (nvidia, amd, cpu, ptx, apple) | `reuse` or `record` | the per-column selection of `tools/identity_columns.py changed-lanes` (the same one each recorder and `tools/reference_refresh.sh` runs), or a decision-table override |
| `gates` | release step that checks or produces (gates, producers, packing, publishing) | `reuse` or `run` | recorded for this commit / a box step's own check (`box_step_owed`) / the decision table |
| `restore` | reused row whose evidence is retained but not on this Mac | `restore` | where to restore it from, before any execution |
| `boxes` | vendor | `rent` (with the lease minutes) or `none` | which owed device rows need it; `none: no owed device work: gate-boxes rents nothing` |
| `publish` | blocking row | `ready` or `blocked` | owed work, evidence to restore, unresolved failures |
| `invalidates` | changed file | the groups whose closure holds it | `<file> invalidates these N groups: ...` |
| `waste` | release | counts and seconds | see "Waste metrics" |

## The decision table

The change since the previous release's source commit (`release_reuse.previous_release()`)
decides the class; every path is classified by `release_plan.KINDS` (first match wins).

| class | when | what runs | what is reused |
|---|---|---|---|
| `metadata` | only docs, changelog, version (`python/mojolearn/_version.py`) or tooling changed | metadata and package checks | every compile group; every identity row recorded on or after the previous release; every numerical gate, from the previous release's evidence; build steps take the previous binaries |
| `repackage` | packaging or loader files changed, no product source | artifact byte checks and the changed loading boundary | compile groups, identity rows, numerical gates |
| `algorithm` | product source reaching some groups | the groups whose closure holds a changed file; the lanes whose closure changed | every other group, lane and gate |
| `shared` | product source reaching every group, or the toolchain pin or build recipe changed | every dependent, each with the changed file in its closure named | nothing that depends on it |

Whatever the class:

- **restore**: a reused row whose evidence is not local but is retained (R2, a retained release
  directory) is restored before any execution step; with no copy anywhere it becomes owed work
  (it is never a reuse without evidence).
- **blocked**: a genuine unresolved failure (a failed step not recorded since) blocks publication.
  The plan never waives anything.
- A box step (a gate on the NVIDIA pod or the AMD droplet) decides its own reuse: a gate adopts a
  retained verdict for the same bytes itself (`release_incremental_gate`, `release_gate_reuse`).
  The plan reports what that check says and never claims a reuse the step would refuse.
- A lane owed since before the previous release is still recorded in a `metadata` release:
  the table only spares the reruns this release's change would have caused.

Enforced in `tools/release.py`:

- `gate-boxes` reads this commit's saved plan. When it says `boxes: none` the step refuses to rent
  ("the release plan owes no device work ... gate-boxes rents nothing"), even when a box step
  reports work: that means evidence to restore or a stale plan, never a rental.
- Every publish step refuses while the saved plan names an unresolved failure that is still not
  recorded, or evidence to restore that is still not on this Mac (`plan_blocks_publication`),
  before the ledger's own pre-publication gate.
- The plan launches nothing. A real run still follows the step graph; the plan is the reasoned
  forecast and the list of evidence publication is gated on.

Tests: `tools/test_release_plan.py` (one fake-input test per table row, the box and publish hooks,
waste.json, interruption recovery).

## The macOS release build runs from one stable checkout path

`repo_path` is a closure-v2 key field of every bincache key (`tools/bincache.py` key_fields): the
binaries embed absolute paths, so a group built in another directory is another key. 0.8.64 was
built in `~/mojolearn-wt/integ-0864` and 0.8.65 in the main checkout
(`/Users/andrewhendel/CascadeProjects/mojolearn`, read from its retained
`macos-bincache/provenance/keys/*.json`): every macOS group missed. Rules:

- Build every macOS release from the same path: the main checkout at the source commit (what
  `Release.source_checkout` uses when HEAD is the source commit). A release frozen elsewhere uses
  `<release root>/<version>/source/<commit12>/`, a new path per release: it compiles every macOS
  group once, and the plan says so.
- The plan compares the path this release builds at with the previous release's retained key
  `repo_path`; when they differ every darwin group is `compile` with both paths named
  (`release_plan.compile_rows`, `macos_paths`).

## The development build of the frozen commit is the macOS wheel's compile

Andrew's rule: one compile per freeze. On 2026-11-07 the Mac compiled 0.8.66 (`bc4ecd4518`) twice:
`tools/development_build.py build --sets identical,host` (79 development groups, in
`~/mojolearn-wt/integ-perf`), then `tools/release.py` step `macos-build` again (15 merged groups,
8,582 s wall, 25,700+ compiler seconds, every group `miss+built-cached` in
`<release>/macos-bincache/`, seeded only from earlier releases). The two could not share a key,
whatever the directory:

| key field | development layout (before) | macOS wheel (`build_release_wheel.sh`) | changes the compile? |
|---|---|---|---|
| key system, cache | `development_build.py` KEY_SCHEMA, `~/.cache/mojolearn-development` | `bincache.py` closure-v2, release-scoped `macos-bincache/` | two disjoint caches |
| groups | `bindings/development_groups.json`: 79 `_mojolearn_dev_*`, one member each | `bindings/merged_groups.json`: 15 `_mojolearn_native*`, 1 to 10 members | yes: other compile units |
| merged root text (`merged.root_sha256`) | one member's `PyInit` | every member's `PyInit` | yes |
| `merged_build.sh` args | `gpu\|host --layout development --group S --outdir --rootdir` | `gpu\|host --group S` | yes (layout); the paths are placeholders |
| declared output (`local.outputs`) | `<cache>/work/<key>/output/S.so` | `python/mojolearn/<set>/S.so` | keyed |
| `MOJOLEARN_COMPILE_JOBS` | the scheduler's share of free cores | `1` (`mac_slot.py run` pins it) | keyed: a codegen split is not shown byte-neutral |
| `repo_path` | where the build ran | the release's source checkout | yes: absolute paths are embedded |
| build environment | v1 sweep, also `MOJOLEARN_MERGED_PYTHON`, `MOJOLEARN_RUNPOD_KEY_FILE` | only what `merged_build.sh`, `build_defines.sh`, `compile_jobs.sh` read | no: closure-v2 already drops both |
| `MOJOLEARN_PACKAGE_BYTE_LM` | `1` | `1` | equal (not a difference) |
| numeric mode, host column, Metal flags | identical; cpu, no accelerator; per flags class | the same | equal |

None of the differing fields is a field no input reads, so no key field is dropped: the fix makes
the compile the same compile. Since lane `post-one-mac-cache`:

- On macOS `tools/development_build.py build` defaults to `--layout release`: it runs every merged
  release group of the selected sets as `tools/bincache.py build ./bindings/merged_build.sh <kind>
  --group <stem>` with the wheel build's per-group environment (`bincache.macos_merged_group`) and
  the release's keyed settings (`bincache.MACOS_RELEASE_COMPILE_ENV`: one compiler worker, byte LM),
  into ONE Mac cache: `--mac-cache`, else `MOJOLEARN_BINCACHE_DIR`, else
  `~/mojolearn-evidence/bincache-mac/`. Groups the cache holds are placed in parallel; the rest
  compile under the memory-packed scheduler (groups in parallel up to the cores and 80% of available
  memory, recorded peaks). Each set's `_mojolearn_merged.json` is then written from the plan, as the
  wheel build writes it. A different `MOJOLEARN_COMPILE_JOBS` or `MOJOLEARN_PACKAGE_BYTE_LM=0` is
  refused, not overridden. `--layout development` keeps the per-binding groups (the Linux default).
- `tools/release.py` step `macos-build` keeps its release-scoped cache (the admitted provenance),
  seeds it from the one Mac cache first (then earlier releases), and after the build copies every
  archive it holds back into the one Mac cache (verified; a key that holds other bytes is a
  conflict and is never replaced).
- A placed group is that group's compile evidence (lane `post-admit-cache-hit`). The
  `macos-release-build` admission (`tools/release_requirements_macos_build.py`) takes a group the
  build log names `miss+built-cached` (compiled now) or `hit` (placed). For both, the closure-v2
  key's source closure, executed inputs, merged membership and build environment reproduce from the
  frozen commit. A placed group also needs the release cache's `hit` row naming the archive's
  bytes, the archive's own record of its compile (`source_commit`, `built`, `build_seconds`,
  `builder`; `bincache.py` writes it into every local archive), the hit's `placements.jsonl` line
  naming the same compile, and the key's closure reproducing at that `source_commit`. Anything
  missing, or a key that does not reproduce, refuses. The admission facts list
  `compiles: {built, placed}` and each group's `compile` (revision, date, seconds, builder), so the
  evidence summary can say per group when it was compiled and that it was placed.

To get the hits, run the development build of the frozen commit in the checkout the release will
build in (`repo_path` is keyed; `Release.source_checkout`: the main checkout when its HEAD is the
source commit, else `--source-checkout`), from the same shell environment class (no extra
`MOJO_*`/`MODULAR_*` or keyed `MOJOLEARN_*` variables):

    cd <release source checkout>   # HEAD = the frozen commit
    bash tools/compile_slot.sh python3 tools/development_build.py build --sets identical,host
    pixi run release <version> ...  # macos-build: `macos identical/<stem>: hit (...)` per group

The `macos-build` step prints `hit` or `compile: <reason>` per group before it starts
(`macos_cache_plan`), naming the field that differs. Expected macOS wheel wall when the development
build already compiled every group: about 3 minutes, against 8,582 s on 2026-11-07. Basis: that
build's log spans 17:59:53Z to 20:22:55Z with compiles from 18:00:18Z to 20:20:52Z, so about 150 s
went to things other than compiling (planning, index, readbacks, staging, packing, the audit). The
15 placements add about 20 s (key and archive verify, four at a time).

## Temporary dependency worktrees

`tools/release_requirement_dependencies.py` analyses a frozen revision in a temporary worktree
(`<tmp>/mojolearn-dependencies-*/tree`). Seven of them stayed registered after 0.8.65 (27 are listed
in the main checkout now): its `finally` never runs when the process is killed. Now:

- every tree is registered (`~/mojolearn-evidence/release-dependencies/worktrees/<id>.json`: pid,
  root, tree, created; `MOJOLEARN_DEPENDENCY_WORKTREES` overrides) as soon as `worktree add`
  succeeds, and unregistered after its removal;
- an atexit hook and a SIGTERM/SIGHUP handler remove this process's live trees;
- `python3 tools/release_requirement_dependencies.py --gc-worktrees [--root R] [--max-age-hours 24]
  [--dry-run]` removes stale ones: under a `mojolearn-dependencies-*` directory, older than a day,
  their registering process gone. A tree whose directory is already gone loses only its own
  `.git/worktrees/<id>` (matched by its gitdir file): the targeted form of `git worktree prune`,
  which would also prune other sessions' missing trees. Nothing under `~/mojolearn-wt/` is touched.

Tests: `tools/test_release_dependency_worktrees.py` (real git, a SIGTERM'd process included).

## Understandable misses (`tools/bincache.py explain`)

    python3 tools/bincache.py explain --since <previous release commit> bindings/merged_build.sh gpu --group <stem>
    MOJOLEARN_BINCACHE_KEY_RULE=v1 python3 tools/bincache.py explain bindings/build_<x>.sh

- `--since REV` prints, per `.mojo` file changed since REV, `<file> invalidates these N groups:
  darwin/identical/..., linux/...` (or `invalidates no merged group`), then the usual verdict.
  The plan carries the same map (`invalidates`).
- Closure-v2 keys never widen: an unresolved import fails the key (`compile: unkeyable: unresolved
  import 'x' in f`, exit 3). The legacy (v1) key, still recomputed by the retained-compile
  verifiers, widened silently; with `MOJOLEARN_BINCACHE_KEY_RULE=v1` explain prints
  `legacy key: scope=tree ...; unresolved import 'x' in <file> widened the closure`
  (`bincache.WIDENED`, kept outside the key fields so v1 keys are unchanged).

## Waste metrics (`waste.json`)

Minimal: the numbers the plan already has plus step times.

- **Release**: `<release root>/<version>/<commit12>/waste.json` after every real run
  (`Release.write_waste`): groups reused vs rebuilt (with seconds when known), lanes reused vs
  recorded (from the saved plan), `steps` (seconds per step; `run_step` now records `seconds` in
  every step record), `box_wait_seconds` (the gate-boxes step) and `admission_seconds` (the record
  step). `--plan` prints a two-line summary and the delta against the newest earlier version's
  `waste.json` (`release_plan.previous_waste`).
- **Reference refresh**: `<REFRESH_HOME>/<sha12>/waste.json` after each admission: lanes reused vs
  recorded per column, this invocation's seconds planning, waiting on boxes and admitting.

## ONE LAYOUT: ship the development layout? (go/no-go, measured)

Andrew, 2026-11-05: "maybe just use the dev layout for the release? downside slightly bigger
packages?" Measured statically, without a compile, by `tools/one_layout.py measure` from the retained
0.8.65 release (`~/mojolearn-evidence/release/0.8.65/6519a54c6579`: its five Linux wheels, its macOS
wheel and its `macos-bincache` merged artifacts), the Mac's development cache
(`~/.cache/mojolearn-development/artifacts`: all 33 identical members and 13 of the 46 host members,
built 2026-10-29), the closures at 9ee5208756, the per-project limits in
`packaging/pypi_limits.json` and each project's bytes on PyPI on 2026-11-04. Full output:
`~/mojolearn-evidence/release-planner-notes/one-layout.json`.

**Recommendation: HOLD (not yet GO).** Every Linux wheel fits with room (largest pessimistic
estimate 80 MiB of 250 for `mojolearn-nvidia`); the macOS wheel would be 86.0 MiB (central) to
110.1 MiB (pessimistic) against `mojolearn`'s 100 MiB. Switch when one Mac development host build
(the 33 unmeasured host members) puts the macOS estimate under 100 MiB, or when pypi/support#12773
grants 250 MiB; then ship the development layout, retire the merged build, one cache for lanes
and releases, and re-record every lane on every vendor once.

### Size per wheel against its project's limit

Development estimate = the retained wheel with each set's compressed native bytes scaled by that
set's measured development/merged ratio (zlib 6 of the unstripped artifacts of both layouts).
`central` fills each unmeasured member with the median measured member; `pessimistic` grows each
group with unmeasured members (and every Linux group, whose grouping differs from the Mac's) by
the worst ratio measured on a fully measured group (2.48, `_mojolearn_native_sequence`).

| wheel (0.8.65) | project | limit | merged (shipped) | development central | development pessimistic |
|---|---|---:|---:|---:|---:|
| macOS `mojolearn-...-macosx_26_0_arm64.whl` | mojolearn | 100 MiB | 64.3 MiB | 86.0 MiB | **110.1 MiB** |
| `mojolearn_nvidia` (sm_89) | mojolearn-nvidia | 250 MiB | 33.1 MiB | 46.1 MiB | 80.0 MiB |
| `mojolearn_nvidia_ptx` (sm_80) | mojolearn-nvidia-ptx | 100 MiB | 30.3 MiB | 42.7 MiB | 74.7 MiB |
| **next: one `mojolearn_nvidia` (sm_89 + PTX sm_80)** | mojolearn-nvidia | 250 MiB | 63.4 MiB (sum) | 88.8 MiB (sum) | 154.7 MiB (sum) |

From the release after 0.8.65 the PTX set packs into `mojolearn-nvidia` (Andrew 2026-11-05); the
pack's size check prints both sets per wheel against the 250 MiB limit. The sums above are an upper
bound: the two sets share no file, so the combined wheel is about their sum.
| `mojolearn_amd` (gfx942) | mojolearn-amd | 250 MiB | 24.7 MiB | 34.4 MiB | 59.5 MiB |
| `mojolearn_core` (host) | mojolearn-core | 100 MiB | 32.4 MiB | 40.4 MiB | 72.3 MiB |
| Linux meta `mojolearn` | mojolearn | 100 MiB | 0.2 MiB | 0.2 MiB | 0.2 MiB |

Measured ratios (Mac): identical 1.413 (33 of 33 members; per merged group 1.03 to 2.48), host
1.297 central (13 of 46 members; `runtime_classical_host` 23.4 MiB and `tokenizer_host` 24.7 MiB
alone are as large as their merged groups). The largest single native file is unchanged in kind
(24.7 MiB development vs 26.4 MiB merged): no single file is near a limit; the wheel is the unit.

Project size (PyPI 10 GiB default; #12774/#12775 open): `mojolearn` holds 5.74 GiB, so 67 more
releases fit at today's 64.5 MiB per release (macOS + meta) and 50 at the central 86.2 MiB.
`mojolearn-nvidia` 2.25 GiB (239 -> 171 releases), `mojolearn-amd` 1.42 GiB (355 -> 255),
`mojolearn-core` 0.27 GiB (307 -> 246), `mojolearn-nvidia-ptx` 0.43 GiB (323 -> 229).

### Load cost

| | merged (release) | development |
|---|---:|---:|
| native files per platform (identical + host) | 13 Mac / 9 Linux | 79 / 79 |
| dlopens for a full import (every member loaded) | 13 / 9 | 79 / 79 |
| mapped native bytes, Mac identical set | 112.8 MiB | 153.5 MiB (+36%) |
| loader index (`_mojolearn_merged.json`, all sets) | 2.5 KB | 4.8 KB |

`python/mojolearn/_merged.py` loads a member lazily from the file its index names, so a typical fit
opens its owner's file (and its host owner's): one or two smaller files in the development layout,
one or two larger ones in the merged layout. A full import pays 79 dlopens instead of 9 to 13;
its time has not been measured (RUN OWED below). The Mojo runtime libraries (`.dylibs`/`.libs`)
are shared either way, so no runtime global is duplicated.

### Compile scope per change (closure files, commit-weighted since 2026-10-04)

| platform/set | merged groups | development groups | full build, merged -> development | expected files per change, merged -> development |
|---|---:|---:|---:|---:|
| darwin/identical | 8 | 33 | 2,796 -> 4,684 | 707.9 -> 705.8 |
| darwin/host | 5 | 46 | 2,857 -> 4,600 | 1,063.1 -> 799.7 |
| linux/identical | 4 | 33 | 2,266 -> 4,684 | 767.8 -> 705.8 |
| linux/host | 5 | 46 | 2,857 -> 4,600 | 1,063.1 -> 799.7 |

A change compiles about as many files in either layout (identical) or 25% fewer (host); a cold
build reads 1.6 to 2.1 times as many. The saving is not a smaller compile per change: it is that
the release stops compiling a second time.

### Dev-to-release build gap (deliverable c)

Closed on macOS since lane `post-one-mac-cache`: the development build there compiles the release
layout itself ("The development build of the frozen commit is the macOS wheel's compile" above).
The analysis below is the development layout's, which Linux still uses.

`tools/development_build.py` caches one artifact per development group
(`bindings/development_groups.json`, one logical owner each) under its own key schema; the release
builds `bindings/merged_groups.json` groups through `bindings/merged_build.sh` and `tools/bincache.py`.
No compiled unit is shared: the smallest merged group has two members (`_mojolearn_native_trees_m1`,
`_mojolearn_native_sequence_m1`) and every development unit has one, so the two layouts never have
the same root text or key. Every release compile therefore repeats work a lane's development build
of the same commit already did: per changed file about 1,771 closure files on the Mac (708 identical
+ 1,063 host) and 1,831 per Linux leg, on top of the development build's 1,506; a cold release
repeats 5,653 (Mac) and 5,123 (Linux leg) closure files. Measured seconds: the 0.8.65 macOS merged
build took 709 s for 13 groups (0.8.64: 12,863 s). The development receipts' seconds were taken on
a loaded Mac under memory pressure (`runtime_classical_host` 2,150 s, paged) and are not compared.

Reuse across the two layouts would need a supported step that links per-owner compiled units into
one merged extension library (each `PyInit_<member>` and the shared Mojo code in one image). The
release builds one shared library per `mojo build` root and no such link path is documented or
qualified, so nothing is unified beyond what the compiler supports (no workaround; if the merged
layout is kept, the ask for Modular is a documented object-level link of Mojo extension modules). The loading boundary
that differs is the loader index (`_mojolearn_merged.json`, one file per member instead of one per
group), qualified by the existing index check (`tools/merged_bindings.py check-index`) and the
installed-wheel self-test (`python -m mojolearn verify --self-test`).

### Can the grouping move bits?

Unknown. One `mojo build` is one compilation unit; the compiler documents no guarantee that a
function compiles to the same machine code in a different unit (inlining and floating-point
contraction decisions can depend on what else the unit holds). So the switch re-records every lane
on every vendor together once (428 lanes in the NVIDIA column at 0.8.65), which the identity rules allow
within a version; nothing else about identity changes.

### The switch (not implemented here: not contained)

The switch touches the release builders on both platforms, the packers, the loader index, three
retained-evidence verifiers and a shipped runtime file. When the size condition above holds:

1. `tools/merged_bindings.py`: the release layout reads `bindings/development_groups.json`
   (`plan(..., layout="development")` becomes the release plan); `check-index` checks that index.
2. `bindings/merged_build.sh` + `tools/development_build.py`: one builder and one key schema; the
   release compiles the development groups through it, and its artifacts land in the one cache
   (development cache keyed by closure, the R2 partitions for Linux).
3. `packaging/macos/build_release_wheel.sh` (`merged_plan`, the index check before packing) and
   `tools/release_macos_assembly.py` (`INDEX`, group rows).
4. `packaging/linux/build_sets.sh` (pool jobs, `reindex_merged`) and `packaging/linux/pack_wheel.py`
   (`layout` "merged" check, native globs; stems `_mojolearn_dev_*` either ship as is or are renamed
   once, a one-time key change).
5. `tools/development_build.py`: drop "a development index deliberately fails the release layout
   check".
6. `tools/release_reuse.py` (`MERGED_INDEX` per set) and `tools/release_host_composition.py`
   (`host_files`).
7. The four that hard-code group names: `tools/release_github_build.py` `UNIT_PEAK_GIB` (one
   measured build of the new units first), `python/mojolearn/_verification_light_reference.py`
   (witnesses by group file name; a runtime file, so the switch release re-records anyway),
   `tools/qualify_release_host_boundary.py` and `tools/release_incremental_gate.py`
   (`_mojolearn_native_tokenizer_host`).
8. Retire the merged-group evidence readers' expectations (`tools/apple_retained_groups.py`,
   `tools/release_ptx_assembly.py`, `tools/release_linux_group_composition.py`): one rebuild.
9. `tools/release.py` `step_macos_build`: seed from the development cache, not the earlier
   releases' `macos-bincache`.
10. Docs: a TOMBSTONE-style note here and in `docs/build/COMPILE_CLOSURES.md` for the merged build
    mode (what it was, the last release that shipped it, the commit that retired it). It is a
    build mode, not a code experiment: no `experiments/removed/` patch.
11. Re-record every lane on every vendor once (one reference refresh with the full scope).

## Interruption recovery: what resumes after a disconnected controller

| step | resumes? | how, or what is owed |
|---|---|---|
| macOS compile (`macos-build`) | yes at group granularity | runs in the controller's foreground. Fixed here: a real run ignores SIGHUP (inherited by the build) and `say` drops output it cannot print, so a closed terminal no longer kills it. A killed controller loses only the group in flight: each finished group is a bincache archive and hits on the rerun. OWED: run it detached like the Linux legs (pid, exit and provenance files). |
| Linux legs (`linux-builds`) | yes | detached (`start_new_session`) with pid, exit and provenance files; `leg_status` adopts a running or finished leg. |
| cross-compile (GitHub) | partly | a rerun dispatches a new workflow run (a new token per run) instead of re-attaching to one in flight; a finished run is reused (`cross_compile_reuse`). OWED: save the dispatch token in the state before dispatching and re-attach to it. |
| GPU execution (gates, records) | yes | box jobs persist to R2 from the box (`tools/box_persist_run.sh`); the tags are in the logs and `boxes-down` fetches them. OWED: the hold and lease renewal (`BoxKeeper`) is a thread of the controller; while the controller is down nothing renews the AMD lease (240 min) or holds the NVIDIA pod (deleted 60 idle minutes after its last job), so a long job can outlive its box. Renewal outside the controller is not small. |
| upload (`publish-*`) | yes | each publish checks the index first ("already on PyPI", the plugin marker), so a rerun never uploads twice. |
| admission (reference refresh) | yes | every decision lands in its `state.json` before the next step; a rerun fetches, waits or admits (`tools/reference_refresh.py`, IDEMPOTENT). |
| admission (release `record`) | reruns the step | not resumable mid-step; the step reruns from its start. Not verified here that every admission it runs is idempotent: OWED. |

## RUN OWED (orchestrator only)

- One Mac development host build (`tools/development_build.py build --sets host`, the 33 members not
  in the cache) and `tools/one_layout.py measure` again: decides the macOS row of ONE LAYOUT.
- Load time of the development layout (import + first fit) on Apple, NVIDIA and AMD.
- `python3 tools/release_requirement_dependencies.py --gc-worktrees --root <main checkout>` after
  the 0.8.65 publish (27 dependency worktrees are registered there now; never during a publish).
- A first `pixi run release <next> --plan` on the integration branch, to compare its rows with what
  the release then runs.
