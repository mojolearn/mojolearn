# Ops tools (copies; git is the source of truth)
lq and compile_slot.sh were lost when ~/mojolearn-evidence top-level files were wiped (2026-10-05 ~23:46) and restored from the T9 backup.
Live copies: ~/mojolearn-evidence/lq/lq and ~/mojolearn-ops/. compile_slot.sh now lives at tools/compile_slot.sh (2026-10-28); tools/ops/compile_slot.sh and ~/mojolearn-evidence/compile_slot.sh forward to it. Never keep the only copy of a script in a logs or evidence folder.

## Incremental development builds

Internal engineering path: `tools/development_build.py` builds independent
logical binding owners from `bindings/development_groups.json`: 33 configured
`identical` units (including optional byte-LM) and 46 host units. Each native type
keeps one owner; classical, model selection, manifold and Sequential
orchestration have independent runtime entries. Target/platform
selection can omit an optional unit.
Release scripts keep `bindings/merged_groups.json` and its compact merged files.
The development layout's native compilation, public runtime behavior and
Apple/NVIDIA/AMD identity still require qualification. Controller tests do not
establish numerical equivalence or a compile-time speedup.

The compact release host group includes all four runtime builders as logical
modules, so release imports use the same native entries. Classical, model-selection, manifold and
Sequential implementation edits invalidate their own development units; shared array/admission/error
helpers still invalidate every owner that imports them. Cold development builds
duplicate common code across units, so their time, peak memory and total binary
size must be measured separately from the compact release build.

The classical owner registers `TreesResident` once and contains its allocation,
use and destruction. Import setup aliases its original native entries and type
onto the legacy tokenizer module; no Python operation router is installed.
Tokenizers built before this ownership split are refused before the second type
owner loads. Use a complete matching host layout and a fresh process. If the new
owner is absent, legacy operation names retain native missing-owner refusals;
the small stateless attribute-descriptor helper remains available for metadata
and optimizer import setup. Adding a missing binary requires a fresh process.

Run from an isolated development worktree. Install its pinned default Pixi
environment first. Planning performs no compilation or activation:

```sh
python3 tools/development_build.py plan --sets host --json /path/to/host-plan.json
python3 tools/development_build.py plan --sets identical,host --json /path/to/plan.json
```

Linux GPU planning/builds additionally need one explicit target, for example
`MOJOLEARN_GPU_ARCHS=sm_89` or `MOJOLEARN_GPU_ARCHS=gfx942`. The host plan removes
that GPU target and selects the CPU column. Apple uses the existing compiler
flag classes; it needs no target override. There is no development `fast` lane.

Compiler workers default to eight for Linux AMD, two for Linux NVIDIA/host,
and one on macOS. `MOJOLEARN_COMPILE_JOBS` overrides all of these;
`MOJOLEARN_AMD_COMPILE_JOBS` overrides the AMD default when the general override
is absent. The effective count is part of each cache key. `--jobs` instead
controls concurrent library compilations; its CPU reservation uses the largest
compiler worker count among the selected pending groups.

The orchestrator builds on the Mac through the existing semaphore:

```sh
bash ~/mojolearn-evidence/compile_slot.sh \
  python3 tools/development_build.py build --sets identical,host \
  --evidence ~/mojolearn-evidence/development-builds/my-run
```

Repeat that command in the **same worktree and cache location** after editing.
Unchanged groups reuse their retained binaries. Repository and cache paths are
part of the key because they can reach compiler/linker output; moving to a new
worktree intentionally does not claim the old binaries are reusable. Keys also
include transitive sources, generators, actual compiler/stdlib package bytes,
locked dependencies, flags and target. An unresolved import records
`tree-fallback`, widening invalidation rather than guessing it is unrelated.
`--cache` defaults to `~/.cache/mojolearn-development` and must be outside the
source tree. Full logs, plan and per-group receipts remain under `--evidence`.

`--group trees` or `--group host:runtime_model_selection` limits which missing groups compile.
If another required group is stale or absent, the selected artifacts are kept
in cache but activation returns **3** and preserves the existing loader indexes.
Omit `--group` to build all missing groups. Successful activation stages
immutable files under each set's `.development/<key>/` directory and replaces
its complete loader index atomically. Start a fresh Python process afterward;
already loaded native modules are not hot-reloaded. Remove or relocate explicit
per-binding `.so` overrides first: the controller refuses them because they
would shadow the development index. Use one build controller for the worktree;
do not run a release build there concurrently.

`--jobs N --memory-per-job-gib G` allows parallel groups once `G` is a measured
peak for this source and target. The pool respects available memory and CPU
allocation, including Linux cgroups. `MOJOLEARN_COMPILE_JOBS` controls workers
inside each compiler and defaults to one. Keep the existing Mac semaphore.

To return this worktree to release layout, rebuild the selected complete sets
using the original builder (and the same Linux target when applicable):

```sh
MOJOLEARN_NUMERIC_MODE=identical MOJOLEARN_PACKAGE_BYTE_LM=1 \
  bash ~/mojolearn-evidence/compile_slot.sh sh bindings/merged_build.sh gpu
MOJOLEARN_NUMERIC_MODE=identical MOJOLEARN_PACKAGE_BYTE_LM=1 \
  bash ~/mojolearn-evidence/compile_slot.sh \
  env -u MOJOLEARN_GPU_ARCHS MOJOLEARN_TARGET_COLUMN=cpu sh bindings/merged_build.sh host
```

The complete rebuild replaces all development member mappings. Do not merely
point the index at an old release binary: its sources may have changed. Hidden
development artifacts can remain for reuse; existing release package-data globs
do not include them, and release layout checks refuse a development index.
Final releases still go through the existing release builder and qualification
gates. Plan/build exit **0** means controller success, **1** means failure, and
**3** means an incomplete selected build with no activation; it is not a release
qualification result.

## Scripts adopted from ~/mojolearn-evidence (2026-10-28)

Scripts that current rules, tools or docs still reference were copied into the repo so they are versioned.
The originals stay in place; edit the repo copy first.

| Repo path | Origin | Purpose |
|---|---|---|
| `tools/compile_slot.sh` | `~/mojolearn-evidence/compile_slot.sh` | Mac compile semaphore: 4 machine-wide slots (mkdir locks under `~/mojolearn-evidence/compile-slots`), `nice -n 19`; refuses dedicated Apple `fast` builds unless `MOJOLEARN_INTEG_BUILD=1`. The origin path is now a shim that execs the main checkout's copy (falling back to `~/mojolearn-evidence/compile_slot.legacy-20261028.sh` until this lands on main). `tools/ops/compile_slot.sh` forwards to it. |
| `tools/ops/archive_all_boxes.sh` | `~/mojolearn-evidence/archive_all_boxes.sh` | Archives every live rented box (from `devpods/*/state.env` and `do-amd-steward/state.env`) through `tools/box_archive.sh`, then prunes R2-confirmed local copies to 3 per box. Named in `tools/box_archive.sh`. Its launchd job is not installed at present. |
| `tools/ops/lq` | `~/mojolearn-ops/lq` (refreshed 2026-10-28 from the 2026-10-24 live copy) | Box job queue CLI named in CLAUDE.md. `~/mojolearn-ops/lq` stays the live copy. |
| `tools/apple_fast_ops/apple_watch.sh`, `tools/apple_fast_ops/m2_bq.sh` | `~/mojolearn-evidence/` | Already versioned, byte-identical; nothing copied. |

### Not adopted (one-off)

Not named by CLAUDE.md, AGENTS.md, current tools or the current briefs, or superseded by a repo tool:

- `archive_box.sh`: hard-coded addresses of released boxes; superseded by `tools/box_archive.sh`.
- `box-watch.sh`, `m2compile.sh`: named only in Oct 3-4 state notes.
- One-off release chains, watchers, wave and build helpers: `algos-trees-push.sh`, `anc2-build.sh`, `anc2-record.sh`, `build-trees-io-all.sh`, `cloudmac-newmac-bootstrap.sh`, `cloudmac-release-2116.sh`, `err1.sh`, `errs.sh`, `gfs_syms.sh`, `hosts_readonly_check.sh`, `hotaisle_stock_watch.sh`, `lanes.sh`, `lean_worktree-identical-experiments.sh`, `lease_steward.sh`, `mac_release.sh`, `ni_at.sh`, `night_chain.sh`, `nmu-compile.sh`, `nomatrix_filter.sh`, `ols-admit-run.sh`, `pull.sh`, `r2ls.sh`, `ref-all-host.sh`, `release-0837-chain.sh`, `release-0837-chain10.sh`, `release-0837-chain2.sh`, `release-0837-chain3.sh`, `release-0837-chain4.sh`, `release-0837-chain5.sh`, `release-0837-chain6.sh`, `release-0837-chain7.sh`, `release-0837-chain8.sh`, `release-0837-chain9.sh`, `release-three-2116.sh`, `release-ultra-0900.sh`, `release-ultra-when-idle.sh`, `release_0839.sh`, `release_nv_when_idle.sh`, `repo_audit.sh`, `rerun_failed_20261010.sh`, `sm_fix_test.sh`, `sm_pin_experiment.sh`, `stage_softmax_mcd.sh`, `stop_compiles.sh`, `symhist_builds.sh`, `symiter_builds.sh`, `thin-neural-count.sh`, `watch_release_0839.sh`, `wave-v2-identity-summary.sh`, `wave-v2-stage-harness.sh`, `wave-v2-timing.sh`, `wave-v2-wait-all.sh`, `wave-v2-wait-identity.sh`, `wave-v2-xv.sh`, `wip_snapshot_loop.sh`, `wt-reclaim.sh`, `wt_idle_pixi_scan.sh`.
