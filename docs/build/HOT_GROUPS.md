# Hot compile groups: what each merged group costs, the guard split and the host split

Lane split-hot-groups, 2026-11-04, on integ/tooling-20261104 at 9ee5208756. Everything here is static: the
retained release receipts under `~/mojolearn-evidence/release/*/` (fields only), the closure-v2 import walk
(`tools/compile_closures.py`) and `git log`. Nothing was compiled. Follows docs/build/COMPILE_CLOSURES.md
(closure-v2 keys), whose "recommended order" this lane carries out: split the hot shared file, then split the
compiler-limited groups selectively instead of a wholesale regroup.

Scripts and raw outputs: `~/mojolearn-evidence/split-hot-groups-notes/` (collect_receipts.py, group_table.py,
host_clusters.py, define_reach*.py, split_guards2.py, write_modules.py, edit_bindings.py, history_attrib2.py,
scope.py and their JSON).

## 1. Per merged group: retained compile time, peak memory, closure, change frequency

Sources: Linux = every GitHub shard of the 0.8.63, 0.8.64 and 0.8.65 legs (`legs/<leg>/GHA/jobs.tsv` names the
shard's group, `shards/<n>/build_logs/<set>_merged_<group>.log` its `BINCACHE ... seconds=` compile line,
`shards/<n>/memory.txt` the cgroup peak); macOS = every `macos-build.log` (`BINCACHE` line after
`== merged <set>/<group>.so`; no peak is recorded on the Mac). Misses only (hits excluded). Closure and commit
counts are at 9ee5208756 (closure-v2 walk; commits since 2026-10-04 that touched any file of the closure,
out of 4,830 `.mojo` commits).

| platform | set | group | members | closure files | commits since 2026-10-04 touching the closure | retained builds | compile s min / median / max | peak GiB max (median) |
|---|---|---|---:|---:|---:|---:|---|---|
| darwin | identical | `_mojolearn_native` | 3 | 320 | 593 | 15 | 28 / 63 / 176 | not recorded |
| darwin | identical | `_mojolearn_native_m1` | 6 | 539 | 673 | 16 | 53 / 96 / 912 | not recorded |
| darwin | identical | `_mojolearn_native_trees` | 8 | 564 | 703 | 16 | 51 / 99 / 612 | not recorded |
| darwin | identical | `_mojolearn_native_trees_m1` | 2 | 92 | 273 | 16 | 12 / 21 / 118 | not recorded |
| darwin | identical | `_mojolearn_native_sequence` | 5 | 340 | 813 | 16 | 40 / 79 / 339 | not recorded |
| darwin | identical | `_mojolearn_native_sequence_m1` | 2 | 318 | 976 | 16 | 32 / 58 / 618 | not recorded |
| darwin | identical | `_mojolearn_native_cluster` | 3 | 236 | 446 | 16 | 20 / 42 / 164 | not recorded |
| darwin | identical | `_mojolearn_native_cluster_m1` | 4 | 387 | 701 | 16 | 34 / 58 / 481 | not recorded |
| darwin | host | `_mojolearn_native_host` | 7 | 702 | 1475 | 16 | 112 / 217 / 3925 | not recorded |
| darwin | host | `_mojolearn_native_tokenizer_host` | 6 | 741 | 1928 | 16 | 99 / 182 / 3973 | not recorded |
| darwin | host | `_mojolearn_native_neural_host` | 9 | 421 | 1046 | 15 | 58 / 110 / 606 | not recorded |
| darwin | host | `_mojolearn_native_forest_host` | 15 | 440 | 581 | 16 | 30 / 58 / 188 | not recorded |
| darwin | host | `_mojolearn_native_x_host` | 9 | 555 | 1142 | 16 | 46 / 95 / 751 | not recorded |
| linux | identical | `_mojolearn_native` | 9 | 660 | 793 | 15 | 332 / 508 / 574 | 13.67 (10.32) |
| linux | identical | `_mojolearn_native_trees` | 10 | 609 | 720 | 15 | 346 / 579 / 656 | 12.95 (10.18) |
| linux | identical | `_mojolearn_native_sequence` | 7 | 511 | 1208 | 15 | 434 / 693 / 802 | 11.58 (8.21) |
| linux | identical | `_mojolearn_native_cluster` | 7 | 486 | 746 | 15 | 227 / 399 / 436 | 13.11 (9.29) |
| linux | host | `_mojolearn_native_host` | 7 | 702 | 1475 | 13 | 1146 / 1563 / 2281 | 14.75 (14.27) |
| linux | host | `_mojolearn_native_tokenizer_host` | 6 | 741 | 1928 | 12 | 1039 / 1397 / 2106 | 14.33 (13.97) |
| linux | host | `_mojolearn_native_neural_host` | 9 | 421 | 1046 | 13 | 348 / 512 / 577 | 11.67 (7.51) |
| linux | host | `_mojolearn_native_forest_host` | 15 | 440 | 581 | 13 | 107 / 153 / 212 | 9.01 (4.92) |
| linux | host | `_mojolearn_native_x_host` | 9 | 555 | 1142 | 13 | 320 / 410 / 491 | 11.58 (7.54) |

The two host groups `_mojolearn_native_host` and `_mojolearn_native_tokenizer_host` are the outliers on every
axis: 3 to 10 times the compile seconds of the 440-file forest group, the highest peaks (14.75 / 14.33 GiB on a
16 GiB runner) and the most-changed closures (1,475 / 1,928 commits).

## 2. Closure clusters of the two largest host groups

Per member: closure files, commits touching it, files no other member of the group reads.

| group | member | files | commits | own files |
|---|---|---:|---:|---:|
| `_mojolearn_native_host` (702) | `_mojolearn_runtime_classical_host` | 442 | 1,210 | 391 |
| | `_mojolearn_estimators_host` | 191 | 466 | 69 |
| | `_mojolearn_metrics_host` | 106 | 312 | 34 |
| | `_mojolearn_linalg_host` | 100 | 348 | 9 |
| | `_mojolearn_solver_host` | 85 | 350 | 25 |
| | `_mojolearn_core_host` | 74 | 328 | 14 |
| | `_mojolearn_preprocessing_host` | 15 | 238 | 2 |
| `_mojolearn_native_tokenizer_host` (741) | `_mojolearn_tokenizer_host` | 655 | 1,906 | 626 |
| | `_mojolearn_tsa_host` | 57 | 251 | 15 |
| | `_mojolearn_forecast_host` | 53 | 254 | 1 |
| | `_mojolearn_x_metrics_host` | 39 | 238 | 22 |
| | `_mojolearn_resample_host` | 27 | 220 | 10 |
| | `_mojolearn_arima_host` | 22 | 217 | 1 |

Each group is one big member plus small neighbours that share almost nothing with it (the runtime classical host
shares at most 37 files with any neighbour; the tokenizer at most 17). The neighbours change far less often than
the big member, yet every change to the big member's 391 / 626 own files recompiled them too.

## 3. The guard split (implemented)

`core/six_lane_experiment_guards.mojo` (476 body lines, 348 asserts, 142 TOMBSTONE lines) was in every group's
closure through `checks/numerics.mojo` and changed in 204 commits since 2026-10-04. It is now nine modules:

| module | body lines | bindings that import it | Linux groups reached (of 11) | macOS groups (of 15) | guard commits since 10-04 now in it |
|---|---:|---:|---:|---:|---:|
| `core/six_lane_experiment_guards.mojo` (shared) | 70 | 81 (via `checks/numerics.mojo`) | 11 | 15 | 21 |
| `core/experiment_guards_neural.mojo` | 235 | 19 | 4 | 5 | 54 |
| `core/experiment_guards_cluster.mojo` | 60 | 15 | 3 | 4 | 32 |
| `core/experiment_guards_linear.mojo` | 51 | 11 | 4 | 5 | 27 |
| `core/experiment_guards_trees.mojo` | 42 | 9 | 2 | 3 | 30 |
| `core/experiment_guards_decomposition.mojo` | 39 | 3 | 2 | 2 | 19 |
| `core/experiment_guards_base.mojo` | 23 | 10 | 4 | 5 | 13 |
| `core/experiment_guards_series.mojo` | 17 | 7 | 2 | 3 | 8 |
| `core/experiment_guards_kernel.mojo` | 10 | 7 | 2 | 2 | 5 |

- **Who imports what.** Each binding `bindings/<name>.mojo` imports `<FAMILY>_GUARDS_OK` from its one family module and
  asserts it on the first line of `build_<name>()` (the module body every per-binding and merged build compiles), so
  a compile of any binding evaluates its family's guards. The shared file is still asserted through
  `GLOBAL_NUMERIC_MODE` (`checks/numerics.mojo`, unchanged). Every guard module is a leaf (`std` imports only).
  `x_neighbors/gen.py` emits the same two lines for the generated x_neighbors bindings.
- **Placement** (`split_guards2.py`; per entry in docs/build/guard_split.json): an entry whose defines are read in
  code (`is_defined`/`get_defined_*` outside comments) by bindings of one family goes to that family; read by
  several families, to the shared file; read nowhere (removed or retired defines) to the family that its
  family-local evidence names unanimously (family-local files mentioning it, its experiments/removed patch paths),
  else the family its name belongs to, else the shared file. A removed define changes no build, so its placement
  only chooses which module keeps the record; `tools/six_lane_grid.py` still evaluates every module.
- **Permanent record, byte for byte.** Entries moved whole, in order. An entry is a comment run with the code lines
  it heads; a comment run heading code lines of several families (the grid-lane batch headers) is carried verbatim
  to each module that receives one of its lines (65 extra copies of 37 header lines); `# ---- lane` blocks moved
  whole; a line using a `comptime` name moved with it. Proof (`python3 tools/experiment_guards.py check`, also
  `tools/test_experiment_guards.py`): the 360 code lines appear exactly once (sorted-line sha256 b090fc25...ceb
  before and after), all 116 comment lines appear (sorted unique sha256 0f05de92...876 before and after), every
  module keeps the old relative order, 348 asserts before and after.
- **Old references.** `docs/TOMBSTONES.md` entries cite `core/six_lane_experiment_guards.mojo:<line>`; those entries are
  permanent and are not rewritten. docs/build/guard_split.json maps every line of the pre-split file (9ee5208756)
  to its module. The procedure texts that name the old file (CLAUDE.md step 4 of "Deleting an experiment",
  docs/TOMBSTONES.md lines 17 and 28) are the owner's to update: a refusal now goes in the family module
  (`python3 tools/experiment_guards.py files` lists them), a cross-family one in the shared file.
- **New refusals** go in the family module of the code they guard; the rebuild fence
  (`tools/hooks/rebuild_fence.py`) says so when the shared file is touched. No family module is reached by more
  than 20 build scripts (the fence's line), so a family-local refusal needs no `MOJOLEARN_WIDE_REBUILD_OK`.

History, attributed line by line (`history_attrib2.py`: each commit's added lines, located in today's modules;
31 commits whose lines were later rewritten are counted as shared): of 204 guard commits, 138 would have touched
one family module, 14 several, 21 the shared file. Linux group rebuilds per leg from guard edits: 1,836
(204 x 9) -> 1,102 (2 to 4 of 11 groups for 142 commits, 11 for 52); macOS 2,652 -> 1,460.

## 4. The host-group split (implemented)

`bindings/merged_groups.json`, along the clusters above (same members, sources and flags; link units only):

| before | after | closure files |
|---|---|---:|
| `_mojolearn_native_host` (7 members, 702) | `_mojolearn_native_host`: core, linalg, estimators, metrics, preprocessing, solver | 313 |
| | `_mojolearn_native_classical_host`: runtime_classical | 443 |
| `_mojolearn_native_tokenizer_host` (6 members, 741) | `_mojolearn_native_tokenizer_host`: tokenizer | 656 |
| | `_mojolearn_native_series_host`: tsa, arima, forecast, resample, x_metrics | 117 |

The kept stems hold the core host binding (`_mojolearn_native_host.so`, named by packaging tests) and the tokenizer
host (`_mojolearn_native_tokenizer_host.so`, the forest boundary file), so those file names do not move. Consumers:

- `tools/merged_bindings.py` `member_file`: the merged file that holds a member, from the manifest or an installed index.
- `tools/release_incremental_gate.py` `HOST` and `tools/qualify_release_host_boundary.py` (installed index) name the
  tokenizer host's file through it instead of the stem.
- `tools/release_github_build.py` `unit_peak_gib`: a stem with no measured `UNIT_PEAK_GIB` row plans at the largest
  measured row of its tier and vendor (14 GiB host, cuda and hip), not the 48 GiB whole-directory peak; a split only
  removes members, so the largest measured group bounds it. Its first build records its own peak; raise or add the
  row then. The two kept rows stay (measured on a superset; never lowered).
- `python/mojolearn/_verification_light_reference.py`: unchanged. Its witness paths are informational; the light
  verifier compares module -> sha256 and judges changed native bytes by their outputs. It is regenerated by
  `tools/light_verifier_reference.py` at the next release.
- `tools/test_box_merged_verify.py` counts groups from the manifest.
- Packaging (`_mojolearn_*_host.so` globs), the loader index, `bindings/merged_build.sh`,
  `packaging/linux/build_sets.sh`, `tools/release_reuse.py`, `tools/release_host_composition.py`,
  `tools/development_build.py` and `bindings/development_groups.json` (same member set) already follow the manifest.

Expected effect (`scope.py`, 4,830 `.mojo` commits since 2026-10-04): host closure files recompiled per commit
868.5 -> 765.1 (-12%); largest host closure 739 -> 656; host full build 2,857 -> 2,954 files in 7 groups instead
of 5 (+97 files compiled twice, the shared core of the split groups). On the GitHub legs each group is its own
shard, so the host wall time is bounded by the slowest group: today 1,563 s median for `_mojolearn_native_host`;
after, two compiles of 313 and 443 files. Their seconds and peaks are measured by their first build.

## 5. One-time cost and per-change scope

- **One-time:** every closure changes once (the shared guard file shrank, all 81 binding files gained two lines,
  and the host groups have new stems): the first build after this lane misses every group (15 macOS, 11 per Linux
  leg), and every identity lane's closure changes once, so the next `tools/record_identity_column.sh` re-records all
  lanes (513) on each column. Bits do not change (module boundaries only), so the re-record must reproduce the
  admitted digests; a mismatch is a bug.
- **After:** a family-local refusal or tombstone rebuilds 2 to 4 of 11 Linux groups (2 to 5 of 15 on macOS) and
  re-records only the lanes whose closure holds that family's bindings (3 to 19 of 81 bindings), where it rebuilt
  9 of 9 / 13 of 13 and re-recorded every lane; a shared guard edit still rebuilds everything. A change inside the
  runtime classical host or the tokenizer host compiles 443 / 656 files instead of 702 / 741, and a change to
  their small neighbours compiles 313 / 117.

## 6. The 2026-11-07 split: the classical and tokenizer hosts by family

Lane post-split-host-groups, on main at bc4ecd4518. Static, like sections 1 to 5: nothing was compiled.

**Why.** On the 0.8.66 macOS wheel build (release run e, `macos-bincache/provenance/provenance.tsv`, column 5)
`_mojolearn_native_classical_host` took 8,163 s and `_mojolearn_native_tokenizer_host` 8,119 s (0.8.65: 4,282 and
4,214 s), then `_mojolearn_native_x_host` 2,540 s; every other host group took 54 to 1,005 s. The Mac builds its
groups four at a time, heaviest first, so the wheel cannot finish before its longest group. Each of the two was one
member after section 4, and the native entry point program (docs/plans/NATIVE_ENTRY_POINTS_AFTER_0866.md) had
added most of its code to them. The same two members compiled alone, one worker, in 1,020 s (5.93 GiB peak) and
1,131 s (5.08 GiB) on the same Mac that day (development cache, `~/.cache/mojolearn-development/peaks/`): the
release's 8x comes from four such compiles paging together on 16 GiB (plan section 4, lesson 11).

**What moved.** Each member registered many independent C entries; they now register in their own bindings, one
merged group each (preprocessing and linear share one), so each group is one `mojo build` of only what its entries
reach:

- `classical_native_call` stays one entry. Its tree, density, saved-model inference and host archive branches and its
  `family_*` branch (with the coded routes) moved unchanged into `bindings/classical_trees_door.mojo` and
  `bindings/classical_family_door.mojo`, registered by `_mojolearn_runtime_trees_host` and
  `_mojolearn_runtime_family_host` as `trees_native_call` and `family_native_call`. The classical door hands them the
  caller's own argument tuple (`PyObject_Call` of the entry, no Python frame), so the steps, their order, the contexts
  and the error mapping are the ones it ran before. Import setup retains the two entries in the classical connection
  context (`python/mojolearn/_native_classical_bridge.py` `_classical_doors`); `bindings/classical_door_routes.mojo`
  (a leaf) names which operations go where, so `family_native` and `trees_public_native` left the classical compile.
- The other entries moved as they were: the same function under the same name. The tokenizer host keeps a native
  refusal under every moved name (`runtime_classical_unavailable_c`, the 0.8.60 classical precedent), and
  `host_surface.HOST_IMPORT_ALIASES` replaces each with its owner's entry at import, one spec per owner, the
  classical owner first. The ownership witness `tokenizer_host_runtime_classical_abi` is 2: an older tokenizer that
  registers the moved entries itself is refused with the complete-layout message.
- Types: every Python type stays registered where it was (the classical host for the resident tree types, the
  compiled-binding snapshot, the scaler snapshot and the linear-algebra pin; the tokenizer host for its neural and
  generation types), because import setup loads those owners first; the two whose only allocator moved
  (`ParallelByteOwner`, `RemoteStateLifetime`) moved with it, under lookup-guarded registrars.
- `_mojolearn_native_x_host` gave its byte LM, model-selection and manifold members to
  `_mojolearn_native_orchestration_host` (link units only).

**The groups.** Closure files: the closure-v2 walk (`tools/compile_closures.py closure_of`). Reachable functions: the
union over the binding's exports of `tools/lane_select.py` `_MojoDefGraph` closures (it does not see the indexed
`neural_block_c[i]` / `causal_public_c[i]` registrations, before or after, so the two neural rows are low). Expected
seconds: the parent's measured single-worker seconds per reachable function (classical 1,020 s / 6,207, tokenizer
1,131 s / 5,711; 0.164 and 0.198 s), times the group's functions; the x groups from their members' measured
single-worker seconds (development runs mac-n1, mac-n11, mac-n13, the largest of each). A group's functions are
the union over its members. A linear share of a superlinear compile, so an
upper estimate; the first build measures each group.

| group | members | closure files | reachable functions | expected s, one worker (Mac) |
|---|---|---:|---:|---:|
| `_mojolearn_native_classical_host` | runtime_classical | 163 | 789 | 130 |
| `_mojolearn_native_classical_trees_host` | runtime_trees | 316 | 2,156 | 355 |
| `_mojolearn_native_classical_family_host` | runtime_family | 353 | 3,148 | 520 |
| `_mojolearn_native_classical_prep_host` | runtime_prep, runtime_linear | 210 | 2,229 | 365 |
| `_mojolearn_native_classical_decomp_host` | runtime_decomp | 217 | 1,700 | 280 |
| `_mojolearn_native_tokenizer_host` | tokenizer | 322 | 2,247 | 445 |
| `_mojolearn_native_catalog_host` | runtime_catalog | 334 | 866 | 170 |
| `_mojolearn_native_xseq_host` | runtime_xseq | 289 | 1,914 | 380 |
| `_mojolearn_native_byte_trainer_host` | runtime_byte_trainer | 267 | 1,165 | 230 |
| `_mojolearn_native_parallel_host` | runtime_parallel | 383 | 2,132 | 420 |
| `_mojolearn_native_neural_doors_host` | runtime_neural | 223 | 1,381 | 275 |
| `_mojolearn_native_orchestration_host` | byte_lm, runtime_model_selection, runtime_manifold | 287 | 2,222 | 190 |
| `_mojolearn_native_x_host` | x_linear, x_cluster, x_neighbors, x_decomp, x_prep, x_ann | 342 | 4,321 | 340 |
| before: `_mojolearn_native_classical_host` | runtime_classical | 488 | 6,207 | 1,020 measured |
| before: `_mojolearn_native_tokenizer_host` | tokenizer | 691 | 5,711 | 1,131 measured |

The largest expected group is about 520 s on one worker, against 1,020 and 1,131 s for the two it replaces. At four
builds at a time the release saw 1.8x to 8x its single-worker seconds per group (neural host 1,005 s, x host 2,540 s,
the two giants above); a group of 0.5 to 3 GiB does not page with three others beside it, so the release should
see the low end of that range. The host set grows from 7 to 17 groups on both platforms (the Linux sets read the
same manifest); `tools/release_github_build.py` mints 24 upload-slot jobs per set (4 identical + 17 host compiles +
the assemble job), and every new group's sizing row starts at its parent's peak until its first build records its own.

**One-time cost.** Every split group is a new compile key, and the tokenizer host is in every lane's closure, so the
first build after this lane misses all 17 host groups on every platform and the next identity record re-records
every lane whose closure holds any of the twelve split bindings (in practice every lane). Bits do not change: the
same functions compile from the same sources with the same flags into different link units, and the two doors run
the moved branches unchanged; a digest that differs is a bug.

**Regenerated after the first compile:** `python/mojolearn/_verification_light_reference.py`
(`tools/light_verifier_reference.py`, at the next release), the sizing rows above from measured peaks, and this
table's expected seconds from the measured ones.
