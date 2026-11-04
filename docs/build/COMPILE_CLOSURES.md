# Compile closures and small compiles

What every Mojolearn compile reads, why the 0.8.65 compiles missed the binary cache, the key rule
that fixes it, a regrouping proposal, and what each common change compiles before and after.
Everything here is static: `git archive` trees, the retained release records and the import walk
that keys the cache. Nothing was compiled to write it.

Regenerate the tables (about five minutes on the Mac, no compiler):

    python3 tools/compile_closures.py report --old <previous release commit> --new <commit> \
        --since <date> --release-keys <old macos-bincache/provenance/keys> <new .../keys> \
        --pattern device_plan=core/device_plan.mojo --pattern kmeans=cluster/impl/detail/kmeans.mojo \
        --json closures.json
    python3 tools/compile_closures.py render --json closures.json

Per group, before anything compiles:

    python3 tools/compile_closures.py plan --old <previous release commit> --new <commit>
    python3 tools/bincache.py explain bindings/merged_build.sh gpu --group <stem>   # with the build's env
    bash tools/box_merged_verify.sh --plan nvidia|amd <sha>                          # on a box

## Summary

<!-- summary: hand-written from the generated tables below -->

## Why the releases compiled everything

Three separate causes, each enough on its own to miss every group:

1. **The whole tree was the source key of every merged group.** `bindings/merged_build.sh`
   compiles a generated root (`"$root"`), so `tools/bincache.py script_plan` could not read a root
   from its compile line and the v1 key fell back to every `.mojo` file of the tree for every
   group. Any edit anywhere moved all 13 macOS keys and all 9 keys of each Linux leg. The per-binding
   scripts never hit the fallback: all of them resolve with zero unresolved imports.
2. **Per-run and per-leg variables were keyed.** `MOJOLEARN_SLOT_TOKEN` (the `tools/mac_slot.py`
   lease token, random per run) is in every macOS key: the two macOS builds of 0.8.65 at the same
   commit (07:50Z and 09:07Z) missed 13 of 13 against each other. `MOJOLEARN_MERGED_KEY` hashed the
   whole manifest and `tools/merged_bindings.py`, so a one-group edit or a new generator flag moved
   every group. On Linux, `MOJOLEARN_CUDA_CODE_FORMAT` (read by no `bindings/` script) split the
   vendor-neutral host groups of the sm_80 leg from the others.
3. **The macOS cache is release-scoped and the build path moved.** `tools/release.py` points
   `MOJOLEARN_BINCACHE_DIR` at `<release>/<sha>/macos-bincache`, so a new release starts empty; and
   0.8.64 was built in `~/mojolearn-wt/integ-0864` while 0.8.65 was built in the main checkout,
   and the repository path is a key field (absolute paths are embedded in the binaries).

Between 0.8.64 and 0.8.65 specifically, the precise closures show the misses were also *real*:
the multi-GPU encapsulation changed `core/device_plan.mojo` and its neighbours, which reach 12 of
the 13 macOS groups and all 9 Linux groups, and every group's closure changed (the table below
lists the files). The fix does not make that pair hit; it makes every rerun, every tooling-only
range and every family-local change hit. Measured: from `0814836ffb` to `6519a54c65` (13 commits,
no `.mojo` change), `tools/compile_closures.py plan` reports 22 of 22 groups unchanged, where the
v1 macOS keys moved on every run.

## The key fix: rule closure-v2 (`tools/bincache.py`)

- **Source** = the import closure of the build's own roots. For a merged build the roots are
  `bindings/<member>.mojo` of each member of the group it compiles (`-I . -I bindings`, the
  resolution `mojo build` uses). There is no whole-tree fallback: an import of this tree that names
  no file raises `UnresolvedImport` with the import and its file (`bincache.py key`/`explain` exit 3;
  a build compiles without the cache and logs `BINCACHE ERROR ... unresolved import 'x' in f`).
  The generated Unicode table is keyed by its tracked generator and pin, so the key is the same
  before and after the generator first runs.
- **Merged** = per group: directory, stem, flags class, members and the sha256 of the exact root
  text. This replaces `MOJOLEARN_MERGED_KEY`.
- **Inputs** = the shell and Python files the build executes, followed from the script through
  every non-comment reference, including `"$(dirname -- "$0")/build_defines.sh"` (not keyed by v1).
  Shell files hash their executable lines, so a comment edit moves nothing.
- **Environment** = `MOJO_*`/`MODULAR_*` (the toolchain reads them), `MOJOLEARN_BUILD_PIXI_ENV`,
  and only the `MOJOLEARN_*` variables a keyed input reads on a non-comment line (dynamic names
  such as `MOJOLEARN_${FAMILY}_HOST_OUTDIR` become patterns), minus the documented lists:
  `NON_BUILD_ENV`, `NON_BUILD_PREFIXES` and the new `NON_BUILD_ENV_V2`
  (`MOJOLEARN_SLOT_TOKEN`, `MOJOLEARN_MERGED_KEY`, `MOJOLEARN_MERGED_PYTHON`,
  `MOJOLEARN_RUNPOD_KEY_FILE`, `MOJOLEARN_REUSE_DIR`, `MOJOLEARN_REUSE_PLAN`,
  `MOJOLEARN_MERGED_BINDINGS`, `MOJOLEARN_CUDA_CODE_FORMAT`, `MOJOLEARN_BUILD_TIERS`).
  `MOJOLEARN_COMPILE_JOBS` stays keyed (a codegen split has not been shown byte-neutral). A
  sabotage or fault-injection variable still makes a negative control even when nothing reads it.
- **Device**: a merged GPU build names its target (`MOJOLEARN_GPU_ARCHS` on Linux, pinned Metal
  flags on macOS), so the box's own GPU is not keyed; host builds keep it (host bytes have been
  seen to depend on the build machine, docs/MODULAR_ASKS.md, 2026-10-31).
- Unchanged: toolchain, numeric mode, target column, image, OS, repository path, arguments.

v1 stays computable (`key_fields(rule="v1")`, `MOJOLEARN_BINCACHE_SOURCE=tree`), and the
retained-compile verifiers (`tools/release_ptx_assembly.py`,
`tools/release_linux_group_composition.py`, `tools/release_compile_subunit_evidence.py`) accept
either rule: closure-v2 fields are recomputed at the original revision
(`bincache.merged_fields_reproduce`). `tools/prebuild_changed.py` predicts closure-v2 keys. The
first build under the new rule misses once everywhere (a new key for every group).

## Where the small compiles now apply

- **Identity verifier box job** (`tools/identity_box_job.sh` -> `tools/box_merged_verify.sh`):
  one group at a time through `tools/bincache.py`, staged under `build/box-verify/`, then moved
  into the run's `artifacts/` with the index written from the plan. A group whose key is in the
  box's hot directory (`/root/.mojolearn_bincache/hot`, kept across jobs) or a staged R2 map is
  placed. Verify must still be VERIFIED before the record; activation is unchanged.
  `BOX_PLAN=1` / `--plan` prints `hit (...)` or `compile: <reason>` per group and builds nothing.
  `tools/identity_box_job_cpu_ptx.sh` compiles nothing (it reuses that tree).
- **macOS release build** (`tools/release.py step_macos_build`): the release-scoped cache is
  seeded from every earlier release's `macos-bincache` (`MOJOLEARN_BINCACHE_SEED_DIRS`; a verified
  archive is copied in, so the release keeps its own complete provenance), and the step prints the
  per-group plan before `build_release_wheel.sh` starts.
- **Linux legs** (`packaging/linux/build_sets.sh` through `tools/bincache.py`, the R2 partition per
  arch and image): closure-v2 keys repeat across releases for unchanged groups, and the three
  legs share the host groups. `tools/release.py step_linux_builds` prints the static per-group
  plan against the previous release commit first.
- **Development builds** (`tools/development_build.py`) already key each development group by its
  exact closure (they never used the merged-script fallback); unchanged.

<!-- generated by tools/compile_closures.py render: begin -->

### Merged groups at 6519a54c65: what each compile reads

Closure = distinct Mojo source files the group's members import (closure-v2 walk, `-I . -I bindings`), generated sources counted as their generator. The v1 key hashed every one of the tree's 3938 `.mojo` files for every merged group (its compile line names a generated root; no root was readable, so the fallback fired).

| platform | set | group | flags | members | closure files | changed f48ed127e6..6519a54c65 |
|---|---|---|---|---:|---:|---|
| darwin | identical | `_mojolearn_native` | default | 3 | 320 | 19: `bindings/_mojolearn.mojo`, `cluster/impl/detail/kmeans.mojo`, `cluster/impl/detail/min_cluster_distance_compute.mojo` ... |
| darwin | identical | `_mojolearn_native_m1` | metal1 | 6 | 539 | 18: `cluster/impl/detail/kmeans.mojo`, `cluster/impl/detail/min_cluster_distance_compute.mojo`, `cluster/multi_gpu.mojo` ... |
| darwin | identical | `_mojolearn_native_trees` | default | 8 | 564 | 34: `bindings/_mojolearn_gbdt.mojo`, `bindings/_mojolearn_resample.mojo`, `bindings/_mojolearn_svm.mojo` ... |
| darwin | identical | `_mojolearn_native_trees_m1` | metal1 | 2 | 92 | 2: `core/device_plan.mojo`, `core/neural_context.mojo` |
| darwin | identical | `_mojolearn_native_sequence` | default | 5 | 340 | 10: `core/device_plan.mojo`, `core/multi_gpu.mojo`, `core/neural_context.mojo` ... |
| darwin | identical | `_mojolearn_native_sequence_m1` | metal1 | 2 | 318 | 10: `autograd/distributed_checkpoint.mojo`, `autograd/distributed_entries.mojo`, `autograd/distributed_program.mojo` ... |
| darwin | identical | `_mojolearn_native_cluster` | default | 3 | 236 | 15: `cluster/impl/detail/kmeans.mojo`, `cluster/impl/detail/min_cluster_distance_compute.mojo`, `cluster/multi_gpu.mojo` ... |
| darwin | identical | `_mojolearn_native_cluster_m1` | metal1 | 4 | 387 | 12: `cluster/impl/detail/kmeans.mojo`, `cluster/impl/detail/min_cluster_distance_compute.mojo`, `cluster/multi_gpu.mojo` ... |
| darwin | host | `_mojolearn_native_host` | host | 7 | 702 | 8: `core/device_plan.mojo`, `core/gemm.mojo`, `core/gram_multi_gpu.mojo` ... |
| darwin | host | `_mojolearn_native_tokenizer_host` | host | 6 | 739 | 8: `autograd/ops_api.mojo`, `autograd/program_runtime.mojo`, `bindings/embedding_layer_native.mojo` ... |
| darwin | host | `_mojolearn_native_neural_host` | host | 9 | 421 | 12: `autograd/distributed_checkpoint.mojo`, `autograd/distributed_entries.mojo`, `autograd/distributed_program.mojo` ... |
| darwin | host | `_mojolearn_native_forest_host` | host | 15 | 440 | 16: `core/device_plan.mojo`, `core/gemm.mojo`, `core/gram_multi_gpu.mojo` ... |
| darwin | host | `_mojolearn_native_x_host` | host | 9 | 555 | 2: `core/device_plan.mojo`, `training/optimizer_public_common.mojo` |
| linux | identical | `_mojolearn_native` | default | 9 | 660 | 23: `bindings/_mojolearn.mojo`, `cluster/impl/detail/kmeans.mojo`, `cluster/impl/detail/min_cluster_distance_compute.mojo` ... |
| linux | identical | `_mojolearn_native_trees` | default | 10 | 609 | 34: `bindings/_mojolearn_gbdt.mojo`, `bindings/_mojolearn_resample.mojo`, `bindings/_mojolearn_svm.mojo` ... |
| linux | identical | `_mojolearn_native_sequence` | default | 7 | 511 | 20: `autograd/distributed_checkpoint.mojo`, `autograd/distributed_entries.mojo`, `autograd/distributed_program.mojo` ... |
| linux | identical | `_mojolearn_native_cluster` | default | 7 | 486 | 18: `cluster/impl/detail/kmeans.mojo`, `cluster/impl/detail/min_cluster_distance_compute.mojo`, `cluster/multi_gpu.mojo` ... |
| linux | host | `_mojolearn_native_host` | host | 7 | 702 | 8: `core/device_plan.mojo`, `core/gemm.mojo`, `core/gram_multi_gpu.mojo` ... |
| linux | host | `_mojolearn_native_tokenizer_host` | host | 6 | 739 | 8: `autograd/ops_api.mojo`, `autograd/program_runtime.mojo`, `bindings/embedding_layer_native.mojo` ... |
| linux | host | `_mojolearn_native_neural_host` | host | 9 | 421 | 12: `autograd/distributed_checkpoint.mojo`, `autograd/distributed_entries.mojo`, `autograd/distributed_program.mojo` ... |
| linux | host | `_mojolearn_native_forest_host` | host | 15 | 440 | 16: `core/device_plan.mojo`, `core/gemm.mojo`, `core/gram_multi_gpu.mojo` ... |
| linux | host | `_mojolearn_native_x_host` | host | 9 | 555 | 2: `core/device_plan.mojo`, `training/optimizer_public_common.mojo` |

### Per-binding build scripts

81 per-binding scripts under `bindings/` (`build_defines.sh`, `build_host_family.sh` and `merged_build.sh` are not one binding's compile). Closure sizes: min 15, median 90, max 657 files. Unresolved imports: none. The only script whose compile line `script_plan` cannot read is `bindings/merged_build.sh` (the generated root `$root`), which is why every merged key fell back to the whole tree under v1.

### Files every darwin group reads (13 groups): 11

`checks/__init__.mojo`, `checks/identity_trace_read.mojo`, `checks/kernel_matrix.mojo`, `checks/numerics.mojo`, `core/__init__.mojo`, `core/hostptr.mojo`, `core/identity_trace.mojo`, `core/package_profile.mojo`, `core/six_lane_experiment_guards.mojo`, `core/step_phase.mojo`, `experiments/__init__.mojo`

### Files every linux group reads (9 groups): 40

`bindings/native_loaded_module.mojo`, `bindings/py_balanced.mojo`, `checks/__init__.mojo`, `checks/identity_trace_read.mojo`, `checks/kernel_matrix.mojo`, `checks/kernel_matrix_gemm.mojo`, `checks/numerics.mojo`, `checks/rtf_seam.mojo`, `checks/soft_f64.mojo`, `core/__init__.mojo`, `core/apple_air.mojo`, `core/device_plan.mojo`, `core/host_parallel.mojo`, `core/hostptr.mojo`, `core/identity_trace.mojo`, `core/launch_log.mojo`, `core/native_call_output.mojo`, `core/native_descriptor_abi.mojo`, `core/native_descriptor_call.mojo`, `core/package_profile.mojo`, `core/philox.mojo`, `core/six_lane_experiment_guards.mojo`, `core/step_phase.mojo`, `experiments/__init__.mojo`, `experiments/apple_fast/gemm/scoped_dispatch.mojo`, `gemm/__init__.mojo`, `gemm/afn_apple_fast.mojo`, `gemm/afn_apple_fast2.mojo`, `gemm/checks/__init__.mojo`, `gemm/checks/gemm_identical.mojo`, `gemm/contract.mojo`, `metrics/__init__.mojo`, `metrics/checks/__init__.mojo`, `metrics/checks/device_io.mojo`, `providers/__init__.mojo`, `providers/python/__init__.mojo`, `providers/python/neural_descriptor_pyapi.mojo`, `x_linear/__init__.mojo`, `x_linear/checks/__init__.mojo`, `x_linear/ops.mojo`

### Files ranked by the groups they reach (Linux, 9 groups; top 30)

| file | groups | commits since 2026-10-04 |
|---|---:|---:|
| `core/__init__.mojo` | 9 |  |
| `checks/numerics.mojo` | 9 |  |
| `core/package_profile.mojo` | 9 |  |
| `core/six_lane_experiment_guards.mojo` | 9 | 203 |
| `checks/__init__.mojo` | 9 |  |
| `checks/kernel_matrix.mojo` | 9 |  |
| `core/apple_air.mojo` | 9 |  |
| `gemm/__init__.mojo` | 9 |  |
| `gemm/checks/__init__.mojo` | 9 |  |
| `gemm/checks/gemm_identical.mojo` | 9 | 45 |
| `checks/kernel_matrix_gemm.mojo` | 9 |  |
| `checks/rtf_seam.mojo` | 9 |  |
| `gemm/contract.mojo` | 9 |  |
| `gemm/afn_apple_fast.mojo` | 9 |  |
| `gemm/afn_apple_fast2.mojo` | 9 |  |
| `experiments/__init__.mojo` | 9 |  |
| `experiments/apple_fast/gemm/scoped_dispatch.mojo` | 9 |  |
| `core/step_phase.mojo` | 9 |  |
| `core/device_plan.mojo` | 9 |  |
| `core/host_parallel.mojo` | 9 |  |
| `core/identity_trace.mojo` | 9 |  |
| `checks/identity_trace_read.mojo` | 9 |  |
| `checks/soft_f64.mojo` | 9 |  |
| `metrics/__init__.mojo` | 9 |  |
| `metrics/checks/__init__.mojo` | 9 |  |
| `metrics/checks/device_io.mojo` | 9 |  |
| `bindings/py_balanced.mojo` | 9 |  |
| `bindings/native_loaded_module.mojo` | 9 |  |
| `core/hostptr.mojo` | 9 |  |
| `core/philox.mojo` | 9 |  |

### Hot shared files: commits since 2026-10-04 x Linux groups reached (top 15)

4830 commits since 2026-10-04 touched a `.mojo` file. Each row is how many group rebuilds that file alone caused on one Linux leg.

| file | commits | groups | group rebuilds |
|---|---:|---:|---:|
| `core/six_lane_experiment_guards.mojo` | 203 | 9 | 1827 |
| `gemm/checks/gemm_identical.mojo` | 45 | 9 | 405 |
| `providers/python/neural_descriptor_pyapi.mojo` | 22 | 9 | 198 |
| `sequence/exec_device.mojo` | 50 | 3 | 150 |
| `experiments/classical_identical_ideas/linear_controls.mojo` | 23 | 6 | 138 |
| `autograd/ops_api.mojo` | 45 | 3 | 135 |
| `sequence/dispatch.mojo` | 43 | 3 | 129 |
| `bindings/native_loaded_module.mojo` | 14 | 9 | 126 |
| `bindings/host_archive_native.mojo` | 62 | 2 | 124 |
| `bindings/_mojolearn_tokenizer_host.mojo` | 123 | 1 | 123 |
| `autograd/tape.mojo` | 39 | 3 | 117 |
| `experiments/apple_fast/gemm/scoped_dispatch.mojo` | 13 | 9 | 117 |
| `x_decomp/device.mojo` | 53 | 2 | 106 |
| `sequence/ops.mojo` | 20 | 5 | 100 |
| `bindings/classical_attributes.mojo` | 20 | 5 | 100 |

### Why every macOS key changed between the two releases (from their keys/<key>.json)

| group (args) | v1 fields that differ | v1 scope (files) |
|---|---|---|
| `gpu --group _mojolearn_native` | `build_env.MOJOLEARN_SLOT_TOKEN`, `repo_path`, `source` | tree (3943 -> 3949) |
| `gpu --group _mojolearn_native_cluster` | `build_env.MOJOLEARN_SLOT_TOKEN`, `repo_path`, `source` | tree (3943 -> 3949) |
| `gpu --group _mojolearn_native_cluster_m1` | `build_env.MOJOLEARN_SLOT_TOKEN`, `repo_path`, `source` | tree (3943 -> 3949) |
| `gpu --group _mojolearn_native_m1` | `build_env.MOJOLEARN_SLOT_TOKEN`, `repo_path`, `source` | tree (3943 -> 3949) |
| `gpu --group _mojolearn_native_sequence` | `build_env.MOJOLEARN_SLOT_TOKEN`, `repo_path`, `source` | tree (3943 -> 3949) |
| `gpu --group _mojolearn_native_sequence_m1` | `build_env.MOJOLEARN_SLOT_TOKEN`, `repo_path`, `source` | tree (3943 -> 3949) |
| `gpu --group _mojolearn_native_trees` | `build_env.MOJOLEARN_SLOT_TOKEN`, `repo_path`, `source` | tree (3943 -> 3949) |
| `gpu --group _mojolearn_native_trees_m1` | `build_env.MOJOLEARN_SLOT_TOKEN`, `repo_path`, `source` | tree (3943 -> 3949) |
| `host --group _mojolearn_native_forest_host` | `build_env.MOJOLEARN_SLOT_TOKEN`, `repo_path`, `source` | tree (3943 -> 3949) |
| `host --group _mojolearn_native_host` | `build_env.MOJOLEARN_SLOT_TOKEN`, `repo_path`, `source` | tree (3943 -> 3949) |
| `host --group _mojolearn_native_neural_host` | `build_env.MOJOLEARN_SLOT_TOKEN`, `repo_path`, `source` | tree (3943 -> 3949) |
| `host --group _mojolearn_native_tokenizer_host` | `build_env.MOJOLEARN_SLOT_TOKEN`, `repo_path`, `source` | tree (3943 -> 3949) |
| `host --group _mojolearn_native_x_host` | `build_env.MOJOLEARN_SLOT_TOKEN`, `repo_path`, `source` | tree (3943 -> 3949) |

### Regrouping by closure overlap (proposal)

Start from one group per member and merge, each step, the pair that saves the most shared compile work per unit of added incremental scope, until the whole set reads no more files than today; never above the largest current group (the peak-memory proxy); macOS flags classes stay apart. `expected` is the commit-weighted mean number of closure files recompiled per changed file since 2026-10-04.

| platform/set | members | groups now -> proposed | full-build files now -> proposed | largest group | expected files per change now -> proposed |
|---|---:|---|---|---|---|
| darwin/host | 46 | 5 -> 19 | 2857 -> 2848 | 739 -> 710 | 1053 -> 683 (-35%) |
| darwin/identical | 33 | 8 -> 13 | 2796 -> 2787 | 564 -> 484 | 695 -> 523 (-25%) |
| linux/host | 46 | 5 -> 19 | 2857 -> 2848 | 739 -> 710 | 1053 -> 683 (-35%) |
| linux/identical | 33 | 4 -> 8 | 2266 -> 2198 | 660 -> 623 | 758 -> 577 (-24%) |

Proposed linux/host groups:

- 710 files: `_mojolearn_runtime_model_selection_host`, `_mojolearn_runtime_sequential_host`, `_mojolearn_tokenizer_host`, `_mojolearn_x_sequence_host`
- 456 files: `_mojolearn_runtime_classical_host`, `_mojolearn_runtime_manifold_host`
- 248 files: `_mojolearn_estimators_host`, `_mojolearn_kernel_methods_host`, `_mojolearn_linalg_host`, `_mojolearn_x_decomp_host`
- 149 files: `_mojolearn_byte_lm_host`, `_mojolearn_neural_host`, `_mojolearn_training_host`, `_mojolearn_transformer_host`
- 144 files: `_mojolearn_core_host`, `_mojolearn_metrics_host`
- 130 files: `_mojolearn_svm_host`
- 125 files: `_mojolearn_forest_host`, `_mojolearn_rf_host`, `_mojolearn_trees_host`
- 120 files: `_mojolearn_hdbscan_host`, `_mojolearn_hdbscan_infer_host`, `_mojolearn_solver_host`
- 119 files: `_mojolearn_gp_host`, `_mojolearn_gp_infer_host`, `_mojolearn_mixture_host`, `_mojolearn_mixture_infer_host`, `_mojolearn_x_cluster_host`
- 87 files: `_mojolearn_mamba_host`
- 83 files: `_mojolearn_gbdt_host`
- 83 files: `_mojolearn_preprocessing_host`, `_mojolearn_x_cnn_host`, `_mojolearn_x_neighbors_host`
- 71 files: `_mojolearn_arima_host`, `_mojolearn_forecast_host`, `_mojolearn_tsa_host`
- 68 files: `_mojolearn_x_linear_host`
- 68 files: `_mojolearn_ivf_host`, `_mojolearn_ivf_search_host`, `_mojolearn_x_ann_host`
- 64 files: `_mojolearn_x_prep_host`
- 48 files: `_mojolearn_x_trees_host`
- 39 files: `_mojolearn_x_metrics_host`
- 36 files: `_mojolearn_embedding_host`, `_mojolearn_embedding_infer_host`, `_mojolearn_resample_host`

Proposed linux/identical groups:

- 623 files: `_mojolearn`, `_mojolearn_estimators`, `_mojolearn_hdbscan`, `_mojolearn_linalg`, `_mojolearn_metrics`, `_mojolearn_solver`, `_mojolearn_x_decomp`
- 511 files: `_mojolearn_byte_lm`, `_mojolearn_embedding`, `_mojolearn_mamba`, `_mojolearn_training`, `_mojolearn_transformer`, `_mojolearn_x_cnn`, `_mojolearn_x_sequence`
- 409 files: `_mojolearn_gp`, `_mojolearn_ivf`, `_mojolearn_kernel_methods`, `_mojolearn_mixture`, `_mojolearn_preprocessing`, `_mojolearn_resample`, `_mojolearn_svm`, `_mojolearn_x_ann`, `_mojolearn_x_cluster`, `_mojolearn_x_neighbors`
- 229 files: `_mojolearn_gbdt`
- 140 files: `_mojolearn_rf`, `_mojolearn_trees`, `_mojolearn_x_trees`
- 111 files: `_mojolearn_x_metrics`, `_mojolearn_x_prep`
- 98 files: `_mojolearn_arima`, `_mojolearn_tsa`
- 77 files: `_mojolearn_x_linear`

### Compile scope of single-file changes: groups (closure files) recompiled, now -> proposed

| change | darwin/host | darwin/identical | linux/host | linux/identical |
|---|---|---|---|---|
| device_plan | 5/5 (2857) -> 9/19 (2130) | 7/8 (2478) -> 11/13 (2358) | 5/5 (2857) -> 9/19 (2130) | 4/4 (2266) -> 7/8 (2087) |
| gbdt | 0/5 (0) -> 0/19 (0) | 1/8 (564) -> 1/13 (229) | 0/5 (0) -> 0/19 (0) | 1/4 (609) -> 1/8 (229) |
| guards | 5/5 (2857) -> 19/19 (2848) | 8/8 (2796) -> 13/13 (2787) | 5/5 (2857) -> 19/19 (2848) | 4/4 (2266) -> 8/8 (2198) |
| kmeans | 0/5 (0) -> 0/19 (0) | 4/8 (1482) -> 4/13 (1164) | 0/5 (0) -> 0/19 (0) | 2/4 (1146) -> 2/8 (1032) |
| mamba | 0/5 (0) -> 0/19 (0) | 1/8 (340) -> 1/13 (340) | 0/5 (0) -> 0/19 (0) | 1/4 (511) -> 1/8 (511) |
| sequence | 2/5 (1160) -> 1/19 (710) | 2/8 (658) -> 2/13 (658) | 2/5 (1160) -> 1/19 (710) | 1/4 (511) -> 1/8 (511) |
| x_decomp | 0/5 (0) -> 0/19 (0) | 3/8 (1246) -> 2/13 (742) | 0/5 (0) -> 0/19 (0) | 2/4 (1146) -> 1/8 (623) |


<!-- generated by tools/compile_closures.py render: end -->

## Target workflow: three common changes, before and after

<!-- workflow: hand-written from the generated tables -->

## Regrouping: why it is a proposal, and who consumes group names

Changing `bindings/merged_groups.json` changes the `.so` names shipped in the wheel. These
consumers are driven from the manifest: the loader index `_mojolearn_merged.json` (written from the
plan by `tools/merged_bindings.py index`, read by `python/mojolearn/_merged.py`),
`bindings/merged_build.sh`, `packaging/linux/build_sets.sh` (pool jobs, `reindex_merged`),
`packaging/macos/build_release_wheel.sh` (`merged_plan`), `tools/release_reuse.py` (`MERGED_INDEX`
per set), `tools/release_host_composition.py` (per-file maps keyed by the manifest's files),
`tools/development_build.py` and this report. These name groups explicitly and would have to change
with it, so the regrouping is not implemented here:

- `tools/release_github_build.py` `UNIT_PEAK_GIB`: the measured peak per (tier, group, vendor)
  that sizes each GitHub shard; a new group has no measured peak until it is built once.
- `python/mojolearn/_verification_light_reference.py`: the shipped light-verification reference
  lists witnesses by group file name.
- `tools/qualify_release_host_boundary.py` and `tools/release_incremental_gate.py`: name
  `_mojolearn_native_tokenizer_host`.
- Retained release evidence (`tools/apple_retained_groups.py`, `tools/release_ptx_assembly.py`,
  `tools/release_linux_group_composition.py` receipts) records today's group stems; the code reads
  the stems from the manifest, so a regroup simply retires that evidence (one rebuild).

Smaller groups also mean more compiler processes and more shared files compiled once per group:
the proposal keeps the full-build file count at or below today's, and no group above today's
largest closure (the peak-memory proxy), but the memory of a new group is only known once it is
compiled. Recommended order: land closure-v2 (the bulk of the saving), then split the hot shared
files named above, then regroup with one measured build of the new groups.
