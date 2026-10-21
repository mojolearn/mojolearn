# Finish leftovers audit (2026-10-21, lane finish-leftovers)

Goal: nothing from the core-completion effort stays outside main. Main at `9f9bdb4a8` contains
`origin/finish/core-completion-20261020` (merge `da729f6ce`). This audits the sixteen sibling `origin/finish/*-20261020`
lanes that fed it.

## Method

For each branch: `mb=$(git merge-base <branch> origin/main)`, the files in `git diff --name-only $mb <branch>`, and for each
file whose blob differs between the branch and `origin/main`:

1. Commit level: `git cherry origin/main <branch>`. Commits marked `+` (not patch-equivalent) were matched by subject to a
   main commit touching the same file (cherry-picks resolved against later integration changes).
2. File level: whether the branch's final blob appears in main's history for that path (main took it, then moved on).
3. Line level: lines the branch added over `mb` that are absent from main's file. Each remaining case was read; every one
   is a later main refactor of the same lines (for example the `ops.py` distributed/program shells now resolved through one
   native namespace, `ops.py:1549`; the cross-vendor Python callback paths replaced by refusals; keepalive `_ = x^` lines
   removed under the no-compensating-references rule; `_mojolearn_x_sequence*.mojo` imports widened with
   `partition_enter_py`).

Decisions: **(a)** main's version is newer or a superset; **(b)** branch content missing from main, ported here;
**(c)** obsolete. No file needed (b) or (c).

## Table

| Branch (head) | Files changed / differing from main | Commits patch-equal / subject-matched | Decision | Main commit(s) carrying it |
|---|---|---|---|---|
| cross-vendor-native-only-20261020 (`09649670a`) | 14 / 5 | 2 / 0 | (a) | patch-equal commits, merge da729f6ce |
| distributed-20261020 (`9b25c725f`) | 54 / 26 | 39 / 6 | (a) | e00fadb75 3a9fbbfea 771360c75 1a59edd96 386e6bbfc 8019a0688 + patch-equal |
| generation-20261020 (`ef5e2117c`) | 89 / 33 | 82 / 1 | (a) | 6482a2465 content present in main tools/hooks (no_host_routes.py:71,747) + patch-equal |
| generation-graph-repair-20261020 (`2c0c3839e`) | 111 / 18 | 49 / 7 | (a) | f872423a4 c8e5eaa58 01df751f6 9b214640d 5503de345 9b4fdce42; 24544a393 content present (classical_native.mojo:161) + patch-equal |
| graph-20261020 (`3ab5cc608`) | 47 / 20 | 25 / 13 | (a) | 6f6e32b0e a757313bd 920a82a43 7970b3778 631e36d88 58e5e42d3 fba58c078 921584b15 78bf1cbca 3a285fa1d 17c15739d ae57bcda6; 1e5165c7c content present + patch-equal |
| graph-hoist-20261020 (`bf8e3e04c`) | 27 / 8 | 10 / 0 | (a) | patch-equal commits, merge da729f6ce |
| graph-process-20261020 (`03b8a802d`) | 2 / 2 | 1 / 0 | (a) | patch-equal commits, merge da729f6ce |
| installed-native-audit-20261020 (`b3927aba2`) | 13 / 1 | 7 / 0 | (a) | patch-equal commits, merge da729f6ce |
| native-autograd-conv-smoke-20261020 (`c4003c15e`) | 6 / 1 | 1 / 0 | (a) | patch-equal commits, merge da729f6ce |
| native-classical-public-20261020 (`a94152138`) | 12 / 5 | 4 / 0 | (a) | patch-equal commits, merge da729f6ce |
| native-optimizer-smoke-20261020 (`786b5a058`) | 7 / 3 | 2 / 0 | (a) | patch-equal commits, merge da729f6ce |
| native-svm-smoke-20261020 (`94d6be425`) | 6 / 3 | 0 / 1 | (a) | 85dfb0db8 |
| parallel-causal-20261020 (`2234c29ca`) | 22 / 5 | 13 / 0 | (a) | patch-equal commits, merge da729f6ce |
| parallel-neural-admission-20261020 (`7cea0dea1`) | 38 / 1 | 16 / 0 | (a) | patch-equal commits, merge da729f6ce |
| pca-native-owner-20261020 (`b781f9111`) | 9 / 1 | 5 / 0 | (a) | patch-equal commits, merge da729f6ce |
| samba-public-20261020 (`e479dd506`) | 26 / 1 | 16 / 0 | (a) | patch-equal commits, merge da729f6ce |

Result: every branch is (a). Nothing was ported; lane/finish-leftovers carries only this record and the Modular ask.
The sixteen `origin/finish/*-20261020` branches (and the integration branch) can be retired once Andrew agrees.

## Differing files, per branch

`main-history` names the main commit whose blob equals the branch's final blob (main took it, then changed it later);
`merged` means the branch blob never existed alone on main because main combined it with another lane's edits of the
same file; every branch commit on that file has a same-subject main commit touching it.


- **cross-vendor-native-only-20261020**:
  - `bindings/byte_checkpoint_io.mojo` (a, main-history `f629ba34d`)
  - `bindings/cross_vendor_admission.mojo` (a, main-history `cc4c14732`)
  - `bindings/cross_vendor_native.mojo` (a, main-history `f629ba34d`)
  - `docs-site/api_exclude.txt` (a, main-history `cc4c14732`)
  - `python/mojolearn/tests/test_cross_vendor_worker_native.py` (a, main-history `cc4c14732`)

- **distributed-20261020**:
  - `autograd/distributed_program.mojo` (a, main-history `5e69d77ae`)
  - `autograd/distributed_pyapi.mojo` (a, main-history `1a59edd96`)
  - `autograd/partition_pyapi.mojo` (a, main-history `771360c75`)
  - `autograd/partition_runtime.mojo` (a, main-history `771360c75`)
  - `autograd/partition_stage.mojo` (a, main-history `771360c75`)
  - `bindings/_mojolearn_x_sequence.mojo` (a, merged)
  - `bindings/_mojolearn_x_sequence_host.mojo` (a, merged)
  - `bindings/cross_vendor_native.mojo` (a, main-history `823c3adec`)
  - `bindings/native_process_pool.mojo` (a, main-history `2e09ef628`)
  - `bindings/parallel_neural_native.mojo` (a, main-history `2e09ef628`)
  - `docs-site/api_exclude.txt` (a, merged)
  - `python/mojolearn/__main__.py` (a, merged)
  - `python/mojolearn/_parallel_neural.py` (a, main-history `2e09ef628`)
  - `python/mojolearn/_partitioned_program.py` (a, main-history `771360c75`)
  - `python/mojolearn/_surface_sequence.py` (a, merged)
  - `python/mojolearn/_verify_distributed_program.py` (a, main-history `386e6bbfc`)
  - `python/mojolearn/_verify_native_runtime.py` (a, main-history `9f4b6c781`)
  - `python/mojolearn/_verify_partitioned_program.py` (a, main-history `386e6bbfc`)
  - `python/mojolearn/cross_vendor.py` (a, main-history `e28c8bcf3`)
  - `python/mojolearn/ops.py` (a, merged)
  - `python/mojolearn/tests/test_cross_vendor_worker_native.py` (a, main-history `c9cf36e65`)
  - `python/mojolearn/tests/test_native_process_pool.py` (a, main-history `6bd069b3f`)
  - `python/mojolearn/tests/test_parallel_neural_native.py` (a, main-history `e66167c17`)
  - `tools/hooks/no_host_routes.py` (a, merged)
  - `tools/test_native_runtime_audit.py` (a, main-history `9f4b6c781`)
  - `training/checkpoint.mojo` (a, main-history `771360c75`)

- **generation-20261020**:
  - `bindings/_mojolearn_tokenizer_host.mojo` (a, main-history `76a795a6c`)
  - `bindings/causal_lm_public.mojo` (a, main-history `0edc97a26`)
  - `bindings/cross_vendor_native.mojo` (a, main-history `823c3adec`)
  - `bindings/native_json.mojo` (a, main-history `02b6c3693`)
  - `bindings/native_process_pool.mojo` (a, merged)
  - `bindings/parallel_causal_native.mojo` (a, main-history `0edc97a26`)
  - `bindings/parallel_mlp_native.mojo` (a, merged)
  - `bindings/parallel_neural_native.mojo` (a, main-history `2e09ef628`)
  - `bindings/samba_config_native.mojo` (a, main-history `8b1d4c487`)
  - `bindings/samba_construct_native.mojo` (a, main-history `88814f11b`)
  - `bindings/samba_public_native.mojo` (a, main-history `694269a1b`)
  - `bindings/samba_state_native.mojo` (a, main-history `77a1d820c`)
  - `bindings/samba_step_native.mojo` (a, main-history `77a1d820c`)
  - `python/mojolearn/__main__.py` (a, main-history `e0c7ff181`)
  - `python/mojolearn/_native_neural_bridge.py` (a, main-history `39855f94b`)
  - `python/mojolearn/_native_samba_bridge.py` (a, main-history `694269a1b`)
  - `python/mojolearn/_parallel_neural.py` (a, main-history `2e09ef628`)
  - `python/mojolearn/_samba_impl.py` (a, main-history `88814f11b`)
  - `python/mojolearn/_training_impl.py` (a, main-history `ec468a4e1`)
  - `python/mojolearn/_verify_native_runtime.py` (a, main-history `9f4b6c781`)
  - `python/mojolearn/cross_vendor.py` (a, main-history `e28c8bcf3`)
  - `python/mojolearn/models/causal_lm.py` (a, main-history `2ffd0bc99`)
  - `python/mojolearn/neural_inference.py` (a, main-history `2ffd0bc99`)
  - `python/mojolearn/parallel_training.py` (a, main-history `302609f98`)
  - `python/mojolearn/tests/test_cross_vendor_worker_native.py` (a, main-history `e28c8bcf3`)
  - `python/mojolearn/tests/test_native_process_pool.py` (a, main-history `6bd069b3f`)
  - `python/mojolearn/tests/test_parallel_byte_native.py` (a, main-history `ec5e24a23`)
  - `python/mojolearn/tests/test_parallel_mlp_native.py` (a, main-history `290ba5311`)
  - `python/mojolearn/tests/test_parallel_neural_native.py` (a, main-history `2e09ef628`)
  - `python/mojolearn/tests/test_samba_construct_native.py` (a, main-history `77a1d820c`)
  - `python/mojolearn/tests/test_samba_state_native.py` (a, main-history `ec468a4e1`)
  - `python/mojolearn/tests/test_samba_step_native.py` (a, main-history `694269a1b`)
  - `tools/test_native_runtime_audit.py` (a, main-history `9f4b6c781`)

- **generation-graph-repair-20261020**:
  - `autograd/ops_frontend.mojo` (a, main-history `22d8807f0`)
  - `autograd/program_runtime.mojo` (a, merged)
  - `bindings/_mojolearn_tokenizer_host.mojo` (a, merged)
  - `bindings/classical_attributes.mojo` (a, main-history `8a6b082d5`)
  - `bindings/generator_draw_native.mojo` (a, main-history `5da86eae3`)
  - `packaging/native_only/README.md` (a, main-history `f88b4eba8`)
  - `packaging/native_only/cases/smoke_umap.mojo` (a, main-history `4446206da`)
  - `packaging/native_only/manifest.json` (a, main-history `f88b4eba8`)
  - `packaging/native_only/test_inventory.py` (a, main-history `f88b4eba8`)
  - `python/mojolearn/_native_samba_bridge.py` (a, main-history `8c43027e5`)
  - `python/mojolearn/_training_impl.py` (a, main-history `5da86eae3`)
  - `python/mojolearn/ops.py` (a, main-history `7ac02bc35`)
  - `python/mojolearn/tests/test_classical_ols_native_owner.py` (a, main-history `72f11f70d`)
  - `python/mojolearn/tests/test_generator_draw_native.py` (a, main-history `5da86eae3`)
  - `python/mojolearn/tests/test_native_classical_public.py` (a, main-history `8a6b082d5`)
  - `python/mojolearn/tests/test_sequential_native_admission.py` (a, main-history `00f4e2434`)
  - `tools/native_only_smoke.py` (a, main-history `5503de345`)
  - `tools/test_native_only_smoke.py` (a, main-history `5503de345`)

- **graph-20261020**:
  - `autograd/ops_api.mojo` (a, main-history `5cd7295cd`)
  - `autograd/ops_frontend.mojo` (a, main-history `ae57bcda6`)
  - `autograd/program_control.mojo` (a, main-history `1d0eb5fc7`)
  - `autograd/program_optimize.mojo` (a, main-history `0df913a1b`)
  - `autograd/program_pyapi.mojo` (a, main-history `17c15739d`)
  - `autograd/program_runtime.mojo` (a, main-history `17c15739d`)
  - `autograd/program_tools.mojo` (a, main-history `1d0eb5fc7`)
  - `autograd/python_errors.mojo` (a, main-history `fba58c078`)
  - `autograd/weight_frontend.mojo` (a, main-history `690391a32`)
  - `bindings/_mojolearn_x_sequence.mojo` (a, merged)
  - `bindings/_mojolearn_x_sequence_host.mojo` (a, merged)
  - `bindings/native_json.mojo` (a, main-history `02b6c3693`)
  - `docs-site/api_exclude.txt` (a, merged)
  - `docs/plans/GRAPH_OPTIMIZATION.md` (a, main-history `ae57bcda6`)
  - `python/mojolearn/__main__.py` (a, merged)
  - `python/mojolearn/_surface_sequence.py` (a, merged)
  - `python/mojolearn/_verify_graph.py` (a, main-history `ae57bcda6`)
  - `python/mojolearn/ops.py` (a, merged)
  - `tools/test_verify_graph.py` (a, main-history `920a82a43`)
  - `training/checkpoint.mojo` (a, merged)

- **graph-hoist-20261020**:
  - `autograd/higher.mojo` (a, main-history `f96500cfc`)
  - `autograd/ops_api.mojo` (a, main-history `f96500cfc`)
  - `autograd/program_optimize.mojo` (a, main-history `1c13c16c2`)
  - `autograd/program_pyapi.mojo` (a, main-history `1c13c16c2`)
  - `autograd/program_runtime.mojo` (a, merged)
  - `docs-site/api_exclude.txt` (a, main-history `0ce90a5ed`)
  - `python/mojolearn/_verify_graph.py` (a, main-history `f96500cfc`)
  - `python/mojolearn/ops.py` (a, merged)

- **graph-process-20261020**:
  - `bindings/native_process_pool.mojo` (a, main-history `86a5bd86c`)
  - `python/mojolearn/tests/test_native_process_pool.py` (a, main-history `86a5bd86c`)

- **installed-native-audit-20261020**:
  - `packaging/native_only/cases/autograd_conv_core.mojo` (a, main-history `fe6827d25`)

- **native-autograd-conv-smoke-20261020**:
  - `packaging/native_only/cases/autograd_conv_core.mojo` (a, main-history `fe6827d25`)

- **native-classical-public-20261020**:
  - `bindings/classical_native.mojo` (a, main-history `855f109f3`)
  - `docs/plans/PUBLIC_CLASSICAL_NATIVE_AUDIT.md` (a, main-history `855f109f3`)
  - `python/mojolearn/_native_classical_bridge.py` (a, main-history `8d334bee6`)
  - `python/mojolearn/tests/test_classical_host_native_owner.py` (a, main-history `855f109f3`)
  - `python/mojolearn/tests/test_native_classical_public.py` (a, main-history `8a6b082d5`)

- **native-optimizer-smoke-20261020**:
  - `packaging/native_only/README.md` (a, main-history `b18954cf5`)
  - `packaging/native_only/manifest.json` (a, main-history `b18954cf5`)
  - `packaging/native_only/test_inventory.py` (a, main-history `b18954cf5`)

- **native-svm-smoke-20261020**:
  - `packaging/native_only/README.md` (a, main-history `85dfb0db8`)
  - `packaging/native_only/manifest.json` (a, main-history `85dfb0db8`)
  - `packaging/native_only/test_inventory.py` (a, merged)

- **parallel-causal-20261020**:
  - `bindings/samba_public_native.mojo` (a, main-history `ec468a4e1`)
  - `bindings/samba_state_native.mojo` (a, main-history `ec468a4e1`)
  - `python/mojolearn/_native_samba_bridge.py` (a, merged)
  - `python/mojolearn/_training_impl.py` (a, main-history `5e5c161fb`)
  - `python/mojolearn/tests/test_samba_state_native.py` (a, main-history `ec468a4e1`)

- **parallel-neural-admission-20261020**:
  - `autograd/ops_api.mojo` (a, main-history `b6254881b`)

- **pca-native-owner-20261020**:
  - `bindings/_mojolearn_tokenizer_host.mojo` (a, main-history `b1e55cfb0`)

- **samba-public-20261020**:
  - `docs/plans/GRAPH_OPTIMIZER_FULL_WORKLOAD_AB_2026-10-20.json` (a, main-history `e71247121`)
