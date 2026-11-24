# Verify through `device=`: the inventory (lane vd-inventory, 2026-11-24)

Read from source on `lane/w3-integrate` ebcfe79e54 and from the round-6 coverage audit
(`~/mojolearn-evidence/w3-integrate-6-gate/gate6-3cda/gate6-3cda8a05b3/audit.json`, head 3cda8a05b3). Nothing was
compiled, imported or run. The generator is `~/mojolearn-evidence/vd-inventory/build_tables.py` (one pass, 10 worker
processes over the files), with its inputs beside it: `tools/device_class_table.py` output (`devclass.tsv`),
`python/mojolearn/host_surface.py --json` (`hs.json`) and `git grep -n -w MOJOLEARN_VENDOR` (`vendor_raw.txt`).

How the columns are derived (static estimates, not runtime facts):

- `takes_device`: functions from their signature; estimator classes from `tools/device_class_table.py`; other
  classes from the `__init__` signature along their bases or `_mojolearn_device = True`. Methods and properties
  `inherits` when their object takes `device`.
- CPU / GPU implementation: the class `_BINDING` (else the bindings named in its source file), mapped through the
  host surface manifest (`routes` and `classes`); a GPU binding counts when its build script exists. The Mojo door is
  the audit's `native_entries`.
- Sabotage arm: per lane. CPU side when a lane is in an arm of `tools/identity_lanes/coverage_sabotage.tsv` that
  rebuilds a `_host` binding, or a training lane of a host family with a `sabotage_define`; GPU side when an arm
  rebuilds a GPU binding.
- Pytest class: keyword and signal counts per file (raises/types/interop against digest/allclose/hash, `device=`
  uses, the name). DUPLICATE and DEVICE rows are candidates to read before any deletion, not a deletion list.

## Counts

| table | rows | split |
|---|---|---|
| `endpoints.tsv` | 4,494 | mapped 1,591; not numerical 953; unmapped 1,950. takes `device`: yes 215, inherits 3,368, no 778, `devices=` only 128, unresolved 5. No CPU implementation 210, no GPU implementation 152. Mapped endpoints by sabotage arm: CPU and GPU 838, CPU only 655, GPU only 2, none 96. Mapped endpoints on a lane whose reference is OWED: 168 (70 lanes). Unmapped proposals: GENERIC LANE 1,341, OWN LANE 606, NOT REASONABLE 3 |
| `functions_without_device.tsv` | 272 | 197 functions, 75 constructors. Side chosen today by `MOJOLEARN_VENDOR` 209, by `devices=` (GPU list, no CPU side) 43, always CPU 6, no compiled numeric side (metadata, files, profiles) 12, verify/check entries 2 |
| `vendor_switch_uses.tsv` | 362 | write 256, read 35, doc/message 71. Goes once verify passes `device=`: 45 (verify, check, identity-column tools, 2 sabotage patches); goes with its DUPLICATE test: up to 35; stays: product selector 25, native process pool 2, packaging 35, tooling 71, bench/experiments 46, docs 17, stored records 83 |
| `pytest_files.tsv` | 1,332 files, 11,147 tests | TOOLING 601, BOUNDARY 511, DUPLICATE 181 (817 tests), DEVICE 39 (502 tests). In `python/mojolearn/tests`: BOUNDARY 501, DUPLICATE 174, DEVICE 36, TOOLING 27 |

## The ten largest gaps

1. **Generic lanes absent: 1,341 unmapped endpoints.** `get_params`/`set_params` 391, `digest_`/`training_digest_`
   391, attribute access and fitted properties 251, `check` 196, pickling 78. One lane per mechanism covers them.
2. **Own lanes absent: 606 unmapped endpoints.** `score` 73, `fit_transform` 49, `call` 43, `state_dict` /
   `load_state_dict` 58, `parameters`/`named_parameters` 25, `fit_predict` 16, `forecast` 10, `inverse_transform` 11;
   109 of them belong to objects with no lane at all (`Sequential` 33 unmapped, `ParallelByteLanguageModelTrainer`
   19, `models.ParallelCausalLM` 19, `importers._program.ImportedProgram` 17, `Array` 16).
3. **No GPU-side fault arm on 655 mapped endpoints.** Their lanes are caught only by the CPU host gate's
   `MOJOLEARN_HOST_SABOTAGE`; GPU arms reach 258 lanes. Plan item 4 needs both sides.
4. **No fault arm at all on 96 mapped endpoints:** `ops.Session` 85, `ops.Program` 9, `model_selection.default_folds`,
   `model_selection.split_descriptor`.
5. **197 public functions take no `device`.** For 147 the side is the process's binding set (`MOJOLEARN_VENDOR`):
   metrics 69, linalg 16, model selection 12, training 14, top-level 39 (feature selection, solvers, `matmul`,
   `cross_entropy`, `layer_norm_*`, ...). Only `import_model(...).call` takes `device`.
6. **75 computing constructors take no `device`:** the optimizers (Adam, AdamW, SGD, LAMB, Lion, RMSprop, Adagrad,
   Adamax, NAdam, Adafactor), the LR schedules (5), the splitters (14), the GP kernels (4), the explainers (3), RFE,
   LanguageModelConfig/Trainer/Inference, SambaInference, MoEBlock, LayerNorm, the tokenizers, the `models.*`
   containers, `Array`, `ops.Program`.
7. **`device="cpu"` is refused on a GPU set for 15 estimator classes:** 9 reach the shared constructor door but
   their CPU implementation is "not connected to device= yet" (CalibratedClassifierCV, MultiOutput x2, OneClassSVM,
   OneVsRestClassifier, Stacking x2, Voting x2), and 6 compiled constructors do not name `device` (Adam, AdamW, SGD,
   Kernel/Permutation/TreeExplainer). Verify cannot pass `device="cpu"` to them until that is connected.
8. **One-sided endpoints:** 210 have no CPU implementation (ParallelCausalLM 25, LanguageModelTrainer 23,
   ParallelByteLanguageModelTrainer 22, LanguageModelConfig 15, the parallel drivers) and 152 no GPU implementation
   (HostGBDT, HostForest, LanguageModelInference, `models.*` file and tokenizer objects). For these verify gets one
   digest; the other side must refuse in words.
9. **70 lanes have their reference OWED** (168 mapped endpoints sit on them).
10. **Pytest that verify would replace: 181 DUPLICATE files (817 tests) and 39 DEVICE files (502 tests).** Each DEVICE
    file keeps its refusal tests (bad device word, device array to a CPU model, callbacks); the count of refusal
    sites per file is in the table.

## What the source contradicts in the plan

- **`MOJOLEARN_VENDOR` cannot be deleted from the product** (plan item 2, "if nothing else needs the variable").
  `_backend.py` reads it to choose cuda or hip on a mixed box and to load the CPU-only set on a GPU box (split core
  and flat layouts), and the native process pool writes it into every child's environment
  (`bindings/native_process_pool_door.mojo:163`). Only the verify and check uses go (`_verify_recipe.py:656`,
  `_check_both.py:749`), plus the identity-column tools that start a CPU child.
- **Two sabotage patches fault one side by reading `MOJOLEARN_VENDOR` at run time**
  (`x_decomp/checks/sabotage/e2e_p2_options.patch`, `x_neighbors/checks/sabotage/870_gamma_scale_device_column.patch`).
  With both sides in one process they fault both or neither; they must key on the call's device route.
- **"1,369 native registrations have a sabotage arm"** is per registration closure. Per lane and side, the arms
  reach the GPU side for 258 lanes only (gap 3).
- **"Public functions have no `device` parameter"** holds, but 43 of them (the `parallel_*` drivers and their
  classes) take `devices=`, a GPU list, and have no CPU side: `device=` does not apply to them as written.
- **"`device=None` keeps today's choice; `cpu` and `gpu` select the side"** already holds for 94 estimator classes;
  for the 15 in gap 7 the constructor refuses `device="cpu"` on a GPU set today, so item 2 depends on connecting
  them first.
- **`mojolearn.check` (`_check_both.py`) uses the same CPU child process as verify.** The plan names only
  `_verify_recipe.py`; `check` and the per-class `.check` endpoints (196 unmapped) move with it.

## Doubts

- CPU/GPU columns come from binding names in source files; a file that names several bindings can over-claim a side.
  `inherits` assumes every method follows its object's device; optimizers also move state with
  `params_to_device()`.
- The CPU-only sabotage count includes the host gate's whole-family define, which faults the host binding but is
  not one of the release audit's per-definition arms.
- Pytest classes are heuristic; read every DUPLICATE file against its named lane before the deletion lane deletes
  it. TOOLING files are never deletion candidates.
