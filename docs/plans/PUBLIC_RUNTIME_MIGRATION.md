# Public runtime migration

Andrew, 2026-10-21. This plan defines the contract for every public method, how we measure it, where the
package stands today, and the order of the migration.

## The contract

Python connects the user to Mojo, and Mojo does the work. A public method (`fit`, `predict`, `transform`,
`score`, `partial_fit`, `sample`, `decision_function`, `predict_proba`, `generate`, `step`, `save`/`load`,
...) runs in this order:

    user call -> thin Python binding (forward arguments and buffer references)
              -> ONE native call that validates, prepares, executes and produces the results
              -> return

The owner contract in `AGENTS.md` is: "A public method is one binding call that hands over buffers and returns the result; nothing else runs in Python."
Outside that native call, Python does not convert or prepare inputs, choose execution routes,
allocate working buffers, sequence several native calls or post-process results. Import-time connection
is distinct from a public constructor or an operation's first call. Native code must not call back into
Python functions, methods, descriptors or input protocols. PyPI and the
scikit-learn compatible interface are only the shell. They are never a reason to keep implementation
logic in Python.

A standalone native smoke test does not prove that a public method is native, and neither does a green
kernel check. Only the method's own call path proves it.

Migration correctness is separate from a performance claim. Reuse valid correctness and compilation
evidence; do not start a performance campaign just to rename or move code. A speedup claim or default
promotion still requires the full-workload evidence in `AGENTS.md`. Within one release, identical-mode
bits must match Apple, NVIDIA and AMD; bits may change between versions. Prefer bit-exact moves: the
same operations in the same order.

## The measure

The operation baseline below remains a regression check, not a completion claim.
For one broader static ledger, run
`python3 tools/public_runtime_inventory.py --endpoints-json /path/to/endpoints.json`.
This uses the same public export and definition resolver, retaining aliases and
inherited constructors, methods, state/checkpoint operations, property accessors,
refusal-only endpoints and declared parameter names/defaults. Each definition has
an AST digest so evidence can be attached to the source it actually exercised.
Runtime, correctness, identity, fault, options and failures have separate evidence
fields; all start unreviewed. An algorithm-level lane or a static test reference
does not fill those fields. Dynamic installers and external-base members still
need runtime enumeration; native-shaped calls still need transitive tracing.

`tools/public_runtime_inventory.py` is a static triage measure, not runtime certification. It reads the code statically and imports nothing.
For every public name `mojolearn` exports, it takes these operation methods: `fit`, `partial_fit`,
`predict`, `predict_proba`, `decision_function`, `transform`, `fit_transform`, `fit_predict`,
`inverse_transform`, `score`, `sample`, `generate`, `step`, `forward`, `train_step`, `save`, `load`,
`export` and `__call__`. A public function is measured too, with `call` as its method name.

The export set is built from:

- the export walk in `packaging/native_only/inventory.py`: the literal `__all__` of the package and of
  its `_expansion_*` modules;
- the rows of `packaging/native_only/manifest.json`, which the census marks `not_algorithm` or not.

Rows that resolve to the same definition are measured once. Two kinds of method are skipped, because
they are not operations: one whose body only raises, and a public constant.

For each method, the tool walks the method body and, transitively, every mojolearn helper the body
calls, including methods on objects the path builds (`pr = _Prog(); pr.run()`). The walk stops at a
native call. A call counts as native when it is:

- an entry called on a binding, such as `_backend.binding(...).entry(...)`, `self._bind().entry(...)`,
  `binding.entry(...)` where the binding was passed in as a parameter, `self._runtime.entry(...)`, or
  `_native_neural_bridge.RUNTIME.entry(...)`;
- a local `fn = b.entry` or `getattr(b, "entry")`;
- a dispatcher from a module-level connect, such as `_ENTRY(self, "fit", args, _CONTEXT)`;
- an unresolved `_native_*` call.

Helpers that return a binding are binding acquisition. They are not counted and not walked. Calls into
the conversion layer (`_buffer`, `_array`, `_arrays`, `_cast_in`, `_bufcheck`, `_sk_validate`,
`_backend`) count as shell calls and are not walked either.

The tool writes one TSV row per method:

| column | meaning |
|---|---|
| `native_calls` | native call sites on the path; `native_entries` counts the distinct entries |
| `py_statements` | Python statements in every function the path reaches |
| `loops` | loops and comprehensions. `untagged_loops` counts those without `# driver-loop:` or `# glue`, leaving out loops over literals |
| `route_branches` | `if` statements that do more than raise and that make a call in an arm |
| `numpy_calls` | NumPy calls. `numpy_work` counts those that are not conversion |
| `allocations` | `np.empty/zeros/ones/full[_like]`, `_buffer.empty/zeros/full`, `*alloc*`, `Array(...)` |
| `helpers` | Python functions walked |

Each method gets one of three verdicts:

- **native**: the method body makes the one native call itself. The path has one native call site, no
  Python helper, no loop, no route branch, no allocation and no NumPy work. A no-op such as the `fit` of a
  stateless transform also counts as native.
- **python**: the path reaches no native call, or it does NumPy work, or it has an untagged loop.
- **orchestrated**: everything else, such as several native calls, a route choice, a Python allocation,
  a tagged driver loop, or one native call reached through a Python helper.

These commands run the tool:

    python3 tools/public_runtime_inventory.py --summary             # per-family numbers
    python3 tools/public_runtime_inventory.py --out inventory.tsv   # every method
    python3 tools/public_runtime_inventory.py --explain mojolearn.KMeans.fit   # the walk, function by function
    python3 tools/public_runtime_inventory.py --check               # what CI runs
    python3 tools/public_runtime_inventory.py --baseline            # after a migration lands

`.github/workflows/public-runtime.yml` runs `--check` on every push that touches `python/mojolearn/`
and on manual dispatch. It uses a free runner and needs only python3. The check fails when:

- the number of non-native methods grows past `tools/hooks/public_runtime_baseline.tsv`;
- the number of python methods grows;
- any method's verdict gets worse (native -> orchestrated -> python).

When a migration lands, refresh the baseline in the same commit with `--baseline`. The baseline only
shrinks.

The tool has limits:

- It is static. A method that is installed at runtime, such as one set with `setattr` by an import-time
  installer, is measured by its source definition. A migration therefore lands when the class's own
  source method body is the one-call shell.
- A call on an object whose type the tool cannot infer (`self.best_estimator_.predict`) counts as
  unresolved. It is not walked.
- The statement and branch totals are summed per method. A shared helper counts once for every method
  that reaches it.

## Reviewed remaining work (2026-10-21, source 6174c30ad)

The executable inventory still reports **996 operation rows: 81 native-shaped, 396 orchestrated,
519 python**. Its 915 non-native rows refer to 695 distinct definition locations; inheritance and
shared helpers mean they are not 915 independent algorithm rewrites. Eight of the 81 native-shaped
rows are no-op `fit` methods. A separate public/MRO walk found 1,906 method rows outside the census's
operation-name list, including constructors, `backward`, state/checkpoint methods and descriptors.
Those omitted rows need classification; they are not 1,906 established violations.

The review artifacts are [the ordered work plan and failure records](runtime_completion_review_20261021.json)
and [every remaining census row, assigned to a workstream](runtime_completion_remaining_20261021.tsv).
Evidence is under `~/mojolearn-evidence/runtime-completion-review-20261021-182318/`.
The earlier baseline below is retained as historical evidence, not the current count.

### Close these shared gaps first

1. **Make completion measurable (`R0`).** The census skips binding getters and conversion helpers,
   stops at native-looking calls, and does not inspect native-to-Python callbacks. It omits constructors,
   much state/checkpoint handling and other public methods. Expand the inventory and add strict cold
   and repeated whole-operation runtime checks. Green baseline CI means debt did not increase.
2. **Remove verified Python execution from migrated paths (`R1`).** The late-NumPy path stores Python
   `_connect_numpy` in the classical context and calls it from `classical_attributes.mojo`. Optimizer
   `step`, clipping and accumulation still call Python `_binding`/`_load` selectors. Both violate the
   contract above even when the static census calls the public method native. Preserve no-NumPy,
   NumPy-first and NumPy-later inputs and outputs; connect them through supported native code.
3. **Close the retained failures on coherent binaries (`V0`–`V2`).** The retained `t5.log` has 204 passes
   and 34 failures. It lacks a complete source/loaded-binary receipt, so neither stale-build dismissal
   nor 34 independent algorithm-bug claims are justified. Finish the existing frozen integration build,
   retain provenance, and rerun the affected cases after their fixes.
4. **Validate the fault gate and finish its evidence (`S0`–`S1`).** Four negative probes demonstrated
   false passes: divergent clean output, different source/vendor, a missing faulted fixture, and a
   missing-binary refusal. Require valid baseline/provenance, complete requested coverage, a legitimate
   fault-triggered refusal or changed result, and restoration to clean output.

The 34 failures have these concrete work items:

| Cases | Cause or diagnostic | Required action |
|---|---|---|
| 24 | Supported buffers rejected by shared Sequential admission | Trace the classical context's actual ndarray/scalar/buffer-owner types; preserve ordinary inputs and poisoned-hook refusals. The precise underlying cause awaits coherent-run provenance. |
| 1 | Importing mojolearn eagerly loads NumPy | Repair the import path and audit no/first/late-NumPy operations, including the first call. |
| 3 | Host export manifests disagree with the source scanner | Add genuine new exports; teach the scanner `def_py_c_function` so six existing ops entries are not falsely absent. |
| 3 | Historical evidence paths absent from the test worktree | The named paths exist in main. Verify retained bytes and use `MOJOLEARN_TEST_EVIDENCE_ROOT`; do not recreate old measurements. |
| 1 | Tokenizer sabotage define not found by the test | Follow its moved source in `tokenizer/impl/tokenizer_host_shell_impl.mojo`. |
| 2 | Reference/coverage metadata disagrees | Reconcile `nystroem-poly` fixture accounting and `cross-vendor-fold` comparability with actual admitted evidence. |

### Fault-run evidence that arrived during the review

`lane/run-sabotage-arms` at `62e93bd37` records 51 arms; main integrated that work at `52fd0f4fc` during
this review. Re-reading its raw `.sabrun/out` records found fault detection and restored clean
signatures for every lane of those arms in the reviewed `6174c30ad` manifest. This is
**base fixture, one repeat, eligible CPU/Metal columns**, not an all-vendor qualification. Retain and
link that work; do not rerun it wholesale. Reconcile historical manifest/summary differences from the
raw records and the merged rank-deficient RSVD case. One arm's commit labels differ across clean/faulted runs, with a diff confined to
another fault patch and the manifest; retain the product-source equivalence explanation.

The current manifest has **54 arms**, and all 54 patches apply. The three without complete retained
run records are `coverage-core-py2mojo_rows`, `coverage-sequence-opt_public`, and
`coverage-training-optimizer_public`. Run those on the appropriate frozen builds after hardening the
acceptance check. Tie every admission to loaded-binding provenance and the actual refusal reason.

### Remaining migration workstreams

| Workstream | Non-native census rows | Boundary to migrate |
|---|---:|---|
| M1: preprocessing and metrics | 204 | Shared admission, buffers, conversion, results and preprocessing/metric orchestration |
| M2: linear and decomposition | 193 | Public fit/predict/transform/score and associated state/archive paths |
| M3: ensemble | 132 | Forest/boosting orchestration and shared model wrappers |
| M4: neural, sequence and byte-LM | 135 | Remaining trainers, blocks, schedules, state/checkpoints and whole-operation boundaries |
| M5: neighbors and clustering | 120 | Index/query setup, clustering orchestration and results |
| M6: remaining estimator families | 131 | Model selection, SVM, kernels, GP, manifold, mixtures, linalg and resampling |
| Total | 915 | Exact class/method/source rows are in the companion TSV |

Do not wait for all families to finish before qualifying a slice. Each slice must preserve interfaces
and existing numerical kernels, pass independent correctness and strict runtime checks, and retain
same-version identity evidence for affected targets. Audit omitted lifecycle methods with each slice.

At the final review check, the existing build at `3837ab3f0` had passed host and Apple fast; Apple
identical was still running. The final release gate (`Q0`) still needs a coherent installed build, resolution of the recorded
`logistic-multiclass`/`hf-checkpoint` mismatches, affected NVIDIA/AMD qualification, explicit neural-training
verification, and actual multi-device evidence for supported parallel paths. The routine verifier
excludes neural training. Keep the bounded GitHub no-Python check and Python correctness tests; a
separate Python-free product is not required to finish this migration.

## Historical baseline (main f180228eb)

The baseline covers 383 public names and 996 operation methods: **53 native, 406 orchestrated, 537
python**. The path reaches 4,579 native call sites and 189,994 Python statements, counted per method.

| family | names | methods | native | orchestrated | python | native calls | py statements |
|---|---:|---:|---:|---:|---:|---:|---:|
| preprocessing | 35 | 135 | 0 | 62 | 73 | 740 | 47,324 |
| ensemble | 32 | 132 | 0 | 49 | 83 | 739 | 30,690 |
| linear_model | 27 | 102 | 0 | 53 | 49 | 335 | 15,721 |
| decomposition | 28 | 101 | 0 | 86 | 15 | 932 | 21,560 |
| sequence | 41 | 81 | 0 | 4 | 77 | 162 | 7,079 |
| neighbors | 23 | 73 | 0 | 33 | 40 | 148 | 9,159 |
| metrics | 69 | 69 | 0 | 9 | 60 | 314 | 22,652 |
| neural_network | 19 | 53 | 8 | 3 | 42 | 396 | 6,868 |
| cluster | 10 | 47 | 0 | 30 | 17 | 86 | 4,156 |
| neural_inference | 9 | 28 | 22 | 6 | 0 | 94 | 1,080 |
| training | 25 | 27 | 10 | 16 | 1 | 53 | 1,069 |
| model_selection | 15 | 25 | 0 | 0 | 25 | 80 | 4,895 |
| svm | 5 | 23 | 0 | 10 | 13 | 106 | 4,517 |
| manifold | 8 | 20 | 0 | 10 | 10 | 257 | 5,790 |
| linalg | 15 | 17 | 0 | 13 | 4 | 29 | 951 |
| kernel_methods | 3 | 15 | 0 | 4 | 11 | 14 | 1,360 |
| gaussian_process | 4 | 13 | 0 | 2 | 11 | 39 | 3,129 |
| mixture | 2 | 13 | 0 | 11 | 2 | 13 | 1,032 |
| models | 3 | 11 | 10 | 0 | 1 | 10 | 11 |
| resample | 5 | 5 | 0 | 4 | 1 | 11 | 239 |
| ops | 3 | 4 | 3 | 0 | 1 | 9 | 398 |
| language_model | 2 | 2 | 0 | 1 | 1 | 12 | 314 |
| **total** | **383** | **996** | **53** | **406** | **537** | **4,579** | **189,994** |

The 53 native methods are the neural runtime's one-call shells: the `CausalLM`, Mamba, transformer and
Samba blocks' `forward`, `step` and `__call__`, the parallel trainers' `train_step`, `ops.Session.step`,
and stateless CNN layer `fit`s. No classical estimator method is native today.

| operation | methods | native | orchestrated | python |
|---|---:|---:|---:|---:|
| fit | 197 | 9 | 88 | 100 |
| call (functions) | 149 | 3 | 50 | 96 |
| predict | 118 | 0 | 58 | 60 |
| score | 92 | 0 | 9 | 83 |
| transform | 73 | 0 | 55 | 18 |
| fit_transform | 72 | 0 | 38 | 34 |
| load / save | 98 | 3 | 0 | 95 |
| predict_proba | 44 | 0 | 35 | 9 |
| others | 153 | 38 | 73 | 42 |

These are the classical estimators the migration starts from:

| method | native calls | statements | loops | route branches | allocations | helpers | verdict |
|---|---:|---:|---:|---:|---:|---:|---|
| LinearRegression.fit | 16 | 293 | 2 | 31 | 26 | 39 | orchestrated |
| LinearRegression.predict | 1 | 8 | 0 | 0 | 1 | 0 | orchestrated |
| Ridge.fit | 16 | 192 | 2 | 20 | 26 | 22 | orchestrated |
| LogisticRegression.fit | 9 | 317 | 12 | 30 | 12 | 19 | python |
| PCA.fit | 14 | 389 | 6 | 37 | 10 | 45 | orchestrated |
| KMeans.fit | 1 | 89 | 0 | 10 | 5 | 4 | python (host NumPy RandomState for a callable `init`) |
| SVC.fit | 22 | 613 | 32 | 49 | 34 | 36 | python |
| SVR.fit | 8 | 371 | 11 | 31 | 11 | 23 | python |
| RandomForestClassifier.fit | 10 | 396 | 17 | 38 | 10 | 28 | python |
| KNeighborsClassifier.fit | 4 | 296 | 12 | 31 | 1 | 23 | python |
| StandardScaler.fit | 6 | 332 | 28 | 36 | 9 | 41 | orchestrated |
| GaussianProcessRegressor.fit | 3 | 220 | 19 | 30 | 6 | 20 | python |

The 42 directly defined exported-class `fit` methods in `python-runtime-scope-20261021/direct-public-methods.json`
all appear here. So do its 83 rows in total: 56 are orchestrated and 27 python, and none is native. The
four bodies that are a single return call, such as `RandomForestClassifier.fit -> _fit_with_tree_start`,
are judged by the helpers they reach.

## Order of the migration

### 1. Shared infrastructure first

Most Python on these paths is not owned by any one family. The same steps repeat across KMeans,
linear/logistic, PCA, SVC/SVR, forests, neighbors, preprocessing and Gaussian processes. Moving each
shared step into Mojo once removes it from every family that reaches it. In this order:

1. **Shared validation and label handling.** Each family reaches these helpers:

   | helpers | methods | families |
   |---|---:|---:|
   | `_labels` (`flatten_labels`, `sorted_classes`, `unique_inverse`, `encode_labels`, `label_kind`) | 230+ | 11 to 12 |
   | `linear_model._shape_of` | 170 | 13 |
   | `_sample_weight.check_sample_weight` (cluster, ensemble, kernel_methods, linear_model, neighbors, preprocessing, svm) | 81 | 7 |

   Target: one native argument-and-label entry that the public shell calls as part of its one native call.
   The entry takes the raw buffers plus the weight and label options. It returns encoded labels, classes
   and validated weights inside the same native operation, not as a separate Python step.
2. **Input preparation and buffer handling.**

   | helpers | methods | families |
   |---|---:|---:|
   | `_arena_io` (`DeviceCache`, `pack_ins`, `pack_outs`, `input_ranges`, `rows_slot`) | 235 | 12 |
   | Python program builders: `_expansion_metrics._Prog` (`put`, `alloc`, `stage`, `run`), plus the per-family copies such as `_expansion_prep._Prog` | 130 | 12 |

   The Python allocations come from these, which is why preprocessing alone counts 1,538 allocations
   summed over its methods. Target: the native operation owns staging, scratch allocation and the program
   it runs. Python passes the input buffers and receives the outputs.
3. **Route selection.** Route branches are the largest single count, summed per method: preprocessing
   5,274, ensemble 3,002, metrics 2,302, decomposition 1,875 and linear_model 1,489. They choose the
   numeric mode, vendor/CPU, engine (forest sequential vs resident groves), solver and the fast/identical
   arms in Python. Target: Python resolves the mode once, chooses the binding and passes the request.
   The native entry picks the route.
4. **Result handling.**

   | operation | methods | python | path |
   |---|---:|---:|---|
   | `score` | 92 | 83 | `_expansion_metrics` (`_Reg`, `_Sums`, `_execute`, `r2_score`, `accuracy_fraction`) on top of `predict`, so it pays both paths |
   | `save` / `load` | 98 | 95 | `_serialize`: `.npz`/`.npy` encode and decode, header parsing, dtype description. 14 families reach it |

   Target: a native `score` per estimator that predicts and scores in one call, and a native save/load
   entry. `linear_model._saved_mode`, `_restore_mode` and `_check_saved_by` move with save/load.

### 2. Families with native owners already staged

1. **linear_model: LinearRegression and Ridge.** `python/mojolearn/_native_classical_public.py` holds the
   one-call shells (`entry(self, "ols_fit" | "ridge_fit" | "predict_public" | "score_public" | "save" |
   "load", args, context)`). `_native_classical_bridge.connect` and `connect_host` hold the compiled
   dispatcher. Today `_install` runs only in qualification. To land, `linear_model.LinearRegression` and
   `Ridge` define those shells as their own source methods over the connected dispatcher, through a
   module-level `_ENTRY, _CONTEXT = connect(...)`, which the measure counts as native. The old Python
   paths are then deleted: 16 native calls, 26 allocations and 31 route branches today. Expected after:
   10 methods native.
2. **PCA.** `python/mojolearn/_native_pca_bridge.connect` connects the native PCA owner (`pca_fit`,
   `pca_transform`, `inverse_transform`, `pca_rank_finish`, optional whiten/full entries). Today
   `PCA.fit` makes 14 native calls through 45 helpers. `PCA.fit`, `transform`, `fit_transform` and
   `inverse_transform` become one dispatcher call each, the same way as linear_model.

### 3. The rest by size (non-native methods today)

1. preprocessing (135)
2. ensemble (132): forests and GBDT adapters
3. the rest of linear_model (92): LogisticRegression, SGD, GLM, CV
4. decomposition (101)
5. sequence (81)
6. neighbors (73)
7. metrics (69): its functions become one native entry each once result handling is native
8. cluster (47): KMeans first
9. neural_network (45)
10. model_selection (25)
11. svm (23): SVC/SVR
12. manifold (20)
13. training (17)
14. linalg (17)
15. kernel_methods (15)
16. gaussian_process (13)
17. mixture (13)
18. neural_inference (6)
19. resample (5)
20. language_model (2)
21. ops (1)
22. models (1)

A family migration lane rewrites the family's public method bodies into one-call shells. In the same
commit, it deletes the Python path it replaced and refreshes the baseline with `--baseline`. The commit
must show the family's non-native count going down and no verdict getting worse.
