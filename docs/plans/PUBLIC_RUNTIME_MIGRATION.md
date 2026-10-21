# Public runtime migration

Andrew, 2026-10-21. This plan defines the contract for every public method, how we measure it, where the
package stands today, and the order of the migration.

## The contract

Python connects the user to Mojo, and Mojo does the work. A public method (`fit`, `predict`, `transform`,
`score`, `partial_fit`, `sample`, `decision_function`, `predict_proba`, `generate`, `step`, `save`/`load`,
...) runs in this order:

    user call -> thin Python binding (argument checks, choose the binding, pass buffers)
              -> ONE native call that prepares, executes and produces the results
              -> return

Outside that native call, Python does not prepare inputs beyond conversion, choose execution routes,
allocate working buffers, sequence several native calls or post-process results. PyPI and the
scikit-learn compatible interface are only the shell. They are never a reason to keep implementation
logic in Python.

A standalone native smoke test does not prove that a public method is native, and neither does a green
kernel check. Only the method's own call path proves it.

There are no before/after A/Bs for the migration itself. Bits may change within a version, and when they
do they are recorded on every vendor together. Prefer bit-exact moves: the same operations in the same
order.

## The measure

`tools/public_runtime_inventory.py` is the measure. It reads the code statically and imports nothing.
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

## Today (main f180228eb, baseline tools/hooks/public_runtime_baseline.tsv)

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
