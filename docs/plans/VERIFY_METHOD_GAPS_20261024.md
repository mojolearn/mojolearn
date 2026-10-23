# Changed public-method verifier gaps — 2026-10-24

The retained audit at `~/mojolearn-evidence/audit-storage-coverage-20261023/verify-coverage.json`
compared `fc8d425fb` with `857c0c176`: 975 changed public methods, 953 with a
verifier lane, 22 without one. Its process exited zero; that is an audit completion
status, not proof that every method was covered.

No existing **public verifier call** was found for these 22 endpoints in the
retained audit and targeted lane-source review. Shared native implementations
being reached by other methods do not establish coverage of these public doors.
The review found no alias exception that justifies a mapper whitelist or
reassignment to an unrelated lane. Existing pytest evidence is separate from
verifier coverage; the new direct calls make the required reach reviewable.

| Endpoint(s) | Existing tests in the audited source | New consumed-output lane |
|---|---|---|
| `DynamicTheta.fit`, `.predict` | Optional statsforecast comparison returns without exercising the endpoint when statsforecast is absent | `public-theta-variants` |
| `OptimizedTheta.fit`, `.predict` | No direct real test found; `sequence-theta` exercised AutoTheta and DynamicOptimizedTheta instead | `public-theta-variants` |
| `ensemble.catboost_auto_learning_rate` | Real table/default comparisons in `test_gbdt_catboost_defaults.py` and public helper checks in `test_parallel_public_drivers.py` | `public-numeric-metadata` |
| `johnson_lindenstrauss_min_dim` | No direct real test found | `public-numeric-metadata` |
| `training.vendor_used` | Real provenance checks in `test_training_surface.py` and other training controllers | `public-numeric-metadata` |
| `sparse_encode` | DictionaryLearning/SparseCoder tests cover shared internals, not this direct function | `public-sparse-encode` |
| `training.residual_dropout`, `.residual_dropout_backward` | No direct public test found in the package test directory | `public-residual-dropout` |
| `model_selection.check_cv`, `.get_scorer`, `.get_scorer_names`, `.make_scorer` | No direct real test found for these public functions | `public-scorer-factories` |
| `model_selection.validation_curve`, `.learning_curve` | `_RowSpy` tests in `test_py_bugs.py` test orchestration with a fake estimator | `public-validation-learning-curves` |
| `model_selection.RandomizedSearchCV.fit`, `.predict`, `.score`, `.predict_proba`, `.decision_function`, `.transform` | No direct real test found | `public-randomized-search` |

`tools/identity_lanes/public_gap_methods.py` adds seven independent lanes. Each
calls the actual public endpoint and consumes its outputs. Vendor identity is
checked against the selected backend; its name deliberately differs between
vendors, so the lane hashes the checked result rather than asserting identical
vendor strings. This is metadata/provenance coverage, not a numerical equivalence
claim about that string.

New package tests cover Theta batch/single-series equivalence without an optional
oracle dependency, the stated Johnson–Lindenstrauss bound and refusals, analytical
sparse-coding results on an orthonormal dictionary, and residual-dropout forward /
backward mask consistency and refusals. Model-selection tests use real Ridge,
LogisticRegression and StandardScaler objects; curve results are checked against
separately fitted fold models and independently computed losses. Search tests
consume refit state and each delegated response. Existing CatBoost table tests
remain required.

## Qualification status

**Authored only: no tests, native executions, builds or reference recordings were
run by this source lane.** The seven verifier lanes are OWED until executed and
admitted through the existing process. Their numerical parts require the normal
Apple/NVIDIA/AMD identical contract and the declared CPU coverage. Existing raw
references and failure evidence remain unchanged.

The twelve model-selection endpoints also depend on the active native CV/search
migration. Scorer registration alone does not qualify search `set_params`, curve
folds, estimator cloning, delegated responses or every built-in family. Any failure
or missing implementation must remain explicit; these lanes must not be skipped,
expected-failed or marked qualified to make the coverage count look green.

Root-owned validation:

```sh
pixi run -e test python -m pytest \
  python/mojolearn/tests/test_public_verifier_gap_methods.py \
  python/mojolearn/tests/test_public_model_selection_verifier_gaps.py \
  python/mojolearn/tests/test_gbdt_catboost_defaults.py
python3 tools/verify_coverage_audit.py --base fc8d425fb --json /RETAINED/PATH/verify-coverage.json
```

The full audit must show the actual new lane reach; a passing authoring/AST check
alone does not establish it. Execute the new lanes against the frozen native
candidate, retain cell coverage/exits/provenance, review fault-arm applicability,
and record/admit matching references. Do not repeat already qualified unrelated
lanes merely because these tests and verifier fragments were added.
