Full classification input preparation
====================================

`six_lane_prepare_full_classification.py` prepares a **new input variant**,
`classification-full-v1`. It does not register that variant, create a timing
queue, run models, compile, or admit results. Original 1M/100k inputs and their
receipts remain untouched. A directory named `rows-full` is not coverage proof.

Planning reads committed source and retained small JSON only:

```sh
python3 tools/six_lane_prepare_full_classification.py plan \
  --source-sha REVIEWED_COMMIT \
  --audit remaining-expanded-audit.json \
  --inventory retained-small-dataset-metadata.json \
  --output classification-full-plan.json
```

The eligibility audit must match the exact benchmark source hash. Plans retain
each original registration and its parameter declarations; they exclude lanes
with intrinsic subsets. The currently reviewed Apple scope contains 48 cells
using `cls`, `raw`, and `cat` blocks. Lane-specific transforms and constructors
remain in the original harness. Exact resolved settings, output scope, artifact
provenance and independent quality still require later variant admission.

Population and transformations
------------------------------

* Taxi uses the original loader's credit-card filter, then its own last 500,000
  filtered trips as queries: 4,110,786 training rows. Regression's all-trip
  5,250,086-row training population is not interchangeable.
* Istella uses all 2,043,304 training rows and the **first** 500,000 rows of the
  separate test file, with binary target `relevance > 0`.
* `cls` retains sentinel cleanup and the original float64 mean/std recipe fitted
  on full training data. `raw` retains sentinel cleanup without standardization.
* Taxi categorical data uses the supported categorical loader's train-fitted
  dense codes, original five categorical columns and unknown-query bucket.
  Istella retains the original eight highest-variance columns and 16 quantile
  codes, fitted on all training rows. This is dataset preprocessing, not a
  kernel dispatch rule. Removing row sampling changes fitted preprocessing;
  the output is therefore explicitly a new input variant.

Only after root review, an exact frozen checkout may run `prepare` with the
reviewed plan digest, original archive directory and a **new** output directory.
Preparation refuses until the original CPU14 and its selector-repair status are
terminal, then acquires the canonical M3 `gpu.lock` nonblockingly and checks the
statuses again. It hashes original archives, uses existing source loaders, and
records actual shapes, array hashes, categorical column/edge choices, archive
hashes and source provenance. Hashes are not invented during planning.

Conventional `cls-taxi.npz` names exist only inside the new variant directory so
the original loaders can read them. Sidecars mark `changes_frozen_race=true` and
the variant. Never attach these files to an old capped race. Failed/partial
preparation remains on disk and must use a fresh directory on retry.

This helper is preparation-only. It contains no estimator runtime changes and
does not substitute unsupported Mojo/Modular operations. No preparation has
been run as part of its initial delivery.
