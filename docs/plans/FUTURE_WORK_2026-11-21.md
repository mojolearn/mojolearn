# Future work, 2026-11-21

Work Andrew put on hold at the end of the device=/check/boundary wave ("no major new projects"). Nothing here is
started. Each item gets its own brief and lane when Andrew reopens it.

## 1. Categorical features in the oblivious-tree (CatBoost-style) booster (Andrew, 2026-11-21: important)

Where `GradientBoosting` stands against CatBoost on categorical input today:

| | CatBoost | Mojolearn |
|---|---|---|
| Input | any values (text or numbers) | codes 0 to k-1, every code present in the training rows; anything else is refused by name |
| Row order | shuffles the rows itself | uses the rows in the order given, so the caller must shuffle (with a recorded seed) |
| Combinations of categorical columns (e.g. site x scanner) | builds them | not in `GradientBoosting`; only `ExperimentalTwoLevelFeatureFreq` (`python/mojolearn/ensemble.py:1198`, one tree, depth two, RMSE) has a limited version, and its note says it does not imply general support |

Code today: `gbdt/options/catboost_options.mojo:857` (`combination_ctrs` NOT HONORED; `check()` refuses
`max_ctr_complexity` above 1), `gbdt/models/hash.mojo` (the combination hash fold already exists),
`docs/plans/gbdt-cpu-ctr-eval-20261027.md` (CTR evaluation notes).

What CatBoost does with a combination, which is what we must match: while it grows each oblivious tree, at every
level after the first it forms new candidate features by pairing the categorical features (and the numeric
splits) already in the tree with each other categorical feature, up to `max_ctr_complexity` columns per
combination (CatBoost's default is 4). Each combination is a new category: the hash of the row's values in those
columns. Each new category gets its target statistic (CTR: smoothed mean target, counter, buckets, with the
priors) computed in the ORDERED way (each row sees only the rows before it in a random permutation, so its own
target never leaks into its feature), and the split is chosen among them like any other. Combinations picked by
the trees are kept, and their CTR tables over all training rows go into the model for predict. This is how
CatBoost learns interactions such as site x scanner without the user building the column.

What Mojolearn has now: single categorical columns only (one-hot when cardinality is small, otherwise a single-column
ordered CTR, `gbdt/ctrs/`), over the rows in the caller's order; permutation-dependent CTR datasets under
`boosting_type='Ordered'` are listed NOT IMPLEMENTED (`gbdt/NOT_IMPLEMENTED.tsv`).

Target: match CatBoost's categorical handling in full (Andrew, 2026-11-21), including combinations with the
NUMERIC splits already in the tree, not only categorical x categorical. Every piece runs on the GPU and on the CPU,
and in `identical` mode (the default) the GPU and CPU give the same bits as each other and across vendors. `fast`
gets the same features, judged on speed and quality only. There is no `deterministic` mode: it is retired and
`identical` replaces it (only the refusal tests for the old value remain).

Work, in order:
1. **Native category encoding at fit.** Accept arbitrary category values (strings, integers, unseen codes) and map
   them to codes in Mojo at UNPACK, with the map stored in the model and applied at predict (unseen value -> the
   CTR prior, as CatBoost does). No Python dict or loop over values.
2. **Seeded permutation inside the fit.** The ordered CTR / ordered boosting permutation is drawn in Mojo from
   `random_state`, identical on every vendor and the host column (a counter-based generator keyed on the row
   index, not a sequential shuffle). The caller no longer shuffles; the receipt records the seed.
3. **Feature combinations (tree CTRs, `max_ctr_complexity` > 1, CatBoost's default 4).** Greedy combination of
   each categorical feature with the features already in the oblivious tree: the categorical features AND the
   binarized numeric splits (each numeric split contributes its 0/1 side to the combination hash, as in CatBoost), combination hashes on the GPU with the existing
   fold (`gbdt/models/hash.mojo`), CTR tables per combination. Remove the `check()` refusal and the
   `ExperimentalTwoLevelFeatureFreq` class (tombstone) once the general route covers it.
4. **Permutation-dependent CTRs under `boosting_type='Ordered'`** (`gbdt/NOT_IMPLEMENTED.tsv`), for single
   columns and combinations.
5. Every step: `identical` gives the same bits on the GPU and the CPU and on every vendor (fixed fold order and the
   counter-based permutation on both devices; the CPU route is the same algorithm, never a different one); quality compared against CatBoost on the same
   explicit parameters (Race fairness); A/B on NVIDIA per the measurement rules; no tuning to a board dataset.

## 2. Held from the device=/boundary wave (2026-11-20/21)

- Member ABI rollout (`core/member_abi.mojo`, note `dw-core-MEMBER-ABI.md`): bagging, AdaBoost, meta-estimators,
  explainers, model-selection member-calling doors.
- Arena coordinated change.
- Label-step call sites for the 28 classifiers.
- Pools onto the typed frame.
- Byte trainer native state.
- CausalLM checkpoint, construct and config doors; the CausalLM resident session lease (stale-capture window:
  Andrew to decide).

## 3. Public API matches what users already know (Andrew, 2026-11-21)

Goal: someone who knows scikit-learn, PyTorch or CatBoost/XGBoost/LightGBM can use Mojolearn without reading our
docs. Work:
1. A generated table, per public class, of its scikit-learn twin: constructor parameter names, defaults and
   accepted values; methods (`fit`, `predict`, `predict_proba`, `transform`, `score`, `partial_fit`, ...); fitted
   attributes (`coef_`, `classes_`, `n_features_in_`, `feature_importances_`, ...). Every difference is listed and
   either fixed or written down with its reason (an identity requirement, a feature we do not have).
2. Run scikit-learn's own estimator checks (`check_estimator`) on every class that claims a twin, as a test of the
   Python boundary.
3. Gradient boosting: CatBoost / XGBoost / LightGBM parameter names accepted where the meaning is the same.
4. Neural models and the Torch adapter: PyTorch conventions (`state_dict`, `train()`/`eval()`, ~~device names~~ device is 'gpu' or 'cpu' only (Andrew, 2026-11-21),
   dtype names) where the meaning is the same.
5. Nothing here may add Python work inside a call: renames and defaults are resolved at construction.

## 4. Owed small items (not projects)

- ~~A/B `abb1b` never ran (`mcd-lockstep-off`, `sgd-ps-preload`).~~ Both timed in the 2026-11-24 A/B jobs and deleted
  (docs/TOMBSTONES.md).
- Arm batch b2: `fast-vendor-math` fails to compile on sm_89 (twice); b3 never built.
- Call-profile report for long-row profile nvc1-0005 (report phase failed rc=8 after a capped nsys pass): run
  `tools/ops/call_profile.py report` by hand and fix the tool so a capped pass still writes the report.
- Defect: `bindings/archive_schema_native_door.mojo:982` reads unlocked `ANN_REGISTRY` / `IVF_FLAT_REGISTRY`
  with the GIL released.
- First GPU run of `tools/ops/feature_proof_job.sh` (device=, check, receipt/reproduce, verify options) once the
  integration compile is green.
