# The benchmark board

`tools/bench_board.py` is one script, the same file on every box. It times
mojolearn from the installed PyPI wheel against the opponent libraries on the
same box, in the same run, with the same settings, interleaved round by round,
and records quality next to every time. See the script's docstring for the
full contract. In short:

| box | modes | opponents |
|---|---|---|
| Apple (Metal) | `fast` and `identical`, interleaved in one race (trees, classical, classical2); `identical` only (neural) | CatBoost, XGBoost, LightGBM, scikit-learn, umap-learn, statsmodels and faiss-cpu on the CPU; torch on MPS (classical, neural) and on the CPU (neural `*-infer`) |
| NVIDIA | `identical` | CatBoost, XGBoost and LightGBM GPU arms, cuML, cuVS, torch CUDA (the rosters of `tools/bench_all_ours.sh`); scikit-learn and statsmodels on the CPU where cuML has no such estimator |
| AMD | `identical` | XGBoost ROCm where the image has it, otherwise the CPU learners on all cores (scikit-learn, umap-learn, statsmodels, faiss-cpu included); torch ROCm |

- Families: trees (`gbdt-symmetric`, `gbdt-symmetric-1000`, `gbdt-depthwise`, `gbdt-lossguide`, `rf`,
  `et`, `iforest`, and the GBDT task lanes `gbdt-rank-yetirank`,
  `gbdt-rank-pairlogit`, `gbdt-multiclass`, `gbdt-categorical`, below) and classical (`kmeans`, `pca`, `ols`, `knn`, `kde`, `svc`,
  `dbscan`, `hdbscan`) on taxi and Istella-S, classical2 (23 lanes, below) on
  taxi and Istella-S or seeded synthetic series, and neural (16 lanes,
  below) on inputs the driver builds from seed 7.
- Neural is `identical` only on every vendor, because the wheel builds its
  neural surface in that tier only. `--modes fast` with the neural family is
  refused by name; a FAST-only Apple run passes
  `--families trees,classical,classical2`.
- One seed per race: 7, or 42 where the cuML benchmark sets 42 (`spectral`,
  `algos/target-encoder`). Five timed rounds after one warm-up (`--rounds`).
- Settings: NVIDIA's own benchmark values on every lane their harnesses
  cover, on our arm and every opponent arm (below, "NVIDIA's harnesses").
- Output: one directory with `board.json` (box fingerprint and every cell) and
  `BOARD.md`. Bulky state (venv, wheel download, classical blocks) goes in
  `--cache`, which defaults to `<out>/cache`. Rerunning the same command
  resumes: finished races are skipped and failed ones are retried
  (`--skip-failed` turns that off). `--rerun trees/rf/,trees/et/ --rerun-before <UTC ISO>` runs
  finished races again (after a driver fix), keeping each earlier record under
  `superseded` in `board.json`. A resume on a different box or a
  different wheel is refused.
- Data is never downloaded. taxi and Istella-S come from R2
  (`docs/REMOTE_DATA_R2.md`), and the ranking lanes also read
  `gbm-bench/istella/istella_rank.npz` (the query ids). Without them the
  script refuses and prints the staging command. A neural-only run
  (`--families neural`) needs no dataset.
- Always check the plan first: `python3 tools/bench_board.py --dry-run
  --vendor apple` (or `nvidia`, `amd`). On 2026-09-29 the plan has 441
  races on every vendor (19 trees, 16 classical, 44 classical2, 16 neural,
  346 algos). That comes to 1,939 fit cells on Apple, 1,695 on NVIDIA and
  1,525 on AMD, 435 of them our CPU tier (below). Inference adds 1,310 cells
  on Apple, 1,144 on NVIDIA and 1,038 on AMD; `--no-infer` times training
  only.

## NVIDIA's harnesses

The board takes its settings from NVIDIA's two public benchmark harnesses.
For every lane one of them covers, each parameter the harness sets
explicitly is the board's value, on our arm and on every opponent arm. Where
the harness leaves a work setting at each library's default, the board
pins ONE value on every arm (the PINNED table), because the libraries'
defaults differ and an unset default is how two arms end up solving two
problems. `tools/bench_board_harness.py` holds the copied values, the
source file and commit of each, and the lane mapping; every race's settings
carry `config` (the harness, entry, file URL and commit, or "the board's
own settings (no NVIDIA harness entry)"), and BOARD.md prints it under the
race. `tools/bench_board_params.py` refuses a race whose arms differ in the
seed or in a shared parameter.

| harness | file | commit |
|---|---|---|
| cuML benchmark (RAPIDS) | [python/cuml/cuml/benchmark/algorithms.py](https://github.com/rapidsai/cuml/blob/e0f7a4e31578c8eeef376f3ce715d846bfee8d4c/python/cuml/cuml/benchmark/algorithms.py) (`AlgorithmPair` shared_args, cpu_args, cuml_args) | `e0f7a4e31578c8eeef376f3ce715d846bfee8d4c` |
| NVIDIA gbm-bench | [algorithms.py](https://github.com/NVIDIA/gbm-bench/blob/73a976b036249ff9d8cb30cf9082bb414b911379/algorithms.py) (`shared_params`, each library's `configure`) and `runme.py` (`-ntrees` 500) | `73a976b036249ff9d8cb30cf9082bb414b911379` |

What the board takes from them:

- Trees (gbm-bench): `gbdt-symmetric`, `gbdt-depthwise`, `gbdt-lossguide`,
  `gbdt-multiclass` and `gbdt-categorical` run 500 trees, max_depth 8,
  learning rate 0.1, L2 1 and 256 leaves (LightGBM's `max_leaves`, which is
  2^8 on every arm), and on a binary task `scale_pos_weight` =
  len(y_train) / count_nonzero(y_train) on CatBoost, XGBoost and LightGBM
  (ours: `class_weights=[1, that]`). `rf` runs gbm-bench's forest values,
  max_depth 8 and 500 trees. gbm-bench gives CatBoost `MultiClassOneVsAll`
  beside XGBoost's softmax; the board keeps one loss, softmax, on every arm
  of `gbdt-multiclass`.
- `gbdt-symmetric-1000` is `gbdt-symmetric` at 1000 trees, with the same
  arms, datasets and pinned settings. Oblivious trees are weaker per tree,
  and 1000 is CatBoost's own default iteration count
  ([boosting_options.cpp](https://github.com/catboost/catboost/blob/e628c03fb0e6b760592652a995163f26be7ea7d3/catboost/private/libs/options/boosting_options.cpp#L13), `IterationCount("iterations", 1000)`,
  commit `e628c03fb0e6b760592652a995163f26be7ea7d3`).
- Classical (cuML): `kmeans` 8 clusters, `init='k-means++'`, 300
  iterations, `n_init=1`, `oversampling_factor=0` on ours and cuML; `pca`
  10 components; `knn` 64 neighbors (scikit-learn `algorithm='brute'`);
  `kde` Gaussian kernel, bandwidth 1.0; `dbscan` eps 3, min_samples 2
  (scikit-learn `algorithm='brute'`); `svc` RBF kernel.
- Classical2 (cuML): `umap` 5 neighbors, 500 epochs; `spectral` 8 clusters,
  nearest-neighbors affinity, 10 neighbors, `n_init=1`, seed 42; `tsvd` 10
  components; `elasticnet` alpha 0.1, l1_ratio 0.5; `agglomerative` 8
  clusters, single linkage; `svr` RBF kernel.
- Algos (cuML): `incremental-pca`, `gaussian-rp` and `sparse-rp` 10
  components; `target-encoder` smooth 0, 4 folds, seed 42 (cuML folds
  interleaved); `onehot` dense output, unknown categories ignored;
  `sgd-clf` and `sgd-reg` 100 epochs at a constant rate 0.005.
- The pairs that pass nothing (`ols`, `hdbscan`, `logreg`, `linearsvc`,
  `ridge`, `lasso`, `linearsvr`, `knn-clf`, `knn-reg`, `kernel-ridge`,
  `tsne`, the naive Bayes classifiers, the scalers and encoders) keep the
  board's pinned values: the harness leaves every parameter at the
  library default there.

PINNED (the harness leaves these at library defaults that differ):

| parameter | value on every arm | why |
|---|---|---|
| seed | 7 (42 on spectral and target-encoder, where cuML's benchmark sets 42) | gbm-bench sets none and most cuML pairs set none; each library's default differs (XGBoost 0, CatBoost 0, LightGBM its own, scikit-learn and cuML None) |
| bins | 254 borders = 255 bins on every boosted arm | defaults differ (ours 128, CatBoost CPU 254 borders, CatBoost GPU 128, XGBoost 256, LightGBM 255 bins) |
| row and column sampling | none (CatBoost and ours bootstrap_type 'No', subsample 1.0 and colsample 1.0 elsewhere) | CatBoost's default bootstrap samples rows; the others do not |
| boosting_type | Plain | CatBoost's default is data-dependent (Ordered on small pools) |
| max_leaves | 2 ** max_depth = 256 on every boosted arm | gbm-bench gives LightGBM 256; the others take the same cap at depth 8 |
| leaf estimation, split floors, borders, nan_mode, boost_from_average | as speed_gbdt_arm.lane_config pins them | the libraries' defaults differ in meaning (lane_config docstring) |
| rf n_bins | 128 (ours and cuML) | cuML's default; scikit-learn searches exact thresholds and has no bin count |
| rf max_features, bootstrap, max_samples, min_samples_leaf | 'sqrt' (classification) or 1.0, True, 1.0, 1 | each library's own default for the task, pinned so a default change cannot move one arm |
| kmeans tol | 1e-7 | scikit-learn and cuML default 1e-4, ours its own; the board keeps one value (ours and cuML refuse 0) |
| every other parameter a lane sets today | its current value, on every arm | the harness leaves it at the library default |

No harness entry, so the board's own settings stand: `et`, `iforest`,
`gbdt-rank-yetirank`, `gbdt-rank-pairlogit`; classical2 `gmm`, `gpr`,
`gpc`, `nystroem`, `rbf-sampler`, `arima`, `ets`, `ivf`,
`spectral-embedding`; every neural lane (the neural races keep their own
settings); and every algos lane not listed above.

The seed check. An arm whose constructor has no seed parameter at all draws
nothing and is recorded as `seed: none (deterministic)`. A third-party arm
that draws random numbers without a seed argument (cuVS IndexParams, cuGraph
louvain, faiss HNSW) or through a seeded function argument (SelectKBest's
mutual information) keeps an EXCEPTIONS row with its reason. An arm that
has a seed parameter holding anything but the lane's seed refuses the race.

## Our CPU tier (`ours-cpu`)

mojolearn trains and predicts on a CPU-only install through its host
bindings. The public switch is `MOJOLEARN_VENDOR=cpu` before import
(`python/mojolearn/_backend.py`): no GPU set loads, and the host bindings
under `mojolearn/host/` answer. They build IDENTICAL only. The board races
this tier on every vendor as the arm `ours-cpu`. It is the same public
estimator as `ours`, run in a worker process started under the switch, and
it races the CPU opponents already on the board (scikit-learn, the CPU
learners of XGBoost, LightGBM and CatBoost, statsmodels, umap-learn,
faiss-cpu). The worker reads back `mojolearn.vendor()` and the binding's tier
before it times anything. If the installed wheel did not load its CPU set,
the cell is REFUSED by name, so a Metal or CUDA fit is never labelled CPU.
`--no-cpu-arm` (leg knob `MOJOLEARN_BOARD_NO_CPU_ARM=1`) turns the arm off.

- Planned on all 12 trees races, all 16 classical and all 44 classical2
  races, and the 10 neural lanes that run on the GPU: 82 cells on every
  vendor. Each of these estimators routes to a host family
  (`host_surface.routed_modules`). A GBDT configuration that the host side
  does not restate (`host_surface.NO_CPU_PATH`) is refused by name in its
  cell.
- Not planned on the six neural `*-infer` lanes, whose `ours` arm already is
  the CPU path (the `*Inference` classes), and not on a FAST-only run. The
  dry run and the board's "Not covered" name both.
- Trees: the trees driver runs every arm in one process, and the switch
  applies to a whole process. So `forest_speed_arm.py --ours-cpu` adds a
  proxy arm (`bench/speed/forest_board_arms.py`) whose worker loads the same
  rows through the same loader and builds the same estimator. The proxy takes
  its turn in the round-robin like every other arm. The conductor's clock
  covers the worker's fit plus one pipe round trip, and the worker's own
  clock is printed beside it (`FSPEED-CPU-ROUND`). Predictions and scores
  come back after the clock. The classical, classical2 and neural racers
  already run one worker per arm, so there `ours-cpu` is just one more
  worker.
- Quality: `bits_equal_vs_ours_identical` on the `ours-cpu` cell compares
  its output with our GPU IDENTICAL arm's in the same race. The classical,
  classical2 and neural drivers compare the saved output arrays byte for
  byte, including every training step's loss. Trees compare the prediction
  hash of the last timed round, and in the inference phase they compare the
  prediction vectors (`FSPEED-INFER-AGREE arms=ours,ours-cpu`).
- `ours CPU / arm` is its median over each opponent's median. Our CPU and GPU
  times are never divided by each other. "Our CPU tier at a glance" lists
  each race's CPU median, the bit check and the CPU opponents.
- The macOS wheel ignored `MOJOLEARN_VENDOR=cpu` through 0.8.22 (the flat
  layout returned before reading it). Main fixes this in `_backend._layout`.
  On a 0.8.22 Mac, `ours-cpu` is therefore REFUSED by name. On the Linux
  wheel the switch works from 0.8.22.

## Memory

Every arm of every fit cell records `peak_host_mb` and `peak_gpu_mb` in
`board.json` and in `BOARD.md`. Each value is the highest per-round peak over
the timed rounds; the warm-up is recorded apart. Every value carries its
method (`memory.host_method`, `memory.gpu_method`), which is printed under
each table. `tools/bench_board_probe.py` resets and reads each peak around
the timed call, outside the clock:

| what | how |
|---|---|
| host, Linux | `VmHWM` after writing 5 to `/proc/self/clear_refs` (the kernel's resettable peak RSS) |
| host, macOS | `proc_pid_rusage` `ri_interval_max_phys_footprint` after `proc_reset_footprint_interval`: the peak physical footprint over the round. On Apple silicon it includes Metal buffers |
| host, child processes | resident size of the worker's descendants (joblib or loky pools, torch.compile workers) at the round's end, as `memory.children_mb` |
| GPU, torch on CUDA or ROCm | `torch.cuda.max_memory_allocated`, reset before the round |
| GPU, torch on MPS | `torch.mps.driver_allocated_memory` at the round's end |
| GPU, ours, cuML, cuVS, XGBoost, CatBoost, LightGBM | the driver's per-process figure at the round's end: `nvidia-smi --query-compute-apps` or `rocm-smi --showpids`. This is the process total, context and pools included |
| GPU, Apple (non-torch) | none: there is no per-process counter, and the Metal share is inside the host footprint |
| CPU arms | GPU none |

The classical, classical2 and neural racers run one arm per worker, so each
figure belongs to its own arm. The trees driver runs its arms in one
process. Its host peak is still per arm, because it is reset around each fit
(`FSPEED-MEM` lines, `--mem`). Its GPU figure is the whole process, and the
method says so. The inference cells of the classical lanes carry memory too;
the trees inference cells do not.

## Inference

Training is not the only clock. After a race's fit rounds, every arm predicts
with its own last fitted model from those rounds (no fit is retimed), on the
same rows and in the same output kind, one warm-up and then the timed rounds,
arms interleaved. These are separate cells (`infer_cells` in `board.json`)
with their own ratios, under each race's fit table and in "Inference at a
glance". Our FAST and IDENTICAL predictions on the same rows are compared bit
for bit.

Trees (`forest_speed_arm.py --infer`; the flag is off by default and the
driver's output is unchanged without it) time two batches: `test`, the
held-out split the accuracy column scores, and `large`, the first 1,000,000
training rows (capped at the training rows). Every clock is host rows in and
host predictions out. The output is P(class 1) on the binary tasks and the
anomaly score for iforest.

| arm | the timed call | why this path |
|---|---|---|
| ours | `predict_proba(X)[:, 1]`; iforest `score_samples(X)` | the public surface. iforest rebuilds its forest inside every scoring call (DEVIATION 874), so its clock includes a build |
| XGBoost | `Booster.inplace_predict(X)`; on a CUDA booster, `inplace_predict(cupy.asarray(X))` then `cupy.asnumpy` | XGBoost documents in-place prediction as its fastest path; host rows on a CUDA booster fall back to a DMatrix, so the rows go up and the result comes back inside the clock |
| LightGBM | `Booster.predict(X)` | its predict runs on the CPU whatever device trained it |
| CatBoost | `predict_proba(X, task_type=...)`, GPU on the `-gpu` arm | CatBoost's own GPU apply. If a build refuses it, the arm applies on the CPU and says so |
| scikit-learn | `predict_proba(X)` or `score_samples(X)`, `n_jobs=-1` | its only path |
| cuML | RF converted once to FIL outside the clock (a model load), then FIL `predict_proba(X)`; IsolationForest `score_samples(X)` | FIL is cuML's forest inference |

Each arm's call is printed under its table (the driver's FSPEED-INFER-PATH
line). Quality: the FSPEED-ACC metric recomputed from the timed output on the
held-out rows (`<metric>_matches_fit` says it equals the fit-time value), and
on our FAST arm `bits_equal_vs_ours_identical` and
`max_abs_diff_vs_ours_identical`.

Classical (`classical_two_datasets.py race --infer`, off by default there
too): kmeans `predict`, pca `transform`, ols `predict` and svc `predict` on
the eval rows (the 500,000 test rows for kmeans, pca and ols; 10,000 for
svc). The clock span is the fit's: ours takes host rows and returns host
results; the torch and cuML arms upload the rows before their clock, which
ends at the device synchronize (`SPAN-ASYMMETRIC`). Quality: kmeans
`eval_inertia` and `label_agreement_own_centers`, pca
`transform_max_rel_err_own_fp64`, ols `r2_eval`, `rmse_eval` and
`predict_max_rel_err_own_fp64`, svc `accuracy_eval`, each against a float64
NumPy evaluation of the arm's own fitted model, plus `bits_equal_vs_ours` on
every arm (on `ours-fast` it is the FAST against IDENTICAL check). kNN
(`kneighbors`) and KDE (`score_samples`) already time inference as their race;
DBSCAN and HDBSCAN have no predict.

The GBDT task lanes time inference the same way: the whole probability
matrix on `gbdt-multiclass` (quality `mlogloss` and `accuracy`), raw scores
on `gbdt-rank-*` (NDCG@10, NDCG@5 and MAP on the test batch), and P(class 1)
on `gbdt-categorical`, where CatBoost and XGBoost predict from the frame kind
their fit took (int64 categorical columns; a pandas `CategoricalDtype`
frame through `XGBClassifier.predict_proba`), built inside the clock as the
fit clock builds it.

Not covered yet: a single-row latency batch; ONNX, Treelite and other export paths; the
classical2 family's predict calls as separate inference cells (its lanes
define their own clocks in `tools/bench_board_more.py`); svc
`decision_function`.

## The GBDT task lanes

Four trees lanes race the public `GradientBoosting` on tasks beyond binary
and regression, through the same driver (`bench/speed/forest_speed_arm.py`),
the same interleaving, FAST and IDENTICAL on Apple and IDENTICAL on NVIDIA
and AMD, with inference timed after the fit rounds. `gbdt-multiclass` and
`gbdt-categorical` take gbm-bench's values (500 trees, depth 8, 256 leaves,
learning rate 0.1, L2 1.0); the two ranking lanes, which gbm-bench does not
have, keep 100 trees, depth 6 and 64 leaves. All four pin 254 borders (255
bins), no bagging, Plain boosting and seed 7. Each lane's objectives and
every mismatch (one line each, with its reason) are `TASK_LANES` in
`tools/speed_gbdt_arm.py`; the board copies them into the race's
`settings.lane_config` and the driver prints them as `FSPEED-NOTE
metric=mismatch` lines.

| lane | board dataset (driver dataset) | ours | CatBoost | XGBoost | LightGBM | quality |
|---|---|---|---|---|---|---|
| `gbdt-rank-yetirank` | Istella-S (`istellarank`) | `YetiRank`, `group_id` | `CatBoostRanker` `YetiRank` | `XGBRanker` `rank:ndcg` | `LGBMRanker` `lambdarank` | NDCG@10, NDCG@5, MAP |
| `gbdt-rank-pairlogit` | Istella-S (`istellarank`) | `PairLogit`, `group_id` | `CatBoostRanker` `PairLogit` | `XGBRanker` `rank:pairwise` | not raced | NDCG@10, NDCG@5, MAP |
| `gbdt-multiclass` | taxi (`taximc`), Istella-S (`istellamc`) | `MultiClass` | `MultiClass` | `multi:softprob` | `multiclass` | multi-logloss, accuracy |
| `gbdt-categorical` | taxi (`taxicat`) | `cat_features` (CTR) | native `cat_features` | `enable_categorical` | `categorical_feature` | logloss, AUC |

- Growers. Our GradientBoosting fits the ranking and multiclass losses on
  SymmetricTree only (Depthwise and Lossguide refuse by name, as CatBoost's
  GPU learner does). Those lanes race CatBoost on the same oblivious grower,
  and XGBoost (depthwise) and LightGBM (leaf-wise) at the lane's depth
  (8 and 256 leaves for multiclass, 6 and 64 for ranking), labeled. The categorical lane runs Lossguide, which all four grow.
- Objectives. LambdaMART (`rank:ndcg`, `lambdarank`) and YetiRank are
  different losses. Each is that library's closest objective, and the lane
  says so. `rank:pairwise` is PairLogit's pairwise logistic loss with
  XGBoost's own pair sampling. LightGBM has no pairwise logistic objective,
  so `gbdt-rank-pairlogit` has no LightGBM arm. In every task lane LightGBM
  keeps `min_child_samples` 20 and `min_child_weight` 1e-3 at its defaults.
  At 0 (the gbdt lanes' value) LightGBM 4.7.0 aborts the first lambdarank,
  multiclass and categorical tree ("Check failed:
  (best_split_info.left_count) > (0)" in the Apple smoke).
- Ranking data. The train rows are `istella_speed.npz`'s. The query ids and
  the whole 681,250-row test split come from `istella_rank.npz`
  (`gbm-bench/istella/istella_rank.npz`, staged with the board's defaults).
  `--rows` cuts at a query boundary. XGBoost gets each query renumbered by
  order of appearance, because it refuses unsorted qids. NDCG uses gain
  2^grade - 1 and breaks score ties pessimistically. A query with no
  relevant document scores 1.0 in NDCG and in MAP (binary relevance, grade
  above 0, over the whole list). This is `tools/speed_gbdt_rank.py`'s
  definition, and a test holds the two implementations equal.
- Multiclass data. `taximc` is the tip share of the fare on card trips, cut
  at 20%, 25% and 30% into 4 classes (23.7%, 18.4%, 28.0% and 29.9% of the
  trips). Class 1 and above is exactly the binary taxi label. `istellamc` is
  Istella-S's grade 0..4 as 5 classes. XGBoost and LightGBM grow one tree per
  class per round, so the fit verdict divides their tree and leaf counts by
  the class count and says so (`FSPEED-FIT-NOTE per_class`).
- Categorical data. `taxicat` is the binary taxi task with vendor, rate
  code, store-and-forward flag, pickup zone and dropoff zone declared
  categorical on every arm. The two zone columns have about 260 categories
  each, so they reach the CTR and partition-search paths. Codes are dense
  within the train slice, and a test value never seen in training goes to
  one unknown bucket per column. criteo, the driver's other categorical set,
  is not in the R2 store (`bench/results/dataset_store/manifest.tsv` has no
  row for it), so it is not a board dataset. Nothing was staged for it.
  `forest_speed_arm.py --lane gbdt-categorical --dataset criteo` runs it
  where it has been fetched.
- Smoke (Apple M4, wheel 0.8.22, `--rows 20000 --rounds 1`, 2026-09-26):
  5 races ran, and all 24 fit cells and 48 inference cells were `ok` with no
  refusal. This is a plumbing check, not a board.

## The classical2 family

`tools/bench_board_more.py` races the wheel's remaining classical estimators.
It uses the classical racer's worker protocol, interleaving and JSON shape,
and its block helpers (stride samples, the Istella sentinel clean, the fit
rows' standardization). Every estimator's binding ships a FAST tier in 0.8.22
(`mojolearn._backend._CLASSICAL_FAST`), so on Apple every lane races `ours`
(IDENTICAL) and `ours-fast` (FAST) side by side. The tier is read back from
the binary. `_mojolearn_solver` (Lasso, ElasticNet) and `_mojolearn_tsa`
(ExponentialSmoothing) carry no numeric-mode constant in 0.8.22, so for
those the tier is read from the directory the loaded binary sits in, and the
cell says so.

| lane | ours | Apple and AMD opponents | NVIDIA opponents | quality |
|---|---|---|---|---|
| `umap` | `UMAP` | umap-learn seeded (one thread, its rule) and unseeded (every core) | cuML UMAP (exact kNN) | trustworthiness k=15 |
| `spectral-embedding` | `SpectralEmbedding` | scikit-learn | cuML, scikit-learn | trustworthiness k=15 |
| `gmm` | `GaussianMixture` | scikit-learn | scikit-learn (cuML has none) | held-out mean log-likelihood, BIC |
| `logreg`, `linearsvc` | `LogisticRegression`, `LinearSVC` | scikit-learn | cuML | held-out accuracy (and log loss) |
| `ridge`, `lasso`, `elasticnet`, `linearsvr` | the same names | scikit-learn | cuML | held-out R2, RMSE |
| `tsvd` | `TruncatedSVD` | scikit-learn (arpack) | cuML | explained-variance ratio sum, reconstruction error |
| `knn-clf`, `knn-reg` | `KNeighborsClassifier`, `KNeighborsRegressor` | scikit-learn | cuML | held-out accuracy, R2 |
| `spectral`, `agglomerative` | `SpectralClustering`, `AgglomerativeClustering` | scikit-learn | cuML (and scikit-learn for spectral) | silhouette, ARI vs ours, cluster count |
| `gpr`, `gpc` | `GaussianProcessRegressor`, `GaussianProcessClassifier` | scikit-learn | scikit-learn (cuML has none) | held-out RMSE, R2, log predictive density; accuracy, log loss |
| `svr`, `kernel-ridge` | `SVR`, `KernelRidge` | scikit-learn | cuML | held-out R2, RMSE |
| `nystroem`, `rbf-sampler` | `Nystroem`, `RBFSampler` | scikit-learn | scikit-learn (cuML has none) | kernel approximation error |
| `arima` | `ARIMA` (one batched fit) | statsmodels (one fit per series, joblib over every core) | cuML, statsmodels | mean llf, mean AIC, forecast and in-sample RMSE |
| `ets` | `ExponentialSmoothing` | statsmodels | cuML, statsmodels | forecast and in-sample RMSE |
| `ivf` | `IVFIndex` | faiss-cpu IVF-Flat | cuVS `ivf_flat` | recall@10 vs float64 brute force |

The rows, every matched parameter, what is inside the clock, and each
mismatch that could not be avoided (with its one-line reason) are in
`LANE_CONFIG` in the driver. The board copies them into every cell's
`settings.lane_config` and prints them under each race. The main ones:

- Sizes are the classical racer's: 1,000,000 fit and 100,000 held-out stride
  rows for the linear lanes and TruncatedSVD. The O(n^2) and O(n^3) lanes take
  a documented stride subset, the same rows for every arm: UMAP and
  SpectralEmbedding 20,000 rows, kNN 200,000 fit rows and 4,000 queries,
  spectral and agglomerative 10,000, GaussianMixture 100,000 and 20,000, GP
  3,000 and 3,000, SVR and KernelRidge 10,000 and 10,000. The IVF lane reads
  the knn lane's block (400,000 index rows, 4,000 queries).
- The time series are synthetic, as in the repo's own ARIMA quality work: 64
  ARMA(1,1) series of 2,100 points and 64 hourly series with a period-24
  season of 1,488 points, all from `default_rng(7)`. The last 100 and 48
  points of each are held out for the forecast error.
- The GP regressor uses `alpha=2**-20`, the one ridge IDENTICAL accepts besides
  0, and carries its noise in a `WhiteKernel(1e-2)` on every arm. Without it
  the float32 factor of the kernel matrix does not exist on taxi's
  near-duplicate rows, and ours refuses to predict from that fit.
- umap-learn with `random_state=7` runs one thread (its own rule), so the
  board also races it unseeded on every core, and says so.
- cuML's Holt-Winters has only its heuristic initialization. Ours runs
  `initialization_method='estimated'`, its default and statsmodels'.

Opponent pins (`MORE_PINS` in `tools/bench_board.py`): umap-learn 0.5.12,
pynndescent 0.6.0, numba 0.67.0, statsmodels 0.15.0 and faiss-cpu 1.15.1 on
Apple and AMD, and statsmodels 0.15.0 on NVIDIA. cuML and cuVS come from the
rapids set in `tools/opponent_wheels.sh`. They are installed only when
classical2 is planned. The dry run and the board list each opponent that is
not planned on a vendor, with the reason (for example faiss-gpu, and cuML's
missing GaussianMixture, GP, Nystroem and RBFSampler).

## The algos family (the algorithm expansion)

`tools/bench_board_algos.py` races every algorithm of the algorithm
expansion (`docs/lanes/ALGORITHM_EXPANSION_PLAN.md`: the nine lane tables,
their Additions and the long tail) against its opponents. It speaks the
classical2 worker protocol (one persistent worker per arm, one warm-up, the
timed rounds interleaved with the arm order rotated, outputs saved after the
clock for a float64 NumPy quality pass in the conductor) with two additions:

- Training AND inference are timed. Each round reports the lane's fit
  (`fit`, `fit_predict`, an index build, a forward + backward, the optimizer
  steps) and, where the lane has one, the inference call (predict /
  transform / search / forward / forecast) on the held-out rows. The
  inference timings become the race's inference cells. Time series lanes
  time fit + forecast together on every arm (statsforecast and the
  per-series libraries forecast in the same call), and ALS is judged from
  its factors, so those lanes have no separate inference cell.
- **SKIPPED: not built yet.** Our side calls the public class by name,
  `mojolearn.<Name>`, the first of the lane's candidate names that the
  INSTALLED wheel exports. Until a lane merges its class, our arms answer
  "skipped" and the cell reads `SKIPPED: not built yet`, never an error,
  while the opponents race. So a board run on a wheel from any point today
  times whatever exists. The dry run marks each race `in source` or
  `not built yet: SKIPPED` from the source tree's `_expansion_<lane>.py`
  `__all__` lists (a hint only; the board asks the wheel).

Our arms are `ours` (IDENTICAL), `ours-fast` (FAST, Apple) and `ours-cpu`
on every lane, neural and CNN included (every expansion binding builds FAST
and IDENTICAL). Opponents, fastest real implementation per box:

| group | opponents |
|---|---|
| scikit-learn-shaped estimators (linear, cluster, neighbors, decomp, prep, trees wrappers) | scikit-learn on every core; cuML on NVIDIA where it has the estimator (MBSGD, Lars, IncrementalPCA, random projections, naive Bayes, TSNE, the cuML preprocessing classes, TargetEncoder, forest-of-one for the decision trees) |
| DART classifier and regressor | LightGBM (`boosting='dart'`) CPU, XGBoost (`booster='dart'`) CPU, and CUDA on NVIDIA |
| SHAP | shap (TreeExplainer, KernelExplainer, PermutationExplainer), XGBoost/LightGBM `pred_contribs` (GPUTreeShap on CUDA), cuML's Kernel and Permutation explainers; our TreeExplainer explains our RandomForestRegressor of the same size (it takes RF, ExtraTrees, DecisionTree and DART models) |
| IVF-PQ, IVF-SQ, IVF-RaBitQ, refine, sample filter, CAGRA | faiss-cpu (HNSW for CAGRA), cuVS on NVIDIA |
| PageRank, connected components, Louvain | networkx, cuGraph on NVIDIA, on the 10-NN graph of 20,000 rows (ours takes a dense adjacency, its class's contract, built before the clock) |
| LSTM, GRU, RNN classifiers and regressors | torch `nn.LSTM`/`nn.GRU`/`nn.RNN` + a linear head trained with Adam for the same epochs and batch size, at every fast setting of the box, on 24-step windows of taxi-hourly and synthetic series |
| LayerNorm, MoE block, Conv1d/2d, pooling, BatchNorm, Dropout2d, ResNet block, GCN, GraphSAGE | torch at every fast setting of the box (eager/compile x fp32/TF32/bf16, TF32 on NVIDIA only; PyG for GCN and SAGE), the same weights loaded into every arm (ours through `load_state_dict` or `set_weights`) |
| RMSprop, Adagrad, Adamax, NAdam, Adafactor | `torch.optim` eager and compiled step, fp32 |
| AutoARIMA, Theta, Croston, damped ETS, STL, VAR, GARCH, Prophet | statsforecast, statsmodels, arch, prophet (one fit per series, joblib over every core); cuML AutoARIMA on NVIDIA |
| SVGP | GPyTorch variational GP on the GPU and CPU |
| ALS | implicit (CPU; its GPU arm refuses by name when the wheel has no CUDA) |
| LU solve, lstsq, randomized SVD | NumPy/LAPACK, torch.linalg on the GPU, CuPy on NVIDIA, scikit-learn `randomized_svd` |

Data: taxi and Istella-S through the classical2 blocks, and the family's own
blocks built in its untimed prep from R2 keys only: `text` (byte-bigram
counts of 2 KB documents of `corpus/enwik8/input.txt` and
`corpus/pile_github/input.txt`, label = which corpus), `taxi-hourly` (hourly
pickups of the 64 busiest zones over January and February 2024, from the taxi
npz), `taxi-zones` (trip counts, (day, hour, pickup zone) x dropoff zone) and
the kNN graph of 100,000 cls rows. The second kind beside a real series set
is a seeded synthetic one. Dense linear algebra, a layer's input tensor and
an optimizer's gradients are seeded tensors. **Not in R2:** an image set
(CIFAR/ImageNet) and an implicit-feedback set (MovieLens/Last.fm); the CNN
layers run on seeded tensors and ALS on the taxi and text counts until one
is staged. The corpora are looked up under `$MOJOLEARN_CORPUS_ROOT`,
`~/r2-stage` and `<repo>/training`; a missing one is refused with the staging
command.

Every lane's settings, rows, quality metric and each unavoidable mismatch
are in `LANES` in the driver (`python3 tools/bench_board_algos.py table`
prints them) and in every cell's `settings.lane_config`. Opponent pins
(`PINS`): statsmodels 0.15.0, statsforecast 2.1.1, arch 8.0.0, prophet
1.4.0, networkx 3.6.1, shap 0.51.0, implicit 0.7.3, torch-geometric
2.8.0.post1, gpytorch 1.15.2 and faiss-cpu 1.15.1 on every vendor, and
cugraph-cu12 26.8.0 from the rapids index on NVIDIA. Not raced, with the
reason: LinearSVC/LinearSVR and SpectralEmbedding (already classical2
lanes), the LR schedulers (a scalar per step), Lion and LAMB torch arms
(torch.optim has neither; ours races alone).

The class contract the board calls (scikit-learn names for estimators;
`fit(Y)` + `forecast(h)` for forecasters; `fit(indptr, indices)` for graph
algorithms; `fit(index).search(queries)` for ANN; `load_state_dict`,
`__call__`/`forward` and `backward(dy)` for layers; `step(grads)` for
optimizers) is in the driver's docstring. A lane whose class answers under a
different name or shape tells lane `bench`, which adds it; the algorithm
lanes do not edit the board.

## The neural family

`tools/bench_board_neural.py` times the wheel's public neural Python API
against torch on the same box. Our arm is IDENTICAL (the only tier the neural
surface builds). Following the rule that our IDENTICAL is compared with the
opponent's fastest supported setting (`bench/OPPONENT_REFERENCE.md`), every
lane races one torch arm per fast setting torch supports there. These are the
columns of `tools/torch_lm_step_opponent.py`, with the setting in the arm name:

| arm | torch setting | planned on |
|---|---|---|
| `torch-eager-fp32` | eager, float32, TF32 off | every vendor |
| `torch-compile-fp32` | `torch.compile` (inductor, default mode), TF32 off | every vendor |
| `torch-eager-tf32` | eager, float32 matmuls in TF32 | NVIDIA only |
| `torch-compile-tf32` | compile with TF32 on | NVIDIA only |
| `torch-eager-bf16` | bf16 autocast mixed precision (parameters, gradients and AdamW state stay float32) | every vendor, probed on the device |
| `torch-compile-bf16` | compile inside the same autocast | every vendor, probed on the device |

TF32 is an NVIDIA CUDA tensor-core matmul mode. On ROCm and MPS torch accepts
the flag and changes nothing, so those arms are not planned there, and the
board says so in "Not covered". An arm that torch cannot run on the box is
refused by name in its cell and never falls back to another setting or
device. Examples are bf16 on a GPU without it, or `torch.compile` failing on
MPS. The TF32 and bf16 arms run at another precision than ours, and their
quality columns show how far each one lands from our output.

On Apple (torch 2.13, M4, small smoke), `torch.compile` and bf16 autocast both
work on MPS for the LM, GEMM, transformer, Mamba-2 and MLP lanes. Inductor's
Metal code generator fails on the Mamba-3 reference, and so on Samba, which
contains it. Those four compile cells are REFUSED by name. On the CPU,
compile and bf16 work for every lane.

| lane | ours (public API) | runs on | torch twin |
|---|---|---|---|
| `lm-train-step` | `LanguageModelTrainer(resident=True, step_result='lean').train_step(ids)` | GPU | `tools/torch_lm_step_opponent.py` `build_model` (SDPA), `torch.optim.AdamW` |
| `lm-forward` | `LanguageModelTrainer(resident=True).logits(ids)` | GPU | the same twin, logits |
| `gemm` | `mojolearn.linalg.matmul(a, b)` | GPU | `a @ b` |
| `transformer-forward` | `TransformerBlock(weights, n_heads, n_kv_heads, head_dim).forward(x)` | GPU | `tools/speed_torch_seq.py` `LlamaEager.block(sdpa=True)` |
| `transformer-infer` | `TransformerBlockInference(...).forward(x)` | CPU | the same, on the CPU |
| `mamba1-forward` / `mamba1-infer` | `Mamba1Block` / `Mamba1BlockInference(weights).forward(x)` | GPU / CPU | `mamba/corpus/gen_corpus.py` `block_forward` (mamba_ssm's `selective_scan_ref`, a per-token loop; eager arms only) |
| `mamba2-forward` / `mamba2-infer` | `Mamba2Block` / `Mamba2BlockInference(weights).forward(x)` | GPU / CPU | `gen_corpus.py` `m2_forward` (chunked SSD reference) |
| `mamba3-forward` / `mamba3-infer` | `Mamba3Block` / `Mamba3BlockInference(weights).forward(x)` | GPU / CPU | `gen_corpus.py` `m3_forward` (SISO reference) |
| `samba-train-step` | `SambaStack(config, weights).train_step(inputs, targets)` | GPU | embedding, then per layer `m3_forward` or `LlamaEager.block`, then RMSNorm and the tied head; mean CE, `torch.optim.AdamW` |
| `samba-forward` | `SambaStack(config, weights).forward(inputs)` | GPU | the same stack, logits |
| `samba-infer` | `SambaInference(config, weights).forward(inputs)` | CPU | the same stack on the CPU |
| `mlp-train-step` | `SmallMLPTrainer(w1, b1, w2, b2).train_step(X, y)` | GPU | `F.linear`, ReLU, `F.linear`; mean CE, `torch.optim.AdamW` |
| `mlp-infer` | `MLPInference(w1, b1, w2, b2).predict_logits(X)` | CPU | the same MLP on the CPU |

The `*-infer` lanes are the public CPU inference classes, which run on the
host binding. They race `torch-cpu-<setting>` arms (eager and compile, fp32
and bf16). Every twin already existed in the repo, and none was written for
the board. In the Apple smoke, each fp32 torch twin matched our output to
about 1e-7 relative (Samba about 7e-7), and each train step's losses matched
to about 1e-6. The Mamba opponents are pure-PyTorch reference
implementations, not mamba-ssm's fused CUDA or Triton kernels (the board does
not install mamba-ssm). A Mamba ratio is therefore against a reference, and
the board says so. The Mamba-1 reference needs `einops`, which the board
installs (`einops==0.8.1`) when the neural family is planned.

Every arm starts from the same inputs, built by the conductor from seed 7.
The LM parameters are `normal(0, .02)` with +1 on every norm, and the batches
come from the byte stream (the installed mojolearn package's own `.py`
sources, sorted and concatenated, sha256 recorded). The transformer weights
are `normal(0, .02)` with +1 on the norms. The Mamba weights are uniform over
`gen_corpus.py`'s default range for each tensor. Samba follows
`SambaStack`'s initializer rules, and our worker refuses unless
`SambaConfig.registry()` matches the conductor's list. The MLP gets
uniform(+-1/sqrt(fan_in)) weights, standard normal X and labels 0..2. Every
clock is host in, host out, synchronized. Training state stays on the device
between steps on both sides, so round r is step r + 1. AdamW uses lr 1e-3,
betas 0.9 and 0.999, eps 1e-8 and weight decay 0.01 on both sides (Samba
without clipping).

Quality: train lanes report `loss_first_step`, `loss_last_step`, and on each
torch arm `loss_first_abs_diff_vs_ours` and `loss_last_abs_diff_vs_ours`.
Forward lanes report `max_abs_diff_vs_ours` and `max_rel_diff_vs_ours` (max
abs difference over max abs of ours) on each torch arm. The logit lanes add
`mean_nll`, and `gemm` adds `max_rel_err_vs_fp64`.

`--neural-shape full` (the default) has these shapes:
- LM: the 20,453,376-parameter control shape of
  `tools/torch_lm_step_opponent.py` (batch 1, length 2048, d_model 384, 6
  heads, head_dim 64, intermediate 1024, 8 layers, vocab 8192).
- GEMM: a 4096 cube.
- Transformer block: that shape's block.
- Mamba blocks: batch 1, length 2048, d_model 384.
- Samba: batch 2, length 512, d_model 384, vocab 256, layers
  mamba3+attention+mamba3+attention, 6 heads, intermediate 1024.
- MLP: 256 rows.
- The CPU lanes cap the length at 512.

The full shape has not yet been run on the Mac. The GPT-3-small target shape
is left off the board. `--neural-shape small` is a plumbing smoke, and the
board says SMOKE. On an Apple M4 all 16 small lanes (76 cells, one round)
took about 3 minutes. `--rows` does not apply to neural lanes.

Not covered in the neural family:
- the blocks' backward (the VJPs), decode `step`, ragged `lengths` and
  carried state;
- `SmallMLPTrainer.predict_logits`;
- mamba-ssm's fused kernels;
- `torch.compile` on the Mamba-1 reference (a per-token Python loop).

Not covered in the classical2 family: `RadiusNeighbors`, the preprocessing
scalers, `Cholesky`, and the `parallel_*` and `Distributed*` wrappers.
Taxi-derived time series are not used.

## Remote Mac (Apple Metal)

Do not run this on the orchestrating laptop. The work is heavy, and that
machine is shared. The steps below are for a separate benchmark Mac,
`bench@bench-mac`, whose home directory is `/Users/bench`.

On the Mac that holds `~/.mojolearn_r2`, stage the data and ship the tree:

```sh
MOJOLEARN_STAGE_BOX_HOME=/Users/bench sh tools/dataset_store.sh stage "bench@bench-mac" \
    gbm-bench/taxi/taxi_speed.npz gbm-bench/istella/istella_speed.npz \
    gbm-bench/istella/istella_rank.npz
ssh bench@bench-mac 'rm -rf ~/mojolearn-board && mkdir -p ~/mojolearn-board'
git archive --format=tar HEAD | ssh bench@bench-mac 'tar -x -C ~/mojolearn-board'
git rev-parse HEAD | ssh bench@bench-mac 'cat > ~/mojolearn-board/SHIPPED_COMMIT.txt'
```

On the benchmark Mac, inside `tmux`, run the plan and then the board.
`--base-python` must be an interpreter the wheel supports.

```sh
cd ~/mojolearn-board
python3 tools/bench_board.py --dry-run --mojolearn-version 0.8.22
python3 tools/bench_board.py --mojolearn-version 0.8.22 \
    --base-python /opt/homebrew/bin/python3.12 \
    --out ~/board-runs/apple-0.8.22 --cache ~/board-cache \
    2>&1 | tee -a ~/board-runs/apple-0.8.22.log
```

If the run stops, rerun the same command. Back on the orchestrator, fetch the
result:

```sh
rsync -a bench@bench-mac:board-runs/apple-0.8.22/ "$HOME/mojolearn-evidence/bench-board/apple-0.8.22/"
```

## NVIDIA leg (DigitalOcean H100)

The body is `tools/bench_board_leg.sh`. The runner stages taxi and Istella-S
from R2 into `/root/datasets/gbm-bench` before the body starts. A full board
takes many hours, so it uses a segment lease rather than the one-hour cap.
The RunPod runner `tools/gemm_remote_leg.sh` is capped at 60 minutes and
passes no environment to the body, so it can only run a small `--lanes`
subset there.

```sh
MOJOLEARN_DO_TOKEN_FILE=$HOME/.mojolearn_do_token \
MOJOLEARN_GEMM_LEG_EXTRA=tools/bench_board_leg.sh \
MOJOLEARN_GEMM_LEG_OUT=$HOME/mojolearn-evidence/bench-board/$(date -u +%Y-%m-%d_%H%M%S)-nvidia-h100 \
MOJOLEARN_DO_EXTRA_ENV='MOJOLEARN_BOARD_VERSION=0.8.22' \
bash tools/do_extra_leg.sh nv --segment-lease 720 --dollar-cap 60 --skip-gates
```

The result arrives at `<leg out>/remote/bench-board/`. NVIDIA uses the image's
CUDA torch through `--system-site-packages`. cuML is installed from the pinned
`rapids-*` set in `tools/opponent_wheels.sh`.

## AMD leg (Hot Aisle MI300X)

The `13core` spec is the one whose CPU opponent rows are comparable. The body
builds a Python 3.12 venv, because the pinned torch ROCm wheels are cp312.

```sh
MOJOLEARN_HOTAISLE_SPEC=13core MOJOLEARN_HOTAISLE_LANE=bench-board \
MOJOLEARN_GEMM_LEG_EXTRA=tools/bench_board_leg.sh \
MOJOLEARN_GEMM_LEG_OUT=$HOME/mojolearn-evidence/bench-board/$(date -u +%Y-%m-%d_%H%M%S)-amd-mi300x \
MOJOLEARN_HOTAISLE_EXTRA_ENV='MOJOLEARN_BOARD_VERSION=0.8.22' \
bash tools/hotaisle_leg.sh amd --rent --segment-lease 900 --dollar-cap 60 --skip-gates
```

The DigitalOcean MI325X works the same way:
`bash tools/do_extra_leg.sh amd --segment-lease 900 --dollar-cap 60 --skip-gates`,
with the environment passed in `MOJOLEARN_DO_EXTRA_ENV`.

## Knobs the leg body reads

Values contain no spaces, and lists are separated by commas.

| variable | effect |
|---|---|
| `MOJOLEARN_BOARD_VERSION` | the mojolearn version to install (required) |
| `MOJOLEARN_BOARD_ROWS` | row cap for a smoke run; the board is then marked SMOKE |
| `MOJOLEARN_BOARD_LANES` / `_FAMILIES` / `_DATASETS` / `_ROUNDS` | narrow the plan |
| `MOJOLEARN_BOARD_NEURAL_SHAPE` | `full` (default) or `small` for a neural smoke |
| `MOJOLEARN_BOARD_NO_INFER` | `1` times training only (no inference cells) |
| `MOJOLEARN_BOARD_NO_CPU_ARM` | `1` leaves out our CPU tier (`ours-cpu`) |
| `MOJOLEARN_BOARD_OUT`, `MOJOLEARN_BOARD_CACHE` | result directory (fetched) and cache (not fetched; the classical and classical2 blocks live here) |

A smoke leg, for example:
`MOJOLEARN_DO_EXTRA_ENV='MOJOLEARN_BOARD_VERSION=0.8.22 MOJOLEARN_BOARD_ROWS=20000 MOJOLEARN_BOARD_ROUNDS=1 MOJOLEARN_BOARD_LANES=rf,kmeans'`.

## Reading the board

`BOARD.md` gives times, ratios and quality, and never states a direction. A
ratio column is our median divided by the opponent's median (`ours IDENTICAL /
arm`, `ours FAST / arm`, `ours CPU / arm`). Our FAST, IDENTICAL and CPU arms
are never divided by each other, because the FAST ratio is the cost of
identity and not a result (ENGINEERING_RULES 0b-iii). The "Quality at a
glance" table puts our FAST value, our IDENTICAL value, our CPU value and each
opponent's
value side by side for every lane and dataset; "Inference at a glance" does
the same for the inference medians per batch, with whether our FAST and
IDENTICAL predictions agree bit for bit. For comparability, trees carry
`FSPEED-FIT-VERDICT` and classical and neural lanes carry the clock span
(`SPAN-ASYMMETRIC` names an opponent whose clock excludes an upload or a fit
that ours includes). A missing arm shows as `UNKNOWN` or `REFUSED(reason)`,
never as a blank.
