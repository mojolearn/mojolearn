# Algorithm expansion: lane briefs

These are the briefs for the nine lanes in
[ALGORITHM_EXPANSION_PLAN.md](ALGORITHM_EXPANSION_PLAN.md). Every lane gets
**THE COMMON BRIEF** plus its own section. The reference sources are pinned
under `~/CascadeProjects/upstream/`: scikit-learn, cuml-v26.08.00,
cuvs-v26.08.00, raft-v26.08.00, lightgbm, xgboost and catboost.

## THE COMMON BRIEF

**Your setup**

- Worktree: `~/mojolearn-wt/algos-<lane>`, on branch `lane/algos-<lane>`,
  created from `origin/main`. Never touch `~/CascadeProjects/mojolearn`.
- Pod: `tools/dev_pod.sh up <lane>` (it retries while RunPod is out of
  stock, `MOJOLEARN_DEVPOD_RETRY_MINUTES`, default 60). Sync with
  `tools/dev_pod.sh sync <lane> <worktree>`. Run with
  `tools/dev_pod.sh run <lane> '<cmd>'` (it runs in `/root/mojolearn`).
  The sync is a tar with no `.git`: before your first check, give the pod
  a git tree at your base commit (the lane check applies the sabotage with
  `git apply`):
  `tools/dev_pod.sh run <lane> 'git init -q; git remote add origin https://github.com/mojolearn/mojolearn.git; git fetch -q --depth 50 origin <base sha>; git reset -q <base sha>'`
  where `<base sha>` is `git merge-base HEAD origin/main` in your worktree.
  After every `git merge origin/main` (step 8) redo this at the new base,
  before the next `sync`: a sabotage patch that applies on the laptop but
  not on the pod is almost always a stale pod tree, not a bad patch
  (plan, R6).
- Heartbeat: run `tools/dev_pod.sh extend <lane> 120` every hour while you
  are working. If you stop calling it, the pod ends on its own.
- Evidence goes to `~/mojolearn-evidence/algos-<lane>/`, never into the
  repo.
- The MacBook is off limits: no build, no test, no Metal, no CPU work. The
  Apple column is the two cloud Macs behind `tools/apple_steward.py`.

**Order of work for each algorithm (one algorithm per commit)**

1. **Reference.** Read the reference source. Name the file and line in your
   Mojo file's header. Every option you do not carry gets a row in your
   module's `NOT_IMPLEMENTED.tsv`, or is refused by name at the API.
2. **Implement in Mojo.** One source for CPU and GPU, with no vendor
   branches.
   - IDENTICAL: fixed reduction order and fixed tie-breaks. No atomics in
     any reduction whose order could change the result. No float64 on the
     device. Every operand through `ftz`, every division `identical_div`,
     every product `identical_mul` (`checks/numerics.mojo`).
   - Never let a NaN reach an output: 0/0 is a NaN whose payload is the
     vendor's (NVIDIA `0x7fffffff`, x86 `0xffc00000`), so a zero-variance or
     all-zero column must be handled explicitly. The prep proof dummy read
     DISAGREE on the `dupes` fixture for exactly this reason.
   - FAST: allowed to use a different schedule, and allowed to be not
     identical, but never lower in quality (step 8).
   - **EVERY NUMERIC SEAM GETS THE FULL DISCIPLINE, not just the
     algorithm.** A seam is any place where two legal spellings can give
     different bits: a reduction or fold order, an FMA contraction, a
     denormal/ftz point, a computed NaN (IDENTITY_PATHS.md Clause B), a sign
     or tie convention (Clause A: value-first clamps), an RNG mapping, a
     sort or tie-break, a clamp or compare at a boundary. Each gets one of
     IDENTITY_PATHS.md "The rule"'s three moves (PIN, REPLACE, REFUSE) and
     ALL FIVE of:
     a. **a host oracle**, `<module>/checks/<name>_oracle.mojo`, restating
        the seam as plain host code, with a check driver
        `<module>/checks/<name>_check.mojo` asserting device == oracle BIT
        FOR BIT (pattern: `holtwinters/checks/hw_oracle.mojo`, `hw_check.mojo`,
        pixi task `check-holtwinters`);
     b. **a fixture that is shown to separate** the pinned spelling from the
        unpinned one BEFORE it is trusted: the check first computes both
        spellings on the fixture and refuses as VACUOUS when they agree
        (CONTRIBUTING.md "Numerical changes"; pattern: the halving-vs-sequential
        guard in `core/column_stats_identity_check.mojo`). Plant `-0.0`,
        subnormals, exact ties and all-zero columns where the seam reads them;
     c. **a sabotage arm per seam, BUILT AND RUN**, that bites (the check
        fails naming the seam). A sabotage that does not move a bit on some
        backend (an "Apple-null" arm) is recorded as a REACH FAILURE of that
        fixture on that backend, never as a pass;
     d. **a DEVIATION number** from your lane's range (below), in the code
        comment at the seam and in your module's contract/README;
     e. **a card stage**: the seam's intermediate recorded through
        `core.identity_trace.IdentityTrace` (`record_device`/`record`) so
        `python3 tools/identity_trace_diff.py <a>.card <b>.card` localizes a
        cross-vendor difference to the stage (patterns: `holtwinters/checks/`,
        `tools/e2_mojo_cards.sh`);
     plus **one row in your section of IDENTITY_PATHS.md** from your row
     range, naming the seam, the move, the DEVIATION and the check. Commands
     (on your pod; on the Macs the steward runs them too, see step 7):
     ```
     tools/with_identical_mode.sh pixi run mojo run -I . <module>/checks/<name>_check.mojo   # oracle + fixture + card
     pixi run mojo run -I . <module>/checks/<name>_check.mojo                                 # FAST arm: recorded, no claim
     # each sabotage arm: a SOURCE patch, git apply, rerun the check, it must FAIL, git apply -R
     python3 tools/identity_trace_diff.py <box A>/<name>.card <box B>/<name>.card             # cross-box
     ```
     List every check driver of your lane, one repo-relative path per line,
     in `tools/identity_lanes/<lane>.checks` (a file you own; pixi.toml is
     shared, so no per-lane pixi task). `tools/algos_lane_check.sh` runs each
     listed driver under `tools/with_identical_mode.sh` before its GPU/CPU
     diff and fails if any exits nonzero; the diff stays the Python-level
     end-to-end check on top. **The `.checks` file is REQUIRED from the
     first algorithm you register**: the tool today only prints a note when
     it is missing (plan, R2), so the orchestrator refuses to merge a lane
     whose fragment registers an identity lane and has no `.checks` listing,
     or whose listing does not name a driver for every seam in its ledger
     rows. The per-seam sabotage arms are yours to run (one patch each,
     above) and to REPORT at each commit, by patch name and result; no tool
     runs them yet (plan, R3). The steward runs the lane check with the one
     end-to-end `--sabotage` patch you submit.
3. **Python class**, sklearn-shaped (`fit` / `predict` / `transform` /
   `predict_proba` where the reference has them), in your door
   `python/mojolearn/_expansion_<lane>.py` (or imported there), listed in its
   `__all__`. It resolves its binding with
   `_backend.binding("_mojolearn_x_<lane>", mode)`; on a CPU-only install the
   same call returns your host binding, so the host binding exports the GPU
   binding's function names with the same address contract.
4. **Correctness sanity** on the pod against the reference library at a
   tolerance, on tiny data. This is a sanity check, not an identity claim.
5. **Verifier lane + CPU route.** A SMALL fixture: non-uniform data, ties
   exercised, and every code path reached. Register it in your identity
   fragment `tools/identity_lanes/<lane>.py` (`@lane("...")` plus its
   `_batch_decl`), declare its CPU route in `python/mojolearn/_surface_<lane>.py`
   (your family `x_<lane>`: `training_lanes`, `exports`, `host_modules`;
   `TRAINING_LANE_NAMES`; `PUBLIC_PENDING_LANES = {lane: "no reference"}`) and
   `GPU_BINDINGS = ("_mojolearn_x_<lane>",)`. Then, on the pod:
   `tools/dev_pod.sh run <lane> 'sh tools/algos_lane_check.sh <lanes>'`
   must print `RESULT: PASS`. The tool derives every binding the lanes run
   (GPU and CPU host) from the lane map, rebuilds any that is missing or
   stale, fits on the GPU and on the CPU host binding, and diffs train,
   infer, model and batch. NOTHING COMPARED, a missing host binding, or a
   refused stage is a failure, never a pass.
6. **Sabotage.** Write a patch that edits the SOURCE (never a `-D` define;
   define-only builds reuse cached kernels) to flip a reduction order or a
   tie-break. Then
   `tools/dev_pod.sh run <lane> 'sh tools/algos_lane_check.sh <lanes> --sabotage /root/<patch>'`
   must print `RESULT: PASS`: AGREE, then DISAGREE under the patch, then
   AGREE after `git apply -R` (the tool applies, rebuilds and reverses; never
   `git checkout`). Copy the patch to the pod first
   (`cat <patch> | tools/dev_pod.sh run <lane> 'cat > /root/<patch>'`) and
   keep it under your evidence dir.
7. **Commit, push the branch, push the commit to the gating Mac, and submit
   to the Apple stewards:**
   ```
   git push -u origin lane/algos-<lane>
   tools/cloudmac.sh push m2pro <sha>          # REQUIRED: the Mac's origin is its own bare repo,
                                               # not GitHub; without this the steward fails at checkout (plan, R1)
   tools/apple_steward.py submit --lane <lane> --commit <sha> --verify-lanes <lanes> --sabotage <patch>
   ```
   The deferred Mac (M3 Ultra) gets the same push from the orchestrator at
   `flush-deferred` time; do not ssh to it.
   It queues the request for BOTH cloud Macs (M2 Pro, M3 Ultra); each runs the
   same `tools/algos_lane_check.sh --sabotage` against Metal and the Arm CPU
   host bindings. The M3 Ultra is DEFERRED while it runs a GPT-3 training
   segment: its copy is spooled on the laptop and shipped when that run ends
   (never ssh to it). Keep going on the next algorithm; don't wait for the
   verdict. `tools/apple_steward.py status` shows PASS once the gating Mac
   (M2 Pro) passed; any FAIL, including a later M3 Ultra one, comes back to
   you to fix.
8. **Merge it yourself** once the algorithm is green on your pod (NVIDIA +
   CPU) AND the M2 Pro steward reads PASS:
   ```
   git fetch origin && git merge origin/main      # never rebase pushed history
   # rerun step 5 (and step 6's patch) on your pod against the merged tree
   git push origin HEAD:main                      # fast-forward only
   ```
   If the push is rejected, main moved: repeat from `git fetch`. Never
   force-push, never push anything the lane check did not just pass.
9. **Speed, after identity passes.** Speed work covers FAST and IDENTICAL
   (neural included, per FINAL DECISIONS), at 1M+ rows (or the family's
   realistic large shape), timed before and after on your own pod.
   - Every IDENTICAL speed change must re-pass steps 5 to 8. It is one
     source, so an NVIDIA speedup can move Apple bits.
   - Every FAST change needs a paired quality check against the reference:
     at least 5 seeds and at least 2 datasets. Revert on any loss.
   - No claims against opponents in this sprint; those come later through
     `tools/bench_board.py`.

**Rules**

- Never `git stash`. Never `git add -A`. Check `git rev-parse --abbrev-ref HEAD`
  before every commit.
- Touch only the files your lane owns (next section).
- Rebuild every binding that imports a shared module you changed (the lane
  check rebuilds what the lanes it checks run; a shared module reaches
  other lanes too).
- **Stop and report; don't widen.** Report instead of patching when a fix
  would need:
  - a change to shared core code (GEMM, reductions, RNG, serialization,
    `core/`), or any file another lane owns, or
  - float64 on the device, or
  - something you cannot make identical.
- Report at each commit: the algorithm, the sha, the pod verdict, the
  sabotage result, the two Mac verdicts, and anything refused.

## Shared registries: what you may touch

Every registry the nine lanes share reads per-lane files, so no lane edits a
shared list. For lane `<lane>` (one of `linear cluster neighbors decomp prep
sequence trees cnn ann`) the files it OWNS are exactly:

| file | what it holds | read by |
|---|---|---|
| `python/mojolearn/_expansion_<lane>.py` | the public classes (`__all__`); optional `CLASSICAL_HOST_BASENAMES` and `classical_host_formats()` for saved-model CPU routes | `mojolearn/__init__.py` (binds `__all__`, refuses a clash by name), `_classical_host.py` |
| `python/mojolearn/_surface_<lane>.py` | literal data only: `GPU_BINDINGS`, `FAMILIES` (one family, `x_<lane>`), `TRAINING_LANE_NAMES`, `PUBLIC_PENDING_LANES`, each bound ONCE (a second binding is not yet refused and silently wins; plan, R8) | `host_surface.py` (and through it `_backend`, every packaging list, the CPU gate) |
| `tools/identity_lanes/<lane>.py` | the lane's identity lanes and their `_batch_decl`/part declarations, written against identity_break's API | `tools/identity_break.py` (executed in its namespace), the wheel (as `mojolearn/_identity_lane_<lane>.py`), `tools/lane_accounting.py`, `tools/lane_select.py` |
| `tools/classical_host_lanes/<lane>.py` | classical gate probes, only for lanes the family declares as `inference_lanes` | `tools/classical_host_gate.py` |
| `bindings/_mojolearn_x_<lane>.mojo`, `bindings/build_x_<lane>.sh` | the lane's ONE GPU binding and its build script (copy `bindings/build_preprocessing.sh`; fast + identical, refuse deterministic, EVERY lane, `sequence` and `cnn` included) | `_backend`, packaging (through `GPU_BINDINGS`) |
| `bindings/_mojolearn_x_<lane>_host.mojo`, `bindings/build_x_<lane>_host.sh` | the ONE host binding (the shim is two lines: `exec sh "$(dirname -- "$0")/build_host_family.sh" x_<lane> "$@"`); exports `x_<lane>_host_{numeric_mode,vendor,column,sabotage}` plus the GPU binding's names | `host_surface` family, `_backend._HOST_MODULES` |
| the lane's own new Mojo module directories (`naive_bayes/` for prep, and so on) and their `NOT_IMPLEMENTED.tsv` | kernels, host oracles | the two bindings |
| the lane's own new tests under `python/mojolearn/tests/test_x_<lane>_*.py` | | |

The loaders enforce ownership, so a violation fails loudly instead of
colliding: a fragment may name only its own family and bindings
(`host_surface._read_expansion_fragment`), an identity fragment may add only
its own lanes and may rebind no existing harness name (prefix helpers with
`_<lane>_`; `identity_break._load_lane_fragments`), and a public name that is
already public is refused at import. `tools/lane_select.py` attributes a
change to a fragment to that lane's lanes only.

**Not yours, even though your algorithm touches the concept:**
`python/mojolearn/__init__.py`, `host_surface.py`, `_backend.py`,
`_classical_host.py`, `tools/identity_break.py`, `tools/classical_host_gate.py`,
every `packaging/` file, `python/.gitignore` (it already ignores
`mojolearn/_mojolearn_x_*.so`), `verify_reference/table.json`,
`_verification_catalog.py`, and any existing binding or module directory. If
you need one of them changed, stop and report.

**FAST on every lane, neural included.** Every expansion lane's GPU
binding builds FAST and IDENTICAL (`host_surface.EXPANSION_IDENTICAL_ONLY`
is empty; Andrew, 2026-09-27). There is no identical-only expansion lane.
The four EXISTING neural bindings stay identical only until their FAST tier
is built, which is the orchestrator's work and not yours; never edit
`packaging/linux/build_sets.sh`'s IDENTICAL_ONLY lists or `_backend.py`'s
tier table.

## Numbers each lane owns

Highest DEVIATION in the tree at prep time: **3133**. Highest
`IDENTITY_PATHS.md` ledger row: **96** (its "Row-number registry" table now
carries these ranges). Use only your own; never renumber anything that exists.

| lane | DEVIATION range | IDENTITY_PATHS rows |
|---|---|---|
| linear | 5000-5099 | 100-109 |
| cluster | 5100-5199 | 110-119 |
| neighbors | 5200-5299 | 120-129 |
| decomp | 5300-5399 | 130-139 |
| prep | 5400-5499 | 140-149 |
| sequence | 5500-5599 | 150-159 |
| trees | 5600-5699 | 160-169 |
| cnn | 5700-5799 | 170-179 |
| ann | 5800-5899 | 180-189 |

A new IDENTITY_PATHS row is the one edit to that file a lane makes: replace
the `(no rows yet)` line of YOUR section under "Algorithm expansion rows" at
the end of `IDENTITY_PATHS.md`, in your range. The sections are separated so
nine lanes' edits never touch the same lines.

---

## Lane 1: linear

Machinery to reuse: `glm/` (QN/OLS), `solver/` (CD), `cholesky/`
(`choleskyRank1Update`), the training optimizers, and
`parked/glm-qn-losses-wip-2026-08-23.patch`.

| algorithm | reference |
|---|---|
| SGDClassifier / SGDRegressor | sklearn `linear_model/_stochastic_gradient.py`; cuML `cpp/src/solver/sgd.cuh` (MBSGD) |
| PoissonRegressor / GammaRegressor / TweedieRegressor | sklearn `linear_model/_glm/glm.py` |
| HuberRegressor | sklearn `linear_model/_huber.py` |
| BayesianRidge / ARDRegression | sklearn `linear_model/_bayes.py` |
| LinearSVC / LinearSVR | sklearn `svm/_classes.py` (liblinear); cuML `svm/linear*` |
| Lars / LassoLars | sklearn `linear_model/_least_angle.py`; cuML `cpp/src/solver/lars_impl.cuh` |
| QuantileRegressor | sklearn `linear_model/_quantile.py` (it uses an LP; use an IRLS/ADMM formulation with a fixed iteration order and name it) |

**Additions (2026-09-27, after the table):** Perceptron,
PassiveAggressiveClassifier/Regressor, RidgeClassifier, SGDOneClassSVM
(sklearn `linear_model/_perceptron.py`, `_passive_aggressive.py`,
`_ridge.py`, `_stochastic_gradient.py`; all variants of your SGD);
RidgeCV, LassoCV, ElasticNetCV, LogisticRegressionCV (sklearn
`linear_model/_ridge.py`, `_coordinate_descent.py`, `_logistic.py`; the
fold order is `cross_val_score`'s); IsotonicRegression (sklearn
`isotonic.py`; parallel PAVA by prefix scan, name the reference for the
scan form).

## Lane 2: clustering

Machinery to reuse: `cluster/` (KMeans, k-means++), `kde/`, `dbscan/`,
`hdbscan/` core distances, and GEMM.

| algorithm | reference |
|---|---|
| MiniBatchKMeans | sklearn `cluster/_kmeans.py` (MiniBatchKMeans; the batch order comes from the seed) |
| BisectingKMeans | sklearn `cluster/_bisect_k_means.py` |
| MeanShift | sklearn `cluster/_mean_shift.py` |
| OPTICS | sklearn `cluster/_optics.py` (the ordering loop is sequential; tie-break by index) |
| AffinityPropagation | sklearn `cluster/_affinity_propagation.py` |

**Additions (2026-09-27, after the table):** BayesianGaussianMixture
(sklearn `mixture/_bayesian_mixture.py`; reuse `mixture/`, whose
`NOT_IMPLEMENTED.tsv` already carries the row).

## Lane 3: neighbors + kernel

Machinery to reuse: `neighbors/` (brute kNN, radius), the SVM solver
(`svm/`), `kernel_methods/`, and eigh.

| algorithm | reference |
|---|---|
| LocalOutlierFactor | sklearn `neighbors/_lof.py` |
| NearestCentroid | sklearn `neighbors/_nearest_centroid.py` |
| OneClassSVM | sklearn `svm/_classes.py` (OneClassSVM, libsvm nu-formulation); already a row in `kernel_methods/NOT_IMPLEMENTED.tsv` |
| KernelPCA | sklearn `decomposition/_kernel_pca.py` |

**Additions (2026-09-27, after the table):** PolynomialCountSketch,
AdditiveChi2Sampler, SkewedChi2Sampler (sklearn `kernel_approximation.py`;
`kernel_methods/NOT_IMPLEMENTED.tsv` carries the first); LabelPropagation,
LabelSpreading (sklearn `semi_supervised/_label_propagation.py`; kNN graph
plus a fixed-order iteration); KNNImputer (sklearn `impute/_knn.py`).

## Lane 4: decomposition + linalg

Machinery to reuse: `decomposition/` (PCA, TSVD, Jacobi), `gemm/`, QR, eigh,
SVD, `spectral/` (its embedding step), and `cholesky/`.

| algorithm | reference |
|---|---|
| IncrementalPCA | sklearn `decomposition/_incremental_pca.py` |
| GaussianRandomProjection / SparseRandomProjection | sklearn `random_projection.py` (the RNG comes from our counter-based RNG, not numpy's) |
| SpectralEmbedding | sklearn `manifold/_spectral_embedding.py` |
| NMF | sklearn `decomposition/_nmf.py` (multiplicative update first; CD second) |
| FastICA | sklearn `decomposition/_fastica.py` |
| FactorAnalysis | sklearn `decomposition/_factor_analysis.py` |
| LU solve | LAPACK getrf/getrs semantics, with partial pivoting and pivot ties broken by lowest index |
| lstsq / randomized SVD | numpy `linalg.lstsq`; sklearn `utils/extmath.py` `randomized_svd`; RAFT `linalg/rsvd.cuh` |

**Additions (2026-09-27, after the table):** CCA, PLSRegression (sklearn
`cross_decomposition/_pls.py`); SparsePCA, MiniBatchSparsePCA,
DictionaryLearning (sklearn `decomposition/_sparse_pca.py`, `_dict_learning.py`);
LatentDirichletAllocation (sklearn `decomposition/_lda.py`, variational EM
in a fixed order); Isomap, MDS, LocallyLinearEmbedding (sklearn
`manifold/_isomap.py`, `_mds.py`, `_locally_linear.py`; kNN plus eigh);
EllipticEnvelope / MinCovDet (sklearn `covariance/_robust_covariance.py`);
ALS matrix factorization for implicit-feedback recommendation (reference:
the `implicit` library's `als.py`; GEMM plus batched least squares, a clear
GPU win).

## Lane 5: preprocessing + naive Bayes & discriminant analysis (NEW 13th family)

Machinery to reuse: `preprocessing/` (standard and min-max scalers), GBDT
quantile binning and CTR target statistics, covariance + eigh, and
`metrics/` reductions.

| algorithm | reference |
|---|---|
| RobustScaler, MaxAbsScaler | sklearn `preprocessing/_data.py` |
| OneHotEncoder / OrdinalEncoder | sklearn `preprocessing/_encoders.py` |
| TargetEncoder | sklearn `preprocessing/_target_encoder.py` (the cross-fit fold order comes from the seed) |
| SimpleImputer | sklearn `impute/_base.py` |
| KBinsDiscretizer | sklearn `preprocessing/_discretization.py` |
| GaussianNB, MultinomialNB, BernoulliNB | sklearn `naive_bayes.py` |
| LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis | sklearn `discriminant_analysis.py` |

This lane creates the new family's module directory (`naive_bayes/`).

**Additions (2026-09-27, after the table):** QuantileTransformer,
PowerTransformer, Normalizer, PolynomialFeatures, SplineTransformer,
Binarizer (sklearn `preprocessing/_data.py`, `_polynomial.py`);
LabelEncoder, LabelBinarizer, MultiLabelBinarizer (`preprocessing/_label.py`);
IterativeImputer (`impute/_iterative.py`, round-robin order fixed by
column index); VarianceThreshold, SelectKBest with f_classif, chi2,
f_regression and mutual_info, RFE as a wrapper (`feature_selection/`);
ComplementNB, CategoricalNB (`naive_bayes.py`).

## Lane 6: sequence (neural + time series)

Machinery to reuse: `training/` (MLP, Adam/AdamW/SGD, `clip_grad_norm_`),
the Mamba scan, `arima/`, `tsa/` (KPSS, `select_d`) and `holtwinters/`.

| algorithm | reference |
|---|---|
| LSTM, GRU | PyTorch `nn.LSTM` / `nn.GRU` gate equations and layouts (the pod has torch; use it for sanity only) |
| RMSprop, Adagrad | PyTorch `torch.optim.RMSprop` / `Adagrad` update rules |
| AutoARIMA order search | cuML `python/cuml/cuml/tsa/auto_arima.pyx`; `batched_arima.cu` `information_criterion` (rows already in `arima/NOT_IMPLEMENTED.tsv`) |
| STL | statsmodels `STL` (loess inner/outer loops); pip-install it on the pod for sanity |
| VAR | statsmodels `VAR` (OLS per equation, fixed lag order) |

**FAST is on for this lane.** Your GPU binding builds FAST and IDENTICAL
like every other expansion lane (`host_surface.EXPANSION_IDENTICAL_ONLY` is
empty). Step 9 applies in full: FAST and IDENTICAL speed work, each FAST
change with its paired quality check. The existing neural bindings
(`_mojolearn_training`, `_mojolearn_mamba`, `_mojolearn_transformer`) you
reuse stay identical only; call them in IDENTICAL from your FAST arm, or
reimplement the piece inside your own binding, and never edit their build
scripts or `packaging/linux/build_sets.sh`.

**Additions (2026-09-27, after the table):** MLPClassifier / MLPRegressor,
sklearn-shaped over `SmallMLPTrainer` (sklearn `neural_network/_multilayer_perceptron.py`
for the API; the trainer is ours); a vanilla RNN after LSTM (PyTorch
`nn.RNN`); Lion, Adafactor, LAMB, Adamax, NAdam (their papers' update
rules; PyTorch `torch.optim` where it has them); LR schedulers step,
exponential and one-cycle (`torch.optim.lr_scheduler`); LayerNorm beside
RMSNorm (`torch.nn.LayerNorm`); Theta and Croston forecasters and
damped-trend ETS (statsforecast `models.py`); GARCH (the `arch` package,
`univariate/volatility.py`).

## Lane 7: trees

Machinery to reuse: the RF/ET tree builder (`ensemble/`, `extratrees/`) and
the GBDT boosting loop (`gbdt/`). **Coordinate:** the FAST trees work
(`lane/apple-fast-trees`) edits the same builder and bindings. Rebase often,
and never rebuild the shared trees `.so` while another job is using it.

| algorithm | reference |
|---|---|
| DecisionTreeClassifier / DecisionTreeRegressor | sklearn `tree/_classes.py` (CART); in effect our RF with one tree, no bootstrap, all features |
| BaggingClassifier / BaggingRegressor | sklearn `ensemble/_bagging.py` |
| AdaBoostClassifier / AdaBoostRegressor | sklearn `ensemble/_weight_boosting.py` (SAMME; AdaBoost.R2) |
| DART | LightGBM `src/boosting/dart.hpp` (the tree-drop order comes from the seed) |

**Additions (2026-09-27, after the table):** RandomTreesEmbedding (sklearn
`ensemble/_forest.py`); VotingClassifier/Regressor, StackingClassifier/Regressor,
MultiOutputClassifier/Regressor, OneVsRestClassifier, CalibratedClassifierCV
(wrappers: sklearn `ensemble/_voting.py`, `_stacking.py`, `multioutput.py`,
`multiclass.py`, `calibration.py`); SHAP explainers over our forests and
GBDT: TreeExplainer first (the `shap` package's `explainers/_tree.py`
algorithm, exact per-tree), then KernelExplainer and PermutationExplainer
(cuML `cpp/src/explainer/kernel_shap.cu`, `permutation_shap.cu`).

## Lane 8: CNN (HARD)

Machinery to reuse: identical GEMM, `training/`, and the MLP.

| algorithm | reference |
|---|---|
| Conv1d / Conv2d (+ MaxPool / AvgPool), forward and backward | PyTorch `nn.Conv1d/2d` semantics. Implement as im2col + our GEMM so the identical GEMM contract carries over. The backward pass's weight-gradient reduction order is the identity risk; pin it. |

The deliverable can be forward + backward + a small CNN trainer. If
something cannot be made identical, the deliverable is a design note plus a
named refusal. The one true atomic hazard is the backward pass's col2im:
a scatter-add over overlapping receptive fields. Write it as a gather per
input pixel in a fixed order (the Jacobi move `umap/optimizer_identical_device.mojo`
made), never as atomics. Max pooling's tie is `identical_fmax`'s (the sign
of zero); BatchNorm's statistics are pinned folds whose partial count is a
function of the shape, never of the core count (IDENTITY_PATHS row 7).
FAST is on for this lane too: the binding builds FAST and IDENTICAL, and
step 9 applies in full (FAST conv is where cuDNN-style algorithm choice
lives; IDENTICAL stays im2col onto the pinned GEMM).

**Additions (2026-09-27, after Conv lands):** BatchNorm (pinned folds,
partial count from the shape), Dropout2d (Philox), global average and max
pooling, one ResNet basic block (PyTorch `torchvision.models.resnet.BasicBlock`).

## Lane 9: ANN + t-SNE (HARD)

Machinery to reuse: `ivf/` (IVF-Flat), `neighbors/`, KMeans, and GEMM.

| algorithm | reference |
|---|---|
| IVF-PQ | cuVS `cpp/src/neighbors/ivf_pq/` (codebook training is KMeans; the LUT and top-k tie-breaks by index) |
| CAGRA | cuVS `cpp/src/neighbors/cagra*` (the graph build and optimize order is the identity risk) |
| t-SNE | cuML `cpp/src/tsne/`; sklearn `manifold/_t_sne.py` (exact or FFT-interpolation gradients; the Barnes-Hut atomics are NOT allowed under IDENTICAL) |

**Write the design note first.** It says how each reference's
nondeterminism (atomics, build order) gets a fixed order. Then implement in
this order: IVF-PQ, t-SNE, CAGRA. Ending with a design note and a named
refusal for CAGRA is an acceptable outcome. Starting points the tree
already holds: IVF-PQ's codebooks are `cluster/`'s k-means per subspace,
its encoding a per-subspace argmin with the index tie-break of
IDENTITY_PATHS row 22, its lookup table and code sum fixed-order chains
with no atomics, its top-k the composite-key selector (rows 11 and 23);
t-SNE's attractive term is the UMAP CSR fold, its repulsive term either
exact (a per-vertex fold over all points ascending, O(n^2) per iteration)
or Barnes-Hut over a tree built by Morton-code radix sort rather than
atomic insertion; CAGRA's default cuVS build goes through IVF-PQ plus exact
refine, so it depends on IVF-PQ landing first, and its reverse-edge
insertion and search queue are the parts that need a rank-based rewrite
(row 23's 32-lane pin). NN-descent stays refused by name
(`hdbscan/NOT_IMPLEMENTED.tsv`).

**Additions (2026-09-27, after IVF-PQ lands):** IVF-SQ and IVF-RaBitQ
(cuVS `ivf_sq/`, `ivf_rabitq/`; quantization arms on the same index), the
refine step (cuVS `refine.cuh`), and the sample filter (both already rows
in `ivf/NOT_IMPLEMENTED.tsv`).
