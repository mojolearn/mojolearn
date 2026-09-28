# linear: progress

Design (pass 1): every fit is one sequential Mojo function in `x_linear/`
(`fit_dispatch`), called directly by the host binding and from a ONE-THREAD
kernel by the GPU binding, so CPU and GPU run the same operation sequence;
scoring (`x_linear_decision`) is one thread per (row, output). Parallel fit
schedules are pass 2 speed work. Sanity: `cd python && PYTHONPATH=/root/skl
python -m mojolearn.tests.test_x_linear_sanity <case>` on the pod (sklearn
installed with `pip --target /root/skl --no-deps scikit-learn scipy joblib
threadpoolctl cloudpickle narwhals`).

Skipped from the table: LinearSVC / LinearSVR are already public
(`mojolearn.svm`, lanes linear-svc*, linear-svr*).

| algorithm | commit | lanes | pod gate |
|---|---|---|---|
| SGDClassifier / SGDRegressor | 7f23b8d5c | x-sgd-clf, x-sgd-reg | sanity PASS; RESULT: PASS (AGREE on x-sgd-clf,x-sgd-reg), RTX 4090 |
| PoissonRegressor / GammaRegressor / TweedieRegressor | 273ecb094 | x-glm-poisson, x-glm-gamma, x-glm-tweedie | sanity PASS (coef within 2e-4 of sklearn); RESULT: PASS (AGREE on all three) |
| HuberRegressor | b81926452 | x-huber | sanity PASS (coef within 2e-5 of sklearn); RESULT: PASS (AGREE on x-huber) |
| BayesianRidge / ARDRegression | 762aa5ef9 | x-bayes-ridge, x-ard | sanity PASS (coef within 2e-6 of sklearn); RESULT: PASS (AGREE on both) |
| Lars / LassoLars | 90634cac4 | x-lars, x-lasso-lars | sanity PASS (coef within 2e-6 of sklearn); RESULT: PASS (AGREE on both) |
| QuantileRegressor | 73aa9bc9b | x-quantile | sanity PASS (objective within 3e-7 relative of sklearn's LP optimum); RESULT: PASS (AGREE) |
| Perceptron | 01e1dcdd0 | x-perceptron | sanity PASS (accuracy within 0.04 of sklearn, 2 and 3 classes); RESULT: PASS (AGREE) |
| PassiveAggressiveClassifier / PassiveAggressiveRegressor | 3d534315b | x-pa-clf, x-pa-reg | sanity PASS (accuracy within 0.02, R2 within 4e-4 of sklearn); RESULT: PASS (AGREE on both) |
| SGDOneClassSVM | feb2ad310 | x-sgd-ocsvm | sanity PASS (outlier fraction equal to sklearn's at nu 0.1 and 0.5); RESULT: PASS (AGREE) |
| RidgeClassifier | f372aaf47 | x-ridge-clf | sanity PASS (coef within 1e-6 of sklearn, predictions equal); RESULT: PASS (AGREE) |
| RidgeCV | e8dfdf91f | x-ridge-cv | sanity PASS (same alpha_, best_score_ within 4e-7 relative); RESULT: PASS (AGREE) |
| LassoCV | 0f6bad966 | x-lasso-cv | sanity PASS (same alpha_, mse_path_ within 1.2e-6 relative); RESULT: PASS (AGREE) |
| ElasticNetCV | 468117252 | x-enet-cv | sanity PASS (same alpha_ and l1_ratio_, mse_path_ within 1e-6 relative); RESULT: PASS (AGREE) |
| LogisticRegressionCV | 3b545850a | x-logistic-cv (and x-huber re-gated: lbfgs.mojo line search changed) | sanity PASS (StratifiedKFold ids equal, same C_, proba within 2e-4); RESULT: PASS (AGREE on x-logistic-cv, x-huber) |
| IsotonicRegression | 7e02a9bc4 | x-isotonic | sanity PASS (thresholds equal to sklearn, predict within 3e-7, same NaN mask); RESULT: PASS (AGREE) |

## Pass 2

Seam ledger: DEVIATIONS 5000-5009, IDENTITY_PATHS rows 100-109, x_linear/README.md.
Check driver x_linear/checks/seams_check.mojo (oracle seams_oracle.mojo), ten
arms in tools/identity_lanes/linear.checks. End-to-end sabotage for the
stewards: x_linear/checks/sabotage/e2e_device_fold.patch (every lane but
x-isotonic) and e2e_device_pava.patch (x-isotonic; the interp arm did not bite, replaced): device-only source edits.

| step | commit | result |
|---|---|---|
| seams + ledger | 0e7290fb4 (isotonic e2e arm swapped in 406335b92) | NVIDIA RTX 4090: `algos_lane_check.sh <all 21> --pass 2` RESULT: PASS, all 10 arms FAIL under their patch and PASS after reversal, 21 lanes AGREE |
| option: GLM sample_weight (Poisson/Gamma/Tweedie) | 651e359c8 | sanity PASS (weighted coef within 1.2e-4 of sklearn); lanes x-glm-* AGREE, x-glm-poisson-sw AGREE; opt_glm_sample_weight.patch (device drops weights) DISAGREE then AGREE; x-glm-poisson/gamma/tweedie hashes UNCHANGED vs the pass-2 run |
| option: sample_weight for HuberRegressor, QuantileRegressor, SGDClassifier/Regressor, Perceptron, PassiveAggressive*, SGDOneClassSVM, RidgeClassifier, RidgeCV, BayesianRidge, LogisticRegressionCV; class_weight for SGDClassifier, Perceptron, PassiveAggressiveClassifier, RidgeClassifier, LogisticRegressionCV | 289237990 | sanity PASS against sklearn for each; 8 new lanes (x-huber-sw, x-quantile-sw, x-sgd-clf-w, x-sgd-reg-sw, x-ridge-clf-w, x-ridge-cv-sw, x-bayes-ridge-sw, x-logistic-cv-w) AGREE; every device-only weights arm DISAGREE then AGREE; the 26 earlier lanes' hashes UNCHANGED |
| option: positive for LassoLars, LassoCV, ElasticNetCV | (the commit that adds this row) | sanity PASS (coef within 2e-3 of sklearn, all >= 0); lanes x-lasso-lars-pos, x-lasso-cv-pos AGREE; device-only arms DISAGREE then AGREE; earlier lanes UNCHANGED |

Directive 000 (session 3): the seam bites above were recorded before the
lane-check fix (3084ca09c). Re-run ONCE on the RTX 4090 pod with the fixed tool
(box tools/algos_lane_check.py md5 = origin/main's) at 7be5da1e3:
`algos_lane_check.sh x-isotonic --pass 2`. All ten arms (5000-5009, 5006 as
seam_5006_host_nan.patch) BUILD, RUN and FAIL under their patch, PASS after
reversal; no BROKEN arm; clean lane AGREE. Done; never repeat.

Stewards (the earlier m2pro FAILs were the 5006 arm that is null on Arm, fixed
in a3156303f): submitted at 7be5da1e3 to m2pro + do-amd:
- 1790535754001-linear-7be5da1e3d: the 31 lanes other than x-isotonic, sabotage e2e_device_fold.patch
- 1790535762888-linear-7be5da1e3d: x-isotonic, sabotage e2e_device_pava.patch

VERDICT: both PASS on m2pro and do-amd (m3ultra deferred, spooled). The 21
expansion algorithms and their 11 option lanes are proven on CPU, NVIDIA,
AMD and Apple.

Merged to main per directive 0000b (pod gates: test_host_surface 196 passed
on the full tree, test_lane_select OK: 0 failure(s)).

Still in phase 1 (LANE CHARTER, main 3af84e8a0): the family also covers the
EXISTING models (LinearRegression, Ridge, Lasso, ElasticNet,
LogisticRegression, LinearSVC/SVR). Audit each against the phase-1 bar (lane
with CPU+GPU paths; per-seam host oracle, separating fixture, biting sabotage,
DEVIATION, card stage; AGREE on NVIDIA, AMD, Apple) and close the gaps.
Next phase: (2) option parity. Remaining NOT IMPLEMENTED rows in
x_linear/NOT_IMPLEMENTED.tsv (SGD early_stopping/average/warm_start/partial_fit,
GLM/Huber warm_start, Tweedie power ranges, Bayes compute_score/return_std,
Lars paths and LarsCV/LassoLarsCV/LassoLarsIC, RidgeClassifier solvers/positive,
RidgeCV k-fold/multi-target, CD CV splitters/selection=random/sample_weight,
LogisticRegressionCV l1/elasticnet) AND the existing public linear models
(python/mojolearn/linear_model.py: LinearRegression, Ridge, Lasso, ElasticNet,
LogisticRegression) against sklearn and cuML options.

## Phase 1 (b), the EXISTING models (session 4, 2026-09-27)

Branch commits c3d1df385..8d979a0bf (seam arms for glm/solver, the 527 gate
fix, DEVIATIONS 550/551 -> 5010/5011, `.core` joins existing lanes to
linear.checks, lane_select attributes `.core`), then:

- NVIDIA RTX 4090 run at 16f5dd898 (`algos_lane_check.sh <17 core lanes>
  --pass 2`): every seam arm in linear.checks (x_linear 5000-5009 and glm/
  solver 527, 545, 547, 549 x2, 552, 705-708, 714, 610, 612, 2620-2622)
  BUILD, RUN, FAIL under its patch, PASS after reversal. 16 of 17 lanes
  AGREE; logistic-unpenalized-no-intercept/dupes infer + batch DIVERGENT.
- ROOT CAUSE (found this session): MAX's CPU pool threads run with MXCSR
  FTZ|DAZ (0x9fe0; the calling thread 0x1fa0). The float64 sigmoid link
  `1/(1+exp(709.39))` = 8.2e-309 flushed to 0 on the CPU column's pool
  tasks and not on the GPU binding's serial host link. IDENTICAL at
  MOJOLEARN_CPU_THREADS=1. The reference x86 record (EPYC 9655) predates
  the threaded host predict. FIX: core/host_fp_env.mojo
  (`host_ieee_fp_enter/leave`, MXCSR bits 15+6, Arm FPCR.FZ) around every
  task of core/classical_host_predict.mojo's four row splits and
  glm/estimator.mojo::qn_softmax_host. OWED ELSEWHERE (not this lane's
  code): every other host `sync_parallelize` site runs on the same FTZ pool
  (knn/forest host predict, kde oracle, gbdt/metrics/... host oracles); a
  float64 result that can be subnormal there has the same defect. Reported
  to the orchestrator for the `cpu` lane.
- CURRENT DIRECTIVES (DeviceContext): x_linear/device.mojo now uses ONE
  process-lifetime context (`linear_ctx`); python/mojolearn/tests/
  test_x_linear_repeat.py fits all 21 estimators twice per binding in one
  process, GPU == host.

Gate at 0bda3a368+progress (RunPod balance went negative mid-session: the
RTX 4090 pod was deleted and `dev_pod.sh up` is refused, so this ran on the
lane's Hot Aisle MI300X box `linear-amd`, CPU column Xeon Platinum 8470):
- `algos_lane_check.sh <17 core lanes>,pca,pca-whiten,pca-full-whiten,tsvd,
  qn-squared,qn-absolute --pass 2 --sabotage glm/checks/sabotage/
  e2e_existing_device.patch`: every seam arm in linear.checks FAILS under
  its patch and PASSES after reversal; ALL 23 clean lanes AGREE (hip vs
  CPU), logistic-unpenalized-no-intercept included; the e2e device
  sabotage DISAGREEs on all 17 linear lanes + qn-squared/qn-absolute and
  every lane AGREEs again after reversal. RESULT: FAIL only because the
  glm sabotage does not reach pca/pca-whiten/pca-full-whiten/tsvd (they
  are in the run as unchanged-bits controls for classical_host_predict,
  not as sabotage targets): expected, not a defect.
- test_x_linear_repeat + test_host_surface: 201 passed (hip + host).
- tools/test_lane_select.py: OK, 0 failure(s) (inputs changed: new
  core/host_fp_env.mojo, linear.core, lane_select.py).

NVIDIA gate (session 5, 2026-09-28, RunPod RTX 4090 pod `linear`, CPU
column EPYC 7542, tree 42b29f691 = 7325a0415 + origin/main):
- `algos_lane_check.sh <the 17 lanes of linear.core> --pass 2 --sabotage
  glm/checks/sabotage/e2e_existing_device.patch`: every seam arm of
  linear.checks BUILD, RUN, FAIL under its patch, PASS after reversal; all
  17 clean lanes AGREE (logistic-unpenalized-no-intercept included: the
  host_fp_env fix holds on x86 EPYC); the e2e sabotage DISAGREEs on all 17
  and every lane AGREEs after reversal. RESULT: PASS.
- `pca,pca-whiten,pca-full-whiten,tsvd,qn-squared,qn-absolute` as clean
  controls: all AGREE, RESULT: PASS.
- test_x_linear_repeat (cuda + host) PASS; test_host_surface 200 passed.
- core/host_parallel.mojo was not on main at merge time, so
  core/host_fp_env.mojo merged as is (the cpu lane absorbs it).
Merged to main with bench/x_linear_speed.py (the speed board). Phase 1 is
CLOSED. Post-merge: ONE batched steward submit (Apple + do-amd) for the 17
existing lanes, sabotage e2e_existing_device.patch (ids below).
- 2026-09-28 steward 1790561256037-linear-fd405e35a: m2pro, m3ultra, m4-a PASS; do-amd still working at 05:20Z.

## Step 0 coverage audit (session 6, 2026-09-28)

Audited list (every public linear-family estimator; lane = CPU arm + GPU arm
in tools/identity_break.py or tools/identity_lanes/linear.py; sabotage = a
source edit that bites):

| estimators | lanes | sabotage that bites |
|---|---|---|
| the 21 x_linear estimators (SGDClassifier/Regressor, Poisson/Gamma/Tweedie, Huber, BayesianRidge, ARD, Lars, LassoLars, Quantile, Perceptron, PassiveAggressive C/R, SGDOneClassSVM, RidgeClassifier, RidgeCV, LassoCV, ElasticNetCV, LogisticRegressionCV) + 11 option lanes | the 31 x-* lanes of linear.py but x-isotonic | x_linear/checks/sabotage/e2e_device_fold.patch (NVIDIA pass-2 gate, steward 1790535754001) |
| IsotonicRegression | x-isotonic | e2e_device_pava.patch (steward 1790535762888) |
| LinearRegression, Ridge, Lasso, ElasticNet, LogisticRegression, LinearSVC, LinearSVR | the 17 lanes of linear.core | glm/checks/sabotage/e2e_existing_device.patch (NVIDIA gate session 5) |
| QNRegressor | qn-squared, qn-absolute | e2e_existing_device.patch (DISAGREE on both, AMD gate session 4) |

GAP FIXED: QNRegressor (python/mojolearn/linear_model.py) had lanes but no
family owner: qn-squared and qn-absolute are now in linear.core, so checking
them runs linear.checks (its glm/checks/qn_losses_check.mojo arms 707, 708,
714 are QN's seams). tools/test_lane_select.py: OK, 0 failure(s) (central
AMD box CPU, 2026-09-28).
