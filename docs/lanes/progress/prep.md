# prep: progress

Lane 5 (preprocessing + naive Bayes & discriminant analysis), pass 1.
Worktree `~/mojolearn-wt/algos-prep2`, branch `lane/algos-prep2`, pod `prep` (NVIDIA H100).

## How the lane computes
One binding entry, `x_prep_run`, runs a program of UNITS over one float32 arena
(x_prep/common.mojo): the GPU binding launches one thread per unit
(x_prep/device.mojo), the host binding loops the same units
(x_prep/host/program.mojo). Units: x_prep/prims.mojo (columns, encoders,
dense), x_prep/eigh.mojo (Jacobi), naive_bayes/nb.mojo, naive_bayes/da.mojo;
op table x_prep/units.mojo == `_OPS` in python/mojolearn/_expansion_prep.py.
Sanity tests need scikit-learn: on the pod it is in /root/skl
(`PYTHONPATH=python:/root/skl pixi run -e default python python/mojolearn/tests/test_x_prep_<x>.py`).

## Merged (pass 1: builds, sklearn sanity, CPU == NVIDIA AGREE)
| algorithm | commit | lanes | verdict |
|---|---|---|---|
| RobustScaler, MaxAbsScaler | ea9fa6c00 | x-prep-robust-scaler, x-prep-maxabs-scaler | AGREE (batch 9, infer 9, train 9; cuda H100 vs CPU) |
| OneHotEncoder, OrdinalEncoder | a8d0e06b8 | x-prep-ordinal-encoder,x-prep-onehot-encoder | batch 9, infer 9, train 9; cuda H100 vs CPU |
| TargetEncoder | d0ca493f6 | x-prep-target-encoder | batch 9, infer 9, train 9; cuda H100 vs CPU |
| SimpleImputer | 2ba3a6c0d | x-prep-simple-imputer | batch 9, infer 9, train 9; cuda H100 vs CPU |
| KBinsDiscretizer | afa14e9dd | x-prep-kbins | batch 9, infer 9, train 9; cuda H100 vs CPU |
| GaussianNB, MultinomialNB, BernoulliNB | ae2cee425 | x-prep-gaussian-nb,x-prep-multinomial-nb,x-prep-bernoulli-nb | batch 9, infer 9, train 9; cuda H100 vs CPU |
| LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis | 6f33de672 | x-prep-lda,x-prep-qda | batch 9, infer 9, train 9; cuda H100 vs CPU |
| QuantileTransformer | faf2a80e5 | x-prep-quantile-transformer | batch 9, infer 9, train 9; cuda H100 vs CPU |
| PowerTransformer | 2df64e382 | x-prep-power-transformer | batch 9, infer 9, train 9; cuda H100 vs CPU |
| Normalizer | b27e52137 | x-prep-normalizer | batch 9, infer 9, train 9; cuda H100 vs CPU |
| PolynomialFeatures | f8c283f4a | x-prep-polynomial-features | batch 9, infer 9, train 9; cuda H100 vs CPU |
| SplineTransformer | 032612ed1 | x-prep-spline-transformer | batch 9, infer 9, train 9; cuda H100 vs CPU |
| Binarizer | 5a893acab | x-prep-binarizer | batch 9, infer 9, train 9; cuda H100 vs CPU |
| LabelEncoder | e62864c81 | x-prep-label-encoder | infer 9, train 9; cuda H100 vs CPU |
| LabelBinarizer | 87df7824e | x-prep-label-binarizer | infer 9, train 9; cuda H100 vs CPU |
| MultiLabelBinarizer | 3c46f0142 | x-prep-multilabel-binarizer | infer 9, train 9; cuda H100 vs CPU |
| IterativeImputer | 77e436512 | x-prep-iterative-imputer | batch 9, infer 9, train 9; cuda H100 vs CPU |
| VarianceThreshold | 9a474af44 | x-prep-variance-threshold | batch 9, infer 9, train 9; cuda H100 vs CPU |
| SelectKBest with f_classif, chi2, f_regression | 7a3c408c0 | x-prep-select-kbest | batch 9, infer 9, train 9; cuda H100 vs CPU |
| mutual_info_classif, mutual_info_regression | ad898a159 | x-prep-mutual-info | infer 9, train 9; cuda H100 vs CPU |
| RFE | 3061ce9b8 | x-prep-rfe | batch 9, infer 9, train 9; cuda H100 vs CPU |
| ComplementNB | bdde41d02 | x-prep-complement-nb | batch 9, infer 9, train 9; cuda H100 vs CPU |
| CategoricalNB | (this commit) | x-prep-categorical-nb | batch 9, infer 9, train 9; cuda H100 vs CPU |

## Pass 2
| step | commit | verdict |
|---|---|---|
| option parity: SimpleImputer(add_indicator=True) | (this commit) | x-prep-simple-imputer-indicator AGREE (batch 9, infer 9, train 9); x-prep-simple-imputer SAME BITS |
| option parity: RobustScaler(unit_variance=True) | (this commit) | x-prep-robust-scaler-unit-variance AGREE (batch 9, infer 9, train 9); robust / quantile lanes SAME BITS |
| end-to-end sabotage `e2e_host_branch.patch` on the 21 summing lanes | 310afeee0 | AGREE, DISAGREE under it on all 21, AGREE after reversal (H100) |
| option parity: priors / class_prior (GaussianNB, the discrete NBs, LDA incl. renormalisation, QDA) | (this commit) | x-prep-priors AGREE (infer 9, train 9); existing NB/DA lanes' cells SAME BITS vs the pass-2 run |
| seam proof on NVIDIA: 10 seams (DEVIATIONS 5400-5409, IDENTITY_PATHS rows 140-149), `x_prep/seams/prep_check.mojo` host AND device vs oracle, 10 sabotage arms RED, all 28 lanes AGREE with `--pass 2` | (this commit) | PASS on H100 (`algos_lane_check.sh <28 lanes> --pass 2`) |
| option parity: inverse_transform of QuantileTransformer, PowerTransformer, KBinsDiscretizer, LabelBinarizer, OrdinalEncoder, OneHotEncoder; OrdinalEncoder encoded_missing_value (the reference default: a NaN category is written NaN, not its index) and the unknown_value / encoded_missing_value collision checks | (this commit) | x-prep-inverse-transforms AGREE, DISAGREE under e2e_host_branch; x-prep-encoder-options AGREE, DISAGREE under e2e_store_branch; the 31 earlier lanes SAME BITS (H100) |
| seam arms re-proved on the fixed lane check (CURRENT DIRECTIVES 000): seam_5401_contraction.patch was a BROKEN ARM (an import inside a loop: it never built); fixed, and all 10 arms now build, run and FAIL | (this commit) | PASS on H100 (`--pass 2`); done, never repeat |
| option parity: TargetEncoder StratifiedKFold folds for a binary / multiclass target (the reference's `_make_test_folds`, shuffle by splitmix64) | (this commit) | x-prep-target-encoder AGREE; its cross-fit cells MOVED (18, intended: the folds are now stratified); unshuffled folds equal sklearn's exactly (test) |
| option parity: LinearDiscriminantAnalysis solver 'lsqr' / 'eigen', shrinkage None / 'auto' (Ledoit-Wolf) / constant, store_covariance; QuadraticDiscriminantAnalysis solver 'eigen' + shrinkage, store_covariance | (this commit) | x-prep-da-solvers AGREE, DISAGREE under e2e_host_branch |
| option parity: sample_weight for GaussianNB / MultinomialNB / ComplementNB / BernoulliNB / CategoricalNB; CategoricalNB min_categories | (this commit) | x-prep-nb-weights AGREE, DISAGREE under e2e_host_branch; every other prep lane (and par-gpc-predict, which the selector picks) AGREE, SAME BITS |
| option parity (on lane/algos-prep2, steward pending): the reference's score edges (f_classif NaN for a constant feature / single class, +inf for a within-class-constant one; chi2 NaN for an all-zero feature; canonical NaN word), f_regression force_finite=False, r_regression (new), RFE importance_getter str / callable; RFE tie rule written as DIFFERS BY NAME (numpy's unstable argsort) | (this commit) | x-prep-score-edges AGREE (--pass 2), DISAGREE under e2e_host_branch; every other prep lane AGREE and SAME BITS vs inv3 except x-prep-select-kbest/dupes (intended: that fixture's constant and all-zero columns now score NaN, as the reference) (H100) |
| option parity (on lane/algos-prep2, steward pending): PolynomialFeatures order='F' (same bits, column-major), OrdinalEncoder / OneHotEncoder categories=<list> | (this commit) | x-prep-encoder-categories AGREE (--pass 2), DISAGREE under e2e_store_branch; x-prep-polynomial-features SAME BITS |
| steward: 1790529624248 / 1790529633391 (4de76eb6e1) PASS m2pro + do-amd; 1790530038647 (x-prep-simple-imputer-indicator) PASS; 1790535515159 / 1790535524761 (d2e61ed9d3: LDA/QDA solvers, NB sample_weight, stratified TargetEncoder folds, seam 5401 fix) PASS | - | PASS |
| option parity (on lane/algos-prep2, steward pending): OneHotEncoder / OrdinalEncoder min_frequency, max_categories, handle_unknown 'infrequent_if_exist' / 'warn', OneHotEncoder drop=<list>; LabelBinarizer multilabel y; SimpleImputer strategy=<callable>; LDA / QDA covariance_estimator; TargetEncoder categories=<list>, cv=<splitter> / (train, test) pairs; partial_fit for GaussianNB, MultinomialNB, ComplementNB, BernoulliNB, CategoricalNB (category_count_ exposed); KBinsDiscretizer every numpy quantile_method. New units indicator, code_counts, remap_codes, add_arrays, gnb_merge, cat_counts, cat_flp (ops 87-93). New lanes x-prep-user-objects, x-prep-nb-partial (sum), x-prep-infrequent, x-prep-label-binarizer-multilabel, x-prep-kbins-methods (store) | (this commit) | the 43 selected lanes AGREE (--pass 2, H100, evidence run-parity2); the 5 new lanes DISAGREE under e2e_host_branch / e2e_store_branch and AGREE after reversal; the 38 existing lanes SAME BITS vs run-edges; test_x_prep_parity PASS on GPU and CPU (scikit-learn 1.9.1); test_host_surface 196 passed; test_lane_select OK |
| option parity: mutual_info discrete_features=True / mask / indices (contingency MI unit mi_dd; Ross with a feature's categories against the noised target, unit mi_dc); FIX at the root: the reference's 1e-10 tie-breaking noise vanished in the float32 add (tied columns read MI 0), now kept as a second word and every distance compared as a (primary, noise) pair (DEVIATION 5407 widened, IDENTITY_PATHS row 147, seam arm 5407 regenerated); singleton classes / categories refused as the reference | (this commit) | 44 selected lanes AGREE (--pass 2 with the seam arms, H100, evidence run-mi); x-prep-mi-discrete and x-prep-mutual-info DISAGREE under e2e_host_branch, AGREE after reversal (sab-mi); every other lane SAME BITS vs run-parity2; x-prep-mutual-info moved only on its tie fixtures (intended); test_x_prep_selection PASS on GPU and CPU |
| option parity: KBinsDiscretizer sample_weight (units kbins_gw / kbins_wq / kbins_wkm: the reference's `_weighted_percentile` over distinct values with summed weights, averaged or not; the nonzero-weight range for uniform; a weighted Lloyd for kmeans; a weighted resample above subsample; other quantile methods refused as the reference) | (this commit) | x-prep-kbins-weights AGREE (--pass 2), DISAGREE under e2e_host_branch (sab-p2b); test_x_prep_parity PASS on GPU and CPU |
| option parity: SplineTransformer knots=<array>, extrapolation 'linear' (derivative at the edge) / 'periodic' (knot wrap, exact float32 fmod), sample_weight (weighted percentile / nonzero-weight range), handle_missing 'zeros', order 'F'; degree-0 'constant' writes the reference's rule; the reference's degree <= 1 'linear' loop bug and degree-0 'constant' crash DIFFER BY NAME | (this commit) | x-prep-spline-options AGREE (--pass 2), DISAGREE under e2e_host_branch; x-prep-spline-transformer SAME BITS; test_x_prep_spline PASS on GPU and CPU |
| option parity: IterativeImputer imputation_order 'random', n_nearest_features (|corr|-weighted draws; ii_sub / ii_br over a predictor mask), sample_posterior (ii_sigma + ii_post: the predictive std and a truncated-normal draw by inversion, AS 241 PPND7), add_indicator, estimator=<any> (the rounds in Python); FIX at the root: the stop is the reference's matrix inf-norm (largest row sum of the change), not the elementwise max | (this commit) | x-prep-iterative-options AGREE (--pass 2), DISAGREE under e2e_host_branch; x-prep-iterative-imputer SAME BITS; test_x_prep_iterative PASS on GPU and CPU |
| tools/algos_lane_check.py names this box's GPU target (sm_NN / gfxNNN) for build_byte_lm.sh, which refused a Linux build without one and failed any check whose byte LM binding went stale | (this commit) | the 48 selected lanes AGREE with --pass 2 and all 10 seam arms bite (run-p2b, H100); every existing lane SAME BITS vs run-mi |
| steward FAIL 1790537100517 fixed at the root: `apple_steward.py submit` coalesced the lane's queued store-lane request (e2e_store_branch) into a sum-lane request and ran all 43 lanes under e2e_host_branch alone, so the 12 store lanes read 'sabotage not seen' (the lane's code was not at fault: each lane's clean AGREE passed, and the same store lanes PASS under e2e_store_branch in 1790537100516 / 1790542801200). Coalescing now takes only a request carrying the same patch bytes; store lanes resubmitted as 1790553231197 (b23b38412, e2e_store_branch) | b23b38412 (merged) | in-memory check of the filter; the live submit left 1790549984236 (host patch) queued as submitted |
| CURRENT DIRECTIVES (x_* second GPU call): x_prep/device.mojo now holds ONE process-lifetime DeviceContext (`x_prep_ctx`, the x_cnn `_Global` pattern); python/mojolearn/tests/test_x_prep_twice.py runs 21 estimators' programs twice in one process | (this commit) | new pod lohadbfeo1vnr5 (H100): fresh SAME BITS reference at main f237f1996 (ref-main: the 48 lanes of lanes_p2b.txt, --pass 2, RESULT PASS; ref-main-sc: the 5 binding scaler lanes, PASS); at the branch (run-branch) all 54 AGREE (--pass 2, the 10 seam arms bite) and samebits.py reads SAME BITS on all 53 existing lanes (a perturbed hash reads MOVED: the check can fail); test_x_prep_twice PASS on GPU and CPU |
| option parity item 5 (scalers): StandardScaler / MinMaxScaler NaN (fit ignores it per feature, n_samples_seen_ a vector when counts differ, a never-seen feature has NaN statistics; transform keeps NaN, refuses inf), StandardScaler sample_weight (scalar or vector, zeros skip), partial_fit for both (first batch == fit bit for bit; MinMax: running extrema, params by the binding's own minmax_fit over [data_min_, data_max_]; Standard: batch stats by fit's own path merged by gnb_merge with K = d, d = 1), copy=False (in place into a writable float32 C-contiguous caller buffer), save/load of a vector or weight-sum n_samples_seen_. A finite unweighted call is unchanged (the binding's standard_fit / minmax_fit / transforms). New x_prep units scaler_stats / std_scale / nan_keep (ops 101-103); gnb_merge KEEP flag (param 11; the naive Bayes callers pad 0, SAME BITS): two exactly constant parts keep their value (FIX at the root: a constant column's merged mean drifted an ulp and the next merge read a 1e-7 scale). New lane x-prep-scaler-options; tests test_x_prep_scaler_options.py | (this commit) | x-prep-scaler-options AGREE (--pass 2, run-branch), DISAGREE under e2e_host_branch and AGREE after reversal (sab-scaler); minmax-scaler, minmax-scaler-clip, standard-scaler, standard-scaler-no-mean, standard-scaler-no-std and every prep lane SAME BITS vs ref-main / ref-main-sc; test_x_prep_scaler_options (9), test_x_prep_scalers, test_x_prep_twice, test_x_prep_parity: 21 passed on GPU and CPU (scikit-learn 1.9.1); test_host_surface 200 passed; test_lane_select OK (H100 pod) |

## Next
PHASE: option parity (2 in the LANE CHARTER at the top of docs/lanes/ALGORITHM_EXPANSION_PLAN.md), still open:
only resample/ remains of item 5 (the scalers are merged, see the Pass 2 table).
- Pod `prep` = lohadbfeo1vnr5 (H100, up 2026-09-28 01:08Z, lease 480 min; `dev_pod.sh extend prep`). Its /root/mojolearn is a git
  checkout at main f237f1996 plus the branch's files (qsync from pod_base = f237f1996); scikit-learn 1.9.1 + pytest in /root/skl
  (installed with the system pip: `python3 -m pip install --target /root/skl --python-version 3.X --only-binary=:all:`, then
  remove /root/skl/numpy*; the pixi env has no pip). SAME BITS references on the pod: /root/mojolearn-evidence/algos-prep/run-branch
  (all 54 lanes, the merged state). NEVER `pkill -f <name>` through `dev_pod.sh run` when <name> is in the command itself: it kills
  the ssh shell first (it did, twice, this session); kill by PID.
- Steward (post-merge, one request per lane per hour): STEWARD_PLACEHOLDER
  Earlier FAILs, both closed: 1790526554859 (310afeee0) ran all 29 lanes under e2e_host_branch, which cannot reach the 8 store lanes
  it named; those lanes are re-proved under e2e_store_branch by 1790553231197 (m3ultra-b, m2pro, m4-a PASS; do-amd queued) and the
  sum lanes by 1790549984236 (PASS on all four). 1790537100517 was the coalescing fault fixed at b23b38412. RULE: a request's patch
  must reach every lane in it: sum lanes (sum_lanes.txt + x-prep-scaler-options) with e2e_host_branch, store lanes (store_lanes.txt)
  with e2e_store_branch, never mixed; the binding scaler lanes (standard-scaler*, minmax-scaler*) with neither (no prep patch reaches
  the _mojolearn_preprocessing binding).
- Item 5, resample/ (resample/NOT_IMPLEMENTED.tsv): BCa is refused for want of an inverse normal CDF; x_prep/iterative.mojo has a
  float32 AS 241 PPND7 (`_ppnd7`) and `_phi` (portable_erff): move them to checks/numerics.mojo as a NEW SEAM with the full proof
  (host oracle, separating fixture, sabotage, DEVIATION, card stage, IDENTITY_PATHS row; gate it as check-division gates
  portable_divf), then BCa; then paired=False, permutation_type 'samples' / 'pairings', resample replace=False / stratify /
  sample_weight. Sparse output stays REFUSED BY NAME (no sparse Array).
- Then: FAST speed (3), IDENTICAL speed (4), CPU speed (5), one phase per session.
Helper scripts (not in the repo): ~/mojolearn-evidence/algos-prep/{qsync,gate,commit,mkpatches,addlane,samebits}.sh|py;
qsync sends every file differing from ~/mojolearn-evidence/algos-prep/pod_base (so it also carries main's
changes merged into the branch). mkpatches.py regenerates the 10 seam arms from the sources (fixed this
session for 5401 and 5407; it reproduces the committed patches). The latest full clean run of every prep
lane (the "same bits" reference) is on the pod at /root/mojolearn-evidence/algos-prep/run-p2b.
