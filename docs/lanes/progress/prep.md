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

## Next
PHASE: option parity (2 in the LANE CHARTER at the top of docs/lanes/ALGORITHM_EXPANSION_PLAN.md), still open.
The charter makes the family include the EXISTING members too: StandardScaler, MinMaxScaler
(python/mojolearn/preprocessing.py, binding _mojolearn_preprocessing) and resampling (bootstrap,
permutation_test, monte_carlo_integrate; python/mojolearn/resample.py). Their parity is owed.
- Steward: two requests per commit (sum lanes + e2e_host_branch, store lanes + e2e_store_branch;
  lane lists in ~/mojolearn-evidence/algos-prep/{sum_lanes,store_lanes}.txt, which include every lane
  so far). Merge gate (CURRENT DIRECTIVES 0000b): (m2pro OR m3ultra) PASS and do-amd PASS.
  NOT YET MERGED (branch lane/algos-prep2): score edges / encoder categories / order F (requests
  1790537091369 / 1790537100516 at 23b11a7ba5) and the parity batch (requests 1790542790533 /
  1790542801200 at bf2bc9f081; the later requests cover every lane, so their PASS is enough). Next session: run
  `apple_steward.py status`; when 1790542790533 and 1790542801200 pass the gate, `git fetch origin && git
  merge origin/main && git push origin HEAD:main`; on FAIL fix at the root and resubmit.
- Option parity still NOT IMPLEMENTED (x_prep/NOT_IMPLEMENTED.tsv), in order:
  1. mutual_info discrete_features=True / mask / indices: discrete x + discrete y is the contingency
     MI (sklearn mutual_info_score; a new unit over per-column codes from _fit_categories + lookup);
     discrete x + continuous y is mi_cd with the roles swapped (labels = the feature's codes, Z = the
     noised y), needing a per-column NUSED in mi_reduce; discrete columns get no scaling or noise.
  2. KBinsDiscretizer sample_weight: quantile with averaged_inverted_cdf / inverted_cdf only (others
     ValueError, as the reference): the weighted percentile over DISTINCT values with their summed
     weights (equivalent to sklearn `_weighted_percentile`; zero-weight groups skipped, the p = 0 level
     takes the first positive-weight value, average when cdf - p <= float32 eps); uniform / kmeans take
     min / max over nonzero-weight rows and a weighted Lloyd over the distinct values; with subsample,
     a weighted resample. kbins_edges_unit has 11 params: add X, W and a 2n-per-column scratch (14 max).
  3. SplineTransformer knots=<array>, extrapolation 'linear' / 'periodic', sample_weight.
  4. IterativeImputer estimator=<any>, sample_posterior, n_nearest_features, imputation_order
     'random', add_indicator.
  5. The existing members: StandardScaler (partial_fit, sample_weight, NaN-ignoring fit, copy=False,
     with_mean/with_std already), MinMaxScaler (partial_fit, NaN handling, copy=False), resampling
     options vs scipy.stats (bootstrap method / confidence_level / n_resamples / batch,
     permutation_test permutation_type / alternative, monte_carlo). Read their modules first and add
     NOT_IMPLEMENTED rows for what is missing.
  Sparse output stays REFUSED BY NAME (no sparse Array).
- Then: FAST speed (3), IDENTICAL speed (4), CPU speed (5), one phase per session.
Helper scripts (not in the repo): ~/mojolearn-evidence/algos-prep/{qsync,gate,commit,mkpatches,addlane,samebits}.sh|py;
qsync sends every file differing from ~/mojolearn-evidence/algos-prep/pod_base. The latest full clean
run of every prep lane (the "same bits" reference) is on the pod at
/root/mojolearn-evidence/algos-prep/run-parity2.
