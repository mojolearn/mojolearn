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
| CURRENT DIRECTIVES (x_* second GPU call): x_prep/device.mojo now holds ONE process-lifetime DeviceContext (`x_prep_ctx`, the x_cnn `_Global` pattern); python/mojolearn/tests/test_x_prep_twice.py runs 21 estimators' programs twice in one process | (branch, NOT MERGED) | test_x_prep_twice PASS on H100 (TWICE OK cuda) and CPU; the full clean lane run for SAME BITS (run-ctx) was cut off when the pod died |

## Next
PHASE: option parity (2 in the LANE CHARTER at the top of docs/lanes/ALGORITHM_EXPANSION_PLAN.md), still open:
only the EXISTING members remain (item 5 below). Items 1-4 are merged (see the Pass 2 table).
- BLOCKED 2026-09-28 00:00Z: the prep pod (c5xxkk5za86bcn) vanished mid lane check (RunPod 404) and `dev_pod.sh up prep` is refused: 'Your account balance is too low to rent a pod'. Andrew must top up RunPod. Then: `dev_pod.sh up prep`, a FULL sync (new box: `MOJOLEARN_DEVPOD_FULL_SYNC=1`), reset ~/mojolearn-evidence/algos-prep/pod_base, reinstall scikit-learn in /root/skl, and (no run-p2b on the new box) record a fresh SAME BITS reference: run every lane in lanes_p2b.txt at origin/main (before the x_prep_ctx commit) and again at the branch; samebits.py must read SAME BITS; then run test_x_prep_twice + test_host_surface and merge lane/algos-prep2.
- Steward (post-merge): 1790549984236 (sum lanes, e2e_host_branch) do-amd PASS, Macs queued; 1790553231197 (store lanes, e2e_store_branch) queued. Check `apple_steward.py status`; a FAIL is a fix commit at the root.
- test_host_surface and test_lane_select on a merge: the pod copy must be a git checkout (test_lane_select
  reads HEAD) and hold the export-ignored bench/results files (git archive drops them): archive + `git
  ls-files bench/results | tar`, link the pod's built .so files, `git init && git add . && git commit`.
- Item 5, the existing members (x_prep/NOT_IMPLEMENTED.tsv rows at the end; resample/NOT_IMPLEMENTED.tsv):
  StandardScaler sample_weight, StandardScaler / MinMaxScaler partial_fit (first batch == fit's bits),
  NaN-ignoring fit + NaN pass-through transform, copy=False. They live in binding _mojolearn_preprocessing
  with recorded release references: add entry points (or x_prep units), never change standard_fit /
  minmax_fit bits. Resampling: BCa is refused for want of an inverse normal CDF; x_prep/iterative.mojo
  now has a float32 AS 241 PPND7 (`_ppnd7`) and `_phi` (portable_erff): move them to checks/numerics.mojo
  and BCa can land; then paired=False, permutation_type 'samples' / 'pairings', resample replace=False /
  stratify / sample_weight, per resample/NOT_IMPLEMENTED.tsv.
  Sparse output stays REFUSED BY NAME (no sparse Array).
- Then: FAST speed (3), IDENTICAL speed (4), CPU speed (5), one phase per session.
Helper scripts (not in the repo): ~/mojolearn-evidence/algos-prep/{qsync,gate,commit,mkpatches,addlane,samebits}.sh|py;
qsync sends every file differing from ~/mojolearn-evidence/algos-prep/pod_base (so it also carries main's
changes merged into the branch). mkpatches.py regenerates the 10 seam arms from the sources (fixed this
session for 5401 and 5407; it reproduces the committed patches). The latest full clean run of every prep
lane (the "same bits" reference) is on the pod at /root/mojolearn-evidence/algos-prep/run-p2b.
