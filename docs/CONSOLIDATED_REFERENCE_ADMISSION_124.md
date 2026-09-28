# Scoped reference admission: 124 lanes

Admitted 124 lanes from complete historical nine-fixture records, corroborated by current Metal and HIP base-fixture records at `308878e8067914f7864db5db9e4d66a413cca349`. Both current GPU columns also agreed with their local CPU column. All 766 numerical comparisons against the candidate matched. This admits historical reference evidence; it does not claim the current build reran all nine fixtures or establish a current CUDA column.

The source used to validate the candidate was `9d64cb98b284afb3f0af94ffaf39aa127e19bbb6`. All 108 targeted repair lanes are excluded. The unresolved physical-device `par-gmm` lane remains held.

The strict builder checked fixture and held-out hashes, lane and batch revisions, backend witnesses, property protocols, completeness, and multiple independent device classes. Original historical record bytes and source SHA witnesses are retained in `bench/results/identity_break/2026-09-28_consolidated-historical-witnesses/`. Current original record bytes are retained in `bench/results/identity_break/2026-09-28_admitted-124/current/`; its adjacent `admission.json` contains per-record SHA-256 hashes, source commits and completed GPU/CPU verdicts. Each JSON file preserves the original bytes.

Only these lanes' table cells and revision/admission metadata changed. All 628 original record entries remain an unchanged prefix; every unselected table cell remains unchanged. Only the selected lanes' resolved no-reference/stale-reference holds were removed. No native fits were run for this admission.

Base table SHA-256: `b7a9781136e8107ae35f064a843cfbba9d4df6f6d3b896d3a053d1897e5f94e4`.

Promoted table SHA-256: `ba6fe7fa5bea203e09b9ad20fdb93abe51d82744dff09b2651a17ac377392053`.

## Validation

Reference-admission and host-surface tests: 285 passed. Two unrelated fixture-floor tests exposed the existing fragment-discovery issue for the six newly revised GLM/TreeSHAP lanes; these are excluded from this admission. Table integrity checks confirmed the preserved record prefix and unselected cells.

## Admitted lanes

- `gmm`
- `gmm-sample`
- `ivf-filter`
- `pca-inverse`
- `pca-whiten-inverse`
- `trees-adaboost-clf`
- `trees-adaboost-reg`
- `trees-bagging-clf`
- `trees-bagging-reg`
- `trees-calibrated`
- `trees-dart-clf`
- `trees-dart-reg`
- `trees-dt-clf`
- `trees-dt-random`
- `trees-dt-reg`
- `trees-et-deviance`
- `trees-gbdt-multirmse`
- `trees-multioutput`
- `trees-onevsrest`
- `trees-oob-cv-link`
- `trees-random-embedding`
- `trees-rf-weighted`
- `trees-shap-kernel`
- `trees-shap-permutation`
- `trees-stacking-clf`
- `trees-stacking-reg`
- `trees-voting-clf`
- `trees-voting-reg`
- `tsvd-inverse`
- `x-ann-cagra`
- `x-ann-cagra-filter`
- `x-ann-filter`
- `x-ann-ivf-pq`
- `x-ann-ivf-rabitq`
- `x-ann-ivf-sq`
- `x-ann-refine`
- `x-ann-refine-euclidean`
- `x-ann-tsne`
- `x-ann-tsne-pca`
- `x-ard`
- `x-bayes-ridge`
- `x-bayes-ridge-sw`
- `x-cluster-affinity-propagation`
- `x-cluster-agglo-connectivity`
- `x-cluster-agglo-linkages`
- `x-cluster-ap-precomputed`
- `x-cluster-bgmm`
- `x-cluster-bgmm-covtypes`
- `x-cluster-bgmm-inits`
- `x-cluster-bisecting-kmeans`
- `x-cluster-bisecting-options`
- `x-cluster-dbscan-metrics`
- `x-cluster-gmm-options`
- `x-cluster-hdbscan-epsilon`
- `x-cluster-kmeans-init`
- `x-cluster-meanshift`
- `x-cluster-meanshift-binned`
- `x-cluster-minibatch-kmeans`
- `x-cluster-minibatch-options`
- `x-cluster-minibatch-partial`
- `x-cluster-optics`
- `x-cluster-optics-metrics`
- `x-cluster-spectral-affinities`
- `x-cnn-batchnorm`
- `x-cnn-bn-options`
- `x-cnn-conv-options`
- `x-cnn-conv1d`
- `x-cnn-conv2d`
- `x-cnn-globalpool`
- `x-cnn-pool`
- `x-cnn-pool-options`
- `x-cnn-resnet-block`
- `x-cnn-trainer`
- `x-cnn-trainer-options`
- `x-decomp-dict-learning`
- `x-decomp-grp`
- `x-decomp-lda`
- `x-decomp-pca-randomized`
- `x-decomp-pls`
- `x-decomp-sparse-pca`
- `x-decomp-srp`
- `x-enet-cv`
- `x-huber`
- `x-huber-sw`
- `x-lars`
- `x-lasso-cv`
- `x-lasso-cv-pos`
- `x-lasso-lars`
- `x-lasso-lars-pos`
- `x-logistic-cv`
- `x-logistic-cv-w`
- `x-neighbors-additive-chi2`
- `x-neighbors-gamma-scale`
- `x-neighbors-km-kernels`
- `x-neighbors-knn-imputer`
- `x-neighbors-kpca`
- `x-neighbors-krr-options`
- `x-neighbors-label-propagation`
- `x-neighbors-label-spreading`
- `x-neighbors-lof`
- `x-neighbors-nearest-centroid`
- `x-neighbors-ocsvm`
- `x-neighbors-poly-sketch`
- `x-neighbors-skewed-chi2`
- `x-neighbors-svc-multiclass`
- `x-neighbors-svc-probability`
- `x-neighbors-svc-sigmoid`
- `x-neighbors-svgp`
- `x-neighbors-svm-weights`
- `x-neighbors-svr-kernels`
- `x-pa-clf`
- `x-pa-reg`
- `x-perceptron`
- `x-quantile`
- `x-quantile-sw`
- `x-ridge-clf`
- `x-ridge-clf-w`
- `x-ridge-cv`
- `x-ridge-cv-sw`
- `x-sgd-clf`
- `x-sgd-clf-w`
- `x-sgd-ocsvm`
- `x-sgd-reg`
- `x-sgd-reg-sw`
