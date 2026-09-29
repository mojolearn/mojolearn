# Bench board coverage

Every public mojolearn algorithm (python/mojolearn/__init__.py `__all__`, the ten `_expansion_*.py` doors, and the functions of mojolearn.linalg, resample, training) against the board's race ids (`python3 tools/bench_board.py --dry-run --vendor <v>`), 2026-09-29, branch lane/bench-board-m3ultra. Configuration objects, state containers, GP kernel objects, GBDT loss objects, modules and runtime switches are not algorithms and are not listed.

Every lane is planned on all three vendors (apple, nvidia, amd): the same 264 lanes and 464 races on each (2026-09-29, branch lane/bench-board-harness). Only the ARMS differ by vendor: cuML, cuVS, cuGraph, CuPy, implicit-gpu and xgboost-gpu are CUDA only; torch TF32 arms are NVIDIA only; our FAST arm races on Apple only (NVIDIA and AMD race IDENTICAL, plus ours-cpu everywhere).

| | count |
|---|---|
| public algorithms listed | 260 |
| raced before 2026-09-29 | 215 |
| added 2026-09-29 (new lanes) | 27 |
| added 2026-09-29, second pass (the 15 whose reason was not good) | 15 |
| not raced (reason below) | 1 |
| EXCLUDED (Andrew, 2026-09-29): multi-GPU, not raced | 2 |

## The 13 families (README.md, "Algorithms")

The board's own five driver families are trees, classical, classical2, neural and algos; the README's 13 algorithm families cut across them:

| family | algorithms listed | raced | not raced | excluded |
|---|---|---|---|---|
| boosting | 7 | 7 | 0 | 0 |
| trees and ensembles | 19 | 19 | 0 | 0 |
| clustering | 12 | 12 | 0 | 0 |
| neighbors, density and search | 20 | 19 | 0 | 1 |
| linear models | 27 | 27 | 0 | 0 |
| kernel methods and Gaussian processes | 14 | 14 | 0 | 0 |
| decomposition and manifold | 30 | 30 | 0 | 0 |
| time series | 18 | 18 | 0 | 0 |
| preprocessing, probabilistic models and resampling | 40 | 39 | 1 | 0 |
| neural blocks and optimizers | 41 | 41 | 0 | 0 |
| convolutional and graph networks | 15 | 15 | 0 | 0 |
| language models and tokenization | 6 | 5 | 0 | 1 |
| linear algebra | 11 | 11 | 0 | 0 |

## Table

| algorithm | family | lane id(s) | opponents | vendors | status |
|---|---|---|---|---|---|
| AdaBoostClassifier | boosting | algos/adaboost-clf | sklearn-cpu | apple, nvidia, amd | raced |
| AdaBoostRegressor | boosting | algos/adaboost-reg | sklearn-cpu | apple, nvidia, amd | raced |
| DARTClassifier | boosting | algos/dart | lightgbm-cpu; xgboost-cpu; xgboost-gpu (nvidia) | apple, nvidia, amd | raced |
| DARTRegressor | boosting | algos/dart-reg | lightgbm-cpu; xgboost-cpu; xgboost-gpu (nvidia) | apple, nvidia, amd | raced |
| GradientBoosting | boosting | trees/gbdt-symmetric, trees/gbdt-depthwise, trees/gbdt-lossguide, trees/gbdt-rank-yetirank, trees/gbdt-rank-pairlogit, trees/gbdt-multiclass, trees/gbdt-categorical | catboost-cpu (apple, amd); catboost-gpu (nvidia) (per lane) | apple, nvidia, amd | raced |
| GradientBoostingClassifier / GradientBoostingRegressor | boosting | trees/gbdt-symmetric | catboost-cpu (apple, amd); catboost-gpu (nvidia) | apple, nvidia, amd | raced (scikit-learn-shaped doors over GradientBoosting: raced through the gbdt-* lanes' engine) |
| Ordered boosting (GradientBoosting boosting_type='Ordered') | boosting | trees/gbdt-ordered | catboost-cpu (apple, amd); catboost-gpu (nvidia) | apple, nvidia, amd | raced |
| BaggingClassifier | trees and ensembles | algos/bagging-clf | sklearn-cpu | apple, nvidia, amd | raced |
| BaggingRegressor | trees and ensembles | algos/bagging-reg | sklearn-cpu | apple, nvidia, amd | raced |
| CalibratedClassifierCV | trees and ensembles | algos/calibrated | sklearn-cpu | apple, nvidia, amd | raced |
| DecisionTreeClassifier | trees and ensembles | algos/decision-tree-clf | sklearn-cpu; cuml-gpu (nvidia) | apple, nvidia, amd | raced |
| DecisionTreeRegressor | trees and ensembles | algos/decision-tree-reg | sklearn-cpu; cuml-gpu (nvidia) | apple, nvidia, amd | raced |
| ExtraTreesClassifier / ExtraTreesRegressor | trees and ensembles | trees/et | sklearn-et-cpu; lightgbm-cpu (apple, amd) | apple, nvidia, amd | raced (the class follows each dataset's task) |
| IsolationForest | trees and ensembles | trees/iforest | cuml-iforest-gpu (nvidia); sklearn-iforest-cpu (apple, amd) | apple, nvidia, amd | raced |
| KernelExplainer | trees and ensembles | algos/kernel-shap | shap-cpu; cuml-gpu (nvidia) | apple, nvidia, amd | raced |
| MultiOutputClassifier | trees and ensembles | algos/multioutput-clf | sklearn-cpu | apple, nvidia, amd | raced |
| MultiOutputRegressor | trees and ensembles | algos/multioutput-reg | sklearn-cpu | apple, nvidia, amd | raced |
| OneVsRestClassifier | trees and ensembles | algos/ovr | sklearn-cpu | apple, nvidia, amd | raced |
| PermutationExplainer | trees and ensembles | algos/permutation-shap | shap-cpu; cuml-gpu (nvidia) | apple, nvidia, amd | raced |
| RandomForestClassifier / RandomForestRegressor | trees and ensembles | trees/rf | cuml-rf-gpu (nvidia); lightgbm-cpu (apple, amd); sklearn-rf-cpu (apple, amd) | apple, nvidia, amd | raced (the class follows each dataset's task) |
| RandomTreesEmbedding | trees and ensembles | algos/random-trees-embedding | sklearn-cpu | apple, nvidia, amd | raced |
| StackingClassifier | trees and ensembles | algos/stacking-clf | sklearn-cpu | apple, nvidia, amd | raced |
| StackingRegressor | trees and ensembles | algos/stacking-reg | sklearn-cpu | apple, nvidia, amd | raced |
| TreeExplainer | trees and ensembles | algos/tree-shap | lightgbm-cpu; shap-cpu; xgboost-cpu; xgboost-gpu (nvidia) | apple, nvidia, amd | raced |
| VotingClassifier | trees and ensembles | algos/voting-clf | sklearn-cpu | apple, nvidia, amd | raced |
| VotingRegressor | trees and ensembles | algos/voting-reg | sklearn-cpu | apple, nvidia, amd | raced |
| AffinityPropagation | clustering | algos/affinity-prop | sklearn-cpu | apple, nvidia, amd | raced |
| AgglomerativeClustering | clustering | classical2/agglomerative | cuml-gpu (nvidia); sklearn-cpu (apple, amd) | apple, nvidia, amd | raced |
| BayesianGaussianMixture | clustering | algos/bayesian-gmm | sklearn-cpu | apple, nvidia, amd | raced |
| BisectingKMeans | clustering | algos/bisecting-kmeans | sklearn-cpu | apple, nvidia, amd | raced |
| DBSCAN | clustering | classical/dbscan | cuml-gpu (nvidia); sklearn-cpu (apple, amd) | apple, nvidia, amd | raced |
| GaussianMixture | clustering | classical2/gmm | sklearn-cpu | apple, nvidia, amd | raced |
| HDBSCAN | clustering | classical/hdbscan | cuml-gpu (nvidia); sklearn-cpu (apple, amd) | apple, nvidia, amd | raced |
| KMeans | clustering | classical/kmeans | torch-gpu; cuml-gpu (nvidia); sklearn-cpu (apple, amd) | apple, nvidia, amd | raced |
| MeanShift | clustering | algos/meanshift | sklearn-cpu | apple, nvidia, amd | raced |
| MiniBatchKMeans | clustering | algos/minibatch-kmeans | sklearn-cpu | apple, nvidia, amd | raced |
| OPTICS | clustering | algos/optics | sklearn-cpu | apple, nvidia, amd | raced |
| SpectralClustering | clustering | classical2/spectral | sklearn-cpu; cuml-gpu (nvidia) | apple, nvidia, amd | raced |
| CagraIndex | neighbors, density and search | algos/cagra | faiss-cpu; cuvs-gpu (nvidia) | apple, nvidia, amd | raced |
| connected_components | neighbors, density and search | algos/connected-components | networkx-cpu; cugraph-gpu (nvidia) | apple, nvidia, amd | raced |
| IVFIndex | neighbors, density and search | classical2/ivf | cuvs-gpu (nvidia); faiss-cpu (apple, amd) | apple, nvidia, amd | raced |
| IVFPQIndex | neighbors, density and search | algos/ivf-filter, algos/ivf-pq, algos/ivf-refine | faiss-cpu (per lane) | apple, nvidia, amd | raced |
| IVFRaBitQIndex | neighbors, density and search | algos/ivf-rabitq | faiss-cpu | apple, nvidia, amd | raced |
| IVFSQIndex | neighbors, density and search | algos/ivf-sq | faiss-cpu; cuvs-gpu (nvidia) | apple, nvidia, amd | raced |
| KernelDensity | neighbors, density and search | classical/kde | cuml-gpu (nvidia); sklearn-cpu (apple, amd) | apple, nvidia, amd | raced |
| KNeighborsClassifier | neighbors, density and search | classical2/knn-clf | cuml-gpu (nvidia); sklearn-cpu (apple, amd) | apple, nvidia, amd | raced |
| KNeighborsRegressor | neighbors, density and search | classical2/knn-reg | cuml-gpu (nvidia); sklearn-cpu (apple, amd) | apple, nvidia, amd | raced |
| KNNImputer | neighbors, density and search | algos/knn-imputer | sklearn-cpu | apple, nvidia, amd | raced |
| LabelPropagation | neighbors, density and search | algos/label-propagation | sklearn-cpu | apple, nvidia, amd | raced |
| LabelSpreading | neighbors, density and search | algos/label-spreading | sklearn-cpu | apple, nvidia, amd | raced |
| LocalOutlierFactor | neighbors, density and search | algos/lof | sklearn-cpu | apple, nvidia, amd | raced |
| Louvain | neighbors, density and search | algos/louvain | networkx-cpu; cugraph-gpu (nvidia) | apple, nvidia, amd | raced |
| NearestCentroid | neighbors, density and search | algos/nearest-centroid | sklearn-cpu | apple, nvidia, amd | raced |
| NearestNeighbors | neighbors, density and search | classical/knn | torch-gpu; cuml-gpu (nvidia); sklearn-cpu (apple, amd) | apple, nvidia, amd | raced |
| PageRank | neighbors, density and search | algos/pagerank | networkx-cpu; cugraph-gpu (nvidia) | apple, nvidia, amd | raced |
| RadiusNeighbors | neighbors, density and search | algos/radius-neighbors | sklearn-cpu | apple, nvidia, amd | added 2026-09-29 |
| refine | neighbors, density and search | algos/ivf-refine | faiss-cpu; cuvs-gpu (nvidia) | apple, nvidia, amd | raced (the re-rank step the ivf-refine lane times) |
| DistributedIVFIndex and parallel_* modules | neighbors, density and search | none | | | EXCLUDED (Andrew, 2026-09-29): multi-GPU, not raced |
| ARDRegression | linear models | algos/ard | sklearn-cpu | apple, nvidia, amd | raced |
| BayesianRidge | linear models | algos/bayesian-ridge | sklearn-cpu | apple, nvidia, amd | raced |
| ElasticNet | linear models | classical2/elasticnet | cuml-gpu (nvidia); sklearn-cpu (apple, amd) | apple, nvidia, amd | raced |
| ElasticNetCV | linear models | algos/enet-cv | sklearn-cpu | apple, nvidia, amd | raced |
| GammaRegressor | linear models | algos/gamma | sklearn-cpu | apple, nvidia, amd | raced |
| HuberRegressor | linear models | algos/huber | sklearn-cpu | apple, nvidia, amd | raced |
| IsotonicRegression | linear models | algos/isotonic | sklearn-cpu | apple, nvidia, amd | raced |
| Lars | linear models | algos/lars | sklearn-cpu; cuml-gpu (nvidia) | apple, nvidia, amd | raced |
| Lasso | linear models | classical2/lasso | cuml-gpu (nvidia); sklearn-cpu (apple, amd) | apple, nvidia, amd | raced |
| LassoCV | linear models | algos/lasso-cv | sklearn-cpu | apple, nvidia, amd | raced |
| LassoLars | linear models | algos/lasso-lars | sklearn-cpu | apple, nvidia, amd | raced |
| LinearRegression | linear models | classical/ols | torch-gpu; cuml-gpu (nvidia); sklearn-cpu (apple, amd); torch-gpu-eigh (nvidia, amd) | apple, nvidia, amd | raced |
| LogisticRegression | linear models | classical2/logreg | cuml-gpu (nvidia); sklearn-cpu (apple, amd) | apple, nvidia, amd | raced |
| LogisticRegressionCV | linear models | algos/logreg-cv | sklearn-cpu | apple, nvidia, amd | raced |
| PassiveAggressiveClassifier | linear models | algos/pa-clf | sklearn-cpu | apple, nvidia, amd | raced |
| PassiveAggressiveRegressor | linear models | algos/pa-reg | sklearn-cpu | apple, nvidia, amd | raced |
| Perceptron | linear models | algos/perceptron | sklearn-cpu | apple, nvidia, amd | raced |
| PoissonRegressor | linear models | algos/poisson | sklearn-cpu | apple, nvidia, amd | raced |
| QuantileRegressor | linear models | algos/quantile | sklearn-cpu | apple, nvidia, amd | raced |
| Ridge | linear models | classical2/ridge | cuml-gpu (nvidia); sklearn-cpu (apple, amd) | apple, nvidia, amd | raced |
| RidgeClassifier | linear models | algos/ridge-clf | sklearn-cpu | apple, nvidia, amd | raced |
| RidgeCV | linear models | algos/ridge-cv | sklearn-cpu | apple, nvidia, amd | raced |
| SGDClassifier | linear models | algos/sgd-clf | sklearn-cpu; cuml-gpu (nvidia) | apple, nvidia, amd | raced |
| SGDOneClassSVM | linear models | algos/sgd-ocsvm | sklearn-cpu | apple, nvidia, amd | raced |
| SGDRegressor | linear models | algos/sgd-reg | sklearn-cpu; cuml-gpu (nvidia) | apple, nvidia, amd | raced |
| TweedieRegressor | linear models | algos/tweedie | sklearn-cpu | apple, nvidia, amd | raced |
| QNRegressor | linear models | algos/qn-reg | sklearn-cpu (LinearRegression); cuml-gpu (cuml.solvers.QN, nvidia) | apple, nvidia, amd | raced |
| AdditiveChi2Sampler | kernel methods and Gaussian processes | algos/additive-chi2 | sklearn-cpu | apple, nvidia, amd | raced |
| GaussianProcessClassifier | kernel methods and Gaussian processes | classical2/gpc | sklearn-cpu | apple, nvidia, amd | raced |
| GaussianProcessRegressor | kernel methods and Gaussian processes | classical2/gpr | sklearn-cpu | apple, nvidia, amd | raced |
| KernelRidge | kernel methods and Gaussian processes | classical2/kernel-ridge | cuml-gpu (nvidia); sklearn-cpu (apple, amd) | apple, nvidia, amd | raced |
| LinearSVC | kernel methods and Gaussian processes | classical2/linearsvc | cuml-gpu (nvidia); sklearn-cpu (apple, amd) | apple, nvidia, amd | raced |
| LinearSVR | kernel methods and Gaussian processes | classical2/linearsvr | cuml-gpu (nvidia); sklearn-cpu (apple, amd) | apple, nvidia, amd | raced |
| Nystroem | kernel methods and Gaussian processes | classical2/nystroem | sklearn-cpu | apple, nvidia, amd | raced |
| OneClassSVM | kernel methods and Gaussian processes | algos/ocsvm | sklearn-cpu | apple, nvidia, amd | raced |
| PolynomialCountSketch | kernel methods and Gaussian processes | algos/poly-count-sketch | sklearn-cpu | apple, nvidia, amd | raced |
| RBFSampler | kernel methods and Gaussian processes | classical2/rbf-sampler | sklearn-cpu | apple, nvidia, amd | raced |
| SkewedChi2Sampler | kernel methods and Gaussian processes | algos/skewed-chi2 | sklearn-cpu | apple, nvidia, amd | raced |
| SVC | kernel methods and Gaussian processes | classical/svc | cuml-gpu (nvidia); sklearn-cpu (apple, amd) | apple, nvidia, amd | raced |
| SVGP | kernel methods and Gaussian processes | algos/svgp | gpytorch-cpu; gpytorch-gpu | apple, nvidia, amd | raced |
| SVR | kernel methods and Gaussian processes | classical2/svr | cuml-gpu (nvidia); sklearn-cpu (apple, amd) | apple, nvidia, amd | raced |
| AlternatingLeastSquares | decomposition and manifold | algos/als | implicit-cpu; implicit-gpu (nvidia) | apple, nvidia, amd | raced |
| CCA | decomposition and manifold | algos/cca | sklearn-cpu | apple, nvidia, amd | raced |
| ClassicalMDS | decomposition and manifold | algos/classical-mds | sklearn-cpu | apple, nvidia, amd | raced |
| DictionaryLearning | decomposition and manifold | algos/dict-learning | sklearn-cpu | apple, nvidia, amd | raced |
| EllipticEnvelope | decomposition and manifold | algos/elliptic-envelope | sklearn-cpu | apple, nvidia, amd | raced |
| FactorAnalysis | decomposition and manifold | algos/factor-analysis | sklearn-cpu | apple, nvidia, amd | raced |
| FastICA | decomposition and manifold | algos/fastica | sklearn-cpu | apple, nvidia, amd | raced |
| GaussianRandomProjection | decomposition and manifold | algos/gaussian-rp | sklearn-cpu; cuml-gpu (nvidia) | apple, nvidia, amd | raced |
| IncrementalPCA | decomposition and manifold | algos/incremental-pca | sklearn-cpu; cuml-gpu (nvidia) | apple, nvidia, amd | raced |
| Isomap | decomposition and manifold | algos/isomap | sklearn-cpu | apple, nvidia, amd | raced |
| KernelPCA | decomposition and manifold | algos/kernel-pca | sklearn-cpu | apple, nvidia, amd | raced |
| LatentDirichletAllocation | decomposition and manifold | algos/lda | sklearn-cpu | apple, nvidia, amd | raced |
| LocallyLinearEmbedding | decomposition and manifold | algos/lle | sklearn-cpu | apple, nvidia, amd | raced |
| MDS | decomposition and manifold | algos/mds | sklearn-cpu | apple, nvidia, amd | raced |
| MinCovDet | decomposition and manifold | algos/min-cov-det | sklearn-cpu | apple, nvidia, amd | raced |
| MiniBatchDictionaryLearning | decomposition and manifold | algos/mb-dict-learning | sklearn-cpu | apple, nvidia, amd | raced |
| MiniBatchSparsePCA | decomposition and manifold | algos/mb-sparse-pca | sklearn-cpu | apple, nvidia, amd | raced |
| NMF | decomposition and manifold | algos/nmf | sklearn-cpu | apple, nvidia, amd | raced |
| PCA | decomposition and manifold | classical/pca | torch-gpu; cuml-gpu (nvidia); sklearn-cpu (apple, amd) | apple, nvidia, amd | raced |
| PLSCanonical | decomposition and manifold | algos/pls-canonical | sklearn-cpu | apple, nvidia, amd | raced |
| PLSRegression | decomposition and manifold | algos/pls | sklearn-cpu | apple, nvidia, amd | raced |
| sparse_encode | decomposition and manifold | algos/sparse-coder | sklearn-cpu | apple, nvidia, amd | added 2026-09-29 (the encoder SparseCoder.transform calls) |
| SparseCoder | decomposition and manifold | algos/sparse-coder | sklearn-cpu | apple, nvidia, amd | added 2026-09-29 |
| SparsePCA | decomposition and manifold | algos/sparse-pca | sklearn-cpu | apple, nvidia, amd | raced |
| SparseRandomProjection | decomposition and manifold | algos/sparse-rp | sklearn-cpu; cuml-gpu (nvidia) | apple, nvidia, amd | raced |
| SpectralEmbedding | decomposition and manifold | classical2/spectral-embedding | sklearn-cpu; cuml-gpu (nvidia) | apple, nvidia, amd | raced |
| TruncatedSVD | decomposition and manifold | classical2/tsvd | cuml-gpu (nvidia); sklearn-cpu (apple, amd) | apple, nvidia, amd | raced |
| TSNE | decomposition and manifold | algos/tsne | sklearn-cpu; cuml-gpu (nvidia) | apple, nvidia, amd | raced |
| UMAP | decomposition and manifold | classical2/umap | cuml-gpu (nvidia); umap-learn-cpu (apple, amd); umap-learn-cpu-unseeded (apple, amd) | apple, nvidia, amd | raced |
| johnson_lindenstrauss_min_dim | decomposition and manifold | algos/jl-min-dim | sklearn-cpu | apple, nvidia, amd | raced |
| ARIMA | time series | classical2/arima | statsmodels-cpu; cuml-gpu (nvidia) | apple, nvidia, amd | raced |
| AutoARIMA | time series | algos/autoarima | statsforecast-cpu; cuml-gpu (nvidia) | apple, nvidia, amd | raced |
| AutoTheta | time series | algos/auto-theta | statsforecast-cpu | apple, nvidia, amd | added 2026-09-29 |
| CrostonClassic | time series | algos/croston | statsforecast-cpu | apple, nvidia, amd | raced |
| CrostonOptimized | time series | algos/croston-optimized | statsforecast-cpu | apple, nvidia, amd | added 2026-09-29 |
| CrostonSBA | time series | algos/croston-sba | statsforecast-cpu | apple, nvidia, amd | added 2026-09-29 |
| DampedETS | time series | algos/damped-ets | statsforecast-cpu; statsmodels-cpu | apple, nvidia, amd | raced (ETS(model='AAN', damped=True) by construction) |
| DynamicOptimizedTheta | time series | algos/dynamic-optimized-theta | statsforecast-cpu | apple, nvidia, amd | added 2026-09-29 |
| DynamicTheta | time series | algos/dynamic-theta | statsforecast-cpu | apple, nvidia, amd | added 2026-09-29 |
| ETS | time series | algos/damped-ets | statsforecast-cpu; statsmodels-cpu | apple, nvidia, amd | raced |
| ExponentialSmoothing | time series | classical2/ets | statsmodels-cpu; cuml-gpu (nvidia) | apple, nvidia, amd | raced |
| GARCH | time series | algos/garch | arch-cpu | apple, nvidia, amd | raced |
| OptimizedTheta | time series | algos/optimized-theta | statsforecast-cpu | apple, nvidia, amd | added 2026-09-29 |
| ProphetForecaster | time series | algos/prophet | prophet-cpu | apple, nvidia, amd | raced |
| STL | time series | algos/stl | statsmodels-cpu | apple, nvidia, amd | raced |
| Theta | time series | algos/theta | statsforecast-cpu; statsmodels-cpu | apple, nvidia, amd | raced |
| VAR | time series | algos/var | statsmodels-cpu | apple, nvidia, amd | raced |
| kpss_test / select_d | time series | algos/kpss, algos/select-d | statsmodels-cpu (kpss, ours' lag count passed) | apple, nvidia, amd | raced |
| BernoulliNB | preprocessing, probabilistic models and resampling | algos/bernoulli-nb | sklearn-cpu; cuml-gpu (nvidia) | apple, nvidia, amd | raced |
| Binarizer | preprocessing, probabilistic models and resampling | algos/binarizer | sklearn-cpu; cuml-gpu (nvidia) | apple, nvidia, amd | raced |
| CategoricalNB | preprocessing, probabilistic models and resampling | algos/categorical-nb | sklearn-cpu; cuml-gpu (nvidia) | apple, nvidia, amd | raced |
| chi2 | preprocessing, probabilistic models and resampling | algos/select-chi2 | sklearn-cpu | apple, nvidia, amd | raced (as SelectKBest(score_func=chi2)) |
| ComplementNB | preprocessing, probabilistic models and resampling | algos/complement-nb | sklearn-cpu; cuml-gpu (nvidia) | apple, nvidia, amd | raced |
| cross_val_score | preprocessing, probabilistic models and resampling | algos/cross-val-score | sklearn-cpu | apple, nvidia, amd | added 2026-09-29 (model selection) |
| f_classif | preprocessing, probabilistic models and resampling | algos/select-f-classif | sklearn-cpu | apple, nvidia, amd | raced (as SelectKBest(score_func=f_classif)) |
| f_regression | preprocessing, probabilistic models and resampling | algos/select-f-regression | sklearn-cpu | apple, nvidia, amd | raced (as SelectKBest(score_func=f_regression)) |
| GaussianNB | preprocessing, probabilistic models and resampling | algos/gaussian-nb | sklearn-cpu; cuml-gpu (nvidia) | apple, nvidia, amd | raced |
| IterativeImputer | preprocessing, probabilistic models and resampling | algos/iterative-imputer | sklearn-cpu | apple, nvidia, amd | raced |
| KBinsDiscretizer | preprocessing, probabilistic models and resampling | algos/kbins | sklearn-cpu; cuml-gpu (nvidia) | apple, nvidia, amd | raced |
| LabelBinarizer | preprocessing, probabilistic models and resampling | algos/label-binarizer | sklearn-cpu; cuml-gpu (nvidia) | apple, nvidia, amd | raced |
| LabelEncoder | preprocessing, probabilistic models and resampling | algos/label-encoder | sklearn-cpu; cuml-gpu (nvidia) | apple, nvidia, amd | raced |
| LinearDiscriminantAnalysis | preprocessing, probabilistic models and resampling | algos/lda-clf | sklearn-cpu | apple, nvidia, amd | raced |
| MaxAbsScaler | preprocessing, probabilistic models and resampling | algos/maxabs-scaler | sklearn-cpu; cuml-gpu (nvidia) | apple, nvidia, amd | raced |
| MinMaxScaler | preprocessing, probabilistic models and resampling | algos/minmax-scaler | sklearn-cpu; cuml-gpu (nvidia) | apple, nvidia, amd | added 2026-09-29 |
| MultiLabelBinarizer | preprocessing, probabilistic models and resampling | algos/multilabel-binarizer | sklearn-cpu | apple, nvidia, amd | raced |
| MultinomialNB | preprocessing, probabilistic models and resampling | algos/multinomial-nb | sklearn-cpu; cuml-gpu (nvidia) | apple, nvidia, amd | raced |
| mutual_info_classif | preprocessing, probabilistic models and resampling | algos/select-mutual-info | sklearn-cpu | apple, nvidia, amd | raced (as SelectKBest(score_func=mutual_info_classif)) |
| mutual_info_regression | preprocessing, probabilistic models and resampling | algos/select-mutual-info-reg | sklearn-cpu | apple, nvidia, amd | added 2026-09-29 (as SelectKBest(score_func=mutual_info_regression)) |
| Normalizer | preprocessing, probabilistic models and resampling | algos/normalizer | sklearn-cpu; cuml-gpu (nvidia) | apple, nvidia, amd | raced |
| OneHotEncoder | preprocessing, probabilistic models and resampling | algos/onehot | sklearn-cpu; cuml-gpu (nvidia) | apple, nvidia, amd | raced |
| OrdinalEncoder | preprocessing, probabilistic models and resampling | algos/ordinal | sklearn-cpu | apple, nvidia, amd | raced |
| PolynomialFeatures | preprocessing, probabilistic models and resampling | algos/poly-features | sklearn-cpu; cuml-gpu (nvidia) | apple, nvidia, amd | raced |
| PowerTransformer | preprocessing, probabilistic models and resampling | algos/power-transformer | sklearn-cpu; cuml-gpu (nvidia) | apple, nvidia, amd | raced |
| QuadraticDiscriminantAnalysis | preprocessing, probabilistic models and resampling | algos/qda | sklearn-cpu | apple, nvidia, amd | raced |
| QuantileTransformer | preprocessing, probabilistic models and resampling | algos/quantile-transformer | sklearn-cpu; cuml-gpu (nvidia) | apple, nvidia, amd | raced |
| r_regression | preprocessing, probabilistic models and resampling | algos/select-r-regression | sklearn-cpu | apple, nvidia, amd | added 2026-09-29 (as SelectKBest(score_func=r_regression)) |
| resample.bootstrap | preprocessing, probabilistic models and resampling | algos/bootstrap | scipy-cpu | apple, nvidia, amd | added 2026-09-29 |
| resample.permutation_test | preprocessing, probabilistic models and resampling | algos/permutation-test | scipy-cpu | apple, nvidia, amd | added 2026-09-29 |
| resample.resample / resample_indices | preprocessing, probabilistic models and resampling | algos/resample | sklearn-cpu | apple, nvidia, amd | added 2026-09-29 (resample_indices is the draw inside resample()) |
| RFE | preprocessing, probabilistic models and resampling | algos/rfe | sklearn-cpu | apple, nvidia, amd | raced |
| RobustScaler | preprocessing, probabilistic models and resampling | algos/robust-scaler | sklearn-cpu; cuml-gpu (nvidia) | apple, nvidia, amd | raced |
| SelectKBest | preprocessing, probabilistic models and resampling | algos/select-chi2, algos/select-f-classif, algos/select-f-regression, algos/select-mutual-info, algos/select-mutual-info-reg, algos/select-r-regression | sklearn-cpu | apple, nvidia, amd | raced |
| SimpleImputer | preprocessing, probabilistic models and resampling | algos/simple-imputer | sklearn-cpu; cuml-gpu (nvidia) | apple, nvidia, amd | raced |
| SplineTransformer | preprocessing, probabilistic models and resampling | algos/spline | sklearn-cpu | apple, nvidia, amd | raced |
| StandardScaler | preprocessing, probabilistic models and resampling | algos/standard-scaler | sklearn-cpu; cuml-gpu (nvidia) | apple, nvidia, amd | added 2026-09-29 |
| TargetEncoder | preprocessing, probabilistic models and resampling | algos/target-encoder | sklearn-cpu; cuml-gpu (nvidia) | apple, nvidia, amd | raced |
| VarianceThreshold | preprocessing, probabilistic models and resampling | algos/variance-threshold | sklearn-cpu | apple, nvidia, amd | raced |
| resample.monte_carlo_integrate | preprocessing, probabilistic models and resampling | none | | | NOT RACED: no pinned library has a plain seeded Monte Carlo integrator (numpy and torch have none; scipy.integrate.qmc_quad is quasi-Monte Carlo over a QMCEngine) (opponent: none) |
| Adafactor | neural blocks and optimizers | algos/adafactor | torch-compile-fp32; torch-eager-fp32 | apple, nvidia, amd | raced |
| Adagrad | neural blocks and optimizers | algos/adagrad | torch-compile-fp32; torch-eager-fp32 | apple, nvidia, amd | raced |
| Adam | neural blocks and optimizers | algos/adam | torch-compile-fp32; torch-eager-fp32 | apple, nvidia, amd | added 2026-09-29 |
| Adamax | neural blocks and optimizers | algos/adamax | torch-compile-fp32; torch-eager-fp32 | apple, nvidia, amd | raced |
| AdamW | neural blocks and optimizers | algos/adamw | torch-compile-fp32; torch-eager-fp32 | apple, nvidia, amd | added 2026-09-29 |
| Embedding | neural blocks and optimizers | algos/embedding | torch-compile-fp32; torch-eager-fp32 | apple, nvidia, amd | added 2026-09-29 |
| GRUClassifier | neural blocks and optimizers | algos/gru-clf | torch-compile-bf16; torch-compile-fp32; torch-eager-bf16; torch-eager-fp32; torch-compile-tf32 (nvidia); torch-eager-tf32 (nvidia) | apple, nvidia, amd | raced |
| GRURegressor | neural blocks and optimizers | algos/gru-reg | torch-compile-bf16; torch-compile-fp32; torch-eager-bf16; torch-eager-fp32; torch-compile-tf32 (nvidia); torch-eager-tf32 (nvidia) | apple, nvidia, amd | raced |
| LAMB | neural blocks and optimizers | algos/lamb |  | apple, nvidia, amd | raced |
| layer_norm_backward | neural blocks and optimizers | algos/layernorm | torch-compile-bf16; torch-compile-fp32; torch-eager-bf16; torch-eager-fp32; torch-compile-tf32 (nvidia); torch-eager-tf32 (nvidia) | apple, nvidia, amd | raced (the functions under LayerNorm) |
| layer_norm_forward | neural blocks and optimizers | algos/layernorm | torch-compile-bf16; torch-compile-fp32; torch-eager-bf16; torch-eager-fp32; torch-compile-tf32 (nvidia); torch-eager-tf32 (nvidia) | apple, nvidia, amd | raced |
| LayerNorm | neural blocks and optimizers | algos/layernorm | torch-compile-bf16; torch-compile-fp32; torch-eager-bf16; torch-eager-fp32; torch-compile-tf32 (nvidia); torch-eager-tf32 (nvidia) | apple, nvidia, amd | raced |
| Lion | neural blocks and optimizers | algos/lion |  | apple, nvidia, amd | raced |
| LSTMClassifier | neural blocks and optimizers | algos/lstm-clf | torch-compile-bf16; torch-compile-fp32; torch-eager-bf16; torch-eager-fp32; torch-compile-tf32 (nvidia); torch-eager-tf32 (nvidia) | apple, nvidia, amd | raced |
| LSTMRegressor | neural blocks and optimizers | algos/lstm-reg | torch-compile-bf16; torch-compile-fp32; torch-eager-bf16; torch-eager-fp32; torch-compile-tf32 (nvidia); torch-eager-tf32 (nvidia) | apple, nvidia, amd | raced |
| Mamba1Block | neural blocks and optimizers | neural/mamba1-forward | torch-eager-bf16; torch-eager-fp32; torch-eager-tf32 (nvidia) | apple, nvidia, amd | raced |
| Mamba1BlockInference | neural blocks and optimizers | neural/mamba1-infer | torch-cpu-eager-bf16; torch-cpu-eager-fp32 | apple, nvidia, amd | raced |
| Mamba2Block | neural blocks and optimizers | neural/mamba2-forward | torch-compile-bf16; torch-compile-fp32; torch-eager-bf16; torch-eager-fp32; torch-compile-tf32 (nvidia); torch-eager-tf32 (nvidia) | apple, nvidia, amd | raced |
| Mamba2BlockInference | neural blocks and optimizers | neural/mamba2-infer | torch-cpu-compile-bf16; torch-cpu-compile-fp32; torch-cpu-eager-bf16; torch-cpu-eager-fp32 | apple, nvidia, amd | raced |
| Mamba3Block | neural blocks and optimizers | neural/mamba3-forward | torch-compile-bf16; torch-compile-fp32; torch-eager-bf16; torch-eager-fp32; torch-compile-tf32 (nvidia); torch-eager-tf32 (nvidia) | apple, nvidia, amd | raced |
| Mamba3BlockInference | neural blocks and optimizers | neural/mamba3-infer | torch-cpu-compile-bf16; torch-cpu-compile-fp32; torch-cpu-eager-bf16; torch-cpu-eager-fp32 | apple, nvidia, amd | raced |
| MLPClassifier | neural blocks and optimizers | algos/mlp-clf | sklearn-cpu | apple, nvidia, amd | raced |
| MLPInference | neural blocks and optimizers | neural/mlp-infer | torch-cpu-compile-bf16; torch-cpu-compile-fp32; torch-cpu-eager-bf16; torch-cpu-eager-fp32 | apple, nvidia, amd | raced |
| MLPRegressor | neural blocks and optimizers | algos/mlp-reg | sklearn-cpu | apple, nvidia, amd | raced |
| MoEBlock | neural blocks and optimizers | algos/moe | torch-compile-bf16; torch-compile-fp32; torch-eager-bf16; torch-eager-fp32; torch-compile-tf32 (nvidia); torch-eager-tf32 (nvidia) | apple, nvidia, amd | raced |
| NAdam | neural blocks and optimizers | algos/nadam | torch-compile-fp32; torch-eager-fp32 | apple, nvidia, amd | raced |
| RMSprop | neural blocks and optimizers | algos/rmsprop | torch-compile-fp32; torch-eager-fp32 | apple, nvidia, amd | raced |
| RNNClassifier | neural blocks and optimizers | algos/rnn-clf | torch-compile-bf16; torch-compile-fp32; torch-eager-bf16; torch-eager-fp32; torch-compile-tf32 (nvidia); torch-eager-tf32 (nvidia) | apple, nvidia, amd | raced |
| RNNRegressor | neural blocks and optimizers | algos/rnn-reg | torch-compile-bf16; torch-compile-fp32; torch-eager-bf16; torch-eager-fp32; torch-compile-tf32 (nvidia); torch-eager-tf32 (nvidia) | apple, nvidia, amd | raced |
| SambaInference | neural blocks and optimizers | neural/samba-infer | torch-cpu-compile-bf16; torch-cpu-compile-fp32; torch-cpu-eager-bf16; torch-cpu-eager-fp32 | apple, nvidia, amd | raced |
| SambaStack | neural blocks and optimizers | neural/samba-train-step, neural/samba-forward | torch-compile-bf16; torch-compile-fp32; torch-eager-bf16; torch-eager-fp32; torch-compile-tf32 (nvidia); torch-eager-tf32 (nvidia) | apple, nvidia, amd | raced |
| SGD | neural blocks and optimizers | algos/sgd | torch-compile-fp32; torch-eager-fp32 | apple, nvidia, amd | added 2026-09-29 |
| SmallMLPTrainer | neural blocks and optimizers | neural/mlp-train-step | torch-compile-bf16; torch-compile-fp32; torch-eager-bf16; torch-eager-fp32; torch-compile-tf32 (nvidia); torch-eager-tf32 (nvidia) | apple, nvidia, amd | raced |
| TransformerBlock | neural blocks and optimizers | neural/transformer-forward | torch-compile-bf16; torch-compile-fp32; torch-eager-bf16; torch-eager-fp32; torch-compile-tf32 (nvidia); torch-eager-tf32 (nvidia) | apple, nvidia, amd | raced |
| TransformerBlockInference | neural blocks and optimizers | neural/transformer-infer | torch-cpu-compile-bf16; torch-cpu-compile-fp32; torch-cpu-eager-bf16; torch-cpu-eager-fp32 | apple, nvidia, amd | raced |
| clip_grad_norm_ | neural blocks and optimizers | algos/clip-grad-norm | torch-eager-fp32; torch-compile-fp32 | apple, nvidia, amd | raced |
| cross_entropy (and training.* forward/backward helpers) | neural blocks and optimizers | algos/cross-entropy | torch-eager-fp32; torch-compile-fp32 | apple, nvidia, amd | raced |
| ExponentialLR | neural blocks and optimizers | algos/lr-exponential | torch-cpu (torch.optim.lr_scheduler) | apple, nvidia, amd | raced |
| OneCycleLR | neural blocks and optimizers | algos/lr-onecycle | torch-cpu (torch.optim.lr_scheduler) | apple, nvidia, amd | raced |
| StepLR | neural blocks and optimizers | algos/lr-step | torch-cpu (torch.optim.lr_scheduler) | apple, nvidia, amd | raced |
| training LR schedules (ConstantLR, WarmupLinearLR, WarmupCosineLR) | neural blocks and optimizers | algos/lr-constant, algos/lr-warmup-linear, algos/lr-warmup-cosine | torch-cpu (LinearLR, SequentialLR, CosineAnnealingLR) | apple, nvidia, amd | raced |
| AdaptiveAvgPool2d | convolutional and graph networks | algos/global-avgpool | torch-compile-bf16; torch-compile-fp32; torch-eager-bf16; torch-eager-fp32; torch-compile-tf32 (nvidia); torch-eager-tf32 (nvidia) | apple, nvidia, amd | raced |
| AdaptiveMaxPool2d | convolutional and graph networks | algos/global-maxpool | torch-compile-bf16; torch-compile-fp32; torch-eager-bf16; torch-eager-fp32; torch-compile-tf32 (nvidia); torch-eager-tf32 (nvidia) | apple, nvidia, amd | raced |
| AvgPool1d | convolutional and graph networks | algos/avgpool1d | torch-compile-bf16; torch-compile-fp32; torch-eager-bf16; torch-eager-fp32; torch-compile-tf32 (nvidia); torch-eager-tf32 (nvidia) | apple, nvidia, amd | raced |
| AvgPool2d | convolutional and graph networks | algos/avgpool2d | torch-compile-bf16; torch-compile-fp32; torch-eager-bf16; torch-eager-fp32; torch-compile-tf32 (nvidia); torch-eager-tf32 (nvidia) | apple, nvidia, amd | raced |
| BasicBlock | convolutional and graph networks | algos/resnet-block | torch-compile-bf16; torch-compile-fp32; torch-eager-bf16; torch-eager-fp32; torch-compile-tf32 (nvidia); torch-eager-tf32 (nvidia) | apple, nvidia, amd | raced |
| BatchNorm1d | convolutional and graph networks | algos/batchnorm1d | torch-compile-bf16; torch-compile-fp32; torch-eager-bf16; torch-eager-fp32; torch-compile-tf32 (nvidia); torch-eager-tf32 (nvidia) | apple, nvidia, amd | raced |
| BatchNorm2d | convolutional and graph networks | algos/batchnorm2d | torch-compile-bf16; torch-compile-fp32; torch-eager-bf16; torch-eager-fp32; torch-compile-tf32 (nvidia); torch-eager-tf32 (nvidia) | apple, nvidia, amd | raced |
| CNNClassifier | convolutional and graph networks | algos/cnn-clf | torch-compile-bf16; torch-compile-fp32; torch-eager-bf16; torch-eager-fp32; torch-compile-tf32 (nvidia); torch-eager-tf32 (nvidia) | apple, nvidia, amd | raced |
| Conv1d | convolutional and graph networks | algos/conv1d | torch-compile-bf16; torch-compile-fp32; torch-eager-bf16; torch-eager-fp32; torch-compile-tf32 (nvidia); torch-eager-tf32 (nvidia) | apple, nvidia, amd | raced |
| Conv2d | convolutional and graph networks | algos/conv2d | torch-compile-bf16; torch-compile-fp32; torch-eager-bf16; torch-eager-fp32; torch-compile-tf32 (nvidia); torch-eager-tf32 (nvidia) | apple, nvidia, amd | raced |
| Dropout2d | convolutional and graph networks | algos/dropout2d | torch-compile-fp32; torch-eager-fp32; torch-compile-tf32 (nvidia); torch-eager-tf32 (nvidia) | apple, nvidia, amd | raced |
| GCNConv | convolutional and graph networks | algos/gcn | torch-compile-bf16; torch-compile-fp32; torch-eager-bf16; torch-eager-fp32; torch-compile-tf32 (nvidia); torch-eager-tf32 (nvidia) | apple, nvidia, amd | raced |
| MaxPool1d | convolutional and graph networks | algos/maxpool1d | torch-compile-bf16; torch-compile-fp32; torch-eager-bf16; torch-eager-fp32; torch-compile-tf32 (nvidia); torch-eager-tf32 (nvidia) | apple, nvidia, amd | raced |
| MaxPool2d | convolutional and graph networks | algos/maxpool2d | torch-compile-bf16; torch-compile-fp32; torch-eager-bf16; torch-eager-fp32; torch-compile-tf32 (nvidia); torch-eager-tf32 (nvidia) | apple, nvidia, amd | raced |
| SAGEConv | convolutional and graph networks | algos/graphsage | torch-compile-bf16; torch-compile-fp32; torch-eager-bf16; torch-eager-fp32; torch-compile-tf32 (nvidia); torch-eager-tf32 (nvidia) | apple, nvidia, amd | raced |
| LanguageModelTrainer | language models and tokenization | neural/lm-train-step, neural/lm-forward | torch-compile-bf16; torch-compile-fp32; torch-eager-bf16; torch-eager-fp32; torch-compile-tf32 (nvidia); torch-eager-tf32 (nvidia) | apple, nvidia, amd | raced |
| BpeTokenizer (GPT2Tokenizer) and BpeVocabularyTrainer | language models and tokenization | algos/bpe-encode, algos/bpe-train | hf-tokenizers-cpu (tokenizers==0.23.2) | apple, nvidia, amd | raced |
| LanguageModelHostTrainer | language models and tokenization | neural/lm-host-train-step | torch-cpu-* | apple, nvidia, amd | raced |
| LanguageModelInference | language models and tokenization | neural/lm-infer | torch-cpu-* | apple, nvidia, amd | raced |
| ParallelByteLanguageModelTrainer / ParallelNeuralTrainer / PooledByteLanguageModelTrainer / OffloadedByteLanguageModelTrainer | language models and tokenization | none | | | EXCLUDED (Andrew, 2026-09-29): multi-GPU, not raced |
| SmallByteLanguageModelTrainer | language models and tokenization | neural/lm-train-step, neural/lm-forward | torch-* (it IS LanguageModelTrainer, language_model.py) | apple, nvidia, amd | raced |
| Cholesky | linear algebra | algos/cholesky | numpy-cpu; torch-gpu; cupy-gpu (nvidia) | apple, nvidia, amd | added 2026-09-29 |
| linalg.eigh | linear algebra | algos/eigh | numpy-cpu; torch-gpu; cupy-gpu (nvidia) | apple, nvidia, amd | added 2026-09-29 |
| linalg.qr | linear algebra | algos/qr | numpy-cpu; torch-gpu; cupy-gpu (nvidia) | apple, nvidia, amd | added 2026-09-29 |
| linalg.svd / linalg.svdvals | linear algebra | algos/svd | numpy-cpu; torch-gpu; cupy-gpu (nvidia) | apple, nvidia, amd | added 2026-09-29 (svdvals is svd(compute_uv=False)) |
| lstsq | linear algebra | algos/lstsq | numpy-cpu; torch-gpu; cupy-gpu (nvidia) | apple, nvidia, amd | raced |
| lu_factor | linear algebra | algos/lu-factor | scipy-cpu; torch-gpu; cupy-gpu (nvidia) | apple, nvidia, amd | added 2026-09-29 |
| lu_solve | linear algebra | algos/lu-factor | scipy-cpu; torch-gpu; cupy-gpu (nvidia) | apple, nvidia, amd | added 2026-09-29 (the second call of the lu-factor lane) |
| matmul | linear algebra | neural/gemm | torch-compile-bf16; torch-compile-fp32; torch-eager-bf16; torch-eager-fp32; torch-compile-tf32 (nvidia); torch-eager-tf32 (nvidia) | apple, nvidia, amd | raced |
| randomized_svd | linear algebra | algos/randomized-svd | sklearn-cpu; torch-gpu | apple, nvidia, amd | raced |
| solve | linear algebra | algos/lu-solve | numpy-cpu; torch-gpu; cupy-gpu (nvidia) | apple, nvidia, amd | raced |
| linalg.matmul_bf16 / linalg.matmul_int8 | linear algebra | neural/gemm-bf16, neural/gemm-int8 | torch-*-bf16; torch-*-int8 (torch._int_mm, nvidia only; ours alone on apple and amd) | apple, nvidia, amd | raced |
