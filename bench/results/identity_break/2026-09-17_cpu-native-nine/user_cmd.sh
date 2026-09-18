#!/bin/bash
set -uo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 MOJOLEARN_CPU_THREADS=1
export MOJOLEARN_IDENTITY_GBDT_CTR_MODELS="$PWD/bench/results/identity_break/2026-09-15_gbdt-ctr-tables/models"
.pixi/envs/test/bin/python - <<'PYBODY'
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import os, subprocess, json
out=Path(os.environ['LEG_OUT'])/'records';out.mkdir(parents=True,exist_ok=True)
lanes=['arima', 'arima-011', 'arima-seasonal-c', 'bootstrap', 'byte-lm-host-infer', 'byte-lm-host-infer-threaded', 'byte-lm-host-train', 'cross-entropy-arms', 'cross-val', 'dbscan', 'dbscan-brute-l1', 'dbscan-weighted', 'elasticnet', 'elasticnet-l2end-no-intercept', 'et-clf-entropy-bestfirst', 'et-reg-bootstrap-parallel', 'gbdt-adapter-clf', 'gbdt-adapter-reg', 'gbdt-categorical-ctr', 'gbdt-categorical-ctr-tables', 'gbdt-depthwise', 'gbdt-exact-mae', 'gbdt-feature-freq', 'gbdt-lossguide', 'gbdt-multiclass', 'gbdt-onevsall', 'gbdt-ordered-rmse', 'gbdt-pointwise-l2-bayesian-eval', 'gbdt-rmse', 'gbdt-symmetric', 'gbdt-tensor-ctr-tables', 'gemm-pinned', 'gemm-transposed', 'gp-normalize-y', 'gp-sample-y-normalize', 'iforest-tuned', 'kde', 'kde-cosine-minkowski', 'kde-epanechnikov-l1', 'kde-exponential-chebyshev', 'kde-linear-cosine', 'kde-tophat-sqeuclidean', 'kde-weighted', 'kmeans', 'kmeans-array', 'kmeans-classic-pp', 'kmeans-cosine', 'kmeans-random', 'kmeans-weighted', 'knn', 'knn-chebyshev', 'knn-clf', 'knn-clf-distance', 'knn-cosine', 'knn-manhattan', 'knn-minkowski-p3', 'knn-rbc', 'knn-reg', 'knn-reg-distance', 'knn-sqeuclidean', 'lasso', 'logistic', 'logistic-elasticnet', 'logistic-l1', 'logistic-multiclass', 'logistic-unpenalized-no-intercept', 'metrics', 'metrics-classification', 'minmax-scaler', 'minmax-scaler-clip', 'mlp', 'monte-carlo', 'ols', 'ols-no-intercept', 'ols-weighted', 'optim-adam-clip', 'optim-sgd', 'pca', 'pca-full-whiten', 'pca-whiten', 'permutation-test', 'radius', 'radius-chebyshev', 'radius-manhattan', 'radius-minkowski-p3', 'rf-clf', 'rf-clf-balanced-parallel', 'rf-clf-entropy-log2-noboot', 'rf-reg', 'rf-reg-gamma-ig', 'rf-reg-poisson', 'ridge', 'ridge-no-intercept', 'spectral-precomputed', 'standard-scaler', 'standard-scaler-no-mean', 'standard-scaler-no-std', 'training-primitives', 'tsvd']
results={}
def run(lane):
 with (out/(lane+'.log')).open('w') as log:
  code=subprocess.run(['.pixi/envs/test/bin/python','tools/verify_cpu_batch.py','--lanes',lane,'--out',str(out/lane),'--timeout','1200'],stdout=log,stderr=subprocess.STDOUT).returncode
 return lane,code
with ThreadPoolExecutor(max_workers=2) as pool:
 for future in as_completed([pool.submit(run,lane) for lane in lanes]):
  lane,code=future.result();results[lane]=code
  (out/'batch-exits.json').write_text(json.dumps(results,indent=2)+'\n')
  print(lane,code,flush=True)
raise SystemExit(int(any(results.values())))
PYBODY
