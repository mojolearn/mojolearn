#!/bin/bash
set -euo pipefail
unset PYTHONPATH PYTHONHOME
cd /Users/andrewhendel/mojolearn-evidence/cpu-verification-completion
python3 build_candidate_wheel.py
installed-env/bin/python check_candidate_wheel.py transformer-bf16w,transformer-int8w,mamba1-bf16w,mamba1-int8w,mamba2-bf16w,mamba2-int8w,mamba3-bf16w,mamba3-int8w,mlp-bf16w,mlp-int8w,samba-bf16w,samba-int8w lowbit-twelve --native-nine
installed-env/bin/python check_candidate_wheel.py bootstrap,cross-val,elasticnet,elasticnet-l2end-no-intercept,gmm,gmm-random-init,iforest,iforest-tuned,kernel-ridge,kmeans,kmeans-array,kmeans-classic-pp,kmeans-random,kmeans-weighted,lasso,mamba1,mamba2,minmax-scaler,minmax-scaler-clip,nystroem,optim-adam-clip,optim-sgd,permutation-test,radius-chebyshev,radius-minkowski-p3,rbf-sampler,standard-scaler,standard-scaler-no-mean,standard-scaler-no-std repaired-twenty-nine
