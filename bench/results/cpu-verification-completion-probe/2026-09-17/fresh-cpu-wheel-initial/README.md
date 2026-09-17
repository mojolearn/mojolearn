# Initial full fresh-host CPU wheel replay

All 150 available CPU lanes ran all nine fixtures twice, one local numerical
worker, in 1451 seconds. 121 lanes pass; 29 correctly report OWED references.
There are ZERO numerical DIVERGENT or execution REFUSED cells. The failures
are missing references or old N/A entries for now-supported model, batch or
step/full parts.
They remain failures, not passes or numerical reference evidence.

Every native host family was rebuilt from frozen `7f5b786ae`. The installed
wheel's Python source is `b6132eec0`; `wheel-receipt.json` hashes the artifact
and staged runtime/bindings. This is Python 3.14 on the current Apple M4 Mac,
not a final release qualification. No PyPI upload.

`reports.json.gz` stores all 150 full original reports without repeating
identical top-level metadata. Run `python unpack_reports.py FRESH_DIRECTORY`
to reconstruct them byte for byte; each is checked against its original
SHA-256. This round trip was verified for every report before bundling.
The unbundled originals also remain in the external evidence directory.

OWED lanes: bootstrap, cross-val, elasticnet, elasticnet-l2end-no-intercept, gmm, gmm-random-init, iforest, iforest-tuned, kernel-ridge, kmeans, kmeans-array, kmeans-classic-pp, kmeans-random, kmeans-weighted, lasso, mamba1, mamba2, minmax-scaler, minmax-scaler-clip, nystroem, optim-adam-clip, optim-sgd, permutation-test, radius-chebyshev, radius-minkowski-p3, rbf-sampler, standard-scaler, standard-scaler-no-mean, standard-scaler-no-std.
