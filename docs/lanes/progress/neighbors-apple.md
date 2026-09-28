# neighbors-apple: progress

Apple (Metal) speed for the neighbors family: k-NN, radius neighbors, KDE,
SVC / SVR, KernelRidge, GP regressor / classifier, Nystroem, RBFSampler.
Brief: ~/mojolearn-evidence/apple_speed_brief.md. Branch lane/neighbors-apple,
merged later by the gate runners. Home Mac for speed jobs: m4pro-b.

Board: `bench/x_neighbors_speed.py` (taxi and HIGGS, first eight columns
standardized; each case one load run, then the minimum of REPS timed runs of
fit and of predict; a digest of the outputs; a quality number for FAST).
Arms: `bench/x_neighbors_ab.sh` (build-define arms, forward then reverse
order, default rebuilt at the end).

## Changes on the branch

| commit | change | modes | bits |
|---|---|---|---|
| 9765af975 | SMO outer loop: device fold order, rescan-free scatter, lagged NaN read (FAST_SMO_SYNCS) now also IDENTICAL on Apple; fused update_f stays FAST | IDENTICAL | same by construction |
| b2ab8d372 | svm, kernel_methods, gaussian_process estimators: one process-lifetime DeviceContext per binding and tier | both | same by construction |
| 8f71669af | kNN A/B arm define `MOJOLEARN_EXPERIMENTAL_KNN_WARPBOUND_GUARD` | arm only | - |

## Speed requests (m4pro-b)

## Identity requests

- 1790581787983-neighbors-fc211b48ea: 29 lanes (svc*, svr*, x-neighbors svm
  lanes, kernel-ridge*, nystroem*, rbf-sampler, gp*, gpc*), sabotage
  `~/mojolearn-evidence/neighbors-apple/device_outputs_sabotage.patch`
  (device entry points only: SVM b + 1e-3, kernel_methods / GP first
  uploaded and first downloaded float moved).

## Before -> after
